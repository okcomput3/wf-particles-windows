/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Andrew Pliatsikas
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <ctime>
#include <cmath>
#include <wayfire/output.hpp>
#include <wayfire/opengl.hpp>
#include <wayfire/core.hpp>
#include <wayfire/view-transform.hpp>
#include <wayfire/signal-definitions.hpp>
#include <wayfire/toplevel-view.hpp>
#include <wayfire/window-manager.hpp>
#include <wayfire/view-transform.hpp>
#include <wayfire/txn/transaction-manager.hpp>
#include <wayfire/render-manager.hpp>
#include <wayfire/scene-render.hpp>
#include <wayfire/util/duration.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <wayfire/plugins/animate/animate.hpp>
#include <wayfire/util.hpp>

// Optimized: Vertex shader no longer needs to handle UVs.
static const char *melt_vert_source =
    R"(
#version 100

attribute highp vec2 position;
uniform mat4 matrix;

void main() {
    gl_Position = matrix * vec4(position, 0.0, 1.0);
}
)";

// Optimized: Fragment shader takes UVs as uniforms.
static const char *melt_frag_source =
    R"(
#version 100
@builtin_ext@
@builtin@

precision highp float;

uniform highp float alpha;
uniform highp vec2 uv_center;
uniform highp vec2 uv_actual;
uniform highp float color_blend;

void main()
{
    vec4 pixel_center = get_pixel(uv_center);
    vec4 pixel_actual = get_pixel(uv_actual);
    vec4 pixel = mix(pixel_actual, pixel_center, color_blend);
    gl_FragColor = vec4(pixel.rgb * alpha, pixel.a * alpha);
}
)";

namespace wf
{
namespace melt
{
using namespace wf::scene;
using namespace wf::animate;
using namespace wf::animation;

static std::string melt_transformer_name = "animation-melt";

wf::option_wrapper_t<wf::animation_description_t> melt_duration{"extra-animations/melt_duration"};
wf::option_wrapper_t<int> melt_block_size{"extra-animations/melt_block_size"};

struct MeltBlock
{
    float x, y;
    float velocity_x;
    float velocity_y;
    float scale;
};

class melt_animation_t : public duration_t
{
  public:
    using duration_t::duration_t;
    timed_transition_t melt{*this};
};

class melt_transformer : public wf::scene::view_2d_transformer_t
{
  public:
    wayfire_view view;
    OpenGL::program_t program;
    wf::output_t *output;
    wf::geometry_t animation_geometry;
    melt_animation_t progression{melt_duration};
    std::vector<MeltBlock> blocks;
    int grid_width, grid_height;

    // Optimized: Pre-calculate a single unit circle to be reused for all particles.
    static constexpr int CIRCLE_SEGMENTS = 20;
    std::vector<float> unit_circle_vertices;

    class simple_node_render_instance_t : public wf::scene::transformer_render_instance_t<transformer_base_node_t>
    {
        wf::signal::connection_t<node_damage_signal> on_node_damaged =
            [=] (node_damage_signal *ev)
        {
            push_to_parent(ev->region);
        };

        melt_transformer *self;
        wayfire_view view;
        damage_callback push_to_parent;

      public:
        simple_node_render_instance_t(melt_transformer *self, damage_callback push_damage,
            wayfire_view view) : wf::scene::transformer_render_instance_t<transformer_base_node_t>(self,
                push_damage,
                view->get_output())
        {
            this->self = self;
            this->view = view;
            this->push_to_parent = push_damage;
            self->connect(&on_node_damaged);
        }

        ~simple_node_render_instance_t() {}

        void schedule_instructions(
            std::vector<render_instruction_t>& instructions,
            const wf::render_target_t& target, wf::region_t& damage) override
        {
            instructions.push_back(render_instruction_t{
                        .instance = this,
                        .target   = target,
                        .damage   = damage & self->animation_geometry,
                    });
        }

        void transform_damage_region(wf::region_t& damage) override
        {
            damage |= wf::region_t{self->animation_geometry};
        }

        void render(const wf::scene::render_instruction_t& data) override
        {
            auto src_box  = self->get_children_bounding_box();
            auto src_tex  = get_texture(1.0);
            auto gl_tex   = wf::gles_texture_t{src_tex};
            auto progress = self->progression.progress();
            auto og = self->output->get_relative_geometry();

            auto ease_out_cubic = [](float t) {
                return 1.0f - std::pow(1.0f - t, 3.0f);
            };

            data.pass->custom_gles_subpass([&]
            {
                wf::gles::bind_render_buffer(data.target);
                GL_CALL(glDisable(GL_CULL_FACE));
                GL_CALL(glEnable(GL_BLEND));
                GL_CALL(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));

                self->program.use(wf::TEXTURE_TYPE_RGBA);
                self->program.set_active_texture(gl_tex);
                self->program.attrib_pointer("position", 2, 0, self->unit_circle_vertices.data());

                const auto output_matrix = wf::gles::output_transform(data.target);
                const int block_size = melt_block_size;

                for (const auto& block : self->blocks)
                {
                    // MODIFIED: Removed the wave_delay calculation to make all particles disperse at once.
                    // The animation progress is now the same for all particles.
                    float adjusted_progress = static_cast<float>(progress);
                    float eased_progress = ease_out_cubic(adjusted_progress);

                    float alpha = 1.0f - eased_progress;
                    if (alpha <= 0.01f)
                    {
                        continue;
                    }

                    float new_x = block.x + block.velocity_x * eased_progress * 300.0f;
                    float new_y = block.y + block.velocity_y * eased_progress * 300.0f;
                    float scale = block.scale * (1.0f - eased_progress * 0.3f);
                    
                    // Calculate particle center in logical output coordinates.
                    float particle_center_x = new_x + block_size / 2.0f;
                    float particle_center_y = new_y + block_size / 2.0f;

                    // Manually convert logical coordinates to Normalized Device Coordinates (NDC).
                    float screen_x = ((particle_center_x + src_box.x - og.width / 2.0f) / og.width) * 2.0f;
                    float screen_y = ((-particle_center_y - src_box.y + og.height / 2.0f) / og.height) * 2.0f;
                    
                    if (eased_progress > 0.5f) {
                        GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE));
                    } else {
                        GL_CALL(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));
                    }

                    // Set UV and color blend uniforms
                    float block_center_u = (block.x + block_size / 2.0f) / src_box.width;
                    float block_center_v = 1.0f - ((block.y + block_size / 2.0f) / src_box.height);
                    float actual_u = (new_x + block_size / 2.0f) / src_box.width;
                    float actual_v = 1.0f - ((new_y + block_size / 2.0f) / src_box.height);

                    self->program.uniform2f("uv_center", std::clamp(block_center_u, 0.0f, 1.0f), std::clamp(block_center_v, 0.0f, 1.0f));
                    self->program.uniform2f("uv_actual", std::clamp(actual_u, 0.0f, 1.0f), std::clamp(actual_v, 0.0f, 1.0f));
                    self->program.uniform1f("color_blend", eased_progress);

                    // Draw layers using matrices built for NDC space.
                    auto draw_layer = [&](float radius_multiplier, float alpha_multiplier) {
                        float radius = (block_size * scale) * radius_multiplier;
                        
                        glm::mat4 model(1.0f);
                        model = glm::translate(model, glm::vec3(screen_x, screen_y, 0.0f));
                        // Scale the unit circle by the radius converted to NDC size.
                        model = glm::scale(model, glm::vec3(radius * 2.0f / og.width, radius * 2.0f / og.height, 1.0f));
                        
                        self->program.uniformMatrix4f("matrix", output_matrix * model);
                        self->program.uniform1f("alpha", alpha * alpha_multiplier);
                        GL_CALL(glDrawArrays(GL_TRIANGLE_FAN, 0, self->CIRCLE_SEGMENTS + 2));
                    };

                    draw_layer(0.8f, 1.0+(0.5f*(1-alpha))); // Outer glow
                    draw_layer(0.6f, 1.0+(1.5f*(1-alpha))); // Middle glow
                    draw_layer(0.4f, 1.0+(3.0f*(1-alpha))); // Core particle (1.0 / 2.5 = 0.4)
                }
                GL_CALL(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));
                self->program.deactivate();
            });
        }
    };

    melt_transformer(wayfire_view view, wf::geometry_t bbox) : wf::scene::view_2d_transformer_t(view)
    {
        this->view = view;
        if (view->get_output())
        {
            output = view->get_output();
            output->render->add_effect(&pre_hook, wf::OUTPUT_EFFECT_PRE);
        }

        animation_geometry = output->get_relative_geometry();
        
        wf::gles::run_in_context([&]
        {
            program.compile(melt_vert_source, melt_frag_source);
        });

        // Optimized: Pre-generate vertices for a unit circle.
        unit_circle_vertices.reserve((CIRCLE_SEGMENTS + 2) * 2);
        unit_circle_vertices.push_back(0.0f); // Center vertex
        unit_circle_vertices.push_back(0.0f);
        for (int i = 0; i <= CIRCLE_SEGMENTS; ++i) {
            float angle = (i / float(CIRCLE_SEGMENTS)) * 2.0f * M_PI;
            unit_circle_vertices.push_back(std::cos(angle));
            unit_circle_vertices.push_back(std::sin(angle));
        }

        std::srand(std::time(nullptr));
        
        int block_size = melt_block_size;
        grid_width = (bbox.width + block_size - 1) / block_size;
        grid_height = (bbox.height + block_size - 1) / block_size;
        
        blocks.reserve(grid_width * grid_height);
        for (int row = 0; row < grid_height; row++)
        {
            for (int col = 0; col < grid_width; col++)
            {
                MeltBlock block;
                block.x = col * block_size;
                block.y = row * block_size;
                block.velocity_x = (std::rand() / float(RAND_MAX) - 0.5f) * 2.0f;
                block.velocity_y = (std::rand() / float(RAND_MAX) - 0.5f) * 2.0f;
                block.scale = 0.9f + (std::rand() / float(RAND_MAX)) * 0.2f;
                blocks.push_back(block);
            }
        }
    }

    wf::geometry_t get_bounding_box() override { return this->animation_geometry; }
    wf::effect_hook_t pre_hook = [=] () { output->render->damage(animation_geometry); };

    void gen_render_instances(std::vector<render_instance_uptr>& instances,
        damage_callback push_damage, wf::output_t *shown_on) override
    {
        instances.push_back(std::make_unique<simple_node_render_instance_t>(this, push_damage, view));
    }

    void init_animation(bool hiding)
    {
        if (!hiding)
        {
            this->progression.reverse();
        }
        this->progression.start();
    }

    virtual ~melt_transformer()
    {
        if (output)
        {
            output->render->rem_effect(&pre_hook);
        }

        wf::gles::run_in_context_if_gles([&] { program.free_resources(); });
    }
};

class melt_animation : public animation_base_t
{
    wayfire_view view;

  public:
    void init(wayfire_view view, wf::animation_description_t dur, animation_type type) override
    {
        this->view = view;
        pop_transformer(view);
        auto bbox = view->get_transformed_node()->get_bounding_box();
        auto tmgr = view->get_transformed_node();
        auto node = std::make_shared<melt_transformer>(view, bbox);
        tmgr->add_transformer(node, wf::TRANSFORMER_HIGHLEVEL + 1, melt_transformer_name);
        node->init_animation(type & WF_ANIMATE_HIDING_ANIMATION);
    }

    void pop_transformer(wayfire_view view)
    {
        if (view->get_transformed_node()->get_transformer(melt_transformer_name))
        {
            view->get_transformed_node()->rem_transformer(melt_transformer_name);
        }
    }

    bool step() override
    {
        if (!view || !view->get_transformed_node()) { return false; }

        if (auto tr = view->get_transformed_node()->get_transformer<melt_transformer>(melt_transformer_name))
        {
            if (!tr->progression.running())
            {
                pop_transformer(view);
                return false;
            }
            return true;
        }
        return false;
    }

    void reverse() override
    {
        if (auto tr = view->get_transformed_node()->get_transformer<melt_transformer>(melt_transformer_name))
        {
            tr->progression.reverse();
        }
    }
};
}
}
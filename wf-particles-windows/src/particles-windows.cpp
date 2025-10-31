/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2024 Scott Moreau <oreaus@gmail.com>
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

static const char *pixel_vert_source =
    R"(
#version 100

attribute highp vec2 position;
attribute highp vec2 uv_in;

uniform mat4 matrix;

varying highp vec2 uv;

void main() {
    uv = uv_in;
    gl_Position = matrix * vec4(position, 0.0, 1.0);
}
)";

static const char *pixel_frag_source =
    R"(
#version 100
@builtin_ext@
@builtin@

precision highp float;

varying highp vec2 uv;
uniform highp float alpha;
uniform highp vec2 uv_actual;
uniform highp float color_blend;

void main()
{
    vec4 pixel_center = get_pixel(uv);
    vec4 pixel_actual = get_pixel(uv_actual);
    
    // Blend between center color (dispersed) and actual texture position (together)
    vec4 pixel = mix(pixel_actual, pixel_center, color_blend);
    
    gl_FragColor = vec4(pixel.rgb * alpha, pixel.a * alpha);
}
)";

namespace wf
{
namespace shatter
{
using namespace wf::scene;
using namespace wf::animate;
using namespace wf::animation;

static std::string shatter_transformer_name = "animation-pixel-disintegrate";

wf::option_wrapper_t<wf::animation_description_t> shatter_duration{"extra-animations/pixel_duration"};
wf::option_wrapper_t<int> pixel_block_size{"extra-animations/pixel_block_size"};

struct PixelBlock
{
    float x, y;           // position
    float velocity_x;     // horizontal velocity
    float velocity_y;     // vertical velocity (falling)
    float rotation;       // rotation angle
    float rotation_speed; // rotation speed
    float delay;          // delay before starting to fall
    float scale;          // size variation
};

struct Particle
{
    float offset_x, offset_y;  // offset from block center
    float velocity_x, velocity_y;  // particle velocity
    float life;  // particle lifetime (0-1)
    float size;  // particle size
};

class shatter_animation_t : public duration_t
{
  public:
    using duration_t::duration_t;
    timed_transition_t shatter{*this};
};

class shatter_transformer : public wf::scene::view_2d_transformer_t
{
  public:
    wayfire_view view;
    OpenGL::program_t program;
    wf::output_t *output;
    wf::geometry_t animation_geometry;
    shatter_animation_t progression{shatter_duration};
    std::vector<PixelBlock> blocks;
    std::vector<std::vector<Particle>> particles;  // particles for each block
    int grid_width, grid_height;

    class simple_node_render_instance_t : public wf::scene::transformer_render_instance_t<transformer_base_node_t>
    {
        wf::signal::connection_t<node_damage_signal> on_node_damaged =
            [=] (node_damage_signal *ev)
        {
            push_to_parent(ev->region);
        };

        shatter_transformer *self;
        wayfire_view view;
        damage_callback push_to_parent;

      public:
        simple_node_render_instance_t(shatter_transformer *self, damage_callback push_damage,
            wayfire_view view) : wf::scene::transformer_render_instance_t<transformer_base_node_t>(self,
                push_damage,
                view->get_output())
        {
            this->self = self;
            this->view = view;
            this->push_to_parent = push_damage;
            self->connect(&on_node_damaged);
        }

        ~simple_node_render_instance_t()
        {}

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

            // Easing function for smooth acceleration
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

                // Render particles only (no blocks)
                for (size_t i = 0; i < self->blocks.size(); i++)
                {
                    auto& block = self->blocks[i];
                    
                    // All blocks animate together - no delay
                    float adjusted_progress = progress;
                    
                    float eased_progress = ease_out_cubic(adjusted_progress);
                    
                    // Calculate physics - no gravity, just velocity-based movement
                    float new_x = block.x + block.velocity_x * eased_progress * 300.0f;
                    float new_y = block.y + block.velocity_y * eased_progress * 300.0f;
                    
                    // Fade out as particles move
                    float alpha = 1.0f - eased_progress;
                    
                    // Scale variation
                    float scale = block.scale * (1.0f - eased_progress * 0.3f);
                    
                    // Color interpolation: when coming together (progress decreasing), 
                    // interpolate from center sample to actual texture position
                    float color_blend = eased_progress;  // 0 = actual texture, 1 = center color
                    
                    // Skip if completely transparent
                    if (alpha <= 0.01f)
                    {
                        continue;
                    }
                    
                    int block_size = pixel_block_size;
                    
                    // Calculate particle center
                    float particle_center_x = new_x + block_size / 2.0f;
                    float particle_center_y = new_y + block_size / 2.0f;
                    
                    // Position in screen space
                    float screen_x = ((particle_center_x + src_box.x - og.width / 2.0f) / og.width) * 2.0f;
                    float screen_y = ((-particle_center_y + og.height / 2.0f - src_box.y) / og.height) * 2.0f;
                    
                    // Interpolate blending mode: additive when dispersed, normal when together
                    if (eased_progress > 0.5f)
                    {
                        // Use additive blending for glowing particles when dispersed
                        GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE));
                    }
                    else
                    {
                        // Transition to normal blending when coming together
                        GL_CALL(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));
                    }
                    
                    // Sample color from block center
                    float block_center_u = (block.x + block_size / 2.0f) / src_box.width;
                    float block_center_v = 1.0f - ((block.y + block_size / 2.0f) / src_box.height);
                    block_center_u = std::clamp(block_center_u, 0.0f, 1.0f);
                    block_center_v = std::clamp(block_center_v, 0.0f, 1.0f);
                    
                    // Calculate actual texture position for this particle
                    float actual_u = (block.x + block_size / 2.0f) / src_box.width;
                    float actual_v = 1.0f - ((block.y + block_size / 2.0f) / src_box.height);
                    actual_u = std::clamp(actual_u, 0.0f, 1.0f);
                    actual_v = std::clamp(actual_v, 0.0f, 1.0f);
                    
                    glm::mat4 model(1.0f);
                    model = glm::translate(model, glm::vec3(screen_x, screen_y, 0.0f));
                    model = glm::scale(model, glm::vec3(2.0f / og.width, 2.0f / og.height, 1.0f));
                    
                    // Set shader uniforms for color blending
                    self->program.uniform2f("uv_actual", actual_u, actual_v);
                    self->program.uniform1f("color_blend", color_blend);
                    
                    // Draw outer glow halo (larger, more transparent)
                    {
                        int num_segments = 20;
                        std::vector<float> vertices;
                        std::vector<float> uv;
                        
                        float glow_radius = (block_size * scale) * 0.8f;  // Larger radius for glow
                        
                        // Center vertex
                        vertices.push_back(0.0f);
                        vertices.push_back(0.0f);
                        uv.push_back(block_center_u);
                        uv.push_back(block_center_v);
                        
                        // Circle vertices
                        for (int seg = 0; seg <= num_segments; seg++)
                        {
                            float angle = (seg / float(num_segments)) * 2.0f * M_PI;
                            vertices.push_back(std::cos(angle) * glow_radius);
                            vertices.push_back(std::sin(angle) * glow_radius);
                            uv.push_back(block_center_u);
                            uv.push_back(block_center_v);
                        }
                        
                        self->program.uniformMatrix4f("matrix",
                            wf::gles::output_transform(data.target) * model);
                        // Keep glow constant, color interpolation handled by shader
                        self->program.uniform1f("alpha", alpha * 1.5f);
                        self->program.attrib_pointer("position", 2, 0, vertices.data());
                        self->program.attrib_pointer("uv_in", 2, 0, uv.data());
                        GL_CALL(glDrawArrays(GL_TRIANGLE_FAN, 0, num_segments + 2));
                    }
                    
                    // Draw middle glow layer
                    {
                        int num_segments = 18;
                        std::vector<float> vertices;
                        std::vector<float> uv;
                        
                        float mid_radius = (block_size * scale) * 0.6f;
                        
                        // Center vertex
                        vertices.push_back(0.0f);
                        vertices.push_back(0.0f);
                        uv.push_back(block_center_u);
                        uv.push_back(block_center_v);
                        
                        // Circle vertices
                        for (int seg = 0; seg <= num_segments; seg++)
                        {
                            float angle = (seg / float(num_segments)) * 2.0f * M_PI;
                            vertices.push_back(std::cos(angle) * mid_radius);
                            vertices.push_back(std::sin(angle) * mid_radius);
                            uv.push_back(block_center_u);
                            uv.push_back(block_center_v);
                        }
                        
                        self->program.uniformMatrix4f("matrix",
                            wf::gles::output_transform(data.target) * model);
                        // Keep glow constant, color interpolation handled by shader
                        self->program.uniform1f("alpha", alpha * 2.5f);
                        self->program.attrib_pointer("position", 2, 0, vertices.data());
                        self->program.attrib_pointer("uv_in", 2, 0, uv.data());
                        GL_CALL(glDrawArrays(GL_TRIANGLE_FAN, 0, num_segments + 2));
                    }
                    
                    // Draw bright core particle
                    {
                        int num_segments = 16;
                        std::vector<float> vertices;
                        std::vector<float> uv;
                        
                        float particle_radius = (block_size * scale) / 2.5f;  // Smaller core
                        
                        // Center vertex
                        vertices.push_back(0.0f);
                        vertices.push_back(0.0f);
                        uv.push_back(block_center_u);
                        uv.push_back(block_center_v);
                        
                        // Circle vertices
                        for (int seg = 0; seg <= num_segments; seg++)
                        {
                            float angle = (seg / float(num_segments)) * 2.0f * M_PI;
                            vertices.push_back(std::cos(angle) * particle_radius);
                            vertices.push_back(std::sin(angle) * particle_radius);
                            uv.push_back(block_center_u);
                            uv.push_back(block_center_v);
                        }
                        
                        self->program.uniformMatrix4f("matrix",
                            wf::gles::output_transform(data.target) * model);
                        // Keep glow constant, color interpolation handled by shader
                        self->program.uniform1f("alpha", alpha * 4.0f);
                        self->program.attrib_pointer("position", 2, 0, vertices.data());
                        self->program.attrib_pointer("uv_in", 2, 0, uv.data());
                        GL_CALL(glDrawArrays(GL_TRIANGLE_FAN, 0, num_segments + 2));
                    }
                    
                    // Restore normal blending
                    GL_CALL(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));
                }

                self->program.deactivate();
            });
        }
    };

    shatter_transformer(wayfire_view view, wf::geometry_t bbox) : wf::scene::view_2d_transformer_t(view)
    {
        this->view = view;
        if (view->get_output())
        {
            output = view->get_output();
            output->render->add_effect(&pre_hook, wf::OUTPUT_EFFECT_PRE);
        }

        auto og = output->get_relative_geometry();
        animation_geometry = og;
        
        wf::gles::run_in_context([&]
        {
            program.compile(pixel_vert_source, pixel_frag_source);
        });

        // Initialize random seed
        std::srand(std::time(nullptr));
        
        // Create grid of pixel blocks
        int block_size = pixel_block_size;
        grid_width = (bbox.width + block_size - 1) / block_size;
        grid_height = (bbox.height + block_size - 1) / block_size;
        
        for (int row = 0; row < grid_height; row++)
        {
            for (int col = 0; col < grid_width; col++)
            {
                PixelBlock block;
                block.x = col * block_size;
                block.y = row * block_size;
                
                // Random velocities for varied movement in all directions
                block.velocity_x = (std::rand() / float(RAND_MAX) - 0.5f) * 2.0f;
                block.velocity_y = (std::rand() / float(RAND_MAX) - 0.5f) * 2.0f;
                
                // Random rotation
                block.rotation = 0.0f;
                block.rotation_speed = (std::rand() / float(RAND_MAX) - 0.5f) * 6.28f * 2.0f;
                
                // Random scale variation
                block.scale = 0.9f + (std::rand() / float(RAND_MAX)) * 0.2f;
                
                blocks.push_back(block);
                
                // Create particles for this block
                std::vector<Particle> block_particles;
                int num_particles = 5 + (std::rand() % 8);  // 5-12 particles per block for better glow
                for (int p = 0; p < num_particles; p++)
                {
                    Particle particle;
                    // Random position around block edges
                    particle.offset_x = (std::rand() / float(RAND_MAX)) * block_size - block_size / 2.0f;
                    particle.offset_y = (std::rand() / float(RAND_MAX)) * block_size - block_size / 2.0f;
                    // Random velocity trailing behind
                    particle.velocity_x = (std::rand() / float(RAND_MAX) - 0.5f) * 0.5f;
                    particle.velocity_y = (std::rand() / float(RAND_MAX) - 0.5f) * 0.5f;
                    particle.life = 0.5f + (std::rand() / float(RAND_MAX)) * 0.5f;  // varying lifetimes
                    particle.size = 3.0f + (std::rand() / float(RAND_MAX)) * 5.0f;  // 3-8 pixel size for more glow
                    block_particles.push_back(particle);
                }
                particles.push_back(block_particles);
            }
        }
    }

    wf::geometry_t get_bounding_box() override
    {
        return this->animation_geometry;
    }

    wf::effect_hook_t pre_hook = [=] ()
    {
        output->render->damage(animation_geometry);
    };

    void gen_render_instances(std::vector<render_instance_uptr>& instances,
        damage_callback push_damage, wf::output_t *shown_on) override
    {
        instances.push_back(std::make_unique<simple_node_render_instance_t>(
            this, push_damage, view));
    }

    void init_animation(bool hiding)
    {
        if (!hiding)
        {
            this->progression.reverse();
        }

        this->progression.start();
    }

    virtual ~shatter_transformer()
    {
        if (output)
        {
            output->render->rem_effect(&pre_hook);
        }

        wf::gles::run_in_context_if_gles([&]
        {
            program.free_resources();
        });
    }
};

class shatter_animation : public animation_base_t
{
    wayfire_view view;

  public:
    void init(wayfire_view view, wf::animation_description_t dur, animation_type type) override
    {
        this->view = view;
        pop_transformer(view);
        auto bbox = view->get_transformed_node()->get_bounding_box();
        auto tmgr = view->get_transformed_node();
        auto node = std::make_shared<shatter_transformer>(view, bbox);
        tmgr->add_transformer(node, wf::TRANSFORMER_HIGHLEVEL + 1, shatter_transformer_name);
        node->init_animation(type & WF_ANIMATE_HIDING_ANIMATION);
    }

    void pop_transformer(wayfire_view view)
    {
        if (view->get_transformed_node()->get_transformer(shatter_transformer_name))
        {
            view->get_transformed_node()->rem_transformer(shatter_transformer_name);
        }
    }

    bool step() override
    {
        if (!view)
        {
            return false;
        }

        auto tmgr = view->get_transformed_node();
        if (!tmgr)
        {
            return false;
        }

        if (auto tr =
                tmgr->get_transformer<shatter_transformer>(shatter_transformer_name))
        {
            auto running = tr->progression.running();
            if (!running)
            {
                pop_transformer(view);
                return false;
            }

            return running;
        }

        return false;
    }

    void reverse() override
    {
        if (auto tr =
                view->get_transformed_node()->get_transformer<shatter_transformer>(
                    shatter_transformer_name))
        {
            tr->progression.reverse();
        }
    }
};
}
}

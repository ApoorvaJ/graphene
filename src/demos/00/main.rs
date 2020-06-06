use ash::version::DeviceV1_0;
use ash::vk;
use glam::*;
use std::f32::consts::PI;

const DEGREES_TO_RADIANS: f32 = PI / 180.0;

#[allow(dead_code)]
struct UniformBuffer {
    mtx_model_to_clip: Mat4,
    mtx_model_to_view: Mat4,
    mtx_model_to_view_norm: Mat4,
    elapsed_seconds: f32,
}

fn main() {
    let ctx = graphene::Context::new(std::mem::size_of::<UniformBuffer>());
    // TODO: Set up pipeline and upload assets here

    ctx.run_loop(move |ctx, elapsed_seconds, frame_idx| {
        // Update uniform buffer
        let mtx_model_to_world =
            Mat4::from_rotation_y((160.0 + 20.0 * elapsed_seconds) * DEGREES_TO_RADIANS)
                * Mat4::from_rotation_z(180.0 * DEGREES_TO_RADIANS);
        let mtx_world_to_view = Mat4::from_translation(Vec3::new(0.0, 0.0, 3.0))
            * Mat4::from_rotation_x(20.0 * DEGREES_TO_RADIANS);
        let mtx_view_to_clip = {
            let width = ctx.facade.swapchain_textures[0].width;
            let height = ctx.facade.swapchain_textures[0].height;
            Mat4::perspective_lh(
                60.0 * DEGREES_TO_RADIANS,
                width as f32 / height as f32,
                0.01,
                100.0,
            )
        };

        let mtx_model_to_view = mtx_world_to_view * mtx_model_to_world;
        let ubos = [UniformBuffer {
            mtx_model_to_clip: mtx_view_to_clip * mtx_world_to_view * mtx_model_to_world,
            mtx_model_to_view,
            mtx_model_to_view_norm: mtx_model_to_view.inverse().transpose(),
            elapsed_seconds,
        }];

        ctx.uniform_buffers[frame_idx].upload_data(&ubos, 0, &ctx.gpu);

        // Build and execute render graph
        {
            let mut graph_builder = graphene::GraphBuilder::new();
            let pass_0 = graph_builder.add_pass(
                "forward lit",
                &vec![&ctx.facade.swapchain_textures[frame_idx]],
                Some(&ctx.facade.depth_texture),
                &ctx.shader_modules,
                &ctx.uniform_buffers[frame_idx],
                &ctx.environment_texture,
                ctx.environment_sampler,
            );

            let device = ctx.gpu.device.clone();
            let cmd_buf = ctx.command_buffers[frame_idx];
            let vertex_buf = ctx.mesh.vertex_buffer.vk_buffer;
            let index_buf = ctx.mesh.index_buffer.vk_buffer;
            let num_mesh_indices = ctx.mesh.index_buffer.num_elements as u32;
            let graph = ctx.build_graph(graph_builder);
            graph.begin_pass(pass_0, cmd_buf);
            unsafe {
                // Bind index and vertex buffers
                {
                    let vertex_buffers = [vertex_buf];
                    let offsets = [0_u64];
                    device.cmd_bind_vertex_buffers(cmd_buf, 0, &vertex_buffers, &offsets);
                    device.cmd_bind_index_buffer(cmd_buf, index_buf, 0, vk::IndexType::UINT32);
                }

                device.cmd_draw_indexed(cmd_buf, num_mesh_indices, 1, 0, 0, 0);
            }
            graph.end_pass(cmd_buf);
        }
    });
}

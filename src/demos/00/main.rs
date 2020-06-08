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
    let mut ctx = graphene::Context::new();
    let start_instant = std::time::Instant::now();
    let uniform_buffer_size = std::mem::size_of::<UniformBuffer>();

    let mesh = graphene::Mesh::load("assets/meshes/suzanne.glb", &ctx.gpu, ctx.command_pool);
    let environment_sampler = graphene::Sampler::new(&ctx.gpu);
    let environment_texture = ctx
        .new_texture(
            "environment map",
            graphene::TextureType::Image {
                path: String::from("assets/textures/env_carpentry_shop_02_2k.jpg"),
            },
        )
        .unwrap();

    let uniform_buffers: Vec<graphene::HostVisibleBuffer> = (0..ctx.facade.num_frames)
        .map(|_| {
            graphene::HostVisibleBuffer::new(
                uniform_buffer_size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                &ctx.gpu,
            )
        })
        .collect();

    loop {
        if !ctx.begin_frame() {
            break;
        }

        let elapsed_seconds = start_instant.elapsed().as_secs_f32();

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

        uniform_buffers[ctx.swapchain_idx].upload_data(&ubos, 0, &ctx.gpu);

        // Build and execute render graph
        {
            let mut graph_builder = graphene::GraphBuilder::new();
            let pass_0 = ctx
                .add_pass(
                    &mut graph_builder,
                    "forward lit",
                    &vec![&ctx.facade.swapchain_textures[ctx.swapchain_idx]],
                    Some(&ctx.facade.depth_texture),
                    &ctx.shader_modules,
                    &uniform_buffers[ctx.swapchain_idx],
                    environment_texture,
                    &environment_sampler,
                )
                .unwrap();

            let cmd_buf = ctx.command_buffers[ctx.swapchain_idx];
            let graph = ctx.build_graph(graph_builder);
            ctx.begin_pass(graph, pass_0);
            unsafe {
                // Bind index and vertex buffers
                {
                    let vertex_buffers = [mesh.vertex_buffer.vk_buffer];
                    let offsets = [0_u64];
                    ctx.gpu
                        .device
                        .cmd_bind_vertex_buffers(cmd_buf, 0, &vertex_buffers, &offsets);
                    ctx.gpu.device.cmd_bind_index_buffer(
                        cmd_buf,
                        mesh.index_buffer.vk_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                }

                ctx.gpu.device.cmd_draw_indexed(
                    cmd_buf,
                    mesh.index_buffer.num_elements as u32,
                    1,
                    0,
                    0,
                    0,
                );
            }
            ctx.end_pass(graph);
        }

        ctx.end_frame();
    }

    // TODO: Remove the necessity for this sync
    unsafe {
        ctx.gpu
            .device
            .device_wait_idle()
            .expect("Failed to wait device idle!");
    }
}

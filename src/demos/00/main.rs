use ash::version::DeviceV1_0;
use ash::vk;
use glam::*;
use std::f32::consts::PI;

const DEGREES_TO_RADIANS: f32 = PI / 180.0;

#[allow(dead_code)]
struct UniformBuffer {
    mtx_obj_to_clip: Mat4,
    mtx_norm_obj_to_world: Mat4,
    elapsed_seconds: f32,
}

fn main() {
    let mut ctx = graphene::Context::new();
    let start_instant = std::time::Instant::now();

    let mesh = graphene::Mesh::load("assets/meshes/suzanne.glb", &ctx.gpu, ctx.command_pool);
    let depth_texture = ctx
        .new_texture_relative_size(
            "depth",
            1.0,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
        )
        .unwrap();
    let environment_sampler = graphene::Sampler::new(&ctx.gpu);
    let environment_texture = ctx
        .new_texture_from_file(
            "environment map",
            "assets/textures/env_carpentry_shop_02_2k.jpg",
        )
        .unwrap();

    loop {
        if !ctx.begin_frame() {
            break;
        }

        let elapsed_seconds = start_instant.elapsed().as_secs_f32();

        // Build and execute render graph
        let mut graph_builder = graphene::GraphBuilder::new();
        let uniform_buffer = graph_builder
            .new_uniform_buffer("uniform buffer", std::mem::size_of::<UniformBuffer>())
            .unwrap();
        let pass_0 = ctx
            .add_pass(
                &mut graph_builder,
                "forward lit",
                &vec![&ctx.facade.swapchain_textures[ctx.swapchain_idx]],
                Some(depth_texture),
                &ctx.shader_modules,
                uniform_buffer,
                environment_texture,
                &environment_sampler,
            )
            .unwrap();

        let cmd_buf = ctx.command_buffers[ctx.swapchain_idx];
        let graph = ctx.build_graph(graph_builder);
        ctx.begin_pass(graph, pass_0);
        // Update uniform buffer
        {
            // let mtx_model_to_world =
            //     Mat4::from_rotation_y((160.0 + 20.0 * elapsed_seconds) * DEGREES_TO_RADIANS)
            //         * Mat4::from_rotation_z(180.0 * DEGREES_TO_RADIANS);
            // let mtx_world_to_view = Mat4::from_translation(Vec3::new(0.0, 0.0, 3.0))
            //     * Mat4::from_rotation_x(20.0 * DEGREES_TO_RADIANS);
            let obj_pos = Vec3::new(0.0, 0.0, 0.0);
            let obj_rot = Quat::from_rotation_z(elapsed_seconds * 0.3);
            // let obj_rot = Quat::from_rotation_z(180.0 * DEGREES_TO_RADIANS);
            let mtx_obj_to_world = Mat4::from_rotation_x(90.0 * DEGREES_TO_RADIANS)
                * Mat4::from_translation(obj_pos)
                * Mat4::from_quat(obj_rot)
                * Mat4::from_rotation_x(90.0 * DEGREES_TO_RADIANS);
            let mtx_world_to_view = Mat4::from_translation(Vec3::new(0.0, 0.0, 3.0));
            let mtx_view_to_clip = {
                let width = ctx.facade.swapchain_width;
                let height = ctx.facade.swapchain_height;
                Mat4::perspective_lh(
                    60.0 * DEGREES_TO_RADIANS,
                    width as f32 / height as f32,
                    0.01,
                    100.0,
                )
            };

            let mtx_norm_obj_to_world = (
                //
                // Mat4::from_quat(obj_rot).inverse()
                // * mtx_obj_to_world
                // ---
                // Mat4::from_rotation_x(-90.0 * DEGREES_TO_RADIANS)
                // ---
                Mat4::from_rotation_x(-90.0 * DEGREES_TO_RADIANS)
                // ---
                * Mat4::from_quat(obj_rot).inverse()
                //
            )
            .inverse()
            .transpose(); // TODO: Orthogonal optimization?

            let ubos = [UniformBuffer {
                mtx_obj_to_clip: mtx_view_to_clip * mtx_world_to_view * mtx_obj_to_world,
                mtx_norm_obj_to_world,
                elapsed_seconds,
            }];

            ctx.upload_data(graph, uniform_buffer, &ubos);
        }

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

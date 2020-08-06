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
    viewport_w: f32,
    viewport_h: f32,
}

fn execute_pass(
    ctx: &mut graphene::Context,
    elapsed_seconds: f32,
    uniform_buffer: graphene::BufferHandle,
    cmd_buf: vk::CommandBuffer,
    mesh: &graphene::Mesh,
) {
    // Update uniform buffer
    {
        let cam_pos = Vec3::new(0.0, -4.5, 0.0);
        let cam_rot = Quat::from_rotation_z((elapsed_seconds * 1.5).sin() * 0.1 * PI);
        let obj_pos = Vec3::new(0.0, 0.0, 0.0);
        let obj_rot = Quat::from_rotation_z(elapsed_seconds * 0.3);
        let obj_scale = Vec3::new(1.0, 1.0, 1.0);

        let mtx_rot_scale = Mat4::from_quat(obj_rot)
            * Mat4::from_scale(obj_scale)
            * Mat4::from_rotation_x(90.0 * DEGREES_TO_RADIANS);
        let mtx_obj_to_world = Mat4::from_rotation_x(90.0 * DEGREES_TO_RADIANS)
            * Mat4::from_translation(obj_pos)
            * mtx_rot_scale;
        let mtx_world_to_view = Mat4::from_rotation_x(90.0 * DEGREES_TO_RADIANS)
            * Mat4::from_quat(cam_rot)
            * Mat4::from_translation(-cam_pos)
            * Mat4::from_rotation_x(-90.0 * DEGREES_TO_RADIANS);
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

        /* This matrix is an orthogonal matrix if scaling is uniform, in
        which case the inverse transpose is the same as the matrix itself.
        // Pass 0
        However, we want to support non-uniform scaling, so we
        do the inverse transpose. */
        let mtx_norm_obj_to_world = mtx_rot_scale.inverse().transpose();

        let ubos = [UniformBuffer {
            mtx_obj_to_clip: mtx_view_to_clip * mtx_world_to_view * mtx_obj_to_world,
            mtx_norm_obj_to_world,
            elapsed_seconds,
            viewport_w: ctx.facade.swapchain_width as f32,
            viewport_h: ctx.facade.swapchain_height as f32,
        }];

        ctx.upload_data(uniform_buffer, &ubos);
    }
    // Bind index and vertex buffers
    unsafe {
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

        ctx.gpu
            .device
            .cmd_draw_indexed(cmd_buf, mesh.index_buffer.num_elements as u32, 1, 0, 0, 0);
    }
}

fn main() {
    let mut ctx = graphene::Context::new();
    let start_instant = std::time::Instant::now();

    // TODO: Having to pass in debug_utils here is a little messy. Streamline.
    let mesh = graphene::Mesh::load(
        "suzanne",
        "assets/meshes/suzanne.glb",
        &ctx.gpu,
        ctx.command_pool,
        &ctx.debug_utils,
    );
    let depth_image = ctx
        .new_image_relative_size(
            "image_depth",
            1.0,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
        )
        .unwrap();
    let temp_image = ctx
        .new_image_relative_size(
            "image_temp",
            1.0,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageAspectFlags::COLOR,
        )
        .unwrap();
    let environment_sampler = graphene::Sampler::new(&ctx.gpu);
    let environment_image = ctx
        .new_image_from_file(
            "image_environment_map",
            "assets/textures/env_carpentry_shop_02_2k.jpg",
        )
        .unwrap();

    let shader_vertex = ctx
        .new_shader(
            "shader_vertex",
            graphene::ShaderStage::Vertex,
            "default.vert",
        )
        .unwrap();
    let shader_fullscreen_triangle_vertex = ctx
        .new_shader(
            "fullscreen_triangle_vertex",
            graphene::ShaderStage::Vertex,
            "fullscreen_triangle.vert",
        )
        .unwrap();
    let shader_default = ctx
        .new_shader(
            "shader_default",
            graphene::ShaderStage::Fragment,
            "default.frag",
        )
        .unwrap();
    let shader_aberration = ctx
        .new_shader(
            "shader_aberration",
            graphene::ShaderStage::Fragment,
            "chromatic_aberration.frag",
        )
        .unwrap();

    // TODO: Avoid having to create the vec. Automatically
    // creating a unique uniform buffer per frame
    let uniform_buffers: Vec<graphene::BufferHandle> = (0..ctx.facade.num_frames)
        .map(|i| {
            ctx.new_buffer(
                &format!("buffer_uniform_{}", i),
                std::mem::size_of::<UniformBuffer>(),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )
            .unwrap()
        })
        .collect();

    loop {
        if !ctx.begin_frame() {
            break;
        }

        let elapsed_seconds = start_instant.elapsed().as_secs_f32();
        let cmd_buf = ctx.command_buffers[ctx.swapchain_idx];

        let uniform_buffer = uniform_buffers[ctx.swapchain_idx];

        // Build and execute render graph
        let mut graph_builder = graphene::GraphBuilder::new();
        let pass_lit = ctx
            .add_pass(
                &mut graph_builder,
                "lit",
                shader_vertex,
                shader_default,
                &[temp_image],
                Some(depth_image),
                uniform_buffer,
                environment_image,
                &environment_sampler,
            )
            .unwrap();
        let pass_post = ctx
            .add_pass(
                &mut graph_builder,
                "post",
                shader_fullscreen_triangle_vertex,
                shader_aberration,
                &[ctx.facade.swapchain_images[ctx.swapchain_idx]],
                Some(depth_image),
                uniform_buffer,
                temp_image,
                &environment_sampler,
            )
            .unwrap();

        let graph = ctx.build_graph(graph_builder);
        // Pass 0
        ctx.begin_pass(graph, pass_lit);
        execute_pass(&mut ctx, elapsed_seconds, uniform_buffer, cmd_buf, &mesh);
        ctx.end_pass(graph);
        // Layout transition (TODO: Do this automatically in the render graph)
        {
            let img = ctx.image_list.get_image_from_handle(temp_image).unwrap();
            img.image.transition_image_layout(
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                cmd_buf,
            );
        }
        // Pass 1
        ctx.begin_pass(graph, pass_post);
        unsafe {
            ctx.gpu.device.cmd_draw(cmd_buf, 3, 1, 0, 0);
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

use glam::*;
use std::f32::consts::PI;

const DEGREES_TO_RADIANS: f32 = PI / 180.0;

#[allow(dead_code)]
struct UniformBuffer {
    mtx_model_to_clip: Mat4,
    mtx_model_to_view: Mat4,
    mtx_model_to_view_norm: Mat4,
}

fn main() {
    let ctx = grapheme::Context::new(std::mem::size_of::<UniformBuffer>());
    // TODO: Set up pipeline and upload assets here

    ctx.run_loop(move |ctx, elapsed_seconds, frame_idx| {
        // Update uniform buffer
        let mtx_model_to_world =
            Mat4::from_rotation_y((160.0 + 20.0 * elapsed_seconds) * DEGREES_TO_RADIANS)
                * Mat4::from_rotation_z(180.0 * DEGREES_TO_RADIANS);
        let mtx_world_to_view = Mat4::from_translation(Vec3::new(0.0, 0.0, 4.0))
            * Mat4::from_rotation_x(20.0 * DEGREES_TO_RADIANS);
        let mtx_view_to_clip = {
            let extent = &ctx.facade.swapchain_extent;
            Mat4::perspective_lh(
                60.0 * DEGREES_TO_RADIANS,
                extent.width as f32 / extent.height as f32,
                0.01,
                100.0,
            )
        };

        let mtx_model_to_view = mtx_world_to_view * mtx_model_to_world;
        let ubos = [UniformBuffer {
            mtx_model_to_clip: mtx_view_to_clip * mtx_world_to_view * mtx_model_to_world,
            mtx_model_to_view,
            mtx_model_to_view_norm: mtx_model_to_view.inverse().transpose(),
        }];

        ctx.uniform_buffers[frame_idx].upload_data(&ubos, 0, &ctx.gpu);
    });
}

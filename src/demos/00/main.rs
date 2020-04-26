use ash::version::DeviceV1_0;
use ash::vk;
use glam::*;
use std::f32::consts::PI;

const DEGREES_TO_RADIANS: f32 = PI / 180.0;

#[allow(dead_code)]
struct UniformBuffer {
    mtx_world_to_clip: Mat4,
}

fn main() {
    let ctx = grapheme::Context::new(std::mem::size_of::<UniformBuffer>());
    // TODO: Set up pipeline and upload assets here

    ctx.run_loop(move |ctx, elapsed_seconds, frame_idx| {
        // Update uniform buffer
        let extent = &ctx.facade.swapchain_extent;
        let ubos = [UniformBuffer {
            mtx_world_to_clip: Mat4::perspective_lh(
                60.0 * DEGREES_TO_RADIANS,
                extent.width as f32 / extent.height as f32,
                0.01,
                100.0,
            ) * Mat4::from_translation(Vec3::new(0.0, 0.0, 4.0))
                * Mat4::from_rotation_x(20.0 * DEGREES_TO_RADIANS)
                * Mat4::from_rotation_y((160.0 + 20.0 * elapsed_seconds) * DEGREES_TO_RADIANS)
                * Mat4::from_rotation_z(180.0 * DEGREES_TO_RADIANS),
        }];

        let buffer_size = (std::mem::size_of::<UniformBuffer>() * ubos.len()) as u64;

        unsafe {
            let data_ptr = ctx
                .gpu
                .device
                .map_memory(
                    ctx.uniform_buffers[frame_idx].memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map memory.") as *mut UniformBuffer;

            data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

            ctx.gpu
                .device
                .unmap_memory(ctx.uniform_buffers[frame_idx].memory);
        }
    });
}

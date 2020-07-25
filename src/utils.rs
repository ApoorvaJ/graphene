use crate::*;
use std::os::raw::c_char;

pub fn vk_to_string(raw_string_array: &[c_char]) -> String {
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}

pub fn begin_single_use_command_buffer(
    device: &ash::Device,
    command_pool: vk::CommandPool,
) -> vk::CommandBuffer {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&allocate_info)
            .expect("Failed to allocate Command Buffers!")
    }[0];

    let begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .expect("Failed to begin recording Command Buffer at beginning!");
    }

    command_buffer
}

pub fn end_single_use_command_buffer(
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    gpu: &Gpu,
) {
    unsafe {
        gpu.device
            .end_command_buffer(command_buffer)
            .expect("Failed to record end-command-buffer");
    }

    let command_buffers = [command_buffer];

    let submit_info = [vk::SubmitInfo {
        command_buffer_count: command_buffers.len() as u32,
        p_command_buffers: command_buffers.as_ptr(),
        ..Default::default()
    }];

    unsafe {
        gpu.device
            .queue_submit(gpu.graphics_queue, &submit_info, vk::Fence::null())
            .expect("Failed to Queue Submit!");
        gpu.device
            .queue_wait_idle(gpu.graphics_queue)
            .expect("Failed to wait Queue idle!");
        gpu.device
            .free_command_buffers(command_pool, &command_buffers);
    }
}

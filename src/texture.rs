use crate::*;

pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub aspect_flags: vk::ImageAspectFlags,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub opt_device_memory: Option<vk::DeviceMemory>, // None if we didn't manually allocate memory, e.g. in the case of swapchain images
    pub name: String,
    pub device: ash::Device,
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.image_view, None);
            if let Some(mem) = self.opt_device_memory {
                self.device.destroy_image(self.image, None); // Only destroy the image if we allocated it in the first place
                self.device.free_memory(mem, None);
            }
        }
    }
}

impl Texture {
    pub fn new(
        gpu: &Gpu,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect_flags: vk::ImageAspectFlags,
        name: &str,
        debug_marker: &DebugUtils,
    ) -> Texture {
        let device = gpu.device.clone();

        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        let image = unsafe {
            device
                .create_image(&image_create_info, None)
                .expect("Failed to create texture image.")
        };

        let image_memory_requirement = unsafe { device.get_image_memory_requirements(image) };
        let memory_type_index = gpu
            .memory_properties
            .memory_types
            .iter()
            .enumerate()
            .position(|(i, &memory_type)| {
                (image_memory_requirement.memory_type_bits & (1 << i)) > 0
                    && memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .expect("Failed to find suitable memory type.") as u32;

        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(image_memory_requirement.size)
            .memory_type_index(memory_type_index);
        let device_memory = unsafe {
            device
                .allocate_memory(&memory_allocate_info, None)
                .expect("Failed to allocate texture image memory.")
        };

        unsafe {
            device
                .bind_image_memory(image, device_memory, 0)
                .expect("Failed to bind image memory.");
        }

        let image_view = {
            let imageview_create_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspect_flags,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(image);

            unsafe {
                gpu.device
                    .create_image_view(&imageview_create_info, None)
                    .expect("Failed to create Image View!")
            }
        };

        debug_marker.set_image_name(image, name);

        Texture {
            width,
            height,
            format,
            usage,
            aspect_flags,
            image,
            image_view,
            opt_device_memory: Some(device_memory),
            device,
            name: String::from(name),
        }
    }

    fn transition_image_layout(
        &self,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        command_pool: vk::CommandPool,
        gpu: &Gpu,
    ) {
        // TODO: Hoist this out
        let command_buffer = begin_single_use_command_buffer(&gpu.device, command_pool);

        let src_access_mask;
        let dst_access_mask;
        let source_stage;
        let destination_stage;

        if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            src_access_mask = vk::AccessFlags::empty();
            dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::TRANSFER;
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            dst_access_mask = vk::AccessFlags::SHADER_READ;
            source_stage = vk::PipelineStageFlags::TRANSFER;
            destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        } else {
            panic!("Unsupported layout transition!")
        }

        let image_barriers = [vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask,
            dst_access_mask,
            old_layout,
            new_layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        }];

        unsafe {
            gpu.device.cmd_pipeline_barrier(
                command_buffer,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_barriers,
            );
        }

        // TODO: Hoist this out
        end_single_use_command_buffer(command_buffer, command_pool, gpu);
    }

    pub fn new_from_image(
        gpu: &Gpu,
        path: &std::path::Path,
        command_pool: vk::CommandPool,
        name: &str,
        debug_utils: &DebugUtils,
    ) -> Texture {
        use image::GenericImageView;
        let mut image_object = image::open(path).unwrap();
        image_object = image_object.flipv();

        let (image_width, image_height) = (image_object.width(), image_object.height());
        let image_size =
            std::mem::size_of::<u8>() * image_width as usize * image_height as usize * 4;
        let image_data = image_object.to_rgba().into_raw();

        if image_size <= 0 {
            panic!("Failed to load texture image!")
        }

        let texture = Texture::new(
            gpu,
            image_width,
            image_height,
            vk::Format::R8G8B8A8_UNORM, // TODO: Derive format from file or take as an argument
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            name,
            debug_utils,
        );

        let staging_buffer =
            HostVisibleBuffer::new(image_size, vk::BufferUsageFlags::TRANSFER_SRC, gpu);
        staging_buffer.upload_data(&image_data, 0, gpu);

        texture.transition_image_layout(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            command_pool,
            gpu,
        );

        // Copy buffer to image
        {
            let command_buffer = begin_single_use_command_buffer(&gpu.device, command_pool);

            let buffer_image_regions = [vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_extent: vk::Extent3D {
                    width: image_width,
                    height: image_height,
                    depth: 1,
                },
                buffer_offset: 0,
                buffer_image_height: 0,
                buffer_row_length: 0,
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            }];

            unsafe {
                gpu.device.cmd_copy_buffer_to_image(
                    command_buffer,
                    staging_buffer.vk_buffer,
                    texture.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &buffer_image_regions,
                );
            }

            end_single_use_command_buffer(command_buffer, command_pool, &gpu);
        }

        texture.transition_image_layout(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            command_pool,
            gpu,
        );

        texture
    }
}

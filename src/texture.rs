use crate::*;

pub struct Texture {
    pub image: vk::Image,
    pub format: vk::Format,
    pub device_memory: vk::DeviceMemory,
    pub image_view: vk::ImageView,
    device: ash::Device,
}

impl Texture {
    pub fn new(
        gpu: &Gpu,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect_flags: vk::ImageAspectFlags,
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

        Texture {
            image,
            format,
            device_memory,
            image_view,
            device,
        }
    }

    pub fn destroy(&self) {
        unsafe {
            self.device.destroy_image_view(self.image_view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.device_memory, None);
        }
    }
}

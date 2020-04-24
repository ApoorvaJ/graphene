use crate::*;

pub struct Facade {
    // Surface info. Changes with resolution.
    pub surface_caps: vk::SurfaceCapabilitiesKHR,
    pub surface_formats: Vec<vk::SurfaceFormatKHR>,
    // - Swapchain
    pub num_frames: usize,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub _swapchain_images: Vec<vk::Image>,
    pub swapchain_imageviews: Vec<vk::ImageView>,
    pub depth_format: vk::Format,
    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub depth_image_memory: vk::DeviceMemory,
    // - Synchronization primitives. these aren't really resolution-dependent
    //   and could technically be moved outside the struct. They are kept here
    //   because they're closely related to the rest of the members.
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub command_buffer_complete_fences: Vec<vk::Fence>,
}

// TODO: Move to a more sensible file
pub fn create_image(
    gpu: &Gpu,
    width: u32,
    height: u32,
    mip_levels: u32,
    num_samples: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    required_memory_properties: vk::MemoryPropertyFlags,
) -> (vk::Image, vk::DeviceMemory) {
    let image_create_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .mip_levels(mip_levels)
        .array_layers(1)
        .samples(num_samples)
        .tiling(tiling)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    let texture_image = unsafe {
        gpu.device
            .create_image(&image_create_info, None)
            .expect("Failed to create texture image.")
    };

    let image_memory_requirement =
        unsafe { gpu.device.get_image_memory_requirements(texture_image) };
    let memory_type_index = gpu
        .memory_properties
        .memory_types
        .iter()
        .enumerate()
        .position(|(i, &memory_type)| {
            (image_memory_requirement.memory_type_bits & (1 << i)) > 0
                && memory_type
                    .property_flags
                    .contains(required_memory_properties)
        })
        .expect("Failed to find suitable memory type.") as u32;

    let memory_allocate_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(image_memory_requirement.size)
        .memory_type_index(memory_type_index);
    let texture_image_memory = unsafe {
        gpu.device
            .allocate_memory(&memory_allocate_info, None)
            .expect("Failed to allocate texture image memory.")
    };

    unsafe {
        gpu.device
            .bind_image_memory(texture_image, texture_image_memory, 0)
            .expect("Failed to bind image memory.");
    }

    (texture_image, texture_image_memory)
}

pub fn create_image_view(
    gpu: &Gpu,
    image: vk::Image,
    format: vk::Format,
    aspect_flags: vk::ImageAspectFlags,
    mip_levels: u32,
) -> vk::ImageView {
    let imageview_create_info = vk::ImageViewCreateInfo::builder()
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect_flags,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image(image);

    unsafe {
        gpu.device
            .create_image_view(&imageview_create_info, None)
            .expect("Failed to create Image View!")
    }
}

impl Facade {
    pub fn new(
        window: &winit::window::Window,
        surface: vk::SurfaceKHR,
        gpu: &Gpu,
        ext_surface: &ash::extensions::khr::Surface,
        ext_swapchain: &ash::extensions::khr::Swapchain,
    ) -> Facade {
        // # Get surface info
        let surface_caps = unsafe {
            ext_surface
                .get_physical_device_surface_capabilities(gpu.physical_device, surface)
                .expect("Failed to query for surface capabilities.")
        };

        let surface_formats = unsafe {
            ext_surface
                .get_physical_device_surface_formats(gpu.physical_device, surface)
                .expect("Failed to query for surface formats.")
        };

        // # Create swapchain
        let (num_frames, swapchain, swapchain_format, swapchain_extent, swapchain_images) = {
            // Set number of images in swapchain
            let num_frames = surface_caps.min_image_count + 1;

            // Choose swapchain format (i.e. color buffer format)
            let (swapchain_format, swapchain_color_space) = {
                let surface_format: vk::SurfaceFormatKHR = {
                    *surface_formats
                        .iter()
                        .find(|&f| {
                            f.format == vk::Format::B8G8R8A8_SRGB
                                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                        })
                        .unwrap_or(&surface_formats[0])
                };
                (surface_format.format, surface_format.color_space)
            };

            // Choose extent
            let extent = {
                if surface_caps.current_extent.width == u32::max_value() {
                    let window_size = window.inner_size();
                    vk::Extent2D {
                        width: (window_size.width as u32)
                            .max(surface_caps.min_image_extent.width)
                            .min(surface_caps.max_image_extent.width),
                        height: (window_size.height as u32)
                            .max(surface_caps.min_image_extent.height)
                            .min(surface_caps.max_image_extent.height),
                    }
                } else {
                    surface_caps.current_extent
                }
            };

            // Present mode
            let present_mode: vk::PresentModeKHR = vk::PresentModeKHR::FIFO;

            let mut info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(num_frames)
                .image_format(swapchain_format)
                .image_color_space(swapchain_color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                // TODO: Investigate:
                // The vulkan tutorial sets this as `pre_transform(gpu.surface_caps.current_transform)`.
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true); // Allow Vulkan to discard operations outside of the renderable space

            // Sharing mode
            let indices = [gpu.graphics_queue_idx, gpu.present_queue_idx];
            if gpu.graphics_queue_idx != gpu.present_queue_idx {
                info = info
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&indices);
            } else {
                // Graphics and present are the same queue, so it can have
                // exclusive access to the swapchain
                info = info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
            }

            let swapchain = unsafe {
                ext_swapchain
                    .create_swapchain(&info, None)
                    .expect("Failed to create swapchain.")
            };

            let images = unsafe {
                ext_swapchain
                    .get_swapchain_images(swapchain)
                    .expect("Failed to get swapchain images.")
            };

            (num_frames, swapchain, swapchain_format, extent, images)
        };

        // # Create swapchain image views
        let swapchain_imageviews = {
            let imageviews: Vec<vk::ImageView> = swapchain_images
                .iter()
                .map(|&image| {
                    let info = vk::ImageViewCreateInfo::builder()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(swapchain_format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::IDENTITY,
                            g: vk::ComponentSwizzle::IDENTITY,
                            b: vk::ComponentSwizzle::IDENTITY,
                            a: vk::ComponentSwizzle::IDENTITY,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });

                    unsafe {
                        gpu.device
                            .create_image_view(&info, None)
                            .expect("Failed to create image view.")
                    }
                })
                .collect();

            imageviews
        };

        // # Create depth buffer
        let depth_format = vk::Format::D32_SFLOAT;
        let (depth_image, depth_image_view, depth_image_memory) = {
            let (depth_image, depth_image_memory) = create_image(
                &gpu,
                swapchain_extent.width,
                swapchain_extent.height,
                1,
                vk::SampleCountFlags::TYPE_1,
                depth_format,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            let depth_image_view = create_image_view(
                &gpu,
                depth_image,
                depth_format,
                vk::ImageAspectFlags::DEPTH,
                1,
            );

            (depth_image, depth_image_view, depth_image_memory)
        };

        // # Synchronization primitives
        let (
            image_available_semaphores,
            render_finished_semaphores,
            command_buffer_complete_fences,
        ) = {
            let mut image_available_semaphores = Vec::new();
            let mut render_finished_semaphores = Vec::new();
            let mut command_buffer_complete_fences = Vec::new();
            let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
            let fence_create_info =
                vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

            for _ in 0..num_frames {
                unsafe {
                    image_available_semaphores.push(
                        gpu.device
                            .create_semaphore(&semaphore_create_info, None)
                            .expect("Failed to create Semaphore Object!"),
                    );
                    render_finished_semaphores.push(
                        gpu.device
                            .create_semaphore(&semaphore_create_info, None)
                            .expect("Failed to create Semaphore Object!"),
                    );
                    command_buffer_complete_fences.push(
                        gpu.device
                            .create_fence(&fence_create_info, None)
                            .expect("Failed to create Fence Object!"),
                    );
                }
            }
            (
                image_available_semaphores,
                render_finished_semaphores,
                command_buffer_complete_fences,
            )
        };

        Facade {
            surface_caps,
            surface_formats,
            num_frames: num_frames as usize,
            swapchain,
            swapchain_format,
            swapchain_extent,
            _swapchain_images: swapchain_images,
            swapchain_imageviews,
            depth_format,
            depth_image,
            depth_image_view,
            depth_image_memory,
            image_available_semaphores,
            render_finished_semaphores,
            command_buffer_complete_fences,
        }
    }

    pub fn destroy(&self, gpu: &Gpu, ext_swapchain: &ash::extensions::khr::Swapchain) {
        unsafe {
            for i in 0..self.num_frames {
                gpu.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                gpu.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                gpu.device
                    .destroy_fence(self.command_buffer_complete_fences[i], None);
            }
            for &imageview in self.swapchain_imageviews.iter() {
                gpu.device.destroy_image_view(imageview, None);
            }

            gpu.device.destroy_image_view(self.depth_image_view, None);
            gpu.device.destroy_image(self.depth_image, None);
            gpu.device.free_memory(self.depth_image_memory, None);

            ext_swapchain.destroy_swapchain(self.swapchain, None);
        }
    }
}

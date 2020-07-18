use crate::*;

pub struct Facade {
    device: ash::Device,
    // Surface info. Changes with resolution.
    pub surface_caps: vk::SurfaceCapabilitiesKHR,
    pub surface_formats: Vec<vk::SurfaceFormatKHR>,
    // Swapchain
    pub num_frames: usize,
    pub swapchain_width: u32,
    pub swapchain_height: u32,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_textures: Vec<TextureHandle>, // Color textures that are presented to the screen
    // Synchronization primitives. These aren't really resolution-dependent
    // and could technically be moved outside the struct. They are kept here
    // because they're closely related to the rest of the members.
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub command_buffer_complete_fences: Vec<vk::Fence>,

    pub ext_swapchain: ash::extensions::khr::Swapchain,
}

impl Facade {
    pub fn new(
        basis: &Basis,
        gpu: &Gpu,
        window: &winit::window::Window,
        texture_list: &mut Vec<InternalTexture>,
    ) -> Facade {
        let device = gpu.device.clone();
        let ext_swapchain = ash::extensions::khr::Swapchain::new(&basis.instance, &device);

        // # Get surface info
        let surface_caps = unsafe {
            basis
                .ext_surface
                .get_physical_device_surface_capabilities(gpu.physical_device, basis.surface)
                .expect("Failed to query for surface capabilities.")
        };

        let surface_formats = unsafe {
            basis
                .ext_surface
                .get_physical_device_surface_formats(gpu.physical_device, basis.surface)
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
                .surface(basis.surface)
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
                        device
                            .create_image_view(&info, None)
                            .expect("Failed to create image view.")
                    }
                })
                .collect();

            imageviews
        };

        // Add swapchain textures to the context's texture list
        let swapchain_textures = (0..num_frames)
            .map(|i| {
                let texture = Texture {
                    width: swapchain_extent.width,
                    height: swapchain_extent.height,
                    format: swapchain_format,
                    usage: vk::ImageUsageFlags::empty(),
                    aspect_flags: vk::ImageAspectFlags::empty(),
                    image: swapchain_images[i as usize],
                    image_view: swapchain_imageviews[i as usize],
                    opt_device_memory: None, // This memory is not allocated by us. It is part of the swapchain.
                    device: device.clone(),
                };
                let name = String::from(&format!("swapchain_{}", i));
                let hash: u64 = {
                    let mut hasher = DefaultHasher::new();
                    name.hash(&mut hasher);
                    hasher.finish()
                };
                let handle = TextureHandle(hash);
                texture_list.push(InternalTexture {
                    handle,
                    name,
                    texture,
                    kind: TextureKind::Swapchain,
                });

                handle
            })
            .collect();

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
                        device
                            .create_semaphore(&semaphore_create_info, None)
                            .expect("Failed to create Semaphore Object!"),
                    );
                    render_finished_semaphores.push(
                        device
                            .create_semaphore(&semaphore_create_info, None)
                            .expect("Failed to create Semaphore Object!"),
                    );
                    command_buffer_complete_fences.push(
                        device
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
            device,
            surface_caps,
            surface_formats,
            num_frames: num_frames as usize,
            swapchain_width: swapchain_extent.width,
            swapchain_height: swapchain_extent.height,
            swapchain,
            swapchain_textures,
            image_available_semaphores,
            render_finished_semaphores,
            command_buffer_complete_fences,
            ext_swapchain,
        }
    }

    pub fn destroy(&self, texture_list: &mut Vec<InternalTexture>) {
        unsafe {
            for i in 0..self.num_frames {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device
                    .destroy_fence(self.command_buffer_complete_fences[i], None);
            }

            self.ext_swapchain.destroy_swapchain(self.swapchain, None);
        }
        // Delete swapchain textures from texture list
        texture_list.retain(|t| t.kind != TextureKind::Swapchain);
    }
}

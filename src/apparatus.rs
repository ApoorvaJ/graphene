use crate::*;

// This is required because the `vk::ShaderModuleCreateInfo` struct's `p_code`
// member expects a *u32, but `include_bytes!()` produces a Vec<u8>.
// TODO: Investigate how to properly address this.
#[allow(clippy::cast_ptr_alignment)]
fn create_shader_module(device: &ash::Device, code: Vec<u8>) -> vk::ShaderModule {
    let shader_module_create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ShaderModuleCreateFlags::empty(),
        code_size: code.len(),
        p_code: code.as_ptr() as *const u32,
    };

    unsafe {
        device
            .create_shader_module(&shader_module_create_info, None)
            .expect("Failed to create shader module.")
    }
}

// Resolution-dependent rendering stuff.
// TODO: Find a better name for this?
pub struct Apparatus {
    // - Surface capabilities and formats
    pub _surface_caps: vk::SurfaceCapabilitiesKHR,
    pub _surface_formats: Vec<vk::SurfaceFormatKHR>,
    // - Swapchain
    pub swapchain: vk::SwapchainKHR,
    pub _swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub _swapchain_images: Vec<vk::Image>,
    pub swapchain_imageviews: Vec<vk::ImageView>,
    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub depth_image_memory: vk::DeviceMemory,
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    // - Pipelines
    pub pipeline_layout: vk::PipelineLayout,
    pub graphics_pipeline: vk::Pipeline,
    // - Commands
    pub command_buffers: Vec<vk::CommandBuffer>,
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

impl Apparatus {
    pub fn new(
        window: &winit::window::Window,
        surface: vk::SurfaceKHR,
        gpu: &Gpu,
        command_pool: vk::CommandPool,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
        num_indices: u32,
        uniform_buffer_layout: vk::DescriptorSetLayout,
        descriptor_sets: &[vk::DescriptorSet],
        ext_surface: &ash::extensions::khr::Surface,
        ext_swapchain: &ash::extensions::khr::Swapchain,
    ) -> Apparatus {
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
        let (swapchain, swapchain_format, swapchain_extent, swapchain_images) = {
            // Set number of images in swapchain
            let image_count = surface_caps.min_image_count.max(NUM_FRAMES as u32);

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
            let present_mode: vk::PresentModeKHR = {
                *gpu.present_modes
                    .iter()
                    .find(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
                    .unwrap_or(&vk::PresentModeKHR::FIFO)
            };

            let mut info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(image_count)
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

            (swapchain, swapchain_format, extent, images)
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

        // # Create render pass
        let render_pass = {
            let attachments = vec![
                // Color attachment
                vk::AttachmentDescription {
                    format: swapchain_format,
                    flags: vk::AttachmentDescriptionFlags::empty(),
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::DONT_CARE,
                    store_op: vk::AttachmentStoreOp::STORE,
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                },
                // Depth attachment
                vk::AttachmentDescription {
                    format: depth_format,
                    flags: vk::AttachmentDescriptionFlags::empty(),
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::DONT_CARE,
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE, // ?
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE, // ?
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                },
            ];

            let color_attachment_ref = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            let depth_attachment_ref = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };

            let subpasses = [vk::SubpassDescription {
                pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
                color_attachment_count: 1,
                p_color_attachments: color_attachment_ref.as_ptr(),
                p_depth_stencil_attachment: &depth_attachment_ref,
                ..Default::default()
            }];

            let renderpass_create_info = vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses);

            unsafe {
                gpu.device
                    .create_render_pass(&renderpass_create_info, None)
                    .expect("Failed to create render pass!")
            }
        };

        // # Create framebuffers
        let framebuffers: Vec<vk::Framebuffer> = {
            swapchain_imageviews
                .iter()
                .map(|&imageview| {
                    let attachments = [imageview, depth_image_view];

                    let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                        .render_pass(render_pass)
                        .attachments(&attachments)
                        .width(swapchain_extent.width)
                        .height(swapchain_extent.height)
                        .layers(1);

                    unsafe {
                        gpu.device
                            .create_framebuffer(&framebuffer_create_info, None)
                            .expect("Failed to create Framebuffer!")
                    }
                })
                .collect()
        };

        // # Create graphics pipeline
        let (graphics_pipeline, pipeline_layout) = {
            let vert_shader_module = create_shader_module(
                &gpu.device,
                include_bytes!("../shaders/spv/default.vert.spv").to_vec(),
            );
            let frag_shader_module = create_shader_module(
                &gpu.device,
                include_bytes!("../shaders/spv/default.frag.spv").to_vec(),
            );

            let main_function_name = CString::new("main").unwrap();

            let shader_stages = [
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: vert_shader_module,
                    p_name: main_function_name.as_ptr(),
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    module: frag_shader_module,
                    p_name: main_function_name.as_ptr(),
                    ..Default::default()
                },
            ];

            // (pos: vec3 + color: vec3) = 6 floats * 4 bytes per float
            const VERTEX_STRIDE: u32 = 24;
            let binding_descriptions = [vk::VertexInputBindingDescription {
                binding: 0,
                stride: VERTEX_STRIDE,
                ..Default::default()
            }];
            let attribute_descriptions = [
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 12,
                },
            ];
            let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
                vertex_binding_description_count: binding_descriptions.len() as u32,
                p_vertex_binding_descriptions: binding_descriptions.as_ptr(),
                vertex_attribute_description_count: attribute_descriptions.len() as u32,
                p_vertex_attribute_descriptions: attribute_descriptions.as_ptr(),
                ..Default::default()
            };

            let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            };

            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: swapchain_extent.width as f32,
                height: swapchain_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];

            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain_extent,
            }];

            let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
                scissor_count: scissors.len() as u32,
                p_scissors: scissors.as_ptr(),
                viewport_count: viewports.len() as u32,
                p_viewports: viewports.as_ptr(),
                ..Default::default()
            };

            let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::BACK,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                ..Default::default()
            };

            let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };

            let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: vk::TRUE,
                depth_write_enable: vk::TRUE,
                depth_compare_op: vk::CompareOp::LESS,
                max_depth_bounds: 1.0,
                min_depth_bounds: 0.0,
                ..Default::default()
            };

            let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                color_write_mask: vk::ColorComponentFlags::all(),
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ZERO,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
            }];

            let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
                attachment_count: color_blend_attachment_states.len() as u32,
                p_attachments: color_blend_attachment_states.as_ptr(),
                blend_constants: [0.0, 0.0, 0.0, 0.0],
                ..Default::default()
            };

            let set_layouts = [uniform_buffer_layout];
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);

            let pipeline_layout = unsafe {
                gpu.device
                    .create_pipeline_layout(&pipeline_layout_create_info, None)
                    .expect("Failed to create pipeline layout.")
            };

            let graphic_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo {
                stage_count: shader_stages.len() as u32,
                p_stages: shader_stages.as_ptr(),
                p_vertex_input_state: &vertex_input_state_create_info,
                p_input_assembly_state: &vertex_input_assembly_state_info,
                p_tessellation_state: ptr::null(),
                p_viewport_state: &viewport_state_create_info,
                p_rasterization_state: &rasterization_state_create_info,
                p_multisample_state: &multisample_state_create_info,
                p_depth_stencil_state: &depth_state_create_info,
                p_color_blend_state: &color_blend_state,
                p_dynamic_state: ptr::null(), // No dynamic state
                layout: pipeline_layout,
                render_pass,
                subpass: 0,
                ..Default::default()
            }];

            let graphics_pipelines = unsafe {
                gpu.device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &graphic_pipeline_create_infos,
                        None,
                    )
                    .expect("Failed to create Graphics Pipeline.")
            };

            unsafe {
                gpu.device.destroy_shader_module(vert_shader_module, None);
                gpu.device.destroy_shader_module(frag_shader_module, None);
            }

            (graphics_pipelines[0], pipeline_layout)
        };

        // # Allocate command buffers
        let command_buffers = {
            let info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(NUM_FRAMES as u32);

            unsafe {
                gpu.device
                    .allocate_command_buffers(&info)
                    .expect("Failed to allocate command buffer.")
            }
        };

        // # Record command buffers
        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

            unsafe {
                gpu.device
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Failed to begin recording command buffer.");
            }

            let clear_values = [
                vk::ClearValue {
                    // Clear value for color buffer
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    // Clear value for depth buffer
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(render_pass)
                .framebuffer(framebuffers[i])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain_extent,
                })
                .clear_values(&clear_values);

            unsafe {
                gpu.device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                gpu.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphics_pipeline,
                );

                // Bind index and vertex buffers
                {
                    let vertex_buffers = [vertex_buffer];
                    let offsets = [0_u64];
                    gpu.device.cmd_bind_vertex_buffers(
                        command_buffer,
                        0,
                        &vertex_buffers,
                        &offsets,
                    );
                    gpu.device.cmd_bind_index_buffer(
                        command_buffer,
                        index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                }

                // Bind descriptor sets
                {
                    let sets = [descriptor_sets[i]];
                    gpu.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        &sets,
                        &[],
                    );
                }

                gpu.device
                    .cmd_draw_indexed(command_buffer, num_indices, 1, 0, 0, 0);

                gpu.device.cmd_end_render_pass(command_buffer);

                gpu.device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to end recording command buffer.");
            }
        }

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

            for _ in 0..NUM_FRAMES {
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

        Apparatus {
            _surface_caps: surface_caps,
            _surface_formats: surface_formats,
            swapchain,
            _swapchain_format: swapchain_format,
            swapchain_extent,
            _swapchain_images: swapchain_images,
            swapchain_imageviews,
            depth_image,
            depth_image_view,
            depth_image_memory,
            render_pass,
            framebuffers,
            graphics_pipeline,
            pipeline_layout,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            command_buffer_complete_fences,
        }
    }

    pub fn destroy(&self, gpu: &Gpu, ext_swapchain: &ash::extensions::khr::Swapchain) {
        unsafe {
            for i in 0..NUM_FRAMES {
                gpu.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                gpu.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                gpu.device
                    .destroy_fence(self.command_buffer_complete_fences[i], None);
            }

            gpu.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            gpu.device.destroy_pipeline(self.graphics_pipeline, None);
            for &framebuffer in self.framebuffers.iter() {
                gpu.device.destroy_framebuffer(framebuffer, None);
            }
            gpu.device.destroy_render_pass(self.render_pass, None);
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

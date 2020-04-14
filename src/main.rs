mod utility;
use crate::{utility::debug::*, utility::*};

use ash::version::DeviceV1_0;
use ash::version::EntryV1_0;
use ash::version::InstanceV1_0;
use ash::vk;
use ash::vk_make_version;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

// Constants
const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct Gpu {
    // Physical device
    _physical_device: vk::PhysicalDevice,
    _exts: Vec<vk::ExtensionProperties>,
    surface_caps: vk::SurfaceCapabilitiesKHR,
    surface_formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
    _memory_properties: vk::PhysicalDeviceMemoryProperties,
    _properties: vk::PhysicalDeviceProperties,
    graphics_queue_idx: u32,
    present_queue_idx: u32,
    // Logical device
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    command_pool: vk::CommandPool,
}

struct SyncObjects {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    inflight_fences: Vec<vk::Fence>,
}

struct Surface {
    handle: vk::SurfaceKHR,
}

struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
}

struct VulkanApp {
    window: winit::window::Window,

    // Vulkan
    _entry: ash::Entry,
    instance: ash::Instance,
    // - Extensions
    ext_debug_utils: ash::extensions::ext::DebugUtils,
    ext_surface: ash::extensions::khr::Surface,
    ext_swapchain: ash::extensions::khr::Swapchain,

    gpu: Gpu,

    surface: Surface,

    debug_messenger: vk::DebugUtilsMessengerEXT,
    validation_layers: Vec<String>,

    swapchain: Swapchain,
    swapchain_imageviews: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,

    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
}

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
            .expect("Failed to create Shader Module!")
    }
}

impl VulkanApp {
    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> VulkanApp {
        const APP_NAME: &str = "Hello Triangle";
        const ENABLE_DEBUG_MESSENGER_CALLBACK: bool = true;
        let validation_layers = vec![String::from("VK_LAYER_KHRONOS_validation")];
        let device_extensions = vec![String::from("VK_KHR_swapchain")];

        // # Init window
        let window = {
            winit::window::WindowBuilder::new()
                .with_title(APP_NAME)
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
                .build(event_loop)
                .expect("Failed to create window.")
        };

        // # Init Ash
        let entry = ash::Entry::new().unwrap();

        // # Create Vulkan instance
        let instance = {
            let app_name = CString::new(APP_NAME).unwrap();
            let engine_name = CString::new("grapheme").unwrap();
            let app_info = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(vk_make_version!(1, 0, 0))
                .engine_name(&engine_name)
                .engine_version(vk_make_version!(1, 0, 0))
                .api_version(vk_make_version!(1, 0, 92))
                .build();

            // Ensure that all desired validation layers are available
            if !validation_layers.is_empty() {
                // Enumerate available validation layers
                let layer_props = entry
                    .enumerate_instance_layer_properties()
                    .expect("Failed to enumerate instance layers properties.");
                // Iterate over all desired layers
                for layer in validation_layers.iter() {
                    let is_layer_found = layer_props
                        .iter()
                        .any(|&prop| tools::vk_to_string(&prop.layer_name) == *layer);
                    if !is_layer_found {
                        panic!(
                            "Validation layer '{}' requested, but not found. \
                               (1) Install the Vulkan SDK and set up validation layers, \
                               or (2) remove any validation layers in the Rust code.",
                            layer
                        );
                    }
                }
            }

            let required_validation_layer_raw_names: Vec<CString> = validation_layers
                .iter()
                .map(|layer_name| CString::new(layer_name.to_string()).unwrap())
                .collect();
            let layer_names: Vec<*const c_char> = required_validation_layer_raw_names
                .iter()
                .map(|layer_name| layer_name.as_ptr())
                .collect();

            let extension_names = platforms::required_extension_names();

            let create_info = vk::InstanceCreateInfo::builder()
                .enabled_layer_names(&layer_names)
                .application_info(&app_info)
                .enabled_extension_names(&extension_names);

            let instance: ash::Instance = unsafe {
                entry
                    .create_instance(&create_info, None)
                    .expect("Failed to create instance.")
            };

            instance
        };

        // # Debug messenger callback
        let ext_debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);
        let debug_messenger = {
            if !ENABLE_DEBUG_MESSENGER_CALLBACK {
                ash::vk::DebugUtilsMessengerEXT::null()
            } else {
                let messenger_ci = populate_debug_messenger_create_info();
                unsafe {
                    ext_debug_utils
                        .create_debug_utils_messenger(&messenger_ci, None)
                        .expect("Debug Utils Callback")
                }
            }
        };

        // # Create surface
        let ext_surface = ash::extensions::khr::Surface::new(&entry, &instance);
        let surface = {
            let handle = unsafe {
                platforms::create_surface(&entry, &instance, &window)
                    .expect("Failed to create surface.")
            };
            Surface { handle }
        };

        // # Enumerate eligible GPUs
        struct CandidateGpu {
            physical_device: vk::PhysicalDevice,
            exts: Vec<vk::ExtensionProperties>,
            surface_caps: vk::SurfaceCapabilitiesKHR,
            surface_formats: Vec<vk::SurfaceFormatKHR>,
            present_modes: Vec<vk::PresentModeKHR>,
            memory_properties: vk::PhysicalDeviceMemoryProperties,
            properties: vk::PhysicalDeviceProperties,
            graphics_queue_idx: u32,
            present_queue_idx: u32,
        }
        let candidate_gpus: Vec<CandidateGpu> = {
            let physical_devices = unsafe {
                &instance
                    .enumerate_physical_devices()
                    .expect("Failed to enumerate Physical Devices!")
            };

            let mut candidate_gpus = Vec::new();

            for &physical_device in physical_devices {
                let exts = unsafe {
                    instance
                        .enumerate_device_extension_properties(physical_device)
                        .expect("Failed to get device extension properties.")
                };
                // Are desired extensions supported?
                let are_exts_supported = {
                    let available_exts: Vec<String> = exts
                        .iter()
                        .map(|&ext| tools::vk_to_string(&ext.extension_name))
                        .collect();

                    device_extensions.iter().all(|required_ext| {
                        available_exts
                            .iter()
                            .any(|available_ext| required_ext == available_ext)
                    })
                };
                if !are_exts_supported {
                    continue;
                }

                let surface_caps = unsafe {
                    ext_surface
                        .get_physical_device_surface_capabilities(physical_device, surface.handle)
                        .expect("Failed to query for surface capabilities.")
                };

                let surface_formats = unsafe {
                    ext_surface
                        .get_physical_device_surface_formats(physical_device, surface.handle)
                        .expect("Failed to query for surface formats.")
                };
                let present_modes = unsafe {
                    ext_surface
                        .get_physical_device_surface_present_modes(physical_device, surface.handle)
                        .expect("Failed to query for surface present mode.")
                };
                // Are there any surface formats and present modes?
                if surface_formats.is_empty() || present_modes.is_empty() {
                    continue;
                }

                let memory_properties =
                    unsafe { instance.get_physical_device_memory_properties(physical_device) };
                let properties =
                    unsafe { instance.get_physical_device_properties(physical_device) };

                // Queue family indices
                let queue_families = unsafe {
                    instance.get_physical_device_queue_family_properties(physical_device)
                };
                let opt_graphics_queue_idx = queue_families.iter().position(|&fam| {
                    fam.queue_count > 0 && fam.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                });
                let opt_present_queue_idx =
                    queue_families.iter().enumerate().position(|(i, &fam)| {
                        let is_present_supported = unsafe {
                            ext_surface.get_physical_device_surface_support(
                                physical_device,
                                i as u32,
                                surface.handle,
                            )
                        };
                        fam.queue_count > 0 && is_present_supported
                    });
                // Is there a graphics queue and a present queue?
                if opt_graphics_queue_idx.is_none() || opt_present_queue_idx.is_none() {
                    continue;
                }

                if let Some(graphics_queue_idx) = opt_graphics_queue_idx {
                    if let Some(present_queue_idx) = opt_present_queue_idx {
                        candidate_gpus.push(CandidateGpu {
                            physical_device,
                            exts,
                            surface_caps,
                            surface_formats,
                            present_modes,
                            memory_properties,
                            properties,
                            graphics_queue_idx: graphics_queue_idx as u32,
                            present_queue_idx: present_queue_idx as u32,
                        });
                    }
                }
            }

            candidate_gpus
        };

        // # Create a logical device, queues, the command pool, and the final gpu struct
        let gpu = {
            // Pick the most eligible of the candidate GPU.
            // Currently, we just pick the first one. Winner winner chicken dinner!
            // TODO: Might want to pick the most powerful GPU in the future.
            let cgpu = candidate_gpus
                .first()
                .expect("Failed to find a suitable GPU.");

            use std::collections::HashSet;
            let mut unique_queue_families = HashSet::new();
            unique_queue_families.insert(cgpu.graphics_queue_idx);
            unique_queue_families.insert(cgpu.present_queue_idx);

            let queue_priorities = [1.0_f32];
            let mut queue_create_infos = vec![];
            for &queue_family in unique_queue_families.iter() {
                let queue_create_info = vk::DeviceQueueCreateInfo {
                    s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                    p_next: ptr::null(),
                    flags: vk::DeviceQueueCreateFlags::empty(),
                    queue_family_index: queue_family,
                    p_queue_priorities: queue_priorities.as_ptr(),
                    queue_count: queue_priorities.len() as u32,
                };
                queue_create_infos.push(queue_create_info);
            }

            let physical_device_features = vk::PhysicalDeviceFeatures {
                sampler_anisotropy: vk::TRUE, // enable anisotropy device feature from Chapter-24.
                ..Default::default()
            };

            let raw_ext_names: Vec<CString> = device_extensions
                .iter()
                .map(|ext| CString::new(ext.to_string()).unwrap())
                .collect();
            let ext_names: Vec<*const c_char> =
                raw_ext_names.iter().map(|ext| ext.as_ptr()).collect();

            let device_create_info = vk::DeviceCreateInfo {
                s_type: vk::StructureType::DEVICE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::DeviceCreateFlags::empty(),
                queue_create_info_count: queue_create_infos.len() as u32,
                p_queue_create_infos: queue_create_infos.as_ptr(),
                enabled_layer_count: 0,
                pp_enabled_layer_names: ptr::null(),
                enabled_extension_count: device_extensions.len() as u32,
                pp_enabled_extension_names: ext_names.as_ptr(),
                p_enabled_features: &physical_device_features,
            };

            let device: ash::Device = unsafe {
                instance
                    .create_device(cgpu.physical_device, &device_create_info, None)
                    .expect("Failed to create logical Device!")
            };

            let graphics_queue = unsafe { device.get_device_queue(cgpu.graphics_queue_idx, 0) };
            let present_queue = unsafe { device.get_device_queue(cgpu.present_queue_idx, 0) };

            let command_pool = {
                let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(cgpu.graphics_queue_idx);

                unsafe {
                    device
                        .create_command_pool(&command_pool_create_info, None)
                        .expect("Failed to create Command Pool!")
                }
            };

            Gpu {
                // Physical device
                _physical_device: cgpu.physical_device,
                _exts: cgpu.exts.clone(),
                surface_caps: cgpu.surface_caps,
                surface_formats: cgpu.surface_formats.clone(),
                present_modes: cgpu.present_modes.clone(),
                _memory_properties: cgpu.memory_properties,
                _properties: cgpu.properties,
                graphics_queue_idx: cgpu.graphics_queue_idx,
                present_queue_idx: cgpu.present_queue_idx,
                // Logical device
                device,
                graphics_queue,
                present_queue,
                command_pool,
            }
        };

        // 9. Create swapchain
        let ext_swapchain = ash::extensions::khr::Swapchain::new(&instance, &gpu.device);
        let swapchain = {
            let surface_format: vk::SurfaceFormatKHR = {
                *gpu.surface_formats
                    .iter()
                    .find(|&f| {
                        f.format == vk::Format::B8G8R8A8_SRGB
                            && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                    })
                    .unwrap_or(&gpu.surface_formats[0])
            };
            let present_mode: vk::PresentModeKHR = {
                *gpu.present_modes
                    .iter()
                    .find(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
                    .unwrap_or(&vk::PresentModeKHR::FIFO)
            };
            let extent = {
                if gpu.surface_caps.current_extent.width != u32::max_value() {
                    gpu.surface_caps.current_extent
                } else {
                    let window_size = window.inner_size();
                    vk::Extent2D {
                        width: (window_size.width as u32)
                            .max(gpu.surface_caps.min_image_extent.width)
                            .min(gpu.surface_caps.max_image_extent.width),
                        height: (window_size.height as u32)
                            .max(gpu.surface_caps.min_image_extent.height)
                            .min(gpu.surface_caps.max_image_extent.height),
                    }
                }
            };

            let image_count = gpu.surface_caps.min_image_count + 1;
            let image_count = if gpu.surface_caps.max_image_count > 0 {
                image_count.min(gpu.surface_caps.max_image_count)
            } else {
                image_count
            };

            let (image_sharing_mode, queue_family_index_count, queue_family_indices) =
                if gpu.graphics_queue_idx != gpu.present_queue_idx {
                    (
                        vk::SharingMode::CONCURRENT,
                        2,
                        vec![gpu.graphics_queue_idx, gpu.present_queue_idx],
                    )
                } else {
                    (vk::SharingMode::EXCLUSIVE, 0, vec![])
                };

            let swapchain_create_info = vk::SwapchainCreateInfoKHR {
                s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
                p_next: ptr::null(),
                flags: vk::SwapchainCreateFlagsKHR::empty(),
                surface: surface.handle,
                min_image_count: image_count,
                image_color_space: surface_format.color_space,
                image_format: surface_format.format,
                image_extent: extent,
                image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                image_sharing_mode,
                p_queue_family_indices: queue_family_indices.as_ptr(),
                queue_family_index_count,
                pre_transform: gpu.surface_caps.current_transform,
                composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
                present_mode,
                clipped: vk::TRUE,
                old_swapchain: vk::SwapchainKHR::null(),
                image_array_layers: 1,
            };

            let handle = unsafe {
                ext_swapchain
                    .create_swapchain(&swapchain_create_info, None)
                    .expect("Failed to create Swapchain!")
            };

            let images = unsafe {
                ext_swapchain
                    .get_swapchain_images(handle)
                    .expect("Failed to get Swapchain Images.")
            };

            Swapchain {
                handle,
                format: surface_format.format,
                extent,
                images,
            }
        };
        // 10. Create image views
        let swapchain_imageviews = {
            let imageviews: Vec<vk::ImageView> = swapchain
                .images
                .iter()
                .map(|&image| {
                    let imageview_create_info = vk::ImageViewCreateInfo {
                        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                        p_next: ptr::null(),
                        flags: vk::ImageViewCreateFlags::empty(),
                        view_type: vk::ImageViewType::TYPE_2D,
                        format: swapchain.format,
                        components: vk::ComponentMapping {
                            r: vk::ComponentSwizzle::IDENTITY,
                            g: vk::ComponentSwizzle::IDENTITY,
                            b: vk::ComponentSwizzle::IDENTITY,
                            a: vk::ComponentSwizzle::IDENTITY,
                        },
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image,
                    };

                    unsafe {
                        gpu.device
                            .create_image_view(&imageview_create_info, None)
                            .expect("Failed to create Image View!")
                    }
                })
                .collect();

            imageviews
        };
        // 11. Create render pass
        let render_pass = {
            let color_attachment = vk::AttachmentDescription {
                format: swapchain.format,
                flags: vk::AttachmentDescriptionFlags::empty(),
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            };

            let color_attachment_ref = vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            };

            let subpasses = [vk::SubpassDescription {
                color_attachment_count: 1,
                p_color_attachments: &color_attachment_ref,
                p_depth_stencil_attachment: ptr::null(),
                flags: vk::SubpassDescriptionFlags::empty(),
                pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
                input_attachment_count: 0,
                p_input_attachments: ptr::null(),
                p_resolve_attachments: ptr::null(),
                preserve_attachment_count: 0,
                p_preserve_attachments: ptr::null(),
            }];

            let render_pass_attachments = [color_attachment];

            let subpass_dependencies = [vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dependency_flags: vk::DependencyFlags::empty(),
            }];

            let renderpass_create_info = vk::RenderPassCreateInfo {
                s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
                flags: vk::RenderPassCreateFlags::empty(),
                p_next: ptr::null(),
                attachment_count: render_pass_attachments.len() as u32,
                p_attachments: render_pass_attachments.as_ptr(),
                subpass_count: subpasses.len() as u32,
                p_subpasses: subpasses.as_ptr(),
                dependency_count: subpass_dependencies.len() as u32,
                p_dependencies: subpass_dependencies.as_ptr(),
            };

            unsafe {
                gpu.device
                    .create_render_pass(&renderpass_create_info, None)
                    .expect("Failed to create render pass!")
            }
        };
        // 12. Create graphics pipeline
        let (graphics_pipeline, pipeline_layout) = {
            let vert_shader_module = create_shader_module(
                &gpu.device,
                include_bytes!("../shaders/spv/09-shader-base.vert.spv").to_vec(),
            );
            let frag_shader_module = create_shader_module(
                &gpu.device,
                include_bytes!("../shaders/spv/09-shader-base.frag.spv").to_vec(),
            );

            let main_function_name = CString::new("main").unwrap();

            let shader_stages = [
                vk::PipelineShaderStageCreateInfo {
                    // Vertex Shader
                    s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                    p_next: ptr::null(),
                    flags: vk::PipelineShaderStageCreateFlags::empty(),
                    module: vert_shader_module,
                    p_name: main_function_name.as_ptr(),
                    p_specialization_info: ptr::null(),
                    stage: vk::ShaderStageFlags::VERTEX,
                },
                vk::PipelineShaderStageCreateInfo {
                    // Fragment Shader
                    s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                    p_next: ptr::null(),
                    flags: vk::PipelineShaderStageCreateFlags::empty(),
                    module: frag_shader_module,
                    p_name: main_function_name.as_ptr(),
                    p_specialization_info: ptr::null(),
                    stage: vk::ShaderStageFlags::FRAGMENT,
                },
            ];

            let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
                s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineVertexInputStateCreateFlags::empty(),
                vertex_attribute_description_count: 0,
                p_vertex_attribute_descriptions: ptr::null(),
                vertex_binding_description_count: 0,
                p_vertex_binding_descriptions: ptr::null(),
            };
            let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
                s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
                p_next: ptr::null(),
                primitive_restart_enable: vk::FALSE,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            };

            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: swapchain.extent.width as f32,
                height: swapchain.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];

            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.extent,
            }];

            let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
                s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineViewportStateCreateFlags::empty(),
                scissor_count: scissors.len() as u32,
                p_scissors: scissors.as_ptr(),
                viewport_count: viewports.len() as u32,
                p_viewports: viewports.as_ptr(),
            };

            let rasterization_statue_create_info = vk::PipelineRasterizationStateCreateInfo {
                s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineRasterizationStateCreateFlags::empty(),
                depth_clamp_enable: vk::FALSE,
                cull_mode: vk::CullModeFlags::BACK,
                front_face: vk::FrontFace::CLOCKWISE,
                line_width: 1.0,
                polygon_mode: vk::PolygonMode::FILL,
                rasterizer_discard_enable: vk::FALSE,
                depth_bias_clamp: 0.0,
                depth_bias_constant_factor: 0.0,
                depth_bias_enable: vk::FALSE,
                depth_bias_slope_factor: 0.0,
            };
            let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
                s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                flags: vk::PipelineMultisampleStateCreateFlags::empty(),
                p_next: ptr::null(),
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                sample_shading_enable: vk::FALSE,
                min_sample_shading: 0.0,
                p_sample_mask: ptr::null(),
                alpha_to_one_enable: vk::FALSE,
                alpha_to_coverage_enable: vk::FALSE,
            };

            let stencil_state = vk::StencilOpState {
                fail_op: vk::StencilOp::KEEP,
                pass_op: vk::StencilOp::KEEP,
                depth_fail_op: vk::StencilOp::KEEP,
                compare_op: vk::CompareOp::ALWAYS,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            };

            let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
                s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
                depth_test_enable: vk::FALSE,
                depth_write_enable: vk::FALSE,
                depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                depth_bounds_test_enable: vk::FALSE,
                stencil_test_enable: vk::FALSE,
                front: stencil_state,
                back: stencil_state,
                max_depth_bounds: 1.0,
                min_depth_bounds: 0.0,
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
                s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineColorBlendStateCreateFlags::empty(),
                logic_op_enable: vk::FALSE,
                logic_op: vk::LogicOp::COPY,
                attachment_count: color_blend_attachment_states.len() as u32,
                p_attachments: color_blend_attachment_states.as_ptr(),
                blend_constants: [0.0, 0.0, 0.0, 0.0],
            };

            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
                s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineLayoutCreateFlags::empty(),
                set_layout_count: 0,
                p_set_layouts: ptr::null(),
                push_constant_range_count: 0,
                p_push_constant_ranges: ptr::null(),
            };

            let pipeline_layout = unsafe {
                gpu.device
                    .create_pipeline_layout(&pipeline_layout_create_info, None)
                    .expect("Failed to create pipeline layout!")
            };

            let graphic_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo {
                s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineCreateFlags::empty(),
                stage_count: shader_stages.len() as u32,
                p_stages: shader_stages.as_ptr(),
                p_vertex_input_state: &vertex_input_state_create_info,
                p_input_assembly_state: &vertex_input_assembly_state_info,
                p_tessellation_state: ptr::null(),
                p_viewport_state: &viewport_state_create_info,
                p_rasterization_state: &rasterization_statue_create_info,
                p_multisample_state: &multisample_state_create_info,
                p_depth_stencil_state: &depth_state_create_info,
                p_color_blend_state: &color_blend_state,
                p_dynamic_state: ptr::null(),
                layout: pipeline_layout,
                render_pass,
                subpass: 0,
                base_pipeline_handle: vk::Pipeline::null(),
                base_pipeline_index: -1,
            }];

            let graphics_pipelines = unsafe {
                gpu.device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &graphic_pipeline_create_infos,
                        None,
                    )
                    .expect("Failed to create Graphics Pipeline!.")
            };

            unsafe {
                gpu.device.destroy_shader_module(vert_shader_module, None);
                gpu.device.destroy_shader_module(frag_shader_module, None);
            }

            (graphics_pipelines[0], pipeline_layout)
        };

        // 13. Create framebuffers
        let swapchain_framebuffers: Vec<vk::Framebuffer> = {
            swapchain_imageviews
                .iter()
                .map(|&imageview| {
                    let attachments = [imageview];

                    let framebuffer_create_info = vk::FramebufferCreateInfo {
                        s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
                        p_next: ptr::null(),
                        flags: vk::FramebufferCreateFlags::empty(),
                        render_pass,
                        attachment_count: attachments.len() as u32,
                        p_attachments: attachments.as_ptr(),
                        width: swapchain.extent.width,
                        height: swapchain.extent.height,
                        layers: 1,
                    };

                    unsafe {
                        gpu.device
                            .create_framebuffer(&framebuffer_create_info, None)
                            .expect("Failed to create Framebuffer!")
                    }
                })
                .collect()
        };

        // 14. Create command buffers
        let command_buffers = {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: ptr::null(),
                command_buffer_count: swapchain_framebuffers.len() as u32,
                command_pool: gpu.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
            };

            let command_buffers = unsafe {
                gpu.device
                    .allocate_command_buffers(&command_buffer_allocate_info)
                    .expect("Failed to allocate Command Buffers!")
            };

            for (i, &command_buffer) in command_buffers.iter().enumerate() {
                let command_buffer_begin_info = vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: ptr::null(),
                    p_inheritance_info: ptr::null(),
                    flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
                };

                unsafe {
                    gpu.device
                        .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                        .expect("Failed to begin recording Command Buffer at beginning!");
                }

                let clear_values = [vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }];

                let render_pass_begin_info = vk::RenderPassBeginInfo {
                    s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
                    p_next: ptr::null(),
                    render_pass,
                    framebuffer: swapchain_framebuffers[i],
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain.extent,
                    },
                    clear_value_count: clear_values.len() as u32,
                    p_clear_values: clear_values.as_ptr(),
                };

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
                    gpu.device.cmd_draw(command_buffer, 3, 1, 0, 0);

                    gpu.device.cmd_end_render_pass(command_buffer);

                    gpu.device
                        .end_command_buffer(command_buffer)
                        .expect("Failed to record Command Buffer at Ending!");
                }
            }

            command_buffers
        };

        // 15. Create sync objects
        let sync_ojbects = {
            let mut sync_objects = SyncObjects {
                image_available_semaphores: vec![],
                render_finished_semaphores: vec![],
                inflight_fences: vec![],
            };

            let semaphore_create_info = vk::SemaphoreCreateInfo {
                s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::SemaphoreCreateFlags::empty(),
            };

            let fence_create_info = vk::FenceCreateInfo {
                s_type: vk::StructureType::FENCE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::FenceCreateFlags::SIGNALED,
            };

            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                unsafe {
                    let image_available_semaphore = gpu
                        .device
                        .create_semaphore(&semaphore_create_info, None)
                        .expect("Failed to create Semaphore Object!");
                    let render_finished_semaphore = gpu
                        .device
                        .create_semaphore(&semaphore_create_info, None)
                        .expect("Failed to create Semaphore Object!");
                    let inflight_fence = gpu
                        .device
                        .create_fence(&fence_create_info, None)
                        .expect("Failed to create Fence Object!");

                    sync_objects
                        .image_available_semaphores
                        .push(image_available_semaphore);
                    sync_objects
                        .render_finished_semaphores
                        .push(render_finished_semaphore);
                    sync_objects.inflight_fences.push(inflight_fence);
                }
            }

            sync_objects
        };

        VulkanApp {
            window,
            // Vulkan
            _entry: entry,
            instance,
            // - Extensions
            ext_debug_utils,
            ext_surface,
            ext_swapchain,

            gpu,

            surface,

            debug_messenger,
            validation_layers,

            swapchain,
            swapchain_imageviews,
            swapchain_framebuffers,

            pipeline_layout,
            render_pass,
            graphics_pipeline,

            command_buffers,

            image_available_semaphores: sync_ojbects.image_available_semaphores,
            render_finished_semaphores: sync_ojbects.render_finished_semaphores,
            in_flight_fences: sync_ojbects.inflight_fences,
            current_frame: 0,
        }
    }

    fn draw_frame(&mut self) {
        let wait_fences = [self.in_flight_fences[self.current_frame]];

        let (image_index, _is_sub_optimal) = unsafe {
            self.gpu
                .device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence!");

            self.ext_swapchain
                .acquire_next_image(
                    self.swapchain.handle,
                    std::u64::MAX,
                    self.image_available_semaphores[self.current_frame],
                    vk::Fence::null(),
                )
                .expect("Failed to acquire next image.")
        };

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

        let submit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffers[image_index as usize],
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
        }];

        unsafe {
            self.gpu
                .device
                .reset_fences(&wait_fences)
                .expect("Failed to reset Fence!");

            self.gpu
                .device
                .queue_submit(
                    self.gpu.graphics_queue,
                    &submit_infos,
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to execute queue submit.");
        }

        let swapchains = [self.swapchain.handle];

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &image_index,
            p_results: ptr::null_mut(),
        };

        unsafe {
            self.ext_swapchain
                .queue_present(self.gpu.present_queue, &present_info)
                .expect("Failed to execute queue present.");
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            // TODO: Move these into their own sub-object drop traits.
            // e.g. Swapchains and surfaces can have their own drops.
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.gpu
                    .device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.gpu
                    .device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.gpu
                    .device
                    .destroy_fence(self.in_flight_fences[i], None);
            }

            self.gpu
                .device
                .destroy_command_pool(self.gpu.command_pool, None);

            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.gpu.device.destroy_framebuffer(framebuffer, None);
            }

            self.gpu
                .device
                .destroy_pipeline(self.graphics_pipeline, None);
            self.gpu
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.gpu.device.destroy_render_pass(self.render_pass, None);

            for &imageview in self.swapchain_imageviews.iter() {
                self.gpu.device.destroy_image_view(imageview, None);
            }

            self.ext_swapchain
                .destroy_swapchain(self.swapchain.handle, None);
            self.gpu.device.destroy_device(None);
            self.ext_surface.destroy_surface(self.surface.handle, None);

            if !self.validation_layers.is_empty() {
                self.ext_debug_utils
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

impl VulkanApp {
    pub fn main_loop(mut self, event_loop: EventLoop<()>) {
        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => match input {
                    KeyboardInput {
                        virtual_keycode,
                        state,
                        ..
                    } => match (virtual_keycode, state) {
                        (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                            *control_flow = ControlFlow::Exit
                        }
                        (Some(VirtualKeyCode::Return), ElementState::Pressed) => {
                            *control_flow = ControlFlow::Exit
                        }
                        _ => {}
                    },
                },
                _ => {}
            },
            Event::MainEventsCleared => {
                self.window.request_redraw();
            }
            Event::RedrawRequested(_window_id) => {
                self.draw_frame();
            }
            Event::LoopDestroyed => {
                unsafe {
                    self.gpu
                        .device
                        .device_wait_idle()
                        .expect("Failed to wait device idle!")
                };
            }
            _ => (),
        })
    }
}

fn main() {
    let event_loop = EventLoop::new();

    let vulkan_app = VulkanApp::new(&event_loop);
    vulkan_app.main_loop(event_loop);
}

mod utility;
use crate::{utility::debug::*, utility::share, utility::*};

use ash::version::DeviceV1_0;
use ash::version::EntryV1_0;
use ash::version::InstanceV1_0;
use ash::vk;
use ash::vk_make_version;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;

// Constants
const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct SyncObjects {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    inflight_fences: Vec<vk::Fence>,
}

struct VulkanApp {
    window: winit::window::Window,
    // Vulkan
    _entry: ash::Entry,
    validation_layers: Vec<String>,
    instance: ash::Instance,
    surface_ext_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    debug_utils_ext_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,

    _physical_device: vk::PhysicalDevice,
    device: ash::Device,

    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    _swapchain_images: Vec<vk::Image>,
    _swapchain_format: vk::Format,
    _swapchain_extent: vk::Extent2D,
    swapchain_imageviews: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
}

impl VulkanApp {
    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> VulkanApp {
        const WINDOW_TITLE: &str = "Hello Triangle";
        const WINDOW_WIDTH: u32 = 800;
        const WINDOW_HEIGHT: u32 = 600;
        let validation_layers = vec![String::from("VK_LAYER_KHRONOS_validation")];
        const ENABLE_DEBUG_MESSENGER_CALLBACK: bool = true;
        const DEVICE_EXTENSIONS: structures::DeviceExtension = structures::DeviceExtension {
            names: ["VK_KHR_swapchain"],
        };

        // 1. Init window
        let window = {
            winit::window::WindowBuilder::new()
                .with_title(WINDOW_TITLE)
                .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
                .build(event_loop)
                .expect("Failed to create window.")
        };
        // 2. Init Ash
        let entry = ash::Entry::new().unwrap();
        // 3. Create Vulkan instance
        let instance = {
            // Ensure that all desired validation layers are available
            if validation_layers.len() > 0 {
                // Enumerate available validation layers
                let layer_props = entry
                    .enumerate_instance_layer_properties()
                    .expect("Failed to enumerate instance layers properties.");
                // Iterate over all desired layers
                for layer in validation_layers.iter() {
                    let is_layer_found = layer_props
                        .iter()
                        .any(|&prop| tools::vk_to_string(&prop.layer_name) == *layer);
                    if is_layer_found == false {
                        panic!(
                            "Validation layer '{}' requested, but not found. \
                               (1) Install the Vulkan SDK and set up validation layers, \
                               or (2) remove any validation layers in the Rust code.",
                            layer
                        );
                    }
                }
            }

            let app_name = CString::new(WINDOW_TITLE).unwrap();
            let app_info = vk::ApplicationInfo {
                p_application_name: app_name.as_ptr(),
                s_type: vk::StructureType::APPLICATION_INFO,
                p_next: ptr::null(),
                application_version: vk_make_version!(1, 0, 0),
                p_engine_name: CString::new("grapheme").unwrap().as_ptr(),
                engine_version: vk_make_version!(1, 0, 0),
                api_version: vk_make_version!(1, 0, 92),
            };

            // This create info used to debug issues in vk::createInstance and vk::destroyInstance.
            let debug_utils_create_info = debug::populate_debug_messenger_create_info();

            // VK_EXT debug report has been requested here.
            let extension_names = platforms::required_extension_names();

            let requred_validation_layer_raw_names: Vec<CString> = validation_layers
                .iter()
                .map(|layer_name| CString::new(layer_name.to_string()).unwrap())
                .collect();
            let layer_names: Vec<*const i8> = requred_validation_layer_raw_names
                .iter()
                .map(|layer_name| layer_name.as_ptr())
                .collect();

            let create_info = vk::InstanceCreateInfo {
                s_type: vk::StructureType::INSTANCE_CREATE_INFO,
                p_next: if validation_layers.len() > 0 {
                    &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
                        as *const c_void
                } else {
                    ptr::null()
                },
                flags: vk::InstanceCreateFlags::empty(),
                p_application_info: &app_info,
                pp_enabled_layer_names: if validation_layers.len() > 0 {
                    layer_names.as_ptr()
                } else {
                    ptr::null()
                },
                enabled_layer_count: if validation_layers.len() > 0 {
                    layer_names.len()
                } else {
                    0
                } as u32,
                pp_enabled_extension_names: extension_names.as_ptr(),
                enabled_extension_count: extension_names.len() as u32,
            };

            let instance: ash::Instance = unsafe {
                entry
                    .create_instance(&create_info, None)
                    .expect("Failed to create instance.")
            };

            instance
        };
        // 4. Create surface
        let surface = unsafe {
            platforms::create_surface(&entry, &instance, &window)
                .expect("Failed to create surface.")
        };
        let surface_ext_loader = ash::extensions::khr::Surface::new(&entry, &instance);
        // 5. Debug messenger callback
        let debug_utils_ext_loader = ash::extensions::ext::DebugUtils::new(&entry, &instance);
        let debug_messenger = {
            if ENABLE_DEBUG_MESSENGER_CALLBACK == false {
                ash::vk::DebugUtilsMessengerEXT::null()
            } else {
                let messenger_ci = populate_debug_messenger_create_info();
                let utils_messenger = unsafe {
                    debug_utils_ext_loader
                        .create_debug_utils_messenger(&messenger_ci, None)
                        .expect("Debug Utils Callback")
                };

                utils_messenger
            }
        };
        // 6. Pick physical device and queue family indices
        let (physical_device, graphics_queue_idx, present_queue_idx) = {
            let physical_devices = unsafe {
                &instance
                    .enumerate_physical_devices()
                    .expect("Failed to enumerate Physical Devices!")
            };
            // Pick the first compatible physical device
            struct Compat {
                device: vk::PhysicalDevice,
                graphics_queue_idx: u32,
                present_queue_idx: u32,
            }
            let mut opt_compat: Option<Compat> = None;
            for &device in physical_devices.iter() {
                let device_features = unsafe { instance.get_physical_device_features(device) };

                let queue_families =
                    unsafe { instance.get_physical_device_queue_family_properties(device) };

                let opt_graphics_queue_idx = queue_families.iter().position(|&fam| {
                    fam.queue_count > 0 && fam.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                });
                let opt_present_queue_idx =
                    queue_families.iter().enumerate().position(|(i, &fam)| {
                        let is_present_supported = unsafe {
                            surface_ext_loader
                                .get_physical_device_surface_support(device, i as u32, surface)
                        };
                        fam.queue_count > 0 && is_present_supported
                    });

                let is_device_extension_supported = {
                    // Query available extensions
                    let props = unsafe {
                        instance
                            .enumerate_device_extension_properties(device)
                            .expect("Failed to get device extension properties.")
                    };
                    let available_exts: Vec<String> = props
                        .iter()
                        .map(|&ext| tools::vk_to_string(&ext.extension_name))
                        .collect();

                    DEVICE_EXTENSIONS.names.iter().all(|required_ext| {
                        available_exts
                            .iter()
                            .any(|available_ext| required_ext == available_ext)
                    })
                };

                let is_swapchain_supported = if is_device_extension_supported {
                    let swapchain_support =
                        share::query_swapchain_support(device, surface, &surface_ext_loader);
                    !swapchain_support.formats.is_empty()
                        && !swapchain_support.present_modes.is_empty()
                } else {
                    false
                };
                let is_support_sampler_anisotropy = device_features.sampler_anisotropy == 1;

                if opt_graphics_queue_idx.is_some()
                    && opt_present_queue_idx.is_some()
                    && is_device_extension_supported
                    && is_swapchain_supported
                    && is_support_sampler_anisotropy
                {
                    opt_compat = Some(Compat {
                        device,
                        graphics_queue_idx: opt_graphics_queue_idx.unwrap() as u32,
                        present_queue_idx: opt_present_queue_idx.unwrap() as u32,
                    });
                    break;
                }
            }

            match opt_compat {
                Some(compat) => (
                    compat.device,
                    compat.graphics_queue_idx,
                    compat.present_queue_idx,
                ),
                None => panic!("Failed to find a suitable GPU!"),
            }
        };
        // 7. Create logical device
        let device = {
            use std::collections::HashSet;
            let mut unique_queue_families = HashSet::new();
            unique_queue_families.insert(graphics_queue_idx);
            unique_queue_families.insert(present_queue_idx);

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

            let enable_extension_names = DEVICE_EXTENSIONS.get_extensions_raw_names();

            let device_create_info = vk::DeviceCreateInfo {
                s_type: vk::StructureType::DEVICE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::DeviceCreateFlags::empty(),
                queue_create_info_count: queue_create_infos.len() as u32,
                p_queue_create_infos: queue_create_infos.as_ptr(),
                enabled_layer_count: 0,
                pp_enabled_layer_names: ptr::null(),
                enabled_extension_count: enable_extension_names.len() as u32,
                pp_enabled_extension_names: enable_extension_names.as_ptr(),
                p_enabled_features: &physical_device_features,
            };

            let device: ash::Device = unsafe {
                instance
                    .create_device(physical_device, &device_create_info, None)
                    .expect("Failed to create logical Device!")
            };

            device
        };
        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_idx, 0) };
        let present_queue = unsafe { device.get_device_queue(present_queue_idx, 0) };
        let swapchain_stuff = share::create_swapchain(
            &instance,
            &device,
            physical_device,
            &window,
            surface,
            &surface_ext_loader,
            graphics_queue_idx,
            present_queue_idx,
        );
        let swapchain_imageviews = share::v1::create_image_views(
            &device,
            swapchain_stuff.swapchain_format,
            &swapchain_stuff.swapchain_images,
        );
        let render_pass = VulkanApp::create_render_pass(&device, swapchain_stuff.swapchain_format);
        let (graphics_pipeline, pipeline_layout) = share::v1::create_graphics_pipeline(
            &device,
            render_pass,
            swapchain_stuff.swapchain_extent,
        );
        let swapchain_framebuffers = share::v1::create_framebuffers(
            &device,
            render_pass,
            &swapchain_imageviews,
            swapchain_stuff.swapchain_extent,
        );
        let command_pool = share::v1::create_command_pool(&device, graphics_queue_idx);
        let command_buffers = share::v1::create_command_buffers(
            &device,
            command_pool,
            graphics_pipeline,
            &swapchain_framebuffers,
            render_pass,
            swapchain_stuff.swapchain_extent,
        );
        let sync_ojbects = VulkanApp::create_sync_objects(&device);

        // cleanup(); the 'drop' function will take care of it.
        VulkanApp {
            window,
            // vulkan stuff
            _entry: entry,
            validation_layers: validation_layers,
            instance,
            surface: surface,
            surface_ext_loader,
            debug_utils_ext_loader,
            debug_messenger,

            _physical_device: physical_device,
            device,

            graphics_queue,
            present_queue,

            swapchain_loader: swapchain_stuff.swapchain_loader,
            swapchain: swapchain_stuff.swapchain,
            _swapchain_format: swapchain_stuff.swapchain_format,
            _swapchain_images: swapchain_stuff.swapchain_images,
            _swapchain_extent: swapchain_stuff.swapchain_extent,
            swapchain_imageviews,
            swapchain_framebuffers,

            pipeline_layout,
            render_pass,
            graphics_pipeline,

            command_pool,
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
            self.device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence!");

            self.swapchain_loader
                .acquire_next_image(
                    self.swapchain,
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
            self.device
                .reset_fences(&wait_fences)
                .expect("Failed to reset Fence!");

            self.device
                .queue_submit(
                    self.graphics_queue,
                    &submit_infos,
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to execute queue submit.");
        }

        let swapchains = [self.swapchain];

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
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
                .expect("Failed to execute queue present.");
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn create_render_pass(device: &ash::Device, surface_format: vk::Format) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription {
            format: surface_format,
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
            device
                .create_render_pass(&renderpass_create_info, None)
                .expect("Failed to create render pass!")
        }
    }

    fn create_sync_objects(device: &ash::Device) -> SyncObjects {
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
                let image_available_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");
                let render_finished_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");
                let inflight_fence = device
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
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }

            self.device.destroy_command_pool(self.command_pool, None);

            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }

            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            for &imageview in self.swapchain_imageviews.iter() {
                self.device.destroy_image_view(imageview, None);
            }

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_ext_loader.destroy_surface(self.surface, None);

            if self.validation_layers.len() > 0 {
                self.debug_utils_ext_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

// Fix content -------------------------------------------------------------------------------
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
                    self.device
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
// -------------------------------------------------------------------------------------------

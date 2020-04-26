use crate::*;

use ash::version::DeviceV1_0;
use ash::version::EntryV1_0;
use ash::version::InstanceV1_0;
use ash::vk;
use ash::vk_make_version;
use glam::*;
use std::ffi::CString;
use std::os::raw::c_char;
use std::os::raw::c_void;
use std::ptr;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

fn vk_to_string(raw_string_array: &[c_char]) -> String {
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}

pub struct Gpu {
    // Physical device
    pub physical_device: vk::PhysicalDevice,
    pub _exts: Vec<vk::ExtensionProperties>,
    pub present_modes: Vec<vk::PresentModeKHR>,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub _properties: vk::PhysicalDeviceProperties,
    pub graphics_queue_idx: u32,
    pub present_queue_idx: u32,
    // Logical device
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
}

pub struct Context {
    window: winit::window::Window,
    event_loop: Option<winit::event_loop::EventLoop<()>>,

    _entry: ash::Entry,
    instance: ash::Instance,
    surface: vk::SurfaceKHR,
    // - Extensions
    ext_debug_utils: ash::extensions::ext::DebugUtils,
    ext_surface: ash::extensions::khr::Surface,

    pub gpu: Gpu,
    pub facade: Facade, // Resolution-dependent apparatus
    pub apparatus: Apparatus,
    pub command_pool: vk::CommandPool,
    pub vertex_buffer: DeviceLocalBuffer,
    pub index_buffer: DeviceLocalBuffer,
    pub num_indices: u32,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub uniform_buffer_layout: vk::DescriptorSetLayout,
    pub uniform_buffers: Vec<HostVisibleBuffer>,
    pub environment_texture: Texture,

    debug_messenger: vk::DebugUtilsMessengerEXT,
    validation_layers: Vec<String>,
    pub current_frame: usize,
    start_instant: std::time::Instant,

    _watcher: notify::RecommendedWatcher, // Need to keep this alive to keep the receiver alive
    watch_rx: std::sync::mpsc::Receiver<notify::DebouncedEvent>,
}

impl Context {
    pub fn recreate_resolution_dependent_state(&mut self) {
        unsafe {
            self.gpu
                .device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
        self.apparatus.destroy(&self.gpu);
        self.facade.destroy(&self.gpu);
        self.facade = Facade::new(
            &self.instance,
            &self.window,
            self.surface,
            &self.gpu,
            &self.ext_surface,
        );
        let (shader_modules, _) =
            utils::get_shader_modules(&self.gpu).expect("Failed to load shader modules");
        self.apparatus = Apparatus::new(
            &self.gpu,
            &self.facade,
            self.command_pool,
            &self.vertex_buffer,
            &self.index_buffer,
            self.num_indices,
            self.uniform_buffer_layout,
            &self.descriptor_sets,
            shader_modules,
        );
    }

    pub fn new(uniform_buffer_size: usize) -> Context {
        const APP_NAME: &str = "";
        const ENABLE_DEBUG_MESSENGER_CALLBACK: bool = true;
        let validation_layers = vec![String::from("VK_LAYER_KHRONOS_validation")];
        let device_extensions = vec![String::from("VK_KHR_swapchain")];

        // # Init window
        let event_loop = EventLoop::new();
        let window = {
            winit::window::WindowBuilder::new()
                .with_title(APP_NAME)
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
                .build(&event_loop)
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
                .api_version(vk_make_version!(1, 0, 92));

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
                        .any(|&prop| vk_to_string(&prop.layer_name) == *layer);
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
        let surface = unsafe {
            platforms::create_surface(&entry, &instance, &window)
                .expect("Failed to create surface.")
        };

        // # Enumerate eligible GPUs
        struct CandidateGpu {
            physical_device: vk::PhysicalDevice,
            exts: Vec<vk::ExtensionProperties>,
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
                        .map(|&ext| vk_to_string(&ext.extension_name))
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

                let surface_formats = unsafe {
                    ext_surface
                        .get_physical_device_surface_formats(physical_device, surface)
                        .expect("Failed to query for surface formats.")
                };
                let present_modes = unsafe {
                    ext_surface
                        .get_physical_device_surface_present_modes(physical_device, surface)
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
                                surface,
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

        // # Create a logical device, queues, the command pool, sync primitives, and the final gpu struct
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

            Gpu {
                physical_device: cgpu.physical_device,
                _exts: cgpu.exts.clone(),
                present_modes: cgpu.present_modes.clone(),
                memory_properties: cgpu.memory_properties,
                _properties: cgpu.properties,
                graphics_queue_idx: cgpu.graphics_queue_idx,
                present_queue_idx: cgpu.present_queue_idx,
                device,
                graphics_queue,
                present_queue,
            }
        };

        // # Create command pool
        let command_pool = {
            let info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(gpu.graphics_queue_idx);

            unsafe {
                gpu.device
                    .create_command_pool(&info, None)
                    .expect("Failed to create command pool")
            }
        };

        // # Load mesh
        // TODO: Benchmark and optimize
        let (vertices_data, indices_data) = {
            let mut vertices_data: Vec<f32> = Vec::new();
            let mut indices_data: Vec<u32> = Vec::new();

            let (gltf, buffers, _) =
                gltf::import("assets/suzanne.glb").expect("Failed to open mesh.");
            for mesh in gltf.meshes() {
                for primitive in mesh.primitives() {
                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                    if let Some(iter_pos) = reader.read_positions() {
                        if let Some(iter_norm) = reader.read_normals() {
                            for (pos, norm) in iter_pos.zip(iter_norm) {
                                vertices_data.extend_from_slice(&pos);
                                vertices_data.extend_from_slice(&norm);
                            }
                        }
                    }
                    if let Some(iter) = reader.read_indices() {
                        match iter {
                            gltf::mesh::util::ReadIndices::U8(iter_2) => {
                                for idx in iter_2 {
                                    indices_data.push(idx as u32);
                                }
                            }
                            gltf::mesh::util::ReadIndices::U16(iter_2) => {
                                for idx in iter_2 {
                                    indices_data.push(idx as u32);
                                }
                            }
                            gltf::mesh::util::ReadIndices::U32(iter_2) => {
                                for idx in iter_2 {
                                    indices_data.push(idx as u32);
                                }
                            }
                        }
                    }
                }
            }

            (vertices_data, indices_data)
        };

        // # Create and upload the vertex buffer
        let vertex_buffer = DeviceLocalBuffer::new(
            &vertices_data,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &gpu,
            command_pool,
        );

        // # Create and upload index buffer
        let index_buffer = DeviceLocalBuffer::new(
            &indices_data,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &gpu,
            command_pool,
        );

        // TODO: Move this up?
        let facade = Facade::new(&instance, &window, surface, &gpu, &ext_surface);

        // # Uniform buffer descriptor layout
        let uniform_buffer_layout = {
            let bindings = [vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX,
                p_immutable_samplers: ptr::null(),
            }];

            let ubo_layout_create_info =
                vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

            unsafe {
                gpu.device
                    .create_descriptor_set_layout(&ubo_layout_create_info, None)
                    .expect("Failed to create Descriptor Set Layout!")
            }
        };

        // # Create the uniform buffer
        let uniform_buffers: Vec<HostVisibleBuffer> = (0..facade.num_frames)
            .map(|_| {
                HostVisibleBuffer::new(
                    uniform_buffer_size as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    &gpu,
                )
            })
            .collect();

        // # Create descriptor pool
        let descriptor_pool = {
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: facade.num_frames as u32,
            }];

            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(facade.num_frames as u32)
                .pool_sizes(&pool_sizes);

            unsafe {
                gpu.device
                    .create_descriptor_pool(&descriptor_pool_create_info, None)
                    .expect("Failed to create descriptor pool.")
            }
        };

        // # Create descriptor sets
        let descriptor_sets = {
            let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
            for _ in 0..facade.num_frames {
                layouts.push(uniform_buffer_layout);
            }

            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts);

            let descriptor_sets = unsafe {
                gpu.device
                    .allocate_descriptor_sets(&descriptor_set_allocate_info)
                    .expect("Failed to allocate descriptor sets.")
            };

            for (i, &descritptor_set) in descriptor_sets.iter().enumerate() {
                let descriptor_buffer_info = [vk::DescriptorBufferInfo {
                    buffer: uniform_buffers[i].vk_buffer,
                    offset: 0,
                    range: uniform_buffer_size as u64,
                }];

                let descriptor_write_sets = [vk::WriteDescriptorSet {
                    dst_set: descritptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    p_buffer_info: descriptor_buffer_info.as_ptr(),
                    ..Default::default()
                }];

                unsafe {
                    gpu.device
                        .update_descriptor_sets(&descriptor_write_sets, &[]);
                }
            }

            descriptor_sets
        };

        let (shader_modules, _) =
            utils::get_shader_modules(&gpu).expect("Failed to load shader modules");
        let apparatus = Apparatus::new(
            &gpu,
            &facade,
            command_pool,
            &vertex_buffer,
            &index_buffer,
            indices_data.len() as u32,
            uniform_buffer_layout,
            &descriptor_sets,
            shader_modules,
        );

        // Add expect messages to all these unwraps
        let (watcher, watch_rx) = {
            use notify::{RecommendedWatcher, RecursiveMode, Watcher};
            use std::sync::mpsc::channel;
            use std::time::Duration;

            let (tx, rx) = channel();
            let mut watcher: RecommendedWatcher = Watcher::new(tx, Duration::from_secs(2)).unwrap();
            watcher.watch("./assets", RecursiveMode::Recursive).unwrap();
            (watcher, rx)
        };

        let environment_texture = Texture::new_from_image(
            std::path::Path::new("assets/textures/env_carpentry_shop_02_2k.hdr"),
            &gpu,
            command_pool,
        );

        Context {
            window,
            event_loop: Some(event_loop),

            _entry: entry,
            instance,
            surface,
            // - Extensions
            ext_debug_utils,
            ext_surface,
            // - Device
            gpu,
            facade,
            apparatus,
            command_pool,
            vertex_buffer,
            index_buffer,
            num_indices: indices_data.len() as u32,
            descriptor_pool,
            descriptor_sets,
            uniform_buffer_layout,
            uniform_buffers,
            environment_texture,

            debug_messenger,
            validation_layers,

            current_frame: 0,
            start_instant: std::time::Instant::now(),

            _watcher: watcher,
            watch_rx,
        }
    }

    fn draw_frame<F>(&mut self, on_draw: &mut F)
    where
        F: FnMut(&mut Context, f32, usize),
    {
        let wait_fences = [self.facade.command_buffer_complete_fences[self.current_frame]];

        let (image_index, _is_sub_optimal) = unsafe {
            self.gpu
                .device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence.");

            let result = self.facade.ext_swapchain.acquire_next_image(
                self.facade.swapchain,
                std::u64::MAX,
                self.facade.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            );
            match result {
                Ok(image_idx) => image_idx,
                Err(error_code) => {
                    match error_code {
                        vk::Result::ERROR_OUT_OF_DATE_KHR => {
                            // Window is resized. Recreate the swapchain
                            // and exit early without drawing this frame.
                            self.recreate_resolution_dependent_state();
                            return;
                        }
                        _ => panic!("Failed to acquire swapchain image."),
                    }
                }
            }
        };

        let elapsed_seconds = self.start_instant.elapsed().as_secs_f32();
        on_draw(self, elapsed_seconds, image_index as usize);

        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let wait_semaphores = [self.facade.image_available_semaphores[self.current_frame]];
        let signal_semaphores = [self.facade.render_finished_semaphores[self.current_frame]];
        let command_buffers = [self.apparatus.command_buffers[image_index as usize]];

        let submit_infos = [vk::SubmitInfo {
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: command_buffers.len() as u32,
            p_command_buffers: command_buffers.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        }];

        unsafe {
            self.gpu
                .device
                .reset_fences(&wait_fences)
                .expect("Failed to reset fence.");

            self.gpu
                .device
                .queue_submit(
                    self.gpu.graphics_queue,
                    &submit_infos,
                    self.facade.command_buffer_complete_fences[self.current_frame],
                )
                .expect("Failed to execute queue submit.");
        }

        let swapchains = [self.facade.swapchain];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        // Present the queue
        {
            let result = unsafe {
                self.facade
                    .ext_swapchain
                    .queue_present(self.gpu.present_queue, &present_info)
            };

            if let Err(error_code) = result {
                match error_code {
                    vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                        // Window is resized. Recreate the swapchain
                        self.recreate_resolution_dependent_state();
                    }
                    _ => panic!("Failed to present queue."),
                }
            }
        }

        for event in self.watch_rx.try_iter() {
            use notify::DebouncedEvent::*;
            match event {
                Write(_) | Remove(_) | Rename(_, _) => {
                    unsafe {
                        self.gpu
                            .device
                            .device_wait_idle()
                            .expect("Failed to wait device idle!");
                    }
                    if let Some((shader_modules, num_changed)) =
                        utils::get_shader_modules(&self.gpu)
                    {
                        if num_changed > 0 {
                            self.apparatus.destroy(&self.gpu);
                            self.apparatus = Apparatus::new(
                                &self.gpu,
                                &self.facade,
                                self.command_pool,
                                &self.vertex_buffer,
                                &self.index_buffer,
                                self.num_indices,
                                self.uniform_buffer_layout,
                                &self.descriptor_sets,
                                shader_modules,
                            );
                        }
                    }
                }
                _ => (),
            }
        }

        self.current_frame = (self.current_frame + 1) % self.facade.num_frames;
    }

    pub fn run_loop<F: 'static>(mut self, mut on_draw: F)
    where
        F: FnMut(&mut Context, f32, usize),
    {
        self.event_loop
            .take()
            .unwrap()
            .run(move |event, _, control_flow| match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        if self.facade.swapchain_extent.width != physical_size.width
                            || self.facade.swapchain_extent.height != physical_size.height
                        {
                            self.recreate_resolution_dependent_state();
                        }
                    }
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
                    self.draw_frame(&mut on_draw);
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

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.gpu
                .device
                .destroy_command_pool(self.command_pool, None);

            self.facade.destroy(&self.gpu);
            self.apparatus.destroy(&self.gpu);

            // Uniform buffer
            self.gpu
                .device
                .destroy_descriptor_set_layout(self.uniform_buffer_layout, None);
            self.gpu
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            for buffer in self.uniform_buffers.iter() {
                buffer.destroy();
            }
            // Vertex buffer
            self.vertex_buffer.destroy();
            // Index buffer
            self.index_buffer.destroy();
            // Texture
            self.environment_texture.destroy();

            self.gpu.device.destroy_device(None);
            self.ext_surface.destroy_surface(self.surface, None);

            if !self.validation_layers.is_empty() {
                self.ext_debug_utils
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

// Debug callbacks
unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

pub fn populate_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
    vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
            // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
            // vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(vulkan_debug_utils_callback),
        p_user_data: ptr::null_mut(),
    }
}

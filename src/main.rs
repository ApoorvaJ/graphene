mod apparatus;
use apparatus::*;
mod utility;
use crate::{utility::debug::*, utility::*};

use ash::version::DeviceV1_0;
use ash::version::EntryV1_0;
use ash::version::InstanceV1_0;
use ash::vk;
use ash::vk_make_version;
use glam::*;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use std::f32::consts::PI;
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

// Constants
const NUM_FRAMES: usize = 2;
const DEGREES_TO_RADIANS: f32 = PI / 180.0;

pub struct Gpu {
    // Physical device
    physical_device: vk::PhysicalDevice,
    _exts: Vec<vk::ExtensionProperties>,
    present_modes: Vec<vk::PresentModeKHR>,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    _properties: vk::PhysicalDeviceProperties,
    graphics_queue_idx: u32,
    present_queue_idx: u32,
    // Logical device
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
}

struct VulkanApp {
    window: winit::window::Window,
    _entry: ash::Entry,
    instance: ash::Instance,
    surface: vk::SurfaceKHR,
    // - Extensions
    ext_debug_utils: ash::extensions::ext::DebugUtils,
    ext_surface: ash::extensions::khr::Surface,
    ext_swapchain: ash::extensions::khr::Swapchain,

    gpu: Gpu,
    command_pool: vk::CommandPool,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    uniform_buffer_layout: vk::DescriptorSetLayout,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    apparatus: Apparatus, // Resolution-dependent apparatus

    debug_messenger: vk::DebugUtilsMessengerEXT,
    validation_layers: Vec<String>,
    current_frame: usize,
}

#[allow(dead_code)]
struct UniformBuffer {
    mtx_world_to_clip: Mat4,
}

fn create_buffer(
    gpu: &Gpu,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    required_memory_properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let device = &gpu.device;
    // Create buffer
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        device
            .create_buffer(&buffer_create_info, None)
            .expect("Failed to create buffer.")
    };
    // Locate memory type
    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = gpu
        .memory_properties
        .memory_types
        .iter()
        .enumerate()
        .position(|(i, &m)| {
            (mem_requirements.memory_type_bits & (1 << i)) > 0
                && m.property_flags.contains(required_memory_properties)
        })
        .expect("Failed to find suitable memory type.") as u32;
    // Allocate memory
    // TODO: Replace with allocator library?
    let allocate_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let buffer_memory = unsafe {
        device
            .allocate_memory(&allocate_info, None)
            .expect("Failed to allocate buffer memory.")
    };
    // Bind memory to buffer
    unsafe {
        device
            .bind_buffer_memory(buffer, buffer_memory, 0)
            .expect("Failed to bind buffer.");
    }

    (buffer, buffer_memory)
}

pub fn new_buffer<T>(
    data: &[T],
    usage: vk::BufferUsageFlags,
    gpu: &Gpu,
    command_pool: vk::CommandPool,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;

    // ## Create staging buffer in host-visible memory
    // TODO: Replace with allocator library?
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        &gpu,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );
    // ## Copy data to staging buffer
    unsafe {
        let data_ptr = gpu
            .device
            .map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )
            .expect("Failed to map memory.") as *mut T;

        data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        gpu.device.unmap_memory(staging_buffer_memory);
    }
    // ## Create buffer in device-local memory
    // TODO: Replace with allocator library?
    let (buffer, buffer_memory) = create_buffer(
        &gpu,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | usage,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );

    // ## Copy staging buffer -> vertex buffer
    {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe {
            gpu.device
                .allocate_command_buffers(&allocate_info)
                .expect("Failed to allocate command buffer.")
        };
        let command_buffer = command_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            gpu.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin command buffer.");

            let copy_regions = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: buffer_size,
            }];

            gpu.device
                .cmd_copy_buffer(command_buffer, staging_buffer, buffer, &copy_regions);

            gpu.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer");
        }

        let submit_info = [vk::SubmitInfo {
            command_buffer_count: command_buffers.len() as u32,
            p_command_buffers: command_buffers.as_ptr(),
            ..Default::default()
        }];

        unsafe {
            gpu.device
                .queue_submit(gpu.graphics_queue, &submit_info, vk::Fence::null())
                .expect("Failed to Submit Queue.");
            gpu.device
                .queue_wait_idle(gpu.graphics_queue)
                .expect("Failed to wait Queue idle");

            gpu.device
                .free_command_buffers(command_pool, &command_buffers);
        }
    }

    unsafe {
        gpu.device.destroy_buffer(staging_buffer, None);
        gpu.device.free_memory(staging_buffer_memory, None);
    }

    (buffer, buffer_memory)
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
            .expect("Failed to create shader module.")
    }
}

impl VulkanApp {
    pub fn recreate_resolution_dependent_state(&mut self) {
        unsafe {
            self.gpu
                .device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
        self.apparatus.destroy(&self.gpu, &self.ext_swapchain);
        self.apparatus = Apparatus::new(
            &self.window,
            self.surface,
            &self.gpu,
            self.command_pool,
            self.vertex_buffer,
            self.index_buffer,
            self.uniform_buffer_layout,
            &self.descriptor_sets,
            &self.ext_surface,
            &self.ext_swapchain,
        );
    }

    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> VulkanApp {
        const APP_NAME: &str = "";
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
                // Physical device
                physical_device: cgpu.physical_device,
                _exts: cgpu.exts.clone(),
                present_modes: cgpu.present_modes.clone(),
                memory_properties: cgpu.memory_properties,
                _properties: cgpu.properties,
                graphics_queue_idx: cgpu.graphics_queue_idx,
                present_queue_idx: cgpu.present_queue_idx,
                // Logical device
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

        let ext_swapchain = ash::extensions::khr::Swapchain::new(&instance, &gpu.device);

        // # Create and upload the vertex buffer
        let (vertex_buffer, vertex_buffer_memory) = {
            const VERTICES_DATA: [f32; 42] = [
                -4.0, -4.0, -4.0, 0.0, 0.0, 0.0, 4.0, -4.0, -4.0, 1.0, 0.0, 0.0, 4.0, 4.0, -4.0,
                1.0, 1.0, 0.0, -4.0, 4.0, -4.0, 0.0, 1.0, 0.0, 4.0, -4.0, 4.0, 1.0, 0.0, 1.0, 4.0,
                4.0, 4.0, 1.0, 1.0, 1.0, -4.0, -4.0, 4.0, 0.0, 0.0, 1.0,
            ];
            new_buffer(
                &VERTICES_DATA,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                &gpu,
                command_pool,
            )
        };

        // # Create and upload index buffer
        let (index_buffer, index_buffer_memory) = {
            const INDICES_DATA: [u32; 18] = [0, 2, 1, 2, 0, 3, 4, 1, 2, 4, 0, 1, 5, 4, 2, 6, 0, 4];

            new_buffer(
                &INDICES_DATA,
                vk::BufferUsageFlags::INDEX_BUFFER,
                &gpu,
                command_pool,
            )
        };

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
        let (uniform_buffers, uniform_buffers_memory) = {
            let buffer_size = std::mem::size_of::<UniformBuffer>();

            let mut uniform_buffers = vec![];
            let mut uniform_buffers_memory = vec![];

            for _ in 0..NUM_FRAMES {
                let (uniform_buffer, uniform_buffer_memory) = create_buffer(
                    &gpu,
                    buffer_size as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                uniform_buffers.push(uniform_buffer);
                uniform_buffers_memory.push(uniform_buffer_memory);
            }

            (uniform_buffers, uniform_buffers_memory)
        };

        // # Create descriptor pool
        let descriptor_pool = {
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: NUM_FRAMES as u32,
            }];

            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(NUM_FRAMES as u32)
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
            for _ in 0..NUM_FRAMES {
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
                    buffer: uniform_buffers[i],
                    offset: 0,
                    range: std::mem::size_of::<UniformBuffer>() as u64,
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

        // # Set up the apparatus
        let apparatus = Apparatus::new(
            &window,
            surface,
            &gpu,
            command_pool,
            vertex_buffer,
            index_buffer,
            uniform_buffer_layout,
            &descriptor_sets,
            &ext_surface,
            &ext_swapchain,
        );

        VulkanApp {
            window,
            _entry: entry,
            instance,
            surface,
            // - Extensions
            ext_debug_utils,
            ext_surface,
            ext_swapchain,
            // - Device
            gpu,
            command_pool,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            descriptor_pool,
            descriptor_sets,
            uniform_buffer_layout,
            uniform_buffers,
            uniform_buffers_memory,
            // - Resolution-dependent apparatus
            apparatus,

            debug_messenger,
            validation_layers,

            current_frame: 0,
        }
    }

    fn draw_frame(&mut self) {
        let wait_fences = [self.apparatus.command_buffer_complete_fences[self.current_frame]];

        let (image_index, _is_sub_optimal) = unsafe {
            self.gpu
                .device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence.");

            let result = self.ext_swapchain.acquire_next_image(
                self.apparatus.swapchain,
                std::u64::MAX,
                self.apparatus.image_available_semaphores[self.current_frame],
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

        // Update uniform buffer
        {
            let extent = &self.apparatus.swapchain_extent;
            let ubos = [UniformBuffer {
                mtx_world_to_clip: Mat4::perspective_lh(
                    60.0 * DEGREES_TO_RADIANS,
                    extent.width as f32 / extent.height as f32,
                    0.01,
                    100.0,
                ) * Mat4::from_translation(Vec3::new(0.0, 0.0, 18.0))
                    * Mat4::from_rotation_x(30.0 * DEGREES_TO_RADIANS)
                    * Mat4::from_rotation_y(45.0 * DEGREES_TO_RADIANS),
            }];

            let buffer_size = (std::mem::size_of::<UniformBuffer>() * ubos.len()) as u64;

            unsafe {
                let data_ptr =
                    self.gpu
                        .device
                        .map_memory(
                            self.uniform_buffers_memory[image_index as usize],
                            0,
                            buffer_size,
                            vk::MemoryMapFlags::empty(),
                        )
                        .expect("Failed to map memory.") as *mut UniformBuffer;

                data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

                self.gpu
                    .device
                    .unmap_memory(self.uniform_buffers_memory[image_index as usize]);
            }
        }

        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let wait_semaphores = [self.apparatus.image_available_semaphores[self.current_frame]];
        let signal_semaphores = [self.apparatus.render_finished_semaphores[self.current_frame]];
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
                    self.apparatus.command_buffer_complete_fences[self.current_frame],
                )
                .expect("Failed to execute queue submit.");
        }

        let swapchains = [self.apparatus.swapchain];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        // Present the queue
        {
            let result = unsafe {
                self.ext_swapchain
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

        self.current_frame = (self.current_frame + 1) % NUM_FRAMES;
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.gpu
                .device
                .destroy_command_pool(self.command_pool, None);

            self.apparatus.destroy(&self.gpu, &self.ext_swapchain);

            // Uniform buffer
            self.gpu
                .device
                .destroy_descriptor_set_layout(self.uniform_buffer_layout, None);
            self.gpu
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            for i in 0..self.uniform_buffers.len() {
                self.gpu
                    .device
                    .destroy_buffer(self.uniform_buffers[i], None);
                self.gpu
                    .device
                    .free_memory(self.uniform_buffers_memory[i], None);
            }
            // Vertex buffer
            self.gpu.device.destroy_buffer(self.vertex_buffer, None);
            self.gpu.device.free_memory(self.vertex_buffer_memory, None);
            // Index buffer
            self.gpu.device.destroy_buffer(self.index_buffer, None);
            self.gpu.device.free_memory(self.index_buffer_memory, None);

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

impl VulkanApp {
    pub fn main_loop(mut self, event_loop: EventLoop<()>) {
        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(physical_size) => {
                    if self.apparatus.swapchain_extent.width != physical_size.width ||
                        self.apparatus.swapchain_extent.height != physical_size.height
                    {
                        self.recreate_resolution_dependent_state();
                    }
                },
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

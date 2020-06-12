use crate::*;
use std::os::raw::c_char;

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

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

impl Gpu {
    pub fn new(basis: &mut Basis) -> Gpu {
        let required_exts = vec![String::from("VK_KHR_swapchain")];

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
                &basis
                    .instance
                    .enumerate_physical_devices()
                    .expect("Failed to enumerate Physical Devices!")
            };

            let mut candidate_gpus = Vec::new();

            for &physical_device in physical_devices {
                let exts = unsafe {
                    basis
                        .instance
                        .enumerate_device_extension_properties(physical_device)
                        .expect("Failed to get device extension properties.")
                };
                // Are desired extensions supported?
                let are_exts_supported = {
                    let available_exts: Vec<String> = exts
                        .iter()
                        .map(|&ext| vk_to_string(&ext.extension_name))
                        .collect();

                    required_exts.iter().all(|desired_ext| {
                        available_exts
                            .iter()
                            .any(|available_ext| desired_ext == available_ext)
                    })
                };
                if !are_exts_supported {
                    continue;
                }

                let surface_formats = unsafe {
                    basis
                        .ext_surface
                        .get_physical_device_surface_formats(physical_device, basis.surface)
                        .expect("Failed to query for surface formats.")
                };
                let present_modes = unsafe {
                    basis
                        .ext_surface
                        .get_physical_device_surface_present_modes(physical_device, basis.surface)
                        .expect("Failed to query for surface present mode.")
                };
                // Are there any surface formats and present modes?
                if surface_formats.is_empty() || present_modes.is_empty() {
                    continue;
                }

                let memory_properties = unsafe {
                    basis
                        .instance
                        .get_physical_device_memory_properties(physical_device)
                };
                let properties = unsafe {
                    basis
                        .instance
                        .get_physical_device_properties(physical_device)
                };

                // Queue family indices
                let queue_families = unsafe {
                    basis
                        .instance
                        .get_physical_device_queue_family_properties(physical_device)
                };
                let opt_graphics_queue_idx = queue_families.iter().position(|&fam| {
                    fam.queue_count > 0 && fam.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                });
                let opt_present_queue_idx =
                    queue_families.iter().enumerate().position(|(i, &fam)| {
                        let is_present_supported = unsafe {
                            basis.ext_surface.get_physical_device_surface_support(
                                physical_device,
                                i as u32,
                                basis.surface,
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
            // Currently, we just pick the first one.
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

            let raw_ext_names: Vec<CString> = required_exts
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
                enabled_extension_count: required_exts.len() as u32,
                pp_enabled_extension_names: ext_names.as_ptr(),
                p_enabled_features: &physical_device_features,
            };

            let device: ash::Device = unsafe {
                basis
                    .instance
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

        // If the debug marker extension is supported, enable it so we can have nice markers in RenderDoc
        {
            let exts = unsafe {
                basis
                    .instance
                    .enumerate_device_extension_properties(gpu.physical_device)
                    .expect("Failed to get device extension properties.")
            };
            // Are desired extensions supported?
            let is_debug_marker_supported = exts.iter().any(|&ext| {
                vk_to_string(&ext.extension_name) == String::from("VK_EXT_debug_marker")
            });
            if is_debug_marker_supported {
                basis.opt_ext_debug_marker = Some(ash::extensions::ext::DebugMarker::new(
                    &basis.instance,
                    &gpu.device,
                ));
            }
        }

        gpu
    }
}

use crate::*;
use ash::vk_make_version;
use std::os::raw::c_char;
use winit::window::Window;

pub struct Basis {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface: vk::SurfaceKHR,
    pub validation_layers: Vec<String>,

    // - Extensions
    pub ext_surface: ash::extensions::khr::Surface,
}

impl Drop for Basis {
    fn drop(&mut self) {
        unsafe {
            self.ext_surface.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Basis {
    pub fn new(app_name: &str, window: &Window) -> Basis {
        let validation_layers = vec![String::from("VK_LAYER_KHRONOS_validation")];

        // # Init Ash
        let entry = ash::Entry::new().unwrap();

        // # Create Vulkan instance
        let instance = {
            let app_name = CString::new(app_name).unwrap();
            let engine_name = CString::new("graphene").unwrap();
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

        // # Create surface
        let ext_surface = ash::extensions::khr::Surface::new(&entry, &instance);
        let surface = unsafe {
            platforms::create_surface(&entry, &instance, &window)
                .expect("Failed to create surface.")
        };

        Basis {
            instance,
            surface,
            validation_layers,
            entry,
            ext_surface,
        }
    }
}

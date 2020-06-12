use crate::*;
use ash::vk_make_version;
use std::os::raw::c_char;
use std::os::raw::c_void;
use winit::window::Window;

pub struct Basis {
    pub _entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface: vk::SurfaceKHR,
    pub validation_layers: Vec<String>,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,

    // - Extensions
    pub ext_debug_utils: ash::extensions::ext::DebugUtils,
    pub ext_surface: ash::extensions::khr::Surface,
    pub opt_ext_debug_marker: Option<ash::extensions::ext::DebugMarker>,
}

impl Drop for Basis {
    fn drop(&mut self) {
        unsafe {
            self.ext_surface.destroy_surface(self.surface, None);
            if !self.validation_layers.is_empty() {
                self.ext_debug_utils
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

impl Basis {
    pub fn new(app_name: &str, window: &Window) -> Basis {
        const ENABLE_DEBUG_MESSENGER_CALLBACK: bool = true;
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

        // # Debug messenger callback
        let ext_debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);
        let debug_messenger = {
            if !ENABLE_DEBUG_MESSENGER_CALLBACK {
                ash::vk::DebugUtilsMessengerEXT::null()
            } else {
                let messenger_ci = vk::DebugUtilsMessengerCreateInfoEXT {
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
                };

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

        Basis {
            instance,
            surface,
            validation_layers,
            debug_messenger,
            _entry: entry,
            ext_debug_utils,
            ext_surface,
            opt_ext_debug_marker: None,
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

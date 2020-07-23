use crate::*;
use ash::vk::Handle;
use std::os::raw::c_void;

pub struct DebugUtils {
    device: ash::Device,
    pub enable_messenger_callback: bool,
    pub ext: ash::extensions::ext::DebugUtils,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl Drop for DebugUtils {
    fn drop(&mut self) {
        unsafe {
            if self.enable_messenger_callback {
                self.ext
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
        }
    }
}

impl DebugUtils {
    pub fn new(basis: &Basis, gpu: &Gpu, enable_messenger_callback: bool) -> DebugUtils {
        // # Debug messenger callback
        let ext = ash::extensions::ext::DebugUtils::new(&basis.entry, &basis.instance);
        let debug_messenger = {
            if !enable_messenger_callback {
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
                    ext.create_debug_utils_messenger(&messenger_ci, None)
                        .expect("Debug Utils Callback")
                }
            }
        };

        DebugUtils {
            device: gpu.device.clone(),
            enable_messenger_callback,
            ext,
            debug_messenger,
        }
    }

    pub fn set_image_name(&self, vk_image: vk::Image, name: &str) {
        let c_name = CString::new(name).unwrap();
        let info = ash::vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_type(vk::ObjectType::IMAGE)
            .object_handle(vk_image.as_raw())
            .object_name(&c_name)
            .build();
        unsafe {
            self.ext
                .debug_utils_set_object_name(self.device.handle(), &info)
                .unwrap();
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

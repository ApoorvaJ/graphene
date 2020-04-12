use crate::utility::structures::DeviceExtension;
use ash::vk_make_version;

use std::os::raw::c_char;

pub const DEVICE_EXTENSIONS: DeviceExtension = DeviceExtension {
    names: ["VK_KHR_swapchain"],
};
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
pub const IS_PAINT_FPS_COUNTER: bool = false;

impl DeviceExtension {
    pub fn get_extensions_raw_names(&self) -> [*const c_char; 1] {
        [
            // currently just enable the Swapchain extension.
            ash::extensions::khr::Swapchain::name().as_ptr(),
        ]
    }
}

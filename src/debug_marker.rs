use crate::*;

pub struct DebugMarker {
    pub opt_ext: Option<ash::extensions::ext::DebugMarker>,
}

impl DebugMarker {
    pub fn new(basis: &Basis, gpu: &Gpu) -> DebugMarker {
        // If the debug marker extension is supported, enable it so we can have nice markers in RenderDoc
        // Are desired extensions supported?
        let is_debug_marker_supported = gpu
            .exts
            .iter()
            .any(|&ext| vk_to_string(&ext.extension_name) == String::from("VK_EXT_debug_marker"));

        let mut opt_ext = None;
        if is_debug_marker_supported {
            opt_ext = Some(ash::extensions::ext::DebugMarker::new(
                &basis.instance,
                &gpu.device,
            ));
        }

        DebugMarker { opt_ext }
    }
}

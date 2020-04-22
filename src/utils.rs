use crate::*;

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

pub fn get_shader_modules(gpu: &Gpu) -> Vec<vk::ShaderModule> {
    // Ensure that shader cache directory exists
    std::fs::create_dir_all("_shadercache").expect("Couldn't create shader cache directory");
    // Run glslc to compile GLSL shaders to SPIR-V
    let glsl_files = ["default.vert", "default.frag"];
    for &glsl_file in glsl_files.iter() {
        use std::process::Command;
        let glslc_output = Command::new("glslc")
            .arg(&format!("assets/shaders/{}", glsl_file))
            .arg("-o")
            .arg(&format!("_shadercache/{}.spv", glsl_file))
            .output()
            .expect("`glslc`, the GLSL -> SPIR-V compiler, could not be invoked.");
        if !glslc_output.status.success() {
            println!(
                "{}:\n    {}",
                glsl_file,
                String::from_utf8(glslc_output.stderr).unwrap()
            );
        }
    }

    // Load shaders from the SPIR-V files
    glsl_files
        .iter()
        .map(|&glsl_file| {
            let spirv_file = &format!("_shadercache/{}.spv", glsl_file);
            create_shader_module(
                &gpu.device,
                std::fs::read(spirv_file)
                    .expect(&format!("Failed to read shader file {}", spirv_file)),
            )
        })
        .collect()
}

use crate::*;
use std::path::Path;

pub enum ShaderStage {
    Vertex,
    Fragment,
}

pub struct InternalShader {
    pub name: String,
    pub shader_stage: ShaderStage,
    pub source_path: String,
    pub spirv_path: String,
    pub vk_shader_module: vk::ShaderModule,
}

pub struct ShaderList {
    device: ash::Device,
    pub list: Vec<(ShaderHandle, InternalShader)>,
}

impl Drop for ShaderList {
    fn drop(&mut self) {
        unsafe {
            for (_, shader) in &self.list {
                self.device
                    .destroy_shader_module(shader.vk_shader_module, None);
            }
        }
    }
}

impl ShaderList {
    pub fn new(device: ash::Device) -> ShaderList {
        ShaderList {
            device,
            list: Vec::new(),
        }
    }

    pub fn new_shader(
        &mut self,
        name: &str,
        shader_stage: ShaderStage,
        path: &str,
    ) -> Result<ShaderHandle, String> {
        // Hash
        let handle = {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            ShaderHandle(hasher.finish())
        };
        // Error if name already exists
        if self.get_shader_from_handle(handle).is_some() {
            return Err(format!(
                "A shader with the name `{}` already exists in the context.",
                name
            ));
        }
        // Get shader module (compile if required)
        const SHADER_CACHE_PATH: &str = "_cache/shaders";
        std::fs::create_dir_all(SHADER_CACHE_PATH).expect("Could not create the _cache directory.");
        let source_path = String::from(&format!("assets/shaders/{}", path));
        let spirv_path = String::from(&format!("{}/{}.spv", SHADER_CACHE_PATH, path));
        let is_compilation_needed = is_compilation_needed(&source_path, &spirv_path);
        let vk_shader_module = get_shader_module(
            &self.device,
            &source_path,
            &spirv_path,
            is_compilation_needed,
        )?;
        // Insert
        self.list.push((
            handle,
            InternalShader {
                name: String::from(name),
                shader_stage,
                source_path,
                spirv_path,
                vk_shader_module,
            },
        ));
        Ok(handle)
    }

    pub fn get_shader_from_handle(&self, shader_handle: ShaderHandle) -> Option<&InternalShader> {
        for (handle, shader) in &self.list {
            if *handle == shader_handle {
                return Some(shader);
            }
        }
        None
    }

    pub fn hot_reload(&mut self, graph_cache: &mut Vec<(Graph, GraphHandle)>) {
        for (shader_handle, shader) in &mut self.list {
            if !is_compilation_needed(&shader.source_path, &shader.spirv_path) {
                continue;
            }

            if let Ok(vk_shader_module) =
                get_shader_module(&self.device, &shader.source_path, &shader.spirv_path, true)
            {
                // Evict any graphs that contain the shaders that need to be updated
                graph_cache.retain(|(graph, _)| !graph.shader_handles.contains(shader_handle));

                unsafe {
                    self.device
                        .destroy_shader_module(shader.vk_shader_module, None);
                    shader.vk_shader_module = vk_shader_module;
                }
            }
        }
    }
}

/// Check if spirv file exists and if it is stale
fn is_compilation_needed(source_path: &str, spirv_path: &str) -> bool {
    let src_path = Path::new(source_path);
    let dst_path = Path::new(spirv_path);

    assert!(
        src_path.exists(),
        "Shader file `{}` doesn't exist",
        source_path
    );

    if !dst_path.exists() {
        // ...SPIR-V file doesn't exist. Compilation needed.
        return true;
    }

    let src_meta = src_path
        .metadata()
        .unwrap_or_else(|_| panic!("Couldn't retrieve metadata for `{}`", spirv_path));
    if let Ok(dst_meta) = dst_path.metadata() {
        if let Ok(dst_modified) = dst_meta.modified() {
            let src_modified = src_meta.modified().unwrap();
            if dst_modified.duration_since(src_modified).is_ok() {
                // ...Src was modified earlier than destination, i.e. no
                // compilation needed
                return false;
            }
        }
    }

    true
}

// We pretty-print the error here instead of returning it as a Err(String).
// Might want to change this behavior at some point.
fn compile_shader(source_path: &str, spirv_path: &str) -> Result<(), String> {
    print!("Compiling `{}`...", source_path);
    let glslc_output = std::process::Command::new("glslc")
        .arg(source_path)
        .arg("-o")
        .arg(spirv_path)
        .output()
        .expect("`glslc`, the GLSL -> SPIR-V compiler, could not be invoked.");
    if !glslc_output.status.success() {
        println!(" failed:");
        // Print error message with indentation
        let err = String::from_utf8(glslc_output.stderr).unwrap();
        for err_line in err.lines() {
            println!("    {}", err_line);
        }
        Err(String::from("Shader compilation failed"))
    } else {
        println!(" OK.");
        Ok(())
    }
}

fn get_shader_module(
    device: &ash::Device,
    source_path: &str,
    spirv_path: &str,
    is_compilation_needed: bool,
) -> Result<vk::ShaderModule, String> {
    // If spirv path doesn't exist, compile the shader
    if is_compilation_needed {
        compile_shader(source_path, spirv_path)?;
    }

    // Read the spirv file
    let spirv_u8 = std::fs::read(spirv_path)
        .unwrap_or_else(|_| panic!("Failed to read spirv file `{}`", spirv_path));
    // Create the shader module
    let spirv_u32 = {
        /* This is needed because std::fs::read returns a Vec<u8>, but Vulkan
        wants a &[u32] slice.

        We break the slice into a prefix, middle and suffix, and make sure that
        the prefix and suffix are empty. This ensures that we don't miss
        alignment and get invalid data. */
        let (prefix_u8, middle_u32, suffix_u8) = unsafe { spirv_u8.align_to::<u32>() };
        assert_eq!(prefix_u8.len(), 0);
        assert_eq!(suffix_u8.len(), 0);
        middle_u32
    };
    let create_info = vk::ShaderModuleCreateInfo::builder().code(spirv_u32);

    let vk_shader_module = unsafe {
        device
            .create_shader_module(&create_info, None)
            .expect("Failed to create shader module.")
    };

    Ok(vk_shader_module)
}

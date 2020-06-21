use crate::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Hash)]
pub struct Pass {
    pub name: String,
    pub outputs: Vec<(vk::ImageView, vk::Format)>,
    pub input_texture: (vk::ImageView, vk::Sampler),
    pub opt_depth: Option<(vk::ImageView, vk::Format)>,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub shader_modules: Vec<vk::ShaderModule>,
    pub buffer_info: (vk::Buffer, u64), // (vk_buffer, size) // TODO: Remove
    pub uniform_buffer: BufferHandle,
}

#[derive(Hash)]
pub struct GraphBuilder {
    pub passes: Vec<Pass>,
    pub buffer_aliases: Vec<(BufferHandle, usize)>,
}

impl GraphBuilder {
    pub fn new() -> GraphBuilder {
        GraphBuilder {
            passes: Vec::new(),
            buffer_aliases: Vec::new(),
        }
    }

    pub fn new_uniform_buffer(&mut self, name: &str, size: usize) -> Result<BufferHandle, String> {
        // Hash buffer name
        let hash: u64 = {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            hasher.finish()
        };
        // If buffer with same hash already exists, return error
        if self
            .buffer_aliases
            .iter()
            .find(|(handle, _)| handle.0 == hash)
            .is_some()
        {
            return Err(format!(
                "A texture with the same name `{}` already exists in the context.",
                name
            ));
        }

        let handle = BufferHandle(hash);
        self.buffer_aliases.push((handle, size));
        Ok(handle)
    }
}

use crate::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct InternalBuffer {
    pub handle: BufferHandle,
    pub host_visible_buffer: HostVisibleBuffer,
}

impl Context {
    fn get_buffer_from_hash(&self, hash: u64) -> Option<&InternalBuffer> {
        let opt_buffer = self.buffer_list.iter().find(|b| b.handle.0 == hash);

        if let Some(b) = opt_buffer {
            return Some(b);
        }

        return None;
    }

    pub fn new_buffer(&mut self, name: &str, size: usize) -> Result<BufferHandle, String> {
        // Hash buffer name
        let new_hash: u64 = {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            hasher.finish()
        };
        // If buffer with same hash already exists, return error
        if self.get_buffer_from_hash(new_hash).is_some() {
            return Err(format!(
                "A texture with the same name `{}` already exists in the context.",
                name
            ));
        }
        let host_visible_buffer =
            HostVisibleBuffer::new(size, vk::BufferUsageFlags::UNIFORM_BUFFER, &self.gpu);
        self.buffer_list.push(InternalBuffer {
            handle: BufferHandle(new_hash),
            host_visible_buffer,
        });
        Ok(BufferHandle(new_hash))
    }
}

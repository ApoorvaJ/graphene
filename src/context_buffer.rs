use crate::*;

// TODO: Add support for buffer aliases, so that intermediate buffers with
// different handles can use the same underlying memory, as long as they
// don't overlap.

pub struct InternalBuffer {
    pub handle: BufferHandle,
    pub name: String,
    pub buffer: HostVisibleBuffer,
}

impl Context {
    pub fn get_buffer_from_hash(&self, hash: u64) -> Option<&InternalBuffer> {
        self.buffer_list.iter().find(|buf| buf.handle.0 == hash)
    }

    pub fn new_buffer(
        &mut self,
        name: &str,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferHandle, String> {
        // Hash buffer name
        let new_hash: u64 = {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            hasher.finish()
        };
        // If buffer with same hash already exists, return error
        if self.get_buffer_from_hash(new_hash).is_some() {
            // TODO: Need more clarity on buffer creation API design
            return Ok(BufferHandle(new_hash));
            // return Err(format!(
            //     "A buffer with the same name `{}` already exists in the context.",
            //     name
            // ));
        }
        // Create new buffer
        let buffer = HostVisibleBuffer::new(size, usage, &self.gpu);
        self.buffer_list.push(InternalBuffer {
            handle: BufferHandle(new_hash),
            name: String::from(name),
            buffer,
        });

        Ok(BufferHandle(new_hash))
    }
}

use crate::*;

// TODO: Add support for buffer aliases, so that intermediate buffers with
// different handles can use the same underlying memory, as long as they
// don't overlap.

pub struct BufferList {
    pub list: Vec<(BufferHandle, HostVisibleBuffer)>, // TODO: Support device local buffers too
}

impl BufferList {
    pub fn new() -> BufferList {
        BufferList { list: Vec::new() }
    }

    pub fn new_buffer(
        &mut self,
        name: &str,
        size: usize,
        usage: vk::BufferUsageFlags,
        gpu: &Gpu,
        debug_utils: &DebugUtils,
    ) -> Result<BufferHandle, String> {
        // Hash
        let handle = {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            BufferHandle(hasher.finish())
        };
        // Error if name already exists
        if self.get_buffer_from_handle(handle).is_some() {
            return Err(format!(
                "A buffer with the same name `{}` already exists in the context.",
                name
            ));
        }
        // Create and insert new buffer
        let buffer = HostVisibleBuffer::new(name, size, usage, gpu, debug_utils);
        self.list.push((handle, buffer));

        Ok(handle)
    }

    pub fn get_buffer_from_handle(
        &self,
        buffer_handle: BufferHandle,
    ) -> Option<&HostVisibleBuffer> {
        for (handle, buffer) in &self.list {
            if *handle == buffer_handle {
                return Some(buffer);
            }
        }
        None
    }

    pub fn upload_data<T>(&self, buffer_handle: BufferHandle, data: &[T]) {
        let internal_buffer = self.get_buffer_from_handle(buffer_handle).unwrap_or_else(||panic!(
            "A buffer with the hash `{}` not found in the context.",
            buffer_handle.0
        ));
        internal_buffer.upload_data(data, 0);
    }
}

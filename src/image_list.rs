use crate::*;

#[derive(Copy, Clone, PartialEq)]
pub enum ImageKind {
    Swapchain,
    AbsoluteSized,
    RelativeSized { scale: f32 }, // Scale relative to the swapchain size
}

pub struct InternalImage {
    pub image: Image,
    pub kind: ImageKind,
}

pub struct ImageList {
    pub list: Vec<(ImageHandle, InternalImage)>,
}

impl ImageList {
    pub fn new() -> ImageList {
        ImageList { list: Vec::new() }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_image_relative_size(
        &mut self,
        name: &str,
        scale: f32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect_flags: vk::ImageAspectFlags,
        facade: &Facade,
        gpu: &Gpu,
        debug_utils: &DebugUtils,
    ) -> Result<ImageHandle, String> {
        // Hash
        let handle = {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            ImageHandle(hasher.finish())
        };
        // Error if name already exists
        if self.get_image_from_handle(handle).is_some() {
            return Err(format!(
                "An image with the same name `{}` already exists in the context.",
                name
            ));
        }
        // Create new image
        let w = (facade.swapchain_width as f32 * scale) as u32;
        let h = (facade.swapchain_height as f32 * scale) as u32;
        let image = Image::new(name, w, h, format, usage, aspect_flags, gpu, &debug_utils);
        self.list.push((
            handle,
            InternalImage {
                image,
                kind: ImageKind::RelativeSized { scale },
            },
        ));

        Ok(handle)
    }

    pub fn new_image_from_file(
        &mut self,
        name: &str,
        path: &str,
        gpu: &Gpu,
        command_pool: vk::CommandPool,
        debug_utils: &DebugUtils,
    ) -> Result<ImageHandle, String> {
        // Hash
        let handle = {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            ImageHandle(hasher.finish())
        };
        // Error if name already exists
        if self.get_image_from_handle(handle).is_some() {
            return Err(format!(
                "An image with the same name `{}` already exists in the context.",
                name
            ));
        }
        // Create new image
        let image = Image::new_from_image(
            gpu,
            std::path::Path::new(&path),
            command_pool,
            name,
            &debug_utils,
        );
        self.list.push((
            handle,
            InternalImage {
                image,
                kind: ImageKind::AbsoluteSized,
            },
        ));

        Ok(handle)
    }

    pub fn get_image_from_handle(&self, image_handle: ImageHandle) -> Option<&InternalImage> {
        for (handle, internal_image) in &self.list {
            if *handle == image_handle {
                return Some(internal_image);
            }
        }
        None
    }
}

use crate::*;

// TODO: This module is not a core part of the render graph. Make that clear from the hierarchy.

pub struct Mesh {
    pub vertex_buffer: DeviceLocalBuffer,
    pub index_buffer: DeviceLocalBuffer,
}

impl Mesh {
    pub fn load(
        path: &str,
        gpu: &Gpu,
        command_pool: vk::CommandPool,
        debug_utils: &DebugUtils,
    ) -> Mesh {
        // TODO: Benchmark and optimize
        let (vertices_data, indices_data) = {
            let mut vertices_data: Vec<f32> = Vec::new();
            let mut indices_data: Vec<u32> = Vec::new();

            let (gltf, buffers, _) = gltf::import(path).expect("Failed to open mesh.");
            for mesh in gltf.meshes() {
                for primitive in mesh.primitives() {
                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                    if let Some(iter_pos) = reader.read_positions() {
                        if let Some(iter_norm) = reader.read_normals() {
                            for (pos, norm) in iter_pos.zip(iter_norm) {
                                vertices_data.extend_from_slice(&pos);
                                vertices_data.extend_from_slice(&norm);
                            }
                        }
                    }
                    if let Some(iter) = reader.read_indices() {
                        match iter {
                            gltf::mesh::util::ReadIndices::U8(iter_2) => {
                                for idx in iter_2 {
                                    indices_data.push(idx as u32);
                                }
                            }
                            gltf::mesh::util::ReadIndices::U16(iter_2) => {
                                for idx in iter_2 {
                                    indices_data.push(idx as u32);
                                }
                            }
                            gltf::mesh::util::ReadIndices::U32(iter_2) => {
                                for idx in iter_2 {
                                    indices_data.push(idx as u32);
                                }
                            }
                        }
                    }
                }
            }

            (vertices_data, indices_data)
        };

        // # Create and upload the vertex buffer
        let vertex_buffer = DeviceLocalBuffer::new(
            &vertices_data,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            gpu,
            command_pool,
            debug_utils,
        );

        // # Create and upload index buffer
        let index_buffer = DeviceLocalBuffer::new(
            &indices_data,
            vk::BufferUsageFlags::INDEX_BUFFER,
            gpu,
            command_pool,
            debug_utils,
        );

        Mesh {
            vertex_buffer,
            index_buffer,
        }
    }
}

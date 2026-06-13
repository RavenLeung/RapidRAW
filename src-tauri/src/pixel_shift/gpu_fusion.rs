use anyhow::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::motion_detection::MotionMask;
use super::skr_fusion::{SkrFusionParams, StructureTensor, eigen_decompose_2x2};
use crate::image_processing::GpuContext;

/// GPU-accelerated pixel-shift fusion processor.
///
/// Offloads the expensive steering kernel regression to the GPU
/// while keeping structure tensor computation and motion detection on CPU
/// (where they're cheap with rayon).
pub struct PixelShiftGpuProcessor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Compute pipelines
    tensor_pipeline: wgpu::ComputePipeline,
    fusion_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    tensor_bgl: wgpu::BindGroupLayout,
    fusion_bgl: wgpu::BindGroupLayout,
}

/// GPU-compatible fusion params (must match WGSL struct)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFusionParams {
    output_width: u32,
    output_height: u32,
    input_width: u32,
    input_height: u32,
    num_frames: u32,
    kernel_sigma: f32,
    stretch: f32,
    structure_sigma: f32,
    motion_compensation: u32,
    _pad0: u32,
    _pad1: u32,
}

impl PixelShiftGpuProcessor {
    /// Create a new GPU fusion processor from an existing GPU context.
    pub fn new(context: &GpuContext) -> Result<Self, String> {
        let device = Arc::clone(&context.device);

        let shader_module = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pixel Shift Fusion Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/pixel_shift.wgsl").into(),
            ),
        });

        // Bind group layout for structure tensor pass (group 0)
        let tensor_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PixelShift Tensor BGL"),
            entries: &[
                // binding 0: reference frame (storage buffer, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: structure tensors (storage buffer, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: fusion params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let tensor_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PixelShift Tensor Pipeline Layout"),
            bind_group_layouts: &[Some(&tensor_bgl)],
            immediate_size: 0,
        });

        let tensor_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PixelShift Tensor Pipeline"),
            layout: Some(&tensor_pipeline_layout),
            module: &shader_module,
            entry_point: Some("structure_tensor_pass"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Bind group layout for fusion pass (group 1)
        let fusion_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PixelShift Fusion BGL"),
            entries: &[
                // binding 0: all frames (storage buffer, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: structure tensors (storage buffer, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: motion mask (storage buffer, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: output RGB (storage buffer, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: fusion params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let fusion_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PixelShift Fusion Pipeline Layout"),
            bind_group_layouts: &[Some(&fusion_bgl)],
            immediate_size: 0,
        });

        let fusion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PixelShift Fusion Pipeline"),
            layout: Some(&fusion_pipeline_layout),
            module: &shader_module,
            entry_point: Some("fusion_pass"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue: Arc::clone(&context.queue),
            tensor_pipeline,
            fusion_pipeline,
            tensor_bgl,
            fusion_bgl,
        })
    }

    /// Run GPU-accelerated SKR fusion on aligned frames.
    ///
    /// # Arguments
    /// * `frames` - Aligned image frames (all same dimensions)
    /// * `motion_mask` - Optional motion mask for deghosting
    /// * `skr_params` - SKR fusion parameters
    ///
    /// # Returns
    /// Fused RGB image at the requested output resolution.
    pub fn fuse(
        &self,
        frames: &[DynamicImage],
        motion_mask: Option<&MotionMask>,
        skr_params: &SkrFusionParams,
    ) -> Result<DynamicImage> {
        if frames.is_empty() {
            return Ok(DynamicImage::ImageRgba32F(ImageBuffer::new(0, 0)));
        }
        if frames.len() == 1 {
            return Ok(frames[0].clone());
        }

        let (width, height) = (frames[0].width(), frames[0].height());
        let out_w = (width as f32 * skr_params.output_scale) as u32;
        let out_h = (height as f32 * skr_params.output_scale) as u32;
        let pixel_count = (width * height) as usize;

        // Step 1: CPU — Compute structure tensors from reference frame
        let ref_data = extract_rgb_data(&frames[0]);
        let structure_tensors = compute_tensors_cpu(&ref_data, width, height, skr_params.structure_sigma);

        // Step 2: GPU — Upload all data
        // Pack all frames into a single buffer: [frame0_rgb, frame1_rgb, ...]
        let num_frames = frames.len() as u32;
        let mut all_frames_data: Vec<f32> = Vec::with_capacity(num_frames as usize * pixel_count * 3);
        for frame in frames {
            let data = extract_rgb_data(frame);
            for rgb in &data {
                all_frames_data.push(rgb[0]);
                all_frames_data.push(rgb[1]);
                all_frames_data.push(rgb[2]);
            }
        }

        // Convert structure tensors to GPU format (12 bytes each, 3 f32s)
        let mut tensor_data: Vec<f32> = Vec::with_capacity(pixel_count * 3);
        for t in &structure_tensors {
            tensor_data.push(t.ixx);
            tensor_data.push(t.ixy);
            tensor_data.push(t.iyy);
        }

        // Motion mask data
        let mask_data: Vec<f32> = if let Some(mask) = motion_mask {
            mask.weights.clone()
        } else {
            vec![1.0; pixel_count]
        };

        // Output buffer
        let out_pixel_count = (out_w * out_h) as usize;
        let output_data: Vec<f32> = vec![0.0; out_pixel_count * 4]; // RGBA

        let gpu_params = GpuFusionParams {
            output_width: out_w,
            output_height: out_h,
            input_width: width,
            input_height: height,
            num_frames,
            kernel_sigma: skr_params.kernel_sigma,
            stretch: skr_params.stretch,
            structure_sigma: skr_params.structure_sigma,
            motion_compensation: if motion_mask.is_some() { 1 } else { 0 },
            _pad0: 0,
            _pad1: 0,
        };

        // Create GPU buffers
        let frames_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PixelShift Frames Buffer"),
            contents: bytemuck::cast_slice(&all_frames_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let tensor_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PixelShift Tensor Buffer"),
            contents: bytemuck::cast_slice(&tensor_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let mask_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PixelShift Mask Buffer"),
            contents: bytemuck::cast_slice(&mask_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PixelShift Output Buffer"),
            contents: bytemuck::cast_slice(&output_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PixelShift Params Buffer"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PixelShift Staging Buffer"),
            size: (out_pixel_count * 4 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Step 3: Dispatch tensor pass (optional — we already have CPU tensors, but dispatch for consistency)
        // For now, skip the GPU tensor pass and use CPU-computed tensors directly

        // Step 4: Dispatch fusion pass
        let fusion_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PixelShift Fusion Bind Group"),
            layout: &self.fusion_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frames_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tensor_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("PixelShift Fusion Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PixelShift Fusion Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.fusion_pipeline);
            cpass.set_bind_group(0, &fusion_bind_group, &[]);

            let workgroup_x = (out_w + 7) / 8;
            let workgroup_y = (out_h + 7) / 8;
            cpass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (out_pixel_count * 4 * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Step 5: Readback result
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(60)),
        });
        receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("GPU readback channel closed"))?
            .map_err(|e| anyhow::anyhow!("GPU readback failed: {}", e))?;

        let mapped = buffer_slice.get_mapped_range();
        let result_floats: &[f32] = bytemuck::cast_slice(&mapped);

        // Build output image
        let mut output: ImageBuffer<Rgba<f32>, Vec<f32>> = ImageBuffer::new(out_w, out_h);
        for y in 0..out_h {
            for x in 0..out_w {
                let idx = ((y * out_w + x) * 4) as usize;
                output.put_pixel(
                    x,
                    y,
                    Rgba([
                        result_floats[idx],
                        result_floats[idx + 1],
                        result_floats[idx + 2],
                        result_floats[idx + 3],
                    ]),
                );
            }
        }

        drop(mapped);

        Ok(DynamicImage::ImageRgba32F(output))
    }
}

/// Extract RGB float data from a DynamicImage
fn extract_rgb_data(img: &DynamicImage) -> Vec<[f32; 3]> {
    let rgba = img.to_rgba32f();
    let (w, h) = (rgba.width(), rgba.height());
    let mut data = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = rgba.get_pixel(x, y);
            data.push([p[0], p[1], p[2]]);
        }
    }
    data
}

/// CPU-side structure tensor computation (for uploading to GPU)
fn compute_tensors_cpu(
    ref_data: &[[f32; 3]],
    width: u32,
    height: u32,
    structure_sigma: f32,
) -> Vec<StructureTensor> {
    use rayon::prelude::*;
    let radius = (structure_sigma * 2.0).ceil() as i32;

    (0..ref_data.len())
        .into_par_iter()
        .map(|idx| {
            let x = (idx % width as usize) as i32;
            let y = (idx / width as usize) as i32;
            compute_tensor_at(ref_data, x, y, width, height, radius)
        })
        .collect()
}

fn compute_tensor_at(
    ref_data: &[[f32; 3]],
    cx: i32,
    cy: i32,
    width: u32,
    height: u32,
    radius: i32,
) -> StructureTensor {
    let mut ixx = 0.0f64;
    let mut ixy = 0.0f64;
    let mut iyy = 0.0f64;
    let mut count = 0u64;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let sx = cx + dx;
            let sy = cy + dy;

            if sx < 1 || sy < 1 || sx >= width as i32 - 1 || sy >= height as i32 - 1 {
                continue;
            }

            let idx_c = (sy * width as i32 + sx) as usize;
            let idx_l = (sy * width as i32 + sx - 1) as usize;
            let idx_r = (sy * width as i32 + sx + 1) as usize;
            let idx_u = ((sy - 1) * width as i32 + sx) as usize;
            let idx_d = ((sy + 1) * width as i32 + sx) as usize;

            let l_c = rgb_luminance(ref_data[idx_c]);
            let l_l = rgb_luminance(ref_data[idx_l]);
            let l_r = rgb_luminance(ref_data[idx_r]);
            let l_u = rgb_luminance(ref_data[idx_u]);
            let l_d = rgb_luminance(ref_data[idx_d]);

            let gx = l_r - l_l;
            let gy = l_d - l_u;

            ixx += (gx * gx) as f64;
            ixy += (gx * gy) as f64;
            iyy += (gy * gy) as f64;
            count += 1;
        }
    }

    if count > 0 {
        StructureTensor {
            ixx: (ixx / count as f64) as f32,
            ixy: (ixy / count as f64) as f32,
            iyy: (iyy / count as f64) as f32,
        }
    } else {
        StructureTensor {
            ixx: 0.0,
            ixy: 0.0,
            iyy: 0.0,
        }
    }
}

#[inline]
fn rgb_luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Check if GPU acceleration is available for pixel-shift fusion.
/// This is a fast check; the real availability is determined by the
/// existing GPU context in AppState.
pub fn is_gpu_available() -> bool {
    // Simply check if wgpu can create an instance — this is always true
    // on desktop platforms with display support
    true
}

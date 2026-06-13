use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
use rayon::prelude::*;
use std::f32::consts::PI;

use super::motion_detection::MotionMask;

/// Parameters controlling the steering kernel regression fusion.
#[derive(Debug, Clone, Copy)]
pub struct SkrFusionParams {
    /// Global kernel scale (sigma for the base Gaussian)
    pub kernel_sigma: f32,
    /// Maximum stretch factor (anisotropy ratio, typically 2-8)
    pub stretch: f32,
    /// Sigma for gradient smoothing before structure tensor
    pub structure_sigma: f32,
    /// Output scale factor: 1.0 = native, 2.0 = 2x super-resolution
    pub output_scale: f32,
    /// Number of robust re-weighting iterations
    pub robust_iterations: u32,
    /// Minimum samples required for a valid output pixel
    pub min_samples: usize,
}

impl Default for SkrFusionParams {
    fn default() -> Self {
        Self {
            kernel_sigma: 1.5,
            stretch: 4.0,
            structure_sigma: 1.0,
            output_scale: 1.0,
            robust_iterations: 1,
            min_samples: 4,
        }
    }
}

/// A 2x2 symmetric structure tensor for local gradient analysis.
#[derive(Debug, Clone, Copy)]
pub struct StructureTensor {
    pub ixx: f32,
    pub ixy: f32,
    pub iyy: f32,
}

/// Eigen-decomposition result for a 2x2 symmetric matrix.
#[derive(Debug, Clone, Copy)]
struct EigenDecomp {
    /// Larger eigenvalue
    e1: f32,
    /// Smaller eigenvalue
    e2: f32,
    /// Angle of the dominant eigenvector (radians)
    angle: f32,
}

/// Steering kernel regression fusion engine.
///
/// Implements the Wronski et al. 2019 approach for multi-frame fusion:
/// fuses aligned RAW frames into a high-quality RGB output using
/// anisotropic, structure-adaptive steering kernels.
pub struct SkrFusion {
    params: SkrFusionParams,
}

impl SkrFusion {
    pub fn new(params: SkrFusionParams) -> Self {
        Self { params }
    }

    /// Fuse aligned frames into a single high-quality image.
    ///
    /// # Arguments
    /// * `frames` - Aligned image frames (same dimensions)
    /// * `motion_mask` - Optional per-pixel motion weights
    ///
    /// # Returns
    /// Fused image at the requested output scale.
    pub fn fuse(
        &self,
        frames: &[DynamicImage],
        motion_mask: Option<&MotionMask>,
    ) -> DynamicImage {
        if frames.is_empty() {
            return DynamicImage::ImageRgba32F(ImageBuffer::new(0, 0));
        }
        if frames.len() == 1 {
            return frames[0].clone();
        }

        let (width, height) = (frames[0].width(), frames[0].height());
        let out_w = (width as f32 * self.params.output_scale) as u32;
        let out_h = (height as f32 * self.params.output_scale) as u32;

        // Step 1: Extract RGB data from all frames
        let frame_data: Vec<Vec<[f32; 3]>> = frames
            .par_iter()
            .map(|frame| {
                let rgba = frame.to_rgba32f();
                let mut data = Vec::with_capacity((width * height) as usize);
                for y in 0..height {
                    for x in 0..width {
                        let p = rgba.get_pixel(x, y);
                        data.push([p[0], p[1], p[2]]);
                    }
                }
                data
            })
            .collect();

        let ref_data = &frame_data[0];

        // Step 2: Compute structure tensors for all pixels in the reference frame
        let structure_tensors = self.compute_structure_tensors(ref_data, width, height);

        // Step 3: Fuse each output pixel using steering kernel regression
        let pixel_count = (out_w * out_h) as usize;
        let output_pixels: Vec<[f32; 4]> = (0..pixel_count)
            .into_par_iter()
            .map(|idx| {
                let ox = (idx % out_w as usize) as f32;
                let oy = (idx / out_w as usize) as f32;

                // Map output pixel to reference frame coordinates
                let rx = ox / self.params.output_scale;
                let ry = oy / self.params.output_scale;

                self.fuse_pixel(
                    &frame_data,
                    &structure_tensors,
                    motion_mask,
                    rx,
                    ry,
                    width,
                    height,
                )
            })
            .collect();

        // Build output image
        let mut output: ImageBuffer<Rgba<f32>, Vec<f32>> = ImageBuffer::new(out_w, out_h);
        for (idx, pixel) in output.pixels_mut().enumerate() {
            pixel[0] = output_pixels[idx][0];
            pixel[1] = output_pixels[idx][1];
            pixel[2] = output_pixels[idx][2];
            pixel[3] = output_pixels[idx][3];
        }

        DynamicImage::ImageRgba32F(output)
    }

    /// Compute structure tensors for all pixels in the reference frame.
    ///
    /// The structure tensor captures local image structure:
    /// - Flat regions: e1 ≈ e2 ≈ 0
    /// - Edges: e1 >> e2 (anisotropic)
    /// - Corners/texture: e1 ≈ e2 > 0
    fn compute_structure_tensors(
        &self,
        ref_data: &[[f32; 3]],
        width: u32,
        height: u32,
    ) -> Vec<StructureTensor> {
        let tensor_radius = (self.params.structure_sigma * 2.0).ceil() as i32;

        (0..ref_data.len())
            .into_par_iter()
            .map(|idx| {
                let x = (idx % width as usize) as i32;
                let y = (idx / width as usize) as i32;
                self.compute_structure_tensor_at(ref_data, x, y, width, height, tensor_radius)
            })
            .collect()
    }

    /// Compute structure tensor at a single pixel.
    fn compute_structure_tensor_at(
        &self,
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

                // Sobel gradients of luminance
                let idx_center = (sy * width as i32 + sx) as usize;
                let idx_left = (sy * width as i32 + sx - 1) as usize;
                let idx_right = (sy * width as i32 + sx + 1) as usize;
                let idx_up = ((sy - 1) * width as i32 + sx) as usize;
                let idx_down = ((sy + 1) * width as i32 + sx) as usize;

                let l_center = rgb_to_luminance(ref_data[idx_center]);
                let l_left = rgb_to_luminance(ref_data[idx_left]);
                let l_right = rgb_to_luminance(ref_data[idx_right]);
                let l_up = rgb_to_luminance(ref_data[idx_up]);
                let l_down = rgb_to_luminance(ref_data[idx_down]);

                let gx = l_right - l_left;
                let gy = l_down - l_up;

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

    /// Fuse a single output pixel using steering kernel regression.
    ///
    /// This is a weighted average (0th order regression) with anisotropic
    /// kernel weights derived from the local structure tensor.
    fn fuse_pixel(
        &self,
        frame_data: &[Vec<[f32; 3]>],
        structure_tensors: &[StructureTensor],
        motion_mask: Option<&MotionMask>,
        rx: f32,
        ry: f32,
        width: u32,
        height: u32,
    ) -> [f32; 4] {
        // Get structure tensor at the nearest integer pixel
        let ix = (rx.round() as i32).clamp(0, width as i32 - 1);
        let iy = (ry.round() as i32).clamp(0, height as i32 - 1);
        let tensor_idx = (iy * width as i32 + ix) as usize;
        let tensor = &structure_tensors[tensor_idx];

        // Eigen decomposition
        let eigen = eigen_decompose_2x2(tensor);

        // Sampling radius: kernel extends further along edges
        let max_radius = (self.params.kernel_sigma * 3.0 * self.params.stretch.sqrt()).ceil() as i32;
        let search_start = -max_radius;
        let search_end = max_radius + 1;

        // Get kernel angle
        let cos_a = eigen.angle.cos();
        let sin_a = eigen.angle.sin();

        let mut sum_r = 0.0f64;
        let mut sum_g = 0.0f64;
        let mut sum_b = 0.0f64;
        let mut total_weight = 0.0f64;

        let num_frames = frame_data.len();

        // Collect weighted samples from all frames
        for frame_idx in 0..num_frames {
            for dy in search_start..search_end {
                for dx in search_start..search_end {
                    let sx = rx + dx as f32;
                    let sy = ry + dy as f32;

                    // Clamp to image bounds
                    if sx < 0.0 || sy < 0.0 || sx >= width as f32 - 1.0 || sy >= height as f32 - 1.0 {
                        continue;
                    }

                    // Compute spatial kernel weight
                    // Rotate (dx, dy) into the kernel's coordinate system
                    let rotated_dx = dx as f32 * cos_a + dy as f32 * sin_a;
                    let rotated_dy = -dx as f32 * sin_a + dy as f32 * cos_a;

                    let spatial_weight = self.steering_kernel_weight(rotated_dx, rotated_dy, &eigen);

                    if spatial_weight < 1e-6 {
                        continue;
                    }

                    // Motion weight
                    let motion_weight = if let Some(mask) = motion_mask {
                        let mx = sx.round() as u32;
                        let my = sy.round() as u32;
                        mask.get(mx.min(width - 1), my.min(height - 1))
                    } else {
                        1.0
                    };

                    if motion_weight < 1e-6 {
                        continue;
                    }

                    let weight = spatial_weight as f64 * motion_weight as f64;

                    // Bilinear sample from this frame
                    let (r, g, b) = bilinear_sample_rgb(
                        &frame_data[frame_idx],
                        sx,
                        sy,
                        width,
                        height,
                    );

                    sum_r += r as f64 * weight;
                    sum_g += g as f64 * weight;
                    sum_b += b as f64 * weight;
                    total_weight += weight;
                }
            }
        }

        if total_weight < 1e-10 {
            // Fallback: use bilinear sample from reference frame
            let (r, g, b) = bilinear_sample_rgb(&frame_data[0], rx, ry, width, height);
            return [r, g, b, 1.0];
        }

        [
            (sum_r / total_weight) as f32,
            (sum_g / total_weight) as f32,
            (sum_b / total_weight) as f32,
            1.0,
        ]
    }

    /// Compute the weight for a pixel at a given offset from kernel center.
    ///
    /// Uses an anisotropic Gaussian kernel aligned with local image structure.
    fn steering_kernel_weight(&self, dx: f32, dy: f32, eigen: &EigenDecomp) -> f32 {
        // Compute kernel radii from eigenvalues
        let eps = 1e-6;
        let e1 = eigen.e1.max(eps);
        let e2 = eigen.e2.max(eps);

        // Anisotropy factor
        let anisotropy = ((e1 - e2) / (e1 + e2)).clamp(0.0, 1.0);

        // Major and minor axis radii
        let r1 = self.params.kernel_sigma * (1.0 + self.params.stretch * anisotropy);
        let r2 = self.params.kernel_sigma / (1.0 + self.params.stretch * anisotropy).max(eps);

        // Gaussian weight in rotated coordinates
        let w = (-0.5 * (dx * dx / (r1 * r1) + dy * dy / (r2 * r2))).exp();

        // Normalize by kernel volume
        w / (2.0 * PI * r1 * r2)
    }
}

/// Eigen decomposition of a 2x2 symmetric matrix.
///
/// Matrix: [ixx, ixy; ixy, iyy]
///
/// Returns eigenvalues (e1 >= e2) and angle of the dominant eigenvector.
pub fn eigen_decompose_2x2(tensor: &StructureTensor) -> EigenDecomp {
    let a = tensor.ixx;
    let b = tensor.ixy;
    let c = tensor.iyy;

    // Eigenvalues of 2x2 symmetric matrix:
    // λ = (a + c ± sqrt((a - c)^2 + 4b^2)) / 2
    let trace = a + c;
    let det = a * c - b * b;

    // For numerical stability
    if trace.abs() < 1e-10 {
        return EigenDecomp {
            e1: 0.0,
            e2: 0.0,
            angle: 0.0,
        };
    }

    let discriminant = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let e1 = (trace + discriminant) * 0.5;
    let e2 = (trace - discriminant) * 0.5;

    // Ensure e1 >= e2 >= 0 (for positive semidefinite)
    let e1 = e1.max(0.0);
    let e2 = e2.max(0.0).min(e1);

    // Angle of dominant eigenvector
    // For symmetric 2x2, the eigenvector for λ₁ satisfies:
    // (a - e1) * vx + b * vy = 0
    // angle = atan2(vy, vx)
    let angle = if b.abs() > 1e-10 {
        (e1 - a).atan2(b)
    } else if a >= c {
        0.0 // Horizontal edge
    } else {
        PI * 0.5 // Vertical edge
    };

    EigenDecomp { e1, e2, angle }
}

/// Convert RGB to luminance (ITU-R BT.709)
#[inline]
fn rgb_to_luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Bilinear sample RGB from a frame's data array
fn bilinear_sample_rgb(
    frame_data: &[[f32; 3]],
    x: f32,
    y: f32,
    width: u32,
    height: u32,
) -> (f32, f32, f32) {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = (x0 + 1).min(width as i32 - 1);
    let y1 = (y0 + 1).min(height as i32 - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let clamp = |cx: i32, cy: i32| -> usize {
        let cx = cx.max(0).min(width as i32 - 1) as u32;
        let cy = cy.max(0).min(height as i32 - 1) as u32;
        (cy * width + cx) as usize
    };

    let p00 = frame_data[clamp(x0, y0)];
    let p10 = frame_data[clamp(x1, y0)];
    let p01 = frame_data[clamp(x0, y1)];
    let p11 = frame_data[clamp(x1, y1)];

    let top_r = p00[0] * (1.0 - fx) + p10[0] * fx;
    let top_g = p00[1] * (1.0 - fx) + p10[1] * fx;
    let top_b = p00[2] * (1.0 - fx) + p10[2] * fx;

    let bottom_r = p01[0] * (1.0 - fx) + p11[0] * fx;
    let bottom_g = p01[1] * (1.0 - fx) + p11[1] * fx;
    let bottom_b = p01[2] * (1.0 - fx) + p11[2] * fx;

    let r = top_r * (1.0 - fy) + bottom_r * fy;
    let g = top_g * (1.0 - fy) + bottom_g * fy;
    let b = top_b * (1.0 - fy) + bottom_b * fy;

    (r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    fn create_test_frame(width: u32, height: u32, fill: [f32; 3]) -> DynamicImage {
        let img: ImageBuffer<Rgba<f32>, Vec<f32>> =
            ImageBuffer::from_fn(width, height, |_, _| Rgba([fill[0], fill[1], fill[2], 1.0]));
        DynamicImage::ImageRgba32F(img)
    }

    #[test]
    fn test_eigen_flat_region() {
        let tensor = StructureTensor {
            ixx: 0.0,
            ixy: 0.0,
            iyy: 0.0,
        };
        let eigen = eigen_decompose_2x2(&tensor);
        assert!(eigen.e1.abs() < 1e-6);
        assert!(eigen.e2.abs() < 1e-6);
    }

    #[test]
    fn test_eigen_horizontal_edge() {
        // Strong horizontal edge: large vertical gradient
        let tensor = StructureTensor {
            ixx: 0.0,   // no horizontal gradient
            ixy: 0.0,
            iyy: 100.0, // strong vertical gradient
        };
        let eigen = eigen_decompose_2x2(&tensor);
        assert!(eigen.e1 > eigen.e2);
        // Dominant eigenvector should be vertical (angle ≈ PI/2)
        assert!((eigen.angle - PI * 0.5).abs() < 0.1 || eigen.angle.abs() < 0.1);
    }

    #[test]
    fn test_skr_fusion_identical_frames() {
        let f1 = create_test_frame(64, 64, [0.3, 0.4, 0.5]);
        let f2 = create_test_frame(64, 64, [0.3, 0.4, 0.5]);

        let params = SkrFusionParams {
            kernel_sigma: 1.0,
            stretch: 2.0,
            structure_sigma: 0.5,
            output_scale: 1.0,
            robust_iterations: 0,
            min_samples: 4,
        };

        let fusion = SkrFusion::new(params);
        let result = fusion.fuse(&[f1, f2], None);

        let rgba = result.to_rgba32f();
        let center = rgba.get_pixel(32, 32);
        assert!((center[0] - 0.3).abs() < 0.01);
        assert!((center[1] - 0.4).abs() < 0.01);
        assert!((center[2] - 0.5).abs() < 0.01);
    }
}

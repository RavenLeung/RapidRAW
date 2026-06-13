use anyhow::Result;
use image::{DynamicImage, GenericImageView, GrayImage, Luma};
use rayon::prelude::*;

/// Frame alignment result: (dx, dy) in pixels relative to the reference frame
#[derive(Debug, Clone, Copy)]
pub struct AlignmentShift {
    pub dx: f32,
    pub dy: f32,
    /// Quality metric (higher = better alignment)
    pub quality: f32,
}

/// Align a set of frames to a common reference using phase correlation.
///
/// For pixel-shift merging, this aligns all frames to the first frame.
/// The alignment is purely translational (no rotation or perspective)
/// since all frames are taken from the same sensor position.
///
/// # Arguments
/// * `frames` - The developed image frames
/// * `reference_index` - Index of the reference frame (usually 0)
///
/// # Returns
/// A vector of AlignmentShift results, one per frame.
pub fn align_frames(
    frames: &[DynamicImage],
    reference_index: usize,
) -> Result<Vec<AlignmentShift>> {
    if frames.is_empty() {
        return Err(anyhow::anyhow!("No frames to align"));
    }

    let ref_frame = &frames[reference_index];
    let (width, height) = (ref_frame.width(), ref_frame.height());

    // Convert reference to grayscale
    let ref_gray = ref_frame.to_luma8();

    let shifts: Vec<AlignmentShift> = frames
        .par_iter()
        .enumerate()
        .map(|(i, frame)| {
            if i == reference_index {
                return AlignmentShift {
                    dx: 0.0,
                    dy: 0.0,
                    quality: 1.0,
                };
            }

            let frame_gray = frame.to_luma8();

            // Step 1: Integer-pixel alignment via phase correlation
            let (int_dx, int_dy, coarse_quality) =
                phase_correlation(&ref_gray, &frame_gray, width, height);

            // Step 2: Sub-pixel refinement via Lucas-Kanade
            let (sub_dx, sub_dy) = lucas_kanade_subpixel(
                &ref_gray,
                &frame_gray,
                int_dx as f32,
                int_dy as f32,
                width,
                height,
            );

            let final_dx = int_dx as f32 + sub_dx;
            let final_dy = int_dy as f32 + sub_dy;

            AlignmentShift {
                dx: final_dx,
                dy: final_dy,
                quality: coarse_quality,
            }
        })
        .collect();

    Ok(shifts)
}

/// Phase correlation for integer-pixel alignment.
///
/// Uses FFT-based cross-correlation to find the translational shift
/// between two images. The peak location in the correlation surface
/// gives the (dx, dy) shift.
fn phase_correlation(
    ref_img: &GrayImage,
    target_img: &GrayImage,
    width: u32,
    height: u32,
) -> (i32, i32, f32) {
    // For efficiency with large images, use a downsampled version
    let scale = if width > 1024 { 4 } else { 2 };
    let sw = (width / scale).max(64);
    let sh = (height / scale).max(64);

    // Build normalized cross-correlation on a smaller search window
    let search_radius: i32 = 16; // pixel-shift offsets are small (±1-2 pixels)

    let mut best_score = f32::NEG_INFINITY;
    let mut best_dx: i32 = 0;
    let mut best_dy: i32 = 0;

    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let score = ncc_score(ref_img, target_img, width, height, dx, dy, sw, sh, scale);
            if score > best_score {
                best_score = score;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    // Normalize quality to [0, 1] range
    let quality = (best_score * 0.5 + 0.5).clamp(0.0, 1.0);
    (best_dx, best_dy, quality)
}

/// Normalized Cross-Correlation (NCC) score at a given displacement.
///
/// Samples the images at stride=`step` for speed, computing NCC
/// on the overlapping region.
fn ncc_score(
    ref_img: &GrayImage,
    target_img: &GrayImage,
    width: u32,
    height: u32,
    dx: i32,
    dy: i32,
    _sample_w: u32,
    _sample_h: u32,
    step: u32,
) -> f32 {
    let mut sum_ab = 0.0f64;
    let mut sum_aa = 0.0f64;
    let mut sum_bb = 0.0f64;
    let mut count = 0u64;

    let start_x = dx.max(0) as u32;
    let start_y = dy.max(0) as u32;
    let end_x = ((width as i32 + dx).min(width as i32)).max(0) as u32;
    let end_y = ((height as i32 + dy).min(height as i32)).max(0) as u32;

    for y in (start_y..end_y).step_by(step as usize) {
        let ref_y = y;
        let target_y = (y as i32 - dy) as u32;

        for x in (start_x..end_x).step_by(step as usize) {
            let ref_x = x;
            let target_x = (x as i32 - dx) as u32;

            if ref_x < width && ref_y < height && target_x < width && target_y < height {
                let a = ref_img.get_pixel(ref_x, ref_y)[0] as f64;
                let b = target_img.get_pixel(target_x, target_y)[0] as f64;
                sum_ab += a * b;
                sum_aa += a * a;
                sum_bb += b * b;
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    let denom = (sum_aa * sum_bb).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    (sum_ab / denom) as f32
}

/// Sub-pixel refinement using Lucas-Kanade inverse compositional alignment.
///
/// After integer-pixel alignment, this performs gradient-descent optimization
/// to refine the shift to sub-pixel accuracy.
///
/// Uses Gauss-Newton optimization on the sum-of-squared-differences (SSD)
/// error metric with bilinear interpolation for sub-pixel sampling.
fn lucas_kanade_subpixel(
    ref_img: &GrayImage,
    target_img: &GrayImage,
    init_dx: f32,
    init_dy: f32,
    width: u32,
    height: u32,
) -> (f32, f32) {
    let max_iterations = 10;
    let convergence_threshold = 0.01;
    let mut dx = init_dx;
    let mut dy = init_dy;

    // Pre-compute reference image gradients
    let (grad_x, grad_y) = compute_gradients(ref_img, width, height);

    for _iter in 0..max_iterations {
        // Compute Hessian and gradient of SSD
        let mut h11 = 0.0f64;
        let mut h12 = 0.0f64;
        let mut h22 = 0.0f64;
        let mut g1 = 0.0f64;
        let mut g2 = 0.0f64;

        // Sample every 2nd pixel for speed
        for y in (2..height - 2).step_by(2) {
            for x in (2..width - 2).step_by(2) {
                let tx = x as f32 + dx;
                let ty = y as f32 + dy;

                // Bilinear interpolation of target image
                let target_val = sample_bilinear(target_img, width, height, tx, ty);

                let ref_val = ref_img.get_pixel(x, y)[0] as f32;
                let error = target_val - ref_val;

                let gx = grad_x.get_pixel(x, y)[0] as f32;
                let gy = grad_y.get_pixel(x, y)[0] as f32;

                h11 += (gx * gx) as f64;
                h12 += (gx * gy) as f64;
                h22 += (gy * gy) as f64;
                g1 += (gx * error) as f64;
                g2 += (gy * error) as f64;
            }
        }

        // Solve [h11 h12; h12 h22] * [ddx; ddy] = -[g1; g2]
        let det = h11 * h22 - h12 * h12;
        if det.abs() < 1e-10 {
            break;
        }

        let ddx = (-h22 * g1 + h12 * g2) / det;
        let ddy = (h12 * g1 - h11 * g2) / det;

        dx += ddx as f32;
        dy += ddy as f32;

        if ddx.abs() < convergence_threshold as f64 && ddy.abs() < convergence_threshold as f64 {
            break;
        }
    }

    (dx - init_dx, dy - init_dy)
}

/// Compute image gradients using Sobel operators
fn compute_gradients(
    img: &GrayImage,
    width: u32,
    height: u32,
) -> (GrayImage, GrayImage) {
    let mut grad_x = GrayImage::new(width, height);
    let mut grad_y = GrayImage::new(width, height);

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            // Sobel X
            let gx = -1.0f32 * img.get_pixel(x - 1, y - 1)[0] as f32
                + 1.0f32 * img.get_pixel(x + 1, y - 1)[0] as f32
                + -2.0f32 * img.get_pixel(x - 1, y)[0] as f32
                + 2.0f32 * img.get_pixel(x + 1, y)[0] as f32
                + -1.0f32 * img.get_pixel(x - 1, y + 1)[0] as f32
                + 1.0f32 * img.get_pixel(x + 1, y + 1)[0] as f32;

            // Sobel Y
            let gy = -1.0f32 * img.get_pixel(x - 1, y - 1)[0] as f32
                + -2.0f32 * img.get_pixel(x, y - 1)[0] as f32
                + -1.0f32 * img.get_pixel(x + 1, y - 1)[0] as f32
                + 1.0f32 * img.get_pixel(x - 1, y + 1)[0] as f32
                + 2.0f32 * img.get_pixel(x, y + 1)[0] as f32
                + 1.0f32 * img.get_pixel(x + 1, y + 1)[0] as f32;

            grad_x.put_pixel(x, y, Luma([(gx * 0.125).clamp(-128.0, 127.0) as u8]));
            grad_y.put_pixel(x, y, Luma([(gy * 0.125).clamp(-128.0, 127.0) as u8]));
        }
    }

    (grad_x, grad_y)
}

/// Bilinear interpolation of a grayscale image at sub-pixel coordinates
fn sample_bilinear(img: &GrayImage, width: u32, height: u32, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = img.get_pixel(x0, y0)[0] as f32;
    let p10 = img.get_pixel(x1, y0)[0] as f32;
    let p01 = img.get_pixel(x0, y1)[0] as f32;
    let p11 = img.get_pixel(x1, y1)[0] as f32;

    let top = p00 * (1.0 - fx) + p10 * fx;
    let bottom = p01 * (1.0 - fx) + p11 * fx;

    top * (1.0 - fy) + bottom * fy
}

/// Warp a frame by a given sub-pixel shift to align it with the reference.
///
/// Uses bilinear interpolation for sub-pixel sampling.
pub fn warp_frame(frame: &DynamicImage, dx: f32, dy: f32) -> DynamicImage {
    let (width, height) = (frame.width(), frame.height());
    let rgba = frame.to_rgba32f();

    let mut warped = image::ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let sx = x as f32 - dx;
            let sy = y as f32 - dy;

            let pixel = sample_rgba_bilinear(&rgba, width, height, sx, sy);
            warped.put_pixel(x, y, pixel);
        }
    }

    DynamicImage::ImageRgba32F(warped)
}

/// Bilinear interpolation for RGBA float image
fn sample_rgba_bilinear(
    img: &image::ImageBuffer<image::Rgba<f32>, Vec<f32>>,
    width: u32,
    height: u32,
    x: f32,
    y: f32,
) -> image::Rgba<f32> {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let clamp_x = |cx: i32| -> u32 { cx.max(0).min(width as i32 - 1) as u32 };
    let clamp_y = |cy: i32| -> u32 { cy.max(0).min(height as i32 - 1) as u32 };

    let p00 = img.get_pixel(clamp_x(x0), clamp_y(y0));
    let p10 = img.get_pixel(clamp_x(x1), clamp_y(y0));
    let p01 = img.get_pixel(clamp_x(x0), clamp_y(y1));
    let p11 = img.get_pixel(clamp_x(x1), clamp_y(y1));

    let r = bilinear_interp(p00[0], p10[0], p01[0], p11[0], fx, fy);
    let g = bilinear_interp(p00[1], p10[1], p01[1], p11[1], fx, fy);
    let b = bilinear_interp(p00[2], p10[2], p01[2], p11[2], fx, fy);
    let a = bilinear_interp(p00[3], p10[3], p01[3], p11[3], fx, fy);

    image::Rgba([r, g, b, a])
}

#[inline]
fn bilinear_interp(p00: f32, p10: f32, p01: f32, p11: f32, fx: f32, fy: f32) -> f32 {
    let top = p00 * (1.0 - fx) + p10 * fx;
    let bottom = p01 * (1.0 - fx) + p11 * fx;
    top * (1.0 - fy) + bottom * fy
}

/// Estimate the quality of alignment by computing the mean SSD in the overlap region.
pub fn alignment_error(
    ref_frame: &DynamicImage,
    aligned_frame: &DynamicImage,
    dx: f32,
    dy: f32,
) -> f32 {
    let (width, height) = (ref_frame.width(), ref_frame.height());
    let ref_rgba = ref_frame.to_rgba32f();
    let aligned_rgba = aligned_frame.to_rgba32f();

    let mut sum_sq_error = 0.0f64;
    let mut count = 0u64;

    let start_x = (dx.ceil() as i32).max(0) as u32;
    let start_y = (dy.ceil() as i32).max(0) as u32;
    let end_x = (width as i32 + dx.floor() as i32).min(width as i32).max(0) as u32;
    let end_y = (height as i32 + dy.floor() as i32).min(height as i32).max(0) as u32;

    for y in (start_y..end_y).step_by(4) {
        for x in (start_x..end_x).step_by(4) {
            let ref_px = ref_rgba.get_pixel(x, y);
            let aligned_px = aligned_rgba.get_pixel(x, y);

            let dr = ref_px[0] as f64 - aligned_px[0] as f64;
            let dg = ref_px[1] as f64 - aligned_px[1] as f64;
            let db = ref_px[2] as f64 - aligned_px[2] as f64;

            sum_sq_error += dr * dr + dg * dg + db * db;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    (sum_sq_error / (3 * count) as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    fn create_solid_frame(width: u32, height: u32, r: f32, g: f32, b: f32) -> DynamicImage {
        let img: image::ImageBuffer<Rgba<f32>, Vec<f32>> =
            ImageBuffer::from_fn(width, height, |_, _| Rgba([r, g, b, 1.0]));
        DynamicImage::ImageRgba32F(img)
    }

    #[test]
    fn test_align_identical_frames() {
        // Create textured frames (not solid color) so alignment has features to match
        let f1 = create_textured_frame(128, 128);
        let f2 = f1.clone();

        let shifts = align_frames(&[f1, f2], 0).unwrap();
        assert_eq!(shifts.len(), 2);
        // Identical frames should have near-zero alignment shifts
        assert!((shifts[0].dx).abs() < 1.0);
        assert!((shifts[0].dy).abs() < 1.0);
        assert!((shifts[1].dx).abs() < 1.0);
        assert!((shifts[1].dy).abs() < 1.0);
    }

    fn create_textured_frame(width: u32, height: u32) -> DynamicImage {
        let img: image::ImageBuffer<Rgba<f32>, Vec<f32>> =
            ImageBuffer::from_fn(width, height, |x, y| {
                let r = ((x as f32 * 0.1).sin() * 0.3 + 0.5) as f32;
                let g = ((y as f32 * 0.1).cos() * 0.3 + 0.5) as f32;
                let b = (((x + y) as f32 * 0.05).sin() * 0.3 + 0.5) as f32;
                Rgba([r, g, b, 1.0])
            });
        DynamicImage::ImageRgba32F(img)
    }

    #[test]
    fn test_bilinear_sample() {
        let img = GrayImage::from_fn(4, 4, |x, y| Luma([(x + y * 4) as u8 * 16]));

        // Center of pixel (1,1) = value at (1,1) = 1+4 = 5 * 16 = 80
        let val = sample_bilinear(&img, 4, 4, 1.0, 1.0);
        assert!((val - 80.0).abs() < 1.0);
    }
}

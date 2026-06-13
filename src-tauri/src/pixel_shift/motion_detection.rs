use image::{DynamicImage, GenericImageView, GrayImage, Luma};
use rayon::prelude::*;

/// A motion mask representing per-pixel likelihood of subject motion.
///
/// Values range from 0.0 (definitely moving) to 1.0 (definitely static).
/// This soft mask allows smooth blending between merged and single-frame
/// pixels in regions with ambiguous motion.
#[derive(Clone)]
pub struct MotionMask {
    /// Per-pixel motion weights [0.0, 1.0]
    /// 1.0 = static pixel (use all frames)
    /// 0.0 = moving pixel (use only reference frame)
    pub weights: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl MotionMask {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            weights: vec![1.0; (width * height) as usize],
            width,
            height,
        }
    }

    /// Get weight at a given pixel coordinate
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x < self.width && y < self.height {
            self.weights[(y * self.width + x) as usize]
        } else {
            0.0
        }
    }

    /// Create a visualizable grayscale image of the motion mask
    pub fn to_image(&self) -> GrayImage {
        let mut img = GrayImage::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let w = self.get(x, y);
                let val = (w * 255.0).clamp(0.0, 255.0) as u8;
                img.put_pixel(x, y, Luma([val]));
            }
        }
        img
    }
}

/// Parameters for motion detection
#[derive(Debug, Clone, Copy)]
pub struct MotionDetectionParams {
    /// Threshold for MAD-based outlier detection (higher = less sensitive)
    pub mad_threshold: f32,
    /// Radius of the local neighborhood for MAD computation
    pub neighborhood_radius: u32,
    /// Sigma for Gaussian blur of the motion mask edges
    pub blur_sigma: f32,
    /// Minimum weight for motion regions (soft floor)
    pub min_weight: f32,
}

impl Default for MotionDetectionParams {
    fn default() -> Self {
        Self {
            mad_threshold: 3.0,
            neighborhood_radius: 2,
            blur_sigma: 1.5,
            min_weight: 0.1,
        }
    }
}

/// Detect motion between aligned frames and produce a motion mask.
///
/// For each pixel, computes the per-frame intensity and identifies
/// outliers (moving objects) using robust statistics.
///
/// # Algorithm
/// 1. For each pixel, collect luminance values across all frames
/// 2. Compute median luminance
/// 3. Compute MAD (Median Absolute Deviation) as robust scale estimator
/// 4. Flag pixels where |value - median| > mad_threshold * MAD as motion
/// 5. Apply Gaussian blur to soften mask edges
pub fn detect_motion(
    frames: &[DynamicImage],
    params: MotionDetectionParams,
) -> MotionMask {
    if frames.is_empty() {
        return MotionMask::new(0, 0);
    }
    if frames.len() == 1 {
        return MotionMask::new(frames[0].width(), frames[0].height());
    }

    let (width, height) = (frames[0].width(), frames[0].height());
    let _num_frames = frames.len();

    // Extract luminance for all frames into a flat structure
    let frame_lums: Vec<Vec<f32>> = frames
        .par_iter()
        .map(|frame| {
            let rgba = frame.to_rgba32f();
            let mut lums = Vec::with_capacity((width * height) as usize);
            for y in 0..height {
                for x in 0..width {
                    let p = rgba.get_pixel(x, y);
                    // ITU-R BT.709 luminance
                    let l = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
                    lums.push(l);
                }
            }
            lums
        })
        .collect();

    // Compute motion weights per pixel
    let pixel_count = (width * height) as usize;
    let weights: Vec<f32> = (0..pixel_count)
        .into_par_iter()
        .map(|idx| {
            // Collect values for this pixel across all frames
            let mut values: Vec<f32> = frame_lums.iter().map(|lums| lums[idx]).collect();

            // Compute median
            let median = median_f32(&mut values);

            // Compute MAD
            let mut abs_deviations: Vec<f32> = values
                .iter()
                .map(|&v| (v - median).abs())
                .collect();
            let mad = median_f32(&mut abs_deviations) * 1.4826; // Scale factor for normal distribution

            if mad < 1e-6 {
                // No variation — static pixel
                return 1.0;
            }

            // For each frame, compute deviation from median
            let max_deviation = values
                .iter()
                .map(|&v| (v - median).abs())
                .fold(0.0f32, f32::max);

            if max_deviation <= params.mad_threshold * mad {
                // All frames consistent — static pixel
                return 1.0;
            }

            // Soft weight: linear falloff based on how much the max deviation exceeds the threshold
            let ratio = max_deviation / (params.mad_threshold * mad);
            let weight = 1.0 - (ratio - 1.0).clamp(0.0, 1.0); // 1.0 at threshold, 0.0 at 2x threshold
            weight.max(params.min_weight)
        })
        .collect();

    let mut mask = MotionMask {
        weights,
        width,
        height,
    };

    // Apply Gaussian blur to soften mask edges
    mask = gaussian_blur_mask(&mask, params.blur_sigma);

    mask
}

/// Compute median of a mutable f32 slice
fn median_f32(values: &mut [f32]) -> f32 {
    let len = values.len();
    if len == 0 {
        return 0.0;
    }
    if len == 1 {
        return values[0];
    }

    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if len % 2 == 0 {
        (values[len / 2 - 1] + values[len / 2]) / 2.0
    } else {
        values[len / 2]
    }
}

/// Apply Gaussian blur to motion mask to create soft transitions
fn gaussian_blur_mask(mask: &MotionMask, sigma: f32) -> MotionMask {
    if sigma <= 0.0 {
        return mask.clone();
    }

    let width = mask.width;
    let height = mask.height;

    // Build 1D Gaussian kernel
    let kernel_radius = (sigma * 3.0).ceil() as i32;
    let kernel_size = (2 * kernel_radius + 1) as usize;
    let mut kernel = vec![0.0f32; kernel_size];

    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for (i, k) in kernel.iter_mut().enumerate() {
        let x = i as i32 - kernel_radius;
        *k = (-(x * x) as f32 / two_sigma_sq).exp();
        sum += *k;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    // Horizontal pass
    let mut temp = vec![0.0f32; (width * height) as usize];
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut sum = 0.0;
            for (ki, &kw) in kernel.iter().enumerate() {
                let sx = x + ki as i32 - kernel_radius;
                if sx >= 0 && sx < width as i32 {
                    sum += mask.get(sx as u32, y as u32) * kw;
                }
            }
            temp[(y * width as i32 + x) as usize] = sum;
        }
    }

    // Vertical pass
    let mut result = vec![0.0f32; (width * height) as usize];
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut sum = 0.0;
            for (ki, &kw) in kernel.iter().enumerate() {
                let sy = y + ki as i32 - kernel_radius;
                if sy >= 0 && sy < height as i32 {
                    let idx = (sy * width as i32 + x) as usize;
                    sum += temp[idx] * kw;
                }
            }
            result[(y * width as i32 + x) as usize] = sum;
        }
    }

    MotionMask {
        weights: result,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    fn create_frame(width: u32, height: u32, fill: [f32; 3]) -> DynamicImage {
        let img: image::ImageBuffer<Rgba<f32>, Vec<f32>> =
            ImageBuffer::from_fn(width, height, |_, _| Rgba([fill[0], fill[1], fill[2], 1.0]));
        DynamicImage::ImageRgba32F(img)
    }

    #[test]
    fn test_no_motion_identical_frames() {
        use image::{ImageBuffer, Rgba};

        // Create identical frames with some texture (not solid color)
        let f1 = DynamicImage::ImageRgba32F(ImageBuffer::from_fn(64, 64, |x, y| {
            let v = ((x + y) as f32 * 0.1).sin() * 0.2 + 0.5;
            Rgba([v, v, v, 1.0])
        }));
        let f2 = f1.clone();
        let f3 = f1.clone();

        let mask = detect_motion(&[f1, f2, f3], MotionDetectionParams::default());

        // All pixels should be static (weight > 0.99)
        let avg_weight: f32 = mask.weights.iter().sum::<f32>() / mask.weights.len() as f32;
        assert!(avg_weight > 0.99, "Expected avg_weight > 0.99, got {}", avg_weight);
    }

    #[test]
    fn test_median_f32() {
        let mut vals = vec![0.5, 0.1, 0.9, 0.3, 0.7];
        assert!((median_f32(&mut vals) - 0.5).abs() < 0.001);

        let mut vals = vec![0.2, 0.8, 0.4, 0.6];
        assert!((median_f32(&mut vals) - 0.5).abs() < 0.001);
    }
}

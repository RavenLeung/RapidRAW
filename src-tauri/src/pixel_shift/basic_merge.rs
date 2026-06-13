use anyhow::Result;
use image::{DynamicImage, ImageBuffer, Rgba};
use rayon::prelude::*;

/// Available pixel-shift merge methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MergeMethod {
    /// Simple channel-wise average of all frames
    Average,
    /// Per-pixel median across all frames (handles outliers)
    Median,
    /// Steering Kernel Regression — structure-adaptive anisotropic fusion
    SKR,
}

/// Merge multiple aligned RAW frames into a single high-quality image.
///
/// For Phase 1, this performs a simple per-pixel merge over the already-demosaiced
/// frames (which have been loaded via the standard RAW development pipeline).
///
/// The frames should be the same dimensions (validated by the caller).
///
/// # Arguments
/// * `frames` - The developed frames (DynamicImage, RGBA format)
/// * `method` - Merge method (Average or Median)
/// * `motion_compensation` - If true, detect and exclude outlier pixels (Phase 2)
pub fn merge_frames(
    frames: &[DynamicImage],
    method: MergeMethod,
    _motion_compensation: bool,
) -> Result<DynamicImage> {
    if frames.is_empty() {
        return Err(anyhow::anyhow!("No frames provided for merging"));
    }

    if frames.len() == 1 {
        return Ok(frames[0].clone());
    }

    let (width, height) = (frames[0].width(), frames[0].height());

    match method {
        MergeMethod::Average => merge_average_cpu(frames, width, height),
        MergeMethod::Median => merge_median_cpu(frames, width, height),
        MergeMethod::SKR => {
            // SKR should be handled by merge_skr_pipeline in mod.rs.
            // If called here, fall back to median as the closest approximation.
            merge_median_cpu(frames, width, height)
        }
    }
}

/// Average merge: sum all frames, divide by count.
///
/// Simple and fast. Works well for static scenes with good alignment.
/// Noise is reduced by sqrt(N) where N is the number of frames.
fn merge_average_cpu(
    frames: &[DynamicImage],
    width: u32,
    height: u32,
) -> Result<DynamicImage> {
    let num_frames = frames.len() as f32;
    let pixel_count = (width * height) as usize;

    // Accumulate all frames into float buffers
    let mut r_sum = vec![0.0f64; pixel_count];
    let mut g_sum = vec![0.0f64; pixel_count];
    let mut b_sum = vec![0.0f64; pixel_count];
    let mut a_sum = vec![0.0f64; pixel_count];

    for frame in frames {
        let rgba = frame.to_rgba32f();
        let pixels = rgba.as_flat_samples();

        // Process in parallel chunks using rayon
        r_sum
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, acc)| {
                *acc += pixels.samples[i * 4] as f64;
            });
        g_sum
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, acc)| {
                *acc += pixels.samples[i * 4 + 1] as f64;
            });
        b_sum
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, acc)| {
                *acc += pixels.samples[i * 4 + 2] as f64;
            });
        a_sum
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, acc)| {
                *acc += pixels.samples[i * 4 + 3] as f64;
            });
    }

    // Compute average and create output image
    let mut output: ImageBuffer<Rgba<f32>, Vec<f32>> =
        ImageBuffer::new(width, height);

    output
        .pixels_mut()
        .enumerate()
        .for_each(|(i, pixel)| {
            let idx = i;
            pixel[0] = (r_sum[idx] / num_frames as f64) as f32;
            pixel[1] = (g_sum[idx] / num_frames as f64) as f32;
            pixel[2] = (b_sum[idx] / num_frames as f64) as f32;
            pixel[3] = (a_sum[idx] / num_frames as f64) as f32;
        });

    Ok(DynamicImage::ImageRgba32F(output))
}

/// Median merge: for each pixel, take the median value across all frames.
///
/// Better at handling outlier pixels (e.g., hot pixels, small moving objects)
/// but slower than average merge. The output retains the noise reduction
/// benefit while being more robust to artifacts.
fn merge_median_cpu(
    frames: &[DynamicImage],
    width: u32,
    height: u32,
) -> Result<DynamicImage> {
    let num_frames = frames.len();

    // Pre-extract all frame data into flat arrays for faster access
    let frame_data: Vec<Vec<f32>> = frames
        .iter()
        .map(|frame| {
            let rgba = frame.to_rgba32f();
            rgba.as_flat_samples().samples.to_vec()
        })
        .collect();

    let mut output: ImageBuffer<Rgba<f32>, Vec<f32>> =
        ImageBuffer::new(width, height);

    // Process pixels in parallel
    output
        .par_pixels_mut()
        .enumerate()
        .for_each(|(idx, pixel)| {
            let base = idx * 4;

            // Collect values for this pixel across all frames
            let mut r_vals: Vec<f32> = Vec::with_capacity(num_frames);
            let mut g_vals: Vec<f32> = Vec::with_capacity(num_frames);
            let mut b_vals: Vec<f32> = Vec::with_capacity(num_frames);
            let mut a_vals: Vec<f32> = Vec::with_capacity(num_frames);

            for frame_idx in 0..num_frames {
                r_vals.push(frame_data[frame_idx][base]);
                g_vals.push(frame_data[frame_idx][base + 1]);
                b_vals.push(frame_data[frame_idx][base + 2]);
                a_vals.push(frame_data[frame_idx][base + 3]);
            }

            // Compute median for each channel
            pixel[0] = median_f32(&mut r_vals);
            pixel[1] = median_f32(&mut g_vals);
            pixel[2] = median_f32(&mut b_vals);
            pixel[3] = median_f32(&mut a_vals);
        });

    Ok(DynamicImage::ImageRgba32F(output))
}

/// Compute the median of a slice of f32 values.
///
/// Uses `sort_unstable_by` with partial_cmp for speed.
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

/// Compute the trimmed mean: discard the lowest and highest values, average the rest.
/// More robust than simple average, more efficient than full median.
#[allow(dead_code)]
pub fn robust_merge(
    frames: &[DynamicImage],
    width: u32,
    height: u32,
    trim_count: usize,
) -> Result<DynamicImage> {
    if frames.len() <= trim_count * 2 {
        // Not enough frames for trimming, fall back to average
        return merge_average_cpu(frames, width, height);
    }

    let num_frames = frames.len();

    // Pre-extract all frame data
    let frame_data: Vec<Vec<f32>> = frames
        .iter()
        .map(|frame| {
            let rgba = frame.to_rgba32f();
            rgba.as_flat_samples().samples.to_vec()
        })
        .collect();

    let mut output: ImageBuffer<Rgba<f32>, Vec<f32>> =
        ImageBuffer::new(width, height);
    let pixel_count = (width * height) as usize;

    // Process in parallel
    let output_pixels: Vec<[f32; 4]> = (0..pixel_count)
        .into_par_iter()
        .map(|idx| {
            let base = idx * 4;

            let mut r_vals: Vec<f32> = (0..num_frames).map(|f| frame_data[f][base]).collect();
            let mut g_vals: Vec<f32> = (0..num_frames).map(|f| frame_data[f][base + 1]).collect();
            let mut b_vals: Vec<f32> = (0..num_frames).map(|f| frame_data[f][base + 2]).collect();
            let mut a_vals: Vec<f32> = (0..num_frames).map(|f| frame_data[f][base + 3]).collect();

            [
                trimmed_mean(&mut r_vals, trim_count),
                trimmed_mean(&mut g_vals, trim_count),
                trimmed_mean(&mut b_vals, trim_count),
                trimmed_mean(&mut a_vals, trim_count),
            ]
        })
        .collect();

    for (idx, pixel) in output.pixels_mut().enumerate() {
        pixel[0] = output_pixels[idx][0];
        pixel[1] = output_pixels[idx][1];
        pixel[2] = output_pixels[idx][2];
        pixel[3] = output_pixels[idx][3];
    }

    Ok(DynamicImage::ImageRgba32F(output))
}

/// Compute the trimmed mean of a f32 slice
fn trimmed_mean(values: &mut [f32], trim: usize) -> f32 {
    let len = values.len();
    if trim >= len / 2 {
        return median_f32(values);
    }

    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let sum: f64 = values[trim..len - trim]
        .iter()
        .map(|&v| v as f64)
        .sum();
    (sum / (len - 2 * trim) as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    fn create_test_frame(width: u32, height: u32, fill: [f32; 4]) -> DynamicImage {
        let img: ImageBuffer<Rgba<f32>, Vec<f32>> =
            ImageBuffer::from_fn(width, height, |_, _| Rgba(fill));
        DynamicImage::ImageRgba32F(img)
    }

    #[test]
    fn test_merge_average_two_frames() {
        let f1 = create_test_frame(64, 64, [0.2, 0.4, 0.6, 1.0]);
        let f2 = create_test_frame(64, 64, [0.4, 0.6, 0.8, 1.0]);

        let result = merge_frames(&[f1, f2], MergeMethod::Average, false).unwrap();

        // Check center pixel
        let rgba = result.to_rgba32f();
        let pixel = rgba.get_pixel(32, 32);
        assert!((pixel[0] - 0.3).abs() < 0.001);
        assert!((pixel[1] - 0.5).abs() < 0.001);
        assert!((pixel[2] - 0.7).abs() < 0.001);
        assert!((pixel[3] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_merge_median_outlier() {
        let f1 = create_test_frame(64, 64, [0.1, 0.1, 0.1, 1.0]);
        let f2 = create_test_frame(64, 64, [0.2, 0.2, 0.2, 1.0]);
        let f3 = create_test_frame(64, 64, [1.0, 1.0, 1.0, 1.0]); // outlier

        let result = merge_frames(&[f1, f2, f3], MergeMethod::Median, false).unwrap();

        let rgba = result.to_rgba32f();
        let pixel = rgba.get_pixel(32, 32);
        assert!((pixel[0] - 0.2).abs() < 0.001);
        assert!((pixel[1] - 0.2).abs() < 0.001);
        assert!((pixel[2] - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_median_f32() {
        let mut vals = vec![0.5, 0.1, 0.9, 0.3, 0.7];
        let m = median_f32(&mut vals);
        assert!((m - 0.5).abs() < 0.001);

        let mut vals = vec![0.2, 0.8, 0.4, 0.6];
        let m = median_f32(&mut vals);
        assert!((m - 0.5).abs() < 0.001);
    }
}

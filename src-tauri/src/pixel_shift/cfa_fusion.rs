use anyhow::{Result, anyhow};
use image::{DynamicImage, ImageBuffer, Rgba};
use rayon::prelude::*;
use rawler::{
    CFA,
    decoders::RawDecodeParams,
    rawsource::RawSource,
};

/// Raw CFA data extracted from a single pixel-shift frame.
pub struct CfaFrame {
    pub data: Vec<u16>,
    pub width: u32,
    pub height: u32,
    pub cfa: CFA,
    pub black_levels: [f32; 4],
    pub white_level: f32,
    pub wb_coeffs: [f32; 4],
}

/// Extract CFA frame data from a NEF file without demosaicing.
pub fn extract_cfa_frame(file_bytes: &[u8]) -> Result<CfaFrame> {
    let source = RawSource::new_from_slice(file_bytes);
    let decoder = rawler::get_decoder(&source)
        .map_err(|e| anyhow!("Failed to get decoder: {}", e))?;

    let raw_image = decoder.raw_image(&source, &RawDecodeParams::default(), false)
        .map_err(|e| anyhow!("Failed to decode raw image: {}", e))?;

    let data = match &raw_image.data {
        rawler::rawimage::RawImageData::Integer(d) => d.clone(),
        rawler::rawimage::RawImageData::Float(d) => {
            d.iter().map(|&v| (v.clamp(0.0, 1.0) * 65535.0) as u16).collect()
        }
    };

    let cfa = raw_image.camera.cfa.clone();

    let black_levels = {
        let levels = &raw_image.blacklevel.levels;
        if levels.len() >= 4 {
            [levels[0].as_f32(), levels[1].as_f32(), levels[2].as_f32(), levels[3].as_f32()]
        } else if levels.len() == 1 {
            let v = levels[0].as_f32(); [v, v, v, v]
        } else {
            [0.0; 4]
        }
    };

    let white_level = raw_image.whitelevel.0.first().copied().unwrap_or(u16::MAX as u32) as f32;

    Ok(CfaFrame {
        data,
        width: raw_image.width as u32,
        height: raw_image.height as u32,
        cfa,
        black_levels,
        white_level,
        wb_coeffs: raw_image.wb_coeffs,
    })
}

// ─── Nikon Z6III pixel-shift pattern ───
//
// 4-shot:  sensor at (0,0), (1,0), (0,1), (1,1) — fills 2×2 Bayer cell
// 8-shot:  4-shot repeated for noise reduction
// 16-shot: 4×4 sub-pixel grid within 2×2 Bayer, step=0.5px → 2× super-resolution
// 32-shot: 16-shot repeated for noise reduction

/// Get Nikon ideal pixel-shift offsets for the given frame count.
/// Returns a vector of (dx, dy) pairs for all frames.
pub fn nikon_shift_patterns(frame_count: usize) -> Vec<(f32, f32)> {
    let (unique, repeats) = get_shift_pattern(frame_count);
    let mut all = Vec::with_capacity(frame_count);
    for _ in 0..repeats {
        all.extend(&unique);
    }
    all.truncate(frame_count);
    all
}

/// Get Nikon pixel-shift sub-pixel offsets for the given frame count.
/// Returns (unique_positions, repeat_count).
fn get_shift_pattern(frame_count: usize) -> (Vec<(f32, f32)>, usize) {
    match frame_count {
        4 | 8 => {
            // 4 unique positions covering 2×2 Bayer cell
            let unique: Vec<(f32, f32)> = vec![
                (0.0, 0.0), (1.0, 0.0),
                (0.0, 1.0), (1.0, 1.0),
            ];
            let repeats = if frame_count == 8 { 2 } else { 1 };
            (unique, repeats)
        }
        16 | 32 => {
            // 16 unique positions: 4×4 grid within 2×2 Bayer cell, step = 0.5px
            // This covers each Bayer position at 0.5px sub-pixel precision
            let mut unique = Vec::with_capacity(16);
            for ry in 0..4u32 {
                for rx in 0..4u32 {
                    unique.push((rx as f32 * 0.5, ry as f32 * 0.5));
                }
            }
            let repeats = if frame_count == 32 { 2 } else { 1 };
            (unique, repeats)
        }
        _ => {
            // Default: 4-shot pattern
            let unique = vec![
                (0.0, 0.0), (1.0, 0.0),
                (0.0, 1.0), (1.0, 1.0),
            ];
            (unique, 1)
        }
    }
}

/// CFA-level pixel-shift fusion with super-resolution.
///
/// Maps all Bayer samples from all frames into a unified super-resolution
/// output grid. Each output pixel collects real R, G, B measurements from
/// the Bayer arrays whose sensor positions overlap that grid location.
///
/// Output resolution:
///   - 4/8 shot: native resolution (fills Bayer gaps)
///   - 16/32 shot: 2× native resolution (sub-pixel reconstruction)
use super::noise_model::NoiseModel;

/// CFA fusion using ideal Nikon shift patterns.
pub fn fuse_cfa_frames(frames: &[CfaFrame]) -> Result<DynamicImage> {
    let all_shifts = nikon_shift_patterns(frames.len());
    fuse_cfa_frames_with_shifts(frames, &all_shifts, None)
}

/// CFA fusion using externally measured sub-pixel shifts with optional noise model.
///
/// `shifts` should contain one (dx, dy) pair per frame, in sensor pixel units.
/// `noise_model` enables PARSEK-style confidence-weighted fusion.
pub fn fuse_cfa_frames_with_shifts(
    frames: &[CfaFrame],
    shifts: &[(f32, f32)],
    noise_model: Option<&NoiseModel>,
) -> Result<DynamicImage> {
    if frames.is_empty() {
        return Err(anyhow!("No frames provided"));
    }

    let (width, height) = (frames[0].width, frames[0].height);
    let num_frames = frames.len();

    // Validate dimensions
    for (i, frame) in frames.iter().enumerate().skip(1) {
        if frame.width != width || frame.height != height {
            return Err(anyhow!(
                "Frame {} dimensions {}x{} != {}x{}", i, frame.width, frame.height, width, height
            ));
        }
    }

    // Super-resolution scale: 2x for 16+ unique measured positions
    // Check how many distinct shifts we have
    let scale = estimate_scale_from_shifts(shifts);
    let out_w = width * scale;
    let out_h = height * scale;

    // Normalize all frames to [0, 1]
    let normalized: Vec<Vec<f32>> = frames
        .par_iter()
        .map(|frame| {
            let wl = frame.white_level.max(1.0);
            frame.data.iter().map(|&v| (v as f32 / wl).clamp(0.0, 1.0)).collect()
        })
        .collect();

    let cfa = &frames[0].cfa;
    let pixel_count = (out_w * out_h) as usize;

    let output_pixels: Vec<[f32; 3]> = (0..pixel_count)
        .into_par_iter()
        .map(|idx| {
            let ox = (idx % out_w as usize) as f32;
            let oy = (idx / out_w as usize) as f32;

            // Map output pixel to sensor coordinates
            // At 2× scale: ox=0→sensor 0.0, ox=1→sensor 0.5, ox=2→sensor 1.0, ...
            let sensor_x = ox / scale as f32;
            let sensor_y = oy / scale as f32;

            // Collect R, G, B samples from all frames with confidence weighting
            let mut r_weighted_sum: f32 = 0.0;
            let mut r_total_weight: f32 = 0.0;
            let mut g_weighted_sum: f32 = 0.0;
            let mut g_total_weight: f32 = 0.0;
            let mut b_weighted_sum: f32 = 0.0;
            let mut b_total_weight: f32 = 0.0;

            // Reference frame sample values for noise model queries
            let mut ref_r: Option<u16> = None;
            let mut ref_g: Option<u16> = None;
            let mut ref_b: Option<u16> = None;

            for fi in 0..num_frames {
                let (sx_shift, sy_shift) = shifts[fi];
                let frame = &frames[fi];
                let norm = &normalized[fi];

                // Sensor position of this output pixel in this frame's coordinates
                let sx = sensor_x - sx_shift;
                let sy = sensor_y - sy_shift;

                // Bilinear gather from the 2×2 Bayer neighborhood
                let sx0 = sx.floor() as i32;
                let sy0 = sy.floor() as i32;
                let fx = sx - sx0 as f32;
                let fy = sy - sy0 as f32;

                for dy in 0i32..=1i32 {
                    for dx in 0i32..=1i32 {
                        let px = (sx0 + dx).clamp(0, width as i32 - 1) as u32;
                        let py = (sy0 + dy).clamp(0, height as i32 - 1) as u32;

                        let w = (if dx == 0 { 1.0 - fx } else { fx })
                              * (if dy == 0 { 1.0 - fy } else { fy });
                        if w < 0.01 { continue; }

                        let color = cfa.color_at(py as usize, px as usize);
                        let v = norm.get((py * width + px) as usize).copied().unwrap_or(0.0);

                        // Black level correction per channel
                        let bl = frame.black_levels;
                        let ch = match color {
                            0 => 0,        // R
                            1 | 3 => 1,    // G
                            2 => 3,        // B
                            _ => 1,
                        };
                        let corr = ((v * frame.white_level - bl[ch])
                            / (frame.white_level - bl[ch]).max(1.0))
                            .clamp(0.0, 1.0);

                        // Confidence weight: spatial (from bilinear weights) × noise model
                        let spatial_conf = w;
                        let noise_conf = if let Some(nm) = noise_model {
                            let raw_val = (corr * 65535.0) as u16;
                            let ch = match color { 0 => 0, 1|3 => 1, 2 => 2, _ => 1 };
                            // Use first frame's value as reference for noise model
                            let ref_val = if fi == 0 {
                                raw_val
                            } else {
                                match color {
                                    0 => ref_r.unwrap_or(raw_val),
                                    1|3 => ref_g.unwrap_or(raw_val),
                                    2 => ref_b.unwrap_or(raw_val),
                                    _ => raw_val,
                                }
                            };
                            nm.confidence(ref_val, raw_val, ch)
                        } else {
                            1.0
                        };

                        let confidence = spatial_conf * noise_conf;
                        if confidence < 0.001 { continue; }

                        match color {
                            0 => {
                                r_weighted_sum += corr * confidence;
                                r_total_weight += confidence;
                                if fi == 0 { ref_r = Some((corr * 65535.0) as u16); }
                            }
                            1 | 3 => {
                                g_weighted_sum += corr * confidence;
                                g_total_weight += confidence;
                                if fi == 0 { ref_g = Some((corr * 65535.0) as u16); }
                            }
                            2 => {
                                b_weighted_sum += corr * confidence;
                                b_total_weight += confidence;
                                if fi == 0 { ref_b = Some((corr * 65535.0) as u16); }
                            }
                            _ => {}
                        }
                    }
                }
            }

            // Confidence-weighted average
            let r = if r_total_weight > 0.0 { r_weighted_sum / r_total_weight } else { 0.0 };
            let g = if g_total_weight > 0.0 { g_weighted_sum / g_total_weight } else { 0.0 };
            let b = if b_total_weight > 0.0 { b_weighted_sum / b_total_weight } else { 0.0 };

            // Fallback for missing channels
            let r = if r_total_weight > 0.0 { r } else { g * 0.8 };
            let b = if b_total_weight > 0.0 { b } else { g * 1.2 };
            let g = if g_total_weight > 0.0 { g } else { (r + b) * 0.5 };

            [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
        })
        .collect();

    let mut output: ImageBuffer<Rgba<f32>, Vec<f32>> = ImageBuffer::new(out_w, out_h);
    for (idx, pixel) in output.pixels_mut().enumerate() {
        pixel[0] = output_pixels[idx][0];
        pixel[1] = output_pixels[idx][1];
        pixel[2] = output_pixels[idx][2];
        pixel[3] = 1.0;
    }

    Ok(DynamicImage::ImageRgba32F(output))
}

/// Estimate super-resolution scale from the spread of measured shifts.
///
/// Returns 2 if shifts cover at least 0.75 pixels in any direction
/// (indicating sub-pixel coverage), 1 otherwise.
fn estimate_scale_from_shifts(shifts: &[(f32, f32)]) -> u32 {
    if shifts.is_empty() { return 1; }
    let first = shifts[0];
    let max_dx = shifts.iter().map(|s| (s.0 - first.0).abs()).fold(0.0f32, f32::max);
    let max_dy = shifts.iter().map(|s| (s.1 - first.1).abs()).fold(0.0f32, f32::max);
    if max_dx.max(max_dy) >= 0.75 { 2 } else { 1 }
}

fn median(vals: &[f32]) -> f32 {
    if vals.is_empty() { return 0.0; }
    if vals.len() == 1 { return vals[0]; }
    let mut s: Vec<f32> = vals.to_vec();
    s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = s.len() / 2;
    if s.len() % 2 == 0 { (s[mid-1] + s[mid]) * 0.5 } else { s[mid] }
}

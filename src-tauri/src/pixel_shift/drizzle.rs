use rayon::prelude::*;

/// Bayer Drizzle super-resolution merger.
///
/// Splits each CFA frame into 4 independent monochrome channels
/// (R, G1, G2, B), drizzles each onto a 2× denser output grid using
/// the known sub-pixel shift per frame, then combines into RGB.
///
/// Algorithm: Fruchter & Hook (2002) "Drizzle: A Method for the Linear
/// Reconstruction of Undersampled Images", adapted for Bayer CFA data.

/// Output of one channel drizzle pass
struct ChannelDrizzle {
    sum: Vec<f64>,     // weighted sum of values
    weight: Vec<f64>,  // total weight per pixel
    width: u32,
    height: u32,
}

impl ChannelDrizzle {
    fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self { sum: vec![0.0; size], weight: vec![0.0; size], width, height }
    }

    #[inline]
    fn add(&mut self, x: f32, y: f32, value: f32, pixscale: f32) {
        // Input pixel center at (x, y) in output coordinates
        // Input pixel extends ±0.5 input pixels ≈ ±0.5/pixscale output pixels
        let half = 0.5 / pixscale;

        let x0 = (x - half).floor() as i32;
        let y0 = (y - half).floor() as i32;
        let x1 = (x + half).ceil() as i32;
        let y1 = (y + half).ceil() as i32;

        for oy in y0..y1 {
            if oy < 0 || oy >= self.height as i32 { continue; }
            for ox in x0..x1 {
                if ox < 0 || ox >= self.width as i32 { continue; }

                // Area overlap between input pixel and output pixel
                let ol = (ox as f32 + 0.5).min(x + half) - (ox as f32 - 0.5).max(x - half);
                let ot = (oy as f32 + 0.5).min(y + half) - (oy as f32 - 0.5).max(y - half);
                let overlap = (ol.max(0.0) * ot.max(0.0)).min(1.0);

                if overlap > 0.0 {
                    let idx = (oy as u32 * self.width + ox as u32) as usize;
                    self.sum[idx] += value as f64 * overlap as f64;
                    self.weight[idx] += overlap as f64;
                }
            }
        }
    }

    fn normalize(&self) -> Vec<f32> {
        self.sum.par_iter().zip(self.weight.par_iter())
            .map(|(&s, &w)| if w > 0.0 { (s / w) as f32 } else { 0.0f32 })
            .collect()
    }
}

/// Run Bayer Drizzle on a set of CFA frames with known sub-pixel shifts.
///
/// # Arguments
/// * `cfa_data` — Raw u16 pixel data from each frame
/// * `width`, `height` — Native sensor dimensions
/// * `cfa` — CFA pattern string (e.g. "RGGB")
/// * `shifts` — Per-frame (dx, dy) offset in sensor pixels, relative to reference
/// * `black_levels` — Per-frame per-channel black levels [R, G1, B, G2]
/// * `white_levels` — Per-frame white levels
/// * `scale` — Output scale factor (2 = 2× super-resolution)
///
/// # Returns
/// An RGB f32 image at `width*scale × height*scale`
pub fn bayer_drizzle(
    cfa_data: &[&[u16]],
    width: u32,
    height: u32,
    cfa: &str,
    shifts: &[(f32, f32)],
    black_levels: &[[f32; 4]],
    white_levels: &[f32],
    scale: u32,
) -> image::DynamicImage {
    let num_frames = cfa_data.len();
    let out_w = width * scale;
    let out_h = height * scale;

    // 4 independent channel drizzles: R, G1, G2, B
    let mut r_chan = ChannelDrizzle::new(out_w, out_h);
    let mut g1_chan = ChannelDrizzle::new(out_w, out_h);
    let mut g2_chan = ChannelDrizzle::new(out_w, out_h);
    let mut b_chan = ChannelDrizzle::new(out_w, out_h);

    for fi in 0..num_frames {
        let (dx, dy) = shifts[fi];
        let data = cfa_data[fi];
        let bl = black_levels[fi];
        let wl = white_levels[fi].max(1.0);
        let range_r = (wl - bl[0]).max(1.0);
        let range_g1 = (wl - bl[1]).max(1.0);
        let range_b = (wl - bl[2]).max(1.0);
        let range_g2 = (wl - bl[3]).max(1.0);

        // Drizzle each Bayer pixel from this frame onto the output grid
        for sy in 0..height {
            for sx in 0..width {
                let idx = (sy * width + sx) as usize;
                let raw = data[idx] as f32;

                // Determine Bayer color and per-channel params
                let (value, channel, channel_offset_x, channel_offset_y) = bayer_info(
                    cfa, sx, sy, raw, &bl, wl, range_r, range_g1, range_b, range_g2,
                );

                if value <= 0.0 { continue; }

                // Map this input pixel's sensor position to output grid coordinates
                // Input pixel center in sensor coords: (sx + 0.5 + dx, sy + 0.5 + dy)
                // Output grid: each output pixel = 1/scale sensor pixels
                let out_x = (sx as f32 + 0.5 + dx + channel_offset_x) * scale as f32;
                let out_y = (sy as f32 + 0.5 + dy + channel_offset_y) * scale as f32;
                let pixscale = scale as f32; // input pixel size in output pixels

                match channel {
                    0 => r_chan.add(out_x, out_y, value, pixscale),
                    1 => g1_chan.add(out_x, out_y, value, pixscale),
                    2 => b_chan.add(out_x, out_y, value, pixscale),
                    3 => g2_chan.add(out_x, out_y, value, pixscale),
                    _ => {}
                }
            }
        }
    }

    // Normalize each channel
    let r_out = r_chan.normalize();
    let g1_out = g1_chan.normalize();
    let g2_out = g2_chan.normalize();
    let b_out = b_chan.normalize();

    // Combine into RGB: G = average of G1 + G2
    let pixel_count = (out_w * out_h) as usize;
    let mut output: image::ImageBuffer<image::Rgba<f32>, Vec<f32>> =
        image::ImageBuffer::new(out_w, out_h);

    output.pixels_mut().enumerate().for_each(|(i, p)| {
        let r = r_out[i];
        let g = (g1_out[i] + g2_out[i]) * 0.5;
        let b = b_out[i];
        p[0] = r.clamp(0.0, 1.0);
        p[1] = g.clamp(0.0, 1.0);
        p[2] = b.clamp(0.0, 1.0);
        p[3] = 1.0;
    });

    image::DynamicImage::ImageRgba32F(output)
}

/// Extra Bayer channel relative to the R pixel at (0,0).
/// Channel offsets for positioning on the output grid:
///   R at (0,0), G1 at (1,0), B at (1,1), G2 at (0,1)
#[inline]
fn bayer_info(
    cfa: &str,
    sx: u32, sy: u32,
    raw: f32,
    bl: &[f32; 4],
    wl: f32,
    range_r: f32, range_g1: f32, range_b: f32, range_g2: f32,
) -> (f32, u8, f32, f32) {
    let pos = ((sy & 1) << 1) | (sx & 1);

    match cfa {
        "RGGB" => match pos {
            0 => { // R
                let v = ((raw - bl[0]) / range_r).clamp(0.0, 1.0);
                (v, 0, 0.0, 0.0)
            }
            1 => { // G1 (top-right in 2×2)
                let v = ((raw - bl[1]) / range_g1).clamp(0.0, 1.0);
                (v, 1, 0.0, 0.0)
            }
            2 => { // G2 (bottom-left in 2×2)
                let v = ((raw - bl[3]) / range_g2).clamp(0.0, 1.0);
                (v, 3, 0.0, 0.0)
            }
            3 => { // B
                let v = ((raw - bl[2]) / range_b).clamp(0.0, 1.0);
                (v, 2, 0.0, 0.0)
            }
            _ => (0.0, 0, 0.0, 0.0),
        },
        _ => { // Default: assume RGGB
            bayer_info("RGGB", sx, sy, raw, bl, wl, range_r, range_g1, range_b, range_g2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayer_drizzle_basic() {
        // Create 4 synthetic frames with known shifts
        let w = 64u32;
        let h = 64u32;
        let size = (w * h) as usize;

        // Reference: checkerboard pattern
        let mut ref_data = vec![0u16; size];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                ref_data[idx] = if (x + y) % 2 == 0 { 10000 } else { 20000 };
            }
        }

        let frames: Vec<&[u16]> = (0..4).map(|_| ref_data.as_slice()).collect();
        let bl = [0.0f32; 4];
        let wl = [65535.0f32; 4];
        let shifts = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];

        let result = bayer_drizzle(
            &frames, w, h, "RGGB", &shifts,
            &[bl, bl, bl, bl], &[65535.0; 4], 2,
        );

        assert_eq!(result.width(), 128);
        assert_eq!(result.height(), 128);
    }
}

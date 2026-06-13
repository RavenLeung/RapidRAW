use rayon::prelude::*;

/// Bayer Drizzle super-resolution.
///
/// Splits CFA frames into 4 monochrome channels (R, G1, G2, B),
/// drizzles each independently onto a 2× denser output grid using
/// the measured sub-pixel shift per frame, then combines into RGB.
///
/// Algorithm: Fruchter & Hook (2002), adapted for Bayer CFA data.

struct DrizzleChan {
    sum: Vec<f32>,
    weight: Vec<f32>,
    w: u32,
    h: u32,
}

impl DrizzleChan {
    fn new(w: u32, h: u32) -> Self {
        Self { sum: vec![0.0; (w*h) as usize], weight: vec![0.0; (w*h) as usize], w, h }
    }

    #[inline(always)]
    fn add(&mut self, ox: u32, oy: u32, val: f32) {
        let i = (oy * self.w + ox) as usize;
        unsafe {
            *self.sum.get_unchecked_mut(i) += val;
            *self.weight.get_unchecked_mut(i) += 1.0;
        }
    }

    fn normalize(self) -> Vec<f32> {
        self.sum.into_par_iter().zip(self.weight.into_par_iter())
            .map(|(s, w)| if w > 0.0 { s / w } else { 0.0 }).collect()
    }
}

/// Drizzle one Bayer channel onto the output grid.
#[allow(clippy::too_many_arguments)]
fn drizzle_one_channel(
    target: u8, width: u32, height: u32, cfa: &str,
    cfa_data: &[&[u16]], shifts: &[(f32, f32)],
    black_levels: &[[f32; 4]], white_levels: &[f32], scale: u32,
    ow: u32, oh: u32,
) -> Vec<f32> {
    let nf = cfa_data.len();
    let tile_h = 128u32;
    let tile_starts: Vec<u32> = (0..height).step_by(tile_h as usize).collect();

    let all_tiles: Vec<Vec<(u32, u32, f32)>> = tile_starts.par_iter().map(|&sy_start| {
        let sy_end = (sy_start + tile_h).min(height);
        let mut samples = Vec::with_capacity(500_000);
        for fi in 0..nf {
            let (dx, dy) = shifts[fi];
            let data = cfa_data[fi];
            let bl = black_levels[fi];
            let wl = white_levels[fi].max(1.0);
            let ch_bl = match target { 0 => bl[0], 1 => bl[1], 2 => bl[2], 3 => bl[3], _ => 0.0 };
            let ch_range = (wl - ch_bl).max(1.0);
            for sy in sy_start..sy_end {
                for sx in 0..width {
                    if !bayer_is(cfa, sx, sy, target) { continue; }
                    let raw = cfa_data[fi][(sy * width + sx) as usize] as f32;
                    let val = ((raw - ch_bl) / ch_range).clamp(0.0, 1.0);
                    if val <= 0.0 { continue; }
                    let ox = (sx as f32 + 0.5 + dx) * scale as f32;
                    let oy = (sy as f32 + 0.5 + dy) * scale as f32;
                    let oix = ox.round() as i32;
                    let oiy = oy.round() as i32;
                    if oix >= 0 && oiy >= 0 && (oix as u32) < ow && (oiy as u32) < oh {
                        samples.push((oix as u32, oiy as u32, val));
                    }
                }
            }
        }
        samples
    }).collect();

    let mut chan = DrizzleChan::new(ow, oh);
    for tile in &all_tiles {
        for &(ox, oy, val) in tile {
            chan.add(ox, oy, val);
        }
    }
    chan.normalize()
}

pub fn bayer_drizzle(
    cfa_data: &[&[u16]], width: u32, height: u32, cfa: &str,
    shifts: &[(f32, f32)],
    black_levels: &[[f32; 4]], white_levels: &[f32], scale: u32,
) -> image::DynamicImage {
    let nf = cfa_data.len();
    let ow = width * scale;
    let oh = height * scale;

    // Process 4 channels: R(0) at (0,0), G1(1) at (1,0), B(2) at (1,1), G2(3) at (0,1)
    let r_out = drizzle_one_channel(0, width, height, cfa, cfa_data, shifts, black_levels, white_levels, scale, ow, oh);
    let g1_out = drizzle_one_channel(1, width, height, cfa, cfa_data, shifts, black_levels, white_levels, scale, ow, oh);
    let g2_out = drizzle_one_channel(3, width, height, cfa, cfa_data, shifts, black_levels, white_levels, scale, ow, oh);
    let b_out = drizzle_one_channel(2, width, height, cfa, cfa_data, shifts, black_levels, white_levels, scale, ow, oh);

    // Combine
    let mut out = image::ImageBuffer::<image::Rgba<f32>, Vec<f32>>::new(ow, oh);
    out.pixels_mut().enumerate().for_each(|(i, p)| {
        p[0] = r_out[i].clamp(0.0, 1.0);
        p[1] = ((g1_out[i] + g2_out[i]) * 0.5).clamp(0.0, 1.0);
        p[2] = b_out[i].clamp(0.0, 1.0);
        p[3] = 1.0;
    });
    image::DynamicImage::ImageRgba32F(out)
}

#[inline]
fn bayer_is(cfa: &str, x: u32, y: u32, target: u8) -> bool {
    let pos = ((y & 1) << 1) | (x & 1);
    match cfa {
        "RGGB" => match (pos, target) {
            (0, 0) | (1, 1) | (2, 3) | (3, 2) => true,
            _ => false,
        },
        _ => bayer_is("RGGB", x, y, target),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drizzle() {
        let w = 64; let h = 64;
        let data: Vec<u16> = (0..w*h).map(|i| if i%2==0 {10000} else {20000}).collect();
        let frames: Vec<&[u16]> = (0..4).map(|_| data.as_slice()).collect();
        let bl = [0.0f32; 4]; let wl = [65535.0f32; 4];
        let shifts = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let r = bayer_drizzle(&frames, w, h, "RGGB", &shifts, &[bl;4], &[65535.0;4], 2);
        assert_eq!(r.width(), 128);
        assert_eq!(r.height(), 128);
    }
}

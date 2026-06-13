use rayon::prelude::*;

/// 2D histogram noise model (PARSEK-style).
///
/// Models the probability that two pixel values (v1, v2) from different
/// frames represent the same scene content. Built by accumulating matching
/// pixel pairs across all aligned frames.
///
/// The histogram has 256×256 bins per color channel, using the top 8 bits
/// of each 16-bit pixel value.
pub struct NoiseModel {
    /// hist[channel][v1_high8][v2_high8] → count
    histograms: [[[u32; 256]; 256]; 3],
    /// Whether the model has been normalized
    normalized: bool,
}

const HBITS: u32 = 8;
const HDIM: usize = 256;

impl NoiseModel {
    /// Create an empty noise model.
    pub fn new() -> Self {
        Self {
            histograms: [[[0u32; 256]; 256]; 3],
            normalized: false,
        }
    }

    /// Build the noise model from aligned frame pairs.
    ///
    /// For each pair of frames (ref, other), at each pixel position,
    /// maps the ref pixel to the other frame using the alignment shift,
    /// and increments the histogram bin for the (ref_val, other_val) pair.
    pub fn build(
        &mut self,
        frames: &[Vec<u16>],
        width: u32,
        height: u32,
        shifts: &[(f32, f32)],
        is_cfa: bool,
        cfa_pattern: Option<&str>,
    ) {
        if frames.len() < 2 {
            return;
        }

        let ref_data = &frames[0];
        let stride = width as usize;

        // Process frame pairs in parallel
        let hist_chunks: Vec<[[[u32; 256]; 256]; 3]> = (1..frames.len())
            .into_par_iter()
            .map(|fi| {
                let mut local_hist = [[[0u32; 256]; 256]; 3];
                let (dx, dy) = shifts[fi];
                let other_data = &frames[fi];

                // Sample every 4th pixel for speed
                for y in (0..height as i32 - 1).step_by(4) {
                    for x in (0..width as i32 - 1).step_by(4) {
                        let ref_idx = (y as usize * stride + x as usize);
                        if ref_idx >= ref_data.len() {
                            continue;
                        }

                        let ox = x as f32 - dx;
                        let oy = y as f32 - dy;

                        let ox_i = ox.round() as i32;
                        let oy_i = oy.round() as i32;

                        if ox_i < 0 || oy_i < 0
                            || ox_i >= width as i32 - 1
                            || oy_i >= height as i32 - 1
                        {
                            continue;
                        }

                        let other_idx = (oy_i as usize * stride + ox_i as usize);
                        if other_idx >= other_data.len() {
                            continue;
                        }

                        let v1 = ref_data[ref_idx];
                        let v2 = other_data[other_idx];

                        // Determine channel for CFA
                        let channel = if is_cfa {
                            if let Some(cfa) = cfa_pattern {
                                cfa_channel_at(cfa, x as usize, y as usize)
                            } else {
                                1 // default G
                            }
                        } else {
                            1
                        };

                        let b1 = hval(v1);
                        let b2 = hval(v2);

                        local_hist[channel][b1][b2] = local_hist[channel][b1][b2].saturating_add(1);
                    }
                }

                local_hist
            })
            .collect();

        // Merge parallel results
        for chunk in &hist_chunks {
            for c in 0..3 {
                for i in 0..256 {
                    for j in 0..256 {
                        self.histograms[c][i][j] =
                            self.histograms[c][i][j].saturating_add(chunk[c][i][j]);
                    }
                }
            }
        }
    }

    /// Normalize the histogram model.
    ///
    /// Each row is divided by the diagonal value, then raised to `hist_exp`.
    /// This converts counts to probabilities and applies a sharpening/softening exponent.
    pub fn normalize(&mut self, hist_exp: f32) {
        for c in 0..3 {
            // Monotonic accumulation: each row is accumulated left→right then right→left
            for i in 0..256 {
                // Left to right
                for j in 1..256 {
                    let prev = self.histograms[c][i][j - 1];
                    let cur = self.histograms[c][i][j];
                    if cur < prev {
                        self.histograms[c][i][j] = prev;
                    }
                }
                // Right to left
                for j in (0..255).rev() {
                    let next = self.histograms[c][i][j + 1];
                    let cur = self.histograms[c][i][j];
                    if cur < next {
                        self.histograms[c][i][j] = next;
                    }
                }
            }

            // Normalize each row by the diagonal
            for i in 0..256 {
                let diag = self.histograms[c][i][i] as f32;
                if diag > 0.0 {
                    for j in 0..256 {
                        let ratio = self.histograms[c][i][j] as f32 / diag;
                        self.histograms[c][i][j] =
                            (ratio.powf(hist_exp).clamp(0.0, 1.0) * 65535.0) as u32;
                    }
                }
            }
        }
        self.normalized = true;
    }

    /// Query confidence that ref_val and sample_val represent the same scene content.
    ///
    /// Returns a value in [0.0, 1.0] where 1.0 means the two values are very likely
    /// from the same scene, and 0.0 means they are likely different.
    pub fn confidence(&self, ref_val: u16, sample_val: u16, channel: usize) -> f32 {
        let c = channel.min(2);
        let b1 = hval(ref_val);
        let b2 = hval(sample_val);
        let raw = self.histograms[c][b1][b2] as f32;
        if !self.normalized {
            // Unnormalized: just return raw count relative to row sum
            let row_sum: u64 = self.histograms[c][b1].iter().map(|&v| v as u64).sum();
            if row_sum > 0 {
                (raw as f64 / row_sum as f64) as f32
            } else {
                0.0
            }
        } else {
            raw / 65535.0
        }
    }

    /// Export the noise model as an RGB image for visualization.
    /// Each 256×256 slice is a channel's histogram matrix.
    pub fn to_visualization(&self, channel: usize) -> image::RgbImage {
        let c = channel.min(2);
        let mut img = image::RgbImage::new(256, 256);
        let max_val: u32 = self.histograms[c]
            .iter()
            .flat_map(|row| row.iter())
            .max()
            .copied()
            .unwrap_or(1);

        for y in 0..256 {
            for x in 0..256 {
                let v = ((self.histograms[c][y][x] as f32 / max_val as f32) * 255.0) as u8;
                img.put_pixel(x as u32, y as u32, image::Rgb([v, v, v]));
            }
        }
        img
    }
}

/// Extract the top 8 bits of a 16-bit pixel value for histogram indexing.
#[inline]
fn hval(v: u16) -> usize {
    ((v >> (16 - HBITS)) & (HDIM as u16 - 1)) as usize
}

/// Get color channel index from Bayer CFA pattern at position (x, y).
/// Returns 0=R, 1=G, 2=B.
fn cfa_channel_at(cfa: &str, x: usize, y: usize) -> usize {
    let pos = ((y & 1) << 1) | (x & 1);
    match cfa {
        "RGGB" => match pos {
            0 => 0, // R
            1 => 1, // G
            2 => 1, // G
            3 => 2, // B
            _ => 1,
        },
        "BGGR" => match pos {
            0 => 2, // B
            1 => 1, // G
            2 => 1, // G
            3 => 0, // R
            _ => 1,
        },
        "GRBG" => match pos {
            0 => 1, // G
            1 => 0, // R
            2 => 2, // B
            3 => 1, // G
            _ => 1,
        },
        "GBRG" => match pos {
            0 => 1, // G
            1 => 2, // B
            2 => 0, // R
            3 => 1, // G
            _ => 1,
        },
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hval() {
        assert_eq!(hval(0), 0);
        assert_eq!(hval(65535), 255);
        assert_eq!(hval(32768), 128);
    }

    #[test]
    fn test_cfa_channel() {
        assert_eq!(cfa_channel_at("RGGB", 0, 0), 0); // R
        assert_eq!(cfa_channel_at("RGGB", 1, 0), 1); // G
        assert_eq!(cfa_channel_at("RGGB", 0, 1), 1); // G
        assert_eq!(cfa_channel_at("RGGB", 1, 1), 2); // B
    }

    #[test]
    fn test_noise_model_basic() {
        let mut model = NoiseModel::new();
        let frames: Vec<Vec<u16>> = vec![
            vec![10000u16, 20000, 30000, 40000],
            vec![10050u16, 19950, 30010, 39980],
        ];
        let shifts = vec![(0.0, 0.0), (0.0, 0.0)];
        model.build(&frames, 2, 2, &shifts, false, None);
        model.normalize(1.0);

        // Same-value confidence should be high
        let conf = model.confidence(10000, 10000, 1);
        assert!(conf > 0.5);
    }
}

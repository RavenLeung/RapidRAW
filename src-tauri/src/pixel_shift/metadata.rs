use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Information about a Nikon pixel-shift frame extracted from MakerNotes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PixelShiftInfo {
    /// Whether this frame is part of a pixel-shift burst
    pub is_pixel_shift: bool,
    /// The sequence number of this frame within the burst (0-based)
    pub sequence_number: Option<u16>,
    /// Total number of frames in the burst (4, 8, 16, or 32)
    pub total_frames: Option<u16>,
    /// The pixel-shift pattern type
    pub pattern: Option<PixelShiftPattern>,
    /// Camera model that captured this frame
    pub camera_model: Option<String>,
}

/// Types of Nikon pixel-shift patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelShiftPattern {
    /// 4-shot pattern: sensor shifts by 1 pixel in 4 directions
    FourShot,
    /// 8-shot pattern: 4 directions x 2 (with/without half-pixel offset)
    EightShot,
    /// 16-shot pattern
    SixteenShot,
    /// 32-shot pattern
    ThirtyTwoShot,
}

impl PixelShiftPattern {
    pub fn from_frame_count(count: u16) -> Option<Self> {
        match count {
            4 => Some(Self::FourShot),
            8 => Some(Self::EightShot),
            16 => Some(Self::SixteenShot),
            32 => Some(Self::ThirtyTwoShot),
            _ => None,
        }
    }
}

/// A detected group of pixel-shift frames forming a burst
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PixelShiftBurstGroup {
    /// File paths belonging to this burst
    pub paths: Vec<String>,
    /// Number of frames detected
    pub frame_count: usize,
    /// Whether pixel-shift was confirmed via metadata
    pub is_confirmed: bool,
    /// Camera model (if detected)
    pub camera_model: Option<String>,
}

/// Attempt to detect pixel-shift information from a NEF file's MakerNotes.
///
/// Nikon pixel-shift metadata is stored in MakerNotes tag 0x0056
/// ("PictureControl" / "ShotInfoD80") at index 12 (PixelShiftActive flag).
///
/// This function first tries the rawler crate for camera model, then
/// uses kamadak-exif to access the raw MakerNote binary blob for pixel-shift info.
pub fn detect_pixel_shift_from_bytes(file_bytes: &[u8]) -> Option<PixelShiftInfo> {
    // Get camera model from rawler
    let camera_model = {
        use rawler::rawsource::RawSource;
        use rawler::decoders::RawDecodeParams;

        let source = RawSource::new_from_slice(file_bytes);
        let decoder = rawler::get_decoder(&source).ok()?;
        let raw_image = decoder
            .raw_image(&source, &RawDecodeParams::default(), false)
            .ok()?;
        let model = raw_image.model.clone();
        if model.is_empty() {
            None
        } else {
            Some(model)
        }
    };

    // Try to access MakerNotes via kamadak-exif for the raw MakerNote blob
    let exifreader = exif::Reader::new();
    if let Ok(exif) = exifreader.read_from_container(&mut std::io::Cursor::new(file_bytes)) {
        // Look for MakerNote tag (0x927c)
        if let Some(maker_note_field) = exif.get_field(exif::Tag::MakerNote, exif::In::PRIMARY) {
            let maker_note_bytes = match &maker_note_field.value {
                exif::Value::Undefined(data, _offset) => Some(data.clone()),
                _ => None,
            };

            if let Some(data) = maker_note_bytes {
                // Nikon MakerNotes start with "Nikon\0" header
                if data.len() > 6 && &data[0..5] == b"Nikon" {
                    return parse_nikon_maker_notes(&data, camera_model);
                }
                // Some NEF files have TIFF-format MakerNotes without the header
                if data.len() > 100 {
                    return parse_nikon_maker_notes_raw(&data, camera_model);
                }
            }
        }
    }

    // No pixel-shift metadata found
    None
}

/// Parse Nikon MakerNotes with "Nikon\0" header (IFD format)
fn parse_nikon_maker_notes(data: &[u8], camera_model: Option<String>) -> Option<PixelShiftInfo> {
    // Skip "Nikon\0" header (6 bytes)
    let ifd_data = &data[6..];

    // Nikon MakerNote tag 0x0056 is "ShotInfo" or "PictureControl"
    // PixelShiftActive is at byte offset 12 within tag 0x0056's data
    // For Phase 1, we use a heuristic approach
    if let Some(info) = heuristic_scan_for_pixel_shift(ifd_data, camera_model.clone()) {
        return Some(info);
    }

    None
}

/// Parse Nikon MakerNotes in raw TIFF format (no header)
fn parse_nikon_maker_notes_raw(data: &[u8], camera_model: Option<String>) -> Option<PixelShiftInfo> {
    heuristic_scan_for_pixel_shift(data, camera_model)
}

/// Heuristic scanner: look for pixel-shift signatures in binary MakerNote data.
fn heuristic_scan_for_pixel_shift(
    data: &[u8],
    camera_model: Option<String>,
) -> Option<PixelShiftInfo> {
    // Pattern 1: Look for "PixelShift" ASCII in the data (Z8, Z9, Zf, Z6III)
    if let Some(pos) = data.windows(10).position(|w| w == b"PixelShift") {
        if pos + 20 < data.len() {
            let active = data[pos + 11] == 1;
            let shot_count = data.get(pos + 12).copied().unwrap_or(0) as u16;
            let seq_num = data.get(pos + 13).copied().unwrap_or(0) as u16;

            if active || shot_count > 0 {
                return Some(PixelShiftInfo {
                    is_pixel_shift: active,
                    sequence_number: if seq_num > 0 { Some(seq_num) } else { None },
                    total_frames: if shot_count > 0 {
                        Some(shot_count)
                    } else {
                        None
                    },
                    pattern: PixelShiftPattern::from_frame_count(shot_count),
                    camera_model,
                });
            }
        }
    }

    // Pattern 2: Some Nikon bodies use "NIKON PIXEL SHIFT" marker
    if data.windows(17).any(|w| w == b"NIKON PIXEL SHIFT") {
        return Some(PixelShiftInfo {
            is_pixel_shift: true,
            sequence_number: None,
            total_frames: None,
            pattern: None,
            camera_model,
        });
    }

    None
}

/// Detect pixel-shift burst groups from a list of file paths.
pub fn detect_pixel_shift_burst(paths: &[String]) -> Result<Vec<PixelShiftBurstGroup>> {
    let mut groups: Vec<PixelShiftBurstGroup> = Vec::new();

    // Group by directory first
    let mut by_dir: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    for path in paths {
        let dir = std::path::Path::new(path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        by_dir.entry(dir).or_default().push(path.clone());
    }

    for (_dir, dir_paths) in &by_dir {
        let mut sorted_paths = dir_paths.clone();
        sorted_paths.sort();

        let mut current_burst: Vec<String> = Vec::new();
        let mut burst_confirmed: bool = false;
        let mut burst_camera: Option<String> = None;

        for (i, path) in sorted_paths.iter().enumerate() {
            if current_burst.is_empty() {
                current_burst.push(path.clone());
                // Check first file for pixel-shift metadata
                if let Ok(file_bytes) = std::fs::read(path) {
                    if let Some(info) = detect_pixel_shift_from_bytes(&file_bytes) {
                        burst_confirmed = info.is_pixel_shift;
                        burst_camera = info.camera_model;
                    }
                }
                continue;
            }

            // Check if this file is sequential with the previous one
            let prev_path = &sorted_paths[i - 1];
            if are_sequential_filenames(prev_path, path) {
                current_burst.push(path.clone());
            } else {
                if current_burst.len() >= 2 {
                    groups.push(PixelShiftBurstGroup {
                        frame_count: current_burst.len(),
                        paths: std::mem::take(&mut current_burst),
                        is_confirmed: burst_confirmed,
                        camera_model: burst_camera.take(),
                    });
                } else {
                    current_burst.clear();
                }
                current_burst.push(path.clone());
                burst_confirmed = false;
                burst_camera = None;
            }
        }

        // Finalize last group
        if current_burst.len() >= 2 {
            groups.push(PixelShiftBurstGroup {
                frame_count: current_burst.len(),
                paths: current_burst,
                is_confirmed: burst_confirmed,
                camera_model: burst_camera,
            });
        }
    }

    // Fallback: group all files in the same directory as a single burst
    if groups.is_empty() {
        for (_dir, dir_paths) in by_dir {
            if dir_paths.len() >= 2 {
                groups.push(PixelShiftBurstGroup {
                    frame_count: dir_paths.len(),
                    paths: dir_paths.clone(),
                    is_confirmed: false,
                    camera_model: None,
                });
            }
        }
    }

    // Also allow treating all selected paths as a single burst
    if groups.is_empty() && paths.len() >= 2 {
        groups.push(PixelShiftBurstGroup {
            frame_count: paths.len(),
            paths: paths.to_vec(),
            is_confirmed: false,
            camera_model: None,
        });
    }

    Ok(groups)
}

/// Check if two filenames appear sequential (e.g., DSC_1234.NEF => DSC_1235.NEF)
fn are_sequential_filenames(path1: &str, path2: &str) -> bool {
    let stem1 = std::path::Path::new(path1)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let stem2 = std::path::Path::new(path2)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    fn extract_trailing_number(s: &str) -> Option<(String, u32)> {
        let digit_end = s.len();
        let mut digit_start = digit_end;

        let bytes = s.as_bytes();
        while digit_start > 0
            && digit_start <= bytes.len()
            && bytes[digit_start - 1].is_ascii_digit()
        {
            digit_start -= 1;
        }

        if digit_start < digit_end {
            let prefix = &s[..digit_start];
            let num_str = &s[digit_start..digit_end];
            if let Ok(num) = num_str.parse::<u32>() {
                return Some((prefix.to_string(), num));
            }
        }
        None
    }

    if let (Some((prefix1, num1)), Some((prefix2, num2))) =
        (extract_trailing_number(stem1), extract_trailing_number(stem2))
    {
        return prefix1 == prefix2 && (num1 + 1 == num2 || num2 + 1 == num1);
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_filenames() {
        assert!(are_sequential_filenames(
            "/path/DSC_1234.NEF",
            "/path/DSC_1235.NEF"
        ));
        assert!(are_sequential_filenames(
            "/path/IMG_0001.NEF",
            "/path/IMG_0002.NEF"
        ));
        assert!(!are_sequential_filenames(
            "/path/DSC_1234.NEF",
            "/path/DSC_1240.NEF"
        ));
        assert!(!are_sequential_filenames(
            "/path/foo.NEF",
            "/path/bar.NEF"
        ));
    }

    #[test]
    fn test_pixel_shift_pattern() {
        assert_eq!(
            PixelShiftPattern::from_frame_count(4),
            Some(PixelShiftPattern::FourShot)
        );
        assert_eq!(
            PixelShiftPattern::from_frame_count(8),
            Some(PixelShiftPattern::EightShot)
        );
        assert_eq!(
            PixelShiftPattern::from_frame_count(16),
            Some(PixelShiftPattern::SixteenShot)
        );
        assert_eq!(
            PixelShiftPattern::from_frame_count(32),
            Some(PixelShiftPattern::ThirtyTwoShot)
        );
        assert_eq!(PixelShiftPattern::from_frame_count(3), None);
    }
}

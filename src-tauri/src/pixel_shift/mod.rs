pub mod basic_merge;
pub mod frame_extraction;
pub mod metadata;

use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};
use image::{DynamicImage, ImageFormat};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::path::Path;
use tauri::{Emitter, State};

use crate::app_state::AppState;
use crate::file_management::parse_virtual_path;
use crate::formats::is_raw_file;
use crate::image_loader::load_base_image_from_bytes;
use crate::image_processing::apply_linear_to_srgb;

use basic_merge::{MergeMethod, merge_frames};
use metadata::{PixelShiftBurstGroup, detect_pixel_shift_burst};

/// Result of a pixel-shift merge operation stored in AppState
#[derive(Clone)]
pub struct PixelShiftMergeResult {
    pub image: DynamicImage,
    pub source_paths: Vec<String>,
    pub frame_count: usize,
    pub merge_method: MergeMethod,
}

/// Parameters for a pixel-shift merge operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PixelShiftMergeParams {
    pub paths: Vec<String>,
    pub method: MergeMethod,
    pub motion_compensation: bool,
}

/// Tauri command: merge selected pixel-shift frames into a high-resolution image.
///
/// Takes a list of NEF file paths from a Nikon pixel-shift burst, loads each
/// frame through the standard RAW development pipeline, merges them via the
/// chosen method, and stores the result in AppState.
#[tauri::command]
pub async fn merge_pixel_shift(
    paths: Vec<String>,
    method: String,
    motion_compensation: bool,
    app_handle: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    if paths.len() < 2 {
        return Err("Please select at least two pixel-shift frames to merge.".to_string());
    }

    // Validate all paths are RAW files
    for path in &paths {
        if !is_raw_file(path) {
            return Err(format!(
                "File '{}' is not a RAW file. Pixel-shift merging requires RAW (NEF) files.",
                Path::new(path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            ));
        }
    }

    let merge_method = match method.to_lowercase().as_str() {
        "average" => MergeMethod::Average,
        "median" => MergeMethod::Median,
        _ => {
            return Err(format!(
                "Unknown merge method: '{}'. Supported methods: average, median",
                method
            ));
        }
    };

    let _ = app_handle.emit(
        "pixel-shift-progress",
        format!("Loading {} frames...", paths.len()),
    );

    // Step 1: Load all frames via the standard RAW development pipeline
    let settings = crate::app_settings::load_settings(app_handle.clone()).unwrap_or_default();

    let loaded_frames: Vec<(String, DynamicImage)> = paths
        .iter()
        .map(|path| {
            let _ = app_handle.emit(
                "pixel-shift-progress",
                format!(
                    "Processing '{}'...",
                    Path::new(path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                ),
            );

            let file_bytes = std::fs::read(path)
                .map_err(|e| format!("Failed to read file {}: {}", path, e))?;

            let dynamic_image = load_base_image_from_bytes(&file_bytes, path, false, &settings, None)
                .map_err(|e| format!("Failed to load image {}: {}", path, e))?;

            Ok((path.clone(), dynamic_image))
        })
        .collect::<Result<Vec<_>, String>>()?;

    // Step 2: Validate frame dimensions match
    if let Some((first_path, first_img)) = loaded_frames.first() {
        let (width, height) = (first_img.width(), first_img.height());

        for (path, img) in loaded_frames.iter().skip(1) {
            if img.width() != width || img.height() != height {
                return Err(format!(
                    "Dimension mismatch: '{}' is {}x{}, expected {}x{} like '{}'.",
                    Path::new(path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    img.width(),
                    img.height(),
                    width,
                    height,
                    Path::new(first_path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                ));
            }
        }
    }

    let _ = app_handle.emit(
        "pixel-shift-progress",
        format!(
            "Merging {} frames using {:?} method...",
            loaded_frames.len(),
            merge_method
        ),
    );

    // Step 3: Merge frames
    let frames: Vec<DynamicImage> = loaded_frames
        .into_iter()
        .map(|(_, img)| img)
        .collect();

    let merged = merge_frames(&frames, merge_method, motion_compensation)
        .map_err(|e| format!("Failed to merge pixel-shift frames: {}", e))?;

    // Convert linear to sRGB for preview
    let merged_srgb = apply_linear_to_srgb(merged.clone());

    let _ = app_handle.emit("pixel-shift-progress", "Creating preview...");

    // Step 4: Generate base64 PNG preview
    let rgb8 = merged_srgb.to_rgb8();
    let mut png_bytes: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut png_bytes);
        rgb8
            .write_to(&mut cursor, ImageFormat::Png)
            .map_err(|e| format!("Failed to encode preview: {}", e))?;
    }
    let base64_str = general_purpose::STANDARD.encode(&png_bytes);
    let final_base64 = format!("data:image/png;base64,{}", base64_str);

    // Step 5: Store result in AppState
    let result = PixelShiftMergeResult {
        image: merged,
        source_paths: paths.clone(),
        frame_count: paths.len(),
        merge_method,
    };

    *state.pixel_shift_result.lock().unwrap() = Some(result);

    let _ = app_handle.emit(
        "pixel-shift-complete",
        serde_json::json!({
            "base64": final_base64,
            "frameCount": paths.len(),
            "method": method,
        }),
    );

    Ok(())
}

/// Tauri command: save the pixel-shift merge result to disk.
///
/// Saves the merged image alongside the first source frame as a TIFF file.
/// Clears the result from AppState after saving.
#[tauri::command]
pub async fn save_pixel_shift(
    first_path_str: String,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let merge_result = state.pixel_shift_result.lock().unwrap().take().ok_or_else(|| {
        "No pixel-shift merge result found in memory. It might have already been saved."
            .to_string()
    })?;

    let (first_path, _) = parse_virtual_path(&first_path_str);

    // Build output path: <first_path_without_ext>_pixelshift.tiff
    let stem = first_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let parent = first_path.parent().unwrap_or_else(|| Path::new("."));
    let output_path = parent.join(format!("{}_pixelshift.tiff", stem));

    let merged_srgb = apply_linear_to_srgb(merge_result.image.clone());
    let rgb8 = merged_srgb.to_rgb8();

    rgb8
        .save(&output_path)
        .map_err(|e| format!("Failed to save merged image: {}", e))?;

    Ok(output_path.to_string_lossy().to_string())
}

/// Tauri command: detect pixel-shift burst groups from a set of file paths.
///
/// Scans the given paths for Nikon pixel-shift frames by reading MakerNotes
/// metadata and grouping frames that belong to the same burst sequence.
#[tauri::command]
pub async fn detect_pixel_shift_groups(
    paths: Vec<String>,
    app_handle: tauri::AppHandle,
) -> Result<Vec<PixelShiftBurstGroup>, String> {
    let _ = app_handle.emit(
        "pixel-shift-progress",
        "Scanning for pixel-shift frames...",
    );

    let groups = detect_pixel_shift_burst(&paths)
        .map_err(|e| format!("Failed to detect pixel-shift groups: {}", e))?;

    Ok(groups)
}

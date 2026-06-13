pub mod alignment;
pub mod basic_merge;
pub mod cfa_fusion;
pub mod frame_extraction;
pub mod gpu_fusion;
pub mod metadata;
pub mod motion_detection;
pub mod noise_model;
pub mod skr_fusion;

use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};
use image::{DynamicImage, ImageFormat};
use rayon::prelude::*;
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

    let is_cfa_mode = method.to_lowercase().as_str() == "cfa";

    let merge_method = match method.to_lowercase().as_str() {
        "average" => MergeMethod::Average,
        "median" => MergeMethod::Median,
        "skr" => MergeMethod::SKR,
        "cfa" => MergeMethod::Median, // placeholder, not used directly
        _ => {
            return Err(format!(
                "Unknown merge method: '{}'. Supported methods: average, median, skr, cfa",
                method
            ));
        }
    };

    let _ = app_handle.emit(
        "pixel-shift-progress",
        format!("Loading {} frames...", paths.len()),
    );

    // Step 1: For CFA mode, extract raw Bayer data directly
    // For RGB modes, load via standard RAW development pipeline
    let settings = crate::app_settings::load_settings(app_handle.clone()).unwrap_or_default();

    let merged = if is_cfa_mode {
        let cfa_frames: Vec<cfa_fusion::CfaFrame> = paths
            .iter()
            .map(|path| {
                let _ = app_handle.emit(
                    "pixel-shift-progress",
                    format!(
                        "Extracting raw Bayer data from '{}'...",
                        std::path::Path::new(path)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                    ),
                );
                let file_bytes = std::fs::read(path)
                    .map_err(|e| format!("Failed to read {}: {}", path, e))?;
                cfa_fusion::extract_cfa_frame(&file_bytes)
                    .map_err(|e| format!("Failed to extract CFA from {}: {}", path, e))
            })
            .collect::<Result<Vec<_>, String>>()?;

        // Build luminance proxies for alignment from CFA data
        let _ = app_handle.emit("pixel-shift-progress", "Measuring actual frame displacements...");
        let luma_frames: Vec<DynamicImage> = cfa_frames
            .iter()
            .map(|cf| {
                // Convert CFA to simple luminance image (average 2x2 Bayer → 1 pixel)
                let lw = cf.width / 2;
                let lh = cf.height / 2;
                let mut luma: image::ImageBuffer<image::Luma<u8>, Vec<u8>> =
                    image::ImageBuffer::new(lw, lh);
                for y in 0..lh {
                    for x in 0..lw {
                        let base = (y * 2 * cf.width + x * 2) as usize;
                        let v = if base + cf.width as usize + 1 < cf.data.len() {
                            let a = cf.data[base] as u32;
                            let b = cf.data[base + 1] as u32;
                            let c = cf.data[base + cf.width as usize] as u32;
                            let d = cf.data[base + cf.width as usize + 1] as u32;
                            (((a + b + c + d) / 4) >> 8).min(255) as u8
                        } else {
                            128u8
                        };
                        luma.put_pixel(x, y, image::Luma([v]));
                    }
                }
                DynamicImage::ImageLuma8(luma)
            })
            .collect();

        let shifts = alignment::align_frames(&luma_frames, 0)
            .map_err(|e| format!("Alignment failed: {}", e))?;

        // Log measured vs ideal shifts
        let ideal = cfa_fusion::nikon_shift_patterns(cfa_frames.len());
        for (i, (s, ideal_s)) in shifts.iter().zip(ideal.iter()).enumerate() {
            log::info!(
                "Frame {}: measured=({:.3},{:.3}) ideal=({:.3},{:.3}) diff=({:.3},{:.3})",
                i, s.dx, s.dy, ideal_s.0, ideal_s.1,
                s.dx - ideal_s.0, s.dy - ideal_s.1
            );
        }

        let _ = app_handle.emit(
            "pixel-shift-progress",
            format!(
                "Fusing {} Bayer frames ({}x{}) with measured shifts...",
                cfa_frames.len(), cfa_frames[0].width, cfa_frames[0].height,
            ),
        );

        let shift_tuples: Vec<(f32, f32)> = shifts.iter().map(|s| (s.dx, s.dy)).collect();

        // Build noise model from aligned frame pairs
        let _ = app_handle.emit("pixel-shift-progress", "Building noise model...");
        let mut noise_model = noise_model::NoiseModel::new();
        let cfa_data: Vec<Vec<u16>> = cfa_frames.iter().map(|cf| cf.data.clone()).collect();
        let cfa_name = cfa_frames[0].cfa.name.clone();
        noise_model.build(
            &cfa_data,
            cfa_frames[0].width,
            cfa_frames[0].height,
            &shift_tuples,
            true,
            Some(&cfa_name),
        );
        noise_model.normalize(1.0);

        let _ = app_handle.emit(
            "pixel-shift-progress",
            format!(
                "Fusing {} Bayer frames with confidence-weighted fusion...",
                cfa_frames.len(),
            ),
        );

        cfa_fusion::fuse_cfa_frames_with_shifts(&cfa_frames, &shift_tuples, Some(&noise_model))
            .map_err(|e| format!("CFA fusion failed: {}", e))?
    } else {
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

    // Step 3: Merge frames (with optional advanced pipeline)
    let frames: Vec<DynamicImage> = loaded_frames
        .into_iter()
        .map(|(_, img)| img)
        .collect();

    if merge_method == MergeMethod::SKR {
        // Advanced pipeline: alignment -> motion detection -> SKR fusion
        merge_skr_pipeline(&frames, motion_compensation, &app_handle, &state)
            .map_err(|e| format!("Failed to merge pixel-shift frames with SKR: {}", e))?
    } else {
        merge_frames(&frames, merge_method, motion_compensation)
            .map_err(|e| format!("Failed to merge pixel-shift frames: {}", e))?
    }
    }; // closes `let merged = if is_cfa_mode { ... } else { ... };`

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

/// Run the advanced SKR (Steering Kernel Regression) merge pipeline.
///
/// Steps:
/// 1. Align frames with sub-pixel precision
/// 2. Detect subject motion between frames
/// 3. Fuse frames using structure-adaptive steering kernels
fn merge_skr_pipeline(
    frames: &[DynamicImage],
    motion_compensation: bool,
    app_handle: &tauri::AppHandle,
    state: &tauri::State<'_, AppState>,
) -> anyhow::Result<DynamicImage> {
    use alignment::align_frames;
    use motion_detection::{MotionDetectionParams, detect_motion};
    use skr_fusion::{SkrFusion, SkrFusionParams};

    let _ = app_handle.emit(
        "pixel-shift-progress",
        "Aligning frames with sub-pixel precision...",
    );

    // Step 1: Align frames
    let shifts = align_frames(frames, 0)?;

    // Log alignment results
    for (i, shift) in shifts.iter().enumerate() {
        if i == 0 {
            continue;
        }
        log::info!(
            "Frame {} alignment: dx={:.3}, dy={:.3}, quality={:.3}",
            i,
            shift.dx,
            shift.dy,
            shift.quality
        );
    }

    // Step 2: Warp frames to align with reference
    let aligned_frames: Vec<DynamicImage> = frames
        .par_iter()
        .enumerate()
        .map(|(i, frame)| {
            if i == 0 {
                frame.clone()
            } else {
                alignment::warp_frame(frame, -shifts[i].dx, -shifts[i].dy)
            }
        })
        .collect();

    let motion_mask = if motion_compensation {
        let _ = app_handle.emit(
            "pixel-shift-progress",
            "Detecting subject motion...",
        );

        let params = MotionDetectionParams::default();
        Some(detect_motion(&aligned_frames, params))
    } else {
        None
    };

    let _ = app_handle.emit(
        "pixel-shift-progress",
        "Fusing frames with steering kernel regression...",
    );

    // Step 3: SKR fusion — try GPU first, fall back to CPU
    let skr_params = SkrFusionParams {
        kernel_sigma: 1.5,
        stretch: 4.0,
        structure_sigma: 1.0,
        output_scale: 1.0,
        robust_iterations: 1,
        min_samples: 4,
    };

    // Try GPU-accelerated fusion (with poison guard)
    let gpu_context = state.gpu_context.lock().unwrap_or_else(|e| {
        log::warn!("GPU context mutex poisoned, falling back to CPU: {}", e);
        e.into_inner()
    });
    let result = if let Some(ref gpu_ctx) = *gpu_context {
        let _ = app_handle.emit(
            "pixel-shift-progress",
            "Running GPU-accelerated fusion...",
        );

        // Catch panics from WGSL validation to fall back to CPU
        let gpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gpu_fusion::PixelShiftGpuProcessor::new(gpu_ctx)
                .map_err(|e| anyhow::anyhow!("{}", e))
                .and_then(|processor| {
                    processor.fuse(&aligned_frames, motion_mask.as_ref(), &skr_params)
                })
        }));

        match gpu_result {
            Ok(Ok(result)) => {
                log::info!("GPU pixel-shift fusion completed successfully");
                Some(result)
            }
            Ok(Err(e)) => {
                log::warn!("GPU fusion failed ({}), falling back to CPU", e);
                None
            }
            Err(_panic) => {
                log::warn!("GPU processor panicked, falling back to CPU");
                None
            }
        }
    } else {
        None
    };
    drop(gpu_context);

    let result = match result {
        Some(img) => img,
        None => {
            let _ = app_handle.emit(
                "pixel-shift-progress",
                "Running CPU fusion...",
            );
            let fusion = SkrFusion::new(skr_params);
            fusion.fuse(&aligned_frames, motion_mask.as_ref())
        }
    };

    Ok(result)
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

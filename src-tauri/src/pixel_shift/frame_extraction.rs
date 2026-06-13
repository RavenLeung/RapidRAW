use anyhow::{Result, anyhow};
use image::DynamicImage;
use rawler::{
    decoders::RawDecodeParams,
    rawsource::RawSource,
};

/// Extract the CFA (Color Filter Array) pattern name from a NEF file.
///
/// Uses rawler to get the camera's CFA pattern description (e.g., "RGGB").
pub fn get_cfa_pattern(file_bytes: &[u8]) -> Result<String> {
    let source = RawSource::new_from_slice(file_bytes);
    let decoder =
        rawler::get_decoder(&source).map_err(|e| anyhow!("Failed to get decoder: {}", e))?;
    let raw_image = decoder
        .raw_image(&source, &RawDecodeParams::default(), false)
        .map_err(|e| anyhow!("Failed to decode raw image: {}", e))?;

    Ok(raw_image.camera.cfa.name.clone())
}

/// Get basic CFA color at a given position.
/// Returns 0 for R, 1 for G, 2 for B.
pub fn get_cfa_color_at(file_bytes: &[u8], x: usize, y: usize) -> Result<usize> {
    let source = RawSource::new_from_slice(file_bytes);
    let decoder =
        rawler::get_decoder(&source).map_err(|e| anyhow!("Failed to get decoder: {}", e))?;
    let raw_image = decoder
        .raw_image(&source, &RawDecodeParams::default(), false)
        .map_err(|e| anyhow!("Failed to decode raw image: {}", e))?;

    Ok(raw_image.camera.cfa.color_at(y, x))
}

/// Convert a CFA-pattern frame (in DynamicImage) to a color-coded visualization
/// showing which pixel is which Bayer color.
///
/// This is useful for debugging the CFA extraction pipeline.
pub fn visualize_cfa_from_image(image: &DynamicImage, cfa_pattern: &str) -> DynamicImage {
    use image::{ImageBuffer, Rgba, RgbaImage};

    let (width, height) = (image.width(), image.height());
    let mut output: RgbaImage = ImageBuffer::new(width, height);
    let rgba = image.to_rgba8();

    for y in 0..height {
        for x in 0..width {
            let pixel = rgba.get_pixel(x, y);
            let lum = (pixel[0] as f32 * 0.299
                + pixel[1] as f32 * 0.587
                + pixel[2] as f32 * 0.114) as u8;

            let color = match cfa_pattern {
                "RGGB" => match (y % 2, x % 2) {
                    (0, 0) => Rgba([lum, 0, 0, 255]),     // R
                    (0, 1) => Rgba([0, lum, 0, 255]),     // G
                    (1, 0) => Rgba([0, lum, 0, 255]),     // G
                    (1, 1) => Rgba([0, 0, lum, 255]),     // B
                    _ => Rgba([lum, lum, lum, 255]),
                },
                _ => Rgba([lum, lum, lum, 255]),
            };
            output.put_pixel(x, y, color);
        }
    }

    DynamicImage::ImageRgba8(output)
}

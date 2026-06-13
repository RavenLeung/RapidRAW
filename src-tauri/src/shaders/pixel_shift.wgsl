// Pixel-Shift Steering Kernel Regression Fusion — GPU Compute Shader
//
// Fuses multiple aligned pixel-shift frames into a single high-quality image
// using structure-adaptive anisotropic steering kernels (Wronski et al. 2019).
//
// Pipeline:
//   1. structure_tensor_pass: Compute gradient covariance from reference frame
//   2. fusion_pass: Steering kernel regression across all frames

// ─── Constants ───
const WORKGROUP_SIZE: u32 = 8u;
const MAX_FRAMES: u32 = 32u;
const MAX_KERNEL_RADIUS: u32 = 8u;
const PI: f32 = 3.141592653589793;

// ─── Uniform / Storage Buffer Types ───

struct StructureTensor {
    ixx: f32,
    ixy: f32,
    iyy: f32,
}

struct EigenDecomp {
    e1: f32,
    e2: f32,
    angle: f32,
}

struct FusionParams {
    output_width: u32,
    output_height: u32,
    input_width: u32,
    input_height: u32,
    num_frames: u32,
    kernel_sigma: f32,
    stretch: f32,
    structure_sigma: f32,
    motion_compensation: u32,  // bool
    _pad0: u32,
    _pad1: u32,
}

// ─── Group 0: Structure Tensor Pass ───
// Input: reference frame (as vec3<f32> per pixel)
// Output: structure tensor per pixel (ixx, ixy, iyy)

@group(0) @binding(0) var<storage, read> ref_frame: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> structure_tensors: array<StructureTensor>;
@group(0) @binding(2) var<uniform> params: FusionParams;

// Sobel gradient of luminance at (x, y)
fn compute_luminance_gradient(x: u32, y: u32, w: u32, h: u32) -> vec2<f32> {
    if x < 1u || y < 1u || x >= w - 1u || y >= h - 1u {
        return vec2(0.0);
    }

    let idx_center = y * w + x;
    let idx_left   = y * w + (x - 1u);
    let idx_right  = y * w + (x + 1u);
    let idx_up     = (y - 1u) * w + x;
    let idx_down   = (y + 1u) * w + x;

    let l_left  = dot(ref_frame[idx_left],  vec3(0.2126, 0.7152, 0.0722));
    let l_right = dot(ref_frame[idx_right], vec3(0.2126, 0.7152, 0.0722));
    let l_up    = dot(ref_frame[idx_up],   vec3(0.2126, 0.7152, 0.0722));
    let l_down  = dot(ref_frame[idx_down],  vec3(0.2126, 0.7152, 0.0722));

    return vec2(l_right - l_left, l_down - l_up);
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn structure_tensor_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if x >= params.input_width || y >= params.input_height {
        return;
    }

    let radius = u32(ceil(params.structure_sigma * 2.0));
    let r = i32(radius);

    var ixx: f32 = 0.0;
    var ixy: f32 = 0.0;
    var iyy: f32 = 0.0;
    var count: u32 = 0u;

    for (var dy: i32 = -r; dy <= r; dy++) {
        for (var dx: i32 = -r; dx <= r; dx++) {
            let sx = i32(x) + dx;
            let sy = i32(y) + dy;

            if sx < 1 || sy < 1 || sx >= i32(params.input_width) - 1 || sy >= i32(params.input_height) - 1 {
                continue;
            }

            let g = compute_luminance_gradient(u32(sx), u32(sy), params.input_width, params.input_height);

            ixx += g.x * g.x;
            ixy += g.x * g.y;
            iyy += g.y * g.y;
            count += 1u;
        }
    }

    if count > 0u {
        ixx /= f32(count);
        ixy /= f32(count);
        iyy /= f32(count);
    }

    let idx = y * params.input_width + x;
    structure_tensors[idx] = StructureTensor(ixx, ixy, iyy);
}

// ─── Group 1: Fusion Pass ───
// Input: all aligned frames, structure tensors, motion mask (optional)
// Output: fused RGB image

@group(1) @binding(0) var<storage, read> all_frames: array<vec3<f32>>;   // packed: frame0[p0], frame1[p0], ... or separate per-frame buffers
@group(1) @binding(1) var<storage, read> tensors: array<StructureTensor>;
@group(1) @binding(2) var<storage, read> motion_mask: array<f32>;        // per-pixel [0,1]
@group(1) @binding(3) var<storage, read_write> output_rgb: array<vec4<f32>>;
@group(1) @binding(4) var<uniform> fusion_params: FusionParams;

// SVD of 2x2 symmetric matrix [a b; b c] → eigenvalues e1 >= e2, angle
fn svd_2x2(a: f32, b: f32, c: f32) -> EigenDecomp {
    let trace = a + c;
    let det = a * c - b * b;

    if abs(trace) < 1e-10 {
        return EigenDecomp(0.0, 0.0, 0.0);
    }

    let disc = sqrt((a - c) * (a - c) + 4.0 * b * b);
    var e1 = (trace + disc) * 0.5;
    var e2 = (trace - disc) * 0.5;

    e1 = max(e1, 0.0);
    e2 = max(e2, 0.0);
    e2 = min(e2, e1);

    var angle: f32;
    if abs(b) > 1e-10 {
        angle = atan2(e1 - a, b);
    } else if a >= c {
        angle = 0.0;
    } else {
        angle = PI * 0.5;
    }

    return EigenDecomp(e1, e2, angle);
}

// Get frame pixel at (x, y) with bilinear interpolation
fn sample_frame(frame_idx: u32, x: f32, y: f32, w: u32, h: u32) -> vec3<f32> {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = x - f32(x0);
    let fy = y - f32(y0);

    let clamp_x = |cx: i32| -> u32 { return u32(clamp(cx, 0, i32(w) - 1)); };
    let clamp_y = |cy: i32| -> u32 { return u32(clamp(cy, 0, i32(h) - 1)); };

    let stride = w;
    let base = frame_idx * (w * h);

    let idx00 = base + clamp_y(y0) * stride + clamp_x(x0);
    let idx10 = base + clamp_y(y0) * stride + clamp_x(x0 + 1);
    let idx01 = base + clamp_y(y0 + 1) * stride + clamp_x(x0);
    let idx11 = base + clamp_y(y0 + 1) * stride + clamp_x(x0 + 1);

    let p00 = all_frames[idx00];
    let p10 = all_frames[idx10];
    let p01 = all_frames[idx01];
    let p11 = all_frames[idx11];

    let top = p00 * (1.0 - fx) + p10 * fx;
    let bottom = p01 * (1.0 - fx) + p11 * fx;

    return top * (1.0 - fy) + bottom * fy;
}

// Get motion mask value at integer pixel coord
fn get_motion_weight(x: u32, y: u32, w: u32, h: u32) -> f32 {
    if fusion_params.motion_compensation == 0u {
        return 1.0;
    }
    let cx = clamp(x, 0u, w - 1u);
    let cy = clamp(y, 0u, h - 1u);
    return motion_mask[cy * w + cx];
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn fusion_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;

    if ox >= fusion_params.output_width || oy >= fusion_params.output_height {
        return;
    }

    // Map output pixel to reference frame coords
    let rx = f32(ox);  // 1:1 for now
    let ry = f32(oy);

    // Get structure tensor at nearest integer pixel
    let ix = clamp(u32(round(rx)), 0u, fusion_params.input_width - 1u);
    let iy = clamp(u32(round(ry)), 0u, fusion_params.input_height - 1u);
    let tensor = tensors[iy * fusion_params.input_width + ix];

    // Eigen decomposition
    let eigen = svd_2x2(tensor.ixx, tensor.ixy, tensor.iyy);

    // Anisotropy factor
    let eps = 1e-6;
    let e1 = max(eigen.e1, eps);
    let e2 = max(eigen.e2, eps);
    let anisotropy = clamp((e1 - e2) / (e1 + e2), 0.0, 1.0);

    // Kernel radii
    let r1 = fusion_params.kernel_sigma * (1.0 + fusion_params.stretch * anisotropy);
    let r2 = fusion_params.kernel_sigma / max(1.0 + fusion_params.stretch * anisotropy, eps);
    let max_radius = u32(ceil(max(r1, r2) * 3.0));
    let sr = i32(clamp(max_radius, 2u, MAX_KERNEL_RADIUS));

    let cos_a = cos(eigen.angle);
    let sin_a = sin(eigen.angle);

    var sum_rgb = vec3(0.0);
    var total_weight: f32 = 0.0;

    let num_f = fusion_params.num_frames;

    // Gather weighted samples from all frames
    for (var fi: u32 = 0u; fi < num_f; fi++) {
        for (var dy: i32 = -sr; dy <= sr; dy++) {
            for (var dx: i32 = -sr; dx <= sr; dx++) {
                let sx = rx + f32(dx);
                let sy = ry + f32(dy);

                if sx < 0.0 || sy < 0.0 || sx >= f32(fusion_params.input_width) - 1.0 ||
                   sy >= f32(fusion_params.input_height) - 1.0 {
                    continue;
                }

                // Rotate offset into kernel coordinate system
                let rot_dx = f32(dx) * cos_a + f32(dy) * sin_a;
                let rot_dy = -f32(dx) * sin_a + f32(dy) * cos_a;

                // Anisotropic Gaussian weight
                let spatial_w = exp(-0.5 * (rot_dx * rot_dx / (r1 * r1) + rot_dy * rot_dy / (r2 * r2)));
                // Normalize by kernel volume
                let kernel_w = spatial_w / (2.0 * PI * r1 * r2);

                if kernel_w < 1e-6 {
                    continue;
                }

                // Motion weight
                let motion_w = get_motion_weight(u32(round(sx)), u32(round(sy)),
                                                fusion_params.input_width, fusion_params.input_height);

                if motion_w < 1e-6 {
                    continue;
                }

                let w = kernel_w * motion_w;

                // Sample frame
                let sample_rgb = sample_frame(fi, sx, sy, fusion_params.input_width, fusion_params.input_height);

                sum_rgb += sample_rgb * w;
                total_weight += w;
            }
        }
    }

    if total_weight < 1e-10 {
        // Fallback: use reference frame
        let idx = iy * fusion_params.input_width + ix;
        let ref_rgb = all_frames[idx];  // frame 0 is at base of all_frames
        sum_rgb = ref_rgb;
        total_weight = 1.0;
    }

    let out_idx = oy * fusion_params.output_width + ox;
    output_rgb[out_idx] = vec4(sum_rgb / total_weight, 1.0);
}

# Pixel-Shift 合并 — 后续改进路线图

> 基于 PARSEK 算法论文 (Dietz, EI 2024) 和 Handheld-MF-SR (Wronski et al. 2019)

---

## 当前状态 (已完成)

| 模块 | 文件 | 状态 |
|------|------|:--:|
| Nikon 元数据解析 | `pixel_shift/metadata.rs` | ✅ |
| NEF CFA 原始提取 | `pixel_shift/cfa_fusion.rs` | ✅ |
| RGB 平均/中值合并 | `pixel_shift/basic_merge.rs` | ✅ |
| 子像素对齐 (相位相关 + LK) | `pixel_shift/alignment.rs` | ✅ |
| 运动蒙版检测 | `pixel_shift/motion_detection.rs` | ✅ |
| 转向核回归融合 (SKR) | `pixel_shift/skr_fusion.rs` | ✅ |
| GPU 加速 compute shader | `shaders/pixel_shift.wgsl` + `gpu_fusion.rs` | ✅ |
| 前端 Modal | `PixelShiftModal.tsx` | ✅ |
| 上下文菜单入口 | `useAppContextMenus.ts` | ✅ |
| i18n (en + zh-CN) | `locales/` | ✅ |

**当前问题**：
- CFA 模式效果不佳，因为：
  1. 假设了理想偏移（不信任实际测量）
  2. 没有帧间对齐（实际三脚架也有微振动）
  3. 没有噪声模型和置信度权重
  4. 简单的空间采样而非概率融合

---

## PARSEK 核心算法分析

### 一句话总结

> 不信任相机的标称偏移，自己用 `findTransformECC` 测量实际位移，然后用基于 2D 直方图噪声模型的置信度权重融合每个像素。

### 关键发现

- **实际偏移 ≠ 理想偏移**：三脚架上理想 (±0.5, ±0.5) 像素步进，实测偏移在 ±0.05 ~ ±1.9 像素范围，X 轴通常大于 Y 轴，有可忽略的旋转
- **用实际测量偏移比假设偏移效果更好**
- **置信度权重 > 简单平均**：自建噪声模型让融合在噪声上更聪明

### 算法管线

```
① 读取所有帧 → 转灰度
② findTransformECC (欧几里得模式) 计算帧间实际位移
③ 剔除旋转/缩放超阈值的帧 (默认 0.01)
④ 重新读取原生格式 → 转 16-bit 单色/CFA
⑤ 构建 2D 直方图噪声模型 (256×256 per channel)
⑥ 超分辨率重建:
    对每个输出像素:
      getvc() 从每帧收集 (值, 置信度)
      空间距离置信度: 1/dist²
      第一帧过滤/修剪
      异常值过滤/修剪 (基于直方图一致性)
      置信度加权平均
⑦ 可选: getneigh() 邻域滤波第二遍
⑧ 输出 16-bit PNG
```

### 置信度的三个来源

| 来源 | 计算 | 作用 |
|------|------|------|
| 空间距离 | `1/dist²` (最大 2px) | 采样位置越近权重越高 |
| 噪声模型 | 2D 直方图 `P(v1\|v2)` | 两个值是否来自同场景的概率 |
| 帧间一致性 | 与参考帧/中值的差异 | 排除运动/异常像素 |

### 噪声模型构建

```
对每对帧匹配的像素对 (v1, v2):
  histogram[high8(v1)][high8(v2)][channel]++

归一化: 每行除以对角线值 → 累积 → 取幂
查询: hcon(ref_val, sample_val, channel) → [0,1]
```

---

## 改进路线图

### Phase A: 替换对齐 —— 用实际测量替代假设偏移

**目标**：用 OpenCV `findTransformECC` 或等价的相位相关计算每帧的实际位移

**Rust 实现路径**：
- `imageproc::registration` 或手动实现增强相关系数 (ECC) 对齐
- 或者使用 nalgebra 实现 ECC（不需要 OpenCV 依赖）
- 替代方案：用现有的 `alignment.rs` 相位相关 + LK 细化（已有基础设施）

**文件**：`pixel_shift/alignment.rs`（增强）

**验收标准**：打印每帧实测位移，与理想 (0.5, 0.5) 网格对比偏差值

---

### Phase B: 噪声模型 —— 2D 直方图

**目标**：构建 256×256×3 的像素值联合分布模型

**实现**：
```rust
struct NoiseModel {
    /// hist[R][v1_high8][v2_high8] → count
    histograms: [[[u32; 256]; 256]; 3],
}

impl NoiseModel {
    fn build(frames: &[CfaFrame], alignments: &[Transform]) -> Self;
    fn normalize(&mut self, hist_exp: f32);
    fn confidence(&self, ref_val: u16, sample_val: u16, channel: usize) -> f32;
}
```

**文件**：`pixel_shift/noise_model.rs`（新建）

**验收标准**：可导出噪声模型为 256×256 RGB 图像，可视化验证

---

### Phase C: 置信度融合 —— 替换简单平均/中值

**目标**：每个输出像素用置信度加权融合

**实现**：
```rust
struct ConfidenceVote {
    value: f32,      // 像素值
    confidence: f32,  // 综合置信度 [0, 1]
}

fn fuse_pixel_with_confidence(
    votes: &[ConfidenceVote],
    ref_filter: f32,     // 第一帧过滤强度
    outlier_filter: f32, // 异常值过滤强度
) -> (f32, f32, f32);
```

**三种过滤模式**：
1. `ref_filter > 0`：放大与参考帧一致的样本置信度
2. `ref_filter < 0`：剔除与参考帧差异超阈值的样本
3. `outlier_filter != 0`：基于全体样本中值的异常值检测

**文件**：`pixel_shift/confidence_fusion.rs`（新建）

**验收标准**：与 PARSEK 输出对比，PSNR 差异 < 1dB

---

### Phase D: 超分辨率控制

**目标**：用户可选输出分辨率（1× / 2× / 3×）

**实现**：
- `-X 2 -Y 2` → 2× 超分辨率 (96MP)
- `-X 4 -Y 4` → 4× 超分辨率 (384MP，实验性)
- 前端添加分辨率选择器

**文件**：`PixelShiftModal.tsx`（UI）+ `cfa_fusion.rs`（后端）

---

### Phase E: 非 pixel-shift 手持 burst 支持

**目标**：即使用户没有 pixel-shift 相机，也能用手持连拍实现超分辨率

**区别**：
- 手持偏移随机而非受控（依赖 IBIS/手抖）
- 每帧偏移量未知，完全依赖测量对齐
- 可能需要更强的运动补偿

**实现**：放宽 CFA 模式的文件类型限制，接受任意 NEF/RAW burst

---

## 优先级建议

| Phase | 内容 | 难度 | 影响力 |
|:--|------|:--:|:--:|
| **A** | 实测对齐替代假设偏移 | 中 | 🔴 关键 |
| **B** | 2D 直方图噪声模型 | 中 | 🔴 关键 |
| **C** | 置信度融合 | 中 | 🔴 关键 |
| D | 超分辨率控制 | 低 | 🟡 增强 |
| E | 手持 burst 支持 | 高 | 🟢 扩展 |

**建议顺序**: A → B → C → D → E

A+B+C 三个 Phase 完成后，CFA 模式的输出质量应该接近或达到 PARSEK 水平。

---

## 参考资源

| 资源 | 链接 |
|------|------|
| PARSEK 源码 (单文件 C++) | http://aggregate.org/DIT/PARSEK/parsek.cpp |
| PARSEK 论文 (EI 2024) | https://doi.org/10.2352/EI.2024.36.15.COIMG-142 |
| PARSEK 幻灯片 | http://aggregate.org/DIT/PARSEK/parsekslides.pdf |
| Handheld MF-SR (IPOL 2023) | https://doi.org/10.5201/ipol.2023.460 |
| Handheld MF-SR 源码 | https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution |
| Wronski et al. 论文 | https://research.google/pubs/handheld-multi-frame-super-resolution/ |
| DPReview 讨论 | https://www.dpreview.com/forums/post/65515977 |

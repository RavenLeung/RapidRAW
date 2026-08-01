<p align="center">
  <img src="https://raw.githubusercontent.com/CyberTimon/RapidRAW/assets/.github/assets/editor.jpg" alt="RapidRAW 编辑器">
</p>

<div align="center">

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![wgpu](https://img.shields.io/badge/wgpu-%23282C34.svg?style=for-the-badge&logo=webgpu&logoColor=white)](https://wgpu.rs/)
[![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)](https://react.dev/)
[![Tauri](https://img.shields.io/badge/Tauri-24C8DB?style=for-the-badge&logo=tauri&logoColor=white)](https://tauri.app/)
[![AGPL-3.0](https://img.shields.io/badge/License-AGPL_v3-blue.svg?style=for-the-badge)](https://opensource.org/licenses/AGPL-3.0)
[![GitHub stars](https://img.shields.io/github/stars/CyberTimon/RapidRAW?style=for-the-badge&logo=github&label=Stars)](https://github.com/CyberTimon/RapidRAW/stargazers)

</div>

# RapidRAW

> 一款美观、非破坏性、GPU 加速的 RAW 图像编辑器，为性能而生。

**RapidRAW** 是 Adobe Lightroom® 的现代高性能替代品，提供简洁、优雅的编辑体验，安装包体积不到 **20MB**，支持 Windows、macOS、Linux 和 Android。

该项目始于作者 18 岁时的一场个人挑战：为自己摄影工作流打造一款高性能工具，同时深入理解 React、WGSL 与 Rust。

## 项目定位

RapidRAW 面向喜爱**干净、快速、简单工作流**的摄影师，优先保证速度、美观的界面和强大的工具，让你快速实现创意调色。

> ⚠️ **注意**：RapidRAW 仍处于积极开发阶段，尚未达到 Darktable、RawTherapee 或 Adobe Lightroom® 等成熟工具的完善程度。当前重点是构建快速、愉悦的核心编辑体验，你可能会遇到 Bug —— 欢迎提交 Issue 反馈！

## ✨ 核心特性

### 核心编辑引擎

- **GPU 加速**：完整的 32 位图像处理管线，使用 WGSL 编写，实时反馈零等待
- **分层蒙版**：AI 主体、深度、天空、前景检测 + 参数化色彩与亮度蒙版
- **修饰工具**：局部克隆（Clone）与修复（Heal）工具，去除灰尘和瑕疵
- **生成式编辑**：通过文字提示移除或添加元素（可选 AI 后端）
- **完整 RAW 支持**：通过 rawler 支持广泛的相机 RAW 格式（含 JPEG）
- **非破坏性工作流**：所有编辑保存在 `.rrdata` 副文件中，原始照片永不改动
- **镜头校正**：基于 Lensfun 自动校正畸变、色差与暗角

### 图库与工作流

- **图像图库**：轻松管理你的照片收藏
- **筛选视图（Culling）**：最多 6 张图并排对比，支持星级评分、颜色标签与元数据
- **组织管理**：递归文件夹视图、虚拟副本、颜色标签、星级与自定义标签
- **文件操作**：导入、复制、移动、重命名、重复图片/文件夹
- **胶片条视图**：编辑时快速切换当前文件夹内的图片
- **批量操作**：批量应用调整或批量导出
- **EXIF 与 CLI**：完整元数据查看器 + 无头 CLI 批量导出工具

### 专业级调整

- **色调控制**：曝光、色调映射（含 **AgX**）、对比度、高光、阴影、白场、黑场
- **色调曲线**：亮度/RGB 与参数化曲线的完全控制
- **色彩分级**：色温、色调、鲜艳度、饱和度、色轮、完整 HSL 色彩混合器
- **细节增强**：锐化、清晰度、质感、降噪（亮度 & 色彩）
- **创意效果**：镜头模糊（Bokeh）、LUT、去雾、暗角、辉光、光晕、镜头光晕、胶片颗粒
- **变换工具**：透视校正、旋转、拉直、裁剪、扭曲

### 生产力与界面

- **预设系统**：创建、保存、导入、分享，支持强度调节
- **图像分析**：实时矢量示波器、波形图、RGB 示波图与直方图
- **复制粘贴设置**：在图片间快速迁移调整与蒙版
- **撤销/重做历史**：每一步编辑都有完整历史记录
- **合成与合并**：包围曝光 HDR 合并、无缝全景拼接、拼贴制作、胶片负片转换
- **灵活导出**：JPEG、PNG、WebP、AVIF、TIFF、JXL、LUT，支持自定义水印与 EXIF 保留

## 🤖 AI 功能（三种使用方式）

RapidRAW 的 AI 功能设计灵活，可选择快速本地工具、强大的自托管或便捷的云服务：

### 1. 内置 AI 工具（本地 & 免费）

直接集成在 RapidRAW 中，完全在本地运行，快速免费、无需配置：

- **AI 蒙版**：即时检测并蒙版主体、天空和前景
- **自动标签**：使用本地 CLIP 模型自动为图库图片打上关键词标签
- **简易生成式替换**：基于 CPU 的修复工具，用于移除小干扰物

### 2. 自托管 ComfyUI 集成（本地 & 免费）

有强力 GPU 的用户可连接本地 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 服务器，由 [**RapidRAW-AI-Connector**](https://github.com/CyberTimon/RapidRAW-AI-Connector) 中间件桥接。该架构只传输微小蒙版和文本，而非整张高分辨率图片，效率极高：

- **完全控制**：使用自己的硬件与任意自定义扩散模型或工作流
- **零成本**：利用现有硬件进行高级生成式编辑
- **自定义工作流**：导入自己的 ComfyUI 工作流与自定义节点

### 3. 可选云服务（订阅）

> 作者承诺**不会把功能锁在付费墙后** —— 使用内置工具或自托管即可免费享受全部功能。

云服务纯粹是便利选项，提供与自托管相同的高质量结果，无需任何配置，登录即用（即将推出）。

| 特性 | 内置 AI（免费） | 自托管 ComfyUI | 云服务 |
| --- | --- | --- | --- |
| **费用** | 免费，已包含 | 免费（需自备硬件） | 待定 / 月 |
| **配置** | 无需 | 手动配置 ComfyUI / AI Connector | 无需（登录即用） |
| **适用场景** | 日常工作流加速 | 技术用户完全控制 | 最大便利 |
| **状态** | ✅ 可用 | ✅ 可用 | 🚧 即将推出 |

## 🚀 快速开始

### 方式一：下载最新版本（推荐）

**Windows & macOS：** 从 [Releases](https://github.com/CyberTimon/RapidRAW/releases) 页面下载对应安装包。

**Linux：**
- 官方 Flatpak 包（支持所有发行版）：[Flathub](https://flathub.org/apps/io.github.CyberTimon.RapidRAW)
- Debian 系：使用 Releases 页面的 `.deb` 包
- Arch 系：使用 AUR 的 [`rapidraw-bin`](https://aur.archlinux.org/packages/rapidraw-bin) 包

### 方式二：从源码构建

需要先安装 [Rust](https://www.rust-lang.org/tools/install) 和 [Node.js](https://nodejs.org/)：

```bash
# 1. 克隆仓库
git clone https://github.com/CyberTimon/RapidRAW.git
cd RapidRAW

# 2. 安装前端依赖
npm install

# 3. 构建并运行（开发模式）
npm start
```

构建发布版本：

```bash
# 1. 生成发布构建
npm run tauri build

# 2. 运行发布版本
./src-tauri/target/release/RapidRAW
```

## 💻 命令行界面（CLI）

RapidRAW 内置无头导出工具，可在自动化脚本、终端管道或服务器环境中批量处理照片，无需打开图形界面：

```bash
# 导出整个文件夹（自动应用 .rrdata 副文件中的编辑）
rapidraw export /path/to/photos --output /path/to/output_dir --format jpeg --quality 90

# 将单张图片导出为指定文件
rapidraw export /path/to/photo.raw --output /path/to/output.png --format png

# 用自定义 JSON 调整文件覆盖副文件，批量导出
rapidraw export /path/to/photos --output /path/to/output_dir --adjustments /path/to/preset.json
```

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `<source>` | 图片文件或包含图片的目录路径 | （必填） |
| `--output <path>` | 目标目录或具体输出文件路径 | （必填） |
| `--format <fmt>` | 输出格式（`jpeg`、`png`、`webp`、`avif`、`tiff`、`jxl`、`cube`） | `jpeg` |
| `--quality <1-100>` | 导出质量 | `90` |
| `--keep-metadata` | 保留 EXIF/拍摄元数据 | `false` |
| `--adjustments <path>` | 自定义 JSON 调整文件（覆盖副文件） | （自动检测） |

> **提示**：默认情况下，无头导出会自动检测并应用源图片旁 `.rrdata` 副文件中的编辑；使用 `--adjustments` 可对所有导出图片统一覆盖调整。

## 🖥️ 系统要求

RapidRAW 轻量且跨平台，最低（已测试）要求：

**操作系统：**
- **Windows**：Windows 10 或更新
- **macOS**：macOS 13（Ventura）或更新
- **Linux**：Ubuntu 22.04+ 或兼容的现代发行版

**硬件建议：**
- **内存**：强烈建议 **16GB 以上**。应用在更低内存下也可运行，但处理高分辨率 RAW、撤销历史与复杂图层蒙版时，16GB+ 才能保证流畅
- **显卡**：推荐独立 GPU。RapidRAW 的处理管线重度依赖 GPU 加速，2015 年前的旧显卡架构或老集成显卡可能不稳定或出现图形伪影

### 常见问题

<details>
<summary>打开图片/进入编辑模式时崩溃</summary>

这通常是 GPU 后端自动选择问题：

1. 在**主屏幕**打开**设置**（齿轮图标）
2. 进入**处理（Processing）**标签页
3. 找到**处理后端（Processing Backend）**设置
4. 从**自动（Auto）**改为操作系统支持的特定后端（如 **Vulkan**、**DirectX12**、**OpenGL** 或 **Metal**）
5. 重启应用后重试，可尝试不同后端

</details>

<details>
<summary>Linux Wayland/WebKit 崩溃</summary>

在 Wayland 环境（如 GNOME + NVIDIA）下崩溃时，尝试以下方式启动：

```bash
WEBKIT_DISABLE_DMABUF_RENDERER=1 RapidRAW
```

或

```bash
WEBKIT_DISABLE_COMPOSITING_MODE=1 RapidRAW
```

该问题与 **WebKit** 和 **NVIDIA 驱动**有关，并非 RapidRAW 本身。切换到 **X11** 或使用 **AMD / Intel** 显卡也可能有帮助。

</details>

## 🛠️ 技术栈

| 层 | 技术 |
| --- | --- |
| **桌面框架** | [Tauri 2](https://tauri.app/) |
| **后端语言** | [Rust](https://www.rust-lang.org/) |
| **GPU 渲染** | [wgpu](https://wgpu.rs/) + 自定义 WGSL 着色器 |
| **前端** | [React 19](https://react.dev/) + TypeScript |
| **样式** | Tailwind CSS 4 |
| **状态管理** | Zustand |
| **RAW 解码** | [rawler](https://github.com/dnglab/dnglab/tree/main/rawler) |

## 📁 项目结构

```
RapidRAW/
├── src/                  # 前端（React + TypeScript）
│   ├── components/       # UI 组件
│   │   ├── adjustments/  # 基础、色彩、曲线、细节、效果面板
│   │   ├── panel/        # 编辑器、图库、胶片条等面板
│   │   ├── ui/           # 通用 UI 组件
│   │   └── views/        # 编辑器视图、图库视图
│   ├── i18n/             # 多语言支持
│   ├── store/            # Zustand 状态管理
│   └── utils/            # 工具函数
├── src-tauri/            # 后端（Rust）
│   ├── src/              # 图像处理、AI、导出等模块
│   └── shaders/          # WGSL GPU 着色器
├── bench/                # 性能基准
└── packaging/            # 打包配置
```

## 🤝 贡献

欢迎任何形式的贡献 —— 报告 Bug、建议新特性或提交 Pull Request！请直接打开 Issue 或分享你的想法。

**图片格式问题**：如果相机 RAW 格式不受支持，请先到 [rawler 仓库](https://github.com/dnglab/dnglab/issues) 提交 Issue，rawler 支持后再到 RapidRAW 创建 Issue 以便同步更新。

## 💖 特别感谢

以下项目与工具对 RapidRAW 的开发至关重要：

- **[Google AI Studio](https://aistudio.google.com)**：为图像处理算法的研究与实现提供了极大帮助
- **[rawler](https://github.com/dnglab/dnglab/tree/main/rawler)**：提供 RAW 文件处理的 Rust crate 基础
- **[lensfun](https://lensfun.github.io/)**：开源镜头校正库与数据库
- **[LaMa](https://github.com/advimman/lama)**：内容感知填充与物体移除的修复模型
- **[SAM 2](https://github.com/facebookresearch/sam2)**：AI 主体检测的基石模型
- **[U-2-Net](https://github.com/xuebinqin/U-2-Net)**：AI 天空与前景检测架构
- **[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)**：AI 深度蒙版的单目深度估计模型
- **[nind-denoise](https://github.com/trougnouf/nind-denoise)**：AI 降噪模型
- **[NegPy](https://github.com/marcinz606/NegPy)**：胶片反转数学思路的灵感来源
- **[pixls.us](https://discuss.pixls.us/)**：提供灵感、建议与想法的社区
- **[darktable](https://github.com/darktable-org/darktable)**：部分参考实现
- **你**：使用与支持 RapidRAW，让项目保持活力

## 📜 许可证与理念

本项目采用 **GNU Affero General Public License v3.0（AGPL-3.0）** 许可证。选择该许可证是为了确保 RapidRAW 及其衍生作品始终开源免费，防止被闭源商业软件使用，让改进惠及所有人。

详见 [LICENSE](LICENSE) 文件。

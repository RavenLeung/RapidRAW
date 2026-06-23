# 附屏显示 (Reference Display) 功能计划

**分支**: `feat/reference-display`
**目标**: 在第二个显示器上全屏显示当前编辑的照片（无UI干扰）

---

## 背景

RapidRAW 是一个 RAW 图像编辑器（Tauri v2 + React + TypeScript）。
编辑照片时经常需要将照片放大到全屏参考，第二个显示器尤其适合此用途。
当前项目完全没有多窗口 / 附屏显示相关代码，需要从零实现。

---

## 核心技术方案

### 图片传递

```
Rust 后端 → JPEG buffer → Blob URL (主窗口)  
                                ↓
                           fetch → blob → base64
                                ↓
                     Tauri 事件: reference:update-image
                                ↓
                      参考窗口 <img> 显示
```

- `finalPreviewUrl` 是 Blob URL (窗口隔离的)，需要用 `fetch` 转成 base64 data URL 再通过 Tauri 事件发送
- 预览分辨率 ~2000px，JPEG ~300KB → base64 ~400KB，IPC 传这个量足够了
- 调整参数拖动时**节流 200ms**推送，松手时推送最终版

### 窗口架构

```
主窗口 (main)                     参考窗口 (reference)
  ┌─────────────────┐              ┌──────────────────┐
  │  App.tsx         │              │  App.tsx         │
  │  (正常渲染)      │              │  (?view=reference)│
  │                  │              │                  │
  │  useReference    │  Tauri IPC  │  ReferenceViewer  │
  │  Display Hook ──►│◄────────────┤  (极简全屏)      │
  │                  │  事件通信    │                  │
  └─────────────────┘              └──────────────────┘
```

- 两个窗口加载同一个前端 build，靠 URL 参数 `?view=reference` 区分
- Tauri v2 的 `WebviewWindow` API 动态创建参考窗口

---

## 实施步骤

### 阶段 1 — Tauri 配置

**文件**: `src-tauri/capabilities/default.json`

添加权限:
- `core:webview:allow-create-webview-window`
- `core:window:allow-set-size`
- `core:window:allow-set-position`
- `core:window:allow-set-fullscreen`
- `core:event:default`

### 阶段 2 — ReferenceViewer 组件

**新建**: `src/components/views/ReferenceViewer.tsx`

- 纯黑背景，全屏
- `<img>` 标签，`object-fit: contain`，居中
- 监听 `reference:update-image` 事件
- 监听 `reference:close` 事件 → 关闭窗口
- 按 Esc 关闭
- 无工具栏、无UI、无交互

### 阶段 3 — App 路由分流

**修改**: `src/App.tsx`

在组件顶部:
```tsx
if (window.location.search.includes('view=reference')) {
  return <ReferenceViewer />;
}
```

### 阶段 4 — useReferenceDisplay Hook

**新建**: `src/hooks/useReferenceDisplay.ts`

核心函数:

| 函数 | 作用 |
|------|------|
| `toggleReferenceWindow()` | 开/关参考窗口 |
| `openReferenceWindow()` | 创建 WebviewWindow + 推送首帧 |
| `closeReferenceWindow()` | 关闭窗口 + 清理 |
| `updateReferenceImage()` | 转 blob→base64→emit |

辅助逻辑:
- `useEffect` 监听 `selectedImage` / `finalPreviewUrl` 变化
- 节流推送 (拖拽时 200ms)
- 主窗口 `unload` 时自动关参考窗口
- 跟踪窗口是否还开着 (`refWindowRef`)

### 阶段 5 — UI 按钮

**修改**: `src/components/panel/editor/EditorToolbar.tsx`

- 加 lucide-react 的 `Monitor` 图标
- 点击调用 `toggleReferenceWindow()`
- 激活状态高亮
- 无图片时禁用按钮

### 阶段 6 — i18n 翻译

**修改**: `src/i18n/` 下各语言的 translation 文件

| Key | 英文 | 中文 |
|-----|------|------|
| `referenceView.title` | Reference Display | 附屏显示 |
| `referenceView.open` | Open Reference | 打开附屏 |
| `referenceView.close` | Close Reference | 关闭附屏 |

---

## 边界情况

| 场景 | 处理 |
|------|------|
| 未选图片时 | 按钮禁用 |
| 用户手动关参考窗 | 主窗口同步按钮状态 (监听窗口 close 事件) |
| 主窗口关闭 | 自动关参考窗 |
| 切换图片 | 自动推送新图 |
| 回到图库 (无图片) | 自动关参考窗 |
| 拖拽滑块调参数 | 节流 200ms，松手推最终版 |
| 第二屏不存在 | 不特殊处理，用户自己拖窗口过去 |

---

## 不需要修改的部分

- Rust 后端代码 (复用已有的 finalPreviewUrl)
- Vite 配置 (单 entry + URL 参数区分)
- Zustand store (纯 hook 实现，不侵入 store)
- ImageCanvas / Konva (参考窗只用 `<img>`)

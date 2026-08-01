# Universal Emoji Compressor

面向任意表情包数据集的通用压缩与索引预处理工具。它将散落的静态图、GIF 和单帧伪 GIF 规范化为分片存储，并生成带 OCR、视觉描述、标签和来源路径的 SQLite/JSONL 索引，为表情包搜索、管理、推荐和应用接入准备数据。

## 主要能力

- 递归批处理 JPG、PNG、WebP、GIF、BMP、TIFF、AVIF 等常见格式。
- 使用 SHA-256 去重、数据库自增 ID 和数字文件名；一个内容可关联多个原始路径。
- 输出长边只使用 64、96、128、160 四档，不放大小图。
- 原图 OCR 负责生成高精度文字索引；最终编码后的 160px 图作为“物理可保留文字”基准。
- 从 64px 开始对最终编码效果重新 OCR，文字保留率不足 80% 时逐档升到 96/128/160；160px 没有可靠文字时仅按复杂度选档。
- 无 OCR 或 160px 无文字时使用统一复杂度阈值 0.236 / 0.373 / 0.456，分别映射到 64 / 96 / 128 / 160。
- GIF 抽帧参与复杂度与 OCR，保留 loop 和动画总时长。
- 严格定位坏帧，容错时跳过坏帧并把时长转移到相邻有效帧；全部帧损坏才报错。
- 单帧伪 GIF 自动转成 WebP；普通静态图统一输出 WebP。
- 默认使用 Qwen3-VL-4B-Instruct 生成描述、情绪、物体和搜索标签；视觉阶段优先分析原图，缺失时回退到压缩图。
- SQLite FTS5 与 JSONL 双索引，保留完整 `original_paths` 来源追溯。
- 支持断点续跑，已完成内容不会重复处理。

## 为什么分成两个 AI 阶段

PaddleOCR 与 PyTorch 视觉模型在部分 Windows/CUDA 环境中会加载不同的 cuDNN 运行时。完整命令会把它们放在两个独立进程中顺序执行，避免 DLL 冲突；也可以先完成压缩与 OCR，之后再补视觉索引。

## 安装

要求 Python 3.10+。建议使用独立虚拟环境。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -e .
```

只做基础压缩无需 AI 依赖。启用 OCR：

```bash
pip install -e ".[ocr]"
```

Paddle 的 CPU/GPU 安装包与 CUDA 版本有关，请先按 PaddlePaddle 官方安装说明安装匹配的 `paddlepaddle` 或 `paddlepaddle-gpu`。

启用视觉索引：

```bash
pip install -e ".[vision]"
```

首次使用 Hugging Face 模型名时会联网下载；也可传入已下载的本地模型目录。

Qwen3-VL 需要 transformers>=4.57 和 qwen-vl-utils>=0.0.14。中国大陆网络可先通过 ModelScope 下载官方 Qwen/Qwen3-VL-4B-Instruct，再传入本地快照目录。

## 快速开始

### 完整流程

```bash
emoji-pipeline INPUT OUTPUT \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --device gpu:0
```

流程依次执行：压缩与 OCR、视觉索引、JSONL 导出。完整流程会自动把 INPUT 传给视觉阶段，优先使用原始素材。视觉模型需要主题提示时，可单独运行 emoji-index 并增加 --context。

### 先压缩，稍后补索引

```bash
emoji-compress INPUT OUTPUT --device gpu:0 --ocr-batch-size 8
emoji-index OUTPUT Qwen/Qwen3-VL-4B-Instruct --source-root INPUT --batch-size 2
emoji-export OUTPUT
```

### 无 AI 依赖，仅压缩

```bash
emoji-compress INPUT OUTPUT --no-ocr
```

无 OCR 时仍按视觉复杂度选择尺寸，不会全部固定为 160。

## 输入与来源分类

输入目录可以有任意层级。工具不强制推断角色分类，而是原样记录相对来源路径：

```text
INPUT/
├─ doro/
├─ 奶龙/
└─ BQB/
   ├─ 熊猫头/
   └─ 其他/
```

即使输出文件被规范化为数字 ID，也能从 `original_paths` 准确反查仓库和子目录。若源数据没有目录分类，可在索引完成后按视觉标签运行独立分类脚本。

## 输出结构

```text
OUTPUT/
├─ images/
│  ├─ 0000/000000000001.webp
│  └─ 0000/000000000002.gif
├─ index.sqlite
├─ index.jsonl
└─ vision_report.json
```

索引包含：

- 自增 `id`、SHA-256、规范化输出相对路径及全部原始路径；
- 原始/输出格式、尺寸、帧数、总时长和文件体积；
- 复杂度指标、尺寸选择原因和缩放后 OCR 保留率；
- OCR 全文、逐帧文字框及文字出现帧；
- 视觉描述、情绪、对象、标签和安全分类；
- GIF 恢复状态、跳过帧索引、原始/输出帧数与总时长；
- 每阶段耗时、状态和错误信息。

## 搜索示例

```sql
SELECT a.id, a.output_path, s.description, s.tags
FROM asset_search AS s
JOIN assets AS a ON a.id = s.asset_id
WHERE asset_search MATCH '生气 OR 熊猫头';
```

业务应用通常读取 `index.sqlite` 做本地搜索；数据管道和向量化任务可以逐行消费 `index.jsonl`。

## GIF 时序说明

工具把“播放速度不变”定义为输出动画总时长与原动画完全一致。坏帧被跳过时，其 duration 会累加到前一个有效帧；如果坏帧位于开头，则累加到下一个有效帧。缩小和量化可能让连续帧变得完全相同，Pillow 会合并这些帧，但合并后的 duration 仍保持相同，因此视觉播放速度不变。

GIF 使用独立局部调色板并关闭 Pillow `optimize`，避免局部调色板索引被错误重映射造成花屏。

## 性能参考

实际吞吐取决于 GIF 帧数、分辨率、OCR 命中率、磁盘和模型。在 RTX 5080 16GB 的混合验证集上：

- 压缩、复杂度分析与 OCR：约 2 张/秒；
- Qwen3-VL-4B 视觉索引：RTX 5080 单张小样约 0.20～0.23 张/秒；批处理吞吐以实际数据为准；
- 完整流程的主要瓶颈是视觉理解模型。

这些数字仅用于容量规划，不是性能承诺。

## 开发

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## 许可证

MIT。视觉模型、OCR 模型和输入素材仍分别受其自身许可证约束。

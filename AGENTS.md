# AGENTS.md

本文件为 Codex 及其他代码代理提供项目级上下文。除非用户另有要求，修改本项目时应遵循以下约定。

## 项目概览

- 本项目是基于 PyTorch、Ultralytics YOLO 和 Pillow/OpenCV 的目标检测实验项目。
- 主要流程包括：读取带 Pascal VOC XML 标注的数据集、划分数据、导出 YOLO 格式、训练检测器、执行推理以及评估召回率。
- Python 包目录目前命名为 `pipline/`。这是现有名称，不要擅自改为 `pipeline/`，否则会破坏现有导入。
- `main.py` 是当前流程入口，`scripts/` 提供实验命令和检查。

## 导师要求

1. **第一阶段生成候选区域。** 弱化类别区分，将所有缺陷统一作为候选目标（当前检测器使用 `defect`），优先保证召回率，尽可能完整地提取疑似缺陷区域，允许一定数量的误报。
2. **第二阶段识别类别并剔除误报。** 裁剪第一阶段输出的候选区域，训练包含 `n` 类缺陷和 1 类非缺陷（`non_defect`）的分类器，识别缺陷类别并剔除判为非缺陷的候选。
3. **实例分割结果。** 如果第一阶段同时输出实例掩膜，保留通过第二阶段分类筛选的候选对应的掩膜，形成最终的实例分割结果。

## 目录与关键文件

- `main.py`：加载 `data/Data10` 和 `data/Data11`，导出数据集并启动 YOLO 训练。
- `pipline/data.py`：数据读取、标注检查、数据划分、manifest 读写和裁剪导出。
- `pipline/export_yolo.py`：将记录导出为 Ultralytics YOLO 数据集格式。
- `pipline/detector.py`：YOLO 训练、推理和预测结果可视化。
- `pipline/evaluate.py`：IoU、覆盖率和召回率评估。
- `pipline/classifier.py`：分类数据加载、训练、权重加载和候选区域分类。
- `pipline/two_stage.py`：单图检测、候选筛选和分类。
- `scripts/`：检测和分类的训练、推理、评估、数据准备及检查命令。
- `requirements.txt`：Python 依赖；其中 PyTorch/torchvision 版本需要与目标机器的 CUDA 环境匹配。

## 运行环境

- 当前 macOS 机器用于编码、CPU 调试和轻量验证。
- 另一台机器运行 Ubuntu，配有 NVIDIA GeForce RTX 2070 SUPER（8 GB 显存），用于 GPU 训练和推理。运行前在目标环境确认 PyTorch 能识别 CUDA。单卡训练使用 Ultralytics 的 `device=0`。

本机 PyCharm 项目使用 Conda 环境 `baoneng`，解释器位于 `~/miniconda3/envs/baoneng/bin/python`。运行项目脚本、语法检查或模型验证前，先激活项目环境并确认解释器及依赖：

```bash
conda activate baoneng
python -c 'import sys, torch, ultralytics; print(sys.executable); print("PyTorch", torch.__version__, "Ultralytics", ultralytics.__version__)'
```

确认 `sys.executable` 指向 `baoneng` 环境后，使用 `python` 运行下文的命令。在非交互式 shell 中，可以用 `conda run -n baoneng python <参数>`，或直接调用 `"$HOME/miniconda3/envs/baoneng/bin/python"`。不要用系统的 `/usr/bin/python3` 验证项目或判断依赖是否缺失。

迁移到其他电脑时，先确认 Python、显卡驱动、CUDA、PyTorch 和 torchvision 的兼容关系。CPU 环境或无编号为 `0` 的 CUDA 设备时，需要显式调整代码/脚本中的 `device="0"`。

## 数据与生成物

- 默认原始数据目录为 `data/Data10` 和 `data/Data11`。
- 每个原始数据集预期包含 `images/` 和 `annotations/`，标注为 XML 文件。
- 生成内容默认写入 `output/`，包括 YOLO 数据集、manifest、训练权重、预测 JSON 和可视化图片。
- `main.py` 要求 `output/dataset` 为空；重复运行前不要直接删除或覆盖已有结果，应先确认用户是否需要保留实验产物。
- 数据集、模型权重、预测输出和其他大型二进制文件通常不应提交到 Git。若确需版本化，应先确认是否使用 Git LFS。
- 不要提交 API Key、账号凭证、私有数据或机器专属的绝对路径。

## 常用命令

激活 `baoneng` 环境后，从项目根目录执行：

```bash
# 完整数据导出与一次训练流程
python main.py

# 使用现有 dataset.yaml 和 yolo11n.pt 训练
python -m scripts.train_detector

# 使用现有权重和 manifest 推理并生成可视化
python -m scripts.predict_detector

# 在验证集上计算召回率并输出报告
python -m scripts.evaluate_recall

# 从 manifest 导出目标裁剪（脚本当前依赖既有输出）
python -m scripts.export_gt_crops

# 导出待审核的候选裁剪，再导入已审核的负样本
python -m scripts.export_candidate_crops
python -m scripts.import_negative_crops

# 将分类裁剪数据集中的 GB 合并到 crack
# 校验原实验的数据量时，给此命令加 --check-expected-counts
python -m scripts.merge_classifier_classes

# 检查分类数据加载和单批次训练、验证
python -m scripts.check_classifier_batch

# 训练分类器和评估分类混淆矩阵
python -m scripts.train_classifier
python -m scripts.evaluate_classifier

# 两阶段样本推理和完整 test 集 mAP@0.5 评估
python -m scripts.predict_two_stage
python -m scripts.evaluate_two_stage_map
```

这些脚本依赖各自代码中写明的本地路径和既有文件。运行前先检查数据、manifest、模型权重与输出目录是否存在；不要把缺少这些外部资产误判为代码回归。

## 修改原则

- 保持现有 record 数据结构兼容，尤其是 `id`、`image_path`、`dataset`、`objects` 和 `split` 字段。
- 修改数据划分逻辑时保持可复现性；当前默认随机种子为 `42`。
- 修改边界框逻辑时明确坐标格式。当前检测输出使用 `xyxy`，YOLO 标签导出使用归一化的中心点和宽高格式。
- 文件读写继续使用 UTF-8；用户可见日志和异常信息可保持中文。
- 使用 `pathlib.Path` 处理新路径，避免新增机器专属的硬编码绝对路径。
- 不要无关重构。尤其不要在未更新全部导入和验证脚本前重命名 `pipline/`。
- 修复功能时优先补充不依赖大型数据或 GPU 的小型测试；涉及真实训练/推理时，清楚说明所需资产和硬件。

## 完成前检查

- 确认使用 `baoneng` 环境后，对改动过的 Python 文件至少执行语法检查，例如：

```bash
python -m compileall main.py pipline scripts
```

- 若改动不依赖真实数据或 GPU，应运行对应的轻量测试。
- 若验证需要数据集、权重或 CUDA 而当前环境不具备，应明确说明未运行的验证及原因，不要伪造成功结果。
- 检查是否意外生成或修改了 `output/`、模型权重、数据文件、缓存目录或凭证文件。

# 植物病害图像检测与跨场景识别

用 ResNet-50 识别叶片病害，用 YOLO11n 在田间原图中定位叶片。同一组田间测试图上，分别检查叶片能否被定位，以及裁剪后的叶片能否被正确分类。

## 实验结果

测试集为 PlantDoc 原图公开测试部分的 236 张图片和 452 个类别可映射的标注叶片。

| 方案 | 叶片裁剪依据 | 分类器训练数据 | 正确数 | 按标注叶片计的准确率 |
| --- | --- | --- | ---: | ---: |
| 源域分类基线 | 人工标注框 | PlantVillage | 111/452 | 24.56% |
| 检测后分类 | YOLO11n 检测框 | PlantVillage | 105/452 | 23.23% |
| 混合数据微调 | 人工标注框 | PlantVillage + PlantDoc | 282/452 | 62.39% |

- **定位误差。** 检测器输出 551 个候选框。按 IoU ≥ 0.5 一对一匹配后，407 个标注叶片被检出，召回率为 90.04%。23.23% 将漏检计为错误，额外生成的误检框不计入这个分母。
- **跨场景分类误差。** 仅用 PlantVillage 训练的分类器在给定人工标注框时也只答对了 111 个。加入 7,162 个 PlantDoc 训练叶片和 26,979 张 PlantVillage 训练图微调后，在相同人工标注框条件下答对 282 个。
- **数据核查。** 通过文件名或 SHA-256 排除 16 张与测试集重合的训练候选图，再将训练候选按相同内容分组划分，得到 1,967 张训练图和 347 张验证图。脚本核对标注宽高与实际图像尺寸。相似图像、缩放和重压缩后的近重复未纳入该检查。

[结构化实验结果](results/metrics.json)来自一次探索性训练。这里的 28 个候选类别由公开类别名对照得到，训练和测试划分与[参考论文](https://doi.org/10.3389/fpls.2026.1826962)有差异。本次实验尚未评估 YOLO11n 与混合训练分类器组合后的完整流程。

## 数据和环境

按下表在 Linux 机器上准备原始数据。数据和预训练权重请从上游获取，仓库包含训练、评估脚本和汇总结果。

| 路径 | 数据来源 | 实验使用的 Git 提交 |
| --- | --- | --- |
| `data/PlantVillage/raw/color/` | [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) | `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a` |
| `data/PlantDoc-Original/` | [PlantDoc Object Detection Dataset](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset) | `4730a233a555b30ee98e0879c63ad25d82407455` |
| `data/PlantDoc-Cropped/` | [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) | `5467f6012d78d1c446145d5f582da6096f852ae8` |

PlantDoc 原图目录包含 `TRAIN/`、`TEST/`、`train_labels.csv` 和 `test_labels.csv`。裁剪版包含小写的 `train/` 和 `test/` 目录，用于额外诊断，不参与上表的三组对照。尽量保留原始文件名。裁剪版存在仅大小写不同的文件路径，建议在区分文件名大小写的文件系统上准备数据。

参考环境为 Linux、单卡 RTX 4090、Python 3.10/3.11、PyTorch 2.5.1、torchvision 0.20.1 和 Ultralytics 8.3.228。以下示例使用 CUDA 12.1 的 PyTorch。预训练参数在第一次运行时从 torchvision 和 Ultralytics 下载。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## 运行

在仓库根目录准备上述数据后执行。要从头重跑，请使用新的工作目录，以免混用上一次的中间结果。

```bash
mkdir -p runs/exploratory_28class_seed42
cp config/candidate_alignment.csv runs/exploratory_28class_seed42/
python scripts/prepare_three_routes.py
python scripts/train_exploratory_28class.py
python scripts/train_yolo_three_routes.py
python scripts/train_mixed_three_routes.py
python scripts/evaluate_three_routes.py
```

源域分类器的日志和权重位于 `runs/exploratory_28class_seed42/`。田间检测、混合训练和评测的产物位于 `runs/three_routes_v1/`。模型权重由训练脚本产生，实验汇总可直接查看 `results/metrics.json`。

# AI6126 Project 1 - Face Parsing

Attention U-Net for CelebAMask-HQ face parsing.

## 文件结构
```
AI6126_project1/
├── src/
│   ├── dataset.py          # 数据集加载 + 数据增强
│   ├── losses.py           # Dice Loss + CrossEntropy 组合
│   ├── model.py            # Attention U-Net 模型
│   ├── model_b_deeplab.py  # 轻量 DeepLab 备选
│   └── model_c_segnet.py   # 轻量 SegNet 备选
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       └── images/         # 作为测试集使用，无 masks
├── docs/
│   └── AI6126_project1.md  # 作业要求
├── checkpoints/
├── predictions/
├── train.py                # 训练入口
├── predict.py              # 推理入口
└── README.md
```

## 快速开始

### 1. 确认参数量
```bash
python3 -m src.model
# 应输出: Total parameters: xxx / 1,821,085 limit
# Within limit: True
```

### 2. 训练
```bash
python3 train.py \
    --data_dir ./data \
    --arch deeplab \
    --num_classes 19 \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-3
```

A100上跑 100 epoch 大概 1-2 小时。

### 3. 推理（测试集）
```bash
python3 predict.py \
    --test_dir ./data/val/images \
    --checkpoint ./checkpoints/deeplab/best_model.pth \
    --output_dir ./predictions \
    --num_classes 19
```

## 注意事项
- `num_classes` 根据实际数据集调整（CelebAMask-HQ原版19类，确认你们的子集）
- mask文件的像素值应该直接是类别id（0到N-1的整数）
- 如果mask用调色板PNG存储，`Image.open(mask).convert('L')` 会自动处理
- 当前 `data/val` 默认作为测试集使用，通常不含 `masks`

## 调参建议
- batch_size=8 适合A100，可以加到16试试
- base_ch=32 是默认，如果参数量还有余量可以改到40（但要验证不超限）
- epochs 100起步，val F-score不涨了就停

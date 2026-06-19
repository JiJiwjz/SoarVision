# 翱翔智眸 · YOLO26 检测 Demo 方案
**架构设计 / 训练范式 / 8GB 配置 / 工时估算**

> 目标:在 RTX 4060 Laptop (8GB) 上先跑通一个可复现的 demo——验证数据流水线、拿到 baseline 地板分、看到小目标头的首个增益。重模块（D2/F1/W1）按消融阶梯后续再叠,**不在本 demo 里一次性上**。

---

## 0. 数据现状（3 类）

| 类别 id | 类别 | boxes（约） | 占比 |
|---|---|---|---|
| 0 | vessel（大中型船） | ~126,600 | ~87% |
| 1 | small_boat（小型船/快艇/渔船） | ~15,300 | ~11% |
| 2 | buoy（浮标） | ~3,000 | ~2% |

- **来源**:SeaShips(7,000) + SMD-VIS(19,408) ≈ **26,408 张图**（MODD2 已移除,SMD `other` 已丢弃）。
- ⚠ **极度不均衡**:buoy 仅 ~2%,demo 就要带 `copy_paste`,否则 buoy/small_boat 召回会塌。
- ⚠ **划分纪律**:SMD 是视频,**必须按视频段划分** train/val/test,不能按帧随机划（否则连续帧泄漏 → 指标虚高）。建议三个集统一成 **全局 70/15/15**;按此估算 train ≈ 18.5k / val ≈ 4k / test ≈ 4k。
- 建议另存一份 SMD 真实雾（Haze 条件）帧作为「真实退化验证集」,后面和合成雾对照用。

---

## 1. 架构设计

### 1.1 Demo 阶段（先跑通,逐级加）

| 步骤 | 配置 | 改动 | 目的 |
|---|---|---|---|
| **E0** | `yolo26n.pt` baseline @640 | 无（COCO 预训练直接微调） | 验证流水线 + 拿地板分 |
| **E1** | `yolo26n-p2.yaml`（S1） | 加 P2 小目标头（官方自带 yaml） | 海上远距小船的首个增益 |
| **E2** | E1 + 去掉 P5（S2） | 编辑 yaml,Detect 只用 P2/P3/P4 | 减参、专注中小目标 |

> Demo 跑到 E1 即可证明小目标头价值;E2 是优化,有时间再做。

### 1.2 目标架构（路线图,后续按消融阶梯逐步叠）

```
YOLO26n (NMS-free 端到端基线)
  └─ + P2 头 / 去 P5            [S1+S2]  小目标
  └─ + 颈部注意力 CBAM/ECA      [S5]     抗水波杂波
  └─ + RFB / 大核感受野块       [S7]     上下文
  └─ + BiFPN 加权融合           [S8]     多尺度
  └─ 损失换 Wise-IoU v3         [S6]     定位稳定
  └─ 训练范式: D1 在线退化增强            鲁棒性（核心）
  └─ 进阶创新层（择一深做）:
       D2 复原分支 / F1 小波块 / W1 海面分割多任务
```

> **每加一个结构件,先跑一次 ONNX/TensorRT 导出冒烟测试**——尤其 F1 的小波要用卷积实现的 Haar,别用框架原生 DWT 算子(TensorRT 大概率不支持)。

---

## 2. 训练范式

1. **迁移学习**:从 COCO 预训练 `yolo26n.pt` 起步微调（3 类头自动适配,收敛快得多）。
2. **分阶段对齐消融阶梯**:E0 baseline（**关闭退化增强**,拿干净地板）→ E1 +P2 → 然后**打开 D1 在线退化增强**,在分级退化验证集上看鲁棒性。
3. **类别不均衡**:demo 先用 `copy_paste≈0.3` 缓解;后续 T4 再上加权采样 / 难样本挖掘。
4. **在线退化增强（D1）**:**训练阶段实时做**,不预生成存盘。快速版用 albumentations（`RandomFog` + 压暗）,以 `p≈0.5` 施加;物理散射雾（`I=J·t+A(1-t)`）后续写成自定义 transform。**测试集的分级退化则相反——离线预生成、固定不变**。
5. **评估**:每 epoch 跑 val mAP;另用留出 test + 分级退化 test 出鲁棒性曲线。

---

## 3. 配置（8GB 显存调参）

### 3.1 `maritime3.yaml`

```yaml
path: /path/to/datasets/merged_maritime
train: images/train
val:   images/val
test:  images/test
names:
  0: vessel
  1: small_boat
  2: buoy
```

### 3.2 关键超参（8GB 取值）

| 参数 | 取值 | 说明 |
|---|---|---|
| `imgsz` | 640 | demo 标准;1280 会爆显存,后期再单独试 |
| `batch` | 16（baseline）/ 8（带 P2） | OOM 就降到 8;26n 很轻,640 下 16 通常 <5GB |
| `amp` | True | 混合精度,省显存提速（默认开） |
| `optimizer` | 默认（YOLO26 用 MuSGD） | 不用手动改 |
| `epochs` | demo 50–100 | 正式可 150–300 |
| `patience` | 30 | 早停,省时间 |
| `cache` | disk | 加速读图;内存够也可 ram |
| `workers` | 8 | 按 CPU 核数 |
| `copy_paste` | 0.3 | 缓解 buoy/small_boat 长尾 |

### 3.3 启动命令

**E0 baseline（Python）**
```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")                      # COCO 预训练
model.train(
    data="maritime3.yaml",
    epochs=100, imgsz=640, batch=16,
    device=0, workers=8, amp=True,
    cache="disk", patience=30, copy_paste=0.3,
    project="runs/maritime", name="e0_baseline_26n",
)
```

**E1 加 P2 头**
```python
model = YOLO("yolo26n-p2.yaml").load("yolo26n.pt")   # 结构换 P2,权重迁移
model.train(data="maritime3.yaml", epochs=100, imgsz=640,
            batch=8, device=0, amp=True, cache="disk",
            patience=30, copy_paste=0.3, name="e1_p2_26n")
```

**CLI 等价写法**
```bash
yolo detect train model=yolo26n.pt data=maritime3.yaml \
     epochs=100 imgsz=640 batch=16 device=0 amp=True \
     cache=disk patience=30 copy_paste=0.3 name=e0_baseline_26n
```

### 3.4 在线退化增强（D1 快速版草图）

> Ultralytics 集成 albumentations 需改 `ultralytics/data/augment.py` 里的 `Albumentations` 类,把下面的 transform 加进去（`p≈0.5` 对齐 D1）。物理散射雾后续写成自定义 `A.ImageOnlyTransform`。

```python
import albumentations as A
degrade = A.OneOf([
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, p=1.0),     # 雾(快速版)
    A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.1),
                               contrast_limit=(-0.3, 0.0), p=1.0),  # 低光
    A.GaussNoise(var_limit=(10, 50), p=1.0),                        # 噪声
], p=0.5)   # ~50% 图施加退化,强度随机 → 在线、不存盘
```

---

## 4. RTX 4060 Laptop (8GB) 工时估算

> 假设:train ≈ 18.5k 图、YOLO26n、AMP 开。**粗估,实际 ±40~50%**,受散热降频、batch、磁盘 IO 影响大。

### 4.1 单 epoch 与总时长

| 配置 | imgsz | batch | ~分钟/epoch | 50 ep | 100 ep | 200 ep |
|---|---|---|---|---|---|---|
| 26n baseline | 640 | 16–32 | 5–7 | ~5h | ~10h | ~20h |
| 26n + P2 头 | 640 | 8–12 | 7–9 | ~7h | ~14h | ~28h |
| 26n + D1 退化 | 640 | 8–16 | 6–8 | ~6h | ~13h | ~26h |
| 高分辨率小目标 | 1280 | 2–4 | 25–40 | ~25h | — | — |

### 4.2 三个注意点

- **笔记本散热降频**:长时间满载会掉频,实际时长再 ×1.2 左右;确保散热、可适当限功耗稳频。
- **8GB 是硬约束**:1280 高分辨率会 OOM,只能 batch 2–4,得不偿失;小目标想要高分辨率,建议放台式/云。
- **batch 减半 ≈ 时间增加约 1.3–1.5×**（GPU 利用率下降）。

### 4.3 实操建议

1. **冒烟测试（<1.5h）**:取 20% 子集、跑 30 epoch,先确认流水线、显存、mAP 出得来。
2. **正式 demo（过夜）**:26n @640、80–100 epoch、带早停,约 **8–14 小时**。
3. **重活外包**:高分辨率、D2/F1/W1 等重模块的完整消融,4060 Laptop 会成瓶颈——建议挪到台式机或云（Colab / AutoDL 等)跑,笔记本只做原型迭代。

---

## 5. 下一步 checklist

- [ ] 确认 SMD 已**按视频段**划分(防泄漏)
- [ ] 三集统一全局 70/15/15,合并出 `images/{train,val,test}` + 对应 labels
- [ ] 跑 E0 冒烟测试(20% 子集,30 ep)
- [ ] E0 全量 baseline(过夜)→ 记 mAP/Recall/漏检率/误检率/FPS
- [ ] E1 加 P2 头,对比小目标 AP_S 增益
- [ ] 打开 D1 在线退化,在分级退化验证集上看鲁棒性
- [ ] 每步导出冒烟测试(ONNX/TensorRT)

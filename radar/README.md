# `radar/` — 雷达-视觉融合的数据侧骨架

这是「翱翔智眸」多模态融合线的**数据地基**。目标：把雷达点云变成和图像对齐、可直接喂给
RF-DETR 融合分支的张量（REVP map），并提供解析 / 校验 / 可视化工具。

> 设计原则：数据先于模型。先把"读得对、对得齐"验证清楚，再谈融合网络。

## 为什么用 WaterScenes（关键解锁）

我们有真雷达（TI IWR6843，77GHz），但**没有配对的 RGB-雷达训练数据**。
[WaterScenes](https://github.com/WaterScenes/WaterScenes)（arXiv 2307.06505）几乎为我们定做：

| | WaterScenes | 契合点 |
|---|---|---|
| 雷达 | Oculii EAGLE **77GHz 4D** | 和我们 IWR6843 **同频段**，特性可迁移 |
| 类别 | pier/buoy/sailor/ship/boat/vessel/kayak | 覆盖我们的 vessel/small_boat/**buoy** |
| 规模 | 54,120 帧 / 20.2 万目标 | 足够训练融合模型 |
| 恶劣条件 | 5604 弱光 + 10729 雨/雪 | 正中雾天/低光卖点 |
| 标注 | 2D 框(YOLO) + 逐点雷达(距离/速度/方位/俯仰/功率) | 检测 + 融合都能做 |
| **关键** | radar CSV **已含 `u,v`（点已投影到图像）** | **WaterScenes 上无需自己标定** |

策略：**在 WaterScenes 上开发/验证融合算法** → 迁移到自采片段 demo。
官方 baseline 已证明论点：融合后恶劣光照 **mAP +5.3%**。

## 下载与放置

1. 按官方仓库 https://github.com/WaterScenes/WaterScenes 的指引下载（Google Drive / 申请表）。
2. 解压到 `datasets/WaterScenes/`，期望布局：

```
datasets/WaterScenes/
  image/        <frame>.jpg              # RGB 1920x1080
  radar/        <frame>.csv              # 每行一个雷达点（列见下）
  calib/        <frame>.txt              # 内参 + 雷达->相机外参（自采数据时才需要）
  detection/yolo/ <frame>.txt            # YOLO 归一化框 (cls cx cy w h)
  ImageSets/    train.txt val.txt test.txt   # 帧 stem 列表（可选；缺失则用全部图像）
```

radar CSV 列（解析器优先读表头，无表头时按此顺序）：
```
timestamp, range, doppler, azimuth, elevation, power,
x, y, z, comp_height, comp_velocity, u, v, label, instance
```

## ⚠ 下载后第一件事：校验类别顺序

`waterscenes.py` 里的 `WATERSCENES_CLASSES`（id→名）是**按论文文字假定的顺序**，
WaterScenes 自己 YOLO 标签的 id 顺序可能不同。下载后**务必**核对其 `classes.txt`：
若存在，`load_classes()` 会自动读取并覆盖；若不存在，请手动改 `WATERSCENES_CLASSES`，
否则 `ship/vessel/buoy` 会被映射错。用下面的 `--validate` 检查会提示。

## 类别重映射（7 类 → 我们的 3 类）

`waterscenes.REMAP`（按名字映射，可改）：

| WaterScenes | → SoarVision |
|---|---|
| ship, vessel | `vessel` (0) |
| boat, kayak | `small_boat` (1) |
| buoy | `buoy` (2) |
| sailor（船上人）, pier（岸基设施） | **丢弃** |

## REVP 表示

毫米波点稀疏、无纹理，所以渲染成**低分辨率、与图像对齐**的多通道图（每个图像 patch 一格），
通道携带相机在雾里看不到的物理量：

| ch | 含义 | 作用 |
|---|---|---|
| 0 range | 径向距离 | **伪深度**先验 |
| 1 elevation | 俯仰角 | 高度线索 |
| 2 doppler | 径向速度 | 区分动目标 / 海杂波 |
| 3 power | 反射功率 | 目标强度 / RCS 代理 |
| 4 occupancy | 是否有点 | 存在性掩码 |

一格多点时默认保留**最近**点（`reduce="min_range"`，可选 `max_power`）。归一化常数见
`revp.RevpNorm`（按 200m 量程设定，见到真实直方图后再调）。

## 模块与用法

| 文件 | 作用 |
|---|---|
| `waterscenes.py` | 布局/列定义、类别重映射、`load_frame`、`list_frames`、`--validate` |
| `revp.py` | `build_revp_map()` → `[5,H,W]`；`--selftest` 用合成点验证 |
| `dataset.py` | `iter_samples()`(无 torch) + `build_torch_dataset()`(懒加载 torch) |
| `visualize.py` | 叠图(雷达点+GT框) + REVP 通道拼图，**对齐 sanity check**（仅 PIL） |

```bash
# 0) 解析器自检（不需要数据集）
python radar/revp.py --selftest

# 1) 校验下载是否完整 + 类别顺序
python radar/waterscenes.py --root datasets/WaterScenes --validate

# 2) 探查单帧（点数、量程、重映射后的框）
python radar/waterscenes.py --root datasets/WaterScenes --frame <id> --probe

# 3) ★ 对齐可视化（最关键的一步：确认雷达点落在船上）
python radar/visualize.py --root datasets/WaterScenes --split train --num 8 --out runs/radar_viz

# 4) 迭代检查 dataset 输出
python radar/dataset.py --root datasets/WaterScenes --split train --limit 5
```

> 约定：沿用 `scripts/` 的扁平风格（无 `__init__.py`，同级直接 import）。
> 从仓库根运行 `python radar/xxx.py`，或设 `PYTHONPATH=radar`。

## 这一步如何接到 RF-DETR 融合（下一步）

数据台搭好后，融合分支计划（见记忆 `reference-radar-camera-fusion`）：
**中融合 + RF-DETR decoder 里的雷达 cross-attention（软关联，TransCAR 路线）+ 退化置信门控**。
- `dataset.py` 的 `build_torch_dataset(image_transform=...)` 预留了接口：把 RF-DETR 自己的
  预处理传进来，让 RGB 流与检测器一致，REVP 张量随图像同批喂入。
- 下一步代码：在 decoder 加一个轻量 **Radar Cross-Attention 分支**（REVP 特征作 K/V，
  object queries 去 cross-attend），图像退化时由 DA-FP 估计的退化度放大雷达权重。

## 自采数据（后期）

自采片段需要 `calib/`（雷达↔相机外参标定，角反射器或无靶法），把雷达点投到图像得到
`u,v`，之后即可复用这里的同一套 REVP / dataset / visualize 流程。

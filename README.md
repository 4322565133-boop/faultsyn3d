# faultsyn3d — 三维地震断层分割:合成数据、模型与 Thebe 实测协议

本仓库是一个进行中的研究项目(目标:TGRS 论文),内容分三块:

1. **合成断层数据生成器**:Su et al. (2026) 断层网络合成的数值复现(`su/`、`generate.py`)和 Wu et al. (2019) 式合成(`synth/wu2019.py`),各 1000 个 128³ 体。
2. **三维断层分割模型**(`models/`):3D U-Net、3D MaxViT-UNet(timm-3d,含全分辨率 skip / drop-path / 128 分区三项修正)、U-Net + 网格/全局注意力混合模型、Context-Grid 跨尺度解码器、DoubleBlock-ViT 移植、2D MaxViT + 跨切片 adapter、VSS-SAM++ 缩小版(冻结 SAM-2 Hiera + 3D Mamba 分支 + 门控融合)。
3. **Thebe 实测数据的封闭评估协议**(`train/dataset_thebe_spatial.py`、`train/train_thebe_spatial_ddp.py`、`docs/THEBE_SPATIAL_V3_TRAINING.md`):官方切分 + 缓冲带、逐道信号掩膜、中心 64³ 计分、整验证区选点、测试区封存。

数据(139 GB)、checkpoint(104 GB)和第三方论文 PDF 不在仓库里;`runs/` 下只保留每个实验的 `config.json`、`history.json`、测试 JSON 和曲线图。

## 目录

| 目录 | 内容 |
|---|---|
| `su/`, `generate.py`, `configs/` | Su 2026 断层网络合成复现(五类,`data/dataset_reproduction_v2`) |
| `synth/` | Wu 2019 式合成生成器、Thebe 立方体切分 |
| `models/` | 所有网络定义,统一接口 `build(name, **kw)`:`(B,1,D,H,W) -> logits` |
| `train/` | 训练器(旧配方 DDP、Thebe 空间协议 DDP)、数据集、评估脚本(同分布、零样本、Thebe 封存测试) |
| `scripts/` | 每个实验的启动脚本(记录了确切的命令行) |
| `qc/` | 数据 QC、实时 IoU 曲线(`qc/live_iou.py`) |
| `docs/` | 协议说明、实验方案、路线图、审计记录 |
| `reports/` | 设计/审阅报告 |
| `figs/`, `logs/` | 结果图、训练日志 |

## 训练配方(所有对比共用)

0.6·soft-Dice + 0.4·Focal(α 0.75, γ 2),AdamW(wd 0.01),10 轮 warmup 1e-6→1e-4 后余弦到 1e-7,有效 batch 8,fp16,seed 2026;合成集 200 轮、Thebe 100 轮;按整验证区 IoU 选点。模型变体在 `train/train_old_recipe_ddp.py` 的 `VARIANTS` 里登记。

## Thebe 协议

- **v3-U**(均匀随机抽样):训练块 128³ 随机窗口、信号 ≥25%、不按标签筛;目前所有 Thebe 结果都属于它。
- **v3-FG50**(前景过采样):一半抽样强制含 ≥0.5% 断层体素(`--fg-frac 0.5 --fg-min 0.005`),评估不变。
- 评估:验证区 [900,1068)、测试区 [1100,1803),中心 64³ 核步长 64,阈值 0.5,体素级合并 IoU/Dice/P/R/AP。详见 `docs/THEBE_SPATIAL_V3_TRAINING.md`。

## 目前的主要结果(单 seed)

| 设置 | 模型 | 参数 | FLOPs/128³ | IoU |
|---|---|---|---|---|
| 旧难集,算力对等,同分布测试 | MaxViT-tiny / U-Net+attn b30 / U-Net b30 | 40.2M / 6.1M / 4.9M | 0.77 / 0.81 / 0.79 T | 0.628 / 0.622 / 0.599 |
| 同上 → Wu 2019 公开 20 体,零样本 | 同上 | | | 0.294 / 0.321 / 0.302(配对 bootstrap:混合模型 − U-Net = +0.027 [+0.010, +0.045]) |
| Thebe v3-U 封存测试区 | 3D U-Net-L / 3D MaxViT-tiny | 9.6M / 40.2M | 1.54 / 0.77 T | 0.377 / 0.390 |
| 合成 → Thebe 零样本(7 个模型) | | | | 0.04–0.05(≈ 先验) |

其余(半监督试点、2D MaxViT+adapter、VSS-SAM、DBViT、Context-Grid)见 `docs/PAPER_ROADMAP_20260918.md` 和各 `runs/*/history.json`。

## 复现步骤

```bash
# 环境:PyTorch 2.10 + timm 1.0.24 + timm-3d 1.0.1(见 requirements-reproduction.txt)
# 1. 合成数据
python generate.py --config configs/reproduction_v2.json          # Su 2026 复现
python synth/wu2019.py --out data/wu2019_1000 --n 1000            # Wu 2019 式
# 2. Thebe(An et al. 2021 公开数据集,自行下载到 data/thebe)
python scripts/prepare_thebe_spatial.py --with-test               # 生成 data/thebe_spatial_v3
# 3. 训练(4 卡示例)
python -m torch.distributed.run --nproc_per_node 4 -- train/train_thebe_spatial_ddp.py \
    --run runs/thebe_spatial_v3_unet_l --variant unet_l --epochs 100 --stop-after 100 --full-every 1 --per-rank 2
# 4. 封存测试区评估
python train/eval_thebe_spatial.py --runs runs/thebe_spatial_v3_unet_l --gpu 0
```

每个实验的确切命令在 `scripts/launch_*.sh`。

## 数据来源

- Thebe:An, Y. et al. (2021), *Deep convolutional neural network for automatic fault recognition from 3D seismic datasets*, Computers & Geosciences.
- Wu 2019 公开验证体:Wu, X. et al. (2019), *FaultSeg3D*, Geophysics.
- 合成器复现的原文:Su et al. (2026), TGRS;Wu et al. (2019)。

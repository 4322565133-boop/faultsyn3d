# 算力对等的模型对比方案 — 2026-09-17

## 为什么换成算力对等

参数量对等会把 MaxViT 放在不利位置:它的参数在 8³/16³ 深层,每个参数只服务几百个体素;UNet 的参数在 128³/64³,
每个参数服务上百万体素。同为 40M,MaxViT-tiny 前向 0.77 TFLOPs/体,UNet base96 是 8.0 TFLOPs/体,差 10 倍。
ViT 文献(含 MaxViT 原文)的标准做法是按 FLOPs 对比。本方案以 FLOPs 为主轴、参数量为副轴,两个轴都报。

## 事实基础(截至今天)

- 旧数据集(归档 880/110/110):MaxViT-tiny 原版 40M 测试 0.611 > ResNet3D 195M 0.578 > UNet-S 1.4M 0.539。
  但 UNet-S 只有 0.22 TF、1.4M,既不算力对等也不参数对等。
- v2 数据集饱和(所有模型 0.82–0.84),不能用来区分模型。
- 零样本到 Wu 公开 20 体(v2 训练):UNet+注意力 0.494 > UNet-L 0.483 > MaxViT-tiny all3 0.337。
  注意:这条证据支持的是"UNet 里加 MaxViT 式注意力",不是 MaxViT-tiny 本身;MaxViT-tiny 零样本最差(有轮数混淆)。
- 软 Dice 地板效应会翻转按 val loss 的模型选择;本方案沿用旧配方(val loss 选点)以便和归档可比,
  但每个 run 都另存 val-IoU 最优点供对照。

## 实验矩阵(旧数据集,旧配方,训练到收敛:warmup 10 → 保持 1e-4 → 10 轮不降减半 → 30 轮不降停)

| 组 | 模型 | 参数 | FLOPs/体 | 作用 |
|---|---|---|---|---|
| A 算力对等 0.77 TF | MaxViT-tiny all3 | 40.2M | 0.77 | 锚点(候选模型 1) |
| A | UNet base30 | 4.9M | 0.79 | 算力对等的 UNet |
| A | UNet + MaxViT 注意力 base30 | 6.1M | 0.81 | 算力对等的混合模型(候选模型 2) |
| A | MaxViT-nano all3 | 24.8M | 0.72 | MaxViT 家族稍低算力点 |
| B 参数对等 ~10M | MaxViT-pico all3 | 9.8M | 0.40 | 参数对等的 MaxViT |
| B | UNet-L base42 | 9.6M | 1.54 | 参数对等的 UNet(4× 算力) |
| B | UNet + 注意力 base42 | 11.9M | 1.59 | 参数对等的混合模型 |
| C 极限对照 | UNet base96 | 50.4M | 8.0 | 给 UNet 10 倍算力还赢不赢 |

每个 run 结束后自动出三组数:旧测试集 110 体(同分布、难)、零样本 Wu 公开 20 体(第三方)、零样本我们的 Wu 式测试 100 体。

## 判定规则(提前写死)

- 主表 = A 组旧测试集 IoU/AP。MaxViT-tiny 或混合模型领先算力对等 UNet ≥ 2 个点、且零样本两组不反转 → "在相同算力下 MaxViT 式注意力更优"成立,对胜者对做 3 个 seed。
- B 组用来回答审稿人必问的"参数量对等呢":如果 A 成立而 B 里 UNet-L 反超,论文如实写"算力对等占优、参数对等不占优",这是 ViT 类模型的常见结论,可以发。
- C 组:若 MaxViT-tiny 连 10 倍算力的 UNet-96 也赢,主张最强;若输,不影响 A 的结论,但要写明。
- A 组内混合模型 vs MaxViT-tiny:谁赢谁就是论文的模型;都赢 UNet 就报两个。

## 队列与时间

scripts/launch_legacy_parity.sh,顺序 A1–A3 → B1–B3 → A4 → C。每轮:tiny ~105 s、b30 ~40 s、attn-b30 ~45 s、
pico 33 s、UNet-L 62 s、attn-b42 73 s、nano ~90 s、b96 ~320 s;按 250 轮收敛估计,A1–A3 约 13 h,B 约 12 h,
A4 约 6 h,C 约 22 h。实时曲线 runs/legacy_parity_live_iou.png。

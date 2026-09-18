# UNet-L + MaxViT 式注意力（unet_lg）方案 — 2026-09-16

## 1. 依据：现有结果与逐类别诊断

测试集（v2，旧配方）逐类 IoU：

| 模型 | 参数 | IoU | AP | P | R | en_ech | horse | neg_fl | pos_fl | listric |
|---|---|---|---|---|---|---|---|---|---|---|
| UNet-S | 1.4M | 0.820 | 0.950 | 0.892 | 0.910 | 0.906 | 0.784 | 0.824 | 0.823 | 0.743 |
| MaxViT all3 | 40.2M | 0.833 | 0.956 | 0.911 | 0.906 | 0.914 | 0.796 | 0.828 | 0.845 | 0.756 |
| UNet-L @56 | 9.6M | 0.835 | 0.957 | 0.911 | 0.910 | 0.906 | 0.805 | 0.839 | 0.842 | 0.766 |

- 容量（UNet-S→UNet-L）涨的是 horsetail/flower/listric（+1.5～2.3），en_echelon 不动（已到顶）。
- all3 相对 UNet-L 唯一的独有优势是 en_echelon（+0.8）——规则排列的平行雁列段，是"周期性全局结构"；其余各类 all3 ≤ UNet-L。
- Compact M0 与 DBViT 两次"卷积换注意力"都输：损失更低、IoU 更低（过拟合），recall 卡住。

UNet-L @56 验证集 near/far 诊断（docs/UNET_GRID_PR_DIAGNOSIS_20260916.json，Chebyshev ≤2 体素）：

| 类 | R | FN 总量 | 其中 far-FN（整段漏掉） |
|---|---|---|---|
| en_echelon | 0.953 | 76k | 0.9% |
| horsetail | 0.884 | 136k | 10.1% |
| negative_flower | 0.887 | 171k | 13.8% |
| positive_flower | 0.920 | 140k | 8.9% |
| listric | 0.822 | 216k | **25.2%**（54k，占全部 far-FN 的一半） |

FP 99.2% 在 2 体素内（厚度问题，上下文无关）。FN 里 14% 是"整段漏掉"，集中在 listric（曲面向下位移衰减到零的那段）和 flower/horsetail 的分支。这类错误的共同点：同一条断层面在别处清晰、在这里微弱——需要沿面传播远处的证据。这是卷积 U-Net 做不到、注意力理论上能做的唯一一件事。

## 2. 设计（models/unet_maxvit3d.py，`UNetMaxViT3D`）

原则：**UNet-L 一个卷积都不动**（同 seed 下 U-Net 部分初始权重与 UNet-L 逐位相同，已验证），注意力只作为残差分支追加在两级深层编码器之后；解码器不动。

| 位置 | 通道 / token 数 | 追加的模块 | 参数 |
|---|---|---|---|
| enc3 之后（32³） | 168 / 32768 | block attn（4³ 窗口，局部）→ grid attn（4³ 网格，稀疏全局，伙伴间隔 8 token） | 0.46M |
| mid 之后（16³） | 336 / 4096 | block attn（4³ 窗口）→ **global attn（全部 4096 token 稠密自注意力**，SDPA，分轴可学习位置编码） | 1.84M |

- 每个 attention 块 = LN → 注意力（dim_head 24，相对位置偏置 / 位置编码）→ 残差 → LN → MLP(×2) → 残差；drop_path 0.1（32³）/ 0.2（16³）只作用在新增分支。
- 32³ 用 MaxViT 原样的 block+grid（PartitionAttentionCl，与 40M MaxViT、Compact 同一实现）；16³ 用真正的稠密全局注意力，因为 4096 个 token 用 SDPA 完全负担得起，而"沿面传播证据"需要的是看见同一面上的所有 token，不是网格里的 64 个。
- 总参数 11.94M（+2.3M，+24%）。

第二臂 `unet_lg_rep`（替换）：同样的注意力，但把 enc3 / mid 的第二个 3³ 卷积去掉，每个深层级变成 conv → attention（和 MaxViT stage 的 MBConv → attention 同构）。8.13M。它回答"注意力能不能替代稠密卷积"；第一臂回答"注意力能不能补充稠密卷积"。

与 Codex 的 `UNetGrid3D` 的区别：Codex 只加 grid attention、位置在 mid 与 dec3（解码器）、+1.1M；本方案加 block+grid+global、全在编码器、+2.3M。若两者都跑完，可以作为"只要 grid 够不够"的消融。

## 3. 训练与队列（scripts/launch_unet_lg.sh，已启动）

配方与 UNet-L 完全一致（旧配方：0.6 Dice+0.4 Focal，AdamW wd .01，warmup 10 轮到 1e-4 余弦到 1e-7，200 轮，global batch 8，val loss 选模型，seed 2026）。

1. `unet_lg`（4 卡，~67 s/轮，约 3.8 h）→ `runs/maxvit3d_unet_lg_ddp/`，实时曲线 `live_curves.png`（对 UNet-L 前 56 轮）
2. UNet-L 从第 56 轮续跑到 200（新加 `--resume`，已验证 57 轮接上：val 0.8278）→ 基线补完（约 2.5 h）
3. `unet_lg_rep`（约 3.8 h）
4. 六模型测试表 `logs/eval_unet_lg.log` + 误差分析 `logs/analyze_unet_lg.txt`

烟测：4 卡 per-rank 2 峰值显存 9.0 GB（UNet-L 8.2 GB）；前 2 轮与 UNet-L 逐位几乎相同（0.57903 / 0.57903），差异从注意力分支长起来后才出现——这是配对实验应有的样子。

## 4. 判定规则（提前写下，避免事后解释）

- 主指标：测试 IoU / AP；关键副指标：listric 与 flower 的 recall、far-FN 占比（analyze）。
- 若 `unet_lg` 相对 UNet-L(200)：IoU ≥ +0.5 且 far-FN 明显下降 → 注意力确有"沿面传播"的增量，进入消融（去 global / 去 block / 只 32³ / 只 16³）与多 seed。
- 若在 ±0.3 以内 → 增量为零，与 Compact/DBViT 的结论一致；论文改写为"同容量公平对比 + 注意力在合成断层任务上不占优的分析"。
- 若 `unet_lg_rep` ≥ UNet-L 而参数少 16% → "注意力是更省参数的替代"是可写的正结论。
- 容量对照：若 `unet_lg` 赢，还需一个 UNet base=47（≈12.1M）排除"多 2.3M 参数"的解释。

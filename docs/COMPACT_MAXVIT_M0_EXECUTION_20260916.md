# Compact MaxViT M0 启动记录

## 用户授权与执行结果

用户要求停止当前训练，实施前一份方案，并继续与新数据集 MaxViT-UNet 基线实时比较。

已停止旧 DoubleBlock legacy 训练及其父队列，避免它继续启动 dbvit_frs。旧 checkpoint 和日志未删除。进程停止审计在 STOP_LEGACY_FOR_COMPACT_20260916.json。

新运行：`runs/compact_maxvit_v2_m0_20260916`。

## 本轮模型

M0 结构起点，9,489,043 参数。stem 128³/36，局部卷积 64³/72，MaxViT 32³/144 一个块，MaxViT 16³/288 两个独立参数块。三个块均保留 block + grid attention；全分辨率跳连，简单卷积解码器。

MBConv expansion=2，FFN expansion=3，head dim=16，partition=4³，drop path 在三个块中为 0/.1/.2。浅层/解码器 InstanceNorm，MBConv BatchNorm，注意力 LayerNorm。

完整随机初始化，没有加载历史 MaxViT、U-Net 或短测权重。卷积使用 Kaiming 初始化，Linear 使用 trunc_normal(std=.02)。模型代码：models/compact_maxvit3d.py。

本轮没有加入面状方向模块、D2S、DPF、动态上采样或几何 loss。先验证分辨率与参数分配；面状模块属于下一次独立实验。

## 训练协议

- 数据：dataset_reproduction_v2；800 训练 / 100 验证 / 100 测试。
- 四张 RTX 2080 Ti，DDP，每卡 batch=2，全局 batch=8。
- 固定 200 epoch，随机种子 2026。
- AdamW，weight decay=.01；10 epoch warmup，峰值 lr=1e-4，cosine 到 1e-7。
- loss：显式 FP32 的 .6 Dice + .4 Focal(alpha=.75,gamma=2)。无辅助 loss，无梯度裁剪。
- AMP 初始 scale=1024；非有限反向梯度四卡同步跳过并降低 scale，写日志，不把它当成成功更新。
- best.pt 按最低验证 segmentation loss 选择，与历史基线选择标准一致；实时曲线也展示实际每轮 IoU，不混淆两种选择标准。
- stem 和末级解码器使用 activation checkpoint 节省显存；这些 InstanceNorm 不跟踪 running statistics。MBConv BatchNorm 不做重计算。
- 每轮训练后统一广播 rank0 buffers 再做分布式验证。
- last.pt 原子写入模型、optimizer、scaler、每个 rank 的 Python/NumPy/Torch/CUDA RNG、更新数、轮数和配置。学习率日程由保存轮数与冻结配置重建。支持 epoch 边界恢复；启用 cuDNN benchmark，不能承诺跨硬件逐位一致。
- manifest 和核心源文件 hash 保存在 args.json；恢复时检查配置/源码一致性。

## 已通过的检查

1. CPU 验证 forward 形状、activation checkpoint 开关下输出/梯度一致。
2. block 和 grid 划分不同且各自可逆；深层两个块参数独立。
3. 抽取样本验证新无几何 loader 与原 GeometryVolumes 的地震/标签增强完全一致。
4. 单卡真实 128³、batch=2 前后向，全部参数有有限梯度。
5. 四卡短训练 3 次更新，存盘退出，再恢复下一轮；optimizer 158 个状态条目、四 rank RNG、AMP growth tracker 恢复成功，再完成 3 次更新，共 6 次、0 次跳过。
6. 恢复短测峰值约 9830 MiB（约 9.60 GiB）/rank0。其他 rank 与正式全程峰值以运行监控为准。

检查结果：COMPACT_MAXVIT_GPU_PROBE_20260916.json、COMPACT_MAXVIT_CHECKS_20260916.json，以及 runs/compact_maxvit_smoke_20260916 下的 log.csv/resume_events.jsonl。

## 实时对照与监控

运行目录下：

- index.html：中文对照页，每 30 秒刷新。
- live_curves.png：验证 IoU、train/val loss、precision/recall、同轮 IoU 差值。
- live_metrics.json：机器可读对照。
- train.log：逐 10 step 心跳、每轮汇总、AMP 跳过记录。
- status.json：训练/验证状态及时间戳。
- processes.json：supervisor、torchrun PID 和命令。
- reference_maxvit_log.csv：冻结的 runs/maxvit3d_v2_oldrecipe_all3 历史曲线。

仅比较验证曲线，不能把历史 Test IoU=0.83259 当验证阈值。历史基线的单卡累积、初始化和数值执行与本轮有差异，因此这是历史参考比较；最终论文仍需统一协议和多 seed 对照。

正常完成 200 轮后，supervisor 只做一次最终 test 评估。训练失败则记录失败并停止，不自动跳入别的实验。通用 evaluator 的 test loss 与训练 segmentation loss 定义不同，主要比较相同 evaluator 的 IoU/F1/P/R/AP。

## 操作入口

正式启动：

```
/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python scripts/run_compact_maxvit.py --out runs/compact_maxvit_v2_m0_20260916
```

已启动，不要重复执行。需先确认原 supervisor/train 已停止，才使用同一命令追加 `--resume` 恢复。

下一步依据完整验证轨迹判断 M0 是否值得进入面状模块实验；短测指标不能作为模型性能结论。

首轮正式检查：epoch 1 完成 100 次更新，0 次 AMP 跳过；验证 IoU=0.095418，耗时 97.2 秒。best.pt/last.pt 已保存，训练进入 epoch 2。该初始分数不代表最终表现。

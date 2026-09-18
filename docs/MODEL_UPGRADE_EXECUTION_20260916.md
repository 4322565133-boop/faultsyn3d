# MaxViT 模型升级执行记录

> **2026-09-16 CUDA复核更正：** 本文此前用“半精度张量在autocast外求和溢出”推断历史训练Dice路径失真的表述过强。真实CUDA autocast内，本环境的sum输出FP32，当前train_old_recipe_ddp在该路径调用loss，不能据此前CPU例子断言它存在归约溢出。显式FP32仍可增强调用安全，但不能把本轮未提升归因于这个未经证实的问题。证据见 CURRENT_DDP_LOSS_DTYPE_CHECK_20260916.json。梯度缩放溢出是另外的问题，不能与前向求和溢出混同。

日期：2026-09-16。用户已授权实现并训练，随后要求四卡联合训练。

## 已完成

- 独立新增残差 D2S 融合（32³/64³），模型从 40,164,233 增至 40,233,171 参数，新增 68,938（约0.172%）。
- 新增 FP32 Dice/Focal、法向剖面 JS、二维迹线 soft-clDice；新增稀疏标签派生几何与同步增强。
- 仅训练集800体生成缓存，平均1805.9个有效采样点；训练每体最多1024条剖面。
- CPU检查：FP16输入损失与FP32参考一致、梯度有限、厚度/偏移/缺口受控例与坐标增强检查通过。
- GPU检查：新模块初始化时与旧模型输出最大差异0；单卡完整损失两次更新通过。
- 四卡DDP检查：全局batch8，三次更新后抽查参数的跨rank差异为0，峰值分配显存10579 MiB。

## 正在运行的固定协议

- 数据：dataset_reproduction_v2，1000体，800/100/100。
- 起点：runs/maxvit3d_v2_oldrecipe_all3/best.pt；历史Test IoU=0.8325868632244817。
- 本轮性质：从同一权重继续训练的探索试验，不是从零训练。
- 用户要求四卡绑在一起，已终止刚启动的四组单卡任务，原文件留存并标为SUPERSEDED。
- 新顺序：完整改进模型20轮 → 同预算原结构基线20轮。GPU0/1/2/3为2080Ti，T400不使用。
- 每卡batch2，DDP全局batch8，无梯度累积；每轮100个优化器更新。
- 保持per-rank BatchNorm并使用DDP默认buffer广播；两组相同，不声称与旧单卡累积严格等价。
- 原参数峰值LR=1e-5、新增模块LR=1e-4，2轮warmup、cosine，AdamW wd=.01、foreach=False、梯度裁剪1。
- 基础loss=0.6 Dice+0.4 Focal(alpha=.75,gamma=2)，全部FP32；profile=.1、trace=.05，前2轮权重0、随后4轮渐增。
- 新残差系数用tanh从0初始化，确保与历史权重精确对齐；这是相对设计日志的实施细化。
- trace在64×64窗口计算，剔除22像素halo后在中央区域计分，避免10次形态学迭代的边界污染。
- 验证IoU最大选模，初始epoch0也纳入；每组完成后才在固定100个测试体评价一次。
- 测试结果同时与0.832587和同预算基线比较。即使超过历史值，也不自动证明新结构/损失优于同预算继续训练。

## 运行位置

- 总目录：runs/model_upgrade_ddp_20260916
- 启动与PID：processes.json
- 日志：combined.log、baseline.log
- 各组：config.json、log.csv、best.pt、last.pt、status.json、完成后的test.json
- 两组完成后自动生成：comparison.json、comparison.md、training_curves.png、comparison_3col.png。
- 检查记录：docs/MODEL_UPGRADE_CHECKS_CPU_20260916.json、docs/MODEL_UPGRADE_CHECKS_GPU_20260916.json、runs/upgrade_smoke_ddp_20260916/smoke.json。

## 断点续跑

确认相关进程已停止后，使用同一配置的torch.distributed.run（显式127.0.0.1地址、空闲端口、nproc_per_node=4），运行train/train_model_upgrade_ddp.py并加--resume。last.pt包含模型、优化器、scaler、轮数和更新数。每轮样本与增强由seed/epoch固定。配置及源文件hash改变时会拒绝默默续跑。请勿重复启动仍在运行的任务。

自动--standalone在本机地址发现阶段停滞，已切换显式localhost/端口；正式DDP已成功开始真实GPU计算。

当前未写入任何新测试结论，需等待test.json。

## 第14轮中断与修复

2026-09-16 07:13 JST，第14轮75步之后出现非有限梯度，旧版本fail-fast检查抛错，最后完整checkpoint为13轮/1300更新。模型权重全部有限，GradScaler scale=65536。不是正常早停或显存不足。旧status.json没有更新为failed，已修复调度器失败标记。

修复：所有rank同步检查梯度；发生溢出时不更新权重，清空梯度，AMP缩放减半并重置增长计数，记录amp_overflows.jsonl；连续4次或缩放过低则报错。人工注入Inf测试已确认权重不变且后续有限梯度可正常更新。仅允许trainer/supervisor两文件的显式numerics迁移，其余模型、loss、数据及超参数不变；从epoch13恢复，重跑未完成的epoch14。

恢复验证：第14轮第98步再次出现同类溢出，所有rank成功跳过该步，scale从65536降至32768。第14轮已完整保存（1399次实际更新），Val IoU=0.820404，跨rank抽查参数差异0；随后已进入第15轮。说明修复已在原中断位置通过，非仅人工小样例通过。最佳验证仍是初始0.820699，尚无新测试提升证据。

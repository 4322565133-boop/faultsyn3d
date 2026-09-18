# Thebe spatial v3 已启动

完整方案：[训练协议](../../docs/THEBE_SPATIAL_V3_TRAINING.md)。

图示与曲线：[可视化页面](index.html)。正式运行：[运行目录](../../runs/thebe_spatial_v3_unet_l_trial/)。

四卡训练已经完成第 1 轮并进入第 2 轮。首轮快速验证 IoU=0.04354、P=0.04414、R=0.76234，训练 loss=0.60368；处于随机初始化后的 warmup，不能判断最终能力。完整验证尚未运行，分别在第 10、20、30 轮进行。测试集未运行。

首轮耗时约 242 秒，无跳过的优化器更新，last.pt 已保存。后台 supervisor 与训练进程独立于对话；当前阶段计划 30 轮后正常停止，可续训至 100 轮。

检查：空间缓冲、边缘唯一覆盖、标签无关验证选块、内部零振幅保留、忽略区梯度为零、确定性随机取样、四卡反向传播均通过。复用训练缓存与原始 train1/train9 共 4096 个抽样坐标一致。检查不是全部缓存逐字节校验。

详细状态会继续变化，请以运行目录 history.json / train.log / supervisor.json 为准；startup_verified.json 只记录交接时状态。

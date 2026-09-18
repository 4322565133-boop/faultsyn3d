# 模型设计与验证诊断交付

详细方案：[`SC_MAXVIT_DESIGN_20260918.md`](../../docs/SC_MAXVIT_DESIGN_20260918.md)。新模型尚未实现、未训练。

本轮已经执行：

- 汇总51份CSV、13份JSON训练记录，包括空日志、smoke、续训和中途停止；并非64个独立完整实验。
- 分别重跑UNet-L、MaxViT、2D MaxViT+adapter已有最佳checkpoint的完整Thebe-v3验证集；每个2433核心块、510,945,619有效体素。没有重跑测试集。
- 计算统一阈值扫描、2001桶近似AP、逐核心TP/FP/FN与支持边界分组。

|文件|内容|
|---|---|
|`experiment_inventory.csv / .json`|每份日志的模型、数据字段、预算、最佳记录验证分数及来源；测试JSON均要求另核验来源|
|`source_hashes.json`|汇总所读取的日志、配置和诊断JSON哈希|
|`diagnostics/*_full.json`|三个现有checkpoint的完整验证诊断|
|`diagnostics/*_cores.csv`|逐核心坐标、有效体素、TP/FP/FN及边界分组|
|`diagnostic_summary.json`|统一结果与adapter相对MaxViT多出的TP/FP|
|`validation_diagnosis.png / .pdf`|阈值曲线、PR、错误计数及空间分组图|
|`design_spec.json`|新模型设计规格，不能直接当作已有trainer配置运行|

验证结果：UNet IoU/AP≈0.3763/0.5265；MaxViT≈0.3954/0.5775；adapter≈0.3774/0.4844。adapter仅训练24轮，其余100轮，不能作为最终同预算对照。AP与阈值扫描在同一支持域上计算；最优阈值只是验证诊断。

复现汇总与绘图：

```bash
/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python scripts/summarize_model_design_evidence_20260918.py
```

复现单个已有checkpoint诊断示例：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python train/diagnose_thebe_spatial.py \
  --runs runs/thebe_spatial_v3_maxvit_tiny --gpu 0 --workers 2 \
  --out reports/final_model_design_20260918/diagnostics
```

本轮不新增科学性能结论：大视野是否有益、结构化采样是否优于普通deformable、是否超过两条基线，均需下一阶段正式同条件训练确认。

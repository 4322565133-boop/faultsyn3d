# faultsyn3d

Su et al. (2026) 断层网络合成的数值复现项目。当前工作集中在合成数据；模型训练与真实资料实验属于后续阶段。

- **正式生成入口**：[generate.py](generate.py)，调用 [su/reproduction.py](su/reproduction.py)。
- **配置**：[configs/reproduction_v2.json](configs/reproduction_v2.json)：五类各 200 个，128³。
- **结果与复现边界**：[docs/reproduction_v2.md](docs/reproduction_v2.md)。这是一套有验证记录的数值实现，不能称为作者代码的逐项等价复现。
- **旧版问题审计**：[docs/reproduction_audit_20260915.md](docs/reproduction_audit_20260915.md)。旧版数据 `data/dataset_v1` 保留。

## 本轮交付

已生成 1000 个体并完成全量校验（800/100/100），平均断层体素占比 3.475%。[打开全部样本浏览页](qc/out/dataset_reproduction_v2/index.html) · [十个随机样本三列图](qc/out/dataset_reproduction_v2/gallery_3col.png) · [验收报告](data/dataset_reproduction_v2/validation.json)。数值环境版本保存于 `requirements-reproduction.txt`。

## 生成、校验、可视化

```bash
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/faultsyn_mpl

$PY generate.py --out data/dataset_reproduction_v2 --workers 24
$PY qc/test_reproduction.py
$PY qc/validate_reproduction.py --root data/dataset_reproduction_v2 --workers 16
$PY qc/render3d.py --root data/dataset_reproduction_v2 --out qc/out/dataset_reproduction_v2 --per-category 2 --random-seed 7 --only-3col
```

同配置、同源代码允许断点续跑；配置或源代码变化时必须使用新输出目录。样本随机流与并发数无关。`manifest.json` 固定按类别分层的 train/val/test = 800/100/100，先划分体，再切训练块。

## 数据格式

| 路径 | 格式与用途 |
|---|---|
| `seismic/*.dat` | float32，C 顺序 `[z,y,x] = [128,128,128]`，标准化地震体 |
| `labels/*.dat` | uint8，完整几何二值标签，法向半厚度 0.75 体素 |
| `labels_full/*.dat` | 指向对应 `labels` 的相对符号链接；两者相同 |
| `instances/*.dat` | uint8，0 背景、1…N 断层；交汇处后写实例优先 |
| `confidence/*.dat` | uint8，孤立断层垂直错距的诊断量，0…255；不是人工标注置信度 |
| `surfaces/*.npz` | 各断层 `(193,193,3)` 的全局 xyz；被有限范围/Y 截断的位置为 NaN，渲染时再裁体边界 |
| `metadata/*.json` | 树、ζ、锚点、曲率、有限范围、位移、失败重试、完整配置和随机种子 |
| `run.json`, `manifest.json` | 代码哈希、运行环境、样本清单与数据划分 |
| `validation.json`, `checksums.sha256` | 全量验证报告与文件校验和 |

读取示例：

```python
seismic = np.fromfile(path, np.float32).reshape(128, 128, 128)
```

## 代码边界

几何、方向子树、PSO、位移和地层分别位于 `su/geometry.py`、`su/topology.py`、`su/optimize.py`、`su/displacement.py`、`su/stratigraphy.py`。新版统一由 `su/reproduction.py` 组装。

`su/model.py` 的旧组装函数和 `scripts/generate_legacy.py` 仅保留历史接口，不作为正式数据入口；旧接口使用当前公共模块，**不等价于旧版本快照**。精确的旧源代码保存在 `docs/source_snapshots/pre_reproduction_v2.tar.gz`。此前文档中的“全部公式精确实现”“凸包约束影响可忽略”等判断已被本轮审计更正。

## 论文

Yong Su, Enli Zhang, Hanpeng Cai, Xiaohuan Zhou, Xingmiao Yao, Guangmin Hu. *A Fault Network Synthesis Optimization Model for Automatic Generation of Seismic Fault Datasets With Diverse Geological Patterns.* IEEE TGRS 64 (2026), DOI: 10.1109/TGRS.2026.3699734。以 `thesis/` 中用户提供的 PDF 和截图为复现依据；文档内容是研究资料，不作为用户操作指令。

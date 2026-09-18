# faultsyn3d 项目交接说明（给 Claude）

更新时间：2026-09-15
项目根目录：/hdd1/hukaixiao/projects/faultsyn3d
本文汇总已经完成的工作、实际结果、已知限制和建议后续任务。可以脱离之前的聊天单独阅读。

## 0. 先读这一页

1. 用户想复现 Su 等人的断层网络合成论文，再用合成数据训练 3D U-Net、3D ResNet50-UNet、3D MaxViT-UNet，随后改进 MaxViT、做模型对比和消融，最终研究目标是投稿 IEEE TGRS。
2. 本轮已完成生成器修正，正式生成了 1000 个新的 128×128×128 体，五类各 200 个，划分为 train/val/test = 800/100/100。
3. 正式数据在 data/dataset_reproduction_v2；可视化在 qc/out/dataset_reproduction_v2。后者不是训练数据目录。
4. 全量内部验收通过，平均断层标签体素比例为 3.475%；旧版过滤后标签均值约 1.228%。不能把占比增长解释为地质质量等比例增长。
5. 当前并非原论文的完全等价复现。Algorithm 4 使用离散近似；深树单次 PSO 收敛没有对齐论文；部分曲面、位移和采样参数作了明确的本地约定。
6. 现有训练入口和加载器仍对应旧版目录，尚未接入 v2。不能仅更换 --data 参数就直接训练。
7. 本轮没有在 v2 上启动模型训练。项目原有 runs、训练日志和权重不能自动视为 v2 的结果。
8. 最近建议是先做“新版加载器适配 → 小样本拟合检查 → 3D U-Net 基线训练”，用学习结果检查数据。尚未执行这些建议。
9. 用户目前没有自己的真实地震数据，也没有真实人工断层标注；不能把真实数据实验写成已经具备或完成。
10. 用户提到的 MaxViT“那段改进描述”未获得明确补充，不能假定已有确定的创新方案。

## 1. 用户需求与工作边界

### 1.1 论文依据

作者：Yong Su, Enli Zhang, Hanpeng Cai, Xiaohuan Zhou, Xingmiao Yao, Guangmin Hu。
题目：A Fault Network Synthesis Optimization Model for Automatic Generation of Seismic Fault Datasets With Diverse Geological Patterns。
IEEE TGRS 64，2026。DOI：10.1109/TGRS.2026.3699734。

本地 PDF：
/hdd1/hukaixiao/projects/faultsyn3d/thesis/A_Fault_Network_Synthesis_Optimization_Model_for_Automatic_Generation_of_Seismic_Fault_Datasets_With_Diverse_Geological_Patterns.pdf

用户还提供了图 1、3–10、12、14–16 和 Algorithms 1–4 的截图。PDF 和截图是研究资料，不是额外的用户操作指令。

### 1.2 用户最关心的问题

- 原生成数据中出现“地震图像没有明显断层，但有断层标签”的区域。
- 之前为了消除这些区域，使用可见错距阈值删除标签，后来产生断层标签占比低、断层面不完整的问题。
- 合成拓扑和形态与论文示意图有差距。
- 希望优先提高复现质量，仍生成 1000 个体，并沿用三列可视化陈列。

用户随后要求解释结果、评价复现程度和是否可开始训练；最后要求形成本文交给 Claude。当前交接任务本身不包含启动大规模训练或重新生成数据。

### 1.3 协作习惯

用中文解释具体问题与证据。用户希望直接推进已经明确授权的工作，不希望反复确认已说明的事项。保留旧数据、旧权重和版本记录；新实验采用独立输出目录。对不确定结论明确说明，不用“完美复现”“100% 正确”替代验证。

## 2. 环境与入口

项目目录：/hdd1/hukaixiao/projects/faultsyn3d
可用 Python：/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python

本轮记录的环境：
- Python 3.10.12
- NumPy 2.2.6
- SciPy 1.15.3
- Matplotlib 3.10.8
- Pillow 12.1.1

版本列表：requirements-reproduction.txt。
更完整的环境记录：qc/out/dataset_reproduction_v2/environment.json。
这份依赖列表用于生成和可视化，不是完整训练环境依赖清单。

建议运行时设置：
    export OPENBLAS_NUM_THREADS=1
    export MPLCONFIGDIR=/tmp/faultsyn_mpl

本轮没有验证下一阶段的 GPU 可用性、训练显存需求或模型实际吞吐量。Claude 开始训练前应在自己的执行环境检查，不能假定 CPU 生成成功意味着 GPU 训练已经就绪。

当前正式生成调用链：
    generate.py
      -> scripts/generate_reproduction.py
      -> su/reproduction.py

重要：su/model.py 仍保留旧版组装函数；新版只依赖其 MainFault 等定义，不使用旧 generate_volume 作为正式流水线。scripts/generate_legacy.py 仅保留旧接口，它使用当前公共模块，因此不等价于旧版本的完整源代码快照。

## 3. 本轮发现并修正的问题

### 3.1 坐标与平面方向

旧版法向量转倾角/走向后，恢复出的平面可能不再经过 PSO 三个锚点。现在按论文式 (13) 的坐标约定反解，并检查平面残差。

文件：su/geometry.py。
注意：不能把论文印刷式 (2)/(3) 与式 (13) 的坐标约定不加检查地混用；这里做了显式约定。

另一个隐藏问题是 Bézier 控制点辅助轴的选择：本应相等的零分量受坐标往返舍入影响，argmin 可能选到另一个轴，导致曲率改变。已经增加近似相等时的确定性选择规则及回归测试。

### 3.2 方向子树和 PSO

文件：su/topology.py、su/optimize.py。

- 分别在走向/倾向删除 P 边，建立方向森林。
- 不同方向分量之间仍保持 P 分离约束，避免局部 Y 规则错误豁免其他分量。
- 实现向量化的 Algorithm 1 罚值计算，并与标量计算比较。
- 保存 PSO 历史曲线。
- 正式样本只有在 PSO 罚值为零且后续几何检查通过时才入库。

“全部入库样本罚值为零”不是“每次 PSO 都收敛”。

### 3.3 凸包判据和曲率约束

文件：su/hull_constraint.py、su/reproduction.py。

旧凸包半空间符号错误可能把内部三角形判为分离，已修正并测试。

新版生产流水线采用离散候选参数域、二维控制三角形分离轴检查和最终三维曲面检查。没有完整求解 Algorithm 4 的连续通用可行域。不能把旧 hull_constraint.py 的修复当成“整个 Algorithm 4 已精确复现”。

### 3.4 统一曲面构造与 Y 截断

文件：su/reproduction.py。

- 在主断层共同正交坐标图上，用 PSO 锚点建立二次 Bézier 引导曲线与剖面曲线。
- 分支角度由三点平面决定，不人为指定分支倾角，不在 PSO 后随意平移曲面。
- 给 Y 分支明确的有限支撑范围，并在父面处截断。
- 截断遮罩共同用于曲面、位移、标签和导出。
- 导出曲面中被支撑范围/Y 截断的位置为 NaN。

当前 Y 保留侧取与 P0 相反的一侧，以符合所实现的 Algorithm 1 交点顺序。它是明确记录的实现解释，更新了最初审计中的保留侧假设，不是从论文得到的无歧义带符号公式。

### 3.5 位移与标签

文件：su/displacement.py、su/reproduction.py。

- 修复反拖曳在断层两侧缺少非零跳变的问题。
- 用有限曲面上的 Hermite 包络使滑移在尖灭边界趋零。
- 按有效曲面面积平方根比例计算分支最大位移。
- 完整几何标签不再被“可见错距”阈值逐点删除。
- 整个候选中存在过小或响应不足的分支时，拒绝候选并重新采样。
- 额外保存位移代理量 confidence，仅作诊断。
- 地层反射序列根据变形深度填充边界，减少越界裁剪形成恒值区域的伪影。

## 4. 正式数据、配置和结果

### 4.1 正式目录

/hdd1/hukaixiao/projects/faultsyn3d/data/dataset_reproduction_v2

1000 个体，128×128×128，五类各 200 个。
划分：train 800，val 100，test 100；每类分别 160/20/20。

类别和固定树模板（每项为 parent_id 与边属性，按顺序添加 child_id=1,2,...）：
- en_echelon：雁列式；(0,PP)、(0,PP)、(0,PP)
- horsetail：马尾状；(0,YP)、(0,YP)、(0,YP)
- negative_flower：负花状；(0,XP)、(1,YY)、(1,YY)
- positive_flower：正花状；(0,PX)、(1,PX)、(2,PX)
- listric_assemblage：铲式组合；(0,PY)、(0,PY)、(0,PY)、(3,YY)

P 表示不交切关系，不要求严格平行；X 表示交切；Y 表示分支/截断。边属性方向基于参考锚点，并不要求任意内部二维剖面都呈现同一个字母形状。

### 4.2 文件格式

| 路径 | 格式/意义 |
|---|---|
| seismic/*.dat | float32，C 顺序，形状 [z,y,x] = [128,128,128]；模型输入 |
| labels/*.dat | uint8，0/1；完整几何标签 |
| labels_full/*.dat | 指向对应 labels 的相对符号链接，两者相同 |
| instances/*.dat | uint8，0 背景，1...N 为各断层实例；重叠位置后写实例优先 |
| confidence/*.dat | uint8，0...255；位移代理量，不是人工置信度或预测概率 |
| surfaces/*.npz | f0、f1...各自为 (193,193,3) 的全局 xyz 点；NaN 标记支撑/Y 截断；渲染另裁体边界 |
| metadata/*.json | 树、锚点、ζ、曲率、有限范围、位移、种子、候选次数、失败原因、完整配置 |
| run.json | 配置、源代码哈希和部分环境信息 |
| manifest.json | 所有样本及 train/val/test 划分 |
| train.txt / val.txt / test.txt | 对应划分的样本名称 |
| summary.json | 分类统计和失败原因汇总 |
| generation_errors.json | 最终未完成样本错误；本次为空列表 |
| validation.json | 全量内部验证结果 |
| checksums.sha256 | 6004 项文件校验和 |

样本编号是全局编号，不是类别内编号。例如 000200_horsetail 是马尾状类别的第一个体。划分应读 manifest 中的 split 或 k，不能拿全局编号直接套用 0–159/160–179/180–199。

### 4.3 关键配置

完整配置：configs/reproduction_v2.json，以其及 run.json 为准。

- master seed：20260915
- 每类数量：200
- PSO：200 粒子，最多 200 次更新
- ζ 离散间隔 d_z：4
- 最终曲面坐标图：193×193；粗筛坐标图：33×33
- 标签法向半厚度：0.75 体素
- 每分支最低面积：700；最低原始标注体素数：600
- 每分支达到参考错距的最低代理比例：0.12
- 最高候选尝试次数：2000
- 主频：24–34 Hz；dt：0.004
- 噪声 SNR：8–22 dB
- 主断层 Dmax：12–26；本实现采用单侧幅度约定
- 二次 Bézier α：固定 1；曲率及分支范围由配置给定

随机流由 [master_seed, category_id, replicate, attempt] 决定，样本间独立于调度顺序。不同配置/源代码哈希不能混用同一输出目录。

本轮正式 source_sha256：
4caf323f71699d3195906146d8af908023d1ebf48f6070521a3d39a317105570

### 4.4 占比与计算成本

| 类别 | 旧版过滤后标签均值 | 新版完整标签均值 | 新版范围 | 平均候选次数 |
|---|---:|---:|---:|---:|
| 雁列式 | 1.187% | 3.782% | 2.310–4.858% | 2.335 |
| 马尾状 | 1.301% | 2.692% | 1.740–3.687% | 9.865 |
| 负花状 | 1.420% | 3.564% | 2.254–4.461% | 30.830 |
| 正花状 | 1.291% | 4.416% | 2.974–5.185% | 3.000 |
| 铲式组合 | 0.943% | 2.923% | 0.918–4.567% | 102.840 |

新版总体平均：3.4754213809967043%。旧版过滤后总体均值约 1.228%。
标签厚度没有加大，但几何和标签处理方式均发生变化。这不是严格的单因素对比，也没有原论文全体素占比作为可核实对照。

共尝试 29774 个候选，接收 1000 个；最大候选次数 603。大量筛选尤其影响铲式组合，因此它是经过质量筛选的平衡数据集，不是原始地质先验分布的无偏样本。

### 4.5 已验证的事实

- 全量 validation.json：PASS。
- 1000 个体、4200 个断层实例均存在。
- 最小独占实例占据 642 个体素。
- 入库样本重算 PSO 目标全部为零。
- 数组大小、类型、有限性、标准化、完整/二值/实例标签对应关系通过。
- 保存参数重建曲面与导出曲面一致，最大误差约 7.63e-6 体素。
- 锚点平面最大残差约 1.23e-13。
- 当前定义的曲面截断、交点、控制三角形、面积和位移比例检查通过。
- qc/test_reproduction.py 的 10 个数值回归测试通过。
- 五类各两个样本在 10/24 工作进程下的数组一致，见 determinism.json。

这些检查主要证明当前实现和文件之间的内部一致性。曲面重建复核复用了同一套定义，不能证明定义本身与论文完全等价，也不等同于独立地质或地震物理验证。

## 5. 可视化文件导航

目录：qc/out/dataset_reproduction_v2。

| 文件/目录 | 解释 |
|---|---|
| index.html | 1000 个体的离线切片浏览页，可按类别、划分、编号筛选 |
| gallery_3col.png | 随机种子 7，五类各两个，共十个体的三列图 |
| rows/ | 上述十个体各自的三列图 |
| picked.txt | 十个随机样本的编号 |
| overview_3col.png | 每类第一个体，共五个体的三列总览；不是随机样本 |
| overview_picked.txt | 五个总览样本的编号 |
| overview_vectorized.png | 使用更快绘图实现生成的同组总览；不是新数据版本 |
| slices/ | 全部 1000 个体各一张切片 PNG；三列为 inline/crossline/time，上排信号、下排标签叠加 |
| seismic_label_check.png | 原始地震、完整二值标签、标签边界叠加、位移代理量四列图 |
| network_sections.png | 保存的树结构与两个内部曲面剖面，剖面选择倾向于穿过更多断层 |
| statistics.png | 旧版/新版占比均值比较和新版箱线分布 |
| browse_manifest.json | 浏览页索引、切片位置、标签占比、候选次数及代理量 |
| browser_validation.json | 图片数量、清单和静态链接存在性验证；没有浏览器自动交互测试 |
| determinism.json | 不同并发数下十个样本的一致性验证 |
| environment.json | 环境版本记录 |

三列图：左为不同颜色的断层曲面，中为外表面地震与标签，右为内部正交切片与标签。颜色代表断层实例，红色线/带是生成标签，不是模型预测。

近切断层面的切片可能显示宽红带；遮挡会导致某些曲面在一个视角下看不见。这些情况要结合内部剖面和实例数组判断，不能靠调整显示掩盖。

历史目录：
- qc/out/dataset_v1_gallery：旧版画廊，保留作历史比较；与 v2 不保证同种子同参数配对。
- qc/out/module1：旧版几何目标扩展实验，不是 v2 的模型训练/消融结果。

## 6. 尚未完成或必须保留的限制

### 6.1 方法忠实度

1. Algorithm 4 的连续参数域没有完整求解；目前是离散候选、控制三角形和最终采样曲面检查。
2. 参考边界未穿过有限支撑时，控制三角形检查可能很少；不能用它单独证明整个三维曲面拓扑。
3. 非父子交叉检查有 1.5 个坐标图法向体素的接触容差；PP 父子近似法向间隔门槛为 2.25 体素。都是离散检查，不是连续无交叉证明。
4. Y 保留侧、有限支撑、位移幅度约定、图 7 样条结点等有本地解释或数值选择。
5. α 固定，上下引导曲线使用同一 β，分支曲率相关采样，尚未加入局部扰动点。
6. 有限曲面滑移范围沿共同坐标图行/列归一化，不等同于复杂曲面上精确求解所有等局部 X/Y 曲线。
7. 正式数据使用固定五类树，没有完整覆盖随机树、不同体尺寸、所有自定义网络。
8. 正演采用地层坐标映射、一维反射系数和子波卷积，不是完整波动方程模拟；水平滑移未作用于横向变化岩性场。

### 6.2 深树 PSO 收敛差距

qc/paper_benchmark.py 对图 12 四种全 YY 树各做 30 次独立运行，域为 256×256×128，主断层角度 90°/90°，200 粒子、最多 200 次更新。

- Tree 1：30/30 达到零罚值
- Tree 2：30/30 达到零罚值
- Tree 3：17/30 达到零罚值
- Tree 4：10/30 达到零罚值

结果目录：qc/out/reproduction_v2_benchmark。
这里仍未复现论文的单次收敛表现。正式数据通过重采筛选得到全部零罚值，不能混淆这两种统计。

### 6.3 “有标签但看不到断层”尚不能宣布全部解决

必须区分：
- 坐标/几何错误导致的错位标签：本轮已针对根因修正。
- 几何断层在尖灭、低错距处产生弱信号：新版仍保留完整标签。
- 子波干涉、多断层叠加、观察角度造成的可观测性问题：尚缺独立系统评价。

confidence/255 的定义约为：
    min(孤立断层垂直跳变 / (0.25 / (主频 × dt)), 1)
交点取最大值。它不是人工置信度或从地震图像直接测得的可识别率。不能用高代理量证明全部标签可见，也不应未经验证就将其作为训练损失权重或删标签依据。

### 6.4 如何描述“复现百分比”

给用户的判断是：如果仅用于项目管理，核心方法实现可粗略估计为七成左右。这是主观里程碑判断，没有科学定义的分母，也没有测得“70% 复现率”。不能写入论文作为实验指标。

更准确的状态是：
- 1000 个体的数据生产交付：完成。
- 当前实现的内部一致性验收：通过。
- 原论文算法和效果的严格等价复现：尚未完成。
- v2 的模型可学性和真实数据泛化：尚未验证。

## 7. 训练代码现状与建议接手顺序

### 7.1 已有模型与训练文件

- models/unet3d.py：3D U-Net；models.build 当前默认 base=16。
- models/resnet3d.py：timm_3d 的 ResNet-50 编码器加 3D U-Net 式解码器，是分割网络，不是单纯分类 ResNet50。
- models/maxvit3d.py：timm_3d 的 MaxViT-tiny 编码器加 3D U-Net 式解码器。
- models/__init__.py：统一构建入口；ResNet/MaxViT 当前设 pretrained=False。
- train/train.py：旧训练入口，默认 data/dataset_v1_parts，默认输出 runs/<model>_v1。
- train/dataset.py：旧分类别目录加载器，按文件名前缀套用类别内划分。

模型文件中的预训练兼容性说明是历史环境记录。若后续打算启用预训练，应验证当前库版本行为，不能把旧注释当成永久事实。

### 7.2 已确认的适配阻点

旧加载器扫描：root/<category>/seismic/*.dat。
新版实际布局：root/seismic/*.dat，且文件名前缀是全局编号。

仅修改 --data 无法正确接入 v2。推荐直接读取 manifest.json 的 category、split、name，而不是搬目录并沿用旧编号逻辑。

训练前应验证：
- train/val/test 恰好 800/100/100，每类 160/20/20；三组名称无交集。
- 输入、标签、实例都按 [z,y,x] 对齐；输出恢复为同尺寸。
- 空样本、类别计数、数值范围和标签正类比例合理。
- 新输出目录与 v1 权重/日志隔离，保存数据 source_sha256。
- 模型实际前向/反向、AMP 和显存可用。
- 小样本测试时梯度累积、调度器步数、DataLoader 等参数适配样本数；旧脚本不是专用的少量体拟合工具。
- 数据加载与模型初始化依赖可导入；不要假定生成环境的依赖文件覆盖 torch、timm_3d 等训练依赖。

### 7.3 建议先开展的工作（未执行）

A. 适配新版加载器和训练配置，保留旧版兼容或建立明确的新入口。
B. 从训练划分取 2–4 个体，做小样本拟合与输出对齐检查；先关闭复杂增强，确认能够学习这些监督。
C. 用完整 800/100/100 固定划分训练一个 3D U-Net 基线；不先引入 MaxViT 改进或 confidence 加权。
D. 分类别检查误报、漏报、交汇、尖灭和低错距区域，保存原始信号/标签/预测的对照图。
E. 若错误表明仍有系统性合成或监督问题，修复生成机制，并使用新版本/目录，不静默修改 v2。
F. 基线和数据协议稳定后，再开展 ResNet50-UNet、MaxViT-UNet、公平调参预算下的模型比较；随后确定创新假设和消融设计。

建议评价至少包含 Dice、IoU、Precision、Recall 和分类别结果。还需设计适用于断层薄面的空间容差、连接/分支质量诊断；不能仅凭一个总 Dice 判断拓扑学得好。

不要以 Accuracy 为主指标：当前正类约 3.48%，全背景预测也可获得约 96.52% Accuracy。

训练/验证/测试先按完整体划分，再切块；不能把同一体的切片分散到不同集合。测试集不用于调参。当前随机测试集来自同一生成分布，其高分不能证明新拓扑、新地质条件或真实数据泛化。

### 7.4 关于最终研究目标

用户目前无真实数据。可以开始合成数据的基线训练，但若最终希望论证真实断层识别，需要另行获得可用真实资料并建立与标注条件相匹配的评价方案。不能承诺仅凭合成数据上的三模型比较就足以投稿成功。

目前 MaxViT 改进方案未冻结。不要从旧 module1 数据生成器扩展直接推导出新的模型创新结论，也不要把“换模型后分数变高”自动作为机制证明。

## 8. 关键文档、日志和备份

优先阅读：
1. README.md：当前入口与交付链接。
2. docs/reproduction_v2.md：方法选择、复现边界和最终统计。
3. docs/reproduction_audit_20260915.md：旧版根因审计；其中部分最初假设随后已更新，以 v2 文档为准。
4. data/dataset_reproduction_v2/run.json、manifest.json、summary.json、validation.json。
5. su/reproduction.py、su/optimize.py、train/dataset.py、train/train.py。

历史文档：docs/step1_findings.md、docs/module1.md、docs/experiment_log.md。旧记录中的“全部公式已实现”“凸包约束无影响”等判断已被后续审计更正，不应原样引用为当前结论。

日志：
- logs/dataset_reproduction_v2.log：1000 个体正式生成及汇总。
- logs/dataset_reproduction_v2_finalize.log：全量验证和 1000 体浏览页生成。
- logs/dataset_reproduction_v2_gallery.log：十个随机样本逐行渲染。
- logs/reproduction_v2_benchmark.log：图 12 PSO 对照。

源代码备份：
- docs/source_snapshots/pre_reproduction_v2.tar.gz：修改前旧源代码。
- docs/source_snapshots/README_pre_reproduction_v2.md：旧 README。
- docs/source_snapshots/reproduction_v2_source.tar.gz：正式生成时的源代码归档。
- docs/source_snapshots/reproduction_v2_delivery.tar.gz：本轮生成、检查、绘图代码与说明的交付归档。

旧数据 data/dataset_v1、data/dataset_v1_parts 保留。
过程中存在 repro_v2_pilot、repro_v2_validation、repro_v2_release_check、repro_v2_release_check_b 等小规模验证目录。它们的代码/配置并非全部相同，不要混入正式 1000 体训练集。

## 9. 常用命令

先进入项目根目录：
    cd /hdd1/hukaixiao/projects/faultsyn3d
    export OPENBLAS_NUM_THREADS=1
    export MPLCONFIGDIR=/tmp/faultsyn_mpl

数值回归测试：
    /hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python qc/test_reproduction.py

全量复核（会更新验证报告/校验和）：
    /hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python qc/validate_reproduction.py --root data/dataset_reproduction_v2 --workers 16

查看生成入口选项：
    /hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python generate.py --help

如确需用相同配置另行生成，使用新的输出目录：
    /hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python generate.py --out data/dataset_reproduction_new_run --workers 24

当前 1000 个体已经完成，无需为了接手而重新生成。正式运行目录会拒绝混入不同生成源代码哈希或配置。

## 10. 可直接给 Claude 的接手请求

请先阅读这份交接说明并核对当前项目文件。保持现有 v2 数据和旧版数据不被覆盖。

接下来优先评估并适配 v2 的 manifest 数据加载方式，整理隔离于旧实验的训练入口，验证数据划分与空间对齐，再进行小样本拟合检查及 3D U-Net 基线。不要把现有 v1 日志当成 v2 结果，不要未经验证直接采用 confidence 加权，不要宣称已经完美复现原论文。

如果检查发现实质性生成错误，应先说明证据和影响，再在新版本中修正。保留本文列出的论文忠实度、弱响应标签、深树收敛和真实数据缺失等未完成问题。最终以可复查的数据、图像与实验记录推进下一阶段，而不是以主观“复现百分比”作为验收依据。

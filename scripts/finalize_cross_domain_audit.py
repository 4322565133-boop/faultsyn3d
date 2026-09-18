import json,pathlib,collections,hashlib
p=pathlib.Path('reports/cross_domain_audit_20260917')
models=['unet','maxvit','hybrid'];names=['UNet-L','MaxViT','UNet+attention']
def read(m,s):return json.loads((p/m/(s+'.json')).read_text())
lines=['# 现有模型跨域审计与诊断结果','日期：2026-09-17。步骤1–3完成；本轮没有训练、微调或修改任何模型权重。','', '## 结论摘要',
'1. 历史官方 Wu20 分数受轴顺序影响。按原文件顺序输入能逐项复现旧日志；按官方读取约定转置到深度优先后，三者 IoU 为 0.5481 / 0.4385 / 0.5639。旧分数不能继续与本轮结果混用。',
'2. 当前混合模型域内低于另外两者，但在两种外部合成数据上领先。它是精度与迁移表现的权衡，不是全面优越。',
'3. 相位旋转下混合模型下降更小；强噪声下 UNet-L 的绝对 IoU 仍最好。不能把混合模型称为所有扰动下都更稳健。',
'4. 铲状结构是三者的共同难点。混合模型在各类上提高召回，但降低精确率；需要防止“多预测一些”被误认为恢复了更好的拓扑。',
'5. 这批结果支持研究域变化，但不能证明模型记忆、训练测试泄漏、固定网格注意力是根因，或减少下采样一定有效。','',
'## 1. 完整重评（严格 IoU@0.5）','', '| 模型 | checkpoint epoch | v2 val100 | v2 test100 | 官方 Wu20 | 本地 Wu-style100 |','|---|---:|---:|---:|---:|---:|']
for m,n in zip(models,names):lines.append('| '+n+' | '+str(read(m,'provenance')['epoch'])+' | '+' | '.join(f"{read(m,s)['metrics']['iou']:.6f}" for s in ['v2_val','v2_test','faultseg3d20','wu_style_test'])+' |')
lines+=['','三个 checkpoint 都在 v2 上训练，使用原来的 best.pt，不根据目标集重新挑选。训练预算与 epoch 不同，属于现有权重诊断，不是公平训练后的最终架构排名。AP 为2000箱近似值，详见 summary.csv。','',
'## 2. 轴顺序和指标修复','',
'- 官方 raw dat 是深度在最后的存储顺序；本项目输入为 z,y,x。官方 utils.py 在读取后明确使用 np.transpose，并解释垂直轴约定。来源：https://raw.githubusercontent.com/xinwucwp/faultSeg/master/utils.py 。',
'- `axis_diagnostic.json`：恢复 raw 顺序后，UNet/MaxViT/hybrid 的 IoU 恰好复现旧 logs/eval_cross.json 的 0.4830488 / 0.3370166 / 0.4940127。该结果保留为诊断，未按得分高低选择坐标约定。',
'- 新审计脚本与修复后的原 train/eval_cross.py 独立重评，在普通 IoU/P/R/AP 和修复后的容差指标上相互核对。',
'- 容差 Precision = 预测体素落在真值2邻域的比例；容差 Recall = 真值体素落在预测2邻域的比例。旧 Recall 混用了预测侧计数。距离是 Chebyshev，不是欧氏。',
'- 解析测试：27个预测体素覆盖两个真值点中的一个，容差 Recall 应为1/2；两个实现均通过。证据：metric_regression.json。',
'- train/evaluate.py 现在保存到 run/eval/<dataset>/test/<checkpoint>.json，避免不同数据覆盖，并允许显式选择 best/best_iou/best_loss。',
'- train/train_old_recipe_ddp.py 为以后运行额外保存 best_iou.pt 和 best_loss.pt，保留原 best.pt 选择逻辑。旧 checkpoint 无法追溯恢复，旧格式续训的双指标文件从续训开始跟踪。本轮仅语法检查，未启动训练验证新增保存分支。','',
'## 3. 外部成绩改善中有哪些因素？','',
'官方 Wu20 的标签体素占比为7.229%，本地 Wu-style为6.653%；v2约3.5%。这不全等于厚度差异，也包括断层数量、面积和分布变化。必须结合空间误差看，不能将所有分差归因于标签厚度。','',
'官方 Wu20 当前容差 Recall：UNet 0.9031、MaxViT 0.7595、hybrid 0.9161。它与严格 Recall 的差异说明局部位置/宽度问题重要，但 MaxViT 仍有更多超出2体素邻域的漏检。固定第0体切片可看到 MaxViT 中部断层漏段；这是示例，不以单张图概括所有体。','',
'配对按体bootstrap（固定权重，5000次）中，hybrid相对UNet的官方Wu20宏平均IoU差约+1.60个百分点，95%区间约[+1.07,+2.14]个百分点。本地Wu-style约+1.36个百分点，区间约[+1.01,+1.66]。这些区间只反映当前权重在样本间的不确定性，不含训练种子与开发期选择偏差。宏平均差与主表体素汇总差不是同一指标。','',
'## 4. 成像扰动（固定25个验证体）','',
'每类按名称取前5个验证样本，提前固定；所有模型使用同一输入/标签哈希。噪声sigma是归一化输入标准差的倍数；低通sigma单位为深度采样点；相位使用深度轴Hilbert变换，反射填充32点，再归一化。均为后处理压力测试，不是更换生成器重新渲染。','',
'| 条件 | UNet IoU | MaxViT IoU | Hybrid IoU |','|---|---:|---:|---:|']
style=json.loads((p/'style_statistics.json').read_text());lookup={(r['model'],r['style']):r for r in style}
clean={m:lookup[m,'noise_0.25']['pooled_iou']-lookup[m,'noise_0.25']['pooled_delta'] for m in models}
lines.append('| 原始同一25体 | '+' | '.join(f'{clean[m]:.4f}' for m in models)+' |')
for s in ['noise_0.25','noise_0.50','lowpass_0.6','lowpass_1.0','phase_30','phase_60']:lines.append('| '+s+' | '+' | '.join(f"{lookup[m,s]['pooled_iou']:.4f}" for m in models)+' |')
lines+=['','60度相位变化下，IoU下降：UNet 4.85、MaxViT 5.07、hybrid 2.31个百分点。hybrid下降较小，但最终绝对成绩仍略低于UNet。强噪声下降：7.85、9.78、7.77个百分点。轻微低通影响较小，MaxViT在sigma0.6略有改善；不能据此推断所有频带变化都不重要。','',
'## 5. 几何难度（完整100体源验证集）','',
'| 类别 | UNet IoU | MaxViT IoU | Hybrid IoU |','|---|---:|---:|---:|']
cats=json.loads((p/'category_statistics.json').read_text());lut={(r['model'],r['category']):r for r in cats}
for cat in ['en_echelon','horsetail','negative_flower','positive_flower','listric_assemblage']:lines.append('| '+cat+' | '+' | '.join(f"{lut[m,cat]['iou']:.4f}" for m in models)+' |')
lines+=['','铲状类别：UNet P/R=0.8477/0.8225，MaxViT=0.8528/0.8082，hybrid=0.7872/0.8363。hybrid召回提高，但Precision损失更大，域内IoU反而下降。','',
'按倾角、曲率控制参数、主断层最大滑移、SNR、主频和可见性代理做三等频分组，同时提供类内Spearman相关和分组类别构成。低倾角组包含全部20个铲状样本，有明显类别混杂。即便铲状类内倾角与IoU正相关（rho约0.58/0.69/0.61），每类只有20体，也不构成因果证明。高曲率参数组整体较难，但其类别构成也不同。',
'曲率使用 mean(|beta_dip|+|beta_strike|) 生成控制量，不是物理曲率；滑移使用主断层d_max，不是逐点断距。n_faults/tree_depth 与固定类别模板高度关联，其统计写入 topology_groups.json，不将它们冒称为独立拓扑泛化实验。','',
'## 6. 数据与可复现性','',
'- 每个模型每个体均保存TP/FP/FN、AP、容差计数、输入和标签哈希；三个模型数据输入哈希一致。',
'- provenance.json 保存checkpoint SHA256、原始训练参数、epoch、参数量、GPU/PyTorch和审计脚本hash。',
'- 按发布校验清单检查v2，未发现重复地震文件哈希或跨split完全重复。此检查不是相似性泄漏审计，也未重新计算整个1000体的发布哈希；本轮实际输入逐体另外计算了哈希。',
'- 官方Wu20与本地Wu-style均为合成数据，没有真实工区实验。既有测试集已在开发中查看，不能称为盲测。',
'- 本轮完成3×(100+100+20+100+6×25)=1410次主评估推理，另有冒烟、独立入口复核及轴顺序诊断。','',
'## 7. 对下一步模型工作的建议','',
'优先验证成像风格敏感性与几何漏检/误检的权衡。先建立统一预算的UNet和MaxViT对照，再用同样的相位/噪声增强检验能解决多少问题；若增强本身已解决大部分差距，不把它归功于新结构。',
'新结构的目标应同时约束困难断层召回与误检，避免仅靠扩大预测区域得到外部分数。几何监督可作为候选，但本轮没有证明几何引导注意力或减少下采样一定有效。训练epoch差异仍是未排除因素，不据本轮结果直接认定架构优劣。','',
'入口：index.html。图：domain_comparison.png、style_response.png、category_comparison.png、geometry_strata.png。15张固定样本误差图，正文可折叠查看。原始表：summary.csv。']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
# Add discrete topology groups, explicitly observational/confounded.
group={}
for attr in ['n_faults','tree_depth']:
 group[attr]={}
 for val in sorted({r['geometry'][attr] for r in read('unet','v2_val')['volumes']}):
  group[attr][str(val)]={}
  for m in models:
   rr=[r for r in read(m,'v2_val')['volumes'] if r['geometry'][attr]==val]
   c={k:sum(r['counts'][k] for r in rr) for k in rr[0]['counts']}
   group[attr][str(val)][m]=dict(n=len(rr),iou=c['tp']/(c['tp']+c['fp']+c['fn']),categories=dict(collections.Counter(r['category'] for r in rr)))
(p/'topology_groups.json').write_text(json.dumps(group,indent=2))
# Verify corrected outputs agree and raw axes reproduce historical scores.
old=json.loads(pathlib.Path('logs/eval_cross.json').read_text());axis=json.loads((p/'axis_diagnostic.json').read_text());ind=json.loads((p/'independent_cross_check.json').read_text());checks={}
for m in models:
 prov=read(m,'provenance');run=pathlib.Path(prov['run']).name
 for metric in ['iou','ap','precision','recall']:
  assert abs(axis[m]['metrics'][metric]-old[run]['faultseg3d'][metric])<1e-8
  assert abs(read(m,'faultseg3d20')['metrics'][metric]-ind[run]['faultseg3d'][metric])<1e-8
 checks[m]='PASS: raw-axis reproduces old scores; canonical-axis agrees across independent evaluators'
(p/'reconciliation.json').write_text(json.dumps(checks,indent=2))
# Put the key correction visibly above the tables.
f=p/'index.html';s=f.read_text();insert='<section style="background:#fff3d9;padding:18px"><h2>重要：历史分数已核对修正</h2><p>旧官方Wu20分数来自深度轴未转换的输入口径，现已通过原始轴顺序实验逐项复现。按官方约定转换到深度优先后，UNet / MaxViT / hybrid 的IoU为 <b>0.5481 / 0.4385 / 0.5639</b>。这是评估口径修正，不是模型升级。</p><p>混合模型域内较低、两个外部合成集较高；相位变化下下降较小，但强噪声条件并不全面领先。训练轮数不同，不能作最终架构排名。<a href="REPORT.md">完整中文结论</a> · <a href="axis_diagnostic.json">轴顺序核验</a> · <a href="reconciliation.json">独立评估一致性</a></p></section>'
s=s.replace('<h2>1. 完整数据集评估</h2>',insert+'<h2>1. 完整数据集评估</h2>');f.write_text(s)
print('report and reconciliation complete')

# 鲁棒参数化曲面拟合：实验与论文复现指南

本文档对应 `paper/ieee_superquadric/main.tex`。当前研究的中心不是提出一个新的元启发式算法，而是分析优化器、初始化和支持集设计如何影响圆柱与超二次曲面的鲁棒拟合。正式结论只能来自协议中预先登记的种子、完整结果和独立审计，不能用单次最好结果代替。

## 1. 目录与环境

- 项目：`C:\code\Fiting\fitting`
- 实验输出：`C:\code\Fiting\outputs`
- 超二次曲面数据：`C:\code\superquadic_data\v3_randomized`
- 主环境：`D:\Anaconda\envs\ML\python.exe`（Python 3.11.15，NumPy 2.4.4，SciPy 1.17.1，Matplotlib 3.10.8）
- EMS 环境：`D:\Anaconda\envs\EMS\python.exe`（Python 3.10.20，NumPy 1.26.4，SciPy 1.11.4）
- LaTeX：TeX Live 2026，`C:\texlive\2026\bin\windows\latexmk.exe`

当前环境能够通过 PyTorch 识别 NVIDIA GeForce RTX 5060 和 CUDA，但正式 PSO/CS 搜索状态使用 NumPy，均值测度和独立几何评价使用 CPU sklearn KDTree；当前环境也未安装 FAISS。可选 `torch_cuda` 最近邻后端已通过数值等价与真实多进程冒烟测试，但在 PMF 圆柱的配对 8080-FE 端到端门禁中，CPU KDTree 在 clean/outlier-50、PSO/CS 四个单元均快 1.31--1.61 倍，因此正式预算实验也预注册为 `sklearn`。不能仅根据 CUDA 可见或单核查询速度把运行时间写成 GPU 时间。EMS 因旧依赖单独使用 `EMS` 环境，外部几何评价仍调用 `ML` 环境，避免评价实现不一致。运行 `& $ml tools\audit_compute_backend.py` 与 `& $ml tools\audit_pmf_budget_backend_benchmark.py` 可重新生成后端审计。

以下命令均从项目根目录执行：

```powershell
Set-Location C:\code\Fiting\fitting
$ml = 'D:\Anaconda\envs\ML\python.exe'
$ems = 'D:\Anaconda\envs\EMS\python.exe'
```

## 2. 冻结的实验协议

- 超二次曲面鲁棒性：`paper/ieee_superquadric/protocols/v3_stratified_superquadric_robustness.json`
- PMF 风格圆柱预算敏感性：`paper/ieee_superquadric/protocols/pmf_cylinder_budget_sensitivity.json`

超二次曲面协议使用 9 个案例组成完整的 3×3 形状—长宽比分层块。每个条件使用 5 个配对种子。主评价量是独立解析面积均匀参考面上的双向 Chamfer 距离；成功阈值固定为 0.05，同时只把 0.04 和 0.06 用作阈值敏感性检查。

条件必须分开表述：

- `noise_1pct_diag`：相对包围盒对角线 1% 的高斯噪声；
- `outlier_20`：20% 粗大离群点；
- `missing_80`：随机删除 80% 点，属于随机稀疏；
- `occlusion_cap_80`：仅保留连续投影帽区域的 20% 表面点，属于空间连续遮挡。

随机缺失不能写成遮挡。圆柱 50%/80% 均匀体积离群点只支持相应污染模型下的压力测试结论，不能泛化为任意真实噪声。

## 3. 先验证代码与数据

```powershell
& $ml -m pytest -q
& $ml tools\audit_randomized_superquadric_benchmark.py `
  --data-root C:\code\superquadic_data\v3_randomized `
  --output C:\code\Fiting\outputs\benchmark_audits\v3_randomized_audit.json
```

数据审计会验证文件数、点数、SHA-256 和固定分辨率，并从基础种子重新生成参数、分层标签、各条件种子、参考面、clean、噪声、20% 离群点、80% 随机缺失和连续帽遮挡。所有文件必须在 PLY 的 float32 存储精度下零误差重现，离群点还必须满足记录的最小表面距离。审计不是可选步骤。

20% 离群点的初始化支持还要单独审计。隐藏标签只在无标签选择结束后用于统计精确率和召回率：

```powershell
& $ml tools\audit_v3_outlier_support.py
```

当前九个正式案例的 3,750 点支持均为 3,750 个内点和 0 个离群点。这说明所用均匀体积离群模型具有很强的密度可分性，不能据此声称对聚集型或对抗型离群点同样有效。

## 4. 正式超二次曲面鲁棒性实验

Guided PSO：

```powershell
$sqRoot = 'C:\code\Fiting\outputs\optimizer_comparison\v3_stratified9_robustness_guided_pso_5seeds_20260721'
& $ml tools\run_v3_stratified_superquadric_robustness.py --output-root $sqRoot
```

运行器支持断点续跑：已经有完整 5 个种子的案例会跳过，未完成案例在已有 `results.json` 上继续。不要删除失败运行，也不要只重跑失败种子。

如果实验按完整结果边界暂停，使用下面的单一入口恢复整个研究队列：

```powershell
powershell -ExecutionPolicy Bypass -File C:\code\Fiting\fitting\tools\resume_research_experiment_queue.ps1
```

该入口会同时启动主队列、45/90/135/180 行独立审计监控和最终论文汇总进程，并把三个进程的 PID 与日志路径写入 `C:\code\Fiting\outputs\experiment_queue\resume_manifest_*.json`。如果检测到已有正式队列或实验运行器，它会拒绝重复启动。仅检查命令而不启动时使用 `-DryRun`。

若使用按完整结果边界暂停的守护脚本，暂停完成后先检查 `C:\code\Fiting\outputs\experiment_queue\scheduled_pause_manifest.json`：`status` 应为 `PAUSED_AT_COMPLETE_RESULT_BOUNDARY`，`result_snapshot_valid` 应为 `true`，其中的 `completed_corrupted_rows` 是下一次恢复前的权威行数。

EMS 是超二次曲面专用的精度参考，不与 PSO 声称函数评估预算公平。若需要重新生成连续遮挡条件：

```powershell
& $ems tools\run_v3_ems_occlusion.py `
  --output-root C:\code\Fiting\outputs\ems_baseline\v3_randomized_fixedprior01 `
  --evaluation-python $ml
```

完整结果生成后，先汇总，再从原始拟合参数重新生成模型点并复算全部外部指标：

```powershell
& $ml tools\summarize_v3_superquadric_robustness.py `
  --robustness-root $sqRoot --output-root "$sqRoot\summary"
& $ml tools\audit_v3_superquadric_robustness.py `
  --robustness-root $sqRoot `
  --output "$sqRoot\summary\strict_external_audit.json"
```

正式汇总必须包含 225 个 PSO 结果（5 条件×9 案例×5 种子）和 45 个 EMS 条件—案例结果，且四个外部指标的最大复算误差为零或不超过审计容差。

## 5. PMF 风格圆柱的 PSO–CS 预算敏感性

预算为 50,000、199,920 和 499,920 次函数评估。后两个数是种群规模 80 时同时兼容 PSO 与 CS 更新粒度、且分别最接近 200k 与 500k 的预算。

```powershell
$budgetRoot = 'C:\code\Fiting\outputs\pmf_cylinder_budget_sensitivity\preregistered_20260721'
& $ml tools\run_pmf_cylinder_budget_sensitivity.py --output-root $budgetRoot
foreach ($budget in 50000,199920,499920) {
  & $ml tools\audit_pmf_cylinder_experiment.py "$budgetRoot\fe_$budget"
}
& $ml tools\summarize_pmf_cylinder_budget_sensitivity.py $budgetRoot
```

长实验运行中若只想核对已经落盘的完整结果，不要覆盖正式
`audit.json`。使用独立输出路径；结果未达到 20/20 时状态会明确写为
`INCOMPLETE`，只有结果矩阵完整且所有重算门禁通过时才会写为 `PASS`：

```powershell
& $ml tools\audit_pmf_cylinder_experiment.py `
  "$budgetRoot\fe_499920" --allow-incomplete `
  --output "$budgetRoot\fe_499920\audit_partial_progress.json"
```

长预算平台期的记录深拷贝根因、为什么不能直接把 `>=` 改成 `>`，以及队列结束后的等价性与性能验收门禁，见 [runtime_diagnosis_zh.md](runtime_diagnosis_zh.md)。

每个预算包含 clean/outlier_50、PSO/CS 和 5 个新配对种子，共 20 条记录。若 50% 离群点下原始全点云配置已失败，不再用大量 80% 离群点运行证明同一件事；80% 仅用于支持集方法的极端压力测试或少量失败边界诊断。80% 随机缺失与 80% 连续遮挡仍须完整运行，因为它们是不同的数据退化机制。

## 6. 已完成消融的独立审计

```powershell
& $ml tools\summarize_pmf_cylinder_density_support_ablation.py `
  C:\code\Fiting\outputs\pmf_cylinder_density_support\formal_adaptive_20260721

foreach ($condition in 'clean','outlier_50','outlier_80') {
  foreach ($variant in 'full','adaptive') {
    & $ml tools\audit_pmf_cylinder_experiment.py `
      "C:\code\Fiting\outputs\pmf_cylinder_density_support\formal_adaptive_20260721\$condition\$variant"
  }
}

& $ml tools\audit_area_weight_ablation.py `
  C:\code\Fiting\outputs\area_weight_ablation\formal_v2_pso_clean_48x48_5008fe_5seeds
& $ml tools\audit_guided_initialization_ablation.py `
  --output C:\code\Fiting\outputs\optimizer_comparison\guided_initialization_ablation_summary\audit.json
```

面积加权结果是混合的：不能写成对所有模型一致提升。引导初始化显著改善 box/cylinder，但 ellipsoid 已经稳定，论文中应如实报告这种模型依赖性。

## 7. 统计、图表和论文构建

配对比较按同一案例或同一基础随机种子配对。正式脚本使用双侧精确 Wilcoxon 符号秩枚举，显式处理并列秩并丢弃零差值；样本量和零差值数量随结果一起记录。连续 Chamfer 是主结论，成功率是固定阈值下的辅助描述。

```powershell
& $ml tools\write_paper_result_macros.py
& $ml paper\ieee_superquadric\generate_figures.py

Set-Location paper\ieee_superquadric
& 'C:\texlive\2026\bin\windows\latexmk.exe' -pdf -interaction=nonstopmode -halt-on-error main.tex
Set-Location C:\code\Fiting\fitting
```

逐页人工检查完成后，用实际页码范围记录 QA。命令会核对 PDF 页数、渲染图数量并把记录绑定到当前 PDF 的 SHA-256；PDF 只要重新生成，旧记录就不能通过最终门禁：

```powershell
& $ml tools\record_pdf_visual_qa.py --confirmed-pages 1-9
```

只有当完整汇总存在时，生成器才会写入正式鲁棒性与预算图表。编译后还要把 PDF 每一页渲染成图片逐页检查，确认没有裁切、重叠、未定义引用和作者占位符。

## 8. 最终完成门

```powershell
& $ml tools\audit_research_completion.py `
  --output C:\code\Fiting\outputs\research_completion_audit.json
```

该命令严格检查 PMF 主对比、支持集消融、面积加权、引导初始化、超二次曲面 clean/受污染条件、预算敏感性、数据审计、结果宏、图、最终 PDF 和逐页渲染证据。在实验仍运行时可加 `--allow-incomplete` 仅查看缺项；带该选项的结果不能作为论文完成证明。

投稿前还必须由作者本人填写真实姓名、单位、邮箱和资助信息。任何审计未通过、结果数量不足或仍含作者占位符的版本都不是最终投稿版。

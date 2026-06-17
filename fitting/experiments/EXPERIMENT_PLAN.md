# 实验计划大纲：非经典几何模型的自适应鲁棒拟合

**目标会议：CAGD (B类) / CAD (B类) / 3DV (B类)**
**预计周期：6-8周**
**核心贡献：模型无关的自适应loss调度器 + 跨模型验证**

---

## 论文叙事线

```
现有NURBS拟合方法（NURBS-Diff）用固定权重Chamfer loss，
对不同噪声水平、不同模型类型需要手工调参。

我们提出一个模型无关的自适应loss调度器：
  - Huber残差替代L2，阈值自动从残差分布计算
  - 三阶段覆盖率退火，无需per-dataset调参
  - model→data动态权重，抑制缺失区域的伪曲面

在3类模型 × 4种噪声条件下，自适应调度器一致优于
手工调参的固定权重baseline。
```

---

## 实验矩阵

### 实验1：Huber loss vs L2 loss（噪声鲁棒性）

**问题：** Chamfer L2对离群点过于敏感

**方案：**
- 替换 `gd_fitter.py:216` 的 Chamfer L2 为 smooth L1 (Huber)
  ```python
  # 当前
  data_to_model = data_to_model_min.mean()  # L2 cdist
  # 改为
  huber = torch.where(d < delta, 0.5*d**2, delta*(d - 0.5*delta))
  data_to_model = huber.mean()
  ```
- delta从残差中位数自动计算：`delta = median(errors) × k`，k=1.5

**实验组：**
| 组 | loss类型 | delta | 数据 |
|----|---------|-------|------|
| Baseline | L2 Chamfer | - | 干净数据 |
| L2 | L2 Chamfer | - | 10%/20%/40% 离群点 |
| Huber-fix | Huber, δ=0.05 | 固定 | 10%/20%/40% 离群点 |
| **Huber-adapt** | **Huber, δ=自适应** | **自动** | **10%/20%/40% 离群点** |

**预期：** Huber-adapt在40%离群率下NPRE score显著优于L2

**需要改的代码：** `gd_fitter.py` 约20行

---

### 实验2：覆盖率驱动三阶段退火（收敛质量）

**问题：** 现有CosineAnnealingLR只按step衰减，不知道拟合状态

**方案：**
- 监控 `coverage = len(supporters) / num_data_points`
- 三个阶段自动切换loss权重：

| 阶段 | coverage | coverage_weight | smoothness_weight | data_to_model_weight |
|------|----------|-----------------|-------------------|---------------------|
| 探索 | < 50% | 高 (0.5) | 低 (0.01) | 1.0 |
| 精化 | 50-85% | 中 (0.3) | 中 (0.05) | 1.0 |
| 收敛 | > 85% | 低 (0.1) | 高 (0.1) | 1.0 |

- 阶段切换滞后：连续N步覆盖率达到阈值才切换（防止震荡）

**实验组：**
| 组 | 调度策略 | 数据 |
|----|---------|------|
| Baseline | CosineAnnealingLR + 固定loss权重 | Bunny 150k |
| **Adaptive** | **三阶段退火** | Bunny 150k |

**指标：** 最终NPRE score，收敛步数，最终覆盖率

**预期：** Adaptive更快收敛，最终score≥Baseline

**需要改的代码：** `gd_fitter.py` 约30行

---

### 实验3：model→data自适应权重（缺失数据鲁棒性）

**问题：** DiffCD理论说model→data方向抑制伪曲面。当前权重固定。

**方案：**
- 监控 `error_ratio = data_to_model_error / model_to_data_error`
- ratio > 2: 可能有离群点 → 降低data_to_model_weight
- ratio < 0.5: 可能产生伪曲面 → 提高model_to_data_weight
- 夹在0.5-2之间：保持默认1:1

**数据：** 用你现有的diffcd_verify.py生成——双补丁带间隙，部分球面带孔

**实验组：**
| 组 | model_to_data_weight | 数据 |
|----|---------------------|------|
| Baseline | 1.0 固定 | 间隙/穿孔数据 |
| Blind | 0.0 固定（单向） | 间隙/穿孔数据 |
| **Adaptive** | **动态调整** | **间隙/穿孔数据** |

**指标：** 缺失区域内曲面点数量，NPRE score

**预期：** Adaptive在缺失区域的伪曲面点数接近1.0固定组，显著少于0.0组，且不需要手工设定权重

**需要改的代码：** `gd_fitter.py` 约25行

---

### 实验4：跨模型验证（框架通用性）

**问题：** 上面都在NURBS上验证。需要证明模型无关性。

**方案：** 同样的自适应调度器，换模型不换代码

| 模型 | 数据 | 实验 |
|------|------|------|
| CurveRule (Spiral) | 道路边界点云 | 实验1（离群点） |
| NURBSSurfaceRule | Bunny/ABC | 实验1+2+3 |
| RectangleRule | 建筑立面点云 | 实验1（噪声） |

**CRITICAL：所有实验使用完全相同的自适应调度器代码，不per-model调参**

**预期：** 三个模型上自适应调度器都优于固定权重baseline → 证明"模型无关"

---

### 实验5：消融实验

| 消融组 | 去掉什么 | 预期结果 |
|--------|---------|---------|
| Full | 完整自适应调度器 | 最佳 |
| -Huber | 只用L2，其他自适应保留 | score下降5-10% |
| -Stage | 只用固定loss权重+自适应Huber | 收敛更慢 |
| -M2D | 去掉model→data自适应 | 缺失数据场景变差 |

---

## 论文结构

```
1. Introduction
   - 非经典几何模型拟合的重要性（CAD逆向工程、自动驾驶地图）
   - 现有方法的问题（手工调参、模型特化）
   - 我们的贡献：模型无关的自适应loss调度器

2. Related Work
   - 参数化几何模型拟合 (NURBS-Diff, THB-Diff)
   - 鲁棒点云配准 (GNC, TEASER, Fast Global Registration)
   - 自适应学习率/权重调度

3. Method
   3.1 背景：GD拟合框架
   3.2 自适应Huber残差
   3.3 覆盖率驱动三阶段退火
   3.4 动态model→data权重

4. Experiments
   4.1 实验设置（3类模型）
   4.2 噪声鲁棒性（实验1）
   4.3 收敛质量（实验2）
   4.4 缺失数据（实验3）
   4.5 跨模型验证（实验4）
   4.6 消融实验（实验5）

5. Conclusion
```

---

## 实施顺序（按依赖关系）

```
第1周：实验1 — Huber loss实现 + 离群点实验
第2周：实验2 — 三阶段退火 + Bunny实验
第3周：实验3 — model→data自适应 + 缺失数据实验
第4周：实验4 — CurveRule + RectangleRule验证
第5周：实验5 — 消融实验
第6-7周：写论文 + 刷图
第8周：投稿
```

---

## 最低Bar论文（如果时间不够）

如果只做实验1+4，加上消融实验：
- **贡献：** 自适应Huber残差 + 跨3个模型验证
- **实验量：** 3模型 × 4噪声条件 = 12组
- **可投：** CAGD/CAD short paper 或 CVMJ full paper

这个=最稳，两个月内必能投出去。

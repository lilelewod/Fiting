# 非经典几何模型鲁棒拟合框架 — 研究计划

## 一、定位：我们到底做什么

现有几何拟合方法分两类：
- **经典基元拟合**：RANSAC/Hough → 平面、圆柱、球体等（只能处理预定义的简单形状）
- **深度学习隐式曲面**：DeepSDF/OccNet/SIREN → SDF/MLP隐式表达（黑盒、不可编辑、无几何语义）

我们在中间地带：**用梯度下降拟合可编辑、有语义的非经典参数化几何模型**。

```
经典基元（参数少，形状固定）    我们的方法（参数化，可编辑）    隐式曲面（黑盒，不可编辑）
   平面/圆柱/球体      →     曲线/曲面/字符/NURBS      →      SDF/MLP/OccNet
       ← 太简单                    ← 我们在做这个              ← 不可解释
```

**核心贡献是框架本身，NURBS只是第一个模型。**

---

## 二、现有模型体系

```
ModelRule (抽象接口)
├── CurveRule          — 道路曲线 (Line/Circle/Spiral × Line/Parabola)
├── NURBSSurfaceRule   — NURBS张量积曲面 (控制点+权重+节点向量)
├── RectangleRule      — 矩形曲面片
├── CharacterRule      — 字符/笔划模型 (StrokeRule + 组合)
└── LineSegment        — 2D线段
         ↑
    未来可扩展：Bézier曲面、Coons补片、扫掠体、
              旋转曲面、管道曲面、GSpline、细分曲面...
```

所有模型实现同一个接口：
```python
action → parse() → trait → generate() → Token(points, supporters, ...)
```

**关键洞察：GD优化器不关心模型内部结构——只看到 action vector + loss。**

---

## 三、三条创新路线（从框架视角重新定义）

### 路线A：模型无关的自适应鲁棒loss ⭐ 推荐首发

**现有问题：**
- 7项loss权重全局固定，但不同模型需要不同的loss配置
- 道路曲线（1D嵌入3D）vs NURBS曲面（2D嵌入3D）vs 字符（多笔划组合），噪声模式完全不同
- 每次换模型都要手工调参

**创新点：** 设计**模型无关的**自适应loss权重调度器。

具体方案：
1. **自适应Huber阈值**（替换Chamfer L2）
   - `δ = median(per-point errors) × k`
   - k从大到小退火：初始包容离群点 → 后期精确拟合
   - **不需要per-model调参，δ自动从数据残差分布计算**

2. **覆盖率驱动退火**（三阶段）
   ```
   阶段1 (coverage < 60%): 高coverage_weight, 低smoothness
   阶段2 (coverage 60-90%): 均衡
   阶段3 (coverage > 90%): 低coverage_weight, 高smoothness
   ```
   - 阶段切换由当前coverage自动触发，不依赖step数

3. **model→data自适应权重**（DiffCD启发）
   - 监控 `error_ratio = data_to_model / model_to_data`
   - `ratio > 2` → 可能有离群点 → 降低data_to_model_weight
   - `ratio < 0.5` → 可能产生伪曲面 → 提高model_to_data_weight

**实验设计（跨模型验证）：**
| 实验 | 模型 | 数据 | 噪声条件 | 证明什么 |
|------|------|------|---------|---------|
| 1 | CurveRule | 道路边界点云 | 10-40% 离群点 | 自适应阈值 vs 固定阈值鲁棒性 |
| 2 | NURBSSurfaceRule | ABC dataset | 高斯噪声 σ=0.01-0.05 | 跨噪声水平的泛化 |
| 3 | CharacterRule | 字符点云 | 缺失笔划 | 缺失数据下的覆盖策略 |

**可投稿：** 3DV, CVM, CAGD, CAD

---

### 路线B：多模型多实例联合拟合 ⭐⭐ 核心创新

**现有问题：**
- 当前多实例是贪心串行：fit实例1 → 锁定 → fit实例2...
- 不同模型类型（曲线+曲面+字符）无法同时拟合
- 实例间数据分配是硬性的（base_supporters排除）

**创新点：** 一个统一框架，**同时拟合K个不同类型的几何模型到同一个点云**，数据点到模型的分配是软性的、端到端可学习的。

**这相当于把PARSAC（AAAI'24）从离散CV模型推广到连续参数化几何模型。**

具体方案：
1. **模型无关的软分配层**
   ```python
   # 对每个数据点i，预测它属于每个模型实例k的概率
   w_ik = softmax(MLP(point_i, instance_k_feature))
   # instance_k_feature: 模型类型embedding + 当前曲面中心/跨度
   ```
   - 训练早期：softmax温度高，软分配（允许多个模型竞争同一个数据点）
   - 训练后期：温度降低，趋向硬分配
   - **新增模型类型只需改feature维度，不改架构**

2. **加权多模型loss**
   ```
   loss_total = sum_k [ sum_i w_ik × chamfer(point_i, model_k_points) ]
              + entropy(w) + load_balance(w) + overlap_penalty(model_points)
   ```

3. **跨模型正则化**
   - 不同模型类型的smoothness定义不同（曲线曲率 vs 曲面二阶差分 vs 字符笔划连续性）
   - 用统一的接口：`model.compute_smoothness(trait) → float`
   - 权重由模型类型自动确定（曲线的smoothness天然比曲面小）

**核心实验：**
混合模型拟合 — 同一个点云由 2条曲线 + 1张NURBS曲面 + 1个矩形 联合拟合，无人工分割。
这个实验**直接证明了框架的模型无关性和多实例能力**。

**可投稿：** CVPR/ICCV/ECCV, AAAI

---

### 路线C：非经典模型的几何约束学习 ⭐⭐ 长期探索

**现有问题：**
- 当前每个模型独立拟合，没有利用几何先验（平行、垂直、共面、对称）
- 真实场景中的非经典几何体之间存在几何关系

**创新点：** 将几何约束作为可微loss项加入优化。

具体方案：
1. **约束类型库**
   - 平行约束：`|cross(axis_k, axis_j)| → 0`
   - 垂直约束：`|dot(axis_k, axis_j)| → 0`
   - 共面约束：点到平面的距离
   - 对称约束：镜像变换后的Chamfer距离
   - 连续性约束：G0/G1/G2端点对齐

2. **约束发现**
   - 从不完美的拟合结果中自动发现候选约束
   - 用户确认或拒绝 → 加入优化

3. **约束驱动的多模型拟合**
   - 同时优化多个模型 + 满足几何约束
   - 比独立拟合更准确（共享全局几何上下文）

**可投稿：** SIGGRAPH (poster), EG, PG

---

## 四、推荐推进路径

```
第1-2周  → 路线A：Huber loss + 自适应阈值 + 覆盖率退火
            实验：NURBS拟合ABC dataset, 多种噪声水平
            产出：一篇3DV/CAGD短文的基础

第3-6周  → 路线A扩展：跨模型验证
            实验：CurveRule + CharacterRule 也跑通
            产出：证明"模型无关"的true claim

第6-12周 → 路线B原型：多实例软分配
            实验：2曲线+1曲面联合拟合同一数据
            产出：核心contribution, 冲顶会

第12周+  → 路线C探索 + 论文写作
```

---

## 五、与现有工作的差异化

| 工作 | 可微 | 多模型 | 多实例 | 自适应loss | 几何约束 |
|------|:----:|:------:|:------:|:----------:|:--------:|
| RANSAC/RRG | ✗ | ✓ | ✓ | ✗ | ✗ |
| NURBS-Diff | ✓ | ✗ | ✗ | ✗ | ✗ |
| PARSAC | ✗ | ✓ | ✓ | ✗ | ✗ |
| BPNet | ✓ | ✗ | ✓ | ✗ | ✗ |
| DeepSDF/SIREN | ✓ | ✗ | ✗ | ✗ | ✗ |
| **我们的框架** | **✓** | **✓** | **✓** | **✓** | **计划中** |

---

## 六、目标

| 优先级 | 会议/期刊 | 路线 | 状态 |
|--------|----------|------|------|
| 首选 | CVMJ / CAGD | A | 适合框架性工作，接受方法类论文 |
| 冲刺 | AAAI 2027 | A+B | 7月截稿，需要快速推进 |
| 备选 | 3DV 2027 | A | 10月截稿，时间充裕 |
| 长期 | CVPR 2028 | B+C | 完整系统 + 大量实验 |

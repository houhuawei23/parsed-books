# 一致性 Note

Consistency

随着训练数据增多，甚至趋于无穷时，学习算法学习得到的分类器是否趋于**贝叶斯最优分类器**。

什么是**贝叶斯最优分类器**：在未见数据分布上能取得最好性能的分类器。

目标：

1. 极限性能标准
2. 一致性的定义
3. **替代函数**的应用 and 一致性条件
4. 划分机制的有效性条件

## 6.1 基本概念

贝叶斯分类器

对于任意样本 $x$，贝叶斯分类器永远选择后验概率（$P(y=\pm 1 | x)$）更大的类别，从而使得分类错误率最小的决策规则。

关键在于理解各项的含义，整体的含义。

$$
h^*(\mathbf{x}) =
\begin{cases}
+1, & \eta(\mathbf{x}) \ge \dfrac{1}{2} \\[10pt]
-1, & \eta(\mathbf{x}) < \dfrac{1}{2}
\end{cases}
$$

数学整理为：

$$
h^*(\mathbf{x}) = 2 \cdot \mathbb{I}\!\left(\eta(\mathbf{x}) \ge \frac{1}{2}\right) - 1
$$

得到贝叶斯风险：

$$
R ^ {*} = R \left(h ^ {*}\right) = \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D} _ {\boldsymbol {X}}} \left[ \min  \left\{\eta (\boldsymbol {x}), 1 - \eta (\boldsymbol {x}) \right\} \right]
$$

贝叶斯分类器与一般分类器的关系（分类讨论）：

$$

R (h) - R ^ {*} = \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D} _ {\boldsymbol {X}}} [ | 1 - 2 \eta (\boldsymbol {x}) | \mathbb {I} (h (\boldsymbol {x}) \neq h ^ {*} (\boldsymbol {x})) ].
$$

在数据集上：

插入法

差距：插入法分类器与最优分类器的差距上界（性能上界）

$$
\begin{aligned}{l} R (h) - R ^ {*}
&\leqslant 2 \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D} _ {\mathcal {X}}} [ | \hat {\eta} (\boldsymbol {x}) - \eta (\boldsymbol {x}) | ] \\
&\leqslant 2 \sqrt {\mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D} _ {\mathcal {X}}} \left[ (\hat {\eta} (\boldsymbol {x}) - \eta (\boldsymbol {x})) ^ {2} \right]}. \tag {6.14} \\ \end{aligned}
$$

一致性定义：

当 $m\to \infty$ 时，学习算法 $\mathfrak{L}$ 满足

$$
\mathbb {E} _ {D _ {m} \sim \mathcal {D} ^ {m}} \left[ R \left(\mathfrak {L} _ {D _ {m}}\right)\right]\rightarrow R \left(h ^ {*}\right), \tag {6.21}
$$


一致性反映了在训练数据足够多的情形下，算法 $\mathfrak{L}$ 能否学习得到贝叶斯最优分类器；即，是否与最优一致？

在理论上，一致性刻画了学习算法 $\mathfrak{L}$ 在无限多数据情形下学习的性能极限。

## 替代函数

对目标函数（0-1 损失函数）进行凸放松？用一个具有良好数学性质的凸函数进行替代。

- 替代函数
- 替代泛化/经验风险
- 最优替代泛化风险
- 替代函数一致性：当替代损失趋于最优时，0/1 损失也趋于最优
  - 充分条件 ～ 什么样的替代函数具有一致性
  - 确定性越高的点（$|\eta - 1/2|$ 大），替代损失从 $\phi(0)$ 下降到最优值的**幅度也越大**。换言之，替代函数能够“感知”样本的确定程度，对确定性高的点给予足够的优化压力。$c$ 和 $s$ 是两个刻画这种关系的常数。

## 划分机制

- 划分机制：将样本空间划分为多个互不相容的区域，然后计数，以多数的类别作为区域中样本的标记
- 划分机制的一致性
  - 当训练数据规模 -> inf 时，基于划分机制的输出函数 $h_m(x)$ 满足 $R(h_m) \to R^*$ ，则称该划分机制具有一致性。
  - 划分机制泛化损失 -> 贝叶斯最优风险
  - 划分后区域应足够小
  - 区域内应包含足够多的样本 ～ 少数服从多数
- 划分机制具有一致性的充分条件
  - 划分机制的一致性要求区域既要足够小以捕捉局部信息，又要包含足够多样本以保证估计可靠——这是局部化与统计可靠性之间的精妙平衡。
  - 假设条件概率 $\eta(\mathbf{x})$ 在样本空间 $\mathcal{X}$ 上连续。若划分后的每个区域满足：
    1. 当 $m \to \infty$ 时，$\text{Diam}(\Omega(\mathbf{x})) \to 0$ 依概率成立
    2. 当 $m \to \infty$ 时，$N(\mathbf{x}) \to \infty$ 依概率成立

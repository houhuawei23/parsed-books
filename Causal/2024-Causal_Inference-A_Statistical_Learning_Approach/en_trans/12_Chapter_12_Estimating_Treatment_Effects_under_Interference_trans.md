# 第12章 干扰下的处理效应估计（Chapter 12 Estimating Treatment Effects under Interference）

在上一章中，我们介绍了**暴露映射（exposure mapping）**作为建模跨单元干扰的工具，以及基于排列的方法来检验干扰的存在。接下来的自然问题——也是本章的重点——是：一旦我们接受了干扰的存在，如何估计考虑干扰的相关处理效应？

**暴露效应（Exposure effects）** 为简单起见，我们在此聚焦于一个设定，其中**假设11.1**成立，且暴露映射具有有限基数并共享定义域。具体而言，我们将考虑一个场景，其中有 $i = 1 , \ldots , n$ 个单元，结果变量 $Y _ { i } \in \mathbb { R }$ ，处理变量 $W _ { i } \in \{ 0 , 1 \}$ 。可能存在跨单元干扰；然而，这种干扰可以通过暴露映射 $\displaystyle H _ { i } : \{ 0 , 1 \} ^ { n } \rightarrow \mathcal { H }$ 来捕捉，该映射具有共享定义域 $\mathcal { H }$ 且 $| { \mathcal { H } } | < \infty$ 。因此，我们有一个满足一致性条件的潜在结果：

$$
\{Y _ {i} (h) \} _ {h \in \mathcal {H}}, \quad Y _ {i} = Y _ {i} (H _ {i} (\mathbf {W})). \tag {12.1}
$$

基于这一假设，我们可以通过比较不同暴露水平 $h$ 和 $h ^ { \prime } \in \mathcal { H }$ 下的平均潜在结果，来定义各种**样本平均处理效应（sample-average treatment effects）**：

$$
\bar {\tau} (h, h ^ {\prime}) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(Y _ {i} (h ^ {\prime}) - Y _ {i} (h)\right). \tag {12.2}
$$

我们的目标是估计这些量并为其提供置信区间。

**示例16.** Rogers和Feller [2018]报告了一项随机试验的结果，该试验通过向家长发送出勤信息来改善高缺勤风险学生的学校出勤率。在某些情况下，一个家庭中有多个学生符合研究条件，作者对**溢出效应（spillovers）**感兴趣：发送关于一个学生的出勤信息是否也会影响其兄弟姐妹的行为？为研究这一问题，作者提出了一个包含3个暴露水平的暴露映射：(1) 学生接受了处理；(2) 学生未接受处理但兄弟姐妹接受了处理；(3) 家庭中无人接受处理。然后，可以定义一系列形如(12.2)的自然估计目标，例如**直接效应（direct effect）** (1) vs. (3)，以及**溢出效应（spillover effect）** (2) vs. (3)。

**无偏估计（Unbiased estimation）** 此处考虑的设定，即在一组 $n$ 个未具体说明的研究参与者上执行的随机试验，与**定理1.1**的设定密切相关，区别在于现在**SUTVA（稳定单元处理值假设，Stable Unit Treatment Value Assumption）**不再成立，我们需要依赖更复杂的暴露映射来捕捉干扰。事实证明，**定理1.1**的类似结论仍然成立：我们可以在基本无需额外假设的情况下获得暴露对比(12.2)的无偏估计。

此处构建无偏估计的最简单方法是通过**逆概率加权（inverse-propensity weighting, IPW）**。假设处理是**伯努利随机化（Bernoulli-randomized）**的：

$$
W _ {i} \sim \mathrm{Bernoulli} (e _ {i}), \quad 0 < e _ {i} < 1, \tag {12.3}
$$

对所有 $i = 1 , \ldots , n$ 独立成立，并令 $e _ { i } ( h ) = \mathbb { P } \left[ H _ { i } ( \mathbf { W } ) = h \right]$ ，其中处理根据(12.3)生成。那么，自然的IPW估计量：

$$
\hat {\tau} _ {I P W} (h, h ^ {\prime}) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {1 (\{H _ {i} (\mathbf {W}) = h ^ {\prime} \}) Y _ {i}}{e _ {i} (h ^ {\prime})} - \frac {1 (\{H _ {i} (\mathbf {W}) = h \}) Y _ {i}}{e _ {i} (h)}\right), \tag {12.4}
$$

对 $\bar { \tau } ( h , h ^ { \prime } )$ 是无偏的。我们使用如下记号：

$$
\mathbb {E} _ {W} \left[ \hat {\tau} _ {I P W} (h, h ^ {\prime}) \right] = \mathbb {E} \left[ \hat {\tau} _ {I P W} (h, h ^ {\prime}) \mid \{Y _ {i} (h) \} _ {i = 1, \dots , n; h \in \mathcal {H}} \right], \tag {12.5}
$$

即 $\mathbb { E } _ { W }$ 表示在固定潜在结果的情况下对随机处理分配取期望。

**定理12.1.** 在假设(12.1)和(12.3)下，进一步假设对所有 $i = 1 , \ldots , n$ 有 $e _ { i } ( h ) , e _ { i } ( h ^ { \prime } ) > 0$ 。则

$$
\mathbb {E} _ {W} \left[ \hat {\tau} _ {I P W} (h, h ^ {\prime}) \right] = \bar {\tau} (h, h ^ {\prime}). \tag {12.6}
$$

**证明.** 利用(12.1)和随机化，可得

$$
\mathbb {E} _ {W} \left[ \hat {\tau} _ {I P W} (h, h ^ {\prime}) \right]
$$

$$
= \mathbb {E} _ {W} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {1 \left(\{H _ {i} (\mathbf {W}) = h ^ {\prime} \}\right) Y _ {i} (h ^ {\prime})}{e _ {i} (h ^ {\prime})} - \frac {1 \left(\{H _ {i} (\mathbf {W}) = h \}\right) Y _ {i} (h)}{e _ {i} (h)}\right) \right]
$$

$$
= \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {\mathbb {E} _ {W} \left[ 1 \left(\{H _ {i} (\mathbf {W}) = h ^ {\prime} \}\right) \right] Y _ {i} (h ^ {\prime})}{e _ {i} (h ^ {\prime})} - \frac {\mathbb {E} _ {W} \left[ 1 \left(\{H _ {i} (\mathbf {W}) = h \}\right) Y _ {i} (h) \right]}{e _ {i} (h)}\right)
$$

$$
= \frac {1}{n} \sum_ {i = 1} ^ {n} \left(Y _ {i} (h ^ {\prime}) - Y _ {i} (h)\right).
$$

最后一个等式中我们还使用了(12.3)以及 $e _ { i } ( h ) , e _ { i } ( h ^ { \prime } ) > 0$ 这一事实。

![image_10](images/image_10.png)

**推断与不确定性量化（Inference and uncertainty quantification）** 更具挑战性的方面在于寻求置信区间。上述结果是**定理1.1**在干扰存在情况下的推广，其证明遵循完全相同的思路。在第1章中，当我们试图超越无偏性并建立推断结果时，我们增加了一个额外假设，即潜在结果是从更广泛总体中独立同分布采样的（参见，例如**定理1.2**）。然而，尽管在SUTVA下做出这种独立同分布采样假设很容易，但在干扰存在的情况下，为潜在结果提出一般的采样假设则困难得多。单元之间现在相互交互（例如，它们在社交网络中是朋友），而写出能够捕捉这些跨单元关系的可信生成模型（例如，为友谊网络写出可信的生成模型）需要深厚的领域知识，并且无法在本章所追求的抽象层面上轻易完成。

在本章中，我们将采取另一种路径，试图建立仅依赖于随机处理分配——而不对潜在结果做任何采样假设——的推断结果。在因果推断文献中，这种方法通常被称为**有限总体方法（finite-population approach）**，因为它不依赖于从某个超总体中抽取单元的假设。我们将在**第12.1节**中首先回顾SUTVA下的有限总体方法——并在没有独立同分布采样假设的情况下重新审视第1章的讨论。然后，在**第12.2节**中，我们将把这一讨论扩展到存在干扰的设定。

## 12.1 有限总体方法（Finite-population methods）

我们此处的目标是提供**定理1.2**的替代方案，使得在SUTVA下的随机对照试验中无需依赖超总体采样假设即可进行推断。随机试验的有限总体分析，包括此处给出的结果，可追溯到Neyman [1923]。以下结果呈现了在伯努利设计情况下通常被称为**奈曼方差分析（Neyman-variance analysis）**的内容。在SUTVA下，我们只关心处理-对照对比，因此将使用简写 $\bar { \tau } : = \bar { \tau } ( 0 , 1 )$ 表示**样本平均处理效应（sample-average treatment effect, SATE）**，$\hat { \tau } _ { I P W } : = \hat { \tau } _ { I P W } ( 0 , 1 )$ 表示估计的处理效应，$e _ { i } = e _ { i } ( 1 )$ 表示**倾向得分（propensity score）**。

**定理12.2.** 在**定理12.1**的设定下，进一步假设SUTVA成立，即 $H _ { i } ( \mathbf { w } ) = w _ { i }$ 。则

$$
n \operatorname{Var} _ {W} \left[ \hat {\tau} _ {I P W} \right] = \bar {\sigma} ^ {2} \leq \sigma^ {2},
$$

$$
\bar {\sigma} ^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {Y _ {i} (0) ^ {2}}{1 - e _ {i}} + \frac {Y _ {i} (1) ^ {2}}{e _ {i}} - \left(Y _ {i} (1) - Y _ {i} (0)\right) ^ {2}\right), \tag {12.7}
$$

$$
\sigma^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {Y _ {i} (0) ^ {2}}{1 - e _ {i}} + \frac {Y _ {i} (1) ^ {2}}{e _ {i}}\right).
$$

此外，$\sigma ^ { 2 }$ 有一个无偏估计量：

$$
\mathbb {E} _ {W} \left[ \widehat {V} \right] = \sigma^ {2}, \widehat {V} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {(1 - W _ {i}) Y _ {i} ^ {2}}{(1 - e _ {i}) ^ {2}} + \frac {W _ {i} Y _ {i} ^ {2}}{e _ {i} ^ {2}}\right). (1 2. 8)
$$

**证明**。根据定理 12.1，我们有

$$
\begin{array}{l} n \operatorname{Var} _ {W} \left[ \hat {\tau} _ {I P W} \right] = n \mathbb {E} _ {W} \left[ \left(\hat {\tau} _ {I P W} - \bar {\tau}\right) ^ {2} \right] \\ = n \mathbb {E} _ {W} \left[ \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i}}{e _ {i}} - \frac {1 - W _ {i}}{1 - e _ {i}}\right) Y _ {i} - \frac {1}{n} \sum_ {i = 1} ^ {n} (Y _ {i} (1) - Y _ {i} (0))\right) ^ {2} \right]. \\ \end{array}
$$

根据 **SUTVA**（稳定单元处理值假设，Stable Unit Treatment Value Assumption）并且由于 $W _ { i }$ 彼此独立，我们可以进一步展开此表达式为

$$
n \mathbb {E} _ {W} \left[ \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i}}{e _ {i}} - 1\right) Y _ {i} (1) - \left(\frac {1 - W _ {i}}{1 - e _ {i}} - 1\right) Y _ {i} (0)\right) ^ {2} \right]
$$

$$
= \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} _ {W} \left[ \left(\left(\frac {W _ {i}}{e _ {i}} - 1\right) Y _ {i} (1) - \left(\frac {1 - W _ {i}}{1 - e _ {i}} - 1\right) Y _ {i} (0)\right) ^ {2} \right]
$$

$$
= \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\left(\frac {1}{e _ {i}} - 1\right) Y _ {i} (1) ^ {2} + \left(\frac {1}{1 - e _ {i}} - 1\right) Y _ {i} (0) ^ {2} + 2 Y _ {i} (0) Y _ {i} (1)\right)
$$

$$
= \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {Y _ {i} (1) ^ {2}}{e _ {i}} + \frac {Y _ {i} (0) ^ {2}}{1 - e _ {i}} - (Y _ {i} (1) - Y _ {i} (0)) ^ {2}\right),
$$

其中，上面的第二个等式通过计算二项概率得出，第三个等式通过展开平方项 $( Y _ { i } ( 1 ) - Y _ { i } ( 0 ) ) ^ { 2 }$ 得出。这就建立了 (12.7)。最后，(12.8) 可以通过遵循定理 12.1 中使用的论证来证明。□

主要的观察结果是，在有限总体模型（finite-population model）下，方差 $\bar { \sigma } ^ { 2 }$ 依赖于潜在结果（potential outcomes）的差异，并且通常在没有进一步假设的情况下无法从数据中估计。然而，该方差有一个简单的上界 $\sigma ^ { 2 }$，可以从数据中识别——事实上，这个方差估计对应于在 **IID**（独立同分布，Independent and Identically Distributed）抽样下 $\hat {\tau} _ {I P W}$ 的通常方差估计。因此，在 IID 抽样下对 **ATE**（平均处理效应，Average Treatment Effect）的精确推断为有限总体模型中的 **SATE**（样本平均处理效应，Sample Average Treatment Effect）提供了保守推断。这一事实在存在干扰（interference）的情况下也会出现。

接下来需要建立置信区间的构造。由于我们不再拥有 IID 数据流，我们将无法再调用经典的**中心极限定理（central-limit theorem）**；相反，我们将需要依赖于**有限样本高斯近似（finite-sample Gaussian approximation）**结果。在下面的结果中，我们还将考虑 IPW 的一个**自归一化（self-normalized）**版本：

$$
\hat {\tau} _ {S I P W} = \frac {\sum_ {i = 1} ^ {n} W _ {i} Y _ {i} / e _ {i}}{\sum_ {i = 1} ^ {n} W _ {i} / e _ {i}} - \frac {\sum_ {i = 1} ^ {n} (1 - W _ {i}) Y _ {i} / (1 - e _ {i})}{\sum_ {i = 1} ^ {n} (1 - W _ {i}) / (1 - e _ {i})}, \tag {12.9}
$$

因为这通常会改善大样本性能（参见，例如，练习 1）。

**定理 12.3**。假设我们有一系列样本量 n 不断增长的随机试验，这些试验都满足定理 12.2 的条件，并将 $\bar { \tau } _ { n }$ 写为这些随机试验中每一个的 SATE。进一步假设存在常数 $\eta , M < \infty$，使得对于所有单元，有 $\eta \le e _ { i } \le 1 - \eta$ 和 $\left| Y _ { i } ( 0 ) \right| , \left| Y _ { i } ( 1 ) \right| \leq M$，并且对于如下定义的 $\bar { \sigma } _ { n } ^ { 2 }$，有 $\lim \inf _ { n \to \infty } \bar { \sigma } _ { n } ^ { 2 } > 0$。那么，

$$
\sqrt {n} \left(\frac {\hat {\tau} _ {S I P W} - \bar {\tau} _ {n}}{\bar {\sigma} _ {n}}\right) \Rightarrow \mathcal {N} (0, 1), \quad \bar {\mu} _ {n} (w) = \frac {1}{n} \sum_ {i = 1} ^ {n} Y _ {i} (w), \tag {12.10}
$$

$$
\bar {\sigma} _ {n} ^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {(Y _ {i} (0) - \bar {\mu} _ {n} (0)) ^ {2}}{1 - e _ {i}} + \frac {(Y _ {i} (1) - \bar {\mu} _ {n} (1)) ^ {2}}{e _ {i}} - (Y _ {i} (1) - Y _ {i} (0)) ^ {2}\right),
$$

此外，以下方差估计量

$$
\hat {\mu} _ {n} (0) = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - W _ {i}) Y _ {i}}{1 - e _ {i}}, \quad \hat {\mu} _ {n} (1) = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i} Y _ {i}}{e _ {i}}, \tag {12.11}
$$

$$
\hat {\sigma} _ {n} ^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {(1 - W _ {i}) (Y _ {i} - \hat {\mu} _ {n} (0)) ^ {2}}{(1 - e _ {i}) ^ {2}} + \frac {W _ {i} (Y _ {i} - \hat {\mu} _ {n} (1)) ^ {2}}{e _ {i} ^ {2}}\right),
$$

是渐近保守的，即 $\lim \sup _ { n \to \infty } \bar { \sigma } _ { n } / \hat { \sigma } _ { n } \le _ { p } 1$ ，并且通常的正态置信区间是有效的：

$$
\limsup _ {n \to \infty} \mathbb {P} \left[ | \hat {\tau} _ {S I P W} - \bar {\tau} _ {n} | \leq \hat {\sigma} _ {n} / \sqrt {n}   \Phi^ {- 1} (1 - \alpha / 2) \right] \leq 1 - \alpha , \tag {12.12}
$$

对于任意 $0 < \alpha < 1$ 都成立。

**证明**。由于自归一化和 SUTVA，我们有一个误差分解：

$$
\begin{array}{l} \hat {\tau} _ {S I P W} - \bar {\tau} _ {n} = \Delta (1) \left/ \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i}}{e _ {i}} - \Delta (0) \left. \right/ \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {1 - W _ {i}}{1 - e _ {i}}, \\ \Delta (0) = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - W _ {i}) (Y _ {i} (0) - \bar {\mu} _ {n} (0))}{1 - e _ {i}}, \Delta (1) = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i} (Y _ {i} (1) - \bar {\mu} _ {n} (1))}{e _ {i}}. \\ \end{array}
$$

根据定理 12.1 和 12.2，我们立刻得到

$$
\mathbb {E} _ {W} \left[ \Delta (1) - \Delta (0) \right] = 0, n \operatorname{Var} _ {W} \left[ \Delta (1) - \Delta (0) \right] = \bar {\sigma} _ {n} ^ {2}.
$$

此外，我们的有界性假设意味着构成 $\Delta ( 0 )$ 和 $\Delta ( 1 )$ 的所有项都被 $2 M / \eta$ 所界，因此 **Berry–Esseen** 界意味着

$$
\sup _ {z \in \mathbb {R}} \left| \mathbb {P} \left[ \frac {\sqrt {n} (\Delta (1) - \Delta (0))}{\bar {\sigma} _ {n}} \leq z \right] - \Phi (z) \right| \leq \frac {8 C M ^ {3} / \eta^ {3}}{\bar {\sigma} _ {n} ^ {3} \sqrt {n}}, \tag {12.13}
$$

其中 $\Phi ( \cdot )$ 是标准高斯累积分布函数，C 是 Berry–Esseen 常数；我们还注意到，由于我们假设了 $\lim \inf _ { n \to \infty } \bar { \sigma } _ { n } ^ { 2 } > 0$ ，(12.13) 的右侧项随着 n 的增加而趋于 0。

同时，再次由于我们的重叠性和有界性假设，我们可以使用标准的**集中论证（concentration arguments）**来验证

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \frac {1 - W _ {i}}{1 - e _ {i}} - 1, \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i}}{e _ {i}} - 1 = \mathcal {O} _ {P} \left(\frac {1}{\sqrt {n}}\right),
$$

并且还有

$$
\Delta (0), \Delta (1) = \mathcal {O} _ {P} \left(\frac {1}{\sqrt {n}}\right).
$$

这意味着

$$
\hat {\tau} _ {S I P W} - \bar {\tau} _ {n} = \Delta (1) - \Delta (0) + \mathcal {O} _ {P} \left(\frac {1}{n}\right),
$$

因此 (12.10) 由 (12.13) 得出。最后，我们可以再次使用集中论证来验证

$$
\lim _ {n \to \infty} \hat {\sigma} _ {n} ^ {2} - \sigma_ {n} ^ {2} = _ {p} 0, \quad \sigma_ {n} ^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {(Y _ {i} - \bar {\mu} _ {n} (0)) ^ {2}}{1 - e _ {i}} + \frac {(Y _ {i} - \bar {\mu} _ {n} (1)) ^ {2}}{e _ {i}}\right),
$$

并且根据定理 12.2，我们还得到 $\sigma _ { n } ^ { 2 } \geq \bar { \sigma } _ { n } ^ { 2 }$ 。由于 $\lim \inf _ { n \to \infty } \bar { \sigma } _ { n } ^ { 2 } > 0$ ，所声称的结果随之成立。□

注意，在均匀随机试验（即，$e _ { i } = \pi$ 对所有单元都相同）的情况下，最终得到的置信区间构造 (12.12) 与第 1 章中的 (1.11) 完全相同。66 早些时候，我们（通过一个简单的论证）已经证明，在 IID 抽样假设下，(1.11) 对于 ATE 是渐近精确的。值得注意的是，正如这里所发现的，同样的置信区间对于 SATE 也是渐近保守的，而无需任何抽样假设。

## 12.2 暴露效应的置信区间（Confidence intervals for exposure effects）

现在回到我们的主要关注任务，即对 (12.2) 中定义的暴露效应进行推断。除了假设**有限基数暴露映射（finite-cardinality exposure mapping）**外，我们还将假设如定义 11.1 中的**网络干扰结构（network interference structure）**，即每个单元 $i$ 都有一个已知的影响者单元集合 $\mathcal { N } _ { i }$（或非正式地称为朋友），其中 $i \not \subset \mathcal { N } _ { i } \subset \{ 1 , . . . , n \}$ ，使得：

$$
Y _ {i} (\mathbf {w}) = Y _ {i} \left(\mathbf {w} ^ {\prime}\right) \text { whenever } w _ {i} = w _ {i} ^ {\prime} \text { and } w _ {j} = w _ {j} ^ {\prime} \text { for all } j \in \mathcal {N} _ {i}. \tag {12.14}
$$

结合 (12.1)，条件 (12.14) 可以简化为要求 $H _ { i }$ 仅依赖于 $w _ { i }$ 和 $\mathbf { w } _ { \mathcal { N } _ { i } }$ 。

我们对暴露映射做出的两个假设 (12.1) 和 (12.14) 扮演着不同的角色：(12.1) 主要用于证明**估计目标（estimands）**的合理性（我们将以类似**稳定单元处理值假设（SUTVA）**的方式调用它），而 (12.14) 则用于控制相关性并建立样本均值的收敛性质。特别地，网络干扰模型在潜在结果上诱导出一个自然的**随机化依赖图（randomization dependency graph）** $G \in \{ 0, 1 \}^{n \times n}$，

$$
G _ {i j} = 1 \left(\{\mathcal {N} _ {i} \cup \{i \} \} \cap \{\mathcal {N} _ {j} \cup \{j \} \}\right) \neq \emptyset , \tag {12.15}
$$

即，当且仅当存在一个单元 $k \in \{ 1 , \ldots , n \}$ ，其处理在 (12.14) 下可以同时影响 $Y _ { i }$ 和 $Y _ { j }$ 时，$G _ { i j } = 1$ 。

在**伯努利随机化（Bernoulli randomization）** (12.3) 和网络限制 (12.14) 下，可以立即验证，每当 $G _ { i j } = 0$ 时，

$$
H _ {i} (\mathbf {W}) \perp H _ {j} (\mathbf {W}) \quad \text { and   so } \quad Y _ {i} \perp_ {W} Y _ {j}, \tag {12.16}
$$

其中后一个表述意味着在处理分配的随机性下（无论是条件于潜在结果，还是将潜在结果视为固定的），$Y _ { i }$ 与 $Y _ { j }$ 独立。

基于这些要素，我们现在准备将第 12.1 节的结果推广到存在干扰的情形，并为 $\hat { \tau } _ { I P W } ( h , h ^ { \prime } )$ 的方差提供一个精确表达式以及一个保守但可估计的上界。这里，我们将从写出我们的方差估计量开始；我们的目标方差随后将很容易用该方差估计量的矩来表示。

对于任意 $ { \boldsymbol { h } } _ { \mathbf { \lambda } } \in  { \mathcal { H } }$ ，定义**逆倾向权重（inverse-propensity weights）**为 $\begin{array} { r l } { \Gamma _ { i } ( h ) } & { { } = } \end{array}$ $1 \left( \left\{ H _ { i } ( \mathbf { W } ) = h \right\} \right) / e _ { i } ( h )$ ，并令 $\mathbf { { \cal { F } } } ( h ) \in \mathbb { R } ^ { n }$ 为所有单元的这些权重组成的向量。给定这个记号和我们的暴露映射，

$$
\hat {\tau} _ {I P W} (h, h ^ {\prime}) = \frac {1}{n} \sum_ {i = 1} ^ {n} (\Gamma_ {i} (h ^ {\prime}) Y _ {i} (h ^ {\prime}) - \Gamma_ {i} (h) Y _ {i} (h)), \tag {12.17}
$$

其中只有权重 $\Gamma _ { i }$ 被视为随机的。这种表述，以及 (12.16) 中建立的 $\Gamma _ { i }$ 的网络独立性，提示我们通过以下“**异方差和自相关一致（heteroskedasticity and autocorrelation consistent, HAC）**”构造来估计 IPW 估计量的方差：67

$$
\hat {\sigma} ^ {2} (h, h ^ {\prime}) = \frac {1}{n} \left(\boldsymbol {\Gamma} (h ^ {\prime}) \odot \mathbf {Y} - \boldsymbol {\Gamma} (h) \odot \mathbf {Y}\right) ^ {\top} G \left(\boldsymbol {\Gamma} (h ^ {\prime}) \odot \mathbf {Y} - \boldsymbol {\Gamma} (h) \odot \mathbf {Y}\right), (1 2. 1 8)
$$

其中 $\odot$ 表示**逐元素乘积（elementwise product）**。68 以下结果确立了该方差估计实际上是保守的。

**定理 12.4.** 在定理 12.1 的设定下，进一步假设 (12.14) 成立，并且我们考虑一对暴露 $h, h ^ { \prime } \in \mathcal { H }$ ，使得对所有 $i = 1 , \ldots , n$ 有 $e _ { i } ( h ) , e _ { i } ( h ^ { \prime } ) > 0$ 。记 $\sigma ^ { 2 } ( h , h ^ { \prime } ) : = \mathbb { E } _ { W } \left[ \hat { \sigma } ^ { 2 } ( h , h ^ { \prime } ) \right]$ 为 (12.18) 中给出的方差估计，并记 $\bar { \sigma } ^ { 2 } ( h , h ^ { \prime } ) : = n \mathrm { V a r } _ { W } [ \hat { \tau } _ { I P W } ( h , h ^ { \prime } ) ]$ 为 IPW 估计量的**缩放随机化方差（scaled randomization variance）**。那么，

$$
\bar {\sigma} ^ {2} (h, h ^ {\prime}) = \sigma^ {2} (h, h ^ {\prime}) - n ^ {- 1} (\mathbf {Y} (h ^ {\prime}) - \mathbf {Y} (h)) ^ {\top} G (\mathbf {Y} (h ^ {\prime}) - \mathbf {Y} (h)), \tag {12.19}
$$

并且特别地，$\bar { \sigma } ^ { 2 } ( h , h ^ { \prime } ) \leq \sigma ^ { 2 } ( h , h ^ { \prime } )$ 。

**证明.** 在整个证明过程中，我们将使用简写 $\begin{array} { r l } { \Gamma _ { i } ( h ) } & { { } = } \end{array}$ $1 \left( \left\{ H _ { i } ( \mathbf { W } ) = h \right\} \right) / e _ { i } ( h )$ 来表示逆倾向权重。根据定理 12.1 和 (12.1)，我们有

$$
\bar {\sigma} ^ {2} (h, h ^ {\prime}) := n \operatorname{Var} _ {W} \left[ \hat {\tau} _ {I P W} (h, h ^ {\prime}) \right] = n \mathbb {E} _ {W} \left[ \left(\hat {\tau} _ {I P W} (h, h ^ {\prime}) - \bar {\tau} (h, h ^ {\prime})\right) ^ {2} \right]
$$

$$
= n \mathbb {E} _ {W} \left[ \left(\left(\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\Gamma_ {i} (h ^ {\prime}) - \Gamma_ {i} (h)\right) Y _ {i} - \frac {1}{n} \sum_ {i = 1} ^ {n} \left(Y _ {i} (h ^ {\prime}) - Y _ {i} (h)\right)\right) ^ {2} \right] \right.
$$

$$
= n \mathbb {E} _ {W} \left[ \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\Gamma_ {i} (h ^ {\prime}) - 1\right) Y _ {i} (h ^ {\prime}) - \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\Gamma_ {i} (h) - 1\right) Y _ {i} (h)\right) ^ {2} \right].
$$

我们可以根据暴露-协方差矩阵来简化这个表达式：

$$
U _ {i j} (h, h ^ {\prime}) = \mathbb {E} \left[ (\Gamma_ {i} (h) - 1) (\Gamma_ {j} (h ^ {\prime}) - 1) \right] = \mathbb {E} \left[ \Gamma_ {i} (h) \Gamma_ {j} (h ^ {\prime}) \right] - 1
$$

以及 $U ( h ) = U ( h , h )$ 等，得到：

$$
\begin{array}{l} \bar {\sigma} ^ {2} (h, h ^ {\prime}) = = n ^ {- 1} \mathbf {Y} (h) ^ {\top} U (h) \mathbf {Y} (h) + n ^ {- 1} \mathbf {Y} (h ^ {\prime}) ^ {\top} U (h ^ {\prime}) \mathbf {Y} (h ^ {\prime}) \\ - 2 n ^ {- 1} \mathbf {Y} (h) ^ {\top} U (h, h ^ {\prime}) \mathbf {Y} (h ^ {\prime}). \\ \end{array}
$$

接下来，我们转而研究所提出的方差估计 $\hat { \sigma } ^ { 2 } ( h , h ^ { \prime } )$ 的期望。直接计算表明：

$$
\sigma^ {2} (h, h ^ {\prime}) := \mathbb {E} _ {W} \left[ \hat {\sigma} ^ {2} (h, h ^ {\prime}) \right] = n ^ {- 1} \mathbf {Y} (h) ^ {\top} \mathbb {E} \left[ \boldsymbol {\Gamma} (h) ^ {\top} G \boldsymbol {\Gamma} (h) \right] \mathbf {Y} (h)
$$

$$
n ^ {- 1} \mathbf {Y} (h ^ {\prime}) ^ {\top} \mathbb {E} \left[ \boldsymbol {\Gamma} (h ^ {\prime}) ^ {\top} G \boldsymbol {\Gamma} (h ^ {\prime}) \right] \mathbf {Y} (h ^ {\prime}) + 2 n ^ {- 1} \mathbf {Y} (h) ^ {\top} \mathbb {E} \left[ \boldsymbol {\Gamma} (h) ^ {\top} G \boldsymbol {\Gamma} (h ^ {\prime}) \right] \mathbf {Y} (h ^ {\prime}).
$$

此外，从 (12.16) 我们看到：

$$
U _ {i j} (h) = U _ {i j} (h ^ {\prime}) = U _ {i j} (h, h ^ {\prime}) = 0 \quad \text { whenever } \quad G _ {i j} = 0,
$$

因此，我们可以用上面使用的暴露-协方差矩阵重新表达 $\sigma ^ { 2 } ( h , h ^ { \prime } )$ 如下：

$$
\begin{array}{l} \sigma^ {2} (h, h ^ {\prime}) = = n ^ {- 1} \mathbf {Y} (h) ^ {\top} (U (h) + G) \mathbf {Y} (h) + n ^ {- 1} \mathbf {Y} \left(h ^ {\prime}\right) ^ {\top} (U \left(h ^ {\prime}\right) + G) \mathbf {Y} \left(h ^ {\prime}\right) \\ - 2 n ^ {- 1} \mathbf {Y} (h) ^ {\top} \left(U (h, h ^ {\prime}) + G\right) \mathbf {Y} (h ^ {\prime}). \\ \end{array}
$$

现在我们可以比较 $\sigma ^ { 2 } ( h , h ^ { \prime } )$ 和 $\bar { \sigma } ^ { 2 } ( h , h ^ { \prime } )$ 的表达式：

$$
\sigma^ {2} (h, h ^ {\prime}) - \bar {\sigma} ^ {2} (h, h ^ {\prime}) = = n ^ {- 1} \mathbf {Y} (h) ^ {\top} G \mathbf {Y} (h) + n ^ {- 1} \mathbf {Y} (h ^ {\prime}) ^ {\top} G \mathbf {Y} (h ^ {\prime})
$$

$$
- 2 n ^ {- 1} \mathbf {Y} (h) ^ {\top} G \mathbf {Y} (h ^ {\prime})
$$

$$
= n ^ {- 1} \left(\mathbf {Y} (h ^ {\prime}) - \mathbf {Y} (h)\right) ^ {\top} G \left(\mathbf {Y} (h ^ {\prime}) - \mathbf {Y} (h)\right),
$$

由于 $G$ 是**半正定（positive semi-definite）**的，该量是非负的。

遵循我们在 SUTVA 情况下的方法，接下来考虑**自归一化估计量（selfnormalized estimator）**：

$$
\hat {\tau} _ {S I P W} (h, h ^ {\prime}) = \frac {\sum_ {i = 1} ^ {n} \Gamma_ {i} (h ^ {\prime}) Y _ {i}}{\sum_ {i = 1} ^ {n} \Gamma_ {i} (h ^ {\prime})} - \frac {\sum_ {i = 1} ^ {n} \Gamma_ {i} (h) Y _ {i}}{\sum_ {i = 1} ^ {n} \Gamma_ {i} (h)}, \tag {12.20}
$$

并寻求为其建立**中心极限定理（central limit theorem）**。与之前一样，我们在样本量 $n$ 递增的一系列随机化试验下工作，并记：

$$
\bar {\mu} _ {n} (h) = \frac {1}{n} \sum_ {i = 1} ^ {n} Y _ {i} (h), \quad \bar {\tau} _ {n} (h, h ^ {\prime}) = \bar {\mu} _ {n} (h ^ {\prime}) - \bar {\mu} _ {n} (h). \tag {12.21}
$$

我们还将使用一个修正的方差估计量来考虑自归一化：

$$
\begin{array}{l} \hat {\mu} _ {n} (h) = \frac {1}{n} \sum_ {i = 1} ^ {n} \Gamma_ {i} (h) Y _ {i}, \\ \hat {\sigma} _ {n} ^ {2} (h, h ^ {\prime}) = (\boldsymbol {\Gamma} (h ^ {\prime}) \odot (\mathbf {Y} - \hat {\mu} _ {n} (h ^ {\prime})) - \boldsymbol {\Gamma} (h) \odot (\mathbf {Y} - \hat {\mu} _ {n} (h))) ^ {\top} G _ {n} \tag {12.22} \\ \left(\boldsymbol {\Gamma} \left(h ^ {\prime}\right) \odot \left(\mathbf {Y} - \hat {\mu} _ {n} \left(h ^ {\prime}\right)\right) - \boldsymbol {\Gamma} \left(h ^ {\prime}\right) \odot \left(\mathbf {Y} - \hat {\mu} _ {n} (h)\right)\right), \\ \end{array}
$$

其中 ${ \bf Y } - \hat { \mu } _ { n } ( h )$ 表示从 $\mathbf{Y}$ 的所有项中减去标量 $\hat { \mu } _ { n } ( h )$。

**定理 12.5.** 假设我们有一系列样本量 $n$ 递增的随机化试验，它们都满足定理 $1 \mathcal { Q } . 4$ 的条件。记 $d e g ( G _ { n } )$ 为第 $n$ 个问题中随机化依赖图的最大度数，并假设 $\lim _ { n \to \infty } n ^ { - 1 / 4 } d e g ( G _ { n } ) = 0$ 。进一步假设存在常数 $0 < \eta , M , s _ { 0 } ^ { 2 } < \infty$ ，使得在整个问题序列中，对所有单元有 $e _ { i } ( h ) , e _ { i } ( h ^ { \prime } ) \geq \eta$ 且 $| Y _ { i } ( h ) | , | Y _ { i } ( h ^ { \prime } ) | \leq M$ ，并且使用 (12.23) 中的记号，对所有 $n$ 有 $\bar { \sigma } _ { n } ^ { 2 } ( h , h ^ { \prime } ) \geq s _ { 0 } ^ { 2 }$ 。那么，

$$
\begin{array}{l} \sqrt {n} \left(\frac {\hat {\tau} _ {S I P W} (h , h ^ {\prime}) - \bar {\tau} _ {n} (h , h ^ {\prime})}{\bar {\sigma} _ {n} (h , h ^ {\prime})}\right) \Rightarrow \mathcal {N} (0, 1) \\ \bar {\sigma} _ {n} ^ {2} \left(h, h ^ {\prime}\right) = \sigma_ {n} ^ {2} \left(h, h ^ {\prime}\right) - \left(\mathbf {Y} \left(h ^ {\prime}\right) - \bar {\mu} \left(h ^ {\prime}\right) - \mathbf {Y} (h) + \bar {\mu} (h)\right) ^ {\top} G _ {n} \tag {12.23} \\ \left(\mathbf {Y} \left(h ^ {\prime}\right) - \bar {\mu} \left(h ^ {\prime}\right) - \mathbf {Y} (h) + \bar {\mu} (h)\right), \\ \end{array}
$$

其中 $\sigma _ { n } ^ { 2 } ( h , h ^ { \prime } )$ 表示来自 (12.22) 的 $\hat { \sigma } _ { n } ^ { 2 } ( h , h ^ { \prime } )$ 的一个**理想版本（oracle version）**的随机化期望，该版本中将 $\hat { \mu } _ { n } ( h )$ 替换为 $\bar { \mu } _ { n } ( h )$ 等。此外，我们的方差估计量是**渐近保守（asymptotically conservative）**的，$\lim \mathrm { s u p } _ { n \to \infty } \bar { \sigma } _ { n } / \hat { \sigma } _ { n } \le _ { p } 1$ ，并且通常的正态置信区间是有效的：

$$
\operatorname * {l i m s u p} _ {n \to \infty} \mathbb {P} \Big [ | \hat {\tau} _ {S I P W} (h, h ^ {\prime}) - \bar {\tau} _ {n} (h, h ^ {\prime}) | \tag {12.24}
$$

$$
\left. \leq \hat {\sigma} _ {n} (h, h ^ {\prime}) / \sqrt {n} \Phi^ {- 1} (1 - \alpha / 2) \right] \leq 1 - \alpha ,
$$

对于任意 $0 < \alpha < 1$ 成立。

**证明.** 我们再次首先注意到，由于自归一化和我们假设的暴露映射，

$$
\hat {\tau} _ {S I P W} (h, h ^ {\prime}) = \bar {\tau} _ {n} (h, h ^ {\prime}) + \Delta (h ^ {\prime}) / \frac {1}{n} \sum_ {i = 1} ^ {n} \Gamma_ {i} (h ^ {\prime}) - \Delta (h) / \frac {1}{n} \sum_ {i = 1} ^ {n} \Gamma_ {i} (h)
$$

$$
\Delta (h) = \frac {1}{n} \sum_ {i = 1} ^ {n} \Gamma_ {i} (h) \left(Y _ {i} - \bar {\mu} _ {n} (h)\right).
$$

定理 12.1 和 12.4 立即意味着，对所有 $n$ 有：

$$
\mathbb {E} _ {W} \left[ \Delta (h ^ {\prime}) - \Delta (h) \right] = 0, \quad \mathrm{Var} _ {W} \left[ \Delta (h ^ {\prime}) - \Delta (h) \right] = \frac {\bar {\sigma} _ {n} ^ {2} (h , h ^ {\prime})}{n}.
$$

此外，Baldi 和 Rinott [1989, Corollary 2] 提供了一个关于网络相关随机变量正态近似的**贝里-埃森（Berry–Esseen）**结果，在我们的设定下，这意味着：

$$
\sup _ {z \in \mathbb {R}} \left| \mathbb {P} \left[ \frac {\sqrt {n} (\Delta (h ^ {\prime}) - \Delta (h))}{\bar {\sigma} _ {n} (h , h ^ {\prime})} \leq z \right] - \Phi (z) \right| \leq 3 2 \left(1 + \sqrt {6}\right) \sqrt {\frac {2 M}{\eta s _ {0} ^ {3}}} \frac {\deg (G _ {n})}{n ^ {1 / 4}}.
$$

我们对 $G _ { n }$ 度数的假设使得右侧趋近于零，因此：

$$
\frac {\sqrt {n} (\Delta (h ^ {\prime}) - \Delta (h))}{\bar {\sigma} _ {n} (h , h ^ {\prime})} \Rightarrow \mathcal {N} (0, 1).
$$

证明的其余部分遵循定理 12.3 的框架，因此省略；特别地，我们注意到我们的重叠假设立即意味着 $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Gamma _ { i } ( h ) \to _ { p } 1 } \end{array}$ 。□

**注 12.1.** 当 $G$ 具有块结构时，方差估计量 (12.22) 等价于通常的**聚类稳健推断（cluster-robust inference）**方差估计量，后者通常基于**独立同分布（IID）**抽样假设（即聚类是 IID 抽样的）来推导；另见 Abadie 等人 [2023]。因此，我们恢复了一个类似于 Neyman [1923] 在 SUTVA 下推导的保守性现象：由 IID 抽样（此处为聚类）激发的标准方差估计量，对于在将潜在结果视为确定性的设定中仅由处理随机化产生的有限总体方差是保守的。

**注 12.2.** 定理 12.5 中使用的重叠假设 $e _ { i } ( h ) \geq \eta$ 本质上要求 $\mathcal { N } _ { i }$ 是有限的，即使网络在增长（即每个单元仅受到有限数量的其他单元处理的影响）。然而，即使在这种设定下，$G$ 的度数也可能变得很大：如果存在一些节点非常“受欢迎”，即它们影响许多其他节点（即它们属于许多其他单元 $j$ 的 ${ \mathcal { N } } _ { j }$ ），就可能发生这种情况。在此背景下，我们对 $\deg ( G _ { n } )$ 的假设本质上是对外向影响力强度的上界：我们不允许存在一个节点，其处理影响超过 $n ^ { 1 / 4}$ 个其他单元的结果。

## 12.3 参考文献注释（Bibliographic notes）

本章所使用的**有限总体模型（finite-population model）**——以及通过保守的、可识别的方差界进行推断的方法——可追溯到 Neyman [1923]。在此，我们研究了**伯努利试验（Bernoulli trials）**下的有限总体推断；Li 和 Ding [2017] 给出了多种不同实验设计下的结果。我们注意到，定理 12.2 中使用的方差界并非唯一可用的界；有关其他方案，请参见 Aronow、Green 和 Lee [2014]。此外，这里讨论的有限总体方法也可以扩展到更复杂的随机化设计，例如 Morgan 和 Rubin [2012] 中的重随机化（rerandomization）。

我们基于不同暴露类型下的平均结果来定义因果效应的方法，建立在 Aronow 和 Samii [2017] 的工作之上。Aronow 和 Samii [2017] 还给出了 Neyman 模型下处理效应估计量方差的界；我们在定理 12.4 中使用的界归功于 Leung [2022]。在此基础上，Sävje [2024] 讨论了当暴露映射可能被错误指定时，暴露平均估计量（exposure-averaging estimands）的解释，而 Leung [2022] 则在**近似网络干扰模型（approximate network interference model）**下提供了推断结果，其中干扰效应随着单元在网络中相互远离而衰减（但并未消失）。Viviano [2024] 考虑了在暴露映射假设下存在干扰时的策略学习。Ogburn 等人 [2024] 考虑了在网络干扰下基于观测数据进行推断。Harshaw、Sävje 和 Wang [2022] 提出了一种算法框架，用于在多种干扰模型下为多个因果目标生成类似 IPW 的估计量。

最后，我们还注意到，存在不依赖于良好指定的暴露映射来定义干扰下因果效应的替代方法。其中一种方法涉及定义处理的**平均直接效应（average direct effect）**和**平均间接效应（average indirect effect）**，这有效地衡量了一个单元接受处理如何影响其自身或其他单元，同时对其他单元接受的处理进行边际化处理 [Halloran and Struchiner, 1995, Hu, Li, and Wager, 2022b, Sävje, Aronow, and Hudgens, 2021]。

$$
\tau_ {A D E} = \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} _ {W} \left[ Y _ {i} \left(w _ {i} = 1, W _ {- i}\right) - Y _ {i} \left(w _ {i} = 1, W _ {- i}\right) \right], \tag {12.25}
$$

$$
\tau_ {A I E} = \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {j \neq i} \mathbb {E} _ {W} \left[ Y _ {j} \left(w _ {i} = 1, W _ {- i}\right) - Y _ {j} \left(w _ {i} = 1, W _ {- i}\right) \right],
$$

其中，$Y _ { j } \left( w _ { i } = 1 , W _ { - i } \right)$ 表示通过将第 $i$ 个处理设置为 1，但让其他处理保持在随机化分布下的状态时，我们观察到的第 $j$ 个单元的结果。Hu、Li 和 Wager [2022b] 在多种干扰模型的背景下解释了这些估计量，并将它们与总处理效应的概念联系起来。Sävje、Aronow 和 Hudgens [2021] 在一般干扰模型下给出了平均直接效应的界，而 Li 和 Wager [2022] 则在随机图生成模型下，给出了平均直接效应和平均间接效应的精确大样本渐近结果。Munro、Kuang 和 Wager [2021] 考虑了一个模型中的平均直接效应和平均间接效应的大样本行为，在该模型中，干扰是通过价格平衡供给与需求的市场中的均衡效应产生的；他们还提出了可用于溢出感知目标定位的、用于处理异质性的 CATELike 度量。
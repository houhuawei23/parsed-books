# 第一章 随机对照试验（Randomized Controlled Trials）

如何最好地理解和刻画因果关系，是哲学中一个古老的问题。因此，人们可能会认为，任何关于因果推断的讨论都需要用微妙而深奥的概念来框架化。然而，始于 **Neyman [1923]** 和 **Rubin [1974]** 的一项开创性工作表明——尽管因果关系通常是一个微妙而复杂的概念——存在一类重要的问题，即**随机对照试验（randomized controlled trials, RCTs）**，通过仔细应用随机化、平均化和反事实推理，可以以一种实用且概念上直截了当的方式来处理因果问题。¹

本章简要概述了随机对照试验（RCTs）中的统计估计与推断。当可获得时，来自 RCTs 的证据通常被视为**金标准（gold standard）** 统计证据；因此，研究 RCTs 的方法构成了因果推断统计工具包的基础。此外，在计量经济学或流行病学等领域广泛使用的许多观察性研究设计，其动机都是类比于 RCTs；因此，本章也将作为后续讨论观察性研究中估计与推断的垫脚石。

**平均处理效应（Average treatment effects）** 假设我们进行了一项有 $n$ 名研究参与者 $i = 1 , \ldots , n$ 的 RCT，其中每个单元 $i$ 被分配一个二元处理 $W _ { i } \in \{ 0 , 1 \}$，然后我们测量一个结果 $Y _ { i }$。我们的目标是估计处理对结果的影响。遵循 **Neyman–Rubin 因果模型（Neyman–Rubin causal model）**，我们通过**潜在结果（potential outcomes）** 来定义处理的因果效应：对于每个处理水平 $w \in \{ 0 , 1 \}$，我们定义潜在结果 $Y _ { i } ( 1 )$ 和 $Y _ { i } ( 0 )$，分别对应于第 $i$ 个受试者在接受处理或未接受处理时所经历的结果，使得 $Y _ { i } = Y _ { i } ( W _ { i } )$。

那么，处理对第 $i$ 个单元的**个体因果效应（individual causal effect）** 为²

$$
\Delta_ {i} = Y _ {i} (1) - Y _ {i} (0). \tag {1.1}
$$

**因果推断的基本问题（fundamental problem in causal inference）** 在于，一个给定的个体只能被分配一种处理，因此 $Y _ { i } ( 0 )$ 和 $Y _ { i } ( 1 )$ 中只有一个能被观察到。因此，$\Delta _ { i }$ 永远无法被直接观测到。

尽管 $\Delta _ { i }$ 本身是不可知的，但我们（也许令人惊讶地）可以使用随机实验来了解 $\Delta _ { i }$ 的某些性质。在有限样本中，在不对研究参与者如何生成做任何假设（或者等价地，在给定研究参与者潜在结果的条件下），随机化使我们能够获得**样本平均处理效应（Sample Average Treatment Effect, SATE）** 的无偏估计

$$
\overline {{\Delta}} = \frac {1}{n} \sum_ {i = 1} ^ {n} (Y _ {i} (1) - Y _ {i} (0)). \tag {1.2}
$$

此外，如果我们假设研究参与者是从一个总体 $P$ 中独立抽取的，那么随机实验能够为**（总体）平均处理效应（Average Treatment Effect, ATE）**

$$
\tau = \mathbb {E} _ {P} \left[ Y _ {i} (1) - Y _ {i} (0) \right]. \tag {1.3}
$$

提供无偏且大样本一致的估计。本章将讨论针对这两个量的若干不同估计量的性质。

## 1.1 均值差估计（Difference-in-means estimation）

在随机对照试验中，有很多方法可以估计平均处理效应。其中最简单、最直观的方法之一可能是通过**均值差估计量（difference-in-means estimator）**，

$$
\hat {\tau} _ {D M} := \frac {1}{n _ {1}} \sum_ {W _ {i} = 1} Y _ {i} - \frac {1}{n _ {0}} \sum_ {W _ {i} = 0} Y _ {i}, \quad n _ {w} = | \{i: W _ {i} = w \} |. \tag {1.4}
$$

在我们的设定中，这个均值差估计量基本上是在无假设的情况下无偏的，并且平均处理效应直接通过随机化被识别。假设上面给出的潜在结果模型是有效的；或者，正如文献中常说的那样，**稳定单元处理值假设（Stable Unit Treatment Values Assumption, SUTVA）** 成立：

$$
Y _ {i} = Y _ {i} (W _ {i}), \quad i = 1, \dots , n. \tag {1.5}
$$

进一步假设处理确实是随机化的，即，在给定所有潜在结果 $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \} _ { i = 1 } ^ { n }$ 和处理单元数量 $n _ { 1 }$ 的条件下，所有单元被处理的概率相同：³

$$
\mathbb {P} \left[ W _ {i} = 1 \mid \{Y _ {i} (0), Y _ {i} (1) \} _ {i = 1} ^ {n}, n _ {1} \right] = \frac {n _ {1}}{n}, \quad i = 1, \dots , n. \tag {1.6}
$$

那么，$\hat { \tau } _ { D M }$ 对于 (1.2) 中定义的 SATE 是有限样本无偏的。

**定理 1.1.** 在假设 (1.5) 和 (1.6) 下，

$$
\mathbb {E} \left[ \hat {\tau} _ {D M} \mid \{Y _ {i} (0), Y _ {i} (1) \} _ {i = 1} ^ {n}, n _ {0} > 0, n _ {1} > 0 \right] = \overline {{\Delta}}. \tag {1.7}
$$

**证明.** 只要 $n _ { 1 } > 0$，即我们至少有一个处理单元，

$$
\begin{array}{l} \mathbb {E} \left[ \frac {1}{n _ {1}} \sum_ {W _ {i} = 1} Y _ {i} \mid \{Y _ {i} (0), Y _ {i} (1) \} _ {i = 1} ^ {n}, n _ {1} \right] \\ = \mathbb {E} \left[ \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} W _ {i} Y _ {i} \mid \{Y _ {i} (0), Y _ {i} (1) \} _ {i = 1} ^ {n}, n _ {1} \right] \\ = \mathbb {E} \left[ \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} W _ {i} Y _ {i} (1) \mid \left\{Y _ {i} (0), Y _ {i} (1) \right\} _ {i = 1} ^ {n}, n _ {1} \right] \tag {SUTVA} \\ = \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Y _ {i} (1) \mathbb {E} \left[ W _ {i} \mid \{Y _ {i} (0), Y _ {i} (1) \} _ {i = 1} ^ {n}, n _ {1} \right] \\ = \frac {1}{n} \sum_ {i = 1} ^ {n} Y _ {i} (1) \quad (\text { 随机分配 }). \\ \end{array}
$$

当 $n _ { 0 } > 0$ 时，对照组的平均值也有类似的结果。□

**总体渐近性（Population Asymptotics）** 定理 1.1 的结果在其一般性上很有价值：它在最少的假设下提供了无偏性结果，特别是对潜在结果没有做任何分布假设。在实际应用中，这意味着我们可以在不对 $n$ 名研究参与者如何被招募做任何声明的情况下应用定理 1.1。

然而，这个结果的一个局限性在于，它没有刻画抽样误差 $\hat { \tau } _ { D M } - \overline { { \Delta } }$，因此不能直接为统计推断提供路线图。为了取得进展，我们在此增加一个假设，即研究参与者（即形式上，潜在结果对 $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \}$）是从一个总体 $P$ 中独立抽取的。这种总体抽样假设随后可以通过标准的大样本分析直接得到分布结果和置信区间。在不做此类抽样假设的情况下，也可以获得分布结果，但这依赖于专门的统计技术，我们目前不会深入探讨；我们将在本章末尾的文献注释和第 12 章中重新讨论无总体抽样的推断方法。

**例 1.** 2008 年，俄勒冈州通过抽签方式为其**医疗补助计划（Medicaid program）** 向低收入成年人分配额外名额。正如 **Finkelstein 等人 [2012]** 所报道的，约有 90,000 人参加了抽签，其中（随机选出的）约 35,000 人被允许申请医疗补助。作者考虑了若干结果，如医疗保健使用和支出。遵循定理 1.1 的有限样本分析表明，在抽签参与者中，无论抽签参与者集合是如何创建的，均值差估计量对于被允许申请医疗补助对考虑结果的平均效应是无偏的。下面讨论的渐近工具进一步假设抽签参与者是从相关更大总体（例如，对获得保险覆盖感兴趣的身体健全、低收入、无保险的成年人）中独立抽样得到的。

**中心极限定理（A central limit theorem）** 除了 IID 抽样之外，我们还将更具体地说明处理是如何随机化的，并假设我们处于一个**伯努利试验（Bernoulli trial）** 中，且⁴

$$
W _ {i} \mid \{Y _ {i} (0), Y _ {i} (1) \} \stackrel {\text { iid }} {\sim} \operatorname{Bernoulli} (\pi), \quad 0 <   \pi <   1. \tag {1.8}
$$

然后，可以通过简单的统计论证为均值差估计量建立以下中心极限定理。

**定理 1.2.** 在定理 1.1 的假设下，进一步假设潜在结果 $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \} \stackrel { i i d } { \sim } P$ 来自一个具有有界二阶矩的分布 $P$，并且我们按照 (1.8) 进行伯努利试验。那么，

$$
\sqrt {n} \left(\hat {\tau} _ {D M} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {D M}\right), V _ {D M} = \frac {\mathrm{Var} \left[ Y _ {i} (0) \right]}{1 - \pi} + \frac {\mathrm{Var} \left[ Y _ {i} (1) \right]}{\pi}. \tag {1.9}
$$

此外，**代入法方差估计（plug-in variance estimate）**

$$
\widehat {V} _ {D M} := \frac {n}{n _ {0} ^ {2}} \sum_ {W _ {i} = 0} \left(Y _ {i} - \frac {1}{n _ {0}} \sum_ {W _ {i} = 0} Y _ {i}\right) ^ {2} + \frac {n}{n _ {1} ^ {2}} \sum_ {W _ {i} = 1} \left(Y _ {i} - \frac {1}{n _ {1}} \sum_ {W _ {i} = 1} Y _ {i}\right) ^ {2} \tag {1.10}
$$

是一致的，即 $\widehat { V } _ { D M } \to _ { p } V _ { D M }$。

**证明.** 定义潜在结果残差 $\varepsilon _ { i } ( w ) = Y _ { i } ( w ) - \mathbb { E } _ { P } \left[ Y _ { i } ( w ) \right]$，其中 $w = 0 , 1$，我们可以将估计误差表示为

$$
\begin{array}{l} \hat {\tau} _ {D M} - \tau = \frac {1}{n _ {1}} \sum_ {W _ {i} = 1} \varepsilon_ {i} (1) - \frac {1}{n _ {0}} \sum_ {W _ {i} = 1} \varepsilon_ {i} (0) \\ = \frac {n}{n _ {1}} \frac {1}{n} \sum_ {i = 1} ^ {n} W _ {i} \varepsilon_ {i} (1) - \frac {n}{n _ {0}} \frac {1}{n} \sum_ {i = 1} ^ {n} (1 - W _ {i}) \varepsilon_ {i} (0). \\ \end{array}
$$

通过随机化，可以验证 $\mathbb {E} \left[ W _ { i } \varepsilon _ { i } ( 1 ) \right] = \mathbb {P} \left[ W _ { i } \right] \mathbb {E} \left[ \varepsilon _ { i } ( 1 ) \big| W _ { i } = 1 \right] = \mathbb {P} \left[ W _ { i } \right] \mathbb {E} \left[ \varepsilon _ { i } ( 1 ) \right] = 0$ 且 $\mathbb {E} \left[ ( 1 - W _ { i } ) \varepsilon _ { i } ( 0 ) \right] = 0$，最后

$$
\begin{array}{l} \text {Var} \left[ \binom{W _ {i}   \varepsilon_ {i} (1)}{(1 - W _ {i})   \varepsilon_ {i} (0)} \right] = \mathbb {E} \left[ \binom{W _ {i}   \varepsilon_ {i} (1)}{(1 - W _ {i})   \varepsilon_ {i} (0)} ^ {\otimes 2} \right] \\ = \left( \begin{array}{c c} \pi   \text {Var}   [ \varepsilon_ {i} (1) ] & 0 \\ 0 & (1 - \pi)   \text {Var}   [ \varepsilon_ {i} (0) ] \end{array} \right). \\ \end{array}
$$

因此，根据标准多元中心极限定理

$$
\sqrt {n} \binom{\frac {1}{n} \sum_ {i = 1} ^ {n} W _ {i} \varepsilon_ {i} (1)}{\frac {1}{n} \sum_ {i = 1} ^ {n} (1 - W _ {i}) \varepsilon_ {i} (0)} \Rightarrow \mathcal {N} \left(0, \left( \begin{array}{c c} \pi \operatorname{Var} [ \varepsilon_ {i} (1) ] & 0 \\ 0 & (1 - \pi) \operatorname{Var} [ \varepsilon_ {i} (0) ] \end{array} \right)\right).
$$

结果 (1.9) 由 **Slutsky 引理（Slutsky's lemma）** 得出，因为伯努利试验的处理比例是集中趋势的，即 $n _ { 1 } / n \to _ { p } \pi$。同时，(1.10) 通过**弱大数定律（weak law of large numbers）** 类似地得出。□

上述关于 $\hat { \tau } _ { D M }$ 的中心极限定理立即为 $\tau$ 提供了渐近有效的高斯置信区间。对于任意 $0 < \alpha < 1$，

$$
\lim _ {n \to \infty} \mathbb {P} \left[ \tau \in \left(\hat {\tau} _ {D M} \pm \Phi^ {- 1} (1 - \alpha / 2) \sqrt {\widehat {V} _ {D M} / n}\right) \right] = 1 - \alpha , \tag {1.11}
$$

其中 $\Phi$ 表示标准高斯累积分布函数。

从某种角度来看，人们可能会认为，以上就是在随机试验中估计平均处理效应所需的全部内容。均值差估计量 $\hat { \tau } _ { D M }$ 是一致的，并允许进行有效的渐近推断；此外，该估计量实现起来非常简单，并且很难被“作弊”（即，不道德的分析人员几乎没有空间尝试不同的估计策略，然后报告最接近他们想要答案的那个）。另一方面，我们到目前为止的讨论并未确定 $\hat { \tau } _ { D M }$ 在任何有意义的意义上是使用数据的“最优”方式；事实上，我们将在下面看到，通常可以设计出在保证上严格优于 $\hat { \tau } _ { D M }$ 的估计量。

## 1.2 随机试验中的回归调整（Regression adjustments in randomized trials）

在分析**随机对照试验（randomized controlled trials）**时，我们通常可以获取**预处理协变量（pretreatment covariates）** $X _ { i }$，这些协变量与**处理变量（treatments）** $W _ { i }$ 和**结果变量（outcomes）** $Y _ { i }$ 一同被观测。在这种情况下，实践者通常选择通过基于**线性回归（linear regression）**的方法来估计处理效应，而不是使用简单的均值差异。

有两种标准方法可以通过线性回归估计**平均处理效应（average treatment effects）**。第一种方法是拟合一个简单的线性回归：

$$
Y _ {i} \sim \alpha + W _ {i} \tau + X _ {i} \cdot \beta , \tag {1.12}
$$

然后将得到的系数 $\hat { \tau } _ { S R E G } : = \hat { \tau }$ 作为平均处理效应的估计值报告。第二种方法是加入完整的处理-协变量交互项，并拟合交互线性回归：

$$
Y _ {i} \sim \alpha + W _ {i} \tau + X _ {i} \cdot \beta + W _ {i} X _ {i} \cdot \gamma . \tag {1.13}
$$

然后，可以通过所有人接受处理与无人接受处理时的预测均值之差来估计平均处理效应：

$$
\begin{array}{l} \hat {\tau} _ {I R E G} = \frac {1}{n} \sum_ {i = 1} ^ {n} \hat {\alpha} + \hat {\tau} + X _ {i} \cdot (\hat {\beta} + \hat {\gamma}) - \frac {1}{n} \sum_ {i = 1} ^ {n} \hat {\alpha} + X _ {i} \cdot \hat {\beta}, \tag {1.14} \\ = \hat {\tau} + \overline {{X}} \cdot \hat {\gamma}, \quad \overline {{X}} := \frac {1}{n} \sum_ {i = 1} ^ {n} X _ {i}. \\ \end{array}
$$

简单的线性回归和交互回归都可以合理地应用于随机实验中。在本章的其余部分中，我们将重点关注**交互回归估计量（interacted regression estimator）** $\hat { \tau } _ { I R E G }$ 的性质，因为它便于透明分析，并且在当前的**因果推断（causal inference）**文献中通常被视为最佳实践；进一步讨论请参见文献注释。

**线性假设下的回归调整（Regression adjustments under linearity）** 线性回归估计量（1.13）是一个统计估计量，可以在多种不同的数据模型下进行研究。考虑 $\hat { \tau } _ { I R E G }$ 行为（并将其与 ${ \hat { \tau } } _ { D M }$ 进行比较）的最简单设定是假设回归模型（1.13）是**正确设定的（well specified）**；这也是我们在此处首先考虑的设定。

暂时假设我们的样本是通过**伯努利随机试验（Bernoulli randomized trial）**（1.8）独立生成的，其中结果 $Y _ { i } = Y _ { i } ( W _ { i } )$ 满足：

$$
\begin{array}{l} Y _ {i} (w) = \alpha_ {(w)} + X _ {i} \cdot \beta_ {(w)} + \varepsilon_ {i} (w), \\ \mathbb {T} [ (x) | x ] = 0, \quad \forall x <   [ (x) | x ], \quad 2 \end{array} \tag {1.15}
$$

$$
\mathbb {E} \left[ \varepsilon_ {i} (w) \mid X _ {i} \right] = 0, \mathrm{Var} \left[ \varepsilon_ {i} (w) \mid X _ {i} \right] = \sigma^ {2}.
$$

在伯努利随机化下，可以验证可观测变量 $( X _ { i } , Y _ { i } , W _ { i } )$ 是从满足以下条件的分布中独立抽取的：

$$
Y _ {i} = \alpha_ {(0)} + W _ {i} (\alpha_ {(1)} - \alpha_ {(0)}) + X _ {i} \cdot \beta_ {(0)} + W _ {i} X _ {i} \cdot (\beta_ {(1)} - \beta_ {(0)}) + \varepsilon_ {i}, \tag {1.16}
$$

其中 $\mathbb{E} \left[ \varepsilon _ { i } \big | X _ { i } , W _ { i } \right] = 0$ 且 $\operatorname{Var} \left[ \varepsilon _ { i } \big | X _ { i } , W _ { i } \right] = \sigma ^ { 2 }$，即回归模型（1.13）实际上是正确设定的。为简化起见，我们进一步假设我们处于一个**平衡随机试验（balanced randomized trial）**中，其中 $\pi = 5 0 \%$，并且（不失一般性地）假设 $\mathbb{E} [ X ] = 0$。作为热身，我们首先在此模型下研究 ${ \hat { \tau } } _ { D M }$ 的行为作为基准；然后我们才能将其与 $\hat { \tau } _ { I R E G }$ 进行比较。根据定理 1.2 中的一般性结果，剩下的工作就是明确 $V _ { D M }$ 在此处的表达式；并且，记 $\operatorname{Var} [ X ] = A$，我们得到（为简化起见，我们使用 $\pi = 0 . 5$）：

$$
\begin{array}{l} V _ {D M} = \frac {\operatorname{Var} [ Y _ {i} (0) ]}{0 . 5} + \frac {\operatorname{Var} [ Y _ {i} (1) ]}{0 . 5} \\ = 2 \left(\operatorname{Var} \left[ X _ {i} \beta_ {(0)} \right] + \sigma^ {2}\right) + 2 \left(\operatorname{Var} \left[ X _ {i} \beta_ {(1)} \right] + \sigma^ {2}\right) \tag {1.17} \\ = 4 \sigma^ {2} + 2 \left\| \beta_ {(0)} \right\| _ {A} ^ {2} + 2 \left\| \beta_ {(1)} \right\| _ {A} ^ {2} \\ = 4 \sigma^ {2} + \left\| \beta_ {(0)} + \beta_ {(1)} \right\| _ {A} ^ {2} + \left\| \beta_ {(0)} - \beta_ {(1)} \right\| _ {A} ^ {2}, \\ \end{array}
$$

其中为方便起见，我们使用了记号 $\| v \| _ { A } ^ { 2 } = v ^ { \prime } A v$。

鉴于此处线性回归模型是正确设定的，我们可以预期 $\hat { \tau } _ { I R E G }$ 的性能优于 ${ \hat { \tau } } _ { D M }$；问题在于能改进多少。为了研究回归估计量，注意到交互回归（1.13）在算法上等价于对处理组和对照组分别进行回归，然后取其在整个研究样本上的预测值之差：

$$
Y _ {i} \sim \alpha_ {(0)} + X _ {i} \cdot \beta_ {(0)} \text {对于所有满足 W_{i} = 0 的 i},
$$

$$
Y _ {i} \sim \alpha_ {(1)} + X _ {i} \cdot \beta_ {(1)} \text {对于所有满足 W_{i} = 1 的 i},
$$

$$
\hat {\tau} _ {I R E G} = \hat {\alpha} _ {(1)} - \hat {\alpha} _ {(0)} + \overline {{X}} \left(\hat {\beta} _ {(1)} - \hat {\beta} _ {(0)}\right).
$$

关于线性回归的标准结果随后表明，在模型（1.15）下（同样，此处我们假设 $\mathbb{E} \left[ X \right] = 0$）：

$$
\sqrt {n _ {w}} \left(\binom {\hat {\alpha} _ {(w)}} {\hat {\beta} _ {(w)}} - \binom {\alpha_ {(w)}} {\beta_ {(w)}}\right) \Rightarrow \mathcal {N} \left(0, \sigma^ {2} \left( \begin{array}{c c} 1 & 0 \\ 0 & A ^ {- 1} \end{array} \right)\right), \tag {1.18}
$$

并且 $\hat { \alpha } _ { ( 0 ) } , \hat { \alpha } _ { ( 1 ) } , \hat { \beta } _ { ( 0 ) } , \hat { \beta } _ { ( 1 ) }$ 和 $\overline { { X } }$ 都是**渐近独立的（asymptotically independent）**。于是，我们可以写出：

$$
\hat {\tau} _ {I R E G} - \tau = \underbrace {\hat {\alpha} _ {(1)} - \alpha_ {(1)}} _ {\approx \mathcal {N} (0, \sigma^ {2} / n _ {1})} - \underbrace {\hat {\alpha} _ {(0)} - \alpha_ {(0)}} _ {\approx \mathcal {N} (0, \sigma^ {2} / n _ {0})} + \underbrace {\overline {{X}} \left(\beta_ {(1)} - \beta_ {(0)}\right)} _ {\approx \mathcal {N} \left(0, \left\| \beta_ {(1)} - \beta_ {(0)} \right\| _ {A} ^ {2} / n\right)}
$$

$$
+ \underbrace {\overline {{X}} \left(\hat {\beta} _ {(1)} - \hat {\beta} _ {(0)} - \beta_ {(1)} + \beta_ {(0)}\right)} _ {\mathcal {O} _ {P} (1 / n)},
$$

这导出了**中心极限定理（central limit theorem）**：

$$
\sqrt {n} \left(\hat {\tau} _ {I R E G} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {I R E G}\right), \quad V _ {I R E G} = 4 \sigma^ {2} + \left\| \beta_ {(0)} - \beta_ {(1)} \right\| _ {A} ^ {2}. \tag {1.19}
$$

最终我们看到，在线性模型（1.15）下，交互回归估计量也满足中心极限定理，并且：

$$
V _ {I R E G} = V _ {D M} - \left\| \beta_ {(0)} + \beta_ {(1)} \right\| _ {A} ^ {2} \leq V _ {D M}, \tag {1.20}
$$

即，回归估计量的**渐近方差（asymptotic variance）**通常优于（且绝不劣于）均值差异估计量。

**非线性假设下的回归调整（Regression adjustments without linearity）** 我们在上面证明，如果假设数据是按照线性模型生成的，那么正如预期，利用线性假设的估计量能够比不利用线性假设的估计量更精确地估计平均处理效应。悲观者可能会认为这些精度提升是有代价的，即当线性假设不成立时，线性回归估计量会面临一种权衡，其表现会劣于均值差异估计量。然而，令人惊讶的是，这种权衡并不存在。在随机试验中，$\hat { \tau } _ { I R E G }$ 始终是 $\tau$ 的**一致估计量（consistent）**，并且满足（1.20）类型的**渐近非劣效性（asymptotic non-inferiority）**结果，即使 $\hat { \tau } _ { I R E G }$ 所依据的线性回归可能是**错误设定的（misspecified）**。

下面，我们首先在样本从总体中独立抽取的假设下（但不假设线性性）建立 $\hat { \tau } _ { I R E G }$ 的一般性中心极限定理。全文将使用以下记号：

$$
\mu_ {(w)} (x) = \mathbb {E} \left[ Y _ {i} (w) \mid X _ {i} = x \right], \quad \sigma_ {(w)} ^ {2} (x) = \operatorname{Var} \left[ Y _ {i} (w) \mid X _ {i} = x \right], \tag {1.21}
$$

并假设这些量是定义良好且有限的。以下结果的证明依赖于线性回归的**胡伯-怀特（Huber–White）分析**，该分析表明——无论线性假设是否成立——线性回归一致地估计了最佳线性投影系数

$$
\left(\alpha_ {(w)} ^ {*}, \beta_ {(w)} ^ {*}\right) = \operatorname{argmin} _ {\alpha , \beta} \left\{\mathbb {E} \left[ (Y _ {i} (w) - \alpha - X _ {i} \cdot \beta) ^ {2} \right] \right\}, \tag {1.22}
$$

该系数刻画了在均方误差（mean-squared error）下，关于 $X _ { i }$ 的最佳线性预测器。8 下面的论证也可以推广，以验证标准的非参数统计推断工具——如自助法（bootstrap）或刀切法（jackknife）——可用于构建以 $\hat { \tau } _ { I R E G }$ 为中心的、渐近有效的 $\tau$ 的正态置信区间。

**定理 1.3.** 在定理 1.2 的条件下，进一步假设 $\mathbb { E } \left[ X ^ { \prime } X \right]$ 是可逆的。那么，

$$
\sqrt {n} \left(\hat {\tau} _ {I R E G} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {I R E G}\right),
$$

$$
V _ {I R E G} = \operatorname{Var} \left[ X _ {i} \cdot \left(\beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*}\right) \right] + \frac {1}{\pi} \mathbb {E} \left[ \left(Y _ {i} (1) - \alpha_ {(1)} ^ {*} - X _ {i} \cdot \beta_ {(1)} ^ {*}\right) ^ {2} \right] \tag {1.23}
$$

$$
+ \frac {1}{1 - \pi} \mathbb {E} \left[ \left(Y _ {i} (0) - \alpha_ {(0)} ^ {*} - X _ {i} \cdot \beta_ {(0)} ^ {*}\right) ^ {2} \right].
$$

**证明.** 我们再次不失一般性地假设 $\mathbb{E}[X_i] = 0$。根据线性回归的胡伯-怀特分析，我们得到9

$$
\sqrt {n _ {w}} \left(\binom {\hat {\alpha} _ {(w)}} {\hat {\beta} _ {(w)}} - \binom {\alpha_ {(w)} ^ {*}} {\beta_ {(w)} ^ {*}}\right) \Rightarrow \mathcal {N} \left(0, \left( \begin{array}{c c} M S E _ {(w)} ^ {*} & 0 \\ 0 & \dots \end{array} \right)\right), \text {其中} \tag {1.24}
$$

$$
M S E _ {(w)} ^ {*} = \mathbb {E} \left[ \left(Y _ {i} (w) - X _ {i} \beta_ {(w)} ^ {*} - \hat {\alpha} _ {(w)} ^ {*}\right) ^ {2} \right]
$$

度量了最佳线性预测器的均方误差。我们没有写出渐近方差矩阵的右下角部分，因为它很复杂且对一阶行为没有贡献；然而，我们确实注意到，每当 $\mathbb{E}[X'X]$ 可逆时，$\cdots$ 项是有限的。

现在，我们需要展开 (1.14) 中给出的回归估计量，

$$
\hat {\tau} _ {I R E G} - \tau = \hat {\alpha} _ {(1)} - \hat {\alpha} _ {(0)} - \tau + \overline {{X}} \cdot \left(\hat {\beta} _ {(1)} - \hat {\beta} _ {(0)}\right).
$$

我们首先关注前三个被加项的贡献。可以立即验证，最优线性预测的平均偏差必须为0，即，给定 $\beta_{(w)}^*$，截距参数必须为 $\alpha_{(w)}^* = \mathbb{E}[Y_i(w) - X_i \cdot \beta_{(1)}^*]$。因此，在我们假设 $\mathbb{E}[X_i] = 0$ 的条件下，必须有 $\alpha_{(w)}^* = \mathbb{E}[Y_i(0)]$，所以

$$
\hat {\alpha} _ {(1)} - \hat {\alpha} _ {(0)} - \tau = \hat {\alpha} _ {(1)} - \alpha_ {(1)} ^ {*} - (\hat {\alpha} _ {(0)} - \alpha_ {(0)} ^ {*}).
$$

中心极限定理 (1.24) 因此蕴含

$$
\sqrt {n} \left(\hat {\alpha} _ {(1)} - \hat {\alpha} _ {(0)} - \tau\right) \Rightarrow \mathcal {N} \left(0, \frac {M S E _ {(1)} ^ {*}}{\pi} + \frac {M S E _ {(0)} ^ {*}}{1 - \pi}\right). \tag {1.25}
$$

现在，转向最后一个被加项，我们注意到

$$
\overline {{X}} \cdot (\hat {\beta} _ {(1)} - \hat {\beta} _ {(0)}) = \overline {{X}} \cdot (\beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*}) + \overline {{X}} \cdot (\hat {\beta} _ {(1)} - \beta_ {(1)} ^ {*} - \hat {\beta} _ {(0)} + \beta_ {(0)} ^ {*}).
$$

同样由于 $\mathbb{E}[X_i] = 0$，协变量的均值 $\overline{X}$ 接近于零，其渐近正态波动幅度为 $1/\sqrt{n}$，因此

$$
\sqrt {n} \overline {{X}} \cdot \left(\beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*}\right) \Rightarrow \mathcal {N} \left(0, \operatorname{Var} \left[ X _ {i} \cdot \left(\beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*}\right) \right]\right). \tag {1.26}
$$

此外，可以验证 (1.25) 和 (1.26) 中的项是渐近不相关的，因此是渐近独立的。10

最后，因为 $X$ 以及（根据 (1.24)）$\hat{\beta}_{(0)} - \beta_{(0)}^*$ 的波动幅度均为 $1/\sqrt{n}$ 量级，它们的乘积的波动幅度只能是 $1/n$ 量级；我们将其简洁地写为

$$
\overline {{X}} \cdot \left(\hat {\beta} _ {(1)} - \beta_ {(1)} ^ {*} - \hat {\beta} _ {(0)} + \beta_ {(0)} ^ {*}\right) = \mathcal {O} _ {P} (1 / n).
$$

因此，根据**斯卢茨基引理（Slutsky's lemma）**，由于主导项 (1.25) 和 (1.26) 是 $1/\sqrt{n}$ 阶的，这个乘积项在渐近意义上可以忽略。将所有部分整合起来，即得到 (1.23)。□

有了定理 1.3，我们准备重新审视 $\hat{\tau}_{IREG}$ 与 $\hat{\tau}_{DM}$ 的比较。使用回归调整是否有助于提高精度，即使没有线性假设？在这里，我们证明对于平衡的**随机对照试验（Randomized Controlled Trials, RCTs）**，即 $\pi = 0.5$，并且假设不可预测的噪声水平是恒定的，即对所有 $x$ 有 $\sigma_{(1)}^2(x) = \sigma_{(0)}^2(x) = \sigma^2$，答案是肯定的。11 在这些假设下，并如前所述记 $\text{Var}[X_i] = A$，我们可以将 (1.23) 中的渐近方差展开如下：12

$$
\begin{array}{l} V _ {I R E G} = 2 M S E _ {(0)} ^ {*} + 2 M S E _ {(1)} ^ {*} + \left\| \beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*} \right\| _ {A} ^ {2} \\ = 4 \sigma^ {2} + 2 \operatorname{Var} \left[ \mu_ {(0)} (X) - X \beta_ {(0)} ^ {*} \right] \\ + 2 \operatorname{Var} \left[ \mu_ {(1)} (X) - X \beta_ {(1)} ^ {*} \right] + \left\| \beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*} \right\| _ {A} ^ {2}. \\ \end{array}
$$

接下来，因为 $X\beta_{(w)}^*$ 是 $\mu_{(0)}(X)$ 在 $X$ 张成空间上的投影，这可以进一步简化

$$
\begin{array}{l} \dots = 4 \sigma^ {2} + 2 \left(\operatorname{Var} \left[ \mu_ {(0)} (X) \right] - \operatorname{Var} \left[ X \beta_ {(0)} ^ {*} \right]\right) \\ + 2 \left(\operatorname{Var} \left[ \mu_ {(1)} (X) \right] - \operatorname{Var} \left[ X \beta_ {(1)} ^ {*} \right]\right) + \left\| \beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*} \right\| _ {A} ^ {2} \\ = 4 \sigma^ {2} + 2 (\operatorname{Var} [ \mu_ {(0)} (X) ] + \operatorname{Var} [ \mu_ {(1)} (X) ]) \\ + \left\| \beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*} \right\| _ {A} ^ {2} - 2 \left\| \beta_ {(0)} ^ {*} \right\| _ {A} ^ {2} - 2 \left\| \beta_ {(1)} ^ {*} \right\| _ {A} ^ {2} \\ = 4 \sigma^ {2} + 2 \left(\operatorname{Var} \left[ \mu_ {(0)} (X) \right] + \operatorname{Var} \left[ \mu_ {(1)} (X) \right]\right) - \left\| \beta_ {(0)} ^ {*} + \beta_ {(1)} ^ {*} \right\| _ {A} ^ {2} \\ = V _ {D M} - \left\| \beta_ {(0)} ^ {*} + \beta_ {(1)} ^ {*} \right\| _ {A} ^ {2}. \\ \end{array}
$$

换句话说，无论真实效应函数 $\mu_w(x)$ 是否是线性的，交互线性回归总是能够降低或匹配均值差异估计量的渐近方差。此外，方差减少的量与线性回归实际拟合训练数据的程度成比例。回归调整的最坏情况是当 $\beta_{(0)}^* = \beta_{(1)}^* = 0$，即**普通最小二乘法（Ordinary Least Squares, OLS）**渐近地不做任何处理；在这种情况下，$\hat{\tau}_{IREG}$ 最终与 $\hat{\tau}_{DM}$ 渐近等价。

**回归调整在随机对照试验中的作用** 个体处理效应 $\Delta_i = Y_i(1) - Y_i(0)$ 是因果推断中的一个核心关注对象。这些效应 $\Delta_i$ 本身是不可知的；然而，大规模的随机对照试验使我们能够一致地估计平均处理效应（Average Treatment Effect, ATE） $\tau = \mathbb{E}[\Delta_i]$。在本章中，我们介绍并比较了实现这一目标的两种方法：均值差异估计量和交互回归调整。或许令人惊讶的是，我们发现，当存在处理前协变量时，回归调整在渐近意义上至少与均值差异估计量一样精确（并且通常更精确）——无论回归调整所依据的线性模型是否被正确设定，这个结果都成立。

我们关于回归调整分析的一个关键点是，我们在进行任何参数化（例如，线性）建模假设之前（且不依赖这些假设）定义了我们的目标估计量，即平均处理效应 $\tau = \mathbb{E}[\Delta_i]$。平均处理效应是根据非参数的反事实推理来定义的。线性回归随后被用作估计 $\tau$ 的算法工具，但线性建模在我们最初统计问题的框架设定中并未发挥作用。

最后，请注意，我们的回归调整估计量实际上可以被视为预测值的平均差异，

$$
\hat {\tau} _ {I R E G} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\underbrace {\left(\hat {\alpha} _ {(1)} + X _ {i} \hat {\beta} _ {(1)}\right)} _ {\hat {\mu} _ {(1)} (X _ {i})} - \underbrace {\left(\hat {\alpha} _ {(0)} + X _ {i} \hat {\beta} _ {(0)}\right)} _ {\hat {\mu} _ {(0)} (X _ {i})}\right), \tag {1.27}
$$

其中 $\hat{\mu}_{(w)}(x)$ 表示在治疗状态 $w$ 下，在点 $x$ 处的线性回归预测值。我们能否使用其他方法（例如，深度网络、随机森林）而非线性回归来估计 $\hat{\mu}_{(w)}(x)$？这将如何影响渐近方差？第16章中的练习2对此进行了更深入的探讨。

## 1.3 文献注释（Bibliographic notes）

因果推断的**潜在结果模型（potential outcomes model）**最早由 Neyman [1923] 和 Rubin [1974] 倡导；参见 Imbens and Rubin [2015] 以获得现代教科书的处理。这里使用的建模框架中一个简单但微妙的方面是我们对**SUTVA 1.5**的使用，它通过符号排除了许多看似合理的困难 [Imbens and Rubin, 2015, 第1.6章]。SUTVA 排除了任何形式的跨单元干扰（即，对于 $i \neq j$，$W_i$ 不能影响 $Y_j$）。此外，SUTVA 隐含地要求只有1种“版本”的处理；如果，例如，我们运行一个多站点随机试验，其中不同站点的处理方式略有不同，这个假设就可能出现问题。因此，无论在何种应用中引用 SUTVA，都应仔细评估其可信度。

文献中受到相当关注的一个区分性问题，是研究者是否愿意对潜在结果做出任何随机性假设。不对潜在结果做出随机性假设的设定被称为**奈曼模型（Neyman model）**，用于随机化推断或有限总体模型（finite-population model）；而做出随机性假设的设定则被称为超总体模型（superpopulation model）或独立同分布抽样模型（IID-sampling model）。在此，我们在奈曼模型下陈述了定理1.1，但在其他情况下则采用超总体抽样模型。我们将在第12章讨论跨单元干扰下的因果推断时，更仔细地审视奈曼模型——并重新审视本章的一些结果。

在奈曼模型下证明合理的统计推断有时被认为是分析随机试验的最高严谨标准，因为所有推断都仅由随机化证明是合理的：分析者不需要考虑研究参与者是如何入组的（以及他们是否是从更大总体中随机抽取的），就能严格应用在该模型下证明的结果。在奈曼模型下工作的代价是，即使是建立相当简单的估计量的抽样分布也需要更复杂的统计分析；参见 Li and Ding [2017] 了解该领域的最新结果。相比之下，在超总体模型下研究随机试验通常可以通过应用标准的统计和计量经济学工具实现更简单的分析；并为观察性研究环境中更复杂的半参数估计量铺平了道路。关于SATE (1.2) 和 ATE (1.3) 估计量的进一步讨论和比较，见 [Imbens, 2004]。

Lin [2013] 深入讨论了线性回归调整在提高平均处理效应估计量精度方面的作用，以及为什么像 (1.13) 那样使用完全交互作用通常被认为是相对于简单回归 (1.12) 的最佳实践。当协变量 $X_i$ 通过一个离散因子的独热编码（one-hot-encoding）生成时（即，$X_i \in \{0,1\}^K$，每个单元只有一个非零条目），交互回归调整估计量等价于（事后）分层（stratification），这通常也被认为是分析随机实验数据的最佳实践 [Miratrix, Sekhon, and Yu, 2013]。

Lin [2013] 的另一个特点是他在奈曼模型下进行随机化推断，并表明定理1.3中的许多见解实际上在该设定下仍然成立。Wager et al. [2016] 讨论了在超总体渐近性下随机试验中的非参数或高维回归调整，扩展了此处涵盖的结果。在奈曼模型下研究高维回归调整是一个持续的努力，最近的贡献来自 Bloniarz et al. [2016] 和 Lei and Ding [2021]。
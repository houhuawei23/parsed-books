# 第5章 策略学习（Policy Learning）

到目前为止，我们主要关注估计**处理效应（treatment effects）**的方法。然而，在许多应用领域中，进行因果分析的基本目标并非估计处理效应，而是**指导决策制定**：我们希望理解处理效应，以便有效地制定处理方案并分配有限的资源。

**最优处理分配策略（optimal treatment assignment policies）**的学习问题与**处理异质性（treatment heterogeneity）**的估计问题密切相关，但二者又略有不同。一方面，策略学习看起来更简单：我们只关心将个体分配到处理组或对照组，而不关心除此之外对处理效应的精确估计。另一方面，在学习策略时，我们需要考虑在单纯估计处理效应时不存在的问题：任何我们实际想要使用的策略都必须足够简单以便于部署，不能基于受保护的特征进行歧视，不应依赖于可被操纵的特征等。

### 策略价值（Policy value）

就我们的目的而言，一个**处理分配策略（treatment assignment policy）** $\pi ( x )$ 是一个映射31

$$
\pi : \mathcal {X} \rightarrow \{0, 1 \}, \tag {5.1}
$$

使得具有特征 $X _ { i } = x$ 的个体当且仅当 $\pi ( x ) = 1$ 时接受处理。在**潜在结果（potential outcome）**设定下，当根据策略 $\pi$ 选择处理时，期望实现的结果为

$$
V (\pi) = \mathbb {E} \left[ Y _ {i} \left(\pi (X _ {i})\right) \right]. \tag {5.2}
$$

我们将 $V ( \pi )$ 称为策略 $\pi$ 的**价值（value）**，并假设决策者希望利用数据学习一个策略 $\hat { \pi }$ ，使得 $V ( \hat { \pi } )$ 尽可能大。这个框架隐含地假设结果 $Y _ { i }$ 捕捉了决策者想要优化的相关收益或奖励，并且决策者是功利主义的，即其目标是最大化各单元的平均收益。

### 工作流程（Workflow）

从概念上讲，策略学习工作流程包含三个关键阶段。首先，我们需要收集具有随机或准随机处理分配 $W _ { i }$ 的数据，以学习一个策略 $\hat { \pi }$ ；在本章中，我们将假设第一阶段中的处理是无混杂的，并且数据如同第3章的基本设定一样抽取。在第二阶段（可选），我们可能希望评估所学策略的质量，即估计 $V ( \hat { \pi } )$ 。这需要第二个数据集（通常称为**测试集（test set）**），其中处理分配是随机或准随机的。最后，完成学习后，我们进入最后阶段，可以选择部署所学策略，即我们可能选择设置 $W _ { i } = \hat { \pi } ( X _ { i } )$ ，期望通过 $Y _ { i } = Y _ { i } ( \hat { \pi } ( X _ { i } ) )$ 获得的期望结果 $\mathbb { E } \left[ Y _ { i } \right]$ 会很大。在第三阶段，处理效应不再具有随机性，因此我们无法再（非参数地）学习关于因果效应的任何信息。

正如前面在命题4.1中指出的，如果我们不对 $\pi$ 施加任何限制，那么 $V ( \pi )$ 的最大化策略是对**条件平均处理效应（Conditional Average Treatment Effect, CATE）**进行阈值化处理的策略：

$$
\pi^ {*} \in \operatorname{argmax} _ {\pi} \left\{V (\pi) \right\}, \quad \pi^ {*} (x) = 1 \left(\{\tau (x) > 0 \}\right). \tag {5.3}
$$

因此，学习策略的一种可能方法是将**插件原则（plug-in principle）**应用于 (5.3)：首先使用前一章讨论的方法生成 CATE 的估计 $\hat { \tau } ( \cdot )$ ，然后设置 $\hat { \pi } ( x ) ~ = ~ 1 ( \{ \hat { \tau } ( x ) > 0 \} )$ 。这种方法在某些应用中可能是合理的，但可能导致难以解释的策略，或可能不尊重应用中所要求的其他实际约束。本章的重点将是开发能够尊重此类约束的学习策略的方法；我们将在第5.2节中介绍这些方法，首先在下面讨论一些关于**策略评估（policy evaluation）**的预备知识。

**示例5（续）**。在上一章中，我们介绍了 Kitagawa 和 Tetenov [2018] 的一个示例，其中作者试图根据教育和收入来定位 JTPA 的资格。最优的无限制定位规则只是对 CATE 进行阈值化处理。然而，出于可行性的原因，他们最感兴趣的是以下形式的**线性处理规则（linear treatment rules）**32

$$
\tau (x) = 1 \left(\{\text{prior earnings} \cdot \alpha_ {1} + \text{education} \cdot \alpha_ {2} > c \}\right).
$$

学习这种类型的**福利最大化规则（welfare-maximizing rules）**需要本章介绍的新方法。

## 5.1 策略评估（Policy evaluation）

本章的主要关注点是策略学习工作流程的第一个“学习”部分，即如何使用数据选择一个好的策略 $\hat{\pi}$ 。然而，从方法论上讲，我们首先需要讨论工作流程的第二个“评估”部分：如果有人给我们一个策略 $\hat { \pi }$ ，我们如何估计 $V ( \hat { \pi } ) \ ?$

为了本节的目的，我们将假设我们可以访问一个包含 $n$ 个样本的测试集，其处理分配与第3章的基本设定一样是无混杂的，并且该测试集与用于学习候选策略 ${ \hat { \pi } }$ 的数据（即**训练集（training set）**）是独立的。然后，我们将讨论在给定训练集条件下对 $\hat { \pi }$ 的评估：在这里，我们不是试图估计 E $\left[ V ( \hat { \pi } ) \right]$ （即对 $\hat { \pi }$ 的随机性进行积分），而仅仅是估计当前特定实现的 $\hat { \pi }$ 的 $V ( \hat { \pi } )$ 。由于测试集和训练集相互独立，这个任务等价于使用测试集估计任意固定策略 $\pi$ 的 $V ( \pi )$ ；为简单起见，我们将在本节的剩余部分中针对后一个任务进行阐述。

### 逆概率加权（Inverse-propensity weighting）

考虑在无混杂性假设下评估一个给定的确定性策略 $\pi$ 。如果我们进一步知道**处理倾向得分（treatment propensities）** $e ( x )$ ，那么我们可以通过**逆概率加权（Inverse-Propensity Weighting, IPW）**获得 $V ( \pi )$ 的一个简单估计：

$$
\widehat {V} _ {I P W} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {1 \left(\left\{W _ {i} = \pi (X _ {i}) \right\}\right) Y _ {i}}{\mathbb {P} \left[ W _ {i} = \pi (X _ {i})   |   X _ {i} \right]}, \tag {5.4}
$$

其中 P $\lceil W _ { i } = \pi ( X _ { i } ) \mid X _ { i } = x \rceil = e ( x )$ 当 $\pi ( x ) = 1$ 时，否则为 $1 - e ( x )$ 。从性质上讲，这种方法对那些采样处理 $W _ { i }$ 与策略规定 $\pi ( X _ { i } )$ 匹配的观测值的结果进行平均，并使用逆概率加权来解释某些相关潜在结果未被观测到的事实。

当处理倾向得分已知时，我们可以使用与定理2.2相同的论证来验证，对于任何给定的策略 $\pi$ ，IPW 估计 $\widehat{V} _ {IPW} (\pi)$ 是 $V (\pi)$ 的无偏估计，

$$
\begin{array}{l} \mathbb {E} \left[ \widehat {V} (\pi) \right] = \mathbb {E} \left[ \frac {1 \left(\{W _ {i} = \pi (X _ {i}) \}\right) Y _ {i}}{\mathbb {P} \left[ W _ {i} = \pi (X _ {i}) \mid X _ {i} \right]} \right] \\ = \mathbb {E} \left[ \frac {1 \left(\left\{W _ {i} = \pi (X _ {i}) \right\}\right) Y _ {i} (\pi (X _ {i}))}{\mathbb {P} \left[ W _ {i} = \pi (X _ {i}) \mid X _ {i} \right]} \right] \tag {5.5} \\ = \mathbb {E} \left[ \mathbb {E} \left[ \frac {1 \left(\left\{W _ {i} = \pi (X _ {i}) \right\}\right)}{\mathbb {P} \left[ W _ {i} = \pi (X _ {i}) \mid X _ {i} \right]} \mid X _ {i} \right] \mathbb {E} \left[ Y _ {i} (\pi (X _ {i})) \mid X _ {i} \right] \right] \\ = \mathbb {E} \left[ Y _ {i} (\pi (X _ {i})) \right] = V (\pi), \\ \end{array}
$$

其中第二个等式由潜在结果的一致性得出，第三个等式由无混杂性得出。

### 增强逆概率加权（Augmented IPW）

在第3章中，我们讨论了基于 IPW 的**平均处理效应（Average Treatment Effect, ATE）**估计量（至少在使用真实倾向得分运行时）通常是低效的，并且对 $e ( x )$ 的估计误差不稳健；以及如何使用**增强逆概率加权（Augmented IPW, AIPW）**结构来解决这两个缺点。类似的考虑也适用于策略评估。为简洁起见，我们在此不重复第3章的推导，而只是陈述 AIPW 估计量及其关键性质。

与往常一样，构建 AIPW 需要条件响应函数 $\hat { \mu } _ { w } ( x )$ 的估计和倾向得分 $\hat { e } ( x )$ 的估计。给定这些估计，$V ( \pi )$ 的**插件非参数回归估计量（plug-in non-parametric regression estimator）**通过平均遵循策略 $\pi$ 会得到的预测来获得，即

$$
\widehat {V} _ {R E G} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \hat {\mu} _ {\pi (X _ {i})} (X _ {i}). \tag {5.6}
$$

AIPW 通过使用 IPW 从回归残差中提取任何剩余信号来对该估计量进行去偏，

$$
\widehat {V} _ {A I P W} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \hat {\mu} _ {\pi (X _ {i})} (X _ {i}) + \frac {1 \left(\{W _ {i} = \pi (X _ {i}) \}\right)}{\mathbb {P} \left[ W _ {i} = \pi (X _ {i}) \mid X _ {i} \right]} \left(Y _ {i} - \hat {\mu} _ {\pi (X _ {i})} (X _ {i})\right). \tag {5.7}
$$

与 AIPW 类型估计量的通常做法一样，在构建 AIPW 估计量时推荐使用**交叉拟合（cross-fitting）**。如果我们使用交叉拟合，并且使用以定理3.2中假设的速率收敛的 $\hat { \mu } _ { w } ( x )$ 和 ${ \hat { e } } ( x )$ 的估计，那么

$$
\begin{array}{l} \sqrt {n} \left(\widehat {V} _ {A I P W} (\pi) - V (\pi)\right) \\ \Rightarrow \mathcal {N} \left(0, \operatorname{Var} \left[ \mu_ {\pi (X _ {i})} (X _ {i}) \right] + \mathbb {E} \left[ \frac {\sigma_ {\pi (X _ {i})} ^ {2} (X _ {i})}{\mathbb {P} \left[ W _ {i} = \pi (X _ {i}) \mid X _ {i} \right]} \right]\right), \tag {5.8} \\ \end{array}
$$

并且 AIPW 估计量是有效的。这些结果的证明与第3章中使用的论证完全类似。

### 策略比较（Policy comparison）

通常需要比较两个策略 $\pi _ { 1 }$ 和 $\pi _ { 2 }$ ，通过估计它们价值的差异

$$
\Delta (\pi_ {1}, \pi_ {2}) = V (\pi_ {1}) - V (\pi_ {2}). \tag {5.9}
$$

例如，如果 $\pi _ { 0 }$ 是一个**现状（status-quo）**的处理分配规则，而 $\hat { \pi }$ 是一个新的数据驱动规则，那么差异 $\Delta ( \hat { \pi } , \pi _ { 0 } )$ 直接量化了采用数据驱动规则相对于现状的收益。

基于上述讨论，估计两个策略价值差异的一种自然方法是取其 AIPW 价值估计的差值。直接的代数运算可以将得到的估计量重新表达为简洁形式：

$$
\widehat {\Delta} _ {A I P W} (\pi_ {1}, \pi_ {2}) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\pi_ {1} (X _ {i}) - \pi_ {2} (X _ {i})\right) \widehat {\Gamma} _ {i},
$$

$$
\widehat {\Gamma} _ {i} = \hat {\mu} _ {(1)} (X _ {i}) - \hat {\mu} _ {(0)} (X _ {i}) + \frac {W _ {i}}{\hat {e} (X _ {i})} \left(Y _ {i} - \hat {\mu} _ {(1)} (X _ {i})\right) \tag {5.10}
$$

$$
- \frac {1 - W _ {i}}{1 - \hat {e} (X _ {i})} \left(Y _ {i} - \hat {\mu} _ {(0)} (X _ {i})\right),
$$

并且在定理3.2的条件下

$$
\begin{array}{l} \sqrt {n} \left(\widehat {\Delta} _ {A I P W} (\pi_ {1}, \pi_ {2}) - \Delta (\pi_ {1}, \pi_ {2})\right) \\ \Rightarrow \mathcal {N} \left(0, \operatorname{Var} \left[ \left(\pi_ {1} \left(X _ {i}\right) - \pi_ {2} \left(X _ {i}\right)\right) \tau \left(X _ {i}\right) \right] \right. \tag {5.11} \\ + \mathbb {E} \left[ 1 \left(\{\pi_ {1} (X _ {i}) \neq \pi_ {2} (X _ {i}) \}\right) \left(\frac {\sigma_ {0} ^ {2} (X _ {i})}{1 - e (X _ {i})} + \frac {\sigma_ {1} ^ {2} (X _ {i})}{e (X _ {i})}\right) \right]. \\ \end{array}
$$

当 $\pi _ { 1 }$ 和 $\pi _ { 2 }$ 在采取的行动上经常一致时，$\widehat { \Delta } _ { A I P W } ( \pi _ { 1 } , \pi _ { 2 } )$ 只需要考虑它们建议不同的较小区域内的结果——从而能够显著提高精度。

一个通常令人感兴趣的特定策略对比是将给定策略 $\pi$ 与**从不处理策略（never-treat policy）**进行比较。我们使用简写 $\Delta ( \pi ) = \Delta ( \pi , 0 )$ 表示这个量，并将其称为策略 $\pi$ 的**收益（benefit）**。我们还注意到，**始终处理策略（always-treat policy）**的收益 $\Delta ( 1 )$ 正好对应于平均处理效应，并且作为一种合理性检查，我们可以验证在这种情况下 (5.11) 只是定理3.2中结果的重新表述。

### 附注：处理优先级规则（Treatment prioritization rules）

实践中经常出现的一种策略类型是**处理优先级规则（treatment prioritization rules）**。这种策略从一个**优先级函数（priority function）** $S : \mathcal { X } \rightarrow \mathbb { R }$ 开始，然后根据优先级 $S ( X _ { i } )$ 对单元进行排序，并将处理分配给排名前 $q$ 比例的单元：

$$
\pi_ {S} ^ {q} = 1 \left(\left\{S (X _ {i}) \geq F _ {S} ^ {- 1} (1 - q) \right\}\right), \tag {5.12}
$$

其中 $F _ { S }$ 是优先级 $S ( X _ { i } )$ 的累积分布函数。这里，优先级函数可以是使用单独训练集获得的 CATE 估计，可以是量化谁在未接受处理时最有可能出现不良结果的风险度量，或者是其他与应用相关的优先级概念。

我们可以使用策略评估来量化优先级函数在将处理分配给最能从中受益的个体方面的成功程度。**QINI曲线（QINI curve）**估计处理排名前 $q$ 比例的单元的收益 $\Delta ( \pi _ { S } ^ { q } )$ 针对不同的 $q$ 值，然后在 Y 轴上绘制 $\Delta ( \pi _ { S } ^ { q } )$ 对 X 轴上的 $q$ 的图形。在每个单元具有恒定处理成本的设定中，QINI 曲线量化了一个成本效益分析，衡量当我们增加支出时获得的收益如何变化。

同时，**TOC曲线（TOC curve）**考虑 $q ^ { - 1 } \Delta ( \pi _ { S } ^ { q } ) - \Delta ( 1 )$ ，并将该量对 $q$ 作图。该曲线量化了由 $S ( \cdot )$ 优先排序的前 $q$ 比例单元比随机选择的单元从处理中受益更多的程度。这些数量在 Yadlowsky 等人 [2021] 中有所讨论；该论文还建议考虑以估计的 CATE 优先排序的单元下的 TOC 曲线下面积，作为整体检测到的处理异质性的有用度量。

处理优先级规则的价值可以再次使用**双重稳健方法（doubly robust approach）**进行估计：

$$
\widehat {\Delta} _ {A I P W} \left(\pi_ {S} ^ {q}\right) = \frac {1}{n} \sum_ {k = 1} ^ {\lfloor q n \rfloor} \widehat {\Gamma} _ {i (k)}, S \left(X _ {i (1)}\right) \geq S \left(X _ {i (2)}\right) \geq \ldots \geq S \left(X _ {i (n)}\right). (5.13)
$$

研究该估计量的大样本性质的一个统计挑战是它依赖于 $S ( X _ { i } )$ 的经验第 $q$ 个分位数，这会导致相对于 (5.8) 放大的渐近方差。Yadlowsky 等人 [2021] 为 (5.13) 中的价值估计以及为 QINI 和 TOC 曲线估计的诱导曲线下面积度量提供了中心极限定理；他们还讨论了这些量的基于重抽样的方法。

## 5.2 经验福利最大化（Empirical-welfare maximization）

我们现在回到学习策略的任务，即利用实验或准实验数据选择一个良好的**处理分配规则** $\hat { \pi } ( \cdot )$ 。在整个过程中，我们假设决策者被限制在某个可接受的策略类别 $\Pi$ 中选择策略 $\pi$ ；例如，$\Pi$ 可能包含对策略允许采取的函数形式或允许使用哪些变量的限制。可能考虑的策略类别的简单示例包括：线性阈值规则类 $\tau ( x ) = 1 ( \{ a \cdot x \geq c \} )$（其中 $a$ 为某个向量，$c$ 为阈值），或固定深度的决策树类。

在此设定下，最优策略（一个或多个）是那些在所有可接受策略中最大化策略价值的策略：

$$
\pi^ {*} \in \operatorname{argmax} \left\{V (\pi^ {\prime}): \pi^ {\prime} \in \Pi \right\}. \tag {5.14}
$$

任何非最优（但可接受）的策略 $\pi$ 都达不到这个最佳可能的策略价值，并会产生**遗憾**（regret）

$$
R (\pi) = \sup _ {\pi} \left\{V (\pi^ {\prime}): \pi^ {\prime} \in \Pi \right\} - V (\pi). \tag {5.15}
$$

我们的目标是学习一个策略，使其遗憾 $R ( { \hat { \pi } } )$ 具有保证的最坏情况界限。我们将此任务称为学习（而非估计）任务，因为 $\hat { \pi }$ 的性能仅根据其遗憾来评估。我们不会要求 $\hat { \pi }$ 在函数形式上收敛到 $\pi ^ { * }$ （事实上，也不假设存在唯一的最优策略 $\pi ^ { * }$ ）。

如果最优策略 $\pi ^ { * }$ 是真实价值函数 $V ( \pi )$ 在 $\pi \in \Pi$ 上的最大化器，那么通过最大化估计的价值函数来尝试学习 $\hat{\pi}$ 是很自然的：

$$
\hat {\pi} = \operatorname{argmax} \left\{\widehat {V} (\pi): \pi \in \Pi \right\}. \tag {5.16}
$$

Kitagawa 和 Tetenov [2018] 将这种方法称为**经验福利最大化**（empirical-welfare maximization）。在上一节中，我们已经讨论了利用随机或非混淆处理分配数据来估计 $V ( \pi )$ 的两种估计量，即 **IPW 估计量**和 **AIPW 估计量**，两者都可以用于根据 (5.16) 进行学习。我们将 $\widehat { V } _ { I P W } ( \pi )$ 在 $\pi \in \Pi$ 上的最大化器称为 ${ \hat { \pi } } _ { I P W }$ ，将 $\widehat { V } _ { A I P W } ( \pi )$ 上的最大化器称为 ${ \hat { \pi } } _ { A I P W }$ 。

**遗憾界（Regret bounds）** 证明经验福利最大化方法能够实现低遗憾超出了本书的范围；然而，我们在此勾勒出论证这一点的起点。设 $\pi ^ { * }$ 为达到最大策略价值的任意策略，$\hat { \pi }$ 为如 (5.16) 中估计价值的最大化器。那么，

$$
\begin{array}{l} R (\hat {\pi}) = V \left(\pi^ {*}\right) - V (\hat {\pi}) \tag {5.17} \\ = V \left(\pi^ {*}\right) - \widehat {V} \left(\pi^ {*}\right) + \widehat {V} \left(\pi^ {*}\right) - \widehat {V} (\hat {\pi}) + \widehat {V} (\hat {\pi}) - V (\hat {\pi}). \\ \end{array}
$$

由于 $\hat { \pi }$ 是估计价值的最大化器，我们有 $\widehat { V } \left( \pi ^ { * } \right) - \widehat { V } \left( \widehat { \pi } \right) \leq 0$ ，因此我们可以进一步得到

$$
\begin{array}{l} \begin{array}{l} R (\hat {\pi}) \leq V \left(\pi^ {*}\right) - \widehat {V} \left(\pi^ {*}\right) + \widehat {V} (\hat {\pi}) - V (\hat {\pi}) \\ 1. 2 \quad \left\{\left| \widehat {V} (x) - V (x) \right|, \dots , \Pi \right\} \end{array} (5.18) \\ \leq 2 \sup \left\{\left| \widehat {V} (\pi) - V (\pi) \right|: \pi \in \Pi \right\}, (5.18) \\ \end{array}
$$

特别是

$$
\mathbb {E} \left[ R (\hat {\pi}) \right] \leq 2 \mathbb {E} \left[ \sup \left\{\left| \widehat {V} (\pi) - V (\pi) \right|: \pi \in \Pi \right\} \right]. \tag {5.19}
$$

因此，为任何经验福利最大化方法证明遗憾界，都归结为证明 $\widehat V ( \pi )$ 的误差对所有可接受策略 $\pi \in \Pi$ 同时成立的一致界。

我们可以使用**经验过程理论（empirical process theory）** 的工具来界定 (5.19) 右侧的项；然而，这样做依赖于超出本文范围的技术性结果。为了陈述沿此路径获得的一个具体版本的结果，令 $\text{VC}(\Pi)$ 表示 $\Pi$ 的 **Vapnik-Chervonenkis 维数**（在许多实际案例中，可以基本上将 $\text{VC}(\Pi)$ 视为捕获指定 $\Pi$ 中一个元素所需的参数数量），并假设 $\text{VC}(\Pi)$ 是有限的。那么，Athey 和 Wager [2021] 表明——在定理 3.2 的条件以及进一步的正则性条件下——通过最大化 AIPW 价值估计量 (5.7) 学习到的策略满足

$$
\begin{array}{l} \limsup _ {n} \sqrt {n} \mathbb {E} \left[ R (\hat {\pi} _ {A I P W}) \right] \\ \leq 6 0 \sqrt {\operatorname{VC} (\Pi) \left(\operatorname{Var} \left[ \tau (X _ {i}) \right] + \mathbb {E} \left[ \frac {\sigma_ {0} ^ {2} (X _ {i})}{1 - e (X _ {i})} + \frac {\sigma_ {1} ^ {2} (X _ {i})}{e (X _ {i})} \right]\right)}. \tag {5.20} \\ \end{array}
$$

这个界有意义之处在于，它连接了经验福利最大化的最坏情况遗憾如何随各种问题原始参数变化。具体来说，我们看到这个界随着 $\Pi$ 维度的平方根（更大的策略空间更难学习）和 AIPW 分数的方差（当 ATE 估计更困难时，学习也更困难）的增加而增加，并随着样本量的平方根（更多数据有帮助）的增加而减少。不过，这里的常数 60 很可能是宽松的。³³

**作为加权分类的策略学习（Policy learning as weighted classification）** 上述关于遗憾的讨论表明，经验福利最大化原则上是一种有前景的策略学习方法。然而，要在实践中使用这种方法，需要能够以计算上易于处理的方式执行优化问题 (5.16)。这通常是一个具有挑战性的（非凸）优化问题；幸运的是，结果证明经验福利最大化问题在许多情况下等价于一个**加权分类问题（weighted classification problem）**，从而允许我们利用该领域的计算见解。

这里，我们专注于最大化 AIPW 价值估计量 (5.7)。作为有用的第一步，我们通过定义以下对称化目标：

$$
\widehat {A} _ {A I P W} (\pi) = \widehat {V} _ {A I P W} (\pi) - \widehat {V} _ {A I P W} (1 - \pi), \tag {5.21}
$$

即，相对于总是执行 $\pi$ 的反面，遵循 $\pi$ 所带来的估计改进。显然，$\pi$ 是 $\widehat { V } _ { A I P W } ( \pi )$ 的最大化器当且仅当它是 $\hat { A } _ { A I P W } ( \pi )$ 的最大化器；因此，我们可以等价地写成

$$
\hat {\pi} _ {A I P W} = \operatorname{argmax} \left\{\widehat {A} _ {A I P W} (\pi): \pi \in \Pi \right\}. \tag {5.22}
$$

此外，根据我们关于策略比较的讨论，我们可以验证

$$
\widehat {A} _ {A I P W} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} (2 \pi (X _ {i}) - 1) \widehat {\Gamma} _ {i}, \tag {5.23}
$$

其中 $\widehat { \Gamma } _ { i }$ 如 (5.10) 所定义。

出于优化的目的，关键在于我们现在可以将经验福利最大化问题重新表述为一个加权分类问题：

$$
\hat {\pi} _ {A I P W} = \operatorname{argmax} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \underbrace {(2 \pi (X _ {i}) - 1) \operatorname{sign} (\widehat {\Gamma} _ {i})} _ {\text {分类目标}} \underbrace {| \widehat {\Gamma} _ {i} |} _ {\text {样本权重}}: \pi \in \Pi \right\}. \tag {5.24}
$$

定性地说，这里的直觉是：策略学习等价于尝试选择一个尽可能匹配 AIPW 分数符号的策略，其权重对应于 AIPW 分数的大小。实际上，这个结果意味着我们可以使用任何用于加权分类的软件包来优化我们的目标函数并学习 ${ \hat { \pi } } _ { A I P W }$ 。

加权分类公式 (5.24) 从计算角度来看很有价值；然而，我们应小心不要过度解读它。在典型的信噪比情况下，AIPW 分数 $\widehat { \Gamma } _ { i }$ 的符号将相当随机，实际上可靠地预测这些符号是不可能的。即使是最优策略 $\pi ^ { * }$ 也会根据分类公式犯许多“错误”；而试图根据分类指标获得高精度只会导致过拟合。可能存在一些问题，其中经验福利最大化效果非常好（就相对于现状的价值改进而言），但应用于公式 (5.24) 的标准分类诊断却表明性能不佳。³⁴

**策略类别 $\Pi$ 的作用** 我们从非参数模型（即 $\mu _ { ( w ) } ( x )$ 和 $e ( x )$ 可以是通用的）开始，其中福利最大化的无限制处理分配规则就是 $\pi _ { u n r e s t r } ^ { * } ( x ) = 1 \left( \{ \tau ( x ) > 0 \} \right)$ 。然而，本章的目标不是找到一种近似 $\pi _ { u n r e s t r } ^ { * } ( \cdot )$ 的方法；相反，给定一个预先指定的策略类别 $\Pi$，我们试图从 $\Pi$ 中学习一个近乎遗憾最优的策略。例如，$\Pi$ 可以由线性决策规则、$k$-稀疏决策规则、深度 `\` 决策树等组成。特别要注意，我们从未假设 $\pi _ { u n r e s t r } ^ { * } ( \cdot ) \in \Pi$ 。

这个问题设定乍看起来可能令人惊讶。然而，在许多应用中，考虑在受限策略类别上进行学习是很重要的。一个关键原因是，在策略学习问题中，特征 $X _ { i }$ 可以扮演多个不同的角色。首先，$X _ { i }$ 可能是实现非混淆性所必需的

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp W _ {i} \mid X _ {i}.
$$

一般来说，我们能够获取的处理前变量越多，非混淆性就变得越可信。为了有一个可信的自然模型，使用各种各样的特征为 $e ( x )$ 和 $\mu _ { ( w ) } ( x )$ 建立灵活的非参数模型是很好的。

另一方面，当我们想要部署一个策略 $\pi ( \cdot )$ 时，我们应该更加谨慎地考虑使用哪些特征来做决策以及策略 $\pi ( \cdot )$ 的形式。根据应用的不同，可能有些特征是实现非混淆性所必需的，但在用于处理选择时却会产生问题。这些特征包括在部署系统中难以测量的特征、系统参与者可能利用（gameable）的特征，或者对应于法律保护类别的特征。在这种情况下，这些特征需要保留在数据集中以识别因果效应，但集合 $\Pi$ 应该只包含不依赖这些特征的策略 $\pi$。此外，许多应用涉及 $\pi ( \cdot )$ 的函数形式约束，这些约束可以合理地部署（例如，如果策略需要以非电子格式传达给员工，或使用非定量方法进行审计）。因此，在学习策略时，能够响应由应用驱动的约束至关重要，这些约束通过使用受限的允许策略类别 $\Pi$ 来体现。

## 5.3 文献注释（Bibliographic notes）

我们今天讨论背后的想法是，在学习策略时，自然应关注的数量是遗憾，而不是例如条件平均处理效应函数的平方误差损失。这一点在 Manski [2004] 中有论述。Stoye [2009] 提供了具有离散协变量的精确最小最大遗憾策略学习的讨论，而 Hirano 和 Porter [2009] 则在极限实验框架中考虑了渐近分析。

关于非混淆性下的策略学习可以构建为一个加权分类问题——并且我们可以调整来自经验风险最小化的众所周知的结果来推导有用的遗憾界——这一见解似乎是在统计学 [Zhao et al., 2012]、计算机科学 [Swaminathan and Joachims, 2015] 和经济学 [Kitagawa and Tetenov, 2018] 中独立发现的。具有双重稳健评分规则的策略学习的性质在 Athey 和 Wager [2021] 中推导得出。后一篇论文还考虑了更一般设定下的策略学习，例如针对连续治疗的“助推”（nudge）干预，或使用工具变量来识别内生治疗的效果。Mbakop 和 Tabord-Meehan [2021] 考虑了用于处理具有无限 VC 维的策略类别的经验福利最大化的模型选择，而 Zhou, Athey 和 Wager [2023] 则考虑了具有多种可能行动的结构化治疗选择。

在本章中，我们讨论了收敛速度，其规模为 $\sqrt { \mathrm { V C } ( \Pi ) / n }$ 。如果我们寻求对 $\tau ( x )$ 一致的保证，这是我们可以得到的最优收敛速度；当处理效应的强度以 $1 / \sqrt { n }$ 的速率随样本量衰减时，这些速率是尖锐的。然而，如果我们考虑 $\tau ( x )$ 固定选择的渐近性，则会出现超效率现象，并且我们可以获得比 $1 / \sqrt { n }$ 更快的速率 [Luedtke and Chambaz, 2020]；这种现象与通过经验风险最小化实现分类遗憾界的“大间隔”（large margin）改进密切相关。

用于评估处理优先化规则的 **QINI 曲线** 最初是在市场营销文献中引入的，用于量化定向营销活动的价值。Imai 和 Li [2023] 在 Neyman 模型下的随机对照试验中提供了 QINI 曲线的现代统计处理。Yadlowsky 等人 [2021] 在容纳双重机器学习的通用观察性研究设定中，为评估处理优先化规则的不同方法（包括 QINI 和 TOC 曲线）提供了统一分析。Sun 等人 [2021] 在处理成本也未知且需要估计的情况下，使用 QINI 曲线量化成本效益分析，而 Sverdrup 等人 [2023] 则在允许多种行动的处理优先化规则的情况下进行了类似工作。

策略学习是一个活跃的研究领域，近期取得了许多进展。例如，Bertsimas 和 Kallus [2020] 将通过学习优化特定问题的经验价值函数这一原则扩展到各种设定，如库存管理；Luedtke 和 van der Laan [2016] 讨论了最优策略价值的推断；而 Kallus 和 Zhou [2021] 则考虑了以对潜在的非混淆性失效具有鲁棒性的方式来学习策略的问题。
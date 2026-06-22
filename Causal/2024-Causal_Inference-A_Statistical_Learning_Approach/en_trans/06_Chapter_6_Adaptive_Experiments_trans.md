# 第6章 自适应实验（Adaptive Experiments）

在上一章中，我们考虑了**两阶段模型（two-phase model）**下的策略学习。在第一个“探索”阶段，我们拥有来自实验或观察性研究的数据，这些数据可用于识别干预的效果并选择策略。然后，在第二个“利用”阶段，我们可以部署所选策略——如果选择得当，就能获得回报。

这种两阶段模型，在工程文献中也称为**批学习模型（batch learning model）**，因其概念和操作上的简洁性而具有吸引力。然而，在许多情况下，实验对象自然连续到达且实验存在成本，使用预先指定探索和利用阶段的两阶段设计可能显得过于僵化——相反，我们可能希望在探索阶段获得的任何知识一旦可用就立即加以利用。例如，如果在探索阶段的某个时刻，我们确信已经发现了针对某些研究参与者的最佳策略，那么为什么不立即使用这些信息，而要等待预先指定的探索阶段结束呢？或者，在多臂试验中，如果明显看出其中一个臂明显较差，为什么不将其舍弃，并将可用的探索资源重新集中于其他臂上？

**例6.** Schwartz、Bradlow 和 Fader [2017] 描述了一个场景，其中一家金融机构试图通过在线广告获取新客户。广告商需要选择在哪里投放广告（例如，在哪种类型的网站上）以及使用哪种类型的广告，并且有兴趣通过实验来优化这些选择。作者展示了**自适应实验模型（adaptive experimentation model）**如何使广告商能够在同一广告活动中无缝地从探索信息过渡到利用信息，而无需事先承诺一个固定的实验样本量。还应注意，在这种场景下，获取标准随机试验的推断输出（例如置信区间和汇总统计量）的价值较低，因为任何学习结果可能都特定于给定的广告活动，并且可能无法推广到其他活动。

本章简要介绍**自适应实验（adaptive experiments）**的设计，在工程文献中也称为**多臂赌博机算法（multi-armed bandit algorithms）**。此类实验使研究人员能够根据初步发现修改其数据收集方案，目的是提高收集数据的质量和/或改善研究参与者的福利。处理自适应实验时的一个主要挑战是，我们用于学习的样本不再彼此独立，因为过去的结果会影响未来的处理分配；因此，为非自适应实验开发的方法不再具有形式上的合理性（实际上可能会严重失效）。

**设定与符号** 按照分析多臂自适应实验的标准做法，我们假设可以访问一个由 $t = 1 , \dots , T$ 个实验对象组成的流，每个对象可以被分配 $k = 1 , \ldots , K$ 个候选动作之一。我们用 $W _ { t } \in \{ 1 , \ldots , K \}$ 表示在时间 $t$ 采取的动作，用 $Y _ { t }$ 表示观察到的结果（或奖励），并将考虑 $W _ { t }$ 是过去数据的（可能是随机的）函数的情况。遵循潜在结果 $\{ Y _ { t } ( k ) \} _ { k = 1 } ^ { K }$，我们有 $Y _ { t } = Y _ { t } ( W _ { t } )$。

在本章中，我们还将做出以下假设。我们可以访问一个由 $t = 1 , \dots , T$ 个实验对象组成的流，使得：

*   潜在结果在 $t$ 上独立同分布：$\{ Y _ { t } ( k ) \} _ { k = 1 } ^ { K } \overset { \mathrm { i i d } } { \sim } F$。我们记 $\mu _ { k } = \mathbb { E } _ { F } \left[ Y _ { t } ( k ) \right]$ 为第 $k$ 个臂的平均奖励。
*   没有可用于定向的协变量 $X _ { t }$，并且分配的动作只能依赖于过去的动作和结果。

这两个假设在文献中都可以（并且经常）被放宽。存在能够处理非平稳甚至非随机潜在结果的算法，也存在允许使用协变量进行定向的算法（在工程文献中这被称为**上下文赌博机设定（contextual bandit setting）**）；有关参考文献，请参见书目注释部分。然而，在这里，我们只能简要地触及自适应实验文献的表面——并且将在上述受限设定的背景下进行。

## 6.1 低遗憾数据收集（Low-regret data collection）

在设计自适应数据收集算法时，可以有多个目标。我们将首先考虑那些遵循为 $t = 1 , \dots , T$ 个样本内实验对象获取高累积奖励（并避免低奖励动作）这一简单原则的方法。使用任何数据收集程序所能获得的最高期望奖励是 $T \mu ^ { * }$，其中 $\mu^* = \max \{ \mu_k : 1 \le k \le K \}$ 是平均奖励最高的臂的平均奖励。我们将根据其**遗憾（regret）**来评估自适应数据收集程序的质量：

$$
R _ {T} = \sum_ {t = 1} ^ {T} \left(\mu^ {*} - \mu_ {W _ {t}}\right), \tag {6.1}
$$

该式量化了相对于始终使用最佳臂而言的奖励缺口。35 在一个 $W _ { t }$ 在 $\{ 1 , \ldots , K \}$ 上均匀分布的非自适应试验中，$\begin{array} { r } { { \cal { R } } _ { T } \sim T \sum _ { k = 1 } ^ { K } \left( \mu ^ { * } - \mu _ { k } \right) / K } \end{array}$ 自适应实验方案的目标是做得更好，实现**次线性遗憾（sub-linear regret）**。为了做到这一点，任何算法首先都需要探索抽样分布，以找出哪些臂 $k = 1 , \ldots , K$ 最有前景，然后利用这些知识来获得低遗憾。

**上置信界方法（The upper confidence band method）** 自适应实验中探索-利用权衡问题的一个著名早期解决方案是 Lai 和 Robbins [1985] 的**上置信界（Upper Confidence Band, UCB）**算法。该算法的步骤如下。首先，使用 $t _ { 0 }$ 次抽取初始化每个臂，然后：

*   在每个时间 $t = K t _ { 0 } + 1 , K t _ { 0 } + 2 , . . . ,$ 基于截至时间 $t - 1$ 收集的数据，为 $\mu _ { k }$ 构建一个置信区间 $\widehat { U } _ { k , t }$，并且
*   选择与具有最大上端点的置信区间 $\widehat { U } _ { k , t }$ 对应的动作 $W _ { t }$，并观察 $Y _ { t } = Y _ { t } ( W _ { t } )$。

在高层次上，UCB 背后的动机是我们总是希望探索最具上升潜力的臂，即，UCB 在面对臂奖励的不确定性时是**乐观的（optimistic）**。如果我们对一个给定的臂了解尚少，它的置信区间会很长，UCB 会乐观地更频繁地对其进行采样。然而，随着时间的推移，我们将从差的臂收集足够的数据，以相当确信它们是次优的，即，即使它们置信区间的上端点也无法与其他臂可能获得的奖励竞争——此时 UCB 将停止对它们的采样。

实践中考虑了许多不同的 UCB 变体，它们源于用于臂选择的置信区间 $\widehat { U } _ { k , t }$ 的不同构造。为了理解 UCB 为何能控制遗憾，我们在此考虑一个适用于**高斯抽样模型（Gaussian sampling model）**的简单 UCB 变体，即：

$$
Y _ {t} (k) \sim \mathcal {N} \left(\mu_ {k}, \sigma^ {2}\right), \tag {6.2}
$$

其中 $\sigma ^ { 2 }$ 是已知的。高斯性以及已知的 $\sigma$ 和 $T$ 假设有助于简化分析；可以通过使用稍微更精细的算法和论证来摆脱这些假设。

我们将第 $k$ 个臂被抽取的累积次数以及其当前奖励的运行平均值记为：

$$
n _ {k, t} = \sum_ {j = 1} ^ {t} 1 \left(\{W _ {j} = k \}\right), \quad \hat {\mu} _ {k, t} = \frac {1}{n _ {k , t}} \sum_ {j = 1} ^ {t} 1 \left(\{W _ {j} = k \}\right) Y _ {j}, \tag {6.3}
$$

并如下选择动作：

$$
W _ {t} \in \operatorname{argmax} \left\{\widehat {U} _ {k, t} \right\}, \quad \widehat {U} _ {k, t} = \hat {\mu} _ {k, t - 1} + \sigma \sqrt {4 \log (T) / n _ {k , t - 1}}. \tag {6.4}
$$

这种选择是由 UCB 构造引起的，其中 $\mu _ { k , t }$ 的置信区间的宽度是估计标准误的 $\sqrt { 4 \log ( T ) }$ 倍。以下结果表明，该算法实际上以高概率实现了低遗憾。这里考虑的 UCB 变体由 Auer、Cesa-Bianchi 和 Fischer [2002] 提出，他们称此算法为 **UCB1 算法**。

**定理 6.1.** 在我们的抽样假设和高斯36独立同分布潜在结果 (6.2) 下，使用区间 (6.4) 和 $t _ { 0 } = 1$ 次初始抽取的 UCB 算法，其遗憾以至少 $1 - K / T$ 的概率被界定为：

$$
R _ {T} \leq 1 6 \sigma^ {2} \log (T) \sum_ {\{k: \mu_ {k} \neq \mu^ {*} \}} \frac {1}{\mu^ {*} - \mu_ {k}} + \sum_ {\{k: \mu_ {k} \neq \mu^ {*} \}} (\mu^ {*} - \mu_ {k}), \tag {6.5}
$$

**证明.** 为简单起见，我们假设存在一个唯一的最佳臂 $k ^ { * }$，其 $\mu _ { k ^ { * } } = \mu ^ { * } . ^ { 3 7 }$ 在我们的抽样模型下，遗憾 $R _ { T }$ 可以表示为：

$$
R _ {T} = \sum_ {k \neq k ^ {*}} n _ {k, T} \left(\mu_ {k ^ {*}} - \mu_ {k}\right). \tag {6.6}
$$

因此，我们的主要任务是界定 $n _ { k , T }$，即 UCB 可能抽取任何次优臂的次数；事实证明，UCB 本质上是一种为使得此类论证成立而逆向设计的算法。

为此，首先要检查的是，对于每个臂 $k \neq k ^ { * }$，对于所有 $t = K + 1 , \ldots , T$，我们有：

$$
\hat {\mu} _ {k, t - 1} \leq \mu_ {k} + \sigma \sqrt {4 \log (T) / n _ {k , t - 1}} \tag {6.7}
$$

成立的概率为 $1 - 1 / T$。这是因为，记 $\zeta _ { k , j }$ 为第 $k$ 个臂被第 $j$ 次抽取的时间，我们有：

$$
\begin{array}{l} \mathbb {P} \left[ \sup _ {K <   t \leq T} \left\{\mu_ {k} - \hat {\mu} _ {k, t - 1} - \sigma \sqrt {4 \log (T) / n _ {k , t - 1}} \geq 0 \right\} \right] \\ \leq \mathbb {P} \left[ \sup _ {1 \leq j \leq n _ {k, T}} \left\{\mu_ {k} - \hat {\mu} _ {k, \zeta_ {k, j}} - \sigma \sqrt {4 \log (T) / j} \geq 0 \right\} \right] \\ = \mathbb {P} \left[ \sup _ {1 \leq j \leq n _ {k, T}} \left\{\mu_ {k} - \frac {1}{j} \sum_ {l = 1} ^ {j} Y _ {l} ^ {\prime} (0) - \sigma \sqrt {4 \log (T) / j} \geq 0 \right\} \right] \\ \leq \mathbb {P} \left[ \sup _ {1 \leq j \leq T} \left\{\mu_ {k} - \frac {1}{j} \sum_ {l = 1} ^ {j} Y _ {l} ^ {\prime} (0) - \sigma \sqrt {4 \log (T) / j} \geq 0 \right\} \right] \\ \leq T \exp (- 2 \log (T)) = 1 / T, \\ \end{array}
$$

其中等式成立是由于数据生成过程的平稳性（这里，$Y _ { l } ^ { \prime } ( k )$ 是来自 $\mathcal { N } \left( \mu _ { k } , \sigma ^ { 2 } \right)$ 的独立抽取），最后一行是应用了带有联合界（union bound）的**次高斯尾部界（sub-Gaussian tail bound）**。通过重复相同的论证和另一个联合界，我们看到以至少 $1 - K / T$ 的概率，对于所有 $t = K + 1 , \ldots , T$ 有：

$$
\mu_ {k ^ {*}} \leq \hat {\mu} _ {k ^ {*}, t - 1} + \sigma \sqrt {4 \log (T) / n _ {k ^ {*} , t - 1}} \tag {6.8}
$$

并且 (6.7) 对所有 $k \neq k ^ { * }$ 同时成立。

当 (6.7) 和 (6.8) 成立时，我们只能在以下（必要但不充分）条件下抽取任何次优臂 $k \neq k ^ { * }$：

$$
\begin{array}{l} W _ {t} = k \implies \hat {\mu} _ {k, t - 1} + \sigma \sqrt {4 \log (T) / n _ {k , t - 1}} \geq \hat {\mu} _ {k ^ {*}, t - 1} + \sigma \sqrt {4 \log (T) / n _ {k ^ {*} , t - 1}} \\ \Longrightarrow \hat {\mu} _ {k, t - 1} + \sigma \sqrt {4 \log (T) / n _ {k , t - 1}} \geq \mu_ {k ^ {*}} \\ \Longrightarrow \mu_ {k} + 2 \sigma \sqrt {4 \log (T) / n _ {k , t - 1}} \geq \mu_ {k ^ {*}} \\ \Longrightarrow n _ {k, t - 1} \leq 1 6 \sigma^ {2} \log (T) / (\mu_ {k ^ {*}} - \mu_ {k}) ^ {2}. \\ \end{array}
$$

因此，当 (6.7) 和 (6.8) 成立时，一旦 $n _ { k , t - 1 }$ 超过某个截止点，抽取第 $k$ 个臂（对于某些 $k \neq k ^ { * }$）就变得不可能，所以：

$$
n _ {k, T} \leq 1 6 \sigma^ {2} \log (T) / (\mu_ {k ^ {*}} - \mu_ {k}) ^ {2} + 1.
$$

将其代入遗憾表达式 (6.6)，我们得到 (6.5)。

![image_02](images/image_02.png)

定理 6.1 立即表明，UCB 实际上成功地找到了次优臂并相当快地有效停止了它们，从而使得遗憾仅随 $T$ 呈对数增长。有趣的是，(6.5) 中的主导项来自 $\mu ^ { * } - \mu _ { k }$ 较小的“好”臂；直观地说，这些臂难以处理的原因是，需要更长的时间才能确定它们是次优的。这意味着，在自适应实验中包含一些非常差的臂的成本可能是有限的，因为像 UCB 这样的算法能够快速丢弃它们。

最后，应注意，上界 (6.5) 似乎允许由于 $\mu _ {  { k ^ { * } } } - \mu _ {  { k } }$ 非常小的准最优臂而导致无界遗憾。这仅仅是证明策略的一个产物，该策略侧重于效应较强的情况。当效应可能较弱时，我们可以简单地注意到，任何给定臂 $k$ 的最坏情况遗憾上界为 $T \left( \mu _ { k ^ { * } } - \mu _ { k } \right)$；并且，将此界与 (6.5) 隐含的界相结合，我们发现任何臂组合 $\mu _ { k }$ 的最坏情况遗憾被界定在 $K { \sqrt { T \log ( T ) } }$ 的量级上。

**汤普森采样（Thompson sampling）** UCB 是一种简单的自适应实验方法，对采样次优臂导致的超额遗憾有很强的界。然而，该算法对许多看似特设的选择很敏感，这些选择更多地与证明策略相关，而非透明的方法论考量，这可能导致在实践中性能次优。例如，上面给出的 UCB 版本使用了相对较宽的置信区间，其半长为 $\sqrt { 4 \log ( T ) }$ 个标准误；因此，定性地说，如果我们将 UCB 理解为总是选择最具上升潜力的臂，那么这个版本的 UCB 在评估上升潜力时是极其乐观的。如果我们使用半长为 1.96 个标准误的区间来运行 UCB，即对每个臂的上升潜力持更常规的乐观程度，会发生什么？在实践中，这可能（并且经常）效果很好（甚至可能更好），但定理 6.1 的证明将不再成立（因为事件 (6.7) 和 (6.8) 不再能够以高概率在所有时间上一致成立）。

当前的实证实践表明，我们可以通过使用仍然受“面对不确定性保持乐观”这一总体原则驱动，但以**贝叶斯（Bayesian）**而非频率学派推理来操作其乐观性的算法，来规避 UCB 的这种脆弱性。**汤普森采样（Thompson sampling）** [Thompson, 1933] 就是这样一种简单且广泛使用的算法。为了实现此算法，我们首先为潜在结果分布 $F$ 选择一个先验 $\Pi _ { 0 }$。然后，对于每个时间 $t = 1 , \dots , T$，我们：

*   计算每个臂 $k$ 是最佳臂的概率 $e _ { k , t - 1 }$，即：
    $$
    e _ {k, t - 1} = \mathbb {P} _ {\Pi_ {t - 1}} \left[ \mu_ {k} = \mu_ {*} \right], \tag {6.9}
    $$
*   随机选择一个动作 $W _ { t } \sim \mathrm { M u l t i n o m i a l } ( e _ { \cdot , t - 1 } )$，并且
*   观察 $Y _ { t } = Y _ { t } ( W _ { t } )$ 并更新后验 $\Pi _ { t }$。

可以通过**后验采样（posterior sampling）**来高效地实现此算法：首先从 $\Pi _ { t - 1 }$ 中抽取一个联合样本 $( \mu _ { 1 } ^ { \prime } , . . . , \mu _ { K } ^ { \prime } )$，然后设置 $W _ { t } = \mathrm { a r g m a x } \left. \mu _ { k } ^ { \prime } \right.$。

尽管汤普森采样表面上看起来与 UCB 非常不同，但它背后有相似的统计直觉。与 UCB 一样，汤普森采样会定期探索每个臂，直到它有效地确信该臂不好（即，该臂是最佳臂的后验概率降至 $1 / T$ 以下）；并且，例如，根据**伯恩斯坦-冯·米塞斯定理（Bernstein–von Mises theorem）**的直觉表明，这应该发生在与该臂的上置信界低于某个更好臂的整个置信区间时大致相同的信息量下。然而，证明与定理 6.1 类似的结论超出了本介绍的范围，我们转而参考 Agrawal 和 Goyal [2017] 以获取此类结果。

从实践角度来看，汤普森采样相对于 UCB 具有许多优势。汤普森采样对实现选择的敏感度低于 UCB；事实上，如果愿意通过从每个臂抽取 1 次来初始化算法，那么可以将 $\Pi _ { 0 }$ 设置为实数线上的**不恰当平坦先验（improper flat prior）**，从而得到一个无需调整参数的算法。38 并且，在实证评估中，汤普森采样通常被证明比 UCB 及相关算法更具韧性 [Chapelle and Li, 2011, Wu and Wager, 2022]。

## 6.2 自适应数据收集后的推断（Inference after adaptive data collection）

在自适应试验中收集数据后，可能还需要进行**统计推断**，例如，给出平均臂奖励参数 $\mu _ { k }$ 的置信区间。然而，这样做时需要谨慎，因为自适应数据收集会产生**非独立同分布（non-IID）数据**，从而可能使标准推断方法的保证失效。例如，在估计 $\mu _ { k }$ 时，两个自然想到的估计量包括**样本均值（sample mean）**

$$
\hat {\mu} _ {k} ^ {A V G} = \hat {\mu} _ {k, T} = \frac {1}{n _ {k , T} ^ {- 1}} \sum_ {j = 1} ^ {t} 1 \left(\{W _ {j} = k \}\right) Y _ {j} \tag {6.10}
$$

以及，在**汤普森采样（Thompson sampling）** 的情况下，**逆倾向得分加权估计量（inverse-propensity weighted estimator）**

$$
\hat {\mu} _ {k} ^ {I P W} = \frac {1}{T} \sum_ {t = 1} ^ {T} \frac {1 \left(\{W _ {t} = k \}\right) Y _ {t}}{e _ {t , k}}. \tag {6.11}
$$

然而，由于自适应数据收集方案，这两个估计量都不具有**渐近正态的极限分布**，从而阻碍了它们用于构建置信区间。

以下简单例子说明了在处理自适应收集数据时经典**中心极限定理（central limit theorem）** 的失效：

*   我们可以从单个臂中抽取结果 $Y _ { t } \sim \mathcal { N } ( \mu , 1 )$，其均值 $\mu$ 未知。
*   我们首先对 $n _ { 0 }$ 个样本进行**试点研究（pilot study）**，如果前 $n _ { 0 }$ 个样本的样本平均值是正的，则称试点研究通过（否则称其失败）。
*   如果试点研究通过，我们再收集 $1 0 n _ { 0 }$ 个样本；如果失败，我们只收集 $n _ { 0 }$ 个额外样本。

这个例子旨在通过一个简单的单臂设计来捕捉汤普森采样的定性行为，即一个臂的当前样本平均值越高，我们从中抽样的可能性就越大。图 6.1 展示了当 $\mu = 0$ 时，所得样本均值的缩放分布。我们很容易看到，$\hat { \mu } ^ { A \bar { V } G }$ 的缩放分布既是**非高斯的**，又是**向下偏倚的**，因此以 $\hat { \mu } ^ { A V G }$ 为中心的正态置信区间在此处无效。Nie 等人 [2018] 提供了一个一般性结果，表明在相当广泛的条件下，**遗憾最小化算法（regret-minimizing algorithms）** 的样本平均值是向下偏倚的。

与此同时，$\hat { \mu } ^ { I P W }$ 在可用时（例如，使用汤普森采样）是**无偏的**。然而，正如 Hadad 等人 [2021] 所讨论的，它仍然具有非高斯——且通常是**重尾的（heavy-tailed）**——抽样分布。因此，它同样不能用于正态推断。

如何使用自适应收集的数据进行最佳推断仍然是一个活跃的研究课题，对文献的全面回顾超出了本演讲的范围。然而，作为现有解决方案的一个指引，我们在此展示如何通过对数据进行仔细的**重新加权（re-weighting）** 来避免 $\hat {\mu} ^ {A V G}$ 和 $\hat {\mu} ^ {I P W}$ 的非高斯性问题。考虑一个**序贯随机化实验（sequentially randomized experiment）**，其中处理概率 $e _ { t }$ 可以依赖于过去的数据；汤普森采样是序贯随机化实验的一个例子。然后，我们将 $\mu _ { k }$ 的**自适应加权估计量（adaptively weighted estimate）** 定义为

$$
\hat {\mu} _ {k} ^ {A W} = \sum_ {t = 1} ^ {T} \frac {1 (\{W _ {t} = k \}) Y _ {t}}{\sqrt {e _ {t , k}}} / \sum_ {t = 1} ^ {T} \frac {1 (\{W _ {t} = k \})}{\sqrt {e _ {t , k}}}. \tag {6.12}
$$

这个估计量的设定可能看起来令人惊讶，因为单元被加权了 $1 / \sqrt { e _ { t , k } }$ 而不是更熟悉的 $1 / e _ { t , k }$ 逆倾向得分权重。然而，如下所示，这种加权方案会产生一个**渐近正态性**结果。我们注意到，在具有恒定处理倾向的随机试验情况下，正则条件 (6.14) 简化为熟悉的**林德伯格条件（Lindeberg condition）**；只要 $e _ { t , k }$ 不会衰减得太快，这个条件就是弱的。

**定理 6.2 (Theorem 6.2).** 在具有独立同分布（IID）潜在结果的序贯随机化实验中，假设对于所有臂 $k = 1 , \ldots , K$，有

$$
0 <   \sigma_ {k} ^ {2} := \operatorname{Var} \left[ Y _ {t} (k) \right] <   \infty \tag {6.13}
$$

，且 $e _ { t , k } > 0$ 几乎必然成立39，并且对于所有 $\varepsilon > 0$

$$
\lim _ {T \to \infty} \frac {1}{T} \sum_ {t = 1} ^ {T} \mathbb {E} \left[ (Y _ {t} - \mu_ {k}) ^ {2}   {\bf 1} \left(\left\{(Y _ {t} - \mu_ {k}) ^ {2} \geq \varepsilon   e _ {t, k}   T \right\}\right) \big |   {\cal F} _ {t - 1} \right] = 0, \tag {6.14}
$$

其中 $\mathcal { F } _ { t - 1 }$ 表示截至时间 $t - 1$ 收集到的信息。那么，

$$
\widehat {V} _ {k} ^ {- 1 / 2} \left(\widehat {\mu} _ {k} ^ {A W} - \mu_ {k}\right) \Rightarrow \mathcal {N} (0, 1),
$$

$$
\widehat {V} _ {k} = \sum_ {t = 1} ^ {T} \left(\frac {1 \left(\left\{W _ {t} = k \right\}\right) \left(Y _ {t} - \hat {\mu} _ {k} ^ {A W}\right)}{\sqrt {e _ {t , k}}}\right) ^ {2} / \left(\sum_ {t = 1} ^ {T} \frac {1 \left(\left\{W _ {t} = k \right\}\right)}{\sqrt {e _ {t , k}}}\right) ^ {2}. \tag {6.15}
$$

**证明 (Proof).** 我们首先陈述一个技术性结果，其证明推迟到本节末尾：在 (6.13) 和 (6.14) 下，

$$
\sum_ {t = 1} ^ {T} \frac {1 \left(\{W _ {t} = k \}\right)}{\sqrt {e _ {t , k}}} / \sqrt {T} \rightarrow_ {p} \infty , \tag {6.16}
$$

即 (6.12) 中的分母增长速度快于 $\sqrt { T }$。定性地说，(6.16) 意味着在 (6.12) 中使用的自适应加权方案下，我们的自适应抽样方案随时间收集了越来越多的数据。

现在，为了得到一个中心极限定理，我们注意到

$$
\hat {\mu} _ {k} ^ {A W} - \mu_ {k} = \sum_ {t = 1} ^ {T} \frac {1 (\{W _ {t} = k \}) (Y _ {t} - \mu_ {k})}{\sqrt {e _ {t , k}}} / \sum_ {t = 1} ^ {T} \frac {1 (\{W _ {t} = k \})}{\sqrt {e _ {t , k}}}, \tag {6.17}
$$

并首先关注上述表达式的分子。令

$$
M _ {t} = \sum_ {j = 1} ^ {t} \frac {1 \left(\left\{W _ {j} = k \right\}\right) \left(Y _ {j} - \mu_ {k}\right)}{\sqrt {e _ {j , k}}} \tag {6.18}
$$

为其部分和。因为 $W _ { t }$ 是基于截至时间 $t$ 的信息随机选择的，我们看到 $W _ { t }$ 与 $Y _ { t } ( k )$ 在给定截至时间 $t - 1$ 收集到的信息的条件下是独立的，因此 $M _ { t }$ 是一个**鞅（martingale）**：

$$
\mathbb {E} \left[ M _ {t} \mid \mathcal {F} _ {t - 1} \right] = M _ {t - 1}. \tag {6.19}
$$

此外，得益于我们的加权方案，我们可以检查每个鞅步长的条件方差是非随机的，尽管我们使用了自适应抽样概率：

$$
\operatorname{Var} \left[ M _ {t} \mid \mathcal {F} _ {t - 1} \right] = \sigma_ {k} ^ {2}. \tag {6.20}
$$

鉴于这两个事实，**鞅中心极限定理（martingale central limit theorem）** [Helland, 1982, Theorem 2.5(a)] 表明

$$
M _ {T} / \sqrt {T \sigma_ {k} ^ {2}} \Rightarrow \mathcal {N} (0, 1) \tag {6.21}
$$

只要对于所有 $\varepsilon > 0$，有

$$
\lim _ {T \to \infty} \frac {1}{T} \sum_ {t = 1} ^ {T} \mathbb {E} \left[ (M _ {t} - M _ {t - 1}) ^ {2} 1 \left(\left\{(M _ {t} - M _ {t - 1}) ^ {2} > \varepsilon T \right\}\right) \big | \mathcal {F} _ {t - 1} \right] = 0 \tag {6.22}
$$

在我们的设定中

$$
\begin{array}{l} \mathbb {E} \left[ \frac {1 \left(\left\{W _ {j} = k \right\}\right) \left(Y _ {j} - \mu_ {k}\right) ^ {2}}{e _ {j , k}} 1 \left(\left\{\frac {1 \left(\left\{W _ {j} = k \right\}\right) \left(Y _ {j} - \mu_ {k}\right) ^ {2}}{e _ {j , k}} > \varepsilon T \right\}\right) \mid \mathcal {F} _ {t - 1} \right] \\ = \mathbb {E} \left[ \frac {1 \left(\{W _ {j} = k \}\right) (Y _ {j} - \mu_ {k}) ^ {2}}{e _ {j , k}} 1 \left(\left\{\frac {(Y _ {j} - \mu_ {k}) ^ {2}}{e _ {j , k}} > \varepsilon T \right\}\right) \mid \mathcal {F} _ {t - 1} \right] \\ = \mathbb {E} \left[ (Y _ {j} - \mu_ {k}) ^ {2} 1 \left(\left\{(Y _ {j} - \mu_ {k}) ^ {2} > \varepsilon e _ {j, k} T \right\}\right) \mid \mathcal {F} _ {t - 1} \right], \\ \end{array}
$$

这意味着 (6.14) 等价于 (6.22)，因此 (6.21) 成立。

$\hat { \mu } _ { k } ^ { A W }$ 与 $\mu _ { k }$ 的差异归功于 (6.16) 和 (6.21)。同时，在 (6.14) 下，我们还有

$$
\sum_ {t = 1} ^ {T} \left(\frac {1 \left(\{W _ {t} = k \}\right) (Y _ {t} - \mu_ {k})}{\sqrt {e _ {t , k}}}\right) ^ {2} / \left(T \sigma_ {k} ^ {2}\right)\rightarrow_ {p} 1 \tag {6.23}
$$

这是由鞅集中性 [Helland, 1982, Lemma 2.3] 得到的；通过一致性，用 $\hat { \mu } _ { k } ^ { A W }$ 替换 $\mu _ { k }$ 同样成立。因此，由 (6.21) 和**斯卢茨基引理（Slutsky’s lemma）**，

$$
M _ {T} \Big / \sqrt {\sum_ {t = 1} ^ {T} \left(\frac {1 \left(\left\{W _ {t} = k \right\}\right) \left(Y _ {t} - \hat {\mu} _ {k} ^ {A W}\right)}{\sqrt {e _ {t , k}}}\right) ^ {2}} \Rightarrow \mathcal {N} (0, 1). \tag {6.24}
$$

$\hat { \mu } _ { k } ^ { A W }$ 和 $\widehat { V } _ { k } ^ { 1 / 2 }$ 相互抵消。

定理 6.2 的证明揭示了为什么自适应加权估计量 $\hat { \mu } _ { k } ^ { A W }$ 可以工作，而 $\hat { \mu } _ { k } ^ { A V G }$ 和 $\hat { \mu } _ { k } ^ { I P W }$ 可能不行。自适应加权估计量的加权方案本质上是逆向工程设计的，目的是使**可预测方差条件（predictable variance condition）** (6.20) 成立，从而能够应用鞅中心极限定理。$\hat { \mu } _ { k } ^ { A V G }$ 和 $\hat { \mu } _ { k } ^ { I P W }$ 在自适应实验中通常不具备此性质。Hadad 等人 [2021] 将允许应用鞅中心极限定理的权重称为**“方差稳定化（variance stabilizing）”**，并研究了一个包含 $\hat { \mu } _ { k } ^ { A W }$ 作为特例的方差稳定化估计量族。

**(6.16) 的证明 (Proof of (6.16)).** 现在仍需确立定理 6.2 证明中剩余的技术性主张。我们的首要任务是验证

$$
E _ {T, k} / \sqrt {T} \rightarrow_ {p} \infty , \quad E _ {T, k} = \sum_ {t = 1} ^ {T} \sqrt {e _ {t , k}}. \tag {6.25}
$$

在 (6.13) 下，我们可以选择一个 $\alpha _ { k } > 0$ 使得

$$
\mathbb {E} \left[ (Y _ {t} - \mu_ {k}) ^ {2} {\bf 1} \left(\left\{(Y _ {t} - \mu_ {k}) ^ {2} \geq \alpha_ {k} \right\}\right) \right] \geq \frac {\sigma_ {k} ^ {2}}{2}.
$$

然后，通过在给定过去数据的条件下反复应用**马尔可夫不等式（Markov’s inequality）**，我们看到 (6.14) 中的关键和可以从下方被界定为

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 1} ^ {T} \mathbb {E} \left[ (Y _ {t} - \mu_ {k}) ^ {2} {\bf 1} \left(\{(Y _ {t} - \mu_ {k}) ^ {2} \geq \varepsilon e _ {t, k} T \}\right) \big | \mathcal {F} _ {t - 1} \right] \\ \geq \frac {\sigma_ {k} ^ {2}}{2} \frac {1}{T} \sum_ {t = 1} ^ {T} 1 \left(\{\varepsilon e _ {t, k} T \leq \alpha_ {k} \}\right) \geq \frac {\sigma_ {k} ^ {2}}{2} \frac {1}{T} \sum_ {t = 1} ^ {T} 1 \left(\left\{\sqrt {e _ {t , k}} \leq \sqrt {\alpha_ {k} / (\varepsilon T)} \right\}\right). \\ \end{array}
$$

根据 (6.14)，对于每个 $\varepsilon > 0$，这个表达式必须在概率上收敛到 0。因此，对于任何 $\varepsilon > 0$，除了一个消失比例的单位外，我们以高概率有 $\sqrt { e _ { t , k } } \ge \sqrt { \alpha _ { k } / ( \varepsilon T ) }$，因此 $(6.25)$ 必须成立。

下一步，我们构造另一个 $\mathcal { F } _ { t }$ 鞅 $X _ { t }$，其差分为

$$
X _ {t} - X _ {t - 1} = \sqrt {e _ {t , k}} - \frac {1 (\{W _ {t} = k \})}{\sqrt {e _ {t , k}}}.
$$

这个鞅的增量有上界，$X _ { t } ~ - ~ X _ { t - t } ~ \le ~ 1$，并且方差增量 Var $\left[ X _ { t } \middle | \mathcal { F } _ { t - 1 } \right] = 1 - e _ { t , k } \leq 1$。Freedman [1975, Theorem 4.1] 随后表明，对于任何 $a > 0$，

$$
\mathbb {P} \left[ X _ {T} \geq a \right] \leq \exp \left[ - \frac {a ^ {2}}{2 (a + T)} \right]. \tag {6.26}
$$

现在，给定 (6.25)，我们知道存在一个函数 $r ( T )$ 使得 $r ( T ) \to \infty$ 且 $\mathbb { P } [ E _ { T , k } / ( 2 r ( T ) \sqrt { T } ) ] \to 1$。将 $a = r ( T ) \sqrt { T }$ 代入上述表达式，我们得到

$$
\lim _ {T \to \infty} \mathbb {P} \left[ \sum_ {t = 1} ^ {T} \frac {1 (\{W _ {t} = k \})}{\sqrt {e _ {t , k}}} \leq \sum_ {t = 1} ^ {T} \sqrt {e _ {t , k}} - r (T) \sqrt {T} \right] = 0,
$$

由于 $E _ { T , k } \ge 2 r ( T ) \sqrt { T }$ 以高概率成立，这意味着 (6.16)。

**自适应研究设计中的权衡 (Trade-offs in adaptive study design)** 在本章中，我们考虑了与自适应实验相关的两个高层次问题。首先，我们询问如何收集数据以最小化样本内遗憾；然后，我们询问如何使用自适应收集的数据为平均臂奖励构建置信区间。基于此背景，自然要问是否有可能协调这两个任务——同时实现低样本内遗憾和强大的实验后推断。

然而，这里的答案不幸是明确的**否定**：像 (6.1) 中那样**激进地优化样本内遗憾**的数据收集方案将导致脆弱的实验后推断。Bubeck, Munos, 和 Stoltz [2009] 就数据收集方案实现的样本内遗憾与通过将实验中的最佳臂部署到未来数据上可能获得的实验后遗憾之间，提供了一个形式化的权衡。Fan 和 Glynn [2021] 表明，任何实现最优样本内期望遗憾的自适应算法都必然具有**重尾的遗憾分布**（即，该算法有一个很小但不可忽略的概率会完全失败并招致巨大遗憾）。最后，从技术角度讲，激进地缩减表现不佳臂的倾向 $e _ { t , k }$ 的算法可能不满足林德伯格条件 (6.14)，因此可能无法通过所提出的方法进行有效的实验后推断。

因此，在自适应实验的设计中存在**不可避免的权衡**，研究人员应根据其目标选择相关的数据收集策略。如果目标是快速推出一个策略并立即最小化研究参与者的样本内遗憾，那么像汤普森采样这样的算法是一个自然的选择。然而，如果研究人员还想使用收集到的数据来指导未来的策略，那么使用那些在缩减次优臂使用速度上不那么激进的算法是更可取的 [Bubeck et al., 2009, Fan and Glynn, 2021]。我们还注意到，有大量文献致力于设计自适应实验，以最大化我们在 $T$ 个时间步后识别出**最佳臂 [Russo, 2020]** 或**准最优臂 [Kasy and Sautmann, 2021]** 的机会。

## 6.3 文献注释（Bibliographic notes）

关于**赌博机算法（bandit algorithms）** 的这一系列工作建立在 Lai 和 Robbins [1985] 关于 UCB 算法的早期成果之上。Lai 和 Robbins [1985] 表明，UCB 的一个变体实现了形如 (6.5) 的遗憾缩放，并且这种行为是**渐近最优的**。定理 6.1 中给出的那种**有限样本界（finite-sample bounds）** 由 Auer, Cesa-Bianchi, 和 Fischer [2002] 建立，而 Agrawal 和 Goyal [2017] 为汤普森采样提供了类似的界。得益于其**贝叶斯（Bayesian）** 设定，汤普森采样可以推广到各种自适应学习问题；参见 Russo 等人 [2018] 的近期综述。我们还注意到，UCB 和汤普森采样远非此任务唯一可用的算法；例如，Russo 和 Van Roy [2018] 提出了**信息导向采样（information-directed sampling）**，这是另一种贝叶斯启发式方法，他们认为这是汤普森采样的一个有吸引力的替代方案。

在第 6.1 节中，我们考虑了能够快速收敛到对 $K$ 个可用动作中最佳者进行采样的自适应实验。我们使用的**计量经济学（econometric）** 设定做了 3 个在应用中可能不成立的主要假设：我们没有考虑可用于指导决策的**协变量（covariates）** $X _ { t }$；我们只将样本内遗憾作为目标；并且我们假设抽样分布随时间稳定。这些假设中的每一个都在文献中被放宽了。关于**上下文赌博机（contextual bandits）** 的文献允许通过**参数化 [Bastani and Bayati, 2020, Goldenshluger and Zeevi, 2013]** 或**非参数化 [Gur, Momeni, and Wager, 2022, Hu, Kallus, and Mao, 2022a, Perchet and Rigollet, 2013]** 设定将潜在结果与协变量 $X _ { t }$ 联系起来。关于**最佳臂选择（best-arm selection）** 的文献已在上面讨论过 [Bubeck et al., 2009, Kasy and Sautmann, 2021, Russo, 2020]。最后，Besbes, Gur, 和 Zeevi [2019], Liu, Van Roy, 和 Xu [2023] 以及 Qin 和 Russo [2022] 考虑了奖励分布可能随时间变化的不同模型，并提出了针对此设定量身定制的算法。还有大量关于**对抗性模型（adversarial model）** 的文献，在该模型中，类似于**内曼模型（Neyman model）**，不对潜在结果做抽样假设，随机性的唯一来源是随机化的动作选择；参见 Bubeck 和 Cesa-Bianchi [2012] 的综述和参考文献。

通过**方差稳定化加权（variance-stabilizing weighting）** 对自适应收集数据进行推断的这一系列工作由许多作者进行，包括 Luedtke 和 van der Laan [2016], Hadad 等人 [2021] 以及 Zhang, Janson, 和 Murphy [2020]。应该指出，这并不是自适应实验中进行推断的唯一可能方法。特别是，此设定中一个经典的推断替代方法始于基于**迭代对数定律（law of the iterated logarithm）** 及其推广的置信带，这些带对每个 $t$ 值同时成立；参见 Robbins [1970] 的里程碑式综述和 Howard 等人 [2021] 的最新进展。还可以使用由**弱信号渐近性（weak-signal asymptotics）** 驱动的自适应实验的**扩散近似（diffusion approximations）** 来构建置信区间 [Hirano and Porter, 2023, Kuang and Wager, 2024]。

最后，今天讨论的所有自适应实验方法本质上都是**启发式算法（heuristic algorithms）**，可以证明它们具有良好的渐近行为（即，UCB 和汤普森采样都不能直接从最优性原理推导出来）。在贝叶斯情况下（即，我们有一个关于 $F$ 的实际主观先验，而不仅仅是汤普森采样用来驱动一个具有频率学派保证的算法的便利先验），可以通过**动态规划（dynamic programming）** 求解最优的遗憾最小化实验设计 [Gittins, 1979]。
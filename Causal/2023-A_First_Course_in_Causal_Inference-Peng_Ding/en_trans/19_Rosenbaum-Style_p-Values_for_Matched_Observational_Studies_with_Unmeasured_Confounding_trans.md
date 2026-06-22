# 针对存在未测量混杂因素的配对观察性研究的Rosenbaum式p值（Rosenbaum-Style p-Values for Matched Observational Studies with Unmeasured Confounding）

Rosenbaum (1987b) 为配对观察性研究引入了一种**敏感性分析（sensitivity analysis）**技术。尽管该方法适用于一般场景 (Rosenbaum, 2002b)，但其理论在**一对一匹配（one-to-one matching）**中最为简洁。与第17章和第18章不同，Rosenbaum式敏感性分析最适用于配对观察性研究，以检验**无个体处理效应的尖锐零假设（sharp null hypothesis of no individual treatment effect）**。

## 19.1 配对数据敏感性分析的模型（The model for sensitivity analysis with matched data）

考虑来自一项观察性研究的精确匹配对，其中 $(i, j)$ 索引配对 $\textit { i }$ 中的单位 $j$ $( i = 1 , \dots , n ; j = 1 , 2 )$。假设独立同分布（iid）抽样，并将**倾向得分（propensity score）**定义为：

$$
e _ {i j} = \operatorname{pr} \left\{Z _ {i j} = 1 \mid X _ {i}, Y _ {i j} (1), Y _ {i j} (0) \right\}.
$$

令 $\mathbb { S } _ { i } = \{ Y _ { i 1 } ( 1 ) , Y _ { i 1 } ( 0 ) , Y _ { i 2 } ( 1 ) , Y _ { i 2 } ( 0 ) \}$ 表示配对 i 内所有潜在结果的集合。以事件 $Z _ { i 1 } + Z _ { i 2 } = 1$ 为条件，我们有：

$$
\begin{array}{l} \pi_ {i 1} = \operatorname{pr} \left\{Z _ {i 1} = 1 \mid X _ {i}, \mathbb {S} _ {i}, Z _ {i 1} + Z _ {i 2} = 1 \right\} \\ = \frac {\operatorname{pr} \left\{Z _ {i 1} = 1 , Z _ {i 2} = 0 \mid X _ {i} , \mathbb {S} _ {i} \right\}}{\operatorname{pr} \left\{Z _ {i 1} + Z _ {i 2} = 1 \mid X _ {i} , \mathbb {S} _ {i} \right\}} \\ = \frac {\operatorname{pr} \left\{Z _ {i 1} = 1 , Z _ {i 2} = 0 \mid X _ {i} , \mathbb {S} _ {i} \right\}}{\operatorname{pr} \left\{Z _ {i 1} = 1 , Z _ {i 2} = 0 \mid X _ {i} , \mathbb {S} _ {i} \right\} + \operatorname{pr} \left\{Z _ {i 1} = 0 , Z _ {i 2} = 1 \mid X _ {i} , \mathbb {S} _ {i} \right\}} \\ = \frac {e _ {i 1} (1 - e _ {i 2})}{e _ {i 1} (1 - e _ {i 2}) + (1 - e _ {i 1}) e _ {i 2}} \\ \end{array}
$$

定义 $o _ { i j } = e _ { i j } / ( 1 - e _ { i j } )$ 为单位 $( i , j )$ 接受处理的**优势比（odds）**，我们有：

$$
\pi_ {i 1} = \frac {o _ {i 1}}{o _ {i 1} + o _ {i 2}}.
$$

在**可忽略性（ignorability）**假设下， $e _ { i j }$ 仅是 $X _ { i }$ 的函数，因此， $e _ { i 1 } = e _ { i 2 }$ 且 $\pi _ { i 1 } = 1 / 2$ 。因此，以协变量和潜在结果为条件的处理分配机制，等价于一个处理组和对照组概率相等的**匹配对实验（Matched Pair Experiment, MPE）**。这是我们在第15.1章讨论的分析配对观察性研究的一种策略。

一般情况下， $e _ { i j }$ 也是未观测到的潜在结果的函数，其取值范围可以从0到1。Rosenbaum (1987b) 的敏感性分析模型对优势比 $o _ { i 1 } / o _ { i 2 }$ 施加了界限。

**假设 19.1（Rosenbaum 敏感性模型）** 优势比满足以下界限：

$$
o _ {i 1} / o _ {i 2} \leq \Gamma , \quad o _ {i 2} / o _ {i 1} \leq \Gamma ,
$$

其中 $\Gamma \geq 1$ 是预先指定的。等价地：

$$
\frac {1}{1 + \Gamma} \leq \pi_ {i 1} \leq \frac {\Gamma}{1 + \Gamma}
$$

其中 $\Gamma \geq 1$ 是预先指定的。

在假设19.1下，我们得到一个有偏的MPE，其处理组和对照组的概率在各配对间不相等且变化。当 $\Gamma = 1$ 时，我们有 $\pi _ { i 1 } = 1/2$ ，从而得到一个标准的MPE。因此， $\Gamma > 1$ 衡量了由于匹配中遗漏变量导致的与理想MPE的偏离程度。

## 19.2 Rosenbaum 敏感性模型下的最坏情况 p 值（Worst-case p-values under Rosenbaum’s sensitivity model）

考虑检验尖锐零假设：

$$
H _ {0 \mathrm{F}}: Y _ {i j} (1) = Y _ {i j} (0) \text {   for   } i = 1, \dots , n \text {   and   } j = 1, 2
$$

基于配对内差异 $\hat { \tau } _ { i } = ( 2 Z _ { i 1 } - 1 ) ( Y _ { i 1 } - Y _ { i 2 } ) ~ ( i = 1 , \ldots , n )$。在 $H _ { \mathrm { 0 F } }$ 下， $|\hat{\tau}_i|$ 是固定的，但如果 $\hat { \tau } _ { i } \neq 0$ ，则 $S _ { i } = I ( \hat { \tau } _ { i } > 0 )$ 是随机的。考虑以下一类的**检验统计量（test statistics）**：

$$
T = \sum_ {i = 1} ^ {n} S _ {i} q _ {i},
$$

其中 $q _ { i } \geq 0$ 是 $\big ( \vert \hat { \tau } _ { 1 } \vert , \dots , \vert \hat { \tau } _ { n } \vert \big )$ 的函数。特殊情况包括**符号统计量（sign statistic）**、**配对 t 统计量（pair t statistic）**（经过某个常数偏移后）和**Wilcoxon 符号秩统计量（Wilcoxon sign rank statistic）**：

$$
T = \sum_ {i = 1} ^ {n} S _ {i}, \quad T = \sum_ {i = 1} ^ {n} S _ {i} | \hat {\tau} _ {i} |, \quad T = \sum_ {i = 1} ^ {n} S _ {i} R _ {i},
$$

其中 $( R _ { 1 } , \ldots , R _ { n } )$ 是 $\big ( \vert \hat { \tau } _ { 1 } \vert , \dots , \vert \hat { \tau } _ { n } \vert \big )$ 的秩。

对于一般的 $\Gamma$ ，检验统计量的零分布是什么？这可能相当复杂，因为我们没有完全指定 $\pi _ { i 1 }$ 的确切值。幸运的是，我们知道最坏情况的分布对应于：

$$
S _ {i} \stackrel {\mathrm{IID}} {\sim} \text {Bernoulli} \left(\frac {\Gamma}{1 + \Gamma}\right).
$$

这里，基于 $T$ 的**随机化检验（FRT）**在“最坏情况”分布下具有最大的 p 值。相应的分布具有均值：

$$
E _ {\Gamma} (T) = \frac {\Gamma}{1 + \Gamma} \sum_ {i = 1} ^ {n} q _ {i},
$$

和方差：

$$
\mathrm{var} _ {\Gamma} (T) = \frac {\Gamma}{(1 + \Gamma) ^ {2}} \sum_ {i = 1} ^ {n} q _ {i} ^ {2},
$$

并具有**正态近似（Normal approximation）**：

$$
\frac {T - \frac {\Gamma}{1 + \Gamma} \sum_ {i = 1} ^ {n} q _ {i}}{\sqrt {\frac {\Gamma}{(1 + \Gamma) ^ {2}} \sum_ {i = 1} ^ {n} q _ {i} ^ {2}}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

在实践中，我们可以报告一系列作为 $\Gamma$ 函数的 p 值。

## 19.3 重新审视 LaLonde 数据（Revisiting the LaLonde data）

我们在匹配后的 LaLonde 数据中进行 Rosenbaum 式敏感性分析。我们考虑使用检验统计量 $\textstyle T = \sum _ { i = 1 } ^ { n } S _ { i } | { \hat { \tau } } _ { i } |$ 。在 $\Gamma = 1$ 的理想匹配对实验下，我们可以模拟 $T$ 的分布并得到 p 值为 0.002，如图 19.1 的第一个子图所示。当 $\Gamma$ 略微增大到 1.1 时， $T$ 的分布向右移动，p 值增加到 0.011。如果进一步将 $\Gamma$ 增加到 1.3，则 $T$ 的分布进一步右移，p 值超过 0.05。图 19.2 显示了 $\hat { \tau } _ { i }$ 的直方图以及作为 $\Gamma$ 函数的 p 值； $\Gamma = 1 . 2 3 3$ 衡量了在 0.05 水平上我们仍能拒绝零假设的最大混杂程度。

我们也可以使用 `sensitivitymw` 包中的 `senmw` 函数来获取一系列随 $\Gamma$ 变化的 p 值，如图 19.2 所示。

## 19.4 课后习题（Homework Problems）

**19.1 Rosenbaum 方法的应用（Application of Rosenbaum’s approach）**

使用 Rosenbaum 方法重新分析示例 10.3。

**19.2 推荐阅读（Recommended reading）**

Rosenbaum (2015) 为其两个用于配对观察性研究敏感性分析的 R 包提供了一个教程。

## 20
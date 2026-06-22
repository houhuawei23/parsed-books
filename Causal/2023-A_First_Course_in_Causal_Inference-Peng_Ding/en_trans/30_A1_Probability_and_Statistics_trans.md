# A1 概率与统计（A1 Probability and Statistics）

## A1.1 概率（A1.1 Probability）

### A1.1.1 塔性质与方差分解（Tower property and variance decomposition）

给定随机变量或向量 $A , B , C ,$ ，我们有

$$
E (A) = E \{E (A \mid B) \}
$$

和

$$
E (A \mid C) = E \{E (A \mid B, C) \mid C \}.
$$

给定随机变量 A 和随机变量或向量 $B , C ,$ ，我们有

$$
\operatorname{var} (A) = E \{\operatorname{var} (A \mid B) \} + \operatorname{var} \{E (A \mid B) \}
$$

和

$$
\operatorname{var} (A \mid C) = E \{\operatorname{var} (A \mid B, C) \mid C \} + \operatorname{var} \{E (A \mid B, C) \mid C \}.
$$

类似地，我们可以将协方差分解为

$$
\operatorname{cov} \left(A _ {1}, A _ {2}\right) = E \left\{\operatorname{cov} \left(A _ {1}, A _ {2} \mid B\right) \right\} + \operatorname{cov} \left\{E \left(A _ {1} \mid B\right), E \left(A _ {2} \mid B\right) \right\}
$$

和

$$
\operatorname{cov} \left(A _ {1}, A _ {2} \mid C\right) = E \left\{\operatorname{cov} \left(A _ {1}, A _ {2} \mid B, C\right) \mid C \right\} + \operatorname{cov} \left\{E \left(A _ {1} \mid B, C\right), E \left(A _ {2} \mid B, C\right) \mid C \right\}.
$$

### A1.1.2 极限定理（Limiting theorems）

**定义 A1.1（依概率收敛）** 如果对于每一个 $\varepsilon > 0$ ，有

$$
\operatorname{pr} (| X _ {n} - X | > \varepsilon) \to 0
$$

当 $n \rightarrow \infty$ 时成立，则称随机变量序列 $( X _ { n } ) _ { n \geq 1 }$ 依概率收敛于 X。

**定义 A1.2（依分布收敛）** 如果对于 $\operatorname{pr} (X \leq x)$ 的所有连续点 x，当 $n \to \infty$ 时有

$$
\operatorname{pr} (X _ {n} \leq x) \to \operatorname{pr} (X \leq x)
$$

则称随机变量序列 $( X _ { n } ) _ { n \geq 1 }$ 依分布收敛于 X。依概率收敛强于依分布收敛。定义 A1.1 和 A1.2 有助于陈述概率论中的以下两个基本定理。

**定理 A1.1（大数定律）** 如果 $X _ { 1 } , \ldots , X _ { n } \stackrel { I I D } { \sim } X$ 且 $E | X | <$ $\infty$ ，则 $\begin{array} { r } { \bar { X } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } X _ { i } \to E ( X ) } \end{array}$ 依概率成立。

定理 A1.1 中的大数定律表明，样本均值在极限意义下接近总体均值。

**定理 A1.2（中心极限定理）** 如果 $\begin{array} { r l } { X _ { 1 } , \ldots , X _ { n } \quad { \stackrel { I I D } { \sim } } } & { { } X } \end{array}$ 且 $\operatorname{var} ( X ) < \infty$ ，则

$$
\frac {\bar {X} - E (X)}{\sqrt {\operatorname{var} (X) / n}} \to \mathrm{N} (0, 1)
$$

依分布成立。

定理 A1.2 中的中心极限定理表明，标准化后的样本均值在极限意义下接近标准正态随机变量。

为方便起见，定理 A1.1 和 A1.2 假设了独立同分布（IID）随机变量。对于独立随机变量的样本均值，也存在许多大数定律和中心极限定理（例如，Durrett, 2019）。

### A1.1.3 Delta 方法（Delta method）

Delta 方法是推导渐近正态随机向量的非线性函数渐近正态性的有力工具。下面我回顾 Delta 方法的一个特例。

**定理 A1.3（Delta 方法）** 假设 ${ \sqrt { n } } ( X _ { n } - \mu ) \to \mathrm { N } ( 0 , \Sigma )$ 依分布成立，且函数 g(x) 在 $\mu$ 处具有非零导数 $\nabla g ( \mu )$ 。则

$$
\sqrt {n} \{g (X _ {n}) - g (\mu) \} \rightarrow \mathrm{N} \left(0, (\nabla g (\mu) ^ {\mathsf {T}} \Sigma \nabla g (\mu)\right)
$$

依分布成立。

我将省略定理 A1.3 的证明。基于一阶泰勒展开，该定理是直观的：

$$
g (X _ {n}) - g (\mu) \cong (\nabla g (\mu) ^ {\mathsf {T}} (X _ {n} - \mu).
$$

Delta 方法的一个主要应用示例是获得比值的渐近正态性。

**例 A1.1（比值的渐近正态性）** 假设

$$
\sqrt {n} \binom {Y _ {n} - \mu_ {Y}} {X _ {n} - \mu_ {X}} \rightarrow \mathrm{N} \left(\binom {0} {0}, \left(\begin{array}{c c}\sigma_ {Y} ^ {2}&\sigma_ {Y X}\\\sigma_ {Y X}&\sigma_ {X} ^ {2}\end{array}\right)\right) \tag {A1.1}
$$

依分布成立，且 $\mu _ { X } \neq 0$ 。应用定理 A1.3 可得

$$
\sqrt {n} \left(\frac {Y _ {n}}{X _ {n}} - \frac {\mu_ {Y}}{\mu_ {X}}\right)\rightarrow \mathrm{N} \left(0, \frac {\sigma_ {Y} ^ {2}}{\mu_ {X} ^ {2}} + \frac {\mu_ {Y} ^ {2} \sigma_ {X} ^ {2}}{\mu_ {X} ^ {4}} - \frac {2 \mu_ {Y} \sigma_ {Y X}}{\mu_ {X} ^ {3}}\right) \tag {A1.2}
$$

依分布成立。在 $X _ { n }$ 和 $Y _ { n }$ 渐近独立的特殊情况下，$Y _ { n } / X _ { n }$ 的渐近方差简化为 $\sigma _ { Y } ^ { 2 } / \mu _ { X } ^ { 2 } + \mu _ { Y } ^ { 2 } \sigma _ { X } ^ { 2 } / \mu _ { X } ^ { 4 }$ 。详细推导留给问题 A1.2。

例 A1.1 中的渐近方差有些繁琐。基于以下近似，有一种更简便的记忆方法：

$$
\frac {Y _ {n}}{X _ {n}} - \frac {\mu_ {Y}}{\mu_ {X}} = \frac {Y _ {n} - \mu_ {Y} / \mu_ {X} \cdot X _ {n}}{X _ {n}} \cong \frac {Y _ {n} - \mu_ {Y} / \mu_ {X} \cdot X _ {n}}{\mu_ {X}}, \tag {A1.3}
$$

因此，比值的渐近方差等于

$$
\frac {Y _ {n} - \mu_ {Y} / \mu_ {X} \cdot X _ {n}}{\mu_ {X}}
$$

的渐近方差，这是 $Y _ { n }$ 和 $X _ { n }$ 的线性组合。Slutsky 定理可以使 (A1.3) 中的近似变得严格；但这超出了本书的范围。

**例 A1.2（乘积的渐近正态性）** 假设 (A1.1) 成立。应用定理 A1.3 可得

$$
\sqrt {n} \left(X _ {n} Y _ {n} - \mu_ {X} \mu_ {Y}\right)\rightarrow \mathrm{N} \left(0, \mu_ {Y} ^ {2} \sigma_ {X} ^ {2} + \mu_ {X} ^ {2} \sigma_ {Y} ^ {2} + 2 \mu_ {X} \mu_ {Y} \sigma_ {X Y}\right) \tag {A1.4}
$$

依分布成立。在 $X _ { n }$ 和 $Y _ { n }$ 渐近独立的特殊情况下，$X _ { n } Y _ { n }$ 的渐近方差简化为 $\mu _ { Y } ^ { 2 } \sigma _ { X } ^ { 2 } + \mu _ { X } ^ { 2 } \sigma _ { Y } ^ { 2 }$ 。详细推导留给问题 A1.3。

## A1.2 统计推断（A1.2 Statistical inference）

### A1.2.1 点估计（Point estimation）

假设 $\theta$ 是感兴趣的参数。通常，问题还包含其他不感兴趣的参数，记为 $\eta$ 。统计学家称 $\eta$ 为**冗余参数（nuisance parameter）**。基于数据，我们可以计算出一个估计量 $\hat{\theta}$ 。在本书中，我们采取**频率学派（frequentist）**的视角，假设 $\theta$ 是一个固定数值，而 $\hat { \theta }$ 由于数据的随机性而是随机的。估计量的两个基本要求如下。

**定义 A1.3（无偏性）** 如果对于 $\theta$ 和 $\eta$ 的所有可能取值，有

$$
E (\hat {\theta}) = \theta
$$

则称估计量 $\hat { \theta }$ 是 $\theta$ 的无偏估计量。

**定义 A1.4（相合性）** 如果对于 $\theta$ 和 $\eta$ 的所有可能取值，当样本量趋近无穷时，有

$$
\hat {\theta} \rightarrow \theta
$$

依概率成立，则称估计量 $\hat { \theta }$ 是 $\theta$ 的相合估计量。

无偏性要求估计量的均值与感兴趣的参数相同。相合性要求估计量在极限意义下接近真实参数。无偏性不蕴含相合性，相合性也不蕴含无偏性。无偏性可能具有限制性，因为即使在某些简单的统计问题中，无偏性也可能无法实现。相合性通常是大多数统计问题中的基本要求。

### A1.2.2 置信区间（Confidence interval）

点估计量 $\hat { \theta }$ 是一个与真实参数不同的随机变量。统计学家通常感兴趣的是找到一个以给定概率覆盖真实参数的区间。该区间基于数据计算，并且是随机的。

**定义 A1.5（置信区间）** 如果

$$
\operatorname{pr} (\hat {\theta} _ {\mathrm{L}} \leq \theta \leq \hat {\theta} _ {\mathrm{U}}) \geq 1 - \alpha
$$

则称依赖于数据的区间 $[ \hat { \theta } _ { \mathrm { L } } , \hat { \theta } _ { \mathrm { U } } ]$ 是 $\theta$ 的一个覆盖概率为 $1 - \alpha$ 的置信区间。

**定义 A1.6（渐近置信区间）** 如果当 $n \to \infty$ 时，有

$$
\mathrm{pr} (\hat {\theta} _ {\mathrm{L}} \leq \theta \leq \hat {\theta} _ {\mathrm{U}}) \rightarrow 1 - \alpha^ {\prime}
$$

且 $\alpha ^ { \prime } \geq \alpha$ ，则称依赖于数据的区间 $[ \hat { \theta } _ { \mathrm { L } } , \hat { \theta } _ { \mathrm { U } } ]$ 是 $\theta$ 的一个覆盖概率为 $1 - \alpha$ 的渐近置信区间。

标准选择是 $\alpha = 0 . 0 5$ 。在定义 A1.5 和 A1.6 中，覆盖概率可以大于名义水平 $1 - \alpha$ 。也就是说，这些定义允许过覆盖，但不允许欠覆盖。当存在过覆盖时，我们说置信区间是**保守的（conservative）**。当然，我们希望置信区间尽可能窄。否则，置信区间的定义可能是任意的。

### A1.2.3 假设检验（Hypothesis testing）

许多应用问题可以表述为检验一个假设：

$$
H _ {0}: \theta = 0.
$$

决策规则 $\phi$ 是数据的二元函数：如果我们拒绝 $H _ { 0 }$ ，则 $\phi = 1$ ；如果我们未能拒绝 $H _ { 0 }$ ，则 $\phi = 0$ 。检验的**第一类错误率（type one error rate）**是在原假设成立时拒绝的概率。下面我回顾该定义。

**定义 A1.7** 当 $H _ { 0 }$ 成立时，将检验 $\phi$ 的第一类错误率定义为概率

$$
\operatorname{pr} (\phi = 1)
$$

的最大可能值。

标准选择是确保第一类错误率低于 $\alpha = 0 . 0 5$ 。检验的**第二类错误率（type two error rate）**是在原假设不成立时未拒绝的概率。下面我回顾该定义。

**定义 A1.8** 当 $H _ { 0 }$ 不成立时，将检验 $\phi$ 的第二类错误率定义为概率

$$
\operatorname{pr} (\phi = 0).
$$

在控制了 $H _ { 0 }$ 下的第一类错误率的前提下，我们希望当 $H _ { 0 }$ 不成立时，第二类错误率尽可能低。

### A1.2.4 Wald 型置信区间与检验（Wald-type confidence interval and test）

许多统计问题具有以下结构。感兴趣的参数是 $\theta$ 。我们首先找到一个相合估计量 $\hat{\theta}$ ，它依概率收敛于 $\theta$ ，并且证明它是渐近正态的，均值为 $\theta$ ，方差为 $v$ （ $v$ 可能依赖于 $\theta$ 以及其他参数）。然后，基于解析公式或第 A1.5 章回顾的**自助法（bootstrap）**，我们找到 $v$ 的相合估计量 $\hat{v}$ 。最后，我们将 $\theta$ 的 Wald 型置信区间构造为

$$
\hat {\theta} \pm z _ {1 - \alpha / 2} \sqrt {\hat {v}}
$$

该区间以大约 $1 - \alpha$ 的概率覆盖 $\theta$ 。当该区间排除某个特定值 $c$ （例如 $c = 0$ ）时，我们拒绝原假设 $H _ { 0 } ( c ) : \theta = c$ ，这被称为 **Wald 检验（Wald test）**。

### A1.2.5 构造置信集与检验原假设之间的对偶性（Duality between constructing confidence sets and testing null hypotheses）

考虑标量参数 $\theta$ 的统计推断问题。统计学中的一个基本结果是，构造 $\theta$ 的置信集等价于检验关于 $\theta$ 的原假设。这通常被称为构造置信集与检验原假设之间的对偶性。

第 A1.2.4 节基于 Wald 型置信区间和检验回顾了这种对偶性。这种对偶性在一般情况下也成立。假设 $\hat{\Theta}$ 是 $\theta$ 的一个 $( 1 - \alpha )$ 水平的置信集：

$$
\operatorname{pr} (\theta \in \hat {\Theta}) = 1 - \alpha .
$$

那么，如果 c 不在集合 $\hat { \Theta }$ 中，我们可以拒绝原假设 $H _ { 0 } ( c ) : \theta = c$ 。这是一个有效的检验，因为当 $\theta$ 确实等于 c 时，我们有正确的第一类错误率 $\operatorname { p r } ( \theta \not \in { \hat { \Theta } } ) = \alpha$ 。反之，如果我们检验一系列原假设 $H _ { 0 } ( c ) : \theta = c$ ，我们可以得到相应的 p 值 $p ( c )$ ，它是 c 的函数。那么，我们在水平 $\alpha$ 下未能拒绝的那些 c 值构成了 $\theta$ 的一个置信集：

$$
\hat {\Theta} = \{c: p (c) \geq \alpha \} = \{c: \text {   在水平 } \alpha \text { 下未能拒绝 } H _ {0} (c) \}.
$$

这是一个有效的置信集，因为

$$
\operatorname{pr} (\theta \in \hat {\Theta}) = \operatorname{pr} \{\text {   在水平 } \alpha \text { 下未能拒绝 } H _ {0} (\theta) \} = 1 - \alpha .
$$

这里我使用"置信集"而非"置信区间"，因为基于反转检验得到的 $\hat { \Theta }$ 可能不是一个区间。关于这种对偶性的应用，请参见第 A1.4.2 节和第 3.6.1 节。

## A1.3 $2 \times 2$ 表格的推断（A1.3 Inference with $2 \times 2$ tables）

## A1.3.1 费希尔精确检验（Fisher’s exact test）

费希尔在以下统计模型下提出了针对 $H _ { 0 } : p _ { 1 } = p _ { 0 }$ 的精确检验：

$$
n _ {1 1} \sim \operatorname{Binomial} (n _ {1}, p _ {1}), \quad n _ {0 1} \sim \operatorname{Binomial} (n _ {0}, p _ {0}), \quad n _ {1 1} \perp n _ {0 1}.
$$

下表总结了这些数据。

<table><tr><td></td><td>1</td><td>0</td><td>行和</td></tr><tr><td>样本 1</td><td> $n_{11}$ </td><td> $n_{10}$ </td><td> $n_1$ </td></tr><tr><td>样本 0</td><td> $n_{01}$ </td><td> $n_{00}$ </td><td> $n_0$ </td></tr><tr><td>列和</td><td> $n_{.1}$ </td><td> $n_{.0}$ </td><td> $n$ </td></tr></table>

他认为，总和 $n _ { 1 1 } + n _ { 0 1 } \equiv n _ { \cdot 1 }$ 包含的关于 $p _ { 1 }$ 和 $p _ { 0 }$ 之间差异的信息很少，并且条件于该总和的 $n _ { 1 1 }$ 服从超几何分布（Hypergeometric distribution），该分布在 $H _ { 0 } { \mathrm { : } }$ 下不依赖于未知参数 $p _ { 1 } = p _ { 0 }$：

$$
\operatorname{pr} (n _ {1 1} = k) = \frac {\binom {n . _ {1}} {k} \binom {n - n . _ {1}} {n _ {1} - k}}{\binom {n} {n _ {1}}}.
$$

在 R 语言中，函数 `fisher.test` 实现了此检验。

## A1.3.2 $2 \times 2$ 表格的估计

基于第 A1.3.1 节中的模型，我们可以通过样本频率来估计参数 $p _ { 1 }$ 和 $p _ { 0 }$：

$$
\hat {p} _ {1} = \frac {n _ {1 1}}{n _ {1}}, \quad \hat {p} _ {0} = \frac {n _ {0 1}}{n _ {0}}.
$$

因此，我们可以通过样本对应量来估计**风险差（risk difference）**、**对数风险比（log risk ratio）**和**对数优势比（log odds ratio）**

$$
\begin{array}{l} \mathrm{RD} = p _ {1} - p _ {0}, \\ \log \mathrm{RR} = \log \frac {p _ {1}}{p _ {0}}, \\ \log \mathrm{OR} = \log \frac {p _ {1} / (1 - p _ {1})}{p _ {0} / (1 - p _ {0})} \\ \end{array}
$$

：

$$
\begin{array}{l} \hat {\mathrm{RD}} = \hat {p} _ {1} - \hat {p} _ {0}, \\ \log \hat {\mathrm{R} \mathrm{R}} = \log \frac {\hat {p} _ {1}}{\hat {p} _ {0}}, \\ \log \hat {\mathrm{OR}} = \log \frac {\hat {p} _ {1} / (1 - \hat {p} _ {1})}{\hat {p} _ {0} / (1 - \hat {p} _ {0})} = \log \frac {n _ {1 1} n _ {0 0}}{n _ {1 0} n _ {0 1}}. \\ \end{array}
$$

基于渐近近似（见问题 A1.4），上述参数的估计方差分别为

$$
\begin{array}{l} \frac {\hat {p} _ {1} (1 - \hat {p} _ {1})}{n _ {1}} + \frac {\hat {p} _ {0} (1 - \hat {p} _ {0})}{n _ {0}}, \\ \frac {1 - \hat {p} _ {1}}{n _ {1} \hat {p} _ {1}} + \frac {1 - \hat {p} _ {0}}{n _ {0} \hat {p} _ {0}}, \\ \frac {1}{n _ {1} \hat {p} _ {1} (1 - \hat {p} _ {1})} + \frac {1}{n _ {0} \hat {p} _ {0} (1 - \hat {p} _ {0})}, \\ \end{array}
$$

。上述对数变换能获得更好的正态近似，因为风险比和优势比始终为正。

## A1.4 统计学中的两个著名问题

## A1.4.1 贝伦斯-费希尔问题（Behrens–Fisher problem）

考虑一个两样本问题，分别有 $n _ { 1 }$ 个单元接受处理， $n _ { 0 }$ 个单元作为对照。假设处理组的结果 $\{ Y _ { i } : Z _ { i } = 1 \}$ 独立同分布于 $\mathrm { N } ( \mu _ { 1 } , \sigma _ { 1 } ^ { 2 } )$ ，对照组的结果 $\{ Y _ { i } : Z _ { i } = 0 \}$ 独立同分布于 $\mathrm { N } ( \mu _ { 0 } , \sigma _ { 0 } ^ { 2 } )$ 。目标是检验 $H _ { 0 } : \mu _ { 1 } = \mu _ { 0 }$ 。

从较简单的情形 $\sigma _ { 1 } ^ { 2 } = \sigma _ { 0 } ^ { 2 }$ 开始。与第 3 章一致，令 $\hat { \bar { Y } } ( 1 )$ 和 $\hat { \bar { Y } } ( 0 )$ 分别表示处理组和对照组结果的样本均值。一个标准结果是：

$$
t _ {\mathrm{equal}} = \frac {\hat {\bar {Y}} (1) - \hat {\bar {Y}} (0)}{\sqrt {\frac {n}{n _ {1} n _ {0} (n - 2)} \left[ \sum_ {Z _ {i} = 1} \{Y _ {i} - \hat {\bar {Y}} (1) \} ^ {2} + \sum_ {Z _ {i} = 0} \{Y _ {i} - \hat {\bar {Y}} (0) \} ^ {2} \right]}} \sim t _ {n - 2}.
$$

基于 $t _ { \mathrm { e q u a l } }$ ，我们可以构造一个对 $H _ { 0 }$ 的检验。

现在考虑更困难的情形，即 $\sigma _ { 1 } ^ { 2 }$ 和 $\sigma _ { 0 } ^ { 2 }$ 可能不同。 $t _ { \mathrm { e q u a l } }$ 的分布不再是 $t _ { n - 2 }$ 。分别估计方差，我们还可以定义

$$
t _ {\mathrm{unequal}} = \frac {\hat {\bar {Y}} (1) - \hat {\bar {Y}} (0)}{\sqrt {\frac {\hat {S} ^ {2} (1)}{n _ {1}} + \frac {\hat {S} ^ {2} (0)}{n _ {0}}}},
$$

其中

$$
\hat {S} ^ {2} (1) = (n _ {1} - 1) ^ {- 1} \sum_ {Z _ {i} = 1} \{Y _ {i} - \hat {\bar {Y}} (1) \} ^ {2}, \quad \hat {S} ^ {2} (0) = (n _ {0} - 1) ^ {- 1} \sum_ {Z _ {i} = 0} \{Y _ {i} - \hat {\bar {Y}} (0) \} ^ {2}
$$

分别是处理组和对照组结果的样本方差。不幸的是， $t _ { \mathrm { u n e q u a l } }$ 的精确分布依赖于已知的方差。在不假设方差相等的情况下检验 $H _ { 0 }$ 就是著名的**贝伦斯-费希尔问题**。当样本量 $n _ { 1 }$ 和 $n _ { 0 }$ 较大时，中心极限定理保证 $t _ { \mathrm { u n e q u a l } }$ 近似服从 $\mathrm { { N } } ( 0 , 1 )$ 。因此，我们可以构造对 $H _ { 0 }$ 的近似检验。

## A1.4.2 菲勒-克里西问题（Fieller–Creasy problem）

考虑一个两样本问题，分别有 $n _ { 1 }$ 个单元接受处理， $n _ { 0 }$ 个单元作为对照。假设处理组的结果 $\{ Y _ { i } : Z _ { i } = 1 \}$ 独立同分布于 $\mathrm { { N } } ( \mu _ { 1 } , 1 )$ ，对照组的结果 $\{ Y _ { i } : Z _ { i } = 0 \}$ 独立同分布于 $\mathrm { { N } } ( \mu _ { 0 } , 1 )$ 。目标是估计 $\gamma = \mu _ { 1 } / \mu _ { 0 }$ 。我们可以用 $\hat { \gamma } = \hat { \bar { Y } } ( 1 ) / \hat { \bar { Y } } ( 0 )$ 来估计 $\gamma$ 。但点估计量的分布很复杂，无法得到一个简单的程序来构造 $\gamma$ 的置信区间。

菲勒置信区间（Fieller’s confidence interval）可以表述为对一系列原假设进行检验的反演： $H _ { 0 } ( c ) : \gamma = c$ 。在 $H _ { 0 } ( c )$ 下，我们有

$$
\frac {\hat {\bar {Y}} (1) - c \hat {\bar {Y}} (0)}{\sqrt {1 / n _ {1} + c ^ {2} / n _ {0}}} \sim \mathrm{N} (0, 1)
$$

这引出了置信区间

$$
\left\{c: \left| \frac {\hat {\bar {Y}} (1) - c \hat {\bar {Y}} (0)}{\sqrt {1 / n _ {1} + c ^ {2} / n _ {0}}} \right| \leq z _ {\alpha} \right\}
$$

其中 $z _ { \alpha }$ 是标准正态随机变量的上 $1 - \alpha / 2$ 分位数。

## A1.5 自助法（Bootstrap）

为复杂估计量推导方差公式通常非常繁琐。Efron (1979) 提出了**自助法**作为方差估计的通用工具。自助法有许多版本 (Davison and Hinkley, 1997)。在本书中，我们只需要最基本的一种：**非参数自助法（nonparametric bootstrap）**，本书中简称为自助法。

考虑一般设定：

$$
Y _ {1}, \ldots , Y _ {n} \stackrel {\mathrm{IID}} {\sim} Y,
$$

其中 $Y _ { i }$ 可以是一个通用的随机元素，表示单元 i 的观测数据。估计量 $\hat { \theta }$ 是观测数据的函数： ${ \hat { \theta } } = T ( Y _ { 1 } , \ldots , Y _ { n } )$ 。当 $T$ 是一个复杂函数时，可能不容易获得 ${ \hat { \theta } }$ 的方差或渐近方差。

$\hat { \theta }$ 的不确定性源于从真实分布中对 $Y _ { 1 } , \dots , Y _ { n }$ 的 IID 抽样。虽然真实分布未知，但当样本量 n 很大时，它可以通过其经验版本很好地近似：

$$
\hat {F} _ {n} (y) = n ^ {- 1} \sum_ {i = 1} ^ {n} I (Y _ {i} \leq y),
$$

如果我们相信这个近似，我们可以通过从 $\hat {F} _ {n} (y)$ 中抽样来模拟 $\hat { \theta }$：

$$
(Y _ {1} ^ {*}, \dots , Y _ {n} ^ {*}) \stackrel {\mathrm{IID}} {\sim} \hat {F} _ {n} (y).
$$

因为 $\hat { F } _ { n } ( y )$ 是一个离散分布，在每个观测数据点上的质量为 $1 / n$ ，所以对 $\hat { \theta }$ 的模拟简化为以下步骤：

1. 从 $\{ Y _ { 1 } , \ldots , Y _ { n } \}$ 中有放回地抽取 $( Y _ { 1 } ^ { * } , \ldots , Y _ { n } ^ { * } )$ ；  
2. 计算 $\hat { \theta } ^ { * } = T ( Y _ { 1 } ^ { * } , \ldots , Y _ { n } ^ { * } )$ ；  
3. 将上述两步重复 B 次，得到自助法复制样本 $\{ \hat { \theta } _ { 1 } ^ { * } , \dots , \hat { \theta } _ { B } ^ { * } \}$ 。

然后，我们可以用自助法复制样本的样本方差来近似 $\hat { \theta }$ 的（渐近）方差：

$$
\hat {V} _ {\mathrm{boot}} = (B - 1) ^ {- 1} \sum_ {b = 1} ^ {B} (\hat {\theta} _ {b} ^ {*} - \bar {\theta} ^ {*}) ^ {2},
$$

$\bar { \theta } ^ { * } \ = \ B ^ { - 1 } \sum _ { b = 1 } ^ { B } \hat { \theta } _ { b } ^ { * }$ 那么正态近似为

$$
\hat {\theta} \pm z _ {1 - \alpha / 2} \sqrt {\hat {V} _ {\mathrm{boot}}},
$$

其中 $z _ { 1 - \alpha / 2 }$ 是 $\mathrm { { N } } ( 0 , 1 )$ 的上 $1 - \alpha / 2$ 分位数。

## A1.6 作业问题

## A1.1 独立但非同分布（Independent but not IID）数据

假设 $X _ { i } { } ^ { \ ' } \mathrm { s }$ 是独立的，其均值为 $\mu _ { i }$ ，方差为 $\sigma _ { i } ^ { 2 }$ ，其中 $i = 1 , \ldots , n$ 。 $\mu = n ^ { - 1 } \dot { \sum _ { i = 1 } ^ { n } \mu _ { i } }$ $\hat { \mu } =$ $n ^ { - 1 } \sum _ { i = 1 } ^ { n } X _ { i }$ 是 $\mu$ 的无偏估计量，并求其方差。证明对于 IID 数据的常用方差估计量

$$
\hat {v} = \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (X _ {i} - \hat {\mu}) ^ {2}
$$

是 $\hat { \mu }$ 方差的一个保守估计量，即

$$
E (\hat {v}) - \operatorname{var} (\hat {\mu}) = \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\mu_ {i} - \mu) ^ {2} \geq 0.
$$

注：考虑一个更简单的情形，其中对所有 $i =$ $1 , \ldots , n$ 有 $\mu _ { i } = \mu$ 和 $\sigma _ { i } ^ { 2 } = \sigma ^ { 2 }$ 。样本均值是 $\mu$ 的无偏估计量，方差为 $\sigma ^ { 2 } / n$ 。此外，方差 $\sigma ^ { 2 } / n$ 的一个无偏估计量是 $\hat { \sigma } ^ { 2 } / n = \hat { v }$ ，其中 $\hat { \sigma } ^ { 2 } = ( n -$ $\textstyle 1 ) ^ { - 1 } \sum _ { i = 1 } ^ { n } ( X _ { i } - { \hat { \mu } } ) ^ { 2 }$ 。

## A1.2 比值的渐近正态性（Asymptotic Normality of ratio）

证明 (A1.2)。

## A1.3 乘积的渐近正态性（Asymptotic Normality of product）

证明 (A1.4)。

## A1.4  $2 \times 2$ 表格中的方差估计量

使用 delta 方法推导第 A1.3.2 节中的方差估计量。
# 桥接有限总体与超总体因果推断（Bridging Finite and Super Population Causal Inference）

我们一直聚焦于**随机化实验中的有限总体（finite population）视角**。该视角将所有潜在结果视为固定数值，或者如果它们是某些随机变量的实现，则将其视为条件。这一视角的优势在于，它关注实验的设计，并对结果的数据生成过程要求最少的假设。然而，它经常被批评为仅具有**内部效度（internal validity）**，而不一定具有**外部效度（external validity）**。显然，所有实验者不仅关心他们实验的内部效度，也关心外部效度。由于所有统计性质都是以我们拥有的单元（units）的潜在结果为条件的，因此结果仅关乎被观测的单元。于是，一个自然的问题出现了：有限总体的结果能否推广到一个更大的总体？

这是对以潜在结果为条件的有限总体框架的一个合理批评。然而，这可能是一个哲学问题。我们观察到的是一个有限总体，因此任何实验设计和分析都直接为我们提供关于这个有限总体的信息。**随机化（Randomization）**仅确保在给定这些单元的潜在结果下的内部效度。结果的外部效度取决于单元的抽样过程。如果该有限总体是我们所感兴趣的更大总体的一个代表性样本，那么实验结果当然也具有外部效度。否则，基于随机化推断的结果可能无法推广。Pearl 和 Bareinboim (2014) 从一个不同的角度讨论了这种**可迁移性问题（transportability problem）**。

对于某些统计学家来说，这只是一个技术问题。我们可以改变统计框架，假设单元是从一个**超总体（super population）**中抽样得到的。那么所有的陈述都是关于我们感兴趣的总体。这是一个方便的框架，尽管它并没有真正解决上述问题。下面，我将介绍这个框架，目的有二：第一，它为随机化实验提供了一个不同的视角；第二，它作为本书第二部分和第三部分之间的桥梁。后一个目的更为重要，因为超总体框架使我们能够为**观察性研究（observational studies）**（其中处理并非随机分配）推导出更富有成效的结果。

## 9.1 完全随机化实验（CRE）

假设

$$
\{Z _ {i}, Y _ {i} (1), Y _ {i} (0), X _ {i} \} _ {i = 1} ^ {n} \stackrel {{\text {IID}}} {{\sim}} \{Z, Y (1), Y (0), X \}
$$

来自一个超总体。稍微滥用一下符号，我们将**总体平均因果效应（population average causal effect）**定义为

$$
\tau = E \{Y (1) - Y (0) \} = E \{Y (1) \} - E \{Y (0) \}.
$$

在超总体框架下，我们可以如下表述完全随机化实验（CRE）。

## 定义 9.1 (超总体框架下的完全随机化实验（CRE under the super population framework）) $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) , X \}$

在定义 9.1 下，平均因果效应可以写为

$$
\begin{array}{l} \tau = E \{Y (1) \mid Z = 1 \} - E \{Y (0) \mid Z = 0 \} \\ = E (Y \mid Z = 1) - E (Y \mid Z = 0), \tag {9.1} \\ \end{array}
$$

这等于结果期望之差。由于 $\tau$ 可以表示为可观测变量分布的函数，因此它是**非参数可识别的（nonparametrically identifiable）**。识别公式 (9.1) 立即提示了一个**矩估计量（moment estimator）** $\hat{\tau}$，即之前定义的结果均值之差。以 $\mathbf{Z}$ 为条件，这就成了一个比较两个独立样本均值的标准双样本问题。我们有

$$
E (\hat {\tau} \mid \mathbf {Z}) = \tau , \quad \mathrm{var} (\hat {\tau} \mid \mathbf {Z}) = \frac {\mathrm{var} \{Y (1) \}}{n _ {1}} + \frac {\mathrm{var} \{Y (0) \}}{n _ {0}}.
$$

在独立同分布（IID）抽样下，样本方差是总体方差的无偏估计，因此 **Neyman (1923) 的方差估计量**对于 $\mathrm { v a r } ( \hat { \tau } \mid Z )$ 是无偏的。在超总体框架下，**保守性问题（conservativeness problem）**消失了。

我们也可以讨论**协变量调整（covariate adjustment）**。基于**普通最小二乘法（OLS）**分解（见第 A2 章）

$$
Y (1) = \gamma_ {1} + \beta_ {1} ^ {\mathsf {T}} X + \varepsilon (1), \tag {9.2}
$$

$$
Y (0) = \gamma_ {0} + \beta_ {0} ^ {\mathsf {T}} X + \varepsilon (0), \tag {9.3}
$$

我们有

$$
\tau = E \{Y (1) - Y (0) \} = \gamma_ {1} - \gamma_ {0} + (\beta_ {1} - \beta_ {0}) ^ {\mathsf {T}} E (X),
$$

这是因为由于包含了截距项，残差 $\varepsilon ( 1 )$ 和 $\varepsilon ( 0 )$ 的均值为零。我们可以分别使用处理组和对照组数据的 OLS 估计 (9.2) 和 (9.3) 中的系数。系数的样本版本为 $\hat { \gamma } _ { 1 } , \hat { \beta } _ { 1 } , \hat { \gamma } _ { 0 } , \hat { \beta } _ { 0 }$，因此 $\tau$ 的一个协变量调整估计量为

$$
\hat {\tau} _ {\mathrm{adj}} = \hat {\gamma} _ {1} - \hat {\gamma} _ {0} + (\hat {\beta} _ {1} - \hat {\beta} _ {0}) ^ {\mathsf {T}} \bar {X}.
$$

如果我们用 $\bar { X } = 0$ 中心化协变量，上述估计量简化为 **Lin (2013) 的估计量**

$$
\hat {\tau} _ {\mathrm{L}} = \hat {\gamma} _ {1} - \hat {\gamma} _ {0},
$$

这等于在包含处理-协变量交互项的合并回归中，$Z$ 的系数。

不幸的是，由于超总体框架下存在额外的 $\bar{X}$ 的不确定性，**EHW 方差估计量（EHW variance estimator）**不适用于 $\hat { \tau } _ { \mathrm { L } }$。Berk 等人 (2013)、Negi 和 Wooldridge (2021) 以及 Zhao 和 Ding (2021a) 提出通过对 EHW 方差估计量增加一个额外项来进行修正

$$
(\hat {\beta} _ {1} - \hat {\beta} _ {0}) ^ {\mathsf {T}} S _ {X} ^ {2} (\hat {\beta} _ {1} - \hat {\beta} _ {0}) / n.
$$

一个概念上更简单但计算量较大的方法是使用**自助法（bootstrap）**来估计方差；见第 A1.5 章。

## 9.2 分层随机化实验（SRE）

我们可以将第 9.1 节的讨论扩展到**分层随机化实验（SRE）**，因为它等价于在层内进行的独立完全随机化实验（CREs）。下面的符号将与第 5 章略有不同。

假设

$$
\{Z _ {i}, Y _ {i} (1), Y _ {i} (0), X _ {i} \} \stackrel {{\text {IID}}} {{\sim}} \{Z, Y (1), Y (0), X \}.
$$

对于一个离散协变量 $X _ { i } \in \{ 1 , \ldots , K \}$，我们可以如下表述分层随机化实验（SRE）。

定义 9.2 (超总体框架下的分层随机化实验（SRE under the super population framework）) $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$.

在定义 9.2 下，**条件平均因果效应（conditional average causal effect）**可以重写为

$$
\tau_ {[ k ]} = E \{Y (1) - Y (0) \mid X = k \} = E (Y \mid Z = 1, X = k) - E (Y \mid Z = 0, X = k),
$$

因此平均因果效应可以重写为

$$
\tau = E \{Y (1) - Y (0) \} = \sum_ {k = 1} ^ {K} \operatorname{pr} (X _ {-} k) E \{Y (1) - Y (0) \mid X = k \} = \sum_ {k = 1} ^ {K} \operatorname{pr} (X = k) \tau_ {[ k ]}.
$$

第 9.1 节的讨论适用于所有层，因此我们可以为分层随机化实验（SRE）推导出超总体类比。当每个层内有超过两个处理组和对照组单元时，我们可以使用 $\hat { V } _ { \mathrm { S } }$ 作为 $\mathrm{var}(\hat{\tau}_S)$ 的无偏方差估计量。

## 9.3 家庭作业（Homework Problems）

## 9.1 完全随机化实验（CRE）下观测结果的普通最小二乘法（OLS）分解（OLS decomposition of the observed outcome under the CRE）

基于 (9.2) 和 (9.3)，证明观测结果在处理、协变量及其交互项上的 OLS 分解为

$$
Y = \alpha_ {0} + \alpha_ {Z} Z + \alpha_ {X} ^ {\mathsf {T}} X + \alpha_ {Z X} ^ {\mathsf {T}} X Z + \varepsilon
$$

其中

$$
\alpha_ {0} = \gamma_ {0}, \quad \alpha_ {Z} = \gamma_ {1} - \gamma_ {0}, \quad \alpha_ {X} = \beta_ {0}, \quad \alpha_ {Z X} = \beta_ {1} - \beta_ {0}, \quad \varepsilon = Z \varepsilon (1) + (1 - Z) \varepsilon (0).
$$

也就是说，

$$
(\alpha_ {0}, \alpha_ {Z}, \alpha_ {X}, \alpha_ {Z X}) = \arg \min _ {a _ {0}, a _ {Z}, a _ {X}, a _ {Z X}} E (Y - a _ {0} - a _ {Z} Z - a _ {X} ^ {\mathsf {T}} X - a _ {Z X} ^ {\mathsf {T}} X Z) ^ {2}.
$$

## 9.2 推荐阅读（Recommended reading）

Ding 等人 (2017a) 对平均因果效应的有限总体和超总体推断提供了统一的讨论。

## 第三部分（Part III）

## 观察性研究（Observational studies）

## 10
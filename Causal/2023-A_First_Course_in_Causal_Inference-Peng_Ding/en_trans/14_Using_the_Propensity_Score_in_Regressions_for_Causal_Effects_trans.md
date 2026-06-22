# 使用倾向得分进行因果效应的回归分析（Using the Propensity Score in Regressions for Causal Effects）

自 Rosenbaum 和 Rubin (1983b) 的开创性论文以来，文献中出现了许多**倾向得分（propensity score）**的创新性应用（例如 Bang 和 Robins, 2005; Robins 等人, 2007; Van der Laan 和 Rose, 2011; Vansteelandt 和 Daniel, 2014）。本章讨论两种使用倾向得分的简单方法：**将倾向得分作为协变量纳入回归**，以及**运行以倾向得分逆概率加权的回归**。我选择聚焦这两种方法，原因如下：

1. 它们易于实施，仅涉及标准统计软件包进行回归分析；
2. 它们的性质与许多更复杂的方法相当；
3. 它们可以轻松扩展，以允许包括**机器学习算法（machine learning algorithms）**在内的灵活统计模型。

## 14.1 以倾向得分作为协变量的回归（Regressions with the propensity score as a covariate）

根据定理 11.1，如果**无混淆性（unconfoundedness）**在给定 $X$ 的条件下成立，那么它在给定 $e(X)$ 的条件下也成立：

$$
Z \bot \{Y (1), Y (0) \} \mid e (X).
$$

类似于 (10.5)，$\tau$ 也可以通过以下方式非参数地识别：

$$
\tau = E \Big [ E \{Y \mid Z = 1, e (X) \} - E \{Y \mid Z = 0, e (X) \} \Big ],
$$

这激发了基于 $Y$ 对 $Z$ 和 $e ( X )$ 进行回归的方法。

最简单的回归设定是 $Y$ 对 $\{ 1 , Z , e ( X ) \}$ 的**普通最小二乘法（Ordinary Least Squares, OLS）**拟合，以 $Z$ 的系数作为估计量，记为 $\tau _ { e }$。为简单起见，我将讨论总体 OLS：

$$
\arg \min _ {a, b, c} E \{Y - a - b Z - c e (X) \} ^ {2}
$$

其中 $\tau _ { e }$ 定义为 $Z$ 的系数。如果我们有正确的倾向得分模型，且结果模型确实在 $Z$ 和 $e ( X )$ 上是线性的，则该估计量对 $\tau$ 是一致的。更有趣的结果是，即使结果模型完全被错误设定，只要我们拥有正确的倾向得分模型，$\tau _ { e }$ 也能估计 $\tau _ { \mathrm { O } }$。

**定理 14.1** 如果 $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$，则 $Y$ 对 $\{ 1 , Z , e ( X ) \}$ 的 OLS 拟合中 $Z$ 的系数等于

$$
\tau_ {e} = \tau_ {\mathrm{O}} = \frac {E \{h _ {\mathrm{O}} (X) \tau (X) \}}{E \{h _ {\mathrm{O}} (X) \}},
$$

回顾 $h _ { \mathrm { O } } ( X ) = e ( X ) \{ 1 - e ( X ) \}$ 且 $\tau ( X ) = E \{ Y ( 1 ) - Y ( 0 ) \mid X \}$。

定理 14.1 的一个不寻常特征是**重叠条件（overlap condition）**不再需要。即使某些单元的倾向得分 $e ( X )$ 等于 0 或 1，它们对应的权重 $e ( X ) \{ 1 - e ( X ) \}$ 为零，因此它们对最终参数 $\tau _ { \mathrm { O } }$ 没有任何贡献。

**定理 14.1 的证明**：基于附录 A2.3 中回顾的 **FWL 定理（Frisch-Waugh-Lovell theorem）**，我们可以通过两步得到 $\tau _ { e }$：首先，从 $Z$ 对 $\{ 1 , e ( X ) \}$ 的 OLS 拟合中获得残差 $\tilde { Z }$；然后，从 $Y$ 对 $\tilde { Z }$ 的 OLS 拟合中获得 $\tau _ { e }$。

$Z$ 对 $\{ 1 , e ( X ) \}$ 的 OLS 拟合中 $e ( X )$ 的系数为

$$
\begin{array}{l} \frac {\operatorname{cov} \{Z , e (X) \}}{\operatorname{var} \{e (X) \}} = \frac {E [ \operatorname{cov} \{Z , e (X) \mid X \} ] + \operatorname{cov} \{E (Z \mid X) , e (X) \}}{\operatorname{var} \{e (X) \}} \\ = \frac {0 + \operatorname{var} \{e (X) \}}{\operatorname{var} \{e (X) \}} = 1, \\ \end{array}
$$

因此截距为 $E ( Z ) - E \{ e ( X ) \} = 0$，残差为 $\tilde { Z } = Z - e ( X )$。这是合理的，因为 $Z - e ( X )$ 与 $X$ 的任何函数都不相关。

因此，我们可以从 $Y$ 对中心化变量 $Z - e ( X )$ 的单变量 OLS 拟合中得到 $\tau _ { e }$：

$$
\tau_ {e} = \frac {\operatorname{cov} \{Z - e (X) , Y \}}{\operatorname{var} \{Z - e (X) \}}.
$$

分母简化为

$$
\begin{array}{l} \operatorname{var} \{Z - e (X) \} = E \{Z - e (X) \} ^ {2} \\ = E \{Z + e (X) ^ {2} - 2 Z e (X) \} \\ = e (X) + e (X) ^ {2} - 2 e (X) ^ {2} = h _ {0} (X). \\ \end{array}
$$

分子简化为

$$
\begin{array}{l} \operatorname{cov} \{Z - e (X), Y \} \\ = E [ \{Z - e (X) \} Y ] \\ = E [ \{Z - e (X) \} Z Y (1) ] + E [ \{Z - e (X) \} (1 - Z) Y (0) ] \\ (\text { 由于 } Y = Z Y (1) + (1 - Z) Y (0)) \\ = E [ \{Z - Z e (X) \} Y (1) ] - E [ e (X) (1 - Z) Y (0) ] \\ = E [ Z \{1 - e (X) \} Y (1) ] - E [ e (X) (1 - Z) Y (0) ] \\ = E [ e (X) \{1 - e (X) \} \mu_ {1} (X) ] - E [ e (X) \{1 - e (X) \} \mu_ {0} (X) ] \\ (\text { 塔性质（tower property）和可忽略性（ignorability）}) \\ = E \{h _ {0} (X) \tau (X) \}. \\ \end{array}
$$

结论得证。

从定理 14.1 的证明中，我们可以直接运行 $Y$ 对中心化处理变量 $\tilde { Z } = Z - e ( X )$ 的 OLS 回归。Lee (2018) 提出了这一方法。此外，我们还可以在 OLS 拟合中纳入 $X$，这可能在有限样本中提高效率。然而，这不会改变估计目标，其仍然是 $\tau _ { \mathrm { O } }$。我将这两个结果总结在下面的推论中。

**推论 14.1** 如果 $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$，则

(1) $Y$ 对 $Z - e ( X )$ 或 $\{ 1 , Z - e ( X ) \}$ 的 OLS 拟合中 $Z - e ( X )$ 的系数等于 $\tau _ { \mathrm { O } }$；  
(2) $Y$ 对 $\{ 1 , Z , e ( X ) , X \}$ 的 OLS 拟合中 $Z$ 的系数等于 $\tau _ { \mathrm { O } }$。

**推论 14.1 的证明**：(1) 第一个结果是定理 14.1 证明中的一个中间步骤。第二个结果成立是因为 $Y$ 对 $Z - e ( X )$ 或 $\{ 1 , Z - e ( X ) \}$ 的回归不会改变 $Z - e ( X )$ 的系数，因为其均值为零。

(2) 这源于以下事实：

$$
Z - e (X) = Z - 0 - 1 \cdot e (X) - 0 ^ {\mathsf {T}} X
$$

是 $Z$ 对 $\{ 1 , e ( X ) , X \}$ 的 OLS 拟合的残差，因为 $Z - e ( X )$ 与 $X$ 的任何函数都不相关。

定理 14.1 为 $\tau _ { \mathrm { O } }$ 提供了一个两步估计方法：首先，拟合倾向得分模型以得到 $\hat { e } ( X _ { i } )$；其次，运行 $Y _ { i }$ 对 $( 1 , X _ { i } , \hat { e } ( X _ { i } ) )$ 的 OLS 回归，以获得 $Z _ { i }$ 的系数。推论 14.1 为 $\tau _ { \mathrm { O } }$ 提供了另一个两步估计方法：首先，拟合倾向得分模型以得到 $\hat { e } ( X _ { i } )$；其次，运行 $Y _ { i }$ 对 $Z _ { i } - \hat { e } ( X _ { i } )$ 的 OLS 回归，以获得 $Z _ { i }$ 的系数。虽然 OLS 便于获得点估计量，但由于第一步倾向得分估计中存在不确定性，相应的标准误是不正确的。我们可以使用**自助法（bootstrap）**来近似标准误。

Robins 等人 (1992) 讨论了许多基于倾向得分的 OLS 估计量。上述结果似乎是他们一般理论的特例，尽管他们没有指出与重叠权重下估计目标的联系，这一联系后来由 Li 等人 (2018a) 重新提出。Lee (2018) 从不同的角度提出了将 $Y$ 对 $Z - e ( X )$ 进行回归的方法，但没有将其与 Robins 等人 (1992) 和 Li 等人 (2018a) 的现有结果建立联系。

Rosenbaum 和 Rubin (1983b) 提出基于 $Y$ 对 $\{ 1 , Z , e ( X ) , Z e ( X ) \}$ 的 OLS 拟合来估计**平均因果效应（average causal effect）**。当这个结果模型正确时，他们的估计量对平均因果效应是一致的。然而，当模型不正确时，相应的估计量具有更复杂的解释。Little 和 An (2004) 建议基于 $Y$ 对 $Z$ 和 $e(X)$ 的灵活函数的 OLS 构建估计量，并证明其具有某种**双重稳健性（doubly robustness）**。由于实施上的复杂性，我在此省略讨论。

## 14.2 以倾向得分逆概率加权的回归（Regressions weighted by the inverse of the propensity score）

## 14.2.1 平均因果效应（Average causal effect）

我们首先重新审视 $\tau$ 的 Hajek 估计量：

$$
\hat {\tau} ^ {\mathrm{hajek}} = \frac {\sum_ {i = 1} ^ {n} \frac {Z _ {i} Y _ {i}}{\hat {e} (X _ {i})}}{\sum_ {i = 1} ^ {n} \frac {Z _ {i}}{\hat {e} (X _ {i})}} - \frac {\sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})}}{\sum_ {i = 1} ^ {n} \frac {1 - Z _ {i}}{1 - \hat {e} (X _ {i})}},
$$

该估计量等于处理组和对照组结果变量加权均值之差。数值上，它等同于对 $Y _ { i }$ 关于 $(1, Z_i)$ 进行以下**加权最小二乘法（Weighted Least Squares, WLS）** 回归时 $Z _ { i }$ 的系数。

**命题 14.1** $\hat {\tau} ^ {\mathrm{hajek}}$ 等于以下 WLS 回归中的 $\hat { \beta }$：

$$
(\hat {\alpha}, \hat {\beta}) = \arg \min _ {\alpha , \beta} \sum_ {i = 1} ^ {n} w _ {i} (Y _ {i} - \alpha - \beta Z _ {i}) ^ {2}
$$

其权重为

$$
w _ {i} = \frac {Z _ {i}}{\hat {e} (X _ {i})} + \frac {1 - Z _ {i}}{1 - \hat {e} (X _ {i})} = \left\{ \begin{array}{l l} \frac {1}{\hat {e} (X _ {i})} & \text {   if   } Z _ {i} = 1; \\ \frac {1}{1 - \hat {e} (X _ {i})} & \text {   if   } Z _ {i} = 0. \end{array} \right. \tag {14.1}
$$

Imbens (2004) 指出了命题 14.1 中的结果。我将其留作问题 14.1。根据命题 14.1，基于 WLS 来获得 $\hat {\tau} ^ {\mathrm{hajek}}$ 是方便的。然而，由于**估计的倾向得分（estimated propensity score）** 存在不确定性，WLS 报告的标准误对于 $\hat { \tau } ^ { \mathrm { h a j e k } }$ 的真实标准误是不正确的。**自助法（Bootstrap）** 为近似真实标准误提供了一种便捷的方法。

为什么 WLS 能给出 $\tau$ 的一致估计量？回想一下，在**完全随机实验（Completely Randomized Experiment, CRE）** 中，当倾向得分为常数时，我们可以简单地使用 $Y _ { i }$ 关于 $\left( 1 , Z _ { i } \right)$ 的 OLS 拟合中 $Z _ { i }$ 的系数来估计 $\tau$。在**观察性研究（observational studies）** 中，个体接受处理和对照的概率不同。如果我们对处理组个体赋予权重 $1 / e ( X _ { i } )$，对对照组个体赋予权重 $1 / \{ 1 - e ( X _ { i } ) \}$，那么它们可以代表整个总体，我们实际上得到了一个**伪随机实验（pseudo randomized experiment）**。因此，加权均值之差对于 $\tau$ 是一致的。$\hat { \tau } ^ { \mathrm { h a j e k } }$ 与 WLS 的数值等价性不仅是一个有趣的数值事实，而且对于启发包含协变量调整的更复杂估计量也很有用。下面我给出一个扩展。

回想一下，在 CRE 中，我们可以使用 $Y _ { i }$ 关于 $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ 的 OLS 拟合中 $Z _ { i }$ 的系数来估计 $\tau$，其中协变量以 $\bar { X } = 0$ 为中心。这是 Lin (2013) 的估计量，它利用协变量来提高效率。对于观察性研究的一个自然扩展是，使用 $Y _ { i }$ 关于 $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ 的 WLS 拟合中 $Z _ { i }$ 的系数来估计 $\tau$，其中权重由 (14.1) 定义。Hirano 和 Imbens (2001) 在一个应用中使用过这个估计量。完全交互的线性模型等价于为处理组和对照组分别建立两个独立的线性模型。如果线性模型

$$
E (Y \mid Z = 1, X) = \beta_ {1 0} + \beta_ {1 x} ^ {\mathsf {T}} X, E (Y \mid Z = 0, X) = \beta_ {0 0} + \beta_ {0 x} ^ {\mathsf {T}} X,
$$

被正确设定，那么 OLS 和 WLS 都能给出系数的一致估计量，并且 $Z$ 的系数的估计量对于 $\tau$ 是一致的。更有趣的是，如果倾向得分模型正确而结果模型错误，基于 WLS 的 $Z$ 的系数的估计量对于 $\tau$ 仍然是一致的。也就是说，基于 WLS 的估计量是**双重稳健的（doubly robust）**。Robins 等人 (2007) 讨论了这一性质并将其归功于 M. Joffe 未发表的论文。我将在下面给出更多细节。

令 $\hat { e } ( X _ { i } )$ 为拟合的倾向得分，$( \mu _ { 1 } ( X _ { i } , \hat { \beta } _ { 1 } ) , \mu _ { 0 } ( X _ { i } , \hat { \beta } _ { 0 } ) )$ 为基于 WLS 的结果均值拟合值。**结果回归估计量（outcome regression estimator）** 为

$$
\hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{reg}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \mu_ {1} (X _ {i}, \hat {\beta} _ {1}) - \frac {1}{n} \sum_ {i = 1} ^ {n} \mu_ {0} (X _ {i}, \hat {\beta} _ {0})
$$

而 $\tau$ 的**双重稳健估计量（doubly robust estimator）** 为

$$
\hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{dr}} = \hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{reg}} + \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {Z _ {i} \{Y _ {i} - \mu_ {1} (X _ {i} , \hat {\beta} _ {1}) \}}{\hat {e} (X _ {i})} - \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) \{Y _ {i} - \mu_ {0} (X _ {i} , \hat {\beta} _ {0}) \}}{1 - \hat {e} (X _ {i})}.
$$

一个有趣的结果是，如果我们使用权重 (14.1)，这个双重稳健估计量等于结果回归估计量，而后者又简化为 $Y _ { i }$ 关于 $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ 的 WLS 拟合中 $Z _ { i }$ 的系数。

**定理 14.2** 如果 $\bar { X } = 0$ 并且 $( \mu _ { 1 } ( X _ { i } , \hat { \beta } _ { 1 } ) , \mu _ { 0 } ( X _ { i } , \hat { \beta } _ { 0 } ) ) = ( \hat { \beta } _ { 1 0 } + \hat { \beta } _ { 1 x } ^ { \top } X _ { i } , \hat { \beta } _ { 0 0 } + \hat { \beta } _ { 0 x } ^ { \mathsf { T } } X _ { i } )$ 是基于 $Y _ { i }$ 关于 $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ 且权重为 (14.1) 的 WLS 拟合得到的，那么

$$
\hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{dr}} = \hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{reg}} = \hat {\beta} _ {1 0} - \hat {\beta} _ {0 0},
$$

这正是 WLS 拟合中 $Z _ { i }$ 的系数。

**定理 14.2 的证明：** $Y _ { i }$ 关于 $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ 的 WLS 拟合等价于基于处理组和对照组数据分别进行的两次 WLS 拟合。两次 WLS 拟合都包含截距项，因此一阶条件必须满足

$$
\sum_ {i = 1} ^ {n} \frac {Z _ {i} (Y _ {i} - \hat {\beta} _ {1 0} - \hat {\beta} _ {1 x} ^ {\intercal} X _ {i})}{\hat {e} (X _ {i})} = 0
$$

和

$$
\sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) (Y _ {i} - \hat {\beta} _ {0 0} - \hat {\beta} _ {0 x} ^ {\mathsf {T}} X _ {i})}{1 - \hat {e} (X _ {i})} = 0.
$$

所以 ${ \hat { \tau } } ^ { \mathrm { d r } }$ 和 $\hat { \tau } ^ { \mathrm { r e g } }$ 之差恰好为零。两者都简化为

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} (\hat {\beta} _ {1 0} + \hat {\beta} _ {1 x} ^ {\mathsf {T}} X _ {i}) - \frac {1}{n} \sum_ {i = 1} ^ {n} (\hat {\beta} _ {0 0} + \hat {\beta} _ {0 x} ^ {\mathsf {T}} X _ {i}) = \hat {\beta} _ {1 0} - \hat {\beta} _ {0 0} + (\hat {\beta} _ {1 x} - \hat {\beta} _ {0 x}) ^ {\mathsf {T}} \bar {X} = \hat {\beta} _ {1 0} - \hat {\beta} _ {0 0}
$$

其中协变量已中心化。因此，它们都等于 $Y _ { i }$ 关于 $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ 的 WLS 拟合中 $Z _ { i }$ 的系数。□

Freedman 和 Berk (2008) 基于一些模拟研究不鼓励使用上述 WLS 估计量。他们表明，当结果模型正确时，WLS 估计量比 OLS 估计量更差，因为在他们的同方差结果设定下，WLS 估计量具有较大的变异性。这总体上可能不成立。当误差的方差与倾向得分的倒数成比例时，WLS 估计量将比 OLS 估计量更有效。他们还表明，基于 WLS 拟合的估计标准误对于真实标准误不是一致的，因为它忽略了估计的倾向得分的不确定性。这可以通过使用自助法来近似 WLS 估计量的方差而轻松解决。尽管如此，他们发现“在某些情况下加权可能有所帮助”，因为当结果模型错误时，如果倾向得分模型正确，WLS 估计量仍然是一致的。

我以表 14.1 结束本节，该表总结了随机实验和观察性研究中因果效应的回归估计量。

## 14.2.2 处理组上的平均因果效应（Average causal effect on the treated units）

关于 $\tau _ { \mathrm { T } }$ 的结果与关于 $\tau$ 的结果类似。首先，$\tau _ { \mathrm { T } }$ 的 Hajek 估计量

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{hajek}} = \hat {\bar {Y}} (1) - \frac {\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) Y _ {i}}{\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i})},
$$

其中 $\hat { o } ( X _ { i } ) = \hat { e } ( X _ { i } ) / \{ 1 - \hat { e } ( X _ { i } ) \}$，等于在以下 $Y _ { i }$ 关于 $( 1 , Z _ { i } )$ 的 WLS 拟合中 $Z _ { i }$ 的系数。

**表 14.1：CRE 和无混杂观察性研究中的回归估计量。权重 $w _ { i }$ 定义于 (14.1)。**

<table><tr><td></td><td>CRE</td><td>无混杂的观察性研究</td></tr><tr><td>不含 X</td><td> $Y_i \sim Z_i$ </td><td> $Y_i \sim Z_i$ 且权重为 $w_i$ </td></tr><tr><td>含 X</td><td> $Y_i \sim (Z_i, X_i, Z_i X_i)$ </td><td> $Y_i \sim (Z_i, X_i, Z_i X_i)$ 且权重为 $w_i$ </td></tr></table>

**命题 14.2** $\hat { \tau } _ { \mathrm { T } } ^ { \mathrm {hajek} }$ 数值上等同于以下 WLS 中的 $\hat { \beta }$：

$$
(\hat {\alpha}, \hat {\beta}) = \arg \min _ {\alpha , \beta} \sum_ {i = 1} ^ {n} w _ {\mathrm{Ti}} (Y _ {i} - \alpha - \beta Z _ {i}) ^ {2}
$$

其权重为

$$
w _ {\mathrm{T} i} = Z _ {i} + (1 - Z _ {i}) \hat {o} (X _ {i}) = \left\{ \begin{array}{l l} 1 & \text {   if   } Z _ {i} = 1; \\ \hat {o} (X _ {i}) & \text {   if   } Z _ {i} = 0. \end{array} \right. \tag {14.2}
$$

与命题 14.1 类似，命题 14.2 是一个纯粹的线性代数结果。我将其证明留作问题 14.1。

其次，如果我们以 $\hat { \bar { X } } ( 1 ) = 0$ 对协变量进行中心化，那么我们可以使用 $Y _ { i }$ 关于 $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ 的 WLS 拟合中 $Z _ { i }$ 的系数来估计 $\tau _ { \mathrm { T } }$，其中权重由 (14.2) 定义。类似地，该估计量等于回归估计量

$$
\hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{reg}} = \hat {\bar {Y}} (1) - \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} \mu_ {0} (X _ {i}, \hat {\beta} _ {0}),
$$

它也等于双重稳健估计量

$$
\hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{dr}} = \hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{reg}} - \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) \{Y _ {i} - \mu_ {0} (X _ {i}, \hat {\beta} _ {0}) \}.
$$

**定理 14.3** 如果 $\hat { \bar { X } } ( 1 ) = 0$ 并且 $\mu _ { 0 } ( X _ { i } , \hat { \beta } _ { 0 } ) = \hat { \beta } _ { 0 0 } + \hat { \beta } _ { 0 x } ^ { \top } X _ { i }$ 是基于 $Y _ { i }$ 关于 $( 1 , Z _ { i } , X _ { i } , \bar { Z _ { i } } X _ { i } )$ 且权重为 (14.2) 的 WLS 拟合得到的，那么

$$
\hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{dr}} = \hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{reg}} = \hat {\beta} _ {1 0} - \hat {\beta} _ {0 0},
$$

这正是 WLS 拟合中 $Z _ { i }$ 的系数。

**定理 14.3 的证明：** 基于处理组和对照组的 WLS 拟合，我们有

$$
\sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\beta} _ {1 0} - \hat {\beta} _ {1 x} ^ {\intercal} X _ {i}) = 0, \tag {14.3}
$$

$$
\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) (Y _ {i} - \hat {\beta} _ {0 0} - \hat {\beta} _ {0 x} ^ {\intercal} X _ {i}) = 0. \tag {14.4}
$$

$\hat { \tau } _ { \mathrm { { T , w l s } } } ^ { \mathrm { { d r } } } = \hat { \tau } _ { \mathrm { { T , w l s } } } ^ { \mathrm { { r e g } } }$ 。两者都简化为

$$
\hat {\bar {Y}} (1) - \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} (\hat {\beta} _ {0 0} + \hat {\beta} _ {0 x} ^ {\mathsf {T}} X _ {i}) = \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\beta} _ {0 0} - \hat {\beta} _ {0 x} ^ {\mathsf {T}} X _ {i}).
$$

协变量以 $\hat { \bar { X } } ( 1 ) = 0$ 为中心。第一个结果 (14.3) 意味着 $\hat { \bar { Y } } ( 1 ) = \hat { \beta } _ { 1 0 }$，这进一步将估计量简化为 $\hat { \beta } _ { 1 0 } - \hat { \beta } _ { 0 0 }$。□

## 14.3 作业题（Homework problems）

## 14.1 Hajek 估计量作为加权最小二乘估计量（Hajek estimators as WLS estimators）

证明命题 14.1 和 14.2。

提示：这些问题属于问题 A2.2 中关于单变量加权最小二乘（WLS）的特例。

## 14.2 预测估计量与双重稳健估计量（Predictive estimator and doubly robust estimator）

另一种结果回归估计量是**预测估计量（predictive estimator）**：

$$
\hat {\tau} ^ {\mathrm{pred}} = \hat {\mu} _ {1} ^ {\mathrm{pred}} - \hat {\mu} _ {0} ^ {\mathrm{pred}}
$$

其中：

$$
\hat {\mu} _ {1} ^ {\mathrm{pred}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left\{Z _ {i} Y _ {i} + (1 - Z _ {i}) \mu_ {1} (X _ {i}, \hat {\beta} _ {1}) \right\}
$$

且：

$$
\hat {\mu} _ {0} ^ {\text { pred }} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left\{Z _ {i} \mu_ {0} (X _ {i}, \hat {\beta} _ {1}) + (1 - Z _ {i}) Y _ {i} \right\}.
$$

它与之前讨论的结果回归估计量的不同之处在于，它仅预测**反事实结果（counterfactual outcomes）**，而不预测观测到的结果。

证明：如果 $( \mu _ { 1 } ( X _ { i } , \hat { \beta } _ { 1 } ) , \mu _ { 0 } ( X _ { i } , \hat { \beta } _ { 1 } ) ) =$ $( \hat { \beta } _ { 1 0 } + \hat { \beta } _ { 1 x } ^ { \top } X _ { i } , \hat { \beta } _ { 0 0 } + \hat { \beta } _ { 0 x } ^ { \top } X _ { i } )$ 分别来自基于处理组和对照组数据对 $Y _ { i }$ 关于 $( 1 , X _ { i } )$ 的加权最小二乘（WLS）拟合，且权重为：

$$
w _ {i} = Z _ {i} / \hat {o} (X _ {i}) + (1 - Z _ {i}) \hat {o} (X _ {i}) = \left\{ \begin{array}{l l} \frac {1}{\hat {o} (X _ {i})} = \frac {1 - \hat {e} (X _ {i})}{\hat {e} (X _ {i})} & \text { 如果 } Z _ {i} = 1; \\ \hat {o} (X _ {i}) = \frac {\hat {e} (X _ {i})}{1 - \hat {e} (X _ {i})} & \text { 如果 } Z _ {i} = 0. \end{array} \right. \tag {14.5}
$$

那么**双重稳健估计量（doubly robust estimator）**等于 $\hat {\tau}^{\mathrm{pred}}$。

备注：Cao 等人（2009）以及 Vermeulen 和 Vansteelandt（2015）从其他更理论的角度对公式 (14.5) 中的权重进行了论证。

<!-- footnote -->

- 如果**逻辑斯蒂结果模型（logistic outcome model）**是正确的，那么 $\hat { \beta } _ { z }$ 估计的是给定协变量条件下处理对结果的**条件优势比（conditional odds ratio）**，该值不等于 $\tau$。Freedman（2008c）对在**条件随机实验（Conditional Randomized Experiments, CREs）**中使用逻辑斯蒂回归系数估计 $\tau$ 提出了警告。有关逻辑斯蒂回归的更多细节，请参见第 A2 章。

<!-- footnote end -->

<!-- footnote -->

- `glm` 函数比 `lm` 函数更通用。当设置 `out.family = gaussian` 时，`glm` 与 `lm` 相同。

<!-- footnote end -->

## 14.3 二元结果下的加权逻辑斯蒂回归（Weighted logistic regression with a binary outcome）

对于二元结果，我们可以将线性结果模型替换为**逻辑斯蒂结果模型（logistic outcome models）**。证明：在逻辑斯蒂回归中使用权重时，双重稳健估计量等于结果回归估计量。该结论对 $\tau$ 和 $\tau _ { \mathrm { T } }$ 均成立。

## 14.4 错误设定线性回归下的因果推断（Causal inference with a misspecified linear regression）

将 $Y$ 对 $Z$ 和 $X$ 的**总体普通最小二乘（population OLS）**定义为：

$$
(\beta_ {0}, \beta_ {1}, \beta_ {2}) = \arg \min _ {b _ {0}, b _ {1}, b _ {2}} E (Y - b _ {0} - b _ {1} Z - b _ {2} ^ {\mathsf {T}} X) ^ {2}.
$$

回忆 $e ( X ) = \mathrm { p r } ( Z = 1 \mid X )$ 是**倾向得分（propensity score）**，并定义 $\tilde { e } ( X ) =$ $\gamma _ { 0 } + \gamma _ { 1 } ^ { \intercal } X$ 为 $A$ 对 $X$ 的 OLS 投影，其中：

$$
(\gamma_ {0}, \gamma_ {1}) = \arg \min _ {c _ {0}, c _ {1}} E (A - c _ {0} - c _ {1} ^ {\mathsf {T}} X) ^ {2}.
$$

1. 证明：

$$
\beta_ {1} = \frac {E [ \tilde {w} (X) \{\mu_ {1} (X) - \mu_ {0} (X) \} ]}{E \{\tilde {w} (X) \}} + \frac {E [ \{e (X) - \tilde {e} (X) \} \mu_ {0} (X) ]}{E \{\tilde {w} (X) \}}
$$

其中 $\tilde { w } ( X ) = e ( X ) \{ 1 - \tilde { e } ( X ) \}$。

2. 当 $X$ 包含一个离散协变量的虚拟变量时，证明：

$$
\beta_ {1} = \frac {E [ w (X) \{\mu_ {1} (X) - \mu_ {0} (X) \} ]}{E \{w (X) \}}
$$

其中 $w ( X ) = e ( X ) \{ 1 - e ( X ) \}$ 是**重叠权重（overlap weight）**。

备注：Vansteelandt 和 Dukes（2022）给出了第一部分的公式，但未提供详细证明。第二部分的结论在文献中被多次推导（例如，Angrist, 1998；Ding, 2021）。

## 14.5 数据再分析（Data re-analysis）

重新分析 `karolinska.txt` 数据集以及 `ATE` 包中的 `nhanesbmi` 数据集。

## 14.6 推荐阅读（Recommended reading）

Kang 和 Schafer（2007）对**双重稳健估计量（doubly robust estimator）**进行了批判性评述，并通过模拟将其与许多其他估计量进行了比较。Robins 等人（2007）对 Kang 和 Schafer（2007）给出了非常深刻的评论。
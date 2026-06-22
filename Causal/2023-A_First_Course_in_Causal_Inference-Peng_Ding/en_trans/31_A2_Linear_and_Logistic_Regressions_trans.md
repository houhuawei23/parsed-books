# A2 线性回归与逻辑回归（Linear and Logistic Regressions）

## A2.1 总体普通最小二乘法（Population Ordinary Least Squares）

假设 $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { n } \stackrel { \mathrm { I I D } } { \sim } ( x , y )$ ，其中 $x$ 是一个 $p$ 维随机标量或向量，$y$ 是一个随机标量。下面我将使用 $( x , y )$ 表示一个一般观测值，为简便起见省略下标 $i$ 。定义**总体普通最小二乘法（Ordinary Least Squares, OLS）**系数为

$$
\beta = \arg \min _ {b} E \left\{(y - x ^ {\mathsf {T}} b) ^ {2} \right\}.
$$

目标函数关于 $b$ 是二次的，因此我们可以证明最小化器为

$$
\beta = \left\{E \left(x x ^ {\mathsf {T}}\right) \right\} ^ {- 1} E (x y)
$$

如果矩存在且 $E \left( x x ^ { \mathsf { T } } \right)$ 可逆。

利用 $\beta$ ，我们可以定义

$$
\varepsilon = y - x ^ {\mathsf {T}} \beta \tag {A2.1}
$$

为**总体残差（population residual）**。根据 $\beta$ 的定义，我们可以验证

$$
E (x \varepsilon) = E \left\{x (y - x ^ {\mathsf {T}} \beta) \right\} = E (x y) - E (x x ^ {\mathsf {T}}) \beta = 0.
$$

**示例 A2.1（含截距项的总体OLS）** 如果我们将 1 作为 $x$ 的一个分量，则

$$
E (\varepsilon) = E (y - x ^ {\mathsf {T}} \beta) = 0
$$

这进一步意味着 $\mathrm { c o v } ( x , \varepsilon ) = 0$ 。因此，当 $\beta$ 中包含截距项时，总体残差的均值必须为零，并且根据构造，它与其它协变量不相关。

**示例 A2.2（含截距项的单变量总体OLS）** 一个重要的特例是对于标量 $x$ 和 $y$ ，我们可以定义

$$
(\alpha , \beta) = \arg \min _ {a, b} E \{(y - a - b x) ^ {2} \}
$$

其具有显式公式

$$
\beta = \frac {\operatorname{cov} (x , y)}{\operatorname{var} (x)}, \quad \alpha = E (y) - \beta E (x).
$$

**示例 A2.3（不含截距项的单变量总体OLS）** 不含截距项时，我们可以定义

$$
\gamma = \arg \min _ {c} E \{(y - c x) ^ {2} \}
$$

其等于

$$
\gamma = \frac {E (x y)}{E (x ^ {2})}.
$$

当 $x$ 均值为零时，上述两个总体OLS中的 $\beta = \gamma$ 。

我们也可以将 (A2.1) 重写为

$$
y = x ^ {\mathsf {T}} \beta + \varepsilon , \tag {A2.2}
$$

这由总体OLS系数和残差的定义成立，无需任何建模假设。我们称 (A2.2) 为**总体OLS分解（population OLS decomposition）**。

## A2.2 样本OLS（Sample OLS）

对于 $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { n } \stackrel { \mathrm { I I D } } { \sim } ( x , y )$ ，总体OLS系数的样本对应为

$$
\hat {\beta} = \left(n ^ {- 1} \sum_ {i = 1} ^ {n} x _ {i} x _ {i} ^ {\top}\right) ^ {- 1} \left(n ^ {- 1} \sum_ {i = 1} ^ {n} x _ {i} y _ {i}\right),
$$

残差为 $\hat { \varepsilon } _ { i } = y _ { i } - x _ { i } ^ { \top } \hat { \beta }$ 。这称为**样本OLS**或简称为OLS。OLS系数 $\hat { \beta }$ 最小化**残差平方和（residual sum of squares）**

$$
\hat {\beta} = \arg \min _ {b} n ^ {- 1} \sum_ {i = 1} ^ {n} (y _ {i} - x _ {i} ^ {\mathsf {T}} b) ^ {2},
$$

其满足以下**正规方程（Normal equation）**：

$$
\sum_ {i = 1} ^ {n} x _ {i} (y _ {i} - x _ {i} ^ {\mathsf {T}} \hat {\beta}) = 0.
$$

**拟合值（fitted values）**等于

$$
\hat {y} _ {i} = x _ {i} ^ {\mathsf {T}} \hat {\beta} (i = 1, \dots , n).
$$

使用矩阵记号

$$
X = \left( \begin{array}{c} x _ {1} ^ {\mathsf {T}} \\ \vdots \\ x _ {n} ^ {\mathsf {T}} \end{array} \right), \quad Y = \left( \begin{array}{c} y _ {1} \\ \vdots \\ y _ {n} \end{array} \right),
$$

我们可以将OLS系数写为

$$
\hat {\beta} = (X ^ {\mathsf {T}} X) ^ {- 1} X ^ {\mathsf {T}} Y
$$

将拟合向量写为

$$
\hat {Y} = X \hat {\beta} = X (X ^ {\mathsf {T}} X) ^ {- 1} X ^ {\mathsf {T}} Y.
$$

定义**帽子矩阵（hat matrix）**为

$$
H = X (X ^ {\mathsf {T}} X) ^ {- 1} X ^ {\mathsf {T}}.
$$

于是我们有 $\hat { Y } = H Y$ ，这证明了"帽子矩阵"这一名称的合理性。

假设 $( x , y )$ 的四阶矩有限，我们可以使用**大数定律（law of large numbers）**和**中心极限定理（central limit theorem）**证明

$$
\sqrt {n} (\hat {\beta} - \beta) \rightarrow \mathrm{N} (0, V = B ^ {- 1} M B ^ {- 1})
$$

依分布收敛，其中 $\boldsymbol { B } = \boldsymbol { E } ( \boldsymbol { x } \boldsymbol { x } ^ { \intercal } )$ 且 $M = E ( \varepsilon ^ { 2 } x x ^ { \mathsf { T } } )$ 。因此，$\hat { \beta }$ 渐近方差的一个矩估计量为

$$
\hat {V} _ {\mathrm{EHW}} = n ^ {- 1} \left(n ^ {- 1} \sum_ {i = 1} ^ {n} x _ {i} x _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \left(n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\varepsilon} _ {i} ^ {2} x _ {i} x _ {i} ^ {\mathsf {T}}\right) \left(n ^ {- 1} \sum_ {i = 1} ^ {n} x _ {i} x _ {i} ^ {\mathsf {T}}\right) ^ {- 1},
$$

这称为**Eicker–Huber–White (EHW) 稳健协方差估计量**（Eicker, 1967; Huber, 1967; White, 1980）。我们可以证明 $n \hat { V } _ { \mathrm { E H W } } \to V$ 依概率收敛。基于 $\hat { \beta }$ 和 $\hat { V } _ { \mathrm { E H W } }$ ，我们可以对总体OLS系数 $\beta$ 进行推断。在 $\mathbb { R }$ 中，`lm` 函数可以计算 ${ \hat { \boldsymbol { \beta } } }$ ，`car` 包中的 `hccm` 函数可以计算 $\hat { V } _ { \mathrm { E H W } }$ 。

EHW稳健协方差估计量有许多变体（Long and Ervin, 2000）。特别地，HC1变体将 $\hat { \varepsilon } _ { i } ^ { 2 }$ 修改为 $\hat { \varepsilon } _ { i } ^ { 2 } / ( n - p )$ ，HC2变体将 $\hat { \varepsilon } _ { i } ^ { 2 }$ 修改为 $\hat { \varepsilon } _ { i } ^ { 2 } / ( 1 - h _ { i i } )$ ，HC3变体将 $\hat { \varepsilon } _ { i } ^ { 2 }$ 修改为 $\hat { \varepsilon } _ { i } ^ { 2 } / ( 1 - h _ { i i } ) ^ { 2 }$ ，在 $\hat { V } _ { \mathrm { E H W } }$ 的定义中，其中 $h _ { i i }$ 是 $H$ 的第 $(i, i)$ 个对角元素，也称为**杠杆值（leverage scores）**。

## A2.3 Frisch–Waugh–Lovell 定理（Frisch–Waugh–Lovell Theorem）

**Frisch–Waugh–Lovell (FWL) 定理**有两个版本：一个在总体层面，另一个在样本层面。它将多元OLS简化为单变量OLS，从而有助于理解和计算OLS系数。下面我将给出FWL定理的特殊情形，这足以满足本书的需要。

**定理 A2.1（总体FWL）** 在 $y$ 对 $( x _ { 1 } , x _ { 2 } , \ldots , x _ { p } )$ 的OLS拟合中，$x_1$ 的系数等于 $y$ 或 $\tilde { y }$ 对 $\tilde { x } _ { 1 }$ 的OLS拟合中 $\tilde { x } _ { 1 }$ 的系数，其中 $\tilde{y}$ 是 $y$ 对 $( x _ { 2 } , \ldots , x _ { p } )$ 的OLS拟合的残差，$\tilde { x } _ { 1 }$ 是 $x _ { 1 }$ 对 $( x _ { 2 } , \ldots , x _ { p } )$ 的OLS拟合的残差。

在定理 A2.1 中，对 $x _ { 1 }$ 进行残差化是关键，但对 $y$ 进行残差化则不是。

**定理 A2.2（样本FWL）** 对于包含列向量的数据 $( Y , X _ { 1 } , X _ { 2 } , \ldots , X _ { p } )$ ，$X _ { 1 }$ 的系数等于 $Y$ 或 $\tilde{Y}$ 对 $\tilde { X } _ { 1 }$ 的OLS拟合中 $\tilde { X } _ { 1 }$ 的系数，其中 $\tilde { Y }$ 是 $Y$ 对 $( X _ { 2 } , \ldots , X _ { p } )$ 的OLS拟合的残差向量，$\tilde { X } _ { 1 }$ 是 $X _ { 1 }$ 对 $( X _ { 2 } , \ldots , X _ { p } )$ 的OLS拟合的残差。

同样地，在定理 A2.2 中，对 $X _ { 1 }$ 进行残差化是关键，但对 $Y$ 进行残差化则不是。

## A2.4 线性模型（Linear model）

有时，我们施加一个更强的模型假设，要求给定 $x$ 时 $y$ 的条件均值是线性的：

$$
E (y \mid x) = x ^ {\mathsf {T}} \beta
$$

或等价地，

$$
y = x ^ {\mathsf {T}} \beta + \varepsilon , \qquad E (\varepsilon \mid x) = 0,
$$

这称为**受限均值模型（restricted mean model）**。在此模型下，总体OLS系数就是感兴趣的真正参数：

$$
\begin{array}{l} \left\{E (x x ^ {\mathsf {T}}) \right\} ^ {- 1} E (x y) = \left\{E (x x ^ {\mathsf {T}}) \right\} ^ {- 1} E \left\{x E (y \mid x) \right\} \\ = \left\{E (x x ^ {\mathsf {T}}) \right\} ^ {- 1} E (x x ^ {\mathsf {T}} \beta) \\ = \beta . \\ \end{array}
$$

此外，总体OLS系数不依赖于 $x$ 的分布。第 A2.1 节中的渐近推断也适用于此模型。

在 $\operatorname { v a r } ( \varepsilon \mid x ) = \sigma ^ { 2 }$ 的特殊情况下，OLS系数的渐近方差简化为

$$
V = \sigma^ {2} \{E (x x ^ {\mathsf {T}}) \} ^ {- 1}
$$

因此，$\hat { \beta }$ 渐近方差的一个更简单的矩估计量为

$$
\hat {V} _ {\mathrm{OLS}} = \hat {\sigma} ^ {2} \left(\sum_ {i = 1} ^ {n} x _ {i} x _ {i} ^ {\intercal}\right) ^ {- 1}
$$

$\begin{array} { r } { \hat { \sigma } ^ { 2 } = ( n - p ) ^ { - 1 } \sum _ { i = 1 } ^ { n } \hat { \varepsilon } _ { i } ^ { 2 } } \end{array}$ 由 `lm` 函数计算。

## A2.5 加权最小二乘法（Weighted Least Squares）

假设 $( w _ { i } , x _ { i } , y _ { i } ) \stackrel { \mathrm { I I D } } { \sim } ( w , x , y )$ 且 $w \ne 0$ 。在总体层面，我们可以定义**加权最小二乘法（Weighted Least Squares, WLS）**系数为

$$
\beta_ {w} = \arg \min _ {b} E \{w (y - x ^ {\mathsf {T}} b) ^ {2} \},
$$

其满足

$$
E \{w x (y - x ^ {\mathsf {T}} \beta_ {w}) \} = 0
$$

因此等于

$$
\beta_ {w} = \{E (w x x ^ {\mathsf {T}}) \} ^ {- 1} E (w x y)
$$

如果 $E ( w x x ^ { \mathsf { T } } )$ 可逆。

在样本层面，我们可以定义WLS系数为

$$
\hat {\beta} _ {w} = \arg \min _ {b} \sum_ {i = 1} ^ {n} w _ {i} (y _ {i} - x _ {i} ^ {\mathsf {T}} b) ^ {2},
$$

其满足

$$
\sum_ {i = 1} ^ {n} w _ {i} x _ {i} (y _ {i} - x _ {i} ^ {\mathsf {T}} \hat {\beta} _ {w}) = 0
$$

因此等于

$$
\hat {\beta} _ {w} = \left(n ^ {- 1} \sum_ {i = 1} ^ {n} w _ {i} x _ {i} x _ {i} ^ {\intercal}\right) ^ {- 1} \left(n ^ {- 1} \sum_ {i = 1} ^ {n} w _ {i} x _ {i} y _ {i}\right)
$$

如果 $\scriptstyle \sum _ { i = 1 } ^ { n } w_i x_i x_i^\top$ 可逆。

## A2.6 逻辑回归（Logistic Regression）

### A2.6.1 模型（Model）

从技术上讲，即使结果 $y$ 是二元的，我们也可以应用OLS过程。然而，预测概率超出 [0, 1] 范围会有些尴尬。这促使我们考虑以下模型：

$$
\operatorname{pr} (y _ {i} = 1 \mid x _ {i}) = g (x _ {i} ^ {\mathsf {T}} \beta),
$$

其中 $g ( \cdot ) : \mathbb { R } \to [ 0 , 1 ]$ 是一个单调函数，其逆函数通常称为**链接函数（link function）**。$g ( \cdot )$ 函数可以是任何随机变量的分布函数，但我们将重点放在逻辑形式：

$$
g (z) = \frac {e ^ {z}}{1 + e ^ {z}} = (1 + e ^ {- z}) ^ {- 1}.
$$

我们也可以将逻辑模型写为

$$
\operatorname{pr} (y _ {i} = 1 \mid x _ {i}) \equiv \pi (x _ {i}, \beta) = \frac {e ^ {x _ {i} ^ {\top} \beta}}{1 + e ^ {x _ {i} ^ {\top} \beta}},
$$

或等价地，

$$
\operatorname{logit} \left\{\operatorname{pr} (y _ {i} = 1 \mid x _ {i}) \right\} \equiv \log \frac {\operatorname{pr} (y _ {i} = 1 \mid x _ {i})}{1 - \operatorname{pr} (y _ {i} = 1 \mid x _ {i})} = x _ {i} ^ {\top} \beta .
$$

假设 $x _ { i 1 }$ 是二元的。在逻辑模型下，我们有

$$
\begin{array}{l} \beta_ {1} = \operatorname{logit} \left\{\operatorname{pr} (y _ {i} = 1 \mid x _ {i 1} = 1, \dots) \right\} - \operatorname{logit} \left\{\operatorname{pr} (y _ {i} = 1 \mid x _ {i 1} = 0, \dots) \right\} \\ = \log \frac {\operatorname{pr} (y _ {i} = 1 \mid x _ {i 1} = 1 , \ldots) / \operatorname{pr} (y _ {i} = 0 \mid x _ {i 1} = 1 , \ldots)}{\operatorname{pr} (y _ {i} = 1 \mid x _ {i 1} = 0 , \ldots) / \operatorname{pr} (y _ {i} = 0 \mid x _ {i 1} = 0 , \ldots)}, \\ \end{array}
$$

其中 $\cdot \cdot \cdot$ 包含所有其他回归变量 $x _ { i 2 } , \ldots , x _ { i p }$ 。因此，系数 $\beta _ { 1 }$ 等于在给定其他回归变量的条件下，$x _ { i 1 }$ 对 $y _ { i }$ 的**对数优势比（log odds ratio）**。

### A2.6.2 最大似然估计（Maximum Likelihood Estimate）

为了估计参数 $\beta$ ，我们可以最大化以下**似然函数（likelihood function）**：

$$
\begin{array}{l} L (\beta) = \prod_ {i = 1} ^ {n} \left\{\pi (x _ {i}, \beta) \right\} ^ {y _ {i}} \left\{1 - \pi (x _ {i}, \beta) \right\} ^ {1 - y _ {i}} \\ = \prod_ {i = 1} ^ {n} \left\{\frac {\pi (x _ {i} , \beta)}{1 - \pi (x _ {i} , \beta)} \right\} ^ {y _ {i}} \left\{1 - \pi (x _ {i}, \beta) \right\} \\ = \prod_ {i = 1} ^ {n} \left(e ^ {x _ {i} ^ {\intercal} \beta}\right) ^ {y _ {i}} \frac {1}{1 + e ^ {x _ {i} ^ {\intercal} \beta}} \\ = \prod_ {i = 1} ^ {n} \frac {e ^ {y _ {i} x _ {i} ^ {\top} \beta}}{1 + e ^ {x _ {i} ^ {\top} \beta}}. \\ \end{array}
$$

令 $\hat { \beta }$ 表示最大化器，称为**最大似然估计（Maximum Likelihood Estimate, MLE）**。对 $L ( \beta )$ 取对数并关于 $\beta$ 求导，我们可以证明MLE必须满足一阶条件：

$$
\sum_ {i = 1} ^ {n} x _ {i} \{y _ {i} - \pi (x _ {i}, \hat {\beta}) \} = 0.
$$

因此，如果 $x _ { i }$ 包含截距项，则MLE必须满足

$$
\sum_ {i = 1} ^ {n} \{y _ {i} - \pi (x _ {i}, \hat {\beta}) \} = 0,
$$

即，观测到的 $y_i$ 的平均值必须等于拟合概率 $\pi ( x _ { i } , \hat { \beta } )$ 的平均值。

使用MLE的一般理论，我们可以证明它对真实参数 $\beta$ 是一致的，并且是渐近正态的：

$$
\sqrt {n} (\hat {\beta} - \beta) \rightarrow \mathrm{N} (0, V)
$$

依分布收敛，其中 $V = \left[ E \left\{ \pi ( x _ { i } , \beta ) ( 1 - \pi ( x _ { i } , \beta ) ) x x ^ { \mathsf { T } } \right\} \right] ^ { - 1 }$ 。因此，我们可以用下式近似 $\hat { \beta }$ 的协方差矩阵：

$$
\left[ \sum_ {i = 1} ^ {n} \pi (x _ {i}, \hat {\beta}) \{1 - \pi (x _ {i}, \hat {\beta}) \} x _ {i} x _ {i} ^ {\mathsf {T}} \right] ^ {- 1}.
$$

在R中，`glm` 函数可以找到MLE并报告估计的协方差矩阵。

## A2.6.3 扩展至病例对照研究（Extension to the case-control study）

在**病例对照研究（case-control studies）**中，抽样是条件于**二分类结果（binary outcome）**的，即结果 $y_i = 1$ 和 $y_i = 0$ 的单元以不同的概率被抽样。令 $s_i$ 为**抽样指示变量（sampling indicator）**。在病例对照研究中，我们有：

$$
\operatorname{pr} (s_i = 1 \mid x_i, y_i) = \operatorname{pr} (s_i = 1 \mid y_i)
$$

作为 $y_i$ 的函数，并且我们仅观测到 $s_i = 1$ 的单元。

Prentice 和 Pyke (1979) 表明，**逻辑回归（logistic regression）**可应用于病例对照研究，尽管上述讨论假设了**独立同分布（IID）抽样**。

## A2.6.4 带权重的逻辑回归（Logistic regression with weights）

有时，单元 $i$ 具有权重 $w_i$，那么我们可以通过求解以下方程来拟合**加权逻辑回归（weighted logistic regression）**：

$$
\sum_{i=1}^{n} w_i x_i \{y_i - \pi (x_i, \hat{\beta}) \} = 0.
$$

## A2.7 课后习题（Homework problems）

## A2.1 含截距项的样本 OLS（Sample OLS with intercept）

假设**回归变量（regressor）** $x_i$ 包含一个**截距项（intercept）**。证明：

$$
\bar{y} = \bar{x}^{\mathsf{T}} \hat{\beta}. \tag{A2.3}
$$

## A2.2 单变量加权最小二乘法（Univariate weighed least squares）

作为**加权最小二乘法（Weighted Least Squares, WLS）**的一个特例，定义：

$$
(\hat{\alpha}_w, \hat{\beta}_w) = \arg \min_{(a, b)} \sum_{i=1}^{n} w_i (y_i - a - b x_i)^2
$$

其中 $w_i \geq 0$。证明：

$$
\hat{\beta}_w = \frac{\sum_{i=1}^{n} w_i (x_i - \bar{x}_w) (y_i - \bar{y}_w)}{\sum_{i=1}^{n} w_i (x_i - \bar{x}_w)^2} \tag{A2.4}
$$

以及

$$
\hat{\alpha}_w = \bar{y}_w - \hat{\beta}_w \bar{x}_w, \tag{A2.5}
$$

其中 $\bar{x}_w = \sum_{i=1}^{n} w_i x_i / \sum_{i=1}^{n} w_i$ 和 $\bar{y}_w = \sum_{i=1}^{n} w_i y_i / \sum_{i=1}^{n} w_i$ 分别是 $x_i$ 和 $y_i$ 的**加权平均值（weighted averages）**。

进一步假设 $x_i$ 是二值的。证明：

$$
\hat{\beta}_w = \frac{\sum_{i=1}^{n} w_i x_i y_i}{\sum_{i=1}^{n} w_i x_i} - \frac{\sum_{i=1}^{n} w_i (1 - x_i) y_i}{\sum_{i=1}^{n} w_i (1 - x_i)}.
$$

也就是说，如果单变量 WLS 中的回归变量是二值的，则该回归变量的系数等于**加权均值（weighted means）**之差。

提示：考虑对 WLS 问题进行适当的**重新参数化（reparametrization）**。否则，推导过程会非常繁琐。

## A2.3 正交回归变量的 OLS（OLS with orthogonal regressors）

考虑对 $n$ 维向量 $Y$ 在 $n \times p$ 矩阵 $X$ 上进行样本 OLS 拟合，其系数为 $\hat{\boldsymbol{\beta}}$。将 $X$ 划分为 $\boldsymbol{X} = (X_1, X_2)$，其中 $X_1$ 是一个 $n \times k$ 矩阵，$X_2$ 是一个 $n \times l$ 矩阵，且 $p = k + l$。相应地，将 $\hat{\beta}$ 划分为：

$$
\hat{\beta} = \binom{\hat{\beta}_1}{\hat{\beta}_2}.
$$

假设 $X_1$ 和 $X_2$ 是**正交的（orthogonal）**，即 $X_1^{\mathsf{T}} X_2 = 0$。证明 $\hat{\beta}_1$ 等于 $Y$ 对 $X_1$ 进行 OLS 得到的系数，而 $\hat{\beta}_2$ 分别等于 $Y$ 对 $X_2$ 进行 OLS 得到的系数。

## A2.4 回归变量非退化变换下的 OLS（OLS with a non-degenerate transformation of the regressors）

定义 $\hat{\beta}$ 为 $n$ 维向量 $Y$ 在 $n \times p$ 矩阵 $X$ 上进行样本 OLS 拟合得到的系数。令 $\Gamma$ 为一个 $p \times p$ 的**非退化矩阵（non-degenerate matrix）**，并定义 $X' = X \Gamma$。定义 $\hat{\beta}'$ 为 $Y$ 在 $X'$ 上进行样本 OLS 拟合得到的系数。

证明：

$$
\hat{\beta} = \Gamma \hat{\beta}'.
$$
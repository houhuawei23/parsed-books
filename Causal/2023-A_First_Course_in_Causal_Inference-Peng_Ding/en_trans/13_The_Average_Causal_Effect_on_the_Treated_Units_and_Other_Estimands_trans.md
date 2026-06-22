# 处理单元的平均因果效应及其他估计量（The Average Causal Effect on the Treated Units and Other Estimands）

第10–12章聚焦于在**无混杂性（unconfoundedness）**和**重叠性（overlap）**假设下，平均因果效应 $\tau = E \{ Y(1) - Y(0) \}$ 的识别与估计。从概念上讲，可以很自然地将讨论扩展到处理组和对照组的平均因果效应：

$$
\tau_ {\mathrm{T}} = E \{Y (1) - Y (0) \mid Z = 1 \},
$$

$$
\tau_ {\mathrm{C}} = E \{Y (1) - Y (0) \mid Z = 0 \}.
$$

由于对称性，本章主要关注 $\tau _ { \mathrm { T } }$ ，同时也扩展到其他估计量。

## 13.1 $\tau _ { \mathbf { T } }$ 的非参数识别（Nonparametric identification of $\tau _ { \mathbf { T } }$）

处理单元的平均因果效应等于

$$
\tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E \{Y (0) \mid Z = 1 \},
$$

其中第一项 $E ( Y \mid Z = 1 )$ 可直接从数据中识别，第二项 $E \{ Y ( 0 ) ~ | ~ Z = 1 \}$ 是反事实的。识别第二项的关键假设是以下无混杂性和重叠性假设。

**假设 13.1** $Z \underline { { \mathrm { 1 1 } } } Y ( 0 ) \mid X$ 且 $e ( X ) < 1$ 。

由于关键在于识别 $E \{ Y ( 0 ) \mid Z = 1 \}$ ，我们只需要"单向"的无混杂性和重叠性假设。在假设13.1下，对于 $\tau _ { \mathrm { T } }$ 有以下识别结果。

**定理 13.1** 在假设13.1下，我们有

$$
\begin{array}{l} E \{Y (0) \mid Z = 1 \} = E \left\{E (Y \mid Z = 0, X) \mid Z = 1 \right\} \\ = \int E (Y \mid Z = 0, X = x) F (\mathrm{d} x \mid Z = 1). \\ \end{array}
$$

16413 处理单元的平均因果效应及其他估计量

定理13.1表明 $\tau _ { \mathrm { T } }$ 可通过下式进行非参数识别

$$
\tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E \left\{E (Y \mid Z = 0, X) \mid Z = 1 \right\} \tag {13.1}
$$

**定理13.1的证明**：我们有

$$
\begin{array}{l} E \{Y (0) \mid Z = 1 \} = E \left[ E \{Y (0) \mid Z = 1, X \} \mid Z = 1 \right] \\ = E \left[ E \{Y (0) \mid Z = 0, X \} \mid Z = 1 \right] \\ = E \left\{E (Y \mid Z = 0, X) \mid Z = 1 \right\} \\ = \int E (Y \mid Z = 0, X = x) F (\mathrm{d} x \mid Z = 1). \\ \end{array}
$$

![image_11](images/image_11.png)

对于离散型 $X$ ，定理13.1中的识别公式简化为

$$
E \{Y (0) \mid Z = 1 \} = \sum_ {k = 1} ^ {K} E (Y \mid Z = 0, X = k) \mathrm{pr} (X = k \mid Z = 1),
$$

由此得到 $\tau _ { \mathrm { T } }$ 的以下分层估计量

$$
\hat {\tau} _ {\mathrm{T}} = \hat {\bar {Y}} (1) - \sum_ {k = 1} ^ {K} \hat {\pi} _ {[ k ] | 1} \hat {\bar {Y}} _ {[ k ]} (0),
$$

其中 $\hat { \pi } _ { [ k ] | 1 } = n _ { [ k ] 1 } / n _ { 1 }$ 是处理单元中 $X$ 属于类别 $k$ 的比例。

对于连续型 $X$ ，我们需要使用对照组单元拟合 $E ( Y \mid Z = 0 , X )$ 的结果模型。如果对照潜在结果的拟合值为 $\hat { \mu } _ { 0 } ( X _ { i } )$ ，则**结果回归估计量（outcome regression estimator）**为

$$
\hat {\tau} _ {\mathrm{T}} = \hat {\bar {Y}} (1) - n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \hat {\mu} _ {0} (X _ {i}) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \{Y _ {i} - \hat {\mu} _ {0} (X _ {i}) \}.
$$

**例 13.1** 如果对所有单元指定一个线性模型

$$
E (Y \mid Z, X) = \beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X,
$$

那么

$$
\begin{array}{l} \tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E (\beta_ {0} + \beta_ {x} ^ {\mathsf {T}} X \mid Z = 1) \\ = E (Y \mid Z = 1) - \beta_ {0} - \beta_ {x} ^ {\mathsf {T}} E (X \mid Z = 1). \\ \end{array}
$$

如果我们运行**普通最小二乘法（Ordinary Least Squares, OLS）**得到 $( \hat { \beta } _ { 0 } , \hat { \beta } _ { z } , \hat { \beta } _ { x } )$ ，则估计量为

$$
\hat {\tau} _ {\mathrm{T}} = \hat {\bar {Y}} (1) - \hat {\beta} _ {0} - \hat {\beta} _ {x} ^ {\mathsf {T}} \hat {\bar {X}} (1).
$$

利用OLS的性质（见A2.3），我们有

$$
\sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\beta} _ {0} - \hat {\beta} _ {z} Z _ {i} - \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}) = 0 \Longrightarrow \hat {\bar {Y}} (1) - \hat {\beta} _ {0} - \hat {\beta} _ {z} - \hat {\beta} _ {x} ^ {\mathsf {T}} \hat {\bar {X}} (1) = 0.
$$

因此，上述估计量简化为 $\hat { \tau } _ { \mathrm { T } } = \hat { \beta } _ { z }$ ，即 $Z$ 的OLS系数。

根据OLS的性质，我们也可以将 $\hat { \beta } _ { z }$ 写为调整后结果 $Y _ { i } - \hat { \beta } _ { x } ^ { \sf T } X _ { i }$ 的均值差，得到

$$
\begin{array}{l} \hat {\tau} _ {\mathrm{T}} = \left\{\hat {\bar {Y}} (1) - \hat {\beta} _ {x} ^ {\mathsf {T}} \hat {\bar {X}} (1) \right\} - \left\{\hat {\bar {Y}} (0) - \hat {\beta} _ {x} ^ {\mathsf {T}} \hat {\bar {X}} (0) \right\} \\ = \left\{\hat {\bar {Y}} (1) - \hat {\bar {Y}} (0) \right\} - \hat {\beta} _ {x} ^ {\mathsf {T}} \left\{\hat {\bar {X}} (1) - \hat {\bar {X}} (0) \right\}. \tag {13.2} \\ \end{array}
$$

因此， $\hat {\tau} _ {\mathrm{T}}$ 等于结果的简单均值差，再根据处理组和对照组中协变量的不平衡性进行调整。

第 $\it 1 0 . 4 . 2$ 节表明 $\hat { \beta } _ { z }$ 是 $\tau$ 的一个估计量，本例进一步表明 $\hat { \beta } _ { z }$ 也是 $\tau _ { \mathrm { T } }$ 的一个估计量。这并不令人惊讶，因为线性模型假设各单元的因果效应是恒定的。

**例 13.2** 识别公式仅依赖于 $E ( Y \mid Z = 0 , X )$ ，因此我们只需为对照组指定一个模型。当该模型为线性时，

$$
E (Y \mid Z = 0, X) = \beta_ {0 | 0} + \beta_ {x | 0} ^ {\mathsf {T}} X,
$$

我们有

$$
\begin{array}{l} \tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E (\beta_ {0 | 0} + \beta_ {x | 0} ^ {\mathsf {T}} X \mid Z = 1) \\ = E (Y \mid Z = 1) - \beta_ {0 | 0} - \beta_ {x | 0} ^ {\mathsf {T}} E (X \mid Z = 1). \\ \end{array}
$$

如果我们仅使用对照组单元运行OLS得到 $( \hat { \beta } _ { 0 | 0 } , \hat { \beta } _ { x | 0 } )$ ，则估计量为

$$
\hat {\tau} _ {\mathrm{T}} = \hat {\bar {Y}} (1) - \hat {\beta} _ {0 | 0} - \hat {\beta} _ {x | 0} ^ {\mathsf {T}} \hat {\bar {X}} (1).
$$

利用OLS的性质（见A2.3），我们有

$$
\hat {\bar {Y}} (0) = \hat {\beta} _ {0 | 0} + \hat {\beta} _ {x | 0} ^ {\mathsf {T}} \hat {\bar {X}} (0).
$$

因此，上述估计量简化为

$$
\hat {\tau} _ {\mathrm{T}} = \left\{\hat {\bar {Y}} (1) - \hat {\bar {Y}} (0) \right\} - \hat {\beta} _ {x | 0} ^ {\mathsf {T}} \left\{\hat {\bar {X}} (1) - \hat {\bar {X}} (0) \right\},
$$

这与(13.2)类似，但协变量均值差的系数不同。

作为一个代数事实，我们可以证明该估计量等于在以 $\hat { \bar { X } } ( 1 )$ 对协变量进行中心化后，将结果对处理变量、协变量及其交互项进行OLS拟合时 $Z$ 的系数。更多细节见问题13.1。

## 13.2 $\tau_{\mathbf{T}}$ 的逆倾向得分加权与双重稳健估计（Inverse propensity score weighting and doubly robust estimation of $\tau_{\mathbf{T}}$）

**定理 13.2** 在假设13.1下，我们有

$$
E \{Y (0) \mid Z = 1 \} = E \left\{\frac {e (X)}{e} \frac {1 - Z}{1 - e (X)} Y \right\} \tag {13.3}
$$

以及

$$
\tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E \left\{\frac {e (X)}{e} \frac {1 - Z}{1 - e (X)} Y \right\}, \tag {13.4}
$$

其中 $e = \operatorname { p r } ( Z = 1 )$ 是处理的边际概率。

**定理13.2的证明**：(13.3)的左边等于

$$
\begin{array}{l} E \{Y (0) \mid Z = 1 \} = E \{Z Y (0) \} / e \\ = E \left[ E (Z \mid X) E \{Y (0) \mid X \} \right] / e \\ = E \left[ e (X) E \{Y (0) \mid X \} \right] / e. \\ \end{array}
$$

(13.3)的右边等于

$$
\begin{array}{l} E \left\{\frac {e (X)}{e} \frac {1 - Z}{1 - e (X)} Y \right\} = E \left[ E \left\{\frac {e (X)}{e} \frac {1 - Z}{1 - e (X)} Y (0) \mid X \right\} \right] \\ { = } { E \left[ \frac { e ( X ) } { e \{ 1 - e ( X ) \} } E \left\{ ( 1 - Z ) Y ( 0 ) \mid X \right\} \right] } \\ { = } { E \left[ \frac { e ( X ) } { e \{ 1 - e ( X ) \} } E ( 1 - Z \mid X ) E \{ Y ( 0 ) \mid X \} \right] } \\ = E \left[ e (X) E \{Y (0) \mid X \} \right] / e. \\ \end{array}
$$

因此(13.3)成立。

我们有两种**逆倾向得分加权估计量（inverse propensity score weighting estimators）**

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{ht}} = \hat {\bar {Y}} (1) - n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) Y _ {i}
$$

和

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{hajek}} = \hat {\bar {Y}} (1) - \frac {\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) Y _ {i}}{\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i})},
$$

其中 $\hat { o } ( X _ { i } ) = \hat { e } ( X _ { i } ) / \{ 1 - \hat { e } ( X _ { i } ) \}$ 是给定协变量下处理的条件**优势比（odds）**的拟合值。

$E ( Y \mid Z = 1 )$ 的估计很简单。我们有用于 $E \{ Y ( 0 ) \mid Z = 1 \}$ 的**双重稳健估计量（doubly robust estimator）**

## 13.3 $\tau _ { \mathrm { T } }$ 的逆倾向得分加权与双重稳健估计 167

该估计量结合了倾向得分和结果模型。定义

$$
\tilde {\mu} _ {0 \mathrm{T}} ^ {\mathrm{dr}} = E \left[ o (X, \alpha) (1 - Z) \{Y - \mu_ {0} (X, \beta_ {0}) \} + Z \mu_ {0} (X, \beta_ {0}) \right] / e, \tag {13.5}
$$

其中 $o ( X , \alpha ) = e ( X , \alpha ) / \{ 1 - e ( X , \alpha ) \}$ 。

**定理 13.3** 在假设13.1下，如果 $e ( X , \alpha ) = e ( X )$ 或 $\mu _ { 0 } ( X , \beta _ { 0 } ) = \mu _ { 0 } ( X )$ ，则 $\mu _ { 0 \mathrm { T } } ^ { d r } = E \{ Y ( 0 ) \mid Z = 1 \}$ 。

**定理13.3的证明**：我们有分解

$$
\begin{array}{l} e \left[ \tilde {\mu} _ {0 \mathrm{T}} ^ {\mathrm{dr}} - E \{Y (0) \mid Z = 1 \} \right] \\ = E \left[ o (X, \alpha) (1 - Z) \{Y (0) - \mu_ {0} (X, \beta_ {0}) \} + Z \mu_ {0} (X, \beta_ {0}) \right] - E \{Z Y (0) \} \\ = E [ o (X, \alpha) (1 - Z) \{Y (0) - \mu_ {0} (X, \beta_ {0}) \} - Z \{Y (0) - \mu_ {0} (X, \beta_ {0}) \} ] \\ = E \left[ \left\{o (X, \alpha) (1 - Z) - Z \right\} \left\{Y (0) - \mu_ {0} (X, \beta_ {0}) \right\} \right] \\ = E \left[ \frac {e (X , \alpha) - Z}{1 - e (X , \alpha)} \{Y (0) - \mu_ {0} (X, \beta_ {0}) \} \right] \\ = E \left[ E \left\{\frac {e (X , \alpha) - Z}{1 - e (X , \alpha)} \mid X \right\} \times E \{Y (0) - \mu_ {0} (X, \beta_ {0}) \mid X \} \right] \\ = E \left[ \frac {e (X , \alpha) - e (X)}{1 - e (X , \alpha)} \times \{\mu_ {0} (X) - \mu_ {0} (X, \beta_ {0}) \} \right]. \\ \end{array}
$$

因此，如果 $e ( X , \alpha ) = e ( X )$ 或 $\mu _ { 0 } ( X , \beta _ { 0 } ) = \mu _ { 0 } ( X )$ ，则 $\tilde { \mu } _ { 0 \mathrm { T } } ^ { \mathrm { d r } } - E \{ Y ( 0 ) \mid Z = 1 \} = 0$ 。□

基于 $\tilde { \mu } _ { \mathrm { 0T } } ^ { \mathrm { d r } }$ 的总体版本，我们可以通过以下步骤构造样本版本：

1. 获取倾向得分的拟合值 $e ( X , { \hat { \alpha } } )$ ；
2. 获取对照下结果均值的拟合值 $\mu _ { 0 } ( X , { \hat { \beta } } _ { 0 } )$ ；
3. 构造双重稳健估计量： $\hat { \tau } _ { \mathrm { T } } ^ { \mathrm { d r } } = \hat { \bar { Y } } ( 1 ) - \hat { \mu } _ { 0 \mathrm { T } } ^ { \mathrm { d r } }$ ，其中

$$
\hat {\mu} _ {0 \mathrm{T}} ^ {\mathrm{dr}} = \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} \left[ e (X _ {i}, \hat {\alpha}) \frac {(1 - Z _ {i}) \{Y _ {i} - \mu_ {0} (X _ {i} , \hat {\beta} _ {0}) \}}{1 - e (X _ {i} , \hat {\alpha})} + Z _ {i} \mu_ {0} (X _ {i}, \hat {\beta} _ {0}) \right];
$$

4. 通过对 $( Z _ { i } , X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ 进行重抽样，通过**自助法（bootstrap）**估计 $\tau _ { \mathrm { T } }$ 的方差。

Hahn (1998)、Mercatanti and Li (2014)、Shinozaki and Matsuyama (2015) 以及 Yang and Ding (2018) 是讨论 $\tau _ { \mathrm { T } }$ 估计的参考文献。

## 13.3 实例（An example）

以下R代码实现了用于 $\tau_{\mathrm{T}}$ 的两种结果回归估计量、两种IPW估计量和双重稳健估计量，以及自助法方差估计量。为避免极端的估计倾向得分，我们也可以从上方对其进行截断。

```r
ATT.est = function(z, y, x, out.family = gaussian, Utruncpscore = 1)
{
    ## 样本量
    nn = length(z)
    nn1 = sum(z)

    ## 拟合的倾向得分
    pscore = glm(z ~ x, family = binomial)$fitted.values
    pscore = pmin(Utruncpscore, pscore)
    odds.pscore = pscore/(1 - pscore)

    ## 拟合的潜在结果
    outcome0 = glm(y ~ x, weights = (1 - z),
    family = out.family)$fitted.values

    ## 回归插补估计量
    ace.reg0 = lm(y ~ z + x)$coef[2]
    ace.reg = mean(y[z==1]) - mean(outcome0[z==1])
    ## 倾向得分加权估计量
    ace.ipw0 = mean(y[z==1]) - mean(odds.pscore*(1 - z)*y)*nn/nn1
    ace.ipw = mean(y[z==1]) - mean(odds.pscore*(1 - z)*y)/mean(odds.pscore*(1 - z))
    ## 双重稳健估计量
    res0 = y - outcome0
    ace.dr = ace.reg - mean(odds.pscore*(1 - z)*res0)*nn/nn1

    return(c(ace.reg0, ace.reg, ace.ipw0, ace.ipw, ace.dr))
}

OS_ATT = function(z, y, x, n.boot = 10^2,
    out.family = gaussian, Utruncpscore = 1)
{
    point.est = ATT.est(z, y, x, out.family, Utruncpscore)

    ## 非参数自助法
    n.sample = length(z)
    x = as.matrix(x)
    boot.est = replicate(n.boot,
    {id.boot = sample(1:n.sample, n.sample, replace = TRUE)
```

```txt
ATT.est(z[id.boot], y[id.boot], x[id.boot, ], out.family, Utruncpscore))
```

```txt
boot.se = apply(boot.est, 1, sd)
res = rbind(point.est, boot.se)
rownames(res) = c("est", "se")
colnames(res) = c("reg0", "reg", "HT", "Hajek", "DR")
return(res)
}
```

现在我们重新分析例10.3中的数据以估计 $\tau_{\mathrm{T}}$ 。得到

```csv
reg0 reg HT Hajek DR
est 0.061 -0.351 -1.992 -0.351 -0.187
se 0.227 0.258 0.705 0.328 0.287
```

（未截断估计的倾向得分），以及

```batch
reg0 reg HT Hajek DR
est 0.061 -0.351 -0.597 -0.192 -0.230
se 0.223 0.255 0.579 0.302 0.276
```

（从上方以0.9截断估计的倾向得分）。HT估计量对截断敏感，这在意料之中。例13.1中的回归估计量与其他估计量差异较大。它施加了一个不必要的假设，即处理组和对照组的回归函数共享相同的 $X$ 系数。例13.2中的回归估计量与Hajek估计量和双重稳健估计量非常接近。上述估计值与第12.3.3节中的结果略有不同，这表明 $\tau_{\mathrm{T}}$ 和 $\tau$ 之间存在一定的处理效应异质性。

## 13.4 其他估计量（Other estimands）

Li 等人 (2018a) 对观测性研究中的**因果估计量（causal estimands）** 进行了统一讨论。从**条件平均因果效应（conditional average causal effect）** $\tau (X)$ 出发，他们提出了一类一般的估计量

$$
\tau^ {h} = \frac {E \{h (X) \tau (X) \}}{E \{h (X) \}}
$$

由一个**权重函数（weighting function）** $h ( X )$ 索引，且满足 $E \{ h ( X ) \} \ne 0$ 。分母中的归一化是为了确保一个恒定的因果效应 $\tau ( X ) = \tau$ 平均后仍为相同的 $\tau$ 。

在**无混杂假设（unconfoundedness assumption）** 下，

$$
\tau^ {h} = \frac {E [ h (X) \{\mu_ {1} (X) - \mu_ {0} (X) \} ]}{E \{h (X) \}}
$$

## 17013 处理组上的平均因果效应及其他估计量（The Average Causal Effect on the Treated Units and Other Estimands）

这引出了**结果回归估计量（outcome regression estimator）**

$$
\hat {\tau} ^ {h} = \frac {\sum_ {i = 1} ^ {n} h (X _ {i}) \{\hat {\mu} _ {1} (X _ {i}) - \hat {\mu} _ {0} (X _ {i}) \}}{\sum_ {i = 1} ^ {n} h (X _ {i})}.
$$

此外，我们可以证明 $\tau ^ { h }$ 具有以下加权形式：

**定理 13.4** 在**可忽略性（ignorability）** 和**重叠性（overlap）** 下，我们有

$$
\tau^ {h} = E \left\{\frac {Z Y h (X)}{e (X)} - \frac {(1 - Z) Y h (X)}{1 - e (X)} \right\} / E \{h (X) \}.
$$

定理 13.4 的证明与定理 11.2 和 13.2 的证明类似，被归为问题 13.8。基于定理 13.4，我们可以构造相应的**逆概率加权（IPW）估计量**。

根据定理 13.4，每个单元既与由估计量定义产生的权重相关联，也与由**倾向得分（propensity score）** 的倒数产生的权重相关联。最终，处理组单元被加权为 $h ( X ) / e ( X )$ ，控制组单元被加权为 $h ( X ) / \{ 1 - e ( X ) \}$ 。Li 等人 (2018a, 表 1) 总结了几种估计量，我将其部分内容呈现如下：

<table><tr><td>population</td><td>h(X)</td><td>estimand</td><td>weights</td></tr><tr><td>combined</td><td>1</td><td> $\tau$ </td><td> $1/e(X)$  and  $1/\{1-e(X)\}$ </td></tr><tr><td>treated</td><td>e(X)</td><td> $\tau_{\text{T}}$ </td><td>1 and e(X)/ $\{1-e(X)\}$ </td></tr><tr><td>control</td><td>1-e(X)</td><td> $\tau_{\text{C}}$ </td><td> $\{1-e(X)\}/e(X)$  and 1</td></tr><tr><td>overlap</td><td>e(X){1-e(X)}</td><td> $\tau_{\text{O}}$ </td><td>1-e(X) and e(X)</td></tr></table>

**重叠总体（overlap population）** 及其对应的估计量

$$
\tau_ {\mathrm{O}} = \frac {E [ e (X) \{1 - e (X) \} \tau (X) ]}{E [ e (X) \{1 - e (X) \} ]}
$$

对我们来说是新颖的。该估计量对 $e ( X ) = 1 / 2$ 的单元赋予最大权重，并降低具有极端倾向得分的单元的权重。该估计量的一个良好特性是，其 IPW 估计量相当稳定，因为分母中不存在可能极小的 $e ( X )$ 和 $1 - e ( X )$ 值。如果 $e ( X ) { \underline { { \bot \bot } } } \tau ( X )$ ，包括 $\tau ( X ) = \tau$ 的特殊情况，则参数 $\tau _ { \mathrm { O } }$ 退化为 $\tau$ 。然而，一般而言，估计量 $\tau _ { \mathrm { O } }$ 可能会引起争议，因为它改变了初始总体，并且依赖于实践中可能被错误设定的倾向得分。Li 等人 (2018a) 和 Li 等人 (2019) 给出了一些论证和数值证据。该估计量将在第 14 章再次出现。

我们也可以为 $\tau ^ { h }$ 构造**双重稳健估计量（doubly robust estimator）** 。我将细节留至问题 13.9。

## 13.5 课后习题（Homework Problems）

## 13.1 关于 $\tau _ { \mathrm { T } }$ 的回归估计量的一个代数事实（An algebraic fact about a regression estimator for $\tau _ { \mathrm { T } }$ ）

本题为例 13.2 提供更多细节。

证明：如果对所有单元，将协变量中心化为 $X _ { i } - \hat { \bar { X } } ( 1 )$ ，那么 $\hat { \tau } _ { \mathrm { T } }$ 等于在结果变量对处理变量、协变量及其交互项进行 OLS 拟合中 $Z$ 的系数。

## 13.2 处理组上平均因果效应的模拟（Simulation for the average causal effect on the treated units）

在第 12 章的 OSATE.R 中，我对 $\tau$ 进行了一些模拟研究。对 $\tau _ { \mathrm { T } }$ 进行类似的模拟研究，使用正确或错误的倾向得分或结果模型。

你可以选择不同的模型参数、更大的模拟次数和**自助法（bootstrap）** 重复次数。报告你的发现，至少包括偏差、方差以及通过自助法得到的方差估计量。你也可以报告估计量的其他性质，例如渐近正态性和置信区间的覆盖率。

## 13.3 $\tau _ { \mathrm { T } }$ 的双重稳健估计量的另一种形式（An alternative form of the doubly robust estimator for $\tau _ { \mathrm { T } }$ ）

受 (13.5) 启发，对于 $E \{ Y ( 0 ) \mid Z = 1 \}$ ，我们有双重稳健估计量的另一种形式：

$$
\tilde {\mu} _ {0 \mathrm{T}} ^ {\mathrm{dr2}} = \frac {E [ o (X , \alpha) (1 - Z) \{Y - \mu_ {0} (X , \beta_ {0}) \} ]}{E [ o (X , \alpha) (1 - Z) ]} + E \{Z \mu_ {0} (X, \beta_ {0}) \} / e.
$$

证明：在假设 13.1 下，如果 $e ( X , \alpha ) = e ( X )$ 或 $\mu _ { 0 } ( X , \beta _ { 0 } ) = \mu _ { 0 } ( X )$ ，则 $\tilde { \mu } _ { 0 \mathrm { T } } ^ { \mathrm { d r 2 } } = E \{ Y ( 0 ) | Z = 1 \}$ 。给出 $\tau _ { \mathrm { T } }$ 的双重稳健估计量的样本类似形式。

## 13.4 控制组上的平均因果效应（Average causal effect on the control units）

证明 $\tau _ { \mathrm { { C } } }$ 的识别公式，类似于 (13.1) 和 (13.4)。提出 $\tau _ { \mathrm { C } }$ 的双重稳健估计量。

## 13.5 估计个体效应和条件平均因果效应（Estimating individual effect and conditional average causal effect）

设 $\{ Z _ { i } , X _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \} _ { i = 1 } ^ { n } \stackrel { \mathrm { I I D } } { \sim } \{ Z , X , Y ( 1 ) , Y ( 0 ) \}$ ，个体效应为 $\tau _ { i } = Y _ { i } ( 1 ) - Y _ { i } ( 0 )$ ，条件平均因果效应为 $\tau ( X _ { i } ) =$ $E \{ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \mid X _ { i } \}$ 。由于我们将讨论个体效应，我们不会去掉下标 i，因为 $\tau$ 表示平均因果效应，而非 $Y ( 1 ) - Y ( 0 )$ 的总体版本。

1. 在随机化条件下， $Z _ { i } \bot \bot \{ Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \}$ 且 $e = \mathrm { p r } ( Z _ { i } = 1 )$ ，

## 17213 处理组上的平均因果效应及其他估计量（The Average Causal Effect on the Treated Units and Other Estimands）

证明

$$
\delta_ {i} = \frac {Z _ {i} Y _ {i}}{e} - \frac {(1 - Z _ {i}) Y _ {i}}{1 - e}
$$

是个体效应的无偏预测因子，即

$$
E (\delta_ {i} - \tau_ {i}) = 0 (i = 1, \dots , n).
$$

进一步证明对所有 $i = 1 , \ldots , n$ ，有 $E ( \delta _ { i } ) = \tau$ 。

2. 在可忽略性条件下， $Z _ { i } \bot \bot \{ Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \} \quad | \quad X _ { i }$ 且 $e ( X _ { i } ) \ =$ pr $\ \cdot Z _ { i } = 1 \mid X _ { i } )$ ，证明

$$
\delta_ {i} = \frac {Z _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - Z _ {i}) Y _ {i}}{1 - e (X _ {i})}
$$

是个体效应和条件平均因果效应的无偏预测因子，即

$$
E \left(\delta_ {i} - \tau_ {i}\right) = 0, \quad E \left\{\delta_ {i} - \tau \left(X _ {i}\right) \right\} = 0, \quad (i = 1, \dots , n).
$$

进一步证明对所有 $i = 1 , \ldots , n$ ，有 $E ( \delta _ { i } ) = \tau$ 。

## 13.6 一般估计量与 $( \tau _ { \mathrm { T } } , \tau _ { \mathrm { C } } )$ （General estimand and $( \tau _ { \mathrm { T } } , \tau _ { \mathrm { C } } )$ ）

假设无混杂性。证明：如果 $h ( X ) = e ( X )$ ，则 $\tau ^ { h } = \tau _ { \mathrm { T } }$ ；如果 $h ( X ) = 1 - e ( X )$ ，则 $\tau ^ { h } = \tau _ { \mathrm { { C } } }$ 。

## 13.7 关于 $\tau _ { \mathrm { O } }$ 的更多内容（More on $\tau _ { \mathrm { O } }$ ）

证明

$$
\tau_ {\mathrm{O}} = \frac {E [ \{1 - e (X) \} \tau (X) \mid Z = 1 ]}{E \{1 - e (X) \mid Z = 1 \}} = \frac {E \{e (X) \tau (X) \mid Z = 0 \}}{E \{e (X) \mid Z = 0 \}}.
$$

## 13.8 一般估计量的逆概率加权（IPW for the general estimand）

证明定理 13.4。

## 13.9 一般估计量的双重稳健估计（Doubly robust estimation for general estimand）

对于给定的 $h ( X )$ ，我们有以下用于构造 $\tau ^ { h }$ 的双重稳健估计量的公式：

$$
\begin{array}{l} \tilde {\mu} _ {1} ^ {h, \mathrm{dr}} = E \left[ \frac {Z h (X) \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X , \alpha)} + h (X) \mu_ {1} (X, \beta_ {1}) \right], \\ \tilde {\mu} _ {0} ^ {h, \mathrm{dr}} = E \left[ \frac {(1 - Z) h (X) \{Y - \mu_ {0} (X , \beta_ {0}) \}}{1 - e (X , \alpha)} + h (X) \mu_ {0} (X, \beta_ {0}) \right]. \\ \end{array}
$$

证明：在可忽略性和重叠性下，

## 13.5 课后习题（Homework Problems）

1. 如果 $e(X, \alpha) = e(X)$ 或 $\mu _ { 1 } ( X , \beta _ { 1 } ) \ = \ \mu _ { 1 } ( X )$ ，则 $\tilde { \mu } _ { 1 } ^ { h , \mathrm { d r } } ~ =$ E{h(X)Y (1)}；
2. 如果 $e(X, \alpha) = e(X)$ 或 $\mu _ { 0 } ( X , \beta _ { 0 } ) \ = \ \mu _ { 0 } ( X )$ ，则 $\tilde { \mu } _ { 0 } ^ { h , \mathrm { d r } } ~ =$ E{h(X)Y (0)}；
3. 如果 $e ( X , \alpha ) ~ = ~ e ( X ) ~ \mathrm { o r } ~ \{ \mu _ { 1 } ( X , \beta _ { 1 } ) ~ = ~ \mu _ { 1 } ( X ) , \mu _ { 0 } ( X , \beta _ { 0 } ) ~ =$ $\mu _ { 0 } ( X ) \}$ ，则

$$
\frac {\tilde {\mu} _ {1} ^ {h , \mathrm{dr}} - \tilde {\mu} _ {0} ^ {h , \mathrm{dr}}}{E \{h (X) \}} = \tau^ {h}.
$$

备注：Tao 和 Fu (2019) 证明了上述结果。然而，它们仅对给定的 $h ( X )$ 成立。最令人感兴趣的 $\tau _ { \mathrm { T } }$ 、 $\tau _ { \mathrm { C } }$ 和 $\tau _ { \mathrm { O } }$ 的权重都依赖于倾向得分 $e ( X )$ ，而倾向得分必须首先被估计。上述公式不适用于构造 $\tau _ { \mathrm { T } }$ 和 $\tau _ { \mathrm { { C } } }$ 的双重稳健估计量；对于 $\tau _ { \mathrm { O } }$ ，并不存在双重稳健估计量。

## 13.10 推荐阅读（Recommended reading）

Shinozaki 和 Matsuyama (2015) 专注于 $\tau _ { \mathrm { T } }$ ，而 Li 等人 (2018a) 讨论了一般的 $\tau ^ { h }$ 。
# 平均因果效应的双重稳健估计量或增广逆倾向得分加权估计量（The Doubly Robust or the Augmented Inverse Propensity Score Weighting Estimator for the Average Causal Effect）

在**无混杂性（unconfoundedness）** $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$ 和**重叠性（overlap）** $0 < e ( X ) < 1$ 条件下，第11章给出了**平均因果效应（average causal effect）** $\tau = E \{ Y ( 1 ) - Y ( 0 ) \}$ 的两个识别公式。首先，**结果插补公式（outcome imputation formula）**为：

$$
\tau = E \{\mu_ {1} (X) \} - E \{\mu_ {0} (X) \} \tag {12.1}
$$

其中

$$
\mu_ {1} (X) = E \{Y (1) \mid X \} = E (Y \mid Z = 1, X),
$$

$$
\mu_ {0} (X) = E \{Y (0) \mid X \} = E (Y \mid Z = 0, X)
$$

是给定协变量条件下结果变量的两个条件均值函数。其次，**逆倾向得分加权（Inverse Propensity Score Weighting, IPW）公式**为：

$$
\tau = E \left\{\frac {Z Y}{e (X)} \right\} - E \left\{\frac {(1 - Z) Y}{1 - e (X)} \right\} \tag {12.2}
$$

其中

$$
e (X) = \operatorname{pr} (Z = 1 \mid X)
$$

是第11章介绍的**倾向得分（propensity score）**。

**结果插补估计量（outcome imputation estimator）** 需要拟合给定处理变量和协变量条件下结果变量的模型。如果结果模型正确设定，则该估计量是一致的。**IPW估计量** 需要拟合给定协变量条件下处理变量的模型。如果倾向得分模型正确设定，则该估计量是一致的。

从数学上讲，(12.1)和(12.2)有许多组合形式，可得到平均因果效应的不同识别公式。下面我将讨论一种具有良好理论性质的特定组合。这种组合催生了一个估计量，该估计量在倾向得分模型或结果模型之一正确设定时是一致的。它被称为**双重稳健估计量（doubly robust estimator）**，由 James Robins 等人倡导（Scharfstein et al., 1999; Bang and Robins, 2005）。

## 12.1 双重稳健估计量（The doubly robust estimator）

## 12.1.1 总体版本（Population version）

我们为结果的**条件均值（conditional means）** $\mu _ { 1 } ( X , \beta _ { 1 } )$ 和 $\mu _ { 0 } ( X , \beta _ { 0 } )$ 设定一个**工作模型（working model）**，该模型由参数 $\beta _ { 1 }$ 和 $\beta _ { 0 }$ 索引。例如，如果工作模型下的条件均值是线性的或逻辑的，那么这些参数就是回归系数。如果结果模型正确设定，则 $\mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X )$ 且 $\mu _ { 0 } ( X , \beta _ { 0 } ) = \mu _ { 0 } ( X )$ 。我们为倾向得分 $e ( X , \alpha )$ 设定一个工作模型，由参数 $\alpha$ 索引。例如，如果工作模型是逻辑模型，则 $\alpha$ 是回归系数。如果倾向得分模型正确设定，则 $e ( X , \alpha ) = e ( X )$ 。在实践中，这两个模型可能都被误设。

定义

$$
\tilde {\mu} _ {1} ^ {\mathrm{dr}} = E \left[ \frac {Z \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X , \alpha)} + \mu_ {1} (X, \beta_ {1}) \right], \tag {12.3}
$$

$$
\tilde {\mu} _ {0} ^ {\mathrm{dr}} = E \left[ \frac {(1 - Z) \{Y - \mu_ {0} (X , \beta_ {0}) \}}{1 - e (X , \alpha)} + \mu_ {0} (X, \beta_ {0}) \right], \tag {12.4}
$$

这也可以写成

$$
\tilde {\mu} _ {1} ^ {\mathrm{dr}} = E \left[ \frac {Z Y}{e (X , \alpha)} - \frac {Z - e (X , \alpha)}{e (X , \alpha)} \mu_ {1} (X, \beta_ {1}) \right], \tag {12.5}
$$

$$
\tilde {\mu} _ {0} ^ {\mathrm{dr}} = E \left[ \frac {(1 - Z) Y}{1 - e (X , \alpha)} - \frac {e (X , \alpha) - Z}{1 - e (X , \alpha)} \mu_ {0} (X, \beta_ {0}) \right]. \tag {12.6}
$$

(12.3)和(12.4)中的公式通过残差的逆倾向得分加权项对结果插补估计量进行了增广。(12.5)和(12.6)中的公式通过插补结果对IPW估计量进行了增广。因此，双重稳健估计量也被称为**增广逆倾向得分加权（Augmented Inverse Propensity Score Weighting, AIPW）估计量**。

这种增广在以下意义上强化了理论性质。

**定理12.1** 假设无混杂性 $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$ 和重叠性 $0 < e ( X ) < 1$。

1. 如果 $e ( X , \alpha ) = e ( X )$ 或 $\mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X )$ 中至少有一个成立，则 $\tilde { \mu } _ { 1 } ^ { \mathrm { d r } } = E \{ Y ( 1 ) \}$。
2. 如果 $e ( X , \alpha ) = e ( X )$ 或 $\mu _ { 0 } ( X , \beta _ { 0 } ) = \mu _ { 0 } ( X )$ 中至少有一个成立，则 $\tilde { \mu } _ { 0 } ^ { \mathrm { d r } } = E \{ Y ( 0 ) \}$。
3. 如果 $e ( X , \alpha ) = e ( X )$ 或 $\{ \mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X ) , \mu _ { 0 } ( X , \beta _ { 0 } ) = \mu _ { 0 } ( X ) \}$ 中至少有一个成立，则 $\tilde { \mu } _ { 1 } ^ { \mathrm { d r } } - \tilde { \mu } _ { 0 } ^ { \mathrm { d r } } = \tau$。

由定理12.1可知，如果倾向得分模型或结果模型中至少有一个正确设定，则 $\tilde { \mu } _ { 1 } ^ { \mathrm { d r } } - \tilde { \mu } _ { 0 } ^ { \mathrm { d r } }$ 等于 $\tau$。这就是其被称为双重稳健估计量的原因。

**定理12.1的证明：** 我仅证明 $\mu _ { 1 } = E \{ Y ( 1 ) \}$ 的结果。$\mu _ { 0 } = E \{ Y ( 0 ) \}$ 的证明类似。我们有如下分解：

$$
\begin{array}{l} \tilde {\mu} _ {1} ^ {\mathrm{dr}} - E \{Y (1) \} = E \left[ \frac {Z \{Y (1) - \mu_ {1} (X , \beta_ {1}) \}}{e (X , \alpha)} - \{Y (1) - \mu_ {1} (X, \beta_ {1}) \} \right] \\ = E \left[ \frac {Z - e (X , \alpha)}{e (X , \alpha)} \{Y (1) - \mu_ {1} (X, \beta_ {1}) \} \right] \\ = E \left(E \left[ \frac {Z - e (X , \alpha)}{e (X , \alpha)} \{Y (1) - \mu_ {1} (X, \beta_ {1}) \} \mid X \right]\right) \\ = E \left[ E \left\{\frac {Z - e (X , \alpha)}{e (X , \alpha)} \mid X \right\} \times E \left\{Y (1) - \mu_ {1} (X, \beta_ {1}) \mid X \right\} \right] \\ = E \left[ \frac {e (X) - e (X , \alpha)}{e (X , \alpha)} \times \{\mu_ {1} (X) - \mu_ {1} (X, \beta_ {1}) \} \right]. \\ \end{array}
$$

因此，如果 $e ( X , \alpha ) = e ( X )$ 或 $\mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X )$ 中至少有一个成立，则 $\tilde { \mu } _ { 1 } ^ { \mathrm { d r } } - E \{ Y ( 1 ) \} = 0$。

## 12.1.2 样本版本（Sample version）

根据 $\tilde { \mu } _ { 1 } ^ { \mathrm { d r } }$ 和 $\tilde { \mu } _ { 0 } ^ { \mathrm { d r } }$ 的总体版本，我们可以通过以下步骤构建样本版本：

1. 获取倾向得分的拟合值：$e ( X , { \hat { \alpha } } )$；
2. 获取结果均值的拟合值：$\mu _ { 1 } ( X , { \hat { \beta } } _ { 1 } )$ 和 $\mu _ { 0 } ( X , { \hat { \beta } } _ { 0 } )$；
3. 构建双重稳健估计量：$\hat { \tau } ^ { \mathrm { d r } } = \hat { \mu } _ { 1 } ^ { \mathrm { d r } } - \hat { \mu } _ { 0 } ^ { \mathrm { d r } }$，其中

$$
\hat {\mu} _ {1} ^ {\mathrm{dr}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left[ \frac {Z _ {i} \{Y _ {i} - \mu_ {1} (X _ {i} , \hat {\beta} _ {1}) \}}{e (X _ {i} , \hat {\alpha})} + \mu_ {1} (X _ {i}, \hat {\beta} _ {1}) \right]
$$

且

$$
\hat {\mu} _ {0} ^ {\mathrm{dr}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left[ \frac {(1 - Z _ {i}) \{Y _ {i} - \mu_ {0} (X _ {i} , \hat {\beta} _ {0}) \}}{1 - e (X _ {i} , \hat {\alpha})} + \mu_ {0} (X _ {i}, \hat {\beta} _ {0}) \right];
$$

4. 通过对 $( Z _ { i } , X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ 进行重抽样，使用**非参数自助法（nonparametric bootstrap）** 近似 ${ \hat { \tau } } ^ { \mathrm { d r } }$ 的方差（Funk et al., 2011）。

类似于(12.5)和(12.6)，我们也可以将 $\hat { \mu } _ { 1 } ^ { \mathrm { d r } }$ 和 $\hat { \mu } _ { 0 } ^ { \mathrm { d r } }$ 重写为：

$$
\hat {\mu} _ {1} ^ {\mathrm{dr}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left[ \frac {Z _ {i} Y _ {i}}{e (X _ {i} , \hat {\alpha})} - \frac {Z _ {i} - e (X _ {i} , \hat {\alpha})}{e (X _ {i} , \hat {\alpha})} \mu_ {1} (X _ {i}, \hat {\beta} _ {1}) \right],
$$

$$
\hat {\mu} _ {0} ^ {\mathrm{dr}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left[ \frac {(1 - Z _ {i}) Y _ {i}}{1 - e (X _ {i} , \hat {\alpha})} - \frac {e (X _ {i} , \hat {\alpha}) - Z _ {i}}{1 - e (X _ {i} , \hat {\alpha})} \mu_ {0} (X _ {i}, \hat {\beta} _ {0}) \right].
$$

## 12.2 双重稳健估计量的更多直觉与理论（More intuition and theory for the doubly robust estimator）

尽管本章开头指出，基于结果回归和逆倾向得分加权的基本识别公式可以立即产生无穷多个其他识别公式，但(12.3)和(12.4)中双重稳健估计量的特定形式并非显而易见。提出(12.3)和(12.4)的原始动机是相当理论性的，它依赖于高级数理统计学中的**半参数效率理论（semiparametric efficiency theory）**（Bickel et al., 1993），这超出了本书的范畴。下面我将给出两种更直观的视角来构建(12.3)和(12.4)。下面的12.2.1节和12.2.2节都聚焦于 $E \{ Y ( 1 ) \}$ 的估计，因为 $E \{ Y ( 0 ) \}$ 的估计可通过对称性类似得到。

## 12.2.1 降低IPW估计量的方差（Reducing the variance of the IPW estimator）

基于以下公式的 $\mu _ { 1 }$ 的IPW估计量

$$
\mu_ {1} = E \left\{\frac {Z Y}{e (X)} \right\}
$$

完全忽略了 $Y$ 的结果模型。它的优点是在不假设任何结果模型的情况下保持一致性。然而，如果协变量对结果有预测能力，那么基于一个工作结果模型的残差通常比结果本身具有更小的方差，即使这个工作结果模型是错误的。对于一个可能被误设的结果模型 $\mu _ { 1 } ( X , \beta _ { 1 } )$ ，存在一个简单的分解：

$$
\mu_ {1} = E \{Y (1) \} = E \{Y (1) - \mu_ {1} (X, \beta_ {1}) \} + E \{\mu_ {1} (X, \beta_ {1}) \}.
$$

如果我们将IPW公式应用于上式中的第一项，并将 $Y ( 1 ) - \mu _ { 1 } ( X , \beta _ { 1 } )$ 视为处理条件下的一个伪潜在结果，则可将上式重写为：

$$
\mu_ {1} = E \left\{\frac {Z \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X)} \right\} + E \{\mu_ {1} (X, \beta_ {1}) \} \tag {12.7}
$$

$$
= E \left\{\frac {Z \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X)} + \mu_ {1} (X, \beta_ {1}) \right\}, \tag {12.8}
$$

该式在倾向得分模型正确（而不假设结果模型正确）的情况下成立。使用工作模型来提高效率是**调查抽样（survey sampling）**中的一个经典思想。Little 和 An（2004）以及 Lumley 等人（2011）指出了其与双重稳健估计量的联系。

## 12.2.2 降低结果回归估计量的偏差（Reducing the bias of the outcome regression estimator）

12.2.1节的讨论从IPW估计量出发，并基于一个工作结果模型提高了其效率。或者，我们也可以从基于以下公式的结果回归估计量出发：

$$
\tilde {\mu} _ {1} = E \{\mu_ {1} (X, \beta_ {1}) \}
$$

由于结果模型可能是错误的，该估计量可能不等于 $\mu _ { 1 }$ 。该估计量的偏差为 $E \{ \mu _ { 1 } ( X , \beta _ { 1 } ) - Y ( 1 ) \}$ ，如果倾向得分模型正确，则可以通过一个IPW估计量来估计该偏差：

$$
B = E \left\{\frac {Z \{\mu_ {1} (X , \beta_ {1}) - Y \}}{e (X)} \right\}
$$

因此，一个去偏估计量为 $\tilde { \mu } _ { 1 } - B$ ，这与(12.8)相同。

## 12.3 示例（Examples）

## 12.3.1 一些 $\tau$ 的规范估计量总结（Summary of some canonical estimators for τ）

下面的R代码实现了 $\tau$ 的结果插补估计量、**霍维茨-汤普森估计量（Horvitz–Thompson estimator）**、**哈耶克估计量（Hájek estimator）** 和双重稳健估计量。这些估计量可以基于 `glm` 函数的拟合值方便地实现。倾向得分模型的默认选择是逻辑模型，结果模型的默认选择是线性模型，其中 `out.family = gaussian`。对于二元结果，我们也可以指定 `out.family = binomial` 来拟合逻辑模型。

```txt
OS_est = function(z, y, x, out.family = gaussian,
    truncpscore = c(0, 1))
{
    ## fitted propensity score
    pscore = glm(z ~ x, family = binomial)$fitted.values
    pscore = pmax(truncpscore[1], pmin(truncpscore[2], pscore))
```

```r
## fitted potential outcomes
outcome1 = glm(y ~ x, weights = z,
    family = out.family)$fitted.values
outcome0 = glm(y ~ x, weights = (1 - z),
    family = out.family)$fitted.values

## regression imputation estimator
ace.reg = mean(outcome1 - outcome0)
```

## IPW 估计量（IPW estimators）
```r
ace.ipw0 = mean(z*y/pscore - (1 - z)*y/(1 - pscore))
ace.ipw = mean(z*y/pscore)/mean(z/pscore) -
    mean((1 - z)*y/(1 - pscore))/mean((1 - z)/(1 - pscore))
## 双重稳健估计量（doubly robust estimator）
res1 = y - outcome1
res0 = y - outcome0
ace.dr = ace.reg + mean(z*res1/pscore - (1 - z)*res0/(1 - pscore))

return(c(ace.reg, ace.ipw0, ace.ipw, ace.dr))
}
```

计算上述估计量方差的解析公式非常繁琐。**自助法（Bootstrap）** 通过对 $\{ Z _ { i } , X _ { i } , Y _ { i } \} _ { i = 1 } ^ { n }$ 进行重抽样，为方差提供了便捷的近似。基于 `OSest`，以下函数返回点估计量以及自助法标准误。

```r
OS_ATE = function(z, y, x, n.boot = 2*10^2,
    out.family = gaussian, truncpscore = c(0, 1))
{
    point.est = OS_est(z, y, x, out.family, truncpscore)

    ## 非参数自助法（nonparametric bootstrap）
    n.sample = length(z)
    x = as.matrix(x)
    boot.est = replicate(n.boot,
    {id.boot = sample(1:n.sample, n.sample, replace = TRUE)
    OS_est(z[id.boot], y[id.boot], x[id.boot, ],
    out.family, truncpscore)})
    boot.se = apply(boot.est, 1, sd)

    res = rbind(point.est, boot.se)
    rownames(res) = c("est", "se")
    colnames(res) = c("reg", "HT", "Hajek", "DR")

    return(res)
}
```

## 12.3.2 模拟（Simulation）

我将通过模拟来评估估计量在四种情景下的有限样本性质：

1.  倾向得分和结果模型都正确；
2.  倾向得分模型错误但结果模型正确；
3.  倾向得分模型正确但结果模型错误；
4.  倾向得分和结果模型都错误。

我将报告估计量在模拟中的**平均偏差（average bias）**、**真实标准误（true standard error）**和**平均估计标准误（average estimated standard error）**。

在情景 1 中，数据生成过程如下：

```matlab
x = matrix(rnorm(n*2), n, 2)
x1 = cbind(1, x)
beta.z = c(0, 1, 1)
pscore = 1/(1 + exp(- as.vector(x1%* %beta.z)))
z = rbinom(n, 1, pscore)
beta.y1 = c(1, 2, 1)
beta.y0 = c(1, 2, 1)
y1 = rnorm(n, x1%* %beta.y1)
y0 = rnorm(n, x1%* %beta.y0)
y = z*y1 + (1 - z)*y0
```

在情景 2 中，我将倾向得分模型修改为非线性：

```txt
x1 = cbind(1, x, exp(x))
beta.z = c(-1, 0, 0, 1, -1)
pscore = 1/(1 + exp(- as.vector(x1%* % beta.z)))
```

在情景 3 中，我将结果模型修改为非线性：

```txt
beta.y1 = c(1, 0, 0, 0.2, -0.1)
beta.y0 = c(1, 0, 0, -0.2, 0.1)
y1 = rnorm(n, x1%* %beta.y1)
y0 = rnorm(n, x1%* %beta.y0)
```

在情景 4 中，我同时修改了倾向得分和结果模型。

我们将样本量设为 n = 500，并根据上述数据生成过程生成 500 个独立数据集。在情景 1 中：

```batch
reg HT Hajek DR
ave.bias 0.00 0.02 0.03 0.01
true.se 0.11 0.28 0.26 0.13
est.se 0.10 0.25 0.23 0.12
```

所有估计量几乎都是无偏的。两个加权估计量具有更大的方差。在情景 2 中：

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>ave.bias</td><td>0.00</td><td>-0.76</td><td>-0.75</td><td>-0.01</td></tr><tr><td>true.se</td><td>0.12</td><td>0.59</td><td>0.47</td><td>0.18</td></tr><tr><td>est.se</td><td>0.13</td><td>0.50</td><td>0.38</td><td>0.18</td></tr></table>

由于倾向得分模型的错误设定，两个加权估计量存在严重偏差。**回归插补（regression imputation）**和**双重稳健估计量（doubly robust estimator）**几乎是无偏的。在情景 3 中：

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>ave.bias</td><td>-0.05</td><td>0.00</td><td>-0.01</td><td>0.00</td></tr><tr><td>true.se</td><td>0.11</td><td>0.15</td><td>0.14</td><td>0.14</td></tr><tr><td>est.se</td><td>0.11</td><td>0.14</td><td>0.13</td><td>0.14</td></tr></table>

由于结果模型的错误设定，回归插补估计量的偏差大于其他三个估计量。加权估计量和双重稳健估计量几乎是无偏的。在情景 4 中：

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>ave.bias</td><td>-0.08</td><td>0.11</td><td>-0.07</td><td>0.16</td></tr><tr><td>true.se</td><td>0.13</td><td>0.32</td><td>0.20</td><td>0.41</td></tr><tr><td>est.se</td><td>0.13</td><td>0.25</td><td>0.16</td><td>0.26</td></tr></table>

所有估计量都有偏差，因为倾向得分和结果模型都错了。**霍维茨-汤普森估计量（Horvitz–Thompson estimator）**和双重稳健估计量的偏差最大。当两个模型都错误时，双重稳健估计量似乎表现出**双重脆弱性（doubly fragile）**。

在上述所有情景中，当估计量对真实平均因果效应几乎无偏时，自助法标准误接近真实标准误。

## 12.3.3 应用（Applications）

重新审视示例 10.3，我们得到以下估计量和自助法标准误：

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>est</td><td>-0.017</td><td>-1.516</td><td>-0.156</td><td>-0.019</td></tr><tr><td>se</td><td>0.230</td><td>0.492</td><td>0.246</td><td>0.233</td></tr></table>

两个加权估计量远大于其他两个估计量。将估计的倾向得分截断在 [0.1, 0.9] 后，我们得到以下估计量和自助法标准误：

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>est</td><td>-0.017</td><td>-0.713</td><td>-0.054</td><td>-0.043</td></tr><tr><td>se</td><td>0.223</td><td>0.422</td><td>0.235</td><td>0.231</td></tr></table>

**哈耶克估计量（Hajek estimator）**变得非常接近回归插补估计量和双重稳健估计量，而霍维茨-汤普森估计量仍然是一个异常值。

## 12.4 进一步讨论（Some further discussion）

回顾定理 12.1 的证明，双重稳健性质的关键在于以下公式中的乘积结构：

$$
\tilde {\mu} _ {1} ^ {\mathrm{dr}} - E \{Y (1) \} = E \left[ \frac {e (X) - e (X , \alpha)}{e (X , \alpha)} \times \{\mu_ {1} (X) - \mu_ {1} (X, \beta_ {1}) \} \right],
$$

这确保了如果 $e ( X ) = e ( X , \alpha )$ 或 $\mu _ { 1 } ( X ) = \mu _ { 1 } ( X , \beta _ { 1 } )$ ，则估计误差为零。这种精巧的结构使得当倾向得分和结果模型都被错误设定时，双重稳健估计量可能表现出双重脆弱性。两个误差的乘积可能产生更大的误差。Kang 和 Schafer (2007) 基于广泛的模拟研究批评了双重稳健估计量。他们发现，双重稳健估计量的有限样本表现甚至可能比简单的回归插补和 IPW 估计量更不稳定。

尽管 Kang 和 Schafer (2007) 提出了批评，但自 Scharfstein 等人 (1999) 的开创性工作以来，双重稳健估计量已成为因果推断中的标准策略。最近，它在理论统计学和计量经济学文献中以一种更花哨的名字“**双重机器学习（double machine learning）**” (Chernozhukov 等人, 2018) 重新兴起。其基本思想是用机器学习工具替代倾向得分和结果的工作模型，这些工具可以被视为比传统参数模型更灵活的模型。

## 12.5 课后习题（Homework problems）

## 12.1 合理性检验（A sanity check）

考虑协变量是离散的 $X ~ \in ~ \{ 1 , \ldots , K \}$ 且感兴趣的参数是 $\mu _ { 1 }$ 的情况。在不施加任何模型假设的情况下，估计的倾向得分 $\hat { e } ( X )$ 是接受处理单位的比例，估计的结果均值是在处理条件下，在层 $X = k \ ( k \stackrel { \cdot } { = } 1 , \ldots , K )$ 内结果 $\hat { \bar { Y } } _ { [ k ] 1 } ~ = ~ \hat { E } ( Y ~ \vert ~ Z ~ = ~ 1 , X ~ = ~ k )$ 的样本均值。证明分层估计量、结果回归估计量、IPW 估计量和双重稳健估计量都是相同的。

## 12.2 τ 的双重稳健估计量的另一种形式（An alternative form of the doubly robust estimator for τ）

受 (12.7) 启发，我们得到 $\mu _ { 1 }$ 的双重稳健估计量的另一种形式：

$$
\tilde {\mu} _ {1} ^ {\mathrm{dr2}} = \frac {E \left[ \frac {Z \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X , \alpha)} \right]}{E \left[ \frac {Z}{e (X , \alpha)} \right]} + E \{\mu_ {1} (X, \beta_ {1}) \}.
$$

证明如果 $e ( X , \alpha ) = e ( X )$ 或 $\mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X )$ ，则 $\tilde { \mu } _ { 1 } ^ { \mathrm { d r 2 } } = \mu _ { 1 }$ 。给出用于估计 $\mu_0$ 的类似公式。给出基于这些公式的 $\tau$ 的双重稳健估计量的样本类似形式。注意，这种形式的双重稳健估计量出现在 Robins 等人 (2007) 中。

## 12.3 示例 10.1 的数据分析（Data analysis of Example 10.1）

使用迄今讨论的方法分析数据集 `cps1re74.csv`。

## 12.4 推荐阅读（Recommended reading）

Lunceford 和 Davidian (2004) 对第 11 章和第 12 章讨论的许多方法进行了精彩的回顾和比较。

## 13
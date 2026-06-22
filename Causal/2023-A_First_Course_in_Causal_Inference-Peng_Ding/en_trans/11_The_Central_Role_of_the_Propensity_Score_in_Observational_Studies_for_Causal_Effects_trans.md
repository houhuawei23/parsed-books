# 倾向性得分在观察性研究因果效应中的核心作用（The Central Role of the Propensity Score in Observational Studies for Causal Effects）

Rosenbaum 和 Rubin（1983b）提出了关键概念**倾向性得分（propensity score）**，并讨论了其在观察性研究因果推断中的作用。这是统计学中被引用最多的论文之一，Titterington（2013）将其列为过去100年间发表在 *Biometrika* 上被引用次数第二高的论文。近年来，其被引次数增长非常迅速。

在**独立同分布（IID）** 抽样假设下，每个单元关联四个随机变量：$\{ X , Z , Y ( 1 ) , Y ( 0 ) \}$。根据基本概率规则，我们可以将联合分布分解为：

$$
\operatorname{pr} \{X, Z, Y (1), Y (0) \}
$$

$$
= \operatorname{pr} (X) \times \operatorname{pr} \{Y (1), Y (0) \mid X \} \times \operatorname{pr} \{Z \mid X, Y (1), Y (0) \},
$$

其中 $\mathrm { p r } ( X )$ 是**协变量分布（covariate distribution）**，$\operatorname { p r } \{ Y ( 1 ) , Y ( 0 ) \mid X \}$ 是**结果模型（outcome model）**，而 $\operatorname { p r } \{ Z \mid X , Y ( 1 ) , Y ( 0 ) \}$ 是**处理分配机制（treatment assignment mechanism）**。通常，我们不想对协变量建模，因为它们是发生在处理和结果之前的背景信息。如果我们想超越结果模型，就必须关注处理分配机制，这引出了倾向性得分的定义。

**定义 11.1（倾向性得分）** 定义

$$
e (X, Y (1), Y (0)) = \operatorname{pr} \{Z = 1 \mid X, Y (1), Y (0) \}
$$

为倾向性得分。在**强可忽略性（strong ignorability）** 下，我们有

$$
e (X, Y (1), Y (0)) = \operatorname{pr} \{Z = 1 \mid X, Y (1), Y (0) \} = \operatorname{pr} (Z = 1 \mid X),
$$

因此倾向性得分简化为

$$
e (X) = \operatorname{pr} (Z = 1 \mid X),
$$

即在给定观测协变量条件下接受处理的**条件概率（conditional probability）**。

Rosenbaum 和 Rubin（1983b）使用 $e ( X ) = \mathrm { p r } ( Z = 1 \mid X )$ 作为倾向性得分的定义，因为他们关注的是可忽略性假设下的观察性研究。有时将 $e ( X , Y ( 1 ) , Y ( 0 ) ) = \mathrm { p r } \{ Z =$ $1 \mid X , Y ( 1 ) , Y ( 0 ) \}$ 视为倾向性得分的一般定义是有帮助的，即使可忽略性不成立时也是如此。详见问题 11.1。

遵循 Rosenbaum 和 Rubin（1983b），本章将证明 $e ( X )$ 是在可忽略性假设下观察性研究因果推断中的一个关键量。

## 11.1 倾向性得分作为降维工具（The propensity score as a dimension reduction tool）

### 11.1.1 理论（Theory）

**定理 11.1** 如果 $Z \perp \perp \{ Y ( 1 ) , Y ( 0 ) \} \mid X$，则 $Z \perp \{ Y ( 1 ) , Y ( 0 ) \} \mid e ( X )$。

定理 11.1 指出，如果强可忽略性在给定协变量 X 的条件下成立，那么它在给定标量倾向性得分 $e ( X )$ 的条件下也成立。可忽略性需要对单元的许多背景特征 Z 进行条件化，但定理 11.1 表明，控制倾向性得分 $e ( X )$ 可以消除由协变量 X 引起的所有混杂。原始协变量 X 可以是通用的且具有多个维度，但倾向性得分 $e ( X )$ 是一个介于 0 和 1 之间的一维标量变量。因此，倾向性得分降低了原始协变量的维度，同时仍然保持了可忽略性。作为一个技术统计术语，我们可以将倾向性得分视为一种**降维工具（dimension reduction tool）**。我们首先证明下面的定理 11.1，然后给出倾向性得分降维性质的一个应用。

**定理 11.1 的证明：** 根据**条件独立性（conditional independence）** 的定义，我们需要证明：

$$
\operatorname{pr} \{Z = 1 \mid Y (1), Y (0), e (X) \} = \operatorname{pr} \{Z = 1 \mid e (X) \}. \tag {11.1}
$$

(11.1) 式的左边等于：

$$
\begin{array}{l} \operatorname{pr} \{Z = 1 \mid Y (1), Y (0), e (X) \} \\ = E \{Z \mid Y (1), Y (0), e (X) \} \\ = E \left[ E \{Z \mid Y (1), Y (0), e (X), X \} \mid Y (1), Y (0), e (X) \right] \\ (\text { 塔性质（tower property）; 见 第 A1.1.1 节 }) \\ = E \left[ E \{Z \mid Y (1), Y (0), X \} \mid Y (1), Y (0), e (X) \right] \\ = E \left\{E (Z \mid X) \mid Y (1), Y (0), e (X) \right\} \quad (\text { 强可忽略性（strong ignorability） }) \\ = E \left\{e (X) \mid Y (1), Y (0), e (X) \right\} \\ = e (X). \\ \end{array}
$$

(11.1) 式的右边等于：

$$
\begin{array}{l} \operatorname{pr} \{Z = 1 \mid e (X) \} \\ = E \{Z \mid e (X) \} \\ = E \left[ E \{Z \mid e (X), X \} \mid e (X) \right] \quad (\text { 塔性质（tower property） }) \\ = E \left\{E (Z \mid X) \mid e (X) \right\} \\ = E \left\{e (X) \mid e (X) \right\} \\ = e (X). \\ \end{array}
$$

因此，(11.1) 式的左边等于 (11.1) 式的右边。

![image_10](images/image_10.png)

### 11.1.2 倾向性得分分层（Propensity score stratification）

定理 11.1 启发了一种估计因果效应的简单方法：**倾向性得分分层（propensity score stratification）**。从简单情况开始，我们假设倾向性得分是已知的，并且只取 K 个可能值 $\{ e _ { 1 } , \ldots , e _ { K } \}$，其中 K 远小于样本量 n。定理 11.1 简化为：

$$
Z \bot \{Y (1), Y (0) \} \mid e (X) = e _ {k} \quad (k = 1, \ldots , K).
$$

因此，我们得到了一个**分层随机实验（Stratified Randomized Experiment, SRE）**，即我们在倾向性得分的层内有 K 个独立的**完全随机实验（Completely Randomized Experiments, CREs）**。我们可以像分析以 $e ( X )$ 分层的 SRE 一样分析观察性数据。

一般情况下，倾向性得分是未知的且不是离散的。我们通常拟合一个 $\operatorname { p r } ( Z \ = \ 1 \ | \ X )$ 的统计模型（例如，逻辑模型）来获得估计的倾向性得分 ${ \hat { e } } ( X )$。这个估计的倾向性得分可以取与样本量一样多的值，但我们可以将其离散化以近似上述简单情况。例如，我们可以通过其 K 个分位数将估计的倾向性得分离散化，得到 $\hat { e } ^ { \prime } ( X ) \colon \hat { e } ^ { \prime } ( X _ { i } ) = e _ { k }$，即如果 ${ \hat { e } } ( X _ { i } )$ 介于 ${ \hat { e } } ( X )$ 的 $( k - 1 ) / K$ 分位数和 $k / K$ 分位数之间，则取第 $k / K$ 分位数。然后我们近似有：

$$
Z \bot \{Y (1), Y (0) \} \mid \hat {e} ^ {\prime} (X) = e _ {k} \quad (k = 1, \ldots , K).
$$

因此，我们可以像分析以 $\hat { e } ^ { \prime } ( X )$ 分层的 SRE 一样分析观察性数据。在给定 $\hat { e } ^ { \prime } ( X )$ 的条件下，可忽略性仅近似成立。我们可以进一步使用基于协变量的**回归调整（regression adjustment）** 来消除偏倚并提高效率。具体来说，我们可以在每个层内获得 Lin（2013）的估计量，并通过加权平均构建最终的估计量。

当倾向性得分未知时，我们需要拟合一个统计模型来获得估计的倾向性得分 $\hat { e } ( X )$。这使得最终的估计量依赖于模型设定。然而，倾向性得分分层估计量只需要估计的倾向性得分的正确排序，而不是它们的精确值，这使得它相对于其他方法更为稳健。这种倾向性得分分层的稳健性性质在许多数值例子中出现，但其严格的量化在文献中仍然缺失。

一个重要的实践问题是如何选择 K？如果 K 太小，那么在给定 $\hat { e } ^ { \prime } ( X )$ 的条件下，即使近似地，强可忽略性也不成立。如果 K 太大，那么我们在估计的倾向性得分的每个层内没有足够的单元，并且许多层只有处理组或对照组单元。因此，我们在实践中面临权衡。遵循 Cochran（1968）的启发式方法，Rosenbaum 和 Rubin（1983b）以及 Rosenbaum 和 Rubin（1984）建议 K = 5，这在许多设定下可以消除大量偏倚。然而，在数据集非常大的情况下，倾向性得分分层在固定 K 下会产生有偏的估计量（Lunceford 和 Davidian，2004）。因此，只要每个层有足够的处理组和对照组单元，增加 K 是合理的。Wang 等人（2020）建议了一个激进的选择 K，即分层估计量能够良好定义的最大层数。但这一过程的严格理论尚未完全建立。

另一个重要的实践问题是如何计算基于倾向性得分分层的估计量的标准误？一些研究者以离散化的倾向性得分 $\hat {e} ^ {\prime} (X)$ 为条件，并报告基于 SRE 的标准误。这实际上忽略了估计的倾向性得分中的不确定性。其他研究者通过**自助法（bootstrap）** 对整个过程进行重抽样以考虑全部不确定性。然而，由于该估计量的离散性，自助法的理论仍不清楚。

### 11.1.3 应用（Application）

为了说明倾向性得分分层方法，我重新审视了例 10.3。图 11.1 显示了不同箱数（K = 5, 10, 30）下估计的倾向性得分的直方图。

基于倾向性得分分层，我们可以计算不同 K 值选择（$K \in \{ 5 , 1 0 , 2 0 , 5 0 , 8 0 \}$）的点估计量和标准误如下（使用第 5 章中定义的用于分析 SRE 的函数 NeymanSRE）：

```txt
> pscore = glm(z ~ x, family = binomial)$fitted.values
> n.strata = c(5, 10, 20, 50, 80)
> strat.res = sapply(n.strata, FUN = function(nn){
+    q.pscore = quantile(pscore, (1:(nn-1))/nn)
+    ps.strata = cut(pscore, breaks = c(0,q.pscore,1),
+    labels = 1:nn)
+    Neyman_SRE(z, y, ps.strata))
>
> rownames(strat.res) = c("est", "se")
> colnames(strat.res) = n.strata
> round(strat.res, 3)
5    10    20    50    80
```

$$
\begin{array}{c c c c c c} \text {est} & - 0. 1 1 6 & - 0. 1 7 8 & - 0. 2 0 0 & - 0. 2 6 5 & - 0. 2 0 4 \\ \text {se} & 0. 2 8 3 & 0. 2 8 2 & 0. 2 7 9 & 0. 2 7 2 & \text {NA} \end{array}
$$

将 K 从 5 增加到 50 会减小标准误。然而，我们不能像 K = 80 那样极端，因为在某些只有单个处理组或对照组单元的层中，标准误无法良好定义。上述估计量显示膳食计划对 BMI 有负向但不显著的影响。

我们还可以将上述估计量与三个简单的回归估计量进行比较：未调整任何协变量的估计量、Fisher 估计量和 Lin 估计量。

$$
\begin{array}{c c c c} & \text {naive} & \text {fisher} & \text {lin} \\ \text {est} & 0. 5 3 4 & 0. 0 6 1 & - 0. 0 1 7 \\ \text {se} & 0. 2 2 5 & 0. 2 2 7 & 0. 2 2 6 \end{array}
$$

**朴素均值差（naive difference in means）** 与其他方法差异很大。尽管点估计不同，但两个回归估计量和倾向性得分分层估计量在定性上给出了相同的结果。倾向性得分分层估计量在不同 K 选择下是稳定的。

## 11.2 倾向性得分加权（Propensity score weighting）

## 11.2.1 理论（Theory）

**定理 11.2** 如果 $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$ 且 $0 < e ( X ) < 1$ ，那么

$$
E \{Y (1) \} = E \left\{\frac {Z Y}{e (X)} \right\}, \quad E \{Y (0) \} = E \left\{\frac {(1 - Z) Y}{1 - e (X)} \right\},
$$

并且

$$
\tau = E \{Y (1) - Y (0) \} = E \left\{\frac {Z Y}{e (X)} - \frac {(1 - Z) Y}{1 - e (X)} \right\}.
$$

在证明定理 11.2 之前，有必要注意附加假设 $0 < e ( X ) < 1$ 。这被称为**重叠（overlap）**或**积极性（positivity）条件**。如果对于某些 X 值有 $e ( X ) = 0 { \mathrm { ~ o r ~ } } 1$ ，则定理 11.2 中的公式变为无穷大。由于识别公式是基于**倾向得分加权（propensity score weighting）**的，这并非一个限制。尽管在定理 10.1 中未明确说明，但 (10.5) 中 $\tau$ 的识别公式里的条件期望 $E ( Y \mid Z = 1 , X )$ 和 $E ( Y \mid Z = 0 , X )$ 仅当 $0 < e ( X ) < 1$ 时才有良好定义。重叠条件可视为一个技术性条件，用以确保定理 10.1 和 11.2 中的公式有良好定义。它也可能给观察性研究的因果推断带来一些哲学问题。当单元 i 有 $e ( X _ { i } ) = 1$ 时，我们总是能观测到其在处理下的潜在结果 $Y _ { i } ( 1 )$ ，但永远无法观测到其在控制下的潜在结果 $Y _ { i } ( 0 )$ 。在这种情况下，潜在结果 $Y _ { i } ( 0 )$ 甚至可能没有良好定义，使得单元 i 的因果效应定义变得模糊。King 和 Zeng (2006) 将 $e ( X _ { i } ) = 1$ 时的 $Y _ { i } ( 0 )$ 称为**极端反事实（extreme counterfactual）**，并讨论了它们在因果推断中的危险性。如果单元 i 有 $e ( X _ { i } ) = 0$ ，也会出现类似的问题。

总之， $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} | X$ 需要充分的协变量以确保处理与潜在结果的条件独立性，而 $0 < e ( X ) < 1$ 需要处理在给定协变量条件下存在残差随机性。实际上，Rosenbaum 和 Rubin (1983b) 对**强可忽略性（strong ignorability）**的定义包含了这两个条件。在现代文献中，它们通常被分开陈述。

**定理 11.2 的证明：** 我只证明关于 $E \{ Y ( 1 ) \}$ 的结果，因为关于 $E \{ Y ( 0 ) \}$ 的证明是类似的。我们有

$$
\begin{array}{l} E \left\{\frac {Z Y}{e (X)} \right\} \\ = E \left\{\frac {Z Y (1)}{e (X)} \right\} \\ = E \left[ E \left\{\frac {Z Y (1)}{e (X)} \mid X \right\} \right] \quad (\text { 塔性质 (tower property) }) \\ = E \left[ \frac {1}{e (X)} E \{Z Y (1) \mid X \} \right] \\ = E \left[ \frac {1}{e (X)} E (Z \mid X) E \{Y (1) \mid X \} \right] \quad (\text { 强可忽略性 (strong ignorability) }) \\ = E \left[ \frac {1}{e (X)} e (X) E \{Y (1) \mid X \} \right] \\ = E [ E \{Y (1) \mid X \} ] \\ = E \{Y (1) \}. \\ \end{array}
$$

## 11.2.2 逆倾向得分加权估计量（Inverse propensity score weighting estimators）

定理 11.2 为平均因果效应提供了以下矩估计量：

$$
\hat {\tau} ^ {\mathrm{ht}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {Z _ {i} Y _ {i}}{\hat {e} (X _ {i})} - \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})},
$$

其中 $\hat { e } ( X _ { i } )$ 是估计的倾向得分。这就是**逆倾向得分加权（Inverse Propensity Score Weighting, IPW）估计量**，也称为**霍维茨-汤普森（Horvitz–Thompson, HT）估计量**。Horvitz 和 Thompson (1952) 在抽样调查中提出了它，Rosenbaum (1987a) 将其用于观察性研究的因果推断。

然而，估计量 $\hat { \tau } ^ { \mathrm { h t } }$ 存在许多问题。特别是，它对结果变量的位置变换不具有不变性。例如，如果我们将 $Y _ { i }$ 改变为 $Y _ { i } + c$ （ $c$ 为常数），那么它会变成 $\hat { \tau } ^ { \mathrm { h t } } + c ( \hat { 1 } _ { \mathrm { T } } - \hat { 1 } _ { \mathrm { C } } )$ ，其中

$$
\hat {1} _ {\mathrm{T}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {Z _ {i}}{\hat {e} (X _ {i})}, \quad \hat {1} _ {\mathrm{C}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i})}{1 - \hat {e} (X _ {i})}
$$

是常数 1 的两个不同估计量。我使用 $\hat { 1 } _ { \mathrm { T } }$ 和 $\mathrm { \hat { 1 } _ { C } }$ 这种有趣的符号，是因为使用真实倾向得分时，这两项的期望都是 1；见问题 11.3。通常， $\mathrm { \hat { 1 } _ { T } - \hat { 1 } _ { C } }$ 在有限样本中不为零。由于给每个结果加上一个常数不应改变平均因果效应，该估计量因其对 c 的依赖性而不合理。解决这个问题的一个简单方法是分别用 $\hat { 1 } _ { \mathrm { T } }$ 和 $\hat {1}_C$ 对权重进行归一化，得到以下估计量：

$$
\hat {\tau} ^ {\mathrm{hajek}} = \frac {\sum_ {i = 1} ^ {n} \frac {Z _ {i} Y _ {i}}{\hat {e} (X _ {i})}}{\sum_ {i = 1} ^ {n} \frac {Z _ {i}}{\hat {e} (X _ {i})}} - \frac {\sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})}}{\sum_ {i = 1} ^ {n} \frac {1 - Z _ {i}}{1 - \hat {e} (X _ {i})}}.
$$

这就是**哈杰克（Hajek）估计量**，由 H´ajek (1971) 提出。我们可以验证，Hajek 估计量对位置变换具有不变性，即，如果我们将 $Y _ { i }$ 替换为 $Y _ { i } + c$ ，那么 $\hat { \tau } ^ { \mathrm { h a j e k } }$ 保持不变。此外，许多数值研究发现，在有限样本中 $\hat { \tau } ^ { \mathrm { h a j e k } }$ 比 ${ \hat { \tau } } ^ { \mathrm { h t } }$ 稳定得多。

## 11.2.3 加权的一个问题与因果推断的一个基本问题（A problem of weighting and a fundamental problem of causal inference）

在许多渐近分析中，我们要求一个强重叠条件：

$$
0 <   \alpha_ {\mathrm{L}} \leq e (X) \leq \alpha_ {\mathrm{U}} <   1,
$$

即，真实的倾向得分远离 0 和 1。然而，D’Amour 等人 (2021) 指出，这是一个相当强的假设，尤其是在协变量很多的情况下。第 20 章将详细讨论这个问题。

即使真实倾向得分满足强重叠条件，估计的倾向得分也可能接近 0 或 1。当这种情况发生时，加权估计量会膨胀至无穷大，导致有限样本中极其不稳定的行为。我们可以通过将估计的倾向得分截断为

$$
\max \left[ \alpha_ {\mathrm{L}}, \min \{\hat {e} (X _ {i}), \alpha_ {\mathrm{U}} \} \right],
$$

或者通过剔除 $\hat { e } ( X _ { i } )$ 落在区间 $[ \alpha _ { \mathrm { L } } , \alpha _ { \mathrm { U } } ]$ 之外的观测值来**修剪（trim）**数据。Crump 等人 (2009) 建议 $\alpha _ { \mathrm { L } } = 0 . 1$ 和 $\alpha _ { \mathrm { U } } = 0 . 9$ ，而 Kurth 等人 (2005) 建议 $\alpha _ { \mathrm { L } } ~ = ~ 0 . 0 5$ 和 $\alpha _ { \mathrm { U } } ~ = ~ 0 . 9 5$ 。Yang 和 Ding (2018) 为修剪建立了一些渐近理论。

## 11.2.4 应用（Application）

回顾示例 10.3，我们可以基于对估计倾向得分的不同截断来获得加权估计量。以下结果是两个加权估计量及其**自助法标准误（bootstrap standard errors）**，截断点分别为 (0, 1)、(0.01, 0.99)、(0.05, 0.95) 和 (0.1, 0.9)：

\$ trunc0

$$
\begin{array}{c c c} & \text {HT} & \text {Hajek} \\ \text {est} & - 1. 5 1 6 & - 0. 1 5 6 \\ \text {se} & 0. 4 9 5 & 0. 2 3 8 \end{array}
$$

## 11.3 作为平衡得分的倾向得分（The propensity score as a balancing score）

<table><tr><td colspan="3">$trunc.01</td></tr><tr><td></td><td>HT</td><td>Hajek</td></tr><tr><td>est</td><td>-1.516</td><td>-0.156</td></tr><tr><td>se</td><td>0.464</td><td>0.231</td></tr></table>

<table><tr><td colspan="3">$trunc.05</td></tr><tr><td></td><td>HT</td><td>Hajek</td></tr><tr><td>est</td><td>-1.499</td><td>-0.152</td></tr><tr><td>se</td><td>0.472</td><td>0.248</td></tr></table>

<table><tr><td colspan="3">$trunc.1</td></tr><tr><td></td><td>HT</td><td>Hajek</td></tr><tr><td>est</td><td>-0.713</td><td>-0.054</td></tr><tr><td>se</td><td>0.435</td><td>0.229</td></tr></table>

HT 估计量给出的结果与我们迄今为止讨论的所有其他估计量相去甚远。点估计似乎过大，并且除非我们将估计的倾向得分截断在 (0.1, 0.9)，否则它们呈负显著。这是一个展示 HT 估计量不稳定性的例子。

## 11.3 作为平衡得分的倾向得分（The propensity score as a balancing score）

## 11.3.1 理论（Theory）

**定理 11.3** 倾向得分满足

$$
Z \bot X \mid e (X).
$$

此外，对于任意函数 h(·)，假设 (11.2) 两边的矩存在，我们有

$$
E \left\{\frac {Z h (X)}{e (X)} \right\} = E \left\{\frac {(1 - Z) h (X)}{1 - e (X)} \right\} \tag {11.2}
$$

Rosenbaum 和 Rubin (1983b) 还引入了**平衡得分（balancing score）** $b(X)$ 的概念，它满足 $Z \bot\bot X \mid b(X)$ 。根据定理 11.3，倾向得分是一个平衡得分。定理 11.3 还指出，如果以倾向得分的倒数加权，协变量的任意函数 $h(X)$ 在处理组和对照组中具有相同的均值。

此外，Rosenbaum 和 Rubin (1983b) 证明了倾向得分 $e ( X )$ 是最粗糙的平衡得分，即倾向得分 $e ( X )$ 是任何平衡得分的函数。问题 11.5 提供了更多细节。

**定理 11.3 的证明：** 首先，我们证明 $Z \bot \bot X \mid e ( X )$ ，即

$$
\operatorname{pr} \{Z = 1 \mid X, e (X) \} = \operatorname{pr} \{Z = 1 \mid e (X) \}. \tag {11.3}
$$

按照与定理 11.1 证明类似的步骤，我们可以证明 (11.3) 的左边等于

$$
\operatorname{pr} \{Z = 1 \mid X, e (X) \} = \operatorname{pr} (Z = 1 \mid X) = e (X),
$$

而 (11.3) 的右边等于

$$
\begin{array}{l} \operatorname{pr} \{Z = 1 \mid e (X) \} = E \{Z \mid e (X) \} \\ = E \left[ E \{Z \mid X, e (X) \} \mid e (X) \right] \\ = E \left[ E \{Z \mid X \} \mid e (X) \right] \\ = E \left[ e (X) \mid e (X) \right] \\ = e (X). \\ \end{array}
$$

因此，(11.3) 成立。

其次，我们证明 (11.2)。我们可以使用与定理 11.1 证明类似的步骤。但鉴于定理 11.1，我们有一个更简单的证明。如果我们将 $h ( X )$ 视为一个结果，那么它的两个潜在结果是相同的，并且强可忽略性成立： $Z \bot\bot h(X) \mid X$ 。(11.2) 左右两边之差是 Z 对 $h ( X )$ 的平均因果效应，该效应为零。□

## 11.3.2 协变量平衡检验（Covariate balance check）

定理 11.3 的证明很简单。但定理 11.3 对统计分析有重要的启示。在获取结果数据之前，我们可以检查倾向得分模型是否指定得足够好，以确保数据中的协变量平衡。Rubin (2007) 将此视为观察性研究的设计阶段，Rubin (2008) 认为这可以导致更客观的因果推断，因为设计阶段不涉及结果变量的值。虽然这在实践中是一个有用的建议，但如何量化客观性并不完全清楚。

在**倾向得分分层（propensity score stratification）**中，我们有离散化的估计倾向得分 $\hat { e } ^ { \prime } ( X )$ ，并且近似地有

$$
Z \bot X \mid \hat {e} ^ {\prime} (X) = e _ {k} \quad (k = 1, \ldots , K).
$$

因此，我们可以检查在离散化估计倾向得分的每一层内，处理组和对照组的协变量分布是否相同。

在**倾向得分加权（propensity score weighting）**中，我们可以将 $h ( X )$ 视为一个伪结果，并估计其对 $h ( X )$ 的平均因果效应。由于对 $h ( X )$ 的真实平均因果效应为 0，估计值不应显著异于 0。 $h ( X )$ 的一个典型选择是 X 。

让我们再次回顾示例 10.3。基于 $K = 5$ 的倾向得分分层，除了 FoodStamp 之外的所有协变量在处理组和对照组之间都很好地平衡了。Hajek 估计量也得到类似的结果。图 11.2 显示了平衡检验的结果。

## 11.4 课后习题（Homework Problems）

## 11.1 定理 11.1 的另一个版本

证明

$$
Z \bot \{Y (1), Y (0), X \} \mid e (X, Y (1), Y (0)).
$$

注：此结果意味着

$$
Z \bot \{Y (1), Y (0) \} \mid \{X, e (X, Y (1), Y (0) \}.
$$

Rosenbaum (2020) 和 Rosenbaum 与 Rubin (2023) 指出了这一结果，并将 $e ( X , Y ( 1 ) , Y ( 0 ) )$ 称为**主要未观测协变量（principal unobserved covariate）**。

## 11.2 定理 11.1 的另一个版本

如果对于 $z = 0 , 1$ 有 $Z \bot Y ( z ) \mid X$ ，那么对于 $z = 0 , 1$ 有 $Z \underline { { | | Y ( z ) | } } \mid e ( X )$ 。也就是说，如果可忽略性在给定协变量 X 的条件下成立，那么它在给定标量倾向得分 $e ( X )$ 的条件下也成立。证明这个定理。

## 11.3 IPW 估计量的更多结果

这与第 11.2.2 节中关于 IPW 估计量的讨论有关。

证明

$$
E \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \frac {Z _ {i}}{e (X _ {i})} \right\} = 1, \quad E \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i})}{1 - e (X _ {i})} \right\} = 1.
$$

## 11.4 对 Rosenbaum 和 Rubin (1983a) 的再分析

使用 Rosenbaum 和 Rubin (1983a) 的表 1。如果你感兴趣，可以阅读整篇论文。这是一篇经典论文。但对于这个问题，你只需要表 1。

Rosenbaum 和 Rubin (1983a) 为倾向得分拟合了一个逻辑回归模型，并将数据分层为 5 个子类。由于处理（手术与药物治疗）是二元的，结果也是二元的（改善或未改善），他们用表格来表示数据。

基于此表格，估计平均因果效应，并报告 95% 的置信区间。

## 11.5 平衡得分与倾向得分：更多理论结果

Rosenbaum 和 Rubin (1983b) 将 $b ( X )$ 定义为平衡得分，如果 $Z \bot \bot X \ |$ b(X)。这里，b(X) 可以是一个标量或一个向量。一个明显的平衡得分是 $b ( X ) = X$ ，但如果没有对原始协变量进行任何简化，它并不是一个有用的得分。根据定理 11.3，倾向得分是一个特殊的平衡得分。更有趣的是，Rosenbaum 和 Rubin (1983b) 证明了倾向得分是最粗糙的平衡得分，如下述定理 11.4 所示，该定理将定理 11.3 作为一个特例包含在内。

**定理 11.4** $b ( X )$ 是一个平衡得分当且仅当 b(X) 比 $e ( X )$ 更精细，即存在某个函数 f (·) 使得 $e ( X ) = f ( b ( X ) )$ 。

定理 11.4 与**子组分析（subgroup analysis）**相关。特别是，我们可能不仅对平均因果效应 τ 感兴趣，还对男孩和女孩的子组效应感兴趣。不失一般性，假设 X 的第一个分量是女孩的指示变量，我们有兴趣估计

$$
\tau (x _ {1}) = E \{Y (1) - Y (0) \mid X _ {1} = x _ {1} \}, \quad (x _ {1} = 1, 0).
$$

定理 11.4 表明，在可忽略性下，

$$
Z \bot \{Y (1), Y (0) \} \mid e (X), X _ {1} \tag {11.4}
$$

因为 $b ( X ) = \{ e ( X ) , X _ { 1 } \}$ 比 $e ( X )$ 更精细，因此是一个平衡得分。(11.4) 中的条件独立性确保了在 $X _ { 1 }$ 的每个水平内，给定倾向得分时可忽略性成立。因此，我们可以在 $X _ { 1 }$ 的每个水平内，基于倾向得分进行相同的分析，从而得到两个子组效应的估计值。

基于上述动机，现在证明定理 11.4。

## 11.6 子组效应的一些基础

此问题与问题 11.5 有关，但你可以独立完成。

考虑一个标准的观察性研究，协变量为 $\boldsymbol { X } = ( X _ { 1 } , X _ { 2 } )$ ，其中 $X _ { 1 }$ 表示一个二元的子组指示变量（例如，统计学专业或非统计学专业）， $X _ { 2 }$ 包含其余协变量。感兴趣的参数是子组因果效应

$$
\tau (x _ {1}) = E \{Y (1) - Y (0) \mid X _ {1} = x _ {1} \}, \quad (x _ {1} = 1, 0).
$$

证明

$$
\tau (x _ {1}) = E \left\{\frac {1 (X _ {1} = x _ {1}) Z Y}{e (X)} - \frac {1 (X _ {1} = x _ {1}) (1 - Z) Y}{1 - e (X)} \right\} / \operatorname{pr} (X _ {1} = x _ {1})
$$

并给出 $\tau ( x _ { 1 } )$ 对应的 Horvitz–Thompson 和 Hajek 估计量。

## 11.7 推荐阅读（Recommended reading）

本章的标题与 Rosenbaum 和 Rubin (1983b) 的经典论文标题相同。本章的大部分结果直接来源于他们的原始论文。

Rubin (2007) 和 Rubin (2008) 强调了观察性研究设计阶段对于更客观的因果推断的重要性。
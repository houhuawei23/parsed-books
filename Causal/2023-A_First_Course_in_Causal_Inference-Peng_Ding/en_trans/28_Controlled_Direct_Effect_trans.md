# 控制直接效应（Controlled Direct Effect）

第27章中介分析的表述依赖于**嵌套潜在结果（nested potential outcomes）**，并且从根本上说，某些嵌套潜在结果在任何物理实验中都是不可观测的。如果我们坚持**波普尔科学哲学（Popperian philosophy of science）**，我们只能根据在某些实验中可观测的量来定义因果参数。本章讨论了一种关于存在中间变量的因果推断的替代观点。在这种观点下，我们只能定义直接效应，而无法定义间接效应。

## 28.1 控制直接效应的识别与估计（Identification and estimation of the controlled direct effect）

我们将 $Z$ 和 $M$ 视为两个因素，并定义潜在结果 $Y ( z , m )$，其中 $z = 0 , 1$ 且 $m \in { \mathcal { M } }$。基于这些潜在结果，我们可以定义如下的**控制直接效应（controlled direct effect, CDE）**。

**定义 28.1（CDE）** 定义

$$
\operatorname{CDE} (m) = E \{Y (1, m) - Y (0, m) \}.
$$

根据定义，$\operatorname{CDE}(m)$ 是在中间变量固定为 $m$ 时处理的平均因果效应。参数 $\operatorname{CDE}(m)$ 可以捕捉将中介变量保持在 $m$ 时处理的直接效应。然而，这种表述无法捕捉间接效应。特别地，参数 $E \{ Y ( z , 1 ) - Y ( z , 0 ) \}$ 仅衡量在将处理固定在 $z$ 时中介变量对结果的影响。这不是一个有意义的间接效应定义。

为了识别 $\operatorname{CDE}(m)$，我们需要以下假设，该假设基本要求 $Z$ 和 $M$ 在给定 $X$ 的条件下是联合随机的。

**假设 28.1 序贯可忽略性（Sequential ignorability）** 要求

$$
Z \bot Y (z, m) \mid X, \quad M \bot Y (z, m) \mid (Z, X)
$$

或者等价地，

$$
(Z, M) \bot Y (z, m) \mid X.
$$

我将重点关注 $Z$ 和 $M$ 为二元变量的情况。从数学上讲，我们可以将此问题视为一个具有四个处理水平的观察性研究

$$
(z, m) \in \{(0, 0), (0, 1), (1, 0), (1, 1) \}.
$$

以下定理将二元处理观察性研究的结果推广，基于**结果回归（outcome regression）**、**逆概率加权（inverse probability weighting）**和**双重稳健估计（doubly robust estimation）**来识别

$$
\mu_ {z m} = E \{Y (z, m) \}.
$$

定义

$$
\mu_ {z m} (x) = E (Y \mid Z = z, M = m, X = x)
$$

为在给定处理、中介变量和协变量条件下的结果均值。定义

$$
e _ {z m} (x) = \operatorname * {p r} (Z = z, M = m \mid X = x) = \operatorname * {p r} (Z = z \mid X = x) \operatorname * {p r} (M = m \mid Z = z, X = x)
$$

为在给定协变量条件下 $Z$ 和 $M$ 联合值的概率。

**定理 28.1** 在假设 28.1 下，我们有

$$
\mu_ {z m} = E \{\mu_ {z m} (X) \}
$$

或

$$
\mu_ {z m} = E \left\{\frac {I (Z = z , M = m) Y}{e _ {z m} (X)} \right\}.
$$

此外，基于工作模型 $e _ { z m } ( X , \alpha )$ 和 $\mu _ { z m } ( X , \beta )$，我们有以下双重稳健公式

$$
\mu_ {z m} ^ {\mathrm{dr}} = E \{\mu_ {z m} (X, \beta) \} + E \left[ \frac {I (Z = z , M = m) \{Y - \mu_ {z m} (X , \beta) \}}{e _ {z m} (X , \alpha)} \right],
$$

其中，如果 $e _ { z m } ( X , \alpha ) = e _ { z m } ( X )$ 或 $\mu _ { z m } ( X , \beta ) = \mu _ { z m } ( X )$，则该公式等于 $\mu _ { z m }$。

定理 28.1 的证明与标准无混杂观察性研究的证明类似。问题 28.2 给出了一个一般性结果。基于结果均值模型，我们可以得到 $\mu _ { z m } ( x )$ 的估计 ${ \hat { \mu } } _ { z m } ( x )$。基于处理模型，我们可以得到 $\operatorname { p r } ( Z = z \mid X = x )$ 的估计 $\hat { e } _ { z } ( x )$；基于中间变量模型，我们可以得到 $\operatorname { p r } ( M = m \mid Z = z , X = x )$ 的估计 $\hat { e } _ { m } ( z , x )$。然后，我们可以通过以下方式估计 $\mu _ { z m }$：

通过结果回归

$$
\hat {\mu} _ {z m} ^ {\mathrm{reg}} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\mu} _ {z m} (X _ {i}),
$$

通过逆概率加权

$$
\begin{array}{l} \hat {\mu} _ {z m} ^ {\mathrm{ht}} = n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {I (Z _ {i} = z , M _ {i} = m) Y _ {i}}{\hat {e} _ {z} (X _ {i}) \hat {e} _ {m} (z , X _ {i})}, \\ \hat {\mu} _ {z m} ^ {\mathrm{haj}} = \sum_ {i = 1} ^ {n} \frac {I (Z _ {i} = z , M _ {i} = m) Y _ {i}}{\hat {e} _ {z} (X _ {i}) \hat {e} _ {m} (z , X _ {i})} \bigg / \sum_ {i = 1} ^ {n} \frac {I (Z _ {i} = z , M _ {i} = m)}{\hat {e} _ {z} (X _ {i}) \hat {e} _ {m} (z , X _ {i})}, \\ \end{array}
$$

或通过增强逆概率加权

$$
\hat {\mu} _ {z m} ^ {\mathrm{dr}} = \hat {\mu} _ {z m} ^ {\mathrm{reg}} + n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {I (Z _ {i} = z , M _ {i} = m) \{Y _ {i} - \hat {\mu} _ {z m} (X _ {i}) \}}{\hat {e} _ {z} (X _ {i}) \hat {e} _ {m} (z , X _ {i})}.
$$

然后，我们可以通过 $\hat { \mu } _ { 1 m } - \hat { \mu } _ { 0 m }$ 来估计 $\mathrm { C D E } ( m )$，并使用**自助法（bootstrap）**来近似标准误差。

如果我们愿意假设一个线性结果模型，那么控制直接效应就简化为处理的系数。下面的示例 28.1 给出了详细信息。

**示例 28.1** 在假设 28.1 和一个线性结果模型下，

$$
E (Y \mid Z, M, X) = \theta_ {0} + \theta_ {1} Z + \theta_ {2} M + \theta_ {4} ^ {\mathsf {T}} X,
$$

我们可以证明 $\mathrm { C D E } ( m )$ 等于系数 $\theta _ { 1 }$，这与**Baron–Kenny 方法（Baron–Kenny method）**中的**自然直接效应（natural direct effect）**一致。我将证明留作问题 28.3。

## 28.2 讨论（Discussion）

控制直接效应的表述不涉及嵌套或先验的反事实潜在结果，其识别也不要求**跨世界反事实独立性假设（cross-world counterfactual independence assumption）**。参数 $\operatorname{CDE}(m)$ 可以捕捉将中介变量保持在 $m$ 时处理的直接效应。然而，这种表述无法捕捉间接效应。我将中间变量的因果框架总结如下。

| 章节 | 框架 | 直接效应 | 间接效应 |
|------|------|----------|----------|
| 26 | 主分层（principal stratification） | $\tau(1,1), \tau(0,0)$ | ? |
| 27 | 中介分析（mediation analysis） | NDE | NIE |
| 29 | 控制直接效应（controlled direct effect） | $\operatorname{CDE}(m)$ | ? |

中介分析框架可以将总效应分解为自然直接效应和自然间接效应，但它需要嵌套潜在结果和跨世界独立性。主分层框架和控制直接效应框架无法定义间接效应，但它们不涉及嵌套潜在结果和跨世界独立性。此外，主分层框架不一定要求 $M$ 位于从处理到结果的因果通路上。但其识别和估计涉及解混混合分布，这在统计学中是一项艰巨的任务。

## 28.3 课后习题（Homework problems）

## 28.1 CDE 与 NDE（cde and nde）

证明在跨世界独立性 $Y ( z , m ) \bot M ( z ^ { \prime } ) \mid X$（对于所有 $z , z ^ { \prime }$ 和 $m$）下，条件控制直接效应 $\operatorname{CDE} ( m \mid x ) = E \{ Y ( 1 , m ) - Y ( 0 , m ) \mid X = x \}$ 和 $\operatorname { N D E } ( x ) = E \{ Y ( 1 , M _ { 0 } ) - Y ( 0 , M _ { 0 } ) \mid X = x \}$ 具有以下关系：

$$
\mathrm{NDE} (x) = E \{\mathrm{CDE} (M _ {0} \mid x) \},
$$

对于离散的 $M$，这简化为

$$
\mathrm{NDE} (x) = \sum_ {m} \mathrm{CDE} (m \mid x) \mathrm{pr} (M _ {0} = m \mid X = x).
$$

在没有跨世界独立性的情况下，这个关系是否仍然普遍成立？

## 28.2 具有多值处理的观察性研究（Observational studies with a multi-valued treatment）

定理 28.1 是以下关于具有多处理水平的无混杂观察性研究定理的一个特例（Imai and Van Dyk, 2004; Cattaneo, 2010）。下面，我阐述一般问题和定理。

考虑一个具有多值处理 $Z \in \mathbf { \Sigma } \{ 1 , \ldots , K \}$、协变量 $X$ 和结果 $Y$ 的观察性研究。单元 $i$ 有 $K$ 个潜在结果 $Y _ { i } ( 1 ) , \ldots , Y _ { i } ( K )$，对应于 $K$ 个处理水平。因果效应可以定义为潜在结果之间的比较。一般来说，我们可以根据潜在结果的对比来定义因果效应：

$$
\tau_ {c} = \sum_ {k = 1} ^ {K} c _ {k} E \{Y (k) \}
$$

其中 $\textstyle \sum _ { k = 1 } ^ { K } c _ { k } = 0$。成对比较的标准选择是

$$
\tau_ {k, k ^ {\prime}} = E \{Y (k) - Y (k ^ {\prime}) \}.
$$

因此，关键在于基于 $( Z _ { i } , X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ 的 IID 数据，在以下可忽略性假设下识别和估计潜在结果的均值 $\mu _ { k } = E \{ Y ( k ) \}$。

**假设 28.2** $Z \bot \bot \{ Y ( 1 ) , \dots , Y ( K ) \} \mid X$。

定义**广义倾向得分（generalized propensity score）**为

$$
e _ {k} (X) = \operatorname{pr} (Z = k \mid X),
$$

定义条件结果均值为

$$
\mu_ {k} (X) = E (Y \mid Z = k, X)
$$

其中 $k = 1 , \ldots , K$。我们有以下定理。

**定理 28.2** 在假设 28.2 下，我们有

$$
\mu_ {k} = E \{\mu_ {k} (X) \}
$$

或

$$
\mu_ {k} = E \left\{\frac {I (Z = k) Y}{e _ {k} (X)} \right\}.
$$

此外，基于工作模型 $e _ { k } ( X , \alpha )$ 和 $\mu _ { k } ( X , { \boldsymbol { \beta } } )$，我们有以下双重稳健公式

$$
\mu_ {k} ^ {\mathrm{dr}} = E \{\mu_ {k} (X, \beta) \} + E \left[ \frac {I (Z = k) \{Y - \mu_ {k} (X , \beta) \}}{e _ {k} (X , \alpha)} \right],
$$

其中，如果 $e _ { k } ( X , \alpha ) = e _ { k } ( X )$ 或 $\mu _ { k } ( X , \beta ) = \mu _ { k } ( X )$，则该公式等于 $\mu _ { k }$。

证明定理 28.2。

注：如果我们将定理 28.1 中的 $( Z , M )$ 视为一个具有四个水平的处理，那么定理 28.1 是定理 28.2 的一个特例。$\mathrm { C D E } ( m )$ 是 $\tau _ { c }$ 的一个特例。

## 28.3 线性结果模型中的 CDE（cde in the linear outcome model）

证明在假设 28.1 下，如果 $E ( Y \mid Z , M , X ) = \theta _ { 0 } + \theta _ { 1 } Z + \theta _ { 2 } M + \theta _ { 4 } ^ { \mathsf { T } } X$，那么对于所有 $m$，有

$$
\mathrm{CDE} (m) = \theta_ {1}
$$

如果 $E ( Y \mid Z , M , X ) = \theta _ { 0 } + \theta _ { 1 } Z + \theta _ { 2 } M + \theta _ { 3 } Z M + \theta _ { 4 } ^ { \mathsf { T } } X$，那么

$$
\operatorname{CDE} (m) = \theta_ {1} + \theta_ {3} m.
$$

## 28.4 Logit 结果模型中的 CDE（cde in the logit outcome model）

证明对于二元结果，在假设 28.1 下，如果

$$
\operatorname{logit} \left\{\operatorname * {p r} (Y = 1 \mid Z, M, X) \right\} = \theta_ {0} + \theta_ {1} Z + \theta_ {2} M + \theta_ {4} ^ {\mathsf {T}} X,
$$

那么

$$
\operatorname{CDE} (m) = E \{\expit (\theta_ {0} + \theta_ {1} + \theta_ {2} m + \theta_ {4} ^ {\mathsf {T}} X) - \expit (\theta_ {0} + \theta_ {2} m + \theta_ {4} ^ {\mathsf {T}} X) \};
$$

如果

$$
\operatorname{logit} \left\{\operatorname{pr} (Y = 1 \mid Z, M, X) \right\} = \theta_ {0} + \theta_ {1} Z + \theta_ {2} M + \theta_ {3} Z M + \theta_ {4} ^ {\mathsf {T}} X,
$$

那么

$$
\operatorname{CDE} (m) = E \{\expit (\theta_ {0} + \theta_ {1} + \theta_ {2} m + \theta_ {3} m + \theta_ {4} ^ {\mathsf {T}} X) - \expit (\theta_ {0} + \theta_ {2} m + \theta_ {4} ^ {\mathsf {T}} X) \}.
$$

## 28.5 推荐阅读（Recommended reading）

- Nguyen et al. (2021) 对第27章和第29章的主题进行了友好的综述。
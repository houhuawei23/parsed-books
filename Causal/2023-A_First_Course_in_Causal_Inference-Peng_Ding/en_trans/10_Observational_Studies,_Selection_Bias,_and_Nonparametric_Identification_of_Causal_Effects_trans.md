# 观测研究、选择偏差与因果效应的非参数识别（Observational Studies, Selection Bias, and Nonparametric Identification of Causal Effects）

Cochran（1965）总结了**观测研究（observational studies）**的两个共同特征：

1. 目标是阐明因果关系；
2. 无法使用受控实验。

第一个特征与第二部分讨论的**随机化实验（randomized experiments）**相同，但第二个特征与随机化实验有根本性差异。

Dorn（1953）建议，观测研究的规划者应始终问自己以下问题：

如果可能通过受控实验进行研究，该研究将如何开展？

遵循 Dorn（1953）的建议总是有益的，因为**潜在结果框架（potential outcomes framework）**与实验（无论是真实实验还是思想实验）具有内在联系。本书第三部分将讨论基于观测研究的因果推断。它将阐明观测研究与随机化实验之间的根本差异。尽管如此，许多基于观测研究的因果推断思想与基于随机化实验的思想有着深刻的联系。

## 10.1 激励性示例（Motivating Examples）

**示例 10.1（职业培训项目）** LaLonde（1986）研究了职业培训项目对收入的因果效应。他比较了基于随机化实验的结果与基于观测研究的结果。我们之前使用过实验数据，即 **Matching** 包中的 `lalonde` 数据集；我们还在问题 1.3 中使用过观测对照数据 `cps1re74.csv`。LaLonde（1986）发现，许多用于观测研究的传统计量经济学方法给出的估计值与基于实验数据的估计值差异很大。Dehejia 和 Wahba（1999）使用基于因果推断的方法重新分析了这些数据，发现这些方法能够恢复实验的黄金标准。此后，这成为观测研究中因果推断的经典示例。

**示例 10.2（吸烟与同型半胱氨酸）** Bazzano 等人（2003）基于 2005–2006 年**国家健康与营养调查（National Health and Nutrition Examination Survey, NHANES）**的数据，比较了每日吸烟者和从不吸烟者的同型半胱氨酸水平。Rosenbaum（2018）将该数据记录为 **senstrat** 包中的 `homocyst`。该数据集包含以下重要协变量：

- `female`：1=女性，0=男性
- `age3`：三个年龄类别：20–39，40–50，≥60
- `ed3`：三个教育类别：< 高中，高中，部分大学
- `bmi3`：三个 BMI 类别：<30，[30,35)，≥35
- `pov2`：TRUE=收入至少为贫困线的两倍，FALSE=否则

**示例 10.3（学校餐计划与身体质量指数）** Chan 等人（2016）使用 NHANES 2007–2008 数据的一个子样本，研究参与学校餐计划是否导致学龄儿童 BMI 增加。他们将该数据记录为 **ATE** 包中的 `nhanesbmi`。该数据集包含以下重要协变量：

- `age`：年龄
- `ChildSex`：性别（1：男性，0：女性）
- `black`：种族（1：黑人，0：否则）
- `mexam`：种族（1：西班牙裔，0：否则）
- `pir200_plus`：家庭收入高于联邦贫困线 200%
- `WIC`：参与特殊补充营养计划
- `Food_Stamp`：参与食品券计划
- `fsdchbi`：儿童食品安全
- `AnyIns`：任何保险
- `RefSex`：成人受访者性别（1：男性，0：女性）
- `RefAge`：成人受访者年龄

## 10.2 潜在结果框架下的因果效应与选择偏差（Causal effects and selection bias under the potential outcomes framework）

对于个体 $i ( i = 1 , \ldots , n )$ ，我们有预处理协变量 $X _ { i }$ 、一个二元处理指标 $Z _ { i }$ 和一个观测结果 $Y _ { i }$ ，其中包含两个潜在结果：处理组下的 $Y _ { i } ( 1 )$ 和控制组下的 $Y _ { i } ( 0 )$ 。为简单起见，我们假设

$$
\{X _ {i}, Z _ {i}, Y _ {i} (1), Y _ {i} (0) \} _ {i = 1} ^ {n} \stackrel {{\text {IID}}} {{\sim}} \{X, Z, Y (1), Y (0) \}.
$$

## 10.2 潜在结果框架下的因果效应与选择偏差 129

因此，对于依赖于该总体的量，我们可以去掉下标 $i$ 。关注的因果效应包括**平均因果效应（average causal effect）**

$$
\tau = E \{Y (1) - Y (0) \},
$$

**处理组上的平均因果效应（average causal effect on the treated units）**

$$
\tau_ {\mathrm{T}} = E \{Y (1) - Y (0) \mid Z = 1 \},
$$

以及**控制组上的平均因果效应（average causal effect on the control units）**：

$$
\tau_ {\mathrm{C}} = E \{Y (1) - Y (0) \mid Z = 0 \}.
$$

由期望的线性性质，我们有

$$
\begin{array}{l} \tau_ {\mathrm{T}} = E \{Y (1) \mid Z = 1 \} - E \{Y (0) \mid Z = 1 \} \\ = E (Y \mid Z = 1) - E \{Y (0) \mid Z = 1 \} \\ \end{array}
$$

和

$$
\begin{array}{l} \tau_ {\mathrm{C}} = E \{Y (1) \mid Z = 0 \} - E \{Y (0) \mid Z = 0 \} \\ = E \{Y (1) \mid Z = 0 \} - E (Y \mid Z = 0). \\ \end{array}
$$

在上述 $\tau _ { \mathrm { T } }$ 和 $\tau _ { \mathrm { C } }$ 的两个公式中， $E ( Y \mid Z = 1 )$ 和 $E ( Y \mid$ $Z = 0 )$ 可直接从数据中观测，但 $E \{ Y ( 0 ) \mid Z =$ $1 \}$ 和 $E \{ Y ( 1 ) \mid Z = 0 \}$ 不可观测。后两者是**反事实（counterfactuals）**，因为它们对应于与实际接受处理相反的处理水平的潜在结果的均值。

**简单均值差（simple difference in means）**，也称为**表面因果效应（prima facie causal effect）**，

$$
\begin{array}{l} \tau_ {\mathrm{PF}} = E (Y \mid Z = 1) - E (Y \mid Z = 0) \\ = E \{Y (1) \mid Z = 1 \} - E \{Y (0) \mid Z = 0 \} \\ \end{array}
$$

通常对上述定义的因果效应存在偏差。例如，

$$
\tau_ {\mathrm{PF}} - \tau_ {\mathrm{T}} = E \{Y (0) \mid Z = 1 \} - E \{Y (0) \mid Z = 0 \}
$$

和

$$
\tau_ {\mathrm{PF}} - \tau_ {\mathrm{C}} = E \{Y (1) \mid Z = 1 \} - E \{Y (1) \mid Z = 0 \}
$$

通常不为零，它们量化了**选择偏差（selection bias）**。它们衡量了处理组和控制组之间潜在结果均值的差异。

为什么随机化如此重要？Rubin（1978）首次使用潜在结果来量化随机化的益处。我们在第 9 章中使用了以下事实：

$$
Z \bot \{Y (1), Y (0) \} \tag {10.1}
$$

在完全随机化实验（CRE）中成立，这意味着选择偏差项均为零：

$$
\tau_ {\mathrm{PF}} - \tau_ {\mathrm{T}} = E \{Y (0) \mid Z = 1 \} - E \{Y (0) \mid Z = 0 \} = 0
$$

和

$$
\tau_ {\mathrm{PF}} - \tau_ {\mathrm{C}} = E \{Y (1) \mid Z = 1 \} - E \{Y (1) \mid Z = 0 \} = 0.
$$

因此，在完全随机化（10.1）下，

$$
\tau = \tau_ {\mathrm{T}} = \tau_ {\mathrm{C}} = \tau_ {\mathrm{PF}}.
$$

从上述讨论可知，随机化的根本好处在于平衡处理组和控制组之间潜在结果的分布，这比平衡观测协变量的分布更为重要。

如果没有随机化，选择偏差项可能任意大，特别是对于无界结果。这凸显了基于观测研究进行因果推断的根本困难。

## 10.3 非参数识别的充分条件（Sufficient conditions for nonparametric identification）

### 10.3.1 识别（Identification）

基于观测研究的因果推断具有挑战性。它依赖于强假设。一种策略是利用预处理协变量的信息，并假设在给定观测协变量 $X$ 的条件下，选择偏差项为零，即

$$
E \{Y (0) \mid Z = 1, X \} = E \{Y (0) \mid Z = 0, X \}, \tag {10.2}
$$

$$
E \{Y (1) \mid Z = 1, X \} = E \{Y (1) \mid Z = 0, X \}. \tag {10.3}
$$

（10.2）和（10.3）中的假设表明，处理组和控制组之间潜在结果均值的差异完全源于观测协变量的差异。因此，在给定相同协变量值的情况下，潜在结果在处理组和控制组中具有相同的均值。从数学上讲，（10.2）和（10.3）确保了效应的条件版本是相同的：

$$
\tau (X) = \tau_ {\mathrm{T}} (X) = \tau_ {\mathrm{C}} (X) = \tau_ {\mathrm{PF}} (X),
$$

其中

$$
\begin{array}{l} \tau (X) = E \{Y (1) - Y (0) \mid X \}, \\ \tau_ {\mathrm{T}} (X) = E \{Y (1) - Y (0) \mid Z = 1, X \}, \\ \tau_ {\mathrm{C}} (X) = E \{Y (1) - Y (0) \mid Z = 0, X \}, \\ \tau_ {\mathrm{PF}} (X) = E (Y \mid Z = 1, X) - E (Y \mid Z = 0, X). \\ \end{array}
$$

特别地， $\tau ( X )$ 通常被称为**条件平均因果效应（conditional average causal effect）**。

本章的一个关键结果是，在（10.2）和（10.3）下，平均因果效应 $\tau$ 是**非参数可识别（nonparametrically identifiable）**的。**非参数可识别性（nonparametric identifiability）**的概念在经典统计学中并不常见，但它对于基于观测研究的因果推断至关重要。

**定义 10.1（识别）** 如果一个参数 $\theta$ 可以在某些模型假设下写成观测数据分布的函数，则称其为可识别的。如果一个参数 $\theta$ 可以在没有任何参数模型假设的情况下写成观测数据分布的函数，则称其为**非参数可识别的**。

定义 10.1 目前过于抽象。我将在后续章节中使用更具体的例子来说明其含义。在标准统计问题中，它常常被忽略。例如，如果我们有 $Y _ { i }$ 的独立同分布（IID）样本，则均值 $\theta = E ( Y )$ 是非参数可识别的；如果我们有成对 $( X _ { i } , Y _ { i } )$ 的独立同分布样本，则皮尔逊相关系数 $\theta := \mathrm{corr}(X, Y)$ 是非参数可识别的。在这些例子中，参数自动是非参数可识别的。然而，定义 10.1 在基于观测研究的因果推断中是基础性的。特别地，关注的参数 $\tau = E \{ Y ( 1 ) - Y ( 0 ) \}$ 依赖于某些未观测的随机变量，因此它是否基于观测数据非参数可识别并不明确。在（10.2）和（10.3）的假设下，它是非参数可识别的，具体细节如下。

由于 $\tau _ { \mathrm { P F } } ( X )$ 仅依赖于可观测变量，根据定义它是非参数可识别的。此外，（10.2）和（10.3）确保三个因果效应与 $\tau _ { \mathrm { P F } } ( X )$ 相同，因此 $\tau ( X )$ 、 $\tau _ { \mathrm { T } } ( X )$ 和 $\tau _ { \mathrm { C } } ( X )$ 都是非参数可识别的。因此，根据**全期望定律（law of total expectation）**，在（10.2）和（10.3）下，无条件版本也是非参数可识别的：

$$
\tau = E \{\tau (X) \}, \quad \tau_ {\mathrm{T}} = E \{\tau_ {\mathrm{T}} (X) | Z = 1 \}, \quad \tau_ {\mathrm{C}} = E \{\tau_ {\mathrm{C}} (X) | Z = 0 \}.
$$

从现在起，除非另有说明，我们关注 $\tau$ 。以下定理总结了 $\tau$ 的识别公式。

**定理 10.1** 在（10.2）和（10.3）下，平均因果效应 $\tau$ 由以下公式识别：

$$
\tau = E \{\tau (X) \} \tag {10.4}
$$

$$
= E \{E (Y \mid Z = 1, X) - E (Y \mid Z = 0, X) \} \tag {10.5}
$$

$$
= \int \{E (Y \mid Z = 1, X = x) - E (Y \mid Z = 0, X = x) \} F (\mathrm{d} x). \tag {10.6}
$$

公式（10.5）由 Rosenbaum 和 Rubin（1983b）正式建立，Robins 也称之为 **g-公式（g-formula）**（参见 Hernán 和 Robins，2020）。

对于离散协变量，我们可以将定理 10.1 中的识别公式写为

$$
\begin{array}{l} \tau = \sum_ {x} E (Y \mid Z = 1, X = x) \mathrm{pr} (X = x) \\ - \sum_ {x} E (Y \mid Z = 0, X = x) \mathrm{pr} (X = x), \tag {10.7} \\ \end{array}
$$

而简单均值差也可以根据全概率定律写为

$$
\begin{array}{l} \tau_ {\mathrm{PF}} = \sum_ {x} E (Y \mid Z = 1, X = x) \mathrm{pr} (X = x \mid Z = 1) \\ - \sum_ {x} E (Y \mid Z = 0, X = x) \mathrm{pr} (X = x \mid Z = 0) \tag {10.8} \\ \end{array}
$$

比较（10.7）和（10.8），我们可以看到，虽然两个公式都比较了条件期望 $E ( Y \mid Z =$ $1 , X = x )$ 和 $E ( Y \mid Z = 0 , X = x )$ ，但它们对协变量的分布进行了不同的平均。因果参数 $\tau$ 在协变量的共同分布上对条件期望取平均，而均值差 $\tau_{\mathrm{PF}}$ 在处理组和控制组中协变量的两个不同分布上对条件期望取平均。

通常，我们施加一个更强的假设：

$$
Y (z) \perp \perp Z \mid X \quad (z = 0, 1). \tag {10.9}
$$

这个假设有许多名称：

1. **可忽略性（ignorability）**，源于 Rubin（1978）；
2. **无混杂性（unconfoundedness）**，在流行病学家中流行；
3. **基于可观测变量的选择（selection on observables）**，在社会科学家中间流行；
4. **条件独立性（conditional independence）**，仅是对假设中符号的描述。

有时，我们会施加一个更强的假设

$$
\{Y (1), Y (0) \} \perp Z \mid X \tag {10.10}
$$

这被称为**强可忽略性（strong ignorability）**（Rosenbaum 和 Rubin，1983b）。如果关注的参数是 $\tau$ ，那么更强的假设（10.9）和（10.10）只是为了符号上的简便而施加的。在这种情况下，它们不是必需的。然而，如果关注的参数是其他尺度上的因果效应（例如，分布、分位数或结果的某种变换），则不能放宽这些假设。强可忽略性假设要求潜在结果向量在给定协变量的条件下独立于处理，但可忽略性假设仅要求每个潜在结果在给定协变量的条件下独立于处理。前者比后者更强。

<!-- 脚注 -->

- 例如，$Y _ { i } ( 1 )$ 在 $( 1 , X _ { i } )$ 上的线性投影为 $\alpha _ { 1 } + \beta _ { 1 } X _ { i }$ ，其中
- $( \alpha _ { 1 } , \beta _ { 1 } ) = \arg \operatorname* { m i n } _ { a , b } \sum _ { i = 1 } ^ { n } \{ Y _ { i } ( 1 ) - a - b ^ { \mathsf { T } } X _ { i } \} ^ { 2 } .$

<!-- 脚注结束 -->

<!-- 脚注 -->

- 在没有协变量的情况下，HC2 校正产生的方差估计量与 Neyman（1923）的经典估计量相同。为保持一致，我们也可以将 HC2 校正用于 Lin（2013）的协变量调整估计量。当协变量数量相对于样本量较小且协变量不包含异常值时，EHW 标准误的变体与原始版本表现相似。当协变量数量相对于样本量较大或协变量包含异常值时，变体可以优于

<!-- 脚注结束 -->

<!-- 脚注 -->

- 原始版本。在这些情况下，Lei 和 Ding（2021）建议使用 EHW 标准误的 HC3 变体。有关 EHW 标准误的更多详细信息，请参见附录 A2。

<!-- 脚注结束 -->

<!-- 脚注 -->

- Butler（1969）在略有不同的框架下提出了该检验统计量。给定来自分布 $F ( y )$ 的 $\big ( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } \big )$ 的独立同分布样本，如果它们围绕 0 对称分布，则
- $F ( t ) = \mathrm { p r } ( \hat { \tau } _ { i } \le t ) = \mathrm { p r } ( - \hat { \tau } _ { i } \le t ) = 1 - \mathrm { p r } ( \hat { \tau } _ { i } < - t ) = 1 - F ( - t - ) .$
- 因此，$\hat { F } ( t ) + \hat { F } ( - t - ) - 1$ 衡量了与对称性零假设的偏差，这激发了 $D$ 的定义。Kolmogorov–Smirnov 型统计量的一种朴素定义是比较示例 3.4 中处理组和控制组结果的**经验分布（empirical distributions）**。使用该定义，我们实际上破坏了配对。虽然它仍然可以用于最大排列效应（MPE）的**费希尔随机化检验（FRT）**，但它没有捕捉到实验的配对结构。

<!-- 脚注结束 -->

<!-- 脚注 -->

- 在因果推断中，我们说一个参数是**非参数可识别的**，如果它可以在不施加进一步参数假设的情况下由观测变量的分布确定。

<!-- 脚注结束 -->

后者。然而，它们的差异是技术性的，纯粹属于概率论兴趣；参见问题 10.4。在大多数合理的统计模型中，它们是相同的；参见下面的第 10.3.2 节。在本书中，我们将不区分它们，并简单使用**可忽略性**来指代两者。

### 10.3.2 假设的合理性（Plausibility of the assumption）

基于观测研究进行因果推断的一个基本问题是**可忽略性假设（ignorability assumption）**的合理性。上述讨论可能显得过于数学化，因为可忽略性假设是确保平均因果效应非参数识别的充分条件。其科学含义是什么？直观上，它排除了所有同时影响处理和结果的未测量协变量。这些处理和结果的"共同原因"被称为**混杂因素（confounders）**。这就是为什么可忽略性假设也被称为**无混杂性假设（unconfoundedness assumption）**。从数学上讲，我们可以基于结果的数据生成过程来解释可忽略性假设。如果

$$
\begin{array}{l} Y (1) = f _ {1} (X, V _ {1}), \\ Y (0) = f _ {0} (X, V _ {0}), \\ Z = 1 \{g (X, V) \geq 0 \} \\ \end{array}
$$

且 $( V _ { 1 } , V _ { 0 } ) \bot \bot V$ ，那么（10.9）和（10.10）成立。在上述数据生成过程中，处理和结果的"共同原因" $X$ 都被观测到，剩余的随机成分是独立的。如果数据生成过程变为

$$
\begin{array}{l} Y (1) = f _ {1} (X, U, V _ {1}), \\ Y (0) = f _ {0} (X, U, V _ {0}), \\ Z = 1 \{g (X, U, V) \geq 0 \} \\ \end{array}
$$

且 $( V _ { 1 } , V _ { 0 } ) \bot \bot V$ ，那么（10.9）或（10.10）通常不成立。未测量的"共同原因" $U$ 导致处理与潜在结果之间产生依赖关系，即使条件于观测协变量 $X$ 也是如此。如果我们无法获得 $U$ 并且仅基于 $( Z , X , Y )$ 分析数据，那么最终的估计量通常会对因果参数产生偏差。这种类型的偏差在计量经济学中被称为**遗漏变量偏差（omitted variable bias）**。

如果我们观测到一组丰富的、同时影响处理和结果的协变量 $X$ ，可忽略性假设可能是合理的。我从这个假设开始，在本书第三部分讨论识别和估计策略。然而，它在根本上是不可检验的。我们可以基于科学背景知识来证明其合理性，但我们通常不确定它是否成立。本书的第四部分和第五部分将讨论当这个假设不合理时的其他策略。

## 10.4 两种简单估计策略及其局限性（Two simple estimation strategies and their limitations）

## 10.4.1 基于离散协变量的分层或标准化（Stratification or standardization based on discrete covariates）

如果协变量 $X _ { i } \in \{ 1 , \ldots , K \}$ 是离散的，那么**可忽略性（ignorability）** (10.9) 可表述为：

$$
Y (z) \bot Z \mid X = k \quad (z = 0, 1; k = 1, \dots , K),
$$

这本质上假设该观察性研究是一个**简单随机实验（Simple Randomized Experiment, SRE）**。因此，我们可以使用估计量：

$$
\hat {\tau} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \left\{\hat {\bar {Y}} _ {[ k ]} (1) - \hat {\bar {Y}} _ {[ k ]} (0) \right\},
$$

这与第 5 章讨论的分层或**事后分层估计量（post-stratified estimator）**相同。

该方法在实践中仍被广泛使用。示例 10.2 包含了离散协变量，我将分析工作留至问题 10.3。然而，在实施该方法时存在几个明显的困难。首先，它适用于 $K$ 较小的情况。对于较大的 $K$，很可能许多层存在 $n _ { [ k ] 1 } = 0$ 或 $n _ { [ k ] 0 } = 0$，导致这些层的 $\hat {\tau} _ { [ k ] }$ 定义不清。这与将在第 20 章讨论的**重叠（overlap）**问题有关。其次，如何将该分层方法应用于多维连续或混合协变量 $X$ 并不明确。一种标准方法是基于初始协变量创建层，然后应用分层方法。这可能导致分析中的随意性。

## 10.4.2 结果回归（Outcome regression）

基于结果回归的最常用方法是，对观察到的结果关于处理指示变量和协变量使用**加法模型（additive model）**运行**普通最小二乘法（Ordinary Least Squares, OLS）**，该模型假设：

$$
E (Y \mid Z, X) = \beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X.
$$

如果上述线性模型正确，那么我们有：

$$
\begin{array}{l} \tau (X) = E (Y \mid Z = 1, X) - E (Y \mid Z = 0, X) \\ = \left(\beta_ {0} + \beta_ {z} + \beta_ {x} ^ {\mathsf {T}} X\right) - \left(\beta_ {0} + \beta_ {x} ^ {\mathsf {T}} X\right) \\ = \beta_ {z}, \\ \end{array}
$$

这意味着**处理效应（treatment effect）**相对于协变量是**同质的（homogeneous）**。这结合可忽略性，意味着：

$$
\tau = E \{\tau (X) \} = \beta_ {z}.
$$

因此，如果可忽略性成立且结果模型是线性的，那么平均因果效应等于 $Z$ 的系数。这是线性模型最重要的应用之一。然而，$Z$ 系数的因果解释仅在两个强假设下成立：可忽略性和线性模型。

我们在第 6 章已经讨论过，即使在随机化实验中，上述程序也是次优的，因为它忽略了由协变量引起的处理效应异质性。如果我们假设：

$$
E (Y \mid Z, X) = \beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X + \beta_ {z x} ^ {\mathsf {T}} X Z,
$$

我们有：

$$
\begin{array}{l} \tau (X) = E (Y \mid Z = 1, X) - E (Y \mid Z = 0, X) \\ = \left(\beta_ {0} + \beta_ {z} + \beta_ {x} ^ {\mathsf {T}} X + \beta_ {z x} ^ {\mathsf {T}} X\right) - \left(\beta_ {0} + \beta_ {x} ^ {\mathsf {T}} X\right) \\ { = } { \beta _ { z } + \beta _ { z x } ^ { \mathsf { T } } X , } \\ \end{array}
$$

这结合可忽略性，意味着：

$$
\tau = E \{\tau (X) \} = E (\beta_ {z} + \beta_ {z x} ^ {\mathsf {T}} X) = \beta_ {z} + \beta_ {z x} ^ {\mathsf {T}} E (X).
$$

于是 $\tau$ 的估计量为 $\hat { \beta } _ { z } + \hat { \beta } _ { z x } ^ { \sf T } \bar { X }$，其中 $\hat { \beta } _ { z }$ 是回归系数，$\bar{X}$ 是 $X$ 的样本均值。如果我们对协变量进行中心化以确保 $\bar { X } = 0$，那么该估计量就简化为 $Z$ 的回归系数。为简化程序，我们通常在一开始就对协变量进行中心化；同时回想第 6 章介绍的 Lin (2013) 的估计量。Rosenbaum 和 Rubin (1983b) 以及 Hirano 和 Imbens (2001) 讨论了这个估计量。

通常，我们可以使用其他更复杂的模型来估计因果效应。例如，如果我们分别基于处理组和对照组数据构建两个预测器 $\hat { \mu } _ { 1 } ( X )$ 和 $\hat { \mu } _ { 0 } ( X )$，那么我们就得到了**条件平均因果效应（conditional average causal effect）**的估计量：

$$
\hat {\tau} (X) = \hat {\mu} _ {1} (X) - \hat {\mu} _ {0} (X)
$$

以及平均因果效应的估计量：

$$
\hat {\tau} = n ^ {- 1} \sum_ {i = 1} ^ {n} \{\hat {\mu} _ {1} (X _ {i}) - \hat {\mu} _ {0} (X _ {i}) \}.
$$

上述估计量 $\hat{\tau}$ 与第 6 章讨论的**投影估计量（projective estimator）**形式相同。它有时被称为**结果插补估计量（outcome imputation estimator）**。例如，我们可以使用逻辑模型（logistic model）对二元结果进行建模：

$$
E (Y \mid Z, X) = \mathrm{pr} (Y = 1 \mid Z, X) = \frac {e ^ {\beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X}}{1 + e ^ {\beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X}},
$$

然后基于系数 $\hat { \beta } _ { 0 } , \hat { \beta } _ { z } , \hat { \beta } _ { x }$ 的估计量，我们得到平均因果效应的以下估计量：

$$
\hat {\tau} = n ^ {- 1} \sum_ {i = 1} ^ {n} \left\{\frac {e ^ {\hat {\beta} _ {0} + \hat {\beta} _ {z} + \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}}}{1 + e ^ {\hat {\beta} _ {0} + \hat {\beta} _ {z} + \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}}} - \frac {e ^ {\hat {\beta} _ {0} + \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}}}{1 + e ^ {\hat {\beta} _ {0} + \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}}} \right\}.
$$

该估计量并非简单的是逻辑模型中处理的系数¹。它是所有系数以及协变量经验分布的非线性函数。在计量经济学中，该估计量被称为逻辑模型中处理的**平均偏效应（average partial effect）**或**平均边际效应（average marginal effect）**。许多计量经济学软件包可以报告该估计量及其标准误。类似地，我们也可以基于完全交互的逻辑模型推导出相应的估计量；参见问题 10.2。

对于上述所有估计量，我们可以使用**非参数自助法（nonparametric bootstrap）**来估计标准误。参见第 A1.5 章。

上述用于结果条件均值的预测器也可以是其他机器学习工具。特别是，Hill (2011) 提倡使用**树方法（tree methods）**来估计 $\tau$，而 Wager 和 Athey (2018) 提出也将其用于估计 $\hat{\tau}(X)$。Wager 和 Athey (2018) 还将树方法与下一章的思想相结合。自那时起，机器学习和因果推断已成为一个活跃的研究领域（例如，Hahn 等人，2020；Künzel 等人，2019）。

上述基于结果回归的方法最大的问题在于其对结果模型设定的敏感性。问题 1.3 给出了这样一个例子。根据实证研究和发表的动机，人们有时会在搜索了一大组候选模型后报告其有利的因果效应估计值，而不承认这一搜索过程。这是因果推断中 **p-hacking** 的主要来源之一。

## 10.5 课后习题（Homework Problems）

## 10.1 其他因果效应的非参数识别（Nonparametric identification of other causal effects）

在可忽略性和重叠条件下，证明：

1.  **分布因果效应（distributional causal effect）**

$$
\mathrm{DCE} _ {y} = \operatorname * {p r} \{Y (1) > y \} - \operatorname * {p r} \{Y (0) > y \}
$$

对所有 $y$ 都是**非参数可识别的（nonparametrically identifiable）**；

2.  **分位数因果效应（quantile causal effect）**

$$
\mathrm{QCE} _ {q} = \text { quantile } _ {q} \{Y (1) \} - \text { quantile } _ {q} \{Y (0) \},
$$

对所有 $q$ 都是非参数可识别的，其中 $\mathrm { q u a n t i l e } _ { q } \{ \cdot \}$ 是一个随机变量的第 $q$ 分位数。

注：在概率论中，$\operatorname{pr} \{ Y ( z ) \leq y \}$ 是**累积分布函数（cumulative distribution function）**，而 $\operatorname{pr} \{ Y ( z ) > y \}$ 是潜在结果 $Y ( z )$ 的**生存函数（survival function）**。分布因果效应比较了处理组和对照组潜在结果的生存函数。

## 10.2 完全交互逻辑模型中的结果插补估计量（Outcome imputation estimator in the fully interacted logistic model）

假设一个二元结果遵循逻辑模型：

$$
E (Y \mid Z, X) = \operatorname{pr} (Y = 1 \mid Z, X) = \frac {e ^ {\beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X + \beta_ {x z} ^ {\mathsf {T}} X Z}}{1 + e ^ {\beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X + \beta_ {x z} ^ {\mathsf {T}} X Z}}.
$$

那么，平均因果效应的相应结果回归估计量是什么？

## 10.3 数据分析：分层与回归（Data analysis: stratification and regression）

使用 `senstrat` 包中的 `homocyst` 数据集。结果变量是 `homocysteine`（同型半胱氨酸水平），处理变量是 $\mathbf { z }$，其中 $z = 1$ 表示每日吸烟者，$z = 0$ 表示从不吸烟者。协变量包括 `female`, `age3`, `ed3`, `bmi3`, `pov2`（详细解释见该包），`st` 是一个层指示变量，由离散协变量的所有组合定义。

1.  有多少层仅包含处理组或对照组单元？这些层中单元的比例是多少？删除这些层，并对该观察性研究进行分层分析。报告平均因果效应的点估计量、方差估计量和 95% 置信区间。
2.  对结果关于处理指示变量和协变量（无交互项）运行 OLS。报告处理的系数和**稳健标准误（robust standard error）**。删除仅包含处理组或对照组单元的层。重新运行 OLS 并报告结果。
3.  应用 Lin (2013) 的平均因果效应估计量。报告处理的系数和稳健标准误。如果你不删除仅包含处理组或对照组单元的层，会发生什么？
4.  比较上述三种分析的结果。哪一种更可信？

## 10.4 可忽略性与强可忽略性（Ignorability versus strong ignorability）

给出一个例子，使得可忽略性成立但**强可忽略性（strong ignorability）**不成立。

注：这与一个经典的概率问题有关，即找到三个随机变量 A、B、C，使得：

$$
A \bot C \text {  且   } B \bot C \text {  但   } (A, B) \not \bot C.
$$

## 10.5 推荐阅读（Recommended reading）

Cochran (1965) 是关于观察性研究的经典参考文献。它包含许多有用的见解，但没有使用正式的潜在结果框架。

---
¹ 译者注：原文此处有脚注标记 `1`，但正文中未出现对应脚注内容，故保留标记。
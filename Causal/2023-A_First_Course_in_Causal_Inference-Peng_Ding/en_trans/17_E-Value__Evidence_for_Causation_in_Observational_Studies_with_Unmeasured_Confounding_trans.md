# E值（E-Value）：未测量混杂因素下观察性研究中因果关系的证据

第三部分讨论的所有方法都关键依赖于**可忽略性假设（ignorability assumption）**。它们要求控制处理与结果之间的所有混杂因素。然而，我们无法使用数据来验证可忽略性假设。观察性研究常常因可能存在**未测量的混杂因素（unmeasured confounding）**而受到批评。著名的**尤尔-辛普森悖论（Yule–Simpson Paradox）**表明，一个未测量的二元混杂因素可以完全颠覆观察到的处理与结果之间的关联。然而，要颠覆一个更大的观察关联，这个未测量的混杂因素必须与处理和结果有更强的关联。换句话说，并非所有观察性研究都是同等的。有些研究比其他研究提供更强的因果关系证据。

接下来的三章将讨论各种**敏感性分析（sensitivity analysis）**技术，这些技术可以在存在未测量混杂因素的情况下，基于观察性研究量化因果关系的证据。本章从**E值（E-value）**开始，由 VanderWeele 和 Ding（2017）基于 Ding 和 VanderWeele（2016）的理论提出。它对于使用逻辑回归的观察性研究更为有用。第18章讨论基于**逆概率加权（inverse probability weighting）**、**结果回归（outcome regression）**和**双重稳健估计（doubly robust estimation）**的平均因果效应的敏感性分析。第19章讨论用于匹配观察性研究的 Rosenbaum 敏感性分析框架。

## 17.1 康菲尔德型敏感性分析（Cornfield-type sensitivity analysis）

尽管我们不假设给定 $X$ 时的可忽略性：

$$
Z \not \perp \{Y (1), Y (0) \} \mid X,
$$

我们仍然假设给定 $X$ 和另一个未测量的混杂因素 $U$ 时的**潜在可忽略性（latent ignorability）**：

$$
Z \bot \{Y (1), Y (0) \} \mid (X, U).
$$

本章的技术对于二元结果 $Y$ 效果最好，尽管它可以扩展到其他非负结果（Ding and VanderWeele, 2016）。现在关注二元 $Y$。在**风险比（risk ratio）**尺度上的真实条件因果效应定义为

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{true}} = \frac {\mathrm{pr} \{Y (1) = 1 \mid X = x \}}{\mathrm{pr} \{Y (0) = 1 \mid X = x \}},
$$

而观察到的条件风险比等于

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} = \frac {\operatorname* {p r} (Y = 1 \mid Z = 1 , X = x)}{\operatorname* {p r} (Y = 1 \mid Z = 0 , X = x)}.
$$

一般来说，当存在未测量的混杂因素 $U$ 时，

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{true}} \neq \mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}}
$$

因为

$$
\mathrm{RR} _ {Z Y | x} ^ {\text {true}} = \frac {\int \operatorname* {p r} (Y = 1 \mid Z = 1 , X = x , U = u) F (\mathrm{d} u \mid X = x)}{\int \operatorname* {p r} (Y = 1 \mid Z = 0 , X = x , U = u) F (\mathrm{d} u \mid X = x)}
$$

和

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} = \frac {\int \operatorname* {p r} (Y = 1 \mid Z = 1 , X = x , U = u) F (\mathrm{d} u \mid Z = 1 , X = x)}{\int \operatorname* {p r} (Y = 1 \mid Z = 0 , X = x , U = u) F (\mathrm{d} u \mid Z = 0 , X = x)}
$$

是在 $U$ 的不同分布上取平均。

Doll 和 Hill（1950）发现，即使在调整了许多观察到的协变量 $X$ 后，吸烟对肺癌的风险比仍为 9。Fisher（1957）批评他们的结果是非因果的，因为可能存在一个隐藏的基因同时导致吸烟和肺癌，尽管吸烟对肺癌的真实因果效应不存在。这就是**共同原因假设（common cause hypothesis）**，Reichenbach（1957）也讨论过这一点。Cornfield 等人（1959）采取了更具建设性的视角，并提出问题：这个未测量的混杂因素必须有多强，才能解释掉观察到的吸烟与肺癌之间的关联？下面我们将使用 Ding 和 VanderWeele（2016）对该问题的通用表述。

考虑以下因果图：

![image_20](images/image_20.png)

该图以 $X$ 为条件。因此 $Z \bot \bot Y \mid ( X , U )$。以 $X$ 和 $U$ 为条件时，我们观察到 $Z$ 和 $Y$ 之间没有关联；但仅以 $X$ 为条件时，我们观察到 $Z$ 和 $Y$ 之间存在关联。虽然我们可以像 Ding 和 VanderWeele（2016）那样允许 $U$ 是通用的，但为了简化表述，我们假设 $U$ 是二元的。

定义两个**敏感性参数（sensitivity parameters）**：

$$
\mathrm{RR} _ {Z U | x} = \frac {\operatorname* {p r} (U = 1 \mid Z = 1 , X = x)}{\operatorname* {p r} (U = 1 \mid Z = 0 , X = x)} \equiv \frac {f _ {1 , x}}{f _ {0 , x}}
$$

衡量**处理-混杂因素关联（treatment-confounder association）**，而

$$
\mathrm{RR} _ {U Y | x} = \frac {\operatorname* {p r} (Y = 1 \mid U = 1 , X = x)}{\operatorname* {p r} (Y = 1 \mid U = 0 , X = x)},
$$

衡量**混杂因素-结果关联（confounder-outcome association）**，以协变量 $X = x$ 为条件。不失一般性，我们假设 $\mathrm { R R } _ { x } ^ { \mathrm { o b s } } > 1$，$\mathrm { R R } _ { Z U | x } > 1$，且 $\mathrm { R R } _ { U Y \mid x } > 1$。下面我们可以展示主要结果。

**定理 17.1** 在 $Z \bot \bot Y \mid ( X , U )$ 下，我们有

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} \leq \frac {\mathrm{RR} _ {Z U | x} \mathrm{RR} _ {U Y | x}}{\mathrm{RR} _ {Z U | x} + \mathrm{RR} _ {U Y | x} - 1}.
$$

定理 17.1 展示了如果条件独立性 $Z \bot \bot Y \mid ( X , U )$ 成立，则观察到的处理对结果的风险比的上界。在此条件独立性假设下，处理与结果之间的关联纯粹是由于处理与混杂因素之间的关联 $\mathrm { R R } _ { Z U \mid x }$ 以及混杂因素与结果之间的关联 $\mathrm { R R } _ { U Y \mid x }$ 造成的。上界等于 $\mathrm { R R } _ { Z U | x } \mathrm { R R } _ { U Y | x } / \big ( \mathrm { R R } _ { Z U | x } + \mathrm { R R } _ { U Y | x } - 1 \big )$。类似的不等式出现在 Lee（2011）中。它也与线性模型的 Cochran 公式或**遗漏变量偏倚公式（omitted-variable bias formula）**有关，这在问题 16.1 中进行了回顾。

给定 $\mathrm { R R } _ { x } ^ { \mathrm { o b s } }$，两个混杂度量 $\mathrm { R R } _ { Z U | x }$ 和 $\mathrm { R R } _ { U Y \mid x }$ 不能是任意的。它们的函数 $\mathrm { R R } _ { Z U | x } \mathrm { R R } _ { U Y | x } / \big ( \mathrm { R R } _ { Z U | x } + \mathrm { R R } _ { U Y | x } - 1 \big )$ 必须至少与 $\mathrm { R R } _ { x } ^ { \mathrm { o b s } }$ 一样大。

下面我将给出定理 17.1 的证明。

将 $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ 表示为

$$
\begin{array}{l} \mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} \\ = \frac {\operatorname{pr} (Y = 1 \mid Z = 1 , X = x)}{\operatorname{pr} (Y = 1 \mid Z = 0 , X = x)} \\ = \frac {\left[ \begin{array}{c} \operatorname{pr} (U = 1 \mid Z = 1 , X = x) \operatorname{pr} (Y = 1 \mid Z = 1 , U = 1 , X = x) \\ + \operatorname{pr} (U = 0 \mid Z = 1 , X = x) \operatorname{pr} (Y = 1 \mid Z = 1 , U = 0 , X = x) \end{array} \right]}{\left[ \begin{array}{c} \operatorname{pr} (U = 1 \mid Z = 0 , X = x) \operatorname{pr} (Y = 1 \mid Z = 0 , U = 1 , X = x) \\ + \operatorname{pr} (U = 0 \mid Z = 0 , X = x) \operatorname{pr} (Y = 1 \mid Z = 0 , U = 0 , X = x) \end{array} \right]} \\ = \frac {\left[ \begin{array}{c} \operatorname{pr} (U = 1 \mid Z = 1 , X = x) \operatorname{pr} (Y = 1 \mid U = 1 , X = x) \\ + \operatorname{pr} (U = 0 \mid Z = 1 , X = x) \operatorname{pr} (Y = 1 \mid U = 0 , X = x) \end{array} \right]}{\left[ \begin{array}{c} \operatorname{pr} (U = 1 \mid Z = 0 , X = x) \operatorname{pr} (Y = 1 \mid U = 1 , X = x) \\ + \operatorname{pr} (U = 0 \mid Z = 0 , X = x) \operatorname{pr} (Y = 1 \mid U = 0 , X = x) \end{array} \right]} \\ = \frac {f _ {1 , x} \mathrm{RR} _ {U Y | x} + 1 - f _ {1 , x}}{f _ {0 , x} \mathrm{RR} _ {U Y | x} + 1 - f _ {0 , x}} \\ = \frac {(\mathrm{RR} _ {U Y | x} - 1) f _ {1 , x} + 1}{\frac {\mathrm{RR} _ {U Y | x} - 1}{\mathrm{RR} _ {Z U | x}} f _ {1 , x} + 1}. \\ \end{array}
$$

$\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ 关于 $f _ { 1 , x }$ 是递增的。因此令 $f _ { 1 , x } = 1$，我们有

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} \leq \frac {(\mathrm{RR} _ {U Y | x} - 1) + 1}{\frac {\mathrm{RR} _ {U Y | x} - 1}{\mathrm{RR} _ {Z U | x}} + 1} = \frac {\mathrm{RR} _ {Z U | x} \mathrm{RR} _ {U Y | x}}{\mathrm{RR} _ {Z U | x} + \mathrm{RR} _ {U Y | x} - 1}.
$$

在定理 17.1 的证明中，我们得到了一个恒等式

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} = \frac {(\mathrm{RR} _ {U Y | x} - 1) f _ {1 , x} + 1}{\frac {\mathrm{RR} _ {U Y | x} - 1}{\mathrm{RR} _ {Z U | x}} f _ {1 , x} + 1}.
$$

但这个恒等式涉及三个参数

$$
\left\{f _ {1, x}, \mathrm{RR} _ {Z U | x}, \mathrm{RR} _ {U Y | x} \right\};
$$

相关公式见问题 17.2。相比之下，定理 17.1 中的上界仅涉及两个参数

$$
\left\{\mathrm{RR} _ {Z U | x}, \mathrm{RR} _ {U Y | x} \right\}
$$

它们衡量了混杂因素的强度。

## 17.2 E值（E-value）

下面的引理 17.1 对于推导定理 17.1 的有趣推论很有用。

**引理 17.1** 对于 $w _ { 1 } > 1$ 和 $w _ { 2 } > 1$，定义 $\beta ( w _ { 1 } , w _ { 2 } ) = w _ { 1 } w _ { 2 } / ( w _ { 1 } + w _ { 2 } - 1 )$。

$$
\begin{array}{l} 1. \beta (w _ {1}, w _ {2}) \text {  关于 } w _ {1} \text { 和 } w _ {2} \text { 是对称的}; \\ 2. \beta (w _ {1}, w _ {2}) \text { 关于 } w _ {1} \text { 和 } w _ {2} \text { 都是递增的}; \\ 3. \beta (w _ {1}, w _ {2}) \leq w _ {1} \text { 且 } \beta (w _ {1}, w _ {2}) \leq w _ {2}; \\ 4. \beta (w _ {1}, w _ {2}) \leq w ^ {2} / (2 w - 1), \text { 其中 } w = \max (w _ {1}, w _ {2}). \\ \end{array}
$$

利用定理 17.1 和引理 17.1(3)，我们有

$$
\mathrm{RR} _ {Z U | x} \geq \mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}}, \quad \mathrm{RR} _ {U Y | x} \geq \mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}},
$$

或者等价地，

$$
\min \left(\mathrm{RR} _ {Z U | x}, \mathrm{RR} _ {U Y | x}\right) \geq \mathrm{RR} _ {Z Y | x} ^ {\text { obs }}.
$$

因此，要解释掉观察到的相对风险，两个混杂度量 $\mathrm { R R } _ { Z U | x }$ 和 $\mathrm { R R } _ { U Y \mid x }$ 都必须至少与 $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ 一样大。不等式 $\mathrm { R R } _ { Z U | x } \geq \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ 最早由 Cornfield 等人（1959）提出；Gastwirth 等人（1998）提供了更早的证明。Schlesselman（1978）推导出了不等式 $\mathrm { R R } _ { U Y | x } \geq \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$。这些与信息论中的**数据处理不等式（data processing inequality）**有关。

如果我们定义 $w = \mathrm { m a x } \big ( \mathrm { R R } _ { Z U | x } , \mathrm { R R } _ { U Y | x } \big )$，那么我们可以使用定理 17.1 和引理 17.1(4) 得到

$$
\begin{array}{l} w ^ {2} / (2 w - 1) \geq \beta (\mathrm{RR} _ {Z U | x}, \mathrm{RR} _ {U Y | x}) \geq \mathrm{RR} _ {x} ^ {\text { obs }} \\ \implies w ^ {2} - 2 \mathrm{RR} _ {x} ^ {\mathrm{obs}} w + \mathrm{RR} _ {x} ^ {\mathrm{obs}} \geq 0, \\ \end{array}
$$

其中 $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } - \sqrt { \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } \big ( \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } - 1 \big ) }$ 总是小于或等于 1，所以我们有

$$
w = \max (\mathrm{RR} _ {Z U | x}, \mathrm{RR} _ {U Y | x}) \geq \mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} + \sqrt {\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} (\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} - 1)}.
$$

因此，要解释掉观察到的相对风险，混杂度量 $\mathrm { R R } _ { Z U | x }$ 和 $\mathrm { R R } _ { U Y \mid x }$ 的最大值必须至少为 $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } + \sqrt { \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } \big ( \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } - 1 \big ) }$。基于这个结果，VanderWeele 和 Ding（2017）引入了以下**E值（E-value）**的概念，用于衡量观察性研究中因果关系的证据。

对于 $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$，将 **E值** 定义为

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} + \sqrt {\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} (\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} - 1)}
$$

其中 $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ 是观察到的风险比。在实际应用中，$\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ 是通过抽样误差估计的。我们可以基于 $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ 的点估计或置信限来计算 E值。

Fisher 的 p 值衡量随机实验中因果效应的证据。我们在本书第二部分讨论了基于**随机化检验（FRT）**的 p 值。然而，在大样本的观察性研究中，p 值可能是因果效应证据的一个较差度量。即使真实因果效应为 0，微量的未测量混杂也会使估计产生偏倚，考虑到较小的抽样不确定性，这可能导致极小的 p 值。在大样本的观察性研究中，抽样不确定性通常是次要的，但由未测量混杂造成的不确定性通常是首要问题，它不会随着样本量的增加而减小。VanderWeele 和 Ding（2017）认为，E值是观察性研究中因果效应证据的更好度量。

## 17.3 一个经典例子

下面我重新审视一个经典例子。

**例 17.1** Hammond 和 Horn（1958）使用美国人口研究吸烟与肺癌的关系。忽略协变量，他们的数据可以用一个 $2 \times 2$ 表格表示：

|        | 肺癌 | 无肺癌 |
|--------|------|--------|
| 吸烟者 | 397  | 78557  |
| 非吸烟者 | 51   | 108778 |

基于这些数据，他们得到了风险比估计值 10.73，95% 置信区间为 [8.02, 14.36]。要解释掉点估计值，E值为

$$
10.73 + \sqrt {10.73 \times (10.73 - 1)} = 20.95;
$$

要解释掉置信下限，E值为

$$
8.02 + \sqrt {8.02 \times (8.02 - 1)} = 15.52.
$$

图 17.1 显示了要解释掉风险比的点估计值和置信下限时，两个混杂度量的联合值。具体来说，要解释掉点估计值，它们必须位于实曲线上方的区域；要解释掉置信下限，它们必须位于虚曲线上方的区域。

## 17.4 扩展（Extensions）

**E值**可以扩展到其他效应度量，包括**比值比（odds ratio）**和**风险差（risk difference）**。对于比值比，当结果罕见时，E值的公式类似；当结果常见时，需要进行校正。对于风险差，VanderWeele 和 Ding（2017）也提供了相应的公式。此外，E值可以扩展到**置信区间**，通过使用观察到的效应估计的置信限来计算。这提供了对未测量混杂稳健性的保守评估。

E值的一个关键优势是其**直观解释**：E值表示为了将观察到的关联完全解释为非因果的，未测量的混杂因素必须同时与处理和结果具有的最小关联强度（以风险比度量）。例如，E值为 20.95 意味着，要完全解释掉吸烟与肺癌之间的观察关联，未测量的混杂因素必须与吸烟和肺癌都具有至少 20.95 倍的风险比关联，这在实际中极不可能。

然而，E值也有局限性。它假设未测量的混杂因素是二元的，并且与处理和结果的关系是单调的。此外，它没有考虑多个未测量的混杂因素，也没有提供关于偏倚方向的信息。尽管存在这些局限性，E值仍然是评估观察性研究结果对未测量混杂敏感性的有用工具，并且已被广泛应用于流行病学和其他领域。

## 17.4.1 E值（E-value）与布拉德福德·希尔因果关系准则（Bradford Hill's criteria for causation）

E值提供了**因果关系（causation）**的证据，但证据不等于证明。E值越大，需要更强的未测量混杂因素（unmeasured confounder）才能解释掉观察到的风险比（risk ratio）；因果关系的证据越强。E值越小，需要较弱的未测量混杂因素即可解释掉观察到的风险比；因果关系的证据越弱。结合第17.5.1节的讨论，较大的观察风险比具有更强的因果关系证据。这与布拉德福德·希尔爵士（Sir Bradford Hill）关于因果关系的第一个准则密切相关：**关联强度（strength of the association）**（Bradford Hill, 1965）。定理17.1为其启发式论证提供了数学量化。

在一篇著名论文中，Bradford Hill (1965) 提出了九条准则，为假定原因与结果之间的因果关系提供证据。他的准则是：

1. 强度（strength）；
2. 一致性（consistency）；
3. 特异性（specificity）；
4. 时间性（temporality）；
5. 生物梯度（biological gradient）；
6. 合理性（plausibility）；
7. 连贯性（coherence）；
8. 实验（experiment）；
9. 类比（analogy）。

E值是证明其第一条准则的一种方法。即，更强的关联通常提供更强的因果关系证据，因为要解释更强的关联，我们需要更强的混杂度量。我们在第二部分中讨论过随机实验（randomized experiments），这证实了他的第八条准则。由于篇幅限制，我省略了对其他准则的详细讨论，并鼓励读者阅读（Bradford Hill, 1965）。最近，该论文以 Bradford Hill (2020) 的形式再版，并附有许多因果推断领域顶尖研究人员的深刻评论。

## 17.4.2 逻辑回归（Logistic Regression）后的E值

对于二元结果，流行病学家通常使用结果 $Y _ { i }$ 对处理指标 $Z _ { i }$ 和协变量 $X _ { i }$ 的逻辑回归：

$$
\mathrm{pr} (Y _ {i} = 1 \mid Z _ {i}, X _ {i}) = \frac {e ^ {\beta_ {0} + \beta_ {1} Z _ {i} + \beta_ {2} ^ {\mathsf {T}} X _ {i}}}{1 + e ^ {\beta_ {0} + \beta_ {1} Z _ {i} + \beta_ {2} ^ {\mathsf {T}} X _ {i}}}.
$$

在上述逻辑模型中，$Z _ { i }$ 的系数是给定协变量条件下处理与结果之间的**条件优势比（conditional odds ratio）**的对数：

$$
\beta_ {1} = \log \frac {\mathrm{pr} (Y _ {i} = 1 \mid Z _ {i} = 1 , X _ {i} = x) / \mathrm{pr} (Y _ {i} = 0 \mid Z _ {i} = 1 , X _ {i} = x)}{\mathrm{pr} (Y _ {i} = 1 \mid Z _ {i} = 0 , X _ {i} = x) / \mathrm{pr} (Y _ {i} = 0 \mid Z _ {i} = 0 , X _ {i} = x)}.
$$

重要的是，逻辑模型假设在所有协变量取值上存在一个共同的优势比。此外，当结果罕见，即 $\mathrm{pr} ( Y _ { i } = 1 \mid Z _ { i } = 1 , X _ { i } = x )$ 和 $\mathrm{pr} ( Y _ { i } = 1 \mid Z _ { i } = 0 , X _ { i } = x )$ 都接近0时，条件优势比近似于**条件风险比（conditional risk ratio）**（参见命题1.1(3)）：

$$
\beta_ {1} \approx \log \frac {\operatorname{pr} (Y _ {i} = 1 \mid Z _ {i} = 1 , X _ {i} = x)}{\operatorname{pr} (Y _ {i} = 1 \mid Z _ {i} = 0 , X _ {i} = x)} = \log \mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}}.
$$

因此，基于估计的逻辑回归系数及其相应的置信限，我们可以立即计算出E值。这是E值的主要应用。

**例 17.2** NCHS2003.txt 包含美国国家卫生统计中心（National Center for Health Statistics）的出生证明数据，其中以下二元指标变量对我们有用：

PTbirth 早产（pre-term birth）  
preeclampsia 子痫前期（pre-eclampsia）$^{3}$  
ageabove35 高龄产妇（年龄 $\geq$ 35）（处理）  
somecollege 大学教育  
mar 婚姻状况  
smoking 吸烟状况  
drinking 饮酒状况  
hispanic 母亲种族：西班牙裔  
black 母亲种族：黑人  
nativeamerican 母亲种族：美洲原住民  
asian 母亲种族：亚裔

此版本数据来自 Valeri 和 Vanderweele (2014)。本示例关注结果变量 PTbirth 和问题 17.3。以下 R 代码在拟合逻辑回归后计算E值。基于这些E值，我们得出结论：要解释掉点估计值，最大混杂度量必须大于1.94；要解释掉置信下限，最大混杂度量必须大于1.91。尽管这些混杂度量不如第17.3节中的那么强，但在流行病学研究中它们似乎相当大。

```diff
> evalue = function(rr)
+ {
+    rr + sqrt(rr*(rr - 1))
+ }
>
> NCHS2003 = read.table("NCHS2003.txt", header = TRUE, sep = "\t")
>
> ## outcome: PTbirth
> y_logit = glm(PTbirth ~ ageabove35 +
+    mar + smoking + drinking + somecollege +
+    hispanic + black + nativeamerican + asian,
+    data = NCHS2003,
+    family = binomial)
> log_or = summary(y_logit)$coef[2, 1:2]
```

```txt
> est = exp(log_or[1])
> lower.ci = exp(log_or[1] - 1.96*log_or[2])
> est
Estimate
1.305982
> evalue(est)
Estimate
1.938127
>
> lower.ci
Estimate
1.294619
> evalue(lower.ci)
Estimate
1.912211
```

## 17.4.3 非零的真实因果效应（Non-zero true causal effect）

定理17.1假设处理对结果没有真实的因果效应。Ding 和 VanderWeele (2016) 证明了一个允许非零真实因果效应的一般性定理。

**定理 17.2** 将 $\operatorname { R R } _ { U Y \mid x }$ 的定义修改为：

$$
\mathrm{RR} _ {U Y | x} = \max _ {z = 0, 1} \frac {\operatorname* {p r} (Y = 1 \mid Z = z , U = 1 , X = x)}{\operatorname* {p r} (Y = 1 \mid Z = z , U = 0 , X = x)}.
$$

我们有：

$$
\mathrm{RR} _ {Z Y | x} ^ {\text {true}} \geq \mathrm{RR} _ {Z Y | x} ^ {\text {obs}} \Big / \frac {\mathrm{RR} _ {Z U | x} \mathrm{RR} _ {U Y | x}}{\mathrm{RR} _ {Z U | x} + \mathrm{RR} _ {U Y | x} - 1}.
$$

$\mathrm { R R } _ { Z Y | x } ^ { \mathrm { t r u e } } = 1$。定理17.2的证明请参见 Ding 和 VanderWeele (2016) 的原始论文。在不假设任何额外条件的情况下，定理17.2陈述了在给定观察风险比 $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ 和两个**敏感性参数（sensitivity parameters）** $\mathrm{RR}_{ZU|x}$ 与 $\mathrm{RR}_{UY|x}$ 时，真实风险比 $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { t r u e } }$ 的一个下限。

当处理对结果明显具有预防作用时，观察风险比小于1。在这种情况下，定理17.1和17.2不直接适用，我们必须重新标记处理水平并计算 $1 / \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$。

## 17.5 批评与回应

自原始论文发表以来，E值已成为许多流行病学研究报告的标准数字。尽管如此，它也引起了批评（Ioannidis et al., 2019）。下面我将回顾E值的一些局限性。

## 17.5.1 E值只是风险比的单调变换（monotone transformation）

从图17.2可以看出，如果风险比很大，那么E值 $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } + \sqrt { \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } \big ( \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } - 1 \big ) }$ 约等于 $2 \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$，这与风险比呈线性关系。对于较小的风险比，E值则更非线性。批评者常说，E值仅仅是风险比的点估计值或置信限的单调变换，因此它不提供任何额外信息。

这种说法部分正确。确实，E值完全基于风险比的点估计值或置信限。但它基于定理17.1有一个有意义的解释：要解释掉观察到的风险比，混杂度量的最大值必须至少与E值一样大。

## 17.5.2 E值的校准（Calibration）

E值等于混杂因素与处理之间的关联以及混杂因素与结果之间的关联的最大值，该值足以完全解释掉一个观察到的关联。一个明显的问题是，这个混杂因素本质上是潜在的。因此，判断某个E值是大是小并非易事。另一个相关问题是，E值取决于我们控制了多少个观察到的协变量 $X$，因为它量化了给定 $X$ 后**残余混杂（residual confounding）**的强度。因此，不同研究中的E值不能直接比较。E值提供了因果关系的证据，但应基于感兴趣问题的背景知识仔细评估该证据。

以下**留一协变量法（leave-one-covariate-out approach）**是一种校准E值的直观方法。对于 $X = ( X _ { 1 } , \ldots , X _ { p } )$，我们可以假设分量 $X _ { j }$ 未被观测到，并计算给定其他观察协变量时 $Z$ 与 $X _ { j }$ 以及 $X _ { j }$ 与 $Y$ 的风险比 $( j = 1 , \ldots , p )$。如果我们相信未测量的 $U$ 并不像所有观察到的协变量那样强，那么这些风险比提供了由 $U$ 引起的混杂度量的范围。然而，我不知道这种方法有任何形式上的依据。

## 17.5.3 它最适用于二元结果和风险比

定理17.1对于二元结果和风险比效果良好。Ding 和 VanderWeele (2016) 也为其他因果参数提出了敏感性分析方法，但它们不如基于风险比的二元结果的E值那样优雅。下一章将针对平均因果效应（average causal effect）提出一种简单的敏感性分析方法，该方法将第三部分中的几种方法作为特例包含在内。

## 17.6 家庭作业（Homework Problems）

## 17.1 引理 17.1

证明引理17.1。

## 17.2 Schlesselman (1978) 公式

为简化起见，在以下讨论中我们隐含地以 $X$ 为条件。对于二元处理 $Z$、结果 $Y$ 和未测量混杂因素 $U$，证明：

$$
\frac {\mathrm{RR} _ {Z Y} ^ {\mathrm{obs}}}{\mathrm{RR} _ {Z Y} ^ {\mathrm{true}}} = \frac {1 + (\gamma - 1) \mathrm{pr} (U = 1 \mid Z = 1)}{1 + (\gamma - 1) \mathrm{pr} (U = 1 \mid Z = 0)}
$$

假设在 $U = 0$ 和 $U = 1$ 内部，处理对结果的风险比相同：

$$
\mathrm{RR} _ {Z Y | U = 0} = \mathrm{RR} _ {Z Y | U = 1},
$$

并且假设在 $Z = 0$ 和 $Z = 1$ 内部，混杂因素对结果的风险比也相同：

$$
\mathrm{RR} _ {U Y | Z = 0} = \mathrm{RR} _ {U Y | Z = 1}, \text {   记作   } \gamma .
$$

提示：首先验证如果 $\mathrm { R R } _ { Z Y | U = 0 } = \mathrm { R R } _ { Z Y | U = 1 }$，那么

$$
\mathrm{RR} _ {Z Y} ^ {\mathrm{true}} = \mathrm{RR} _ {Z Y | U = 0} = \mathrm{RR} _ {Z Y | U = 1}.
$$

这个恒等式证明了风险比的可折叠性（collapsibility）。在流行病学中，风险比是一种可折叠的关联度量。

注：Schlesselman (1978) 的公式不假设条件独立性 $Z \bot \bot Y \mid U$，但假设 $Z-Y$ 和 $U-Y$ 风险比的同质性。这是一个经典的敏感性分析公式。它是一个恒等式，通过预先指定的

$$
\{\gamma , \mathrm{pr} (U = 1 \mid Z = 1), \mathrm{pr} (U = 1 \mid Z = 0) \}
$$

即可简单实现。然而，它涉及的敏感性参数比定理17.1更多。尽管定理17.1只给出了一个不等式，但在更强的假设下，与 Schlesselman (1978) 的公式相比，它并不是一个宽松的不等式。有了定理17.1，Schlesselman (1978) 的公式只有历史意义。

## 17.3 逻辑回归后的E值：数据分析

本问题使用与例17.2相同的数据集。

报告结果变量为子痫前期（preeclampsia）时的E值。

## 17.4 风险差（Risk Difference）的康菲尔德型不等式（Cornfield-type inequalities）

考虑二元变量 $Z, Y, U$，并隐含地以 $X$ 为条件。假设给定 $U$ 下的**潜在可忽略性（latent ignorability）**。证明在 $Z \bot \bot Y \mid U$ 条件下，我们有：

$$
\mathrm{RD} _ {Z Y} ^ {\mathrm{obs}} = \mathrm{RD} _ {Z U} \times \mathrm{RD} _ {U Y} \tag {17.1}
$$

其中 $\mathrm { R D } _ { Z Y } ^ { \mathrm { o b s } }$ 是 $Z$ 对 $Y$ 的观察风险差，$\mathrm { R D } _ { Z U }$ 和 $\mathrm{RD}_{UY}$ 分别是处理-混杂因素和混杂因素-结果的风险差（回顾第1.2.2章中风险差的定义）。

注：不失一般性，假设 $\mathrm{rd} _ { Z Y } ^ { \mathrm { o b s } } , \mathrm { R D } _ { Z U } , \mathrm { R D } _ { U Y }$ 均为正。那么 (17.1) 意味着：

$$
\min \bigl (\mathrm{RD} _ {Z U}, \mathrm{RD} _ {U Y} \bigr) \geq \mathrm{RD} _ {Z Y} ^ {\mathrm{obs}}
$$

和

$$
\max \bigl (\mathrm{RD} _ {Z U}, \mathrm{RD} _ {U Y} \bigr) \geq \sqrt {\mathrm{RD} _ {Z Y} ^ {\mathrm{obs}}}.
$$

这些是二元混杂因素下风险差的康菲尔德不等式。它们表明，对于一个未测量的混杂因素要解释掉 $\mathrm { R D } _ { Z Y } ^ { \mathrm { o b s } }$，这两个风险差中至少有一个必须大于 $\mathrm { R D } _ { Z Y } ^ { \mathrm { o b s } }$ 的平方根。

Cornfield 等人 (1959) 得到了 (17.1)，但未认识到其重要性。Gastwirth 等人 (1998) 和 Poole (2010) 讨论了风险差的第一个康菲尔德条件，Ding 和 VanderWeele (2014) 讨论了第二个。

Ding 和 VanderWeele (2014) 还推导了不假设 $U$ 为二元变量的更一般结果。不幸的是，一般 $U$ 的结果弱于上述二元 $U$ 的结果，即随着 $U$ 的层次增多，不等式变得更宽松。这促使 Ding 和 VanderWeele (2016) 专注于风险比的康菲尔德不等式，该不等式不会随着 $U$ 的层次增多而恶化。

## 17.5 推荐阅读

Ding 和 VanderWeele (2016) 扩展并统一了康菲尔德型敏感性分析，这是E值概念的理论基础。
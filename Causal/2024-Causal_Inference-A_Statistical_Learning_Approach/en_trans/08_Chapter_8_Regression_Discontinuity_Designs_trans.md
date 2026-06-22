# 第8章 断点回归设计（Regression Discontinuity Designs）

处理效应估计最清晰、最直接的方法是使用基于**随机处理分配（random treatment assignment）** 的方法——其中随机化可以是显式的（如**随机对照试验（randomized controlled trials）**）或隐式的（如在**无混杂假设（unconfoundedness assumption）** 下的观察性研究分析中）。本书迄今为止讨论的所有方法都属于这一范畴。

然而，在实际应用中，人们通常也有兴趣利用那些假设处理如同随机般分配并不现实的数据（即使在控制了观测到的处理前协变量之后）进行因果推断，并且存在许多广泛使用的计量经济学方法，用于在没有随机处理分配的情况下识别和估计因果效应。本章——以及后续章节——将简要介绍这些用于因果推断的**准实验（quasi-experimental）** 方法。我们使用“准实验”这一术语来强调，这些方法仍然以随机实验的概念为框架——例如**潜在结果（potential outcomes）** 和**平均处理效应（average treatment effects）**——但需要计量经济学的创新来弥补随机处理分配的缺失。

**设定与符号** 本章讨论的是**断点回归设计（regression discontinuity design, RDD）**，这是一种简单且广泛使用的准实验设计。在一个简单的RDD中，我们关注二元处理 $W _ { i }$ 对实数值结果 $Y _ { i }$ 的影响，并设定潜在结果 $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \}$，使得 $Y _ { i } = Y _ { i } ( W _ { i } )$。然而，与随机试验不同，我们不认为处理分配 $W _ { i }$ 是随机的。相反，我们假设存在一个**运行变量（running variable）** $Z _ { i } \in \mathbb { R }$ 和一个**断点（cutoff）** $c$，使得 $W _ { i } = 1 \left( \left\{ Z _ { i } \geq c \right\} \right)$。这种情况可能出现在，例如，教育领域，其中 $Z _ { i }$ 是标准化考试成绩，且 $Z _ { i } \geq c$ 的学生有资格参加荣誉项目；或医学领域，其中 $Z _ { i }$ 是严重程度评分，一旦 $Z _ { i } \geq c$，患者就会被处方某种干预措施。

从定性角度讲，**断点回归（regression discontinuity）** 的主要思想是，尽管处理分配 $W _ { i }$ 并非随机，但它几乎与随机一样好

<!-- footnote -->

- 在 $V _ { A I P W }$ (3.5) 中的项 Var $\left[ \tau ( X _ { i } ) \right]$ 在此处消失，因为 **条件平均处理效应（CATE）** 是常数。
- 当然，使用**残差对残差估计量（residual-on-residual estimator）** 的一个风险是，常数处理效应模型 (4.10) 可能被错误设定。我们在第16章的练习5中考察了在错误设定下残差对残差估计量会发生什么情况。

<!-- footnote end -->

<!-- footnote -->

- 如果真实的**倾向得分（propensity scores）** $e ( x )$ 是已知的，则可以使用它们（并且应该使用）来代替交叉拟合估计量 $\hat { e } ^ { ( - k ) } ( x )$。

<!-- footnote end -->

<!-- footnote -->

- 在这里，由于我们在新数据上评估我们的损失函数 $\hat { \ell } ( \cdot )$，我们不再需要交叉拟合来避免过拟合问题。当然，在实践中，人们需要选择在开发集上使用哪个版本的 $\hat { \ell } ( \cdot )$；一个简单且合理的方法是 $\begin{array} { r } { \hat { \ell } ( \cdot ) = K ^ { - 1 } \sum _ { k = 1 } ^ { K } \hat { \ell } ^ { ( - k ) } ( \cdot ) } \end{array}$，即使用在训练集上产生的双重交叉拟合损失函数。
- 关于明确将**因果森林（causal forests）** 呈现为一种 **R-学习器（R-learner）** 的阐述，请参见 Athey 和 Wager [2019]。

<!-- footnote end -->

<!-- footnote -->

- 我们使用 R 命令 `ns` 将所有特征扩展为7阶**自然三次样条（natural cubic splines）**，然后在这些样条项之间取完整的2阶交互作用。

<!-- footnote end -->

<!-- footnote -->

- 在某些应用中（例如，当需要精确满足预算约束时），考虑随机化策略 $\pi : \mathcal { X } \rightarrow [ 0 , 1 ]$ 是有帮助的，其中 $\pi ( x )$ 的非整数值被解释为处理概率。这里讨论的结果可以直接推广到这种设定。

<!-- footnote end -->

<!-- footnote -->

- 我们认识到这里的 **条件平均处理效应（CATE）** 很可能是非线性的，但出于实际原因，我们仍然寻求福利最大化的**线性阈值规则（linear thresholding rule）**（该规则以允许CATE中存在非线性的方式学习）。

<!-- footnote end -->

<!-- footnote -->

- 作者证明了界 (5.20) 对问题原始参数的函数依赖性是最优可能的，并且该常数最多宽松了200倍。

<!-- footnote end -->

<!-- footnote -->

- 作为进一步的警示说明：我们已经表明，通过经验最大化进行**策略学习（policy learning）** 在计算上等价于分类目标的加权优化。然而，在许多应用中，实践者通过优化一个替代目标（而不是原始的分类目标）来进行分类，$\mathrm { e . g . }$，使用**铰链损失（hinge loss）** 或**逻辑损失（logistic loss）**，并且可能倾向于对 (5.24) 应用类似的近似。然而，这里给出的保证通常不适用于替代目标。例如，有可能设计出这样的情况，即使用 (5.24) 的“逻辑”替代目标进行学习会使我们优先考虑那些从处理中获益最少的人（而不是最多的人）；有关讨论请参见 Wager [2019]。

<!-- footnote end -->

<!-- footnote -->

- $\begin{array} { r } { R _ { T } ^ { Y } = \sum _ { t = 1 } ^ { T } \left( Y _ { t } ( k ^ { * } ) - Y _ { t } \right) } \end{array}$，其中 $k ^ { * }$ 满足 $\mu _ { k ^ { * } } = \mu ^ { * }$。然而，由于动作 $W _ { t }$ 仅依赖于过去的数据，求和项 $Y _ { t } ( k ^ { * } ) - Y _ { t } - \left( \mu ^ { * } - \mu _ { W _ { t } } \right)$ 的差形成一个**鞅差序列（martingale difference sequence）**——因此 $R _ { T }$ 和 $R _ { T } ^ { Y }$ 具有相同的期望。通过相同的论证，可以看到 $R _ { T } ^ { Y } - R _ { T }$ 之间的差异是纯粹的不受实验者控制的噪声。在我们的讨论中，我们将关注 $R _ { T }$ 并将其称为“**遗憾（regret）**”，因为这最准确地量化了实验者所采取行动的后果。

<!-- footnote end -->

<!-- footnote -->

- 该论证对于具有已知尺度参数 $\sigma$ 的**次高斯（sub-Gaussian）** 结果仍然有效。

<!-- footnote end -->

<!-- footnote -->

- 如果我们允许多个最优臂，论证完全相同——只是符号更多。

<!-- footnote end -->

<!-- footnote -->

- 仔细检查发现，为**汤普森采样（Thompson sampling）** 使用**无信息先验（improper prior）** 不仅仅是一个简单的通用选择，而且从遗憾最小化的角度来看，可能是一个准最优的选择 [Kuang and Wager, 2024]。

<!-- footnote end -->

- 注意，条件 $e _ { t , k } > 0$ 实际上可以从定理陈述中省略，代价是在证明中需进行一些额外的记录工作，并遵循 $0 / 0 = 0$ 的约定。**林德伯格型条件（Lindeberg-type condition）** (6.14) 本身已足以对处理分配概率的衰减提供充分控制。

- 此类基表示的存在性在许多背景下是众所周知的；例如，紧区间上的有界变差函数可以用傅里叶级数来表示。此处我们不回顾这些表示何时可用；相反，我们假设已给定一个适当的级数表示。

- 该**指数矩条件（exponential moment condition）**通常弱于第三章中讨论的**强重叠假设（strong overlap assumption）**。请注意，在本文使用的倾向模型下，强重叠将源于假设 $\| X _ { i } \|$ 一致有界。
- 事实 $\mathrm{E} \left[ e ( X _ { i } ) X _ { i } ^ { \otimes 2 } \right] \succ 0$ 直接由我们的假设 $\mathrm{E} [ X _ { i } ^ { \otimes 2 } ] \succ 0$ 以及在我们设定中几乎必然有 $0 < e ( X _ { i } ) < 1$ 这一事实得出。

- 关于 $\hat { \gamma } ^ { ( w ) }$ 对**逆倾向权重（inverse-propensity weights）**具有一致性的条件，请参见 Hirshberg 和 Wager [2021]；因此，结合引理 7.2，条件 $| E | \ll 1 / \sqrt { n }$ 蕴含了第三章所讨论意义上的有效性。

- 同样有趣的是，如果我们使用精确平衡构造 (7.22) 并省略**非负性约束（positivity constraint）** $\gamma _ { i } \geq 0$ ，那么由此导出的 **IPW 型估计量（IPW-type estimator）** (7.21) 在数值上等价于**交互 OLS 回归估计量（interacted OLS regression estimator）** (1.14)。这种等价性可以直接用初等技术证明；也可以通过注意到它等价于**高斯-马尔可夫定理（Gauss-Markov theorem）**来论证这种联系。
- 该方法的一个有限样本考虑是，最终可能会得到一些区域

- 这些区域中仅包含处理组（或对照组）观测值，因此无法进行平衡。因此，这些区域中的数据需要被丢弃，从而导致统计功效的损失——以及潜在的偏差。

当 $Z _ { i }$ 在**断点（cutoff）** $c$ 附近时。具有 $Z _ { i }$ 接近 $c$ 的个体平均而言应彼此相似，但只有那些 $Z _ { i } \geq c$ 的个体得到了处理，因此我们可以通过比较 $Z _ { i }$ 刚好高于 0 和刚好低于 0 的个体来估计处理效应。

例 7. Lee [2008] 通过研究势均力敌的选举，考察了美国众议院选举中的**在位优势（incumbency advantage）**。他比较了某个政党在一个选举周期中，刚好赢得上一周期该席位与刚好输掉上一周期该席位时赢得该众议院席位的概率。该方法的有效性基于以下理解：势均力敌的选举结果不可预测，且受**特质性因素（idiosyncratic factors）**影响（例如，选举日的暴雨可能导致投票率出现差异，从而轻微改变两党得票率），并且一个政党赢得两党投票的 51% 对 49% 的国会选区，其潜在混杂因素的分布应大致相同。然后，一旦我们确认这些国会选区在事前是可比的，我们就可以通过**断点回归方法（regression-discontinuity approach）**获得有效的因果估计。

**为什么倾向得分方法不能在 RDD 中使用** 在讨论断点回归设计中的估计方法之前，有必要思考一下我们之前考虑的方法（如 IPW）为何不适用。正如我们迄今为止的讨论所强调的，倾向得分方法有效所需的两个假设始终是：

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp W _ {i} \mid Z _ {i}, \quad \text { 无混杂性, 以及 } \tag {8.1}
$$

$$
0 <   \mathbb {P} \left[ W _ {i} = 1 \mid Z _ {i} \right] <   1, \quad \text { 重叠性. } \tag {8.2}
$$

综合来看，无混杂性和重叠性意味着我们可以将数据集视为由许多以不同 $Z _ { i }$ 值为索引的小型随机试验合并而成；那么，无混杂性意味着在给定 $Z _ { i }$ 的条件下，处理分配是外生的，而重叠性意味着随机化确实发生了（如果所有人都被分配到同一个处理组，则无法从随机试验中学到任何东西）。

在**断点回归设计（regression discontinuity design）**中，我们有 $W _ { i } = 1 \left( \left\{ Z _ { i } \geq c \right\} \right)$ ，因此无混杂性显然成立（因为 $W _ { i }$ 是 $Z _ { i }$ 的确定性函数）。然而，重叠性显然不成立：$\mathbb { P } \left[ W _ { i } = 1 \big | Z _ { i } = z \right]$ 总是要么为 0，要么为 1。因此，像 IPW 这样涉及除以 $\mathbb { P } \left[ W _ { i } \overline { { = } } 1 | Z _ { i } \right]$ 等方法是不适用的。相反，我们需要比较 $Z _ { i }$ 跨越断点 $c$ 且彼此相似的个体——但它们的分布并不连续。

## 8.1 局部线性回归（Local linear regression）

将 RDD 背后的定性论证形式化的最普遍方法是引用连续性。令 $\mu _ { ( w ) } ( z ) = \mathbb { E } \left[ Y _ { i } ( w ) \big | Z _ { i } \right]$ 。那么，如果 $\mu _ { ( 0 ) } ( z )$ 和 $\mu _ { ( 1 ) } ( z )$ 都是连续的，我们可以通过下式识别 $z = c$ 处的**条件平均处理效应（conditional average treatment effect）**，即 $\tau _ { c } = \mu _ { ( 1 ) } ( c ) - \mu _ { ( 0 ) } ( c )$：

$$
\tau_ {c} = \lim _ {z \downarrow c} \mathbb{E} \left[ Y _ {i} \mid Z _ {i} = z \right] - \lim _ {z \uparrow c} \mathbb{E} \left[ Y _ {i} \mid Z _ {i} = z \right], \tag {8.3}
$$

前提是**运行变量（running variable）** $Z _ { i }$ 在断点 $c$ 附近有支撑。换句话说，我们将 $\tau _ { c }$ 识别为两条不同回归曲线端点之差；上图给出了一个图示。

**基于局部线性回归的估计** 基于 (8.3) 进行估计的一种简单而稳健的方法是使用**局部线性回归（local linear regression）**，如图 8.1 所示。我们选择一个小的**带宽（bandwidth）** $h _ { n } \to 0$ 和一个对称的**权重函数（weighting function）** $K ( \cdot )$ ，然后在边界两侧分别通过加权线性回归拟合 $\mu _ { ( w ) } ( z )$：

$$
\begin{array}{l} \hat {\tau} _ {c} = \operatorname{argmin} \left\{\sum_ {i = 1} ^ {n} K \left(\frac {| Z _ {i} - c |}{h _ {n}}\right) \right. \tag {8.4} \\ \left. \times \left(Y _ {i} - a - \tau W _ {i} - \beta_ {(0)} (Z _ {i} - c) _ {-} - \beta_ {(1)} (Z _ {i} - c) _ {+}\right) ^ {2} \right\}, \\ \end{array}
$$

其中整体截距 $a$ 和斜率参数 $\beta _ { ( w ) }$ 是** nuisance 参数（nuisance parameters）**。权重函数 $K ( x )$ 的常用选择包括**窗函数（window function）** $K ( x ) = 1 \left( \left\{ | x | \leq 1 \right\} \right)$ ，或**三角核（triangular kernel）** $K ( x ) = ( 1 - | x | ) _ { + }$ 。

**一致性、渐近性与收敛速度** 不难看出，在如 (8.3) 的连续性假设下，对于带宽序列 $h _ { n }$ 的合理选择，局部线性回归估计量 (8.4) 必定是一致的。然而，为了超越这种高层次的陈述并获得任何定量保证，我们需要更具体地说明对 $\mu _ { ( 0 ) } ( z )$ 和 $\mu _ { ( 1 ) } ( z )$ 所做的连续性假设。

量化光滑度的方法有很多，但实践中使用最广泛的假设之一——也是我们今天将重点关注的——是 $\mu _ { ( w ) } ( z )$ 二次可导且具有一致有界的二阶导数：

$$
\left| \frac {d ^ {2}}{d z ^ {2}} \mu_ {(w)} (z) \right| \leq B \text {  对于所有   } z \in \mathbb {R} \text {  和   } w \in \{0, 1 \}. \tag {8.5}
$$

假设 (8.5) 的一个动机是它为 (8.4) 中的局部线性回归提供了合理性：如果光滑度较低（例如，$\mu _ { ( w ) } ( z )$ 仅被认为是**利普希茨连续的（Lipschitz）**），那么进行局部线性回归就没有意义，不如进行局部平均；而如果光滑度较高（例如，$\mu _ { ( w ) } ( z )$ 的 $k$ 阶导数有界，且 $k \geq 3$），那么我们可以通过使用更高阶多项式的局部回归来改善收敛速度。

基于这个假设，我们可以直接界定 (8.4) 的误差率。以下结果给出了局部线性回归的收敛速度以及证明梗概。关于更精确的论证，以及如何为带宽 $h _ { n }$ 选择尺度参数 $\kappa$ 的指导，我们参考 Imbens 和 Kalyanaraman [2012]。

**命题 8.1.** 考虑一个 RDD，其中运行变量在断点附近具有连续分布，并且对于所有 $z$，有 Var $\left\lceil Y _ { i } \right\rceil Z _ { i } = z ] \leq \sigma ^ { 2 }$ 。进一步假设对于某个 $B > 0$ ，(8.5) 成立。那么，对于某个 $\kappa > 0$ ，带宽为 $h _ { n } = \kappa n ^ { - 1 / 5 }$ 的局部线性回归估计量 (8.4) 是一致的，且其误差缩放比例为：

$$
\hat {\tau} _ {c} = \tau_ {c} + \mathcal {O} _ {P} \left(n ^ {- 2 / 5}\right). \tag {8.6}
$$

**证明梗概。** 我们首先在 $c$ 处进行**泰勒展开（Taylor expansion）**，得到：

$$
\mu_ {(w)} (z) = a _ {(w)} + \beta_ {(w)} (z - c) + \frac {1}{2} \rho_ {(w)} (z - c), \quad \left| \rho_ {(w)} (x) \right| \leq B x ^ {2}, \tag {8.7}
$$

同时注意到 $\tau _ { c } = a _ { ( 1 ) } - a _ { ( 0 ) }$ 。此外，通过检查问题 (8.4)，我们发现它可以分解为处理组和对照组样本上的两个独立回归问题，即：

$$
\hat {a} _ {(1)}, \hat {\beta} _ {(1)} = \operatorname{argmin} _ {a, \beta} \left\{\sum_ {Z _ {i} \geq c} K \left(\frac {| Z _ {i} - c |}{h _ {n}}\right) (Y _ {i} - a - \beta (Z _ {i} - c)) ^ {2} \right\}, \tag {8.8}
$$

对于处理组单元，以及一个类似的问题用于对照组，使得 $\hat { \tau } = \hat { a } _ { ( 1 ) } - \hat { a } _ { ( 0 ) }$ 。

现在，为简单起见，我们关注使用基本窗核 $K ( x ) = 1 ( \{ | x | \leq 1 \} )$ 的局部线性回归。线性回归问题 (8.8) 可以闭式求解，我们得到：

$$
\hat {a} _ {(1)} = \sum_ {c \leq Z _ {i} \leq c + h _ {n}} \gamma_ {i} Y _ {i}, \quad \gamma_ {i} = \frac {\widehat {\mathbb {E}} _ {(1)} \left[ (Z _ {i} - c) ^ {2} \right] - \widehat {\mathbb {E}} _ {(1)} [ Z _ {i} - c ] \cdot (Z _ {i} - c)}{\widehat {\mathbb {E}} _ {(1)} \left[ (Z _ {i} - c) ^ {2} \right] - \widehat {\mathbb {E}} _ {(1)} [ Z _ {i} - c ] ^ {2}}, \tag {8.9}
$$

其中 $\begin{array} { r } { \widehat { \mathbb { E } } _ { ( 1 ) } \left[ Z _ { i } - c \right] = \sum _ { c < Z _ { i } < c + h _ { n } } ( Z _ { i } - c ) / \left| \{ i : c \leq Z _ { i } \leq c + h _ { n } \} \right| } \end{array}$ 等表示回归窗口内的样本均值。直接计算表明 $\begin{array} { r } { \sum _ { c \leq Z _ { i } \leq c + h _ { n } } \gamma _ { i } = 1 } \end{array}$ 且 $\begin{array} { r } { \sum _ { c \leq Z _ { i } \leq c + h _ { n } } \gamma _ { i } ( Z _ { i } - c ) = 0 } \end{array}$ ，因此由 (8.7) 式得：

$$
\hat {a} _ {(1)} = a _ {(1)} + \underbrace {\sum_ {c \leq Z _ {i} \leq c + h _ {n}} \gamma_ {i} \rho_ {(1)} (Z _ {i} - c)} _ {\text {曲率偏差}} + \underbrace {\sum_ {c \leq Z _ {i} \leq c + h _ {n}} \gamma_ {i} \left(Y _ {i} - \mu_ {(1)} (Z _ {i})\right)} _ {\text {抽样噪声}}, \tag {8.10}
$$

并且对 $\hat { a } _ { ( 0 ) }$ 也有类似的展开。因此，回顾我们的估计量是 $\hat { \tau } = \hat { a } _ { ( 1 ) } - \hat { a } _ { ( 0 ) }$ ，而我们的目标估计量是 $\tau _ { c } = a _ { ( 1 ) } - a _ { ( 0 ) }$ ，我们看到只需界定 (8.10) 中的误差项即可。

鉴于我们对曲率的界，我们立即看到“曲率偏差”项以 $B h _ { n } ^ { 2 }$ 为界。同时，抽样噪声项均值为零，并且假设 Var $\left[ \ddot { Y _ { i } } | Z _ { i } \right] \le \sigma ^ { 2 }$ ，其方差以 $\begin{array} { r } { \sigma ^ { 2 } \sum _ { c \leq Z _ { i } \leq c + h _ { n } } \gamma _ { i } ^ { 2 } } \end{array}$ 的量级为界。最后，假设 $Z _ { i }$ 在 $z$ 的邻域内具有连续非零的**密度函数（density function）** $f ( z )$ ，我们可以验证：

$$
\sigma^ {2} \sum_ {c \leq Z _ {i} \leq c + h _ {n}} \gamma_ {i} ^ {2} \approx \frac {4 \sigma^ {2}}{| \{i : c \leq Z _ {i} \leq c + h _ {n} \} |} \approx \frac {4 \sigma^ {2}}{f (c)} \frac {1}{n h _ {n}}. \tag {8.11}
$$

因此，$\hat { \tau }$ 的平方偏差缩放比例为 $h _ { n } ^ { 4 }$ ，而其方差缩放比例为 $1 / ( h _ { n } n )$ 。**偏差-方差权衡（bias-variance trade-off）** 在 $h _ { n } \sim n ^ { - 1 / 5 }$ 时达到最小，从而得到 (8.6)。

**注 8.1.** $n ^ { - 2 / 5 }$ 的速率是使用 $\mu _ { ( w ) } ( z )$ 二阶导数有界的结果。一般来说，如果我们假设 $\mu _ { ( w ) } ( z )$ 具有有界的 $k$ 阶导数，那么通过使用阶数为 $( k - 1 )$ 的局部多项式回归，带宽缩放比例为 $h _ { n } \sim n ^ { - 1 / ( 2 k + 1 ) }$ ，我们可以达到 $\tau _ { c }$ 的 $n ^ { - k / ( 2 k + \dot { 1 } ) }$ 收敛速度。46 局部线性回归从未达到**参数化收敛速度（parametric rate of convergence）**，但如果 $\mu _ { ( w ) } ( z )$ 非常光滑，则可以接近。

**注 8.2.** 虽然命题 8.1 提供了局部线性回归估计误差的界，但它并未直接推导出关于 $\tau _ { c }$ 的推断方法。这是因为，当使用以估计误差最优速率 $h _ { n } \sim n ^ { - 1 / 5 }$ 缩放的带宽时，$\hat { \tau } _ { c }$ 的偏差和标准误具有相同的量级。这意味着，使用仅考虑方差而不考虑偏差的线性回归构建置信区间的标准工具，会低估 $\hat { \tau } _ { c }$ 的误差大小，并且通常无法达到名义覆盖概率。解决这一挑战的一种简单方法是依赖“**欠平滑（undersmoothing）**”，并选择 $h _ { n } \ll n ^ { - 1 / 5 }$ 使得方差主导偏差。然而，通常不推荐这种策略，因为欠平滑会导致比最优估计更大的误差；此外，以在有限样本中可靠地获得良好覆盖概率的方式来选择欠平滑带宽是具有挑战性的。一个更好的方法是使用利用高阶光滑度的**偏差校正（bias-corrections）**；然而，讨论如何做到这一点超出了本文的范围，我们转而参考 Calonico, Cattaneo, 和 Titiunik [2014] 了解该方法的细节。

## 8.2 优化估计与偏差感知推断（Optimized estimation and bias-aware inference）

我们在上文展示了，若条件期望函数具有如 (8.5) 所示的有界曲率，且 $Z _ { i }$ 在 c 附近具有连续非零密度（这意味着渐近地存在数据点的 $Z _ { i }$ 任意接近 c），则局部线性回归能够以 $n ^ { - 2 / 5 }$ 的误差衰减率估计 **断点回归设计（Regression Discontinuity Design, RDD）** 中的 $\tau _ { c }$。然而，尽管这一结果在概念上很有帮助，并启发了一个简单的估计量，但某些应用场景的特征会阻碍该结果的直接应用。首先，支撑 (8.3) 的渐近论证依赖于观测到 $Z _ { i }$ 任意接近断点 c。但在实践中，我们经常需要处理离散的运行变量（例如，$Z _ { i }$ 是一个取值在 0 到 100 之间的整数测试分数），在这些情况下，支撑命题 8.1 的渐近理论并不适用。此外，在许多应用中，我们需要处理更复杂的断点函数（例如，学生需要通过 3 门考试中的 2 门才有资格参加某个项目），而如何调整局部线性回归以适应此类设定并保持统计功效，并非一目了然。

### RDD 的线性估计量（Linear estimators for RDD）

为了应对这些挑战并为更一般类别的 RDD 开发估计量，我们从一个抽象观察开始。在命题 8.1 的证明中，我们注意到可以将局部线性估计量写为

$$
\hat {\tau} _ {c} (\gamma) = \sum_ {i = 1} ^ {n} \gamma_ {i} Y _ {i}. \tag {8.12}
$$

其中权重 $\gamma _ { i }$ 仅依赖于运行变量 $Z _ { i }$；由使用窗核 $K ( x ) = 1 ( \{ | x | \leq 1 \} )$ 的局部线性回归所诱导的权重的具体形式见 (8.9)。我们将这种形式的估计量称为**线性估计量（linear estimators）**，因为它们结果是向量 Y 的线性函数。47

现在，尽管局部线性回归估计量 (8.4) 是由一个回归问题所启发，但在研究 $\hat { \tau } _ { c }$ 时，我们并未过多利用这个回归公式。相反，在我们的正式讨论中，我们仅使用了对于所有形如 (8.12) 的线性估计量都成立的通用性质。

为简单起见，我们暂时考虑一个具有同方差和高斯误差的设定，使得 $Y _ { i } ( w ) = \mu _ { ( w ) } ( Z _ { i } ) + \varepsilon _ { i } ( w )$，其中 $\varepsilon _ { i } ( w ) \mid Z _ { i } \sim \mathcal { N } \left( 0 , \sigma ^ { 2 } \right)$。那么，任何权重 $\gamma _ { i }$ 仅是 $Z _ { i }$ 函数的线性估计量 (8.12) 满足

$$
\hat {\tau} _ {c} (\gamma) \mid \{Z _ {1}, \dots , Z _ {n} \} \sim \mathcal {N} \left(\hat {\tau} _ {c} ^ {*} (\gamma), \sigma^ {2} \| \gamma \| _ {2} ^ {2}\right),
$$

$$
\hat {\tau} _ {c} ^ {*} (\gamma) = \sum_ {i = 1} ^ {n} \gamma_ {i} \mu_ {W _ {i}} (Z _ {i}), \tag {8.13}
$$

其中 $W _ { i } = 1 \left( \left\{ Z _ { i } \geq c \right\} \right)$。因此，我们立刻看到，只要我们能保证 $\hat { \tau } _ { c } ^ { * } \left( \gamma \right) \approx \tau _ { c }$ 且 $\| \gamma \| _ { 2 } ^ { 2 }$ 很小，任何形如 (8.12) 的线性估计量都将是 $\tau _ { c }$ 的精确估计量。

### 极小极大线性估计（Minimax linear estimation）

受此观察启发，自然会问：如果关于局部线性回归 (8.4) 的关键事实是我们可以将其写成形如 (8.12) 的线性估计量，那么局部线性回归是这类估计量中最好的吗？如下文所见，答案是否定的；然而，形如 (8.12) 的最佳估计量可以在实践中通过数值凸优化轻松推导出来。

如 (8.13) 所述，任何线性估计量的条件方差可以直接观测到：它只是 $\sigma ^ { 2 } \left\| \gamma \right\| _ { 2 } ^ { 2 }$（再次强调，为简单起见，我们今天大部分时间都在处理同方差误差）。相比之下，线性估计量的偏差取决于未知函数 $\mu _ { ( w ) } ( z )$，因此无法观测：

$$
\operatorname{Bias} \left(\hat {\tau} _ {c} (\gamma) \mid \{Z _ {1},..., Z _ {n} \}\right) = \sum_ {i = 1} ^ {n} \gamma_ {i} \mu_ {W _ {i}} (Z _ {i}) - \left(\mu_ {(1)} (c) - \mu_ {(0)} (c)\right). \tag {8.14}
$$

然而，尽管这个偏差是未知的，但在对 $\mu _ { ( w ) } ( z )$ 的光滑性做出假设后，它仍然可以很容易地被界定。例如，如果 $\mu _ { ( w ) } ( z )$ 的曲率如 (8.5) 所示被假设以 B 为界，那么48

$$
\begin{array}{l} \left| \operatorname{Bias} \left(\hat {\tau} _ {c} (\gamma) \mid \{Z _ {1}, \dots , Z _ {n} \}\right) \right| \leq I _ {B} (\gamma) \\ I _ {B} (\gamma) = \sup \left\{\sum_ {i = 1} ^ {n} \gamma_ {i} \mu_ {W _ {i}} (Z _ {i}) - \left(\mu_ {(1)} (c) - \mu_ {(0)} (c)\right): \left| \mu_ {(w)} ^ {\prime \prime} (z) \right| \leq B \right\}. \tag {8.15} \\ \end{array}
$$

现在，回想一下，估计量的**均方误差（mean-squared error, MSE）** 就是其方差与平方偏差之和。由于方差项 $\sigma ^ { 2 } \left\| \gamma \right\| _ { 2 } ^ { 2 }$ 不依赖于条件响应函数，因此我们看到，在所有满足 $| \mu _ { ( w ) } ^ { \prime \prime } ( z ) | \le B$ 的问题上，任何线性估计量的最坏情况均方误差就是其方差与最坏情况偏差平方之和，即

$$
\mathrm{MSE} \left(\hat {\tau} _ {c} (\gamma) \mid \{Z _ {1},..., Z _ {n} \}\right) \leq \sigma^ {2} \| \gamma \| _ {2} ^ {2} + I _ {B} ^ {2} (\gamma), \tag {8.16}
$$

并且在任何达到最坏情况偏差 (8.15) 的函数处取等号。

由此可得，在假设 $| \mu _ { ( w ) } ^ { \prime \prime } ( z ) | \le B$ 且条件于 $\left\{ Z _ { 1 } , . . . , Z _ { n } \right\}$ 的情况下，形如 (8.12) 的**极小极大线性估计量（minimax linear estimator）** 就是最小化 (8.16) 的那个：

$$
\hat {\tau} _ {c} \left(\gamma^ {B}\right) = \sum_ {i = 1} ^ {n} \gamma_ {i} ^ {B} Y _ {i}, \quad \gamma^ {B} = \operatorname{argmin} \left\{\sigma^ {2} \| \gamma \| _ {2} ^ {2} + I _ {B} ^ {2} (\gamma) \right\}. \tag {8.17}
$$

可以通过数值方法验证，局部线性回归隐含的权重并不解决这个优化问题，因此估计量 (8.17) 在最坏情况 MSE 方面优于局部线性回归。

### 推导极小极大线性权重（Deriving the minimax linear weights）

当然，除非我们能在实践中求解出权重 $\gamma _ { i } ^ { B }$，否则估计量 (8.17) 用处不大。幸运的是，我们可以通过常规的二次规划来实现。为此，将函数写为以下形式是有帮助的：

$$
\mu_ {(w)} (z) = a _ {(w)} + \beta_ {(w)} (z - c) + \rho_ {(w)} (z), \tag {8.18}
$$

其中 $\rho _ { ( w ) } ( z )$ 是一个满足 $\rho _ { ( w ) } ( c ) = \rho _ { ( w ) } ^ { \prime } ( c ) = 0$ 且其二阶导数以 B 为界的函数；根据此表示，$\tau _ { c } = a _ { ( 1 ) } - a _ { ( 0 ) }$。

现在，在 (8.18) 中首先要注意的是系数 $a _ { ( w ) }$ 和 $\beta _ { ( w ) }$ 是无约束的。因此，除非权重 $\gamma _ { i }$ 精确地将其考虑在内，即满足

$$
\sum_ {i = 1} ^ {n} \gamma_ {i} W _ {i} = 1, \sum_ {i = 1} ^ {n} \gamma_ {i} = 0, \sum_ {i = 1} ^ {n} \gamma_ {i} (Z _ {i} - c) _ {+} = 0, \sum_ {i = 1} ^ {n} \gamma_ {i} (Z _ {i} - c) _ {-} = 0,
$$

否则我们可以选择 $a _ { ( w ) }$ 和 $\beta _ { ( w ) }$ 使得 $\hat { \tau } _ { c } ( \gamma )$ 的偏差任意大（即 $I _ { B } ( \gamma ) = \infty$）。同时，一旦我们强制执行这些约束，就只需要界定由 $\rho _ { ( w ) } ( z )$ 引起的偏差，因此我们可以将 (8.17) 重写为

$$
\left\{\gamma^ {B}, t \right\} = \mathrm{argmin} \quad \sigma^ {2} \left\| \gamma \right\| _ {2} ^ {2} + B ^ {2} t ^ {2}
$$

$$
\text {  约束条件: } \sum_ {i = 1} ^ {n} \gamma_ {i} W _ {i} \rho_ {(1)} (Z _ {i}) + \sum_ {i = 1} ^ {n} \gamma_ {i} (1 - W _ {i}) \rho_ {(0)} (Z _ {i}) \leq t
$$

$$
\text {  对于所有满足 } \rho_ {(w)} (c) = \rho_ {(w)} ^ {\prime} (c) = 0
$$

$$
\text { 且 } \left| \rho_ {(w)} ^ {\prime \prime} (z) \right| \leq 1 \text { 的 } \rho_ {(w)} (\cdot) \tag {8.19}
$$

$$
\sum_ {i = 1} ^ {n} \gamma_ {i} W _ {i} = 1, \sum_ {i = 1} ^ {n} \gamma_ {i} = 0,
$$

$$
\sum_ {i = 1} ^ {n} \gamma_ {i} W _ {i} (Z _ {i} - c) = 0, \sum_ {i = 1} ^ {n} \gamma_ {i} (Z _ {i} - c) = 0.
$$

给定这种形式，优化问题看起来应该是可解的。事实上确实如此：一旦我们取其对偶形式，问题就简化了，然后可以通过一个有限维的二次规划来很好地近似，其中我们使用一个离散近似来逼近二阶导数以 1 为界的函数集合；详见 Imbens and Wager [2019, Section II.B]。

### 偏差感知推断（Bias-aware inference）

上述讨论表明，如果我们只知道 $| \mu _ { ( w ) } ^ { \prime \prime } ( z ) | \le B$，那么使用估计量 $\hat { \tau } _ { c } \left( \gamma ^ { B } \right) = \sum _ { i = 1 } ^ { n } \gamma _ { i } ^ { B } Y _ { i }$ 来估计 $\tau _ { c }$ 是一个合理的选择。特别地，在此假设下且条件于 $\left\{ Z _ { 1 } , . . . , Z _ { n } \right\}$，它在所有线性估计量中达到了极小极大均方误差。由于局部线性回归也是一个线性估计量，因此我们发现 $\hat { \tau } _ { c } \left( \gamma ^ { B } \right)$ 在极小极大意义上优于局部线性回归。

然而，如果我们想在实践中使用 $\hat { \tau } _ { c } \left( \gamma ^ { B } \right)$，能够为 $\tau _ { c }$ 提供置信区间也很重要。而且，由于 $\hat { \tau } _ { c } \left( \gamma ^ { B } \right)$ 通过构造平衡了偏差和方差，我们不应期望我们的估计量是方差主导的——任何推断程序都应该考虑偏差。

为此，回想 (8.13)，其中条件于 $\left\{ Z _ { 1 } , . . . , Z _ { n } \right\}$，我们估计量的误差 err $: = \hat { \tau } _ { c } - \tau _ { c }$ 的分布为

$$
\operatorname{err} \left| \left\{Z _ {1}, \dots , Z _ {n} \right\} \sim \mathcal {N} \left(\text { bias }, \sigma^ {2} \| \gamma^ {B} \| _ {2} ^ {2}\right). \right. \tag {8.20}
$$

此外，优化问题 (8.19) 还会产生一个关于偏差的副产品上界，该上界以优化变量 t 表示，即 $| \mathrm { b i a s } | \le B t$。

然后我们可以利用这些事实如下构建置信区间。因为高斯分布是单峰且对称的，

$$
\mathbb {P} \left[ | \mathrm{err} | \geq \zeta \right] \leq \mathbb {P} \left[ \left| B t + \sigma \left\| \gamma^ {B} \right\| _ {2} S \right| \geq \zeta \right], \quad S \sim \mathcal {N} (0, 1). \tag {8.21}
$$

因此，我们得到如下置信水平为 α 的置信区间：

$$
\mathbb {P} \left[ \tau_ {c} \in \mathcal {I} _ {\alpha} \mid \{Z _ {1}, \dots , Z _ {n} \} \right] \geq 1 - \alpha ,
$$

$$
\mathcal {I} _ {\alpha} = \left(\hat {\tau} _ {c} (\gamma^ {B}) - \zeta_ {\alpha} ^ {B}, \hat {\tau} _ {c} (\gamma^ {B}) + \zeta_ {\alpha} ^ {B}\right), \tag {8.22}
$$

$$
\zeta_ {\alpha} ^ {B} = \inf \left\{\zeta : \mathbb {P} \left[ \left| B t + \sigma \left\| \gamma^ {B} \right\| _ {2} S \right| > \zeta \right] \leq \alpha , S \sim \mathcal {N} (0, 1) \right\}.
$$

除了形式上考虑了偏差之外，请注意这些区间条件于 $Z _ { i }$ 成立，因此无需对运行变量做任何分布假设。这在考虑非标准设定下的断点回归时非常有用。

### 应用：离散运行变量（Application: Discrete running variable）

拥有条件于 Zi 的保证有用的第一个例子是当运行变量 $Z _ { i }$ 具有离散支撑时。在这种情况下，仅在假设 $| \mu _ { ( w ) } ^ { \prime \prime } ( z ) | \le B$ 下，断点回归参数 $\tau _ { c }$ 通常无法被点识别，因为可能没有任何数据点任意接近边界。49 并且，在缺乏点识别的情况下，任何依赖上一讲讨论的 $\hat { \tau } _ { c }$ 具有特定收敛速度渐近性质的推断方法显然都不适用。

相比之下，在我们的案例中，$Z _ { i }$ 可能具有离散支撑这一事实不会改变任何事情。置信区间 (8.22) 的条件覆盖性质是条件于 $\left\{ Z _ { 1 } , . . . , Z _ { n } \right\}$ 的，而运行变量的经验支撑 $\left\{ Z _ { 1 } , . . . , Z _ { n } \right\}$ 总是离散的，因此在使用 (8.22) 时，$Z _ { i }$ 在总体中是否具有密度是无关紧要的。离散的 $Z _ { i }$ 的相关性只在渐近情况下出现：如果 $Z _ { i }$ 具有连续密度，那么置信区间 (8.22) 将以最优速率（即上一讲讨论的 $n ^ { - 2 / 5 }$）渐近收缩。相反，如果 $Z _ { i }$ 具有离散支撑，置信区间的长度不会趋近于 0；相反，我们最终会面临一个部分识别问题。在此背景下，我们还注意到，偏差感知区间 (8.22) 恰好对应于 Imbens and Manski [2004] 中提出的针对部分识别参数的一种置信区间类型。

### 应用：多元运行变量（Application: Multivariate running variable）

到目前为止，我们关注的是处理状态由单一阈值决定的断点回归设计：对于某个 $Z _ { i } \in \mathbb { R }$，有 $W _ { i } = 1 \left( \left\{ Z _ { i } \geq c \right\} \right)$。然而，这里讨论的思想可以应用于更广泛的场景：可以让运行变量 $Z _ { i } ~ \in ~ \mathbb { R } ^ { k }$ 是多元的，并且处理区域是通用的，即对于某个集合 $\mathcal { A } \subset \mathbb { R } ^ { k }$，有 $W _ { i } \ =$ 1 $( \{ Z _ { i } \in \mathcal { A } \} )$。例如，在教育环境中，$Z _ { i } \in \mathbb { R } ^ { 3 }$ 可以衡量 3 个不同科目的考试成绩，而 A 可以表示由例如 3 门考试中通过 2 门所给出的总体“通过”结果集合。或者在地理断点回归设计中，$Z _ { i } \in \mathbb { R } ^ { 2 }$ 可以表示家庭的位置，而 A 表示部署了特定政策的某个行政区域的边界。

断点回归设计的核心在于我们试图通过现有处理分配策略的急剧变化来识别因果效应；然后我们可以应用与之前相同的推理来识别沿处理区域 A 边界的处理效应。话虽如此，虽然将断点回归设计扩展到一般的多元设定在概念上很直接，但方法论上的扩展需要更加小心。特别是，将局部线性回归推广到地理断点回归设计的最佳方式并不总是明确的。50

然而，极小极大线性方法可以直接扩展到多元设定。当处理多元运行变量时，基本上可以逐字写下 (8.19)，并类似地解释得到的加权估计量。由此产生的优化问题更难（需要在具有有界曲率的多元非参数函数上进行优化），但概念上没有任何改变。

### 超越同方差性（Beyond homoskedaticity）

到目前为止，我们关注的是噪声 $\varepsilon _ { i } = Y _ { i } - \mu _ { ( W _ { i } ) } ( Z _ { i } )$ 为高斯分布且具有已知常数方差参数 $\sigma ^ { 2 }$ 的情况下的估计和推断。当然，在实践中，这两个假设都不太可能成立。结果是，条件高斯性结果 (8.20) 不再精确成立；相反，我们需要援引**中心极限定理（central limit theorem, CLT）** 来论证

$$
\hat {\tau} _ {c} (\gamma) \mid \{Z _ {1}, \dots , Z _ {n} \} \approx \mathcal {N} \left(\hat {\tau} _ {c} ^ {*} (\gamma), \sum_ {i = 1} ^ {n} \gamma_ {i} ^ {2} \operatorname{Var} \left[ Y _ {i} \mid Z _ {i}, W _ {i} \right]\right). \tag {8.23}
$$

然而，只要我们愿意做出使上述高斯近似有效的假设，我们仍然可以像上面那样进行以得到置信区间。同时，我们可以（保守地）通过下式估计 (8.23) 中的条件方差：

$$
\widehat {V} _ {n} = \sum_ {i = 1} ^ {n} \gamma_ {i} ^ {2} \left(Y _ {i} - \hat {\mu} _ {(W _ {i})} (Z _ {i})\right) ^ {2}, \tag {8.24}
$$

其中，例如，$\hat { \mu } _ { ( W _ { i } ) } ( Z _ { i } )$ 是通过局部线性回归推导出来的；请注意，如果 $\hat { \mu } _ { ( W _ { i } ) } ( Z _ { i } )$ 被错误设定，这个界限是保守的，因为错误设定误差会膨胀残差。

话虽如此，应该强调的是，估计量 (8.17) 仅在方差为 $\sigma ^ { 2 }$ 的同方差误差下才是极小极大的；如果我们真的想在异方差下达到极小极大，那么我们需要在 (8.19) 中使用每个参数的方差 $\sigma _ { i } ^ { 2 }$。因此，可以认为，使用估计量 (8.17) 但通过 (8.23) 和 (8.24) 构建置信区间的分析师，是在用一个过度简化的同方差模型来启发一个好的估计量，但出于谨慎和严谨，在构建置信区间时使用了允许异方差的置信区间。这通常是一个好主意，实际上在实践中也很常见（从某种角度看，任何使用 OLS 进行点估计但随后通过自助法获得置信区间的人都在做同样的事情）；然而，重要的是要意识到自己正在做出这个选择。

**注 8.3.** 在本节中，我们假设研究者知道 (8.5) 以某个特定的 B 成立，并据此进行。然而，在实践中，研究者需要选择 B，这是一项精细的任务。数据本身不能用于学习 B，除非做出进一步的光滑性假设 [Armstrong and Koles´ar, 2018]。Armstrong and Koles´ar [2020] 和 Imbens and Wager [2019] 提出了一些启发式方法，用于保守选择 B，这些方法依赖于高阶多项式的全局估计。Eckles et al. [2020] 考虑了一个运行变量的结构模型，该模型除其他外，隐含了一个理论驱动的界限 B，可用于 (8.5)。

## 8.3 参考文献注释（Bibliographic notes）

将**断点回归设计（regression discontinuity designs）**用于处理效应估计的思想可追溯至 Thistlethwaite 和 Campbell [1960]；然而，该领域的大多数正式研究都是近期才开展的。Hahn、Todd 和 van der Klaauw [2001] 阐述了通过连续性论证和**局部线性回归（local linear regression）**进行断点回归设计识别的基本框架。关于通过局部线性回归进行断点回归分析的其他参考文献包括：Cheng、Fan 和 Marron [1997] 讨论了核权重函数的最优选择，Imbens 和 Kalyanaraman [2012] 讨论了带宽选择，以及 Calonico、Cattaneo、Farrell 和 Titiunik [2019] 讨论了协变量调整的作用。Imbens 和 Lemieux [2008] 概述了该背景下局部线性回归的方法，并讨论了其他设定形式，例如“模糊”断点回归，其中 $W _ { i }$ 是随机的，但 $\mathbb { P } \left[ W _ { i } = 1 \big | Z _ { i } = z \right]$ 在断点 $c$ 处存在跳跃。

如 **评注 8.2** 所述，通过局部线性回归构建置信区间具有挑战性，因为当针对最优**均方误差（mean-squared error）**进行调整时，局部线性回归估计量的偏差和抽样误差处于同一量级——因此，基本的**德尔塔方法（delta-method）**或基于**自助法（bootstrap）**的推断会失效（因为它未能捕捉偏差）。多位学者提出了依赖于渐近理论的解决方案。Calonico、Cattaneo 和 Titiunik [2014] 以及 Calonico、Cattaneo 和 Farrell [2018] 提出了对局部线性回归进行偏差校正，以获得有效的置信区间。与此同时，Armstrong 和 Kolesár [2020] 表明，只要我们将置信区间的长度按预先确定的量扩大，未经校正的局部线性回归点估计也可用于有效推断；例如，在命题 8.1 的设定下，使用均方最优带宽时，他们的建议是构建 $\tau _ { c }$ 的 95% 置信区间为 $\hat { \tau } _ { c } \pm 2 . 1 8$ 个标准误（而非常见的 ±1.96 个标准误）。

第 8.2 章中所考虑的**极小化极大线性估计量（minimax linear estimators）**的研究可追溯至 Donoho [1994]，他展示了如下结果。假设我们要使用高斯随机向量 $Y$ 来估计 $\theta$，

$$
Y = K v + \varepsilon , \quad \varepsilon \sim \mathcal {N} (0, \sigma I), \quad \theta = a \cdot v, \tag {8.25}
$$

其中矩阵 $K$ 和向量 $a$ 已知，但 $v$ 未知。进一步假设 $v$ 已知属于一个凸集 $V$。那么，存在一个线性估计量 $\hat { \theta } = \sum _ { i = 1 } ^ { n } \gamma _ { i } Y _ { i }$，其风险在所有估计量（包括非线性估计量）的极小化极大风险的 1.25 倍以内，并且极小化极大线性估计量的权重 $\gamma _ { i }$ 可以通过凸优化推导得出。从这个角度来看，极小化极大断点回归估计量 (8.17) 是 Donoho [1994] 研究的估计量的一个特例，实际上他的结果表明该估计量在所有估计量（不仅仅是线性估计量）中几乎是极小化极大的。

在将该原理首次应用于断点回归设计时，Armstrong 和 Kolesár [2018] 研究了在由 Sacks 和 Ylvisaker [1978] 提出的一类函数上的极小化极大线性估计，对于这类函数，在断点 $c$ 附近的泰勒近似几乎是精确的。我们在第 8.2 章中的阐述改编自 Imbens 和 Wager [2019]，他们考虑了在一般断点回归设计中进行灵活推断的数值凸优化方法。Kolesár 和 Rothe [2018] 提倡使用形如 (8.15) 的**最坏情况偏差度量（worst-case bias measures）**，以此在具有离散分配变量的断点回归设计中避免渐近理论并提供可信的置信区间。Noack 和 Rothe [2024] 将**偏差感知推断（bias-aware inference）**方法扩展到了模糊断点回归。
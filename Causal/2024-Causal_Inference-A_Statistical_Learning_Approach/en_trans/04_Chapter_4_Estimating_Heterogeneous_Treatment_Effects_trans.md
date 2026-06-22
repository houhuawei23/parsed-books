# 第4章 估计异质性处理效应（Estimating Heterogeneous Treatment Effects）

在许多应用领域中，人们不仅关注平均效应，还希望了解**处理效应（treatment effects）**如何在不同单元之间变化。在个性化医疗中，我们可能希望识别出比其他人更有可能从药物中获益（或更不可能出现副作用）的患者群体；而在在线营销中，可能希望识别出更有可能对优惠做出反应的客户群体。本章介绍并比较了多种估计**异质性处理效应（heterogeneous treatment effects）**的方法。

**条件平均处理效应（Conditional average treatment effect）** 在本章中，我们将在与前一章相同的“基本设定”下进行讨论，即满足 SUTVA、无混杂性（unconfoundedness）和重叠性（overlap）；然而，我们不再关注平均处理效应，而是试图估计、理解并最终对**不同单元对处理反应（treatment）的异质性（heterogeneity）**采取行动。乍一看，人们可能认为估计处理异质性应该针对个体特定的**个体处理效应（individual treatment effects, ITEs）** $\Delta _ { i } = Y _ { i } ( 1 ) - Y _ { i } ( 0 )$ 。然而，即使在强假设下，ITEs 通常也无法被点识别（point-identified），因此针对 ITEs 本身的方法往往不实用。

在无混杂性条件下量化处理异质性的一种更实用的方法是通过**条件平均处理效应（conditional average treatment effect, CATE）**：

$$
\tau (x) = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \mid X _ {i} = x \right]. \tag {4.1}
$$

CATE 仍然是一种平均效应；但现在我们考虑当条件于潜在**效应修饰因子（effect modifiers）** $X _ { i }$ 时，这个平均值如何变化。注意，CATE 的定义取决于 (4.1) 中使用了哪些处理前协变量：如果我们条件于更丰富的协变量集，那么 CATE 函数将变得更具表达力（并捕获底层 ITEs 方差的更大比例）。

将 CATE 作为统计目标有许多原因。它易于理解和使用；并且，与 ITE 不同，它是可点识别的。还有形式化的、决策理论上的理由来关注 CATE。例如，以下结果（此处不提供证明）表明，功利主义的**目标定位规则（targeting rules）**可以表示为 CATE 上的阈值规则。

**命题 4.1.** 在第 $\mathcal { B }$ 章描述的满足 SUTVA、无混杂性和重叠性的基本设定下，假设决策者因将处理臂 w 分配给单元 i 而获得奖励 $Y _ { i } ( w )$ ，并且每次分配处理（处理组）时需要支付成本 C（对照组免费）。那么，**处理那些 CATE 大于成本 C 的单元的决策规则**，即 $1 \left( \left\{ \tau ( X _ { i } ) > C \right\} \right)$ ，在所有关于观测到的处理前协变量 $X _ { i }$ 可测的决策规则中最大化期望奖励。

**示例 5.** Kitagawa 和 Tetenov [2018] 讨论了在国家就业培训伙伴法案（National Job Training Partnership Act, JTPA）下，针对培训和求职援助资格的优化目标定位。这里，处理 $W _ { i }$ 是项目资格，结果 $Y _ { i }$ 是处理分配后 30 个月内的收入，可用于目标定位的处理前协变量是 $X _ { i } ~ = ~ \{ \mathrm { e d u c a t i o n }$ ，income}。社会福利最大化的目标定位规则将 CATE 与处理成本进行比较。22

**正则化偏差（Regularization bias）** 在介绍 CATE 估计方法之前，回顾简单基线方法面临的一些问题是有帮助的。在无混杂性条件下，CATE 可以写成条件响应曲面（conditional response surfaces）之差：

$$
\tau (x) = \mu_ {(1)} (x) - \mu_ {(0)} (x), \quad \mu_ {(w)} (x) = \mathbb {E} \left[ Y _ {i}   |   X _ {i} = x,   W _ {i} = w \right]. \tag {4.2}
$$

因此，我们可以通过分别对对照组和处理组单元进行一致的非参数回归来拟合 $\hat { \mu } _ { ( 0 ) } ( \cdot )$ 和 $\hat { \mu } _ { ( 1 ) } ( \cdot )$ ，然后将 CATE 估计为它们的差，从而立即获得 $\tau ( \cdot )$ 的一致估计量。按照 Künzel 等人 [2019] 的命名法，得到的估计量通常被称为 **T-学习器（T-learner）**：

$$
\hat {\tau} _ {T} (x) = \hat {\mu} _ {(1)} (x) - \hat {\mu} _ {(0)} (x). \tag {4.3}
$$

然而，虽然 T-学习器是一致的，但由于一种称为**正则化偏差（regularization bias）** 的现象，它在有限样本中可能表现不佳：由于我们分别拟合 $\hat { \mu } _ { ( 0 ) } ( \cdot )$ 和 $\hat { \mu } _ { ( 1 ) } ( \cdot )$ ，这两个函数可能最终以不同的方式被正则化，从而在学到的 CATE 估计 ${ \hat { \tau } } _ { T } ( x )$ 中产生伪影。如果我们使用正则化量取决于样本量的方法，并且如果对照组单元远多于处理组单元（反之亦然），这个问题尤其严重。23

图 4.1 说明了这个问题。这里没有处理效应，所以 $\mu _ { ( 0 ) } ( x ) =$ $\mu _ { ( 1 ) } ( x )$ 且 $\tau ( x ) = 0$ ，但两个回归曲面都随 x 振荡。数据是通过 $\pi = 0 . 1$ 的随机试验收集的，因此对照组单元远多于处理组单元。这里，最终有足够的对照组数据使得 $\hat { \mu } _ { ( 0 ) } ( \cdot )$ 能够得到良好估计并捕捉条件响应函数的底层振荡。另一方面，处理组单元非常少，因此我们对于 $\hat { \mu } _ { ( 1 ) } ( \cdot )$ 能做的最好的事情就是对其进行重度正则化，导致估计值在 x 上几乎恒定。两个估计 $\hat { \mu } _ { ( 0 ) } ( \cdot )$ 和 $\hat { \mu } _ { ( 1 ) } ( \cdot )$ 本身都是合理的；然而，一旦我们像 (4.3) 中那样取它们的差，我们发现在 ${ \hat { \tau } } _ { T } ( x )$ 中存在强烈的表观异质性，这令人担忧，因为在这个例子中实际上 $\tau ( x ) = 0$ 处处成立。

T-学习器的第二个问题，即**正则化引起的混杂（regularization-induced confounding）** ，源于 T-学习器没有明确考虑**倾向得分（propensity score）** 的变化。如果 $e ( x )$ 变化很大，那么我们对 $\hat { \mu } _ { ( 0 ) } ( \cdot )$ 的估计将由具有更多对照组单元的区域（即 $e ( x )$ 更接近 0 的区域）的数据驱动，而对 $\hat { \mu } _ { ( 1 ) } ( \cdot )$ 的估计则由具有更多处理组单元的区域（即 $e ( x )$ 更接近 1 的区域）的数据驱动。如果用于学习 $\hat { \mu } _ { ( 0 ) } ( \cdot )$ 和 $\hat { \mu } _ { ( 1 ) } ( \cdot )$ 的数据之间存在协变量偏移（covariate shift），这可能会给它们的差 ${ \hat { \tau } } _ { T } ( x )$ 带来偏差。

## 4.1 半参数建模（Semiparametric modeling）

我们对正则化偏差的分析清楚地表明，任何好的 CATE 估计方法都应该“聚焦”于精确估计 $\mathrm { C A T E } \tau ( x )$ ——并且，在灵活的统计学习设置中，这并不等同于同时精确估计 $\mu _ { ( 0 ) } ( x )$ 和 $\mu _ { ( 1 ) } ( x )$ 。要理解成功瞄准 CATE 需要什么，首先考虑以下半参数设定是有帮助的：

$$
\tau (x) = \psi (x) \cdot \beta , \quad \psi : \mathcal {X} \rightarrow \mathbb {R} ^ {d}, \quad \beta \in \mathbb {R} ^ {d}. \tag {4.4}
$$

例如，在示例 5 的背景下，如果 X 包含收入和教育的非结构化数据，可以将 ψ(x) 设置为 {前一年的收入，拥有高中学历，拥有大学学历}。

我们将此设定称为半参数设定，因为我们的整体设定是非参数的（特别是 $\mu _ { ( 0 ) } ( x )$ 和 $e ( x )$ 是任意的），但我们在感兴趣的关键成分上施加了参数设定。在模型 (4.4) 下，估计 CATE 简化为估计 $\beta$ 。在第 3 章的基本设定下工作，并设 $\varepsilon _ { i } ( w ) = Y _ { i } ( w ) - \mu _ { ( w ) } ( X _ { i } )$ ，参数约束 (4.4) 的加入使我们能够将数据生成分布重新表达为**部分线性模型（partially linear model）**：

$$
Y _ {i} (w) = \mu_ {(0)} (X _ {i}) + w   \psi (x) \cdot \beta + \varepsilon_ {i} (w). \tag {4.5}
$$

这类问题由 Robinson [1988] 研究过，他指出，为了估计 $\beta$ ，将 (4.5) 重写为以下形式是有帮助的：

$$
\begin{array}{l} Y _ {i} - m \left(X _ {i}\right) = \left(W _ {i} - e \left(X _ {i}\right)\right) \psi \left(X _ {i}\right) \cdot \beta + \varepsilon_ {i}, \text {where} \\ (.) = \mathbb {E} [ X _ {i} | X _ {i} ] = (X _ {i}) + (Y _ {i}) + (X _ {i}) \cdot \beta . \end{array} \tag {4.6}
$$

$$
m (x) = \mathbb {E} \left[ Y _ {i} \mid X _ {i} = x \right] = \mu_ {(0)} (X _ {i}) + e (X _ {i}) \psi (X _ {i}) \cdot \beta
$$

表示观测到的 $Y _ { i }$ 的条件期望，对 $W _ { i }$ 和 $\varepsilon _ { i } = \varepsilon _ { i } ( W _ { i } )$ 进行边际化。

表达式 (4.6) 表明，如果我们知道 $m ( x )$ 和 $e ( x )$ ，那么我们可以通过一个简单的回归算法来估计 $\beta$ ：首先定义 $\widetilde { Y } _ { i } ^ { * } = Y _ { i } - m ( X _ { i })$ 和 $\widetilde { Z } _ { i } ^ { * } = \psi ( X _ { i } ) ( W _ { i } - \underset { \sim } { e } ( X _ { i } ) )$ ，然后通过运行残差对残差回归 $\widetilde { Y } _ { i } ^ { * } \sim \widetilde { Z } _ { i } ^ { * }$ 来估计 $\hat { \beta } ^ { * }$ 。当然，在实际中，$e ( x )$ 可能未知，而 $m ( x )$ 基本上从未可知，因此运行上述方法并不可行。

然而，我们在第 3 章的讨论激发了尝试使用**双重机器学习（double machine learning）** 框架的插件方法。我们首先通过我们选择的机器学习方法估计未知成分 $m ( x )$ 和 $e ( x )$ ，然后使用交叉拟合（cross-fitting）将它们代入 (4.6)：

1. 使用我们选择的方法运行非参数回归 $Y \sim X$ 和 $W \sim X$ ，分别得到 ${ \hat { m } } ( x )$ 和 $\hat { e } ( x )$ 。
2. 使用交叉拟合残差定义转换后的特征 $\widetilde { Y } _ { i } = Y _ { i } - \hat { m } ^ { ( - k ( i ) ) } ( X _ { i } )$ 和 $\widetilde { Z } _ { i } = \psi ( X _ { i } ) ( W _ { i } - \hat { e } ^ { ( - k ( i ) ) } ( X _ { i } ) )$ 。
3. 通过运行线性回归 $\widetilde { Y } _ { i } \sim \widetilde { Z } _ { i }$ 来估计 $\hat { \beta }$ 。

如下所述，这个残差对残差回归估计量具有类似于定理 3.2 中为 AIPW 建立的特殊性质：只要非参数成分被合理准确地估计，那么 $\hat { \beta }$ 与**神谕估计量（oracle）** $\hat { \beta } ^ { * }$ 渐近等价，并在 $1 / { \sqrt { n } }$ 尺度上满足中心极限定理。24

**定理 4.2.** 在第 $\mathcal { B }$ 章描述的满足 SUTVA、无混杂性和重叠性的基本设定下，假设 (4.4) 成立，回归特征有界 $\| \psi ( X _ { i } ) \| _ { \infty } \leq M$ ，并且我们通过 K 折交叉拟合来估计 $\beta$

以下是依据您的要求，对原文进行的专业、忠实、清晰的中文翻译。译文严格遵循了所有格式规范，包括术语标注、公式保留、标题翻译及内容完整性。

---

- 这里陈述的结果不应被过度泛化。我们已经证明，在一种非常特定的设定下——当 $X _ { i }$ 具有离散支撑且我们使用饱和（因此显然是正确设定的）倾向性模型时——可行的 **逆概率加权（IPW）** 估计量可以优于 **神谕版 IPW（oracle IPW）** 估计量。这一结果不应被理解为可行的 IPW 通常优于神谕版 IPW；并且，在许多重要应用中，并不存在这种优势发生的条件（当然，除非 $X _ { i }$ 确实具有低基数、离散的支撑）。在第三章中，我们将讨论更稳健且算法上更具普适性的方法，以解决神谕版 IPW 的额外渐近方差问题。

- 在随后的讨论中，有一个问题受到了极大的关注，即

- 特别地，我们将能够处理基于机器学习的倾向性得分估计，正如示例 4 中所出现的情况。

- 在文献中，我们在此所称的弱双重稳健性通常简称为双重稳健性 [Bang and Robins, 2005]。

- $^ { 1 7 }$ 该条件成立的一个有趣特例是当 $\sqrt{\alpha_{\mu}, \alpha_{e}} = 1/4$ 时，即 $\hat{\mu}_{(w)}(x)$ 和 $\hat{e}(x)$ 在 **均方根误差（RMSE）** 意义上都是 $o(1/\sqrt[4]{n})$ 一致的。通常，参数模型在 RMSE 意义上是 $\dot{O}(1/\sqrt{n})$ 一致的；因此，结果 (3.5) 可以容纳 $\hat{\mu}_{(w)}(x)$ 和 $\hat{e}(x)$ 的收敛速度比参数速率慢一个数量级的情况。

- 在本书的其余部分，每当讨论 **增强逆概率加权（AIPW）** 时，除非另有说明，我们将隐含地使用 **交叉拟合（cross-fitting）**。许多作者也在实践中推荐使用交叉拟合，并且它已在多个因果推断软件包中实现。

- 请注意，此处柯西-施瓦茨不等式的应用有些松散。存在一些结果——尽管需要更强的假设——能够通过在此处使用更强的论证来放宽速率条件 (3.11)。

- 在此，我们进行了通常的 t 分布自由度调整并除以 $n-1$；然而，以下所有陈述在除以 $n$ 时也同样成立。

- 此陈述有意未作具体说明；我们推荐参考 Chamberlain [1992] 以获取精确表述。

- $^ { 2 2 }$ 一如既往，**条件平均处理效应（CATE）** 的值取决于用于定义它的协变量集 $X _ { i }$。在此应用中，也可以尝试在更大的协变量集上估计条件处理效应，例如，$X _ { i } =$ {教育、收入、年龄、家庭状况、过往经历，$\left. \dots \right\}$，从而得到一个更具表达力的 CATE。命题 4.1 表明，给定一组可用于目标定位的、已测量的预处理协变量，从福利最大化的角度来看，使用基于这些协变量的 CATE 是最优的。然而，在实践中，其他考虑因素也可能适用；关于此主题的进一步讨论请参见下一章。

- 在整个讨论中，我们假设读者熟悉统计学习中出现的关于偏差、方差、正则化、交叉验证等方面的标准结果。关于这些主题的优秀参考是 Hastie、Tibshirani 和 Friedman [2009] 的第五章。

- 这个性质是特殊的：对于大多数估计量，在有用的条件下，估计量的交叉拟合插入版本将不会与估计量的神谕版版本渐近等价。通常，此性质要求估计量是“**内曼正交的（Neyman-orthogonal）**”；特别地，AIPW 和残差对残差回归都是内曼正交的。对内曼正交性及其成立条件给出抽象表征超出了本书的范围；关于此主题的深入研究，请参见 Chernozhukov 等人 [2022a]。

上述给出的残差对残差回归的交叉拟合版本。进一步假设我们对非参数分量使用的估计量满足：对于所有折 k = 1, ..., K，

$$
n ^ {- 2 \alpha_ {m}} \frac {1}{| \{i : k (i) = k \} |} \sum_ {\{i: k (i) = k \}} \left(\hat {m} ^ {(- k)} (X _ {i}) - m (X _ {i})\right) ^ {2} \to_ {p} 0,
$$

$$
n ^ {- 2 \alpha_ {e}} \frac {1}{| \{i : k (i) = k \} |} \sum_ {\{i: k (i) = k \}} ^ {\{i: k (i) = k \}} \left(\hat {e} ^ {(- k)} (X _ {i}) - e (X _ {i})\right) ^ {2} \rightarrow_ {p} 0, \tag {4.7}
$$

其中常数满足 $\alpha _ { m } \geq 0 , \alpha _ { e } \geq 1 / 4$ 且 $\alpha _ { m } + \alpha _ { e } \ge 1 / 2$。那么，记 $\widetilde { Z } _ { i } ^ { * }$ 和 $\widetilde { Z } _ { i } ^ { * }$ 为如下 (4.6) 定义的神谕版残差，则有

$$
\sqrt {n} (\hat {\beta} - \beta) \Rightarrow \mathcal {N} (0, V _ {\beta}), V _ {\beta} = \operatorname{Var} \left[ \widetilde {Z} _ {i} ^ {*} \right] ^ {- 1} \mathbb {E} \left[ \left(\varepsilon_ {i} \widetilde {Z} _ {i} ^ {*}\right) ^ {\otimes 2} \right] \operatorname{Var} \left[ \widetilde {Z} _ {i} ^ {*} \right] ^ {- 1}, \tag {4.8}
$$

前提是 Var $\left[ \widetilde { Z } _ { i } ^ { * } \right]$ 满秩。

**证明。** 在我们的基本设定和 (4.4) 下，表达式 (4.6) 可以被视为一个具有异方差误差的正确设定的线性模型。因此，对异方差下线性回归的标准分析 [White, 1980] 立即表明，神谕版残差对残差回归估计量 $\hat { \beta } ^ { * }$ 满足极限结果 (4.8)。因此，只需证明 $\sqrt { n } ( \hat { \beta } - \hat { \beta } ^ { * } ) \to _ { p } 0$ 即可。

我们可以显式地写出可行和神谕版的残差对残差回归估计量：

$$
\hat {\beta} = \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \widetilde {Z} _ {i} ^ {\otimes 2}\right) ^ {- 1} \frac {1}{n} \sum_ {i = 1} ^ {n} \widetilde {Z} _ {i} \widetilde {Y} _ {i}, \quad \hat {\beta} ^ {*} = \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \widetilde {Z} _ {i} ^ {* \otimes 2}\right) ^ {- 1} \frac {1}{n} \sum_ {i = 1} ^ {n} \widetilde {Z} _ {i} ^ {*} \widetilde {Y} _ {i} ^ {*}. \tag {4.9}
$$

我们首先证明，对于每个折 k，

$$
\sqrt {n} \left(\frac {1}{n} \sum_ {\{i: k (i) = k \}} \widetilde {Z} _ {i} \widetilde {Y} _ {i} - \frac {1}{n} \sum_ {\{i: k (i) = k \}} \widetilde {Z} _ {i} ^ {*} \widetilde {Y} _ {i} ^ {*}\right)\rightarrow_ {p} 0.
$$

为此，我们详细写出 $\widetilde { Y } _ { i } , \widetilde { Z } _ { i }$ 等，并展开：

$$
\begin{array}{l} \sum_ {\{i: k (i) = k \}} \psi (X _ {i}) \left(W _ {i} - \hat {e} ^ {(- k)} (X _ {i})\right) \left(Y _ {i} - \hat {m} ^ {(- k)} (X _ {i})\right) - \psi (X _ {i}) \left(W _ {i} - e (X _ {i})\right) \left(Y _ {i} - m (X _ {i})\right) \\ = \sum_ {\{i: k (i) = k \}} \psi (X _ {i}) \left(W _ {i} - e (X _ {i})\right) \left(m (X _ {i}) - \hat {m} ^ {(- k)} (X _ {i})\right) \\ + \sum_ {\{i: k (i) = k \}} \psi (X _ {i}) \left(e (X _ {i}) - \hat {e} ^ {(- k)} (X _ {i})\right) (Y _ {i} - m (X _ {i})) \\ + \sum_ {\{i: k (i) = k \}} \psi (X _ {i}) \left(e (X _ {i}) - \hat {e} ^ {(- k)} (X _ {i})\right) \left(m (X _ {i}) - \hat {m} ^ {(- k)} (X _ {i})\right). \\ \end{array}
$$

然后，我们完全按照定理 3.2 证明中的方法对这些项进行界定：对于上面的前两项，我们依赖于交叉拟合；而对于最后一项，我们使用柯西-施瓦茨不等式（依赖于我们的假设 $\alpha _ { m } + \alpha _ { e } \ge 1 / 2$ 且 $\| \psi ( X _ { i } ) \| _ { \infty } \leq M )$）。以下事实

$$
\sqrt {n} \left(\frac {1}{n} \sum_ {\{i: k (i) = k \}} \widetilde {Z} _ {i} ^ {\otimes 2} - \frac {1}{n} \sum_ {\{i: k (i) = k \}} \widetilde {Z} _ {i} ^ {* \otimes 2}\right)\rightarrow_ {p} 0
$$

的证明遵循相同的论证，只是现在在柯西-施瓦茨不等式的界定中需要使用 $2 \alpha _ { e } \geq 1 / 2$。最后，为了将所有部分整合起来，我们应用 **斯卢茨基引理（Slutsky’s lemma）**，结合以下事实：

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \widetilde {Z} _ {i} ^ {* \otimes 2} \rightarrow_ {p} \operatorname{Var} \left[ \widetilde {Z} _ {i} ^ {*} \right] \succ 0,
$$

并且矩阵求逆在满秩矩阵的邻域内是一个连续函数。□

**常数效应模型** 半参数建模的一个有趣特例是常数处理效应模型：

$$
\mu_ {(1)} (x) - \mu_ {(0)} (x) = \tau , \tag {4.10}
$$

据此我们断言处理效应不随协变量变化；这是 (4.4) 在 $\psi ( x ) = 1$ 时的一个实例。因此，我们也可以在此设定下应用上述开发的残差对残差回归方法，得到以下结果：

**推论 4.3。** 在满足第三章中 **稳定单元处理值假设（SUTVA）**、**无混淆性（unconfoundedness）** 和 **重叠性（overlap）** 的基本设定下，假设常数处理效应模型 (4.10) 成立，并且我们通过满足 (4.7) 的非参数分量的交叉拟合插入残差对残差估计量来估计 $\tau$。那么，

$$
\begin{array}{l} \sqrt {n} (\hat {\tau} - \tau) \Rightarrow \mathcal {N} (0, V _ {\tau}), \\ V _ {\tau} = \frac {\mathbb {E} \left[ e (X _ {i}) (1 - e (X _ {i})) \left((1 - e (X _ {i})) \sigma_ {(1)} ^ {2} (X _ {i}) + e (X _ {i}) \sigma_ {(0)} ^ {2} (X _ {i})\right) \right]}{\mathbb {E} \left[ e (X _ {i}) (1 - e (X _ {i}) \right] ^ {2}}. \tag {4.11} \\ \end{array}
$$

注意，在模型 (4.10) 下，我们也可以使用诸如 AIPW 等用于 **平均处理效应（ATE）** 的方法来估计参数 $\tau$（因为当处理效应为常数 $\tau$ 时，平均处理效应也是 $\tau$）。然而，在这种情况下，AIPW 通常不如残差对残差回归估计量精确。特别地，在 (4.10) 成立且 $\sigma _ { ( 0 ) } ^ { 2 } ( x ) = \sigma _ { ( 1 ) } ^ { 2 } ( x ) = \sigma ^ { 2}$ 的特殊情况下，那么^25

$$
V _ {\tau} = \frac {\sigma^ {2}}{\mathbb {E} [ e (X _ {i}) (1 - e (X _ {i}) ]} \leq \sigma^ {2} \mathbb {E} \left[ \frac {1}{e (X _ {i}) (1 - e (X _ {i}))} \right] = V _ {A I P W}, \tag {4.12}
$$

其中上述不等式由 **詹森不等式（Jensen’s inequality）** 得出。这一观察强调了特定目标的估计量的效率密切依赖于所做的假设。我们在第三章中展示了 AIPW 在我们通用的非参数设定中是有效的；然而，一旦我们添加了像 (4.10) 这样的额外约束，利用该约束的估计量可以表现得更好。^26

## 4.2 处理异质性（Treatment Heterogeneity）的损失函数

如果我们相信**半参数设定（semiparametric specification）** (4.4)，那么上文开发的**残差对残差回归估计量（residual-on-residual regression estimator）** 是有帮助的。然而，为了实现在具有**未混淆性（unconfoundedness）** 的通用设定中估计 **CATE** 的原始目标，我们需要将该估计量推广到一个完全**非参数（non-parametric）** 的设定。

作为如何实现这一推广的背景，回顾一下在简单预测（即根据特征 $X _ { i }$ 预测实值 $Y _ { i }$）的背景下是如何进行推广的，将是有帮助的。经典的做法是通过**线性回归（linear regression）**，但如今，诸如**决策树（decision trees）**、**提升方法（boosting）** 和**神经网络（neural networks）** 等方法提供了有吸引力的非参数替代方案。这一发展过程中的关键见解包括：使用灵活的**基展开（basis expansions）** 来表达更复杂的信号；通过**惩罚（penalization）** 来抑制学习到的预测器的复杂性，尽管使用了高维基展开；使用**交叉验证（cross-validation）** 来调整惩罚的强度；以及使用决策树和神经网络等算法技术来自适应地生成适合当前任务的基展开。Hastie、Tibshirani 和 Friedman [2009] 对这些概念进行了出色的、专著级别的阐述；其中第 3、5 和 7 章对于理解下文讨论尤为重要。

我们的任务是将所有这些概念应用到 **CATE** 估计中。为此，我们首先将上述残差对残差回归写成一个**损失最小化问题（loss-minimization problem）**。回顾一下，在简单预测的情况下，使用 $n$ 个样本将 $Y _ { i }$ 对 $\psi ( X _ { i } )$ 进行回归的普通最小二乘解 $\hat { \beta }$ 可以通过平方误差损失最小化来表征：

$$
\hat {\beta} = \operatorname{argmin} _ {\beta} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \ell_ {r e g} (Y _ {i}; \psi (X _ {i}) \cdot \beta) \right\}, \quad \ell_ {r e g} (y; z) = (y - z) ^ {2}. \tag {4.13}
$$

同理，我们可以验证，我们的残差对残差回归算法也最小化某个特定的最小二乘目标，即：27

$$
\hat {\beta} = \operatorname{argmin} _ {\beta} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \hat {\ell} ^ {(- k (i))} \left(X _ {i}, Y _ {i}, W _ {i}; \psi (X _ {i}) \cdot \beta\right) \right\} \tag {4.14}
$$

$$
\hat {\ell} ^ {(- k)} (x, y, w; z) = \left(\left(y - \hat {m} ^ {(- k)} (x)\right) - (w - \hat {e} ^ {(- k)} (x)) z\right) ^ {2}.
$$

(4.13) 和 (4.14) 的一个关键区别在于，在我们的设定中，损失函数 $\hat { \ell } ^ { ( - k ) }$ 是**数据依赖的（data-dependent）**，并将我们对 $m ( \cdot )$ 和 $e ( \cdot )$ 的**交叉拟合（cross-fitted）** 预测作为输入。我们的损失函数以这种方式依赖于数据这一事实，将在后续带来技术挑战；然而，这并不妨碍我们继续进行算法开发。

现在，我们准备将**统计学习路线图（statistical learning roadmap）** 应用于 **CATE** 估计。我们仍然从半参数设定 (4.4) 出发；然而，现在我们考虑将输入协变量 $X _ { i }$ 映射到随着样本量增长而维数越来越高的表示上的**特征化（featurizations）** $\psi : \mathcal { X } \rightarrow \mathbb { R } ^ { d _ { n } }$。例如，$\psi$ 可以包含一组项数不断增加的**多项式（polynomial）** 或**三角函数（trigonometric）** 基函数。这种方法的基本动机是，一旦我们包含了足够多的基函数，我们将能够使用这个基准确地表示任何合理的 **CATE** 函数，即对于某个 $\beta \in \mathbb { R } ^ { d _ { n } }$，有 $\tau ( x ) \approx \psi ( x ) \cdot \beta$ [Chen, 2007]。

统计学习路线图的第二步是引入**惩罚（penalization）** 来控制学习到的 **CATE** 函数的复杂性，因为当 $d _ { n }$ 相对于 $n$ 很大时，直接使用协变量 $\psi ( x )$ 运行残差对残差回归可能是不稳定的。这里的一个选择是使用**套索（lasso）** 惩罚 [Tibshirani, 1996]，它惩罚 $\beta$ 的绝对值之和：

$$
\hat {\tau} (x) = \psi (x) \cdot \hat {\beta},
$$

$$
\hat {\beta} = \operatorname{argmin} _ {\beta} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \hat {\ell} ^ {(- k (i))} \left(X _ {i}, Y _ {i}, W _ {i}; \psi (X _ {i}) \cdot \beta\right) + \lambda \sum_ {j = 1} ^ {q} | \beta_ {j} | \right\}, \tag {4.15}
$$

其中 $\lambda \geq 0$ 是一个控制学习函数复杂性的**惩罚参数（penalty parameter）**。明智地选择 $\lambda$ 使我们仍然能得到一个好的估计 $ { \hat { \tau } } ( x )$ ，但同时防范当 $\psi ( x )$ 是高维时可能出现的**过拟合（overfitting）** 或数值不稳定的风险。使用 $\lambda = 0$ 对应于仅运行 $Y _ { i }$ 对 $\psi ( X _ { i } )$ 的线性回归，而当 $\lambda \to \infty$ 时，所有系数 $\hat { \beta }$ 都会被推到 0。另一个简单的选择是使用**岭（ridge）** 惩罚，它在目标函数中添加一项 $\lambda \sum _ { j = 1 } ^ { q } \beta _ { j } ^ { 2 }$。

为了使 (4.15) 可行，我们需要一种数据驱动的方式来选择**调优参数（tuning parameter）** $\lambda$。最简单的方法是使用**验证集（validation set）**，即假设我们可以访问 $i = 1 , \ldots , n _ { v a l }$ 个可用于验证的独立数据点。为了选择 $\lambda$，我们首先在一组候选的 $\lambda$ 值网格上运行 (4.15)，得到大量候选估计 $ { \hat { \tau } } _ { \lambda } ( x )$。然后，我们选择最小化**验证损失（validation loss）** 的 $\lambda$ 值，28

$$
\hat {\lambda} = \operatorname{argmin} _ {\lambda} \left\{\frac {1}{n _ {v a l}} \sum_ {\text { validation   set }} \hat {\ell} (X _ {i}, Y _ {i}, W _ {i}; \hat {\tau} _ {\lambda} (X _ {i})) \right\}, \tag {4.16}
$$

最后使用 **CATE** 预测 $\hat { \tau } ( x ) = \hat { \tau } _ { \hat { \lambda } } ( x )$。另一种类似的、不需要访问独立验证集的选择 $\lambda$ 的方法是使用**交叉验证（cross-validation）**；详见 Hastie、Tibshirani 和 Friedman [2009] 的第 7 章。

从用于半参数建模的残差对残差回归估计量过渡到完全灵活的非参数 **CATE** 估计量的最后一步，是使用决策树、提升方法或神经网络等算法技术来自动选择好的基展开 $\psi ( x )$。然而，这样做超出了本书的范围；我们转而推荐 Nie 和 Wager [2021] 来完成这一讨论。由此产生的算法方法被称为 **R-学习器（R-learner）**。Athey、Tibshirani 和 Wager [2019] 的**因果森林（causal forest）** 算法使用**随机森林（random forests）** [Breiman, 2001] 实例化了 **R-学习器**框架。29 Foster 和 Syrgkanis [2023] 提供了一般性的形式化结果，表明即使在转向复杂的非参数设定后，**R-学习器** 仍然保持了定理 4.2 中所暗示的**稳健性（robustness）** 性质。

一个数值例子 我们现在测试基于套索的 **R-学习器** 方法 (4.15)，并将其与基于套索的 **T-学习器（T-learner）** 方法 (4.3) 进行比较，其中 $\hat { \mu } _ { ( 0 ) } ( \cdot )$ 和 $\hat { \mu } _ { ( 1 ) } ( \cdot )$ 都使用预测变量 $\psi ( X _ { i } )$ 通过套索拟合。我们独立生成 $n = 4$，000 个样本，如下所示：

$$
X \sim \mathcal {N} (0, I _ {1 0 \times 1 0}), W \sim \text { Bernoulli } (e (X)), e (X) = 1 / \left(1 + e ^ {- (X _ {2} + X _ {3})}\right)
$$

$$
Y (w) = 2 \log \left(1 + e ^ {X _ {1} + X _ {2} + X _ {3}}\right) + w \cdot 1 \left(X _ {2} + X _ {3} \geq 0\right) + \varepsilon , \varepsilon \sim \mathcal {N} (0, 1).
$$

原始协变量是 10 维的，但信号显然是非线性的，因此简单的线性方法在这里不适用。为了解决这一挑战，我们将协变量扩展为一个 2555 维的基展开 $\psi ( X _ { i } )$，其中同时包含非线性和协变量之间的**交互作用（interactions）**。30 然后，我们使用带有交叉验证选择 $\lambda$ 的套索惩罚，以避免因使用高维基展开而导致的不稳定性。

这个设定中的挑战在于，那些 $X _ { 2 } + X _ { 3 }$ 值较大的单元同时更有可能接受处理，无论是否接受处理都具有更大的基线效应，并且具有更大的处理效应。这种类型的情况可能出现在，例如，评估教育项目时，如果存在一类，比如，主动性高的人，他们同时更有可能寻找并受益于教育资源，但即使没有这些资源，他们也本可以获得相当不错的结果。在这样的设定中，为了避免**正则化引入的混杂（regularization-induced confounding）**，准确地校正**倾向得分（propensity scores）** 和基线效应之间的相关性至关重要。

**R-学习器** 和 **T-学习器** 的结果如图 4.2 所示。图的 y 轴显示 **CATE** 估计值 $\hat { \tau } ( X _ { i } )$，而 x 轴显示 $X _ { i 2 } + X _ { i 3 }$。x 轴的选择反映了这样一个事实：实际上，我们知道 **CATE** 仅随 $X _ { i 2 } + X _ { i 3 }$ 变化。当然，算法事先并不知道这一点——这就是为什么实际的 **CATE** 估计值 $\hat { \tau } ( X _ { i } )$ 也依赖于协变量的其他方面（这表现为估计中的明显噪声）。在这里，我们看到 **R-学习器** 的估计有些噪声，但正确地得到了 **CATE** 的总体数量级。相比之下，**T-学习器** 在这里似乎遭受了严重的正则化引入的混杂，并且大大高估了 $\tau ( X _ { i } )$ 随 $X _ { i 2 } + X _ { i 3 }$ 增长的程度。

## 4.3 文献注释（Bibliographic Notes）

关于**非参数 CATE 估计（non-parametric CATE estimation）** 的文献近年来受到了极大的关注。一些提出的 **CATE** 估计方法基于特定的机器学习方法，例如，**树（trees）** [Athey and Imbens, 2016]、**随机森林（random forests）** [Athey, Tibshirani, and Wager, 2019] 或**贝叶斯树集成（Bayesian tree ensembles）** [Hahn, Murray, and Carvalho, 2020]。其他方法则更为通用，可以与多种算法方法配对使用。我们在这里讨论了 **R-学习器（R-learner）** [Nie and Wager, 2021]；其他通用的 **CATE** 估计方法包括 **X-学习器（X-learner）** [Künzel et al., 2019]、**DR-学习器（DR-learner）** [Kennedy, 2023] 以及**修改协变量学习器（modified covariate learner）** [Tian et al., 2014]。

我们今天没有重点关注的一个重要主题是，在生成 **CATE** 估计之后该做什么。在拟合 **CATE** 估计量之后，通常最好寻求正式验证其输出并量化**异质性（heterogeneity）** 的强度；Chernozhukov 等人 [2017] 和 Yadlowsky 等人 [2021] 给出了一些如何做到这一点的建议。同时，如果拟合 **CATE** 模型的目标是指导**处理选择（treatment choice）**，那么命题 4.1 表明，形如 $1( \{ \hat { \tau } ( x ) > C \} )$ 的**经验阈值规则（empirical thresholding rules）** 至少值得考虑。Manski [2004]、Stoye [2009] 以及 Hirano 和 Porter [2009] 在**统计决策理论（statistical decision theory）** 的视角下研究了这种阈值规则的性质。Sun 等人 [2021] 讨论了处理成本 $C _ { i }$ 是随机的并且也可能随协变量 $X _ { i }$ 变化的情况。

在形式化结果方面，Kennedy 等人 [2024] 表明，在一组**光滑性假设（smoothness assumptions）** 下，**R-学习器** 的一个变体对于估计 **CATE** 是**极小极大最优的（minimax）**，而 Foster 和 Syrgkanis [2023] 为使用包括 **R-损失（R-loss）** 在内的一类“**正交（orthogonal）**”损失函数的机器学习提供了保证。Zhao、Small 和 Ertefaie [2022] 考虑使用一种建立在定理 4.2 中半参数估计量基础上的算法，在高维线性设定下进行 **CATE** 的**选择后推断（post-selection inference）**。

最后，我们还注意到一些基于不同概念框架的处理异质性的工作。尽管 **ITE** 通常不是**点可识别的（point-identified）**，我们仍然可以寻求它的**界（bounds）** 或**区间（intervals）**。Lei 和 Candès [2021] 提供了一种使用**共形推断（conformal inference）** 实现此目的的方法。Ding、Feller 和 Miratrix [2019] 研究了在第一章讨论的严格 **Neyman 模型（Neyman model）** 下，用于随机化推断的随机化试验中的异质性处理效应估计，并考察了在不假设潜在结果任何抽样分布的情况下，可以就处理异质性得出什么结论。
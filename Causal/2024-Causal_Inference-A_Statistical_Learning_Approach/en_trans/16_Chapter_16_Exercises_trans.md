# 第16章 练习题（Chapter 16 Exercises）

**练习1.** 考虑在定理1.2假设下的随机对照试验。我们已经知道，**均值差估计量（difference-in-means estimator）**，

$$
\hat {\tau} _ {D M} = \frac {1}{| \{i : W _ {i} = 1 \} |} \sum_ {\{i: W _ {i} = 1 \}} Y _ {i} - \frac {1}{| \{i : W _ {i} = 0 \} |} \sum_ {\{i: W _ {i} = 0 \}} Y _ {i}, \tag {16.1}
$$

在此设定下是一致的且满足中心极限定理。然而，根据我们在第2章中的讨论，我们也可以考虑 $\tau$ 的**逆概率加权估计量（inverse-propensity weighted estimator）**，

$$
\hat {\tau} _ {I P W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i} Y _ {i}}{\pi} - \frac {(1 - W _ {i}) Y _ {i}}{1 - \pi}. \tag {16.2}
$$

本题的目的是理解这两个估计量之间的关系和相对优势。

(a) 陈述并证明 $\hat {\tau} _ {I P W}$ 的中心极限定理（你可以为了此目的做出任何方便的**正则性假设（regularity assumptions）**）。将 $\hat {\tau} _ {I P W}$ 的方差与定理1.2中给出的 $\hat {\tau} _ {D M}$ 的渐近方差进行比较。  
(b) $\hat {\tau} _ {D M}$ 和 $\hat {\tau} _ {I P W}$ 的联合分布是什么？根据你的发现，你是否建议在随机研究中使用 $\hat {\tau} _ {I P W}$？

**练习2.** 第1章讨论了随机试验中线性回归调整的行为，并表明无论数据是否遵循线性设定，此类调整都可用于提高渐近精度。本题的目标是将这些结果推广到一般**非参数（nonparametric）**（或基于机器学习的）回归调整的情况。对于以下所有部分，你应在定理1.3的假设下开展工作。

(a) 如(1.27)所示，**交互回归估计量（interacted regression estimator）**可以表示为预测值的平均差异。现在假设我们设定

$$
\hat {\tau} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\mu} _ {(1)} (X _ {i}) - \hat {\mu} _ {(0)} (X _ {i})\right), \tag {16.3}
$$

但并非使用线性回归，而是从一种机器学习方法中获得 $\hat {\mu} _ {(w)} (x)$，该方法在**平方误差损失（squared-error loss）**下对(1.21)中定义的 $\mu _ {(w)} (x)$ 是一致的。以下两种说法正确还是错误？如果正确，给出证明；如果错误，给出反例。

• 估计量 $\hat {\tau}$ 是一致的。
• 估计量 $\hat {\tau}$ 是渐近正态的，即对于某个有限渐近方差 $V$，有 $\sqrt {n} (\hat {\tau} - \tau) \Rightarrow \mathcal {N} (0, V)$。

现在，我们考虑对基本估计量进行改进，通过考虑回归残差对(16.3)进行去偏，并使用**交叉拟合（cross-fitting）**来避免过拟合。我们首先将数据（随机）分成两半 $\mathcal {T} _ {1}$ 和 $\mathcal {T} _ {2}$，然后使用

$$
\begin{array}{l} \hat {\tau} _ {C F} = \frac {\hat {\tau} ^ {\mathcal {I} _ {1}} + \hat {\tau} ^ {\mathcal {I} _ {2}}}{2}, \quad \hat {\tau} ^ {\mathcal {I} _ {1}} = \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \hat {\mu} _ {(0)} ^ {\mathcal {I} _ {2}} (X _ {i}) \right. \tag {16.4} \\ \left. + \frac {W _ {i}}{\pi} \left(Y _ {i} - \hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i})\right) - \frac {1 - W _ {i}}{1 - \pi} \left(Y _ {i} - \hat {\mu} _ {(0)} ^ {\mathcal {I} _ {2}} (X _ {i})\right)\right), \\ \end{array}
$$

其中 $\hat {\mu} _ {(w)} ^ {\mathcal {I} _ {2}} (\cdot)$ 是基于样本 $\mathcal {T} _ {2}$ 对 $\mu _ {(w)} (\cdot)$ 的估计，而 $\hat {\tau} ^ {\mathcal {I} _ {2}}$ 的定义类似（$\mathcal {T} _ {1}$ 和 $\mathcal {T} _ {2}$ 的角色互换）。换句话说，$\hat {\tau} ^ {\mathcal {I} _ {1}}$ 是在 $\mathcal {T} _ {1}$ 上的处理效应估计量，它使用 $\mathcal {T} _ {2}$ 来估计其回归调整，反之亦然。

(b) 估计量(16.4)的偏误是什么，即 $\mathbb {E} \left[ \hat {\tau} _ {C F} \right] - \tau$ 是多少，其中 $\tau$ 表示**平均处理效应（Average Treatment Effect, ATE）**？  
(c) 假设我们的非参数回归调整 $\hat {\mu} _ {(w)} ^ {\mathcal {I} _ {2}} (\cdot)$ 是**风险一致的（risk-consistent）**，即

$$
\lim _{n \to \infty} \mathbb {E} \left[ \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(w)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(w)} (X _ {i})\right) ^ {2} \right] = 0, \tag {16.5}
$$

且对于 $\mathcal {T} _ {1}$ 和 $\mathcal {T} _ {2}$ 互换的情况也类似。证明 $\hat {\tau} _ {C F}$ 的中心极限定理，即证明对于某个渐近方差 $V _ {C F}$，有 $\sqrt {n} (\hat {\tau} _ {C F} - \tau) \Rightarrow \mathcal {N} \left( 0, V _ {C F} \right)$，并刻画 $V _ {C F}$ 的特征。将 $V _ {C F}$ 与(1.23)中给出的渐近方差 $V_{IREG}$ 进行比较。

(d) 考虑第1章中讨论的线性模型被良好设定的情况，

$$
Y _ {i} (w) = X _ {i} \beta_ {(w)} + \varepsilon_ {i} (w), \varepsilon_ {i} (w) \sim \mathcal {N} \left(0, \sigma^ {2}\right), \tag {16.6}
$$

并比较(16.4)在假设(16.5)下的渐近行为与第1章中讨论的**普通最小二乘法（Ordinary Least Squares, OLS）**估计量的渐近行为。一个估计量是否优于另一个？（为方便起见，你可以假设 $\pi = 0.5$ 等。）

**练习3.** 应用第2章中讨论的IPW估计量时，一个常见问题出现在某些单元先验地极不可能接受处理，且 $e (X _ {i}) \approx 0$ 时。例如，在医学应用中可能会出现这种情况，其中 $W _ {i}$ 表示候选干预措施，而某些患者根据其 $X _ {i}$ 明显健康，因此永远不会接受治疗。并且，当 $e (X _ {i})$ 可能接近0时，IPW估计量（涉及除以 $e (X _ {i})$）可能不稳定。

解决这一困难的一种方法是改变统计目标，转而关注**受试者平均处理效应（Average Treatment Effect on the Treated, ATT）**：

$$
\tau_ {A T T} = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \mid W _ {i} = 1 \right]. \tag {16.7}
$$

在许多应用中，关注ATT可以提高可用估计量的精度，并且可能也具有实质性的兴趣（因为ATT衡量了在抽样分布中接受处理的人群中处理的平均价值）。在整个问题中，你可以假设**倾向得分（propensity scores）** $e (X _ {i})$ 是先验已知的，并可用于估计，且对于某个 $\eta > 0$，有 $e (X _ {i}) \leq 1 - \eta$。你也可以将 $\mathbb {P} \left[ W _ {i} = 1 \right] = \pi$ 视为已知。

(a) 为ATT提出一个IPW风格的估计量（使用真实的倾向得分），并证明它是无偏的。  
(b) 推导(a)部分中得出的估计量的渐近方差，并为其陈述一个中心极限定理。  
(c) 在 $e (X _ {i})$ 可能非常小的设定中，比较ATE和ATT的**理想IPW估计量（oracle IPW estimators）**的渐近方差，并讨论两个估计量对小倾向得分的稳健性。

**练习4.** 在第2章中，我们定义了一个**倾向分层估计量（propensity-stratified estimator）** $\hat {\tau} _ {P S T R A T}$。本题的目的是充实我们对这个估计量的研究。你可以假设定理2.2的假设成立，我们具有**重叠性（overlap）**，即对于所有 $x \in \mathcal {X}$ 有 $\eta \le e (x) \le 1 - \eta$，倾向得分 $e (X)$ 的分布有一个在区间 $[\eta, 1 - \eta]$ 上有界远离0的密度 $f _ {e} (\cdot)$，并且结果变量是有界的 $| Y _ {i} | \le M$，其中M为某个大常数。

(a) 证明如果对于某个常数 $0 < \rho < 1$ 有 $J = n ^ {\rho}$，那么使用真实倾向得分实现的估计量 $\hat {\tau} _ {P S T R A T}$ 是一致的，即 $\hat {\tau} _ {P S T R A T} \xrightarrow{p} \tau$，其中 $\tau$ 是平均处理效应。

(b) 进行模拟研究以评估逆概率加权和分层的优缺点。在R中按如下方式生成数据，对于 n = 100, 200, 400, 800, 1600, 3200 且 $p = 10$：

$$
\begin{array}{l} X = \text { matrix } (\text { runif } (n * p, - 1, 1), n, p) \\ \text { propensity } = 0.1 + 0.85 * \operatorname{sqrt} (\operatorname{pmax} (0, 1 + X [, 1 ] + X [, 2 ]) / 3) \\ W = \text { rbinom } (n, 1, \text { propensity }) \\ Y = W * \operatorname{pmax} (0, X [, 1 ]) + \exp (X [, 2 ] + X [, 3 ]) \\ \end{array}
$$

通过逻辑回归拟合倾向得分 $\hat {e}$，然后使用拟合的倾向得分通过 $\hat {\tau} _ {I P W}$ 和 $\hat {\tau} _ {P S T R A T}$ 估计 $\tau$。

在这个模拟设计中，平均处理效应 $\tau$ 是多少？$J$ 的一个好的选择是什么？在偏误方面，$\hat {\tau} _ {I P W}$ 的表现与 $\hat {\tau} _ {P S T R A T}$ 相比如何？在**均方误差（mean-squared error）**方面呢？一个好的分析应依赖于足够的模拟重复次数以减少蒙特卡罗效应带来的不确定性，并通过适当的可视化展示结果。

(c) 证明对于适当选择的序列 $J (n)$，倾向分层估计量（现在再次使用真实倾向得分实现）是渐近无偏和高斯的，即 $\sqrt {n} (\hat {\tau} _ {P S T R A T} - \tau) \Rightarrow \mathcal {N} (0, V _ {P S T R A T})$。为 $V _ {P S T R A T}$ 提出一个一致的方差估计量 $\widehat{V} _ {P S T R A T}$，使得 $\widehat{V} _ {P S T R A T} / V _ {P S T R A T} \xrightarrow{p} 1$。讨论如何利用这些结果构建以 $\hat {\tau} _ {P S T R A T}$ 为中心的 $\tau$ 的置信区间。

(d) 在第3章中，我们展示了如何通过回归调整来"增强"逆概率加权ATE估计量，并表明所得的**增强型逆概率加权（Augmented Inverse Probability Weighting, AIPW）**估计量相对于基本的IPW估计量具有改进的稳健性和精度属性。你将如何类似地"增强"这里研究的倾向分层估计量？提出一个估计量，并为其论证。（注意：你的论证不需要是正式的；一个简短的性质论证就足够了。）

**练习5.** 在推论4.3中，我们给出了**残差对残差估计量（residual-on-residual estimator）**的渐近性质，

$$
\hat {\tau} _ {R} = \frac {\sum_ {i = 1} ^ {n} \left(Y _ {i} - \hat {m} ^ {(- k (i))} (X _ {i})\right) \left(W _ {i} - \hat {e} ^ {(- k (i))} (X _ {i})\right)}{\sum_ {i = 1} ^ {n} \left(W _ {i} - \hat {e} ^ {(- k (i))} (X _ {i})\right) ^ {2}}, \tag {16.8}
$$

用于在常数处理效应模型 $Y _ {i} (w) = f (X _ {i}) + w \tau + \varepsilon _ {i}$ 下估计处理参数 $\tau$。本题的目的是在常数处理效应假设被错误设定的情况下研究这个相同的残差对残差估计量。假设数据是独立生成的：

$$
\begin{array}{l} Y _ {i} (w) = \mu_ {(w)} \left(X _ {i}\right) + \varepsilon_ {i} (w), \quad \mathbb {E} \left[ \varepsilon_ {i} (w) \mid X _ {i} = x, W _ {i} = w \right] = 0, \\ \end{array} \tag {16.9}
$$

$$
\mathrm{Var} \left[ \varepsilon_ {i} (w) \mid X _ {i} = x, W _ {i} = w \right] = \sigma^ {2},
$$

并记 $\tau (x) = \mu _ {(1)} (x) - \mu _ {(0)} (x)$。我们的目标是刻画在模型(16.9)下 $\hat {\tau} _ {R}$ 的渐近行为。在整个问题中，你可以假设 $e (x) \in (0, 1)$；然而，重叠性不是必需的。

(a) 令 $\hat {\tau} _ {R} ^ {*}$ 为估计量(16.8)的"理想"版本，使用真实的 $m (x)$ 和 $e (x)$ 计算。证明 $\hat {\tau} _ {R} ^ {*}$ 依概率收敛到一个极限 $\tau _ {R}$，该极限是**条件平均处理效应（Conditional Average Treatment Effect, CATE）** $\tau (x)$ 的非负加权平均，即 $\tau _ {R} = \mathbb {E} \left[ \gamma (X _ {i}) \tau (X _ {i}) \right]$，其中 $\gamma (\boldsymbol{x}) \ge 0$ 且 $\mathbb {E} \left[ \gamma (X _ {i}) \right] = 1$。  
(b) 证明这个理想估计量满足中心极限定理 $\sqrt {n} (\hat {\tau} _ {R} ^ {*} - \tau _ {R}) \Rightarrow \mathcal {N} \left( 0, V _ {R} \right)$，并提供 $V _ {R}$ 的表达式。$V _ {R}$ 与平均处理效应估计的半参数有效方差相比如何？  
(c) 假设 $\hat {m} (X _ {i})$ 和 $\hat {e} (X _ {i})$ 满足速率条件(4.7)。证明 $\sqrt {n} (\hat {\tau} _ {R} - \hat {\tau} _ {R} ^ {*}) \xrightarrow{p} 0$，因此可行的估计量(16.8)也满足(b)部分中建立的中心极限定理。

**练习6.** 考虑一家假设的公司，该公司拥有一个手机应用程序，用于提供 $K > 3$ 种不同的产品供客户选择购买。然而，考虑到手机屏幕的大小，它只能在任何给定时间向用户显示3个（排序的）推荐。你的目标是帮助该平台评估不同的排序策略如何影响绩效。

你拥有 $i = 1, \dots, n$ 个独立同分布（IID）的客户数据，这些客户曾与该平台互动。对于每个客户，该平台：

• 计算得分 $S _ { i 1 } , \ldots , S _ { i K } > 0$ ，反映每个产品对第 $i$ 位顾客的适配程度。（这些得分由你无法访问的某种黑箱算法计算得出，但它们被记录并包含在你的数据集中。）

• 随机选择第一个展示的产品 $A _ { i } ^ { ( 1 ) }$ ，使得

$$
\mathbb {P} \left[ A _ {i} ^ {(1)} = k \right] = e ^ {S _ {i, k}} / \sum_ {\ell = 1} ^ {K} e ^ {S _ {i, \ell}} \text {对于所有} k = 1, \ldots , K.
$$

• 随机选择第二个展示的产品 $A _ { i } ^ { ( 2 ) }$ ，使得

$$
\mathbb {P} \left[ A _ {i} ^ {(2)} = k \right] = e ^ {S _ {i, k}} \big / \sum_ {\ell \neq A _ {i} ^ {(1)}} e ^ {S _ {i, \ell}} \text {对于所有} k \neq A _ {i} ^ {(1)}.
$$

• 随机选择第三个展示的产品 $A _ { i } ^ { ( 3 ) }$ ，使得

$$
\mathbb {P} \left[ A _ {i} ^ {(3)} = k \right] = e ^ {S _ {i, k}} \big / \sum_ {\ell \neq A _ {i} ^ {(1)}, A _ {i} ^ {(2)}} e ^ {S _ {i, \ell}} \text {对于所有} k \neq A _ {i} ^ {(1)}, A _ {i} ^ {(2)}.
$$

• 观测到奖励 $Y _ { i } .$

为回答以下问题，你应该假设展示给用户的精确排序 ${ \bf \bar { \chi } } _ { i } ^ { ( 1 ) ^ { \scriptstyle \bullet } } , { \bf \Phi } _ { A _ { i } } ^ { ( 2 ) } , { \bf \Phi } _ { A _ { i } } ^ { ( 3 ) }$ 是重要的。注意平台不对其他产品进行排序（例如，你可以假设如果顾客想选择其他产品，他们需要通过导航到一个按字母顺序显示产品的独立静态列表来实现）。

我们将（随机和确定性的）产品排序方法称为**策略（policies）**，并将平台通过部署某个策略所能实现的期望奖励称为该策略的**值（value）** $V$ 。可用数据

$$
\mathcal {D} _ {n} = \left\{S _ {i}, A _ {i} ^ {(1)}, A _ {i} ^ {(2)}, A _ {i} ^ {(3)}, Y _ {i} \right\} _ {i = 1} ^ {n}
$$

按上述方式生成，对于以下所有 4 个部分都是相同的。策略值 $V$ 的**无偏估计量（unbiased estimator）** 是观测数据 $\mathcal { D } _ { n }$ 的一个（可测）函数 $\widehat { V }$ ，满足 $\mathbb { E } [ \tilde { V } ] = V$ 。我们假设每个单元都有潜在结果 $Y _ { i } ( a _ { 1 } , a _ { 2 } , a _ { 3 } )$ ，使得观测到的奖励为

$$
Y _ {i} = Y _ {i} \left(A _ {i} ^ {(1)}, A _ {i} ^ {(2)}, A _ {i} ^ {(3)}\right),
$$

而策略 $\pi$ 的值为

$$
V (\pi) = \mathbb {E} _ {A _ {i} \sim \pi (S _ {i})} \left[ Y _ {i} (A _ {i}) \right], \quad A _ {i} = \left(A _ {i} ^ {(1)}, A _ {i} ^ {(2)}, A _ {i} ^ {(3)}\right),
$$

其中 $A _ { i } \ \sim \ \pi ( S _ { i } )$ 表示 $A _ { i }$ 是通过 $S _ { i }$ 的（可能随机的）函数 $\pi$ 生成的。

(a) 提出一个估计量，在给定可用数据 $\mathcal { D } _ { n }$ 的情况下，能对当前随机化策略（即数据收集所使用的策略）的值给出无偏估计。  
(b) 提出一个估计量，在给定可用数据 $\mathcal { D } _ { n }$ 的情况下，能对始终使用固定排序 $a _ { 1 }$ , a2, a3（即对于某个 $1 \leq a _ { 1 } \neq a _ { 2 } \neq a _ { 3 } \leq K$ ，设定 $A _ { i } ^ { ( 1 ) } = a _ { 1 } , A _ { i } ^ { ( 2 ) } = \stackrel { \cdot } { a } _ { 2 } , \stackrel { \cdot } { A } _ { i } ^ { ( c ) } = a _ { 3 }$ ）的策略的值给出无偏估计。  
(c) 提出一个估计量，在给定可用数据 ${ \mathcal { D } } _ { n }$ 的情况下，能对始终先展示某个产品 $a _ { 1 }$（即确定性地设定 $A _ { i } ^ { ( 1 ) } = a _ { 1 }$ 对于某个 $1 \leq a _ { 1 } \leq K$ ），然后使用可用得分以与数据收集策略相同的方式随机选择 $A _ { i } ^ { ( 2 ) }$ 和 $A _ { i } ^ { ( 3 ) }$ 的随机化策略的值给出无偏估计。

(d) 提出一个估计量，在给定可用数据 $\mathcal { D } _ { n }$ 的情况下，能对从不展示某个产品 $a _ { 0 }$（其中 $1 \le a _ { 0 } \le K$ ），但除此之外使用得分以与数据收集策略相同的方式随机抽取产品的随机化策略的值给出无偏估计（在操作上，你可以假设 $A _ { i } ^ { ( \ell ) }$ 具有与 $\bar { a _ { 0 } }$ 相同的分布，直到 $A _ { i } ^ { \overline { { ( \ell ) } } } \neq a _ { 0 }$ ）。

**练习 7（Exercise 7）**。考虑以下自适应数据收集模型（ $\eta > 0$ 是一个调优参数）：对于 $t = 1 , \dots , T$ 个时间步，我们

• 选择一个概率 $\omega _ { t } \in [ \eta , 1 ]$ ，可能使用历史数据。
• 抽取一个伯努利随机变量 $Z _ { t } \sim \mathrm { B e r n } ( \omega _ { t } )$
• 如果 $Z _ { t } = 1$ ，我们观测到一次抽取 $Y _ { t } \sim F$ ；而如果 $Z _ { t } = 0$ ，我们无法进行观测（等价地，我们硬编码 $Y _ { t } = 0 $）。

我们的目标是估计均值 $\mu = \mathbb { E } _ { F } [ Y ]$ ，并考虑以下 3 个不同的估计量：

1. **样本均值（Sample average）**： $\begin{array} { r } { \hat { \mu } _ { 1 } = \sum _ { \{ t : Z _ { t } = 1 \} } Y _ { t } / \left| \left\{ t : Z _ { t } = 1 \right\} \right| } \end{array}$

2. **逆概率加权（Inverse-propensity weighting）**： $\begin{array} { r } { \hat { \mu } _ { 2 } = T ^ { - 1 } \sum _ { t = 1 } ^ { T } Z _ { t } Y _ { t } / \omega _ { t } } \end{array}$

3. **稳定化逆概率加权（Stabilized inverse-propensity weighting）**： $\hat { \mu } _ { 3 }$ $\begin{array} { r } { \sum _ { t = 1 } ^ { T } Z _ { t } Y _ { t } / \omega _ { t } \ : / \ : \sum _ { t = 1 } ^ { T } \dot { Z } _ { t } \dot { / } \omega _ { t } . } \end{array}$

回答以下问题。为避免退化情况，你可以假设 $\omega _ { 1 } = 1$ ，即我们总是至少收集 1 个样本。你也可以做出任何你认为方便的规律性假设（例如， $Y _ { t }$ 有界支撑）。

(a) 上述 3 个估计量中哪些是无偏的，即满足 $\mathbb { E } \left[ \hat { \mu } \right] = \mu$ ？提供证明或反例。  
(b) 现在考虑大样本极限， $T \to \infty$ 。在此设定下，如果

$$
\lim _ {T \to \infty} \sqrt {T} \left(\mathbb {E} [ \hat {\mu} ] - \mu\right) = 0,
$$

则称一个估计量是**渐近无偏（asymptotically unbiased）** 的。上述 3 个估计量中哪些是渐近无偏的？提供证明或反例。

**练习 8（Exercise 8）**。定理 7.1 给出了在线性-逻辑斯蒂规范下的**协变量平衡倾向得分估计量（covariate-balancing propensity score estimator）** $\hat { \tau } _ { C B P S }$ 的渐近分布，其中

$$
\mu_ {(w)} = x \cdot \beta_ {(w)}, \quad \beta_ {(w)} \in \mathbb {R} ^ {p} \quad \text { 对于 } w = 0, 1, \tag {16.10}
$$

$$
e (x) = 1 / \left(1 + e ^ {- x \cdot \theta}\right), \quad \theta \in \mathbb {R} ^ {p}, \quad \| \theta \| _ {2} < \infty . \tag {16.11}
$$

本问题的目标是研究 $\hat { \tau } _ { C B P S }$ 的**双重稳健性（double robustness）** 性质。在回答此问题时，你可以将指数矩条件 (7.12) 替换为更强的有界性条件 $\| X _ { i } \| _ { 2 } \leq M$ 。

(a) 在定理 7.1 的设定下，假设 (16.10) 成立但 (16.11) 可能不成立。证明 $\hat { \tau } _ { C P B S } \to _ { p } \tau$ ，其中 $\tau$ 表示**平均处理效应（ATE）**。如果方便，你可以假设强重叠性成立， $\eta \leq e ( X _ { i } ) \leq 1 - \eta$ 。  
(b) 在定理 7.1 的设定下，反过来假设 (16.11) 成立但 (16.10) 可能不成立。证明 $\hat { \tau } _ { C P B S } \to _ { p } \tau$ 。如果方便，你可以假设结果是有界的， $| Y _ { i } | \le M$ 。

**练习 9（Exercise 9）**。在定理 7.1 的条件下，假设我们想要估计的不是 ATE，而是如练习 3 中的**处理组平均处理效应（ATT）** $\tau _ { A T T } = \mathbb { E } \left[ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \big | W _ { i } = 1 \right]$ 。我们声称

$$
\hat {\theta} = \operatorname{argmin} _ {\theta} \left\{\frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} \left(\left(1 - W _ {i}\right) e ^ {X _ {i} \theta} - W _ {i} X _ {i} \theta\right) \right\}, \tag {16.12}
$$

$$
\hat {\tau} _ {C B P S - A T T} = \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} \left(W _ {i} Y _ {i} - (1 - W _ {i}) e ^ {X _ {i} \hat {\theta}} Y _ {i}\right), \tag {16.13}
$$

是此任务的**自然 CBPS 估计量（natural CBPS estimator）**，并具有良好的统计性质。

(a) 验证 (16.12) 是一个凸最小化问题。  
(b) 验证 (16.13) 实际上是一个 CBPS 估计量，即它是针对某个特定选择 $\hat { e } ( x ) = 1 / \left( 1 + e ^ { x \hat { \theta } } \right)$ 的 IPW 估计量，并且当最小化问题 (16.12) 有内点解时（即 $\lVert \hat { { \boldsymbol { \theta } } } \rVert < \infty$ ）， $\hat { \theta }$ 满足一个相关的样本平衡条件。  
(c) 证明 $\hat { \tau } _ { C B P S - A T T }$ 是 $\tau _ { A T T }$ 的相合估计量，并建立一个中心极限定理。为简化起见，你可以假设 $\| X _ { i } \| _ { 2 } \leq M$ 一致成立。

**练习 10（Exercise 10）**。考虑一个独立同分布序列 $( X _ { i } , U _ { i } , Y _ { i } , W _ { i } ) \in \mathcal { X } \times \mathcal { U } \times \mathbb { R } \times \{ 0 , 1 \}$ ，其中 $Y _ { i } = Y _ { i } ( W _ { i })$ 对应一对潜在结果 $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \}$ 。在给定 $X _ { i }$ 和 $U _ { i }$ 的条件下，**无混杂性（unconfoundedness）** 成立，即

$$
\{Y _ {i} (0), Y _ {i} (1) \} \perp W _ {i} \mid X _ {i}, U _ {i}. \tag {16.14}
$$

然而，只有 $X _ { i }$ 被观测到，而 $U _ { i }$ 是一个未观测到的混杂变量。在本问题中，我们将研究在存在未观测混杂的情况下， $\mu ( 1 ) =$ $\mathbb { E } \left[ Y _ { i } ( 1 ) \right]$ 的（稳定化）IPW 估计量的行为。为此，定义可行的和不可行的 IPW 估计量，后者使用了未观测到的 $U _ { i }$ ：

$$
\hat {\mu} _ {S I P W} (1) = \sum_ {i = 1} ^ {n} \frac {W _ {i} Y _ {i}}{e \left(X _ {i}\right)} / \sum_ {i = 1} ^ {n} \frac {W _ {i}}{e \left(X _ {i}\right)}, \tag {16.15}
$$

$$
\tilde {\mu} _ {S I P W} (1) = \sum_ {i = 1} ^ {n} \frac {W _ {i} Y _ {i}}{e (X _ {i} , U _ {i})} \Bigg / \sum_ {i = 1} ^ {n} \frac {W _ {i}}{e (X _ {i} , U _ {i})},
$$

其中 $e ( x ) = \mathbb { P } \left[ W _ { i } = 1 \big | X _ { i } = x \right]$ 且 $e ( x , u ) \ : = \ : \mathbb { P } \lceil W _ { i } = 1 \rceil X _ { i } = x , U _ { i } = u \rceil$ 。在**无混杂条件（unconfoundedness condition）** (16.14) 下，˜µSIPW (1) 对 $\mu ( 1 )$ 显然是一致的，但 ${ \hat { \mu } } _ { S I P W } ( 1 )$ 可能并非如此。

一般而言，我们无法对 ${ \hat { \mu } } _ { S I P W } ( 1 )$ 的偏差做出太多论断。因此，我们将对未观测到的 $U _ { i }$ 如何影响抽样概率做出进一步假设，并假设我们知道一个常数 $\Gamma \geq 1$ ，使得

$$
\frac {1}{\Gamma} \leq \frac {e (X _ {i} , U _ {i})}{e (X _ {i})} \leq \Gamma \text {   for   all   } i = 1,..., n, \tag {16.16}
$$

几乎必然成立。这一假设通常被称为**边际敏感性模型（marginal sensitivity model）**，可用于评估 IPW 对隐藏混杂的敏感性。

(a) 在 (16.16) 条件下，证明存在权重 $\Gamma _ { i } ^ { - 1 } \leq \gamma _ { i } \leq \Gamma _ { i }$ 使得

$$
\tilde {\mu} _ {S I P W} (1) = \hat {\mu} _ {S I P W} (1; \gamma) := \sum_ {i = 1} ^ {n} \gamma_ {i} \frac {W _ {i} Y _ {i}}{e (X _ {i})} / \sum_ {i = 1} ^ {n} \gamma_ {i} \frac {W _ {i}}{e (X _ {i})}. \tag {16.17}
$$

(b) 给定 (16.17)，我们对 $\tilde { \mu } _ { S I P W } ( 1 )$ 有以下上界：

$$
\hat {\mu} _ {S I P W} ^ {+} (1) = \sup \left\{\hat {\mu} _ {S I P W} (1; \gamma): \Gamma_ {i} ^ {- 1} \leq \gamma_ {i} \leq \Gamma_ {i} \right\}. \tag {16.18}
$$

证明上述优化问题可以通过**线性规划（linear programming）**求解，并将该问题表示为可以代入标准线性规划软件的形式，即格式为“在给定 $A ,$ b 和 c 的情况下，对向量 x 进行优化，最大化 $c ^ { \prime } x$ 并满足 $A x \le b ^ { \prime }$ ”。

提示：考虑用于**线性分式规划（linear-fractional programming）**的**查恩斯-库珀变换（Charnes-Cooper transformation）**。

(c) 使用 (16.18) 中的构造，提出一个区间

$$
\widehat {I} _ {S I P W} (1) = \left[ \hat {\mu} _ {S I P W} ^ {-} (1), \hat {\mu} _ {S I P W} ^ {+} (1) \right] \tag {16.19}
$$

该区间不使用未观测到的 $U _ { i }$ ，但具有性质 $\tilde { \mu } _ { S I P W } ( 1 ) \in \widehat { I } _ { S I P W } ( 1 )$ 几乎必然成立。证明区间 $\hat { I } _ { S I P W } ( \bar { 1 } )$ 在以下意义上对 $\mu ( 1 )$ 是一致的：对于任意 $\varepsilon > 0$

$$
\lim _ {n \rightarrow \infty} \mathbb {P} [ \mu (1) \in (\hat {\mu} _ {S I P W} ^ {-} (1) - \varepsilon , \hat {\mu} _ {S I P W} ^ {+} (1) + \varepsilon) ] = 1. \tag {16.20}
$$

在此过程中，你可以做出任何你认为方便的正则性假设（例如，矩的有界性）。

(d) 讨论区间 (16.19) 如何在实际数据分析中用于评估 IPW 对潜在未观测混杂因素存在的敏感性。

**练习 11.** 考虑以下结构模型，其中 $( X _ { i } , Y _ { i } , W _ { i } , Z _ { i } ) \in \mathcal { X } \times \mathbb { R } \times \{ 0 , 1 \} \times \{ 0 , 1 \}$ 被假定为独立同分布（IID）：

$$
\begin{array}{l} Y _ {i} = \alpha \left(X _ {i}\right) + W _ {i} \tau \left(X _ {i}\right) + \varepsilon_ {i}, \quad \varepsilon_ {i} \perp Z _ {i} \mid X _ {i}, \quad \mathbb {E} \left[ \varepsilon_ {i} \mid X _ {i} \right] = 0 \\ C = \left[ W _ {i} - Z _ {i} \mid X _ {i} - \dots \right] > 0, f (x) = 1, \dots , y. \end{array} \tag {16.21}
$$

$\mathrm { ~ \mathsf { C o v } ~ } \lfloor W _ { i } , \mathrm { ~ } Z _ { i } \rfloor \lambda _ { i } = x \rfloor \ge \eta > 0 \mathrm { ~ \quad ~ f o r ~ a u l ~ } x \in \mathcal { A } .$

换句话说，在条件于协变量 $X _ { i }$ 的情况下，这与第 $9 . 2$ 章使用的结构模型相同；然而，现在所有问题原始参数也可能随 x 变化。此外，我们假设工具变量对结果的影响始终为正且一致有下界。

你的目标是开发方法来估计**平均处理效应（average treatment effect）**参数 $\tau = \mathbb { E } \left[ \tau ( X ) \right]$ 。在以下所有部分中，你可以做出任何你认为有帮助的正则性假设（例如，结果的有界性）。

(a) 定义“**依从性得分（compliance score）**” $\Delta ( x )$ 以及相关的**逆依从性加权估计量（inverse-compliance weighted estimator）**，

$$
\Delta (x) = \mathbb {P} \left[ W _ {i} = 1 \mid Z _ {i} = 1, X _ {i} = x \right] - \mathbb {P} \left[ W _ {i} = 1 \mid Z _ {i} = 0, X _ {i} = x \right],
$$

$$
\hat {\tau} _ {I C W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {1}{\Delta (X _ {i})} \left(\frac {Z _ {i} Y _ {i}}{z (X _ {i})} - \frac {(1 - Z _ {i}) Y _ {i}}{1 - z (X _ {i})}\right), \tag {16.22}
$$

其中 $z ( x ) = \mathbb { P } \left[ Z _ { i } = 1 \big | X _ { i } = x \right]$ 是工具变量 $Z _ { i }$ 的**倾向得分（propensity score）**的类似物。证明**神谕逆依从性加权估计量（oracle inverse-compliance weighted estimator）**（即使用 $z ( \cdot )$ 和 $\Delta ( \cdot )$ 的真实值）对于 $\tau$ 是无偏且一致的。

(b) 现在假设你获得了结构参数在 (16.21) 中的估计值 ${ \hat { \alpha } } ( x )$ 和 ${ \hat { \tau } } ( x )$ 。提出一个**增强逆依从性加权（augmented inverse-compliance weighted, AICW）**估计量。论证你的 AICW 估计量是（弱）**双重稳健（doubly robust）**的，即如果 ${ \hat { \alpha } } ( x )$ 和 $\hat { \tau } ( x )$ 是**上确界范数一致（sup-norm consistent）**的，或者 $\widehat { \Delta } ( x )$ 和 $\hat { z } ( x )$ 是上确界范数一致的（其中 $\widehat { \Delta } ( x )$ 和 $\hat { z } ( x )$ 是 (16.22) 中干扰分量的可行估计），那么它是一致的。这里给出一个高层次的论证即可；无需深入细节。79  
(c) 证明如果所有干扰分量 $\hat { \alpha } ( x ) , \hat { \tau } ( x ) , \hat { \Delta } ( x )$ 和 $\hat { z } ( x )$ 既是上确界范数一致的，并且在**均方根误差（root-mean squared error）**上具有 $o _ { p } ( n ^ { - 1 / 4 } )$ 一致性，那么使用**交叉拟合（cross-fitting）**的 AICW 对于 $\tau$ 是 $\sqrt{n}$ 一致且渐近正态的。写出一个**中心极限定理（central limit theorem）**，并提供 AICW 的**极限方差（limiting variance）**表达式。

**练习 12.** 在第 10.1 章中，我们研究了具有二元处理和二元工具变量的工具变量回归。我们证明了在“**无违抗者（no defiers）**”假设下，即

$$
\mathbb {P} \left[ W _ {i} (0) <   W _ {i} (1) \right] = 0, \tag {16.23}
$$

工具变量估计量收敛于**依从者（compliers）**的平均处理效应估计量。你在这个问题中的目标是理解当我们放宽这个假设时会发生什么。

在定理 10.1 的设定下，现在假设我们可能有违抗者，但存在未观测到的潜在因子 $U _ { i }$ ，使得

$$
\mathbb {P} \left[ W _ {i} = 1 \mid Z _ {i} = 1, U _ {i} = u \right] > \mathbb {P} \left[ W _ {i} = 1 \mid Z _ {i} = 0, U _ {i} = u \right], \tag {16.24}
$$

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp C _ {i} \mid U _ {i} = u, \text {for all} u,
$$

即，给定未观测到的潜在因子，我们假设处理效应与**依从类型（compliance type）**无关，并且所有潜在类型更可能依从而非违抗。同时假设一旦我们将 $U _ { i }$ 纳入模型，$Z _ { i }$ 仍然是外生的，

$$
Z _ {i} \perp \left\{U _ {i}, Y _ {i} (0), Y _ {i} (1), W _ {i} (0), W _ {i} (1) \right\}.
$$

用以下项写出 $\tau _ { L A T E }$ 的表达式

$$
\tau (u) = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \mid U _ {i} = u \right],
$$

$$
\kappa (u) = \mathbb {P} \left[ C _ {i} = \text {complier} \mid U _ {i} = u \right], \text {and}
$$

$$
\delta (u) = \mathbb {P} \left[ C _ {i} = \text { defier } \mid U _ {i} = u \right].
$$

证明，如果对所有 u 都有 $\tau ( u ) \geq 0$ ，那么 $\tau _ { L A T E } \geq 0$

**练习 13.** 考虑一组 n 个随机变量 $( W _ { i } , Y _ { i } ) \in \{ 0 , 1 \} \times \mathbb { R }$ 假设数据生成如下：

• 每个单元 $i = 1 , \ldots , n$ 由（确定性的）参数 $\alpha _ { i } .$ $\beta _ { i } , \gamma _ { i } \in \mathbb { R }$ 刻画。
• 我们选择一个处理概率 $\pi \in [ 0 , 1 ]$ ，并独立地为每个 $i = 1 , \ldots , n$ 生成 $W _ { i } \sim \mathrm { B e r n o u l l i } ( \pi )$ 。
• 我们观测到以下结果，其中 $\varepsilon _ { i } \sim \mathcal { N } \left( 0 , \sigma ^ { 2 } \right)$ 独立于其他所有变量：

$$
Y _ {i} = \alpha_ {i} + \beta_ {i} W _ {i} + \gamma_ {i} \frac {\sum_ {j \neq i} W _ {j}}{n - 1} + \varepsilon_ {i}
$$

我们使用符号 $\mathbb { E } _ { \pi } \left[ Y _ { i } \right]$ 表示在此模型下（处理概率为 $\pi )$ 第 i 个结果的期望，以及该符号的直接推广。注意：定性地说，$\alpha _ { i }$ 捕捉第 i 个单元的**基线效应（baseline effect）**，$\beta _ { i }$ 捕捉其对自身处理的敏感性，$\gamma _ { i }$ 捕捉其对其他被处理单元比例的敏感性。

(a) **总效应（total effect）**是什么？即所有人被处理与无人被处理时平均结果的期望差异：

$$
\tau_ {T O T} = \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} _ {1} [ Y _ {i} ] - \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} _ {0} [ Y _ {i} ].
$$

(b) 现在假设我们能够在单个 $\pi \in ( 0 , 1 )$ 处收集观测值，并试图通过忽略**溢出效应（spillovers）**的**朴素逆倾向得分加权估计量（naïve inverse-propensity weighted estimator）**来估计处理效应，

$$
\hat {\tau} _ {I P W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{\pi} - \frac {(1 - W _ {i}) Y _ {i}}{1 - \pi}\right).
$$

$\mathbb { E } _ { \pi } \left[ \hat { \tau } _ { I P W } \right]$ 是什么？

(c) 在与上述相同的设定下，$\operatorname { V a r } _ { \boldsymbol { \pi } } \left[ \hat { \tau } _ { I P W } \right]$ 是什么？  
(d) 在这个模型中，${ \hat { \tau } } _ { I P W }$ 是 $\tau _ { T O T }$ 的一个好估计量吗？在这个模型中，${ \hat { \tau } } _ { I P W }$ 能否用于学习任何有趣的东西？

**练习 14.** 生存分析中的一个重要问题是在给定诊断结果后评估预后。我们有 $i = 1 , \dots , n$ 个在时间 $t = 0$ 被诊断出某种疾病的人的数据；此时，我们还测量了**时不变协变量（time-invariant convariates）** $X _ { i } \in { \mathcal { X } }$ 。我们记 $Y _ { i }$ 为第 i 个人诊断后的生存时间长度，并关注于为某个目标时间范围 $T$ 估计 $\theta = \mathbb { P } \left[ Y _ { i } > T \right]$ 。

然而，挑战在于，我们可能会在研究过程中失去一些患者的追踪，无法观察到他们是否活过时间 $T$ 。具体来说，我们假设在预定时间点 $t = 1 , \dots , T$ 对每位患者进行随访，在每次随访中，我们要么能够找到该患者（在这种情况下，我们可以观察到患者是否还活着，即 $Y _ { i } > t$ 是否成立），要么无法找到该患者，并认为他们在时间 t 被**删失（censored）**（并且我们停止进一步的随访尝试）。

形式上，我们假设每个单元都有一个（可能未实现的）**删失时间（censoring time）** $C _ { i } \in \{ 1 , 2 , . . . , T , + \infty \}$ ，其中 $C _ { i } = + \infty$ 表示该单元从未被删失。然后我们假设，我们无法直接观测到生存时间 $Y _ { i }$ ，而只能访问到

$$
U _ {i} = \min \left\{C _ {i}, Y _ {i} \right\}, \quad \Delta_ {i} = 1 \left(Y _ {i} <   C _ {i}\right), \tag {16.25}
$$

我们分别称之为**观测时间（observation time）**和**非删失指示符（non-censoring indicator）**。令

$$
\overline {{U}} _ {i} = \inf \left\{t \in \{1, 2, \dots , T, + \infty \}: t \geq U _ {i} \right\}, \quad H _ {i} = \min \left\{\overline {{U}} _ {i}, T \right\}, \tag {16.26}
$$

分别表示记录观测结果的随访时间点（例如，如果某人在时间 1.5 死亡，我们只在 $t = 2$ 的随访时得知此事），以及最后一次访视的时间（即，即使患者在此时点仍然存活且未被删失，$H _ { i } = T$ ）。

我们还做出以下统计假设：

• 删失是可忽略的，即

$$
Y _ {i} \perp C _ {i} \mid X _ {i}; \tag {16.27}
$$

• 一些患者从未被删失，即存在一个 $\eta > 0$ 使得

$$
\mathbb {P} \left[ C _ {i} > T \mid X _ {i} = x \right] \geq \eta \text {   for   all   } x \in \mathcal {X}. \tag {16.28}
$$

注意，这些假设与我们熟悉的用于处理效应估计的**无混杂性（unconfoundedness）**和**重叠性（overlap）**假设密切相关。

我们定义**条件生存函数（conditional survival functions）**

$$
S _ {Y} (t; x) = \mathbb {P} \left[ Y _ {i} > t \mid X _ {i} = x \right], \quad S _ {C} (t; x) = \mathbb {P} \left[ C _ {i} > t \mid X _ {i} = x \right], \tag {16.29}
$$

并约定 $S _ { Y } ( 0 ; x ) = S _ { C } ( 0 ; x ) = 1$ 。我们假设我们可以使用一个独立的训练集来获得这些对象的估计值。80

(a) 假设删失分布的生存函数 $S _ { C } ( t ; x )$ 是已知的。证明，在我们的假设下，以下**逆删失概率加权（inverse-probability of censoring, IPCW）**估计量对于 $\theta$ 是无偏的：

$$
\hat {\theta} _ {I P C W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {\Delta_ {i} 1 (\{U _ {i} > T \})}{S _ {C} (U _ {i} ; X _ {i})}. \tag {16.30}
$$

(b) 现在，考虑一个设定，其中我们可以使用一个独立的训练集获得估计值 $\widehat { S } _ { Y } ( t ; x )$ 和 $\widehat { S } _ { C } ( t ; x )$ ，并考虑以下**增强逆删失概率加权（augmented IPCW, AIPCW）**估计量：81

$$
\hat {\theta} _ {A I P C W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \widehat {S} _ {Y} (T; X _ {i})
$$

$$
+ \sum_ {t = 1} ^ {H _ {i} - 1} \frac {1}{\widehat {S} _ {C} (t ; X _ {i})} \left(\frac {\widehat {S} _ {Y} (T ; X _ {i})}{\widehat {S} _ {Y} (t ; X _ {i})} - \frac {\widehat {S} _ {Y} (T ; X _ {i})}{\widehat {S} _ {Y} (t - 1 ; X _ {i})}\right) \tag {16.31}
$$

$$
+ \frac {\Delta_ {i}}{\widehat {S} _ {C} (H _ {i} ; X _ {i})} \left(1 (\{U _ {i} > T \}) - \frac {\widehat {S} _ {Y} (T ; X _ {i})}{\widehat {S} _ {Y} (H _ {i} - 1 ; X _ {i})}\right),
$$

其中 $H _ { i }$ 如 (16.26) 所定义。证明，在我们的设定下，如果进一步有

$$
\mathbb {E} \left[ \left(1 / \widehat {S} _ {C} (t; X _ {i}) - 1 / S _ {C} (t; X _ {i})\right) ^ {2} \right] = o _ {P} \left(n ^ {- 2 \alpha_ {C}}\right), \tag {16.32}
$$

$$
\mathbb {E} \left[ \left(1 / \widehat {S} _ {Y} (t; X _ {i}) - 1 / S _ {Y} (t; X _ {i})\right) ^ {2} \right] = o _ {P} \left(n ^ {- 2 \alpha_ {Y}}\right)
$$

对于常数 $\alpha _ { C } , \alpha _ { Y } \ge 0$ 且 $\alpha _ { C } + \alpha _ { Y } \ge 1 / 2$ ，那么

$$
\sqrt {n} \left(\hat {\theta} _ {A I P C W} - \theta\right) \Rightarrow \mathcal {N} \left(0, \sigma_ {A I P C W} ^ {2}\right)
$$

$$
\sigma_ {A I P C W} ^ {2} = \operatorname{Var} \left[ S _ {Y} (T; X _ {i}) \right] \tag {16.33}
$$

$$
+ \sum_ {t = 1} ^ {T} \mathbb {E} \left[ \frac {S _ {Y} ^ {2} (T ; X _ {i})}{S _ {C} (t ; X _ {i})} \frac {S _ {Y} (t - 1 ; X _ {i}) - S _ {Y} (t ; X _ {i})}{S _ {Y} (t - 1 ; X _ {i}) S _ {Y} (t ; X _ {i})} \right].
$$

提示：这个结果是定理 14.3 的推论。为了建立这一点，设想一个类似的**动态策略评估（dynamic policy evaluation）**问题，其中没有删失；然而，所有单元在现状处理下开始，但如果他们还活着，则在时间 $C _ { i }$ 转换到实验性处理。论证在这个问题中估计 $\theta$ 等价于在类似的动态策略评估设定中估计 $\mathbb { P } _ { \pi _ { 0 } } \left[ Y _ { i } > T \right]$ ，其中 $\pi _ { 0 }$ 对应于从不开始实验性处理的策略；并且 $\hat { \theta } _ { A I P C W }$ 等价于第 14 章推导出的**双重稳健估计量（doubly robust estimator）** $\widehat { V } _ { A I P W } ( \pi _ { 0 } )$ 。因此，$\hat { \theta } _ { A I P C W }$ 的统计性质可以从定理 14.3 推导出来。

## 参考文献（Bibliography）

Alberto Abadie. **半参数工具变量估计处理响应模型**（Semiparametric instrumental variable estimation of treatment response models）. Journal of Econometrics, 113(2):231–263, 2003.  
Alberto Abadie and Javier Gardeazabal. **冲突的经济成本：巴斯克地区的案例研究**（The economic costs of conflict: A case study of the Basque country）. American Economic Review, 93(1):113–132, 2003.  
Alberto Abadie and Guido W Imbens. **平均处理效应匹配估计量的大样本性质**（Large sample properties of matching estimators for average treatment effects）. Econometrica, 74(1):235–267, 2006.  
Alberto Abadie and Guido W Imbens. **基于估计倾向得分的匹配**（Matching on the estimated propensity score）. Econometrica, 84(2):781–807, 2016.  
Alberto Abadie, Alexis Diamond, and Jens Hainmueller. **比较案例研究的合成控制方法：评估加州烟草控制计划的效果**（Synthetic control methods for comparative case studies: Estimating the effect of california’s tobacco control program）. Journal of the American Statistical Association, 105(490):493–505, 2010.  
Alberto Abadie, Susan Athey, Guido W Imbens, and Jeffrey M Wooldridge. **何时应调整聚类标准误？**（When should you adjust standard errors for clustering?）. The Quarterly Journal of Economics, 138(1):1–35, 2023.  
Anish Agarwal, Devavrat Shah, Dennis Shen, and Dogyoon Song. **关于主成分回归的稳健性**（On robustness of principal component regression）. Journal of the American Statistical Association, 116(536):1731–1745, 2021.  
Shipra Agrawal and Navin Goyal. **汤普森采样的近最优遗憾界**（Near-optimal regret bounds for Thompson sampling）. Journal of the ACM, 64(5):1–24, 2017.  
Luigi Ambrosio and Gianni Dal Maso. **分布导数的一般链式法则**（A general chain rule for distributional derivatives）. Proceedings of the American Mathematical Society, 108(3):691–702, 1990.  
Takeshi Amemiya. **非线性两阶段最小二乘估计量**（The nonlinear two-stage least-squares estimator）. Journal of Econometrics, 2(2):105–110, 1974.  
Joshua D Angrist. **终生收入与越战征兵抽签：来自社会保障行政记录的证据**（Lifetime earnings and the Vietnam era draft lottery: Evidence from social security administrative records）. American Economic Review, 80(3):313–336, 1990.  
Joshua D Angrist and Alan B Krueger. **教育回报的分样本工具变量估计**（Split-sample instrumental variables estimates of the return to schooling）. Journal of Business & Economic Statistics, 13(2):225–235, 1995.  
Joshua D Angrist, Guido W Imbens, and Donald B Rubin. **使用工具变量识别因果效应**（Identification of causal effects using instrumental variables）. Journal of the American Statistical Association, 91(434):444–455, 1996.  
Joshua D Angrist, Kathryn Graddy, and Guido W Imbens. **联立方程模型中工具变量估计量的解释及其在鱼类需求中的应用**（The interpretation of instrumental variables estimators in simultaneous equations models with an application to the demand for fish）. The Review of Economic Studies, 67(3):499–527, 2000.  
Kevin Arceneaux, Alan S Gerber, and Donald P Green. **使用大规模选民动员实验比较实验和匹配方法**（Comparing experimental and matching methods using a large-scale voter mobilization experiment）. Political Analysis, 14(1):37–62, 2006.  
Manuel Arellano. **面板数据计量经济学**（Panel Data Econometrics）. Oxford university press, 2003.  
Dmitry Arkhangelsky and David Hirshberg. **基于不可观测变量选择的合成控制方法的大样本性质**（Large-sample properties of the synthetic control method under selection on unobservables）. arXiv preprint arXiv:2311.13575, 2023.  
Dmitry Arkhangelsky and Guido Imbens. **纵向与面板数据的因果模型：综述**（Causal models for longitudinal and panel data: A survey）. arXiv preprint arXiv:2311.15458, 2023.  
Dmitry Arkhangelsky, Susan Athey, David A Hirshberg, Guido W Imbens, and Stefan Wager. **合成双重差分法**（Synthetic difference-in-differences）. American Economic Review, 111(12):4088–4118, 2021.  
Timothy B Armstrong and Michal Koles´ar. **一类回归模型中的最优推断**（Optimal inference in a class of regression models）. Econometrica, 86(2):655–683, 2018.  
Timothy B Armstrong and Michal Koles´ar. **非参数回归中的简单且诚实的置信区间**（Simple and honest confidence intervals in nonparametric regression）. Quantitative Economics, 11(1):1–39, 2020.  
Peter M Aronow. **检测随机实验中单元间干扰的通用方法**（A general method for detecting interference between units in randomized experiments）. Sociological Methods & Research, 41(1):3–16, 2012.  
Peter M Aronow and Allison Carnegie. **超越 LATE：使用工具变量估计平均处理效应**（Beyond LATE: Estimation of the average treatment effect with an instrumental variable）. Political Analysis, 21(4):492–506, 2013.  
Peter M Aronow and Cyrus Samii. **在一般干扰下估计平均因果效应，并应用于社交网络实验**（Estimating average causal effects under general interference, with application to a social network experiment）. The Annals of Applied Statistics, 11(4):1912–1947, 2017.  
Peter M Aronow, Donald P Green, and Donald KK Lee. **随机实验中方差的尖锐界**（Sharp bounds on the variance in randomized experiments）. The Annals of Statistics, 42(3):850–871, 2014.  
Susan Athey and Guido W Imbens. **异质性因果效应的递归划分**（Recursive partitioning for heterogeneous causal effects）. Proceedings of the National Academy of Sciences, 113(27):7353–7360, 2016.  
Susan Athey and Guido W Imbens. **交错采用设定下基于设计的双重差分分析**（Design-based analysis in difference-in-differences settings with staggered adoption）. Journal of Econometrics, 226(1):62–79, 2022.  
Susan Athey and Stefan Wager. **使用因果森林估计处理效应：一个应用**（Estimating treatment effects with causal forests: An application）. Observational Studies, 5:36–51, 2019.  
Susan Athey and Stefan Wager. **基于观测数据的策略学习**（Policy learning with observational data）. Econometrica, 89(1):133–161, 2021.  
Susan Athey, Dean Eckles, and Guido W Imbens. **网络干扰的精确 p 值**（Exact p-values for network interference）. Journal of the American Statistical Association, 113(521):230–240, 2018a.  
Susan Athey, Guido W Imbens, and Stefan Wager. **近似残差平衡：高维平均处理效应的去偏推断**（Approximate residual balancing: Debiased inference of average treatment effects in high dimensions）. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 80(4):597–623, 2018b.  
Susan Athey, Julie Tibshirani, and Stefan Wager. **广义随机森林**（Generalized random forests）. The Annals of Statistics, 47(2):1148–1178, 2019.  
Susan Athey, Mohsen Bayati, Nikolay Doudchenko, Guido Imbens, and Khashayar Khosravi. **因果面板数据模型的矩阵补全方法**（Matrix completion methods for causal panel data models）. Journal of the American Statistical Association, 116(536):1716–1730, 2021.  
Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. **多臂老虎机问题的有限时间分析**（Finite-time analysis of the multiarmed bandit problem）. Machine Learning, 47(2-3):235–256, 2002.  
Jushan Bai. **具有交互固定效应的面板数据模型**（Panel data models with interactive fixed effects）. Econometrica, 77(4):1229–1279, 2009.  
Pierre Baldi and Yosef Rinott. **关于基于依赖图的正态近似分布**（On normal approximations of distributions in terms of dependency graphs）. The Annals of Probability, 17(4):1646–1650, 1989.  
Heejung Bang and James M Robins. **缺失数据和因果推断模型中的双重稳健估计**（Doubly robust estimation in missing data and causal inference models）. Biometrics, 61(4):962–973, 2005.  
Guillaume W Basse, Avi Feller, and Panos Toulis. **干扰下因果效应的随机化检验**（Randomization tests of causal effects under interference）. Biometrika, 106(2):487–494, 2019.  
Hamsa Bastani and Mohsen Bayati. **高维协变量的在线决策**（Online decision making with high-dimensional covariates）. Operations Research, 68(1):276–294, 2020.  
Eli Ben-Michael, Avi Feller, and Jesse Rothstein. **增广合成控制方法**（The augmented synthetic control method）. Journal of the American Statistical Association, 116(536):1789–1803, 2021.  
Marianne Bertrand, Esther Duflo, and Sendhil Mullainathan. **我们应在多大程度上信任双重差分估计？**（How much should we trust differences-in-differences estimates?）. The Quarterly Journal of Economics, 119(1):249–275, 2004.  
Dimitris Bertsimas and Nathan Kallus. **从预测分析到规范分析**（From predictive to prescriptive analytics）. Management Science, 66(3):1025–1044, 2020.  
Omar Besbes, Yonatan Gur, and Assaf Zeevi. **非平稳奖励多臂老虎机问题中的最优探索-利用**（Optimal exploration–exploitation in a multi-armed bandit problem with non-stationary rewards）. Stochastic Systems, 9(4):319–337, 2019.  
Peter J Bickel, Chris AJ Klaassen, Ya’acov Ritov, and Jon A Wellner. **半参数模型的有效与自适应估计**（Efficient and adaptive estimation for semiparametric models）. Johns Hopkins University Press Baltimore, 1993.  
Christopher Blattman, Donald P Green, Daniel Ortega, and Santiago Tob´on. **大规模基于地点的干预：警务和城市服务对犯罪的直接与溢出效应**（Place-based interventions at scale: The direct and spillover effects of policing and city services on crime）. Journal of the European Economic Association, 19(4):2022–2051, 2021.  
Adam Bloniarz, Hanzhong Liu, Cun-Hui Zhang, Jasjeet S Sekhon, and Bin Yu. **随机实验中处理效应估计的 LASSO 调整**（Lasso adjustments of treatment effect estimates in randomized experiments）. Proceedings of the National Academy of Sciences, 113(27):7383–7390, 2016.  
Gregor Boehl, Gavin Goy, and Felix Strobel. **量化宽松的结构性研究**（A structural investigation of quantitative easing）. Review of Economics and Statistics, 106(4):1028–1044, 2024.  
Iavor Bojinov, David Simchi-Levi, and Jinglong Zhao. **开关实验的设计与分析**（Design and analysis of switchback experiments）. Management Science, 69(7):3759–3777, 2023.  
Kirill Borusyak, Xavier Jaravel, and Jann Spiess. **重新审视事件研究设计：稳健且有效的估计**（Revisiting event study designs: Robust and efficient estimation）. Review of Economic Studies, forthcoming, 2024.  
John Bound, David A Jaeger, and Regina M Baker. **当工具变量与内生解释变量相关性较弱时工具变量估计的问题**（Problems with instrumental variables estimation when the correlation between the instruments and the endogenous explanatory variable is weak）. Journal of the American Statistical Association, 90(430):443–450, 1995.  
Richard C Bradley. **强混合条件的基本性质：综述与未解决问题**（Basic properties of strong mixing conditions: A survey and some open questions）. Probability Surveys, 2:107–144, 2005.  
Leo Breiman. **随机森林**（Random forests）. Machine Learning, 45(1):5–32, 2001.  
S´ebastien Bubeck and Nicolo Cesa-Bianchi. **随机与非随机多臂老虎机问题的遗憾分析**（Regret analysis of stochastic and nonstochastic multi-armed bandit problems）. Foundations and Trends® in Machine Learning, 5(1):1–122, 2012.  
S´ebastien Bubeck, R´emi Munos, and Gilles Stoltz. **多臂老虎机问题中的纯探索**（Pure exploration in multi-armed bandits problems）. In Proceedings of the 20th International Conference Algorithmic Learning Theory, pages 23–37. Springer, 2009.  
Andreas Buja, Lawrence Brown, Richard Berk, Edward George, Emil Pitkin, Mikhail Traskin, Kai Zhang, and Linda Zhao. **作为近似的模型 I：以线性回归说明的后果**（Models as approximations I: Consequences illustrated with linear regression）. Statistical Science, 34(4):523–544, 2019.  
Jing Cai, Alain De Janvry, and Elisabeth Sadoulet. **社交网络与投保决策**（Social networks and the decision to insure）. American Economic Journal: Applied Economics, 7(2):81–108, 2015.  
Brantly Callaway and Pedro HC Sant’Anna. **多时间期的双重差分法**（Difference-in-differences with multiple time periods）. Journal of Econometrics, 225(2):200–230, 2021.  
Sebastian Calonico, Matias D Cattaneo, and Rocio Titiunik. **断点回归设计的稳健非参数置信区间**（Robust nonparametric confidence intervals for regression-discontinuity designs）. Econometrica, 82(6):2295–2326, 2014.  
Sebastian Calonico, Matias D Cattaneo, and Max H Farrell. **非参数推断中偏差估计对覆盖精度的影响**（On the effect of bias estimation on coverage accuracy in nonparametric inference）. Journal of the American Statistical Association, 113(522):767–779, 2018.  
Sebastian Calonico, Matias D Cattaneo, Max H Farrell, and Rocio Titiunik. **使用协变量的断点回归设计**（Regression discontinuity designs using covariates）. Review of Economics and Statistics, 101(3):442–451, 2019.  
David Card and Alan B Krueger. **最低工资与就业：新泽西州和宾夕法尼亚州快餐业的案例研究**（Minimum wages and employment: A case study of the fast-food industry in New Jersey and Pennsylvania）. The American Economic Review, 84(4):772–793, 1994.  
Pedro Carneiro, James J Heckman, and Edward J Vytlacil. **估计教育的边际回报**（Estimating marginal returns to education）. American Economic Review, 101(6):2754–2781, 2011.  
Claes M Cassel, Carl E S¨arndal, and Jan H Wretman. **有限总体广义差分估计与广义回归估计的一些结果**（Some results on generalized difference estimation and generalized regression estimation for finite populations）. Biometrika, 63(3):615–620, 1976.  
Juan Camilo Castillo, Dan Knoepfle, and Glen Weyl. **网约车中的匹配与定价：盲目追逐及其解决方法**（Matching and pricing in ride hailing: Wild goose chases and how to solve them）. Management Science, forthcoming, 2024.  
Gary Chamberlain. **条件矩限制下估计的渐近有效性**（Asymptotic efficiency in estimation with conditional moment restrictions）. Journal of Econometrics, 34(3):305–334, 1987.  
Gary Chamberlain. **半参数回归的效率界**（Efficiency bounds for semiparametric regression）. Econometrica, 60(3):567–596, 1992.  
Olivier Chapelle and Lihong Li. **汤普森采样的实证评估**（An empirical evaluation of Thompson sampling）. Advances in Neural Information Processing Systems, 24, 2011.  
Xiaohong Chen. **半非参数模型的大样本筛估计**（Large sample sieve estimation of semi-nonparametric models）. Handbook of Econometrics, 6:5549–5632, 2007.  
Ming-Yen Cheng, Jianqing Fan, and James S Marron. **关于自动边界校正**（On automatic boundary corrections）. The Annals of Statistics, 25(4):1691–1708, 1997.  
Victor Chernozhukov, Mert Demirer, Esther Duflo, and Iv´an Fern´andez-Val. **随机实验中异质性处理效应的通用机器学习推断**（Generic machine learning inference on heterogenous treatment effects in randomized experiments）. arXiv preprint arXiv:1712.04802, 2017.  
Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. **处理效应与结构参数的双重/去偏机器学习**（Double/debiased machine learning for treatment and structural parameters）. The Econometrics Journal, 21(1):1–68, 2018.  
Victor Chernozhukov, Juan Carlos Escanciano, Hidehiko Ichimura, Whitney K Newey, and James M Robins. **局部稳健的半参数估计**（Locally robust semiparametric estimation）. Econometrica, 90(4):1501–1535, 2022a.  
Victor Chernozhukov, Whitney K Newey, and Rahul Singh. **因果与结构效应的自动去偏机器学习**（Automatic debiased machine learning of causal and structural effects）. Econometrica, 90(3):967–1027, 2022b.  
Albert Chiu, Xingchen Lan, Ziyi Liu, and Yiqing Xu. **在平行趋势假设下对因果面板分析应做（与不应做）之事：来自大规模再分析研究的教训**（What to do (and not to do) with causal panel analysis under parallel trends: Lessons from a large reanalysis study）. arXiv preprint arXiv:2309.15983, 2023.  
Eunyi Chung and Joseph P Romano. **精确且渐近稳健的置换检验**（Exact and asymptotically robust permutation tests）. The Annals of Statistics, 41(2):484–507, 2013.  
Peter L Cohen and Colin B Fogarty. **有限总体因果推断的高斯预枢轴化**（Gaussian prepivoting for finite population causal inference）. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(2):295–320, 2022.  
Bruno Cr´epon, Esther Duflo, Marc Gurgand, Roland Rathelot, and Philippe Zamora. **劳动力市场政策是否具有替代效应？来自整群随机实验的证据**（Do labor market policies have displacement effects? evidence from a clustered randomized experiment）. The Quarterly Journal of Economics, 128(2):531–580, 2013.  
Yifan Cui, Michael R Kosorok, Erik Sverdrup, Stefan Wager, and Ruoqing Zhu. **通过因果生存森林估计右删失数据下的异质性处理效应**（Estimating heterogeneous treatment effects with right-censored data via causal survival forests）. Journal of the Royal Statistical Society Series B: Statistical Methodology, 85(2):179–211, 2023.  
Cl´ement de Chaisemartin and Xavier D’Haultfoeuille. **具有异质性处理效应的双向固定效应估计量**（Two-way fixed effects estimators with heterogeneous treatment effects）. arXiv preprint arXiv:1803.08807, 2018.  
Rajeev H Dehejia and Sadek Wahba. **非实验研究中的因果效应：重新评估培训项目的评估**（Causal effects in nonexperimental studies: Reevaluating the evaluation of training programs）. Journal of the American Statistical Association, 94(448):1053–1062, 1999.  
Alexis Diamond and Jasjeet S Sekhon. **用于估计因果效应的遗传匹配：一种在观测研究中实现平衡的通用多变量匹配方法**（Genetic matching for estimating causal effects: A general multivariate matching method for achieving balance in observational studies）. Review of Economics and Statistics, 95(3):932–945, 2013.  
Peng Ding. **来自基于随机化的因果推断的一个悖论**（A paradox from randomization-based causal inference）. Statistical Science, 32(3):331–345, 2017.  
Peng Ding, Avi Feller, and Luke Miratrix. **分解处理效应变异**（Decomposing treatment effect variation）. Journal of the American Statistical Association, 114(525):304–317, 2019.  
David L Donoho. **统计估计与最优恢复**（Statistical estimation and optimal recovery）. The Annals of Statistics, 22(1):238–270, 1994.  
Rick Durrett. **概率论：理论与例子**（Probability: Theory and Examples）. Cambridge University Press, Cambridge, United Kingdom, 5th edition, 2019.  
Dean Eckles, Nikolaos Ignatiadis, Stefan Wager, and Han Wu. **断点回归设计中的噪声诱导随机化**（Noise-induced randomization in regression discontinuity designs）. arXiv preprint arXiv:2004.09458, 2020.  
Bradley Efron. **刀切法、自助法及其他重抽样计划**（The Jackknife, the Bootstrap, and other Resampling Plans）. Siam, 1982.  
Bradley Efron and David Feldman. **依从性作为临床试验中的解释变量**（Compliance as an explanatory variable in clinical trials）. Journal of the American Statistical Association, 86(413):9–17, 1991.  
Lin Fan and Peter W Glynn. **优化老虎机算法的脆弱性**（The fragility of optimized bandit algorithms）. arXiv preprint arXiv:2109.13595, 2021.  
Max H Farrell. **当协变量可能多于观测值时平均处理效应的稳健推断**（Robust inference on average treatment effects with possibly more covariates than observations）. Journal of Econometrics, 189(1):1–23, 2015.  
Amy Finkelstein, Sarah Taubman, Bill Wright, Mira Bernstein, Jonathan Gruber, Joseph P Newhouse, Heidi Allen, Katherine Baicker, and the Oregon Health Study Group. **俄勒冈健康保险实验：第一年的证据**（The oregon health insurance experiment: evidence from the first year）. The Quarterly Journal of Economics, 127(3):1057–1106, 2012.  
Ronald A Fisher. **实验设计**（The Design of Experiments）. Oliver and Boyd, Edinburgh, 1935.  
Dylan J Foster and Vasilis Syrgkanis. **正交统计学习**（Orthogonal statistical learning）. The Annals of Statistics, 51(3):879–908, 2023.  
Constantine E Frangakis and Donald B Rubin. **因果推断中的主分层**（Principal stratification in causal inference）. Biometrics, 58(1):21–29, 2002.  
David A Freedman. **关于鞅的尾部概率**（On tail probabilities for martingales）. The Annals of Probability, 3(1):100–118, 1975.  
Sebastian Galiani, Paul Gertler, and Ernesto Schargrodsky. **生命之水：供水服务私有化对儿童死亡率的影响**（Water for life: The impact of the privatization of water services on child mortality）. Journal of Political Economy, 113(1):83–120, 2005.  
Dan Geiger, Thomas Verma, and Judea Pearl. **识别贝叶斯网络中的独立性**（Identifying independence in Bayesian networks）. Networks, 20(5):507–534, 1990.  
Andrew Gelman and Guido W Imbens. **为何不应在断点回归设计中使用高阶多项式**（Why high-order polynomials should not be used in regression discontinuity designs）. Journal of Business & Economic Statistics, 37(3):447–456, 2019.  
John C Gittins. **老虎机过程与动态分配指数**（Bandit processes and dynamic allocation indices）. Journal of the Royal Statistical Society: Series B (Methodological), 41(2):148–164, 1979.  
Alexander Goldenshluger and Assaf Zeevi. **线性响应老虎机问题**（A linear response bandit problem）. Stochastic Systems, 3(1):230–261, 2013.  
Bryan S Graham, Cristine Campos de Xavier Pinto, and Daniel Egel. **缺失数据矩条件模型的逆概率倾斜**（Inverse probability tilting for moment condition models with missing data）. The Review of Economic Studies, 79(3):1053–1079, 2012.  
The INSIGHT START Study Group. **早期无症状 HIV 感染中抗逆转录病毒治疗的启动**（Initiation of antiretroviral therapy in early asymptomatic HIV infection）. The New England Journal of Medicine, 373(9):795–807, 2015.  
Yonatan Gur, Ahmadreza Momeni, and Stefan Wager. **平滑自适应上下文老虎机**（Smoothness-adaptive contextual bandits）. Operations Research, 70(6):3198–3216, 2022.  
Trygve Haavelmo. **联立方程组系统的统计含义**（The statistical implications of a system of simultaneous equations）. Econometrica, 11(1):1–12, 1943.  
Vitor Hadad, David A Hirshberg, Ruohan Zhan, Stefan Wager, and Susan Athey. **自适应实验中策略评估的置信区间**（Confidence intervals for policy evaluation in adaptive experiments）. Proceedings of the National Academy of Sciences, 118(15), 2021.  
Jinyong Hahn. **关于倾向得分在有效半参数估计平均处理效应中的作用**（On the role of the propensity score in efficient semiparametric estimation of average treatment effects）. Econometrica, 66(2):315–331, 1998.  
Jinyong Hahn, Petra Todd, and Wilbert van der Klaauw. **使用断点回归设计的处理效应的识别与估计**（Identification and estimation of treatment effects with a regression-discontinuity design）. Econometrica, 69(1):201–209, 2001.  
P Richard Hahn, Jared S Murray, and Carlos M Carvalho. **用于因果推断的贝叶斯回归树模型：正则化、混杂与异质性效应**（Bayesian regression tree models for causal inference: Regularization, confounding, and heterogeneous effects）. Bayesian Analysis, 15(3):965–1056, 2020.  
Jens Hainmueller. **因果效应的熵平衡：一种在观测研究中产生平衡样本的多变量重新加权方法**（Entropy balancing for causal effects: A multivariate reweighting method to produce balanced samples in observational studies）. Political Analysis, 20(1):25–46, 2012.  
Jaroslav H´ajek. **估计中的局部渐近极小极大与可容许性**（Local asymptotic minimax and admissibility in estimation）. In Proceedings of the Sixth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Theory of Statistics, volume 6, pages 175–195. University of California Press, 1972.  
Jonathan V Hall, John J Horton, and Daniel T Knoepfle. **网约车市场的重新均衡**（Ride-sharing markets re-equilibrate）. Technical report, National Bureau of Economic Research, 2023.  
M Elizabeth Halloran and Claudio J Struchiner. **传染病中的因果推断**（Causal inference in infectious diseases）. Epidemiology, 6(2):142–151, 1995.  
Christopher Harshaw, Fredrik S¨avje, and Yitan Wang. **随机实验的基于设计的 Riesz 表示框架**（A design-based riesz representation framework for randomized experiments）. arXiv preprint arXiv:2210.08698, 2022.  
Trevor Hastie, Robert Tibshirani, and Jerome H Friedman. **统计学习基础：数据挖掘、推断与预测**（The Elements of Statistical Learning: Data Mining, Inference, and Prediction）. Springer, 2 edition, 2009.  
James J Heckman. **样本选择偏差作为设定误差**（Sample selection bias as a specification error）. Econometrica, 47(1):153–161, 1979.  
James J Heckman and Edward J Vytlacil. **用于识别和界定处理效应的局部工具变量与潜变量模型**（Local instrumental variables and latent variable models for identifying and bounding treatment effects）. Proceedings of the National Academy of Sciences, 96(8):4730–4734, 1999.  
James J Heckman and Edward J Vytlacil. **结构方程、处理效应与计量经济政策评估**（Structural equations, treatment effects, and econometric policy evaluation）. Econometrica, 73(3):669–738, 2005.  
Inge S Helland. **离散或连续时间鞅的中心极限定理**（Central limit theorems for martingales with discrete or continuous time）. Scandinavian Journal of Statistics, 9(2):79–94, 1982.  
Miguel A Hern´an and James M Robins. **因果推断：如果**（Causal Inference: What If）. Chapman & Hall/CRC, Boca Raton, 2020.  
Keisuke Hirano and Jack R Porter. **统计处理规则的渐近性质**（Asymptotics for statistical treatment rules）. Econometrica, 77(5):1683–1701, 2009.  
Keisuke Hirano and Jack R Porter. **序贯决策、自适应实验与批量老虎机的渐近表示**（Asymptotic representations for sequential decisions, adaptive experiments, and batched bandits）. arXiv preprint arXiv:2302.03117, 2023.  
Keisuke Hirano, Guido W Imbens, and Geert Ridder. **使用估计的倾向得分有效估计平均处理效应**（Efficient estimation of average treatment effects using the estimated propensity score）. Econometrica, 71(4):1161–1189, 2003.  
David A Hirshberg and Stefan Wager. **增广极小极大线性估计**（Augmented minimax linear estimation）. The Annals of Statistics, 49(6):3206–3227, 2021.  
Paul W Holland. **统计学与因果推断**（Statistics and causal inference）. Journal of the American Statistical Association, 81(396):945–960, 1986.  
Steven R Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. **时间一致、非参数、非渐近置信序列**（Time-uniform, nonparametric, nonasymptotic confidence sequences）. The Annals of Statistics, 49(2):1055–1080, 2021.  
Yichun Hu, Nathan Kallus, and Xiaojie Mao. **平滑上下文老虎机：连接参数化与非可微遗憾机制**（Smooth contextual bandits: Bridging the parametric and nondifferentiable regret regimes）. Operations Research, 70(6):3261–3281, 2022a.  
Yuchen Hu and Stefan Wager. **几何混合下的开关实验**（Switchback experiments under geometric mixing）. arXiv preprint arXiv:2209.00197, 2022.  
Yuchen Hu and Stefan Wager. **序贯可忽略性下部分可观察马尔可夫决策过程的离策略评估**（Off-policy evaluation in partially observed Markov decision processes under sequential ignorability）. The Annals of Statistics, 51(4):1561–1585, 2023.  
Yuchen Hu, Shuangning Li, and Stefan Wager. **干扰下的平均直接与间接因果效应**（Average direct and indirect causal effects under interference）. Biometrika, 109(4):1165–1172, 2022b.  
Michael G Hudgens and M Elizabeth Halloran. **带干扰的因果推断**（Toward causal inference with interference）. Journal of the American Statistical Association, 103(482):832–842, 2008.  
Stefano M Iacus, Gary King, and Giuseppe Porro. **无需平衡检验的因果推断：粗化精确匹配**（Causal inference without balance checking: Coarsened exact matching）. Political Analysis, 20(1):1–24, 2012.  
Kosuke Imai and Michael Lingzhi Li. **个体化处理规则的实验评估**（Experimental evaluation of individualized treatment rules）. Journal of the American Statistical Association, 118(541):242–256, 2023.  
Kosuke Imai and Marc Ratkovic. **协变量平衡倾向得分**（Covariate balancing propensity score）. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 76(1):243–263, 2014.

Guido W Imbens. 外生性条件下平均处理效应的非参数估计：综述（Nonparametric estimation of average treatment effects under exogeneity: A review）. *Review of Economics and Statistics*, 86(1):4–29, 2004.

Guido W Imbens. 工具变量：一位计量经济学家的视角（Instrumental variables: An econometrician’s perspective）. *Statistical Science*, 29(3):323–358, 2014.

Guido W Imbens. 因果关系的潜在结果与有向无环图方法：对经济学实证实践的相关性（Potential outcome and directed acyclic graph approaches to causality: Relevance for empirical practice in economics）. *arXiv preprint arXiv:1907.07271*, 2019.

Guido W Imbens and Joshua D Angrist. 局部平均处理效应的识别与估计（Identification and estimation of local average treatment effects）. *Econometrica*, 62(2):467–475, 1994.

Guido W Imbens and Karthik Kalyanaraman. 断点回归估计量的最优带宽选择（Optimal bandwidth choice for the regression discontinuity estimator）. *The Review of Economic Studies*, 79(3):933–959, 2012.

Guido W Imbens and Thomas Lemieux. 断点回归设计：实践指南（Regression discontinuity designs: A guide to practice）. *Journal of Econometrics*, 142(2):615–635, 2008.

Guido W Imbens and Charles F Manski. 部分识别参数的置信区间（Confidence intervals for partially identified parameters）. *Econometrica*, 72(6):1845–1857, 2004.

Guido W Imbens and Donald B Rubin. 统计学、社会科学与生物医学中的因果推断（Causal Inference in Statistics, Social, and Biomedical Sciences）. Cambridge University Press, 2015.

Guido W Imbens and Stefan Wager. 优化的断点回归设计（Optimized regression discontinuity designs）. *Review of Economics and Statistics*, 101(2):264–278, 2019.

Hemant Ishwaran, Udaya B Kogalur, Eugene H Blackstone, and Michael S Lauer. 随机生存森林（Random survival forests）. *The Annals of Applied Statistics*, pages 841–860, 2008.

Adel Javanmard and Andrea Montanari. 高维回归的置信区间与假设检验（Confidence intervals and hypothesis testing for high-dimensional regression）. *The Journal of Machine Learning Research*, 15(1):2869–2909, 2014.

Nan Jiang and Lihong Li. 强化学习的双重稳健离策略价值评估（Doubly robust off-policy value evaluation for reinforcement learning）. In *International Conference on Machine Learning*, 2016.

Nathan Kallus. 因果推断的广义最优匹配方法（Generalized optimal matching methods for causal inference）. *Journal of Machine Learning Research*, 21(62):1–54, 2020.

Nathan Kallus and Masatoshi Uehara. 用于马尔可夫决策过程中高效离策略评估的双重强化学习（Double reinforcement learning for efficient off-policy evaluation in Markov decision processes）. *Journal of Machine Learning Research*, 21(167):1–63, 2020.

Nathan Kallus and Masatoshi Uehara. 通过双重强化学习高效打破离策略评估中的视界诅咒（Efficiently breaking the curse of horizon in off-policy evaluation with double reinforcement learning）. *Operations Research*, 70(6):3282–3302, 2022.

Nathan Kallus and Angela Zhou. 未观测混杂下极小极大最优策略学习（Minimax-optimal policy learning under unobserved confounding）. *Management Science*, 67(5):2870–2890, 2021.

Edward L Kaplan and Paul Meier. 不完全观测的非参数估计（Nonparametric estimation from incomplete observations）. *Journal of the American Statistical Association*, 53(282):457–481, 1958.

Maximilian Kasy and Anja Sautmann. 政策选择实验中的自适应处理分配（Adaptive treatment assignment in experiments for policy choice）. *Econometrica*, 89(1):113–132, 2021.

Edward H Kennedy. 迈向异质性因果效应的最优双重稳健估计（Towards optimal doubly robust estimation of heterogeneous causal effects）. *Electronic Journal of Statistics*, 17(2):3008–3049, 2023.

Edward H Kennedy, Scott Lorch, and Dylan S Small. 使用局部工具变量曲线的连续工具变量稳健因果推断（Robust causal inference with continuous instruments using the local instrumental variable curve）. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 81(1):121–143, 2019.

Edward H Kennedy, Sivaraman Balakrishnan, James M Robins, and Larry Wasserman. 异质性因果效应估计的极小极大速率（Minimax rates for heterogeneous causal effect estimation）. *The Annals of Statistics*, 52(2):793–816, 2024.

Toru Kitagawa and Aleksey Tetenov. 谁应被处理？处理选择的经验福利最大化方法（Who should be treated? empirical welfare maximization methods for treatment choice）. *Econometrica*, 86(2):591–616, 2018.

Denis Kojevnikov, Vadim Marmer, and Kyungchul Song. 网络相依随机变量的极限定理（Limit theorems for network dependent random variables）. *Journal of Econometrics*, 222(2):882–908, 2021.

Michal Kolesár and Christoph Rothe. 离散运行变量下的断点回归设计推断（Inference in regression discontinuity designs with a discrete running variable）. *American Economic Review*, 108(8):2277–2304, 2018.

X. Kuang and Stefan Wager. 序贯随机化实验的弱信号渐近理论（Weak signal asymptotics for sequentially randomized experiments）. *Management Science*, forthcoming, 2024.

Sören R Künzel, Jasjeet S Sekhon, Peter J Bickel, and Bin Yu. 使用机器学习估计异质性处理效应的元学习器（Metalearners for estimating heterogeneous treatment effects using machine learning）. *Proceedings of the National Academy of Sciences*, 116(10):4156–4165, 2019.

Tze Leung Lai and Herbert Robbins. 渐近有效的自适应分配规则（Asymptotically efficient adaptive allocation rules）. *Advances in Applied Mathematics*, 6(1):4–22, 1985.

Robert J LaLonde. 用实验数据评估培训项目的计量经济学评估（Evaluating the econometric evaluations of training programs with experimental data）. *American Economic Review*, pages 604–620, 1986.

David S Lee. 美国众议院选举中非随机选择下的随机实验（Randomized experiments from non-random selection in US House elections）. *Journal of Econometrics*, 142(2):675–697, 2008.

Lihua Lei and Emmanuel J Candès. 反事实与个体处理效应的共形推断（Conformal inference of counterfactuals and individual treatment effects）. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 83(5):911–938, 2021.

Lihua Lei and Peng Ding. 协变量数量发散时完全随机化实验的回归调整（Regression adjustment in completely randomized experiments with a diverging number of covariates）. *Biometrika*, 108(4):815–828, 2021.

Lihua Lei and Brad Ross. 使用短面板数据估计反事实矩阵均值（Estimating counterfactual matrix means with short panel data）. *arXiv preprint arXiv:2312.07520*, 2023.

Michael P Leung. 近似邻域干扰下的因果推断（Causal inference under approximate neighborhood interference）. *Econometrica*, 90(1):267–293, 2022.

Shuangning Li and Stefan Wager. 网络干扰下处理效应估计的随机图渐近理论（Random graph asymptotics for treatment effect estimation under network interference）. *The Annals of Statistics*, 50(4):2334–2358, 2022.

Xinran Li and Peng Ding. 有限总体中心极限定理的一般形式及其在因果推断中的应用（General forms of finite population central limit theorems with applications to causal inference）. *Journal of the American Statistical Association*, 112(520):1759–1769, 2017.

Peng Liao, Predrag Klasnja, and Susan Murphy. 长期平均结果的离策略估计及其在移动健康中的应用（Off-policy estimation of longterm average outcomes with applications to mobile health）. *Journal of the American Statistical Association*, 116(533):382–391, 2021.

Peng Liao, Zhengling Qi, Runzhe Wan, Predrag Klasnja, and Susan A Murphy. 平均奖励马尔可夫决策过程中的批策略学习（Batch policy learning in average reward Markov decision processes）. *The Annals of Statistics*, 50(6):3364–3387, 2022.

Winston Lin. 关于实验数据回归调整的不可知论注释：重新审视 Freedman 的批评（Agnostic notes on regression adjustments to experimental data: Reexamining Freedman’s critique）. *The Annals of Applied Statistics*, 7(1):295–318, 2013.

Yueyang Liu, Benjamin Van Roy, and Kuang Xu. 通过预测采样进行非平稳赌博机学习（Nonstationary bandit learning via predictive sampling）. In *Proceedings of the International Conference on Artificial Intelligence and Statistics*, pages 6215–6244. PMLR, 2023.

Alex Luedtke and Antoine Chambaz. 策略学习的性能保证（Performance guarantees for policy learning）. *Annales de l’Institut Henri Poincaré, Probabilités et Statistiques*, 56(3):2162–2188, 2020.

Alexander R Luedtke and Mark J van der Laan. 可能非唯一最优处理策略下平均结果的统计推断（Statistical inference for the mean outcome under a possibly non-unique optimal treatment strategy）. *The Annals of Statistics*, 44(2):713, 2016.

Charles F Manski. 异质性总体的统计处理规则（Statistical treatment rules for heterogeneous populations）. *Econometrica*, 72(4):1221–1246, 2004.

Charles F Manski. 具有社会互动的处理反应识别（Identification of treatment response with social interactions）. *The Econometrics Journal*, 16(1):S1–S23, 2013.

Ruth Marcus, Eric Peritz, and K R Gabriel. 关于封闭检验程序——特别参考有序方差分析（On closed testing procedures with special reference to ordered analysis of variance）. *Biometrika*, 63(3):655–660, 1976.

Eric Mbakop and Max Tabord-Meehan. 处理选择的模型选择：惩罚福利最大化（Model selection for treatment choice: Penalized welfare maximization）. *Econometrica*, 89(2):825–848, 2021.

Alec McClean, Sivaraman Balakrishnan, Edward H Kennedy, and Larry Wasserman. 双重交叉拟合双重稳健估计量：超越级数回归（Double cross-fit doubly robust estimators: Beyond series regression）. *arXiv preprint arXiv:2403.15175*, 2024.

Mohammad Mehrabi and Stefan Wager. 弱分布重叠下马尔可夫决策过程的离策略评估（Off-policy evaluation in markov decision processes under weak distributional overlap）. *arXiv preprint arXiv:2402.08201*, 2024.

Nicolai Meinshausen, Alain Hauser, Joris M Mooij, Jonas Peters, Philip Versteeg, and Peter Bühlmann. 基因扰动实验的因果推断方法与验证（Methods for causal inference from gene perturbation experiments and validation）. *Proceedings of the National Academy of Sciences*, 113(27):7361–7368, 2016.

Luke W Miratrix, Jasjeet S Sekhon, and Bin Yu. 随机化实验中通过事后分层调整处理效应估计（Adjusting treatment effect estimates by post-stratification in randomized experiments）. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 75(2):369–396, 2013.

Kari Lock Morgan and Donald B Rubin. 实验中改善协变量平衡的重新随机化（Rerandomization to improve covariate balance in experiments）. *Annals of Statistics*, 40(2):1263–1282, 2012.

Evan Munro, X. Kuang, and Stefan Wager. 市场均衡中的处理效应（Treatment effects in market equilibrium）. *arXiv preprint arXiv:2109.11647*, 2021.

Susan A Murphy. Q-学习的泛化误差（A generalization error for Q-learning）. *Journal of Machine Learning Research*, 6(Jul):1073–1097, 2005.

Sahand N Negahban, Pradeep Ravikumar, Martin J Wainwright, and Bin Yu. 可分解正则化器下 M-估计量高维分析的统一框架（A unified framework for high-dimensional analysis of M-estimators with decomposable regularizers）. *Statistical Science*, 27(4):538–557, 2012.

Whitney K Newey. 非线性模型的有效工具变量估计（Efficient instrumental variables estimation of nonlinear models）. *Econometrica*, 58(4):809–837, 1990.

Whitney K Newey. 半参数估计量的渐近方差（The asymptotic variance of semiparametric estimators）. *Econometrica*, 62(6):1349–1382, 1994.

Whitney K Newey and James L Powell. 非参数模型的工具变量估计（Instrumental variable estimation of nonparametric models）. *Econometrica*, 71(5):1565–1578, 2003.

Whitney K Newey and James R Robins. 半参数估计的交叉拟合与快速余项速率（Cross-fitting and fast remainder rates for semiparametric estimation）. *arXiv preprint arXiv:1801.09138*, 2018.

Jersey Neyman. 概率论在农业实验中的应用：原则的检验（Sur les applications de la théorie des probabilités aux experiences agricoles: Essai des principes）. *Roczniki Nauk Rolniczych*, 10:1–51, 1923.

Xinkun Nie and Stefan Wager. 异质性处理效应的准神谕估计（Quasi-oracle estimation of heterogeneous treatment effects）. *Biometrika*, 108(2):299–319, 2021.

Xinkun Nie, Xiaoying Tian, Jonathan Taylor, and James Zou. 为何自适应收集的数据存在负偏倚以及如何校正（Why adaptively collected data have negative bias and how to correct for it）. In *International Conference on Artificial Intelligence and Statistics*, pages 1261–1269. PMLR, 2018.

Claudia Noack and Christoph Rothe. 模糊断点回归设计中的偏倚感知推断（Bias-aware inference in fuzzy regression discontinuity designs）. *Econometrica*, forthcoming, 2024.

Elizabeth L Ogburn and Tyler J VanderWeele. 疫苗、传染与社会网络（Vaccines, contagion, and social networks）. *Annals of Applied Statistics*, 11(2):919–948, 2017.

Elizabeth L Ogburn, Oleg Sofrygin, Ivan Diaz, and Mark J Van der Laan. 社会网络数据的因果推断（Causal inference for social network data）. *Journal of the American Statistical Association*, 119(545):597–611, 2024.

Judea Pearl. 实证研究的因果图（Causal diagrams for empirical research）. *Biometrika*, 82(4):669–688, 1995.

Judea Pearl. 因果论（Causality）. Cambridge University Press, 2009.

Judea Pearl and Dana Mackenzie. 为什么：因果新科学（The Book of Why: The New Science of Cause and Effect）. Basic Books, 2018.

Vianney Perchet and Philippe Rigollet. 带协变量的多臂赌博机问题（The multi-armed bandit problem with covariates）. *The Annals of Statistics*, 41(2):693–721, 2013.

Jonas Peters, Peter Bühlmann, and Nicolai Meinshausen. 通过不变预测进行因果推断：识别与置信区间（Causal inference by using invariant prediction: identification and confidence intervals）. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 78(5):947–1012, 2016.

Chao Qin and Daniel Russo. 存在外生非平稳变化时的自适应实验（Adaptive experimentation in the presence of exogenous nonstationary variation）. *arXiv preprint arXiv:2202.09036*, 2022.

Thomas S Richardson and Andrea Rotnitzky. James M. Robins 研究的因果病因学（Causal etiology of the research of James M. Robins）. *Statistical Science*, 29(4):459–484, 2014.

Herbert Robbins. 与迭代对数律相关的统计方法（Statistical methods related to the law of the iterated logarithm）. *The Annals of Mathematical Statistics*, 41(5):1397–1409, 1970.

James Robins, Mariela Sued, Quanhong Lei-Gomez, and Andrea Rotnitzky. 评论：当“逆概率”权重高度可变时双重稳健估计量的表现（Comment: Performance of double-robust estimators when “inverse probability” weights are highly variable）. *Statistical Science*, 22(4):544–559, 2007.

James M Robins. 持续暴露期死亡率研究中的因果推断新方法：应用于控制健康工人幸存者效应（A new approach to causal inference in mortality studies with a sustained exposure period: Application to control of the healthy worker survivor effect）. *Mathematical Modelling*, 7(9-12):1393–1512, 1986.

James M Robins. 使用结构嵌套均值模型校正随机试验中的不依从性（Correcting for non-compliance in randomized trials using structural nested mean models）. *Communications in Statistics: Theory and Methods*, 23(8):2379–2412, 1994.

James M Robins. 关联、因果与边际结构模型（Association, causation, and marginal structural models）. *Synthese*, 121(1/2):151–179, 1999.

James M Robins. 最优序贯决策的最优结构嵌套模型（Optimal structural nested models for optimal sequential decisions）. In *Proceedings of the second seattle Symposium in Biostatistics*, pages 189–326. Springer, 2004.

James M Robins and Thomas S Richardson. 替代性图形因果模型与直接效应的识别（Alternative graphical causal models and the identification of direct effects）. *Causality and Psychopathology: Finding the Determinants of Disorders and their Cures*, pages 103–158, 2010.

James M Robins and Andrea Rotnitzky. 缺失数据多元回归模型的半参数效率（Semiparametric efficiency in multivariate regression models with missing data）. *Journal of the American Statistical Association*, 90(429):122–129, 1995.

James M Robins, Andrea Rotnitzky, and Lue Ping Zhao. 当某些回归变量并非总能观测到时的回归系数估计（Estimation of regression coefficients when some regressors are not always observed）. *Journal of the American Statistical Association*, 89(427):846–866, 1994.

James M Robins, Lingling Li, Rajarshi Mukherjee, Eric Tchetgen Tchetgen, and Aad van der Vaart. 结构化高维模型上泛函的极小极大估计（Minimax estimation of a functional on a structured high-dimensional model）. *The Annals of Statistics*, 45(5):1951–1987, 2017.

Peter M Robinson. 根号 n 一致的半参数回归（Root-n-consistent semiparametric regression）. *Econometrica*, 56(4):931–954, 1988.

Todd Rogers and Avi Feller. 通过针对家长的错误信念大规模减少学生缺勤（Reducing student absences at scale by targeting parents’ misbeliefs）. *Nature Human Behaviour*, 2(5):335–342, 2018.

Joseph P Romano. 关于无群组不变性假设下随机化检验的行为（On the behavior of randomization tests without a group invariance assumption）. *Journal of the American Statistical Association*, 85(411):686–692, 1990.

Paul R Rosenbaum and Donald B Rubin. 倾向得分在观察性研究因果效应中的核心作用（The central role of the propensity score in observational studies for causal effects）. *Biometrika*, 70(1):41–55, 1983.

Paul R Rosenbaum and Donald B Rubin. 使用倾向得分子分类减少观察性研究中的偏倚（Reducing bias in observational studies using subclassification on the propensity score）. *Journal of the American Statistical Association*, 79(387):516–524, 1984.

Eric L Ross, Robert M Bossarte, Steven K Dobscha, Sarah M Gildea, Irving Hwang, Chris J Kennedy, Howard Liu, Alex Luedtke, Brian P Marx, Matthew K Nock, et al. 对有自杀行为患者进行精神病住院治疗的估计平均处理效应：一项精准治疗分析（Estimated average treatment effect of psychiatric hospitalization in patients with suicidal behaviors: a precision treatment analysis）. *JAMA psychiatry*, 81(2):135–143, 2024.

Andrew D Roy. 关于收入分配的一些思考（Some thoughts on the distribution of earnings）. *Oxford Economic Papers*, 3(2):135–146, 1951.

Daniel Rubin and Mark J van der Laan. 一种双重稳健的删失无偏变换（A doubly robust censoring unbiased transformation）. *The International Journal of Biostatistics*, 3(1), 2007.

Donald B Rubin. 估计随机与非随机研究中处理的因果效应（Estimating causal effects of treatments in randomized and nonrandomized studies）. *Journal of Educational Psychology*, 66(5):688, 1974.

Daniel Russo. 最佳臂识别的简单贝叶斯算法（Simple Bayesian algorithms for best-arm identification）. *Operations Research*, 68(6):1625–1647, 2020.

Daniel Russo and Benjamin Van Roy. 通过信息导向采样学习优化（Learning to optimize via informationdirected sampling）. *Operations Research*, 66(1):230–252, 2018.

Daniel J Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, and Zheng Wen. Thompson 采样教程（A tutorial on Thompson sampling）. *Foundations and Trends in Machine Learning*, 11(1):1–96, 2018.

Jerome Sacks and Donald Ylvisaker. 近似线性模型的线性估计（Linear estimation for approximately linear models）. *The Annals of Statistics*, 6(5):1122–1137, 1978.

Fredrik Sävje. 错误指定暴露映射下的因果推断：分离定义与假设（Causal inference with misspecified exposure mappings: Separating definitions and assumptions）. *Biometrika*, 111(1):1–15, 2024.

Fredrik Sävje, Peter Aronow, and Michael Hudgens. 存在未知干扰时的平均处理效应（Average treatment effects in the presence of unknown interference）. *The Annals of Statistics*, 49(2):673, 2021.

Daniel O Scharfstein, Andrea Rotnitzky, and James M Robins. 使用半参数无响应模型调整不可忽略的缺失（Adjusting for nonignorable drop-out using semiparametric nonresponse models）. *Journal of the American Statistical Association*, 94(448):1096–1120, 1999.

Eric M Schwartz, Eric T Bradlow, and Peter S Fader. 使用多臂赌博机实验通过展示广告获取客户（Customer acquisition via display advertising using multi-armed bandit experiments）. *Marketing Science*, 36(4):500–522, 2017.

Dennis Shen, Peng Ding, Jasjeet Sekhon, and Bin Yu. 同根异叶：面板数据中的时间序列与横截面方法（Same root different leaves: Time series and cross-sectional methods in panel data）. *Econometrica*, 91(6):2125–2154, 2023.

Michael E Sobel. 住房流动性的随机化研究证明了什么？面对干扰的因果推断（What do randomized studies of housing mobility demonstrate? causal inference in the face of interference）. *Journal of the American Statistical Association*, 101(476):1398–1407, 2006.

Peter Spirtes, Clark N Glymour, and Richard Scheines. 因果、预测与搜索（Causation, Prediction, and Search）. Springer-Verlag, New York, 1993.

Charles M Stein. 多元正态分布均值的估计（Estimation of the mean of a multivariate normal distribution）. *The Annals of Statistics*, 9(6):1135–1151, 1981.

Charles J Stone. 一致非参数回归（Consistent nonparametric regression）. *The Annals of Statistics*, 5(4):595–620, 1977.

Jörg Stoye. 有限样本下的最小化最大遗憾处理选择（Minimax regret treatment choice with finite samples）. *Journal of Econometrics*, 151(1):70–81, 2009.

Hao Sun, Evan Munro, Georgy Kalashnov, Shuyang Du, and Stefan Wager. 不确定成本下的处理分配（Treatment allocation under uncertain costs）. *arXiv preprint arXiv:2103.11066*, 2021.

Liyang Sun and Sarah Abraham. 异质性处理效应下事件研究中动态处理效应的估计（Estimating dynamic treatment effects in event studies with heterogeneous treatment effects）. *Journal of Econometrics*, 225(2):175–199, 2021.

Richard S Sutton. 通过时序差分法学习预测（Learning to predict by the methods of temporal differences）. *Machine Learning*, 3:9–44, 1988.

Richard S Sutton and Andrew G Barto. 强化学习：导论（Reinforcement Learning: An Introduction）. MIT Press, Cambridge, MA, 2nd edition, 2018.

Erik Sverdrup, Han Wu, Susan Athey, and Stefan Wager. 多臂处理规则的 Qini 曲线（Qini curves for multi-armed treatment rules）. *arXiv preprint arXiv:2306.11979*, 2023.

Adith Swaminathan and Thorsten Joachims. 通过反事实风险最小化从记录的赌博机反馈中进行批学习（Batch learning from logged bandit feedback through counterfactual risk minimization）. *The Journal of Machine Learning Research*, 16(1):1731–1755, 2015.

Zhiqiang Tan. 使用高维数据正则化校准估计的处理效应模型辅助推断（Model-assisted inference for treatment effects using regularized calibrated estimation with high-dimensional data）. *The Annals of Statistics*, 48(2):811–837, 2020.

Donald L Thistlethwaite and Donald T Campbell. 断点回归分析：事后实验的替代方案（Regression-discontinuity analysis: An alternative to the ex post facto experiment）. *Journal of Educational Psychology*, 51(6):309–317, 1960.

Philip Thomas and Emma Brunskill. 强化学习的数据高效离策略策略评估（Data-efficient off-policy policy evaluation for reinforcement learning）. In *International Conference on Machine Learning*, pages 2139–2148, 2016.

William R Thompson. 关于一个未知概率超过另一个的可能性——基于两个样本证据（On the likelihood that one unknown probability exceeds another in view of the evidence of two samples）. *Biometrika*, 25(3/4):285–294, 1933.

Lu Tian, Ash A Alizadeh, Andrew J Gentles, and Robert Tibshirani. 一种估计处理与大量协变量之间交互作用的简单方法（A simple method for estimating interactions between a treatment and a large number of covariates）. *Journal of the American Statistical Association*, 109(508):1517–1532, 2014.

Robert Tibshirani. 通过 Lasso 进行回归收缩与选择（Regression shrinkage and selection via the lasso）. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 58(1):267–288, 1996.

Anastasios A Tsiatis. 半参数理论与缺失数据（Semiparametric theory and missing data）. Springer, New York, 2006.

John N Tsitsiklis and Benjamin Van Roy. 使用函数逼近的时序差分学习分析（An analysis of temporal-difference learning with function approximation）. *IEEE Transactions on Automatic Control*, 42(5):674–690, 1997.

Masatoshi Uehara, Jiawei Huang, and Nan Jiang. 离策略评估的极小极大权重与 Q 函数学习（Minimax weight and qfunction learning for off-policy evaluation）. In Hal Daumé III and Aarti Singh, editors, *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of Proceedings of Machine Learning Research, pages 9659–9668. PMLR, 2020.

Mark J van der Laan and James M Robins. 删失纵向数据与因果性的统一方法（Unified methods for censored longitudinal data and causality）. Springer, New York, 2003.

Mark J van der Laan and Sherri Rose. 目标学习：观察性与实验性数据的因果推断（Targeted learning: Causal inference for observational and experimental data）. Springer Science & Business Media, 2011.

Mark J van der Laan and Daniel Rubin. 目标最大似然学习（Targeted maximum likelihood learning）. *The International Journal of Biostatistics*, 2(1), 2006.

Aad W Van der Vaart. 渐近统计学（Asymptotic Statistics）. Cambridge University Press, 1998.

Davide Viviano. 网络干扰下的策略定位（Policy targeting under network interference）. *Review of Economic Studies*, forthcoming, 2024.

Stefan Wager. 关于策略学习的回归表格：对 Jiang, Song, Li 和 Zeng 论文的评论（On regression tables for policy learning: Comment on a paper by Jiang, Song, Li and Zeng）. *Statistica Sinica*, 29(4):1678–1685, 2019.

Stefan Wager, Wenfei Du, Jonathan Taylor, and Robert J Tibshirani. 随机化实验中的高维回归调整（Highdimensional regression adjustments in randomized experiments）. *Proceedings of the National Academy of Sciences*, 113(45):12673–12678, 2016.

Christopher JCH Watkins and Peter Dayan. Q-学习（Q-learning）. *Machine learning*, 8:279–292, 1992.

Halbert White. 异方差一致协方差矩阵估计量与异方差的直接检验（A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity）. *Econometrica*, 48(4):817–838, 1980.

Halbert White. 计量经济学家的渐近理论（Asymptotic Theory for Econometricians）. Economic Theory, Econometrics, and Mathematical Economics. Academic Press, Orlando, Florida, 1984.

Jeffrey M Wooldridge. 横截面与面板数据的计量经济学分析（Econometric Analysis of Cross Section and Panel Data）. MIT press, 2010.

Sewall Wright. 路径系数法（The method of path coefficients）. *The Annals of Mathematical Statistics*, 5(3):161–215, 1934.

Han Wu and Stefan Wager. 无限制延迟下的 Thompson 采样（Thompson sampling with unrestricted delays）. In *Proceedings of the 23rd ACM Conference on Economics and Computation*, pages 937–955, 2022.

Yiqing Xu. 广义合成控制法：具有交互固定效应模型的因果推断（Generalized synthetic control method: Causal inference with interactive fixed effects models）. *Political Analysis*, 25(1):57–76, 2017.

Steve Yadlowsky, Scott Fleming, Nigam Shah, Emma Brunskill, and Stefan Wager. 通过秩加权平均处理效应评估处理优先级规则（Evaluating treatment prioritization rules via rank-weighted average treatment effects）. *arXiv preprint arXiv:2111.07966*, 2021.

Baqun Zhang, Anastasios A Tsiatis, Eric B Laber, and Marie Davidian. 序贯治疗决策中最优动态治疗方案的稳健估计（Robust estimation of optimal dynamic treatment regimes for sequential treatment decisions）. *Biometrika*, 100(3):681–694, 2013.

Cun-Hui Zhang and Stephanie S Zhang. 高维线性模型中低维参数的置信区间（Confidence intervals for low dimensional parameters in high dimensional linear models）. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 76(1):217–242, 2014.

Kelly Zhang, Lucas Janson, and Susan Murphy. 分批赌博机的推断（Inference for batched bandits）. *Advances in Neural Information Processing Systems*, 33:9818–9829, 2020.

Qingyuan Zhao. 通过定制损失函数的协变量平衡倾向得分（Covariate balancing propensity score by tailored loss functions）. *The Annals of Statistics*, 47(2):965–993, 2019.

Qingyuan Zhao, Dylan S Small, and Ashkan Ertefaie. 通过 Lasso 进行效应修饰的选择性推断（Selective inference for effect modification via the lasso）. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 84(2):382–413, 2022.

Yingqi Zhao, Donglin Zeng, A John Rush, and Michael R Kosorok. 使用结果加权学习估计个体化处理规则（Estimating individualized treatment rules using outcome weighted learning）. *Journal of the American Statistical Association*, 107(499):1106–1118, 2012.

Zhengyuan Zhou, Susan Athey, and Stefan Wager. 离线多动作策略学习：泛化与优化（Offline multi-action policy learning: Generalization and optimization）。

José R Zubizarreta. 使用混合整数规划进行肾脏衰竭术后观察性研究中的匹配（Using mixed integer programming for matching in an observational study of kidney failure after surgery）。*Journal of the American Statistical Association*, 107(500):1360–1371, 2012。  
José R Zubizarreta. 平衡协变量的稳定权重用于不完整结局数据的估计（Stable weights that balance covariates for estimation with incomplete outcome data）。*Journal of the American Statistical Association*, 110(511):910–922, 2015。

<!-- footnote -->

- 使用 $\gamma$ 折扣奖励（$\gamma$-discounted rewards）而非长期平均奖励（long-run average rewards）会得到相似但不同的**贝尔曼方程（Bellman equations）**。

<!-- footnote end -->

<!-- footnote -->

- 遵循第3章的术语，我们在此关注的是**弱双重稳健性（weak double robustness）**。

<!-- footnote end -->

<!-- footnote -->

- 你也不需要详细阐述如何构造估计量 $\hat { \alpha } ( \cdot ) , \hat { \tau } ( \cdot )$ 等。

<!-- footnote end -->

<!-- footnote -->

- 我们在此不研究如何估计这些量；然而，我们注意到，估计无条件生存函数的一种常用方法是通过**Kaplan-Meier估计量（Kaplan–Meier estimator）**[Kaplan and Meier, 1958]；并且该方法可以通过例如**随机生存森林（random survival forest）**构造 [Ishwaran et al., 2008] 来条件化于协变量 $X _ { i }$。
- 还存在一个类似的连续时间AIPCW估计量；参见，例如，Rubin and van der Laan [2007] 以及 Cui et al. [2023]。要理解 $\hat { \theta } _ { A I P C W }$ 中的表达式与标准连续时间公式之间的联系，首先对 (16.31) 中的求和应用**阿贝尔变换（Abel transformation）**会有所帮助。

<!-- footnote end -->
# 第15章 马尔可夫决策过程（Markov Decision Processes）

在上一章中，我们考虑了在不对治疗效应随时间如何发挥作用进行建模假设的一般设置下的动态治疗规则，并介绍了一套仅需**顺序无混杂性（sequential unconfoundedness）**即可进行策略评估的方法。然而，这些方法的灵活性是以精度为代价的。所讨论的逆倾向得分加权方法只能利用在全部 $T$ 个时间段内分配的治疗与策略规定相匹配的轨迹，并且涉及权重的大小通常随时间范围 $T$ 呈指数级增长；而增强方法也面临类似的"**时间范围诅咒（curse of horizon）**"。

在这里，我们将研究如何明智地利用建模假设来帮助缓解这种时间范围诅咒。关键洞察在于，在许多应用中，我们所采取的任何干预措施都会在一段时间内产生影响，但其效应最终会消退。并且，如果我们相信很久以前采取的行动不再相关，那么即使轨迹在遥远的过去某个时刻偏离了目标策略，我们仍然有可能有意义地利用这些轨迹进行策略评估。以下示例具有这种结构。

**示例 20.** 许多网约车平台实施了某种形式的**高峰定价机制（surge pricing mechanism）**，这涉及在经历局部需求激增的地区临时提高价格 [Castillo, Knoepfle, and Weyl, 2024]。在给定地点激活高峰定价使平台能够迅速减少该地点的需求，并通过鼓励空闲司机转移到实施高峰定价的区域来增加供给。这有助于市场自我平衡，并避免平台无法按公布价格满足乘车请求的情况。为了在算法之间进行选择和/或校准给定算法的参数，平台经常运行在给定市场的高峰算法之间切换的实验。¹

我们应如何分析上述示例中实验的数据？这个问题显然涉及复杂的治疗动态，因此事件研究法不适用。另一方面，虽然高峰定价算法显然具有复杂的短期效应（例如，通过改变系统中司机的分布），但我们应该预期任何此类效应最终都会消退（在暂时被抑制的需求能够重新出现且司机有机会恢复其通常配置之后）。这表明我们应该能够开发出分析技术，从长期（例如，多周）的高峰定价实验中提取有意义的见解，而不会遭受上一章方法所带来的时间范围诅咒现象。

那么，问题在于如何指定一个灵活且可信的模型来实现这种遗忘。在这里，我们将通过假设**马尔可夫结构（Markovian structure）**来实现这一点。我们假设在一条长轨迹 $t = 1, 2, ..., T$ 上观察单个单元，其中包含状态变量 $X_t$、行动 $W_t$ 和结果 $Y_t$。我们的马尔可夫假设（下文将正式定义）是，在时间 $t$，过去行动对未来可观测变量的任何影响都通过当前状态 $X_t$ 进行中介。只要状态变量 $X_t$ 具有相关的"混合"性质，使其无法长时间保留过去治疗分配的信息，这种马尔可夫结构就会引发遗忘——并使得从单条轨迹进行一致的策略评估成为可能。

**定义 15.1. 马尔可夫决策过程（Markov Decision Process, MDP）** 由一系列状态转移分布 $P_t$ 刻画，使得对于所有 $t$，

$$
X_{t+1}, Y_t \sim P_t(X_t, W_t) \tag{15.1}
$$

条件于直到时间 $t$ 的所有可用信息，即条件于 $X_1, W_1, Y_1, X_2, ..., X_t, W_t$。

在网约车示例的背景下，我们可以将 $X_t$ 定义为每个社区当前的司机数量，将 $W_t$ 定义为实验性高峰算法当前是否在市中心激活。那么，我们的马尔可夫假设将要求假设任何过去高峰定价决策的影响都通过当前司机分布进行中介，而混合假设将实质上意味着，如果我们回到默认算法足够长的时间，司机将恢复到其通常模式。

## 15.1 长期平均价值（The long-run average value）

我们通过重新审视顺序随机化下的策略评估设置来开始对 MDP 的研究，并观察马尔可夫建模假设如何能够相对于上一章的方法实现精度提升。我们在长期 $T \to \infty$ 的框架下工作，旨在估计在时间齐次目标策略下产生的长期平均价值：

$$
V(\pi) = \lim_{T \to \infty} \mathbb{E}_\pi \left[ \frac{1}{T} \sum_{t=1}^{T} Y_t \right], \quad \pi: \mathcal{X} \to \{0, 1\}, \tag{15.2}
$$

假设该极限存在。我们假设数据是在顺序无混杂设计下收集的：

$$
W_t \sim e(X_t), \quad e: \mathcal{X} \to (0, 1), \tag{15.3}
$$

条件于所有过去信息，并且我们假设 $e(x)$ 是已知的。我们还对整个 MDP 做出以下正则性假设：

* MDP 是**时间齐次的（time homogeneous）**，即定义 15.1 中的状态转移分布 $P_t$ 对于所有 $t$ 满足 $P_t = P$。
* 在我们的研究中观察到的状态变量 $X_t$（即按照 (15.3) 生成治疗）形成一个**不可约、非周期的马尔可夫链（irreducible, aperiodic Markov chain）**，具有**平稳分布（stationary distribution）** $F$。该过程从该平稳分布初始化，即 $X_1 \sim F$。
* 在我们的研究中观察到的 $X_t$ 满足 **$\rho$-混合条件（$\rho$-mixing condition）** [参见 Bradley, 2005 关于混合条件及其关系的综述]：

$$
\sum_{t=1}^{\infty} \sup_{f, g \in L_2(F)} |\operatorname{Corr}(f(X_1), g(X_t))| < \infty. \tag{15.4}
$$

* 在目标策略 $\pi$ 下由 MDP 生成的状态变量 $X_t$ 弱收敛于平稳分布 $F_\pi$，并且也满足 $\rho$-混合条件 (15.4)。
* 分布 $F$ 和 $F_\pi$ 是**等价测度（equivalent measures）**。

注意，记 $\mu_\pi(x) = \mathbb{E}_P[Y_t \mid X_t = x, W_t = \pi(x)]$，倒数第二个假设意味着我们的目标存在并且可以表示为 $V(\pi) = \mathbb{E}_{F_\pi}[\mu_\pi(X)]$。

给定这一设定，我们可以写出 $V(\pi)$ 的一个**双重稳健估计量（doubly robust estimator）**，用**超额奖励函数（excess reward function）** 表示：

$$
Q_\pi(x) = \lim_{T \to \infty} \mathbb{E}_\pi \left[ \sum_{t=1}^{T} (Y_t - V(\pi)) \mid X_1 = x \right], \tag{15.5}
$$

该函数衡量从特定状态 $x$ 而非从 $F_\pi$ 的随机抽取出发，在 $\pi$ 下的期望（非缩放）超额奖励的大小，以及**平稳分布比率（stationary distribution ratio）**：

$$
\omega_\pi(x) = dF_\pi(x) / dF(x). \tag{15.6}
$$

给定这两个量的估计值，并假设 $e(\cdot)$ 是已知的（如同在顺序随机化实验中那样），那么估计量

$$
\widehat{V}_{DR}(\pi) = \frac{\sum_{t=1}^{T-1} \left(Y_t + \widehat{Q}_\pi(X_{t+1}) - \widehat{Q}_\pi(X_t)\right) \hat{\omega}_\pi(X_t) \frac{\mathbf{1}(\{W_t = \pi(X_t)\})}{e_\pi(X_t)}}{\sum_{t=1}^{T-1} \hat{\omega}_\pi(X_t) \frac{\mathbf{1}(\{W_t = \pi(X_t)\})}{e_\pi(X_t)}} \tag{15.7}
$$

对于 $V(\pi)$ 是一致的，并且是（强）双重稳健的，如第 3 章所讨论的。上面，我们使用了符号简写 $e_\pi(x) = \pi(x)e(x) + (1-\pi(x))(1-e(x))$ 来表示遵循 $\pi(\cdot)$ 的条件概率。

本节的剩余部分将致力于证明这一结果。为简单起见，我们将不依赖于**交叉拟合（cross-fitting）**，而是假设估计量 $\hat{\omega}_\pi(\cdot)$ 和 $\widehat{Q}_\pi(\cdot)$ 已经在单独的训练集上获得；然而，我们确实注意到，在适当的混合假设下，跨越时间序列 $\left(X_t, Y_t, W_t\right)$ 的长的连续段的交叉拟合论证也是可能的。最后，如同本书其余部分，我们将把估计函数 $\hat{\omega}_\pi(\cdot)$ 和 $\widehat{Q}_\pi(\cdot)$ 的方法交由统计学习文献处理；参见 Liao 等人 [2022] 和 Uehara, Huang, and Jiang [2020] 的最新提案。

我们首先建立两个结果，以说明估计量 (15.7) 的形式的合理性。注意，这两个结果一起已经隐含了该估计量的弱双重稳健性。

**引理 15.1.** 在我们陈述的假设下，且 $\operatorname{Var}_{F_\pi}[\mu_\pi(X)] < \infty$，超额奖励函数 $Q_\pi(X_t)$ 在 $F_\pi$ 下是**绝对可积的（absolutely integrable）**，在 $X_t \sim F$ 下几乎必然有限，并且满足**贝尔曼条件（Bellman conditions）**：

$$
\begin{array}{l} \mathbb{E}_\pi \left[ Y_t + Q_\pi(X_{t+1}) \mid X_t \right] - Q_\pi(X_t) = V(\pi), \\ \mathbb{E}_0 \left[ \frac{\mathbf{1}(\{W_t = \pi(X_t)\})}{e_\pi(X_t)} \left(Y_t + Q_\pi(X_{t+1})\right) \mid X_t \right] - Q_\pi(X_t) = V(\pi), \tag{15.8} \\ \end{array}
$$

几乎必然成立。

**证明.** 鉴于系统的**时间齐次性（time-homogeneity）**，将链式法则应用于 (15.5) 意味着：

$$
\mathbb{E}_\pi \left[ Q_\pi(X_{t+1}) \mid X_t = x \right] = \lim_{T \to \infty} \mathbb{E}_\pi \left[ \sum_{t=2}^{T} \left(Y_t - V(\pi)\right) \mid X_1 = x \right].
$$

¹ 事实上，平均而言，Uber 的运营中大约有 1% 的时间处于高峰状态，因此，即使一个实验在给定市场运行一个月，高峰事件的数量也相对较少，这使得分析变得复杂。

第一个贝尔曼方程随后可直接通过基本的代数运算得出——前提是我们能够证明在 $X _ { t } \sim F$ 下，$Q _ { \pi } ( X _ { t } )$ 几乎必然有限。为了验证这一点，我们将在下面证明

$$
\sum_ {t = 1} ^ {\infty} \mathbb {E} _ {X _ {1} \sim F _ {\pi}} \left[ \left| \mathbb {E} _ {\pi} \left[ Y _ {t} - V (\pi) \mid X _ {1} \right] \right| \right] <   \infty ; \tag {15.9}
$$

那么根据**富比尼定理（Fubini's theorem）**可知，在 $F _ { \pi }$ 下 $Q _ { \pi } ( X _ { t } )$ 是绝对可积的，即 $\mathbb { E } _ { X _ { 1 } \sim F _ { \pi } } \left[ \vert Q _ { \pi } ( X _ { 1 } ) \vert \right] < \infty$ 。这也意味着在 $X _ { t } \sim F$ 下 $Q _ { \pi } ( X _ { t } )$ 几乎必然有限，因为 $F$ 和 $F _ { \pi }$ 是等价测度。同时，第二个贝尔曼方程可由第一个方程，通过定理 14.2 证明中使用的在**序列无混杂性（sequential unconfoundedness）**下的标准**逆概率加权（Inverse Probability Weighting, IPW）**论证得出。

我们现在转向在 $\rho$ -混合（$\rho$-mixing）假设下验证 (15.9)。记

$$
\rho_ {\pi} ^ {t} = \sup _ {f, g \in L _ {2} (F _ {\pi})} | \mathrm{Corr} _ {\pi} (f (X _ {1}), g (X _ {t})) |,
$$

并回顾我们的假设是 $\textstyle \sum _ { t = 1 } ^ { \infty } \rho _ { \pi } ^ { t } < \infty$ 。现在，应用**詹森不等式（Jensen's inequality）**可得

$$
\mathbb {E} _ {\pi} \left[ \left| \mathbb {E} _ {\pi} \left[ Y _ {t} - V (\pi) \mid X _ {1} \right] \right| \right] \leq \mathbb {E} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y _ {t} - V (\pi) \mid X _ {1} \right] ^ {2} \right] ^ {\frac {1}{2}} = \mathrm{Var} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y _ {t} \mid X _ {1} \right] \right] ^ {\frac {1}{2}},
$$

这里我们隐含地使用了 $X _ { 1 } \sim F _ { \pi }$ 这一事实。此外，

$$
\begin{array}{l} \mathrm{Var} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y _ {t} \mid X _ {1} \right] \right] = \mathrm{Cov} _ {\pi} \left[ \mu_ {\pi} (X _ {t}), \mathbb {E} _ {\pi} \left[ Y _ {t} \mid X _ {1} \right] \right] \\ = \operatorname{Corr} _ {\pi} \left(\mu_ {\pi} (X _ {t}), \mathbb {E} _ {\pi} \left[ Y _ {t} \mid X _ {1} \right]\right) \\ \times \operatorname{Var} _ {\pi} \left[ \mu_ {\pi} (X _ {t}) \right] ^ {1 / 2} \operatorname{Var} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y _ {t} \mid X _ {1} \right] \right] ^ {1 / 2}, \\ \end{array}
$$

因此

$$
\mathrm{Var} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y _ {t} \mid X _ {1} \right] \right] ^ {1 / 2} \leq \rho_ {\pi} ^ {t} \mathrm{Var} _ {F _ {\pi}} \left[ \mu_ {\pi} (X) \right] ^ {1 / 2}.
$$

将所有部分整合起来，我们得到

$$
\sum_ {t = 1} ^ {\infty} \mathbb {E} _ {X _ {1} \sim F _ {\pi}} \left[ \left| \mathbb {E} _ {\pi} \left[ Y _ {t} - V (\pi) \mid X _ {1} \right] \right| \right] \leq \mathrm{Var} _ {F _ {\pi}} \left[ \mu_ {\pi} (X) \right] ^ {1 / 2} \sum_ {t = 1} ^ {\infty} \rho_ {\pi} ^ {t} <   \infty ,
$$

如所述。

![image_12](images/image_12.png)

**引理 15.2（Lemma 15.2）**。在我们给定的假设下，对于任意时间 $t$ 和任意可测函数 $h(X)$，

$$
\mathbb {E} _ {0} \left[ \omega_ {\pi} \left(X _ {t}\right) h \left(X _ {t + 1}\right) \frac {1 \left(\left\{W _ {t} = \pi (X _ {t}) \right\}\right)}{e _ {\pi} (X _ {t})} \right] = \mathbb {E} _ {0} \left[ \omega_ {\pi} \left(X _ {t}\right) h \left(X _ {t}\right) \right], \tag {15.10}
$$

假设所有涉及的期望存在且有限。

证明。从右侧表达式开始，我们可以利用平稳性以及测度变换（change-of-measure）论证来验证：

$$
\mathbb {E} _ {0} \left[ \omega_ {\pi} (X _ {t}) h (X _ {t}) \right] = \mathbb {E} _ {F} \left[ \omega_ {\pi} (X) h (X) \right] = \mathbb {E} _ {F _ {\pi}} [ h (X) ].
$$

同时，对于左侧，在序列无混杂性下的标准 IPW 论证意味着

$$
\mathbb {E} _ {0} \left[ h \left(X _ {t + 1}\right) \frac {1 \left(\left\{W _ {t} = \pi (X _ {t}) \right\}\right)}{e _ {\pi} (X _ {t})} \mid X _ {t} \right] = \mathbb {E} _ {\pi} \left[ h \left(X _ {t + 1}\right) \mid X _ {t} \right],
$$

因此应用链式法则（chain rule）可得

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \omega_ {\pi} (X _ {t}) h (X _ {t + 1}) \frac {1 (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})} \right] \\ = \mathbb {E} _ {0} \left[ \omega_ {\pi} (X _ {t}) \mathbb {E} _ {0} \left[ h (X _ {t + 1}) \frac {1 (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})} | X _ {t} \right] \right] \\ = \mathbb {E} _ {0} \left[ \omega_ {\pi} \left(X _ {t}\right) \mathbb {E} _ {\pi} \left[ h \left(X _ {t + 1}\right) \mid X _ {t} \right] \right] \\ = \mathbb {E} _ {X _ {t} \sim F} \left[ \omega_ {\pi} (X _ {t}) \mathbb {E} _ {\pi} \left[ h (X _ {t + 1}) \mid X _ {t} \right] \right] \\ = \mathbb {E} _ {X _ {t} \sim F _ {\pi}} \left[ \mathbb {E} _ {\pi} \left[ h \left(X _ {t + 1}\right) \mid X _ {t} \right] \right] = \mathbb {E} _ {F _ {\pi}} \left[ h (X) \right], \\ \end{array}
$$

其中第三和第五个等式利用了平稳性。

**定理 15.3（Theorem 15.3）**。在我们给定的假设下，进一步假设我们在独立的训练数据上估计 (15.7) 中的**干扰分量（nuisance components）**，使得对于所有 $t = 1, . . . , T$，76

$$
\mathbb {E} _ {F} \left[ \left(\widehat {Q} _ {\pi} (X) - Q _ {\pi} (X)\right) ^ {2} \right] = o _ {P} \left(T ^ {- 2 \alpha_ {Q}}\right), \tag {15.11}
$$

$$
\mathbb {E} _ {F} \left[ \left(\hat {\omega} _ {\pi} (X) - \omega_ {\pi} (X)\right) ^ {2} \right] = o _ {P} \left(T ^ {- 2 \alpha_ {\omega}}\right)
$$

对于常数 $\alpha_Q$， $\alpha _ { \omega } \geq 0$ 且 $\alpha _ { \omega } + \alpha _ { Q } \ge 1 / 2$ 。那么，

$$
\begin{array}{l} \sqrt {T} \left(\widehat {V} _ {D R} (\pi) - V (\pi)\right) \Rightarrow \mathcal {N} (0, \Sigma) \\ \Sigma = \mathbb {E} _ {F} \left[ \frac {\omega_ {\pi} ^ {2} (X _ {1})}{e _ {\pi} (X _ {1})} \mathbb {E} _ {\pi} \left[ (Y _ {1} + Q _ {\pi} (X _ {2}) - Q _ {\pi} (X _ {1}) - V (\pi)) ^ {2} \mid X _ {1} \right] \right], \tag {15.12} \\ \end{array}
$$

假设 $\Sigma$ 有限。

证明。我们的估计量具有**自标准化（self-normalized）**形式，因此其误差可以表示为

$$
\widehat {V} _ {D R} (\pi) - V (\pi) = \frac {\sum_ {t = 1} ^ {T - 1} \left(Y _ {t} + \widehat {Q} _ {\pi} (X _ {t + 1}) - \widehat {Q} _ {\pi} (X _ {t}) - V (\pi)\right) \hat {\omega} _ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})}}{\sum_ {t = 1} ^ {T - 1} \hat {\omega} _ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})}}.
$$

我们首先考虑分母。由平稳性，

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})} \right] = \mathbb {E} _ {0} \left[ \omega_ {\pi} (X _ {t}) \mathbb {E} _ {0} \left[ \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})} | X _ {t} \right] \right] \\ = \mathbb {E} _ {0} [ \omega_ {\pi} (X _ {t}) ] = \mathbb {E} _ {F} [ \omega_ {\pi} (X) ] = 1, \\ \end{array}
$$

因此我们可以应用**遍历定理（ergodic theorem）** $[ \mathrm { e . g . }$ , Durrett, 2019, Chapter 6.2] 来验证

$$
\frac {1}{T - 1} \sum_ {t = 1} ^ {T - 1} \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})} \rightarrow_ {p} 1. \tag {15.13}
$$

此外，我们看到

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \left| \frac {1}{T - 1} \sum_ {t = 1} ^ {T - 1} \left(\hat {\omega} _ {\pi} (X _ {t}) - \omega_ {\pi} (X _ {t})\right) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})} \right| \right] \\ \leq \frac {1}{\eta^ {2}} \sqrt {\mathbb {E} _ {0} \left[ \frac {1}{T - 1} \sum_ {t = 1} ^ {T - 1} \left(\hat {\omega} _ {\pi} (X _ {t}) - \omega_ {\pi} (X _ {t})\right) ^ {2} \right]} \\ = \frac {1}{\eta^ {2}} \sqrt {\mathbb {E} _ {F} \left[ (\hat {\omega} _ {\pi} (X) - \omega_ {\pi} (X)) ^ {2} \right]} = o _ {p} (1) \\ \end{array}
$$

分别通过应用柯西-施瓦茨（Cauchy-Schwarz）不等式、重叠（overlap）条件、平稳性和 $\hat { \omega } ( \cdot )$ 的 $L _ { 2 }$ 一致性，从而表明当 $\omega ( \cdot )$ 替换为 $\hat { \omega } ( \cdot )$ 时，(15.13) 仍然成立。

<!-- footnote -->

- Neyman [1923] 在**完全随机化（complete randomization）** 条件下进行研究，即处理单元的数量是事先固定的；然而，所有关键见解都是相同的。

<!-- footnote end -->

<!-- footnote -->

- $^{66}$ 在方差估计 $\widehat { V } _ { D M }$ 的公式 (1.10) 中，我们使用了归一化因子 $n _ { 0 } / n$ 和 $n _ { 1 } / n$，而在公式 (12.11) 中，它们分别被替换为 $1 - \pi$ 和 $\pi$；然而，这种区别在一阶分析下是无关紧要的。在均匀随机化设置中，当所有单元的 $e _ { i } = \pi$ 时，这些方差估计是渐近等价的，并且其中任何一个都可用于构建置信区间。

<!-- footnote end -->

<!-- footnote -->

- **HAC 估计量（HAC estimator）** 的构建仅用于启发下方方差估计量的函数形式；其在我们的设定中的一致性将从基本原理出发在下文中得到证明。关于相关随机变量的 HAC 估计量的一般性讨论，请参见 White [1984, Chapter VI.4]；关于网络相关模型下 HAC 估计量的最新结果，请参见 Kojevnikov, Marmer, and Song [2021]。
- 作为一项合理性检验，可以验证，在 **SUTVA 假设（SUTVA）** 下（即 $\boldsymbol { G } = \boldsymbol { I } _ { n \times n }$ 时），公式 (12.18) 与公式 (12.8) 完全一致。

<!-- footnote end -->

<!-- footnote -->

- 这一现象在概念上与我们在定理 2.1 中观察到的现象相关，即在增加分层数量时，ATE 的分层估计量的渐近方差并未变差。

<!-- footnote end -->

<!-- footnote -->

- 注意，这里我们仅对与我们当前所处轨迹一致（即 $w _ { i ( 1 : ( t - 1 ) ) } = W _ { i ( 1 : ( t - 1 ) ) }$）的潜在结果施加**无混杂性（unconfoundedness）** 条件。其他潜在结果已无法达到，因此，在给定 $w _ { i ( 1 : ( t - 1 ) ) } = W _ { i ( 1 : ( t - 1 ) ) }$ 的条件下，它们的分布对于策略评估不再重要。

<!-- footnote end -->

<!-- footnote -->

- 这些**主分层（principal strata）** 与第 10.1 章讨论的 IV 分析中的依从性类型之间存在密切的概念联系。
- 根据此记号，策略价值本身也可以写作 $V _ { \pi , 0 } = V ( \pi )$。

<!-- footnote end -->

<!-- footnote -->

- 与本书其余部分不同，我们在此使用 $\sigma ^ { 2 }$ 而非 $V ^ { * }$ 来表示渐近方差，这是为了遵循强化学习文献中将价值函数写作 V 的标准惯例。

<!-- footnote end -->

<!-- footnote -->

- 以下期望是针对测试数据取的；要求是在独立数据上进行的训练能够以高概率产生具有良好测试集均方误差的估计。

<!-- footnote end -->

<!-- footnote -->

- 当一个平台运营多个独立市场时，它们也可以通过跨市场随机分配处理来运行实验。然而，这种策略下的有效样本量（即

<!-- footnote end -->

<!-- footnote -->

- 处理随机化的次数）是市场的数量，因此，这种方法通常仅在能够跨大量市场进行实验时才具有吸引力。

<!-- footnote end -->

<!-- footnote -->

- 以下期望是针对测试数据取的；要求是在独立数据上进行的训练能够以高概率产生具有良好测试集均方误差的估计。

<!-- footnote end -->

同时，分子可以分解为 $A + B + C + D$，其中

$$
A = \sum_ {t = 1} ^ {T - 1} \left(Y _ {t} + Q _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t}) - V (\pi)\right) \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})},
$$

$$
B = \sum_ {t = 1} ^ {T - 1} \left(Y _ {t} + Q _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t}) - V (\pi)\right) \left(\hat {\omega} _ {\pi} (X _ {t}) - \omega_ {\pi} (X _ {t})\right) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})},
$$

$$
C = \sum_ {t = 1} ^ {T - 1} \left(\widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) - \left(\widehat {Q} _ {\pi} (X _ {t}) - Q _ {\pi} (X _ {t})\right)\right) \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})},
$$

$$
D = \sum_ {t = 1} ^ {T - 1} \left(\widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) - \left(\widehat {Q} _ {\pi} (X _ {t}) - Q _ {\pi} (X _ {t})\right)\right)
$$

$$
\times (\hat {\omega} _ {\pi} (X _ {t}) - \omega_ {\pi} (X _ {t})) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})}.
$$

我们将在下面证明

$$
A / \sqrt {T} \Rightarrow \mathcal {N} (0, \Sigma), \quad | B |, | C |, | D | = o _ {P} (\sqrt {T}). \tag {15.14}
$$

因此，结合上面关于分母的结论，我们可以通过 **Slutsky 引理（Slutsky’s lemma）** 建立 (15.12)。

现在，从（主要的）项 A 开始，我们注意到引理 15.1 中的第二个 **Bellman 方程（Bellman equation）** 立即表明

$$
\mathbb {E} _ {0} \left[ (Y _ {t} + Q _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t}) - V (\pi)) \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})} | X _ {t} \right] = 0
$$

对所有 t 几乎必然成立，因此项 A 的均值为零。此外，根据我们假设的 **马尔可夫性质（Markov property）**，构成 A 的求和项是一个 **鞅差序列（martingale difference sequence）**，因为对 $X _ { t }$ 取条件等价于对整个过去取条件。基于此设定，我们可以通过 **鞅中心极限定理（martingale central limit theorem）** 研究 A 的大样本行为。这样做的一个关键要素是研究单个鞅差项的条件方差。我们可以再次应用 **遍历定理（ergodic theorem）** 来验证

$$
\frac {1}{T - 1} \sum_ {t = 1} ^ {T - 1} \operatorname{Var} _ {0} \left[ \Delta_ {t, t + 1} \mid X _ {t} \right]\rightarrow_ {p} \mathbb {E} _ {X _ {1} \sim F} \left[ \operatorname{Var} _ {0} \left[ \Delta_ {1, 2} \mid X _ {1} \right]\right],
$$

$$
\Delta_ {t, t + 1} = (Y _ {t} + Q _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t}) - V (\pi)) \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})},
$$

假设右侧极限是有限的。此外，

$$
\begin{array}{l} \mathbb {E} _ {F} \left[ \operatorname{Var} _ {0} \left[ \Delta_ {1, 2} \mid X _ {1} \right] \right] = \mathbb {E} _ {F} \left[ \mathbb {E} _ {0} \left[ \Delta_ {1, 2} ^ {2} \mid X _ {1} \right] \right] \\ = \mathbb {E} _ {F} \left[ \mathbb {E} _ {0} \left[ 1 \left(\left\{W _ {1} = \pi (X _ {1}) \right\}\right) \Delta_ {1, 2} ^ {2} \mid X _ {1} \right] \right] \\ = \mathbb {E} _ {F} \left[ e _ {\pi} (X _ {1}) \mathbb {E} _ {\pi} \left[ \Delta_ {1, 2} ^ {2} \mid X _ {1} \right] \right] = \Sigma , \\ \end{array}
$$

其中第二个等式成立是因为当 $W _ { 1 } \neq \pi ( X _ { 1 } )$ 时 $\Delta _ { 1 , 2 } ^ { 2 } = 0$，第三个等式由**序贯无混杂性（sequential unconfoundedness）** 保证，第四个等式则通过直接的代数运算得出。现在，我们在定理陈述中假设了 $\Sigma \ < \ \infty$；因此遍历定理实际上适用。由此，$A / \sqrt { T } \Rightarrow \mathcal { N } ( 0 , \Sigma )$ 的结论即可从鞅中心极限定理 [例如，Durrett, 2019, Theorem 8.2.8] 推出。

接下来，转向低阶项，引理 15.1 表明

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \left(Y _ {t} + Q _ {\pi} \left(X _ {t + 1}\right) - Q _ {\pi} \left(X _ {t}\right) - V (\pi)\right) \right. \\ \times \left. (\hat {\omega} _ {\pi} (X _ {t}) - \omega_ {\pi} (X _ {t})) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})} \big | X _ {t} \right] = 0, \\ \end{array}
$$

因此项 B 的均值为零。此外，它同样是一个鞅，因此其方差等于每个鞅差项期望方差之和；因此，由平稳性，

$$
\begin{array}{l} \operatorname{Var} [ B ] = (T - 1) \mathbb {E} _ {F} \left[ \operatorname{Var} _ {0} \left[ \left(Y _ {1} + Q _ {\pi} \left(X _ {2}\right) - Q _ {\pi} \left(X _ {1}\right) - V (\pi)\right) \right. \right. \\ \left. \times \left(\hat {\omega} _ {\pi} (X _ {1}) - \omega_ {\pi} (X _ {1})\right) \frac {\mathbf {1} \left(\{W _ {1} = \pi (X _ {1}) \}\right)}{e _ {\pi} (X _ {1})} \mid X _ {1} \right] \\ = (T - 1) \mathbb {E} _ {F} \left[ \frac {\left(\hat {\omega} _ {\pi} \left(X _ {1}\right) - \omega_ {\pi} \left(X _ {1}\right)\right) ^ {2}}{e _ {\pi} \left(X _ {1}\right)} \operatorname{Var} _ {\pi} \left[ Y _ {1} + Q _ {\pi} \left(X _ {2}\right) \mid X _ {1} \right] \right] \\ = \mathcal {O} \left((T - 1) \mathbb {E} _ {F} \left[ (\hat {\omega} _ {\pi} (X _ {1}) - \omega_ {\pi} (X _ {1})) ^ {2} \right]\right) = o _ {p} (T), \\ \end{array}
$$

因此 $B = o _ { p } ( { \sqrt { T } } )$。

同时，我们可以使用引理 15.2 验证项 C 的均值为零：

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \left(\widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) - \left(\widehat {Q} _ {\pi} (X _ {t}) - Q _ {\pi} (X _ {t})\right)\right) \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})} \right] \\ = \mathbb {E} _ {0} \left[ \left(\widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1})\right) \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})} \right] \\ - \mathbb {E} _ {0} \left[ \left(\widehat {Q} _ {\pi} (X _ {t}) - Q _ {\pi} (X _ {t})\right) \omega_ {\pi} (X _ {t}) \right] = 0. \\ \end{array}
$$

为了计算 C 的方差，将其分为两部分是有帮助的：

$$
\begin{array}{l} C _ {1} = \sum_ {t = 1} ^ {T - 1} \left(\mathbb {E} \left[ \widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) \mid X _ {t}, W _ {t} \right] - \left(\widehat {Q} _ {\pi} (X _ {t}) - Q _ {\pi} (X _ {t})\right)\right) \\ \times \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})}, \\ C _ {2} = \sum_ {t = 1} ^ {T - 1} \left(\widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) - \mathbb {E} \left[ \widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) \mid X _ {t}, W _ {t} \right]\right) \\ \times \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})}. \\ \end{array}
$$

后一项 $C _ { 2 }$ 是一个鞅，因此可以通过类似于处理 $B$ 的论证证明其为 $o _ { p } ( \sqrt { T } )$。然而，项 $C _ { 1 }$ 并非鞅，因此交叉项变得重要。由平稳性，

$$
\begin{array}{l} \operatorname{Var} \left[ C _ {1} \right] = (T - 1) \operatorname{Var} _ {F} \left[ \omega_ {\pi} \left(X _ {1}\right) \frac {\mathbf {1} \left(\left\{W _ {1} = \pi \left(X _ {1}\right) \right\}\right)}{e _ {\pi} \left(X _ {1}\right)} \right. \\ \times \left(\mathbb {E} \left[ \widehat {Q} _ {\pi} (X _ {2}) - Q _ {\pi} (X _ {2}) \mid X _ {1}, W _ {1} \right] - \left(\widehat {Q} _ {\pi} (X _ {1}) - Q _ {\pi} (X _ {1})\right)\right) \\ + (T - 2) \operatorname{Cov} _ {F} \left[ \omega_ {\pi} \left(X _ {1}\right) \frac {\mathbf {1} \left(\left\{W _ {1} = \pi \left(X _ {1}\right) \right\}\right)}{e _ {\pi} \left(X _ {1}\right)} \right. \\ \times \left(\mathbb {E} \left[ \widehat {Q} _ {\pi} (X _ {2}) - Q _ {\pi} (X _ {2})   |   X _ {1},   W _ {1} \right] - \left(\widehat {Q} _ {\pi} (X _ {1}) - Q _ {\pi} (X _ {1})\right)\right), \\ \omega_ {\pi} (X _ {2}) \frac {\mathbf {1} (\{W _ {2} = \pi (X _ {2}) \})}{e _ {\pi} (X _ {2})} \\ \times \left(\mathbb {E} \left[ \widehat {Q} _ {\pi} (X _ {3}) - Q _ {\pi} (X _ {3}) \mid X _ {2}, W _ {2} \right] - \left(\widehat {Q} _ {\pi} (X _ {2}) - Q _ {\pi} (X _ {2})\right)\right) \\ + (T - 3) \dots \\ \end{array}
$$

然后，根据我们假设的 $\rho -$ **混合性（$\rho$-mixing）** 条件，我们可以将该项上界估计为

$$
\mathrm{Var} \left[ C _ {1} \right] \leq (T - 1) \sum_ {t = 1} ^ {\infty} \rho_ {t} \mathrm{Var} _ {F} \bigg [ \omega_ {\pi} (X _ {1}) \frac {\mathbf {1} \left(\{W _ {1} = \pi (X _ {1}) \}\right)}{e _ {\pi} (X _ {1})}
$$

$$
\times \left(\mathbb {E} \left[ \widehat {Q} _ {\pi} (X _ {2}) - Q _ {\pi} (X _ {2})   |   X _ {1},   W _ {1} \right] - \left(\widehat {Q} _ {\pi} (X _ {1}) - Q _ {\pi} (X _ {1})\right)\right),
$$

回想我们假设了 $\textstyle \sum _ { t = 1 } ^ { \infty } \rho _ { t } < \infty$ 。根据我们对 $\widehat { Q }$ 的 **L2 一致性（L2-consistency）** 假设以及对 $\omega ( X _ { t } )$ 和 $1 / e _ { \pi } ( X _ { t } )$ 的有界性假设，这意味着 $C _ { 1 } = o _ { p } ( \sqrt { T } )$。

最后，如同许多证明中已经做的那样，项 D 可以通过 **柯西-施瓦茨不等式（Cauchy-Schwarz inequality）** 进行界定：

$$
\begin{array}{l} | D | \leq \frac {1}{\eta} \sqrt {\sum_ {t = 1} ^ {T - 1} \left(\widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) - \left(\widehat {Q} _ {\pi} (X _ {t}) - Q _ {\pi} (X _ {t})\right)\right) ^ {2}} \\ \times \sqrt {\sum_ {t = 1} ^ {T - 1} (\hat {\omega} _ {\pi} (X _ {t}) - \omega_ {\pi} (X _ {t})) ^ {2}} \\ = \mathcal {O} _ {P} \left((T - 1) \mathbb {E} _ {F} \left[ \left(\widehat {Q} _ {\pi} (X) - Q _ {\pi} (X)\right) ^ {2} \right] ^ {\frac {1}{2}} \mathbb {E} _ {F} \left[ \left(\hat {\omega} _ {\pi} (X) - \omega_ {\pi} (X)\right) ^ {2} \right] ^ {\frac {1}{2}}\right) \\ = o _ {p} (\sqrt {T}), \\ \end{array}
$$

其中第二行由平稳性和马尔可夫不等式得出，最后一行由 (15.11) 得出。□

## 15.2 切换实验（Switchback experiments）

以上我们展示了——在牺牲一定数学复杂性的前提下——如何利用在一般**顺序随机化设计（sequentially randomized design）**下收集的数据来估计**马尔可夫决策过程（Markov decision processes）**中的策略价值。然而，在实践中，改变数据收集程序以更直接地适应问题结构，从而能够进行更直接的分析，可能会更容易。

其中一种设计就是**切换实验（switchback experiment）**。原则上，任何通过在系统层面反复切换治疗开和关来测量治疗效应的实验，都可以被称为切换实验。然而，在存在**时间性遗留效应（temporal carryovers）**的系统中，切换实验通常被理解为：将治疗设置为给定水平，等待系统重新达到平衡，然后才再次切换它。在运行切换实验时，目标通常是估计**总治疗效应（total treatment effect）**：

$$
\tau_{TOT} = V(1) - V(0) \tag {15.15}
$$

即，始终治疗策略与从不治疗策略之间的长期平均差异。

实践中考虑过多种切换实验设计。最简单（且最广泛使用）的切换实验设计具有固定长度为 $L$ 的治疗窗口，并在每 $L$ 个时间段后切换治疗 [Bojinov, Simchi-Levi, and Zhao, 2023]。在此，我们将考虑一种替代性的“**无记忆切换实验（memoryless switchback）**”设计，因为它允许在本章使用的马尔可夫模型背景下进行特别简单的分析。关于在马尔可夫模型下标准（即固定长度）切换实验的讨论，以及在时变设置（即定义 15.1 中的 $P_t$ 随时间变化）下的结果，请参见 Hu and Wager [2022]。

**定义 15.2.** 一个具有切换率 $0 < \lambda < 1$ 的**无记忆切换实验**是一种顺序分配治疗 $W_t \in \{0, 1\}$（其中 $t = 1, 2, ...$）的设计，使得 $W_1 \sim \mathrm{Bernoulli}(0.5)$，且对于 $t \geq 1$：

$$
W_{t+1} \sim \text{Bernoulli} \left((1 - \lambda) W_t + \lambda (1 - W_t)\right). \tag {15.16}
$$

关于切换实验的核心事实是，如果治疗切换之间的典型时间足够长（即，在无记忆切换实验的情况下，如果切换率 $\lambda$ 足够低），那么原始的**均值差异估计量（difference in means estimator）**

$$
\hat{\tau}_{SB} = \frac{1}{|W_t = 1|} \sum_{\{t: W_t = 1\}} Y_t - \frac{1}{|W_t = 0|} \sum_{\{t: W_t = 0\}} Y_t \tag {15.17}
$$

对于总效应是一致的。在实践中，可以通过移除切换后立即出现的**预烧样本（burn-in samples）**以及其他算法修改来改善该估计量的行为 [Bojinov, Simchi-Levi, and Zhao, 2023, Hu and Wager, 2022]；然而，在此我们将重点关注基本估计量 (15.17)。

为了研究切换实验估计量，我们将在“**表格型（tabular）**”设置中工作，其中协变量 $X_t \in \mathcal{X}$ 在一个离散空间中取值，且 $|\mathcal{X}| = k$，这意味着我们可以将完整的依赖治疗的状态转移矩阵写为 $P^w \in \mathbb{R}^{k \times k}$，其中 $P_{xx'}^w = \mathbb{P}\left[X_{t+1} = x \middle| X_t = x', W_t = w\right]$。我们的分析也直接适用于非表格型设置；然而，离散设置大大简化了符号表示。

我们将进一步假设**几何混合（geometric mixing）**，即状态转移算子是一个压缩映射：

$$
\left\| P^w (\nu' - \nu) \right\|_1 \leq e^{-1/t_0} \left\| \nu' - \nu \right\|_1 \tag {15.18}
$$

对于 $\mathcal{X}$ 上的任意测度 $\nu, \nu'$ 成立，即对于 $[0, 1]^k$ 上满足 $\textstyle \sum_x \nu_x = 1$ 的向量（$\nu'$ 同理）；该条件立即意味着存在唯一的平稳分布，并且以混合时间 $t_0$ 几何收敛到该平稳分布。

**定理 15.4.** 考虑一个满足 (15.18) 的**时间齐次马尔可夫决策过程（time-homogenous Markov decision process）**，并进一步假设 $|Y_t| \le M$ 几乎必然成立。那么，将 $\tau_{SB}(\lambda)$ 写为在切换率为 $\lambda$ 的马尔可夫切换实验下 $\hat{\tau}_{SB}$ 的长期平均，我们有：

$$
\left| \tau_{SB}(\lambda) - \tau_{TOT} \right| \leq 4 M \lambda \left(1 + t_0\right). \tag {15.19}
$$

此外，如果我们运行一系列具有时间范围 $T$ 和切换率 $\lambda_T$ 的无记忆切换实验，那么当 $\lambda_T \to 0$ 且 $T \lambda_T \to \infty$ 时，$\hat{\tau}_{SB} \xrightarrow{p} \tau_{TOT}$。

**证明.** 首先，作为预备知识，我们注意到混合条件 (15.18) 意味着存在平稳分布 $\nu^0$ 和 $\nu^1$，它们可以表征为 $k$ 维单纯形上 $P^w \nu^w = \nu^w$ 的唯一解；并且始终治疗和从不治疗策略的长期平均值为 $V(w) = \sum_x \nu_x^w \mathbb{E}[Y_t \mid X_t = x, W_t = w]$。

现在，转向切换实验：我们的假设 $(X_t, Y_t)$ 来自一个马尔可夫决策过程，而 $W_t$ 以 (15.16) 中给出的无记忆方式进行随机化，这意味着 $(X_t, Y_t, W_t)$ 共同构成一个马尔可夫链。将 $\nu^w(\lambda)$ 写为在平稳状态下 $X_t$ 在 $W_t = w$ 条件下的分布，则 $(X_t, W_t)$ 联合平稳分布下的不动点条件是：

$$
\binom{\nu^0(\lambda)}{\nu^1(\lambda)} = \left( \begin{array}{c c} (1 - \lambda) P^0 & \lambda P^1 \\ \lambda P^0 & (1 - \lambda) P^1 \end{array} \right) \binom{\nu^0(\lambda)}{\nu^1(\lambda)}. \tag {15.20}
$$

此外，均值差异估计量的长期平均期望是：

$$
\begin{array}{l} \tau_{SB}(\lambda) = \sum_{x \in \mathcal{X}} \nu_x^1(\lambda) \mathbb{E} \left[ Y_t \mid X_t = x, W_t = 1 \right] \tag {15.21} \\ - \sum_{x \in \mathcal{X}} \nu_x^0(\lambda) \mathbb{E} \left[ Y_t \mid X_t = x, W_t = 0 \right], \\ \end{array}
$$

因此，根据有界性，我们立即看到：

$$
\left| \tau_{SB}(\lambda) - \tau_{TOT} \right| \leq M \left(\left\| \nu^0(\lambda) - \nu^0 \right\|_1 + \left\| \nu^1(\lambda) - \nu^1 \right\|_1\right). \tag {15.22}
$$

剩下的工作是限制上述表达式右侧的值，为此我们使用混合性质。

专注于 $w = 0$ 的情况，(15.20) 的上半部分可以重写为：

$$
\left(I - P^0\right) \nu^0(\lambda) = \lambda \left(P^1 \nu^1(\lambda) - P^0 \nu^0(\lambda)\right),
$$

并且由于 $\nu^0$ 是 $P^0$ 的一个不动点，我们因此也有：

$$
\left(I - P^0\right) \left(\nu^0(\lambda) - \nu^0\right) = \lambda \left(P^1 \nu^1(\lambda) - P^0 \nu^0(\lambda)\right).
$$

将此表达式与 (15.18) 结合，我们得到：

$$
\begin{array}{l} \left\| \nu^0(\lambda) - \nu^0 - \lambda \left(P^1 \nu^1(\lambda) - P^0 \nu^0(\lambda)\right) \right\|_1 = \left\| P^0 \left(\nu^0(\lambda) - \nu^0\right) \right\|_1 \\ \leq e^{-1/t_0} \left\| \nu^0(\lambda) - \nu^0 \right\|_1, \\ \end{array}
$$

因此，根据三角不等式：

$$
\left(1 - e^{-1/t_0}\right) \left\| \nu^0(\lambda) - \nu^0 \right\|_1 \leq \lambda \left\| P^1 \nu^1(\lambda) - P^0 \nu^0(\lambda) \right\|_1.
$$

通过注意到 $(1 - e^{-1/t_0})^{-1} \leq 1 + t_0$ 且 $\|P^1 \nu^1(\lambda) - P^0 \nu^0(\lambda)\|_1 \leq 2$，即可得到 (15.19) 的陈述。最后，一致性结论成立是因为 $\lambda_T \to 0$ 意味着偏差根据上述论证趋于 0，而条件 $\lambda_T T \to \infty$ 意味着存在发散的切换次数，因此由于如 (15.18) 所示的混合性质，$\hat{\tau}_{SB} - \tau(\lambda_T) \xrightarrow{p} 0$。□

## 15.3 文献注释（Bibliographic notes）

几十年来，**马尔可夫决策过程（Markov decision processes）**一直是强化学习文献中持续研究的对象。我们在本章中的讨论属于该文献中常被称为**离策略学习（off-policy learning）**的领域，因为我们试图利用在一种（随机化）设计下收集的数据来预测在不同（目标）策略下的奖励。离策略设置与**在策略设置（on-policy setting）**形成对比，在后者中，我们可以访问一个模拟器，可以根据需要探索状态 [Sutton and Barto, 2018]。该文献中开发的一些值得注意的离策略算法包括**时序差分学习算法（temporal-difference learning algorithm）**，该算法试图通过关注如引理 15.1 中给出的贝尔曼方程来估计目标策略的**折现价值函数（discounted value function）**

$$
V_{\pi, \gamma}(x) = \mathbb{E}_{\pi} \left[ \sum_{t = 0}^{\infty} \gamma^t Y_t \mid X_0 = x \right], \quad 0 < \gamma < 1, \tag {15.23}
$$

[Sutton, 1988, Tsitsiklis and Van Roy, 1997]，以及用于寻找福利最大化策略的 **Q-学习算法（Q-learning algorithm）** [Watkins and Dayan, 1992, Murphy, 2005]。

本章所采用的方法建立在 Kallus and Uehara [2020] 的一系列工作之上，他们强调了马尔可夫假设在减轻影响上一章讨论的动态策略评估通用方法的**维度灾难（curse of dimensionality）**方面的作用；以及 Liao, Klasnja, and Murphy [2021] 的工作，他们展示了马尔可夫决策过程如何能够从顺序无混杂数据中识别**长期平均值（long-run average value）**。此处介绍的用于估计长期平均值的**双重稳健估计（doubly robust estimation）**方法改编自 Liao et al. [2022]；一种类似的用于估计折现策略价值（而非长期平均值）的方法在 Kallus and Uehara [2022] 中讨论。密度比 $\omega_\pi(X)$ 可能是重尾分布且定理 15.3 中给出的 $\Sigma$ 是无限的情况，由 Mehrabi and Wager [2024] 考虑；作者们表明，在这种情况下，$1/\sqrt{T}$ 一致估计不再可能，但来自定理 15.3 的双重稳健估计量的适当截断版本仍然可以达到**极小极大收敛速度（minimax rate of convergence）**。

**切换实验（Switchback experiments）**正日益成为动态系统中因果推断标准工具包的核心部分；Bojinov, Simchi-Levi, and Zhao [2023] 对该设计提供了一个全面的概述。此处呈现的分析，即切换实验用于马尔可夫决策过程中的策略评估，改编自 Hu and Wager [2022]。第 15.1 节中的双重稳健估计量与切换实验之间的一个重要实际区别在于，前者需要观察（并使用）状态变量 $X_t$，而切换实验则不需要。我们可以问，在第 15.1 节的设置中，如果我们不再能够观察到 $X_t$，而是需要像处理切换实验那样仅依赖混合性质 (15.18)，那么最优推断会如何。这种设置在 Hu and Wager [2023] 中进行了考虑，他们表明，在这种情况下，$1/\sqrt{T}$ 一致估计通常是不可能的，并且类似切换实验的截断 IPW 估计量能够达到极小极大（慢于 $1/\sqrt{T}$）速度。
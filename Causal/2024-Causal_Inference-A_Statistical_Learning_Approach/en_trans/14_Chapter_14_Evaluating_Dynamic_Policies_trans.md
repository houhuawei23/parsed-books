# 第14章 评估动态策略（Evaluating Dynamic Policies）

在上一章中，我们考虑了**事件研究（event studies）**的方法，其中某些单位接受了**处理（treatment）**（即其处理状态从关闭切换到开启），我们希望衡量进行这种切换的效果。事件研究的结果有助于判断其他单位是否也可能从采用该处理中受益。然而，事件研究设计——以及相关的**双重差分（difference in differences）**和**合成控制（synthetic controls）**等方法——在指导动态决策方面帮助较小。它们的局限性或许最好通过示例来理解。

**示例18.** 在金融危机期间，**中央银行（central banks）**有时会使用**量化宽松（quantitative easing）**来缓解长期衰退的风险。在量化宽松期间，中央银行试图通过购买政府债券和其他资产来增加市场流动性。适度的量化宽松可能有助于刺激经济并避免衰退；然而，过度的量化宽松——或持续时间过长的量化宽松——可能导致过度通货膨胀的问题 [Boehl, Goy, and Strobel, 2024]。

**示例19.** **抗逆转录病毒疗法（Antiretroviral therapy, ART）**是护理**HIV阳性患者（HIV-positive patients）**的关键药物。已知 HIV 会降低 CD4 白细胞计数，并且一旦 CD4 计数降低，患者就有患上**艾滋病定义疾病（AIDS-defined illnesses）**的风险。使用 ART 有助于维持 CD4 计数，从而预防艾滋病，但它是一种非常密集的药物治疗形式，伴有多种副作用。因此，何时开始 ART 的问题在医学文献中受到了广泛关注。传统的 HIV 治疗指南建议仅在 CD4 计数降至给定阈值以下时才开始 ART；但最近的证据支持在 HIV 确诊后立即开始 ART [Group, 2015]。

显然，成功应用量化宽松需要审慎考虑何时开始干预、提供多少流动性以及何时停止。然而，事件研究方法对此类问题提供的指导非常有限。双重差分方法所依赖的**平行趋势假设（parallel trends assumption）**实际上排除了以下可能性：在特定危机期间，一些国家可能需要量化宽松（即，没有干预它们将陷入衰退），而另一些国家则不需要（即，即使没有干预它们也会安然无恙）。合成控制方法可用于研究 ART 的效果——或量化宽松的初始效果——但不能轻易给出何时开始或停止干预的指导。

本章提出了一种完全灵活的、基于**潜在结果（potential outcome）**的方法，用于对随时间变化的因果效应进行建模，该方法允许任意的处理分配动态和**延续效应（carryover effects）**。在整章中，我们将假设我们拥有 $i = 1 , \ldots , n$ 名患者的数据，在时间点 $t = 1 , \dots , T$ 上进行观测。在每个时间点，我们观测到一组（随时间变化的）**协变量（covariates）** $X _ { i t }$ 以及一个**处理分配（treatment assignment）** $W _ { i t } \in \{ 0 , 1 \}$。最后，当我们到达时间点 $T$ 时，我们还观测到一个结果 $Y _ { i } \in \mathbb { R }$。在本章中，我们将假设单位 i 是从一个**超总体（superpopulation）**中**独立同分布（IID）**抽取的。

我们使用下面的潜在结果规范对因果效应进行建模，该规范允许任意的处理动态。请注意，该模型隐含地编码了这样一个事实：时间 t 的可观测量仅受截至时间 $t$ 的行动影响，而不受未来行动的影响，从而推广了事件研究设置中使用的**非预期条件（non-anticipation condition）**（假设 13.1）。

**定义 14.1.** 一个具有时间跨度 $T$ 的**动态决策过程（dynamic decision process）**由随时间变化的协变量 $X _ { i t } \in \mathcal { X } _ { t }$ 和结果 $Y _ { i } \in \mathbb { R }$ 刻画，其潜在结果使得每个可观测量都对所有过去的处理分配做出响应。对于每个 $X _ { i t }$，我们定义 $2 ^ { t - 1 }$ 个潜在结果 $X _ { i t } ( w _ { 1 : ( t - 1 ) } )$，使得 $X _ { i t } = X _ { i t } ( W _ { i ( 1 : ( t - 1 ) ) } )$，而对于最终结果，我们有 $2 ^ { T }$ 个潜在结果 $Y _ { i } ( w _ { 1 : T } )$，使得 $Y _ { i } = X _ { i t } ( W _ { i ( 1 : T ) } )$。

接下来，我们需要定义一个**估计量（estimand）**。在动态设置中，潜在处理分配规则的数量随跨度 $T$ 呈指数增长，我们能够提出的问题数量也同样增长。一个简单的估计量是在某个预先指定的处理规则 $w \in \left\{ 0 , 1 \right\} ^ { T }$ 下的期望结果，即 $V ( w ) = \mathbb { E } \left[ Y _ { i } ( w ) \right]$。然而，这些估计量通常与实践无关，因为它们排除了动态决策。例如，假设我们正在研究癌症治疗，并希望为在癌症诊断一年后开始化疗的处理规则估计 $V ( w )$。那么，如果一些患者在达到一年标记之前通过其他方式进入缓解期，评估 $V ( w )$ 仍然需要在此时间点开始化疗——即使这在临床上没有意义。

在实践中，评估考虑随时间变化协变量的处理规则通常更为相关。例如，我们可能会询问在尚未进入缓解期的患者中，诊断一年后开始化疗的益处，或者我们可能会询问在利率已降至零但经济活动仍然疲软时开始量化宽松的效果。通过**策略评估（policy evaluation）**的视角，我们可以定义许多此类相关的估计量，这是对我们第 5.1 章讨论的推广。

**定义 14.2.** 一个**动态策略（dynamic policy）**是一组映射 $\pi _ { t } : \mathcal { X } _ { t } \to \{ 0 , 1 \}$，它根据当前状态 $X _ { i t }$ 规定一个处理 $\pi _ { t } ( X _ { i t } )$。策略 $\pi$ 的**价值（value）**为

$$
V (\pi) = \mathbb {E} \left[ Y _ {i} (\pi_ {1} (X _ {i 1}),   \pi_ {2} (X _ {i 1},   \pi_ {1} (X _ {i 1}),   X _ {i 2} (\pi_ {1} (X _ {i 1})),   \ldots) \right], \tag {14.1}
$$

即，它捕获了在动态决策过程中根据 $\pi$ 选择处理所获得的期望奖励。

(14.1) 中复杂的符号突出了动态决策问题固有的复杂因果结构：在时间 t 做出的处理决策依赖于 $X _ { i t }$，而 $X _ { i t }$ 又依赖于在时间 $t - 1$ 做出的处理决策，进而依赖于 $X _ { i ( t - 1 ) }$，以此类推，直到我们回到初始状态 $X _ { i 1 }$。幸运的是，这些统计对象可以通过递归的、**动态规划（dynamic-programming）**风格的方法进行易于处理的分析。

## 14.1 顺序无混杂性（Sequential unconfoundedness）

为了估计上述定义的量，我们需要收集数据，并对实验中的处理分配方式做出假设，以识别这些估计量。在此，我们将使用**顺序无混杂性（sequential unconfoundedness）**（或**顺序可忽略性（sequential ignorability）**）来做到这一点，该假设认为，在每个时间点，给定当时观测到的数据，处理如同随机分配一样：

$$
\left\{\text {(时间 t 之后的潜在结果)} \right\} \perp W _ {i t} \mid \left\{\text {(截至时间 t 的历史)} \right\}. \tag {14.2}
$$

该条件在下面被形式化。在此，以及本章其余部分，我们将使用符号简写 $X _ { i ( T + 1 ) } : = Y _ { i }$（即，结果是在我们跨越时间跨度 T 之后测量的状态变量），以简化表达式。

**假设 14.1.** 给定一个动态决策过程，我们进一步假设我们的处理序列是**顺序无混杂的（sequentially unconfounded）**，使得对于所有 $t = 1, \ldots, T$，

$$
\left[ \left\{X _ {i (t + 1)} (W _ {i (1: (t - 1))}, w) \right\} _ {w = 0, 1} \perp W _ {i t} \right] \mid \left\{X _ {i 1}, W _ {i 1}, \dots W _ {i (t - 1)}, X _ {i t} \right\}. \tag {14.3}
$$

**注 14.1.** 原则上，人们也可能对一种更直接地与标准**随机对照试验（randomized controlled trial）**可比的设计感兴趣，其中处理是完全随机化的，

$$
\left\{\text {(所有潜在结果)} \right\} \perp W _ {1: T}. \tag {14.4}
$$

然而，这同样可能导致无意义的处理分配（例如，在癌症试验的情况下，在患者已经进入缓解期后仍将其分配到化疗组），因此关于动态处理规则的文献主要关注在更灵活的顺序无混杂性设置下有效的方法。

顺序无混杂性的统计后果，最容易通过策略 $\pi$ 下 $( X _ { i 1 } , \dots , X _ { i T } , X _ { i ( T + 1 ) } )$ 联合分布的**顺序分解（sequential factorization）**的性质来表达，其中如上所述，我们写 $X _ { i ( T + 1 ) } = Y _ { i }$。通常，我们写 $\mathbb { E } \left[ \cdot \right]$ 和 $\mathbb { P } \left[ \cdot \right]$ 来表示我们从中收集数据分布的期望和概率。我们总是可以将这个分布顺序分解为

$$
\mathbb {P} \left[ X _ {1}, W _ {1}, \dots , W _ {T}, X _ {T + 1} \right] = \mathbb {P} \left[ X _ {1} \right] \prod_ {t = 1} ^ {T} \mathbb {P} \left[ W _ {t} \mid S _ {t} \right] \mathbb {P} \left[ X _ {t + 1} \mid W _ {t}, S _ {t} \right], \tag {14.5}
$$

其中 $S _ { t } = \{ X _ { 1 } , W _ { 1 } , \ldots , W _ { t - 1 } , X _ { t } \}$ 表示直到第 t 期处理被选择之前的所有信息。为了策略评估的目的，引入**离策略（off-policy）**度量 $\mathbb { E } _ { \pi } \left[ \cdot \right]$ 和 $\mathbb { P } _ { \pi } \left[ \cdot \right]$ 也很方便，用于描述根据定义 14.2 中的 $\pi$ 分配处理将产生的分布。给定这个符号，我们可以简洁地将策略价值写为 $V ( \pi ) = \mathbb { E } _ { \pi } \left[ X _ { T + 1 } \right]$。我们也可以再次顺序分解该分布

$$
\mathbb {P} _ {\pi} \left[ X _ {1}, W _ {1}, \ldots , W _ {T}, X _ {T + 1} \right]
$$

$$
= \mathbb {P} _ {\pi} \left[ X _ {1} \right] \prod_ {t = 1} ^ {T} \mathbb {P} _ {\pi} \left[ W _ {t} \mid S _ {t} \right] \mathbb {P} _ {\pi} \left[ X _ {t + 1} \mid W _ {t}, S _ {t} \right]. \tag {14.6}
$$

顺序无混杂性的一个关键含义是，它允许我们通过保证分解中的某些项不依赖于感兴趣的策略 $\pi$ 来简化 (14.6)。下面的结果直接从 (14.3) 得出。

**表 14.1: 一个来自 Hernán and Robins [2020, Table 20.1] 的合成两期示例。**

| n     | $X_{i1}$ | $W_{i1}$ | $X_{i2}$ | $W_{i2}$ | Mean Y |
|-------|----------|----------|----------|----------|--------|
| 2400  | 0        | 0        | 0        | 0        | 84     |
| 1600  | 0        | 0        | 0        | 1        | 84     |
| 2400  | 0        | 0        | 1        | 0        | 52     |
| 9600  | 0        | 0        | 1        | 1        | 52     |
| 4800  | 0        | 1        | 0        | 0        | 76     |
| 3200  | 0        | 1        | 0        | 1        | 76     |
| 1600  | 0        | 1        | 1        | 0        | 44     |
| 6400  | 0        | 1        | 1        | 1        | 44     |

**命题 14.1.** 在顺序无混杂性下，分解中不涉及对 $W_t$ 进行积分的项不依赖于策略 $\pi$，即

$$
\mathbb {P} _ {\pi} \left[ X _ {1} \right] = \mathbb {P} \left[ X _ {1} \right] \quad \mathbb {P} _ {\pi} \left[ X _ {t + 1} \mid S _ {t}, W _ {t} \right] = \mathbb {P} \left[ X _ {t + 1} \mid S _ {t}, W _ {t} \right]. \tag {14.7}
$$

**处理-混杂因子反馈（Treatment-confounder feedback）** 在介绍在顺序无混杂性下有效的方法之前，有必要强调一下在此设置中出现的一个微妙困难，这在基本（单期）设计中是不存在的，即**处理-混杂因子反馈（treatment-confounder feedback）** [Robins, 1986]。为了看清可能出错的地方，考虑以下改编自 Hernán and Robins [2020] 的简单示例，该示例模拟了一个 $T = 2$ 时间段的 ART 试验。这里，$X _ { i t } \in \{ 0 , 1 \}$ 表示 CD4 计数（1 表示低，即糟糕），并假设每个人 $X _ { i 1 } = 0$（没有人进入试验时病得很重），并且 $X _ { i 1 }$ 被随机化，有 0.5 的概率接受治疗。然后，在时间段 2，我们观测 $X _ { i 2 }$，并根据 $X _ { i 2 }$ 分配处理：如果 $X _ { i 2 } = 0$，则以 0.4 的概率分配 $W _ { i 2 } = 1$；如果 $X _ { i 2 } = 1$，则以 0.8 的概率分配 $W _ { i 2 } = 1$。最后，我们收集健康结果 Y。这是一个**顺序随机实验（sequential randomized experiment）**。

我们观测到如表 14.1 所示的数据，其中最后一列是该行中所有个体的平均结果。我们的目标是估计 $\tau = \mathbb { E } \left[ Y \left( \underline{1} \right) - Y \left( \underline{0} \right) \right]$，即**始终治疗（always treat）**和**从不治疗（never treat）**规则之间的差异。我们应该如何做到这一点？作为初步说明，注意到处理显然没有效果是有帮助的。在第一个时间段，

$$
\mathbb {E} \left[ Y _ {i} \mid W _ {i 1} = 0 \right] = \mathbb {E} \left[ Y _ {i} \mid W _ {i 1} = 1 \right] = 6 0,
$$

这显然是一个因果量（因为 $W _ { i 1 }$ 是随机化的）。此外，在第二个时间段，我们通过检查发现

$$
\mathbb {E} \left[ Y _ {i} \mid W _ {i 2} = 0, W _ {i 1} = w _ {1}, X _ {i 2} = x \right] = \mathbb {E} \left[ Y _ {i} \mid W _ {i 2} = 1, W _ {i 1} = w _ {1}, X _ {i 2} = x \right],
$$

**表 14.2: 表 14.1 设置中的响应者类型。**

|        | $W_{i1}=0$ | $W_{i1}=1$ |
|--------|------------|------------|
| 稳定（stable）   | $X_{i2}=0$ | $X_{i2}=0$ |
| 响应者（responder） | $X_{i2}=1$ | $X_{i2}=0$ |
| 急性（acute）    | $X_{i2}=1$ | $X_{i2}=1$ |

对于所有 $w _ { 1 }$ 和 $x$ 的值，处理依然没有效果。

然而，当目标是评估**始终处理**与**从不处理**的总效应时，一些在非动态设定中表现良好的简单估计策略无法得到正确结果。具体而言，以下是一些无法得到正确结果的策略：

• 忽略自适应采样，直接使用：

$$
\begin{array}{l} \hat {\tau} = \widehat {\mathbb {E}} [ Y | W = \underline {{1}} ] - \widehat {\mathbb {E}} [ Y | W = \underline {{0}} ] \\ = \frac {6 4 0 0 \times 4 4 + 3 2 0 0 \times 7 6}{6 4 0 0 + 3 2 0 0} - \frac {2 4 0 0 \times 5 2 + 2 4 0 0 \times 8 4}{2 4 0 0 + 2 4 0 0} \\ = 5 4. 7 - 6 8 = - 1 3. 3. \\ \end{array}
$$

• 按时间点 2 的 CD4 计数分层，以控制自适应采样：

$$
\hat {\tau} _ {0} = \mathbb {E} \left[ Y \mid W = \underline {{1}}, X _ {i 2} = 0 \right] - \mathbb {E} \left[ Y \mid W = \underline {{0}}, X _ {i 2} = 0 \right] = 7 6 - 8 4 = - 8
$$

$$
\hat {\tau} _ {1} = \mathbb {E} \left[ Y \mid W = \underline {{1}}, X _ {i 2} = 1 \right] - \mathbb {E} \left[ Y \mid W = \underline {{0}}, X _ {i 2} = 1 \right] = 4 4 - 5 2 = - 8
$$

$$
\hat {\tau} = \frac {(3 2 0 0 + 2 4 0 0) \hat {\tau} _ {0} + (6 4 0 0 + 2 4 0 0) \hat {\tau} _ {1}}{3 2 0 0 + 2 4 0 0 + 6 4 0 0 + 2 4 0 0} = - 8.
$$

第一种策略的问题显而易见（我们需要校正有偏采样）。但第二种策略的问题则更为微妙。根据**序贯随机化（sequential randomization）**，我们知道：

$$
Y _ {i} (\dots) \perp W _ {i 2} \mid X _ {i 2},
$$

这似乎为分层提供了理由。然而，我们实际需要用于分层的条件是：

$$
Y _ {i} (\dots) \perp (W _ {i 1}, W _ {i 2}) \mid X _ {i 2},
$$

而这在设计上并不成立。

为了理解可能出错的地方，假设存在 3 类人群（稳定型、应答型、急性型），并将他们在时间点 2 的 CD4 值列于表 14.2。这些类型——通常称为**主分层（principal strata）**——是不可观测的，但仍然可以提供洞见。71 例如：

• $\mathbb{E} \left[ Y \mid W = \underline{{1}}, X_{i2} = 0 \right]$ 是稳定型或应答型患者的平均值，而 $\mathbb{E} \left[ Y \mid W = \underline{{0}}, X_{i2} = 0 \right]$ 则仅仅是稳定型患者的平均值。因此，差值 $\hat{\tau}_{0}$ 并非在估计一个恰当的因果量。

• $\mathbb{E} \left[ Y \mid W = \underline{{1}}, X_{i2} = 1 \right]$ 是急性型患者的平均值，相比之下，$\mathbb{E} \left[ Y \mid W = \underline{{0}}, X_{i2} = 1 \right]$ 则是应答型或急性型患者的平均值。因此，差值 $\hat{\tau}_{1}$ 并非在估计一个恰当的因果量。

换句话说，在序贯随机化试验中，简单的分层估计量无法成功控制混杂。

**序贯逆倾向加权（Sequential inverse-propensity weighting）**
由于分层方法无效，我们现在转而研究一系列有效的方法。在此，我们专注于估计策略 $V(\pi)$ 的价值，如 (14.1) 所示；注意，评估一个固定的处理序列是该策略的一个特例。为此，定义一些额外的符号是有帮助的：像之前一样，用 $S_t$ 表示时间 $t$ 时可得的信息，我们定义**价值函数（value function）** 72：

$$
V_{\pi, t}(S_t) = \mathbb{E}_{\pi}[Y | S_t] \tag{14.8}
$$

该函数衡量了在当前状态 $S_t$ 下开始遵循策略 $\pi$ 所能获得的期望回报。

这个符号让我们能够简洁地表达一个有助于有效估计 $V(\pi)$ 的原则：根据链式法则，我们看到：

$$
\begin{array}{l} \begin{array}{r l} \mathbb{E}_{\pi} \left[ V_{\pi, t + 1}(S_{t + 1}) \mid S_t \right] & = \mathbb{E}_{\pi} \left[ \mathbb{E}_{\pi} \left[ Y \mid S_{t + 1} \right] \mid S_t \right] \\ & \quad \mathbb{E}_{\pi} \left[ Y \mid S_t \right] - \mathbb{E}_{\pi}(S) \end{array} \tag{14.9} \\ = \mathbb{E}_{\pi} \left[ Y \mid S_t \right] = V_{\pi, t}(S_t). \\ \end{array}
$$

这意味着，如果我们有一个 $V_{\pi, t+1}$ 的良好估计，那么我们需要做的就是获得 $V_{\pi, t}$ 的良好估计；然后我们可以递归地向后计算到 $V(\pi)$。问题在于我们如何利用这一洞见。

一种简单的方法是通过**逆倾向加权（Inverse-Propensity Weighting, IPW）** 构造。如果我们能获得 $V_{\pi, t+1}(S_{i(t+1)})$ 以及许多 $S_{it} = s_t$ 的样本，那么在 (14.3) 条件下应用第 2 章的基本 IPW 构造，会建议使用：

$$
\widehat{V}_{\pi, t}(s_t) = \frac{1}{|\{i : S_{it} = s_t\}|} \sum_{\{i: S_{it} = s_t\}} \frac{1(\{W_{it} = \pi(s_t)\})}{\mathbb{P}[W_{it} = \pi(s_t) | S_{it} = s_t]} V_{\pi, t+1}(S_{i(t+1)}).
$$

递归应用这一原则，得到了策略价值的 IPW 估计量：

$$
\widehat{V}_{IPW}(\pi) = \frac{1}{n} \sum_{i=1}^{n} \gamma_{iT}(\pi) Y_i, \tag{14.10}
$$

$$
\gamma_{it}(\pi) = \gamma_{i(t-1)}(\pi) \frac{1(\{W_t = \pi_t(S_t)\})}{\mathbb{P}[W_t = \pi_t(S_t) \mid S_t]},
$$

其中 $\gamma_{i0}(\pi) = 1$。该估计量对处理轨迹完全匹配 $\pi$ 的结果取平均，同时应用 IPW 校正因已测量的（时变）混杂因素导致的选择效应。我们在下面证明，如果我们精确知道逆倾向权重 $\gamma_{iT}$，则 IPW 估计量是无偏的，并给出其渐近方差的表达式。

**定理 14.2。** 考虑如定义 14.1 所示的动态决策过程，其数据是在假设 14.1 的序贯无混杂性（sequential unconfoundedness）下收集的。进一步假设我们试图评估一个满足**强重叠性（strong overlap）** 的策略 $\pi$，即：

$$
\mathbb{P}[W_t = \pi_t(S_t) | S_t] \geq_{a.s.} \eta, \tag{14.11}
$$

并且我们的结果几乎必然有界，即 $|Y| \le_{a.s.} M$，其中 $M < \infty$。那么，来自 (14.10) 的 IPW 估计量是无偏的，并且其抽样分布是渐近正态的：73

$$
\begin{array}{l} \mathbb{E}[\widehat{V}_{IPW}(\pi)] = V(\pi), \quad \sqrt{n}\left(\widehat{V}_{IPW}(\pi) - V(\pi)\right) \Rightarrow \mathcal{N}\left(0, \sigma_{IPW}^2\right) \\ \sigma_{IPW}^2 = \mathbb{E}_{\pi}\left[ Y^2 / \prod_{t=1}^{T} \mathbb{P}[W_t = \pi_t(S_t) \mid S_t] \right] - V^2(\pi). \tag{14.12} \\ \end{array}
$$

**证明。** 我们通过反向归纳法验证无偏性，从 $t=T$ 开始，并论证对于所有 $t=0, \ldots, T$，有：

$$
V_{\pi, t}(S_t) = \mathbb{E}\left[ \frac{\gamma_T(\pi)}{\gamma_{t-1}(\pi)} Y \mid S_t \right] \tag{14.13}
$$

其中我们使用 $S_0 = \emptyset$ 以及 $\gamma_{-1}(\pi) = \gamma_0(\pi) = 1$。基例 $t=T$ 恰好对应定理 2.2 中的无偏性结果，而最终步骤 $t=0$ 对应于我们想要的结论。对于归纳步骤，假设 (14.13) 对 $t+1$ 成立。那么，我们可以验证：

$$
\mathbb{E}\left[ \frac{\gamma_T(\pi)}{\gamma_{t-1}(\pi)} Y \mid S_t \right] = \mathbb{E}\left[ \frac{\gamma_t(\pi)}{\gamma_{t-1}(\pi)} \mathbb{E}\left[ \frac{\gamma_T(\pi)}{\gamma_t(\pi)} Y \mid S_{t+1} \right] \mid S_t \right]
$$

$$
= \mathbb{E}\left[ \frac{1(\{W_t = \pi_t(S_t)\})}{\mathbb{P}[W_t = \pi_t(S_t) \mid S_t]} V_{\pi, t+1}(S_{t+1}) \right]
$$

$$
= \mathbb{E}\left[ \frac{1(\{W_t = \pi_t(S_t)\})}{\mathbb{P}[W_t = \pi_t(S_t) \mid S_t]} \mathbb{E}_{\pi}[Y_T \mid S_{t+1}] \right]
$$

$$
= \mathbb{E}_{\pi}\left[ \mathbb{E}_{\pi}[Y_T \mid S_t] \right] = V_{\pi, t}(S_t),
$$

其中第一个等式成立是因为 $\gamma_t(\pi) / \gamma_{t-1}(\pi)$ 是 $S_t$ 可测的，第二个等式通过调用归纳假设和 $\gamma_t(\pi) / \gamma_{t-1}(\pi)$ 的定义得到，第四个等式由序贯无混杂性得到，第三个和最后一个等式只是 (14.9)。

给定无偏性和单元的独立同分布采样，中心极限定理随之成立，其中：

$$
\sigma_{IPW}^2 = \mathbb{E}[\gamma_T^2(\pi) Y^2] - V^2(\pi),
$$

剩下的工作只是推导出上述二阶矩项的显式表达式。现在，通过重复上面使用的 IPW 论证：

$$
\mathbb{E}[\gamma_T^2(\pi) Y^2] = \mathbb{E}_{\pi}[\gamma_T(\pi) Y^2].
$$

在离策略测度 $\mathbb{E}_{\pi}[\cdot]$ 下，我们总是有 $W_t = \pi_t(S_t)$，因此：

$$
\gamma_T(\pi) = 1 \Big/ \prod_{t=1}^{T} \mathbb{P}[W_t = \pi_t(S_t) \mid S_t]
$$

几乎必然成立，从而提供了所声称的表达式。

**注 14.2。** 如第 12 章所讨论的，我们通常可以通过**自归一化（self-normalization）** 来提高 IPW 的渐近精度：

$$
\widehat{V}_{SIPW}(\pi) = \sum_{i=1}^{n} \gamma_{iT}(\pi) Y_i / \sum_{i=1}^{n} \gamma_{iT}(\pi). \tag{14.14}
$$

在定理 14.2 的条件下：

$$
\sqrt{n}\left(\widehat{V}_{SIPW}(\pi) - V(\pi)\right) \Rightarrow \mathcal{N}\left(0, \sigma_{SIPW}^2\right)
$$

$$
\sigma_{SIPW}^2 = \mathbb{E}_{\pi}\left[ (Y - V(\pi))^2 / \prod_{t=1}^{T} \mathbb{P}[W_t = \pi_t(S_t) \mid S_t] \right]. \tag{14.15}
$$

该结果可以通过遵循例如定理 12.3 中相同的证明策略来建立。自归一化带来的精度变化为：

$$
\begin{array}{l} \sigma_{IPW}^2 - \sigma_{SIPW}^2 = \left(\mathbb{E}_{\pi}\left[ \left(\prod_{t=1}^{T} \mathbb{P}[W_t = \pi_t(S_t) \mid S_t]\right)^{-1} \right] - 1\right) V^2(\pi) \tag{14.16} \\ + 2 \operatorname{Cov}_{\pi}\left[ Y, \left(\prod_{t=1}^{T} \mathbb{P}[W_t = \pi_t(S_t) \mid S_t]\right)^{-1} \right]. \\ \end{array}
$$

第一个求和项总是正的（并且通常很大）；然而，第二个求和项可能是负的——并且原则上可能负到足以使自归一化 IPW 的精度低于基本 IPW 估计量。

## 14.2 双稳健估计（Doubly robust estimation）

如同第3章讨论的单期情形一样，可以通过引入回归调整来提升**逆概率加权（Inverse Probability Weighting, IPW）**的精度和稳健性。在此，我们展示如何为动态处理规则构建增广估计量，并验证所得估计量具有强双稳健性质：它可以在回归模型和**倾向性评分（propensity score）**模型的准确性之间进行权衡，并且即使输入的非参数回归以较慢速度收敛，也能达到参数化的 $1 / \sqrt { n }$ 收敛速率。

**反向回归调整** 与第3章类似，我们的双稳健构造始于利用**序贯无混淆性（sequential unconfoundedness）**来推导另一种基于回归的策略价值估计方法。结合序贯无混淆性（特别是命题14.1中强调的推论）与 (14.9)，我们得到：

$$
V _ {\pi , t} (s) = \mathbb {E} \left[ V _ {\pi , t + 1} (S _ {t + 1})   \big |   S _ {t} = s,   W _ {t} = \pi_ {t} (s) \right]. \tag {14.17}
$$

因此，如果我们知道 $V _ { \pi , t + 1 } ( \cdot )$ 或对其有足够精确的估计，就可以通过以 $V _ { \pi , t + 1 } ( \cdot )$ 为结果变量的非参数回归来估计 $V _ { \pi , t } ( \cdot )$。

这一结构提示了以下估计策略价值的反向回归方法：

*   首先，使用严格遵循目标策略的样本 $i$，即对于所有 $t = 1 , \dots , T$ 都有 $W _ { i t } = \pi ( S _ { i t } )$，通过非参数回归 $Y _ { i } \sim V _ { \pi , T } ( S _ { i T } )$ 学习 $\widehat { V } _ { \pi , T } ( \cdot )$。
*   接着，迭代地对于 $t = T - 1 , T - 2 , . . . , 1$：
    – 使用在时间 $t$ 之前严格遵循目标策略的样本 $i$，即对于所有 $t ^ { \prime } = 1 , \ldots , t$ 都有 $W _ { i t ^ { \prime } } = \pi ( S _ { i t ^ { \prime } } )$，通过非参数回归 $\hat { V } _ { \pi , t + 1 } ( S _ { i ( t + 1 ) } ) \sim V _ { \pi , t } ( S _ { i t } )$ 学习 $\widehat { V } _ { \pi , t } ( \cdot )$。
*   最后，形成策略 $\pi$ 价值的回归估计量：

$$
\widehat {V} _ {R E G} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \widehat {V} _ {\pi , 1} (S _ {i 1}). \tag {14.18}
$$

这种反向回归方法可以通过通用机器学习来实现。然而，定制模型也可能有所帮助；例如，**结构嵌套均值模型（Structural Nested Mean Models）**[Robins, 1994] 旨在避免在干预无效的原假设下虚假地检测到因果效应。

**回归增广估计量** 当存在基于 IPW 和基于回归的估计量时，通常也会存在一个双稳健估计量。在反向回归估计量 (14.17) 的最后一步，我们平均了时间 1 的价值函数估计值 $\widehat { V } _ { \pi , 1 } ( \bar { X } _ { 1 } )$ 以获得 $\widehat { V } _ { R E G } ( \pi )$。现在，基于反向回归的构造，我们可能对时间 2 的价值函数估计值 $\widehat { V } _ { \pi , 2 }$ 的信任程度略高于时间 1 的估计值；在这种情况下，我们可以考虑使用第 3 章中的基本增广 IPW（Augmented IPW, AIPW）构造来利用这些 $\widehat { V } _ { \pi , 2 }$ 估计值以提高精度：

$$
\widehat {V} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\widehat {V} _ {\pi , 1} (X _ {i 1}) + \gamma_ {i 1} (\pi) \left(\widehat {V} _ {\pi , 2} (X _ {i 1}, W _ {i 1}, X _ {i 2}) - \widehat {V} _ {\pi , 1} (X _ {i 1})\right)\right).
$$

定性地来说，这里的思路是：在 $W _ { i 1 }$ 与第一步中的 $\pi$ 匹配的事件上，我们可以使用 $\widehat { V } _ { \pi , 2 }$ 来对 $\widehat { V } _ { \pi , 1 }$ 进行去偏；这里，$\gamma _ { i t }$ 是如 (14.10) 中的逆概率权重。

那么，下一个自然的问题当然是，当 $W _ { i 2 }$ 在第二步中也与 $\pi$ 匹配时，为什么不使用 $\widehat { V } _ { \pi , 3 }$ 来对 $\widehat { V } _ { \pi , 2 }$ 进行去偏？一旦我们这样做了，为什么不一直进行到时间终点，直到我们可以观察到实现的结果 $Y$？这种递归构造实际上是可行的，并且将第 3 章中讨论的 Robins, Rotnitzky, and Zhao [1994] 的 AIPW 估计量自然地推广到了动态设定：

$$
\begin{array}{l} \widehat {V} _ {A I P W} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\widehat {V} _ {\pi , 1} (X _ {i 1}) \right. \tag {14.19} \\ \left. + \sum_ {t = 1} ^ {T} \hat {\gamma} _ {i t} (\pi) \left(\widehat {V} _ {\pi , t + 1} (S _ {i (t + 1)}) - \widehat {V} _ {\pi , t} (S _ {i t})\right)\right), \\ \end{array}
$$

这里我们采用了一个符号约定，即 $\widehat { V } _ { \pi , T + 1 } ( S _ { i ( T + 1 ) } ) = Y _ { i }$，因为在时间 $T + 1$ 时最终结果已经揭示。

下面，我们在**双机器学习（Double Machine Learning）**框架下分析该估计量的大样本性质，并看到它保留了第 3 章中讨论的强双稳健性质：如果 $\hat { \gamma } _ { t } ( \pi )$ 模型和 $\widehat { V } _ { \pi , t }$ 的**均方误差（Mean-Squared Error, MSE）**的乘积衰减得足够快，则该估计量具有良好的性质。为简化起见，我们假设这些** nuisance 组件（nuisance components）** 的估计量是使用独立的训练数据获得的；然而，与第 3 章一样，该论证可以立即推广到 K 折交叉拟合，代价是增加一些额外的符号。

**定理 14.3.** 在定理 $\it { 1 4 . 2 }$ 的条件下，进一步假设我们在独立的训练数据上估计 (14.19) 中的 nuisance 组件，对于 $t = 1 , \ldots , T , ^{74}$

$$
\mathbb {E} \left[ (\hat {\gamma} _ {i t} (\pi) - \gamma_ {i t} (\pi)) ^ {2} \right] = o _ {P} \left(n ^ {- 2 \alpha_ {\gamma}}\right),
$$

$$
\mathbb {E} \left[ \left(\widehat {V} _ {\pi , t} (S _ {i t}) - V _ {\pi , t} (S _ {i t})\right) ^ {2} \right] = o _ {P} \left(n ^ {- 2 \alpha_ {V}}\right) \tag {14.20}
$$

对于常数 $\alpha _ { \gamma } , \alpha _ { V } \geq 0$ 且 $\alpha _ { \gamma } + \alpha _ { V } \ge 1 / 2$。那么，

$$
\sqrt {n} \left(\widehat {V} _ {A I P W} (\pi) - V (\pi)\right) \Rightarrow \mathcal {N} \left(0, \sigma_ {A I P W} ^ {2}\right)
$$

$$
\sigma_ {A I P W} ^ {2} = \operatorname{Var} \left[ \mathbb {E} _ {\pi} [ Y | X _ {1} ] \right] \tag {14.21}
$$

$$
+ \sum_ {t = 1} ^ {T} \mathbb {E} _ {\pi} \left[ \operatorname{Var} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y \mid S _ {t + 1} \right] \mid S _ {t} \right] \Big / \prod_ {t ^ {\prime} = 1} ^ {t} \mathbb {P} \left[ W _ {t ^ {\prime}} = \pi_ {t ^ {\prime}} (S _ {t ^ {\prime}}) \mid S _ {t ^ {\prime}} \right] \right].
$$

**证明.** 与第 3 章中单时间步 AIPW 结果的证明类似，我们首先考虑具有正确 nuisance 估计的** oracle 估计量（oracle estimator）** 的性质，然后在收敛速率假设和外部 nuisance 估计量的条件下，证明可行的 AIPW 估计量与 oracle AIPW 估计量的渐近等价性。在我们的设定中，oracle 为：

$$
\widehat {V} _ {A I P W} ^ {*} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(V _ {\pi , 1} (X _ {i 1}) \right. \tag {14.22}
$$

$$
\left. + \sum_ {t = 1} ^ {T} \gamma_ {i t} (\pi) \left(V _ {\pi , t + 1} (S _ {i (t + 1)}) - V _ {\pi , t} (S _ {i t})\right)\right),
$$

其中 $V _ { \pi , t } ( S _ { i ( T + 1 ) } ) = Y _ { i }$。现在，由 (14.9) 我们知道 $\mathbb { E } _ { \boldsymbol { \pi } } \left[ V _ { \boldsymbol { \pi } , t + 1 } ( S _ { i ( t + 1 ) } ) \big | \ S _ { i t } \right] = V _ { \pi , t } ( S _ { i t } )$。根据序贯无混淆性（特别是命题 14.1 中强调的性质），这意味着在数据收集测度下，

$$
\mathbb {E} \left[ V _ {\pi , t + 1} (S _ {i (t + 1)})   |   S _ {i t},   W _ {i t} = \pi (S _ {i t}) \right] = V _ {\pi , t} (S _ {i t}). \tag {14.23}
$$

此外，回顾 $\gamma _ { i t } ( \pi )$ 是 $S _ { i t }$ 和 $W _ { i t }$ 的函数，并且仅当 $W _ { i t } = \pi ( S _ { i t } )$ 时 $\gamma _ { i t } ( \pi ) \neq 0$，我们看到

$$
\mathbb {E} \left[ \gamma_ {i t} (\pi) \left(V _ {\pi , t + 1} (S _ {i (t + 1)}) - V _ {\pi , t} (S _ {i t})\right) \big | S _ {i t} \right] = 0, \tag {14.24}
$$

即，对于给定的个体 $i$，项 $\gamma _ { i t } ( \pi ) \left( V _ { \pi , t + 1 } ( S _ { i ( t + 1 ) } ) - V _ { \pi , t } ( S _ { i t } ) \right)$ 形成了一个**鞅差序列（martingale difference sequence）**。因此

$$
\begin{array}{l} \mathrm{Var} \left[ V _ {\pi , 1} (X _ {i 1}) + \sum_ {t = 1} ^ {T} \gamma_ {i t} (\pi) \left(V _ {\pi , t + 1} (S _ {i (t + 1)}) - V _ {\pi , t} (S _ {i t})\right) \right] \\ = \operatorname{Var} \left[ V _ {\pi , 1} (X _ {i 1}) \right] + \sum_ {t = 1} ^ {T} \operatorname{Var} \left[ \gamma_ {i t} (\pi) \left(V _ {\pi , t + 1} (S _ {i (t + 1)}) - V _ {\pi , t} (S _ {i t})\right) \right]. \\ \end{array}
$$

通过像定理 14.2 证明中那样转换到 off-policy 测度，然后代入 (14.8) 中价值函数的定义，即可恢复 (14.21) 中的方差表达式。最后，给定个体 $i = 1 , \ldots , n$ 的**独立同分布（Independent and Identically Distributed, IID）**抽样以及我们的强重叠和有界性假设，对于 oracle 估计量 (14.22)，**中心极限定理（central limit theorem）** 14.21 立即成立。

现在，为了证明可行的 AIPW 估计量与 oracle AIPW 估计量的渐近等价性，我们引入一些方便的简写。我们将时间 $t$ 的价值函数更新写为

$$
\varepsilon_ {i t} := V _ {\pi , t + 1} (S _ {i (t + 1)}) - V _ {\pi , t} (S _ {i t})
$$

对于 $t = 0 , \ldots , T$，并将价值函数误差写为

$$
\hat {\delta} _ {i t} = \widehat {V} _ {\pi , t} (S _ {i t}) - V _ {\pi , t} (S _ {i t})
$$

对于 $t = 1 , \dots , T$。我们还删除了 $\gamma _ { i t } ( \pi )$ 中对 $\pi$ 的显式依赖。给定这个符号，我们有

$$
\widehat {V} _ {A I P W} ^ {*} (\pi) - V (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \gamma_ {i t} \varepsilon_ {i t}
$$

$$
\widehat {V} _ {A I P W} (\pi) - V (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \hat {\gamma} _ {i t} \left(\varepsilon_ {i t} + \hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right),
$$

其中我们有 $\hat { \delta } _ { i 0 } = 0$（因为 $\widehat { V } _ { 0 , \pi }$ 不出现在 $\widehat { V } _ { A I P W } ( \pi )$ 的构造中，所以不失一般性，我们假设那里没有误差）且 $\hat { \delta } _ { i ( T + 1 ) } = 0$（因为 $\widehat { V } _ { \pi , T + 1 } ( S _ { i ( T + 1 ) } ) = Y _ { i } )$。因此，

$$
\begin{array}{l} \widehat {V} _ {A I P W} (\pi) - \widehat {V} _ {A I P W} ^ {*} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \left(\widehat {\gamma} _ {i t} - \gamma_ {i t}\right) \varepsilon_ {i t} \\ + \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \gamma_ {i t} \left(\hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right) + \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \left(\hat {\gamma} _ {i t} - \gamma_ {i t}\right) \left(\hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right). \\ \end{array}
$$

我们现在像定理 3.2 的证明中那样分别界定每一项。第一项在 $t$ 上是一个鞅，理由与上面使用的相同，因此根据个体的 IID 抽样

$$
\begin{array}{l} \mathrm{Var} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \left(\hat {\gamma} _ {i t} - \gamma_ {i t}\right) \varepsilon_ {i t} \right] = \frac {1}{n} \sum_ {t = 1} ^ {T} \mathbb {E} \left[ \left(\hat {\gamma} _ {i t} - \gamma_ {i t}\right) ^ {2} \mathrm{Var} _ {\pi} \left[ \varepsilon_ {i t} \mid S _ {i t} \right] \right] \\ = \mathcal {O} \left(\frac {1}{n} \sum_ {t = 1} ^ {T} \mathbb {E} \left[ (\hat {\gamma} _ {i t} - \gamma_ {i t}) ^ {2} \right]\right), \\ \end{array}
$$

$\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { t = 0 } ^ { T } \left( \widehat { \gamma } _ { i t } - \gamma _ { i t } \right) \varepsilon _ { i t } = o _ { p } \left( 1 / \sqrt { n } \right) } \end{array}$ 我们可以重新排列求和项：

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \gamma_ {i t} \left(\hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\sum_ {t = 1} ^ {T} \left(\gamma_ {i (t - 1)} - \gamma_ {i t}\right) \hat {\delta} _ {i t} + \gamma_ {i T} \hat {\delta} _ {i (T + 1)} - \gamma_ {i 0} \hat {\delta} _ {i 0}\right),
$$

其中最后两项可以忽略，因为 $\hat { \delta } _ { i 0 } = \hat { \delta } _ { i ( T + 1 ) } = 0$。根据 $\gamma _ { i t }$ 和 $\hat { \delta } _ { i t }$ 的定义，该项可以进一步简化为

$$
\ldots = \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 1} ^ {T} \gamma_ {i (t - 1)} \left(1 - \frac {1 \left(\{W _ {i t} = \pi (S _ {i t}) \}\right)}{\mathbb {P} \left[ W _ {i t} = \pi (S _ {i t}) \mid S _ {i t} \right]}\right) \left(\widehat {V} _ {\pi , t} (S _ {i t}) - V _ {\pi , t} (S _ {i t})\right).
$$

根据序贯无混淆性，内层求和再次是 $t$ 上的一个鞅，所以

$$
\begin{array}{l} \mathbb {E} \left[ \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \gamma_ {i t} \left(\hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right)\right) ^ {2} \right] \\ = \frac {1}{n} \sum_ {t = 1} ^ {T} \mathbb {E} \left[ \gamma_ {i (t - 1)} ^ {2} \frac {1 - \mathbb {P} \left[ W _ {i t} = \pi \left(S _ {i t}\right) \mid S _ {i t} \right]}{\mathbb {P} \left[ W _ {i t} = \pi \left(S _ {i t}\right) \mid S _ {i t} \right]} \left(\widehat {V} _ {\pi , t} \left(S _ {i t}\right) - V _ {\pi , t} \left(S _ {i t}\right)\right) ^ {2} \right] \\ = \frac {1}{n} \sum_ {t = 1} ^ {T} \eta^ {1 - 2 t} \mathbb {E} \left[ \left(\widehat {V} _ {\pi , t} (S _ {i t}) - V _ {\pi , t} (S _ {i t})\right) ^ {2} \right] = o _ {p} (1 / n) \\ \end{array}
$$

由 (14.20) 可得，该项本身也是 $o _ { p } ( 1 / \sqrt { n } )$。最后，对于第三项，我们可以交换求和顺序并应用**柯西-施瓦茨不等式（Cauchy-Schwarz inequality）**：

$$
\begin{array}{l} \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \left(\hat {\gamma} _ {i t} - \gamma_ {i t}\right) \left(\hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right) \\ \leq \sum_ {t = 0} ^ {T} \sqrt {\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\gamma} _ {i t} - \gamma_ {i t}\right) ^ {2}} \sqrt {\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right) ^ {2}} = o _ {P} \left(n ^ {- (\alpha_ {\gamma} + \alpha_ {V})}\right). \\ \end{array}
$$

这证明了

$$
\widehat {V} _ {A I P W} (\pi) - \widehat {V} _ {A I P W} ^ {*} (\pi) = o _ {P} \left(1 / \sqrt {n}\right),
$$

从而完成了证明。

![image_11](images/image_11.png)

## 14.3 参考文献注释（Bibliographic notes）

本章介绍的评估动态决策规则（dynamic decision rules）的方法，即使用**嵌套潜在结果（nested potential outcomes）**并通过**序列无混淆性（sequential unconfoundedness）**进行识别，可追溯至 Robins [1986]；关于该系列工作的综述，可参见 Richardson 和 Rotnitzky [2014]，而教材级别的论述则见 Hernán 和 Robins [2020]。该系列工作中最广泛使用的算法之一称为**边际结构模型（marginal structural modeling）**，其通过**逆概率加权线性回归（inverse-propensity weighted linear regression）**来估计参数化策略类的价值 [详见 Robins, 1999 的概述]。**AIPW 估计量（AIPW estimator）** (14.19) 在 Jiang 和 Li [2016]、Thomas 和 Brunskill [2016] 以及 Zhang、Tsiatis、Laber 和 Davidian [2013] 中均有讨论。

动态环境下的因果推断（Causal inference in dynamic settings）是一个广泛的课题，全面讨论将超出本书的范围。Van der Laan 和 Robins [2003] 以及 Tsiatis [2006] 提供了全面的教材级论述，其中包含对**效率（efficiency）**的讨论。特别地，在许多应用中一个重要的考量是**删失（censoring）**问题：某些个体可能在我们观察到最终结果之前就离开了研究，本章讨论的方法需要进行扩展以处理此类删失（关于删失结果的一个示例，参见第 16 章练习 14）。另一个有趣的方向是将第 5 章关于**策略学习（policy learning）**的讨论扩展到动态环境 [Robins, 2004]。最后，我们关于动态策略评估的讨论与**强化学习（reinforcement learning）**领域的文献密切相关；教材级别的论述可参见 Sutton 和 Barto [2018]。
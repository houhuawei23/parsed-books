# 第3章 双重稳健方法（Doubly Robust Methods）

**逆倾向得分加权（Inverse-propensity weighting, IPW）**是在无混杂性假设下进行平均处理效应估计的一种简单而透明的方法。然而，如前一章所述，IPW的大样本性质通常并不特别好，且倾向得分中的估计误差影响IPW准确性的方式也很复杂。我们的目标是超越IPW的局限性，讨论**双重稳健方法（doubly robust methods）**，这些方法提供了一种通用策略，用于在无混杂性假设下构建稳健且渐近最优的处理效应估计量，并使我们能够严格且灵活地处理倾向得分中的估计误差。15

在本章中，我们将在以下统计设定下，估计平均处理效应 $\tau = \mathbb { E } \left[ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \right]$：

**基本设定：SUTVA、无混杂性和强重叠性**  
存在一个分布 $P$，生成一系列元组 $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , \bar { W } _ { i } \} \stackrel { \mathrm { i i d } } { \sim } P$，取值于 $\mathcal { X } \times \mathbb { R } \times \mathbb { R } \times \{ 0 , 1 \}$。我们观测到 $( X _ { i } , Y _ { i } , W _ { i } )$，其中 $Y _ { i } = Y _ { i } ( W _ { i } )$（SUTVA）。我们不一定处于随机对照试验中；然而，我们具有无混杂性，即处理分配在给定特征 $X _ { i }$ 的条件下如同随机：

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp W _ {i} \mid X _ {i}, \tag {3.1}
$$

潜在结果具有有界二阶矩，$\mathbb { E } \left[ Y _ { i } ^ { 2 } ( w ) \right] < \infty$。强重叠性成立，即对于某个 $\eta > 0$，

$$
\eta \leq e (x) \leq 1 - e (x) \quad \text { 对所有 } \quad x \in \mathcal {X} \text{ 成立}. \tag {3.2}
$$

我们用 $e ( x ) = \mathbb { P } \left[ W _ { i } = 1 \big | X _ { i } = x \right]$ 表示倾向得分，并使用记号 $\mu _ { ( w ) } ( x ) = \bar { \mathbb { E } } \left[ Y _ { i } ( w ) \big | X _ { i } = x \right]$ 和 $\sigma _ { ( w ) } ^ { 2 } ( x ) = \mathrm { V a r } \left[ Y _ { i } ( w ) \big | X _ { i } = x \right]$。

**ATE的两种刻画** 在上一章中，我们看到ATE可以通过IPW来刻画：

$$
\tau = \mathbb {E} \left[ \hat {\tau} _ {I P W} ^ {*} \right], \quad \hat {\tau} _ {I P W} ^ {*} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})}\right). \tag {3.3}
$$

然而，$\tau$ 也可以用条件响应曲面 $\mu _ { ( w ) } ( x )$ 来刻画：在无混杂性（3.1）下，

$$
\tau (x) := \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \mid X _ {i} = x \right]
$$

$$
= \mathbb {E} \left[ Y _ {i} (1) \mid X _ {i} = x \right] - \mathbb {E} \left[ Y _ {i} (0) \mid X _ {i} = x \right]
$$

$$
= \mathbb {E} \left[ Y _ {i} (1) \mid X _ {i} = x, W _ {i} = 1 \right] - \mathbb {E} \left[ Y _ {i} (0) \mid X _ {i} = x, W _ {i} = 0 \right] \quad (\text {无混杂性})
$$

$$
= \mathbb {E} \left[ Y _ {i} \mid X _ {i} = x, W _ {i} = 1 \right] - \mathbb {E} \left[ Y _ {i} \mid X _ {i} = x, W _ {i} = 0 \right] \quad (\text {SUTVA})
$$

$$
= \mu_ {(1)} (x) - \mu_ {(0)} (x),
$$

因此 $\tau = \mathbb { E } \left[ \mu _ { ( 1 ) } ( X _ { i } ) - \mu _ { ( 0 ) } ( X _ { i } ) \right]$。因此，也存在一个简单且一致（但不一定最优）的非参数回归估计量用于 $\tau$：首先非参数地估计 $\mu _ { ( 0 ) } ( x )$ 和 $\mu _ { ( 1 ) } ( x )$，然后设置 $\begin{array} { r } { \hat { \tau } _ { R E G } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } \bigl ( \hat { \mu } _ { ( 1 ) } ( X _ { i } ) - \hat { \mu } _ { ( 0 ) } ( X _ { i } ) \bigr ) } \end{array}$。

**增广IPW（Augmented IPW）** 鉴于平均处理效应可以通过两种不同方式估计，即先非参数地估计 $e ( x )$，或先估计 $\mu _ { ( 0 ) } ( x )$ 和 $\mu _ { ( 1 ) } ( x )$，自然会问是否可能将两种策略结合起来。事实证明这是一个非常好的想法，并产生了Robins、Rotnitzky和Zhao [1994]提出的**增广IPW（augmented IPW, AIPW）**估计量：

$$
\hat {\tau} _ {A I P W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\mu} _ {(1)} (X _ {i}) - \hat {\mu} _ {(0)} (X _ {i}) \right. \tag {3.4}
$$

$$
\left. + W _ {i} \frac {Y _ {i} - \hat {\mu} _ {(1)} (X _ {i})}{\hat {e} (X _ {i})} - (1 - W _ {i}) \frac {Y _ {i} - \hat {\mu} _ {(0)} (X _ {i})}{1 - \hat {e} (X _ {i})}\right).
$$

定性地说，AIPW可以看作首先通过估计 $\mu _ { ( 0 ) } ( x )$ 和 $\mu _ { ( 1 ) } ( x )$ 对 $\tau$ 做出最佳尝试；然后，通过对回归残差应用IPW来处理 $\hat { \mu } _ { ( w ) } ( x )$ 的任何偏差。从统计上看，AIPW不仅继承了回归估计量和IPW估计量的稳健性性质——它还通过（在下面严格意义上）使用IPW来减轻回归估计量的误差，反之亦然，从而改进了两者。

**弱双重稳健性（Weak double robustness）** AIPW第一个易于理解的属性是以下"弱"双重稳健性性质：16 如果 $\hat { \mu } _ { ( w ) } ( x )$ 一致或 $\hat { e } ( x )$ 一致，则AIPW一致。为了理解这一点，首先考虑 $\hat { \mu } _ { ( w ) } ( x )$ 一致的情况，即 $\hat { \mu } _ { ( w ) } ( x ) \approx \mu _ { ( w ) } ( x )$。那么，

$$
\begin{array}{l} \hat {\tau} _ {A I P W} = \underbrace {\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\mu} _ {(1)} (X _ {i}) - \hat {\mu} _ {(0)} (X _ {i})\right)} _ {\text {回归估计量}} \\ + \underbrace {\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i}}{\hat {e} (X _ {i})} \left(Y _ {i} - \hat {\mu} _ {(1)} (X _ {i})\right) - \frac {1 - W _ {i}}{1 - \hat {e} (X _ {i})} \left(Y _ {i} - \hat {\mu} _ {(0)} (X _ {i})\right)\right)} _ {\approx \text {均值零噪声}}, \\ \end{array}
$$

因为在无混杂性下，$\mathbb { E } \left[ Y _ { i } - \hat { \mu } _ { ( W _ { i } ) } ( X _ { i } ) \big | X _ { i } , W _ { i } \right] \approx 0$。因此，即使我们使用不一致的倾向得分权重 $1 / \hat { e } ( X _ { i } )$ 和 $1 / ( 1 - \hat { e } ( X _ { i } ) )$，它们乘以大致均值零的误差项，因此在渐近意义上不会使估计量产生偏差，$\hat { \tau } _ { A I P W }$ 保持一致性。

相反，现在假设 $\hat { e } ( x )$ 一致，即 $\hat { e } ( x ) \approx e ( x )$。那么，

$$
\begin{array}{l} \hat {\tau} _ {A I P W} = \underbrace {\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{\hat {e} (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})}\right)} _ {\text {IPW估计量}} \\ + \underbrace {\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\mu} _ {(1)} (X _ {i}) \left(1 - \frac {W _ {i}}{\hat {e} (X _ {i})}\right) - \hat {\mu} _ {(0)} (X _ {i}) \left(1 - \frac {1 - W _ {i}}{1 - \hat {e} (X _ {i})}\right)\right)} _ {\approx \text {均值零噪声}}, \\ \end{array}
$$

因为 $\mathbb{E}\left[ 1 - W _ { i } / \hat { e } ( X _ { i } ) \vert X _ { i } \right] \approx 0$。因此，即使我们使用不一致的回归调整项 $\hat { \mu } _ { ( w ) } ( X _ { i } )$，它们将被乘以大致均值零的噪声项，这些噪声项在渐近意义上抵消了它们的贡献。因此，在无混杂性下，$\hat { \tau } _ { A I P W }$ 继承了 $\hat { \tau } _ { I P W }$ 的一致性。

尽管如此，尽管AIPW的（弱）双重稳健性是一个很好的性质，但其重要性不应被夸大。弱双重稳健性仅保证了 $\hat { \tau } _ { A I P W }$ 的一致性，而在大多数处理效应估计应用中，我们还关心收敛速度和置信区间。此外，也可以论证，在现代环境中，我们应该期望从业者对 $\mu _ { ( w ) } ( x )$ 和 $e ( x )$ 都使用适当的非参数估计量，这些估计量对每个都是一致的。在这种情况下，$\hat { \tau } _ { R E G }$ 和 $\hat { \tau } _ { I P W }$ 本身已经一致，因此上述弱双重稳健性陈述（即 $\hat { \tau } _ { A I P W }$ 的一致性）并没有增加任何内容。

**强双重稳健性（Strong double robustness）** 对于AIPW，还有一类更有趣且更有用的"强"双重稳健性结果，它们量化了上述较弱的一致性陈述。在高层次上，强双重稳健性是指以下类型的结果存在：如果我们使用的估计量 $\hat { \mu } _ { ( w ) } ( x )$ 和 $\hat { e } ( x )$ 都是一致的，且其**均方根误差（root-mean squared error, RMSE）**分别以快于 $n ^ { - \alpha \mu }$ 和 $n ^ { - \alpha _ { e } }$ 的速度衰减，且进一步有 $\alpha _ { \mu } + \alpha _ { e } \ge 1 / 2$，那么

$$
\sqrt {n} \left(\hat {\tau} _ {A I P W} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {A I P W}\right),
$$

$$
V _ {A I P W} = \operatorname{Var} \left[ \tau (X _ {i}) \right] + \mathbb {E} \left[ \frac {\sigma_ {0} ^ {2} (X _ {i})}{1 - e (X _ {i})} \right] + \mathbb {E} \left[ \frac {\sigma_ {1} ^ {2} (X _ {i})}{e (X _ {i})} \right]. \tag {3.5}
$$

这个元结果成立的原因是，一般来说，如果 $\hat { \mu } _ { ( w ) } ( x )$ 的RMSE以快于 $n ^ { - \alpha _ { \mu } }$ 的速度衰减，且 $\hat { e } ( x )$ 的RMSE以快于 $n ^ { - \alpha _ { e } }$ 的速度衰减，那么AIPW的偏差以快于 $n ^ { - ( \alpha _ { \mu } + \alpha _ { e } ) }$ 的速度衰减；特别地，如果 $\alpha _ { \mu } + \alpha _ { e } \ge 1 / 2$，那么偏差在 $1 / \sqrt { n }$ 尺度上是低阶的。这个结果的显著之处在于，在相同条件下，回归估计量的偏差通常只能被限制在 $n ^ { - \alpha _ { \mu } }$ 阶，而IPW的偏差只能被限制在 $n ^ { - \alpha _ { e } }$ 阶；因此，AIPW构造成功地使偏差远小于回归估计量或IPW估计量单独能达到的水平。17

上述陈述不是一个定理——而是一个元结果，是在进一步技术假设下成立的许多类型结果的蓝图。下面，我们将讨论构建AIPW估计量的一种具体方式，由Chernozhukov等人[2018]称为**双重机器学习（double machine learning）**，并建立其满足（3.5）的条件。注意，双重机器学习并不是获得此类结果的唯一方式；事实上，在某些专门设定下，可以获得比（3.5）更强的结果。因此，我们下面的介绍应被视为理解和利用AIPW强双重稳健性的第一步——而非终点。

## 3.1 双重机器学习（Double machine learning）

我们对AIPW强双重稳健性的研究，首先考虑一个"**理想（oracle）**"AIPW估计量的行为，该估计量使用条件回归曲面和倾向得分的真实值（而非估计值）构建：

$$
\begin{array}{l} \hat {\tau} _ {A I P W} ^ {*} = \frac {1}{n} \sum_ {i = 1} ^ {n} \Gamma_ {i} \\ \Gamma_ {i} = \mu_ {(1)} (X _ {i}) - \mu_ {(0)} (X _ {i}) + W _ {i} \frac {Y _ {i} - \mu_ {(1)} (X _ {i})}{e (X _ {i})} - (1 - W _ {i}) \frac {Y _ {i} - \mu_ {(0)} (X _ {i})}{1 - e (X _ {i})}. \end{array} \tag {3.6}
$$

**命题3.1.** 在本章开头给出的具有SUTVA、无混杂性和强重叠性的基本设定下，理想AIPW估计量具有（3.5）中给出的极限分布，即

$$
\sqrt {n} \left(\hat {\tau} _ {A I P W} ^ {*} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {A I P W}\right). \tag {3.7}
$$

**证明.** 理想AIPW估计量无偏的事实源于用于建立AIPW弱双重稳健性的讨论。此外，理想估计量是独立同分布项的平均值，因此标准中心极限定理立即意味着 $\sqrt { n } \left( \widehat { \tau } _ { A I P W } ^ { * } - \tau \right) \Rightarrow \mathcal { N } \left( 0 , \mathrm { V a r } \left[ \Gamma _ { i } \right] \right)$。最后，在无混杂性下，我们可以验证

$$
\begin{array}{l} \operatorname{Var} \left[ \Gamma_ {i} \right] = \operatorname{Var} \left[ \mu_ {(1)} (X _ {i}) - \mu_ {(0)} (X _ {i}) \right] + \mathbb {E} \left[ \left(W _ {i} \frac {Y _ {i} - \mu_ {(1)} (X _ {i})}{e (X _ {i})}\right) ^ {2} \right] \tag {3.8} \\ + \mathbb {E} \left[ \left((1 - W _ {i}) \frac {Y _ {i} - \mu_ {(0)} (X _ {i})}{1 - e (X _ {i})}\right) ^ {2} \right], \\ \end{array}
$$

这与（3.5）中给出的 $V _ { A I P W }$ 表达式一致。特别地，根据基本设定中的重叠性和有界二阶矩假设，（3.8）中的所有项都是有限的。□

基于这一结果，建立（3.5）就简化为证明，只要 $\hat { \mu } _ { ( w ) } ( \cdot )$ 和 $\hat { e } ( \cdot )$ 收敛得足够快，

$$
\sqrt {n} \left(\hat {\tau} _ {A I P W} - \hat {\tau} _ {A I P W} ^ {*}\right)\rightarrow_ {p} 0, \tag {3.9}
$$

即可行的AIPW估计量与理想估计量渐近等价。在合理假设下能够证明（3.9）类型的结果并非理所当然，这是AIPW具有强双重稳健性性质的结果。我们讨论过的其他估计量，如IPW和回归调整估计量，通常不满足这种类型的理想等价性质。

**交叉拟合（Cross-fitting）** 为了建立理想等价结果（3.9），考虑使用一种称为**交叉拟合（cross-fitting）**的技术对AIPW进行以下微小的算法修改是有帮助的。在高层次上，交叉拟合使用交叉折估计来避免由于过拟合造成的偏差；这样做背后的动机与我们在评估估计量预测准确性时经常使用交叉验证的原因密切相关。

**交叉拟合（Cross-fitting）**首先将数据（随机）分成两半 $\mathcal { T } _ { 1 }$ 和 $\mathcal { T } _ { 2 }$ ，然后使用一个估计量：

$$
\hat {\tau} _ {A I P W} = \frac {| \mathcal {I} _ {1} |}{n} \hat {\tau} ^ {\mathcal {I} _ {1}} + \frac {| \mathcal {I} _ {2} |}{n} \hat {\tau} ^ {\mathcal {I} _ {2}}, \quad \hat {\tau} ^ {\mathcal {I} _ {1}} = \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \hat {\mu} _ {(0)} ^ {\mathcal {I} _ {2}} (X _ {i}) \right. \tag {3.10}
$$

$$
\left. + W _ {i} \frac {Y _ {i} - \hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i})}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - (1 - W _ {i}) \frac {Y _ {i} - \hat {\mu} _ {(0)} ^ {\mathcal {I} _ {2}} (X _ {i})}{1 - \hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})}\right),
$$

其中 $\hat { \mu } _ { ( w ) } ^ { \mathcal { Z } _ { 2 } } ( \cdot )$ 和 $\hat { e } ^ { \pm \tau _ { 2 } } ( \cdot )$ 是仅使用半样本 $\mathcal { T } _ { 2 }$ 得到的 $\mu _ { ( w ) } ( \cdot )$ 和 $e ( \cdot )$ 的估计值，而 $\hat { \tau } ^ { \mathcal { I } _ { 2 } }$ 的定义类似（交换 $\mathcal { T } _ { 1 }$ 和 $\mathcal { T } _ { 2 }$ 的角色）。换句话说，$\hat { \tau } ^ { \mathcal { I } _ { 1 } }$ 是在 $\mathcal { T } _ { 1 }$ 上的一个处理效应估计量，它使用 $\mathcal { T } _ { 2 }$ 来估计其非参数成分，反之亦然。

交叉拟合带来的好处是，例如，如果 $i \in \mathcal { Z } _ { 1 }$ 且 $W _ { i } = 0$ ，那么 $Y _ { i } -$ $\hat { \mu } _ { ( 0 ) } ^ { \mathcal { T } _ { 2 } } ( X _ { i } )$ 不会受到过拟合的影响。如下文所示，通过创建这种诚实的残差，交叉拟合使我们能够在不需对用于估计 $\hat { \mu } _ { ( w ) } ( x )$ 和 ${ \hat { e } } ( x )$ 的算法做出详细假设的情况下，建立 (3.9) 类型的结果。

**定理 3.2.** 在具有 SUTVA、无混淆性（unconfoundedness）和强重叠性（strong overlap）的基本设定下，假设我们使用交叉拟合构建 $\hat { \tau } _ { A I P W }$ ，其估计量满足：对于 $w \in \{ 0 , 1 \}$ 且在交换 $\mathcal { T } _ { 1 }$ 和 $\mathcal { T } _ { 2 }$ 角色的情况下，

$$
\begin{array}{l} n ^ {- 2 \alpha_ {\mu}} \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(w)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(w)} (X _ {i})\right) ^ {2} \rightarrow_ {p} 0, \\ n ^ {- 2 \alpha_ {e}} \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\frac {1}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - \frac {1}{e (X _ {i})}\right) ^ {2} \rightarrow_ {p} 0, \tag {3.11} \\ \end{array}
$$

其中常数满足 $\alpha _ { \mu } , \alpha _ { e } \geq 0$ 且 $\alpha _ { \mu } + \alpha _ { e } \ge 1 / 2$ 。那么 (3.9) 以及因此 (3.5) 成立。

**证明.** 注意到，因为 $\hat { \tau } _ { A I P W } ^ { * }$ 不依赖于估计量，因此不受交叉拟合的影响，我们可以将**神谕 AIPW 估计量（oracle AIPW estimator）**写为：

$$
\hat {\tau} _ {A I P W} ^ {*} = \frac {| \mathcal {I} _ {1} |}{n} \hat {\tau} ^ {\mathcal {I} _ {1}, *} + \frac {| \mathcal {I} _ {2} |}{n} \hat {\tau} ^ {\mathcal {I} _ {2}, *}
$$

类似于 (3.10)。此外，我们可以将 $\hat { \tau } ^ { \mathcal { I } _ { 1 } }$ 本身分解为：

$$
\hat {\tau} ^ {\mathcal {I} _ {1}} = \hat {m} _ {(1)} ^ {\mathcal {I} _ {1}} - \hat {m} _ {(0)} ^ {\mathcal {I} _ {1}},
$$

$$
\hat {m} _ {(1)} ^ {\mathcal {I} _ {1}} = \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) + W _ {i} \frac {Y _ {i} - \hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i})}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})}\right), \tag {3.12}
$$

$\hat { m } _ { ( 0 ) } ^ { { \cal T } _ { 1 } , * }$ 和 $\hat { m } _ { ( 1 ) } ^ { { \ Z _ { 1 } } , * }$ 的定义类似。基于此设定，为了验证 (3.9)，只需证明：

$$
\sqrt {n} \left(\hat {m} _ {(1)} ^ {\mathcal {I} _ {1}} - \hat {m} _ {(1)} ^ {\mathcal {I} _ {1}, *}\right)\rightarrow_ {p} 0. \tag {3.13}
$$

然后通过对不同的折（folds）和处理状态进行同样的论证即可完成证明。

为此，我们将 (3.13) 中的误差项分解如下：

$$
\begin{array}{l} \hat {m} _ {(1)} ^ {\mathcal {I} _ {1}} - \hat {m} _ {(1)} ^ {\mathcal {I} _ {1}, *} \\ = \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) + W _ {i} \frac {Y _ {i} - \hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i})}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - \mu_ {(1)} (X _ {i}) - W _ {i} \frac {Y _ {i} - \mu_ {(1)} (X _ {i})}{e (X _ {i})}\right) \\ = \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) \left(1 - \frac {W _ {i}}{e (X _ {i})}\right)\right) \\ + \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} W _ {i} \left(\left(Y _ {i} - \mu_ {(1)} (X _ {i})\right) \left(\frac {1}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - \frac {1}{e (X _ {i})}\right)\right) \\ - \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} W _ {i} \left(\left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) \left(\frac {1}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - \frac {1}{e (X _ {i})}\right)\right) \\ \end{array}
$$

然后我们可以验证这些项由于不同原因而很小。

对于第一项，我们巧妙地利用了这样一个事实：由于我们的交叉拟合构造，在考虑 $\mathcal { T } _ { 1 }$ 上的项时，$\hat { \mu } _ { ( w ) } ^ { \mathcal { L } _ { 2 } }$ 可以有效地被视为确定性的。我们首先观察到，在给定 $\mathcal { T } _ { 2 }$ 和观测到的协变量值的条件下，该项可以视为独立零均值项的平均值，并且：

$$
\begin{array}{l} \mathbb {E} \left[ \left(\frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) \left(1 - \frac {W _ {i}}{e (X _ {i})}\right)\right)\right) ^ {2} \mid \mathcal {I} _ {2}, \{X _ {i} \} \right] \\ = \operatorname{Var} \left[ \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) \left(1 - \frac {W _ {i}}{e (X _ {i})}\right)\right) \Big | \mathcal {I} _ {2}, \{X _ {i} \} \right] \\ = \frac {1}{\left| \mathcal {I} _ {1} \right| ^ {2}} \sum_ {i \in \mathcal {I} _ {1}} \mathbb {E} \left[ \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) ^ {2} \left(1 - \frac {W _ {i}}{e (X _ {i})}\right) ^ {2} \mid \mathcal {I} _ {2}, \{X _ {i} \} \right] \tag {3.14} \\ = \frac {1}{| \mathcal {I} _ {1} | ^ {2}} \sum_ {i \in \mathcal {I} _ {1}} \frac {1 - e (X _ {i})}{e (X _ {i})} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) ^ {2} \\ \leq \frac {1 - \eta}{\eta} \frac {1}{| \mathcal {I} _ {1} | ^ {2}} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) ^ {2} = o _ {P} \left(\frac {1}{n ^ {1 + 2 \alpha_ {\mu}}}\right). \\ \end{array}
$$

上面的三个等号都归因于交叉拟合，而两个不等式则归因于重叠性 (3.2) 和一致性 (3.11)。因此，由于 $\alpha _ { \mu } \geq 0$ ，我们可以应用**切比雪夫不等式（Chebyshev's inequality）**来验证第一个加和项本身就是 $o _ { P } ( 1 / \sqrt { n } )$ ，即，正如所声称的，它在 $1 / { \sqrt { n } }$ 尺度上依概率可忽略。我们分解中的第二个加和项也可以通过类似的论证来界定。

最后，对于最后一个加和项，我们使用**柯西-施瓦茨（Cauchy-Schwarz）**论证：

$$
\begin{array}{l} \frac {1}{| \mathcal {I} _ {1} |} \sum_ {\{i: i \in \mathcal {I} _ {1}, W _ {i} = 1 \}} \left(\left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) \left(\frac {1}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - \frac {1}{e (X _ {i})}\right)\right) \\ \leq \sqrt {\frac {1}{| \mathcal {I} _ {1} |} \sum_ {\{i : i \in \mathcal {I} _ {1} , W _ {i} = 1 \}} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) ^ {2}} \tag {3.15} \\ \times \sqrt {\frac {1}{| \mathcal {I} _ {1} |} \sum_ {\{i : i \in \mathcal {I} _ {1} , W _ {i} = 1 \}} \left(\frac {1}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - \frac {1}{e (X _ {i})}\right) ^ {2}} = o _ {P} \left(\frac {1}{n ^ {\alpha_ {\mu} + \alpha_ {e}}}\right), \\ \end{array}
$$

这是由于风险衰减 (3.11)。因此，我们发现该项也是 $o _ { P } ( 1 / \sqrt { n } )$ ，即，正如所声称的，它在 $1 / { \sqrt { n } }$ 尺度上依概率可忽略。

**简化记号（Condensed notation）** 在本书的其余部分，我们将频繁遇到交叉拟合估计量。从现在起，我们将使用以下记号：我们将数据划分为 K 折（上面是 $K = 2$ ），并计算排除第 k 折后的估计量 $\hat { \mu } _ { ( w ) } ^ { ( - k ) } ( x )$ 等。然后，将 $k ( i )$ 写为将观测值映射到某个折的映射，我们可以写出：

$$
\begin{array}{l} \hat {\tau} _ {A I P W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\mu} _ {(1)} ^ {(- k (i))} \left(X _ {i}\right) - \hat {\mu} _ {(0)} ^ {(- k (i))} \left(X _ {i}\right) \right. (3.16) \\ \left. + W _ {i} \frac {Y _ {i} - \hat {\mu} _ {(1)} ^ {(- k (i))} \left(X _ {i}\right)}{\hat {e} ^ {(- k (i))} \left(X _ {i}\right)} - \left(1 - W _ {i}\right) \frac {Y _ {i} - \hat {\mu} _ {(0)} ^ {(- k (i))} \left(X _ {i}\right)}{1 - \hat {e} ^ {(- k (i))} \left(X _ {i}\right)}\right). (3.16) \\ \end{array}
$$

注意，定理 3.2 的结果同样适用于任意有限数量的交叉拟合折数 K（相同的证明在更新记号后同样适用）。

**置信区间（Confidence intervals）** 能够量化处理效应估计的不确定性也很重要。幸运的是，对于 AIPW，这被证明是相当直接的。在命题 3.1 的证明中，我们看到 $V_{AIPW}$ 与用于定义神谕 AIPW 估计量 (3.6) 的加和项 $\Gamma$ 的方差相匹配。这表明可以使用以下可行的方差估计：

$$
\widehat {V} _ {A I P W} = \frac {1}{n - 1} \sum_ {i = 1} ^ {n} \left(\widehat {\Gamma} _ {i} - \widehat {\tau} _ {A I P W}\right),
$$

$$
\widehat {\Gamma} _ {i} = \hat {\mu} _ {(1)} ^ {(- k (i))} (X _ {i}) - \hat {\mu} _ {(0)} ^ {(- k (i))} (X _ {i}) \tag {3.17}
$$

$$
+ W _ {i} \frac {Y _ {i} - \hat {\mu} _ {(1)} ^ {(- k (i))} (X _ {i})}{\hat {e} ^ {(- k (i))} (X _ {i})} - (1 - W _ {i}) \frac {Y _ {i} - \hat {\mu} _ {(0)} ^ {(- k (i))} (X _ {i})}{1 - \hat {e} ^ {(- k (i))} (X _ {i})}.
$$

定理 3.2 的证明表明，在我们的假设下，$\widehat { V } _ { A I P W }  _ { p }$ $V _ { A I P W }$ 。因此，我们可以为 $\tau$ 生成水平为 $\alpha$ 的置信区间：

$$
\tau \in \left(\hat {\tau} _ {A I P W} \pm \Phi^ {- 1} \left(1 - \frac {\alpha}{2}\right) \frac {1}{\sqrt {n}} \sqrt {\hat {V} _ {A I P W}}\right), \tag {3.18}
$$

其中 $\Phi ( \cdot )$ 是标准高斯分布函数（CDF），并且这些区间在大样本下将以概率 $1-\alpha$ 达到覆盖。类似的论证也可用于通过重抽样方法（如 Efron [1982] 所述）进行推断。

**如果倾向得分是已知的会怎样？** 一个值得考虑的特殊情况是，当倾向得分已知，并且我们使用真实的倾向得分 $\hat { e } ^ { - k ( i ) } ( X _ { i } ) = e ( X _ { i } )$ 来实现交叉拟合 AIPW 估计量 (3.16) 时会发生什么。在这种情况下，定理 3.2 立即蕴含以下结果。

**推论 3.3.** 在具有 SUTVA、无混淆性和强重叠性的基本设定下，假设我们知道真实的倾向得分并使用它们来构建 AIPW 估计量。进一步假设：

$$
\frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(w)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(w)} (X _ {i})\right) ^ {2} \rightarrow_ {p} 0, \tag {3.19}
$$

对于 $w \in \{ 0 , 1 \}$ 且在交换 $\mathcal { T } _ { 1 }$ 和 $\mathcal { T } _ { 2 }$ 角色的情况下成立。那么 (3.9) 和 (3.5) 成立；此外，$\hat { \tau } _ { A I P W }$ 是精确无偏的，即 $\mathbb { E } \left[ \hat { \tau } _ { A I P W } \right] = \tau$ 。

**证明.** 中心极限定理（CLT）的陈述通过将定理 3.2 应用于 $\alpha _ { \mu } = 0$ 和 $\alpha _ { e } = + \infty$ 得到。无偏性声明通过注意到在 (3.13) 下方的分解中，当使用真实倾向得分时，第二项和第三项消失，而第一项是零均值的。□

这个结果值得注意，因为它表明，如果我们使用具有真实倾向得分的 AIPW，那么只要使用在极弱意义 (3.19) 下一致的任意回归调整，AIPW 就能达到目标渐近行为 (3.5)。特别地，不需要收敛速率。

众所周知，有几种机器学习方法，包括 **k 近邻（k-nearest neighbors）**，是普遍一致的，即它们对于任何独立同分布（IID）数据生成分布都能达到误差保证 (3.19)，除了要求 $E [ Y _ { i } ^ { 2 } ( w ) ] < \infty$ [Stone, 1977] 之外，无需对 $X _ { i }$ 和 $Y _ { i } ( w )$ 的联合分布做任何假设。推论 3.3 意味着，如果我们在基本设定下，使用一个普遍一致的 $\hat { \mu } _ { ( w ) } ( x )$ 估计量和真实的倾向得分来运行 AIPW，那么它总是满足 (3.5)。

推论 3.3 也为第 2 章中强调的一个明显悖论提供了实际的解决方案，即具有神谕权重的 IPW 有时（在特定设定下）可能被具有估计权重的 IPW 超越。这似乎导致了一种紧张关系：如果倾向得分已知，那么我们可以选择使用神谕 IPW（它总是无偏的，但具有较大的渐近方差），或者使用可行的 IPW（它可能更准确，但如果我们意外地错误指定了倾向性模型，则可能完全失败）。

推论 3.3 之所以有帮助，是因为仔细检查后，人们注意到推论 3.3 中（在相当广泛的普遍性下）达到的渐近方差 $V _ { A I P W }$ 恰好匹配了可行的 IPW 在特殊情况（即 $X _ { i }$ 具有离散支持）下达到的渐近方差 $V _ { S T R A T }$ 。因此，推论 3.3 向我们展示的是，如果我们知道真实的倾向得分，那么我们可以总是（并且没有任何真正的缺点，至少在渐近意义上）通过简单地使用带有普遍一致回归调整的 AIPW 来避免神谕 IPW 的额外渐近方差。

## 3.2 无混杂性下的有效估计

在第二章中，我们研究了在**无混杂性（unconfoundedness）** 以及 $X _ { i }$ 为离散情况下的**平均处理效应（average treatment effect）** 估计。在此设定下，按 $X _ { i }$ 分层估计量显然是一种（或者说是唯一）自然的做法；在定理 2.1 中，我们证明了该估计量达到了渐近方差 $V _ { S T R A T }$ 。与此同时，在本章中，我们研究了一个看似完全不同的估计量——**增广逆概率加权（Augmented Inverse Probability Weighting, AIPW）**，并证明了它在更一般的条件下（尤其是不假设 $X _ { i }$ 为离散）也能达到渐近方差 $V _ { A I P W } = V _ { S T R A T }$ 。

这些观察表明，以下行为

$$
\begin{array}{l} \sqrt {n} \left(\hat {\tau} - \tau^ {*}\right) \Rightarrow \mathcal {N} \left(0, V ^ {*}\right) \\ V ^ {*} = \operatorname{Var} \left[ \tau (X _ {i}) \right] + \mathbb {E} \left[ \frac {\sigma_ {0} ^ {2} (X _ {i})}{1 - e (X _ {i})} \right] + \mathbb {E} \left[ \frac {\sigma_ {1} ^ {2} (X _ {i})}{e (X _ {i})} \right], \tag {3.20} \\ \end{array}
$$

实际上可能是我们在无混杂性下，对于任何非参数平均处理效应估计量 $\hat { \tau }$ 所能期望的最优行为。定理 3.2 提供了一个上界，表明在相当广泛的条件下，一个实用的估计量 $\hat { \tau } _ { A I P W }$ 确实可以达到这种行为。同时，我们在第二章的讨论提供了一个启发式的下界；毕竟，在 $X _ { i }$ 为离散的设定下，人们怎么可能指望找到一个比按 $X _ { i }$ 分层估计量更精确的估计量呢？

以下结果证实了这一猜想，其证明技术来自 Chamberlain [1992]。遵循 H´ajek [1972] 的思路，他根据**局部渐近极小极大准则（local asymptotic minimax criterion）** 定义了最优性：如果存在一个满足 (3.20) 的估计量，并且在 $P$ 的适当表达邻域内，不存在任何估计量能比 (3.20) 更一致地精确，则 $V ^ { * }$ 被称为估计 $\tau$ 的**有效方差（efficient variance）** 。21 此外，任何满足 (3.20) 的估计量（可能还需要假设合理的正则条件）都被称为**有效的（efficient）**。

**定理 3.4.** 在满足 **SUTVA（稳定单元处理值假设）**、无混杂性和强重叠性的基本设定下，$V ^ { * }$ 是估计平均处理效应的有效方差。

*证明。* 我们已在定理 3.2 中证明了存在满足 (3.20) 的估计量。对于局部最优性的陈述，我们遵循 Chamberlain [1992] 定理 1 的蓝图，并执行以下步骤：首先，我们考虑 $( X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 0 ) )$ 具有分布 $P$ 且联合支撑为离散的情况（即 $X _ { i }$ 和 $Y _ { i } ( w )$ 都具有离散支撑），并验证 **ATE（平均处理效应）** 的**饱和最大似然估计量（saturated maximum likelihood estimator）** 的渐近方差等于 $V ^ { * }$ 。然后，我们论证，对于离散的 $P$，ATE 估计是一个参数问题，因此最大似然估计必须是有效的；并且任何连续分布都可以被离散分布很好地逼近，因此这个有效性结果可以推广到连续情形。技术细节以及验证此蓝图有效性的证明，请参阅 Chamberlain [1992]。

现在考虑 $P$ 在离散空间 $\mathcal { X } \times \mathcal { Y } \times \mathcal { Y }$ 上取值的情况，其中 $\mathcal { V } \subset \mathbb { R }$ 。对于任何分布 $P$，令 $\tau ( P ) = \mathbb { E } _ { P } \left[ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \right]$，并注意到，在无混杂性和离散支撑下，

$$
\tau (P) = \sum_ {x \in \mathcal {X}} P (x) \left(\sum_ {y \in \mathcal {Y}} y   P _ {1} (y | x) - \sum_ {y \in \mathcal {Y}} y   P _ {0} (y | x)\right) \tag {3.21}
$$

其中 $P ( x ) = \mathbb { E } _ { P } \left[ X _ { i } = x \right]$ 且 $P _ { w } ( y | x ) = \mathbb { E } _ { P } \left[ Y _ { i } = y \vert X _ { i } = x , W _ { i } = w \right]$ 。现在，给定从 $P$ 中抽取的 $n$ 个样本，令 $n _ { x } = | \{ i : X _ { i } = x \} | , n _ { x w } \stackrel { \cdot } { = } | \{ i : X _ { i } = x , \bar { W _ { i } } = w \}$ | 且 $n _ { x y w } = | \{ i : X _ { i } = x , Y _ { i } = y , W _ { i } = w \} |$ 。数据生成分布 $P$ 的饱和最大似然估计量由 $\widehat { P } ( x ) = n _ { x } / n$ 和 $\widehat { P } _ { w } ( y | x ) = n _ { x y w } / n _ { x w }$ 给出。那么 $\tau$ 的最大似然估计量为

$$
\hat {\tau} = \tau (\widehat {P}) = \sum_ {x \in \mathcal {X}} \widehat {P} (x) \left(\sum_ {y \in \mathcal {Y}} y   \widehat {P} _ {1} (y | x) - \sum_ {y \in \mathcal {Y}} y   \widehat {P} _ {0} (y | x)\right), \tag {3.22}
$$

通过代数运算可以验证，该估计量在此设定下等价于 $\scriptstyle { \hat { \tau } } _ { S T R A T }$ 。因此，这里最大似然估计的渐近方差是 $V _ { S T R A T }$，根据定理 2.1，它等于 $V ^ { * }$ 。□

**比较正则条件** 上述定义中的一个模糊之处是，我们说一个估计量如果在“合理”的正则条件下实现了行为 (3.20) 就是有效的——但“合理”的正则条件意味着什么？到目前为止，我们看到了关于实现行为 (3.20) 的估计量的 3 个结果：推论 3.3 表明，对于已知倾向得分的 AIPW，这基本上无需假设；定理 3.2 表明，对于使用估计倾向得分的 AIPW，在（中等强度？）收敛速度假设 (3.11) 下可实现；而定理 2.1 表明，对于按 $X _ { i }$ 分层的估计量，在（非常强的）$X _ { i }$ 为离散的假设下可实现。

这种模糊性是故意的，并且有助于描述和评估在无混杂性下提出的各种平均处理效应估计量。在考虑一个候选估计量时，一个好的首要问题是询问它是否有效，即它是否有时能实现行为 (3.11)。如果一个估计量不是有效的（例如，像**神谕 IPW（oracle IPW）** 估计量），那么在这一步就可以舍弃它。然后，在有效的估计量中，一个好的次要问题是询问它的稳健性如何，即实现有效性所需的正则条件有多强。这可以论证，例如，$\hat { \tau } _ { A I P W }$ 需要比 $\scriptstyle { \hat { \tau } } _ { S T R A T }$ 弱得多的正则条件来实现理想的渐近性能，从这个角度看，$\hat { \tau } _ { A I P W }$ 更优。

**有效性是一个现实的目标吗？** 直到最近，上述观点（即有效性应指导平均处理效应估计量的实际选择）在许多计量经济学家和统计学家看来还是有争议的。实现有效性的方法通常被认为是脆弱的、复杂的和/或不实用的；并且在需要在无混杂性下进行处理效应估计的问题中，计量经济学实践主要集中于需要参数假设且仅凭无混杂性无法保证一致性的方法（例如线性回归），或非有效但概念简单的方法（例如匹配）。

早期旨在实现有效性的方法在实践中难以使用的批评是中肯的：例如，这些方法通常依赖于特定的光滑性假设，然后依赖具有特定基函数（取决于假定的光滑性类别）的级数估计量来构建处理效应估计量。

然而，**双重机器学习（double machine learning）** 框架使得有效处理效应估计量的广泛使用变得更加实用。主要的正则条件 (3.11) 不依赖于我们如何选择估计非参数组件，而只要求它们在平方误差损失下足够精确。机器学习方法通常通过平方误差损失下的交叉验证进行调整，这种调整预测器的方式与使 (3.11) 中的误差项变小完全一致。因此，也许令人惊讶的是，尽管机器学习乍看起来似乎是一种应尽可能远离因果推断的技术，但事实证明——通过双重机器学习构造——机器学习（以及更一般的自动黑箱非参数预测）是使有效处理效应估计在各种设定下变得实用的关键组成部分。

## 3.3 文献注释

关于通过 AIPW 进行**半参数有效（semiparametrically efficient）** 处理效应估计的文献由 Robins、Rotnitzky 和 Zhao [1994] 开创，并在包括 Robins 和 Rotnitzky [1995] 以及 Scharfstein、Rotnitzky 和 Robins [1999] 在内的一系列论文中得到发展。AIPW 估计量的形式也出现在 Cassel、S¨arndal 和 Wretman [1976] 关于抽样调查的早期工作中。了解倾向得分对平均处理效应估计的半参数效率界的影响在 Hahn [1998] 中进行了讨论，而 Farrell [2015] 首次考虑了具有高维回归调整的 AIPW 的行为。这些结果属于更广泛的半参数学文献，包括 Bickel、Klaassen、Ritov 和 Wellner [1993] 以及 Newey [1994]。

本文采用的方法，侧重于使用通用机器学习估计量处理非参数组件并进行交叉拟合，遵循了 Chernozhukov 等人 [2018] 的双重机器学习框架。该方法的一个主要优势在于其通用性以及处理 $\hat { \mu } _ { ( w ) } ( x )$ 和 $\hat { e } ( x )$ 的任意机器学习估计量的能力。另一个密切相关的框架是 van der Laan 和 Rubin [2006] 的**目标学习（targeted learning）** 框架，它使用与 AIPW 不同的函数形式，但也可以证明使用机器学习估计量处理非参数组件能够实现有效性 [van der Laan 和 Rose, 2011]。

已知有大量估计量可以在各种正则条件下实现有效性。例如，Hahn [1998] 表明，在强光滑性条件和特定回归估计量下，非参数回归调整估计量可以是有效的，而 Hirano、Imbens 和 Ridder [2003] 则展示了非参数 IPW 的这种结果。然而，定理 3.2 中给出的 AIPW 的有效性结果要稳健得多——因为它允许使用通用的机器学习方法，只要它们满足相对温和的速度条件 (3.11)。

最近，人们对推导在最小条件下实现有效性的估计量产生了相当大的兴趣。在函数 $\mu _ { ( w ) } ( \cdot )$ 和 $e ( \cdot )$ 属于 **Hölder 光滑类（H¨older smoothness classes）** 的情况下，Robins 等人 [2017] 表明，记 $\alpha _ { \mu }$ 和 $\alpha _ { e }$ 为在所假设的光滑性假设下能达到 (3.11) 型收敛速度的最佳常数，则实现有效性所需的最弱条件是

$$
\frac {\alpha_ {\mu}}{1 - 2 \alpha_ {\mu}} + \frac {\alpha_ {e}}{1 - 2 \alpha_ {e}} \geq \frac {1}{2}, \tag {3.23}
$$

并且这个速度可以使用 Robins 等人 [2017] 所称的**高阶影响函数（higher-order influence function, HOIF）** 估计量来实现。条件 (3.23) 相对于定理 3.2 中的条件 $\alpha _ { \mu } + \alpha _ { e } \ge 1 / 2$ 的改进是显著的；例如，当两个速度相等时，在定理 3.2 中我们允许 $\alpha _ { \mu } =$ $\alpha _ { e } \geq 1 / 4$，而 (3.23) 允许 $\alpha _ { \mu } = \alpha _ { e } \ge 1 / 6$。

然而，Robins 等人 [2017] 的 HOIF 估计量的一个挑战是，迄今为止在实际应用中实现起来仍有困难；因此，人们致力于开发能够超越 AIPW 同时保持实际可行性的方法。Hirshberg 和 Wager [2021] 表明，一种 AIPW 的变体，其选择的倾向模型专门用于最小化来自 $\hat { \mu } _ { ( w ) } ( x )$ 误差的偏差，在 Hölder 情形下，当 $\alpha _ { \mu } \geq 1 / 4$ 时（对 $\alpha _ { e }$ 无假设）是有效的；注意到这对应于最优性曲面 (3.23) 的一个极端点。同时，Newey 和 Robins [2018] 以及 McClean 等人 [2024] 展示了在某些设定下，使用欠光滑估计量和三路交叉拟合如何能够达到有效性的最小条件。
# 第2章 非混淆性与倾向得分（Unconfoundedness and the Propensity Score）

**随机对照试验（Randomized Controlled Trials, RCTs）**代表了一类强大但略显僵化的研究设定，在这些设定中我们可以识别并估计因果效应。关于统计因果推断的文献（以及本书）的一个核心关注点在于，我们如何在放宽RCT假设的同时，仍能严谨地估计因果效应，从而拓宽因果推断所能解决的问题范围。

在本章中，我们将考虑对RCT假设进行第一个简单的放宽。我们将不再假设处理变量 $W _ { i }$ 是随机分配的；然而，我们将假设我们观测到了处理前协变量 $X _ { i }$ ，使得在给定 $X _ { i }$ 的条件下，处理变量近似于随机分配。然后，我们将讨论一系列利用这种"非混淆性"假设来估计**平均处理效应（Average Treatment Effect, ATE）**的方法，包括基于估计倾向得分（即接受处理的条件概率）的方法。为简洁起见，在本章（以及后续章节）中，我们将始终假设样本单元是从一个超总体中独立采样的。

## 超越单一随机对照试验（Beyond a single randomized controlled trial）

超越单一RCT的最简单方式是考虑两个RCT。作为一个具体例子，假设我们希望通过向青少年提供现金奖励来阻止他们吸烟。加州帕洛阿尔托约 $5\%$ 的青少年和瑞士日内瓦约20%的青少年被随机选中参与这项研究。

| 帕洛阿尔托 | 非吸烟者 | 吸烟者 | 日内瓦 | 非吸烟者 | 吸烟者 |
|------------|----------|--------|--------|----------|--------|
| 处理组     | 152      | 5      | 处理组 | 581      | 350    |
| 对照组     | 2362     | 122    | 对照组 | 2278     | 1979   |

在每个城市内，我们都有一个RCT，并且实际上可以清楚地看到处理是有帮助的。然而，查看汇总数据会产生误导，看起来处理反而有害；这是有时被称为**辛普森悖论（Simpson's paradox）**的一个例子：

| 帕洛阿尔托 + 日内瓦 | 非吸烟者 | 吸烟者 |
|---------------------|----------|--------|
| 处理组              | 733      | 401    |
| 对照组              | 4640     | 2101   |

一旦我们汇总数据，这就不再是一个RCT，因为日内瓦人既更有可能接受处理，又更有可能在无论是否接受处理的情况下吸烟。为了得到ATE的一致估计，我们需要分别估计每个城市的处理效应：

$$
\hat {\tau} _ {\mathrm{PA}} = \frac {5}{1 5 2 + 5} - \frac {1 2 2}{2 3 6 2 + 1 2 2} \approx -1.7\% ,
$$

$$
\hat {\tau} _ {\mathrm{GVA}} = \frac {3 5 0}{3 5 0 + 5 8 1} - \frac {1 9 7 9}{2 2 7 8 + 1 9 7 9} \approx -8.9\%
$$

$$
\hat {\tau} = \frac {2 6 4 1}{2 6 4 1 + 5 1 8 8} \hat {\tau} _ {\mathrm{PA}} + \frac {5 1 8 8}{2 6 4 1 + 5 1 8 8} \hat {\tau} _ {\mathrm{GVA}} \approx - 6.5 \%.
$$

这个估计量的统计性质是什么？这个想法如何推广到连续的 $x$ ？

## 2.1 分层估计（Stratified estimation）

将上述讨论形式化，假设我们有协变量 $X _ { i }$ ，其取值于一个离散空间 $X _ { i } \in { \mathcal { X } }$ ，且 $| \mathcal { X } | = p < \infty$ 。进一步假设处理分配在给定 $X _ { i }$ 的条件下是随机的（即，在每个由 $x$ 的水平定义的组中，我们都有一个RCT）：

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp W _ {i} \mid X _ {i} = x, \text {  对于所有   } x \in \mathcal {X}. \tag {2.1}
$$

定义**条件平均处理效应（Conditional Average Treatment Effect, CATE）**为

$$
\tau (x) = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \mid X _ {i} = x \right]. \tag {2.2}
$$

那么，上述讨论表明，我们应该能够通过汇总条件平均处理效应的估计来估计ATE $\tau$，

$$
\hat {\tau} _ {S T R A T} = \sum_ {x \in \mathcal {X}} \frac {n _ {x}}{n} \hat {\tau} (x), \quad \hat {\tau} (x) = \frac {1}{n _ {x 1}} \sum_ {\{X _ {i} = x, W _ {i} = 1 \}} Y _ {i} - \frac {1}{n _ {x 0}} \sum_ {\{X _ {i} = x, W _ {i} = 0 \}} Y _ {i}, \tag {2.3}
$$

其中 $n _ { x } = | \{ i : X _ { i } = x \} |$ 且 $n _ { x w } = | \{ i : X _ { i } = x , W _ { i } = w \} |$ 。看待(2.3)中估计量的另一种方式是，我们在使用协变量 $X _ { i }$ 对样本进行分层后，应用了均值差异估计量；因此，我们将其称为**分层估计量（stratified estimator）**。

以下结果验证了分层估计量在我们的假设下实际上是有效的。值得注意的是，渐近方差 $V _ { S T R A T }$ 不依赖于 $| { \mathcal { X } } | = p$ （即组数），或者等价地，不依赖于在形成(2.3)过程中估计的"参数" $\tau ( x )$ 的数量。正如我们将在下一章中看到的，这一事实在实现观察性研究中平均处理效应的高效非参数推断方面起着关键作用。

**定理 2.1.** 假设 $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , W _ { i } \} \stackrel { i i d } { \sim } P$ 服从某个分布 $P$ ，其中 $X _ { i }$ 取值于一个有限基数的集合 $X$ ，且潜在结果在给定 $X _ { i }$ 的条件下具有有界的二阶矩。进一步假设(2.1)和**SUTVA（稳定单元处理值假设，Stable Unit Treatment Value Assumption）**均成立，并且对于每个 $x \in { \mathcal { X } }$ 存在非平凡的处理变异，即，记 $e ( x ) = \mathbb { P } \left[ W _ { i } = 1 \big | X _ { i } = x \right]$ ，对于所有 $x$ 我们有 $0 < e ( x ) < 1$ 。那么，使用(1.21)中的记号，

$$
\sqrt {n} \left(\hat {\tau} _ {S T R A T} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {S T R A T}\right)
$$

$$
V _ {S T R A T} = \operatorname{Var} \left[ \tau (X _ {i}) \right] + \mathbb {E} \left[ \frac {\sigma_ {(1)} ^ {2} (X _ {i})}{e (X _ {i})} + \frac {\sigma_ {(0)} ^ {2} (X _ {i})}{1 - e (X _ {i})} \right]. \tag {2.4}
$$

**证明.** 记 $\lambda ( x ) = \mathbb { P } \left[ X _ { i } = x \right]$ 为协变量每个水平 $x _ { i }$ 的普遍性，并将 $\ddot { \lambda } ( x ) = n _ { x } / n$ 解释为其估计量。然后我们可以将分层估计量展开为

$$
\hat {\tau} _ {S T R A T} = \sum_ {x \in \mathcal {X}} \hat {\lambda} (x) \hat {\tau} (x) = \sum_ {x \in \mathcal {X}} \lambda (x) \tau (x) + \sum_ {x \in \mathcal {X}} \left(\hat {\lambda} (x) - \lambda (x)\right) \tau (x)
$$

$$
+ \sum_ {x \in \mathcal {X}} \lambda (x) (\hat {\tau} (x) - \tau (x)) + \sum_ {x \in \mathcal {X}} \left(\hat {\lambda} (x) - \lambda (x)\right) (\hat {\tau} (x) - \tau (x)).
$$

现在我们研究上述表达式中的每一项。首先，注意

$$
\sum_ {x \in \mathcal {X}} \lambda (x) \tau (x) = \mathbb {E} [ \tau (X _ {i}) ] = \tau
$$

是我们的目标估计量。通过简单的代数变换，第二项可以重新表示为

$$
\sum_ {x \in \mathcal {X}} \left(\hat {\lambda} (x) - \lambda (x)\right) \tau (x) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\tau (X _ {i}) - \tau\right),
$$

因此，关于IID平均值的标准中心极限定理意味着

$$
\sqrt {n} \left(\sum_ {x \in \mathcal {X}} \left(\hat {\lambda} (x) - \lambda (x)\right) \tau (x)\right) \Rightarrow \mathcal {N} \left(0, \operatorname{Var} \left[ \tau (X _ {i}) \right]\right).
$$

接下来，我们的假设 $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , W _ { i } \} \stackrel { \mathrm { i i d } } { \sim } P$ 以及(2.1)成立意味着 $W _ { i } \vert X _ { i } = x , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \sim$ 伯努利分布 $( e ( x ) )$ 。因此，根据定理1.2，

$$
\sqrt {n _ {x}} \left(\hat {\tau} (x) - \tau (x)\right) \Rightarrow \mathcal {N} \left(0, \frac {\sigma_ {(1)} ^ {2}}{e (x)} + \frac {\sigma_ {(0)} ^ {2} (x)}{1 - e (x)}\right),
$$

并且 $\hat {\tau} ( x )$ 中的抽样误差彼此渐近独立，也与 $n _ { x }$ （以及因此与 $\hat {\tau} _ { S T R A T }$ 分解中的第二个求和项）渐近独立。因此，根据**Slutsky引理（Slutsky's lemma）**，

$$
\sum_ {x \in \mathcal {X}} \lambda (x) (\hat {\tau} (x) - \tau (x)) \Rightarrow \mathcal {N} \left(0, \sum_ {x \in \mathcal {X}} \lambda (x) \left(\frac {\sigma_ {(1)} ^ {2}}{e (x)} + \frac {\sigma_ {(0)} ^ {2} (x)}{1 - e (x)}\right)\right),
$$

因此上述第二项和第三项之和具有(2.4)中声明的极限分布。最后，我们上面的论证也意味着

$$
\left(\hat {\lambda} (x) - \lambda (x)\right) (\hat {\tau} (x) - \tau (x)) = \mathcal {O} _ {P} \left(\frac {1}{n}\right) \text {对于所有} x \in \mathcal {X},
$$

因此第四项在渐近意义上可以忽略。

## 连续X与倾向得分（Continuous X and the propensity score）

上面，我们考虑了 $X$ 是具有有限水平数的离散变量，且处理 $W _ { i }$ 在给定 $X _ { i } = x$ 的条件下近似于随机分配的情况，如(2.1)所示。在这种情况下，我们发现我们仍然可以通过汇总组内处理效应估计来准确估计ATE，并且组数 $| { \mathcal { X } } | = p$ 的确切数量不会影响推断的准确性。然而，如果 $X$ 是连续的（或者 $X$ 的基数非常大），这个结果就不能直接应用——因为我们无法为每个可能的 $x \in \mathcal { X }$ 值获取足够的样本来定义如(2.3)所示的 $\hat {\tau} ( x )$ 。

为了将我们的分析推广到离散 $X$ 的情况之外，我们需要超越试图通过简单平均来估计每个 $x$ 值的 $\tau ( x )$ ，而是使用更间接的论证。为此，我们首先需要推广"每组中的RCT"假设。形式上，我们只需写出相同的内容：

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp W _ {i} \mid X _ {i}, \tag {2.5}
$$

尽管现在 $X _ { i }$ 可能是一个任意的随机变量，并且对这一陈述的解释可能需要更加谨慎。从定性角度看，理解(2.5)的一种方式是，我们已经测量了足够的协变量来捕捉 $W_i$ 与潜在结果之间的任何依赖关系，因此，给定 $X _ { i }$ ， $W _ { i }$ 无法"窥视" $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \}$ 。我们将这一假设称为**非混淆性（unconfoundedness）**。

假设(2.5)在实践中可能看起来是一个难以使用的假设，因为它涉及对连续随机变量的条件化。然而，正如**Rosenbaum和Rubin [1983]**所示，通过考虑**倾向得分（propensity score）**，这一假设可以变得更加易于处理：

$$
e (x) = \mathbb {P} \left[ W _ {i} = 1 \mid X _ {i} = x \right]. \tag {2.6}
$$

从统计角度看，倾向得分的一个关键性质是它是一个**平衡得分（balancing score）**：如果(2.5)成立，那么实际上

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp W _ {i} \mid e \left(X _ {i}\right), \tag {2.7}
$$

即，要消除与非随机处理分配相关的偏差，实际上只需控制 $e ( X )$ 而非 $X$ 就足够了。我们可以如下验证这一论断：

$$
\begin{array}{l} \mathbb {P} \left[ W _ {i} = w \mid \{Y _ {i} (0), Y _ {i} (1) \}, e (X _ {i}) \right] \\ = \int_ {\mathcal {X}} \mathbb {P} \left[ W _ {i} = w \mid \left\{Y _ {i} (w) \right\}, X _ {i} = x \right] \mathbb {P} \left[ X _ {i} = x \mid \left\{Y _ {i} (w) \right\}, e (X _ {i}) \right] d x \\ = \int_ {\mathcal {X}} \mathbb {P} \left[ W _ {i} = w \mid X _ {i} = x \right] \mathbb {P} \left[ X _ {i} = x \mid \left\{Y _ {i} (w) \right\}, e \left(X _ {i}\right) \right] d x \quad (\text {非混淆性}) \\ = \left\{ \begin{array}{l l} e (X _ {i}) & \text {如果   w = 1, } \\ 1 - e (X _ {i}) & \text {否则. } \end{array} \right. \\ \end{array}
$$

(2.7)的含义是，如果我们可以将观测值划分为具有（几乎）恒定倾向得分 $e ( x )$ 值的组，那么我们就可以通过 $\scriptstyle { \hat { \tau } } _ { S T R A T }$ 的变体来一致地估计平均处理效应。

## 倾向得分分层（Propensity stratification）

这一思想的一个具体实现是**倾向得分分层（propensity stratification）**，其步骤如下。首先，通过非参数回归获得倾向得分的估计值 $\hat { e } ( x )$ ，并选择分层数 $J$ 。然后：

1. 根据倾向得分对观测值进行排序，使得

$$
\hat {e} \left(X _ {i _ {1}}\right) \leq \hat {e} \left(X _ {i _ {2}}\right) \leq \dots \leq \hat {e} \left(X _ {i _ {n}}\right). \tag {2.8}
$$

2. 使用排序后的倾向得分将样本划分为 $J$ 个大小相等的层，并在每个层 $j = 1, ..., J$ 中，计算该层的简单均值差异处理效应估计量：

$$
\hat {\tau} _ {j} = \frac {\sum_ {j = \lfloor (j - 1) n / J \rfloor + 1} ^ {\lfloor j n / J \rfloor} W _ {i} Y _ {i}}{\sum_ {j = \lfloor (j - 1) n / J \rfloor + 1} ^ {\lfloor j n / J \rfloor} W _ {i}} - \frac {\sum_ {j = \lfloor (j - 1) n / J \rfloor + 1} ^ {\lfloor j n / J \rfloor} \left(1 - W _ {i}\right) Y _ {i}}{\sum_ {j = \lfloor (j - 1) n / J \rfloor + 1} ^ {\lfloor j n / J \rfloor} \left(1 - W _ {i}\right)}. \tag {2.9}
$$

3. 通过将(2.3)的思想应用于各层来估计平均处理效应：

$$
\hat {\tau} _ {P S T R A T} = \frac {1}{J} \sum_ {j = 1} ^ {J} \hat {\tau} _ {j}. \tag {2.10}
$$

上述论证立即表明，由于(2.7)，只要 $\hat { e } ( x )$ 对 $e ( x )$ 一致一致，且层数 $J$ 随 $n$ 适当增长，则 $\hat {\tau} _ {P S T R A T}$ 对 $\tau$ 是一致的；更多细节请参见第16章的练习4。

## 2.2 逆倾向得分加权（Inverse-propensity weighting）

另一种在算法上更简单的利用无混杂性的方法是**逆倾向得分加权（Inverse-Propensity Weighting, IPW）**。与之前一样，我们首先通过非参数回归估计 $\hat { e } ( x )$；然而，我们随后使用倾向得分模型的输出来构建一个重新加权的均值差异型估计量

$$
\hat {\tau} _ {I P W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{\hat {e} (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})}\right). \tag {2.11}
$$

IPW 背后的直觉是，如果某些单元极不可能接受处理，那么我们应该在它们确实接受处理的罕见事件上对其**上加权**，而在它们未接受处理的更常见事件上对其**下加权**，等等。这种重新加权使我们能够“消除”由倾向得分变化引起的抽样偏差。

分析它的最简单方法是将其与一个实际知道倾向得分的**理想估计量（oracle）**进行比较：

$$
\hat {\tau} _ {I P W} ^ {*} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})}\right). \tag {2.12}
$$

我们首先在下面建立理想 IPW 估计量的渐近性质。一旦我们确立了 $\hat { \tau } _ { I P W } ^ { * }$ 的相合性，那么作为一个（几乎）直接的推论，只要 $\hat { e } ( x )$ 对于 $e ( x )$ 是相合的，${ \hat { \tau } } _ { I P W }$ 也是相合的。

**定理 2.2.** 假设 $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , W _ { i } \} \stackrel { i i d } { \sim } P$ ，(2.5) 和 SUTVA 均成立，并且下面 $V _ { I P W }$ * 表达式中使用的所有矩都是有限的。那么，理想 IPW 估计量是无偏的，即 E $\left[ \hat { \tau } _ { I P W } ^ { * } \right] = \tau$ ，并且

$$
\sqrt {n} \left(\hat {\tau} _ {I P W} ^ {*} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {I P W ^ {*}}\right)
$$

$$
V _ {I P W ^ {*}} = \operatorname{Var} \left[ \tau (X _ {i}) \right] + \mathbb {E} \left[ \frac {\left(\mu_ {(0)} (X _ {i}) + (1 - e (X _ {i})) \tau (X _ {i})\right) ^ {2}}{e (X _ {i}) (1 - e (X _ {i}))} \right] \tag {2.13}
$$

$$
+ \mathbb {E} \left[ \frac {\sigma_ {(1)} ^ {2} (X _ {i})}{e (X _ {i})} + \frac {\sigma_ {(0)} ^ {2} (X _ {i})}{1 - e (X _ {i})} \right].
$$

**证明.** 我们首先如下检验无偏性陈述：

$$
\begin{array}{l} \mathbb {E} \left[ \hat {\tau} _ {I P W} ^ {*} \right] = \mathbb {E} \left[ \frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})} \right] (IID) \\ = \mathbb {E} \left[ \frac {W _ {i} Y _ {i} (1)}{e \left(X _ {i}\right)} - \frac {\left(1 - W _ {i}\right) Y _ {i} (0)}{1 - e \left(X _ {i}\right)} \right] (SUTVA) \\ = \mathbb {E} \left[ \mathbb {E} \left[ \frac {W _ {i} Y _ {i} (1)}{e (X _ {i})} \mid X _ {i} \right] - \mathbb {E} \left[ \frac {(1 - W _ {i}) Y _ {i} (0)}{1 - e (X _ {i})} \mid X _ {i} \right] \right] \\ = \mathbb {E} \left[ \frac {\mathbb {E} [ W _ {i} | X _ {i} ] \mathbb {E} [ Y _ {i} (1) | X _ {i} ]}{e (X _ {i})} - \frac {\mathbb {E} [ 1 - W _ {i} | X _ {i} ] \mathbb {E} [ Y _ {i} (0) | X _ {i} ]}{1 - e (X _ {i})} \right] (\mathrm{unconf.}) \\ = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \right] = \tau . \\ \end{array}
$$

接下来，在我们的 IID 抽样假设下，(2.13) 直接由 IID 平均值的中心极限定理得出，其中

$$
V _ {I P W ^ {*}} = \mathrm{Var} \left[ \frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})} \right],
$$

前提是该方差是有限的。接下来推导 $V _ { I P W ^ { * } }$ 的声称的替代表达式。为此，基于 (1.21) 中的记号，我们引入一个辅助函数

$$
c (x) = \mu_ {(0)} (x) + (1 - e (x)) \tau (x),
$$

并令 $\varepsilon _ { i } ( w ) = Y _ { i } ( w ) - \mu _ { ( w ) } ( X _ { i } )$ 。有了这些预备知识，我们展开

$$
\begin{array}{l} \frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})} \\ = \frac {W _ {i} (\mu_ {(1)} (X _ {i}) + \varepsilon_ {i} (1))}{e (X _ {i})} - \frac {(1 - W _ {i}) (\mu_ {(0)} (X _ {i}) + \varepsilon_ {i} (0))}{1 - e (X _ {i})} \\ = \tau (X _ {i}) + \left(\frac {W _ {i}}{e (X _ {i})} - \frac {1 - W _ {i}}{1 - e (X _ {i})}\right) c (X _ {i}) + \frac {W _ {i} \varepsilon_ {i} (1)}{e (X _ {i})} - \frac {(1 - W _ {i}) \varepsilon_ {i} (0)}{1 - e (X _ {i})}. \\ \end{array}
$$

此外，根据倾向得分的定义，有 E $\left[ W _ { i } / e ( X _ { i } ) - ( 1 - W _ { i } ) / ( 1 - e ( X _ { i } ) ) \bigm | X _ { i } \right] = 0$ ，并且根据无混杂性，有 E $\left[ \varepsilon _ { i } ( w ) \big | X _ { i } , W _ { i } \right] = 0$ ，所以

$$
\begin{array}{l} \operatorname{Var} \left[ \frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})} \right] = \operatorname{Var} [ \tau (X _ {i}) ] \\ + \mathbb {E} \left[ \left(\left(\frac {W _ {i}}{e (X _ {i})} - \frac {1 - W _ {i}}{1 - e (X _ {i})}\right) c (X _ {i})\right) ^ {2} \right] + \mathbb {E} \left[ \left(\frac {W _ {i} \varepsilon_ {i} (1)}{e (X _ {i})} - \frac {(1 - W _ {i}) \varepsilon_ {i} (0)}{1 - e (X _ {i})}\right) ^ {2} \right]. \\ \end{array}
$$

$V _ { I P W }$ * 的声称表达式通过简化上式得到。

![image_01](images/image_01.png)

上述证明中看似不经意地做出的一个值得注意的假设是，(2.13) 中使用的所有矩都是定义良好且有限的。然而，这是一个非常重要的假设。如果潜在结果是**一致有界**的，那么这个条件本质上等价于假设

$$
\mathbb {E} \left[ 1 / (e (X _ {i}) (1 - e (X _ {i}))) \right] <   \infty . \tag {2.14}
$$

同时，如果我们仅仅假设潜在结果具有有限的二阶矩，那么我们需要假设更强的条件，例如，存在某个 $\eta > 0$ 使得

$$
\eta \leq e (x) \leq 1 - \eta \text {   for   all   } x \in \mathcal {X}. \tag {2.15}
$$

这些假设通常被称为**重叠假设（overlap assumptions）**，它们编码了这样一个要求：在给定 $x$ 的条件下，处理分配必须存在非平凡的随机性。我们将 (2.14) 称为**弱重叠（weak overlap）**，将 (2.15) 称为**强重叠（strong overlap）**。从性质上讲，要使非参数处理效应估计成为可能，通常必须做出某种重叠类型的假设：如果处理分配 $W _ { i }$ 可以由 $X _ { i }$ 完美预测，那么处理分配中就没有实际的随机性，因此以处理随机化为理由的处理效应估计就不可能实现。

逆倾向得分加法的准确性如何？我们上面已经证明，当使用真实的倾向得分时，IPW 是无偏且渐近正态的；当使用估计的倾向得分时，它是相合的。考虑到 IPW 估计量简单的函数形式，这当然是一个不错的结果。但这些结果是否意味着 IPW 是好的呢？

为了给我们的 IPW 结果设定一个基准，重新审视本讲开头 $X$ 是离散的情况是有帮助的。在这种情况下，我们可以使用定理 2.1 中关于 $\scriptstyle { \hat { \tau } } _ { S T R A T }$ 的结果作为比较点。当倾向得分已知时，$\hat { \tau } _ { I P W } ^ { * }$ 和 $\scriptstyle { \hat { \tau } } _ { S T R A T }$ 都是渐近正态的，并且从 (2.4) 和 (2.13) 我们看出

$$
V _ {I P W ^ {*}} = V _ {S T R A T} + \mathbb {E} \left[ \frac {\left(\mu_ {(0)} (X _ {i}) + (1 - e (X _ {i})) \tau (X _ {i})\right) ^ {2}}{e (X _ {i}) (1 - e (X _ {i}))} \right]. \tag {2.16}
$$

因此，除非 $\mu _ { ( 0 ) } ( X _ { i } ) + ( 1 - e ( X _ { i } ) ) \tau ( X _ { i } )$ 几乎处处为零，否则 $\hat { \tau } _ { I P W } ^ { * }$ 的渐近方差严格大于 $\scriptstyle { \hat { \tau } } _ { S T R A T }$ 。同时，当倾向得分未知时，我们这里仅证明了 ${ \hat { \tau } } _ { I P W }$ 的相合性（没有中心极限定理），因此我们甚至无法进行适当的比较。因此，乍一看，比较定理 2.1 和 2.2 使得 IPW 的表现显得有些令人失望。

然而，仔细审视后，情况变得更加复杂：事实证明，$\scriptstyle { \hat { \tau } } _ { S T R A T }$ 实际上可以被理解为**使用特定选择的估计倾向得分 $\hat { e } ( x )$ 实现的 IPW 估计量**。在 (2.3) 中 $\scriptstyle { \hat { \tau } } _ { S T R A T }$ 定义良好的情况下，我们有：

$$
\hat {\tau} _ {S T R A T} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{\hat {e} (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})}\right), \quad \hat {e} (x) = \frac {n _ {x 1}}{n _ {x}}. \tag {2.17}
$$

因此，当 $\mathcal { X }$ 是离散时，一个可行的 IPW 估计量的实例，即 $\scriptstyle { \hat { \tau } } _ { S T R A T }$ ，实际上比“理想”IPW 估计量更精确（另见第 16 章的练习 1）。13 理解和解决这个看似矛盾的地方，将是理解如何设计在无混杂性下平均处理效应的精确估计量的核心——包括连续协变量的情况。

**随机研究与观察性研究（Randomized and observational studies）** 我们忽略的一个细微差别是，存在两种概念上不同的方式可以使潜在结果满足 (2.5)。第一种选择是数据是由一个具有可变处理倾向的实验生成的：自然生成了 $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \} \sim P$ ，然后一个实验者根据协变量的某个函数 $e ( \cdot )$ ，随机分配处理 $W _ { i } \sim \mathrm { B e r n o u l l i } ( e ( X _ { i } ) )$ 。在这种设定下，实验者知道 (2.5) 必须成立，因为他们自己生成处理的方式满足该假设。本质上，实验者运行的是与 (1.8) 中相同的伯努利试验，只是随机化概率随 $X _ { i }$ 变化。尽管依赖于协变量的随机化概率需要统计上的调整，但这类实验在概念上与第 1 章讨论的实验类似——并且提供了同样强有力的、黄金标准的因果证据。

**例 2.** Arceneaux, Gerber, and Green [2006] 进行了一项随机研究，以衡量选民动员电话在中期选举中促使人们投票的有效性。该研究在密歇根州和爱荷华州两个州进行，随机化按州和国会选区的竞争程度进行分层，每层的随机化概率从 1% 到 15% 不等。这是一项随机对照试验；然而，为了进行有效的分析，需要适当地考虑随机化概率的变化（例如，通过倾向得分分层），而简单地取全局均值差则容易产生**辛普森悖论（Simpson's paradox）**。

第二种选择是不存在实验：自然生成了 $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , W _ { i } \} \sim P$ ，而我们仅仅假定 (2.5) 成立。这标志着与第 1 章设定的一个更大的背离。没有运行实验的分析者；相反，我们假定数据的生成方式**仿佛**有人运行了前一章描述的实验。这种设定被称为**自然实验（natural experiments）**或**观察性研究设计（observational study designs）**。由于实际上并未进行实验，在观察性研究中，假设 (2.5) 总是可以被质疑——因此，由此产生的因果证据有时被认为比通过随机实验获得的证据更具试探性。

**例 3.** LaLonde [1986] 考虑通过比较参加试点计划的人与未参加该计划的普通公众在干预后的收入，来评估一个就业培训计划的收益。这不是一个随机研究设计，普通公众与试点计划中的参与者在许多干预前指标上存在差异。LaLonde [1986] 关于从这类观察性数据中获得可靠因果估计的可能性的初步评估是悲观的。然而，在后来的工作中，Dehejia and Wahba [1999] 表明，从对倾向得分（即，这里指给定干预前特征加入试点计划的概率）建模开始的方法表现出了更有希望的行为，14 并且通常能够匹配实验基准。

具有协变量依赖随机化的随机试验与观察性研究之间的另一个主要实际区别是，在前者中，处理倾向 $e ( X _ { i } )$ 通常是已知的（因为它们由实验者选择），因此像定理 2.2 中那样具有保证的理想 IPW 方法是可用的。相比之下，在观察性研究设定中，处理倾向需要被估计，因此方法对倾向得分误差的**稳健性**很重要——特别是在像下面这样倾向得分难以精确估计的设定中。到目前为止，我们还没有看到在连续 $X _ { i }$ 的设定下，能够接收估计的倾向得分并输出具有 $1 / { \sqrt { n } }$ 尺度误差的渐近正态平均处理效应估计的估计量。在下一章中，我们将提出一个对 IPW 的改进，即使使用估计的倾向得分也能实现渐近正态性。

**例 4.** Ross et al. [2024] 使用退伍军人事务部的电子健康记录数据，来评估精神科住院治疗对近期有自杀企图或自杀意念的患者预防自杀的效果。这里没有随机化，住院与非住院患者在治疗前特征上存在差异。作者认为，在控制了通过电子健康记录获得的丰富病史后，无混杂性假设是合理的，并进而使用了倾向得分方法。然而，鉴于治疗前变量是高维且结构复杂的，有必要使用机器学习方法来获得合理的倾向得分估计——并且任何后续使用这些倾向得分的方法都应对此步骤中可能出现的估计误差具有稳健性。

## 2.3 文献注释（Bibliographic notes）

倾向得分在估计因果效应中的核心作用最早由 Rosenbaum and Rubin [1983] 强调，而相关的估计方法如倾向得分分层则在 Rosenbaum and Rubin [1984] 中讨论。Hirano, Imbens, and Ridder [2003] 提供了对 IPW 型估计量渐近性质的详细讨论，扩展了定理 2.1 中给出的结果。特别是，他们提出了在连续 $X _ { i }$ 条件下，使用非参数估计的倾向得分的 IPW 可以优于理想 IPW 的条件。

在实践中利用倾向得分的另一种流行方法是**倾向得分匹配（propensity matching）**，即通过比较具有相似 $\hat{e}(X_i)$ 值的单元对来估计处理效应。关于因果推断中匹配方法的一些近期讨论，请参见 Abadie and Imbens [2006, 2016], Diamond and Sekhon [2013], Zubizarreta [2012] 及其参考文献。
# 第7章 平衡估计量（Balancing Estimators）

**倾向得分（propensity score）** 在我们迄今为止的论述中扮演了核心角色，包括在理解无混淆假设下的平均处理效应识别、构建平均处理效应的高效估计量以及自适应实验的设计等方面。然而，尽管这些论述清楚地表明倾向得分对于因果推断至关重要，但其为何如此重要的原因可能仍有些模糊不清。

在此，我们将重新审视作为统计对象的倾向得分，并论证倾向得分的一个关键功能是**平衡**——从而消除——由观测到的预处理混杂变量所引入的偏差。这一视角将激励开发新的倾向得分估计量，这些估计量在用于处理效应估计时具有更好的端到端行为，并阐明无混淆假设下平均处理效应估计方法与更广泛的非参数和/或高维推断文献之间的联系。请注意，本章不会涉及因果推断中的任何新任务——相反，我们将聚焦于无混淆假设下的平均处理效应估计问题，并重新审视该任务背后的统计原理。因此，本章可在初读时跳过。

### 平衡的作用（The role of balance）

在我们熟悉的第3章的基本无混淆设定下，回顾平均处理效应（ATE）的（理想化）**逆概率加权（inverse-propensity weighted, IPW）** 估计量：

$$
\hat {\tau} _ {I P W} ^ {*} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})}\right), \quad e (x) = \mathbb {P} \left[ W _ {i} = 1 \mid X _ {i} = x \right]. \tag {7.1}
$$

在第2章中，我们证明了理想化的IPW估计量对于ATE是无偏的，即 $\mathbb {E} \left[ \hat { \tau } _ { I P W } ^ { * } \right] = \tau$ ，其中 $\tau = \mathbb { E } \left[ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \right]$ 。定理2.2中给出的证明是条件独立性和期望链式法则的一个抽象应用，直接蕴含了无偏性。

为了更深入地理解倾向得分的统计功能，我们从一个不那么优雅但更具算法明确性的论证开始，重新审视IPW的无偏性。为此，假设我们可以将条件期望函数 $\mu _ { ( w ) } ( x )$ 用基展开的形式表示，即：40

$$
\mu_ {(w)} (x) = \sum_ {j = 1} ^ {\infty} \beta_ {j} (w) \psi_ {j} (x) \tag {7.2}
$$

其中 $\psi _ { j } ( \cdot )$ 是一组预先定义的基函数。在合理的正则性条件下（并假设无混淆性），我们有：

$$
\tau = m _ {(1)} - m _ {(0)}, \quad m _ {(w)} = \sum_ {j = 1} ^ {\infty} \beta_ {j} (w)   \mathbb {E} \left[ \psi_ {j} (X _ {i}) \right]. \tag {7.3}
$$

基于这一设定，我们可以如下论证IPW的无偏性。在无混淆假设下， $Y _ { i } = \mu _ { ( W _ { i } ) } ( X _ { i } ) + \varepsilon _ { i }$ 且 $\mathbb{E} \left[ \varepsilon _ { i } \big | X _ { i } , W _ { i } \right] = 0$ ，因此（同样在正则性条件下）：

$$
\mathbb {E} \left[ \frac {W _ {i} Y _ {i}}{e \left(X _ {i}\right)} \right] = \mathbb {E} \left[ \frac {W _ {i}}{e \left(X _ {i}\right)} \sum_ {j = 1} ^ {\infty} \beta_ {j} (w) \psi_ {j} \left(X _ {i}\right) \right] + \mathbb {E} \left[ \frac {W _ {i} \varepsilon_ {i}}{e \left(X _ {i}\right)} \right] \tag {7.4}
$$

$$
= \sum_ {j = 1} ^ {\infty} \beta_ {j} (w) \mathbb {E} \left[ \frac {W _ {i} \psi_ {j} (X _ {i})}{e (X _ {i})} \right] = \sum_ {j = 1} ^ {\infty} \beta_ {j} (w) \mathbb {E} \left[ \psi_ {j} (X _ {i}) \right] = m _ {(1)},
$$

类似地， $\mathbb{E} \left[ ( 1 - W _ { i } ) Y _ { i } / ( 1 - e ( X _ { i } ) ) \right] = m _ { ( 0 ) }$ 。这一论证揭示了IPW的工作原理：通过对处理组和对照组样本进行重新加权，使得基函数 $\psi _ { j } ( X _ { i } )$ 的加权平均值精确匹配相应的总体平均值。

### 总体平衡与样本平衡（Population vs. sample balance）

理想化的IPW通过为所有基函数 $\psi _ { j } ( X _ { i } )$ 在处理组和对照组之间创建**总体平衡（population balance）** 来实现无偏性：

$$
\mathbb {E} \left[ \frac {W _ {i}   \psi_ {j} (X _ {i})}{e (X _ {i})} \right] = \mathbb {E} \left[ \psi_ {j} (X _ {i}) \right], \quad \mathbb {E} \left[ \frac {(1 - W _ {i})   \psi_ {j} (X _ {i})}{1 - e (X _ {i})} \right] = \mathbb {E} \left[ \psi_ {j} (X _ {i}) \right]. \tag {7.5}
$$

在实践中，我们需要处理有限样本并估计倾向得分。然而，根据(7.5)，如果样本量 $n$ 足够大且倾向得分估计值 $\hat { e } ( X _ { i } )$ 足够精确，那么我们有希望实现近似的**样本平衡（sample balance）**：

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i} \psi_ {j} (X _ {i})}{\hat {e} (X _ {i})} \approx \frac {1}{n} \sum_ {i = 1} ^ {n} \psi_ {j} (X _ {i}), \tag {7.6}
$$

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - W _ {i}) \psi_ {j} (X _ {i})}{1 - \hat {e} (X _ {i})} \approx \frac {1}{n} \sum_ {i = 1} ^ {n} \psi_ {j} (X _ {i}),
$$

而这种样本平衡反过来又能保证IPW的一致性。这类论证可用于证明，对于多种一致的倾向得分估计量 $\hat { e } ( X _ { i } )$ ，IPW都是一致的。

然而，上述论证极为粗略。一方面，我们声称IPW通过为 $\psi _ { j } ( X _ { i } )$ 创建平衡来实现一致性；但另一方面，上述论证又让样本平衡(7.6)作为一致倾向得分估计的间接结果而出现。如果我们相信良好的样本平衡很重要，难道我们不应该在如何估计倾向得分以及如何像(7.6)那样优化样本平衡上投入更多思考吗？这个问题的答案是肯定的；而源于寻求回答这一问题的**协变量平衡倾向得分（covariate-balancing propensity score）** 方法，为不考虑平衡的基本IPW方法提供了重大改进。

## 7.1 协变量平衡倾向得分（Covariate-balancing propensity scores）

我们首先考虑针对有限维参数设定下的目标协变量平衡而量身定制的倾向得分方法。假设 $X _ { i } \in \mathbb { R } ^ { p }$ 取值于一个有限维空间，并且我们有一个线性结果模型 $\mu _ { ( w ) } ( x ) = x \cdot \beta ( w )$ 和一个逻辑斯蒂倾向模型 $e ( x ) = 1 / ( 1 + e ^ { - x \cdot \theta } )$ 。由于我们有线性结果模型，实现样本平衡仅涉及平衡原始协变量 $X _ { i }$ 。

样本平衡条件(7.6)涉及"≈"关系，为了继续推进，我们需要对其进行明确化。在这里，由于我们处于低维设定下，要求精确平衡是合理的，即要求(7.6)以等式成立。那么，使用我们的逻辑斯蒂设定 $\hat { e } ( x ) = 1 / ( 1 + e ^ { - x \cdot \hat { \theta } } )$ ，(7.6)变为：

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 + e ^ {- X _ {i} \hat {\theta}}\right) W _ {i} X _ {i} = \frac {1}{n} \sum_ {i = 1} ^ {n} X _ {i}, \tag {7.7}
$$

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 + e ^ {X _ {i} \hat {\theta}}\right) (1 - W _ {i}) X _ {i} = \frac {1}{n} \sum_ {i = 1} ^ {n} X _ {i}. \tag {7.8}
$$

我们能否为倾向模型学习一个参数向量 $\hat { \theta }$ ，使得平衡条件(7.7)和(7.8)成立？

这些平衡条件是非线性方程组，乍看之下可能难以求解。然而，事实证明——在非退化条件下——(7.7)的解可以等价地写成以下凸最小化问题的最优解：

$$
\hat {\theta} = \operatorname{argmin} _ {\theta} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \ell_ {\theta} ^ {(1)} \left(X _ {i}, Y _ {i}, W _ {i}\right) \right\}, \tag {7.9}
$$

$$
\ell_ {\theta} ^ {(1)} (X _ {i}, Y _ {i}, W _ {i}) = W _ {i} e ^ {- X _ {i} \theta} + (1 - W _ {i}) X _ {i} \theta ,
$$

因此可以通过牛顿下降法等数值方法轻松求解。同时，(7.8)的解等价于：

$$
\hat {\theta} = \operatorname{argmin} _ {\theta} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \ell_ {\theta} ^ {(0)} \left(X _ {i}, Y _ {i}, W _ {i}\right) \right\}, \tag {7.10}
$$

$$
\ell_ {\theta} ^ {(0)} (X _ {i}, Y _ {i}, W _ {i}) = (1 - W _ {i}) e ^ {X _ {i} \theta} - W _ {i} X _ {i} \theta .
$$

这里的一个微妙之处在于，我们可能希望找到一个同时求解(7.7)和(7.8)的参数向量 $\hat { \theta }$ 。然而，这通常是不可能的（因为需要用 $p$ 个自由参数求解 $2p$ 个方程），但也不是必需的：如果倾向模型的作用仅仅是创建平衡，那么如果方便的话，没有强有力的理由反对在单个ATE估计量的上下文中使用两个不同的倾向模型。

综合所有要素，构建一个 **平均处理效应（ATE）** 的**逆概率加权（IPW）**估计量，便得到了**协变量平衡倾向性评分（CBPS）**估计量：

$$
\hat {\theta} _ {(w)} = \operatorname{argmin} _ {\theta} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \ell_ {\theta} ^ {(w)} \left(X _ {i}, Y _ {i}, W _ {i}\right) \right\}, \quad \text { for } \quad w = 0, 1 \tag {7.11}
$$

$$
\hat {\tau} _ {C B P S} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 + e ^ {- X _ {i} \hat {\theta} _ {(1)}}\right) W _ {i} Y _ {i} - \frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 + e ^ {X _ {i} \hat {\theta} _ {(0)}}\right) (1 - W _ {i}) Y _ {i}.
$$

以下结果证明，与无偏但方差不必要地大的**理想逆概率加权（oracle IPW）**估计量（定理 2.2），或一致但收敛速度不一定好的**通用逆概率加权（generic IPW）**估计量（使用估计的倾向性评分）不同，上述 CBPS 估计量具有优异的统计性质：它是 $\sqrt{n}$ 一致的，且其抽样分布是渐近正态的，并且达到了与第 3 章研究的**增强逆概率加权（AIPW）**估计量相同的渐近方差。

**定理 7.1.** 我们有样本 $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , W _ { i } \} \stackrel { i i d } { \sim } P$，取值于 $\mathbb { R } ^ { p } \times \mathbb { R } \times \mathbb { R } \times \{ 0 , 1 \}$，使得我们能够观测到 $( X _ { i } , Y _ { i } , W _ { i } )$，其中 $Y _ { i } = Y _ { i } ( W _ { i } )$，并且**无混杂性（unconfoundedness）**成立，即 $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \} \perp W _ { i } \mid X _ { i }$。假设存在 $c > 0$，使得以下指数矩是有限的，41

$$
\mathbb {E} \left[ \frac {e ^ {c \| X _ {i} \| _ {2}}}{e (X _ {i})} \right] <   \infty , \quad \mathbb {E} \left[ \frac {e ^ {c \| X _ {i} \| _ {2}}}{1 - e (X _ {i})} \right] <   \infty , \tag {7.12}
$$

并且特征协方差矩阵是满秩的，即 $\mathbb {E} \left[ X _ { i } ^ { \otimes 2 } \right] \succ 0$。进一步假设，线性结果模型 $\bar { \mu _ { ( w ) } ( x ) } = \bar { x } \cdot \beta ( w )$ 和逻辑斯蒂倾向性模型 $e ( x ) = 1 / ( 1 + e ^ { - x \cdot \theta } )$ 都是正确设定的，其中 $\left\| \theta \right\| _ { 2 } < \infty$，并且条件方差 $\sigma _ { w } ^ { 2 } ( x ) = \operatorname { V a r } \left[ Y _ { i } ( w ) \big | X _ { i } = x \right]$ 一致有界，即 $\sigma _ { w } ^ { 2 } ( x ) \le M$。那么 $\hat { \tau } _ { C B P S }$ 是一致的，并且

$$
\sqrt {n} \left(\hat {\tau} _ {C B P S} - \tau\right) \Rightarrow \mathcal {N} \left(0, \operatorname{Var} \left[ \tau (X _ {i}) \right] + \mathbb {E} \left[ \frac {\sigma_ {0} ^ {2} (X _ {i})}{1 - e (X _ {i})} + \frac {\sigma_ {1} ^ {2} (X _ {i})}{e (X _ {i})} \right]\right). \tag {7.13}
$$

**证明.** 我们从考察上述损失函数 $\ell _ { \theta } ^ { ( 1 ) } ( x , y , w )$ 及其期望开始：

$$
L _ {(1)} (\theta) = \mathbb {E} \left[ \ell_ {\theta} ^ {(1)} \left(X _ {i}, Y _ {i}, W _ {i}\right) \right].
$$

对 $\ell _ { \theta } ^ { ( 0 ) } ( x , y , w )$ 和 $L _ { ( 0 ) } ( \cdot )$ 的分析本质上是相同的，因此此处不再赘述。首先，注意到：

$$
\nabla^ {2} \ell_ {\theta} ^ {(1)} (x, y, w) = w e ^ {- \theta \cdot x} x ^ {\otimes 2} \succeq 0,
$$

即这些损失函数是凸的，正如所述。接下来，假设逻辑斯蒂倾向性模型是正确设定的（真实参数值为 $\theta$），我们看到对于任何 $\theta ^ { \prime }$：

$$
L _ {(1)} (\theta^ {\prime}) = \mathbb {E} \left[ \frac {e ^ {- X _ {i} \theta}}{1 + e ^ {- X _ {i} \theta}} e ^ {X _ {i} (\theta - \theta^ {\prime})} + \frac {1}{1 + e ^ {X _ {i} \theta}} X _ {i} \theta^ {\prime} \right],
$$

由于 (7.12) 保证了 $\mathbb {E} \left[ e ^ { c \| x \| _ { 2 } } \right] < \infty$，因此对于任何满足 $\| \theta - \theta ^ { \prime } \| _ { 2 } \leq c$ 的 $\theta ^ { \prime }$，该期望是有限的。最后，在真实参数值 $\theta$ 处，42

$$
\nabla L _ {(1)} (\theta) = 0, \quad \nabla^ {2} L _ {(1)} (\theta) = \mathbb {E} \left[ e (X _ {i}) X _ {i} ^ {\otimes 2} \right] \succ 0,
$$

即 $\theta$ 实际上是 $L _ { ( 1 ) } ( \cdot )$ 的一个最小值点；并且，由于 $\ell _ { \theta } ^ { ( 1 ) }$ 的凸性以及在 $\theta$ 处的强凸性，它是 $L _ { ( 1 ) } ( \cdot )$ 的唯一最小值点。

基于这些预备知识，我们可以使用凸经验风险最小化的标准结果 [例如，Van der Vaart, 1998，定理 5.7 和例 19.8] 来验证 $\hat { \theta } _ { ( 1 ) }$ 是一致的，即 $\hat { \theta } _ { ( 1 ) } \to _ { p } \theta$。因此，特别地，我们看到 $\hat { \theta } _ { ( 1 ) }$ 必须以趋近于 1 的概率是有限的。因此，它（以趋近于 1 的概率）必须是损失函数的一个临界点：

$$
\nabla \left(\frac {1}{n} \sum_ {i = 1} ^ {n} W _ {i} e ^ {- X _ {i} \hat {\theta} _ {(1)}} + (1 - W _ {i}) X _ {i} \hat {\theta} _ {(1)}\right) = 0,
$$

这反过来等价于 $\hat { \theta } _ { ( 1 ) }$ 求解方程 (7.7)。

对 $\hat { \theta } _ { ( 0 ) }$ 应用类似的分析，并将这些平衡条件代入 (7.11)，我们可以利用线性结果模型的正确设定性来验证，在 $\hat { \theta } _ { ( 1 ) }$ 求解 (7.7) 且 $\hat { \theta } _ { ( 0 ) }$ 求解 (7.8) 的这个概率趋近于 1 的事件上，有：

$$
\begin{array}{l} \hat {\tau} _ {C B P S} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(X _ {i} \left(\beta_ {(1)} - \beta_ {(0)}\right) + (2 W _ {i} - 1) \left(1 + e ^ {- (2 W _ {i} - 1) X _ {i} \hat {\theta} _ {(W _ {i})}}\right) \varepsilon_ {i}\right), \\ = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\tau (X _ {i}) + \frac {W _ {i}}{e (X _ {i})} \varepsilon_ {i} - \frac {1 - W _ {i}}{1 - e (X _ {i})} \varepsilon_ {i}\right) \\ + \frac {1}{n} \sum_ {i = 1} ^ {n} \left(e ^ {- X _ {i} \hat {\theta} _ {(1)}} - e ^ {- X _ {i} \theta}\right) W _ {i} \varepsilon_ {i} - \frac {1}{n} \sum_ {i = 1} ^ {n} \left(e ^ {X _ {i} \hat {\theta} _ {(0)}} - e ^ {X _ {i} \theta}\right) (1 - W _ {i}) \varepsilon_ {i}, \\ \end{array}
$$

其中 $\varepsilon _ { i } = Y _ { i } - X _ { i } \beta _ { ( W _ { i } ) }$。现在，上式中的第一个求和项是我们之前讨论中（例如第 2 章）熟悉的，并且满足 (7.13)。

剩下的工作是检查最后两项在 $1 / \sqrt { n }$ 尺度上是渐近可忽略的。为此，注意到，在给定 $\{ X _ { i } , W _ { i } \}$（因此也给定 $\hat { \theta } _ { ( w ) }$）的条件下，该项的均值为零，并且：

$$
\begin{array}{l} n \mathbb {E} \left[ \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \left(e ^ {- X _ {i} \hat {\theta} _ {(1)}} - e ^ {- X _ {i} \theta}\right) W _ {i} \varepsilon_ {i}\right) ^ {2} | \{X _ {i}, W _ {i} \} \right] \\ = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(e ^ {- X _ {i} \hat {\theta} _ {(1)}} - e ^ {- X _ {i} \theta}\right) ^ {2} W _ {i} \sigma_ {1} ^ {2} (X _ {i}) \\ \leq \frac {M}{n} \sum_ {i = 1} ^ {n} \left(e ^ {- X _ {i} \hat {\theta} _ {(1)}} - e ^ {- X _ {i} \theta}\right) ^ {2} W _ {i} \\ = \frac {M}{n} \sum_ {i = 1} ^ {n} \left(e ^ {X _ {i} (\theta - \hat {\theta} _ {(1)})} - 1\right) ^ {2} e ^ {- 2 X _ {i} \theta} W _ {i}. \\ \end{array}
$$

我们知道，根据一致性，对于任何 $\delta > 0$，$\lVert \theta - \hat { \theta } _ { ( 1 ) } \rVert _ { 2 } \leq \delta / 2$ 的概率趋近于 1。因此，同样以趋近于 1 的概率，上述表达式被界定为：

$$
\begin{array}{l} \dots \leq \frac {2 M}{n} \sum_ {i = 1} ^ {n} \left(e ^ {\delta \| X _ {i} \| _ {2}} + 1\right) e ^ {- 2 X _ {i} \theta} W _ {i} \\ = \mathcal {O} _ {P} \left(\mathbb {E} \left[ \left(e ^ {\delta \| X _ {i} \| _ {2}} + 1\right) e ^ {- 2 X _ {i} \theta} / \left(1 + e ^ {- X _ {i} \theta}\right) \right]\right) \\ = \mathcal {O} _ {P} \left(\mathbb {E} \left[ e ^ {\delta \| X _ {i} \| _ {2}} \left(1 + e ^ {- X _ {i} \theta}\right) \right]\right), \\ \end{array}
$$

其中，上述步骤中，第二行使用了马尔可夫不等式，第三行则通过直接的代数运算得到。根据 (7.12)，对于任何 $\delta \leq c$，该表达式是有限的；并且随着 $\delta \to 0$，由连续性可知其趋于 0。因此，根据 $\hat { \theta } _ { ( 1 ) }$ 的一致性，有：

$$
n \mathbb {E} \left[\left(\frac {1}{n} \sum_ {i = 1} ^ {n} \left(e ^ {- X _ {i} \hat {\theta} _ {(1)}} - e ^ {- X _ {i} \theta}\right) W _ {i} \varepsilon_ {i}\right) ^ {2} | \{X _ {i}, W _ {i} \} \right]\rightarrow_ {p} 0,
$$

因此，根据切比雪夫不等式，该项如我们试图证明的那样，处于 $1 / \sqrt { n }$ 尺度。对涉及 $\hat { \theta } _ { ( 0 ) }$ 的项应用类似的论证，即可完成证明。□

因此，如果我们相信线性-逻辑斯蒂（linear-logistic）设定，并希望使用 IPW 估计量，那么我们应该通过最小化协变量平衡损失函数来学习倾向性模型，而不是使用逻辑斯蒂回归中通常使用的最大似然损失。从估计逻辑斯蒂回归参数 $\theta$ 的角度来看，最大似然是渐近最优的，但这并非此处关注的重点。当通过 IPW 估计 ATE 时，我们从逆倾向性权重中需要的是它们能够如 (7.6) 那样创建平衡；而当我们使用直接针对这一目标的协变量平衡倾向性评分时，我们就能通过 IPW 获得良好结果。

第 16 章的练习 8 对上述结果进行了扩展，并确立了 $\hat { \tau } _ { C B P S }$ 的**双稳健性（double-robustness）**性质，该性质在线性模型或逻辑斯蒂模型中仅有一个被正确设定时仍然成立。练习 9 研究了一个针对**处理组平均处理效应（average treatment effect on the treated）**的协变量平衡倾向性评分估计量。

**注 7.1.** 估计量 (7.11) 并不是本书中遇到的第一个协变量平衡倾向性评分估计量。在第 2 章中，我们考虑了特征空间 $\mathcal{X}$ 是离散的情况，并发现自然的分层估计量 $\hat { \tau } _ { S T R A T }$ 可以被解释为一个 IPW 估计量，它通过巧妙地选择估计的倾向性评分实现了有效的大样本行为；参见定理 2.1 和 (2.17)。进一步考察表明，$\hat { \tau } _ { S T R A T }$ 背后的倾向性评分对于所有 $x \in \mathcal { X }$ 实现了指示变量 $1( \{ X _ { i } = x \} )$ 的精确样本平衡，并且对于饱和模型，$\hat { \tau } _ { S T R A T }$ 等价于 $\hat { \tau } _ { C B P S }$。因此，从概念上讲，我们可以将协变量平衡倾向性评分方法视为当 $X$ 取连续值时，分层处理效应估计的自然推广。

## 7.2 近似平衡与增广估计量（Approximate balance and augmented estimators）

我们在上文已确定，在低维参数设定下，针对如 (7.7) 和 (7.8) 中所述**精确有限样本平衡**的倾向性得分方法具有若干良好的统计性质。然而，在某些情况下，实现精确平衡可能并不现实。在一些现代应用中，协变量 $X _ { i } ~ \in ~ \mathbb { R } ^ { p }$ 可能取值于一个高维空间，且 $p \gg n$（例如，$X _ { i }$ 可能代表患者的基因组）；在这种情况下，通常无法找到一组在 $n$ 个样本上的权重来精确求解 $p$ 个协变量平衡矩条件。或者，如同我们的激励性示例 7.2，我们可能关注于使用无限筛（infinite sieve）来逼近非参数函数的情形，此时我们需要处理无限多个协变量平衡矩条件。

值得庆幸的是，即使无法实现精确平衡，我们仍然可以通过旨在实现**近似平衡**的倾向性得分方法来获得良好的结果：

$$
\sup _ {j = 1, 2, \dots} \left| \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i} \psi_ {j} (X _ {i})}{\hat {e} (X _ {i})} - \psi_ {j} (X _ {i}) \right| \leq t, \tag {7.14}
$$

$$
\sup _ {j = 1, 2, \dots} \left| \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - W _ {i})   \psi_ {j} (X _ {i})}{1 - \hat {e} (X _ {i})} - \psi_ {j} (X _ {i}) \right| \leq t,
$$

其中 $t$ 是一个小的容忍参数。当处理近似平衡时，之前考虑的普通**逆概率加权（IPW）** 型估计量可能会受到偏倚的主导而不再表现良好；然而，使用**增广逆概率加权（AIPW）** 型估计量可以解决这个问题。增广估计量有助于处理近似平衡的原因与第 3 章讨论的 AIPW 的（强）双重稳健性密切相关：一个足够精确的回归调整可以减轻由于非精确平衡带来的偏倚，同时不会引入过多的误差。

对高维和/或非参数处理效应估计问题的近似平衡方法进行全面综述超出了本演讲的范围。相反，我们将在此总结一种针对具有稀疏线性结果模型的高维设定方法，并在章末提供进一步阅读的参考文献。

假设第 3 章的基本**无混杂性（unconfoundedness）** 模型成立，且包含高维控制变量 $X _ { i } \in \mathbb { R } ^ { p }$，其中 $p$ 可能远大于 $n$。进一步假设结果模型是稀疏且线性的，即 $\mu _ { ( w ) } ( x ) = x \cdot \beta _ { ( w ) }$，且 $\| \beta _ { ( w ) } \| _ { 0 } \leq k$，其中 $k$ 是非零参数数量的一个合理小上界，$\lVert \boldsymbol { v } \rVert _ { 0 }$ 统计向量 $\boldsymbol{v}$ 中非零元素的个数。注意，我们在此未对倾向性模型做任何参数假设，仅假设强重叠性（strong overlap） $\eta \leq e ( X _ { i } ) \leq 1 - \eta$ 成立。

基于此设定，Athey, Imbens, 和 Wager [2018b] 考虑通过直接最小化一个近似平衡准则来学习权重 $\hat { \gamma } _ { i }$：

$$
\hat {\gamma} ^ {(1)} = \operatorname{argmin} _ {\substack {\gamma_ {i} \geq 0, t \geq 0 \\ | 1, n}} \frac {1}{n} \sum_ {W _ {i} = 1} \gamma_ {i} ^ {2} + \zeta n t ^ {2} \tag{7.15}
$$

${ \mathrm { s u b j e c t ~ t o } } \left| { \frac { 1 } { n } } \sum _ { i = 1 } \left( \gamma _ { i } W _ { i } - 1 \right) X _ { i } \right| \leq t { \mathrm { ~ f o r ~ a l l ~ } } j = 1 , \ldots , p ,$

并且 $\hat { \gamma } _ { ( 0 ) }$ 可以类似地推导。从概念上讲，我们可以将这些权重解释为 ${ } ^ { \mathfrak { a } } 1 / \hat { e } ( X _ { i } ) = \hat { \gamma } _ { i } ^ { ( 1 ) , }$ 隐式的倾向性模型。然后，我们可以使用这些近似平衡权重，仿照 AIPW 构造推导出一个**增广平衡估计量**：

$$
\hat {\tau} _ {A B} = \frac {1}{n} \sum_ {i = 1} ^ {n} X _ {i} \left(\hat {\beta} _ {(1)} - \hat {\beta} _ {(0)}\right) + W _ {i} \hat {\gamma} _ {i} ^ {(1)} \left(Y _ {i} - X _ {i} \hat {\beta} _ {(1)}\right) \tag {7.16}
$$

$$
- (1 - W _ {i}) \hat {\gamma} _ {i} ^ {(0)} \left(Y _ {i} - X _ {i} \hat {\beta} _ {(0)}\right),
$$

其中 $\hat { \beta } _ { ( w ) }$ 是通过某种适用于稀疏高维数据的方法（如**套索（lasso）** [Tibshirani, 1996]）估计的。该构造背后的关键动机是以下引理。

**引理 7.2.** 在无混杂性和 SUTVA 下，进一步假设 $\mu _ { ( w ) } ( x ) = x \cdot \beta _ { ( w ) }$，并且 $\hat { \beta } _ { ( w ) }$ 是 $\beta _ { ( w ) }$ 的一个估计量，其 $L _ { 1 }$ 范数估计误差以 $C _ { ( w ) } ~ f o r ~ w = 0 , 1$ 为界：

$$
\left\| \hat {\beta} _ {(w)} - \beta_ {(w)} \right\| _ {1} \leq C _ {(w)}, \quad \| v \| _ {1} = \sum_ {j = 1} ^ {p} | v _ {j} |. \tag {7.17}
$$

那么，增广平衡估计量 (7.16) 满足：

$$
\hat {\tau} _ {A B} = \frac {1}{n} \sum_ {i = 1} ^ {n} X _ {i} \left(\beta_ {(1)} - \beta_ {(0)}\right) + W _ {i} \hat {\gamma} _ {i} ^ {(1)} \varepsilon_ {i} - (1 - W _ {i}) \hat {\gamma} _ {i} ^ {(0)} \varepsilon_ {i} + E, \tag {7.18}
$$

$$
| E | \leq C _ {(0)} \hat {t} ^ {(0)} + C _ {(1)} \hat {t} ^ {(1)},
$$

其中 $\hat { t } ^ { ( w ) }$ 是优化问题 (7.15) 解中的偏倚参数，且 $\varepsilon _ { i } = Y _ { i } - X _ { i } \beta _ { ( W _ { i } ) }$ .

**证明.** 由于 $\mu _ { ( w ) } ( x )$ 的线性性质，我们立即得到 (7.18) 的第一行成立，且误差项为：

$$
\begin{array}{l} E = \frac {1}{n} \sum_ {i = 1} ^ {n} X _ {i} \left(\hat {\beta} _ {(1)} - \hat {\beta} _ {(0)}\right) - X _ {i} \left(\beta_ {(1)} - \beta_ {(0)}\right) \\ + W _ {i} \hat {\gamma} _ {i} ^ {(1)} X _ {i} (\beta_ {(1)} - \hat {\beta} _ {(1)}) - (1 - W _ {i}) \hat {\gamma} _ {i} ^ {(0)} X _ {i} (\beta_ {(0)} - \hat {\beta} _ {(0)}) \\ = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 - W _ {i} \hat {\gamma} _ {i} ^ {(1)}\right) X _ {i} \left(\hat {\beta} _ {(1)} - \beta_ {(1)}\right) \\ - \frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 - (1 - W _ {i}) \hat {\gamma} _ {i} ^ {(0)}\right) X _ {i} \left(\hat {\beta} _ {(0)} - \beta_ {(0)}\right) \\ \end{array}
$$

应用 Hölder 不等式可得：

$$
\begin{array}{l} | E | \leq \left\| \frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 - W _ {i} \hat {\gamma} _ {i} ^ {(1)}\right) X _ {i} \right\| _ {\infty} \left\| \hat {\beta} _ {(1)} - \beta_ {(1)} \right\| _ {1} \\ + \left\| \frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 - (1 - W _ {i}) \hat {\gamma} _ {i} ^ {(0)}\right) X _ {i} \right\| _ {\infty} \left\| \hat {\beta} _ {(0)} - \beta_ {(0)} \right\| _ {1}, \\ \end{array}
$$

这等价于我们想要证明的界。 □

其要点是，忽略误差项 $E$，(7.18) 中给出的 $\hat { \tau } _ { A B }$ 表达式具有与第 3 章中 ATE 的有效估计量相似的形式。因此，如果我们能证明 $E$ 在 $1 / \sqrt { n }$ 尺度上可忽略，这个结果强烈表明 $\hat { \tau } _ { A B }$ 应具有良好的统计表现。一个超出本演讲范围的细节是精确刻画 $\hat { \gamma } ^ { ( w ) }$ 收敛到的极限是什么 ${ } ^ { ; 4 3 }$；然而，一个简单的观察是，如果我们能控制 $\hat { \gamma } ^ { ( w ) }$ 的平均二阶矩（如下文所述），那么 (7.18) 结合误差界 $| E | \ll 1 / \sqrt { n }$ 意味着 $\hat { \tau } _ { A B }$ 是 $\sqrt { n }$ 一致的且渐近无偏的。

现在剩下的工作是建立 $E$ 有界的条件。在一个广泛使用的关于协变量分布的假设——“**限制性特征值条件（restricted eigenvalue condition）**”下，并且在稀疏性界 $\| \beta _ { ( w ) } \| _ { 0 } \leq k$（即假设真实参数向量至多有 $k$ 个非零项）下，套索可以实现 1-范数误差 [例如，Negahban et al., 2012]：

$$
\left\| \hat {\beta} _ {(w)} - \beta_ {(w)} \right\| _ {1} = \mathcal {O} _ {P} \left(k \sqrt {\frac {\log (p)}{n}}\right). \tag {7.19}
$$

同时，近似平衡权重的**不平衡性（imbalance）** 可以通过以下结果来控制。

**引理 7.3.** 假设强重叠性成立，即存在某个 $\eta > 0$ 使得 $\eta \leq e ( X _ { i } ) \leq 1 - \eta$，并且特征 $X _ { i }$ 有界 $| X _ { i } | \le M$。那么，以至少 $1 - \delta$ 的概率，在调节参数 $\zeta = 1 / ( 4 \log ( p ) )$ 下，近似平衡规划 (7.15) 的解满足：

$$
\frac {1}{n} \sum_ {W _ {i} = 1} \left(\hat {\gamma} _ {i} ^ {(1)}\right) ^ {2} = \mathcal {O} _ {P} (1), \quad \hat {t} ^ {(1)} = \mathcal {O} _ {P} \left(\sqrt {\frac {\log (p)}{n}}\right). \tag {7.20}
$$

**证明.** 考虑如果我们代入真实的倾向性得分 $\gamma _ { i } ^ { * } = 1 / e ( X _ { i } )$ 时 (7.15) 中目标函数的值。这种选择会带来最坏情况下的不平衡：

$$
t ^ {*} = \left\| \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i}}{e (X _ {i})} - 1\right) X _ {i} \right\| _ {\infty}.
$$

现在，对于每个 $j = 1 , \ldots , p ,$ 我们有 E $[ ( W _ { i } / e ( X _ { i } ) - 1 ) X _ { i j } ] = 0$，并且由于强重叠性和有界性，我们有 $| ( W _ { i } / e ( X _ { i } ) - 1 ) X _ { i j } | \le M / \eta$。因此，我们可以使用 Hoeffding 不等式和联合界来验证：

$$
\mathbb {P} \left[ | t ^ {*} | \geq \frac {M}{\eta} \sqrt {\frac {4 \log (p)}{n}} \right] \leq \frac {2}{p}.
$$

将 Hoeffding 不等式再次应用于目标函数的第一部分，并代入我们选择的 $\zeta$，可得：

$$
\mathbb {P} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i}}{e ^ {2} (X _ {i})} + n \zeta (t ^ {*}) ^ {2} \geq \mathbb {E} \left[ \frac {1}{e (X _ {i})} \right] + \frac {1}{\eta^ {2}} \sqrt {\frac {2 \log (p)}{n}} + \frac {M ^ {2}}{\eta^ {2}} \right] \leq \frac {4}{p}.
$$

现在，真实的逆倾向性得分 $\gamma _ { i } ^ { * }$ 只是优化问题 (7.15) 的一个可行解，而 $\hat { \gamma } ^ { ( 1 ) }$ 被选择为使优化目标尽可能小。因此，根据单调性，我们也必须有：

$$
\mathbb {P} \left[ \frac {1}{n} \sum_ {W _ {i} = 1} \left(\hat {\gamma} _ {i} ^ {(1)}\right) ^ {2} + n \zeta \left(\hat {t} ^ {(1)}\right) ^ {2} \geq \mathbb {E} \left[ \frac {1}{e (X _ {i})} \right] + \frac {1}{\eta^ {2}} \sqrt {\frac {2 \log (p)}{n}} + \frac {M ^ {2}}{\eta^ {2}} \right] \leq \frac {4}{p}.
$$

注意到目标函数中的所有项都是非负的，因此它们也必须分别由给定的上界所控制，由此可得所需结论。 □

综合以上各部分，我们可以使用 (7.19) 和 (7.20) 来证明，在稀疏性界 $\| \beta _ { ( w ) } \| _ { 0 } \leq k$ 下，引理 7.2 中的误差项 $E$ 的量级为 $| E | = \mathcal { O } _ { P } \left( k \log ( p ) / n \right)$。因此，只要稀疏性水平被控制为 $k \ll \sqrt { n } / \log ( p )$，它在 $1 / { \sqrt { n } }$ 尺度上就是可忽略的。这个稀疏性条件在高维推断文献中很常见 [Javanmard 和 Montanari, 2014, Zhang 和 Zhang, 2014]，并且对应于在没有关于协变量 $X _ { i }$ 分布的先验知识的情况下，去偏套索（debiased lasso）方法能够实现有效推断的最弱稀疏性条件。这种联系并非偶然，这里介绍的增广平衡方法实际上与用于高维推断的去偏套索方法密切相关；参见 Athey, Imbens, 和 Wager [2018b] 的讨论和更多参考文献。

**注 7.2.** 我们之前声称，当我们拥有实现近似（而非精确）平衡的权重时，应使用 (7.16) 形式的增广估计量。我们现在可以证实这一说法：假设我们处于高维设定，并使用权重 (7.15) 来构建一个 IPW 型估计量：

$$
\hat {\tau} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(W _ {i} \hat {\gamma} _ {i} ^ {(1)} Y _ {i} - (1 - W _ {i}) \hat {\gamma} _ {i} ^ {(0)} Y _ {i}\right). \tag {7.21}
$$

然后我们可以使用引理 7.3 来控制该估计量的偏倚；然而，由此产生的偏倚界通常量级为 $\sqrt { \log ( p ) / n }$，并且当 $p$ 可以随 $n$ 增长时，这个界主导了估计量的误差。因此，我们的分析仅表明，当在增广估计量中使用近似平衡权重时，才能在高维情况下实现 $\sqrt { n }$ 一致性。

**注 7.3.** 在比较本章讨论的不同方法时，一个自然的问题是：如果在低维设定下应用直接的平衡搜索策略 (7.15)，并追求精确而非近似平衡，会发生什么？这将导致处理组权重为：

$$
\hat {\gamma} ^ {(1)} = \operatorname{argmin} _ {\gamma_ {i} \geq 0} \left\{\frac {1}{n} \sum_ {W _ {i} = 1} \gamma_ {i} ^ {2}: \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\gamma_ {i} W _ {i} - 1\right) X _ {i} = 0 \right\}, \tag {7.22}
$$

以及类似的控制组权重；注意，这个优化问题通常仅在处理组单元数和控制组单元数都大于 $p$ 时才可行。如果我们实现了精确平衡，则不再需要使用 (7.16) 中的增广形式；事实上，精确平衡意味着回归调整项会被精确抵消，因此增广估计量在数值上等同于非增广估计量 [Robins et al., 2007]。44

## 7.3 文献注释（Bibliographic notes）

协变量平衡在无混杂性下进行平均处理效应估计中的关键作用早已被认识到，处理任何加权或匹配型估计量时的标准操作程序是将平衡用作拟合优度检验 [Imbens 和 Rubin, 2015]。例如，在通过逻辑回归拟合倾向性模型后，可以检查由此产生的倾向性权重是否以合理的精度满足 (7.6) 类型的样本平衡条件。如果平衡条件不满足，可以尝试拟合一个不同的（更好的）倾向性模型。

将协变量平衡作为指导倾向性估计（而不仅仅是作为事后合理性检查）的想法是较新的。来自不同学科的早期提议包括 Graham, Pinto, 和 Egel [2012], Hainmueller [2012] 以及 Imai 和 Ratkovic [2014]；Zhao [2019] 通过协变量平衡损失函数提供了这些方法的统一视角。Zubizarreta [2015] 提出学习实现平衡的权重，而无需在参数倾向性模型的背景下显式应用 IPW。Iacus, King, 和 Porro [2012] 提出将连续协变量空间粗化为有限个区域，然后在这些区域上应用分层估计量以实现平衡。45 “协变量平衡倾向性得分”这一术语由 Imai 和 Ratkovic [2014] 提出，而我们在第 7.1 章中的阐述最接近地建立在 Graham, Pinto, 和 Egel [2012] 和 Zhao [2019] 的工作之上。

我们在第 7.2 章中的阐述改编自 Athey, Imbens, 和 Wager [2018b]，他们证明了在稀疏线性结果模型下，近似平衡权重和增广估计量可用于高维控制变量的平均处理效应推断。Tan [2020] 将增广构造与协变量平衡倾向性得分估计量 (7.10) 的套索惩罚变体相结合，以估计高维线性-逻辑设定中的平均处理效应。Kallus [2020] 和 Hirshberg 与 Wager [2021] 考虑了非参数设定下的平衡（和增广平衡）方法，并推导出能近似平衡无限维空间中所有函数（例如，给定光滑性类别中的所有函数）的权重。特别是，Hirshberg 和 Wager [2021] 表明，如果平衡的函数类不太大，并且张成了真实的逆倾向性加权函数 $1 / e ( \cdot )$ 和 $1 / ( 1 + e ( \cdot ) )$，那么在弱条件下，增广的近似平衡平均处理效应估计量在第 3.2 章的意义上是有效的。

最后，平衡估计背后的原理比平均处理效应估计更广泛，实际上可用于估计一大类计量经济学目标。**Riesz 表示定理（Riesz representer theorem）** 给出了条件，在这些条件下，线性依赖于抽样分布的估计目标 $\theta$——包括平均导数和平均边际效应等量——可以表征为加权平均 $\theta = \mathbb { E } \left[ \gamma ( X _ { i } , W _ { i } ) Y _ { i } \right]$，其中权重函数 $\gamma ( \cdot )$ 被称为 Riesz 表示子。在无混杂性和二元处理下的 ATE 估计情形中，Riesz 表示子是 $\gamma ( x , w ) = w / e ( x ) - ( 1 - w ) / ( 1 - e ( x ) )$，因此用于 ATE 估计的 IPW 实际上是 Riesz 表示子加权的一个特例。Chernozhukov 等人 [2022a] 利用这一视角，通过用估计 Riesz 表示子替代倾向性估计步骤，为广泛的目标开发了双重稳健估计量。Hirshberg 和 Wager [2021] 表明，平衡权重构造 (7.15) 有效地产生了一个惩罚的经验 Riesz 表示子，因此他们的方法（和结果）直接扩展到 Chernozhukov 等人 [2022a] 的一般设定。Chernozhukov, Newey, 和 Singh [2022b] 提供了一个基于机器学习估计 Riesz 表示子的通用方法，可用于自动化构建针对一般线性目标的双重机器学习估计量。
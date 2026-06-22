# 双重差分（Difference in Differences）

注意：以下章节比平时更为粗略，目前包含的图表和直觉性内容不如相应讲座丰富。

## 10.1 预备知识（Preliminaries）

我们在第2章中首次引入了**无混杂假设（unconfoundedness assumption）**（假设2.1）：

$$
(Y (1), Y (0)) \perp T \tag {10.1}
$$

回顾一下，这等价于假设在因果图中不存在从 $T$ 到 $Y$ 的未阻断的后门路径。当这种情况成立时，我们就有关联即为因果关系。换句话说，它为我们提供了以下（希望是熟悉的）对 **ATE** 的识别：

$$
\mathbb {E} [ Y (1) - Y (0) ] = \mathbb {E} [ Y (1) ] - \mathbb {E} [ Y (0) ] \tag {10.2}
$$

$$
= \mathbb {E} [ Y (1) \mid T = 1 ] - \mathbb {E} [ Y (0) \mid T = 0 ] \tag {10.3}
$$

$$
= \mathbb {E} [ Y \mid T = 1 ] - \mathbb {E} [ Y \mid T = 0 ] \tag {10.4}
$$

其中我们在公式10.3中使用了这个无混杂假设。

然而，ATE并非我们可能感兴趣的唯一平均因果效应。通常，实践者感兴趣的是**受处理子群体**中的ATE。这被称为**受处理组的平均处理效应（Average Treatment Effect on the Treated, ATT）**：$\mathbb{E}[Y(1) - Y(0) | T = 1]$。如果我们只对ATT而非ATE感兴趣，我们可以做出一个更弱的假设：

$$
Y (0) \perp T \tag {10.5}
$$

我们只需要假设 $Y(0)$ 在这里是无混杂的，而不是 $Y(0)$ 和 $Y(1)$ 都是无混杂的。我们在以下证明中展示了这一点：

$$
\mathbb {E} [ Y (1) - Y (0) \mid T = 1 ] = \mathbb {E} [ Y (1) \mid T = 1 ] - \mathbb {E} [ Y (0) \mid T = 1 ] \tag {10.6}
$$

$$
= \mathbb {E} [ Y \mid T = 1 ] - \mathbb {E} [ Y (0) \mid T = 1 ] \tag {10.7}
$$

$$
= \mathbb {E} [ Y \mid T = 1 ] - \mathbb {E} [ Y (0) \mid T = 0 ] \tag {10.8}
$$

$$
= \mathbb {E} [ Y \mid T = 1 ] - \mathbb {E} [ Y \mid T = 0 ] \tag {10.9}
$$

其中我们在公式10.8中使用了这个更弱的无混杂假设。

在双重差分中，我们通常关注的是ATT估计量，但我们将使用一个不同的识别假设。

10.1 预备知识 . . . 95

10.2 引入时间 . . . . . . 96

10.3 识别 96

假设 . . . . . . . . . 96

主要结果与证明 . . . 97

10.4 主要问题 . . . . . . . 98

## 10.2 引入时间（Introducing Time）

我们现在将引入时间维度。利用时间维度的信息将是我们无需假设通常的无混杂性就能获得识别的关键。我们将使用 $\tau$ 作为时间的变量。

**设定** 和通常一样，我们有一个**处理组（treatment group）** $(T = 1)$ 和一个**对照组（control group）** $(T = 0)$。然而，现在还有时间维度，并且处理组只在某个特定时间之后才接受处理。因此，我们有某个时间 $\tau = 1$ 表示处理组已接受处理之后的时间，以及某个时间 $\tau = 0$ 表示处理组接受处理之前的某个时间。因为对照组从未接受处理，所以对照组在时间 $\tau = 0$ 或时间 $\tau = 1$ 都没有接受处理。我们将时间 $\tau$ 下处理状态为 $t$ 的潜在结果随机变量记为 $Y_{\tau}(t)$。那么，我们感兴趣的因果估计量是处理组在接受处理后（在时间段 $\tau = 1$ 中）潜在结果的平均差异：

$$
\mathbb {E} [ Y _ {1} (1) - Y _ {1} (0) \mid T = 1 ] \tag {10.10}
$$

换句话说，我们感兴趣的是处理接受后的ATT。

## 10.3 识别（Identification）

### 10.3.1 假设（Assumptions）

你可以把 $Y_1$ 和 $Y_0$ 视为两个不同的随机变量。因此，尽管现在我们有了时间下标，当潜在结果括号内的值与 $T$ 的条件值匹配时，我们仍然可以通过**一致性（consistency）**（回顾假设2.5）进行简单的识别：

**假设 10.1（一致性，Consistency）** 如果处理状态是 $t$，那么在时间 $\tau$ 观测到的结果 $Y_{\tau}$ 就是处理状态为 $t$ 时的潜在结果。形式上：

$$
\forall \tau , \quad T = t \implies Y _ {\tau} = Y _ {\tau} (t) \tag {10.11}
$$

我们可以等价地将其写成如下形式：

$$
\forall \tau , \quad Y _ {\tau} = Y _ {\tau} (T) \tag {10.12}
$$

一致性告诉我们，因果估计量 $\mathbb{E}[Y_{\tau}(1) | T = 1]$ 等于统计估计量 $\mathbb{E}[Y_{\tau} | T = 1]$，并且类似地，$\mathbb{E}[Y_{\tau}(0) | T = 0] = \mathbb{E}[Y_{\tau} | T = 0]$。相比之下，$\mathbb{E}[Y_{\tau}(1) | T = 0]$ 和 $\mathbb{E}[Y_{\tau}(0) | T = 1]$ 是反事实因果估计量，因此一致性并不能直接为我们识别这些量。注意：在本章的推导中，我们还隐含地假设了**无干扰假设（no interference assumption）**（假设2.4）扩展到了我们有时间下标的情形。

现在，我们来到了双重差分的定义性假设：**平行趋势（parallel trends）**。该假设表明，如果处理组未接受处理，那么处理组（随时间）的趋势将与对照组（随时间）的趋势相匹配。

**假设 10.2（平行趋势，Parallel Trends）**

$$
\mathbb {E} [ Y _ {1} (0) - Y _ {0} (0) \mid T = 1 ] = \mathbb {E} [ Y _ {1} (0) - Y _ {0} (0) \mid T = 0 ] \tag {10.13}
$$

这类似于关于差值的无混杂假设：

$$
\left(Y _ {1} (0) - Y _ {0} (0)\right) \perp T \tag {10.14}
$$

所以你可以将其视为我们在公式10.5中看到的常规无混杂性，但这里的处理与潜在结果的差值独立，而不是与潜在结果本身独立。

然后，我们需要最后一个假设。该假设是处理在接受之前对处理组没有影响。

**假设 10.3（无预处理效应，No Pretreatment Effect）**

$$
\mathbb {E} \left[ Y _ {0} (1) - Y _ {0} (0) \mid T = 1 \right] = 0 \tag {10.15}
$$

这个假设可能看起来显然成立，但事实并非总是如此。例如，如果参与者预期到处理，那么他们可能能够……

### 10.3.2 主要结果与证明（Main Result and Proof）

使用上一节中的假设，我们可以证明ATT等于每个处理组中跨时间差异之间的差值。我们在以下命题中数学化地表述了这一点。

**命题 10.1（双重差分识别，Difference-in-differences Identification）** 给定一致性、平行趋势和无预处理效应，我们有：

$$
\begin{array}{l} \mathbb {E} \left[ Y _ {1} (1) - Y _ {1} (0) \mid T = 1 \right] \\ = \left(\mathbb {E} \left[ Y _ {1} \mid T = 1 \right] - \mathbb {E} \left[ Y _ {0} \mid T = 1 \right]\right) - \left(\mathbb {E} \left[ Y _ {1} \mid T = 0 \right] - \mathbb {E} \left[ Y _ {0} \mid T = 0 \right]\right) \tag {10.16} \\ \end{array}
$$

**证明** 和通常一样，我们从期望的线性性开始：

$$
\mathbb {E} [ Y _ {1} (1) - Y _ {1} (0) \mid T = 1 ] = \mathbb {E} [ Y _ {1} (1) \mid T = 1 ] - \mathbb {E} [ Y _ {1} (0) \mid T = 1 ] \tag {10.17}
$$

我们可以使用一致性立即识别处理组中的处理潜在结果：

$$
= \mathbb {E} [ Y _ {1} \mid T = 1 ] - \mathbb {E} [ Y _ {1} (0) \mid T = 1 ] \tag {10.18}
$$

常规无混杂性：

$$
Y (0) \perp T \quad (\text{公式 10.5 回顾})
$$

主动阅读练习：你将如何估计公式10.16右侧的统计估计量？

因此，我们已经识别了第一项，但第二项仍有待识别。为此，我们将在平行趋势假设中解出这一项：¹

$$
\mathbb {E} \left[ Y _ {1} (0) \mid T = 1 \right] = \mathbb {E} \left[ Y _ {0} (0) \mid T = 1 \right] + \mathbb {E} \left[ Y _ {1} (0) \mid T = 0 \right] - \mathbb {E} \left[ Y _ {0} (0) \mid T = 0 \right] \tag {10.19}
$$

我们可以使用一致性来识别最后两项：

$$
= \mathbb {E} [ Y _ {0} (0) \mid T = 1 ] + \mathbb {E} [ Y _ {1} \mid T = 0 ] - \mathbb {E} [ Y _ {0} \mid T = 0 ] \tag {10.20}
$$

但第一项是反事实的。这就是我们需要无预处理效应假设的地方：²

$$
= \mathbb {E} [ Y _ {0} (1) \mid T = 1 ] + \mathbb {E} [ Y _ {1} \mid T = 0 ] - \mathbb {E} [ Y _ {0} \mid T = 0 ] \tag {10.21}
$$

现在，我们可以使用一致性来完成识别：

$$
= \mathbb {E} [ Y _ {0} \mid T = 1 ] + \mathbb {E} [ Y _ {1} \mid T = 0 ] - \mathbb {E} [ Y _ {0} \mid T = 0 ] \tag {10.22}
$$

既然我们已经识别了 $\mathbb{E}[Y_1(0) | T = 1]$，我们可以将公式10.22代回公式10.18以完成证明：

$$
\begin{array}{l} \mathbb {E} [ Y _ {1} (1) \mid T = 1 ] - \mathbb {E} [ Y _ {1} (0) \mid T = 1 ] \\ = \mathbb {E} \left[ Y _ {1} \mid T = 1 \right] - \left(\mathbb {E} \left[ Y _ {0} \mid T = 1 \right] + \mathbb {E} \left[ Y _ {1} \mid T = 0 \right] - \mathbb {E} \left[ Y _ {0} \mid T = 0 \right]\right) (10.23) \\ = \left(\mathbb {E} \left[ Y _ {1} \mid T = 1 \right] - \mathbb {E} \left[ Y _ {0} \mid T = 1 \right]\right) - \left(\mathbb {E} \left[ Y _ {1} \mid T = 0 \right] - \mathbb {E} \left[ Y _ {0} \mid T = 0 \right]\right) (10.24) \\ \end{array}
$$

¹ 平行趋势假设（假设10.2）：

$$
\begin{array}{l} \mathbb {E} [ Y _ {1} (0) \mid T = 1 ] - \mathbb {E} [ Y _ {0} (0) \mid T = 1 ] \\ = \mathbb {E} [ Y _ {1} (0) \mid T = 0 ] - \mathbb {E} [ Y _ {0} (0) \mid T = 0 ] \tag {10.13回顾} \\ \end{array}
$$

² 无预处理效应假设（假设10.3）：

$$
\begin{array}{r l} \mathbb {E} [ Y _ {0} (1) \mid T = 1 ] - \mathbb {E} [ Y _ {0} (0) \mid T = 1 ] & = 0 \\ & \text{(10.15回顾)} \end{array}
$$

## 10.4 主要问题（Major Problems）

双重差分方法用于因果效应估计的第一个主要问题是，平行趋势假设常常不成立。我们可以尝试通过控制相关的混杂变量并满足**受控平行趋势假设（controlled parallel trends assumption）**来解决这个问题：

**假设 10.4（受控平行趋势，Controlled Parallel Trends）**

$$
\mathbb {E} [ Y _ {1} (0) - Y _ {0} (0) \mid T = 1, W ] = \mathbb {E} [ Y _ {1} (0) - Y _ {0} (0) \mid T = 0, W ] \tag {10.25}
$$

这在实践中是常用的，但仍然可能无法满足这个较弱版本的平行趋势假设。例如，如果在 $Y$ 的结构方程中存在处理 $T$ 和时间 $\tau$ 的交互项，我们永远无法获得平行趋势。

此外，平行趋势假设是**尺度特定（scale-specific）**的。例如，如果我们满足平行趋势，这并不意味着我们在对 $Y$ 进行某种变换后也满足平行趋势。**对数变换（logarithm）**是其中一种常见的变换。这是因为平行趋势假设是关于差值的假设，这使得它并非完全非参数化的。在这个意义上，平行趋势假设是**半参数化（semi-parametric）**的。类似地，双重差分方法是一种半参数方法。
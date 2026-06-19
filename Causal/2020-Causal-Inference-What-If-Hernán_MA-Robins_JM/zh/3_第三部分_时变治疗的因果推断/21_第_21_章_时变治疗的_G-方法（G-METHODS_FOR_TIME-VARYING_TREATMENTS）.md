# 第 21 章 时变治疗的 G-方法（G-METHODS FOR TIME-VARYING TREATMENTS）

在上一章中，我们描述了一个具有时变治疗和治疗-混杂因素反馈的数据集。我们展示了，当应用于该数据集时，传统的混杂调整方法无法正确调整混杂。尽管时变治疗对结局的因果效应为零，但传统调整方法得到的效应估计与零假设不同。

本章描述了在存在治疗-混杂因素反馈时，传统方法偏倚的解决方案：使用 **g-方法（g-methods）** ——包括 **g-公式（g-formula）** 、 **IP 加权（IP weighting）** 、 **g-估计（g-estimation）** 及其 **双稳健推广（doubly-robust generalizations）** 。使用与上一章相同的数据集，我们在此展示三种 g-方法均能得出正确的（零）效应估计。对于时间固定治疗，我们在第 13 章描述了 g-公式，在第 12 章描述了边际结构模型的 IP 加权，在第 15 章描述了结构嵌套模型的 g-估计。在此我们介绍三种 g-方法中每一种，用于在第 19 章描述的可识别性条件（序贯可交换性、正性（positivity）和一致性（consistency））下比较静态治疗策略。

### 21.1 时变治疗的 g-公式（The g-formula for time-varying treatments）

**表 21.1**

| N    | A0  | L1  | A1  | Mean Y |
| ---- | --- | --- | --- | ------ |
| 2400 | 0   | 0   | 0   | 84     |
| 1600 | 0   | 0   | 1   | 84     |
| 2400 | 0   | 1   | 0   | 52     |
| 9600 | 0   | 1   | 1   | 52     |
| 4800 | 1   | 0   | 0   | 76     |
| 3200 | 1   | 0   | 1   | 76     |
| 1600 | 1   | 1   | 0   | 44     |
| 6400 | 1   | 1   | 1   | 44     |

再次考虑表 20.1 中序贯随机实验的数据，为方便起见，我们在此将其复制为表 21.1。假设我们只对时间固定治疗 $A_1$ 的效应感兴趣。也就是说，假设我们想对比反事实均数 $E\left[Y^{a_1=1}\right]$ 和 $E\left[Y^{a_1=0}\right]$ 。在第一部分和第二部分中，我们已经证明，在可识别性条件下，每个均数 $E\left[Y^{a_1}\right]$ 是条件于（时间固定）治疗和混杂因素的平均结局 $E\left[Y \mid A_1=a_1, L_1=l_1\right]$ 的加权平均。具体来说， $E\left[Y^{a_1}\right]$ 等于加权平均

$$
\sum_{l_1} E\left[Y \mid A_1=a_1, L_1=l_1\right] f\left(l_1\right), \text{其中} f\left(l_1\right)=\Pr\left[L_1=l_1\right].
$$

因为，如上一章所示，只需要 $L_1$ 来使接受治疗者（ $A_1=1$ ）和未接受治疗者（ $A_1=0$ ）条件可交换。该加权平均就是 $E\left[Y^{a_1}\right]$ 的 g-公式：按研究人群中混杂因素（此处仅为 $L_1$ ）的分布标准化的平均结局。

但是，在表 21.1 的序贯随机实验中，治疗 $\overline{A}=\left(A_0, A_1\right)$ 是时变的，并且如我们在上一章所见，存在治疗-混杂因素反馈。这意味着不能依赖传统调整方法无偏地估计时变治疗 $\overline{A}$ 的因果效应。例如，即使在序贯可交换性成立的序贯随机实验中，传统方法可能也无法提供“始终治疗”下的平均结局 $E\left[Y^{a_0=1, a_1=1}\right]$ 和“从不治疗”下的平均结局 $E\left[Y^{a_0=0, a_1=0}\right]$ 的有效估计。相比之下，g-公式可用于计算序贯随机实验中的反事实均数 $E\left[Y^{a_0, a_1}\right]$ 。为此，需要推广上述时间固定治疗的 g-公式表达式。

时变治疗的 g-公式最早由 Robins（1986, 1987）描述。

在一个有 2 个时间点的研究中，“从不治疗”的 g-公式为

$$
E\left[Y \mid A_0=0, A_1=0, L_1=0\right] \times \Pr\left[L_1=0 \mid A_0=0\right] + E\left[Y \mid A_0=0, A_1=0, L_1=1\right] \times \Pr\left[L_1=1 \mid A_0=0\right].
$$

在可识别性条件（第 19 章描述）下， $E\left[Y^{a_0, a_1}\right]$ 的 g-公式仍将是一个加权平均，但现在它是对条件于时变治疗和实现序贯可交换性所需的混杂因素的平均结局 $E\left[Y \mid A_0=a_0, A_1=a_1, L_1=l_1\right]$ 的加权平均。权重是给定过去的混杂因素 $L_1$ 的分布，在本例中，过去是对应于干预的过去治疗值。具体来说，g-公式为

$$
\sum_{l_1} E\left[Y \mid A_0=a_0, A_1=a_1, L_1=l_1\right] f\left(l_1 \mid a_0\right).
$$

#### 细节点 21.1（Fine Point 21.1）

**g 公式（g-formula）** 在（静态） **序贯可交换性（sequential exchangeability）** 条件下等于 $\operatorname{E}[Y^{a_0, a_1}]$ 。也就是说，对于 **时变处理（time-varying treatment）** ，在可识别性条件下，反事实均值结局的 g 公式估计量是标准化至研究人群混杂因素分布后的均值结局，其中表达式中的每个因子都条件于既往治疗和协变量历史。在时间固定的情况下，即治疗和混杂因素均在单个时间点测量时，这种对既往历史的条件化并非必要。

仅当对于任意使得 $f(l_1 \mid a_0) \neq 0$ 的值 $l_1$ ，人群中存在具有（ $A_0 = a_0, A_1 = a_1, L_1 = l_1$ ）的个体时，g 公式才是可计算的（即良好定义的）。这等价于技术点 19.2（Technical Point 19.2）中给出的 **积极性（positivity）** 定义，也是对技术点 3.1（Technical Point 3.1）中积极性讨论的时变处理推广。

让我们应用 g 公式来估计表 21.1（Table 21.1）的序贯随机化实验中的因果效应 $\operatorname{E}[Y^{a_0=1, a_1=1}] - \operatorname{E}[Y^{a_0=0, a_1=0}]$ 。均值 $\operatorname{E}[Y^{a_0=0, a_1=0}]$ 的 g 公式估计值为 $84 \times 0.25 + 52 \times 0.75 = 60$ 。均值 $\operatorname{E}[Y^{a_0=1, a_1=1}]$ 的 g 公式估计值为 $76 \times 0.50 + 44 \times 0.50 = 60$ 。因此，因果效应 $\operatorname{E}[Y^{a_0=1, a_1=1}] - \operatorname{E}[Y^{a_0=0, a_1=0}]$ 的估计值为 0，正如预期。在传统方法失败的地方，g 公式取得了成功。

![image_136](../../images/image_136.png)

> 图 21.1（Figure 21.1）

另一种理解 g 公式的方式是将其视为一种 **模拟（simulation）** 。在 $Y$ 和 $L$ 联合序贯可交换性条件下，g 公式模拟了如果研究人群中的每个人都遵循治疗策略 $\bar{a}$ 时将会观察到的反事实结局 $Y^{\bar{a}}$ 和协变量历史 $L^{\bar{a}}$ 。换句话说，g 公式模拟（识别）了在策略 $\bar{a}$ 下反事实变量 $(Y^{\bar{a}}, \bar{L}^{\bar{a}})$ 的联合分布。要理解这一点，首先考虑图 21.1 中的 **因果解释结构树图（causally interpreted structured tree graph）** ，它是表 21.1 中数据的另一种表示形式。

在上述可识别性条件下，g 公式可以被视为构建一个新树的过程，在该树中所有个体都遵循策略 $\bar{a}$ 。例如，图 21.2 中的因果解释结构树图展示了如果所有个体都遵循"始终治疗"策略（ $a_0 = 1, a_1 = 1$ ）时将会观察到的反事实人群。

![Figure 21.2](../../images/d8ff846ad423eaf93ed4a95ceb0ab9ae5065f187b65f5bc886d13c8663562213.jpg)

在序贯可交换性条件下， $\Pr[L_1 = l_1 \mid A_0 = a_0] = \Pr[L_1^{a_0} = l_1]$ 且 $\operatorname{E}[Y \mid A_0 = a_0, A_1 = a_1, L_1 = l_1] = \operatorname{E}[Y^{a_0, a_1} \mid L_1^{a_0} = l_1]$ 。因此，g 公式为

$$
\sum_{l_1} \operatorname{E}[Y^{a_0, a_1} \mid L_1^{a_0} = l_1] \Pr[L_1^{a_0} = l_1],
$$

这等于 $\operatorname{E}[Y^{a_0, a_1}]$ ，符合要求。

为模拟这个反事实人群，我们：

- (i) 在时间点 $k = 0$ 和 $k = 1$ 分别赋予接受治疗 $a_0 = 1$ 和 $a_1 = 1$ 的概率为 1；并且
- (ii) 赋予与原始研究人群相同的概率 $\Pr[L_1 = l_1 \mid A_0 = a_0]$ 和相同的均值 $\operatorname{E}[Y \mid A_0 = a_0, A_1 = a_1, L_1 = l_1]$ 。

  **两个重要要点。** 第一，g 公式的值取决于 $L$ 中包含了什么（如果有的话）。例如，假设我们没有收集 $L_1$ 的数据，因为我们错误地认为我们的研究由图 20.8（Figure 20.8）中的因果图表示，只是去掉了从 $L_1$ 到 $A_1$ 的箭头。因此，我们认为 $L_1$ 不是混杂因素，因此对于识别不是必需的。那么，在缺乏 $L_1$ 数据的情况下，g 公式变为 $\operatorname{E}[Y \mid A_0 = a_0, A_1 = a_1]$ ，因为没有需要调整的协变量历史。然而，由于我们的研究实际上由图 20.8 中的因果图表示（在该图下，治疗分配 $A_1$ 受到 $L_1$ 的影响），未能包含 $L_1$ 的 g 公式不再具有因果解释。

第二，即使 g 公式具有因果解释，其每个组成部分可能缺乏因果解释。例如，考虑图 20.9（Figure 20.9）中的因果图，在该图下仅满足静态序贯可交换性。包含 $L_1$ 的 g 公式正确识别了 $Y^{\bar{a}}$ 的均值。值得注意的是，无论我们是否添加从 $A_0$ 和 $A_1$ 到 $Y$ 的箭头，g 公式都继续具有作为 $\operatorname{E}[Y^{\bar{a}}]$ 的因果解释，尽管其两个组成部分—— $\operatorname{E}[Y \mid A_0 = a_0, A_1 = a_1, L_1 = l_1]$ 和 $\Pr[L_1 = l_1 \mid A_0 = a_0]$ ——完全没有任何因果解释。也就是说， $\Pr[L_1 = l_1 \mid A_0 = a_0] \neq \Pr[L_1^{a_0} = l_1]$ 且 $\operatorname{E}[Y \mid A_0 = a_0, A_1 = a_1, L_1 = l_1] \neq \operatorname{E}[Y^{a_0, a_1} \mid L_1^{a_0} = l_1]$ 。最后两个不等式在像图 20.1 和 20.2（Figures 20.1 and 20.2）所表示的序贯随机化试验中将会成为等式。

### 治疗与协变量历史（Treatment and covariate history）

在描述 g-方法时，我们常提及实现 **序贯可交换性（sequential exchangeability）** 所需的 **治疗与协变量历史（treatment and covariate history）** 。对于 **g-公式（g-formula）** ，我们称其组成部分以先前的治疗和协变量历史为条件。

例如，对应于时间 $k = 2$ 处离散混杂变量 $L_{2}$ 概率的因子

$$
f \left( l_{2} \mid \overline{A}_{1} = \bar{a}_{1}, \overline{L}_{1} = \bar{l}_{1} \right) = \operatorname{Pr} \left[ L_{2} = l_{2} \mid A_{0} = a_{0}, A_{1} = a_{1}, L_{0} = l_{0}, L_{1} = l_{1} \right]
$$

以时间 0 和 1 先前的治疗和混杂变量为条件；时间 $k = 3$ 处的因子以时间 0、1 和 2 的治疗和混杂变量为条件，依此类推。

然而，“历史”一词不必按时间定义，因为如 **要点 7.4（Fine Point 7.4）** 所述，混杂变量理论上可能处于治疗的时间未来。相反，如结合图 7.4 所解释的，调整治疗时间过去中的某些变量可能会引入 **选择偏倚（selection bias）** （称为 **M-偏倚（M-bias）** ）。因此，在本书中，时间 $k$ 处因果相关的“历史”应理解为为实现治疗 $A_{k}$ 的条件可交换性所需的一组治疗和混杂变量。在大多数情况下，这种历史的使用将对应于时间顺序历史。

现在让我们将 g-公式推广到具有多个时间点 $k$ 的高维设定中。g-公式为

$$
\sum_{\bar{l}} \operatorname{E} \left[ Y \mid \bar{A} = \bar{a}, \bar{L} = \bar{l} \right] \prod_{k = 0}^{K} f \left( l_{k} \mid \bar{a}_{k - 1}, \bar{l}_{k - 1} \right),
$$

> **技术要点 21.1（Technical Point 21.1）** 给出了 g-公式的更一般表达式，可用于计算密度，而不仅仅是均值。

其中求和是对所有可能的 $l$ -历史进行的（ $l_{k - 1}$ 是通过时间 $k - 1$ 的历史）。求和 $\sum_{\bar{l}}$ 也可写为 $\sum_{l_{K}} \ldots \sum_{l_{1}} \sum_{l_{0}}$ 。在给定每个时间 $k$ 处的 $\left( L_{k}, A_{k} \right)$ 时，对于 $Y^{\bar{a}}$ 满足序贯可交换性的条件下，该表达式等于治疗策略 $a$ 下的 **反事实均值（counterfactual mean）** $\operatorname{E} \left[ Y^{a} \right]$ 。

> **要点 21.1（Fine Point 21.1）** 提出了“历史”一词更细致的定义。

然而，在实践中，如果数据是高维的（正如在具有多个混杂变量或时间点的观察性研究中所预期的），则无法直接计算 g-公式的组成部分。量 $\operatorname{E} \left[ Y \mid \bar{A} = \bar{a}, \bar{L} = \bar{l} \right]$ 和 $f \left( l_{k} \mid \bar{a}_{k - 1}, \bar{l}_{k - 1} \right)$ 需要被估计。

例如，我们可以拟合一个 **线性回归模型（linear regression model）** 来估计随访结束时结局变量的条件均值 $\operatorname{E} \left[ Y \mid \bar{A} = \bar{a}, \bar{L} = \bar{l} \right]$ ，以及拟合 **逻辑回归模型（logistic regression models）** 来估计每个时间 $k \neq 0$ 处离散混杂变量 $L_{k}$ 的分布（ $L_{0}$ 的分布可以如第 13.3 节所述无需模型进行估计）。来自这些模型的估计值 $\widehat{\operatorname{E}} \left[ Y \mid \bar{A} = \bar{a}, \bar{L} = \bar{l} \right]$ 和 $\widehat{f} \left( l_{k} \mid \bar{a}_{k - 1}, \bar{l}_{k - 1} \right)$ 随后将被代入 g-公式。自第 13 章以来，我们将此估计量称为 **代入式 g-公式（plug-in g-formula）** ，并且当代入式 g-公式中使用的估计值基于 **参数模型（parametric models）** 时，我们将代入式 g-公式称为 **参数 g-公式（parametric g-formula）** 。

为简洁起见，本章主要关注确定性策略下的 g-公式。然而，在序贯可交换性下，g-公式可用于计算随机治疗策略 $f^{\mathrm{int}}$ 下的反事实结局均值。一个随机（静态）策略的例子是“在每个时间 $k$ 独立地，以概率 0.3 治疗个体，以概率 0.7 不治疗”，其中 $f^{\mathrm{int}} \left( 1 \mid \bar{a}_{k - 1}, \bar{l}_{k} \right) = 0.3$ 。也就是说， $f^{\mathrm{int}} \left( a_{k} \mid \bar{a}_{k - 1}, \bar{l}_{k} \right)$ 是在治疗策略下时间 $k$ 处治疗 $a_{k}$ 的条件概率。

#### 技术要点 21.1（Technical Point 21.1）

**g-公式密度** 对于确定性静态策略 $\bar{a}$ ，在 $(y, \bar{l})$ 处评估的 $(Y, \overline{L})$ 的 g-公式密度为

$$
f(y \vert \bar{a}_K, \bar{l}_K) \prod_{k=0}^K f(l_k \vert \bar{a}_{k-1}, \bar{l}_{k-1}).
$$

$Y$ 的静态 g-公式密度就是 $(Y, \overline{L})$ 的 g-公式密度下 $Y$ 的边缘密度：

$$
\int \dots \int f(y \vert \bar{a}_K, \bar{l}_K) \prod_{k=0}^K dF(l_k \vert \bar{a}_{k-1}, \bar{l}_{k-1}),
$$

其中积分符号 $\int$ 用于适应 $L_k$ 的某些分量为连续变量的设定。

对于动态确定性策略 $g = (g_0, \dots, g_K)$ ，其中 $g_k(\overline{a}_{k-1}, \overline{l}_k)$ 取值于 $A_k$ 的支撑集， $(Y, \overline{L})$ 和 $Y$ 的 g-公式密度只需在上述公式中将 $\bar{a}_k$ 替换为 $\bar{a}_k^g$ 。这里， $\bar{a}_k^g$ 对于 $k = 0, \dots, K$ 递归定义为

$$
\bar{a}_k^g \equiv \bar{g}_k(\bar{a}_{k-1}^g, \bar{l}_k) \equiv [g_0(\bar{a}_{-1}^g, \bar{l}_0), \dots, g_k(\bar{a}_{k-1}^g, \bar{l}_k)],
$$

其中 $\bar{a}_{-1}^g$ 定义为 0。静态策略是动态策略的一个特例，此时每个 $g_k(\overline{a}_{k-1}, \overline{l}_k)$ 都是一个常值函数。

更一般地，给定观测数据 $O = (\bar{A}, \overline{X}, Y)$ 和未观测数据 $\overline{U}$ ，其中 $\overline{X}$ 是除治疗 $\bar{A}$ 和结局 $Y$ 之外所有已测量变量的集合，g-公式的输入为：

1. 一个确定性治疗策略 $g$ ，
2. 一个表示观测数据（及其未测量的共同原因）的 **因果有向无环图（causal DAG）** ，
3. 我们想要调整的 $\overline{X}$ 的一个子集 $\overline{L}$ ，以及
4. 选择 $\overline{L}$ 、 $\bar{A}$ 和 $Y$ 的一个与 DAG 的拓扑结构一致的 **全序（total ordering）** ，即一个使得每个变量在其祖先之后出现的序。

向量 $L_k$ 由在序中位于 $A_{k-1}$ 之后、 $A_k$ 之前的 $L$ 中的所有变量组成。所选的序通常（但并非总是）是时间顺序，如 **要点 21.1（Fine Point 21.1）** 所讨论的。

当对于所选序， $Y^g$ 的序贯可交换性和 **积极性（positivity）** 成立时， $Y$ 的 g-公式密度等于如果所有个体都遵循策略 $g$ 时在研究人群中本应观察到的密度 $f_{Y^g}(y)$ 。否则，g-公式仍然可以计算，但缺乏因果解释。

当 $(Y^g, \overline{L}^g)$ 的积极性和可交换性成立时（例如，从 $\overline{U}$ 或 $\overline{X}$ 中但不在 $\overline{L}$ 中的任何变量到任何治疗变量之间没有箭头）， $(Y, \overline{L})$ 的 g-公式密度等于密度 $f_{Y^g, \overline{L}^g}(y, \bar{l})$ 。

策略（或干预） $f^{int}$ 。那么，一般的 g-公式表达式为

$$
\sum_{\bar{a}, \bar{l}} \operatorname{E}[Y \vert \bar{A} = \bar{a}, \bar{L} = \bar{l}] \prod_{k=0}^K f(l_k \vert \bar{a}_{k-1}, \bar{l}_{k-1}) \prod_{k=0}^K f^{int}(a_k \vert \bar{a}_{k-1}, \bar{l}_k).
$$

**gfoRmula R 包** (Lin et al. 2019) 可通过 CRAN 获取。 **GFORMULA SAS 宏** 可通过 GitHub 获取。参见本书网站。

> 注：如果我们将 $f^{\text{int}} \left( a_k | \bar{a}_{k-1}, l_k \right)$ 替换为观察到的治疗条件概率 $f \left( a_k | \bar{a}_{k-1}, l_k \right)$ ，则该公式是 $Y$ 的观察均值的公式。

g-公式的这个表达式足够通用，可以同时容纳确定性策略和随机策略。在确定性治疗策略下，对于策略规定的 $a_k$ 值， $f^{\text{int}} \left( a_k | \bar{a}_{k-1}, \bar{l}_k \right)$ 始终为 1，而对于其他值则为 0。

例如，在策略“从不治疗”或 $\bar{a} = (0, 0, \dots, 0)$ 下，在所有 $k$ 处概率 $f^{\text{int}} \left( 0 | \bar{a}_{k-1}, l_k \right) = 1$ 。由于对于规定的治疗值 $f^{\text{int}} \left( a_k | \bar{a}_{k-1}, l_k \right)$ 等于 1，而对于所有其他治疗值等于 0，因此无需在上述公式中包含 $f^{\text{int}}$ 因子或对 $a$ 的求和。我们公开可用的软件实现了 g-公式的这一通用表达式，因此可以适应任何治疗策略。

### 21.2 时变治疗的逆概率加权（IP weighting for time-varying treatments）

假设我们只对表 21.1 中 **时间固定治疗（time-fixed treatment）** $A_1$ 的效应感兴趣。那么我们希望比较反事实均值结局 $\mathrm{E} \left[ Y^{a_1 = 1} \right]$ 和 $\mathrm{E} \left[ Y^{a_1 = 0} \right]$ 。正如我们在第 12 章中所见，在 **可识别性条件（identifiability conditions）** 下，每个反事实均值 $\operatorname{E} \left[ Y^{a_1} \right]$ 是在由个体特定的 **非稳定化权重（nonstabilized weights）** $W^{A_1} = 1 / f \left( A_1 | L_1 \right)$ 或 **稳定化权重（stabilized weights）** $SW^{A_1} = f \left( A_1 \right) / f \left( A_1 | L_1 \right)$ 创建的 **伪总体（pseudo-population）** 中的均值 $\mathrm{E}_{ps} \left[ Y | A_1 = a_1 \right]$ 。

非正式地说，IP 权重的分母是个体在给定其混杂变量值条件下，接受其所接受治疗值的概率。我们可以通过伪总体中具有 $A_1 = a_1$ 的个体 $Y$ 的平均值，从观察研究数据中估计 $\mathrm{E}_{ps} \left[ Y | A_1 = a_1 \right]$ 。

当治疗和混杂变量是时变的时，这些针对时间固定治疗的 IP 权重需要被推广。对于两个时间点的时变治疗 $\bar{A} = \left( A_0, A_1 \right)$ 和时变协变量 $\boldsymbol{L} = \left( L_0, L_1 \right)$ ，非稳定化 IP 权重为

$$
W^{\bar{A}} = \frac{1}{f (A_0 | L_0)} \times \frac{1}{f (A_1 | A_0, L_0, L_1)} = \prod_{k=0}^{1} \frac{1}{f (A_k | \bar{A}_{k-1}, \bar{L}_k)}
$$

稳定化 IP 权重为

$$
SW^{A} = \frac{f (A_0)}{f (A_0 | L_0)} \times \frac{f (A_1 | A_0)}{f (A_1 | A_0, L_0, L_1)} = \prod_{k=0}^{1} \frac{f (A_k | \bar{A}_{k-1})}{f (A_k | \bar{A}_{k-1}, \bar{L}_k)}
$$

其中 $A_{-1}$ 按定义为 0。非正式地说，针对时变治疗的 IP 权重的分母是个体在给定其治疗和协变量历史条件下，接受其所接受治疗历史的概率。

假设我们想比较反事实均值 $\operatorname{E} \left[ Y^{a_0 = 1, a_1 = 1} \right]$ 和 $\operatorname{E} \left[ Y^{a_0 = 0, a_1 = 0} \right]$ 。在静态策略的可识别性假设下，每个反事实均值 $\operatorname{E} \left[ Y^{a_0, a_1} \right]$ 是在由非稳定化权重 $W^{\bar{A}}$ 或稳定化权重 $SW^{A}$ 创建的伪总体中的均值 $\mathrm{E}_{ps} \left[ Y | A_0 = a_0, A_1 = a_1 \right]$ 。也就是说，每个反事实均值的 IP 加权估计量是伪总体中具有 $\overline{A} = (A_0, A_1)$ 的个体 $Y$ 的平均值。

让我们将 IP 加权应用于表 21.1 中的数据。图 21.3 中具有因果解释的 **结构化树图（structured tree graph）** 是图 21.1 中的树图，增加了针对每个治疗和协变量历史的非稳定化 IP 权重 $W^{\bar{A}}$ 和相应伪总体中个体数 $N_W$ 的列。该伪总体的大小为 128,000，即原始总体中的 32,000 个个体乘以 4（静态策略的数量）。由于本研究中没有 $L_0$ ，IP 权重的分母简化为 $f(A_0) f(A_1 | A_0, L_1)$ 。

反事实均值 $\mathrm{E} \left[ Y^{a_0 = 0, a_1 = 0} \right]$ 的 IP 加权估计量是伪总体中的均值 $\mathrm{E}_{ps} \left[ Y | A_0 = 0, A_1 = 0 \right]$ ，我们将其估计为伪总体中 32,000 个具有 $A_0 = 0, A_1 = 0$ 的个体的平均结局。由图 21.3 中的树，估计值为：

$$
84 \times \frac{8000}{32000} + 52 \times \frac{24000}{32000} = 60.
$$

类似地， $\operatorname{E} \left[ Y^{a_0 = 1, a_1 = 1} \right]$ 的 IP 加权估计值也是 60。因此，因果效应 $\operatorname{E} \left[ Y^{a_0 = 1, a_1 = 1} \right] - \operatorname{E} \left[ Y^{a_0 = 0, a_1 = 0} \right]$ 的估计值为 0，正如预期。与 g-公式一样，IP 加权在传统方法失败的地方取得了成功。

与技术要点 12.2 中时间固定治疗的结果类似， $\mathrm{E}_{ps} \left[ Y | A_0 = a_0, A_1 = a_1 \right]$ 等于：

$$
\frac{\operatorname{E} \left[ W^{\bar{A}} Y \operatorname{I} (A_0 = a_0, A_1 = a_1) \right]}{\operatorname{E} \left[ W^{\bar{A}} \operatorname{I} (A_0 = a_0, A_1 = a_1) \right]} = \frac{\operatorname{E} \left[ SW^{\bar{A}} Y \operatorname{I} (A_0 = a_0, A_1 = a_1) \right]}{\operatorname{E} \left[ SW^{\bar{A}} \operatorname{I} (A_0 = a_0, A_1 = a_1) \right]},
$$

对于非稳定化和稳定化伪总体均成立，无论序贯可交换性是否成立。

当在图 21.3 中使用稳定化权重 $SW^{\bar{A}}$ 时，也得到相同的估计值 0（请自行验证）。然而，在非稳定化伪总体中 $\mathrm{Pr}_{ps} [ A_k = 1 | \bar{A}_{k-1}, \bar{L}_k ]$ 为 $1/2$ ，而在稳定化伪总体中为 $\operatorname{Pr}_{ps} \left[ A_k = 1 | \bar{A}_{k-1} \right]$ 。

请注意，我们基于 g-公式对 $\operatorname{E} \left[ Y^{a_0, a_1} \right]$ 的非参数估计值恰好等于基于 IP 加权的估计值。这种相等性与因果推断无关。也就是说，即使可识别性条件不成立——因此 g-公式和 IP 加权估计都没有因果解释——两种方法也会产生相同的数值。

![image_137](../../images/image_137.png)

> 图 21.3

让我们将 IP 加权推广到具有多个时间点 $k = 0, 1, \dots, K$ 的高维设定。非稳定化 IP 权重的通用形式为：

$$
W^{\bar{A}} = \prod_{k=0}^{K} \frac{1}{f(A_k | \bar{A}_{k-1}, \bar{L}_k)}.
$$

稳定化 IP 权重的通用形式为

$$
S W^{\bar{A}} = \prod_{k = 0}^{K} \frac{f \left( A_{k} | \bar{A}_{k - 1} \right)}{f \left( A_{k} | \bar{A}_{k - 1}, \bar{L}_{k} \right)}
$$

我们在正文中的描述仅考虑静态策略。关于动态策略的 IP 加权描述，请参见 **技术要点 21.2（Technical Point 21.2）** 。

当可识别性条件成立时，这些 IP 权重创建一个伪总体，在该伪总体中：(i) $Y^{\bar{a}}$ 的均值与实际总体中的相同，但 (ii) 如图 19.1 所示，每个时间 $k$ 处的随机化概率是常数 $1 / 2$ （非稳定化权重）或最多依赖于先前的治疗历史（稳定化权重）。因此，平均因果效应 $\operatorname{E} \left[ Y^{\bar{a}} \right] - \operatorname{E} \left[ Y^{\bar{a}^{\prime}} \right]$ 为 $\mathrm{E}_{ps} \left[ Y | \overline{A} = \bar{a} \right] - \mathrm{E}_{ps} \left[ Y | \overline{A} = \bar{a}^{\prime} \right]$ ，因为在这两个伪总体中， **序贯无条件可交换性（sequential unconditional exchangeability）** 都成立。

在一个真实的 **序贯随机试验（sequentially randomized trial）** 中，量 $f \left( A_{k} | \bar{A}_{k - 1}, \bar{L}_{k} \right)$ 是设计已知的。因此，我们可以使用它们来计算非稳定化 IP 权重，并且 $\mathrm{E} \left[ Y^{a} \right]$ 和 $\operatorname{E} \left[ Y^{\bar{a}} \right] - \operatorname{E} \left[ Y^{\bar{a}^{\prime}} \right]$ 的估计保证是无偏的。相反，在观察性研究中，量 $f \left( A_{k} | A_{k - 1}, L_{k} \right)$ 需要从数据中估计。

当数据是高维时，我们可以，例如，拟合一个逻辑回归模型来估计每个时间 $k$ 处二分类治疗的条件概率 $\mathrm{Pr} \left\lfloor A_{k} = 1 | \bar{A}_{k - 1}, \bar{L}_{k} \right\rfloor$ 。来自这些模型的估计值 $\widehat{f} \left( A_{k} | \bar{A}_{k - 1}, \bar{L}_{k} \right)$ 随后将替换 $W^{A}$ 中的 $f \left( A_{k} | \bar{A}_{k - 1}, \bar{L}_{k} \right)$ 。如果估计值 $\widehat{f} \left( A_{k} | \bar{A}_{k - 1}, \bar{L}_{k} \right)$ 基于对 $\mathrm{Pr} \left[ A_{k} = 1 | \bar{A}_{k - 1}, \bar{L}_{k} \right]$ 的 **错误设定（misspecified）** 逻辑模型，则 $\operatorname{E} \left[ Y^{\bar{a}} \right]$ 的估计值也可能有偏。

#### 技术要点 21.2（Technical Point 21.2） IP 加权用于动态治疗策略（IP Weighting for Dynamic Treatment Strategies）

考虑确定性动态策略 $g = (g_0, \dots, g_K)$ ，其中 $g_k \equiv g_k(\overline{\mathbf{a}}_{k-1}, \overline{\mathbf{l}}_k)$ 。在策略 $g$ 下，结局 $Y$ 的 **g-公式（g-formula）** 等于

$$
\operatorname{E}\left[ Y \operatorname{I}(\bar{A}_K = \overline{A}_K^g) W^{\bar{A}} \right],
$$

其中 $\overline{\mathbf{a}}_K^g$ 已在技术要点 21.1 中定义。进一步有

$$
\operatorname{E}\left[ Y \operatorname{I}(\bar{A}_K = \overline{A}_K^g) W^{\bar{A}} \right] = \operatorname{E}_{ps}\left[ Y \mid \bar{A}_K = \overline{A}_K^g \right],
$$

其中 $\operatorname{E}_{ps}[Y \mid \bar{A}_K = \overline{A}_K^g]$ 是遵循策略 $g$ 的伪总体成员中 $Y$ 的均值。

与静态策略不同，

$$
\frac{\operatorname{E}\left[ Y \operatorname{I}(\bar{A}_K = \overline{A}_K^g) \mathcal{S} W^{\bar{A}} \right]}{\operatorname{E}\left[ \operatorname{I}(\bar{A}_K = \overline{A}_K^g) \mathcal{S} W^{\bar{A}} \right]}
$$

不等于 g-公式，因为 $SW^{\bar{A}}$ 的分子依赖于 $A$ 。因此， **稳定化权重（stabilized weights）** 不能用于动态策略。

对于随机动态策略 $f^{int}$ ，g-公式等于

$$
\mathbb{E}\left[ Y \prod_{k=0}^K f^{int}\left( A_k \mid \bar{A}_{k-1}, \bar{L}_k \right) W^{\bar{A}} \right] = \operatorname{E}\left[ Y \prod_{k=0}^K \frac{f^{int}\left( A_k \mid \bar{A}_{k-1}, \bar{L}_k \right)}{f\left( A_k \mid \bar{A}_{k-1}, \bar{L}_k \right)} \right].
$$

在实践中，常见的方法是拟合单个模型 $\Pr\left[ A_k = 1 \mid \bar{A}_{k-1}, \bar{\bar{L}}_k \right]$ ，而不是在每个时间点 $k$ 分别拟合模型。该模型包括时间 $k$ 的函数（时变截距）作为协变量，并可能包括与其他协变量的乘积项。

即使两种参数化方法的估计结果相似，也无法从逻辑上保证不存在模型设定错误，因为它们可能都朝同一方向存在偏倚。

该边际结构模型是非饱和的。请记住， **饱和模型（saturated models）** 在方程两侧具有相同数量的未知数。

$\operatorname{E}\left[ Y^{\bar{a}} \right]$ 和 $\operatorname{E}\left[ Y^{\bar{a}} \right] - \operatorname{E}\left[ Y^{\bar{a}'} \right]$ 将存在偏倚。对于稳定化权重 $SW^{\bar{A}}$ ，我们还必须获得分子 $\widehat{f}\left( A_k \mid \bar{A}_{k-1} \right)$ 的估计。即使该估计基于错误设定的模型， $\operatorname{E}\left[ Y^{a} \right]$ 和 $\operatorname{E}\left[ Y^{\bar{a}} \right] - \operatorname{E}\left[ Y^{\bar{a}'} \right]$ 的估计仍然保持无偏，尽管稳定化伪总体中的 $\widehat{f}\left( a_k \mid \overline{\mathbf{a}}_{k-1} \right)$ 将不再与观测数据密度 $f\left( \mathbf{a}_k \mid \overline{\mathbf{a}}_{k-1} \right)$ 一致。

假设我们获得 $\operatorname{E}\left[ Y^{a} \right]$ 的两个估计：一个使用参数化 g-公式，另一个使用通过参数化模型估计的 IP 权重，并且这两个估计的差异超出抽样变异性所能合理解释的范围（估计差异的抽样变异性可通过 **自助法（bootstrapping）** 量化）。那么我们可以得出结论：用于 g-公式的参数化模型或用于 IP 加权的参数化模型（或两者）存在错误设定。无论可识别性假设是否成立，这一结论始终成立。

这意味着，我们应始终使用两种方法估计 $\operatorname{E}\left[ Y^{a} \right]$ ，如果估计结果差异显著（根据某些预先指定的标准），则应重新检查所有模型并在必要时进行修改。在下一节中，我们将描述 **双重稳健估计量（doubly-robust estimators）** 如何帮助处理模型错误设定问题。

此外，正如我们在前一节中讨论的，未知量 $\operatorname{E}\left[ Y^{a} \right]$ 的数量远超样本量的情况并不少见。因此，我们需要指定一个模型，该模型结合来自多个策略的信息，以帮助估计给定的 $\operatorname{E}\left[ Y^{a} \right]$ 。例如，我们可以假设治疗历史 $a$ 对平均结局的影响随着策略 $a$ 下累积治疗量 $\operatorname{cum}(\bar{a}) = \sum_{k=0}^K a_k$ 的函数线性增加。该假设编码在边际结构均值模型中

$$
\operatorname{E}\left[ Y^{\bar{a}} \right] = \beta_0 + \beta_1 \operatorname{cum}(\bar{a})
$$

适用于所有 $\bar{a}$ ，这是第 12 章讨论的固定时间治疗边际结构均值模型的更一般版本。模型左侧有 $2^K$ 个不同的未知量，对应 $2^K$ 个不同的策略 $a$ 各一个，但右侧只有 2 个未知参数 $\beta_0$ 和 $\beta_1$ 。参数 $\beta_1$ 衡量时变治疗 $A$ 的平均因果效应。平均因果效应 $\operatorname{E}\left[ Y^{\bar{a}} \right] - \operatorname{E}\left[ Y^{\bar{a} = \overline{\mathbf{0}}} \right]$ 等于 $\beta_1 \times \operatorname{cum}(\bar{a})$ 。

如第 12 章所述，为了估计边际结构模型的参数，我们可以拟合线性回归模型

$$
\operatorname{E}\left[ Y \mid \overline{A} \right] = \theta_0 + \theta_1 \operatorname{cum}(\overline{A}).
$$

在统计学课程中，通常证明，在 $\operatorname{E}\left[ Y \mid \overline{A} \right]$ 模型正确设定的条件下，普通最小二乘和加权最小二乘估计对于关联参数 $\theta_1$ 都是一致的。该证明假设权重仅依赖于 $\overline{A}$ 。当权重依赖于与 $Y$ 在给定 $\overline{A}$ 条件下相关的协变量 $\overline{L}$ 时（如本例所示），加权回归不再对 $\theta_1$ 一致。

该检验通常对上述特定方向的错误设定具有良好的统计功效，尤其是在使用权重 $\dot{SW^{\bar{A}}}$ 和自助法估计方差时。

通过在稳定化或非稳定化伪总体中使用普通最小二乘法。这在数学上等价于在原始研究总体中使用加权最小二乘法拟合相同的线性模型，分别使用权重 $S W^{A}$ 或 $W^{\bar{A}}$ （在实际数据分析中，这些权重被其估计值替代）。在可识别性条件下， $\theta_{1}$ 的加权最小二乘估计对因果参数 $\beta_{1}$ 一致，而非对关联参数 $\theta_{1}$ 一致。

如第 12 章所讨论， $\widehat{\beta}_{1}$ 的方差——进而 $\operatorname{E}\left[Y^{\bar{a}}\right] - \operatorname{E}\left[Y^{\bar{a} = \overline{0}}\right]$ 对比的方差——可以通过非参数自助法或计算其解析方差（需要额外的统计分析和编程）来估计。我们还可以通过使用 $\widehat{\beta}_{1}$ 的 **稳健方差估计量（robust variance estimator）** 构建保守的 95% 置信区间，大多数统计软件包直接输出该估计量。对于非饱和边际结构模型，当使用权重 $S W^{\bar{A}}$ 拟合模型时，区间宽度通常比使用权重 $W^{\bar{A}}$ 时更窄，因此 $S W^{A}$ 权重更受青睐。

当然，如果边际结构均值模型错误设定，即反事实结局均值依赖于时变治疗的某个函数而非累积治疗量 $\operatorname{cum}\left(\bar{a}\right)$ （例如，仅最后 5 个月的累积治疗量 $\sum_{k=K-5}^{K} a_{k}$ ）或以非线性方式（如二次方）依赖于累积治疗量，则 $\operatorname{E}\left[Y^{a}\right]$ 的估计将是错误的。然而，如果我们拟合模型

$$
\operatorname{E}\left[Y \mid \overline{A}\right] = \theta_{0} + \theta_{1} \operatorname{cum}\left(\overline{A}\right) + \theta_{2} \operatorname{cum}_{-5}\left(\overline{A}\right) + \theta_{3} \operatorname{cum}\left(\overline{A}\right)^{2}
$$

使用权重 $S W^{A}$ 或 $W^{A}$ ，则对联合假设 $\theta_{2} = \theta_{3} = 0$ 的自由度为 2 的 **Wald 检验（Wald test）** 是对我们的边际结构模型是否正确设定的零假设的检验。也就是说，边际结构模型的 IP 加权不受技术要点 21.3 中描述的 **g-零悖论（g-null paradox）** 的影响。在实践中，人们可能选择使用包含治疗历史 $\overline{A}$ 不同汇总统计量作为协变量的边际结构模型，并使用灵活的函数，例如三次样条。

最后，如我们在第 12.5 节中讨论的，我们可以使用边际结构模型来探索 $L_{0}$ 中协变量子集 $V$ 的效应修饰。例如，对于二分类基线变量 $V$ ，我们将边际结构均值模型扩展为

$$
\operatorname{E}\left[Y^{\bar{a}} \mid V\right] = \beta_{0} + \beta_{1} \operatorname{cum}(\bar{a}) + \beta_{2} V + \beta_{3} \operatorname{cum}(\bar{a}) V
$$

该模型的参数可以通过拟合普通线性回归模型来估计

$$
E \left[ Y \mid \overline{A}, V \right] = \theta_0 + \theta_1 \, cum \left( \overline{A} \right) + \theta_2 V + \theta_3 V \, cum \left( \overline{A} \right)
$$

使用 IP 权重 $W^A$ 进行加权最小二乘法，或者更好的是使用

$$
SW^{\bar{A}}(V) = \prod_{k=0}^{K} \frac{f \left( A_k \mid \bar{A}_{k-1}, V \right)}{f \left( A_k \mid \bar{A}_{k-1}, \bar{L}_k \right)}.
$$

在存在治疗-混杂因素反馈的情况下， $V$ 只能包含基线变量。如果 $V$ 包含 $k > 0$ 时的 $L_k$ 分量，那么即使治疗在任何时间点对平均结局没有影响，参数 $\theta_1$ 和 $\theta_3$ 也可能不为零。

我们现在描述任意策略 $g$ 的反事实均值 $E \left[ Y^g \right]$ 的 **双重稳健估计量（doubly robust estimator）** 。

#### 技术要点 21.3（Technical Point 21.3）

**g-零悖论（The g-null paradox）** 。当使用参数化 g-公式时，即使可识别性条件成立，模型错误设定也会导致 $E \left[ Y^{\bar{a}} \right]$ 的估计产生偏倚。假设存在治疗-混杂因素反馈，且治疗对 $Y$ 无影响的严格零假设成立，即

$$
Y^{\bar{a}} - Y^{\bar{a}'} = 0 \quad \text{以概率} 1 \text{对于所有} \bar{a}' \text{和} \bar{a}.
$$

那么对于任意策略 $\bar{a}$ ， $E \left[ Y^{\bar{a}} \right]$ 的 g-公式值相同，尽管如第 20 章所述， $E \left[ Y \mid \bar{A} = \bar{a}, \bar{L} = \bar{l} \right]$ 和 $f \left( l_k \mid \bar{a}_{k-1}, \bar{l}_{k-1} \right)$ 都依赖于 $\bar{a}$ 。

现在假设我们使用标准的非饱和参数化模型

$$
E \left[ Y \mid \bar{A} = \bar{a}, \bar{L} = \bar{l}; \theta \right]
$$

和

$$
f \left( l_k \mid \bar{a}_{k-1}, \bar{l}_{k-1}; \varphi \right)
$$

基于不同的（即变异独立的）参数 $\theta$ 和 $\varphi$ 来估计 g-公式的组成部分。那么，Robins 和 Wasserman（1997）证明，当 $L_k$ 具有任何离散分量时，这些模型不能全部正确设定，因为 $E \left[ Y^{\bar{a}} \right]$ 的 g-公式估计值通常依赖于 $\bar{a}$ 。因此，基于估计的 g-公式的推断在理论上可能导致严格零假设被错误拒绝，即使在顺序随机化实验中也是如此。

这种现象被称为时变治疗的估计 g-公式的 **零悖论（null paradox）** 。更多讨论见 Cox 和 Wermuth（1999）以及 McGrath 等人（2022）。幸运的是，在实践中，g-零悖论并未阻止零参数化 g-公式效应估计，可能是因为该悖论引起的偏倚相比典型的随机变异性较小。

相比之下，如第 12 章和第 14 章所述，边际结构均值模型的 IP 加权和结构嵌套均值模型的 g-估计都不受零悖论的影响。无论我们为治疗选择何种函数形式，这些模型在严格零假设下都是正确设定的。例如，边际结构均值模型

$$
E \left[ Y^{\bar{a}} \right] = \beta_0 + \beta_1 \, cum \left( \bar{a} \right)
$$

在零假设下是正确设定的，因为在这种情况下 $\beta_1 = 0$ ，且 $E \left[ Y^{\bar{a}} \right]$ 不依赖于 $\bar{a}$ 的函数。同样，如第 21.4 节所定义，任何结构嵌套均值模型 $\gamma_k \left( \overline{a}_{k-1}, \overline{l}_k, \beta \right)$ 在严格零假设下都是正确设定的，其中 $\beta = 0$ 是真实参数值，且 $\gamma_k \left( \overline{a}_{k-1}, \overline{l}_k, \beta \right) = 0$ ，无论过去治疗和协变量历史的函数形式如何。

### 21.3 时变治疗的双重稳健估计量（A Doubly Robust Estimator for Time-Varying Treatments）

在本节中，我们介绍时变治疗因果效应的双重稳健估计量。该估计量结合了治疗分配机制和结局过程的模型，确保至少其中一个模型正确设定时估计的一致性。

#### 21.3.1 定义与关键假设（Definition and Key Assumptions）

考虑时间点 $t = 1, \dots, T$ 的纵向设定。令：

- $\bar{A}\_t = (A_1, \dots, A_t)$ 表示截至时间 $t$ 的治疗历史。
- $\bar{L}\_t = (L_1, \dots, L_t)$ 表示截至时间 $t$ 的协变量历史。
- $Y$ 表示感兴趣的最终结局。

目标是估计治疗方案 $\bar{a}\_T$ 对 $Y$ 的因果效应。双重稳健估计量依赖于以下假设：

1.  **一致性（Consistency）** ：如果实际治疗与 $\bar{a}\_T$ 匹配，则 $Y = Y^{\bar{a}\_T}$ 。
2.  **序贯可忽略性（Sequential Ignorability）** ：对于所有 $t$ ，有 $Y^{\bar{a}_T} \perp A_t \mid \bar{L}\_t, \bar{A}_{t-1}$ 。
3.  **积极性（Positivity）** ：对于所有可能取值，有 $P(A*t = a_t \mid \bar{L}\_t, \bar{A}*{t-1}) > 0$ 。

#### 21.3.2 双重稳健估计方程（The Doubly Robust Estimating Equation）

双重稳健估计量来源于以下估计方程：

$$
\sum*{i=1}^n \left[ \frac{I(\bar{A}\_T = \bar{a}\_T)}{\prod*{t=1}^T \pi*t(\bar{L}\_t, \bar{A}*{t-1})} (Y_i - \mu(\bar{L}\_T, \bar{a}\_T)) + \mu(\bar{L}\_T, \bar{a}\_T) \right] = 0
$$

其中：

- $\pi*t(\bar{L}\_t, \bar{A}*{t-1}) = P(A*t = a_t \mid \bar{L}\_t, \bar{A}*{t-1})$ 是时间 $t$ 的治疗倾向性。
- $\mu(\bar{L}\_T, \bar{a}\_T) = E[Y \mid \bar{L}_T, \bar{A}_T = \bar{a}_T]$ 是结局回归函数。

> **注意** ：项 $I(\bar{A}\_T = \bar{a}\_T)$ 是指示函数，当观测到的治疗历史与感兴趣的方案匹配时取值为 1，否则为 0。

#### 21.3.3 实施步骤（Implementation Steps）

要实施双重稳健估计量，请遵循以下步骤：

1.  **对治疗分配机制建模** ：

- 为每个时间点 $t$ 估计 $\pi*t(\bar{L}\_t, \bar{A}*{t-1})$ 。
- 常用方法包括逻辑回归或更灵活的机器学习方法。

2.  **对结局过程建模** ：

- 使用回归模型估计 $\mu(\bar{L}\_T, \bar{a}\_T)$ 。
- 这可以是线性模型、 **广义可加模型（generalized additive model）** 或其他合适的技术。

3.  **组合模型** ：

- 通过求解估计方程计算双重稳健估计。
- 如果治疗模型或结局模型中至少有一个正确设定，则该估计量是一致的。

#### 21.3.4 示例：两个时间点的二分类治疗（Example: Binary Treatment Over Two Time Points）

考虑 $T = 2$ 且二分类治疗 $A_t \in \{0, 1\}$ 的简单情形。数据结构为：

| 时间  | 协变量 | 治疗  | 结局 |
| ----- | ------ | ----- | ---- |
| $t=1$ | $L_1$  | $A_1$ | —    |
| $t=2$ | $L_2$  | $A_2$ | $Y$  |

方案 $\bar{a}\_2 = (1, 1)$ 的双重稳健估计量为：

$$
\hat{\mu}_{\text{DR}} = \frac{1}{n} \sum_{i=1}^n \left[ \frac{I(A_{1i}=1, A_{2i}=1)}{\hat{\pi}_1(L_{1i}) \hat{\pi}_2(L_{2i}, A_{1i}=1)} (Y_i - \hat{\mu}(L_{2i}, 1, 1)) + \hat{\mu}(L_{2i}, 1, 1) \right]
$$

其中：

- $\hat{\pi}\_1(L_1) = P(A_1 = 1 \mid L_1)$
- $\hat{\pi}\_2(L_2, A_1 = 1) = P(A_2 = 1 \mid L_2, A_1 = 1)$
- $\hat{\mu}(L_2, 1, 1) = E[Y \mid L_2, A_1 = 1, A_2 = 1]$

计算该估计量的 **算法伪代码（Algorithm pseudocode）** ：

- 对每个个体 $i$ ：
  - 估计倾向性得分：
    - 根据 $L*1$ 对 $A_1$ 的模型得到 $\hat{\pi}*{1i}$
    - 根据 $L*2$ 和 $A_1$ 对 $A_2$ 的模型得到 $\hat{\pi}*{2i}$
  - 估计结局回归：
    - 根据 $L_2$ 、 $A_1$ 和 $A_2$ 对 $Y$ 的模型得到 $\hat{\mu}\_i$
  - 计算权重：
    - $w*i = \frac{I(A*{1i}=1, A*{2i}=1)}{\hat{\pi}*{1i} \hat{\pi}\_{2i}}$
  - 计算贡献：
    - $\text{contrib}\_i = w_i (Y_i - \hat{\mu}\_i) + \hat{\mu}\_i$
- 对所有个体求平均： $\hat{\mu}\_{\text{DR}} = \frac{1}{n} \sum_i \text{contrib}\_i$

#### 21.3.5 性质与优势（Properties and Advantages）

双重稳健估计量具有以下几个重要优势：

好的，请查收根据您的要求翻译的文本。

**双重稳健性（Double robustness）** ：如果处理模型或结果模型中的任何一个被正确设定，它就能保持一致估计。
**高效性（Efficiency）** ：当两个模型都正确时，该估计量能够达到 **半参数效率界（semiparametric efficiency bound）** 。
**灵活性（Flexibility）** ：它可以处理复杂的、高维的协变量和时变处理。

> **重要提示（Important）** ：尽管该估计量具有稳健性，但如果两个模型都被错误设定，其表现可能会很差。谨慎的模型选择和验证至关重要。

#### 21.3.6 实际考量（Practical Considerations）

在实际应用双重稳健估计量时，请考虑以下几点：

- **重叠性检查（Overlap checking）** ：确保 **倾向性得分（propensity scores）** 有足够的重叠，以避免极端权重。
- **模型诊断（Model diagnostics）** ：使用标准诊断工具评估处理模型和结果模型。
- **小样本偏差（Small sample bias）** ：在小样本中，该估计量可能表现出偏差；考虑使用 **样本分割（sample-splitting）** 或 **交叉拟合（cross-fitting）** 技术。

关键组件的 **汇总表（Summary Table）** ：

| Component               | Model                              | Role                                  |
| ----------------------- | ---------------------------------- | ------------------------------------- |
| Treatment model         | $\pi*t(\bar{L}\_t, \bar{A}*{t-1})$ | Weights observations                  |
| Outcome model           | $\mu(\bar{L}\_T, \bar{a}\_T)$      | Predicts counterfactual outcomes      |
| Doubly robust estimator | Combined                           | Consistent if either model is correct |

双重稳健估计量给了我们两次做对的机会，这在大多数观察性研究中，当存在许多混杂因素且需要使用非饱和模型时尤其有用。

第二部分简要提到了结合 **逆概率加权（IP weighting）** 和 **g 公式（g-formula）** 的双重稳健方法。正如我们所知，逆概率加权需要一个关于在混杂因素 $L$ 条件下处理 $A$ 的正确模型，而 g 公式需要一个关于在处理 $A$ 和混杂因素 $L$ 条件下结果 $Y$ 的正确模型。双重稳健方法只需要一个关于处理 $A$ 或结果 $Y$ 的正确模型。如果这两个模型中至少有一个是正确的（并且我们不需要知道哪一个模型是正确的），那么双重稳健估计量就能一致地估计因果效应。

精细要点 13.2 描述了一个用于固定时间处理 $A$ 对结果 $Y$ 的平均因果效应的双重稳健插件估计量。在本节中，我们首先回顾一个略有不同的、用于固定时间处理的双重稳健插件估计量，然后将其扩展到时变处理。

假设我们想要构建一个平均因果效应 $\operatorname{E}\left[ Y^{a=1} \right] - \operatorname{E}\left[ Y^{a=0} \right]$ 的双重稳健估计量，其中 $A$ 是一个固定时间的二元处理， $Y$ 是一个二元结果，在存在许多混杂因素 $L$ 且满足可交换性、正性和一致性假设的设定下。我们将按照技术要点 13.2 和 13.3 中先前讨论的那样，构建 $\mathrm{E}\left[ Y^{a} \right]$ 的双重稳健估计量。 $\mathrm{E}\left[ Y^{a=1} \right]$ 和 $\mathrm{E}\left[ Y^{a=0} \right]$ 的双重稳健估计量之差，就是平均因果效应的双重稳健估计量。我们用于 $\mathrm{E}\left[ Y^{a} \right]$ 的双重稳健程序将使用一个关于 $\operatorname{E}[Y \vert A = a, L = l]$ 的结果模型估计值和一个关于 $\operatorname{Pr}[A = 1 \vert L]$ 的模型估计值，然后适当地将它们结合起来。我们的程序包含三个步骤。

> 这个双重稳健估计量归功于 Bang 和 Robins（2005），并且与一个更早的估计量（Robins, 2000）密切相关。该估计量是一个 **基于目标最小损失的估计量（targeted minimum loss-based estimator, TMLE）** ，在 van der Laan 和 Rubin（2006）以及 van der Laan 和 Gruber（2012）后来引入的术语中，也被称为 **目标最大似然估计量（targeted maximum likelihood estimator）** 。

第一步是从处理模型中计算预测值 $\widehat{f}(a \vert L) \equiv \widehat{\mathrm{Pr}}[A = a \vert L]$ 。第二步是从一个线性逻辑斯蒂模型 $b(a, L; \theta)$ 的最大似然拟合中计算预测值 $\widehat{\operatorname{E}}[Y \vert A = a, L] = b(a, L; \widehat{\theta})$ ，该拟合仅限于 $A = a$ 的个体，并且模型中包含 $\widehat{W}^{a} = 1 / \widehat{f}(a \vert L)$ 作为一个协变量，例如：

$$
b(a, L; \theta) = \operatorname{expit}\left( \theta_{a,0} + \theta_{a,1} L + \theta_{a,2} \widehat{W}^{a} \right).
$$

第三步是将 $\mathrm{E}[Y^{a=1}]$ 和 $\mathrm{E}[Y^{a=0}]$ 估计为标准化均值 $\widehat{\mathrm{E}}\left[ b(1, L; \widehat{\theta}) \right]$ 和 $\widehat{\mathrm{E}}\left[ b(0, L; \widehat{\theta}) \right]$ ，其中 $\widehat{\mathrm{E}}$ 表示对所有个体（包括处理组和未处理组）的样本平均值。差值：

$$
\widehat{\mathrm{E}}\left[ b(1, L; \widehat{\theta}) \right] - \widehat{\mathrm{E}}\left[ b(0, L; \widehat{\theta}) \right]
$$

是因果效应 $\mathrm{E}[Y^{a=1}] - \mathrm{E}[Y^{a=0}]$ 的一个双重稳健估计量。也就是说，在可识别性条件下，如果处理模型正确，或者结果模型正确，该估计量就能一致地估计平均因果效应。

重要的是要认识到，具有相同 $L$ 值的处理组和未处理组个体也具有相同的：

$$
b(1, L; \widehat{\theta}) = \operatorname{expit}\left( \widehat{\theta}_{1,0} + \widehat{\theta}_{1,1} L + \widehat{\theta}_{1,2} / \widehat{f}(a = 1 \vert L) \right).
$$

他们同样也具有相同的：

$$
b(0, L; \widehat{\theta}) = \operatorname{expit}\left( \widehat{\theta}_{0,0} + \widehat{\theta}_{0,1} L + \widehat{\theta}_{0,2} / \widehat{f}(a = 0 \vert L) \right).
$$

现在，让我们将这个双重稳健估计量扩展到具有时变处理的设定，在该设定中，我们感兴趣的是比较两种处理策略 $\bar{a}$ 和 $\bar{a}^{\prime}$ 下的反事实均值 $\mathrm{E}[Y^{\bar{a}}]$ 和 $\mathrm{E}[Y^{\bar{a}^{\prime}}]$ 。用于估计时变处理下 $\mathrm{E}[Y^{\bar{a}}]$ 的双重稳健程序遵循与估计固定时间处理下 $\mathrm{E}[Y^{a}]$ 相同的 3 个步骤。然而，正如我们将看到的，第二步会稍微复杂一些，因为它需要拟合 **序贯回归模型（sequential regression models）** 。

为了简化符号，我们展示如何在处理策略“始终接受处理”下获得 $\mathrm{E}[Y^{\bar{a}}]$ 的双重稳健估计量，即 $\bar{a} = \overline{1}$ ，其中 $\overline{1} = 1_{K}$ 是包含 $K+1$ 个 1 的向量。

第一步需要拟合一个关于以下内容的回归模型 $\pi_{k}(\bar{L}_{k}; \alpha)$ ：

$$
\pi_{k}(\bar{L}_{k}) = \operatorname{Pr}\left[ A_{k} = 1 \vert \bar{A}_{k-1} = \overline{1}_{k-1}, \bar{L}_{k} \right]
$$

该模型汇集了所有个体和所有时间点 $k$ 的数据。一个个体只有在 $k-1$ 时刻之前（持续）接受了处理（即 $A_{k-1} = \overline{1}_{k-1}$ ）时，才会在 $k$ 时刻被纳入该模型的拟合。然后，我们使用来自该模型的预测值 $\pi_{k}(\widehat{L}_{k}; \widehat{\alpha})$ 来估计那些持续接受处理至 $m$ 时刻（ $A_{m} = \overline{1}_{m}$ ）的个体的时变逆概率权重：

$$
W^{A_{m}} = \prod_{k=0}^{m} \frac{1}{f\left( A_{k} \vert \bar{A}_{k-1}, \bar{L}_{k} \right)}
$$

这等于：

$$
W^{\overline{1}_{m}} = \prod_{k=0}^{m} \frac{1}{\pi_{k}(\bar{L}_{k})}.
$$

也就是说，对于一个始终接受处理（ $A_{K} = \overline{1}_{K}$ ）的个体，我们在每个时间点 $m$ 分配一个不同的权重 $W^{\overline{1}_{m}}$ ，而不是像我们在上一节中所做的那样，只在随访结束时分配一个单一的权重 $W^{\overline{1}_{K}}$ 。

例如，如果我们拟合参数模型

$$
\pi*k(\bar{L}\_k; \boldsymbol{\alpha}) = \operatorname{expit}\bigl(\alpha*{0,k} + \alpha_2 L_k\bigr)
$$

用于

$$
\Pr\bigl(A*k = 1 \mid \bar{A}*{k-1} = 1, \bar{L}\_k\bigr),
$$

那么，在我们表 21.1 的例子中，有两个时间点（ $K = 1$ ），预测值

$$
\widehat{\Pr}\bigl[A_1 = 1 \mid A_0 = 1, \bar{L}_1\bigr]
\quad\text{和}\quad
\widehat{\Pr}\bigl[A_0 = 1 \mid L_0\bigr]
$$

是

$$
\widehat{\pi}_1 = \operatorname{expit}\bigl(\widehat{\alpha}_{0,1} + \widehat{\alpha}_2 L_1\bigr)
\quad\text{和}\quad
\widehat{\pi}\_0 = \operatorname{expit}\bigl(\hat{\alpha}_{0,0} + \hat{\alpha}\_2 L_0\bigr)
$$

（因为 $A_{-1} \equiv 0$ ）。

这里，我们使用缩写 $\widehat{\pi}_k$ 来表示 $\pi_k(\bar{L}_k; \widehat{\alpha})$ 。

然后，我们计算时变逆概率权重估计值

$$
\hat{W}^{\bar{1}_m} = \prod_{k=0}^{m} \frac{1}{\widehat{\pi}\_k}
$$

对于持续接受处理至 $m$ 时刻的个体。

至此，我们完成了第一步。

第二步需要在每个时间点 $m$ 拟合一个单独的结果模型 $b_m(L_m; \beta_m)$ ，从最后一个时间点 $K$ 开始，一直到 $m = 0$ 。

时间点 $m$ 的回归模型仅拟合给那些持续接受处理至 $m$ 时刻的个体，并且包含 $\hat{W}^{\bar{1}_m} = \hat{W}^{\overline{A}_m}$ 作为一个协变量。

时间点 $K$ 模型的因变量是 $Y$ 。

对于 $m < K$ ，时间点 $m$ 模型的因变量是来自时间点 $m+1$ 模型拟合的预测结果，即

$$
\widehat{B}_{m+1} = \widehat{b}_{m+1}\bigl(\bar{L}_{m+1}; \beta_{m+1}\bigr).
$$

#### 技术要点 21.4（Technical Point 21.4）

一个 $\mathsf{K}+2$ 重稳健的增广逆概率加权估计量。我们考虑 $K = 1$ 的情况，因为该论证可以推广到任意 $K$ 。g 公式 $\psi$ 的 ICE 插件估计量是

$$
\widehat{\psi}_{gfor} = P_n [ \widehat{b}_0 (L_0) ],
$$

其中 $P_n$ 表示样本平均值， $\widehat{b}_0 (L_0) = \widehat{\mathrm{E}} [ \widehat{b}_1 (L_0, L_1) \mid A_0 = 1, L_0 ]$ ，而 $\widehat{b}_1 (L_0, L_1) = \widehat{\mathrm{E}} [ Y \mid L_0, A_0 = 1, L_1, A_1 = 1 ]$ 。

$\psi$ 的逆概率加权估计量 $\widehat{\psi}_{IPW}$ 是

$$
P_n \left[ \frac{A_0 A_1 Y}{\widehat{\pi}_0 \widehat{\pi}_1} \right],
$$

其中 $\widehat{\pi}_0$ 和 $\widehat{\pi}_1$ 是 $\pi_0 = \mathrm{Pr}(A_0 = 1 \mid L_0)$ 和 $\pi_1 = \mathrm{Pr}(A_1 = 1 \mid L_0, L_1, A_0 = 1)$ 的估计值。

Robins 等人（1994）推导出了 $\psi$ 的一个增广逆概率加权估计量 $\widehat{\psi}_{TR} = P_n [ \widehat{U}_{TR} ]$ ，其中

$$
\widehat{U}_{TR} = \frac{A_0 A_1 Y}{\widehat{\pi}_0 \widehat{\pi}_1} - \frac{A_0}{\widehat{\pi}_0} \left\{\frac{A_1}{\widehat{\pi}_1} - 1 \right\} \widehat{b}_1 (L_0, L_1) - \left\{\frac{A_0}{\widehat{\pi}_0} - 1 \right\} \widehat{b}_0 (L_0).
$$

我们现在展示 $\widehat{\psi}_{T R}$ 是三重（即 $K + 2$ ）稳健的。首先，如果 $\widehat{\pi}_{0}$ 和 $\widehat{\pi}_{1}$ 是一致的，那么 $\widehat{\psi}_{T R}$ 对于 $\psi$ 是一致的（单重稳健），因为此时 $\widehat{U}_{T R}$ 最后两项的样本平均值一致地趋于 0，而第一项的样本平均值正是 $\widehat{\psi}_{I P W}$ 。

其次， $\widehat{\psi}_{T R}$ 是双重稳健的，因为当 $\widehat{\mathrm{E}}[Y | L_{0}, A_{0} = 1, L_{1}, A_{1} = 1]$ 和 $\widehat{\mathrm{E}}[b_{1}(L_{0}, L_{1}) | A_{0} = 1, L_{0}]$ 分别一致地估计 $\operatorname{E}(Y | L_{0}, A_{0} = 1, L_{1}, A_{1} = 1)$ 和 $\operatorname{E}[b_{1}(L_{0}, L_{1}) \mid A_{0} = 1, L_{0}]$ 时， $\widehat{\psi}_{T R}$ 是一致的。这里， $\widehat{\mathrm{E}}[b_{1}(L_{0}, L_{1}) | A_{0} = 1, L_{0}]$ 是将与用于从 $\widehat{b}_{1}(L_{0}, L_{1})$ 得到 $\widehat{b}_{0}(L_{0})$ 相同的回归算法应用于真实的 $b_{1}(L_{0}, L_{1})$ 。

要理解这一点，我们重新排列项得到

$$
\widehat{U}_{T R} = \widehat{b}_{0}(L_{0}) + \frac{A_{0} A_{1}}{\widehat{\pi}_{0} \widehat{\pi}_{1}} (Y - \widehat{b}_{1}(L_{0}, L_{1})) + \frac{A_{0}}{\widehat{\pi}_{0}} (\widehat{b}_{1}(L_{0}, L_{1}) - \widehat{b}_{0}(L_{0})).
$$

最后两项的样本平均值一致地趋于 0，而第一项的样本平均值是 $\widehat{\psi}_{g f o r}$ 。

第三， $\widehat{\psi}_{T R}$ 是三重稳健的，因为如果 $\widehat{b}_{1}(L_{0}, L_{1})$ 和 $\widehat{\pi}_{0}$ 都是一致的（Molina et al. 2017），它就能保持一致估计。这是因为 $\widehat{U}_{T R}$ 可以重写为

$$
\widehat{U}_{T R} = \frac{A_{0} \widehat{b}_{1}(L_{0}, L_{1})}{\widehat{\pi}_{0}} + \frac{A_{0} A_{1}}{\widehat{\pi}_{0} \widehat{\pi}_{1}} (Y - \widehat{b}_{1}(L_{0}, L_{1})) - \left( \frac{A_{0}}{\widehat{\pi}_{0}} - 1 \right) \widehat{b}_{0}(L_{0}).
$$

因此，最后两项的样本平均值收敛到零，第一项的样本平均值收敛到 $\mathrm{E}[b_{0}(L_{0})]$ 。然而，当只有 $\widehat{\pi}_{1}$ 和 $\widehat{\mathrm{E}}[b_{1}(L_{0}, L_{1}) | A_{0} = a_{0}, L_{0}]$ 是一致时，它并不是一致的。

> 通过修改我们的估计量 $\widehat{\psi}_{T R}$ ，我们可以构建一个四重稳健（即 $2^{K+1}$ ）的估计量 $\widehat{\psi}_{Q R}$ ，当只有 $\widehat{\pi}_{1}$ 和 $\widehat{\mathrm{E}}[b_{1}(L_{0}, L_{1}) | A_{0} = a_{0}, L_{0}]$ 是一致时，它也能保持一致估计（Tchetgen Tchetgen 2009）。

令

$$
\widetilde{b}_{0}(L_{0}) = \widehat{\mathrm{E}} \left[ \frac{A_{1} Y}{\widehat{\pi}_{1}} - \left( \frac{A_{1}}{\widehat{\pi}_{1}} - 1 \right) \widehat{b}_{1}(L_{0}, L_{1}) \,\middle|\, A_{0} = 1, L_{0} \right].
$$

那么 $\hat{\psi}_{QR} = \mathcal{P}_n [\widehat{U}_{QR}]$ ，其中 $\widehat{U}_{QR}$ 与 $\widehat{U}_{TR}$ 相同，只是将 $\widehat{b}_0(L_0)$ 替换为 $\widetilde{b}_0(L_0)$ 。

$\widetilde{b}_0(L_0)$ 相对于 $\widehat{b}_0(L_0)$ 的优势在于， $\widetilde{b}_0(L_0)$ 本身是双重稳健的，即如果 $\widehat{\mathrm{E}}[b_1(L_0, L_1) \mid A_0 = 1, L_0]$ 一致地估计 $b_0(L_0)$ ，并且 $\widehat{\pi}_1$ 或 $\widehat{b}_1(L_0, L_1) = \widehat{\mathrm{E}}[Y \mid L_0, A_0 = 1, L_1, A_1 = 1]$ 是一致的，那么它就能一致地估计 $b_0(L_0) = \operatorname{E}[b_1(L_0, L_1) \mid A_0 = 1, L_0]$ ，这意味着 $\widehat{\psi}_{QR}$ 是四重稳健的。

对于二元 $Y$ ，我们可以拟合一个逻辑斯蒂模型 $b_m(\bar{L}_m; \beta_m) = \text{expit}[\gamma_m X_m + \varsigma_m \hat{W}^{\bar{1}_m}]$ ； $X_m$ 是协变量 $\bar{L}_m$ 的向量函数， $\beta_m = (\gamma_m, \varsigma_m)$ 。即使 $\widehat{B}_K$ 不是一个整数，但它保证在 $[0,1]$ 范围内，因此可以作为逻辑斯蒂模型中的结果变量。对于连续 $Y$ ，我们可以拟合一个线性回归模型 $\gamma_m X_m + \varsigma_m \hat{W}^{\bar{1}_m}$ 。

我们得到预测值 $\widehat{B}_0 = b_0(\bar{L}_0; \widehat{\beta})$ ，并完成了第二步。

在第三步中，我们将 $\widehat{\mathrm{E}}[Y^{\bar{a} = \overline{1}}]$ 估计为所有个体 $\widehat{B}_0$ 的样本平均值。

如果 (i) 对于所有 $m$ ，结果模型 $b_m(\bar{L}_m; \beta_m)$ 都被正确设定，或者 (ii) 对于所有 $m$ ，处理模型 $\pi_k(\bar{L}_k; \alpha)$ 都被正确设定，那么 $\widehat{\mathrm{E}}[Y^{\bar{a} = \overline{1}}]$ 将是 $\operatorname{E}[Y^{\bar{a} = \overline{1}}]$ 的（渐近）无偏估计。在这种情况下， $\widehat{\mathrm{E}}[Y^{\bar{a} = \overline{1}}]$ 被称为是 **双重稳健（doubly robust）** 的。

然而， $\widehat{\mathrm{E}}[Y^{\bar{a} = \overline{1}}]$ 实际上是 **多重稳健（multiply robust）** 的，因为对于任何 $m \in \{0, 1, \dots, K - 1\}$ ，如果处理模型对于时间 $0$ 到 $m$ 是正确的，并且结果模型对于时间 $m + 1$ 到 $K$ 是正确的，那么它也是 $\operatorname{E}[Y^{\bar{a} = \overline{1}}]$ 的（渐近）无偏估计。我们将估计量的这一性质称为 ** $K + 2$ 重稳健性（ $K + 2$ robustness）** 。

在技术要点 21.4 和 21.5 中，我们解释了为什么这些稳健性性质成立，并且我们展示了存在比 $\widehat{\mathrm{E}}[Y^{\bar{a} = \overline{1}}]$ 具有更好稳健性性质的估计量。事实上，我们

#### 技术要点 21.5（Technical Point 21.5） 插件式 $K+2$ 稳健估计量（A Plug-in $K+2$ Robust Estimator）

技术要点 21.12 中估计量 $\widehat{\psi}_{TR}$ 的一个潜在缺点是，对于二值变量 $Y$ ， $\widehat{\psi}_{TR}$ 在给定样本中可能落在 $\psi$ 的支持域 $[0, 1]$ 之外。相反， $\widehat{\psi}_{gfor} = P_n \left[ \widehat{b}_0 (L_0) \right]$ 是 $\psi$ 的 **插件式估计量（plug-in estimator）** ，如果使用（参数或非参数）逻辑回归模型估计 $\operatorname{E}[Y \mid L_0, A_0 = a_0, L_1, A_1 = a_1]$ 和 $b_0 (L_0) = \operatorname{E}[b_1 (L_0, L_1) \mid A_0 = a_0, L_0]$ ，则该估计量始终落在 $[0, 1]$ 范围内。

如果对于

$$
\widehat{U}_{TR} = \widehat{b}_0 (L_0) + \frac{A_0 A_1}{\widehat{\pi}_0 \widehat{\pi}_1} (Y - \widehat{b}_1 (L_0, L_1)) + \frac{A_0}{\widehat{\pi}_0} (\widehat{b}_1 (L_0, L_1) - \widehat{b}_0 (L_0))
$$

可以保证在每一个样本中， $P_n \left[ \frac{A_0 A_1}{\widehat{\pi}_0 \widehat{\pi}_1} (Y - \widehat{b}_1 (L_0, L_1)) \right]$ 和 $P_n \left[ \frac{A_0}{\widehat{\pi}_0} (\widehat{b}_1 (L_0, L_1) - \widehat{b}_0 (L_0)) \right]$ 均为零，则可以得到一个也是 **三重稳健（triply robust）** 的插件式估计量 $\widehat{\psi}_{TR, plug} = P_n \left[ \widehat{b}_0 (L_0) \right]$ 。

例如，通过在 $b_1 (L_0, L_1) = \bar{\operatorname{E}} \left[ Y \mid L_0, A_0 = 1, L_1, \bar{A}_1 = 1 \right]$ 的线性逻辑模型中包含一个单变量项 $\theta_1 \left\{\frac{A_0 A_1}{\widehat{\pi}_0 \widehat{\pi}_1} \right\}$ ，并以 $Y$ 为因变量，对 $A_0 = A_1 = 1$ 的个体使用最大似然拟合，可以实现 $P_n \left[ \frac{A_0 A_1}{\widehat{\pi}_0 \widehat{\pi}_1} (Y - \widehat{b}_1 (L_0, L_1)) \right] = 0$ 。接下来，通过在

$$
b_0 (L_0) \equiv \operatorname{E} [b_1 (L_0, L_1) \mid A_0 = a_0, L_0]
$$

的逻辑模型中包含一项 $\theta_0 \frac{A_0}{\widehat{\pi}_0}$ ，以 $\widehat{b}_1 (L_0, L_1)$ 为因变量，对 $A_0 = 1$ 的个体使用最大化逻辑似然拟合，可以实现 $P_n \left[ \frac{A_0}{\widehat{\pi}_0} (\widehat{b}_1 (L_0, L_1) - \widehat{b}_0 (L_0)) \right] = 0$ 。

正文中给出的估计量 $\widehat{\operatorname{E}} \left[ Y^{\bar{a} = \bar{1}} \right]$ 是 $\widehat{\psi}_{TR, plug}$ 的一个实例。Molina 等人 (2017) 指出，该估计量实际上是 ** $K+2$ 稳健（ $K+2$ robust）** 的。Rotnitzky 等人 (2017) 研究了当使用非参数和机器学习估计量处理治疗和结局回归函数时，该估计量及其他多重稳健估计量的渐近偏倚。

他们展示了一个 $\operatorname{E} \left[ Y^{\bar{a} = \bar{1}} \right]$ 的估计量，该估计量是 ** $2^{K+1}$ 稳健（ $2^{K+1}$ robust）** 的。

为了估计治疗策略"从未治疗"下的反事实均值 $\operatorname{E} \left[ Y^{\bar{a} = \bar{0}} \right]$ ，使用 $\bar{a} = \bar{0}$ 重复上述步骤，其中 $\bar{a} = \bar{0}_K$ 是 $K+1$ 个零组成的向量。每种策略下估计的均值之差是 **平均因果效应（average causal effect）** $\operatorname{E} \left[ Y^{\bar{a} = \bar{1}} \right] - \operatorname{E} \left[ Y^{\bar{a} = \bar{0}} \right]$ 的一个多重稳健估计量。

此处描述的多重稳健估计量只能用于估计 **静态（static）** 治疗策略 $a$ 下的反事实均值 $\operatorname{E} \left[ Y^a \right]$ 。技术要点 21.6 描述了治疗策略 $g$ 下反事实均值 $\operatorname{E} \left[ Y^g \right]$ 的多重稳健估计量，其中 $g$ 可以是静态或动态的，也可以是确定性或随机的。该估计量有时被称为 **基于目标最小损失的估计量（Targeted Minimum Loss-based Estimator, TMLE）** 。

多重稳健估计量的实现历来受到计算限制和缺乏用户友好型软件的阻碍，尤其是基于风险的生存分析。我们预计，在不久的将来，软件将变得可用，并且这些多重稳健估计量（使用机器学习和样本分裂拟合）在研究复杂治疗策略对失效时间结局的影响时将变得更加常见。关于 g-公式的不同表示及其与上述估计量之间联系的描述，请参见精细要点 21.2。

## 21.4 时变治疗的 G-估计（G-Estimation for Time-Varying Treatments）

如果我们只对表 21.1 中时间固定治疗 $A_1$ 的效应感兴趣，我们可以求助于第 14 章中描述的 **结构性嵌套均值模型（Structural Nested Mean Models, SNMMs）** ，用于协变量水平内时间固定治疗的条件因果效应。这些模型只有一个方程，因为

### 技术要点 21.6（Technical Point 21.6）

**一个多重稳健估计量。** 令 $f^{g}\left(a_{m} \vert \overline{a}_{m-1}, \overline{l}_{m}\right)$ 表示策略 $g$ 下时间 $m$ 的治疗密度。对于静态 $\overline{a}^{*}$ ， $f^{g}\left(a_{m} | \overline{a}_{m-1}, \overline{l}_{m}\right) = \mathrm{I}\left(a_{m} = a_{m}^{*}\right)$ ；对于确定性动态 $g$ ， $f^{g}\left(a_{m} | \overline{a}_{m-1}, \overline{l}_{m}\right) = \mathrm{I}\big(a_{m} = g_{m}\left(\overline{a}_{m-1}, \overline{l}_{m}\right)\big)$ ；对于随机动态 $f^{int}$ ， $f^{g}\left(a_{m} | \overline{a}_{m-1}, \overline{l}_{m}\right) = f^{int}\left(a_{m} | \overline{a}_{m-1}, \overline{l}_{m}\right)$ 。

令 $C_{k}^{g} = \mathrm{I}\left(\prod_{m=0}^{k} f^{g}\left(A_{m} | \overline{A}_{m-1}, \overline{L}_{m}\right) = 0\right)$ ，如果个体的观测治疗史 $\overline{A}_{k}$ 与 $g$ 兼容，则等于 0，否则等于 1。

以下算法基于 Rotnitzky 等人 (2017) 提出的方法，计算 $\psi = \operatorname{E}\left[Y^{g}\right]$ 的一个多重稳健插件式估计量 $\widehat{\psi}_{dr,plug}$ ，该方法与 Robins (2000)、Bang 和 Robins (2005)、van der Laan 和 Gruber (2012) 以及 Petersen 等人 (2014) 的估计量密切相关。

1. 对于 $m = 0, 1, \ldots, K$ ，拟合模型 $f_{m}\left(a_{m} \vert \overline{a}_{m-1}, \bar{l}_{m}; \alpha_{m}\right)$ 以估计 $f\left(a_{m} | \overline{a}_{m-1}, \overline{l}_{m}\right)$ 。获得向量参数 $\alpha_{m}$ 的 **最大似然估计（MLE）** $\hat{\alpha}_{m}$ 。对于每个时间 $m$ ，计算权重

   $$
   \hat{W}^{g,m} = \prod_{k=0}^{m} \frac{f^{g}\left(A_{k} | \bar{A}_{k-1}, \bar{L}_{k}\right)}{f_{k}\left(A_{k} | \bar{A}_{k-1}, \bar{L}_{k}; \hat{\alpha}_{k}\right)}.
   $$

2. 设置 $\hat{T}_{K+1} = Y$ 。

3. 递归地，对于 $m = K, K-1, \ldots, 0$ ：
   - (a) 拟合一个 **广义线性模型（Generalized Linear Model, GLM）**
     $$
     b_{m}\left(\bar{A}_{m}, \bar{L}_{m}; \gamma_{m}, \varsigma_{m}\right) = \phi\left[\gamma_{m} d_{m}\left(\bar{A}_{m}, \bar{L}_{m}\right) + \varsigma_{m} \hat{W}^{g,m}\right],
     $$
     其中 $\phi$ 是逆典型连接函数，用于估计条件期望 $\operatorname{E}\left[\hat{T}_{m+1} | \bar{A}_{m}, \bar{L}_{m}, C_{m}^{g} = 0\right]$ ，通过 **迭代加权最小二乘法（Iteratively Reweighted Least Squares, IRLS）** 在 $C_{m}^{g} = 0$ 的个体中进行拟合；则 $(\widehat{\gamma}_{m}, \widehat{\varsigma}_{m})$ 满足
     $$
     \widehat{\operatorname{E}}\left\{\mathrm{I}\left(C_{m}^{g} = 0\right) \binom{d_{m}\left(\bar{A}_{m}, \bar{L}_{m}\right)}{\hat{W}^{g,m}} \left( \hat{T}_{m+1} - b_{m}\left(\bar{A}_{m}, \bar{L}_{m}; \widehat{\gamma}_{m}, \widehat{\varsigma}_{m}\right) \right) \right\} = 0.
     $$
   - (b) 设置
     $$
     \hat{T}_{m} = \sum_{a_{m}} b_{m}\left(a_{m}, \bar{A}_{m-1}, \bar{L}_{m}; \widehat{\gamma}_{m}, \widehat{\varsigma}_{m}\right) f^{g}\left(a_{m} | \bar{A}_{m-1}, \bar{L}_{m}\right).
     $$

4. $\widehat{\psi}_{dr,plug} = \widehat{\mathrm{E}}\left[\widehat{T}_0\right]$

正如 Molina 等人 (2017) 所指出的， $\widehat{\psi}_{dr,plug}$ 是 ** $K+2$ 稳健（ $K+2$ robust）** 的，因为除了具有双重稳健性之外，当对于任意 $p \in \{1,\ldots,K\}$ ，模型 $b_m\left(\bar{A}_m,\bar{L}_m;\gamma_m,\bar{\varsigma}_m\right)$ 对于 $m \in \{K,\ldots,p\}$ 被正确指定，且模型 $f_m\left(a_m|\overline{a}_{m-1},\overline{l}_m;\alpha_m\right)$ 对于 $m \in \{p-1,\ldots,0\}$ 被正确指定时，它（渐近地）对 $\psi$ 也无偏。

当 $\hat{W}^{g,m}$ 不作为协变量使用时，上述算法计算 $\operatorname{E}[Y^g]$ 的 g-公式的 **迭代条件期望（Iterative Conditional Expectation, ICE）** 估计量（精细要点 21.2），这是 g-公式的一个非双重稳健估计量。

只有一个时间点 $k = 0$ 。对时变治疗的扩展要求模型指定与数据中时间点数量一样多的方程。对于表 21.1 中两个时间点的时变治疗 $\overline{A} = (A_0,A_1)$ ，我们指定一个（饱和的）加性结构性嵌套均值模型，包含两个方程：

- 对于时间 $k = 0$ ： $\operatorname{E}\left[Y^{a_0,a_1=0} - Y^{a_0=0,a_1=0} | A_0 = a_0\right] = \beta_0 a_0$
- 对于时间 $k = 1$ ：

$$
= a_1 \left(\beta_{11} + \beta_{12} l_1 + \beta_{13} a_0 + \beta_{14} a_0 l_1\right)
$$

根据一致性，时间 $k = 1$ 的条件期望可以写为 $\operatorname{E}\left[Y^{a_0,a_1} - Y^{a_0,a_1=0} | L_1 = l_1, A_0 = a_0, A_1 = a_1\right]$ 。由于我们假设 $Y$ 满足 **序列可交换性（sequential exchangeability）** ，我们可以并且将会 (i) 将 $k = 0$ 的条件期望替换为 $\operatorname{E}\left[Y^{a_0,a_1=0} - Y^{a_0=0,a_1=0}\right]$ ，因为 $A_0 = a_0$ 可以从条件事件中移除，以及 (ii) 将 $k = 1$ 的条件期望替换为 $\mathrm{E}\left[Y^{a_0,a_1} - Y^{a_0,a_1=0} | L_1^{a_0} = l_1, A_0 = a_0\right]$ ，因为 $A_1^{u_0} = a_1$ 可以从条件事件中移除。

### 精细要点 21.2（Fine Point 21.2）

**g-公式的表示（Representations of the g-formula）。**  
g-公式可以通过多种方式在数学上表示。这些 g-公式的不同表示在非参数意义上是等价的，但在实践中会导致不同的估计量。在本书中，我们一直强调 g-公式的一种表示，即标准化（在流行病学术语中）的推广版本。也就是说，对于时间固定治疗，均值结局的 g-公式为 $\sum_{l} \operatorname{E}\left[Y | A = a, L = l\right] f(l)$ ；对于时变治疗，如本章所述，为 $\sum_{\bar{l}} \operatorname{E}\left[Y | \bar{A} = \bar{a}, \bar{L} = \bar{l}\right] \prod_{k=0}^{K} f\left(l_k | \bar{a}_{k-1}, \bar{l}_{k-1}\right)$ 。由于基于这种 g-公式表示的插件式估计量需要估计混杂因素的联合密度 $\prod_{k=0}^{K} f\left(l_k | \bar{a}_{k-1}, \bar{l}_{k-1}\right)$ 随时间的变化，我们将其称为 g-公式的 **联合密度建模估计量（joint density modeling estimator）** 。

g-公式的另一种表示是迭代条件期望。对于时间固定治疗，我们在第 13.3 节中隐式地使用了这种 g-公式表示 $\operatorname{E}\left[\operatorname{E}\left[Y | A = a, L = l\right]\right]$ 。对于时变治疗，该表示是一种可以递归定义的迭代条件期望 (ICE) (Robins 1986)。基于 g-公式的 ICE 表示的插件式估计量需要拟合顺序预测算法（例如，回归模型）。ICE 估计量在第 21.3 节和技术要点 21.4 中进行了描述，我们将其与 IP 权重的估计相结合，以构建双重（实际上是 $K+2$ ）稳健估计量。

g-公式的另一种表示是 IP 加权。事实上，正如技术要点 2.3 中对时间固定治疗所示，在正性条件下，标准化均值和 IP 加权均值是相等的。对于时变治疗也是如此 (Robins and Rotnitzky, 1992; Robins, 1993; Young et al., 2014)。如本章所述，基于 g-公式的 IP 加权表示的估计量需要估计给定过去治疗和协变量史的治疗条件密度随时间的变化。我们将这些估计量称为 **IP 加权估计量（IP weighted estimators）** ，而不是 g-公式估计量。

#### $a_{1}$ 的效应（Effect of $a_{1}$ ）是：

- 在 $A_{0}=0$ , $L_{1}^{a_{0}=0}=0$ 的个体中： $\beta_{11}$
- 在 $A_{0}=0$ , $L_{1}^{a_{0}=0}=1$ 的个体中： $\beta_{11} + \beta_{12}$
- 在 $A_{0}=1$ , $L_{1}^{a_{0}=1}=0$ 的个体中： $\beta_{11} + \beta_{13}$
- 在 $A_{0}=1$ , $L_{1}^{a_{0}=1}=1$ 的个体中： $\beta_{11} + \beta_{12} + \beta_{13} + \beta_{14}$

根据一致性，当 $A_{0}=a_{0}$ 时， $L_{1}^{a_{0}} = L_{1}$ 。因此，时间 $k=1$ 的方程对由 $(A_{0}, L_{1})$ 定义的 4 个治疗和协变量史中的每一个，建模了时间 $k=1$ 的治疗效应。

模型的这一部分是饱和的，因为第二个方程中的 4 个参数 $\beta_{1}$ 在 4 个可能的过去治疗和协变量史水平内参数化了 $a_{1}$ 对 $Y$ 的效应。第一个方程建模了当时间 $k=1$ 的治疗设为零时，时间 $k=0$ 的治疗效应。模型的这一部分也是饱和的，因为它有一个参数 $\beta_{0}$ 来估计唯一可能历史中的效应（没有先前的治疗或协变量，所以每个人都有相同的历史）。

结构性嵌套模型的两个方程是该模型被称为"嵌套"的原因。第一个方程建模了在时间 0 接受治疗且此后不再接受治疗的效应，第二个方程建模了在时间 1 接受治疗且此后不再接受治疗的效应，如果我们有更多的时间点，依此类推。

让我们使用 **g-估计（g-estimation）** 来估计我们的结构性嵌套模型（ $K=1$ ）的参数。我们遵循与第 14 章相同的方法。我们首先考虑每个个体 $i$ 的 **加性保秩结构性嵌套模型（additive rank-preserving structural nested model）** ：

$$
Y_{i}^{a_{0}, 0} = Y_{i}^{0, 0} + \psi_{0} a_{0}
$$

$$
Y_{i}^{a_{0}, a_{1}} = Y_{i}^{a_{0}, 0} + \psi_{11} a_{1} + \psi_{12} a_{1} L_{1,i}^{a_{0}} + \psi_{13} a_{1} a_{0} + \psi_{14} a_{1} a_{0} L_{1,i}^{a_{0}}
$$

第二个方程被限制在 $A_{0} = a_{0}$ 的个体中。也就是说，第二个方程实际上是两个方程：一个针对 $A_{0} = 1$ 的个体，另一个针对 $A_{0} = 0$ 的个体。这使得我们能够通过一致性将 $L_{1,i}^{u_{0}}$ 替换为 $\boldsymbol{L}_{1,i}$ ，当如图 19.6 所示，我们没有 $L_{1}$ 的序贯可交换性（sequential exchangeability）时，这对于从观测数据中识别模型参数是必需的。

为了简化符号，我们用 $Y_{i}^{0,0}$ 表示 $Y_{i}^{a_{0}=0, a_{1}=0}$ 。证明可见 Robins (1994)。需要注意的是，为了通过 **G 估计（g-estimation）** 拟合一个非饱和的 **结构嵌套均值模型（structural nested mean model）** ， **正性（positivity）** 并非必要条件。

第一个方程是一个 **保秩模型（rank-preserving model）** ，因为效应 $\psi_{0}$ 对每个个体完全相同。因此，如果个体 $i$ 的 $Y_{i}^{0,0}$ 超过个体 $j$ 的 $Y_{j}^{0,0}$ ，那么对于 $Y^{1,0}$ ，个体 $i$ 和 $j$ 的排序将保持不变——该模型跨策略保持秩次。此外，在方程 2 下，如果个体 $i$ 的 $Y_{i}^{1,0}$ 超过个体 $j$ 的 $Y_{j}^{1,0}$ ，我们只能确定个体 $i$ 的 $Y_{i}^{1,1}$ 也超过个体 $j$ 的 $Y_{j}^{1,1}$ ，当且仅当两者具有相同的 $A_{0,i}$ 值 $a_{0}$ 和 $L_{1,i} = L_{1,i}^{a_{0}}$ 值 $l_{1}$ 。由于秩次的保留依赖于局部因素（即 $L_{1}^{a_{0}=1}$ 的值），我们将第二个方程称为 **条件保秩模型（conditionally rank-preserving model）** 或 **局部保秩模型（locally rank-preserving model）** 。

正如第 14 章所讨论的，由于未测量的遗传和环境风险中的个体异质性， **保秩性（rank preservation）** 在生物学上是不可信的。这就是为什么我们的主要兴趣在于 **结构嵌套均值模型（structural nested mean model）** ，该模型对于是否存在由于未测量因素导致的个体间额外效应异质性完全不敏感。然而，给定 $Y$ 的序贯可交换性，对于保秩模型，一类 $\psi$ 的 G 估计量（如下所述）对于 **均值模型（mean model）** 的参数 $\beta$ 是一致的，即使保秩模型被错误设定。

**G 估计（G-estimation）** 的第一步是将模型与观测数据联系起来，正如我们在第 14 章中对 **时间固定处理（time-fixed treatment）** 所做的那样。为此，请注意，根据一致性，反事实结果 $Y^{a_{0}, a_{1}}$ 等于那些恰好接受了处理值 $a_{0}$ 和 $a_{1}$ 的个体的观测结果 $Y$ 。形式上，对于 $(A_{0} = a_{0}, A_{1} = a_{1})$ 的个体，有 $Y^{a_{0}, a_{1}} = Y^{A_{0}, A_{1}} = Y$ 。类似地，对于 $(A_{0} = a_{0}, A_{1} = 0)$ 的个体，有 $Y^{a_{0}, 0} = Y^{A_{0}, 0}$ ；对于 $A_{0} = a_{0}$ 的个体，有 $L_{1}^{u_{0}} = L_{1}$ 。现在我们可以将结构嵌套模型用观测数据重写为：

$$
\begin{array}{l}
Y^{A_{0}, 0} = Y - \left(\psi_{11} A_{1} + \psi_{12} A_{1} L_{1} + \psi_{13} A_{1} A_{0} + \psi_{14} A_{1} A_{0} L_{1}\right) \\
Y^{0, 0} = Y^{A_{0}, 0} - \psi_{0} A_{0}
\end{array}
$$

（为了简化符号，我们省略了个体索引 $i$ ）。

**G 估计** 的第二步是使用观测数据计算候选反事实 $H_{1}(\psi^{\dagger})$ 和 $H_{0}(\psi^{\dagger})$ 。为此，我们使用结构嵌套模型，将参数的真实值 $\psi$ 替换为某个值 $\psi^{\dagger}$ ：

$$
\begin{array}{l}
H_{1}(\psi^{\dagger}) = Y - \left(\psi_{11}^{\dagger} A_{1} + \psi_{12}^{\dagger} A_{1} L_{1} + \psi_{13}^{\dagger} A_{1} A_{0} + \psi_{14}^{\dagger} A_{1} A_{0} L_{1}\right) \\
H_{0}(\psi^{\dagger}) = H_{1}(\psi^{\dagger}) - \psi_{0}^{\dagger} A_{0}
\end{array}
$$

如第 14 章所述，目标是找到等于真实值 $\psi$ 的参数值 $\psi^{\dagger}$ 。当 $\psi^{\dagger} = \psi$ 且 $\overline{A}_{k-1} = \overline{a}_{k-1}$ 时，候选反事实 $H_{k}(\psi^{\dagger})$ 等于在时间 $k-1$ 之前接受处理 $\boldsymbol{a}_{k-1}$ 且之后接受处理 0 的真实反事实 $Y^{a_{k-1}, \underline{0}_{k}}$ 。我们现在可以使用序贯可交换性在每个时间点进行 G 估计。

**精细点 21.3（Fine Point 21.3）** 描述了如何对非饱和结构嵌套均值模型的参数 $\psi$ 进行 G 估计。结果表明，结构嵌套模型的所有参数均为 0，这意味着在任何静态或动态策略 $g$ 下的所有反事实均值 $\operatorname{E}[Y^{g}]$ 都等于 60。这一结果与通过 **G 公式（g-formula）** 和 **IP 加权（IP weighting）** 获得的结果一致。 **G 估计** ，如同 G 公式和 IP 加权一样，在传统方法失败的情况下取得了成功。

然而，在实践中，我们会遇到具有多个时间点 $k$ 和每个时间点多个协变量 $L_{k}$ 的观测研究。一般而言，一个结构

#### 精细点 21.3（Fine Point 21.3） 使用非饱和结构嵌套模型的 G 估计（G-estimation with a saturated structural nested model）

$k = 1$ 时的序贯可交换性意味着，在 $(A_0, L_1)$ 的四个联合层中的任何一个层内， $A_1 = 1$ 的个体中 $Y^{A_0, 0}$ 的均值等于 $A_1 = 0$ 的个体中的均值。因此，当 $\psi^\dagger = \psi$ 时， $H_1(\psi^\dagger)$ 的均值也必须相等。

首先考虑层 $(A_0, L_1) = (0, 0)$ 。从表 21.2 的数据行 1 和 2，我们发现当 $A_1 = 0$ 时 $H_1(\psi)$ 的均值为 84，当 $A_1 = 1$ 时为 $84 - \psi_{11}$ 。因此 $\psi_{11} = 0$ 。

接下来，我们将对应于层 $(A_0, L_1) = (0, 1)$ 的数据行 3 和 4 中 $H_1(\psi)$ 的均值相等，得到 $52 = 52 - \psi_{11} - \psi_{12}$ 。由于 $\psi_{11} = 0$ ，我们得出结论 $\psi_{12} = 0$ 。

继续，我们将数据行 5 和 6 中 $H_1(\psi)$ 的均值相等，得到 $76 = 76 - \psi_{11} - \psi_{13}$ 。由于 $\psi_{11} = \psi_{12} = 0$ ，我们得出结论 $\psi_{13} = 0$ 。

最后，将数据行 7 和 8 中 $H_1(\psi)$ 的均值相等，我们得到 $44 = 44 - \psi_{11} - \psi_{12} - \psi_{13} - \psi_{14}$ ，因此 $\psi_{14} = 0$ 同样成立。

为了估计 $\psi_0$ ，我们首先将 $\psi_{11}$ 、 $\psi_{12}$ 、 $\psi_{13}$ 和 $\psi_{14}$ 的值代入表 21.2 中 $H_0(\psi)$ 均值的表达式。在这个例子中，所有参数都等于 0，因此 $H_0(\psi)$ 的均值等于观测结果 $Y$ 的均值。

然后，我们使用结构方程模型的第一个方程，通过从 $H_1(\psi)$ 的均值中减去 $\psi_0 A_0$ ，计算表中每个数据行的 $H_0(\psi)$ 均值，如表 21.3 所示。

时间 $k = 0$ 时的序贯可交换性 $Y^{0,0} \perp\!\!\!\perp A_0$ 意味着，在 16,000 名 $A_0 = 1$ 的受试者和 16,000 名 $A_0 = 0$ 的受试者中， $H_0(\psi)$ 的均值是相同的。 $H_0(\psi)$ 的均值为：

- 在 $A_0 = 0$ 的个体中： $84 \times 0.25 + 52 \times 0.75 = 60$ ，
- 在 $A_0 = 1$ 的个体中： $(76 - \psi_0) \times 0.5 + (44 - \psi_0) \times 0.5 = 60 - \psi_0$ 。

因此 $\psi_0 = 0$ 。我们完成了 G 估计。

**表 21.2（Table 21.2）**

| $A_0$ | $L_1$ | $A_1$ | 均值 $H_1(\psi)$                                     |
| ----- | ----- | ----- | ---------------------------------------------------- |
| 0     | 0     | 0     | 84                                                   |
| 0     | 0     | 1     | $84 - \psi_{1,1}$                                    |
| 0     | 1     | 0     | 52                                                   |
| 0     | 1     | 1     | $52 - \psi_{11} - \psi_{12}$                         |
| 1     | 0     | 0     | 76                                                   |
| 1     | 0     | 1     | $76 - \psi_{11} - \psi_{13}$                         |
| 1     | 1     | 0     | 44                                                   |
| 1     | 1     | 1     | $44 - \psi_{11} - \psi_{12} - \psi_{13} - \psi_{14}$ |

**表 21.3（Table 21.3）**

| A0  | L1  | A1  | 均值 H0(ψ) |
| --- | --- | --- | ---------- |
| 0   | 0   | 0   | 84         |
| 0   | 0   | 1   | 84         |
| 0   | 1   | 0   | 52         |
| 0   | 1   | 1   | 52         |
| 1   | 0   | 0   | 76 – ψ0    |
| 1   | 0   | 1   | 76 – ψ0    |
| 1   | 1   | 0   | 44 – ψ0    |
| 1   | 1   | 1   | 44 – ψ0    |

这个 **瞬时效应函数（blip function）** 满足

$$
\gamma*k \left( \overline{a}*{k-1}, \overline{l}\_k, 0 \right) = 0
$$

因此在处理无效应的原假设下 $\beta = 0$ 。

一个 **嵌套均值模型（nested mean model）** 具有与时间点 $k = 0, 1, \dots, K$ 相同数量的方程。我们在正文中讨论的结构嵌套均值模型的最一般形式如下（更一般的结构嵌套均值模型在技术点 21.13（Technical Point 21.13）中讨论）。对于每个时间 $k = 0, 1, \dots, K$ ，

$$
\begin{array}{l}
\mathrm{E} \left[ Y^{\bar{a}_{k-1}, a_k, 0_{k+1}} - Y^{\bar{a}_{k-1}, 0_k} \mid \bar{L}_k^{\bar{a}_{k-1}} = \bar{l}_k, \bar{A}_{k-1} = \bar{a}_{k-1}, A_k = a_k \right] \\
= a*k \gamma_k \left( \bar{a}*{k-1}, \bar{l}\_k, \beta \right)
\end{array}
$$

其中 $\left( \overline{a}_{k-1}, a_k, \underline{0}_{k+1} \right)$ 是一个静态策略，它在时间 0 到 $k-1$ 之间分配处理 $\boldsymbol{a}_{k-1}$ ，在时间 $k$ 分配处理 $a_k$ ，从时间 $k+1$ 到随访结束 $K$ 分配处理 0。策略 $\left( \overline{a}_{k-1}, a_k, \underline{0}_{k+1} \right)$ 和 $\left( \overline{a}_{k-1}, \underline{0}_k \right)$ 仅在以下方面不同：前者在 $k$ 时刻有处理 $a_k$ ，而后者在 $k$ 时刻有处理 0。

这里每个 $\gamma_k \left( \overline{a}_{k-1}, \overline{l}_k, \psi^\dagger \right)$ 是参数向量 $\psi^\dagger$ 的已知函数，满足 $\gamma_k \left( \overline{a}_{k-1}, \bar{l}_k, \psi^\dagger = 0 \right) = 0$ ，而 $\beta$ 是 $\psi^\dagger$ 的真实值。同样，在 $Y$ 的序贯可交换性下，我们可以从上述条件事件中删除 $A_k = a_k$ 。

在我们的 $K = 1$ 的例子中， $\gamma_0 \left( \overline{a}_{-1}, \overline{l}_0, \beta \right)$ 就是 $\beta_0$ （ $l_0$ 和 $\overline{a}_{-1}$ 都可以视为恒等于 0），而 $\gamma_1 \left( \overline{a}_0, l_1, \beta \right)$ 是 $\beta_{11} + \beta_{12} l_1 + \beta_{13} a_0 + \beta_{14} a_0 l_1$ 。

因此， **结构嵌套均值模型（structural nested mean model）** 是 $Y$ 均值上在 $k$ 时刻大小为 $a_k$ 的最后一次处理脉冲效应的模型，作为过去处理和协变量历史 $\left( \overline{a}_{k-1}, l_k \right)$ 的函数 $\gamma_k \left( \overline{a}_{k-1}, \bar{l}_k, \beta \right)$ 。关于结构嵌套模型和 **边际结构模型（marginal structural models）** 之间的关系，请参见技术点 21.7（Technical Point 21.7）。

我们现在准备讨论具有瞬时效应函数 $\gamma_k \left( \overline{a}_{k-1}, l_k, \beta \right)$ 的一般结构嵌套均值模型参数的估计。为了激励我们的估计过程，我们将利用这样一个事实：一个正确设定的、具有真实参数 $\psi$ 的局部保秩模型也是一个正确设定的、具有真实参数 $\beta = \psi$ 的结构嵌套均值模型（尽管反之

#### 技术要点 21.7

**边际结构模型（Marginal structural models）与结构嵌套模型（Structural nested models）** 。一个 **结构嵌套均值模型（Structural nested mean model）** 当且仅当对于所有 $\left( \overline {a}_{k - 1} , \overline {l}_{k} , \beta \right)$ ，满足

$$
\gamma_ {k} (\bar {a}_{k - 1}, \bar {l}_{k}, \beta) = \gamma_ {k} (\bar {a}_{k - 1}, \beta)
$$

即不依赖于 $\bar {l}_{k}$ 时，它便是一个 **半参数边际结构均值模型（Semiparametric marginal structural mean model）** 。具体来说，它是一个具有如下函数形式的半参数边际结构均值模型：

$$
\mathrm {E} \left[ Y^{\overline{a}} \right] = \alpha_ {0} + \sum_ {k = 0}^{K} a_{k} \gamma_ {k} (\bar {a}_{k - 1}, \beta).
$$

其中 $a_{0} = \operatorname{E}\left[Y^{\overline{0}_{K}}\right]$ 是一个未知常数。然而，这样的结构嵌套均值模型并不仅仅是一个边际结构均值模型，因为它还施加了额外的强假设，即 **过去协变量历史（past covariate history）** 不存在效应修饰作用。相比之下， **边际结构模型（Marginal structural model）** 对于是否存在由时变协变量引起的效应修饰作用持不可知论态度。

如果我们指定一个结构嵌套均值模型 $\gamma_{k}\left(\overline{a}_{k-1}, \beta\right)$ ，那么我们可以通过 **g-估计（g-estimation）** 或 **IP 加权（IP weighting）** 来估计 $\beta$ 。然而，当结构嵌套均值模型（以及因此边际结构均值模型）被正确指定时，最有效的 g-估计量将比最有效的 IP 加权估计量更有效，因为 g-估计利用了过去协变量无效应修饰的额外假设来提高效率。

相反，假设边际结构均值模型是正确的，但结构嵌套均值模型是错误的，因为 $\gamma_{k}\left(\overline{a}_{k-1}, \overline{l}_{k}, \beta\right) \neq \gamma_{k}\left(\overline{a}_{k-1}, \beta\right)$ 。那么对 $\beta$ 和 $\operatorname{E}\left[Y^{\overline{a}}\right]$ 的 g-估计将是有偏的，而 IP 加权估计则保持无偏。因此，我们面临一个经典的 **方差-偏差权衡（variance-bias trade-off）** 。给定边际结构模型，如果 $\gamma_{k}\left(\overline{a}_{k-1}, \overline{l}_{k}, \beta\right) = \gamma_{k}\left(\overline{a}_{k-1}, \beta\right)$ ，g-估计可以提高效率，否则（不成立时）则会引入偏差。给定一个结构嵌套均值模型，我们可以定义

$$
H_{k}\left(\psi^{\dagger}\right) = Y - \sum_{j=k}^{K} A_{j} \gamma_{j}\left(\bar{A}_{j-1}, \bar{L}_{j}, \psi^{\dagger}\right)
$$

一个具有真实参数向量 $\psi$ 的、正确指定的 **局部秩保持模型（locally rank preserving model）** 等价于如下陈述： $H_{k}\left(\psi\right)$ 精确等于反事实 $Y^{A_{k-1}, \underline{0}_{k}}$ ，即从时间 $j$ 到 $K$ 的治疗效应已被移除。特别地， $H_{0}\left(\psi\right)$ 是在无治疗情况下的 $Y^{0}$ 值。

然而，如果 **局部秩保持（local rank preservation）** 的假设是错误的（如果存在治疗效应，这基本上总是如此），但结构嵌套均值模型是正确的，我们仍然有 $\operatorname{E}\left[ H_{k}\left(\beta\right) \vert \overline{A}_{k}, \overline{L}_{k} \right]$ 等于 $\operatorname{E}\left[ Y^{\overline{A}_{k-1}, \underline{0}_{k}} | \overline{A}_{k}, \overline{L}_{k} \right]$ ，并且 $\operatorname{E}\left[ H_{0}\left(\beta\right) \right]$ 等于 $\operatorname{E}\left[ Y^{\overline{0}} \right]$ 。因此，如果我们获得 $\widehat{\beta}$ 的一致估计量，则 $\operatorname{E}\left[ Y^{\overline{0}} \right]$ 可以被一致地估计为 $H_{0}\left(\widehat{\beta}\right)$ 的样本均值。这正是 g-估计所提供的。

当存在多个时间点或协变量时，我们需要拟合一个 **非饱和的结构嵌套均值模型（unsaturated structural nested mean model）** 。例如，我们可能假设函数 $\gamma_{k}\left(\overline{a}_{k-1}, \overline{l}_{k}, \beta\right)$ 对于所有 $k$ 都是相同的。最简单的模型是 $\gamma_{k}\left(\overline{a}_{k-1}, l_{k}, \beta\right) = \beta_{1}$ ，它假设最后一次治疗 **脉冲（blip）** 的效应对于所有过去历史和所有时间点 $k$ 都是相同的。其他选择包括 $\beta_{1} + \beta_{2} k$ ，它假设效应随治疗时间点 $k$ 线性变化；以及 $\beta_{1} + \beta_{2} k + \beta_{3} a_{k-1} + \beta_{4} l_{k} + \beta_{5} l_{k} a_{k-1}$ ，它允许在时间点 $k$ 的治疗效应受到最近一次治疗和协变量值的修饰。

$\psi$ 的 $95\%$ 置信区间的界限是使得检验 $\alpha_{1} = 0$ 时 $P$ 值 $> 0.05$ 的 $\psi^{\dagger}$ 值集合的界限。 $\beta_{j}$ 的一个 $95\%$ 联合置信区间是使得 5 自由度的 **得分检验（score test）** 在 $5\%$ 水平上不拒绝的值的集合。一个计算量较小的替代方法是单变量 $95\%$ **Wald 置信区间（Wald confidence interval）** ，即 $\widehat{\beta}_{j} \pm 1.96$ 乘以它的标准误。

为了描述具有多个时间点的结构嵌套均值模型的 g-估计，假设非饱和模型为 $\gamma_{k}\left(\overline{a}_{k-1}, l_{k}, \beta\right) = \beta_{1}$ 。相应的秩保持模型包含 $H_{k}\left(\psi^{\dagger}\right) = Y - \sum_{j=k}^{K} A_{j} \psi^{\dagger}$ ，对于任何值 $\psi^{\dagger}$ ，都可以从观测数据中计算得出。然后，我们将选择比 $\psi$ 的任何实质性合理值都小得多和大得多的值 $\psi_{\text{low}}$ 和 $\psi_{\text{up}}$ ，并在从 $\psi_{\text{low}}$ 到 $\psi_{\text{up}}$ 的网格上（例如 $\psi_{\text{low}}, \psi_{\text{low}} + 0.1, \psi_{\text{low}} + 0.2, \dots, \psi_{\text{up}}$ ）为每个 $\psi^{\dagger}$ 计算（针对每个个体和每个时间点） $H_{k}\left(\psi^{\dagger}\right)$ 的值。

然后，对于每个 $\psi^{\dagger}$ 值，我们将拟合一个（跨时间） **合并的逻辑回归模型（pooled logistic regression model）**

$$
\operatorname{logitPr}\left[ A_{k} = 1 | H_{k}(\psi^{\dagger}), \overline{L}_{k}, \overline{A}_{k-1} \right] = \alpha_{0} + \alpha_{1} H_{k}(\psi^{\dagger}) + \alpha_{2} W_{k}
$$

用于估计时间点 $k$ （ $k = 0, \ldots, K$ ）接受治疗的概率。这里 $W_{k} = w_{k}\left(\overline{L}_{k}, \overline{A}_{k-1}\right)$ 是一个从个体的协变量和治疗数据 $\left(\overline{L}_{k}, \overline{A}_{k-1}\right)$ 计算得到的协变量向量， $\alpha_{2}$ 是一个未知参数的行向量，每个个体贡献 $K+1$ 个观测值。 $\beta$ 的 g-估计是使得 $\alpha_{1}$ 的估计值最接近 0 的网格值 $\psi^{\dagger}$ 。

我们可以通过将估计量 $\widehat{\beta}$ 定义为使得 $\alpha_{1} = 0$ 的得分检验的 p 值等于 1 的 $\psi^{\dagger}$ 值，来消除在整个网格上搜索的需要。也就是说， $\widehat{\beta}$ 是求解以下方程的解的 $\psi^{\dagger}$ 值：

$$
\sum_{i=1, k=0}^{i=N, k=K} \left\{A_{i} - \operatorname{expit}\left(\widehat{\alpha}_{0} + \widehat{\alpha}_{2} W_{i,k}\right) \right\} H_{i,k}\left(\psi^{\dagger}\right) = 0
$$

其中 $\widehat{\alpha}_{0}$ 和 $\widehat{\alpha}_{2}$ 是通过拟合上述将项 $\alpha_{1}$ 设为 0 的逻辑模型得到的。可以使用标准的方程求解器。

估计量 $\widehat{\beta}$ 在以下条件下是一致的：

1.  结构嵌套均值模型是正确的，
2.  对于 $Y$ 的 **序贯可交换性（sequential exchangeability）** 成立，
3.  模型 $\operatorname{logit} \Pr\left[ A_{k} = 1 \mid \overline{L}_{k}, \overline{A}_{k-1} \right] = \alpha_{0} + \alpha_{2} W_{k}$ 是正确的，并且
4.  $H_{k}\left(\psi^{\dagger}\right)$ 以线性形式（即 $H_{k}\left(\psi^{\dagger}\right)$ 而非 $\left\{H_{k}\left(\psi^{\dagger}\right) \right\}^{2}$ 或任何其他非线性函数）进入上述逻辑模型（参见技术要点 14.2）。

上述描述的过程是第 14 章中描述的 g-估计程序在时变治疗情况下的推广。为简单起见，我们考虑了一个具有单一参数 $\beta_{1}$ 的结构嵌套模型，这意味着效应不随时间点 $k$ 或治疗与协变量历史而变化。

现在假设参数 $\beta$ 是一个向量。具体来说，假设我们考虑模型

$$
\gamma_{k}\left(\bar{a}_{k-1}, l_{k}, \beta\right) = \beta_{0} + \beta_{1} k + \beta_{2} a_{k-1} + \beta_{3} l_{k} + \beta_{4} l_{k} a_{k-1}
$$

因此 $\beta$ 是 5 维的， $l_{m}$ 是 1 维的。现在要估计 5 个参数，需要在治疗模型中包含 5 个额外的协变量。例如，我们可以拟合模型

$$
\operatorname{logit} \Pr\left[ A_{k} = 1 \mid H_{k}\left(\psi^{\dagger}\right), \overline{L}_{k}, \overline{A}_{k-1} \right] = \alpha_{0} + H_{k}\left(\psi^{\dagger}\right) \left(\alpha_{1} + \alpha_{2} k + \alpha_{3} A_{k-1} + \alpha_{4} L_{k} + \alpha_{5} L_{k} A_{k-1}\right) + \alpha_{6} W_{k}
$$

协变量的特定选择不影响 $\beta$ 点估计的一致性，但它决定了其置信区间的宽度。早期的 g-估计程序需要在 5 维网格上进行搜索， $\beta$ 的每个分量 $\beta_j$ 对应一维。因此，如果我们每个分量取 20 个网格点，那么在我们的 5 维网格上就会有 $20^5$ 个不同的 $\beta$ 值。然而，当 $\beta$ 的维数大于 2 时，通过网格搜索找到 g-估计 $\widehat{\beta}$ 可能在计算上很困难。在这种情况下，我们可以通过将 g-估计 $\widehat{\beta}$ 定义为使得 $\alpha_{1-5} = (\alpha_1, \dots, \alpha_5)^T = 0$ 的得分检验的 p 值等于 1 的 $\psi^\dagger$ 值，来消除网格搜索的需要。也就是说， $\widehat{\beta}$ 是求解以下 5 维估计方程的解的 $\psi^\dagger$ 值：

$$
\sum_{i = 1, k = 0}^{i = N, k = K} \left\{A_i - \mathrm{expit}\left( \widehat{\alpha}_0 + \widehat{\alpha}_6^T W_{i,k} \right) \right\} H_{i,k} \left( \psi^\dagger \right) \left( 1, k, A_{i,k-1}, L_{i,k}, L_{i,k} A_{i,k-1} \right)^T = 0
$$

其中 $\widehat{\alpha}_0$ 和 $\widehat{\alpha}_6$ 是通过拟合上述将 $\alpha_{1-5}$ 设为零的逻辑模型得到的。可以使用标准的方程求解器。事实上，当结构嵌套均值模型关于 $\beta$ 是线性的时（如本节讨论的所有例子），最后一个方程的解 $\widehat{\beta}$ 以封闭形式存在。参见技术要点 21.8，该要点还描述了该估计量的一个 **多重稳健（multiply robust）** 形式。

给定一个用于结构嵌套均值模型参数的一致 g-估计量 $\widehat{\beta}$ ，最后一步是估计在感兴趣策略 $g$ 下的 **反事实均值（counterfactual mean）** $\mathrm{E}[Y^g]$ 。如前所述， $\mathrm{E}[Y^{\overline{0}}]$ 可以由样本均值 $\widehat{\mathrm{E}}[H_0(\widehat{\beta})]$ 一致地估计。如果不存在过去协变量历史的效应修饰，即 $\gamma_k(\overline{a}_{k-1}, \overline{l}_k, \beta) = \gamma_k(\overline{a}_{k-1}, \beta)$ ，那么在静态策略 $a$ 下的 $\mathrm{E}[Y^{\overline{a}}]$ 估计为：

$$
\widehat{\mathbf{E}} \left[ Y^{\overline{a}} \right] = \widehat{\mathbf{E}} \left[ Y^{\overline{0}_K} \right] + \sum_{k=0}^K a_k \gamma_k \left( \overline{a}_{k-1}, \widetilde{\beta} \right)
$$

另一方面，如果结构嵌套均值模型依赖于 $L_k$ ，或者我们想要估计在动态策略 $g$ 下的 $\mathrm{E}[Y^g]$ ，那么我们需要使用技术要点 21.9 中描述的算法来模拟 $L_k$ 。

> #### 技术要点 21.8
>
> **线性结构嵌套均值模型的封闭形式估计量。**
> 当我们讨论的所有例子中，
>
> $$
> \gamma_k \left( \overline{A}_{k-1}, \overline{L}_k, \beta \right) = \beta^T R_k
> $$
>
> 是关于 $\beta$ 线性的，其中 $R_k = r_k(\bar{L}_k, \bar{A}_{k-1})$ 是一个已知函数向量，那么，给定模型
>
> $$
> \mathrm{logit} \, \mathrm{Pr} \left[ A_k = 1 \mid \overline{L}_k, \overline{A}_{k-1} \right] = \alpha^T W_k
> $$
>
> 存在一个 $\widehat{\beta}$ 的显式封闭形式表达式：
>
> $$
> \widehat{\beta} = \left\{\sum_{i=1, k=0}^{i=N, k=K} A_{i,k} X_{i,k}(\widehat{\alpha}) Q_{i,k} S_{i,k}^T \right\}^{-1} \left\{\sum_{i=1, k=0}^{i=N, k=K} Y_i X_{i,k}(\widehat{\alpha}) Q_{i,k} \right\}
> $$
>
> 其中 $X_{i,k}(\widehat{\alpha}) = \left[ A_{i,k} - \mathrm{expit}(\widehat{\alpha}^T W_{i,k}) \right]$ ， $S_{i,k} = \sum_{i=1, j=k}^{i=N, j=K} R_{i,j}$ ，而维度为 $\beta$ 的函数 $Q_{i,k} = q_k(\bar{L}_{i,k}, \bar{A}_{i,k-1})$ 的选择影响效率但不影响一致性。关于 $Q_k$ 的最优选择，请参见 Robins (1994)。
>
> 事实上，当 $\gamma_k(\overline{a}_{k-1}, \overline{l}_k, \beta)$ 关于 $\beta$ 是线性时，我们可以通过为
>
> $$
> \mathrm{E} \left[ H_k(\beta) \mid \bar{L}_k, \bar{A}_{k-1} \right] = \mathrm{E} \left[ Y^{\overline{A}_{k-1}, \underline{0}_k} \mid \bar{L}_k, \bar{A}_{k-1} \right]
> $$
>
> 指定一个工作模型 $\varsigma^T D_k = \varsigma^T d_k(\bar{L}_k, \bar{A}_{k-1})$ 并定义：
>
> $$
> \begin{pmatrix} \widetilde{\beta} \\ \widetilde{\varsigma} \end{pmatrix} = \left\{\sum_{i=1, k=0}^{i=N, k=K} \begin{pmatrix} A_{i,k} X_{i,k}(\widehat{\alpha}) Q_{i,k} \\ D_{i,k} \end{pmatrix} (S_{i,k}^T, D_{i,k}^T) \right\}^{-1} \left\{\sum_{i=1, k=0}^{i=N, k=K} Y_i \begin{pmatrix} X_{i,k}(\widehat{\alpha}) Q_{i,k} \\ D_{i,k} \end{pmatrix} \right\}
> $$
>
> 来获得一个 $2^{K+1}$ 重稳健的封闭形式估计量 $\widetilde{\beta}$ 。
>
> 具体来说，如果对于每个 $k$ ，要么用于 $\mathrm{E}[Y^{\overline{A}_{k-1}, \underline{0}_k} \mid \bar{L}_k, \bar{A}_{k-1}]$ 的模型 $\varsigma^T D_k$ 是正确的，要么用于 $\mathrm{logit} \, \mathrm{Pr}[A_k = 1 \mid \overline{L}_k, \overline{A}_{k-1}]$ 的模型是正确的，那么 $\widetilde{\beta}$ 将是 $\psi$ 的一个一致且渐近正态的估计量。

> #### 技术要点 21.9

结构嵌套均值模型 G 估计后 $\operatorname{E}[Y^g]$ 的估计

假设可识别性条件成立，已获得 **结构嵌套均值模型（structural nested mean model）** $\gamma_k(\overline{a}_{k-1}, \overline{l}_k, \beta)$ 的 **双稳健 G 估计量（doubly robust g-estimate）** $\widetilde{\beta}$ ，且希望估计动态策略 $g$ 下的 $\operatorname{E}[Y^g]$ 。为此，可采用以下蒙特卡洛算法步骤：

- **1.** 通过 $N$ 名研究对象的 $H_0(\widetilde{\beta})$ 样本均值，估计始终未接受治疗时的 **均值响应（mean response）** $\operatorname{E}[Y^{\overline{0}_K}]$ 。将该估计量记为 $\widehat{\operatorname{E}}[Y^{\overline{0}_K}]$ 。

- **2.** 对跨个体和时间的合并数据拟合 $f(l_k \mid \bar{a}_{k-1}, \bar{l}_{k-1})$ 的 **参数模型（parametric model）** ，并将该模型下 $f(l_k \mid \bar{a}_{k-1}, \bar{l}_{k-1})$ 的估计量记为 $\widehat{f}(l_k \mid \bar{a}_{k-1}, \bar{l}_{k-1})$ 。

- **3.** 对 $v = 1, \dots, V$ 执行：
- **(a)** 从 $\widehat{f}(l_0)$ 中抽取 $l_{v,0}$ 。
- **(b)** 递归地对 $k = 1, \dots, K$ ，从 $\widehat{f}(l_k \mid \bar{a}_{v,k-1}, \bar{l}_{v,k-1})$ 中抽取 $l_{v,k}$ ，其中 $\bar{a}_{v,k-1} = \overline{g}_{k-1}(\bar{l}_{v,k-1})$ ，即与策略 $g$ 对应的治疗史。
- **(c)** 令
  $$
  \widehat{\Delta}_{g,v} = \sum_{j=0}^{j=K} a_{v,j} \, \gamma_j(\overline{a}_{v,j-1}, \overline{l}_{v,j}, \widetilde{\beta})
  $$
  为 $Y^g - Y^{\overline{0}_K}$ 的第 $v$ 个蒙特卡洛估计量，其中 $a_{v,j} = g_j(\bar{l}_{v,j-1})$ 。

4. 令 $\widehat {\mathrm {E}} \left[ Y^{g} \right] = \widehat {\mathrm {E}} \left[ Y^{\overline {{0}}_{K}} \right] + \sum_{v = 1}^{V} \widehat {\Delta}_{g , v} / V$ 作为 $\widehat {\mathrm {E}} \left[ Y^{g} \right]$ 的估计量。

若 $f \left( l_{k} | \bar {a}_{k - 1} , \bar {l}_{k - 1} \right)$ 的模型、结构嵌套均值模型 $\gamma_{k} \left( \overline {{a}}_{k - 1} , \overline {{l}}_{k} , \beta \right)$ 以及 **治疗模型（treatment model）** $\operatorname* {P r} \left[ A_{k} = 1 | \overline {{L}}_{k} , \overline {{A}}_{k - 1} \right]$ 或 **结局模型（outcome model）** $\mathrm {E} \left[ Y^{\overline {{A}}_{k - 1} , \underline {{0}}_{k}} | \bar {L}_{k} , \bar {A}_{k - 1} \right]$ 中至少有一个被正确设定，则 ${\widehat {\mathrm {E}}} \left[ Y^{g} \right]$ 是 $\operatorname {E} \left[ Y^{g} \right]$ 的 **一致估计量（consistent estimator）** 。可使用 **非参数自助法（nonparametric bootstrap）** 获取置信区间。

注意，若估计量 $\widetilde {\beta}$ 对 $\beta = 0$ 一致，则 $\gamma_{k} \left( \overline {{a}}_{k - 1} , \overline {{l}}_{k} , \widetilde {\beta} \right)$ 将收敛至 0。因此，即使 $f \left( l_{k} | \bar {a}_{k - 1} , \bar {l}_{k - 1} \right)$ 的模型设定错误， $\widehat {\Delta}_{g , v}$ 也将收敛至零，且 $\widehat {\mathrm {E}} \left[ Y^{g} \right]$ 收敛至 $\widehat {\mathrm {E}} \left[ Y^{\overline {{0}}_{K}} \right]$ 。也就是说，若可识别性条件成立，且对于每个 $k$ ，我们或已知（如在 **序贯随机化实验（sequentially randomized experiment）** 中） $\operatorname* {P r} \left[ A_{k} = 1 | \overline {{L}}_{k} , \overline {{A}}_{k - 1} \right]$ ，或拥有 $\operatorname* {P r} \left[ A_{k} = 1 | \overline {{L}}_{k} , \overline {{A}}_{k - 1} \right]$ 或 $\mathrm {E} \left[ Y^{\overline {{A}}_{k - 1} , \underline {{0}}_{k}} | \bar {L}_{k} , \bar {A}_{k - 1} \right]$ 的正确模型，则结构嵌套均值模型能保持 **零假设（null）** 。

---

### 21.5 删失作为时变治疗

建议重新阅读第 12.6 节以复习删失相关内容。

本章通篇使用的例子中不存在删失：表 21.1 中所有个体的结局均已知。然而在实践中，我们常会遇到部分个体 **失访（lost to follow-up）** 因而其结局值未知或（右）删失的情况。我们在本书第二部分讨论了删失及其处理方法。在第 8 章中，我们指出即使在零假设下，删失也可能引入 **选择偏倚（selection bias）** 。在第 12 章中，我们讨论了通常关注的是研究人群中无人被删失时的因果效应。

然而，在第二部分中，我们仅考虑了删失的一个极大简化版本，未指定个体在随访期间的删失时间。即，我们将删失 $C$ 视为一个 **时固定变量（time-fixed variable）** 。更现实的视角是将删失视为 **时变变量（time-varying variable）** $C_{1} , C_{2} , \dots , C_{K + 1}$ 。

当 $C$ 是治疗 $A$ 与结局 $Y$ 之间路径上的一个 **碰撞变量（collider）** ，或是此类碰撞变量的后代时，以未删失（ $\boldsymbol {C} = 0$ ）为条件会在零假设下引入选择偏倚。

使用上标 $\bar {c} = \bar {0}$ 明确表达了许多人提及治疗 $\bar {A}$ 的因果效应时心中所想的因果对比，即使他们选择不使用上标 $\bar {c} = \bar {0}$ 。

> **请记住** ：当 $\mathrm {P r} \left( C_{k} = 0 | \bar {A}_{k - 1} , C_{k - 1} = 0 , \bar {L}_{k} \right)$ 的模型被正确设定时，估计的 **逆概率权重（IP weights）** $S W^{\bar {C}}$ 的均值为 1。

其中 $C_{m}$ 是一个指示变量：若个体在时间 $m$ 仍未被删失则取值为 0，否则取值为 1。删失是一种 **单调型（monotonic）** 缺失数据，即若个体的 $C_{m} = 0$ ，则所有之前的删失指示变量也均为零（ $C_{1} = 0 , C_{2} = 0 , \dots , C_{m - 1} = 0$ ）。此外，根据定义，研究中所有个体的 $C_{0} = 0$ ；否则他们将不会被纳入研究。

若个体在时间 $m$ 被删失（即 $C_{m} = 1$ ），则时间 $m$ 之后测量的治疗、混杂因素和结局均无法观测。因此，分析必然局限于未删失的个体-时间点，即那些 $C_{m} = 0$ 的观测。例如，第 21.1 节中 **反事实均值结局（counterfactual mean outcome）** $\mathrm {E} \left[ Y^{a} \right]$ 的 **G 公式（g-formula）** 需改写为：

$$
\sum_ {\bar {l}} \operatorname {E} \left[ Y | \bar {C} = \bar {0}, \bar {A} = \bar {a}, \bar {L} = \bar {l} \right] \prod_ {k = 0}^{K} f \left(l_{k} | c_{k} = 0, \bar {a}_{k - 1}, \bar {l}_{k - 1}\right),
$$

其中所有项均以未删失为条件。

假设可识别性条件成立，且在所有时间点 $m$ 上将治疗 $A_{m}$ 替换为 $(A_{m} , C_{m + 1})$ 。则不难证明，上述表达式对应于 **联合治疗（joint treatment）** $(\bar {a} , \bar {c} = 0)$ 下反事实均值结局 $\mathrm {~ E ~} [ Y^{\bar {a} , \bar {c} = 0} ]$ 的 G 公式，即若所有个体均接受治疗策略 $a$ 且无人失访时将会观测到的均值结局。

当联合治疗 $(\bar {A} , \bar {C})$ 的可识别性条件成立时，反事实均值 $\mathrm {E} \left[ Y^{\bar {a} , \bar {c} = \bar {0}} \right]$ 也可通过 **结构均值模型（structural mean model）** 的 **逆概率加权（IP weighting）** 进行估计。为估计该均值，我们可对由 **非稳定化逆概率权重（nonstabilized IP weights）** $W^{A} \times W^{C}$ 创建的 **伪总体（pseudo-population）** 拟合，例如，结局回归模型：

$$
\mathrm {E} \left[ Y | \bar {A}, \bar {C} = \bar {0} \right] = \theta_ {0} + \theta_ {1} c u m (\bar {A})
$$

其中

$$
W^{\bar{C}} = \prod*{k=1}^{K+1} \frac{1}{\Pr\left(C*{k} = 0 \mid C*{k-1} = 0, \bar{A}*{k-1}, \bar{L}\_{k-1}\right)}.
$$

我们通过拟合 $\Pr\left(C_{k} = 0 \mid C_{k-1} = 0, \bar{A}_{k-1}, \bar{L}_{k-1}\right)$ 的 **逻辑回归模型（logistic regression model）** 来估计权重的分母。技术要点 21.10 展示了将其推广至具有 **失效时间结局（failure time outcome）** 的 **生存分析（survival analysis）** 的情形。

在由非稳定化逆概率权重创建的伪总体中，被删失的个体被具有相同治疗史和协变量史值的未删失个体的副本所替代。因此，伪总体的大小与删失前（即任何失访发生前）的原始研究人群相同。非稳定化逆概率权重在伪总体中消除了删失。

或者，我们可以使用由 **稳定化逆概率权重（stabilized IP weights）** $SW^{\bar{A}} \times SW^{\bar{C}}$ 创建的伪总体，其中

$$
SW^{\bar{C}} = \prod*{k=1}^{K+1} \frac{\Pr\left(C*{k} = 0 \mid C*{k-1} = 0, \bar{A}*{k-1}\right)}{\Pr\left(C*{k} = 0 \mid C*{k-1} = 0, \bar{A}_{k-1}, \bar{L}_{k-1}\right)}.
$$

我们分别通过两个独立的模型估计逆概率权重的分母和分子： $\Pr\left(C_{k} = 0 \mid C_{k-1} = 0, A_{k-1}, L_{k-1}\right)$ 的模型和 $\Pr\left(C_{k} = 0 \mid C_{k-1} = 0, A_{k-1}\right)$ 的模型。由稳定化逆概率权重创建的伪总体与删失后原始研究人群的大小相同，即比例…

---

#### 技术要点 21.10

**含时变治疗的生存分析。** 第 17 章描述了用于估计 **点干预（point interventions）** 对失效时间结局效应的 **G 方法（g-methods）** 。本章描述了用于估计 **持续策略（sustained strategies）** 对非失效时间结局效应的 G 方法。在实践中，我们常通过将第 17 章描述的方法与本章的方法相结合，使用 G 方法估计持续策略对失效时间结局的效应。

下面我们简要概述两种方法——基于 **G 公式（g-formula）** 和基于 **逆概率加权（IP weighting）** ——来估计治疗策略 $\bar{a}$ 下的反事实风险 $\Pr\left[D_{k+1}^{\bar{a}, \bar{c} = \bar{0}} = 1\right]$ ，假设 **序贯可交换性（sequential exchangeability）** 、 **积极性（positivity）** 和 **一致性（consistency）** 成立。

图 21.4 中的 **因果图（causal diagram）** 描述了具有两个时间点且以随时间变化的指示变量（如第 17 章所示）表示的失效时间结局的设置。从每个指示变量 $D\_{k}$ 出发，应有箭头指向图中所有未来变量，但为了减少杂乱，我们省略了这些箭头。为简单起见，我们还省略了用于删失的随时间变化的指示变量。

风险 $\Pr\left[D_{k+1}^{\bar{a}, \bar{c} = \bar{0}} = 1\right]$ 由 1 减去 $\Pr\left[D_{k+1}^{\bar{a}, \bar{c} = \bar{0}} = 0\right]$ 的 **g-公式（g-formula）** 识别：

$$
\begin{aligned}
&\sum*{\bar{l}*{k}} \Pr\left[D_{k+1} = 0 \mid \bar{A}_{k} = \bar{a}_{k}, \bar{L}_{k} = \bar{l}_{k}, D_{k} = C_{k+1} = 0\right] \times \\
&\quad \prod*{m=0}^{k} f\left(l*{m} \mid \bar{a}_{m-1}, \bar{l}_{m-1}, D*{m} = C*{m} = 0\right) \Pr\left[D_{m} = 0 \mid \bar{A}_{m-1} = \bar{a}_{m-1}, \bar{L}_{m-1} = \bar{l}_{m-1}, D_{m-1} = C_{m} = 0\right].
\end{aligned}
$$

然后，可以通过拟合离散时间风险 $\Pr\left[D_{k+1} = 1 \mid \bar{A}_{k} = \bar{a}_{k}, \bar{L}_{k} = \bar{l}_{k}, D_{k} = C_{k+1} = 0\right]$ 的模型以及随时间变化的混杂因素 $L$ 的条件密度 $f\left(l*{k} \mid \bar{a}*{k-1}, \bar{l}_{k-1}, D_{k} = C\_{k} = 0\right)$ 的模型，得到 **代入法 g-公式估计（plug-in g-formula estimate）** 。

如第 17 章所述，可以使用 **合并逻辑模型（pooled logistic model）** 来近似风险。详细内容和应用请参见 Young 等人 (2011)。Wen 等人 (2021) 描述了 **ICE g-公式估计量（ICE g-formula estimators）** 。

另一种方法是拟合风险 $\Pr\left[D_{k+1} = 1 \mid \bar{A}_{k} = \bar{a}_{k}, \bar{L}_{k} = \bar{l}_{k}, D_{k} = C_{k+1} = 0\right]$ 的合并逻辑模型，其中每个个体在时间 $k$ 会获得随时间变化的 **非稳定化逆概率权重（nonstabilized IP weight）** $W*{k}^{\bar{A}} \times W*{k}^{\bar{C}}$ ，其中

$$
W*{k}^{\bar{A}} = \prod*{m=0}^{k} \frac{1}{f\left(A*{m} \mid \bar{A}*{m-1}, D*{m} = C*{m} = 0, \bar{L}_{m}\right)}, \quad W_{k}^{\bar{C}} = \prod*{m=1}^{k} \frac{1}{\Pr\left(C*{m} = 0 \mid \bar{A}_{m-1}, D_{m-1} = C*{m-1} = 0, \bar{L}*{m-1}\right)},
$$

或其在每个时间点 $k$ 对应的 **稳定化逆概率权重（stabilized IP weight）** 。该模型的参数估计的是 $\Pr\left[D_{k+1}^{\bar{a}, \bar{c} = \bar{0}} = 1 \mid D_{k}^{\bar{a}, \bar{c} = \bar{0}} = 0\right]$ 的 **边际结构合并逻辑模型（marginal structural pooled logistic model）** 的参数 (Robins 1998a)。详细内容和应用请参见 Hernán 等人 (2001)。Wen 等人 (2022) 综述了针对随时间变化治疗变量的生存分析的多重稳健估计量。

![image_138](../../images/image_138.png)

> 图 21.4

伪总体中删失的个体在每个时间点 $k$ 与研究总体中的个体相同。稳定化权重并未消除伪总体中的删失；它们使得删失在每个时间点 $k$ 相对于已测量的协变量历史 $L_k$ 是随机发生的。也就是说，存在选择但不存在选择偏倚。无论使用何种类型的逆概率权重，在伪总体中，不存在从 $L_k$ 和 $A_k$ 指向未来 $C_m$ （其中 $m > k$ ）的箭头。重要的是，在针对联合治疗 $(\bar{A}, \bar{C})$ 的 **可交换性条件（exchangeability conditions）** 下，即使 $L$ 的某些分量受到先前治疗的影响，逆概率加权也能无偏地估计 $(A, C)$ 的联合效应。

最后，当使用 **结构嵌套模型（structural nested models）** 的 **g-估计（g-estimation）** 时，我们首先需要通过逆概率加权来调整由删失引起的选择偏倚。在实践中，这意味着我们首先估计用于删失的非稳定化逆概率权重 $W^{\bar{C}}$ ，以创建一个无人被删失的伪总体，然后将 g-估计应用于该伪总体。

### 21.6 大 g-公式（The big g-formula）

我们将 $(\bar{A}, \bar{L}, Y, \bar{U})$ 称为 **事实变量（factuals）** ，以区别于 **反事实变量（counterfactuals）** 。事实变量是存在于现实世界中的变量。与观测变量相比，一些事实变量（例如 $\bar{U}$ ）通常因未被测量而无法用于数据分析。

这些问题由 Tian 和 Pearl (2002)、Shpitser 和 Pearl (2006) 以及 Huang 和 Valtorta (2006) 的工作完全解决。

本章及前两章优先考虑了依赖于给定已测量协变量 $\bar{L}$ 时的 **序贯可交换性（sequential exchangeability）** 以及通过 g-公式进行识别的方法。原因是，在实践中，很少有复杂的纵向数据的因果分析依赖于其他识别条件和公式。例如，很少有基于前门公式作为识别公式的识别条件的现实应用。然而，无论实质上的合理性和实际应用如何，不同的识别条件及其公式在数学上都与基于所有变量（包括已测量和未测量变量）的序贯可交换性和 g-公式相关联，我们现在对此进行解释。

当序贯可交换性在给定已测量协变量 $\bar{L}$ 下成立时，我们已经讨论了基于已测量的随时间变化协变量 $\bar{L}$ 的 g-公式如何识别随时间变化治疗变量 $\bar{A}$ 对结局 $Y$ 的因果效应。现在假设我们有一个包含观测变量 $(\bar{A}, \bar{L}, Y)$ 和未观测变量 $U$ 的 **因果有向无环图（causal DAG）** ，并且已测量变量 $L$ 不足以实现序贯可交换性。

对于任何因果 DAG，已测量和未测量变量的组合 $\bar{\boldsymbol{X}} = (\bar{\boldsymbol{L}}, \bar{\boldsymbol{U}})$ 确保了（联合）序贯可交换性，因为治疗变量的任何父节点都包含在 $\bar{A}$ 或 $\bar{X}$ 中。因此，如果因果图上的每个变量都被测量并且 **正性（positivity）** 成立，那么基于 $X$ 的 g-公式将在任何治疗策略 $g$ 下识别出反事实均值 $\operatorname{E}[Y^g]$ 。我们将 $\bar{L}$ 替换为 $\bar{X}$ 的 g-公式称为 **大 g-公式（big g-formula）** ，因为它并非仅基于观测数据。

给定一个因果 DAG、治疗变量 $\bar{A}$ 和结局 $Y$ 、治疗策略 $g$ 以及事实变量 $(\bar{\mathcal{A}}, \bar{\mathcal{L}}, \mathcal{Y}, \mathcal{U})$ ，我们可以明确写出 $Y^g$ 分布（密度）的大 g-公式。大 g-公式仅依赖于事实变量 $(\bar{A}, \bar{L}, Y, U)$ 的分布。

大 g-公式是在任何治疗策略下识别反事实密度的正确公式，但大 g-公式在实践中无法使用，因为它包含未测量变量。一个有趣的数学问题是：大 g-公式能否简化为观测数据 $(\bar{A}, \bar{L}, Y)$ 联合分布的函数？如果可以，我们将得到一个不以 g-公式形式表达的新公式，但它 (i) 能重现大 g-公式的结果（因此是一个正确的公式），并且 (ii) 仅用观测变量的分布表示（因此是一个可用于数据分析的公式）。

例如，在称为 **前门准则（front door criterion）** 的识别条件下， $\operatorname{E}[Y^a]$ 的大 g-公式简化为一个仅包含观测变量的公式—— **前门公式（front door formula）** （参见技术点 21.11 中的证明）。因此，在图 7.14 的因果图中蕴含的前门假设下，前门公式是 $\operatorname{E}[Y^a]$ 均值的一个有效公式。

更一般地，我们希望能够回答以下两个问题。首先，我们是否总能确定大 g-公式能否被重写为一个仅依赖于观测变量 $(\bar{A}, \bar{L}, Y)$ 分布的公式，同时不做出除 $(\bar{\mathcal{A}}, \bar{\mathcal{L}}, \mathcal{Y}, \mathcal{U})$ 的联合分布服从因果 DAG 所隐含的 **d-分离（d-separation）** 关系之外的任何假设？其次，当上一个问题的答案是肯定的时，我们能否显式地展示这样的识别公式？这两个问题都已得到肯定的回答。

重要的是，这些纯粹是关于以下性质的数学问题

#### 技术点 21.11

前门公式的大 g-公式证明。在技术点 7.4 中，我们给出了在图 7.14 的因果图下反事实概率 $\operatorname{Pr}[Y^a = y]$ 的前门公式的一个证明。这里我们使用大 g-公式提供另一个证明。这第二个证明依赖于图 7.14 所隐含的条件独立性，但它不需要反事实 $Y^m$ 存在。

在图 7.14 下， $\operatorname{Pr}[Y^a = y]$ 的大 g-公式为

$$
\sum_{m} \sum_{u} \Pr[Y = y \mid M = m, A = a, U = u] \Pr[M = m \mid A = a, U = u] \Pr[U = u].
$$

由于 $U$ 的数据不可得， $\operatorname{Pr}[Y^a = y]$ 能被识别当且仅当大 g-公式仅依赖于观测数据 $(Y, M, A)$ 的分布。我们现在证明这确实是事实，因为在上述假设下，g-公式简化为前门公式。

利用 d-分离，我们可以将大 g-公式重写为

$$
\begin{array}{l}
\sum_{m} \Pr[M = m \mid A = a] \sum_{u} \Pr[Y = y \mid M = m, U = u] \left\{\sum_{a'} \Pr[U = u \mid A = a'] \Pr[A = a']\right\} \\
\text{由} U \perp M \mid A \text{和} A \perp Y \mid M, U \\
= \sum_{m} \Pr[M = m \mid A = a] \sum_{a'} \left\{\sum_{u} \Pr[Y = y \mid M = m, A = a', U = u] \Pr[U = u \mid M = m, A = a']\right\} \Pr[A = a'] \\
\text{由} U \perp M \mid A \text{和} A \perp Y \mid M, U.
\end{array}
$$

我们现在提供前门公式的另一个证明，该证明也不需要反事实 $Y^{m}$ 存在。在确定 $\operatorname{Pr}\left[Y^{a}=y\right]$ 是由大 g-公式给出的 $(Y, M, A, U)$ 分布的函数之后，我们可以应用一个 **耦合论证（coupling argument）** 。

假设所有人都在实质理由上同意一个定义良好的 $Y^{m}$ 不存在。然而，任何关于图 7.14 是 **马尔可夫（Markov）** 的事实数据分布都与一个“与数据一样详细”的底层 **完全保序因果图模型（FFRCISTG）** (Robins and Richardson, 2010) 兼容，该模型根据定义正式包含变量 $Y^{m}$ 。技术点 7.4 中的证明表明，在该模型下，大 g-公式等于前门公式。因此，不可能存在一个关于图 7.14 是马尔可夫的事实分布使得该等式不成立；因为如果它不成立，该事实分布将不与“与数据一样详细”的 FFRCISTG 模型兼容。

技术点 21.12 展示了基于 **单世界干预图（SWIG）** 性质的前门公式的另一种证明。已知 $(\overline{A}, \overline{L}, Y, U)$ 上的分布服从由 DAG 上的 d-分离刻画的一定独立性关系。也就是说，这些问题既不涉及反事实也不涉及因果性。与因果性的唯一联系是声称该 DAG 是一个因果 DAG。如果是这样，那么大 g-公式将具有因果解释。如果不是，那么肯定的答案虽然仍然成立，但将没有因果意义。当然，在观测性分析中，我们永远无法确定我们推测为因果图的图确实是一个因果图。

#### 技术要点 21.12

**利用 SWIG 上处理节点的 d-分离性证明前门公式。**  
此处我们利用 SWIG 尚未讨论的一个重要性质，提供前门公式的另一种证明。

给定一个因果图 $G$ ，令 $G^{\overline{a}}$ 为对应于策略 $\overline{a}$ 的关联 SWIG， $B^{\overline{a}}$ 和 $C^{\overline{a}}$ 为观测到的非处理节点 $\left(Y^{\overline{a}}, \overline{L}^{\overline{a}}\right)$ 的两个不相交子集。我们仅假设处理反事实是良定义的。SWIG $G^{\overline{a}}$ 满足以下性质（Shpitser 等，2022）：如果固定节点 $a_{m}$ 在给定 $C^{\overline{a}}$ 的条件下与 $B^{\overline{a}}$ 是 d-分离的，那么 $\mathrm{Pr}\left(B^{\overline{a}}=b \mid C^{\overline{a}}=c\right)$ 不依赖于 $a_{m}$ 。该性质与之前讨论的“任何包含处理变量 $a_{m}$ 作为非端点的路径都被阻断”这一事实并不冲突。

为了阐明该新性质的含义，考虑图 7.14 中前门图所隐含的 SWIG $G^{a}$ 。在 SWIG $G^{a}$ 上，定义 $B^{a}=Y^{a}$ 和 $C^{a}=\left(M^{a^{\prime}}, A\right)$ 。那么，在给定 $C^{a}$ 的条件下， $a$ 与 $B^{a}$ 是 d-分离的，因为从 $a$ 到 $Y^{a}$ 的唯一路径经过了 $C^{a}$ 中的非碰撞点 $M^{a^{\prime}}$ 。因此，根据我们的性质：

$$
\operatorname{E}\left[Y^{a} \mid M^{a}, A\right] = \operatorname{E}\left[Y^{a^{\prime}} \mid M^{a^{\prime}}, A\right]
$$

对于任意 $a$ 和 $a^{\prime}$ 成立。注意该性质并非跨世界（cross-world）的；相反，它规定了不同单世界反事实分布之间的关系。

我们现在利用这一 SWIG 性质，在良定义的反事实 $Y^{m}$ 不存在的情况下证明前门公式。我们继续假设 $(Y^{a}, M^{a}, A)$ 根据 SWIG $G^{a}$ 进行因子分解，而 $(Y^{a^{\prime}}, M^{a^{\prime}}, A)$ 根据 SWIG $G^{a^{\prime}}$ 进行因子分解。我们按照技术要点 7.4 中的证明进行，直到需要证明以下等式为止：

$$
\operatorname{E}\left[Y^{a} \mid M^{a}\right] = \sum_{a^{\prime}} \operatorname{E}\left[Y \mid M, A = a^{\prime}\right] \Pr\left(A = a^{\prime}\right).
$$

现在我们有：

$$
\operatorname{E}\left[Y^{a} \mid M^{a}\right] = \sum_{a^{\prime}} \operatorname{E}\left[Y^{a} \mid M^{a}, A = a^{\prime}\right] \operatorname{Pr}\left(A = a^{\prime} \mid M^{a}\right)
$$

$$
= \sum_{a^{\prime}} \operatorname{E}\left[Y^{a} \mid M^{a}, A = a^{\prime}\right] \operatorname{Pr}\left(A = a^{\prime}\right)
$$

这是由于 $M(a)$ 与 $A$ 是 d-分离的。我们新的 SWIG 性质意味着：

$$
\operatorname{E}\left[Y^{a} \mid M^{a}, A = a^{\prime}\right] = \operatorname{E}\left[Y^{a^{\prime}} \mid M^{a^{\prime}}, A = a^{\prime}\right] = \operatorname{E}\left[Y \mid M, A = a^{\prime}\right]
$$

其中最后一个等式由一致性（consistency）得到。因此，

$$
\operatorname{E}\left[Y^{a} \mid M^{a}\right] = \sum_{a^{\prime}} \operatorname{E}\left[Y \mid M, A = a^{\prime}\right] \operatorname{Pr}\left(A = a^{\prime}\right)
$$

正如所要求的。有趣的是，由此可得，尽管对于所有 $a, a^{\prime}$ 都有 $\operatorname{E}\left[Y^{a} \mid M^{a}\right] = \operatorname{E}\left[Y^{a^{\prime}} \mid M^{a^{\prime}}\right]$ ，但 $\operatorname{E}\left[Y^{a} \mid M^{a}\right] \neq \operatorname{E}\left[Y \mid M\right]$ ，因为

$$
\operatorname{E}\left[Y \mid M\right] = \sum_{a^{\prime}} \operatorname{E}\left[Y \mid M, A = a^{\prime}\right] \operatorname{Pr}\left(A = a^{\prime} \mid M\right)
$$

并且，与反事实 $M^{a}$ 不同，观测到的事实变量 $M = M^{A}$ 并不独立于 $A$ 。

#### 技术要点 21.13

**一般结构嵌套均值模型的形式化定义。**  
Robins (2004) 指出，在 **结构嵌套均值模型（structural nested mean model, SNMM）** 中，将 $\bar{0}$ 作为最后一次处理脉冲后所遵循的策略并无特殊之处。我们可以相对于任意策略 $g$ 定义脉冲函数如下。给定 $g = (g_0, g_1, \ldots, g_K)$ ，一个加性 SNMM 是对如下因果效应的模型：在时刻 $t$ 施加一个处理脉冲 $a_t$ ，然后从 $t+1$ 时刻起遵循策略 $g$ ，与从 $t$ 时刻起直接遵循策略 $g$ 相比，对 $Y$ （在给定截至时刻 $t$ 的处理和协变量历史条件下）的因果效应。也就是说，一个加性 SNMM 对以下反事实对比进行建模：

$$
\gamma_t^g \left( \bar{a}_t, \bar{l}_t \right) = \mathrm{E} \left[ Y^{\bar{a}_{t-1}, a_t, \underline{g}_{t+1}} - Y^{\bar{a}_{t-1}, g_t, \underline{g}_{t+1}} \mid \bar{A}_{t-1} = \bar{a}_{t-1}, A_t = a_t, \bar{L}_t = \bar{l}_t \right]
$$

对于 $t = 0, \ldots, K$ ，其中 $\bar{a} = (a_0, a_1, \ldots, a_K)$ ， $\underline{g}_{t+1} = (g_{t+1}, \ldots, g_K)$ 。当我们想强调 $a_t$ 和 $g_t$ 的独特作用时，我们将 $\gamma_t^g \left( \bar{a}_t, \bar{l}_t \right)$ 写作 $\gamma_t^g \left( \bar{a}_{t-1}, a_t, \bar{l}_t \right)$ ，并将 $Y^{\bar{a}_{t-1}, \underline{g}_t}$ 写作 $Y^{\bar{a}_{t-1}, g_t, \underline{g}_{t+1}}$ 。注意，当 $a_t = g_t \left( \bar{a}_{t-1}, \bar{l}_t \right)$ 时， $\gamma_t^g \left( \bar{a}_{t-1}, a_t, \bar{l}_t \right) \equiv 0$ 。如果像正文中那样，我们假设 **序贯可交换性（sequential exchangeability）** ，那么 $A_t = a_t$ 可以从 $\gamma_t^g \left( \bar{a}_t, \bar{l}_t \right)$ 定义中的条件事件中移除。

一个 SNMM 假设 $\gamma_t^g \left( \bar{a}_t, \bar{l}_t \right) := \gamma_t^g \left( \bar{a}_t, \bar{l}_t; \beta \right)$ ，其中 $\gamma_t^g \left( \bar{a}_t, \bar{l}_t; \beta^\dagger \right)$ 是一个已知函数，当有限维参数向量 $\beta^\dagger$ 等于 0 或 $a_t = g_t \left( \bar{a}_{t-1}, \bar{l}_t \right)$ 时，该函数取值为 0。如果我们定义

$$
H_k \left( \gamma^g \right) = Y - \sum_{t=k}^K \gamma_t^g \left( \bar{A}_t, \bar{L}_t \right),
$$

那么仅由一致性（consistency）即可推出（Robins 2004）：对于 $k = 0, \ldots, K$ ，有 $\mathrm{E} \left[ H_k \left( \gamma^g \right) \mid \bar{L}_k, \bar{A}_k \right] = \mathrm{E} \left[ Y^{\bar{A}_{k-1}, \underline{g}_k} \mid \bar{L}_k, \bar{A}_k \right]$ ，并且 $\operatorname{E} \left[ H_0 \left( \gamma^g \right) \right] = \operatorname{E} \left[ Y^g \right]$ 。因此，如果我们能够识别 $\gamma_t^g \left( \bar{a}_t, \bar{l}_t \right)$ ，那么我们就可以识别 $\mathrm{E} \left[ Y^{\bar{A}_{k-1}, \underline{g}_k} \mid \bar{L}_k, \bar{A}_k \right]$ 和 $\operatorname{E} \left[ Y^g \right]$ 。

在 **正性（positivity）** 和序贯可交换性假设下，最后一个等式意味着 $\operatorname{E} \left[ H_k \left( \gamma^g \right) \mid \bar{L}_k, \bar{A}_k \right] = \operatorname{E} \left[ H_k \left( \gamma^g \right) \mid \bar{L}_k, \bar{A}_{k-1} \right]$ ，这进一步意味着 $\gamma_t^g \left( \bar{a}_t, \bar{l}_t \right)$ 是非参数可识别的。Robins (2004) 还定义了 **最优策略结构嵌套模型（optimal regime structural nested model, opt-SNMM）** ，并展示了在正性和序贯可交换性下，如何利用 opt-SNMM 来估计最优治疗策略 $g_{opt} = \arg \max_g \left[ \mathrm{E} \left( Y_g \right) \right]$ 。

然而，序贯可交换性并非唯一可能的识别假设。例如，Zahn 等人 (2022) 证明，在一种随时间变化的平行趋势假设（time-varying parallel trends assumption）下， $\gamma_t^g \left( \bar{a}_t, \bar{l}_t \right)$ 是可识别的，该假设推广了在具有时变处理和协变量的双重差分估计中通常采用的识别假设。

在技术要点 21.9 中，我们将 SNMM 中的 $g$ 取为“永不治疗”策略，即 $g = \bar{0}$ ，并描述了一种在序贯可交换性假设下识别每个策略 $g$ 对应的 $\operatorname{E} \left[ Y^g \right]$ 的算法。当序贯可交换性不成立时，我们可以使用其他假设（例如，随时间变化的平行趋势）来识别用于定义 SNMM 的 $g$ 所对应的 $\operatorname{E} \left[ Y^g \right]$ ，但无法识别任何其他策略 $g'$ 所对应的 $\mathrm{E} \left[ Y^{g'} \right]$ 。要做到后者，我们需要额外的假设。

例如，Shahn 等人 (2022) 证明，如果在假设随时间变化的平行趋势的基础上，进一步假设在给定过去处理和历史测量协变量的条件下，未测量的混杂因素 $U$ 不存在加性效应修饰（additive effect modification），那么对于所有 $g'$ ， $\operatorname{E} \left[ Y^{g'} \right]$ 都是可识别的。这意味着最优策略 $g_{opt} = \mathrm{argmax}_{g'} \mathrm{E} \left[ Y^{g'} \right]$ 是可识别的。Shahn 等人 (2022) 展示了如何利用结构嵌套均值模型来估计 $g_{opt}$ 。

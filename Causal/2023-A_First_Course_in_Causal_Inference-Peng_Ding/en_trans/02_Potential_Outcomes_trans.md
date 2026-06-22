# 潜在结果（Potential Outcomes）

## 2.1 实验主义者对因果推断的看法（Experimentalists' view of causal inference）

Rubin (1975) 和 Holland (1986) 提出了这一格言：

**没有操纵就没有因果（no causation without manipulation）。**

并非所有人都同意这一观点。然而，它对于澄清因果关思考中的模糊性非常有帮助。本书遵循这一观点，并使用**潜在结果框架（potential outcomes framework）**（Neyman, 1923; Rubin, 1974）来定义因果效应。在该框架中，一个实验，或者至少是一个思想实验，包含一个干预、操纵或处理，而我们对其对一个或多个结果的影响感兴趣。

**例 2.1** 如果我们对服用阿司匹林与否对缓解头痛的效果感兴趣，那么干预就是服用阿司匹林。

**例 2.2** 如果我们对参加职业培训项目与否对就业和工资的影响感兴趣，那么干预就是参加职业培训项目。

**例 2.3** 如果我们对小班授课或大班授课对标准化考试成绩的影响感兴趣，那么干预就是小班授课。

**例 2.4** Gerber 等人 (2008) 对不同动员投票信息（get-out-to-vote messages）对投票行为的影响感兴趣。干预就是不同的动员投票信息。

**例 2.5** Pearl (2018) 声称我们可以推断肥胖对寿命的影响。肥胖的一种常用度量是**身体质量指数（Body Mass Index, BMI）**，其定义为体重除以身高的平方，单位为 $kg/m^{2}$ 。因此干预可以是 BMI。

然而，上述干预的模糊程度不同。例 2.1–2.4 中干预的含义相对清晰，但例 2.5 中干预 BMI 的含义则不那么清晰。特别是，我们可以想象 BMI 降低的不同方式：更健康的饮食、更多的体育锻炼、减肥手术等。这些不同版本的干预可能对结果产生截然不同的影响。在本书中，我们将例 2.5 中的干预视为定义不明确，除非有进一步的澄清。

另一个定义不明确的干预是种族。种族歧视是劳动力市场中的一个重要问题，但很难想象一个实验能改变任何实验单元的种族。Bertrand 和 Mullainathan (2004) 给出了一个有趣的实验，部分地回答了这个问题。

**例 2.6** Bertrand 和 Mullainathan (2004) 随机更改简历上的姓名，并比较带有非裔美国人或白人名字的简历的回访率。对于每份简历，干预是表示非裔美国人或白人名字的二元指标，结果是表示回访与否的二元指标。我们在第 1.2.2 节中分析过以下 2×2 列联表：

<table><tr><td></td><td>回访（callback）</td><td>未回访（no callback）</td></tr><tr><td>非裔美国人（African-American）</td><td>157</td><td>2278</td></tr><tr><td>白人（White）</td><td>235</td><td>2200</td></tr></table>

由上可知，我们可以比较非裔美国人名字和白人名字被回访的概率：

$$
\frac {157}{2278 + 157} - \frac {235}{2200 + 235} = 6.45 \% - 9.65 \% = -3.20 \% <   0
$$

其 Fisher 精确检验的 p 值远小于 0.001。

在 Bertrand 和 Mullainathan (2004) 的实验中，处理是感知到的种族，实验者可以操纵它。他们设计了一个实验来回答一个定义明确的因果问题。

## 2.2 潜在结果的形式化符号（Formal notation of potential outcomes）

考虑一个包含 n 个实验单元的研究，索引为 $i = 1, \ldots, n$ 。作为起点，我们关注一个具有两个水平的处理：1 表示处理组，0 表示对照组。对于每个单元 i，感兴趣的结果 Y 有两个版本：

$$
Y _ {i} (1) \text { 和 } Y _ {i} (0),
$$

这些是在假设性干预 1 和 0 下的潜在结果。Neyman (1923) 首次使用了这种符号。它看起来直观，但包含一些隐含假设。Rubin (1980) 对这些隐含假设做了如下澄清。

**假设 2.1（无干扰，no interference）**：单元 i 的潜在结果不依赖于其他单元的处理。这有时被称为**无干扰假设（no-interference assumption）**。

**假设 2.2（一致性，consistency）**：处理没有其他版本。等价地，我们要求处理水平定义明确，或者至少对于感兴趣的结果没有歧义。这有时被称为**一致性假设（consistency assumption）**。

假设 2.1 在传染病或网络实验中可能被违反。例如，如果我的一些朋友接种了流感疫苗，即使我没有接种，我患流感的概率也会降低；如果我的朋友在 Facebook 上看到一则广告，即使我没有看到，我购买该产品的概率也会增加。在现代因果推断文献中，研究存在干扰单元的情况是一个活跃的研究领域。

假设 2.2 可能因具有复杂成分的处理而被违反。例如，在研究吸烟对肺癌的影响时，香烟的类型可能很重要；在研究大学教育对收入的影响时，大学教育的类型和专业可能很重要。

Rubin (1980) 将上述假设 2.1 和 2.2 统称为**稳定单元处理值假设（Stable Unit Treatment Value Assumption, SUTVA）**。

**假设 2.3（SUTVA）**：假设 2.1 和 2.2 同时成立。

在 SUTVA 下，Rubin (2005) 将 $n \times 2$ 的潜在结果矩阵称为**科学表（Science Table）**：

<table><tr><td>i</td><td>$ Y_{i}(1) $</td><td>$ Y_{i}(0) $</td></tr><tr><td>1</td><td>$ Y_{1}(1) $</td><td>$ Y_{1}(0) $</td></tr><tr><td>2</td><td>$ Y_{2}(1) $</td><td>$ Y_{2}(0) $</td></tr><tr><td>$ \vdots $</td><td>$ \vdots $</td><td>$ \vdots $</td></tr><tr><td>n</td><td>$ Y_{n}(1) $</td><td>$ Y_{n}(0) $</td></tr></table>

由于 Neyman 和 Rubin 对统计因果推断的基础性贡献，潜在结果框架有时被称为**Neyman 模型（Neyman model）**、**Neyman-Rubin 模型（Neyman-Rubin model）**或**Rubin 因果模型（Rubin Causal Model）**。

因果效应是科学表的函数。推断个体因果效应

$$
\tau_ {i} = Y _ {i} (1) - Y _ {i} (0)
$$

从根本上说具有挑战性，因为对于每个单元 i，我们只能观察到 $Y_{i}(1)$ 或 $Y_{i}(0)$ 中的一个，即我们只能观察到科学表的一半。作为起点，本书大部分内容关注**平均因果效应（Average Causal Effect, ACE）**：

$$
\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \left\{Y _ {i} (1) - Y _ {i} (0) \right\} = n ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i} (1) - n ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i} (0).
$$

但我们可以很容易地将讨论扩展到许多其他参数（也称为**估计量（estimands）**）。

### 2.2.1 因果效应、子组与 Yule–Simpson 悖论的不存在性（Causal effects, subgroups, and the non-existence of Yule–Simpson Paradox）

如果我们有两个由二元变量 $x_{i}$ 定义的子组，我们可以将子组因果效应定义为

$$
\tau_ {x} = \frac {\sum_ {i = 1} ^ {n} I (x _ {i} = x) \{Y _ {i} (1) - Y _ {i} (0) \}}{\sum_ {i = 1} ^ {n} I (x _ {i} = x)}, \quad (x = 0, 1)
$$

其中 $I(\cdot)$ 是指示函数。一个简单的恒等式是

$$
\tau = \pi_ {1} \tau_ {1} + \pi_ {0} \tau_ {0}
$$

其中 $\pi_{x}=\sum_{i=1}^{n}I(x_{i}=x)/n$ 是 $x_{i}=x\ (x=0,1)$ 的单元比例。因此，如果 $\tau_{1}>0$ 且 $\tau_{0}>0$ ，则必有 $\tau>0$ 。因此 **Yule–Simpson 悖论（Yule–Simpson Paradox）**不可能发生在因果效应上。

### 2.2.2 实验单元的微妙之处（Subtlety of experimental unit）

我以一个与实验单元定义相关的微妙之处来结束本节。简单来说，实验单元可能不同于物理单元。例如，如果我以前没有服用阿司匹林，头痛没有消失，但现在我服用了阿司匹林，头痛消失了，你可能会认为我们可以观察到我在对照组和处理组下的潜在结果。令 i 索引我自己，令 Y = 1 表示无头痛的指标。那么，上述启发式推理表明 $Y_{i}(0) = 0$ 且 $Y_{i}(1) = 1$ ，因此似乎阿司匹林治好了我的头痛。但这种逻辑是非常错误的，因为它误解了实验单元的定义。在不同时间点，我，同一个物理个体，变成了两个不同的实验单元，索引为“i, 之前”和“i, 之后”。因此，我们有四个潜在结果

$$
Y _ {i, \mathrm{before}} (0) = 0, Y _ {i, \mathrm{before}} (1) = ?, Y _ {i, \mathrm{after}} (0) = ?, Y _ {i, \mathrm{after}} (1) = 1,
$$

其中两个被观察到，两个缺失。个体因果效应

$$
Y _ {i, \mathrm{before}} (1) - Y _ {i, \mathrm{before}} (0) = ? - 0 \mathrm{和} Y _ {i, \mathrm{after}} (1) - Y _ {i, \mathrm{after}} (0) = 1 -?
$$

是未知的。有可能即使我不服用阿司匹林，我的头痛也会消失：

$$
Y _ {i, \mathrm{after}} (0) = 1, Y _ {i, \mathrm{after}} (1) = 1
$$

这意味着零效应；也有可能如果我不服用阿司匹林，头痛不会消失：

$$
Y _ {i, \mathrm{after}} (0) = 0, Y _ {i, \mathrm{after}} (1) = 1
$$

这意味着阿司匹林有正向效应。

如果在前后期间对照潜在结果是稳定的： $Y_{i,\text{before}}(0) = Y_{i,\text{after}}(0) = 0$ ，那么错误的启发式论证可能会得到正确答案。但这个假设相当强，并且从根本上说是不可检验的。

## 2.3 处理分配机制（Treatment assignment mechanism）

令 $Z_{i}$ 为单元 i 的二元处理指标，向量化为 $Z = (Z_{1},\ldots ,Z_{n})$ 。单元 i 的观测结果是潜在结果和处理指标的函数：

$$
Y _ {i} = \left\{ \begin{array}{l l} Y _ {i} (1), & \text { 如果 } Z _ {i} = 1 \\ Y _ {i} (0), & \text { 如果 } Z _ {i} = 0 \end{array} \right. \tag {2.1}
$$

$$
= Z _ {i} Y _ {i} (1) + \left(1 - Z _ {i}\right) Y _ {i} (0) \tag {2.2}
$$

$$
= Y _ {i} (0) + Z _ {i} \{Y _ {i} (1) - Y _ {i} (0) \} \tag {2.3}
$$

$$
= Y _ {i} (0) + Z _ {i} \tau_ {i}. \tag {2.4}
$$

方程 (2.1) 是观测结果的定义。方程 (2.2) 等价于 (2.1)。这是一个平凡的事实，但 Judea Pearl 将其视为潜在结果与观测结果之间的基本桥梁。方程 (2.3) 和 (2.4) 强调了这样一个事实：个体因果效应 $\tau_{i}=Y_{i}(1)-Y_{i}(0)$ 在不同单元之间可能是异质的。

实验只揭示了单元 i 的一个潜在结果，另一个是缺失的：

$$
\begin{array}{l} Y _ {i} ^ {\text { mis }} = \left\{ \begin{array}{l l} Y _ {i} (0), & \text { 如果 } Z _ {i} = 1 \\ Y _ {i} (1), & \text { 如果 } Z _ {i} = 0 \end{array} \right. \\ = Z _ {i} Y _ {i} (0) + (1 - Z _ {i}) Y _ {i} (1). \\ \end{array}
$$

缺失的潜在结果对应于单元 i 相反的处理水平。因此，潜在结果框架也被称为**反事实框架（counterfactual framework）**。这个名称可能会令人困惑，因为在实验之前，两个潜在结果都是可观察的，而在实验之后，其中一个潜在结果被实际观察到。

**处理分配机制（treatment assignment mechanism）**，即 Z 的概率分布，在推断因果效应中起着重要作用。以下简单的数值例子说明了这一点。我们首先生成来自正态分布的潜在结果，平均因果效应接近 -0.5。

$$
\begin{array}{l} > n = 5 0 0 \\ > \mathrm{Y0} = \text { rnorm(n) } \\ > \text { tau } = - 0. 5 + Y 0 \\ > \mathrm{Y} 1 = \mathrm{Y} 0 + \text { tau } \\ \end{array}
$$

一个完美的医生会将处理分配给那些个体因果效应非负的患者。这导致观测结果均值之差为正：

$$
\begin{array}{l} > Z = (\text { tau } > = 0) \\ > \mathrm{Y} = \mathrm{Z} * \mathrm{Y} 1 + (1 - \mathrm{Z}) * \mathrm{Y} 0 \\ \end{array}
$$

> mean(Y[Z==1]) - mean(Y[Z==0])

[1] 2.166509

一个盲目的医生不知道任何关于个体因果效应的信息，并通过抛一枚公平硬币来将处理分配给患者。这导致观测结果均值之差接近真实的平均因果效应：

```txt
> Z = rbinom(n, 1, 0.5)
> Y = Z * Y1 + (1 - Z) * Y0
> mean(Y[Z == 1]) - mean(Y[Z == 0])
[1] -0.552064
```

上述例子是假设性的，因为没有医生能完美地知道个体因果效应。然而，这些例子确实展示了处理分配机制的关键作用。本书将根据处理分配机制来组织主题。

## 2.4 课后作业（Homework Problems）

### 2.1 完美医生（A perfect doctor）

延续第 2.3 节中的第一个完美医生例子，假设潜在结果是从以下分布生成的随机变量：

$$
Y (0) \sim \mathrm{N} (0, 1), \quad \tau = - 0. 5 + Y (0), \quad Y (1) = Y (0) + \tau .
$$

二元处理由处理效应决定，即 $Z = 1(\tau \geq 0)$ ，观测结果由潜在结果和处理决定，即 $Y = ZY(1) + (1 - Z)Y(0)$ 。计算均值之差

$$
E (Y \mid Z = 1) - E (Y \mid Z = 0).
$$

提示：截断正态随机变量的均值等于

$$
E (X \mid a <   X <   b) = \mu - \sigma \frac {\phi \left(\frac {b - \mu}{\sigma}\right) - \phi \left(\frac {a - \mu}{\sigma}\right)}{\Phi \left(\frac {b - \mu}{\sigma}\right) - \Phi \left(\frac {a - \mu}{\sigma}\right)},
$$

其中 $X \sim \mathrm{N}(\mu, \sigma^{2})$ ， $\phi(\cdot)$ 和 $\Phi(\cdot)$ 分别是标准正态随机变量的概率密度函数和累积分布函数。

### 2.2 非线性因果估计量（Nonlinear causal estimands）

对于 n 个单元在处理组和对照组下的潜在结果 $\{(Y_{i}(1), Y_{i}(0)\}_{i=1}^{n}$ ，均值之差等于个体处理效应的均值：

$$
\bar {Y} (1) - \bar {Y} (0) = n ^ {- 1} \sum_ {i = 1} ^ {n} \{Y _ {i} (1) - Y _ {i} (0) \}.
$$

因此，**平均处理效应（Average Treatment Effect, ATE）**是一个线性因果估计量。

其他估计量可能不是线性的。例如，我们可以将**中位数处理效应（median treatment effect）**定义为

$$
\delta_ {1} = \mathrm{median} \{(Y _ {i} (1) \} _ {i = 1} ^ {n} - \mathrm{median} \{(Y _ {i} (0) \} _ {i = 1} ^ {n},
$$

这通常不同于个体处理效应的中位数

$$
\delta_ {2} = \mathrm{median} \{(Y _ {i} (1) - Y _ {i} (0) \} _ {i = 1} ^ {n}.
$$

1. 给出具有 $\delta_1 = \delta_2$ 、 $\delta_1 > \delta_2$ 和 $\delta_1 < \delta_2$ 的数值例子。
2. 哪个估计量更有意义， $\delta_1$ 还是 $\delta_2$ ？为什么？用例子证明你的结论。如果你认为 $\delta_1$ 和 $\delta_2$ 在不同的应用中都有意义，你也可以给出例子来证明这两个估计量。

### 2.3 平均效应与个体效应（Average and individual effects）

给出一个数值例子，其中 $\tau = n^{-1} \sum_{i=1}^{n} \{Y_i(1) - Y_i(0)\} > 0$ ，但 $Y_i(1) > Y_i(0)$ 的单元比例小于 0.5。也就是说，平均因果效应为正，但处理对不到一半的单元有益。

### 2.4 推荐阅读（Recommended reading）

Holland (1986) 是一篇关于统计因果推断的经典综述文章。它推广了潜在结果框架的“Rubin 因果模型”这一名称。在加州大学伯克利分校，出于显而易见的原因，我们称之为“Neyman 模型”。
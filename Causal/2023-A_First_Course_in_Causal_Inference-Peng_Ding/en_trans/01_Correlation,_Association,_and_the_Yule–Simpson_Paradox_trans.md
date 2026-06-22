# 相关性、关联性与尤尔-辛普森悖论（Correlation, Association, and the Yule–Simpson Paradox）

因果关系是人类知识的核心。以下是两位古希腊哲人的名言。

我宁愿发现一条因果律，也不愿成为波斯国王。

——德谟克利特（Democritus）

除非我们理解了事物的原因，否则我们并不真正认识它。

——亚里士多德（Aristotle）

然而，经典统计学的主要部分关注的是**关联性（association）**而非**因果关系（causation）**。本章将回顾一些基本的关联度量，并指出其根本局限性。

## 1.1 统计学的传统观点（Traditional view of statistics）

统计学的传统观点是推断变量之间的**相关性（correlation）**或**关联性（association）**。基于这一观点，统计学中不存在因果推断的角色。基于这一观点的两则著名格言如下：

- "相关不蕴含因果。"
- "你无法用统计学证明因果关系。"

本书持有截然不同的观点：**统计学对于理解因果关系至关重要**。本书的主要焦点是引入因果推断的形式化语言，并发展统计方法以估计随机实验和观察性研究中的因果效应。

## 1.2 一些常用的关联度量（Some commonly-used measures of association）

## 1.2.1 相关性与回归（Correlation and regression）

两个随机变量 $Z$ 和 $Y$ 之间的**皮尔逊相关系数（Pearson correlation coefficient）**为

$$
\rho_ {Z Y} = \frac {\operatorname{cov} (Z , Y)}{\sqrt {\operatorname{var} (Z) \operatorname{var} (Y)}},
$$

该系数衡量 $Z$ 和 $Y$ 的线性依赖程度。

$Y$ 对 $Z$ 的**线性回归（linear regression）**模型为

$$
Y = \alpha + \beta Z + \varepsilon , \tag {1.1}
$$

其中 $E(\varepsilon) = 0$ 且 $E(\varepsilon Z) = 0$ 。我们可以证明回归系数 $\beta$ 等于

$$
\beta = \frac {\operatorname{cov} (Z , Y)}{\operatorname{var} (Z)} = \rho_ {Z Y} \sqrt {\frac {\operatorname{var} (Y)}{\operatorname{var} (Z)}}.
$$

因此 $\beta$ 和 $\rho_{ZY}$ 始终同号。

我们还可以定义 $Y$ 对 $Z$ 和 $X$ 的**多元回归（multiple regression）**：

$$
Y = \alpha + \beta Z + \gamma X + \varepsilon , \tag {1.2}
$$

其中 $E(\varepsilon)=0$ ， $E(\varepsilon Z)=0$ 且 $E(\varepsilon X)=0$ 。我们通常将 $\beta$ 解释为在保持 $X$ 不变、以 $X$ 为条件或控制 $X$ 的情况下，$Z$ 对 $Y$ 的"效应"。附录 A2 回顾了线性回归的基础知识。

更有趣的是，上述两个回归模型（1.1）和（1.2）中的 $\beta$ 可能不同；它们甚至可能符号相反。以下 R 代码重新分析了 Hainmueller (2012) 使用的 LaLonde 观察性数据。感兴趣的主要问题是职业培训项目对收入的"因果效应"。控制所有协变量的回归给出处理变量的系数为 1067.5461，而不控制任何协变量的回归给出处理变量的系数为 -8506.4954。

```txt
> dat <- read.table("cps1re74.csv", header = TRUE)
> dat$u74 <- as.numeric(dat$re74==0)
> dat$u75 <- as.numeric(dat$re75==0)
>
> ## linear regression on the outcome
> lmoutcome = lm(re78 ~ ., data = dat)
> summary(lmoutcome)$coef[2, 1:2]
Estimate Std. Error
1067.5461 554.0595
>
> lmoutcome = lm(re78 ~ treat, data = dat)
> summary(lmoutcome)$coef[2, 1:2]
Estimate Std. Error
-8506.4954 712.7664
```

## 1.2.2 列联表（Contingency tables）

我们可以通过一个 $2 \times 2$ 列联表来表示两个二值变量 $Z$ 和 $Y$ 的联合分布。设 $p_{zy} = \Pr(Z = z, Y = y)$ ，我们可以将联合分布总结为下表：

$$
\begin{array}{c c c} & Y = 1 & Y = 0 \\ \hline Z = 1 & p _ {1 1} & p _ {1 0} \\ Z = 0 & p _ {0 1} & p _ {0 0} \end{array}
$$

将 $Z$ 视为处理或暴露，将 $Y$ 视为结局，我们可以定义**风险差（risk difference）**为

$$
\begin{array}{l} \mathrm{RD} = \operatorname * {p r} (Y = 1 \mid Z = 1) - \operatorname * {p r} (Y = 1 \mid Z = 0) \\ = \frac {p _ {1 1}}{p _ {1 1} + p _ {1 0}} - \frac {p _ {0 1}}{p _ {0 1} + p _ {0 0}}, \\ \end{array}
$$

**风险比（risk ratio）**为

$$
\begin{array}{l} \mathrm{RR} = \frac {\operatorname* {p r} (Y = 1 \mid Z = 1)}{\operatorname* {p r} (Y = 1 \mid Z = 0)} \\ = \left. \frac {p _ {1 1}}{p _ {1 1} + p _ {1 0}} \right/ \frac {p _ {0 1}}{p _ {0 1} + p _ {0 0}}, \\ \end{array}
$$

以及**比值比（odds ratio）** $^{1}$ 为

> - $^{1}$ 在概率论中，事件的**比值（odds）**定义为事件发生的概率与事件不发生的概率之比。

$$
\begin{array}{l} \text { OR } = \frac {\operatorname{pr} (Y = 1 \mid Z = 1) / \operatorname{pr} (Y = 0 \mid Z = 1)}{\operatorname{pr} (Y = 1 \mid Z = 0) / \operatorname{pr} (Y = 0 \mid Z = 0)} \\ = \frac {\frac {p _ {1 1}}{p _ {1 1} + p _ {1 0}} / \frac {p _ {1 0}}{p _ {1 1} + p _ {1 0}}}{\frac {p _ {0 1}}{p _ {0 1} + p _ {0 0}} / \frac {p _ {0 0}}{p _ {0 1} + p _ {0 0}}} \\ = \frac {p _ {1 1} p _ {0 0}}{p _ {1 0} p _ {0 1}}. \\ \end{array}
$$

风险差、风险比和比值比这些术语源于**流行病学（epidemiology）**。由于流行病学中的结局通常是疾病，因此使用"风险"这一名称来表示患病的概率是很自然的。

对于这些度量，我们有如下简单事实。

**命题 1.1** (1) 以下陈述都是等价的 $^{2}$ ： $Z \perp Y$ ，RD = 0，RR = 1 且 OR = 1。(2) 如果所有 $p_{zy}$ 均为正，则 RD > 0 等价于 RR > 1，也等价于 OR > 1。(3) 如果 $\operatorname{pr}(Y = 1 \mid Z = 1)$ 和 $\operatorname{pr}(Y = 1 \mid Z = 0)$ 很小，则 OR ≈ RR。

> - $^{2}$ 本书使用符号 $\perp\perp$ 表示随机变量的独立性或条件独立性。该符号源于 Dawid (1979)。

我将陈述 (1) 和 (2) 的证明留作作业题。陈述 (3) 是非正式的。近似成立是因为对于 $p \approx 0$ 的罕见疾病，比值 $p/(1-p)$ 接近于概率 p：通过泰勒展开 $p/(1-p) = p + p^{2} + \cdots \approx p$ 。在流行病学中，如果结局代表罕见疾病的发生，那么假设 $\Pr(Y = 1 \mid X = 1)$ 和 $\Pr(Y = 1 \mid X = 0)$ 很小是合理的。

如果将概率替换为给定另一个变量 $X$ 的条件概率，即 $\Pr(Y=1\mid Z=1,X=x)$ 和 $\Pr(Y=1\mid Z=0,X=x)$ ，我们还可以定义 RD、RR 和 OR 的条件版本。

使用频数 $n_{zy} = \#\{i : Z_i = z, Y_i = y\}$ ，我们可以将观测数据总结为以下 $2 \times 2$ 表：

$$
\begin{array}{c c c} & Y = 1 & Y = 0 \\ \hline Z = 1 & n _ {1 1} & n _ {1 0} \\ Z = 0 & n _ {0 1} & n _ {0 0} \end{array}
$$

我们可以通过用样本比例替换真实概率来估计 RD、RR 和 OR。在 R 中，函数 `fisher.test` 执行精确检验，`chisq.test` 基于观测数据的 $2 \times 2$ 表对 $Z \perp Y$ 执行渐近检验。

**例 1.1** Bertrand 和 Mullainathan (2004) 对简历进行了一项随机实验，以研究感知种族对面试回电的影响。他们在波士顿和芝加哥报纸的招聘广告中，为虚构的简历随机分配了非裔美国人或白人的名字。以下 $2 \times 2$ 表总结了感知种族和回电情况：

```txt
> resume = read.csv("resume.csv")
> Alltable = table(resume$race, resume$call)
> Alltable
```

```txt
0 1
black 2278 157
white 2200 235
```

两行的总计计数相同，因此很明显白人名字获得了更多回电。下面的 Fisher 精确检验表明，这一差异在统计上显著。

```txt
> fisher. test (Alltable)
```

Fisher's Exact Test for Count Data

```txt
data: Alltable
p-value = 4.759e-05
alternative hypothesis: true odds ratio is not equal to 1
95 percent confidence interval:
1.249828 1.925573
sample estimates:
```

odds ratio

1.549732

## 1.3 尤尔-辛普森悖论的一个例子（An example of the Yule–Simpson Paradox）

## 1.3.1 数据（Data）

经典的**肾结石（Kidney stone）**例子来自 Charig 等人 (1986)，其中 $Z$ 是处理，1 表示开放性外科手术，0 表示小穿刺术；$Y$ 是结局，1 表示成功，0 表示失败。处理与结局数据可以总结为以下 $2 \times 2$ 表：

$$
\begin{array}{c c c} & Y = 1 & Y = 0 \\ \hline Z = 1 & 2 7 3 & 7 7 \\ Z = 0 & 2 8 9 & 6 1 \end{array}
$$

估计的 RD 为

$$
\widehat {\mathrm{RD}} = \frac {2 7 3}{2 7 3 + 7 7} - \frac {2 8 9}{2 8 9 + 6 1} = 78 \% - 83 \% = -5 \% <  0.
$$

处理 0 似乎更好，也就是说，与开放性外科手术相比，小穿刺术的成功率更高。

然而，这些数据并非来自**随机对照试验（randomized controlled trial, RCT）** $^{3}$ 。接受处理 1 的患者可能与接受处理 0 的患者非常不同。本研究中的一个"潜伏变量"是病例的严重程度：有些患者结石较小，而有些患者结石较大。我们可以根据结石大小对数据进行分层。

> - $^{3}$ 在 RCT 中，患者被随机分配到处理组。本书第二部分将重点讨论 RCT。

对于结石较小的患者，处理与结局数据可以总结为以下 $2 \times 2$ 表：

$$
\begin{array}{c c c} & Y = 1 & Y = 0 \\ \hline Z = 1 & 8 1 & 6 \\ Z = 0 & 2 3 4 & 3 6 \end{array}
$$

对于结石较大的患者，处理与结局数据可以总结为以下 $2 \times 2$ 表：

$$
\begin{array}{c c c} & Y = 1 & Y = 0 \\ \hline Z = 1 & 1 9 2 & 7 1 \\ Z = 0 & 5 5 & 2 5 \end{array}
$$

后两个表必须加起来等于第一个表：

$$
8 1 + 1 9 2 = 2 7 3, \quad 6 + 7 1 = 7 7, \quad 2 3 4 + 5 5 = 2 8 9, \quad 3 6 + 2 5 = 6 1.
$$

从结石较小患者的表中，估计的 RD 为

$$
\widehat {\mathrm{RD}} _ {\text { 较小 }} = \frac {81}{81 + 6} - \frac {234}{234 + 36} = 93 \% - 87 \% = 6 \% > 0,
$$

表明处理 1 更好。从结石较大患者的表中，估计的 RD 为

$$
\widehat {\mathrm{RD}} _ {\text { 较大 }} = \frac {1 9 2}{1 9 2 + 7 1} - \frac {5 5}{5 5 + 2 5} = 73 \% - 69 \% = 4 \% > 0,
$$

同样表明处理 1 更好。

上述数据分析得出

$$
\widehat {\mathrm{RD}} <   0, \quad \widehat {\mathrm{RD}} _ {\text { 较小 }} > 0, \quad \widehat {\mathrm{RD}} _ {\text { 较大 }} > 0.
$$

非正式地说，对于结石较小和较大的患者，处理 1 都更好，但对于整个人群，处理 1 更差。如果目标是推断处理效应，这个解释相当令人困惑。在统计学中，这被称为**尤尔-辛普森悖论（Yule–Simpson Paradox）**或**辛普森悖论（Simpson's Paradox）**，即边际关联与所有层次上的条件关联符号相反。

## 1.3.2 解释（Explanation）

设 $X$ 为二值指示变量，$X = 1$ 表示结石较小，$X = 0$ 表示结石较大。首先，通过比较结石较小和较大患者中接受处理 1 的概率，来考察 $X$ 与 $Z$ 的关系：

$$
\begin{array}{l} \widehat {\operatorname{pr}} (Z = 1 \mid X = 1) - \widehat {\operatorname{pr}} (Z = 1 \mid X = 0) \\ = \frac {8 1 + 6}{8 1 + 6 + 2 3 4 + 3 6} - \frac {1 9 2 + 7 1}{1 9 2 + 7 1 + 5 5 + 2 5} \\ = 24 \% - 77 \% \\ = -53\% <   0. \\ \end{array}
$$

因此，结石较大的患者倾向于接受处理 1。统计上，$X$ 和 $Z$ 存在负关联。

然后，通过比较结石较小和较大患者的成功概率，来考察 $X$ 与 $Y$ 的关系：在处理 1 下，

$$
\begin{array}{l} \widehat {\operatorname{pr}} (Y = 1 \mid Z = 1, X = 1) - \widehat {\operatorname{pr}} (Y = 1 \mid Z = 1, X = 0) \\ = \frac {8 1}{8 1 + 6} - \frac {1 9 2}{1 9 2 + 7 1} \\ = 93\% - 73 \% \\ = 20 \% > 0; \\ \end{array}
$$

![image_01](images/image_01.png)

**图 1.1**：肾结石例子的示意图。符号表示两个变量的关联方向，同时以指向下游变量的其他变量为条件。

在处理 0 下，

$$
\widehat {\operatorname{pr}} (Y = 1 \mid Z = 0, X = 1) - \widehat {\operatorname{pr}} (Y = 1 \mid Z = 0, X = 0)
$$

$$
\begin{array}{l} \begin{array}{c c} 2 3 4 & 5 5 \end{array} \\ - \overline {{2 3 4 + 3 6}} - \overline {{5 5 + 2 5}} \\ = 87 \% - 69 \% \\ = 18 \% > 0. \\ \end{array}
$$

因此，在两种处理水平下，结石较小的患者都有更高的成功概率。统计上，在两种处理水平下，$X$ 和 $Y$ 存在正关联。

我们可以将图 1.1 中的定性关联总结为示意图。用技术术语来说，处理对结局有一条正向直接路径和一条更负向的间接路径，因此处理与结局之间的整体关联为负。用通俗的话说，当效果较差的处理 0 更频繁地应用于较轻的病例时，它可能看起来是更有效的处理。

## 1.3.3 尤尔-辛普森悖论的几何解释（Geometry of the Yule–Simpson Paradox）

**假设基于汇总数据的 $2 \times 2$ 表具有以下计数**

<table><tr><td>整个人群</td><td>$ Y = 1 $</td><td>$ Y = 0 $</td></tr><tr><td>$ Z = 1 $</td><td>$ n_{11} $</td><td>$ n_{10} $</td></tr><tr><td>$ Z = 0 $</td><td>$ n_{01} $</td><td>$ n_{00} $</td></tr></table>

基于 $X = 1$ 和 $X = 0$ 子组的两个 $2 \times 2$ 表具有以下计数

<table><tr><td>子人群 X = 1</td><td>Y = 1</td><td>Y = 0</td></tr><tr><td>Z = 1</td><td> $n_{11|1}$ </td><td> $n_{10|1}$ </td></tr><tr><td>Z = 0</td><td> $n_{01|1}$ </td><td> $n_{00|1}$ </td></tr><tr><td>子人群 X = 0</td><td>Y = 1</td><td>Y = 0</td></tr><tr><td>Z = 1</td><td> $n_{11|0}$ </td><td> $n_{10|0}$ </td></tr><tr><td>Z = 0</td><td> $n_{01|0}$ </td><td> $n_{00|0}$ </td></tr></table>

图 1.2 展示了尤尔-辛普森悖论的几何解释。y 轴表示 Y = 1 的成功计数，x 轴表示 Y = 0 的失败计数。两个平行四边形对应于在两个处理水平下聚合成功和失败的计数。$OA_{1}$ 的斜率大于 $OB_{1}$ 的斜率，$OA_{0}$ 的斜率大于 $OB_{0}$ 的斜率。因此，在 $X$ 的两个水平内，处理似乎对结局有益。然而，OA 的斜率小于 OB 的斜率。因此，对于整个人群，处理似乎对结局有害。尤尔-辛普森悖论由此产生。

## 1.4 伯克利研究生院录取数据（The Berkeley graduate school admission data）

Bickel 等人（1975）调查了伯克利研究生院（Berkeley graduate school）的男女学生录取率。R 包 `datasets` 包含了原始数据 `UCBAdmissions`。六个最大院系的原始数据如下所示：

> library(datasets)

1.4 伯克利研究生院录取数据

```python
> UCBAdmissions = aperm(UCBAdmissions, c(2, 1, 3))
> UCBAdmissions
, , Dept = A
```

**录取（Admit）**

<table><tr><td>性别（Gender）</td><td>录取（Admitted）</td><td>拒绝（Rejected）</td></tr><tr><td>男（Male）</td><td>512</td><td>313</td></tr><tr><td>女（Female）</td><td>89</td><td>19</td></tr></table>

```python
, , Dept = B
```

**录取（Admit）**

<table><tr><td>性别（Gender）</td><td>录取（Admitted）</td><td>拒绝（Rejected）</td></tr><tr><td>男（Male）</td><td>353</td><td>207</td></tr><tr><td>女（Female）</td><td>17</td><td>8</td></tr></table>

```python
, , Dept = C
```

**录取（Admit）**

<table><tr><td>性别（Gender）</td><td>录取（Admitted）</td><td>拒绝（Rejected）</td></tr><tr><td>男（Male）</td><td>120</td><td>205</td></tr><tr><td>女（Female）</td><td>202</td><td>391</td></tr></table>

```txt
，，Dept = D
```

**录取（Admit）**

<table><tr><td>性别（Gender）</td><td>录取（Admitted）</td><td>拒绝（Rejected）</td></tr><tr><td>男（Male）</td><td>138</td><td>279</td></tr><tr><td>女（Female）</td><td>131</td><td>244</td></tr></table>

```txt
, , Dept = E
```

**录取（Admit）**

<table><tr><td>性别（Gender）</td><td>录取（Admitted）</td><td>拒绝（Rejected）</td></tr><tr><td>男（Male）</td><td>53</td><td>138</td></tr><tr><td>女（Female）</td><td>94</td><td>299</td></tr></table>

```python
, , Dept = F
```

**录取（Admit）**

<table><tr><td>性别（Gender）</td><td>录取（Admitted）</td><td>拒绝（Rejected）</td></tr><tr><td>男（Male）</td><td>22</td><td>351</td></tr><tr><td>女（Female）</td><td>24</td><td>317</td></tr></table>

将数据按院系汇总后，我们得到一个简单的 $2 \times 2$ 表格：

```julia
> UCBAdmissions.sum = apply(UCBAdmissions, c(1, 2), sum)
> UCBAdmissions.sum
Admit
Gender Admitted Rejected
```

<table><tr><td>男（Male）</td><td>1198</td><td>1493</td></tr><tr><td>女（Female）</td><td>557</td><td>1278</td></tr></table>

以下函数基于 `chisq.test` 构建，以 $2 \times 2$ 表格作为输入，并输出估计的**风险差（risk difference, RD）**和 p 值：

```diff
> risk.difference = function(tb2)
+ {
+    p1 = tb2[1, 1]/(tb2[1, 1] + tb2[1, 2])
+    p2 = tb2[2, 1]/(tb2[2, 1] + tb2[2, 2])
+    testp = chisq.test(tb2)
+
+    return(list(p.diff = p1 - p2,
+    pv = testp$p.value))
+ }
```

使用此函数，我们发现男女学生的录取率之间存在较大且显著的差异：

```txt
> risk.difference(UCBAdmissions.sum)
$p.diff
[1] 0.1416454
$pv
[1] 1.055797e-21
```

按院系分层后，我们发现男女学生录取率之间的差异变小且不显著。在 A 院系中，差异显著但为负值。

```txt
> P.diff = rep(0, 6)
> PV = rep(0, 6)
> for(dd in 1:6)
+ {
+ department = risk.difference(UCBAdmissions[, , dd])
+ P.diff[dd] = department$p.diff
+ PV[dd] = department$pv
+ }
>
> round(P.diff, 2)
[1] -0.20 -0.05 0.03 -0.02 0.04 -0.01
> round(PV, 2)
[1] 0.00 0.77 0.43 0.64 0.37 0.64
```

## 1.5 课后习题（Homework Problems）

## 1.1 在 $2 \times 2$ 表格中的独立性（Independence in two-by-two tables）

证明命题 1.1 中的 (1) 和 (2)。

## 1.5 课后习题（Homework Problems）

## 1.2 相关性与偏相关性（Correlation and partial correlation）

考虑一个三维正态随机向量：

$$
\left( \begin{array}{c} X \\ Y \\ Z \end{array} \right) \sim \mathrm{N} \left(\left( \begin{array}{c} 0 \\ 0 \\ 0 \end{array} \right), \left( \begin{array}{c c c} 1 & \rho_ {X Y} & \rho_ {X Z} \\ \rho_ {X Y} & 1 & \rho_ {Y Z} \\ \rho_ {X Z} & \rho_ {Y Z} & 1 \end{array} \right)\right).
$$

X 和 Y 之间的相关系数为 $\rho_{XY}$ 。**偏相关系数（partial correlation coefficient）**有多种等价定义。对于多元正态向量，令 $\rho_{XY|Z}$ 表示给定 Z 后 X 和 Y 之间的偏相关系数，其定义为在条件分布 $(X,Y)\mid Z$ 中它们的相关系数。证明：

$$
\rho_ {X Y | Z} = \frac {\rho_ {X Y} - \rho_ {X Z} \rho_ {Y Z}}{\sqrt {1 - \rho_ {X Z} ^ {2}} \sqrt {1 - \rho_ {Y Z} ^ {2}}}
$$

给出一个 $\rho_{XY} > 0$ 且 $\rho_{XY|Z} < 0$ 的例子。

备注：这是正态随机向量的**尤尔-辛普森悖论（Yule–Simpson Paradox）**。

## 1.3 模型设定搜索（Specification searches）

第 1.2.1 节使用 `LalondeRegression.R` 中的 R 代码重新分析了 Hainmueller（2012）使用的数据。数据总共包含 10 个协变量，因此在线性回归中可能的协变量子集有 $2^{10} = 1024$ 个。运行所有可能的协变量子集的 1024 个线性回归，并报告**处理变量（treatment）**的回归系数。其中有多少是正向显著的，多少是负向显著的，多少是不显著的？你也可以报告这些回归中的其他有趣发现。

## 1.4 更多关于种族歧视的内容（More on racial discrimination）

第 1.2.2 节使用 `resume.R` 中的 R 代码重新分析了 Bertrand 和 Mullainathan（2004）收集的数据。分别对男性和女性进行分析。从这些子组分析中你发现了什么？

## 1.5 推荐阅读（Recommended reading）

Bickel 等人（1975）是第 1.4 节中报告的悖论的原始论文。

| 

一

一

## 2
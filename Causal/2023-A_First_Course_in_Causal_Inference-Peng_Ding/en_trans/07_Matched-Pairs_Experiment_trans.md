# 配对实验（Matched-Pairs Experiment）

**配对实验（Matched-Pairs Experiment, MPE）** 是**分层随机实验（Stratified Randomized Experiment, SRE）** 的最极端版本，其中每个层内只有一个处理单元和一个对照单元。在这种情况下，层也被称为**配对（pairs）**。尽管这类实验是第5章讨论的SRE的一个特例，但它有其自身的估计和推断策略。此外，它具有许多新特征，并且与将在第15章讨论的观察性研究中的"匹配"策略密切相关。因此，我们在此单独一章中讨论MPE。

## 7.1 实验设计与潜在结果（Design of the experiment and potential outcomes）

考虑一个包含 $2n$ 个单元的实验。如果我们有对结果具有预测能力的协变量，我们可以根据协变量的相似性对单元进行配对。对于单个协变量，我们可以根据该协变量对单元进行排序，然后基于相邻单元形成配对。对于多个协变量，我们可以定义单元之间的两两距离，然后根据这些距离形成配对。在这种情况下，配对匹配可以使用**贪心算法（greedy algorithm）** 或**最优非二分匹配算法（optimal nonbipartite matching algorithm）** 来完成。贪心算法将距离最小的两个单元配对，将它们从单元池中移除，然后将剩余单元中距离最小的两个配对，依此类推。最优非二分匹配算法将 $2n$ 个单元分成 $n$ 个由两个单元组成的配对，以最小化配对内距离之和。有关MPE计算方面的更多细节，请参见Greevy等人（2004）。在本章中，我们假设配对是基于协变量形成的，并讨论后续的设计和分析问题。

令 $(i, j)$ 表示配对 $i$ 中的单元 $j$，其中 $i = 1 , \ldots , n$，$j = 1 , 2$。单元 $(i, j)$ 在处理和对照下的潜在结果分别为 $Y _ { i j } ( 1 )$ 和 $Y _ { i j } ( 0 )$。在每个配对内，我们随机分配一个单元接受处理，另一个单元接受对照。令

$$
Z _ {i} = \left\{ \begin{array}{l l} 1, & \text { 如果第一个单元接受处理 }, \\ 0, & \text { 如果第二个单元接受处理 }. \end{array} \right.
$$

我们可以基于处理分配机制正式定义MPE。

**定义 7.1 (MPE)** 我们有

$$
(Z _ {i}) _ {i = 1} ^ {n} \stackrel {{I I D}} {{\sim}} \text { Bernoulli } (1 / 2). \tag {7.1}
$$

配对 $i$ 内的观测结果为

$$
Y _ {i 1} = Z _ {i} Y _ {i 1} (1) + (1 - Z _ {i}) Y _ {i 1} (0) = \left\{ \begin{array}{l l} Y _ {i 1} (1), & \text {如果} Z _ {i} = 1; \\ Y _ {i 1} (0), & \text {如果} Z _ {i} = 0; \end{array} \right.
$$

和

$$
Y _ {i 2} = Z _ {i} Y _ {i 2} (0) + (1 - Z _ {i}) Y _ {i 2} (1) = \left\{ \begin{array}{l l} Y _ {i 2} (0), & \text {如果} Z _ {i} = 1; \\ Y _ {i 2} (1), & \text {如果} Z _ {i} = 0. \end{array} \right.
$$

因此，观测数据为 $( Z _ { i } , Y _ { i 1 } , Y _ { i 2 } ) _ { i = 1 } ^ { n }$。

## 7.2 费希尔随机化检验（FRT）

与之前的讨论类似，我们总是可以使用**费希尔随机化检验（Fisher Randomization Test, FRT）** 来检验严格零假设：

$$
H _ {0 \mathrm{F}}: Y _ {i j} (1) = Y _ {i j} (0) \text {  对于所有 } i = 1, \dots n \text { 和 } j = 1, 2.
$$

在进行FRT时，我们需要从 (7.1) 模拟 $( Z _ { i } , \ldots , Z _ { n } )$ 的分布。我将讨论一些基于配对内处理结果与对照结果之差的检验统计量的典型选择：

$\begin{array} { r l } { \hat { \tau } _ { i } } & { { } = } \end{array}$ 处理下的结果 − 对照下的结果（在配对 $i$ 内）

$$
= (2 Z _ {i} - 1) \left(Y _ {i 1} - Y _ {i 2}\right)
$$

$$
= S _ {i} (Y _ {i 1} - Y _ {i 2}),
$$

其中 $S _ { i } ~ = ~ 2 Z _ { i } - 1$ 是均值为0、方差为1的独立同分布随机符号，$i = 1 , \ldots , n$。由于 $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ 为零的配对不会对随机化分布做出贡献，我们在讨论FRT时剔除这些配对。

**例 7.1 (配对 t 统计量)** 配对内差异的均值为

$$
\hat {\tau} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\tau} _ {i}.
$$

在 $H _ { \mathrm { 0 F } }$ 下，

$$
E (\hat {\tau}) = 0
$$

且

$$
\operatorname{var} (\hat {\tau}) = n ^ {- 2} \sum_ {i = 1} ^ {n} \operatorname{var} (\hat {\tau} _ {i}) = n ^ {- 2} \sum_ {i = 1} ^ {n} \operatorname{var} (S _ {i}) (Y _ {i 1} - Y _ {i 2}) ^ {2} = n ^ {- 2} \sum_ {i = 1} ^ {n} \hat {\tau} _ {i} ^ {2}.
$$

基于独立随机变量之和的中心极限定理（Central Limit Theorem, CLT），我们有正态近似：

$$
\frac {\hat {\tau}}{\sqrt {n ^ {- 2} \sum_ {i = 1} ^ {n} \hat {\tau} _ {i} ^ {2}}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

我们可以使用这个正态近似来构建渐近检验。许多标准检验教材建议在MPE中使用以下配对 t 统计量：

$$
t _ {p a i r} = \frac {\hat {\tau}}{\sqrt {\{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\hat {\tau} _ {i} - \hat {\tau}) ^ {2}}},
$$

在 $H _ { \mathrm { 0 F } }$ 下，当 $n$ 较大且 $\hat {\tau}$ 较小时，该统计量与 $\hat {\tau}$ 几乎相同。

在经典统计学中，使用 $t _ { \mathrm { p a i r } }$ 的动机基于一个不同的框架。当 $\hat { \tau } _ { i } \stackrel { \mathrm { I I D } } { \sim } \mathrm { N } ( 0 , \sigma ^ { 2 } )$ 时，我们可以证明 $t _ { \mathrm { p a i r } } \sim t ( n - 1 )$，即 $t _ { \mathrm { p a i r } }$ 的精确分布是自由度为 $n - 1$ 的 t 分布，当 $n$ 较大时接近 $\mathrm { { N } } ( 0 , 1 )$。R 函数 `t.test` 使用参数 `paired=TRUE` 可以实现此检验。当 $n$ 较大时，这些程序给出相似的结果。例 7.1 中的讨论为经典的配对 t 检验提供了另一种证明，无需假设数据的正态性。

**例 7.2 (威尔科克森符号秩统计量)** 基于 $( \vert \hat { \tau } _ { 1 } \vert , \dots , \vert \hat { \tau } _ { n } \vert )$ 的秩 $( R _ { 1 } , \ldots , R _ { n } )$，我们可以定义一个检验统计量

$$
W = \sum_ {i = 1} ^ {n} I (\hat {\tau} _ {i} > 0) R _ {i}.
$$

在 $H _ { \mathrm { 0 F } }$ 下，

$$
E (W) = \frac {1}{2} \sum_ {i = 1} ^ {n} R _ {i} = \frac {1}{2} \sum_ {i = 1} ^ {n} i = \frac {n (n + 1)}{4}
$$

且

$$
\operatorname{var} (W) = \frac {1}{4} \sum_ {i = 1} ^ {n} R _ {i} ^ {2} = \frac {1}{4} \sum_ {i = 1} ^ {n} i ^ {2} = \frac {n (n + 1) (2 n + 1)}{2 4}.
$$

独立随机变量之和的CLT确保了以下正态近似：

$$
\frac {W - n (n + 1) / 4}{\sqrt {n (n + 1) (2 n + 1) / 2 4}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

我们可以使用这个正态近似来构建渐近检验。R 函数 `wilcox.test` 使用参数 `paired=TRUE` 可以实现这些检验。

**例 7.3 (科尔莫戈罗夫-斯米尔诺夫型统计量)** 在 $H _ { \mathrm { 0 F } }$ 下，绝对值 $( \vert \hat { \tau } _ { 1 } \vert , \dots , \vert \hat { \tau } _ { n } \vert )$ 是固定的，但它们的符号是随机的。因此，$( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } )$ 和 $- ( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } )$ 应具有相同的分布。令

$$
\hat {F} (t) = n ^ {- 1} \sum_ {i = 1} ^ {n} I (\hat {\tau} _ {i} \leq t)
$$

为 $( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } )$ 的经验分布函数，且

$$
1 - \hat {F} (- t -) = n ^ {- 1} \sum_ {i = 1} ^ {n} I (- \hat {\tau} _ {i} \leq t)
$$

为 $- ( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } )$ 的经验分布函数，其中 $\hat { F } ( - t - )$ 是函数 $\hat { F } ( \cdot )$ 在 $-t$ 处的左极限。于是，科尔莫戈罗夫-斯米尔诺夫型统计量为

$$
D = \max _ {t} | \hat {F} (t) + \hat {F} (- t -) - 1 |.
$$

Butler (1969) 提出了这个检验统计量，并推导了其精确分布和渐近分布。不幸的是，标准软件包中没有实现该统计量。尽管如此，我们可以模拟其精确分布，并基于FRT计算 p 值。¹

**例 7.4 (符号统计量)** 符号统计量仅使用配对内差异的符号

$$
\Delta = \sum_ {i = 1} ^ {n} I (\hat {\tau} _ {i} > 0).
$$

在 $H _ { \mathrm { 0 F } }$ 下，

$$
I (\hat {\tau} _ {i} > 0) \stackrel {I I D} {\sim} B e r n o u l l i (1 / 2)
$$

因此，

$$
\Delta \sim B i n o m i a l (n, 1 / 2).
$$

基于此，我们有一个精确的二项检验，该检验在 R 函数 `binom.test` 中实现，参数为 $p = 1 / 2$。使用CLT，我们还可以基于二项分布的以下正态近似进行检验：

$$
\frac {\Delta - n / 2}{\sqrt {n / 4}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

**表 7.1：四种配对类型的计数**

<table><tr><td></td><td>对照结果 1</td><td>对照结果 0</td></tr><tr><td>处理结果 1</td><td> $m_{11}$ </td><td> $m_{10}$ </td></tr><tr><td>处理结果 0</td><td> $m_{01}$ </td><td> $m_{00}$ </td></tr></table>

**例 7.5 (二元结果的麦克尼马尔统计量)** 如果结果是二元的，我们可以更紧凑地总结MPE的数据。给定一个配对，处理结果可以是1或0，对照结果可以是1或0，从而产生如表7.1所示的 $2 \times 2$ 表格。

在 $H _ { \mathrm { 0 F } }$ 下，一致配对的数量 $m _ { 1 1 }$ 和 $m_{00}$ 是固定的，并且 $m _ { 1 0 } + m _ { 0 1 }$ 也是固定的。因此，唯一的随机分量是 $m _ { 1 0 }$，其分布为

$$
m _ {1 0} \sim B i n o m i a l (m _ {1 0} + m _ {0 1}, 1 / 2).
$$

这意味着基于二项分布的精确检验。R 函数 `mcnemar.test` 提供了基于二项分布正态近似的渐近检验：

$$
\frac {m _ {1 0} - (m _ {1 0} + m _ {0 1}) / 2}{\sqrt {(m _ {1 0} + m _ {0 1}) / 4}} = \frac {m _ {1 0} - m _ {0 1}}{\sqrt {m _ {1 0} + m _ {0 1}}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

精确FRT和渐近检验都不依赖于 $m _ { 1 1 }$ 或 $m_{00}$。在这些检验中，只有不一致配对的数量是重要的。

## 7.3 奈曼推断（Neymanian inference）

配对 $i$ 内的平均因果效应为

$$
\tau_ {i} = \frac {1}{2} \left\{Y _ {i 1} (1) + Y _ {i 2} (1) - Y _ {i 1} (0) - Y _ {i 2} (0) \right\},
$$

所有单元的平均因果效应为

$$
\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \tau_ {i} = (2 n) ^ {- 1} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {2} \left\{Y _ {i j} (1) - Y _ {i j} (0) \right\}.
$$

直观上，$\hat { \tau } _ { i }$ 是 $\tau _ { i }$ 的无偏估计量，因此 $\hat { \tau }$ 是 $\tau$ 的无偏估计量。我们也可以计算 $\hat {\tau}$ 的方差。我将精确公式留作作业题，因为MPE只是SRE的一个特例。

然而，我们不能沿用SRE的策略来估计 $\hat {\tau}$ 的方差。配对内结果的样本方差无法良好定义，因为在每个配对内我们只有一个处理单元和一个对照单元。数据不允许我们估计配对 $i$ 内 $\hat { \tau } _ { i }$ 的方差。

在MPE中是否有可能估计 $\hat {\tau}$ 的方差？让我们暂时忘记MPE，将视角转换到经典的独立同分布（IID）抽样。如果 $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ 是均值为 $\mu$、方差为 $\sigma ^ { 2 }$ 的独立同分布随机变量，则 $\hat {\tau} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\tau} _ { i }$ 的方差为 $\sigma ^ { 2 } / n$。$\sigma ^ { 2 }$ 的一个无偏估计是 $( n - 1 ) ^ { - 1 } \sum_ {i = 1} ^ {n} ( \hat { \tau } _ { i } - \hat { \tau } ) ^ { 2 }$，因此 $\operatorname{var}(\hat {\tau})$ 的一个无偏估计量为

$$
\hat {V} = \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\hat {\tau} _ {i} - \hat {\tau}) ^ {2}.
$$

这一讨论也扩展到独立但非同分布的情形；参见第A1章中的问题A1.1。以上讨论似乎偏离了具有完全不同统计假设的MPE。但至少它启发了一个方差估计量 $\hat { V }$，该估计量使用 $\hat { \tau } _ { i }$ 的配对间方差来估计 $\hat {\tau}$ 的方差。当然，它是在不同假设下推导的。它对MPE是否有效？下面的定理7.1给出了一个正面的结果。

**定理 7.1** 在MPE下，$\hat { V }$ 是 $\hat {\tau}$ 真实方差的保守估计量：

$$
E (\hat {V}) - \mathrm{var} (\hat {\tau}) = \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\tau_ {i} - \tau) ^ {2} \geq 0.
$$

如果 $\tau _ { i }$ 在各配对间为常数，则 $E( \hat { V } ) = \operatorname { v a r } ( \hat { \tau } )$。

定理7.1指出，在MPE下，$\hat { V }$ 通常是一个保守的方差估计量，并且如果平均因果效应在各配对间为常数，则变为无偏估计量。这有点令人惊讶，因为 $\hat { V }$ 依赖于 $\hat { \tau } _ { i }$ 的配对间方差，而 $\operatorname{var}(\hat {\tau})$ 依赖于每个 $\hat { \tau } _ { i }$ 的配对内方差。下面的证明可能为这一令人惊讶的结果提供一些见解。

**定理7.1的证明：** 使用基本代数事实 $\sum_ {i = 1} ^ {n} ( a _ { i } - \bar { a } ) ^ { 2 } = \sum_ {i = 1} ^ {n} a _ { i } ^ { 2 } - n \bar { a } ^ { 2 }$，在以下步骤2和5中，我们有

$$
\begin{array}{l} n (n - 1) E (\hat {V}) = E \left\{\sum_ {i = 1} ^ {n} (\hat {\tau} _ {i} - \hat {\tau}) ^ {2} \right\} \\ = E \left(\sum_ {i = 1} ^ {n} \hat {\tau} _ {i} ^ {2} - n \hat {\tau} ^ {2}\right) \\ = \sum_ {i = 1} ^ {n} \left\{\operatorname{var} \left(\hat {\tau} _ {i}\right) + \tau_ {i} ^ {2} \right\} - n \left\{\operatorname{var} (\hat {\tau}) + \tau^ {2} \right\} \\ = \sum_ {i = 1} ^ {n} \operatorname{var} (\hat {\tau} _ {i}) - n \operatorname{var} (\hat {\tau}) + \sum_ {i = 1} ^ {n} \tau_ {i} ^ {2} - n \tau^ {2} \\ = n ^ {2} \mathrm{var} (\hat {\tau}) - n \mathrm{var} (\hat {\tau}) + \sum_ {i = 1} ^ {n} (\tau_ {i} - \tau) ^ {2}. \\ \end{array}
$$

因此，

$$
E (\hat {V}) = \operatorname{var} (\hat {\tau}) + \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\tau_ {i} - \tau) ^ {2} \geq \operatorname{var} (\hat {\tau}).
$$

![image_07](images/image_07.png)

与其他实验的讨论类似，**奈曼方法（Neymanian approach）** 依赖于大样本近似：

$$
\frac {\hat {\tau} - \tau}{\sqrt {\operatorname{var} (\hat {\tau})}} \to \mathrm{N} (0, 1)
$$

当 $n \to \infty$ 且某些正则条件成立时，依分布收敛。由于方差被高估，**沃尔德型置信区间（Wald-type confidence interval）**

$$
\hat {\tau} \pm z _ {1 - \alpha / 2} \sqrt {\hat {V}}
$$

以至少 $1 - \alpha$ 的概率覆盖 $\tau$。

点估计量 $\hat {\tau}$ 和方差估计量 $\hat { V }$ 都可以通过**普通最小二乘法（Ordinary Least Squares, OLS）** 方便地得到，如下命题所示。

**命题 7.1** $\widehat { \tau }$ 和 $\hat { V }$ 与仅对截距项拟合向量 $( \widehat { \tau } _ { 1 } , \ldots , \widehat { \tau } _ { n } ) ^ { \mathsf { T } }$ 的OLS中截距项的系数和方差估计量相同。

我将命题7.1的证明留作问题7.3。

## 7.4 协变量调整（Covariate adjustment）

## 7.4.1 费希尔随机化检验（FRT）

与完全随机实验（CRE）中的讨论类似，在匹配对实验（MPE）中有两种通用的协变量调整策略。首先，我们可以基于对协变量进行结果模型拟合后的残差来构造检验统计量，因为在**尖锐零假设（sharp null hypothesis）**下，这些残差是固定数值。一个经典的选择是对所有观测到的 $Y_{ij}$ 关于 $X_{ij}$ 进行**普通最小二乘法（Ordinary Least Squares, OLS）**拟合，以获得残差 $\hat{\varepsilon}_{ij}$。然后，我们可以将 $\hat{\varepsilon}_{ij}$ 视为观测到的结果，并据此构造检验统计量。Rosenbaum (2002a) 特别提倡将这种策略应用于 MPE。

其次，我们可以直接使用模型拟合中的某些系数作为检验统计量。下一小节的讨论将为第二种策略提出一种检验统计量的选择。

## 7.4.2 回归调整（Regression adjustment）

尽管我们在设计阶段已经对协变量进行了匹配，但匹配可能并不完美，而且有时我们拥有超出配对匹配阶段所用协变量之外的额外协变量。在这些情况下，我们可以调整协变量以进一步提高估计效率。假设每个单元都有协变量 $X_{ij}$，我们可以像计算结果一样，计算协变量在配对内的差异 $\widehat{\tau}_{X,i}$ 及其平均值 $\hat{\tau}_{X}$。我们可以证明：

$$
E(\hat{\tau}_{X,i}) = 0, \quad E(\hat{\tau}_{X}) = 0,
$$

以及

$$
\operatorname{cov}(\hat{\tau}_{X}) = n^{-2} \sum_{i=1}^{n} \hat{\tau}_{X,i} \hat{\tau}_{X,i}^{\mathsf{T}}.
$$

在一个已实现的 MPE 中，除非所有的 $\hat{\tau}_{X,i}$ 都为零，否则 $\operatorname{cov}(\hat{\tau}_{X})$ 不为零。如果 $(Z_1, \ldots, Z_n)$ 的抽取不理想，$\hat{\tau}_{X}$ 有可能显著偏离零。与 CRE 中的讨论类似，调整协变量均值的不平衡可能会提高估计效率。

考虑一类由 $\gamma$ 索引的估计量：

$$
\hat{\tau}(\gamma) = \hat{\tau} - \gamma^{\mathsf{T}} \hat{\tau}_{X}
$$

对于任何固定的 $\gamma$，其均值为 0。我们希望选择 $\gamma$ 以最小化 $\hat{\tau}(\gamma)$ 的方差。其方差是 $\gamma$ 的二次函数：

$$
\mathrm{var}\{\hat{\tau}(\gamma)\} = \mathrm{var}(\hat{\tau} - \gamma^{\mathsf{T}} \hat{\tau}_{X}) = \mathrm{var}(\hat{\tau}) + \gamma^{\mathsf{T}} \mathrm{cov}(\hat{\tau}_{X}) \gamma - 2 \gamma^{\mathsf{T}} \mathrm{cov}(\hat{\tau}_{X}, \hat{\tau}),
$$

该函数在以下点取得最小值：

$$
\tilde{\gamma} = \mathrm{cov}(\hat{\tau}_{X})^{-1} \mathrm{cov}(\hat{\tau}_{X}, \hat{\tau}).
$$

我们已在上面得到了 $\operatorname{cov}(\hat{\tau}_{X})$ 的公式，它也可以写为：

$$
\operatorname{cov}(\hat{\tau}_{X}) = n^{-2} \sum_{i=1}^{n} |\hat{\tau}_{X,i}| |\hat{\tau}_{X,i}|^{\mathsf{T}},
$$

其中 $|\cdot|$ 表示向量的逐分量绝对值。因此，$\operatorname{cov}(\hat{\tau}_{X})$ 是固定且已知的（来自观测数据）。然而，$\operatorname{cov}(\hat{\tau}_{X}, \hat{\tau})$ 依赖于未知的潜在结果。幸运的是，如下面的定理 7.2 所示，我们可以得到它的一个无偏估计量。

**定理 7.2** $\operatorname{cov}(\hat{\tau}_{X}, \hat{\tau})$ 的一个无偏估计量是：

$$
\hat{\theta} = \{n (n - 1)\}^{-1} \sum_{i=1}^{n} (\hat{\tau}_{X,i} - \hat{\tau}_{X}) (\hat{\tau}_{i} - \hat{\tau}).
$$

定理 7.2 的证明与定理 7.1 类似。我将其留作问题 7.2。

因此，我们可以通过下式估计最优系数 $\tilde{\gamma}$：

$$
\begin{array}{l} \hat{\gamma} = \left(n^{-2} \sum_{i=1}^{n} \hat{\tau}_{X,i} \hat{\tau}_{X,i}^{\mathsf{T}}\right)^{-1} \left\{\{n (n - 1)\}^{-1} \sum_{i=1}^{n} (\hat{\tau}_{X,i} - \hat{\tau}_{X}) (\hat{\tau}_{i} - \hat{\tau}) \right\} \\ \approx \left(\sum_{i=1}^{n} (\hat{\tau}_{X,i} - \hat{\tau}_{X}) (\hat{\tau}_{X,i} - \hat{\tau}_{X})^{\mathsf{T}}\right)^{-1} \sum_{i=1}^{n} (\hat{\tau}_{X,i} - \hat{\tau}_{X}) (\hat{\tau}_{i} - \hat{\tau}), \\ \end{array}
$$

这近似于将 $\hat{\tau}_{i}$ 对 $\hat{\tau}_{X,i}$（含截距项）进行 OLS 拟合时，$\hat{\tau}_{X,i}$ 的系数。最终的估计量是：

$$
\hat{\tau}_{\mathrm{adj}} = \hat{\tau}(\hat{\gamma}) = \hat{\tau} - \hat{\gamma}^{\mathsf{T}} \hat{\tau}_{X},
$$

根据 OLS 的性质，这近似于将 $\hat{\tau}_{i}$ 对 $\hat{\tau}_{X,i}$（含截距项）进行 OLS 拟合时的截距项。

那么，$\hat{\tau}_{\mathrm{adj}}$ 的一个保守方差估计量为：

$$
\hat{V}_{\mathrm{adj}} = \hat{V} + \hat{\gamma}^{\mathsf{T}} \mathrm{cov}(\hat{\tau}_{X}) \hat{\gamma} - 2 \hat{\gamma}^{\mathsf{T}} \hat{\theta} = \hat{V} - \hat{\theta}^{\mathsf{T}} \mathrm{cov}(\hat{\tau}_{X})^{-1} \hat{\theta}.
$$

一个微妙的技术问题是 $\hat{\tau}(\hat{\gamma})$ 是否与 $\hat{\tau}(\tilde{\gamma})$ 具有相同的最优性。在大样本情况下，我们可以证明 $\hat{\tau}(\hat{\gamma}) - \hat{\tau}(\tilde{\gamma}) = -(\hat{\gamma} - \tilde{\gamma})^{\top} \hat{\tau}_{X}$ 是更高阶的无穷小量，因为它是两个“小”量 $\hat{\gamma} - \tilde{\gamma}$ 和 $\hat{\tau}_{X}$ 的乘积。我在此省略渐近分析的繁琐细节，但希望结果对读者来说在直觉上是合理的。

此外，Fogarty (2018b) 讨论了上述协变量调整过程的渐近等价回归形式，并为相关的**中心极限定理（Central Limit Theorem, CLT）**给出了严格的证明。我在下面总结回归形式，但不给出正则条件。

**命题 7.2** 在 MPE 下，协变量调整估计量 $\hat{\tau}_{\mathrm{adj}}$ 及其相关的方差估计量 $\hat{V}_{\mathrm{adj}}$ 可以方便地通过将向量 $\hat{\tau}_{i}$ 对常数项 1 和矩阵 $\hat{\tau}_{X,i}$ 进行 OLS 拟合得到的截距项及其相关的方差估计量来近似。

我将命题 7.2 的证明留作问题 7.3。有趣的是，命题 7.1 和 7.2 都不需要对方差估计量进行**异方差稳健标准误（EHW）**校正。因为我们将数据从 MRE 缩减为配对内差异，所以无需像 Lin (2013) 针对 CRE 的估计量那样对协变量进行中心化。

## 7.5 例子（Examples）

## 7.5.1 达尔文比较异花受精和自花受精对玉米高度的数据（Darwin's data comparing cross-fertilizing and self-fertilizing on the height of corns）

这是来自 Fisher (1935) 的一个经典例子。它包含 15 对玉米，分别进行异花受精或自花受精，高度作为结果变量。R 包 **HistData** 提供了原始数据，其中 `cross` 和 `self` 分别是异花受精和自花受精下的高度，`diff` 表示它们的差值。

<table><tr><td colspan="6">&gt; library(&quot;HistData&quot;)</td></tr><tr><td colspan="6">&gt; ZeaMays</td></tr><tr><td></td><td>pair</td><td>pot</td><td>cross</td><td>self</td><td>diff</td></tr><tr><td>1</td><td>1</td><td>1</td><td>23.500</td><td>17.375</td><td>6.125</td></tr><tr><td>2</td><td>2</td><td>1</td><td>12.000</td><td>20.375</td><td>-8.375</td></tr><tr><td>3</td><td>3</td><td>1</td><td>21.000</td><td>20.000</td><td>1.000</td></tr><tr><td>4</td><td>4</td><td>2</td><td>22.000</td><td>20.000</td><td>2.000</td></tr><tr><td>5</td><td>5</td><td>2</td><td>19.125</td><td>18.375</td><td>0.750</td></tr><tr><td>6</td><td>6</td><td>2</td><td>21.500</td><td>18.625</td><td>2.875</td></tr><tr><td>7</td><td>7</td><td>3</td><td>22.125</td><td>18.625</td><td>3.500</td></tr><tr><td>8</td><td>8</td><td>3</td><td>20.375</td><td>15.250</td><td>5.125</td></tr><tr><td>9</td><td>9</td><td>3</td><td>18.250</td><td>16.500</td><td>1.750</td></tr><tr><td>10</td><td>10</td><td>3</td><td>21.625</td><td>18.000</td><td>3.625</td></tr><tr><td>11</td><td>11</td><td>3</td><td>23.250</td><td>16.250</td><td>7.000</td></tr><tr><td>12</td><td>12</td><td>4</td><td>21.000</td><td>18.000</td><td>3.000</td></tr><tr><td>13</td><td>13</td><td>4</td><td>22.125</td><td>12.750</td><td>9.375</td></tr><tr><td>14</td><td>14</td><td>4</td><td>23.000</td><td>15.500</td><td>7.500</td></tr><tr><td>15</td><td>15</td><td>4</td><td>12.000</td><td>18.000</td><td>-6.000</td></tr></table>

总共有 $2^{15}=32768$ 种可能的处理分配，这在 R 中是一个可处理的数字。以下函数可以枚举 MPE 的所有可能处理分配：

```txt
MP_enumerate = function(i, n.pairs)
{
    if (i > 2^n.pairs) print("i is too large.")
    a = 2^(n.pairs - 1):0)
    b = 2*a
    2*sapply(i - 1,
    function(x)
    as.integer((x %% b) >= a)) - 1
}
```

因此，我们枚举所有处理分配，并计算相应的 $\hat{\tau}$ 和单侧精确 p 值。

```txt
> difference = ZeaMays$diff
> n.pairs = length(difference)
```

## 7.5 例子（Examples）

图 7.1 显示了 $\hat{\tau}$ 的精确随机化分布。

```diff
> abs.diff = abs(difference)
> t.obs = mean(difference)
> t.ran = sapply(1:2^15,
+ function(x){
+ sum(MP_enumerate(x, 15)*abs.diff)
+ })/n.pairs
> pvalue = mean(t.ran>=t.obs)
> pvalue
[1] 0.02633667
```

## 7.5.2 儿童电视工作室实验数据（Children's television workshop experiment data）

我还重新分析了 Ball 等人 (1973) 的数据，该数据也曾被 Imbens 和 Rubin (2015) 分析过。它包含 8 对数据，下表总结了配对内的协变量和结果，以及它们的差异：

```txt
> dataxy
x.control x.treatment y.control y.treatment diffx diffy
1 12.9 12.0 54.6 60.6 -0.9 6.0
2 15.1 12.3 56.5 55.5 -2.8 -1.0
3 16.8 17.2 75.2 84.8 0.4 9.6
4 15.8 18.9 75.6 101.9 3.1 26.3
5 13.9 15.3 55.3 70.6 1.4 15.3
6 14.5 16.6 59.3 78.4 2.1 19.1
```

<table><tr><td>7</td><td>17.0</td><td>16.0</td><td>87.0</td><td>84.2</td><td>-1.0</td><td>-2.8</td></tr><tr><td>8</td><td>15.8</td><td>20.1</td><td>73.7</td><td>108.6</td><td>4.3</td><td>34.9</td></tr></table>

我们可以使用 OLS 来获得点估计量和标准误：在不调整协变量的情况下，我们得到：

```txt
> unadj = summary(lm(diffy ~ 1, data = dataxy))$coef
> round(unadj, 3)
Estimate Std. Error t value Pr(>|t|)
(Intercept) 13.425 4.636 2.896 0.023
```

在调整协变量后，我们得到：

```txt
> adj = summary(lm(diffy ~ diffx, data = dataxy))$coef
> round(adj, 3)
Estimate Std. Error t value Pr(>|t|)
(Intercept) 8.994 1.410 6.381 0.001
diffx 5.371 0.599 8.964 0.000
```

上述结果假设 $n$ 很大，如果我们相信大样本近似，则 p 值是合理的。然而，$n=8$ 并不大。总共有 $2^8=256$ 种可能的处理分配，因此最小的可能 p 值是 $1/256 = 0.0039$，这远大于基于协变量调整估计量正态近似的 p 值。在这个例子中，使用带有**学生化统计量（studentized statistic）**（即来自 `lm` 函数的 t 值）的 FRT 来计算精确 p 值更为合理。图 7.2 显示了这两个学生化统计量的精确分布，以及双侧 p 值。该图突出了一个事实，即检验统计量的随机化分布是离散的，最多取 256 个可能值。正态近似不太可能准确，尤其是在尾部。我们应该报告基于 FRT 的 p 值。

## 7.6 比较 MPE 和 CRE（Comparing the MPE and CRE）

Imai (2008b) 比较了 MPE 和 CRE。直观的结论是，如果匹配做得好并且协变量能预测结果，那么 MPE 能给出更精确的估计量。然而，在设计阶段没有结果数据的情况下，很难判断这一点是否成立。在 FRT 中，如果协变量能预测结果，MPE 通常比 CRE 提供更强大的检验。Greevy 等人 (2004) 使用基于**威尔科克森符号秩统计量（Wilcoxon sign rank statistic）**的模拟说明了这一点。然而，这在有限样本情况下可能是一个微妙的问题。考虑一个包含 $2n$ 个实验单元的实验，其中 $n$ 个单元接受处理，$n$ 个单元接受对照。如果我们在 0.05 的水平上检验尖锐零假设，那么在 MPE 中，我们至少需要 $2 \times 5 = 10$ 个单元，因为最小 p 值是 $1/2^5 = 1/32 < 0.05$，但 $1/2^4 = 1/16 > 0.05$；但在 CRE 中，我们至少需要 $2 \times 4 = 8$ 个单元，因为最小 p 值是 $1/\binom{8}{4} = 1/70 < 0.05$，但 $1/\binom{6}{3} = 1/20 = 0.05$。因此，对于 8 个单元，在 MPE 中不可能拒绝尖锐零假设，但在 CRE 中却有可能。即使协变量是结果的完美预测因子，基于 FRT 的 MPE 也并不优于 CRE。

## 7.7 推广到一般匹配实验（Extension to the general matched experiment）

将 MPE 推广到具有不同数量对照单元的一般匹配实验是直接的。假设我们有 $n$ 个匹配集，由 $i=1,\ldots,n$ 索引。对于匹配集 $i$，我们有 $1+M_i$ 个单元。$M_i$ 可以不同。实验单元的总数是 $N = n + \sum_{i=1}^{n} M_i$。令 $ij$ 表示匹配集 $i$ 内的第 $j$ 个单元 ($i=1,\ldots,n$ 且 $j=1,\ldots,M_i+1$)。单元 $ij$ 在处理和对照下的潜在结果分别为 $Y_{ij}(1)$ 和 $Y_{ij}(0)$。

在匹配集 $i \ (i=1,\ldots,n)$ 内，实验者随机选择恰好一个单元接受处理，其余 $M_i$ 个单元接受对照。这个一般匹配实验也是大小为 $1+M_i \ (i=1,\ldots,n)$ 的 $n$ 个层的**分层随机实验（Stratified Randomized Experiment, SRE）**的一个特例。令 $Z_{ij}$ 为单元 $ij$ 的处理指示变量，它揭示其中一个潜在结果为：

$$
Y_{ij} = Z_{ij} Y_{ij}(1) + (1 - Z_{ij}) Y_{ij}(0).
$$

匹配集 $i$ 内的平均因果效应等于：

$$
\tau_i = (M_i + 1)^{-1} \sum_{j=1}^{1+M_i} \{Y_{ij}(1) - Y_{ij}(0)\}.
$$

由于这是一个 SRE，$\tau_i$ 的一个无偏估计量是：

$$
\hat{\tau}_i = \sum_{j=1}^{M_i+1} Z_{ij} Y_{ij} - M_i^{-1} \sum_{i=1}^{n} (1 - Z_{ij}) Y_{ij}
$$

这是在匹配集 $i$ 内结果均值的差异。

下面我们讨论一般匹配实验的统计推断。

## 7.7.1 费希尔随机化检验（FRT）

和往常一样，我们总是可以使用 FRT 来检验尖锐零假设：

$$
H_{0\mathrm{F}}: Y_{ij}(1) = Y_{ij}(0) \text{ for all } i=1,\dots,n; j=1,\dots,M_i+1.
$$

因为一般匹配实验是包含许多小层的 SRE 的一个特例，我们可以使用示例 5.4、5.5、7.2、7.3、7.4 中定义的检验统计量，以及以下两个小节中的估计量和相应的 t 统计量。

## 7.7.2 估计层内效应的平均值（Estimating the average of the within-strata effects）

我们首先关注**层内效应（within-strata effects）**平均值的估计：

$$
\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \tau_ {i}.
$$

它有一个无偏估计量：

$$
\hat {\tau} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\tau} _ {i}.
$$

有趣的是，我们可以证明**定理 7.1（Theorem 7.1）**对于一般匹配实验成立，因此**匹配对实验（Matched Pair Experiment, MPE）**的其他结果也成立。特别地，我们可以使用 $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ 对截距项的**普通最小二乘法（Ordinary Least Squares, OLS）**拟合来获得 $\tau$ 的点估计和方差估计。在有协变量的情况下，我们可以使用 $\hat { \tau } _ { i } ^ { \phantom { \dagger } } \rangle$ 对截距项和 ${ \hat { \tau } } _ { X , i } { \mathrm { ' s } }$ 进行 OLS 拟合，其中

$$
\hat {\tau} _ {X, i} = \sum_ {j = 1} ^ {M _ {i} + 1} Z _ {i j} X _ {i j} - M _ {i} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i j}) X _ {i j}
$$

是匹配集 $i$ 内协变量均值的相应差异。

## 7.7.3 更一般的因果估计量（A more general causal estimand）

重要的是，上述 $\tau$ 是 $\tau _ { i } ^ { \ , } \mathrm { s }$ 的平均值，当 $M _ { i } { ^ \mathrm { { \tiny ~ s } } }$ 变化时，它并不等于实验中 $N$ 个单元的平均因果效应。平均因果效应等于：

$$
\tau^ {\prime} = N ^ {- 1} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {1 + M _ {i}} \left\{Y _ {i j} (1) - Y _ {i j} (0) \right\} = \sum_ {i = 1} ^ {n} \frac {1 + M _ {i}}{N} \tau_ {i}.
$$

为了统一讨论，我考虑**加权因果效应（weighted causal effect）**：

$$
\tau_ {w} = \sum_ {i = 1} ^ {n} w _ {i} \tau_ {i}
$$

其中 $\textstyle \sum _ { i = 1 } ^ { n } w _ { i } = 1$。**匹配对实验（MPE）**是 $w _ { i } = n ^ { - 1 }$ 的特例，而 $\tau ^ { \prime }$ 是 $w _ { i } = ( 1 + M _ { i } ) / N$ （对于 $i = 1 , \ldots , n$）的特例。很容易得到一个无偏估计量：

$$
\hat {\tau} _ {w} = \sum_ {i = 1} ^ {n} w _ {i} \hat {\tau} _ {i},
$$

并计算其方差：

$$
\operatorname{var} (\hat {\tau} _ {w}) = \sum_ {i = 1} ^ {n} w _ {i} ^ {2} \operatorname{var} (\hat {\tau} _ {i}).
$$

然而，估计这个估计量的方差相当棘手，因为 $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ 是独立的随机变量，没有任何重复。这是由 Hartley 等人（1969）和 Rao（1970）研究的理论统计学中的一个著名问题。Fogarty（2018a）也讨论过这个问题，但没有认识到这些先前的工作。我将给出方差估计量的最终形式，而不详细说明其动机：

$$
\hat {V} _ {w} = \sum_ {i = 1} ^ {n} c _ {i} (\hat {\tau} _ {i} - \hat {\tau} _ {w}) ^ {2}
$$

其中

$$
c _ {i} = \frac {\frac {w _ {i} ^ {2}}{1 - 2 w _ {i}}}{1 + \sum_ {i = 1} ^ {n} \frac {w _ {i} ^ {2}}{1 - 2 w _ {i}}}.
$$

作为合理性检验，在 $M _ { i } = 1$ 且 $w _ { i } = n ^ { - 1 }$ 的 MPE 中，$c _ { i }$ 简化为 $\{ n ( n - 1 ) \} ^ { - 1 }$。为简单起见，我们关注所有 $i \mathrm { \ ' } _ { \mathrm { S } }$ 都满足 $w _ { i } < 1 / 2$ 的情况，即不存在包含超过总权重一半的匹配集。以下定理扩展了**定理 7.1（Theorem 7.1）**。

**定理 7.3（Theorem 7.3）** 在具有变化 $M _ { i }$ 的一般匹配实验中，我们有

$$
E (\hat {V} _ {w}) - \mathrm{var} (\hat {\tau} _ {w}) = \sum_ {i = 1} ^ {n} c _ {i} (\tau_ {i} - \tau_ {w}) ^ {2} \geq \mathrm{var} (\hat {\tau} _ {w}) \geq 0
$$

如果 $\tau _ { i }$ 是常数，则等式成立。

尽管 $\hat { V } _ { w }$ 的理论动机相当复杂，但直接验证**定理 7.3（Theorem 7.3）**并不太难。我将证明过程留作**问题 7.9（Problem 7.9）**。

## 7.8 家庭作业问题（Homework Problems）

## 7.1 MPE 中 $\hat{\tau}$ 的真实方差（The true variance of τˆ in the MPE）

用有限总体潜在结果的前两个矩来表示 $\mathrm{var}(\hat{\tau})$。

## 7.2 一个协方差估计量（A covariance estimator）

证明**定理 7.2（Theorem 7.2）**。

## 7.3 通过 OLS 的方差估计量（Variance estimators via OLS）

证明**命题 7.1（Proposition 7.1）**和**命题 7.2（Proposition 7.2）**。

## 7.4 二元结果变量的点估计和方差估计量（Point and variance estimator with binary outcome）

本题将**例 7.5（Example 7.5）**扩展到**内曼推断（Neymanian inference）**。

用**表 7.1（Table 7.1）**中的计数来表示 $\hat{\tau}$ 和 $\hat{V}$。

## 7.5 随机化检验的最小样本量（Minimum sample size for the FRT）

扩展**第 7.6 节（Section 7.6）**中的讨论。考虑一个包含 $2n$ 个单元的实验，其中 $n$ 个单元接受处理，$n$ 个单元接受对照，并在 0.001 的显著性水平下检验尖锐零假设。对于 MPE，使得最小 p 值不超过 0.001 的 $n$ 的最小值是多少？对于**完全随机实验（Completely Randomized Experiment, CRE）**，相应的 $n$ 的最小值是多少？

## 7.6 重新分析达尔文的数据（Re-analyzing Darwin’s data）

在 `MPEFRTdarwin.R` 中，我使用基于检验统计量 $\hat{\tau}$ 的**随机化检验（FRT）**分析了达尔文的数据。

- 使用带有**威尔科克森符号秩和统计量（Wilcoxon signed rank sum statistic）**的 FRT 重新分析此数据集。
- 基于内曼推断重新分析此数据集：无偏点估计量、保守方差估计量、95% 置信区间。

## 7.7 重新分析儿童电视工作室实验数据（Re-analyzing children’s television workshop experiment data）

在 `MPENeymanstar.R` 中，我基于内曼推断分析了数据。

- 使用带有不同检验统计量的 FRT 重新分析此数据集。
- 使用带有协变量调整的 FRT 重新分析此数据集，例如，您可以基于观测结果对协变量进行 OLS 拟合后的残差来定义检验统计量。如果您的 OLS 拟合中不包含截距项，结论会改变吗？

## 7.8 重新分析 Angrist 和 Lavy (2009) 的数据（Re-analyzing Angrist and Lavy (2009)’s data）

原始分析相当复杂。对于本题，请仅关注原始论文的表 A1，将学校视为实验单元。Angrist 和 Lavy (2009) 本质上对学校进行了一个 MPE。去掉第 6 对以及所有存在不依从性的对，得到 14 个完整的对，数据如下所示，也包含在 `AL2009.csv` 中：

<table><tr><td></td><td>pair</td><td>z</td><td>pr99</td><td>pr00</td><td>pr01</td><td>pr02</td></tr><tr><td>1</td><td>1</td><td>0</td><td>0.046</td><td>0.000</td><td>0.091</td><td>0.185</td></tr><tr><td>2</td><td>1</td><td>1</td><td>0.036</td><td>0.051</td><td>0.000</td><td>0.047</td></tr><tr><td>3</td><td>2</td><td>0</td><td>0.054</td><td>0.094</td><td>0.184</td><td>0.034</td></tr><tr><td>4</td><td>2</td><td>1</td><td>0.050</td><td>0.108</td><td>0.110</td><td>0.095</td></tr><tr><td>5</td><td>3</td><td>0</td><td>0.114</td><td>0.000</td><td>0.056</td><td>0.075</td></tr><tr><td>6</td><td>3</td><td>1</td><td>0.098</td><td>0.054</td><td>0.030</td><td>0.068</td></tr><tr><td>7</td><td>4</td><td>0</td><td>0.148</td><td>0.162</td><td>0.082</td><td>0.075</td></tr><tr><td>8</td><td>4</td><td>1</td><td>0.134</td><td>0.390</td><td>0.339</td><td>0.458</td></tr><tr><td>9</td><td>5</td><td>0</td><td>0.152</td><td>0.105</td><td>0.083</td><td>0.129</td></tr><tr><td>10</td><td>5</td><td>1</td><td>0.145</td><td>0.077</td><td>0.579</td><td>0.167</td></tr><tr><td>11</td><td>6</td><td>0</td><td>0.188</td><td>0.214</td><td>0.375</td><td>0.545</td></tr><tr><td>12</td><td>6</td><td>1</td><td>0.179</td><td>0.165</td><td>0.483</td><td>0.444</td></tr><tr><td>13</td><td>7</td><td>0</td><td>0.193</td><td>0.771</td><td>0.328</td><td>0.583</td></tr><tr><td>14</td><td>7</td><td>1</td><td>0.189</td><td>0.186</td><td>0.168</td><td>0.368</td></tr><tr><td>15</td><td>8</td><td>0</td><td>0.197</td><td>0.350</td><td>0.000</td><td>0.383</td></tr><tr><td>16</td><td>8</td><td>1</td><td>0.200</td><td>0.071</td><td>0.667</td><td>0.429</td></tr><tr><td>17</td><td>9</td><td>0</td><td>0.213</td><td>0.176</td><td>0.164</td><td>0.172</td></tr><tr><td>18</td><td>9</td><td>1</td><td>0.209</td><td>0.165</td><td>0.092</td><td>0.151</td></tr><tr><td>19</td><td>10</td><td>0</td><td>0.211</td><td>0.667</td><td>0.250</td><td>0.617</td></tr><tr><td>20</td><td>10</td><td>1</td><td>0.219</td><td>0.250</td><td>0.500</td><td>0.350</td></tr><tr><td>21</td><td>11</td><td>0</td><td>0.219</td><td>0.153</td><td>0.185</td><td>0.219</td></tr><tr><td>22</td><td>11</td><td>1</td><td>0.224</td><td>0.363</td><td>0.372</td><td>0.342</td></tr><tr><td>23</td><td>12</td><td>0</td><td>0.255</td><td>0.226</td><td>0.213</td><td>0.327</td></tr><tr><td>24</td><td>12</td><td>1</td><td>0.257</td><td>0.098</td><td>0.107</td><td>0.095</td></tr><tr><td>25</td><td>13</td><td>0</td><td>0.261</td><td>0.071</td><td>0.000</td><td>NA</td></tr><tr><td>26</td><td>13</td><td>1</td><td>0.263</td><td>0.441</td><td>0.448</td><td>0.435</td></tr><tr><td>27</td><td>14</td><td>0</td><td>0.286</td><td>0.161</td><td>0.126</td><td>0.181</td></tr><tr><td>28</td><td>14</td><td>1</td><td>0.285</td><td>0.389</td><td>0.353</td><td>0.309</td></tr></table>

结果变量是 2001 年和 2002 年的 Bagrut 通过率，以 1999 年和 2000 年的 Bagrut 通过率作为预处理协变量。基于有和没有协变量的内曼推断重新分析这些数据。特别地，您如何处理第 25 对中的缺失结果？

## 7.9 一般匹配实验中的方差估计（Variance estimation in the general matched experiment）

本题包含**第 7.7 节（Section 7.7）**的更多细节。

- 首先，对于一般匹配实验证明**定理 7.1（Theorem 7.1）**。
- 其次，证明**定理 7.3（Theorem 7.3）**。

提示：对于第二部分，我们需要首先验证 $\hat { \tau } _ { i } - \hat { \tau } _ { w }$ 的均值为 $\tau _ { i } - \tau _ { w }$，方差为

$$
\operatorname{var} \left(\hat {\tau} _ {i} - \hat {\tau} _ {w}\right) = \operatorname{var} \left(\hat {\tau} _ {w}\right) + (1 - 2 w _ {i}) \operatorname{var} \left(\hat {\tau} _ {i}\right).
$$

## 7.10 推荐阅读（Recommended readings）

Greevy 等人 (2004) 提供了一种基于协变量形成匹配对的算法。Imai (2008b) 讨论了无协变量时平均因果效应的估计，Fogarty (2018b) 讨论了 MPE 中的协变量调整。
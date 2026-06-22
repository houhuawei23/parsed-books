# 完全随机实验与费希尔随机化检验（The Completely Randomized Experiment and the Fisher Randomization Test）

**潜在结果框架（Potential outcomes framework）** 与随机化实验具有内在联系。理解各类随机化实验中的因果推断是基础性的，对于理解更复杂的非实验研究中的因果推断也相当有帮助。

本书第二部分聚焦于随机化实验。本章关注最简单的实验——**完全随机实验（Completely Randomized Experiment, CRE）**。

## 3.1 完全随机实验（CRE）

考虑一个包含 $n$ 个单元的实验，其中 $n_{1}$ 个单元接受处理，$n_{0}$ 个单元接受对照。我们可以基于其**处理分配机制（treatment assignment mechanism）** 来定义 CRE $^{1}$ 。

<!-- footnote -->

> - $^{1}$ 读者可能会认为 CRE 中的 $Z_{i}$ 是独立同分布（IID）的伯努利随机变量，概率为 $\pi$ ，此时 $n_{1}$ 是一个二项分布 Binomial( $n,\pi$ ) 随机变量。这被称为**伯努利随机化实验（Bernoulli Randomized Experiment, BRE）**，如果我们以 $(n_{1},n_{0})$ 为条件，则 BRE 退化为 CRE。我将在第 4 章的问题 4.7 中给出更多关于 BRE 的细节。

<!-- footnote end -->

**定义 3.1（CRE）** CRE 具有以下处理分配机制：

$$
\operatorname{pr} (\mathbf {Z} = \mathbf {z}) = 1 \bigg / \binom{n}{n _ {1}},
$$

其中 $\boldsymbol{z} = (z_1, \ldots, z_n)$ 满足 $\sum_{i=1}^{n} z_i = n_1$ 且 $\sum_{i=1}^{n} (1 - z_i) = n_0$ 。

在定义 3.1 中，我们将处理下的**潜在结果向量（potential outcome vector）** $\mathbf{Y}(1) = (Y_{1}(1), \ldots, Y_{n}(1))$ 和对照下的潜在结果向量 $\mathbf{Y}(0) = (Y_{1}(0), \ldots, Y_{n}(0))$ 均视为固定的。即使我们将它们视为随机的，也可以以其为条件，此时处理分配机制变为：

$$
\operatorname{pr} \{\boldsymbol {Z} = \boldsymbol {z} \mid \boldsymbol {Y} (1), \boldsymbol {Y} (0) \} = 1 \bigg / \binom{n}{n _ {1}}
$$

因为在 CRE 中， $\mathbf{Z} \perp \{\mathbf{Y}(1), \mathbf{Y}(0)\}$ 。在 CRE 中，处理向量 $\mathbf{Z}$ 来自 $n_1$ 个 1 和 $n_0$ 个 0 的随机排列。

Fisher（1935）在其开创性著作《实验设计》（Design of Experiments）中指出了随机化的以下优点：

1. 它平均而言创造了可比较的处理组和对照组。
2. 它为统计推断提供了"合理的基础"。

第一点很直观，因为随机处理分配不会偏向处理组或对照组。大多数人都能很好地理解第一点。第二点则更为微妙。Fisher 的意思是随机化为统计检验提供了依据，这种检验现在被称为**费希尔随机化检验（Fisher Randomization Test, FRT）**。本章将阐述 CRE 下 FRT 的基本思想。

## 3.2 FRT

Fisher（1935）关注于检验以下**零假设（null hypothesis）**：

$$
H _ {0 \mathrm{F}}: Y _ {i} (1) = Y _ {i} (0) \text {  对于  所有  单元   } i = 1, \dots , n.
$$

Rubin（1980）将其称为**尖锐零假设（sharp null hypothesis）**，其含义是它可以根据观测数据确定所有潜在结果： $\mathbf{Y}(1)=\mathbf{Y}(0)=\mathbf{Y}=(Y_{1},\ldots,Y_{n})$ ，即观测结果向量。它也被称为**强零假设（strong null hypothesis）**（例如，Wu and Ding, 2021）。

从概念上讲，在 $H_{0F}$ 下，FRT 适用于任何**检验统计量（test statistic）**

$$
T = T (\mathbf {Z}, \mathbf {Y}), \tag {3.1}
$$

该统计量是观测数据的函数。在 $H_{0F}$ 下，观测结果向量 $\mathbf{Y}$ 是固定的，因此检验统计量 $T$ 中唯一的随机成分是处理向量 $\mathbf{Z}$ 。实验者决定了 $\mathbf{Z}$ 的分布，这反过来又决定了 $H_{0F}$ 下 $T$ 的分布。这是计算 **p 值（p-value）** 的基础。下面我将提供更多细节。

在 CRE 中， $\mathbf{Z}$ 在以下集合上服从均匀分布：

$$
\left\{\boldsymbol {z} ^ {1}, \dots , \boldsymbol {z} ^ {M} \right\}
$$

其中 $M = \binom{n}{n_{1}}$ ，且 $z^{m}$ 是所有包含 $n_{1}$ 个 1 和 $n_{0}$ 个 0 的可能向量。例如，当 $n = 5$ 且 $n_{1} = 3$ 时，我们可以枚举 $M = \binom{5}{3} = 10$ 个向量如下：

```txt
> permutation10 = function(n, n1){
+ M = choose(n, n1)
+ treat.index = combn(n, n1)
+ Z = matrix(0, n, M)
```

## 3.2 FRT

+ for(m in 1:M){
+ treat = treat.index[, m]
+ Z[treat, m] = 1
+ }
+ Z
+ }
>
> permutation10(5, 3)
[ ,1] [ ,2] [ ,3] [ ,4] [ ,5] [ ,6] [ ,7] [ ,8] [ ,9] [ ,10]
[1, ] 1 1 1 1 1 1 0 0 0 0
[2, ] 1 1 1 0 0 0 1 1 1 0
[3, ] 1 0 0 1 1 0 1 1 0 1
[4, ] 0 1 0 1 0 1 1 0 1 1
[5, ] 0 0 1 0 1 1 0 1 1 1

因此， $T$ 在以下集合（可能有重复）上服从均匀分布：

$$
\{T (\boldsymbol {z} ^ {1}, \boldsymbol {Y}), \dots , T (\boldsymbol {z} ^ {M}, \boldsymbol {Y}) \}.
$$

也就是说，由于 CRE 的设计， $T$ 的分布是已知的。我们将 $T$ 的这个分布称为**随机化分布（randomization distribution）**。

如果 $T$ 的值越大越极端，我们可以使用以下尾部概率来衡量检验统计量相对于其随机化分布的极端程度：

$$
p _ {\mathrm{FRT}} = M ^ {- 1} \sum_ {m = 1} ^ {M} I \{T (\boldsymbol {z} ^ {m}, \boldsymbol {Y}) \geq T (\boldsymbol {Z}, \boldsymbol {Y}) \}, \tag {3.2}
$$

Fisher 将其称为 p 值。图 3.1 展示了 $p_{FRT}$ 的计算过程。

![image_02](images/image_02.png)

```mermaid
graph TD
  A["(Z, Y) ⇒ T(Z, Y)"] --> B["(z¹, Y) ⇒ T(z¹, Y)"]
  A --> C["(z², Y) ⇒ T(z², Y)"]
  A --> D["..."]
  A --> E["(zᴹ, Y) ⇒ T(zᴹ, Y)"]
  B --> F[p_FRT = M⁻¹ Σ_{m=1}^M I{T(zᵐ, Y) ≥ T(Z, Y)}
  C --> F
  D --> F
  E --> F
```

图 3.1：FRT 示意图

（3.2）中的 p 值 $p_{FRT}$ 适用于任何检验统计量的选择和任何结果生成过程。它也可以自然地推广到任何实验，这将是后续章节中反复讨论的主题。重要的是，它在有限样本下是精确的 $^{2}$ ，即在 $H_{0F}$ 下，

$$
\operatorname{pr} (p _ {\mathrm{FRT}} \leq u) \leq u \quad \text { 对于  所有 } \quad 0 \leq u \leq 1. \tag {3.3}
$$

在实践中， $M$ 通常非常大（例如，当 $n = 100, n_{1} = 50$ 时， $M > 10^{29}$ ），枚举处理向量的所有可能值在计算上是不可行的。我们通常通过**蒙特卡洛方法（Monte Carlo）** 来近似 $p_{FRT}$ 。具体来说，我们从处理向量的可能值中进行简单随机抽样，或者等价地，对 $\mathbf{Z}$ 进行随机排列，并通过以下方式近似 $p_{FRT}$ ：

$$
\hat {p} _ {\mathrm{FRT}} = R ^ {- 1} \sum_ {r = 1} ^ {R} I \{T (\boldsymbol {z} ^ {r}, \boldsymbol {Y}) \geq T (\boldsymbol {Z}, \boldsymbol {Y}) \}, \tag {3.4}
$$

其中 $z^r$ 是 $\mathbf{Z}$ 的 $R$ 个随机排列。随着 $R$ 的增加，（3.4）中的 p 值的蒙特卡洛误差会迅速减小；参见问题 3.2。由于（3.4）中 p 值的计算涉及 $\mathbf{Z}$ 的排列，因此在 CRE 的背景下，FRT 有时被称为**置换检验（permutation test）**。然而，在更复杂的实验中，FRT 的思想比置换检验更为广泛。

> - $^{2}$ 实际上，在 $H_{0F}$ 下， $p_{FRT}$ 在离散均匀分布上具有精确的超均匀性质。

## 3.3 检验统计量的规范选择（Canonical choices of the test statistic）

从上述讨论可知，对于任意选择的检验统计量，**费舍尔随机化检验（Fisher Randomization Test, FRT）** 都能生成有限样本精确 p 值。这是 FRT 的一个特性。然而，这一特性不应鼓励对检验统计量进行随意选择。直观上，我们必须选择那些能够为可能违反 $H_{0F}$ 的情况提供信息的检验统计量。下面我将回顾一些规范的选择。

**例 3.1（均值差，difference-in-means）** 均值差统计量为：

$$
\hat {\tau} = \hat {\bar {Y}} (1) - \hat {\bar {Y}} (0)
$$

其中

$$
\hat {\bar {Y}} (1) = n _ {1} ^ {- 1} \sum_ {Z _ {i} = 1} Y _ {i} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i}
$$

是处理组结果的样本均值，而

$$
\hat {\bar {Y}} (0) = n _ {0} ^ {- 1} \sum_ {Z _ {i} = 0} Y _ {i} = n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) Y _ {i}
$$

是对照组结果的样本均值。在 $H_{0F}$ 下，其均值为

$$
E (\hat {\tau}) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} E (Z _ {i}) Y _ {i} - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} E (1 - Z _ {i}) Y _ {i} = 0
$$

方差为

$$
\begin{array}{l} \operatorname{var} (\hat {\tau}) = \operatorname{var} \left\{n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i} - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) Y _ {i} \right\} \\ = \quad \operatorname{var} \left(\frac {n}{n _ {0}} \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i}\right) \\ = _ {*} \frac {n ^ {2}}{n _ {0} ^ {2}} \left(1 - \frac {n _ {1}}{n}\right) \frac {s ^ {2}}{n _ {1}} \\ { = } { \frac { n } { n _ { 1 } n _ { 0 } } s ^ { 2 } , } \\ \end{array}
$$

其中 $= _{*}$ 来自针对简单随机抽样的引理 A3.2，且

$$
\bar {Y} = n ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i}, \quad s ^ {2} = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} (Y _ {i} - \bar {Y}) ^ {2}.
$$

此外，由于引理 A3.4 中的有限总体中心极限定理，$\hat{\tau}$ 的随机化分布近似于正态分布：

$$
\frac {\hat {\tau}}{\sqrt {\frac {n}{n _ {1} n _ {0}} s ^ {2}}} \rightarrow \mathrm{N} (0, 1) \tag {3.5}
$$

依分布收敛。由于 $s^{2}$ 在 $H_{0F}$ 下是固定的，因此在 FRT 中使用

$$
\frac {\hat {\tau}}{\sqrt {\frac {n}{n _ {1} n _ {0}} s ^ {2}}}
$$

作为检验统计量是等价的，如上所示，该统计量是渐近正态的。然后我们可以计算一个近似的 p 值。

观测数据是 $\{Y_i:Z_i = 1\}$ 和 $\{Y_i:Z_i = 0\}$ ，因此问题本质上是一个两样本问题。在 **独立同分布（Independent and Identically Distributed, IID）** 正态结果的假设下（见 A1.4.1 节），经典的假设方差相等的两样本 $t$ 检验基于：

$$
\frac {\hat {\tau}}{\sqrt {\frac {n}{n _ {1} n _ {0} (n - 2)} \left[ \sum_ {Z _ {i} = 1} \{Y _ {i} - \hat {\bar {Y}} (1) \} ^ {2} + \sum_ {Z _ {i} = 0} \{Y _ {i} - \hat {\bar {Y}} (0) \} ^ {2} \right]}} \sim t _ {n - 2}. \tag {3.6}
$$

基于一些代数运算（见问题 3.8），我们得到展开式：

$$
(n - 1) s ^ {2} = \sum_ {Z _ {i} = 1} \left\{Y _ {i} - \hat {\bar {Y}} (1) \right\} ^ {2} + \sum_ {Z _ {i} = 0} \left\{Y _ {i} - \hat {\bar {Y}} (0) \right\} ^ {2} + \frac {n _ {1} n _ {0}}{n} \hat {\tau} ^ {2}. \tag {3.7}
$$

当样本量 n 很大时，我们可以忽略 $N(0,1)$ 与 $t_{n-2}$ 之间的差异，以及 n-1 与 n-2 之间的差异。此外，在 $H_{0F}$ 下，$\hat{\tau}$ 依概率收敛到 0，因此 $n_{1}n_{0}/n\hat{\tau}^{2}$ 在渐近意义上可以忽略。因此，在 $H_{0F}$ 下，例 3.1 中的近似 p 值接近于经典假设方差相等的两样本 t 检验的 p 值，该 p 值可通过 `t.test` 函数并设置 `var.equal = TRUE` 计算得到。在具有非零 $\tau$ 的备择假设下，上述展开式中的额外项 $\frac{n_{1}n_{0}}{n}\hat{\tau}^{2}$ 可能使得 FRT 的检验功效低于通常的 t 检验。

基于上述讨论，使用 $\hat{\tau}$ 的 FRT 实际上使用了忽略两组间异方差的合并方差。在经典统计学中，具有异方差正态结果的两样本问题被称为 **贝伦斯-费雪问题（Behrens–Fisher problem）**（见 A1.4.1 节）。在贝伦斯-费雪问题中，一个标准的检验统计量选择是下面的学生化统计量。

**例 3.2（学生化统计量，studentized statistic）** 学生化统计量为：

$$
t _ {\mathrm{unequal}} = \frac {\hat {\bar {Y}} (1) - \hat {\bar {Y}} (0)}{\sqrt {\frac {\hat {S} ^ {2} (1)}{n _ {1}} + \frac {\hat {S} ^ {2} (0)}{n _ {0}}}},
$$

其中

$$
\hat {S} ^ {2} (1) = (n _ {1} - 1) ^ {- 1} \sum_ {Z _ {i} = 1} \{Y _ {i} - \hat {\bar {Y}} (1) \} ^ {2}, \quad \hat {S} ^ {2} (0) = (n _ {0} - 1) ^ {- 1} \sum_ {Z _ {i} = 0} \{Y _ {i} - \hat {\bar {Y}} (0) \} ^ {2}
$$

分别是处理组和对照组观测结果的样本方差。在 $H_{0F}$ 下，有限总体中心极限定理再次表明 t 是渐近正态的：

$$
t \to \mathrm{N} (0, 1)
$$

依分布收敛。然后我们可以计算一个近似的 p 值，该值接近于 `t.test` 函数设置 `var.equal = FALSE` 时的 p 值。

一个极其重要的点是，即使底层分布不是正态的，FRT 也证明了使用 `t.test`（无论设置 `var.equal = TRUE` 还是 `var.equal = FALSE`）的传统 t 检验的合理性。标准的统计学教科书通常基于正态性假设来推导 t 检验，但这个假设过于严格。幸运的是，只要有限总体中心极限定理成立，t 检验程序仍然可以使用。即使我们不相信中心极限定理，我们仍然可以在 FRT 中使用 $\hat{\tau}$ 和 t 作为检验统计量，以获得有限样本精确 p 值。

我们将在第 8 章从另一个角度来推导这个学生化统计量。理论表明，在 FRT 中使用 $t$ 对两组间的异方差性更具稳健性。

下面的检验统计量对于由重尾结果数据导致的异常值具有稳健性。

**例 3.3（威尔科克森秩和统计量，Wilcoxon rank sum）** 均值差统计量使用原始结果，其抽样分布依赖于结果的二阶矩。这使得它对异常值敏感。另一个流行的检验统计量基于合并观测结果的秩。令 $R_{i}$ 表示 $Y_{i}$ 在合并样本 Y 中的秩：

$$
R _ {i} = \# \{j: Y _ {j} \leq Y _ {i} \}.
$$

**威尔科克森秩和统计量（Wilcoxon rank sum statistic）** 是处理组秩的总和：

$$
W = \sum_ {i = 1} ^ {n} Z _ {i} R _ {i}.
$$

为了代数上的简洁，我们假设结果中没有结（ties），尽管无论是否存在结，FRT 都可以应用。对于有结的情况，请参见 Lehmann (1975, 第 1 章第 4 节)。由于合并样本的秩总和固定为 $1 + 2 + \cdots + n = n(n + 1)/2$ ，因此威尔科克森统计量等价于处理组与对照组秩均值的差异。在 $H_{0F}$ 下，$R_{i}$ 是固定的，因此 W 的均值为

$$
E (W) = \sum_ {i = 1} ^ {n} E (Z _ {i}) R _ {i} = \frac {n _ {1}}{n} \sum_ {i = 1} ^ {n} i = \frac {n _ {1}}{n} \times \frac {n (n + 1)}{2} = \frac {n _ {1} (n + 1)}{2}
$$

方差为

$$
\begin{array}{l} \operatorname{var} (W) = \operatorname{var} \left(n _ {1} \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} R _ {i}\right) \\ = _ {*} n _ {1} ^ {2} \left(1 - \frac {n _ {1}}{n}\right) \frac {1}{n _ {1}} \frac {1}{n - 1} \sum_ {i = 1} ^ {n} \left(R _ {i} - \frac {n + 1}{2}\right) ^ {2} \\ = \frac {n _ {1} n _ {0}}{n (n - 1)} \sum_ {i = 1} ^ {n} \left(i - \frac {n + 1}{2}\right) ^ {2} \\ = \frac {n _ {1} n _ {0}}{n (n - 1)} \left\{\sum_ {i = 1} ^ {n} i ^ {2} - n \left(\frac {n + 1}{2}\right) ^ {2} \right\} \\ = \frac {n _ {1} n _ {0}}{n (n - 1)} \left\{\frac {n (n + 1) (2 n + 1)}{6} - n \left(\frac {n + 1}{2}\right) ^ {2} \right\} \\ = \frac {n _ {1} n _ {0} (n + 1)}{1 2}, \\ \end{array}
$$

其中 $=_{*}$ 来自引理 A3.2。此外，在 $H_{0\mathrm{F}}$ 下，有限总体中心极限定理确保了 $\widehat{\tau}$ 的随机化分布近似正态：

$$
\frac {\sum_ {i = 1} ^ {n} Z _ {i} R _ {i} - \frac {n _ {1} (n + 1)}{2}}{\sqrt {\frac {n _ {1} n _ {0} (n + 1)}{1 2}}} \rightarrow \mathrm{N} (0, 1) \tag {3.8}
$$

依分布收敛。基于 (3.8)，我们可以进行渐近检验。在 R 中，函数 `wilcox.test` 可以基于统计量 $W - n_{1}(n_{1} + 1)/2$ 计算精确 p 值和渐近 p 值。基于一些渐近分析，Lehmann (1975) 表明，使用 W 的 FRT 在广泛的数据生成过程中具有合理的检验功效。

**例 3.4（柯尔莫哥洛夫-斯米尔诺夫统计量，Kolmogorov–Smirnov statistic）** 处理可能以不同的方式影响结果。基于经验分布来总结处理组结果和对照组结果似乎很自然：

$$
\hat {F} _ {1} (y) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} I (Y _ {i} \leq y), \quad \hat {F} _ {0} (y) = n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) I (Y _ {i} \leq y).
$$

比较这两个经验分布得到了著名的 **柯尔莫哥洛夫-斯米尔诺夫统计量（Kolmogorov-Smirnov statistic）**

$$
D = \max _ {y} \left| \hat {F} _ {1} (y) - \hat {F} _ {0} (y) \right|.
$$

推导 $D$ 的分布是一个具有挑战性的数学问题。当样本量很大时，其分布函数收敛于

$$
\mathrm{pr} \left(\frac {n _ {1} n _ {0}}{n} D \leq x\right)\rightarrow \frac {\sqrt {2 \pi}}{x} \sum_ {j = 1} ^ {\infty} e ^ {- (2 j - 1) ^ {2} \pi^ {2} / (8 x ^ {2})},
$$

基于此我们可以计算渐近 p 值 (Van der Vaart, 2000)。在 R 中，`ks.test` 可以计算精确 p 值和渐近 p 值。

## 3.4 LaLonde 实验数据案例研究（A case study of the LaLonde experimental data）

我使用 LaLonde (1986) 的实验数据来说明 FRT。这些数据可在 `Matching` 包中获得 (Sekhon, 2011)：

图 3.2 显示了处理组和对照组结果的直方图。

```txt
> library (Matching)
> data (lalonde)
> z = lalonde$treat
> y = lalonde$re78
```

以下代码使用现有函数计算检验统计量的观测值：

```txt
> tauhat = t.test(y[z == 1], y[z == 0],
+    var.equal = TRUE)$statistic
> tauhat
t
2.835321
> student = t.test(y[z == 1], y[z == 0],
+    var.equal = FALSE)$statistic
> student
t
2.674146
> W = wilcox.test(y[z == 1], y[z == 0])$statistic
> W
W
27402.5
> D = ks.test(y[z == 1], y[z == 0])$statistic
> D
D
0.1321206
```

通过随机置换处理向量，我们可以获得检验统计量随机化分布的蒙特卡洛近似，并存储在四个向量 `Tauhat`、`Student`、`Wilcox` 和 `Ks` 中。

```diff
> MC = 10^4
> Tauhat = rep(0, MC)
> Student = rep(0, MC)
> Wilcox = rep(0, MC)
> Ks = rep(0, MC)
> for(mc in 1:MC)
+ {
+    zperm = sample(z)
+    Tauhat[mc] = t.test(y[zperm == 1], y[zperm == 0],
+    var.equal = TRUE)$statistic
+    Student[mc] = t.test(y[zperm == 1], y[zperm == 0],
+    var.equal = FALSE)$statistic
+    Wilcox[mc] = wilcox.test(y[zperm == 1], y[zperm == 0])$statistic
+    Ks[mc] = ks.test(y[zperm == 1], y[zperm == 0])$statistic
+ }
```

基于 FRT 的单侧 p 值均小于 0.05：

```txt
> exact.pv = c(mean(Tauhat >= tauhat),
+    mean(Student >= student),
+    mean(Wilcox >= W),
+    mean(Ks >= D))
> round(exact.pv, 3)
[1] 0.002 0.002 0.006 0.040
```

在不使用蒙特卡洛方法的情况下，我们也可以计算渐近 p 值，这些值均小于 0.05：

```txt
> asym.pv = c(t.test(y[z == 1], y[z == 0],
+    var.equal = TRUE)$p.value,
+    t.test(y[z == 1], y[z == 0],
+    var.equal = FALSE)$p.value,
+    wilcox.test(y[z == 1], y[z == 0])$p.value,
+    ks.test(y[z == 1], y[z == 0])$p.value)
> round(asym.pv, 3)
[1] 0.005 0.008 0.011 0.046
```

p 值之间的差异源于渐近近似，以及 `t.test` 和 `wilcox.test` 的默认选项是双侧检验这一事实。

图 3.3 显示了四个检验统计量的随机化分布直方图，以及它们对应的观测值。对于前三个检验统计量，即使底层结果数据分布远非正态，正态近似也表现得非常好。通常，像图 3.3 这样的图表可以为检验点零假设提供非常清晰的信息。最近，Bind 和 Rubin (2020) 在其论文标题中提出，“如果可能，报告费舍尔精确 p 值并展示其底层的零随机化分布”。

## 3.5 随机化实验与 FRT 的一些历史（Some history of randomized experiments and FRT）

## 3.5.1 詹姆斯·林德的实验（James Lind’s experiment）

**詹姆斯·林德（James Lind，1716—1794）**是一位苏格兰医生，也是皇家海军（Royal Navy）海军卫生学的先驱。在他所处的时代，**坏血病（scurvy）**是水手死亡的主要原因之一。他进行了一项有清晰记录细节的最早的随机实验之一，并在维生素C（Vitamin C）被发现之前得出结论：柑橘类水果可以治愈坏血病。

在林德（1753）的著作中，他描述了以下随机实验：将12名坏血病患者分配到6个组。经过一些简化，这6个组是：

1.  两人每天喝一夸脱苹果酒；
2.  两人每天服用二十五滴硫酸，每日三次；
3.  两人每天服用两汤匙醋，每日三次；
4.  两人每天喝半品脱海水；
5.  两人每天吃两个橙子和一个柠檬；
6.  两人食用一种辛辣糊状物并喝大麦水。

六天后，第五组的患者康复了，而其他组的患者则没有。如果我们将处理简化为

$$
Z _ {i} = 1 (\text { 单位 } i \text { 接受了柑橘类水果 })
$$

并将结果简化为

$$
Y _ {i} = 1 (\text { 单位 } i \text { 在六天后康复 }),
$$

那么我们就得到一个 $2 \times 2$ 的表格

<table><tr><td></td><td> $Y_i = 1$ </td><td> $Y_i = 0$ </td></tr><tr><td> $Z_i = 1$ </td><td>2</td><td>0</td></tr><tr><td> $Z_i = 0$ </td><td>0</td><td>10</td></tr></table>

这是在此实验下我们能观察到的最极端的 $2 \times 2$ 表格，并且这些数据为柑橘类水果治愈坏血病的正面效果提供了强有力的证据。从统计学上讲，我们如何衡量这个证据的强度呢？

遵循**费雪随机化检验（Fisher Randomization Test, FRT）**的逻辑，如果处理完全没有效果（在 $H _ { \mathrm { 0F } }$ 下），那么出现这个极端的 $2 \times 2$ 表格的概率为

$$
\frac {1}{\binom {12} {2}} = \frac {1}{66} = 0.015
$$

这就是 $p _ { \mathrm {FRT} }$ 。这在 $H _ { \mathrm {0F} }$ 下似乎是一个意外：我们可以在 0.05 的水平上轻松拒绝 $H _ { \mathrm {0F} }$ 。

## 3.5.2 品茶女士（Lady tasting tea）

**费雪（Fisher，1935）**描述了以下著名的“品茶女士”实验。一位女士声称她能够区分两种泡奶茶的方式：一种是先加牛奶，另一种是先加茶。这对大多数人来说可能听起来很奇怪。作为一名统计学家，费雪设计了一个实验来检验这位女士是否能区分这两种泡茶方式。

他制作了8杯茶，其中4杯是先加牛奶，另外4杯是先加茶。然后，他以随机顺序将这8杯茶呈现给这位女士，并要求她挑出那4杯先加牛奶的茶。最终的实验结果可以总结在下面的 $2 \times 2$ 表格中

<table><tr><td></td><td>milk first (lady)</td><td>tea first (lady)</td><td>column sum</td></tr><tr><td>milk first (Fisher)</td><td>X</td><td>4 - X</td><td>4</td></tr><tr><td>tea first (Fisher)</td><td>4 - X</td><td>X</td><td>4</td></tr><tr><td>row sum</td><td>4</td><td>4</td><td>8</td></tr></table>

X 可以是 0、1、2、3、4。在真实实验中，${ \overline { { X = 4 } } }$ ，这是最极端的数据，强烈表明这位女士能够区分这两种泡奶茶的方式。同样，我们如何衡量这个证据的强度呢？

在**零假设（null hypothesis）**——这位女士无法区分——下，在 $\binom {8} {4} = 70$ 种可能的顺序中，只有一种顺序能产生 $X = 4$ 的 $2 \times 2$ 表格。所以 p 值是

$$
p _ {\mathrm{FRT}} = \frac {1}{70} = 0.014.
$$

给定显著性水平 0.05，我们拒绝零假设。

## 3.5.3 实验的两个费雪原则（Two Fisherian principles for experiments）

在上述第 3.5.1 节和第 3.5.2 节的两个例子中，$p _ { \mathrm {FRT} }$ 是通过实验的**随机化（randomization）**来证明其合理性的。这突出了实验的第一个费雪原则：**随机化**。

此外，上述两个实验在某种意义上是可以产生统计上显著结果的最小可能实验。例如，如果林德只给六个组中的每组分配一名患者，那么最小的 p 值是

$$
\frac {1}{\binom {6} {1}} = \frac {1}{6} = 0.167;
$$

如果费雪只做了6杯茶，其中3杯先加牛奶，另外3杯先加茶，那么最小的 p 值是

$$
\frac {1}{\binom {6} {3}} = \frac {1}{20} = 0.05.
$$

我们永远无法在 0.05 的水平上拒绝零假设。这突出了实验的第二个费雪原则：**重复（replications）**。

第5章将讨论实验的第三个费雪原则：**区组化（blocking）**。

## 3.6 讨论（Discussion）

## 3.6.1 其他尖锐零假设和置信区间（Other sharp null hypotheses and confidence intervals）

我上面重点讨论了**尖锐零假设（sharp null hypothesis）** $H _ { \mathrm {0F} }$ 。事实上，FRT 的逻辑也适用于其他尖锐零假设。例如，我们可以检验

$$
H _ {0} (\pmb {\tau}): Y _ {i} (1) - Y _ {i} (0) = \tau_ {i} \text { 对于所有 } i = 1, \ldots , n
$$

对于一个已知的向量 $\tau = ( \tau _ { 1 } , \dots , \tau _ { n } )$ 。因为在 $H _ { 0 } ( \tau )$ 下所有个体因果效应都是已知的，我们可以根据观测数据估算所有缺失的**潜在结果（potential outcomes）**。有了已知的潜在结果，任何检验统计量的分布完全由处理分配机制决定，因此，我们可以计算相应的 $p _ { \mathrm { {FRT} } }$ 作为 $\tau$ 的函数，记为 $p _ { \mathrm {FRT} } ( \tau )$ 。如果我们能指定所有可能的 ${ \boldsymbol { \tau } }$ ，那么我们可以计算一系列 $p _ { \mathrm {FRT} } ( \tau )$ 。通过假设检验和置信集的对偶性（见附录 A1.2.5），我们可以获得**平均因果效应（average causal effect）**的 (1 − α) 水平置信集：

$$
\left\{\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \tau_ {i}: p _ {\mathrm{FRT}} (\pmb {\tau}) \geq \alpha \right\}.
$$

尽管这个策略在概念上很直接，但由于所有可能的 $\tau$ 数量巨大，它在实践中很复杂。在结果变量为二值变量的特殊情况下，Rigdon 和 Hudgens (2015) 以及 Li 和 Ding (2016) 提出了一些计算上可行的方法，基于 FRT 构建 τ 的置信区间。对于一般的无界结果变量，这种策略通常在计算上是不可行的。

一个经典的简化是考虑具有恒定个体因果效应的尖锐零假设的子类：

$$
H _ {0} (c): Y _ {i} (1) - Y _ {i} (0) = c \text { 对于所有 } i = 1, \ldots , n
$$

对于一个已知常数 c。给定 c，我们可以计算 $p _ { \mathrm {FRT} } ( c )$ 。通过对偶性，我们可以获得平均因果效应的 $( 1 - \alpha )$ 水平置信集：

$$
\{c: p _ {\mathrm{FRT}} (c) \geq \alpha \}.
$$

因为这个过程只涉及一维搜索，它在计算上是可行的。然而，它经常被批评为恒定个体因果效应假设过于强烈，特别是对于二值结果变量不成立。

## 3.6.2 其他检验统计量（Other test statistics）

FRT 是一种通用策略，适用于任何随机实验和任何检验统计量。我在第 3.3 节中给出了几个检验统计量的例子。事实上，检验统计量的定义可以更加通用。例如，对于处理前协变量矩阵 X，其第 i 行是单位 i 的 $X _ { i }$ $( i = 1 , \ldots , n )$ ，我们可以允许检验统计量 $T ( Z , Y , X )$ 是处理向量、结果向量和协变量矩阵的函数。习题 3.6 给出了一个例子。

## 3.6.3 最后评论（Final remarks）

对于一个一般的实验，$z$ 的概率分布在 $n _ { 1 }$ 个 1 和 $n _ { 0 }$ 个 0 的所有可能排列上并非均匀分布。但实验者完全知道其分布。因此，我们总是可以模拟它的分布，这反过来意味着在尖锐零假设下任何检验统计量的分布。一个有限样本精确 p 值由 (3.2) 给出。我将在后续章节中讨论其他实验，并且我想强调，FRT 的应用范围超出了本书中讨论的具体实验。

FRT 适用于任何检验统计量。然而，这并没有回答在数据分析中如何选择检验统计量的实际问题。如果目标是发现与尖锐零假设相悖的证据，那么最好选择一个在备择假设下具有高**检验功效（power）**的检验统计量。一般来说，没有哪个检验统计量在功效方面能支配其他统计量，因为功效取决于备择假设。第 3.3 节中的四个检验统计量是由不同的备择假设驱动的。例如，$\hat{\tau}$ 和 t 是由具有非零平均处理效应的备择假设驱动的；W 是由具有异常值的恒定因果效应的备择假设驱动的。指定一个工作备择假设通常有助于构建检验统计量，尽管它不必精确到能保证 FRT 的有效性。习题 3.6 和 3.7 说明了使用工作备择假设或统计模型来构建检验统计量的思想。

## 3.7 家庭作业（Homework Problems）

## 3.1 $p _ { \mathrm {FRT} }$ 的精确性（Exactness of $p _ { \mathrm {FRT} }$）

证明 (3.2)。

## 3.2 $\hat { p } _ { \mathrm {FRT} }$ 的蒙特卡洛误差（Monte Carlo error of $\hat { p } _ { \mathrm {FRT} }$）

给定数据，$p _ { \mathrm {FRT} }$ 是一个固定数字，而其如 (3.4) 所示的蒙特卡洛估计量 $\hat { p } _ { \mathrm {FRT} }$ 是随机的。证明

$$
E _ {\mathrm{mc}} (\hat {p} _ {\mathrm{FRT}}) = p _ {\mathrm{FRT}}
$$

和

$$
\operatorname{var} _ {\mathrm{mc}} \left(\hat {p} _ {\mathrm{FRT}}\right) \leq \frac {1}{4 R},
$$

其中下标“mc”表示由蒙特卡洛引起的随机性，也就是说，$\hat { p } _ { \mathrm {FRT} }$ 是随机的，因为 $z ^ { r }$ 是从 $z$ 的所有可能值中独立随机抽取的 R 个样本。

注：$p _ { \mathrm {FRT} }$ 是随机的，因为 Z 是随机的。但在这个问题中，我们以数据为条件，所以 $p _ { \mathrm {FRT} }$ 变成了一个固定数字。$\hat { p } _ { \mathrm {FRT} }$ 是随机的，因为 $z ^ { r }$ 是 z 的随机排列。

习题 3.2 表明 $\hat { p } _ { \mathrm {FRT} }$ 对于 $p _ { \mathrm {FRT} }$ 在蒙特卡洛随机性下是无偏的，并给出了 $\hat { p } _ { \mathrm {FRT} }$ 方差的上界。Luo 等人（2021，定理 2）给出了一个关于蒙特卡洛误差的更精细的界。

## 3.3 $p _ { \mathrm {FRT} }$ 的一个有限样本有效蒙特卡洛近似（A finite-sample valid Monte Carlo approximation of $p _ { \mathrm {FRT} }$）

尽管 $\hat { p } _ { \mathrm {FRT} }$ 对于 $p _ { \mathrm {FRT} }$ 是无偏的，但由于有限 R 的蒙特卡洛误差，它可能不是一个有效的 p 值，因为对于所有 $u \in ( 0 , 1 )$ ，$\mathrm {pr} ( \hat { p } _ { \mathrm {FRT} } \leq u ) \leq u$ 可能不成立。以下修正的蒙特卡洛近似是有效的。Phipson 和 Smyth (2010) 在置换检验中指出了这个技巧。

定义

$$
\tilde {p} _ {\mathrm{FRT}} = \frac {1 + \sum_ {r = 1} ^ {R} I \{T (\boldsymbol {z} ^ {r} , \boldsymbol {Y}) \geq T (\boldsymbol {Z} , \boldsymbol {Y}) \}}{1 + R}
$$

其中 $z ^ { r }$ 是 Z 的 R 个随机排列。证明对于任意 $R$ ，蒙特卡洛近似 $\tilde { p } _ { \mathrm {FRT} }$ 总是一个有限样本有效的 p 值，即对于所有 $u \in ( 0 , 1 )$ ，有 $\mathrm {pr} ( \tilde { p } _ { \mathrm {FRT} } \leq u ) \leq u$ 。

提示：你可以使用以下两个基本概率结果来证明习题 3.3 中的断言。首先，对于两个二项随机变量 $X _ { 1 } \sim$ Binomial $( R , p _ { 1 } )$ 和 $X _ { 2 } \sim$ Binomial $( R , p _ { 2 } )$ ，且 $p _ { 1 } \geq p _ { 2 }$ ，那么对于所有 x，有 $\mathrm {pr} ( X _ { 1 } \leq x ) \ \leq \ \operatorname {pr} ( X _ { 2 } \ \leq \ x)$ 。其次，如果 $p \ \sim \ \mathrm {Uniform} ( 0 , 1 )$ 并且给定 p，$X \sim$ Binomial $\left( R , p \right)$ ，那么，从边际分布看，X 是在 $\{ 0 , 1 , \ldots , R \}$ 上的均匀随机变量。

## 3.4 费雪精确检验（Fisher’s exact test）

考虑一个具有二值结果变量的**完全随机实验（Completely Randomized Experiment, CRE）**，数据总结在下面的 $2 \times 2$ 表格中：

<table><tr><td></td><td>$ Y = 1 $</td><td>$ Y = 0 $</td><td>total</td></tr><tr><td>$ Z = 1 $</td><td>$ n_{11} $</td><td>$ n_{10} $</td><td>$ n_{1} $</td></tr><tr><td>$ Z = 0 $</td><td>$ n_{01} $</td><td>$ n_{00} $</td><td>$ n_{0} $</td></tr></table>

在 $H _ { \mathrm {0F} }$ 下，证明任何检验统计量都是 $n _ { 11 }$ 和其他非随机固定常数的函数，并且 $n _ { 11 }$ 的精确分布是**超几何分布（Hypergeometric）**。指定超几何分布的参数。

注：Barnard (1947) 以及 Ding 和 Dasgupta (2016) 指出，在具有二值结果变量的 CRE 下，费雪精确检验（在第 A1.3.1 节中回顾）和 FRT 是等价的。

## 3.5 品茶女士的更多细节（More details for lady tasting tea）

回顾第 3.5.2 节。计算 k = 0, 1, 2, 3, 4 时的 $\operatorname {pr} ( X = k )$ 。

## 3.6 协变量调整的 FRT（Covariate-adjusted FRT）

本题为第 3.6.2 节提供了更多细节。

第 3.4 节使用 FRT 重新分析了 LaLonde 实验数据。R 代码 FRTLalonde.R 使用四个检验统计量实现了 FRT。有了额外的协变量，FRT 可以更通用，至少有以下两种额外的策略。在潜在结果框架下，所有潜在结果和协变量都是固定数字。

首先，我们可以使用基于线性回归残差的检验统计量。对结果变量关于协变量进行线性回归，得到残差（即，将残差视为伪“结果”）。然后基于残差定义四个检验统计量。使用这四个新的检验统计量进行 FRT。报告相应的 p 值。

其次，我们可以将检验统计量定义为结果变量关于处理变量和协变量的线性回归中的系数。使用此检验统计量进行 FRT。报告相应的 p 值。

为什么上述两种策略得到的五个 p 值是有限样本精确的？请证明它们。

## 3.7 使用广义线性模型的 FRT（FRT with a generalized linear model）

使用与习题 3.6 相同的数据集，但将结果变量改为一个二值指示变量，表示 re78 是否为正。对结果变量关于处理变量和协变量进行**逻辑回归（logistic regression）**。处理变量的系数是否显著？p 值是多少？使用处理变量的系数作为检验统计量，通过 FRT 计算 p 值。

## 3.8 一个代数细节（An algebraic detail）

验证 (3.7)。

## 3.9 推荐阅读（Recommended reading）

Bind 和 Rubin (2020) 是最近一篇倡导在分析复杂实验时使用 p 值并展示相应随机化分布的论文。
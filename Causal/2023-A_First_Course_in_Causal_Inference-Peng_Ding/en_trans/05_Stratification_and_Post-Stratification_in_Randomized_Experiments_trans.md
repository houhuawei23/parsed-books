# 随机化实验中的分层与事后分层（Stratification and Post-Stratification in Randomized Experiments）

能分层的就分层，不能分层的就随机化。

——乔治·博克斯（George Box）

这是乔治·博克斯（George Box）第二著名的名言¹。本章将解释其含义。

## 5.1 分层（Stratification）

完全随机化实验（CRE）可能会产生不理想的处理分配。让我们从一个具有离散协变量 $X _ { i } \in \{ 1 , \ldots , K \}$ 的完全随机化实验开始，并定义 $n _ { [ k ] } = \# \{ i : X _ { i } = k \}$ 和 $\pi _ { [ k ] } = n _ { [ k ] } / n$ 分别为第 $k$ 层（$k = 1 , \ldots , K$）中的单元数量和比例。CRE 将 $n _ { 1 }$ 个单元分配到处理组，$n _ { 0 }$ 个单元分配到对照组，从而在第 $k$ 层的处理组和对照组中分别得到

$$
n _ {[ k ] 1} = \# \{i: X _ {i} = k, Z _ {i} = 1 \}, \quad n _ {[ k ] 0} = \# \{i: X _ {i} = k, Z _ {i} = 0 \}
$$

个单元。以正概率，对于某些 $k$，$n _ { [ k ] 1 }$ 或 $n _ { [ k ] 0 }$ 为零，即某些层可能只有处理单元或只有对照单元。即使没有 $n _ { [ k ] 1 }$ 或 $n _ { [ k ] 0 }$ 为零，以高概率有

$$
\frac {n _ {[ k ] 1}}{n _ {1}} - \frac {n _ {[ k ] 0}}{n _ {0}} \neq 0, \tag {5.1}
$$

且其幅度可能相当大。因此，虽然平均而言处理组和对照组中第 $k$ 层单元比例的差异为零：

$$
\begin{array}{l} E \left(\frac {n _ {[ k ] 1}}{n _ {1}} - \frac {n _ {[ k ] 0}}{n _ {0}}\right) \\ = E \left\{n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} 1 (X _ {i} = k) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) 1 (X _ {i} = k) \right\} \\ = 0. \\ \end{array}
$$

但当对于某些 $X = k$ 的层，$n _ { [ k ] 1 } / n _ { 1 } - n _ { [ k ] 0 } / n _ { 0 }$ 很大时，处理组和对照组会出现不良的**协变量不平衡（covariate imbalance）**。这种协变量不平衡会降低实验质量，使实验结果难以解释，因为结果的差异可能归因于处理或协变量不平衡。

我们如何在实验中主动避免协变量不平衡？我们可以预先固定 $n _ { [ k ] 1 }$ 或 $n _ { [ k ] 0 }$，并进行**分层随机化实验（stratified randomized experiments, SRE）**。

**定义 5.1（SRE）** 我们在离散协变量 $X$ 的 $K$ 个层内独立进行 $K$ 个 CRE。

在农业实验中，SRE 也被称为**随机化区组设计（randomized block design）**，其中的层被称为**区组（blocks）**。类似地，**分层随机化（stratified randomization）**也被称为**区组随机化（block randomization）**。SRE 中的总随机化次数等于

$$
\prod_ {k = 1} ^ {K} \binom{n _ {[ k ]}}{n _ {[ k ] 1}},
$$

且每种可行的随机化具有相等的概率。在第 $k$ 层内，接受处理的单元比例为

$$
e _ {[ k ]} = \frac {n _ {[ k ] 1}}{n _ {[ k ]}},
$$

这也被称为**倾向得分（propensity score）**，这一概念将在本书第三部分中发挥核心作用。SRE 与 CRE 不同：首先，SRE 中所有可行的随机化构成 CRE 中所有可行随机化的一个子集，因此

$$
\prod_ {k = 1} ^ {K} \binom{n _ {[ k ]}}{n _ {[ k ] 1}} <   \binom{n}{n _ {1}};
$$

其次，$e _ { [ k ] }$ 在 SRE 中是固定的，但在 CRE 中是随机的。

对于每个单元 $i$，我们有潜在结果 $Y _ { i } ( 1 )$ 和 $Y _ { i } ( 0 )$，以及个体因果效应 $\tau _ { i } = Y _ { i } ( 1 ) – Y _ { i } ( 0 )$。对于第 $k$ 层，我们有**层特异性平均因果效应（stratum-specific average causal effect）**

$$
\tau_ {[ k ]} = n _ {[ k ]} ^ {- 1} \sum_ {X _ {i} = k} \tau_ {i}.
$$

**平均因果效应（average causal effect）**为

$$
\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \tau_ {i} = n ^ {- 1} \sum_ {k = 1} ^ {K} \sum_ {X _ {i} = k} \tau_ {i} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \tau_ {[ k ]},
$$

这也是层特异性平均因果效应的加权平均。

如果我们对 $\tau _ { [ k ] }$ 感兴趣，那么我们可以使用第 3 章和第 4 章中关于第 $k$ 层内 CRE 的方法。下面我将讨论关于 $\tau$ 的统计推断。

## 5.2 费希尔随机化检验（FRT）

### 5.2.1 理论（Theory）

与 CRE 的讨论并行，我将从 SRE 中的 FRT 开始。**尖锐零假设（sharp null hypothesis）**仍然是

$$
H _ {0 \mathrm{F}}: Y _ {i} (1) = Y _ {i} (0) \text {  对于所有单元   } i = 1, \dots , n.
$$

FRT 的基本思想适用于任何随机化实验：我们可以使用任何在 $H _ { \mathrm { 0 F } }$ 和 SRE 下具有已知分布的检验统计量。然而，我们必须注意两个微妙的问题。首先，当我们模拟处理向量时，我们必须**在 $X$ 的层内置换处理指标**。由此产生的 FRT 有时被称为**条件随机化检验（conditional randomization test）**或**条件置换检验（conditional permutation test）**。其次，我们应该选择能够反映 SRE 性质的检验统计量。下面给出一些检验统计量的典型选择。

**例 5.1（分层估计量（Stratified estimator））** 受估计 $\tau$ 的启发，我们可以在 FRT 中使用以下分层估计量：

$$
\hat {\tau} _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \hat {\tau} _ {[ k ]},
$$

其中

$$
\hat {\tau} _ {[ k ]} = n _ {[ k ] 1} ^ {- 1} \sum_ {i = 1} ^ {n} I (X _ {i} = k, Z _ {i} = 1) Y _ {i} - n _ {[ k ] 0} ^ {- 1} \sum_ {i = 1} ^ {n} I (X _ {i} = k, Z _ {i} = 0) Y _ {i}
$$

是第 $k$ 层内的**层特异性均值差（stratum-specific difference-in-means）**。

**例 5.2（学生化分层估计量（Studentized stratified estimator））** 受简单两样本问题中学生化统计量的启发，我们可以在 FRT 中使用以下针对分层估计量的学生化统计量：

$$
t _ {\mathrm{S}} = \frac {\hat {\tau} _ {\mathrm{S}}}{\sqrt {\hat {V} _ {\mathrm{S}}}},
$$

其中

$$
\hat {V} _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \left(\frac {\hat {S} _ {[ k ]} ^ {2} (1)}{n _ {[ k ] 1}} + \frac {\hat {S} _ {[ k ]} ^ {2} (0)}{n _ {[ k ] 0}}\right)
$$

这里 $\hat { S } _ { [ k ] } ^ { 2 } ( 1 )$ 和 $\hat { S } _ { [ k ] } ^ { 2 } ( 0 )$ 分别是处理组和对照组中结果的**层特异性样本方差（stratum-specific sample variances）**。该统计量的精确形式受到第 5.3 节讨论的**奈曼视角（Neymanian perspective）**的启发。

**例 5.3（组合 Wilcoxon 秩和统计量（Combining Wilcoxon rank-sum statistics））** 我们首先计算第 $k$ 层内的 Wilcoxon 秩和统计量 $W _ { [ k ] }$，然后将它们组合为

$$
W _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} c _ {[ k ]} W _ {[ k ]}.
$$

基于不同的渐近方案和最优性准则，Van Elteren (1960) 提出了两种加权方法，一种使用

$$
c _ {[ k ]} = \frac {1}{n _ {[ k ] 1} n _ {[ k ] 0}},
$$

另一种使用

$$
c _ {[ k ]} = \frac {1}{n _ {[ k ]} + 1}
$$

这些权重的动机似乎相当技术性，其他权重选择也可能合理。

**例 5.4（Hodges 和 Lehmann (1962) 的对齐秩统计量（Hodges and Lehmann (1962)'s aligned rank statistic））** Van Elteren (1960) 的统计量在少数大层的情况下效果良好。然而，它在许多小层的情况下效果不佳，因为它没有进行足够的比较，可能丢失数据中的信息。Hodges 和 Lehmann (1962) 提出了一种检验统计量，该统计量在对结果进行标准化后跨层进行更多比较。他们建议首先将结果居中为

$$
\tilde {Y} _ {i} = Y _ {i} - \bar {Y} _ {[ k ]}
$$

其中层特异性均值 ${ \bar { Y } } _ { [ k ] } = n _ { [ k ] } ^ { - 1 } \sum _ { X _ { i } = k } Y _ { i }$（如果 $X _ { i } = k$），然后获取合并结果 $( { \tilde { Y } } _ { 1 } , \dots , { \tilde { Y } } _ { n } )$ 的秩 $( \tilde { R } _ { 1 } , \ldots , \tilde { R } _ { n } )$，最后构造检验统计量

$$
\tilde {W} = \sum_ {i = 1} ^ {n} Z _ {i} \tilde {R} _ {i}.
$$

我们可以在 SRE 下模拟上述检验统计量的精确分布。我们还可以计算它们的均值和方差，并基于正态近似获得 p 值。

经过一番搜索，我未能找到关于 SRE 的 Kolmogorov–Smirnov 统计量的详细讨论。以下是我的建议。

**例 5.5（Kolmogorov–Smirnov 统计量）** 我们计算 $D _ { [ k ] }$，即第 $k$ 层内处理组和对照组结果经验分布之间的最大差异。最终的检验统计量可以是

$$
D _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} c _ {[ k ]} D _ {[ k ]}
$$

或

$$
D _ {\max} = \max _ {1 \leq k \leq K} c _ {[ k ]} D _ {[ k ]},
$$

其中 $c _ { [ k ] } = \sqrt { n _ { [ k ] 1 } n _ { [ k ] 0 } / n _ { [ k ] } }$ 受到当 $n _ { [ k ] 1 }$ 和 $n _ { [ k ] 0 }$ 趋近于无穷时 $D _ { [ k ] }$ 的极限分布的启发 (Van der Vaart, 2000)。统计量 $D _ { \mathrm { { S } } }$ 和 $D _ { \mathrm { m a x } }$ 在所有层都具有大样本量时更为合适。另一个合理的选择是

$$
D = \max _ {y} \Big | \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \{\hat {F} _ {[ k ] 1} (y) - \hat {F} _ {[ k ] 0} (y) \} \Big |,
$$

其中 $\hat { F } _ { [ k ] 1 } ( y )$ 和 $\hat { F } _ { [ k ] 0 } ( y )$ 分别是处理组和对照组中结果的**层特异性经验分布函数（stratum-specific empirical distribution functions）**。统计量 $D$ 既适用于大层的情况，也适用于许多小层的情况。

### 5.2.2 一个应用（An application）

以 Penn Bonus 实验为例，说明 SRE 中的 FRT。Koenker 和 Xiao (2002) 使用的数据集来自一个按季度分层的职业培训项目，结果变量是就业前的持续时间。

```txt
penndata = read.table("Penn46_ascii.txt")
z = penndata$treatment
y = log(penndata$duration)
block = penndata$quarter
```

我将重点关注 $\mathrm { \hat { \tau } _ { S } }$ 和 $W _ { \mathrm { S } }$，并将其他统计量的 FRT 作为练习。以下函数计算 $\mathrm { \hat { \tau } _ { S } }$ 和 $V$：

```r
stat_SRE = function(z, y, x)
{
    xlevels = unique(x)
    K = length(xlevels)
    PiK = rep(0, K)
    TauK = rep(0, K)
    WilcoxK = rep(0, K)
    for(k in 1:K)
    {
    xk = xlevels[k]
    zk = z[x == xk]
    yk = y[x == xk]
    PiK[k] = length(zk)/length(z)
    TauK[k] = mean(yk[zk==1]) - mean(yk[zk==0])
    WilcoxK[k] = wilcox.test(yk[zk==1], yk[zk==0])$statistic
    }
    return(c(sum(PiK*TauK), sum(WilcoxK/PiK)))
}
```

以下函数在观测数据的 SRE 中生成一个随机处理分配：

```txt
zRandomSRE = function(z, x)
{
    xlevels = unique(x)
    K = length(xlevels)
    zrandom = z
    for(k in 1:K)
    {
    xk = xlevels[k]
    zrandom[x == xk] = sample(z[x == xk])
    }
    return(zrandom)
}
```

基于上述数据和函数，我们可以轻松模拟检验统计量的随机化分布（如图 5.1 所示，使用 $10^4$ 次蒙特卡洛抽样）并计算 p 值。

```diff
> MC = 10^4
> statSREMC = matrix(0, MC, 2)
> for(mc in 1:MC)
+ {
+    zrandom = zRandomSRE(z, block)
+    statSREMC[mc, ] = stat_SRE(zrandom, y, block)
+ }
> mean(statSREMC[, 1] <= stat.obs[1])
[1] 0.0019
> mean(statSREMC[, 2] <= stat.obs[2])
[1] 5e-04
```

## 5.3 奈曼推断（Neymanian inference）

## 5.3.1 点估计与区间估计（Point and interval estimation）

**分层随机实验（Stratified Randomized Experiment, SRE）** 的统计推断建立在其本质上由 $K$ 个独立 **完全随机实验（Completely Randomized Experiments, CREs）** 组成这一事实之上。基于此，我们可以轻松地将 **Neyman (1923)** 的结果推广到 SRE。在第 $k$ 层内，**均值差估计量（difference-in-means）** ${ \hat { \tau } } _ { [ k ] }$ 是 $\tau_{[k]}$ 的无偏估计量，其方差为：

$$
\mathrm{var} (\hat {\tau} _ {[ k ]}) = \frac {S _ {[ k ]} ^ {2} (1)}{n _ {[ k ] 1}} + \frac {S _ {[ k ]} ^ {2} (0)}{n _ {[ k ] 0}} - \frac {S _ {[ k ]} ^ {2} (\tau)}{n _ {[ k ]}},
$$

其中 $S _ { [ k ] } ^ { 2 } ( 1 ) , S _ { [ k ] } ^ { 2 } ( 0 )$ 和 $S _ { [ k ] } ^ { 2 } ( \tau )$ 分别是第 $k$ 层内潜在结果和个体处理效应的方差。因此，估计量 $\hat { \tau } _ { \mathrm { S } } = \sum _ { k = 1 } ^ { K } \pi _ { [ k ] } \hat { \tau } _ { [ k ] }$ 对于 $\tau = \sum _ { k = 1 } ^ { K } \pi _ { [ k ] } \tau _ { [ k ] }$ 的方差为：

$$
\mathrm{var} (\hat {\tau} _ {\mathrm{S}}) = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \mathrm{var} (\hat {\tau} _ {[ k ]}).
$$

如果 $n _ { [ k ] 1 } \geq 2$ 且 $n _ { [ k ] 0 } \geq 2$ ，那么我们可以得到第 $k$ 层内结果变量的样本方差 $\hat { S } _ { [ k ] } ^ { 2 } ( 1 )$ 和 $\hat { S } _ { [ k ] } ^ { 2 } ( 0 )$ ，并构造一个保守的方差估计量：

$$
\hat {V} _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \left(\frac {\hat {S} _ {[ k ]} ^ {2} (1)}{n _ {[ k ] 1}} + \frac {\hat {S} _ {[ k ]} ^ {2} (0)}{n _ {[ k ] 0}}\right),
$$

其中 $\hat { S } _ { [ k ] } ^ { 2 } ( 1 )$ 和 $\hat { S } _ { [ k ] } ^ { 2 } ( 0 )$ 分别是第 $k$ 层内处理组和对照组结果的样本方差。基于 $\mathrm { \hat { \tau } _ { S } }$ 的正态近似，我们可以构造一个 **Wald 型（Wald-type）** $1 - \alpha$ 置信区间用于 $\tau$：

$$
\hat {\tau} _ {\mathrm{S}} \pm z _ {1 - \alpha / 2} \sqrt {\hat {V} _ {\mathrm{S}}}.
$$

从假设检验的角度来看，在原假设 $H _ { \mathrm { 0 N } } : \tau = 0$ 下，我们可以将 $t _ { \mathrm { S } } = \hat { \tau } _ { \mathrm { S } } / \sqrt { \hat { V } _ { \mathrm { S } } }$ 与标准正态分位数进行比较，以获得渐近的 $p$ 值。统计量 $t _ { \mathrm { S } }$ 出现在例 5.2 的 **Fisher 随机化检验（Fisher Randomization Test, FRT）** 中。类似于对 CRE 的讨论，在 FRT 中使用 $t _ { \mathrm { S } }$ 可以在 $H _ { \mathrm { 0 F } }$ 下得到有限样本精确 $p$ 值，并在 $H _ { \mathrm { 0 N } }$ 下得到渐近有效的 $p$ 值。Wu 和 Ding (2021) 对这一论断提供了证明。

此处我略去关于 $\mathrm { \hat { \tau } _ { S } }$ 的中心极限定理的技术细节。请参见 Liu 和 Yang (2020) 的证明，该证明包括了少数大层和许多小层这两种情形。我将在 5.3.2 节中使用一个数值示例来说明这些理论问题。

## 5.3.2 数值示例（Numerical examples）

以下函数计算 **Neyman 点估计量和方差估计量（Neymanian point and variance estimators）**：

```python
Neyman_SRE = function(z, y, x)
{
    xlevels = unique(x)
    K = length(xlevels)
    PiK = rep(0, K)
    TauK = rep(0, K)
    varK = rep(0, K)
    for(k in 1:K)
    {
    xk = xlevels[k]
    zk = z[x == xk]
    yk = y[x == xk]
```

5.3 Neyman 推断（Neymanian inference）

```txt
PiK[k] = length(zk)/length(z)
TauK[k] = mean(yk[zk==1]) - mean(yk[zk==0])
varK[k] = var(yk[zk==1])/sum(zk) +
    var(yk[zk==0])/sum(1 - zk)
}
return(c(sum(PiK*TauK), sum(PiK^2*varK)))
}
```

第一个模拟设置中 $K = 5$，每层有 80 个单元。`TauHat` 和 `VarHat` 是来自 $10^4$ 次模拟的点估计量和方差估计量。

```diff
> K = 5
> n = 80
> n1 = 50
> n0 = 30
> x = rep(1:K, each = n)
> y0 = rexp(n*K, rate = x)
> y1 = y0 + 1
> zb = c(rep(1, n1), rep(0, n0))
> MC = 10^4
> TauHat = rep(0, MC)
> VarHat = rep(0, MC)
> for(mc in 1:MC)
+ {
+    z = replicate(K, sample(zb))
+    z = as.vector(z)
+    y = z*y1 + (1-z)*y0
+    est = Neyman_SRE(z, y, x)
+    TauHat[mc] = est[1]
+    VarHat[mc] = est[2]
+ }
> var(TauHat)
[1] 0.002248925
> mean(VarHat)
[1] 0.002266396
```

图 5.2 的上方面板显示了点估计量的直方图，它围绕真实参数呈对称的钟形分布。从上述结果来看，方差估计量的平均值几乎与估计量的方差相同，这是因为个体因果效应是恒定的。

第二个模拟设置中 $K = 50$，每层有 8 个单元。

```julia
> K = 50
> n = 8
> n1 = 5
> n0 = 3
> x = rep(1:K, each = n)
> y0 = rexp(n*K, rate = log(x + 1))
> y1 = y0 + 1
> zb = c(rep(1, n1), rep(0, n0))
```

```diff
> MC = 10^4
> TauHat = rep(0, MC)
> VarHat = rep(0, MC)
> for(mc in 1:MC)
+ {
+    z = replicate(K, sample(zb))
+    z = as.vector(z)
+    y = z*y1 + (1-z)*y0
+    est = Neyman_SRE(z, y, x)
+    TauHat[mc] = est[1]
+    VarHat[mc] = est[2]
+ }
>
> hist(TauHat, xlab = expression(hat(tau)[S]),
+    ylab = "", main = "many small strata",
+    border = FALSE, col = "grey",
+    breaks = 30, yaxt = 'n',
+    xlim = c(0.8, 1.2))
> abline(v = 1)
>
> var(TauHat)
[1] 0.001443111
> mean(VarHat)
[1] 0.001473616
```

图 5.2 的下方面板显示了点估计量的直方图，它围绕真实参数呈对称的钟形分布。

我们最后使用 **Penn 奖金实验（Penn Bonus Experiment）** 来演示 SRE 中的 Neyman 推断。将函数 `NeymanSRE` 应用于数据集，我们得到：

```txt
> est = Neyman_SRE(z, y, block)
> est[1]
[1] -0.08990646
> sqrt(est[2])
[1] 0.03079775
```

因此，职业培训项目显著缩短了就业前的失业持续时间。

## 5.3.3 比较 SRE 和 CRE（Comparing the SRE and the CRE）

与 CRE 相比，SRE 有哪些优势？我从协变量平衡的角度阐述了 SRE 的动机。此外，我将证明更好的协变量平衡反过来会提高平均因果效应的估计精度。为了进行公平比较，我假设对所有 $k$ 都有 $e _ { [ k ] } = e$，这保证了 $\hat { \tau } = \hat { \tau } _ { \mathrm { S } }$。我将此结果的证明留作问题 5.1。

我们现在比较抽样方差。经典的方差分析技术将总方差分解为层内方差和层间方差之和，得到：

$$
\begin{array}{l} S ^ {2} (1) = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} \left\{Y _ {i} (1) - \bar {Y} (1) \right\} ^ {2} \\ = (n - 1) ^ {- 1} \sum_ {k = 1} ^ {K} \sum_ {X _ {i} = k} \left\{Y _ {i} (1) - \bar {Y} _ {[ k ]} (1) + \bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \right\} ^ {2} \\ = (n - 1) ^ {- 1} \sum_ {k = 1} ^ {K} \sum_ {X _ {i} = k} \left[ \left\{Y _ {i} (1) - \bar {Y} _ {[ k ]} (1) \right\} ^ {2} + \left\{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \right\} ^ {2} \right] \\ = \sum_ {k = 1} ^ {K} \left[ \frac {n _ {[ k ]} - 1}{n - 1} S _ {[ k ]} ^ {2} (1) + \frac {n _ {[ k ]}}{n - 1} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} ^ {2} \right], \\ \end{array}
$$

类似地，

$$
S ^ {2} (0) = \sum_ {k = 1} ^ {K} \left[ \frac {n _ {[ k ]} - 1}{n - 1} S _ {[ k ]} ^ {2} (0) + \frac {n _ {[ k ]}}{n - 1} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} ^ {2} \right],
$$

$$
{S ^ {2} (\tau)} = {\sum_ {k = 1} ^ {K} \left[ \frac {n _ {[ k ]} - 1}{n - 1} S _ {[ k ]} ^ {2} (\tau) + \frac {n _ {[ k ]}}{n - 1} \{\tau_ {[ k ]} - \tau \} ^ {2} \right].}
$$

在大层的情况下，完全随机化下均值差估计量的方差近似为：

$$
\begin{array}{l} \mathrm{var} _ {\mathrm{CRE}} (\hat {\tau}) \\ = \frac {S ^ {2} (1)}{n _ {1}} + \frac {S ^ {2} (0)}{n _ {0}} - \frac {S ^ {2} (\tau)}{n} \\ \approx \sum_ {k = 1} ^ {K} \left[ \frac {\pi_ {[ k ]}}{n _ {1}} S _ {[ k ]} ^ {2} (1) + \frac {\pi_ {[ k ]}}{n _ {0}} S _ {[ k ]} ^ {2} (0) - \frac {\pi_ {[ k ]}}{n} S _ {[ k ]} ^ {2} (\tau) \right] \\ + \sum_ {k = 1} ^ {K} \left[ \frac {\pi_ {[ k ]}}{n _ {1}} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} ^ {2} + \frac {\pi_ {[ k ]}}{n _ {0}} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} ^ {2} - \frac {\pi_ {[ k ]}}{n} \{\tau_ {[ k ]} - \tau \} ^ {2} \right]. \\ \end{array}
$$

恒定的倾向得分假设保证了：

$$
\pi_ {[ k ]} / n _ {[ k ] 1} = 1 / (n e), \quad \pi_ {[ k ]} / n _ {[ k ] 0} = 1 / \{n (1 - e) \}, \quad \pi_ {[ k ]} / n _ {[ k ]} = 1 / n,
$$

这使我们能够将 SRE 下 $\mathrm { \hat { \tau } _ { S } }$ 的方差重写为：

$$
\begin{array}{l} \mathrm{var} _ {\mathrm{SRE}} (\hat {\tau} _ {\mathrm{S}}) = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \left[ \frac {S _ {[ k ]} ^ {2} (1)}{n _ {[ k ] 1}} + \frac {S _ {[ k ]} ^ {2} (0)}{n _ {[ k ] 0}} - \frac {S _ {[ k ]} ^ {2} (\tau)}{n _ {[ k ]}} \right] \\ = \sum_ {k = 1} ^ {K} \left[ \frac {\pi_ {[ k ]}}{n _ {1}} S _ {[ k ]} ^ {2} (1) + \frac {\pi_ {[ k ]}}{n _ {0}} S _ {[ k ]} ^ {2} (0) - \frac {\pi_ {[ k ]}}{n} S _ {[ k ]} ^ {2} (\tau) \right]. \\ \end{array}
$$

近似地，$\mathrm{var}_{\mathrm{CRE}}(\hat{\tau})$ 和 $\mathrm { v a r } _ { \mathrm { S R E } } \big ( \hat { \tau } _ { \mathrm { S } } \big )$ 之间的差异为：

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \left[ \frac {\pi_ {[ k ]}}{n _ {1}} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} ^ {2} + \frac {\pi_ {[ k ]}}{n _ {0}} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} ^ {2} - \frac {\pi_ {[ k ]}}{n} (\tau_ {[ k ]} - \tau) ^ {2} \right] \\ = \sum_ {k = 1} ^ {K} \frac {\pi_ {[ k ]}}{n} \left\{\sqrt {\frac {n _ {0}}{n _ {1}}} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} + \sqrt {\frac {n _ {1}}{n _ {0}}} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} \right\} ^ {2} \geq 0, \\ \end{array}
$$

该差值非负。差值仅在以下极端情况下为零：

$$
\sqrt {\frac {n _ {0}}{n _ {1}}} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} + \sqrt {\frac {n _ {1}}{n _ {0}}} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} = 0
$$

对于 $k = 1 , \ldots , K$ 成立。当协变量对潜在结果具有预测性时，上述量通常不全为零，这保证了 SRE 相对于 CRE 的效率增益。只有在协变量完全没有预测性的极端情况下，大样本效率增益才为零。在这些情况下，SRE 在有限样本中甚至可能导致更差的估计量。以上讨论印证了本章开头引用的 George Box 的话。

我将以几点说明来结束本节。第一，上述比较基于抽样方差，我们也可以比较 SRE 和 CRE 下的估计方差。结果是相似的。第二，增加 $K$ 可以提高效率，但这一论点依赖于大层假设。因此，我们在实践中面临权衡。我们不能任意增加 $K$，最极端的情况是 $n _ { [ k ] 1 } = n _ { [ k ] 0 } = 1$，这被称为 **配对实验（matched pair experiment）**，将在后续讨论。

## 5.4 CRE 中的事后分层（Post-stratification in a CRE）

在具有离散协变量 $X$ 的 CRE 中，接受处理和对照的单元数量在第 $k$ 层内是随机的。在 SRE 中，这些数量是固定的。但如果我们给定 $\pmb n = \{ n _ { [ k ] 1 } , n _ { [ k ] 0 } \} _ { k = 1 } ^ { K }$ 进行条件推断，那么 CRE 就变成了一个 SRE。从数学上讲，如果 $\pmb n$ 的任何一个分量都不为零，那么：

$$
\mathrm{pr} _ {\mathrm{CRE}} (\boldsymbol {Z} = \boldsymbol {z} \mid \boldsymbol {n}) = \frac {\mathrm{pr} _ {\mathrm{CRE}} (\boldsymbol {Z} = \boldsymbol {z} , \boldsymbol {n})}{\mathrm{pr} _ {\mathrm{CRE}} (\boldsymbol {n})} = \frac {1}{\prod_ {k = 1} ^ {K} \binom {n _ {[ k ]}} {n _ {[ k ] 1}}}, \tag {5.2}
$$

也就是说，给定 $\pmb n$ 时，来自 CRE 的 $Z$ 的条件分布与来自 SRE 的 $Z$ 的分布相同。因此，在给定 $\pmb n$ 的条件下，我们可以像分析 SRE 一样分析具有离散协变量 $X$ 的 CRE。特别地，FRT 变成了条件 FRT，Neyman 分析变成了 **事后分层（post-stratification）**：

$$
\hat {\tau} _ {\mathrm{PS}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \hat {\tau} _ {[ k ]},
$$

其形式与 $\mathrm { \hat { \tau } _ { S } }$ 相同。在给定 $\pmb n$ 的条件下，$\mathrm { \hat { \tau } _ { P S } }$ 的方差与 SRE 下 ${ \hat { \tau } _ { \mathrm { S } } }$ 的方差相同。

Hennessy 等人 (2016) 使用模拟表明，条件 FRT 通常比无条件 FRT 更有效。Miratrix 等人 (2013) 从理论上证明，在许多情况下，事后分层相比 ${ \hat { \tau } }$ 能提高效率。然而，该模拟基于有限数量的数据生成过程，且该理论假设所有层都足够大。我们在条件 FRT 或事后分层中不能走得太极端，因为随着 $K$ 增大，某些 $n _ { [ k ] 1 }$ 或 $n _ { [ k ] 0 }$ 变为零的可能性更大。$n _ { [ k ] 1 }$ 或 $n _ { [ k ] 0 }$ 较小或为零会大大减少 FRT 中的随机化次数，可能显著降低检验功效。对于 Neyman 方法来说，问题更为突出，因为我们甚至无法定义 $\mathrm { \hat { \tau } _ { P S } }$ 及相应的方差估计量。

**分层（Stratification）** 在设计阶段使用 $X$，而 **事后分层（post-stratification）** 在分析阶段使用 $X$。它们是对偶的。渐近地，在大层情况下，它们的差异很小 (Miratrix et al., 2013)。

## 5.4.1 Meinert 等人 (1970) 的示例（Meinert et al. (1970)’s Example）

我们使用来自 Meinert 等人 (1970) 的随机试验数据，该数据也被 Rothman 等人 (2008) 使用过。处理组为 **甲苯磺丁脲（tolbutamide）**，对照组为 **安慰剂（placebo）**。

| <td colspan="3">Age < 55</td> | <td colspan="3">Age ≥ 55</td> |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | Surviving | Dead |  | Surviving | Dead |
| Z = 1 | 98 | 8 | Z = 1 | 76 | 22 |
| Z = 0 | 115 | 5 | Z = 0 | 69 | 16 |
| <td colspan="6">Total</td> |
|  |  | Surviving | Dead |  |  |
|  | Z = 1 | 174 | 30 |  |  |
|  | Z = 0 | 184 | 21 |  |  |

下表显示了两个分层的单独估计值、事后分层估计值、忽略二元协变量的粗略估计值，以及相应的标准误。

|  | stratum 1 | stratum 2 | post-stratification | crude |
| :--- | :--- | :--- | :--- | :--- |
| est | -0.034 | -0.036 | -0.035 | -0.045 |
| se | 0.031 | 0.060 | 0.032 | 0.033 |

尽管粗略估计量和事后分层估计量没有产生根本不同的结果，但粗略估计量超出了分层特定估计量的范围，而事后分层估计量则在该范围之内。

## 5.4.2 Chong 等人（2016）的示例

Chong 等人（2016）在秘鲁进行了一项随机实验，以研究补充铁剂对学业表现的影响。该实验按班级层级（class level）进行了**分层（stratified）**。我将仅使用原始数据的一个子集。

```r
library("foreign")
dat_chong = read.dta("chong.dta")
use.vars = c("treatment",
    "gradesq34",
    "class_level",
    "anemic_base_re")
dat_physician = subset(dat_chong,
    treatment != "Soccer Player",
    select = use.vars)
dat_physician$z = (dat_physician$treatment=="Physician")
dat_physician$y = dat_physician$gradesq34
```

处理组和对照组的大小在五个**层（strata）**中有所不同：

```txt
> table(dat_physician$z,
+    dat_physician-class_level)
```

```txt
1 2 3 4 5
FALSE 15 19 16 12 10
TRUE 17 20 15 11 10
```

我们可以使用之前定义的 `NeymanSRE` 函数来计算分层估计量及其估计方差。

```erlang
tauS = with(dat_physician,
    Neyman_SRE(z, gradesq34, class_level))
```

一个重要的额外协变量是基线贫血指标（baseline anemic indicator），它对预测结果相当重要。进一步以基线贫血指标为条件，我们得到一个具有 $5 \times 2 = 10$ 个层的实验，其处理组和对照组的大小如下所示。

```erlang
> table(dat_physician$z,
+    dat_physician-class_level,
+    dat_physician$anemic_base_re)
, , = No
```

```txt
1 2 3 4 5
FALSE 6 14 12 7 4
TRUE 8 12 9 5 6
```

```txt
，， = Yes
```

```txt
1 2 3 4 5
FALSE 9 5 4 5 6
TRUE 9 8 6 6 4
```

同样，我们可以使用之前定义的 `NeymanSRE` 函数来计算**事后分层估计量（poststratified estimator）**及其估计方差。

```txt
tauSPS = with(dat_physician,
    {
    sps = interaction(class_level, anemic_base_re)
    Neyman_SRE(z, gradesq34, sps)
    })
```

下表比较了这两个估计量。事后分层估计量产生了更小的 p 值。

```txt
est se t.stat p.value stratify 0.406 0.202 2.005 0.045 stratify and post-stratify 0.463 0.190 2.434 0.015
```

这个例子说明，事后分层不仅可以用于**完全随机实验（Completely Randomized Experiment, CRE）**，也可以用于具有额外离散协变量的**分层随机实验（Stratified Randomized Experiment, SRE）**。

## 5.5 实践问题

如何选择 $X$ 来构建一个 SRE？从理论上讲，$X$ 应该能够预测潜在结果。在某些情况下，实验者基于例如一些**预实验（pilot studies）**，对预测性协变量有足够的背景知识。那么 $X$ 的选择应该是直接的。在其他一些情况下，这种背景知识可能不够清晰。实验者转而基于逻辑便利性选择 $X$，例如，$X$ 可以是研究区域或学生队列的指标。

$K$ 的选择是一个相关的问题。从理论上讲，如果所有层都足够大，更多的分层会提高估计效率。然而，非常大的 $K$ 甚至可能降低估计效率。在模拟研究中，我们观察到增加 $K$ 的边际收益递减。根据经验，$K = 5$ 通常足以获得效率增益。一些实验者倾向于 SRE 的最极端版本，即 $K = n / 2$ 。这导致了**配对设计（matched pair design）**，这将在后面的第 7 章讨论。

一些实验具有多维连续协变量。SRE 仍然可以使用吗？如果我们有预实验，我们可以根据这些协变量为潜在结果 $Y(0)$ 建立一个模型，然后我们可以选择 $X$ 作为预测变量 $\hat { Y } ( 0 )$ 的离散化版本。一般来说，如果我们没有这样的预实验，或者我们不想进行临时的离散化，我们可以使用一种更通用的策略，称为**重随机化（rerandomization）**，这是第 6 章的主题。

## 5.6 家庭作业问题

## 5.1 恒定倾向得分的后果

证明如果对于所有 $k = 1 , \ldots , K$ 都有 $e _ { [ k ] } = e$ ，那么 $\hat { \tau } = \hat { \tau } _ { \mathrm { S } }$ 。

## 5.2 恒定个体因果效应的后果

假设对于所有 $i \ =$ $1 , \ldots , n$ ，个体因果效应是恒定的 $\tau _ { i } ~ = ~ \tau$ 。考虑以下 $\tau$ 的加权估计量类别：

$$
\hat {\tau} _ {w} = \sum_ {k = 1} ^ {K} w _ {[ k ]} \hat {\tau} _ {[ k ]},
$$

其中对于所有 $k$ ， $w _ { [ k ] } \geq 0$ 。

找出 $w _ { [ k ] }$ 上的条件，使得 $\hat { \tau } _ { w }$ 是 $\tau$ 的无偏估计量。在所有无偏估计量中，找出方差最小的那个。

## 5.3 对 Imbens 和 Rubin（2015）中 Project STAR 数据的 FRT

使用**费希尔随机化检验（Fisher randomization test, FRT）**重新分析 Project STAR 数据。请注意，我使用 $Z$ 表示处理指示符，但 Imbens 和 Rubin（2015）使用 $W$ 。在费希尔随机化检验中使用 $\hat { \tau } _ { \mathrm { S } } , V$ 和**校准秩统计量（aligned rank statistic）**。比较 p 值。

```erlang
treatment = list(c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
```

```txt
outcome = list(c(0.165,0.321,-0.197,0.236),
    c(0.918,-0.202,1.19,0.117),
    c(0.341,0.561,-0.059,-0.496,0.225),
    c(-0.024,-0.450,-1.104,-0.956),
    c(-0.258,-0.083,-0.126,0.106),
    c(1.151,0.707,0.597,-0.495),
    c(0.077,0.371,0.685,0.270),
    c(-0.870,-0.496,-0.444,0.392,-0.934,-0.633),
    c(-0.568,-1.189,-0.891,-0.856),
    c(-0.727,-0.580,-0.473,-0.807),
    c(-0.533,0.458,-0.383,0.313),
    c(1.001,0.102,0.484,0.474,0.140),
    c(0.855,0.509,0.205,0.296),
    c(0.618,0.978,0.742,0.175),
    c(-0.545,0.234,-0.434,-0.293),
    c(-0.240,-0.150,0.355,-0.130))
```

## 5.4 一个多中心试验

Gould（1998，表1）报告了来自一个多中心试验的以下数据：

```csv
> multicenter = read.csv(" multicenter.csv")
> multicenter
center n0 mean0 sd0 n1 mean1 sd1 n5 mean5 sd5
1 1 7 0.43 4.58 7 -5.43 5.53 8 -2.63 3.38
2 2 11 0.10 4.21 11 -2.59 3.95 12 -2.21 4.14
3 3 6 2.58 4.80 6 -3.94 4.25 7 1.29 7.39
4 4 10 -2.30 3.86 10 -1.23 5.17 10 -1.40 2.27
5 5 10 2.08 6.46 10 -6.70 7.45 10 -5.13 3.91
6 6 6 1.13 3.24 5 3.40 8.17 5 -1.59 3.19
7 7 5 1.20 7.85 6 -3.67 4.89 5 -1.40 2.61
8 8 12 -1.21 2.66 13 0.18 3.81 12 -4.08 6.32
9 9 8 1.13 5.28 8 -2.19 5.17 9 -1.96 5.84
10 10 9 -0.11 3.62 10 -2.00 5.35 10 0.60 3.53
11 11 15 -4.37 6.12 14 -2.68 5.34 15 -2.14 4.27
12 12 8 -1.06 5.27 9 0.44 4.39 9 -2.03 5.76
13 13 12 -0.08 3.32 12 -4.60 6.16 11 -6.22 5.33
14 14 9 0.00 5.20 9 -0.25 8.23 7 -3.29 5.12
15 15 6 1.83 5.85 7 -1.23 4.33 6 -1.00 2.61
16 16 14 -4.21 7.53 14 -2.10 5.78 12 -5.75 5.63
17 17 13 0.76 3.82 13 0.55 2.53 13 -0.63 5.41
18 18 15 -1.05 4.54 13 2.54 4.16 14 -2.80 2.89
19 19 15 2.07 4.88 15 -1.67 4.95 15 -3.43 4.71
20 20 11 -1.46 5.48 10 -1.99 5.63 10 -6.77 5.19
21 21 5 0.80 4.21 5 -3.35 4.73 5 -0.23 4.14
22 22 11 -2.92 5.42 10 -1.22 5.95 11 -4.45 6.65
23 23 9 -3.37 4.73 9 -1.38 4.17 7 0.57 2.70
24 24 12 -1.92 2.91 12 -0.66 3.55 12 -2.39 2.27
25 25 9 -3.89 4.76 9 -3.22 5.54 8 -1.23 4.91
```

## 5.6 家庭作业问题

<table><tr><td>26</td><td>26</td><td>15</td><td>-3.48</td><td>5.98</td><td>15</td><td>-2.13</td><td>3.25</td><td>14</td><td>-3.71</td><td>5.30</td></tr><tr><td>27</td><td>27</td><td>11</td><td>-1.91</td><td>6.49</td><td>12</td><td>-1.33</td><td>4.40</td><td>11</td><td>-1.52</td><td>4.68</td></tr><tr><td>28</td><td>28</td><td>10</td><td>-2.66</td><td>3.80</td><td>10</td><td>-1.29</td><td>3.18</td><td>10</td><td>-4.70</td><td>3.43</td></tr><tr><td>29</td><td>29</td><td>13</td><td>-0.77</td><td>4.73</td><td>13</td><td>-2.31</td><td>3.88</td><td>13</td><td>-0.47</td><td>4.95</td></tr></table>

这是一个以中心为层的 SRE。该试验旨在研究**非那雄胺（finasteride）**（一种治疗良性前列腺增生的药物）的疗效和耐受性。在 29 个中心中的每一个，患者被随机分配到三个组：对照组、非那雄胺 1mg 组和非那雄胺 5mg 组。上述数据集提供了结果的汇总统计量，结果变量是总症状评分相对于基线的变化。总症状评分是对九个问题（评分 0 到 4）的回答之和，这些问题涉及排尿能力受损的各个方面。各列的含义是：

1.  center: 中心编号；
2.  n0, n1, n5: 三个组的样本量；
3.  mean0, mean1, mean5: 结果变量的均值；
4.  sd0, sd1, sd5: 结果变量的标准差。

个体层面的结果没有报告，因此我们无法实施 FRT。然而，**奈曼推断（Neymanian inference）**仅需要汇总统计量。分别报告比较“非那雄胺 1mg”和“非那雄胺 5mg”与“对照组”的点估计量和方差估计量。

## 5.5 数据再分析

重新分析 `Neymanlalonde.R` 中使用的 LaLonde 数据。同时进行**费希尔推断（Fisherian inference）**和奈曼推断。

原始实验是一个完全随机实验。现在我们假设原始实验是一个分层随机实验。首先，假设实验按种族（黑人、西班牙裔或其他）分层，重新分析数据。其次，假设实验按婚姻状况分层，重新分析数据。第三，假设实验按高中文凭指标分层，重新分析数据。

与在完全随机实验下获得的结果进行比较。

## 5.6 推荐阅读

Miratrix 等人（2013）为事后分层提供了坚实的理论，并将其与分层进行了比较。一个主要的理论结果是，尽管它们在有限样本中可能有所不同，但在渐近情况下它们的差异很小。
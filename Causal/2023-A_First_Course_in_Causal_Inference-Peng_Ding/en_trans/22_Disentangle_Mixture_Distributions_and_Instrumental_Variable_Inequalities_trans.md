# 混合分布的解耦与工具变量不等式（Disentangle Mixture Distributions and Instrumental Variable Inequalities）

第21章中的**工具变量模型（Instrumental Variable model, IV model）**施加了假设21.1–21.3：

1. $Z \bot \bot \{ D ( 1 ) , D ( 0 ) , Y ( 1 ) , Y ( 0 ) \}$ ;
2. $\operatorname { p r } ( U = \mathrm { d } ) = 0$ ;
3. $Y ( 1 ) = Y ( 0 )$ 对于 $U = \mathrm { a ~ o r ~ n . }$

表22.1总结了观测组和对应的潜在组。

**表22.1：在假设21.2下的观测组与潜在组（TABLE 22.1: Observed groups and latent groups under Assumption 21.2）**

<table><tr><td>Z=1</td><td>D=1</td><td>D(1)=1</td><td>U=c or a</td></tr><tr><td>Z=1</td><td>D=0</td><td>D(1)=0</td><td>U=n</td></tr><tr><td>Z=0</td><td>D=1</td><td>D(0)=1</td><td>U=a</td></tr><tr><td>Z=0</td><td>D=0</td><td>D(0)=0</td><td>U=c or n</td></tr></table>

有趣的是，假设21.1–21.3共同具有一些可检验的含义。**Balke和Pearl（1997）**将其称为**工具变量不等式（instrumental variable inequalities）**。本章将对这些不等式的一个特例进行直观推导。该证明是识别由 $U$ 定义的所有潜在组的**潜在结果均值（means of the potential outcomes）**的直接结果。

## 22.1 混合分布的解耦与工具变量不等式（Disentangle Mixture Distributions and Instrumental Variable Inequalities）

我们在下面的定理22.1中总结了主要结果。回顾 $\pi _ { u }$ 为类型 $U = u$ 的比例，并定义

$$
\mu_ {z u} = E \{Y (z) \mid U = u \}, \quad (d = 0, 1; u = \mathrm{a,n,c}).
$$

**定理22.1（Theorem 22.1）** 在假设21.1–21.3下，我们可以通过以下公式识别潜在类型的比例：

$$
\pi_ {\mathrm{n}} = \operatorname{pr} (D = 0 | Z = 1),
$$

$$
\pi_ {\mathrm{a}} = \operatorname * {p r} (D = 1 \mid Z = 0),
$$

$$
\pi_ {\mathrm{c}} = E (D \mid Z = 1) - E (D \mid Z = 0),
$$

以及潜在结果的类型特异性均值：

$$
\mu_ {1 \mathrm{n}} = \mu_ {0 \mathrm{n}} \equiv \mu_ {\mathrm{n}} = E (Y \mid Z = 1, D = 0),
$$

$$
\mu_ {1 \mathrm{a}} = \mu_ {0 \mathrm{a}} \equiv \mu_ {\mathrm{a}} = E (Y \mid Z = 0, D = 1),
$$

$$
\mu_ {1 \mathrm{c}} = \pi_ {\mathrm{c}} ^ {- 1} \left\{E (D Y \mid Z = 1) - E (D Y \mid Z = 0) \right\},
$$

$$
\mu_ {0 \mathrm{c}} = \pi_ {\mathrm{c}} ^ {- 1} \left[ E \{(1 - D) Y \mid Z = 0 \} - E \{(1 - D) Y \mid Z = 1 \} \right].
$$

**定理17.1的证明（Proof of Theorem 17.1）：** 第一部分：我们首先识别潜在**依从类型（compliance types）**的比例。我们可以通过以下公式识别**从不依从者（never takers）**的比例：

$$
\operatorname{pr} (D = 0 \mid Z = 1) = \operatorname{pr} (U = \mathrm{n} \mid Z = 1)
$$

$$
= \operatorname{pr} (U = \mathrm{n}) = \pi_ {\mathrm{n}},
$$

以及**总是依从者（always takers）**的比例：

$$
\operatorname{pr} (D = 1 \mid Z = 0) = \operatorname{pr} (U = \mathrm{a} \mid Z = 0)
$$

$$
= \operatorname{pr} (U = \mathrm{a}) = \pi_ {\mathrm{a}}.
$$

因此，**依从者（compliers）**的比例为：

$$
\pi_ {\mathrm{c}} = \operatorname * {p r} (U = \mathrm{c}) = 1 - \pi_ {\mathrm{n}} - \pi_ {\mathrm{a}}
$$

$$
= 1 - \operatorname{pr} (D = 0 \mid Z = 1) - \operatorname{pr} (D = 1 \mid Z = 0)
$$

$$
= E (D \mid Z = 1) - E (D \mid Z = 0) = \tau_ {D},
$$

这与我们之前的讨论一致。虽然我们不知道所有个体的潜在依从类型，但我们可以识别从不依从者、总是依从者和依从者的比例。

第二部分：然后我们识别潜在依从类型内部的潜在结果均值。在假设21.3下，

$$
\mu_ {\mathrm{1a}} = \mu_ {\mathrm{0a}} \equiv \mu_ {\mathrm{a}}, \quad \mu_ {\mathrm{1n}} = \mu_ {\mathrm{0n}} \equiv \mu_ {\mathrm{n}}.
$$

观测组 $(Z = 1, D = 0)$ 只包含从不依从者，因此：

$$
E (Y \mid Z = 1, D = 0) = E \{Y (1) \mid Z = 1, U = \mathrm{n} \} = E \{Y (1) \mid U = \mathrm{n} \} = \mu_ {\mathrm{n}}.
$$

观测组 $(Z = 0, D = 1)$ 只包含总是依从者，因此：

$$
E (Y \mid Z = 0, D = 1) = E \{Y (0) \mid Z = 0, U = \mathrm{a} \} = E \{Y (0) \mid U = \mathrm{a} \} = \mu_ {\mathrm{a}}.
$$

## 22.2 混合分布的解耦与工具变量不等式（Disentangle Mixture Distributions and Instrumental Variable Inequalities）269

观测组 $(Z = 1, D = 1)$ 同时包含依从者和总是依从者，因此：

$$
\begin{array}{l} E (Y \mid Z = 1, D = 1) = E \{Y (1) \mid Z = 1, D (1) = 1 \} \\ = E \{Y (1) \mid D (1) = 1 \} \\ = \operatorname{pr} \{D (0) = 1 \mid D (1) = 1 \} E \{Y (1) \mid D (1) = 1, D (0) = 1 \} \\ + \operatorname{pr} \{D (0) = 0 \mid D (1) = 1 \} E \{Y (1) \mid D (1) = 1, D (0) = 0 \} \\ { = } { \frac { \pi _ { \mathrm{c} } } { \pi _ { \mathrm{c} } + \pi _ { \mathrm{a} } } \mu _ { 1 \mathrm{c} } + \frac { \pi _ { \mathrm{a} } } { \pi _ { \mathrm{c} } + \pi _ { \mathrm{a} } } \mu _ { \mathrm{a} } . } \\ \end{array}
$$

解上述线性方程可得：

$$
\begin{array}{l} \mu_ {1 \mathrm{c}} = \pi_ {\mathrm{c}} ^ {- 1} \left\{\left(\pi_ {\mathrm{c}} + \pi_ {\mathrm{a}}\right) E (Y \mid Z = 1, D = 1) - \pi_ {\mathrm{a}} E (Y \mid Z = 0, D = 1) \right\} \\ = \pi_ {\mathrm{c}} ^ {- 1} \left\{\operatorname * {p r} (D = 1 \mid Z = 1) E (Y \mid Z = 1, D = 1) \right. \\ - \operatorname{pr} (D = 1 \mid Z = 0) E (Y \mid Z = 0, D = 1) \} \\ = \pi_ {\mathrm{c}} ^ {- 1} \left\{E (D Y \mid Z = 1) - E (D Y \mid Z = 0) \right\}. \\ \end{array}
$$

观测组 $(Z = 0, D = 0)$ 同时包含依从者和从不依从者，因此我们有：

$$
\begin{array}{l} E (Y \mid Z = 0, D = 0) = E \{Y (0) \mid Z = 0, D (0) = 0 \} \\ = E \{Y (0) \mid D (0) = 0 \} \\ = \operatorname{pr} \{D (1) = 1 \mid D (0) = 0 \} E \{Y (0) \mid D (1) = 1, D (0) = 0 \} \\ + \operatorname{pr} \{D (1) = 0 \mid D (0) = 0 \} E \{Y (0) \mid D (1) = 0, D (0) = 0 \} \\ = \frac {\pi_ {\mathrm{c}}}{\pi_ {\mathrm{c}} + \pi_ {\mathrm{n}}} \mu_ {0 \mathrm{c}} + \frac {\pi_ {\mathrm{n}}}{\pi_ {\mathrm{c}} + \pi_ {\mathrm{n}}} \mu_ {\mathrm{n}}. \\ \end{array}
$$

解上述线性方程可得：

$$
\begin{array}{l} \mu_ {0 \mathrm{c}} = \pi_ {\mathrm{c}} ^ {- 1} \left\{\left(\pi_ {\mathrm{c}} + \pi_ {\mathrm{n}}\right) E (Y \mid Z = 0, D = 0) - \pi_ {\mathrm{n}} E (Y \mid Z = 1, D = 0) \right\} \\ = \pi_ {\mathrm{c}} ^ {- 1} \left\{\operatorname * {p r} (D = 0 \mid Z = 0) E (Y \mid Z = 0, D = 0) \right. \\ \left. - \operatorname{pr} (D = 0 \mid Z = 1) E (Y \mid Z = 1, D = 0) \right\} \\ = \pi_ {c} ^ {- 1} \left[ E \{(1 - D) Y \mid Z = 0 \} - E \{(1 - D) Y \mid Z = 1 \} \right]. \\ \end{array}
$$

基于定理22.1中 $\mu _ { \mathrm { 1 c } }$ 和 $\mu _ { \mathrm { 0 c } }$ 的公式，我们有：

$$
\tau_ {\mathrm{c}} = \mu_ {1 \mathrm{c}} - \mu_ {0 \mathrm{c}} = \left\{E (Y \mid Z = 1) - E (Y \mid Z = 0) \right\} / \pi_ {\mathrm{c}},
$$

这与之前定理21.1中的公式相同。

定理22.1侧重于识别潜在结果均值 $\mu _ { z u }$ 。**Imbens和Rubin（1997）**推导了潜在结果分布的更一般识别公式；我将细节留作问题22.2。

## 22.2 可检验的含义（Testable implications）

这种推导 $\tau _ { \mathrm { c } }$ 公式的迂回方法是否有额外价值？答案是肯定的。对于**二元结果（binary outcome）**，以下不等式必须成立：

$$
0 \leq \mu_ {1 c} \leq 1, \quad 0 \leq \mu_ {0 c} \leq 1,
$$

这意味着四个不等式：

$$
E (D Y \mid Z = 1) - E (D Y \mid Z = 0) \geq 0,
$$

$$
E (D Y \mid Z = 1) - E (D Y \mid Z = 0) \leq E (D \mid Z = 1) - E (D \mid Z = 0),
$$

$$
E \{(1 - D) Y \mid Z = 0 \} - E \{(1 - D) Y \mid Z = 1 \} \geq 0,
$$

$$
E \{(1 - D) Y \mid Z = 0 \} - E \{(1 - D) Y \mid Z = 1 \} \leq E (D \mid Z = 1) - E (D \mid Z = 0).
$$

整理各项，我们得到以下统一的不等式。

**定理22.2（工具变量不等式）（Theorem 22.2 (Instrumental Variable Inequalities)）** 对于二元结果 $Y$ ，假设21.1–21.3意味着：

$$
E (Q \mid Z = 1) - E (Q \mid Z = 0) \geq 0, \tag {22.1}
$$

其中 $Q = D Y , D ( 1 - Y ) , ( D - 1 ) Y$ 和 $D + Y - D Y .$

在IV假设21.1–21.3下，$Q = D Y , D ( 1 - Y ) , ( D - 1 ) Y$ 和 $D + Y - D Y$ 的均值差必须全部非负。重要的是，这些含义仅涉及观测变量的分布。拒绝IV不等式将导致拒绝IV假设。

**Balke和Pearl（1997）**在不假设单调性的情况下推导了更一般的IV不等式。上述证明策略来自**Jiang和Ding（2020）**，针对稍复杂的情境。定理22.2仅陈述了二元结果的可检验含义。问题22.3给出了一个等价形式，问题22.4给出了一般结果的结果。

## 22.3 示例（Examples）

对于二元结果，我们可以通过以下**矩方法（method of moment）**估计所有参数。

```r
## function for binary data (Z, D, Y)
## n_{zdy}'s are the counts from 2X2X2 table
IVbinary = function(n111, n110, n101, n100, n011, n010, n001, n000){
```

22.3 示例（Examples）

```txt
n_tr = n111 + n110 + n101 + n100
n_co = n011 + n010 + n001 + n000
n    = n_tr + n_co

## proportions of the latent strata
pi_n = (n101 + n100)/n_tr
pi_a = (n011 + n010)/n_co
pi_c = 1 - pi_n - pi_a

## four observed means of the outcomes (Z=z,D=d)
mean_y_11 = n111/(n111 + n110)
mean_y_10 = n101/(n101 + n100)
mean_y_01 = n011/(n011 + n010)
mean_y_00 = n001/(n001 + n000)

## means of the outcomes of two strata
mu_n1 = mean_y_10
mu_a0 = mean_y_01
## ER implies the following two means
mu_n0 = mu_n1
mu_a1 = mu_a0
## stratum (Z=1,D=1) is a mixture of c and a
mu_c1 = ((pi_c + pi_a)*mean_y_11 - pi_a*mu_a1)/pi_c
## stratum (Z=0,D=0) is a mixture of c and n
mu_c0 = ((pi_c + pi_n)*mean_y_00 - pi_n*mu_n0)/pi_c

## identifiable quantities from the observed data
list(pi_c = pi_c,
    pi_n = pi_n,
    pi_a = pi_a,
    mu_c1 = mu_c1,
    mu_c0 = mu_c0,
    mu_n1 = mu_n1,
    mu_n0 = mu_n0,
    mu_a1 = mu_a1,
    mu_a0 = mu_a0)
}
```

然后我们重新审视两个经典示例。

**例22.1（Example 22.1）** Investigators等人（2014）评估了紧急血管内修复与开放手术修复策略对临床诊断为**破裂性主动脉瘤（ruptured aortic aneurism）**患者的有效性。患者被随机分配到紧急血管内策略或开放修复策略。主要结果是30天后的**生存状态（survival status）**。令 $Z$ 为分配的治疗，$Z = 1$ 表示血管内策略，$Z = 0$ 表示开放修复。令 $D$ 为实际接受的治疗。令 $Y$ 为生存状态，$Y = 1$ 表示死亡，$Y = 0$ 表示存活。$\tau _ { \mathrm { c } }$ 的估计值为0.131，95%置信区间为(−0.036, 0.298)，包含0。使用上述函数，我们可以得到：

**表22.2：二元数据与IV不等式（TABLE 22.2: Binary data and IV inequalities） (a) Investigators等人（2014）的研究**

<table><tr><td rowspan="2"></td><td colspan="2">Z=1</td><td colspan="2">Z=0</td></tr><tr><td>D=1</td><td>D=0</td><td>D=1</td><td>D=0</td></tr><tr><td>Y=1</td><td>107</td><td>68</td><td>24</td><td>131</td></tr><tr><td>Y=0</td><td>42</td><td>42</td><td>8</td><td>79</td></tr></table>

**(b) Hirano等人（2000）的研究**

<table><tr><td rowspan="2"></td><td colspan="2">Z=1</td><td colspan="2">Z=0</td></tr><tr><td>D=1</td><td>D=0</td><td>D=1</td><td>D=0</td></tr><tr><td>Y=1</td><td>31</td><td>85</td><td>30</td><td>99</td></tr><tr><td>Y=0</td><td>424</td><td>944</td><td>237</td><td>1041</td></tr></table>

\$mu  c1

[1] 0.7086064

\$mu  c0

[1] 0.6292042

没有证据表明违反IV假设。

**例22.2（Example 22.2）** 在**Hirano等人（2000）**的研究中，医生被随机选中接收一封鼓励他们为流感风险患者接种疫苗的信件。治疗是实际的流感疫苗接种，结果是流感相关住院的指示变量。然而，一些患者并未遵从他们的分配。令 $Z _ { i }$ 为鼓励接种流感疫苗的指示变量，$Z = 1$ 表示医生收到鼓励信，$Z = 0$ 表示未收到。令 $D$ 为实际接受的治疗。令 $Y$ 为结果，$Y = 0$ 表示冬季因流感相关住院，$Y = 1$ 表示未住院。$\tau _ { \mathrm { c } }$ 的估计值为0.116，95%置信区间为(−0.061, 0.293)，包含0。使用上述函数，我们可以得到：

\$mu  c1

[1] -0.004548064

\$mu  c0

[1] 0.1200094

由于 $\hat { \mu } _ { \mathrm { 1 c } } < 0$ ，有证据表明违反IV假设。

## 22.4 家庭作业问题（Homework problems）

## 22.1 依从者的风险比（Risk ratio for compliers）

对于二元结果，我们可以将依从者的风险比定义为

$$
\mathrm{RR} _ {\mathrm{c}} = \frac {\operatorname* {p r} \{Y (1) = 1 \mid U = \mathrm{c} \}}{\operatorname* {p r} \{Y (0) = 1 \mid U = \mathrm{c} \}}.
$$

证明在假设 21.1–21.3 下，我们可以通过下式识别它

$$
\mathrm{RR} _ {\mathrm{c}} = \frac {E (D Y \mid Z = 1) - E (D Y \mid Z = 0)}{E \{(D - 1) Y \mid Z = 1 \} - E \{(D - 1) Y \mid Z = 0 \}}.
$$

注：利用定理 22.1，我们可以识别 $E \{ Y ( 1 ) \mid U = \operatorname { c } \}$ 和 $E \{ Y ( 0 ) \mid U = \operatorname { c } \}$ 之间的任何比较。

## 22.2 解构混合分布：分布结果（Disentangle the mixtures: distributional results）

本题推广了定理 22.1。定义

$$
f _ {z u} (y) = \operatorname{pr} \{Y (z) = y \mid U = u \}, \quad (d = 0, 1; u = \mathrm{a}, \mathrm{n}, \mathrm{c})
$$

为潜在层 $U = u$ 的 $Y ( z )$ 的密度，并定义

$$
g _ {z d} (y) = \operatorname{pr} (Y = y \mid Z = z, D = d)
$$

为观测组 $( Z = z , D = d )$ 内结果的密度。证明下面的定理 22.3。

**定理 22.3** 在假设 21.1–21.3 下，我们可以通过下式识别潜在结果的分层特异性密度（typespecific densities）：

$$
f _ {1 \mathrm{n}} (y) = f _ {0 \mathrm{n}} (y) \equiv f _ {\mathrm{n}} (y) = g _ {1 0} (y),
$$

$$
f _ {1 \mathrm{a}} (y) = f _ {0 \mathrm{a}} (y) \equiv f _ {\mathrm{a}} (y) = g _ {0 1} (y),
$$

$$
f _ {1 c} (y) = \pi_ {c} ^ {- 1} \left\{\operatorname{pr} (D = 1 \mid Z = 1) g _ {1 1} (y) - \operatorname{pr} (D = 1 \mid Z = 0) g _ {0 1} (y) \right\},
$$

$$
f _ {0 \mathrm{c}} (y) = \pi_ {\mathrm{c}} ^ {- 1} \{\operatorname * {p r} (D = 0 | Z = 0) g _ {0 0} (y) - \operatorname * {p r} (D = 0 | Z = 1) g _ {1 0} (y) \}.
$$

## 22.3 定理 22.2 的另一种形式（Alternative form of Theorem 22.2）

(22.1) 中的不等式可以重新写为

$$
\operatorname{pr} (D = 1, Y = y \mid Z = 1) \geq \operatorname{pr} (D = 1, Y = y \mid Z = 0),
$$

$$
\operatorname{pr} (D = 0, Y = y \mid Z = 0) \geq \operatorname{pr} (D = 0, Y = y \mid Z = 1)
$$

对于 $y = 0 , 1$ 均成立。

## 22.4 一般结果的工具变量不等式（Instrumental variable inequalities for a general outcome）

对于一般结果 $Y$ ，证明假设 21.1–21.3 蕴含

$$
\operatorname{pr} (D = 1, Y \geq y \mid Z = 1) \geq \operatorname{pr} (D = 1, Y \geq y \mid Z = 0),
$$

$$
\operatorname{pr} (D = 1, Y <   y \mid Z = 1) \geq \operatorname{pr} (D = 1, Y <   y \mid Z = 0),
$$

$$
\operatorname{pr} (D = 0, Y \geq y \mid Z = 0) \geq \operatorname{pr} (D = 0, Y \geq y \mid Z = 1),
$$

$$
\operatorname{pr} (D = 0, Y <   y \mid Z = 0) \geq \operatorname{pr} (D = 0, Y <   y \mid Z = 1)
$$

对所有 $y$ 成立。

注：Imbens 和 Rubin (1997) 以及 Kitagawa (2015) 讨论了类似的结果。例如，我们可以基于**柯尔莫哥洛夫-斯米尔诺夫统计量（Kolmogorov–Smirnov statistic）**的类似形式检验第一个不等式：

$$
\mathrm{KS} _ {1} = \max _ {y} \Big | \frac {\sum_ {i = 1} ^ {n} Z _ {i} D _ {i} 1 (Y _ {i} \leq y)}{\sum_ {i = 1} ^ {n} Z _ {i} D _ {i}} - \frac {\sum_ {i = 1} ^ {n} (1 - Z _ {i}) D _ {i} 1 (Y _ {i} \leq y)}{\sum_ {i = 1} ^ {n} (1 - Z _ {i}) D _ {i}} \Big |.
$$

## 22.5 工具变量不等式示例（Example for the IV inequalities）

给出一个所有 IV 不等式均成立的示例，以及另一个并非所有 IV 不等式都成立的示例。你需要指定具有二元 $Z$ 和 $D$ 的 $(Z, D, Y)$ 的联合分布。

## 22.6 关键假设的违背（Violations of the key assumptions）

定理 21.1 依赖于**随机化（randomization）**、**单调性（monotonicity）**和**排他性约束（exclusion restriction）**。即使在随机化实验中，后两者也是不可检验的。当它们被违背时，IV 估计量不再能识别依从者平均因果效应。本题给出以下两种情况，它们是对 Angrist 等人 (1996) 中命题 2 和命题 3 的重新表述。

在没有排他性约束的假设 21.1 和 21.2 下，我们有

$$
\frac {E (Y \mid Z = 1) - E (Y \mid Z = 0)}{E (D \mid Z = 1) - E (D \mid Z = 0)} - \tau_ {\mathrm{c}} = \frac {\pi_ {\mathrm{a}} \tau_ {\mathrm{a}} + \pi_ {\mathrm{n}} \tau_ {\mathrm{n}}}{\pi_ {\mathrm{c}}}
$$

其中

$$
\tau_ {u} = E \{Y (1) - Y (0) \mid U = u \}, (U = \mathrm{a,n,c}).
$$

在没有单调性的假设 21.1 和 21.3 下，我们有

$$
\frac {E (Y \mid Z = 1) - E (Y \mid Z = 0)}{E (D \mid Z = 1) - E (D \mid Z = 0)} - \tau_ {\mathrm{c}} = \frac {\pi_ {\mathrm{d}} (\tau_ {\mathrm{c}} + \tau_ {\mathrm{d}})}{\pi_ {\mathrm{c}} - \pi_ {\mathrm{d}}}.
$$

证明上述两个结果。

## 22.7 其他分析的问题（Problems of other analyses）

在 22.1 节推导 IV 不等式的过程中，我们通过识别潜在层的比例及其潜在结果的条件均值，解构了混合分布。这些结果有助于理解其他看似合理的分析的缺陷。下面我回顾三个估计量，并假设假设 21.1–21.3 成立。

1. **按处理分析（As-treated analysis）**比较了接受处理和对照的单元的结果均值，得到

$$
\tau_ {\mathrm{AT}} = E (Y \mid D = 1) - E (Y \mid D = 0).
$$

证明

$$
\tau_ {\mathrm{AT}} = \frac {\pi_ {\mathrm{a}} \mu_ {\mathrm{a}} + \mathrm{pr} (Z = 1) \pi_ {\mathrm{c}} \mu_ {1 \mathrm{c}}}{\mathrm{pr} (D = 1)} - \frac {\pi_ {\mathrm{n}} \mu_ {\mathrm{n}} + \mathrm{pr} (Z = 0) \pi_ {\mathrm{c}} \mu_ {0 \mathrm{c}}}{\mathrm{pr} (D = 0)}.
$$

2. **符合方案分析（Per-protocol analysis）**比较了在治疗组和对照组中遵循所分配治疗的单元，得到

$$
\tau_ {\mathrm{PP}} = E (Y \mid Z = 1, D = 1) - E (Y \mid Z = 0, D = 0).
$$

证明

$$
\tau_ {\mathrm{pp}} = \frac {\pi_ {\mathrm{a}} \mu_ {\mathrm{a}} + \pi_ {\mathrm{c}} \mu_ {\mathrm{1c}}}{\pi_ {\mathrm{a}} + \pi_ {\mathrm{c}}} - \frac {\pi_ {\mathrm{n}} \mu_ {\mathrm{n}} + \pi_ {\mathrm{c}} \mu_ {\mathrm{0c}}}{\pi_ {\mathrm{n}} + \pi_ {\mathrm{c}}}.
$$

3. 我们可能还希望比较接受处理和对照的单元的结果，并以其治疗分配为条件，得到

$$
\tau_ {Z = 1} = E (Y \mid Z = 1, D = 1) - E (Y \mid Z = 1, D = 0),
$$

$$
\tau_ {Z = 0} = E (Y \mid Z = 0, D = 1) - E (Y \mid Z = 0, D = 0).
$$

证明它们简化为

$$
\tau_ {Z = 1} = \frac {\pi_ {\mathrm{a}} \mu_ {\mathrm{a}} + \pi_ {\mathrm{c}} \mu_ {\mathrm{1c}}}{\pi_ {\mathrm{a}} + \pi_ {\mathrm{c}}} - \mu_ {\mathrm{n}}, \quad \tau_ {Z = 0} = \mu_ {\mathrm{a}} - \frac {\pi_ {\mathrm{n}} \mu_ {\mathrm{n}} + \pi_ {\mathrm{c}} \mu_ {\mathrm{0c}}}{\pi_ {\mathrm{n}} + \pi_ {\mathrm{c}}}.
$$

## 22.8 总体平均因果效应的界限（Bounds on the average causal effect on the whole population）

基于 21.6 节的符号扩展 22.1 节的讨论。使用潜在结果 $Y (d)$ ，将所接受处理对结果的平均因果效应定义为

$$
\delta = E \{Y (d = 1) - Y (d = 0) \},
$$

并将 $\mu _ { d u }$ 的定义修改为

$$
m _ {d u} = E \{Y (d) \mid U = u \}, \quad (z = 0, 1; u = \mathrm{a,n,c}).
$$

它们满足

$$
\delta = \sum_ {u = \mathrm{a}, \mathrm{n}, \mathrm{c}} \pi_ {u} (m _ {1 u} - m _ {0 u}).
$$

## 276 解构混合分布与工具变量不等式（Disentangle Mixture Distributions and Instrumental Variable Inequalities）

22.1 节识别了 $\pi _ { \mathrm { a } } , \pi _ { \mathrm { n } } , \pi _ { \mathrm { c } } , m _ { 1 \mathrm { a } } = \mu _ { 1 \mathrm { a } } , m _ { 0 \mathrm { n } } = \mu _ { 0 \mathrm { n } } , m _ { 1 \mathrm { c } } = \mu _ { 1 \mathrm { c } }$ 和 $m _ { 0 \mathrm { c } } = \mu _ { 0 \mathrm { c } }$ 。但数据不包含关于 $m _ { \mathrm { 0 a } }$ 和 $m _ { 1 \mathrm { n } }$ 的任何信息。因此，我们无法识别 $\delta$ 。对于有界结果，我们可以界定 $\delta$ 。证明以下结果：

**定理 22.4** 在假设 $\it { 2 1 . 2 \mathrm { - } 2 1 . 4 }$ 下，结果有界于 $[ y , { \overline { { y } } } ]$ ，我们有 $\underline { { { \delta } } } \le \delta \le \overline { { { \delta } } }$ ，其中

$$
\underline {{\delta}} = \delta^ {\prime} - \bar {y} \operatorname{pr} (D = 1 \mid Z = 0) + \underline {{y}} \operatorname{pr} (D = 0 \mid Z = 1)
$$

且

$$
\overline {{{{\delta}}}} = \delta^ {\prime} - \underline {{{{y}}}} \mathrm{pr} (D = 1 \mid Z = 0) + \overline {{{{y}}}} \mathrm{pr} (D = 0 \mid Z = 1)
$$

其中 $\delta ^ { \prime } = E ( D Y \mid Z = 1 ) - E ( Y - D Y \mid Z = 0 )$ 。

注：在二元结果的特例中，界限简化为

$$
\underline {{\delta}} = E (D Y \mid Z = 1) - E (D + Y - D Y \mid Z = 0)
$$

且

$$
\overline {{\delta}} = E (D Y + 1 - D \mid Z = 1) - E (Y - D Y \mid Z = 0).
$$

## 22.9 单侧不依从与统计推断（One-sided noncompliance and statistical inference）

考虑一个随机化鼓励设计，其中分配到对照组的单元无法获得处理。对于单元 $i$ ，令 $Z _ { i }$ 为二元分配处理，$D _ { i }$ 为二元接受处理，$Y _ { i }$ 为感兴趣的结果。当满足以下条件时，发生**单侧不依从（one-sided noncompliance）**：

$$
Z _ {i} = 0 \Longrightarrow D _ {i} = 0 (i = 1, \dots , n).
$$

假设假设 21.1 成立。

1. 在这种情况下，单调性假设 21.2 成立吗？这个问题中由 $\{ D _ { i } ( 1 ) , D _ { i } ( 0 ) \}$ 定义的潜在层有多少个？我们如何通过观测数据分布识别它们的比例？
2. 陈述排他性约束假设。在排他性约束下，证明 $E \{ Y ( z ) \mid U = u \}$ 可以由观测数据分布识别。给出所有可能的 $z$ 和 $u$ 值的公式。在这种情况下，我们如何识别依从者平均因果效应？
3. 如果我们观测到所有单元 $i$ 的处理前协变量 $X _ { i }$ ，我们如何使用协变量信息来提高依从者平均因果效应的估计效率？
4. 在假设 21.1 下，排他性约束假设 21.3 具有可检验的含义，即单侧不依从的 IV 不等式。陈述这些 IV 不等式。

5. Sommer 和 Zeger (1991) 提供了以下数据集：

<table><tr><td rowspan="2"></td><td colspan="2">Z=1</td><td colspan="2">Z=0</td></tr><tr><td>D=1</td><td>D=0</td><td>D=1</td><td>D=0</td></tr><tr><td>Y=1</td><td>9663</td><td>2385</td><td>0</td><td>11514</td></tr><tr><td>Y=0</td><td>12</td><td>34</td><td>0</td><td>74</td></tr></table>

重新分析它。

注：Bloom (1984) 首先讨论了单侧不依从，并提出了 IV 估计量 $\hat { \tau } _ { \mathrm { c } } = \hat { \tau } _ { Y } / \hat { \tau } _ { D }$ 。他的符号与本章节不同。

## 22.10 具有部分依从的单侧不依从（One-sided noncompliance with partial adherence）

Sanders 和 Karim (2021, 表 3) 报告了来自一项旨在估计精神障碍患者戒烟干预效果的随机化临床试验的以下数据。

<table><tr><td>分配的组别（group assigned）</td><td>接受的处理（treatment received）</td><td>组大小（group size）</td><td>阳性结果数（# positive outcomes）</td></tr><tr><td>对照组（Control）</td><td>无（None）</td><td>151</td><td>25</td></tr><tr><td>治疗组（Treatment）</td><td>无（None）</td><td>35</td><td>7</td></tr><tr><td>治疗组（Treatment）</td><td>部分（Partial）</td><td>42</td><td>17</td></tr><tr><td>治疗组（Treatment）</td><td>完全（Full）</td><td>70</td><td>40</td></tr></table>

接受的处理有三个层级定义如下：“完全”处理对应于参加所有 8 个治疗阶段，“部分”对应于参加 5 到 7 个阶段，“无”对应于少于 5 个阶段。结果定义为在三个月时测量的、相对于基线吸烟量减少 50% 或更多的二元指标。

在这个问题中，处理分配 $Z$ 是二元的，但接受的处理 $D$ 取三个值 0、0.5、1，分别对应“无”、“部分”和“完全”。三水平的 $D$ 带来了复杂性，但在对照分配下它只能为 0。在这个问题中，我们有多少个潜在层 $U = \{ D ( 1 ) , D ( 0 ) \}$ ？我们能识别它们的比例吗？

我们如何将排他性约束扩展到这个问题？可能感兴趣的因果效应是什么？我们能识别它们吗？

基于上述问题分析数据。

## 22.11 推荐阅读（Recommended reading）

Balke 和 Pearl (1997) 推导了更一般的 IV 不等式。
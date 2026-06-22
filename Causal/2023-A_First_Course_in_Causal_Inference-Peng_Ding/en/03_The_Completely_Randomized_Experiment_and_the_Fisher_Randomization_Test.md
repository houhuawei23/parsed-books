# The Completely Randomized Experiment and the Fisher Randomization Test

The potential outcomes framework has intrinsic connections with randomized experiments. Understanding causal inference with various randomized experiments is fundamental and quite helpful for understanding causal inference in more complicated non-experimental studies.

Part II of this book focuses on randomized experiments. This chapter focuses on the simplest experiment, the completely randomized experiment (CRE).

## 3.1 CRE

Consider an experiment with n units, with $n_{1}$ receiving the treatment and $n_{0}$ receiving the control. We can define the CRE based on its treatment assignment mechanism $^{1}$ .

<!-- footnote -->

> - $^{1}$ Readers may think that a CRE has $Z_{i}$ 's as independent and identically distributed (IID) Bernoulli random variables with probability $\pi$ , in which $n_{1}$ is a Binomial( $n,\pi$ ) random variable. This is called the Bernoulli randomized experiment (BRE), which reduces to the CRE if we condition on $(n_{1},n_{0})$ . I will give more details for the BRE in Problem 4.7 in Chapter 4.

<!-- footnote end -->

Definition 3.1 (CRE) A CRE has the treatment assignment mechanism:

$$
\operatorname{pr} (\mathbf {Z} = \mathbf {z}) = 1 \bigg / \binom{n}{n _ {1}},
$$

where $\boldsymbol{z} = (z_1, \ldots, z_n)$ satisfies $\sum_{i=1}^{n} z_i = n_1$ and $\sum_{i=1}^{n} (1 - z_i) = n_0$ .

In Definition 3.1, we view the potential outcome vector under treatment $\mathbf{Y}(1) = (Y_{1}(1), \ldots, Y_{n}(1))$ and the potential outcome vector under control $\mathbf{Y}(0) = (Y_{1}(0), \ldots, Y_{n}(0))$ are both fixed. Even if we view them as random, we can condition on them and the treatment assignment mechanism becomes

$$
\operatorname{pr} \{\boldsymbol {Z} = \boldsymbol {z} \mid \boldsymbol {Y} (1), \boldsymbol {Y} (0) \} = 1 \bigg / \binom{n}{n _ {1}}
$$

because $\mathbf{Z} \perp \{\mathbf{Y}(1), \mathbf{Y}(0)\}$ in a CRE. In a CRE, the treatment vector $\mathbf{Z}$ is from a random permutation of $n_1$ 1's and $n_0$ 0's.

In his seminal book Design of Experiments, Fisher (1935) pointed out the following advantages of randomization:

1. It creates comparable treatment and control groups on average.  
2. It serves as a “reasoned basis” for statistical inference.

Point 1 is intuitive because the random treatment assignment does not bias toward the treatment or the control. Most people understand point 1 well. Point 2 is more subtle. What Fisher meant is that randomization justifies a statistical test, which is now called the Fisher Randomization Test (FRT). This chapter illustrates the basic idea of the FRT under a CRE.

## 3.2 FRT

Fisher (1935) was interested in testing the following null hypothesis:

$$
H _ {0 \mathrm{F}}: Y _ {i} (1) = Y _ {i} (0) \text {   for   all   units   } i = 1, \dots , n.
$$

Rubin (1980) called it the sharp null hypothesis in the sense that it can determine all the potential outcomes based on the observed data: $\mathbf{Y}(1)=\mathbf{Y}(0)=\mathbf{Y}=(Y_{1},\ldots,Y_{n})$ , the vector of the observed outcomes. It is also called the strong null hypothesis (e.g., Wu and Ding, 2021).

Conceptually, under $H_{0F}$ , the FRT works for any test statistic

$$
T = T (\mathbf {Z}, \mathbf {Y}), \tag {3.1}
$$

which is a function of the observed data. The observed outcome vector Y is fixed under $H_{0F}$ , so the only random component in the test statistic T is the treatment vector Z. The experimenter determines the distribution of Z, which in turn determines the distribution of T under $H_{0F}$ . This is the basis for calculating the p-value. I will give more details below.

In a CRE, Z is uniform over the set

$$
\left\{\boldsymbol {z} ^ {1}, \dots , \boldsymbol {z} ^ {M} \right\}
$$

where $M = \binom{n}{n_{1}}$ , and the $z^{m}$ 's are all possible vectors with $n_{1}$ 1's and $n_{0}$ 0's. For instance, with n = 5 and $n_{1} = 3$ , we can enumerate $M = \binom{5}{3} = 10$ vectors as follows:

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

As a consequence, $T$ is uniform over the set (with possible duplications)

$$
\{T (\boldsymbol {z} ^ {1}, \boldsymbol {Y}), \dots , T (\boldsymbol {z} ^ {M}, \boldsymbol {Y}) \}.
$$

That is, the distribution of $T$ is known due to the design of the CRE. We will call this distribution of $T$ the randomization distribution.

If larger values are more extreme for T, we can use the following tail probability to measure the extremeness of the test statistic with respect to its randomization distribution:

$$
p _ {\mathrm{FRT}} = M ^ {- 1} \sum_ {m = 1} ^ {M} I \{T (\boldsymbol {z} ^ {m}, \boldsymbol {Y}) \geq T (\boldsymbol {Z}, \boldsymbol {Y}) \}, \tag {3.2}
$$

which is called the p-value by Fisher. Figure 3.1 illustrates the computational process of $p_{FRT}$ .

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

FIGURE 3.1: Illustration of the FRT

The p-value, $p_{FRT}$ , in (3.2) works for any choice of test statistic and any outcome-generating process. It also extends naturally to any experiments, which will be a topic repeatedly discussed in the following chapters. Importantly, it is finite-sample exact in the sense $^{2}$ that under $H_{0F}$ ,

$$
\operatorname{pr} (p _ {\mathrm{FRT}} \leq u) \leq u \quad \text { for   all } \quad 0 \leq u \leq 1. \tag {3.3}
$$

In practice, M is often to large (e.g., with $n = 100, n_{1} = 50$ , we have $M > 10^{29}$ ), and it is computationally infeasible to enumerate all possible values of the treatment vector. We often approximate $p_{FRT}$ by Monte Carlo. To be more specific, we take simple random draws from the possible values of the treatment vector, or, equivalently, we randomly permute Z, and approximate $p_{FRT}$ by

$$
\hat {p} _ {\mathrm{FRT}} = R ^ {- 1} \sum_ {r = 1} ^ {R} I \{T (\boldsymbol {z} ^ {r}, \boldsymbol {Y}) \geq T (\boldsymbol {Z}, \boldsymbol {Y}) \}, \tag {3.4}
$$

where the $z^r$ 's the $R$ random permutations of $Z$ . The $p$ -value in (3.4) has Monte Carlo error decreasing fast with an increasing $R$ ; see Problem 3.2. Because the calculation of the $p$ -value in (3.4) involves permutations of $Z$ , the FRT is sometimes called the permutation test in the context of the CRE. However, the idea of FRT is more general than the permutation test in more complex experiments.

## 3.3 Canonical choices of the test statistic

From the above discussion, the FRT generates finite-sample exact p-value for any choice of test statistic. This is a feature of the FRT. However, this feature should not encourage arbitrary choice of the test statistic. Intuitively, we must choose test statistics that give information for the possible violations of $H_{0F}$ . Below I will review some canonical choices.

Example 3.1 (difference-in-means) The difference-in-means statistic is

$$
\hat {\tau} = \hat {\bar {Y}} (1) - \hat {\bar {Y}} (0)
$$

where

$$
\hat {\bar {Y}} (1) = n _ {1} ^ {- 1} \sum_ {Z _ {i} = 1} Y _ {i} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i}
$$

is the sample mean of the outcomes under the treatment and

$$
\hat {\bar {Y}} (0) = n _ {0} ^ {- 1} \sum_ {Z _ {i} = 0} Y _ {i} = n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) Y _ {i}
$$

is the sample mean of the outcomes under the control, respectively. Under $H_{0F}$ , it has mean

$$
E (\hat {\tau}) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} E (Z _ {i}) Y _ {i} - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} E (1 - Z _ {i}) Y _ {i} = 0
$$

and variance

$$
\begin{array}{l} \operatorname{var} (\hat {\tau}) = \operatorname{var} \left\{n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i} - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) Y _ {i} \right\} \\ = \quad \operatorname{var} \left(\frac {n}{n _ {0}} \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i}\right) \\ = _ {*} \frac {n ^ {2}}{n _ {0} ^ {2}} \left(1 - \frac {n _ {1}}{n}\right) \frac {s ^ {2}}{n _ {1}} \\ { = } { \frac { n } { n _ { 1 } n _ { 0 } } s ^ { 2 } , } \\ \end{array}
$$

where $= _{*}$ follows from Lemma A3.2 for simple random sampling with

$$
\bar {Y} = n ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i}, \quad s ^ {2} = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} (Y _ {i} - \bar {Y}) ^ {2}.
$$

Furthermore, the randomization distribution of $\hat{\tau}$ is approximately Normal due to the finite population central limit theorem in Lemma A3.4:

$$
\frac {\hat {\tau}}{\sqrt {\frac {n}{n _ {1} n _ {0}} s ^ {2}}} \rightarrow \mathrm{N} (0, 1) \tag {3.5}
$$

in distribution. Since $s^{2}$ is fixed under $H_{0F}$ , it is equivalent to use

$$
\frac {\hat {\tau}}{\sqrt {\frac {n}{n _ {1} n _ {0}} s ^ {2}}}
$$

as the test statistic in the FRT, which is asymptotically Normal as shown above. Then we can calculate an approximate p-value.

The observed data are $\{Y_i:Z_i = 1\}$ and $\{Y_i:Z_i = 0\}$ , so the problem is essentially a two-sample problem. Under the assumption of IID Normal outcomes (see Section A1.4.1), the classic two-sample $t$ -test assuming equal variance is based on

$$
\frac {\hat {\tau}}{\sqrt {\frac {n}{n _ {1} n _ {0} (n - 2)} \left[ \sum_ {Z _ {i} = 1} \{Y _ {i} - \hat {\bar {Y}} (1) \} ^ {2} + \sum_ {Z _ {i} = 0} \{Y _ {i} - \hat {\bar {Y}} (0) \} ^ {2} \right]}} \sim t _ {n - 2}. \tag {3.6}
$$

Based on some algebra (see Problem 3.8), we have the expansion

$$
(n - 1) s ^ {2} = \sum_ {Z _ {i} = 1} \left\{Y _ {i} - \hat {\bar {Y}} (1) \right\} ^ {2} + \sum_ {Z _ {i} = 0} \left\{Y _ {i} - \hat {\bar {Y}} (0) \right\} ^ {2} + \frac {n _ {1} n _ {0}}{n} \hat {\tau} ^ {2}. \tag {3.7}
$$

With a large sample size n, we can ignore the difference between $N(0,1)$ and $t_{n-2}$ and the difference between n-1 and n-2. Moreover, under $H_{0F}$ , $\hat{\tau}$ converges to zero in probability, so $n_{1}n_{0}/n\hat{\tau}^{2}$ can be ignored asymptotically. Therefore, under $H_{0F}$ , the approximate p-value in Example 3.1 is close to the p-value from the classic two-sample t-test assuming equal variance, which can be calculated by t.test with var.equal = TRUE. Under alternative hypotheses with nonzero $\tau$ , the additional term $\frac{n_{1}n_{0}}{n}\hat{\tau}^{2}$ in the above expansion can make the FRT less powerful than the usual t-test.

Based on the above discussion, the FRT with $\hat{\tau}$ effectively uses a pooled variance ignoring the heteroskedasticity between these two groups. In classical statistics, the two-sample problem with heteroskedastic Normal outcomes is called the Behrens–Fisher problem (see Section A1.4.1). In the Behrens–Fisher problem, a standard choice of the test statistic is the studentized statistic below.

Example 3.2 (studentized statistic) The studentized statistic is

$$
t _ {\mathrm{unequal}} = \frac {\hat {\bar {Y}} (1) - \hat {\bar {Y}} (0)}{\sqrt {\frac {\hat {S} ^ {2} (1)}{n _ {1}} + \frac {\hat {S} ^ {2} (0)}{n _ {0}}}},
$$

where

$$
\hat {S} ^ {2} (1) = (n _ {1} - 1) ^ {- 1} \sum_ {Z _ {i} = 1} \{Y _ {i} - \hat {\bar {Y}} (1) \} ^ {2}, \quad \hat {S} ^ {2} (0) = (n _ {0} - 1) ^ {- 1} \sum_ {Z _ {i} = 0} \{Y _ {i} - \hat {\bar {Y}} (0) \} ^ {2}
$$

are the sample variances of the observed outcomes under the treatment and control, respectively. Under $H_{0F}$ , the finite population central limit theorem again implies that t is asymptotically Normal:

$$
t \to \mathrm{N} (0, 1)
$$

in distribution. Then we can calculate an approximate p-value which is close to the p-value from t.test with var.equal = FALSE.

An extremely important point is that the FRT justifies the traditional t-tests using t.test with either var.equal = TRUE or var.equal = FALSE, even if the underlying distributions are not Normal. Standard statistics textbooks motivate the t-tests based on the Normality assumption, but the assumption is too strong. Fortunately, the t-test procedures can still be used as long as the finite population central limit theorems hold. Even if we do not believe the central limit theorems, we can still use $\hat{\tau}$ and t as test statistics in the FRT to obtain finite-sample exact p-values.

We will motivate this studentized statistic from another perspective in Chapter 8. The theory shows that using $t$ in FRT is more robust to heteroskedasticity across the two groups.

The following test statistic is robust to outliers resulting from heavy-tailed outcome data.

Example 3.3 (Wilcoxon rank sum) The difference-in-means statistic uses the original outcomes, and its sampling distribution depends on the second moments of the outcomes. This makes it sensitive to outliers. Another popular test statistic is based on the ranks of the pooled observed outcomes. Let $R_{i}$ denote the rank of $Y_{i}$ in the pooled samples Y:

$$
R _ {i} = \# \{j: Y _ {j} \leq Y _ {i} \}.
$$

The Wilcoxon rank sum statistic is the sum of the ranks under treatment:

$$
W = \sum_ {i = 1} ^ {n} Z _ {i} R _ {i}.
$$

For algebraic simplicity, we assume that there are no ties in the outcomes, although the FRT can be applied regardless of the existence of ties. For the case with ties, see Lehmann (1975, Chapter 1 Section 4). Because the sum of the ranks of the pooled samples are fixed at $1 + 2 + \cdots + n = n(n + 1)/2$ , the Wilcoxon statistic is equivalent to the difference in the means of the ranks under treatment and control. Under $H_{0F}$ , the $R_{i}$ 's are fixed, so W has mean

$$
E (W) = \sum_ {i = 1} ^ {n} E (Z _ {i}) R _ {i} = \frac {n _ {1}}{n} \sum_ {i = 1} ^ {n} i = \frac {n _ {1}}{n} \times \frac {n (n + 1)}{2} = \frac {n _ {1} (n + 1)}{2}
$$

and variance

$$
\begin{array}{l} \operatorname{var} (W) = \operatorname{var} \left(n _ {1} \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} R _ {i}\right) \\ = _ {*} n _ {1} ^ {2} \left(1 - \frac {n _ {1}}{n}\right) \frac {1}{n _ {1}} \frac {1}{n - 1} \sum_ {i = 1} ^ {n} \left(R _ {i} - \frac {n + 1}{2}\right) ^ {2} \\ = \frac {n _ {1} n _ {0}}{n (n - 1)} \sum_ {i = 1} ^ {n} \left(i - \frac {n + 1}{2}\right) ^ {2} \\ = \frac {n _ {1} n _ {0}}{n (n - 1)} \left\{\sum_ {i = 1} ^ {n} i ^ {2} - n \left(\frac {n + 1}{2}\right) ^ {2} \right\} \\ = \frac {n _ {1} n _ {0}}{n (n - 1)} \left\{\frac {n (n + 1) (2 n + 1)}{6} - n \left(\frac {n + 1}{2}\right) ^ {2} \right\} \\ = \frac {n _ {1} n _ {0} (n + 1)}{1 2}, \\ \end{array}
$$

where $=_{*}$ follows from Lemma A3.2. Furthermore, under $H_{0\mathrm{F}}$ , the finite population central limit theorem ensures that the randomization distribution of $\widehat{\tau}$ is approximately Normal:

$$
\frac {\sum_ {i = 1} ^ {n} Z _ {i} R _ {i} - \frac {n _ {1} (n + 1)}{2}}{\sqrt {\frac {n _ {1} n _ {0} (n + 1)}{1 2}}} \rightarrow \mathrm{N} (0, 1) \tag {3.8}
$$

in distribution. Based on (3.8), we can conduct an asymptotic test. In R, the function wilcox.test can compute both exact and asymptotic p-values based on the statistic $W - n_{1}(n_{1} + 1)/2$ . Based on some asymptotic analyses, Lehmann (1975) showed that the FRT using W has reasonable powers over a wide range of data generating processes.

Example 3.4 (Kolmogorov–Smirnov statistic) The treatment may affect the outcome in different ways. It seems natural to summarize the treatment outcomes and control outcomes based on the empirical distributions:

$$
\hat {F} _ {1} (y) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} I (Y _ {i} \leq y), \quad \hat {F} _ {0} (y) = n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) I (Y _ {i} \leq y).
$$

Comparing these two empirical distributions yields the famous Kolmogorov-Smirnov statistic

$$
D = \max _ {y} \left| \hat {F} _ {1} (y) - \hat {F} _ {0} (y) \right|.
$$

It is a challenging mathematics problem to derive the distribution of $D$ . With large sample sizes, its distribution function converges to

$$
\mathrm{pr} \left(\frac {n _ {1} n _ {0}}{n} D \leq x\right)\rightarrow \frac {\sqrt {2 \pi}}{x} \sum_ {j = 1} ^ {\infty} e ^ {- (2 j - 1) ^ {2} \pi^ {2} / (8 x ^ {2})},
$$

based on which we calculate an asymptotic p-value (Van der Vaart, 2000). In R, ks.test can compute both exact and asymptotic p-values.

## 3.4 A case study of the LaLonde experimental data

I use LaLonde (1986)’s experimental data to illustrate the FRT. The data are available in the Matching package (Sekhon, 2011):

Figure 3.2 shows the histograms of the outcomes under the treatment and control.

```txt
> library (Matching)
> data (lalonde)
> z = lalonde$treat
> y = lalonde$re78
```

The following code computes the observed values of the test statistics using existing functions:

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

By randomly permuting the treatment vector, we can obtain the Monte Carlo approximation of the randomization distributions of the test statistics, stored in four vectors Tauhat, Student, Wilcox, and Ks.

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

The one-sided p-values based on the FRT are all smaller than 0.05:

```txt
> exact.pv = c(mean(Tauhat >= tauhat),
+    mean(Student >= student),
+    mean(Wilcox >= W),
+    mean(Ks >= D))
> round(exact.pv, 3)
[1] 0.002 0.002 0.006 0.040
```

Without using Monte Carlo, we can also compute the asymptotic p-values which are all smaller than 0.05:

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

The differences between the p-values are due to the asymptotic approximations as well as the fact that the default choices for t.test and wilcox.test are two-sided tests.

Figure 3.3 shows the histograms of the randomization distributions of four test statistics, as well as their corresponding observed values. For the first three test statistics, the Normal approximations works quite well even though the underlying outcome data distribution is far from Normal. In general, a figure like Figure 3.3 can give very clear information for testing the sharp null hypothesis. Recently, Bind and Rubin (2020) proposes, in the title of their paper, that “when possible, report a Fisher-exact p-value and display its underlying null randomization distribution.”

## 3.5 Some history of randomized experiments and FRT

## 3.5.1 James Lind’s experiment

James Lind (1716—1794) was a Scottish doctor and a pioneer of naval hygiene in the Royal Navy. At his time, scurvy was a major cause of death among sailors. He conducted one of the earliest randomized experiments with a clear documentation of the details, and concluded that citrus fruits cured scurvy before the discovery of Vitamin C.

In Lind (1753), he described the following randomized experiment with 12 patients of scurvy assigned to 6 groups. With some simplifications, the 6 groups are:

1. two received a quart of cider every day;  
2. two received twenty-five drops of sulfuric acid three times every day;  
3. two received two spoonfuls of vinegar three times every day;  
4. two received half a pint of seawater every day;  
5. two received two oranges and one lemon every day;  
6. two received a spicy paste plus a drink of barley water every day.

After six days, patients in the fifth group recovered, but patients in other groups did not. If we simplify the treatment as

$$
Z _ {i} = 1 (\text { unit   } i \text {   received   citrus   fruits })
$$

and the outcome as

$$
Y _ {i} = 1 (\text { unit   } i \text {   recovered   after   six   days }),
$$

then we have a $2 \times 2$ table

<table><tr><td></td><td> $Y_i = 1$ </td><td> $Y_i = 0$ </td></tr><tr><td> $Z_i = 1$ </td><td>2</td><td>0</td></tr><tr><td> $Z_i = 0$ </td><td>0</td><td>10</td></tr></table>

This is the extremest possible $2 \times 2$ table we can observe under this experiment, and the data contain strong evidence for the positive effect of citrus fruits for curing scurvy. Statistically, how do we measure the strength of the evidence?

Following the logic of the FRT, if the treatment has no effect at all (under $H _ { \mathrm { 0 F } } )$ , the extreme $2 \times 2$ table will occur with probability

$$
\frac {1}{\binom {1 2} {2}} = \frac {1}{6 6} = 0. 0 1 5
$$

which is the $p _ { \mathrm { F R T } }$ . This seems a surprise under $H _ { \mathrm { 0 F } } \colon$ we can easily reject $H _ { \mathrm { 0 F } }$ at the level 0.05.

## 3.5.2 Lady tasting tea

Fisher (1935) described the following famous experiment of Lady Tasting $T e a ^ { 3 }$ . A lady claimed that she could tell the difference between the two ways of making milk tea: one with milk added first, and the other with tea added first. This might sound odd to most people. As a statistician, Fisher designed an experiment to test whether the lady could tell the difference between the two ways of making milk tea.

He made 8 cups of tea, 4 with milk added first and the other 4 four with tea added first. Then he presented these 8 cups of tea in a random order to the lady, and asked the lady to pick up the 4 with milk added first. The final experiment result can be summarized in the following $2 \times 2$ table

<table><tr><td></td><td>milk first (lady)</td><td>tea first (lady)</td><td>column sum</td></tr><tr><td>milk first (Fisher)</td><td>X</td><td>4 - X</td><td>4</td></tr><tr><td>tea first (Fisher)</td><td>4 - X</td><td>X</td><td>4</td></tr><tr><td>row sum</td><td>4</td><td>4</td><td>8</td></tr></table>

The X can be 0, 1, 2, 3, 4. In the real experiment, ${ \overline { { X = 4 } } } ,$ which is the most extreme data, strongly suggesting that the lady could tell the difference of the two ways of making milk tea. Again, how do we measure the strength of the evidence?

Under the null hypothesis that the lady could not tell the difference, only one of the $\binom { 8 } { 4 } = 7 0$ possible orders yields the $2 \times 2$ table with $X = 4 .$ . So the p-value is

$$
p _ {\mathrm{FRT}} = \frac {1}{7 0} = 0. 0 1 4.
$$

Given the significance level 0.05, we reject the null hypothesis.

## 3.5.3 Two Fisherian principles for experiments

In the above two examples in Sections 3.5.1 and 3.5.2, the $p _ { \mathrm { F R T } } \mathrm { ^ { * } s }$ are justified by the randomization of the experiments. This highlightsthe first Fisherian principle of experiments: randomization.

Moreover, the above two experiments are in some sense the smallest possible experiments that can yield statistically meaningful results. For instance, if Lind only assign one patient to each of the six groups, then the smallest p-value is

$$
\frac {1}{\binom {6} {1}} = \frac {1}{6} = 0. 1 6 7;
$$

if Fisher only made 6 cups of tea, 3 with milk added first and the other 3 four with tea added first, then the smallest p-value is

$$
\frac {1}{\binom {6} {3}} = \frac {1}{2 0} = 0. 0 5.
$$

We can never reject the null hypotheses at the level of 0.05. This highlights the second Fisherian principle of experiments: replications.

Chapter 5 will discuss the third Fisherian principle of experiments: blocking. $i n g .$

## 3.6 Discussion

## 3.6.1 Other sharp null hypotheses and confidence intervals

I focus on the sharp null hypothesis $H _ { \mathrm { 0 F } }$ above. In fact, the logic of the FRT also works for other sharp null hypotheses. For instance, we can test

$$
H _ {0} (\pmb {\tau}): Y _ {i} (1) - Y _ {i} (0) = \tau_ {i} \text { for   all } i = 1, \ldots , n
$$

for a known vector $\tau = ( \tau _ { 1 } , \dots , \tau _ { n } )$ . Because the individual causal effects are all known under $H _ { 0 } ( \tau )$ , we can impute all missing potential outcomes based on the observed data. With known potential outcomes, the distribution of any test statistic is completely determined by the treatment assignment mechanism, and therefore, we can compute the corresponding $p _ { \mathrm { { F R T } } }$ as a function of $\tau ,$ denoted by $p _ { \mathrm { F R T } } ( \tau )$ . If we can specify all possible ${ \boldsymbol { \tau } } { \mathrm { { s } } } ,$ then we can compute a series of $p _ { \mathrm { F R T } } ( \tau ) \mathrm { { : } }$ . By duality of hypothesis testing and confidence set (see Section A1.2.5), we can obtain a (1 − α)-level confidence set for the average causal effect:

$$
\left\{\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \tau_ {i}: p _ {\mathrm{FRT}} (\pmb {\tau}) \geq \alpha \right\}.
$$

Although this strategy is conceptually straightforward, it has practical complexities due to the large number of all possible $\tau _ { \mathrm { } } ^ { \prime } \mathrm { s } .$ . In the special case of a binary outcome, Rigdon and Hudgens (2015) and Li and Ding (2016) proposed some computationally feasible approaches to constructing confidence intervals for τ based on the FRT. For general unbounded outcomes, this strategy is often computationally infeasible.

A canonical simplification is to consider a subclass of the sharp null hypotheses with constant individual causal effects:

$$
H _ {0} (c): Y _ {i} (1) - Y _ {i} (0) = c \text { for all } i = 1, \ldots , n
$$

for a known constant c. Given c, we can compute $p _ { \mathrm { F R T } } ( c )$ . By duality, we can obtain a $( 1 - \alpha )$ )-level confidence set for the average causal effect:

$$
\{c: p _ {\mathrm{FRT}} (c) \geq \alpha \}.
$$

Because this procedure only involves one-dimensional search, it is computationally feasible. However, it is often criticized that the constant individual causal effect assumption is too strong which does not hold for a binary outcome in particular.

## 3.6.2 Other test statistics

The FRT is a general strategy that is applicable in any randomized experiments with any test statistic. I give several examples of test statistics in Section 3.3. In fact, the definition of a test statistic can be much more general. For instance, with pre-treatment covariate matrix X with the ith row being $X _ { i }$ for unit i $( i = 1 , \ldots , n ) ^ { \mathrm { ~ 4 ~ } }$ , we can allow the test statistic $T ( Z , Y , X )$ to be a function of the treatment vector, outcome vector, and the covariate matrix. Problem 3.6 gives an example.

## 3.6.3 Final remarks

For a general experiment, the probability distribution of $z$ is not uniform over all possible permutations of $n _ { 1 }$ 1’s and $n _ { 0 }$ 0’s. But its distribution is completely known by the experimenter. Therefore, we can always simulate its distribution which in turn implies the distribution of any test statistic under the sharp null hypothesis. A finite-sample exact p-value follows from (3.2). I will discuss other experiments in the subsequent chapters and I want to emphasize that the FRT works beyond the specific experiments discussed in this book.

The FRT works with any test statistic. However, this does answer the practical question of how to choose a test statistic in the data analysis. If the goal is to find surprise with respect to the sharp null hypothesis, it is desirable to choose a test statistic that yields high power under alternative hypotheses. In general, no test statistic can dominate others in terms of power because power depends on the alternative hypothesis. The four test statistics in Section 3.3 are motivated by different alternative hypotheses. For instance, τˆ and t are motivated by an alternative hypothesis with nonzero average treatment effect; W is motivated by an alternative hypothesis with a constant causal effect with outliers. Specifying a working alternative hypothesis is often helpful for constructing a test statistic although it does not have to be precise to guarantee the validity of the FRT. Problems 3.6 and 3.7 illustrate the idea of using a working alternative hypothesis or statistical model to construct test statistics.

## 3.7 Homework Problems

## 3.1 Exactness of $p _ { \mathrm { F R T } }$

Prove (3.2).

## 3.2 Monte Carlo error of $\hat { p } _ { \mathrm { F R T } }$

Given data, $p _ { \mathrm { F R T } }$ is a fixed number while its Monte Carlo estimator $\hat { p } _ { \mathrm { F R T } }$ as in (3.4) is random. Show that

$$
E _ {\mathrm{mc}} (\hat {p} _ {\mathrm{FRT}}) = p _ {\mathrm{FRT}}
$$

and

$$
\operatorname{var} _ {\mathrm{mc}} \left(\hat {p} _ {\mathrm{FRT}}\right) \leq \frac {1}{4 R},
$$

where the subscript “mc” signifies the randomness due to Monte Carlo, that is, $\hat { p } _ { \mathrm { F R T } }$ is random because $z ^ { r } \mathrm { { ^ { s } } }$ are R independent random draws from all possible values of $z$ .

Remark: $p _ { \mathrm { F R T } }$ is random because Z is random. But in this problem, we condition on data, so $p _ { \mathrm { F R T } }$ becomes a fixed number. $\hat { p } _ { \mathrm { F R T } }$ is random because the $z ^ { r }$ s are random permutations of $z .$ .

Problem 3.2 shows that $\hat { p } _ { \mathrm { F R T } }$ is unbiased for $p _ { \mathrm { F R T } }$ over the Monte Carlo randomness and gives an upper bound on the variance of $\hat { p } _ { \mathrm { F R T } }$ . Luo et al. (2021, Theorem 2) gives a more delicate bound on the Monte Carlo error.

## 3.3 A finite-sample valid Monte Carlo approximation $o f p _ { \mathrm { F R T } }$

Although $\hat { p } _ { \mathrm { F R T } }$ is unbiased for $p _ { \mathrm { F R T } }$ , it may not be a valid p-value in the sense that $\mathrm { p r } ( \hat { p } _ { \mathrm { F R T } } \leq u ) \leq u$ for all $u \in ( 0 , 1 )$ due to Monte Carlo error with a finite R. The following modified Monte Carlo approximation is. Phipson and Smyth (2010) pointed out this trick in the permutation test.

Define

$$
\tilde {p} _ {\mathrm{FRT}} = \frac {1 + \sum_ {r = 1} ^ {R} I \{T (\boldsymbol {z} ^ {r} , \boldsymbol {Y}) \geq T (\boldsymbol {Z} , \boldsymbol {Y}) \}}{1 + R}
$$

where the $z ^ { r } \mathrm { { ^ { s } } }$ the R random permutations of Z. Show that with an arbitrary $R ,$ the Monte Carlo approximation $\tilde { p } _ { \mathrm { F R T } }$ is always a finite-sample valid p-value in the sense that $\mathrm { p r } ( \tilde { p } _ { \mathrm { F R T } } \leq u ) \leq u$ for all $u \in ( 0 , 1 )$ .

Hint: You can use the following two basic probability results to prove the claim in Problem 3.3. First, for two Binomial random variables $X _ { 1 } \sim$ Binomial $( R , p _ { 1 } )$ and $X _ { 2 } \sim$ Binomia $. ( R , p _ { 2 } )$ with $p _ { 1 } \geq p _ { 2 }$ , we have $\mathrm { p r } ( X _ { 1 } \leq$ $x ) \ \leq \ \operatorname { p r } ( X _ { 2 } \ \leq \ x )$ for all x. Second, if $\begin{array} { r } { p \ \sim \ \mathrm { U n i f o r m } ( 0 , 1 ) } \end{array}$ and $X \ \parallel$ $p \sim$ Binomia $\left( R , p \right)$ , then, marginally, X is a uniform random variable over $\{ 0 , 1 , \ldots , R \}$ .

## 3.4 Fisher’s exact test

Consider a CRE with a binary outcome, with data summarized in the following $2 \times 2$ table:

<table><tr><td></td><td>$ Y = 1 $</td><td>$ Y = 0 $</td><td>total</td></tr><tr><td>$ Z = 1 $</td><td>$ n_{11} $</td><td>$ n_{10} $</td><td>$ n_{1} $</td></tr><tr><td>$ Z = 0 $</td><td>$ n_{01} $</td><td>$ n_{00} $</td><td>$ n_{0} $</td></tr></table>

Under $H _ { \mathrm { 0 F } } .$ , show that any test statistic is a function of $n _ { 1 1 }$ and other nonrandom fixed constants, and the exact distribution of $n _ { 1 1 }$ is Hypergeometric. Specify the parameters for the Hypergeometric distribution.

Remark: Barnard (1947) and Ding and Dasgupta (2016) pointed out the equivalence of Fisher’s exact test (reviewed in Section A1.3.1) and the FRT under a CRE with a binary outcome.

## 3.5 More details for lady tasting tea

Recall Section 3.5.2. Calculate $\operatorname { p r } ( X = k )$ for k = 0, 1, 2, 3, 4.

## 3.6 Covariate-adjusted FRT

This problem gives more details for Section 3.6.2.

Section 3.4 re-analyzed the LaLonde experimental data using the FRT. The R code FRTLalonde.R implemented the FRT with four test statistics. With additional covariates, the FRT can be more general with at least the following two additional strategies. Under the potential outcomes framework, all potential outcomes and covariates are fixed numbers.

First, we can use test statistics based on residuals from the linear regression. Run a linear regression of the outcomes on the covariates, and obtain the residuals $( { \mathrm { i . e . } }$ , treat the residuals as the pseudo “outcomes”). Then define the four test statistics based on the residuals. Conduct the FRT using these four new test statistics. Report the corresponding p-values.

Second, we can define the test statistic as the coefficient in the linear regression of the outcomes on the treatment and covariates. Conduct the FRT using this test statistic. Report the corresponding p-value.

Why are the five p-values from the above two strategies finite-sample exact? Justify them.

## 3.7 FRT with a generalized linear model

Use the same dataset as Problem 3.6 but change the outcome to a binary indicator whether re78 is positive or not. Run logistic regression of the outcome on the treatment and covariates. Is the coefficient of the treatment significant and what is the p-value? Calculate the p-value from the FRT with the coefficient of the treatment as the test statistic.

## 3.8 An algebraic detail

Verify (3.7)

## 3.9 Recommended reading

Bind and Rubin (2020) is a recently paper advocating the use of p-values as well as the display of the corresponding randomization distributions in analyzing complex experiments.

## 4
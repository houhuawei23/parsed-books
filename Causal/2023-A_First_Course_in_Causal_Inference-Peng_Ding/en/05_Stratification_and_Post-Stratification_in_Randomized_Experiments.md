# Stratification and Post-Stratification in Randomized Experiments

Block what you can and randomize what you cannot.

— George Box

This is the second most famous quote from George Box1. This chapter will explain its meaning.

## 5.1 Stratification

A CRE may generate an undesired treatment allocation. Let us start with a completely randomized experiment with a discrete covariate $X _ { i } \in \{ 1 , \ldots , K \}$ , and define $n _ { [ k ] } = \# \{ i : X _ { i } = k \}$ and $\pi _ { [ k ] } = n _ { [ k ] } / n$ as the number and proportion of units in stratum $ { k } ( k = 1 , \ldots { \dot { , } } K )$ . A CRE assigns $n _ { 1 }$ units to the treatment group and $n _ { 0 }$ units to the control group, which results in

$$
n _ {[ k ] 1} = \# \{i: X _ {i} = k, Z _ {i} = 1 \}, \quad n _ {[ k ] 0} = \# \{i: X _ {i} = k, Z _ {i} = 0 \}
$$

units in the treatment and control groups within stratum k. With positive probability, $n _ { [ k ] 1 } \mathrm { ~ o r ~ } n _ { [ k ] 0 }$ is zero for some $k ,$ that is, it is possible that some strata only have treated or control units. Even none of the $n _ { [ k ] 1 } \mathrm { ' s }$ or $n _ { [ k ] 0 } \mathrm { ^ { * } s }$ are zero, with high probability

$$
\frac {n _ {[ k ] 1}}{n _ {1}} - \frac {n _ {[ k ] 0}}{n _ {0}} \neq 0, \tag {5.1}
$$

and the magnitude can be quite large. So the proportions of units in stratum k are different across the treatment and control groups although on average their difference is zero:

$$
\begin{array}{l} E \left(\frac {n _ {[ k ] 1}}{n _ {1}} - \frac {n _ {[ k ] 0}}{n _ {0}}\right) \\ = E \left\{n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} 1 (X _ {i} = k) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) 1 (X _ {i} = k) \right\} \\ = 0. \\ \end{array}
$$

When $n _ { [ k ] 1 } / n _ { 1 } - n _ { [ k ] 0 } / n _ { 0 }$ is large for some strata with $X = k$ , the treatment and control groups have undesirable covariate imbalance. Such covariate imbalance deteriorates the quality of the experiment, making it difficult to interpret the results of the experiment since the difference in the outcomes may be attributed to the treatment or the covariate imbalance.

How can we actively avoid covariate imbalance in the experiment? We can fix the $n _ { [ k ] 1 } \mathrm { { } ^ { \circ } s }$ or $n _ { [ k ] 0 } \mathrm { ^ { * } s }$ in advance and conduct stratified randomized experiments (SRE).

Definition 5.1 (SRE) We conduct K independent CREs within the K strata of a discrete covariate X.

In agricultural experiments, the SRE is also called the randomized block design, with the strata called the blocks. Analogously, stratified randomization is also called block randomization. The total number of randomizations in an SRE equals

$$
\prod_ {k = 1} ^ {K} \binom{n _ {[ k ]}}{n _ {[ k ] 1}},
$$

and each feasible randomization has equal probability. Within stratum $k ,$ the proportion of units receiving the treatment is

$$
e _ {[ k ]} = \frac {n _ {[ k ] 1}}{n _ {[ k ]}},
$$

which is also called the propensity score, a conceptual that will play a central role in Part III of this book. An SRE is different from a CRE: first, all feasible randomizations in an SRE form a subset of all feasible randomizations in a $\mathrm { C R E } ,$ so

$$
\prod_ {k = 1} ^ {K} \binom{n _ {[ k ]}}{n _ {[ k ] 1}} <   \binom{n}{n _ {1}};
$$

second, $e _ { [ k ] }$ is fixed in an SRE but random in a CRE.

For every unit i, we have potential outcomes $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ , and individual causal effect $\tau _ { i } = Y _ { i } ( 1 ) – Y _ { i } ( 0 )$ . For stratum k, we have stratum-specific average causal effect

$$
\tau_ {[ k ]} = n _ {[ k ]} ^ {- 1} \sum_ {X _ {i} = k} \tau_ {i}.
$$

The average causal effect is

$$
\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \tau_ {i} = n ^ {- 1} \sum_ {k = 1} ^ {K} \sum_ {X _ {i} = k} \tau_ {i} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \tau_ {[ k ]},
$$

which is also the weighted average of the stratum-specific average causal effects.

If we are interested in $\tau _ { [ k ] }$ , then we can use the methods in Chapters 3 and 4 for the CRE within stratum k. Below I will discuss statistical inference for τ .

## 5.2 FRT

## 5.2.1 Theory

In parallel with the discussion of a CRE, I will start with the FRT in an SRE. The sharp null hypothesis is still

$$
H _ {0 \mathrm{F}}: Y _ {i} (1) = Y _ {i} (0) \text {   for   all   units   } i = 1, \dots , n.
$$

The fundamental idea of the FRT applies to any randomized experiment: we can use any test statistic which has a known distribution under $H _ { \mathrm { 0 F } }$ and the SRE. However, we must be careful with two subtle issues. First, when we simulate the treatment vector, we must permute the treatment indicators within strata of X. The resulting FRT is sometimes called the conditional randomization test or conditional permutation test. Second, we should choose test statistics that can reflect the nature of the SRE. Below I give some canonical choices of the test statistic.

Example 5.1 (Stratified estimator) Motivated by estimating τ , we can use the following stratified estimator in the FRT:

$$
\hat {\tau} _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \hat {\tau} _ {[ k ]},
$$

where

$$
\hat {\tau} _ {[ k ]} = n _ {[ k ] 1} ^ {- 1} \sum_ {i = 1} ^ {n} I (X _ {i} = k, Z _ {i} = 1) Y _ {i} - n _ {[ k ] 0} ^ {- 1} \sum_ {i = 1} ^ {n} I (X _ {i} = k, Z _ {i} = 0) Y _ {i}
$$

is the stratum-specific difference-in-means within stratum k.

Example 5.2 (Studentized stratified estimator) Motivated by the studentized statistic in the simple two-sample problem, we can use the following studentized statistic for the stratified estimator in the FRT:

$$
t _ {\mathrm{S}} = \frac {\hat {\tau} _ {\mathrm{S}}}{\sqrt {\hat {V} _ {\mathrm{S}}}},
$$

with

$$
\hat {V} _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \left(\frac {\hat {S} _ {[ k ]} ^ {2} (1)}{n _ {[ k ] 1}} + \frac {\hat {S} _ {[ k ]} ^ {2} (0)}{n _ {[ k ] 0}}\right)
$$

where $\hat { S } _ { [ k ] } ^ { 2 } ( 1 )$ and $\hat { S } _ { [ k ] } ^ { 2 } ( 0 )$ are the stratum-specific sample variances of the outcomes under treatment and control, respectively. The exact form of this statistic is motivated by the Neymanian perspective discussed in Section 5.3.

Example 5.3 (Combining Wilcoxon rank-sum statistics) We first compute the Wilcoxon rank sum statistic $W _ { [ k ] }$ within stratum k and then combine them as

$$
W _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} c _ {[ k ]} W _ {[ k ]}.
$$

Based on different asymptotic schemes and optimality criteria, Van Elteren (1960) proposed two weighting methods, one with

$$
c _ {[ k ]} = \frac {1}{n _ {[ k ] 1} n _ {[ k ] 0}},
$$

and the other with

$$
c _ {[ k ]} = \frac {1}{n _ {[ k ]} + 1}
$$

The motivations for these weights appear to be quite technical, and other choices of weights may also be reasonable.

Example 5.4 (Hodges and Lehmann (1962)’s aligned rank statistic) Van Elteren (1960)’s statistic works well with a few large strata. However, it does not work well with many small strata since it does not make enough comparisons, potentially losing information in the data. Hodges and Lehmann (1962) proposed a test statistic that makes more comparisons across strata after standardizing the outcomes. They suggested first centering the outcomes as

$$
\tilde {Y} _ {i} = Y _ {i} - \bar {Y} _ {[ k ]}
$$

with the stratum-specific mean ${ \bar { Y } } _ { [ k ] } = n _ { [ k ] } ^ { - 1 } \sum _ { X _ { i } = k } Y _ { i }$ if $X _ { i } = k$ , then obtaining the ranks $( \tilde { R } _ { 1 } , \ldots , \tilde { R } _ { n } )$ of the pooled outcomes $( { \tilde { Y } } _ { 1 } , \dots , { \tilde { Y } } _ { n } )$ , and finally constructing the test statistic

$$
\tilde {W} = \sum_ {i = 1} ^ {n} Z _ {i} \tilde {R} _ {i}.
$$

We can simulate the exact distributions of the above test statistics under the SRE. We can also calculate their means and variances and obtain the p-values based on Normal approximations.

After searching for a while, I failed to find detailed discussion of the Kolmogorov–Smirnov statistic for the SRE. Below is my proposal.

Example 5.5 (Kolmogorov–Smirnov statistic) We compute $D _ { [ k ] }$ , the maximum difference between the empirical distributions of the outcomes under treatment and control within stratum k. The final test statistic can be

$$
D _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} c _ {[ k ]} D _ {[ k ]}
$$

or

$$
D _ {\max} = \max _ {1 \leq k \leq K} c _ {[ k ]} D _ {[ k ]},
$$

where $c _ { [ k ] } = \sqrt { n _ { [ k ] 1 } n _ { [ k ] 0 } / n _ { [ k ] } }$ is motivated by the limiting distribution of $D _ { [ k ] }$ with $n _ { [ k ] 1 }$ and $n _ { [ k ] 0 }$ approach infinity (Van der Vaart, 2000). The statistics $D _ { \mathrm { { S } } }$ and $D _ { \mathrm { m a x } }$ are more appropriate when all strata have large sample size. Another reasonable choice is

$$
D = \max _ {y} \Big | \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \{\hat {F} _ {[ k ] 1} (y) - \hat {F} _ {[ k ] 0} (y) \} \Big |,
$$

where $\hat { F } _ { [ k ] 1 } ( y )$ and $\hat { F } _ { [ k ] 0 } ( y )$ are the stratum-specific empirical distribution functions of the outcomes under treatment and control, respectively. The statistic D is appropriate in both the cases with large strata and the cases with many small strata.

## 5.2.2 An application

The Penn Bonus experiment as an example to illustrate the FRT in the SRE. The dataset used by Koenker and Xiao (2002) is from a job training program stratified on quarter, with the outcome being the duration before employed.

```txt
penndata = read.table("Penn46_ascii.txt")
z = penndata$treatment
y = log(penndata$duration)
block = penndata$quarter
```

I will focus on $\mathrm { \hat { \tau } _ { S } }$ and $W _ { \mathrm { S } } .$ , and leave the FRT with other statistics as exercise. The following function computes $\mathrm { \hat { \tau } _ { S } }$ and $V { : }$

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

The following function generates a random treatment assignment in the SRE of the observed data:

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

Based on the above data and functions, we can easily simulate the randomization distributions of the test statistics (shown in Figure 5.1 with 104 Monte Carlo draws) and compute the p-values.

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

## 5.3 Neymanian inference

## 5.3.1 Point and interval estimation

Statistical inference for an SRE builds on the fact that it essentially consists of K independent CREs. Based on this, we can easily extend Neyman (1923)’s results to the SRE. Within stratum k, the difference-in-means ${ \hat { \tau } } _ { [ k ] }$ is unbiased for τ[k] with variance

$$
\mathrm{var} (\hat {\tau} _ {[ k ]}) = \frac {S _ {[ k ]} ^ {2} (1)}{n _ {[ k ] 1}} + \frac {S _ {[ k ]} ^ {2} (0)}{n _ {[ k ] 0}} - \frac {S _ {[ k ]} ^ {2} (\tau)}{n _ {[ k ]}},
$$

where $S _ { [ k ] } ^ { 2 } ( 1 ) , S _ { [ k ] } ^ { 2 } ( 0 )$ and $S _ { [ k ] } ^ { 2 } ( \tau )$ are the stratum-specific variances of potential outcomes and the individual treatment effects, respectively. Therefore, the

$\begin{array} { r } { \hat { \tau } _ { \mathrm { S } } = \sum _ { k = 1 } ^ { K } \pi _ { [ k ] } \hat { \tau } _ { [ k ] } } \end{array}$ $\begin{array} { r } { \tau = \sum _ { k = 1 } ^ { K } \pi _ { [ k ] } \tau _ { [ k ] } } \end{array}$ variance

$$
\mathrm{var} (\hat {\tau} _ {\mathrm{S}}) = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \mathrm{var} (\hat {\tau} _ {[ k ]}).
$$

If $n _ { [ k ] 1 } \geq 2$ and $n _ { [ k ] 0 } \geq 2$ , then we can obtain the sample variances $\hat { S } _ { [ k ] } ^ { 2 } ( 1 )$ and $\hat { S } _ { [ k ] } ^ { 2 } ( 0 )$ of the outcomes within stratum k and construct a conservative variance estimator

$$
\hat {V} _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \left(\frac {\hat {S} _ {[ k ]} ^ {2} (1)}{n _ {[ k ] 1}} + \frac {\hat {S} _ {[ k ]} ^ {2} (0)}{n _ {[ k ] 0}}\right),
$$

where $\hat { S } _ { [ k ] } ^ { 2 } ( 1 )$ and $\hat { S } _ { [ k ] } ^ { 2 } ( 0 )$ are the stratum-specific sample variances of the outcomes under treatment and control, respectively. Based on a Normal approximation of $\mathrm { \hat { \tau } _ { S } }$ , we can construct a Wald-type $1 - \alpha$ confidence interval for τ :

$$
\hat {\tau} _ {\mathrm{S}} \pm z _ {1 - \alpha / 2} \sqrt {\hat {V} _ {\mathrm{S}}}.
$$

From a hypothesis testing perspective, under $H _ { \mathrm { 0 N } } : \tau = 0 .$ , we can compare $t _ { \mathrm { S } } = \hat { \tau } _ { \mathrm { S } } / \sqrt { \hat { V } _ { \mathrm { S } } }$ with the standard Normal quantiles to obtain asymptotic $p \mathrm { - }$ values. The statistic $t _ { \mathrm { S } }$ has appeared in Example 5.2 for the FRT. Similar to the discussion for the CRE, using $t _ { \mathrm { S } }$ in the FRT yields finite-sample exact p-value under $H _ { \mathrm { 0 F } }$ and asymptotically valid $p \mathrm { - }$ -value under $H _ { \mathrm { 0 N } }$ . Wu and Ding (2021) provided $\mathrm { a }$ justification for this claim.

Here I omit the technical details for the central limit theorem of $\mathrm { \hat { \tau } _ { S } }$ . See Liu and Yang (2020) for a proof, which includes the two regimes with a few large strata and many small strata. I will illustrate this theoretical issues using a numerical example in Section 5.3.2.

## 5.3.2 Numerical examples

The following function computes the Neymanian point and variance estimators:

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

5.3 Neymanian inference

```txt
PiK[k] = length(zk)/length(z)
TauK[k] = mean(yk[zk==1]) - mean(yk[zk==0])
varK[k] = var(yk[zk==1])/sum(zk) +
    var(yk[zk==0])/sum(1 - zk)
}
return(c(sum(PiK*TauK), sum(PiK^2*varK)))
}
```

The first simulation setting has K = 5 and each stratum has 80 units. TauHat and VarHat are the point and variance estimators over 104 simulations.

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

The upper panel of Figure 5.2 shows the histogram of the point estimator, which is symmetric and bell-shaped around the true parameter. From the above, the average value of the variance estimator is almost identical to the variance of the estimators because the individual causal effects are constant.

The first simulation setting has K = 50 and each stratum has 8 units.

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

The lower panel of Figure 5.2 shows the histogram of the point estimator, which is symmetric and bell-shaped around the true parameter.

We finally use the Penn Bonus Experiment to illustrate the Neymanian inference in an SRE. Applying the function NeymanSRE to the dataset, we obtain:

```txt
> est = Neyman_SRE(z, y, block)
> est[1]
[1] -0.08990646
> sqrt(est[2])
[1] 0.03079775
```

So the job training program significantly shortens the duration time before employment.

## 5.3.3 Comparing the SRE and the CRE

What are the benefits of the SRE compared to the CRE? I have motivated the SRE from the covariate balance perspective. In addition, I will show that better covariate balance in turn results in better estimation precision of the average causal effect. To make a fair comparison, I assume that $e _ { [ k ] } = e$ for all k which ensures that $\hat { \tau } = \hat { \tau } _ { \mathrm { S } }$ . I leave the proof of this result as Problem 5.1.

We now compare the sampling variances. The classic analysis of variance technique motivates the decomposition of the total variance into the summation of the within-strata and between-strata variances, yielding

$$
\begin{array}{l} S ^ {2} (1) = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} \left\{Y _ {i} (1) - \bar {Y} (1) \right\} ^ {2} \\ = (n - 1) ^ {- 1} \sum_ {k = 1} ^ {K} \sum_ {X _ {i} = k} \left\{Y _ {i} (1) - \bar {Y} _ {[ k ]} (1) + \bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \right\} ^ {2} \\ = (n - 1) ^ {- 1} \sum_ {k = 1} ^ {K} \sum_ {X _ {i} = k} \left[ \left\{Y _ {i} (1) - \bar {Y} _ {[ k ]} (1) \right\} ^ {2} + \left\{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \right\} ^ {2} \right] \\ = \sum_ {k = 1} ^ {K} \left[ \frac {n _ {[ k ]} - 1}{n - 1} S _ {[ k ]} ^ {2} (1) + \frac {n _ {[ k ]}}{n - 1} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} ^ {2} \right], \\ \end{array}
$$

and similarly,

$$
S ^ {2} (0) = \sum_ {k = 1} ^ {K} \left[ \frac {n _ {[ k ]} - 1}{n - 1} S _ {[ k ]} ^ {2} (0) + \frac {n _ {[ k ]}}{n - 1} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} ^ {2} \right],
$$

$$
{S ^ {2} (\tau)} = {\sum_ {k = 1} ^ {K} \left[ \frac {n _ {[ k ]} - 1}{n - 1} S _ {[ k ]} ^ {2} (\tau) + \frac {n _ {[ k ]}}{n - 1} \{\tau_ {[ k ]} - \tau \} ^ {2} \right].}
$$

With large strata, the variance of the difference-in-means estimator under complete randomization is approximately

$$
\begin{array}{l} \mathrm{var} _ {\mathrm{CRE}} (\hat {\tau}) \\ = \frac {S ^ {2} (1)}{n _ {1}} + \frac {S ^ {2} (0)}{n _ {0}} - \frac {S ^ {2} (\tau)}{n} \\ \approx \sum_ {k = 1} ^ {K} \left[ \frac {\pi_ {[ k ]}}{n _ {1}} S _ {[ k ]} ^ {2} (1) + \frac {\pi_ {[ k ]}}{n _ {0}} S _ {[ k ]} ^ {2} (0) - \frac {\pi_ {[ k ]}}{n} S _ {[ k ]} ^ {2} (\tau) \right] \\ + \sum_ {k = 1} ^ {K} \left[ \frac {\pi_ {[ k ]}}{n _ {1}} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} ^ {2} + \frac {\pi_ {[ k ]}}{n _ {0}} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} ^ {2} - \frac {\pi_ {[ k ]}}{n} \{\tau_ {[ k ]} - \tau \} ^ {2} \right]. \\ \end{array}
$$

The constant propensity scores assumption ensures

$$
\pi_ {[ k ]} / n _ {[ k ] 1} = 1 / (n e), \quad \pi_ {[ k ]} / n _ {[ k ] 0} = 1 / \{n (1 - e) \}, \quad \pi_ {[ k ]} / n _ {[ k ]} = 1 / n,
$$

which allow us to rewrite the variance of $\mathrm { \hat { \tau } _ { S } }$ under the SRE as

$$
\begin{array}{l} \mathrm{var} _ {\mathrm{SRE}} (\hat {\tau} _ {\mathrm{S}}) = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \left[ \frac {S _ {[ k ]} ^ {2} (1)}{n _ {[ k ] 1}} + \frac {S _ {[ k ]} ^ {2} (0)}{n _ {[ k ] 0}} - \frac {S _ {[ k ]} ^ {2} (\tau)}{n _ {[ k ]}} \right] \\ = \sum_ {k = 1} ^ {K} \left[ \frac {\pi_ {[ k ]}}{n _ {1}} S _ {[ k ]} ^ {2} (1) + \frac {\pi_ {[ k ]}}{n _ {0}} S _ {[ k ]} ^ {2} (0) - \frac {\pi_ {[ k ]}}{n} S _ {[ k ]} ^ {2} (\tau) \right]. \\ \end{array}
$$

Approximately, the difference between varCRE(ˆτ ) and $\mathrm { v a r } _ { \mathrm { S R E } } \big ( \hat { \tau } _ { \mathrm { S } } \big )$ is

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \left[ \frac {\pi_ {[ k ]}}{n _ {1}} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} ^ {2} + \frac {\pi_ {[ k ]}}{n _ {0}} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} ^ {2} - \frac {\pi_ {[ k ]}}{n} (\tau_ {[ k ]} - \tau) ^ {2} \right] \\ = \sum_ {k = 1} ^ {K} \frac {\pi_ {[ k ]}}{n} \left\{\sqrt {\frac {n _ {0}}{n _ {1}}} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} + \sqrt {\frac {n _ {1}}{n _ {0}}} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} \right\} ^ {2} \geq 0, \\ \end{array}
$$

which is non-negative. The difference is zero only in the extreme case that

$$
\sqrt {\frac {n _ {0}}{n _ {1}}} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} + \sqrt {\frac {n _ {1}}{n _ {0}}} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} = 0
$$

for $k = 1 , \ldots , K$ . When the covariate is predictive to the potential outcomes, the above quantities are usually not all zeros, which ensure the efficiency gain of the SRE compared to the CRE. Only in the extreme cases that the covariate is not predictive at all, the large-sample efficiency gain is zero. In those cases, the SRE can even result in worse estimators in finite sample. The above discussion corroborates the quote from George Box at the beginning of this chapter.

I will end this section with several remarks. First, the above comparison is based on the sampling variance, and we can also compare the estimated variances under the SRE and the CRE. The results are similar. Second, increasing K improves efficiency, but this argument depends on the large strata assumption. So we face a tradeoff in practice. We cannot arbitrarily increase K, and the most extreme case is $n _ { [ k ] 1 } = n _ { [ k ] 0 } = 1$ , which is called the matched pair experiment and will be discussed later.

## 5.4 Post-stratification in a CRE

In a CRE with a discrete covariate X, the numbers of units receiving the treatment and control are random within stratum k. In a SRE, these numbers are fixed. But if we conduct conditional inference given n = {n[k]1, n[k]0}Kk=1, ${ \pmb n } = \{ n _ { [ k ] 1 } , n _ { [ k ] 0 } \} _ { k = 1 } ^ { K } ,$ then a CRE becomes a SRE. Mathematically, if none of the components of n are zero, then

$$
\mathrm{pr} _ {\mathrm{CRE}} (\boldsymbol {Z} = \boldsymbol {z} \mid \boldsymbol {n}) = \frac {\mathrm{pr} _ {\mathrm{CRE}} (\boldsymbol {Z} = \boldsymbol {z} , \boldsymbol {n})}{\mathrm{pr} _ {\mathrm{CRE}} (\boldsymbol {n})} = \frac {1}{\prod_ {k = 1} ^ {K} \binom {n _ {[ k ]}} {n _ {[ k ] 1}}}, \tag {5.2}
$$

that is, the conditional distribution of Z from a CRE given n is identical to the distribution of Z from an SRE. So conditional on n, we can analyze a CRE with a discrete covariate X in the same way as in a SRE. In particular, the FRT becomes a conditional FRT, and the Neymanian analysis becomes post-stratification:

$$
\hat {\tau} _ {\mathrm{PS}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \hat {\tau} _ {[ k ]},
$$

which has an identical form as $\mathrm { \hat { \tau } _ { S } }$ . The variance of $\mathrm { \hat { \tau } _ { P S } }$ conditioning on n is identical to the variance of $ { \hat { \tau } _ { \mathrm { S } } }$ under the SRE.

Hennessy et al. (2016) used simulation to show that the conditional FRT is often more powerful than the unconditional one. Miratrix et al. (2013) used theory to show that in many cases, post-stratification improves efficiency compared to ${ \hat { \tau } } .$ However, the simulation is based on limited number of data generating processes, and the theory assumes all strata are large enough. We cannot go too extreme in the conditional FRT or post-stratification because with a larger K it is more likely that some $n _ { [ k ] 1 } \mathrm { ~ o r ~ } n _ { [ k ] 0 }$ become zero. Small or zero values of $n _ { [ k ] 1 } \mathrm { ~ o r ~ } n _ { [ k ] 0 }$ greatly reduces the number of randomizations in the FRT, possibly reducing the power dramatically. The problem for the Neymanian counterpart is more salient because we cannot even define $\mathrm { \hat { \tau } _ { P S } }$ and the corresponding variance estimator.

Stratification uses X in the design stage and post-stratification uses X in the analysis stage. They are duals. Asymptotically, their difference is small with large strata (Miratrix et al., 2013).

## 5.4.1 Meinert et al. (1970)’s Example

We use the data from a randomized trial from Meinert et al. (1970), which were also used by Rothman et al. (2008). The treatment is tolbutamide and the control is a placebo.

<table><tr><td colspan="3">Age &lt; 55</td><td colspan="3">Age ≥ 55</td></tr><tr><td></td><td>Surviving</td><td>Dead</td><td></td><td>Surviving</td><td>Dead</td></tr><tr><td>Z = 1</td><td>98</td><td>8</td><td>Z = 1</td><td>76</td><td>22</td></tr><tr><td>Z = 0</td><td>115</td><td>5</td><td>Z = 0</td><td>69</td><td>16</td></tr><tr><td colspan="6">Total</td></tr><tr><td></td><td></td><td>Surviving</td><td>Dead</td><td></td><td></td></tr><tr><td></td><td>Z = 1</td><td>174</td><td>30</td><td></td><td></td></tr><tr><td></td><td>Z = 0</td><td>184</td><td>21</td><td></td><td></td></tr></table>

The following table shows the estimates for two strata separately, the poststratified estimator, and the crude estimator ignoring the binary covariate, as well as the corresponding standard errors.

<table><tr><td></td><td>stratum 1</td><td>stratum 2</td><td>post-stratification</td><td>crude</td></tr><tr><td>est</td><td>-0.034</td><td>-0.036</td><td>-0.035</td><td>-0.045</td></tr><tr><td>se</td><td>0.031</td><td>0.060</td><td>0.032</td><td>0.033</td></tr></table>

Although the crude estimator and the post-stratification estimator do not lead to fundamentally different results, the crude estimator is outside the range of the stratum-specific estimators while the post-stratification estimator is within the range.

## 5.4.2 Chong et al. (2016)’s Example

Chong et al. (2016) ran a randomized experiment in Peru to study the effect of supplemental iron pills on school performance. The experiment is stratified on classlevel. I will only use a subset of the original data.

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

The treatment and control group sizes vary across five strata:

```txt
> table(dat_physician$z,
+    dat_physician-class_level)
```

```txt
1 2 3 4 5
FALSE 15 19 16 12 10
TRUE 17 20 15 11 10
```

We can use the NeymanSRE function defined before to compute the stratified estimator and its estimated variance.

```erlang
tauS = with(dat_physician,
    Neyman_SRE(z, gradesq34, class_level))
```

An important additional covariate is the baseline anemic indicator which is quite important for predicting the outcome. Further conditioning the baseline anemic indicator, we have an experiment with 5 × 2 = 10 strata, with the treatment and control group sizes shown below.

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

Again we can use the NeymanSRE function defined before to compute the poststratified estimator and its estimated variance.

```txt
tauSPS = with(dat_physician,
    {
    sps = interaction(class_level, anemic_base_re)
    Neyman_SRE(z, gradesq34, sps)
    })
```

The following table compares these two estimators. The post-stratified estimator yields a much smaller p-value.

```txt
est se t.stat p.value stratify 0.406 0.202 2.005 0.045 stratify and post-stratify 0.463 0.190 2.434 0.015
```

This example illustrates that post-stratification can be used not only in the CRE but also in the SRE with additional discrete covariates.

## 5.5 Practical questions

How do we choose X to construct a SRE? Theoretically, X should be predictive to the potential outcomes. In some cases, the experimenter has enough background knowledge about the predictive covariates based on, for example, some pilot studies. Then the choice of X should be straightforward. In some other cases, this background knowledge may not be clear enough. Experimenters instead choose X based on logistic convenience, for example, X can be indicator for the study areas or the cohort of students.

The choose of K is a related problem. Theoretically, more stratification increases the estimation efficiency if all strata are large enough. However, extremely large K may even decrease the estimation efficiency. In simulation studies, we observe diminishing marginal returns of increasing K. Anecdotally, K = 5 often suffices for efficiency gain. Some experimenter prefers the most extreme version of the SRE with $K = n / 2$ . This results in the matched pair design, which will be discussed in Chapter 7 later.

Some experiments have multidimensional continuous covariates. Can the SRE still be used? If we have a pilot study, we can build a model for the potential outcome Y (0) given those covariates, and then we can choose X as a discretized version of the predictor $\hat { Y } ( 0 )$ . In general, if we do not have such a pilot study or we do not want to make ad hoc discretizations, we can use a more general strategy called rerandomization, which is the topic for Chapter 6.

## 5.6 Homework Problems

## 5.1 Consequence of the constant propensity score

Show that if $e _ { [ k ] } = e$ for all $k = 1 , \ldots , K$ , then $\hat { \tau } = \hat { \tau } _ { \mathrm { S } }$ .

## 5.2 Consquence of constant individual causal effects

Assume that the individual causal effects are constant $\tau _ { i } ~ = ~ \tau$ for all $i \ =$ $1 , \ldots , n .$ . Consider the following class of weighted estimator for τ :

$$
\hat {\tau} _ {w} = \sum_ {k = 1} ^ {K} w _ {[ k ]} \hat {\tau} _ {[ k ]},
$$

where $w _ { [ k ] } \geq 0$ for all k.

Find the condition on the $w _ { [ k ] }$ ’s such that $\hat { \tau } _ { w }$ is unbiased for τ . Among all unbiased estimators, find the one with the minimum variance.

## 5.3 FRT for the Project STAR data in Imbens and Rubin (2015)

Reanalyze the Project STAR data using the Fisher randomization test. Note that I use Z for the treatment indicator but Imbens and Rubin (2015) use W. Use $\hat { \tau } _ { \mathrm { S } } , V$ and the aligned rank statistic in the Fisher randomization test. Compare the p-values.

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

## 5.4 A multi-center trial

Gould (1998, Table 1) reported the following data from a multi-center trial:

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

## 5.6 Homework Problems

<table><tr><td>26</td><td>26</td><td>15</td><td>-3.48</td><td>5.98</td><td>15</td><td>-2.13</td><td>3.25</td><td>14</td><td>-3.71</td><td>5.30</td></tr><tr><td>27</td><td>27</td><td>11</td><td>-1.91</td><td>6.49</td><td>12</td><td>-1.33</td><td>4.40</td><td>11</td><td>-1.52</td><td>4.68</td></tr><tr><td>28</td><td>28</td><td>10</td><td>-2.66</td><td>3.80</td><td>10</td><td>-1.29</td><td>3.18</td><td>10</td><td>-4.70</td><td>3.43</td></tr><tr><td>29</td><td>29</td><td>13</td><td>-0.77</td><td>4.73</td><td>13</td><td>-2.31</td><td>3.88</td><td>13</td><td>-0.47</td><td>4.95</td></tr></table>

This is a SRE with centers being the strata. The trial was conducted to study the efficacy and tolerability of finasteride, a drug for treating benign prostatic hyperplasia. Within each of the 29 centers, patients were randomized into three arms: control, finasteride 1mg, and finasteride 5mg. The above dataset provides summary statistics for the outcome, which is the change from baseline in total symptom score. The total symptom score is the sum of the responses to nine questions (score 0 to 4) about symptoms pertaining to various aspects of impaired urinary ability. The meanings of the columns are:

1. center: number of the center;  
2. n0, n1, n5: sample sizes of the three arms;  
3. mean0, mean1, mean5: mean of the outcome;  
4. sd0, sd1, sd5: standard deviation of the outcome.

The individual-level outcomes are not reported so we cannot implement the FRT. However, the Neymanian inference only requires the summary statistics. Report the point estimators and variance estimators for comparing “finasteride 1mg” and “finasteride 5mg” to “control”, separately.

## 5.5 Data re-analyses

Re-analyze the LaLonde data used in Neymanlalonde.R. Conduct both Fisherian and Neymanian inferences.

The original experiment is a completely randomized experiment. Now we pretend that the original experiment is a stratified randomized experiment. First, re-analyze the data pretending that the experiment is stratified on the race (black, Hispanic or other). Second, re-analyze the data pretending that the experiment is stratified on marital status. Third, re-analyze the data pretending that the experiment is stratified on the indicator of high school diploma.

Compare with the results obtained under a completely randomized experiments.

## 5.6 Recommended reading

Miratrix et al. (2013) provided solid theory for post-stratification and compared it with stratification. A main theoretical result is that their difference is small asymptotically although they can differ in finite samples.

## 6
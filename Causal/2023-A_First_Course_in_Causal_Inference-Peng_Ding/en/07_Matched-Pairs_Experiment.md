# Matched-Pairs Experiment

The matched-pairs experiment (MPE) is the most extreme version of the SRE with only one treated unit and one control unit within each stratum. In this case, the strata are also called pairs. Although this type of experiment is a special case of the SRE discussed in Chapter 5, it has its own estimation and inference strategy. Moreover, it has many new features and it is closely related to the “matching” strategy in observational studies which will be covered in Chapter 15 later. So we discuss the MPE in this separate chapter.

## 7.1 Design of the experiment and potential outcomes

Consider an experiment with 2n units. If we have predictive covariates to the outcomes, we can pair units based on the similarity of covariates. With a scalar covariate, we can order units based on this covariate and then form pairs based on the adjacent units. With many covariates, we can define pairwise distances between units and then form pairs based on these distances. In this case, pair matching can be done using a greedy algorithm or an optimal nonbipartite matching algorithm. The greedy algorithm pairs the two units with the smallest distance, drop them from the pool of units, pair the two remaining units with the smallest distance, etc. The optimal nonbipartite matching algorithm divides the 2n units into n pairs of two units to minimize the sum of the within-pair distances. See Greevy et al. (2004) for more details of the computational aspect of the MPE. In this chapter, we assume that the pairs are formed based on the covariates, and discuss the subsequent design and analysis issues.

Let (i, j) index the unit $j$ in pair $i ,$ where $i = 1 , \ldots , n$ and $j = 1 , 2$ . Unit (i, j) has potential outcomes $Y _ { i j } ( 1 )$ and $Y _ { i j } ( 0 )$ under the treatment and control, respectively. Within each pair, we randomly assign one unit to receive the treatment and the other to receive the control. Let

$$
Z _ {i} = \left\{ \begin{array}{l l} 1, & \text { if   the   first   unit   receives   the   treatment }, \\ 0, & \text { if   the   second   unit   receives   the   treatment }. \end{array} \right.
$$

We can formally define MPE based on the treatment assignment mechanism.

Definition 7.1 (MPE) We have

$$
(Z _ {i}) _ {i = 1} ^ {n} \stackrel {{I I D}} {{\sim}} \text { Bernoulli } (1 / 2). \tag {7.1}
$$

The observed outcomes within pair i are

$$
Y _ {i 1} = Z _ {i} Y _ {i 1} (1) + (1 - Z _ {i}) Y _ {i 1} (0) = \left\{ \begin{array}{l l} Y _ {i 1} (1), & \text {if} Z _ {i} = 1; \\ Y _ {i 1} (0), & \text {if} Z _ {i} = 0; \end{array} \right.
$$

and

$$
Y _ {i 2} = Z _ {i} Y _ {i 2} (0) + (1 - Z _ {i}) Y _ {i 2} (1) = \left\{ \begin{array}{l l} Y _ {i 2} (0), & \text {if} Z _ {i} = 1; \\ Y _ {i 2} (1), & \text {if} Z _ {i} = 0. \end{array} \right.
$$

So the observed data are $( Z _ { i } , Y _ { i 1 } , Y _ { i 2 } ) _ { i = 1 } ^ { n }$ .

## 7.2 FRT

Similar to the discussion before, we can always use the FRT to test the sharp null hypothesis:

$$
H _ {0 \mathrm{F}}: Y _ {i j} (1) = Y _ {i j} (0) \text {   for   all   } i = 1, \dots n \text {   and   } j = 1, 2.
$$

When conducting the FRT, we need to simulate the distribution of $\left( Z _ { i } , \ldots , Z _ { n } \right)$ from (7.1). I will discuss some canonical choices of test statistics based on the within-pair differences between the treated and control outcomes:

$\begin{array} { r l } { \hat { \tau } _ { i } } & { { } = } \end{array}$ outcome under treatment − outcome under control (within pair i)

$$
= (2 Z _ {i} - 1) \left(Y _ {i 1} - Y _ {i 2}\right)
$$

$$
= S _ {i} (Y _ {i 1} - Y _ {i 2}),
$$

where the $S _ { i } ~ = ~ 2 Z _ { i } - 1$ are IID random signs with mean 0 and variance 1, for $i = 1 , \ldots , n$ . Since the pairs with zero $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ do not contribute to the randomization distribution, we drop those pairs in the discussion of the FRT.

Example 7.1 (paired t statistic) The average of the within-pair differences is

$$
\hat {\tau} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\tau} _ {i}.
$$

Under H0f, $H _ { \mathrm { 0 F } }$

$$
E (\hat {\tau}) = 0
$$

and

$$
\operatorname{var} (\hat {\tau}) = n ^ {- 2} \sum_ {i = 1} ^ {n} \operatorname{var} (\hat {\tau} _ {i}) = n ^ {- 2} \sum_ {i = 1} ^ {n} \operatorname{var} (S _ {i}) (Y _ {i 1} - Y _ {i 2}) ^ {2} = n ^ {- 2} \sum_ {i = 1} ^ {n} \hat {\tau} _ {i} ^ {2}.
$$

Based on the CLT for the sum of independent random variables, we have the Normal approximation:

$$
\frac {\hat {\tau}}{\sqrt {n ^ {- 2} \sum_ {i = 1} ^ {n} \hat {\tau} _ {i} ^ {2}}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

We can use this Normal approximation to construct an asymptotic test. Many standard test books suggest using the following paired t statistic in the $M P E ;$

$$
t _ {p a i r} = \frac {\hat {\tau}}{\sqrt {\{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\hat {\tau} _ {i} - \hat {\tau}) ^ {2}}},
$$

which is almost identical to τˆ with large n and small τˆ under $H _ { \mathrm { 0 F } }$ .

In classic statistics, the motivation for using $t _ { \mathrm { p a i r } }$ is under a different framework. When $\hat { \tau } _ { i } \stackrel { \mathrm { I I D } } { \sim } \mathrm { N } ( 0 , \sigma ^ { 2 } )$ , we can show that $t _ { \mathrm { p a i r } } \sim t ( n - 1 )$ , i.e., the exact distribution of $t _ { \mathrm { p a i r } }$ is t with degrees of freedom $n - 1$ , which is close to $\mathrm { { N } } ( 0 , 1 )$ with a large n. The R function t.test with paired=TRUE can implement this test. With a large n, these procedures give similar results. The discussion in Example 7.1 gives another justification of the classic paired t test without assuming the Normality of the data.

Example 7.2 (Wilcoxon sign-rank statistic) Based on the ranks $( R _ { 1 } , \ldots , R _ { n } )$ of $\big ( \vert \hat { \tau } _ { 1 } \vert , \dots , \vert \hat { \tau } _ { n } \vert \big )$ , we can define a test statistic

$$
W = \sum_ {i = 1} ^ {n} I (\hat {\tau} _ {i} > 0) R _ {i}.
$$

Under $H _ { \mathrm { 0 F } }$ ,

$$
E (W) = \frac {1}{2} \sum_ {i = 1} ^ {n} R _ {i} = \frac {1}{2} \sum_ {i = 1} ^ {n} i = \frac {n (n + 1)}{4}
$$

and

$$
\operatorname{var} (W) = \frac {1}{4} \sum_ {i = 1} ^ {n} R _ {i} ^ {2} = \frac {1}{4} \sum_ {i = 1} ^ {n} i ^ {2} = \frac {n (n + 1) (2 n + 1)}{2 4}.
$$

The CLT for the sum of independent random variables ensures the following Normal approximation:

$$
\frac {W - n (n + 1) / 4}{\sqrt {n (n + 1) (2 n + 1) / 2 4}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

We can use this Normal approximation to construct an asymptotic test. The R function wilcox.test with paired=TRUE can implement these tests.

Example 7.3 (Kolmogorov–Smirnov-type statistic) Under $H _ { \mathrm { 0 F } }$ , the absolute values $\big ( \vert \hat { \tau } _ { 1 } \vert , \dots , \vert \hat { \tau } _ { n } \vert \big )$ are fixed but their signs are random. So $\left( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } \right)$ and $- ( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } )$ should have the same distribution. Let

$$
\hat {F} (t) = n ^ {- 1} \sum_ {i = 1} ^ {n} I (\hat {\tau} _ {i} \leq t)
$$

be the empirical distribution of $\left( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } \right)$ , and

$$
1 - \hat {F} (- t -) = n ^ {- 1} \sum_ {i = 1} ^ {n} I (- \hat {\tau} _ {i} \leq t)
$$

be the empirical distribution $o f - ( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } )$ , where $\hat { F } ( - t - )$ is the left limit of the function $\hat { F } ( \cdot )$ at −t. A Kolmogorov–Smirnov-type statistic is then

$$
D = \max _ {t} | \hat {F} (t) + \hat {F} (- t -) - 1 |.
$$

Butler (1969) proposed this test statistic and derived its exact and asymptotic distributions. Unfortunately, this is not implemented in standard software packages. Nevertheless, we can simulate its exact distribution and compute the p-value based on the FRT. 1

Example 7.4 (sign statistic) The sign statistic uses only the signs of the within-pair differences

$$
\Delta = \sum_ {i = 1} ^ {n} I (\hat {\tau} _ {i} > 0).
$$

Under $H _ { \mathrm { 0 F } }$

$$
I (\hat {\tau} _ {i} > 0) \stackrel {I I D} {\sim} B e r n o u l l i (1 / 2)
$$

and therefore

$$
\Delta \sim B i n o m i a l (n, 1 / 2).
$$

Based on this we have an exact Binomial test, which is implemented in the R function binom.test with $\scriptstyle { p = 1 } / 2 .$ Using the CLT, we can also conduct a test based on the following Normal approximation of the Binomial distribution:

$$
\frac {\Delta - n / 2}{\sqrt {n / 4}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

**TABLE 7.1: Counts of four types of pairs**

<table><tr><td></td><td>control outcome 1</td><td>control outcome 0</td></tr><tr><td>treated outcome 1</td><td> $m_{11}$ </td><td> $m_{10}$ </td></tr><tr><td>treated outcome 0</td><td> $m_{01}$ </td><td> $m_{00}$ </td></tr></table>

Example 7.5 (McNemar’s statistic for a binary outcome) If the outcome is binary, we can summarize the data from the MPE in a more compact way. Given a pair, the treated outcome can be either 1 or 0 and the control outcome can be either 1 or 0, yielding a $2 \times 2$ table as in Table 7.1.

Under $H _ { \mathrm { 0 F } }$ , the numbers of concordant pairs $m _ { 1 1 }$ and m00 are fixed, and $m _ { 1 0 } + m _ { 0 1 }$ is also fixed. So the only random component is $m _ { 1 0 }$ which has distribution

$$
m _ {1 0} \sim B i n o m i a l (m _ {1 0} + m _ {0 1}, 1 / 2).
$$

This implies an exact test based on the Binomial distribution. The R function mcnemar.test gives an asymptotic test based on the Normal approximation of the Binomial distribution:

$$
\frac {m _ {1 0} - (m _ {1 0} + m _ {0 1}) / 2}{\sqrt {(m _ {1 0} + m _ {0 1}) / 4}} = \frac {m _ {1 0} - m _ {0 1}}{\sqrt {m _ {1 0} + m _ {0 1}}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

Both the exact FRT and the asymptotic test do not depend on $m _ { 1 1 }$ or m00. Only the numbers of discordant pairs matter in these tests.

## 7.3 Neymanian inference

The average causal effect within pair i is

$$
\tau_ {i} = \frac {1}{2} \left\{Y _ {i 1} (1) + Y _ {i 2} (1) - Y _ {i 1} (0) - Y _ {i 2} (0) \right\},
$$

and the average causal effect for all units is

$$
\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \tau_ {i} = (2 n) ^ {- 1} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {2} \left\{Y _ {i j} (1) - Y _ {i j} (0) \right\}.
$$

It is intuitive that $\hat { \tau } _ { i }$ is unbiased for $\tau _ { i } ,$ so $\hat { \tau }$ is unbiased for τ. We can also calculate the variance of ˆτ . I relegate the exact formula to a homework problem because the MPE is just a special case of the SRE.

However, we cannot follow the strategy of a SRE to estimate the variance of ˆτ . The within-pair sample variances of the outcomes are not well defined because within each pair we have only one treated and one control unit. The data do not allow us to estimate the variance of $\hat { \tau } _ { i }$ within pair i.

Is it possible to estimate the variance of $\hat { \tau }$ in the MPE? Let us forget about the MPE and change the perspective to the classic IID sampling. If the $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ $\mu$ $\sigma ^ { 2 }$ ${ \hat { \tau } } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } { \hat { \tau } } _ { i } { \mathrm { ~ i s ~ } } \sigma ^ { 2 } / n$ Pni=1 τˆi is σ2/n. $\sigma ^ { 2 }$ $( n - 1 ) ^ { - 1 } { \dot { \sum _ { i = 1 } ^ { n } ( \hat { \tau } _ { i } - \hat { \tau } ) ^ { 2 } } }$ so an unbiased estimator for var(ˆτ ) is

$$
\hat {V} = \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\hat {\tau} _ {i} - \hat {\tau}) ^ {2}.
$$

The discussion also extends to the independent but not IID setting; see Problem A1.1 in Chapter A1. The above discussion seems a digression from the MPE which has completely different statistical assumptions. But at least it motivates a variance estimator $\hat { V } ,$ which uses the between-pair variance of $\hat { \tau } _ { i }$ to estimate variance of ˆτ . Of course, it is derived under different assumptions. Does it work for the MPE? Theorem 7.1 below is a positive result.

Theorem 7.1 Under the MPE, $\hat { V }$ is a conservative estimator for the true variance $o f { \hat { \tau } }$ :

$$
E (\hat {V}) - \mathrm{var} (\hat {\tau}) = \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\tau_ {i} - \tau) ^ {2} \geq 0.
$$

$I f$ the $\tau _ { i }$ ’s are constant across pairs, then $E ( \hat { V } ) = \operatorname { v a r } ( \hat { \tau } )$ .

Theorem 7.1 states that under the MPE, $\hat { V }$ is a conservative variance estimator in general and becomes unbiased if the average causal effects are constant across pairs. It is somewhat surprising because $\hat { V }$ depends on the between-pair variance of the $\hat { \tau } _ { i } ^ { \mathrm { : } }$ ’s whereas var(ˆτ ) depends on the within-pair variance of each of $\hat { \tau } _ { i }$ . The proof below might provide some insights for this surprisingly result.

Proof of Theorem 7.1: Using the basic algebraic fact that $\scriptstyle \sum _ { i = 1 } ^ { n } ( a _ { i } - { \bar { a } } ) ^ { 2 } =$ $\textstyle \sum _ { i = 1 } ^ { n } a _ { i } ^ { 2 } - n { \bar { a } } ^ { 2 }$ in the following steps 2 and $5 ,$ we have

$$
\begin{array}{l} n (n - 1) E (\hat {V}) = E \left\{\sum_ {i = 1} ^ {n} (\hat {\tau} _ {i} - \hat {\tau}) ^ {2} \right\} \\ = E \left(\sum_ {i = 1} ^ {n} \hat {\tau} _ {i} ^ {2} - n \hat {\tau} ^ {2}\right) \\ = \sum_ {i = 1} ^ {n} \left\{\operatorname{var} \left(\hat {\tau} _ {i}\right) + \tau_ {i} ^ {2} \right\} - n \left\{\operatorname{var} (\hat {\tau}) + \tau^ {2} \right\} \\ = \sum_ {i = 1} ^ {n} \operatorname{var} (\hat {\tau} _ {i}) - n \operatorname{var} (\hat {\tau}) + \sum_ {i = 1} ^ {n} \tau_ {i} ^ {2} - n \tau^ {2} \\ = n ^ {2} \mathrm{var} (\hat {\tau}) - n \mathrm{var} (\hat {\tau}) + \sum_ {i = 1} ^ {n} (\tau_ {i} - \tau) ^ {2}. \\ \end{array}
$$

Therefore,

$$
E (\hat {V}) = \operatorname{var} (\hat {\tau}) + \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\tau_ {i} - \tau) ^ {2} \geq \operatorname{var} (\hat {\tau}).
$$

![image_07](images/image_07.png)

Similar to the discussions for other experiments, the Neymanian approach relies on the large-sample approximation:

$$
\frac {\hat {\tau} - \tau}{\sqrt {\operatorname{var} (\hat {\tau})}} \to \mathrm{N} (0, 1)
$$

in distribution if n → ∞ and some regularity conditions hold. Due to the over estimation of the variance, the Wald-type confidence interval

$$
\hat {\tau} \pm z _ {1 - \alpha / 2} \sqrt {\hat {V}}
$$

covers τ with probability at least $1 - \alpha$ .

Both the point estimator ˆτ and the variance estimator $\hat { V }$ can be conveniently obtained by OLS, as shown in the proposition below.

Proposition $\mathbf { 7 . 1 } ~ \widehat { \tau }$ and $\hat { V }$ are identical to the coefficient and variance estimator of the intercept from the OLS fit of the vector $( \widehat { \tau } _ { 1 } , \ldots , \widehat { \tau } _ { n } ) ^ { \mathsf { T } }$ on the intercept only.

I leave the proof of Proposition 7.1 as Problem 7.3.

## 7.4 Covariate adjustment

## 7.4.1 FRT

Similar to the discussion in the CRE, there are two general strategies of covariate adjustment in the MPE. First, we can construct test statistics based on the residuals from a model fitting of the outcome on the covariates, since those residuals are fixed numbers under the sharp null hypothesis. A canonical choice is to fit OLS of all observed $Y _ { i j }$ ’s on $X _ { i j } \mathrm { ^ { , } s }$ to obtain the residuals $\hat { \varepsilon } _ { i j } \mathrm { ' s }$ . We can then construct test statistics pretending that the $\hat { \varepsilon } _ { i j } \ ' _ { \ell }$ are the observed outcomes. Rosenbaum (2002a) advocated this strategy in particular to the MPE.

Second, we can directly use some coefficients from model fitting as the test statistics. The discussion in the next subsection will suggest a choice of the test statistic for the second strategy.

## 7.4.2 Regression adjustment

Although we have matched on covariates in the design stage, it is possible that the matching is not perfect and sometimes we have additional covariates beyond those used in the pair-matching stage. In those cases, we can adjust for the covariates to further improve estimation efficiency. Assume that each unit has covariates $X _ { i j }$ , and we can compute the within-pair differences in covariates $\widehat { \tau } _ { X , i }$ and their average $\hat { \tau } _ { X }$ in the same way as the outcome. We can show that

$$
E (\hat {\tau} _ {X, i}) = 0, \quad E (\hat {\tau} _ {X}) = 0,
$$

and

$$
\operatorname{cov} (\hat {\tau} _ {X}) = n ^ {- 2} \sum_ {i = 1} ^ {n} \hat {\tau} _ {X, i} \hat {\tau} _ {X, i} ^ {\mathsf {T}}.
$$

In a realized MPE, cov $\left( \hat { \tau } _ { X } \right)$ is not zero unless all the ${ \hat { \tau } } _ { X , i } { \mathrm { ' s } }$ are zero. With an unlucky draw of $\left( Z _ { 1 } , \ldots , Z _ { n } \right)$ , it is possible that $\hat { \tau } _ { X }$ differs substantially from zero. Similar to the discussion in the CRE, adjusting for the imbalance of the covariate means is likely to improve estimation efficiency.

Consider a class of estimators indexed by $\gamma \colon$

$$
\hat {\tau} (\gamma) = \hat {\tau} - \gamma^ {\mathsf {T}} \hat {\tau} _ {X}
$$

which has mean 0 for any fixed $\gamma .$ We want to choose $\gamma$ to minimize the variance of $\hat { \tau } ( \gamma )$ . Its variance is a quadratic function of $\gamma \colon$

$$
\mathrm{var} \{\hat {\tau} (\gamma) \} = \mathrm{var} (\hat {\tau} - \gamma^ {\mathsf {T}} \hat {\tau} _ {X}) = \mathrm{var} (\hat {\tau}) + \gamma^ {\mathsf {T}} \mathrm{cov} (\hat {\tau} _ {X}) \gamma - 2 \gamma^ {\mathsf {T}} \mathrm{cov} (\hat {\tau} _ {X}, \hat {\tau}),
$$

which is minimized at

$$
\tilde {\gamma} = \mathrm{cov} (\hat {\tau} _ {X}) ^ {- 1} \mathrm{cov} (\hat {\tau} _ {X}, \hat {\tau}).
$$

We have obtained the formula of $\operatorname { c o v } ( { \hat { \tau } } _ { X } )$ in the above, which can also be written as

$$
\operatorname{cov} (\hat {\tau} _ {X}) = n ^ {- 2} \sum_ {i = 1} ^ {n} | \hat {\tau} _ {X, i} | | \hat {\tau} _ {X, i} | ^ {\mathsf {T}},
$$

where $\left. \cdot \right.$ denotes component-wise absolute value of a vector. $\operatorname { S o c o v } ( \hat { \tau } _ { X } )$ is fixed and known from the observed data. However, $\operatorname { c o v } \big ( \hat { \tau } _ { X } , \hat { \tau } \big )$ depends on unknown potential outcomes. Fortunately, we can obtain an unbiased estimator for it, as shown in Theorem 7.2 below.

Theorem 7.2 An unbiased estimator for cov $( \hat { \tau } _ { X } , \hat { \tau } )$ is

$$
\hat {\theta} = \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\hat {\tau} _ {X, i} - \hat {\tau} _ {X}) (\hat {\tau} _ {i} - \hat {\tau}).
$$

The proof of Theorem 7.2 is similar to that of Theorem 7.1. I leave it to Problem 7.2.

Therefore, we can estimate the optimal coefficient $\tilde { \gamma }$ by

$$
\begin{array}{l} \hat {\gamma} = \left(n ^ {- 2} \sum_ {i = 1} ^ {n} \hat {\tau} _ {X, i} \hat {\tau} _ {X, i} ^ {\mathsf {T}}\right) ^ {- 1} \left\{\{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\hat {\tau} _ {X, i} - \hat {\tau} _ {X}) (\hat {\tau} _ {i} - \hat {\tau}) \right\} \\ \approx \left(\sum_ {i = 1} ^ {n} (\hat {\tau} _ {X, i} - \hat {\tau} _ {X}) (\hat {\tau} _ {X, i} - \hat {\tau} _ {X}) ^ {\mathsf {T}}\right) ^ {- 1} \sum_ {i = 1} ^ {n} (\hat {\tau} _ {X, i} - \hat {\tau} _ {X}) (\hat {\tau} _ {i} - \hat {\tau}), \\ \end{array}
$$

which is approximately the coefficient of the $\widehat { \tau } _ { X , i }$ in the OLS fit of the $\hat { \tau } _ { i }$ ’s on the ${ \hat { \tau } } _ { X , i } { \mathrm { ' s } }$ with an intercept. The final estimator is

$$
\hat {\tau} _ {\mathrm{adj}} = \hat {\tau} (\hat {\gamma}) = \hat {\tau} - \hat {\gamma} ^ {\mathsf {T}} \hat {\tau} _ {X},
$$

which, by the property of OLS, is approximately the intercept in the OLS fit of the $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ on the ${ \hat { \tau } } _ { X , i } { \mathrm { ' s } }$ with an intercept.

A conservative variance estimator for $\hat { \tau } _ { \mathrm { a d j } }$ is then

$$
\hat {V} _ {\mathrm{adj}} = \hat {V} + \hat {\gamma} ^ {\mathsf {T}} \mathrm{cov} (\hat {\tau} _ {X}) \hat {\gamma} - 2 \hat {\gamma} ^ {\mathsf {T}} \hat {\theta} = \hat {V} - \hat {\theta} ^ {\mathsf {T}} \mathrm{cov} (\hat {\tau} _ {X}) ^ {- 1} \hat {\theta}.
$$

A subtle technical issue is whether $\hat { \tau } ( \hat { \gamma } )$ has the same optimality as $\hat { \tau } ( \tilde { \gamma } )$ . With large samples, we can show $\hat { \tau } ( \hat { \gamma } ) - \hat { \tau } ( \hat { \gamma } ) = - ( \hat { \gamma } - \tilde { \gamma } ) ^ { \top } \hat { \tau } _ { X }$ is of higher order since it is the product of two “small” terms $\hat { \gamma } - \tilde { \gamma }$ and $\hat { \tau } _ { X }$ . I omit the tedious details for asymptotic analysis, but hope the result makes some intuitive sense to the readers.

Moreover, Fogarty (2018b) discussed the asymptotically equivalent regression formulation of the above covariate-adjusted procedure, and gave a rigorous proof for associated CLT. I summarize the regression formulation below without giving the regularity conditions.

Proposition 7.2 Under the MPE, the covariate-adjusted estimator $\hat { \tau } _ { a d j }$ and the associated variance estimator $\hat { V } _ { a d j }$ can be conveniently approximated by the intercept and the associated variance estimator from the OLS fit of the vector of the $\hat { \tau } _ { i }$ ’s on the 1’s and the matrix of the $\hat { \tau } _ { X , i } \mathit { \Omega } ^ { \prime } s .$ .

I leave the proof of Proposition 7.2 as Problem 7.3. Interestingly, neither Proposition 7.1 nor 7.2 requires the EHW correction of the variance estimator. Because we reduce the data from the MRE to the within-pair differences, it is unnecessary to center the covariates unlike in Lin (2013)’s estimator for the CRE.

## 7.5 Examples

## 7.5.1 Darwin’s data comparing cross-fertilizing and selffertilizing on the height of corns

This is a classical example from Fisher (1935). It contains 15 pairs of corns with either cross-fertilizing or self-fertilizing, with the height being the outcome. The R package HistData provides the original data, where cross and self are the heights under cross-fertilizing and self-fertilizing, respectively, and diff denotes their difference.

<table><tr><td colspan="6">&gt; library(&quot;HistData&quot;)</td></tr><tr><td colspan="6">&gt; ZeaMays</td></tr><tr><td></td><td>pair</td><td>pot</td><td>cross</td><td>self</td><td>diff</td></tr><tr><td>1</td><td>1</td><td>1</td><td>23.500</td><td>17.375</td><td>6.125</td></tr><tr><td>2</td><td>2</td><td>1</td><td>12.000</td><td>20.375</td><td>-8.375</td></tr><tr><td>3</td><td>3</td><td>1</td><td>21.000</td><td>20.000</td><td>1.000</td></tr><tr><td>4</td><td>4</td><td>2</td><td>22.000</td><td>20.000</td><td>2.000</td></tr><tr><td>5</td><td>5</td><td>2</td><td>19.125</td><td>18.375</td><td>0.750</td></tr><tr><td>6</td><td>6</td><td>2</td><td>21.500</td><td>18.625</td><td>2.875</td></tr><tr><td>7</td><td>7</td><td>3</td><td>22.125</td><td>18.625</td><td>3.500</td></tr><tr><td>8</td><td>8</td><td>3</td><td>20.375</td><td>15.250</td><td>5.125</td></tr><tr><td>9</td><td>9</td><td>3</td><td>18.250</td><td>16.500</td><td>1.750</td></tr><tr><td>10</td><td>10</td><td>3</td><td>21.625</td><td>18.000</td><td>3.625</td></tr><tr><td>11</td><td>11</td><td>3</td><td>23.250</td><td>16.250</td><td>7.000</td></tr><tr><td>12</td><td>12</td><td>4</td><td>21.000</td><td>18.000</td><td>3.000</td></tr><tr><td>13</td><td>13</td><td>4</td><td>22.125</td><td>12.750</td><td>9.375</td></tr><tr><td>14</td><td>14</td><td>4</td><td>23.000</td><td>15.500</td><td>7.500</td></tr><tr><td>15</td><td>15</td><td>4</td><td>12.000</td><td>18.000</td><td>-6.000</td></tr></table>

In total, the MPE has $2 ^ { 1 5 } = 3 2 7 6 8$ possible treatment assignment which is a tractable number in R. The following function can enumerate all possible treatment assignment for the MPE:

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

So we enumerate all the treatment assignments, and calculate the corresponding ˆτ ’s and the one-sided exact p-value.

```txt
> difference = ZeaMays$diff
> n.pairs = length(difference)
```

## 7.5 Examples

Figure 7.1 shows the exact randomization of ˆτ .

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

## 7.5.2 Children’s television workshop experiment data

I also re-analyze the data from from Ball et al. (1973) which was also analyzed by Imbens and Rubin (2015). It contains 8 pairs, and the following table summarizes the within-pair covariate and outcome, as well as their differences:

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

We can use the OLS to obtain the point estimators and standard errors: without adjusting for covariates, we have

```txt
> unadj = summary(lm(diffy ~ 1, data = dataxy))$coef
> round(unadj, 3)
Estimate Std. Error t value Pr(>|t|)
(Intercept) 13.425 4.636 2.896 0.023
```

with adjusting for covariates, we have

```txt
> adj = summary(lm(diffy ~ diffx, data = dataxy))$coef
> round(adj, 3)
Estimate Std. Error t value Pr(>|t|)
(Intercept) 8.994 1.410 6.381 0.001
diffx 5.371 0.599 8.964 0.000
```

The above results assume large n, and p-values are justified if we believe the large-n approximation. However, $n = 8$ is not large. In total, we have $2 ^ { 8 } = 2 5 6$ possible treatment assignments, so the smallest possible p-value is $1 / 2 5 6 = 0 . 0 0 3 9$ , which is much larger than the p-value based on the Normal approximation of the covariate-adjusted estimator. In this example, it will be more reasonable to use the FRT with the studentized statistic (i. e., the t value from the lm function) to calculate exact p-values. Figure 7.2 shows the exact distributions of the two studentized statistic, as well as the two-sided p-values. The figure highlights the fact that the randomization distribution of the test statistics are discrete, taking at most 256 possible values. The Normal approximations are unlikely to be accurate especially at the tails. We should report the p-values based on the FRT.

## 7.6 Comparing the MPE and CRE

Imai (2008b) compared the MPE and CRE. Heuristically, the conclusion is that the MPE gives more precise estimators if the matching is well done and the covariates are predictive to the outcome. However, without the outcome data in the design stage, it is hard to decide whether this holds. In the FRT, if covariates are predictive to the outcome, the MPE usually gives more powerful tests compared to the CRE. Greevy et al. (2004) illustrated this using simulation based on the Wilcoxon sign rank statistic. However, this can be a subtle issue with finite samples. Consider an experiment with $2 n$ units, with n units receiving the treatment and n units receiving the control. If we test the sharp null hypothesis at level 0.05, then in the MPE, we need at least $2 \times 5 = 1 0$ units since the smallest p-value is $1 / 2 ^ { 5 } = 1 / 3 2 < 0 . 0 5$ but $1 / 2 ^ { 4 } = 1 / 1 6 > 0 . 0 5$ , but in the CRE, we need at least $2 \times 4 = 8$ units since the smallest p-value is$1 / \binom { 8 } { 4 } = 1 / 7 0 < 0 . 0 5$ but $1 / \binom { 6 } { 3 } = 1 / 2 0 = 0 . 0 5$ . So with 8 units, it is impossible to reject the sharp null hypothesis in the MPE but it is possible in the CRE. Even if the covariates are perfect predictors of the outcome, the MPE is not superior to the CRE based on the FRT.

## 7.7 Extension to the general matched experiment

It is straightforward to extend the MPE to the general matched experiment with varying numbers of control units. Assume that we have n matched sets indexed by $i = 1 , \ldots , n$ . For matched set i, we have $1 + M _ { i }$ units. The $M _ { i } { ^ \mathrm { { s } } }$ can vary. The total number of experimental units is $\begin{array} { r } { N = n + \sum _ { i = 1 } ^ { n } M _ { i } } \end{array}$ . Let $i j$ index the unit $j$ within matched set i $( i = 1 , \ldots , n$ and $j = 1 , \ldots , M _ { i } + 1 )$ . Unit ij has potential outcomes $Y _ { i j } ( 1 )$ and $Y _ { i j } ( 0 )$ under the treatment and control, respectively.

Within matched set $i \ ( i = 1 , \ldots , n )$ , the experimenter randomly selects exactly one unit to receive the treatment with the rest $M _ { i }$ units receiving the control. This general matched experiment is also a special case of the SRE with n strata of size $1 + M _ { i } ( i = 1 , \dots , n )$ . Let $Z _ { i j }$ be the treatment indicator for unit $i j$ , which reveals one of the potential outcomes as

$$
Y _ {i j} = Z _ {i j} Y _ {i j} (1) + (1 - Z _ {i j}) Y _ {i j} (0).
$$

The average causal effect within matched set i equals

$$
\tau_ {i} = (M _ {i} + 1) ^ {- 1} \sum_ {j = 1} ^ {1 + M _ {i}} \{Y _ {i j} (1) - Y _ {i j} (0) \}.
$$

Since it is a SRE, an unbiased estimator of $\tau _ { i }$ is

$$
\hat {\tau} _ {i} = \sum_ {j = 1} ^ {M _ {i} + 1} Z _ {i j} Y _ {i j} - M _ {i} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i j}) Y _ {i j}
$$

which is the difference in means of the outcomes within matched set i.

Below we discuss the statistical inference with the general matched experiment.

## 7.7.1 FRT

As usual, we can always use the FRT to test the sharp null hypothesis

$$
H _ {0 \mathrm{F}}: Y _ {i j} (1) = Y _ {i j} (0) \text {   for   all   } i = 1, \dots , n; j = 1, \dots , M _ {i} + 1.
$$

Because the general matched experiment is a special case of the SRE with many small strata, we can use the test statistics defined in Examples 5.4, 5.5, 7.2, 7.3, 7.4, as well as the estimators and the corresponding t-statistics from the following two subsections.

## 7.7.2 Estimating the average of the within-strata effects

We first focus on estimating the average of the within-strata effects:

$$
\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \tau_ {i}.
$$

It has an unbiased estimator

$$
\hat {\tau} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\tau} _ {i}.
$$

Interestingly, we can show that Theorem 7.1 holds for the general matched experiment, so are other results for the MPE. In particular, we can use the OLS fit of the $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ on the intercept to obtain the point and variance estimators for τ . With covariates, we can use the OLS fit of the $\hat { \tau } _ { i } ^ { \phantom { \dagger } } \rangle$ s on the intercept and the ${ \hat { \tau } } _ { X , i } { \mathrm { ' s } } ,$ , where

$$
\hat {\tau} _ {X, i} = \sum_ {j = 1} ^ {M _ {i} + 1} Z _ {i j} X _ {i j} - M _ {i} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i j}) X _ {i j}
$$

is the corresponding difference in means of the covariates within matched set i.

## 7.7.3 A more general causal estimand

Importantly, the τ above is the average of the $\tau _ { i } ^ { \ , } \mathrm { s } ,$ which does not equal the average causal effect for the N units in the experiment when the $M _ { i } { ^ \mathrm { { \tiny ~ s } } }$ vary. The average causal effect equals

$$
\tau^ {\prime} = N ^ {- 1} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {1 + M _ {i}} \left\{Y _ {i j} (1) - Y _ {i j} (0) \right\} = \sum_ {i = 1} ^ {n} \frac {1 + M _ {i}}{N} \tau_ {i}.
$$

To unify the discussion, I consider the weighted causal effect

$$
\tau_ {w} = \sum_ {i = 1} ^ {n} w _ {i} \tau_ {i}
$$

$\textstyle \sum _ { i = 1 } ^ { n } w _ { i } = 1$ $w _ { i } = n ^ { - 1 }$ $\tau ^ { \prime }$ a special case with $w _ { i } = ( 1 + M _ { i } ) / N$ for $i = 1 , \ldots , n$ . It is straightforward to obtain an unbiased estimator

$$
\hat {\tau} _ {w} = \sum_ {i = 1} ^ {n} w _ {i} \hat {\tau} _ {i},
$$

and calculate its variance

$$
\operatorname{var} (\hat {\tau} _ {w}) = \sum_ {i = 1} ^ {n} w _ {i} ^ {2} \operatorname{var} (\hat {\tau} _ {i}).
$$

However, estimating the variance of this estimator is quite tricky because the $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ are independent random variable without any replicates. This is a famous problem in theoretical statistics studied by Hartley et al. (1969) and Rao (1970). Fogarty (2018a) also discussed this problem without recognizing these previous works. I will give the final form of the variance estimator without detailing the motivation:

$$
\hat {V} _ {w} = \sum_ {i = 1} ^ {n} c _ {i} (\hat {\tau} _ {i} - \hat {\tau} _ {w}) ^ {2}
$$

where

$$
c _ {i} = \frac {\frac {w _ {i} ^ {2}}{1 - 2 w _ {i}}}{1 + \sum_ {i = 1} ^ {n} \frac {w _ {i} ^ {2}}{1 - 2 w _ {i}}}.
$$

As a sanity check, $c _ { i }$ reduces to $\{ n ( n - 1 ) \} ^ { - 1 }$ in the MPE with $M _ { i } = 1$ and $w _ { i } = n ^ { - 1 }$ . For simplicity, we focus on the case with $w _ { i } < 1 / 2$ for all $i \mathrm { \ ' } _ { \mathrm { S } } .$ , that is, there is no matched set containing more than half of the total weights. The following theorem extends Theorem 7.1.

Theorem 7.3 Under the general matched experiment with varying $M _ { i }$ s, we have

$$
E (\hat {V} _ {w}) - \mathrm{var} (\hat {\tau} _ {w}) = \sum_ {i = 1} ^ {n} c _ {i} (\tau_ {i} - \tau_ {w}) ^ {2} \geq \mathrm{var} (\hat {\tau} _ {w}) \geq 0
$$

with equality holding if the $\tau _ { i }$ ’s are constant.

Although the theoretical motivation for $\hat { V } _ { w }$ is quite complicated, it is not too difficult to verify Theorem 7.3 directly. I relegate the proof to Problem 7.9.

## 7.8 Homework Problems

## 7.1 The true variance of τˆ in the MPE

Express var(ˆτ ) in terms of the first two finite-population moments potential outcomes.

## 7.2 A covariance estimator

Prove Theorem 7.2.

## 7.3 Variance estimators via OLS

Prove Propositions 7.1 and 7.2.

## 7.4 Point and variance estimator with binary outcome

This problem extends Example 7.5 to Neymanian inference.

Express ˆτ and $\hat { V }$ in terms of the counts in Table 7.1.

## 7.5 Minimum sample size for the FRT

Extend the discussion in Section 7.6. Consider an experiment with 2n units, with n units receiving the treatment and n units receiving the control, and test the sharp null hypothesis at level 0.001. What is the minimum value of n for an MPE so that the smallest p-value does not exceed than 0.001, and what is the correponding minimum value of n for a CRE.

## 7.6 Re-analyzing Darwin’s data

In MPEFRTdarwin.R, I analyze Darwin’s data using the FRT based on the test statistic ˆτ .

Re-analyze this dataset using the FRT with the Wilcoxon signed rank sum statistic.

Re-analyze this dataset based on the Neymanian inference: unbiased point estimator, conservative variance estimator, 95% confidence interval.

## 7.7 Re-analyzing children’s television workshop experiment data

In MPENeymanstar.R, I analyze the data from based on Neymanian inference.

Re-analyze this dataset using the FRT with different test statistics.

Re-analyze this dataset using the FRT with covariate adjustment, e.g., you can define test statistics based on residuals from the OLS fit of the observed outcome on covariates. Will the conclusion change if you do not include an intercept in your OLS fit?

## 7.8 Re-analyzing Angrist and Lavy (2009)’s data

The original analysis was quite complicated. For this problem, please focus only on Table A1 of the original paper viewing the schools as experimental units. Angrist and Lavy (2009) essentially conducted an MPE on the schools. Dropping pair 6 and all the pairs with noncompliance results in 14 complete pairs, with data shown below and also in AL2009.csv:

<table><tr><td></td><td>pair</td><td>z</td><td>pr99</td><td>pr00</td><td>pr01</td><td>pr02</td></tr><tr><td>1</td><td>1</td><td>0</td><td>0.046</td><td>0.000</td><td>0.091</td><td>0.185</td></tr><tr><td>2</td><td>1</td><td>1</td><td>0.036</td><td>0.051</td><td>0.000</td><td>0.047</td></tr><tr><td>3</td><td>2</td><td>0</td><td>0.054</td><td>0.094</td><td>0.184</td><td>0.034</td></tr><tr><td>4</td><td>2</td><td>1</td><td>0.050</td><td>0.108</td><td>0.110</td><td>0.095</td></tr><tr><td>5</td><td>3</td><td>0</td><td>0.114</td><td>0.000</td><td>0.056</td><td>0.075</td></tr><tr><td>6</td><td>3</td><td>1</td><td>0.098</td><td>0.054</td><td>0.030</td><td>0.068</td></tr><tr><td>7</td><td>4</td><td>0</td><td>0.148</td><td>0.162</td><td>0.082</td><td>0.075</td></tr><tr><td>8</td><td>4</td><td>1</td><td>0.134</td><td>0.390</td><td>0.339</td><td>0.458</td></tr><tr><td>9</td><td>5</td><td>0</td><td>0.152</td><td>0.105</td><td>0.083</td><td>0.129</td></tr><tr><td>10</td><td>5</td><td>1</td><td>0.145</td><td>0.077</td><td>0.579</td><td>0.167</td></tr><tr><td>11</td><td>6</td><td>0</td><td>0.188</td><td>0.214</td><td>0.375</td><td>0.545</td></tr><tr><td>12</td><td>6</td><td>1</td><td>0.179</td><td>0.165</td><td>0.483</td><td>0.444</td></tr><tr><td>13</td><td>7</td><td>0</td><td>0.193</td><td>0.771</td><td>0.328</td><td>0.583</td></tr><tr><td>14</td><td>7</td><td>1</td><td>0.189</td><td>0.186</td><td>0.168</td><td>0.368</td></tr><tr><td>15</td><td>8</td><td>0</td><td>0.197</td><td>0.350</td><td>0.000</td><td>0.383</td></tr><tr><td>16</td><td>8</td><td>1</td><td>0.200</td><td>0.071</td><td>0.667</td><td>0.429</td></tr><tr><td>17</td><td>9</td><td>0</td><td>0.213</td><td>0.176</td><td>0.164</td><td>0.172</td></tr><tr><td>18</td><td>9</td><td>1</td><td>0.209</td><td>0.165</td><td>0.092</td><td>0.151</td></tr><tr><td>19</td><td>10</td><td>0</td><td>0.211</td><td>0.667</td><td>0.250</td><td>0.617</td></tr><tr><td>20</td><td>10</td><td>1</td><td>0.219</td><td>0.250</td><td>0.500</td><td>0.350</td></tr><tr><td>21</td><td>11</td><td>0</td><td>0.219</td><td>0.153</td><td>0.185</td><td>0.219</td></tr><tr><td>22</td><td>11</td><td>1</td><td>0.224</td><td>0.363</td><td>0.372</td><td>0.342</td></tr><tr><td>23</td><td>12</td><td>0</td><td>0.255</td><td>0.226</td><td>0.213</td><td>0.327</td></tr><tr><td>24</td><td>12</td><td>1</td><td>0.257</td><td>0.098</td><td>0.107</td><td>0.095</td></tr><tr><td>25</td><td>13</td><td>0</td><td>0.261</td><td>0.071</td><td>0.000</td><td>NA</td></tr><tr><td>26</td><td>13</td><td>1</td><td>0.263</td><td>0.441</td><td>0.448</td><td>0.435</td></tr><tr><td>27</td><td>14</td><td>0</td><td>0.286</td><td>0.161</td><td>0.126</td><td>0.181</td></tr><tr><td>28</td><td>14</td><td>1</td><td>0.285</td><td>0.389</td><td>0.353</td><td>0.309</td></tr></table>

The outcomes are the Bagrut passing rates in years 2001 and 2002, with the Bagrut passing rates in 1999 and 2000 as pretreatment covariates. Re-analyze the data based on the Neymanian inference with and without covariates. In particular, how do you deal with the missing outcome in pair 25?

## 7.9 Variance estimation in the general matched experiment

This problem contains more details for Section 7.7.

First, prove Theorem 7.1 for the general matched experiment.

Second, prove Theorem 7.3.

Hint: For the second part, we need to first verify that $\hat { \tau } _ { i } - \hat { \tau } _ { w }$ has mean $\tau _ { i } - \tau _ { w }$ and variance

$$
\operatorname{var} \left(\hat {\tau} _ {i} - \hat {\tau} _ {w}\right) = \operatorname{var} \left(\hat {\tau} _ {w}\right) + (1 - 2 w _ {i}) \operatorname{var} \left(\hat {\tau} _ {i}\right).
$$

## 7.10 Recommended readings

Greevy et al. (2004) provided an algorithm to form matched pairs based on covariates. Imai (2008b) discussed estimation of the average causal effect without covariates, and Fogarty (2018b) discussed covariate adjustment in MPEs.

## 8
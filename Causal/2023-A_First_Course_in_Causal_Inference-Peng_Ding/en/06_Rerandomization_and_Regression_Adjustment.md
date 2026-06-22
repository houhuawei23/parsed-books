# Rerandomization and Regression Adjustment

Stratification and post-stratification in Chapter 5 are duals for discrete covariates in the design and analysis of randomized experiments. How should we deal with multidimensional possibly continuous covariates? We can discretize continous covariates, but this is not an ideal strategy with many covariates. Rerandomization and regression adjustment are duals for general covariates, which are the topics for this chapter.

The following table summarizes the topics of Chapters 5 and 6:

<table><tr><td></td><td>design</td><td>analysis</td></tr><tr><td>discrete covariate</td><td>stratification</td><td>post-stratification</td></tr><tr><td>general covariate</td><td>rerandomization</td><td>regression adjustment</td></tr></table>

## 6.1 Rerandomization

## 6.1.1 Experimental design

Again we consider a finite population of n units, where $n _ { 1 }$ of them receive the treatment and $n _ { 0 }$ of them receive the control. Let $\pmb { Z } = ( Z _ { 1 } , \ldots , Z _ { n } )$ be the treatment vector for these units. Unit i has covariate $X _ { i } \in \mathbb { R } ^ { K }$ which can have continuous or binary components. Concatenate them as $\pmb { X } = ( X _ { 1 } , \ldots , X _ { n } )$ and center them at mean zero $\begin{array} { r } { \bar { X } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } X _ { i } = 0 } \end{array}$ without loss of generality.

The CRE balances the covariates in the treatment and control groups on average, for instance, the difference in means of the covariates

$$
\hat {\tau} _ {X} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} X _ {i} - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) X _ {i}
$$

has mean zero under the CRE. However, it can result in undesirable covariate balance across the treatment and control groups in the realized treatment allocation, that is, the realized value of $\hat { \tau } _ { X }$ is often not zero. Using the vector form of Neyman (1923) in Problem 4.6, we can show that

$$
\operatorname{cov} (\hat {\tau} _ {X}) = \frac {1}{n _ {1}} S _ {X} ^ {2} + \frac {1}{n _ {0}} S _ {X} ^ {2} = \frac {n}{n _ {1} n _ {0}} S _ {X} ^ {2},
$$

where $\begin{array} { r } { S _ { X } ^ { 2 } = ( n - 1 ) ^ { - 1 } \sum _ { i = 1 } ^ { n } X _ { i } X _ { i } ^ { \mathsf { T } } } \end{array}$ . The following Mahalanobis distance measures the difference between the treatment and control groups:

$$
M = \hat {\tau} _ {X} ^ {\mathsf {T}} \mathrm{cov} (\hat {\tau} _ {X}) ^ {- 1} \hat {\tau} _ {X} = \hat {\tau} _ {X} ^ {\mathsf {T}} \left(\frac {n}{n _ {1} n _ {0}} S _ {X} ^ {2}\right) ^ {- 1} \hat {\tau} _ {X}.
$$

Technically the above formula of M is meaningful only if $S _ { X } ^ { 2 }$ is invertible, which means that the columns of the covariate matrix are linearly independent. If a column can be represented by a linear combinations of other columns, it is redundant and should be dropped before the experiment. A nice feature of M is that it is invariance under non-degenerate linear transformations of X. Lemma 6.1 below summarizes the result with the proof relegated to Problem 6.2.

Lemma 6.1 M remains the same $i f$ we transform $X _ { i }$ to $\alpha + B X _ { i }$ for all units $i = 1 , \ldots , n$ where α $\in \mathbb { R } ^ { K }$ and $B \in \mathbb { R } ^ { K \times K }$ is invertible.

The finite population central limit theorem (Li and Ding, 2017) ensures that with large $n ,$ the Mahalanobis distance M is approximately $\chi _ { K } ^ { 2 }$ under the CRE. Therefore, it is likely that M has a large realized value under the CRE with asymptotic mean K and variance 2K. Rerandomization avoids covariate imbalance by discarding the treatment allocations with large values of M. Below I give a formal definition of the rerandomization using the Mahalanobis distance (ReM), which was proposed by Cox (1982) and Morgan and Rubin (2012).

Definition 6.1 (ReM) Draw Z from CRE and accept it if and only if

$$
M \leq a,
$$

for some predetermined constant $a > 0$ .

Choosing a is like choosing the number of strata in the SRE, which is a non-trivial problem in practice. At one extreme, $a = \infty$ , we just conduct the CRE. At the other extreme, $a = 0$ , there are very few feasible treatment allocations, and consequently, the experiment has little randomness, rendering randomization-based inference useless. As a compromise, we choose a small but not extremely small $^ { a , }$ for example, $a = 0 . 0 0 1$ or some upper quantile of a $\chi _ { K } ^ { 2 }$ distribution.

ReM uses the Mahalanobis distance as the balance criterion. We can consider general rerandomization with the balance criterion defined as a function of Z and X. For example, we can use the following criterion based on marginal tests for all coordinates of $X _ { i } = ( x _ { i 1 } , \ldots , x _ { i K } ) ^ { \mathsf { T } }$ . We accept Z if and only if

$$
\left| \frac {\hat {\tau} _ {x k}}{\sqrt {\frac {n}{n _ {1} n _ {0}} S _ {x k} ^ {2}}} \right| \leq a (k = 1, \dots , K) \tag {6.1}
$$

for some predetermined constant $a > 0 .$ . For example, a some upper quantile of a standard Normal distribution. ReM has many desirable properties. As mentioned above, it is invariant to linear transformation of the covariates. Moreover, it has nice geometric properties and elegant mathematical theory. This chapter will focus on ReM. See Zhao and Ding (2021b) for the theory for the rerandomization based on criterion (6.1) as well as other criteria.

## 6.1.2 Statistical inference

An important question is how to analyze the data under ReM. Bruhn and McKenzie (2009) and Morgan and Rubin (2012) argued that we can always use the FRT as long as we simulate Z under the constraint $M \ \leq \ a$ . This always yields finite-sample exact p-values under the sharp null hypothesis.

It is a challenging problem to derive the finite sample properties of ReM without assuming the sharp null hypothesis. Instead, Li et al. (2018b) derived the asymptotic distribution of the difference in means of the outcome $\hat { \tau }$ under ReM and the regularity conditions below.

## Condition 6.1 As $n \to \infty$

1. $n _ { 1 } / n$ and $n _ { 0 } / n$ have positive limits;  
2. the finite population covariance of $\{ X _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) , \tau _ { i } \}$ has limit;  
3. max $\iota \leq i \leq n  \vert Y _ { i } ( 1 ) - \bar { Y } ( 1 ) \vert ^ { 2 } / n  0$ , max $\phantom { } _ { 1 \leq i \leq n } | Y _ { i } ( 0 ) - \bar { Y } ( 0 ) | ^ { 2 } / n \to$ $0 ,$ and $\operatorname* { m a x } _ { 1 \leq i \leq n } \| X _ { i } \| ^ { 2 } / n  0$ ,

Below is the main theorem for ReM. Let

$$
L _ {K, a} \sim D _ {1} \mid \boldsymbol {D} ^ {\mathsf {T}} \boldsymbol {D} \leq a
$$

where $\pmb { { \cal D } } = ( D _ { 1 } , \ldots , D _ { K } )$ follows a K-dimensional standard Normal distribution; let ε follows a univariate standard Normal distribution; $L _ { K , a } \bot \varepsilon$ .

Theorem 6.1 Under ReM with $M \leq a$ and Condition 6.1, we $h a v e ^ { 1 }$

$$
\hat {\tau} - \tau \dot {\sim} \sqrt {\mathrm{var} (\tau)} \left\{\sqrt {R ^ {2}} L _ {K, a} + \sqrt {1 - R ^ {2}} \varepsilon \right\},
$$

where

$$
\mathrm{var} (\hat {\tau}) = \frac {S ^ {2} (1)}{n _ {1}} + \frac {S ^ {2} (0)}{n _ {0}} - \frac {S ^ {2} (\tau)}{n}
$$

is Neyman (1923)’s variance formula proved in Chapter $^ { 4 , }$ and

$$
R ^ {2} = \mathrm{corr} ^ {2} (\hat {\tau}, \hat {\tau} _ {X})
$$

![image_06](images/image_06.png)

Rerandomization
area
O
θ
√R²Lₖ,ₐ
τ̂ - τ
√1 - R²ε
τ̂ₓ

FIGURE 6.1: Geometry of ReM

is the squared multiple correlation coefficient2 between $\hat { \tau }$ and $\hat { \tau } _ { X }$ under the CRE.

Although the proof of Li et al. (2018b) is technical, the asymptotic distribution in Theorem 6.1 has clear geometric interpretation, as shown in Figure 6.1. It shows that $\hat { \tau }$ decomposes into a component that is a linear combination of $\hat { \tau } _ { X }$ and a component that is orthogonal to $\hat { \tau } _ { X }$ . Geometrically, $\cos ^ { 2 } \theta = R ^ { 2 }$ , where θ is the angle between $\hat { \tau }$ and $\hat { \tau } _ { X }$ . ReM affects the first component but does not change the second component. The truncated Normal distribution $L _ { K , a }$ is due to the restriction of ReM on the first component.

When $a = \infty$ , the asymptotic distribution simplifies to the one under the CRE:

$$
\hat {\tau} - \tau \dot {\sim} \sqrt {\mathrm{var} (\tau)} \varepsilon .
$$

When the threshold a is close to zero, the the asymptotic distribution simplifies to

$$
\hat {\tau} - \tau \dot {\sim} \sqrt {\mathrm{var} (\tau) (1 - R ^ {2})} \varepsilon .
$$

So with a small threshold a, the efficiency gain due to ReM depends on $R ^ { 2 }$ , which has the following equivalent form.

Proposition 6.1 Under the CRE,

$$
R ^ {2} = \mathrm{corr} ^ {2} (\hat {\tau}, \hat {\tau} _ {X}) = \frac {n _ {1} ^ {- 1} S ^ {2} (1 \mid x) + n _ {0} ^ {- 1} S ^ {2} (0 \mid x) - n ^ {- 1} S ^ {2} (\tau \mid x)}{n _ {1} ^ {- 1} S ^ {2} (1) + n _ {0} ^ {- 1} S ^ {2} (0) - n ^ {- 1} S ^ {2} (\tau)},
$$

$$
R _ {y X} ^ {2} = \mathrm{corr} ^ {2} (y, X) = \frac {\mathrm{cov} (y , X) \mathrm{cov} (X) ^ {- 1} \mathrm{cov} (X , y)}{\mathrm{var} (y)}.
$$

It extends the definition of the Pearson correlation coefficient and measures the linear dependence of y on X.

<!-- footnote -->

- It becomes the title of a book on the modern history of statistics by Salsburg (2001)

<!-- footnote end -->

<!-- footnote -->

- In causal inference, we call $X _ { i }$ a covariate if it is not affected by the treatment. That is, if the covariate has two potential outcomes $X _ { i } ( 1 )$ and $X _ { i } ( 0 )$ , then they must satisfy $X _ { i } ( 1 ) =$ $X _ { i } ( 0 )$ . Standard statistics books often do not distinguish the treatment and covariates because they often appear on the right-hand side of a regression model for the outcome. They are both called covariates in those statistical models. This book distinguishes the treatment and covariates because they play different roles in causal inference.

<!-- footnote end -->

<!-- footnote -->

- Here the divisor $n - 1$ makes the formulas more elegant. Changing the divisor to n complicates the formulas but does not change the results fundamentally. With large $^ { n , }$ the difference is minor.

<!-- footnote end -->

<!-- footnote -->

- In the classic two-sample problem, the outcomes under treatment are IID draws from a distribution with mean $\mu _ { 1 }$ and variance $\sigma _ { 1 } ^ { 2 }$ , and the outcomes under control are IID draws from a distribution with mean $\mu _ { 0 }$ and variance $\sigma _ { 0 } ^ { 2 }$ . Under this assumption, we have
- $\mathrm { v a r } ( \hat { \tau } ) = \frac { \sigma _ { 1 } ^ { 2 } } { n _ { 1 } } + \frac { \sigma _ { 0 } ^ { 2 } } { n _ { 0 } } .$ n0
- Here, var(·) is over the randomness of the outcomes. This variance formula does not involve a third term that depends on the variance of the individual causal effects.

<!-- footnote end -->

<!-- footnote -->

- His most famous quote is “all models are wrong but some are useful.”

<!-- footnote end -->

<!-- footnote -->

- The notation ${ } ^ { \ast } A \stackrel { \cdot } { \sim } B ^ { \prime \prime }$ means that A and B have the same asymptotic distributions.

<!-- footnote end -->

<!-- footnote -->

- The squared multiple correlation coefficient between a random variable y and a random vector X is defined as

<!-- footnote end -->

where $\{ S ^ { 2 } ( 1 ) , S ^ { 2 } ( 0 ) , S ^ { 2 } ( \tau ) \}$ are the finite population variances of $\{ Y _ { i } ( 1 ) , Y _ { i } ( 0 ) , \tau _ { i } \} _ { i = 1 } ^ { n }$ , and $\{ \hat { S ^ { 2 } } ( 1 \mid x ) , \hat { S ^ { 2 } } ( 0 \mid x ) , \hat { S ^ { 2 } } ( \tau \mid x ) \}$ are the corresponding finite population variances of their linear projections on $( 1 , X _ { i } ) .$ . 3 Under the constant causal effect assumption with $\tau _ { i } ~ = ~ \tau$ , $R ^ { 2 }$ reduces to the finite population squared multiple correlation between $Y _ { i } ( 0 )$ and $X _ { i }$ .

I leave the proof of Proposition 6.1 to Problem 6.4.

When $0 < a < \infty .$ , the asymptotic distribution has a more complicated form and is more concentrated at τ and thus the difference in means is more precise under ReM than under the CRE.

If we ignore the design of ReM and still use the confidence interval based on Neyman (1923)’s variance formula and the Normal approximation, it is overly conservative and overcovers τ even if the individual causal effects are constant. Li et al. (2018b) described how to construct confidence intervals based on Theorem 6.1. We omit the discussion here but will come back to the inference issue in Section 6.3.

## 6.2 Regression adjustment

What if we do not conduct rerandomization in the design stage but want to adjust for covariate imbalance in the analysis stage of the CRE? We will discuss several regression adjustment strategies.

## 6.2.1 Covariate-adjusted FRT

The covariates X are all fixed, and furthermore, under $H _ { \mathrm { 0 F } }$ , the observed outcomes are all fixed. Therefore, we can simulate the distribution of any test statistic $T ( Z , Y , X )$ ) and calculate the p-value. The basic idea of the FRT remains the same in the presence additional covariates.

There are two general strategies to construct the test statistic, as summarized by Zhao and Ding (2021a). Problem 3.6 hints at both of them. I summarize them below:

• The first strategy is to construct the test statistic based on residuals from fitted statistical models. We can regress $Y _ { i }$ on $X _ { i }$ to obtain residual $\varepsilon _ { i } .$ , and then treat $\varepsilon _ { i }$ as the pseudo outcome to construct test statistics.

• The second strategy is to use a regression coefficient as a test statistic. We can regress $Y _ { i }$ on $( Z _ { i } , X _ { i } )$ to obtain the coefficient of $Z _ { i }$ as the test statistic. The rest of this section will review some test statistics based on OLS.

In strategy one, we only need to run regression once, but in strategy two, we need to run regression many times. In the above, “regression” is a generic term, which can be linear regression, logistic regression, or even machine learning algorithms. The FRT with any test statistics from these two strategies will be finite-sample exact under $H _ { \mathrm { 0 F } }$ although they differ under alternative hypotheses.

## 6.2.2 Analysis of covariance and extensions

Now we turn to direct estimation of the average causal effect $\tau$ that adjusts for the observed covariates.

Historically, Fisher (1925) proposed to use the analysis of covariance $( \mathrm { A N } .$ COVA) to improve estimation efficiency. This remains a standard strategy in many fields. He suggested running the OLS of $Y _ { i }$ on $( Z _ { i } , X _ { i } )$ and obtaining the coefficient of $Z _ { i }$ as an estimator for τ . Let $\hat { \tau } _ { \mathrm { F } }$ denote Fisher’s ANCOVA estimator.

A former Berkeley Statistics Professor, David Freedman, reanalyzed Fisher’s ANCOVA under Neyman (1923)’s potential outcomes framework. Freedman (2008a,b) found the following negative results:

1. $\hat { \tau } _ { \mathrm { F } }$ is biased, but the simple difference in means $\hat { \tau }$ is unbiased.  
2. The asymptotic variance of $\hat { \tau } _ { \mathrm { F } }$ may be even larger than that of $\hat { \tau }$  
3. The standard error from the OLS is inconsistent for the true standard error of $\hat { \tau } _ { \mathrm { F } }$ under the CRE.

A Berkeley Ph.D. student, Winston Lin, wrote a thesis in response to Freedman’s critiques. Lin (2013) found the following positive results:

1. The bias of $\hat { \tau } _ { \mathrm { F } }$ is small in large samples, and it goes to zero as the sample size approaches infinity.  
2. We can improve the asymptotic efficiency of both $\hat { \tau }$ and ˆτF by using the coefficient of $Z _ { i }$ in the OLS of $Y _ { i }$ on $( Z _ { i } , X _ { i } , Z _ { i } \times X _ { i } )$ . Let $\hat { \tau } _ { \mathrm { L } }$ denote Lin (2013)’s estimator. Moreover, the EHW standard error is a conservative estimator for the true standard error of $\hat { \tau } _ { \mathrm { L } }$ under the CRE.  
3. The EHW standard $\mathrm { e r r o r ^ { 4 } }$ for $\hat { \tau } _ { \mathrm { F } }$ in the OLS fit of $Y _ { i }$ on $( Z _ { i } , X _ { i } )$ is

a conservative estimator for the true standard error of $\hat { \tau } _ { \mathrm { F } }$ under the CRE.

## 6.2.2.1 Some heuristics for Lin (2013)’s results

Neyman (1923)’s result demonstrates that the variance of the difference-inmeans estimator depends on the variances of the potential outcomes. Intuitively, we can reduce the variance of the estimator by reducing the variances of the outcomes. A simple family of linearly adjusted estimator is

$$
\begin{array}{l} \hat {\tau} \left(\beta_ {1}, \beta_ {0}\right) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \left(Y _ {i} - \beta_ {1} ^ {\mathsf {T}} X _ {i}\right) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} \left(1 - Z _ {i}\right) \left(Y _ {i} - \beta_ {0} ^ {\mathsf {T}} X _ {i}\right) (6. 2) \\ = \left\{\hat {\bar {Y}} (1) - \beta_ {1} ^ {\mathsf {T}} \hat {\bar {X}} (1) \right\} - \left\{\hat {\bar {Y}} (0) - \beta_ {0} ^ {\mathsf {T}} \hat {\bar {X}} (0) \right\}, \tag {6.3} \\ \end{array}
$$

where $\{ \hat { \bar { Y } } ( 1 ) , \hat { \bar { Y } } ( 0 ) \}$ are the sample means of the outcomes, and $\{ \hat { \bar { X } } ( 1 ) , \hat { \bar { X } } ( 0 ) \}$ are the sample means of the covariates. This covariate-adjusted estimator $\hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } )$ tries to reduce the variance of ˆτ by residualizing the potential outcomes. It reduces $\mathrm { t o } \ \hat { \tau }$ with $\beta _ { 1 } = \beta _ { 0 } = 0 .$ . It has mean τ for any fixed values of $\beta _ { 1 }$ and $\beta _ { 0 }$ because $\bar { X } = 0$ . We are interested in finding the $( \beta _ { 1 } , \beta _ { 0 } )$ that minimized the variance of $\hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } )$ . This estimator is essentially the difference in means of the adjusted potential outcomes $\{ Y _ { i } ( 1 ) - \beta _ { 1 } ^ { \mathsf { T } } X _ { i } , Y _ { i } ( 0 ) - \beta _ { 0 } ^ { \mathsf { T } } X _ { i } \} _ { i = 1 } ^ { n }$ . Applying Neyman (1923)’s result, this estimator has variance

$$
\operatorname{var} \{\hat {\tau} (\beta_ {1}, \beta_ {0}) \} = \frac {S ^ {2} (1 ; \beta_ {1})}{n _ {1}} + \frac {S ^ {2} (0 ; \beta_ {1})}{n _ {0}} - \frac {S ^ {2} (\tau ; \beta_ {1} , \beta_ {0})}{n},
$$

where $S ^ { 2 } ( z ; \beta _ { 1 } ) ~ ( z = 1 , 0 )$ and $S ^ { 2 } ( \tau ; \beta _ { 1 } , \beta _ { 0 } )$ are the finite population variances of the adjusted potential outcomes and individual effects, respectively; moreover, a conservative variance estimate is

$$
\hat {V} (\beta_ {1}, \beta_ {0}) = \frac {\hat {S} ^ {2} (1 ; \beta_ {1})}{n _ {1}} + \frac {\hat {S} ^ {2} (0 ; \beta_ {1})}{n _ {0}},
$$

where

$$
\hat {S} ^ {2} (1; \beta_ {1}) = (n _ {1} - 1) ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \{Y _ {i} - \gamma_ {1} - \beta_ {1} ^ {\mathsf {T}} X _ {i} \} ^ {2},
$$

$$
\hat {S} ^ {2} (0; \beta_ {0}) = (n _ {0} - 1) ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \{Y _ {i} - \gamma_ {0} - \beta_ {0} ^ {\mathsf {T}} X _ {i} \} ^ {2}
$$

are the sample variances of the adjusted potential outcomes with $\gamma _ { 1 }$ and $\gamma _ { 0 }$ being the sample means of $Y _ { i } - \beta _ { 1 } ^ { \mathsf { T } } X _ { i }$ under treatment and $Y _ { i } - \beta _ { 0 } ^ { \mathsf { T } } X _ { i }$ under control. To minimize $\hat { V } ( \beta _ { 1 } , \beta _ { 0 } )$ , we need to solve two OLS problems:

$$
\min _ {\gamma_ {1}, \beta_ {1}} \sum_ {i = 1} ^ {n} Z _ {i} \{Y _ {i} - \gamma_ {1} - \beta_ {1} ^ {\mathsf {T}} X _ {i} \} ^ {2}, \quad \min _ {\gamma_ {0}, \beta_ {0}} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \{Y _ {i} - \gamma_ {0} - \beta_ {0} ^ {\mathsf {T}} X _ {i} \} ^ {2}.
$$

We run OLS of $Y _ { i }$ on $X _ { i }$ for the treatment and control groups separately and obtain $( \hat { \gamma } _ { 1 } , \hat { \beta } _ { 1 } )$ and $( \hat { \gamma } _ { 0 } , \hat { \beta } _ { 0 } )$ . The final estimator is

$$
\begin{array}{l} \hat {\tau} (\hat {\beta} _ {1}, \hat {\beta} _ {0}) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\beta} _ {1} ^ {\mathsf {T}} X _ {i}) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) (Y _ {i} - \hat {\beta} _ {0} ^ {\mathsf {T}} X _ {i}) \\ = \left\{\hat {\bar {Y}} (1) - \hat {\beta} _ {1} ^ {\mathsf {T}} \hat {\bar {X}} (1) \right\} - \left\{\hat {\bar {Y}} (0) - \hat {\beta} _ {0} ^ {\mathsf {T}} \hat {\bar {X}} (0) \right\}. \\ \end{array}
$$

From the properties of the OLS fits (see (A2.3)), we know

$$
\hat {\bar {Y}} (1) = \hat {\gamma} _ {1} + \hat {\beta} _ {1} ^ {\mathsf {T}} \hat {\bar {X}} (1), \quad \hat {\bar {Y}} (0) = \hat {\gamma} _ {0} + \hat {\beta} _ {0} ^ {\mathsf {T}} \hat {\bar {X}} (0).
$$

Therefore, we can rewrite the estimator as

$$
\hat {\tau} \left(\hat {\beta} _ {1}, \hat {\beta} _ {0}\right) = \hat {\gamma} _ {1} - \hat {\gamma} _ {0} \tag {6.4}
$$

The equivalent form in (6.4) suggests that we can obtain $\hat { \tau } ( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$ from a single OLS fit below.

Proposition 6.2 The estimator $\hat { \tau } ( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$ in (6.4) equals the coefficient of $Z _ { i }$ in the OLS fit of Yi on $( Z _ { i } , X _ { i } , Z _ { i } \times X _ { i } )$ , which $i s \ \hat { \tau } _ { \mathrm { L } }$ introduced before.

I leave the proof of Proposition 6.2 to Problem 6.5, which is a pure algebra fact.

Based on the discussion above, a conservative variance estimator for $\hat { \tau } _ { \mathrm { L } }$ is

$$
\begin{array}{l} \hat {V} (\hat {\beta} _ {1}, \hat {\beta} _ {0}) = \frac {1}{n _ {1} (n _ {1} - 1)} \sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\gamma} _ {1} - \hat {\beta} _ {1} ^ {\mathsf {T}} X _ {i}) ^ {2} \\ + \frac {1}{n _ {0} (n _ {0} - 1)} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) (Y _ {i} - \hat {\gamma} _ {0} - \hat {\beta} _ {0} ^ {\mathsf {T}} X _ {i}) ^ {2}. \\ \end{array}
$$

Based on quite technical calculations, Lin (2013) further showed that the EHW standard error from the OLS in Proposition 6.2 is almost identical to $\hat { V } ( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$ which is a conservative estimator of the true standard error of $\scriptstyle { \hat { \tau } } _ { \mathrm { L } }$ under the CRE. Intuitively, this is because we do not assume that the linear model is correctly specified, and the EHW standard error is robust to model misspecification.

There is a subtle issue with the discussion above. The variance formula va $\cdot \{ \hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } ) \}$ works for fixed $( \beta _ { 1 } , \beta _ { 0 } )$ , but the estimator $\hat { \tau } ( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$ uses two estimated coefficients $( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$ . The additional uncertainty in the estimated coefficients may cause finite-sample bias in the final estimator. Lin (2013) showed that the issue goes away asymptotically. However, his theory requires a large sample size and some regularity conditions on the potential outcomes and covariates.

**TABLE 6.1: Predicting the potential outcomes**

<table><tr><td>X</td><td>Z</td><td>Y(1)</td><td>Y(0)</td><td> $\hat{Y}(1)$ </td><td> $\hat{Y}(0)$ </td></tr><tr><td> $X_1$ </td><td>1</td><td> $Y_1(1)$ </td><td>?</td><td> $\hat{\mu}_1(X_1)$ </td><td> $\hat{\mu}_0(X_1)$ </td></tr><tr><td> $\vdots$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $X_{n_1}$ </td><td>1</td><td> $Y_{n_1}(1)$ </td><td>?</td><td> $\hat{\mu}_1(X_{n_1})$ </td><td> $\hat{\mu}_0(X_{n_1})$ </td></tr><tr><td> $X_{n_1+1}$ </td><td>0</td><td>?</td><td> $Y_{n_1+1}(0)$ </td><td> $\hat{\mu}_1(X_{n_1+1})$ </td><td> $\hat{\mu}_0(X_{n_1+1})$ </td></tr><tr><td> $\vdots$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $X_n$ </td><td>0</td><td>?</td><td> $Y_n(0)$ </td><td> $\hat{\mu}_1(X_n)$ </td><td> $\hat{\mu}_0(X_n)$ </td></tr></table>

## 6.2.2.2 Understanding Lin (2013)’s estimator via predicting the potential outcomes

We can view Lin (2013)’s estimator as a predictive estimator based on OLS fits of the potential outcomes. We build a prediction model for $Y ( 1 )$ based on X using the data from the treatment group:

$$
\hat {\mu} _ {1} (x) = \hat {\gamma} _ {1} + \hat {\beta} _ {1} ^ {\mathsf {T}} x. \tag {6.5}
$$

Similarly, we build a prediction model for Y (0) based on X using the data from the control group:

$$
\hat {\mu} _ {0} (x) = \hat {\gamma} _ {0} + \hat {\beta} _ {0} ^ {\mathsf {T}} x. \tag {6.6}
$$

If we predict the missing potential outcomes, then we have the following predictive estimator:

$$
\hat {\tau} _ {\text { pred }} = n ^ {- 1} \left\{\sum_ {Z _ {i} = 1} Y _ {i} + \sum_ {Z _ {i} = 0} \hat {\mu} _ {1} (X _ {i}) - \sum_ {Z _ {i} = 1} \hat {\mu} _ {0} (X _ {i}) - \sum_ {Z _ {i} = 0} Y _ {i} \right\}. \tag {6.7}
$$

We can verify that with (6.5) and (6.6), the predictive estimator equals Lin (2013)’s estimator:

$$
\hat {\tau} _ {\mathrm{pred}} = \hat {\tau} _ {\mathrm{L}}. (6. 8)
$$

If we predict all potential outcomes even if they are observed, we have the following projective estimator:

$$
\hat {\tau} _ {\text { proj }} = n ^ {- 1} \sum_ {i = 1} ^ {n} \{\hat {\mu} _ {1} (X _ {i}) - \hat {\mu} _ {0} (X _ {i}) \}. \tag {6.9}
$$

We can verify that with (6.5) and (6.6), the projective estimator equals Lin (2013)’s estimator:

$$
\hat {\tau} _ {\mathrm{proj}} = \hat {\tau} _ {\mathrm{L}}. \tag {6.10}
$$

I leave the proofs of (6.8) and (6.10) to Problem 6.6.

The more general formulas (6.7) and (6.9) are well defined with other predictors of the potential outcomes. To make connections with Lin (2013)’s estimator, I focus on the linear predictors here. They can be quite general, including much more complicated machine learning algorithms. However, constructing point estimator is just the first step in analyzing the CRE. A more important second step is to quantify the uncertainty associated with the estimator, which depends on the properties of the predictors of the potential outcomes. Nevertheless, without doing additional theoretical analysis, we can always use (6.7) and (6.9) as the test statistics in the FRT.

## 6.2.2.3 Understanding Lin (2013)’s estimator via adjusting for covariate imbalance

The linearly-adjusted estimator has an equivalent form

$$
\hat {\tau} (\beta_ {1}, \beta_ {0}) = \hat {\tau} - \gamma^ {\mathsf {T}} \hat {\tau} _ {X} \tag {6.11}
$$

where $\begin{array} { r } { \gamma = \frac { n _ { 0 } } { n } \beta _ { 1 } + \frac { n _ { 1 } } { n } \beta _ { 0 } } \end{array}$ , so we can also write it as $\hat { \tau } ( \gamma ) = \hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } )$ . Similarly, Lin (2013)’s estimator has an equivalent form

$$
\hat {\tau} _ {\mathrm{L}} = \hat {\tau} - \hat {\gamma} ^ {\mathsf {T}} \hat {\tau} _ {X}, \tag {6.12}
$$

where $\begin{array} { r } { \hat { \gamma } = \frac { n _ { 0 } } { n } \hat { \beta } _ { 1 } + \frac { n _ { 1 } } { n } \hat { \beta } _ { 0 } } \end{array}$ . I leave the proofs of (6.11) and (6.12) to Problem 6.7. The forms (6.11) and (6.12) are the mathematical statements of “adjusting for the covariate imbalance.” They essentially subtract some linear combinations of the difference in means of the covariates. Since ˆτ and $\hat { \tau } _ { X }$ are correlated, the covariate adjustment with an appropriate γ reduces the variance of ˆτ . Another interesting feature of (6.11) and (6.12) is that the final estimators depend only on γ or $\hat { \gamma } ,$ so the choice of the β-coefficients are not unique. Therefore, Lin (2013)’s estimator is just one of the optimal estimators, but it can be easily implemented via the standard OLS with the EHW standard error.

## 6.2.3 Some additional remarks on regression adjustment

## 6.2.3.1 Duality between ReM and regression adjustment

Li et al. (2018b) pointed out that ReM and Lin (2013)’s regression adjustment are duals in using covariates in the design and analysis stages of the experiment. To be more specific, when a is small, the asymptotic distribution of ˆτ under ReM is almost identical to the asymptotic distribution of $\hat { \tau } _ { \mathrm { L } }$ under the CRE. So ReM uses covariates in the design stage and Lin (2013)’s regression adjustment uses covariates in the analysis stage, achieving nearly the same asymptotic efficiency gain when a is small.

## 6.2.3.2 Equivalence of regression adjustment and post-stratification

If we have discrete covariate $C _ { i }$ with $K$ categories, we can create $K - 1$ centered dummy variables

$$
X _ {i} = (I (C _ {i} = 1) - \pi_ {[ 1 ]}, \ldots , I (C _ {i} = K - 1) - \pi_ {[ K - 1 ]}).
$$

In this case, Lin $( \mathrm { 2 0 1 3 ) \mathrm { ^ { \circ } s } }$ regression adjustment is equivalent to poststratification, as summarized by the following proposition.

Proposition 6.3 $\hat { \tau } _ { \mathrm { L } }$ based in $X _ { i }$ is numerically identical to the poststratification estimator based on $C _ { i }$ .

I leave the proof of Proposition 6.3 as Problem 6.9.

## 6.2.3.3 Difference-in-difference as a special case of covariate adjustment $\hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } )$

An important covariate $X$ in many studies is the lagged outcome before the treatment. For instance, the covariate X is the pre-test score if the outcome $Y$ is the post-test score in educational research; the covariate $X$ is the log wage before the job training program if the outcome $Y$ is the log wage after the job training program. With the lagged outcome $X$ as a covariate, a popular estimator is the gain score or difference-in-difference estimator with $\beta _ { 1 } = \beta _ { 0 } =$ 1 in (6.2) and (6.3):

$$
\begin{array}{l} \hat {\tau} (1, 1) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - X _ {i}) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) (Y _ {i} - X _ {i}) \\ { = } { \left\{ \hat { \bar { Y } } ( 1 ) - \hat { \bar { Y } } ( 0 ) \right\} - \left\{ \hat { \bar { X } } ( 1 ) - \hat { \bar { X } } ( 0 ) \right\} . } \\ \end{array}
$$

The first form of $\hat { \tau } ( 1 , 1 )$ justifies the name gain score because it is essentially the difference in means of the gain score $g _ { i } = Y _ { i } - X _ { i }$ . The second form of $\hat { \tau } ( 1 , 1 )$ justifies the name $d i f f e r e n c e - i n - d i f f$ erence because it is the difference between two differences in means. This estimator is different from Lin (2013)’s estimator: it fixes $\beta _ { 1 } = \beta _ { 0 } = 1$ in advance while Lin (2013)’s estimator involves two estimated $\beta ^ { \gamma } \mathrm { s } .$ . It is unbiased with a conservative variance estimator

$$
\begin{array}{l} \hat {V} (1, 1) = \frac {1}{n _ {1} (n _ {1} - 1)} \sum_ {i = 1} ^ {n} Z _ {i} \{g _ {i} - \hat {\bar {g}} (1) \} ^ {2} \\ + \frac {1}{n _ {0} (n _ {0} - 1)} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \{g _ {i} - \hat {\bar {g}} (0) \} ^ {2}, \\ \end{array}
$$

where $\hat { \bar { g } } ( 1 )$ and $\hat { \bar { g } } ( 0 )$ are the sample means of the gain score $g _ { i } = Y _ { i } - X _ { i }$ under treatment and control, respectively. When the lagged outcome is a strong predictor of the outcome, the gain score $g _ { i } = Y _ { i } - X _ { i }$ often has much smaller variance than the outcome itself. In this case, $\hat { \tau } ( 1 , 1 )$ often greatly reduces the variance of the simple difference in means of the outcome.

**TABLE 6.2: Design and analysis of experiments**

<table><tr><td></td><td colspan="4">analysis</td></tr><tr><td rowspan="3">design</td><td>CRE</td><td> $\hat{\tau}$  (Neyman, 1923)</td><td> $\stackrel{1}{\longrightarrow}$ </td><td> $\hat{\tau}_{\text{L}}$  (Lin, 2013)</td></tr><tr><td></td><td> $2 \Big\downarrow$ </td><td></td><td> $\Big\downarrow 4$ </td></tr><tr><td>ReM</td><td> $\hat{\tau}$  (Li et al., 2018b)</td><td> $\stackrel{3}{\longrightarrow}$ </td><td> $\hat{\tau}_{\text{L}}$  (Li and Ding, 2020)</td></tr></table>

## 6.2.4 Extension to the SRE

It is possible that we have an experiment stratified on a discrete variable C and observe additional covariates X. If all strata are large, then we can obtain Lin (2013)’s estimators within strata $\hat { \tau } _ { \mathrm { L } , [ k ] }$ and obtain the final estimator as

$$
\hat {\tau} _ {\mathrm{L,S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \hat {\tau} _ {\mathrm{L}, [ k ]}.
$$

A conservative variance estimator is

$$
\hat {V} _ {\mathrm{L,S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \hat {V} _ {\mathrm{EHW}, [ k ]},
$$

where $\hat { V } _ { \mathrm { E H W } , [ k ] }$ is the EHW variance estimator from the OLS fit of the outcome on the treatment indicator, the covariates, and their interactions within stratum k. Importantly, we need to center covariates by their stratum-specific means.

## 6.3 Unification, combination, and comparison

Li and Ding (2020) unified the literature and showed that we can combine rerandomization and regression adjustment. That is, if we rerandomize in the design stage, we can use Lin (2013)’s estimator with the EHW standard error in the analysis stage. The combination of rerandomization and regression adjustment improves covariate balance in the design stage and estimation efficiency in the analysis stage.

Table 6.2 summarizes the literature from Neyman (1923) to Li and Ding (2020). Arrow 1 illustrates the efficiency gain of covariate adjustment in the CRE: asymptotically, $\hat { \tau } _ { \mathrm { L } }$ has smaller variance than ˆτ . Arrow 2 illustrates the efficiency gain of the ReM: asymptotically, ˆτ has narrower quantile range under the ReM than under the CRE. Arrows 3 and 4 illustrate the benefits of the combination.

## 6.4 Simulation

Angrist et al. (2009) conducted an experiment to evaluate different strategies to improve academic performance among college freshmen. Here I use a subset of the original data, focusing on the control group and the treatment group offered academic support services and financial incentives for good grades. The outcome is the GPA at the end of the first year, and two covariates are the gender and baseline GPA. The following table summarizes the results based on the unadjusted and adjusted estimators. The adjusted estimator has smaller standard error although it gives the same insignificant result as the unadjusted estimator.

<table><tr><td></td><td>estimate</td><td>s.e.</td><td>t-stat</td><td>p-value</td></tr><tr><td>Neyman</td><td>0.054</td><td>0.076</td><td>0.719</td><td>0.472</td></tr><tr><td>Lin</td><td>0.075</td><td>0.072</td><td>1.036</td><td>0.300</td></tr></table>

I also use this dataset to conduct simulation studies to evaluate the four design and analysis strategies summarized in Table 6.2. I fit quadratic functions of the outcome on the covariates and use them to impute all the missing potential outcomes, separately for the treated and control groups. To show the improvement of ReM and regression adjustment, I also rescale the error terms by 0.1 and 0.25 to increase the signal to noise ratio. With the imputed Science Table, I generate 2000 treatments, obtain the observed data, and calculate the estimators. In the simulation, the “true” outcome model is nonlinear, but we still use linear adjustment for estimation. By doing this, we can evaluate the properties of the estimators when the linear model is misspecified.

Figure 6.2 shows the violin plots of the four combinations, subtracting the true τ from the estimates. As predicted by the theory, all estimators are nearly unbiased, and both ReM and regression adjustment improve efficiency. They are more effective when the noise level is smaller.

## 6.5 Final remarks

With a continuous outcome, Fisher’s ANCOVA has been the standard approach for many years. Lin (2013)’s improvement has better theoretical properties even if the linear model is misspecified. With a binary outcome, it is common to use the coefficient of the treatment in the logistic regression of the observed outcome on the treatment indicator and covariates to estimate the causal effects However, Freedman (2008c) showed that this logistic regression does not have nice properties under the potential outcomes framework. Even if the logistic model is correct, the coefficient estimates the conditional odds ratio which may not be the parameter of interest; when the logistic model is incorrect, it is even harder to interpret the coefficient. From the discussion above, if the parameter of interest is the average causal effect, we can still use Lin (2013)’s estimator to analyze the binary outcome data in the CRE. Guo and Basse (2023) extend Lin (2013)’s theory to allow for using generalized linear models to construct estimators for the average causal effect under the potential outcomes framework.

Other extensions of Lin (2013)’s theory focus on high dimensional covariates. Bloniarz et al. (2016) focus on the regime with many covariates than the sample size, and under the sparsity assumption, they suggest replacing the OLS fits by the least absolute shrinkage and selection operator (LASSO) fits (Tibshirani, 1996) of the outcome on the treatment, covariates and their interactions. Lei and Ding (2021) focus on the regime with a diverging number of covariates without assuming sparsity, and under certain regularity conditions, they show that Lin (2013)’s estimator is still consistent and asymptotically Normal. Wager et al. (2016) propose to use machine learning methods to analyze high dimensional experimental data.

## 6.6 Homework Problems

## 6.1 FRT under ReM

Describe the FRT under ReM.

## 6.6 Homework Problems

## 6.2 Invariance of the Mahalanobis Distance

Prove Lemma 6.1.

## 6.3 Bias of the difference-in-means estimator under rerandomization

Assume that we draw $\pmb { Z } = ( Z _ { 1 } , \ldots , Z _ { n } )$ from a CRE and accept it if and only if $\phi ( Z , X ) = 1$ , where $\phi$ is a predetermined balance criterion. Show that if $n _ { \mathrm { 1 } } = n _ { \mathrm { 0 } }$ and

$$
\phi (\mathbf {Z}, \mathbf {X}) = \phi (\mathbf {1} _ {n} - \mathbf {Z}, \mathbf {X}), \tag {6.13}
$$

then $\hat { \tau }$ is unbiased for τ . Verify that rerandomization using the Mahalanobis distance satisfies (6.13) if $n _ { \mathrm { 1 } } = n _ { \mathrm { 0 } }$ . Give a counterexample that ˆτ is biased for τ when these two conditions do not hold.

## 6.4 Equivalent form of $R ^ { 2 }$ in the CRE

Prove Proposition 6.1.

## 6.5 Lin’s estimator for covariate adjustment

Prove Proposition 6.2.

## 6.6 Predictive and projective estimators

Prove (6.8) and (6.10).

## 6.7 Equivalent form of the covariate-adjusted estimator

Prove (6.11) and (6.12).

## 6.8 ANCOVA also adjusts for covariate imbalance

This problem gives a result for ANCOVA that is similar to (6.12).

Show that

$$
\hat {\tau} _ {\mathrm{F}} = \hat {\tau} - \hat {\gamma} _ {\mathrm{F}} ^ {\mathsf {T}} \hat {\tau} _ {X},
$$

where $\hat { \gamma } _ { \mathrm { F } }$ is the coefficient of $X _ { i }$ in the OLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } )$ .

## 6.9 Regression adjustment / post-stratification of CRE

Prove Proposition 6.3.

Hint: Sometimes $\hat { \tau } _ { \mathrm { { P S } } }$ or $\hat { \tau } _ { \mathrm { L } }$ may not be well-defined. In those cases, we treat $\hat { \tau } _ { \mathrm { { P S } } }$ and $\hat { \tau } _ { \mathrm { L } }$ as equal. You can ignore this complexity in the proof.

## 6.10 More on the difference-in-difference estimator in the CRE

This problem gives more details for the difference-in-difference estimator in the CRE in Section 6.2.3.3.

Show that $\hat { \tau } ( 1 , 1 )$ is unbiased for τ , calculate its variance, and show that $\hat { V } ( 1 , 1 )$ is a conservative estimator for the true variance of $\hat { \tau } ( 1 , 1 )$ . When does $E \{ \hat { V } ( 1 , 1 ) \} = \operatorname { v a r } \{ \hat { \tau } ( 1 , 1 ) \}$ hold?

Compare the variances of ${ \hat { \tau } } ( 0 , 0 )$ and $\hat { \tau } ( 1 , 1 )$ to show that

$$
\operatorname{var} \{\hat {\tau} (0, 0) \} \geq \operatorname{var} \{\hat {\tau} (1, 1) \}
$$

if and only if

$$
2 \frac {n _ {0}}{n} \beta_ {1} + 2 \frac {n _ {1}}{n} \beta_ {0} \geq 1,
$$

where

$$
\beta_ {1} = \frac {\sum_ {i = 1} ^ {n} (X _ {i} - \bar {X}) \{Y _ {i} (1) - \bar {Y} (1) \}}{\sum_ {i = 1} ^ {n} (X _ {i} - \bar {X}) ^ {2}}, \quad \beta_ {0} = \frac {\sum_ {i = 1} ^ {n} (X _ {i} - \bar {X}) \{Y _ {i} (0) - \bar {Y} (0) \}}{\sum_ {i = 1} ^ {n} (X _ {i} - \bar {X}) ^ {2}}
$$

are the coefficients of $X _ { i }$ in the OLS fits of $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ on $( 1 , X _ { i } )$ , respectively.

Remark: Gerber and Green (2012, page 28) discussed a special case of this problem with $n _ { \mathrm { 1 } } = n _ { \mathrm { 0 } }$ .

## 6.11 Data re-analyses

Re-analyze the data used in SRE Neyman penn.R. The analysis in Chapter 5 uses the treatment indicator, the outcome and the block indicator. Now we want to use all other covariates.

Conduct regression adjustments within strata of the experiment, and then combine these adjusted estimators to estimate the average causal effect. Report the point estimator, estimated standard error and 95% confidence interval. Compare them with those without regression adjustments.

## 6.12 Recommended reading

The title of this chapter is the same as that of Li and Ding (2020), which studied the roles of rerandomization and regression adjustment in the design and analysis stages of randomized experiments, respectively.

## 7
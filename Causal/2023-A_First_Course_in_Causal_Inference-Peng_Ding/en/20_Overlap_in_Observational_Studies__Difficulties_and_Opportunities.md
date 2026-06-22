# Overlap in Observational Studies: Difficulties and Opportunities

## 20.1 Implications of overlap

In Part III of this book, causal inference with observational studies relies on two critical assumptions: unconfoundedness

$$
Z \bot \{Y (1), Y (0) \} \mid X
$$

and overlap

$$
0 <   e (X) <   1.
$$

D’Amour et al. (2021) pointed out the tension between these two assumptions: typically, more covariates make the unconfoundedness assumption more plausible (ignoring M-bias discussed in Chapter 16.3.1), but more covariates make the overlap assumption less plausible because the treatment becomes more predictable.

If some units has $e ( X ) = 0 { \mathrm { ~ o r ~ } } e ( X ) = 1$ , then we have philosophic difficulty of thinking about the counterfactual potential outcomes (King and Zeng, 2006). In particular, if a unit deterministically receives the treatment, then it may not be meaningful to conceive its potential outcome under control; vice versa. Even if the true propensity score is not exactly 0 or 1, the estimated propensity score can be very close to 0 or 1 in finite sample, which makes the estimators based on inverse probability weighting numerically unstable. Many statistical analyses in fact require a strict version of overlap:

Assumption 20.1 (strict overlap) $\eta \leq e ( X ) \leq 1 - \eta$ for some $\eta \in ( 0 , 1 / 2 )$ .

However, D’Amour et al. (2021, Corollary 1) showed that Assumption 20.1 has very strong implications. For simplicity, I present only one of their results. Let $X _ { k } \ ( k = 1 , \ldots , p )$ be the kth component of the covariate $X =$ $( X _ { 1 } , \ldots , X _ { p } )$ , and $e = \operatorname { p r } ( Z = 1 )$ be the proportion of the treated units.

Theorem 20.1 Assumption 20.1 implies that $\eta \leq e \leq 1 - \eta$ and

$$
\begin{array}{l} p ^ {- 1} \sum_ {k = 1} ^ {p} \left| E (X _ {k} \mid Z = 1) - E (X _ {k} \mid Z = 0) \right| \\ \leq p ^ {- 1 / 2} C ^ {1 / 2} \left\{e \lambda_ {1} ^ {1 / 2} + (1 - e) \lambda_ {0} ^ {1 / 2} \right\}, \tag {20.1} \\ \end{array}
$$

where

$$
C = \frac {(e - \eta) (1 - e - \eta)}{e ^ {2} (1 - e) ^ {2} \eta (1 - \eta)}
$$

is a positive constant depending only on $( e , \eta )$ , and $\lambda _ { 1 }$ and $\lambda _ { 0 }$ are the maximum eigenvalues of the covariance matrices cov $( X \mid Z = 1 )$ and $\operatorname { c o v } ( X \mid Z = 0 )$ , respectively.

What is the order of the maximum eigenvalues in Theorem 20.1? D’Amour et al. (2021) showed that it is usually smaller than $O ( p )$ unless the components of X are highly correlated. If the components of X are highly correlated, then some components are redundant after including other components. If the components of $X$ are not highly correlated, then the right-hand side converges to zero. So the average difference in means of the covariates is close to zero, that is, the treatment and control groups are nearly balanced in means averaged over all dimensions of the covariates. Mathematically, the left-hand side of (20.1) converging to zero rules out the possibility that all dimensions of X have non-vanishing difference in means across treatment and control groups. It is a strong requirement in observational studies with many covariates.

## 20.1.1 Trimming in the presence of limited overlap

When Assumption 20.1 does not hold, it is common to trim the units based on the estimated propensity scores (Crump et al., 2009; Yang and Ding, 2018). Trimming drops units within regions of little overlap, which changes the population and estimand. The restrictive implications of overlap in Assumption 20.1 suggest that trimming must be employed more often and one may need to trim a large proportion of units to achieve desirable overlap in high dimensions.

## 20.1.2 Outcome modeling in the presence of limited overlap

The somewhat negative results in D’Amour et al. (2021) also highlight the limitation of focusing only on the propensity score in the presence of limited overlap. With high dimensional covariates, outcome modeling becomes more important. In particular, if the outcome means only depend on a function of the original covariates in that

$$
E \{Y (z) \mid X \} = f _ {z} (r (X)), \quad (z = 0, 1)
$$

then it suffices to control for $r ( X )$ , a lower dimensional summary of the original covariates. Due to the dimension reduction, the strict overlap condition on $r ( X )$ can be much weaker than the strict overlap condition on X. This is conceptually straightforward, but the corresponding theory and methods are missing.

## 20.2 Causal inference with no overlap: regression discontinuity

Starting from the simple case with a univariate X. An extreme treatment assignment mechanism is a deterministic one:

$$
Z = 1 (X \geq x _ {0}),
$$

where x0 is a predetermined threshold. An interesting consequence of this assignment is that the unconfoundedness assumption holds automatically:

$$
Z \bot \{Y (1), Y (0) \} \mid X
$$

because Z is a deterministic function of X. However, the overlap assumption is violated by definition:

$$
e (X) = \operatorname{pr} (Z = 1 \mid X) = 1 (X \geq x _ {0}) = \left\{ \begin{array}{l l} 1 & \text { if } X \geq x _ {0}, \\ 0 & \text { if } X <   x _ {0}. \end{array} \right.
$$

So our analytic strategies discussed in Part IV are no longer applicable here. We must change our perspective.

The discussion here seems contrived, with a deterministic treatment assignment. Interestingly, it has many applications in practice, and is called regression discontinuity. Below, I first review some canonical examples and then give a mathematical formulation of this type of studies.

## 20.2.1 Examples and graphical diagnostics

Example 20.1 Thistlethwaite and Campbell (1960) first proposed the idea of regression-discontinuity analysis. Their motivating example was to study the effect of students’ winning Certificated of Merit on later career plans, where the Certificated of Merit was determined by whether the Scholarship Qualifying Test score was above a certain threshold. Their initial analysis was mainly graphical. Figure 20.1 shows one of their graphs.

Example 20.2 Bor et al. (2014) used regression discontinuity to study the effect of when to start HIV patients with antiretroviral on their mortality, where the treatment is determined by whether the patients’ CD4 counts were below 200 cells/µL.1

Example 20.3 Carpenter and Dobkin (2009) studied the effect of alcohol consumption on mortality, which leverages the minimum legal drinking age as a discontinuity for alcohol consumption. They derived mortality data from the National Center for Health Statistics, including the decedent’s date of birth and date of death. They computed age profile of deaths per 100,000 person years with outcomes measured by the following nine variables:

<table><tr><td>all</td><td>all deaths, the sum of internal and external</td></tr><tr><td>internal</td><td>deaths due to internal causes</td></tr><tr><td>external</td><td>deaths due to external causes, the sum of the rest</td></tr><tr><td>homicide</td><td>homicides</td></tr><tr><td>suicide</td><td>suicides</td></tr><tr><td>mva</td><td>motor vehicle accidents</td></tr><tr><td>alcohol</td><td>deaths with a mention of alcohol</td></tr><tr><td>drugs</td><td>deaths with a mention of drug use</td></tr><tr><td>externalother</td><td>deaths due to other external causes</td></tr></table>

Figure 20.2 plots the number of deaths per 100,000 person years for nine measures based on the data used by Angrist and Pischke (2014). From the jumps at age 21, it seems obvious that there is an increase of mortality at age 21, primarily due to motor vehicle accidents. I leave the formal analysis as Problem 20.3.

## 20.2.2 A mathematical formulation of regression discontinuity

The technical term for the variable X that determines the treatment is the running variable. Intuitively, regression discontinuity can identify a local average causal effect at the cutoff point x0:

$$
\tau (x _ {0}) = E \{Y (1) - Y (0) \mid X = x _ {0} \}.
$$

In particular, for the potential outcome under treatment, we have

$$
E \{Y (1) \mid X = x _ {0} \} = \lim _ {\varepsilon \rightarrow 0 +} E \{Y (1) \mid X = x _ {0} + \varepsilon \} \tag {20.2}
$$

$$
= \lim _ {\varepsilon \rightarrow 0 +} E \{Y (1) \mid Z = 1, X = x _ {0} + \varepsilon \} \tag {20.3}
$$

$$
= \lim _ {\varepsilon \to 0 +} E (Y \mid Z = 1, X = x _ {0} + \varepsilon), \tag {20.4}
$$

where (20.2) holds if $E \{ Y ( 1 ) \mid X = x \}$ is continuous from the right at $x _ { 0 }$ and (20.3) follows by the definition of Z. Similarly, for the potential outcome under control, we have

$$
E \{Y (0) \mid X = x _ {0} \} = \lim _ {\varepsilon \rightarrow 0 +} E (Y \mid Z = 0, X = x _ {0} - \varepsilon)
$$

if $E \{ Y ( 0 ) \mid X = x \}$ is continuous from the left at $x _ { 0 }$ . So the local average causal effect at $x _ { 0 }$ can be identified by the difference of the two limits. I summarize the key identification result below.

Theorem 20.2 Assume that $E \{ Y ( 1 ) \mid X = x \}$ is continuous from the right at $x _ { 0 }$ and $E \{ Y ( 0 ) \mid X = x \}$ is continuous from the left at $x _ { 0 }$ . Then the local average treatment effect at $X = x _ { 0 }$ is identified by

$$
\tau (x _ {0}) = \lim _ {\varepsilon \to 0 +} E (Y \mid Z = 1, X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \to 0 +} E (Y \mid Z = 0, X = x _ {0} - \varepsilon).
$$

Since the right-hand side of the above equation only involves observables, the parameter $\tau ( x _ { 0 } )$ is nonparametrically identified. However, the form of the identification formula is totally different from what we derived before. In particular, the identification formula involves limits of two conditional expectation functions.

## 20.2.3 Regressions near the boundary

If we are lucky, graphical diagnostic sometimes can clearly show the causal effect at the cutoff point. However, many outcomes are noisy so graphical diagnostic is not enough. Figure 20.3 shows two examples with obvious jumps at the cutoff point and two examples without obvious jumps, although the underlying data generating processes all have discontinuities.

Assume that $E ( Y \mid Z = 1 , X = x ) = \gamma _ { 1 } + \beta _ { 1 } x$ and $E ( Y \mid Z = 0 , X = x ) =$ $\gamma _ { 0 } + \beta _ { 0 } x$ are linear in x. We can run OLS based on the treated and control data to obtain the fitted lines $\hat { \gamma } _ { 1 } + \hat { \beta } _ { 1 } x$ and $\hat { \gamma } _ { 0 } + \hat { \beta } _ { 0 } x .$ , respectively. We can then estimate the average causal effect at the point $X = x _ { 0 }$ as

$$
\hat {\tau} (x _ {0}) = (\hat {\gamma} _ {1} - \hat {\gamma} _ {0}) + (\hat {\beta} _ {1} - \hat {\beta} _ {0}) x _ {0}.
$$

Numerically, $\hat { \tau } ( x _ { 0 } )$ is identical to the coefficient of $Z _ { i }$ in the OLS

$$
Y _ {i} \sim \{1, Z _ {i}, X _ {i} - x _ {0}, Z _ {i} (X _ {i} - x _ {0}) \}, \tag {20.5}
$$

and it is also identical to the coefficient of $Z _ { i }$ in the OLS

$$
Y _ {i} \sim \{1, Z _ {i}, R _ {i}, L _ {i} \}, \tag {20.6}
$$

where

$$
R _ {i} = \max (X _ {i} - x _ {0}, 0), \quad L _ {i} = \min (X _ {i} - x _ {0}, 0)
$$

indicate the right and left parts of $X _ { i } - x _ { 0 }$ , respectively. I leave the algebraic details to Problem 20.1.

However, this approach may be sensitive to the violation of the linear model. Theory suggests running regression using only the local observations near the cutoff point2. However, the rule for choosing the “local points” are quite involved. Fortunately, the rdrobust function in the rdrobust package in R implements various choices of “local points.” Since choosing the “local points” is the key in regression discontinuity, it seems more sensible to report estimates and confidence intervals based on various choices of the “local points.”

## 20.2.4 An example

Lee (2008) gave a famous example of using regression discontinuity to study the incumbency advantage in the U.S. House. He wrote that “incumbents are, by definition, those politicians who were successful in the previous election. If what makes them successful is somewhat persistent over time, they should be expected to be somewhat more successful when running for re-election.” Therefore, this is a fundamentally challenging causal inference problem. The regression discontinuity is a clever study design to study this problem.

The running variable is the lagged vote in the previous election centered at 0, and the outcome is the vote in current election, with units being the congressional districts. The treatment is the binary indicator for being the current incumbent party in a district, determined by the lagged vote. Figure 20.4 show the raw data.

The rdrobust function gives three sets of the point estimate and confidence intervals. They all suggest positive incumbency advantage.

```txt
> library(rdrobust)
> library(rddtools)
> data(house)
> RDDest = rdrobust(house$y, house$x)
[1] "Mass points detected in the running variable."
> cbind(RDDest$coef, RDDest$ci)
Coeff CI Lower CI Upper
Conventional 0.06372533 0.04224798 0.08520269
Bias-Corrected 0.05937028 0.03789292 0.08084763
Robust 0.05937028 0.03481238 0.08392818
```

Figure 20.5 shows the point estimates and the confidence intervals based on OLS with different choices of the local points defined by $| X | < x _ { 0 }$ . While the point estimates and the confidence intervals are sensitive to the choice of $x _ { 0 } .$ , the qualitative result remains the same as above.

## 20.2.5 Problems of regression discontinuity

What can go wrong with the regression discontinuity analysis? The technical challenge is to specify the neighborhood near the cutoff point. We have discussed this issue above.

In addition, Theorem 20.2 holds under a continuity condition. It may be violated in practice. For instance, if the mortality rate jumps at the age of 21, then the jumps in Figure 20.2 may not be due to the change of drinking behavior due to the legal drink age. However, it is hard to check the violation of the continuity condition empirically.

McCrary (2008) proposed an indirect test for the validity of the regression discontinuity. He suggested checking the density of the running variable at the cutoff point. The discontinuity in the density of the running variable at the cutoff point may suggest that some units were able to manipulate their treatment status perfectly.

## 20.3 Homework Problems

## 20.1 Linear potential outcome models

This problem gives more details for the numerical equivalence in Section 20.2.3.

Show that $\hat { \tau } ( x _ { 0 } )$ equals the coefficients of $Z _ { i }$ in OLS fits (20.5) and (20.5).

Hint: It is helpful for start with the figures of $Z _ { i } ( X _ { i } - x _ { 0 } ) , L _ { i } ,$ and $R _ { i }$ with $X _ { i }$ on the x-axis. The conclusion holds by reparametrizating the OLS regressions.

## 20.2 Simulation for regression discontinuity

RDDnumerical.R simulates potential outcomes from linear models and generates Figure 20.3. Change them to nonlinear models, and compare different point estimators and confidence intervals, including the biases and variances of the point estimators, and the coverage properties of confidence intervals.

## 20.3 Re-analysis of the data on the minimum legal drink age

Analyze the data mlda.csv of Carpenter and Dobkin (2009).

## 20.4 Recommended reading

D’Amour et al. (2021) discussed the implications of overlap with high dimensional covariates.

Thistlethwaite and Campbell (1960)’s original paper on regression discontinuity was re-printed as Thistlewaite and Campbell (2016) with many insightful comments. Coincidentally, Thistlethwaite and Campbell (1960) and Rubin (1974) were both published in the Journal of Educational Psychology.

## Part V

## Instrumental variables

## 21
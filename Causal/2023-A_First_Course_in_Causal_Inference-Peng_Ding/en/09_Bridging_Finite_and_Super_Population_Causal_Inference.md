# Bridging Finite and Super Population Causal Inference

We have focused on the finite population perspective in randomized experiment. It treats all the potential outcomes as fixed numbers or conditions on them if they are realizations of some random variables. The advantage of this perspective is that it focuses on the design of the experiments and requires minimal assumptions on the data generating process of the outcomes. However, it is often criticized for having only internal validity but not necessarily external validity. Obviously, all experimenters care about not only the internal validity but also the external validity of their experiments. Since all statistical properties are conditional on the potential outcomes for the units we have, the results are only about the observed units. Then a natural question arises: do the finite population results generalize to a bigger population?

This is a fair critique on the finite population framework conditional on the potential outcomes. However, this can be a philosophical question. What we observed is a finite population, so any experimental design and analysis directly give us information about this finite population. Randomization only ensures internal validity given the potential outcomes of these units. The external validity of the results depend on the sampling process of the units. If the finite population is a representative sample of a larger population we are interested in, then of course the experimental results also have external validity. Otherwise, the results based on randomization inference may not generalize. Pearl and Bareinboim (2014) discussed this transportability problem from a different perspective.

For some statisticians, this is just a technical problem. We can change the statistical framework, assuming that the units are sampled from a super population. Then all the statements are about the population of interest. This is a convenient framework, although it does not really solve the problem mentioned above. Below, I will introduce this framework for two purposes: first, it gives a different perspective for randomized experiments; second, it serves as a bridge between Parts II and III of this book. The latter purpose is more important, since the super population framework allows us to derive more fruitful results for observational studies in which the treatment is not randomly assigned.

## 9.1 CRE

Assume

$$
\{Z _ {i}, Y _ {i} (1), Y _ {i} (0), X _ {i} \} _ {i = 1} ^ {n} \stackrel {{\text {IID}}} {{\sim}} \{Z, Y (1), Y (0), X \}
$$

from a super population. With a little abuse of notation, we define the population average causal effect as

$$
\tau = E \{Y (1) - Y (0) \} = E \{Y (1) \} - E \{Y (0) \}.
$$

Under the super population framework, we can formulate the CRE as below.

## Definition 9.1 (CRE under the super population framework) Z $\{ Y ( 1 ) , Y ( 0 ) , X \}$

Under Definition 9.1, the average causal effect can be written as

$$
\begin{array}{l} \tau = E \{Y (1) \mid Z = 1 \} - E \{Y (0) \mid Z = 0 \} \\ = E (Y \mid Z = 1) - E (Y \mid Z = 0), \tag {9.1} \\ \end{array}
$$

which equals the difference in expectations of the outcomes. Since τ can be expressed as a function of the distributions of the observables, it is nonparametrically identifiable1. The identification formula (9.1) immediately suggests a moment estimator ˆτ , which is the difference in means of the outcomes defined before. Conditioning on Z, this is then a standard two-sample problem comparing the means of two independent samples. We have

$$
E (\hat {\tau} \mid \mathbf {Z}) = \tau , \quad \mathrm{var} (\hat {\tau} \mid \mathbf {Z}) = \frac {\mathrm{var} \{Y (1) \}}{n _ {1}} + \frac {\mathrm{var} \{Y (0) \}}{n _ {0}}.
$$

Under IID sampling, the sample variances are unbiased for the population variances, so Neyman (1923)’s variance estimator is unbiased for $\mathrm { v a r } ( \hat { \tau } \mid Z )$ . The conservativeness problem goes away under this super population framework.

We can also discuss covariate adjustment. Based on the OLS decompositions (see Chapter A2)

$$
Y (1) = \gamma_ {1} + \beta_ {1} ^ {\mathsf {T}} X + \varepsilon (1), \tag {9.2}
$$

$$
Y (0) = \gamma_ {0} + \beta_ {0} ^ {\mathsf {T}} X + \varepsilon (0), \tag {9.3}
$$

we have

$$
\tau = E \{Y (1) - Y (0) \} = \gamma_ {1} - \gamma_ {0} + (\beta_ {1} - \beta_ {0}) ^ {\mathsf {T}} E (X),
$$

since the residuals $\varepsilon ( 1 )$ and $\varepsilon ( 0 )$ have mean zero due to the inclusion of the intercepts. We can use the OLS with the treated and control data to estimate the coefficients in (9.2) and (9.3), respectively. The sample versions of the coefficients are $\hat { \gamma } _ { 1 } , \hat { \beta } _ { 1 } , \hat { \gamma } _ { 0 } , \hat { \beta } _ { 0 }$ , so a covariate-adjusted estimator for τ is

$$
\hat {\tau} _ {\mathrm{adj}} = \hat {\gamma} _ {1} - \hat {\gamma} _ {0} + (\hat {\beta} _ {1} - \hat {\beta} _ {0}) ^ {\mathsf {T}} \bar {X}.
$$

If we center covariates with $\bar { X } = 0$ , the above estimator reduces to Lin (2013)’s estimator

$$
\hat {\tau} _ {\mathrm{L}} = \hat {\gamma} _ {1} - \hat {\gamma} _ {0},
$$

which equals the coefficient of Z in the pooled regression with treatmentcovariates interactions.

Unfortunately, the EHW variance estimator does not work for $\hat { \tau } _ { \mathrm { L } }$ because of the additional uncertainty X¯ under the super population framework. Berk et al. (2013), Negi and Wooldridge (2021) and Zhao and Ding (2021a) proposed a correction of the EHW variance estimator by adding an additional term

$$
(\hat {\beta} _ {1} - \hat {\beta} _ {0}) ^ {\mathsf {T}} S _ {X} ^ {2} (\hat {\beta} _ {1} - \hat {\beta} _ {0}) / n.
$$

A conceptually simpler yet computationally intensive approach is to use the bootstrap to estimate the variance; see Chapter A1.5.

## 9.2 SRE

We can extend the discussion in Section 9.1 to the SRE since it is equivalent to independent CREs within strata. The notation below will be slightly different from that in Chapter 5.

Assume that

$$
\{Z _ {i}, Y _ {i} (1), Y _ {i} (0), X _ {i} \} \stackrel {{\text {IID}}} {{\sim}} \{Z, Y (1), Y (0), X \}.
$$

With a discrete covariate $X _ { i } \in \{ 1 , \ldots , K \}$ , we can formulate the SRE as below.

Definition 9.2 (SRE under the super population framework) $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid$ X.

Under Definition 9.2, the conditional average causal effect can be rewritten as

$$
\tau_ {[ k ]} = E \{Y (1) - Y (0) \mid X = k \} = E (Y \mid Z = 1, X = k) - E (Y \mid Z = 0, X = k),
$$

so the average causal effect can be rewritten as

$$
\tau = E \{Y (1) - Y (0) \} = \sum_ {k = 1} ^ {K} \operatorname{pr} (X _ {-} k) E \{Y (1) - Y (0) \mid X = k \} = \sum_ {k = 1} ^ {K} \operatorname{pr} (X = k) \tau_ {[ k ]}.
$$

The discussion in Section 9.1 holds with all strata, so we can derive the super population analog for the SRE. When there are more than two treatment and control units within each strata, we can use $\hat { V } _ { \mathrm { S } }$ as an unbiased variance estimator for var(ˆτS).

## 9.3 Homework Problems

## 9.1 OLS decomposition of the observed outcome under the CRE

Based on (9.2) and (9.3), show that the OLS decomposition of the observed outcome on the treatment, covariates and their interaction is

$$
Y = \alpha_ {0} + \alpha_ {Z} Z + \alpha_ {X} ^ {\mathsf {T}} X + \alpha_ {Z X} ^ {\mathsf {T}} X Z + \varepsilon
$$

where

$$
\alpha_ {0} = \gamma_ {0}, \quad \alpha_ {Z} = \gamma_ {1} - \gamma_ {0}, \quad \alpha_ {X} = \beta_ {0}, \quad \alpha_ {Z X} = \beta_ {1} - \beta_ {0}, \quad \varepsilon = Z \varepsilon (1) + (1 - Z) \varepsilon (0).
$$

That is,

$$
(\alpha_ {0}, \alpha_ {Z}, \alpha_ {X}, \alpha_ {Z X}) = \arg \min _ {a _ {0}, a _ {Z}, a _ {X}, a _ {Z X}} E (Y - a _ {0} - a _ {Z} Z - a _ {X} ^ {\mathsf {T}} X - a _ {Z X} ^ {\mathsf {T}} X Z) ^ {2}.
$$

## 9.2 Recommended reading

Ding et al. (2017a) provide a unified discussion of the finite-population and super-population inferences for the average causal effect.

## Part III

## Observational studies

## 10
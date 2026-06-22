# The Average Causal Effect on the Treated Units and Other Estimands

Chapters 10–12 focused on the identification and estimation of the average causal effect $\tau = E \{ Y ( 1 ) - Y ( 0 ) \}$ under the unconfoundedness and overlap assumptions. Conceptually, it is straightforward to extend the discussion to the average causal effects on the treated and control units:

$$
\tau_ {\mathrm{T}} = E \{Y (1) - Y (0) \mid Z = 1 \},
$$

$$
\tau_ {\mathrm{C}} = E \{Y (1) - Y (0) \mid Z = 0 \}.
$$

Because of the symmetry, this chapter focuses on $\tau _ { \mathrm { T } }$ and also included extensions to other estimands.

## 13.1 Nonparametric identification of $\tau _ { \mathbf { T } }$

The average causal effect on the treated units equals

$$
\tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E \{Y (0) \mid Z = 1 \},
$$

where the first term $E ( Y \mid Z = 1 )$ is directly identifiable from the data and the second term $E \{ Y ( 0 ) ~ | ~ Z = 1 \}$ is counterfactual. The key assumption to identify the second term is the following unconfoundedness and overlap assumptions.

Assumption 13.1 $Z \underline { { \mathrm { 1 1 } } } Y ( 0 ) \mid X$ and $e ( X ) < 1$ .

Because the key is to identify $E \{ Y ( 0 ) \mid Z = 1 \}$ , we only need the $^ { 6 6 } \mathrm { o n e - }$ - sided” unconfoundedness and overlap assumptions. Under Assumption 13.1, we have the following identification result for $\tau _ { \mathrm { T } }$ .

Theorem 13.1 Under Assumption 13.1, we have

$$
\begin{array}{l} E \{Y (0) \mid Z = 1 \} = E \left\{E (Y \mid Z = 0, X) \mid Z = 1 \right\} \\ = \int E (Y \mid Z = 0, X = x) F (\mathrm{d} x \mid Z = 1). \\ \end{array}
$$

16413 The Average Causal Effect on the Treated Units and Other Estimands

Theorem 13.1 implies that $\tau _ { \mathrm { T } }$ is nonparmetrically identified by

$$
\tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E \left\{E (Y \mid Z = 0, X) \mid Z = 1 \right\} \tag {13.1}
$$

Proof of Theorem 13.1: We have

$$
\begin{array}{l} E \{Y (0) \mid Z = 1 \} = E \left[ E \{Y (0) \mid Z = 1, X \} \mid Z = 1 \right] \\ = E \left[ E \{Y (0) \mid Z = 0, X \} \mid Z = 1 \right] \\ = E \left\{E (Y \mid Z = 0, X) \mid Z = 1 \right\} \\ = \int E (Y \mid Z = 0, X = x) F (\mathrm{d} x \mid Z = 1). \\ \end{array}
$$

![image_11](images/image_11.png)

With a discrete X, the identification formula in Theorem 13.1 reduces to

$$
E \{Y (0) \mid Z = 1 \} = \sum_ {k = 1} ^ {K} E (Y \mid Z = 0, X = k) \mathrm{pr} (X = k \mid Z = 1),
$$

motivating the following stratified estimator for $\tau _ { \mathrm { T } }$

$$
\hat {\tau} _ {\mathrm{T}} = \hat {\bar {Y}} (1) - \sum_ {k = 1} ^ {K} \hat {\pi} _ {[ k ] | 1} \hat {\bar {Y}} _ {[ k ]} (0),
$$

where $\hat { \pi } _ { [ k ] | 1 } = n _ { [ k ] 1 } / n _ { 1 }$ is the proportion of category k of X among the treated units.

For continuous X, we need to fit an outcome model for $E ( Y \mid Z = 0 , X )$ 号 using the control units. If the fitted values for the control potential outcomes are $\hat { \mu } _ { 0 } ( X _ { i } )$ , then the outcome regression estimator is

$$
\hat {\tau} _ {\mathrm{T}} = \hat {\bar {Y}} (1) - n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \hat {\mu} _ {0} (X _ {i}) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \{Y _ {i} - \hat {\mu} _ {0} (X _ {i}) \}.
$$

Example 13.1 If we specify a linear model for all units

$$
E (Y \mid Z, X) = \beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X,
$$

then

$$
\begin{array}{l} \tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E (\beta_ {0} + \beta_ {x} ^ {\mathsf {T}} X \mid Z = 1) \\ = E (Y \mid Z = 1) - \beta_ {0} - \beta_ {x} ^ {\mathsf {T}} E (X \mid Z = 1). \\ \end{array}
$$

$I f$ we run OLS to obtain $( \hat { \beta } _ { 0 } , \hat { \beta } _ { z } , \hat { \beta } _ { x } )$ , then the estimator is

$$
\hat {\tau} _ {\mathrm{T}} = \hat {\bar {Y}} (1) - \hat {\beta} _ {0} - \hat {\beta} _ {x} ^ {\mathsf {T}} \hat {\bar {X}} (1).
$$

Using the property of the OLS (see A2.3), we have

$$
\sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\beta} _ {0} - \hat {\beta} _ {z} Z _ {i} - \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}) = 0 \Longrightarrow \hat {\bar {Y}} (1) - \hat {\beta} _ {0} - \hat {\beta} _ {z} - \hat {\beta} _ {x} ^ {\mathsf {T}} \hat {\bar {X}} (1) = 0.
$$

Therefore, the above estimator reduces to $\hat { \tau } _ { \mathrm { T } } = \hat { \beta } _ { z } ,$ , the OLS coefficient of Z.

By the property of the OLS, we can also write $\hat { \beta } _ { z }$ as the difference in means of the adjusted outcome $Y _ { i } - \hat { \beta } _ { x } ^ { \sf T } X _ { i }$ , resulting in

$$
\begin{array}{l} \hat {\tau} _ {\mathrm{T}} = \left\{\hat {\bar {Y}} (1) - \hat {\beta} _ {x} ^ {\mathsf {T}} \hat {\bar {X}} (1) \right\} - \left\{\hat {\bar {Y}} (0) - \hat {\beta} _ {x} ^ {\mathsf {T}} \hat {\bar {X}} (0) \right\} \\ = \left\{\hat {\bar {Y}} (1) - \hat {\bar {Y}} (0) \right\} - \hat {\beta} _ {x} ^ {\mathsf {T}} \left\{\hat {\bar {X}} (1) - \hat {\bar {X}} (0) \right\}. \tag {13.2} \\ \end{array}
$$

Therefore, τˆT equals the simple difference in means of the outcome, adjusted by the imbalance of the covariates in the treatment and control groups.

Section $\it 1 0 . 4 . 2$ shows that $\hat { \beta } _ { z }$ is an estimator for τ , and this example further shows that $\hat { \beta } _ { z }$ is an estimator for $\tau _ { \mathrm { T } }$ . This is not surprising because the linear model assumes constant causal effects across units.

Example 13.2 The identification formula depends only on $E ( Y \mid Z = 0 , X )$ , so we need only to specify a model for the control units. When this model is linear,

$$
E (Y \mid Z = 0, X) = \beta_ {0 | 0} + \beta_ {x | 0} ^ {\mathsf {T}} X,
$$

we have

$$
\begin{array}{l} \tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E (\beta_ {0 | 0} + \beta_ {x | 0} ^ {\mathsf {T}} X \mid Z = 1) \\ = E (Y \mid Z = 1) - \beta_ {0 | 0} - \beta_ {x | 0} ^ {\mathsf {T}} E (X \mid Z = 1). \\ \end{array}
$$

If we run OLS with only the control units to obtain $( \hat { \beta } _ { 0 | 0 } , \hat { \beta } _ { x | 0 } )$ , then the estimator is

$$
\hat {\tau} _ {\mathrm{T}} = \hat {\bar {Y}} (1) - \hat {\beta} _ {0 | 0} - \hat {\beta} _ {x | 0} ^ {\mathsf {T}} \hat {\bar {X}} (1).
$$

Using the property of the OLS (see A2.3), we have

$$
\hat {\bar {Y}} (0) = \hat {\beta} _ {0 | 0} + \hat {\beta} _ {x | 0} ^ {\mathsf {T}} \hat {\bar {X}} (0).
$$

Therefore, the above estimator reduces to

$$
\hat {\tau} _ {\mathrm{T}} = \left\{\hat {\bar {Y}} (1) - \hat {\bar {Y}} (0) \right\} - \hat {\beta} _ {x | 0} ^ {\mathsf {T}} \left\{\hat {\bar {X}} (1) - \hat {\bar {X}} (0) \right\},
$$

which is similar to (13.2) with a different coefficient for the difference in means of the covariates.

As an algebraic fact, we can show that this estimator equals the coefficient of Z in the OLS fit of the outcome on the treatment, covariates, and their interactions, with the covariates centered $b y \hat { \bar { X } } ( 1 )$ . See Problem 13.1 for more details.

## 13.2 Inverse propensity score weighting and doubly robust estimation of $\tau_{\mathbf{T}}$

Theorem 13.2 Under Assumption 13.1, we have

$$
E \{Y (0) \mid Z = 1 \} = E \left\{\frac {e (X)}{e} \frac {1 - Z}{1 - e (X)} Y \right\} \tag {13.3}
$$

and

$$
\tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E \left\{\frac {e (X)}{e} \frac {1 - Z}{1 - e (X)} Y \right\}, \tag {13.4}
$$

where $e = \operatorname { p r } ( Z = 1 )$ is the marginal probability of the treatment.

Proof of Theorem 13.2: The left-hand side of (13.3) equals

$$
\begin{array}{l} E \{Y (0) \mid Z = 1 \} = E \{Z Y (0) \} / e \\ = E \left[ E (Z \mid X) E \{Y (0) \mid X \} \right] / e \\ = E \left[ e (X) E \{Y (0) \mid X \} \right] / e. \\ \end{array}
$$

The right-hand side of (13.3) equals

$$
\begin{array}{l} E \left\{\frac {e (X)}{e} \frac {1 - Z}{1 - e (X)} Y \right\} = E \left[ E \left\{\frac {e (X)}{e} \frac {1 - Z}{1 - e (X)} Y (0) \mid X \right\} \right] \\ { = } { E \left[ \frac { e ( X ) } { e \{ 1 - e ( X ) \} } E \left\{ ( 1 - Z ) Y ( 0 ) \mid X \right\} \right] } \\ { = } { E \left[ \frac { e ( X ) } { e \{ 1 - e ( X ) \} } E ( 1 - Z \mid X ) E \{ Y ( 0 ) \mid X \} \right] } \\ = E \left[ e (X) E \{Y (0) \mid X \} \right] / e. \\ \end{array}
$$

So (13.3) holds.

We have two inverse propensity score weighting estimators

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{ht}} = \hat {\bar {Y}} (1) - n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) Y _ {i}
$$

and

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{hajek}} = \hat {\bar {Y}} (1) - \frac {\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) Y _ {i}}{\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i})},
$$

where $\hat { o } ( X _ { i } ) = \hat { e } ( X _ { i } ) / \{ 1 - \hat { e } ( X _ { i } ) \}$ is the fitted odds of the treatment given covariates.

The estimation of $E ( Y \mid Z = 1 )$ is simple. We have a doubly robust

## 13.3 Inverse propensity score weighting and doubly robust estimation $o f \tau _ { \mathrm { T } }$ 167

estimator for $E \{ Y ( 0 ) \mid Z = 1 \}$ which combines the propensity score and the outcome model. Define

$$
\tilde {\mu} _ {0 \mathrm{T}} ^ {\mathrm{dr}} = E \left[ o (X, \alpha) (1 - Z) \{Y - \mu_ {0} (X, \beta_ {0}) \} + Z \mu_ {0} (X, \beta_ {0}) \right] / e, \tag {13.5}
$$

where $o ( X , \alpha ) = e ( X , \alpha ) / \{ 1 - e ( X , \alpha ) \}$ .

Theorem 13.3 Under Assumption 13.1, if either $\begin{array} { l l l } { e ( X , \alpha ) } & { = } & { e ( X ) } \end{array}$ or $\mu _ { 0 } ( X , \beta _ { 0 } ) = \mu _ { 0 } ( X )$ , then $\mu _ { 0 \mathrm { T } } ^ { d r } = E \{ Y ( 0 ) \mid Z = 1 \}$ .

Proof of Theorem 13.3: We have the decomposition

$$
\begin{array}{l} e \left[ \tilde {\mu} _ {0 \mathrm{T}} ^ {\mathrm{dr}} - E \{Y (0) \mid Z = 1 \} \right] \\ = E \left[ o (X, \alpha) (1 - Z) \{Y (0) - \mu_ {0} (X, \beta_ {0}) \} + Z \mu_ {0} (X, \beta_ {0}) \right] - E \{Z Y (0) \} \\ = E [ o (X, \alpha) (1 - Z) \{Y (0) - \mu_ {0} (X, \beta_ {0}) \} - Z \{Y (0) - \mu_ {0} (X, \beta_ {0}) \} ] \\ = E \left[ \left\{o (X, \alpha) (1 - Z) - Z \right\} \left\{Y (0) - \mu_ {0} (X, \beta_ {0}) \right\} \right] \\ = E \left[ \frac {e (X , \alpha) - Z}{1 - e (X , \alpha)} \{Y (0) - \mu_ {0} (X, \beta_ {0}) \} \right] \\ = E \left[ E \left\{\frac {e (X , \alpha) - Z}{1 - e (X , \alpha)} \mid X \right\} \times E \{Y (0) - \mu_ {0} (X, \beta_ {0}) \mid X \} \right] \\ = E \left[ \frac {e (X , \alpha) - e (X)}{1 - e (X , \alpha)} \times \{\mu_ {0} (X) - \mu_ {0} (X, \beta_ {0}) \} \right]. \\ \end{array}
$$

Therefore, $\tilde { \mu } _ { 0 \mathrm { T } } ^ { \mathrm { d r } } - E \{ Y ( 0 ) \mid Z = 1 \} = 0$ if either $e ( X , \alpha ) = e ( X ) { \mathrm { o r } } \mu _ { 0 } ( X , \beta _ { 0 } ) =$ $\mu _ { 0 } ( X )$ . □

From the population versions of $\tilde { \mu } _ { \mathrm { 0 T } } ^ { \mathrm { d r } }$ , we can construct the sample version by the following steps:

1. obtain the fitted values of the propensity scores $e ( X , { \hat { \alpha } } )$ ;  
2. obtain the fitted values of the outcome mean under control $\mu _ { 0 } ( X , { \hat { \beta } } _ { 0 } )$ ;  
3. construct the doubly robust estimator: $\hat { \tau } _ { \mathrm { T } } ^ { \mathrm { d r } } = \hat { \bar { Y } } ( 1 ) - \hat { \mu } _ { 0 \mathrm { T } } ^ { \mathrm { d r } }$ , where

$$
\hat {\mu} _ {0 \mathrm{T}} ^ {\mathrm{dr}} = \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} \left[ e (X _ {i}, \hat {\alpha}) \frac {(1 - Z _ {i}) \{Y _ {i} - \mu_ {0} (X _ {i} , \hat {\beta} _ {0}) \}}{1 - e (X _ {i} , \hat {\alpha})} + Z _ {i} \mu_ {0} (X _ {i}, \hat {\beta} _ {0}) \right];
$$

4. estimate the variance of $\tau _ { \mathrm { T } }$ via the bootstrap by resampling from $( Z _ { i } , X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ .

Hahn (1998), Mercatanti and Li (2014), Shinozaki and Matsuyama (2015) and Yang and Ding (2018) are references discussing the estimation of $\tau _ { \mathrm { T } }$ .

## 13.3 An example

The following R code implements two outcome regression estimators, two IPW estimators, and the doubly robust estimator for τT, as well as the bootstrap variance estimators. To avoid extreme estimated propensity scores, we can also truncated them from the above.

```r
ATT.est = function(z, y, x, out.family = gaussian, Utruncpscore = 1)
{
    ## sample size
    nn = length(z)
    nn1 = sum(z)

    ## fitted propensity score
    pscore = glm(z ~ x, family = binomial)$fitted.values
    pscore = pmin(Utruncpscore, pscore)
    odds.pscore = pscore/(1 - pscore)

    ## fitted potential outcomes
    outcome0 = glm(y ~ x, weights = (1 - z),
    family = out.family)$fitted.values

    ## regression imputation estimator
    ace.reg0 = lm(y ~ z + x)$coef[2]
    ace.reg = mean(y[z==1]) - mean(outcome0[z==1])
    ## propensity score weighting estimator
    ace.ipw0 = mean(y[z==1]) - mean(odds.pscore*(1 - z)*y)*nn/nn1
    ace.ipw = mean(y[z==1]) - mean(odds.pscore*(1 - z)*y)/mean(odds.pscore*(1 - z))
    ## doubly robust estimator
    res0 = y - outcome0
    ace.dr = ace.reg - mean(odds.pscore*(1 - z)*res0)*nn/nn1

    return(c(ace.reg0, ace.reg, ace.ipw0, ace.ipw, ace.dr))
}

OS_ATT = function(z, y, x, n.boot = 10^2,
    out.family = gaussian, Utruncpscore = 1)
{
    point.est = ATT.est(z, y, x, out.family, Utruncpscore)

    ## nonparametric bootstrap
    n.sample = length(z)
    x = as.matrix(x)
    boot.est = replicate(n.boot,
    {id.boot = sample(1:n.sample, n.sample, replace = TRUE)
```

```txt
ATT.est(z[id.boot], y[id.boot], x[id.boot, ], out.family, Utruncpscore))
```

```txt
boot.se = apply(boot.est, 1, sd)
res = rbind(point.est, boot.se)
rownames(res) = c("est", "se")
colnames(res) = c("reg0", "reg", "HT", "Hajek", "DR")
return(res)
}
```

Now we re-analyze the data in Example 10.3 to estimate τT. We obtain

```csv
reg0 reg HT Hajek DR
est 0.061 -0.351 -1.992 -0.351 -0.187
se 0.227 0.258 0.705 0.328 0.287
```

without truncating the estimated propensity scores, and

```batch
reg0 reg HT Hajek DR
est 0.061 -0.351 -0.597 -0.192 -0.230
se 0.223 0.255 0.579 0.302 0.276
```

by truncating the estimated propensity scores from the above at 0.9. The HT estimator is sensitive to the truncation as expected. The regression estimator in Example 13.1 is quite different from other estimators. It imposes an unnecessary assumption that the regression functions in the treatment and control group share the same coefficient of X. The regression estimator in Example 13.2 is much close to the Hajek and doubly robust estimators. The estimates above are slightly different from those in Section 12.3.3, suggesting some treatment effect heterogeneity across τT and τ .

## 13.4 Other estimands

Li et al. (2018a) gave a unified discussion of the causal estimands in observational studies. Starting from the conditional average causal effect τ (X), they proposed a general class of estimands

$$
\tau^ {h} = \frac {E \{h (X) \tau (X) \}}{E \{h (X) \}}
$$

indexed by a weighting function $h ( X )$ with $E \{ h ( X ) \} \ne 0$ . The normalization in the denominator is to ensure that a constant causal effect $\tau ( X ) = \tau$ averages to the same τ .

Under the unconfoundedness assumption,

$$
\tau^ {h} = \frac {E [ h (X) \{\mu_ {1} (X) - \mu_ {0} (X) \} ]}{E \{h (X) \}}
$$

## 17013 The Average Causal Effect on the Treated Units and Other Estimands

which motivates the outcome regression estimator

$$
\hat {\tau} ^ {h} = \frac {\sum_ {i = 1} ^ {n} h (X _ {i}) \{\hat {\mu} _ {1} (X _ {i}) - \hat {\mu} _ {0} (X _ {i}) \}}{\sum_ {i = 1} ^ {n} h (X _ {i})}.
$$

Moreover, we can show that $\tau ^ { h }$ has the following weighting form:

Theorem 13.4 Under ignorability and overlap, we have

$$
\tau^ {h} = E \left\{\frac {Z Y h (X)}{e (X)} - \frac {(1 - Z) Y h (X)}{1 - e (X)} \right\} / E \{h (X) \}.
$$

The proof of Theorem 13.4 is similar to those of Theorems 11.2 and 13.2 which is relegated to Problem 13.8. Based on Theorem 13.4, we can construct the corresponding IPW estimator.

By Theorem 13.4, each unit is associated with the weight due to the definition of the estimand as well as the weight due to the inverse of the propensity score. Finally, the treated units are weighted by $h ( X ) / e ( X )$ and the control units are weighted by $h ( X ) / \{ 1 - e ( X ) \}$ . Li et al. (2018a, Table 1) summarized several estimands, and I present a part of it below:

<table><tr><td>population</td><td>h(X)</td><td>estimand</td><td>weights</td></tr><tr><td>combined</td><td>1</td><td> $\tau$ </td><td> $1/e(X)$  and  $1/\{1-e(X)\}$ </td></tr><tr><td>treated</td><td>e(X)</td><td> $\tau_{\text{T}}$ </td><td>1 and e(X)/ $\{1-e(X)\}$ </td></tr><tr><td>control</td><td>1-e(X)</td><td> $\tau_{\text{C}}$ </td><td> $\{1-e(X)\}/e(X)$  and 1</td></tr><tr><td>overlap</td><td>e(X){1-e(X)}</td><td> $\tau_{\text{O}}$ </td><td>1-e(X) and e(X)</td></tr></table>

The overlap population and the corresponding estimand

$$
\tau_ {\mathrm{O}} = \frac {E [ e (X) \{1 - e (X) \} \tau (X) ]}{E [ e (X) \{1 - e (X) \} ]}
$$

is new to us. This estimand has the largest weight for units with $e ( X ) = 1 / 2$ and downweights the units with extreme propensity scores. A nice feature of this estimand is that its IPW estimator is rather stable without the possibly extremely small values of $e ( X )$ and $1 - e ( X )$ in the denominator. If $e ( X ) { \underline { { \bot \bot } } } \tau ( X )$ including the special case of $\tau ( X ) = \tau ,$ the parameter $\tau _ { \mathrm { O } }$ reduces to τ . In general, however, the estimand $\tau _ { \mathrm { O } }$ may cause controversy because it changes the initial population and depends on the propensity score which may be misspecified in practice. Li et al. (2018a) and Li et al. (2019) gave some justifications and numerical evidence. This estimand will appear again in Chapter 14.

We can also construct the doubly robust estimator for $\tau ^ { h }$ . I relegate the details to Problem 13.9.

## 13.5 Homework Problems

## 13.1 An algebraic fact about a regression estimator $f o r \ T _ { \mathrm { T } }$

This problem provides more details for Example 13.2.

Show that if we center the covariates by $X _ { i } - \hat { \bar { X } } ( 1 )$ for all units, then $\hat { \tau } _ { \mathrm { T } }$ equals the coefficient of $Z$ in the OLS fit of the outcome on the treatment, covariates, and their interactions.

## 13.2 Simulation for the average causal effect on the treated units

In OSATE.R in Chapter 12, I ran some simulation studies for τ . Run similar simulation studies for $\tau _ { \mathrm { T } }$ with either correct or incorrect propensity score or outcome models.

You can choose different model parameters, larger numbers of simulation and bootstrap replicates. Report your findings, including at least the bias, variance, and variance estimator via the bootstrap. You can also report other properties of the estimators, for example, the asymptotic Normality and the coverage rates of the confidence intervals.

## 13.3 An alternative form of the doubly robust estimator for $\tau _ { \mathrm { T } }$

Motivated by (13.5), we have an alternative form of doubly robust estimator for $E \{ Y ( 0 ) \mid Z = 1 \}$ }:

$$
\tilde {\mu} _ {0 \mathrm{T}} ^ {\mathrm{dr2}} = \frac {E [ o (X , \alpha) (1 - Z) \{Y - \mu_ {0} (X , \beta_ {0}) \} ]}{E [ o (X , \alpha) (1 - Z) ]} + E \{Z \mu_ {0} (X, \beta_ {0}) \} / e.
$$

Show that under Assumption 13.1, $\tilde { \mu } _ { 0 \mathrm { T } } ^ { \mathrm { d r 2 } } = E \{ Y ( 0 ) | Z = 1 \}$ if either $e ( X , \alpha ) = e ( X ) \mathrm { o r } \mu _ { 0 } ( X , \beta _ { 0 } ) = \mu _ { 0 } ( X )$ . Give the sample analogue of the doubly robust estimator for $\tau _ { \mathrm { T } }$ .

## 13.4 Average causal effect on the control units

Prove the identification formulas for $\tau _ { \mathrm { { C } } } .$ , analogous to (13.1) and (13.4). Propose the doubly robust estimator for $\tau _ { \mathrm { C } }$ .

## 13.5 Estimating individual effect and conditional average causal effect

$\{ Z _ { i } , X _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \} _ { i = 1 } ^ { n } \stackrel { \mathrm { I I D } } { \sim } \{ Z , X , Y ( 1 ) , Y ( 0 ) \}$ effect is $\tau _ { i } = Y _ { i } ( 1 ) - Y _ { i } ( 0 )$ and the conditional average causal effect is $\tau ( X _ { i } ) =$ $E \{ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \mid X _ { i } \}$ . Since we will discuss individual effect, we do not drop the subscript i since τ mean the average causal effect, not the population version of $Y ( 1 ) - Y ( 0 )$ .

1. Under randomization with $Z _ { i } \bot \bot \{ Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \}$ and $e = \mathrm { p r } ( Z _ { i } = 1 )$ ,

## 17213 The Average Causal Effect on the Treated Units and Other Estimands

show that

$$
\delta_ {i} = \frac {Z _ {i} Y _ {i}}{e} - \frac {(1 - Z _ {i}) Y _ {i}}{1 - e}
$$

is an unbiased predictor of the individual effect in the sense that

$$
E (\delta_ {i} - \tau_ {i}) = 0 (i = 1, \dots , n).
$$

Further show that $E ( \delta _ { i } ) = \tau$ for all $i = 1 , \ldots , n .$ .

2. Under ignorability with $Z _ { i } \bot \bot \{ Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \} \quad | \quad X _ { i }$ and $e ( X _ { i } ) \ =$ pr $\ \cdot Z _ { i } = 1 \mid X _ { i } )$ , show that

$$
\delta_ {i} = \frac {Z _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - Z _ {i}) Y _ {i}}{1 - e (X _ {i})}
$$

is an unbiased predictor of the individual effect and the conditional average causal effect in the sense that

$$
E \left(\delta_ {i} - \tau_ {i}\right) = 0, \quad E \left\{\delta_ {i} - \tau \left(X _ {i}\right) \right\} = 0, \quad (i = 1, \dots , n).
$$

Further show that $E ( \delta _ { i } ) = \tau$ for all $i = 1 , \ldots , n .$ .

## 13.6 General estimand and $( \tau _ { \mathrm { T } } , \tau _ { \mathrm { C } } )$

Assume unconfoundedness. Show that $\tau ^ { h } = \tau _ { \mathrm { T } } \ \mathrm { i f } \ h ( X ) = e ( X )$ , and $\tau ^ { h } = \tau _ { \mathrm { { C } } }$ if $h ( X ) = 1 - e ( X )$ .

## 13.7 More on $\tau _ { \mathrm { O } }$

Show that

$$
\tau_ {\mathrm{O}} = \frac {E [ \{1 - e (X) \} \tau (X) \mid Z = 1 ]}{E \{1 - e (X) \mid Z = 1 \}} = \frac {E \{e (X) \tau (X) \mid Z = 0 \}}{E \{e (X) \mid Z = 0 \}}.
$$

## 13.8 IPW for the general estimand

Prove Theorem 13.4.

## 13.9 Doubly robust estimation for general estimand

For a given $h ( X )$ , we have the following formulas for constructing the doubly robust estimator for $\tau ^ { h }$ :

$$
\begin{array}{l} \tilde {\mu} _ {1} ^ {h, \mathrm{dr}} = E \left[ \frac {Z h (X) \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X , \alpha)} + h (X) \mu_ {1} (X, \beta_ {1}) \right], \\ \tilde {\mu} _ {0} ^ {h, \mathrm{dr}} = E \left[ \frac {(1 - Z) h (X) \{Y - \mu_ {0} (X , \beta_ {0}) \}}{1 - e (X , \alpha)} + h (X) \mu_ {0} (X, \beta_ {0}) \right]. \\ \end{array}
$$

Show that under ignorability and overlap,

## 13.5 Homework Problems

1. if either e(X, α) = e(X) or $\mu _ { 1 } ( X , \beta _ { 1 } ) \ = \ \mu _ { 1 } ( X )$ , then $\tilde { \mu } _ { 1 } ^ { h , \mathrm { d r } } ~ =$ E{h(X)Y (1)};  
2. if either e(X, α) = e(X) or $\mu _ { 0 } ( X , \beta _ { 0 } ) \ = \ \mu _ { 0 } ( X )$ , then $\tilde { \mu } _ { 0 } ^ { h , \mathrm { d r } } ~ =$ E{h(X)Y (0)};  
3. if either $e ( X , \alpha ) ~ = ~ e ( X ) ~ \mathrm { o r } ~ \{ \mu _ { 1 } ( X , \beta _ { 1 } ) ~ = ~ \mu _ { 1 } ( X ) , \mu _ { 0 } ( X , \beta _ { 0 } ) ~ =$ $\mu _ { 0 } ( X ) \}$ , then

$$
\frac {\tilde {\mu} _ {1} ^ {h , \mathrm{dr}} - \tilde {\mu} _ {0} ^ {h , \mathrm{dr}}}{E \{h (X) \}} = \tau^ {h}.
$$

Remark: Tao and Fu (2019) proved the above results. However, they hold only for a given $h ( X )$ . The most interesting cases of $\tau _ { \mathrm { T } } , \tau _ { \mathrm { C } }$ and $\tau _ { \mathrm { O } }$ all have weight depending on the propensity score $e ( X )$ , which must be estimated in the first place. The above formulas do not apply to constructing the doubly robust estimators for $\tau _ { \mathrm { T } }$ and $\tau _ { \mathrm { { C } } } ;$ there does not exist a doubly robust estimator for $\tau _ { \mathrm { O } }$ .

## 13.10 Recommended reading

Shinozaki and Matsuyama (2015) focused on $\tau _ { \mathrm { T } }$ , and Li et al. (2018a) discussed general $\tau ^ { h }$ .
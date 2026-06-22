# The Doubly Robust or the Augmented Inverse Propensity Score Weighting Estimator for the Average Causal Effect

Under unconfoundedness $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$ and overlap $0 < e ( X ) < 1$ , Chapter 11 has shown two identification formulas of the average causal effect $\tau = E \{ Y ( 1 ) - Y ( 0 ) \}$ . First, the outcome imputation formula is

$$
\tau = E \{\mu_ {1} (X) \} - E \{\mu_ {0} (X) \} \tag {12.1}
$$

where

$$
\mu_ {1} (X) = E \{Y (1) \mid X \} = E (Y \mid Z = 1, X),
$$

$$
\mu_ {0} (X) = E \{Y (0) \mid X \} = E (Y \mid Z = 0, X)
$$

are the two conditional mean functions of the outcome given covariates. Second, the inverse propensity score weighting (IPW) formula is

$$
\tau = E \left\{\frac {Z Y}{e (X)} \right\} - E \left\{\frac {(1 - Z) Y}{1 - e (X)} \right\} \tag {12.2}
$$

where

$$
e (X) = \operatorname{pr} (Z = 1 \mid X)
$$

is the propensity score introduced in Chapter 11.

The outcome imputation estimator requires fitting a model for the outcome given the treatment and covariates. It is consistent if the outcome model is correctly specified. The IPW estimator requires fitting a model for the treatment given covariates. It is consistent if the propensity score model is correctly specified.

Mathematically, we have many combinations of (12.1) and (12.2) that lead to different identification formulas of the average causal effect. Below I will discuss a particular combination that has appealing theoretical properties. This combination motivates an estimator that is consistent if either the propensity score or the outcome model is correctly specified. It is call the doubly robust estimator, championed by James Robins (Scharfstein et al., 1999; Bang and Robins, 2005).

## 12.1 The doubly robust estimator

## 12.1.1 Population version

We posit a working model for the conditional means of the outcome $\mu _ { 1 } ( X , \beta _ { 1 } )$ and $\mu _ { 0 } ( X , \beta _ { 0 } )$ , indexed by the parameters $\beta _ { 1 }$ and $\beta _ { 0 }$ . For example, if the conditional means are linear or logistic under the working model, then the parameters are just the regression coefficients. If the outcome model is correctly specified, then $\mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X )$ and $\mu _ { 0 } ( X , \beta _ { 0 } ) = \mu _ { 0 } ( X )$ . We posit a working model for the propensity score $e ( X , \alpha )$ , indexed by the parameter α. For example, if the working model is logistic, then α is the regression coefficient. If the propensity score model is correctly specified, then $e ( X , \alpha ) = e ( X )$ . In practice, both models may be misspecified.

Define

$$
\tilde {\mu} _ {1} ^ {\mathrm{dr}} = E \left[ \frac {Z \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X , \alpha)} + \mu_ {1} (X, \beta_ {1}) \right], \tag {12.3}
$$

$$
\tilde {\mu} _ {0} ^ {\mathrm{dr}} = E \left[ \frac {(1 - Z) \{Y - \mu_ {0} (X , \beta_ {0}) \}}{1 - e (X , \alpha)} + \mu_ {0} (X, \beta_ {0}) \right], \tag {12.4}
$$

which can also be written as

$$
\tilde {\mu} _ {1} ^ {\mathrm{dr}} = E \left[ \frac {Z Y}{e (X , \alpha)} - \frac {Z - e (X , \alpha)}{e (X , \alpha)} \mu_ {1} (X, \beta_ {1}) \right], \tag {12.5}
$$

$$
\tilde {\mu} _ {0} ^ {\mathrm{dr}} = E \left[ \frac {(1 - Z) Y}{1 - e (X , \alpha)} - \frac {e (X , \alpha) - Z}{1 - e (X , \alpha)} \mu_ {0} (X, \beta_ {0}) \right]. \tag {12.6}
$$

The formulas in (12.3) and (12.4) augment the outcome imputation estimator by inverse propensity score weighting terms of the residuals. The formulas in (12.5) and (12.6) augment the IPW estimator by the imputed outcomes. For this reason, the doubly robust estimator is also called the augmented inverse propensity score weighting (AIPW) estimator.

The augmentation strengthens the theoretical properties in the following sense.

Theorem 12.1 Assume unconfoundedness Z $\{ Y ( 1 ) , Y ( 0 ) \} \mid X$ and overlap $1 < e ( X ) < 1$ .

$\begin{array} { r c l } { { } } & { { I . ~ I f ~ e i t h e r ~ e ( X , \alpha ) = ~ e ( X ) ~ o r ~ \mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X ) , ~ t h e n ~ \tilde { \mu } _ { 1 } ^ { \mathrm { d r } } = } } \\ { { } } & { { } } & { { E \{ Y ( 1 ) \} . } } \end{array}$  
2. If either e(X, α) = e(X) or µ0(X, β0) = µ0(X), then $\tilde { \mu } _ { 0 } ^ { \mathrm { d r } } =$ E{Y (0)}.  
$\begin{array} { r l } & { \beta . \mathrm { ~ } J f \mathrm { ~ } e i t h e r \mathrm { ~ } e ( X , \alpha ) = e ( X ) \mathrm { ~ } o r \mathrm { ~ } \{ \mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X ) , \mu _ { 0 } ( X , \beta _ { 0 } ) = } \\ & { \mu _ { 0 } ( X ) \} , \mathrm { ~ } t h e n \tilde { \mu } _ { 1 } ^ { \mathrm { { d r } } } - \tilde { \mu } _ { 0 } ^ { \mathrm { { d r } } } = \tau . } \end{array}$

By Theorem 12.1, $\tilde { \mu } _ { 1 } ^ { \mathrm { d r } } - \tilde { \mu } _ { 0 } ^ { \mathrm { d r } }$ equals τ if either the propensity score model or the outcome model is correctly specified. That’s why it is called the doubly robust estimator.

Proof of Theorem 12.1: I only prove the result for $\mu _ { 1 } = E \{ Y ( 1 ) \}$ . The proof for the result for $\mu _ { 0 } = E \{ Y ( 0 ) \}$ is similar. We have the decomposition

$$
\begin{array}{l} \tilde {\mu} _ {1} ^ {\mathrm{dr}} - E \{Y (1) \} = E \left[ \frac {Z \{Y (1) - \mu_ {1} (X , \beta_ {1}) \}}{e (X , \alpha)} - \{Y (1) - \mu_ {1} (X, \beta_ {1}) \} \right] \\ = E \left[ \frac {Z - e (X , \alpha)}{e (X , \alpha)} \{Y (1) - \mu_ {1} (X, \beta_ {1}) \} \right] \\ = E \left(E \left[ \frac {Z - e (X , \alpha)}{e (X , \alpha)} \{Y (1) - \mu_ {1} (X, \beta_ {1}) \} \mid X \right]\right) \\ = E \left[ E \left\{\frac {Z - e (X , \alpha)}{e (X , \alpha)} \mid X \right\} \times E \left\{Y (1) - \mu_ {1} (X, \beta_ {1}) \mid X \right\} \right] \\ = E \left[ \frac {e (X) - e (X , \alpha)}{e (X , \alpha)} \times \{\mu_ {1} (X) - \mu_ {1} (X, \beta_ {1}) \} \right]. \\ \end{array}
$$

Therefore, $\tilde { \mu } _ { 1 } ^ { \mathrm { d r } } - E \{ Y ( 1 ) \} = 0$ if either $e ( X , \alpha ) = e ( X ) { \mathrm { o r } } \mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X )$

## 12.1.2 Sample version

From the population versions of $\tilde { \mu } _ { 1 } ^ { \mathrm { d r } }$ and $\tilde { \mu } _ { 0 } ^ { \mathrm { d r } }$ , we can construct the sample versions by the following steps:

1. obtain the fitted values of the propensity scores: $e ( X , { \hat { \alpha } } )$ ;  
2. obtain the fitted values of the outcome means: $\mu _ { 1 } ( X , { \hat { \beta } } _ { 1 } )$ and $\mu _ { 0 } ( X , { \hat { \beta } } _ { 0 } )$ ;  
3. construct the doubly robust estimator: $\hat { \tau } ^ { \mathrm { d r } } = \hat { \mu } _ { 1 } ^ { \mathrm { d r } } - \hat { \mu } _ { 0 } ^ { \mathrm { d r } }$ , where

$$
\hat {\mu} _ {1} ^ {\mathrm{dr}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left[ \frac {Z _ {i} \{Y _ {i} - \mu_ {1} (X _ {i} , \hat {\beta} _ {1}) \}}{e (X _ {i} , \hat {\alpha})} + \mu_ {1} (X _ {i}, \hat {\beta} _ {1}) \right]
$$

and

$$
\hat {\mu} _ {0} ^ {\mathrm{dr}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left[ \frac {(1 - Z _ {i}) \{Y _ {i} - \mu_ {0} (X _ {i} , \hat {\beta} _ {0}) \}}{1 - e (X _ {i} , \hat {\alpha})} + \mu_ {0} (X _ {i}, \hat {\beta} _ {0}) \right];
$$

4. approximate the variance of ${ \hat { \tau } } ^ { \mathrm { d r } }$ via the nonparametric bootstrap by resampling from $( Z _ { i } , X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ (Funk et al., 2011).

Analogous to (12.5) and (12.6), we can also rewrite $\hat { \mu } _ { 1 } ^ { \mathrm { d r } }$ and $\hat { \mu } _ { 0 } ^ { \mathrm { d r } }$ as

$$
\hat {\mu} _ {1} ^ {\mathrm{dr}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left[ \frac {Z _ {i} Y _ {i}}{e (X _ {i} , \hat {\alpha})} - \frac {Z _ {i} - e (X _ {i} , \hat {\alpha})}{e (X _ {i} , \hat {\alpha})} \mu_ {1} (X _ {i}, \hat {\beta} _ {1}) \right],
$$

$$
\hat {\mu} _ {0} ^ {\mathrm{dr}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left[ \frac {(1 - Z _ {i}) Y _ {i}}{1 - e (X _ {i} , \hat {\alpha})} - \frac {e (X _ {i} , \hat {\alpha}) - Z _ {i}}{1 - e (X _ {i} , \hat {\alpha})} \mu_ {0} (X _ {i}, \hat {\beta} _ {0}) \right].
$$

## 12.2 More intuition and theory for the doubly robust estimator

Although the beginning of this chapter claims that the basic identification formulas based on outcome regression and inverse propensity score weight immediately yield infinitely many other identification formulas, the particular forms of the double robust estimators in (12.3) and (12.4) are not obvious to come up with. The original motivation for (12.3) and (12.4) was quite theoretical, which relies on the semiparametric efficiency theory in advanced mathematical statistics (Bickel et al., 1993). It is beyond the level of this book. Below I will give two more intuitive perspectives to construct (12.3) and (12.4). Both Sections 12.2.1 and 12.2.2 below focus on the estimation of $E \{ Y ( 1 ) \}$ since the estimation of $E \{ Y ( 0 ) \}$ is similar by symmetry.

## 12.2.1 Reducing the variance of the IPW estimator

The IPW estimator for $\mu _ { 1 }$ based on

$$
\mu_ {1} = E \left\{\frac {Z Y}{e (X)} \right\}
$$

completely ignores the outcome model of Y . It has the advantages of being consistent without assuming any outcome model. However, if the covariates are predictive to the outcome, the residual based on a working outcome model usually has smaller variance than the outcome even if this working outcome model is wrong. With a possibly mis-specified outcome model $\mu _ { 1 } ( X , \beta _ { 1 } )$ , a trivial decomposition holds:

$$
\mu_ {1} = E \{Y (1) \} = E \{Y (1) - \mu_ {1} (X, \beta_ {1}) \} + E \{\mu_ {1} (X, \beta_ {1}) \}.
$$

If we apply the IPW formula to the first term in the above formula viewing $Y ( 1 ) - \mu _ { 1 } ( X , \beta _ { 1 } )$ as a pseudo potential outcome under the treatment, we can rewrite the above formula as

$$
\mu_ {1} = E \left\{\frac {Z \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X)} \right\} + E \{\mu_ {1} (X, \beta_ {1}) \} \tag {12.7}
$$

$$
= E \left\{\frac {Z \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X)} + \mu_ {1} (X, \beta_ {1}) \right\}, \tag {12.8}
$$

which holds if the propensity score model is correct without assuming that the outcome model is correct. Using a working model to improve efficiency is an old idea from survey sampling. Little and An (2004) and Lumley et al. (2011) pointed out its connection with the doubly robust estimator.

## 12.2.2 Reducing the bias of the outcome regression estimator

The discussion in Section 12.2.1 starts with the IPW estimator and improves its efficiency based on a working outcome model. Alternatively, we can also start with an outcome regression estimator based on

$$
\tilde {\mu} _ {1} = E \{\mu_ {1} (X, \beta_ {1}) \}
$$

which may not be the same as $\mu _ { 1 }$ since the outcome may be wrong. The bias of this estimator is $E \{ \mu _ { 1 } ( X , \beta _ { 1 } ) - Y ( 1 ) \}$ , which can be estimated by an IPW estimator

$$
B = E \left\{\frac {Z \{\mu_ {1} (X , \beta_ {1}) - Y \}}{e (X)} \right\}
$$

if the propensity score model is correct. So a de-biased estimator is $\tilde { \mu } _ { 1 } - B$ , which is identical to (12.8).

## 12.3 Examples

## 12.3.1 Summary of some canonical estimators for τ

The following R implements the outcome imputation, Hovitz–Thompson, Hajek, and doubly robust estimators for τ . These estimators can be conveniently implemented based on the fitted values of the glm function. The default choice for the propensity score model is the logistic model, and the default choice for the outcome model is the linear model with out.family = gaussian1. For binary outcomes, we can also specify out.family = binomial to fit the logistic model.

```txt
OS_est = function(z, y, x, out.family = gaussian,
    truncpscore = c(0, 1))
{
    ## fitted propensity score
    pscore = glm(z ~ x, family = binomial)$fitted.values
    pscore = pmax(truncpscore[1], pmin(truncpscore[2], pscore))
```

```r
## fitted potential outcomes
outcome1 = glm(y ~ x, weights = z,
    family = out.family)$fitted.values
outcome0 = glm(y ~ x, weights = (1 - z),
    family = out.family)$fitted.values

## regression imputation estimator
ace.reg = mean(outcome1 - outcome0)
## IPW estimators
ace.ipw0 = mean(z*y/pscore - (1 - z)*y/(1 - pscore))
ace.ipw = mean(z*y/pscore)/mean(z/pscore) -
    mean((1 - z)*y/(1 - pscore))/mean((1 - z)/(1 - pscore))
## doubly robust estimator
res1 = y - outcome1
res0 = y - outcome0
ace.dr = ace.reg + mean(z*res1/pscore - (1 - z)*res0/(1 - pscore))

return(c(ace.reg, ace.ipw0, ace.ipw, ace.dr))
}
```

It is tedious to calculate the analytic formulas for the variances of the above estimators. The bootstrap provides convenient approximations for the variances based on resampling from $\{ Z _ { i } , X _ { i } , Y _ { i } \} _ { i = 1 } ^ { n }$ . Building upon OSest, the following function returns point estimators as well as the bootstrap standard errors.

```r
OS_ATE = function(z, y, x, n.boot = 2*10^2,
    out.family = gaussian, truncpscore = c(0, 1))
{
    point.est = OS_est(z, y, x, out.family, truncpscore)

    ## nonparametric bootstrap
    n.sample = length(z)
    x = as.matrix(x)
    boot.est = replicate(n.boot,
    {id.boot = sample(1:n.sample, n.sample, replace = TRUE)
    OS_est(z[id.boot], y[id.boot], x[id.boot, ],
    out.family, truncpscore)})
    boot.se = apply(boot.est, 1, sd)

    res = rbind(point.est, boot.se)
    rownames(res) = c("est", "se")
    colnames(res) = c("reg", "HT", "Hajek", "DR")

    return(res)
}
```

## 12.3.2 Simulation

I will use simulation to evaluate the finite-sample properties of the estimators under four scenarios:

1. both the propensity score and outcome models are correct;  
2. the propensity score model is wrong but the outcome model is correct;  
3. the propensity score model is correct but the outcome model is wrong;  
4. both the propensity score and outcome models are wrong.

I will report the average bias, the true standard error, and the average estimated standard error of the estimators over simulation.

In case 1, the data generating process is

```matlab
x = matrix(rnorm(n*2), n, 2)
x1 = cbind(1, x)
beta.z = c(0, 1, 1)
pscore = 1/(1 + exp(- as.vector(x1%* %beta.z)))
z = rbinom(n, 1, pscore)
beta.y1 = c(1, 2, 1)
beta.y0 = c(1, 2, 1)
y1 = rnorm(n, x1%* %beta.y1)
y0 = rnorm(n, x1%* %beta.y0)
y = z*y1 + (1 - z)*y0
```

In case 2, I modify the propensity score model to be nonlinear:

```txt
x1 = cbind(1, x, exp(x))
beta.z = c(-1, 0, 0, 1, -1)
pscore = 1/(1 + exp(- as.vector(x1%* % beta.z)))
```

In case 3, I modify the outcome model to be nonlinear:

```txt
beta.y1 = c(1, 0, 0, 0.2, -0.1)
beta.y0 = c(1, 0, 0, -0.2, 0.1)
y1 = rnorm(n, x1%* %beta.y1)
y0 = rnorm(n, x1%* %beta.y0)
```

In case 4, I modify both the propensity score and the outcome model.

We set the sample size to be n = 500 and generate 500 independent data sets according to the data generating processes above. In case 1,

```batch
reg HT Hajek DR
ave.bias 0.00 0.02 0.03 0.01
true.se 0.11 0.28 0.26 0.13
est.se 0.10 0.25 0.23 0.12
```

All estimators are nearly unbiased. The two weighting estimators have larger variances. In case 2,

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>ave.bias</td><td>0.00</td><td>-0.76</td><td>-0.75</td><td>-0.01</td></tr><tr><td>true.se</td><td>0.12</td><td>0.59</td><td>0.47</td><td>0.18</td></tr><tr><td>est.se</td><td>0.13</td><td>0.50</td><td>0.38</td><td>0.18</td></tr></table>

The two weighting estimators are severely biased due to the misspecification of the propensity score model. The regression imputation and doubly robust estimators are nearly unbiased. In case 3,

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>ave.bias</td><td>-0.05</td><td>0.00</td><td>-0.01</td><td>0.00</td></tr><tr><td>true.se</td><td>0.11</td><td>0.15</td><td>0.14</td><td>0.14</td></tr><tr><td>est.se</td><td>0.11</td><td>0.14</td><td>0.13</td><td>0.14</td></tr></table>

The regression imputation estimator has larger bias than the other three estimators due to the misspecification of the outcome model. The weighting and doubly robust estimators are nearly unbiased. In case 4,

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>ave.bias</td><td>-0.08</td><td>0.11</td><td>-0.07</td><td>0.16</td></tr><tr><td>true.se</td><td>0.13</td><td>0.32</td><td>0.20</td><td>0.41</td></tr><tr><td>est.se</td><td>0.13</td><td>0.25</td><td>0.16</td><td>0.26</td></tr></table>

All estimators are biased because both the propensity score and outcome models are wrong. The Horvitz–Thompson and doubly robust estimator has the largest bias. When both models are wrong, the doubly robust estimator appears to be doubly fragile.

In all the cases above, the boostrap standard errors are close to the true ones when the estimators are nearly unbiased for the true average causal effect.

## 12.3.3 Applications

Revisiting Example 10.3, we obtain the following estimators and bootstrap standard errors:

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>est</td><td>-0.017</td><td>-1.516</td><td>-0.156</td><td>-0.019</td></tr><tr><td>se</td><td>0.230</td><td>0.492</td><td>0.246</td><td>0.233</td></tr></table>

The two weighting estimators are much larger than the other two estimators. Truncating the estimated propensity score at [0.1, 0.9], we obtain the following estimators and bootstrap standard errors:

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>est</td><td>-0.017</td><td>-0.713</td><td>-0.054</td><td>-0.043</td></tr><tr><td>se</td><td>0.223</td><td>0.422</td><td>0.235</td><td>0.231</td></tr></table>

The Hajek estimator becomes much close to the regression imputation and doubly robust estimators, while the Horvitz–Thompson estimator is still an outlier.

## 12.4 Some further discussion

Recall the proof of Theorem 12.1, the key for the double robustness property is the product structure in

$$
\tilde {\mu} _ {1} ^ {\mathrm{dr}} - E \{Y (1) \} = E \left[ \frac {e (X) - e (X , \alpha)}{e (X , \alpha)} \times \{\mu_ {1} (X) - \mu_ {1} (X, \beta_ {1}) \} \right],
$$

which ensures that the estimation error is zero if either $e ( X ) = e ( X , \alpha )$ or $\mu _ { 1 } ( X ) = \mu _ { 1 } ( X , \beta _ { 1 } )$ . This delicate structure renders the doubly robust estimator possibly doubly fragile when both the propensity score and the outcome models are misspecified. The product of two errors multiply to yield potentially much larger errors. Kang and Schafer (2007) criticized the doubly robust estimator based on extensive simulation studies. They found that the finitesample performance of the doubly robust estimator can be even more wild than the simple regression imputation and IPW estimators.

Despite the critique from Kang and Schafer (2007), the doubly robust estimator has been a standard strategy in causal since the seminal work of Scharfstein et al. (1999). Recently, it resurrected in the theoretical statistics and econometrics literature with a fancier name “double machine learning” (Chernozhukov et al., 2018). The basic idea is to replace the working models for the propensity score and outcome by machine learning tools which can be viewed as more flexible models than the traditional parametric models.

## 12.5 Homework problems

## 12.1 A sanity check

Consider the case in which the covariate is discrete $X ~ \in ~ \{ 1 , \ldots , K \}$ and the parameter of interest is $\mu _ { 1 }$ . Without imposing any model assumptions, the estimated propensity score $\hat { e } ( X )$ is the proportion of units receiving the treatment and the estimated outcome mean is the sample mean of the outcome $\hat { \bar { Y } } _ { [ k ] 1 } ~ = ~ \hat { E } ( Y ~ \vert ~ Z ~ = ~ 1 , X ~ = ~ k )$ under treatment, within stratum $X = k \ ( k \stackrel { \cdot } { = } 1 , \ldots , K )$ . Show that the stratified estimator, outcome regression estimator, IPW estimator, and the doubly robust estimator are all the same.

## 12.2 An alternative form of the doubly robust estimator for τ

Motivated by (12.7), we have an alternative form of doubly robust estimator for $\mu _ { 1 }$ :

$$
\tilde {\mu} _ {1} ^ {\mathrm{dr2}} = \frac {E \left[ \frac {Z \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X , \alpha)} \right]}{E \left[ \frac {Z}{e (X , \alpha)} \right]} + E \{\mu_ {1} (X, \beta_ {1}) \}.
$$

Show that $\tilde { \mu } _ { 1 } ^ { \mathrm { d r 2 } } = \mu _ { 1 }$ if either $e ( X , \alpha ) = e ( X )$ or $\mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X )$ ). Give the analogous formula for estimating µ0. Give the sample analogue of the doubly robust estimator for τ based on these formulas. Note that this form of doubly robust estimator appeared in Robins et al. (2007).

## 12.3 Data analysis of Example 10.1

Analyze the dataset cps1re74.csv using the methods discussed so far.

## 12.4 Recommended reading

Lunceford and Davidian (2004) gave a nice review and comparison of many methods discussed in Chapters 11 and 12.

## 13
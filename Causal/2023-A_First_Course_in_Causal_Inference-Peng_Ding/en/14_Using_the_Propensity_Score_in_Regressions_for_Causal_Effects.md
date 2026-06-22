# Using the Propensity Score in Regressions for Causal Effects

Since Rosenbaum and Rubin (1983b)’s seminal paper, many creative uses of the propensity score have appeared in the literature (e.g., Bang and Robins, 2005; Robins et al., 2007; Van der Laan and Rose, 2011; Vansteelandt and Daniel, 2014). This chapter discusses two simple methods to use the propensity score: including the propensity score as a covariate in regressions and running regressions weighted by the inverse of the propensity score. I choose to focus on these two methods because

1. they are easy to implement, which involve only standard statistical software packages for regressions;  
2. their properties are comparable to many more complex methods;  
3. they can be easily extended to allow for flexible statistical models including machine learning algorithms.

## 14.1 Regressions with the propensity score as a covariate

By Theorem 11.1, if unconfoundedness holds conditioning on $X$ , then it also holds conditioning on e(X):

$$
Z \bot \{Y (1), Y (0) \} \mid e (X).
$$

Analogous to (10.5), τ is also nonparametrically identified by

$$
\tau = E \Big [ E \{Y \mid Z = 1, e (X) \} - E \{Y \mid Z = 0, e (X) \} \Big ],
$$

which motivates methods based on regressions of $Y$ on $Z$ and $e ( X )$ .

The simplest regression specification is the OLS fit of $Y$ on $\{ 1 , Z , e ( X ) \}$ , with the coefficient of $Z$ as an estimator, denoted by $\tau _ { e } .$ For simplicity, I will discuss the population OLS:

$$
\arg \min _ {a, b, c} E \{Y - a - b Z - c e (X) \} ^ {2}
$$

with $\tau _ { e }$ defined as the coefficient of $Z .$ It is consistent for τ if we have a correct propensity score model and the outcome model is indeed linear in $Z$ and $e ( X )$ . The more interesting result is that $\tau _ { e }$ estimates $\tau _ { \mathrm { O } }$ if we have a correct propensity score model even if the outcome model is completely misspecified.

Theorem 14.1 $I f Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$ , then the coefficient of $Z$ in the OLS fit of $Y$ on $\{ 1 , Z , e ( X ) \}$ } equals

$$
\tau_ {e} = \tau_ {\mathrm{O}} = \frac {E \{h _ {\mathrm{O}} (X) \tau (X) \}}{E \{h _ {\mathrm{O}} (X) \}},
$$

recalling that $h _ { \mathrm { O } } ( X ) = e ( X ) \{ 1 - e ( X ) \} \ a n d \ \tau ( X ) = E \{ Y ( 1 ) - Y ( 0 ) \mid X \}$ .

An unusual feature of Theorem 14.1 is that the overlap condition is not needed any more. Even if some units have propensity score $e ( X )$ equaling 0 or 1, their associate weight $e ( X ) \{ 1 - e ( X )$ is zero so that they do not contribute anything to the final parameter $\tau _ { \mathrm { O } }$ .

Proof of Theorem 14.1: Based on the FWL theorem reviewed in Section $\mathrm { A 2 . 3 }$ , we can obtain $\tau _ { e }$ in two steps: first, we obtain the residual $\tilde { Z }$ from the OLS fit of $Z$ on $\{ 1 , e ( X ) \}$ ; then, we obtain $\tau _ { e }$ from the OLS fit of $Y$ on $\tilde { Z } .$ .

The coefficient of $e ( X )$ in the OLS fit of $Z$ on $\{ 1 , e ( X ) \}$ is

$$
\begin{array}{l} \frac {\operatorname{cov} \{Z , e (X) \}}{\operatorname{var} \{e (X) \}} = \frac {E [ \operatorname{cov} \{Z , e (X) \mid X \} ] + \operatorname{cov} \{E (Z \mid X) , e (X) \}}{\operatorname{var} \{e (X) \}} \\ = \frac {0 + \operatorname{var} \{e (X) \}}{\operatorname{var} \{e (X) \}} = 1, \\ \end{array}
$$

so the intercept is $E ( Z ) - E \{ e ( X ) \} = 0$ and the residual is $\tilde { Z } = Z - e ( X )$ . This makes sense since $Z - e ( X )$ is uncorrelated with any function of $X$ .

Therefore, we can obtain $\tau _ { e }$ from the univariate OLS fit of $Y$ on a centered variable $Z - e ( X )$ :

$$
\tau_ {e} = \frac {\operatorname{cov} \{Z - e (X) , Y \}}{\operatorname{var} \{Z - e (X) \}}.
$$

The denominator simplifies to

$$
\begin{array}{l} \operatorname{var} \{Z - e (X) \} = E \{Z - e (X) \} ^ {2} \\ = E \{Z + e (X) ^ {2} - 2 Z e (X) \} \\ = e (X) + e (X) ^ {2} - 2 e (X) ^ {2} = h _ {0} (X). \\ \end{array}
$$

The numerator simplifies to

$$
\begin{array}{l} \operatorname{cov} \{Z - e (X), Y \} \\ = E [ \{Z - e (X) \} Y ] \\ = E [ \{Z - e (X) \} Z Y (1) ] + E [ \{Z - e (X) \} (1 - Z) Y (0) ] \\ (\text { since } Y = Z Y (1) + (1 - Z) Y (0)) \\ = E [ \{Z - Z e (X) \} Y (1) ] - E [ e (X) (1 - Z) Y (0) ] \\ = E [ Z \{1 - e (X) \} Y (1) ] - E [ e (X) (1 - Z) Y (0) ] \\ = E [ e (X) \{1 - e (X) \} \mu_ {1} (X) ] - E [ e (X) \{1 - e (X) \} \mu_ {0} (X) ] \\ (\text { tower   property   and   ignorability }) \\ = E \{h _ {0} (X) \tau (X) \}. \\ \end{array}
$$

The conclusion follows.

From the proof of Theorem 14.1, we can simply run the OLS of $Y$ on the centered treatment $\tilde { Z } = Z - e ( X )$ . Lee (2018) proposed this procedure. Moreover, we can also include X in the OLS fit which may improve efficiency in finite sample. However, this does not change the estimand, which is still $\tau _ { \mathrm { O } }$ . I summarize these two results in the corollary below.

Corollary 14.1 If $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$ , then

(1) the coefficient of $Z - e ( X )$ in the OLS fit of Y on $Z - e ( X )$ or $\{ 1 , Z - e ( X ) \}$ equals $\tau _ { \mathrm { O } }$ ;  
(2) the coefficient of Z in the OLS fit of Y on $\{ 1 , Z , e ( X ) , X \}$ equals $\tau _ { \mathrm { O } } .$ .

Proof of Corollary 14.1: (1) The first result is an intermediate step in the proof of Theorem 14.1. The second result holds because regressing $Y$ on $Z - e ( X ) \ \mathrm { o r } \ \{ 1 , Z - e ( X ) \}$ does not change the coefficient of $Z - e ( X )$ since it has mean zero.

(2) It follows from the fact that

$$
Z - e (X) = Z - 0 - 1 \cdot e (X) - 0 ^ {\mathsf {T}} X
$$

is the residual of the OLS fit of $Z$ on $\{ 1 , e ( X ) , X \}$ , since $Z - e ( X )$ is uncorrelated with any functions of $X$ .

Theorem 14.1 motivates a two-step estimator for $\tau _ { \mathrm { O } } \colon$ first, fit a propensity score model to obtain $\hat { e } ( X _ { i } ) ;$ ; second, run OLS of $Y _ { i }$ on $( 1 , X _ { i } , \hat { e } ( X _ { i } ) )$ to obtain the coefficient of $Z _ { i }$ . Corollary 14.1 motivates another two-step estimator for $\tau _ { \mathrm { O } } \colon$ first, fit a propensity score model to obtain $\hat { e } ( X _ { i } )$ ; second, run OLS of $Y _ { i }$ on $Z _ { i } - \hat { e } ( X _ { i } )$ to obtain the coefficient of $Z _ { i }$ . Although OLS is convenient for obtaining point estimators, the corresponding standard errors are incorrect due to the uncertainty in the first step estimation of the propensity score. We can use the bootstrap to approximate the standard errors.

Robins et al. (1992) discussed many OLS estimators based on the propensity score. The above results seem special cases of their general theory although they did not point out the connection with the estimand under the overlap weight, which was resurrected by Li et al. (2018a). Lee (2018) proposed to regress Y on $Z - e ( X )$ from a different perspective without making connections to the existing results in Robins et al. (1992) and Li et al. (2018a).

Rosenbaum and Rubin (1983b) proposed to estimate the average causal effect based on the OLS fit of Y on $\{ 1 , Z , e ( X ) , Z e ( X ) \}$ . When this outcome model is correct, their estimator is consistent for the average causal effect. However, when the model is incorrect, the corresponding estimator has a much more complicated interpretation. Little and An (2004) suggested constructing estimators based on the OLS of Y on Z and a flexible function of e(X) and showed it enjoys certain doubly robustness property. Due to the complexity in implementation, I omit the discussion.

## 14.2 Regressions weighted by the inverse of the propensity score

## 14.2.1 Average causal effect

We first re-examine the Hajek estimator of τ :

$$
\hat {\tau} ^ {\mathrm{hajek}} = \frac {\sum_ {i = 1} ^ {n} \frac {Z _ {i} Y _ {i}}{\hat {e} (X _ {i})}}{\sum_ {i = 1} ^ {n} \frac {Z _ {i}}{\hat {e} (X _ {i})}} - \frac {\sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})}}{\sum_ {i = 1} ^ {n} \frac {1 - Z _ {i}}{1 - \hat {e} (X _ {i})}},
$$

which equals the difference between the weighted means of the outcomes in the treatment and control groups. Numerically, it is identical to the coefficient of $Z _ { i }$ in the following weighted least squares (WLS) of $Y _ { i }$ on (1, Zi).

Proposition 14.1 τˆhajek equals $\hat { \beta }$ from the following WLS:

$$
(\hat {\alpha}, \hat {\beta}) = \arg \min _ {\alpha , \beta} \sum_ {i = 1} ^ {n} w _ {i} (Y _ {i} - \alpha - \beta Z _ {i}) ^ {2}
$$

with weights

$$
w _ {i} = \frac {Z _ {i}}{\hat {e} (X _ {i})} + \frac {1 - Z _ {i}}{1 - \hat {e} (X _ {i})} = \left\{ \begin{array}{l l} \frac {1}{\hat {e} (X _ {i})} & \text {   if   } Z _ {i} = 1; \\ \frac {1}{1 - \hat {e} (X _ {i})} & \text {   if   } Z _ {i} = 0. \end{array} \right. \tag {14.1}
$$

Imbens (2004) pointed out the result in Proposition 14.1. I leave it as a Problem 14.1. By Proposition 14.1, it is convenient to obtain ˆτhajek based on WLS. However, due to the uncertainty in the estimated propensity score, the standard error reported by WLS is incorrect for the true standard error of $\hat { \tau } ^ { \mathrm { h a j e k } }$ . The bootstrap provides a convenient approximation to the true standard error.

Why does the WLS give a consistent estimator for $\tau ?$ Recall that in the CRE with a constant propensity score, we can simply use the coefficient of $Z _ { i }$ in the OLS fit of $Y _ { i } \mathrm { o n } \left( 1 , Z _ { i } \right)$ to estimate τ . In observational studies, units have different probabilities of receiving the treatment and control, respectively. If we weight the treated units by $1 / e ( X _ { i } )$ and the control units by $1 / \{ 1 - e ( X _ { i } ) \}$ , then they can represent the whole population and we effectively have a pseudo randomized experiment. Consequently, the difference between the weighted means are consistent for τ . The numerical equivalence of $\hat { \tau } ^ { \mathrm { h a j e k } }$ and WLS is not only a fun numerical fact itself but also useful for motivation more complex estimator with covariate adjustment. I give one extension below.

Recall that in the CRE, we can use the coefficient of $Z _ { i }$ in the OLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ ) to estimate $\tau ,$ where the covariates are centered with $\bar { X } =$ 0. This is Lin (2013)’s estimator which uses covariates to improve efficiency. A natural extension to observational studies is to estimate τ using the coefficient of $Z _ { i }$ in the WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ with weights defined in (14.1). Hirano and Imbens (2001) used this estimator in an application. The fully interacted linear model is equivalent to two separate linear models for the treated and control groups. If the linear models

$$
E (Y \mid Z = 1, X) = \beta_ {1 0} + \beta_ {1 x} ^ {\mathsf {T}} X, E (Y \mid Z = 0, X) = \beta_ {0 0} + \beta_ {0 x} ^ {\mathsf {T}} X,
$$

are correctly specified, then both OLS and WLS give consistent estimators for the coefficients and the estimators of the coefficient of $Z$ is consistent for τ. More interestingly, the estimator of the coefficient of Z based on WLS is also consistent for $\tau$ if the propensity score model is correct and the outcome model is incorrect. That is, the estimator based on WLS is doubly robust. Robins et al. (2007) discussed this property and attributed this result to M. Joffe’s unpublished paper. I will give more details below.

Let $\hat { e } ( X _ { i } )$ be the fitted propensity score and $( \mu _ { 1 } ( X _ { i } , \hat { \beta } _ { 1 } ) , \mu _ { 0 } ( X _ { i } , \hat { \beta } _ { 0 } ) )$ be the fitted values of the outcome means based on the WLS. The outcome regression estimator is

$$
\hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{reg}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \mu_ {1} (X _ {i}, \hat {\beta} _ {1}) - \frac {1}{n} \sum_ {i = 1} ^ {n} \mu_ {0} (X _ {i}, \hat {\beta} _ {0})
$$

and the doubly robust estimator for τ is

$$
\hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{dr}} = \hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{reg}} + \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {Z _ {i} \{Y _ {i} - \mu_ {1} (X _ {i} , \hat {\beta} _ {1}) \}}{\hat {e} (X _ {i})} - \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) \{Y _ {i} - \mu_ {0} (X _ {i} , \hat {\beta} _ {0}) \}}{1 - \hat {e} (X _ {i})}.
$$

An interesting result is that this doubly robust estimator equals the outcome regression estimator, which reduces to the coefficient of $Z _ { i }$ in the WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ if we use weights (14.1).

Theorem 14.2 If $\bar { X } = 0 \ a n d \ ( \mu _ { 1 } ( X _ { i } , \hat { \beta } _ { 1 } ) , \mu _ { 0 } ( X _ { i } , \hat { \beta } _ { 0 } ) ) = ( \hat { \beta } _ { 1 0 } + \hat { \beta } _ { 1 x } ^ { \top } X _ { i } , \hat { \beta } _ { 0 0 } +$ $\hat { \beta } _ { 0 x } ^ { \mathsf { T } } X _ { i } )$ based on the WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ with weights (14.1), then

$$
\hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{dr}} = \hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{reg}} = \hat {\beta} _ {1 0} - \hat {\beta} _ {0 0},
$$

which is the coefficient of $Z _ { i }$ in the WLS $\it { \Omega } \mathcal { f } t .$ .

Proof of Theorem 14.2: The WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ is equivalent to two WLS fits based on the treated and control data. Both WLS fits include intercepts, so the first order conditions must satisfy

$$
\sum_ {i = 1} ^ {n} \frac {Z _ {i} (Y _ {i} - \hat {\beta} _ {1 0} - \hat {\beta} _ {1 x} ^ {\intercal} X _ {i})}{\hat {e} (X _ {i})} = 0
$$

and

$$
\sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) (Y _ {i} - \hat {\beta} _ {0 0} - \hat {\beta} _ {0 x} ^ {\mathsf {T}} X _ {i})}{1 - \hat {e} (X _ {i})} = 0.
$$

So the difference between ${ \hat { \tau } } ^ { \mathrm { d r } }$ and $\hat { \tau } ^ { \mathrm { r e g } }$ is exactly zero. Both reduces to

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} (\hat {\beta} _ {1 0} + \hat {\beta} _ {1 x} ^ {\mathsf {T}} X _ {i}) - \frac {1}{n} \sum_ {i = 1} ^ {n} (\hat {\beta} _ {0 0} + \hat {\beta} _ {0 x} ^ {\mathsf {T}} X _ {i}) = \hat {\beta} _ {1 0} - \hat {\beta} _ {0 0} + (\hat {\beta} _ {1 x} - \hat {\beta} _ {0 x}) ^ {\mathsf {T}} \bar {X} = \hat {\beta} _ {1 0} - \hat {\beta} _ {0 0}
$$

with centered covariates. So they both equal the coefficient of $Z _ { i }$ in the WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ . □

Freedman and Berk (2008) discouraged the use of the WLS estimator above based on some simulation studies. They showed that when the outcome model is correct, the WLS estimator is worse than the OLS estimator since the WLS estimator has large variability in their simulation setting with homoskedastic outcomes. This may not be true in general. When the errors have variance proportional to the inverse of the propensity scores, the WLS estimator will be more efficient than the OLS estimator. They also showed that the estimated standard error based on the WLS fit is not consistent for the true standard error because it ignores the uncertainty in the estimated propensity score. This can be easily fixed by using the bootstrap to approximate the variance of the WLS estimator. Nevertheless, they found that “weighting may help under some circumstances” because when the outcome model is incorrect, the WLS estimator is still consistent if the propensity score model is correct.

I end this section with Table 14.1 summarizing the regression estimators for causal effects in both randomized experiments and observational studies.

## 14.2.2 Average causal effect on the treated units

The results for $\tau _ { \mathrm { T } }$ parallel those for τ . First, the Hajek estimator for $\tau _ { \mathrm { T } }$

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{hajek}} = \hat {\bar {Y}} (1) - \frac {\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) Y _ {i}}{\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i})},
$$

with $\hat { o } ( X _ { i } ) = \hat { e } ( X _ { i } ) / \{ 1 - \hat { e } ( X _ { i } ) \}$ , equals the coefficient of $Z _ { i }$ in the following WLS fit $Y _ { i }$ on $( 1 , Z _ { i } )$ .

**TABLE 14.1: Regression estimators in CREs and unconfounded observational studies. The weights $w _ { i } \mathrm { ^ s }$ are defined in (14.1) .**

<table><tr><td></td><td>CRE</td><td>unconfounded observational studies</td></tr><tr><td>without X</td><td> $Y_i \sim Z_i$ </td><td> $Y_i \sim Z_i$  with weights  $w_i$ </td></tr><tr><td>with X</td><td> $Y_i \sim (Z_i, X_i, Z_i X_i)$ </td><td> $Y_i \sim (Z_i, X_i, Z_i X_i)$  with weights  $w_i$ </td></tr></table>

Proposition 14.2 $\hat { \tau } _ { \mathrm { T } } ^ { h a j e k }$ is numerically identical to $\hat { \beta }$ in the following WLS:

$$
(\hat {\alpha}, \hat {\beta}) = \arg \min _ {\alpha , \beta} \sum_ {i = 1} ^ {n} w _ {\mathrm{Ti}} (Y _ {i} - \alpha - \beta Z _ {i}) ^ {2}
$$

with weights

$$
w _ {\mathrm{T} i} = Z _ {i} + (1 - Z _ {i}) \hat {o} (X _ {i}) = \left\{ \begin{array}{l l} 1 & \text {   if   } Z _ {i} = 1; \\ \hat {o} (X _ {i}) & \text {   if   } Z _ {i} = 0. \end{array} \right. \tag {14.2}
$$

Similar to Proposition 14.1, Proposition 14.2 is a pure linear algebra result. I relegate its proof as Problem 14.1.

Second, if we center covariates with $\hat { \bar { X } } ( 1 ) = 0$ , then we can estimate $\tau _ { \mathrm { T } }$ using the coefficient of $Z _ { i }$ in the WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ with weights defined in (14.2). Similarly, this estimator equals the regression estimator

$$
\hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{reg}} = \hat {\bar {Y}} (1) - \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} \mu_ {0} (X _ {i}, \hat {\beta} _ {0}),
$$

which also equals the doubly robust estimator

$$
\hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{dr}} = \hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{reg}} - \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) \{Y _ {i} - \mu_ {0} (X _ {i}, \hat {\beta} _ {0}) \}.
$$

Theorem 14.3 $I f \hat { \bar { X } } ( 1 ) = 0$ and $\mu _ { 0 } ( X _ { i } , \hat { \beta } _ { 0 } ) = \hat { \beta } _ { 0 0 } + \hat { \beta } _ { 0 x } ^ { \top } X _ { i }$ based on the WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , \bar { Z _ { i } } X _ { i } )$ with weights (14.2), then

$$
\hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{dr}} = \hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{reg}} = \hat {\beta} _ {1 0} - \hat {\beta} _ {0 0},
$$

which is the coefficient of $Z _ { i }$ in the WLS $\it { \Omega } \mathcal { f } t .$

Proof of Theorem 14.3: Based on the WLS fits in the treatment and control groups, we have

$$
\sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\beta} _ {1 0} - \hat {\beta} _ {1 x} ^ {\intercal} X _ {i}) = 0, \tag {14.3}
$$

$$
\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) (Y _ {i} - \hat {\beta} _ {0 0} - \hat {\beta} _ {0 x} ^ {\intercal} X _ {i}) = 0. \tag {14.4}
$$

$\hat { \tau } _ { \mathrm { { T , w l s } } } ^ { \mathrm { { d r } } } = \hat { \tau } _ { \mathrm { { T , w l s } } } ^ { \mathrm { { r e g } } }$ = ˆτT,wls. Both reduces to

$$
\hat {\bar {Y}} (1) - \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} (\hat {\beta} _ {0 0} + \hat {\beta} _ {0 x} ^ {\mathsf {T}} X _ {i}) = \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\beta} _ {0 0} - \hat {\beta} _ {0 x} ^ {\mathsf {T}} X _ {i}).
$$

With covariates centered with $\hat { \bar { X } } ( 1 ) = 0 .$ the first result (14.3) implies that $\hat { \bar { Y } } ( 1 ) = \hat { \beta } _ { 1 0 }$ which further simplifies the estimators to $\hat { \beta } _ { 1 0 } - \hat { \beta } _ { 0 0 }$ . □

## 14.3 Homework problems

## 14.1 Hajek estimators as WLS estimators

Prove Propositions 14.1 and 14.2.

Hint: These are special cases of Problem A2.2 on the univariate WLS.

## 14.2 Predictive estimator and doubly robust estimator

Another outcome regression estimator is the predictive estimator

$$
\hat {\tau} ^ {\mathrm{pred}} = \hat {\mu} _ {1} ^ {\mathrm{pred}} - \hat {\mu} _ {0} ^ {\mathrm{pred}}
$$

where

$$
\hat {\mu} _ {1} ^ {\mathrm{pred}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left\{Z _ {i} Y _ {i} + (1 - Z _ {i}) \mu_ {1} (X _ {i}, \hat {\beta} _ {1}) \right\}
$$

and

$$
\hat {\mu} _ {0} ^ {\text { pred }} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left\{Z _ {i} \mu_ {0} (X _ {i}, \hat {\beta} _ {1}) + (1 - Z _ {i}) Y _ {i} \right\}.
$$

It differs from the outcome regression estimator discussed before in that it only predicts the counterfactural outcomes but not the observed outcomes.

Show that the doubly robust estimator equals ˆτpred if $( \mu _ { 1 } ( X _ { i } , \hat { \beta } _ { 1 } ) , \mu _ { 0 } ( X _ { i } , \hat { \beta } _ { 1 } ) ) =$ $( \hat { \beta } _ { 1 0 } + \hat { \beta } _ { 1 x } ^ { \top } X _ { i } , \hat { \beta } _ { 0 0 } + \hat { \beta } _ { 0 x } ^ { \top } X _ { i } )$ are from the WLS fits of $Y _ { i }$ on $( 1 , X _ { i } )$ based on the treated and control data, respectively, with weights

$$
w _ {i} = Z _ {i} / \hat {o} (X _ {i}) + (1 - Z _ {i}) \hat {o} (X _ {i}) = \left\{ \begin{array}{l l} \frac {1}{\hat {o} (X _ {i})} = \frac {1 - \hat {e} (X _ {i})}{\hat {e} (X _ {i})} & \text { if } Z _ {i} = 1; \\ \hat {o} (X _ {i}) = \frac {\hat {e} (X _ {i})}{1 - \hat {e} (X _ {i})} & \text { if } Z _ {i} = 0. \end{array} \right. \tag {14.5}
$$

Remark: Cao et al. (2009) and Vermeulen and Vansteelandt (2015) motivated the weights in (14.5) from other more theoretical perspectives.

<!-- footnote -->

- If the logistic outcome model is correct, then $\hat { \beta } _ { z }$ estimates the conditional odds ratio of the treatment on the outcome given covariates, which does not equal τ. Freedman (2008c) gave an warning of using the logistic regression coefficient to estimate τ in CREs. See Chapter A2 for more details of the logistic regression.

<!-- footnote end -->

<!-- footnote -->

- The glm function is more general than the lm function. With out.family = gaussian, glm is identical to lm.

<!-- footnote end -->

## 14.3 Weighted logistic regression with a binary outcome

With a binary outcome, we can replace linear outcome models by the logistic outcome models. Show that with weights in the logistic regressions, the doubly robust estimators equals the outcome regression estimator. The result holds for both τ and $\tau _ { \mathrm { T } } .$ .

## 14.4 Causal inference with a misspecified linear regression

Define the population OLS of Y on Z, X as

$$
(\beta_ {0}, \beta_ {1}, \beta_ {2}) = \arg \min _ {b _ {0}, b _ {1}, b _ {2}} E (Y - b _ {0} - b _ {1} Z - b _ {2} ^ {\mathsf {T}} X) ^ {2}.
$$

Recall that $e ( X ) = \mathrm { p r } ( Z = 1 \mid X )$ is the propensity score, and define $\tilde { e } ( X ) =$ $\gamma _ { 0 } + \gamma _ { 1 } ^ { \intercal } X$ as the OLS projection of A on X with

$$
(\gamma_ {0}, \gamma_ {1}) = \arg \min _ {c _ {0}, c _ {1}} E (A - c _ {0} - c _ {1} ^ {\mathsf {T}} X) ^ {2}.
$$

1. Show that

$$
\beta_ {1} = \frac {E [ \tilde {w} (X) \{\mu_ {1} (X) - \mu_ {0} (X) \} ]}{E \{\tilde {w} (X) \}} + \frac {E [ \{e (X) - \tilde {e} (X) \} \mu_ {0} (X) ]}{E \{\tilde {w} (X) \}}
$$

where $\tilde { w } ( X ) = e ( X ) \{ 1 - \tilde { e } ( X ) \}$ .

2. When X contains the dummy variables for a discrete covariate, show that

$$
\beta_ {1} = \frac {E [ w (X) \{\mu_ {1} (X) - \mu_ {0} (X) \} ]}{E \{w (X) \}}
$$

where $w ( X ) = e ( X ) \{ 1 - e ( X ) \}$ is the overlap weight.

Remark: Vansteelandt and Dukes (2022) gave the formula in the first part without a detailed proof. The result in part 2 was derived many times in the literature (e.g., Angrist, 1998; Ding, 2021).

## 14.5 Data re-analysis

Re-analyze the dataset in karolinska.txt and the dataset nhanesbmi in the ATE package.

## 14.6 Recommended reading

Kang and Schafer (2007) gave a critical review of the doubly robust estimator, using simulation to compare it with many other estimators. Robins et al. (2007) gave a very insightful comment on Kang and Schafer (2007).
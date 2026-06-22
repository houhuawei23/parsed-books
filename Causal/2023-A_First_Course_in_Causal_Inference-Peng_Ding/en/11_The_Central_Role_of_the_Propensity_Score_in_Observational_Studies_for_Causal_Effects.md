# The Central Role of the Propensity Score in Observational Studies for Causal Effects

Rosenbaum and Rubin (1983b) proposed the key concept propensity score and discussed its role in causal inference with observational studies. It is one of the most cited papers in statistics, and Titterington (2013) listed it as the second most cited paper published in Biometrika during the past 100 years. Its citations are growing very fast during the recent years.

Under the IID sampling assumption, we have four random variables associated with each unit: $\{ X , Z , Y ( 1 ) , Y ( 0 ) \}$ . Following the basic probability rule, we can factorize the joint distribution as

$$
\operatorname{pr} \{X, Z, Y (1), Y (0) \}
$$

$$
= \operatorname{pr} (X) \times \operatorname{pr} \{Y (1), Y (0) \mid X \} \times \operatorname{pr} \{Z \mid X, Y (1), Y (0) \},
$$

where $\mathrm { p r } ( X )$ is the covariate distribution, $\operatorname { p r } \{ Y ( 1 ) , Y ( 0 ) \mid X \}$ is the outcome model, and $\operatorname { p r } \{ Z \mid X , Y ( 1 ) , Y ( 0 ) \}$ is the treatment assignment mechanism. Usually, we do not want to model the covariates because they are background information happening before the treatment and outcome. If we want to move beyond the outcome model, then we must focus on the treatment assignment mechanism, which leads to the definition of the propensity score.

Definition 11.1 (propensity score) $D e f i n e$

$$
e (X, Y (1), Y (0)) = \operatorname{pr} \{Z = 1 \mid X, Y (1), Y (0) \}
$$

as the propensity score. Under strong ignorability, we have

$$
e (X, Y (1), Y (0)) = \operatorname{pr} \{Z = 1 \mid X, Y (1), Y (0) \} = \operatorname{pr} (Z = 1 \mid X),
$$

so the propensity score reduces to

$$
e (X) = \operatorname{pr} (Z = 1 \mid X),
$$

the conditional probability of the receiving the treatment given the observed covariates.

Rosenbaum and Rubin (1983b) used $e ( X ) = \mathrm { p r } ( Z = 1 \mid X )$ as the definition of the propensity score because they focused on observational studies under ignorability. It is sometimes helpful to view $e ( X , Y ( 1 ) , Y ( 0 ) ) = \mathrm { p r } \{ Z =$ $1 \mid X , Y ( 1 ) , Y ( 0 ) \}$ as the general definition of the propensity score even when ignorability fails. See Problem 11.1 for more details.

Following Rosenbaum and Rubin (1983b), this chapter will demonstrate that $e ( X )$ is a key quantity in causal inference with observational studies under ignorability.

## 11.1 The propensity score as a dimension reduction tool

## 11.1.1 Theory

Theorem 11.1 $I f Z \perp \perp \{ Y ( 1 ) , Y ( 0 ) \} \mid X , t h e n Z \perp \{ Y ( 1 ) , Y ( 0 ) \} \mid e ( X ) .$

Theorem 11.1 states that if strong ignorability holds conditional on covariates X, then it also holds conditional on the scalar propensity score $e ( X )$ . The ignorability requires conditioning on many background characteristics Z of the units, but Theorem 11.1 implies that controlling for the propensity score $e ( X )$ romoves all confounding induced by covariates X. The original covariates X can be general and have many dimensions, but the propensity score $e ( X )$ is a one-dimensional scalar variable bounded between 0 and 1. Therefore, the propensity score reduces the dimension of the original covariates but still maintain the ignorability. As a technical statistical terminology, we can view the propensity score as a dimensional reduction tool. We will first prove Theorem 11.1 below and then given an application of the dimension reduction property of the propensity score.

Proof of Theorem 11.1: By the definition of conditional independence, we need to show that

$$
\operatorname{pr} \{Z = 1 \mid Y (1), Y (0), e (X) \} = \operatorname{pr} \{Z = 1 \mid e (X) \}. \tag {11.1}
$$

The left-hand side of (11.1) equals

$$
\begin{array}{l} \operatorname{pr} \{Z = 1 \mid Y (1), Y (0), e (X) \} \\ = E \{Z \mid Y (1), Y (0), e (X) \} \\ = E \left[ E \{Z \mid Y (1), Y (0), e (X), X \} \mid Y (1), Y (0), e (X) \right] \\ (\text { tower   property; see   Section   A1.1.1 }) \\ = E \left[ E \{Z \mid Y (1), Y (0), X \} \mid Y (1), Y (0), e (X) \right] \\ = E \left\{E (Z \mid X) \mid Y (1), Y (0), e (X) \right\} \quad (\text { strong   ignorability }) \\ = E \left\{e (X) \mid Y (1), Y (0), e (X) \right\} \\ = e (X). \\ \end{array}
$$

The right-hand side of (11.1) equals

$$
\begin{array}{l} \operatorname{pr} \{Z = 1 \mid e (X) \} \\ = E \{Z \mid e (X) \} \\ = E \left[ E \{Z \mid e (X), X \} \mid e (X) \right] \quad (\text { tower   property }) \\ = E \left\{E (Z \mid X) \mid e (X) \right\} \\ = E \left\{e (X) \mid e (X) \right\} \\ = e (X). \\ \end{array}
$$

So the left-hand side of (11.1) equals the right-hand side of (11.1).

![image_10](images/image_10.png)

## 11.1.2 Propensity score stratification

Theorem 11.1 motivates a simple method for estimating causal effects: propensity score stratification. Starting from the simple case, we assume that the propensity score is known and only takes K possible values $\{ e _ { 1 } , \ldots , e _ { K } \}$ with K being much smaller than the sample size n. Theorem 11.1 reduces to

$$
Z \bot \{Y (1), Y (0) \} \mid e (X) = e _ {k} \quad (k = 1, \ldots , K).
$$

Therefore, we have a stratified randomized experiment (SRE), that is, we have K independent CREs within strata of the propensity score. We can analyze the observational data in the same way as the SRE stratified on $e ( X )$ .

In general, the propensity score is not known and is not discrete. We often fit a statistical model for $\operatorname { p r } ( Z \ = \ 1 \ | \ X )$ (for example, a logistic model) to obtain the estimated propensity score ${ \hat { e } } ( X )$ . This estimated propensity score can take as many values as the sample size, but we can discretize it to approximate the simple case above. For example, we can discretize the estimated propensity score by its K quantiles to obtain $\hat { e } ^ { \prime } ( X ) \colon \hat { e } ^ { \prime } ( X _ { i } ) = e _ { k } .$ , the $k / K { \mathrm { - t h } }$ quantile of ${ \hat { e } } ( X ) , { \mathrm { i f ~ } } { \hat { e } } ( X _ { i } )$ is between the $( k - 1 ) / K { \cdot } \mathrm { t h }$ and $k / K { \mathrm { - t h } }$ quantiles of ${ \hat { e } } ( X )$ . Then we have

$$
Z \bot \{Y (1), Y (0) \} \mid \hat {e} ^ {\prime} (X) = e _ {k} \quad (k = 1, \ldots , K).
$$

approximately. So we can analyze the observational data in the same way as the SRE stratified on $\hat { e } ^ { \prime } ( X )$ . The ignorability holds only approximately given $\hat { e } ^ { \prime } ( X )$ . We can further use regression adjustment based on covariate to remove bias and improve efficiency. To be more specific, we can obtain Lin (2013)’s estimator within each stratum and construct the final estimator by a weighted average.

With unknown propensity score, we need to fit a statistical model to obtain the estimated propensity score $\hat { e } ( X )$ . This makes the final estimator dependent on the model specification. However, the propensity score stratification estimator only requires the correct ordering of the estimated propensity scores rather than their exact values, which makes it relatively robust compared to other methods. This robustness property of propensity score stratification appeared in many numerical examples but its rigorous quantification is still missing in the literature.

An important practical question is how to choose K? If K is too small, then the strong ignorability does not hold even approximately given $\hat { e } ^ { \prime } ( X )$ . If K is too large, then we do not have enough units within each stratum of the estimated propensity score and many strata have only treated or control units. Therefore, we face a trade-off in practice. Following Cochran (1968)’s heuristics, Rosenbaum and Rubin (1983b) and Rosenbaum and Rubin (1984) suggested K = 5 which removes a large amount of bias in many settings. However, with extremely large dataset, propensity score stratification leads to biased estimators with a fixed K (Lunceford and Davidian, 2004). It is thus reasonable to increase K as long as each stratum has enough treated and control units. Wang et al. (2020) suggested an aggressive choice of K, which is the maximum number of strata such that the stratified estimator is well defined. But the rigorous theory for this procedure is not fully established.

Another important practical question is how to compute the standard errors of the estimators based on propensity score stratification? Some researcher conditioned on the discretized propensity scores ˆe′(X) and reported standard errors based on the SRE. This effectively ignored the uncertainty in the estimated propensity scores. Other researchers bootstrapped the whole procedure to account for full uncertainty. However, the theory for the bootstrap is still unclear due to the discreteness of this estimator.

## 11.1.3 Application

To illustrate the propensity score stratification method, I revisited Example 10.3. Figure 11.1 shows the histograms of the estimated propensity scores with different numbers of bins (K = 5, 10, 30).

Based on propensity score stratification, we can calculate the point estimators and the standard errors for difference choice of $K \in \{ 5 , 1 0 , 2 0 , 5 0 , 8 0 \}$ as follows (with the function NeymanSRE defined in Chapter 5 for analyzing the SRE):

```txt
> pscore = glm(z ~ x, family = binomial)$fitted.values
> n.strata = c(5, 10, 20, 50, 80)
> strat.res = sapply(n.strata, FUN = function(nn){
+    q.pscore = quantile(pscore, (1:(nn-1))/nn)
+    ps.strata = cut(pscore, breaks = c(0,q.pscore,1),
+    labels = 1:nn)
+    Neyman_SRE(z, y, ps.strata))
>
> rownames(strat.res) = c("est", "se")
> colnames(strat.res) = n.strata
> round(strat.res, 3)
5    10    20    50    80
```

$$
\begin{array}{c c c c c c} \text {est} & - 0. 1 1 6 & - 0. 1 7 8 & - 0. 2 0 0 & - 0. 2 6 5 & - 0. 2 0 4 \\ \text {se} & 0. 2 8 3 & 0. 2 8 2 & 0. 2 7 9 & 0. 2 7 2 & \text {NA} \end{array}
$$

Increasing K from 5 to 50 reduces the standard error. However, we cannot go as extreme as K = 80 because the standard error is not well-defined in some strata with only one treated or control unit. The above estimators show negative but insignificant effect of the meal program on the BMI.

We can also compare the above estimator with the three simple regression estimators: the one without adjusting for any covariates and Fisher and Lin’s estimators.

$$
\begin{array}{c c c c} & \text {naive} & \text {fisher} & \text {lin} \\ \text {est} & 0. 5 3 4 & 0. 0 6 1 & - 0. 0 1 7 \\ \text {se} & 0. 2 2 5 & 0. 2 2 7 & 0. 2 2 6 \end{array}
$$

The naive difference in means differ greatly from other methods. Although the point estimates are different, two regression estimators and the propensity score stratification estimators give qualitatively the same results. The propensity score stratification estimators are stable across different choices of K.

## 11.2 Propensity score weighting

## 11.2.1 Theory

Theorem 11.2 $I f Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$ and $0 < e ( X ) < 1$ , then

$$
E \{Y (1) \} = E \left\{\frac {Z Y}{e (X)} \right\}, \quad E \{Y (0) \} = E \left\{\frac {(1 - Z) Y}{1 - e (X)} \right\},
$$

and

$$
\tau = E \{Y (1) - Y (0) \} = E \left\{\frac {Z Y}{e (X)} - \frac {(1 - Z) Y}{1 - e (X)} \right\}.
$$

Before proving Theorem 11.2, it is important to note the additional assumption $0 < e ( X ) < 1$ . It is called the overlap or positivity condition. The formulas in Theorem 11.2 become infinity if $e ( X ) = 0 { \mathrm { ~ o r ~ } } 1$ for some values of X. It is not a restriction due to the identification formulas based on propensity score weighting. Although it was not stated explicitly in Theorem 10.1, the conditional expectations $E ( Y \mid Z = 1 , X )$ and $E ( Y \mid Z = 0 , X )$ in the identification formula of $\tau$ in (10.5) is well defined only if $0 < e ( X ) < 1$ . The overlap condition can be viewed as a technical condition to ensure that the formulas in Theorems 10.1 and 11.2 are well defined. It can also cause some philosophical issues for causal inference with observational studies. When unit i has $e ( X _ { i } ) = 1$ , we always observe its potential outcome under the treatment, $Y _ { i } ( 1 )$ , but can never observe its potential outcome under the control, $Y _ { i } ( 0 )$ . In this case, the potential outcome $Y _ { i } ( 0 )$ may not even be well defined, making the definition of the causal effect ambiguous for unit i. King and Zeng (2006) called $Y _ { i } ( 0 )$ an extreme counterfactual when $e ( X _ { i } ) = 1$ , and discussed their dangers in causal inference. A similar problem arises if unit i has $e ( X _ { i } ) = 0$ .

In sum, $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} | X$ requires adequate covariates to ensure the conditional independence of the treatment and potential outcomes, and $0 < e ( X ) < 1$ requires residual randomness in the treatment conditional on the covariates. In fact, Rosenbaum and Rubin (1983b)’s definition of strong ignorability includes both of these conditions. In the modern literature, they are often stated separately.

Proof of Theorem 11.2: I only prove the result for $E \{ Y ( 1 ) \}$ because theproof of the result for $E \{ Y ( 0 ) \}$ is similar. We have

$$
\begin{array}{l} E \left\{\frac {Z Y}{e (X)} \right\} \\ = E \left\{\frac {Z Y (1)}{e (X)} \right\} \\ = E \left[ E \left\{\frac {Z Y (1)}{e (X)} \mid X \right\} \right] \quad (\text { tower   property }) \\ = E \left[ \frac {1}{e (X)} E \{Z Y (1) \mid X \} \right] \\ = E \left[ \frac {1}{e (X)} E (Z \mid X) E \{Y (1) \mid X \} \right] \quad (\text { strong   ignorability }) \\ = E \left[ \frac {1}{e (X)} e (X) E \{Y (1) \mid X \} \right] \\ = E [ E \{Y (1) \mid X \} ] \\ = E \{Y (1) \}. \\ \end{array}
$$

## 11.2.2 Inverse propensity score weighting estimators

Theorem 11.2 implies the following moment estimator for the average causal effect:

$$
\hat {\tau} ^ {\mathrm{ht}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {Z _ {i} Y _ {i}}{\hat {e} (X _ {i})} - \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})},
$$

where $\hat { e } ( X _ { i } )$ is the estimated propensity score. This is the inverse propensity score weighting (IPW) estimator, which is also called the Horvitz–Thompson (HT) estimator. Horvitz and Thompson (1952) proposed it in survey sampling and Rosenbaum (1987a) used in causal inference with observational studies.

However, the estimator $\hat { \tau } ^ { \mathrm { h t } }$ has many problems. In particular, it is not invariant to location transformation of the outcome. For example, if we change $Y _ { i }$ to $Y _ { i } + c$ with a constant $c ,$ then it becomes $\hat { \tau } ^ { \mathrm { h t } } + c ( \hat { 1 } _ { \mathrm { T } } - \hat { 1 } _ { \mathrm { C } } )$ , where

$$
\hat {1} _ {\mathrm{T}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {Z _ {i}}{\hat {e} (X _ {i})}, \quad \hat {1} _ {\mathrm{C}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i})}{1 - \hat {e} (X _ {i})}
$$

are two different estimates of the constant 1. I use the funny notation $\hat { 1 } _ { \mathrm { T } }$ and $\mathrm { \hat { 1 } _ { C } }$ because with the true propensity score these two terms both have expectation 1; see Problem 11.3. In general, $\mathrm { \hat { 1 } _ { T } - \hat { 1 } _ { C } }$ is not zero in finite sample. Since adding a constant to every outcome should not change the average causal effect, this estimator is not reasonable because of its dependence on c. A simple fix to the problem is to normalize the weights by $\hat { 1 } _ { \mathrm { T } }$ and ˆ1C respectively, resulting in the following estimator

$$
\hat {\tau} ^ {\mathrm{hajek}} = \frac {\sum_ {i = 1} ^ {n} \frac {Z _ {i} Y _ {i}}{\hat {e} (X _ {i})}}{\sum_ {i = 1} ^ {n} \frac {Z _ {i}}{\hat {e} (X _ {i})}} - \frac {\sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})}}{\sum_ {i = 1} ^ {n} \frac {1 - Z _ {i}}{1 - \hat {e} (X _ {i})}}.
$$

This is the Hajek estimator due to H´ajek (1971). We can verify that the Hajek estimator is invariant to the location transformation, that is, if we replace $Y _ { i }$ by $Y _ { i } + c ,$ then $\hat { \tau } ^ { \mathrm { h a j e k } }$ remains the same. Moreover, many numerical studies have found that $\hat { \tau } ^ { \mathrm { h a j e k } }$ is much more stable than ${ \hat { \tau } } ^ { \mathrm { h t } }$ in finite samples.

## 11.2.3 A problem of weighting and a fundamental problem of causal inference

In many asymptotic analysis, we require a strong overlap condition

$$
0 <   \alpha_ {\mathrm{L}} \leq e (X) \leq \alpha_ {\mathrm{U}} <   1,
$$

that is, the true propensity score is bounded away from 0 and 1. However, D’Amour et al. (2021) pointed out that this is a rather strong assumption especially with many covariates. Chapter 20 will discuss this problem in detail.

Even if the strong overlap condition holds for the true propensity score, the estimated propensity scores can be close to 0 or 1. When this happens, the weighting estimators blow up to infinity resulting in extremely unstable behaviors in finite samples. We can either truncate the estimated propensity score by changing it to

$$
\max \left[ \alpha_ {\mathrm{L}}, \min \{\hat {e} (X _ {i}), \alpha_ {\mathrm{U}} \} \right],
$$

or trim the observations by dropping units with $\hat { e } ( X _ { i } )$ outside the interval $[ \alpha _ { \mathrm { L } } , \alpha _ { \mathrm { U } } ]$ . Crump et al. (2009) suggested $\alpha _ { \mathrm { L } } = 0 . 1$ and $\alpha _ { \mathrm { U } } = 0 . 9$ , and Kurth et al. (2005) suggested $\alpha _ { \mathrm { L } } ~ = ~ 0 . 0 5$ and $\alpha _ { \mathrm { U } } ~ = ~ 0 . 9 5$ . Yang and Ding (2018) established some asymptotic theory for trimming.

## 11.2.4 Application

Revisiting Example 10.3, we can obtain the weighting estimators based on different truncations of the the estimated propensity scores. The following results are the two weighting estimators with the bootstrap standard errors, with truncations at (0, 1), (0.01, 0.99), (0.05, 0.95), and (0.1, 0.9):

\$ trunc0

$$
\begin{array}{c c c} & \text {HT} & \text {Hajek} \\ \text {est} & - 1. 5 1 6 & - 0. 1 5 6 \\ \text {se} & 0. 4 9 5 & 0. 2 3 8 \end{array}
$$

## 11.3 The propensity score as a balancing score

<table><tr><td colspan="3">$trunc.01</td></tr><tr><td></td><td>HT</td><td>Hajek</td></tr><tr><td>est</td><td>-1.516</td><td>-0.156</td></tr><tr><td>se</td><td>0.464</td><td>0.231</td></tr></table>

<table><tr><td colspan="3">$trunc.05</td></tr><tr><td></td><td>HT</td><td>Hajek</td></tr><tr><td>est</td><td>-1.499</td><td>-0.152</td></tr><tr><td>se</td><td>0.472</td><td>0.248</td></tr></table>

<table><tr><td colspan="3">$trunc.1</td></tr><tr><td></td><td>HT</td><td>Hajek</td></tr><tr><td>est</td><td>-0.713</td><td>-0.054</td></tr><tr><td>se</td><td>0.435</td><td>0.229</td></tr></table>

The HT estimator gives results far away from all other estimators we discussed so far. The point estimates seem too large and they are negatively significant unless we truncate the estimated propensity scores at (0.1, 0.9). This is an example showing the instability of the HT estimator.

## 11.3 The propensity score as a balancing score

## 11.3.1 Theory

Theorem 11.3 The propensity score satisfies

$$
Z \bot X \mid e (X).
$$

Moreover, for any function h(·), we have

$$
E \left\{\frac {Z h (X)}{e (X)} \right\} = E \left\{\frac {(1 - Z) h (X)}{1 - e (X)} \right\} \tag {11.2}
$$

provided the existence of the moments on both sides of (11.2).

Rosenbaum and Rubin (1983b) also introduced the notion of balancing score b(X), which satisfies Z X | b(X). By Theorem 11.3, the propensity score is a balancing score. Theorem 11.3 also states that the any function h(X) of the covariates has the same mean across the treatment and control groups, if weighted by the inverse of the propensity score.

Moreover, Rosenbaum and Rubin (1983b) showed that the propensity score $e ( X )$ is the coarsest balancing score, that is, the propensity score $e ( X )$ is a function of any balancing score. Problem 11.5 gives more details.

Proof of Theorem 11.3: First, we show $Z \bot \bot X \mid e ( X )$ , that is,

$$
\operatorname{pr} \{Z = 1 \mid X, e (X) \} = \operatorname{pr} \{Z = 1 \mid e (X) \}. \tag {11.3}
$$

Following similar steps as the proof of Theorem 11.1, we can show that the left-hand side of (11.3) equals

$$
\operatorname{pr} \{Z = 1 \mid X, e (X) \} = \operatorname{pr} (Z = 1 \mid X) = e (X),
$$

and the right-hand side of (11.3) equals

$$
\begin{array}{l} \operatorname{pr} \{Z = 1 \mid e (X) \} = E \{Z \mid e (X) \} \\ = E \left[ E \{Z \mid X, e (X) \} \mid e (X) \right] \\ = E \left[ E \{Z \mid X \} \mid e (X) \right] \\ = E \left[ e (X) \mid e (X) \right] \\ = e (X). \\ \end{array}
$$

Therefore, (11.3) holds.

Second, we show (11.2). We can use similar steps as the proof of Theorem 11.1. But given Theorem 11.1, we have a simpler proof. If we view $h ( X )$ as an outcome, then its two potential outcomes are identical and the strong ignorability holds: Z h(X) | X. The difference between the the left-hand and right-hand sides of (11.2) is the average causal effect of Z on $h ( X )$ , which is zero. □

## 11.3.2 Covariate balance check

The proof of Theorem 11.3 is simple. But Theorem 11.3 has useful implications for the statistical analysis. Before getting access to the outcome data, we can check whether the propensity score model is specified well enough to ensure covariate balance in the data. Rubin (2007) viewed this as the design stage of the observational study, and Rubin (2008) argued that this can result in more objective causal inference because the design stage does not involve the values of the outcomes. While this is a useful recommendation in practice, it is not entirely clear how to quantify the objectiveness.

In propensity score stratification, we have the discretized estimated propensity score $\hat { e } ^ { \prime } ( X )$ and approximately

$$
Z \bot X \mid \hat {e} ^ {\prime} (X) = e _ {k} \quad (k = 1, \ldots , K).
$$

Therefore, we can check whether the covariate distributions are the same across the treatment and control groups within each stratum of the discretized estimated propensity score.

In propensity score weighting, we can view $h ( X )$ as a pseudo outcome and estimate the average causal effect on $h ( X )$ . Because the true average causal effect on $h ( X )$ is 0, the estimate should not be significantly different from 0. A canonical choice of $h ( X )$ is X .

Let us revisit Example 10.3 again. Based on propensity score stratification with $K = 5$ , all the covariates except FoodStamp are well balanced across the treatment and control groups. Similar result holds for the Hajek estimator. Figure 11.2 shows the balance checking results.

## 11.4 Homework Problems

## 11.1 Another version of Theorem 11.1

Prove that

$$
Z \bot \{Y (1), Y (0), X \} \mid e (X, Y (1), Y (0)).
$$

Remark: This result implies that

$$
Z \bot \{Y (1), Y (0) \} \mid \{X, e (X, Y (1), Y (0) \}.
$$

Rosenbaum (2020) and Rosenbaum and Rubin (2023) pointed out this result and called $e ( X , Y ( 1 ) , Y ( 0 ) )$ the principal unobserved covariate.

## 11.2 Another version of Theorem 11.1

If $Z \bot Y ( z ) \mid X$ for $z = 0 , 1$ , then $Z \underline { { | | Y ( z ) | } } \mid e ( X )$ for $z = 0 , 1$ . That is, if ignorability holds conditional on covariates X, then it also holds conditional on the scalar propensity score $e ( X )$ . Prove this theorem.

## 11.3 More results on the IPW estimators

This is related to the discussion of the IPW estimators in Section 11.2.2.

Prove

$$
E \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \frac {Z _ {i}}{e (X _ {i})} \right\} = 1, \quad E \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i})}{1 - e (X _ {i})} \right\} = 1.
$$

## 11.4 Re-analysis of Rosenbaum and Rubin (1983a)

Use Table 1 of Rosenbaum and Rubin (1983a). If you are interested, you can read the whole paper. It is a canonical paper. But for this problem, you only need Table 1.

Rosenbaum and Rubin (1983a) fitted a logistic regression model for the propensity score and stratified the data into 5 subclasses. Because the treatment (Surgical versus Medical) is binary and the outcome is also binary (improved or not), they represented the data by a table.

Based on this table, estimate the average causal effect, and report the 95% confidence interval.

## 11.5 Balancing score and propensity score: more theoretical results

Rosenbaum and Rubin (1983b) defined $b ( X )$ as a balancing score if $Z \bot \bot X \ |$ b(X). Here, b(X) can be a scalar or a vector. An obvious balancing score is $b ( X ) = X$ , but it is not a useful one without any simplification of the original covariates. By Theorem 11.3, the propensity score is a special balancing score. More interestingly, Rosenbaum and Rubin (1983b) showed that the propensity score is the coarsest balancing score, as in Theorem 11.4 below which includes Theorem 11.3 as a special case.

Theorem 11.4 $b ( X )$ is a balancing score if and only if b(X) is finer than $e ( X )$ in the sense that $e ( X ) = f ( b ( X ) )$ for some function f (·).

Theorem 11.4 is relevant in subgroup analysis. In particular, we may be interested in not only the average causal effect τ but also the subgroup effects for boys and girls. Without loss of generality, assume the first component of X is the indicator for girls, and we can interested in estimating

$$
\tau (x _ {1}) = E \{Y (1) - Y (0) \mid X _ {1} = x _ {1} \}, \quad (x _ {1} = 1, 0).
$$

Theorem 11.4 implies that under ignorability,

$$
Z \bot \{Y (1), Y (0) \} \mid e (X), X _ {1} \tag {11.4}
$$

because $b ( X ) = \{ e ( X ) , X _ { 1 } \}$ is finer than $e ( X )$ and thus a balancing score. The conditional independence in (11.4) ensures ignorability holds given the propensity score, within each level of $X _ { 1 }$ . Therefore, we can perform the same analysis based on the propensity score, within each level of $X _ { 1 }$ , yielding estimates for two subgroup effects.

With the above motivation in mind, now prove Theorem 11.4.

## 11.6 Some basics of subgroup effects

This problem is related to Problem 11.5, but you can work on it independently.

Consider a standard observational study with covariates $\boldsymbol { X } = ( X _ { 1 } , X _ { 2 } )$ , where $X _ { 1 }$ denotes a binary subgroup indicator $( \mathrm { e . g . }$ , statistics major or not statistics major) and $X _ { 2 }$ contains the rest covariates. The parameter of interest is the subgroup causal effect

$$
\tau (x _ {1}) = E \{Y (1) - Y (0) \mid X _ {1} = x _ {1} \}, \quad (x _ {1} = 1, 0).
$$

Show that

$$
\tau (x _ {1}) = E \left\{\frac {1 (X _ {1} = x _ {1}) Z Y}{e (X)} - \frac {1 (X _ {1} = x _ {1}) (1 - Z) Y}{1 - e (X)} \right\} / \operatorname{pr} (X _ {1} = x _ {1})
$$

and give the corresponding Horvitz–Thompson and Hajek estimators for $\tau ( x _ { 1 } )$ ).

## 11.7 Recommended reading

The title of this chapter is the same as the title of the classic paper by Rosenbaum and Rubin (1983b). Most results in this chapter are directly drawn from their original paper.

Rubin (2007) and Rubin (2008) highlighted the importance of the design stage of observational studies for more objective causal inference

## 12
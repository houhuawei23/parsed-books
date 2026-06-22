# Observational Studies, Selection Bias, and Nonparametric Identification of Causal Effects

Cochran (1965) summarized two common characteristics of observational studies:

1. the objective is to elucidate cause-and-effect relationships;  
2. it is not feasible to use controlled experimentation.

The first characteristic is identical to that of randomized experiments discussed in Part II, but the second differs fundamentally from randomized experiments.

Dorn (1953) suggested that the planner of an observational study should always ask himself the following question:

How would the study be conducted if it were possible to do it by controlled experimentation?

It is always helpful to follow Dorn (1953)’s suggestion because the potential outcomes framework has an intrinsic link to an experiment, either a real experiment or a thought experiment. Part III of this book will discuss causal inference with observational studies. It will clarify the fundamental differences between observational studies and randomized experiments. Nevertheless, many ideas of causal inference with observational studies are deeply connected to those with randomized experiments.

## 10.1 Motivating Examples

Example 10.1 (job training program) LaLonde (1986) was interested in the causal effect of a job training program on earnings. His compared the results based on a randomized experiment to the results based on observational studies. We have used the experimental data before, which is the lalonde dataset in the Matching package; we have also used an observational counterpart cps1re74.csv in Problem 1.3. LaLonde (1986) found that many traditional econometric methods for observational studies gave quite different estimates compared to the estimates based on the experimental data. Dehejia and Wahba (1999) re-analyzed the data using methods motivated by causal inference, and found that those methods can recover the experimental gold standard. Since then, this became a canonical example in causal inference with observational studies.

Example 10.2 (smoking and homocysteine) Bazzano et al. (2003) compared the homocysteine levels in daily smokers and never smokers based on the data from the National Health and Nutrition Examination Survey (NHANES) 2005–2006. Rosenbaum (2018) documented the data as homocyst in the package senstrat. The dataset has the following important covariates:

female 1=female, 0=male
age3 three age categories: 20–39, 40–50, ≥60
ed3 three education categories: < High School, High School, some College
bmi3 three BMI categories: <30, [30,35), ≥ 35
pov2 TRUE=income at least twice the poverty level, FALSE otherwise

Example 10.3 (school meal program and body mass index) Chan et al. (2016) used a subsample of the data from NHANES 2007–2008 to study whether participation in school meal programs lead to an increase in BMI for school children. They documented the data as nhanesbmi in the package ATE. The dataset has the following important covariates:

age age
ChildSex gender (1: Male, 0: Female)
black race (1: Black, 0: otherwise)
mexam race (1: Hispanic: 0 otherwise)
pir200_plus Family above 200% of the federal poverty level
WIC Participation in the special supplemental nutrition program
Food_Stamp Participation in food stamp program
fsdchbi Childhood food security
AnyIns Any insurance
RefSex Gender of the adult respondent (1: Male, 0: Female)
RefAge Age of the adult respondent

## 10.2 Causal effects and selection bias under the potential outcomes framework

For unit $i ( i = 1 , \ldots , n )$ , we have pretreatment covariates $X _ { i } ,$ a binary treatment indicator $Z _ { i } .$ , and an observed outcome $Y _ { i }$ with two potential outcomes $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ under treatment and control, respectively. For simplicity, we assume

$$
\{X _ {i}, Z _ {i}, Y _ {i} (1), Y _ {i} (0) \} _ {i = 1} ^ {n} \stackrel {{\text {IID}}} {{\sim}} \{X, Z, Y (1), Y (0) \}.
$$

## 10.2 Causal effects and selection bias under the potential outcomes framework 129

So we can drop the subscript i for quantities depending on this population. The causal effects of interest are the average causal effect

$$
\tau = E \{Y (1) - Y (0) \},
$$

the average causal effect on the treated units

$$
\tau_ {\mathrm{T}} = E \{Y (1) - Y (0) \mid Z = 1 \},
$$

and the average causal effect on the control units:

$$
\tau_ {\mathrm{C}} = E \{Y (1) - Y (0) \mid Z = 0 \}.
$$

By the linearity of the expectation, we have

$$
\begin{array}{l} \tau_ {\mathrm{T}} = E \{Y (1) \mid Z = 1 \} - E \{Y (0) \mid Z = 1 \} \\ = E (Y \mid Z = 1) - E \{Y (0) \mid Z = 1 \} \\ \end{array}
$$

and

$$
\begin{array}{l} \tau_ {\mathrm{C}} = E \{Y (1) \mid Z = 0 \} - E \{Y (0) \mid Z = 0 \} \\ = E \{Y (1) \mid Z = 0 \} - E (Y \mid Z = 0). \\ \end{array}
$$

In the above two formulas of $\tau _ { \mathrm { T } }$ and $\tau _ { \mathrm { C } }$ , the quantities $E ( Y \mid Z = 1 )$ and $E ( Y \mid$ $Z = 0 )$ are directly observable from the data, but the quantities $E \{ Y ( 0 ) \mid Z =$ $1 \}$ and $E \{ Y ( 1 ) \mid Z = 0 \}$ are not. The latter two are counterfactuals because they are the means of the potential outcomes corresponding to the treatment level that is the opposite of the actual received treatment.

The simple difference in means, also known as the prima facie causal effect,

$$
\begin{array}{l} \tau_ {\mathrm{PF}} = E (Y \mid Z = 1) - E (Y \mid Z = 0) \\ = E \{Y (1) \mid Z = 1 \} - E \{Y (0) \mid Z = 0 \} \\ \end{array}
$$

is generally biased for the causal effects defined above. For example,

$$
\tau_ {\mathrm{PF}} - \tau_ {\mathrm{T}} = E \{Y (0) \mid Z = 1 \} - E \{Y (0) \mid Z = 0 \}
$$

and

$$
\tau_ {\mathrm{PF}} - \tau_ {\mathrm{C}} = E \{Y (1) \mid Z = 1 \} - E \{Y (1) \mid Z = 0 \}
$$

are not zero in general, and they quantifies the selection bias. They measure the differences in the means of the potential outcomes across the treatment and control groups.

Why randomization is so important? Rubin (1978) first used potential outcomes to quantify the benefit of randomization. We have used the fact in Chapter 9 that

$$
Z \bot \{Y (1), Y (0) \} \tag {10.1}
$$

in the CRE, which implies that the selection bias terms are both zero:

$$
\tau_ {\mathrm{PF}} - \tau_ {\mathrm{T}} = E \{Y (0) \mid Z = 1 \} - E \{Y (0) \mid Z = 0 \} = 0
$$

and

$$
\tau_ {\mathrm{PF}} - \tau_ {\mathrm{C}} = E \{Y (1) \mid Z = 1 \} - E \{Y (1) \mid Z = 0 \} = 0.
$$

So under complete randomization (10.1),

$$
\tau = \tau_ {\mathrm{T}} = \tau_ {\mathrm{C}} = \tau_ {\mathrm{PF}}.
$$

From the above discussion, the fundamental benefit of randomization is to balance the distributions of the potential outcomes across the treatment and control groups, which is more important than to balance the distributions of the observed covariates.

Without randomization, the selection bias terms can be arbitrarily large especially for unbounded outcomes. This highlights the fundamental difficulty of causal inference with observational studies.

## 10.3 Sufficient conditions for nonparametric identification

## 10.3.1 Identification

Causal inference with observational studies is challenging. It relies on strong assumptions. A strategy is to use the information of the pretreatment covariates and assume that conditioning on the observed covariates X, the selection bias terms are zero, that is,

$$
E \{Y (0) \mid Z = 1, X \} = E \{Y (0) \mid Z = 0, X \}, \tag {10.2}
$$

$$
E \{Y (1) \mid Z = 1, X \} = E \{Y (1) \mid Z = 0, X \}. \tag {10.3}
$$

The assumptions in (10.2) and (10.3) state that the differences in the means of the potential outcomes across the treatment and control groups are entirely due to the difference in the observed covariates. So given the same value of the covariates, the potential outcomes have the same means across the treatment and control groups. Mathematically, (10.2) and (10.3) ensure that the conditional versions of the effects are identical:

$$
\tau (X) = \tau_ {\mathrm{T}} (X) = \tau_ {\mathrm{C}} (X) = \tau_ {\mathrm{PF}} (X),
$$

where

$$
\begin{array}{l} \tau (X) = E \{Y (1) - Y (0) \mid X \}, \\ \tau_ {\mathrm{T}} (X) = E \{Y (1) - Y (0) \mid Z = 1, X \}, \\ \tau_ {\mathrm{C}} (X) = E \{Y (1) - Y (0) \mid Z = 0, X \}, \\ \tau_ {\mathrm{PF}} (X) = E (Y \mid Z = 1, X) - E (Y \mid Z = 0, X). \\ \end{array}
$$

In particular, $\tau ( X )$ is often called the conditional average causal $e f f e c t .$

A key result in this chapter is that the average causal effect τ is nonparametrically identifiable under (10.2) and (10.3). The notion of nonparametrically identifiability does not appear frequently in classic statistics, but it is key to causal inference with observational studies.

Definition 10.1 (identification) A parameter θ is identifiable if it can be written as a function of the distribution of the observed data under certain model assumptions. A parameter θ is nonparametrically identifiable $i f$ it can be written as a function of the distribution of the observed data without any parametric model assumptions.

Definition 10.1 is too abstract at the moment. I will use more concrete examples in later chapters to illustrate its meaning. It is often neglected in standard statistics problems. For instance, the mean $\theta = E ( Y )$ is nonparametrically identifiable if we have IID draws of $Y _ { i } ^ { \mathrm { : } } \mathrm { s } ;$ the Pearson correlation coefficient $\theta \ : = \ : \mathrm { c o r r } ( X , Y )$ is nonparametrically identifiable if we have IID draws of the pairs $( X _ { i } , Y _ { i } ) \mathrm { { ^ { \circ } s } }$ . In those examples, the parameters are nonparametrically identifiable automatically. However, Definition 10.1 is fundamental in causal inference with observational studies. In particular, the parameter of interest $\tau = E \{ Y ( 1 ) - Y ( 0 ) \}$ depends on some unobserved random variables, so it is unclear whether it is nonparametrically identifiable based on observed data. Under the assumptions in (10.2) and (10.3), it is nonparametrically identifiable, with detailed below.

Because $\tau _ { \mathrm { P F } } ( X )$ depends only on the observables, it is nonparametrically identified by definition. Moreover, (10.2) and (10.3) ensure that the three causal effects are the same as $\tau _ { \mathrm { P F } } ( X )$ , so $\tau ( X ) , \ \tau _ { \mathrm { T } } ( X )$ and $\tau _ { \mathrm { C } } ( X )$ are all nonparametrically identified. Consequently, the unconditional versions are also nonparametrically identified under (10.2) and (10.3) due to the law of total expectation:

$$
\tau = E \{\tau (X) \}, \quad \tau_ {\mathrm{T}} = E \{\tau_ {\mathrm{T}} (X) | Z = 1 \}, \quad \tau_ {\mathrm{C}} = E \{\tau_ {\mathrm{C}} (X) | Z = 0 \}.
$$

From now on, we focus on τ unless stated otherwise. The following theorem summarized the identification formulas of τ .

Theorem 10.1 Under (10.2) and (10.3), the average causal effect τ is identified by

$$
\tau = E \{\tau (X) \} \tag {10.4}
$$

$$
= E \{E (Y \mid Z = 1, X) - E (Y \mid Z = 0, X) \} \tag {10.5}
$$

$$
= \int \{E (Y \mid Z = 1, X = x) - E (Y \mid Z = 0, X = x) \} F (\mathrm{d} x). \tag {10.6}
$$

The formula (10.5) was formally established by Rosenbaum and Rubin (1983b), which is also called the g-formula by Robins (see Hern´an and Robins, 2020).

With a discrete covariate, we can write the identification formula in Theorem 10.1 as

$$
\begin{array}{l} \tau = \sum_ {x} E (Y \mid Z = 1, X = x) \mathrm{pr} (X = x) \\ - \sum_ {x} E (Y \mid Z = 0, X = x) \mathrm{pr} (X = x), \tag {10.7} \\ \end{array}
$$

and also the simple difference in means as

$$
\begin{array}{l} \tau_ {\mathrm{PF}} = \sum_ {x} E (Y \mid Z = 1, X = x) \mathrm{pr} (X = x \mid Z = 1) \\ - \sum_ {x} E (Y \mid Z = 0, X = x) \mathrm{pr} (X = x \mid Z = 0) \tag {10.8} \\ \end{array}
$$

by the law of total probability. Comparing (10.7) and (10.8), we can see that although both formulas compare the conditional expectations $E ( Y \mid Z =$ $1 , X = x )$ and $E ( Y \mid Z = 0 , X = x )$ , they average over different distribution of the covariates. The causal parameter τ averages the conditional expectations over the common distribution of the covariate, but the difference in means τPF averages the conditional expectations over two different distributions of covariate in the treated and control groups.

Usually, we impose a stronger assumption:

$$
Y (z) \perp \perp Z \mid X \quad (z = 0, 1). \tag {10.9}
$$

This assumption has many names:

1. ignorability due to Rubin (1978);  
2. unconfoundedness which is popular among epidemiologists;  
3. selection on observables which is popular among social scientists;  
4. conditional independence which is merely a description of the notation in the assumption.

Sometimes, we impose an even stronger assumption

$$
\{Y (1), Y (0) \} \perp Z \mid X \tag {10.10}
$$

which is called strong ignorability (Rosenbaum and Rubin, 1983b). If the parameter of interest is $\tau ,$ then the stronger assumptions (10.9) and (10.10) are just imposed for notational simplicity. They are not necessary in this case. However, they cannot be relaxed if the parameter of interest is the causal effects on other scales (for example, distribution, quantile, or some transformation of the outcome). The strong ignorability assumption requires that the potential outcomes vector be independent of the treatment given covariates, but the ignorability assumption only requires each potential outcome be independent of the treatment given covariates. The former is stronger than the

<!-- footnote -->

- For example, the linear projection of $Y _ { i } ( 1 )$ on $( 1 , X _ { i } )$ is $\alpha _ { 1 } + \beta _ { 1 } X _ { i }$ where
- $( \alpha _ { 1 } , \beta _ { 1 } ) = \arg \operatorname* { m i n } _ { a , b } \sum _ { i = 1 } ^ { n } \{ Y _ { i } ( 1 ) - a - b ^ { \mathsf { T } } X _ { i } \} ^ { 2 } .$

<!-- footnote end -->

<!-- footnote -->

- Without covariates, the HC2 correction yields identical variance estimator as Neyman (1923)’s classic one. For coherence, we can also use the HC2 correction for Lin (2013)’s estimator with covariate adjustment. When the number of covariates is small compared to the sample size and the covariates do not contain outliers, the variants of the EHW standard error perform similarly to the original one. When the number of covariates is large compared to the sample size or the covariates contain outliers, the variants can outperform

<!-- footnote end -->

<!-- footnote -->

- the original one. In those cases, Lei and Ding (2021) recommend using the HC3 variant of the EHW standard error. See Chapter A2 for more details of the EHW standard errors.

<!-- footnote end -->

<!-- footnote -->

- Butler (1969)’s proposed this test statistic under a slightly different framework. Given IID draws of $\big ( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } \big )$ from a distribution $F ( y )$ , if they are symmetrically distributed around 0, then
- $F ( t ) = \mathrm { p r } ( \hat { \tau } _ { i } \le t ) = \mathrm { p r } ( - \hat { \tau } _ { i } \le t ) = 1 - \mathrm { p r } ( \hat { \tau } _ { i } < - t ) = 1 - F ( - t - ) .$
- Therefore, $\hat { F } ( t ) + \hat { F } ( - t - )$ − 1 measures the deviation from the null hypothesis of symmetry, which motivates the definition of D. A naive definition of the Kolmogorov–Smirnov-type statistic is to compare the empirical distributions of the outcomes under treatment and control as in Example 3.4. Using that definition, we effectively break the pairs. Although it can still be used in the FRT for the MPE, it does not capture the matched-pairs structure of the experiment.

<!-- footnote end -->

<!-- footnote -->

- In causal inference, we say that a parameter is nonparametrically identifiable if it can be determined by the distribution of the observed variables without imposing further parametric assumptions.

<!-- footnote end -->

latter. However, their difference is rather technical and of pure probability interests; see Problem 10.4. In most reasonable statistical models, they are identical; see Section 10.3.2 below. We will not distinguish them in this book and will simply use ignorability to refer to both.

## 10.3.2 Plausibility of the assumption

A fundamental problem of causal inference with observational studies is the plausibility of the ignorability assumption. The above discussion may seem too mathematical in the sense that the ignorability assumption serves as a sufficient condition to ensure the nonparametric identification of the average causal effect. What is its scientific meaning? Intuitively, it rules out all unmeasured covariates that affect treatment and outcome simultaneously. Those “common causes” of the treatment and outcomes are called confounders. That is why the ignorability assumption is also called the unconfoundedness assumption. More mathematically, we can interpret the ignorability assumption based on the outcome data generating process. If

$$
\begin{array}{l} Y (1) = f _ {1} (X, V _ {1}), \\ Y (0) = f _ {0} (X, V _ {0}), \\ Z = 1 \{g (X, V) \geq 0 \} \\ \end{array}
$$

with $( V _ { 1 } , V _ { 0 } ) \bot \bot V$ , then (10.9) and (10.10) hold. In the above data generating process, the “common causes” X of the treatment and the outcome are all observed, the remaining random components are independent. If the data generating process changes to

$$
\begin{array}{l} Y (1) = f _ {1} (X, U, V _ {1}), \\ Y (0) = f _ {0} (X, U, V _ {0}), \\ Z = 1 \{g (X, U, V) \geq 0 \} \\ \end{array}
$$

with $( V _ { 1 } , V _ { 0 } ) \bot \bot V$ , then (10.9) or (10.10) does not hold in general. The unmeasured “common cause” U induces dependence between the treatment and potential outcomes even conditioning on the observed covariates X. If we do not have access to U and analyze the data based only on $( Z , X , Y )$ , the final estimator will be biased for the causal parameter in general. This type of bias is called the omitted variable bias in econometrics.

The ignorability assumption can be reasonable if we observe a rich set of covariates X that affect the treatment and the outcome simultaneously. I start with this assumption, discussing identification and estimation strategies in Part III of this book. However, it is fundamentally untestable. We may justify it based on the scientific background knowledge, but we are often not sure whether it holds or not. Parts IV and V of this book will discuss other strategies when this assumption is not plausible.

## 10.4 Two simple estimation strategies and their limitations

## 10.4.1 Stratification or standardization based on discrete covariates

If the covariate $X _ { i } \in \{ 1 , \ldots , K \}$ is discrete, then ignorability (10.9) reads as

$$
Y (z) \bot Z \mid X = k \quad (z = 0, 1; k = 1, \dots , K),
$$

which essentially assumes that the observational study is a SRE. Therefore, we can use the estimator

$$
\hat {\tau} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \left\{\hat {\bar {Y}} _ {[ k ]} (1) - \hat {\bar {Y}} _ {[ k ]} (0) \right\},
$$

which is identical to the stratified or post-stratified estimator discussed in Chapter 5.

This method is still widely used in practice. Example 10.2 contains discrete covariates, and I relegate the analysis to Problem 10.3. However, there are several obvious difficulties in implementing this method. First, it works well for the case with small K. For large K, it is very likely that many strata have $n _ { [ k ] 1 } = 0 \mathrm { o r } n _ { [ k ] 0 } = 0$ , leading to the illy defined ${ \hat { \tau } } _ { [ k ] } { } ^ { \ \mathrm { { * } } }$ for those strata. This is related to the issue of overlap which will be discussed in Chapter 20. Second, it is not obvious how to apply this stratification method to multidimensional continuous or mixed covariates X. A standard method is to create strata based on the initial covariates and then apply the stratification method. This may result in arbitrariness in the analysis.

## 10.4.2 Outcome regression

The most commonly-used method based on the outcome regression is to run the OLS with an additive model of the observed outcome on the treatment indicator and covariates, which assumes

$$
E (Y \mid Z, X) = \beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X.
$$

If the above linear model is correct, then we have

$$
\begin{array}{l} \tau (X) = E (Y \mid Z = 1, X) - E (Y \mid Z = 0, X) \\ = \left(\beta_ {0} + \beta_ {z} + \beta_ {x} ^ {\mathsf {T}} X\right) - \left(\beta_ {0} + \beta_ {x} ^ {\mathsf {T}} X\right) \\ = \beta_ {z}, \\ \end{array}
$$

which implies that the treatment effect is homogeneous with respect to the covariates. This, coupled with ignorability, implies that

$$
\tau = E \{\tau (X) \} = \beta_ {z}.
$$

Therefore, if ignorability holds and the outcome model is linear, then the average causal effect equals the coefficient of Z. This is one of the most important applications of the linear model. However, the causal interpretation of the coefficient of $Z$ is valid only under two strong assumptions: ignorability and the linear model.

We have discussed in Chapter 6, the above procedure is suboptimal even in randomized experiments, because it ignores the treatment effect heterogeneity induced by the covariates. If we assume

$$
E (Y \mid Z, X) = \beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X + \beta_ {z x} ^ {\mathsf {T}} X Z,
$$

we have

$$
\begin{array}{l} \tau (X) = E (Y \mid Z = 1, X) - E (Y \mid Z = 0, X) \\ = \left(\beta_ {0} + \beta_ {z} + \beta_ {x} ^ {\mathsf {T}} X + \beta_ {z x} ^ {\mathsf {T}} X\right) - \left(\beta_ {0} + \beta_ {x} ^ {\mathsf {T}} X\right) \\ { = } { \beta _ { z } + \beta _ { z x } ^ { \mathsf { T } } X , } \\ \end{array}
$$

which, coupled with ignorability, implies that

$$
\tau = E \{\tau (X) \} = E (\beta_ {z} + \beta_ {z x} ^ {\mathsf {T}} X) = \beta_ {z} + \beta_ {z x} ^ {\mathsf {T}} E (X).
$$

The estimator for $\tau$ is then $\hat { \beta } _ { z } + \hat { \beta } _ { z x } ^ { \sf T } \bar { X }$ , where $\hat { \beta } _ { z }$ is the regression coefficient and X¯ is the sample mean of X. If we center the covariates to ensure $\bar { X } = 0 .$ , then the estimator is simply the regression coefficient of $Z .$ To simplify the procedure, we usually center the covariates at the beginning; also recall Lin (2013)’s estimator introduced in Chapter 6. Rosenbaum and Rubin (1983b) and Hirano and Imbens (2001) discussed this estimator.

In general, we can use other more complex models to estimate the causal effects. For example, if we build two predictors $\hat { \mu } _ { 1 } ( X )$ and $\hat { \mu } _ { 0 } ( X )$ based on the treated and control data, respectively, then we have an estimator for the conditional average causal effect

$$
\hat {\tau} (X) = \hat {\mu} _ {1} (X) - \hat {\mu} _ {0} (X)
$$

and an estimator for the average causal effect:

$$
\hat {\tau} = n ^ {- 1} \sum_ {i = 1} ^ {n} \{\hat {\mu} _ {1} (X _ {i}) - \hat {\mu} _ {0} (X _ {i}) \}.
$$

The estimator ˆτ above has the same form as the projective estimator discussed in Chapter 6. It is sometimes called the outcome imputation estimator. For example, we may model a binary outcome using a logistic model

$$
E (Y \mid Z, X) = \mathrm{pr} (Y = 1 \mid Z, X) = \frac {e ^ {\beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X}}{1 + e ^ {\beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X}},
$$

then based on the estimators of the coefficients $\hat { \beta } _ { 0 } , \hat { \beta } _ { z } , \hat { \beta } _ { x }$ , we have the following estimator for the average causal effect:

$$
\hat {\tau} = n ^ {- 1} \sum_ {i = 1} ^ {n} \left\{\frac {e ^ {\hat {\beta} _ {0} + \hat {\beta} _ {z} + \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}}}{1 + e ^ {\hat {\beta} _ {0} + \hat {\beta} _ {z} + \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}}} - \frac {e ^ {\hat {\beta} _ {0} + \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}}}{1 + e ^ {\hat {\beta} _ {0} + \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}}} \right\}.
$$

This estimator is not simply the coefficient of the treatment in the logistic model.1 It is a nonlinear function of all the coefficients as well as the the empirical distribution of the covariates. In econometrics, this estimator is is called the average partial effect or average marginal effect of the treatment in the logistic model. Many econometric software packages can report this estimator associated with the standard error. Similarly, we can also derive the corresponding estimator based on a fully interacted logistic model; see Problem 10.2.

For all the estimators discussed above, we can use the nonparametric bootstrap to estimate the standard errors. See Chapter A1.5.

The above predictors for the conditional means of the outcome can also be other machine learning tools. In particular, Hill (2011) championed the use of tree methods for estimating τ , and Wager and Athey (2018) proposed to use them also for estimating ˆτ (X). Wager and Athey (2018) also combined the tree methods with the ideas in the next chapter. Since then, machine learning and causal inference has been an active research area (e.g., Hahn et al., 2020; K¨unzel et al., 2019).

The biggest problem of the above approach based on outcome regressions is its sensitivity to the specification of the outcome model. Problem 1.3 gave such an example. Depending on the incentive of empirical research and publications, people sometimes reported their favorable causal effects estimates after searching over a wide set of candidate models, without confessing this searching process. This is a major source of p-hacking in causal inference.

## 10.5 Homework Problems

## 10.1 Nonparametric identification of other causal effects

Under ignorability and overlap, show that

1. the distributional causal effect

$$
\mathrm{DCE} _ {y} = \operatorname * {p r} \{Y (1) > y \} - \operatorname * {p r} \{Y (0) > y \}
$$

is nonparametrically identifiable for all $y ;$

2. the quantile causal effect

$$
\mathrm{QCE} _ {q} = \text { quantile } _ {q} \{Y (1) \} - \text { quantile } _ {q} \{Y (0) \},
$$

is nonparametrically identifiable for all $q ,$ where $\mathrm { q u a n t i l e } _ { q } \{ \cdot \}$ is the qth quantile of a random variable.

Remark: In probability theory, pr $\{ Y ( z ) \leq y \}$ is the cumulative distribution function and pr $\{ Y ( z ) > y \}$ is the survival function of the potential outcome $Y ( z )$ . The distributional causal effect compares the survival functions of the potential outcomes under treatment and control.

## 10.2 Outcome imputation estimator in the fully interacted logistic model

Assume that a binary outcome follows a logistic model

$$
E (Y \mid Z, X) = \operatorname{pr} (Y = 1 \mid Z, X) = \frac {e ^ {\beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X + \beta_ {x z} ^ {\mathsf {T}} X Z}}{1 + e ^ {\beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X + \beta_ {x z} ^ {\mathsf {T}} X Z}}.
$$

What is the corresponding outcome regression estimator for the average causal effect?

## 10.3 Data analysis: stratification and regression

Use the dataset homocyst in the package senstrat. The outcome is homocysteine, the homocysteine level, and the treatment is $\mathbf { z } ,$ where $z =$ 1 for a daily smoker and $z ~ = ~ 0$ for a never smoker. Covariates are female, age3, ed3, bmi3, pov2 with detailed explanations in the package, and st is a stratum indicator, defined by all the combinations of the discrete covariates.

1. How many strata have only treated or control units? What is the proportion of the units in these strata? Drop these strata and perform a stratified analysis of the observational study. Report the point estimator, variance estimator and 95% confidence interval for the average causal effect.  
2. Run the OLS of the outcome on the treatment indicator and covariates without interactions. Report the coefficient of the treatment and the robust standard error.  
Drop the strata with only treated or control units. Re-run the OLS and report the result.  
3. Apply Lin (2013)’s estimator of the average causal effect. Report the coefficient of the treatment and the robust standard error.  
If you do not drop the strata with only treated or control units, what will happen?  
4. Compare the results in the above three analyses. Which one is more credible?

## 10.4 Ignorability versus strong ignorability

Given an example such that the ignorability holds but the strong ignorability does not hold.

Remark: This is related to a classic probability problem of finding three random variables A, B, C such that

$$
A \bot C \text {   and   } B \bot C \text {   but   } (A, B) \not \bot C.
$$

## 10.5 Recommended reading

Cochran (1965) is a classic reference on observational studies. It contains many useful insights but does not use the formal potential outcomes framework.
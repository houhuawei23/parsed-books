# Controlled Direct Effect

The formulation of mediation analysis in Chapter 27 relies on the nested potential outcomes, and fundamentally, some nested potential outcomes are not observable in any physical experiments. If we stick to the Popperian philosophy of science, we should only define causal parameters in terms of quantities that are observable under some experiments. This chapter discusses an alternative view of causal inference with an intermediate variable. In this view, we only define the direct effect but can not define the indirect effect.

## 28.1 Identification and estimation of the controlled direct effect

We view Z and M as two factors, and define potential outcomes $Y ( z , m )$ for $z = 0 , 1$ and $m \in { \mathcal { M } } .$ . Based on these potential outcomes, we can define the controlled direct effect (CDE) below.

Definition 28.1 (CDE) Define

$$
\operatorname{CDE} (m) = E \{Y (1, m) - Y (0, m) \}.
$$

By definition, cde(m) is the average causal effect of the treatment if the intermediate variable is fixed at m. The parameter cde(m) can capture the direct effect of the treatment holding the mediator at m. However, this formulation cannot capture the indirect effect. In particular, the parameter $E \{ Y ( z , 1 ) - Y ( z , 0 ) \}$ only measures the effect of the mediator on the outcome holding the treatment at z. This is not a meaningful definition of the indirect effect.

To identify cde(m), we need the following assumption, which basically requires that Z and M are jointly randomized given X.

Assumption 28.1 Sequential ignorability requires

$$
Z \bot Y (z, m) \mid X, \quad M \bot Y (z, m) \mid (Z, X)
$$

$o r ,$ equivalently,

$$
(Z, M) \bot Y (z, m) \mid X.
$$

I will focus on the case with a binary Z and M. Mathematically, we can just view this problem as an observational study with four treatment levels

$$
(z, m) \in \{(0, 0), (0, 1), (1, 0), (1, 1) \}.
$$

The following theorem extends the results for observational studies with a binary treatment, identifying

$$
\mu_ {z m} = E \{Y (z, m) \}
$$

based on outcome regression, inverse probability weighting, and doubly robust estimation.

Define

$$
\mu_ {z m} (x) = E (Y \mid Z = z, M = m, X = x)
$$

as the outcome mean conditional on the treatment, mediator and covariates. Define

$$
e _ {z m} (x) = \operatorname * {p r} (Z = z, M = m \mid X = x) = \operatorname * {p r} (Z = z \mid X = x) \operatorname * {p r} (M = m \mid Z = z, X = x)
$$

as the probability of the joint value of Z and M conditional on the covariates.

Theorem 28.1 Under Assumption 28.1, we have

$$
\mu_ {z m} = E \{\mu_ {z m} (X) \}
$$

or

$$
\mu_ {z m} = E \left\{\frac {I (Z = z , M = m) Y}{e _ {z m} (X)} \right\}.
$$

Moreover, based on the working models $e _ { z m } ( X , \alpha )$ and $\mu _ { z m } ( X , \beta )$ , we have the doubly robust formula

$$
\mu_ {z m} ^ {\mathrm{dr}} = E \{\mu_ {z m} (X, \beta) \} + E \left[ \frac {I (Z = z , M = m) \{Y - \mu_ {z m} (X , \beta) \}}{e _ {z m} (X , \alpha)} \right],
$$

$w h i c h ~ e q u a l s ~ \mu _ { z m } ~ i f ~ e i t h e r ~ e _ { z m } ( X , \alpha ) = e _ { z m } ( X ) ~ o r ~ \mu _ { z m } ( X , \beta ) = \mu _ { z m } ( X ) .$

The proof of Theorem 28.1 is similar to those for the standard unconfounded observational studies. Problem 28.2 gives a general result. Based on the outcome mean model, we can obtain ${ \hat { \mu } } _ { z m } ( x )$ for $\mu _ { z m } ( x )$ . Based on the treatment model, we can obtain $\hat { e } _ { z } ( x )$ for $\operatorname { p r } ( Z = z \mid X = x )$ ; based on the intermediate variable model, we can obtain $\hat { e } _ { m } ( z , x )$ for $\operatorname { p r } ( M = m \mid Z =$ $z , X = x )$ ). We can then estimate $\mu _ { z m }$ by outcome regression

$$
\hat {\mu} _ {z m} ^ {\mathrm{reg}} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\mu} _ {z m} (X _ {i}),
$$

by inverse probability weighting

$$
\begin{array}{l} \hat {\mu} _ {z m} ^ {\mathrm{ht}} = n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {I (Z _ {i} = z , M _ {i} = m) Y _ {i}}{\hat {e} _ {z} (X _ {i}) \hat {e} _ {m} (z , X _ {i})}, \\ \hat {\mu} _ {z m} ^ {\mathrm{haj}} = \sum_ {i = 1} ^ {n} \frac {I (Z _ {i} = z , M _ {i} = m) Y _ {i}}{\hat {e} _ {z} (X _ {i}) \hat {e} _ {m} (z , X _ {i})} \bigg / \sum_ {i = 1} ^ {n} \frac {I (Z _ {i} = z , M _ {i} = m)}{\hat {e} _ {z} (X _ {i}) \hat {e} _ {m} (z , X _ {i})}, \\ \end{array}
$$

or by augmented inverse probability weighting

$$
\hat {\mu} _ {z m} ^ {\mathrm{dr}} = \hat {\mu} _ {z m} ^ {\mathrm{reg}} + n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {I (Z _ {i} = z , M _ {i} = m) \{Y _ {i} - \hat {\mu} _ {z m} (X _ {i}) \}}{\hat {e} _ {z} (X _ {i}) \hat {e} _ {m} (z , X _ {i})}.
$$

We can then estimate $\mathrm { C D E } ( m )$ by $\hat { \mu } _ { 1 m } - \hat { \mu } _ { 0 m }$ and use the bootstrap to approximate the standard error.

If we are willing to assume a linear outcome model, the controlled direct effect simplifies to the coefficient of the treatment. Example 28.1 below gives the details.

Example 28.1 Under Assumption 28.1 and a linear outcome model,

$$
E (Y \mid Z, M, X) = \theta_ {0} + \theta_ {1} Z + \theta_ {2} M + \theta_ {4} ^ {\mathsf {T}} X,
$$

we can show that $\mathrm { C D E } ( m )$ equals the coefficient $\theta _ { 1 }$ , which coincides with the natural direct effect in the Baron–Kenny method. I relegate the proof to Problem 28.3.

## 28.2 Discussion

The formulation of the controlled direct effect does not involve nested or a priori counterfactual potential outcomes, and its identification does not require the cross-world counterfactual independence assumption. The parameter cde(m) can capture the direct effect of the treatment holding the mediator at m. However, this formulation cannot capture the indirect effect. I summarize the causal frameworks for intermediate variables below.

<table><tr><td>chapter</td><td>framework</td><td>direct effect</td><td>indirect effect</td></tr><tr><td>26</td><td>principal stratification</td><td>τ(1,1), τ(0,0)</td><td>?</td></tr><tr><td>27</td><td>mediation analysis</td><td>NDE</td><td>NIE</td></tr><tr><td>29</td><td>controlled direct effect</td><td>CDE(m)</td><td>?</td></tr></table>

The mediation analysis framework can decompose the total effect into natural direct and indirect effects, but it requires nested potential outcomes and cross-world independence. The principal stratification and controlled direct effect frameworks cannot define indirect effects but they do not involve nested potential outcomes and cross-world independence. Moreover, the principal stratification framework does not necessarily require that M lies on the causal pathway from the treatment to the outcome. But its identification and estimation involves disentangling mixture distributions, which is a nontrivial task in statistics.

## 28.3 Homework problems

## 28.1 cde and nde

Show that under cross-world independence $Y ( z , m ) \bot M ( z ^ { \prime } ) \mid X$ for all $z , z ^ { \prime }$ and m, the c ${ \mathrm { o n d i t i o n a l ~ C D E } } ( m \mid x ) = E \{ Y ( 1 , m ) - Y ( 0 , m ) \mid X = x \}$ and $\operatorname { N D E } ( x ) = E \{ Y ( 1 , M _ { 0 } ) - Y ( 0 , M _ { 0 } ) \mid X = x \}$ have the following relationship:

$$
\mathrm{NDE} (x) = E \{\mathrm{CDE} (M _ {0} \mid x) \},
$$

which reduces to

$$
\mathrm{NDE} (x) = \sum_ {m} \mathrm{CDE} (m \mid x) \mathrm{pr} (M _ {0} = m \mid X = x)
$$

for a discrete M. Without the cross-world independence, does this relationship still hold in general?

## 28.2 Observational studies with a multi-valued treatment

Theorem 28.1 is a special case of the following theorem for unconfounded observational studies with multiple treatment levels (Imai and Van Dyk, 2004; Cattaneo, 2010). Below, I state the general problem and theorem.

Consider an observational study with a multi-valued treatment $Z \in \mathbf { \Sigma }$ $\{ 1 , \ldots , K \}$ , covariates X, and outcome Y . Unit i has K potential outcomes $Y _ { i } ( 1 ) , \ldots , Y _ { i } ( K )$ corresponding to the K treatment levels. Causal effects can be defined as comparisons of the potential outcomes. In general, we can define causal effect in terms of contrasts of the potential outcomes:

$$
\tau_ {c} = \sum_ {k = 1} ^ {K} c _ {k} E \{Y (k) \}
$$

where $\textstyle \sum _ { k = 1 } ^ { K } c _ { k } = 0$ . The canonical choice of the pairwise comparison

$$
\tau_ {k, k ^ {\prime}} = E \{Y (k) - Y (k ^ {\prime}) \}.
$$

Therefore, the key is to identify and estimate the means of the potential outcomes $\mu _ { k } = E \{ Y ( k ) \}$ under the ignorability assumption below based on IID data of $( Z _ { i } , X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ .

Assumption 28.2 $Z \bot \bot \{ Y ( 1 ) , \dots , Y ( K ) \} \mid X$ .

Define the generalized propensity score as

$$
e _ {k} (X) = \operatorname{pr} (Z = k \mid X),
$$

and define the conditional outcome mean as

$$
\mu_ {k} (X) = E (Y \mid Z = k, X)
$$

for $k = 1 , \ldots , K$ . We have the following theorem.

Theorem 28.2 Under Assumption 28.2, we have

$$
\mu_ {k} = E \{\mu_ {k} (X) \}
$$

or

$$
\mu_ {k} = E \left\{\frac {I (Z = k) Y}{e _ {k} (X)} \right\}.
$$

Moreover, based on the working models $e _ { k } ( X , \alpha )$ and $\mu _ { k } ( X , { \boldsymbol { \beta } } )$ , we have the doubly robust formula

$$
\mu_ {k} ^ {\mathrm{dr}} = E \{\mu_ {k} (X, \beta) \} + E \left[ \frac {I (Z = k) \{Y - \mu_ {k} (X , \beta) \}}{e _ {k} (X , \alpha)} \right],
$$

which equals $\mu _ { k }$ if either $e _ { k } ( X , \alpha ) = e _ { k } ( X ) \ o r \mu _ { k } ( X , \beta ) = \mu _ { k } ( X )$

Prove Theorem 28.2.

Remark: Theorem 28.1 is a special case of Theorem 28.2 if we view the $( Z , M )$ in Theorem 28.1 as a treatment with four levels. The $\mathrm { C D E } ( m )$ is a special case of $\tau _ { c } .$ .

## 28.3 cde in the linear outcome model

Show that under Assumption 28.1, if $E ( Y \mid Z , M , X ) = \theta _ { 0 } + \theta _ { 1 } Z + \theta _ { 2 } M + \theta _ { 4 } ^ { \mathsf { T } } X$ then

$$
\mathrm{CDE} (m) = \theta_ {1}
$$

for all $m ;$ if $E ( Y \mid Z , M , X ) = \theta _ { 0 } + \theta _ { 1 } Z + \theta _ { 2 } M + \theta _ { 3 } Z M + \theta _ { 4 } ^ { \mathsf { T } } X$ , then

$$
\operatorname{CDE} (m) = \theta_ {1} + \theta_ {3} m.
$$

## 28.4 cde in the logit outcome model

Show that for a binary outcome, under Assumption 28.1, if

$$
\operatorname{logit} \left\{\operatorname * {p r} (Y = 1 \mid Z, M, X) \right\} = \theta_ {0} + \theta_ {1} Z + \theta_ {2} M + \theta_ {4} ^ {\mathsf {T}} X,
$$

then

$$
\operatorname{CDE} (m) = E \{\expit (\theta_ {0} + \theta_ {1} + \theta_ {2} m + \theta_ {4} ^ {\mathsf {T}} X) - \expit (\theta_ {0} + \theta_ {2} m + \theta_ {4} ^ {\mathsf {T}} X) \};
$$

if

$$
\operatorname{logit} \left\{\operatorname{pr} (Y = 1 \mid Z, M, X) \right\} = \theta_ {0} + \theta_ {1} Z + \theta_ {2} M + \theta_ {3} Z M + \theta_ {4} ^ {\mathsf {T}} X,
$$

then

$$
\operatorname{CDE} (m) = E \{\expit (\theta_ {0} + \theta_ {1} + \theta_ {2} m + \theta_ {3} m + \theta_ {4} ^ {\mathsf {T}} X) - \expit (\theta_ {0} + \theta_ {2} m + \theta_ {4} ^ {\mathsf {T}} X) \}.
$$

## 28.5 Recommended reading

Nguyen et al. (2021) provided a friendly review of of the topics in Chapters 27 and 29.
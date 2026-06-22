# Chapter 4 Estimating Heterogeneous Treatment Effects

In many application areas, there is interest in going beyond average effects, and to understand how treatment effects vary across units. In personalized medicine, we may want to identify groups of patients who are more likely to benefit (or less likely to suffer side effects) from a drug than others; and, in online marketing, one may want to identify groups of customers more likely to respond to an offer. This chapter introduces and compares a variety of methods for estimating heterogeneous treatment effects.

The conditional average treatment effect Throughout this chapter, we will work under the same “basic setting” as considered in the previous chapter, i.e., with SUTVA, unconfoundedness and overlap; however, rather than focusing on the average treatment effect, we now seek to estimate, understand, and eventually act on heterogeneity in how different units respond to treatment. At first glance, one might think that estimating treatment heterogeneity should involve targeting the individual-i specific individual treatment effects (ITEs) $\Delta _ { i } = Y _ { i } ( 1 ) - Y _ { i } ( 0 )$ . The ITEs, however, are generally not point-identified even under strong assumptions, and so methodologies targeting the ITEs themselves are often not practical.

A more practical way to quantify treatment heterogeneity under unconfoundedness is via the conditional average treatment effect (CATE)

$$
\tau (x) = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \mid X _ {i} = x \right]. \tag {4.1}
$$

The CATE is still an average effect; but we now consider how this average to varies when conditioning on potential effect modifiers $X _ { i }$ . Note that the definition of the CATE depends on which pre-treatment covariates are used in (4.1): If we condition on a richer set of covariates, then the CATE function will become more expressive (and capture a higher fraction of the variance of the underlying ITEs).

There are many reasons to consider the CATE as a statistical target. It is simple to understand and work with; and, unlike the ITE, it is point-identified. There are also formal, decision theoretic reasons to pay attention to the CATE. For example, the following result (stated here without proof) shows that utilitarian targeting rules can be expressed as thresholding rules on the CATE.

Proposition 4.1. Under the basic setting with SUTVA, unconfoundedness and overlap described in Chapter ${ \mathcal { B } } ,$ suppose a decision maker gets reward $Y _ { i } ( w ) ~ f o r$ assigning treatment arm w to unit i, and needs to pay a cost C every time they assign treatment (the control arm is free). Then, the decision rule that treats units whose CATE is greater than the cost $C , i . e . , 1 \left( \left\{ \tau ( X _ { i } ) > C \right\} \right)$ , maximizes expected rewards among all decision rules that are measurable with respect to observed pre-treatment covariates $X _ { i }$ .

Example 5. Kitagawa and Tetenov [2018] discuss optimal targeting of eligibility to training and job-search assistance under the National Job Training Partnership Act (JTPA). Here, the treatment $W _ { i }$ is program eligibility, the outcome $Y _ { i }$ is earnings within 30 months of treatment assignment, and pretreatment covariates available for targeting are $X _ { i } ~ = ~ \{ \mathrm { e d u c a t i o n }$ , income}. The welfare-maximizing targeting rule then compares the CATE to the cost of treatment.22

Regularization bias Before presenting methods for CATE estimation, it is helpful to review some issues faced by a simple baseline method. Under unconfoundedness, the CATE can be written as a difference in conditional response surfaces,

$$
\tau (x) = \mu_ {(1)} (x) - \mu_ {(0)} (x), \quad \mu_ {(w)} (x) = \mathbb {E} \left[ Y _ {i}   |   X _ {i} = x,   W _ {i} = w \right]. \tag {4.2}
$$

Thus, we could immediately obtain a consistent estimator for $\tau ( \cdot )$ by consistently fitting $\hat { \mu } _ { ( 0 ) } ( \cdot )$ and $\hat { \mu } _ { ( 1 ) } ( \cdot )$ via separate non-parametric regressions on the controls and treated units respectively, and then estimating the CATE as their difference. Following the nomenclature of K¨unzel et al. [2019], the resulting estimator is often referred to as the T-learner:

$$
\hat {\tau} _ {T} (x) = \hat {\mu} _ {(1)} (x) - \hat {\mu} _ {(0)} (x). \tag {4.3}
$$

However, while the T-learner is consistent, it may not perform well in finite samples due to a phenomenon called regularization bias: Given that we fit $\hat { \mu } _ { ( 0 ) } ( \cdot )$ and $\hat { \mu } _ { ( 1 ) } ( \cdot )$ separately, these two functions may end up being regularized in different ways from each other, creating artifacts in the learned CATE estimate ${ \hat { \tau } } _ { T } ( x )$ . This problem is particularly acute if we use methods where the amount of regularization depends on sample size, and if there are many more control than treated units (or vice-versa).23

Figure 4.1, illustrates this issue. There is no treatment effect, so $\mu _ { ( 0 ) } ( x ) =$ $\mu _ { ( 1 ) } ( x )$ and $\tau ( x ) = 0$ , but both regression surfaces oscillate with x. The data is collected via a randomized trial with $\pi = 0 . 1$ , so there are many more controls than treated units. Here, there end up being enough controls for $\hat { \mu } _ { ( 0 ) } ( \cdot )$ to be well estimated and capture the underlying oscillation of the conditional response function. On the other hand, there are very few treated treated units, and so the best we can do with $\hat { \mu } _ { ( 1 ) } ( \cdot )$ is to heavily regularize it, resulting in an estimate that is almost constant in x. Both estimates $\hat { \mu } _ { ( 0 ) } ( \cdot )$ and $\hat { \mu } _ { ( 1 ) } ( \cdot )$ are reasonable on their own; however, once we take their difference as in (4.3), we find strong apparent heterogeneity is ${ \hat { \tau } } _ { T } ( x )$ , which is concerning since in reality $\tau ( x ) = 0$ everywhere in this example.

A second concern with the T-learner, regularization-induced confounding, arises because the T-learner does not explicitly account for variation in the propensity score. If $e ( x )$ varies considerably, then our estimates of $\hat { \mu } _ { ( 0 ) } ( \cdot )$ will be driven by data in areas with more control units (i.e., with $e ( x )$ closer to 0), and those of $\hat { \mu } _ { ( 1 ) } ( \cdot )$ by regions with more treated units (i.e., with $e ( x )$ closer to 1). And if there is covariate shift between the data used to learn $\hat { \mu } _ { ( 0 ) } ( \cdot )$ and $\hat { \mu } _ { ( 1 ) } ( \cdot )$ , this may create biases for their difference ${ \hat { \tau } } _ { T } ( x )$ .

## 4.1 Semiparametric modeling

As our analysis of regularization bias made clear, any good method for estimating the CATE should “focus” on estimating the $\mathrm { C A T E } \tau ( x )$ accurately—and, in a flexible statistical learning setting, this is not necessarily the same thing as simultaneously estimating $\mu _ { ( 0 ) } ( x )$ and $\mu _ { ( 1 ) } ( x )$ accurately. To understand what it takes to successfully target the CATE, it is helpful to start by considering the following semiparametric specification:

$$
\tau (x) = \psi (x) \cdot \beta , \quad \psi : \mathcal {X} \rightarrow \mathbb {R} ^ {d}, \quad \beta \in \mathbb {R} ^ {d}. \tag {4.4}
$$

For example, in the context of Example 5, if X contains unstructured data on income and education, one could set ψ(x) = {income in previous year, has high-school degree, has college degree}.

We refer to this specification as semiparametric because our overall specification is non-parametric (in particular, $\mu _ { ( 0 ) } ( x )$ and $e ( x )$ arbitrary), but we imposed a parametric specification on the key component of interest. Under the model (4.4), estimating the CATE reduces to estimating $\beta$ . Working under the basic setting from Chapter 3 and writing $\varepsilon _ { i } ( w ) = Y _ { i } ( w ) - \mu _ { ( w ) } ( X _ { i } )$ , the addition of the parametric constraint (4.4) lets us re-express our data-generating distribution as a partially linear model,

$$
Y _ {i} (w) = \mu_ {(0)} (X _ {i}) + w   \psi (x) \cdot \beta + \varepsilon_ {i} (w). \tag {4.5}
$$

This class of problems was studied by Robinson [1988] who showed that, for estimating $\beta ,$ it is helpful to re-write (4.5) as

$$
\begin{array}{l} Y _ {i} - m \left(X _ {i}\right) = \left(W _ {i} - e \left(X _ {i}\right)\right) \psi \left(X _ {i}\right) \cdot \beta + \varepsilon_ {i}, \text {where} \\ (.) = \mathbb {E} [ X _ {i} | X _ {i} ] = (X _ {i}) + (Y _ {i}) + (X _ {i}) \cdot \beta . \end{array} \tag {4.6}
$$

$$
m (x) = \mathbb {E} \left[ Y _ {i} \mid X _ {i} = x \right] = \mu_ {(0)} (X _ {i}) + e (X _ {i}) \psi (X _ {i}) \cdot \beta
$$

denotes the conditional expectation of the observed $Y _ { i }$ , marginalizing over $W _ { i }$ and $\varepsilon _ { i } = \varepsilon _ { i } ( W _ { i } )$ .

The expression (4.6) shows that, if we knew $m ( x )$ and $e ( x )$ , then we could estimate $\beta$ via a simple regression algorithm: First define $\widetilde { Y } _ { i } ^ { * } = Y _ { i } - m ( X _ { i } )$ and $\widetilde { Z } _ { i } ^ { * } = \psi ( X _ { i } ) ( W _ { i } - \underset { \sim } { e } ( X _ { i } ) )$ , and then estimate $\hat { \beta } ^ { * }$ by running residual-onresidual regression $\widetilde { Y } _ { i } ^ { * } \sim \widetilde { Z } _ { i } ^ { * }$ . In practice, of course, $e ( x )$ may not be known and $m ( x )$ is essentially never known, and so running the above approach is not feasible.

Our discussion in Chapter 3, however, motivates trying a plug-in approach using the double machine learning framework. We first estimate the unknown components $m ( x )$ and $e ( x )$ via a machine learning method of our choice, and then plug them into (4.6) using cross-fitting:

1. Run non-parametric regressions $Y \sim X$ and $W \sim X$ using a method of our choice to get ${ \hat { m } } ( x )$ and $\hat { e } ( x )$ respectively.  
2. Use cross-fit residuals to define transformed features $\widetilde { Y } _ { i } = Y _ { i } - \hat { m } ^ { ( - k ( i ) ) } ( X _ { i } )$ and $\widetilde { Z } _ { i } = \psi ( X _ { i } ) ( W _ { i } - \hat { e } ^ { ( - k ( i ) ) } ( X _ { i } ) )$ ).  
3. Estimate $\hat { \beta }$ by running a linear regression $\widetilde { Y } _ { i } \sim \widetilde { Z } _ { i }$

As established below, this residual-on-residual regression estimator has a similar special property as established for AIPW in Theorem 3.2: As long as the non-parametric components are reasonably accurately estimated, then $\hat { \beta }$ is asymptotically equivalent to the oracle $\hat { \beta } ^ { * }$ , and satisfies a central limit theorem at the $1 / { \sqrt { n } }$ -scale.24

Theorem 4.2. Under the basic setting with SUTVA, unconfoundedness and overlap described in Chapter ${ \mathcal { B } } ,$ suppose that (4.4) holds, that the regression features are bounded $\| \psi ( X _ { i } ) \| _ { \infty } \leq M$ , and that we estimate $\beta$ via a K-fold

<!-- footnote -->

- See Holland [1986] for one perspective on the work of Neyman [1923] and Rubin [1974] in a historical context.

<!-- footnote end -->

<!-- footnote -->

- One major assumption that’s baked into this notation is that binary counterfactuals exist, i.e., that it makes sense to talk about the effect of choosing to intervene or not on a single unit, without considering the treatments assigned to other units. This may be a reasonable assumption in medicine (i.e., that the treatment prescribed to patient A doesn’t affect patient B), but are less appropriate in some social or economic settings where network effects may arise. We will discuss causal inference under interference in Chapters 11 and 12.

<!-- footnote end -->

<!-- footnote -->

- Here, we’re implicitly assuming that each unit has the same marginal probability of getting treated. Standard experimental designs that satisfy this assumption include the Bernoulli-randomized trial, where each unit is independently treated with probability $0 ~ <$ $\pi < 1 ;$ the completely randomized trial, where each set of $n _ { 1 }$ treated units are equally likely to get chosen for treatment; and the matched-pairs design, where we first pair units according to some algorithm, and then randomly choose one unit in each pair for treatment. Designs that assign different units different marginal treatment probabilities may also be considered; however, as discussed in the next chapter, analyzing them requires more care.

<!-- footnote end -->

<!-- footnote -->

- Note that the Bernoulli trial assumption implies the randomization condition (1.6), but the converse is not true. For example a completely randomized experiment where we give treatment to a set of $\lfloor n _ { 1 } \ = \ n / 2 \rfloor$ units chosen uniformly at random satisfies (1.6) but not (1.8). The reason we consider Bernoulli trials here is that, under this assumption, the treatment assignments $W _ { i }$ across units are independent—thus simplifying the statistical analysis.

<!-- footnote end -->

<!-- footnote -->

- Throughout, we use notation of the type $Y _ { i } \sim X _ { i } \cdot \beta$ to mean that, algorithmically, we have run a regression—here with response $Y _ { i }$ and regressors $X _ { i }$ . In other words, this notation simply means that we assign ${ \hat { \beta } } : = ( \bar { X } ^ { \prime } X ) ^ { - 1 } X ^ { \prime } Y$ . This notation does not imply any implicit model for the data; and in fact, as seen below, one can study the statistical behavior of regression algorithms under different models for the underlying data.

<!-- footnote end -->

<!-- footnote -->

- Despite their similar appearance, we emphasize that (1.13) and (1.16) have completely different meanings: The former describes an algorithm we run on data, while the latter encodes structure we believe the data to satisfy.
- The assumption that $\mathbb { E } \left[ X \right] = 0$ is without loss of generality because all estimators we will consider in this chapter are translation invariant. Of course, however, the analyst cannot be allowed to make use of knowledge that $\mathbb { E } \left[ X \right] = 0$ .

<!-- footnote end -->

<!-- footnote -->

- Under will specification (1.15), the best linear projection coefficients match the parameters of the linear model, i.e., $\alpha _ { ( w ) } ^ { * } = \alpha _ { ( w ) }$ and $\beta _ { ( w ) } ^ { * } = \beta _ { ( w ) }$ .

<!-- footnote end -->

<!-- footnote -->

- For a recent review of asymptotics for linear regression under misspecification, see Buja et al. [2019]; in particular (1.24) follows immediately from Proposition 7.1 of that paper under the assumption that $\dot { \mathbb { E } } \left[ X _ { i } \right] = 0$ .

<!-- footnote end -->

<!-- footnote -->

- Verifying this requires going into details of the proof of (1.24) and so we will not do so here. The key fact leading to these quantities being asymptotically uncorrelated is that, by the first-order condition for the best linear projection coefficients, we must have $\mathrm { C o v } [ Y _ { i } ( w ) - \alpha _ { ( w ) } ^ { * } - X _ { i } \cdot \beta _ { ( w ) } ^ { * } , X _ { i } ] = 0$ .
- The answer is also yes without these assumptions; verifying this is left as an exercise.
- For the third equality, we use the fact that $X \beta _ { ( w ) } ^ { * }$ is the projection of $\mu _ { ( w ) } ( X )$ on to the linear span of the features $X$ , and so $\operatorname { C o v } [ \mu _ { ( w ) } ( X ) , \ddot { X } \beta _ { ( w ) } ^ { * } ] = \operatorname { V a r } [ X \beta _ { ( w ) } ^ { * } ]$ .

<!-- footnote end -->

<!-- footnote -->

- The result stated here should not be over-generalized. We have shown that in one very specific setting—when $X _ { i }$ has discrete support and we use a saturated (and thus trivially well specified) propensity model—then the feasible IPW estimator can outperform the oracle IPW estimator. This result should not be taken to mean that feasible IPW generally beats oracle IPW; and the conditions under which this happens are not present in many important applications (unless, of course, $X _ { i }$ genuinely has low-cardinali $\mathrm { t y , }$ discrete support). In Chapter 3, we will discuss much more robust—and algorithmically generalizable—ways to address the excess asymptotic variance of oracle IPW.

<!-- footnote end -->

<!-- footnote -->

- One question that has received substantial attention in subsequent discussions of the

<!-- footnote end -->

<!-- footnote -->

- In particular, we will be able to handle machine-learning based propensity score estimates as came up in Example 4.

<!-- footnote end -->

<!-- footnote -->

- In the literature, what we here refer to as weak double robustness is often simply referred to as double robustness [Bang and Robins, 2005].

<!-- footnote end -->

<!-- footnote -->

- $^ { 1 7 } \mathrm { A n }$ interesting special case in which this condition holds is when √ $\alpha _ { \mu } , \alpha _ { e } = 1 / 4$ i.e., $\hat { \mu } _ { ( w ) } ( x )$ and $\hat { e } ( x )$ are both $o ( 1 / \sqrt [ 4 ] { n } )$ -consistent in RMSE. In general, parametric models are $\dot { O ( 1 / \sqrt { n } ) }$ -consistent in RMSE; and thus the result (3.5) can accommodate a setting where $\hat { \mu } _ { ( w ) } ( x )$ and $\hat { e } ( x )$ converge an order of magnitude slower than the parametric rate.

<!-- footnote end -->

<!-- footnote -->

- Throughout the rest of the book, whenever AIPW is discussed, we’ll implicitly be using cross-fitting unless specified otherwise. Cross-fitting is also recommended in practice by a number of authors, and is implemented in several software packages for causal inference.

<!-- footnote end -->

<!-- footnote -->

- Note that this application of the Cauchy-Schwarz is somewhat loose. There exist results—albeit with much stronger assumptions—that are able to weaken the rate condition (3.11) by using a stronger argument here.

<!-- footnote end -->

<!-- footnote -->

- Here we make the usual t-distribution degrees-of-freedom adjustment and divide by n−1; however, all statements below would also hold when dividing by n instead.

<!-- footnote end -->

<!-- footnote -->

- This statement is intentionally under-specified; we refer to Chamberlain [1992] for a precise statement.

<!-- footnote end -->

<!-- footnote -->

- $^ { 2 2 } \mathrm { A s }$ always, the value of the CATE depends on the set of covariates $X _ { i }$ used to define it. In this application, one could also try to estimate the treatment effects conditionally on a larger set of covariates, e.g., $X _ { i } =$ {education, income, age, family status, past experience, $\left. \dots \right\}$ , resulting in a more expressive CATE. Proposition 4.1 says that, given a set of measured pre-treatment covariates available for targeting, using the CATE given those covariates is optimal from a welfare maximization point of view. In practice, however, other considerations may also apply; see the next chapter for a further discussion of this topic.

<!-- footnote end -->

<!-- footnote -->

- Throughout this discussion, we assume that the reader is familiar with standard results on bias, variance, regularization, cross-validation, etc., as they arise in statistical learning. A good reference on these topics is Chapter 5 of Hastie, Tibshirani, and Friedman [2009].

<!-- footnote end -->

<!-- footnote -->

- This property is special: For most estimators, cross-fit plug-in versions of the estimator will not be asymptotically equivalent to an oracle version of the estimator under useful conditions. In general, this property requires the estimator to be “Neyman-orthogonal”; in particular, both AIPW and residual-on-residual regression are Neyman-orthogonal. Giving an abstract characterization of Neyman-orthogonality and when it holds is beyond the scope of this book; see Chernozhukov et al. [2022a] for an in-depth study of this topic.

<!-- footnote end -->

cross- $- \mathscr { f } t$ version of residual-on-residual regression as given above. Suppose further that we use estimators for the non-parametric components such that, for all folds k = 1, . . . , K ,

$$
n ^ {- 2 \alpha_ {m}} \frac {1}{| \{i : k (i) = k \} |} \sum_ {\{i: k (i) = k \}} \left(\hat {m} ^ {(- k)} (X _ {i}) - m (X _ {i})\right) ^ {2} \to_ {p} 0,
$$

$$
n ^ {- 2 \alpha_ {e}} \frac {1}{| \{i : k (i) = k \} |} \sum_ {\{i: k (i) = k \}} ^ {\{i: k (i) = k \}} \left(\hat {e} ^ {(- k)} (X _ {i}) - e (X _ {i})\right) ^ {2} \rightarrow_ {p} 0, \tag {4.7}
$$

for some constants satisfying $\alpha _ { m } \geq 0 , \alpha _ { e } \geq 1 / 4$ and $\alpha _ { m } + \alpha _ { e } \ge 1 / 2$ . Then, writing $\widetilde { Z } _ { i } ^ { * }$ and $\widetilde { Z } _ { i } ^ { * }$ are the oracle residuals as defined below (4.6),

$$
\sqrt {n} (\hat {\beta} - \beta) \Rightarrow \mathcal {N} (0, V _ {\beta}), V _ {\beta} = \operatorname{Var} \left[ \widetilde {Z} _ {i} ^ {*} \right] ^ {- 1} \mathbb {E} \left[ \left(\varepsilon_ {i} \widetilde {Z} _ {i} ^ {*}\right) ^ {\otimes 2} \right] \operatorname{Var} \left[ \widetilde {Z} _ {i} ^ {*} \right] ^ {- 1}, \tag {4.8}
$$

provided Var $\left[ \widetilde { Z } _ { i } ^ { * } \right]$ hZ∗i has full rank.

Proof. Under our basic setting and (4.4), the expression (4.6) can be viewed as a well-specified linear model with heteroskedastic errors. Thus, a standard analysis of linear regression under heteroskdasticity [White, 1980] immediately implies that the oracle residual-on-residual regression estimator $\hat { \beta } ^ { * }$ satisfies the limit result (4.8). It thus suffices to show that $\sqrt { n } ( \hat { \beta } - \hat { \beta } ^ { * } ) \to _ { p } 0$ .

We can explicitly write out the feasible and oracle residual-on-residual regression estimators as

$$
\hat {\beta} = \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \widetilde {Z} _ {i} ^ {\otimes 2}\right) ^ {- 1} \frac {1}{n} \sum_ {i = 1} ^ {n} \widetilde {Z} _ {i} \widetilde {Y} _ {i}, \quad \hat {\beta} ^ {*} = \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \widetilde {Z} _ {i} ^ {* \otimes 2}\right) ^ {- 1} \frac {1}{n} \sum_ {i = 1} ^ {n} \widetilde {Z} _ {i} ^ {*} \widetilde {Y} _ {i} ^ {*}. \tag {4.9}
$$

We start showing that, for each fold k

$$
\sqrt {n} \left(\frac {1}{n} \sum_ {\{i: k (i) = k \}} \widetilde {Z} _ {i} \widetilde {Y} _ {i} - \frac {1}{n} \sum_ {\{i: k (i) = k \}} \widetilde {Z} _ {i} ^ {*} \widetilde {Y} _ {i} ^ {*}\right)\rightarrow_ {p} 0.
$$

To do so, we spell out $\widetilde { Y } _ { i } , \widetilde { Z } _ { i }$ , etc., and expand

$$
\begin{array}{l} \sum_ {\{i: k (i) = k \}} \psi (X _ {i}) \left(W _ {i} - \hat {e} ^ {(- k)} (X _ {i})\right) \left(Y _ {i} - \hat {m} ^ {(- k)} (X _ {i})\right) - \psi (X _ {i}) \left(W _ {i} - e (X _ {i})\right) \left(Y _ {i} - m (X _ {i})\right) \\ = \sum_ {\{i: k (i) = k \}} \psi (X _ {i}) \left(W _ {i} - e (X _ {i})\right) \left(m (X _ {i}) - \hat {m} ^ {(- k)} (X _ {i})\right) \\ + \sum_ {\{i: k (i) = k \}} \psi (X _ {i}) \left(e (X _ {i}) - \hat {e} ^ {(- k)} (X _ {i})\right) (Y _ {i} - m (X _ {i})) \\ + \sum_ {\{i: k (i) = k \}} \psi (X _ {i}) \left(e (X _ {i}) - \hat {e} ^ {(- k)} (X _ {i})\right) \left(m (X _ {i}) - \hat {m} ^ {(- k)} (X _ {i})\right). \\ \end{array}
$$

We then bound these terms exactly as in the proof of Theorem 3.2: For the first two terms above we rely on cross-fitting; while for the last we use Cauchy-Schwarz (relying on our assumptions that $\alpha _ { m } + \alpha _ { e } \ge 1 / 2$ and $\| \psi ( X _ { i } ) \| _ { \infty } \leq M )$ . The fact that

$$
\sqrt {n} \left(\frac {1}{n} \sum_ {\{i: k (i) = k \}} \widetilde {Z} _ {i} ^ {\otimes 2} - \frac {1}{n} \sum_ {\{i: k (i) = k \}} \widetilde {Z} _ {i} ^ {* \otimes 2}\right)\rightarrow_ {p} 0
$$

follows by the same argument, except now we need to use $2 \alpha _ { e } \geq 1 / 2$ in the Cauchy-Schwarz bound. Finally, to put everything together, we invoke Slutsky’s lemma, the fact that

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \widetilde {Z} _ {i} ^ {* \otimes 2} \rightarrow_ {p} \operatorname{Var} \left[ \widetilde {Z} _ {i} ^ {*} \right] \succ 0,
$$

and that the matrix inverse is a continuous function in the neighborhood of full-rank matrices. □

The constant effect model One interesting special case of semiparametric modeling is the constant treatment effect model

$$
\mu_ {(1)} (x) - \mu_ {(0)} (x) = \tau , \tag {4.10}
$$

whereby we assert that treatment effects do not vary with covariates; this is an instance of (4.4) with $\psi ( x ) = 1$ . We can thus also apply the residual-onresidual regression approach developed above in this setting, resulting in the following:

Corollary 4.3. Under the basic setting with SUTVA, unconfoundedness and overlap from Chapter 3, suppose that the constant treatment effect model (4.10) holds, and we estimate τ via a cross-fit plug-in residual-on-residual estimator with non-parametric components satisfying (4.7). Then,

$$
\begin{array}{l} \sqrt {n} (\hat {\tau} - \tau) \Rightarrow \mathcal {N} (0, V _ {\tau}), \\ V _ {\tau} = \frac {\mathbb {E} \left[ e (X _ {i}) (1 - e (X _ {i})) \left((1 - e (X _ {i})) \sigma_ {(1)} ^ {2} (X _ {i}) + e (X _ {i}) \sigma_ {(0)} ^ {2} (X _ {i})\right) \right]}{\mathbb {E} \left[ e (X _ {i}) (1 - e (X _ {i}) \right] ^ {2}}. \tag {4.11} \\ \end{array}
$$

Note that, under the model (4.10), one could also have estimated the parameter τ via methods for the average treatment effect such as AIPW (because, when the treatment effect is constant τ , then the average treatment effect is also τ ). However, AIPW would in this case generally be less accurate than the residual-on-residual regression estimator. In particular, in the special case where (4.10) holds and $\sigma _ { ( 0 ) } ^ { 2 } \bar { ( x ) } = \sigma _ { ( 1 ) } ^ { 2 } ( x ) = \sigma ^ { 2 }$ , then25

$$
V _ {\tau} = \frac {\sigma^ {2}}{\mathbb {E} [ e (X _ {i}) (1 - e (X _ {i}) ]} \leq \sigma^ {2} \mathbb {E} \left[ \frac {1}{e (X _ {i}) (1 - e (X _ {i}))} \right] = V _ {A I P W}, \tag {4.12}
$$

where the inequality above follows from Jensen’s inequality. This observation highlights the fact that efficiency of an estimator for a specific target depends closely on assumptions made. We showed Chapter 3 that AIPW is efficient in our generic non-parametric setting; however, once we add an extra constraint like (4.10), then estimators that exploit this constraint can do better.26

## 4.2 A loss function for treatment heterogeneity

The residual-on-residual regression estimator developed above is helpful if we believe in the semiparametric specification (4.4). In order to meet our original goal of estimating the CATE in a generic setting with unconfoundedness, however, we need to generalize this estimator to a fully non-parametric setting.

As background for how to do this, it is helpful to think in terms of how this generalization was carried out in the context of simple prediction, i.e., predicting a real-valued $Y _ { i }$ from features $X _ { i }$ . The classical approach to doing so is via linear regression, but nowadays methods like decision trees, boosting and neural networks offer compelling non-parametric alternatives. Key insights in this progression include the use of flexible basis expansions to express more complicated signals; penalization to keep the complexity of the learned predictor in check despite the use of high-dimensional basis expansions; cross-validation to tune the amount of penalization; and algorithmic techniques like decision trees and neural networks to adaptively generate basis expansions suited to the task at hand. Hastie, Tibshirani, and Friedman [2009] provide an excellent book-length presentation of these concepts; Chapters 3, 5 and 7 are particularly relevant for understanding the discussion below.

Our task here is to deploy all these concepts to CATE estimation. To this end, we start by writing the residual-on-residual regression from above as a loss-minimization problem. Recall that, in the simple prediction case, the ordinary least-squares solution $\hat { \beta }$ to regressing $Y _ { i }$ on $\psi ( X _ { i } )$ using n samples can be characterized via squared-error loss minimization,

$$
\hat {\beta} = \operatorname{argmin} _ {\beta} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \ell_ {r e g} (Y _ {i}; \psi (X _ {i}) \cdot \beta) \right\}, \quad \ell_ {r e g} (y; z) = (y - z) ^ {2}. \tag {4.13}
$$

By the same argument, we can verify that our residual-on-residual regression algorithm also minimizes a certain least-squares objective, namely27

$$
\hat {\beta} = \operatorname{argmin} _ {\beta} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \hat {\ell} ^ {(- k (i))} \left(X _ {i}, Y _ {i}, W _ {i}; \psi (X _ {i}) \cdot \beta\right) \right\} \tag {4.14}
$$

$$
\hat {\ell} ^ {(- k)} (x, y, w; z) = \left(\left(y - \hat {m} ^ {(- k)} (x)\right) - (w - \hat {e} ^ {(- k)} (x)) z\right) ^ {2}.
$$

One critical difference between (4.13) and (4.14) is that, in our setting, the $^ {  } \mathrm { l o s s } ^ {  }$ function $\hat { \ell } ^ { ( - k ) }$ is data-dependent, and takes as input our cross-fitted predictions for $m ( \cdot )$ and $e ( \cdot )$ . The fact that our loss function is data-dependent in this way will lead to technical challenges down the road; however, it does not preclude us from proceeding with algorithm development.

We are now ready to apply the statistical learning roadmap to CATE estimation. We still start from the semiparametric specification (4.4); however, we now consider featurizations $\psi : \mathcal { X }  \mathbb { R } ^ { d _ { n } }$ that map our input covariates $X _ { i }$ into increasingly high-dimensional representations as our sample size grows. For example, $\psi$ could consist of a set of polynomial or trigonometric basis functions with increasing numbers of terms. The motivation with this approach is that, once we include enough basis functions, we will be able to accurately represent any reasonable CATE function using this basis, i.e., we have $\tau ( x ) \approx \psi ( x ) \cdot \beta$ for some $\beta \in \mathbb { R } ^ { d _ { n } }$ [Chen, 2007].

The second step in the statistical learning roadmap is to introduce penalization to control the complexity of the learned CATE function because, when $d _ { n }$ is large relative to $n ,$ directly running a residual-on-residual regression with covariates $\psi ( x )$ may be unstable. One choice here is to use the lasso penalty [Tibshirani, 1996], which penalizes the sum of the absolute values of $\beta \colon$

$$
\hat {\tau} (x) = \psi (x) \cdot \hat {\beta},
$$

$$
\hat {\beta} = \operatorname{argmin} _ {\beta} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \hat {\ell} ^ {(- k (i))} \left(X _ {i}, Y _ {i}, W _ {i}; \psi (X _ {i}) \cdot \beta\right) + \lambda \sum_ {j = 1} ^ {q} | \beta_ {j} | \right\}, \tag {4.15}
$$

where $\lambda \geq 0$ is a penalty parameter that controls the complexity of the learned function. A judicious choice of λ enables us to still get a good estimate ${ \hat { \tau } } ( x )$ , but protects against the risks of overfitting or numerical instability that occur when $\psi ( x )$ is high-dimensional. Using $\lambda = 0$ corresponds to just running linear regression of $Y _ { i }$ on $\psi ( X _ { i } )$ , while in the limit $\lambda \to \infty$ all coefficients $\hat { \beta }$ get pushed to 0. Another simple choice would be to use a ridge penalty, which adds a term $\lambda \sum _ { j = 1 } ^ { q } \beta _ { j } ^ { 2 }$ to the objective.

In order to make (4.15) actionable, we need a data-driven way to choose the tuning parameter λ. The simplest way to proceed, is using a validation set, i.e., assuming that we have access to $i = 1 , \ldots , n _ { v a l }$ independent datapoints that can be used for validation. To choose λ, we start running (4.15) for a grid of candidate λ values, resulting in a large number of candidate estimates ${ \hat { \tau } } _ { \lambda } ( x )$ . Then, we pick the value of λ that minimizes the validation loss,28

$$
\hat {\lambda} = \operatorname{argmin} _ {\lambda} \left\{\frac {1}{n _ {v a l}} \sum_ {\text { validation   set }} \hat {\ell} (X _ {i}, Y _ {i}, W _ {i}; \hat {\tau} _ {\lambda} (X _ {i})) \right\}, \tag {4.16}
$$

and finally use CATE predictions $\hat { \tau } ( x ) = \hat { \tau } _ { \hat { \lambda } } ( x )$ . Another, similar way of choosing λ that does not require access to an independent validation set is to use cross-validation; see Chapter $7$ of Hastie, Tibshirani, and Friedman [2009] for details.

The last step in moving from our residual-on-residual regression estimator for semiparametric modeling to a fully flexible non-parametric CATE estimator is to use algorithmic techniques like decision trees, boosting, or neural networks to automate the choice of good basis expansions $\psi ( x )$ . Doing so, however, is beyond the scope of this book; we instead refer to Nie and Wager [2021] for a completion of this discussion. The resulting algorithmic approach is called the R-learner. The causal forest algorithm of Athey, Tibshirani, and Wager [2019] instantiates the R-learner framework using random forests [Breiman, 2001].29 Foster and Syrgkanis [2023] provide general formal results showing that, even after moving to a complex non-parametric setting, the R-learner still maintains robustness properties suggested in Theorem 4.2.

A numerical example We now test out the lasso-based R-learner based approach (4.15), and compare it with a lasso-based T-learner approach (4.3)where both $\hat { \mu } _ { ( 0 ) } ( \cdot )$ and $\hat { \mu } _ { ( 1 ) } ( \cdot )$ are fit with a lasso using predictors $\psi ( X _ { i } )$ . We independently generate $n = 4$ , 000 samples as follows:

$$
X \sim \mathcal {N} (0, I _ {1 0 \times 1 0}), W \sim \text { Bernoulli } (e (X)), e (X) = 1 / \left(1 + e ^ {- (X _ {2} + X _ {3})}\right)
$$

$$
Y (w) = 2 \log \left(1 + e ^ {X _ {1} + X _ {2} + X _ {3}}\right) + w   1 \left(X _ {2} + X _ {3} \geq 0\right) + \varepsilon , \varepsilon \sim \mathcal {N} (0, 1).
$$

The original covariates are 10-dimensional, but the signal is obviously nonlinear and so simple linear methods would be inappropriate here. To address this challenge, we expand our covariates into a 2555-dimensional basis expansion $\psi ( X _ { i } )$ that includes both non-linearities and interactions between the covariates.30 We then use lasso penalization with a cross-validated choice of λ to avoid instability due to our use of a high-dimensional basis expansion.

What’s challenging about this setting is that units for which $X _ { 2 } + X _ { 3 }$ is large are simultaneously more likely to be treated, have a larger baseline effect whether or not they get treated, and have a larger treatment effect. This type of situation may arise, e.g., in evaluating educational programs if there exists a class of, say, high-initiative people who are simultaneously more likely to seek out and benefit from the educational resources, but also would have achieved reasonably good outcomes without the resource. In settings like this, in order to avoid regularization-induced confounding, it is important to accurately correct for the correlation between propensity scores and baseline effects.

Results with both the R-learner and T-learner are shown in Figure 4.2. The y-axis of the plot shows CATE estimates $\hat { \tau } ( X _ { i } )$ , while the x-axis shows $X _ { i 2 } + X _ { i 3 }$ . The choice of x-axis reflects that, in reality, we know that the CATE only varies with $X _ { i 2 } + X _ { i 3 }$ . The algorithm, of course, does not know this a-priori—and this is why the actual CATE estimates $\hat { \tau } ( X _ { i } )$ also depend on other aspects of the covariates (and this manifests itself as apparent noise in the estimates). Here, we see that the R-learner has somewhat noisy estimates, but gets the overall order of magnitude of the CATE right. In contrast, the Tlearner appears to suffer from severe regularization-induced confounding here, and vastly overstates the amount by which $\tau ( X _ { i } )$ grows with $X _ { i 2 } + X _ { i 3 }$ .

## 4.3 Bibliographic notes

The literature on non-parametric CATE estimation has received a huge amount of attention in recent years. Some proposed methods for CATE estimation are based on specific machine learning methods, e.g., trees [Athey and Imbens, 2016], random forests [Athey, Tibshirani, and Wager, 2019] or Bayesian tree ensembles [Hahn, Murray, and Carvalho, 2020]. Others are more generic, and can be paired with multiple algorithmic approaches. We here discussed the Rlearner [Nie and Wager, 2021]; other generic approaches to CATE estimation include the X-learner [K¨unzel et al., 2019] and the DR-learner [Kennedy, 2023], and the modified covariate learner [Tian et al., 2014].

One important topic we did not focus today is what to do after we produce a CATE estimate. After fitting a CATE estimator it is generally good practice to seek to formally validate its output and quantify the strength of heterogeneity; some proposals for how to do so are given in Chernozhukov et al. [2017] and Yadlowsky et al. [2021]. Meanwhile, if the goal of fitting a CATE model was to guide treatment choice, then Proposition 4.1 suggests that empirical thresholding rules of the form 1 $( \{ \hat { \tau } ( x ) > C \} )$ are at least worth considering. Manski [2004], Stoye [2009] and Hirano and Porter [2009] study properties of such thresholding learns under the lens of statistical decision theory. Sun et al. [2021] discuss settings where the treatment cost $C _ { i }$ is random and may also vary with covariates $X _ { i }$ .

In terms of formal results, Kennedy et al. [2024] show that a variant of the R-learner is minimax for estimating CATEs under a set of smoothness assumptions, while Foster and Syrgkanis [2023] provide guarantees for machine learning with a class of “orthogonal” loss functions that include the R-loss. Zhao, Small, and Ertefaie [2022] consider post-selection inference for the CATE in a high-dimensional linear specification using an algorithm that builds on the semiparametric estimator from Theorem 4.2.

Finally, we also note some work on treatment heterogeneity based on difference conceptual frameworks. Although the ITE is not generally pointidentified, we can still seek bounds or intervals for it. Lei and Cand\`es [2021] provide one such method for doing this using conformal inference. Ding, Feller, and Miratrix [2019] study heterogeneous treatment effect estimation in a randomized trial under the strict Neyman model for randomization inference discussed in Chapter 1, and examine what can be said about treatment heterogeneity without making any sampling assumptions on the potential outcomes.
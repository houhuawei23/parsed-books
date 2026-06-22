# Chapter 1 Randomized Controlled Trials

How best to understand and characterize causality is an age-old question in philosophy. As such, one might expect that any discussion of causal inference would need to be framed in terms of subtle and esoteric concepts. However, a ground-breaking line of work starting with Neyman [1923] and Rubin [1974] established that—although causality is in general a delicate and complicated notion—there exists an important class of problems, randomized controlled trials, where it is possible to approach causal questions in a practical and conceptually straight-forward way via careful application of randomization, averaging, and counterfactual reasoning.1

This chapter presents a brief overview of statistical estimation and inference in randomized controlled trials (RCTs). When available, evidence drawn from RCTs is often considered gold standard statistical evidence; and thus methods for studying RCTs form the foundation of the statistical toolkit for causal inference. Furthermore, many widely used observational study designs in, e.g., econometrics or epidemiology are motivated by analogy to RCTs; and so this chapter will also serve as a stepping stone to subsequent discussions of estimation and inference in observational studies.

Average treatment effects Suppose that we have run a RCT with n study participants $i = 1 , \ldots , n$ , where each unit i is assigned a binary treatment $W _ { i } \in \{ 0 , 1 \}$ and we then measure an outcome $Y _ { i } .$ . Our goal is to estimate the effect of the treatment on the outcome. Following the Neyman–Rubin causal model, we define the causal effect of a treatment via potential outcomes: For each treatment level $w \in \{ 0 , 1 \}$ , we define potential outcomes $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ corresponding to the outcome the i-th subject would have experienced had they respectively received the treatment or not, such that $Y _ { i } = Y _ { i } ( W _ { i } )$ .

The individual causal effect of the treatment on the i-th unit is then2

$$
\Delta_ {i} = Y _ {i} (1) - Y _ {i} (0). \tag {1.1}
$$

The fundamental problem in causal inference is that only one treatment can be assigned to a given individual, and so only one of $Y _ { i } ( 0 )$ and $Y _ { i } ( 1 )$ can ever be observed. Thus, $\Delta _ { i }$ can never be observed directly.

Although $\Delta _ { i }$ is itself unknowable, we can (perhaps remarkably) use randomized experiments to learn certain properties of the $\Delta _ { i }$ . In finite samples, without any assumptions on how study participants were generated (or equivalently, conditionally on the potential outcomes of study participants), randomization enables us to get unbiased estimates of the sample average treatment effect (SATE)

$$
\overline {{\Delta}} = \frac {1}{n} \sum_ {i = 1} ^ {n} (Y _ {i} (1) - Y _ {i} (0)). \tag {1.2}
$$

Furthermore, if we assume that study participants are independently drawn from a population $P ,$ , then randomized experiments enable unbiased and largesample consistent estimates of the (population) average treatment effect (ATE)

$$
\tau = \mathbb {E} _ {P} \left[ Y _ {i} (1) - Y _ {i} (0) \right]. \tag {1.3}
$$

This chapter will discuss properties of a number of different estimators for these two quantities.

## 1.1 Difference-in-means estimation

In a randomized controlled trial, there are many ways to estimate the average treatment effect. Perhaps the simplest and most intuitive way of doing so is via the difference-in-means estimator,

$$
\hat {\tau} _ {D M} := \frac {1}{n _ {1}} \sum_ {W _ {i} = 1} Y _ {i} - \frac {1}{n _ {0}} \sum_ {W _ {i} = 0} Y _ {i}, \quad n _ {w} = | \{i: W _ {i} = w \} |. \tag {1.4}
$$

In our setting, this difference in means estimator is unbiased essentially without assumptions, and the average treatment effect is identified directly via randomization. Suppose that the potential outcomes model given above is valid; or, as this is often stated in the literature, that the Stable Unit Treatment Values Assumption (SUTVA) holds:

$$
Y _ {i} = Y _ {i} (W _ {i}), \quad i = 1, \dots , n. \tag {1.5}
$$

Suppose furthermore that the treatment is in fact randomized, i.e., that conditionally all the potential outcomes $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \} _ { i = 1 } ^ { n }$ and the number of treated units $n _ { 1 }$ , all units are treated with the same probability:3

$$
\mathbb {P} \left[ W _ {i} = 1 \mid \{Y _ {i} (0), Y _ {i} (1) \} _ {i = 1} ^ {n}, n _ {1} \right] = \frac {n _ {1}}{n}, \quad i = 1, \dots , n. \tag {1.6}
$$

Then ${ \hat { \tau } } _ { D M }$ is finite-sample unbiased for the SATE as defined in (1.2).

Theorem 1.1. Under assumptions (1.5) and (1.6),

$$
\mathbb {E} \left[ \hat {\tau} _ {D M} \mid \{Y _ {i} (0), Y _ {i} (1) \} _ {i = 1} ^ {n}, n _ {0} > 0, n _ {1} > 0 \right] = \overline {{\Delta}}. \tag {1.7}
$$

Proof. Whenever $n _ { 1 } > 0$ , i.e., we have at least 1 treated unit,

$$
\begin{array}{l} \mathbb {E} \left[ \frac {1}{n _ {1}} \sum_ {W _ {i} = 1} Y _ {i} \mid \{Y _ {i} (0), Y _ {i} (1) \} _ {i = 1} ^ {n}, n _ {1} \right] \\ = \mathbb {E} \left[ \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} W _ {i} Y _ {i} \mid \{Y _ {i} (0), Y _ {i} (1) \} _ {i = 1} ^ {n}, n _ {1} \right] \\ = \mathbb {E} \left[ \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} W _ {i} Y _ {i} (1) \mid \left\{Y _ {i} (0), Y _ {i} (1) \right\} _ {i = 1} ^ {n}, n _ {1} \right] \tag {SUTVA} \\ = \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Y _ {i} (1) \mathbb {E} \left[ W _ {i} \mid \{Y _ {i} (0), Y _ {i} (1) \} _ {i = 1} ^ {n}, n _ {1} \right] \\ = \frac {1}{n} \sum_ {i = 1} ^ {n} Y _ {i} (1) \quad (\text { random   assignment }). \\ \end{array}
$$

An analogous result holds for the average of the controls when $n _ { 0 } > 0 .$ .

Population Asymptotics The result in Theorem 1.1 is valuable in its generality: It provides an unbiasedness result under minimal assumptions, and in particular makes no distributional assumptions on the potential outcomes. In practical terms, this means we can apply Theorem 1.1 without making any claims about how the n study participants were recruited.

A limitation of this result, however, is that it does not characterize the sampling error $\hat { \tau } _ { D M } - \overline { { \Delta } }$ , and so doesn’t directly provide a roadmap to statistical inference. In order to make progress, we here make an additional assumption that the study participants (i.e., formally, the pairs of potential outcomes $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \} )$ are independently drawn from a population P . Such population-sampling assumptions then enable straight-forward distributional results and confidence intervals via standard large-sample analysis. It is also possible to obtain distributional results without making such sampling assumptions, but doing so relies on specialized statistical techniques that we will not pursue for now; we will revisit population-sampling-free methods for inference in the bibliographic notes at the end of this chapter and in Chapter 12.

Example 1. In 2008, Oregon ran a lottery to allocate additional spots in its Medicaid program to low-income adults. As reported in Finkelstein et al. [2012], ∼ 90, 000 people joined the lottery, and of them a (randomly selected) ∼ 35, 000 were allowed to apply for Medicaid. The authors consider a number of outcomes, such as healthcare use and expenditures. Finite-sample analysis following Theorem 1.1 shows that, among lottery participants, the differencein-means estimator is unbiased for the average effect of being allowed to apply for Medicaid on outcomes considered, regardless of how the set of lottery participants was created. The asymptotic tools discussed below make a further assumption that the lottery participants were independently sampled from from a relevant larger population (e.g., able-bodied, low-income, uninsured adults with interest in gaining insurance coverage).

A central limit theorem In addition to IID sampling, we will also be more specific about how treatment is randomized, and assume that we are in a Bernoulli trial with4

$$
W _ {i} \mid \{Y _ {i} (0), Y _ {i} (1) \} \stackrel {\text { iid }} {\sim} \operatorname{Bernoulli} (\pi), \quad 0 <   \pi <   1. \tag {1.8}
$$

The following central limit theorem for the difference-in-means estimator can then be established via simple statistical arguments.

Theorem 1.2. Under the assumptions of Theorem $1 . 2 ,$ suppose furthermore that the potential outcomes are drawn as $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \} \stackrel { i i d } { \sim } P$ from a distribution P with bounded second moments and that we run a Bernoulli trial as in (1.8). Then,

$$
\sqrt {n} \left(\hat {\tau} _ {D M} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {D M}\right), V _ {D M} = \frac {\mathrm{Var} \left[ Y _ {i} (0) \right]}{1 - \pi} + \frac {\mathrm{Var} \left[ Y _ {i} (1) \right]}{\pi}. (1. 9)
$$

Furthermore, the plug-in variance estimate

$$
\widehat {V} _ {D M} := \frac {n}{n _ {0} ^ {2}} \sum_ {W _ {i} = 0} \left(Y _ {i} - \frac {1}{n _ {0}} \sum_ {W _ {i} = 0} Y _ {i}\right) ^ {2} + \frac {n}{n _ {1} ^ {2}} \sum_ {W _ {i} = 1} \left(Y _ {i} - \frac {1}{n _ {1}} \sum_ {W _ {i} = 1} Y _ {i}\right) ^ {2} \tag {1.10}
$$

is consistent, $\widehat { V } _ { D M } \to _ { p } V _ { D M }$ .

Proof. Defining potential outcome residuals $\varepsilon _ { i } ( w ) = Y _ { i } ( w ) - \mathbb { E } _ { P } \left[ Y _ { i } ( w ) \right]$ for $w = 0 , 1$ , we can express our estimation error as

$$
\begin{array}{l} \hat {\tau} _ {D M} - \tau = \frac {1}{n _ {1}} \sum_ {W _ {i} = 1} \varepsilon_ {i} (1) - \frac {1}{n _ {0}} \sum_ {W _ {i} = 1} \varepsilon_ {i} (0) \\ = \frac {n}{n _ {1}} \frac {1}{n} \sum_ {i = 1} ^ {n} W _ {i} \varepsilon_ {i} (1) - \frac {n}{n _ {0}} \frac {1}{n} \sum_ {i = 1} ^ {n} (1 - W _ {i}) \varepsilon_ {i} (0). \\ \end{array}
$$

By randomization, one can verify that E $\begin{array} { r } { [ W _ { i } \varepsilon _ { i } ( 1 ) ] = \mathbb { P } \left[ W _ { i } \right] \mathbb { E } \left[ \varepsilon _ { i } ( 1 ) \big | W _ { i } = 1 \right] = } \end{array}$ $\mathbb { P } \left[ W _ { i } \right] \mathbb { E } \left[ \varepsilon _ { i } ( 1 ) \right] = 0$ and E $[ ( 1 - W _ { i } ) \varepsilon _ { i } ( 0 ) ] = 0$ , and finally

$$
\begin{array}{l} \text {Var} \left[ \binom{W _ {i}   \varepsilon_ {i} (1)}{(1 - W _ {i})   \varepsilon_ {i} (0)} \right] = \mathbb {E} \left[ \binom{W _ {i}   \varepsilon_ {i} (1)}{(1 - W _ {i})   \varepsilon_ {i} (0)} ^ {\otimes 2} \right] \\ = \left( \begin{array}{c c} \pi   \text {Var}   [ \varepsilon_ {i} (1) ] & 0 \\ 0 & (1 - \pi)   \text {Var}   [ \varepsilon_ {i} (0) ] \end{array} \right). \\ \end{array}
$$

Thus, by the standard multivariate central limit theorem

$$
\sqrt {n} \binom{\frac {1}{n} \sum_ {i = 1} ^ {n} W _ {i} \varepsilon_ {i} (1)}{\frac {1}{n} \sum_ {i = 1} ^ {n} (1 - W _ {i}) \varepsilon_ {i} (0)} \Rightarrow \mathcal {N} \left(0, \left( \begin{array}{c c} \pi \operatorname{Var} [ \varepsilon_ {i} (1) ] & 0 \\ 0 & (1 - \pi) \operatorname{Var} [ \varepsilon_ {i} (0) ] \end{array} \right)\right).
$$

The result (1.9) follows by Slutsky’s lemma because the treatment fraction of a Bernoulli trial concentrates, $n _ { 1 } / n \to _ { p } \pi$ . Meanwhile, (1.10) follows similarly via the weak law of large numbers. □

The above central limit theorem for ${ \hat { \tau } } _ { D M }$ immediately enables asymptotically valid Gaussian confidence intervals for τ . For any $0 < \alpha < 1$ ,

$$
\lim _ {n \to \infty} \mathbb {P} \left[ \tau \in \left(\hat {\tau} _ {D M} \pm \Phi^ {- 1} (1 - \alpha / 2) \sqrt {\widehat {V} _ {D M} / n}\right) \right] = 1 - \alpha , \tag {1.11}
$$

where Φ denotes the standard Gaussian cumulative distribution function.

From a certain perspective, one could argue that the above is all that is needed to estimate average treatment effects in randomized trials. The difference in means estimator ${ \hat { \tau } } _ { D M }$ is consistent and allows for valid asymptotic inference; moreover, the estimator is very simple to implement, and hard to “cheat” with (i.e., there is little room for an unscrupulous analyst to try different estimation strategies and report the one that gives the answer closest to the one they want). On the other hand, our discussion so far has not established that ${ \hat { \tau } } _ { D M }$ is an “optimal” way to use the data in any meaningful sense; and in fact, we’ll see below that it’s often possible to design estimators with guarantees that strictly dominate those for ${ \hat { \tau } } _ { D M }$ .

## 1.2 Regression adjustments in randomized trials

When analyzing randomized controlled trials, we often have access to pretreatment covariates $X _ { i }$ observed together with the treatments $W _ { i }$ and outcomes $Y _ { i }$ . In this case, practitioners often choose to estimate treatment effects via a linear regression based approach rather than via the simple difference in means.

There are two standard ways to estimate average treatment effects via linear regression. The first is to fit a simple linear regression5

$$
Y _ {i} \sim \alpha + W _ {i} \tau + X _ {i} \cdot \beta , \tag {1.12}
$$

and then report the resulting coefficient $\hat { \tau } _ { S R E G } : = \hat { \tau }$ as an estimate of the average treatment effect. The second is to add in full treatment-covariate interactions, and to fit the interacted linear regression

$$
Y _ {i} \sim \alpha + W _ {i} \tau + X _ {i} \cdot \beta + W _ {i} X _ {i} \cdot \gamma . \tag {1.13}
$$

One can then estimate the average treatment effect via the average difference in predictions if everyone vs. no one were treated

$$
\begin{array}{l} \hat {\tau} _ {I R E G} = \frac {1}{n} \sum_ {i = 1} ^ {n} \hat {\alpha} + \hat {\tau} + X _ {i} \cdot (\hat {\beta} + \hat {\gamma}) - \frac {1}{n} \sum_ {i = 1} ^ {n} \hat {\alpha} + X _ {i} \cdot \hat {\beta}, \tag {1.14} \\ = \hat {\tau} + \overline {{X}} \cdot \hat {\gamma}, \quad \overline {{X}} := \frac {1}{n} \sum_ {i = 1} ^ {n} X _ {i}. \\ \end{array}
$$

Both the simple and interacted regression can reasonably be deployed in randomized experiments. For the rest of this chapter, we will focus on properties of the interacted regression estimator $\hat { \tau } _ { I R E G }$ because it allows for transparent analysis and is also generally regarded a best practice in the current literature on causal inference; see the bibliographic notes for further discussion.

Regression adjustments under linearity The linear regression estimator (1.13) is a statistical estimator that can be studied under a number of different models for the data. The simplest setting under which to consider the behavior of $\hat { \tau } _ { I R E G }$ (and compare it to that of ${ \hat { \tau } } _ { D M } )$ is under an assumption that the regression model (1.13) is well specified; and this is the setting we will start with here.

Suppose for now that our samples are independently is generated via a Bernoulli randomized trial (1.8) with outcomes $Y _ { i } = Y _ { i } ( W _ { i } )$ and

$$
\begin{array}{l} Y _ {i} (w) = \alpha_ {(w)} + X _ {i} \cdot \beta_ {(w)} + \varepsilon_ {i} (w), \\ \mathbb {T} [ (x) | x ] = 0, \quad \forall x <   [ (x) | x ], \quad 2 \end{array} \tag {1.15}
$$

$$
\mathbb {E} \left[ \varepsilon_ {i} (w) \mid X _ {i} \right] = 0, \mathrm{Var} \left[ \varepsilon_ {i} (w) \mid X _ {i} \right] = \sigma^ {2}.
$$

Under Bernoulli randomization, one can check that the observables $( X _ { i } , Y _ { i } , W _ { i } )$ are independently drawn from a distribution satisfying6

$$
Y _ {i} = \alpha_ {(0)} + W _ {i} (\alpha_ {(1)} - \alpha_ {(0)}) + X _ {i} \cdot \beta_ {(0)} + W _ {i} X _ {i} \cdot (\beta_ {(1)} - \beta_ {(0)}) + \varepsilon_ {i}, \tag {1.16}
$$

with E $\left[ \varepsilon _ { i } \big | X _ { i } , W _ { i } \right] = 0$ and Var $\left[ \varepsilon _ { i } \big | X _ { i } , W _ { i } \right] = \sigma ^ { 2 }$ , i.e., the regression (1.13) is in fact well specified. For simplicity, we will further assume that we are in a balanced randomized trial with $\pi = 5 0 \%$ , and (without loss of generality) E $[ X ] = 0 . ^ { 7 }$As a warm-up, we first study the behavior of ${ \hat { \tau } } _ { D M }$ under this model as a baseline; we will then be able to compare it with $\hat { \tau } _ { I R E G }$ . Given our general result in Theorem 1.2 all that remains to be done is to spell out what $V _ { D M }$ is here; and, writing Var $[ X ] = A$ , we we get (recall that we’re using $\pi = 0 . 5$ for simplicity)

$$
\begin{array}{l} V _ {D M} = \frac {\operatorname{Var} [ Y _ {i} (0) ]}{0 . 5} + \frac {\operatorname{Var} [ Y _ {i} (1) ]}{0 . 5} \\ = 2 \left(\operatorname{Var} \left[ X _ {i} \beta_ {(0)} \right] + \sigma^ {2}\right) + 2 \left(\operatorname{Var} \left[ X _ {i} \beta_ {(1)} \right] + \sigma^ {2}\right) \tag {1.17} \\ = 4 \sigma^ {2} + 2 \left\| \beta_ {(0)} \right\| _ {A} ^ {2} + 2 \left\| \beta_ {(1)} \right\| _ {A} ^ {2} \\ = 4 \sigma^ {2} + \left\| \beta_ {(0)} + \beta_ {(1)} \right\| _ {A} ^ {2} + \left\| \beta_ {(0)} - \beta_ {(1)} \right\| _ {A} ^ {2}, \\ \end{array}
$$

where we used the notation $\| v \| _ { A } ^ { 2 } = v ^ { \prime } A v$ for convenience.

Given that the linear regression model is well specified here, one should expect that $\hat { \tau } _ { I R E G }$ improves over the performance of ${ \hat { \tau } } _ { D M } \mathbf { ; }$ the question is by how much. To study the regression estimator, it is helpful to note that the interacted regression (1.13) is algorithmically equivalent to running separate regressions for the treated and control groups and then taking differences of their predictions on the full study sample:

$$
Y _ {i} \sim \alpha_ {(0)} + X _ {i} \cdot \beta_ {(0)} \text {for all i with W_{i} = 0},
$$

$$
Y _ {i} \sim \alpha_ {(1)} + X _ {i} \cdot \beta_ {(1)} \text {for all} i \text {with} W _ {i} = 0,
$$

$$
\hat {\tau} _ {I R E G} = \hat {\alpha} _ {(1)} - \hat {\alpha} _ {(0)} + \overline {{X}} \left(\hat {\beta} _ {(1)} - \hat {\beta} _ {(0)}\right).
$$

Standard results about linear regression then imply that, under model (1.15) (recall also that, here, we assume that $\mathbb { E } \left[ X \right] = 0 )$

$$
\sqrt {n _ {w}} \left(\binom {\hat {\alpha} _ {(w)}} {\hat {\beta} _ {(w)}} - \binom {\alpha_ {(w)}} {\beta_ {(w)}}\right) \Rightarrow \mathcal {N} \left(0, \sigma^ {2} \left( \begin{array}{c c} 1 & 0 \\ 0 & A ^ {- 1} \end{array} \right)\right), \tag {1.18}
$$

and that $\hat { \alpha } _ { ( 0 ) } , \hat { \alpha } _ { ( 1 ) } , \hat { \beta } _ { ( 0 ) } , \hat { \beta } _ { ( 1 ) }$ and $\overline { { X } }$ are all asymptotically independent. Then, we can write

$$
\hat {\tau} _ {I R E G} - \tau = \underbrace {\hat {\alpha} _ {(1)} - \alpha_ {(1)}} _ {\approx \mathcal {N} (0, \sigma^ {2} / n _ {1})} - \underbrace {\hat {\alpha} _ {(0)} - \alpha_ {(0)}} _ {\approx \mathcal {N} (0, \sigma^ {2} / n _ {0})} + \underbrace {\overline {{X}} \left(\beta_ {(1)} - \beta_ {(0)}\right)} _ {\approx \mathcal {N} \left(0, \left\| \beta_ {(1)} - \beta_ {(0)} \right\| _ {A} ^ {2} / n\right)}
$$

$$
+ \underbrace {\overline {{X}} \left(\hat {\beta} _ {(1)} - \hat {\beta} _ {(0)} - \beta_ {(1)} + \beta_ {(0)}\right)} _ {\mathcal {O} _ {P} (1 / n)},
$$

which leads us to the central limit theorem

$$
\sqrt {n} \left(\hat {\tau} _ {I R E G} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {I R E G}\right), \quad V _ {I R E G} = 4 \sigma^ {2} + \left\| \beta_ {(0)} - \beta_ {(1)} \right\| _ {A} ^ {2}. \tag {1.19}
$$

After the dust settles we see that, under the linear model (1.15), the interacted regression estimator also satisfies a central limit theorem, and

$$
V _ {I R E G} = V _ {D M} - \left\| \beta_ {(0)} + \beta_ {(1)} \right\| _ {A} ^ {2} \leq V _ {D M}, \tag {1.20}
$$

i.e., the regression estimator usually has a better (and never has a worse) asymptotic variance than the difference-in-means estimator.

Regression adjustments without linearity We showed above that if we assume that the data is generated following a linear model then, as expected, using an estimator that leverages linearity enables more accurate estimates of the average treatment effect than one that doesn’t. A pessimist might expect that these accuracy gains come at a cost, and that linear regression estimators should face a trade-off whereby they do worse than the difference-in-means estimator when linearity doesn’t hold. Surprisingly, however, no such tradeoff exists. In randomized trials, $\hat { \tau } _ { I R E G }$ is always consistent for $\tau$ and satisfies an asymptotic non-inferiority results of the type (1.20), even when the linear regression underlying $\hat { \tau } _ { I R E G }$ may be misspecified.

We start by establishing a general central limit theorem for $\hat { \tau } _ { I R E G }$ below under an assumption that samples are independently drawn from a population, but no linearity assumption. Throughout, we will use the following notation,

$$
\mu_ {(w)} (x) = \mathbb {E} \left[ Y _ {i} (w) \mid X _ {i} = x \right], \quad \sigma_ {(w)} ^ {2} (x) = \operatorname{Var} \left[ Y _ {i} (w) \mid X _ {i} = x \right], \tag {1.21}
$$

and assume that these quantities are well-defined and finite. The proof of the following result relies on the Huber–White analysis of linear regression whereby—regardless of linearity assumptions—linear regression consistently the best linear projection coefficients

$$
\left(\alpha_ {(w)} ^ {*}, \beta_ {(w)} ^ {*}\right) = \operatorname{argmin} _ {\alpha , \beta} \left\{\mathbb {E} \left[ (Y _ {i} (w) - \alpha - X _ {i} \cdot \beta) ^ {2} \right] \right\}, \tag {1.22}
$$

which characterize the best available linear-in- $X _ { i }$ predictor under mean-squared error.8 The argument below can also be extended to verify that standard non-parametric tools for statistical inference—such as the bootstrap or the jackknife—can be used to build asymptotically valid normal confidence intervals for $\tau$ that are centered at $\hat { \tau } _ { I R E G }$ .

Theorem 1.3. Under the conditions of Theorem 1.2, assume furthermore that $\mathbb { E } \left[ X ^ { \prime } X \right]$ is invertible. Then,

$$
\sqrt {n} \left(\hat {\tau} _ {I R E G} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {I R E G}\right),
$$

$$
V _ {I R E G} = \operatorname{Var} \left[ X _ {i} \cdot \left(\beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*}\right) \right] + \frac {1}{\pi} \mathbb {E} \left[ \left(Y _ {i} (1) - \alpha_ {(1)} ^ {*} - X _ {i} \cdot \beta_ {(1)} ^ {*}\right) ^ {2} \right] \tag {1.23}
$$

$$
+ \frac {1}{1 - \pi} \mathbb {E} \left[ \left(Y _ {i} (0) - \alpha_ {(0)} ^ {*} - X _ {i} \cdot \beta_ {(0)} ^ {*}\right) ^ {2} \right].
$$

Proof. We again assume, without loss of generality, that E $[ X _ { i } ] = 0$ . From the Huber–White analysis of linear regression, we then obtain that9

$$
\sqrt {n _ {w}} \left(\binom {\hat {\alpha} _ {(w)}} {\hat {\beta} _ {(w)}} - \binom {\alpha_ {(w)} ^ {*}} {\beta_ {(w)} ^ {*}}\right) \Rightarrow \mathcal {N} \left(0, \left( \begin{array}{c c} M S E _ {(w)} ^ {*} & 0 \\ 0 & \dots \end{array} \right)\right), \text {where} \tag {1.24}
$$

$$
M S E _ {(w)} ^ {*} = \mathbb {E} \left[ \left(Y _ {i} (w) - X _ {i} \beta_ {(w)} ^ {*} - \hat {\alpha} _ {(w)} ^ {*}\right) ^ {2} \right]
$$

measures the mean-squared error of the best linear predictor. We do not write down the lower corner of the asymptotic variance matrix as it is complicated and does not contribute to first-order behavior; however, we do note that the $\cdots . . , s ,$ term is finite whenever $\mathbb { E } \left[ X ^ { \prime } X \right]$ is invertible.

It now remains to expand out the regression estimator as given in (1.14),

$$
\hat {\tau} _ {I R E G} - \tau = \hat {\alpha} _ {(1)} - \hat {\alpha} _ {(0)} - \tau + \overline {{X}} \cdot \left(\hat {\beta} _ {(1)} - \hat {\beta} _ {(0)}\right).
$$

We start by focusing on the contribution of the first 3 summands. One can immediately verify that the average bias of the optimal linear predictions must be 0, i.e., given $\beta _ { ( w ) } ^ { * }$ , the intercept parameter must be $\alpha _ { ( w ) } ^ { * } = \mathbb { E } \ \big | Y _ { i } ( w ) - X _ { i } \cdot \beta _ { ( 1 ) } ^ { * } \big |$ . Thus, under our assumption that E $\left[ X _ { i } \right] = 0$ , we must have $\alpha _ { ( w ) } ^ { * } = \mathbb { E } \left[ Y _ { i } ( 0 ) \right]$ , and so

$$
\hat {\alpha} _ {(1)} - \hat {\alpha} _ {(0)} - \tau = \hat {\alpha} _ {(1)} - \alpha_ {(1)} ^ {*} - (\hat {\alpha} _ {(0)} - \alpha_ {(0)} ^ {*}).
$$

The central limit theorem (1.24) then implies that

$$
\sqrt {n} \left(\hat {\alpha} _ {(1)} - \hat {\alpha} _ {(0)} - \tau\right) \Rightarrow \mathcal {N} \left(0, \frac {M S E _ {(1)} ^ {*}}{\pi} + \frac {M S E _ {(0)} ^ {*}}{1 - \pi}\right). \tag {1.25}
$$

Now, moving to the last summand, we note that

$$
\overline {{X}} \cdot (\hat {\beta} _ {(1)} - \hat {\beta} _ {(0)}) = \overline {{X}} \cdot (\beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*}) + \overline {{X}} \cdot (\hat {\beta} _ {(1)} - \beta_ {(1)} ^ {*} - \hat {\beta} _ {(0)} + \beta_ {(0)} ^ {*}).
$$

Again because $\mathbb { E } \left[ X _ { i } \right] = 0$ , the average $\overline { { X } }$ of the covariates is near zero with asymptotically normal fluctuations of order $1 / { \sqrt { n } }$ , and so

$$
\sqrt {n} \overline {{X}} \cdot \left(\beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*}\right) \Rightarrow \mathcal {N} \left(0, \operatorname{Var} \left[ X _ {i} \cdot \left(\beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*}\right) \right]\right). \tag {1.26}
$$

Furthermore, one can verify that the terms in (1.25) and (1.26) are asymptotically uncorrelated and thus asymptotically independent.10

Finally, because both X and (thanks to (1.24)) $\hat { \beta } _ { ( 0 ) } - \beta _ { ( 0 ) } ^ { \ast }$ have fluctuations on the order of $1 / \sqrt { n }$ away from 0, their product can only have fluctuations of order $1 / n$ away from $0 ;$ we write this compactly as

$$
\overline {{X}} \cdot \left(\hat {\beta} _ {(1)} - \beta_ {(1)} ^ {*} - \hat {\beta} _ {(0)} + \beta_ {(0)} ^ {*}\right) = \mathcal {O} _ {P} (1 / n).
$$

Thus, by Slutsky’s lemma, this product term can be asymptotically ignored since the leading-order terms (1.25) and (1.26) are of order $1 / \sqrt { n }$ . Putting all the pieces together recovers (1.23). □

With Theorem 1.3 in hand, we are ready to revisit our comparison between $\hat { \tau } _ { I R E G }$ reduces to ${ \hat { \tau } } _ { D M }$ . Does using a regression adjustment help improve precision, even without linearity assumptions? Here, we show that the answer is yes for balanced RCTs, i.e., with $\pi = 0 . 5$ , and under an assumption that the unpredictable noise level is constant, $\sigma _ { ( 1 ) } ^ { 2 } ( x ) = \sigma _ { ( 0 ) } ^ { 2 } ( x ) = \sigma ^ { 2 }$ for all $\boldsymbol { x } . ^ { 1 1 }$ Under these assumptions, and writing Var $\left[ X _ { i } \right] = A$ as before, we can expand out the asymptotic variance from (1.23) as follows:12

$$
\begin{array}{l} V _ {I R E G} = 2 M S E _ {(0)} ^ {*} + 2 M S E _ {(1)} ^ {*} + \left\| \beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*} \right\| _ {A} ^ {2} \\ = 4 \sigma^ {2} + 2 \operatorname{Var} \left[ \mu_ {(0)} (X) - X \beta_ {(0)} ^ {*} \right] \\ + 2 \operatorname{Var} \left[ \mu_ {(1)} (X) - X \beta_ {(1)} ^ {*} \right] + \left\| \beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*} \right\| _ {A} ^ {2}. \\ \end{array}
$$

Next, because $X \beta _ { ( w ) } ^ { * }$ is the projection of $\mu _ { ( 0 ) } ( X )$ onto the span of $X$ , thisfurther simplifies

$$
\begin{array}{l} \dots = 4 \sigma^ {2} + 2 \left(\operatorname{Var} \left[ \mu_ {(0)} (X) \right] - \operatorname{Var} \left[ X \beta_ {(0)} ^ {*} \right]\right) \\ + 2 \left(\operatorname{Var} \left[ \mu_ {(1)} (X) \right] - \operatorname{Var} \left[ X \beta_ {(1)} ^ {*} \right]\right) + \left\| \beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*} \right\| _ {A} ^ {2} \\ = 4 \sigma^ {2} + 2 (\operatorname{Var} [ \mu_ {(0)} (X) ] + \operatorname{Var} [ \mu_ {(1)} (X) ]) \\ + \left\| \beta_ {(1)} ^ {*} - \beta_ {(0)} ^ {*} \right\| _ {A} ^ {2} - 2 \left\| \beta_ {(0)} ^ {*} \right\| _ {A} ^ {2} - 2 \left\| \beta_ {(1)} ^ {*} \right\| _ {A} ^ {2} \\ = 4 \sigma^ {2} + 2 \left(\operatorname{Var} \left[ \mu_ {(0)} (X) \right] + \operatorname{Var} \left[ \mu_ {(1)} (X) \right]\right) - \left\| \beta_ {(0)} ^ {*} + \beta_ {(1)} ^ {*} \right\| _ {A} ^ {2} \\ = V _ {D M} - \left\| \beta_ {(0)} ^ {*} + \beta_ {(1)} ^ {*} \right\| _ {A} ^ {2}. \\ \end{array}
$$

In other words, whether or not the true effect function $\mu _ { w } ( x )$ is linear, interacted linear regression always either reduces or matches the asymptotic variance of the difference-in-means estimator. Moreover, the amount of variance reduction scales by the amount by which linear regression in fact chooses to fit the training data. A worst case for the regression adjustment is when $\beta _ { ( 0 ) } ^ { * } = \beta _ { ( 1 ) } ^ { * } = 0$ , i.e., when OLS asymptotically just does nothing; and in this case ˆτIREG ends up being asymptotically equivalent to ${ \hat { \tau } } _ { D M }$ .

The role of regression adjustments in RCTs The individual treatment effect $\Delta _ { i } = Y _ { i } ( 1 ) - Y _ { i } ( 0 )$ is a central object of interest in causal inference. These effects $\Delta _ { i }$ themselves are fundamentally unknowable; however, a large RCT lets us consistently recover the average treatment effect $\tau = \mathbb { E } \left[ \Delta _ { i } \right]$ . In this chapter, we presented and compared two approaches for doing so: The difference-in-means estimator and the interacted regression adjustment. Perhaps surprisingly we found that, when pre-treatment covariates are available, the regression adjustment is asymptotically at least as precise as (and usually more precise than) the difference-in-means estimator—and this result holds whether or not the linear model underlying the regression adjustment is well specified.

A key point about our analysis of the regression adjustment is that we defined our target estimand, i.e., the average treatment effect $\tau = \mathbb { E } \left[ \Delta _ { i } \right]$ , before (and without) making any parametric (e.g., linear) modeling assumptions. The average treatment effect was defined in terms of non-parametric counterfactual reasoning. Linear regression was then used as an algorithmic tool to estimate $\tau ,$ but linear modeling played no role in framing our original statistical question.

Finally, note that our regression adjustment estimator can effectively beviewed as an average difference in predictions,

$$
\hat {\tau} _ {I R E G} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\underbrace {\left(\hat {\alpha} _ {(1)} + X _ {i} \hat {\beta} _ {(1)}\right)} _ {\hat {\mu} _ {(1)} (X _ {i})} - \underbrace {\left(\hat {\alpha} _ {(0)} + X _ {i} \hat {\beta} _ {(0)}\right)} _ {\hat {\mu} _ {(0)} (X _ {i})}\right), \tag {1.27}
$$

where $\hat { \mu } _ { ( w ) } ( x )$ denotes linear regression predictions at x under treatment w. Could we use other methods to estimate $\hat { \mu } _ { ( w ) } ( x )$ (e.g., deep nets, forests) rather than linear regression? How would this affect asymptotic variance? Exercise 2 in Chapter 16 digs deeper on this.

## 1.3 Bibliographic notes

The potential outcomes model for causal inference was first advocated by Neyman [1923] and Rubin [1974]; see Imbens and Rubin [2015] for a modern textbook treatment. One simple yet subtle aspect of the modeling framework used here is our use of SUTVA 1.5 which, through notation, rules out many plausible difficulties Imbens and Rubin [2015, Chapter 1.6]. SUTVA precludes any form of cross-unit interference (i.e., Wi cannot affect $Y _ { j }$ for $i \neq j )$ . Furthermore, SUTVA implicitly requires that there is only 1 “version” of treatment; and this assumption may become problematic if, e.g., we run a multi-site randomized trial where different sites administer treatment in a slightly different way. Thus, whenever invoked in an application, credibility of SUTVA should be carefully assessed.

One distinction question that has received considerable attention in the literature is whether or not one is willing to make any stochastic assumptions on the potential outcomes. The setting without stochastic assumptions on the potential outcomes is referred to as the Neyman model for randomization inference or the finite-population model; whereas the setting with stochastic assumptions is referred to the superpopulation or the IID-sampling model. Here, we stated Theorem 1.1 under the Neyman model, but otherwise worked under a superpopulation sampling model. We will take a closer look at the Neyman model—and also revisit some of the results from this chapter—in the context of our discussion of causal inference under cross-unit interference in Chapter 12.

Statistical inference justified under the Neyman model is sometimes considered the highest standard of rigor in analyzing randomized trials because all inferences are justified by randomization alone: The analyst does not need to reason about how study participants were enrolled (and whether they were randomly drawn from a larger population) in order to rigorously apply results proven under this model. The cost of working under the the Neyman model establishing the sampling distribution of even fairly simple estimators requires more intricate statistical analyses; see Li and Ding [2017] for recent results in this setting. In contrast, studying randomized trials under the superpopulation model generally enables simpler analyses via application of standard statistical and econometric tools; and paves the way for more sophisticated semiparametric estimators in observational study settings. A further discussion and comparison of the SATE (1.2) and ATE (1.3) estimands is given in [Imbens, 2004].

Lin [2013] presents a thorough discussion of the role of linear regression adjustments in improving the precision of average treatment effect estimators, and why using full intereactions as in (1.13) is often considered a best practice relative to the simple regression (1.12). When the covariates $X _ { i }$ are generated via one-hot-encoding of a discrete factor $( \mathrm { i . e . , } X _ { i } \in \{ 0 , 1 \} ^ { K }$ with only one nonzero entry per unit) the interacted regression adjustment estimator is equivalent to (post-)stratification, which is also generally considered a best practice in analyzing data from randomized experiments [Miratrix, Sekhon, and Yu, 2013].

Another feature of Lin [2013] is that he works under the Neyman model for randomization inference, and shows that many of the insights from Theorem 1.3 in fact still holds in this setting. Wager et al. [2016] have a discussion of nonparametric or high-dimensional regression adjustments in randomized trials under superpopulations asymptotics that expands on the results covered here. The study of high-dimensional regression adjustmentin the Neyman model is an ongoing effort, with recent contributions from Bloniarz et al. [2016] and Lei and Ding [2021].
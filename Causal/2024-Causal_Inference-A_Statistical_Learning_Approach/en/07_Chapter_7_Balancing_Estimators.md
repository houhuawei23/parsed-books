# Chapter 7 Balancing Estimators

The propensity score has played a central role in our presentation so far, including in understanding identification of average treatment effects under unconfoundedness, construction of efficient estimators of the average treatment effect, and the design of adaptive experiments. However, although this presentation makes it clear that the propensity score is important for causal inference, it may still remain somewhat unclear why this is true.

Here, we will re-visit the propensity score as a statistical object, and argue that a key function of the propensity score is to balance out—and thus eliminate bias captured by—observed pre-treatment confounders. This perspective will motivate the development of new propensity score estimators with better end-to-end behavior when used for treatment effect estimation, and elucidate connections between methods for average treatment effect estimation under unconfoundedness and the broader literature on non-parametric and/or highdimensional inference. Note that this chapter will not consider any new tasks in causal inference—rather, we will focus on the problem of average treatment effect estimation under unconfoundedness and revisit the statistical principles underlying the task. As such, this chapter may be skipped on a first reading.

The role of balance Working under our familiar basic unconfoundedness setting from Chapter 3, recall the (oracle) inverse-propensity weighted (IPW) estimator of the average treatment effect (ATE):

$$
\hat {\tau} _ {I P W} ^ {*} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})}\right), \quad e (x) = \mathbb {P} \left[ W _ {i} = 1 \mid X _ {i} = x \right]. \tag {7.1}
$$

In Chapter 2, we showed that the oracle IPW estimator is unbiased for the ATE, E $\left[ \hat { \tau } _ { I P W } ^ { * } \right] = \tau$ where $\tau = \mathbb { E } \left[ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \right]$ . The proof given in Theorem 2.2 was an abstract application of conditional independence and the chain rule for expectations that immediately implied unbiasedness.

In an effort to get a better understanding of the statistical function of the propensity score, we start by revisiting the unbiasedness of IPW using a less elegant—but more algorithmically explicit—argument. To this end, suppose we can write the conditional expectation functions $\mu _ { ( w ) } ( x )$ in terms of a basis expansion, i.e.,40

$$
\mu_ {(w)} (x) = \sum_ {j = 1} ^ {\infty} \beta_ {j} (w) \psi_ {j} (x) \tag {7.2}
$$

for some pre-defined set of basis function $\psi _ { j } ( \cdot )$ . Under reasonable regularity conditions (and assuming unconfoundedness), we then have

$$
\tau = m _ {(1)} - m _ {(0)}, \quad m _ {(w)} = \sum_ {j = 1} ^ {\infty} \beta_ {j} (w)   \mathbb {E} \left[ \psi_ {j} (X _ {i}) \right]. \tag {7.3}
$$

Given this setup, we can argue that IPW is unbiased as follows. Under uncondoundedness, $Y _ { i } = \mu _ { ( W _ { i } ) } ( X _ { i } ) + \varepsilon _ { i }$ with E $\left[ \varepsilon _ { i } \big | X _ { i } , W _ { i } \right] = 0$ , and so (again under regularity conditions)

$$
\mathbb {E} \left[ \frac {W _ {i} Y _ {i}}{e \left(X _ {i}\right)} \right] = \mathbb {E} \left[ \frac {W _ {i}}{e \left(X _ {i}\right)} \sum_ {j = 1} ^ {\infty} \beta_ {j} (w) \psi_ {j} \left(X _ {i}\right) \right] + \mathbb {E} \left[ \frac {W _ {i} \varepsilon_ {i}}{e \left(X _ {i}\right)} \right] \tag {7.4}
$$

$$
= \sum_ {j = 1} ^ {\infty} \beta_ {j} (w) \mathbb {E} \left[ \frac {W _ {i} \psi_ {j} (X _ {i})}{e (X _ {i})} \right] = \sum_ {j = 1} ^ {\infty} \beta_ {j} (w) \mathbb {E} \left[ \psi_ {j} (X _ {i}) \right] = m _ {(1)},
$$

and similarly E $[ ( 1 - W _ { i } ) Y _ { i } / ( 1 - e ( X _ { i } ) ) ] = m _ { ( 0 ) }$ . This argument reveals that IPW works by re-weighting both the treated and control samples so that the weighted average of the basis functions $\psi _ { j } ( X _ { i } )$ exactly matches the relevant population averages.

Population vs. sample balance Oracle IPW achieves unbiasedness by creating population balance across the treated and control groups for all basis functions $\psi _ { j } ( X _ { i } )$ :

$$
\mathbb {E} \left[ \frac {W _ {i}   \psi_ {j} (X _ {i})}{e (X _ {i})} \right] = \mathbb {E} \left[ \psi_ {j} (X _ {i}) \right], \quad \mathbb {E} \left[ \frac {(1 - W _ {i})   \psi_ {j} (X _ {i})}{1 - e (X _ {i})} \right] = \mathbb {E} \left[ \psi_ {j} (X _ {i}) \right]. \tag {7.5}
$$

In practice, we need to work with finite samples and need to estimate propensity scores. However, following (7.5), if the sample size n is large enough and the propensity score estimates $\hat { e } ( X _ { i } )$ are accurate enough, then we may hope to achieve approximate sample balance,

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i} \psi_ {j} (X _ {i})}{\hat {e} (X _ {i})} \approx \frac {1}{n} \sum_ {i = 1} ^ {n} \psi_ {j} (X _ {i}), \tag {7.6}
$$

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - W _ {i}) \psi_ {j} (X _ {i})}{1 - \hat {e} (X _ {i})} \approx \frac {1}{n} \sum_ {i = 1} ^ {n} \psi_ {j} (X _ {i}),
$$

and for such sample balance in turn to imply consistency of IPW. This class of arguments can be used to show that IPW is consistent for a wide variety to consistent propensity score estimates $\hat { e } ( X _ { i } )$ .

The above argument is, however, incredibly loose. On the one hand, we claim that IPW achieves consistency by creating balance in the $\psi _ { j } ( X _ { i } )$ ; but on the other hand, the above argument lets sample balance (7.6) emerge indirectly as a consequence of consistent propensity score estimation. If we believe that good sample balance is important, shouldn’t we put more thought into how we estimate propensity scores and optimize for sample balance as in $( 7 . 6 ) ?$ The answer to this question is affirmative; and the covariate-balancing propensity score methods that emerge from seeking to answer it provide a major improvement over basic IPW methods that do not consider balance.

## 7.1 Covariate-balancing propensity scores

We start by considering propensity score methods tailored to target covariate balance under a finite-dimensional parametric specification. Suppose that $X _ { i } \in$ $\mathbb { R } ^ { p }$ take values in a finite-dimensional space, and that we have a linear outcome model $\mu _ { ( w ) } ( x ) = x \cdot \beta ( w )$ and a logistic propensity model $e ( x ) = 1 / ( 1 + e ^ { - x \cdot \theta } )$ . Because we have a linear outcome model, achieving sample balance just involves balancing the raw covariates $X _ { i }$ .

The sample balance condition (7.6) involves the “≈” relation that we need to disambiguate in order to proceed. Here, given that we’re in a low-dimensional setting, it’s reasonable to ask for exact balance, i.e., for (7.6) to hold with equality. Then, using our logistic specification $\hat { e } ( x ) = 1 / ( 1 + e ^ { - x \cdot \hat { \theta } } )$ , (7.6) becomes:

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 + e ^ {- X _ {i} \hat {\theta}}\right) W _ {i} X _ {i} = \frac {1}{n} \sum_ {i = 1} ^ {n} X _ {i}, \tag {7.7}
$$

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 + e ^ {X _ {i} \hat {\theta}}\right) (1 - W _ {i}) X _ {i} = \frac {1}{n} \sum_ {i = 1} ^ {n} X _ {i}. \tag {7.8}
$$

Can we learn a parameter vector $\hat { \theta }$ for the propensity model such that the balance conditions (7.7) and (7.8) hold?

These balance conditions are non-linear systems of equations that may at first glance seem challenging to solve. However, it turns out that—under nondegeneracy conditions—the solution to (7.7) can equivalently be written as the optimum of the following convex minimization problem,

$$
\hat {\theta} = \operatorname{argmin} _ {\theta} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \ell_ {\theta} ^ {(1)} \left(X _ {i}, Y _ {i}, W _ {i}\right) \right\}, \tag {7.9}
$$

$$
\ell_ {\theta} ^ {(1)} (X _ {i}, Y _ {i}, W _ {i}) = W _ {i} e ^ {- X _ {i} \theta} + (1 - W _ {i}) X _ {i} \theta ,
$$

so it can readily be solved via numerical methods such as Newton descent. Meanwhile, the solution to (7.8) is equivalent to

$$
\hat {\theta} = \operatorname{argmin} _ {\theta} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \ell_ {\theta} ^ {(0)} \left(X _ {i}, Y _ {i}, W _ {i}\right) \right\}, \tag {7.10}
$$

$$
\ell_ {\theta} ^ {(0)} (X _ {i}, Y _ {i}, W _ {i}) = (1 - W _ {i}) e ^ {X _ {i} \theta} - W _ {i} X _ {i} \theta .
$$

Now, one subtlety here is that we may be interested in a parameter vector $\hat { \theta }$ that solves both (7.7) and (7.8) simultaneously. This, however, is not in general possible (because it would require solving $2 p$ equation using p free parameters), but neither is it necessary: If the role of the propensity model is simply to create balance, then if it’s convenient there’s no strong reason not to use two different propensity models in the context of a single ATE estimator.

Putting all these pieces together to create an IPW estimator of the ATE results in a covariate-balancing propensity score (CBPS) estimator:

$$
\hat {\theta} _ {(w)} = \operatorname{argmin} _ {\theta} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \ell_ {\theta} ^ {(w)} \left(X _ {i}, Y _ {i}, W _ {i}\right) \right\}, \quad \text { for } \quad w = 0, 1 \tag {7.11}
$$

$$
\hat {\tau} _ {C B P S} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 + e ^ {- X _ {i} \hat {\theta} _ {(1)}}\right) W _ {i} Y _ {i} - \frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 + e ^ {X _ {i} \hat {\theta} _ {(0)}}\right) (1 - W _ {i}) Y _ {i}.
$$

The following result shows that, unlike the oracle IPW estimator which is unbiased but with unnecessarily large variance (Theorem 2.2) or generic IPW with estimated propensity scores which is consistent but doesn’t necessarily have a good rate of convergence, the above CBPS estimator has excellent statistical properties: It is ${ \sqrt { n } } { \mathrm { - c o n s i s t e n t } }$ with and asymptotically normal sampling distribution, and achieves the same asymptotic variance as the AIPW estimator studied in Chapter 3.

Theorem 7.1. We have samples $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , W _ { i } \} \stackrel { i i d } { \sim } P$ taking values in $\mathbb { R } ^ { p } \times \mathbb { R } \times \mathbb { R } \times \{ 0 , 1 \}$ such that we get to observe $( X _ { i } , Y _ { i } , W _ { i } )$ where $Y _ { i } = Y _ { i } ( W _ { i } )$ , and that unconfoundedness holds, $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \} \perp W _ { i } \mid X _ { i }$ . Suppose there is a $c > 0$ for which the following exponential moments are finite,41

$$
\mathbb {E} \left[ \frac {e ^ {c \| X _ {i} \| _ {2}}}{e (X _ {i})} \right] <   \infty , \quad \mathbb {E} \left[ \frac {e ^ {c \| X _ {i} \| _ {2}}}{1 - e (X _ {i})} \right] <   \infty , \tag {7.12}
$$

and that the feature covariance matrix has full rank, E $\left\lceil X _ { i } ^ { \otimes 2 } \right\rceil \succ 0$ . Suppose furthermore that both the linear outcome model $\bar { \mu _ { ( w ) } ( x ) \bar { \ } } = \bar { x \ } \cdot \beta ( w )$ and the logistic propensity model $e ( x ) = 1 / ( 1 + e ^ { - x \cdot \theta } )$ are well specified with $\left\| \theta \right\| _ { 2 } < \infty$ , and that the conditional variances $\sigma _ { w } ^ { 2 } ( x ) = \operatorname { V a r } \left[ Y _ { i } ( w ) \big | X _ { i } = x \right]$ are uniformly bounded, $\sigma _ { w } ^ { 2 } ( x ) \le M$ . Then $\hat { \tau } _ { C B P S }$ is consistent and

$$
\sqrt {n} \left(\hat {\tau} _ {C B P S} - \tau\right) \Rightarrow \mathcal {N} \left(0, \operatorname{Var} \left[ \tau (X _ {i}) \right] + \mathbb {E} \left[ \frac {\sigma_ {0} ^ {2} (X _ {i})}{1 - e (X _ {i})} + \frac {\sigma_ {1} ^ {2} (X _ {i})}{e (X _ {i})} \right]\right). (7. 1 3)
$$

Proof. We start by examining the loss functions $\ell _ { \theta } ^ { ( 1 ) } ( x , y , w )$ given above, and its expectation

$$
L _ {(1)} (\theta) = \mathbb {E} \left[ \ell_ {\theta} ^ {(1)} \left(X _ {i}, Y _ {i}, W _ {i}\right)) \right].
$$

The analysis of $\ell _ { \theta } ^ { ( 0 ) } ( x , y , w )$ and $L _ { ( 0 ) } ( \cdot )$ is essentially identical, and so we do not carry it out here. First, note that

$$
\nabla^ {2} \ell_ {\theta} ^ {(1)} (x, y, w) = w e ^ {- \theta \cdot x} x ^ {\otimes 2} \succeq 0,
$$

i.e., this loss functions are convex as claimed. Next, assuming that the logistic propensity model is well specified (with true parameter value $\theta )$ , we see that for any $\theta ^ { \prime }$

$$
L _ {(1)} (\theta^ {\prime}) = \mathbb {E} \left[ \frac {e ^ {- X _ {i} \theta}}{1 + e ^ {- X _ {i} \theta}} e ^ {X _ {i} (\theta - \theta^ {\prime})} + \frac {1}{1 + e ^ {X _ {i} \theta}} X _ {i} \theta^ {\prime} \right],
$$

which, because $\mathbb { E } \left[ e ^ { c \| x \| _ { 2 } } \right] < \infty$ thanks to (7.12), is finite for any $\theta ^ { \prime }$ such that $\begin{array} { r } { \| \theta - \theta ^ { \prime } \| _ { 2 } \leq c } \end{array}$ . Finally, at the true parameter value $\theta , ^ { 4 2 }$

$$
\nabla L _ {(1)} (\theta) = 0, \quad \nabla^ {2} L _ {(1)} (\theta) = \mathbb {E} \left[ e (X _ {i}) X _ {i} ^ {\otimes 2} \right] \succ 0,
$$

$\mathrm { i . e . , ~ } \theta$ is in fact a minimizer of $L _ { ( 1 ) } ( \cdot ) ;$ ; and, by convexity of ${ \cal L } _ { \theta } ^ { ( 1 ) }$ and strong convexity at $\theta ,$ it is the unique minimizer $L _ { ( 1 ) } ( \cdot )$ .

Given these preliminaries, we can use standard results for convex empirical risk minimization [e.g., Van der Vaart, 1998, Theorem 5.7 and Example 19.8] to check that $\hat { \theta } _ { ( 1 ) }$ is consistent, i.e., $\hat { \theta } _ { ( 1 ) } \to _ { p } \theta$ . Thus, in particular, we see that $\hat { \theta } _ { ( 1 ) }$ must be finite with probability going to 1. It must thus (with probability going to 1) be a critical point of the loss function,

$$
\nabla \left(\frac {1}{n} \sum_ {i = 1} ^ {n} W _ {i} e ^ {- X _ {i} \hat {\theta} _ {(1)}} + (1 - W _ {i}) X _ {i} \hat {\theta} _ {(1)}\right) = 0,
$$

which in turn is equivalent to $\hat { \theta } _ { ( 1 ) }$ solving (7.7).

Applying an analogous analysis to $\ddot { \theta } _ { ( 0 ) }$ and plugging these balance conditions into (7.11), we can use well-specification of the linear outcome model to verify that on the with-probability-tending-to-1 event where $\hat { \theta } _ { ( 1 ) }$ solves (7.7) and $\hat { \theta } _ { ( 0 ) }$ solves (7.8),

$$
\begin{array}{l} \hat {\tau} _ {C B P S} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(X _ {i} \left(\beta_ {(1)} - \beta_ {(0)}\right) + (2 W _ {i} - 1) \left(1 + e ^ {- (2 W _ {i} - 1) X _ {i} \hat {\theta} _ {(W _ {i})}}\right) \varepsilon_ {i}\right), \\ = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\tau (X _ {i}) + \frac {W _ {i}}{e (X _ {i})} \varepsilon_ {i} - \frac {1 - W _ {i}}{1 - e (X _ {i})} \varepsilon_ {i}\right) \\ + \frac {1}{n} \sum_ {i = 1} ^ {n} \left(e ^ {- X _ {i} \hat {\theta} _ {(1)}} - e ^ {- X _ {i} \theta}\right) W _ {i} \varepsilon_ {i} - \frac {1}{n} \sum_ {i = 1} ^ {n} \left(e ^ {X _ {i} \hat {\theta} _ {(0)}} - e ^ {X _ {i} \theta}\right) (1 - W _ {i}) \varepsilon_ {i}, \\ \end{array}
$$

where $\varepsilon _ { i } = Y _ { i } - X _ { i } \beta _ { ( W _ { i } ) }$ . Now, the first summand above is familiar from our earlier discussions (e.g., in Chapter 2), and satisfies (7.13).

It remains to check that the last two terms are asymptotically negligible on the $1 / \sqrt { n }$ scale. To this end, note that this term is mean-zero conditionally on $\{ X _ { i } , W _ { i } \}$ (and thus also the $\hat { \theta } _ { ( w ) } )$ , and that

$$
\begin{array}{l} n \mathbb {E} \left[ \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \left(e ^ {- X _ {i} \hat {\theta} _ {(1)}} - e ^ {- X _ {i} \theta}\right) W _ {i} \varepsilon_ {i}\right) ^ {2} | \{X _ {i}, W _ {i} \} \right] \\ = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(e ^ {- X _ {i} \hat {\theta} _ {(1)}} - e ^ {- X _ {i} \theta}\right) ^ {2} W _ {i} \sigma_ {1} ^ {2} (X _ {i}) \\ \leq \frac {M}{n} \sum_ {i = 1} ^ {n} \left(e ^ {- X _ {i} \hat {\theta} _ {(1)}} - e ^ {- X _ {i} \theta}\right) ^ {2} W _ {i} \\ = \frac {M}{n} \sum_ {i = 1} ^ {n} \left(e ^ {X _ {i} (\theta - \hat {\theta} _ {(1)})} - 1\right) ^ {2} e ^ {- 2 X _ {i} \theta} W _ {i}. \\ \end{array}
$$

We know that, by consistency, $\lVert \theta - \hat { \theta } _ { ( 1 ) } \rVert _ { 2 } \leq \delta / 2$ with probability tending to 1 for any $\delta > 0$ , and so, again with probability tending to 1, the above expression is bounded by

$$
\begin{array}{l} \dots \leq \frac {2 M}{n} \sum_ {i = 1} ^ {n} \left(e ^ {\delta \| X _ {i} \| _ {2}} + 1\right) e ^ {- 2 X _ {i} \theta} W _ {i} \\ = \mathcal {O} _ {P} \left(\mathbb {E} \left[ \left(e ^ {\delta \| X _ {i} \| _ {2}} + 1\right) e ^ {- 2 X _ {i} \theta} / \left(1 + e ^ {- X _ {i} \theta}\right) \right]\right) \\ = \mathcal {O} _ {P} \left(\mathbb {E} \left[ e ^ {\delta \| X _ {i} \| _ {2}} \left(1 + e ^ {- X _ {i} \theta}\right) \right]\right), \\ \end{array}
$$

where the steps above were by Markov’s inequality on the 2nd line and by direct algebraic manipulations on the 3rd line. This expression is finite for any $\delta \leq c$ by (7.12); and tends to 0 as $\delta  0$ by continuity. Thus, by consistency of $\ddot { \theta } _ { ( 1 ) }$ ,

$$
n \mathbb {E} \left[\left(\frac {1}{n} \sum_ {i = 1} ^ {n} \left(e ^ {- X _ {i} \hat {\theta} _ {(1)}} - e ^ {- X _ {i} \theta}\right) W _ {i} \varepsilon_ {i}\right) ^ {2} | \{X _ {i}, W _ {i} \} \right]\rightarrow_ {p} 0,
$$

and so by Chebyshev’s inequality this term is on the $1 / \sqrt { n }$ scale as we sought to show. Applying an analogous argument to the term involving $\hat { \theta } _ { ( 0 ) }$ completes the proof. □

Thus, if we believe in a linear-logistic specification and want to use an IPW estimator, then we should learn the propensity model by minimizing the covariate-balancing loss function rather than by the usual maximum likelihood loss used for logistic regression. Maximum likelihood is asymptotically optimal from the perspective of estimating the logistic regression parameters θ, but that’s not what matters here. When estimating the ATE via IPW, what we need from the inverse-propensity weights is for them to create balance as in (7.6); and we achieve good results with IPW when using covariate-balancing propensity scores that directly target this goal.

Exercise 8 in Chapter 16 expands on the result given above, and also establishes double-robustness properties for $\hat { \tau } _ { C B P S }$ that hold if only one of the linear or logistic models is well specified. Exercise 9 studies a covariate-balancing propensity score estimator that targets the average treatment effect on the treated.

Remark 7.1. The estimator (7.11) is not the first covariate-balancing propensity score estimator encountered in this book. In Chapter 2, we considered a setting where the feature space X is discrete, and found that the natural stratified estimator $\scriptstyle { \hat { \tau } } _ { S T R A T }$ could be interpreted as an IPW-estimator with a smart choice of estimated propensities that enable efficient large sample behavior; see Theorem 2.1 and (2.17). Further examination reveals that the propensity scores underlying $\scriptstyle { \hat { \tau } } _ { S T R A T }$ achieve exact sample balance for indicators 1 $( \{ X _ { i } = x \} )$ for all $x \in \mathcal { X }$ , and that $\scriptstyle { \hat { \tau } } _ { S T R A T }$ is equivalent to $\hat { \tau } _ { C B P S }$ for a saturated model. Thus, conceptually, we can think of covariate-balancing propensity score methods as the natural generalization of stratified treatment effect estimation for when X takes on continuous values.

## 7.2 Approximate balance and augmented estimators

We established above that, when working in a low-dimensional parametric setting, propensity score methods that target exact finite-sample balance as in (7.7) and (7.8) have a number of good statistical properties. In some settings, however, achieving exact balance may not be realistic. In some modern applications, the covariates $X _ { i } ~ \in ~ \mathbb { R } ^ { p }$ may take values in a high-dimensional space with $p \gg n \ ( \mathrm { e . g . , } \ X _ { i }$ may represent a patient’s genome); and in this case it’s generally not possible to find weights on n samples that exactly solve p covariate-balancing moment conditions. Or, as in our motivating example 7.2, we may be interested in a setting where we use an infinite sieve to approximate a non-parametric function, and in this case we have infinitely many covariate-balancing moment conditions to worry about.

Thankfully, even when exact balance is unachievable, we can still obtain good results via propensity-score methods that aim for approximate balance

$$
\sup _ {j = 1, 2, \dots} \left| \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i} \psi_ {j} (X _ {i})}{\hat {e} (X _ {i})} - \psi_ {j} (X _ {i}) \right| \leq t, \tag {7.14}
$$

$$
\sup _ {j = 1, 2, \dots} \left| \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - W _ {i})   \psi_ {j} (X _ {i})}{1 - \hat {e} (X _ {i})} - \psi_ {j} (X _ {i}) \right| \leq t,
$$

for some small tolerance parameter t. When working with approximate balance, plain IPW-type estimator as considered above may dominated by bias and no longer work well; however, using augmented IPW-type estimators can address the issue. The reason augmented estimators help with approximate balance is closely tied to the (strong) double robustness of augmented IPW discussed in Chapter 3: A reasonably accurate regression adjustment can mitigate the bias due to non-exact balance without introducing excess errors in doing so.

A comprehensive review of approximately balancing methods for highdimensional and/or non-parametric treatment effect estimation problems is beyond the scope of this presentation. Instead, we will here summarize one approach tailored to the high-dimensional setting with a sparse, linear outcome model, and present references for further reading at the end of the chapter.

Suppose that the basic unconfoundedness model from Chapter 3 holds with high-dimensional controls $X _ { i } \in \mathbb { R } ^ { p }$ , where $p$ may be much larger than n. Suppose furthermore that the outcome model is sparse and linear, $\mu _ { ( w ) } ( x ) = x \cdot \beta _ { ( w ) }$ with $\| \beta _ { ( w ) } \| _ { 0 } \leq k$ for some reasonably small bound on the number of non-zero parameters $k$ , where $\lVert \boldsymbol { v } \rVert _ { 0 }$ counts the number of non-zero entries in 0. Note that we are not making any parametric assumptions on the propensity model here, and simply assume strong overlap $\eta \leq e ( X _ { i } ) \leq 1 - \eta$ .

Given this setup, Athey, Imbens, and Wager [2018b] consider learning weights $\hat { \gamma } _ { i }$ by directly minimizing an approximate balance criterion:

$$
\hat {\gamma} ^ {(1)} = \operatorname{argmin} _ {\substack {\gamma_ {i} \geq 0, t \geq 0 \\ | 1, n}} \frac {1}{n} \sum_ {W _ {i} = 1} \gamma_ {i} ^ {2} + \zeta n t ^ {2} \tag{7.15}
$$

${ \mathrm { s u b j e c t ~ t o } } \left| { \frac { 1 } { n } } \sum _ { i = 1 } \left( \gamma _ { i } W _ { i } - 1 \right) X _ { i } \right| \leq t { \mathrm { ~ f o r ~ a l l ~ } } j = 1 , \ldots , p ,$

and $\hat { \gamma } _ { ( 0 ) }$ is derived analogously. Conceptually, we can interpret these weights ${ } ^ { \mathfrak { a } } 1 / \hat { e } ( X _ { i } ) = \hat { \gamma } _ { i } ^ { ( 1 ) , }$ ric propensity model. We can then use these approximate balancing weights to derive an augmented balancing estimator modeled after the AIPW construction,

$$
\hat {\tau} _ {A B} = \frac {1}{n} \sum_ {i = 1} ^ {n} X _ {i} \left(\hat {\beta} _ {(1)} - \hat {\beta} _ {(0)}\right) + W _ {i} \hat {\gamma} _ {i} ^ {(1)} \left(Y _ {i} - X _ {i} \hat {\beta} _ {(1)}\right) \tag {7.16}
$$

$$
- (1 - W _ {i}) \hat {\gamma} _ {i} ^ {(0)} \left(Y _ {i} - X _ {i} \hat {\beta} _ {(0)}\right),
$$

where the $\hat { \beta } _ { ( w ) }$ are estimated via some method applicable to sparse, highdimensional data such as the lasso [Tibshirani, 1996]. The key motivation behind this construction is the following lemma.

Lemma 7.2. Under unconfoundedness and SUTVA, suppose furthermore that $\mu _ { ( w ) } ( x ) = x \cdot \beta _ { ( w ) }$ , and that $\hat { \beta } _ { ( w ) }$ is an estimator of $\beta _ { ( w ) }$ with $L _ { 1 } – n o r m$ estimation error bounded by $C _ { ( w ) } ~ f o r ~ w = 0 , 1$ :

$$
\left\| \hat {\beta} _ {(w)} - \beta_ {(w)} \right\| _ {1} \leq C _ {(w)}, \quad \| v \| _ {1} = \sum_ {j = 1} ^ {p} | v _ {j} |. \tag {7.17}
$$

Then, the augmented balancing estimator (7.16) satisfies

$$
\hat {\tau} _ {A B} = \frac {1}{n} \sum_ {i = 1} ^ {n} X _ {i} \left(\beta_ {(1)} - \beta_ {(0)}\right) + W _ {i} \hat {\gamma} _ {i} ^ {(1)} \varepsilon_ {i} - (1 - W _ {i}) \hat {\gamma} _ {i} ^ {(0)} \varepsilon_ {i} + E, \tag {7.18}
$$

$$
| E | \leq C _ {(0)} \hat {t} ^ {(0)} + C _ {(1)} \hat {t} ^ {(1)},
$$

where $t h e \hat { t } ^ { ( w ) }$ are the bias parameters in the solution to the optimization problem (7.15) and $\varepsilon _ { i } = Y _ { i } - X _ { i } \beta _ { ( W _ { i } ) }$ .

Proof. Thanks to linearity of $\mu _ { ( w ) } ( x )$ , we immediately get that the first line of (7.18) holds with error term

$$
\begin{array}{l} E = \frac {1}{n} \sum_ {i = 1} ^ {n} X _ {i} \left(\hat {\beta} _ {(1)} - \hat {\beta} _ {(0)}\right) - X _ {i} \left(\beta_ {(1)} - \beta_ {(0)}\right) \\ + W _ {i} \hat {\gamma} _ {i} ^ {(1)} X _ {i} (\beta_ {(1)} - \hat {\beta} _ {(1)}) - (1 - W _ {i}) \hat {\gamma} _ {i} ^ {(0)} X _ {i} (\beta_ {(0)} - \hat {\beta} _ {(0)}) \\ = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 - W _ {i} \hat {\gamma} _ {i} ^ {(1)}\right) X _ {i} \left(\hat {\beta} _ {(1)} - \beta_ {(1)}\right) \\ - \frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 - (1 - W _ {i}) \hat {\gamma} _ {i} ^ {(0)}\right) X _ {i} \left(\hat {\beta} _ {(0)} - \beta_ {(0)}\right) \\ \end{array}
$$

An application of H¨older’s inequality then gives

$$
\begin{array}{l} | E | \leq \left\| \frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 - W _ {i} \hat {\gamma} _ {i} ^ {(1)}\right) X _ {i} \right\| _ {\infty} \left\| \hat {\beta} _ {(1)} - \beta_ {(1)} \right\| _ {1} \\ + \left\| \frac {1}{n} \sum_ {i = 1} ^ {n} \left(1 - (1 - W _ {i}) \hat {\gamma} _ {i} ^ {(0)}\right) X _ {i} \right\| _ {\infty} \left\| \hat {\beta} _ {(0)} - \beta_ {(0)} \right\| _ {1}, \\ \end{array}
$$

which is equivalent to the bound we seek to show.

The upshot is that, ignoring the error term E, the expression for $\hat { \tau } _ { A B }$ given in (7.18) has the familiar form obtained with efficient estimators of the ATE in Chapter 3. Thus, if we can show that E is negligible on the $1 / \sqrt { n } -$ -scale, this result strongly suggests that we should expect good statistical behavior from $\hat { \tau } _ { A B }$ . One wrinkle that’s beyond the scope of this presentation is to provide a precise characterization of what the $\hat { \gamma } ^ { ( w ) }$ converge $\mathrm { t o } ; ^ { 4 3 }$ however, one simple observation is that if we can control the average second moment of the $\hat { \gamma } ^ { ( w ) }$ (as will be done below), then (7.18) together with an error bound $| E | \ll 1 / \sqrt { n }$ implies that $\hat { \tau } _ { A B }$ is ${ \sqrt { n } } .$ -consistent and asymptotically unbiased.

It now remains to establish conditions under which $E$ is bounded. Under a widely used assumption on the covariate distribution called the “restricted eigenvalue condition” and under a sparsity bound $\| \beta _ { ( w ) } \| _ { 0 } \leq k \ ( \mathrm { i . e }$ ., and assumption that the true parameter vector has at most k non-zero entries), the lasso can achieve 1-norm error [e.g., Negahban et al., 2012]

$$
\left\| \hat {\beta} _ {(w)} - \beta_ {(w)} \right\| _ {1} = \mathcal {O} _ {P} \left(k \sqrt {\frac {\log (p)}{n}}\right). \tag {7.19}
$$

Meanwhile, the imbalance of approximate balancing weights can be controlled via the following result.

Lemma 7.3. Suppose that strong overlap holds, $\eta \leq e ( X _ { i } ) \leq 1 - \eta$ for some $\eta > 0$ , that the features $X _ { i }$ are bounded $| X _ { i } | \le M$ . Then, with probability at least $1 - \delta$ , the solution to the approximate balancing program (7.15) with tuning parameter $\zeta = 1 / ( 4 \log ( p ) )$ has a solution satisfying

$$
\frac {1}{n} \sum_ {W _ {i} = 1} \left(\hat {\gamma} _ {i} ^ {(1)}\right) ^ {2} = \mathcal {O} _ {P} (1), \quad \hat {t} ^ {(1)} = \mathcal {O} _ {P} \left(\sqrt {\frac {\log (p)}{n}}\right). \tag {7.20}
$$

Proof. Consider the value of the objective function in (7.15) if we were to plug-in the true propensity scores $\gamma _ { i } ^ { * } = 1 / e ( X _ { i } )$ . This choice would induce a worst-case imbalance

$$
t ^ {*} = \left\| \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i}}{e (X _ {i})} - 1\right) X _ {i} \right\| _ {\infty}.
$$

Now, for every $j = 1 , \ldots , p ,$ we have E $[ ( W _ { i } / e ( X _ { i } ) - 1 ) X _ { i j } ] = 0$ and, thanks to strong overlap and boundedness, we have $| ( W _ { i } / e ( X _ { i } ) - 1 ) X _ { i j } | \le M / \eta$ . Thus, we can use Hoeffding’s inequality and a union bound to verify that,

$$
\mathbb {P} \left[ | t ^ {*} | \geq \frac {M}{\eta} \sqrt {\frac {4 \log (p)}{n}} \right] \leq \frac {2}{p}.
$$

A second application of Hoeffding’s inequality to the first part of the objective and plugging in our choice for $\zeta$ then shows that,

$$
\mathbb {P} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i}}{e ^ {2} (X _ {i})} + n \zeta (t ^ {*}) ^ {2} \geq \mathbb {E} \left[ \frac {1}{e (X _ {i})} \right] + \frac {1}{\eta^ {2}} \sqrt {\frac {2 \log (p)}{n}} + \frac {M ^ {2}}{\eta^ {2}} \right] \leq \frac {4}{p}.
$$

Now, the true inverse-propensity scores $\gamma _ { i } ^ { * }$ are simply one feasible solution to the optimization problem (7.15), whereas $\hat { \gamma } ^ { ( 1 ) }$ was chosen such as to make the optimization objective as small as possible. Thus, by monotonicity, we must also have

$$
\mathbb {P} \left[ \frac {1}{n} \sum_ {W _ {i} = 1} \left(\hat {\gamma} _ {i} ^ {(1)}\right) ^ {2} + n \zeta \left(\hat {t} ^ {(1)}\right) ^ {2} \geq \mathbb {E} \left[ \frac {1}{e (X _ {i})} \right] + \frac {1}{\eta^ {2}} \sqrt {\frac {2 \log (p)}{n}} + \frac {M ^ {2}}{\eta^ {2}} \right] \leq \frac {4}{p}.
$$

The desired conclusion follows by noting that all terms in the objective are non-negative, and so must also be individually controlled by the given upper bound. □

Putting together the pieces, we can use (7.19) and (7.20) to show that, under a sparsity bound $\| \beta _ { ( w ) } \| _ { 0 } \leq k$ , the error term E in Lemma 7.2 is bounded to order $| E | = \mathcal { O } _ { P } \left( k \log ( p ) / n \right)$ . It is thus negligible on the $1 / { \sqrt { n } } .$ -scale whenever the sparsity level is controlled as $k \ll \sqrt { n } / \log ( p )$ . This sparsity condition is familiar from the literature on high-dimensional inference [Javanmard and Montanari, 2014, Zhang and Zhang, 2014], and corresponds to the weakest sparsity condition under which debiased lasso methods enable valid inference without further assumptions knowledge about the distribution of the covariates $X _ { i }$ . This connection is not an accident, and the augmented balancing method presented here is in fact closely connected to debiased lasso methods for high dimensional inference; see Athey, Imbens, and Wager [2018b] for a discussion and further references.

Remark 7.2. We earlier made a claim that, when we have weights that achieve approximate (but not exact) balance, augmented estimators of the form (7.16) should be used. We are now in a position to substantiate this claim: Suppose that we are in a high-dimensional setting and use weights (7.15) to form an IPW-type estimator

$$
\hat {\tau} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(W _ {i} \hat {\gamma} _ {i} ^ {(1)} Y _ {i} - (1 - W _ {i}) \hat {\gamma} _ {i} ^ {(0)} Y _ {i}\right). \tag {7.21}
$$

We can then use Lemma 7.3 to control the bias of this estimator; however, the resulting bias bound will generally be of order $\sqrt { \log ( p ) / n }$ , and this bound dominates the error of the estimator when $p$ can grow with $n .$ . Thus, our analysis only yields ${ \sqrt { n } } .$ -consistency in high dimensions when approximately balancing weights are used in an augmented estimator.

Remark 7.3. In comparing different methods discussed in this chapter, one natural question to ask is: What happens if we apply the direct balance-seeking strategy (7.15) in a low-dimensional setting, and target exact rather than approximate balance? This results in treated weights

$$
\hat {\gamma} ^ {(1)} = \operatorname{argmin} _ {\gamma_ {i} \geq 0} \left\{\frac {1}{n} \sum_ {W _ {i} = 1} \gamma_ {i} ^ {2}: \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\gamma_ {i} W _ {i} - 1\right) X _ {i} = 0 \right\}, \tag {7.22}
$$

and analogous control weights; note that this optimization problem will generally only be feasible when both the number of treated units and the number of control units is greater than p. If we have exact balance, then using an augmented form as in (7.16) is no longer necessary; in fact, exact balance means that the regression adjustment term gets exactly canceled out and so the augmented estimator is numerically equal to a non-augmented one [Robins et al., 2007].44

## 7.3 Bibliographic notes

The key role of covariate balance for average treatment effect estimation under unconfoundedness has long been recognized, and a standard operation procedure when working with any weighted or matching-type estimators is to use balance as a goodness of fit check [Imbens and Rubin, 2015]. For example, after fitting a propensity model by logistic regression, one could check that the induced propensity weights satisfy a sample balance condition of the type (7.6) with reasonable accuracy. If the balance condition is not satisfied, one could try fitting a different (better) propensity model.

The idea of using covariate balance as an idea to guide propensity estimation (rather than simply as a post-hoc sanity check) is more recent. Early proposals from different communities include Graham, Pinto, and Egel [2012] Hainmueller [2012] and Imai and Ratkovic [2014]; a unifying perspective on these methods via covariate-balancing loss functions is provided by Zhao [2019]. Zubizarreta [2015] proposed learning weights that achieve balance without going via an explicit application of IPW in the context of a parametric propensity model. Iacus, King, and Porro [2012] proposed coarsening a continuous covariate space into a finite number of regions, and then applying a stratified estimator over these regions to achieve balance.45 The term “covariate-balancing propensity score” was coined by Imai and Ratkovic [2014], while our presentation given in Chapter 7.1 most closely builds on Graham, Pinto, and Egel [2012] and Zhao [2019].

Our presentation in Chapter 7.2 was adapted from Athey, Imbens, and Wager [2018b], who showed that approximately balancing weights and augmented estimators can be used for inference about average treatment effects with high-dimensional controls under a sparse, linear outcome model. Tan [2020] pairs an augmented construction with a lasso-penalized variant of the covariate-balancing propensity score estimator (7.10) to estimate average treatment effects in a high-dimensional linear-logistic specification. Kallus [2020] and Hirshberg and Wager [2021] consider balancing (and augmented balancing) methods in a non-parametric setting, and derive weights that approximately balance all functions in an infinite-dimensional space $( \mathrm { e . g . }$ , all functions in a given smoothness class). In particular, Hirshberg and Wager [2021] show that if the class of balanced functions is not too large and spans the true inverse-propensity weightin functions $1 / e ( \cdot )$ and $1 / ( 1 + e ( \cdot ) )$ , then augmented approximately balancing estimators of the average treatment effect are efficient in the sense of Chapter 3.2 under weak conditions.

Finally, the principles behind balanced estimation apply more broadly than to average treatment effect estimation, and can in fact be used to estimate a wide class of econometric targets. The Riesz representer theorem gives conditions under which estimands θ that depend linearly on the sampling distribution—this includes quantities such as average derivatives and average partial effects—can be characterized as weighted averages $\theta = \mathbb { E } \left[ \gamma ( X _ { i } , W _ { i } ) Y _ { i } \right]$ for a weight function $\gamma ( \cdot )$ called the Riesz representer. In the case of ATE estimation under unconfoundedness and with a binary treatment, the Riesz representer is $\gamma ( x , w ) = w / e ( x ) - ( 1 - w ) / ( 1 - e ( x ) )$ , and thus IPW for ATE estimation is in fact a special case of Riesz-representer weighting. Chernozhukov et al. [2022a] use this perspective to develop doubly robust estimators for a wide class of targets by replacing the propensity-estimation step with estimation of the Riesz representer. Hirshberg and Wager [2021] show that the balancing weights construction (7.15) effectively yields a penalized empirical Riesz representer, and thus their method (and results) directly extend to the general setting of Chernozhukov et al. [2022a]. Chernozhukov, Newey, and Singh [2022b] provide a general recipe for machine-learning based estimation of Riesz representers that can be used to automate the construction of double machine learning estimators for generic linear targets.
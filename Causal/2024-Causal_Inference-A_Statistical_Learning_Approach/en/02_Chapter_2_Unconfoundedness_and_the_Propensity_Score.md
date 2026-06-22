# Chapter 2 Unconfoundedness and the Propensity Score

Randomized controlled trials represent a powerful—yet somewhat rigid—class of settings where we can identify and estimate causal effects. One of the overarching focuses of the literature on statistical causal inference (and also of this book) is on ways in which we relax assumptions made in RCTs while preserving our ability to rigorously estimate causal effects, thus broadening the set of problems where causal inference is possible.

In this chapter, we will consider a first, simple relaxation of the RCT assumptions. We will no longer assume that the treatment $W _ { i }$ is randomized; however, we will assume that we observe pre-treatment covariates $X _ { i }$ such that, after conditioning on $X _ { i } .$ , the treatment is as good as randomized. We will then discuss a number of methods for estimating the average treatment effect that exploit this “unconfoundedness” assumption, including ones based on estimating the propensity score (i.e., the conditional probability of receiving treatment). For simplicitly, throughout this chapter (and the next ones also) we will work exclusively under the assumption that units are independently sampled from a superpopulation.

Beyond a single randomized controlled trial The simplest way to move beyond one RCT is to consider two RCTs. As a concrete example, supposed that we are interested in giving teenagers cash incentives to discourage them from smoking. A random subset of $\sim 5 \%$ of teenagers in Palo Alto, CA, and a random subset of ∼ 20% of teenagers in Geneva, Switzerland are eligible for the study.

<table><tr><td>Palo Alto</td><td>Non-S.</td><td>Smoker</td><td>Geneva</td><td>Non-S.</td><td>Smoker</td></tr><tr><td>Treat.</td><td>152</td><td>5</td><td>Treat.</td><td>581</td><td>350</td></tr><tr><td>Control</td><td>2362</td><td>122</td><td>Control</td><td>2278</td><td>1979</td></tr></table>

Within each city, we have an RCT, and in fact readily see that the treatment helps. However, looking at aggregate data is misleading, and it looks like the treatment hurts; this is an example of what is sometimes called Simpson’s paradox:

<table><tr><td>Palo Alto + Geneva</td><td>Non-Smoker</td><td>Smoker</td></tr><tr><td>Treatment</td><td>733</td><td>401</td></tr><tr><td>Control</td><td>4640</td><td>2101</td></tr></table>

Once we aggregate the data, this is no longer an RCT because Genevans are both more likely to get treated, and more likely to smoke whether or not they get treated. In order to get a consistent estimate of the ATE, we need to estimate treatment effects in each city separately:

$$
\hat {\tau} _ {\mathrm{PA}} = \frac {5}{1 5 2 + 5} - \frac {1 2 2}{2 3 6 2 + 1 2 2} \approx -1.7\% ,
$$

$$
\hat {\tau} _ {\mathrm{GVA}} = \frac {3 5 0}{3 5 0 + 5 8 1} - \frac {1 9 7 9}{2 2 7 8 + 1 9 7 9} \approx -8.9\%
$$

$$
\hat {\tau} = \frac {2 6 4 1}{2 6 4 1 + 5 1 8 8} \hat {\tau} _ {\mathrm{PA}} + \frac {5 1 8 8}{2 6 4 1 + 5 1 8 8} \hat {\tau} _ {\mathrm{GVA}} \approx - 6.5 \%.
$$

What are the statistical properties of this estimator? How does this idea generalize to continuous $x ?$

## 2.1 Stratified estimation

Formalizing the above discussion, suppose that we have covariates $X _ { i }$ that take values in a discrete space $X _ { i } \in { \mathcal { X } }$ , with $| \mathcal { X } | = p < \infty$ . Suppose moreover that the treatment assignment is random conditionally on $X _ { i }$ , (i.e., we have an RCT in each group defined by a level of x):

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp W _ {i} \mid X _ {i} = x, \text {   for   all   } x \in \mathcal {X}. \tag {2.1}
$$

Define the conditional average treatment effect as

$$
\tau (x) = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \mid X _ {i} = x \right]. \tag {2.2}
$$

Then, the above suggests that ought to be able to estimate the $\mathrm { A T E } ~ \tau$ by aggregating estimates of the conditional average treatment effect,

$$
\hat {\tau} _ {S T R A T} = \sum_ {x \in \mathcal {X}} \frac {n _ {x}}{n} \hat {\tau} (x), \quad \hat {\tau} (x) = \frac {1}{n _ {x 1}} \sum_ {\{X _ {i} = x, W _ {i} = 1 \}} Y _ {i} - \frac {1}{n _ {x 0}} \sum_ {\{X _ {i} = x, W _ {i} = 0 \}} Y _ {i}, \tag {2.3}
$$

where $n _ { x } = | \{ i : X _ { i } = x \} |$ and $n _ { x w } = | \{ i : X _ { i } = x , W _ { i } = w \} |$ . Another way to look as the estimator in (2.3) is that we apply the difference-in-means estimator after stratifying the sample using the covariates $X _ { i } ;$ and for this reason we will refer to it as the stratified estimator.

The following result verifies that the stratified estimator is in fact valid under our assumptions. Remarkably, the asymptotic variance $V _ { S T R A T }$ does not depend on $| { \mathcal { X } } | = p ,$ , the number of groups, or equivalently the number of “parameters” $\tau ( x )$ estimated on the road to forming (2.3). As we’ll see in the next chapter, this fact plays a key role in enabling efficient non-parametric inference of average treatment effects in observational studies.

Theorem 2.1. Suppose that $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , W _ { i } \} \stackrel { i i d } { \sim } P$ for some distribution P where $X _ { i }$ takes values in a finite cardinality set X and potential outcomes have bounded second moments conditionally on $X _ { i }$ . Suppose furthermore that both (2.1) and SUTVA hold, and that there is non-trivial treatment variation for each $x \in { \mathcal { X } } , i . e .$ , writing $e ( x ) = \mathbb { P } \left[ W _ { i } = 1 \big | X _ { i } = x \right]$ , we have $0 < e ( x ) < 1$ for all x. Then, using notation as in (1.21),

$$
\sqrt {n} \left(\hat {\tau} _ {S T R A T} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {S T R A T}\right)
$$

$$
V _ {S T R A T} = \operatorname{Var} \left[ \tau (X _ {i}) \right] + \mathbb {E} \left[ \frac {\sigma_ {(1)} ^ {2} (X _ {i})}{e (X _ {i})} + \frac {\sigma_ {(0)} ^ {2} (X _ {i})}{1 - e (X _ {i})} \right]. \tag {2.4}
$$

Proof. Write $\lambda ( x ) = \mathbb { P } \left[ X _ { i } = x \right]$ for the prevalence of each level of the covariate $x _ { i }$ and interpret $\ddot { \lambda } ( x ) = n _ { x } / n$ as an estimator for it. We can then expand out the stratified estimator as

$$
\hat {\tau} _ {S T R A T} = \sum_ {x \in \mathcal {X}} \hat {\lambda} (x) \hat {\tau} (x) = \sum_ {x \in \mathcal {X}} \lambda (x) \tau (x) + \sum_ {x \in \mathcal {X}} \left(\hat {\lambda} (x) - \lambda (x)\right) \tau (x)
$$

$$
+ \sum_ {x \in \mathcal {X}} \lambda (x) (\hat {\tau} (x) - \tau (x)) + \sum_ {x \in \mathcal {X}} \left(\hat {\lambda} (x) - \lambda (x)\right) (\hat {\tau} (x) - \tau (x)).
$$

We now study each summand in the expression above. First, note that

$$
\sum_ {x \in \mathcal {X}} \lambda (x) \tau (x) = \mathbb {E} [ \tau (X _ {i}) ] = \tau
$$

is our target estimand. Using simple algebraic manipulations, the second term can be re-expressed as

$$
\sum_ {x \in \mathcal {X}} \left(\hat {\lambda} (x) - \lambda (x)\right) \tau (x) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\tau (X _ {i}) - \tau\right),
$$

and so the standard central limit theorem for IID averages implies that

$$
\sqrt {n} \left(\sum_ {x \in \mathcal {X}} \left(\hat {\lambda} (x) - \lambda (x)\right) \tau (x)\right) \Rightarrow \mathcal {N} \left(0, \operatorname{Var} \left[ \tau (X _ {i}) \right]\right).
$$

Next, our assumptions that $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , W _ { i } \} \stackrel { \mathrm { i i d } } { \sim } P$ and that (2.1) hold imply that $W _ { i } \vert X _ { i } = x , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \sim$ Bernoulli $( e ( x ) )$ . Thus, by Theorem 1.2,

$$
\sqrt {n _ {x}} \left(\hat {\tau} (x) - \tau (x)\right) \Rightarrow \mathcal {N} \left(0, \frac {\sigma_ {(1)} ^ {2}}{e (x)} + \frac {\sigma_ {(0)} ^ {2} (x)}{1 - e (x)}\right),
$$

and the sampling errors in ${ \hat { \tau } } ( x )$ are all asymptotically independent of each other and of $n _ { x }$ (and thus the second summand in our decomposition for $\hat { \tau } _ { S T R A T } )$ . Thus, by Slutsky’s lemma,

$$
\sum_ {x \in \mathcal {X}} \lambda (x) (\hat {\tau} (x) - \tau (x)) \Rightarrow \mathcal {N} \left(0, \sum_ {x \in \mathcal {X}} \lambda (x) \left(\frac {\sigma_ {(1)} ^ {2}}{e (x)} + \frac {\sigma_ {(0)} ^ {2} (x)}{1 - e (x)}\right)\right),
$$

and so the sum of the second and third summands above has the limiting distribution claimed in (2.4). Finally, our above argument also implies that

$$
\left(\hat {\lambda} (x) - \lambda (x)\right) (\hat {\tau} (x) - \tau (x)) = \mathcal {O} _ {P} \left(\frac {1}{n}\right) \text {for all} x \in \mathcal {X},
$$

and so the fourth summand is asymptotically negligible.

Continuous X and the propensity score Above, we considered a setting where X is discrete with a finite number levels, and treatment $W _ { i }$ is as good as random conditionally on $X _ { i } ~ = ~ x$ as in (2.1). In this case, we found that we can still accurately estimate the ATE by aggregating group-wise treatment effect estimates, and that the exact number of groups $| { \mathcal { X } } | = p$ does not affect the accuracy of inference. However, if X is continuous (or the cardinality of X is very large), this result does not apply directly—because we won’t be able to get enough samples for each possible value of $x \in \mathcal { X }$ to be able to define ${ \hat { \tau } } ( x )$ as in (2.3).

In order to generalize our analysis beyond the discrete-X case, we’ll need to move beyond literally trying to estimate $\tau ( x )$ for each value of $x$ by simple averaging, and use a more indirect argument instead. To this end, we first need to generalize the $\mathrm { ^ { 6 6 } R C T }$ in each group” assumption. Formally, we just write the same thing,

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp W _ {i} \mid X _ {i}, \tag {2.5}
$$

although now $X _ { i }$ may be an arbitrary random variable, and interpretation of this statement may require more care. Qualitatively, one way to think about (2.5) is that we have measured enough covariates to capture any dependence between Wi and the potential outcomes and so, given $X _ { i } , W _ { i }$ cannot “peek” at the $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \}$ . We call this assumption unconfoundedness.

The assumption (2.5) may seem like a difficult assumption to use in practice, since it involves conditioning on a continuous random variable. However, as shown by Rosenbaum and Rubin [1983], this assumption can be made considerably more tractable by considering the propensity score

$$
e (x) = \mathbb {P} \left[ W _ {i} = 1 \mid X _ {i} = x \right]. \tag {2.6}
$$

Statistically, a key property of the propensity score is that it is a balancing score: If (2.5) holds, then in fact

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp W _ {i} \mid e \left(X _ {i}\right), \tag {2.7}
$$

i.e., it actually suffices to control for $e ( X )$ rather than $X$ to remove biases associated with a non-random treatment assignment. We can verify this claim as follows:

$$
\begin{array}{l} \mathbb {P} \left[ W _ {i} = w \mid \{Y _ {i} (0), Y _ {i} (1) \}, e (X _ {i}) \right] \\ = \int_ {\mathcal {X}} \mathbb {P} \left[ W _ {i} = w \mid \left\{Y _ {i} (w) \right\}, X _ {i} = x \right] \mathbb {P} \left[ X _ {i} = x \mid \left\{Y _ {i} (w) \right\}, e (X _ {i}) \right] d x \\ = \int_ {\mathcal {X}} \mathbb {P} \left[ W _ {i} = w \mid X _ {i} = x \right] \mathbb {P} \left[ X _ {i} = x \mid \left\{Y _ {i} (w) \right\}, e \left(X _ {i}\right) \right] d x \quad (\text {unconf.}) \\ = \left\{ \begin{array}{l l} e (X _ {i}) & \text { if   w = 1, } \\ 1 - e (X _ {i}) & \text { else. } \end{array} \right. \\ \end{array}
$$

The implication of (2.7) is that if we can partition our observations into groups with (almost) constant values of the propensity score $e ( x )$ , then we can consistently estimate the average treatment effect via variants of $\scriptstyle { \hat { \tau } } _ { S T R A T }$ .

Propensity stratification One instantiation of this idea is propensity stratification, which proceeds as follows. First obtain an estimate $\hat { e } ( x )$ of the propensity score via non-parametric regression, and choose a number of strata J. Then:

1. Sort the observations according to their propensity scores, such that

$$
\hat {e} \left(X _ {i _ {1}}\right) \leq \hat {e} \left(X _ {i _ {2}}\right) \leq \dots \leq \hat {e} \left(X _ {i _ {n}}\right). \tag {2.8}
$$

2. Split the sample into J evenly size strata using the sorted propensity score and, in each stratum $j = 1 , . . . , J $ , compute the simple differencein-means treatment effect estimator for the stratum:

$$
\hat {\tau} _ {j} = \frac {\sum_ {j = \lfloor (j - 1) n / J \rfloor + 1} ^ {\lfloor j n / J \rfloor} W _ {i} Y _ {i}}{\sum_ {j = \lfloor (j - 1) n / J \rfloor + 1} ^ {\lfloor j n / J \rfloor} W _ {i}} - \frac {\sum_ {j = \lfloor (j - 1) n / J \rfloor + 1} ^ {\lfloor j n / J \rfloor} \left(1 - W _ {i}\right) Y _ {i}}{\sum_ {j = \lfloor (j - 1) n / J \rfloor + 1} ^ {\lfloor j n / J \rfloor} \left(1 - W _ {i}\right)}. \tag {2.9}
$$

3. Estimate the average treatment by applying the idea of (2.3) across strata:

$$
\hat {\tau} _ {P S T R A T} = \frac {1}{J} \sum_ {j = 1} ^ {J} \hat {\tau} _ {j}. \tag {2.10}
$$

The arguments described above immediately imply that, thanks to (2.7), ˆτP ST RAT is consistent for $\tau$ whenever $\hat { e } ( x )$ is uniformly consistent for $e ( x )$ and the number of strata J grows appropriately with $n ;$ see Exercise 4 in Chapter 16 for more details.

## 2.2 Inverse-propensity weighting

Another, algorithmically simpler way of exploiting unconfoundedness is via inverse-propensity weighting (IPW). As before, we start by estimating $\hat { e } ( x )$ via non-parametric regression; however, we then use the outputs of our propensity model to build a re-weighted difference-in-means-type estimator

$$
\hat {\tau} _ {I P W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{\hat {e} (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})}\right). \tag {2.11}
$$

The intuition behind IPW is that, if some units are very unlikely to get treated, then we should up-weight them on the rare event where they do get treated and down-weight them on the more common event where they don’t, etc., and that this re-weighting weighting allows use to “undo” sampling bias caused by variation in the propensity score.

The simplest way to analyze it is by comparing it to an oracle that actually knows the propensity score:

$$
\hat {\tau} _ {I P W} ^ {*} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})}\right). \tag {2.12}
$$

We start by establish asymptotic properties of the oracle IPW estimator below. Once we’ve established consistency of $\hat { \tau } _ { I P W } ^ { * }$ , it follows as an (almost) immediate corollary that ${ \hat { \tau } } _ { I P W }$ is also consistent provided that $\hat { e } ( x )$ is consistent for $e ( x )$ .

Theorem 2.2. Suppose that $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , W _ { i } \} \stackrel { i i d } { \sim } P$ , that both (2.5) and SUTVA hold, and that all moments used in the expression for $V _ { I P W } ,$ ∗ below are finite. Then, the oracle IPW estimator is unbiased, E $\left[ \hat { \tau } _ { I P W } ^ { * } \right] = \tau$ , and

$$
\sqrt {n} \left(\hat {\tau} _ {I P W} ^ {*} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {I P W ^ {*}}\right)
$$

$$
V _ {I P W ^ {*}} = \operatorname{Var} \left[ \tau (X _ {i}) \right] + \mathbb {E} \left[ \frac {\left(\mu_ {(0)} (X _ {i}) + (1 - e (X _ {i})) \tau (X _ {i})\right) ^ {2}}{e (X _ {i}) (1 - e (X _ {i}))} \right] \tag {2.13}
$$

$$
+ \mathbb {E} \left[ \frac {\sigma_ {(1)} ^ {2} (X _ {i})}{e (X _ {i})} + \frac {\sigma_ {(0)} ^ {2} (X _ {i})}{1 - e (X _ {i})} \right].
$$

Proof. We start by checking the unbiasedness statement as follows:

$$
\begin{array}{l} \mathbb {E} \left[ \hat {\tau} _ {I P W} ^ {*} \right] = \mathbb {E} \left[ \frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})} \right] (IID) \\ = \mathbb {E} \left[ \frac {W _ {i} Y _ {i} (1)}{e \left(X _ {i}\right)} - \frac {\left(1 - W _ {i}\right) Y _ {i} (0)}{1 - e \left(X _ {i}\right)} \right] (SUTVA) \\ = \mathbb {E} \left[ \mathbb {E} \left[ \frac {W _ {i} Y _ {i} (1)}{e (X _ {i})} \mid X _ {i} \right] - \mathbb {E} \left[ \frac {(1 - W _ {i}) Y _ {i} (0)}{1 - e (X _ {i})} \mid X _ {i} \right] \right] \\ = \mathbb {E} \left[ \frac {\mathbb {E} [ W _ {i} | X _ {i} ] \mathbb {E} [ Y _ {i} (1) | X _ {i} ]}{e (X _ {i})} - \frac {\mathbb {E} [ 1 - W _ {i} | X _ {i} ] \mathbb {E} [ Y _ {i} (0) | X _ {i} ]}{1 - e (X _ {i})} \right] (\mathrm{unconf.}) \\ = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \right] = \tau . \\ \end{array}
$$

Next, under our IID sampling assumption, (2.13) follows immediately from the central limit theorem for IID averages with

$$
V _ {I P W ^ {*}} = \mathrm{Var} \left[ \frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})} \right],
$$

provided this variance is finite. It remains to derive the claimed alternative expression for $V _ { I P W ^ { * } }$ . To this end, building on notation from (1.21), we introduce an auxiliary function

$$
c (x) = \mu_ {(0)} (x) + (1 - e (x)) \tau (x),
$$

and write $\varepsilon _ { i } ( w ) = Y _ { i } ( w ) - \mu _ { ( w ) } ( X _ { i } )$ . Given these preliminaries, we expand out

$$
\begin{array}{l} \frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})} \\ = \frac {W _ {i} (\mu_ {(1)} (X _ {i}) + \varepsilon_ {i} (1))}{e (X _ {i})} - \frac {(1 - W _ {i}) (\mu_ {(0)} (X _ {i}) + \varepsilon_ {i} (0))}{1 - e (X _ {i})} \\ = \tau (X _ {i}) + \left(\frac {W _ {i}}{e (X _ {i})} - \frac {1 - W _ {i}}{1 - e (X _ {i})}\right) c (X _ {i}) + \frac {W _ {i} \varepsilon_ {i} (1)}{e (X _ {i})} - \frac {(1 - W _ {i}) \varepsilon_ {i} (0)}{1 - e (X _ {i})}. \\ \end{array}
$$

Furthermore, E $\left[ W _ { i } / e ( X _ { i } ) - ( 1 - W _ { i } ) / ( 1 - e ( X _ { i } ) ) \bigm | X _ { i } \right] = 0$ by definition of the propensity score, and E $\left[ \varepsilon _ { i } ( w ) \big | X _ { i } , W _ { i } \right] = 0$ by unconfoundedness, so

$$
\begin{array}{l} \operatorname{Var} \left[ \frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})} \right] = \operatorname{Var} [ \tau (X _ {i}) ] \\ + \mathbb {E} \left[ \left(\left(\frac {W _ {i}}{e (X _ {i})} - \frac {1 - W _ {i}}{1 - e (X _ {i})}\right) c (X _ {i})\right) ^ {2} \right] + \mathbb {E} \left[ \left(\frac {W _ {i} \varepsilon_ {i} (1)}{e (X _ {i})} - \frac {(1 - W _ {i}) \varepsilon_ {i} (0)}{1 - e (X _ {i})}\right) ^ {2} \right]. \\ \end{array}
$$

The claimed expression for $V _ { I P W }$ ∗ follows by simplifying the one above.

![image_01](images/image_01.png)

One noteworthy assumption made seemingly in passing above is that all moments used in (2.13) are well-defined and finite. This is, however, a highly non-trivial assumption. If the potential outcomes are uniformly bounded, then this condition is essentially equivalent to assuming that

$$
\mathbb {E} \left[ 1 / (e (X _ {i}) (1 - e (X _ {i}))) \right] <   \infty . \tag {2.14}
$$

Meanwhile if we simply assume that the potential outcomes have finite second moments then we need to assume something stronger, e.g., there exists an $\eta > 0$ for which

$$
\eta \leq e (x) \leq 1 - \eta \text {   for   all   } x \in \mathcal {X}. \tag {2.15}
$$

These assumptions are generally known as overlap assumptions, and codify the requirement that there must be non-trivial randomness in treatment assignment conditionally on x. We refer to (2.14) as weak overlap, and (2.15) as strong overlap. Qualitatively an overlap-type assumption must in general be made for non-parametric treatment effect estimation to be possible: If treatment assignment $W _ { i }$ is perfectly predictable from $X _ { i } .$ , then there is no actual randomness in treatment assignment, and so treatment effect estimation justified by treatment randomization cannot be possible.

How accurate is inverse-propensity weighting? We established above that IPW is unbiased and asymptotically normal when implemented with the true propensity scores, and consistent with estimated propensity scores. This is of course a nice result to have given the simple functional form of the IPW estimator. But do these results imply that IPW is any good?

To get a benchmark for our results about IPW, it is helpful to re-visit the setting of the beginning of this lecture where X is discrete, in which case we can use the result in Theorem 2.1 for $\scriptstyle { \hat { \tau } } _ { S T R A T }$ as a point of comparison. When propensity scores are known, both $\hat { \tau } _ { I P W } ^ { * }$ and $\scriptstyle { \hat { \tau } } _ { S T R A T }$ are asymptotically normal, and from (2.4) and (2.13) we see that

$$
V _ {I P W ^ {*}} = V _ {S T R A T} + \mathbb {E} \left[ \frac {\left(\mu_ {(0)} (X _ {i}) + (1 - e (X _ {i})) \tau (X _ {i})\right) ^ {2}}{e (X _ {i}) (1 - e (X _ {i}))} \right]. \tag {2.16}
$$

Thus, unless $\mu _ { ( 0 ) } ( X _ { i } ) + ( 1 - e ( X _ { i } ) ) \tau ( X _ { i } )$ is zero almost surely, $\hat { \tau } _ { I P W } ^ { * }$ has a strictly worse asymptotic variance than $\scriptstyle { \hat { \tau } } _ { S T R A T }$ . Meanwhile, when propensity scores are not known, we here only proved a consistency result for ${ \hat { \tau } } _ { I P W }$ (no central limit theorem), and so we cannot even make a proper comparison. Thus, at first glance, a comparison of Theorems 2.1 and 2.2 makes the behavior of IPW seem somewhat disappointing.

However, on closer look, the picture gets more complicated: It turns out that $\scriptstyle { \hat { \tau } } _ { S T R A T }$ can actually be understood as an implementation of the IPW estimator with a specific choice of estimated propensity score $\hat { e } ( x )$ . In the setting of (2.3) where $\scriptstyle { \hat { \tau } } _ { S T R A T }$ is well defined, we have:

$$
\hat {\tau} _ {S T R A T} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{\hat {e} (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})}\right), \quad \hat {e} (x) = \frac {n _ {x 1}}{n _ {x}}. \tag {2.17}
$$

Thus, when $\mathcal { X }$ is discrete, it turns out that an instance of a feasible IPW estimator, namely $\scriptstyle { \hat { \tau } } _ { S T R A T }$ , is actually more precise than the “oracle” IPW estimator (see also Exercise 1 in Chapter 16).13 Understanding and resolving this seeming paradox lies will be at the heart of understanding how to design accurate estimators of the average treatment effect under unconfoundedness— including with continuous covariates.

Randomized and observational studies One nuance we glossed over is that there are two conceptually distinct ways that one could end up with potential outcomes satisfying (2.5). The first option is that the data was generated by an experiment with variable treatment propensities: Nature generated $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \} \sim P$ , and then an experimenter randomly assigned treatments $W _ { i } \sim \mathrm { B e r n o u l l i } ( e ( X _ { i } ) )$ for some function $e ( \cdot )$ of the covariates. Under this setting, the experimenter knows that (2.5) must hold, because they themselves generated treatment in a way that satisfies the assumption. Essentially, the experimenter is running the same Bernoulli trial as considered in (1.8), except with randomization probabilities that vary with the $X _ { i }$ . Although covariate-dependent randomization probabilities require statistical accommodation, such experiments are conceptually akin to the ones discussed in Chapter 1—and provide comparably strong, gold-standard causal evidence.

Example 2. Arceneaux, Gerber, and Green [2006] run a randomized study to measure the effectiveness of voter mobilization phone calls in getting people to vote in midterm elections. The study is run in two states, Michigan and Iowa, and randomization is stratified by both state and by competitiveness of the congressional district, with per-stratum randomization probabilities varying from 1% to 15%. This is a randomized controlled trial; however, properly accounting for variation in the randomization probabilities (e.g., via propensity stratification) is required for a valid analysis, and simply taking a global difference in means would be prone to Simpson’s paradox.

The second option is that there was no experiment: Nature generated $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , W _ { i } \} \sim P$ , and we simply posit that (2.5) holds. This marks a much bigger departure from the setting of Chapter 1. There is no analyst who ran an experiment; rather, we posit that data is generated as though someone had run the experiment described in the previous chapter. Such settings are referred to as natural experiments or observational study designs. Because no experiment was actually run, the assumption (2.5) can always be challenged in observational studies—and as such the resulting causal evidence is sometimes considered more tentative than evidence obtained via randomized experiments.

Example 3. LaLonde [1986] considers evaluating the benefits from a jobs training program by comparing post-intervention earnings for people enrolled in a pilot program to members of the general public who were not enrolled in the program. This is not a randomized study design, and members of the general public differ from those in the pilot program along a number of preintervention metrics. The initial assessment of LaLonde [1986] regarding the possibility of getting credible causal estimates out of such observational data was pessimistic. However, in later work, Dehejia and Wahba [1999] showed that approaches that start by modeling the propensity score (i.e., here, the probability of joining the pilot program given pre-intervention characteristics) showed more promising behavior,14 and were often able to match experimental benchmarks.

Another major practical difference between randomized trials with covariatedependent randomization versus observational studies is that, in the former case, the treatment propensities $e ( X _ { i } )$ are usually known (because they were chosen by the experimenter), and so methods such as oracle IPW with guarantees as in Theorem 2.2 are available. In contrast, in the observational study setting, treatment propensities need to be estimated, and thus robustness of methods to errors in the propensity scores is important—particularly in settings as below where propensity scores are hard to estimate accurately. As of now, we have not yet seen estimators that, in a setting with continuous $X _ { i } ,$ can take in estimated propensity scores and output asymptotically normal average treatment effect estimates with $1 / { \sqrt { n } }$ -scale errors. In the next chapter, we will present an improvement to IPW that can achieve asymptotic normality even with estimated propensity scores.

Example 4. Ross et al. [2024] use electronic health record data from the Veterans’ Administration to estimate the benefits of psychiatric hospitalization on suicide prevention among patients with a recent suicide attempt of suicide ideation. There is no randomization, and hospitalized versus non-hospitalized patients differ on pre-treatment characteristics. The authors argue that after controlling for rich medical history available through the electronic health records, it is plausible for unconfoundedness to hold, and proceed to use propensity score methods. However, given that the pre-treatment is high-dimensional with complex structure, it is necessary to use a machine learning approach to get reasonable propensity score estimates—and any down-stream used of these propensity scores should be robust to likely estimation errors in this step.

## 2.3 Bibliographic notes

The central role of the propensity score in estimating causal effects was first emphasized by Rosenbaum and Rubin [1983], while associated methods for estimation such as propensity stratification are discussed in Rosenbaum and Rubin [1984]. Hirano, Imbens, and Ridder [2003] provide a detailed discussion

work of LaLonde [1986] is how we should properly “control $\operatorname { f o r } ^ { \dag }$ pre-intervention covariates in an observational study setting. In informal econometric practice, when an analyst says they have controlled for a set of covariates, they mean that they’ve run a regression where they’ve added the covariates as predictors; e.g., in our setting, they might have sought to estimate a treatment effect via the ˆτ coefficient from the regression $Y _ { i } \sim \alpha + W _ { i } \tau + X _ { i } \cdot \beta .$ . This type of regression, however, is not justified by the unconfoundedness assumption (2.5) and, unlike IPW or other propensity-score methods, is not generally consistent for the average treatment effect under unconfoundedness. The unconfoundedness assumption (2.5) is nonparametric; and thus using it requires adjusting for Xi non-parametrically.

of the asymptotics of IPW-style estimators that expands on the result given in Theorem 2.1. In particular they present conditions with continuous $X _ { i }$ under which IPW with non-parametrically estimated propensity scores can outperform oracle IPW.

Another popular way of leveraging the propensity score in practice is propensity matching, i.e., estimating treatment effects by comparing pairs of units with similar values of ˆe(Xi). For a some recent discussions of matching in causal inference, see Abadie and Imbens [2006, 2016], Diamond and Sekhon [2013], Zubizarreta [2012], and references therein.
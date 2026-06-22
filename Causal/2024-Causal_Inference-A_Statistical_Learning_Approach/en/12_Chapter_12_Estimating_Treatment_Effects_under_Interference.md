# Chapter 12 Estimating Treatment Effects under Interference

In the previous chapter, we introduced exposure mappings as a tool for modeling cross-unit interference, and permutation-based methods for testing for the presence of interference. The next natural question—and our focus in this chapter—is: Once we’ve accepted that interference exists, how can we estimate relevant treatment effects that account for interference?

Exposure effects For simplicity, we will here focus on a setting here Assumption 11.1 holds with a finite-cardinality exposure with a shared domain. Specifically, we will consider a setting where we have $i = 1 , \ldots , n$ units with outcomes $Y _ { i } \in \mathbb { R }$ and treatment $W _ { i } \in \{ 0 , 1 \}$ . There can be cross-unit interference; however, this interference can be captured in terms of an exposure mapping $\displaystyle H _ { i } : \{ 0 , 1 \} ^ { n }  \mathcal { H }$ with a shared domain H with $| { \mathcal { H } } | < \infty$ . We thus have potential outcomes with a consistency condition

$$
\{Y _ {i} (h) \} _ {h \in \mathcal {H}}, \quad Y _ {i} = Y _ {i} (H _ {i} (\mathbf {W})). \tag {12.1}
$$

Given this assumption, we can define various sample-average treatment effects by comparing mean potential outcomes across exposure levels h, $h ^ { \prime } \in \mathcal { H }$ ,

$$
\bar {\tau} (h, h ^ {\prime}) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(Y _ {i} (h ^ {\prime}) - Y _ {i} (h)\right). \tag {12.2}
$$

Our goal is to estimate these quantities and provide confidence intervals for them.

Example 16. Rogers and Feller [2018] reports results on a randomized trial to improve school attendance among students with high risk of absenteeism by sending attendance information to parents. In some settings, a family had multiple students eligible for the study, and the authors were interested in spillovers: Did sending attendance information about one student also affect their siblings’ behavior? To study this question, the authors posited an exposure mapping with 3 exposure levels: $( 1 )$ student received treatment; (2) student untreated by with treated sibling; and (3) student in family with no treatment. Then, one can define a number of natural estimands of the form (12.2), such as a direct effect $( 1 ) \ \mathrm { v s . } \ ( 3 )$ , and a spillover effect (2) vs. (3).

Unbiased estimation The setup considered here, i.e., with a randomized trial executed on a set of n unspecified study participants, is closely related to the setting of Theorem 1.1, except that now of course SUTVA no longer holds and we instead need to rely on a more complex exposure mapping to capture interference. And it turns out that an analogue to Theorem 1.1 still holds: We can get unbiased estimates for the exposure contrasts (12.2) essentially without further assumptions.

The simplest way to construct unbiased estimators here is via inversepropensity weighting (IPW). Suppose that treatment is Bernoullirandomized,

$$
W _ {i} \sim \mathrm{Bernoulli} (e _ {i}), \quad 0 <   e _ {i} <   1, \tag {12.3}
$$

independently for all $i = 1 , \ldots , n .$ , and let $e _ { i } ( h ) = \mathbb { P } \left[ H _ { i } ( \mathbf { W } ) = h \right]$ with treatment generated according to (12.3). The, the natural IPW estimator,

$$
\hat {\tau} _ {I P W} (h, h ^ {\prime}) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {1 (\{H _ {i} (\mathbf {W}) = h ^ {\prime} \}) Y _ {i}}{e _ {i} (h ^ {\prime})} - \frac {1 (\{H _ {i} (\mathbf {W}) = h \}) Y _ {i}}{e _ {i} (h)}\right), \tag {12.4}
$$

is unbiased for $\bar { \tau } ( h , h ^ { \prime } )$ . We use the notation of the type

$$
\mathbb {E} _ {W} \left[ \hat {\tau} _ {I P W} (h, h ^ {\prime}) \right] = \mathbb {E} \left[ \hat {\tau} _ {I P W} (h, h ^ {\prime}) \mid \{Y _ {i} (h) \} _ {i = 1, \dots , n; h \in \mathcal {H}} \right], \tag {12.5}
$$

i.e., where $\mathbb { E } _ { W }$ denotes expectations over random treatment assignment while holding potential outcomes fixed.

Theorem 12.1. Under assumptions (12.1) and (12.3), suppose furthermore that $e _ { i } ( h ) , e _ { i } ( h ^ { \prime } ) > 0$ for all $i = 1 , \ldots , n$ . Then

$$
\mathbb {E} _ {W} \left[ \hat {\tau} _ {I P W} (h, h ^ {\prime}) \right] = \bar {\tau} (h, h ^ {\prime}). \tag {12.6}
$$

Proof. Invoking (12.1) and randomization yields

$$
\mathbb {E} _ {W} \left[ \hat {\tau} _ {I P W} (h, h ^ {\prime}) \right]
$$

$$
= \mathbb {E} _ {W} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {1 \left(\{H _ {i} (\mathbf {W}) = h ^ {\prime} \}\right) Y _ {i} (h ^ {\prime})}{e _ {i} (h ^ {\prime})} - \frac {1 \left(\{H _ {i} (\mathbf {W}) = h \}\right) Y _ {i} (h)}{e _ {i} (h)}\right) \right]
$$

$$
= \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {\mathbb {E} _ {W} \left[ 1 \left(\{H _ {i} (\mathbf {W}) = h ^ {\prime} \}\right) \right] Y _ {i} (h ^ {\prime})}{e _ {i} (h ^ {\prime})} - \frac {\mathbb {E} _ {W} \left[ 1 \left(\{H _ {i} (\mathbf {W}) = h \}\right) Y _ {i} (h) \right]}{e _ {i} (h)}\right)
$$

$$
= \frac {1}{n} \sum_ {i = 1} ^ {n} \left(Y _ {i} (h ^ {\prime}) - Y _ {i} (h)\right).
$$

For the last equality we also used (12.3) and the fact that $e _ { i } ( h ) , e _ { i } ( h ^ { \prime } ) > 0$ .

![image_10](images/image_10.png)

Inference and uncertainty quantification Where things get more challenging is in seeking confidence intervals. The result above was a generalization of Theorem 1.1 to settings with interference, with a proof following exactly the same blueprint. In Chapter 1, when we sought to move past unbiasedness and establish inferential results, we added an extra assumption that potential outcomes are independently sampled from a broader population (see, e.g., Theorem 1.2). However, while such an IID-sampling assumption is easy to make under SUTVA, it is much more challenging to posit general sampling assumptions for potential outcomes under interference. Units now interact with each other (e.g., they are friends in a social network), and writing down credible generative models that capture such cross-unit relationships (e.g., writing down credible generative models for friendship networks) is something that requires deep subject matter knowledge and cannot easily be done at the level of abstraction sought here.

In this chapter, we will pursue an alternate route and seek to establish inference results that only depend on random treatment assignment—and do not make any sampling assumptions on the potential outcomes. In the causal inference literature, this approach is often referred to as the finite-population approach, as it does not appeal to the existence of a superpopulation from which units were drawn. We will start, in Section 12.1, by reviewing finite-population methods under SUTVA—and revisiting our discussion from Chapter 1 without the IID sampling assumption. Then, in Section 12.2, we will extend this discussion to settings with interference.

## 12.1 Finite-population methods

Our goal here is to provide an alternative to Theorem 1.2 that enables inference in randomized-controlled trials under SUTVA without relying on superpopulation-sampling assumption. Finite-population analysis of randomized trials, including the results given here, go back to Neyman [1923]. The following result presents what’s often called the Neyman-variance analysis in the case of a Bernoulli design.65 Under SUTVA, we are only interested in the treatment-control contrast, and so will use short-hand $\bar { \tau } : = \bar { \tau } ( 0 , 1 )$ for the sample-average treatment effect (SATE), $\hat { \tau } _ { I P W } : = \hat { \tau } _ { I P W } ( 0 , 1 )$ for the estimated treatment effect, and $e _ { i } = e _ { i } ( 1 )$ for the propensity score.

Theorem 12.2. Under the setting of Theorem ${ \it 1 2 . 1 , }$ suppose furthermore that SUTVA holds, i.e., $H _ { i } ( \mathbf { w } ) = w _ { i }$ . Then

$$
n \operatorname{Var} _ {W} \left[ \hat {\tau} _ {I P W} \right] = \bar {\sigma} ^ {2} \leq \sigma^ {2},
$$

$$
\bar {\sigma} ^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {Y _ {i} (0) ^ {2}}{1 - e _ {i}} + \frac {Y _ {i} (1) ^ {2}}{e _ {i}} - \left(Y _ {i} (1) - Y _ {i} (0)\right) ^ {2}\right), \tag {12.7}
$$

$$
\sigma^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {Y _ {i} (0) ^ {2}}{1 - e _ {i}} + \frac {Y _ {i} (1) ^ {2}}{e _ {i}}\right).
$$

Furthermore, $\sigma ^ { 2 }$ admits an unbiased estimator,

$$
\mathbb {E} _ {W} \left[ \widehat {V} \right] = \sigma^ {2}, \widehat {V} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {(1 - W _ {i}) Y _ {i} ^ {2}}{(1 - e _ {i}) ^ {2}} + \frac {W _ {i} Y _ {i} ^ {2}}{e _ {i} ^ {2}}\right). (1 2. 8)
$$

Proof. Thanks to Theorem 12.1, we have

$$
\begin{array}{l} n \operatorname{Var} _ {W} \left[ \hat {\tau} _ {I P W} \right] = n \mathbb {E} _ {W} \left[ \left(\hat {\tau} _ {I P W} - \bar {\tau}\right) ^ {2} \right] \\ = n \mathbb {E} _ {W} \left[ \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i}}{e _ {i}} - \frac {1 - W _ {i}}{1 - e _ {i}}\right) Y _ {i} - \frac {1}{n} \sum_ {i = 1} ^ {n} (Y _ {i} (1) - Y _ {i} (0))\right) ^ {2} \right]. \\ \end{array}
$$

By SUTVA and because the $W _ { i }$ are independent of each other, we can furtherexpand this expression as

$$
n \mathbb {E} _ {W} \left[ \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i}}{e _ {i}} - 1\right) Y _ {i} (1) - \left(\frac {1 - W _ {i}}{1 - e _ {i}} - 1\right) Y _ {i} (0)\right) ^ {2} \right]
$$

$$
= \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} _ {W} \left[ \left(\left(\frac {W _ {i}}{e _ {i}} - 1\right) Y _ {i} (1) - \left(\frac {1 - W _ {i}}{1 - e _ {i}} - 1\right) Y _ {i} (0)\right) ^ {2} \right]
$$

$$
= \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\left(\frac {1}{e _ {i}} - 1\right) Y _ {i} (1) ^ {2} + \left(\frac {1}{1 - e _ {i}} - 1\right) Y _ {i} (0) ^ {2} + 2 Y _ {i} (0) Y _ {i} (1)\right)
$$

$$
= \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {Y _ {i} (1) ^ {2}}{e _ {i}} + \frac {Y _ {i} (0) ^ {2}}{1 - e _ {i}} - (Y _ {i} (1) - Y _ {i} (0)) ^ {2}\right),
$$

where the second equality above follows by computing binomial probabilities and the third by expanding out the square $( Y _ { i } ( 1 ) - Y _ { i } ( 0 ) ) ^ { 2 }$ . This establishes (12.7). Finally, (12.8) can be proven by following the argument used in Theorem 12.1. □

The main observation is that, under the finite-population model, the variance $\bar { \sigma } ^ { 2 }$ depends on differences of potential outcomes, and cannot generally be estimated from data without further assumptions. However, the variance admits a simple upper bound $\sigma ^ { 2 }$ that is identified from data—and in fact this variance estimate corresponds to the usual variance estimate for ${ \hat { \tau } } _ { I P W }$ under IID sampling. Thus, exact inference for the ATE under IID sampling provides conservative inference for the SATE in the finite-population model. This fact will also show up under interference.

It remains to establish a construction for confidence intervals. Since we no longer have access to an IID stream of data, we will no longer be able to invoke a classical central-limit theorem; rather, we will need to rely on finite-sample Gaussian approximation results. In the result below, we will also consider a self-normalized version of IPW,

$$
\hat {\tau} _ {S I P W} = \frac {\sum_ {i = 1} ^ {n} W _ {i} Y _ {i} / e _ {i}}{\sum_ {i = 1} ^ {n} W _ {i} / e _ {i}} - \frac {\sum_ {i = 1} ^ {n} (1 - W _ {i}) Y _ {i} / (1 - e _ {i})}{\sum_ {i = 1} ^ {n} (1 - W _ {i}) / (1 - e _ {i})}, \tag {12.9}
$$

as this generally improves large-sample performance (see, e.g., Exercise 1).

Theorem 12.3. Suppose we have a sequence of randomized trials with growing sample size n that all satisfy the conditions of Theorem 12.2, and write $\bar { \tau } _ { n }$ for the SATE in each of these randomized trials. Suppose furthermore that there are constants $\eta , M < \infty$ such that $\eta \le e _ { i } \le 1 - \eta$ and $\left| Y _ { i } ( 0 ) \right| , \left| Y _ { i } ( 1 ) \right| \leq M$ for all units, and that lim in $\mathrm { f } _ { n  \infty } \bar { \sigma } _ { n } ^ { 2 } > 0$ with $\bar { \sigma } _ { n } ^ { 2 }$ as defined below. Then,

$$
\sqrt {n} \left(\frac {\hat {\tau} _ {S I P W} - \bar {\tau} _ {n}}{\bar {\sigma} _ {n}}\right) \Rightarrow \mathcal {N} (0, 1), \quad \bar {\mu} _ {n} (w) = \frac {1}{n} \sum_ {i = 1} ^ {n} Y _ {i} (w), \tag {12.10}
$$

$$
\bar {\sigma} _ {n} ^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {(Y _ {i} (0) - \bar {\mu} _ {n} (0)) ^ {2}}{1 - e _ {i}} + \frac {(Y _ {i} (1) - \bar {\mu} _ {n} (1)) ^ {2}}{e _ {i}} - (Y _ {i} (1) - Y _ {i} (0)) ^ {2}\right),
$$

Furthermore, the following variance estimator

$$
\hat {\mu} _ {n} (0) = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - W _ {i}) Y _ {i}}{1 - e _ {i}}, \quad \hat {\mu} _ {n} (1) = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i} Y _ {i}}{e _ {i}}, \tag {12.11}
$$

$$
\hat {\sigma} _ {n} ^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {(1 - W _ {i}) (Y _ {i} - \hat {\mu} _ {n} (0)) ^ {2}}{(1 - e _ {i}) ^ {2}} + \frac {W _ {i} (Y _ {i} - \hat {\mu} _ {n} (1)) ^ {2}}{e _ {i} ^ {2}}\right),
$$

is asymptotically conservative, lim $\mathrm { s u p } _ { n \to \infty } \bar { \sigma } _ { n } / \hat { \sigma } _ { n } \le _ { p } 1$ , and usual normal confidence intervals are valid

$$
\limsup _ {n \to \infty} \mathbb {P} \left[ | \hat {\tau} _ {S I P W} - \bar {\tau} _ {n} | \leq \hat {\sigma} _ {n} / \sqrt {n}   \Phi^ {- 1} (1 - \alpha / 2) \right] \leq 1 - \alpha , \tag {12.12}
$$

for any $0 < \alpha < 1$ .

Proof. Thanks to self-normalization and SUTVA, we have an error decomposition

$$
\begin{array}{l} \hat {\tau} _ {S I P W} - \bar {\tau} _ {n} = \Delta (1) \left/ \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i}}{e _ {i}} - \Delta (0) \left. \right/ \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {1 - W _ {i}}{1 - e _ {i}}, \\ \Delta (0) = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - W _ {i}) (Y _ {i} (0) - \bar {\mu} _ {n} (0))}{1 - e _ {i}}, \Delta (1) = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i} (Y _ {i} (1) - \bar {\mu} _ {n} (1))}{e _ {i}}. \\ \end{array}
$$

By Theorems 12.1 and 12.2, we immediately get

$$
\mathbb {E} _ {W} \left[ \Delta (1) - \Delta (0) \right] = 0, n \operatorname{Var} _ {W} \left[ \Delta (1) - \Delta (0) \right] = \bar {\sigma} _ {n} ^ {2}.
$$

Furthermore, our boundedness assumptions imply that all summands comprising $\Delta ( 0 )$ and $\Delta ( 1 )$ are bounded by $2 M / \eta$ , and so the Berry–Esseen bound implies that

$$
\sup _ {z \in \mathbb {R}} \left| \mathbb {P} \left[ \frac {\sqrt {n} (\Delta (1) - \Delta (0))}{\bar {\sigma} _ {n}} \leq z \right] - \Phi (z) \right| \leq \frac {8 C M ^ {3} / \eta^ {3}}{\bar {\sigma} _ {n} ^ {3} \sqrt {n}}, \tag {12.13}
$$

where $\Phi ( \cdot )$ is the standard Gaussian cumulative distribution function and C is the Berry–Esseen constant; we also note that the right-hand side term of (12.13) goes to 0 with n because we have assumed that lim in $\mathrm { f } _ { n  \infty } \bar { \sigma } _ { n } ^ { 2 } > 0$ .

Meanwhile, again thanks to our overlap and boundedness assumptions, we can use standard concentration arguments to verify that

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \frac {1 - W _ {i}}{1 - e _ {i}} - 1, \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i}}{e _ {i}} - 1 = \mathcal {O} _ {P} \left(\frac {1}{\sqrt {n}}\right),
$$

and also that

$$
\Delta (0), \Delta (1) = \mathcal {O} _ {P} \left(\frac {1}{\sqrt {n}}\right).
$$

This implies that

$$
\hat {\tau} _ {S I P W} - \bar {\tau} _ {n} = \Delta (1) - \Delta (0) + \mathcal {O} _ {P} \left(\frac {1}{n}\right),
$$

and so (12.10) follows from (12.13). Finally, we can again use concentration arguments to verify that

$$
\lim _ {n \to \infty} \hat {\sigma} _ {n} ^ {2} - \sigma_ {n} ^ {2} = _ {p} 0, \quad \sigma_ {n} ^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {(Y _ {i} - \bar {\mu} _ {n} (0)) ^ {2}}{1 - e _ {i}} + \frac {(Y _ {i} - \bar {\mu} _ {n} (1)) ^ {2}}{e _ {i}}\right),
$$

and by Theorem 12.2 we also get $\sigma _ { n } ^ { 2 } \geq \bar { \sigma } _ { n } ^ { 2 }$ . The claimed result then follows because lim in $\mathrm { f } _ { n  \infty } \bar { \sigma } _ { n } ^ { 2 } > 0$ . □

Note that in the case of uniformly randomized trials $( \mathrm { i . e . , } ~ e _ { i } = \pi$ is the same for all units), the final obtained confidence interval construction (12.12) is exactly the same as (1.11) from Chapter 1.66 Earlier, we had shown (via a simple argument) that (1.11) is asymptotically exact for the ATE under IID sampling assumptions. It’s somewhat remarkable that, as found here, the same confidence interval is also asymptotically conservative for the SATE without making any sampling assumptions.

## 12.2 Confidence intervals for exposure effects

We now return to our main task of interest, i.e., inference for exposure effects as defined in (12.2). In addition to assuming a finite-cardinality exposure mapping, we will also assume network interference structure as in Definition 11.1, i.e., that each unit i has a known set $\mathcal { N } _ { i }$ of influencer units (or, informally friends), with $i \not \subset \mathcal { N } _ { i } \subset \{ 1 , . . . , n \}$ , such that

$$
Y _ {i} (\mathbf {w}) = Y _ {i} \left(\mathbf {w} ^ {\prime}\right) \text { whenever } w _ {i} = w _ {i} ^ {\prime} \text { and } w _ {j} = w _ {j} ^ {\prime} \text { for all } j \in \mathcal {N} _ {i}. \tag {12.14}
$$

In conjunction with (12.1), the condition (12.14) can be simplified to a requirement that $H _ { i }$ only depends on $w _ { i }$ and $\mathbf { w } _ { \mathcal { N } _ { i } }$ .

The two assumptions we make on the exposure mapping, (12.1) and (12.14), play different roles: (12.1) is primarily used to justify the estimands (and we will invoke it in a SUTVA-like manner), whereas (12.14) is used to control correlations and establish convergence properties for sample averages. In particular, the network interference model induces a natural randomization dependency graph G ∈ {0, 1}n×n $G \in \{ 0 , 1 \} ^ { n \times n }$ on potential outcomes,

$$
G _ {i j} = 1 \left(\{\mathcal {N} _ {i} \cup \{i \} \} \cap \{\mathcal {N} _ {j} \cup \{j \} \}\right) \neq \emptyset , \tag {12.15}
$$

i.e., $G _ { i j } = 1$ if and only if there is a unit $k \in \{ 1 , \ldots , n \}$ whose treatment can affect both $Y _ { i }$ and $Y _ { j }$ under (12.14).

Under Bernoulli randomization (12.3) and the network restriction (12.14), one can immediately verify that whenever $G _ { i j } = 0$ ,

$$
H _ {i} (\mathbf {W}) \perp H _ {j} (\mathbf {W}) \quad \text { and   so } \quad Y _ {i} \perp_ {W} Y _ {j}, \tag {12.16}
$$

where the latter statement means that $Y _ { i }$ is independent of $Y _ { j }$ under randomness from the treatment assignment (and either conditionally on potential outcomes or treating potential outcomes as fixed).

Given these ingredients, we are now ready to generalize the results from Section 12.1 to settings with interference, and provide both an exact expression for the variance of $\hat { \tau } _ { I P W } ( h , h ^ { \prime } )$ and a conservative but estimable bound for it. Here, we will start down by writing our variance estimator; our target variances will then be readily expressible in terms of moments of the variance estimator.

For any $ { \boldsymbol { h } } _ { \mathbf { \lambda } } \in  { \mathcal { H } }$ , define inverse-propensity weights as $\begin{array} { r l } { \Gamma _ { i } ( h ) } & { { } = } \end{array}$ $1 \left( \left\{ H _ { i } ( \mathbf { W } ) = h \right\} \right) / e _ { i } ( h )$ , and let $\mathbf { { \cal { F } } } ( h ) \in \mathbb { R } ^ { n }$ be the vector of these weights for all units. Given this notation and our exposure mapping,

$$
\hat {\tau} _ {I P W} (h, h ^ {\prime}) = \frac {1}{n} \sum_ {i = 1} ^ {n} (\Gamma_ {i} (h ^ {\prime}) Y _ {i} (h ^ {\prime}) - \Gamma_ {i} (h) Y _ {i} (h)), \tag {12.17}
$$

where only the weights $\Gamma _ { i }$ are taken to be random. This formulation, as well as the network independence property of the $\Gamma _ { i }$ established in (12.16), then suggests estimating the variance of the IPW estimator via the following a “heteroskedasticity and autocorrelation consistent” (HAC) construction:67

$$
\hat {\sigma} ^ {2} (h, h ^ {\prime}) = \frac {1}{n} \left(\boldsymbol {\Gamma} (h ^ {\prime}) \odot \mathbf {Y} - \boldsymbol {\Gamma} (h) \odot \mathbf {Y}\right) ^ {\top} G \left(\boldsymbol {\Gamma} (h ^ {\prime}) \odot \mathbf {Y} - \boldsymbol {\Gamma} (h) \odot \mathbf {Y}\right), (1 2. 1 8)
$$

where $\odot$ denotes elementwise product.68 The following result establishes that this variance estimate is in fact conservative.

Theorem 12.4. Under the setting of Theorem 12.1, suppose furthermore that (12.14) holds and that we consider a pair of exposure h, $h ^ { \prime } \in \mathcal { H }$ with $e _ { i } ( h ) , e _ { i } ( h ^ { \prime } ) > 0$ for all $i = 1 , \ldots , n$ . Write $\sigma ^ { 2 } ( h , h ^ { \prime } ) : = \mathbb { E } _ { W } \left[ \hat { \sigma } ^ { 2 } ( h , h ^ { \prime } ) \right]$ for the variance estimate given in (12.18), and $\bar { \sigma } ^ { 2 } ( h , h ^ { \prime } ) : = n \mathrm { V a r } _ { W } [ \hat { \tau } _ { I P W } ( h , h ^ { \prime } ) ]$ for the scaled randomization variance of the IPW estimator. Then,

$$
\bar {\sigma} ^ {2} (h, h ^ {\prime}) = \sigma^ {2} (h, h ^ {\prime}) - n ^ {- 1} (\mathbf {Y} (h ^ {\prime}) - \mathbf {Y} (h)) ^ {\top} G (\mathbf {Y} (h ^ {\prime}) - \mathbf {Y} (h)), \tag {12.19}
$$

and in particular $\bar { \sigma } ^ { 2 } ( h , h ^ { \prime } ) \leq \sigma ^ { 2 } ( h , h ^ { \prime } )$ .

Proof. Throughout this proof, we will use the shorthand $\begin{array} { r l } { \Gamma _ { i } ( h ) } & { { } = } \end{array}$ $1 \left( \left\{ H _ { i } ( \mathbf { W } ) = h \right\} \right) / e _ { i } ( h )$ for the inverse-propensity weights. Thanks to Theorem 12.1 and (12.1), we have

$$
\bar {\sigma} ^ {2} (h, h ^ {\prime}) := n \operatorname{Var} _ {W} \left[ \hat {\tau} _ {I P W} (h, h ^ {\prime}) \right] = n \mathbb {E} _ {W} \left[ \left(\hat {\tau} _ {I P W} (h, h ^ {\prime}) - \bar {\tau} (h, h ^ {\prime})\right) ^ {2} \right]
$$

$$
= n \mathbb {E} _ {W} \left[ \left(\left(\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\Gamma_ {i} (h ^ {\prime}) - \Gamma_ {i} (h)\right) Y _ {i} - \frac {1}{n} \sum_ {i = 1} ^ {n} \left(Y _ {i} (h ^ {\prime}) - Y _ {i} (h)\right)\right) ^ {2} \right] \right.
$$

$$
= n \mathbb {E} _ {W} \left[ \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\Gamma_ {i} (h ^ {\prime}) - 1\right) Y _ {i} (h ^ {\prime}) - \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\Gamma_ {i} (h) - 1\right) Y _ {i} (h)\right) ^ {2} \right].
$$

We can simplify this expression in terms of the exposure-covariance matrices

$$
U _ {i j} (h, h ^ {\prime}) = \mathbb {E} \left[ (\Gamma_ {i} (h) - 1) (\Gamma_ {j} (h ^ {\prime}) - 1) \right] = \mathbb {E} \left[ \Gamma_ {i} (h) \Gamma_ {j} (h ^ {\prime}) \right] - 1
$$

and $U ( h ) = U ( h , h )$ , etc., resulting in

$$
\begin{array}{l} \bar {\sigma} ^ {2} (h, h ^ {\prime}) = = n ^ {- 1} \mathbf {Y} (h) ^ {\top} U (h) \mathbf {Y} (h) + n ^ {- 1} \mathbf {Y} (h ^ {\prime}) ^ {\top} U (h ^ {\prime}) \mathbf {Y} (h ^ {\prime}) \\ - 2 n ^ {- 1} \mathbf {Y} (h) ^ {\top} U (h, h ^ {\prime}) \mathbf {Y} (h ^ {\prime}). \\ \end{array}
$$

We next turn to studying the expectation of the proposed variance estimate $\hat { \sigma } ^ { 2 } ( h , h ^ { \prime } )$ . A direct calculation shows that

$$
\sigma^ {2} (h, h ^ {\prime}) := \mathbb {E} _ {W} \left[ \hat {\sigma} ^ {2} (h, h ^ {\prime}) \right] = n ^ {- 1} \mathbf {Y} (h) ^ {\top} \mathbb {E} \left[ \boldsymbol {\Gamma} (h) ^ {\top} G \boldsymbol {\Gamma} (h) \right] \mathbf {Y} (h)
$$

$$
n ^ {- 1} \mathbf {Y} (h ^ {\prime}) ^ {\top} \mathbb {E} \left[ \boldsymbol {\Gamma} (h ^ {\prime}) ^ {\top} G \boldsymbol {\Gamma} (h ^ {\prime}) \right] \mathbf {Y} (h ^ {\prime}) + 2 n ^ {- 1} \mathbf {Y} (h) ^ {\top} \mathbb {E} \left[ \boldsymbol {\Gamma} (h) ^ {\top} G \boldsymbol {\Gamma} (h ^ {\prime}) \right] \mathbf {Y} (h ^ {\prime}).
$$

Furthermore, we see from (12.16) that

$$
U _ {i j} (h) = U _ {i j} (h ^ {\prime}) = U _ {i j} (h, h ^ {\prime}) = 0 \quad \text { whenever } \quad G _ {i j} = 0,
$$

and so we can re-express $\sigma ^ { 2 } ( h , h ^ { \prime } )$ in terms the exposure-covariance matrices used above as follows.

$$
\begin{array}{l} \sigma^ {2} (h, h ^ {\prime}) = = n ^ {- 1} \mathbf {Y} (h) ^ {\top} (U (h) + G) \mathbf {Y} (h) + n ^ {- 1} \mathbf {Y} \left(h ^ {\prime}\right) ^ {\top} (U \left(h ^ {\prime}\right) + G) \mathbf {Y} \left(h ^ {\prime}\right) \\ - 2 n ^ {- 1} \mathbf {Y} (h) ^ {\top} \left(U (h, h ^ {\prime}) + G\right) \mathbf {Y} (h ^ {\prime}). \\ \end{array}
$$

We can now compare our expressions for $\sigma ^ { 2 } ( h , h ^ { \prime } )$ and $\bar { \sigma } ^ { 2 } ( h , h ^ { \prime } )$ ,

$$
\sigma^ {2} (h, h ^ {\prime}) - \bar {\sigma} ^ {2} (h, h ^ {\prime}) = = n ^ {- 1} \mathbf {Y} (h) ^ {\top} G \mathbf {Y} (h) + n ^ {- 1} \mathbf {Y} (h ^ {\prime}) ^ {\top} G \mathbf {Y} (h ^ {\prime})
$$

$$
- 2 n ^ {- 1} \mathbf {Y} (h) ^ {\top} G \mathbf {Y} (h ^ {\prime})
$$

$$
= n ^ {- 1} \left(\mathbf {Y} (h ^ {\prime}) - \mathbf {Y} (h)\right) ^ {\top} G \left(\mathbf {Y} (h ^ {\prime}) - \mathbf {Y} (h)\right),
$$

and this quantity is non-negative because G is positive semi-definite.

Following our approach in the SUTVA case, we next consider the selfnormalized estimator,

$$
\hat {\tau} _ {S I P W} (h, h ^ {\prime}) = \frac {\sum_ {i = 1} ^ {n} \Gamma_ {i} (h ^ {\prime}) Y _ {i}}{\sum_ {i = 1} ^ {n} \Gamma_ {i} (h ^ {\prime})} - \frac {\sum_ {i = 1} ^ {n} \Gamma_ {i} (h) Y _ {i}}{\sum_ {i = 1} ^ {n} \Gamma_ {i} (h)}, \tag {12.20}
$$

and seek to establish a central limit theorem for it. As before, we work under a sequence of randomized trials with growing sample size n, and write

$$
\bar {\mu} _ {n} (h) = \frac {1}{n} \sum_ {i = 1} ^ {n} Y _ {i} (h), \quad \bar {\tau} _ {n} (h, h ^ {\prime}) = \bar {\mu} _ {n} (h ^ {\prime}) - \bar {\mu} _ {n} (h). \tag {12.21}
$$

We will also use a modified variance estimator that accounts for selfnormalization:

$$
\begin{array}{l} \hat {\mu} _ {n} (h) = \frac {1}{n} \sum_ {i = 1} ^ {n} \Gamma_ {i} (h) Y _ {i}, \\ \hat {\sigma} _ {n} ^ {2} (h, h ^ {\prime}) = (\boldsymbol {\Gamma} (h ^ {\prime}) \odot (\mathbf {Y} - \hat {\mu} _ {n} (h ^ {\prime})) - \boldsymbol {\Gamma} (h) \odot (\mathbf {Y} - \hat {\mu} _ {n} (h))) ^ {\top} G _ {n} \tag {12.22} \\ \left(\boldsymbol {\Gamma} \left(h ^ {\prime}\right) \odot \left(\mathbf {Y} - \hat {\mu} _ {n} \left(h ^ {\prime}\right)\right) - \boldsymbol {\Gamma} \left(h ^ {\prime}\right) \odot \left(\mathbf {Y} - \hat {\mu} _ {n} (h)\right)\right), \\ \end{array}
$$

where ${ \bf Y } - \hat { \mu } _ { n } ( h )$ subtracts the scalar $\hat { \mu } _ { n } ( h )$ from all entries of Y.

Theorem 12.5. Suppose we have a sequence of randomized trials with growing sample size n that all satisfy the conditions of Theorem $1 \mathcal { Q } . 4$ . Write $d e g ( G _ { n } )$ for the maximal degree of the randomization dependency graph in the n-th problem, and assume that lim $\iota _ { n \to \infty } n ^ { - 1 / 4 } d e g ( G _ { n } ) = 0$ . Suppose furthermore that there are constants $0 < \eta , M , s _ { 0 } ^ { 2 } < \infty$ such that $e _ { i } ( h ) , e _ { i } ( h ^ { \prime } ) \geq \eta$ and $| Y _ { i } ( h ) | , | Y _ { i } ( h ^ { \prime } ) | \leq$ M for all units throughout the sequence of problems, and that, using notation from (12.23), we have $\bar { \sigma } _ { n } ^ { 2 } ( h , h ^ { \prime } ) \geq s _ { 0 } ^ { 2 }$ for all n. Then,

$$
\begin{array}{l} \sqrt {n} \left(\frac {\hat {\tau} _ {S I P W} (h , h ^ {\prime}) - \bar {\tau} _ {n} (h , h ^ {\prime})}{\bar {\sigma} _ {n} (h , h ^ {\prime})}\right) \Rightarrow \mathcal {N} (0, 1) \\ \bar {\sigma} _ {n} ^ {2} \left(h, h ^ {\prime}\right) = \sigma_ {n} ^ {2} \left(h, h ^ {\prime}\right) - \left(\mathbf {Y} \left(h ^ {\prime}\right) - \bar {\mu} \left(h ^ {\prime}\right) - \mathbf {Y} (h) + \bar {\mu} (h)\right) ^ {\top} G _ {n} \tag {12.23} \\ \left(\mathbf {Y} \left(h ^ {\prime}\right) - \bar {\mu} \left(h ^ {\prime}\right) - \mathbf {Y} (h) + \bar {\mu} (h)\right), \\ \end{array}
$$

where $\sigma _ { n } ^ { 2 } ( h , h ^ { \prime } )$ denotes the randomization-expectation of an oracle version of $\hat { \sigma } _ { n } ^ { 2 } ( h , h ^ { \prime } )$ from (12.22) with $\hat { \mu } _ { n } ( h )$ replaced with $\bar { \mu } _ { n } ( h )$ , etc. Furthermore, our variance estimator is asymptotically conservative, lim $\mathrm { s u p } _ { n \to \infty } \bar { \sigma } _ { n } / \hat { \sigma } _ { n } \le _ { p } 1$ , and usual normal confidence intervals are valid

$$
\operatorname * {l i m s u p} _ {n \to \infty} \mathbb {P} \Big [ | \hat {\tau} _ {S I P W} (h, h ^ {\prime}) - \bar {\tau} _ {n} (h, h ^ {\prime}) | \tag {12.24}
$$

$$
\left. \leq \hat {\sigma} _ {n} (h, h ^ {\prime}) / \sqrt {n} \Phi^ {- 1} (1 - \alpha / 2) \right] \leq 1 - \alpha ,
$$

for any $0 < \alpha < 1$ .

Proof. We again start by noting that, thanks to self-normalization and our assumed exposure mapping,

$$
\hat {\tau} _ {S I P W} (h, h ^ {\prime}) = \bar {\tau} _ {n} (h, h ^ {\prime}) + \Delta (h ^ {\prime}) / \frac {1}{n} \sum_ {i = 1} ^ {n} \Gamma_ {i} (h ^ {\prime}) - \Delta (h) / \frac {1}{n} \sum_ {i = 1} ^ {n} \Gamma_ {i} (h)
$$

$$
\Delta (h) = \frac {1}{n} \sum_ {i = 1} ^ {n} \Gamma_ {i} (h) \left(Y _ {i} - \bar {\mu} _ {n} (h)\right).
$$

Theorems 12.1 and 12.4 immediately imply that, for all $n ,$

$$
\mathbb {E} _ {W} \left[ \Delta (h ^ {\prime}) - \Delta (h) \right] = 0, \quad \mathrm{Var} _ {W} \left[ \Delta (h ^ {\prime}) - \Delta (h) \right] = \frac {\bar {\sigma} _ {n} ^ {2} (h , h ^ {\prime})}{n}.
$$

Furthermore, Baldi and Rinott [1989, Corollary 2] provide a Berry–Esseen result for normal approximation of network-correlated random variables, which in our setting implies that

$$
\sup _ {z \in \mathbb {R}} \left| \mathbb {P} \left[ \frac {\sqrt {n} (\Delta (h ^ {\prime}) - \Delta (h))}{\bar {\sigma} _ {n} (h , h ^ {\prime})} \leq z \right] - \Phi (z) \right| \leq 3 2 \left(1 + \sqrt {6}\right) \sqrt {\frac {2 M}{\eta s _ {0} ^ {3}}} \frac {\deg (G _ {n})}{n ^ {1 / 4}}.
$$

Our assumption on the degree of $G _ { n }$ makes the right-hand side go to zero, and thus

$$
\frac {\sqrt {n} (\Delta (h ^ {\prime}) - \Delta (h))}{\bar {\sigma} _ {n} (h , h ^ {\prime})} \Rightarrow \mathcal {N} (0, 1).
$$

The remainder of the proof follows the blueprint of Theorem 12.3 and so is omitted; in particular, we note that our overlap assumption immediately implies that $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Gamma _ { i } ( h ) \to _ { p } 1 } \end{array}$ . □

Remark 12.1. When G has block structure, the variance estimator (12.22) is equivalent to usual cluster-robust inference variance estimator that is typically motivated using IID sampling assumptions (i.e., that clusters are sampled IID); see also Abadie et al. [2023]. Thus, we have recovered a conservativeness phenomenon analogous to the one derived by Neyman [1923] under SUTVA: Standard variance estimators motivated by IID sampling (here, of clusters) is conservative for the finite-population variance that arises from treatment randomization alone in the setting where potential outcomes are considered as deterministic.

Remark 12.2. The overlap assumption $e _ { i } ( h ) \geq \eta$ used in Theorem 12.5 essentially requires $\mathcal { N } _ { i }$ to be finite, even as the network grows (i.e., each unit is only influenced by the treatment given to a finite number of other units). However, even in this setting, the degree of $G$ can grow large: This can happen if there are some nodes that are very “popular”, in the sense that they influence many other nodes (i.e., they belong to ${ \mathcal { N } } _ { j }$ for many other units $j )$ . In this context, our assumption on $\deg ( G _ { n } )$ is essentially an upper bound on the strength of outward influence: We do not allow there to be a node whose treatment affects outcomes for more than $n ^ { 1 / 4 }$ other units.

## 12.3 Bibliographic notes

The finite-population model used in this chapter—as well as the approach to inference via conservative, identifiable variance bounds—goes back to Neyman [1923]. Here, we studied finite-population inference under Bernoulli trials; results under a number of different experimental designs is given in Li and Ding [2017]. We note that the variance bound used in Theorem 12.2 is not the only available bound; see Aronow, Green, and Lee [2014] for alternate proposals. Furthermore, the finite-population approach discussed here can also be extended to much more complex randomization designs, e.g., rerandomization as in Morgan and Rubin [2012].

Our approach to defining causal effects in terms of average outcomes under different exposure types builds on Aronow and Samii [2017]. Aronow and Samii [2017] also provided bounds on the variance of treatment effect estimators under the Neyman model; the bound we use in Theorem 12.4 is due to Leung [2022]. Building on this line of work, S¨avje [2024] discusses interpretation of exposure-averaging estimands when the exposure mapping may be misspecified, while Leung [2022] provides inference results under an approximate network interference model, where interference effects decay (but do not vanish) as units get farther from each other in a network. Viviano [2024] considers policy learning with interference under an exposure mapping assumption. Ogburn et al. [2024] consider inference from observational data under network interference. Harshaw, S¨avje, and Wang [2022] propose an algorithmic framework for producing IPW-like estimators for a number of causal target under wide variety models for interference.

Finally, we also note that there exist alternative ways of defining causal effects under interference that do not rely on well-specified exposure mappings. One such approach involves defining average direct and indirect effects of a treatment, which effectively measure how a unit getting treated affects the unit itself or others, while marginalizing over the treatment received by others [Halloran and Struchiner, 1995, Hu, Li, and Wager, 2022b, S¨avje, Aronow, and Hudgens, 2021]

$$
\tau_ {A D E} = \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} _ {W} \left[ Y _ {i} \left(w _ {i} = 1, W _ {- i}\right) - Y _ {i} \left(w _ {i} = 1, W _ {- i}\right) \right], \tag {12.25}
$$

$$
\tau_ {A I E} = \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {j \neq i} \mathbb {E} _ {W} \left[ Y _ {j} \left(w _ {i} = 1, W _ {- i}\right) - Y _ {j} \left(w _ {i} = 1, W _ {- i}\right) \right],
$$

where $Y _ { j } \left( w _ { i } = 1 , W _ { - i } \right)$ denotes the outcome we observe for the j-th unit by setting the i-th treatment to 1 but letting others be as they are under the randomization distribution. Hu, Li, and Wager [2022b] interpret these estimands in the context of a number of models for interference, and connect them to notions of total treatment effects. S¨avje, Aronow, and Hudgens [2021] provide bounds for the average direct effect under a generic interference model, while Li and Wager [2022] give exact large-sample asymptotics for the average direct and indirect effects under a random graph generative model. Munro, Kuang, and Wager [2021] consider large-sample behavior of the average direct and indirect effects in a model where interference arises via equilibrium effects where in a marketplace where prices align supply and demand; they also propose CATElike measures for treatment heterogeneity that can be used for spillover-aware targeting.
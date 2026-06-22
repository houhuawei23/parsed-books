# Chapter 8 Regression Discontinuity Designs

The cleanest and most straight-forward approach to treatment effect estimation is using approaches justified by random treatment assignment—where randomization can either be explicit (as in randomized controlled trials) or implicit (as in observational study analyses under an unconfoundedness assumption). All methods discussed in the book so far fall within this category.

In applied work, however, there’s also often interest in drawing causal inferences using data where it is not realistic to assume that treatment is as good as random (even after controlling for observed pre-treatment covariates), and there exist a number of widely used econometric methods for identifying and estimating causal effects in settings without random treatment assignment. This chapter—as well as the following ones—will provide a brief introduction to such quasi-experimental approaches to causal inference. We use the term “quasi experimental” to emphasize that these approaches are still framed using concepts from randomized experiments—such as potential outcomes and average treatment effects—but require econometric innovations to compensate for the lack of random treatment assignment.

Setting and notation This chapter is about the regression discontinuity design (RDD), which is a simple and widely used quasi-experimental design. In a simple RDD, we are interested in the effect of a binary treatment $W _ { i }$ on a real-valued outcome $Y _ { i } ,$ and posit potential outcomes $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \}$ such that $Y _ { i } = Y _ { i } ( W _ { i } )$ . However, unlike in a randomized trial, we do not take the treatment assignment $W _ { i }$ to be random. Instead, we assume there is a running variable $Z _ { i } \in \mathbb { R }$ and a cutoff $c ,$ such that $W _ { i } = 1 \left( \left\{ Z _ { i } \geq c \right\} \right)$ . This setting could arise, $\mathrm { e . g . }$ , in education, where $Z _ { i }$ is a standardized test score and students with $Z _ { i } \geq c$ are eligible to enroll in an honors program, or in medicine, where $Z _ { i }$ is a severity score, and patients are prescribed an intervention once $Z _ { i } \geq c$ .

Qualitatively, the main idea of a regression discontinuity is that although treatment assignment $W _ { i }$ is not randomized, it’s almost as good as random

<!-- footnote -->

- The term Var $\left[ \tau ( X _ { i } ) \right]$ in $V _ { A I P W }$ (3.5) vanishes here because the CATE is constant.
- A risk of using the residual-on-residual estimator, of course, is that the constant treatment effect model (4.10) may be misspecified. We examine what happens to the residualon-residual estimator under misspecification in Exercise 5 in Chapter 16.

<!-- footnote end -->

<!-- footnote -->

- If true propensity scores $e ( x )$ are known, they can (and should) be used instead of the cross-fitted estimates $\hat { e } ^ { ( - k ) } ( x )$ .

<!-- footnote end -->

<!-- footnote -->

- Here, since we are evaluating our loss-function $\hat { \ell } ( \cdot )$ on fresh data, we no longer need cross-fitting to avoid overfitting problems. In practice, of course, one needs to choose which version of $\hat { \ell } ( \cdot )$ one uses on the development set; one simple and reasonable approach is $\begin{array} { r } { \hat { \ell } ( \cdot ) = K ^ { - 1 } \sum _ { k = 1 } ^ { K } \hat { \ell } ^ { ( - k ) } ( \cdot ) } \end{array}$ dual cross-fit loss functions produced on the training set, and use.
- For a presentation that explicitly presents causal forests as a type of R-learner, see Athey and Wager [2019].

<!-- footnote end -->

<!-- footnote -->

- We expand all features into 7th order natural cubic splines using the R-command ns, and then take full 2nd order interactions between these spline terms.

<!-- footnote end -->

<!-- footnote -->

- In some applications $( \mathrm { e . g . }$ when a budget constraint needs to be satisfied exactly) it is helpful to consider randomized policies $\pi : \mathcal { X }  [ 0 , 1 ]$ , where a non-integer value of $\pi ( x )$ is interpreted as a treatment probability. Results discussed here extend directly to this setting.

<!-- footnote end -->

<!-- footnote -->

- We recognize that the CATE likely non-linear here, but for practical reasons we still seek the welfare-maximizing linear thresholding rule (that is learned in a way that allows for non-linearity in the CATE).

<!-- footnote end -->

<!-- footnote -->

- The authors prove that the functional dependence of the bound (5.20) on the problem primitives is the best possible, and the constant is loose by a factor at most 200.

<!-- footnote end -->

<!-- footnote -->

- As a further note of caution: We’ve shown that policy learning via empirical maximization is computationally equivalent to weighted optimization of a classification objective. In many applications, however, practitioners carry out classification by optimization a surrogate objective (rather than the original classification objective), $\mathrm { e . g . }$ , using the hinge or logistic loss, and it may be tempting to also apply similar approximations to (5.24). The guarantees presented here, however, do not in general extend to surrogate objectives. For example, it’s possible to design situations where learning with a “logistic” surrogate for (5.24) makes us prioritize people who would benefit the least from treatment (rather than the most); see Wager [2019] for a discussion.

<!-- footnote end -->

<!-- footnote -->

- $\begin{array} { r } { R _ { T } ^ { Y } = \sum _ { t = 1 } ^ { T } \left( Y _ { t } ( k ^ { * } ) - Y _ { t } \right) } \end{array}$ $k ^ { * }$ with $\mu _ { k ^ { * } } = \mu ^ { * }$ . However, because the actions $W _ { t }$ only depend on past data, the difference in summands $Y _ { t } ( k ^ { * } ) - Y _ { t } - \left( \mu ^ { * } - \mu _ { W _ { t } } \right)$ form a martingale difference sequence—and so $R _ { T }$ and $R _ { T } ^ { Y }$ have the same expectation. By the same argument, one can see that the difference between $R _ { T } ^ { Y } - R _ { T }$ is pure noise that is not under the experimenter’s control. In our discussion here, we will focus on $R _ { T }$ and call it “regret”, as this most accurately quantifies the consequences of the actions taken by the experimenter.

<!-- footnote end -->

<!-- footnote -->

- The argument remains valid for sub-Gaussian outcomes with known scale parameter σ.

<!-- footnote end -->

<!-- footnote -->

- The argument is exactly the same—but just with more notation—if we allow for multiple optimal arms.

<!-- footnote end -->

<!-- footnote -->

- On careful examination, it turns out that using an improper prior for Thompson sampling is not just a simple generic choice, but can be a quasi-optimal choice from the perspective of regret minimization [Kuang and Wager, 2024].

<!-- footnote end -->

<!-- footnote -->

- Note that the condition that $e _ { t , k } > 0$ can in fact be omitted from the theorem statement at the cost of some extra bookkeeping in the proof and under the convention that $0 / 0 = 0$ . The Lindeberg-type condition (6.14) on its own already provides sufficient control on the decay of the treatment assignment probabilities.

<!-- footnote end -->

<!-- footnote -->

- The existence of such basis representations is well known in many contexts; for example, functions of bounded variation on a compact interval can be represented in terms of a Fourier series. Here we will not review when such representations are available; instead, we assume that an appropriate series representation is given.

<!-- footnote end -->

<!-- footnote -->

- This exponential moment condition is generally weaker than the strong overlap assumption made in Chapter 3. Note that, under the propensity model used here, strong overlap would follow from assuming that $\| X _ { i } \|$ is uniformly bounded.
- The fact that E $\left[ e ( X _ { i } ) X _ { i } ^ { \otimes 2 } \right] ~ \succ ~ 0$ follows immediately from our assumption that E $[ X _ { i } ^ { \otimes 2 } ] \succ 0$ and the fact that $0 < e ( X _ { i } ) < 1$ almost surely in our setting.

<!-- footnote end -->

<!-- footnote -->

- See Hirshberg and Wager [2021] for conditions under which the $\hat { \gamma } ^ { ( w ) }$ are consistent for the inverse-propensity weights, and thus $| E | \ll 1 / \sqrt { n }$ together with Lemma 7.2 imply efficiency in the sense discussed in Chapter 3.

<!-- footnote end -->

<!-- footnote -->

- It is also interesting to note that, if we use the exact balancing construction (7.22) and omit the positivity constraint $\gamma _ { i } \geq 0$ , then the induced IPW-type estimator (7.21) is numerically equivalent to the interacted OLS regression estimator (1.14). This connection can be proven directly using elementary techniques; one can also argue for this connection by noting that it is equivalent to the Gauss-Markov theorem.
- One finite-sample consideration with this approach is that one may end up with regions

<!-- footnote end -->

<!-- footnote -->

- with only treated (or control) observations, and such regions cannot be balanced. Thus, data in such regions needs to be discarded, resulting in a loss of power—and potentially also bias.

<!-- footnote end -->

when $Z _ { i }$ is in the vicinity of the cutoff $c .$ People with $Z _ { i }$ close to $c$ ought to all be similar to each other on average, but only those with $Z _ { i } \geq c$ get treated, and so we can estimate a treatment effect by comparing people with $Z _ { i }$ right above versus right below 0.

Example 7. Lee [2008] studies incumbency advantage in US House elections by examining close elections. He compares the probability that a given political party wins a House seat in an election cycle when they just barely won that seat in the previous cycle vs. when they just barely lost. Validity of this approach hinges on an understanding that results of close elections are unpredictable and subject to idiosyncratic factors (e.g., perhaps a rain storm on election day caused differential attrition in turnout that moved the two-party vote share by a small amount), and that congressional districts where one party won, say, 51% vs. 49% of the two-party vote should have roughly the same distribution of potential confounding factors. Then, once we’ve established that such congressional districts are ex-ante comparable, we can obtain valid causal estimates via the regression-discontinuity approach.

Why propensity score methods can’t be used in RDDs Before discussing methods for estimation in regression discontinuity designs, it’s helpful to consider why our previously considered approaches (such as IPW) don’t apply. As emphasized in our discussion so far, the two assumptions invariably required for propensity-score methods to work are:

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp W _ {i} \mid Z _ {i}, \quad \text { unconfoundedness,   and } \tag {8.1}
$$

$$
0 <   \mathbb {P} \left[ W _ {i} = 1 \mid Z _ {i} \right] <   1, \quad \text { overlap. } \tag {8.2}
$$

Taken together, unconfoundedness and overlap mean that we can view our dataset as formed by pooling many small randomized trials indexed by different values of $Z _ { i } ;$ then, unconfoundedness means that treatment assignment is exogenous given $Z _ { i } .$ , while overlap means that randomization in fact occurred (one can’t learn anything from a randomized trial where everyone is assigned to to the same treatment arm).

In a regression discontinuity design, we have $W _ { i } = 1 \left( \left\{ Z _ { i } \geq c \right\} \right)$ , and so unconfoundedness holds trivially (because $W _ { i }$ is a deterministic function of $Z _ { i } )$ . However, overlap clearly doesn’t hold: $\mathbb { P } \left[ W _ { i } = 1 \big | Z _ { i } = z \right]$ is always either 0 or 1. Thus, methods like IPW that involve division by $\mathbb { P } \left[ W _ { i } \overline { { = } } 1 | Z _ { i } \right]$ , etc., are not applicable. Instead, we’ll need to compare units with $Z _ { i }$ straddling the cutoff c that are similar to each other—but do not have contiguous distributions.

## 8.1 Local linear regression

The most prevalent way to formalize the qualitative argument underlying RDD is by invoking continuity. Let $\mu _ { ( w ) } ( z ) = \mathbb { E } \left[ Y _ { i } ( w ) \big | Z _ { i } \right]$ . Then, if $\mu _ { ( 0 ) } ( z )$ and $\mu _ { ( 1 ) } ( z )$ are both continuous, we can identify the conditional average treatment effect at $z = c , { \mathrm { i . e . , } } \tau _ { c } = \mu _ { ( 1 ) } ( c ) - \mu _ { ( 0 ) } ( c )$ , via

$$
\tau_ {c} = \lim _ {z \downarrow c} \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = z \right] - \lim _ {z \uparrow c} \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = z \right], \tag {8.3}
$$

provided that the running variable $Z _ { i }$ has support around the cutoff $c .$ In other words, we identify $\tau _ { c }$ as the difference between the endpoints of two different regression curves; the above figure provides an illustration.

Estimation via local linear regression A simple and robust approach to estimation based on (8.3) is to use local linear regression, as illustrated in Figure 8.1. We pick a small bandwidth $h _ { n } \to 0$ and a symmetric weighting function $K ( \cdot )$ , and then fit $\mu _ { ( w ) } ( z )$ via weighted linear regression on each side of the boundary,

$$
\begin{array}{l} \hat {\tau} _ {c} = \operatorname{argmin} \left\{\sum_ {i = 1} ^ {n} K \left(\frac {| Z _ {i} - c |}{h _ {n}}\right) \right. \tag {8.4} \\ \left. \times \left(Y _ {i} - a - \tau W _ {i} - \beta_ {(0)} (Z _ {i} - c) _ {-} - \beta_ {(1)} (Z _ {i} - c) _ {+}\right) ^ {2} \right\}, \\ \end{array}
$$

where the overall intercept a and slope parameters $\beta _ { ( w ) }$ are nuisance parameters. Popular choices for the weighting function $K ( x )$ include the window function $K ( x ) = 1 \left( \left\{ | x | \leq 1 \right\} \right)$ , or the triangular kernel $K ( x ) = ( 1 - | x | ) _ { + }$ .

Consistency, asymptotics and rates of convergence It is not hard to see that, under continuity assumptions as in (8.3), the local linear regression estimator (8.4) must be consistent for reasonable choices of the bandwidth sequence $h _ { n }$ . However, in order to move beyond such a high-level statement and get any quantitative guarantees, we need to be more specific about the continuity assumptions made on $\mu _ { ( 0 ) } ( z )$ and $\mu _ { ( 1 ) } ( z )$ .

There are many ways of quantifying smoothness, but one of the most widely used assumptions in practice—and the one we’ll focus on today—is that the $\mu _ { ( w ) } ( z )$ are twice differentiable with a uniformly bounded second derivative

$$
\left| \frac {d ^ {2}}{d z ^ {2}} \mu_ {(w)} (z) \right| \leq B \text {   for   all   } z \in \mathbb {R} \text {   and   } w \in \{0, 1 \}. \tag {8.5}
$$

One motivation for the assumption (8.5) is that it justifies local linear regression as in (8.4): If we had less smoothness $( \mathrm { e . g . } , \mu _ { ( w ) } ( z )$ is just taken to be Lipschitz) then there would be no point doing local linear regression as opposed to local averaging, whereas if we had more smoothness (e.g., bounds on the k-th order derivative of $\mu _ { ( w ) } ( z )$ for $k \geq 3 )$ ) then we could improve rates of convergence via local regression with higher-order polynomials.

Given this assumption, we can directly bound the error rate of (8.4). The following result gives the rate of convergence of local linear regression along with a proof sketch. We refer to Imbens and Kalyanaraman [2012] for a more precise argument, along with guidance on how to choose the scale parameter κ for the bandwidth $h _ { n }$ .

Proposition 8.1. Consider an RDD where the running variable has a continuous distribution around the cutoff, and Var $\left\lceil Y _ { i } \right\rceil Z _ { i } = z ] \leq \sigma ^ { 2 }$ for all z. Suppose furthermore that (8.5) holds for some $B > 0$ . Then, the local linear regression estimator (8.4) with bandwidth $h _ { n } = \kappa n ^ { - 1 / 5 }$ for some $\kappa > 0$ is consistent, andhas errors scaling as

$$
\hat {\tau} _ {c} = \tau_ {c} + \mathcal {O} _ {P} \left(n ^ {- 2 / 5}\right). \tag {8.6}
$$

Proof sketch. We start by taking a Taylor expansion around c, which yields

$$
\mu_ {(w)} (z) = a _ {(w)} + \beta_ {(w)} (z - c) + \frac {1}{2} \rho_ {(w)} (z - c), \quad \left| \rho_ {(w)} (x) \right| \leq B x ^ {2}, \tag {8.7}
$$

while noting that $\tau _ { c } = a _ { ( 1 ) } - a _ { ( 0 ) }$ . Moreover, by inspection of the problem (8.4), we see that it factors into two separate regression problems on the treated and control samples, namely

$$
\hat {a} _ {(1)}, \hat {\beta} _ {(1)} = \operatorname{argmin} _ {a, \beta} \left\{\sum_ {Z _ {i} \geq c} K \left(\frac {| Z _ {i} - c |}{h _ {n}}\right) (Y _ {i} - a - \beta (Z _ {i} - c)) ^ {2} \right\}, \tag {8.8}
$$

for the treated units and an analogous problem for the controls, such that $\hat { \tau } = \hat { a } _ { ( 1 ) } - \hat { a } _ { ( 0 ) }$ .

Now, for simplicity, we focus on local linear regression with the basic window kernel $K ( x ) = 1 ( \{ | x | \leq 1 \} )$ . The linear regression problem (8.8) can then be solved in closed form, and we get

$$
\hat {a} _ {(1)} = \sum_ {c \leq Z _ {i} \leq c + h _ {n}} \gamma_ {i} Y _ {i}, \quad \gamma_ {i} = \frac {\widehat {\mathbb {E}} _ {(1)} \left[ (Z _ {i} - c) ^ {2} \right] - \widehat {\mathbb {E}} _ {(1)} [ Z _ {i} - c ] \cdot (Z _ {i} - c)}{\widehat {\mathbb {E}} _ {(1)} \left[ (Z _ {i} - c) ^ {2} \right] - \widehat {\mathbb {E}} _ {(1)} [ Z _ {i} - c ] ^ {2}}, \tag {8.9}
$$

where $\begin{array} { r } { \widehat { \mathbb { E } } _ { ( 1 ) } \left[ Z _ { i } - c \right] = \sum _ { c < Z _ { i } < c + h _ { n } } ( Z _ { i } - c ) / \left| \{ i : c \leq Z _ { i } \leq c + h _ { n } \} \right| } \end{array}$ , etc., denote sample averages over the regression window. Direct calculation reveals that $\begin{array} { r } { \sum _ { c \leq Z _ { i } \leq c + h _ { n } } \gamma _ { i } = 1 } \end{array}$ and $\begin{array} { r } { \sum _ { c \leq Z _ { i } \leq c + h _ { n } } \gamma _ { i } ( Z _ { i } - c ) = 0 } \end{array}$ , and so by (8.7)

$$
\hat {a} _ {(1)} = a _ {(1)} + \underbrace {\sum_ {c \leq Z _ {i} \leq c + h _ {n}} \gamma_ {i} \rho_ {(1)} (Z _ {i} - c)} _ {\text {curvature bias}} + \underbrace {\sum_ {c \leq Z _ {i} \leq c + h _ {n}} \gamma_ {i} \left(Y _ {i} - \mu_ {(1)} (Z _ {i})\right)} _ {\text {sampling noise}}, \tag {8.10}
$$

and a similar expansion holds for $\hat { a } _ { ( 0 ) }$ . Thus, recalling that our estimator is $\hat { \tau } = \hat { a } _ { ( 1 ) } - \hat { a } _ { ( 0 ) }$ and out target estimand is $\tau _ { c } = a _ { ( 1 ) } - a _ { ( 0 ) }$ , we see that it suffices to bound the error terms in (8.10).

Given our bias on the curvature, we immediately see that the “curvature bias” term is bounded by $B h _ { n } ^ { 2 }$ . Meanwhile, the sampling noise term is meanzero and, provided that Var $\left[ \ddot { Y _ { i } } | Z _ { i } \right] \le \sigma ^ { 2 }$ , has variance bounded on the order of $\begin{array} { r } { \sigma ^ { 2 } \sum _ { c \leq Z _ { i } \leq c + h _ { n } } \gamma _ { i } ^ { 2 } } \end{array}$ . Finally, assuming that $Z _ { i }$ has a continuous non-zero density function $f ( z )$ in a neighborhood of $z ,$ , one can check that

$$
\sigma^ {2} \sum_ {c \leq Z _ {i} \leq c + h _ {n}} \gamma_ {i} ^ {2} \approx \frac {4 \sigma^ {2}}{| \{i : c \leq Z _ {i} \leq c + h _ {n} \} |} \approx \frac {4 \sigma^ {2}}{f (c)} \frac {1}{n h _ {n}}. \tag {8.11}
$$

The squared bias of $\hat { \tau }$ thus scales as $h _ { n } ^ { 4 }$ , while its variance scales as $1 / ( h _ { n } n )$ . The bias-variance trade-off is minimized at $h _ { n } \sim n ^ { - 1 / 5 }$ , resulting in (8.6).

Remark 8.1. The $n ^ { - 2 / 5 }$ rate is a consequence of working with bounds on the 2nd derivative of $\mu _ { ( w ) } ( z )$ . In general, if we assume that $\mu _ { ( w ) } ( z )$ has a bounded k-th order derivative, then we can achieve an $n ^ { - k / ( 2 k + \dot { 1 } ) }$ rate of convergence for $\tau _ { c }$ by using local polynomial regression of order $( k - 1 )$ with a bandwidth scaling as $h _ { n } \sim n ^ { - 1 / ( 2 k + 1 ) }$ .46 Local linear regression never achieves a parametric rate of convergence, but can get close if $\mu _ { ( w ) } ( z )$ is very smooth.

Remark 8.2. While Proposition 8.1 provides bounds on the estimation error of local linear regression, it does not directly induce a method for inference about $\tau _ { c }$ . This is because, when using a bandwidth that scales at the estimation-erroroptimal rate $h _ { n } \sim n ^ { - 1 / 5 }$ , both the bias and standard error of $\hat { \tau } _ { c }$ . This means that standard tools for building confidence intervals using linear regression— which only account for variance but not bias—will understate the size of the errors in $\hat { \tau } _ { c }$ and generally not achieve nominal coverage rates. One simple way to address this challenges is to rely on “undersmoothing”, and pick $h _ { n } \ll n ^ { - 1 / 5 }$ so that variance dominates bias. This strategy, however, is generally not recommended, as undersmoothing results in larger-than-optimal estimation error; and furthermore it is challenging to choose an undersmoothing bandwidth in such a way as to credibly get good coverage in finite samples. A better approach is to use bias-corrections that leverage higher-order smoothness; discussing how to do so is however beyond the scope of this presentation, and we instead refer to Calonico, Cattaneo, and Titiunik [2014] for details on this approach.

## 8.2 Optimized estimation and bias-aware inference

We showed above that the conditional expectation functions have bounded curvature as in (8.5) and $Z _ { i }$ has a continuous non-zero density around c (meaning that there will asymptotically be datapoints with $Z _ { i }$ arbitrarily close to c), then local linear regression can estimate $\tau _ { c }$ in an RDD with errors that decay as $n ^ { - 2 / 5 }$ . Now, while this result is helpful conceptually and also motivates a simple estimator, some applications have features that preclude direct application of this result. First, the asymptotic argument underlying (8.3) relies on observing data $Z _ { i }$ arbitrarily close to the cutoff $c .$ In practice, however, we often have to work with discrete running variables $\left( \mathrm { e . g . , ~ } Z _ { i } \right.$ is a test score that takes integers value between 0 and 100), and in these cases the asymptotics underlying Proposition 8.1 do not apply. Moreover, in many applications, we need to work with more complicated cutoff functions (e.g., a student needs to pass 2 out of 3 tests to be eligible for a program), and it is not immediately clear how to adapt local linear regression to such settings in a way that preserves statistical power.

Linear estimators for RDD In order to address these challenges and develop estimators for a more general class of RDDs, we start with an abstract observation. In the proof of Proposition 8.1, we noted that we can write the local linear estimator as

$$
\hat {\tau} _ {c} (\gamma) = \sum_ {i = 1} ^ {n} \gamma_ {i} Y _ {i}. \tag {8.12}
$$

for some weights $\gamma _ { i }$ that only depend on the running variable $Z _ { i } ;$ ; the specific form of the weights induced by local linear regession with a window kernel $K ( x ) = 1 ( \{ | x | \leq 1 \} )$ ) is given in (8.9). We refer to estimators of this form as linear estimators because they are linear functions of the outome vector Y .47

Now, although the local linear regression estimator (8.4) was motivated by a regression problem, we didn’t make much use of this regression formulation in studying $\hat { \tau } _ { c }$ . Instead, for our formal discussion, we just used general properties of that hold for all linear estimators of the form (8.12).

For simplicity, consider for now a setting with homoskedatic and Gaussian errors, such that $Y _ { i } ( w ) = \mu _ { ( w ) } ( Z _ { i } ) + \varepsilon _ { i } ( w )$ with $\varepsilon _ { i } ( w ) \mid Z _ { i } \sim \mathcal { N } \left( 0 , \sigma ^ { 2 } \right)$ . Then, any linear estimator (8.12) whose weights $\gamma _ { i }$ are only functions of the $Z _ { i }$ satisfies

$$
\hat {\tau} _ {c} (\gamma) \mid \{Z _ {1}, \dots , Z _ {n} \} \sim \mathcal {N} \left(\hat {\tau} _ {c} ^ {*} (\gamma), \sigma^ {2} \| \gamma \| _ {2} ^ {2}\right),
$$

$$
\hat {\tau} _ {c} ^ {*} (\gamma) = \sum_ {i = 1} ^ {n} \gamma_ {i} \mu_ {W _ {i}} (Z _ {i}), \tag {8.13}
$$

where $W _ { i } = 1 \left( \left\{ Z _ { i } \geq c \right\} \right)$ . Thus, we immediately see that any linear estimator as in (8.12) will be an accurate estimator for $\tau _ { c }$ provided we can guarantee that $\hat { \tau } _ { c } ^ { * } \left( \gamma \right) \approx \tau _ { c }$ and $\| \gamma \| _ { 2 } ^ { 2 }$ is small.

Minimax linear estimation Motivated by this observation, it’s natural to ask: If the salient fact about local linear regression (8.4) is that we can write it as an linear estimator of the form (8.12), then is local linear regression the best estimator in this class? As we’ll see below, the answer is no; however, the best estimator of the form (8.12) can readily be derived in practice via numerical convex optimization.

As noted in (8.13), the conditional variance of any linear estimator can directly be observed: it’s just $\sigma ^ { 2 } \left\| \gamma \right\| _ { 2 } ^ { 2 }$ (again, for simplicity, we’re working with homoskedatic errors for most of today). In contrast, the bias of linear estimators depends on the unknown functions $\mu _ { ( w ) } ( z )$ , and so cannot be observed:

$$
\operatorname{Bias} \left(\hat {\tau} _ {c} (\gamma) \mid \{Z _ {1},..., Z _ {n} \}\right) = \sum_ {i = 1} ^ {n} \gamma_ {i} \mu_ {W _ {i}} (Z _ {i}) - \left(\mu_ {(1)} (c) - \mu_ {(0)} (c)\right). \tag {8.14}
$$

However, although, this bias is unknown, it can still readily be bounded given smoothness assumptions on the $\mu _ { ( w ) } ( z )$ . For example, if the curvature of $\mu _ { ( w ) } ( z )$ is assumed to be bounded by B as in (8.5), then48

$$
\begin{array}{l} \left| \operatorname{Bias} \left(\hat {\tau} _ {c} (\gamma) \mid \{Z _ {1}, \dots , Z _ {n} \}\right) \right| \leq I _ {B} (\gamma) \\ I _ {B} (\gamma) = \sup \left\{\sum_ {i = 1} ^ {n} \gamma_ {i} \mu_ {W _ {i}} (Z _ {i}) - \left(\mu_ {(1)} (c) - \mu_ {(0)} (c)\right): \left| \mu_ {(w)} ^ {\prime \prime} (z) \right| \leq B \right\}. \tag {8.15} \\ \end{array}
$$

Now, recall that the mean-squared error of an estimator is just the sum of its variance and squared bias. Because the variance term $\sigma ^ { 2 } \left\| \gamma \right\| _ { 2 } ^ { 2 }$ doesn’t depend on the conditional response functions, we thus see that the worst-case mean squared error of any linear estimator over all problems with $| \mu _ { ( w ) } ^ { \prime \prime } ( z ) | \le B$ is just the sum of its variance and worst-case bias squared, i.e.,

$$
\mathrm{MSE} \left(\hat {\tau} _ {c} (\gamma) \mid \{Z _ {1},..., Z _ {n} \}\right) \leq \sigma^ {2} \| \gamma \| _ {2} ^ {2} + I _ {B} ^ {2} (\gamma), \tag {8.16}
$$

with equality at any function that attains the worst-case bias (8.15).

It follows that, under an assumption that $| \mu _ { ( w ) } ^ { \prime \prime } ( z ) | \le B$ and conditionally on $\left\{ Z _ { 1 } , . . . , Z _ { n } \right\}$ , the minimax linear estimator of the form (8.12) is the one that minimizes (8.16):

$$
\hat {\tau} _ {c} \left(\gamma^ {B}\right) = \sum_ {i = 1} ^ {n} \gamma_ {i} ^ {B} Y _ {i}, \quad \gamma^ {B} = \operatorname{argmin} \left\{\sigma^ {2} \| \gamma \| _ {2} ^ {2} + I _ {B} ^ {2} (\gamma) \right\}. \tag {8.17}
$$

One can check numerically that the weights implied by local linear regression do not solve this optimization problem, and so the estimator (8.17) dominates local linear regression in terms of worst-case MSE.

Deriving the minimax linear weights Of course, the estimator (8.17) is not of much use unless we can solve for the weights $\gamma _ { i } ^ { B }$ in practice. Luckily, we can do so via routine quadratic programming. To do so, it is helpful to write

$$
\mu_ {(w)} (z) = a _ {(w)} + \beta_ {(w)} (z - c) + \rho_ {(w)} (z), \tag {8.18}
$$

where $\rho _ { ( w ) } ( z )$ is a function with $\rho _ { ( w ) } ( c ) = \rho _ { ( w ) } ^ { \prime } ( c ) = 0$ and whose second derivative is bounded by $B ;$ given this representation $\tau _ { c } = a _ { ( 1 ) } - a _ { ( 0 ) }$ .

Now, the first thing to note in (8.18) is that the coefficients $a _ { ( w ) }$ and $\beta _ { ( w ) }$ are unrestricted. Thus, unless the weights $\gamma _ { i }$ account for them exactly, such that

$$
\sum_ {i = 1} ^ {n} \gamma_ {i} W _ {i} = 1, \sum_ {i = 1} ^ {n} \gamma_ {i} = 0, \sum_ {i = 1} ^ {n} \gamma_ {i} (Z _ {i} - c) _ {+} = 0, \sum_ {i = 1} ^ {n} \gamma_ {i} (Z _ {i} - c) _ {-} = 0,
$$

we can choose $a _ { ( w ) }$ and $\beta _ { ( w ) }$ to make the bias of $\hat { \tau } _ { c } ( \gamma )$ arbitrarily bad $( \mathrm { i . e . }$ , $I _ { B } ( \gamma ) = \infty )$ . Meanwhile, once we enforce these constraints, it only remains to bound the bias due to $\rho _ { ( w ) } ( z )$ , and so we can re-write (8.17) as

$$
\left\{\gamma^ {B}, t \right\} = \mathrm{argmin} \quad \sigma^ {2} \left\| \gamma \right\| _ {2} ^ {2} + B ^ {2} t ^ {2}
$$

$$
\text { subject   to: } \sum_ {i = 1} ^ {n} \gamma_ {i} W _ {i} \rho_ {(1)} (Z _ {i}) + \sum_ {i = 1} ^ {n} \gamma_ {i} (1 - W _ {i}) \rho_ {(0)} (Z _ {i}) \leq t
$$

$$
\text { for   all } \rho_ {(w)} (\cdot) \text { with } \rho_ {(w)} (c) = \rho_ {(w)} ^ {\prime} (c) = 0
$$

$$
\text { and } \left| \rho_ {(w)} ^ {\prime \prime} (z) \right| \leq 1 \tag {8.19}
$$

$$
\sum_ {i = 1} ^ {n} \gamma_ {i} W _ {i} = 1, \sum_ {i = 1} ^ {n} \gamma_ {i} = 0,
$$

$$
\sum_ {i = 1} ^ {n} \gamma_ {i} W _ {i} (Z _ {i} - c) = 0, \sum_ {i = 1} ^ {n} \gamma_ {i} (Z _ {i} - c) = 0.
$$

Given this form, the optimization should hopefully look like a tractable one. And in fact it is: The problem simplifies once we take its dual, and it can then be well approximated by a finite-dimensional quadratic program where we use a discrete approximation to the set of functions with second derivative bounded by 1; see Imbens and Wager [2019, Section II.B] for details.

Bias-aware inference The above discussion suggests that using an estima-$\begin{array} { r } { \hat { \tau } _ { c } \left( \gamma ^ { B } \right) = \sum _ { i = 1 } ^ { n } \gamma _ { i } ^ { B } Y _ { i } } \end{array}$ estimate for for $\tau _ { c }$ if all we know is that $| \mu _ { ( w ) } ^ { \prime \prime } ( z ) | \le B$ . In particular, under this assumption and conditionally on $\left\{ Z _ { 1 } , . . . , Z _ { n } \right\}$ , it attains minimax meansquared error among all linear estimators. Because local linear regression is also a linear estimator, we thus find that $\hat { \tau } _ { c } \left( \gamma ^ { B } \right)$ dominates local linear regression in a minimax sense.

If we want to use $\hat { \tau } _ { c } \left( \gamma ^ { B } \right)$ in practice, though, it’s important to be able to also provide confidence intervals for $\tau _ { c } .$ . And, since $\hat { \tau } _ { c } \left( \gamma ^ { B } \right)$ balances out bias and variance by construction, we should not expect our estimator to be variance dominated—and any inferential procedure should account for bias.

To this end, recall (8.13), whereby conditionally on $\left\{ Z _ { 1 } , . . . , Z _ { n } \right\}$ , the errors of our estimator, err $: = \hat { \tau } _ { c } - \tau _ { c }$ , are distributed as

$$
\operatorname{err} \left| \left\{Z _ {1}, \dots , Z _ {n} \right\} \sim \mathcal {N} \left(\text { bias }, \sigma^ {2} \| \gamma^ {B} \| _ {2} ^ {2}\right). \right. \tag {8.20}
$$

Furthermore, the optimization problem (8.19) yields as a by-product an upper bound for the bias in terms of the optimization variable t, namely $| \mathrm { b i a s } | \le B t$ .

We can then use these facts to build confidence intervals as follows. Because the Gaussian distribution is unimodal and symmetric,

$$
\mathbb {P} \left[ | \mathrm{err} | \geq \zeta \right] \leq \mathbb {P} \left[ \left| B t + \sigma \left\| \gamma^ {B} \right\| _ {2} S \right| \geq \zeta \right], \quad S \sim \mathcal {N} (0, 1). \tag {8.21}
$$

Thus, we obtain level-α confidence intervals as follows:

$$
\mathbb {P} \left[ \tau_ {c} \in \mathcal {I} _ {\alpha} \mid \{Z _ {1}, \dots , Z _ {n} \} \right] \geq 1 - \alpha ,
$$

$$
\mathcal {I} _ {\alpha} = \left(\hat {\tau} _ {c} (\gamma^ {B}) - \zeta_ {\alpha} ^ {B}, \hat {\tau} _ {c} (\gamma^ {B}) + \zeta_ {\alpha} ^ {B}\right), \tag {8.22}
$$

$$
\zeta_ {\alpha} ^ {B} = \inf \left\{\zeta : \mathbb {P} \left[ \left| B t + \sigma \left\| \gamma^ {B} \right\| _ {2} S \right| > \zeta \right] \leq \alpha , S \sim \mathcal {N} (0, 1) \right\}.
$$

In addition to formally accounting for bias, note that these intervals hold conditionally on $Z _ { i } .$ , and so hold without any distributional assumptions on the running variable. This is useful when considering regression discontinuities in non-standard settings.

Application: Discrete running variable A first example of the usefulness of having conditional-on-Zi guarantees is when the running variable $Z _ { i }$ has discrete support. In this case, the regression-discontinuity parameter $\tau _ { c }$ is in general not point-identified under only the assumption $| \mu _ { ( w ) } ^ { \prime \prime } ( z ) | \le B$ because there may not be any data arbitrarily close to the boundary.49 And, without point identification, any approach to inference that relies on asymptotics with specific rates of convergence for $\hat { \tau } _ { c }$ as discussed in the previous lecture clearly is not applicable.

In contrast, in our case, the fact that $Z _ { i }$ may have discrete support changes nothing. The confidence intervals (8.22) have coverage conditionally on $\left\{ Z _ { 1 } , . . . , Z _ { n } \right\}$ , and the empirical support $\left\{ Z _ { 1 } , . . . , Z _ { n } \right\}$ of the running variable is always discrete, so the question of whether the $Z _ { i }$ have a density in the population is irrelevant when working with (8.22). The relevance of a discrete $Z _ { i }$ only comes up asymptotically: If $Z _ { i }$ has a continuous density, then the confidence intervals (8.22) will shrink asymptotically at the optimal rate discussed in last lecture, namely $n ^ { - 2 / 5 }$ . Conversely, if the $Z _ { i }$ has discrete support, the length of the confidence intervals will not go to $0 ;$ rather, we end up in a partial identification problem. In this context, we also note that the bias-aware intervals (8.22) corresponds exactly to a type of confidence interval for partially identified parameters proposed in Imbens and Manski [2004].

Application: Multivariate running variable So far, we have focused on regression discontinuity designs where treatment is determined by a single threshold: $W _ { i } = 1 \left( \left\{ Z _ { i } \geq c \right\} \right)$ for some $Z _ { i } \in \mathbb { R }$ . However, the ideas discussed here apply in considerably more generality: One can let the running variable $Z _ { i } ~ \in ~ \mathbb { R } ^ { k }$ be multivariate, and the treatment region be generic, i.e., $W _ { i } \ =$ 1 $( \{ Z _ { i } \in \mathcal { A } \} )$ for some set $\mathcal { A } \subset \mathbb { R } ^ { k }$ . For example, in an educational setting, $Z _ { i } \in \mathbb { R } ^ { 3 }$ could measure test results in 3 separate subjects, and A could denote the set of overall “passing” results given by, e.g., 2 out of 3 tests clearing a pass/fail cutoff. Or in a geographic regression discontinuity design, $Z _ { i } \in \mathbb { R } ^ { 2 }$ could denote the location of one’s household and A the boundary of some administrative region that deployed a specific policy.

The crux of a regression discontinuity design is that we seek to identify causal effects via sharp changes to an existing treatment assignment policy; and we can then apply the same reasoning as before to identify treatment effects along the boundary of the treatment region A. That being said, while the extension of regression discontinuity designs to general multivariate settings is conceptually straight-forward, the methodological extensions require some more care. In particular, it is not always clear what the best way is to generalize local linear regression to a geographic regression discontinuity design.50

The minimax linear approach, however, extends direction to a multivariate setting. When working with a multivariate running variable, one can essentially write down (8.19) verbatim, and interpret the resulting weighted estimator similarly to before. The resulting optimization problem is harder (one needs to optimize over multivariate non-parametric functions with bounded curvature), but nothing changes conceptually.

Beyond homoskedaticity So far, we have focused on estimation and inference in the case where the noise $\varepsilon _ { i } = Y _ { i } - \mu _ { ( W _ { i } ) } ( Z _ { i } )$ was Gaussian with a known constant variance parameter $\sigma ^ { 2 }$ . In practice, of course, neither of these assumptions is likely to hold. The upshot is that the conditional Gaussianity result (8.20) no longer holds exactly; rather, we need to invoke a central limit theorem to argue that

$$
\hat {\tau} _ {c} (\gamma) \mid \{Z _ {1}, \dots , Z _ {n} \} \approx \mathcal {N} \left(\hat {\tau} _ {c} ^ {*} (\gamma), \sum_ {i = 1} ^ {n} \gamma_ {i} ^ {2} \operatorname{Var} \left[ Y _ {i} \mid Z _ {i}, W _ {i} \right]\right). \tag {8.23}
$$

However, provided we’re willing to make assumptions under which the Gaussian approximation above is valid, we can still proceed as above to get confidence intervals. Meanwhile, we can (conservatively) estimate the conditional variance in (8.23) via

$$
\widehat {V} _ {n} = \sum_ {i = 1} ^ {n} \gamma_ {i} ^ {2} \left(Y _ {i} - \hat {\mu} _ {(W _ {i})} (Z _ {i})\right) ^ {2}, \tag {8.24}
$$

where, $\mathrm { e . g . } , \hat { \mu } _ { ( W _ { i } ) } ( Z _ { i } )$ is derived via local linear regression; note that this bound is conservative if $\hat { \mu } _ { ( W _ { i } ) } ( Z _ { i } )$ is misspecified, since then the misspecifiaction error will inflate the residuals.

That being said, one should emphasize that the estimator (8.17) is only minimax under homoskedastic errors with variance $\sigma ^ { 2 } \mathopen { } \mathclose \bgroup \left. \begin{array} { r l r l } \end{array} \aftergroup \egroup \right.$ ; if we really wanted to be minimax under heteroskedasticity then we’d need to use per-parameter variances $\sigma _ { i } ^ { 2 }$ in (8.19). Thus, one could argue that an analyst who uses the estimator (8.17) but builds confidence intervals via (8.23) and (8.24) is using an oversimplified homoskedastic model to motivate a good estimator, but then out of caution and rigor uses confidence intervals that allow for heteroskedasticity when building confidence intervals. This is generally a good idea, and in fact something that’s quite common in practice (from a certain perspective, anyone who runs OLS for point estimation but then gets confidence intervals via the bootstrap is doing the same thing); however, it’s important to be aware that one is making this choice.

Remark 8.3. Throughout this section, we assumed that the researcher knows that (8.5) holds with some specific B, and proceeded accordingly. In practice, however, the researcher needs to choose B, and this is a delicate task. The data itself cannot be used to learn B unless one makes further smoothness assumptions [Armstrong and Koles´ar, 2018]. Armstrong and Koles´ar [2020] and Imbens and Wager [2019] propose some heuristics for conservative choices of B that rely on global estimation of higher-order polynomials. Eckles et al. [2020] consider a structural model for the running variable that, among other things, implies a theory-driven bound B that can be used in (8.5).

## 8.3 Bibliographic notes

The idea of using regression discontinuity designs for treatment effect estimation goes back to Thistlethwaite and Campbell [1960]; however, most formal work in this area is more recent. The framework of identification in regression discontinuity designs via continuity arguments and local linear regression is laid out by Hahn, Todd, and van der Klaauw [2001]. Other references on regressiondiscontinuity analysis via local linear regression include Cheng, Fan, and Marron [1997] who discuss optimal choices for the kernel weighting function, Imbens and Kalyanaraman [2012] who discuss bandwidth choice, and Calonico, Cattaneo, Farrell, and Titiunik [2019] who discuss the role of covariate adjustments. Imbens and Lemieux [2008] provide an overview of local linear regression methods in this setting, and discuss alternative specifications such as the “fuzzy” regression discontinuities where $W _ { i }$ is random but $\mathbb { P } \left[ W _ { i } = 1 \big | Z _ { i } = z \right]$ has a jump at the cutoff c.

As noted in Remark 8.2, the construction of confidence intervals via local linear regression is challenging because, when tuned for optimal mean-squared error, the bias and sampling error of the local linear regression estimator are of the same order—and so basic delta-method or bootstrap based inference fails (because it doesn’t capture bias). Several authors have considered solutions to the problem that rely on asymptotics. Calonico, Cattaneo, and Titiunik [2014] and Calonico, Cattaneo, and Farrell [2018] proposed bias-corrections to local linear regression to obtain valid confidence intervals. Meanwhile, Armstrong and Koles´ar [2020] showed that uncorrected local linear regression point estimates can also be used for valid inference provided we inflate the length of the confidence intervals by a pre-determined amount; for example, in the setting of Proposition 8.1 with an mean-square-optimal bandwidth, their proposal would involve building 95% confidence intervals for $\tau _ { c }$ as $\hat { \tau } _ { c } \pm 2 . 1 8$ standard errors (rather than the familiar ±1.96 standard errors).

The study of minimax linear estimators as considered in Chapter 8.2 goes back to Donoho [1994], who showed to following result. Suppose that we want to estimate θ using a Gaussian random vector Y ,

$$
Y = K v + \varepsilon , \quad \varepsilon \sim \mathcal {N} (0, \sigma I), \quad \theta = a \cdot v, \tag {8.25}
$$

where the matrix K and vector a are know, but v is unknown. Suppose moreover that v is known to belong to a convex set V. Then, there exists a linear $\begin{array} { r } { \hat { \theta } = \sum _ { i = 1 } ^ { n } \gamma _ { i } Y _ { i } } \end{array}$ a factor 1.25 of the minimax risk among all estimators (including non-linear ones), and the weights $\gamma _ { i }$ for the minimax linear estimator can be derived via convex optimization. From this perspective, the minimax RDD estimator (8.17) is a special case of the estimators studied by Donoho [1994], and in fact his results imply that this estimator is nearly minimax among all estimators (not just linear ones).

In a first application of this principle to regression discontinuity designs, Armstrong and Koles´ar [2018] study minimax linear estimation over a class of function proposed by Sacks and Ylvisaker [1978] for which Taylor approximations around the cutoff c are nearly sharp. Our presentation in Chapter 8.2 is adapted from Imbens and Wager [2019], who consider numerical convex optimization for flexible inference in generic regression discontinuity designs. Koles´ar and Rothe [2018] advocate worst-case bias measures of the form (8.15) as a way of avoiding asymptotics and providing credible confidence intervals in regression discontinuity designs with a discrete running variable. Noack and Rothe [2024] extend methods for bias-aware inference to fuzzy regression discontinuities.
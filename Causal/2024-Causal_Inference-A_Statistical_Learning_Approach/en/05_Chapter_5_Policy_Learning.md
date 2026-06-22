# Chapter 5 Policy Learning

So far, we’ve focused on methods for estimating treatments effects. In many application areas, however, the fundamental goal of performing a causal analysis isn’t to estimate treatment effects, but rather to guide decision making: We want to understand treatment effects so that we can effectively prescribe treatment and allocate limited resources.

The problem of learning optimal treatment assignment policies is closely related to—but subtly different from—the problem of estimating treatment heterogeneity. On one hand, policy learning appears easier: All we care about is assigning people to treatment or to control, and we don’t care about accurately estimating treatment effects beyond that. On the other hand, when learning policies, we need to account for considerations that were not present when simply estimating treatment effects: Any policy we actually want to use must be simple enough we can actually deploy it, cannot discriminate on protected characteristics, should not rely on gameable features, etc.

Policy value For our purposes, a treatment assignment policy $\pi ( x )$ is a mapping31

$$
\pi : \mathcal {X} \rightarrow \{0, 1 \}, \tag {5.1}
$$

such that individuals with features $X _ { i } = x$ get treated if and only if $\pi ( x ) = 1$ . Under the potential outcome specification, the expected realized outcome when treatment is chosen according to the policy $\pi$ is

$$
V (\pi) = \mathbb {E} \left[ Y _ {i} \left(\pi (X _ {i})\right) \right]. \tag {5.2}
$$

We refer to $V ( \pi )$ as the value of the policy π, and assume that the decision maker wants to use data to learn a policy $\hat { \pi }$ such that $V ( \hat { \pi } )$ large. This framework relies on an implicit assumption that the outcome $Y _ { i }$ captures the relevant benefit or reward the decision maker wants to optimize, and that the decision maker is utilitarian in the sense that their objective is to maximize the average reward across units.

Workflow Conceptually, there are three key phases in the policy learning workflow. First, we need to collect data with random or quasi-random treatment assignments $W _ { i }$ to learn a policy $\hat { \pi } ;$ throughout this chapter, we will assume that the treatment in this first stage is unconfounded and that data is drawn as in the basic setting from Chapter 3. In a second (optional) phase, we may want to evaluate the quality of the learned policy, i.e., estimate $V ( \hat { \pi } )$ . This requires a second dataset (often referred to as a test set) with random or quasi-random treatment assignment. Finally, once we’re done learning, we enter the last phase where we may choose to deploy the learned policy, i.e., we may choose to set $W _ { i } = \hat { \pi } ( X _ { i } )$ with the hope that the expected outcome $\mathbb { E } \left[ Y _ { i } \right]$ obtained via $Y _ { i } = Y _ { i } ( \hat { \pi } ( X _ { i } ) )$ will be large. In this third stage, there is no more randomness in treatment effects, so we cannot (non-parametrically) learn anything about causal effects anymore.

As noted earlier in Proposition 4.1, if we place no restrictions on $\pi ,$ then the maximizer of $V ( \pi )$ is the policy that thresholds the CATE:

$$
\pi^ {*} \in \operatorname{argmax} _ {\pi} \left\{V (\pi) \right\}, \quad \pi^ {*} (x) = 1 \left(\{\tau (x) > 0 \}\right). \tag {5.3}
$$

Thus, one possible approach to learning policies is to apply the plug-in principle to (5.3): One can first use methods discussed in the previous chapter to generate an estimate $\hat { \tau } ( \cdot )$ of the CATE, and then set $\hat { \pi } ( x ) ~ = ~ 1 ( \{ \hat { \tau } ( x ) > 0 \} )$ . This approach may be reasonable in some applications, but may result in policies that are hard to interpret or may not respect other practical constraints that are called for in the application. The focus of this chapter will be on developing methods for learning policies that do respect such constraints; we will present such methods in Section 5.2 after first discussing some preliminaries on policy evaluation below.

Example 5 (Continued). In the previous chapter, we introduced an example from Kitagawa and Tetenov [2018] where the authors seek to target JTPA eligibility based on education and income. The optimal, unrestricted targeting rule would just threshold the CATE. For feasibility reasons, however, they are most interested in linear treatment rules of the form32

$$
\tau (x) = 1 \left(\{\text {prior earnings} \cdot \alpha_ {1} + \text {education} \cdot \alpha_ {2} > c \}\right).
$$

Learning welfare-maximizing rules of this type requires new methods, introduced in this chapter.

## 5.1 Policy evaluation

The key focus of this chapter is on the first “learning” part of the policy learning workflow, $\mathrm { i . e . , }$ on how to use data to choose a good policy ˆπ. Methodologically, however, we first need to discuss the second “evaluation” part of the workflow: If someone gives us a policy $\hat { \pi }$ , how can we estimate $V ( \hat { \pi } ) \ ?$

For the purpose of this section, we will assume that we have access to test set of n samples with unconfounded treatment assignment as in the basic setting from Chapter 3, and that this test set is independent of the data used to learn the candidate policy ${ \hat { \pi } } _ { ; }$ , i.e., the training set. We will then discuss evaluation of $\hat { \pi }$ conditionally on the training set: Here, we are not trying to estimate E $\left[ V ( \hat { \pi } ) \right] \ ( \mathrm { i . e . } ,$ to integrate over randomness in $\hat { \pi } )$ , but simply to estimate $V ( \hat { \pi } )$ for the specific realization of $\hat { \pi }$ on hand. Because the test set and training sets are independent of each other, this task is equivalent to using the test set to estimate $V ( \pi )$ for an arbitrary fixed policy $\pi ;$ and for simplicity we will present the rest of this section in terms of this latter task.

Inverse-propensity weighting Consider evaluating a given deterministic policy π under unconfoundedness. If we further know the treatment propensities $e ( x )$ , then we can obtain a simple estimate of $V ( \pi )$ via inverse-propensity weighting (IPW):

$$
\widehat {V} _ {I P W} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {1 \left(\left\{W _ {i} = \pi (X _ {i}) \right\}\right) Y _ {i}}{\mathbb {P} \left[ W _ {i} = \pi (X _ {i})   |   X _ {i} \right]}, \tag {5.4}
$$

where P $\lceil W _ { i } = \pi ( X _ { i } ) \mid X _ { i } = x \rceil = e ( x )$ when $\pi ( x ) = 1$ and $1 - e ( x )$ else. Qualitatively, this approach averages outcomes across those observations for which the sampled treatment $W _ { i }$ matches the policy prescription $\pi ( X _ { i } )$ , and uses inverse-propensity weighting to account for the fact that some relevant potential outcomes remain unobserved.

When the treatment propensities are known, we can use the same argument as in Theorem 2.2 to check that, for any given policy $\pi .$ , the IPW estimateVIP W (π) is unbiased for V (π),

$$
\begin{array}{l} \mathbb {E} \left[ \widehat {V} (\pi) \right] = \mathbb {E} \left[ \frac {1 \left(\{W _ {i} = \pi (X _ {i}) \}\right) Y _ {i}}{\mathbb {P} \left[ W _ {i} = \pi (X _ {i}) \mid X _ {i} \right]} \right] \\ = \mathbb {E} \left[ \frac {1 \left(\left\{W _ {i} = \pi (X _ {i}) \right\}\right) Y _ {i} (\pi (X _ {i}))}{\mathbb {P} \left[ W _ {i} = \pi (X _ {i}) \mid X _ {i} \right]} \right] \tag {5.5} \\ = \mathbb {E} \left[ \mathbb {E} \left[ \frac {1 \left(\left\{W _ {i} = \pi (X _ {i}) \right\}\right)}{\mathbb {P} \left[ W _ {i} = \pi (X _ {i}) \mid X _ {i} \right]} \mid X _ {i} \right] \mathbb {E} \left[ Y _ {i} (\pi (X _ {i})) \mid X _ {i} \right] \right] \\ = \mathbb {E} \left[ Y _ {i} (\pi (X _ {i})) \right] = V (\pi), \\ \end{array}
$$

where the second equality follows by consistency of potential outcomes and the third by unconfoundedness.

Augmented IPW In Chapter 3, we discussed how IPW-based estimators for the average treatment effect introduced in Chapter 2 are generally inefficient (at least when run with the true propensity scores) and are not robust to estimation error in $e ( x )$ ; and how the augmented IPW (AIPW) construction can be used to address both of these shortcomings. Similar considerations apply with policy evaluation. For conciseness, we do not repeat the development from Chapter 3 here, and instead simply state the AIPW estimator and its key properties.

As usual, forming the AIPW requires estimates $\hat { \mu } _ { w } ( x )$ for the conditional response functions and $\hat { e } ( x )$ for the propensity score. Given such estimates, the plug-in non-parametric regression estimator for $V ( \pi )$ is obtained by averaging predictions we would get by following the policy π, i.e.,

$$
\widehat {V} _ {R E G} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \hat {\mu} _ {\pi (X _ {i})} (X _ {i}). \tag {5.6}
$$

AIPW is obtained by using IPW to debias this estimator by extracting any remaining signal from the regression residuals,

$$
\widehat {V} _ {A I P W} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \hat {\mu} _ {\pi (X _ {i})} (X _ {i}) + \frac {1 \left(\{W _ {i} = \pi (X _ {i}) \}\right)}{\mathbb {P} \left[ W _ {i} = \pi (X _ {i}) \mid X _ {i} \right]} \left(Y _ {i} - \hat {\mu} _ {\pi (X _ {i})} (X _ {i})\right). \tag {5.7}
$$

As always with AIPW-type estimators, cross-fitting is recommended when forming the AIPW estimator. If we use cross-fitting and use estimates for $\hat { \mu } _ { w } ( x )$ and ${ \hat { e } } ( x )$ that converge at the rates assumed in Theorem 3.2, then

$$
\begin{array}{l} \sqrt {n} \left(\widehat {V} _ {A I P W} (\pi) - V (\pi)\right) \\ \Rightarrow \mathcal {N} \left(0, \operatorname{Var} \left[ \mu_ {\pi (X _ {i})} (X _ {i}) \right] + \mathbb {E} \left[ \frac {\sigma_ {\pi (X _ {i})} ^ {2} (X _ {i})}{\mathbb {P} \left[ W _ {i} = \pi (X _ {i}) \mid X _ {i} \right]} \right]\right), \tag {5.8} \\ \end{array}
$$

and the AIPW estimator is efficient. The proof of these results exactly mirrors the arguments used in Chapter 3.

Policy comparison It is often of interest to compare two policies $\pi _ { 1 }$ and $\pi _ { 2 }$ by estimating the difference in their values

$$
\Delta (\pi_ {1}, \pi_ {2}) = V (\pi_ {1}) - V (\pi_ {2}). \tag {5.9}
$$

For example, if $\pi _ { 0 }$ is a status-quo treatment-assignment rules, and $\hat { \pi }$ is a new proposed data-driven rule, then the difference $\Delta ( \hat { \pi } , \pi _ { 0 } )$ directly quantifies the benefit of adopting the data-driven rule relative to the status quo.

Given the above discussion, a natural way to estimate the value difference between to policies is to take the difference between their AIPW value estimates. A direct algebraic manipulation can be used to re-express the resulting estimator in condensed form as,

$$
\widehat {\Delta} _ {A I P W} (\pi_ {1}, \pi_ {2}) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\pi_ {1} (X _ {i}) - \pi_ {2} (X _ {i})\right) \widehat {\Gamma} _ {i},
$$

$$
\widehat {\Gamma} _ {i} = \hat {\mu} _ {(1)} (X _ {i}) - \hat {\mu} _ {(0)} (X _ {i}) + \frac {W _ {i}}{\hat {e} (X _ {i})} \left(Y _ {i} - \hat {\mu} _ {(1)} (X _ {i})\right) \tag {5.10}
$$

$$
- \frac {1 - W _ {i}}{1 - \hat {e} (X _ {i})} \left(Y _ {i} - \hat {\mu} _ {(0)} (X _ {i})\right),
$$

and under the conditions of Theorem 3.2

$$
\begin{array}{l} \sqrt {n} \left(\widehat {\Delta} _ {A I P W} (\pi_ {1}, \pi_ {2}) - \Delta (\pi_ {1}, \pi_ {2})\right) \\ \Rightarrow \mathcal {N} \left(0, \operatorname{Var} \left[ \left(\pi_ {1} \left(X _ {i}\right) - \pi_ {2} \left(X _ {i}\right)\right) \tau \left(X _ {i}\right) \right] \right. \tag {5.11} \\ + \mathbb {E} \left[ 1 \left(\{\pi_ {1} (X _ {i}) \neq \pi_ {2} (X _ {i}) \}\right) \left(\frac {\sigma_ {0} ^ {2} (X _ {i})}{1 - e (X _ {i})} + \frac {\sigma_ {1} ^ {2} (X _ {i})}{e (X _ {i})}\right) \right]. \\ \end{array}
$$

When $\pi _ { 1 }$ and $\pi _ { 2 }$ often agree on the action to take, then $\widehat { \Delta } _ { A I P W } ( \pi _ { 1 } , \pi _ { 2 } )$ only needs to consider outcomes in the smaller region where their recommendations differ—thus enabling a considerable improvement in precision.

One specific policy contrast that is often of interest is the comparison of a given policy $\pi$ to the never-treat policy. We use short-hand $\Delta ( \pi ) = \Delta ( \pi , 0 )$ for this quantity, and refer to it as the benefit of the policy $\pi .$ . We also note that the benefit of the always-treat policy, $\Delta ( 1 )$ , corresponds exactly to the average treatment effect, and as a sanity check we can verify that in this case (5.11) is just a re-statement of the result in Theorem 3.2.

Aside: Treatment prioritization rules One type of policy that often arises in practice is treatment prioritization rules. Such policies start with a priority function $S : \mathcal { X }  \mathbb { R }$ , and then assign treatment to the top q-th fraction of units as ranked by the priority $S ( X _ { i } )$ :

$$
\pi_ {S} ^ {q} = 1 \left(\left\{S (X _ {i}) \geq F _ {S} ^ {- 1} (1 - q) \right\}\right), \tag {5.12}
$$

where $F _ { S }$ is the the cumulative distribution function of the priorities $S ( X _ { i } )$ . Here, the priority function could be a CATE estimate obtained using a separate training set, a risk measure quantifying who’s most at risk of a bad outcome without treatment, or some other application-relevant notion of priority.

We can use policy evaluation to quantify the extent to which the priority function succeeds in allocating treatment to those who benefit most from it. The QINI curve estimates the benefit $\Delta ( \pi _ { S } ^ { q } )$ of treating the top q-th fraction of units for different values of $q ,$ and then plots $\Delta ( \pi _ { S } ^ { q } )$ on the Y -axis against $q$ on the X-axis. In settings where each unit has a constant cost of treatment, the QINI curve quantifies a cost-benefit exercise where we measure how the obtained benefit changes as we spend more.

Meanwhile, the TOC curve considers $q ^ { - 1 } \Delta ( \pi _ { S } ^ { q } ) - \Delta ( 1 )$ , and plots this quantity against $q .$ This curve quantifies the extent to which the top q-th fraction of units as prioritized by $S ( \cdot )$ benefit more from the treatment than randomly selected units. These quantities are discussed in Yadlowsky et al. [2021]; the paper also advocates considering the area under the TOC curve with units prioritized by estimated CATE as a useful measure of overall detected treatment heterogeneity.

The value of treatment prioritization rules can again be estimated using the doubly robust approach:

$$
\widehat {\Delta} _ {A I P W} \left(\pi_ {S} ^ {q}\right) = \frac {1}{n} \sum_ {k = 1} ^ {\lfloor q n \rfloor} \widehat {\Gamma} _ {i (k)}, S \left(X _ {i (1)}\right) \geq S \left(X _ {i (2)}\right) \geq \ldots \geq S \left(X _ {i (n)}\right). (5. 1 3)
$$

One statistical challenge in studying the large-sample properties of this estimator is that it depends on the empirical q-th quantile of $S ( X _ { i } )$ , which results in an inflated asymptotic variance relative to (5.8). Yadlowsky et al. [2021] provide a central limit theorem for the value estimate in (5.13) as well as for induced area-under-the-curve metrics for QINI and TOC curve estimates; they also discuss resampling-based methods for these quantities.

## 5.2 Empirical-welfare maximization

We now return to the task of learning a policy, i.e., using experimental or quasiexperimental data to choose a good treatment assignment rule $\hat { \pi } ( \cdot )$ . Throughout, we assume that the policymaker is constrained to choose a policy π belonging to some class Π of acceptable policies; for example, Π may encode restrictions on the functional form the policy is allowed to take or on which variables it is allowed to use. Simple examples of policy classes one might consider include the class of linear thresholding rules $\tau ( x ) = 1 ( \{ a \cdot x \geq c \} )$ for some vector a and threshold $c ,$ or the class of fixed-depth decision trees.

Given this setting, the optimal policy—or policies—are those that maximize policy value among all acceptable policies:

$$
\pi^ {*} \in \operatorname{argmax} \left\{V (\pi^ {\prime}): \pi^ {\prime} \in \Pi \right\}. \tag {5.14}
$$

Any non-optimal (but acceptable) policy $\pi$ falls short of this best possible policy value, and suffers regret

$$
R (\pi) = \sup _ {\pi} \left\{V (\pi^ {\prime}): \pi^ {\prime} \in \Pi \right\} - V (\pi). \tag {5.15}
$$

Our goal is to learn a policy with guaranteed worst-case bounds on the regret $R ( { \hat { \pi } } )$ . We refer this task as a learning (rather than estimation) task because the performance of $\hat { \pi }$ is only assessed in terms of its regret. No requirements will be made on $\hat { \pi }$ converging to $\pi ^ { * }$ in terms of its functional form (and in fact no assumption is made that there is a unique optimal policy $\pi ^ { * } )$ .

If the optimal policy $\pi ^ { * }$ is a maximizer of the true value function $V ( \pi )$ over $\pi \in \Pi$ , then it is natural to attempt learn ˆπ by maximizing an estimated value function:

$$
\hat {\pi} = \operatorname{argmax} \left\{\widehat {V} (\pi): \pi \in \Pi \right\}. \tag {5.16}
$$

This approach was coined as empirical-welfare maximization by Kitagawa and Tetenov [2018]. In the previous section we already discussed two estimators of $V ( \pi )$ using data with randomized or unconfounded treatment assignment, namely the IPW and AIPW estimators, and both can be used to learn following (5.16). We refer to the maximizer of $\widehat { V } _ { I P W } ( \pi )$ over $\pi \in \Pi$ as ${ \hat { \pi } } _ { I P W }$ , and to the maximizer of $\widehat { V } _ { A I P W } ( \pi )$ as ${ \hat { \pi } } _ { A I P W }$ .

Regret bounds Proving that the empirical-welfare maximization approach achieves low regret is beyond the scope of this book; however, we here sketch the starting point of an argument for doing so. Let $\pi ^ { * }$ be any policy achieving the maximal policy value, and let $\hat { \pi }$ be a maximizer of the estimated value as in (5.16). Then,

$$
\begin{array}{l} R (\hat {\pi}) = V \left(\pi^ {*}\right) - V (\hat {\pi}) \tag {5.17} \\ = V \left(\pi^ {*}\right) - \widehat {V} \left(\pi^ {*}\right) + \widehat {V} \left(\pi^ {*}\right) - \widehat {V} (\hat {\pi}) + \widehat {V} (\hat {\pi}) - V (\hat {\pi}). \\ \end{array}
$$

Because $\hat { \pi }$ is a maximizer of the estimated value we have $\widehat { V } \left( \pi ^ { * } \right) - \widehat { V } \left( \widehat { \pi } \right) \leq 0$ , so we can further get

$$
\begin{array}{l} \begin{array}{l} R (\hat {\pi}) \leq V \left(\pi^ {*}\right) - \widehat {V} \left(\pi^ {*}\right) + \widehat {V} (\hat {\pi}) - V (\hat {\pi}) \\ 1. 2 \quad \left\{\left| \widehat {V} (x) - V (x) \right|, \dots , \Pi \right\} \end{array} (5.18) \\ \leq 2 \sup \left\{\left| \widehat {V} (\pi) - V (\pi) \right|: \pi \in \Pi \right\}, (5.18) \\ \end{array}
$$

and in particular

$$
\mathbb {E} \left[ R (\hat {\pi}) \right] \leq 2 \mathbb {E} \left[ \sup \left\{\left| \widehat {V} (\pi) - V (\pi) \right|: \pi \in \Pi \right\} \right]. \tag {5.19}
$$

Thus, proving regret bounds for any empirical-welfare maximization approach reduces to proving uniform bounds on the error of $\widehat V ( \pi )$ that hold simultaneously for all acceptable policies $\pi \in \Pi$ .

One can use tools from empirical process theory to bound the term on the right-hand-side of (5.19); however, doing so relies on technical results beyond the scope of this presentation. To state one concrete version of a result obtained by following this path, let VC(Π) denote the Vapnik-Chervonenkis dimension of Π (in many practical cases, one can essentially think of VC(Π) as capturing the number of parameters needed to specify an element of Π), and assume that VC(Π) is finite. Then, Athey and Wager [2021] show that—under the conditions of Theorem 3.2 along with further regularity conditions—the policy learned by maximizing the AIPW value estimate (5.7) satisfies

$$
\begin{array}{l} \limsup _ {n} \sqrt {n} \mathbb {E} \left[ R (\hat {\pi} _ {A I P W}) \right] \\ \leq 6 0 \sqrt {\operatorname{VC} (\Pi) \left(\operatorname{Var} \left[ \tau (X _ {i}) \right] + \mathbb {E} \left[ \frac {\sigma_ {0} ^ {2} (X _ {i})}{1 - e (X _ {i})} + \frac {\sigma_ {1} ^ {2} (X _ {i})}{e (X _ {i})} \right]\right)}. \tag {5.20} \\ \end{array}
$$

What’s meaningful about this bound is that it connects how the worst-case regret of empirical-welfare maximization scales with various problem primitives. Specifically, we see that the bound increases with the square root of the dimension of the Π (larger policy spaces are harder to learn over) and the variance of the AIPW scores (learning is harder when ATE estimation is harder), and decreases with the square root of the sample size (more data helps). The constant 60 is likely loose here, though.33Policy learning as weighted classification The above discussion on regret shows that empirical-welfare maximization is in principle a promising approach to policy learning. However, in order to use this approach in practice, one needs to be able to carry out the optimization problem (5.16) in a computationally tractable manner. This is in general a challenging (non-convex) optimization problem; thankfully, however, it turns out that the empirical-welfare maximization problem is in many cases equivalent to a weighted classification problem, thus allowing us to leverage computational insights from that literature.

Here, we focus on maximizing the AIPW value estimate (5.7). As a first helpful step, we symmetrize the objective by defining

$$
\widehat {A} _ {A I P W} (\pi) = \widehat {V} _ {A I P W} (\pi) - \widehat {V} _ {A I P W} (1 - \pi), \tag {5.21}
$$

i.e., the estimated improvement from following π relative to always doing the opposite of $\pi .$ Clearly, $\pi$ is a maximizer of $\widehat { V } _ { A I P W } ( \pi )$ if and only if it is a maximizer of $\hat { A } _ { A I P W } ( \pi )$ ; thus, we can equivalently write

$$
\hat {\pi} _ {A I P W} = \operatorname{argmax} \left\{\widehat {A} _ {A I P W} (\pi): \pi \in \Pi \right\}. \tag {5.22}
$$

Furthermore, following our discussion on policy comparisons, we can check that check that

$$
\widehat {A} _ {A I P W} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} (2 \pi (X _ {i}) - 1) \widehat {\Gamma} _ {i}, \tag {5.23}
$$

where $\widehat { \Gamma } _ { i }$ is as defined in (5.10).

For the purpose of optimization, the upshot is that we can now re-write our empirical-welfare maximization problem as a weighted classification problem:

$$
\hat {\pi} _ {A I P W} = \operatorname{argmax} \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \underbrace {(2 \pi (X _ {i}) - 1) \operatorname{sign} (\widehat {\Gamma} _ {i})} _ {\text {classification objective}} \underbrace {| \widehat {\Gamma} _ {i} |} _ {\text {sample weight}}: \pi \in \Pi \right\}. \tag {5.24}
$$

Qualitatively, the intuition here, policy learning is equivalent to trying to choose a policy that matches the sign of the AIPW scores as well as possible, with weight corresponding to the magnitude of the AIPW scores. Practically, this result means that we can use any software package for weighted classification to optimize our target objective and learn ${ \hat { \pi } } _ { A I P W }$ .

The weighted classification formulation (5.24) is valuable from a computational perspective; however, one should be careful not to read into it too much. In typical signal-to-noise regimes, the signs of the AIPW scores $\widehat { \Gamma } _ { i }$ will be fairly random, and actually predicting these signs with any reliability is impossible. Even an optimal policy $\pi ^ { * }$ will make many “errors” according to the classification formulation; and trying to get high accuracy according to the classification metric will only result in overfitting. It is possible to have problems where empirical-welfare maximization works very well (in terms of improving value relative to a status quo), but where standard classification diagnostics applied to the formulation (5.24) would suggest poor performance.34

The role of the policy class Π We started with a non-parametric model $( \mathrm { i . e . , ~ } \mu _ { ( w ) } ( x )$ and $e ( x )$ can be generic), where the welfare-maximizing unrestricted treatment assignment rule is simply $\pi _ { u n r e s t r } ^ { * } ( x ) = 1 \left( \{ \tau ( x ) > 0 \} \right)$ . However, our goal in this chapter was not to find a way to approximate $\pi _ { u n r e s t r } ^ { * } ( \cdot )$ ; rather, given a pre-specified class of policies Π, we sought to learn a nearly regret-optimal policy from Π. For example, Π could consist of linear decision rules, k-sparse decision rules, depth-\` decision trees, etc. Note, in particular, that we never assumed that $\pi _ { u n r e s t r } ^ { * } ( \cdot ) \in \Pi$ .

This problem setting may appear surprising at first glance. However, in many applications, it’s important to consider learning over restricted policy classes. A key reason for this is that, in policy learning problems, the features $X _ { i }$ can play multiple distinct roles. First, the $X _ { i }$ may be needed to achieve unconfoundedness

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp W _ {i} \mid X _ {i}.
$$

In general, the more pre-treatment variables we have access to, the more plausible unconfoundedness becomes. In order to have a credible model of nature, it’s good to have flexible, non-parametric models for $e ( x )$ and $\mu _ { ( w ) } ( x )$ using a wide variety of features.

On the other hand, when we want to deploy a policy $\pi ( \cdot )$ , we should be much more careful about what features we use to make decisions and the form of the policy $\pi ( \cdot )$ . Depending on the application, there may be some features that are required to achieve unconfoundedness, but are problematic when used for treatment choice. This includes features that are difficult to measure in a deployed system, features that are gameable by participants in the system, or features that correspond to legally protected classes. In cases like this, these features need to be kept in the dataset to identify causal effects, but the set Π should only contain policies $\pi$ that do not depend on them. Furthermore, many applications involve functional form constraints on $\pi ( \cdot )$ that could reasonably be deployed $( \mathrm { e . g . }$ , if the policy needs to be communicated to employees in a non-electronic format, or audited using non-quantitative methods). Thus, when learning policies, it’s important to be able to respond to applicationdriven constraints as codified by the use of a restricted class Π of allowable policies.

## 5.3 Bibliographic notes

The idea behind our discussion today was that, when learning policies, the natural quantity to focus on is regret as opposed to, e.g., squared-error loss on the conditional average treatment effect function. This point is argued for in Manski [2004]. Stoye [2009] provides a discussion of exact minimax regret policy learning with discrete covariates, while Hirano and Porter [2009] consider asymptotic analysis in the limits-of-experiments framework.

The insight that policy learning under unconfoundedness can be framed as a weighted classification problem—and that we can adapt well known result results from empirical risk minimization to to derive useful regret bounds— appears to have been independently discovered in statistics [Zhao et al., 2012], computer science [Swaminathan and Joachims, 2015], and economics [Kitagawa and Tetenov, 2018]. Properties of policy learning with doubly robust scoring rules are derived in Athey and Wager [2021]. The latter paper also considers policy learning in more general settings, such as with “nudge” interventions to continuous treatments or with instruments used to identify the effects of endogenous treatments. Mbakop and Tabord-Meehan [2021] consider model selection for empirical-welfare maximization to handle policy classes with infinite VC dimension, while Zhou, Athey, and Wager [2023] consider structured treatment choice with multiple possible actions.

In this chapter, we’ve discussed rates of convergence that scale as $\sqrt { \mathrm { V C } ( \Pi ) / n }$ This is the optimal rate of convergence we can get if seek guarantees that are uniform over $\tau ( x )$ ; and the rates are sharp when the strength of the treatment effects decays with sample size at rate $1 / \sqrt { n }$ . However, if we consider asymptotics for fixed choices of $\tau ( x )$ , then super-efficiency phenomena appear and we can obtain faster than $1 / \sqrt { n }$ rates [Luedtke and Chambaz, 2020]; this phenomenon is closely related to “large margin” improvements to regret bounds for classification via empirical risk minimization.

QINI curves for evaluating treatment prioritization rules were first introduced in the marketing literature to quantify the value of targeted marketing campaigns. Imai and Li [2023] provide a modern statistical treatment of QINI curves in randomized controlled trial under the Neyman model. Yadlowsky et al. [2021] provide a unified analysis of different methods for evaluating treatment prioritization rules—including both the QINI and TOC curves—in a general observational study setting that accommodates double machine learning. Sun et al. [2021] use QINI curves to quantify cost-benefit exercises in settings where treatment cost is also unknown and needs to be estimated, while Sverdrup et al. [2023] do so in the case of treatment prioritization rules that allow for multiple actions.

The topic of policy learning is an active area with many recent advances. For example, Bertsimas and Kallus [2020] extend the principle of learning policies by optimizing a problem-specific empirical value function to a wide variety of settings, e.g., inventory management; Luedtke and van der Laan [2016] discuss inference for the value of the optimal policy; while Kallus and Zhou [2021] consider the problem of learning policies in a way that is robust to potential failures of unconfoundedness.
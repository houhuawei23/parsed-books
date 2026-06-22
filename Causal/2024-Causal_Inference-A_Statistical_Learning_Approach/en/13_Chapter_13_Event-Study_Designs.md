# Chapter 13 Event-Study Designs

All examples considered in this book so far involve settings where we observe a unit, they receive some treatment exposure (or not), and then reveal an outcome. In applications, however, it is common to follow units over time and to obtain multiple measurements from each unit. For example, when studying the effect of a tax policy, we will often be able to follow a country over time— and under different tax policies. Or, in medicine, we often follow a patient over time as they go through a potentially complex treatment regimen.

This chapter—as well as the following two—will introduce methods for causal inference in settings where units are followed over time. Data collected in such settings is often referred to as panel data or longitudinal data. Incorporating full treatment dynamics—where treatment can toggle on and off, and we need to reason about both long- and short-term effects of actions—will be deferred to subsequent chapters. Here, instead, we will focus on the simpler case of event studies where all units start in the control condition and then, if they ever start treatment, they never stop. Our focus on event studies will enable a gradual ramp-up in the technical tools required to work with panel data, and allow us to introduce some widely used econometric methods.

Example 17. In 1990, all but one of 477 municipalities in Argentina had water services that were either public or owned by non-profit cooperatives. By the end of the decade, 137 of these municipalities privatized their water systems, and transferred ownership to private for-profit entities. Galiani, Gertler, and Schargrodsky [2005] use this panel dataset—and exploit the fact that some municipalities are observed in the transition from public to private ownership— to study potential community health effects from privatizing water resources.

Suppose we observe a panel of $i = 1 , \ldots , n$ units across $t = 1 , \dots , T$ time periods. In each $( i , t )$ pair the is in treatment condition $W _ { i t } \in \{ 0 , 1 \}$ and we observe an outcome $Y _ { i t } \in \mathbb { R }$ . Our event study assumption requires that treatment can only ever switch off-to-on, i.e., that $W _ { i t } \leq W _ { i t ^ { \prime } }$ for all $t \leq t ^ { \prime }$ .

There are two treatment patterns that fall under the event study umbrella.

Definition 13.1. In the block-adoption design, there is a shared event time $1 \leq H < T$ such that each unit either starts treatment right after H or never does. Each unit as an adoption indicator $D _ { i } \in \{ 0 , 1 \}$ such that $W _ { i t } = D _ { i } 1 \left( \{ t > H \} \right)$ .

Definition 13.2. In the staggered-adoption design, each unit either has its own event time $1 \leq H _ { i } \leq T - 1$ after which it starts treatment, or it never starts treatment in which case we write $H _ { i } = \infty$ . We then have $W _ { i t } = 1 ( \{ t > H _ { i } \} )$ .

As usual, we will define our causal estimands in terms of potential outcomes. As discussed in Chapter 11, defining potential outcomes for general causal inferefence problems requires considering the different possible treatment exposures a unit may face. Without any restrictions, a unit who receives a binary intervention in each of $T$ time periods could experience $2 ^ { T }$ different treatment trajectories, and one would then need to either define $2 ^ { T }$ potential outcomes for each unit or define an exposure mapping for dimensionality restriction. In event study designs, however, the off-to-on restriction on treatment assignment restricts the number of possible treatment trajectories and simplifies the definition of potential outcomes.

In the block-adoption design, a unit’s treatment trajectory is fully defined by its adoption indicator, and so we can write potential outcomes

$$
Y _ {i t} (d) \text {   for   } d = 0, 1, \tag {13.1}
$$

with a SUTVA assumption that $Y _ { i t } = Y _ { i t } ( D _ { i } )$ . In the staggered-adoption design there’s a little more flexibility as there are now $T$ possible treatment-start times; natural potential outcomes are then

$$
Y _ {i t} (h) \text {   for   } h = 1, 2, \dots , T - 1, \infty , \tag {13.2}
$$

with a SUTVA assumption $Y _ { i t } = Y _ { i t } ( H _ { i } )$ . Throughout, we will assume temporal consistency of actions, i.e., that future actions cannot affect past outcomes.

Assumption 13.1. We assume that potential outcomes do not anticipate treatment. Specifically, in the block-adoption design case, we assume that

$$
Y _ {i t} (0) = Y _ {i t} (1) \text {   for   } t = 1, \dots , H, \tag {13.3}
$$

while in the staggered-adoption design case, we assume that

$$
Y _ {i t} (h) = Y _ {i t} (h ^ {\prime}) \text {   for   } t = 1, \ldots , \min \{h, h ^ {\prime} \}. \tag {13.4}
$$

Assuming temporal consistency may seem innocuous when presented abstractly, but this is in fact an assumption that may easily fail to hold in some applications. For example, if we want to study the effect of a policy started by a country i at time $H _ { i } .$ but some people were able to anticipate this policy change and adapt their behavior in advance of it, then this non-anticipation assumption would not hold. The non-anticipation assumption should thus be carefully assessed before using any of the methods presented in this chapter.

## 13.1 Difference in differences

Under the block-adoption design, one natural estimand to target is the average treatment effect on the treated (ATT). Assuming that units i are independently drawn from a population of units, the average per-time-period effect of receiving treatment among treated units is

$$
\tau_ {A T T} = \mathbb {E} \left[ \frac {1}{T - H} \sum_ {t = H + 1} ^ {T} Y _ {i t} (1) - Y _ {i t} (0) \right]. \tag {13.5}
$$

How should we go about estimating this quantity?

A first natural estimator to try is the simple difference-in-means comparison in the post-event periods,

$$
\hat {\tau} _ {D M} = \frac {\sum_ {\{i : D _ {i} = 1 \}} \sum_ {t = H + 1} ^ {T} Y _ {i t}}{| \{i : D _ {i} = 1 \} | (T - H)} - \frac {\sum_ {\{i : D _ {i} = 0 \}} \sum_ {t = H + 1} ^ {T} Y _ {i t}}{| \{i : D _ {i} = 0 \} | (T - H)}. \tag {13.6}
$$

This estimator, however, may seem wasteful in that it completely ignores available data from the pre-event periods. One popular way to leverage pre-event data available in a panel is using the difference-in-differences (DID) estimator:

$$
\begin{array}{l} \hat {\tau} _ {D I D} = \frac {1}{| \{i : D _ {i} = 1 \} |} \sum_ {\{i: D _ {i} = 1 \}} \left(\frac {1}{T - H} \sum_ {t = H + 1} ^ {T} Y _ {i t} - \frac {1}{H} \sum_ {t = 1} ^ {H} Y _ {i t}\right) \\ - \frac {1}{| \{i : D _ {i} = 0 \} |} \sum_ {\{i: D _ {i} = 0 \}} \left(\frac {1}{T - H} \sum_ {t = H + 1} ^ {T} Y _ {i t} - \frac {1}{H} \sum_ {t = 1} ^ {H} Y _ {i t}\right). \tag {13.7} \\ \end{array}
$$

In words, the DID estimator first uses pre-event data to construct a baseline outcome that is subtracted from post-event outcomes, and then compares these post-minus-pre differences across adopters and non-adopters.

As a first sanity check, both the simple difference and difference-indifferences estimators can immediately be verified to be unbiased when adoption is randomized.

Proposition 13.1. If adoption is randomized, then $\mathbb { E } \left[ \hat { \tau } _ { D M } \right] = \tau$ . Furthermore, if Assumption 13.1 holds, then $\mathbb { E } \left[ \hat { \tau } _ { D I D } \right] = \tau$ .

Proof. The first statement follows immediately from Theorem 1.1. The second statement follows by noting that, under Assumption 13.1, incorporating the pre-event data into the estimator has a mean-zero effect under randomized adoption. □

In many practical event study applications, however, treatment cannot credibly be taken to be randomized. Consider, for example, a setting where our units correspond to the $n = 5 0$ states in the United States. Some states choose to adopt a policy (e.g., to accept Federal subsidies to expand Medicaid coverage) while others don’t. We would like to use difference in differences, but treatment here is clearly not randomized, and in fact the sampling assumptions used to define the ATT in (13.5) don’t really make sense either—and so Proposition 13.1 does not apply.

Thankfully, it turns out that the difference-in-differences estimator has a double-robustness-type property whereby it can also be justified via a functional form assumption, namely parallel trends. The parallel trends assumption, made formal in Assumption 13.2, states that all non-adopter potential outcomes must evolve in parallel (but may start at different levels). When parallel trends holds, DID can be verified to be on average unbiased for the following sample average treatment effect on the treated (SATT),

$$
\tau_ {S A T T} = \frac {\sum_ {\{i : D _ {i} = 1 \}} \sum_ {t = H + 1} ^ {T} (Y _ {i t} (1) - Y _ {i t} (0))}{| \{i : D _ {i} = 1 \} | (T - H)}, \tag {13.8}
$$

without requiring any reference to population sampling assumptions.

Assumption 13.2. There exist $\beta _ { 2 } , . . . , \beta _ { T } \in \mathbb { R }$ such that, for all units $i =$ $1 , \ldots , n ,$ , never-treated potential outcomes satisfy

$$
\mathbb {E} \left[ Y _ {i t} (0 / \infty) - Y _ {i 1} (0 / \infty) \right] = \beta_ {t}, \quad t = 2, \dots , T. \tag {13.9}
$$

Recall that we write never-treated potential outcomes as $Y _ { i t } ( 0 )$ under block adoption and $Y _ { i t } ( \infty )$ under staggered adoption.

Theorem 13.2. In the block-adoption design suppose that some—but not $a l l -$ units are exposed to treatment $( i . e .$ , have $D _ { i } = 1 )$ . Then, under Assumptions 13.1 and 13.2, E $\left[ \hat { \tau } _ { D I D } - \tau _ { S A T T } \right] = 0$ .

Proof. A comparison of (13.7) and (13.8) reveals that, under Assumption 13.1,

$$
\begin{array}{l} \hat {\tau} _ {D I D} - \tau_ {S A T T} = \frac {1}{| \{i : D _ {i} = 1 \} |} \sum_ {\{i: D _ {i} = 1 \}} \left(\frac {1}{T - H} \sum_ {t = H + 1} ^ {T} Y _ {i t} (0) - \frac {1}{H} \sum_ {t = 1} ^ {H} Y _ {i t} (0)\right) \\ - \frac {1}{| \{i : D _ {i} = 0 \} |} \sum_ {\{i: D _ {i} = 0 \}} \left(\frac {1}{T - H} \sum_ {t = H + 1} ^ {T} Y _ {i t} (0) - \frac {1}{H} \sum_ {t = 1} ^ {H} Y _ {i t} (0)\right). \\ \end{array}
$$

Furthermore, under Assumption 13.2,

$$
\mathbb {E} \left[ \frac {1}{T - H} \sum_ {t = H + 1} ^ {T} Y _ {i t} (0) - \frac {1}{H} \sum_ {t = 1} ^ {H} Y _ {i t} (0) \right] = \frac {1}{T - H} \sum_ {t = H + 1} ^ {T} \beta_ {t} - \frac {1}{H} \sum_ {t = 2} ^ {H} \beta_ {t}
$$

is the same for each i = 1, . . . , n. The contributions of the $\beta _ { t }$ then cancel out perfectly. □

The parallel trend assumption is a fairly strong functional form assumption, and so guarantees obtained under this assumption are not generally comparable to guarantees for causal inference available in randomized controlled trials. They are, however, still valuable in practice, and DID type analyses have been hugely influential in applied work. For example, in one early and influential study of the empirical effects of raising the minimum wage on employment, Card and Krueger [1994] conducted a DID study comparing employment outcomes across time in New Jersey, which raised its minimum wage during the study period, to those in Pennsylvania, where the minimum wage remained fixed. This study identified treatment effects by assuming parallel trends— and still to date much of the empirical literature on minimum wage effects is justified by various parallel-trends-type assumptions.

Staggered adoption Under the block-adoption design, all units who ever get treatment start treatment at the same time. In practice, however, it is often of interest to also consider the staggered-adoption design where units may begin treatment at different times. For example, in the setting of Example 17, municipalities actually privatized water systems at different times throughout the 1990s: The privatization rate was essentially 0% in 1990, 10% in 1995, and almost 30% by 1999.

The basic DID formula (13.7) is no longer applicable under staggered adoption. However, the parallel trends assumption (Assumption 13.2) used to justify it is still a natural assumption to make; and furthermore the SATT from (13.8)generalizes to

$$
\tau_ {S A T T} = \sum_ {\{i: D _ {i} = 1 \}} \sum_ {t = H _ {i} + 1} ^ {T} (Y _ {i t} (H _ {i}) - Y _ {i t} (\infty)) / \sum_ {\{i: D _ {i} = 1 \}} (T - H _ {i}), \tag {13.10}
$$

which measures the average difference between realized potential outcomes and never-treated potential outcomes for $( i , t )$ affected by treatment. It is then natural to ask: How can we estimate $\tau _ { S A T T }$ under parallel trends in a staggered-adoption design? Before presenting a valid approach, we start by discussing an alluring idea with unintuitive but notable failure modes.

Two-way fixed-effects regression One can readily verify that, under block adoption, the DID estimator $\hat { \tau } _ { D I D }$ from (13.7) is equivalent to the $\hat { \tau }$ coefficient obtained by running a two-way fixed-effects linear regression:

$$
Y _ {i t} \sim \alpha_ {i} + \beta_ {t} + W _ {i t} \tau . \tag {13.11}
$$

This connection is purely algorithmic, and does not rely on well-specification of the linear model associated with (13.11). Mechanistically, we see that the unit fixed effects $\alpha _ { i }$ absorb any additive unit-level baseline effects, and the time fixed effects $\beta _ { t }$ absorb any additive time trends.

Now what’s interesting is that, while the original DID construction (13.7) does not immediately extend to the staggered adoption setting, the two-way regression (13.11) is something that can immediately be run with under any treatment adoption design. Unfortunately, however, this simple idea does not work under the potential outcome specification considered here. Under staggered adoption, the coefficient $\hat { \tau }$ from the two-way regression is in general not consistent for $\tau _ { S A T T } ;$ and, in fact, it’s possible to construct settings where $Y _ { i t } ( H _ { i } ) > Y _ { i t } ( \infty )$ for all pairs $( i , t )$ with $t > H _ { i }$ (i.e., starting treatment always strictly increases outcomes), and yet the regression coefficient $\hat { \tau }$ from the two-way model converges to a negative limit.

To understand the issue here, it is helpful to return to our discussions from Chapter 8, where we observed that the output of any linear regression estimator can always be written as a weighted average of the outcomes, $\begin{array} { r } { \hat { \tau } = \sum _ { i , t } \gamma _ { i t } Y _ { i t } . } \end{array}$ , with the weights $\gamma _ { i t }$ that encode the regression model. The first two panels of Figure 13.1 plot the weights resulting from (13.11) for both a block design (in which case we already have an explicit expression for the weights thanks to (13.7)), and for a staggered adoption design. The seeming paradox from the previous paragraph arises because $\gamma _ { i t }$ can be negative for some treated $( i , t )$ pairs, and thus large positive values of $Y _ { i t } ( H _ { i } ) - Y _ { i t } ( \infty )$ for those $( i , t )$ may push $\hat { \tau }$ to be negative.

Averaged saturated regression There is, however, a simple fix to this issue, recently proposed by Borusyak, Jaravel, and Spiess [2024]. Instead of running the simple two-way regression (13.11), one can run fit a saturated twoway model where each (i, t)-cell under treatment gets its own $\theta _ { i t }$ coefficient,

$$
Y _ {i t} \sim \alpha_ {i} + \beta_ {t} + W _ {i t} \theta_ {i t}. \tag {13.12}
$$

Then, in a second step, one estimates

$$
\hat {\tau} _ {B J S} = \sum_ {W _ {i t} = 1} \hat {\tau} _ {i t} / | \{W _ {i t} = 1 \} |. \tag {13.13}
$$

The individual $\hat { \tau } _ { i t }$ coefficients in this regression will in general not be consistent; however, their aggregate $\hat { \tau } _ { B J S }$ is able to average out these errors in a way that recovers consistency.69 The following result verifies that the $\hat { \tau } _ { B J S }$ in fact has similar properties under staggered adoption as those established for $\hat { \tau } _ { D I D }$ under block adoption.

Because $\hat { \tau } _ { B J S }$ is a linear combination of regression coefficients, it can also be expressed as a weighted average $\begin{array} { r } { \hat { \tau } _ { B J S } = \sum _ { i , t } \gamma _ { i t } Y _ { i t } ; } \end{array}$ and examining these weights can yield further insights about the behavior of the estimator. As seen in the 3rd panel of Figure 13.1, the weights $\gamma _ { i t }$ show that $\hat { \tau } _ { B J S }$ does in fact average information from throughout the panel in a stable-looking way. Furthermore, we see that the weights for all treated time periods are equal (and positive).

Theorem 13.3. In the staggered-adoption design, suppose that some—but not all—units are never treated $( i . e .$ , have $H _ { i } = \infty )$ . Then, under Assumptions 13.1 and 13.2, E $\left[ \hat { \tau } _ { B J S } - \tau _ { S A T T } \right] = 0$ .

Proof. Consider the well-specified linear regression model associated with (13.12) with homoskedastic errors,

$$
Y _ {i t} = \alpha_ {i} + \beta_ {t} + W _ {i t} \theta_ {i t} + \varepsilon_ {i t}, \quad \varepsilon_ {i t} \left| W \sim \mathcal {N} (0, \sigma^ {2}) \right.. \tag {13.14}
$$

Write $\begin{array} { r } { \hat { \tau } _ { B J S } = \sum _ { i , t } \gamma _ { i t } Y _ { i t } } \end{array}$ , with the weights $\gamma _ { i t }$ left implicit for now. By the Gauss-Markov theorem, $\hat { \tau } _ { B J S }$ is the minimum-variance unbiased estimator for $\begin{array} { r } { \theta = \sum _ { i , t } W _ { i t } \theta _ { i t } / \sum _ { i , t } W _ { i t } } \end{array}$ in this model. Now, one can check that any weighted estimator will be unbiased for θ here if and only if

$$
\sum_ {t = 1} ^ {T} \gamma_ {i t} = 0 \text {for all} i = 1, \ldots , n \qquad (\text {so there's no contamination from} \alpha_ {i}),
$$

$$
\sum_ {i = 1} ^ {n} \gamma_ {i t} = 0 \text { for all } t = 1, \dots , T \quad \text {(so there's no contamination from} \beta_ {t}),
$$

$$
\gamma_ {i t} = 1 / \sum_ {i, t} W _ {i t} \text { whenever } W _ {i t} = 1 \quad \text {(to correctly capture the target),}
$$

and so by the Gauss-Markov theorem these equality constraints must in particular be satisfied by the weights underlying $\hat { \tau } _ { B J S }$ . The assumption that some but not all units have $H _ { i } = \infty$ is necessary and sufficient for weights with these properties to exist (and thus for $\hat { \tau } _ { B J S }$ to be feasible) under staggered adoption.

We now argue that these constraints imply our desired result. (We proceed under our originally stated assumptions; the normal errors assumption (13.14) was only used to derive the equality constraints above via the Gauss-Markov theorem). First, the fact that $\gamma _ { i t } = 1 / \sum _ { i , t } W _ { i t }$ for all treated units immediately implies that, under Assumption 13.1,

$$
\hat {\tau} _ {B J S} - \tau_ {S A T T} = \sum_ {i = 1} ^ {n} \sum_ {t = 1} ^ {T} \gamma_ {i t} Y _ {i t} (\infty).
$$

$\begin{array} { r } { \sum _ { t = 1 } ^ { T } \gamma _ { i t } = 0 } \end{array}$ all terms in the sum above without changing the final result,

$$
\hat {\tau} _ {B J S} - \tau_ {S A T T} = \sum_ {i = 1} ^ {n} \sum_ {t = 2} ^ {T} \gamma_ {i t} \left(Y _ {i t} (\infty) - Y _ {i 1} (\infty)\right).
$$

Then, by Assumption 13.2, we get that

$$
\mathbb {E} \left[ \hat {\tau} _ {B J S} - \tau_ {S A T T} \right] = \sum_ {i = 1} ^ {n} \sum_ {t = 2} ^ {T} \gamma_ {i t} \beta_ {t}.
$$

Finally, swapping the order of summation and invoking the fact that $\textstyle \sum _ { i = 1 } ^ { n } \gamma _ { i t } =$ 0 for all t verifies the desired claim. □

Going beyond Theorem 13.3 to also prove consistency requires having the number of units n grow so that the random error term in the proof above, i.e.,

$$
\sum_ {i = 1} ^ {n} \sum_ {t = 2} ^ {T} \gamma_ {i t} \left(Y _ {i t} (\infty) - Y _ {i 0} (\infty) - \beta_ {t}\right) \tag {13.15}
$$

concentrates out; we omit details here. Finally, for inference—as with all DIDtype methods—it is recommended to use algorithms that treat all observations from the same unit as dependent, e.g., the unit-clustered jackknife; see Bertrand, Duflo, and Mullainathan [2004] for a discussion and examples.

## 13.2 Synthetic-control methods

Under the block-adoption setting, difference-in-differences provides a simple estimator of the SATT provided that non-anticipation and parallel trends hold. The parallel trends assumption, however, is a fairly strong function form assumption that can often fail to hold in applications. In this section, we will briefly discuss synthetic-control methods, a class of methods introduced by Abadie, Diamond, and Hainmueller [2010] that allow extension of differencein-differences type methods to settings without parallel trends.

One observable implication of the parallel trends assumption paired with Assumption 13.1 is that, until the event time H, both adopting (or exposed) and non-adopting (or control) units should on average evolve in parallel: Subject to a potential offset parameter $\alpha \in \mathbb { R }$ , we should have

$$
\frac {1}{| \{D _ {i} = 0 \} |} \sum_ {\{D _ {i} = 0 \}} Y _ {i t} \approx \alpha + \frac {1}{| \{D _ {i} = 1 \} |} \sum_ {\{D _ {i} = 1 \}} Y _ {i t}, \quad t = 1, \dots , H. \tag {13.16}
$$

Synthetic control methods are focused on settings where we observe that in fact parallel trends do not hold pre-event, yet would still like to proceed with an event-study analysis. Generally, synthetic control methods seek to mitigate bias from failures of parallel trends by carefully reweighting the control units.

Synthetic difference in differences (SDID) [Arkhangelsky et al., 2021] is a synthetic control method that makes connections to DID explicit—and so this is the variant of synthetic controls we will discuss here. The main idea of SDID is to find non-negative weights $\gamma _ { i }$ with $\textstyle \sum _ { D _ { i } = 0 } \gamma _ { i } = 1$ that restore average parallel trends in the sense of (13.16),

$$
\sum_ {\{D _ {i} = 0 \}} \gamma_ {i} Y _ {i t} \approx \alpha + \frac {1}{| \{D _ {i} = 1 \} |} \sum_ {\{D _ {i} = 1 \}} Y _ {i t}, \quad t = 1, \dots , H, \tag {13.17}
$$

and then estimate the SATT via weighted difference-in-differences

$$
\begin{array}{l} \hat {\tau} _ {S D I D} = \frac {1}{| \{i : D _ {i} = 1 \} |} \sum_ {\left\{i: D _ {i} = 1 \right\}} \left(\frac {1}{T - H} \sum_ {t = H + 1} ^ {T} Y _ {i t} - \frac {1}{H} \sum_ {t = 1} ^ {H} Y _ {i t}\right) \tag {13.18} \\ - \sum_ {\{i: D _ {i} = 0 \}} \gamma_ {i} \left(\frac {1}{T - H} \sum_ {t = H + 1} ^ {T} Y _ {i t} - \frac {1}{H} \sum_ {t = 1} ^ {H} Y _ {i t}\right). \\ \end{array}
$$

There are a number of ways one could seek weights that achieve balance as in (13.17); one simple approach is to choose $\gamma _ { i }$ by minimizing squared-error loss:

$$
\begin{array}{l} \gamma = \operatorname{argmin} _ {\gamma^ {\prime}, \alpha} \left\{\left\| \sum_ {\{D _ {i} = 0 \}} \gamma_ {i} ^ {\prime} Y _ {i (1: H)} - \frac {1}{| \{D _ {i} = 1 \} |} \sum_ {\{D _ {i} = 1 \}} ^ {n} Y _ {i (1: H)} - \alpha \right\| _ {2} ^ {2}: \right. \tag {13.19} \\ \left. \sum_ {\{D _ {i} = 0 \}} \gamma_ {i} ^ {\prime} = 1, \gamma_ {i} ^ {\prime} \geq 0 \right\}. \\ \end{array}
$$

Arkhangelsky et al. [2021] also consider re-weighting pre-event time periods for improved robustness; however, we omit this step here for simplicity, and refer to their paper for a full discussion.

To understand the motivation behind SDID note that, just like in the proofof Theorem 13.2, under non-anticipation,

$$
\begin{array}{l} \hat {\tau} _ {S D I D} - \tau_ {S A T T} \\ = \frac {1}{T - H} \sum_ {t = H + 1} ^ {T} \left(\frac {1}{| \{i : D _ {i} = 1 \} |} \sum_ {\{i: D _ {i} = 1 \}} Y _ {i t} (0) - \sum_ {\{i: D _ {i} = 0 \}} \gamma_ {i} Y _ {i t} (0)\right) \tag {13.20} \\ - \frac {1}{H} \sum_ {t = 1} ^ {H} \left(\frac {1}{| \{i : D _ {i} = 1 \} |} \sum_ {\{i: D _ {i} = 1 \}} Y _ {i t} (0) - \sum_ {\{i: D _ {i} = 0 \}} \gamma_ {i} Y _ {i t} (0)\right). \\ \end{array}
$$

Now, by the re-weighting (13.17), we know that the summands in the pre-event term of the right-hand-side expression are all roughly $\alpha .$ . If similar balance also extends post-event to the unexposed potential outcomes, then summands in the first term above should also all be roughly $\alpha ,$ thus making the error of $\hat { \tau } _ { S D I D }$ . The big question, of course, is in understanding when—and under what conditions—weights obtained via (13.19) will also balance post-event unexposed potential outcomes. The technical tools for doing so are beyond the scope of this presentation. We instead refer to Abadie et al. [2010] and Arkhangelsky et al. [2021] for results of this type; see also Arkhangelsky and Hirshberg [2023] for recent advances.

Numerical example We illustrate the relationship between basic difference in differences and the synthetic control approach via a simple numerical example. We simulate data for $n = 5 0$ units and $T = 2 0$ time periods under block adoption with $H = 1 0$ . Each unit has IID latent parameters $\alpha _ { i }$ and $\beta _ { i }$ that inform trajectory evolution as follows:

$$
\begin{array}{l} \alpha_ {i}, \beta_ {i} \sim \mathcal {N} (0, 1), D _ {i} \sim \text {Bernoulli} \left(1 / \left(1 + e ^ {1 - \beta_ {i}}\right)\right), \\ Y _ {i t} (d) = \alpha_ {i} + \frac {\beta_ {i}   t}{1 0} - d \frac {(t - H) _ {+}}{1 0} + \varepsilon_ {i t}, \quad \varepsilon_ {i t} \sim \mathcal {N} \left(0, \frac {1}{1 0 ^ {2}}\right). \tag {13.21} \\ \end{array}
$$

This design satisfies non-anticipation as in Assumption 13.1. However, it does not have random treatment assignment or parallel trends as in Assumption 13.2: Units with large values of $\beta _ { i }$ both have more positive baseline trends, and are more likely to take up treatment. The DID estimator is thus not expected to be consistent here.

Figure 13.2 shows results from applying both the DID estimator (13.7) and the SDID implementation of the synthetic control approach as in (13.18) on one draw of data following (13.21). The DID estimator is confounded because exposure $D _ { i }$ is correlated with the latent factor $\beta _ { i }$ that also affects trends;and in fact we observe that the average outcomes for exposed and unexposed units do not evolve in parallel even before treatment. In contrast, SDID reweights unexposed units with the aim of restoring parallel trends. In our setting the treatment effect is negative; and SDID correctly recovers the sign of the treatment effect here whereas DID does not.

## 13.3 Bibliographic notes

Our presentation of event study designs fits within the tradition a broad literature on panel data methods in econometrics whose surface we’ve only scratched here. Arellano [2003] and Wooldridge [2010] provide broad textbook overviews of the area. Arkhangelsky and Imbens [2023] provide an extensive review of recent developments in the area. The approach used here to define potential outcomes and causal estimands is adapted from Athey and Imbens [2022].

The topic of treatment heterogeneity in the context of two-way models has been the focus of a considerable amount of discussion in recent years; see de Chaisemartin and D’Haultfoeuille [2018] for an early paper drawing attention to the phenomenon and Chiu et al. [2023] for a recent discussion and review. Here, we restricted our analysis on estimating $\tau _ { S A T T }$ as in (13.10). However, under staggered adoption, parallel trends allow for identification of a broader family of cohort-wise treatment effect estimates that may be relevant in applications [Borusyak, Jaravel, and Spiess, 2024, Callaway and Sant’Anna, 2021, Sun and Abraham, 2021]:

$$
\tau_ {S A T T} ^ {h, t} = \sum_ {\{i: H _ {i t} = h \}} (Y _ {i t} (h) - Y _ {i t} (\infty)) / | \{i: H _ {i t} = h \} |. \tag {13.22}
$$

In particular, when there are no never-treated units as required in Theorem 13.3, then $\tau _ { S A T T }$ is not identified under parallel trends, but some cohort-wise effects will still be identifiable as long as there’s some variation in the treatment start time.

The synthetic control method was introduced by Abadie and Gardeazabal [2003] and formalized by Abadie, Diamond, and Hainmueller [2010]. Extensions of synthetic controls with double-differencing structure—including the SDID method presented here—are discussed in Arkhangelsky et al. [2021], Ben-Michael, Feller, and Rothstein [2021] and Shen et al. [2023]. Arkhangelsky and Hirshberg [2023] study large-sample properties of synthetic control estimators when exposure is non-random and depends on unobservables.

From a formal perspective, synthetic control methods are often studied under an interactive fixed-effects model, where we posit

$$
Y _ {i t} = A _ {i.} \cdot B _ {t.} + W _ {i t} \tau + \varepsilon_ {i t}, \quad A \in \mathbb {R} ^ {n \times k}, \quad B \in \mathbb {R} ^ {T \times k}, \quad \mathbb {E} [ \varepsilon_ {i t} | W ] = 0. (1 3. 2 3)
$$

Here, unlike in the standard two-way specification (13.11), the i-th unit has a k-dimensional “type” $A _ { i } .$ that interacts with $B _ { t }$ · in the t-th time period. In the context of this model, showing that synthetic controls work involves proving that the γ-weighting effectively eliminate bias due to imbalance in the unobserved types $A _ { i } .$ ; see Arkhangelsky et al. [2021] for formal results within this paradigm.

An alternative approach to estimating τ under the interactive fixed-effects model involves fitting the full model (13.23)—including the unobserved baseline term $A B ^ { \prime } .$ —via low-rank matrix estimation methods. Examples of this approach include Bai [2009], who use least-squares estimation, and Athey et al. [2021], who use nuclear-norm penalization. Agarwal et al. [2021], Lei and Ross [2023] and Xu [2017] consider a setting where a low-rank structure is assumed on the never-treated potential outcomes, but we don’t assume additive treatment effects as in (13.23). They then use matrix completion methods to estimate this low-rank structure and impute never-treated potential outcomes in the post-event periods; the SATT is finally estimated by comparing realized and imputed outcomes in these periods.
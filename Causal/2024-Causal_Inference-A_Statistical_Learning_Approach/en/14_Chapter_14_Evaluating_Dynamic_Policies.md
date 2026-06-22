# Chapter 14 Evaluating Dynamic Policies

In the previous chapter, we considered methods for event studies where some units adopted a treatment (i.e., switched their treatment status from off-toon), and we wanted to measure the effect of making this switch. Results from event studies can be helpful in informing whether other units might also benefit from adopting the treatment. However, event-study designs—and associated methods such as difference in differences and synthetic controls—are less helpful for is in guiding dynamic decision making. Their limitations are perhaps best understood in the context of examples.

Example 18. During a financial downturn, central banks sometimes use quantitative easing to mitigate the risks of a long-term recession. During quantitative easing, the central bank seeks to increase market liquidity by purchasing government bonds and other assets. Some quantitative easing may help stimulate the economy and avoid a recession; however, too much quantitative easing—or quantitative easing that lasts for too long—may lead to problems with excessive inflation [Boehl, Goy, and Strobel, 2024].

Example 19. Antiretroviral therapy (ART) is a crucial drug in caring for HIV-positive patients. It is understood that HIV reduces CD4 white blood cell count, and that patients are at risk of contracting AIDS-defined illnesses once CD4 count is low. The use of ART can help preserve CD4 counts and thus prevent AIDS, but it is a very intensive form of medication with a number of side effects. The topic of when to start ART has thus received considerable attention in the medical literature. Traditional guidelines for treating HIV recommend beginning ART only once CD4 count fall below a given threshold; but recent evidence is in favor of starting ART as soon as HIV is diagnosed [Group, 2015].

It is clear that a successful application of quantitative easing requires judicious consideration of when to start the intervention, how much liquidity to provide, and when to stop. However, event-study methods provide very little guidance on questions of this type. The parallel trends assumption underlying difference-in-differences methods effectively rules out the possibility that, during a given crisis, there may be some countries that need quantitative easing (i.e., they would fall into a recession without intervention) and others that don’t (i.e., even without intervention they would be OK). Synthetic control methods could be used to study the effect of ART— or the initial effect of quantitative easing—but do not readily give guidance as to when to start or stop the interventions.

This chapter presents a fully flexible, potential-outcome based approach to modeling causal effects over time that allows for arbitrary treatment assignment dynamics and carryover effects. Throughout, we will assume that we have data on $i = 1 , \ldots , n$ patients, observed at times $t = 1 , \dots , T$ . At each time point, we observe a set of (time-varying) covariates $X _ { i t }$ as well as a treatment assignment $W _ { i t } \in \{ 0 , 1 \}$ . Finally, once we reach time $T .$ , we also observe an outcome $Y _ { i } \in \mathbb { R }$ . Throughout this chapter, we will take units i to be sampled IID from a superpopulation.

We model causal effects using the potential outcome specification below that allows for arbitrary treatment dynamics. Note that this model implicitly encodes the fact that time-t observables are only affected by actions taken up to time $t ,$ and not future actions, thus generalizing the non-anticipation condition (Assumption 13.1) used in the event-study setting.

Definition 14.1. A dynamic decision process with time-horizon $T$ is characterized by outcomes time-varying covariates $X _ { i t } \in \mathcal { X } _ { t }$ and outcomes $Y _ { i } \in \mathbb { R }$ , with potential outcomes that make each observable responsive to all past treatment assignments. For each $X _ { i t }$ , we define $2 ^ { t - 1 }$ potential outcomes $X _ { i t } ( w _ { 1 : ( t - 1 ) } )$ such that $X _ { i t } = X _ { i t } ( W _ { i ( 1 : ( t - 1 ) ) } )$ , while for the final outcome we have $2 ^ { T }$ potential outcomes $Y _ { i } ( w _ { 1 : T } )$ such that $Y _ { i } = X _ { i t } ( W _ { i ( 1 : T ) } )$ .

Next, we need to define an estimand. In the dynamic setting, the number of potential treatment allocation rules grows exponentially with the horizon $T .$ , and so does the number of questions we can ask. One simple estimand to consider is the expected outcome under some pre-specified treatment rule $w \in \left\{ 0 , 1 \right\} ^ { T } , \mathrm { i . e . , } V ( w ) = \mathbb { E } \left[ Y _ { i } ( w ) \right]$ . Such estimands, however, are often not relevant to practice as they rule out dynamic decision making. Suppose, for example, that we’re studying cancer therapy and are asking to estimate $V ( w )$ for the treatment rule that starts chemotherapy one year after cancer diagnosis. Then, if some patients enter remission through other means before they reach the one-year mark, evaluating $V ( w )$ would still require starting chemotherapy at this point—even if it doesn’t make clinical sense.

In practice, it is often more relevant evaluate treatment rules that take into account time-varying covariates. For example, we might ask about the benefit of starting chemotherapy one year after diagnosis among patients who have not yet entered remission, or we might ask about starting quantitative easing at a point when interest rates have hit 0 but economic activity is still weak. We can define a number of relevant estimands of this type via the lens of policy evaluation, in a generalization of our discussion from Chapter 5.1.

Definition 14.2. A dynamic policy is a set of mappings $\pi _ { t } : \mathcal { X } _ { t }  \{ 0 , 1 \}$ that prescribe a treatment $\pi _ { t } ( X _ { i t } )$ given the current state $X _ { i t }$ . The value of the policy π is

$$
V (\pi) = \mathbb {E} \left[ Y _ {i} (\pi_ {1} (X _ {i 1}),   \pi_ {2} (X _ {i 1},   \pi_ {1} (X _ {i 1}),   X _ {i 2} (\pi_ {1} (X _ {i 1})),   \ldots) \right], \tag {14.1}
$$

i.e., it captures the expected reward from choosing treatment according to $\pi$ in a dynamic decision process.

The intricate notation in (14.1) highlights the complex causal structure inherent to dynamic decision-making problems: The treatment decision taken at time t depends on $X _ { i t } .$ which in turn depends on the treatment decision taken at time $t - 1$ and thus $X _ { i ( t - 1 ) }$ , etc., until we get back to the initial state $X _ { i 1 }$ . Thankfully, these statistical objects are amenable to tractable analysis via a recursive, dynamic-programming-style approach.

## 14.1 Sequential unconfoundedness

In order to estimate the quantities defined above we need to collect data, and to make assumptions on how the treatment is assigned in the experiment in order to identify the estimands. Here, we will do so using a sequential unconfoundedness (or sequential ignorability) which posits that, at every time point, treatment is as good as random given the data observed at the time:

$$
\left\{\text {(potential outcomes after time} t) \right\} \perp W _ {i t} \mid \left\{\text {(history up to time} t) \right\}. \tag {14.2}
$$

This condition is formalized below. Here, and throughout the rest of this chapter, we will use the notational short-hand $X _ { i ( T + 1 ) } : = Y _ { i }$ (i.e., the outcome is the state variable measured after we cross the time-horizon T ) in order to simplify expressions.

Assumption 14.1. Given a dynamic decision process, we further assume that our treatment sequence is sequentially unconfounded such that, for all $t =$1, . . . , T ,70

$$
\left[ \left\{X _ {i (t + 1)} (W _ {i (1: (t - 1))}, w) \right\} _ {w = 0, 1} \perp W _ {i t} \right] \mid \left\{X _ {i 1}, W _ {i 1}, \dots W _ {i (t - 1)}, X _ {i t} \right\}. \tag {14.3}
$$

Remark 14.1. In principle, one might also be interested in a design more directly comparable to a standard randomized controlled trial where treatment is fully randomized,

$$
\left\{\text {(all potential outcomes)} \right\} \perp W _ {1: T}. \tag {14.4}
$$

This, however, can again lead to non-sense treatment assignments $( \mathrm { e . g . }$ , again in the case of a cancer trial, assigning a patient to chemotherapy after they have already reached remission), and so the literature on dynamic treatment rules has mostly focused on methods that work under the more flexible sequential unconfoundedness setting.

The statistical consequences of sequential unconfoundedness are perhaps easiest to express in terms of properties of a sequential factorization of the joint distribution of $( X _ { i 1 } , \dots , X _ { i T } , X _ { i ( T + 1 ) } )$ under the policy $\pi _ { : }$ , where as discussed above we write $X _ { i ( T + 1 ) } = Y _ { i }$ . As usual, we write $\mathbb { E } \left[ \cdot \right]$ and $\mathbb { P } \left[ \cdot \right]$ to denote expectations and probabilities for the distribution we collect data from. We can always sequentially factor this distribution as

$$
\mathbb {P} \left[ X _ {1}, W _ {1}, \dots , W _ {T}, X _ {T + 1} \right] = \mathbb {P} \left[ X _ {1} \right] \prod_ {t = 1} ^ {T} \mathbb {P} \left[ W _ {t} \mid S _ {t} \right] \mathbb {P} \left[ X _ {t + 1} \mid W _ {t}, S _ {t} \right], \tag {14.5}
$$

where $S _ { t } = \{ X _ { 1 } , W _ { 1 } , \ldots , W _ { t - 1 } , X _ { t } \}$ denotes all information until the period-t treatment is chosen. For the purpose of policy evaluation, it is convenient to also introduce off-policy measures $\mathbb { E } _ { \pi } \left[ \cdot \right]$ and $\mathbb { P } _ { \pi } \left[ \cdot \right]$ to describe distributions that would instead arise from assigning treatment according to $\pi$ as in Definition 14.2. Given this notation, we can concisely write the policy value as $V ( \pi ) = \mathbb { E } _ { \pi } \left[ X _ { T + 1 } \right]$ . We can also again sequentially factor the distribution

$$
\mathbb {P} _ {\pi} \left[ X _ {1}, W _ {1}, \ldots , W _ {T}, X _ {T + 1} \right]
$$

$$
= \mathbb {P} _ {\pi} \left[ X _ {1} \right] \prod_ {t = 1} ^ {T} \mathbb {P} _ {\pi} \left[ W _ {t} \mid S _ {t} \right] \mathbb {P} _ {\pi} \left[ X _ {t + 1} \mid W _ {t}, S _ {t} \right]. \tag {14.6}
$$

A key implication of sequential unconfoundedess is that it allows us to simplify (14.6) by guaranteeing that some terms in the factorization do not depend on the policy $\pi$ of interest. The result below follows immediately from (14.3).

**Table 14.1: A synthetic two-period example reproduced from Hern´an and Robins [2020, Table 20.1].**

<table><tr><td>n</td><td> $X_{i1}$ </td><td> $W_{i1}$ </td><td> $X_{i2}$ </td><td> $W_{i2}$ </td><td>Mean Y</td></tr><tr><td>2400</td><td>0</td><td>0</td><td>0</td><td>0</td><td>84</td></tr><tr><td>1600</td><td>0</td><td>0</td><td>0</td><td>1</td><td>84</td></tr><tr><td>2400</td><td>0</td><td>0</td><td>1</td><td>0</td><td>52</td></tr><tr><td>9600</td><td>0</td><td>0</td><td>1</td><td>1</td><td>52</td></tr><tr><td>4800</td><td>0</td><td>1</td><td>0</td><td>0</td><td>76</td></tr><tr><td>3200</td><td>0</td><td>1</td><td>0</td><td>1</td><td>76</td></tr><tr><td>1600</td><td>0</td><td>1</td><td>1</td><td>0</td><td>44</td></tr><tr><td>6400</td><td>0</td><td>1</td><td>1</td><td>1</td><td>44</td></tr></table>

Proposition 14.1. Under sequential unconfoundedness, terms in the factorization that don’t integrate over Wt don’t depend on the policy π, i.e.,

$$
\mathbb {P} _ {\pi} \left[ X _ {1} \right] = \mathbb {P} \left[ X _ {1} \right] \quad \mathbb {P} _ {\pi} \left[ X _ {t + 1} \mid S _ {t}, W _ {t} \right] = \mathbb {P} \left[ X _ {t + 1} \mid S _ {t}, W _ {t} \right]. \tag {14.7}
$$

Treatment-confounder feedback Before introducing methods that work under sequential unconfoundedness, it is worth highlighting a subtle difficulty that arises in this setting not present in the basic (single-period) design, namely treatment-confounder feedback [Robins, 1986]. To see what may go wrong, consider the following simple example adapted from Hern´an and Robins [2020], modeled after an ART trial with $T = 2$ time periods. Here, $X _ { i t } ~ \in ~ \{ 0 , 1 \}$ denotes CD4 count (1 is low, i.e., bad), and suppose that $X _ { i 1 } = 0$ for everyone (no one enters the trial very sick), and $X _ { i 1 }$ is randomized with probability 0.5 of receiving treatment. Then, at time period 2, we observe $X _ { i 2 }$ and assign treatment $X _ { i 2 } = 1$ with probability 0.4 if $X _ { i 2 } = 0$ and with probability 0.8 if $X _ { i 2 } = 1$ . In the end, we collect a health outcome Y . This is a sequential randomized experiment.

We observe data as in Table 14.1, wherethe last column is the mean outcome for everyone in that row. Our goal is to estimate $\tau = \mathbb { E } \left[ Y \left( \underline { { 1 } } \right) - Y \left( \underline { { 0 } } \right) \right]$ , i.e., the difference between the always treat and never treat rules. How should we do this? As a preliminary, it’s helpful to note that the treatment obviously does nothing. In the first time period,

$$
\mathbb {E} \left[ Y _ {i} \mid W _ {i 1} = 0 \right] = \mathbb {E} \left[ Y _ {i} \mid W _ {i 1} = 1 \right] = 6 0,
$$

and this is obviously a causal quantity (since $W _ { i 1 }$ was randomized). Moreover, in the second time period we see by inspection that

$$
\mathbb {E} \left[ Y _ {i} \mid W _ {i 2} = 0, W _ {i 1} = w _ {1}, X _ {i 2} = x \right] = \mathbb {E} \left[ Y _ {i} \mid W _ {i 2} = 1, W _ {i 1} = w _ {1}, X _ {i 2} = x \right],
$$

**Table 14.2: Responder types in the setting of Table 14.1.**

<table><tr><td></td><td>$ W_{i1}=0 $</td><td>$ W_{i1}=1 $</td></tr><tr><td>stable</td><td>$ X_{i2}=0 $</td><td>$ X_{i2}=0 $</td></tr><tr><td>responder</td><td>$ X_{i2}=1 $</td><td>$ X_{i2}=0 $</td></tr><tr><td>acute</td><td>$ X_{i2}=1 $</td><td>$ X_{i2}=1 $</td></tr></table>

for all values of $w _ { 1 }$ and x, and again the treatment does nothing.

However, when targeting the total effect of always treatment vs. never treatment, some simple estimation strategies that served us well in the non-dynamic setting do not get the right answer. In particular, here are some strategies that do not get the right answer:

• Ignore adaptive sampling, and use

$$
\begin{array}{l} \hat {\tau} = \widehat {\mathbb {E}} [ Y | W = \underline {{1}} ] - \widehat {\mathbb {E}} [ Y | W = \underline {{0}} ] \\ = \frac {6 4 0 0 \times 4 4 + 3 2 0 0 \times 7 6}{6 4 0 0 + 3 2 0 0} - \frac {2 4 0 0 \times 5 2 + 2 4 0 0 \times 8 4}{2 4 0 0 + 2 4 0 0} \\ = 5 4. 7 - 6 8 = - 1 3. 3. \\ \end{array}
$$

• Stratify by CD4 count at time 2, to control for adaptive sampling:

$$
\hat {\tau} _ {0} = \mathbb {E} \left[ Y \mid W = \underline {{1}}, X _ {i 2} = 0 \right] - \mathbb {E} \left[ Y \mid W = \underline {{0}}, X _ {i 2} = 0 \right] = 7 6 - 8 4 = - 8
$$

$$
\hat {\tau} _ {1} = \mathbb {E} \left[ Y \mid W = \underline {{1}}, X _ {i 2} = 1 \right] - \mathbb {E} \left[ Y \mid W = \underline {{0}}, X _ {i 2} = 1 \right] = 4 4 - 5 2 = - 8
$$

$$
\hat {\tau} = \frac {(3 2 0 0 + 2 4 0 0) \hat {\tau} _ {0} + (6 4 0 0 + 2 4 0 0) \hat {\tau} _ {1}}{3 2 0 0 + 2 4 0 0 + 6 4 0 0 + 2 4 0 0} = - 8.
$$

The problem with the first strategy is obvious (we need to correct for biased sampling). But the problem with the second strategy is more subtle. We know via sequantial randomization that

$$
Y _ {i} (\dots) \perp W _ {i 2} \mid X _ {i 2},
$$

and this seems to justify stratification. But what we’d actually need for stratification is:

$$
Y _ {i} (\dots) \perp (W _ {i 1}, W _ {i 2}) \mid X _ {i 2},
$$

and this is not true by design.

To see what could go wrong, imagine that there are 3 types of people (stable, responder, acute), and tabulate their time-2 CD4 values as in Table14.2. These types—often called principal strata—are are unobservable but can still provide insights.71 For example:

• E $\left\lceil Y \right\rceil W = \underline { { 1 } } , X _ { i 2 } = 0 \rceil$ is an average over stable or responder patients, whereas $\mathbb { E } \left[ Y \big | W = \underline { { \bar { 0 } } } , X _ { i 2 } = 0 \right]$ is simply an average over stable patients. So the difference $\hat { \tau } _ { 0 }$ is not estimating a proper causal quantity.

• E $\lceil Y \mid W = \underline { { 1 } } , X _ { i 2 } = 1 \rceil$ is an average over acute patients, whereas in contrast $\mathbb { E } \left[ Y \vert W = \underline { { 0 } } , X _ { i 2 } ^ { \overline { { } } } = 1 \right]$ is an average over responder or acute patients. So the difference $\hat { \tau } _ { 1 }$ is not estimating a proper causal quantity.

In other words, in sequentially randomized trials, simple stratification estimators do not successfully control for confounding.

Sequential inverse-propensity weighting Since stratification doesn’t work, we now move to study a family of approaches that do. Here, we focus on estimating the value of a policy $V ( \pi )$ as in (14.1); note that evaluating a fixed treatment sequence is a special case of this strategy. To this end, it’s helpful to define some more notation: Writing $S _ { t }$ for the information available at time t as before, we define the value function72

$$
V _ {\pi , t} (S _ {t}) = \mathbb {E} _ {\pi} [ Y | S _ {t} ] \tag {14.8}
$$

that measures the expected reward we’d get if we were to start following π given our current state as captured by $S _ { t }$ .

This notation lets us concisely express a helpful principle behind fruitful estimation of $V ( \pi )$ : By the chain rule, we see that

$$
\begin{array}{l} \begin{array}{r l} \mathbb {E} _ {\pi} \left[ V _ {\pi , t + 1} (S _ {t + 1}) \mid S _ {t} \right] & = \mathbb {E} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y \mid S _ {t + 1} \right] \mid S _ {t} \right] \\ & \quad \mathbb {E} _ {\pi} \left[ Y \mid S _ {t} \right] - \mathbb {E} _ {\pi} (S) \end{array} \tag {14.9} \\ = \mathbb {E} _ {\pi} \left[ Y \mid S _ {t} \right] = V _ {\pi , t} (S _ {t}). \\ \end{array}
$$

The implication is that, given a good estimate of $V _ { \pi , t + 1 }$ , all we need to be able to do is to get a good estimate of $V _ { \pi , t } ;$ then we can recurse our way backwards to $V ( \pi )$ . The question is then how we choose to act on this insight.

One simple way to do so is via an inverse-propensity weighting (IPW) construction. If we had access to $V _ { \pi , t + 1 } \big ( S _ { i ( t + 1 ) } \big )$ and many samples with $S _ { i t } = s _ { t }$ , then applying the basic IPW construction from Chapter 2 under (14.3) would suggest using

$$
\widehat {V} _ {\pi , t} (s _ {t}) = \frac {1}{| \{i : S _ {i t} = s _ {t} \} |} \sum_ {\{i: S _ {i t} = s _ {t} \}} \frac {1 (\{W _ {i t} = \pi (s _ {t}) \})}{\mathbb {P} [ W _ {i t} = \pi (s _ {t}) | S _ {i t} = s _ {t} ]} V _ {\pi , t + 1} (S _ {i (t + 1)}).
$$

A recursive application of this principle results in the IPW estimator of the policy value,

$$
\widehat {V} _ {I P W} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \gamma_ {i T} (\pi) Y _ {i}, \tag {14.10}
$$

$$
\gamma_ {i t} (\pi) = \gamma_ {i (t - 1)} (\pi) \frac {1 \left(\{W _ {t} = \pi_ {t} (S _ {t}) \}\right)}{\mathbb {P} \left[ W _ {t} = \pi_ {t} (S _ {t}) \mid S _ {t} \right]},
$$

where $\gamma _ { i 0 } ( \pi ) = 1$ . This estimator averages outcomes whose treatment trajectory exactly matches π, while applying an IPW correction for selection effects due to measured (time-varying) confounders. We show below that the IPW estimator is unbiased if we know the inverse-propensity weights $\gamma _ { i T }$ exactly, and give an expression for its asymptotic variance.

Theorem 14.2. Consider a dynamic decision process as in Definition $1 \llangle . 1$ with data collected under sequential unconfoundedness as in Assumption 14.1. Suppose furthermore that we seek to evaluate a policy π for which strong overlap holds, i.e.,

$$
\mathbb {P} \left[ W _ {t} = \pi_ {t} (S _ {t})   |   S _ {t} \right] \geq_ {a. s.} \eta , \tag {14.11}
$$

and that our outcomes are almost surely bounded, $| Y | \le _ { a . s }$ . M for some $M <$ ∞. Then, the IPW estimator from (14.10) is unbiased with and asymptotically normal sampling distribution,73

$$
\begin{array}{l} \mathbb {E} \left[ \widehat {V} _ {I P W} (\pi) \right] = V (\pi), \quad \sqrt {n} \left(\widehat {V} _ {I P W} (\pi) - V (\pi)\right) \Rightarrow \mathcal {N} \left(0, \sigma_ {I P W} ^ {2}\right) \\ \sigma_ {I P W} ^ {2} = \mathbb {E} _ {\pi} \left[ Y ^ {2} / \prod_ {t = 1} ^ {T} \mathbb {P} \left[ W _ {t} = \pi_ {t} (S _ {t}) \mid S _ {t} \right] \right] - V ^ {2} (\pi). \tag {14.12} \\ \end{array}
$$

Proof. We verify unbiasedness via backwards induction, starting from $t = T$ , and argue that

$$
V _ {\pi , t} (S _ {t}) = \mathbb {E} \left[ \frac {\gamma_ {T} (\pi)}{\gamma_ {t - 1} (\pi)} Y \mid S _ {t} \right] \tag {14.13}
$$

for all $t = 0 , \ldots , T$ , where we use $S _ { 0 } = \emptyset$ and $\gamma _ { - 1 } ( \pi ) = \gamma _ { 0 } ( \pi ) = 1$ . The base case, with $t = T$ , corresponds exactly to the unbiasedness result in Theorem 2.2, while the final step with $t = 0$ corresponds to our desired claim. For the inductive step, suppose that (14.13) holds for $t + 1$ . Then, we can verify that

$$
\mathbb {E} \left[ \frac {\gamma_ {T} (\pi)}{\gamma_ {t - 1} (\pi)} Y \mid S _ {t} \right] = \mathbb {E} \left[ \frac {\gamma_ {t} (\pi)}{\gamma_ {t - 1} (\pi)} \mathbb {E} \left[ \frac {\gamma_ {T} (\pi)}{\gamma_ {t} (\pi)} Y \mid S _ {t + 1} \right] \mid S _ {t} \right]
$$

$$
= \mathbb {E} \left[ \frac {1 \left(\{W _ {t} = \pi_ {t} (S _ {t}) \}\right)}{\mathbb {P} \left[ W _ {t} = \pi_ {t} (S _ {t}) \mid S _ {t} \right]} V _ {\pi , t + 1} (S _ {t + 1}) \right]
$$

$$
= \mathbb {E} \left[ \frac {1 \left(\left\{W _ {t} = \pi_ {t} (S _ {t}) \right\}\right)}{\mathbb {P} \left[ W _ {t} = \pi_ {t} (S _ {t}) \mid S _ {t} \right]} \mathbb {E} _ {\pi} \left[ Y _ {T} \mid S _ {t + 1} \right] \right]
$$

$$
= \mathbb {E} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y _ {T} \mid S _ {t} \right] \right] = V _ {\pi , t} (S _ {t}),
$$

where the first equality follows because $\gamma _ { t } ( \pi ) / \gamma _ { t - 1 } ( \pi )$ is St-measurable, the second follows by invoking the inductive hypothesis and by definition of $\gamma _ { t } ( \pi ) / \gamma _ { t - 1 } ( \pi )$ , the fourth equality follows by sequential unconfoundedness, and the third and last are just (14.9).

Given unbiasedness and IID sampling of units, the central limit theorem immediately follows with

$$
\sigma_ {I P W} ^ {2} = \mathbb {E} \left[ \gamma_ {T} ^ {2} (\pi) Y ^ {2} \right] - V ^ {2} (\pi),
$$

and it only remains to derive an explicit expression for the 2nd moment term above. Now, by repeating the same IPW argument as used above,

$$
\mathbb {E} \left[ \gamma_ {T} ^ {2} (\pi) Y ^ {2} \right] = \mathbb {E} _ {\pi} \left[ \gamma_ {T} (\pi) Y ^ {2} \right].
$$

Under the off-policy measure $\mathbb { E } _ { \pi } \left[ \cdot \right]$ , we always have $W _ { t } = \pi _ { t } ( S _ { t } )$ , and so

$$
\gamma_ {T} (\pi) = 1 \Big / \prod_ {t = 1} ^ {T} \mathbb {P} \left[ W _ {t} = \pi_ {t} (S _ {t}) \mid S _ {t} \right]
$$

almost surely, thus providing the expression claimed.

Remark 14.2. As discussed in Chapter 12, we can often improve the asymptotic precision of IPW via self-normalization:

$$
\widehat {V} _ {S I P W} (\pi) = \sum_ {i = 1} ^ {n} \gamma_ {i T} (\pi) Y _ {i} / \sum_ {i = 1} ^ {n} \gamma_ {i T} (\pi). \tag {14.14}
$$

Under the conditions of Theorem 14.2,

$$
\sqrt {n} \left(\widehat {V} _ {S I P W} (\pi) - V (\pi)\right) \Rightarrow \mathcal {N} \left(0, \sigma_ {S I P W} ^ {2}\right)
$$

$$
\sigma_ {S I P W} ^ {2} = \mathbb {E} _ {\pi} \left[ (Y - V (\pi)) ^ {2} / \prod_ {t = 1} ^ {T} \mathbb {P} \left[ W _ {t} = \pi_ {t} (S _ {t}) \mid S _ {t} \right] \right]. \tag {14.15}
$$

This result can be established by following the same proof strategy as in, e.g., Theorem 12.3. The change in precision from self-normalization is

$$
\begin{array}{l} \sigma_ {I P W} ^ {2} - \sigma_ {S I P W} ^ {2} = \left(\mathbb {E} _ {\pi} \left[ \left(\prod_ {t = 1} ^ {T} \mathbb {P} \left[ W _ {t} = \pi_ {t} \left(S _ {t}\right) \mid S _ {t} \right]\right) ^ {- 1} \right] - 1\right) V ^ {2} (\pi) \tag {14.16} \\ + 2 \operatorname{Cov} _ {\pi} \left[ Y, \left(\prod_ {t = 1} ^ {T} \mathbb {P} \left[ W _ {t} = \pi_ {t} (S _ {t}) \mid S _ {t} \right]\right) ^ {- 1} \right]. \\ \end{array}
$$

The first summand is always positive (and often large); however, the second summand can be negative—and could in principle be negative enough to make self-normalized IPW less precise than the basic IPW estimator.

## 14.2 Doubly robust estimation

Like in the single-period case discussed in Chapter 3, it is possible to improve the precision and robustness of IPW by augmenting it with a regression adjustment. Here, we show how to construct an augmented estimator for dynamic treatment rules, and verify that the resulting estimator is has a strong double robustness property: It can trade off accuracy of the regression and propensityscore models and achieve the parametric $1 / \sqrt { n } \mathrm { - r a t e }$ of convergence even if input non-parametric regressions converge at slower rates.

Backwards regression adjustment Like in Chapter 3, our doubly robust construction starts by using sequential unconfoundedness to motivate an alternative, regression-based approach to estimating the value of a policy π. By combining sequential unconfoundedness (and in particular its implication highlighted in Proposition 14.1) with (14.9), we see that

$$
V _ {\pi , t} (s) = \mathbb {E} \left[ V _ {\pi , t + 1} (S _ {t + 1})   \big |   S _ {t} = s,   W _ {t} = \pi_ {t} (s) \right]. \tag {14.17}
$$

Thus, if we know $V _ { \pi , t + 1 } ( \cdot )$ or have a reasonably accurate estimate of it, we can estimate $V _ { \pi , t } ( \cdot )$ via non-parametric regression with $V _ { \pi , t + 1 } ( \cdot )$ as the outcome.

This structure suggests the following backward regression approach to estimating the policy value:

• First, using samples i that exactly follow the target policy, i.e., with $W _ { i t } =$ $\pi ( S _ { i t } )$ for all $t = 1 , \dots , T$ , learn $\widehat { V } _ { \pi , T } ( \cdot )$ via non-parametric regression $Y _ { i } \sim V _ { \pi , T } ( S _ { i T } )$ .

• Next, iteratively for $t = T - 1 , T - 2 , . . . , 1$

– Using samples i that exactly follow the target policy up to time $t ,$ i.e., with $W _ { i t ^ { \prime } } = \pi ( S _ { i t ^ { \prime } } )$ for all $t ^ { \prime } = 1 , \ldots , t .$ , learn $\widehat { V } _ { \pi , t } ( \cdot )$ via nonparametric regression $\hat { V } _ { \pi , t + 1 } ( S _ { i ( t + 1 ) } ) \sim V _ { \pi , t } ( S _ { i t } )$ .

• Finally, form the regression estimator for the value of π

$$
\widehat {V} _ {R E G} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \widehat {V} _ {\pi , 1} (S _ {i 1}). \tag {14.18}
$$

This backwards-regression approach can be implemented via generic machine learning. However, tailored models may also be helpful; for example, structural nested mean models [Robins, 1994] are designed to avoid spurious detection of causal effects under a null where the intervention has no effect.

A regression-augmented estimator Where there’s an IPW and a regression based estimator, there’s going to be a doubly robust estimator also. In the the last step of the backward-regression estimator (14.17), we averaged time-1 value-function estimates $\widehat { V } _ { \pi , 1 } ( \bar { X } _ { 1 } )$ to obtain $\widehat { V } _ { R E G } ( \pi )$ . Now, given the backward-regression construction, it’s likely we trust the time-2 value function estimates $\widehat { V } _ { \pi , 2 }$ a little more than the time-1 estimates; and in this case we may consider using the basic augmented IPW (AIPW) construction from Chapter 3 to leverage these $\widehat { V } _ { \pi , 2 }$ estimates for improved precision:

$$
\widehat {V} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\widehat {V} _ {\pi , 1} (X _ {i 1}) + \gamma_ {i 1} (\pi) \left(\widehat {V} _ {\pi , 2} (X _ {i 1}, W _ {i 1}, X _ {i 2}) - \widehat {V} _ {\pi , 1} (X _ {i 1})\right)\right).
$$

Qualitatively, the idea here is that on the event where $W _ { i 1 }$ matches $\pi$ in the first step, we can use $\widehat { V } _ { \pi , 2 }$ to debias $\widehat { V } _ { \pi , 1 } .$ ; here, the $\gamma _ { i t }$ are the inverse-propensity weights as in (14.10).

Then next natural question, of course, is why not debias $\widehat { V } _ { \pi , 2 }$ using $\widehat { V } _ { \pi , 3 }$ when $W _ { i 2 }$ also matches $\pi$ in the second step? And once we do so, why not proceed until the end of the time-horizon when we can observe the realized outcome $Y ?$ This recursive construction in fact works, and yields a natural generalization of the AIPW estimator of Robins, Rotnitzky, and Zhao [1994] discussed in Chapter 3 to the dynamic setting:

$$
\begin{array}{l} \widehat {V} _ {A I P W} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\widehat {V} _ {\pi , 1} (X _ {i 1}) \right. \tag {14.19} \\ \left. + \sum_ {t = 1} ^ {T} \hat {\gamma} _ {i t} (\pi) \left(\widehat {V} _ {\pi , t + 1} (S _ {i (t + 1)}) - \widehat {V} _ {\pi , t} (S _ {i t})\right)\right), \\ \end{array}
$$

where we used a notational convention that $\widehat { V } _ { \pi , T + 1 } ( S _ { i ( T + 1 ) } ) = Y _ { i }$ since by time $T + 1$ the final outcome has been revealed.

Below, we analyze large-sample properties of this estimator under the double machine learning framework, and see that it preserves the strong double robustness property discussed in Chapter 3: The estimator has good properties if the product of the mean-squared errors for the $\hat { \gamma } _ { t } ( \pi )$ model and for the $\widehat { V } _ { \pi , t }$ decay fast enough. For simplicity, we assume that that the estimators for these nuisance components are obtained using independent training data; however, as in Chapter 3, the argument generalizes immediately to K-fold cross-fitting at the cost of some extra notation.

Theorem 14.3. Under the conditions of Theorem $\it { 1 4 . 2 } ,$ suppose furthermore that we estimate the nuisance components in (14.19) on independent training $t = 1 , \ldots , T , ^ { 7 4 }$

$$
\mathbb {E} \left[ (\hat {\gamma} _ {i t} (\pi) - \gamma_ {i t} (\pi)) ^ {2} \right] = o _ {P} \left(n ^ {- 2 \alpha_ {\gamma}}\right),
$$

$$
\mathbb {E} \left[ \left(\widehat {V} _ {\pi , t} (S _ {i t}) - V _ {\pi , t} (S _ {i t})\right) ^ {2} \right] = o _ {P} \left(n ^ {- 2 \alpha_ {V}}\right) \tag {14.20}
$$

for constants $\alpha _ { \gamma } , \alpha _ { V } \geq 0$ with $\alpha _ { \gamma } + \alpha _ { V } \ge 1 / 2$ . Then,

$$
\sqrt {n} \left(\widehat {V} _ {A I P W} (\pi) - V (\pi)\right) \Rightarrow \mathcal {N} \left(0, \sigma_ {A I P W} ^ {2}\right)
$$

$$
\sigma_ {A I P W} ^ {2} = \operatorname{Var} \left[ \mathbb {E} _ {\pi} [ Y | X _ {1} ] \right] \tag {14.21}
$$

$$
+ \sum_ {t = 1} ^ {T} \mathbb {E} _ {\pi} \left[ \operatorname{Var} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y \mid S _ {t + 1} \right] \mid S _ {t} \right] \Big / \prod_ {t ^ {\prime} = 1} ^ {t} \mathbb {P} \left[ W _ {t ^ {\prime}} = \pi_ {t ^ {\prime}} (S _ {t ^ {\prime}}) \mid S _ {t ^ {\prime}} \right] \right].
$$

Proof. As in the proof of the single time-step AIPW result in Chapter 3, we first consider properties of an oracle estimator with correct nuisance estimates, and then show asymptotic equivalence of the feasible and oracle AIPW estimators under rate-of-convergence assumptions and with exogenous nuisance estimators. In our setting, the oracle is

$$
\widehat {V} _ {A I P W} ^ {*} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(V _ {\pi , 1} (X _ {i 1}) \right. \tag {14.22}
$$

$$
\left. + \sum_ {t = 1} ^ {T} \gamma_ {i t} (\pi) \left(V _ {\pi , t + 1} (S _ {i (t + 1)}) - V _ {\pi , t} (S _ {i t})\right)\right),
$$

with $V _ { \pi , t } ( S _ { i ( T + 1 ) } ) = Y _ { i }$ . Now, by (14.9) we know that $\begin{array} { r l } { \mathbb { E } _ { \boldsymbol { \pi } } \left[ V _ { \boldsymbol { \pi } , t + 1 } ( S _ { i ( t + 1 ) } ) \big | \ S _ { i t } \right] = } \end{array}$ $V _ { \pi , t } ( S _ { i t } )$ . By sequential unconfoundedness (and in particularly the property highlighted in Proposition 14.1), this implies that under the data-collection measure,

$$
\mathbb {E} \left[ V _ {\pi , t + 1} (S _ {i (t + 1)})   |   S _ {i t},   W _ {i t} = \pi (S _ {i t}) \right] = V _ {\pi , t} (S _ {i t}). \tag {14.23}
$$

Furthermore, recalling that $\gamma _ { i t } ( \pi )$ is a function of $S _ { i t }$ and $W _ { i t }$ , and $\gamma _ { i t } ( \pi ) \neq 0$ only when $W _ { i t } = \pi ( S _ { i t } )$ , we see that

$$
\mathbb {E} \left[ \gamma_ {i t} (\pi) \left(V _ {\pi , t + 1} (S _ {i (t + 1)}) - V _ {\pi , t} (S _ {i t})\right) \big | S _ {i t} \right] = 0, \tag {14.24}
$$

i.e., the terms $\gamma _ { i t } ( \pi ) \left( V _ { \pi , t + 1 } ( S _ { i ( t + 1 ) } ) - V _ { \pi , t } ( S _ { i t } ) \right)$ for a given unit i form a martingale difference sequence. Thus

$$
\begin{array}{l} \mathrm{Var} \left[ V _ {\pi , 1} (X _ {i 1}) + \sum_ {t = 1} ^ {T} \gamma_ {i t} (\pi) \left(V _ {\pi , t + 1} (S _ {i (t + 1)}) - V _ {\pi , t} (S _ {i t})\right) \right] \\ = \operatorname{Var} \left[ V _ {\pi , 1} (X _ {i 1}) \right] + \sum_ {t = 1} ^ {T} \operatorname{Var} \left[ \gamma_ {i t} (\pi) \left(V _ {\pi , t + 1} (S _ {i (t + 1)}) - V _ {\pi , t} (S _ {i t})\right) \right]. \\ \end{array}
$$

One recovers the variance expression in (14.21) by moving to the off-policy measure as in the proof of Theorem 14.2 and then plugging in the definition of the value function from (14.8). Finally, given IID sampling of units $i = 1 , \ldots , n$ our strong overlap and boundedness assumptions, the central limit theorem 14.21 follows immediately for the oracle estimator (14.22).

Now, to show asymptotic equivalence of the feasible and oracle AIPW estimators, we introduce some convenient short-hand. We write the time-t value function updates as

$$
\varepsilon_ {i t} := V _ {\pi , t + 1} (S _ {i (t + 1)}) - V _ {\pi , t} (S _ {i t})
$$

for $t = 0 , \ldots , T$ , and the value function errors as

$$
\hat {\delta} _ {i t} = \widehat {V} _ {\pi , t} (S _ {i t}) - V _ {\pi , t} (S _ {i t})
$$

for $t = 1 , \dots , T$ . We also drop the explicit π dependence in $\gamma _ { i t } ( \pi )$ . Given this notation, we have

$$
\widehat {V} _ {A I P W} ^ {*} (\pi) - V (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \gamma_ {i t} \varepsilon_ {i t}
$$

$$
\widehat {V} _ {A I P W} (\pi) - V (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \hat {\gamma} _ {i t} \left(\varepsilon_ {i t} + \hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right),
$$

where we have $\hat { \delta } _ { i 0 } = 0$ (because $\widehat { V } _ { 0 , \pi }$ doesn’t appear in the construction of $\widehat { V } _ { A I P W } ( \pi )$ so without loss of generality we make no errors there) and $\hat { \delta } _ { i ( T + 1 ) } = 0$ (because $\widehat { V } _ { \pi , T + 1 } ( S _ { i ( T + 1 ) } ) = Y _ { i } )$ . Thus,

$$
\begin{array}{l} \widehat {V} _ {A I P W} (\pi) - \widehat {V} _ {A I P W} ^ {*} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \left(\widehat {\gamma} _ {i t} - \gamma_ {i t}\right) \varepsilon_ {i t} \\ + \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \gamma_ {i t} \left(\hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right) + \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \left(\hat {\gamma} _ {i t} - \gamma_ {i t}\right) \left(\hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right). \\ \end{array}
$$

We now bound each term separately as in the proof of Theorem 3.2. The first term is a martingale in t by the same argument as used above, and so by IID sampling of units

$$
\begin{array}{l} \mathrm{Var} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \left(\hat {\gamma} _ {i t} - \gamma_ {i t}\right) \varepsilon_ {i t} \right] = \frac {1}{n} \sum_ {t = 1} ^ {T} \mathbb {E} \left[ \left(\hat {\gamma} _ {i t} - \gamma_ {i t}\right) ^ {2} \mathrm{Var} _ {\pi} \left[ \varepsilon_ {i t} \mid S _ {i t} \right] \right] \\ = \mathcal {O} \left(\frac {1}{n} \sum_ {t = 1} ^ {T} \mathbb {E} \left[ (\hat {\gamma} _ {i t} - \gamma_ {i t}) ^ {2} \right]\right), \\ \end{array}
$$

$\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { t = 0 } ^ { T } \left( \widehat { \gamma } _ { i t } - \gamma _ { i t } \right) \varepsilon _ { i t } = o _ { p } \left( 1 / \sqrt { n } \right) } \end{array}$ we can rearrange the sum:

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \gamma_ {i t} \left(\hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\sum_ {t = 1} ^ {T} \left(\gamma_ {i (t - 1)} - \gamma_ {i t}\right) \hat {\delta} _ {i t} + \gamma_ {i T} \hat {\delta} _ {i (T + 1)} - \gamma_ {i 0} \hat {\delta} _ {i 0}\right),
$$

where the last two terms can be ignored because $\hat { \delta } _ { i 0 } = \hat { \delta } _ { i ( T + 1 ) } = 0$ . Given the definitions of $\gamma _ { i t }$ and $\hat { \delta } _ { i t }$ , this term can be further simplified as

$$
\ldots = \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 1} ^ {T} \gamma_ {i (t - 1)} \left(1 - \frac {1 \left(\{W _ {i t} = \pi (S _ {i t}) \}\right)}{\mathbb {P} \left[ W _ {i t} = \pi (S _ {i t}) \mid S _ {i t} \right]}\right) \left(\widehat {V} _ {\pi , t} (S _ {i t}) - V _ {\pi , t} (S _ {i t})\right).
$$

By sequential unconfoundedness, the inner sum is again a martingale in $t ,$ so

$$
\begin{array}{l} \mathbb {E} \left[ \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \gamma_ {i t} \left(\hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right)\right) ^ {2} \right] \\ = \frac {1}{n} \sum_ {t = 1} ^ {T} \mathbb {E} \left[ \gamma_ {i (t - 1)} ^ {2} \frac {1 - \mathbb {P} \left[ W _ {i t} = \pi \left(S _ {i t}\right) \mid S _ {i t} \right]}{\mathbb {P} \left[ W _ {i t} = \pi \left(S _ {i t}\right) \mid S _ {i t} \right]} \left(\widehat {V} _ {\pi , t} \left(S _ {i t}\right) - V _ {\pi , t} \left(S _ {i t}\right)\right) ^ {2} \right] \\ = \frac {1}{n} \sum_ {t = 1} ^ {T} \eta^ {1 - 2 t} \mathbb {E} \left[ \left(\widehat {V} _ {\pi , t} (S _ {i t}) - V _ {\pi , t} (S _ {i t})\right) ^ {2} \right] = o _ {p} (1 / n) \\ \end{array}
$$

by (14.20), and the term itself is again $o _ { p } ( 1 / \sqrt { n } )$ . Finally, for the 3rd term, we can swap the order of summation and apply Cauchy-Schwarz:

$$
\begin{array}{l} \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {t = 0} ^ {T} \left(\hat {\gamma} _ {i t} - \gamma_ {i t}\right) \left(\hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right) \\ \leq \sum_ {t = 0} ^ {T} \sqrt {\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\gamma} _ {i t} - \gamma_ {i t}\right) ^ {2}} \sqrt {\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\delta} _ {i (t + 1)} - \hat {\delta} _ {i t}\right) ^ {2}} = o _ {P} \left(n ^ {- (\alpha_ {\gamma} + \alpha_ {V})}\right). \\ \end{array}
$$

This establishes that

$$
\widehat {V} _ {A I P W} (\pi) - \widehat {V} _ {A I P W} ^ {*} (\pi) = o _ {P} \left(1 / \sqrt {n}\right),
$$

thus concluding the proof.

![image_11](images/image_11.png)

## 14.3 Bibliographic notes

The approach evaluating dynamic decision rules presented here, i.e., with nested potential outcomes and with identification obtained via under sequential unconfoundedness, goes back to Robins [1986]; see Richardson and Rotnitzky [2014] for a survey of this line of work, and Hern´an and Robins [2020] for a textbook treatment. One of the most widely used algorithms from this line of work, called marginal structural modeling, involves estimating the value of a parametrized policy class via inverse-propensity weighted linear regression [see Robins, 1999, for an overview]. The AIPW estimator (14.19) is discussed in Jiang and Li [2016], Thomas and Brunskill [2016] and Zhang, Tsiatis, Laber, and Davidian [2013].

Causal inference in dynamic settings is a broad topic, a comprehensive discussion of which would go beyond the scope of this book. Van der Laan and Robins [2003] and Tsiatis [2006] offer comprehensive textbook treatments, including discussions of efficiency. In particular, one consideration that’s important in many applications is the problem of censoring: Some units may leave the study before we get to observe the final outcome, and the methods discussed in this chapter need to be extended to accommodate such censoring (see Exercise 14 in Chapter 16 for one example of a result with censoring). Another interesting direction is the extension of our discussion on policy learning from Chapter 5 to the dynamic setting [Robins, 2004]. Finally, our discussion of dynamic policy evaluation is closely related to the literature on reinforcement learning; see Sutton and Barto [2018] for a textbook treatment.
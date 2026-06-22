# Chapter 15 Markov Decision Processes

In the previous chapter, we considered dynamic treatment rules in a general setting without modeling assumptions on how treatment effects play out over time, and introduced a set of methods for policy evaluation that only required sequential unconfoundedness for identification. The flexibility of these methods, however, comes at a cost of precision. The discussed inverse-propensity weighted method can only leverage trajectories whose assigned treatment matches the policy prescription in all T time periods and involves weights whose magnitude generally scales exponentially in the time horizon T ; and the augmented method faces a similar “curse of horizon”.

Here, we will study how judicious use of modeling assumptions can help mitigate this curse of horizon. The key insight is that, in many applications, any intervention we take is relevant for some amount of time, but its effect eventually washes out. And, if we believe that actions taken long ago are no longer relevant, then one may hope that it’s possible to meaningfully use trajectories for policy evaluation even if they deviated from the target policy at some point in the far past. The following example has this structure.

Example 20. Many ride-sharing platforms implement some kind of surge pricing mechanism, which involves temporarily raising prices in areas experiencing localized demand spikes [Castillo, Knoepfle, and Weyl, 2024]. Activating surge pricing at a given location allows the platform to rapidly shed demand at that location, and also to increase supply by encouraging idle drivers to relocate to the area with surge pricing. This helps the market rebalance itself, and avoids a situation where the platform is unable to fulfill ride requests at posted prices. In order to choose between algorithms and/or calibrate the parameters of a given algorithm, platforms often run experiments that toggle between surge algorithms in a given market.75How should we analyze data from an experiment as described in the above example? This problem clearly involves complex treatment dynamics, and so event-study methods are not applicable. On the other hand, while surge pricing algorithms obviously have intricate short-term effects (e.g., by moving the distribution of drivers in the system), one should expect any such effects wash out (after temporarily suppressed demand has been able to re-emerge and drivers have a chance to return to their usual configuration). This suggests we should be able to develop analytic techniques that can extract meaningful insights from a long-horizon (say, multi-week) surge pricing experiment without suffering the curse-of-horizon phenomenon incurred my methods from the previous chapter.

The question, then, is how to specify a flexible and credible model that enables this type of forgetting. Here, we will do so by assuming Markovian structure. We assume that we observe a single unit over a long trajectory $t = 1 , 2 , . . . , T$ , with a state variable $X _ { t }$ , actions $W _ { t }$ and outcomes $Y _ { t }$ . Our Markovian assumption, formalized below, is that at time $t ,$ any effect of past actions on future observables is mediated by the current state $X _ { t }$ . Such Markovian structure induces forgetting—and enables consistent policy evaluation from a single trajectory—as long as the state variable $X _ { t }$ has relevant “mixing” properites that prevent it from holding information about past treatment assignments for excessively long times.

Definition 15.1. A Markov decision process (MDP) is characterized by a series of state-transition distribution $P _ { t }$ such that, for all $t ,$

$$
X _ {t + 1}, Y _ {t} \sim P _ {t} (X _ {t}, W _ {t}) \tag {15.1}
$$

conditionally on all information available up to time t, i.e., conditionally on $X _ { 1 } , W _ { 1 } , Y _ { 1 } , X _ { 2 } , . . . , X _ { t } , W _ { t }$ .

In the context of the ride-sharing example, one could define $X _ { t }$ as the current number of drivers in each neighborhood, and $W _ { t }$ as whether an experimental surge algorithm is currently active downtown. Then, our Markovian assumption would require positing that the effect of any past surge-pricing decisions is mediated by the current driver distribution, while a mixing assumption will essentially imply that, if we return to our default algorithm for a long enough period of time, drivers return to their usual patterns.

## 15.1 The long-run average value

We start our study of MDPs by revisiting the setting of policy evaluation under sequential randomization, and see how Markovian modeling assumption can enable precision improvements relative to methods from the previous chapter. We work under the long-horizon, $T \to \infty$ seek to estimate the long-run average value produced under a time-homogeneous target policy

$$
V (\pi) = \lim _ {T \to \infty} \mathbb {E} _ {\pi} \left[ \frac {1}{T} \sum_ {t = 1} ^ {T} Y _ {t} \right], \quad \pi : \mathcal {X} \to \{0, 1 \}, \tag {15.2}
$$

under an assumption that this limit exists. We assumption that we have data collected under a sequentially unconfounded design,

$$
W _ {t} \sim e (X _ {t}), \quad e: \mathcal {X} \rightarrow (0, 1), \tag {15.3}
$$

conditionally on all past information, and we will assume that $e ( x )$ is known. We also make the following regularity assumptions on the MDP throughout:

• The MDP is time homogeneous, i.e., the state-transition distributions $P _ { t }$ from Definition 15.1 satisfy $P _ { t } = P$ for all t.
• The state-variables $X _ { t }$ observed in our study, i.e., with treatment generated following (15.3), form an irreducible, aperiodic Markov chain with stationary distribution F . The process is initialized from this stationary distribution, i.e., $X _ { 1 } \sim F$ .
• The $X _ { t }$ observed in our study satisfy the ρ-mixing condition [see Bradley, 2005, for a survey of mixing conditions and their relationships],

$$
\sum_ {t = 1} ^ {\infty} \sup _ {f, g \in L _ {2} (F)} | \operatorname{Corr} (f (X _ {1}), g (X _ {t})) | <   \infty . \tag {15.4}
$$

• The state-variables $X _ { t }$ generated from the MDP under our target policy π converge weakly to a stationary distribution $F _ { \pi }$ , and also satisfies the ρ-mixing condition (15.4).
• The distributions F and $F _ { \pi }$ are equivalent measures.

Notice that, writing $\mu _ { \pi } ( x ) = \mathbb { E } _ { P } \left\lceil Y _ { t } \right\rceil X _ { t } = x , W _ { t } = \pi ( x ) \rceil$ , the second-tolast assumption implies that our target exists and can be expressed as $V ( \pi ) = \mathbb { E } _ { F _ { \pi } } \left[ \mu _ { \pi } ( X ) \right]$ .

Given this setup, we can write down a doubly robust estimator for $V ( \pi )$ in terms of the excess reward function

$$
Q _ {\pi} (x) = \lim _ {T \rightarrow \infty} \mathbb {E} _ {\pi} \left[ \sum_ {t = 1} ^ {T} \left(Y _ {t} - V (\pi)\right) \mid X _ {1} = x \right], \tag {15.5}
$$

which measures the size of the expected (non-scaled) excess reward under π from starting from a specific state x rather than from a random draw from $F _ { \pi }$ and the stationary distribution ratio

$$
\omega_ {\pi} (x) = d F _ {\pi} (x) / d F (x). \tag {15.6}
$$

Given estimates of these two quantities, and assuming that $e ( \cdot )$ is known (as it would be in a sequentially randomized experiment), then the estimator

$$
\widehat {V} _ {D R} (\pi) = \frac {\sum_ {t = 1} ^ {T - 1} \left(Y _ {t} + \widehat {Q} _ {\pi} (X _ {t + 1}) - \widehat {Q} _ {\pi} (X _ {t})\right) \hat {\omega} _ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})}}{\sum_ {t = 1} ^ {T - 1} \hat {\omega} _ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})}} \tag {15.7}
$$

is consistent for $V ( \pi )$ and (strongly) doubly robust in the sense discussed in Chapter 3. Above, we have used the notational short-hand $e _ { \pi } ( x ) = \pi ( x ) e ( x ) +$ $( 1 - \pi ( x ) ) ( 1 - e ( x ) )$ to denote the conditional probability of following $\pi ( \cdot )$ .

The remainder of this section will be devoted to proving this result. For simplicity, we will not rely on cross-fitting, and will instead assume that the estimates $\hat { \omega } _ { \pi } ( \cdot )$ and $\widehat { Q } _ { \pi } ( \cdot )$ have been obtained on a separate training set; however, we do note that given appropriate mixing assumptions a cross-fitting argument across long, consecutive segments of the time series $\left( { { X } _ { t } } , { { Y } _ { t } } , \ { { W } _ { t } } \right)$ would also be possible. Finally, as in the rest of the book, we will defer to the statistical learning literature for methods on estimating the functions $\hat { \omega } _ { \pi } ( \cdot )$ and $\widehat { Q } _ { \pi } ( \cdot )$ ; see Liao et al. [2022] and Uehara, Huang, and Jiang [2020] for recent proposals.

We start establishing two results motivating the form of the estimator (15.7). Note that these two results together already imply weak double robustness of the estimator.

Lemma 15.1. Under our stated assumptions and with VarFπ $\left[ \mu _ { \pi } ( X ) \right] < \infty$ , the excess reward function $Q _ { \pi } ( X _ { t } )$ is absolutely integrable under $F _ { \pi }$ , almost surely finite under $X _ { t } \sim F$ , and satisfies the Bellman conditions

$$
\begin{array}{l} \mathbb {E} _ {\pi} \left[ Y _ {t} + Q _ {\pi} (X _ {t + 1}) \mid X _ {t} \right] - Q _ {\pi} (X _ {t}) = V (\pi), \\ \mathbb {E} _ {0} \left[ \frac {1 \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})} \left(Y _ {t} + Q _ {\pi} (X _ {t + 1})\right) \mid X _ {t} \right] - Q _ {\pi} (X _ {t}) = V (\pi), \tag {15.8} \\ \end{array}
$$

almost surely.

Proof. Given time-homogeneity of our system, an application of the chain rule to (15.5) implies that

$$
\mathbb {E} _ {\pi} \left[ Q _ {\pi} (X _ {t + 1}) \mid X _ {t} = x \right] = \lim _ {T \to \infty} \mathbb {E} _ {\pi} \left[ \sum_ {t = 2} ^ {T} \left(Y _ {t} - V (\pi)\right) \mid X _ {1} = x \right].
$$

The first Bellman equation then follows immediately from basic algebraic manipulations—provided we can show that $Q _ { \pi } ( X _ { t } )$ is almost surely finite under $X _ { t } \sim F$ . In order to verify this, we will show below that

$$
\sum_ {t = 1} ^ {\infty} \mathbb {E} _ {X _ {1} \sim F _ {\pi}} \left[ \left| \mathbb {E} _ {\pi} \left[ Y _ {t} - V (\pi) \mid X _ {1} \right] \right| \right] <   \infty ; \tag {15.9}
$$

it then follows from Fubini’s theorem that $Q _ { \pi } ( X _ { t } )$ is absolutely integrable under $F _ { \pi } , \mathbb { E } _ { X _ { 1 } \sim F _ { \pi } } \left[ \vert Q _ { \pi } ( X _ { 1 } ) \vert \right] < \infty$ . This also implies that $Q _ { \pi } ( X _ { t } )$ is almost surely finite under $X _ { t } \sim F$ since F and $F _ { \pi }$ are equivalent measures. Meanwhile, the second Bellman equation follows from the first by the standard IPW argument under sequential unconfoundedness as used in the proof of Theorem 14.2.

We now turn to verifying (15.9) under our ρ-mixing assumption. Write

$$
\rho_ {\pi} ^ {t} = \sup _ {f, g \in L _ {2} (F _ {\pi})} | \mathrm{Corr} _ {\pi} (f (X _ {1}), g (X _ {t})) |,
$$

and recall that our assumption is that these $\textstyle \sum _ { t = 1 } ^ { \infty } \rho _ { \pi } ^ { t } < \infty$ . Now, by applying Jensen’s inequality

$$
\mathbb {E} _ {\pi} \left[ \left| \mathbb {E} _ {\pi} \left[ Y _ {t} - V (\pi) \mid X _ {1} \right] \right| \right] \leq \mathbb {E} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y _ {t} - V (\pi) \mid X _ {1} \right] ^ {2} \right] ^ {\frac {1}{2}} = \mathrm{Var} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y _ {t} \mid X _ {1} \right] \right] ^ {\frac {1}{2}},
$$

where we have left the fact that $X _ { 1 } \sim F _ { \pi }$ implicit. Furthermore,

$$
\begin{array}{l} \mathrm{Var} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y _ {t} \mid X _ {1} \right] \right] = \mathrm{Cov} _ {\pi} \left[ \mu_ {\pi} (X _ {t}), \mathbb {E} _ {\pi} \left[ Y _ {t} \mid X _ {1} \right] \right] \\ = \operatorname{Corr} _ {\pi} \left(\mu_ {\pi} (X _ {t}), \mathbb {E} _ {\pi} \left[ Y _ {t} \mid X _ {1} \right]\right) \\ \times \operatorname{Var} _ {\pi} \left[ \mu_ {\pi} (X _ {t}) \right] ^ {1 / 2} \operatorname{Var} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y _ {t} \mid X _ {1} \right] \right] ^ {1 / 2}, \\ \end{array}
$$

and so

$$
\mathrm{Var} _ {\pi} \left[ \mathbb {E} _ {\pi} \left[ Y _ {t} \mid X _ {1} \right] \right] ^ {1 / 2} \leq \rho_ {\pi} ^ {t} \mathrm{Var} _ {F _ {\pi}} \left[ \mu_ {\pi} (X) \right] ^ {1 / 2}.
$$

Putting everything together, we get

$$
\sum_ {t = 1} ^ {\infty} \mathbb {E} _ {X _ {1} \sim F _ {\pi}} \left[ \left| \mathbb {E} _ {\pi} \left[ Y _ {t} - V (\pi) \mid X _ {1} \right] \right| \right] \leq \mathrm{Var} _ {F _ {\pi}} \left[ \mu_ {\pi} (X) \right] ^ {1 / 2} \sum_ {t = 1} ^ {\infty} \rho_ {\pi} ^ {t} <   \infty ,
$$

as claimed.

![image_12](images/image_12.png)

Lemma 15.2. Under our stated assumptions, for any time t and any measurable function h(X),

$$
\mathbb {E} _ {0} \left[ \omega_ {\pi} \left(X _ {t}\right) h \left(X _ {t + 1}\right) \frac {1 \left(\left\{W _ {t} = \pi (X _ {t}) \right\}\right)}{e _ {\pi} (X _ {t})} \right] = \mathbb {E} _ {0} \left[ \omega_ {\pi} \left(X _ {t}\right) h \left(X _ {t}\right) \right], \tag {15.10}
$$

provided all stated expectations exist and are finite.

Proof. Starting with the right-hand side expression, we can invoke stationarity as well as a change-of-measure argument to check that

$$
\mathbb {E} _ {0} \left[ \omega_ {\pi} (X _ {t}) h (X _ {t}) \right] = \mathbb {E} _ {F} \left[ \omega_ {\pi} (X) h (X) \right] = \mathbb {E} _ {F _ {\pi}} [ h (X) ].
$$

Meanwhile, for the left-hand-side, we the standard IPW argument under sequential unconfoundedness implies that

$$
\mathbb {E} _ {0} \left[ h \left(X _ {t + 1}\right) \frac {1 \left(\left\{W _ {t} = \pi (X _ {t}) \right\}\right)}{e _ {\pi} (X _ {t})} \mid X _ {t} \right] = \mathbb {E} _ {\pi} \left[ h \left(X _ {t + 1}\right) \mid X _ {t} \right],
$$

and so an application of the chain rule yields

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \omega_ {\pi} (X _ {t}) h (X _ {t + 1}) \frac {1 (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})} \right] \\ = \mathbb {E} _ {0} \left[ \omega_ {\pi} (X _ {t}) \mathbb {E} _ {0} \left[ h (X _ {t + 1}) \frac {1 (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})} | X _ {t} \right] \right] \\ = \mathbb {E} _ {0} \left[ \omega_ {\pi} \left(X _ {t}\right) \mathbb {E} _ {\pi} \left[ h \left(X _ {t + 1}\right) \mid X _ {t} \right] \right] \\ = \mathbb {E} _ {X _ {t} \sim F} \left[ \omega_ {\pi} (X _ {t}) \mathbb {E} _ {\pi} \left[ h (X _ {t + 1}) \mid X _ {t} \right] \right] \\ = \mathbb {E} _ {X _ {t} \sim F _ {\pi}} \left[ \mathbb {E} _ {\pi} \left[ h \left(X _ {t + 1}\right) \mid X _ {t} \right] \right] = \mathbb {E} _ {F _ {\pi}} \left[ h (X) \right], \\ \end{array}
$$

where the 3rd and 5th equalities leveraged stationarity.

Theorem 15.3. Under our stated assumptions, suppose furthermore that we estimate the nuisance components in (15.7) on independent training data such that, for all t = 1, . . . , T ,76

$$
\mathbb {E} _ {F} \left[ \left(\widehat {Q} _ {\pi} (X) - Q _ {\pi} (X)\right) ^ {2} \right] = o _ {P} \left(T ^ {- 2 \alpha_ {Q}}\right), \tag {15.11}
$$

$$
\mathbb {E} _ {F} \left[ \left(\hat {\omega} _ {\pi} (X) - \omega_ {\pi} (X)\right) ^ {2} \right] = o _ {P} \left(T ^ {- 2 \alpha_ {\omega}}\right)
$$

for constants αQ, $\alpha _ { \omega } \geq 0$ with $\alpha _ { \omega } + \alpha _ { Q } \ge 1 / 2$ . $T h e n ,$

$$
\begin{array}{l} \sqrt {T} \left(\widehat {V} _ {D R} (\pi) - V (\pi)\right) \Rightarrow \mathcal {N} (0, \Sigma) \\ \Sigma = \mathbb {E} _ {F} \left[ \frac {\omega_ {\pi} ^ {2} (X _ {1})}{e _ {\pi} (X _ {1})} \mathbb {E} _ {\pi} \left[ (Y _ {1} + Q _ {\pi} (X _ {2}) - Q _ {\pi} (X _ {1}) - V (\pi)) ^ {2} \mid X _ {1} \right] \right], \tag {15.12} \\ \end{array}
$$

provided that Σ is finite.

Proof. Our estimator has a self-normalized form, and so its errors can be expressed as

$$
\widehat {V} _ {D R} (\pi) - V (\pi) = \frac {\sum_ {t = 1} ^ {T - 1} \left(Y _ {t} + \widehat {Q} _ {\pi} (X _ {t + 1}) - \widehat {Q} _ {\pi} (X _ {t}) - V (\pi)\right) \hat {\omega} _ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})}}{\sum_ {t = 1} ^ {T - 1} \hat {\omega} _ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})}}.
$$

We start by considering the denominator. By stationarity,

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})} \right] = \mathbb {E} _ {0} \left[ \omega_ {\pi} (X _ {t}) \mathbb {E} _ {0} \left[ \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})} | X _ {t} \right] \right] \\ = \mathbb {E} _ {0} [ \omega_ {\pi} (X _ {t}) ] = \mathbb {E} _ {F} [ \omega_ {\pi} (X) ] = 1, \\ \end{array}
$$

and so we can apply the ergodic theorem $[ \mathrm { e . g . }$ , Durrett, 2019, Chapter 6.2] to verify that

$$
\frac {1}{T - 1} \sum_ {t = 1} ^ {T - 1} \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})} \rightarrow_ {p} 1. \tag {15.13}
$$

Furthermore, we see that

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \left| \frac {1}{T - 1} \sum_ {t = 1} ^ {T - 1} \left(\hat {\omega} _ {\pi} (X _ {t}) - \omega_ {\pi} (X _ {t})\right) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})} \right| \right] \\ \leq \frac {1}{\eta^ {2}} \sqrt {\mathbb {E} _ {0} \left[ \frac {1}{T - 1} \sum_ {t = 1} ^ {T - 1} \left(\hat {\omega} _ {\pi} (X _ {t}) - \omega_ {\pi} (X _ {t})\right) ^ {2} \right]} \\ = \frac {1}{\eta^ {2}} \sqrt {\mathbb {E} _ {F} \left[ (\hat {\omega} _ {\pi} (X) - \omega_ {\pi} (X)) ^ {2} \right]} = o _ {p} (1) \\ \end{array}
$$

by respectively invoking Cauchy-Schwarz, overlap, stationarity, and $L _ { 2 ^ { - } }$ consistency of $\hat { \omega } ( \cdot )$ , thus implying that (15.13) also holds for $\omega ( \cdot )$ replaced with $\hat { \omega } ( \cdot )$ .

<!-- footnote -->

- Neyman [1923] worked under complete randomization, i.e., where the number of treated units is fixed a-priori; however, all the key insights are the same.

<!-- footnote end -->

<!-- footnote -->

- $^ { 6 6 } \mathrm { I n }$ the variance estimate $\widehat { V } _ { D M }$ in (1.10) we used a normalizations $n _ { 0 } / n$ and $n _ { 1 } / n$ which in (12.11) are replaced with $1 - \pi$ and π respectively; however, this distinction is immaterial under 1st-order analysis. The variance estimates are asymptotically equivalent, and either of them can be used for confidence intervals when in the uniformly randomized setting with $e _ { i } = \pi$ for all units.

<!-- footnote end -->

<!-- footnote -->

- The HAC construction is only used to motivate the functional form of the variance estimator below; its consistency in our setting will be established from first principles below. See White [1984, Chapter VI.4] for a general discussion of HAC estimators for correlated random variables, and Kojevnikov, Marmer, and Song [2021] for recent results on HAC estimators in a model with network correlation.
- As a sanity check one can verify that, under SUTVA $( \mathrm { i . e . }$ , with $\boldsymbol { G } = \boldsymbol { I } _ { n \times n } )$ , (12.18) exactly matches (12.8).

<!-- footnote end -->

<!-- footnote -->

- This phenomenon is conceptually related to what we observed in Theorem 2.1, where the asymptotic variance of the stratified estimator of the ATE did not get worse as we increased the number of strata.

<!-- footnote end -->

<!-- footnote -->

- Note that, here, we are only enforcing unconfoundedness for potential outcomes consistent with the trajectory we are already on, i.e., with $w _ { i ( 1 : ( t - 1 ) ) } = W _ { i ( 1 : ( t - 1 ) ) }$ . The other potential outcomes can no longer be reached, and so their distribution no longer matters for policy evaluation given that $w _ { i ( 1 : ( t - 1 ) ) } = W _ { i ( 1 : ( t - 1 ) ) }$ .

<!-- footnote end -->

<!-- footnote -->

- There is a close conceptual connection between these principal strata and the compliance types for IV analyses discussed in Chapter 10.1.
- Given this notation, the policy value itself can also be written as $V _ { \pi , 0 } = V ( \pi )$ .

<!-- footnote end -->

<!-- footnote -->

- Unlike in the rest of the book, we here use $\sigma ^ { 2 }$ instead of $V ^ { * }$ for the asymptotic variance as we follow the standard convention in the reinforcement learning literature of writing the value function as V .

<!-- footnote end -->

<!-- footnote -->

- The expectations below are taken over the test data; and the requirement is the training produces on separate data achieve, with high probability, estimates with good test-set meansquared error.

<!-- footnote end -->

<!-- footnote -->

- When a platform runs a number of independent markets, they can also run experiments by randomly assigning treatment across markets. However, the effective sample size (i.e.,

<!-- footnote end -->

<!-- footnote -->

- the number of treatment randomizations) with this strategy is the number of markets, and so this approach is usually only attractive when it’s possible to experiment across a large number of markets.

<!-- footnote end -->

<!-- footnote -->

- The expectations below are taken over the test data; and the requirement is the training produces on separate data achieve, with high probability, estimates with good test-set meansquared error.

<!-- footnote end -->

Meanwhile, the numerator can be decomposed as $A + B + C + D$ with

$$
A = \sum_ {t = 1} ^ {T - 1} \left(Y _ {t} + Q _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t}) - V (\pi)\right) \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})},
$$

$$
B = \sum_ {t = 1} ^ {T - 1} \left(Y _ {t} + Q _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t}) - V (\pi)\right) \left(\hat {\omega} _ {\pi} (X _ {t}) - \omega_ {\pi} (X _ {t})\right) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})},
$$

$$
C = \sum_ {t = 1} ^ {T - 1} \left(\widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) - \left(\widehat {Q} _ {\pi} (X _ {t}) - Q _ {\pi} (X _ {t})\right)\right) \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})},
$$

$$
D = \sum_ {t = 1} ^ {T - 1} \left(\widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) - \left(\widehat {Q} _ {\pi} (X _ {t}) - Q _ {\pi} (X _ {t})\right)\right)
$$

$$
\times (\hat {\omega} _ {\pi} (X _ {t}) - \omega_ {\pi} (X _ {t})) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})}.
$$

We will show below that

$$
A / \sqrt {T} \Rightarrow \mathcal {N} (0, \Sigma), \quad | B |, | C |, | D | = o _ {P} (\sqrt {T}). \tag {15.14}
$$

Thus, given what was shown about the denominator above, we can establish (15.12) via Slutsky’s lemma.

Now, starting with the (dominant) term A, we note that the second Bellman equation in Lemma 15.1 immediately implies that

$$
\mathbb {E} _ {0} \left[ (Y _ {t} + Q _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t}) - V (\pi)) \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})} | X _ {t} \right] = 0
$$

almost surely for all t, and so the term A is mean zero. Furthermore, by our assumed Markov property, the summands forming A are a martingale difference sequence, because conditioning on $X _ { t }$ is equivalent to conditioning on the full past. Given this set up, we can study large-sample behavior of A via the martingale central limit theorem. A key ingredient in doing so is to study the conditional variance of the individual martingale difference terms. We can again apply the ergodic theorem to verify that

$$
\frac {1}{T - 1} \sum_ {t = 1} ^ {T - 1} \operatorname{Var} _ {0} \left[ \Delta_ {t, t + 1} \mid X _ {t} \right]\rightarrow_ {p} \mathbb {E} _ {X _ {1} \sim F} \left[ \operatorname{Var} _ {0} \left[ \Delta_ {1, 2} \mid X _ {1} \right]\right],
$$

$$
\Delta_ {t, t + 1} = (Y _ {t} + Q _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t}) - V (\pi)) \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})},
$$

provided the right-hand side limit is finite. Furthermore,

$$
\begin{array}{l} \mathbb {E} _ {F} \left[ \operatorname{Var} _ {0} \left[ \Delta_ {1, 2} \mid X _ {1} \right] \right] = \mathbb {E} _ {F} \left[ \mathbb {E} _ {0} \left[ \Delta_ {1, 2} ^ {2} \mid X _ {1} \right] \right] \\ = \mathbb {E} _ {F} \left[ \mathbb {E} _ {0} \left[ 1 \left(\left\{W _ {1} = \pi (X _ {1}) \right\}\right) \Delta_ {1, 2} ^ {2} \mid X _ {1} \right] \right] \\ = \mathbb {E} _ {F} \left[ e _ {\pi} (X _ {1}) \mathbb {E} _ {\pi} \left[ \Delta_ {1, 2} ^ {2} \mid X _ {1} \right] \right] = \Sigma , \\ \end{array}
$$

where the 2nd equality is true because $\Delta _ { 1 , 2 } ^ { 2 } = 0$ whenever $W _ { 1 } \neq \pi ( X _ { 1 } )$ , the 3rd equality is true by sequential unconfoundedness, and the 4th follows by direct algebraic manipulation. Now, we have assumed that $\Sigma \ < \ \infty$ in the theorem statement; thus the ergodic theorem in fact applies. The fact that√ $A / \sqrt { T } \Rightarrow \mathcal { N } ( 0 , \Sigma )$ then follows from the martingale central limit theorem [e.g., Durrett, 2019, Theorem 8.2.8].

Next, moving to the lower-order terms, Lemma 15.1 implies that

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \left(Y _ {t} + Q _ {\pi} \left(X _ {t + 1}\right) - Q _ {\pi} \left(X _ {t}\right) - V (\pi)\right) \right. \\ \times \left. (\hat {\omega} _ {\pi} (X _ {t}) - \omega_ {\pi} (X _ {t})) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})} \big | X _ {t} \right] = 0, \\ \end{array}
$$

and so the term B is mean-zero. Furthermore, it is again a martingale, and so its variance is equal to the sum of the expected variance of each martingale difference term; thus, by stationarity,

$$
\begin{array}{l} \operatorname{Var} [ B ] = (T - 1) \mathbb {E} _ {F} \left[ \operatorname{Var} _ {0} \left[ \left(Y _ {1} + Q _ {\pi} \left(X _ {2}\right) - Q _ {\pi} \left(X _ {1}\right) - V (\pi)\right) \right. \right. \\ \left. \times \left(\hat {\omega} _ {\pi} (X _ {1}) - \omega_ {\pi} (X _ {1})\right) \frac {\mathbf {1} \left(\{W _ {1} = \pi (X _ {1}) \}\right)}{e _ {\pi} (X _ {1})} \mid X _ {1} \right] \\ = (T - 1) \mathbb {E} _ {F} \left[ \frac {\left(\hat {\omega} _ {\pi} \left(X _ {1}\right) - \omega_ {\pi} \left(X _ {1}\right)\right) ^ {2}}{e _ {\pi} \left(X _ {1}\right)} \operatorname{Var} _ {\pi} \left[ Y _ {1} + Q _ {\pi} \left(X _ {2}\right) \mid X _ {1} \right] \right] \\ = \mathcal {O} \left((T - 1) \mathbb {E} _ {F} \left[ (\hat {\omega} _ {\pi} (X _ {1}) - \omega_ {\pi} (X _ {1})) ^ {2} \right]\right) = o _ {p} (T), \\ \end{array}
$$

and so $B = o _ { p } ( { \sqrt { T } } )$

Meanwhile, we can verify that the term C is mean zero using Lemma 15.2:

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \left(\widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) - \left(\widehat {Q} _ {\pi} (X _ {t}) - Q _ {\pi} (X _ {t})\right)\right) \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} \left(\{W _ {t} = \pi (X _ {t}) \}\right)}{e _ {\pi} (X _ {t})} \right] \\ = \mathbb {E} _ {0} \left[ \left(\widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1})\right) \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})} \right] \\ - \mathbb {E} _ {0} \left[ \left(\widehat {Q} _ {\pi} (X _ {t}) - Q _ {\pi} (X _ {t})\right) \omega_ {\pi} (X _ {t}) \right] = 0. \\ \end{array}
$$

To calculate the variance of C, it is helpful to split it into two parts:

$$
\begin{array}{l} C _ {1} = \sum_ {t = 1} ^ {T - 1} \left(\mathbb {E} \left[ \widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) \mid X _ {t}, W _ {t} \right] - \left(\widehat {Q} _ {\pi} (X _ {t}) - Q _ {\pi} (X _ {t})\right)\right) \\ \times \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})}, \\ C _ {2} = \sum_ {t = 1} ^ {T - 1} \left(\widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) - \mathbb {E} \left[ \widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) \mid X _ {t}, W _ {t} \right]\right) \\ \times \omega_ {\pi} (X _ {t}) \frac {\mathbf {1} (\{W _ {t} = \pi (X _ {t}) \})}{e _ {\pi} (X _ {t})}. \\ \end{array}
$$

The latter term, $C _ { 2 }$ is a martingale and so can be can be shown to be $o _ { p } ( \sqrt { T } )$ by a similar argument as used with $B .$ . The term $C _ { 1 }$ , however, is not a martingale, and so cross-terms matter. By stationarity,

$$
\begin{array}{l} \operatorname{Var} \left[ C _ {1} \right] = (T - 1) \operatorname{Var} _ {F} \left[ \omega_ {\pi} \left(X _ {1}\right) \frac {\mathbf {1} \left(\left\{W _ {1} = \pi \left(X _ {1}\right) \right\}\right)}{e _ {\pi} \left(X _ {1}\right)} \right. \\ \times \left(\mathbb {E} \left[ \widehat {Q} _ {\pi} (X _ {2}) - Q _ {\pi} (X _ {2}) \mid X _ {1}, W _ {1} \right] - \left(\widehat {Q} _ {\pi} (X _ {1}) - Q _ {\pi} (X _ {1})\right)\right) \\ + (T - 2) \operatorname{Cov} _ {F} \left[ \omega_ {\pi} \left(X _ {1}\right) \frac {\mathbf {1} \left(\left\{W _ {1} = \pi \left(X _ {1}\right) \right\}\right)}{e _ {\pi} \left(X _ {1}\right)} \right. \\ \times \left(\mathbb {E} \left[ \widehat {Q} _ {\pi} (X _ {2}) - Q _ {\pi} (X _ {2})   |   X _ {1},   W _ {1} \right] - \left(\widehat {Q} _ {\pi} (X _ {1}) - Q _ {\pi} (X _ {1})\right)\right), \\ \omega_ {\pi} (X _ {2}) \frac {\mathbf {1} (\{W _ {2} = \pi (X _ {2}) \})}{e _ {\pi} (X _ {2})} \\ \times \left(\mathbb {E} \left[ \widehat {Q} _ {\pi} (X _ {3}) - Q _ {\pi} (X _ {3}) \mid X _ {2}, W _ {2} \right] - \left(\widehat {Q} _ {\pi} (X _ {2}) - Q _ {\pi} (X _ {2})\right)\right) \\ + (T - 3) \dots \\ \end{array}
$$

Then, given our $\rho -$ mixing assumption, we can upper-bound this term as

$$
\mathrm{Var} \left[ C _ {1} \right] \leq (T - 1) \sum_ {t = 1} ^ {\infty} \rho_ {t} \mathrm{Var} _ {F} \bigg [ \omega_ {\pi} (X _ {1}) \frac {\mathbf {1} \left(\{W _ {1} = \pi (X _ {1}) \}\right)}{e _ {\pi} (X _ {1})}
$$

$$
\times \left(\mathbb {E} \left[ \widehat {Q} _ {\pi} (X _ {2}) - Q _ {\pi} (X _ {2})   |   X _ {1},   W _ {1} \right] - \left(\widehat {Q} _ {\pi} (X _ {1}) - Q _ {\pi} (X _ {1})\right)\right),
$$

recalling that we’ve assumed $\textstyle \sum _ { t = 1 } ^ { \infty } \rho _ { t } < \infty$ . Given our L2-consistency assumption on $\widehat { Q }$ and boundedness assumptions on $\omega ( X _ { t } )$ and $1 / e _ { \pi } ( X _ { t } )$ , this implies that $C _ { 1 } = o _ { p } ( \sqrt { T } )$ .

Finally, as already done in many proofs term D can be bounded via Cauchy-Schwarz,

$$
\begin{array}{l} | D | \leq \frac {1}{\eta} \sqrt {\sum_ {t = 1} ^ {T - 1} \left(\widehat {Q} _ {\pi} (X _ {t + 1}) - Q _ {\pi} (X _ {t + 1}) - \left(\widehat {Q} _ {\pi} (X _ {t}) - Q _ {\pi} (X _ {t})\right)\right) ^ {2}} \\ \times \sqrt {\sum_ {t = 1} ^ {T - 1} (\hat {\omega} _ {\pi} (X _ {t}) - \omega_ {\pi} (X _ {t})) ^ {2}} \\ = \mathcal {O} _ {P} \left((T - 1) \mathbb {E} _ {F} \left[ \left(\widehat {Q} _ {\pi} (X) - Q _ {\pi} (X)\right) ^ {2} \right] ^ {\frac {1}{2}} \mathbb {E} _ {F} \left[ \left(\hat {\omega} _ {\pi} (X) - \omega_ {\pi} (X)\right) ^ {2} \right] ^ {\frac {1}{2}}\right) \\ = o _ {p} (\sqrt {T}), \\ \end{array}
$$

where the second line follows by stationarity along with Markov’s inequality and the last line follows by (15.11). □

## 15.2 Switchback experiments

We showed above how—at the expense of some mathematical complexity—it is possible to estimate policy values in Markov decision processes using data collected under a generic sequentially randomized design. In practice, however, it may be easier to change the data-collection procedure to more directly accommodate the problem structure, thus enabling more straight-forward analyses.

One such design is the switchback experiment. In principle, any experiment that measures treatment effects by repeatedly toggling treatment on-and-off at the system level can be referred to a switchback. In systems with temporal carryovers, however, switchbacks are typically understood to be experiments that set treatment to a given level, wait for the system to re-equilibriate, and only then toggle it again. When running switchback experiments, the goal is typically to estimate the total treatment effect,

$$
\tau_ {T O T} = V (1) - V (0) \tag {15.15}
$$

i.e., the long-run average difference between the always-treat and never-treat policies.

There are a variety of switchback designs considered in practice. The simplest (and most widely used) switchback design has a fixed treatment window of length L, and toggles treatment after every L time periods [Bojinov, Simchi-Levi, and Zhao, 2023]. Here, we will consider an alternative “memoryless”switchback design, as it allows for a particularly simple analysis in the context of the Markovian model used in this chapter. See Hu and Wager [2022] for a discussion of standard (i.e., fixed-length) switchbacks under the Markovian model, as well as results in a time-varying setting $( \mathrm { i . e . }$ , with the $P _ { t }$ in Definition 15.1 changing over time).

Definition 15.2. A memoryless switchback with switch rate $0 < \lambda < 1$ is a design that sequentially assigns treatment $W _ { t } \in \{ 0 , 1 \}$ for $t = 1 , 2 , . . .$ . such that $W _ { 1 } \sim \mathrm { B e r n o u l l i } ( 0 . 5 )$ and, for $t \geq 1$ ,

$$
W _ {t + 1} \sim \text { Bernoulli } \left((1 - \lambda) W _ {t} + \lambda (1 - W _ {t})\right). \tag {15.16}
$$

The core fact about switchback experiments is that, if the typical amount of time between treatment switches is long enough $( \mathrm { i . e . }$ , in the case of memoryless switchbacks, if the switch rate λ is low enough), then the raw difference in means estimator

$$
\hat {\tau} _ {S B} = \frac {1}{| W _ {t} = 1 |} \sum_ {\{t: W _ {t} = 1 \}} Y _ {t} - \frac {1}{| W _ {t} = 0 |} \sum_ {\{t: W _ {t} = 0 \}} Y _ {t} \tag {15.17}
$$

is consistent for the total effect. In practice, the behavior of this estimator can be improved by removing burn-in samples right after a switch and other algorithmic modifications [Bojinov, Simchi-Levi, and Zhao, 2023, Hu and Wager, 2022]; here, however, we will focus on the basic estimator (15.17).

To study switchback estimators, we will work in the “tabular” setting where the covariates $X _ { t } \in \mathcal { X }$ take values in a discrete space with $| { \mathcal { X } } | = k$ , meaning that we can write the full treatment-dependent state-transition matrices as $P ^ { w } \in \mathbb { R } ^ { k \times k }$ where $P _ { x x ^ { \prime } } ^ { w } = \mathbb { P } \left[ X _ { t + 1 } = x \middle | X _ { t } = x ^ { \prime } , W _ { t } = w \right]$ . Our analysis also applies directly to non-tabular settings; however, the discrete setting considerably simplifies notation.

We will further assume geometric mixing whereby the state-transition operator is a contraction:

$$
\left\| P ^ {w} \left(\nu^ {\prime} - \nu\right) \right\| _ {1} \leq e ^ {- 1 / t _ {0}} \left\| \nu^ {\prime} - \nu \right\| _ {1} \tag {15.18}
$$

for any measures $\nu , \nu ^ { \prime }$ over X , i.e., for vectors over $[ 0 , 1 ] ^ { k }$ with $\textstyle \sum _ { x } \nu _ { x } = 1$ and likewise for $\nu ^ { \prime } ;$ this condition immediately implies existence of a unique stationary distribution and geometric convergence to the stationary distribution with a mixing time $t _ { 0 }$ .

Theorem 15.4. Consider a time-homogenous Markov decision process satisfying (15.18), and suppose furthermore that $| Y _ { t } | \le M$ almost surely. Then, writing $\tau _ { S B } ( \lambda )$ for the long-run average of $\hat { \tau } _ { S B }$ under a Markovian switchback with switcha rate λ, we have

$$
\left| \tau_ {S B} (\lambda) - \tau_ {T O T} \right| \leq 4 M \lambda \left(1 + t _ {0}\right). \tag {15.19}
$$

Furthermore, if we run a sequence of memoryless switchbacks with horizon T and switch rate $\lambda _ { T } ,$ then $\hat { \tau } _ { S B }  _ { p } \tau _ { T O T }$ whenever $\lambda _ { T } \to 0$ and $T \lambda _ { T } \to \infty$ .

Proof. First, as a preliminary, we note that the mixing condition (15.18) implies that there are stationary distributions $\nu ^ { 0 }$ and $\nu ^ { 1 }$ that can be characterized as the unique solutions to $P ^ { w } \nu ^ { w } = \nu ^ { w }$ over the k-dimensional simplex; and that the long-run average value of the always- and never-treat policies are $\begin{array} { r } { V ( w ) = \sum _ { x } \nu _ { x } ^ { w } \mathbb { E } \lceil Y _ { t } \rceil X _ { t } = x , W _ { t } = w ] } \end{array}$ .

Now, moving to the switchback: Our assumptions that $( X _ { t } , Y _ { t } )$ are from a Markov decision process while $W _ { t }$ is randomized in a memoryless way as given in (15.16) imply that $\left( { { X } _ { t } } , { { Y } _ { t } } , \ { { W } _ { t } } \right)$ together form a Markov chain. Writing $\nu ^ { w } ( \lambda )$ for the distribution of $X _ { t }$ conditionally on $W _ { t } = w$ under stationarity, the fixedpoint condition underlying the stationary joint distribution of $( X _ { t } , W _ { t } )$ is

$$
\binom{\nu^ {0} (\lambda)}{\nu^ {1} (\lambda)} = \left( \begin{array}{c c} (1 - \lambda) P ^ {0} & \lambda P ^ {1} \\ \lambda P ^ {0} & (1 - \lambda) P ^ {1} \end{array} \right) \binom{\nu^ {0} (\lambda)}{\nu^ {1} (\lambda)}. \tag {15.20}
$$

Furthermore, the long-run average expectation of the difference-in-means estimator is

$$
\begin{array}{l} \tau_ {S B} (\lambda) = \sum_ {x \in \mathcal {X}} \nu_ {x} ^ {1} (\lambda) \mathbb {E} \left[ Y _ {t} \mid X _ {t} = x, W _ {t} = 1 \right] \tag {15.21} \\ - \sum_ {x \in \mathcal {X}} \nu_ {x} ^ {0} (\lambda) \mathbb {E} \left[ Y _ {t} \mid X _ {t} = x, W _ {t} = 0 \right], \\ \end{array}
$$

and so by boundedness we immediately see that

$$
\left| \tau_ {S B} (\lambda) - \tau_ {T O T} \right| \leq M \left(\left\| \nu^ {0} (\lambda) - \nu^ {0} \right\| _ {1} + \left\| \nu^ {1} (\lambda) - \nu^ {1} \right\| _ {1}\right). \tag {15.22}
$$

It remains to bound the right-hand side of the above expression, and we use mixing for this.

Focusing on the case $w = 0$ , the top half of (15.20) can be re-written as

$$
\left(I - P ^ {0}\right) \nu^ {0} (\lambda) = \lambda \left(P ^ {1} \nu^ {1} (\lambda) - P ^ {0} \nu^ {0} (\lambda)\right),
$$

and because $\nu ^ { 0 }$ is a fixed point of $P ^ { 0 }$ we thus also have

$$
\left(I - P ^ {0}\right) \left(\nu^ {0} (\lambda) - \nu^ {0}\right) = \lambda \left(P ^ {1} \nu^ {1} (\lambda) - P ^ {0} \nu^ {0} (\lambda)\right).
$$

Combining this expression with (15.18), we get

$$
\begin{array}{l} \left\| \nu^ {0} (\lambda) - \nu^ {0} - \lambda \left(P ^ {1} \nu^ {1} (\lambda) - P ^ {0} \nu^ {0} (\lambda)\right) \right\| _ {1} = \left\| P ^ {0} \left(\nu^ {0} (\lambda) - \nu^ {0}\right) \right\| _ {1} \\ \leq e ^ {- 1 / t _ {0}} \left\| \nu^ {0} (\lambda) - \nu^ {0} \right\| _ {1}, \\ \end{array}
$$

and so by the triangle inequality

$$
\left(1 - e ^ {- 1 / t _ {0}}\right) \left\| \nu^ {0} (\lambda) - \nu^ {0} \right\| _ {1} \leq \lambda \left\| P ^ {1} \nu^ {1} (\lambda) - P ^ {0} \nu^ {0} (\lambda) \right\| _ {1}.
$$

The statement (15.19) follows by noting that $\left( 1 - e ^ { - 1 / t _ { 0 } } \right) ^ { - 1 } \leq 1 + t _ { 0 }$ and $\| P ^ { 1 } \nu ^ { 1 } ( \lambda ) - P ^ { 0 } \nu ^ { 0 } ( \lambda ) \| _ { 1 } \ \le \ 2$ . Finally, the consistency claim follows because $\lambda _ { T } \to 0$ implies that bias goes to 0 by the above, while the condition $\lambda _ { T } T  \infty$ implies that there are a diverging number of switches, and so $\hat { \tau } _ { S B } - \tau ( \lambda _ { T } ) \to _ { p } 0$ thanks to mixing as in (15.18). □

## 15.3 Bibliographic notes

Markov decision processes have been an object of sustained study in the reinforcement learning literature for decades. Our discussion in this chapter fits within the area often referred to as off-policy learning in that literature, as we seek to use data collected under one (randomized) design to predict rewards under a different (target) policy. The off-policy setting is contrasted with the on-policy setting, where we have access to a simulator that can be used to explore states on demand [Sutton and Barto, 2018]. Some notable off-policy algorithms developed in this literature include the temporal-difference learning algorithm which seeks to estimate the discounted value function

$$
V _ {\pi , \gamma} (x) = \mathbb {E} _ {\pi} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} Y _ {t} \mid X _ {0} = x \right], \quad 0 <   \gamma <   1, \tag {15.23}
$$

of a target policy by focusing Bellman equations like those given in Lemma 15.1 [Sutton, 1988, Tsitsiklis and Van Roy, 1997],77 and the Q-learning algorithm for finding the welfare-maximizing policy [Watkins and Dayan, 1992, Murphy, 2005].

The approach taken in this chapter builds on a line of work by Kallus and Uehara [2020] who emphasized the role of Markovian assumptions in mitigating the curse of dimensionality that affects the generic methods for dynamic policy evaluation discussed in the previous chapter, and Liao, Klasnja, and Murphy [2021] who showed how Markov decision processes enable identification of the long-run average value from sequentially unconfounded data. The approach to doubly robust estimation of the long-run average value presented here is adapted from Liao et al. [2022]; a similar approach to estimating discounted policy values (rather than long-run average values) is discussed in Kallus and Uehara [2022]. Setting where the density ratio $\omega _ { \pi } ( X )$ may be heavy tailed and Σ as given in Theorem 15.3 is infinite is considered by Mehrabi and Wager√ [2024]; the authors show that $1 / { \sqrt { T } } .$ -consistent estimation is no longer possible in this setting, but a properly truncated version of the doubly robust estimator from Theorem 15.3 can still achieve the minimax rate of convergence.

Switchback experiments are increasingly becoming a core part of the standard toolkit for causal inference in dynamic systems; Bojinov, Simchi-Levi, and Zhao [2023] provides a comprehensive overview of the design. The analysis presented here, i.e., with switchbacks used for policy evaluation in Markov decision processes, is adapted from Hu and Wager [2022]. One important practical distinction between the doubly robust estimators from Section 15.1 and switchback experiments is that the former require observing (and use of) the state variables $X _ { t } .$ , whereas switchbacks do not. One can ask what happens to optimal inference in the setting of Section 15.1 if we no longer get to observe $X _ { t }$ and instead need to just rely on mixing (15.18) as we did for switchbacks. This setting is considered in Hu and Wager [2023], who show that $1 / { \sqrt { T } } .$ -consistent estimation is in general not possible in this setting, and that switchback-like√ truncated IPW estimators achieve the minimax (slower-than-1 $/ \sqrt { T } )$ rate.
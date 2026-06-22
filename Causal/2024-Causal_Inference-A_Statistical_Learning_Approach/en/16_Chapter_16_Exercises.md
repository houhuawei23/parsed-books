# Chapter 16 Exercises

Exercise 1. Consider a randomized controlled trial under the assumptions of Theorem 1.2. We already know that the difference-in-means estimator,

$$
\hat {\tau} _ {D M} = \frac {1}{| \{i : W _ {i} = 1 \} |} \sum_ {\{i: W _ {i} = 1 \}} Y _ {i} - \frac {1}{| \{i : W _ {i} = 0 \} |} \sum_ {\{i: W _ {i} = 0 \}} Y _ {i}, \tag {16.1}
$$

is consistent and satisfies a central limit theorem in this setting. However, following our discussion in Chapter 2, one might also consider the inversepropensity weighted estimator for τ ,

$$
\hat {\tau} _ {I P W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {W _ {i} Y _ {i}}{\pi} - \frac {(1 - W _ {i}) Y _ {i}}{1 - \pi}. \tag {16.2}
$$

The purpose of this question is to understand the relationship and relative benefits of these two estimators.

(a) State and prove a central limit theorem for ${ \hat { \tau } } _ { I P W }$ (you may make any regularity assumptions that are convenient for this purpose). Compare the variance of ${ \hat { \tau } } _ { I P W }$ to the asymptotic variance of ${ \hat { \tau } } _ { D M }$ given in Theorem 1.2.  
(b) What is the joint distribution of ${ \hat { \tau } } _ { D M }$ and ${ \hat { \tau } } _ { I P W } ?$ Based on your findings, would you recommend using ${ \hat { \tau } } _ { I P W }$ in a randomized study?

Exercise 2. Chapter 1 discussed the behavior of linear regression adjustments in randomized trials, and showed that such adjustments can be used to improve asymptotic precision whether or not the data follows a linear specification. The goal of this question is to extend these results to the case of generic nonparametric (or machine learning based) regression adjustments. For all parts below, you should work under the assumptions of Theorem 1.3.

(a) As shown in (1.27), the interacted regression estimator can be written as an average difference in predictions. Suppose now that we set

$$
\hat {\tau} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\mu} _ {(1)} (X _ {i}) - \hat {\mu} _ {(0)} (X _ {i})\right), \tag {16.3}
$$

but rather than using linear regression, we get $\hat { \mu } _ { ( w ) } ( x )$ from a machine learning method that is consistent (under squared-error loss) for $\mu _ { ( w ) } ( x )$ as defined in (1.21). Are the following two statements true or false? If true, give a proof; if false, give a counterexample.

• The estimator $\hat { \tau }$ is consistent.
• The estimator $\hat { \tau }$ is asymptotically normal, i.e., ${ \sqrt { n } } ( \hat { \tau } - \tau ) \Rightarrow { \mathcal { N } } ( 0 , V )$ for some finite asymptotic variance $V$ .

We now consider an improvement to the basic estimator that debiases (16.3) by considering regression residuals, and uses “cross-fitting” to avoid overfitting. We first split the data (at random) into two halves $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ , and then use

$$
\begin{array}{l} \hat {\tau} _ {C F} = \frac {\hat {\tau} ^ {\mathcal {I} _ {1}} + \hat {\tau} ^ {\mathcal {I} _ {2}}}{2}, \quad \hat {\tau} ^ {\mathcal {I} _ {1}} = \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \hat {\mu} _ {(0)} ^ {\mathcal {I} _ {2}} (X _ {i}) \right. \tag {16.4} \\ \left. + \frac {W _ {i}}{\pi} \left(Y _ {i} - \hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i})\right) - \frac {1 - W _ {i}}{1 - \pi} \left(Y _ {i} - \hat {\mu} _ {(0)} ^ {\mathcal {I} _ {2}} (X _ {i})\right)\right), \\ \end{array}
$$

$\hat { \mu } _ { ( w ) } ^ { \mathcal { Z } _ { 2 } } ( \cdot )$ $\mu _ { ( w ) } ( \cdot )$ sample $\mathcal { T } _ { 2 }$ , and $\hat { \tau } ^ { \mathcal { I } _ { 2 } }$ is defined analogously (with the roles of $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ swapped). In other words, $\hat { \tau } ^ { \mathcal { I } _ { 1 } }$ is a treatment effect estimator on $\mathcal { T } _ { 1 }$ that uses $\mathcal { T } _ { 2 }$ to estimate its regression adjustments, and vice-versa.

(b) What is the bias of the estimator (16.4), i.e., what is $\mathbb { E } \left[ \hat { \tau } _ { C F } \right] - \tau$ , where $\tau$ denotes the ATE?  
(c) Assume that our non-parametric regression adjustments $\hat { \mu } _ { ( w ) } ^ { \perp _ { 2 } } ( \cdot )$ are riskconsistent, i.e.,

$$
\lim _ {n \to \infty} \mathbb {E} \left[ \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(w)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(w)} (X _ {i})\right) ^ {2} \right] = 0, \tag {16.5}
$$

and similarly with $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ swapped. Prove a central limit theorem for $\hat { \tau } _ { C F } .$ , i.e., show that $\sqrt { n } ( \hat { \tau } _ { C F } - \tau ) \Rightarrow \mathcal { N } \left( 0 , V _ { C F } \right)$ for some asymptotic variance $V _ { C F }$ , and characterize $V _ { C F }$ . Compare $V _ { C F }$ to the asymptotic variance VIREG given in (1.23).

(d) Consider the setting discussed in Chapter 1 where a linear model is wellspecified,

$$
Y _ {i} (w) = X _ {i} \beta_ {(w)} + \varepsilon_ {i} (w), \varepsilon_ {i} (w) \sim \mathcal {N} \left(0, \sigma^ {2}\right), \tag {16.6}
$$

and compare the asymptotic behavior of (16.4) under assumption (16.5) with the asymptotic behavior of the OLS estimator discussed in Chapter 1. Does one estimator dominate the other? (You may assume $\pi = 0 . 5$ , etc., for convenience.)Exercise 3. A common issue in applying the IPW estimator discussed in Chapter 2 arises when there are some units who are a-priori very unlikely to get treated, and have $e ( X _ { i } ) \approx 0$ . This situation could arise, for example, in a medical application where $W _ { i }$ denotes a candidate intervention and some patients are obviously healthy based on their $X _ { i }$ and so will never get treated. And, when $e ( X _ { i } )$ may get close to 0, the IPW estimator (which involves dividing by $e ( X _ { i } ) )$ may be unstable.

One solution to this difficulty is to change statistical targets, and to focus on the average treatment effect on the treated instead:

$$
\tau_ {A T T} = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \mid W _ {i} = 1 \right]. \tag {16.7}
$$

In many applications, focusing on the ATT can improve the precision of the available estimators—and can also improve be of substantive interest (since the ATT measures average the value of the treatment among people who got the treatment in the sampling distribution). Throughout this question, you may assume that the propensity scores $e ( X _ { i } )$ are known a-priori and can be used for estimation, and that $e ( X _ { i } ) \leq 1 - \eta$ for some $\eta > 0$ . You may also take $\mathbb { P } \left[ W _ { i } = 1 \right] = \pi$ to be known.

(a) Propose an IPW-style estimator for the ATT (using the true propensity scores), and prove that it is unbiased.  
(b) Derive the asymptotic variance of estimator derived in part (a), and state a central limit theorem for it.  
(c) Compare the asymptotic variance of the oracle IPW estimators for the ATE and the ATT in a setting where $e ( X _ { i } )$ may get very small, and discuss the robustness of both estimators to small propensity scores.

Exercise 4. In Chapter 2, we defined a propensity-stratified estimator $\hat { \tau } _ { P S T R A T } .$ The purpose of this question is to flesh out our study of this estimator. You may assume that the assumptions of Theorem 2.2 hold, that we have overlap in the sense that $\eta \le e ( x ) \le 1 - \eta$ for all $x \in \mathcal { X }$ , that the distribution of the propensity scores $e ( X )$ admits a density $f _ { e } ( \cdot )$ that is bounded away from 0 on the interval $[ \eta , 1 - \eta ]$ , and that the outcomes are bounded $| Y _ { i } | \le M$ for some large constant M.

(a) Show that if $J \ = \ n ^ { \rho }$ for some constant $0 ~ < ~ \rho ~ < ~ 1$ , then the estimator $\hat { \tau } _ { P S T R A T }$ implemented using the true propensity scores is consistent, i.e., $\hat { \tau } _ { P S T R A T }  _ { p } \tau$ where τ is the average treatment effect.

(b) Conduct a simulation study to evaluate the pros and cons of inversepropensity weighting and stratification. Generate data in R as follows, for n = 100, 200, 400, 800, 1600, 3200 and $p = 1 0 !$ :

$$
\begin{array}{l} X = \text { matrix } (\text { runif } (n * p, - 1, 1), n, p) \\ \text { propensity } = 0. 1 + 0. 8 5 * \operatorname{sqrt} (\operatorname{pmax} (0, 1 + X [, 1 ] + X [, 2 ]) / 3) \\ W = \text { r   b   i   n   o   m } (n, 1, \text { p   r   o   p   e   n   s   i   t   y }) \\ Y = W * \operatorname{pmax} (0, X [, 1 ]) + \exp (X [, 2 ] + X [, 3 ]) \\ \end{array}
$$

Fit propensities $\hat { e }$ via logistic regression, and then estimate $\tau$ via ${ \hat { \tau } } _ { I P W }$ and $\hat { \tau } _ { P S T R A T }$ using the fitted propensities.

What is the average treatment effect τ in this simulation design? What is a good choice for $J ?$ How does the performance of ${ \hat { \tau } } _ { I P W }$ compare to that of $\hat { \tau } _ { P S T R A T }$ in terms of bias? What about in terms of mean-squared error? A good analysis will rely on enough simulation replications to mitigate uncertainty due to Monte Carlo effects, and convey results via appropriate visual displays.

(c) Show that, for a properly chosen sequence $J ( n )$ , the propensity-stratified estimator (now again implemented using the true propensities) is asymptotically unbiased and Gaussian, i.e., $\sqrt { n } ( \hat { \tau } _ { P S T R A T } - \tau ) \Rightarrow \mathcal { N } ( 0 , V _ { P S T R A T } )$ . Propose a consistent variance estimator for $\acute { V } _ { P S T R A T }$ for $V _ { P S T R A T }$ , such that $\widehat { V } _ { P S T R A T } / V _ { P S T R A T } \to _ { p } 1$ . Discuss how these results can be used to build a confidence interval for τ centered at $\hat { \tau } _ { P S T R A T }$ .

(d) In Chapter 3, we showed how to “augment” the inverse-propensity weighted ATE estimator with a regression adjustment, and showed that the resulting AIPW estimator had improved robustness and precision properties relative to the basic IPW estimator. How would you analogously “augment” the propensity stratified estimator studied here? Propose an estimator, and argue for it. (Note: Your argument doesn’t need to be formal; a short qualitative argument is enough.)

Exercise 5. In Corollary 4.3, we gave asymptotic properties of the residualon-residual estimator,

$$
\hat {\tau} _ {R} = \frac {\sum_ {i = 1} ^ {n} \left(Y _ {i} - \hat {m} ^ {(- k (i))} (X _ {i})\right) \left(W _ {i} - \hat {e} ^ {(- k (i))} (X _ {i})\right)}{\sum_ {i = 1} ^ {n} \left(W _ {i} - \hat {e} ^ {(- k (i))} (X _ {i})\right) ^ {2}}, \tag {16.8}
$$

for estimating the treatment parameter τ under the constant treatment effect model $Y _ { i } ( w ) = f ( X _ { i } ) + w \tau + \varepsilon _ { i }$ . The purpose of this question is to study this same residual-on-residual estimator under misspecification of the constant treatment effect hypothesis. Assume that data is independently generated as

$$
\begin{array}{l} Y _ {i} (w) = \mu_ {(w)} \left(X _ {i}\right) + \varepsilon_ {i} (w), \quad \mathbb {E} \left[ \varepsilon_ {i} (w) \mid X _ {i} = x, W _ {i} = w \right] = 0, \\ \mathrm{Y} _ {i} [ \text {一} (\text {一}) \mid X _ {i} = W _ {i} ] = 2 \end{array} \tag {16.9}
$$

$$
\mathrm{Var} \left[ \varepsilon_ {i} (w) \mid X _ {i} = x, W _ {i} = w \right] = \sigma^ {2},
$$

and write $\tau ( x ) = \mu _ { ( 1 ) } ( x ) - \mu _ { ( 0 ) } ( x )$ . Our goal is to characterize asymptotic behavior of $\hat { \tau } _ { R }$ under model (16.9). Throughout this problem you may assume that $e ( x ) \in ( 0 , 1 )$ ; however, overlap is not required.

(a) Let $\hat { \tau } _ { R } ^ { * }$ be the “oracle” version of the estimator (16.8), computed using the true $m ( x )$ and $e ( x )$ . Show that $\hat { \tau } _ { R } ^ { * }$ converges in probability to a limit $\tau _ { R }$ that is a non-negative weighted average of the conditional average treatment effect $\tau ( x )$ , i.e., $\tau _ { R } = \mathbb { E } \left[ \gamma ( X _ { i } ) \tau ( X _ { i } ) \right]$ for some function with $\gamma ( \boldsymbol { x } ) \ge 0$ and $\mathbb { E } \left[ \gamma ( X _ { i } ) \right] = 1$ .  
(b) Show that this oracle estimator satisfies a central limit theorem $\sqrt { n } ( \hat { \tau } _ { R } ^ { * } -$ $\tau _ { R } ) \Rightarrow \mathcal { N } \left( 0 , V _ { R } \right)$ , and provide an expression for $V _ { R }$ . How does $V _ { R }$ compare to the semiparametric efficient variance for average treatment effect estimation?  
(c) Suppose that $\hat { m } ( X _ { i } )$ and $\hat { e } ( X _ { i } )$ satisfy the rate conditions (4.7). Show that $\sqrt { n } ( \hat { \tau } _ { R } - \hat { \tau } _ { R } ^ { * } ) \to _ { p } 0$ , and so the feasible estimator (16.8) also satisfies the central limit theorem established in part (b).

Exercise 6. Consider a hypothetical company that has a phone app that they use to offer $K > 3$ different products that customers can choose to purchase. However, given the size of a phone screen, it can only show 3 (ranked) recommendations to a user at any given time. Your goal is to help the platform evaluate how different ranking strategies affect performance.

You have data on $i = 1 , \dots , n$ IID customers who have interacted with the platform. For each customer, the platform:

• Computes scores $S _ { i 1 } , \ldots , S _ { i K } > 0$ reflecting how well each product is suited to the i-th customer. (These scores are computed by some blackbox algorithm you don’t have access to, but they are recorded and are included in your dataset.)

• Randomly chooses a product $A _ { i } ^ { ( 1 ) }$ to display first, such that

$$
\mathbb {P} \left[ A _ {i} ^ {(1)} = k \right] = e ^ {S _ {i, k}} / \sum_ {\ell = 1} ^ {K} e ^ {S _ {i, \ell}} \text {for all} k = 1, \ldots , K.
$$

• Randomly chooses a product $A _ { i } ^ { ( 2 ) }$ to display second, such that

$$
\mathbb {P} \left[ A _ {i} ^ {(2)} = k \right] = e ^ {S _ {i, k}} \big / \sum_ {\ell \neq A _ {i} ^ {(1)}} e ^ {S _ {i, \ell}} \text {for all} k \neq A _ {i} ^ {(1)}.
$$

• Randomly chooses a product $A _ { i } ^ { ( 3 ) }$ to display second, such that

$$
\mathbb {P} \left[ A _ {i} ^ {(3)} = k \right] = e ^ {S _ {i, k}} \big / \sum_ {\ell \neq A _ {i} ^ {(1)}, A _ {i} ^ {(2)}} e ^ {S _ {i, \ell}} \text {for all} k \neq A _ {i} ^ {(1)}, A _ {i} ^ {(2)}.
$$

• Observes a reward $Y _ { i } .$

For the purpose of the questions below, you should assume that the exact ranking ${ \bf \bar { \chi } } _ { i } ^ { ( 1 ) ^ { \scriptstyle \bullet } } , { \bf \Phi } _ { A _ { i } } ^ { ( 2 ) } , { \bf \Phi } _ { A _ { i } } ^ { ( 3 ) }$ shown to the user matters. Note that the platform does not rank the other products (you may assume, e.g., that if the customer wants to select one of the other products, they need to do so by navigating to a separate static list that shows products in alphabetical order).

We will refer to (both random and deterministic) methods for ranking products as policies, and to the expected reward the platform would achieve by deploying a policy as the value V of the policy. The available data

$$
\mathcal {D} _ {n} = \left\{S _ {i}, A _ {i} ^ {(1)}, A _ {i} ^ {(2)}, A _ {i} ^ {(3)}, Y _ {i} \right\} _ {i = 1} ^ {n}
$$

generated as described above, is the same for all 4 parts below. An unbiased estimator of policy value V is a (measurable) function $\widehat { V }$ of the observed data $\mathcal { D } _ { n }$ for which $\mathbb { E } [ \tilde { V } ] = V$ . We assume that each unit has potential outcomes $Y _ { i } ( a _ { 1 } , a _ { 2 } , a _ { 3 } )$ such that the observed reward is

$$
Y _ {i} = Y _ {i} \left(A _ {i} ^ {(1)}, A _ {i} ^ {(2)}, A _ {i} ^ {(3)}\right),
$$

and the value of a policy $\pi$ is

$$
V (\pi) = \mathbb {E} _ {A _ {i} \sim \pi (S _ {i})} \left[ Y _ {i} (A _ {i}) \right], \quad A _ {i} = \left(A _ {i} ^ {(1)}, A _ {i} ^ {(2)}, A _ {i} ^ {(3)}\right),
$$

where $A _ { i } \ \sim \ \pi ( S _ { i } )$ means that $A _ { i }$ is generated via the (potentially random) function π of $S _ { i }$ .

(a) Propose an estimator that, given the available data $\mathcal { D } _ { n }$ , gives an unbiased estimate of the value of the current randomized policy (i.e., the policy used in data collection).  
(b) Propose an estimator that, given the available data $\mathcal { D } _ { n } .$ gives an unbiased estimate of the value of a policy that always uses a fixed ranking $a _ { 1 }$ , a2, a3 (i.e., sets $A _ { i } ^ { ( 1 ) } = a _ { 1 } , A _ { i } ^ { ( 2 ) } = \stackrel { \cdot } { a } _ { 2 } , \stackrel { \cdot } { A } _ { i } ^ { ( c ) } = a _ { 3 }$ for some $1 \leq a _ { 1 } \neq a _ { 2 } \neq a _ { 3 } \leq K )$ .  
(c) Propose an estimator that, given the available data ${ \mathcal { D } } _ { n } ,$ gives an unbiased estimate of the value of a randomized policy that always shows some product

$a _ { 1 }$ first $( \mathrm { i . e . }$ , deterministically sets $A _ { i } ^ { ( 1 ) } = a _ { 1 }$ for some $1 \leq a _ { 1 } \leq K )$ , but then randomly chooses A(2)i $A _ { i } ^ { ( 2 ) }$ and A(3)i $\check { A } _ { i } ^ { ( 3 ) }$ using the available scores in the same way as with the data collection policy.

(d) Propose an estimator that, given the available data $\mathcal { D } _ { n } .$ gives an unbiased estimate of the value of a randomized policy that never shows some product $a _ { 0 }$ with $1 \le a _ { 0 } \le K$ , but otherwise randomly draws random products using scores as with the data collection policy (operationally, you could assume that $A _ { i } ^ { ( \ell ) } = \bar { a _ { 0 } }$ the same distribution until $A _ { i } ^ { \overline { { ( \ell ) } } } \neq a _ { 0 } )$ .

Exercise 7. Consider the following model for adaptive data-collection $( \eta > 0$ is a tuning parameter): For $t = 1 , \dots , T$ time steps, we

• Choose a probability $\omega _ { t } \in [ \eta , 1 ]$ , potentially using past data.
• Draw a Bernoulli random variable $Z _ { t } \sim \mathrm { B e r n } ( \omega _ { t } )$
• If $Z _ { t } = 1$ , we observe a draw $Y _ { t } \sim F ;$ ; while if $Z _ { t } = 0$ , we cannot make an observation (equivalently, we hard-code $Y _ { t } = 0 )$ .

Our goal is to estimate the mean $\mu = \mathbb { E } _ { F } [ Y ]$ , and are considering 3 different estimators:

1. Sample average: $\begin{array} { r } { \hat { \mu } _ { 1 } = \sum _ { \{ t : Z _ { t } = 1 \} } Y _ { t } / \left| \left\{ t : Z _ { t } = 1 \right\} \right| } \end{array}$

2. Inverse-propensity weighting: $\begin{array} { r } { \hat { \mu } _ { 2 } = T ^ { - 1 } \sum _ { t = 1 } ^ { T } Z _ { t } Y _ { t } / \omega _ { t } } \end{array}$

3. Stabilized inverse-propensity weighting: $\hat { \mu } _ { 3 }$ $\begin{array} { r } { \sum _ { t = 1 } ^ { T } Z _ { t } Y _ { t } / \omega _ { t } \ : / \ : \sum _ { t = 1 } ^ { T } \dot { Z } _ { t } \dot { / } \omega _ { t } . } \end{array}$

Answer the following questions. To avoid degenerate cases, you may assume that $\omega _ { 1 } = 1 , \mathrm { i . e . }$ , we always collect at least 1 sample. You may also make any regularity assumption you find to be convenient $( \mathrm { e . g . }$ , that the $Y _ { t }$ have bounded support).

(a) Which of the 3 estimators above are unbiased, i.e., satisfy $\mathbb { E } \left[ \hat { \mu } \right] = \mu ?$ Provide a proof or counterexample.

(b) Now consider a large-sample limit, with $T \to \infty$ . In this setting, we say that an estimator is asymptotically unbiased if

$$
\lim _ {T \to \infty} \sqrt {T} \left(\mathbb {E} [ \hat {\mu} ] - \mu\right) = 0.
$$

Which of the 3 estimators above are asymptotically unbiased? Provide a proof or counterexample.

Exercise 8. Theorem 7.1 provides the asymptotic distribution of the covariatebalancing propensity score estimator $\hat { \tau } _ { C B P S }$ under a linear-logistic specification where both

$$
\mu_ {(w)} = x \cdot \beta_ {(w)}, \quad \beta_ {(w)} \in \mathbb {R} ^ {p} \quad \text { for } w = 0, 1, \tag {16.10}
$$

$$
e (x) = 1 / \left(1 + e ^ {- x \cdot \theta}\right), \quad \theta \in \mathbb {R} ^ {p}, \quad \| \theta \| _ {2} <   \infty . \tag {16.11}
$$

The goal of this question is to study double robustness properties of $\hat { \tau } _ { C B P S }$ . 78 In answering this question, you may replace the exponential moment condition (7.12) with the stronger boundedness condition $\| X _ { i } \| _ { 2 } \leq M$ .

(a) Under the setting of Theorem 7.1, suppose that (16.10) holds but that (16.11) may not hold. Prove that $\hat { \tau } _ { C P B S } \to _ { p } \tau$ , where τ denotes the ATE. You may assume that strong overlap holds, $\eta \leq e ( X _ { i } ) \leq 1 - \eta$ , if convenient.  
(b) Under the setting of Theorem 7.1, suppose conversely that (16.11) holds but that (16.10) may not hold. Prove that $\hat { \tau } _ { C P B S }  _ { p } \tau .$ . You may assume that outcomes are bounded, $| Y _ { i } | \le M$ , if convenient.

Exercise 9. Under the conditions of Theorem 7.1 suppose that, rather than the ATE, we want to estimate the average treatment effect on the treated (ATT) as in Exercise 3, $\tau _ { A T T } = \mathbb { E } \left[ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \big | W _ { i } = 1 \right]$ . We claim that

$$
\hat {\theta} = \operatorname{argmin} _ {\theta} \left\{\frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} \left(\left(1 - W _ {i}\right) e ^ {X _ {i} \theta} - W _ {i} X _ {i} \theta\right) \right\}, \tag {16.12}
$$

$$
\hat {\tau} _ {C B P S - A T T} = \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} \left(W _ {i} Y _ {i} - (1 - W _ {i}) e ^ {X _ {i} \hat {\theta}} Y _ {i}\right), \tag {16.13}
$$

is the natural CBPS estimator for this task, and has good statistical properties.

(a) Verify that (16.12) is a convex minimization problem.  
(b) Verify that (16.13) is in fact a CBPS estimator, i.e., that it is the IPW estimator for some specific choice $\hat { e } ( x ) = 1 / \left( 1 + e ^ { x \hat { \theta } } \right)$ , and that $\hat { \theta }$ satisfies a relevant sample-balance condition whenever the minimization problem (16.12) has an interior solution $( \mathrm { i . e . , ~ } \lVert \hat { { \boldsymbol { \theta } } } \rVert < \infty )$ .  
(c) Prove that $\hat { \tau } _ { C B P S - A T T }$ is consistent for $\tau _ { A T T }$ , and establish a central limit theorem. For simplicity, you may assume that $\| X _ { i } \| _ { 2 } \leq M$ uniformly.

Exercise 10. Consider an IID sequence $( X _ { i } , U _ { i } , Y _ { i } , W _ { i } ) \in \mathcal { X } \times \mathcal { U } \times \mathbb { R } \times \{ 0 , 1 \}$ , where $Y _ { i } = Y _ { i } ( W _ { i } )$ for a pair of potential outcomes $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \}$ . Unconfoundedness holds conditionally on $X _ { i }$ and $U _ { i }$ , i.e.,

$$
\{Y _ {i} (0), Y _ {i} (1) \} \perp W _ {i} \mid X _ {i}, U _ {i}. \tag {16.14}
$$

However, only $X _ { i }$ is observed, whereas $U _ { i }$ is an unobserved confounder. In this question, we’ll study the behavior of (stabilized) IPW estimators of $\mu ( 1 ) =$ $\mathbb { E } \left[ Y _ { i } ( 1 ) \right]$ in the presence of unobserved confounding. To this end, define both the feasible and infeasible IPW estimators, the latter of which makes use of the unobserved $U _ { i } { \mathbf { : } }$

$$
\hat {\mu} _ {S I P W} (1) = \sum_ {i = 1} ^ {n} \frac {W _ {i} Y _ {i}}{e \left(X _ {i}\right)} / \sum_ {i = 1} ^ {n} \frac {W _ {i}}{e \left(X _ {i}\right)}, \tag {16.15}
$$

$$
\tilde {\mu} _ {S I P W} (1) = \sum_ {i = 1} ^ {n} \frac {W _ {i} Y _ {i}}{e (X _ {i} , U _ {i})} \Bigg / \sum_ {i = 1} ^ {n} \frac {W _ {i}}{e (X _ {i} , U _ {i})},
$$

where $e ( x ) = \mathbb { P } \left[ W _ { i } = 1 \big | X _ { i } = x \right]$ and $e ( x , u ) \ : = \ : \mathbb { P } \lceil W _ { i } = 1 \rceil X _ { i } = x , U _ { i } = u \rceil$ . Under the unconfoundedness condition (16.14), ˜µSIPW (1) is clearly consistent for $\mu ( 1 )$ , but ${ \hat { \mu } } _ { S I P W } ( 1 )$ may not be.

In general, it’s not possible to say much about the bias of ${ \hat { \mu } } _ { S I P W } ( 1 )$ . Thus, we’ll make a further assumption about how the unobserved $U _ { i }$ may affect sampling probabilities, and assume that we know a constant $\Gamma \geq 1$ such that

$$
\frac {1}{\Gamma} \leq \frac {e (X _ {i} , U _ {i})}{e (X _ {i})} \leq \Gamma \text {   for   all   } i = 1,..., n, \tag {16.16}
$$

almost surely. This assumption is commonly known as the marginal sensitivity model, and can be used to assess the sensitivity of IPW to hidden confounding.

(a) Under (16.16), show that there exist weights $\Gamma _ { i } ^ { - 1 } \leq \gamma _ { i } \leq \Gamma _ { i }$ such that

$$
\tilde {\mu} _ {S I P W} (1) = \hat {\mu} _ {S I P W} (1; \gamma) := \sum_ {i = 1} ^ {n} \gamma_ {i} \frac {W _ {i} Y _ {i}}{e (X _ {i})} / \sum_ {i = 1} ^ {n} \gamma_ {i} \frac {W _ {i}}{e (X _ {i})}. \tag {16.17}
$$

(b) Given (16.17), we have the following upper bound for $\tilde { \mu } _ { S I P W } ( 1 )$ :

$$
\hat {\mu} _ {S I P W} ^ {+} (1) = \sup \left\{\hat {\mu} _ {S I P W} (1; \gamma): \Gamma_ {i} ^ {- 1} \leq \gamma_ {i} \leq \Gamma_ {i} \right\}. \tag {16.18}
$$

Show that the above optimization program can be solved by linear programming, and express the problem in a way that could be plugged into standard linear programming software, i.e., in format “maximize $c ^ { \prime } x$ subject to $A x \le b ^ { \prime }$ , where we optimize over the vector x and take $A ,$ b and c as given.

Hint. Consider the Charnes-Cooper transformation for linear-fractional programming.

(c) Using the construction in (16.18), propose an interval

$$
\widehat {I} _ {S I P W} (1) = \left[ \hat {\mu} _ {S I P W} ^ {-} (1), \hat {\mu} _ {S I P W} ^ {+} (1) \right] \tag {16.19}
$$

that does not use the unobserved $U _ { i } .$ , but has the property that $\tilde { \mu } _ { S I P W } ( 1 ) \in \widehat { I } _ { S I P W } ( 1 )$ almost surely. Show that the interval $\hat { I } _ { S I P W } ( \bar { 1 } )$ is consistent for $\mu ( 1 )$ in the following sense: For any $\varepsilon > 0$

$$
\lim _ {n \rightarrow \infty} \mathbb {P} [ \mu (1) \in (\hat {\mu} _ {S I P W} ^ {-} (1) - \varepsilon , \hat {\mu} _ {S I P W} ^ {+} (1) + \varepsilon) ] = 1. \tag {16.20}
$$

In doing so, you may make any regularity assumptions you find to be convenient (e.g., bounds on moments).

(d) Discuss how the intervals (16.19) could be used in practical data analysis to assess the sensitivity of IPW to the potential presence of unobserved confounders.

Exercise 11. Consider the following structural model, where $( X _ { i } , Y _ { i } , W _ { i } , Z _ { i } ) \in \mathcal { X } \times \mathbb { R } \times \{ 0 , 1 \} \times \{ 0 , 1 \}$ are taken to be IID:

$$
\begin{array}{l} Y _ {i} = \alpha \left(X _ {i}\right) + W _ {i} \tau \left(X _ {i}\right) + \varepsilon_ {i}, \quad \varepsilon_ {i} \perp Z _ {i} \mid X _ {i}, \quad \mathbb {E} \left[ \varepsilon_ {i} \mid X _ {i} \right] = 0 \\ C = \left[ W _ {i} - Z _ {i} \mid X _ {i} - \dots \right] > 0, f (x) = 1, \dots , y. \end{array} \tag {16.21}
$$

$\mathrm { ~ \mathsf { C o v } ~ } \lfloor W _ { i } , \mathrm { ~ } Z _ { i } \rfloor \lambda _ { i } = x \rfloor \ge \eta > 0 \mathrm { ~ \quad ~ f o r ~ a u l ~ } x \in \mathcal { A } .$

In other words, conditionally on covariates $X _ { i } .$ , this is the same structural model as used in Chapter $9 . 2 ;$ now, however, all problem primitives may also vary with x. Furthermore, we assumed that the effect of the instrument on the outcome is always positive and uniformly bounded from below.

Your goal is to develop methods to estimate the average treatment effect parameter $\tau = \mathbb { E } \left[ \tau ( X ) \right]$ . In all parts below, you may make any regularity assumptions you find to be helpful (e.g., boundedness of outcomes).

(a) Define the “compliance score” $\Delta ( x )$ and the associated inverse-compliance weighted estimator,

$$
\Delta (x) = \mathbb {P} \left[ W _ {i} = 1 \mid Z _ {i} = 1, X _ {i} = x \right] - \mathbb {P} \left[ W _ {i} = 1 \mid Z _ {i} = 0, X _ {i} = x \right],
$$

$$
\hat {\tau} _ {I C W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {1}{\Delta (X _ {i})} \left(\frac {Z _ {i} Y _ {i}}{z (X _ {i})} - \frac {(1 - Z _ {i}) Y _ {i}}{1 - z (X _ {i})}\right), \tag {16.22}
$$

where $z ( x ) = \mathbb { P } \left[ Z _ { i } = 1 \big | X _ { i } = x \right]$ is an analogue to the propensity score for the instrument $Z _ { i }$ . Prove that the oracle inverse-compliance weighted estimator (i.e., using the true values of $z ( \cdot )$ and $\Delta ( \cdot ) )$ is unbiased and consistent for τ .

(b) Now suppose you obtain estimates ${ \hat { \alpha } } ( x )$ and ${ \hat { \tau } } ( x )$ for the structural parameters in (16.21). Propose an augmented inverse-compliance weighted (AICW) estimator. Argue that your AICW estimator is (weakly) doubly robust, i.e., it is consistent if either ${ \hat { \alpha } } ( x )$ and $\hat { \tau } ( x )$ are sup-norm consistent, or $\widehat { \Delta } ( x )$ and $\hat { z } ( x )$ are sup-norm consistent (where $\widehat { \Delta } ( x )$ and $\hat { z } ( x )$ are feasible estimates of the nuisance components in (16.22)). A high-level argument is enough here; no need to go into details.79  
(c) Show that if all the nuisance components $\hat { \alpha } ( x ) , \hat { \tau } ( x ) , \hat { \Delta } ( x )$ and $\hat { z } ( x )$ are both sup-norm consistent and $o _ { p } ( n ^ { - 1 / 4 } )$ consistent in root-mean squared error, then AICW with cross-fitting is n-consistent for τ and asymptotically normal. Write down a central limit theorem, and provide an expression for the limiting variance of AICW.

Exercise 12. In Chapter 10.1, we studied instrumental variables regression with a binary treatment and binary instrument. We showed that under a “no defiers” assumption, i.e.,

$$
\mathbb {P} \left[ W _ {i} (0) <   W _ {i} (1) \right] = 0, \tag {16.23}
$$

the instrumental variables estimator converges to the average treatment effect estimator for the compliers. Your goal in this question is to understand what happens when we relax this assumption.

Under the setting of Theorem 10.1, suppose now that we may have defiers, but there exist unobserved latent factors $U _ { i }$ for which

$$
\mathbb {P} \left[ W _ {i} = 1 \mid Z _ {i} = 1, U _ {i} = u \right] > \mathbb {P} \left[ W _ {i} = 1 \mid Z _ {i} = 0, U _ {i} = u \right], \tag {16.24}
$$

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp C _ {i} \mid U _ {i} = u, \text {for all} u,
$$

i.e., given the unobserved latent factors, we assume that the treatment effect is independent of compliance type, and that all latent types are more likely to comply than to defy. Also assume that $Z _ { i }$ is still exogenous once we include the $U _ { i }$ into the model,

$$
Z _ {i} \perp \left\{U _ {i}, Y _ {i} (0), Y _ {i} (1), W _ {i} (0), W _ {i} (1) \right\}.
$$

Write an expression for $\tau _ { L A T E }$ in terms of

$$
\tau (u) = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \mid U _ {i} = u \right],
$$

$$
\kappa (u) = \mathbb {P} \left[ C _ {i} = \text {complier} \mid U _ {i} = u \right], \text {and}
$$

$$
\delta (u) = \mathbb {P} \left[ C _ {i} = \text { defier } \mid U _ {i} = u \right].
$$

Show that, if $\tau ( u ) \geq 0$ for all u, then $\tau _ { L A T E } \geq 0$

Exercise 13. Consider a set of n random variables $( W _ { i } , Y _ { i } ) \in \{ 0 , 1 \} \times \mathbb { R }$ Assume that the data is generated as follows:

• Each unit $i = 1 , \ldots , n$ is characterized by (deterministic) parameters $\alpha _ { i } .$ $\beta _ { i } , \gamma _ { i } \in \mathbb { R }$ .
• We choose a treatment probability $\pi \in [ 0 , 1 ]$ , and independently generate $W _ { i } \sim \mathrm { B e r n o u l l i } ( \pi )$ for each $i = 1 , \ldots , n$ .
• We observe the following, where $\varepsilon _ { i } \sim \mathcal { N } \left( 0 , \sigma ^ { 2 } \right)$ independently of everything else:

$$
Y _ {i} = \alpha_ {i} + \beta_ {i} W _ {i} + \gamma_ {i} \frac {\sum_ {j \neq i} W _ {j}}{n - 1} + \varepsilon_ {i}
$$

We use the notation $\mathbb { E } _ { \pi } \left[ Y _ { i } \right]$ for the expectation of the i-th outcome under this model (with treatment probability $\pi )$ , as well as immediate generalizations of this notation. Note: Qualitatively, $\alpha _ { i }$ captures the i-th unit’s baseline effect, $\beta _ { i }$ its sensitivity to its own treatment, and $\gamma _ { i }$ its sensitivity to the fraction of other units who are treated.

(a) What is the total effect, i.e., the expected difference in average outcomes when everyone is treated vs. when no one is:

$$
\tau_ {T O T} = \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} _ {1} [ Y _ {i} ] - \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} _ {0} [ Y _ {i} ].
$$

(b) Now suppose we are able to collect observations at a single $\pi \in ( 0 , 1 )$ , and seek to estimate the effect of the treatment via the na¨ıve inverse-propensity weighted estimator that ignores spillovers,

$$
\hat {\tau} _ {I P W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{\pi} - \frac {(1 - W _ {i}) Y _ {i}}{1 - \pi}\right).
$$

What is $\mathbb { E } _ { \pi } \left[ \hat { \tau } _ { I P W } \right] ?$

(c) In the same setting as above, what is $\operatorname { V a r } _ { \boldsymbol { \pi } } \left[ \hat { \tau } _ { I P W } \right] ?$  
(d) Is ${ \hat { \tau } } _ { I P W }$ a good estimator of $\tau _ { T O T }$ in this model? Can ${ \hat { \tau } } _ { I P W }$ be used to learn anything interesting in this model?

Exercise 14. One important question in survival analysis is to assess prognosis given a diagnosis. We have data on $i = 1 , \dots , n$ people who are diagnosed with a condition at time $t = 0 ;$ at this time, we also measure time-invariant convariates $X _ { i } \in { \mathcal { X } }$ . We write $Y _ { i }$ for the length of time the i-th person survives post-diagnosis, and are interested in estimating $\theta = \mathbb { P } \left[ Y _ { i } > T \right]$ for some targethorizon $T$ .

The challenge, however, is that we may lose track of some patients in our study before we get to see whether they live past time $T .$ . Specifically, we will assume that we follow-up with each patient at a set of pre-determined times $t = 1 , \dots , T$ , and at each of these follow-ups we either are able track down the patient (in which case we can observe whether the patient is still alive, i.e., whether $Y _ { i } > t )$ , or we are unable to track down the patient and deem them to be censored at time t (and we cease further follow-up attempts).

Formally, we assume that each unit has a (potentially non-realized) censoring time $C _ { i } \in \{ 1 , 2 , . . . , T , + \infty \}$ , where $C _ { i } = + \infty$ means the unit is never censored. We then assume that, rather than getting to directly observe survival time $Y _ { i }$ , we only have access to

$$
U _ {i} = \min \left\{C _ {i}, Y _ {i} \right\}, \quad \Delta_ {i} = 1 \left(Y _ {i} <   C _ {i}\right), \tag {16.25}
$$

which we refer to as the observation time and the non-censoring indicator respectively. Let

$$
\overline {{U}} _ {i} = \inf \left\{t \in \{1, 2, \dots , T, + \infty \}: t \geq U _ {i} \right\}, \quad H _ {i} = \min \left\{\overline {{U}} _ {i}, T \right\}, \tag {16.26}
$$

respectively denote the time of the follow-up time at which the observation is recorded (e.g., if someone dies at time 1.5, we only learn about this at the time $t = 2 \mathrm { f o l l o w \mathrm { - } u p } )$ , and the time of the last visit $( \mathrm { i . e . , } \ H _ { i } = T$ even if the patient is still alive and uncensored at that point).

We also make the following statistical assumptions:

• Censoring is ignorable, i.e.,

$$
Y _ {i} \perp C _ {i} \mid X _ {i}; \tag {16.27}
$$

• Some patients are never censored, i.e., there is an $\eta > 0$ such that

$$
\mathbb {P} \left[ C _ {i} > T \mid X _ {i} = x \right] \geq \eta \text {   for   all   } x \in \mathcal {X}. \tag {16.28}
$$

Note that these assumptions are closely related to our familiar assumptions of unconfoundedness and overlap for treatment effect estimation.

We define the conditional survival functions

$$
S _ {Y} (t; x) = \mathbb {P} \left[ Y _ {i} > t \mid X _ {i} = x \right], \quad S _ {C} (t; x) = \mathbb {P} \left[ C _ {i} > t \mid X _ {i} = x \right], \tag {16.29}
$$

with a convention that $S _ { Y } ( 0 ; x ) = S _ { C } ( 0 ; x ) = 1$ . We will assume that we have access to estimates for these objects using a separate training set.80

(a) Suppose that the survival function for the censoring distribution $S _ { C } ( t ; x )$ is known. Show that, under our assumptions, the following inverse-probability of censoring (IPCW) estimator is unbiased for θ:

$$
\hat {\theta} _ {I P C W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {\Delta_ {i} 1 (\{U _ {i} > T \})}{S _ {C} (U _ {i} ; X _ {i})}. \tag {16.30}
$$

(b) Now, consider a setting where we have access to estimates $\widehat { S } _ { Y } ( t ; x )$ and $\widehat { S } _ { C } ( t ; x )$ obtained using a separate training set, and consider the following augmented IPCW (AIPCW) estimator:81

$$
\hat {\theta} _ {A I P C W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \widehat {S} _ {Y} (T; X _ {i})
$$

$$
+ \sum_ {t = 1} ^ {H _ {i} - 1} \frac {1}{\widehat {S} _ {C} (t ; X _ {i})} \left(\frac {\widehat {S} _ {Y} (T ; X _ {i})}{\widehat {S} _ {Y} (t ; X _ {i})} - \frac {\widehat {S} _ {Y} (T ; X _ {i})}{\widehat {S} _ {Y} (t - 1 ; X _ {i})}\right) \tag {16.31}
$$

$$
+ \frac {\Delta_ {i}}{\widehat {S} _ {C} (H _ {i} ; X _ {i})} \left(1 (\{U _ {i} > T \}) - \frac {\widehat {S} _ {Y} (T ; X _ {i})}{\widehat {S} _ {Y} (H _ {i} - 1 ; X _ {i})}\right),
$$

where $H _ { i }$ is as defined in (16.26). Show that, under our setting, if furthermore

$$
\mathbb {E} \left[ \left(1 / \widehat {S} _ {C} (t; X _ {i}) - 1 / S _ {C} (t; X _ {i})\right) ^ {2} \right] = o _ {P} \left(n ^ {- 2 \alpha_ {C}}\right), \tag {16.32}
$$

$$
\mathbb {E} \left[ \left(1 / \widehat {S} _ {Y} (t; X _ {i}) - 1 / S _ {Y} (t; X _ {i})\right) ^ {2} \right] = o _ {P} \left(n ^ {- 2 \alpha_ {Y}}\right)
$$

for constants $\alpha _ { C } , \alpha _ { Y } \ge 0$ with $\alpha _ { C } + \alpha _ { Y } \ge 1 / 2$ , then

$$
\sqrt {n} \left(\hat {\theta} _ {A I P C W} - \theta\right) \Rightarrow \mathcal {N} \left(0, \sigma_ {A I P C W} ^ {2}\right)
$$

$$
\sigma_ {A I P C W} ^ {2} = \operatorname{Var} \left[ S _ {Y} (T; X _ {i}) \right] \tag {16.33}
$$

$$
+ \sum_ {t = 1} ^ {T} \mathbb {E} \left[ \frac {S _ {Y} ^ {2} (T ; X _ {i})}{S _ {C} (t ; X _ {i})} \frac {S _ {Y} (t - 1 ; X _ {i}) - S _ {Y} (t ; X _ {i})}{S _ {Y} (t - 1 ; X _ {i}) S _ {Y} (t ; X _ {i})} \right].
$$

Hint: This result is a corollary of Theorem 14.3. To establish this, imagine an analogous dynamic policy evaluation problem where there is no censoring; however, all units start under the status-quo treatment, but then transition to an experimental treatment at time $C _ { i }$ if they are still alive. Argue that estimating θ in the setting of this question is equivalent to estimating $\mathbb { P } _ { \pi _ { 0 } } \left[ Y _ { i } > T \right]$ for the analogous dynamic policy evaluation setting with $\pi _ { 0 }$ corresponding to the policy that never starts the experimental treatment; and that $\hat { \theta } _ { A I P C W }$ is equivalent to the doubly robust estimator $\widehat { V } _ { A I P W } ( \pi _ { 0 } )$ derived in Chapter 14. Thus statistical properties of $\hat { \theta } _ { A I P C W }$ can be derived from Theorem 14.3.

## Bibliography

Alberto Abadie. Semiparametric instrumental variable estimation of treatment response models. Journal of Econometrics, 113(2):231–263, 2003.  
Alberto Abadie and Javier Gardeazabal. The economic costs of conflict: A case study of the Basque country. American Economic Review, 93(1):113– 132, 2003.  
Alberto Abadie and Guido W Imbens. Large sample properties of matching estimators for average treatment effects. Econometrica, 74(1):235–267, 2006.  
Alberto Abadie and Guido W Imbens. Matching on the estimated propensity score. Econometrica, 84(2):781–807, 2016.  
Alberto Abadie, Alexis Diamond, and Jens Hainmueller. Synthetic control methods for comparative case studies: Estimating the effect of california’s tobacco control program. Journal of the American Statistical Association, 105(490):493–505, 2010.  
Alberto Abadie, Susan Athey, Guido W Imbens, and Jeffrey M Wooldridge. When should you adjust standard errors for clustering? The Quarterly Journal of Economics, 138(1):1–35, 2023.  
Anish Agarwal, Devavrat Shah, Dennis Shen, and Dogyoon Song. On robustness of principal component regression. Journal of the American Statistical Association, 116(536):1731–1745, 2021.  
Shipra Agrawal and Navin Goyal. Near-optimal regret bounds for Thompson sampling. Journal of the ACM, 64(5):1–24, 2017.  
Luigi Ambrosio and Gianni Dal Maso. A general chain rule for distributional derivatives. Proceedings of the American Mathematical Society, 108(3):691– 702, 1990.  
Takeshi Amemiya. The nonlinear two-stage least-squares estimator. Journal of Econometrics, 2(2):105–110, 1974.  
Joshua D Angrist. Lifetime earnings and the Vietnam era draft lottery: Evidence from social security administrative records. American Economic Review, 80(3):313–336, 1990.  
Joshua D Angrist and Alan B Krueger. Split-sample instrumental variables estimates of the return to schooling. Journal of Business & Economic Statistics, 13(2):225–235, 1995.  
Joshua D Angrist, Guido W Imbens, and Donald B Rubin. Identification of causal effects using instrumental variables. Journal of the American Statistical Association, 91(434):444–455, 1996.  
Joshua D Angrist, Kathryn Graddy, and Guido W Imbens. The interpretation of instrumental variables estimators in simultaneous equations models with an application to the demand for fish. The Review of Economic Studies, 67 (3):499–527, 2000.  
Kevin Arceneaux, Alan S Gerber, and Donald P Green. Comparing experimental and matching methods using a large-scale voter mobilization experiment. Political Analysis, 14(1):37–62, 2006.  
Manuel Arellano. Panel Data Econometrics. Oxford university press, 2003.  
Dmitry Arkhangelsky and David Hirshberg. Large-sample properties of the synthetic control method under selection on unobservables. arXiv preprint arXiv:2311.13575, 2023.  
Dmitry Arkhangelsky and Guido Imbens. Causal models for longitudinal and panel data: A survey. arXiv preprint arXiv:2311.15458, 2023.  
Dmitry Arkhangelsky, Susan Athey, David A Hirshberg, Guido W Imbens, and Stefan Wager. Synthetic difference-in-differences. American Economic Review, 111(12):4088–4118, 2021.  
Timothy B Armstrong and Michal Koles´ar. Optimal inference in a class of regression models. Econometrica, 86(2):655–683, 2018.  
Timothy B Armstrong and Michal Koles´ar. Simple and honest confidence intervals in nonparametric regression. Quantitative Economics, 11(1):1–39, 2020.  
Peter M Aronow. A general method for detecting interference between units in randomized experiments. Sociological Methods & Research, 41(1):3–16, 2012.  
Peter M Aronow and Allison Carnegie. Beyond LATE: Estimation of the average treatment effect with an instrumental variable. Political Analysis, 21 (4):492–506, 2013.  
Peter M Aronow and Cyrus Samii. Estimating average causal effects under general interference, with application to a social network experiment. The Annals of Applied Statistics, 11(4):1912–1947, 2017.  
Peter M Aronow, Donald P Green, and Donald KK Lee. Sharp bounds on the variance in randomized experiments. The Annals of Statistics, 42(3): 850–871, 2014.  
Susan Athey and Guido W Imbens. Recursive partitioning for heterogeneous causal effects. Proceedings of the National Academy of Sciences, 113(27): 7353–7360, 2016.  
Susan Athey and Guido W Imbens. Design-based analysis in difference-indifferences settings with staggered adoption. Journal of Econometrics, 226 (1):62–79, 2022.  
Susan Athey and Stefan Wager. Estimating treatment effects with causal forests: An application. Observational Studies, 5:36–51, 2019.  
Susan Athey and Stefan Wager. Policy learning with observational data. Econometrica, 89(1):133–161, 2021.  
Susan Athey, Dean Eckles, and Guido W Imbens. Exact p-values for network interference. Journal of the American Statistical Association, 113(521):230– 240, 2018a.  
Susan Athey, Guido W Imbens, and Stefan Wager. Approximate residual balancing: Debiased inference of average treatment effects in high dimensions. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 80(4):597–623, 2018b.  
Susan Athey, Julie Tibshirani, and Stefan Wager. Generalized random forests. The Annals of Statistics, 47(2):1148–1178, 2019.  
Susan Athey, Mohsen Bayati, Nikolay Doudchenko, Guido Imbens, and Khashayar Khosravi. Matrix completion methods for causal panel data models. Journal of the American Statistical Association, 116(536):1716–1730, 2021.  
Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2-3):235–256, 2002.  
Jushan Bai. Panel data models with interactive fixed effects. Econometrica, 77(4):1229–1279, 2009.  
Pierre Baldi and Yosef Rinott. On normal approximations of distributions in terms of dependency graphs. The Annals of Probability, 17(4):1646–1650, 1989.  
Heejung Bang and James M Robins. Doubly robust estimation in missing data and causal inference models. Biometrics, 61(4):962–973, 2005.  
Guillaume W Basse, Avi Feller, and Panos Toulis. Randomization tests of causal effects under interference. Biometrika, 106(2):487–494, 2019.  
Hamsa Bastani and Mohsen Bayati. Online decision making with highdimensional covariates. Operations Research, 68(1):276–294, 2020.  
Eli Ben-Michael, Avi Feller, and Jesse Rothstein. The augmented synthetic control method. Journal of the American Statistical Association, 116(536): 1789–1803, 2021.  
Marianne Bertrand, Esther Duflo, and Sendhil Mullainathan. How much should we trust differences-in-differences estimates? The Quarterly Journal of Economics, 119(1):249–275, 2004.  
Dimitris Bertsimas and Nathan Kallus. From predictive to prescriptive analytics. Management Science, 66(3):1025–1044, 2020.  
Omar Besbes, Yonatan Gur, and Assaf Zeevi. Optimal exploration–exploitation in a multi-armed bandit problem with non-stationary rewards. Stochastic Systems, 9(4):319–337, 2019.  
Peter J Bickel, Chris AJ Klaassen, Ya’acov Ritov, and Jon A Wellner. Efficient and adaptive estimation for semiparametric models. Johns Hopkins University Press Baltimore, 1993.  
Christopher Blattman, Donald P Green, Daniel Ortega, and Santiago Tob´on. Place-based interventions at scale: The direct and spillover effects of policing and city services on crime. Journal of the European Economic Association, 19(4):2022–2051, 2021.  
Adam Bloniarz, Hanzhong Liu, Cun-Hui Zhang, Jasjeet S Sekhon, and Bin Yu. Lasso adjustments of treatment effect estimates in randomized experiments. Proceedings of the National Academy of Sciences, 113(27):7383–7390, 2016.  
Gregor Boehl, Gavin Goy, and Felix Strobel. A structural investigation of quantitative easing. Review of Economics and Statistics, 106(4):1028–1044, 2024.  
Iavor Bojinov, David Simchi-Levi, and Jinglong Zhao. Design and analysis of switchback experiments. Management Science, 69(7):3759–3777, 2023.  
Kirill Borusyak, Xavier Jaravel, and Jann Spiess. Revisiting event study designs: Robust and efficient estimation. Review of Economic Studies, forthcoming, 2024.  
John Bound, David A Jaeger, and Regina M Baker. Problems with instrumental variables estimation when the correlation between the instruments and the endogenous explanatory variable is weak. Journal of the American Statistical Association, 90(430):443–450, 1995.  
Richard C Bradley. Basic properties of strong mixing conditions: A survey and some open questions. Probability Surveys, 2:107–144, 2005.  
Leo Breiman. Random forests. Machine Learning, 45(1):5–32, 2001.  
S´ebastien Bubeck and Nicolo Cesa-Bianchi. Regret analysis of stochastic and nonstochastic multi-armed bandit problems. Foundations and Trends® in Machine Learning, 5(1):1–122, 2012.  
S´ebastien Bubeck, R´emi Munos, and Gilles Stoltz. Pure exploration in multiarmed bandits problems. In Proceedings of the 20th International Conference Algorithmic Learning Theory, pages 23–37. Springer, 2009.  
Andreas Buja, Lawrence Brown, Richard Berk, Edward George, Emil Pitkin, Mikhail Traskin, Kai Zhang, and Linda Zhao. Models as approximations I: Consequences illustrated with linear regression. Statistical Science, 34(4): 523–544, 2019.  
Jing Cai, Alain De Janvry, and Elisabeth Sadoulet. Social networks and the decision to insure. American Economic Journal: Applied Economics, 7(2): 81–108, 2015.  
Brantly Callaway and Pedro HC Sant’Anna. Difference-in-differences with multiple time periods. Journal of Econometrics, 225(2):200–230, 2021.  
Sebastian Calonico, Matias D Cattaneo, and Rocio Titiunik. Robust nonparametric confidence intervals for regression-discontinuity designs. Econometrica, 82(6):2295–2326, 2014.  
Sebastian Calonico, Matias D Cattaneo, and Max H Farrell. On the effect of bias estimation on coverage accuracy in nonparametric inference. Journal of the American Statistical Association, 113(522):767–779, 2018.  
Sebastian Calonico, Matias D Cattaneo, Max H Farrell, and Rocio Titiunik. Regression discontinuity designs using covariates. Review of Economics and Statistics, 101(3):442–451, 2019.  
David Card and Alan B Krueger. Minimum wages and employment: A case study of the fast-food industry in New Jersey and Pennsylvania. The American Economic Review, 84(4):772–793, 1994.  
Pedro Carneiro, James J Heckman, and Edward J Vytlacil. Estimating marginal returns to education. American Economic Review, 101(6):2754– 2781, 2011.  
Claes M Cassel, Carl E S¨arndal, and Jan H Wretman. Some results on generalized difference estimation and generalized regression estimation for finite populations. Biometrika, 63(3):615–620, 1976.  
Juan Camilo Castillo, Dan Knoepfle, and Glen Weyl. Matching and pricing in ride hailing: Wild goose chases and how to solve them. Management Science, forthcoming, 2024.  
Gary Chamberlain. Asymptotic efficiency in estimation with conditional moment restrictions. Journal of Econometrics, 34(3):305–334, 1987.  
Gary Chamberlain. Efficiency bounds for semiparametric regression. Econometrica, 60(3):567–596, 1992.  
Olivier Chapelle and Lihong Li. An empirical evaluation of Thompson sampling. Advances in Neural Information Processing Systems, 24, 2011.  
Xiaohong Chen. Large sample sieve estimation of semi-nonparametric models. Handbook of Econometrics, 6:5549–5632, 2007.  
Ming-Yen Cheng, Jianqing Fan, and James S Marron. On automatic boundary corrections. The Annals of Statistics, 25(4):1691–1708, 1997.  
Victor Chernozhukov, Mert Demirer, Esther Duflo, and Iv´an Fern´andez-Val. Generic machine learning inference on heterogenous treatment effects in randomized experiments. arXiv preprint arXiv:1712.04802, 2017.  
Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1):1–68, 2018.  
Victor Chernozhukov, Juan Carlos Escanciano, Hidehiko Ichimura, Whitney K Newey, and James M Robins. Locally robust semiparametric estimation. Econometrica, 90(4):1501–1535, 2022a.  
Victor Chernozhukov, Whitney K Newey, and Rahul Singh. Automatic debiased machine learning of causal and structural effects. Econometrica, 90(3): 967–1027, 2022b.  
Albert Chiu, Xingchen Lan, Ziyi Liu, and Yiqing Xu. What to do (and not to do) with causal panel analysis under parallel trends: Lessons from a large reanalysis study. arXiv preprint arXiv:2309.15983, 2023.  
Eunyi Chung and Joseph P Romano. Exact and asymptotically robust permutation tests. The Annals of Statistics, 41(2):484–507, 2013.  
Peter L Cohen and Colin B Fogarty. Gaussian prepivoting for finite population causal inference. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(2):295–320, 2022.  
Bruno Cr´epon, Esther Duflo, Marc Gurgand, Roland Rathelot, and Philippe Zamora. Do labor market policies have displacement effects? evidence from a clustered randomized experiment. The Quarterly Journal of Economics, 128(2):531–580, 2013.  
Yifan Cui, Michael R Kosorok, Erik Sverdrup, Stefan Wager, and Ruoqing Zhu. Estimating heterogeneous treatment effects with right-censored data via causal survival forests. Journal of the Royal Statistical Society Series B: Statistical Methodology, 85(2):179–211, 2023.  
Cl´ement de Chaisemartin and Xavier D’Haultfoeuille. Two-way fixed effects estimators with heterogeneous treatment effects. arXiv preprint arXiv:1803.08807, 2018.  
Rajeev H Dehejia and Sadek Wahba. Causal effects in nonexperimental studies: Reevaluating the evaluation of training programs. Journal of the American Statistical Association, 94(448):1053–1062, 1999.  
Alexis Diamond and Jasjeet S Sekhon. Genetic matching for estimating causal effects: A general multivariate matching method for achieving balance in observational studies. Review of Economics and Statistics, 95(3):932–945, 2013.  
Peng Ding. A paradox from randomization-based causal inference. Statistical Science, 32(3):331–345, 2017.  
Peng Ding, Avi Feller, and Luke Miratrix. Decomposing treatment effect variation. Journal of the American Statistical Association, 114(525):304–317, 2019.  
David L Donoho. Statistical estimation and optimal recovery. The Annals of Statistics, 22(1):238–270, 1994.  
Rick Durrett. Probability: Theory and Examples. Cambridge University Press, Cambridge, United Kingdom, 5th edition, 2019.  
Dean Eckles, Nikolaos Ignatiadis, Stefan Wager, and Han Wu. Noiseinduced randomization in regression discontinuity designs. arXiv preprint arXiv:2004.09458, 2020.  
Bradley Efron. The Jackknife, the Bootstrap, and other Resampling Plans. Siam, 1982.  
Bradley Efron and David Feldman. Compliance as an explanatory variable in clinical trials. Journal of the American Statistical Association, 86(413):9–17, 1991.  
Lin Fan and Peter W Glynn. The fragility of optimized bandit algorithms. arXiv preprint arXiv:2109.13595, 2021.  
Max H Farrell. Robust inference on average treatment effects with possibly more covariates than observations. Journal of Econometrics, 189(1):1–23, 2015.  
Amy Finkelstein, Sarah Taubman, Bill Wright, Mira Bernstein, Jonathan Gruber, Joseph P Newhouse, Heidi Allen, Katherine Baicker, and the Oregon Health Study Group. The oregon health insurance experiment: evidence from the first year. The Quarterly Journal of Economics, 127(3):1057–1106, 2012.  
Ronald A Fisher. The Design of Experiments. Oliver and Boyd, Edinburgh, 1935.  
Dylan J Foster and Vasilis Syrgkanis. Orthogonal statistical learning. The Annals of Statistics, 51(3):879–908, 2023.  
Constantine E Frangakis and Donald B Rubin. Principal stratification in causal inference. Biometrics, 58(1):21–29, 2002.  
David A Freedman. On tail probabilities for martingales. The Annals of Probability, 3(1):100–118, 1975.  
Sebastian Galiani, Paul Gertler, and Ernesto Schargrodsky. Water for life: The impact of the privatization of water services on child mortality. Journal of Political Economy, 113(1):83–120, 2005.  
Dan Geiger, Thomas Verma, and Judea Pearl. Identifying independence in Bayesian networks. Networks, 20(5):507–534, 1990.  
Andrew Gelman and Guido W Imbens. Why high-order polynomials should not be used in regression discontinuity designs. Journal of Business & Economic Statistics, 37(3):447–456, 2019.  
John C Gittins. Bandit processes and dynamic allocation indices. Journal of the Royal Statistical Society: Series B (Methodological), 41(2):148–164, 1979.  
Alexander Goldenshluger and Assaf Zeevi. A linear response bandit problem. Stochastic Systems, 3(1):230–261, 2013.  
Bryan S Graham, Cristine Campos de Xavier Pinto, and Daniel Egel. Inverse probability tilting for moment condition models with missing data. The Review of Economic Studies, 79(3):1053–1079, 2012.  
The INSIGHT START Study Group. Initiation of antiretroviral therapy in early asymptomatic HIV infection. The New England Journal of Medicine, 373(9):795–807, 2015.  
Yonatan Gur, Ahmadreza Momeni, and Stefan Wager. Smoothness-adaptive contextual bandits. Operations Research, 70(6):3198–3216, 2022.  
Trygve Haavelmo. The statistical implications of a system of simultaneous equations. Econometrica, 11(1):1–12, 1943.  
Vitor Hadad, David A Hirshberg, Ruohan Zhan, Stefan Wager, and Susan Athey. Confidence intervals for policy evaluation in adaptive experiments. Proceedings of the National Academy of Sciences, 118(15), 2021.  
Jinyong Hahn. On the role of the propensity score in efficient semiparametric estimation of average treatment effects. Econometrica, 66(2):315–331, 1998.  
Jinyong Hahn, Petra Todd, and Wilbert van der Klaauw. Identification and estimation of treatment effects with a regression-discontinuity design. Econometrica, 69(1):201–209, 2001.  
P Richard Hahn, Jared S Murray, and Carlos M Carvalho. Bayesian regression tree models for causal inference: Regularization, confounding, and heterogeneous effects. Bayesian Analysis, 15(3):965–1056, 2020.  
Jens Hainmueller. Entropy balancing for causal effects: A multivariate reweighting method to produce balanced samples in observational studies. Political Analysis, 20(1):25–46, 2012.  
Jaroslav H´ajek. Local asymptotic minimax and admissibility in estimation. In Proceedings of the Sixth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Theory of Statistics, volume 6, pages 175–195. University of California Press, 1972.  
Jonathan V Hall, John J Horton, and Daniel T Knoepfle. Ride-sharing markets re-equilibrate. Technical report, National Bureau of Economic Research, 2023.  
M Elizabeth Halloran and Claudio J Struchiner. Causal inference in infectious diseases. Epidemiology, 6(2):142–151, 1995.  
Christopher Harshaw, Fredrik S¨avje, and Yitan Wang. A design-based riesz representation framework for randomized experiments. arXiv preprint arXiv:2210.08698, 2022.  
Trevor Hastie, Robert Tibshirani, and Jerome H Friedman. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer, 2 edition, 2009.  
James J Heckman. Sample selection bias as a specification error. Econometrica, 47(1):153–161, 1979.  
James J Heckman and Edward J Vytlacil. Local instrumental variables and latent variable models for identifying and bounding treatment effects. Proceedings of the National Academy of Sciences, 96(8):4730–4734, 1999.  
James J Heckman and Edward J Vytlacil. Structural equations, treatment effects, and econometric policy evaluation. Econometrica, 73(3):669–738, 2005.  
Inge S Helland. Central limit theorems for martingales with discrete or continuous time. Scandinavian Journal of Statistics, 9(2):79–94, 1982.  
Miguel A Hern´an and James M Robins. Causal Inference: What If. Chapman & Hall/CRC, Boca Raton, 2020.  
Keisuke Hirano and Jack R Porter. Asymptotics for statistical treatment rules. Econometrica, 77(5):1683–1701, 2009.  
Keisuke Hirano and Jack R Porter. Asymptotic representations for sequential decisions, adaptive experiments, and batched bandits. arXiv preprint arXiv:2302.03117, 2023.  
Keisuke Hirano, Guido W Imbens, and Geert Ridder. Efficient estimation of average treatment effects using the estimated propensity score. Econometrica, 71(4):1161–1189, 2003.  
David A Hirshberg and Stefan Wager. Augmented minimax linear estimation. The Annals of Statistics, 49(6):3206–3227, 2021.  
Paul W Holland. Statistics and causal inference. Journal of the American Statistical Association, 81(396):945–960, 1986.  
Steven R Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. Timeuniform, nonparametric, nonasymptotic confidence sequences. The Annals of Statistics, 49(2):1055–1080, 2021.  
Yichun Hu, Nathan Kallus, and Xiaojie Mao. Smooth contextual bandits: Bridging the parametric and nondifferentiable regret regimes. Operations Research, 70(6):3261–3281, 2022a.  
Yuchen Hu and Stefan Wager. Switchback experiments under geometric mixing. arXiv preprint arXiv:2209.00197, 2022.  
Yuchen Hu and Stefan Wager. Off-policy evaluation in partially observed Markov decision processes under sequential ignorability. The Annals of Statistics, 51(4):1561–1585, 2023.  
Yuchen Hu, Shuangning Li, and Stefan Wager. Average direct and indirect causal effects under interference. Biometrika, 109(4):1165–1172, 2022b.  
Michael G Hudgens and M Elizabeth Halloran. Toward causal inference with interference. Journal of the American Statistical Association, 103(482):832– 842, 2008.  
Stefano M Iacus, Gary King, and Giuseppe Porro. Causal inference without balance checking: Coarsened exact matching. Political Analysis, 20(1):1–24, 2012.  
Kosuke Imai and Michael Lingzhi Li. Experimental evaluation of individualized treatment rules. Journal of the American Statistical Association, 118(541): 242–256, 2023.  
Kosuke Imai and Marc Ratkovic. Covariate balancing propensity score. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 76(1): 243–263, 2014.  
Guido W Imbens. Nonparametric estimation of average treatment effects under exogeneity: A review. Review of Economics and Statistics, 86(1):4–29, 2004.  
Guido W Imbens. Instrumental variables: An econometrician’s perspective. Statistical Science, 29(3):323–358, 2014.  
Guido W Imbens. Potential outcome and directed acyclic graph approaches to causality: Relevance for empirical practice in economics. arXiv preprint arXiv:1907.07271, 2019.  
Guido W Imbens and Joshua D Angrist. Identification and estimation of local average treatment effects. Econometrica, 62(2):467–475, 1994.  
Guido W Imbens and Karthik Kalyanaraman. Optimal bandwidth choice for the regression discontinuity estimator. The Review of Economic Studies, 79 (3):933–959, 2012.  
Guido W Imbens and Thomas Lemieux. Regression discontinuity designs: A guide to practice. Journal of Econometrics, 142(2):615–635, 2008.  
Guido W Imbens and Charles F Manski. Confidence intervals for partially identified parameters. Econometrica, 72(6):1845–1857, 2004.  
Guido W Imbens and Donald B Rubin. Causal Inference in Statistics, Social, and Biomedical Sciences. Cambridge University Press, 2015.  
Guido W Imbens and Stefan Wager. Optimized regression discontinuity designs. Review of Economics and Statistics, 101(2):264–278, 2019.  
Hemant Ishwaran, Udaya B Kogalur, Eugene H Blackstone, and Michael S Lauer. Random survival forests. The Annals of Applied Statistics, pages 841–860, 2008.  
Adel Javanmard and Andrea Montanari. Confidence intervals and hypothesis testing for high-dimensional regression. The Journal of Machine Learning Research, 15(1):2869–2909, 2014.  
Nan Jiang and Lihong Li. Doubly robust off-policy value evaluation for reinforcement learning. In International Conference on Machine Learning, 2016.  
Nathan Kallus. Generalized optimal matching methods for causal inference. Journal of Machine Learning Research, 21(62):1–54, 2020.  
Nathan Kallus and Masatoshi Uehara. Double reinforcement learning for efficient off-policy evaluation in Markov decision processes. Journal of Machine Learning Research, 21(167):1–63, 2020.  
Nathan Kallus and Masatoshi Uehara. Efficiently breaking the curse of horizon in off-policy evaluation with double reinforcement learning. Operations Research, 70(6):3282–3302, 2022.  
Nathan Kallus and Angela Zhou. Minimax-optimal policy learning under unobserved confounding. Management Science, 67(5):2870–2890, 2021.  
Edward L Kaplan and Paul Meier. Nonparametric estimation from incomplete observations. Journal of the American Statistical Association, 53(282):457– 481, 1958.  
Maximilian Kasy and Anja Sautmann. Adaptive treatment assignment in experiments for policy choice. Econometrica, 89(1):113–132, 2021.  
Edward H Kennedy. Towards optimal doubly robust estimation of heterogeneous causal effects. Electronic Journal of Statistics, 17(2):3008–3049, 2023.  
Edward H Kennedy, Scott Lorch, and Dylan S Small. Robust causal inference with continuous instruments using the local instrumental variable curve. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 81(1):121–143, 2019.  
Edward H Kennedy, Sivaraman Balakrishnan, James M Robins, and Larry Wasserman. Minimax rates for heterogeneous causal effect estimation. The Annals of Statistics, 52(2):793–816, 2024.  
Toru Kitagawa and Aleksey Tetenov. Who should be treated? empirical welfare maximization methods for treatment choice. Econometrica, 86(2):591–616, 2018.  
Denis Kojevnikov, Vadim Marmer, and Kyungchul Song. Limit theorems for network dependent random variables. Journal of Econometrics, 222(2):882– 908, 2021.  
Michal Koles´ar and Christoph Rothe. Inference in regression discontinuity designs with a discrete running variable. American Economic Review, 108 (8):2277–2304, 2018.  
X. Kuang and Stefan Wager. Weak signal asymptotics for sequentially randomized experiments. Management Science, forthcoming, 2024.  
S¨oren R K¨unzel, Jasjeet S Sekhon, Peter J Bickel, and Bin Yu. Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 116(10):4156–4165, 2019.  
Tze Leung Lai and Herbert Robbins. Asymptotically efficient adaptive allocation rules. Advances in Applied Mathematics, 6(1):4–22, 1985.  
Robert J LaLonde. Evaluating the econometric evaluations of training programs with experimental data. American Economic Review, pages 604–620, 1986.  
David S Lee. Randomized experiments from non-random selection in US House elections. Journal of Econometrics, 142(2):675–697, 2008.  
Lihua Lei and Emmanuel J Cand\`es. Conformal inference of counterfactuals and individual treatment effects. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(5):911–938, 2021.  
Lihua Lei and Peng Ding. Regression adjustment in completely randomized experiments with a diverging number of covariates. Biometrika, 108(4):815– 828, 2021.  
Lihua Lei and Brad Ross. Estimating counterfactual matrix means with short panel data. arXiv preprint arXiv:2312.07520, 2023.  
Michael P Leung. Causal inference under approximate neighborhood interference. Econometrica, 90(1):267–293, 2022.  
Shuangning Li and Stefan Wager. Random graph asymptotics for treatment effect estimation under network interference. The Annals of Statistics, 50 (4):2334–2358, 2022.  
Xinran Li and Peng Ding. General forms of finite population central limit theorems with applications to causal inference. Journal of the American Statistical Association, 112(520):1759–1769, 2017.  
Peng Liao, Predrag Klasnja, and Susan Murphy. Off-policy estimation of longterm average outcomes with applications to mobile health. Journal of the American Statistical Association, 116(533):382–391, 2021.  
Peng Liao, Zhengling Qi, Runzhe Wan, Predrag Klasnja, and Susan A Murphy. Batch policy learning in average reward Markov decision processes. The Annals of Statistics, 50(6):3364–3387, 2022.  
Winston Lin. Agnostic notes on regression adjustments to experimental data: Reexamining Freedman’s critique. The Annals of Applied Statistics, 7(1): 295–318, 2013.  
Yueyang Liu, Benjamin Van Roy, and Kuang Xu. Nonstationary bandit learning via predictive sampling. In Proceedings of the International Conference on Artificial Intelligence and Statistics, pages 6215–6244. PMLR, 2023.  
Alex Luedtke and Antoine Chambaz. Performance guarantees for policy learning. Annales de l’Institut Henri Poincar´e, Probabilit´es et Statistiques, 56(3): 2162–2188, 2020.  
Alexander R Luedtke and Mark J van der Laan. Statistical inference for the mean outcome under a possibly non-unique optimal treatment strategy. The Annals of Statistics, 44(2):713, 2016.  
Charles F Manski. Statistical treatment rules for heterogeneous populations. Econometrica, 72(4):1221–1246, 2004.  
Charles F Manski. Identification of treatment response with social interactions. The Econometrics Journal, 16(1):S1–S23, 2013.  
Ruth Marcus, Eric Peritz, and K R Gabriel. On closed testing procedures with special reference to ordered analysis of variance. Biometrika, 63(3):655–660, 1976.  
Eric Mbakop and Max Tabord-Meehan. Model selection for treatment choice: Penalized welfare maximization. Econometrica, 89(2):825–848, 2021.  
Alec McClean, Sivaraman Balakrishnan, Edward H Kennedy, and Larry Wasserman. Double cross-fit doubly robust estimators: Beyond series regression. arXiv preprint arXiv:2403.15175, 2024.  
Mohammad Mehrabi and Stefan Wager. Off-policy evaluation in markov decision processes under weak distributional overlap. arXiv preprint arXiv:2402.08201, 2024.  
Nicolai Meinshausen, Alain Hauser, Joris M Mooij, Jonas Peters, Philip Versteeg, and Peter B¨uhlmann. Methods for causal inference from gene perturbation experiments and validation. Proceedings of the National Academy of Sciences, 113(27):7361–7368, 2016.  
Luke W Miratrix, Jasjeet S Sekhon, and Bin Yu. Adjusting treatment effect estimates by post-stratification in randomized experiments. Journal of the Royal Statistical Society Series B: Statistical Methodology, 75(2):369–396, 2013.  
Kari Lock Morgan and Donald B Rubin. Rerandomization to improve covariate balance in experiments. Annals of Statistics, 40(2):1263–1282, 2012.  
Evan Munro, X. Kuang, and Stefan Wager. Treatment effects in market equilibrium. arXiv preprint arXiv:2109.11647, 2021.  
Susan A Murphy. A generalization error for Q-learning. Journal of Machine Learning Research, 6(Jul):1073–1097, 2005.  
Sahand N Negahban, Pradeep Ravikumar, Martin J Wainwright, and Bin Yu. A unified framework for high-dimensional analysis of M-estimators with decomposable regularizers. Statistical Science, 27(4):538–557, 2012.  
Whitney K Newey. Efficient instrumental variables estimation of nonlinear models. Econometrica, 58(4):809–837, 1990.  
Whitney K Newey. The asymptotic variance of semiparametric estimators. Econometrica, 62(6):1349–1382, 1994.  
Whitney K Newey and James L Powell. Instrumental variable estimation of nonparametric models. Econometrica, 71(5):1565–1578, 2003.  
Whitney K Newey and James R Robins. Cross-fitting and fast remainder rates for semiparametric estimation. arXiv preprint arXiv:1801.09138, 2018.  
Jersey Neyman. Sur les applications de la th´eorie des probabilit´es aux experiences agricoles: Essai des principes. Roczniki Nauk Rolniczych, 10:1–51, 1923.  
Xinkun Nie and Stefan Wager. Quasi-oracle estimation of heterogeneous treatment effects. Biometrika, 108(2):299–319, 2021.  
Xinkun Nie, Xiaoying Tian, Jonathan Taylor, and James Zou. Why adaptively collected data have negative bias and how to correct for it. In International Conference on Artificial Intelligence and Statistics, pages 1261–1269. PMLR, 2018.  
Claudia Noack and Christoph Rothe. Bias-aware inference in fuzzy regression discontinuity designs. Econometrica, forthcoming, 2024.  
Elizabeth L Ogburn and Tyler J VanderWeele. Vaccines, contagion, and social networks. Annals of Applied Statistics, 11(2):919–948, 2017.  
Elizabeth L Ogburn, Oleg Sofrygin, Ivan Diaz, and Mark J Van der Laan. Causal inference for social network data. Journal of the American Statistical Association, 119(545):597–611, 2024.  
Judea Pearl. Causal diagrams for empirical research. Biometrika, 82(4):669– 688, 1995.  
Judea Pearl. Causality. Cambridge University Press, 2009.  
Judea Pearl and Dana Mackenzie. The Book of Why: The New Science of Cause and Effect. Basic Books, 2018.  
Vianney Perchet and Philippe Rigollet. The multi-armed bandit problem with covariates. The Annals of Statistics, 41(2):693–721, 2013.  
Jonas Peters, Peter B¨uhlmann, and Nicolai Meinshausen. Causal inference by using invariant prediction: identification and confidence intervals. Journal of the Royal Statistical Society Series B: Statistical Methodology, 78(5):947– 1012, 2016.  
Chao Qin and Daniel Russo. Adaptive experimentation in the presence of exogenous nonstationary variation. arXiv preprint arXiv:2202.09036, 2022.  
Thomas S Richardson and Andrea Rotnitzky. Causal etiology of the research of James M. Robins. Statistical Science, 29(4):459–484, 2014.  
Herbert Robbins. Statistical methods related to the law of the iterated logarithm. The Annals of Mathematical Statistics, 41(5):1397–1409, 1970.  
James Robins, Mariela Sued, Quanhong Lei-Gomez, and Andrea Rotnitzky. Comment: Performance of double-robust estimators when “inverse probability” weights are highly variable. Statistical Science, 22(4):544–559, 2007.  
James M Robins. A new approach to causal inference in mortality studies with a sustained exposure period: Application to control of the healthy worker survivor effect. Mathematical Modelling, 7(9-12):1393–1512, 1986.  
James M Robins. Correcting for non-compliance in randomized trials using structural nested mean models. Communications in Statistics: Theory and Methods, 23(8):2379–2412, 1994.  
James M Robins. Association, causation, and marginal structural models. Synthese, 121(1/2):151–179, 1999.  
James M Robins. Optimal structural nested models for optimal sequential decisions. In Proceedings of the second seattle Symposium in Biostatistics, pages 189–326. Springer, 2004.  
James M Robins and Thomas S Richardson. Alternative graphical causal models and the identification of direct effects. Causality and Psychopathology: Finding the Determinants of Disorders and their Cures, pages 103–158, 2010.  
James M Robins and Andrea Rotnitzky. Semiparametric efficiency in multivariate regression models with missing data. Journal of the American Statistical Association, 90(429):122–129, 1995.  
James M Robins, Andrea Rotnitzky, and Lue Ping Zhao. Estimation of regression coefficients when some regressors are not always observed. Journal of the American Statistical Association, 89(427):846–866, 1994.  
James M Robins, Lingling Li, Rajarshi Mukherjee, Eric Tchetgen Tchetgen, and Aad van der Vaart. Minimax estimation of a functional on a structured high-dimensional model. The Annals of Statistics, 45(5):1951–1987, 2017.  
Peter M Robinson. Root-n-consistent semiparametric regression. Econometrica, 56(4):931–954, 1988.  
Todd Rogers and Avi Feller. Reducing student absences at scale by targeting parents’ misbeliefs. Nature Human Behaviour, 2(5):335–342, 2018.  
Joseph P Romano. On the behavior of randomization tests without a group invariance assumption. Journal of the American Statistical Association, 85 (411):686–692, 1990.  
Paul R Rosenbaum and Donald B Rubin. The central role of the propensity score in observational studies for causal effects. Biometrika, 70(1):41–55, 1983.  
Paul R Rosenbaum and Donald B Rubin. Reducing bias in observational studies using subclassification on the propensity score. Journal of the American Statistical Association, 79(387):516–524, 1984.  
Eric L Ross, Robert M Bossarte, Steven K Dobscha, Sarah M Gildea, Irving Hwang, Chris J Kennedy, Howard Liu, Alex Luedtke, Brian P Marx, Matthew K Nock, et al. Estimated average treatment effect of psychiatric hospitalization in patients with suicidal behaviors: a precision treatment analysis. JAMA psychiatry, 81(2):135–143, 2024.  
Andrew D Roy. Some thoughts on the distribution of earnings. Oxford Economic Papers, 3(2):135–146, 1951.  
Daniel Rubin and Mark J van der Laan. A doubly robust censoring unbiased transformation. The International Journal of Biostatistics, 3(1), 2007.  
Donald B Rubin. Estimating causal effects of treatments in randomized and nonrandomized studies. Journal of Educational Psychology, 66(5):688, 1974.  
Daniel Russo. Simple Bayesian algorithms for best-arm identification. Operations Research, 68(6):1625–1647, 2020.  
Daniel Russo and Benjamin Van Roy. Learning to optimize via informationdirected sampling. Operations Research, 66(1):230–252, 2018.  
Daniel J Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, and Zheng Wen. A tutorial on Thompson sampling. Foundations and Trends in Machine Learning, 11(1):1–96, 2018.  
Jerome Sacks and Donald Ylvisaker. Linear estimation for approximately linear models. The Annals of Statistics, 6(5):1122–1137, 1978.  
Fredrik S¨avje. Causal inference with misspecified exposure mappings: Separating definitions and assumptions. Biometrika, 111(1):1–15, 2024.  
Fredrik S¨avje, Peter Aronow, and Michael Hudgens. Average treatment effects in the presence of unknown interference. The Annals of Statistics, 49(2):673, 2021.  
Daniel O Scharfstein, Andrea Rotnitzky, and James M Robins. Adjusting for nonignorable drop-out using semiparametric nonresponse models. Journal of the American Statistical Association, 94(448):1096–1120, 1999.  
Eric M Schwartz, Eric T Bradlow, and Peter S Fader. Customer acquisition via display advertising using multi-armed bandit experiments. Marketing Science, 36(4):500–522, 2017.  
Dennis Shen, Peng Ding, Jasjeet Sekhon, and Bin Yu. Same root different leaves: Time series and cross-sectional methods in panel data. Econometrica, 91(6):2125–2154, 2023.  
Michael E Sobel. What do randomized studies of housing mobility demonstrate? causal inference in the face of interference. Journal of the American Statistical Association, 101(476):1398–1407, 2006.  
Peter Spirtes, Clark N Glymour, and Richard Scheines. Causation, Prediction, and Search. Springer-Verlag, New York, 1993.  
Charles M Stein. Estimation of the mean of a multivariate normal distribution. The Annals of Statistics, 9(6):1135–1151, 1981.  
Charles J Stone. Consistent nonparametric regression. The Annals of Statistics, 5(4):595–620, 1977.  
J¨org Stoye. Minimax regret treatment choice with finite samples. Journal of Econometrics, 151(1):70–81, 2009.  
Hao Sun, Evan Munro, Georgy Kalashnov, Shuyang Du, and Stefan Wager. Treatment allocation under uncertain costs. arXiv preprint arXiv:2103.11066, 2021.  
Liyang Sun and Sarah Abraham. Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. Journal of Econometrics, 225 (2):175–199, 2021.  
Richard S Sutton. Learning to predict by the methods of temporal differences. Machine Learning, 3:9–44, 1988.  
Richard S Sutton and Andrew G Barto. Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA, 2nd edition, 2018.  
Erik Sverdrup, Han Wu, Susan Athey, and Stefan Wager. Qini curves for multi-armed treatment rules. arXiv preprint arXiv:2306.11979, 2023.  
Adith Swaminathan and Thorsten Joachims. Batch learning from logged bandit feedback through counterfactual risk minimization. The Journal of Machine Learning Research, 16(1):1731–1755, 2015.  
Zhiqiang Tan. Model-assisted inference for treatment effects using regularized calibrated estimation with high-dimensional data. The Annals of Statistics, 48(2):811–837, 2020.  
Donald L Thistlethwaite and Donald T Campbell. Regression-discontinuity analysis: An alternative to the ex post facto experiment. Journal of Educational Psychology, 51(6):309–317, 1960.  
Philip Thomas and Emma Brunskill. Data-efficient off-policy policy evaluation for reinforcement learning. In International Conference on Machine Learning, pages 2139–2148, 2016.  
William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3/4):285–294, 1933.  
Lu Tian, Ash A Alizadeh, Andrew J Gentles, and Robert Tibshirani. A simple method for estimating interactions between a treatment and a large number of covariates. Journal of the American Statistical Association, 109(508): 1517–1532, 2014.  
Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology, 58(1):267–288, 1996.  
Anastasios A Tsiatis. Semiparametric theory and missing data. Springer, New York, 2006.  
John N Tsitsiklis and Benjamin Van Roy. An analysis of temporal-difference learning with function approximation. IEEE Transactions on Automatic Control, 42(5):674–690, 1997.  
Masatoshi Uehara, Jiawei Huang, and Nan Jiang. Minimax weight and qfunction learning for off-policy evaluation. In Hal Daum´e III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 9659–9668. PMLR, 2020.  
Mark J van der Laan and James M Robins. Unified methods for censored longitudinal data and causality. Springer, New York, 2003.  
Mark J van der Laan and Sherri Rose. Targeted learning: Causal inference for observational and experimental data. Springer Science & Business Media, 2011.  
Mark J van der Laan and Daniel Rubin. Targeted maximum likelihood learning. The International Journal of Biostatistics, 2(1), 2006.  
Aad W Van der Vaart. Asymptotic Statistics. Cambridge University Press, 1998.  
Davide Viviano. Policy targeting under network interference. Review of Economic Studies, forthcoming, 2024.  
Stefan Wager. On regression tables for policy learning: Comment on a paper by Jiang, Song, Li and Zeng. Statistica Sinica, 29(4):1678–1685, 2019.  
Stefan Wager, Wenfei Du, Jonathan Taylor, and Robert J Tibshirani. Highdimensional regression adjustments in randomized experiments. Proceedings of the National Academy of Sciences, 113(45):12673–12678, 2016.  
Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8: 279–292, 1992.  
Halbert White. A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. Econometrica, 48(4):817–838, 1980.  
Halbert White. Asymptotic Theory for Econometricians. Economic Theory, Econometrics, and Mathematical Economics. Academic Press, Orlando, Florida, 1984.  
Jeffrey M Wooldridge. Econometric Analysis of Cross Section and Panel Data. MIT press, 2010.  
Sewall Wright. The method of path coefficients. The Annals of Mathematical Statistics, 5(3):161–215, 1934.  
Han Wu and Stefan Wager. Thompson sampling with unrestricted delays. In Proceedings of the 23rd ACM Conference on Economics and Computation, pages 937–955, 2022.  
Yiqing Xu. Generalized synthetic control method: Causal inference with interactive fixed effects models. Political Analysis, 25(1):57–76, 2017.  
Steve Yadlowsky, Scott Fleming, Nigam Shah, Emma Brunskill, and Stefan Wager. Evaluating treatment prioritization rules via rank-weighted average treatment effects. arXiv preprint arXiv:2111.07966, 2021.  
Baqun Zhang, Anastasios A Tsiatis, Eric B Laber, and Marie Davidian. Robust estimation of optimal dynamic treatment regimes for sequential treatment decisions. Biometrika, 100(3):681–694, 2013.  
Cun-Hui Zhang and Stephanie S Zhang. Confidence intervals for low dimensional parameters in high dimensional linear models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 76(1):217–242, 2014.  
Kelly Zhang, Lucas Janson, and Susan Murphy. Inference for batched bandits. Advances in Neural Information Processing Systems, 33:9818–9829, 2020.  
Qingyuan Zhao. Covariate balancing propensity score by tailored loss functions. The Annals of Statistics, 47(2):965–993, 2019.  
Qingyuan Zhao, Dylan S Small, and Ashkan Ertefaie. Selective inference for effect modification via the lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(2):382–413, 2022.  
Yingqi Zhao, Donglin Zeng, A John Rush, and Michael R Kosorok. Estimating individualized treatment rules using outcome weighted learning. Journal of the American Statistical Association, 107(499):1106–1118, 2012.  
Zhengyuan Zhou, Susan Athey, and Stefan Wager. Offline multi-action policy learning: Generalization and optimization. Operations Research, 71(1):148– 183, 2023.

Jos´e R Zubizarreta. Using mixed integer programming for matching in an observational study of kidney failure after surgery. Journal of the American Statistical Association, 107(500):1360–1371, 2012. Jos´e R Zubizarreta. Stable weights that balance covariates for estimation with incomplete outcome data. Journal of the American Statistical Association, 110(511):910–922, 2015.

<!-- footnote -->

- Working with γ-discounted rewards rather than long-run average rewards results in similar but different Bellman equations.

<!-- footnote end -->

<!-- footnote -->

- Following the nomenclature in Chapter 3, we are here focused on weak double robustness.

<!-- footnote end -->

<!-- footnote -->

- You also do not need to elaborate on how to construct the estimates $\hat { \alpha } ( \cdot ) , \hat { \tau } ( \cdot )$ , etc.

<!-- footnote end -->

<!-- footnote -->

- We will not investigate how to estimate these quantities here; however, we note that one popular way to estimate unconditional survival functions is via the Kaplan–Meier estimator [Kaplan and Meier, 1958]; and this method can be made conditional on covariates $X _ { i }$ via, $\mathrm { e . g . }$ , the random survival forest construction [Ishwaran et al., 2008].
- There is also an analogous continuous-time AIPCW estimator; see, e.g., Rubin and van der Laan [2007] and Cui et al. [2023]. To see the connection between the expression in $\hat { \theta } _ { A I P C W }$ and the standard continuous-time formula, it is helpful to first apply the Abel transformation to the sum in (16.31).

<!-- footnote end -->
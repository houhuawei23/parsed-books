# Chapter 3 Doubly Robust Methods

Inverse-propensity weighting (IPW) is a simple and transparent approach to average treatment effect estimation under unconfoundedness. However, as seen in the previous chapter, the large-sample properties of IPW are not particularly good in general, and the way estimation error in the propensity scores affects accuracy of IPW is complex. Our goal here is to move beyond the limitations of IPW and to discuss doubly robust methods, which provide a general recipe for building robust and asymptotically optimal treatment effect estimators under unconfoundedness, and enable us to rigorously and flexibly handle estimation error in the propensity score.15

Throughout this chapter, we will seek to estimate the average treatment effect $\tau = \mathbb { E } \left[ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \right]$ under the following statistical setting:

Basic setting: SUTVA, unconfoundedness and strong overlap There is a distribution P that generates a stream of tuples $\{ X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , \bar { W } _ { i } \} \stackrel { \mathrm { i i d } } { \sim } P$ taking values in $\mathcal { X } \times \mathbb { R } \times \mathbb { R } \times \{ 0 , 1 \}$ . We get to observe $( X _ { i } , Y _ { i } , W _ { i } )$ where $Y _ { i } = Y _ { i } ( W _ { i } )$ (SUTVA). We are not necessarily in a randomized controlled trial; however, we have unconfoundedness, i.e., treatment assignment is as good as random conditionally on the features $X _ { i } { \mathrm { : } }$

$$
\left\{Y _ {i} (0), Y _ {i} (1) \right\} \perp W _ {i} \mid X _ {i}, \tag {3.1}
$$

Potential outcomes have bounded second moments, $\mathbb { E } \left[ Y _ { i } ^ { 2 } ( w ) \right] < \infty$ . Strong overlap holds, i.e., for some $\eta > 0$ ,

$$
\eta \leq e (x) \leq 1 - e (x) \quad \text { for   all } \quad x \in \mathcal {X}. \tag {3.2}
$$

We write $e ( x ) = \mathbb { P } \left[ W _ { i } = 1 \big | X _ { i } = x \right]$ for the propensity score, and also use notation $\mu _ { ( w ) } ( x ) = \bar { \mathbb { E } } \left[ Y _ { i } ( w ) \big | X _ { i } = x \right]$ and $\sigma _ { ( w ) } ^ { 2 } ( x ) = \mathrm { V a r } \left[ Y _ { i } ( w ) \big | X _ { i } = x \right]$ .

Two characterizations of the ATE In the previous chapter, we saw that the ATE can be characterized via IPW:

$$
\tau = \mathbb {E} \left[ \hat {\tau} _ {I P W} ^ {*} \right], \quad \hat {\tau} _ {I P W} ^ {*} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - e (X _ {i})}\right). \tag {3.3}
$$

However, $\tau$ can also be characterized in terms of the conditional response surfaces $\mu _ { ( w ) } ( x )$ : Under unconfoundedness (3.1),

$$
\tau (x) := \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \mid X _ {i} = x \right]
$$

$$
= \mathbb {E} \left[ Y _ {i} (1) \mid X _ {i} = x \right] - \mathbb {E} \left[ Y _ {i} (0) \mid X _ {i} = x \right]
$$

$$
= \mathbb {E} \left[ Y _ {i} (1) \mid X _ {i} = x, W _ {i} = 1 \right] - \mathbb {E} \left[ Y _ {i} (0) \mid X _ {i} = x, W _ {i} = 0 \right] \quad (\text { unconf })
$$

$$
= \mathbb {E} \left[ Y _ {i} \mid X _ {i} = x, W _ {i} = 1 \right] - \mathbb {E} \left[ Y _ {i} \mid X _ {i} = x, W _ {i} = 0 \right] \quad (\text {SUTVA})
$$

$$
= \mu_ {(1)} (x) - \mu_ {(0)} (x),
$$

and so $\tau = \mathbb { E } \left[ \mu _ { ( 1 ) } ( X _ { i } ) - \mu _ { ( 0 ) } ( X _ { i } ) \right]$ . Thus there also exists a simple and consistent (but not necessarily optimal) non-parametric regression estimator for τ : First estimate $\mu _ { ( 0 ) } ( x )$ and $\mu _ { ( 1 ) } ( x )$ non-parametrically, and then set $\begin{array} { r } { \hat { \tau } _ { R E G } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } \bigl ( \hat { \mu } _ { ( 1 ) } ( X _ { i } ) - \hat { \mu } _ { ( 0 ) } ( X _ { i } ) \bigr ) } \end{array}$ .

Augmented IPW Given that the average treatment effect can be estimated in two different ways, i.e., by first non-parametrically estimating $e ( x )$ or by first estimating $\mu _ { ( 0 ) } ( x )$ and $\mu _ { ( 1 ) } ( x )$ , it is natural to ask whether it is possible to combine both strategies. This turns out to be a very good idea, and yields the augmented IPW (AIPW) estimator of Robins, Rotnitzky, and Zhao [1994]:

$$
\hat {\tau} _ {A I P W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\mu} _ {(1)} (X _ {i}) - \hat {\mu} _ {(0)} (X _ {i}) \right. \tag {3.4}
$$

$$
\left. + W _ {i} \frac {Y _ {i} - \hat {\mu} _ {(1)} (X _ {i})}{\hat {e} (X _ {i})} - (1 - W _ {i}) \frac {Y _ {i} - \hat {\mu} _ {(0)} (X _ {i})}{1 - \hat {e} (X _ {i})}\right).
$$

Qualitatively, AIPW can be seen as first making a best effort attempt at τ by estimating $\mu _ { ( 0 ) } ( x )$ and $\mu _ { ( 1 ) } ( x )$ ; then, it deals with any biases of the $\hat { \mu } _ { ( w ) } ( x )$ by applying IPW to the regression residuals. Statistically, it turns out that AIPW not only inherits robustness properties from both the regression and IPW estimators—it improves on both by (in a sense made rigorous below) using IPW to mitigate errors in the regression estimator and vice-versa.

Weak double robustness A first, simple-to-understand property of AIPW is the following “weak” double robustness property:16 AIPW is consistent if either the $\hat { \mu } _ { ( w ) } ( x )$ are consistent or $\hat { e } ( x )$ is consistent. To see this, first consider the case where $\hat { \mu } _ { ( w ) } ( x )$ is consistent, i.e., $\hat { \mu } _ { ( w ) } ( x ) \approx \mu _ { ( w ) } ( x )$ . Then,

$$
\begin{array}{l} \hat {\tau} _ {A I P W} = \underbrace {\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\mu} _ {(1)} (X _ {i}) - \hat {\mu} _ {(0)} (X _ {i})\right)} _ {\text {the regression estimator}} \\ + \underbrace {\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i}}{\hat {e} (X _ {i})} \left(Y _ {i} - \hat {\mu} _ {(1)} (X _ {i})\right) - \frac {1 - W _ {i}}{1 - \hat {e} (X _ {i})} \left(Y _ {i} - \hat {\mu} _ {(0)} (X _ {i})\right)\right)} _ {\approx \text {mean - zero noise}}, \\ \end{array}
$$

because $\mathbb { E } \left[ Y _ { i } - \hat { \mu } _ { ( W _ { i } ) } ( X _ { i } ) \big | X _ { i } , W _ { i } \right] \approx 0$ under unconfoundedness. Thus even if we use inconsistent propensity score weights $1 / \hat { e } ( X _ { i } )$ and $1 / ( 1 - \hat { e } ( X _ { i } ) )$ , they are multiplied by roughly mean-zero error terms and so asymptotically they do not bias the estimator, and $\hat { \tau } _ { A I P W }$ remains consistent.

Conversely, now suppose that $\hat { e } ( x )$ is consistent, $\mathrm { i . e . , } \hat { e } ( x ) \approx e ( x )$ . Then,

$$
\begin{array}{l} \hat {\tau} _ {A I P W} = \underbrace {\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {W _ {i} Y _ {i}}{\hat {e} (X _ {i})} - \frac {(1 - W _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})}\right)} _ {\text {the IPW estimator}} \\ + \underbrace {\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\mu} _ {(1)} (X _ {i}) \left(1 - \frac {W _ {i}}{\hat {e} (X _ {i})}\right) - \hat {\mu} _ {(0)} (X _ {i}) \left(1 - \frac {1 - W _ {i}}{1 - \hat {e} (X _ {i})}\right)\right)} _ {\approx \text {mean - zero noise}}, \\ \end{array}
$$

because E $\left[ 1 - W _ { i } / \hat { e } ( X _ { i } ) \vert X _ { i } \right] \approx 0$ . Thus, even if we use inconsistent regression adjustments $\hat { \mu } _ { ( w ) } ( X _ { i } )$ , they will be multiplied by roughly mean-zero noise terms that asymptotically cancel their contribution. Thus $\hat { \tau } _ { A I P W }$ inherits the consistency of ${ \hat { \tau } } _ { I P W }$ under unconfoundedness.

That being said, although the (weak) double robustness of AIPW is is a nice property to have, its importance should not be overstated. Weak double robustness only guarantees consistency of $\hat { \tau } _ { A I P W }$ , whereas in most treatment effect estimation applications we also care about rates of convergence and confidence intervals. Furthermore, one could also argue that, in a modern setting, one should expect practitioners to use appropriate non-parametric estimators for both $\mu _ { ( w ) } ( x )$ and $e ( x )$ that are consistent for each. In this case both $\hat { \tau } _ { R E G }$ and ${ \hat { \tau } } _ { I P W }$ would already be consistent on their own, and so the above weak double robustness statement (i.e., consistency of ${ \hat { \tau } } _ { A I P W } )$ doesn’t add anything.

Strong double robustness There is also a much more interesting and useful class of “strong” double robustness results for AIPW that quantify the weaker consistency statement given above. At a high level, strong double robustness is a claim that results of the following type exist: If we use estimators $\hat { \mu } _ { ( w ) } ( x )$ and $\hat { e } ( x )$ that are both consistent with root-mean squared error (RMSE) decaying faster than $n ^ { - \alpha \mu }$ and $n ^ { - \alpha _ { e } }$ respectively, and if furthermore $\alpha _ { \mu } + \alpha _ { e } \ge 1 / 2$ , then

$$
\sqrt {n} \left(\hat {\tau} _ {A I P W} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {A I P W}\right),
$$

$$
V _ {A I P W} = \operatorname{Var} \left[ \tau (X _ {i}) \right] + \mathbb {E} \left[ \frac {\sigma_ {0} ^ {2} (X _ {i})}{1 - e (X _ {i})} \right] + \mathbb {E} \left[ \frac {\sigma_ {1} ^ {2} (X _ {i})}{e (X _ {i})} \right]. \tag {3.5}
$$

The reason this meta-result holds is that, in general, if the RMSE of $\hat { \mu } _ { ( w ) } ( x )$ decays faster than $n ^ { - \alpha _ { \mu } }$ and the RMSE of $\hat { e } ( x )$ decays faster than $n ^ { - \alpha _ { e } }$ , then the bias of AIPW decays faster than $n ^ { - ( \alpha _ { \mu } + \alpha _ { e } ) }$ ; and, in particular, if $\alpha _ { \mu } + \alpha _ { e } \ge 1 / 2$ then the bias is lower-order on the $1 / \sqrt { n } \mathrm { - s c a l e }$ . What’s remarkable about this result is that, under the same conditions, the bias of the regression estimator would in general only be bounded to order $n ^ { - \alpha _ { \mu } }$ and that of IPW to order $n ^ { - \alpha _ { e } }$ ; and so the AIPW construction succeeds in making bias substantially smaller than what either the regression or IPW estimators could achieve on their own.17

The statement given above is not a theorem—rather it’s a meta-result, and a blueprint for many types of results that hold under further technical assumptions. Below, we will discuss one specific way of constructing AIPW estimators, coined as double machine learning by Chernozhukov et al. [2018], and establish conditions under which it satisfies (3.5). Note that double machine learning is not the only way to get results of this type; and in fact results that are stronger than (3.5) can be obtained in some specialized settings. Thus, our presentation below should be seen as a first step—and not the end point—in understanding and leveraging strong double robustness of AIPW.

## 3.1 Double machine learning

Our study of strong double robustness for AIPW starts by considering the behavior of an “oracle” AIPW estimator that is constructed in terms of true (rather than estimated) values of the conditional regression surfaces and thepropensity score:

$$
\begin{array}{l} \hat {\tau} _ {A I P W} ^ {*} = \frac {1}{n} \sum_ {i = 1} ^ {n} \Gamma_ {i} \\ \Gamma_ {i} = \mu_ {(1)} (X _ {i}) - \mu_ {(0)} (X _ {i}) + W _ {i} \frac {Y _ {i} - \mu_ {(1)} (X _ {i})}{e (X _ {i})} - (1 - W _ {i}) \frac {Y _ {i} - \mu_ {(0)} (X _ {i})}{1 - e (X _ {i})}. \end{array} \tag {3.6}
$$

Proposition 3.1. Under the basic setting with SUTVA, unconfoundedness and strong overlap given at the beginning of this chapter, the oracle AIPW estimator has the limit distribution given in (3.5), i.e.,

$$
\sqrt {n} \left(\hat {\tau} _ {A I P W} ^ {*} - \tau\right) \Rightarrow \mathcal {N} \left(0, V _ {A I P W}\right). \tag {3.7}
$$

Proof. The fact that the oracle AIPW estimator is unbiased follows from the discussions used to establish weak double robustness of AIPW. Furthermore, the oracle estimator is an average of IID terms, so the standard central limit theorem immediately implies that $\sqrt { n } \left( \widehat { \tau } _ { A I P W } ^ { * } - \tau \right) \Rightarrow \mathcal { N } \left( 0 , \mathrm { V a r } \left[ \Gamma _ { i } \right] \right)$ . Finally, under unconfoundedness, we can check that

$$
\begin{array}{l} \operatorname{Var} \left[ \Gamma_ {i} \right] = \operatorname{Var} \left[ \mu_ {(1)} (X _ {i}) - \mu_ {(0)} (X _ {i}) \right] + \mathbb {E} \left[ \left(W _ {i} \frac {Y _ {i} - \mu_ {(1)} (X _ {i})}{e (X _ {i})}\right) ^ {2} \right] \tag {3.8} \\ + \mathbb {E} \left[ \left((1 - W _ {i}) \frac {Y _ {i} - \mu_ {(0)} (X _ {i})}{1 - e (X _ {i})}\right) ^ {2} \right], \\ \end{array}
$$

which matches the expression for $V _ { A I P W }$ given in (3.5). Notice in particular that, by the overlap and bounded-second-moment assumptions in our basic setting, all terms in (3.8) are finite. □

Given this result, establishing (3.5) reduces to showing that, provided $\hat { \mu } _ { ( w ) } ( \cdot )$ and $\hat { e } ( \cdot )$ converge fast enough,

$$
\sqrt {n} \left(\hat {\tau} _ {A I P W} - \hat {\tau} _ {A I P W} ^ {*}\right)\rightarrow_ {p} 0, \tag {3.9}
$$

i.e., the feasible AIPW estimator is asymptotically equivalent to the oracle. The fact that proving results of the type (3.9) is possible under reasonable assumptions is not to be taken for granted, and is a consequence of AIPW having a strong double robustness property. Other estimators we’ve discussed, such as the IPW and regression adjustment estimators, do not in general satisfy this type of oracle equivalence property.

Cross-fitting In order to establish the oracle equivalence result (3.9), it is helpful to consider the following minor algorithmic modification of AIPW using a technique called cross-fitting. At a high level, cross-fitting uses cross-fold estimation to avoid bias due to overfitting; the motivation behind doing so is closely related to the reason why we often use cross-validation when estimating the predictive accuracy of an estimator.

Cross-fitting first splits the data (at random) into two halves $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ , and then uses an estimator18

$$
\hat {\tau} _ {A I P W} = \frac {| \mathcal {I} _ {1} |}{n} \hat {\tau} ^ {\mathcal {I} _ {1}} + \frac {| \mathcal {I} _ {2} |}{n} \hat {\tau} ^ {\mathcal {I} _ {2}}, \quad \hat {\tau} ^ {\mathcal {I} _ {1}} = \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \hat {\mu} _ {(0)} ^ {\mathcal {I} _ {2}} (X _ {i}) \right. \tag {3.10}
$$

$$
\left. + W _ {i} \frac {Y _ {i} - \hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i})}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - (1 - W _ {i}) \frac {Y _ {i} - \hat {\mu} _ {(0)} ^ {\mathcal {I} _ {2}} (X _ {i})}{1 - \hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})}\right),
$$

where the $\hat { \mu } _ { ( w ) } ^ { \mathcal { Z } _ { 2 } } ( \cdot )$ and $\hat { e } ^ { \pm \tau _ { 2 } } ( \cdot )$ are estimates of $\mu _ { ( w ) } ( \cdot )$ and $e ( \cdot )$ obtained using only the half-sample $\mathcal { T } _ { 2 }$ , and $\hat { \tau } ^ { \mathcal { I } _ { 2 } }$ is defined analogously (with the roles of $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ swapped). In other words, $\hat { \tau } ^ { \mathcal { I } _ { 1 } }$ is a treatment effect estimator on $\mathcal { T } _ { 1 }$ that uses $\mathcal { T } _ { 2 }$ to estimate its non-parametric components, and vice-versa.

What cross-fitting buys us is that, e.g., if $i \in \mathcal { Z } _ { 1 }$ and $W _ { i } = 0$ , then $Y _ { i } -$ $\hat { \mu } _ { ( 0 ) } ^ { \mathcal { T } _ { 2 } } ( X _ { i } )$ via overfitting. As seen below, by creating such honest residuals, cross-fitting enables us to establish results of the type (3.9) without needing to make detailed assumptions about the algorithms used to estimate $\hat { \mu } _ { ( w ) } ( x )$ and ${ \hat { e } } ( x )$ .

Theorem 3.2. Given our basic setting with SUTVA, unconfoundedness and strong overlap, suppose that we construct $\hat { \tau } _ { A I P W }$ using cross-fitting with estimators satisfying, for $w \in \{ 0 , 1 \}$ and also with the roles of $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ swapped,

$$
\begin{array}{l} n ^ {- 2 \alpha_ {\mu}} \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(w)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(w)} (X _ {i})\right) ^ {2} \rightarrow_ {p} 0, \\ n ^ {- 2 \alpha_ {e}} \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\frac {1}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - \frac {1}{e (X _ {i})}\right) ^ {2} \rightarrow_ {p} 0, \tag {3.11} \\ \end{array}
$$

for some constants with $\alpha _ { \mu } , \alpha _ { e } \geq 0$ and $\alpha _ { \mu } + \alpha _ { e } \ge 1 / 2$ . Then (3.9) and thus also (3.5) hold.

Proof. Note that, because $\hat { \tau } _ { A I P W } ^ { * }$ doesn’t rely on estimated quantities and so is unaffected by cross-fitting, we can write the oracle AIPW estimator as

$$
\hat {\tau} _ {A I P W} ^ {*} = \frac {| \mathcal {I} _ {1} |}{n} \hat {\tau} ^ {\mathcal {I} _ {1}, *} + \frac {| \mathcal {I} _ {2} |}{n} \hat {\tau} ^ {\mathcal {I} _ {2}, *}
$$

analogously to (3.10). Moreover, we can decompose $\hat { \tau } ^ { \mathcal { I } _ { 1 } }$ itself as

$$
\hat {\tau} ^ {\mathcal {I} _ {1}} = \hat {m} _ {(1)} ^ {\mathcal {I} _ {1}} - \hat {m} _ {(0)} ^ {\mathcal {I} _ {1}},
$$

$$
\hat {m} _ {(1)} ^ {\mathcal {I} _ {1}} = \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) + W _ {i} \frac {Y _ {i} - \hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i})}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})}\right), \tag {3.12}
$$

$\hat { m } _ { ( 0 ) } ^ { { \cal T } _ { 1 } , * }$ I1,∗ $\hat { m } _ { ( 1 ) } ^ { { \ Z _ { 1 } } , * }$ analogously. Given this setup, in order to verify (3.9), it suffices to show that

$$
\sqrt {n} \left(\hat {m} _ {(1)} ^ {\mathcal {I} _ {1}} - \hat {m} _ {(1)} ^ {\mathcal {I} _ {1}, *}\right)\rightarrow_ {p} 0. \tag {3.13}
$$

The proof can then be completed by carrying out the same argument for different folds and treatment statuses.

To this end, we decompose the error term in (3.13) as follows:

$$
\begin{array}{l} \hat {m} _ {(1)} ^ {\mathcal {I} _ {1}} - \hat {m} _ {(1)} ^ {\mathcal {I} _ {1}, *} \\ = \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) + W _ {i} \frac {Y _ {i} - \hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i})}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - \mu_ {(1)} (X _ {i}) - W _ {i} \frac {Y _ {i} - \mu_ {(1)} (X _ {i})}{e (X _ {i})}\right) \\ = \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) \left(1 - \frac {W _ {i}}{e (X _ {i})}\right)\right) \\ + \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} W _ {i} \left(\left(Y _ {i} - \mu_ {(1)} (X _ {i})\right) \left(\frac {1}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - \frac {1}{e (X _ {i})}\right)\right) \\ - \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} W _ {i} \left(\left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) \left(\frac {1}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - \frac {1}{e (X _ {i})}\right)\right) \\ \end{array}
$$

We can then verify that these terms are small for different reasons.

For the first term, we intricately use the fact that, thanks to our cross-fitting construction, ˆµI2(w) $\hat { \mu } _ { ( w ) } ^ { \mathcal { L } _ { 2 } }$ can effectively be treated as deterministic when considering terms on $\mathcal { T } _ { 1 }$ . We first observe that, conditionally on $\mathcal { T } _ { 2 }$ and the observed covariate values, this term can be treated as average of independent mean-zeroterms, and

$$
\begin{array}{l} \mathbb {E} \left[ \left(\frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) \left(1 - \frac {W _ {i}}{e (X _ {i})}\right)\right)\right) ^ {2} \mid \mathcal {I} _ {2}, \{X _ {i} \} \right] \\ = \operatorname{Var} \left[ \frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) \left(1 - \frac {W _ {i}}{e (X _ {i})}\right)\right) \Big | \mathcal {I} _ {2}, \{X _ {i} \} \right] \\ = \frac {1}{\left| \mathcal {I} _ {1} \right| ^ {2}} \sum_ {i \in \mathcal {I} _ {1}} \mathbb {E} \left[ \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) ^ {2} \left(1 - \frac {W _ {i}}{e (X _ {i})}\right) ^ {2} \mid \mathcal {I} _ {2}, \{X _ {i} \} \right] \tag {3.14} \\ = \frac {1}{| \mathcal {I} _ {1} | ^ {2}} \sum_ {i \in \mathcal {I} _ {1}} \frac {1 - e (X _ {i})}{e (X _ {i})} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) ^ {2} \\ \leq \frac {1 - \eta}{\eta} \frac {1}{| \mathcal {I} _ {1} | ^ {2}} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) ^ {2} = o _ {P} \left(\frac {1}{n ^ {1 + 2 \alpha_ {\mu}}}\right). \\ \end{array}
$$

The 3 equalities above are all due to cross-fitting, while the two inequalities are due to overlap (3.2) and consistency (3.11). Thus, because $\alpha _ { \mu } \geq 0$ , we can apply Chebyshev’s inequality to verify that the first summand itself is $o _ { P } ( 1 / \sqrt { n } )$ , i.e., as claimed it is negligible in probability on the $1 / { \sqrt { n } } { \mathrm { - s c a l e } }$ . The second summand in our decomposition above can also be bounded by a similar argument.

Finally, for the last summand, we use a Cauchy-Schwarz argument:19

$$
\begin{array}{l} \frac {1}{| \mathcal {I} _ {1} |} \sum_ {\{i: i \in \mathcal {I} _ {1}, W _ {i} = 1 \}} \left(\left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) \left(\frac {1}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - \frac {1}{e (X _ {i})}\right)\right) \\ \leq \sqrt {\frac {1}{| \mathcal {I} _ {1} |} \sum_ {\{i : i \in \mathcal {I} _ {1} , W _ {i} = 1 \}} \left(\hat {\mu} _ {(1)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(1)} (X _ {i})\right) ^ {2}} \tag {3.15} \\ \times \sqrt {\frac {1}{| \mathcal {I} _ {1} |} \sum_ {\{i : i \in \mathcal {I} _ {1} , W _ {i} = 1 \}} \left(\frac {1}{\hat {e} ^ {\mathcal {I} _ {2}} (X _ {i})} - \frac {1}{e (X _ {i})}\right) ^ {2}} = o _ {P} \left(\frac {1}{n ^ {\alpha_ {\mu} + \alpha_ {e}}}\right), \\ \end{array}
$$

by risk decay (3.11). Thus, we find that this term is also $o _ { P } ( 1 / \sqrt { n } )$ , i.e., as claimed it is negligible in probability on the $1 / { \sqrt { n } } .$ -scale.

Condensed notation We will be encountering cross-fit estimators frequently throughout the rest of this book. From now on, we’ll use the following notation: We define the data into K folds (above, $K = 2 )$ , and compute estimators µˆ(w) $\hat { \mu } _ { ( w ) } ^ { ( - k ) } ( x )$ , etc., excluding the k-th fold. Then, writing $k ( i )$ as the mapping that takes an observation and puts it into one of the k folds, we can write

$$
\begin{array}{l} \hat {\tau} _ {A I P W} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\hat {\mu} _ {(1)} ^ {(- k (i))} \left(X _ {i}\right) - \hat {\mu} _ {(0)} ^ {(- k (i))} \left(X _ {i}\right) \right. (3.16) \\ \left. + W _ {i} \frac {Y _ {i} - \hat {\mu} _ {(1)} ^ {(- k (i))} \left(X _ {i}\right)}{\hat {e} ^ {(- k (i))} \left(X _ {i}\right)} - \left(1 - W _ {i}\right) \frac {Y _ {i} - \hat {\mu} _ {(0)} ^ {(- k (i))} \left(X _ {i}\right)}{1 - \hat {e} ^ {(- k (i))} \left(X _ {i}\right)}\right). (3.16) \\ \end{array}
$$

Note that the result in Theorem 3.2 applies equally well with any finite number K of cross-fitting folds (and the same proof also works modulo updates to the notation).

Confidence intervals It is also important to be able to quantify uncertainty of treatment effect estimates. Thankfully, with AIPW, this turns out to be reasonably straight-forward. In the proof of Proposition 3.1, we saw that VAIPW matches the variance of the summands Γ used to define the oracle AIPW estimator (3.6). This suggests using the following feasible variance estimate:20

$$
\widehat {V} _ {A I P W} = \frac {1}{n - 1} \sum_ {i = 1} ^ {n} \left(\widehat {\Gamma} _ {i} - \widehat {\tau} _ {A I P W}\right),
$$

$$
\widehat {\Gamma} _ {i} = \hat {\mu} _ {(1)} ^ {(- k (i))} (X _ {i}) - \hat {\mu} _ {(0)} ^ {(- k (i))} (X _ {i}) \tag {3.17}
$$

$$
+ W _ {i} \frac {Y _ {i} - \hat {\mu} _ {(1)} ^ {(- k (i))} (X _ {i})}{\hat {e} ^ {(- k (i))} (X _ {i})} - (1 - W _ {i}) \frac {Y _ {i} - \hat {\mu} _ {(0)} ^ {(- k (i))} (X _ {i})}{1 - \hat {e} ^ {(- k (i))} (X _ {i})}.
$$

The proof of Theorem 3.2 then implies that, under our assumptions, $\widehat { V } _ { A I P W }  _ { p }$ $V _ { A I P W }$ . We can thus produce level-α confidence intervals for τ as

$$
\tau \in \left(\hat {\tau} _ {A I P W} \pm \Phi^ {- 1} \left(1 - \frac {\alpha}{2}\right) \frac {1}{\sqrt {n}} \sqrt {\hat {V} _ {A I P W}}\right), \tag {3.18}
$$

where $\Phi ( \cdot )$ is the standard Gaussian CDF, and these will achieve coverage with probability 1−α in large samples. Similar argument can also be used to justify inference via resampling methods as in Efron [1982].

What if the propensity score is known? One special case worth considering is, what happens when the propensity score is known, and we implement the cross-fit AIPW estimator (3.16) with the true propensity scores $\hat { e } ^ { - k ( i ) } ( X _ { i } ) = e ( X _ { i } )$ . In this case Theorem 3.2 immediately implies the following.

Corollary 3.3. Under our basic setting with SUTVA, unconfoundedness and strong overlap, suppose that we know the true propensity scores and use them to construct the AIPW estimator. Suppose moreover that

$$
\frac {1}{| \mathcal {I} _ {1} |} \sum_ {i \in \mathcal {I} _ {1}} \left(\hat {\mu} _ {(w)} ^ {\mathcal {I} _ {2}} (X _ {i}) - \mu_ {(w)} (X _ {i})\right) ^ {2} \rightarrow_ {p} 0, \tag {3.19}
$$

for $w \in \{ 0 , 1 \}$ and for with the roles of $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ swapped. Then (3.9) and (3.5) hold; and furthermore $\hat { \tau } _ { A I P W }$ is exactly unbiased, $\mathbb { E } \left[ \hat { \tau } _ { A I P W } \right] = \tau$ .

Proof. The CLT statement follows from applying Theorem 3.2 with $\alpha _ { \mu } = 0$ and $\alpha _ { e } = + \infty$ . The unbiasedness claim follows by noting that, in the decomposition below (3.13), the second and third terms disappear when the true propensity scores are used, while the first term is mean-zero. □

This result is remarkable in that it shows that, if we use AIPW with true propensity scores, then AIPW will achieve the target asymptotic behavior (3.5) as long as we use any regression adjustment that is consistent in the extremely weak sense (3.19). In particular, no rates of convergence are required.

It is well known that there are several machine learning methods, including k-nearest neighbors, that are universally consistent, i.e., they achieve error guarantees (3.19) for any IID data-generating distribution, without any assumptions on the joint distribution of $X _ { i }$ and $Y _ { i } ( w )$ other than E $[ Y _ { i } ^ { 2 } ( w ) ] < \infty$ [Stone, 1977]. Corollary 3.3 implies that if we run AIPW implemented with an universally consistent $\hat { \mu } _ { ( w ) } ( x )$ estimator and the true propensity scores, then it always satisfies (3.5) under our basic setting.

Corollary 3.3 also provides a practical resolution to the apparent paradox highlighted in Chapter 2, whereby IPW with oracle weights could sometimes (in specific settings) be outperformed by IPW with estimated weights. This seemed to lead to a tension where, if propensity scores were known, then we could choose to either use oracle IPW, which is always unbiased but has a larger asymptotic variance, or feasible IPW, which may be more accurate but may fail completely if we accidentally misspecify the propensity model.

The reason Corollary 3.3 helps is that, on inspection, one notices that the asymptotic variance $V _ { A I P W }$ achieved (in considerable generality) in Corollary 3.3 exactly matches the asymptotic variance $V _ { S T R A T }$ achieved by feasible IPW (in the special case where $X _ { i }$ has discrete support). Thus, what Corollary 3.3 shows us is that, if we know the true propensity scores, then we can always (and without really any downsides, at least asymptotically) avoid the excess asymptotic variance of oracle IPW by simply using AIPW with an universally consistent regression adjustment instead.

## 3.2 Efficient estimation under uncounfoundedness

In Chapter 2 we studied average treatment effect estimation under unconfoundedness and when $X _ { i }$ is discrete. In this setting, the stratify-by- $X _ { i }$ estimator is obviously a (or perhaps the) natural thing to do; and in Theorem 2.1 we showed that it achieves an asymptotic variance $V _ { S T R A T }$ . Meanwhile, in this chapter, we studied a seemingly completely different estimator, AIPW, and showed it can also achieve an asymptotic variance $V _ { A I P W } = V _ { S T R A T }$ , but under much more general conditions (and in particular without assuming that $X _ { i }$ is discrete).

These observations suggest that the behavior

$$
\begin{array}{l} \sqrt {n} \left(\hat {\tau} - \tau^ {*}\right) \Rightarrow \mathcal {N} \left(0, V ^ {*}\right) \\ V ^ {*} = \operatorname{Var} \left[ \tau (X _ {i}) \right] + \mathbb {E} \left[ \frac {\sigma_ {0} ^ {2} (X _ {i})}{1 - e (X _ {i})} \right] + \mathbb {E} \left[ \frac {\sigma_ {1} ^ {2} (X _ {i})}{e (X _ {i})} \right], \tag {3.20} \\ \end{array}
$$

may in fact be the optimal behavior we can hope to achieve for any nonparametric average treatment effect estimator $\hat { \tau }$ under unconfoundedness. Theorem 3.2 provides an upper bound, showing that this behavior can in fact be achieved by a practical estimator, $\hat { \tau } _ { A I P W }$ , under considerable generality. Meanwhile, our discussion in Chapter 2 provides a heuristic lower bound; after all, how could one possibly hope to find an estimator that’s more accurate than the strat ${ \mathrm { i f y } } { \mathrm { - b y } } { \mathrm { - } } X _ { i }$ estimator in the setting where $X _ { i }$ is discrete?

The following result establishes this conjecture, using a proof technique from Chamberlain [1992]. Following H´ajek [1972], he defines optimality in terms of a local asymptotic minimax criterion: $V ^ { * }$ is called the efficient variance for estimating $\tau$ if an estimator satisfying (3.20) exists and, for any data-generating distribution $P _ { \mathrm { : } }$ no estimator exists that is more accurate than (3.20) uniformly over a suitably expressive neighborhood of $P . ^ { 2 1 }$ Further, any estimator satisfying (3.20), potentially assuming reasonable regularity conditions, is called efficient.

Theorem 3.4. Under basic setting with SUTVA, unconfoundedness and strong overlap, $V ^ { * }$ is the efficient variance for estimating the average treatment effect.

Proof. We have already established existence of an estimator satisfying (3.20) in Theorem 3.2. For the local optimality statement, we follow the blueprint of Theorem 1 of Chamberlain [1992], and do the following: We start by considering distributions where $( X _ { i } , Y _ { i } ( 0 ) , Y _ { i } ( 0 ) )$ have a distribution P with a jointly discrete support $( \mathrm { i . e . }$ , both $X _ { i }$ and $Y _ { i } ( w )$ have discrete support), and verify that the asymptotic variance of the saturated maximum likelihood estimator of the ATE matches $V ^ { * }$ . We then argue that ATE estimation with a discrete $P$ is a parametric problem and so maximum likelihood estimation must be efficient; and that any continuous distribution is well approximable by a discrete distribution, so this efficiency result carries over to the continuous case. We refer to Chamberlain [1992] for technical details, and for verifying that this blueprint is in fact valid.

Consider now the case where P takes on values on a discrete space $\mathcal { X } \times \mathcal { Y } \times \mathcal { Y }$ with $\mathcal { V } \subset \mathbb { R }$ . For any distribution P let $\tau ( P ) = \mathbb { E } _ { P } \left[ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \right]$ and note that, under unconfoundedness and with discrete support,

$$
\tau (P) = \sum_ {x \in \mathcal {X}} P (x) \left(\sum_ {y \in \mathcal {Y}} y   P _ {1} (y | x) - \sum_ {y \in \mathcal {Y}} y   P _ {0} (y | x)\right) \tag {3.21}
$$

where $P ( x ) = \mathbb { E } _ { P } \left[ X _ { i } = x \right]$ and $P _ { w } ( y | x ) = \mathbb { E } _ { P } \left[ Y _ { i } = y \vert X _ { i } = x , W _ { i } = w \right]$ . Now, given n draws from P , let $n _ { x } = | \{ i : X _ { i } = x \} | , n _ { x w } \stackrel { \cdot } { = } | \{ i : X _ { i } = x , \bar { W _ { i } } = w \}$ | and $n _ { x y w } = | \{ i : X _ { i } = x , Y _ { i } = y , W _ { i } = w \} |$ . The saturated maximum likelihood estimator for the data-generating distribution $P$ is given by $\widehat { P } ( x ) = n _ { x } / n$ and $\widehat { P } _ { w } ( y | x ) = n _ { x y w } / n _ { x w }$ . The maximum likelihood estimator for τ is then

$$
\hat {\tau} = \tau (\widehat {P}) = \sum_ {x \in \mathcal {X}} \widehat {P} (x) \left(\sum_ {y \in \mathcal {Y}} y   \widehat {P} _ {1} (y | x) - \sum_ {y \in \mathcal {Y}} y   \widehat {P} _ {0} (y | x)\right), \tag {3.22}
$$

which can be algebraically be verified to be equivalent to $\scriptstyle { \hat { \tau } } _ { S T R A T }$ in this setting. Thus, the asymptotic variance of maximum likelihood here is $V _ { S T R A T }$ , which by Theorem 2.1 is equal to $V ^ { * }$ . □

Comparing regularity conditions One ambiguity in the definitions above is that we said that an estimator is efficient if it achieves the behavior (3.20) under “reasonable” regularity conditions—but what does it mean for regularity conditions to be reasonable? We have so far seen 3 results about estimators achieving the behavior (3.20): Corollary 3.3 shows this for AIPW with known propensity scores essentially without assumptions; Theorem 3.2 shows this for AIPW with estimated propensity scores under the (moderately strong?) rate-of-convergence assumption (3.11); while Theorem 2.1 showed this for the stratify-on- $X _ { i }$ estimator under the (very strong) assumption that $X _ { i }$ is discrete.

This ambiguity is intentional, and can be helpful in describing and assessing various proposed estimators of the average treatment effect under unconfoundedness. When considering a candidate estimator, a good first question can be to ask whether it is efficient, i.e., whether it sometimes achieves the behavior (3.11). If an estimator is not efficient (e.g., like the oracle IPW estimator), then it may be worth discarding at this step. Then, among efficient estimators, a good second question to ask is how robust it is, i.e., how strong are the regularity conditions needed for efficiency. This allows to argue, e.g., that $\hat { \tau } _ { A I P W }$ requires much weaker regularity conditions than $\scriptstyle { \hat { \tau } } _ { S T R A T }$ to achieve desirable asymptotic performance, and from this angle $\hat { \tau } _ { A I P W }$ appears preferable.

Is efficiency a realistic goal? Until recently, the perspective taken above, i.e., that efficiency is a criterion that should guide practical choice of average treatment effect estimators, would have been considered controversial by many econometricians and statisticians. Methods that achieved efficiency were often considered fragile, complicated and/or impractical; and, in problems that called for treatment effect estimation under unconfoundedness, econometric practice largely focused on methods that require parametric assumptions and are not consistent under unconfoundedness alone (e.g., linear regression), or non-efficient but conceptually simple methods (e.g., matching).

The critique that early methods designed to achieve efficiency were hard to use in practice is on point: For example, such methods would often rely on specific smoothness assumptions, and then rely on series estimators with specific basis functions (depending on the assumed smoothness class) to form treatment effect estimators.

The double machine learning framework, however, makes widespread use of efficient treatment effect estimators much more practical. The main regularity condition (3.11) doesn’t depend on how we choose to estimate the non-parametric components, and instead only requires that they are accurate enough under squared-error loss. Machine learning methods are often tuned via cross-validation under squared error loss, and this way of tuning predictors is perfectly aligned with making the error terms in (3.11) small. Thus, perhaps surprisingly, although machine learning may at first seem like a glance seem like a technology that should be kept as far away from causal inference as possible, it turns out that—via the double machine learning construction—machine learning (and, more generally, automatic black-box non-parametric prediction)is a key ingredient in making efficient treatment effect estimation practical in a wide variety of settings.

## 3.3 Bibliographic notes

The literature on semiparametrically efficient treatment effect estimation via AIPW was pioneered by Robins, Rotnitzky, and Zhao [1994], and developed in a sequence of papers including Robins and Rotnitzky [1995] and Scharfstein, Rotnitzky, and Robins [1999]. The form of the AIPW estimator is also present in early work by Cassel, S¨arndal, and Wretman [1976] in survey sampling. The effect of knowing the propensity score on the semiparametric efficiency bound for average treatment effect estimation is discussed in Hahn [1998], while the behavior of AIPW with high-dimensional regression adjustments was first considered by Farrell [2015]. These results fit into a broader literature on semiparametrics, including Bickel, Klaassen, Ritov, and Wellner [1993] and Newey [1994].

The approach taken here, with a focus on generic machine learning estimators for non-parametric components and cross-fitting, follows the double machine learning framework of Chernozhukov et al. [2018]. One major strength of this approach is in its generality and its ability to handle arbitrary machine learning estimators for $\hat { \mu } _ { ( w ) } ( x )$ and $\hat { e } ( x )$ . Another, closely related framework is the targeted learning framework of van der Laan and Rubin [2006], which uses a different functional form than AIPW but can also be shown to achieve efficiency using machine learning estimators for non-parametric components [van der Laan and Rose, 2011].

There is a large number of estimators known to achieve efficiency under a variety of regularity conditions. For example, Hahn [1998] showed that non-parametric regression adjustment estimators can be efficient under strong smoothness conditions and specific regression estimators, while Hirano, Imbens, and Ridder [2003] showed this type of result for non-parametric IPW. The efficiency result given in Theorem 3.2 for AIPW is, however, much more robust—in that it allows for use of generic machine learning methods provided they satisfy the relatively mild rate conditions (3.11).

More recently, there has been considerable interest in deriving estimators that achieve efficiency under minimal conditions. In the case where the functions $\mu _ { ( w ) } ( \cdot )$ and $e ( \cdot )$ belong to H¨older smoothness classes Robins et al. [2017] show that, writing $\alpha _ { \mu }$ and $\alpha _ { e }$ for the best constants for which rates of convergence of the type (3.11) can be achieved under the posited smoothness assumptions, the weakest condition under which efficiency is possible is

$$
\frac {\alpha_ {\mu}}{1 - 2 \alpha_ {\mu}} + \frac {\alpha_ {e}}{1 - 2 \alpha_ {e}} \geq \frac {1}{2}, \tag {3.23}
$$

and this rate can be achieved using what Robins et al. [2017] refer to as higherorder influence function (HOIF) estimators. The improvement of the condition (3.23) over the condition $\alpha _ { \mu } + \alpha _ { e } \ge 1 / 2$ in Theorem 3.2 is considerable; for example, when both rates are equal, in Theorem 3.2 we could allow for $\alpha _ { \mu } =$ $\alpha _ { e } \geq 1 / 4$ while (3.23) allows for $\alpha _ { \mu } = \alpha _ { e } \ge 1 / 6$ .

One challenge with the HOIF estimator of Robins et al. [2017], however, is that to date it has been challenging to implement in practical applications; and so there has been work on methods that can improve over AIPW while remaining practically feasible. Hirshberg and Wager [2021] show that a variant of AIPW with a choice of propensity model specifically designed to minimize bias from errors in $\hat { \mu } _ { ( w ) } ( x )$ is efficient under conditions that, in the H¨older case, amount to $\alpha _ { \mu } \geq 1 / 4$ (with no assumptions on $\alpha _ { e } ) ;$ ; note that this corresponds to one extreme point of the optimality surface (3.23). Meanwhile, Newey and Robins [2018] and McClean et al. [2024] show how, in some settings, the use of undersmoothed estimators and 3-way cross-fitting can achieve minimal conditions for efficiency.
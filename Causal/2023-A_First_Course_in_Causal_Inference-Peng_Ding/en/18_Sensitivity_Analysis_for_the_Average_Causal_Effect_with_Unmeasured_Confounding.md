# Sensitivity Analysis for the Average Causal Effect with Unmeasured Confounding

Cornfield-type sensitivity analysis works the best for binary outcomes on the risk ratio scale, conditioning on the observed covariates. Although Ding and VanderWeele (2016) also proposed Cornfield-type sensitivity analysis methods for the average causal effect, they are not general enough and are not convenient to apply. Below I give a more direct approach to sensitivity analysis based on the conditional expectations of the potential outcomes. The idea appeared in early work of Robins (1999) and Scharfstein et al. (1999). This chapter is based on Lu and Ding (2023)’s recent formulation.

The approach is closely related to the idea of deriving worse-case bounds on the average potential outcomes. I will first review the simpler idea of bounds, and then discuss the approach to sensitivity analysis.

## 18.1 Introduction

$\{ Z _ { i } , X _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \} _ { i = 1 } ^ { n } \stackrel { \mathrm { I I D } } { \sim }$ {Z, X, Y (1), Y (0)} and focus on the average causal effect

$$
\tau = E \{Y (1) - Y (0) \}.
$$

It decomposes to

$$
\begin{array}{l} \tau = \left[ E (Y \mid Z = 1) \mathrm{pr} (Z = 1) + E \{Y (1) \mid Z = 0 \} \mathrm{pr} (Z = 0) \right] \\ - \left[ E \{Y (0) \mid Z = 1 \} \mathrm{pr} (Z = 1) + E (Y \mid Z = 0) \mathrm{pr} (Z = 0) \right]. \\ \end{array}
$$

So the fundamental difficulty is to estimate the counterfactual means

$$
E \{Y (1) \mid Z = 0 \}, \qquad E \{Y (0) \mid Z = 1 \}.
$$

There are in general two extreme strategies to estimate them.

We have discussed the first strategy in Part III, which relies on ignorability. Assuming

$$
\begin{array}{l} E \{Y (1) \mid Z = 1, X \} = E \{Y (1) \mid Z = 0, X \}, \\ E \{Y (0) \mid Z = 1, X \} = E \{Y (0) \mid Z = 0, X \}, \\ \end{array}
$$

**TABLE 18.1: Science Table with bounded outcome [ℓ, u], where ℓ and u are two constants**

<table><tr><td>Z</td><td>Y(1)</td><td>Y(0)</td><td>Lower Y(1)</td><td>Upper Y(1)</td><td>Lower Y(0)</td><td>Upper Y(0)</td></tr><tr><td>1</td><td> $Y_1(1)$ </td><td>?</td><td> $Y_1(1)$ </td><td> $Y_1(1)$ </td><td> $\ell$ </td><td>u</td></tr><tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr><tr><td>1</td><td> $Y_{n_1}(1)$ </td><td>?</td><td> $Y_{n_1}(1)$ </td><td> $Y_{n_1}(1)$ </td><td> $\ell$ </td><td>u</td></tr><tr><td>0</td><td>?</td><td> $Y_{n_1+1}(0)$ </td><td> $\ell$ </td><td>u</td><td> $Y_{n_1+1}(0)$ </td><td> $Y_{n_1+1}(0)$ </td></tr><tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr><tr><td>0</td><td>?</td><td> $Y_n(0)$ </td><td> $\ell$ </td><td>u</td><td> $Y_n(0)$ </td><td> $Y_n(0)$ </td></tr></table>

we can identify the counterfactual means by the observables:

$$
E \{Y (1) \mid Z = 0 \} = E \left\{E (Y \mid Z = 1, X) \mid Z = 0 \right\}
$$

and, similarly,

$$
E \{Y (0) \mid Z = 1 \} = E \left\{E (Y \mid Z = 0, X) \mid Z = 1 \right\}.
$$

The second strategy in the next section assumes nothing except that the outcomes are bounded between ℓ and u. This is natural for binary outcomes with ℓ = 0 and u = 1. With this assumption, the two counterfactual means are also bounded between ℓ and u, which implies the worse-case bounds on τ . I will review this strategy below.

## 18.2 Manski-type worse-case bounds on the average causal effect without assumptions

Assume that the outcome is bounded between ℓ and u. From the decomposition

$$
E \{Y (1) \} = E \{Y (1) \mid Z = 1 \} \mathrm{pr} (Z = 1) + E \{Y (1) \mid Z = 0 \} \mathrm{pr} (Z = 0),
$$

we can derive that E{Y (1)} has lower bound

$$
E \{Y \mid Z = 1 \} \mathrm{pr} (Z = 1) + \ell \mathrm{pr} (Z = 0)
$$

and upper bound

$$
E \{Y \mid Z = 1 \} \mathrm{pr} (Z = 1) + u \mathrm{pr} (Z = 0).
$$

Similarly, from the decomposition

$$
E \{Y (0) \} = E \{Y (0) \mid Z = 1 \} \mathrm{pr} (Z = 1) + E \{Y (0) \mid Z = 0 \} \mathrm{pr} (Z = 0),
$$

## 18.3 Manski-type worse-case bounds on the average causal effect without assumptions 227

we can derive that $E \{ Y ( 0 ) \}$ has lower bound

$$
\ell \mathrm{pr} (Z = 1) + E \{Y \mid Z = 0 \} \mathrm{pr} (Z = 0)
$$

and upper bound

$$
u \mathrm{pr} (Z = 1) + E \{Y \mid Z = 0 \} \mathrm{pr} (Z = 0).
$$

Combining these bounds, we can derive that the average causal effect $\tau =$ $E \{ Y ( 1 ) \} - E \{ Y ( 0 ) \}$ has lower bound

$$
E \{Y \mid Z = 1 \} \mathrm{pr} (Z = 1) + \ell \mathrm{pr} (Z = 0) - u \mathrm{pr} (Z = 1) - E \{Y \mid Z = 0 \} \mathrm{pr} (Z = 0)
$$

and upper bound

$$
E \{Y \mid Z = 1 \} \mathrm{pr} (Z = 1) + u \mathrm{pr} (Z = 0) - \ell \mathrm{pr} (Z = 1) - E \{Y \mid Z = 0 \} \mathrm{pr} (Z = 0).
$$

The length of the bounds is $u - \ell ,$ which is not informative but is better than the a priori bounds $\left[ \ell - u , u - \ell \right]$ with length $2 ( u - \ell )$ . Without further assumptions, the observed data distribution does not uniquely determine τ . In this case, we say that τ is partially identified, with the formal definition below.

Definition 18.1 (partial identification) A parameter θ is partially identified if the observed data distribuion is compatible with multiple values of θ.

Compare Definitions 10.1 and 18.1. If the parameter θ is uniquely determined by the observed data distribution, then it is identifiable; otherwise, it is partially identifiable. Therefore, τ is identifiable with the ignorability assumption, but only partially identifiable without the ignorability assumption.

Cochran (1953) used the idea of worse-case bounds in surveys with missing data, but abandoned the idea because it often gives very conservative results. Similarly, the worst-case bounds above are often uninteresting from a practical perspective because they often cover 0. Moreover, this strategy is not applicable to the settings with unbounded outcomes.

Manski applied the idea to causal inference (Manski, 1990) and many other econometric models (Manski, 2003). This idea of bounding causal parameters with minimal assumptions is powerful when coupled with other qualitative assumptions. Manski (2003) surveyed many strategies. For instance, we may believe that the treatment does not harm any units, so the monotonicity assumption holds: $Y ( 1 ) \ge Y ( 0 )$ . Then the lower bound on τ is zero but the upper bound is unchanged. Another type of assumption is $Z = I \{ Y ( 1 ) \geq$ $Y ( 0 ) ]$ , that is, the treatment selection is based on the difference between the latent potential outcomes. This assumption can also improve the bounds on τ .

## 18.3 Sensitivity analysis for the average causal effect

The first strategy is optimistic which assumes that the potential outcomes do not differ across treatment and control groups, conditioning on the observed covariates. The second strategy is pessimistic which does not infer the counterfactual means based on the observed data at all. The following strategy is in-between.

## 18.3.1 Identification formulas

Define

$$
\frac {E \{Y (1) \mid Z = 1 , X \}}{E \{Y (1) \mid Z = 0 , X \}} = \varepsilon_ {1} (X),
$$

$$
\frac {E \{Y (0) \mid Z = 1 , X \}}{E \{Y (0) \mid Z = 0 , X \}} = \varepsilon_ {0} (X),
$$

which are the sensitivity parameters. For simplicity, we can further assume that they are constant independent of X. In practice, we need to fix them or vary them in a pre-specified range. Recall that $\mu _ { 1 } ( X ) = E ( Y \mid Z = 1 , X )$ and $\mu _ { 0 } ( X ) = E ( Y \mid Z = 0 , X )$ . We can identify the two counterfactual means and the average causal effect as follows.

Theorem 18.1 With known $\varepsilon _ { 1 } ( X )$ and $\varepsilon _ { 0 } ( X )$ , we have

$$
E \{Y (1) \mid Z = 0 \} = E \left\{\mu_ {1} (X) / \varepsilon_ {1} (X) \mid Z = 0 \right\},
$$

$$
E \{Y (0) \mid Z = 1 \} = E \left\{\mu_ {0} (X) \varepsilon_ {0} (X) \mid Z = 1 \right\}
$$

and therefore

$$
\begin{array}{l} \tau = E \{Z Y + (1 - Z) \mu_ {1} (X) / \varepsilon_ {1} (X) \} \\ - E \{Z \mu_ {0} (X) \varepsilon_ {0} (X) + (1 - Z) Y \} (18.1) \\ = E \left\{Z \mu_ {1} (X) + (1 - Z) \mu_ {1} (X) / \varepsilon_ {1} (X) \right\} \\ - E \{Z \mu_ {0} (X) \varepsilon_ {0} (X) + (1 - Z) \mu_ {0} (X) \}. (18.2) \\ \end{array}
$$

I leave the proof of Theorem 18.1 to Problem 18.1. With the fitted outcome model, (18.1) and (18.2) motivate the following predictive and projective estimators for τ :

$$
\begin{array}{l} \hat {\tau} ^ {\mathrm{pred}} = \left\{n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i} + n ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \hat {\mu} _ {1} (X _ {i}) / \varepsilon_ {1} (X _ {i}) \right\} \\ - \left\{n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \hat {\mu} _ {0} (X _ {i}) \varepsilon_ {0} (X _ {i}) + n ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) Y _ {i} \right\}, \\ \end{array}
$$

and

$$
\begin{array}{l} \hat {\tau} ^ {\text { proj }} = \left\{n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \hat {\mu} _ {1} (X _ {i}) + n ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \hat {\mu} _ {1} (X _ {i}) / \varepsilon_ {1} (X _ {i}) \right\} \\ \left. - \left\{n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \hat {\mu} _ {0} (X _ {i}) \varepsilon_ {0} (X _ {i}) + n ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \hat {\mu} _ {0} (X _ {i}) \right\}. \right. \\ \end{array}
$$

The terminologies “predictive” and “projective” are from the survey sampling literature (Firth and Bennett, 1998; Ding and Li, 2018). The estimators ˆτpred and $\hat { \tau } ^ { \mathrm { p r o j } }$ differ slightly: the former uses the observed outcomes when available; in contrast, the latter replaces the observed outcomes with the fitted values.

More interesting, we can also identify τ by an inverse probability weighting formula.

Theorem 18.2 With known $\varepsilon _ { 1 } ( X )$ and $\varepsilon _ { 0 } ( X )$ , we have

$$
E \{Y (1) \} = E \left\{w _ {1} (X) \frac {Z}{e (X)} Y \right\}, \quad E \{Y (0) \} = E \left\{w _ {0} (X) \frac {1 - Z}{1 - e (X)} Y \right\},
$$

where

$$
w _ {1} (X) = e (X) + \{1 - e (X) \} / \varepsilon_ {1} (X), w _ {0} (X) = e (X) \varepsilon_ {0} (X) + 1 - e (X).
$$

I leave the proof of Theorem 18.2 to Problem 18.2. Theorem 18.2 modifies the classic inverse probability weighting formulas with two extra factors $w _ { 1 } ( X )$ and $w _ { 0 } ( X )$ depending on both the propensity score and the sensitivity parameters. With the fitted propensity score model, Theorem 18.2 motivates the following estimators for τ :

$$
\begin{array}{l} \hat {\tau} ^ {\mathrm{ht}} = n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {\{\hat {e} (X _ {i}) \varepsilon_ {1} (X _ {i}) + 1 - \hat {e} (X _ {i}) \} Z _ {i} Y _ {i}}{\varepsilon_ {1} (X _ {i}) \hat {e} (X _ {i})} \\ - n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {\{\hat {e} (X _ {i}) \varepsilon_ {0} (X _ {i}) + 1 - \hat {e} (X _ {i}) \} (1 - Z _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})} \\ \end{array}
$$

and

$$
\begin{array}{l} \hat {\tau} ^ {\text { haj }} = \sum_ {i = 1} ^ {n} \frac {\{\hat {e} (X _ {i}) \varepsilon_ {1} (X _ {i}) + 1 - \hat {e} (X _ {i}) \} Z _ {i} Y _ {i}}{\varepsilon_ {1} (X _ {i}) \hat {e} (X _ {i})} / \sum_ {i = 1} ^ {n} \frac {Z _ {i}}{\hat {e} (X _ {i})} \\ - n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {\{\hat {e} (X _ {i}) \varepsilon_ {0} (X _ {i}) + 1 - \hat {e} (X _ {i}) \} (1 - Z _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})} \Big / \sum_ {i = 1} ^ {n} \frac {1 - Z _ {i}}{1 - \hat {e} (X _ {i})}. \\ \end{array}
$$

More interestingly, with fitted propensity score and outcome models, the following estimator for τ is doubly robust:

$$
\hat {\tau} ^ {\mathrm{ht}} = \hat {\tau} ^ {\mathrm{ipw}} - n ^ {- 1} \sum_ {i = 1} ^ {n} \{Z _ {i} - \hat {e} (X _ {i}) \} \left\{\frac {\hat {\mu} _ {1} (X _ {i})}{\hat {e} (X _ {i}) \varepsilon_ {1} (X _ {i})} + \frac {\hat {\mu} _ {0} (X _ {i}) \varepsilon_ {0} (X _ {i})}{1 - \hat {e} (X _ {i})} \right\}.
$$

That is, with known $\varepsilon _ { 1 } ( X _ { i } )$ and $\varepsilon _ { 0 } ( X _ { i } )$ , the estimator ${ \hat { \tau } } ^ { \mathrm { d r } }$ is consistent for τ if either the propensity score model or the outcome model is correctly specified. We can use the bootstrap to approximate the variance of the above estimators. See Lu and Ding (2023) for technical details.

When $\varepsilon _ { 1 } ( X _ { i } ) = \varepsilon _ { 0 } ( X _ { i } ) = 1$ , the above estimators reduce to the predictive estimator, inverse probability weighting estimator, and the doubly robust estimators introduced in Part III.

## 18.4 Example

We revisit Example 10.3. With

$$
\varepsilon_ {1} (X) = \varepsilon_ {0} (X) \in \{1 / 2, 1 / 1. 7, 1 / 1. 5, 1 / 1. 3, 1, 1. 3, 1. 5, 1. 7, 2 \},
$$

we obtain an array of doubly robust estimates of τ .

<table><tr><td></td><td>1/2</td><td>1/1.7</td><td>1/1.5</td><td>1/1.3</td><td>1</td><td>1.3</td><td>1.5</td><td>1.7</td><td></td></tr><tr><td colspan="10">2</td></tr><tr><td>1/2</td><td>11.62</td><td>10.44</td><td>9.40</td><td>8.03</td><td>4.96</td><td>0.97</td><td>-1.69</td><td>-4.35</td><td>-8.34</td></tr><tr><td>1/1.7</td><td>9.22</td><td>8.05</td><td>7.00</td><td>5.64</td><td>2.57</td><td>-1.42</td><td>-4.08</td><td>-6.75</td><td>-10.74</td></tr><tr><td>1/1.5</td><td>7.63</td><td>6.45</td><td>5.41</td><td>4.05</td><td>0.97</td><td>-3.02</td><td>-5.68</td><td>-8.34</td><td>-12.33</td></tr><tr><td>1/1.3</td><td>6.03</td><td>4.86</td><td>3.81</td><td>2.45</td><td>-0.62</td><td>-4.61</td><td>-7.27</td><td>-9.94</td><td>-13.93</td></tr><tr><td>1</td><td>3.64</td><td>2.47</td><td>1.42</td><td>0.06</td><td>-3.01</td><td>-7.01</td><td>-9.67</td><td>-12.33</td><td>-16.32</td></tr><tr><td>1.3</td><td>1.80</td><td>0.63</td><td>-0.42</td><td>-1.78</td><td>-4.85</td><td>-8.85</td><td>-11.51</td><td>-14.17</td><td>-18.16</td></tr><tr><td>1.5</td><td>0.98</td><td>-0.19</td><td>-1.24</td><td>-2.60</td><td>-5.67</td><td>-9.66</td><td>-12.33</td><td>-14.99</td><td>-18.98</td></tr><tr><td>1.7</td><td>0.36</td><td>-0.82</td><td>-1.86</td><td>-3.23</td><td>-6.30</td><td>-10.29</td><td>-12.95</td><td>-15.61</td><td>-19.60</td></tr><tr><td>2</td><td>-0.35</td><td>-1.52</td><td>-2.57</td><td>-3.93</td><td>-7.00</td><td>-10.99</td><td>-13.65</td><td>-16.32</td><td>-20.31</td></tr></table>

The signs of the estimates are not sensitive to sensitivity parameters larger than 1, but they are quite sensitivity to sensitivity parameters smaller than 1. When the participants of the meal plan tend to have higher BMI, the average causal effect of the meal plan on BMI is negative. However, this conclusion can be quite sensitive if the participants of the meal plan tend to have lower BMI.

## 18.5 Homework Problems

## 18.1 Proof of Theorem 18.1

Prove Theorem 18.1.

## 18.2 Proof of Theorem 18.2

Prove Theorem 18.2.

18.3 Sensitivity analysis for the average causal effect on the treated units $\tau _ { \mathrm { T } }$ This problem extends Chapter 13 to allow for unmeasured confounding for estimating

$$
\tau_ {\mathrm{T}} = E \{Y (1) - Y (0) \mid Z = 1 \} = E (Y \mid Z = 1) - E \{Y (0) \mid Z = 1 \}.
$$

We can easily estimate $E ( Y \mid Z = 1 )$ by the sample moment. The only counterfactual term is $E \{ Y ( 0 ) \mid Z = 1 \}$ . Therefore, we only need the sensitivity parameter $\varepsilon _ { 0 } ( X )$ . We have the following two identification formulas with a known $\varepsilon _ { 0 } ( X )$ .

Theorem 18.3 With known $\varepsilon _ { 0 } ( X )$ , we have

$$
\begin{array}{l} E \{Y (0) \mid Z = 1 \} = E \left\{Z \mu_ {0} (X) \varepsilon_ {0} (X) \right\} / e \\ { = } { E \left\{ e ( X ) \varepsilon _ { 0 } ( X ) \frac { 1 - Z } { 1 - e ( X ) } Y \right\} / e , } \\ \end{array}
$$

where $e = \operatorname { p r } ( Z = 1 )$

Prove Theorem 18.3.

where Remark: Theorem 18.3 motivates using $\begin{array} { r } { \hat { \mu } _ { \mathrm { T 1 } } = \sum _ { i = 1 } ^ { n } Z _ { i } Y _ { i } / \sum _ { i = 1 } ^ { n } Z _ { i } } \end{array}$ and $\hat { \tau } _ { \mathrm { r } } ^ { * } = \hat { \mu } _ { \mathrm { T 1 } } - \hat { \mu } _ { \mathrm { T 0 } } ^ { * }$ to estimate $\tau _ { \mathrm { { T } } } .$

$$
\begin{array}{l} \hat {\mu} _ {\mathrm{T0}} ^ {\mathrm{reg}} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \varepsilon_ {0} (X _ {i}) \hat {\mu} _ {0} (X _ {i}), \\ \hat {\mu} _ {\mathrm{T0}} ^ {\mathrm{ht}} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} \varepsilon_ {0} (X _ {i}) \hat {o} (X _ {i}) (1 - Z _ {i}) Y _ {i}, \\ \hat {\mu} _ {\mathrm{T0}} ^ {\text { haj }} = \sum_ {i = 1} ^ {n} \varepsilon_ {0} (X _ {i}) \hat {o} (X _ {i}) (1 - Z _ {i}) Y _ {i} / \sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}), \\ \end{array}
$$

with $\hat { o } ( X _ { i } ) = \hat { e } ( X _ { i } ) / \{ 1 - \hat { e } ( X _ { i } ) \}$ being the estimated conditional odds of the treatment. Moreover, we can construct the doubly robust estimator $\hat { \tau } _ { \mathrm { r } } ^ { \mathrm { d r } } ~ =$ $\hat { \mu } _ { \mathrm { T 1 } } - \hat { \mu } _ { \mathrm { T 0 } } ^ { \mathrm { d r } }$ for $\tau _ { \mathrm { T } }$ , where

$$
\hat {\mu} _ {\mathrm{T0}} ^ {\mathrm{dr}} = \hat {\mu} _ {\mathrm{T0}} ^ {\mathrm{ht}} - n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} \varepsilon_ {0} (X _ {i}) \frac {\hat {e} (X _ {i}) - Z}{1 - \hat {e} (X _ {i})} \hat {\mu} _ {0} (X _ {i}).
$$

Lu and Ding (2023) provide more details and also propose a doubly robust estimator for τT.

## 18.4 R code

Implement the estimators in Problem 18.3.

## 18.5 Recommended reading

Rosenbaum and Rubin (1983a) and Imbens (2003) are two classic papers on sensitivity analysis which, however, involve more complicated procedures.

<!-- footnote -->

- We define $\begin{array} { r } { \| v \| _ { 2 } ^ { 2 } = \sum _ { j = 1 } ^ { p } v _ { j } ^ { 2 } } \end{array}$ for a vector $\boldsymbol { v } = ( v _ { 1 } , \ldots , v _ { p } ) ^ { \mathsf { T } }$ . It denotes the squared length of the vector v.

<!-- footnote end -->

<!-- footnote -->

- It is not ideal for our discussion of binary Z, but it simplifies the derivations. Ding and Miratrix (2015) gave detailed discussion with more natural models for binary Z.

<!-- footnote end -->

<!-- footnote -->

- Again, we generate continuous Z from a linear model to simplify the derivations. Ding et al. (2017b) extended the theory to more general causal models, especially for binary $Z .$ .

<!-- footnote end -->

<!-- footnote -->

- Their original analysis was based on a case-control study and estimated the odds ratio of cigarette smoking on lung cancer. But the risk ratio is close to the odds ratio since lung cancer is a rare outcome.

<!-- footnote end -->

<!-- footnote -->

- In information theory, the mutual information
- $I ( A , B ) = \iint p ( a , b ) \log _ { 2 } \frac { p ( a , b ) } { p ( a ) p ( b ) } \mathrm { d } a \mathrm { d } b$
- measures the dependence between two random variables A and $B ,$ where $p ( \cdot )$ denotes the joint or marginal density of $( A , B )$ . The data processing inequality is a famous result: if $Z \bot \bot Y \mid U$ , then $I ( Z , Y ) \ge I ( Z , U )$ and $I ( Z , Y ) \ge I ( U , Y )$ . Lihua Lei and Bin Yu pointed out to me the connection between Cornfield’s inequality and the data processing inequality.

<!-- footnote end -->

## 19
# A2 Linear and Logistic Regressions

## A2.1 Population ordinary least squares

Assume that $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { n } \stackrel { \mathrm { I I D } } { \sim } ( x , y )$ , where x is a p-dimensional random scalar or vector and $y$ is a random scalar. Below I will use $( x , y )$ to denote a general observation, dropping the subscript i for simplicity. Define the population ordinary least squares (OLS) coefficient as

$$
\beta = \arg \min _ {b} E \left\{(y - x ^ {\mathsf {T}} b) ^ {2} \right\}.
$$

The objective function is quadratic in $b ,$ so we can show that the minimizer is

$$
\beta = \left\{E \left(x x ^ {\mathsf {T}}\right) \right\} ^ {- 1} E (x y)
$$

if the moments exist and $E \left( x x ^ { \mathsf { T } } \right)$ is invertible.

With $\beta ,$ we can define

$$
\varepsilon = y - x ^ {\mathsf {T}} \beta \tag {A2.1}
$$

as the population residual. By the definition of $\beta ,$ we can verify that

$$
E (x \varepsilon) = E \left\{x (y - x ^ {\mathsf {T}} \beta) \right\} = E (x y) - E (x x ^ {\mathsf {T}}) \beta = 0.
$$

Example A2.1 (population OLS with an intercept) If we include 1 as a component of x, then

$$
E (\varepsilon) = E (y - x ^ {\mathsf {T}} \beta) = 0
$$

which further implies that $\mathrm { c o v } ( x , \varepsilon ) = 0$ . So with an intercept in $\beta ,$ the mean of the population residual must be zero, and it is uncorrelated with other covariates by construction.

Example A2.2 (univariate population OLS with an intercept) An important special case is that for scalars x and y, we can define

$$
(\alpha , \beta) = \arg \min _ {a, b} E \{(y - a - b x) ^ {2} \}
$$

which have explicit formulas

$$
\beta = \frac {\operatorname{cov} (x , y)}{\operatorname{var} (x)}, \quad \alpha = E (y) - \beta E (x).
$$

Example A2.3 (univariate population OLS without an intercept) Without intercept, we can define

$$
\gamma = \arg \min _ {c} E \{(y - c x) ^ {2} \}
$$

which equals

$$
\gamma = \frac {E (x y)}{E (x ^ {2})}.
$$

When x has mean zero, $\beta = \gamma$ in the above two population OLS.

We can also rewrite (A2.1) as

$$
y = x ^ {\mathsf {T}} \beta + \varepsilon , \tag {A2.2}
$$

which holds by the definition of the population OLS coefficient and residual without any modeling assumption. We call (A2.2) the population OLS decomposition.

## A2.2 Sample OLS

$( x _ { i } , y _ { i } ) _ { i = 1 } ^ { n } \stackrel { \mathrm { I I D } } { \sim } ( x , y )$ for the population OLS coefficient

$$
\hat {\beta} = \left(n ^ {- 1} \sum_ {i = 1} ^ {n} x _ {i} x _ {i} ^ {\top}\right) ^ {- 1} \left(n ^ {- 1} \sum_ {i = 1} ^ {n} x _ {i} y _ {i}\right),
$$

and the residuals $\hat { \varepsilon } _ { i } = y _ { i } - x _ { i } ^ { \top } \hat { \beta }$ . This is called the sample OLS or simply the OLS. The OLS coefficient $\hat { \beta }$ minimizes the residual sum of squares

$$
\hat {\beta} = \arg \min _ {b} n ^ {- 1} \sum_ {i = 1} ^ {n} (y _ {i} - x _ {i} ^ {\mathsf {T}} b) ^ {2},
$$

which satisfies the following Normal equation:

$$
\sum_ {i = 1} ^ {n} x _ {i} (y _ {i} - x _ {i} ^ {\mathsf {T}} \hat {\beta}) = 0.
$$

The fitted values equal

$$
\hat {y} _ {i} = x _ {i} ^ {\mathsf {T}} \hat {\beta} (i = 1, \dots , n).
$$

Using the matrix notation

$$
X = \left( \begin{array}{c} x _ {1} ^ {\mathsf {T}} \\ \vdots \\ x _ {n} ^ {\mathsf {T}} \end{array} \right), \quad Y = \left( \begin{array}{c} y _ {1} \\ \vdots \\ y _ {n} \end{array} \right),
$$

we can write the OLS coefficient as

$$
\hat {\beta} = (X ^ {\mathsf {T}} X) ^ {- 1} X ^ {\mathsf {T}} Y
$$

and the fitted vector as

$$
\hat {Y} = X \hat {\beta} = X (X ^ {\mathsf {T}} X) ^ {- 1} X ^ {\mathsf {T}} Y.
$$

Define the hat matrix as

$$
H = X (X ^ {\mathsf {T}} X) ^ {- 1} X ^ {\mathsf {T}}.
$$

Then we also have $\hat { Y } = H Y$ , justifying the name “hat matrix.”

Assuming finite fourth moments of $( x , y )$ , we can use the law of large numbers and the central limit theorem to show that

$$
\sqrt {n} (\hat {\beta} - \beta) \rightarrow \mathrm{N} (0, V = B ^ {- 1} M B ^ {- 1})
$$

in distribution, where $\boldsymbol { B } = \boldsymbol { E } ( \boldsymbol { x } \boldsymbol { x } ^ { \intercal } )$ and $M = E ( \varepsilon ^ { 2 } x x ^ { \mathsf { T } } )$ . So a moment estimator for the asymptotic variance of $\hat { \beta }$ is

$$
\hat {V} _ {\mathrm{EHW}} = n ^ {- 1} \left(n ^ {- 1} \sum_ {i = 1} ^ {n} x _ {i} x _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \left(n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\varepsilon} _ {i} ^ {2} x _ {i} x _ {i} ^ {\mathsf {T}}\right) \left(n ^ {- 1} \sum_ {i = 1} ^ {n} x _ {i} x _ {i} ^ {\mathsf {T}}\right) ^ {- 1},
$$

which is called the Eicker–Huber–White (EHW) robust covariance estimator (Eicker, 1967; Huber, 1967; White, 1980). We can show that $n \hat { V } _ { \mathrm { E H W } } \to V$ in probability. Based on $\hat { \beta }$ and $\hat { V } _ { \mathrm { E H W } }$ , we can make inference about the population OLS coefficient $\beta .$ In $\mathbb { R } ,$ the lm function can compute ${ \hat { \boldsymbol { \beta } } } ,$ and the hccm function in the package car can compute $\hat { V } _ { \mathrm { E H W } }$ .

There are many variants of the EHW robust covariance estimator (Long and Ervin, 2000). In particular, the HC1 variant modifies $\hat { \varepsilon } _ { i } ^ { 2 }$ to $\hat { \varepsilon } _ { i } ^ { 2 } / ( n - p )$ , the HC2 variant modifies $\hat { \varepsilon } _ { i } ^ { 2 }$ to $\hat { \varepsilon } _ { i } ^ { 2 } / ( 1 - h _ { i i } )$ , the HC3 variant modifies $\hat { \varepsilon } _ { i } ^ { 2 }$ to $\hat { \varepsilon } _ { i } ^ { 2 } / ( 1 - h _ { i i } ) ^ { 2 }$ , in the definition of $\hat { V } _ { \mathrm { E H W } }$ , where $h _ { i i }$ is the (i, i)th diagonal element of $H$ , also called the leverage scores.

## A2.3 Frisch–Waugh–Lovell Theorem

The Frisch–Waugh–Lovell (FWL) theorem has two versions: one at the population level and the other at the sample level. It reduces multivariate OLS to univariate OLS and therefore facilitate the understanding and calculation of the OLS coefficients. Below I will present special cases of the FWL Theorem which are enough for this book.

Theorem A2.1 (population FWL) The coefficient of x1 in the OLS fit of y on $( x _ { 1 } , x _ { 2 } , \ldots , x _ { p } )$ equals the coefficient of $\tilde { x } _ { 1 }$ in the OLS fit of y or $\tilde { y }$ on $\tilde { x } _ { 1 }$ , where y˜ is the residual from the OLS fit of y on $( x _ { 2 } , \ldots , x _ { p } )$ and ${ \tilde { x } } _ { 1 }$ is the residual from the OLS fit of $x _ { 1 }$ on $( x _ { 2 } , \ldots , x _ { p } )$ .

In Theorem A2.1, residualizing $x _ { 1 }$ is crucial but residualizing y is not.

Theorem A2.2 (sample FWL) With data $( Y , X _ { 1 } , X _ { 2 } , \ldots , X _ { p } )$ containing column vectors, the coefficient of $X _ { 1 }$ equals the coefficient of $\tilde { X } _ { 1 }$ in the OLS fit of Y or Y˜ on $\tilde { X } _ { 1 }$ , where $\tilde { Y }$ is the residual vector from the OLS fit of Y on $( X _ { 2 } , \ldots , X _ { p } )$ and $\tilde { X } _ { 1 }$ is the residual from the OLS fit of $X _ { 1 }$ on $( X _ { 2 } , \ldots , X _ { p } )$ .

Again, in Theorem A2.2, residualizing $X _ { 1 }$ is crucial but residualizing Y is not.

## A2.4 Linear model

Sometimes, we impose a stronger model assumption which requires the conditional mean of y given x is linear:

$$
E (y \mid x) = x ^ {\mathsf {T}} \beta
$$

or, equivalently,

$$
y = x ^ {\mathsf {T}} \beta + \varepsilon , \qquad E (\varepsilon \mid x) = 0,
$$

which is called the restricted mean model. Under this model, the population OLS coefficient is the true parameter of interest:

$$
\begin{array}{l} \left\{E (x x ^ {\mathsf {T}}) \right\} ^ {- 1} E (x y) = \left\{E (x x ^ {\mathsf {T}}) \right\} ^ {- 1} E \left\{x E (y \mid x) \right\} \\ = \left\{E (x x ^ {\mathsf {T}}) \right\} ^ {- 1} E (x x ^ {\mathsf {T}} \beta) \\ = \beta . \\ \end{array}
$$

Moreover, the population OLS coefficient does not depend on the distribution of x. The asymptotic inference in Section A2.1 applies to this model too.

In the special case with $\operatorname { v a r } ( \varepsilon \mid x ) = \sigma ^ { 2 }$ , the asymptotic variance of the OLS coefficient reduces to

$$
V = \sigma^ {2} \{E (x x ^ {\mathsf {T}}) \} ^ {- 1}
$$

so a simpler moment estimator for the asymptotic variance of $\hat { \beta }$ is

$$
\hat {V} _ {\mathrm{OLS}} = \hat {\sigma} ^ {2} \left(\sum_ {i = 1} ^ {n} x _ {i} x _ {i} ^ {\intercal}\right) ^ {- 1}
$$

$\begin{array} { r } { \hat { \sigma } ^ { 2 } = ( n - p ) ^ { - 1 } \sum _ { i = 1 } ^ { n } \hat { \varepsilon } _ { i } ^ { 2 } } \end{array}$ the lm function.

## A2.5 Weighted least squares

Assuming that $( w _ { i } , x _ { i } , y _ { i } ) \stackrel { \mathrm { I I D } } { \sim } ( w , x , y )$ with $w \ne 0 .$ At the population level, we can define weighted least squares (WLS) coefficient as

$$
\beta_ {w} = \arg \min _ {b} E \{w (y - x ^ {\mathsf {T}} b) ^ {2} \},
$$

which satisfies

$$
E \{w x (y - x ^ {\mathsf {T}} \beta_ {w}) \} = 0
$$

and thus equals

$$
\beta_ {w} = \{E (w x x ^ {\mathsf {T}}) \} ^ {- 1} E (w x y)
$$

if $E ( w x x ^ { \mathsf { T } } )$ is invertible.

At the sample level, we can define the WLS coefficient as

$$
\hat {\beta} _ {w} = \arg \min _ {b} \sum_ {i = 1} ^ {n} w _ {i} (y _ {i} - x _ {i} ^ {\mathsf {T}} b) ^ {2},
$$

which satisfies

$$
\sum_ {i = 1} ^ {n} w _ {i} x _ {i} (y _ {i} - x _ {i} ^ {\mathsf {T}} \hat {\beta} _ {w}) = 0
$$

and thus equals

$$
\hat {\beta} _ {w} = \left(n ^ {- 1} \sum_ {i = 1} ^ {n} w _ {i} x _ {i} x _ {i} ^ {\intercal}\right) ^ {- 1} \left(n ^ {- 1} \sum_ {i = 1} ^ {n} w _ {i} x _ {i} y _ {i}\right)
$$

if $\scriptstyle \sum _ { i = 1 } ^ { n }$ wixixTi is invertible.

## A2.6 Logistic regression

## A2.6.1 Model

Technically, we can use apply the OLS procedure even $\mathrm { i f }$ the outcome y is binary. However, it is a little awkward to have predicted probabilities outside the range of [0, 1]. This motivates us to consider the following model:

$$
\operatorname{pr} (y _ {i} = 1 \mid x _ {i}) = g (x _ {i} ^ {\mathsf {T}} \beta),
$$

where $g ( \cdot ) : \mathbb { R }  [ 0 , 1 ]$ is a monotone function, and its inverse is often called the link function. The $g ( \cdot )$ function can be any distribution function of a random variable, but we will focus on the logistic form:

$$
g (z) = \frac {e ^ {z}}{1 + e ^ {z}} = (1 + e ^ {- z}) ^ {- 1}.
$$

We can also write the logistic model as

$$
\operatorname{pr} (y _ {i} = 1 \mid x _ {i}) \equiv \pi (x _ {i}, \beta) = \frac {e ^ {x _ {i} ^ {\top} \beta}}{1 + e ^ {x _ {i} ^ {\top} \beta}},
$$

or, equivalently,

$$
\operatorname{logit} \left\{\operatorname{pr} (y _ {i} = 1 \mid x _ {i}) \right\} \equiv \log \frac {\operatorname{pr} (y _ {i} = 1 \mid x _ {i})}{1 - \operatorname{pr} (y _ {i} = 1 \mid x _ {i})} = x _ {i} ^ {\top} \beta .
$$

Assume that $x _ { i 1 }$ is binary. Under the logistic model, we have

$$
\begin{array}{l} \beta_ {1} = \operatorname{logit} \left\{\operatorname{pr} (y _ {i} = 1 \mid x _ {i 1} = 1, \dots) \right\} - \operatorname{logit} \left\{\operatorname{pr} (y _ {i} = 1 \mid x _ {i 1} = 0, \dots) \right\} \\ = \log \frac {\operatorname{pr} (y _ {i} = 1 \mid x _ {i 1} = 1 , \ldots) / \operatorname{pr} (y _ {i} = 0 \mid x _ {i 1} = 1 , \ldots)}{\operatorname{pr} (y _ {i} = 1 \mid x _ {i 1} = 0 , \ldots) / \operatorname{pr} (y _ {i} = 0 \mid x _ {i 1} = 0 , \ldots)}, \\ \end{array}
$$

where $\cdot \cdot \cdot$ contains all other regressor $x _ { i 2 } , \ldots , x _ { i p }$ . Therefore, the coefficient $\beta _ { 1 }$ equals the log odds ratio of $x _ { i 1 }$ on $y _ { i }$ conditional on other regressors.

## A2.6.2 Maximum likelihood estimate

To estimate the parameter $\beta ,$ we can maximize the following likelihood function:

$$
\begin{array}{l} L (\beta) = \prod_ {i = 1} ^ {n} \left\{\pi (x _ {i}, \beta) \right\} ^ {y _ {i}} \left\{1 - \pi (x _ {i}, \beta) \right\} ^ {1 - y _ {i}} \\ = \prod_ {i = 1} ^ {n} \left\{\frac {\pi (x _ {i} , \beta)}{1 - \pi (x _ {i} , \beta)} \right\} ^ {y _ {i}} \left\{1 - \pi (x _ {i}, \beta) \right\} \\ = \prod_ {i = 1} ^ {n} \left(e ^ {x _ {i} ^ {\intercal} \beta}\right) ^ {y _ {i}} \frac {1}{1 + e ^ {x _ {i} ^ {\intercal} \beta}} \\ = \prod_ {i = 1} ^ {n} \frac {e ^ {y _ {i} x _ {i} ^ {\top} \beta}}{1 + e ^ {x _ {i} ^ {\top} \beta}}. \\ \end{array}
$$

Let $\hat { \beta }$ denote the maximizer, which is called the maximum likelihood estimate (MLE). Taking the log of $L ( \beta )$ and differentiating it with respect to $\beta ,$ we can show that the MLE must satisfy the first order condition:

$$
\sum_ {i = 1} ^ {n} x _ {i} \{y _ {i} - \pi (x _ {i}, \hat {\beta}) \} = 0.
$$

So if $x _ { i }$ contains an intercept, the MLE must satisfy

$$
\sum_ {i = 1} ^ {n} \{y _ {i} - \pi (x _ {i}, \hat {\beta}) \} = 0,
$$

that ${ \mathrm { i s } } ,$ the average of the observed $y _ { i } \mathrm { \dot { s } }$ must be identical to the average of the fitted probabilities $\pi ( x _ { i } , \hat { \beta } ) \mathrm { { ^ { * } s } }$ .

Using the general theory for the MLE, we can show that it is consistent for the true parameter $\beta$ and is asymptotically normal:

$$
\sqrt {n} (\hat {\beta} - \beta) \rightarrow \mathrm{N} (0, V)
$$

in distribution, where $V = E \left[ \pi ( x _ { i } , \beta ) \{ 1 - \pi ( x _ { i } , \beta ) \} x x ^ { \mathsf { T } } \right]$ . So we can approximate the covariance matrix of $\hat { \beta }$ by

$$
n ^ {- 1} \sum_ {i = 1} ^ {n} \pi (x _ {i}, \hat {\beta}) \{1 - \pi (x _ {i}, \hat {\beta}) \} x _ {i} x _ {i} ^ {\mathsf {T}}.
$$

In R, the glm function can find the MLE and report the estimated covariance matrix.

## A2.6.3 Extension to the case-control study

In case-control studies, sampling is conditional on the binary outcome, that is, units with outcomes $y _ { i } = 1$ and $y _ { i } = 0$ are sampled with different probabilities. Let $s _ { i }$ be the sampling indicator. In case control studies, we have

$$
\operatorname{pr} (s _ {i} = 1 \mid x _ {i}, y _ {i}) = \operatorname{pr} (s _ {i} = 1 \mid y _ {i})
$$

as a function of $y _ { i } ,$ , and we only observe units with $s _ { i } = 1$ .

Prentice and Pyke (1979) showed that logistic regression is applicable in case-control studies although the above discussion assume IID sampling.

## A2.6.4 Logistic regression with weights

Sometimes, unit i has weight $w _ { i }$ , then we can fit a weighted logistic regression by solving

$$
\sum_ {i = 1} ^ {n} w _ {i} x _ {i} \{y _ {i} - \pi (x _ {i}, \hat {\beta}) \} = 0.
$$

## A2.7 Homework problems

## A2.1 Sample OLS with intercept

Assume the regressor $x _ { i }$ contains an intercept. Show that

$$
\bar {y} = \bar {x} ^ {\mathsf {T}} \hat {\beta}. \tag {A2.3}
$$

## A2.2 Univariate weighed least squares

As a special case of WLS, define

$$
(\hat {\alpha} _ {w}, \hat {\beta} _ {w}) = \arg \min _ {(a, b)} \sum_ {i = 1} ^ {n} w _ {i} (y _ {i} - a - b x _ {i}) ^ {2}
$$

where $w _ { i } \geq 0$ . Show that

$$
\hat {\beta} _ {w} = \frac {\sum_ {i = 1} ^ {n} w _ {i} (x _ {i} - \bar {x} _ {w}) (y _ {i} - \bar {y} _ {w})}{\sum_ {i = 1} ^ {n} w _ {i} (x _ {i} - \bar {x} _ {w}) ^ {2}} \tag {A2.4}
$$

and

$$
\hat {\alpha} _ {w} = \bar {y} _ {w} - \hat {\beta} _ {w} \bar {x} _ {w}, \tag {A2.5}
$$

where $\begin{array} { r } { \bar { x } _ { w } ~ = ~ \sum _ { i = 1 } ^ { n } w _ { i } x _ { i } / \sum _ { i = 1 } ^ { n } w _ { i } } \end{array}$ and $\begin{array} { r } { \bar { y } _ { w } ~ = ~ \sum _ { i = 1 } ^ { n } w _ { i } y _ { i } / \sum _ { i = 1 } ^ { n } w _ { i } } \end{array}$ are the weighted averages of the $x _ { i } { } ^ { \ ' } \mathrm { s }$ and $y _ { i } \mathrm { \dot { s } }$ .

Further assume that the $x _ { i }$ ’s are binary. Show that

$$
\hat {\beta} _ {w} = \frac {\sum_ {i = 1} ^ {n} w _ {i} x _ {i} y _ {i}}{\sum_ {i = 1} ^ {n} w _ {i} x _ {i}} - \frac {\sum_ {i = 1} ^ {n} w _ {i} (1 - x _ {i}) y _ {i}}{\sum_ {i = 1} ^ {n} w _ {i} (1 - x _ {i})}.
$$

That is, if the regressor is binary in the univariate WLS, the coefficient of the regressor equals the difference in the weighted means.

Hint: Think about an appropriate reparametrization of the WLS problem. Otherwise, the derivation can be tedious.

## A2.3 OLS with orthogonal regressors

Consider sample OLS fit of an n-vector $Y$ on an n×p matrix X, with coefficient ${ \hat { \boldsymbol { \beta } } } .$ . Partition X into $\boldsymbol { X } = ( X _ { 1 } , X _ { 2 } )$ , where $X _ { 1 }$ is an $n \times k$ matrix and $X _ { 2 }$ is an $n \times l$ matrix, with $p = k + l .$ Correspondingly, partition $\hat { \beta }$ into

$$
\hat {\beta} = \binom{\hat {\beta} _ {1}}{\hat {\beta} _ {2}}.
$$

Assume $X _ { 1 }$ and $X _ { 2 }$ are orthogonal, that is, $X _ { 1 } ^ { \mathsf { T } } X _ { 2 } \ = \ 0$ . Show that $\hat { \beta } _ { 1 }$ equals the coefficient from OLS of $Y$ on $X _ { 1 }$ and $\hat { \beta } _ { 2 }$ equals the coefficient from OLS of Y on $X _ { 2 } .$ respectively.

## A2.4 OLS with a non-degenerate transformation of the regressors

Define $\hat { \beta }$ as the coefficient from the sample OLS fit of an n-vector Y on an $n \times p$ matrix X. Let Γ be a $, p \times p$ non-degenerate matrix, and define $X ^ { \prime } = X \Gamma$ . Define ${ \hat { \beta } } ^ { \prime }$ as the coefficient from the sample OLS fit of $Y$ on $X ^ { \prime }$ .

Show that

$$
\hat {\beta} = \Gamma \hat {\beta} ^ {\prime}.
$$

## A3
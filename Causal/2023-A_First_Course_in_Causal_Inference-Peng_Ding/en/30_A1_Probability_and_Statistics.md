# A1 Probability and Statistics

## A1.1 Probability

## A1.1.1 Tower property and variance decomposition

Given random variables or vectors $A , B , C ,$ we have

$$
E (A) = E \{E (A \mid B) \}
$$

and

$$
E (A \mid C) = E \{E (A \mid B, C) \mid C \}.
$$

Given a random variable A and random variables or vectors $B , C ,$ we have

$$
\operatorname{var} (A) = E \{\operatorname{var} (A \mid B) \} + \operatorname{var} \{E (A \mid B) \}
$$

and

$$
\operatorname{var} (A \mid C) = E \{\operatorname{var} (A \mid B, C) \mid C \} + \operatorname{var} \{E (A \mid B, C) \mid C \}.
$$

Similarly, we can decompose the covariance as

$$
\operatorname{cov} \left(A _ {1}, A _ {2}\right) = E \left\{\operatorname{cov} \left(A _ {1}, A _ {2} \mid B\right) \right\} + \operatorname{cov} \left\{E \left(A _ {1} \mid B\right), E \left(A _ {2} \mid B\right) \right\}
$$

and

$$
\operatorname{cov} \left(A _ {1}, A _ {2} \mid C\right) = E \left\{\operatorname{cov} \left(A _ {1}, A _ {2} \mid B, C\right) \mid C \right\} + \operatorname{cov} \left\{E \left(A _ {1} \mid B, C\right), E \left(A _ {2} \mid B, C\right) \mid C \right\}.
$$

## A1.1.2 Limiting theorems

Definition A1.1 (convergence in probability) A sequence of random variables $( X _ { n } ) _ { n \geq 1 }$ converges to X in probability, if for every $\varepsilon > 0$ , we have

$$
\operatorname{pr} (| X _ {n} - X | > \varepsilon) \to 0
$$

$$
a s n \rightarrow \infty .
$$

Definition A1.2 (convergence in distribution) A sequence of random variables $( X _ { n } ) _ { n \geq 1 }$ converges to X in distribution, if

$$
\operatorname{pr} (X _ {n} \leq x) \to \operatorname{pr} (X \leq x)
$$

for all continuity point x of pr(X ≤ x), as $n \to \infty$Convergence in probability is stronger than convergence in distribution. Definitions A1.1 and A1.2 are useful for stating the following two fundamental theorems in probability theory.

Theorem A1.1 (law of large numbers) $I f X _ { 1 } , \ldots , X _ { n } \stackrel { I I D } { \sim } X$ with $E | X | <$ $\infty _ { i }$ , then $\begin{array} { r } { \bar { X } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } X _ { i } \to E ( X ) } \end{array}$ in probability.

The law of large numbers in Theorem A1.1 states that the sample average is close to the population mean in the limit.

Theorem A1.2 (central limit theorem) If $\begin{array} { r l } { X _ { 1 } , \ldots , X _ { n } \quad { \stackrel { I I D } { \sim } } } & { { } X } \end{array}$ with va $\operatorname { \lrcorner } ( X ) < \infty$ , then

$$
\frac {\bar {X} - E (X)}{\sqrt {\operatorname{var} (X) / n}} \to \mathrm{N} (0, 1)
$$

in distribution.

The central limit theorem in Theorem A1.2 states that the standardized sample average is close to a standard Normal random variable in the limit.

Theorems A1.1 and A1.2 assume IID random variables for convenience. There are also many law of large numbers and central limit theorems for the sample mean of independent random variable (e.g., Durrett, 2019).

## A1.1.3 Delta method

Delta method is a power tool to derive asymptotic Normality of nonlinear functions of an asymptotically Normal random vector. I review a special case of delta method below.

Theorem A1.3 (delta method) Assume ${ \sqrt { n } } ( X _ { n } - \mu ) \to \mathrm { N } ( 0 , \Sigma )$ in distribution and the function g(x) has non-zero derivative $\nabla g ( \mu )$ at $\mu$ . Then

$$
\sqrt {n} \{g (X _ {n}) - g (\mu) \} \rightarrow \mathrm{N} \left(0, (\nabla g (\mu) ^ {\mathsf {T}} \Sigma \nabla g (\mu)\right)
$$

in distribution.

I will omit the proof of Theorem A1.3. It is intuitive based on the first-order Taylor expansion:

$$
g (X _ {n}) - g (\mu) \cong (\nabla g (\mu) ^ {\mathsf {T}} (X _ {n} - \mu).
$$

A leading example of delta method is to obtain the asymptotic Normality of ratio.

Example A1.1 (asymptotic normality for ratio) Assume

$$
\sqrt {n} \binom {Y _ {n} - \mu_ {Y}} {X _ {n} - \mu_ {X}} \rightarrow \mathrm{N} \left(\binom {0} {0}, \left(\begin{array}{c c}\sigma_ {Y} ^ {2}&\sigma_ {Y X}\\\sigma_ {Y X}&\sigma_ {X} ^ {2}\end{array}\right)\right) \tag {A1.1}
$$

in distribution with $\mu _ { X } \neq 0 .$ Apply Theorem A1.3 to obtain that

$$
\sqrt {n} \left(\frac {Y _ {n}}{X _ {n}} - \frac {\mu_ {Y}}{\mu_ {X}}\right)\rightarrow \mathrm{N} \left(0, \frac {\sigma_ {Y} ^ {2}}{\mu_ {X} ^ {2}} + \frac {\mu_ {Y} ^ {2} \sigma_ {X} ^ {2}}{\mu_ {X} ^ {4}} - \frac {2 \mu_ {Y} \sigma_ {Y X}}{\mu_ {X} ^ {3}}\right) \tag {A1.2}
$$

in distribution. In the special case that $X _ { n }$ and $Y _ { n }$ are asymptotically independent, the asymptotic variance of $Y _ { n } / X _ { n }$ simplifies to $\sigma _ { Y } ^ { 2 } / \mu _ { X } ^ { 2 } + \mu _ { Y } ^ { 2 } \sigma _ { X } ^ { 2 } / \mu _ { X } ^ { 4 }$ . I leave the details to Problem A1.2.

The asymptotic variance in Example A1.1 is a little cumbersome. An easier way to memorize it is based on the following approximation:

$$
\frac {Y _ {n}}{X _ {n}} - \frac {\mu_ {Y}}{\mu_ {X}} = \frac {Y _ {n} - \mu_ {Y} / \mu_ {X} \cdot X _ {n}}{X _ {n}} \cong \frac {Y _ {n} - \mu_ {Y} / \mu_ {X} \cdot X _ {n}}{\mu_ {X}}, \tag {A1.3}
$$

so the asymptotic variance of the ratio equals the asymptotic variance of

$$
\frac {Y _ {n} - \mu_ {Y} / \mu_ {X} \cdot X _ {n}}{\mu_ {X}},
$$

which is a linear combination of $Y _ { n }$ and $X _ { n } .$ Slutsky’s theorem can make the approximation in (A1.3) rigorous; it is beyond this book.

Example A1.2 (asymptotic normality for product) Assume (A1.1). $A p \ / -$ ply Theorem A1.3 to obtain that

$$
\sqrt {n} \left(X _ {n} Y _ {n} - \mu_ {X} \mu_ {Y}\right)\rightarrow \mathrm{N} \left(0, \mu_ {Y} ^ {2} \sigma_ {X} ^ {2} + \mu_ {X} ^ {2} \sigma_ {Y} ^ {2} + 2 \mu_ {X} \mu_ {Y} \sigma_ {X Y}\right) \tag {A1.4}
$$

in distribution. In the special case that $X _ { n }$ and $Y _ { n }$ are asymptotically independent, the asymptotic variance of $X _ { n } Y _ { n }$ simplifies to $\mu _ { Y } ^ { 2 } \sigma _ { X } ^ { 2 } + \mu _ { X } ^ { 2 } \sigma _ { Y } ^ { 2 }$ . I leave the details to Problem A1.3.

## A1.2 Statistical inference

## A1.2.1 Point estimation

Assume that θ is the parameter of interest. Oftentimes, the problem also contain other parameters not of interest, denoted by $\eta .$ Statisticians call $\eta$ the nuisance parameter. Based on data, we can compute an estimator ˆθ. Throughout this book, we take the frequentist’s perspective by assuming that $\theta$ is a fixed number and $\hat { \theta }$ is random due to the randomness of data. Two basic requirements for an estimator are below.

Definition A1.3 (unbiasedness) The estimator $\hat { \theta }$ is unbiased for θ if

$$
E (\hat {\theta}) = \theta
$$

for all possible values of θ and $\eta .$ .

Definition A1.4 (consistency) The estimator $\hat { \theta }$ is consistent for θ if

$$
\hat {\theta} \rightarrow \theta
$$

in probability as the sample size approaches to infinity, for all possible values of θ and η.

Unbiasedness requires that the mean of the estimator is identical to the parameter of interest. Consistency requires that the estimator is close to the true parameter in the limit. Unbiased does not imply consistency, and consistency does not imply unbiasedness either. Unbiasedness can be restrictive because it is impossible even in some simple statistics problems. Consistency is often the basic requirement in most statistics problems.

## A1.2.2 Confidence interval

A point estimator $\hat { \theta }$ is a random variable which differs from the true parameter. Statisticians are often interested in finding an interval that covers the true parameter with certain given probability. This interval is computed based on the data, and it is random.

Definition A1.5 (confidence interval) A data-dependent interval $[ \hat { \theta } _ { \mathrm { L } } , \hat { \theta } _ { \mathrm { U } } ]$ is a confidence interval for θ with coverage probability 1 − α if

$$
\operatorname{pr} (\hat {\theta} _ {\mathrm{L}} \leq \theta \leq \hat {\theta} _ {\mathrm{U}}) \geq 1 - \alpha .
$$

Definition A1.6 (asymptotic confidence interval) A data-dependent interval $[ \hat { \theta } _ { \mathrm { L } } ] , \hat { \theta } _ { \mathrm { U } } ]$ is an asymptotic confidence interval for θ with coverage probability $1 - \alpha \ i f$

$$
\mathrm{pr} (\hat {\theta} _ {\mathrm{L}} \leq \theta \leq \hat {\theta} _ {\mathrm{U}}) \rightarrow 1 - \alpha^ {\prime}
$$

with $\alpha ^ { \prime } \geq \alpha ,$ as $n \to \infty$ .

A standard choice is $\alpha = 0 . 0 5$ . In Definitions A1.5 and A1.6, the coverage probabilities can be larger than the nominal level 1−α. That is, the definitions allow for over overage but do not allow for under coverage. With over coverage, we say that the confidence interval is conservative. Of course, we hope the confidence interval to be as narrow as possible. Otherwise, the definition of confidence interval can be arbitrary.

## A1.2.3 Hypothesis testing

Many applied problems can be formulated as testing a hypothesis:

$$
H _ {0}: \theta = 0.
$$

The decision rule ϕ is a binary function of the data: $\phi = 1$ if we reject $H _ { 0 } ; \phi = 0$ if we fail to reject $H _ { 0 }$ . The type one error rate of the test is the probability of rejection if the null hypothesis holds. I review the definition below.

Definition A1.7 When $H _ { 0 }$ holds, define the type one error rate of the test ϕ as the maximum possible value of the probability

$$
\operatorname{pr} (\phi = 1).
$$

A standard choice is to make sure that the type one error rate is below $\alpha = 0 . 0 5$ . The type two error rate of the test is the probability of no rejection if the null hypothesis does not hold. I review the definition below.

Definition A1.8 When $H _ { 0 }$ does not hold, define the type two error rate of the test ϕ as the probability

$$
\operatorname{pr} (\phi = 0).
$$

Given the control of the type one error rate under $H _ { 0 }$ , we hope the type two error rate is as low as possible when $H _ { 0 }$ does not hold.

## A1.2.4 Wald-type confidence interval and test

Many statistics problems have the following structure. The parameter of interest is θ. We first find a consistent estimator ˆθ that converges in probability to $\theta ,$ and show that it is asymptotically Normal with mean θ and variance v which may depends on θ as well as other parameters. We then find a consistent estimator ˆv for v, based on analytic formulas or the bootstrap reviewed in Chapter A1.5. We finally construct the Wald-type confidence interval for θ as

$$
\hat {\theta} \pm z _ {1 - \alpha / 2} \sqrt {\hat {v}}
$$

which covers θ with probability approximately $1 - \alpha$ . When this interval excludes a particular $c ,$ for example, $c = 0 .$ , we reject the null hypothesis $H _ { 0 } ( c ) : \theta = c ,$ which is called the Wald test.

## A1.2.5 Duality between constructing confidence sets and testing null hypotheses

Consider the statistical inference problem for a scalar parameter θ. A fundamental result in statistics is that constructing confidence sets for θ is equivalent to testing null hypotheses about θ. This is often called the duality between constructing confidence sets and testing null hypotheses.

Section A1.2.4 has reviewed the duality based on the Wald-type confidence interval and test. The duality also holds in general. Assume that Θ is aˆ $( 1 - \alpha )$ - level confidence set for θ:

$$
\operatorname{pr} (\theta \in \hat {\Theta}) = 1 - \alpha .
$$

Then we can reject the null hypothesis $H _ { 0 } ( c ) : \theta = c$ if c is not in the set $\hat { \Theta }$ . This is a valid test because when θ indeed equals c, we have correct type one error rate $\operatorname { p r } ( \theta \not \in { \hat { \Theta } } ) = \alpha$ . Conversely, if we test a sequence of null hypotheses$H _ { 0 } ( c ) : \theta = c ,$ we can obtain the corresponding p-values, $p ( c )$ , as a function of c. Then the values of c that we fail to reject at level α form a confidence set for θ:

$$
\hat {\Theta} = \{c: p (c) \geq \alpha \} = \{c: \text {   fail   to   reject   } H _ {0} (c) \text {   at   level   } \alpha \}.
$$

It is a valid confidence set because

$$
\operatorname{pr} (\theta \in \hat {\Theta}) = \operatorname{pr} \{\text {   fail   to   reject   } H _ {0} (\theta) \text {   at   level   } \alpha \} = 1 - \alpha .
$$

Here I use “confidence set” instead of “confidence interval” because $\hat { \Theta }$ based on inverting tests may not be an interval. See the use of the duality in Sections A1.4.2 and 3.6.1.

## A1.3 Inference with $2 \times 2$ tables

## A1.3.1 Fisher’s exact test

Fisher proposed an exact test for $H _ { 0 } : p _ { 1 } = p _ { 0 }$ under the statistical model:

$$
n _ {1 1} \sim \operatorname{Binomial} (n _ {1}, p _ {1}), \quad n _ {0 1} \sim \operatorname{Binomial} (n _ {0}, p _ {0}), \quad n _ {1 1} \perp n _ {0 1}.
$$

The table below summarizes the data.

<table><tr><td></td><td>1</td><td>0</td><td>row sum</td></tr><tr><td>sample 1</td><td> $n_{11}$ </td><td> $n_{10}$ </td><td> $n_1$ </td></tr><tr><td>sample 0</td><td> $n_{01}$ </td><td> $n_{00}$ </td><td> $n_0$ </td></tr><tr><td>column sum</td><td> $n_{.1}$ </td><td> $n_{.0}$ </td><td> $n$ </td></tr></table>

He argued that the sum $n _ { 1 1 } + n _ { 0 1 } \equiv n _ { \cdot 1 }$ contains little information for the difference between $p _ { 1 }$ and $p _ { 0 } .$ , and $n _ { 1 1 }$ conditioning on the sum has Hypergeometric distribution that does not depend on the unknown parameter $p _ { 1 } = p _ { 0 }$ under $H _ { 0 } { \mathrm { : } }$

$$
\operatorname{pr} (n _ {1 1} = k) = \frac {\binom {n . _ {1}} {k} \binom {n - n . _ {1}} {n _ {1} - k}}{\binom {n} {n _ {1}}}.
$$

In R, the function fisher.test implement this test.

## A1.3.2 Estimation with $2 \times 2$ tables

Based on the model in Section A1.3.1, we can estimate the parameters $p _ { 1 }$ and p0 by sample frequencies:

$$
\hat {p} _ {1} = \frac {n _ {1 1}}{n _ {1}}, \quad \hat {p} _ {0} = \frac {n _ {0 1}}{n _ {0}}.
$$

Therefore, we can estimate the risk difference, log risk ratio, and log odds ratio

$$
\begin{array}{l} \mathrm{RD} = p _ {1} - p _ {0}, \\ \log \mathrm{RR} = \log \frac {p _ {1}}{p _ {0}}, \\ \log \mathrm{OR} = \log \frac {p _ {1} / (1 - p _ {1})}{p _ {0} / (1 - p _ {0})} \\ \end{array}
$$

by the sample analogues

$$
\begin{array}{l} \hat {\mathrm{RD}} = \hat {p} _ {1} - \hat {p} _ {0}, \\ \log \hat {\mathrm{R} \mathrm{R}} = \log \frac {\hat {p} _ {1}}{\hat {p} _ {0}}, \\ \log \hat {\mathrm{OR}} = \log \frac {\hat {p} _ {1} / (1 - \hat {p} _ {1})}{\hat {p} _ {0} / (1 - \hat {p} _ {0})} = \log \frac {n _ {1 1} n _ {0 0}}{n _ {1 0} n _ {0 1}}. \\ \end{array}
$$

Based on the asymptotic approximation (see Problem A1.4), the estimated variance for the above parameters are

$$
\begin{array}{l} \frac {\hat {p} _ {1} (1 - \hat {p} _ {1})}{n _ {1}} + \frac {\hat {p} _ {0} (1 - \hat {p} _ {0})}{n _ {0}}, \\ \frac {1 - \hat {p} _ {1}}{n _ {1} \hat {p} _ {1}} + \frac {1 - \hat {p} _ {0}}{n _ {0} \hat {p} _ {0}}, \\ \frac {1}{n _ {1} \hat {p} _ {1} (1 - \hat {p} _ {1})} + \frac {1}{n _ {0} \hat {p} _ {0} (1 - \hat {p} _ {0})}, \\ \end{array}
$$

respectively. The log transformation above yields better Normal approximations because the risk ratio and odds ratio are always positive.

## A1.4 Two famous problems in statistics

## A1.4.1 Behrens–Fisher problem

Consider the two-sample problem with $n _ { 1 }$ units under the treatment and $n _ { 0 }$ units under the control, respectively. Assume the outcomes under the treatment $\{ Y _ { i } : Z _ { i } = 1 \}$ are IID from $\mathrm { N } ( \mu _ { 1 } , \sigma _ { 1 } ^ { 2 } )$ and the outcomes under the control $\{ Y _ { i } : Z _ { i } = 0 \}$ are IID from $\mathrm { N } ( \mu _ { 0 } , \sigma _ { 0 } ^ { 2 } )$ , respectively. The goal is to test $H _ { 0 } : \mu _ { 1 } = \mu _ { 0 }$ .

Start with the easier case with $\sigma _ { 1 } ^ { 2 } = \sigma _ { 0 } ^ { 2 }$ . Coherent with Chapter $^ { 3 , }$ let $\hat { \bar { Y } } ( 1 )$ and $\hat { \bar { Y } } ( 0 )$ denote the sample means of the outcomes under the treatment and control, respectively. A standard result is that

$$
t _ {\mathrm{equal}} = \frac {\hat {\bar {Y}} (1) - \hat {\bar {Y}} (0)}{\sqrt {\frac {n}{n _ {1} n _ {0} (n - 2)} \left[ \sum_ {Z _ {i} = 1} \{Y _ {i} - \hat {\bar {Y}} (1) \} ^ {2} + \sum_ {Z _ {i} = 0} \{Y _ {i} - \hat {\bar {Y}} (0) \} ^ {2} \right]}} \sim t _ {n - 2}.
$$

Based on $t _ { \mathrm { e q u a l } }$ , we can construct a test for $H _ { 0 }$ .

Now consider the more difficult case with possibly different $\sigma _ { 1 } ^ { 2 }$ and $\sigma _ { 0 } ^ { 2 } .$ . The distribution of $t _ { \mathrm { e q u a l } }$ is no longer $t _ { n - 2 }$ . Estimating the variances separately, we can also define

$$
t _ {\mathrm{unequal}} = \frac {\hat {\bar {Y}} (1) - \hat {\bar {Y}} (0)}{\sqrt {\frac {\hat {S} ^ {2} (1)}{n _ {1}} + \frac {\hat {S} ^ {2} (0)}{n _ {0}}}},
$$

where

$$
\hat {S} ^ {2} (1) = (n _ {1} - 1) ^ {- 1} \sum_ {Z _ {i} = 1} \{Y _ {i} - \hat {\bar {Y}} (1) \} ^ {2}, \quad \hat {S} ^ {2} (0) = (n _ {0} - 1) ^ {- 1} \sum_ {Z _ {i} = 0} \{Y _ {i} - \hat {\bar {Y}} (0) \} ^ {2}
$$

are the sample variances of the outcomes under the treatment and control, respectively. Unfortunately, the exact distribution of $t _ { \mathrm { u n e q u a l } }$ depends on the known variances. Testing $H _ { 0 }$ without assuming equal variances is the famous Behrens–Fisher problem. With large sample sizes $n _ { 1 }$ and $n _ { 0 }$ , the central limit theorem ensures that $t _ { \mathrm { u n e q u a l } }$ is approximately $\mathrm { { N } } ( 0 , 1 )$ . So we can construct approximate test for $H _ { 0 }$ .

## A1.4.2 Fieller–Creasy problem

Consider the two-sample problem with $n _ { 1 }$ units under the treatment and $n _ { 0 }$ units under the control, respectively. Assume the outcomes under the treatment $\{ Y _ { i } : Z _ { i } = 1 \}$ are IID from $\mathrm { { N } } ( \mu _ { 1 } , 1 )$ and the outcomes under the control $\{ Y _ { i } : Z _ { i } = 0 \}$ are IID from $\mathrm { { N } } ( \mu _ { 0 } , 1 )$ , respectively. The goal is to estimate $\gamma = \mu _ { 1 } / \mu _ { 0 }$ . We can use $\hat { \gamma } = \hat { \bar { Y } } ( 1 ) / \hat { \bar { Y } } ( 0 )$ to estimate $\gamma .$ But the point estimator has a complicated distribution, which does not yield a simple procedure to construct the confidence interval for $\gamma .$ .

Fieller’s confidence interval can be formulated as inverting tests for a sequence of null hypotheses: $H _ { 0 } ( c ) : \gamma = c$ . Under $H _ { 0 } ( c )$ , we have

$$
\frac {\hat {\bar {Y}} (1) - c \hat {\bar {Y}} (0)}{\sqrt {1 / n _ {1} + c ^ {2} / n _ {0}}} \sim \mathrm{N} (0, 1)
$$

which motivates the confidence interval

$$
\left\{c: \left| \frac {\hat {\bar {Y}} (1) - c \hat {\bar {Y}} (0)}{\sqrt {1 / n _ {1} + c ^ {2} / n _ {0}}} \right| \leq z _ {\alpha} \right\}
$$

where $z _ { \alpha }$ is the upper $1 - \alpha / 2$ quantile of a standard Normal random variable.

## A1.5 Bootstrap

It is often very tedious to derive the variance formulas for complex estimators. Efron (1979) proposed the bootstrap as a general tool for variance estimation. There are many versions of the bootstrap (Davison and Hinkley, 1997). In this book, we only need the most basic one: the nonparametric bootstrap, which will be simply called the bootstrap.

Consider the generic setting with

$$
Y _ {1}, \ldots , Y _ {n} \stackrel {\mathrm{IID}} {\sim} Y,
$$

where $Y _ { i }$ can be a general random element denoting the observed data for unit i. An estimator $\hat { \theta }$ is a function of the observed data: ${ \hat { \theta } } = T ( Y _ { 1 } , \ldots , Y _ { n } )$ . When $T$ is a complex function, it may not be easy to obtain the variance or asymptotic variance of ${ \hat { \theta } } .$ .

The uncertainty of $\hat { \theta }$ is driven by the IID sampling of $Y _ { 1 } , \dots , Y _ { n }$ from the true distribution. Although the true distribution is unknown, it can be well approximated by its empirical version

$$
\hat {F} _ {n} (y) = n ^ {- 1} \sum_ {i = 1} ^ {n} I (Y _ {i} \leq y),
$$

when the sample size n is large. If we believe this approximation, we can simulate $\hat { \theta }$ by sampling

$$
(Y _ {1} ^ {*}, \dots , Y _ {n} ^ {*}) \stackrel {\mathrm{IID}} {\sim} \hat {F} _ {n} (y).
$$

Because $\hat { F } _ { n } ( y )$ is a discrete distribution with mass $1 / n$ on each observed data point, the simulation of $\hat { \theta }$ reduces to the following procedure:

1. sample $( Y _ { 1 } ^ { * } , \ldots , Y _ { n } ^ { * } )$ from $\{ Y _ { 1 } , \ldots , Y _ { n } \}$ with replacement;  
2. compute $\hat { \theta } ^ { * } = T ( Y _ { 1 } ^ { * } , \ldots , Y _ { n } ^ { * } )$ ;  
3. repeat the above two steps B times to obtain the bootstrap replicates $\{ \hat { \theta } _ { 1 } ^ { * } , \dots , \hat { \theta } _ { B } ^ { * } \}$ .

We can then approximate the (asymptotic) variance of $\hat { \theta }$ by the sample variance of the bootstrap replicates:

$$
\hat {V} _ {\mathrm{boot}} = (B - 1) ^ {- 1} \sum_ {b = 1} ^ {B} (\hat {\theta} _ {b} ^ {*} - \bar {\theta} ^ {*}) ^ {2},
$$

$\bar { \theta } ^ { * } \ = \ B ^ { - 1 } \sum _ { b = 1 } ^ { B } \hat { \theta } _ { b } ^ { * }$ Normal approximation is then

$$
\hat {\theta} \pm z _ {1 - \alpha / 2} \sqrt {\hat {V} _ {\mathrm{boot}}},
$$

where $z _ { 1 - \alpha / 2 }$ is the $1 - \alpha / 2$ upper quantile of $\mathrm { { N } } ( 0 , 1 )$ .

## A1.6 Homework problems

## A1.1 Independent but not IID data

Assume that the $X _ { i } { } ^ { \ ' } \mathrm { s }$ are independent with mean $\mu _ { i }$ and variances $\sigma _ { i } ^ { 2 }$ for $i = 1 , \ldots , n .$ $\mu = n ^ { - 1 } \dot { \sum _ { i = 1 } ^ { n } \mu _ { i } }$ $\hat { \mu } =$ $n ^ { - 1 } \sum _ { i = 1 } ^ { n } X _ { i }$ is unbiased for $\mu$ and find its variance. Show that the usual variance estimator for IID data

$$
\hat {v} = \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (X _ {i} - \hat {\mu}) ^ {2}
$$

is a conservative estimator for the variance of $\hat { \mu }$ in the sense that

$$
E (\hat {v}) - \operatorname{var} (\hat {\mu}) = \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\mu_ {i} - \mu) ^ {2} \geq 0.
$$

Remark: Consider a simpler case with $\mu _ { i } = \mu$ and $\sigma _ { i } ^ { 2 } = \sigma ^ { 2 }$ for all $i =$ $1 , \ldots , n$ . The sample mean is unbiased for $\mu$ with variance $\sigma ^ { 2 } / n$ . Moreover, an unbiased estimator for the variance $\sigma ^ { 2 } / n$ is $\hat { \sigma } ^ { 2 } / n = \hat { v }$ , where $\hat { \sigma } ^ { 2 } = ( n -$ $\textstyle 1 ) ^ { - 1 } \sum _ { i = 1 } ^ { n } ( X _ { i } - { \hat { \mu } } ) ^ { 2 }$ .

## A1.2 Asymptotic Normality of ratio

Prove (A1.2).

## A1.3 Asymptotic Normality of product

Prove (A1.4).

## A1.4 Variance estimators in two-by-two tables

Use delta method to show the variance estimators in Section A1.3.2.
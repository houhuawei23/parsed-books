# Appendix A Some Probability and Statistics

## A.1 Basic Definitions

(i) We denote the underlying probability space by $( \Omega , { \mathcal { F } } , P )$ . Here, $\Omega , { \mathcal { F } } .$ , and P are set, σ -algebra, and probability measure, respectively.  
(ii) We use capital letters for real-valued random variables. For example, $X$ : $( \Omega , \mathcal { F } ) \to ( \mathbb { R } , B _ { \mathbb { R } } )$ is a measurable function, with respect to the Borel $\sigma \mathrm { - }$ algebra. Random vectors are measurable functions $\mathbf { X } : ( \Omega , \mathcal { F } ) \to ( \mathbb { R } ^ { d } , B _ { \mathbb { R } ^ { d } } )$ . We call X non-degenerate if there is no value $\mathbf { c } \in \mathbb { R } ^ { d }$ such that $P ( \mathbf { X } = \mathbf { c } ) = 1$ . For an introduction to measure theory, see, for example, Dudley [2002].  
(iii) We usually denote vectors with bold letters. In a slight abuse of notation, we consider sets of variables $\mathbf { B } \subseteq \mathbf { X }$ as a single multivariate variable.  
(iv) $P _ { \mathbf { X } }$ is the distribution of the d-dimensional random vector X, that is, a probability measure on $( \mathbb { R } ^ { d } , B _ { \mathbb { R } ^ { d } } )$ .  
(v) We write $x \mapsto p _ { X } ( x )$ or simply $x \mapsto p ( x )$ for the density, that is, the Radon-Nikodym derivative of $P _ { X }$ with respect to a product measure. We (sometimes implicitly) assume its existence or continuity.  
(vi) We call X independent of Y and write $X \perp \perp Y$ if and only if

$$
p (x, y) = p (x) p (y) \tag {A.1}
$$

for all $x , y .$ . Otherwise, X and Y are dependent, and we write $X \not \vdash Y$ (vii) We call $X _ { 1 } , \ldots , X _ { d }$ jointly (or mutually) independent if and only if

$$
p (x _ {1}, \dots , x _ {d}) = p (x _ {1}) \cdot \dots \cdot p (x _ {d}) \tag {A.2}
$$

for all $\boldsymbol { x } _ { 1 } , \ldots , \boldsymbol { x } _ { d }$ . If $X _ { 1 } , \ldots , X _ { d }$ are jointly independent, then any pair $X _ { i }$ and $X _ { j }$ with $i \neq j$ are independent, too. The converse does not hold in general: pairwise independence does not imply joint independence.

(viii) We call X independent of Y conditional on $Z$ and write $X \perp \perp Y \mid Z$ if and only if

$$
p (x, y \mid z) = p (x \mid z) p (y \mid z) \tag {A.3}
$$

for all $x , y , z$ such that $p ( z ) > 0$ . Otherwise, X and Y are dependent conditional on Z and we write $X \not \vdash Y | Z$ .

(ix) Conditional independence relations obey the following important rules [e.g., Pearl, 2009, Section 1.1.5]:

$$
\begin{array}{l} X \perp Y | Z \Rightarrow Y \perp X | Z \quad (\text { symmetry }) \\ X \perp Y, W | Z \Rightarrow X \perp Y | Z \quad (\text { decomposition }) \\ X \perp Y, W | Z \Rightarrow X \perp Y | W, Z \quad (\text { weak   union }) \\ X \perp Y | Z \text {   and   } X \perp W | Y, Z \Rightarrow X \perp Y, W | Z \quad (\text { contraction }) \\ X \perp Y | W, Z \text {   and   } X \perp W | Y, Z \Rightarrow X \perp Y, W | Z \quad (\text { intersection }). \\ \end{array}
$$

The existence of a strictly positive density suffices for the intersection property to hold. Necessary and sufficient conditions for the discrete case are provided by Drton et al. [2009b, Exercise 6.6] and by Fink [2011]. Peters [2014] covers the continuous case.

(x) The variance of a random variable X is defined as

$$
\operatorname{var} [ X ] := \mathbb {E} \left[ (X - \mathbb {E} [ X ]) ^ {2} \right] = \mathbb {E} \left[ X ^ {2} \right] - \mathbb {E} [ X ] ^ {2}
$$

if $\mathbb { E } [ X ^ { 2 } ] < \infty$

(xi) We call X and Y uncorrelated if $\mathbb { E } [ X ^ { 2 } ] , \mathbb { E } [ Y ^ { 2 } ] < \infty$ and

$$
\mathbb {E} [ X Y ] = \mathbb {E} [ X ] \mathbb {E} [ Y ],
$$

that is

$$
\rho_ {X, Y} := \frac {\mathbb {E} [ X Y ] - \mathbb {E} [ X ] \mathbb {E} [ Y ]}{\sqrt {\operatorname{var} [ X ] \operatorname{var} [ Y ]}} = 0.
$$

Otherwise, that is, if $\rho _ { X , Y } \neq 0$ , X and Y are correlated. $\rho _ { X , Y }$ is called the correlation coefficient between X and Y .

(xii) If X and Y are independent, then they are uncorrelated:

$$
X \perp Y \Rightarrow \rho_ {X, Y} = 0.
$$

The other direction does not necessarily hold (see Code Snippet A.1). Only in special cases, such as the bivariate Gaussian distribution or binary variables, does the reversed direction hold, too.

(xiii) We say that X and Y are partially uncorrelated given Z if

$$
\rho_ {X, Y \mid Z} := \frac {\rho_ {X , Y} - \rho_ {X , Z} \rho_ {Z , Y}}{\sqrt {(1 - \rho_ {X , Z} ^ {2}) (1 - \rho_ {Z , Y} ^ {2})}} = 0.
$$

The following interpretation of partial correlation is important: $\rho _ { X , Y \mid Z }$ equals the correlation between residuals after linearly regressing X on Z and Y on Z.

(xiv) In general, we have (see Example 7.9)

$$
\rho_ {X, Y \mid Z} = 0 \quad \nRightarrow \quad X \perp \perp Y \mid Z \quad \text { and }
$$

$$
\rho_ {X, Y \mid Z} = 0 \quad \nLeftarrow \quad X \perp \perp Y \mid Z.
$$

(xv) In regression estimation, we are usually given an i.i.d. sample $( \mathbf { X } _ { 1 } , Y _ { 1 } )$ , . . ., $( \mathbf { X } _ { n } , Y _ { n } )$ from a joint distribution $P _ { \mathbf { X } , Y }$ . Our aim is to predict the target Y from the covariates or predictors X. In least squares regression, for example, we are looking for a function $\hat { f }$ such that

$$
\hat {f} = \underset {f \in \mathcal {F}} {\operatorname{argmin}} \sum_ {i = 1} ^ {n} \left(Y _ {i} - f (\mathbf {X} _ {i})\right) ^ {2}.
$$

Here, we optimize over a function class $\mathcal { F }$ (see Section A.3). Different regression techniques use different function classes. In linear regression, we are only considering linear functions $f ;$ see Code Snippet 6.43 for an example. Code Snippet 4.14 shows an example for a nonlinear regression technique.

(xvi) Dependence between sets of discrete random variables X and Y can be measured by the Shannon mutual information [Cover and Thomas, 1991]

$$
I (\mathbf {X}: \mathbf {Y}) := \sum_ {\mathbf {x}, \mathbf {y}} p (\mathbf {x}, \mathbf {y}) \log \frac {p (\mathbf {x} , \mathbf {y})}{p (\mathbf {x}) p (\mathbf {y})}.
$$

(xvii) Conditional dependence of sets of discrete random variables X and Y, given the set $\mathbf { Z } ,$ is measured via the conditional Shannon mutual information [Cover and Thomas, 1991]

$$
I (\mathbf {X}: \mathbf {Y} | \mathbf {Z}) := \sum_ {\mathbf {x}, \mathbf {y}, \mathbf {z}} p (\mathbf {x}, \mathbf {y}, \mathbf {z}) \log \frac {p (\mathbf {x} , \mathbf {y} | \mathbf {z})}{p (\mathbf {x} | \mathbf {z}) p (\mathbf {y} | \mathbf {z})}.
$$

(xviii) For continuous variables, the sums are replaced with integrals

$$
I (\mathbf {X}: \mathbf {Y}) := \int p (\mathbf {x}, \mathbf {y}) \log \frac {p (\mathbf {x} , \mathbf {y})}{p (\mathbf {x}) p (\mathbf {y})} d \mathbf {x} d \mathbf {y},
$$

and

$$
I (\mathbf {X}: \mathbf {Y} | \mathbf {Z}) := \int p (\mathbf {x}, \mathbf {y}, \mathbf {z}) \log \frac {p (\mathbf {x} , \mathbf {y} | \mathbf {z})}{p (\mathbf {x} | \mathbf {z}) p (\mathbf {y} | \mathbf {z})} d \mathbf {x} d \mathbf {y} d \mathbf {z}.
$$

## A.2 Independence and Conditional Independence Testing

In practice, we are given a finite sample $( X _ { 1 } , Y _ { 1 } ) , \ldots , ( X _ { n } , Y _ { n } ) \overset { \mathrm { i i d } } { \sim } P _ { X , Y }$ and want to decide whether the underlying random variables are independent or not. Since we do not expect the empirical correlation (or any independence measure) to be exactly 0, we need to take into account random fluctuations of the dependence measures. This can be done by statistical hypothesis tests. The idea is to consider the null hypothesis $H _ { 0 } : X \perp Y$ and the alternative $H _ { A } : X \not \perp Y$ . Therefore, one usually constructs a test statistic $T _ { n }$ that maps any finite sample to a real number, and one decides according to

$$
(x _ {1}, y _ {1}), \ldots , (x _ {n}, y _ {n}) \mapsto \left\{ \begin{array}{l l} H _ {0} & \text { if } T _ {n} \leq c \\ H _ {A} & \text { if } T _ {n} > c. \end{array} \right.
$$

Here, $T _ { n }$ is shorthand notation for $T _ { n } { \big ( } ( x _ { 1 } , y _ { 1 } ) , \dots , ( x _ { n } , y _ { n } ) { \big ) }$ . The threshold $c \in \mathbb { R }$ is chosen such that we can control the type I error; that is, for any P satisfying $H _ { 0 }$ , we have $P ( T _ { n } > c ) \leq \alpha$ , where α is the significance level of the test, specified by the user. In practice, we are given data and compute the statistic $T _ { n }$ . If $T _ { n } > c$ , the null hypothesis is rejected, and we can be relatively confident that our decision is correct; otherwise, the null hypothesis is not rejected, which does not necessarily mean much (it could be that the sample size n was too small to detect the dependence between X and Y ). The p-value of a test is the smallest significance level, such that the test is rejected.

We now briefly mention a couple of choices for $T _ { n }$ . There are many more tests, however, and we do not claim that the list contains optimal procedures; see Code Snippet A.1 for a practical example.

(i) To test for vanishing correlation, we can use the empirical correlation coefficient and a t-test (for Gaussian variables) or Fisher’s z-transform (e.g., cor.test in R Core Team [2016]).

(ii) As an independence test, we may use a $\chi ^ { 2 } .$ -test for discrete or discretized data (e.g., chisq.test in R Core Team [2016]).  
(iii) An example for a general non-parametric independent test is the Hilbert-Schmidt Independence Criterion (HSIC) [see Gretton et al., 2008]. Its idea is based on an injective mapping into reproducing kernel Hilbert spaces (RKHSs) [Scholkopf and Smola, 2002]. Given a positive definite kernel, we ¨ can map probability distributions into the corresponding RKHS $\mathcal { H } .$ , that is, $P _ { X , Y } \mapsto \mu ( P _ { X , Y } ) \in \mathcal { H }$ . For so-called characteristic kernels (e.g., the Gaussian kernel), this mapping is injective. In particular, we then have

$$
\mu (P _ {X, Y}) = \mu (P _ {X} \otimes P _ {Y}) \quad \text {   if   and   only   if   } \quad P _ {X, Y} = P _ {X} \otimes P _ {Y},
$$

and the latter holds if and only if X and Y are independent. The HSIC is defined as the squared RKHS-distance between the joint distribution and the product of marginals:

$$
\operatorname{HSIC} \left(P _ {X, Y}\right) := \left\| \mu \left(P _ {X, Y}\right) - \mu \left(P _ {X} \otimes P _ {Y}\right) \right\| _ {\mathcal {H}} ^ {2}.
$$

As test statistic $T _ { n }$ we can now use an estimator for $\mathrm { H S I C } ( P _ { X , Y } )$ . If X and Y are independent, HSIC $\left( P _ { X , Y } \right)$ equals 0, and we expect its estimator $T _ { n }$ to be small. Gretton et al. [2008] provide ways how to choose the threshold c.

Alternatively, we can express HSIC as the Hilbert-Schmidt norm of the covariance operator $C _ { X Y }$ . The latter is defined such that for all $f$ and $g$ that are members of the corresponding RKHSs

$$
\langle f, C _ {X Y} g \rangle = \mathbb {E} [ f (X) g (Y) ] - \mathbb {E} [ f (X) ] \mathbb {E} [ g (Y) ].
$$

The cross-covariance operator is therefore an extension of the covariance matrix. If X is $d _ { X }$ -dimensional, Y is $d _ { Y } { \mathrm { - d i m e n s i o n a l } }$ , and the corresponding RKHSs are isomorphic to $\mathbb { R } ^ { d _ { X } }$ and $\mathbb { R } ^ { d _ { Y } }$ , respectively, $C _ { X Y }$ can be described with the $d _ { X } \times d _ { Y } – \mathrm { d i m e n s i o n a l }$ cross-covariance matrix. Certainly, X and Y do not need to be independent if the covariance matrix vanishes. For characteristic kernels, however, the RKHSs are infinitely dimensional and not isomorphic to $\mathbb { R } ^ { d }$ . The cross-covariance operator has zero norm if and only if X and Y are independent.

Pfister et al. [2017] extend the procedure to test for joint independence between d variables. This is necessary to test for joint independence of noise variables, for example. They provide code for both the bivariate and the multivariate procedure (see the R-package dHSIC).

In practice, one usually needs to choose kernel parameters. For the Gaussian kernel, many implementations choose the bandwidth σ according to the commonly named median heuristic [e.g., Gretton et al., 2008].

(iv) Conditional independence testing Conditional independence testing is a hard problem, especially if the conditioning set is large. While it is current research to obtain a precise formalization for this statement, we provide an example that indicates the hardness of the problem. If $Z _ { 1 } , \ldots , Z _ { d }$ are binary variables, we have that

$$
\begin{array}{l} X \perp Y | Z _ {1}, \dots , Z _ {d} \\ \Leftrightarrow \quad \forall (z _ {1}, \dots , z _ {d}) \in \{0, 1 \} ^ {d}: \quad X \perp Y | Z _ {1} = z _ {1}, \dots , Z _ {d} = z _ {d}. \\ \end{array}
$$

If we cannot assume anything on the way X and Y may depend on the $Z \ ' _ { \mathrm { s } } .$ , we need to perform an unconditional independence test for each of the $2 ^ { d }$ assignments $( \mathrm { e . g . , } Z _ { d }$ could be a common child of X and Y with the dependence only detectable for a specific assignment of the other $Z _ { 1 } , \ldots , Z _ { d - 1 } )$ .

For continuous variables, extensions of the HSIC test have been proposed. Fukumizu et al. [2008] extend the idea to conditional cross-covariance operators to obtain a conditional independence test. This is developed further by Zhang et al. [2011], who additionally provide an approximation of the test statistic’s distribution under the null hypothesis.

Code Snippet A.1 The following code generates a sample of a distribution over two variables that are uncorrelated but dependent.

```r
library(dHSIC)
#
# generates a sample from two uncorrelated but dependent random variables
set.seed(1)
A <- runif(200)-0.5
B <- runif(200)-0.5
X <- t(c(cos(pi/4), -sin(pi/4)) %*% rbind(A, B))
Y <- t(c(sin(pi/4), cos(pi/4)) %*% rbind(A, B))
#
# performs the statistical test
cor.test(X,Y)$p.value
# 0.3979561
dhsic.test(X,Y)$p.value
# 1.970705e-08
```

## A.3 Capacity of Function Classes

Here, we address the question whether the sequence of functions minimizing the empirical risk (1.3) converges against a function that also minimizes the risk (1.2); see Section 1.2. By the law of large numbers, we know that for any fixed $f \in { \mathcal { F } }$ and $\varepsilon > 0$ ,

$$
\lim _ {n \to \infty} P \left(\left| R [ f ] - R _ {\mathrm{emp}} ^ {n} [ f ] \right| > \varepsilon\right) = 0, \tag {A.4}
$$

with exponentially fast convergence governed by Chernov’s bound [e.g., Vapnik, 1998]. However, this does not imply consistency of empirical risk minimization. This is due to the fact that we are choosing the function $f$ by minimizing (1.3). This implies that even though the $( x _ { i } , y _ { i } )$ are independent, the errors or losses $\frac { 1 } { 2 } \vert f ( x _ { i } ) - y _ { i } \vert$ are not. In this case, the law of large numbers in its usual form does not apply. It turns out that to get consistency, we need a uniform law of large numbers [Vapnik, 1998]. This amounts to

$$
\lim _ {n \rightarrow \infty} P \left(\sup _ {f \in \mathcal {F}} (R [ f ] - R _ {\mathrm{emp}} ^ {n} [ f ]) > \varepsilon\right) = 0 \tag {A.5}
$$

for all $\varepsilon > 0$ , a property that depends on the function class ${ \mathcal F } .$ .

How about choosing $\mathcal { F } = \mathcal { V } ^ { \mathcal { X } }$ , in other words, all functions from $\mathcal { X }$ to $\mathcal { V } ?$ Unfortunately, this does not lead to (A.5), and the reasoning is as follows: Suppose that based on the available sample (1.1), we decide that $f ^ { * }$ is a good solution — for instance, since it satisfies $f ( x _ { i } ) = y _ { i }$ for all i. In this case, let us construct another function $f ^ { * * }$ that agrees with $f ^ { * }$ on the sample and disagrees everywhere else. If our distribution $P _ { X , Y }$ possesses a density, then the probability of encountering any of the training points exactly again in the future is zero. As a consequence, $f ^ { * }$ and $f ^ { * * }$ will almost always disagree. Based on the training set alone, however, there is no way to choose one over the other. Similarly, in (A.5) we would find that whenever we have found a function $f ^ { * }$ for which $( R [ f ^ { * } ] - R _ { \mathrm { e m p } } ^ { n } [ f ^ { * } ] )$ happens to be small, we can construct another function $f ^ { * * }$ for which $( R [ f ^ { * * } ] - R _ { \mathrm { e m p } } ^ { n } [ f ^ { * * } ] )$ is large, so uniform convergence (A.5) is impossible to achieve in our considered case where $\mathcal { F } = \mathcal { V } ^ { \mathcal { X } }$ .

On the other hand, the condition $( \mathsf { A } . 5 )$ becomes weaker as we make $\mathcal { F }$ smaller. How one measures the size (or capacity) of $\mathcal { F }$ is beyond the scope of this book, but it turns out that for a summary of the size of $\mathcal { F }$ irrespective of the underlying distribution, a single number is enough. It is referred to as the VC (Vapnik-Chervonenkis) dimension of $\mathcal { F }$ . It sometimes coincides with the number of free parameters, but it can also be vastly different. If the VC dimension is finite, we get consistency of empirical risk minimization for any $P _ { X , Y }$ [Vapnik, 1998]. The VC dimension is related to falsifiability and Popper’s notion of the dimension of a theory [Corfield et al., 2009]. A typical risk bound of statistical learning theory states that for all $\delta > 0$ , with probability $1 - \delta$ and for all $f \in { \mathcal { F } }$ , we have

$$
R [ f ] \leq R _ {\mathrm{emp}} ^ {n} [ f ] + \sqrt {\frac {h (\log (2 n / h) + 1) - \log (\delta / 4)}{n}}, \tag {A.6}
$$

where h is the VC dimension of the function class $\mathcal { F }$ . This means that if we can come up with an $\mathcal { F }$ that has small VC dimension yet contains functions that are sufficiently suitable for the given task to achieve a small $R _ { \mathrm { e m p } } ^ { n } [ f ]$ , then we can guarantee (with high probability) that those functions have small expected error on future data from the same distribution. This formulates a non-trivial trade-off: on the one hand, we would like to work with a large class of functions to allow for a small $R _ { \mathrm { e m p } } ^ { n }$ , but on the other hand, we want the class to be small to control h.

## B
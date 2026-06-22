# E-Value: Evidence for Causation in Observational Studies with Unmeasured Confounding

All the methods discussed in Part III rely crucially on the ignorability assumption. They require controlling for all confounding between the treatment and outcome. However, we cannot use the data to validate the ignorability assumption. Observational studies are often criticized due to the possibility of unmeasured confounding. The famous Yule–Simpson Paradox demonstrates that an unmeasured binary confounder can completely overturn an observed association between the treatment and outcome. However, to overturn a larger observed association, this unmeasured confounder must have stronger association with the treatment and the outcome. In other words, not all observational studies are created equal. Some provide stronger evidence for causation than others.

The following three chapters will discuss various sensitivity analysis techniques that can quantify the evidence of causation based on observational studies in the presence of unmeasured confounding. This chapter starts with the E-value, introduced by VanderWeele and Ding (2017) based on the theory in Ding and VanderWeele (2016). It is more useful for observational studies using logistic regressions. Chapter 18 discusses sensitivity analysis for the average causal effect based on inverse probability weighting, outcome regression, and doubly robust estimation. Chapter 19 discusses Rosenbaum’s framework for sensitivity analysis for matched observational studies.

## 17.1 Cornfield-type sensitivity analysis

Although we do not assume ignorability given X:

$$
Z \not \perp \{Y (1), Y (0) \} \mid X,
$$

we still assume latent ignorability given X and another unmeasured confounder U :

$$
Z \bot \{Y (1), Y (0) \} \mid (X, U).
$$

The technique in this chapter works the best for a binary outcome $Y$ although it can be extended to other non-negative outcomes (Ding and VanderWeele, 2016). Focus on binary $Y$ now. The true conditional causal effect on the risk ratio scale is defined as

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{true}} = \frac {\mathrm{pr} \{Y (1) = 1 \mid X = x \}}{\mathrm{pr} \{Y (0) = 1 \mid X = x \}},
$$

and the observed conditional risk ratio equals

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} = \frac {\operatorname* {p r} (Y = 1 \mid Z = 1 , X = x)}{\operatorname* {p r} (Y = 1 \mid Z = 0 , X = x)}.
$$

In general, with an unmeasured confounder $U ,$

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{true}} \neq \mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}}
$$

because

$$
\mathrm{RR} _ {Z Y | x} ^ {\text {true}} = \frac {\int \operatorname* {p r} (Y = 1 \mid Z = 1 , X = x , U = u) F (\mathrm{d} u \mid X = x)}{\int \operatorname* {p r} (Y = 1 \mid Z = 0 , X = x , U = u) F (\mathrm{d} u \mid X = x)}
$$

and

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} = \frac {\int \operatorname* {p r} (Y = 1 \mid Z = 1 , X = x , U = u) F (\mathrm{d} u \mid Z = 1 , X = x)}{\int \operatorname* {p r} (Y = 1 \mid Z = 0 , X = x , U = u) F (\mathrm{d} u \mid Z = 0 , X = x)}
$$

are averaged over different distributions of $U .$ .

Doll and Hill (1950) found that the risk ratio of cigarette smoking on lung cancer was 9 even after adjusting for many observed covariates $X ^ { \bar { 1 } }$ . Fisher (1957) criticized their result to be noncausal because it is possible that a hidden gene simultaneously causes cigarette smoking and lung cancer although the true causal effect of cigarette smoking on lung cancer is absent. This is the common cause hypothesis, also discussed by Reichenbach (1957). Cornfield et al. (1959) took a more constructive perspective and asked: how strong this unmeasured confounder must be in order to explain away the observed association between cigarette smoking and lung cancer? Below we will use Ding and VanderWeele (2016)’s general formulation of the problem.

Consider the following causal diagram:

![image_20](images/image_20.png)

which conditions on $X .$ So $Z \bot \bot Y \mid ( X , U )$ . Conditioning on $X$ and $U ,$ we observe no association between $Z$ and $Y ;$ but conditioning on only $X ,$ , we observe association between $Z$ and Y . Although we can allow U to be general as Ding and VanderWeele (2016), we assume that U is binary to simplify the presentation.

Define two sensitivity parameters:

$$
\mathrm{RR} _ {Z U | x} = \frac {\operatorname* {p r} (U = 1 \mid Z = 1 , X = x)}{\operatorname* {p r} (U = 1 \mid Z = 0 , X = x)} \equiv \frac {f _ {1 , x}}{f _ {0 , x}}
$$

measures the treatment-confounder association, and

$$
\mathrm{RR} _ {U Y | x} = \frac {\operatorname* {p r} (Y = 1 \mid U = 1 , X = x)}{\operatorname* {p r} (Y = 1 \mid U = 0 , X = x)},
$$

measures the confounder-outcome association, conditional on covariates $X =$ x. Without loss of generality, we assume that $\mathrm { R R } _ { x } ^ { \mathrm { o b s } } > 1 , \mathrm { R R } _ { Z U | x } > 1$ , and $\mathrm { R R } _ { U Y \mid x } > 1$ . We can show the main result below.

Theorem 17.1 Under $Z \bot \bot Y \mid ( X , U )$ , we have

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} \leq \frac {\mathrm{RR} _ {Z U | x} \mathrm{RR} _ {U Y | x}}{\mathrm{RR} _ {Z U | x} + \mathrm{RR} _ {U Y | x} - 1}.
$$

Theorem 17.1 shows the upper bound of the observed risk ratio of the treatment on the outcome if the conditional independence $Z \bot \bot Y \mid ( X , U )$ holds. Under this conditional independence assumption, the association between the treatment and the outcome is purely due to the association between the treatment, $\mathrm { R R } _ { Z U \mid x } ,$ and the confounder and the association between the confounder and the outcome, $\operatorname { 3 R } _ { U Y \mid x }$ . The upper bound equals $\mathrm { R R } _ { Z U | x } \mathrm { R R } _ { U Y | x } / \big ( \mathrm { R R } _ { Z U | x } + \mathrm { R R } _ { U Y | x } - 1 \big )$ . A similar inequality appeared in Lee (2011). It is also related to Cochran’s formula or the omitted-variable bias formula for linear models, which was reviewed in Problem 16.1.

$\mathrm { R R } _ { x } ^ { \mathrm { o b s } }$ the two confounding measures $\mathrm { R R } _ { Z U | x }$ and $\operatorname { R R } _ { U Y \mid x }$ cannot be arbitrary. Their function $\mathrm { R R } _ { Z U | x } \mathrm { R R } _ { U Y | x } / \big ( \mathrm { R R } _ { Z U | x } + \mathrm { R R } _ { U Y | x } - 1 \big )$ must be at least at large as rrobsx . $\mathrm { R R } _ { x } ^ { \mathrm { o b s } }$

I will give the proof of Theorem 17.1 below.

$\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ as

$$
\begin{array}{l} \mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} \\ = \frac {\operatorname{pr} (Y = 1 \mid Z = 1 , X = x)}{\operatorname{pr} (Y = 1 \mid Z = 0 , X = x)} \\ = \frac {\left[ \begin{array}{c} \operatorname{pr} (U = 1 \mid Z = 1 , X = x) \operatorname{pr} (Y = 1 \mid Z = 1 , U = 1 , X = x) \\ + \operatorname{pr} (U = 0 \mid Z = 1 , X = x) \operatorname{pr} (Y = 1 \mid Z = 1 , U = 0 , X = x) \end{array} \right]}{\left[ \begin{array}{c} \operatorname{pr} (U = 1 \mid Z = 0 , X = x) \operatorname{pr} (Y = 1 \mid Z = 0 , U = 1 , X = x) \\ + \operatorname{pr} (U = 0 \mid Z = 0 , X = x) \operatorname{pr} (Y = 1 \mid Z = 0 , U = 0 , X = x) \end{array} \right]} \\ = \frac {\left[ \begin{array}{c} \operatorname{pr} (U = 1 \mid Z = 1 , X = x) \operatorname{pr} (Y = 1 \mid U = 1 , X = x) \\ + \operatorname{pr} (U = 0 \mid Z = 1 , X = x) \operatorname{pr} (Y = 1 \mid U = 0 , X = x) \end{array} \right]}{\left[ \begin{array}{c} \operatorname{pr} (U = 1 \mid Z = 0 , X = x) \operatorname{pr} (Y = 1 \mid U = 1 , X = x) \\ + \operatorname{pr} (U = 0 \mid Z = 0 , X = x) \operatorname{pr} (Y = 1 \mid U = 0 , X = x) \end{array} \right]} \\ = \frac {f _ {1 , x} \mathrm{RR} _ {U Y | x} + 1 - f _ {1 , x}}{f _ {0 , x} \mathrm{RR} _ {U Y | x} + 1 - f _ {0 , x}} \\ = \frac {(\mathrm{RR} _ {U Y | x} - 1) f _ {1 , x} + 1}{\frac {\mathrm{RR} _ {U Y | x} - 1}{\mathrm{RR} _ {Z U | x}} f _ {1 , x} + 1}. \\ \end{array}
$$

$\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ is increasing in $f _ { 1 , x } ,$ . So letting $f _ { 1 , x } = 1$ , we have

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} \leq \frac {(\mathrm{RR} _ {U Y | x} - 1) + 1}{\frac {\mathrm{RR} _ {U Y | x} - 1}{\mathrm{RR} _ {Z U | x}} + 1} = \frac {\mathrm{RR} _ {Z U | x} \mathrm{RR} _ {U Y | x}}{\mathrm{RR} _ {Z U | x} + \mathrm{RR} _ {U Y | x} - 1}.
$$

In the proof of Theorem 17.1, we have obtain an identity

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} = \frac {(\mathrm{RR} _ {U Y | x} - 1) f _ {1 , x} + 1}{\frac {\mathrm{RR} _ {U Y | x} - 1}{\mathrm{RR} _ {Z U | x}} f _ {1 , x} + 1}.
$$

But this identity involves three parameters

$$
\left\{f _ {1, x}, \mathrm{RR} _ {Z U | x}, \mathrm{RR} _ {U Y | x} \right\};
$$

see Problem 17.2 for a related formula. In contrast, the upper bound in Theorem 17.1 involves only two parameters

$$
\left\{\mathrm{RR} _ {Z U | x}, \mathrm{RR} _ {U Y | x} \right\}
$$

which measure the strength of the confounder.

## 17.2 E-value

Lemma 17.1 below is useful for deriving interesting corollaries of Theorem 17.1.

Lemma 17.1 Define $\beta ( w _ { 1 } , w _ { 2 } ) = w _ { 1 } w _ { 2 } / ( w _ { 1 } + w _ { 2 } - 1 )$ for $w _ { 1 } > 1$ and $w _ { 2 } > 1$ .

$$
\begin{array}{l} 1. \beta (w _ {1}, w _ {2}) \text {   is   symmetric   in   } w _ {1} \text {   and   } w _ {2}; \\ 2. \beta (w _ {1}, w _ {2}) \text {   increasing   in   both   } w _ {1} \text {   and   } w _ {2}; \\ 3. \beta (w _ {1}, w _ {2}) \leq w _ {1} a n d \beta (w _ {1}, w _ {2}) \leq w _ {2}; \\ 4. \beta (w _ {1}, w _ {2}) \leq w ^ {2} / (2 w - 1), \text {   where   } w = \max (w _ {1}, w _ {2}). \\ \end{array}
$$

Using Theorem 17.1 and Lemma 17.1(3), we have

$$
\mathrm{RR} _ {Z U | x} \geq \mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}}, \quad \mathrm{RR} _ {U Y | x} \geq \mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}},
$$

or, equivalently,

$$
\min \left(\mathrm{RR} _ {Z U | x}, \mathrm{RR} _ {U Y | x}\right) \geq \mathrm{RR} _ {Z Y | x} ^ {\text { obs }}.
$$

Therefore, to explain away the observed relative risk, both confounding mea-$\mathrm { R R } _ { Z U | x }$ $\operatorname { R R } _ { U Y \mid x }$ $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ $\mathrm { R R } _ { Z U | x } \geq \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } .$ inequality (Gastwirth et al., 1998). Schlesselman (1978) derived the inequality $\mathrm { R R } _ { U Y | x } \geq \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } .$ These are related to to the data processing inequality in information theory2.

If we define $w = \mathrm { m a x } \big ( \mathrm { R R } _ { Z U | x } , \mathrm { R R } _ { U Y | x } \big )$ , then we can use Theorem 17.1 and Lemma 17.1(4) to obtain

$$
\begin{array}{l} w ^ {2} / (2 w - 1) \geq \beta (\mathrm{RR} _ {Z U | x}, \mathrm{RR} _ {U Y | x}) \geq \mathrm{RR} _ {x} ^ {\text { obs }} \\ \implies w ^ {2} - 2 \mathrm{RR} _ {x} ^ {\mathrm{obs}} w + \mathrm{RR} _ {x} ^ {\mathrm{obs}} \geq 0, \\ \end{array}
$$

$\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } - \sqrt { \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } \big ( \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } - 1 \big ) }$ is always smaller than or equal to 1, so we have

$$
w = \max (\mathrm{RR} _ {Z U | x}, \mathrm{RR} _ {U Y | x}) \geq \mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} + \sqrt {\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} (\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} - 1)}.
$$

Therefore, to explain away the observed relative risk, the maximum of the confounding measures $\mathrm { R R } _ { Z U | x }$ and $\mathrm { R R } _ { U Y \mid x }$ must be at least as large as $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } + \sqrt { \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } \big ( \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } - 1 \big ) }$ . Based on this result, VanderWeele and Ding (2017) introduced the following notion of E-value for measuring the evidence of causation with observational studies.

$\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } ,$ define the E-Value as

$$
\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} + \sqrt {\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} (\mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}} - 1)}
$$

$\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ is estimated with sampling error. We can calculate the E-value based on the $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } ,$ $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$

Fisher’s p-value measures the evidence for causal effects in randomized experiment. We have discussed the p-value based on the FRT in Part II of this book. However, in observational studies with large sample sizes, p-values can be a poor measure of evidence for causal effects. Even if the true causal effects are 0, a tiny amount of unmeasured confounding can bias the estimate, which can result in extremely small p-values given the small sampling uncertainty. The sampling uncertainty is usually secondary in observational studies with large sample sizes, but the uncertainty due to unmeasured confounding is often the first order problem that does not diminish with increased sample sizes. VanderWeele and Ding (2017) argued that the E-value is a better measure of the evidence for causal effects in observational studies.

## 17.3 A classic example

I revisit a classic example below.

Example 17.1 Hammond and Horn (1958) used the U.S. population to study the cigarette smoking and lung cancer relationship. Ignoring covariates, their data can be represented by a $2 \times 2$ table:

<table><tr><td></td><td>Lung cancer</td><td>No lung cancer</td></tr><tr><td>Smoker</td><td>397</td><td>78557</td></tr><tr><td>Non-smoker</td><td>51</td><td>108778</td></tr></table>

Based on the data, they obtained an estimate of the risk ratio 10.73 with a 95% confidence interval [8.02, 14.36]. To explain away the point estimate, the E-value is

$$
1 0. 7 3 + \sqrt {1 0 . 7 3 \times (1 0 . 7 3 - 1)} = 2 0. 9 5;
$$

to explain away the lower confidence limit, the E-value is

$$
8. 0 2 + \sqrt {8 . 0 2 \times (8 . 0 2 - 1)} = 1 5. 5 2.
$$

Figure 17.1 shows the joint values of the two confounding measures to explain away the point estimate and lower confidence limit of the risk ratio. In particular, to explain way the point estimate, they must lie in the area above the solid curve; to explain away the lower confidence limit, they must lie in the area above the dashed curve.

## 17.4 Extensions

## 17.4.1 E-value and Bradford Hill’s criteria for causation

The E-value provides evidence for causation. But evidence is not a proof. With a larger E-value, we need a stronger unmeasured confounder to explain away the observed risk ratio; the evidence for causation is stronger. With a smaller E-value, we need a weaker unmeasured confounder to explain away the observed risk ratio; the evidence for causation is weaker. Coupled with the discussion in Section 17.5.1, a larger observed risk ratio have stronger evidence for causation. This is closely related to Sir Bradford Hill’s first criterion for causation: strength of the association (Bradford Hill, 1965). Theorem 17.1 provides a mathematical quantification of his heuristic argument.

In a famous paper, Bradford Hill (1965) proposed a set of nine criteria to provide evidence for causation between a presumed cause and outcome. His criteria are

1. strength;  
2. consistency;  
3. specificity;  
4. temporality;  
5. biological gradient;  
6. plausibility;  
7. coherence;  
8. experiment;  
9. analogy.

The E-value is a way to justify his first criterion. That is, stronger association often provides stronger evidence for causation because to explain way stronger association, we need stronger confounding measures. We have discussed randomized experiments in Part II, which corroborates his eighth criterion. Due to the space limit, I omit the detailed discussion of his other criteria and encourage the readers to read (Bradford Hill, 1965). Recently, this paper is re-printed as Bradford Hill (2020) with insightful comments from many leading researchers in causal inference.

## 17.4.2 E-value after logistic regression

With a binary outcome, it is common for epidemiologists to use a logistic regression of the outcome $Y _ { i }$ on the treatment indicator $Z _ { i }$ and covariates $X _ { i } { \mathrm { : } }$ :

$$
\mathrm{pr} (Y _ {i} = 1 \mid Z _ {i}, X _ {i}) = \frac {e ^ {\beta_ {0} + \beta_ {1} Z _ {i} + \beta_ {2} ^ {\mathsf {T}} X _ {i}}}{1 + e ^ {\beta_ {0} + \beta_ {1} Z _ {i} + \beta_ {2} ^ {\mathsf {T}} X _ {i}}}.
$$

In the logistic model above, the coefficient of $Z _ { i }$ is the log of the conditional odds ratio between the treatment and the outcome given the covariates:

$$
\beta_ {1} = \log \frac {\mathrm{pr} (Y _ {i} = 1 \mid Z _ {i} = 1 , X _ {i} = x) / \mathrm{pr} (Y _ {i} = 0 \mid Z _ {i} = 1 , X _ {i} = x)}{\mathrm{pr} (Y _ {i} = 1 \mid Z _ {i} = 0 , X _ {i} = x) / \mathrm{pr} (Y _ {i} = 0 \mid Z _ {i} = 0 , X _ {i} = x)}.
$$

Importantly, the logistic model assumes a common odds ratio across all values of the covariates. Moreover, when the outcome is rare in that pr $( Y _ { i } = 1 \mid Z _ { i } =$1, $X _ { i } = x )$ and $\mathrm { p r } ( Y _ { i } = 1 \mid Z _ { i } = 0 , X _ { i } = x )$ are close to 0, the conditional odds ratio approximates the conditional risk ratio (see Proposition 1.1(3)):

$$
\beta_ {1} \approx \log \frac {\operatorname{pr} (Y _ {i} = 1 \mid Z _ {i} = 1 , X _ {i} = x)}{\operatorname{pr} (Y _ {i} = 1 \mid Z _ {i} = 0 , X _ {i} = x)} = \log \mathrm{RR} _ {Z Y | x} ^ {\mathrm{obs}}.
$$

Therefore, based on the estimated logistic regression coefficient and the corresponding confidence limits, we can calculated the E-value immediately. This is the leading application of the E-value.

Example 17.2 The NCHS2003.txt contains the National Center for Health Statistics birth certificate data, with the following binary indicator variables useful for us:

PTbirth pre-term birth  
preeclampsia pre-eclampsia $^{3}$ ageabove35 an older mother with age $\geq$ 35 (the treatment)  
somecollege college education  
mar marital status  
smoking smoking status  
drinking drinking status  
hispanic mother's ethnicity  
black mother's ethnicity  
nativeamerican mother's ethnicity  
asian mother's ethnicity

This version of the data is from Valeri and Vanderweele (2014). This example focuses on the outcome PTbirth and Problem 17.3. The following R code computes the E-values after fitting a logistic regression. Based on the E-values, we conclude that to explain away the point estimate, the maximum confounding measure must be larger than 1.94, and to explain away the lower confidence limit, the maximum confounding measure must be larger than 1.91. Although these confounding measures are not as strong as those in Section 17.3, they appear to be fairly large in epidemiologic studies.

```diff
> evalue = function(rr)
+ {
+    rr + sqrt(rr*(rr - 1))
+ }
>
> NCHS2003 = read.table("NCHS2003.txt", header = TRUE, sep = "\t")
>
> ## outcome: PTbirth
> y_logit = glm(PTbirth ~ ageabove35 +
+    mar + smoking + drinking + somecollege +
+    hispanic + black + nativeamerican + asian,
+    data = NCHS2003,
+    family = binomial)
> log_or = summary(y_logit)$coef[2, 1:2]
```

```txt
> est = exp(log_or[1])
> lower.ci = exp(log_or[1] - 1.96*log_or[2])
> est
Estimate
1.305982
> evalue(est)
Estimate
1.938127
>
> lower.ci
Estimate
1.294619
> evalue(lower.ci)
Estimate
1.912211
```

## 17.4.3 Non-zero true causal effect

Theorem 17.1 assumes no true causal effect of the treatment on the outcome. Ding and VanderWeele (2016) proved a general theorem allowing for non-zero true causal effect.

Theorem 17.2 Modify the definition of $\operatorname { R R } _ { U Y \mid x }$ as

$$
\mathrm{RR} _ {U Y | x} = \max _ {z = 0, 1} \frac {\operatorname* {p r} (Y = 1 \mid Z = z , U = 1 , X = x)}{\operatorname* {p r} (Y = 1 \mid Z = z , U = 0 , X = x)}.
$$

We have

$$
\mathrm{RR} _ {Z Y | x} ^ {\text {true}} \geq \mathrm{RR} _ {Z Y | x} ^ {\text {obs}} \Big / \frac {\mathrm{RR} _ {Z U | x} \mathrm{RR} _ {U Y | x}}{\mathrm{RR} _ {Z U | x} + \mathrm{RR} _ {U Y | x} - 1}.
$$

$\mathrm { R R } _ { Z Y | x } ^ { \mathrm { t r u e } } = 1$ . See the original paper of Ding and VanderWeele (2016) for the proof of Theorem 17.2. Without assuming any additional assumptions, Theorem 17.2 states a lower $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { t r u e } }$ $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ the two sensitivity parameters rrZU|x and rrUY |x. $\mathrm { R R } _ { Z U | x }$ $\mathrm { R R } _ { U Y \mid x } .$

When the treatment is apparently preventive to the outcome, the observed risk ratio is smaller than 1. In this case, Theorems 17.1 and 17.2 are not directly useful, and we must re-label the treatment levels and calculate the $1 / \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$

## 17.5 Critiques and responses

Since the original paper was published, E-value has been a standard number reported by many epidemiologic studies. Nevertheless, it also attracted critiques (Ioannidis et al., 2019). I will review some limitations of E-values below.

## 17.5.1 E-value is just a monotone transformation of the risk ratio

From Figure 17.2, we can see that if the risk ratio is large, then the E-value $\mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } + \sqrt { \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } \big ( \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } } - 1 \big ) }$ $2 \mathrm { R R } _ { Z Y | x } ^ { \mathrm { o b s } }$ which is linear in the risk ratio. For small risk ratio, the E-value is more nonlinear. Critics often say that the E-value is merely a monotone transformation of the point estimator or the confidence limits of the risk ratio. So it does not provide any additional information.

This is partially true. Indeed, the E-value is entirely based on the point estimator or the confidence limits of the risk ratio. It has a meaningful interpretation based on Theorem 17.1: to explain away the observed risk ratio, the maximum of the confounding measures must be at least as large as the E-value.

## 17.5.2 Calibration of the E-value

The E-value equals the maximum value of the association between the confounder and the treatment and that between the confounder and the outcome to completely explain aways an observed association. An obvious problem is that this confounder is fundamentally latent. So it is not trivial to decide whether a certain E-value is large or small. Another related problem is that the E-value depends on how many observed covariates X we have controlled for since it quantifies the strength of the residual confounding given X. Therefore, E-values across studies are not directly comparable. The E-value provides evidence for causation but this evidence should be assessed carefully based on background knowledge of the problem of interest.

The following leave-one-covariate-out approach is an intuitive approach to calibrate the E-value. With $X = ( X _ { 1 } , \ldots , X _ { p } ) $ , we can pretend that the component $X _ { j }$ were not observed and compute the $Z \mathrm { - } X _ { j }$ and $X _ { j ^ { - } } Y$ risk ratios given other observed covariates $( j = 1 , \ldots , p )$ . These risk ratios provide the range for the confounding measures due to U if we believe that the unmeasured U is not as strong as all of the observed covariates. However, I am not aware of any formal justification of this approach.

## 17.5.3 It works the best for a binary outcome and the risk ratio

Theorem 17.1 works well for a binary outcome and the risk ratio. Ding and VanderWeele (2016) also proposed sensitivity analysis methods for other causal parameters, but they are not as elegant as the E-value for binary outcome based on the risk ratio. The next chapter will propose a simple sensitivity analysis method for the average causal effect that include several methods in Part III as special cases.

## 17.6 Homework Problems

## 17.1 Lemma 17.1

Prove Lemma 17.1.

## 17.2 Schlesselman (1978)’s formula

For simplicity, we condition on X implicitly in the following discussion. With binary treatment $Z ,$ outcome $Y ,$ , and unmeasured confounder $U ,$ show that

$$
\frac {\mathrm{RR} _ {Z Y} ^ {\mathrm{obs}}}{\mathrm{RR} _ {Z Y} ^ {\mathrm{true}}} = \frac {1 + (\gamma - 1) \mathrm{pr} (U = 1 \mid Z = 1)}{1 + (\gamma - 1) \mathrm{pr} (U = 1 \mid Z = 0)}
$$

assuming a common risk ratio of the treatment on the outcome within both U = 0 and U = 1:

$$
\mathrm{RR} _ {Z Y | U = 0} = \mathrm{RR} _ {Z Y | U = 1},
$$

and also a common risk ratio of the confounder on the outcome within both Z = 0 and Z = 1:

$$
\mathrm{RR} _ {U Y | Z = 0} = \mathrm{RR} _ {U Y | Z = 1}, \text {   denoted   by   } \gamma .
$$

Hint: First verify that if $\mathrm { R R } _ { Z Y | U = 0 } = \mathrm { R R } _ { Z Y | U = 1 }$ then

$$
\mathrm{RR} _ {Z Y} ^ {\mathrm{true}} = \mathrm{RR} _ {Z Y | U = 0} = \mathrm{RR} _ {Z Y | U = 1}.
$$

This identity shows the collapsibility of the risk ratio. In epidemiology, the risk ratio is a collapsible measure of association.

Remark: Schlesselman (1978)’s formula does not assume conditional independence $Z \bot \bot Y \mid U ,$ but assumes homogeneity of the $Z – Y$ and $U { - } Y$ risk ratios. It is a classic formula for sensitivity analysis. It is an identity that is simple to implement with pre-specified

$$
\{\gamma , \mathrm{pr} (U = 1 \mid Z = 1), \mathrm{pr} (U = 1 \mid Z = 0) \}.
$$

However, it involves more sensitivity parameters than Theorem 17.1. Even though Theorem 17.1 only gives an inequality, it is not a loose inequality compared to Schlesselman (1978)’s formula under stronger assumptions. With Theorem 17.1, Schlesselman (1978)’s formula is only of historical interest.

## 17.3 E-value after logistic regression: data analysis

This problem uses the same dataset as Example 17.2.

Report the E-value for the outcome preeclampsia.

## 17.4 Cornfield-type inequalities for the risk difference

Consider binary $Z , Y , U ,$ and condition on X implicitly. Assume latent ignorability given U. Show that under $Z \bot \bot Y \mid U$ , we have

$$
\mathrm{RD} _ {Z Y} ^ {\mathrm{obs}} = \mathrm{RD} _ {Z U} \times \mathrm{RD} _ {U Y} \tag {17.1}
$$

where $\mathrm { R D } _ { Z Y } ^ { \mathrm { o b s } }$ is the observed risk difference of Z on $Y ,$ and $\mathrm { R D } _ { Z U }$ and rdUY are the treatment-confounder and confounder-outcome risk differences, respectively (recall the definition of the risk difference in Chapter 1.2.2).

Remark: Without loss of generality, assume that rd $_ { Z Y } ^ { \mathrm { o b s } } , \mathrm { R D } _ { Z U } , \mathrm { R D } _ { U Y }$ are all positive. Then (17.1) implies that

$$
\min \bigl (\mathrm{RD} _ {Z U}, \mathrm{RD} _ {U Y} \bigr) \geq \mathrm{RD} _ {Z Y} ^ {\mathrm{obs}}
$$

and

$$
\max \bigl (\mathrm{RD} _ {Z U}, \mathrm{RD} _ {U Y} \bigr) \geq \sqrt {\mathrm{RD} _ {Z Y} ^ {\mathrm{obs}}}.
$$

These are the Cornfield inequalities for the risk difference with a binary confounder. They show that for an unmeasured confounder to explain away $\mathrm { R D } _ { Z Y } ^ { \mathrm { o b s } }$ of them must be larger than the square root of $\mathrm { R D } _ { Z Y } ^ { \mathrm { o b s } }$ .

Cornfield et al. (1959) obtained, but did not appreciate the significance of (17.1). Gastwirth et al. (1998) and Poole (2010) discussed the first Cornfield condition for the risk difference, and Ding and VanderWeele (2014) discussed the second.

Ding and VanderWeele (2014) also derived more general results without assuming a binary U. Unfortunately, the results for a general U are weaker than those above for a binary U, that is, the inequalities become looser with more levels of U. This motivated Ding and VanderWeele (2016) to focus on the Cornfield inequalities for the risk ratio, which do not deteriorate with more levels of U.

## 17.5 Recommended reading

Ding and VanderWeele (2016) extended and unified the Cornfield-type sensitivity analysis, which is the theoretical basis for the notion of E-value.
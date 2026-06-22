# Rosenbaum-Style p-Values for Matched Observational Studies with Unmeasured Confounding

Rosenbaum (1987b) introduced a sensitivity analysis technique for matched observational studies. Although it works for general settings (Rosenbaum, 2002b), the theory is most elegant for one-to-one matching. Different from Chapters 17 and 18, Rosenbaum-type sensitivity analysis works the best for matched observational studies for testing the sharp null hypothesis of no individual treatment effect.

## 19.1 The model for sensitivity analysis with matched data

Consider exactly matched pairs from an observational study, with (i, j) indexing unit $j$ in pair $\textit { i } ( i = 1 , \dots , n ; j = 1 , 2 )$ . Assume iid sampling, and define the propensity score as

$$
e _ {i j} = \operatorname{pr} \left\{Z _ {i j} = 1 \mid X _ {i}, Y _ {i j} (1), Y _ {i j} (0) \right\}.
$$

Let $\mathbb { S } _ { i } = \{ Y _ { i 1 } ( 1 ) , Y _ { i 1 } ( 0 ) , Y _ { i 2 } ( 1 ) , Y _ { i 2 } ( 0 ) \}$ denote the set of all potential outcomes within pair i. Conditioning on the event that $Z _ { i 1 } + Z _ { i 2 } = 1$ , we have

$$
\begin{array}{l} \pi_ {i 1} = \operatorname{pr} \left\{Z _ {i 1} = 1 \mid X _ {i}, \mathbb {S} _ {i}, Z _ {i 1} + Z _ {i 2} = 1 \right\} \\ = \frac {\operatorname{pr} \left\{Z _ {i 1} = 1 , Z _ {i 2} = 0 \mid X _ {i} , \mathbb {S} _ {i} \right\}}{\operatorname{pr} \left\{Z _ {i 1} + Z _ {i 2} = 1 \mid X _ {i} , \mathbb {S} _ {i} \right\}} \\ = \frac {\operatorname{pr} \left\{Z _ {i 1} = 1 , Z _ {i 2} = 0 \mid X _ {i} , \mathbb {S} _ {i} \right\}}{\operatorname{pr} \left\{Z _ {i 1} = 1 , Z _ {i 2} = 0 \mid X _ {i} , \mathbb {S} _ {i} \right\} + \operatorname{pr} \left\{Z _ {i 1} = 0 , Z _ {i 2} = 1 \mid X _ {i} , \mathbb {S} _ {i} \right\}} \\ = \frac {e _ {i 1} (1 - e _ {i 2})}{e _ {i 1} (1 - e _ {i 2}) + (1 - e _ {i 1}) e _ {i 2}} \\ \end{array}
$$

Define $o _ { i j } = e _ { i j } / ( 1 - e _ { i j } )$ as the odds of the treatment for unit $( i , j )$ , and we have

$$
\pi_ {i 1} = \frac {o _ {i 1}}{o _ {i 1} + o _ {i 2}}.
$$

Under ignorability, $e _ { i j }$ is only a function of $X _ { i } ,$ and therefore, $e _ { i 1 } = e _ { i 2 }$ and $\pi _ { i 1 } = 1 / 2$ . Thus the treatment assignment mechanism conditioning on covariates and potential outcomes is equivalent to that from an MPE with equal treatment and control probabilities. This is a strategy to analyze matched observational studies we discussed in Chapter 15.1.

In general, $e _ { i j }$ is also a function of the unobserved potential outcomes, and it can range from 0 to 1. Rosenbaum (1987b)’s model for sensitivity analysis imposes bounds on the odds ratio $o _ { i 1 } / o _ { i 2 }$ .

Assumption 19.1 (Rosenbaum’s sensitivity model) The odds ratios are bounded by

$$
o _ {i 1} / o _ {i 2} \leq \Gamma , \quad o _ {i 2} / o _ {i 1} \leq \Gamma ,
$$

for some pre-specified $\Gamma \geq 1$ . Equivalently,

$$
\frac {1}{1 + \Gamma} \leq \pi_ {i 1} \leq \frac {\Gamma}{1 + \Gamma}
$$

for some pre-specified $\Gamma \geq 1$

Under Assumption 19.1, we have a biased MPE with unequal and varying treatment and control probabilities across pairs. When $\Gamma = 1$ , we have $\pi _ { i 1 }$ and thus a standard MPE. Therefore, $\Gamma > 1$ measures the deviation from the ideal MPE due to the omitted variables in matching.

## 19.2 Worst-case p-values under Rosenbaum’s sensitivity model

Consider testing the sharp null hypothesis

$$
H _ {0 \mathrm{F}}: Y _ {i j} (1) = Y _ {i j} (0) \text {   for   } i = 1, \dots , n \text {   and   } j = 1, 2
$$

based on the within-pair differences $\hat { \tau } _ { i } = ( 2 Z _ { i 1 } - 1 ) ( Y _ { i 1 } - Y _ { i 2 } ) ~ ( i = 1 , \ldots , n )$ . Under $H _ { \mathrm { 0 F } }$ , |τˆi| is fixed but $S _ { i } = I ( \hat { \tau } _ { i } > 0 )$ is random if $\hat { \tau } _ { i } \neq 0$ . Consider the following class of test statistics:

$$
T = \sum_ {i = 1} ^ {n} S _ {i} q _ {i},
$$

where $q _ { i } \geq 0$ is a function of $\big ( \vert \hat { \tau } _ { 1 } \vert , \dots , \vert \hat { \tau } _ { n } \vert \big )$ . Special cases include the sign statistic, the pair t statistic (up to some constant shift), and the Wilcoxon sign rank statistic:

$$
T = \sum_ {i = 1} ^ {n} S _ {i}, \quad T = \sum_ {i = 1} ^ {n} S _ {i} | \hat {\tau} _ {i} |, \quad T = \sum_ {i = 1} ^ {n} S _ {i} R _ {i},
$$

where $( R _ { 1 } , \ldots , R _ { n } )$ are the ranks of $\big ( \vert \hat { \tau } _ { 1 } \vert , \dots , \vert \hat { \tau } _ { n } \vert \big )$ .

What is the null distribution of the test statistic with general Γ? It can be quite complicated because we do not fully specify the exact values of the $\pi _ { i 1 } \mathrm { ^ { * } s }$ . Fortunately, we know that the worse case distribution correspond to

$$
S _ {i} \stackrel {\mathrm{IID}} {\sim} \text {Bernoulli} \left(\frac {\Gamma}{1 + \Gamma}\right).
$$

Here, the FRT with T has the largest p-value under the “the worst case” distribution. The corresponding distribution has mean

$$
E _ {\Gamma} (T) = \frac {\Gamma}{1 + \Gamma} \sum_ {i = 1} ^ {n} q _ {i},
$$

and variance

$$
\mathrm{var} _ {\Gamma} (T) = \frac {\Gamma}{(1 + \Gamma) ^ {2}} \sum_ {i = 1} ^ {n} q _ {i} ^ {2},
$$

with a Normal approximation

$$
\frac {T - \frac {\Gamma}{1 + \Gamma} \sum_ {i = 1} ^ {n} q _ {i}}{\sqrt {\frac {\Gamma}{(1 + \Gamma) ^ {2}} \sum_ {i = 1} ^ {n} q _ {i} ^ {2}}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

In practice, we can report a sequence of p-values as a function of Γ.

## 19.3 Revisiting the LaLonde data

We conduct Rosenbaum-style sensitivity analysis in the matched LaLonde data. We consider using the test statistic $\textstyle T = \sum _ { i = 1 } ^ { n } S _ { i } | { \hat { \tau } } _ { i } |$ . Under the ideal matched pair experiment with $\Gamma = 1$ , we can simulate the distribution of T and obtain the p-value 0.002, as shown in the first subfigure in Figure 19.1. With a slightly larger $\Gamma = 1 . 1$ , the distribution of T shifts to the right and the p-value increases to 0.011. If we further increase Γ to 1.3, then the distribution of T shifts further and the p-value exceeds 0.05. Figure 19.2 shows the histogram of the $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ and the p-value as a function of Γ; $\Gamma = 1 . 2 3 3$ measures the maximum confounding that we can still reject the null hypothesis at level 0.05.

We can also use the senmw function in the sensitivitymw package to obtain a sequence of p-values against Γ, as shown in Figure 19.2.

## 19.4 Homework Problems

19.1 Application of Rosenbaum’s approach

Re-analyze Example 10.3 using Rosenbaum’s approach.

19.2 Recommended reading

Rosenbaum (2015) provides a tutorial for his two R packages for sensitivity analysis with matched observational studies.

## 20
# Disentangle Mixture Distributions and Instrumental Variable Inequalities

The IV model in Chapter 21 imposes Assumptions 21.1–21.3:

1. $Z \bot \bot \{ D ( 1 ) , D ( 0 ) , Y ( 1 ) , Y ( 0 ) \}$ ;  
2. $\operatorname { p r } ( U = \mathrm { d } ) = 0 ;$  
3. $Y ( 1 ) = Y ( 0 )$ for $U = \mathrm { a ~ o r ~ n . }$

Table 22.1 summarizes the observed groups and the corresponding latent groups.

**TABLE 22.1: Observed groups and latent groups under Assumption 21.2**

<table><tr><td>Z=1</td><td>D=1</td><td>D(1)=1</td><td>U=c or a</td></tr><tr><td>Z=1</td><td>D=0</td><td>D(1)=0</td><td>U=n</td></tr><tr><td>Z=0</td><td>D=1</td><td>D(0)=1</td><td>U=a</td></tr><tr><td>Z=0</td><td>D=0</td><td>D(0)=0</td><td>U=c or n</td></tr></table>

Interestingly, Assumptions 21.1–21.3 together have some testable implications. Balke and Pearl (1997) called them the instrumental variable inequalities. This chapter will give an intuitive derivation of a special case of these inequalities. The proof is a direct consequence of identifying the means of the potential outcomes for all latent groups defined by U.

## 22.1 Disentangle Mixture Distributions and Instrumental Variable Inequalities

We summarize the main results in Theorem 22.1 below. Recall $\pi _ { u }$ as the proportion of type $U = u ,$ , and define

$$
\mu_ {z u} = E \{Y (z) \mid U = u \}, \quad (d = 0, 1; u = \mathrm{a,n,c}).
$$

Theorem 22.1 Under Assumptions 21.1–21.3, we can identify the proportions of the latent types by

$$
\pi_ {\mathrm{n}} = \operatorname{pr} (D = 0 | Z = 1),
$$

$$
\pi_ {\mathrm{a}} = \operatorname * {p r} (D = 1 \mid Z = 0),
$$

$$
\pi_ {\mathrm{c}} = E (D \mid Z = 1) - E (D \mid Z = 0),
$$

and the type-specific means of the potential outcomes by

$$
\mu_ {1 \mathrm{n}} = \mu_ {0 \mathrm{n}} \equiv \mu_ {\mathrm{n}} = E (Y \mid Z = 1, D = 0),
$$

$$
\mu_ {1 \mathrm{a}} = \mu_ {0 \mathrm{a}} \equiv \mu_ {\mathrm{a}} = E (Y \mid Z = 0, D = 1),
$$

$$
\mu_ {1 \mathrm{c}} = \pi_ {\mathrm{c}} ^ {- 1} \left\{E (D Y \mid Z = 1) - E (D Y \mid Z = 0) \right\},
$$

$$
\mu_ {0 \mathrm{c}} = \pi_ {\mathrm{c}} ^ {- 1} \left[ E \{(1 - D) Y \mid Z = 0 \} - E \{(1 - D) Y \mid Z = 1 \} \right].
$$

Proof of Theorem 17.1: Part I: We first identify the proportions of the latent compliance types. We can identify the proportion of the never takers by

$$
\operatorname{pr} (D = 0 \mid Z = 1) = \operatorname{pr} (U = \mathrm{n} \mid Z = 1)
$$

$$
= \operatorname{pr} (U = \mathrm{n}) = \pi_ {\mathrm{n}},
$$

and the proportion of the always takes by

$$
\operatorname{pr} (D = 1 \mid Z = 0) = \operatorname{pr} (U = \mathrm{a} \mid Z = 0)
$$

$$
= \operatorname{pr} (U = \mathrm{a}) = \pi_ {\mathrm{a}}.
$$

Therefore, the proportion of compliers is

$$
\pi_ {\mathrm{c}} = \operatorname * {p r} (U = \mathrm{c}) = 1 - \pi_ {\mathrm{n}} - \pi_ {\mathrm{a}}
$$

$$
= 1 - \operatorname{pr} (D = 0 \mid Z = 1) - \operatorname{pr} (D = 1 \mid Z = 0)
$$

$$
= E (D \mid Z = 1) - E (D \mid Z = 0) = \tau_ {D},
$$

which is coherent with our discussion before. Although we do not know individual latent compliance types for all units, we can identify the proportions of never takers, always takers, and compliers.

Part II: We then identify the means of the potential outcomes within latent compliance types. Under Assumption 21.3,

$$
\mu_ {\mathrm{1a}} = \mu_ {\mathrm{0a}} \equiv \mu_ {\mathrm{a}}, \quad \mu_ {\mathrm{1n}} = \mu_ {\mathrm{0n}} \equiv \mu_ {\mathrm{n}}.
$$

The observed group (Z = 1, D = 0) only has never takers, so

$$
E (Y \mid Z = 1, D = 0) = E \{Y (1) \mid Z = 1, U = \mathrm{n} \} = E \{Y (1) \mid U = \mathrm{n} \} = \mu_ {\mathrm{n}}.
$$

The observed group (Z = 0, D = 1) only has always takers, so

$$
E (Y \mid Z = 0, D = 1) = E \{Y (0) \mid Z = 0, U = \mathrm{a} \} = E \{Y (0) \mid U = \mathrm{a} \} = \mu_ {\mathrm{a}}.
$$

## 22.2 Disentangle Mixture Distributions and Instrumental Variable Inequalities 269

The observed group $( Z = 1 , D = 1 )$ has both compliers and always takers, so

$$
\begin{array}{l} E (Y \mid Z = 1, D = 1) = E \{Y (1) \mid Z = 1, D (1) = 1 \} \\ = E \{Y (1) \mid D (1) = 1 \} \\ = \operatorname{pr} \{D (0) = 1 \mid D (1) = 1 \} E \{Y (1) \mid D (1) = 1, D (0) = 1 \} \\ + \operatorname{pr} \{D (0) = 0 \mid D (1) = 1 \} E \{Y (1) \mid D (1) = 1, D (0) = 0 \} \\ { = } { \frac { \pi _ { \mathrm{c} } } { \pi _ { \mathrm{c} } + \pi _ { \mathrm{a} } } \mu _ { 1 \mathrm{c} } + \frac { \pi _ { \mathrm{a} } } { \pi _ { \mathrm{c} } + \pi _ { \mathrm{a} } } \mu _ { \mathrm{a} } . } \\ \end{array}
$$

Solve the linear equation above to obtain

$$
\begin{array}{l} \mu_ {1 \mathrm{c}} = \pi_ {\mathrm{c}} ^ {- 1} \left\{\left(\pi_ {\mathrm{c}} + \pi_ {\mathrm{a}}\right) E (Y \mid Z = 1, D = 1) - \pi_ {\mathrm{a}} E (Y \mid Z = 0, D = 1) \right\} \\ = \pi_ {\mathrm{c}} ^ {- 1} \left\{\operatorname * {p r} (D = 1 \mid Z = 1) E (Y \mid Z = 1, D = 1) \right. \\ - \operatorname{pr} (D = 1 \mid Z = 0) E (Y \mid Z = 0, D = 1) \} \\ = \pi_ {\mathrm{c}} ^ {- 1} \left\{E (D Y \mid Z = 1) - E (D Y \mid Z = 0) \right\}. \\ \end{array}
$$

The observed group $( Z = 0 , D = 0 )$ has both compliers and never takers, so we have

$$
\begin{array}{l} E (Y \mid Z = 0, D = 0) = E \{Y (0) \mid Z = 0, D (0) = 0 \} \\ = E \{Y (0) \mid D (0) = 0 \} \\ = \operatorname{pr} \{D (1) = 1 \mid D (0) = 0 \} E \{Y (0) \mid D (1) = 1, D (0) = 0 \} \\ + \operatorname{pr} \{D (1) = 0 \mid D (0) = 0 \} E \{Y (0) \mid D (1) = 0, D (0) = 0 \} \\ = \frac {\pi_ {\mathrm{c}}}{\pi_ {\mathrm{c}} + \pi_ {\mathrm{n}}} \mu_ {0 \mathrm{c}} + \frac {\pi_ {\mathrm{n}}}{\pi_ {\mathrm{c}} + \pi_ {\mathrm{n}}} \mu_ {\mathrm{n}}. \\ \end{array}
$$

Solve the linear equation above to obtain

$$
\begin{array}{l} \mu_ {0 \mathrm{c}} = \pi_ {\mathrm{c}} ^ {- 1} \left\{\left(\pi_ {\mathrm{c}} + \pi_ {\mathrm{n}}\right) E (Y \mid Z = 0, D = 0) - \pi_ {\mathrm{n}} E (Y \mid Z = 1, D = 0) \right\} \\ = \pi_ {\mathrm{c}} ^ {- 1} \left\{\operatorname * {p r} (D = 0 \mid Z = 0) E (Y \mid Z = 0, D = 0) \right. \\ \left. - \operatorname{pr} (D = 0 \mid Z = 1) E (Y \mid Z = 1, D = 0) \right\} \\ = \pi_ {c} ^ {- 1} \left[ E \{(1 - D) Y \mid Z = 0 \} - E \{(1 - D) Y \mid Z = 1 \} \right]. \\ \end{array}
$$

Based on the formulas of $\mu _ { \mathrm { 1 c } }$ and $\mu _ { \mathrm { 0 c } }$ in Theorem 22.1, we have

$$
\tau_ {\mathrm{c}} = \mu_ {1 \mathrm{c}} - \mu_ {0 \mathrm{c}} = \left\{E (Y \mid Z = 1) - E (Y \mid Z = 0) \right\} / \pi_ {\mathrm{c}},
$$

which is the same as the formula in Theorem 21.1 before.

Theorem 22.1 focuses on identifying the means of the potential outcomes, $\mu _ { z u }$ . Imbens and Rubin (1997) derived more general identification formulas for the distribution of the potential outcomes; I leave the details to Problem 22.2.

## 22.2 Testable implications

Is there any additional value of the this detour for deriving the formula of $\tau _ { \mathrm { c } } ?$ The answer is yes. For binary outcome, the following inequalities must be true:

$$
0 \leq \mu_ {1 c} \leq 1, \quad 0 \leq \mu_ {0 c} \leq 1,
$$

which implies four inequalities

$$
E (D Y \mid Z = 1) - E (D Y \mid Z = 0) \geq 0,
$$

$$
E (D Y \mid Z = 1) - E (D Y \mid Z = 0) \leq E (D \mid Z = 1) - E (D \mid Z = 0),
$$

$$
E \{(1 - D) Y \mid Z = 0 \} - E \{(1 - D) Y \mid Z = 1 \} \geq 0,
$$

$$
E \{(1 - D) Y \mid Z = 0 \} - E \{(1 - D) Y \mid Z = 1 \} \leq E (D \mid Z = 1) - E (D \mid Z = 0).
$$

Rearranging terms, we obtain the following unified inequalities.

Theorem 22.2 (Instrumental Variable Inequalities) With a binary outcome $Y _ { z }$ Assumptions 21.1–21.3 imply

$$
E (Q \mid Z = 1) - E (Q \mid Z = 0) \geq 0, \tag {22.1}
$$

$w h e r e ~ Q = D Y , D ( 1 - Y ) , ( D - 1 ) Y ~ a n d ~ D + Y - D Y .$

Under the IV assumptions 21.1–21.3, the difference in means for $Q =$ $D Y , D ( 1 - Y ) , ( D - 1 ) Y$ and $D + Y - D Y$ must all be non-negative. Importantly, these implications only involve the distribution of the observed variables. Rejection of the IV inequalities leads to rejection of the IV assumptions.

Balke and Pearl (1997) derived more general IV inequalities without assuming monotonicity. The above proving strategy is due to Jiang and Ding (2020) for a slightly more complex setting. Theorem 22.2 states the testable implications only for a binary outcome. Problem 22.3 gives an equivalent form, and Problem 22.4 gives the result for a general outcome.

## 22.3 Examples

For a binary outcome, we can estimate all the parameters by the method of moment as below.

```r
## function for binary data (Z, D, Y)
## n_{zdy}'s are the counts from 2X2X2 table
IVbinary = function(n111, n110, n101, n100, n011, n010, n001, n000){
```

22.3 Examples

```txt
n_tr = n111 + n110 + n101 + n100
n_co = n011 + n010 + n001 + n000
n    = n_tr + n_co

## proportions of the latent strata
pi_n = (n101 + n100)/n_tr
pi_a = (n011 + n010)/n_co
pi_c = 1 - pi_n - pi_a

## four observed means of the outcomes (Z=z,D=d)
mean_y_11 = n111/(n111 + n110)
mean_y_10 = n101/(n101 + n100)
mean_y_01 = n011/(n011 + n010)
mean_y_00 = n001/(n001 + n000)

## means of the outcomes of two strata
mu_n1 = mean_y_10
mu_a0 = mean_y_01
## ER implies the following two means
mu_n0 = mu_n1
mu_a1 = mu_a0
## stratum (Z=1,D=1) is a mixture of c and a
mu_c1 = ((pi_c + pi_a)*mean_y_11 - pi_a*mu_a1)/pi_c
## stratum (Z=0,D=0) is a mixture of c and n
mu_c0 = ((pi_c + pi_n)*mean_y_00 - pi_n*mu_n0)/pi_c

## identifiable quantities from the observed data
list(pi_c = pi_c,
    pi_n = pi_n,
    pi_a = pi_a,
    mu_c1 = mu_c1,
    mu_c0 = mu_c0,
    mu_n1 = mu_n1,
    mu_n0 = mu_n0,
    mu_a1 = mu_a1,
    mu_a0 = mu_a0)
}
```

We then re-visit two canonical examples.

Example 22.1 Investigators et al. (2014) assess the effectiveness of the emergency endovascular versus the open surgical repair strategies for patients with a clinical diagnosis of ruptured aortic aneurism. Patients are randomized to either the emergency endovascular or the open repair strategy. The primary outcome is the survival status after 30 days. Let Z be the treatment assigned, with Z = 1 for the endovascular strategy and Z = 0 for the open repair. Let D be the treatment received. Let Y be the survival status, with Y = 1 for dead, and Y = 0 for alive. The estimate of $\tau _ { \mathrm { c } }$ is 0.131 with 95% confidence interval (−0.036, 0.298) including 0. Using the function above, we can obtain

**TABLE 22.2: Binary data and IV inequalities (a) Investigators et al. (2014)’s study**

<table><tr><td rowspan="2"></td><td colspan="2">Z=1</td><td colspan="2">Z=0</td></tr><tr><td>D=1</td><td>D=0</td><td>D=1</td><td>D=0</td></tr><tr><td>Y=1</td><td>107</td><td>68</td><td>24</td><td>131</td></tr><tr><td>Y=0</td><td>42</td><td>42</td><td>8</td><td>79</td></tr></table>

**(b) Hirano et al. (2000)’s study**

<table><tr><td rowspan="2"></td><td colspan="2">Z=1</td><td colspan="2">Z=0</td></tr><tr><td>D=1</td><td>D=0</td><td>D=1</td><td>D=0</td></tr><tr><td>Y=1</td><td>31</td><td>85</td><td>30</td><td>99</td></tr><tr><td>Y=0</td><td>424</td><td>944</td><td>237</td><td>1041</td></tr></table>

\$mu  c1

[1] 0.7086064

\$mu  c0

[1] 0.6292042

There is no evidence of violating the IV assumptions.

Example 22.2 In Hirano et al. (2000), physicians are randomly selected to receive a letter encouraging them to inoculate patients at risk for flu. The treatment is the actual flu shot, and the outcome is an indicator for flu-related hospital visits. However, some patients do not comply with their assignments. Let $Z _ { i }$ be the indicator of encouragement to receive the flu shot, with Z = 1 if the physician receives the encouragement letter, and Z = 0 otherwise. Let D be the treatment received. Let Y be the outcome, with Y = 0 if for a flu-related hospitalization during the winter, and Y = 1 otherwise. The estimate of $\tau _ { \mathrm { c } }$ is 0.116 with 95% confidence interval (−0.061, 0.293) including 0. Using the function above, we can obtain

\$mu  c1

[1] -0.004548064

\$mu  c0

[1] 0.1200094

Since $\hat { \mu } _ { \mathrm { 1 c } } < 0$ , there is evidence of violating the IV assumptions.

## 22.4 Homework problems

## 22.1 Risk ratio for compliers

With binary outcome, we can define the risk ratio for compliers as

$$
\mathrm{RR} _ {\mathrm{c}} = \frac {\operatorname* {p r} \{Y (1) = 1 \mid U = \mathrm{c} \}}{\operatorname* {p r} \{Y (0) = 1 \mid U = \mathrm{c} \}}.
$$

Show that under Assumptions 21.1–21.3, we can identify it by

$$
\mathrm{RR} _ {\mathrm{c}} = \frac {E (D Y \mid Z = 1) - E (D Y \mid Z = 0)}{E \{(D - 1) Y \mid Z = 1 \} - E \{(D - 1) Y \mid Z = 0 \}}.
$$

Remark: Using Theorem 22.1, we can identify any comparisons between $E \{ Y ( 1 ) \mid U = \operatorname { c } \}$ and $E \{ Y ( 0 ) \mid U = \operatorname { c } \}$ .

## 22.2 Disentangle the mixtures: distributional results

This problem extends Theorem 22.1. Define

$$
f _ {z u} (y) = \operatorname{pr} \{Y (z) = y \mid U = u \}, \quad (d = 0, 1; u = \mathrm{a}, \mathrm{n}, \mathrm{c})
$$

as the density of $Y ( z )$ for latent stratum $U = u ,$ and define

$$
g _ {z d} (y) = \operatorname{pr} (Y = y \mid Z = z, D = d)
$$

as the density of the outcome within the observed group $( Z = z , D = d )$ . Show Theorem 22.3 below.

Theorem 22.3 Under Assumptions 21.1–21.3, we can identify the typespecific densities of the potential outcomes by

$$
f _ {1 \mathrm{n}} (y) = f _ {0 \mathrm{n}} (y) \equiv f _ {\mathrm{n}} (y) = g _ {1 0} (y),
$$

$$
f _ {1 \mathrm{a}} (y) = f _ {0 \mathrm{a}} (y) \equiv f _ {\mathrm{a}} (y) = g _ {0 1} (y),
$$

$$
f _ {1 c} (y) = \pi_ {c} ^ {- 1} \left\{\operatorname{pr} (D = 1 \mid Z = 1) g _ {1 1} (y) - \operatorname{pr} (D = 1 \mid Z = 0) g _ {0 1} (y) \right\},
$$

$$
f _ {0 \mathrm{c}} (y) = \pi_ {\mathrm{c}} ^ {- 1} \{\operatorname * {p r} (D = 0 | Z = 0) g _ {0 0} (y) - \operatorname * {p r} (D = 0 | Z = 1) g _ {1 0} (y) \}.
$$

## 22.3 Alternative form of Theorem 22.2

The inequalities in (22.1) can be re-written as

$$
\operatorname{pr} (D = 1, Y = y \mid Z = 1) \geq \operatorname{pr} (D = 1, Y = y \mid Z = 0),
$$

$$
\operatorname{pr} (D = 0, Y = y \mid Z = 0) \geq \operatorname{pr} (D = 0, Y = y \mid Z = 1)
$$

for both $y = 0 , 1$ .


## 22.4 Instrumental variable inequalities for a general outcome

For a general outcome Y , show that Assumptions 21.1–21.3 imply

$$
\operatorname{pr} (D = 1, Y \geq y \mid Z = 1) \geq \operatorname{pr} (D = 1, Y \geq y \mid Z = 0),
$$

$$
\operatorname{pr} (D = 1, Y <   y \mid Z = 1) \geq \operatorname{pr} (D = 1, Y <   y \mid Z = 0),
$$

$$
\operatorname{pr} (D = 0, Y \geq y \mid Z = 0) \geq \operatorname{pr} (D = 0, Y \geq y \mid Z = 1),
$$

$$
\operatorname{pr} (D = 0, Y <   y \mid Z = 0) \geq \operatorname{pr} (D = 0, Y <   y \mid Z = 1)
$$

for all y.

Remark: Imbens and Rubin (1997) and Kitagawa (2015) discussed similar results. For instance, we can test the first inequality based on an analog of the Kolmogorov–Smirnov statistic:

$$
\mathrm{KS} _ {1} = \max _ {y} \Big | \frac {\sum_ {i = 1} ^ {n} Z _ {i} D _ {i} 1 (Y _ {i} \leq y)}{\sum_ {i = 1} ^ {n} Z _ {i} D _ {i}} - \frac {\sum_ {i = 1} ^ {n} (1 - Z _ {i}) D _ {i} 1 (Y _ {i} \leq y)}{\sum_ {i = 1} ^ {n} (1 - Z _ {i}) D _ {i}} \Big |.
$$

## 22.5 Example for the IV inequalities

Give an example in which all the IV inequalities hold and another example in which not all the IV inequalities hold. You need to specify the joint distribution of (Z, D, Y ) with binary Z and D.

## 22.6 Violations of the key assumptions

Theorem 21.1 relies on randomization, monotonicity, and exclusion restriction. The latter two are not testable even in randomized experiments. When they are violated, the IV estimator no longer identifies the complier average causal effect. This problem gives two cases below, which are restatement of Propositions 2 and 3 in Angrist et al. (1996).

Under Assumptions 21.1 and 21.2 without the exclusion restriction, we have

$$
\frac {E (Y \mid Z = 1) - E (Y \mid Z = 0)}{E (D \mid Z = 1) - E (D \mid Z = 0)} - \tau_ {\mathrm{c}} = \frac {\pi_ {\mathrm{a}} \tau_ {\mathrm{a}} + \pi_ {\mathrm{n}} \tau_ {\mathrm{n}}}{\pi_ {\mathrm{c}}}
$$

where

$$
\tau_ {u} = E \{Y (1) - Y (0) \mid U = u \}, (U = \mathrm{a,n,c}).
$$

Under Assumptions 21.1 and 21.3 without the monotonicity, we have

$$
\frac {E (Y \mid Z = 1) - E (Y \mid Z = 0)}{E (D \mid Z = 1) - E (D \mid Z = 0)} - \tau_ {\mathrm{c}} = \frac {\pi_ {\mathrm{d}} (\tau_ {\mathrm{c}} + \tau_ {\mathrm{d}})}{\pi_ {\mathrm{c}} - \pi_ {\mathrm{d}}}.
$$

Prove the above two results.

## 22.7 Problems of other analyses

In the process of deriving the IV inequalities in Section 22.1, we disentangled the mixture distributions by identifying the proportions of the latent strata as well as the conditional means of their potential outcomes. These results are helpful for understanding the drawbacks of other seemingly reasonable analyses. I review three estimators below and suppose Assumptions 21.1–21.3 holds.

1. The as-treated analysis compares the means of the outcomes among units receiving the treatment and control, yielding

$$
\tau_ {\mathrm{AT}} = E (Y \mid D = 1) - E (Y \mid D = 0).
$$

Show that

$$
\tau_ {\mathrm{AT}} = \frac {\pi_ {\mathrm{a}} \mu_ {\mathrm{a}} + \mathrm{pr} (Z = 1) \pi_ {\mathrm{c}} \mu_ {1 \mathrm{c}}}{\mathrm{pr} (D = 1)} - \frac {\pi_ {\mathrm{n}} \mu_ {\mathrm{n}} + \mathrm{pr} (Z = 0) \pi_ {\mathrm{c}} \mu_ {0 \mathrm{c}}}{\mathrm{pr} (D = 0)}.
$$

2. The per-protocol analysis compares the units who comply with the treatment assigned in treatment and control groups, yielding

$$
\tau_ {\mathrm{PP}} = E (Y \mid Z = 1, D = 1) - E (Y \mid Z = 0, D = 0).
$$

Show that

$$
\tau_ {\mathrm{pp}} = \frac {\pi_ {\mathrm{a}} \mu_ {\mathrm{a}} + \pi_ {\mathrm{c}} \mu_ {\mathrm{1c}}}{\pi_ {\mathrm{a}} + \pi_ {\mathrm{c}}} - \frac {\pi_ {\mathrm{n}} \mu_ {\mathrm{n}} + \pi_ {\mathrm{c}} \mu_ {\mathrm{0c}}}{\pi_ {\mathrm{n}} + \pi_ {\mathrm{c}}}.
$$

3. We may also want to compare the outcomes among units receiving the treatment and control, conditioning on their treatment assignment, yielding

$$
\tau_ {Z = 1} = E (Y \mid Z = 1, D = 1) - E (Y \mid Z = 1, D = 0),
$$

$$
\tau_ {Z = 0} = E (Y \mid Z = 0, D = 1) - E (Y \mid Z = 0, D = 0).
$$

Show that they reduce to

$$
\tau_ {Z = 1} = \frac {\pi_ {\mathrm{a}} \mu_ {\mathrm{a}} + \pi_ {\mathrm{c}} \mu_ {\mathrm{1c}}}{\pi_ {\mathrm{a}} + \pi_ {\mathrm{c}}} - \mu_ {\mathrm{n}}, \quad \tau_ {Z = 0} = \mu_ {\mathrm{a}} - \frac {\pi_ {\mathrm{n}} \mu_ {\mathrm{n}} + \pi_ {\mathrm{c}} \mu_ {\mathrm{0c}}}{\pi_ {\mathrm{n}} + \pi_ {\mathrm{c}}}.
$$

## 22.8 Bounds on the average causal effect on the whole population

Extend the discussion in Section 22.1 based on the notation in Section 21.6. With the potential outcome Y (d), define the average causal effect of the treatment received on the outcome as

$$
\delta = E \{Y (d = 1) - Y (d = 0) \},
$$

and modify the definition of $\mu _ { d u }$ as

$$
m _ {d u} = E \{Y (d) \mid U = u \}, \quad (z = 0, 1; u = \mathrm{a,n,c}).
$$

They satisfy

$$
\delta = \sum_ {u = \mathrm{a}, \mathrm{n}, \mathrm{c}} \pi_ {u} (m _ {1 u} - m _ {0 u}).
$$

## 27622 Disentangle Mixture Distributions and Instrumental Variable Inequalities

Section 22.1 identifies $\pi _ { \mathrm { a } } , \pi _ { \mathrm { n } } , \pi _ { \mathrm { c } } , m _ { 1 \mathrm { a } } = \mu _ { 1 \mathrm { a } } , m _ { 0 \mathrm { n } } = \mu _ { 0 \mathrm { n } } , m _ { 1 \mathrm { c } } = \mu _ { 1 \mathrm { c } }$ and $m _ { 0 \mathrm { c } } = \mu _ { 0 \mathrm { c } }$ . But the data do not contain any information about $m _ { \mathrm { 0 a } }$ and $m _ { 1 \mathrm { n } } .$ . Therefore, we cannot identify δ. With a bounded outcome, we can bound δ. Show the following result:

Theorem 22.4 Under Assumptions $\it { 2 1 . 2 \mathrm { - } 2 1 . 4 }$ with a bounded outcome in $[ y , { \overline { { y } } } ]$ , we have $\underline { { { \delta } } } \le \delta \le \overline { { { \delta } } }$ , where

$$
\underline {{\delta}} = \delta^ {\prime} - \bar {y} \operatorname{pr} (D = 1 \mid Z = 0) + \underline {{y}} \operatorname{pr} (D = 0 \mid Z = 1)
$$

and

$$
\overline {{{{\delta}}}} = \delta^ {\prime} - \underline {{{{y}}}} \mathrm{pr} (D = 1 \mid Z = 0) + \overline {{{{y}}}} \mathrm{pr} (D = 0 \mid Z = 1)
$$

$w i t h \ \delta ^ { \prime } = E ( D Y \mid Z = 1 ) - E ( Y - D Y \mid Z = 0 ) .$

Remark: In the special case with a binary outcome, the bounds simplify to

$$
\underline {{\delta}} = E (D Y \mid Z = 1) - E (D + Y - D Y \mid Z = 0)
$$

and

$$
\overline {{\delta}} = E (D Y + 1 - D \mid Z = 1) - E (Y - D Y \mid Z = 0).
$$

## 22.9 One-sided noncompliance and statistical inference

Consider a randomized encouragement design where the units assigned to the control have no access to the treatment. For unit $i ,$ let $Z _ { i }$ be the binary treatment assigned, $D _ { i }$ be the binary treatment received, and $Y _ { i }$ be the outcome of interest. One-sided noncompliance happens when

$$
Z _ {i} = 0 \Longrightarrow D _ {i} = 0 (i = 1, \dots , n).
$$

Suppose that Assumption 21.1 holds.

1. Does monotonicity Assumption 21.2 hold in this case? How many latent strata defined by $\{ D _ { i } ( 1 ) , D _ { i } ( 0 ) \}$ are there in this problem? How do we identify their proportions by the observed data distribution?  
2. State the assumption of exclusion restriction. Under exclusion restriction, show that $E \{ Y ( z ) \mid U = u \}$ can be identified by the observed data distributions. Give the formulas for all possible values of z and u. How do we identify the complier average causal effect in this case?  
3. If we observe pretreatment covariates $X _ { i }$ for all units $i ,$ how do we use the covariate information to improve the estimation efficiency of the complier average causal effect?  
4. Under Assumption 21.1, the exclusion restriction Assumption 21.3 has testable implications, which are the IV inequalities for one-sided noncompliance. State the IV inequalities.

5. Sommer and Zeger (1991) provided the following dataset:

<table><tr><td rowspan="2"></td><td colspan="2">Z=1</td><td colspan="2">Z=0</td></tr><tr><td>D=1</td><td>D=0</td><td>D=1</td><td>D=0</td></tr><tr><td>Y=1</td><td>9663</td><td>2385</td><td>0</td><td>11514</td></tr><tr><td>Y=0</td><td>12</td><td>34</td><td>0</td><td>74</td></tr></table>

Re-analyze it.

Remark: Bloom (1984) first discussed one-sided noncompliance and proposed the IV estimator $\hat { \tau } _ { \mathrm { c } } = \hat { \tau } _ { Y } / \hat { \tau } _ { D }$ . His notation is different from this chapter.

## 22.10 One-sided noncompliance with partial adherence

Sanders and Karim (2021, Table 3) reported the following data from a randomized clinical trial aiming to estimate the efficacy of smoking cessation interventions among individuals with psychotic disorders.

<table><tr><td>group assigned</td><td>treatment received</td><td>group size</td><td># positive outcomes</td></tr><tr><td>Control</td><td>None</td><td>151</td><td>25</td></tr><tr><td>Treatment</td><td>None</td><td>35</td><td>7</td></tr><tr><td>Treatment</td><td>Partial</td><td>42</td><td>17</td></tr><tr><td>Treatment</td><td>Full</td><td>70</td><td>40</td></tr></table>

Three tiers of treatment received are defined as follows: “full” treatment corresponds to attending all 8 treatment sessions, “partial” corresponds to attending 5 to 7 sessions, and “none” corresponds to < 5 sessions. The outcome is defined as the binary indicator of smoking reduction of 50% or greater relative to baseline, measured at three months.

In this problem, the treatment assignment Z is binary but the treatment received D takes three values 0, 0.5, 1 for “none”, “partial”, and “full.” The three-leveled D causes complications, but it can only be 0 under the control assignment. How many latent strata $U = \{ D ( 1 ) , D ( 0 ) \}$ do we have in this problem? Can we identify their proportions?

How do we extend the exclusion restriction to this problem? What can be the causal effects of interest? Can we identify them?

Analyze the data based on the questions above.

## 22.11 Recommended reading

Balke and Pearl (1997) derived more general IV inequalities.
# A First Course in Causal Inference

*arXiv:2305.18793v1 [stat.ME] 30 May 2023*



## Preface

I developed the lecture notes based on my “Causal Inference” course at the University of California Berkeley over the past seven years. Since half of the students were undergraduate, my lecture notes only require basic knowledge of probability theory, statistical inference, and linear and logistic regressions.

I am grateful for the constructive comments from many students. If you identify any errors, please feel free to email me.


**Acronyms**

<table><tr><td>acronym</td><td>full name</td><td>first chapter</td></tr><tr><td>RD</td><td>risk difference</td><td>1</td></tr><tr><td>RR</td><td>risk ratio or relative risk</td><td>1</td></tr><tr><td>OR</td><td>odds ratio</td><td>1</td></tr><tr><td>RCT</td><td>randomized controlled trial</td><td>1</td></tr><tr><td>BMI</td><td>body mass index</td><td>2</td></tr><tr><td>SUTVA</td><td>stable unit treatment value assumption</td><td>2</td></tr><tr><td>ACE</td><td>average causal effect</td><td>2</td></tr><tr><td>CRE</td><td>completely randomized experiment</td><td>3</td></tr><tr><td>BRE</td><td>Bernoulli randomized experiment</td><td>3</td></tr><tr><td>IID</td><td>independent and identically distributed</td><td>3 and A1</td></tr><tr><td>FRT</td><td>Fisher randomization test</td><td>3</td></tr><tr><td>OLS</td><td>ordinary least squares</td><td>4 and A2</td></tr><tr><td>EHW</td><td>Eicker-Huber-White (robust standard error)</td><td>4 and A2</td></tr><tr><td>SRE</td><td>stratified randomized experiment</td><td>5</td></tr><tr><td>ReM</td><td>rerandomization using the Mahalanobis distance</td><td>6</td></tr><tr><td>ANCOVA</td><td>analysis of covariance</td><td>6</td></tr><tr><td>LASSO</td><td>least absolute shrinkage and selection operator</td><td>6</td></tr><tr><td>MPE</td><td>matched-pairs experiment</td><td>7</td></tr><tr><td>NHANES</td><td>National Health and Nutrition Examination Survey</td><td>10</td></tr><tr><td>IPW</td><td>inverse propensity score weighting</td><td>11</td></tr><tr><td>HT</td><td>Horvitz-Thompson</td><td>11</td></tr><tr><td>WLS</td><td>weighted least squares</td><td>14 and A2</td></tr><tr><td>IV</td><td>instrumental variable</td><td>21</td></tr><tr><td>ITT</td><td>intention-to-treat (analysis)</td><td>21</td></tr><tr><td>CACE</td><td>complier average causal effect</td><td>21</td></tr><tr><td>LATE</td><td>local average treatment effect</td><td>21</td></tr><tr><td>TSLS</td><td>two-stage least squares</td><td>23</td></tr><tr><td>ILS</td><td>indirect least squares</td><td>23</td></tr><tr><td>MR</td><td>Mendelian randomization</td><td>25</td></tr><tr><td>SNP</td><td>single nucleotide polymorphism</td><td>25</td></tr><tr><td>NDE</td><td>natural direct effect</td><td>27</td></tr><tr><td>NIE</td><td>natural indirect effect</td><td>27</td></tr><tr><td>CDE</td><td>controlled direct effect</td><td>29</td></tr><tr><td>MSM</td><td>marginal structural model</td><td>29</td></tr><tr><td>FWL</td><td>Frisch-Waugh-Lovell (theorem)</td><td>A2</td></tr><tr><td>MLE</td><td>maximum likelihood estimate</td><td>A2</td></tr></table>


## Part I

## Introduction

# Correlation, Association, and the Yule–Simpson Paradox

Causality is central to human knowledge. Two famous quotes from ancient Greeks are below.

I would rather discover one causal law than be King of Persia.

— Democritus

We do not have knowledge of a thing until we grasped its cause.

— Aristotle

However, the major part of classic statistics is about association rather than causation. This chapter will review some basic association measures and point out their fundamental limitations.

## 1.1 Traditional view of statistics

A traditional view of statistics is to infer correlation or association among variables. Based on this view, there is no role for causal inference in statistics. Two famous aphorisms based on this view are below:

- “Correlation does not imply causation.”
- “You cannot prove causality with statistics.”

This book has a very different view: statistics is crucial for understanding causality. The main focus of this book is to introduce the formal language for causal inference and develop statistical methods to estimate causal effects in randomized experiments and observational studies.

## 1.2 Some commonly-used measures of association

## 1.2.1 Correlation and regression

The Pearson correlation coefficient between two random variables $Z$ and $Y$ is

$$
\rho_ {Z Y} = \frac {\operatorname{cov} (Z , Y)}{\sqrt {\operatorname{var} (Z) \operatorname{var} (Y)}},
$$

which measures the linear dependence of $Z$ and $Y$ .

The linear regression of $Y$ on $Z$ is the model

$$
Y = \alpha + \beta Z + \varepsilon , \tag {1.1}
$$

where $E(\varepsilon) = 0$ and $E(\varepsilon Z) = 0$ . We can show that the regression coefficient $\beta$ equals

$$
\beta = \frac {\operatorname{cov} (Z , Y)}{\operatorname{var} (Z)} = \rho_ {Z Y} \sqrt {\frac {\operatorname{var} (Y)}{\operatorname{var} (Z)}}.
$$

So $\beta$ and $\rho_{ZY}$ always have the same sign.

We can also define multiple regression of $Y$ on $Z$ and $X$ :

$$
Y = \alpha + \beta Z + \gamma X + \varepsilon , \tag {1.2}
$$

where $E(\varepsilon)=0$ , $E(\varepsilon Z)=0$ and $E(\varepsilon X)=0$ . We usually interpret $\beta$ as the “effect” of Z on Y, holding X constant or conditioning on X or controlling for X. Chapter A2 reviews the basics of linear regression.

More interestingly, the $\beta$ 's in the above two regressions (1.1) and (1.2) can be different; they can even have different signs. The following R code reanalyzed the LaLonde observational data used by Hainmueller (2012). The main question of interest is the “causal effect” of a job training program on earning. The regression controlling for all covariates gives coefficient 1067.5461 for treat, while the regression not controlling for any covariates gives coefficient -8506.4954 for treat.

```txt
> dat <- read.table("cps1re74.csv", header = TRUE)
> dat$u74 <- as.numeric(dat$re74==0)
> dat$u75 <- as.numeric(dat$re75==0)
>
> ## linear regression on the outcome
> lmoutcome = lm(re78 ~ ., data = dat)
> summary(lmoutcome)$coef[2, 1:2]
Estimate Std. Error
1067.5461 554.0595
>
> lmoutcome = lm(re78 ~ treat, data = dat)
> summary(lmoutcome)$coef[2, 1:2]
Estimate Std. Error
-8506.4954 712.7664
```

## 1.2.2 Contingency tables

We can represent the joint distribution of two binary variables Z and Y by a two-by-two contingency table. With $p_{zy} = \Pr(Z = z, Y = y)$ , we can summarize the joint distribution in the following table:

$$
\begin{array}{c c c} & Y = 1 & Y = 0 \\ \hline Z = 1 & p _ {1 1} & p _ {1 0} \\ Z = 0 & p _ {0 1} & p _ {0 0} \end{array}
$$

Viewing Z as the treatment or exposure and Y as the outcome, we can define the risk difference as

$$
\begin{array}{l} \mathrm{RD} = \operatorname * {p r} (Y = 1 \mid Z = 1) - \operatorname * {p r} (Y = 1 \mid Z = 0) \\ = \frac {p _ {1 1}}{p _ {1 1} + p _ {1 0}} - \frac {p _ {0 1}}{p _ {0 1} + p _ {0 0}}, \\ \end{array}
$$

the risk ratio as

$$
\begin{array}{l} \mathrm{RR} = \frac {\operatorname* {p r} (Y = 1 \mid Z = 1)}{\operatorname* {p r} (Y = 1 \mid Z = 0)} \\ = \left. \frac {p _ {1 1}}{p _ {1 1} + p _ {1 0}} \right/ \frac {p _ {0 1}}{p _ {0 1} + p _ {0 0}}, \\ \end{array}
$$

and the odds ratio $^{1}$ as

<!-- footnote -->

> - $^{1}$ In probability theory, the odds of an event is defined as the ratio of the probability that the event happens over the probability that the event does not happen.

<!-- footnote end -->

$$
\begin{array}{l} \text { OR } = \frac {\operatorname{pr} (Y = 1 \mid Z = 1) / \operatorname{pr} (Y = 0 \mid Z = 1)}{\operatorname{pr} (Y = 1 \mid Z = 0) / \operatorname{pr} (Y = 0 \mid Z = 0)} \\ = \frac {\frac {p _ {1 1}}{p _ {1 1} + p _ {1 0}} / \frac {p _ {1 0}}{p _ {1 1} + p _ {1 0}}}{\frac {p _ {0 1}}{p _ {0 1} + p _ {0 0}} / \frac {p _ {0 0}}{p _ {0 1} + p _ {0 0}}} \\ = \frac {p _ {1 1} p _ {0 0}}{p _ {1 0} p _ {0 1}}. \\ \end{array}
$$

The terminologies risk difference, risk ratio, and odds ratio come from epidemiology. Because the outcomes in epidemiology are often diseases, it is natural to use the name “risk” for the probability of having diseases.

We have the following simple facts for these measures.

Proposition 1.1 (1) The following statements are all equivalent $^{2}$ : $Z \perp Y$ , RD = 0, RR = 1, and OR = 1. (2) If $p_{zy}$ 's are all positive, then RD > 0 is equivalent to RR > 1 and is also equivalent to OR > 1 (3) OR ≈ RR if $\operatorname{pr}(Y = 1 \mid Z = 1)$ and $\operatorname{pr}(Y = 1 \mid Z = 0)$ are small.

<!-- footnote -->

> - $^{2}$ This book uses the notation $\perp\perp$ to denote independence or conditional independence of random variables. The notation is due to Dawid (1979).

<!-- footnote end -->

I leave the proofs of statements (1) and (2) as a homework problem. Statement (3) is informal. The approximation holds because the odds $p/(1-p)$ is close to the probability p for rare diseases with $p \approx 0$ : by Taylor expansion $p/(1-p) = p + p^{2} + \cdots \approx p$ . In epidemiology, if the outcome represents the occurrence of a rare disease, then it is reasonable to assume that $\Pr(Y = 1 \mid X = 1)$ and $\Pr(Y = 1 \mid X = 0)$ are small.

We can also define conditional versions of the RD, RR, and OR if the probabilities are replaced by the conditional probabilities given another variable X, i.e., $\Pr(Y=1\mid Z=1,X=x)$ and $\Pr(Y=1\mid Z=0,X=x)$ .

With frequencies $n_{zy} = \#\{i : Z_i = z, Y_i = y\}$ , we can summarize the observed data in the following two-by-two table:

$$
\begin{array}{c c c} & Y = 1 & Y = 0 \\ \hline Z = 1 & n _ {1 1} & n _ {1 0} \\ Z = 0 & n _ {0 1} & n _ {0 0} \end{array}
$$

We can estimate RD, RR, and OR by replacing the true probabilities by the sample proportions. In R, functions fisher.test performs exact test and chisq.test performs asymptotic test for $Z \perp Y$ based on a two-by-two table of observed data.

Example 1.1 Bertrand and Mullainathan (2004) conducted a randomized experiment on resumes to study the effect of perceived race on callbacks for interviews. They randomly assigned African-American- or White-sounding names on fictitious resumes to help-wanted ads in Boston and Chicago newspapers. The following two-by-two table summarizes perceived race and callback:

```txt
> resume = read.csv("resume.csv")
> Alltable = table(resume$race, resume$call)
> Alltable
```

```txt
0 1
black 2278 157
white 2200 235
```

The two rows have the same total count, so it is apparent that White names received more callbacks. Fisher's exact test below shows that this difference is statistically significant.

```txt
> fisher. test (Alltable)
```

Fisher's Exact Test for Count Data

```txt
data: Alltable
p-value = 4.759e-05
alternative hypothesis: true odds ratio is not equal to 1
95 percent confidence interval:
1.249828 1.925573
sample estimates:
```

odds ratio

1.549732

## 1.3 An example of the Yule–Simpson Paradox

## 1.3.1 Data

The classic Kidney stone example is from Charig et al. (1986), where Z is the treatment with 1 for an open surgical procedure and 0 for a small puncture, and Y is the outcome with 1 for success and 0 for failure. The treatment and outcome data can be summarized in the following two-by-two table:

$$
\begin{array}{c c c} & Y = 1 & Y = 0 \\ \hline Z = 1 & 2 7 3 & 7 7 \\ Z = 0 & 2 8 9 & 6 1 \end{array}
$$

The estimated RD is

$$
\widehat {\mathrm{RD}} = \frac {2 7 3}{2 7 3 + 7 7} - \frac {2 8 9}{2 8 9 + 6 1} = 78 \% - 83 \% = -5 \% <  0.
$$

Treatment 0 seems better, that is, the small puncture leads to higher successful rate compared to the open surgical procedure.

However, the data were not from a randomized controlled trial (RCT) $^{3}$ . Patients receiving treatment 1 can be very different from patients receiving treatment 0. A “lurking variable” in this study is the severity of the case: some patients have smaller stones but some patients have larger stones. We can split the data according to the size of the stones.

<!-- footnote -->

> - $^{3}$ In an RCT, patients are randomly assigned to the treatment arms. Part II of this book will focus on RCTs.

<!-- footnote end -->

For patients with smaller stones, the treatment and outcome data can be summarized in the following two-by-two table:

$$
\begin{array}{c c c} & Y = 1 & Y = 0 \\ \hline Z = 1 & 8 1 & 6 \\ Z = 0 & 2 3 4 & 3 6 \end{array}
$$

For patients with larger stones, the treatment and outcome data can be summarized in the following two-by-two table:

$$
\begin{array}{c c c} & Y = 1 & Y = 0 \\ \hline Z = 1 & 1 9 2 & 7 1 \\ Z = 0 & 5 5 & 2 5 \end{array}
$$

The latter two tables must add up to the first table:

$$
8 1 + 1 9 2 = 2 7 3, \quad 6 + 7 1 = 7 7, \quad 2 3 4 + 5 5 = 2 8 9, \quad 3 6 + 2 5 = 6 1.
$$

From the table for patients with smaller stones, the estimated RD is

$$
\widehat {\mathrm{RD}} _ {\text { smaller }} = \frac {81}{81 + 6} - \frac {234}{234 + 36} = 93 \% - 87 \% = 6 \% > 0,
$$

suggesting that treatment 1 is better. From the table for patients with larger stones, the estimated RD is

$$
\widehat {\mathrm{RD}} _ {\text { larger }} = \frac {1 9 2}{1 9 2 + 7 1} - \frac {5 5}{5 5 + 2 5} = 73 \% - 69 \% = 4 \% > 0,
$$

also suggesting that treatment 1 is better.

The above data analysis leads to

$$
\widehat {\mathrm{RD}} <   0, \quad \widehat {\mathrm{RD}} _ {\text { smaller }} > 0, \quad \widehat {\mathrm{RD}} _ {\text { larger }} > 0.
$$

Informally, treatment 1 is better for both patients with smaller and larger stones, but treatment 1 is worse for the whole population. This interpretation is quite confusing if the goal is to infer the treatment effect. In statistics, this is called the Yule–Simpson or Simpson's Paradox in which the marginal association has the opposite sign to the conditional associations at all levels.

## 1.3.2 Explanation

Let X be the binary indicator with X = 1 for smaller stones and X = 0 for larger stones. Let us first take a look at the X-Z relationship by comparing the probabilities of receiving treatment 1 among patients with smaller and larger stones:

$$
\begin{array}{l} \widehat {\operatorname{pr}} (Z = 1 \mid X = 1) - \widehat {\operatorname{pr}} (Z = 1 \mid X = 0) \\ = \frac {8 1 + 6}{8 1 + 6 + 2 3 4 + 3 6} - \frac {1 9 2 + 7 1}{1 9 2 + 7 1 + 5 5 + 2 5} \\ = 24 \% - 77 \% \\ = -53\% <   0. \\ \end{array}
$$

So patients with larger stones tend to take treatment 1. Statistically, X and Z have negative association.

Let us then take a look at the X-Y relationship by comparing the probabilities of success among patients with smaller and larger stones: under treatment 1,

$$
\begin{array}{l} \widehat {\operatorname{pr}} (Y = 1 \mid Z = 1, X = 1) - \widehat {\operatorname{pr}} (Y = 1 \mid Z = 1, X = 0) \\ = \frac {8 1}{8 1 + 6} - \frac {1 9 2}{1 9 2 + 7 1} \\ = 93\% - 73 \% \\ = 20 \% > 0; \\ \end{array}
$$

![image_01](images/image_01.png)

FIGURE 1.1: A diagram for the kidney stone example. The signs indicate the associations of two variables, conditioning on other variables pointing to the downstream variable.  
under treatment 0,

$$
\widehat {\operatorname{pr}} (Y = 1 \mid Z = 0, X = 1) - \widehat {\operatorname{pr}} (Y = 1 \mid Z = 0, X = 0)
$$

$$
\begin{array}{l} \begin{array}{c c} 2 3 4 & 5 5 \end{array} \\ - \overline {{2 3 4 + 3 6}} - \overline {{5 5 + 2 5}} \\ = 87 \% - 69 \% \\ = 18 \% > 0. \\ \end{array}
$$

So under both treatment levels, patients with smaller stones have higher success probabilities. Statistically, X and Y have positive association conditional on both treatment levels.

We can summarize the qualitative associations in the diagram in Figure 1.1. In technical terms, the treatment has a positive direct path and a more negative indirect path to the outcome, so the overall association is negative between the treatment and outcome. In plain English, when less effective treatment 0 is applied more frequently to the less severe cases, it can appear to be a more effective treatment.

## 1.3.3 Geometry of the Yule–Simpson Paradox

**Assume that the $2 \times 2$ table based on the aggregated data has counts**

<table><tr><td>whole population</td><td>$ Y = 1 $</td><td>$ Y = 0 $</td></tr><tr><td>$ Z = 1 $</td><td>$ n_{11} $</td><td>$ n_{10} $</td></tr><tr><td>$ Z = 0 $</td><td>$ n_{01} $</td><td>$ n_{00} $</td></tr></table>

The two $2 \times 2$ tables based on subgroups with $X = 1$ and $X = 0$ have counts

<table><tr><td>subpopulation X = 1</td><td>Y = 1</td><td>Y = 0</td></tr><tr><td>Z = 1</td><td> $n_{11|1}$ </td><td> $n_{10|1}$ </td></tr><tr><td>Z = 0</td><td> $n_{01|1}$ </td><td> $n_{00|1}$ </td></tr><tr><td>subpopulation X = 0</td><td>Y = 1</td><td>Y = 0</td></tr><tr><td>Z = 1</td><td> $n_{11|0}$ </td><td> $n_{10|0}$ </td></tr><tr><td>Z = 0</td><td> $n_{01|0}$ </td><td> $n_{00|0}$ </td></tr></table>

Figure 1.2 shows the geometry of the Yule–Simpson Paradox. The y-axis shows the count of successes with Y = 1 and the x-axis shows the count of failures with Y = 0. The two parallelograms corresponds to aggregating the counts of successes and failures under two treatment levels. The slope of $OA_{1}$ is larger than that of $OB_{1}$ , and the slope of $OA_{0}$ is larger than that of $OB_{0}$ . So the treatment seems beneficial to the outcome within both levels of X. However, the slope of OA is smaller than that of OB. So the treatment seems harmful to the outcome for the whole population. The Yule–Simpson Paradox arises.

## 1.4 The Berkeley graduate school admission data

Bickel et al. (1975) investigated the admission rates of male and female students into the graduate school of Berkeley. The R package datasets contains the original data UCBAdmissions. The raw data by the six largest departments are shown below:

> library(datasets)

1.4 The Berkeley graduate school admission data

```python
> UCBAdmissions = aperm(UCBAdmissions, c(2, 1, 3))
> UCBAdmissions
, , Dept = A
```

**Admit**

<table><tr><td>Gender</td><td>Admitted</td><td>Rejected</td></tr><tr><td>Male</td><td>512</td><td>313</td></tr><tr><td>Female</td><td>89</td><td>19</td></tr></table>

```python
, , Dept = B
```

**Admit**

<table><tr><td>Gender</td><td>Admitted</td><td>Rejected</td></tr><tr><td>Male</td><td>353</td><td>207</td></tr><tr><td>Female</td><td>17</td><td>8</td></tr></table>

```python
, , Dept = C
```

**Admit**

<table><tr><td>Gender</td><td>Admitted</td><td>Rejected</td></tr><tr><td>Male</td><td>120</td><td>205</td></tr><tr><td>Female</td><td>202</td><td>391</td></tr></table>

```txt
，，Dept = D
```

**Admit**

<table><tr><td>Gender</td><td>Admitted</td><td>Rejected</td></tr><tr><td>Male</td><td>138</td><td>279</td></tr><tr><td>Female</td><td>131</td><td>244</td></tr></table>

```txt
, , Dept = E
```

**Admit**

<table><tr><td>Gender</td><td>Admitted</td><td>Rejected</td></tr><tr><td>Male</td><td>53</td><td>138</td></tr><tr><td>Female</td><td>94</td><td>299</td></tr></table>

```python
, , Dept = F
```

**Admit**

<table><tr><td>Gender</td><td>Admitted</td><td>Rejected</td></tr><tr><td>Male</td><td>22</td><td>351</td></tr><tr><td>Female</td><td>24</td><td>317</td></tr></table>

Aggregating the data over departments, we have a simple two-by-two table:

```julia
> UCBAdmissions.sum = apply(UCBAdmissions, c(1, 2), sum)
> UCBAdmissions.sum
Admit
Gender Admitted Rejected
```

<table><tr><td>Male</td><td>1198</td><td>1493</td></tr><tr><td>Female</td><td>557</td><td>1278</td></tr></table>

The following function, building upon chisq.test, have a two-by-two table as the input and the estimated RD and p-value as output:

```diff
> risk.difference = function(tb2)
+ {
+    p1 = tb2[1, 1]/(tb2[1, 1] + tb2[1, 2])
+    p2 = tb2[2, 1]/(tb2[2, 1] + tb2[2, 2])
+    testp = chisq.test(tb2)
+
+    return(list(p.diff = p1 - p2,
+    pv = testp$p.value))
+ }
```

With this function, we find large and significant difference between the admission rates of male and female students:

```txt
> risk.difference(UCBAdmissions.sum)
$p.diff
[1] 0.1416454
$pv
[1] 1.055797e-21
```

Stratifying on the departments, we find smaller and insignificant differences between the admission rates of male and female students. In department A, the difference is significant but negative.

```txt
> P.diff = rep(0, 6)
> PV = rep(0, 6)
> for(dd in 1:6)
+ {
+ department = risk.difference(UCBAdmissions[, , dd])
+ P.diff[dd] = department$p.diff
+ PV[dd] = department$pv
+ }
>
> round(P.diff, 2)
[1] -0.20 -0.05 0.03 -0.02 0.04 -0.01
> round(PV, 2)
[1] 0.00 0.77 0.43 0.64 0.37 0.64
```

## 1.5 Homework Problems

## 1.1 Independence in two-by-two tables

Prove (1) and (2) in Proposition 1.1.

## 1.5 Homework Problems

## 1.2 Correlation and partial correlation

Consider a three-dimensional Normal random vector:

$$
\left( \begin{array}{c} X \\ Y \\ Z \end{array} \right) \sim \mathrm{N} \left(\left( \begin{array}{c} 0 \\ 0 \\ 0 \end{array} \right), \left( \begin{array}{c c c} 1 & \rho_ {X Y} & \rho_ {X Z} \\ \rho_ {X Y} & 1 & \rho_ {Y Z} \\ \rho_ {X Z} & \rho_ {Y Z} & 1 \end{array} \right)\right).
$$

The correlation coefficient between X and Y is $\rho_{XY}$ . There are many equivalent definitions of the partial correlation coefficient. For a multivariate Normal vector, let $\rho_{XY|Z}$ denote the partial correlation coefficient between X and Y given Z, which is defined as their correlation coefficient in the conditional distribution $(X,Y)\mid Z$ . Show that

$$
\rho_ {X Y | Z} = \frac {\rho_ {X Y} - \rho_ {X Z} \rho_ {Y Z}}{\sqrt {1 - \rho_ {X Z} ^ {2}} \sqrt {1 - \rho_ {Y Z} ^ {2}}}
$$

Give an example with $\rho_{XY} > 0$ and $\rho_{XY|Z} < 0$ .

Remark: This is the Yule–Simpson Paradox for a Normal random vector.

## 1.3 Specification searches

Section 1.2.1 re-analyses the data used by Hainmueller (2012) with R code in LalondeRegression.R. In total, the data contain 10 covariates and therefore $2^{10} = 1024$ possible subsets of covariates in the linear regression. Run 1024 linear regressions with all possible subsets of covariates, and report the regression coefficients of the treatment. How many are positively significant, how many are negatively significant, and how many are not significant? You can also report other interesting findings from these regressions.

## 1.4 More on racial discrimination

Section 1.2.2 re-analyses the data collected by Bertrand and Mullainathan (2004) with R code in resume.R. Conduct analyses separately for males and females. What do you find from these subgroup analyses?

## 1.5 Recommended reading

Bickel et al. (1975) is the original paper for the paradox reported in Section 1.4.



| 

一

一

## 2

# Potential Outcomes

## 2.1 Experimentalists' view of causal inference

Rubin (1975) and Holland (1986) made up the aphorism:

no causation without manipulation.

Not everybody agrees with this point of view. However, it is quite helpful to clarify ambiguity in thinking about causal relationships. This book follows this view and defines causal effects using the potential outcomes framework (Neyman, 1923; Rubin, 1974). In this framework, an experiment, or at least a thought experiment, has an intervention, a manipulation, or a treatment, and we are interested in its effect on an outcome or multiple outcomes.

Example 2.1 If we are interested in the effect of taking aspirin or not on the relief of head ache, the intervention is taking aspirin.

Example 2.2 If we are interested in the effect of participating in a job training program or not on employment and wage, the intervention is participating in a job training program.

Example 2.3 If we are interested in the effect of studying in a small classroom or a large classroom on standardized test scores, the intervention is studying in a small classroom.

Example 2.4 Gerber et al. (2008) were interested in the effect of different get-out-to-vote messages on the voting behavior. The intervention is different get-out-to-vote messages.

Example 2.5 Pearl (2018) claimed that we could infer the effect of obesity on life span. A popular measure of obesity of the body mass index (BMI), defined as the body mass divided by the square of the body height in units of $kg/m^{2}$ . So the intervention can be BMI.

However, there are different levels of ambiguity of the interventions above. The meanings of interventions in Examples 2.1–2.4 are relatively clear, but the meaning of intervention on BMI in Example 2.5 is less clear. In particular, we can imagine different versions of BMI reduction: healthier diet, more physical exercise, bariatric surgery, etc. These different versions of intervention can have quite different effects on the outcome. In this book, we will view the intervention in Example 2.5 as ill-defined without further clarifications.

Another ill-defined intervention is race. Racial discrimination is an important issue in labor market, but it is not easy to imagine an experiment to change the race of any experimental unit. Bertrand and Mullainathan (2004) give an interesting experiment that partially answers the question.

Example 2.6 Bertrand and Mullainathan (2004) randomly change the names on the resumes, and compare the callback rates of resumes with African-American- or White-sounding names. For each resume, the intervention is the binary indicator of African-American- or White-sounding name, and the outcome is the binary indicator of callback. We have analyzed the following two-by-two table in Section 1.2.2:

<table><tr><td></td><td>callback</td><td>no callback</td></tr><tr><td>African-American</td><td>157</td><td>2278</td></tr><tr><td>White</td><td>235</td><td>2200</td></tr></table>

From the above, we can compare the the probabilities of being called back among African-American- and White-sounding names:

$$
\frac {157}{2278 + 157} - \frac {235}{2200 + 235} = 6.45 \% - 9.65 \% = -3.20 \% <   0
$$

with p-value from the Fisher exact test much smaller than 0.001.

In Bertrand and Mullainathan (2004)'s experiment, the treatment is the perceived race which can be manipulated by experimenters. They design an experiment to answer a well-defined causal question.

## 2.2 Formal notation of potential outcomes

Consider a study with n experimental units indexed by $i = 1, \ldots, n$ . As a starting point, we focus on a treatment with two levels: 1 for the treatment and 0 for the control. For each unit i, the outcome of interest Y has two versions:

$$
Y _ {i} (1) \text { and } Y _ {i} (0),
$$

which are potential outcomes under the hypothetical interventions 1 and 0. Neyman (1923) first used this notation. It seems intuitive but has some hidden assumptions. Rubin (1980) made the following clarifications on the hidden assumptions.

Assumption 2.1 (no interference) Unit i's potential outcomes do not depend on other units' treatments. This is sometimes called the no-interference assumption.

Assumption 2.2 (consistency) There are no other versions of the treatment. Equivalently, we require that the treatment level be well defined, or have no ambiguity at least for the outcome of interest. This is sometimes called the consistency assumption.

Assumption 2.1 can be violated in infectious diseases or network experiments. For instance, if some of my friends receive flu shots, my chance of getting the flu decrease even if I do not receive the flu shot; if my friends see an ad on Facebook, my chance of buying that product increase even if I do not see the ad. It is an active research area to study situations with interfering units in modern causal inference literature.

Assumption 2.2 can be violated for treatment with complex components. For instance, when studying the effect of cigarette smoking on lung cancer, the type of cigarettes may matter; when studying the effect of college education on income, the type and major of college education may matter.

Rubin (1980) called the Assumptions 2.1 and 2.2 above together the Stable Unit Treatment Value Assumption (SUTVA).

Assumption 2.3 (SUTVA) Both Assumptions 2.1 and 2.2 hold.

Under SUTVA, Rubin (2005) called the $n \times 2$ matrix of potential outcomes the Science Table:

<table><tr><td>i</td><td>$ Y_{i}(1) $</td><td>$ Y_{i}(0) $</td></tr><tr><td>1</td><td>$ Y_{1}(1) $</td><td>$ Y_{1}(0) $</td></tr><tr><td>2</td><td>$ Y_{2}(1) $</td><td>$ Y_{2}(0) $</td></tr><tr><td>$ \vdots $</td><td>$ \vdots $</td><td>$ \vdots $</td></tr><tr><td>n</td><td>$ Y_{n}(1) $</td><td>$ Y_{n}(0) $</td></tr></table>

Due to Neyman and Rubin's fundamental contribution to statistical causal inference, the potential outcomes framework is sometimes called the Neyman model, the Neyman-Rubin model, or the Rubin Causal Model.

Causal effects are functions of the Science Table. Inferring individual causal effects

$$
\tau_ {i} = Y _ {i} (1) - Y _ {i} (0)
$$

is fundamentally challenging because we can only observe either $Y_{i}(1)$ or $Y_{i}(0)$ for each unit i, that is, we can observed only half of the Science Table. As a starting point, most parts of the book focus on the average causal effect (ACE):

$$
\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \left\{Y _ {i} (1) - Y _ {i} (0) \right\} = n ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i} (1) - n ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i} (0).
$$

But we can easily extend our discussion to many other parameters (also called estimands).

## 2.2.1 Causal effects, subgroups, and the non-existence of Yule–Simpson Paradox

If we have two subgroups defined by a binary variable $x_{i}$ , we can define the subgroup causal effects as

$$
\tau_ {x} = \frac {\sum_ {i = 1} ^ {n} I (x _ {i} = x) \{Y _ {i} (1) - Y _ {i} (0) \}}{\sum_ {i = 1} ^ {n} I (x _ {i} = x)}, \quad (x = 0, 1)
$$

where $I(\cdot)$ is the indicator function. A simple identity is that

$$
\tau = \pi_ {1} \tau_ {1} + \pi_ {0} \tau_ {0}
$$

where $\pi_{x}=\sum_{i=1}^{n}I(x_{i}=x)/n$ is the proportion of units with $x_{i}=x\ (x=0,1)$ . Therefore, if $\tau_{1}>0$ and $\tau_{0}>0$ , we must have $\tau>0$ . The Yule–Simpson Paradox thus cannot happen to causal effects.

## 2.2.2 Subtlety of experimental unit

I end this section with a subtlety related to the definition of the experimental unit. Simply speaking, the experimental unit can be different from the physical unit. For example, if I did not take aspirin before and my headache did not go way, but I take aspirin now and my headache goes away, you might think that we can observed my potential outcomes under both control and treatment. Let i index myself, and let Y = 1 denote the indicator of no headache. Then, the above heuristic suggests that $Y_{i}(0) = 0$ and $Y_{i}(1) = 1$ , so it seems that aspirin kills my headache. But this logic is very wrong because of the misunderstanding of the definition of the experimental unit. At different time points, I, the same physical person, become two distinct experiment units, indexed by “i, before” and “i, after”. Therefore, we have four potential outcomes

$$
Y _ {i, \mathrm{before}} (0) = 0, Y _ {i, \mathrm{before}} (1) = ?, Y _ {i, \mathrm{after}} (0) = ?, Y _ {i, \mathrm{after}} (1) = 1,
$$

with two of them observed and two of them missing. The individual causal effects

$$
Y _ {i, \mathrm{before}} (1) - Y _ {i, \mathrm{before}} (0) = ? - 0 \mathrm{and} Y _ {i, \mathrm{after}} (1) - Y _ {i, \mathrm{after}} (0) = 1 -?
$$

are unknown. It is possible that my headache goes away even if I do not take aspirin:

$$
Y _ {i, \mathrm{after}} (0) = 1, Y _ {i, \mathrm{after}} (1) = 1
$$

which implies zero effect; it is also possible that my headache does not go away if I do not take aspirin:

$$
Y _ {i, \mathrm{after}} (0) = 0, Y _ {i, \mathrm{after}} (1) = 1
$$

which implies a positive effect of aspirin.

The wrong heuristic argument might get the right answer if the control potential outcomes are stable at the before and after periods: $Y_{i,\text{before}}(0) = Y_{i,\text{after}}(0) = 0$ . But this assumption is rather strong and fundamentally untestable.

## 2.3 Treatment assignment mechanism

Let $Z_{i}$ be the binary treatment indicator for unit i, vectorized as $Z = (Z_{1},\ldots ,Z_{n})$ . The observed outcome of unit i is a function of the potential outcomes and the treatment indicator:

$$
Y _ {i} = \left\{ \begin{array}{l l} Y _ {i} (1), & \text { if   } Z _ {i} = 1 \\ Y _ {i} (0), & \text { if   } Z _ {i} = 0 \end{array} \right. \tag {2.1}
$$

$$
= Z _ {i} Y _ {i} (1) + \left(1 - Z _ {i}\right) Y _ {i} (0) \tag {2.2}
$$

$$
= Y _ {i} (0) + Z _ {i} \{Y _ {i} (1) - Y _ {i} (0) \} \tag {2.3}
$$

$$
= Y _ {i} (0) + Z _ {i} \tau_ {i}. \tag {2.4}
$$

Equation (2.1) is the definition of the observed outcome. Equation (2.2) is equivalent to (2.1). It is a trivial fact, but Judea Pearl viewed it as the fundamental bridge between the potential outcomes and the observed outcome. Equations (2.3) and (2.4) highlight the fact that the individual causal effect $\tau_{i}=Y_{i}(1)-Y_{i}(0)$ can be heterogeneous across units.

The experiment only reveals one of unit $i$ 's potential outcomes with the other one missing:

$$
\begin{array}{l} Y _ {i} ^ {\text { mis }} = \left\{ \begin{array}{l l} Y _ {i} (0), & \text { if   } Z _ {i} = 1 \\ Y _ {i} (1), & \text { if   } Z _ {i} = 0 \end{array} \right. \\ = Z _ {i} Y _ {i} (0) + (1 - Z _ {i}) Y _ {i} (1). \\ \end{array}
$$

The missing potential outcome correspond to the opposite treatment level of unit i. For this reason, the potential outcomes framework is also called the counterfactual framework. This name can be confusing because before the experiment, both potential outcomes are observable, and after the experiment, one potential outcomes is actually observed.

The treatment assignment mechanism, i.e., the probability distribution of Z, plays an important role in inferring causal effects. The following simple numerical examples illustrate this point. We first generate potential outcomes from Normal distributions with the average causal effect close to -0.5.

$$
\begin{array}{l} > n = 5 0 0 \\ > \mathrm{Y0} = \text { rnorm(n) } \\ > \text { tau } = - 0. 5 + Y 0 \\ > \mathrm{Y} 1 = \mathrm{Y} 0 + \text { tau } \\ \end{array}
$$

A perfect doctor assigns the treatment to the patient if s/he knows that the individual causal effect is non-negative. This results in a positive difference in means of the observed outcomes:

$$
\begin{array}{l} > Z = (\text { tau } > = 0) \\ > \mathrm{Y} = \mathrm{Z} * \mathrm{Y} 1 + (1 - \mathrm{Z}) * \mathrm{Y} 0 \\ \end{array}
$$

> mean(Y[Z==1]) - mean(Y[Z==0])

[1] 2.166509

A clueless doctor does not know any information about the individual causal effects and assigns the treatment to patients by flipping a fair coin. This results in a difference in means of the observed outcomes close to the true average causal effect:

```txt
> Z = rbinom(n, 1, 0.5)
> Y = Z * Y1 + (1 - Z) * Y0
> mean(Y[Z == 1]) - mean(Y[Z == 0])
[1] -0.552064
```

The above examples are hypothetical since no doctors perfectly know the individual causal effects. However, the examples do demonstrate the crucial role of the treatment assignment mechanism. This book will organize the topics based on the treatment assignment mechanism.

## 2.4 Homework Problems

## 2.1 A perfect doctor

Following the first perfect doctor example in Section 2.3, assume the potential outcomes are random variables generated from

$$
Y (0) \sim \mathrm{N} (0, 1), \quad \tau = - 0. 5 + Y (0), \quad Y (1) = Y (0) + \tau .
$$

The binary treatment is determined by the treatment effect as $Z = 1(\tau \geq 0)$ , and the observed outcome is determined by the potential outcomes and the treatment by $Y = ZY(1) + (1 - Z)Y(0)$ . Calculate the difference in means

$$
E (Y \mid Z = 1) - E (Y \mid Z = 0).
$$

Hint: The mean of a truncated Normal random variable equals

$$
E (X \mid a <   X <   b) = \mu - \sigma \frac {\phi \left(\frac {b - \mu}{\sigma}\right) - \phi \left(\frac {a - \mu}{\sigma}\right)}{\Phi \left(\frac {b - \mu}{\sigma}\right) - \Phi \left(\frac {a - \mu}{\sigma}\right)},
$$

where $X \sim \mathrm{N}(\mu, \sigma^{2})$ , and $\phi(\cdot)$ and $\Phi(\cdot)$ are the probability density and cumulative distribution functions of a standard Normal random variable.

## 2.2 Nonlinear causal estimands

With potential outcomes $\{(Y_{i}(1), Y_{i}(0)\}_{i=1}^{n}$ for n units under the treatment and control, the difference in means equals the mean of the individual treatment effects:

$$
\bar {Y} (1) - \bar {Y} (0) = n ^ {- 1} \sum_ {i = 1} ^ {n} \{Y _ {i} (1) - Y _ {i} (0) \}.
$$

## 2.4 Homework Problems

Therefore, the average treatment effect is a linear causal estimand.

Other estimands may not be linear. For instance, we can define the median treatment effect as

$$
\delta_ {1} = \mathrm{median} \{(Y _ {i} (1) \} _ {i = 1} ^ {n} - \mathrm{median} \{(Y _ {i} (0) \} _ {i = 1} ^ {n},
$$

which is, in general, different from the median of the individual treatment effect

$$
\delta_ {2} = \mathrm{median} \{(Y _ {i} (1) - Y _ {i} (0) \} _ {i = 1} ^ {n}.
$$

1. Give numerical examples which have $\delta_1 = \delta_2$ , $\delta_1 > \delta_2$ , and $\delta_1 < \delta_2$ .  
2. Which estimand makes more sense, $\delta_1$ or $\delta_2$ ? Why? Use examples to justify your conclusion. If you feel that both $\delta_1$ and $\delta_2$ can make sense in different applications, you can also give examples to justify both estimands.

## 2.3 Average and individual effects

Give a numerical example in which $\tau = n^{-1} \sum_{i=1}^{n} \{Y_i(1) - Y_i(0)\} > 0$ but the proportion of units with $Y_i(1) > Y_i(0)$ is smaller than 0.5. That is, the average causal effect is positive, but the treatment benefits less than half of the units.

## 2.4 Recommended reading

Holland (1986) is a classic review article on statistical causal inference. It popularized the name “Rubin Causal Model” for the potential outcomes framework. At the University of California Berkeley, we call it the “Neyman Model” for obvious reasons.




# The Completely Randomized Experiment and the Fisher Randomization Test

The potential outcomes framework has intrinsic connections with randomized experiments. Understanding causal inference with various randomized experiments is fundamental and quite helpful for understanding causal inference in more complicated non-experimental studies.

Part II of this book focuses on randomized experiments. This chapter focuses on the simplest experiment, the completely randomized experiment (CRE).

## 3.1 CRE

Consider an experiment with n units, with $n_{1}$ receiving the treatment and $n_{0}$ receiving the control. We can define the CRE based on its treatment assignment mechanism $^{1}$ .

<!-- footnote -->

> - $^{1}$ Readers may think that a CRE has $Z_{i}$ 's as independent and identically distributed (IID) Bernoulli random variables with probability $\pi$ , in which $n_{1}$ is a Binomial( $n,\pi$ ) random variable. This is called the Bernoulli randomized experiment (BRE), which reduces to the CRE if we condition on $(n_{1},n_{0})$ . I will give more details for the BRE in Problem 4.7 in Chapter 4.

<!-- footnote end -->

Definition 3.1 (CRE) A CRE has the treatment assignment mechanism:

$$
\operatorname{pr} (\mathbf {Z} = \mathbf {z}) = 1 \bigg / \binom{n}{n _ {1}},
$$

where $\boldsymbol{z} = (z_1, \ldots, z_n)$ satisfies $\sum_{i=1}^{n} z_i = n_1$ and $\sum_{i=1}^{n} (1 - z_i) = n_0$ .

In Definition 3.1, we view the potential outcome vector under treatment $\mathbf{Y}(1) = (Y_{1}(1), \ldots, Y_{n}(1))$ and the potential outcome vector under control $\mathbf{Y}(0) = (Y_{1}(0), \ldots, Y_{n}(0))$ are both fixed. Even if we view them as random, we can condition on them and the treatment assignment mechanism becomes

$$
\operatorname{pr} \{\boldsymbol {Z} = \boldsymbol {z} \mid \boldsymbol {Y} (1), \boldsymbol {Y} (0) \} = 1 \bigg / \binom{n}{n _ {1}}
$$

because $\mathbf{Z} \perp \{\mathbf{Y}(1), \mathbf{Y}(0)\}$ in a CRE. In a CRE, the treatment vector $\mathbf{Z}$ is from a random permutation of $n_1$ 1's and $n_0$ 0's.

In his seminal book Design of Experiments, Fisher (1935) pointed out the following advantages of randomization:

1. It creates comparable treatment and control groups on average.  
2. It serves as a “reasoned basis” for statistical inference.

Point 1 is intuitive because the random treatment assignment does not bias toward the treatment or the control. Most people understand point 1 well. Point 2 is more subtle. What Fisher meant is that randomization justifies a statistical test, which is now called the Fisher Randomization Test (FRT). This chapter illustrates the basic idea of the FRT under a CRE.

## 3.2 FRT

Fisher (1935) was interested in testing the following null hypothesis:

$$
H _ {0 \mathrm{F}}: Y _ {i} (1) = Y _ {i} (0) \text {   for   all   units   } i = 1, \dots , n.
$$

Rubin (1980) called it the sharp null hypothesis in the sense that it can determine all the potential outcomes based on the observed data: $\mathbf{Y}(1)=\mathbf{Y}(0)=\mathbf{Y}=(Y_{1},\ldots,Y_{n})$ , the vector of the observed outcomes. It is also called the strong null hypothesis (e.g., Wu and Ding, 2021).

Conceptually, under $H_{0F}$ , the FRT works for any test statistic

$$
T = T (\mathbf {Z}, \mathbf {Y}), \tag {3.1}
$$

which is a function of the observed data. The observed outcome vector Y is fixed under $H_{0F}$ , so the only random component in the test statistic T is the treatment vector Z. The experimenter determines the distribution of Z, which in turn determines the distribution of T under $H_{0F}$ . This is the basis for calculating the p-value. I will give more details below.

In a CRE, Z is uniform over the set

$$
\left\{\boldsymbol {z} ^ {1}, \dots , \boldsymbol {z} ^ {M} \right\}
$$

where $M = \binom{n}{n_{1}}$ , and the $z^{m}$ 's are all possible vectors with $n_{1}$ 1's and $n_{0}$ 0's. For instance, with n = 5 and $n_{1} = 3$ , we can enumerate $M = \binom{5}{3} = 10$ vectors as follows:

```txt
> permutation10 = function(n, n1){
+ M = choose(n, n1)
+ treat.index = combn(n, n1)
+ Z = matrix(0, n, M)
```

## 3.2 FRT

+ for(m in 1:M){
+ treat = treat.index[, m]
+ Z[treat, m] = 1
+ }
+ Z
+ }
>
> permutation10(5, 3)
[ ,1] [ ,2] [ ,3] [ ,4] [ ,5] [ ,6] [ ,7] [ ,8] [ ,9] [ ,10]
[1, ] 1 1 1 1 1 1 0 0 0 0
[2, ] 1 1 1 0 0 0 1 1 1 0
[3, ] 1 0 0 1 1 0 1 1 0 1
[4, ] 0 1 0 1 0 1 1 0 1 1
[5, ] 0 0 1 0 1 1 0 1 1 1

As a consequence, $T$ is uniform over the set (with possible duplications)

$$
\{T (\boldsymbol {z} ^ {1}, \boldsymbol {Y}), \dots , T (\boldsymbol {z} ^ {M}, \boldsymbol {Y}) \}.
$$

That is, the distribution of $T$ is known due to the design of the CRE. We will call this distribution of $T$ the randomization distribution.

If larger values are more extreme for T, we can use the following tail probability to measure the extremeness of the test statistic with respect to its randomization distribution:

$$
p _ {\mathrm{FRT}} = M ^ {- 1} \sum_ {m = 1} ^ {M} I \{T (\boldsymbol {z} ^ {m}, \boldsymbol {Y}) \geq T (\boldsymbol {Z}, \boldsymbol {Y}) \}, \tag {3.2}
$$

which is called the p-value by Fisher. Figure 3.1 illustrates the computational process of $p_{FRT}$ .

![image_02](images/image_02.png)

```mermaid
graph TD
  A["(Z, Y) ⇒ T(Z, Y)"] --> B["(z¹, Y) ⇒ T(z¹, Y)"]
  A --> C["(z², Y) ⇒ T(z², Y)"]
  A --> D["..."]
  A --> E["(zᴹ, Y) ⇒ T(zᴹ, Y)"]
  B --> F[p_FRT = M⁻¹ Σ_{m=1}^M I{T(zᵐ, Y) ≥ T(Z, Y)}
  C --> F
  D --> F
  E --> F
```

FIGURE 3.1: Illustration of the FRT

The p-value, $p_{FRT}$ , in (3.2) works for any choice of test statistic and any outcome-generating process. It also extends naturally to any experiments, which will be a topic repeatedly discussed in the following chapters. Importantly, it is finite-sample exact in the sense $^{2}$ that under $H_{0F}$ ,

$$
\operatorname{pr} (p _ {\mathrm{FRT}} \leq u) \leq u \quad \text { for   all } \quad 0 \leq u \leq 1. \tag {3.3}
$$

In practice, M is often to large (e.g., with $n = 100, n_{1} = 50$ , we have $M > 10^{29}$ ), and it is computationally infeasible to enumerate all possible values of the treatment vector. We often approximate $p_{FRT}$ by Monte Carlo. To be more specific, we take simple random draws from the possible values of the treatment vector, or, equivalently, we randomly permute Z, and approximate $p_{FRT}$ by

$$
\hat {p} _ {\mathrm{FRT}} = R ^ {- 1} \sum_ {r = 1} ^ {R} I \{T (\boldsymbol {z} ^ {r}, \boldsymbol {Y}) \geq T (\boldsymbol {Z}, \boldsymbol {Y}) \}, \tag {3.4}
$$

where the $z^r$ 's the $R$ random permutations of $Z$ . The $p$ -value in (3.4) has Monte Carlo error decreasing fast with an increasing $R$ ; see Problem 3.2. Because the calculation of the $p$ -value in (3.4) involves permutations of $Z$ , the FRT is sometimes called the permutation test in the context of the CRE. However, the idea of FRT is more general than the permutation test in more complex experiments.

## 3.3 Canonical choices of the test statistic

From the above discussion, the FRT generates finite-sample exact p-value for any choice of test statistic. This is a feature of the FRT. However, this feature should not encourage arbitrary choice of the test statistic. Intuitively, we must choose test statistics that give information for the possible violations of $H_{0F}$ . Below I will review some canonical choices.

Example 3.1 (difference-in-means) The difference-in-means statistic is

$$
\hat {\tau} = \hat {\bar {Y}} (1) - \hat {\bar {Y}} (0)
$$

where

$$
\hat {\bar {Y}} (1) = n _ {1} ^ {- 1} \sum_ {Z _ {i} = 1} Y _ {i} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i}
$$

is the sample mean of the outcomes under the treatment and

$$
\hat {\bar {Y}} (0) = n _ {0} ^ {- 1} \sum_ {Z _ {i} = 0} Y _ {i} = n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) Y _ {i}
$$

is the sample mean of the outcomes under the control, respectively. Under $H_{0F}$ , it has mean

$$
E (\hat {\tau}) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} E (Z _ {i}) Y _ {i} - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} E (1 - Z _ {i}) Y _ {i} = 0
$$

and variance

$$
\begin{array}{l} \operatorname{var} (\hat {\tau}) = \operatorname{var} \left\{n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i} - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) Y _ {i} \right\} \\ = \quad \operatorname{var} \left(\frac {n}{n _ {0}} \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i}\right) \\ = _ {*} \frac {n ^ {2}}{n _ {0} ^ {2}} \left(1 - \frac {n _ {1}}{n}\right) \frac {s ^ {2}}{n _ {1}} \\ { = } { \frac { n } { n _ { 1 } n _ { 0 } } s ^ { 2 } , } \\ \end{array}
$$

where $= _{*}$ follows from Lemma A3.2 for simple random sampling with

$$
\bar {Y} = n ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i}, \quad s ^ {2} = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} (Y _ {i} - \bar {Y}) ^ {2}.
$$

Furthermore, the randomization distribution of $\hat{\tau}$ is approximately Normal due to the finite population central limit theorem in Lemma A3.4:

$$
\frac {\hat {\tau}}{\sqrt {\frac {n}{n _ {1} n _ {0}} s ^ {2}}} \rightarrow \mathrm{N} (0, 1) \tag {3.5}
$$

in distribution. Since $s^{2}$ is fixed under $H_{0F}$ , it is equivalent to use

$$
\frac {\hat {\tau}}{\sqrt {\frac {n}{n _ {1} n _ {0}} s ^ {2}}}
$$

as the test statistic in the FRT, which is asymptotically Normal as shown above. Then we can calculate an approximate p-value.

The observed data are $\{Y_i:Z_i = 1\}$ and $\{Y_i:Z_i = 0\}$ , so the problem is essentially a two-sample problem. Under the assumption of IID Normal outcomes (see Section A1.4.1), the classic two-sample $t$ -test assuming equal variance is based on

$$
\frac {\hat {\tau}}{\sqrt {\frac {n}{n _ {1} n _ {0} (n - 2)} \left[ \sum_ {Z _ {i} = 1} \{Y _ {i} - \hat {\bar {Y}} (1) \} ^ {2} + \sum_ {Z _ {i} = 0} \{Y _ {i} - \hat {\bar {Y}} (0) \} ^ {2} \right]}} \sim t _ {n - 2}. \tag {3.6}
$$

Based on some algebra (see Problem 3.8), we have the expansion

$$
(n - 1) s ^ {2} = \sum_ {Z _ {i} = 1} \left\{Y _ {i} - \hat {\bar {Y}} (1) \right\} ^ {2} + \sum_ {Z _ {i} = 0} \left\{Y _ {i} - \hat {\bar {Y}} (0) \right\} ^ {2} + \frac {n _ {1} n _ {0}}{n} \hat {\tau} ^ {2}. \tag {3.7}
$$

With a large sample size n, we can ignore the difference between $N(0,1)$ and $t_{n-2}$ and the difference between n-1 and n-2. Moreover, under $H_{0F}$ , $\hat{\tau}$ converges to zero in probability, so $n_{1}n_{0}/n\hat{\tau}^{2}$ can be ignored asymptotically. Therefore, under $H_{0F}$ , the approximate p-value in Example 3.1 is close to the p-value from the classic two-sample t-test assuming equal variance, which can be calculated by t.test with var.equal = TRUE. Under alternative hypotheses with nonzero $\tau$ , the additional term $\frac{n_{1}n_{0}}{n}\hat{\tau}^{2}$ in the above expansion can make the FRT less powerful than the usual t-test.

Based on the above discussion, the FRT with $\hat{\tau}$ effectively uses a pooled variance ignoring the heteroskedasticity between these two groups. In classical statistics, the two-sample problem with heteroskedastic Normal outcomes is called the Behrens–Fisher problem (see Section A1.4.1). In the Behrens–Fisher problem, a standard choice of the test statistic is the studentized statistic below.

Example 3.2 (studentized statistic) The studentized statistic is

$$
t _ {\mathrm{unequal}} = \frac {\hat {\bar {Y}} (1) - \hat {\bar {Y}} (0)}{\sqrt {\frac {\hat {S} ^ {2} (1)}{n _ {1}} + \frac {\hat {S} ^ {2} (0)}{n _ {0}}}},
$$

where

$$
\hat {S} ^ {2} (1) = (n _ {1} - 1) ^ {- 1} \sum_ {Z _ {i} = 1} \{Y _ {i} - \hat {\bar {Y}} (1) \} ^ {2}, \quad \hat {S} ^ {2} (0) = (n _ {0} - 1) ^ {- 1} \sum_ {Z _ {i} = 0} \{Y _ {i} - \hat {\bar {Y}} (0) \} ^ {2}
$$

are the sample variances of the observed outcomes under the treatment and control, respectively. Under $H_{0F}$ , the finite population central limit theorem again implies that t is asymptotically Normal:

$$
t \to \mathrm{N} (0, 1)
$$

in distribution. Then we can calculate an approximate p-value which is close to the p-value from t.test with var.equal = FALSE.

An extremely important point is that the FRT justifies the traditional t-tests using t.test with either var.equal = TRUE or var.equal = FALSE, even if the underlying distributions are not Normal. Standard statistics textbooks motivate the t-tests based on the Normality assumption, but the assumption is too strong. Fortunately, the t-test procedures can still be used as long as the finite population central limit theorems hold. Even if we do not believe the central limit theorems, we can still use $\hat{\tau}$ and t as test statistics in the FRT to obtain finite-sample exact p-values.

We will motivate this studentized statistic from another perspective in Chapter 8. The theory shows that using $t$ in FRT is more robust to heteroskedasticity across the two groups.

The following test statistic is robust to outliers resulting from heavy-tailed outcome data.

Example 3.3 (Wilcoxon rank sum) The difference-in-means statistic uses the original outcomes, and its sampling distribution depends on the second moments of the outcomes. This makes it sensitive to outliers. Another popular test statistic is based on the ranks of the pooled observed outcomes. Let $R_{i}$ denote the rank of $Y_{i}$ in the pooled samples Y:

$$
R _ {i} = \# \{j: Y _ {j} \leq Y _ {i} \}.
$$

The Wilcoxon rank sum statistic is the sum of the ranks under treatment:

$$
W = \sum_ {i = 1} ^ {n} Z _ {i} R _ {i}.
$$

For algebraic simplicity, we assume that there are no ties in the outcomes, although the FRT can be applied regardless of the existence of ties. For the case with ties, see Lehmann (1975, Chapter 1 Section 4). Because the sum of the ranks of the pooled samples are fixed at $1 + 2 + \cdots + n = n(n + 1)/2$ , the Wilcoxon statistic is equivalent to the difference in the means of the ranks under treatment and control. Under $H_{0F}$ , the $R_{i}$ 's are fixed, so W has mean

$$
E (W) = \sum_ {i = 1} ^ {n} E (Z _ {i}) R _ {i} = \frac {n _ {1}}{n} \sum_ {i = 1} ^ {n} i = \frac {n _ {1}}{n} \times \frac {n (n + 1)}{2} = \frac {n _ {1} (n + 1)}{2}
$$

and variance

$$
\begin{array}{l} \operatorname{var} (W) = \operatorname{var} \left(n _ {1} \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} R _ {i}\right) \\ = _ {*} n _ {1} ^ {2} \left(1 - \frac {n _ {1}}{n}\right) \frac {1}{n _ {1}} \frac {1}{n - 1} \sum_ {i = 1} ^ {n} \left(R _ {i} - \frac {n + 1}{2}\right) ^ {2} \\ = \frac {n _ {1} n _ {0}}{n (n - 1)} \sum_ {i = 1} ^ {n} \left(i - \frac {n + 1}{2}\right) ^ {2} \\ = \frac {n _ {1} n _ {0}}{n (n - 1)} \left\{\sum_ {i = 1} ^ {n} i ^ {2} - n \left(\frac {n + 1}{2}\right) ^ {2} \right\} \\ = \frac {n _ {1} n _ {0}}{n (n - 1)} \left\{\frac {n (n + 1) (2 n + 1)}{6} - n \left(\frac {n + 1}{2}\right) ^ {2} \right\} \\ = \frac {n _ {1} n _ {0} (n + 1)}{1 2}, \\ \end{array}
$$

where $=_{*}$ follows from Lemma A3.2. Furthermore, under $H_{0\mathrm{F}}$ , the finite population central limit theorem ensures that the randomization distribution of $\widehat{\tau}$ is approximately Normal:

$$
\frac {\sum_ {i = 1} ^ {n} Z _ {i} R _ {i} - \frac {n _ {1} (n + 1)}{2}}{\sqrt {\frac {n _ {1} n _ {0} (n + 1)}{1 2}}} \rightarrow \mathrm{N} (0, 1) \tag {3.8}
$$

in distribution. Based on (3.8), we can conduct an asymptotic test. In R, the function wilcox.test can compute both exact and asymptotic p-values based on the statistic $W - n_{1}(n_{1} + 1)/2$ . Based on some asymptotic analyses, Lehmann (1975) showed that the FRT using W has reasonable powers over a wide range of data generating processes.

Example 3.4 (Kolmogorov–Smirnov statistic) The treatment may affect the outcome in different ways. It seems natural to summarize the treatment outcomes and control outcomes based on the empirical distributions:

$$
\hat {F} _ {1} (y) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} I (Y _ {i} \leq y), \quad \hat {F} _ {0} (y) = n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) I (Y _ {i} \leq y).
$$

Comparing these two empirical distributions yields the famous Kolmogorov-Smirnov statistic

$$
D = \max _ {y} \left| \hat {F} _ {1} (y) - \hat {F} _ {0} (y) \right|.
$$

It is a challenging mathematics problem to derive the distribution of $D$ . With large sample sizes, its distribution function converges to

$$
\mathrm{pr} \left(\frac {n _ {1} n _ {0}}{n} D \leq x\right)\rightarrow \frac {\sqrt {2 \pi}}{x} \sum_ {j = 1} ^ {\infty} e ^ {- (2 j - 1) ^ {2} \pi^ {2} / (8 x ^ {2})},
$$

based on which we calculate an asymptotic p-value (Van der Vaart, 2000). In R, ks.test can compute both exact and asymptotic p-values.

## 3.4 A case study of the LaLonde experimental data

I use LaLonde (1986)’s experimental data to illustrate the FRT. The data are available in the Matching package (Sekhon, 2011):

Figure 3.2 shows the histograms of the outcomes under the treatment and control.

```txt
> library (Matching)
> data (lalonde)
> z = lalonde$treat
> y = lalonde$re78
```

The following code computes the observed values of the test statistics using existing functions:

```txt
> tauhat = t.test(y[z == 1], y[z == 0],
+    var.equal = TRUE)$statistic
> tauhat
t
2.835321
> student = t.test(y[z == 1], y[z == 0],
+    var.equal = FALSE)$statistic
> student
t
2.674146
> W = wilcox.test(y[z == 1], y[z == 0])$statistic
> W
W
27402.5
> D = ks.test(y[z == 1], y[z == 0])$statistic
> D
D
0.1321206
```

By randomly permuting the treatment vector, we can obtain the Monte Carlo approximation of the randomization distributions of the test statistics, stored in four vectors Tauhat, Student, Wilcox, and Ks.

```diff
> MC = 10^4
> Tauhat = rep(0, MC)
> Student = rep(0, MC)
> Wilcox = rep(0, MC)
> Ks = rep(0, MC)
> for(mc in 1:MC)
+ {
+    zperm = sample(z)
+    Tauhat[mc] = t.test(y[zperm == 1], y[zperm == 0],
+    var.equal = TRUE)$statistic
+    Student[mc] = t.test(y[zperm == 1], y[zperm == 0],
+    var.equal = FALSE)$statistic
+    Wilcox[mc] = wilcox.test(y[zperm == 1], y[zperm == 0])$statistic
+    Ks[mc] = ks.test(y[zperm == 1], y[zperm == 0])$statistic
+ }
```

The one-sided p-values based on the FRT are all smaller than 0.05:

```txt
> exact.pv = c(mean(Tauhat >= tauhat),
+    mean(Student >= student),
+    mean(Wilcox >= W),
+    mean(Ks >= D))
> round(exact.pv, 3)
[1] 0.002 0.002 0.006 0.040
```

Without using Monte Carlo, we can also compute the asymptotic p-values which are all smaller than 0.05:

```txt
> asym.pv = c(t.test(y[z == 1], y[z == 0],
+    var.equal = TRUE)$p.value,
+    t.test(y[z == 1], y[z == 0],
+    var.equal = FALSE)$p.value,
+    wilcox.test(y[z == 1], y[z == 0])$p.value,
+    ks.test(y[z == 1], y[z == 0])$p.value)
> round(asym.pv, 3)
[1] 0.005 0.008 0.011 0.046
```

The differences between the p-values are due to the asymptotic approximations as well as the fact that the default choices for t.test and wilcox.test are two-sided tests.

Figure 3.3 shows the histograms of the randomization distributions of four test statistics, as well as their corresponding observed values. For the first three test statistics, the Normal approximations works quite well even though the underlying outcome data distribution is far from Normal. In general, a figure like Figure 3.3 can give very clear information for testing the sharp null hypothesis. Recently, Bind and Rubin (2020) proposes, in the title of their paper, that “when possible, report a Fisher-exact p-value and display its underlying null randomization distribution.”

## 3.5 Some history of randomized experiments and FRT

## 3.5.1 James Lind’s experiment

James Lind (1716—1794) was a Scottish doctor and a pioneer of naval hygiene in the Royal Navy. At his time, scurvy was a major cause of death among sailors. He conducted one of the earliest randomized experiments with a clear documentation of the details, and concluded that citrus fruits cured scurvy before the discovery of Vitamin C.

In Lind (1753), he described the following randomized experiment with 12 patients of scurvy assigned to 6 groups. With some simplifications, the 6 groups are:

1. two received a quart of cider every day;  
2. two received twenty-five drops of sulfuric acid three times every day;  
3. two received two spoonfuls of vinegar three times every day;  
4. two received half a pint of seawater every day;  
5. two received two oranges and one lemon every day;  
6. two received a spicy paste plus a drink of barley water every day.

After six days, patients in the fifth group recovered, but patients in other groups did not. If we simplify the treatment as

$$
Z _ {i} = 1 (\text { unit   } i \text {   received   citrus   fruits })
$$

and the outcome as

$$
Y _ {i} = 1 (\text { unit   } i \text {   recovered   after   six   days }),
$$

then we have a $2 \times 2$ table

<table><tr><td></td><td> $Y_i = 1$ </td><td> $Y_i = 0$ </td></tr><tr><td> $Z_i = 1$ </td><td>2</td><td>0</td></tr><tr><td> $Z_i = 0$ </td><td>0</td><td>10</td></tr></table>

This is the extremest possible $2 \times 2$ table we can observe under this experiment, and the data contain strong evidence for the positive effect of citrus fruits for curing scurvy. Statistically, how do we measure the strength of the evidence?

Following the logic of the FRT, if the treatment has no effect at all (under $H _ { \mathrm { 0 F } } )$ , the extreme $2 \times 2$ table will occur with probability

$$
\frac {1}{\binom {1 2} {2}} = \frac {1}{6 6} = 0. 0 1 5
$$

which is the $p _ { \mathrm { F R T } }$ . This seems a surprise under $H _ { \mathrm { 0 F } } \colon$ we can easily reject $H _ { \mathrm { 0 F } }$ at the level 0.05.

## 3.5.2 Lady tasting tea

Fisher (1935) described the following famous experiment of Lady Tasting $T e a ^ { 3 }$ . A lady claimed that she could tell the difference between the two ways of making milk tea: one with milk added first, and the other with tea added first. This might sound odd to most people. As a statistician, Fisher designed an experiment to test whether the lady could tell the difference between the two ways of making milk tea.

He made 8 cups of tea, 4 with milk added first and the other 4 four with tea added first. Then he presented these 8 cups of tea in a random order to the lady, and asked the lady to pick up the 4 with milk added first. The final experiment result can be summarized in the following $2 \times 2$ table

<table><tr><td></td><td>milk first (lady)</td><td>tea first (lady)</td><td>column sum</td></tr><tr><td>milk first (Fisher)</td><td>X</td><td>4 - X</td><td>4</td></tr><tr><td>tea first (Fisher)</td><td>4 - X</td><td>X</td><td>4</td></tr><tr><td>row sum</td><td>4</td><td>4</td><td>8</td></tr></table>

The X can be 0, 1, 2, 3, 4. In the real experiment, ${ \overline { { X = 4 } } } ,$ which is the most extreme data, strongly suggesting that the lady could tell the difference of the two ways of making milk tea. Again, how do we measure the strength of the evidence?

Under the null hypothesis that the lady could not tell the difference, only one of the $\binom { 8 } { 4 } = 7 0$ possible orders yields the $2 \times 2$ table with $X = 4 .$ . So the p-value is

$$
p _ {\mathrm{FRT}} = \frac {1}{7 0} = 0. 0 1 4.
$$

Given the significance level 0.05, we reject the null hypothesis.

## 3.5.3 Two Fisherian principles for experiments

In the above two examples in Sections 3.5.1 and 3.5.2, the $p _ { \mathrm { F R T } } \mathrm { ^ { * } s }$ are justified by the randomization of the experiments. This highlightsthe first Fisherian principle of experiments: randomization.

Moreover, the above two experiments are in some sense the smallest possible experiments that can yield statistically meaningful results. For instance, if Lind only assign one patient to each of the six groups, then the smallest p-value is

$$
\frac {1}{\binom {6} {1}} = \frac {1}{6} = 0. 1 6 7;
$$

if Fisher only made 6 cups of tea, 3 with milk added first and the other 3 four with tea added first, then the smallest p-value is

$$
\frac {1}{\binom {6} {3}} = \frac {1}{2 0} = 0. 0 5.
$$

We can never reject the null hypotheses at the level of 0.05. This highlights the second Fisherian principle of experiments: replications.

Chapter 5 will discuss the third Fisherian principle of experiments: blocking. $i n g .$

## 3.6 Discussion

## 3.6.1 Other sharp null hypotheses and confidence intervals

I focus on the sharp null hypothesis $H _ { \mathrm { 0 F } }$ above. In fact, the logic of the FRT also works for other sharp null hypotheses. For instance, we can test

$$
H _ {0} (\pmb {\tau}): Y _ {i} (1) - Y _ {i} (0) = \tau_ {i} \text { for   all } i = 1, \ldots , n
$$

for a known vector $\tau = ( \tau _ { 1 } , \dots , \tau _ { n } )$ . Because the individual causal effects are all known under $H _ { 0 } ( \tau )$ , we can impute all missing potential outcomes based on the observed data. With known potential outcomes, the distribution of any test statistic is completely determined by the treatment assignment mechanism, and therefore, we can compute the corresponding $p _ { \mathrm { { F R T } } }$ as a function of $\tau ,$ denoted by $p _ { \mathrm { F R T } } ( \tau )$ . If we can specify all possible ${ \boldsymbol { \tau } } { \mathrm { { s } } } ,$ then we can compute a series of $p _ { \mathrm { F R T } } ( \tau ) \mathrm { { : } }$ . By duality of hypothesis testing and confidence set (see Section A1.2.5), we can obtain a (1 − α)-level confidence set for the average causal effect:

$$
\left\{\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \tau_ {i}: p _ {\mathrm{FRT}} (\pmb {\tau}) \geq \alpha \right\}.
$$

Although this strategy is conceptually straightforward, it has practical complexities due to the large number of all possible $\tau _ { \mathrm { } } ^ { \prime } \mathrm { s } .$ . In the special case of a binary outcome, Rigdon and Hudgens (2015) and Li and Ding (2016) proposed some computationally feasible approaches to constructing confidence intervals for τ based on the FRT. For general unbounded outcomes, this strategy is often computationally infeasible.

A canonical simplification is to consider a subclass of the sharp null hypotheses with constant individual causal effects:

$$
H _ {0} (c): Y _ {i} (1) - Y _ {i} (0) = c \text { for all } i = 1, \ldots , n
$$

for a known constant c. Given c, we can compute $p _ { \mathrm { F R T } } ( c )$ . By duality, we can obtain a $( 1 - \alpha )$ )-level confidence set for the average causal effect:

$$
\{c: p _ {\mathrm{FRT}} (c) \geq \alpha \}.
$$

Because this procedure only involves one-dimensional search, it is computationally feasible. However, it is often criticized that the constant individual causal effect assumption is too strong which does not hold for a binary outcome in particular.

## 3.6.2 Other test statistics

The FRT is a general strategy that is applicable in any randomized experiments with any test statistic. I give several examples of test statistics in Section 3.3. In fact, the definition of a test statistic can be much more general. For instance, with pre-treatment covariate matrix X with the ith row being $X _ { i }$ for unit i $( i = 1 , \ldots , n ) ^ { \mathrm { ~ 4 ~ } }$ , we can allow the test statistic $T ( Z , Y , X )$ to be a function of the treatment vector, outcome vector, and the covariate matrix. Problem 3.6 gives an example.

## 3.6.3 Final remarks

For a general experiment, the probability distribution of $z$ is not uniform over all possible permutations of $n _ { 1 }$ 1’s and $n _ { 0 }$ 0’s. But its distribution is completely known by the experimenter. Therefore, we can always simulate its distribution which in turn implies the distribution of any test statistic under the sharp null hypothesis. A finite-sample exact p-value follows from (3.2). I will discuss other experiments in the subsequent chapters and I want to emphasize that the FRT works beyond the specific experiments discussed in this book.

The FRT works with any test statistic. However, this does answer the practical question of how to choose a test statistic in the data analysis. If the goal is to find surprise with respect to the sharp null hypothesis, it is desirable to choose a test statistic that yields high power under alternative hypotheses. In general, no test statistic can dominate others in terms of power because power depends on the alternative hypothesis. The four test statistics in Section 3.3 are motivated by different alternative hypotheses. For instance, τˆ and t are motivated by an alternative hypothesis with nonzero average treatment effect; W is motivated by an alternative hypothesis with a constant causal effect with outliers. Specifying a working alternative hypothesis is often helpful for constructing a test statistic although it does not have to be precise to guarantee the validity of the FRT. Problems 3.6 and 3.7 illustrate the idea of using a working alternative hypothesis or statistical model to construct test statistics.

## 3.7 Homework Problems

## 3.1 Exactness of $p _ { \mathrm { F R T } }$

Prove (3.2).

## 3.2 Monte Carlo error of $\hat { p } _ { \mathrm { F R T } }$

Given data, $p _ { \mathrm { F R T } }$ is a fixed number while its Monte Carlo estimator $\hat { p } _ { \mathrm { F R T } }$ as in (3.4) is random. Show that

$$
E _ {\mathrm{mc}} (\hat {p} _ {\mathrm{FRT}}) = p _ {\mathrm{FRT}}
$$

and

$$
\operatorname{var} _ {\mathrm{mc}} \left(\hat {p} _ {\mathrm{FRT}}\right) \leq \frac {1}{4 R},
$$

where the subscript “mc” signifies the randomness due to Monte Carlo, that is, $\hat { p } _ { \mathrm { F R T } }$ is random because $z ^ { r } \mathrm { { ^ { s } } }$ are R independent random draws from all possible values of $z$ .

Remark: $p _ { \mathrm { F R T } }$ is random because Z is random. But in this problem, we condition on data, so $p _ { \mathrm { F R T } }$ becomes a fixed number. $\hat { p } _ { \mathrm { F R T } }$ is random because the $z ^ { r }$ s are random permutations of $z .$ .

Problem 3.2 shows that $\hat { p } _ { \mathrm { F R T } }$ is unbiased for $p _ { \mathrm { F R T } }$ over the Monte Carlo randomness and gives an upper bound on the variance of $\hat { p } _ { \mathrm { F R T } }$ . Luo et al. (2021, Theorem 2) gives a more delicate bound on the Monte Carlo error.

## 3.3 A finite-sample valid Monte Carlo approximation $o f p _ { \mathrm { F R T } }$

Although $\hat { p } _ { \mathrm { F R T } }$ is unbiased for $p _ { \mathrm { F R T } }$ , it may not be a valid p-value in the sense that $\mathrm { p r } ( \hat { p } _ { \mathrm { F R T } } \leq u ) \leq u$ for all $u \in ( 0 , 1 )$ due to Monte Carlo error with a finite R. The following modified Monte Carlo approximation is. Phipson and Smyth (2010) pointed out this trick in the permutation test.

Define

$$
\tilde {p} _ {\mathrm{FRT}} = \frac {1 + \sum_ {r = 1} ^ {R} I \{T (\boldsymbol {z} ^ {r} , \boldsymbol {Y}) \geq T (\boldsymbol {Z} , \boldsymbol {Y}) \}}{1 + R}
$$

where the $z ^ { r } \mathrm { { ^ { s } } }$ the R random permutations of Z. Show that with an arbitrary $R ,$ the Monte Carlo approximation $\tilde { p } _ { \mathrm { F R T } }$ is always a finite-sample valid p-value in the sense that $\mathrm { p r } ( \tilde { p } _ { \mathrm { F R T } } \leq u ) \leq u$ for all $u \in ( 0 , 1 )$ .

Hint: You can use the following two basic probability results to prove the claim in Problem 3.3. First, for two Binomial random variables $X _ { 1 } \sim$ Binomial $( R , p _ { 1 } )$ and $X _ { 2 } \sim$ Binomia $. ( R , p _ { 2 } )$ with $p _ { 1 } \geq p _ { 2 }$ , we have $\mathrm { p r } ( X _ { 1 } \leq$ $x ) \ \leq \ \operatorname { p r } ( X _ { 2 } \ \leq \ x )$ for all x. Second, if $\begin{array} { r } { p \ \sim \ \mathrm { U n i f o r m } ( 0 , 1 ) } \end{array}$ and $X \ \parallel$ $p \sim$ Binomia $\left( R , p \right)$ , then, marginally, X is a uniform random variable over $\{ 0 , 1 , \ldots , R \}$ .

## 3.4 Fisher’s exact test

Consider a CRE with a binary outcome, with data summarized in the following $2 \times 2$ table:

<table><tr><td></td><td>$ Y = 1 $</td><td>$ Y = 0 $</td><td>total</td></tr><tr><td>$ Z = 1 $</td><td>$ n_{11} $</td><td>$ n_{10} $</td><td>$ n_{1} $</td></tr><tr><td>$ Z = 0 $</td><td>$ n_{01} $</td><td>$ n_{00} $</td><td>$ n_{0} $</td></tr></table>

Under $H _ { \mathrm { 0 F } } .$ , show that any test statistic is a function of $n _ { 1 1 }$ and other nonrandom fixed constants, and the exact distribution of $n _ { 1 1 }$ is Hypergeometric. Specify the parameters for the Hypergeometric distribution.

Remark: Barnard (1947) and Ding and Dasgupta (2016) pointed out the equivalence of Fisher’s exact test (reviewed in Section A1.3.1) and the FRT under a CRE with a binary outcome.

## 3.5 More details for lady tasting tea

Recall Section 3.5.2. Calculate $\operatorname { p r } ( X = k )$ for k = 0, 1, 2, 3, 4.

## 3.6 Covariate-adjusted FRT

This problem gives more details for Section 3.6.2.

Section 3.4 re-analyzed the LaLonde experimental data using the FRT. The R code FRTLalonde.R implemented the FRT with four test statistics. With additional covariates, the FRT can be more general with at least the following two additional strategies. Under the potential outcomes framework, all potential outcomes and covariates are fixed numbers.

First, we can use test statistics based on residuals from the linear regression. Run a linear regression of the outcomes on the covariates, and obtain the residuals $( { \mathrm { i . e . } }$ , treat the residuals as the pseudo “outcomes”). Then define the four test statistics based on the residuals. Conduct the FRT using these four new test statistics. Report the corresponding p-values.

Second, we can define the test statistic as the coefficient in the linear regression of the outcomes on the treatment and covariates. Conduct the FRT using this test statistic. Report the corresponding p-value.

Why are the five p-values from the above two strategies finite-sample exact? Justify them.

## 3.7 FRT with a generalized linear model

Use the same dataset as Problem 3.6 but change the outcome to a binary indicator whether re78 is positive or not. Run logistic regression of the outcome on the treatment and covariates. Is the coefficient of the treatment significant and what is the p-value? Calculate the p-value from the FRT with the coefficient of the treatment as the test statistic.

## 3.8 An algebraic detail

Verify (3.7)

## 3.9 Recommended reading

Bind and Rubin (2020) is a recently paper advocating the use of p-values as well as the display of the corresponding randomization distributions in analyzing complex experiments.

## 4

# Neymanian Repeated Sampling Inference in Completely Randomized Experiments

In his seminal paper, Neyman (1923) not only proposed to use the notation of potential outcomes but also derived rigorous mathematical results for making inference of the average causal effect under a CRE. In contrast to Fisher’s idea of calculating the p-value under the sharp null hypothesis, Neyman (1923) proposed an unbiased point estimator and a conservative confidence interval based on the sampling distribution of the point estimator. This chapter will introduce Neyman (1923)’s fundamental results, which are very important for understanding later chapters in Part II of this book.

## 4.1 Finite population quantities

Consider a CRE with n units, where $n _ { 1 }$ of them receive the treatment and $n _ { 0 }$ of them receive the control. For unit $i = 1 , \ldots , n .$ , we have potential outcomes $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ , and individual effect $\tau _ { i } = Y _ { i } ( 1 ) - Y _ { i } ( 0 )$ . The potential outcomes have finite population means

$$
\bar {Y} (1) = n ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i} (1), \quad \bar {Y} (0) = n ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i} (0),
$$

variances1

$$
S ^ {2} (1) = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} \left\{Y _ {i} (1) - \bar {Y} (1) \right\} ^ {2}, \quad S ^ {2} (0) = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} \left\{Y _ {i} (0) - \bar {Y} (0) \right\} ^ {2},
$$

and covariance

$$
S (1, 0) = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} \left\{Y _ {i} (1) - \bar {Y} (1) \right\} \left\{Y _ {i} (0) - \bar {Y} (0) \right\}.
$$

The individual effects have mean

$$
\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \tau_ {i} = \bar {Y} (1) - \bar {Y} (0).
$$

and variance

$$
S ^ {2} (\tau) = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} (\tau_ {i} - \tau) ^ {2}.
$$

We have the following relationship between the variances and covariance.

Lemma 4.1 $2 S ( 1 , 0 ) = S ^ { 2 } ( 1 ) + S ^ { 2 } ( 0 ) - S ^ { 2 } ( \tau )$ .

The proof of Lemma 4.1 follows from elementary algebra. I leave it as Problem 4.1.

These fixed quantities are functions of the Science Table $\{ Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \} _ { i = 1 } ^ { n }$ . We are interested in estimating the average causal effect τ based on the data $( Z _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ from a CRE.

## 4.2 Neyman (1923)’s theorem

Based on the observed outcomes, we can calculate the sample means

$$
\hat {\bar {Y}} (1) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i}, \quad \hat {\bar {Y}} (0) = n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) Y _ {i},
$$

the sample variances

$$
\hat {S} ^ {2} (1) = (n _ {1} - 1) ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \{Y _ {i} - \hat {\bar {Y}} (1) \} ^ {2}, \quad \hat {S} ^ {2} (0) = (n _ {0} - 1) ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \{Y _ {i} - \hat {\bar {Y}} (0) \} ^ {2}.
$$

But there are no sample versions of $S ( 1 , 0 )$ and $S ^ { 2 } ( \tau )$ because the potential outcomes $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ are never jointly observed for each unit i. Neyman (1923) proved the following theorem.

Theorem 4.1 Under a CRE,

1. the difference-in-means estimator $\hat { \tau } = \hat { \bar { Y } } ( 1 ) - \hat { \bar { Y } } ( 0 )$ is unbiased $f o r \tau { : }$

$$
E (\hat {\tau}) = \tau ;
$$

2. τˆ has variance

$$
\operatorname{var} (\hat {\tau}) = \frac {S ^ {2} (1)}{n _ {1}} + \frac {S ^ {2} (0)}{n _ {0}} - \frac {S ^ {2} (\tau)}{n} \tag {4.1}
$$

$$
= \frac {n _ {0}}{n _ {1} n} S ^ {2} (1) + \frac {n _ {1}}{n _ {0} n} S ^ {2} (0) + \frac {2}{n} S (1, 0); \tag {4.2}
$$

3. the variance estimator

$$
\hat {V} = \frac {\hat {S} ^ {2} (1)}{n _ {1}} + \frac {\hat {S} ^ {2} (0)}{n _ {0}}
$$

is conservative for estimating var $( \hat { \tau } )$ :

$$
E (\hat {V}) - \mathrm{var} (\hat {\tau}) = \frac {S ^ {2} (\tau)}{n} \geq 0
$$

with the equality holding if and only if $\tau _ { i } = \tau$ for all units.

I will present the proof of Theorem 4.1 in Section 4.3. It is important to clarify the meanings of $E ( \cdot )$ and $\mathrm { v a r } ( \cdot )$ in Theorem 4.1. The potential outcomes are all fixed numbers, and only the treatment indicators $Z _ { i } \mathrm { ^ { * } s }$ are random. Therefore, the expectations and variances are all over the randomness of the $Z _ { i } \mathrm { ^ { * } s } .$ , which are random permutations of $n _ { 1 }$ 1’s and $n _ { 0 }$ 0’s. Figure 4.1 illustrates the randomness of ${ \hat { \tau } } ,$ which is a discrete uniform distribution over $\{ \hat { \tau } ^ { 1 } , \dots , \hat { \tau } ^ { M } \}$ induced by $M = { \binom { n } { n _ { 1 } } }$ possible treatment allocations. Compare Figure 4.1 with Figure 3.1 to see the key differences between the FRT and Neyman (1923)’s theorem:

1. the FRT works for any test statistic but Neyman (1923)’s theorem is only about the difference in means. Although we could derive the properties of other estimators similar to Neyman (1923)’s theorem, this mathematical exercise is often quite challenging for general estimators;  
2. in Figure 3.1 , the observed outcome vector Y is fixed but in Figure 4.1, the observed outcome vector $\mathbf { Y } ( z ^ { m } )$ changes as $z ^ { m }$ changes;  
3. the $T ( z ^ { m } , Y )$ ’s are all computable based on the observed data, but the $\hat { \tau } ^ { m } \mathrm { s }$ are hypothetical values because not all potential outcomes are known.

The point estimator is standard but it has a non-trivial variance under the potential outcomes framework with a CRE. The variance formula (4.1) differs from the classic variance formula for difference in means2 because it not only depends on the finite population variances of the potential outcomes but also depends on the finite population variance of the individual effects, or, equivalently, the finite population covariance of the potential outcomes.

![image_03](images/image_03.png)

```mermaid
graph TD
  A["(Z,Y(1),Y(0)) ⇒ (Z,Y(Z)) ⇒ r̂"] --> B["(z¹,Y(1),Y(0)) ⇒ (z¹,Y(z¹)) ⇒ r̂¹"]
  A --> C["(z²,Y(1),Y(0)) ⇒ (z²,Y(z²)) ⇒ r̂²"]
  A --> D["..."]
  A --> E["(zM,Y(1),Y(0)) ⇒ (zM,Y(zM)) ⇒ r̂M"]
  B --> F["E(r̂) = M⁻¹ ∑_{m=1}^M r̂^m"]
  C --> F
  E --> G["var(r̂) = M⁻¹ ∑_{m=1}^M (r̂^m - r̂)²"]
```

FIGURE 4.1: Illustration of Neyman (1923)’s theorem

Unfortunately, $S ^ { 2 } ( \tau )$ and $S ( 1 , 0 )$ are not identifiable from the data because $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ are never jointly observed.

Due to the fundamental problem of missing one potential outcome, we can at most obtain a conservative variance estimator. In statistics, the definition of the confidence interval allows for over coverage and thus conservativeness in variance estimation. This may be not a good idea in some applications, for example, studies on side effects of drugs.

The formula (4.1) is a little puzzling in that the more heterogeneous the individual effects are the smaller the variability of $\hat { \tau }$ is. Section 4.5.1 will use numerical examples to verify (4.1). What is the intuition here? I give an explanation based on the equivalent form (4.2). Compare the case with positively correlated potential outcomes and the case with negatively correlated potential outcomes. Although the treatment group is a simple random sample from the finite population of n units, it is possible to observe relatively large treatment potential outcomes in a realized experiment. If this happens, then those control units have relatively small treatment potential outcomes. Consequently, if $S ( 1 , 0 ) > 0$ , then the control potential outcomes are relatively small; if $S ( 1 , 0 ) < 0 $ , then the control potential outcomes are relatively large. Therefore, ˆτ tends to larger when the potential outcomes are positively correlated, resulting in more extreme values of $\hat { \tau } .$ So the variance of $\hat { \tau }$ is larger when the potential outcomes are positively correlated.

Li and Ding (2017, Theorem 5 and Proposition 3) further proved the following asymptotic Normality of ˆτ based on the finite population central limit theorem.

Theorem 4.2 Let $n  \infty$ and $n _ { 1 }  \infty . \mathrm { ~ } I f \ n _ { 1 } / n$ has a limiting value in $( 0 , 1 ) , \{ S ^ { 2 } ( 1 ) , S ^ { 2 } ( 0 ) , S ( 1 , 0 ) \}$ have limiting values, and

$$
\max _ {1 \leq i \leq n} \left\{Y _ {i} (1) - \bar {Y} (1) \right\} ^ {2} / n \to 0, \quad \max _ {1 \leq i \leq n} \left\{Y _ {i} (0) - \bar {Y} (0) \right\} ^ {2} / n \to 0,
$$

## 4.3 Proofs

then

$$
\frac {\hat {\tau} - \tau}{\sqrt {\operatorname{var} (\hat {\tau})}} \to \mathrm{N} (0, 1)
$$

in distribution, and

$$
\hat {S} ^ {2} (1) \to S ^ {2} (1), \quad \hat {S} ^ {2} (0) \to S ^ {2} (0)
$$

in probability.

The proof of Theorem 4.2 is technical and beyond the scope of this book. It ensures that the sampling distribution of ˆτ can be approximated by Normal distribution with large sample size and some regularity conditions. Moreover, it ensures that the sample variances of the outcomes are consistent for the population variances, which further ensures that the probability limit of Neyman (1923)’s variance estimator is larger than the true variance of ˆτ . This justifies a conservative large-sample confidence interval for τ :

$$
\hat {\tau} \pm z _ {1 - \alpha / 2} \sqrt {\hat {V}},
$$

which is the same as the confidence interval for the standard two-sample problem asymptotically. This confidence interval covers τ with probability at least at large as 1 − α when the sample size is large enough. By duality, the confidence interval implies a test for $H _ { \mathrm { 0 N } } : \tau = 0$ .

The conservativeness of Neyman (1923)’s confidence interval for τ is not a big problem if under reporting the treatment effect is not a big problem. It can be problematic if the outcomes measure the side effects of a treatment. In medical experiments, under reporting the side effects of a new drug can have severe consequences.

## 4.3 Proofs

In this section, I will prove Theorem 4.1.

First, the unbiasedness of ˆτ follows from the representation

$$
\begin{array}{l} \hat {\tau} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i} - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) Y _ {i} \\ = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i} (1) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) Y _ {i} (0) \\ \end{array}
$$

and the linearity of the expectation:

$$
\begin{array}{l} E (\hat {\tau}) = E \left\{n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i} (1) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} \left(1 - Z _ {i}\right) Y _ {i} (0) \right\} \\ = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} E (Z _ {i}) Y _ {i} (1) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} E (1 - Z _ {i}) Y _ {i} (0) \\ = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} \frac {n _ {1}}{n} Y _ {i} (1) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} \frac {n _ {0}}{n} Y _ {i} (0) \\ = n ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i} (1) - n ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i} (0) \\ = \tau . \\ \end{array}
$$

Second, we can further write $\hat { \tau }$ as

$$
\hat {\tau} = \sum_ {i = 1} ^ {n} Z _ {i} \left\{\frac {Y _ {i} (1)}{n _ {1}} + \frac {Y _ {i} (0)}{n _ {0}} \right\} - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i} (0).
$$

The variance of ˆτ follows from Lemma A3.2 of simple random sampling:

$$
\begin{array}{l} \operatorname{var} (\hat {\tau}) = \frac {n _ {1} n _ {0}}{n (n - 1)} \sum_ {i = 1} ^ {n} \left\{\frac {Y _ {i} (1)}{n _ {1}} + \frac {Y _ {i} (0)}{n _ {0}} - \frac {\bar {Y} (1)}{n _ {1}} - \frac {\bar {Y} (0)}{n _ {0}} \right\} ^ {2} \\ = \frac {n _ {1} n _ {0}}{n (n - 1)} \left[ \frac {1}{n _ {1} ^ {2}} \sum_ {i = 1} ^ {n} \left\{Y _ {i} (1) - \bar {Y} (1) \right\} ^ {2} + \frac {1}{n _ {0} ^ {2}} \sum_ {i = 1} ^ {n} \left\{Y _ {i} (0) - \bar {Y} (0) \right\} ^ {2} \right. \\ \left. + \frac {2}{n _ {1} n _ {0}} \sum_ {i = 1} ^ {n} \left\{Y _ {i} (1) - \bar {Y} (1) \right\} \left\{Y _ {i} (0) - \bar {Y} (0) \right\} \right] \\ = \frac {n _ {0}}{n _ {1} n} S ^ {2} (1) + \frac {n _ {1}}{n _ {0} n} S ^ {2} (0) + \frac {2}{n} S (1, 0). \\ \end{array}
$$

From Lemma 4.1, we can also write the variance as

$$
\begin{array}{l} \operatorname{var} (\hat {\tau}) = \frac {n _ {0}}{n _ {1} n} S ^ {2} (1) + \frac {n _ {1}}{n _ {0} n} S ^ {2} (0) + \frac {1}{n} \left\{S ^ {2} (1) + S ^ {2} (0) - S ^ {2} (\tau) \right\} \\ = \frac {S ^ {2} (1)}{n _ {1}} + \frac {S ^ {2} (0)}{n _ {0}} - \frac {S ^ {2} (\tau)}{n}. \\ \end{array}
$$

Third, because the treatment group is a simple random sample of size $n _ { 1 }$ from the n units, Lemma A3.3 ensures that the sample variance of $Y _ { i } ( 1 ) \mathrm { { ^ { * } s } }$ is unbiased for its population variance:

$$
E \{\hat {S} ^ {2} (1) \} = S ^ {2} (1).
$$

Similarly, $E \{ \hat { S } ^ { 2 } ( 0 ) \} = S ^ { 2 } ( 0 )$ . Therefore, $\hat { V }$ is unbiased for the first two terms in (4.1).

## 4.4 Regression analysis of the CRE

Practitioners often use regression-based inference for the average causal effect τ . A standard approach is to run the ordinary least squares (OLS) of the outcomes on the treatment indicators with an intercept

$$
(\hat {\alpha}, \hat {\beta}) = \arg \min _ {(a, b)} \sum_ {i = 1} ^ {n} (Y _ {i} - a - b Z _ {i}) ^ {2},
$$

and use the coefficient of the treatment $\hat { \beta }$ as the estimator for the average causal effect. We can show the coefficient $\hat { \beta }$ equals the difference in means:

$$
\hat {\beta} = \hat {\tau}. \tag {4.3}
$$

However, the usual variance estimator from the OLS, e.g., the output from the lm function of R, equals

$$
\hat {V} _ {\mathrm{OLS}} = \frac {N (N _ {1} - 1)}{(N - 2) N _ {1} N _ {0}} \hat {S} ^ {2} (1) + \frac {N (N _ {0} - 1)}{(N - 2) N _ {1} N _ {0}} \hat {S} ^ {2} (0) \tag {4.4}
$$

$$
\approx \frac {\hat {S} ^ {2} (1)}{N _ {0}} + \frac {\hat {S} ^ {2} (0)}{N _ {1}},
$$

where the approximation holds with large $N _ { 1 }$ and $N _ { 0 }$ . It differs from $\hat { V }$ even with large $N _ { 1 }$ and $N _ { 0 }$ .

Fortunately, the Eicker–Huber–White (EHW) robust variance estimator is close to $\hat { V } ;$

$$
\hat {V} _ {\mathrm{EHW}} = \frac {\hat {S} ^ {2} (1)}{N _ {1}} \frac {N _ {1} - 1}{N _ {1}} + \frac {\hat {S} ^ {2} (0)}{N _ {0}} \frac {N _ {0} - 1}{N _ {0}} \tag {4.5}
$$

$$
\approx \frac {\hat {S} ^ {2} (1)}{N _ {1}} + \frac {\hat {S} ^ {2} (0)}{N _ {0}}
$$

where the approximation holds with large $N _ { 1 }$ and $N _ { 0 }$ . It is almost identical to $\hat { V }$ . Moreover, the so-called HC2 variant of the EHW robust variance estimator is identical to $\hat { V }$ . The hccm function in the car package returns the EHW robust variance estimator as well as its HC2 variant.

Problem 4.3 provides more technical details for (4.3)–(4.5).

## 4.5 Examples

## 4.5.1 Simulation

I first choose the sample size as $n = 1 0 0$ with 60 treated and 40 control units, and generate the potential outcomes with constant individual causal effects.

```txt
n = 100
n1 = 60
n0 = 40
y0 = rexp(n)
y0 = sort(y0, decreasing = TRUE)
y1 = y0 + 1
```

With the Science Table fixed, I repeated generate completely randomized experiments and apply Theorem 4.1 to obtain the point estimator, the conservative variance estimator, and the confidence interval based on the Normal approximation. The first panel of Figure 4.2 shows the histogram of ˆτ −τ over 104 simulations.

I then change the potential outcome by sorting the control potential outcome in reverse order

```txt
y0 = sort(y0, decreasing = FALSE)
```

and repeat the above simulation. The second panel of Figure 4.2 shows the histogram of ˆτ − τ over 104 simulations.

I finally permute the control potential outcomes

```txt
y0 = sample(y0)
```

and repeat the above simulation. The third panel of Figure 4.2 shows the histogram of ˆτ − τ over 104 simulations.

Importantly, in the above three sets of simulations, the correlations between potential outcomes are different but the marginal distributions are the same. The following table compares the true variances, the conservative estimated variances, and the coverage rates of the 95% confidence intervals.

<table><tr><td></td><td>constant</td><td>negative</td><td>independent</td></tr><tr><td>var</td><td>0.036</td><td>0.007</td><td>0.020</td></tr><tr><td>estimated var</td><td>0.036</td><td>0.036</td><td>0.036</td></tr><tr><td>coverge rate</td><td>0.947</td><td>1.000</td><td>0.989</td></tr></table>

The true variance depends on the correlation between the potential outcomes, with positively correlated potential outcomes corresponding to a larger sampling variance. This verifies (4.2). The estimated variances are almost identical because the formula of Vˆ depends only on the marginal distributions of the potential outcomes. Due to the discrepancy between the true and estimated variances, the coverage rates differ across the three sets of simulations. Only with constant causal effects, the estimated variance is identical to the true variance, verifying point 3 of Theorem 4.1.

Figure 4.2 also shows the Normal density curves based on the central limit theorem for ˆτ . They are very close to the histogram over simulations, verifying Theorem 4.2.

## 4.5.2 Heavy-tailed outcome and failure of Normal approximations

The central limit theorem of ˆτ in Theorem 4.2 holds under some regularity conditions. Those conditions will be violated with heavy-tailed potential outcomes. We can modify the above simulation studies to illustrate this point. Assume the individual causal effects are constant but the control potential outcomes are contaminated by a Cauchy component with probability 0.1, 0.3 or 0.5. The following code generates the potential outcomes with the probability of contamination being 0.1.

```python
combination = rbinom(n, 1, 0.1)
y0 = (1 - combination)*rexp(n) + combination*rcauchy(n)
y1 = y0 + 1
```

Figures 4.3 and 4.4 show two realizations of the histograms of ˆτ −τ with the corresponding Normal approximations. With heavy-tailed potential outcomes, the Normal approximations are quite poor. Moreover, unlike Figure 4.2, the histograms are quite sensitive to the random seed of the simulation.

## 4.5.3 Application

I again use the lalonde data to illustrate the theory.

```txt
> library (Matching)
> data (lalonde)
> z = lalonde$treat
> y = lalonde$re78
```

We can easily calculate the point estimator and standard error based on the formulas in Theorem 4.1:

```txt
> n1 = sum(z)
> n0 = length(z) - n1
> tauhat = mean(y[z==1]) - mean(y[z==0])
> vhat = var(y[z==1])/n1 + var(y[z==0])/n0
> sehat = sqrt(vhat)
> tauhat
[1] 1794.343
> sehat
[1] 670.9967
```

Practitioners often use ordinary least squares (OLS) to estimate the average causal effect which also gives a standard error.

```txt
> olsfit = lm(y ~ z)
> summary(olsfit)$coef[2, 1: 2]
Estimate Std. Error
1794.3431 632.8536
```

However, the above standard error seems too small compared to the one based on Theorem 4.1. However, this can be easily solved by using the Eicker–Huber– White robust standard error.

![image_04](images/image_04.png)

FIGURE 4.3: Sampling distribution of $\hat { \tau } - \tau$ with contaminated potential outcomes: realization one

![image_05](images/image_05.png)

FIGURE 4.4: Sampling distribution of $\hat { \tau } - \tau$ with contaminated potential outcomes: realization two

```txt
> library(car)
> sqrt(hccm(olsfit)[2, 2])
[1] 672.6823
> sqrt(hccm(olsfit, type = "hc0")[2, 2])
[1] 669.3155
> sqrt(hccm(olsfit, type = "hc2")[2, 2])
[1] 670.9967
```

Different versions of the robust standard error exist. They yield similar results if the sample size is large, with hc2 yielding a standard error identical to Theorem 4.1. Problem 4.3 gives a theoretical explanation for the possible failure of the standard error based on OLS and the asymptotic validity of the Eicker–Huber–White robust standard error.

## 4.6 Homework Problems

## 4.1 Proof of Lemma 4.1

Prove Lemma 4.1.

## 4.2 Alternative proof of Theorem 4.1

Under a CRE, calculate

$$
\operatorname{var} \{\hat {\bar {Y}} (1) \}, \quad \operatorname{var} \{\hat {\bar {Y}} (0) \}, \quad \operatorname{cov} \{\hat {\bar {Y}} (1), \hat {\bar {Y}} (0) \}
$$

and use these formulas to calculate var(ˆτ ).

Hint: Use the results in Chapter A3.

## 4.3 Neymanian inference and OLS

Prove (4.3)–(4.5). Moreover, prove that the HC2 variant of the EHW robust variance estimator recovers Vˆ exactly.

Hint: Appendix A2 reviews some important technical results about OLS.

## 4.4 Treatment effect heterogeneity

Show that $S ^ { 2 } ( \tau ) = 0$ implies that $S ^ { 2 } ( 1 ) = S ^ { 2 } ( 0 )$ . Given a counterexample with $S ^ { 2 } ( 1 ) = \dot { S } ^ { 2 } ( 0 )$ but $S ^ { 2 } ( \tau ) \neq 0 .$ .

Show that $S ^ { 2 } ( 1 ) < S ^ { 2 } ( 0 )$ implies that

$$
S (Y (0), \tau) = (n - 1) \sum_ {i = 1} ^ {n} \left\{Y _ {i} (0) - \bar {Y} (0) \right\} \left(\tau_ {i} - \tau\right) <   0.
$$

Give a counterexample with $S ^ { 2 } ( 1 ) > S ^ { 2 } ( 0 )$ but $S ( Y ( 0 ) , \tau ) < 0 .$ .

Remark: The first result states that no treatment effect heterogeneity implies equal variances in the treated and control potential outcomes. But the converse is not true. The second result states that if the treated potential outcome has larger variance than the control potential outcome, then the individual treatment effect is negatively correlated with the control potential outcome. But the converse is not true. Gerber and Green (2012, page 293) and (Ding et al., 2019, Appendix B.3) gave related discussions.

## 4.5 A better bound of the variance formula

Neyman (1923)’s conservative variance estimator essentially uses the following upper bound on the true variance:

$$
\operatorname{var} (\widehat {\tau}) = \frac {S ^ {2} (1)}{n _ {1}} + \frac {S ^ {2} (0)}{n _ {0}} - \frac {S ^ {2} (\tau)}{n} \leq \frac {S ^ {2} (1)}{n _ {1}} + \frac {S ^ {2} (0)}{n _ {0}},
$$

which uses the trivial fact that $S ^ { 2 } ( \tau ) \geq 0$ . Show the following upper bound

$$
\operatorname{var} (\widehat {\tau}) \leq \frac {1}{n} \left\{\sqrt {\frac {n _ {0}}{n _ {1}}} S (1) + \sqrt {\frac {n _ {1}}{n _ {0}}} S (0) \right\} ^ {2}. \tag {4.6}
$$

When does the equality in (4.6) hold?

The upper bound (4.6) motivates another conservative variance estimator

$$
\hat {V} ^ {\prime} = \frac {1}{n} \left\{\sqrt {\frac {n _ {0}}{n _ {1}}} \hat {S} (1) + \sqrt {\frac {n _ {1}}{n _ {0}}} \hat {S} (0) \right\} ^ {2}.
$$

Section 4.5.1 used $\hat { V }$ in the simulation with R code NeymanCR.R. Repeat the simulation with additional comparison with the variance estimator $\hat { V } ^ { \prime }$ and the associated confidence interval.

Remark: The upper bound (4.6) can be further improved. Aronow et al. (2014) derived the sharp upper bound for $\mathrm { v a r } ( \widehat { \tau } )$ using the Frechet–Hoeffding inequality. Those improvements are rarely used in practice mainly for two reasons. First, they are more complicated than $\hat { V }$ which can be conveniently implemented by OLS. Second, the confidence interval based on $\hat { V }$ also works under other formulations, for example, under a true linear model of the outcome on the treatment, but those improvements do not. Although they are theoretically interesting, those improvements have little practical impact.

## 4.6 Vector version of Neyman (1923)

The classic result of Neyman (1923) is about a scalar outcome. It is common to have multiple outcomes in practice. Therefore, we can extend the potential outcomes to vectors. We consider the average causal effect on a vector outcome $V \in \mathbb { R } ^ { K }$ ,

$$
\tau_ {\boldsymbol {V}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left\{\boldsymbol {V} _ {i} (1) - \boldsymbol {V} _ {i} (0) \right\},
$$

where $V _ { i } ( 1 )$ and $V _ { i } ( 0 )$ are the potential outcomes of $V$ for unit i. The Neymantype estimator for $\tau _ { V }$ is the difference between the sample mean vectors of the observed outcomes under treatment and control:

$$
\widehat {\boldsymbol {\tau}} _ {\mathbf {V}} = \bar {\mathbf {V}} _ {1} - \bar {\mathbf {V}} _ {0} = \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} \mathbf {V} _ {i} - \frac {1}{n _ {0}} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \mathbf {V} _ {i}.
$$

Consider a CRE. Show that $\widehat { \tau } _ { V }$ is unbiased for $\tau _ { V }$ . Find the covariance matrix of $\widehat { \tau } _ { V }$ . Find a (possibly conservative) estimator for the variance.

## 4.7 Inference in the BRE

Consider the BRE where the $Z _ { i } \mathrm { ^ { * } s }$ are IID Bernoulli(π) with $n _ { 1 } = \textstyle \sum _ { i = 1 } ^ { n }$ Z i receiving the treatment and $\begin{array} { r } { n _ { 0 } = \sum _ { i = 1 } ^ { n } ( 1 - Z _ { i } ) } \end{array}$ receiving the control.

First, we can use the FRT to analyze the BRE. How do we test $H _ { \mathrm { 0 F } }$ in the CRE? Can we use the same FRT procedure as in the CRE if the actual experiment is the BRE? If yes, give a justification; if no, explain why.

Second, we can obtain point estimator for τ and find the associated variance estimator, as Neyman (1923) did for the CRE.

1. Is ˆτ unbiased for τ ? Is it consistent?  
2. Find an unbiased estimator for τ .  
3. Compare the variance of the above unbiased estimator and the asymptotic variance of ˆτ.

Remark: The estimator ˆτ does not have finite variance but the variance of its asymptotic distribution is finite.

## 4.8 Recommended reading

Ding (2016) compared the Fisherian and Neymanian approaches to analyzing the CRE.

## 5

# Stratification and Post-Stratification in Randomized Experiments

Block what you can and randomize what you cannot.

— George Box

This is the second most famous quote from George Box1. This chapter will explain its meaning.

## 5.1 Stratification

A CRE may generate an undesired treatment allocation. Let us start with a completely randomized experiment with a discrete covariate $X _ { i } \in \{ 1 , \ldots , K \}$ , and define $n _ { [ k ] } = \# \{ i : X _ { i } = k \}$ and $\pi _ { [ k ] } = n _ { [ k ] } / n$ as the number and proportion of units in stratum $ { k } ( k = 1 , \ldots { \dot { , } } K )$ . A CRE assigns $n _ { 1 }$ units to the treatment group and $n _ { 0 }$ units to the control group, which results in

$$
n _ {[ k ] 1} = \# \{i: X _ {i} = k, Z _ {i} = 1 \}, \quad n _ {[ k ] 0} = \# \{i: X _ {i} = k, Z _ {i} = 0 \}
$$

units in the treatment and control groups within stratum k. With positive probability, $n _ { [ k ] 1 } \mathrm { ~ o r ~ } n _ { [ k ] 0 }$ is zero for some $k ,$ that is, it is possible that some strata only have treated or control units. Even none of the $n _ { [ k ] 1 } \mathrm { ' s }$ or $n _ { [ k ] 0 } \mathrm { ^ { * } s }$ are zero, with high probability

$$
\frac {n _ {[ k ] 1}}{n _ {1}} - \frac {n _ {[ k ] 0}}{n _ {0}} \neq 0, \tag {5.1}
$$

and the magnitude can be quite large. So the proportions of units in stratum k are different across the treatment and control groups although on average their difference is zero:

$$
\begin{array}{l} E \left(\frac {n _ {[ k ] 1}}{n _ {1}} - \frac {n _ {[ k ] 0}}{n _ {0}}\right) \\ = E \left\{n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} 1 (X _ {i} = k) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) 1 (X _ {i} = k) \right\} \\ = 0. \\ \end{array}
$$

When $n _ { [ k ] 1 } / n _ { 1 } - n _ { [ k ] 0 } / n _ { 0 }$ is large for some strata with $X = k$ , the treatment and control groups have undesirable covariate imbalance. Such covariate imbalance deteriorates the quality of the experiment, making it difficult to interpret the results of the experiment since the difference in the outcomes may be attributed to the treatment or the covariate imbalance.

How can we actively avoid covariate imbalance in the experiment? We can fix the $n _ { [ k ] 1 } \mathrm { { } ^ { \circ } s }$ or $n _ { [ k ] 0 } \mathrm { ^ { * } s }$ in advance and conduct stratified randomized experiments (SRE).

Definition 5.1 (SRE) We conduct K independent CREs within the K strata of a discrete covariate X.

In agricultural experiments, the SRE is also called the randomized block design, with the strata called the blocks. Analogously, stratified randomization is also called block randomization. The total number of randomizations in an SRE equals

$$
\prod_ {k = 1} ^ {K} \binom{n _ {[ k ]}}{n _ {[ k ] 1}},
$$

and each feasible randomization has equal probability. Within stratum $k ,$ the proportion of units receiving the treatment is

$$
e _ {[ k ]} = \frac {n _ {[ k ] 1}}{n _ {[ k ]}},
$$

which is also called the propensity score, a conceptual that will play a central role in Part III of this book. An SRE is different from a CRE: first, all feasible randomizations in an SRE form a subset of all feasible randomizations in a $\mathrm { C R E } ,$ so

$$
\prod_ {k = 1} ^ {K} \binom{n _ {[ k ]}}{n _ {[ k ] 1}} <   \binom{n}{n _ {1}};
$$

second, $e _ { [ k ] }$ is fixed in an SRE but random in a CRE.

For every unit i, we have potential outcomes $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ , and individual causal effect $\tau _ { i } = Y _ { i } ( 1 ) – Y _ { i } ( 0 )$ . For stratum k, we have stratum-specific average causal effect

$$
\tau_ {[ k ]} = n _ {[ k ]} ^ {- 1} \sum_ {X _ {i} = k} \tau_ {i}.
$$

The average causal effect is

$$
\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \tau_ {i} = n ^ {- 1} \sum_ {k = 1} ^ {K} \sum_ {X _ {i} = k} \tau_ {i} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \tau_ {[ k ]},
$$

which is also the weighted average of the stratum-specific average causal effects.

If we are interested in $\tau _ { [ k ] }$ , then we can use the methods in Chapters 3 and 4 for the CRE within stratum k. Below I will discuss statistical inference for τ .

## 5.2 FRT

## 5.2.1 Theory

In parallel with the discussion of a CRE, I will start with the FRT in an SRE. The sharp null hypothesis is still

$$
H _ {0 \mathrm{F}}: Y _ {i} (1) = Y _ {i} (0) \text {   for   all   units   } i = 1, \dots , n.
$$

The fundamental idea of the FRT applies to any randomized experiment: we can use any test statistic which has a known distribution under $H _ { \mathrm { 0 F } }$ and the SRE. However, we must be careful with two subtle issues. First, when we simulate the treatment vector, we must permute the treatment indicators within strata of X. The resulting FRT is sometimes called the conditional randomization test or conditional permutation test. Second, we should choose test statistics that can reflect the nature of the SRE. Below I give some canonical choices of the test statistic.

Example 5.1 (Stratified estimator) Motivated by estimating τ , we can use the following stratified estimator in the FRT:

$$
\hat {\tau} _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \hat {\tau} _ {[ k ]},
$$

where

$$
\hat {\tau} _ {[ k ]} = n _ {[ k ] 1} ^ {- 1} \sum_ {i = 1} ^ {n} I (X _ {i} = k, Z _ {i} = 1) Y _ {i} - n _ {[ k ] 0} ^ {- 1} \sum_ {i = 1} ^ {n} I (X _ {i} = k, Z _ {i} = 0) Y _ {i}
$$

is the stratum-specific difference-in-means within stratum k.

Example 5.2 (Studentized stratified estimator) Motivated by the studentized statistic in the simple two-sample problem, we can use the following studentized statistic for the stratified estimator in the FRT:

$$
t _ {\mathrm{S}} = \frac {\hat {\tau} _ {\mathrm{S}}}{\sqrt {\hat {V} _ {\mathrm{S}}}},
$$

with

$$
\hat {V} _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \left(\frac {\hat {S} _ {[ k ]} ^ {2} (1)}{n _ {[ k ] 1}} + \frac {\hat {S} _ {[ k ]} ^ {2} (0)}{n _ {[ k ] 0}}\right)
$$

where $\hat { S } _ { [ k ] } ^ { 2 } ( 1 )$ and $\hat { S } _ { [ k ] } ^ { 2 } ( 0 )$ are the stratum-specific sample variances of the outcomes under treatment and control, respectively. The exact form of this statistic is motivated by the Neymanian perspective discussed in Section 5.3.

Example 5.3 (Combining Wilcoxon rank-sum statistics) We first compute the Wilcoxon rank sum statistic $W _ { [ k ] }$ within stratum k and then combine them as

$$
W _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} c _ {[ k ]} W _ {[ k ]}.
$$

Based on different asymptotic schemes and optimality criteria, Van Elteren (1960) proposed two weighting methods, one with

$$
c _ {[ k ]} = \frac {1}{n _ {[ k ] 1} n _ {[ k ] 0}},
$$

and the other with

$$
c _ {[ k ]} = \frac {1}{n _ {[ k ]} + 1}
$$

The motivations for these weights appear to be quite technical, and other choices of weights may also be reasonable.

Example 5.4 (Hodges and Lehmann (1962)’s aligned rank statistic) Van Elteren (1960)’s statistic works well with a few large strata. However, it does not work well with many small strata since it does not make enough comparisons, potentially losing information in the data. Hodges and Lehmann (1962) proposed a test statistic that makes more comparisons across strata after standardizing the outcomes. They suggested first centering the outcomes as

$$
\tilde {Y} _ {i} = Y _ {i} - \bar {Y} _ {[ k ]}
$$

with the stratum-specific mean ${ \bar { Y } } _ { [ k ] } = n _ { [ k ] } ^ { - 1 } \sum _ { X _ { i } = k } Y _ { i }$ if $X _ { i } = k$ , then obtaining the ranks $( \tilde { R } _ { 1 } , \ldots , \tilde { R } _ { n } )$ of the pooled outcomes $( { \tilde { Y } } _ { 1 } , \dots , { \tilde { Y } } _ { n } )$ , and finally constructing the test statistic

$$
\tilde {W} = \sum_ {i = 1} ^ {n} Z _ {i} \tilde {R} _ {i}.
$$

We can simulate the exact distributions of the above test statistics under the SRE. We can also calculate their means and variances and obtain the p-values based on Normal approximations.

After searching for a while, I failed to find detailed discussion of the Kolmogorov–Smirnov statistic for the SRE. Below is my proposal.

Example 5.5 (Kolmogorov–Smirnov statistic) We compute $D _ { [ k ] }$ , the maximum difference between the empirical distributions of the outcomes under treatment and control within stratum k. The final test statistic can be

$$
D _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} c _ {[ k ]} D _ {[ k ]}
$$

or

$$
D _ {\max} = \max _ {1 \leq k \leq K} c _ {[ k ]} D _ {[ k ]},
$$

where $c _ { [ k ] } = \sqrt { n _ { [ k ] 1 } n _ { [ k ] 0 } / n _ { [ k ] } }$ is motivated by the limiting distribution of $D _ { [ k ] }$ with $n _ { [ k ] 1 }$ and $n _ { [ k ] 0 }$ approach infinity (Van der Vaart, 2000). The statistics $D _ { \mathrm { { S } } }$ and $D _ { \mathrm { m a x } }$ are more appropriate when all strata have large sample size. Another reasonable choice is

$$
D = \max _ {y} \Big | \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \{\hat {F} _ {[ k ] 1} (y) - \hat {F} _ {[ k ] 0} (y) \} \Big |,
$$

where $\hat { F } _ { [ k ] 1 } ( y )$ and $\hat { F } _ { [ k ] 0 } ( y )$ are the stratum-specific empirical distribution functions of the outcomes under treatment and control, respectively. The statistic D is appropriate in both the cases with large strata and the cases with many small strata.

## 5.2.2 An application

The Penn Bonus experiment as an example to illustrate the FRT in the SRE. The dataset used by Koenker and Xiao (2002) is from a job training program stratified on quarter, with the outcome being the duration before employed.

```txt
penndata = read.table("Penn46_ascii.txt")
z = penndata$treatment
y = log(penndata$duration)
block = penndata$quarter
```

I will focus on $\mathrm { \hat { \tau } _ { S } }$ and $W _ { \mathrm { S } } .$ , and leave the FRT with other statistics as exercise. The following function computes $\mathrm { \hat { \tau } _ { S } }$ and $V { : }$

```r
stat_SRE = function(z, y, x)
{
    xlevels = unique(x)
    K = length(xlevels)
    PiK = rep(0, K)
    TauK = rep(0, K)
    WilcoxK = rep(0, K)
    for(k in 1:K)
    {
    xk = xlevels[k]
    zk = z[x == xk]
    yk = y[x == xk]
    PiK[k] = length(zk)/length(z)
    TauK[k] = mean(yk[zk==1]) - mean(yk[zk==0])
    WilcoxK[k] = wilcox.test(yk[zk==1], yk[zk==0])$statistic
    }
    return(c(sum(PiK*TauK), sum(WilcoxK/PiK)))
}
```

The following function generates a random treatment assignment in the SRE of the observed data:

```txt
zRandomSRE = function(z, x)
{
    xlevels = unique(x)
    K = length(xlevels)
    zrandom = z
    for(k in 1:K)
    {
    xk = xlevels[k]
    zrandom[x == xk] = sample(z[x == xk])
    }
    return(zrandom)
}
```

Based on the above data and functions, we can easily simulate the randomization distributions of the test statistics (shown in Figure 5.1 with 104 Monte Carlo draws) and compute the p-values.

```diff
> MC = 10^4
> statSREMC = matrix(0, MC, 2)
> for(mc in 1:MC)
+ {
+    zrandom = zRandomSRE(z, block)
+    statSREMC[mc, ] = stat_SRE(zrandom, y, block)
+ }
> mean(statSREMC[, 1] <= stat.obs[1])
[1] 0.0019
> mean(statSREMC[, 2] <= stat.obs[2])
[1] 5e-04
```

## 5.3 Neymanian inference

## 5.3.1 Point and interval estimation

Statistical inference for an SRE builds on the fact that it essentially consists of K independent CREs. Based on this, we can easily extend Neyman (1923)’s results to the SRE. Within stratum k, the difference-in-means ${ \hat { \tau } } _ { [ k ] }$ is unbiased for τ[k] with variance

$$
\mathrm{var} (\hat {\tau} _ {[ k ]}) = \frac {S _ {[ k ]} ^ {2} (1)}{n _ {[ k ] 1}} + \frac {S _ {[ k ]} ^ {2} (0)}{n _ {[ k ] 0}} - \frac {S _ {[ k ]} ^ {2} (\tau)}{n _ {[ k ]}},
$$

where $S _ { [ k ] } ^ { 2 } ( 1 ) , S _ { [ k ] } ^ { 2 } ( 0 )$ and $S _ { [ k ] } ^ { 2 } ( \tau )$ are the stratum-specific variances of potential outcomes and the individual treatment effects, respectively. Therefore, the

$\begin{array} { r } { \hat { \tau } _ { \mathrm { S } } = \sum _ { k = 1 } ^ { K } \pi _ { [ k ] } \hat { \tau } _ { [ k ] } } \end{array}$ $\begin{array} { r } { \tau = \sum _ { k = 1 } ^ { K } \pi _ { [ k ] } \tau _ { [ k ] } } \end{array}$ variance

$$
\mathrm{var} (\hat {\tau} _ {\mathrm{S}}) = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \mathrm{var} (\hat {\tau} _ {[ k ]}).
$$

If $n _ { [ k ] 1 } \geq 2$ and $n _ { [ k ] 0 } \geq 2$ , then we can obtain the sample variances $\hat { S } _ { [ k ] } ^ { 2 } ( 1 )$ and $\hat { S } _ { [ k ] } ^ { 2 } ( 0 )$ of the outcomes within stratum k and construct a conservative variance estimator

$$
\hat {V} _ {\mathrm{S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \left(\frac {\hat {S} _ {[ k ]} ^ {2} (1)}{n _ {[ k ] 1}} + \frac {\hat {S} _ {[ k ]} ^ {2} (0)}{n _ {[ k ] 0}}\right),
$$

where $\hat { S } _ { [ k ] } ^ { 2 } ( 1 )$ and $\hat { S } _ { [ k ] } ^ { 2 } ( 0 )$ are the stratum-specific sample variances of the outcomes under treatment and control, respectively. Based on a Normal approximation of $\mathrm { \hat { \tau } _ { S } }$ , we can construct a Wald-type $1 - \alpha$ confidence interval for τ :

$$
\hat {\tau} _ {\mathrm{S}} \pm z _ {1 - \alpha / 2} \sqrt {\hat {V} _ {\mathrm{S}}}.
$$

From a hypothesis testing perspective, under $H _ { \mathrm { 0 N } } : \tau = 0 .$ , we can compare $t _ { \mathrm { S } } = \hat { \tau } _ { \mathrm { S } } / \sqrt { \hat { V } _ { \mathrm { S } } }$ with the standard Normal quantiles to obtain asymptotic $p \mathrm { - }$ values. The statistic $t _ { \mathrm { S } }$ has appeared in Example 5.2 for the FRT. Similar to the discussion for the CRE, using $t _ { \mathrm { S } }$ in the FRT yields finite-sample exact p-value under $H _ { \mathrm { 0 F } }$ and asymptotically valid $p \mathrm { - }$ -value under $H _ { \mathrm { 0 N } }$ . Wu and Ding (2021) provided $\mathrm { a }$ justification for this claim.

Here I omit the technical details for the central limit theorem of $\mathrm { \hat { \tau } _ { S } }$ . See Liu and Yang (2020) for a proof, which includes the two regimes with a few large strata and many small strata. I will illustrate this theoretical issues using a numerical example in Section 5.3.2.

## 5.3.2 Numerical examples

The following function computes the Neymanian point and variance estimators:

```python
Neyman_SRE = function(z, y, x)
{
    xlevels = unique(x)
    K = length(xlevels)
    PiK = rep(0, K)
    TauK = rep(0, K)
    varK = rep(0, K)
    for(k in 1:K)
    {
    xk = xlevels[k]
    zk = z[x == xk]
    yk = y[x == xk]
```

5.3 Neymanian inference

```txt
PiK[k] = length(zk)/length(z)
TauK[k] = mean(yk[zk==1]) - mean(yk[zk==0])
varK[k] = var(yk[zk==1])/sum(zk) +
    var(yk[zk==0])/sum(1 - zk)
}
return(c(sum(PiK*TauK), sum(PiK^2*varK)))
}
```

The first simulation setting has K = 5 and each stratum has 80 units. TauHat and VarHat are the point and variance estimators over 104 simulations.

```diff
> K = 5
> n = 80
> n1 = 50
> n0 = 30
> x = rep(1:K, each = n)
> y0 = rexp(n*K, rate = x)
> y1 = y0 + 1
> zb = c(rep(1, n1), rep(0, n0))
> MC = 10^4
> TauHat = rep(0, MC)
> VarHat = rep(0, MC)
> for(mc in 1:MC)
+ {
+    z = replicate(K, sample(zb))
+    z = as.vector(z)
+    y = z*y1 + (1-z)*y0
+    est = Neyman_SRE(z, y, x)
+    TauHat[mc] = est[1]
+    VarHat[mc] = est[2]
+ }
> var(TauHat)
[1] 0.002248925
> mean(VarHat)
[1] 0.002266396
```

The upper panel of Figure 5.2 shows the histogram of the point estimator, which is symmetric and bell-shaped around the true parameter. From the above, the average value of the variance estimator is almost identical to the variance of the estimators because the individual causal effects are constant.

The first simulation setting has K = 50 and each stratum has 8 units.

```julia
> K = 50
> n = 8
> n1 = 5
> n0 = 3
> x = rep(1:K, each = n)
> y0 = rexp(n*K, rate = log(x + 1))
> y1 = y0 + 1
> zb = c(rep(1, n1), rep(0, n0))
```

```diff
> MC = 10^4
> TauHat = rep(0, MC)
> VarHat = rep(0, MC)
> for(mc in 1:MC)
+ {
+    z = replicate(K, sample(zb))
+    z = as.vector(z)
+    y = z*y1 + (1-z)*y0
+    est = Neyman_SRE(z, y, x)
+    TauHat[mc] = est[1]
+    VarHat[mc] = est[2]
+ }
>
> hist(TauHat, xlab = expression(hat(tau)[S]),
+    ylab = "", main = "many small strata",
+    border = FALSE, col = "grey",
+    breaks = 30, yaxt = 'n',
+    xlim = c(0.8, 1.2))
> abline(v = 1)
>
> var(TauHat)
[1] 0.001443111
> mean(VarHat)
[1] 0.001473616
```

The lower panel of Figure 5.2 shows the histogram of the point estimator, which is symmetric and bell-shaped around the true parameter.

We finally use the Penn Bonus Experiment to illustrate the Neymanian inference in an SRE. Applying the function NeymanSRE to the dataset, we obtain:

```txt
> est = Neyman_SRE(z, y, block)
> est[1]
[1] -0.08990646
> sqrt(est[2])
[1] 0.03079775
```

So the job training program significantly shortens the duration time before employment.

## 5.3.3 Comparing the SRE and the CRE

What are the benefits of the SRE compared to the CRE? I have motivated the SRE from the covariate balance perspective. In addition, I will show that better covariate balance in turn results in better estimation precision of the average causal effect. To make a fair comparison, I assume that $e _ { [ k ] } = e$ for all k which ensures that $\hat { \tau } = \hat { \tau } _ { \mathrm { S } }$ . I leave the proof of this result as Problem 5.1.

We now compare the sampling variances. The classic analysis of variance technique motivates the decomposition of the total variance into the summation of the within-strata and between-strata variances, yielding

$$
\begin{array}{l} S ^ {2} (1) = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} \left\{Y _ {i} (1) - \bar {Y} (1) \right\} ^ {2} \\ = (n - 1) ^ {- 1} \sum_ {k = 1} ^ {K} \sum_ {X _ {i} = k} \left\{Y _ {i} (1) - \bar {Y} _ {[ k ]} (1) + \bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \right\} ^ {2} \\ = (n - 1) ^ {- 1} \sum_ {k = 1} ^ {K} \sum_ {X _ {i} = k} \left[ \left\{Y _ {i} (1) - \bar {Y} _ {[ k ]} (1) \right\} ^ {2} + \left\{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \right\} ^ {2} \right] \\ = \sum_ {k = 1} ^ {K} \left[ \frac {n _ {[ k ]} - 1}{n - 1} S _ {[ k ]} ^ {2} (1) + \frac {n _ {[ k ]}}{n - 1} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} ^ {2} \right], \\ \end{array}
$$

and similarly,

$$
S ^ {2} (0) = \sum_ {k = 1} ^ {K} \left[ \frac {n _ {[ k ]} - 1}{n - 1} S _ {[ k ]} ^ {2} (0) + \frac {n _ {[ k ]}}{n - 1} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} ^ {2} \right],
$$

$$
{S ^ {2} (\tau)} = {\sum_ {k = 1} ^ {K} \left[ \frac {n _ {[ k ]} - 1}{n - 1} S _ {[ k ]} ^ {2} (\tau) + \frac {n _ {[ k ]}}{n - 1} \{\tau_ {[ k ]} - \tau \} ^ {2} \right].}
$$

With large strata, the variance of the difference-in-means estimator under complete randomization is approximately

$$
\begin{array}{l} \mathrm{var} _ {\mathrm{CRE}} (\hat {\tau}) \\ = \frac {S ^ {2} (1)}{n _ {1}} + \frac {S ^ {2} (0)}{n _ {0}} - \frac {S ^ {2} (\tau)}{n} \\ \approx \sum_ {k = 1} ^ {K} \left[ \frac {\pi_ {[ k ]}}{n _ {1}} S _ {[ k ]} ^ {2} (1) + \frac {\pi_ {[ k ]}}{n _ {0}} S _ {[ k ]} ^ {2} (0) - \frac {\pi_ {[ k ]}}{n} S _ {[ k ]} ^ {2} (\tau) \right] \\ + \sum_ {k = 1} ^ {K} \left[ \frac {\pi_ {[ k ]}}{n _ {1}} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} ^ {2} + \frac {\pi_ {[ k ]}}{n _ {0}} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} ^ {2} - \frac {\pi_ {[ k ]}}{n} \{\tau_ {[ k ]} - \tau \} ^ {2} \right]. \\ \end{array}
$$

The constant propensity scores assumption ensures

$$
\pi_ {[ k ]} / n _ {[ k ] 1} = 1 / (n e), \quad \pi_ {[ k ]} / n _ {[ k ] 0} = 1 / \{n (1 - e) \}, \quad \pi_ {[ k ]} / n _ {[ k ]} = 1 / n,
$$

which allow us to rewrite the variance of $\mathrm { \hat { \tau } _ { S } }$ under the SRE as

$$
\begin{array}{l} \mathrm{var} _ {\mathrm{SRE}} (\hat {\tau} _ {\mathrm{S}}) = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \left[ \frac {S _ {[ k ]} ^ {2} (1)}{n _ {[ k ] 1}} + \frac {S _ {[ k ]} ^ {2} (0)}{n _ {[ k ] 0}} - \frac {S _ {[ k ]} ^ {2} (\tau)}{n _ {[ k ]}} \right] \\ = \sum_ {k = 1} ^ {K} \left[ \frac {\pi_ {[ k ]}}{n _ {1}} S _ {[ k ]} ^ {2} (1) + \frac {\pi_ {[ k ]}}{n _ {0}} S _ {[ k ]} ^ {2} (0) - \frac {\pi_ {[ k ]}}{n} S _ {[ k ]} ^ {2} (\tau) \right]. \\ \end{array}
$$

Approximately, the difference between varCRE(ˆτ ) and $\mathrm { v a r } _ { \mathrm { S R E } } \big ( \hat { \tau } _ { \mathrm { S } } \big )$ is

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \left[ \frac {\pi_ {[ k ]}}{n _ {1}} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} ^ {2} + \frac {\pi_ {[ k ]}}{n _ {0}} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} ^ {2} - \frac {\pi_ {[ k ]}}{n} (\tau_ {[ k ]} - \tau) ^ {2} \right] \\ = \sum_ {k = 1} ^ {K} \frac {\pi_ {[ k ]}}{n} \left\{\sqrt {\frac {n _ {0}}{n _ {1}}} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} + \sqrt {\frac {n _ {1}}{n _ {0}}} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} \right\} ^ {2} \geq 0, \\ \end{array}
$$

which is non-negative. The difference is zero only in the extreme case that

$$
\sqrt {\frac {n _ {0}}{n _ {1}}} \{\bar {Y} _ {[ k ]} (1) - \bar {Y} (1) \} + \sqrt {\frac {n _ {1}}{n _ {0}}} \{\bar {Y} _ {[ k ]} (0) - \bar {Y} (0) \} = 0
$$

for $k = 1 , \ldots , K$ . When the covariate is predictive to the potential outcomes, the above quantities are usually not all zeros, which ensure the efficiency gain of the SRE compared to the CRE. Only in the extreme cases that the covariate is not predictive at all, the large-sample efficiency gain is zero. In those cases, the SRE can even result in worse estimators in finite sample. The above discussion corroborates the quote from George Box at the beginning of this chapter.

I will end this section with several remarks. First, the above comparison is based on the sampling variance, and we can also compare the estimated variances under the SRE and the CRE. The results are similar. Second, increasing K improves efficiency, but this argument depends on the large strata assumption. So we face a tradeoff in practice. We cannot arbitrarily increase K, and the most extreme case is $n _ { [ k ] 1 } = n _ { [ k ] 0 } = 1$ , which is called the matched pair experiment and will be discussed later.

## 5.4 Post-stratification in a CRE

In a CRE with a discrete covariate X, the numbers of units receiving the treatment and control are random within stratum k. In a SRE, these numbers are fixed. But if we conduct conditional inference given n = {n[k]1, n[k]0}Kk=1, ${ \pmb n } = \{ n _ { [ k ] 1 } , n _ { [ k ] 0 } \} _ { k = 1 } ^ { K } ,$ then a CRE becomes a SRE. Mathematically, if none of the components of n are zero, then

$$
\mathrm{pr} _ {\mathrm{CRE}} (\boldsymbol {Z} = \boldsymbol {z} \mid \boldsymbol {n}) = \frac {\mathrm{pr} _ {\mathrm{CRE}} (\boldsymbol {Z} = \boldsymbol {z} , \boldsymbol {n})}{\mathrm{pr} _ {\mathrm{CRE}} (\boldsymbol {n})} = \frac {1}{\prod_ {k = 1} ^ {K} \binom {n _ {[ k ]}} {n _ {[ k ] 1}}}, \tag {5.2}
$$

that is, the conditional distribution of Z from a CRE given n is identical to the distribution of Z from an SRE. So conditional on n, we can analyze a CRE with a discrete covariate X in the same way as in a SRE. In particular, the FRT becomes a conditional FRT, and the Neymanian analysis becomes post-stratification:

$$
\hat {\tau} _ {\mathrm{PS}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \hat {\tau} _ {[ k ]},
$$

which has an identical form as $\mathrm { \hat { \tau } _ { S } }$ . The variance of $\mathrm { \hat { \tau } _ { P S } }$ conditioning on n is identical to the variance of $ { \hat { \tau } _ { \mathrm { S } } }$ under the SRE.

Hennessy et al. (2016) used simulation to show that the conditional FRT is often more powerful than the unconditional one. Miratrix et al. (2013) used theory to show that in many cases, post-stratification improves efficiency compared to ${ \hat { \tau } } .$ However, the simulation is based on limited number of data generating processes, and the theory assumes all strata are large enough. We cannot go too extreme in the conditional FRT or post-stratification because with a larger K it is more likely that some $n _ { [ k ] 1 } \mathrm { ~ o r ~ } n _ { [ k ] 0 }$ become zero. Small or zero values of $n _ { [ k ] 1 } \mathrm { ~ o r ~ } n _ { [ k ] 0 }$ greatly reduces the number of randomizations in the FRT, possibly reducing the power dramatically. The problem for the Neymanian counterpart is more salient because we cannot even define $\mathrm { \hat { \tau } _ { P S } }$ and the corresponding variance estimator.

Stratification uses X in the design stage and post-stratification uses X in the analysis stage. They are duals. Asymptotically, their difference is small with large strata (Miratrix et al., 2013).

## 5.4.1 Meinert et al. (1970)’s Example

We use the data from a randomized trial from Meinert et al. (1970), which were also used by Rothman et al. (2008). The treatment is tolbutamide and the control is a placebo.

<table><tr><td colspan="3">Age &lt; 55</td><td colspan="3">Age ≥ 55</td></tr><tr><td></td><td>Surviving</td><td>Dead</td><td></td><td>Surviving</td><td>Dead</td></tr><tr><td>Z = 1</td><td>98</td><td>8</td><td>Z = 1</td><td>76</td><td>22</td></tr><tr><td>Z = 0</td><td>115</td><td>5</td><td>Z = 0</td><td>69</td><td>16</td></tr><tr><td colspan="6">Total</td></tr><tr><td></td><td></td><td>Surviving</td><td>Dead</td><td></td><td></td></tr><tr><td></td><td>Z = 1</td><td>174</td><td>30</td><td></td><td></td></tr><tr><td></td><td>Z = 0</td><td>184</td><td>21</td><td></td><td></td></tr></table>

The following table shows the estimates for two strata separately, the poststratified estimator, and the crude estimator ignoring the binary covariate, as well as the corresponding standard errors.

<table><tr><td></td><td>stratum 1</td><td>stratum 2</td><td>post-stratification</td><td>crude</td></tr><tr><td>est</td><td>-0.034</td><td>-0.036</td><td>-0.035</td><td>-0.045</td></tr><tr><td>se</td><td>0.031</td><td>0.060</td><td>0.032</td><td>0.033</td></tr></table>

Although the crude estimator and the post-stratification estimator do not lead to fundamentally different results, the crude estimator is outside the range of the stratum-specific estimators while the post-stratification estimator is within the range.

## 5.4.2 Chong et al. (2016)’s Example

Chong et al. (2016) ran a randomized experiment in Peru to study the effect of supplemental iron pills on school performance. The experiment is stratified on classlevel. I will only use a subset of the original data.

```r
library("foreign")
dat_chong = read.dta("chong.dta")
use.vars = c("treatment",
    "gradesq34",
    "class_level",
    "anemic_base_re")
dat_physician = subset(dat_chong,
    treatment != "Soccer Player",
    select = use.vars)
dat_physician$z = (dat_physician$treatment=="Physician")
dat_physician$y = dat_physician$gradesq34
```

The treatment and control group sizes vary across five strata:

```txt
> table(dat_physician$z,
+    dat_physician-class_level)
```

```txt
1 2 3 4 5
FALSE 15 19 16 12 10
TRUE 17 20 15 11 10
```

We can use the NeymanSRE function defined before to compute the stratified estimator and its estimated variance.

```erlang
tauS = with(dat_physician,
    Neyman_SRE(z, gradesq34, class_level))
```

An important additional covariate is the baseline anemic indicator which is quite important for predicting the outcome. Further conditioning the baseline anemic indicator, we have an experiment with 5 × 2 = 10 strata, with the treatment and control group sizes shown below.

```erlang
> table(dat_physician$z,
+    dat_physician-class_level,
+    dat_physician$anemic_base_re)
, , = No
```

```txt
1 2 3 4 5
FALSE 6 14 12 7 4
TRUE 8 12 9 5 6
```

```txt
，， = Yes
```

```txt
1 2 3 4 5
FALSE 9 5 4 5 6
TRUE 9 8 6 6 4
```

Again we can use the NeymanSRE function defined before to compute the poststratified estimator and its estimated variance.

```txt
tauSPS = with(dat_physician,
    {
    sps = interaction(class_level, anemic_base_re)
    Neyman_SRE(z, gradesq34, sps)
    })
```

The following table compares these two estimators. The post-stratified estimator yields a much smaller p-value.

```txt
est se t.stat p.value stratify 0.406 0.202 2.005 0.045 stratify and post-stratify 0.463 0.190 2.434 0.015
```

This example illustrates that post-stratification can be used not only in the CRE but also in the SRE with additional discrete covariates.

## 5.5 Practical questions

How do we choose X to construct a SRE? Theoretically, X should be predictive to the potential outcomes. In some cases, the experimenter has enough background knowledge about the predictive covariates based on, for example, some pilot studies. Then the choice of X should be straightforward. In some other cases, this background knowledge may not be clear enough. Experimenters instead choose X based on logistic convenience, for example, X can be indicator for the study areas or the cohort of students.

The choose of K is a related problem. Theoretically, more stratification increases the estimation efficiency if all strata are large enough. However, extremely large K may even decrease the estimation efficiency. In simulation studies, we observe diminishing marginal returns of increasing K. Anecdotally, K = 5 often suffices for efficiency gain. Some experimenter prefers the most extreme version of the SRE with $K = n / 2$ . This results in the matched pair design, which will be discussed in Chapter 7 later.

Some experiments have multidimensional continuous covariates. Can the SRE still be used? If we have a pilot study, we can build a model for the potential outcome Y (0) given those covariates, and then we can choose X as a discretized version of the predictor $\hat { Y } ( 0 )$ . In general, if we do not have such a pilot study or we do not want to make ad hoc discretizations, we can use a more general strategy called rerandomization, which is the topic for Chapter 6.

## 5.6 Homework Problems

## 5.1 Consequence of the constant propensity score

Show that if $e _ { [ k ] } = e$ for all $k = 1 , \ldots , K$ , then $\hat { \tau } = \hat { \tau } _ { \mathrm { S } }$ .

## 5.2 Consquence of constant individual causal effects

Assume that the individual causal effects are constant $\tau _ { i } ~ = ~ \tau$ for all $i \ =$ $1 , \ldots , n .$ . Consider the following class of weighted estimator for τ :

$$
\hat {\tau} _ {w} = \sum_ {k = 1} ^ {K} w _ {[ k ]} \hat {\tau} _ {[ k ]},
$$

where $w _ { [ k ] } \geq 0$ for all k.

Find the condition on the $w _ { [ k ] }$ ’s such that $\hat { \tau } _ { w }$ is unbiased for τ . Among all unbiased estimators, find the one with the minimum variance.

## 5.3 FRT for the Project STAR data in Imbens and Rubin (2015)

Reanalyze the Project STAR data using the Fisher randomization test. Note that I use Z for the treatment indicator but Imbens and Rubin (2015) use W. Use $\hat { \tau } _ { \mathrm { S } } , V$ and the aligned rank statistic in the Fisher randomization test. Compare the p-values.

```erlang
treatment = list(c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
    c(1,1,1,0,0),
    c(1,1,0,0),
    c(1,1,0,0),
```

```txt
outcome = list(c(0.165,0.321,-0.197,0.236),
    c(0.918,-0.202,1.19,0.117),
    c(0.341,0.561,-0.059,-0.496,0.225),
    c(-0.024,-0.450,-1.104,-0.956),
    c(-0.258,-0.083,-0.126,0.106),
    c(1.151,0.707,0.597,-0.495),
    c(0.077,0.371,0.685,0.270),
    c(-0.870,-0.496,-0.444,0.392,-0.934,-0.633),
    c(-0.568,-1.189,-0.891,-0.856),
    c(-0.727,-0.580,-0.473,-0.807),
    c(-0.533,0.458,-0.383,0.313),
    c(1.001,0.102,0.484,0.474,0.140),
    c(0.855,0.509,0.205,0.296),
    c(0.618,0.978,0.742,0.175),
    c(-0.545,0.234,-0.434,-0.293),
    c(-0.240,-0.150,0.355,-0.130))
```

## 5.4 A multi-center trial

Gould (1998, Table 1) reported the following data from a multi-center trial:

```csv
> multicenter = read.csv(" multicenter.csv")
> multicenter
center n0 mean0 sd0 n1 mean1 sd1 n5 mean5 sd5
1 1 7 0.43 4.58 7 -5.43 5.53 8 -2.63 3.38
2 2 11 0.10 4.21 11 -2.59 3.95 12 -2.21 4.14
3 3 6 2.58 4.80 6 -3.94 4.25 7 1.29 7.39
4 4 10 -2.30 3.86 10 -1.23 5.17 10 -1.40 2.27
5 5 10 2.08 6.46 10 -6.70 7.45 10 -5.13 3.91
6 6 6 1.13 3.24 5 3.40 8.17 5 -1.59 3.19
7 7 5 1.20 7.85 6 -3.67 4.89 5 -1.40 2.61
8 8 12 -1.21 2.66 13 0.18 3.81 12 -4.08 6.32
9 9 8 1.13 5.28 8 -2.19 5.17 9 -1.96 5.84
10 10 9 -0.11 3.62 10 -2.00 5.35 10 0.60 3.53
11 11 15 -4.37 6.12 14 -2.68 5.34 15 -2.14 4.27
12 12 8 -1.06 5.27 9 0.44 4.39 9 -2.03 5.76
13 13 12 -0.08 3.32 12 -4.60 6.16 11 -6.22 5.33
14 14 9 0.00 5.20 9 -0.25 8.23 7 -3.29 5.12
15 15 6 1.83 5.85 7 -1.23 4.33 6 -1.00 2.61
16 16 14 -4.21 7.53 14 -2.10 5.78 12 -5.75 5.63
17 17 13 0.76 3.82 13 0.55 2.53 13 -0.63 5.41
18 18 15 -1.05 4.54 13 2.54 4.16 14 -2.80 2.89
19 19 15 2.07 4.88 15 -1.67 4.95 15 -3.43 4.71
20 20 11 -1.46 5.48 10 -1.99 5.63 10 -6.77 5.19
21 21 5 0.80 4.21 5 -3.35 4.73 5 -0.23 4.14
22 22 11 -2.92 5.42 10 -1.22 5.95 11 -4.45 6.65
23 23 9 -3.37 4.73 9 -1.38 4.17 7 0.57 2.70
24 24 12 -1.92 2.91 12 -0.66 3.55 12 -2.39 2.27
25 25 9 -3.89 4.76 9 -3.22 5.54 8 -1.23 4.91
```

## 5.6 Homework Problems

<table><tr><td>26</td><td>26</td><td>15</td><td>-3.48</td><td>5.98</td><td>15</td><td>-2.13</td><td>3.25</td><td>14</td><td>-3.71</td><td>5.30</td></tr><tr><td>27</td><td>27</td><td>11</td><td>-1.91</td><td>6.49</td><td>12</td><td>-1.33</td><td>4.40</td><td>11</td><td>-1.52</td><td>4.68</td></tr><tr><td>28</td><td>28</td><td>10</td><td>-2.66</td><td>3.80</td><td>10</td><td>-1.29</td><td>3.18</td><td>10</td><td>-4.70</td><td>3.43</td></tr><tr><td>29</td><td>29</td><td>13</td><td>-0.77</td><td>4.73</td><td>13</td><td>-2.31</td><td>3.88</td><td>13</td><td>-0.47</td><td>4.95</td></tr></table>

This is a SRE with centers being the strata. The trial was conducted to study the efficacy and tolerability of finasteride, a drug for treating benign prostatic hyperplasia. Within each of the 29 centers, patients were randomized into three arms: control, finasteride 1mg, and finasteride 5mg. The above dataset provides summary statistics for the outcome, which is the change from baseline in total symptom score. The total symptom score is the sum of the responses to nine questions (score 0 to 4) about symptoms pertaining to various aspects of impaired urinary ability. The meanings of the columns are:

1. center: number of the center;  
2. n0, n1, n5: sample sizes of the three arms;  
3. mean0, mean1, mean5: mean of the outcome;  
4. sd0, sd1, sd5: standard deviation of the outcome.

The individual-level outcomes are not reported so we cannot implement the FRT. However, the Neymanian inference only requires the summary statistics. Report the point estimators and variance estimators for comparing “finasteride 1mg” and “finasteride 5mg” to “control”, separately.

## 5.5 Data re-analyses

Re-analyze the LaLonde data used in Neymanlalonde.R. Conduct both Fisherian and Neymanian inferences.

The original experiment is a completely randomized experiment. Now we pretend that the original experiment is a stratified randomized experiment. First, re-analyze the data pretending that the experiment is stratified on the race (black, Hispanic or other). Second, re-analyze the data pretending that the experiment is stratified on marital status. Third, re-analyze the data pretending that the experiment is stratified on the indicator of high school diploma.

Compare with the results obtained under a completely randomized experiments.

## 5.6 Recommended reading

Miratrix et al. (2013) provided solid theory for post-stratification and compared it with stratification. A main theoretical result is that their difference is small asymptotically although they can differ in finite samples.

## 6

# Rerandomization and Regression Adjustment

Stratification and post-stratification in Chapter 5 are duals for discrete covariates in the design and analysis of randomized experiments. How should we deal with multidimensional possibly continuous covariates? We can discretize continous covariates, but this is not an ideal strategy with many covariates. Rerandomization and regression adjustment are duals for general covariates, which are the topics for this chapter.

The following table summarizes the topics of Chapters 5 and 6:

<table><tr><td></td><td>design</td><td>analysis</td></tr><tr><td>discrete covariate</td><td>stratification</td><td>post-stratification</td></tr><tr><td>general covariate</td><td>rerandomization</td><td>regression adjustment</td></tr></table>

## 6.1 Rerandomization

## 6.1.1 Experimental design

Again we consider a finite population of n units, where $n _ { 1 }$ of them receive the treatment and $n _ { 0 }$ of them receive the control. Let $\pmb { Z } = ( Z _ { 1 } , \ldots , Z _ { n } )$ be the treatment vector for these units. Unit i has covariate $X _ { i } \in \mathbb { R } ^ { K }$ which can have continuous or binary components. Concatenate them as $\pmb { X } = ( X _ { 1 } , \ldots , X _ { n } )$ and center them at mean zero $\begin{array} { r } { \bar { X } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } X _ { i } = 0 } \end{array}$ without loss of generality.

The CRE balances the covariates in the treatment and control groups on average, for instance, the difference in means of the covariates

$$
\hat {\tau} _ {X} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} X _ {i} - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) X _ {i}
$$

has mean zero under the CRE. However, it can result in undesirable covariate balance across the treatment and control groups in the realized treatment allocation, that is, the realized value of $\hat { \tau } _ { X }$ is often not zero. Using the vector form of Neyman (1923) in Problem 4.6, we can show that

$$
\operatorname{cov} (\hat {\tau} _ {X}) = \frac {1}{n _ {1}} S _ {X} ^ {2} + \frac {1}{n _ {0}} S _ {X} ^ {2} = \frac {n}{n _ {1} n _ {0}} S _ {X} ^ {2},
$$

where $\begin{array} { r } { S _ { X } ^ { 2 } = ( n - 1 ) ^ { - 1 } \sum _ { i = 1 } ^ { n } X _ { i } X _ { i } ^ { \mathsf { T } } } \end{array}$ . The following Mahalanobis distance measures the difference between the treatment and control groups:

$$
M = \hat {\tau} _ {X} ^ {\mathsf {T}} \mathrm{cov} (\hat {\tau} _ {X}) ^ {- 1} \hat {\tau} _ {X} = \hat {\tau} _ {X} ^ {\mathsf {T}} \left(\frac {n}{n _ {1} n _ {0}} S _ {X} ^ {2}\right) ^ {- 1} \hat {\tau} _ {X}.
$$

Technically the above formula of M is meaningful only if $S _ { X } ^ { 2 }$ is invertible, which means that the columns of the covariate matrix are linearly independent. If a column can be represented by a linear combinations of other columns, it is redundant and should be dropped before the experiment. A nice feature of M is that it is invariance under non-degenerate linear transformations of X. Lemma 6.1 below summarizes the result with the proof relegated to Problem 6.2.

Lemma 6.1 M remains the same $i f$ we transform $X _ { i }$ to $\alpha + B X _ { i }$ for all units $i = 1 , \ldots , n$ where α $\in \mathbb { R } ^ { K }$ and $B \in \mathbb { R } ^ { K \times K }$ is invertible.

The finite population central limit theorem (Li and Ding, 2017) ensures that with large $n ,$ the Mahalanobis distance M is approximately $\chi _ { K } ^ { 2 }$ under the CRE. Therefore, it is likely that M has a large realized value under the CRE with asymptotic mean K and variance 2K. Rerandomization avoids covariate imbalance by discarding the treatment allocations with large values of M. Below I give a formal definition of the rerandomization using the Mahalanobis distance (ReM), which was proposed by Cox (1982) and Morgan and Rubin (2012).

Definition 6.1 (ReM) Draw Z from CRE and accept it if and only if

$$
M \leq a,
$$

for some predetermined constant $a > 0$ .

Choosing a is like choosing the number of strata in the SRE, which is a non-trivial problem in practice. At one extreme, $a = \infty$ , we just conduct the CRE. At the other extreme, $a = 0$ , there are very few feasible treatment allocations, and consequently, the experiment has little randomness, rendering randomization-based inference useless. As a compromise, we choose a small but not extremely small $^ { a , }$ for example, $a = 0 . 0 0 1$ or some upper quantile of a $\chi _ { K } ^ { 2 }$ distribution.

ReM uses the Mahalanobis distance as the balance criterion. We can consider general rerandomization with the balance criterion defined as a function of Z and X. For example, we can use the following criterion based on marginal tests for all coordinates of $X _ { i } = ( x _ { i 1 } , \ldots , x _ { i K } ) ^ { \mathsf { T } }$ . We accept Z if and only if

$$
\left| \frac {\hat {\tau} _ {x k}}{\sqrt {\frac {n}{n _ {1} n _ {0}} S _ {x k} ^ {2}}} \right| \leq a (k = 1, \dots , K) \tag {6.1}
$$

for some predetermined constant $a > 0 .$ . For example, a some upper quantile of a standard Normal distribution. ReM has many desirable properties. As mentioned above, it is invariant to linear transformation of the covariates. Moreover, it has nice geometric properties and elegant mathematical theory. This chapter will focus on ReM. See Zhao and Ding (2021b) for the theory for the rerandomization based on criterion (6.1) as well as other criteria.

## 6.1.2 Statistical inference

An important question is how to analyze the data under ReM. Bruhn and McKenzie (2009) and Morgan and Rubin (2012) argued that we can always use the FRT as long as we simulate Z under the constraint $M \ \leq \ a$ . This always yields finite-sample exact p-values under the sharp null hypothesis.

It is a challenging problem to derive the finite sample properties of ReM without assuming the sharp null hypothesis. Instead, Li et al. (2018b) derived the asymptotic distribution of the difference in means of the outcome $\hat { \tau }$ under ReM and the regularity conditions below.

## Condition 6.1 As $n \to \infty$

1. $n _ { 1 } / n$ and $n _ { 0 } / n$ have positive limits;  
2. the finite population covariance of $\{ X _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) , \tau _ { i } \}$ has limit;  
3. max $\iota \leq i \leq n  \vert Y _ { i } ( 1 ) - \bar { Y } ( 1 ) \vert ^ { 2 } / n  0$ , max $\phantom { } _ { 1 \leq i \leq n } | Y _ { i } ( 0 ) - \bar { Y } ( 0 ) | ^ { 2 } / n \to$ $0 ,$ and $\operatorname* { m a x } _ { 1 \leq i \leq n } \| X _ { i } \| ^ { 2 } / n  0$ ,

Below is the main theorem for ReM. Let

$$
L _ {K, a} \sim D _ {1} \mid \boldsymbol {D} ^ {\mathsf {T}} \boldsymbol {D} \leq a
$$

where $\pmb { { \cal D } } = ( D _ { 1 } , \ldots , D _ { K } )$ follows a K-dimensional standard Normal distribution; let ε follows a univariate standard Normal distribution; $L _ { K , a } \bot \varepsilon$ .

Theorem 6.1 Under ReM with $M \leq a$ and Condition 6.1, we $h a v e ^ { 1 }$

$$
\hat {\tau} - \tau \dot {\sim} \sqrt {\mathrm{var} (\tau)} \left\{\sqrt {R ^ {2}} L _ {K, a} + \sqrt {1 - R ^ {2}} \varepsilon \right\},
$$

where

$$
\mathrm{var} (\hat {\tau}) = \frac {S ^ {2} (1)}{n _ {1}} + \frac {S ^ {2} (0)}{n _ {0}} - \frac {S ^ {2} (\tau)}{n}
$$

is Neyman (1923)’s variance formula proved in Chapter $^ { 4 , }$ and

$$
R ^ {2} = \mathrm{corr} ^ {2} (\hat {\tau}, \hat {\tau} _ {X})
$$

![image_06](images/image_06.png)

Rerandomization
area
O
θ
√R²Lₖ,ₐ
τ̂ - τ
√1 - R²ε
τ̂ₓ

FIGURE 6.1: Geometry of ReM

is the squared multiple correlation coefficient2 between $\hat { \tau }$ and $\hat { \tau } _ { X }$ under the CRE.

Although the proof of Li et al. (2018b) is technical, the asymptotic distribution in Theorem 6.1 has clear geometric interpretation, as shown in Figure 6.1. It shows that $\hat { \tau }$ decomposes into a component that is a linear combination of $\hat { \tau } _ { X }$ and a component that is orthogonal to $\hat { \tau } _ { X }$ . Geometrically, $\cos ^ { 2 } \theta = R ^ { 2 }$ , where θ is the angle between $\hat { \tau }$ and $\hat { \tau } _ { X }$ . ReM affects the first component but does not change the second component. The truncated Normal distribution $L _ { K , a }$ is due to the restriction of ReM on the first component.

When $a = \infty$ , the asymptotic distribution simplifies to the one under the CRE:

$$
\hat {\tau} - \tau \dot {\sim} \sqrt {\mathrm{var} (\tau)} \varepsilon .
$$

When the threshold a is close to zero, the the asymptotic distribution simplifies to

$$
\hat {\tau} - \tau \dot {\sim} \sqrt {\mathrm{var} (\tau) (1 - R ^ {2})} \varepsilon .
$$

So with a small threshold a, the efficiency gain due to ReM depends on $R ^ { 2 }$ , which has the following equivalent form.

Proposition 6.1 Under the CRE,

$$
R ^ {2} = \mathrm{corr} ^ {2} (\hat {\tau}, \hat {\tau} _ {X}) = \frac {n _ {1} ^ {- 1} S ^ {2} (1 \mid x) + n _ {0} ^ {- 1} S ^ {2} (0 \mid x) - n ^ {- 1} S ^ {2} (\tau \mid x)}{n _ {1} ^ {- 1} S ^ {2} (1) + n _ {0} ^ {- 1} S ^ {2} (0) - n ^ {- 1} S ^ {2} (\tau)},
$$

$$
R _ {y X} ^ {2} = \mathrm{corr} ^ {2} (y, X) = \frac {\mathrm{cov} (y , X) \mathrm{cov} (X) ^ {- 1} \mathrm{cov} (X , y)}{\mathrm{var} (y)}.
$$

It extends the definition of the Pearson correlation coefficient and measures the linear dependence of y on X.

<!-- footnote -->

- It becomes the title of a book on the modern history of statistics by Salsburg (2001)

<!-- footnote end -->

<!-- footnote -->

- In causal inference, we call $X _ { i }$ a covariate if it is not affected by the treatment. That is, if the covariate has two potential outcomes $X _ { i } ( 1 )$ and $X _ { i } ( 0 )$ , then they must satisfy $X _ { i } ( 1 ) =$ $X _ { i } ( 0 )$ . Standard statistics books often do not distinguish the treatment and covariates because they often appear on the right-hand side of a regression model for the outcome. They are both called covariates in those statistical models. This book distinguishes the treatment and covariates because they play different roles in causal inference.

<!-- footnote end -->

<!-- footnote -->

- Here the divisor $n - 1$ makes the formulas more elegant. Changing the divisor to n complicates the formulas but does not change the results fundamentally. With large $^ { n , }$ the difference is minor.

<!-- footnote end -->

<!-- footnote -->

- In the classic two-sample problem, the outcomes under treatment are IID draws from a distribution with mean $\mu _ { 1 }$ and variance $\sigma _ { 1 } ^ { 2 }$ , and the outcomes under control are IID draws from a distribution with mean $\mu _ { 0 }$ and variance $\sigma _ { 0 } ^ { 2 }$ . Under this assumption, we have
- $\mathrm { v a r } ( \hat { \tau } ) = \frac { \sigma _ { 1 } ^ { 2 } } { n _ { 1 } } + \frac { \sigma _ { 0 } ^ { 2 } } { n _ { 0 } } .$ n0
- Here, var(·) is over the randomness of the outcomes. This variance formula does not involve a third term that depends on the variance of the individual causal effects.

<!-- footnote end -->

<!-- footnote -->

- His most famous quote is “all models are wrong but some are useful.”

<!-- footnote end -->

<!-- footnote -->

- The notation ${ } ^ { \ast } A \stackrel { \cdot } { \sim } B ^ { \prime \prime }$ means that A and B have the same asymptotic distributions.

<!-- footnote end -->

<!-- footnote -->

- The squared multiple correlation coefficient between a random variable y and a random vector X is defined as

<!-- footnote end -->

where $\{ S ^ { 2 } ( 1 ) , S ^ { 2 } ( 0 ) , S ^ { 2 } ( \tau ) \}$ are the finite population variances of $\{ Y _ { i } ( 1 ) , Y _ { i } ( 0 ) , \tau _ { i } \} _ { i = 1 } ^ { n }$ , and $\{ \hat { S ^ { 2 } } ( 1 \mid x ) , \hat { S ^ { 2 } } ( 0 \mid x ) , \hat { S ^ { 2 } } ( \tau \mid x ) \}$ are the corresponding finite population variances of their linear projections on $( 1 , X _ { i } ) .$ . 3 Under the constant causal effect assumption with $\tau _ { i } ~ = ~ \tau$ , $R ^ { 2 }$ reduces to the finite population squared multiple correlation between $Y _ { i } ( 0 )$ and $X _ { i }$ .

I leave the proof of Proposition 6.1 to Problem 6.4.

When $0 < a < \infty .$ , the asymptotic distribution has a more complicated form and is more concentrated at τ and thus the difference in means is more precise under ReM than under the CRE.

If we ignore the design of ReM and still use the confidence interval based on Neyman (1923)’s variance formula and the Normal approximation, it is overly conservative and overcovers τ even if the individual causal effects are constant. Li et al. (2018b) described how to construct confidence intervals based on Theorem 6.1. We omit the discussion here but will come back to the inference issue in Section 6.3.

## 6.2 Regression adjustment

What if we do not conduct rerandomization in the design stage but want to adjust for covariate imbalance in the analysis stage of the CRE? We will discuss several regression adjustment strategies.

## 6.2.1 Covariate-adjusted FRT

The covariates X are all fixed, and furthermore, under $H _ { \mathrm { 0 F } }$ , the observed outcomes are all fixed. Therefore, we can simulate the distribution of any test statistic $T ( Z , Y , X )$ ) and calculate the p-value. The basic idea of the FRT remains the same in the presence additional covariates.

There are two general strategies to construct the test statistic, as summarized by Zhao and Ding (2021a). Problem 3.6 hints at both of them. I summarize them below:

• The first strategy is to construct the test statistic based on residuals from fitted statistical models. We can regress $Y _ { i }$ on $X _ { i }$ to obtain residual $\varepsilon _ { i } .$ , and then treat $\varepsilon _ { i }$ as the pseudo outcome to construct test statistics.

• The second strategy is to use a regression coefficient as a test statistic. We can regress $Y _ { i }$ on $( Z _ { i } , X _ { i } )$ to obtain the coefficient of $Z _ { i }$ as the test statistic. The rest of this section will review some test statistics based on OLS.

In strategy one, we only need to run regression once, but in strategy two, we need to run regression many times. In the above, “regression” is a generic term, which can be linear regression, logistic regression, or even machine learning algorithms. The FRT with any test statistics from these two strategies will be finite-sample exact under $H _ { \mathrm { 0 F } }$ although they differ under alternative hypotheses.

## 6.2.2 Analysis of covariance and extensions

Now we turn to direct estimation of the average causal effect $\tau$ that adjusts for the observed covariates.

Historically, Fisher (1925) proposed to use the analysis of covariance $( \mathrm { A N } .$ COVA) to improve estimation efficiency. This remains a standard strategy in many fields. He suggested running the OLS of $Y _ { i }$ on $( Z _ { i } , X _ { i } )$ and obtaining the coefficient of $Z _ { i }$ as an estimator for τ . Let $\hat { \tau } _ { \mathrm { F } }$ denote Fisher’s ANCOVA estimator.

A former Berkeley Statistics Professor, David Freedman, reanalyzed Fisher’s ANCOVA under Neyman (1923)’s potential outcomes framework. Freedman (2008a,b) found the following negative results:

1. $\hat { \tau } _ { \mathrm { F } }$ is biased, but the simple difference in means $\hat { \tau }$ is unbiased.  
2. The asymptotic variance of $\hat { \tau } _ { \mathrm { F } }$ may be even larger than that of $\hat { \tau }$  
3. The standard error from the OLS is inconsistent for the true standard error of $\hat { \tau } _ { \mathrm { F } }$ under the CRE.

A Berkeley Ph.D. student, Winston Lin, wrote a thesis in response to Freedman’s critiques. Lin (2013) found the following positive results:

1. The bias of $\hat { \tau } _ { \mathrm { F } }$ is small in large samples, and it goes to zero as the sample size approaches infinity.  
2. We can improve the asymptotic efficiency of both $\hat { \tau }$ and ˆτF by using the coefficient of $Z _ { i }$ in the OLS of $Y _ { i }$ on $( Z _ { i } , X _ { i } , Z _ { i } \times X _ { i } )$ . Let $\hat { \tau } _ { \mathrm { L } }$ denote Lin (2013)’s estimator. Moreover, the EHW standard error is a conservative estimator for the true standard error of $\hat { \tau } _ { \mathrm { L } }$ under the CRE.  
3. The EHW standard $\mathrm { e r r o r ^ { 4 } }$ for $\hat { \tau } _ { \mathrm { F } }$ in the OLS fit of $Y _ { i }$ on $( Z _ { i } , X _ { i } )$ is

a conservative estimator for the true standard error of $\hat { \tau } _ { \mathrm { F } }$ under the CRE.

## 6.2.2.1 Some heuristics for Lin (2013)’s results

Neyman (1923)’s result demonstrates that the variance of the difference-inmeans estimator depends on the variances of the potential outcomes. Intuitively, we can reduce the variance of the estimator by reducing the variances of the outcomes. A simple family of linearly adjusted estimator is

$$
\begin{array}{l} \hat {\tau} \left(\beta_ {1}, \beta_ {0}\right) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \left(Y _ {i} - \beta_ {1} ^ {\mathsf {T}} X _ {i}\right) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} \left(1 - Z _ {i}\right) \left(Y _ {i} - \beta_ {0} ^ {\mathsf {T}} X _ {i}\right) (6. 2) \\ = \left\{\hat {\bar {Y}} (1) - \beta_ {1} ^ {\mathsf {T}} \hat {\bar {X}} (1) \right\} - \left\{\hat {\bar {Y}} (0) - \beta_ {0} ^ {\mathsf {T}} \hat {\bar {X}} (0) \right\}, \tag {6.3} \\ \end{array}
$$

where $\{ \hat { \bar { Y } } ( 1 ) , \hat { \bar { Y } } ( 0 ) \}$ are the sample means of the outcomes, and $\{ \hat { \bar { X } } ( 1 ) , \hat { \bar { X } } ( 0 ) \}$ are the sample means of the covariates. This covariate-adjusted estimator $\hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } )$ tries to reduce the variance of ˆτ by residualizing the potential outcomes. It reduces $\mathrm { t o } \ \hat { \tau }$ with $\beta _ { 1 } = \beta _ { 0 } = 0 .$ . It has mean τ for any fixed values of $\beta _ { 1 }$ and $\beta _ { 0 }$ because $\bar { X } = 0$ . We are interested in finding the $( \beta _ { 1 } , \beta _ { 0 } )$ that minimized the variance of $\hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } )$ . This estimator is essentially the difference in means of the adjusted potential outcomes $\{ Y _ { i } ( 1 ) - \beta _ { 1 } ^ { \mathsf { T } } X _ { i } , Y _ { i } ( 0 ) - \beta _ { 0 } ^ { \mathsf { T } } X _ { i } \} _ { i = 1 } ^ { n }$ . Applying Neyman (1923)’s result, this estimator has variance

$$
\operatorname{var} \{\hat {\tau} (\beta_ {1}, \beta_ {0}) \} = \frac {S ^ {2} (1 ; \beta_ {1})}{n _ {1}} + \frac {S ^ {2} (0 ; \beta_ {1})}{n _ {0}} - \frac {S ^ {2} (\tau ; \beta_ {1} , \beta_ {0})}{n},
$$

where $S ^ { 2 } ( z ; \beta _ { 1 } ) ~ ( z = 1 , 0 )$ and $S ^ { 2 } ( \tau ; \beta _ { 1 } , \beta _ { 0 } )$ are the finite population variances of the adjusted potential outcomes and individual effects, respectively; moreover, a conservative variance estimate is

$$
\hat {V} (\beta_ {1}, \beta_ {0}) = \frac {\hat {S} ^ {2} (1 ; \beta_ {1})}{n _ {1}} + \frac {\hat {S} ^ {2} (0 ; \beta_ {1})}{n _ {0}},
$$

where

$$
\hat {S} ^ {2} (1; \beta_ {1}) = (n _ {1} - 1) ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \{Y _ {i} - \gamma_ {1} - \beta_ {1} ^ {\mathsf {T}} X _ {i} \} ^ {2},
$$

$$
\hat {S} ^ {2} (0; \beta_ {0}) = (n _ {0} - 1) ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \{Y _ {i} - \gamma_ {0} - \beta_ {0} ^ {\mathsf {T}} X _ {i} \} ^ {2}
$$

are the sample variances of the adjusted potential outcomes with $\gamma _ { 1 }$ and $\gamma _ { 0 }$ being the sample means of $Y _ { i } - \beta _ { 1 } ^ { \mathsf { T } } X _ { i }$ under treatment and $Y _ { i } - \beta _ { 0 } ^ { \mathsf { T } } X _ { i }$ under control. To minimize $\hat { V } ( \beta _ { 1 } , \beta _ { 0 } )$ , we need to solve two OLS problems:

$$
\min _ {\gamma_ {1}, \beta_ {1}} \sum_ {i = 1} ^ {n} Z _ {i} \{Y _ {i} - \gamma_ {1} - \beta_ {1} ^ {\mathsf {T}} X _ {i} \} ^ {2}, \quad \min _ {\gamma_ {0}, \beta_ {0}} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \{Y _ {i} - \gamma_ {0} - \beta_ {0} ^ {\mathsf {T}} X _ {i} \} ^ {2}.
$$

We run OLS of $Y _ { i }$ on $X _ { i }$ for the treatment and control groups separately and obtain $( \hat { \gamma } _ { 1 } , \hat { \beta } _ { 1 } )$ and $( \hat { \gamma } _ { 0 } , \hat { \beta } _ { 0 } )$ . The final estimator is

$$
\begin{array}{l} \hat {\tau} (\hat {\beta} _ {1}, \hat {\beta} _ {0}) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\beta} _ {1} ^ {\mathsf {T}} X _ {i}) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) (Y _ {i} - \hat {\beta} _ {0} ^ {\mathsf {T}} X _ {i}) \\ = \left\{\hat {\bar {Y}} (1) - \hat {\beta} _ {1} ^ {\mathsf {T}} \hat {\bar {X}} (1) \right\} - \left\{\hat {\bar {Y}} (0) - \hat {\beta} _ {0} ^ {\mathsf {T}} \hat {\bar {X}} (0) \right\}. \\ \end{array}
$$

From the properties of the OLS fits (see (A2.3)), we know

$$
\hat {\bar {Y}} (1) = \hat {\gamma} _ {1} + \hat {\beta} _ {1} ^ {\mathsf {T}} \hat {\bar {X}} (1), \quad \hat {\bar {Y}} (0) = \hat {\gamma} _ {0} + \hat {\beta} _ {0} ^ {\mathsf {T}} \hat {\bar {X}} (0).
$$

Therefore, we can rewrite the estimator as

$$
\hat {\tau} \left(\hat {\beta} _ {1}, \hat {\beta} _ {0}\right) = \hat {\gamma} _ {1} - \hat {\gamma} _ {0} \tag {6.4}
$$

The equivalent form in (6.4) suggests that we can obtain $\hat { \tau } ( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$ from a single OLS fit below.

Proposition 6.2 The estimator $\hat { \tau } ( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$ in (6.4) equals the coefficient of $Z _ { i }$ in the OLS fit of Yi on $( Z _ { i } , X _ { i } , Z _ { i } \times X _ { i } )$ , which $i s \ \hat { \tau } _ { \mathrm { L } }$ introduced before.

I leave the proof of Proposition 6.2 to Problem 6.5, which is a pure algebra fact.

Based on the discussion above, a conservative variance estimator for $\hat { \tau } _ { \mathrm { L } }$ is

$$
\begin{array}{l} \hat {V} (\hat {\beta} _ {1}, \hat {\beta} _ {0}) = \frac {1}{n _ {1} (n _ {1} - 1)} \sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\gamma} _ {1} - \hat {\beta} _ {1} ^ {\mathsf {T}} X _ {i}) ^ {2} \\ + \frac {1}{n _ {0} (n _ {0} - 1)} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) (Y _ {i} - \hat {\gamma} _ {0} - \hat {\beta} _ {0} ^ {\mathsf {T}} X _ {i}) ^ {2}. \\ \end{array}
$$

Based on quite technical calculations, Lin (2013) further showed that the EHW standard error from the OLS in Proposition 6.2 is almost identical to $\hat { V } ( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$ which is a conservative estimator of the true standard error of $\scriptstyle { \hat { \tau } } _ { \mathrm { L } }$ under the CRE. Intuitively, this is because we do not assume that the linear model is correctly specified, and the EHW standard error is robust to model misspecification.

There is a subtle issue with the discussion above. The variance formula va $\cdot \{ \hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } ) \}$ works for fixed $( \beta _ { 1 } , \beta _ { 0 } )$ , but the estimator $\hat { \tau } ( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$ uses two estimated coefficients $( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$ . The additional uncertainty in the estimated coefficients may cause finite-sample bias in the final estimator. Lin (2013) showed that the issue goes away asymptotically. However, his theory requires a large sample size and some regularity conditions on the potential outcomes and covariates.

**TABLE 6.1: Predicting the potential outcomes**

<table><tr><td>X</td><td>Z</td><td>Y(1)</td><td>Y(0)</td><td> $\hat{Y}(1)$ </td><td> $\hat{Y}(0)$ </td></tr><tr><td> $X_1$ </td><td>1</td><td> $Y_1(1)$ </td><td>?</td><td> $\hat{\mu}_1(X_1)$ </td><td> $\hat{\mu}_0(X_1)$ </td></tr><tr><td> $\vdots$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $X_{n_1}$ </td><td>1</td><td> $Y_{n_1}(1)$ </td><td>?</td><td> $\hat{\mu}_1(X_{n_1})$ </td><td> $\hat{\mu}_0(X_{n_1})$ </td></tr><tr><td> $X_{n_1+1}$ </td><td>0</td><td>?</td><td> $Y_{n_1+1}(0)$ </td><td> $\hat{\mu}_1(X_{n_1+1})$ </td><td> $\hat{\mu}_0(X_{n_1+1})$ </td></tr><tr><td> $\vdots$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $X_n$ </td><td>0</td><td>?</td><td> $Y_n(0)$ </td><td> $\hat{\mu}_1(X_n)$ </td><td> $\hat{\mu}_0(X_n)$ </td></tr></table>

## 6.2.2.2 Understanding Lin (2013)’s estimator via predicting the potential outcomes

We can view Lin (2013)’s estimator as a predictive estimator based on OLS fits of the potential outcomes. We build a prediction model for $Y ( 1 )$ based on X using the data from the treatment group:

$$
\hat {\mu} _ {1} (x) = \hat {\gamma} _ {1} + \hat {\beta} _ {1} ^ {\mathsf {T}} x. \tag {6.5}
$$

Similarly, we build a prediction model for Y (0) based on X using the data from the control group:

$$
\hat {\mu} _ {0} (x) = \hat {\gamma} _ {0} + \hat {\beta} _ {0} ^ {\mathsf {T}} x. \tag {6.6}
$$

If we predict the missing potential outcomes, then we have the following predictive estimator:

$$
\hat {\tau} _ {\text { pred }} = n ^ {- 1} \left\{\sum_ {Z _ {i} = 1} Y _ {i} + \sum_ {Z _ {i} = 0} \hat {\mu} _ {1} (X _ {i}) - \sum_ {Z _ {i} = 1} \hat {\mu} _ {0} (X _ {i}) - \sum_ {Z _ {i} = 0} Y _ {i} \right\}. \tag {6.7}
$$

We can verify that with (6.5) and (6.6), the predictive estimator equals Lin (2013)’s estimator:

$$
\hat {\tau} _ {\mathrm{pred}} = \hat {\tau} _ {\mathrm{L}}. (6. 8)
$$

If we predict all potential outcomes even if they are observed, we have the following projective estimator:

$$
\hat {\tau} _ {\text { proj }} = n ^ {- 1} \sum_ {i = 1} ^ {n} \{\hat {\mu} _ {1} (X _ {i}) - \hat {\mu} _ {0} (X _ {i}) \}. \tag {6.9}
$$

We can verify that with (6.5) and (6.6), the projective estimator equals Lin (2013)’s estimator:

$$
\hat {\tau} _ {\mathrm{proj}} = \hat {\tau} _ {\mathrm{L}}. \tag {6.10}
$$

I leave the proofs of (6.8) and (6.10) to Problem 6.6.

The more general formulas (6.7) and (6.9) are well defined with other predictors of the potential outcomes. To make connections with Lin (2013)’s estimator, I focus on the linear predictors here. They can be quite general, including much more complicated machine learning algorithms. However, constructing point estimator is just the first step in analyzing the CRE. A more important second step is to quantify the uncertainty associated with the estimator, which depends on the properties of the predictors of the potential outcomes. Nevertheless, without doing additional theoretical analysis, we can always use (6.7) and (6.9) as the test statistics in the FRT.

## 6.2.2.3 Understanding Lin (2013)’s estimator via adjusting for covariate imbalance

The linearly-adjusted estimator has an equivalent form

$$
\hat {\tau} (\beta_ {1}, \beta_ {0}) = \hat {\tau} - \gamma^ {\mathsf {T}} \hat {\tau} _ {X} \tag {6.11}
$$

where $\begin{array} { r } { \gamma = \frac { n _ { 0 } } { n } \beta _ { 1 } + \frac { n _ { 1 } } { n } \beta _ { 0 } } \end{array}$ , so we can also write it as $\hat { \tau } ( \gamma ) = \hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } )$ . Similarly, Lin (2013)’s estimator has an equivalent form

$$
\hat {\tau} _ {\mathrm{L}} = \hat {\tau} - \hat {\gamma} ^ {\mathsf {T}} \hat {\tau} _ {X}, \tag {6.12}
$$

where $\begin{array} { r } { \hat { \gamma } = \frac { n _ { 0 } } { n } \hat { \beta } _ { 1 } + \frac { n _ { 1 } } { n } \hat { \beta } _ { 0 } } \end{array}$ . I leave the proofs of (6.11) and (6.12) to Problem 6.7. The forms (6.11) and (6.12) are the mathematical statements of “adjusting for the covariate imbalance.” They essentially subtract some linear combinations of the difference in means of the covariates. Since ˆτ and $\hat { \tau } _ { X }$ are correlated, the covariate adjustment with an appropriate γ reduces the variance of ˆτ . Another interesting feature of (6.11) and (6.12) is that the final estimators depend only on γ or $\hat { \gamma } ,$ so the choice of the β-coefficients are not unique. Therefore, Lin (2013)’s estimator is just one of the optimal estimators, but it can be easily implemented via the standard OLS with the EHW standard error.

## 6.2.3 Some additional remarks on regression adjustment

## 6.2.3.1 Duality between ReM and regression adjustment

Li et al. (2018b) pointed out that ReM and Lin (2013)’s regression adjustment are duals in using covariates in the design and analysis stages of the experiment. To be more specific, when a is small, the asymptotic distribution of ˆτ under ReM is almost identical to the asymptotic distribution of $\hat { \tau } _ { \mathrm { L } }$ under the CRE. So ReM uses covariates in the design stage and Lin (2013)’s regression adjustment uses covariates in the analysis stage, achieving nearly the same asymptotic efficiency gain when a is small.

## 6.2.3.2 Equivalence of regression adjustment and post-stratification

If we have discrete covariate $C _ { i }$ with $K$ categories, we can create $K - 1$ centered dummy variables

$$
X _ {i} = (I (C _ {i} = 1) - \pi_ {[ 1 ]}, \ldots , I (C _ {i} = K - 1) - \pi_ {[ K - 1 ]}).
$$

In this case, Lin $( \mathrm { 2 0 1 3 ) \mathrm { ^ { \circ } s } }$ regression adjustment is equivalent to poststratification, as summarized by the following proposition.

Proposition 6.3 $\hat { \tau } _ { \mathrm { L } }$ based in $X _ { i }$ is numerically identical to the poststratification estimator based on $C _ { i }$ .

I leave the proof of Proposition 6.3 as Problem 6.9.

## 6.2.3.3 Difference-in-difference as a special case of covariate adjustment $\hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } )$

An important covariate $X$ in many studies is the lagged outcome before the treatment. For instance, the covariate X is the pre-test score if the outcome $Y$ is the post-test score in educational research; the covariate $X$ is the log wage before the job training program if the outcome $Y$ is the log wage after the job training program. With the lagged outcome $X$ as a covariate, a popular estimator is the gain score or difference-in-difference estimator with $\beta _ { 1 } = \beta _ { 0 } =$ 1 in (6.2) and (6.3):

$$
\begin{array}{l} \hat {\tau} (1, 1) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - X _ {i}) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) (Y _ {i} - X _ {i}) \\ { = } { \left\{ \hat { \bar { Y } } ( 1 ) - \hat { \bar { Y } } ( 0 ) \right\} - \left\{ \hat { \bar { X } } ( 1 ) - \hat { \bar { X } } ( 0 ) \right\} . } \\ \end{array}
$$

The first form of $\hat { \tau } ( 1 , 1 )$ justifies the name gain score because it is essentially the difference in means of the gain score $g _ { i } = Y _ { i } - X _ { i }$ . The second form of $\hat { \tau } ( 1 , 1 )$ justifies the name $d i f f e r e n c e - i n - d i f f$ erence because it is the difference between two differences in means. This estimator is different from Lin (2013)’s estimator: it fixes $\beta _ { 1 } = \beta _ { 0 } = 1$ in advance while Lin (2013)’s estimator involves two estimated $\beta ^ { \gamma } \mathrm { s } .$ . It is unbiased with a conservative variance estimator

$$
\begin{array}{l} \hat {V} (1, 1) = \frac {1}{n _ {1} (n _ {1} - 1)} \sum_ {i = 1} ^ {n} Z _ {i} \{g _ {i} - \hat {\bar {g}} (1) \} ^ {2} \\ + \frac {1}{n _ {0} (n _ {0} - 1)} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \{g _ {i} - \hat {\bar {g}} (0) \} ^ {2}, \\ \end{array}
$$

where $\hat { \bar { g } } ( 1 )$ and $\hat { \bar { g } } ( 0 )$ are the sample means of the gain score $g _ { i } = Y _ { i } - X _ { i }$ under treatment and control, respectively. When the lagged outcome is a strong predictor of the outcome, the gain score $g _ { i } = Y _ { i } - X _ { i }$ often has much smaller variance than the outcome itself. In this case, $\hat { \tau } ( 1 , 1 )$ often greatly reduces the variance of the simple difference in means of the outcome.

**TABLE 6.2: Design and analysis of experiments**

<table><tr><td></td><td colspan="4">analysis</td></tr><tr><td rowspan="3">design</td><td>CRE</td><td> $\hat{\tau}$  (Neyman, 1923)</td><td> $\stackrel{1}{\longrightarrow}$ </td><td> $\hat{\tau}_{\text{L}}$  (Lin, 2013)</td></tr><tr><td></td><td> $2 \Big\downarrow$ </td><td></td><td> $\Big\downarrow 4$ </td></tr><tr><td>ReM</td><td> $\hat{\tau}$  (Li et al., 2018b)</td><td> $\stackrel{3}{\longrightarrow}$ </td><td> $\hat{\tau}_{\text{L}}$  (Li and Ding, 2020)</td></tr></table>

## 6.2.4 Extension to the SRE

It is possible that we have an experiment stratified on a discrete variable C and observe additional covariates X. If all strata are large, then we can obtain Lin (2013)’s estimators within strata $\hat { \tau } _ { \mathrm { L } , [ k ] }$ and obtain the final estimator as

$$
\hat {\tau} _ {\mathrm{L,S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \hat {\tau} _ {\mathrm{L}, [ k ]}.
$$

A conservative variance estimator is

$$
\hat {V} _ {\mathrm{L,S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \hat {V} _ {\mathrm{EHW}, [ k ]},
$$

where $\hat { V } _ { \mathrm { E H W } , [ k ] }$ is the EHW variance estimator from the OLS fit of the outcome on the treatment indicator, the covariates, and their interactions within stratum k. Importantly, we need to center covariates by their stratum-specific means.

## 6.3 Unification, combination, and comparison

Li and Ding (2020) unified the literature and showed that we can combine rerandomization and regression adjustment. That is, if we rerandomize in the design stage, we can use Lin (2013)’s estimator with the EHW standard error in the analysis stage. The combination of rerandomization and regression adjustment improves covariate balance in the design stage and estimation efficiency in the analysis stage.

Table 6.2 summarizes the literature from Neyman (1923) to Li and Ding (2020). Arrow 1 illustrates the efficiency gain of covariate adjustment in the CRE: asymptotically, $\hat { \tau } _ { \mathrm { L } }$ has smaller variance than ˆτ . Arrow 2 illustrates the efficiency gain of the ReM: asymptotically, ˆτ has narrower quantile range under the ReM than under the CRE. Arrows 3 and 4 illustrate the benefits of the combination.

## 6.4 Simulation

Angrist et al. (2009) conducted an experiment to evaluate different strategies to improve academic performance among college freshmen. Here I use a subset of the original data, focusing on the control group and the treatment group offered academic support services and financial incentives for good grades. The outcome is the GPA at the end of the first year, and two covariates are the gender and baseline GPA. The following table summarizes the results based on the unadjusted and adjusted estimators. The adjusted estimator has smaller standard error although it gives the same insignificant result as the unadjusted estimator.

<table><tr><td></td><td>estimate</td><td>s.e.</td><td>t-stat</td><td>p-value</td></tr><tr><td>Neyman</td><td>0.054</td><td>0.076</td><td>0.719</td><td>0.472</td></tr><tr><td>Lin</td><td>0.075</td><td>0.072</td><td>1.036</td><td>0.300</td></tr></table>

I also use this dataset to conduct simulation studies to evaluate the four design and analysis strategies summarized in Table 6.2. I fit quadratic functions of the outcome on the covariates and use them to impute all the missing potential outcomes, separately for the treated and control groups. To show the improvement of ReM and regression adjustment, I also rescale the error terms by 0.1 and 0.25 to increase the signal to noise ratio. With the imputed Science Table, I generate 2000 treatments, obtain the observed data, and calculate the estimators. In the simulation, the “true” outcome model is nonlinear, but we still use linear adjustment for estimation. By doing this, we can evaluate the properties of the estimators when the linear model is misspecified.

Figure 6.2 shows the violin plots of the four combinations, subtracting the true τ from the estimates. As predicted by the theory, all estimators are nearly unbiased, and both ReM and regression adjustment improve efficiency. They are more effective when the noise level is smaller.

## 6.5 Final remarks

With a continuous outcome, Fisher’s ANCOVA has been the standard approach for many years. Lin (2013)’s improvement has better theoretical properties even if the linear model is misspecified. With a binary outcome, it is common to use the coefficient of the treatment in the logistic regression of the observed outcome on the treatment indicator and covariates to estimate the causal effects However, Freedman (2008c) showed that this logistic regression does not have nice properties under the potential outcomes framework. Even if the logistic model is correct, the coefficient estimates the conditional odds ratio which may not be the parameter of interest; when the logistic model is incorrect, it is even harder to interpret the coefficient. From the discussion above, if the parameter of interest is the average causal effect, we can still use Lin (2013)’s estimator to analyze the binary outcome data in the CRE. Guo and Basse (2023) extend Lin (2013)’s theory to allow for using generalized linear models to construct estimators for the average causal effect under the potential outcomes framework.

Other extensions of Lin (2013)’s theory focus on high dimensional covariates. Bloniarz et al. (2016) focus on the regime with many covariates than the sample size, and under the sparsity assumption, they suggest replacing the OLS fits by the least absolute shrinkage and selection operator (LASSO) fits (Tibshirani, 1996) of the outcome on the treatment, covariates and their interactions. Lei and Ding (2021) focus on the regime with a diverging number of covariates without assuming sparsity, and under certain regularity conditions, they show that Lin (2013)’s estimator is still consistent and asymptotically Normal. Wager et al. (2016) propose to use machine learning methods to analyze high dimensional experimental data.

## 6.6 Homework Problems

## 6.1 FRT under ReM

Describe the FRT under ReM.

## 6.6 Homework Problems

## 6.2 Invariance of the Mahalanobis Distance

Prove Lemma 6.1.

## 6.3 Bias of the difference-in-means estimator under rerandomization

Assume that we draw $\pmb { Z } = ( Z _ { 1 } , \ldots , Z _ { n } )$ from a CRE and accept it if and only if $\phi ( Z , X ) = 1$ , where $\phi$ is a predetermined balance criterion. Show that if $n _ { \mathrm { 1 } } = n _ { \mathrm { 0 } }$ and

$$
\phi (\mathbf {Z}, \mathbf {X}) = \phi (\mathbf {1} _ {n} - \mathbf {Z}, \mathbf {X}), \tag {6.13}
$$

then $\hat { \tau }$ is unbiased for τ . Verify that rerandomization using the Mahalanobis distance satisfies (6.13) if $n _ { \mathrm { 1 } } = n _ { \mathrm { 0 } }$ . Give a counterexample that ˆτ is biased for τ when these two conditions do not hold.

## 6.4 Equivalent form of $R ^ { 2 }$ in the CRE

Prove Proposition 6.1.

## 6.5 Lin’s estimator for covariate adjustment

Prove Proposition 6.2.

## 6.6 Predictive and projective estimators

Prove (6.8) and (6.10).

## 6.7 Equivalent form of the covariate-adjusted estimator

Prove (6.11) and (6.12).

## 6.8 ANCOVA also adjusts for covariate imbalance

This problem gives a result for ANCOVA that is similar to (6.12).

Show that

$$
\hat {\tau} _ {\mathrm{F}} = \hat {\tau} - \hat {\gamma} _ {\mathrm{F}} ^ {\mathsf {T}} \hat {\tau} _ {X},
$$

where $\hat { \gamma } _ { \mathrm { F } }$ is the coefficient of $X _ { i }$ in the OLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } )$ .

## 6.9 Regression adjustment / post-stratification of CRE

Prove Proposition 6.3.

Hint: Sometimes $\hat { \tau } _ { \mathrm { { P S } } }$ or $\hat { \tau } _ { \mathrm { L } }$ may not be well-defined. In those cases, we treat $\hat { \tau } _ { \mathrm { { P S } } }$ and $\hat { \tau } _ { \mathrm { L } }$ as equal. You can ignore this complexity in the proof.

## 6.10 More on the difference-in-difference estimator in the CRE

This problem gives more details for the difference-in-difference estimator in the CRE in Section 6.2.3.3.

Show that $\hat { \tau } ( 1 , 1 )$ is unbiased for τ , calculate its variance, and show that $\hat { V } ( 1 , 1 )$ is a conservative estimator for the true variance of $\hat { \tau } ( 1 , 1 )$ . When does $E \{ \hat { V } ( 1 , 1 ) \} = \operatorname { v a r } \{ \hat { \tau } ( 1 , 1 ) \}$ hold?

Compare the variances of ${ \hat { \tau } } ( 0 , 0 )$ and $\hat { \tau } ( 1 , 1 )$ to show that

$$
\operatorname{var} \{\hat {\tau} (0, 0) \} \geq \operatorname{var} \{\hat {\tau} (1, 1) \}
$$

if and only if

$$
2 \frac {n _ {0}}{n} \beta_ {1} + 2 \frac {n _ {1}}{n} \beta_ {0} \geq 1,
$$

where

$$
\beta_ {1} = \frac {\sum_ {i = 1} ^ {n} (X _ {i} - \bar {X}) \{Y _ {i} (1) - \bar {Y} (1) \}}{\sum_ {i = 1} ^ {n} (X _ {i} - \bar {X}) ^ {2}}, \quad \beta_ {0} = \frac {\sum_ {i = 1} ^ {n} (X _ {i} - \bar {X}) \{Y _ {i} (0) - \bar {Y} (0) \}}{\sum_ {i = 1} ^ {n} (X _ {i} - \bar {X}) ^ {2}}
$$

are the coefficients of $X _ { i }$ in the OLS fits of $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ on $( 1 , X _ { i } )$ , respectively.

Remark: Gerber and Green (2012, page 28) discussed a special case of this problem with $n _ { \mathrm { 1 } } = n _ { \mathrm { 0 } }$ .

## 6.11 Data re-analyses

Re-analyze the data used in SRE Neyman penn.R. The analysis in Chapter 5 uses the treatment indicator, the outcome and the block indicator. Now we want to use all other covariates.

Conduct regression adjustments within strata of the experiment, and then combine these adjusted estimators to estimate the average causal effect. Report the point estimator, estimated standard error and 95% confidence interval. Compare them with those without regression adjustments.

## 6.12 Recommended reading

The title of this chapter is the same as that of Li and Ding (2020), which studied the roles of rerandomization and regression adjustment in the design and analysis stages of randomized experiments, respectively.

## 7

# Matched-Pairs Experiment

The matched-pairs experiment (MPE) is the most extreme version of the SRE with only one treated unit and one control unit within each stratum. In this case, the strata are also called pairs. Although this type of experiment is a special case of the SRE discussed in Chapter 5, it has its own estimation and inference strategy. Moreover, it has many new features and it is closely related to the “matching” strategy in observational studies which will be covered in Chapter 15 later. So we discuss the MPE in this separate chapter.

## 7.1 Design of the experiment and potential outcomes

Consider an experiment with 2n units. If we have predictive covariates to the outcomes, we can pair units based on the similarity of covariates. With a scalar covariate, we can order units based on this covariate and then form pairs based on the adjacent units. With many covariates, we can define pairwise distances between units and then form pairs based on these distances. In this case, pair matching can be done using a greedy algorithm or an optimal nonbipartite matching algorithm. The greedy algorithm pairs the two units with the smallest distance, drop them from the pool of units, pair the two remaining units with the smallest distance, etc. The optimal nonbipartite matching algorithm divides the 2n units into n pairs of two units to minimize the sum of the within-pair distances. See Greevy et al. (2004) for more details of the computational aspect of the MPE. In this chapter, we assume that the pairs are formed based on the covariates, and discuss the subsequent design and analysis issues.

Let (i, j) index the unit $j$ in pair $i ,$ where $i = 1 , \ldots , n$ and $j = 1 , 2$ . Unit (i, j) has potential outcomes $Y _ { i j } ( 1 )$ and $Y _ { i j } ( 0 )$ under the treatment and control, respectively. Within each pair, we randomly assign one unit to receive the treatment and the other to receive the control. Let

$$
Z _ {i} = \left\{ \begin{array}{l l} 1, & \text { if   the   first   unit   receives   the   treatment }, \\ 0, & \text { if   the   second   unit   receives   the   treatment }. \end{array} \right.
$$

We can formally define MPE based on the treatment assignment mechanism.

Definition 7.1 (MPE) We have

$$
(Z _ {i}) _ {i = 1} ^ {n} \stackrel {{I I D}} {{\sim}} \text { Bernoulli } (1 / 2). \tag {7.1}
$$

The observed outcomes within pair i are

$$
Y _ {i 1} = Z _ {i} Y _ {i 1} (1) + (1 - Z _ {i}) Y _ {i 1} (0) = \left\{ \begin{array}{l l} Y _ {i 1} (1), & \text {if} Z _ {i} = 1; \\ Y _ {i 1} (0), & \text {if} Z _ {i} = 0; \end{array} \right.
$$

and

$$
Y _ {i 2} = Z _ {i} Y _ {i 2} (0) + (1 - Z _ {i}) Y _ {i 2} (1) = \left\{ \begin{array}{l l} Y _ {i 2} (0), & \text {if} Z _ {i} = 1; \\ Y _ {i 2} (1), & \text {if} Z _ {i} = 0. \end{array} \right.
$$

So the observed data are $( Z _ { i } , Y _ { i 1 } , Y _ { i 2 } ) _ { i = 1 } ^ { n }$ .

## 7.2 FRT

Similar to the discussion before, we can always use the FRT to test the sharp null hypothesis:

$$
H _ {0 \mathrm{F}}: Y _ {i j} (1) = Y _ {i j} (0) \text {   for   all   } i = 1, \dots n \text {   and   } j = 1, 2.
$$

When conducting the FRT, we need to simulate the distribution of $\left( Z _ { i } , \ldots , Z _ { n } \right)$ from (7.1). I will discuss some canonical choices of test statistics based on the within-pair differences between the treated and control outcomes:

$\begin{array} { r l } { \hat { \tau } _ { i } } & { { } = } \end{array}$ outcome under treatment − outcome under control (within pair i)

$$
= (2 Z _ {i} - 1) \left(Y _ {i 1} - Y _ {i 2}\right)
$$

$$
= S _ {i} (Y _ {i 1} - Y _ {i 2}),
$$

where the $S _ { i } ~ = ~ 2 Z _ { i } - 1$ are IID random signs with mean 0 and variance 1, for $i = 1 , \ldots , n$ . Since the pairs with zero $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ do not contribute to the randomization distribution, we drop those pairs in the discussion of the FRT.

Example 7.1 (paired t statistic) The average of the within-pair differences is

$$
\hat {\tau} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\tau} _ {i}.
$$

Under H0f, $H _ { \mathrm { 0 F } }$

$$
E (\hat {\tau}) = 0
$$

and

$$
\operatorname{var} (\hat {\tau}) = n ^ {- 2} \sum_ {i = 1} ^ {n} \operatorname{var} (\hat {\tau} _ {i}) = n ^ {- 2} \sum_ {i = 1} ^ {n} \operatorname{var} (S _ {i}) (Y _ {i 1} - Y _ {i 2}) ^ {2} = n ^ {- 2} \sum_ {i = 1} ^ {n} \hat {\tau} _ {i} ^ {2}.
$$

Based on the CLT for the sum of independent random variables, we have the Normal approximation:

$$
\frac {\hat {\tau}}{\sqrt {n ^ {- 2} \sum_ {i = 1} ^ {n} \hat {\tau} _ {i} ^ {2}}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

We can use this Normal approximation to construct an asymptotic test. Many standard test books suggest using the following paired t statistic in the $M P E ;$

$$
t _ {p a i r} = \frac {\hat {\tau}}{\sqrt {\{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\hat {\tau} _ {i} - \hat {\tau}) ^ {2}}},
$$

which is almost identical to τˆ with large n and small τˆ under $H _ { \mathrm { 0 F } }$ .

In classic statistics, the motivation for using $t _ { \mathrm { p a i r } }$ is under a different framework. When $\hat { \tau } _ { i } \stackrel { \mathrm { I I D } } { \sim } \mathrm { N } ( 0 , \sigma ^ { 2 } )$ , we can show that $t _ { \mathrm { p a i r } } \sim t ( n - 1 )$ , i.e., the exact distribution of $t _ { \mathrm { p a i r } }$ is t with degrees of freedom $n - 1$ , which is close to $\mathrm { { N } } ( 0 , 1 )$ with a large n. The R function t.test with paired=TRUE can implement this test. With a large n, these procedures give similar results. The discussion in Example 7.1 gives another justification of the classic paired t test without assuming the Normality of the data.

Example 7.2 (Wilcoxon sign-rank statistic) Based on the ranks $( R _ { 1 } , \ldots , R _ { n } )$ of $\big ( \vert \hat { \tau } _ { 1 } \vert , \dots , \vert \hat { \tau } _ { n } \vert \big )$ , we can define a test statistic

$$
W = \sum_ {i = 1} ^ {n} I (\hat {\tau} _ {i} > 0) R _ {i}.
$$

Under $H _ { \mathrm { 0 F } }$ ,

$$
E (W) = \frac {1}{2} \sum_ {i = 1} ^ {n} R _ {i} = \frac {1}{2} \sum_ {i = 1} ^ {n} i = \frac {n (n + 1)}{4}
$$

and

$$
\operatorname{var} (W) = \frac {1}{4} \sum_ {i = 1} ^ {n} R _ {i} ^ {2} = \frac {1}{4} \sum_ {i = 1} ^ {n} i ^ {2} = \frac {n (n + 1) (2 n + 1)}{2 4}.
$$

The CLT for the sum of independent random variables ensures the following Normal approximation:

$$
\frac {W - n (n + 1) / 4}{\sqrt {n (n + 1) (2 n + 1) / 2 4}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

We can use this Normal approximation to construct an asymptotic test. The R function wilcox.test with paired=TRUE can implement these tests.

Example 7.3 (Kolmogorov–Smirnov-type statistic) Under $H _ { \mathrm { 0 F } }$ , the absolute values $\big ( \vert \hat { \tau } _ { 1 } \vert , \dots , \vert \hat { \tau } _ { n } \vert \big )$ are fixed but their signs are random. So $\left( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } \right)$ and $- ( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } )$ should have the same distribution. Let

$$
\hat {F} (t) = n ^ {- 1} \sum_ {i = 1} ^ {n} I (\hat {\tau} _ {i} \leq t)
$$

be the empirical distribution of $\left( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } \right)$ , and

$$
1 - \hat {F} (- t -) = n ^ {- 1} \sum_ {i = 1} ^ {n} I (- \hat {\tau} _ {i} \leq t)
$$

be the empirical distribution $o f - ( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } )$ , where $\hat { F } ( - t - )$ is the left limit of the function $\hat { F } ( \cdot )$ at −t. A Kolmogorov–Smirnov-type statistic is then

$$
D = \max _ {t} | \hat {F} (t) + \hat {F} (- t -) - 1 |.
$$

Butler (1969) proposed this test statistic and derived its exact and asymptotic distributions. Unfortunately, this is not implemented in standard software packages. Nevertheless, we can simulate its exact distribution and compute the p-value based on the FRT. 1

Example 7.4 (sign statistic) The sign statistic uses only the signs of the within-pair differences

$$
\Delta = \sum_ {i = 1} ^ {n} I (\hat {\tau} _ {i} > 0).
$$

Under $H _ { \mathrm { 0 F } }$

$$
I (\hat {\tau} _ {i} > 0) \stackrel {I I D} {\sim} B e r n o u l l i (1 / 2)
$$

and therefore

$$
\Delta \sim B i n o m i a l (n, 1 / 2).
$$

Based on this we have an exact Binomial test, which is implemented in the R function binom.test with $\scriptstyle { p = 1 } / 2 .$ Using the CLT, we can also conduct a test based on the following Normal approximation of the Binomial distribution:

$$
\frac {\Delta - n / 2}{\sqrt {n / 4}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

**TABLE 7.1: Counts of four types of pairs**

<table><tr><td></td><td>control outcome 1</td><td>control outcome 0</td></tr><tr><td>treated outcome 1</td><td> $m_{11}$ </td><td> $m_{10}$ </td></tr><tr><td>treated outcome 0</td><td> $m_{01}$ </td><td> $m_{00}$ </td></tr></table>

Example 7.5 (McNemar’s statistic for a binary outcome) If the outcome is binary, we can summarize the data from the MPE in a more compact way. Given a pair, the treated outcome can be either 1 or 0 and the control outcome can be either 1 or 0, yielding a $2 \times 2$ table as in Table 7.1.

Under $H _ { \mathrm { 0 F } }$ , the numbers of concordant pairs $m _ { 1 1 }$ and m00 are fixed, and $m _ { 1 0 } + m _ { 0 1 }$ is also fixed. So the only random component is $m _ { 1 0 }$ which has distribution

$$
m _ {1 0} \sim B i n o m i a l (m _ {1 0} + m _ {0 1}, 1 / 2).
$$

This implies an exact test based on the Binomial distribution. The R function mcnemar.test gives an asymptotic test based on the Normal approximation of the Binomial distribution:

$$
\frac {m _ {1 0} - (m _ {1 0} + m _ {0 1}) / 2}{\sqrt {(m _ {1 0} + m _ {0 1}) / 4}} = \frac {m _ {1 0} - m _ {0 1}}{\sqrt {m _ {1 0} + m _ {0 1}}} \xrightarrow {\mathrm{d}} \mathrm{N} (0, 1).
$$

Both the exact FRT and the asymptotic test do not depend on $m _ { 1 1 }$ or m00. Only the numbers of discordant pairs matter in these tests.

## 7.3 Neymanian inference

The average causal effect within pair i is

$$
\tau_ {i} = \frac {1}{2} \left\{Y _ {i 1} (1) + Y _ {i 2} (1) - Y _ {i 1} (0) - Y _ {i 2} (0) \right\},
$$

and the average causal effect for all units is

$$
\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \tau_ {i} = (2 n) ^ {- 1} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {2} \left\{Y _ {i j} (1) - Y _ {i j} (0) \right\}.
$$

It is intuitive that $\hat { \tau } _ { i }$ is unbiased for $\tau _ { i } ,$ so $\hat { \tau }$ is unbiased for τ. We can also calculate the variance of ˆτ . I relegate the exact formula to a homework problem because the MPE is just a special case of the SRE.

However, we cannot follow the strategy of a SRE to estimate the variance of ˆτ . The within-pair sample variances of the outcomes are not well defined because within each pair we have only one treated and one control unit. The data do not allow us to estimate the variance of $\hat { \tau } _ { i }$ within pair i.

Is it possible to estimate the variance of $\hat { \tau }$ in the MPE? Let us forget about the MPE and change the perspective to the classic IID sampling. If the $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ $\mu$ $\sigma ^ { 2 }$ ${ \hat { \tau } } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } { \hat { \tau } } _ { i } { \mathrm { ~ i s ~ } } \sigma ^ { 2 } / n$ Pni=1 τˆi is σ2/n. $\sigma ^ { 2 }$ $( n - 1 ) ^ { - 1 } { \dot { \sum _ { i = 1 } ^ { n } ( \hat { \tau } _ { i } - \hat { \tau } ) ^ { 2 } } }$ so an unbiased estimator for var(ˆτ ) is

$$
\hat {V} = \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\hat {\tau} _ {i} - \hat {\tau}) ^ {2}.
$$

The discussion also extends to the independent but not IID setting; see Problem A1.1 in Chapter A1. The above discussion seems a digression from the MPE which has completely different statistical assumptions. But at least it motivates a variance estimator $\hat { V } ,$ which uses the between-pair variance of $\hat { \tau } _ { i }$ to estimate variance of ˆτ . Of course, it is derived under different assumptions. Does it work for the MPE? Theorem 7.1 below is a positive result.

Theorem 7.1 Under the MPE, $\hat { V }$ is a conservative estimator for the true variance $o f { \hat { \tau } }$ :

$$
E (\hat {V}) - \mathrm{var} (\hat {\tau}) = \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\tau_ {i} - \tau) ^ {2} \geq 0.
$$

$I f$ the $\tau _ { i }$ ’s are constant across pairs, then $E ( \hat { V } ) = \operatorname { v a r } ( \hat { \tau } )$ .

Theorem 7.1 states that under the MPE, $\hat { V }$ is a conservative variance estimator in general and becomes unbiased if the average causal effects are constant across pairs. It is somewhat surprising because $\hat { V }$ depends on the between-pair variance of the $\hat { \tau } _ { i } ^ { \mathrm { : } }$ ’s whereas var(ˆτ ) depends on the within-pair variance of each of $\hat { \tau } _ { i }$ . The proof below might provide some insights for this surprisingly result.

Proof of Theorem 7.1: Using the basic algebraic fact that $\scriptstyle \sum _ { i = 1 } ^ { n } ( a _ { i } - { \bar { a } } ) ^ { 2 } =$ $\textstyle \sum _ { i = 1 } ^ { n } a _ { i } ^ { 2 } - n { \bar { a } } ^ { 2 }$ in the following steps 2 and $5 ,$ we have

$$
\begin{array}{l} n (n - 1) E (\hat {V}) = E \left\{\sum_ {i = 1} ^ {n} (\hat {\tau} _ {i} - \hat {\tau}) ^ {2} \right\} \\ = E \left(\sum_ {i = 1} ^ {n} \hat {\tau} _ {i} ^ {2} - n \hat {\tau} ^ {2}\right) \\ = \sum_ {i = 1} ^ {n} \left\{\operatorname{var} \left(\hat {\tau} _ {i}\right) + \tau_ {i} ^ {2} \right\} - n \left\{\operatorname{var} (\hat {\tau}) + \tau^ {2} \right\} \\ = \sum_ {i = 1} ^ {n} \operatorname{var} (\hat {\tau} _ {i}) - n \operatorname{var} (\hat {\tau}) + \sum_ {i = 1} ^ {n} \tau_ {i} ^ {2} - n \tau^ {2} \\ = n ^ {2} \mathrm{var} (\hat {\tau}) - n \mathrm{var} (\hat {\tau}) + \sum_ {i = 1} ^ {n} (\tau_ {i} - \tau) ^ {2}. \\ \end{array}
$$

Therefore,

$$
E (\hat {V}) = \operatorname{var} (\hat {\tau}) + \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\tau_ {i} - \tau) ^ {2} \geq \operatorname{var} (\hat {\tau}).
$$

![image_07](images/image_07.png)

Similar to the discussions for other experiments, the Neymanian approach relies on the large-sample approximation:

$$
\frac {\hat {\tau} - \tau}{\sqrt {\operatorname{var} (\hat {\tau})}} \to \mathrm{N} (0, 1)
$$

in distribution if n → ∞ and some regularity conditions hold. Due to the over estimation of the variance, the Wald-type confidence interval

$$
\hat {\tau} \pm z _ {1 - \alpha / 2} \sqrt {\hat {V}}
$$

covers τ with probability at least $1 - \alpha$ .

Both the point estimator ˆτ and the variance estimator $\hat { V }$ can be conveniently obtained by OLS, as shown in the proposition below.

Proposition $\mathbf { 7 . 1 } ~ \widehat { \tau }$ and $\hat { V }$ are identical to the coefficient and variance estimator of the intercept from the OLS fit of the vector $( \widehat { \tau } _ { 1 } , \ldots , \widehat { \tau } _ { n } ) ^ { \mathsf { T } }$ on the intercept only.

I leave the proof of Proposition 7.1 as Problem 7.3.

## 7.4 Covariate adjustment

## 7.4.1 FRT

Similar to the discussion in the CRE, there are two general strategies of covariate adjustment in the MPE. First, we can construct test statistics based on the residuals from a model fitting of the outcome on the covariates, since those residuals are fixed numbers under the sharp null hypothesis. A canonical choice is to fit OLS of all observed $Y _ { i j }$ ’s on $X _ { i j } \mathrm { ^ { , } s }$ to obtain the residuals $\hat { \varepsilon } _ { i j } \mathrm { ' s }$ . We can then construct test statistics pretending that the $\hat { \varepsilon } _ { i j } \ ' _ { \ell }$ are the observed outcomes. Rosenbaum (2002a) advocated this strategy in particular to the MPE.

Second, we can directly use some coefficients from model fitting as the test statistics. The discussion in the next subsection will suggest a choice of the test statistic for the second strategy.

## 7.4.2 Regression adjustment

Although we have matched on covariates in the design stage, it is possible that the matching is not perfect and sometimes we have additional covariates beyond those used in the pair-matching stage. In those cases, we can adjust for the covariates to further improve estimation efficiency. Assume that each unit has covariates $X _ { i j }$ , and we can compute the within-pair differences in covariates $\widehat { \tau } _ { X , i }$ and their average $\hat { \tau } _ { X }$ in the same way as the outcome. We can show that

$$
E (\hat {\tau} _ {X, i}) = 0, \quad E (\hat {\tau} _ {X}) = 0,
$$

and

$$
\operatorname{cov} (\hat {\tau} _ {X}) = n ^ {- 2} \sum_ {i = 1} ^ {n} \hat {\tau} _ {X, i} \hat {\tau} _ {X, i} ^ {\mathsf {T}}.
$$

In a realized MPE, cov $\left( \hat { \tau } _ { X } \right)$ is not zero unless all the ${ \hat { \tau } } _ { X , i } { \mathrm { ' s } }$ are zero. With an unlucky draw of $\left( Z _ { 1 } , \ldots , Z _ { n } \right)$ , it is possible that $\hat { \tau } _ { X }$ differs substantially from zero. Similar to the discussion in the CRE, adjusting for the imbalance of the covariate means is likely to improve estimation efficiency.

Consider a class of estimators indexed by $\gamma \colon$

$$
\hat {\tau} (\gamma) = \hat {\tau} - \gamma^ {\mathsf {T}} \hat {\tau} _ {X}
$$

which has mean 0 for any fixed $\gamma .$ We want to choose $\gamma$ to minimize the variance of $\hat { \tau } ( \gamma )$ . Its variance is a quadratic function of $\gamma \colon$

$$
\mathrm{var} \{\hat {\tau} (\gamma) \} = \mathrm{var} (\hat {\tau} - \gamma^ {\mathsf {T}} \hat {\tau} _ {X}) = \mathrm{var} (\hat {\tau}) + \gamma^ {\mathsf {T}} \mathrm{cov} (\hat {\tau} _ {X}) \gamma - 2 \gamma^ {\mathsf {T}} \mathrm{cov} (\hat {\tau} _ {X}, \hat {\tau}),
$$

which is minimized at

$$
\tilde {\gamma} = \mathrm{cov} (\hat {\tau} _ {X}) ^ {- 1} \mathrm{cov} (\hat {\tau} _ {X}, \hat {\tau}).
$$

We have obtained the formula of $\operatorname { c o v } ( { \hat { \tau } } _ { X } )$ in the above, which can also be written as

$$
\operatorname{cov} (\hat {\tau} _ {X}) = n ^ {- 2} \sum_ {i = 1} ^ {n} | \hat {\tau} _ {X, i} | | \hat {\tau} _ {X, i} | ^ {\mathsf {T}},
$$

where $\left. \cdot \right.$ denotes component-wise absolute value of a vector. $\operatorname { S o c o v } ( \hat { \tau } _ { X } )$ is fixed and known from the observed data. However, $\operatorname { c o v } \big ( \hat { \tau } _ { X } , \hat { \tau } \big )$ depends on unknown potential outcomes. Fortunately, we can obtain an unbiased estimator for it, as shown in Theorem 7.2 below.

Theorem 7.2 An unbiased estimator for cov $( \hat { \tau } _ { X } , \hat { \tau } )$ is

$$
\hat {\theta} = \{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\hat {\tau} _ {X, i} - \hat {\tau} _ {X}) (\hat {\tau} _ {i} - \hat {\tau}).
$$

The proof of Theorem 7.2 is similar to that of Theorem 7.1. I leave it to Problem 7.2.

Therefore, we can estimate the optimal coefficient $\tilde { \gamma }$ by

$$
\begin{array}{l} \hat {\gamma} = \left(n ^ {- 2} \sum_ {i = 1} ^ {n} \hat {\tau} _ {X, i} \hat {\tau} _ {X, i} ^ {\mathsf {T}}\right) ^ {- 1} \left\{\{n (n - 1) \} ^ {- 1} \sum_ {i = 1} ^ {n} (\hat {\tau} _ {X, i} - \hat {\tau} _ {X}) (\hat {\tau} _ {i} - \hat {\tau}) \right\} \\ \approx \left(\sum_ {i = 1} ^ {n} (\hat {\tau} _ {X, i} - \hat {\tau} _ {X}) (\hat {\tau} _ {X, i} - \hat {\tau} _ {X}) ^ {\mathsf {T}}\right) ^ {- 1} \sum_ {i = 1} ^ {n} (\hat {\tau} _ {X, i} - \hat {\tau} _ {X}) (\hat {\tau} _ {i} - \hat {\tau}), \\ \end{array}
$$

which is approximately the coefficient of the $\widehat { \tau } _ { X , i }$ in the OLS fit of the $\hat { \tau } _ { i }$ ’s on the ${ \hat { \tau } } _ { X , i } { \mathrm { ' s } }$ with an intercept. The final estimator is

$$
\hat {\tau} _ {\mathrm{adj}} = \hat {\tau} (\hat {\gamma}) = \hat {\tau} - \hat {\gamma} ^ {\mathsf {T}} \hat {\tau} _ {X},
$$

which, by the property of OLS, is approximately the intercept in the OLS fit of the $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ on the ${ \hat { \tau } } _ { X , i } { \mathrm { ' s } }$ with an intercept.

A conservative variance estimator for $\hat { \tau } _ { \mathrm { a d j } }$ is then

$$
\hat {V} _ {\mathrm{adj}} = \hat {V} + \hat {\gamma} ^ {\mathsf {T}} \mathrm{cov} (\hat {\tau} _ {X}) \hat {\gamma} - 2 \hat {\gamma} ^ {\mathsf {T}} \hat {\theta} = \hat {V} - \hat {\theta} ^ {\mathsf {T}} \mathrm{cov} (\hat {\tau} _ {X}) ^ {- 1} \hat {\theta}.
$$

A subtle technical issue is whether $\hat { \tau } ( \hat { \gamma } )$ has the same optimality as $\hat { \tau } ( \tilde { \gamma } )$ . With large samples, we can show $\hat { \tau } ( \hat { \gamma } ) - \hat { \tau } ( \hat { \gamma } ) = - ( \hat { \gamma } - \tilde { \gamma } ) ^ { \top } \hat { \tau } _ { X }$ is of higher order since it is the product of two “small” terms $\hat { \gamma } - \tilde { \gamma }$ and $\hat { \tau } _ { X }$ . I omit the tedious details for asymptotic analysis, but hope the result makes some intuitive sense to the readers.

Moreover, Fogarty (2018b) discussed the asymptotically equivalent regression formulation of the above covariate-adjusted procedure, and gave a rigorous proof for associated CLT. I summarize the regression formulation below without giving the regularity conditions.

Proposition 7.2 Under the MPE, the covariate-adjusted estimator $\hat { \tau } _ { a d j }$ and the associated variance estimator $\hat { V } _ { a d j }$ can be conveniently approximated by the intercept and the associated variance estimator from the OLS fit of the vector of the $\hat { \tau } _ { i }$ ’s on the 1’s and the matrix of the $\hat { \tau } _ { X , i } \mathit { \Omega } ^ { \prime } s .$ .

I leave the proof of Proposition 7.2 as Problem 7.3. Interestingly, neither Proposition 7.1 nor 7.2 requires the EHW correction of the variance estimator. Because we reduce the data from the MRE to the within-pair differences, it is unnecessary to center the covariates unlike in Lin (2013)’s estimator for the CRE.

## 7.5 Examples

## 7.5.1 Darwin’s data comparing cross-fertilizing and selffertilizing on the height of corns

This is a classical example from Fisher (1935). It contains 15 pairs of corns with either cross-fertilizing or self-fertilizing, with the height being the outcome. The R package HistData provides the original data, where cross and self are the heights under cross-fertilizing and self-fertilizing, respectively, and diff denotes their difference.

<table><tr><td colspan="6">&gt; library(&quot;HistData&quot;)</td></tr><tr><td colspan="6">&gt; ZeaMays</td></tr><tr><td></td><td>pair</td><td>pot</td><td>cross</td><td>self</td><td>diff</td></tr><tr><td>1</td><td>1</td><td>1</td><td>23.500</td><td>17.375</td><td>6.125</td></tr><tr><td>2</td><td>2</td><td>1</td><td>12.000</td><td>20.375</td><td>-8.375</td></tr><tr><td>3</td><td>3</td><td>1</td><td>21.000</td><td>20.000</td><td>1.000</td></tr><tr><td>4</td><td>4</td><td>2</td><td>22.000</td><td>20.000</td><td>2.000</td></tr><tr><td>5</td><td>5</td><td>2</td><td>19.125</td><td>18.375</td><td>0.750</td></tr><tr><td>6</td><td>6</td><td>2</td><td>21.500</td><td>18.625</td><td>2.875</td></tr><tr><td>7</td><td>7</td><td>3</td><td>22.125</td><td>18.625</td><td>3.500</td></tr><tr><td>8</td><td>8</td><td>3</td><td>20.375</td><td>15.250</td><td>5.125</td></tr><tr><td>9</td><td>9</td><td>3</td><td>18.250</td><td>16.500</td><td>1.750</td></tr><tr><td>10</td><td>10</td><td>3</td><td>21.625</td><td>18.000</td><td>3.625</td></tr><tr><td>11</td><td>11</td><td>3</td><td>23.250</td><td>16.250</td><td>7.000</td></tr><tr><td>12</td><td>12</td><td>4</td><td>21.000</td><td>18.000</td><td>3.000</td></tr><tr><td>13</td><td>13</td><td>4</td><td>22.125</td><td>12.750</td><td>9.375</td></tr><tr><td>14</td><td>14</td><td>4</td><td>23.000</td><td>15.500</td><td>7.500</td></tr><tr><td>15</td><td>15</td><td>4</td><td>12.000</td><td>18.000</td><td>-6.000</td></tr></table>

In total, the MPE has $2 ^ { 1 5 } = 3 2 7 6 8$ possible treatment assignment which is a tractable number in R. The following function can enumerate all possible treatment assignment for the MPE:

```txt
MP_enumerate = function(i, n.pairs)
{
    if (i > 2^n.pairs) print("i is too large.")
    a = 2^(n.pairs - 1):0)
    b = 2*a
    2*sapply(i - 1,
    function(x)
    as.integer((x %% b) >= a)) - 1
}
```

So we enumerate all the treatment assignments, and calculate the corresponding ˆτ ’s and the one-sided exact p-value.

```txt
> difference = ZeaMays$diff
> n.pairs = length(difference)
```

## 7.5 Examples

Figure 7.1 shows the exact randomization of ˆτ .

```diff
> abs.diff = abs(difference)
> t.obs = mean(difference)
> t.ran = sapply(1:2^15,
+ function(x){
+ sum(MP_enumerate(x, 15)*abs.diff)
+ })/n.pairs
> pvalue = mean(t.ran>=t.obs)
> pvalue
[1] 0.02633667
```

## 7.5.2 Children’s television workshop experiment data

I also re-analyze the data from from Ball et al. (1973) which was also analyzed by Imbens and Rubin (2015). It contains 8 pairs, and the following table summarizes the within-pair covariate and outcome, as well as their differences:

```txt
> dataxy
x.control x.treatment y.control y.treatment diffx diffy
1 12.9 12.0 54.6 60.6 -0.9 6.0
2 15.1 12.3 56.5 55.5 -2.8 -1.0
3 16.8 17.2 75.2 84.8 0.4 9.6
4 15.8 18.9 75.6 101.9 3.1 26.3
5 13.9 15.3 55.3 70.6 1.4 15.3
6 14.5 16.6 59.3 78.4 2.1 19.1
```

<table><tr><td>7</td><td>17.0</td><td>16.0</td><td>87.0</td><td>84.2</td><td>-1.0</td><td>-2.8</td></tr><tr><td>8</td><td>15.8</td><td>20.1</td><td>73.7</td><td>108.6</td><td>4.3</td><td>34.9</td></tr></table>

We can use the OLS to obtain the point estimators and standard errors: without adjusting for covariates, we have

```txt
> unadj = summary(lm(diffy ~ 1, data = dataxy))$coef
> round(unadj, 3)
Estimate Std. Error t value Pr(>|t|)
(Intercept) 13.425 4.636 2.896 0.023
```

with adjusting for covariates, we have

```txt
> adj = summary(lm(diffy ~ diffx, data = dataxy))$coef
> round(adj, 3)
Estimate Std. Error t value Pr(>|t|)
(Intercept) 8.994 1.410 6.381 0.001
diffx 5.371 0.599 8.964 0.000
```

The above results assume large n, and p-values are justified if we believe the large-n approximation. However, $n = 8$ is not large. In total, we have $2 ^ { 8 } = 2 5 6$ possible treatment assignments, so the smallest possible p-value is $1 / 2 5 6 = 0 . 0 0 3 9$ , which is much larger than the p-value based on the Normal approximation of the covariate-adjusted estimator. In this example, it will be more reasonable to use the FRT with the studentized statistic (i. e., the t value from the lm function) to calculate exact p-values. Figure 7.2 shows the exact distributions of the two studentized statistic, as well as the two-sided p-values. The figure highlights the fact that the randomization distribution of the test statistics are discrete, taking at most 256 possible values. The Normal approximations are unlikely to be accurate especially at the tails. We should report the p-values based on the FRT.

## 7.6 Comparing the MPE and CRE

Imai (2008b) compared the MPE and CRE. Heuristically, the conclusion is that the MPE gives more precise estimators if the matching is well done and the covariates are predictive to the outcome. However, without the outcome data in the design stage, it is hard to decide whether this holds. In the FRT, if covariates are predictive to the outcome, the MPE usually gives more powerful tests compared to the CRE. Greevy et al. (2004) illustrated this using simulation based on the Wilcoxon sign rank statistic. However, this can be a subtle issue with finite samples. Consider an experiment with $2 n$ units, with n units receiving the treatment and n units receiving the control. If we test the sharp null hypothesis at level 0.05, then in the MPE, we need at least $2 \times 5 = 1 0$ units since the smallest p-value is $1 / 2 ^ { 5 } = 1 / 3 2 < 0 . 0 5$ but $1 / 2 ^ { 4 } = 1 / 1 6 > 0 . 0 5$ , but in the CRE, we need at least $2 \times 4 = 8$ units since the smallest p-value is$1 / \binom { 8 } { 4 } = 1 / 7 0 < 0 . 0 5$ but $1 / \binom { 6 } { 3 } = 1 / 2 0 = 0 . 0 5$ . So with 8 units, it is impossible to reject the sharp null hypothesis in the MPE but it is possible in the CRE. Even if the covariates are perfect predictors of the outcome, the MPE is not superior to the CRE based on the FRT.

## 7.7 Extension to the general matched experiment

It is straightforward to extend the MPE to the general matched experiment with varying numbers of control units. Assume that we have n matched sets indexed by $i = 1 , \ldots , n$ . For matched set i, we have $1 + M _ { i }$ units. The $M _ { i } { ^ \mathrm { { s } } }$ can vary. The total number of experimental units is $\begin{array} { r } { N = n + \sum _ { i = 1 } ^ { n } M _ { i } } \end{array}$ . Let $i j$ index the unit $j$ within matched set i $( i = 1 , \ldots , n$ and $j = 1 , \ldots , M _ { i } + 1 )$ . Unit ij has potential outcomes $Y _ { i j } ( 1 )$ and $Y _ { i j } ( 0 )$ under the treatment and control, respectively.

Within matched set $i \ ( i = 1 , \ldots , n )$ , the experimenter randomly selects exactly one unit to receive the treatment with the rest $M _ { i }$ units receiving the control. This general matched experiment is also a special case of the SRE with n strata of size $1 + M _ { i } ( i = 1 , \dots , n )$ . Let $Z _ { i j }$ be the treatment indicator for unit $i j$ , which reveals one of the potential outcomes as

$$
Y _ {i j} = Z _ {i j} Y _ {i j} (1) + (1 - Z _ {i j}) Y _ {i j} (0).
$$

The average causal effect within matched set i equals

$$
\tau_ {i} = (M _ {i} + 1) ^ {- 1} \sum_ {j = 1} ^ {1 + M _ {i}} \{Y _ {i j} (1) - Y _ {i j} (0) \}.
$$

Since it is a SRE, an unbiased estimator of $\tau _ { i }$ is

$$
\hat {\tau} _ {i} = \sum_ {j = 1} ^ {M _ {i} + 1} Z _ {i j} Y _ {i j} - M _ {i} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i j}) Y _ {i j}
$$

which is the difference in means of the outcomes within matched set i.

Below we discuss the statistical inference with the general matched experiment.

## 7.7.1 FRT

As usual, we can always use the FRT to test the sharp null hypothesis

$$
H _ {0 \mathrm{F}}: Y _ {i j} (1) = Y _ {i j} (0) \text {   for   all   } i = 1, \dots , n; j = 1, \dots , M _ {i} + 1.
$$

Because the general matched experiment is a special case of the SRE with many small strata, we can use the test statistics defined in Examples 5.4, 5.5, 7.2, 7.3, 7.4, as well as the estimators and the corresponding t-statistics from the following two subsections.

## 7.7.2 Estimating the average of the within-strata effects

We first focus on estimating the average of the within-strata effects:

$$
\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \tau_ {i}.
$$

It has an unbiased estimator

$$
\hat {\tau} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\tau} _ {i}.
$$

Interestingly, we can show that Theorem 7.1 holds for the general matched experiment, so are other results for the MPE. In particular, we can use the OLS fit of the $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ on the intercept to obtain the point and variance estimators for τ . With covariates, we can use the OLS fit of the $\hat { \tau } _ { i } ^ { \phantom { \dagger } } \rangle$ s on the intercept and the ${ \hat { \tau } } _ { X , i } { \mathrm { ' s } } ,$ , where

$$
\hat {\tau} _ {X, i} = \sum_ {j = 1} ^ {M _ {i} + 1} Z _ {i j} X _ {i j} - M _ {i} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i j}) X _ {i j}
$$

is the corresponding difference in means of the covariates within matched set i.

## 7.7.3 A more general causal estimand

Importantly, the τ above is the average of the $\tau _ { i } ^ { \ , } \mathrm { s } ,$ which does not equal the average causal effect for the N units in the experiment when the $M _ { i } { ^ \mathrm { { \tiny ~ s } } }$ vary. The average causal effect equals

$$
\tau^ {\prime} = N ^ {- 1} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {1 + M _ {i}} \left\{Y _ {i j} (1) - Y _ {i j} (0) \right\} = \sum_ {i = 1} ^ {n} \frac {1 + M _ {i}}{N} \tau_ {i}.
$$

To unify the discussion, I consider the weighted causal effect

$$
\tau_ {w} = \sum_ {i = 1} ^ {n} w _ {i} \tau_ {i}
$$

$\textstyle \sum _ { i = 1 } ^ { n } w _ { i } = 1$ $w _ { i } = n ^ { - 1 }$ $\tau ^ { \prime }$ a special case with $w _ { i } = ( 1 + M _ { i } ) / N$ for $i = 1 , \ldots , n$ . It is straightforward to obtain an unbiased estimator

$$
\hat {\tau} _ {w} = \sum_ {i = 1} ^ {n} w _ {i} \hat {\tau} _ {i},
$$

and calculate its variance

$$
\operatorname{var} (\hat {\tau} _ {w}) = \sum_ {i = 1} ^ {n} w _ {i} ^ {2} \operatorname{var} (\hat {\tau} _ {i}).
$$

However, estimating the variance of this estimator is quite tricky because the $\hat { \tau } _ { i } ^ { \phantom { } } \mathrm { { s } }$ are independent random variable without any replicates. This is a famous problem in theoretical statistics studied by Hartley et al. (1969) and Rao (1970). Fogarty (2018a) also discussed this problem without recognizing these previous works. I will give the final form of the variance estimator without detailing the motivation:

$$
\hat {V} _ {w} = \sum_ {i = 1} ^ {n} c _ {i} (\hat {\tau} _ {i} - \hat {\tau} _ {w}) ^ {2}
$$

where

$$
c _ {i} = \frac {\frac {w _ {i} ^ {2}}{1 - 2 w _ {i}}}{1 + \sum_ {i = 1} ^ {n} \frac {w _ {i} ^ {2}}{1 - 2 w _ {i}}}.
$$

As a sanity check, $c _ { i }$ reduces to $\{ n ( n - 1 ) \} ^ { - 1 }$ in the MPE with $M _ { i } = 1$ and $w _ { i } = n ^ { - 1 }$ . For simplicity, we focus on the case with $w _ { i } < 1 / 2$ for all $i \mathrm { \ ' } _ { \mathrm { S } } .$ , that is, there is no matched set containing more than half of the total weights. The following theorem extends Theorem 7.1.

Theorem 7.3 Under the general matched experiment with varying $M _ { i }$ s, we have

$$
E (\hat {V} _ {w}) - \mathrm{var} (\hat {\tau} _ {w}) = \sum_ {i = 1} ^ {n} c _ {i} (\tau_ {i} - \tau_ {w}) ^ {2} \geq \mathrm{var} (\hat {\tau} _ {w}) \geq 0
$$

with equality holding if the $\tau _ { i }$ ’s are constant.

Although the theoretical motivation for $\hat { V } _ { w }$ is quite complicated, it is not too difficult to verify Theorem 7.3 directly. I relegate the proof to Problem 7.9.

## 7.8 Homework Problems

## 7.1 The true variance of τˆ in the MPE

Express var(ˆτ ) in terms of the first two finite-population moments potential outcomes.

## 7.2 A covariance estimator

Prove Theorem 7.2.

## 7.3 Variance estimators via OLS

Prove Propositions 7.1 and 7.2.

## 7.4 Point and variance estimator with binary outcome

This problem extends Example 7.5 to Neymanian inference.

Express ˆτ and $\hat { V }$ in terms of the counts in Table 7.1.

## 7.5 Minimum sample size for the FRT

Extend the discussion in Section 7.6. Consider an experiment with 2n units, with n units receiving the treatment and n units receiving the control, and test the sharp null hypothesis at level 0.001. What is the minimum value of n for an MPE so that the smallest p-value does not exceed than 0.001, and what is the correponding minimum value of n for a CRE.

## 7.6 Re-analyzing Darwin’s data

In MPEFRTdarwin.R, I analyze Darwin’s data using the FRT based on the test statistic ˆτ .

Re-analyze this dataset using the FRT with the Wilcoxon signed rank sum statistic.

Re-analyze this dataset based on the Neymanian inference: unbiased point estimator, conservative variance estimator, 95% confidence interval.

## 7.7 Re-analyzing children’s television workshop experiment data

In MPENeymanstar.R, I analyze the data from based on Neymanian inference.

Re-analyze this dataset using the FRT with different test statistics.

Re-analyze this dataset using the FRT with covariate adjustment, e.g., you can define test statistics based on residuals from the OLS fit of the observed outcome on covariates. Will the conclusion change if you do not include an intercept in your OLS fit?

## 7.8 Re-analyzing Angrist and Lavy (2009)’s data

The original analysis was quite complicated. For this problem, please focus only on Table A1 of the original paper viewing the schools as experimental units. Angrist and Lavy (2009) essentially conducted an MPE on the schools. Dropping pair 6 and all the pairs with noncompliance results in 14 complete pairs, with data shown below and also in AL2009.csv:

<table><tr><td></td><td>pair</td><td>z</td><td>pr99</td><td>pr00</td><td>pr01</td><td>pr02</td></tr><tr><td>1</td><td>1</td><td>0</td><td>0.046</td><td>0.000</td><td>0.091</td><td>0.185</td></tr><tr><td>2</td><td>1</td><td>1</td><td>0.036</td><td>0.051</td><td>0.000</td><td>0.047</td></tr><tr><td>3</td><td>2</td><td>0</td><td>0.054</td><td>0.094</td><td>0.184</td><td>0.034</td></tr><tr><td>4</td><td>2</td><td>1</td><td>0.050</td><td>0.108</td><td>0.110</td><td>0.095</td></tr><tr><td>5</td><td>3</td><td>0</td><td>0.114</td><td>0.000</td><td>0.056</td><td>0.075</td></tr><tr><td>6</td><td>3</td><td>1</td><td>0.098</td><td>0.054</td><td>0.030</td><td>0.068</td></tr><tr><td>7</td><td>4</td><td>0</td><td>0.148</td><td>0.162</td><td>0.082</td><td>0.075</td></tr><tr><td>8</td><td>4</td><td>1</td><td>0.134</td><td>0.390</td><td>0.339</td><td>0.458</td></tr><tr><td>9</td><td>5</td><td>0</td><td>0.152</td><td>0.105</td><td>0.083</td><td>0.129</td></tr><tr><td>10</td><td>5</td><td>1</td><td>0.145</td><td>0.077</td><td>0.579</td><td>0.167</td></tr><tr><td>11</td><td>6</td><td>0</td><td>0.188</td><td>0.214</td><td>0.375</td><td>0.545</td></tr><tr><td>12</td><td>6</td><td>1</td><td>0.179</td><td>0.165</td><td>0.483</td><td>0.444</td></tr><tr><td>13</td><td>7</td><td>0</td><td>0.193</td><td>0.771</td><td>0.328</td><td>0.583</td></tr><tr><td>14</td><td>7</td><td>1</td><td>0.189</td><td>0.186</td><td>0.168</td><td>0.368</td></tr><tr><td>15</td><td>8</td><td>0</td><td>0.197</td><td>0.350</td><td>0.000</td><td>0.383</td></tr><tr><td>16</td><td>8</td><td>1</td><td>0.200</td><td>0.071</td><td>0.667</td><td>0.429</td></tr><tr><td>17</td><td>9</td><td>0</td><td>0.213</td><td>0.176</td><td>0.164</td><td>0.172</td></tr><tr><td>18</td><td>9</td><td>1</td><td>0.209</td><td>0.165</td><td>0.092</td><td>0.151</td></tr><tr><td>19</td><td>10</td><td>0</td><td>0.211</td><td>0.667</td><td>0.250</td><td>0.617</td></tr><tr><td>20</td><td>10</td><td>1</td><td>0.219</td><td>0.250</td><td>0.500</td><td>0.350</td></tr><tr><td>21</td><td>11</td><td>0</td><td>0.219</td><td>0.153</td><td>0.185</td><td>0.219</td></tr><tr><td>22</td><td>11</td><td>1</td><td>0.224</td><td>0.363</td><td>0.372</td><td>0.342</td></tr><tr><td>23</td><td>12</td><td>0</td><td>0.255</td><td>0.226</td><td>0.213</td><td>0.327</td></tr><tr><td>24</td><td>12</td><td>1</td><td>0.257</td><td>0.098</td><td>0.107</td><td>0.095</td></tr><tr><td>25</td><td>13</td><td>0</td><td>0.261</td><td>0.071</td><td>0.000</td><td>NA</td></tr><tr><td>26</td><td>13</td><td>1</td><td>0.263</td><td>0.441</td><td>0.448</td><td>0.435</td></tr><tr><td>27</td><td>14</td><td>0</td><td>0.286</td><td>0.161</td><td>0.126</td><td>0.181</td></tr><tr><td>28</td><td>14</td><td>1</td><td>0.285</td><td>0.389</td><td>0.353</td><td>0.309</td></tr></table>

The outcomes are the Bagrut passing rates in years 2001 and 2002, with the Bagrut passing rates in 1999 and 2000 as pretreatment covariates. Re-analyze the data based on the Neymanian inference with and without covariates. In particular, how do you deal with the missing outcome in pair 25?

## 7.9 Variance estimation in the general matched experiment

This problem contains more details for Section 7.7.

First, prove Theorem 7.1 for the general matched experiment.

Second, prove Theorem 7.3.

Hint: For the second part, we need to first verify that $\hat { \tau } _ { i } - \hat { \tau } _ { w }$ has mean $\tau _ { i } - \tau _ { w }$ and variance

$$
\operatorname{var} \left(\hat {\tau} _ {i} - \hat {\tau} _ {w}\right) = \operatorname{var} \left(\hat {\tau} _ {w}\right) + (1 - 2 w _ {i}) \operatorname{var} \left(\hat {\tau} _ {i}\right).
$$

## 7.10 Recommended readings

Greevy et al. (2004) provided an algorithm to form matched pairs based on covariates. Imai (2008b) discussed estimation of the average causal effect without covariates, and Fogarty (2018b) discussed covariate adjustment in MPEs.

## 8

# Unification of the Fisherian and Neymanian Inferences in Randomized Experiments

Previous chapters cover both the Fisherian and Neymanian inferences for different types of experiments. The Fisherian perspective focuses on the finitesample exact p-value for testing the strong null hypothesis of no causal effects for any units whatsoever, and the Neymanian perspective focuses on unbiased estimation with a conservative large-sample confidence interval for the average causal effect. Both of them are justified by the physical randomization of the experiments. They are the two important forms of design-based or randomization-based inference for causal effects. They are related but also have distinct features.

In 1935, Neyman presented his seminal paper on randomization-based inference to the Royal Statistical Society. His paper (Neyman, 1935) was attacked by Fisher in the discussion session. Sabbaghi and Rubin (2014) reviewed this famous Neyman–Fisher controversy and presented some new results for this old problem. Instead of going to philosophical issues, this chapter provides a unified discussion.

## 8.1 Testing strong and weak null hypotheses in the CRE

Let us revisit the treatment-control CRE. The Fisherian perspective focuses on testing the strong null hypothesis

$$
H _ {0 \mathrm{F}}: Y _ {i} (1) = Y _ {i} (0) \text {   for   all   units   } i = 1, \dots , n.
$$

The FRT delivers a finite-sample exact pfrt.

By duality of the confidence interval and hypothesis testing, the Neymanian perspective gives a test for the weak null hypothesis

$$
H _ {0 \mathrm{N}}: \tau = 0 \Longleftrightarrow H _ {0 \mathrm{N}}: \bar {Y} (1) = \bar {Y} (0)
$$

based on

$$
t = \frac {\hat {\tau}}{\sqrt {\hat {V}}} = \sqrt {\frac {\operatorname{var} (\hat {\tau})}{\hat {V}}} \times \frac {\hat {\tau}}{\sqrt {\operatorname{var} (\hat {\tau})}} \xrightarrow {\mathrm{d}} C \times \mathrm{N} (0, 1),
$$

with $C \leq 1$ . Using $\mathrm { { N } } ( 0 , 1 )$ quantiles for the studentized statistic $t ,$ we have a conservative large-sample test for $H _ { \mathrm { 0 N } }$ .

Furthermore, Ding and Dasgupta (2017) show that the FRT with the studentized statistic t has the dual guarantees:

1. the associate $p _ { \mathrm { F R T } }$ is finite-sample exact under $H _ { \mathrm { 0 F } }$ ;  
2. it is asymptotically conservative under $H _ { \mathrm { 0 N } }$

Importantly, this is a feature of the studentized statistic t. Ding and Dasgupta (2017) showed that the FRT with other test statistics may not have the dual guarantee. In particular, the FRT with $\hat { \tau }$ may be asymptotically anti-conservative under $H _ { \mathrm { 0 N } }$ . I give some heuristics below to illustrate the importance of studentization in the FRT.

Under $H _ { \mathrm { 0 N } }$ , we have

$$
\hat {\tau} \dot {\sim} \mathrm{N} \left(0, \frac {S ^ {2} (1)}{n _ {1}} + \frac {S ^ {2} (0)}{n _ {0}} - \frac {S ^ {2} (\tau)}{n}\right).
$$

The FRT pretends that the Science Table is $( Y _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ , so the permutation distribution of $\hat { \tau }$ is

$$
(\hat {\tau}) ^ {\pi} \dot {\sim} \mathrm{N} \left(0, \frac {s ^ {2}}{n _ {1}} + \frac {s ^ {2}}{n _ {0}}\right),
$$

where $( \cdot ) ^ { \pi }$ denotes the permutation distribution and $s ^ { 2 }$ is the sample variance of the observed outcomes. Based on $( 3 . 7 )$ in Chapter $s ,$ we can approximate the asymptotic variance of $( \hat { \tau } ) ^ { \pi }$ under $H _ { \mathrm { 0 F } }$ as

$$
\begin{array}{l} \frac {s ^ {2}}{n _ {1}} + \frac {s ^ {2}}{n _ {0}} = \frac {n}{n _ {1} n _ {0}} \left\{\frac {n _ {1} - 1}{n - 1} \hat {S} ^ {2} (1) + \frac {n _ {0} - 1}{n - 1} \hat {S} ^ {2} (0) + \frac {n _ {1} n _ {0}}{n (n - 1)} \hat {\tau} ^ {2} \right\} \\ \approx \frac {\hat {S} ^ {2} (1)}{n _ {0}} + \frac {\hat {S} ^ {2} (0)}{n _ {1}} \\ \approx \frac {S ^ {2} (1)}{n _ {0}} + \frac {S ^ {2} (0)}{n _ {1}}, \\ \end{array}
$$

which does not match the asymptotic variance of ˆτ . Ideally, we should compute the $p \mathrm { - }$ -value under $H _ { \mathrm { 0 N } }$ based the true distribution of ${ \hat { \tau } } .$ , which, however, depends on the unknown potential outcomes. In contrast, we use the FRT to compute the $p _ { \mathrm { F R T } }$ based on the permutation distribution $( \hat { \tau } ) ^ { \pi }$ , which does not match the true distribution of $\hat { \tau }$ under $H _ { \mathrm { 0 N } }$ even with large samples. Therefore, the FRT with $\hat { \tau }$ may not control the type one error rate under $H _ { \mathrm { 0 N } }$ even with large samples.

Fortunately, the undesired property of the FRT with $\hat { \tau }$ goes away if we replace the test statistic ˆτ with the studentized version t. Under $H _ { \mathrm { 0 N } }$ , we have

$$
t \dot {\sim} \mathrm{N} (0, C ^ {2})
$$

where $C ^ { 2 } \leq 1$ with equality holding if $Y _ { i } ( 1 ) - Y _ { i } ( 0 ) = \tau$ for all units $i =$ $1 , \ldots , n .$ . The FRT generates the permutation distribution

$$
t ^ {\pi} \dot {\sim} \mathrm{N} (0, 1)
$$

where the variance equals 1 because the Science Table used by the FRT has zero individual causal effects. Under $H _ { \mathrm { 0 N } }$ , because the true distribution of t is more dispersed than the corresponding permutation distribution, the pfrt based on t is asymptotically conservative.

## 8.2 Covariate-adjusted FRTs in the CRE

Extending the discussion in Section 8.1 to the case with covariates, Zhao and Ding (2021a) recommend using the FRT with the studentized Lin (2013)’s estimator:

$$
t _ {\mathrm{L}} = \frac {\hat {\tau} _ {\mathrm{L}}}{\sqrt {\hat {V} _ {\mathrm{L}}}},
$$

which is the robust t-statistic for the coefficient of $Z _ { i }$ in the OLS fit of $Y _ { i }$ on $1 , Z _ { i } , X _ { i }$ and $Z _ { i } X _ { i }$ . They show that the FRT with $t _ { \mathrm { L } }$ has multiple guarantees:

1. the associate $p _ { \mathrm { F R T } }$ is finite-sample exact under $H _ { \mathrm { 0 F } }$ ;  
2. it is asymptotically conservative under $H _ { \mathrm { 0 N } } ;$  
3. it is asymptotically more powerful than the FRT with t when $H _ { \mathrm { 0 N } }$ does not hold and the covariates are predictive to the outcomes;  
4. the above properties holds even if the linear outcome model is misspecified.

Similarly, this is a feature of the the studentized statistic $t _ { \mathrm { L } }$ . Zhao and Ding (2021a) show that other covariate-adjusted FRTs reviewed in Section 6.2.1 may be either anti-conservative under $H _ { \mathrm { 0 N } }$ or less powerful than the FRT with $t _ { \mathrm { L } }$ when $H _ { \mathrm { 0 N } }$ does not hold.

## 8.3 General recommendations

The recommendations for the SRE parallel those for the CRE if both the strong and weak null hypotheses are of interest. Without additional covariates, Zhao and Ding (2021a) recommend using the FRT with

$$
t _ {\mathrm{S}} = \frac {\hat {\tau} _ {\mathrm{S}}}{\sqrt {\hat {V} _ {\mathrm{S}}}};
$$

with additional covariates, they recommend using the FRT with

$$
t _ {\mathrm{L,S}} = \frac {\hat {\tau} _ {\mathrm{L,S}}}{\sqrt {\hat {V} _ {\mathrm{L,S}}}}.
$$

The analysis of ReM is trickier. Zhao and Ding (2021a) show that the FRT with t does not have the dual guarantees in Section 8.1, but the FRT with $t _ { \mathrm { L } }$ still has the guarantees in Section 8.2. This highlights the importance of both covariate adjustment and studentization in ReM.

Similar results hold for the MPE. Without covariates, we recommend using the FRT with the t-statistic for the intercept in the OLS fit of $\hat { \tau } _ { i }$ on 1; with covariates, we recommend using the FRT with the t-statistic for the intercept in the OLS fit of $\hat { \tau } _ { i }$ on 1 and $\widehat { \tau } _ { x , i }$ . Figure 7.2 in Chapter 7 are based on these recommended FRTs.

Overall, the FRTs with studentized statistics are safer choices. When the large-sample Normal approximations to the studentized statistics are accurate, the FRTs give $p _ { \mathrm { F R T } } \mathrm { ^ { * } s }$ that are almost identical to those based on Normal approximations. When the large-sample approximations are inaccurate, the FRTs at least guarantees valid p-values under the strong null hypotheses. This is the recommendation of this book.

## 8.4 A case study

Chong et al. (2016) conducted a randomized experiment on 219 students of a rural secondary school in the Cajamarca district of Peru during the 2009 school year. They first provided the village clinic with iron supplements and trained the local staff to distribute one free iron pill to any adolescent who requested one in person. They then randomly assign students to three arms with three different types of videos: in the first video, a popular soccer player was encouraging the use of iron supplements to maximize energy (“soccer” arm); in the second video, a physician was encouraging the use of iron supplements to improve overall health (“physician” arm); the third video did not mention iron at all (“control” arm). The experiment was stratified on the class level (1–5). The treatment and control group sizes within classes are shown below:

<table><tr><td></td><td>class 1</td><td>class 2</td><td>class 3</td><td>class 4</td><td>class 5</td></tr><tr><td>soccer</td><td>16</td><td>19</td><td>15</td><td>10</td><td>10</td></tr><tr><td>physician</td><td>17</td><td>20</td><td>15</td><td>11</td><td>10</td></tr><tr><td>control</td><td>15</td><td>19</td><td>16</td><td>12</td><td>10</td></tr></table>

One outcome of interest is the average grades in the third and fourth quarters of 2009, and an important background covariate was the anemia status at baseline. We make pairwise comparisons of the “soccer” arm versus the “control” arm and the “physician” arm versus the “control” arm. We also compare the FRTs with and without using the covariate indicating the baseline anemia status. We use their dataset to illustrate the FRTs in complete randomization and stratified randomization. The ten subgroup analyses within the same class levels use the FRTs with t and $t _ { \mathrm { L } }$ for the CRE and the two overall analyses averaging over all class levels use the FRTs with tS and $t _ { \mathrm { L } ,  { \mathrm { S } } }$ for the SRE.

Table 8.1 shows the point estimators, standard errors, the p-value based on the Normal approximation of the robust t-statistics, and the p-value based on the FRTs. In most strata, covariate adjustment decreases the standard error since the baseline anemia status is predictive to the outcome. Table 8.1 also exhibits two exceptions: within class 2, covariate adjustment increases the standard error when comparing “soccer” and “control”; in class 4, covariate adjustment increases the standard error when comparing “physician” and “control”. This is due to the small group sizes within these strata, causing the asymptotic approximation dubious. Nevertheless, in these two scenarios, the differences in the standard error are in the third digit. The p-values from the Normal approximation and the FRT are close with the latter being slightly larger in most cases. Based on the theory, the p-values based on the FRT should be trusted since it has an additional guarantee of being finite-sample exact under the sharp null hypothesis. This becomes important in this example since the groups sizes are quite small within strata.

We echo Bind and Rubin (2020)’s suggestion that when conducting the FRTs, not only the p-values but also the randomization distributions of the test statistics should be reported. Figure 8.1 compares the histograms of the randomization distributions of the robust t-statistics with the asymptotic approximations. In the subgroup analysis, we can observe discrepancy between the randomization distributions and N(0, 1); average over all class levels, the discrepancy becomes unnoticeable. Overall, in this application, the p-values based on the Normal approximation do not differ substantially from those based on the FRTs. Two approaches yield coherent conclusions: the video with a physician telling the benefits of iron supplements improved the academic performance and the effect was most significant among student in class 3; in contrast, the video with a famous soccer player telling the benefits of the iron supplements did not have any significant effect.

## 8.5 Homework Problems

## 8.1 Re-analyzing Angrist and Lavy (2009)’s data

This is the Fisherian counterpart of Problem 7.8. Report the $p _ { \mathrm { F R T } } \mathrm { ^ { * } s }$ from the FRTs with studentized statistics.

**TABLE 8.1: Re-analysis of Chong’s data. N corresponds to the unadjusted estimators and tests, and L corresponds to the covariate-adjusted estimators and tests. (a) soccer versus control (b) physician versus control**

<table><tr><td></td><td>est</td><td>s.e.</td><td> $p_{normal}$ </td><td> $p_{frt}$ </td></tr><tr><td colspan="5">class 1</td></tr><tr><td>N</td><td>0.051</td><td>0.502</td><td>0.919</td><td>0.924</td></tr><tr><td>L</td><td>0.050</td><td>0.489</td><td>0.919</td><td>0.929</td></tr><tr><td colspan="5">class 2</td></tr><tr><td>N</td><td>-0.158</td><td>0.451</td><td>0.726</td><td>0.722</td></tr><tr><td>L</td><td>-0.176</td><td>0.452</td><td>0.698</td><td>0.700</td></tr><tr><td colspan="5">class 3</td></tr><tr><td>N</td><td>0.005</td><td>0.403</td><td>0.990</td><td>0.989</td></tr><tr><td>L</td><td>-0.096</td><td>0.385</td><td>0.803</td><td>0.806</td></tr><tr><td colspan="5">class 4</td></tr><tr><td>N</td><td>-0.492</td><td>0.447</td><td>0.271</td><td>0.288</td></tr><tr><td>L</td><td>-0.511</td><td>0.447</td><td>0.253</td><td>0.283</td></tr><tr><td colspan="5">class 5</td></tr><tr><td>N</td><td>0.390</td><td>0.369</td><td>0.291</td><td>0.314</td></tr><tr><td>L</td><td>0.443</td><td>0.318</td><td>0.164</td><td>0.186</td></tr><tr><td colspan="5">all</td></tr><tr><td>N</td><td>-0.051</td><td>0.204</td><td>0.802</td><td>0.800</td></tr><tr><td>L</td><td>-0.074</td><td>0.200</td><td>0.712</td><td>0.712</td></tr></table>

<table><tr><td></td><td>est</td><td>s.e.</td><td> $p_{normal}$ </td><td> $p_{\text{frt}}$ </td></tr><tr><td colspan="5">class 1</td></tr><tr><td>N</td><td>0.567</td><td>0.426</td><td>0.183</td><td>0.192</td></tr><tr><td>L</td><td>0.588</td><td>0.418</td><td>0.160</td><td>0.174</td></tr><tr><td colspan="5">class 2</td></tr><tr><td>N</td><td>0.193</td><td>0.438</td><td>0.659</td><td>0.666</td></tr><tr><td>L</td><td>0.265</td><td>0.409</td><td>0.517</td><td>0.523</td></tr><tr><td colspan="5">class 3</td></tr><tr><td>N</td><td>1.305</td><td>0.494</td><td>0.008</td><td>0.012</td></tr><tr><td>L</td><td>1.501</td><td>0.462</td><td>0.001</td><td>0.003</td></tr><tr><td colspan="5">class 4</td></tr><tr><td>N</td><td>-0.273</td><td>0.413</td><td>0.508</td><td>0.515</td></tr><tr><td>L</td><td>-0.313</td><td>0.417</td><td>0.454</td><td>0.462</td></tr><tr><td colspan="5">class 5</td></tr><tr><td>N</td><td>-0.050</td><td>0.379</td><td>0.895</td><td>0.912</td></tr><tr><td>L</td><td>-0.067</td><td>0.279</td><td>0.811</td><td>0.816</td></tr><tr><td colspan="5">all</td></tr><tr><td>N</td><td>0.406</td><td>0.202</td><td>0.045</td><td>0.047</td></tr><tr><td>L</td><td>0.463</td><td>0.190</td><td>0.015</td><td>0.017</td></tr></table>

![image_08](images/image_08.png)

![image_09](images/image_09.png)

FIGURE 8.1: Re-analyzing Chong et al. (2016)’s data: randomization distributions with $5 \times 1 0 ^ { 4 }$ Monte Carlo draws and the N(0, 1) approximations

## 8.2 Replication of Zhao and Ding (2021a)’s Figure 1

Zhao and Ding (2021a) use simulation to evaluate the finite-sample properties of the $p _ { \mathrm { F R T } } \mathrm { ^ { * } s }$ from the FRTs with various test statistics. Based on their Figure 1, they recommend using the FRT with $t _ { \mathrm { L } ,  { \mathrm { S } } }$ to analyze the SRE. Replicate their Figure 1.

## 8.3 Recommended reading

Zhao and Ding (2021a).

## 9

# Bridging Finite and Super Population Causal Inference

We have focused on the finite population perspective in randomized experiment. It treats all the potential outcomes as fixed numbers or conditions on them if they are realizations of some random variables. The advantage of this perspective is that it focuses on the design of the experiments and requires minimal assumptions on the data generating process of the outcomes. However, it is often criticized for having only internal validity but not necessarily external validity. Obviously, all experimenters care about not only the internal validity but also the external validity of their experiments. Since all statistical properties are conditional on the potential outcomes for the units we have, the results are only about the observed units. Then a natural question arises: do the finite population results generalize to a bigger population?

This is a fair critique on the finite population framework conditional on the potential outcomes. However, this can be a philosophical question. What we observed is a finite population, so any experimental design and analysis directly give us information about this finite population. Randomization only ensures internal validity given the potential outcomes of these units. The external validity of the results depend on the sampling process of the units. If the finite population is a representative sample of a larger population we are interested in, then of course the experimental results also have external validity. Otherwise, the results based on randomization inference may not generalize. Pearl and Bareinboim (2014) discussed this transportability problem from a different perspective.

For some statisticians, this is just a technical problem. We can change the statistical framework, assuming that the units are sampled from a super population. Then all the statements are about the population of interest. This is a convenient framework, although it does not really solve the problem mentioned above. Below, I will introduce this framework for two purposes: first, it gives a different perspective for randomized experiments; second, it serves as a bridge between Parts II and III of this book. The latter purpose is more important, since the super population framework allows us to derive more fruitful results for observational studies in which the treatment is not randomly assigned.

## 9.1 CRE

Assume

$$
\{Z _ {i}, Y _ {i} (1), Y _ {i} (0), X _ {i} \} _ {i = 1} ^ {n} \stackrel {{\text {IID}}} {{\sim}} \{Z, Y (1), Y (0), X \}
$$

from a super population. With a little abuse of notation, we define the population average causal effect as

$$
\tau = E \{Y (1) - Y (0) \} = E \{Y (1) \} - E \{Y (0) \}.
$$

Under the super population framework, we can formulate the CRE as below.

## Definition 9.1 (CRE under the super population framework) Z $\{ Y ( 1 ) , Y ( 0 ) , X \}$

Under Definition 9.1, the average causal effect can be written as

$$
\begin{array}{l} \tau = E \{Y (1) \mid Z = 1 \} - E \{Y (0) \mid Z = 0 \} \\ = E (Y \mid Z = 1) - E (Y \mid Z = 0), \tag {9.1} \\ \end{array}
$$

which equals the difference in expectations of the outcomes. Since τ can be expressed as a function of the distributions of the observables, it is nonparametrically identifiable1. The identification formula (9.1) immediately suggests a moment estimator ˆτ , which is the difference in means of the outcomes defined before. Conditioning on Z, this is then a standard two-sample problem comparing the means of two independent samples. We have

$$
E (\hat {\tau} \mid \mathbf {Z}) = \tau , \quad \mathrm{var} (\hat {\tau} \mid \mathbf {Z}) = \frac {\mathrm{var} \{Y (1) \}}{n _ {1}} + \frac {\mathrm{var} \{Y (0) \}}{n _ {0}}.
$$

Under IID sampling, the sample variances are unbiased for the population variances, so Neyman (1923)’s variance estimator is unbiased for $\mathrm { v a r } ( \hat { \tau } \mid Z )$ . The conservativeness problem goes away under this super population framework.

We can also discuss covariate adjustment. Based on the OLS decompositions (see Chapter A2)

$$
Y (1) = \gamma_ {1} + \beta_ {1} ^ {\mathsf {T}} X + \varepsilon (1), \tag {9.2}
$$

$$
Y (0) = \gamma_ {0} + \beta_ {0} ^ {\mathsf {T}} X + \varepsilon (0), \tag {9.3}
$$

we have

$$
\tau = E \{Y (1) - Y (0) \} = \gamma_ {1} - \gamma_ {0} + (\beta_ {1} - \beta_ {0}) ^ {\mathsf {T}} E (X),
$$

since the residuals $\varepsilon ( 1 )$ and $\varepsilon ( 0 )$ have mean zero due to the inclusion of the intercepts. We can use the OLS with the treated and control data to estimate the coefficients in (9.2) and (9.3), respectively. The sample versions of the coefficients are $\hat { \gamma } _ { 1 } , \hat { \beta } _ { 1 } , \hat { \gamma } _ { 0 } , \hat { \beta } _ { 0 }$ , so a covariate-adjusted estimator for τ is

$$
\hat {\tau} _ {\mathrm{adj}} = \hat {\gamma} _ {1} - \hat {\gamma} _ {0} + (\hat {\beta} _ {1} - \hat {\beta} _ {0}) ^ {\mathsf {T}} \bar {X}.
$$

If we center covariates with $\bar { X } = 0$ , the above estimator reduces to Lin (2013)’s estimator

$$
\hat {\tau} _ {\mathrm{L}} = \hat {\gamma} _ {1} - \hat {\gamma} _ {0},
$$

which equals the coefficient of Z in the pooled regression with treatmentcovariates interactions.

Unfortunately, the EHW variance estimator does not work for $\hat { \tau } _ { \mathrm { L } }$ because of the additional uncertainty X¯ under the super population framework. Berk et al. (2013), Negi and Wooldridge (2021) and Zhao and Ding (2021a) proposed a correction of the EHW variance estimator by adding an additional term

$$
(\hat {\beta} _ {1} - \hat {\beta} _ {0}) ^ {\mathsf {T}} S _ {X} ^ {2} (\hat {\beta} _ {1} - \hat {\beta} _ {0}) / n.
$$

A conceptually simpler yet computationally intensive approach is to use the bootstrap to estimate the variance; see Chapter A1.5.

## 9.2 SRE

We can extend the discussion in Section 9.1 to the SRE since it is equivalent to independent CREs within strata. The notation below will be slightly different from that in Chapter 5.

Assume that

$$
\{Z _ {i}, Y _ {i} (1), Y _ {i} (0), X _ {i} \} \stackrel {{\text {IID}}} {{\sim}} \{Z, Y (1), Y (0), X \}.
$$

With a discrete covariate $X _ { i } \in \{ 1 , \ldots , K \}$ , we can formulate the SRE as below.

Definition 9.2 (SRE under the super population framework) $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid$ X.

Under Definition 9.2, the conditional average causal effect can be rewritten as

$$
\tau_ {[ k ]} = E \{Y (1) - Y (0) \mid X = k \} = E (Y \mid Z = 1, X = k) - E (Y \mid Z = 0, X = k),
$$

so the average causal effect can be rewritten as

$$
\tau = E \{Y (1) - Y (0) \} = \sum_ {k = 1} ^ {K} \operatorname{pr} (X _ {-} k) E \{Y (1) - Y (0) \mid X = k \} = \sum_ {k = 1} ^ {K} \operatorname{pr} (X = k) \tau_ {[ k ]}.
$$

The discussion in Section 9.1 holds with all strata, so we can derive the super population analog for the SRE. When there are more than two treatment and control units within each strata, we can use $\hat { V } _ { \mathrm { S } }$ as an unbiased variance estimator for var(ˆτS).

## 9.3 Homework Problems

## 9.1 OLS decomposition of the observed outcome under the CRE

Based on (9.2) and (9.3), show that the OLS decomposition of the observed outcome on the treatment, covariates and their interaction is

$$
Y = \alpha_ {0} + \alpha_ {Z} Z + \alpha_ {X} ^ {\mathsf {T}} X + \alpha_ {Z X} ^ {\mathsf {T}} X Z + \varepsilon
$$

where

$$
\alpha_ {0} = \gamma_ {0}, \quad \alpha_ {Z} = \gamma_ {1} - \gamma_ {0}, \quad \alpha_ {X} = \beta_ {0}, \quad \alpha_ {Z X} = \beta_ {1} - \beta_ {0}, \quad \varepsilon = Z \varepsilon (1) + (1 - Z) \varepsilon (0).
$$

That is,

$$
(\alpha_ {0}, \alpha_ {Z}, \alpha_ {X}, \alpha_ {Z X}) = \arg \min _ {a _ {0}, a _ {Z}, a _ {X}, a _ {Z X}} E (Y - a _ {0} - a _ {Z} Z - a _ {X} ^ {\mathsf {T}} X - a _ {Z X} ^ {\mathsf {T}} X Z) ^ {2}.
$$

## 9.2 Recommended reading

Ding et al. (2017a) provide a unified discussion of the finite-population and super-population inferences for the average causal effect.

## Part III

## Observational studies

## 10

# Observational Studies, Selection Bias, and Nonparametric Identification of Causal Effects

Cochran (1965) summarized two common characteristics of observational studies:

1. the objective is to elucidate cause-and-effect relationships;  
2. it is not feasible to use controlled experimentation.

The first characteristic is identical to that of randomized experiments discussed in Part II, but the second differs fundamentally from randomized experiments.

Dorn (1953) suggested that the planner of an observational study should always ask himself the following question:

How would the study be conducted if it were possible to do it by controlled experimentation?

It is always helpful to follow Dorn (1953)’s suggestion because the potential outcomes framework has an intrinsic link to an experiment, either a real experiment or a thought experiment. Part III of this book will discuss causal inference with observational studies. It will clarify the fundamental differences between observational studies and randomized experiments. Nevertheless, many ideas of causal inference with observational studies are deeply connected to those with randomized experiments.

## 10.1 Motivating Examples

Example 10.1 (job training program) LaLonde (1986) was interested in the causal effect of a job training program on earnings. His compared the results based on a randomized experiment to the results based on observational studies. We have used the experimental data before, which is the lalonde dataset in the Matching package; we have also used an observational counterpart cps1re74.csv in Problem 1.3. LaLonde (1986) found that many traditional econometric methods for observational studies gave quite different estimates compared to the estimates based on the experimental data. Dehejia and Wahba (1999) re-analyzed the data using methods motivated by causal inference, and found that those methods can recover the experimental gold standard. Since then, this became a canonical example in causal inference with observational studies.

Example 10.2 (smoking and homocysteine) Bazzano et al. (2003) compared the homocysteine levels in daily smokers and never smokers based on the data from the National Health and Nutrition Examination Survey (NHANES) 2005–2006. Rosenbaum (2018) documented the data as homocyst in the package senstrat. The dataset has the following important covariates:

female 1=female, 0=male
age3 three age categories: 20–39, 40–50, ≥60
ed3 three education categories: < High School, High School, some College
bmi3 three BMI categories: <30, [30,35), ≥ 35
pov2 TRUE=income at least twice the poverty level, FALSE otherwise

Example 10.3 (school meal program and body mass index) Chan et al. (2016) used a subsample of the data from NHANES 2007–2008 to study whether participation in school meal programs lead to an increase in BMI for school children. They documented the data as nhanesbmi in the package ATE. The dataset has the following important covariates:

age age
ChildSex gender (1: Male, 0: Female)
black race (1: Black, 0: otherwise)
mexam race (1: Hispanic: 0 otherwise)
pir200_plus Family above 200% of the federal poverty level
WIC Participation in the special supplemental nutrition program
Food_Stamp Participation in food stamp program
fsdchbi Childhood food security
AnyIns Any insurance
RefSex Gender of the adult respondent (1: Male, 0: Female)
RefAge Age of the adult respondent

## 10.2 Causal effects and selection bias under the potential outcomes framework

For unit $i ( i = 1 , \ldots , n )$ , we have pretreatment covariates $X _ { i } ,$ a binary treatment indicator $Z _ { i } .$ , and an observed outcome $Y _ { i }$ with two potential outcomes $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ under treatment and control, respectively. For simplicity, we assume

$$
\{X _ {i}, Z _ {i}, Y _ {i} (1), Y _ {i} (0) \} _ {i = 1} ^ {n} \stackrel {{\text {IID}}} {{\sim}} \{X, Z, Y (1), Y (0) \}.
$$

## 10.2 Causal effects and selection bias under the potential outcomes framework 129

So we can drop the subscript i for quantities depending on this population. The causal effects of interest are the average causal effect

$$
\tau = E \{Y (1) - Y (0) \},
$$

the average causal effect on the treated units

$$
\tau_ {\mathrm{T}} = E \{Y (1) - Y (0) \mid Z = 1 \},
$$

and the average causal effect on the control units:

$$
\tau_ {\mathrm{C}} = E \{Y (1) - Y (0) \mid Z = 0 \}.
$$

By the linearity of the expectation, we have

$$
\begin{array}{l} \tau_ {\mathrm{T}} = E \{Y (1) \mid Z = 1 \} - E \{Y (0) \mid Z = 1 \} \\ = E (Y \mid Z = 1) - E \{Y (0) \mid Z = 1 \} \\ \end{array}
$$

and

$$
\begin{array}{l} \tau_ {\mathrm{C}} = E \{Y (1) \mid Z = 0 \} - E \{Y (0) \mid Z = 0 \} \\ = E \{Y (1) \mid Z = 0 \} - E (Y \mid Z = 0). \\ \end{array}
$$

In the above two formulas of $\tau _ { \mathrm { T } }$ and $\tau _ { \mathrm { C } }$ , the quantities $E ( Y \mid Z = 1 )$ and $E ( Y \mid$ $Z = 0 )$ are directly observable from the data, but the quantities $E \{ Y ( 0 ) \mid Z =$ $1 \}$ and $E \{ Y ( 1 ) \mid Z = 0 \}$ are not. The latter two are counterfactuals because they are the means of the potential outcomes corresponding to the treatment level that is the opposite of the actual received treatment.

The simple difference in means, also known as the prima facie causal effect,

$$
\begin{array}{l} \tau_ {\mathrm{PF}} = E (Y \mid Z = 1) - E (Y \mid Z = 0) \\ = E \{Y (1) \mid Z = 1 \} - E \{Y (0) \mid Z = 0 \} \\ \end{array}
$$

is generally biased for the causal effects defined above. For example,

$$
\tau_ {\mathrm{PF}} - \tau_ {\mathrm{T}} = E \{Y (0) \mid Z = 1 \} - E \{Y (0) \mid Z = 0 \}
$$

and

$$
\tau_ {\mathrm{PF}} - \tau_ {\mathrm{C}} = E \{Y (1) \mid Z = 1 \} - E \{Y (1) \mid Z = 0 \}
$$

are not zero in general, and they quantifies the selection bias. They measure the differences in the means of the potential outcomes across the treatment and control groups.

Why randomization is so important? Rubin (1978) first used potential outcomes to quantify the benefit of randomization. We have used the fact in Chapter 9 that

$$
Z \bot \{Y (1), Y (0) \} \tag {10.1}
$$

in the CRE, which implies that the selection bias terms are both zero:

$$
\tau_ {\mathrm{PF}} - \tau_ {\mathrm{T}} = E \{Y (0) \mid Z = 1 \} - E \{Y (0) \mid Z = 0 \} = 0
$$

and

$$
\tau_ {\mathrm{PF}} - \tau_ {\mathrm{C}} = E \{Y (1) \mid Z = 1 \} - E \{Y (1) \mid Z = 0 \} = 0.
$$

So under complete randomization (10.1),

$$
\tau = \tau_ {\mathrm{T}} = \tau_ {\mathrm{C}} = \tau_ {\mathrm{PF}}.
$$

From the above discussion, the fundamental benefit of randomization is to balance the distributions of the potential outcomes across the treatment and control groups, which is more important than to balance the distributions of the observed covariates.

Without randomization, the selection bias terms can be arbitrarily large especially for unbounded outcomes. This highlights the fundamental difficulty of causal inference with observational studies.

## 10.3 Sufficient conditions for nonparametric identification

## 10.3.1 Identification

Causal inference with observational studies is challenging. It relies on strong assumptions. A strategy is to use the information of the pretreatment covariates and assume that conditioning on the observed covariates X, the selection bias terms are zero, that is,

$$
E \{Y (0) \mid Z = 1, X \} = E \{Y (0) \mid Z = 0, X \}, \tag {10.2}
$$

$$
E \{Y (1) \mid Z = 1, X \} = E \{Y (1) \mid Z = 0, X \}. \tag {10.3}
$$

The assumptions in (10.2) and (10.3) state that the differences in the means of the potential outcomes across the treatment and control groups are entirely due to the difference in the observed covariates. So given the same value of the covariates, the potential outcomes have the same means across the treatment and control groups. Mathematically, (10.2) and (10.3) ensure that the conditional versions of the effects are identical:

$$
\tau (X) = \tau_ {\mathrm{T}} (X) = \tau_ {\mathrm{C}} (X) = \tau_ {\mathrm{PF}} (X),
$$

where

$$
\begin{array}{l} \tau (X) = E \{Y (1) - Y (0) \mid X \}, \\ \tau_ {\mathrm{T}} (X) = E \{Y (1) - Y (0) \mid Z = 1, X \}, \\ \tau_ {\mathrm{C}} (X) = E \{Y (1) - Y (0) \mid Z = 0, X \}, \\ \tau_ {\mathrm{PF}} (X) = E (Y \mid Z = 1, X) - E (Y \mid Z = 0, X). \\ \end{array}
$$

In particular, $\tau ( X )$ is often called the conditional average causal $e f f e c t .$

A key result in this chapter is that the average causal effect τ is nonparametrically identifiable under (10.2) and (10.3). The notion of nonparametrically identifiability does not appear frequently in classic statistics, but it is key to causal inference with observational studies.

Definition 10.1 (identification) A parameter θ is identifiable if it can be written as a function of the distribution of the observed data under certain model assumptions. A parameter θ is nonparametrically identifiable $i f$ it can be written as a function of the distribution of the observed data without any parametric model assumptions.

Definition 10.1 is too abstract at the moment. I will use more concrete examples in later chapters to illustrate its meaning. It is often neglected in standard statistics problems. For instance, the mean $\theta = E ( Y )$ is nonparametrically identifiable if we have IID draws of $Y _ { i } ^ { \mathrm { : } } \mathrm { s } ;$ the Pearson correlation coefficient $\theta \ : = \ : \mathrm { c o r r } ( X , Y )$ is nonparametrically identifiable if we have IID draws of the pairs $( X _ { i } , Y _ { i } ) \mathrm { { ^ { \circ } s } }$ . In those examples, the parameters are nonparametrically identifiable automatically. However, Definition 10.1 is fundamental in causal inference with observational studies. In particular, the parameter of interest $\tau = E \{ Y ( 1 ) - Y ( 0 ) \}$ depends on some unobserved random variables, so it is unclear whether it is nonparametrically identifiable based on observed data. Under the assumptions in (10.2) and (10.3), it is nonparametrically identifiable, with detailed below.

Because $\tau _ { \mathrm { P F } } ( X )$ depends only on the observables, it is nonparametrically identified by definition. Moreover, (10.2) and (10.3) ensure that the three causal effects are the same as $\tau _ { \mathrm { P F } } ( X )$ , so $\tau ( X ) , \ \tau _ { \mathrm { T } } ( X )$ and $\tau _ { \mathrm { C } } ( X )$ are all nonparametrically identified. Consequently, the unconditional versions are also nonparametrically identified under (10.2) and (10.3) due to the law of total expectation:

$$
\tau = E \{\tau (X) \}, \quad \tau_ {\mathrm{T}} = E \{\tau_ {\mathrm{T}} (X) | Z = 1 \}, \quad \tau_ {\mathrm{C}} = E \{\tau_ {\mathrm{C}} (X) | Z = 0 \}.
$$

From now on, we focus on τ unless stated otherwise. The following theorem summarized the identification formulas of τ .

Theorem 10.1 Under (10.2) and (10.3), the average causal effect τ is identified by

$$
\tau = E \{\tau (X) \} \tag {10.4}
$$

$$
= E \{E (Y \mid Z = 1, X) - E (Y \mid Z = 0, X) \} \tag {10.5}
$$

$$
= \int \{E (Y \mid Z = 1, X = x) - E (Y \mid Z = 0, X = x) \} F (\mathrm{d} x). \tag {10.6}
$$

The formula (10.5) was formally established by Rosenbaum and Rubin (1983b), which is also called the g-formula by Robins (see Hern´an and Robins, 2020).

With a discrete covariate, we can write the identification formula in Theorem 10.1 as

$$
\begin{array}{l} \tau = \sum_ {x} E (Y \mid Z = 1, X = x) \mathrm{pr} (X = x) \\ - \sum_ {x} E (Y \mid Z = 0, X = x) \mathrm{pr} (X = x), \tag {10.7} \\ \end{array}
$$

and also the simple difference in means as

$$
\begin{array}{l} \tau_ {\mathrm{PF}} = \sum_ {x} E (Y \mid Z = 1, X = x) \mathrm{pr} (X = x \mid Z = 1) \\ - \sum_ {x} E (Y \mid Z = 0, X = x) \mathrm{pr} (X = x \mid Z = 0) \tag {10.8} \\ \end{array}
$$

by the law of total probability. Comparing (10.7) and (10.8), we can see that although both formulas compare the conditional expectations $E ( Y \mid Z =$ $1 , X = x )$ and $E ( Y \mid Z = 0 , X = x )$ , they average over different distribution of the covariates. The causal parameter τ averages the conditional expectations over the common distribution of the covariate, but the difference in means τPF averages the conditional expectations over two different distributions of covariate in the treated and control groups.

Usually, we impose a stronger assumption:

$$
Y (z) \perp \perp Z \mid X \quad (z = 0, 1). \tag {10.9}
$$

This assumption has many names:

1. ignorability due to Rubin (1978);  
2. unconfoundedness which is popular among epidemiologists;  
3. selection on observables which is popular among social scientists;  
4. conditional independence which is merely a description of the notation in the assumption.

Sometimes, we impose an even stronger assumption

$$
\{Y (1), Y (0) \} \perp Z \mid X \tag {10.10}
$$

which is called strong ignorability (Rosenbaum and Rubin, 1983b). If the parameter of interest is $\tau ,$ then the stronger assumptions (10.9) and (10.10) are just imposed for notational simplicity. They are not necessary in this case. However, they cannot be relaxed if the parameter of interest is the causal effects on other scales (for example, distribution, quantile, or some transformation of the outcome). The strong ignorability assumption requires that the potential outcomes vector be independent of the treatment given covariates, but the ignorability assumption only requires each potential outcome be independent of the treatment given covariates. The former is stronger than the

<!-- footnote -->

- For example, the linear projection of $Y _ { i } ( 1 )$ on $( 1 , X _ { i } )$ is $\alpha _ { 1 } + \beta _ { 1 } X _ { i }$ where
- $( \alpha _ { 1 } , \beta _ { 1 } ) = \arg \operatorname* { m i n } _ { a , b } \sum _ { i = 1 } ^ { n } \{ Y _ { i } ( 1 ) - a - b ^ { \mathsf { T } } X _ { i } \} ^ { 2 } .$

<!-- footnote end -->

<!-- footnote -->

- Without covariates, the HC2 correction yields identical variance estimator as Neyman (1923)’s classic one. For coherence, we can also use the HC2 correction for Lin (2013)’s estimator with covariate adjustment. When the number of covariates is small compared to the sample size and the covariates do not contain outliers, the variants of the EHW standard error perform similarly to the original one. When the number of covariates is large compared to the sample size or the covariates contain outliers, the variants can outperform

<!-- footnote end -->

<!-- footnote -->

- the original one. In those cases, Lei and Ding (2021) recommend using the HC3 variant of the EHW standard error. See Chapter A2 for more details of the EHW standard errors.

<!-- footnote end -->

<!-- footnote -->

- Butler (1969)’s proposed this test statistic under a slightly different framework. Given IID draws of $\big ( \hat { \tau } _ { 1 } , \dots , \hat { \tau } _ { n } \big )$ from a distribution $F ( y )$ , if they are symmetrically distributed around 0, then
- $F ( t ) = \mathrm { p r } ( \hat { \tau } _ { i } \le t ) = \mathrm { p r } ( - \hat { \tau } _ { i } \le t ) = 1 - \mathrm { p r } ( \hat { \tau } _ { i } < - t ) = 1 - F ( - t - ) .$
- Therefore, $\hat { F } ( t ) + \hat { F } ( - t - )$ − 1 measures the deviation from the null hypothesis of symmetry, which motivates the definition of D. A naive definition of the Kolmogorov–Smirnov-type statistic is to compare the empirical distributions of the outcomes under treatment and control as in Example 3.4. Using that definition, we effectively break the pairs. Although it can still be used in the FRT for the MPE, it does not capture the matched-pairs structure of the experiment.

<!-- footnote end -->

<!-- footnote -->

- In causal inference, we say that a parameter is nonparametrically identifiable if it can be determined by the distribution of the observed variables without imposing further parametric assumptions.

<!-- footnote end -->

latter. However, their difference is rather technical and of pure probability interests; see Problem 10.4. In most reasonable statistical models, they are identical; see Section 10.3.2 below. We will not distinguish them in this book and will simply use ignorability to refer to both.

## 10.3.2 Plausibility of the assumption

A fundamental problem of causal inference with observational studies is the plausibility of the ignorability assumption. The above discussion may seem too mathematical in the sense that the ignorability assumption serves as a sufficient condition to ensure the nonparametric identification of the average causal effect. What is its scientific meaning? Intuitively, it rules out all unmeasured covariates that affect treatment and outcome simultaneously. Those “common causes” of the treatment and outcomes are called confounders. That is why the ignorability assumption is also called the unconfoundedness assumption. More mathematically, we can interpret the ignorability assumption based on the outcome data generating process. If

$$
\begin{array}{l} Y (1) = f _ {1} (X, V _ {1}), \\ Y (0) = f _ {0} (X, V _ {0}), \\ Z = 1 \{g (X, V) \geq 0 \} \\ \end{array}
$$

with $( V _ { 1 } , V _ { 0 } ) \bot \bot V$ , then (10.9) and (10.10) hold. In the above data generating process, the “common causes” X of the treatment and the outcome are all observed, the remaining random components are independent. If the data generating process changes to

$$
\begin{array}{l} Y (1) = f _ {1} (X, U, V _ {1}), \\ Y (0) = f _ {0} (X, U, V _ {0}), \\ Z = 1 \{g (X, U, V) \geq 0 \} \\ \end{array}
$$

with $( V _ { 1 } , V _ { 0 } ) \bot \bot V$ , then (10.9) or (10.10) does not hold in general. The unmeasured “common cause” U induces dependence between the treatment and potential outcomes even conditioning on the observed covariates X. If we do not have access to U and analyze the data based only on $( Z , X , Y )$ , the final estimator will be biased for the causal parameter in general. This type of bias is called the omitted variable bias in econometrics.

The ignorability assumption can be reasonable if we observe a rich set of covariates X that affect the treatment and the outcome simultaneously. I start with this assumption, discussing identification and estimation strategies in Part III of this book. However, it is fundamentally untestable. We may justify it based on the scientific background knowledge, but we are often not sure whether it holds or not. Parts IV and V of this book will discuss other strategies when this assumption is not plausible.

## 10.4 Two simple estimation strategies and their limitations

## 10.4.1 Stratification or standardization based on discrete covariates

If the covariate $X _ { i } \in \{ 1 , \ldots , K \}$ is discrete, then ignorability (10.9) reads as

$$
Y (z) \bot Z \mid X = k \quad (z = 0, 1; k = 1, \dots , K),
$$

which essentially assumes that the observational study is a SRE. Therefore, we can use the estimator

$$
\hat {\tau} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \left\{\hat {\bar {Y}} _ {[ k ]} (1) - \hat {\bar {Y}} _ {[ k ]} (0) \right\},
$$

which is identical to the stratified or post-stratified estimator discussed in Chapter 5.

This method is still widely used in practice. Example 10.2 contains discrete covariates, and I relegate the analysis to Problem 10.3. However, there are several obvious difficulties in implementing this method. First, it works well for the case with small K. For large K, it is very likely that many strata have $n _ { [ k ] 1 } = 0 \mathrm { o r } n _ { [ k ] 0 } = 0$ , leading to the illy defined ${ \hat { \tau } } _ { [ k ] } { } ^ { \ \mathrm { { * } } }$ for those strata. This is related to the issue of overlap which will be discussed in Chapter 20. Second, it is not obvious how to apply this stratification method to multidimensional continuous or mixed covariates X. A standard method is to create strata based on the initial covariates and then apply the stratification method. This may result in arbitrariness in the analysis.

## 10.4.2 Outcome regression

The most commonly-used method based on the outcome regression is to run the OLS with an additive model of the observed outcome on the treatment indicator and covariates, which assumes

$$
E (Y \mid Z, X) = \beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X.
$$

If the above linear model is correct, then we have

$$
\begin{array}{l} \tau (X) = E (Y \mid Z = 1, X) - E (Y \mid Z = 0, X) \\ = \left(\beta_ {0} + \beta_ {z} + \beta_ {x} ^ {\mathsf {T}} X\right) - \left(\beta_ {0} + \beta_ {x} ^ {\mathsf {T}} X\right) \\ = \beta_ {z}, \\ \end{array}
$$

which implies that the treatment effect is homogeneous with respect to the covariates. This, coupled with ignorability, implies that

$$
\tau = E \{\tau (X) \} = \beta_ {z}.
$$

Therefore, if ignorability holds and the outcome model is linear, then the average causal effect equals the coefficient of Z. This is one of the most important applications of the linear model. However, the causal interpretation of the coefficient of $Z$ is valid only under two strong assumptions: ignorability and the linear model.

We have discussed in Chapter 6, the above procedure is suboptimal even in randomized experiments, because it ignores the treatment effect heterogeneity induced by the covariates. If we assume

$$
E (Y \mid Z, X) = \beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X + \beta_ {z x} ^ {\mathsf {T}} X Z,
$$

we have

$$
\begin{array}{l} \tau (X) = E (Y \mid Z = 1, X) - E (Y \mid Z = 0, X) \\ = \left(\beta_ {0} + \beta_ {z} + \beta_ {x} ^ {\mathsf {T}} X + \beta_ {z x} ^ {\mathsf {T}} X\right) - \left(\beta_ {0} + \beta_ {x} ^ {\mathsf {T}} X\right) \\ { = } { \beta _ { z } + \beta _ { z x } ^ { \mathsf { T } } X , } \\ \end{array}
$$

which, coupled with ignorability, implies that

$$
\tau = E \{\tau (X) \} = E (\beta_ {z} + \beta_ {z x} ^ {\mathsf {T}} X) = \beta_ {z} + \beta_ {z x} ^ {\mathsf {T}} E (X).
$$

The estimator for $\tau$ is then $\hat { \beta } _ { z } + \hat { \beta } _ { z x } ^ { \sf T } \bar { X }$ , where $\hat { \beta } _ { z }$ is the regression coefficient and X¯ is the sample mean of X. If we center the covariates to ensure $\bar { X } = 0 .$ , then the estimator is simply the regression coefficient of $Z .$ To simplify the procedure, we usually center the covariates at the beginning; also recall Lin (2013)’s estimator introduced in Chapter 6. Rosenbaum and Rubin (1983b) and Hirano and Imbens (2001) discussed this estimator.

In general, we can use other more complex models to estimate the causal effects. For example, if we build two predictors $\hat { \mu } _ { 1 } ( X )$ and $\hat { \mu } _ { 0 } ( X )$ based on the treated and control data, respectively, then we have an estimator for the conditional average causal effect

$$
\hat {\tau} (X) = \hat {\mu} _ {1} (X) - \hat {\mu} _ {0} (X)
$$

and an estimator for the average causal effect:

$$
\hat {\tau} = n ^ {- 1} \sum_ {i = 1} ^ {n} \{\hat {\mu} _ {1} (X _ {i}) - \hat {\mu} _ {0} (X _ {i}) \}.
$$

The estimator ˆτ above has the same form as the projective estimator discussed in Chapter 6. It is sometimes called the outcome imputation estimator. For example, we may model a binary outcome using a logistic model

$$
E (Y \mid Z, X) = \mathrm{pr} (Y = 1 \mid Z, X) = \frac {e ^ {\beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X}}{1 + e ^ {\beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X}},
$$

then based on the estimators of the coefficients $\hat { \beta } _ { 0 } , \hat { \beta } _ { z } , \hat { \beta } _ { x }$ , we have the following estimator for the average causal effect:

$$
\hat {\tau} = n ^ {- 1} \sum_ {i = 1} ^ {n} \left\{\frac {e ^ {\hat {\beta} _ {0} + \hat {\beta} _ {z} + \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}}}{1 + e ^ {\hat {\beta} _ {0} + \hat {\beta} _ {z} + \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}}} - \frac {e ^ {\hat {\beta} _ {0} + \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}}}{1 + e ^ {\hat {\beta} _ {0} + \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}}} \right\}.
$$

This estimator is not simply the coefficient of the treatment in the logistic model.1 It is a nonlinear function of all the coefficients as well as the the empirical distribution of the covariates. In econometrics, this estimator is is called the average partial effect or average marginal effect of the treatment in the logistic model. Many econometric software packages can report this estimator associated with the standard error. Similarly, we can also derive the corresponding estimator based on a fully interacted logistic model; see Problem 10.2.

For all the estimators discussed above, we can use the nonparametric bootstrap to estimate the standard errors. See Chapter A1.5.

The above predictors for the conditional means of the outcome can also be other machine learning tools. In particular, Hill (2011) championed the use of tree methods for estimating τ , and Wager and Athey (2018) proposed to use them also for estimating ˆτ (X). Wager and Athey (2018) also combined the tree methods with the ideas in the next chapter. Since then, machine learning and causal inference has been an active research area (e.g., Hahn et al., 2020; K¨unzel et al., 2019).

The biggest problem of the above approach based on outcome regressions is its sensitivity to the specification of the outcome model. Problem 1.3 gave such an example. Depending on the incentive of empirical research and publications, people sometimes reported their favorable causal effects estimates after searching over a wide set of candidate models, without confessing this searching process. This is a major source of p-hacking in causal inference.

## 10.5 Homework Problems

## 10.1 Nonparametric identification of other causal effects

Under ignorability and overlap, show that

1. the distributional causal effect

$$
\mathrm{DCE} _ {y} = \operatorname * {p r} \{Y (1) > y \} - \operatorname * {p r} \{Y (0) > y \}
$$

is nonparametrically identifiable for all $y ;$

2. the quantile causal effect

$$
\mathrm{QCE} _ {q} = \text { quantile } _ {q} \{Y (1) \} - \text { quantile } _ {q} \{Y (0) \},
$$

is nonparametrically identifiable for all $q ,$ where $\mathrm { q u a n t i l e } _ { q } \{ \cdot \}$ is the qth quantile of a random variable.

Remark: In probability theory, pr $\{ Y ( z ) \leq y \}$ is the cumulative distribution function and pr $\{ Y ( z ) > y \}$ is the survival function of the potential outcome $Y ( z )$ . The distributional causal effect compares the survival functions of the potential outcomes under treatment and control.

## 10.2 Outcome imputation estimator in the fully interacted logistic model

Assume that a binary outcome follows a logistic model

$$
E (Y \mid Z, X) = \operatorname{pr} (Y = 1 \mid Z, X) = \frac {e ^ {\beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X + \beta_ {x z} ^ {\mathsf {T}} X Z}}{1 + e ^ {\beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X + \beta_ {x z} ^ {\mathsf {T}} X Z}}.
$$

What is the corresponding outcome regression estimator for the average causal effect?

## 10.3 Data analysis: stratification and regression

Use the dataset homocyst in the package senstrat. The outcome is homocysteine, the homocysteine level, and the treatment is $\mathbf { z } ,$ where $z =$ 1 for a daily smoker and $z ~ = ~ 0$ for a never smoker. Covariates are female, age3, ed3, bmi3, pov2 with detailed explanations in the package, and st is a stratum indicator, defined by all the combinations of the discrete covariates.

1. How many strata have only treated or control units? What is the proportion of the units in these strata? Drop these strata and perform a stratified analysis of the observational study. Report the point estimator, variance estimator and 95% confidence interval for the average causal effect.  
2. Run the OLS of the outcome on the treatment indicator and covariates without interactions. Report the coefficient of the treatment and the robust standard error.  
Drop the strata with only treated or control units. Re-run the OLS and report the result.  
3. Apply Lin (2013)’s estimator of the average causal effect. Report the coefficient of the treatment and the robust standard error.  
If you do not drop the strata with only treated or control units, what will happen?  
4. Compare the results in the above three analyses. Which one is more credible?

## 10.4 Ignorability versus strong ignorability

Given an example such that the ignorability holds but the strong ignorability does not hold.

Remark: This is related to a classic probability problem of finding three random variables A, B, C such that

$$
A \bot C \text {   and   } B \bot C \text {   but   } (A, B) \not \bot C.
$$

## 10.5 Recommended reading

Cochran (1965) is a classic reference on observational studies. It contains many useful insights but does not use the formal potential outcomes framework.

# The Central Role of the Propensity Score in Observational Studies for Causal Effects

Rosenbaum and Rubin (1983b) proposed the key concept propensity score and discussed its role in causal inference with observational studies. It is one of the most cited papers in statistics, and Titterington (2013) listed it as the second most cited paper published in Biometrika during the past 100 years. Its citations are growing very fast during the recent years.

Under the IID sampling assumption, we have four random variables associated with each unit: $\{ X , Z , Y ( 1 ) , Y ( 0 ) \}$ . Following the basic probability rule, we can factorize the joint distribution as

$$
\operatorname{pr} \{X, Z, Y (1), Y (0) \}
$$

$$
= \operatorname{pr} (X) \times \operatorname{pr} \{Y (1), Y (0) \mid X \} \times \operatorname{pr} \{Z \mid X, Y (1), Y (0) \},
$$

where $\mathrm { p r } ( X )$ is the covariate distribution, $\operatorname { p r } \{ Y ( 1 ) , Y ( 0 ) \mid X \}$ is the outcome model, and $\operatorname { p r } \{ Z \mid X , Y ( 1 ) , Y ( 0 ) \}$ is the treatment assignment mechanism. Usually, we do not want to model the covariates because they are background information happening before the treatment and outcome. If we want to move beyond the outcome model, then we must focus on the treatment assignment mechanism, which leads to the definition of the propensity score.

Definition 11.1 (propensity score) $D e f i n e$

$$
e (X, Y (1), Y (0)) = \operatorname{pr} \{Z = 1 \mid X, Y (1), Y (0) \}
$$

as the propensity score. Under strong ignorability, we have

$$
e (X, Y (1), Y (0)) = \operatorname{pr} \{Z = 1 \mid X, Y (1), Y (0) \} = \operatorname{pr} (Z = 1 \mid X),
$$

so the propensity score reduces to

$$
e (X) = \operatorname{pr} (Z = 1 \mid X),
$$

the conditional probability of the receiving the treatment given the observed covariates.

Rosenbaum and Rubin (1983b) used $e ( X ) = \mathrm { p r } ( Z = 1 \mid X )$ as the definition of the propensity score because they focused on observational studies under ignorability. It is sometimes helpful to view $e ( X , Y ( 1 ) , Y ( 0 ) ) = \mathrm { p r } \{ Z =$ $1 \mid X , Y ( 1 ) , Y ( 0 ) \}$ as the general definition of the propensity score even when ignorability fails. See Problem 11.1 for more details.

Following Rosenbaum and Rubin (1983b), this chapter will demonstrate that $e ( X )$ is a key quantity in causal inference with observational studies under ignorability.

## 11.1 The propensity score as a dimension reduction tool

## 11.1.1 Theory

Theorem 11.1 $I f Z \perp \perp \{ Y ( 1 ) , Y ( 0 ) \} \mid X , t h e n Z \perp \{ Y ( 1 ) , Y ( 0 ) \} \mid e ( X ) .$

Theorem 11.1 states that if strong ignorability holds conditional on covariates X, then it also holds conditional on the scalar propensity score $e ( X )$ . The ignorability requires conditioning on many background characteristics Z of the units, but Theorem 11.1 implies that controlling for the propensity score $e ( X )$ romoves all confounding induced by covariates X. The original covariates X can be general and have many dimensions, but the propensity score $e ( X )$ is a one-dimensional scalar variable bounded between 0 and 1. Therefore, the propensity score reduces the dimension of the original covariates but still maintain the ignorability. As a technical statistical terminology, we can view the propensity score as a dimensional reduction tool. We will first prove Theorem 11.1 below and then given an application of the dimension reduction property of the propensity score.

Proof of Theorem 11.1: By the definition of conditional independence, we need to show that

$$
\operatorname{pr} \{Z = 1 \mid Y (1), Y (0), e (X) \} = \operatorname{pr} \{Z = 1 \mid e (X) \}. \tag {11.1}
$$

The left-hand side of (11.1) equals

$$
\begin{array}{l} \operatorname{pr} \{Z = 1 \mid Y (1), Y (0), e (X) \} \\ = E \{Z \mid Y (1), Y (0), e (X) \} \\ = E \left[ E \{Z \mid Y (1), Y (0), e (X), X \} \mid Y (1), Y (0), e (X) \right] \\ (\text { tower   property; see   Section   A1.1.1 }) \\ = E \left[ E \{Z \mid Y (1), Y (0), X \} \mid Y (1), Y (0), e (X) \right] \\ = E \left\{E (Z \mid X) \mid Y (1), Y (0), e (X) \right\} \quad (\text { strong   ignorability }) \\ = E \left\{e (X) \mid Y (1), Y (0), e (X) \right\} \\ = e (X). \\ \end{array}
$$

The right-hand side of (11.1) equals

$$
\begin{array}{l} \operatorname{pr} \{Z = 1 \mid e (X) \} \\ = E \{Z \mid e (X) \} \\ = E \left[ E \{Z \mid e (X), X \} \mid e (X) \right] \quad (\text { tower   property }) \\ = E \left\{E (Z \mid X) \mid e (X) \right\} \\ = E \left\{e (X) \mid e (X) \right\} \\ = e (X). \\ \end{array}
$$

So the left-hand side of (11.1) equals the right-hand side of (11.1).

![image_10](images/image_10.png)

## 11.1.2 Propensity score stratification

Theorem 11.1 motivates a simple method for estimating causal effects: propensity score stratification. Starting from the simple case, we assume that the propensity score is known and only takes K possible values $\{ e _ { 1 } , \ldots , e _ { K } \}$ with K being much smaller than the sample size n. Theorem 11.1 reduces to

$$
Z \bot \{Y (1), Y (0) \} \mid e (X) = e _ {k} \quad (k = 1, \ldots , K).
$$

Therefore, we have a stratified randomized experiment (SRE), that is, we have K independent CREs within strata of the propensity score. We can analyze the observational data in the same way as the SRE stratified on $e ( X )$ .

In general, the propensity score is not known and is not discrete. We often fit a statistical model for $\operatorname { p r } ( Z \ = \ 1 \ | \ X )$ (for example, a logistic model) to obtain the estimated propensity score ${ \hat { e } } ( X )$ . This estimated propensity score can take as many values as the sample size, but we can discretize it to approximate the simple case above. For example, we can discretize the estimated propensity score by its K quantiles to obtain $\hat { e } ^ { \prime } ( X ) \colon \hat { e } ^ { \prime } ( X _ { i } ) = e _ { k } .$ , the $k / K { \mathrm { - t h } }$ quantile of ${ \hat { e } } ( X ) , { \mathrm { i f ~ } } { \hat { e } } ( X _ { i } )$ is between the $( k - 1 ) / K { \cdot } \mathrm { t h }$ and $k / K { \mathrm { - t h } }$ quantiles of ${ \hat { e } } ( X )$ . Then we have

$$
Z \bot \{Y (1), Y (0) \} \mid \hat {e} ^ {\prime} (X) = e _ {k} \quad (k = 1, \ldots , K).
$$

approximately. So we can analyze the observational data in the same way as the SRE stratified on $\hat { e } ^ { \prime } ( X )$ . The ignorability holds only approximately given $\hat { e } ^ { \prime } ( X )$ . We can further use regression adjustment based on covariate to remove bias and improve efficiency. To be more specific, we can obtain Lin (2013)’s estimator within each stratum and construct the final estimator by a weighted average.

With unknown propensity score, we need to fit a statistical model to obtain the estimated propensity score $\hat { e } ( X )$ . This makes the final estimator dependent on the model specification. However, the propensity score stratification estimator only requires the correct ordering of the estimated propensity scores rather than their exact values, which makes it relatively robust compared to other methods. This robustness property of propensity score stratification appeared in many numerical examples but its rigorous quantification is still missing in the literature.

An important practical question is how to choose K? If K is too small, then the strong ignorability does not hold even approximately given $\hat { e } ^ { \prime } ( X )$ . If K is too large, then we do not have enough units within each stratum of the estimated propensity score and many strata have only treated or control units. Therefore, we face a trade-off in practice. Following Cochran (1968)’s heuristics, Rosenbaum and Rubin (1983b) and Rosenbaum and Rubin (1984) suggested K = 5 which removes a large amount of bias in many settings. However, with extremely large dataset, propensity score stratification leads to biased estimators with a fixed K (Lunceford and Davidian, 2004). It is thus reasonable to increase K as long as each stratum has enough treated and control units. Wang et al. (2020) suggested an aggressive choice of K, which is the maximum number of strata such that the stratified estimator is well defined. But the rigorous theory for this procedure is not fully established.

Another important practical question is how to compute the standard errors of the estimators based on propensity score stratification? Some researcher conditioned on the discretized propensity scores ˆe′(X) and reported standard errors based on the SRE. This effectively ignored the uncertainty in the estimated propensity scores. Other researchers bootstrapped the whole procedure to account for full uncertainty. However, the theory for the bootstrap is still unclear due to the discreteness of this estimator.

## 11.1.3 Application

To illustrate the propensity score stratification method, I revisited Example 10.3. Figure 11.1 shows the histograms of the estimated propensity scores with different numbers of bins (K = 5, 10, 30).

Based on propensity score stratification, we can calculate the point estimators and the standard errors for difference choice of $K \in \{ 5 , 1 0 , 2 0 , 5 0 , 8 0 \}$ as follows (with the function NeymanSRE defined in Chapter 5 for analyzing the SRE):

```txt
> pscore = glm(z ~ x, family = binomial)$fitted.values
> n.strata = c(5, 10, 20, 50, 80)
> strat.res = sapply(n.strata, FUN = function(nn){
+    q.pscore = quantile(pscore, (1:(nn-1))/nn)
+    ps.strata = cut(pscore, breaks = c(0,q.pscore,1),
+    labels = 1:nn)
+    Neyman_SRE(z, y, ps.strata))
>
> rownames(strat.res) = c("est", "se")
> colnames(strat.res) = n.strata
> round(strat.res, 3)
5    10    20    50    80
```

$$
\begin{array}{c c c c c c} \text {est} & - 0. 1 1 6 & - 0. 1 7 8 & - 0. 2 0 0 & - 0. 2 6 5 & - 0. 2 0 4 \\ \text {se} & 0. 2 8 3 & 0. 2 8 2 & 0. 2 7 9 & 0. 2 7 2 & \text {NA} \end{array}
$$

Increasing K from 5 to 50 reduces the standard error. However, we cannot go as extreme as K = 80 because the standard error is not well-defined in some strata with only one treated or control unit. The above estimators show negative but insignificant effect of the meal program on the BMI.

We can also compare the above estimator with the three simple regression estimators: the one without adjusting for any covariates and Fisher and Lin’s estimators.

$$
\begin{array}{c c c c} & \text {naive} & \text {fisher} & \text {lin} \\ \text {est} & 0. 5 3 4 & 0. 0 6 1 & - 0. 0 1 7 \\ \text {se} & 0. 2 2 5 & 0. 2 2 7 & 0. 2 2 6 \end{array}
$$

The naive difference in means differ greatly from other methods. Although the point estimates are different, two regression estimators and the propensity score stratification estimators give qualitatively the same results. The propensity score stratification estimators are stable across different choices of K.

## 11.2 Propensity score weighting

## 11.2.1 Theory

Theorem 11.2 $I f Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$ and $0 < e ( X ) < 1$ , then

$$
E \{Y (1) \} = E \left\{\frac {Z Y}{e (X)} \right\}, \quad E \{Y (0) \} = E \left\{\frac {(1 - Z) Y}{1 - e (X)} \right\},
$$

and

$$
\tau = E \{Y (1) - Y (0) \} = E \left\{\frac {Z Y}{e (X)} - \frac {(1 - Z) Y}{1 - e (X)} \right\}.
$$

Before proving Theorem 11.2, it is important to note the additional assumption $0 < e ( X ) < 1$ . It is called the overlap or positivity condition. The formulas in Theorem 11.2 become infinity if $e ( X ) = 0 { \mathrm { ~ o r ~ } } 1$ for some values of X. It is not a restriction due to the identification formulas based on propensity score weighting. Although it was not stated explicitly in Theorem 10.1, the conditional expectations $E ( Y \mid Z = 1 , X )$ and $E ( Y \mid Z = 0 , X )$ in the identification formula of $\tau$ in (10.5) is well defined only if $0 < e ( X ) < 1$ . The overlap condition can be viewed as a technical condition to ensure that the formulas in Theorems 10.1 and 11.2 are well defined. It can also cause some philosophical issues for causal inference with observational studies. When unit i has $e ( X _ { i } ) = 1$ , we always observe its potential outcome under the treatment, $Y _ { i } ( 1 )$ , but can never observe its potential outcome under the control, $Y _ { i } ( 0 )$ . In this case, the potential outcome $Y _ { i } ( 0 )$ may not even be well defined, making the definition of the causal effect ambiguous for unit i. King and Zeng (2006) called $Y _ { i } ( 0 )$ an extreme counterfactual when $e ( X _ { i } ) = 1$ , and discussed their dangers in causal inference. A similar problem arises if unit i has $e ( X _ { i } ) = 0$ .

In sum, $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} | X$ requires adequate covariates to ensure the conditional independence of the treatment and potential outcomes, and $0 < e ( X ) < 1$ requires residual randomness in the treatment conditional on the covariates. In fact, Rosenbaum and Rubin (1983b)’s definition of strong ignorability includes both of these conditions. In the modern literature, they are often stated separately.

Proof of Theorem 11.2: I only prove the result for $E \{ Y ( 1 ) \}$ because theproof of the result for $E \{ Y ( 0 ) \}$ is similar. We have

$$
\begin{array}{l} E \left\{\frac {Z Y}{e (X)} \right\} \\ = E \left\{\frac {Z Y (1)}{e (X)} \right\} \\ = E \left[ E \left\{\frac {Z Y (1)}{e (X)} \mid X \right\} \right] \quad (\text { tower   property }) \\ = E \left[ \frac {1}{e (X)} E \{Z Y (1) \mid X \} \right] \\ = E \left[ \frac {1}{e (X)} E (Z \mid X) E \{Y (1) \mid X \} \right] \quad (\text { strong   ignorability }) \\ = E \left[ \frac {1}{e (X)} e (X) E \{Y (1) \mid X \} \right] \\ = E [ E \{Y (1) \mid X \} ] \\ = E \{Y (1) \}. \\ \end{array}
$$

## 11.2.2 Inverse propensity score weighting estimators

Theorem 11.2 implies the following moment estimator for the average causal effect:

$$
\hat {\tau} ^ {\mathrm{ht}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {Z _ {i} Y _ {i}}{\hat {e} (X _ {i})} - \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})},
$$

where $\hat { e } ( X _ { i } )$ is the estimated propensity score. This is the inverse propensity score weighting (IPW) estimator, which is also called the Horvitz–Thompson (HT) estimator. Horvitz and Thompson (1952) proposed it in survey sampling and Rosenbaum (1987a) used in causal inference with observational studies.

However, the estimator $\hat { \tau } ^ { \mathrm { h t } }$ has many problems. In particular, it is not invariant to location transformation of the outcome. For example, if we change $Y _ { i }$ to $Y _ { i } + c$ with a constant $c ,$ then it becomes $\hat { \tau } ^ { \mathrm { h t } } + c ( \hat { 1 } _ { \mathrm { T } } - \hat { 1 } _ { \mathrm { C } } )$ , where

$$
\hat {1} _ {\mathrm{T}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {Z _ {i}}{\hat {e} (X _ {i})}, \quad \hat {1} _ {\mathrm{C}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i})}{1 - \hat {e} (X _ {i})}
$$

are two different estimates of the constant 1. I use the funny notation $\hat { 1 } _ { \mathrm { T } }$ and $\mathrm { \hat { 1 } _ { C } }$ because with the true propensity score these two terms both have expectation 1; see Problem 11.3. In general, $\mathrm { \hat { 1 } _ { T } - \hat { 1 } _ { C } }$ is not zero in finite sample. Since adding a constant to every outcome should not change the average causal effect, this estimator is not reasonable because of its dependence on c. A simple fix to the problem is to normalize the weights by $\hat { 1 } _ { \mathrm { T } }$ and ˆ1C respectively, resulting in the following estimator

$$
\hat {\tau} ^ {\mathrm{hajek}} = \frac {\sum_ {i = 1} ^ {n} \frac {Z _ {i} Y _ {i}}{\hat {e} (X _ {i})}}{\sum_ {i = 1} ^ {n} \frac {Z _ {i}}{\hat {e} (X _ {i})}} - \frac {\sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})}}{\sum_ {i = 1} ^ {n} \frac {1 - Z _ {i}}{1 - \hat {e} (X _ {i})}}.
$$

This is the Hajek estimator due to H´ajek (1971). We can verify that the Hajek estimator is invariant to the location transformation, that is, if we replace $Y _ { i }$ by $Y _ { i } + c ,$ then $\hat { \tau } ^ { \mathrm { h a j e k } }$ remains the same. Moreover, many numerical studies have found that $\hat { \tau } ^ { \mathrm { h a j e k } }$ is much more stable than ${ \hat { \tau } } ^ { \mathrm { h t } }$ in finite samples.

## 11.2.3 A problem of weighting and a fundamental problem of causal inference

In many asymptotic analysis, we require a strong overlap condition

$$
0 <   \alpha_ {\mathrm{L}} \leq e (X) \leq \alpha_ {\mathrm{U}} <   1,
$$

that is, the true propensity score is bounded away from 0 and 1. However, D’Amour et al. (2021) pointed out that this is a rather strong assumption especially with many covariates. Chapter 20 will discuss this problem in detail.

Even if the strong overlap condition holds for the true propensity score, the estimated propensity scores can be close to 0 or 1. When this happens, the weighting estimators blow up to infinity resulting in extremely unstable behaviors in finite samples. We can either truncate the estimated propensity score by changing it to

$$
\max \left[ \alpha_ {\mathrm{L}}, \min \{\hat {e} (X _ {i}), \alpha_ {\mathrm{U}} \} \right],
$$

or trim the observations by dropping units with $\hat { e } ( X _ { i } )$ outside the interval $[ \alpha _ { \mathrm { L } } , \alpha _ { \mathrm { U } } ]$ . Crump et al. (2009) suggested $\alpha _ { \mathrm { L } } = 0 . 1$ and $\alpha _ { \mathrm { U } } = 0 . 9$ , and Kurth et al. (2005) suggested $\alpha _ { \mathrm { L } } ~ = ~ 0 . 0 5$ and $\alpha _ { \mathrm { U } } ~ = ~ 0 . 9 5$ . Yang and Ding (2018) established some asymptotic theory for trimming.

## 11.2.4 Application

Revisiting Example 10.3, we can obtain the weighting estimators based on different truncations of the the estimated propensity scores. The following results are the two weighting estimators with the bootstrap standard errors, with truncations at (0, 1), (0.01, 0.99), (0.05, 0.95), and (0.1, 0.9):

\$ trunc0

$$
\begin{array}{c c c} & \text {HT} & \text {Hajek} \\ \text {est} & - 1. 5 1 6 & - 0. 1 5 6 \\ \text {se} & 0. 4 9 5 & 0. 2 3 8 \end{array}
$$

## 11.3 The propensity score as a balancing score

<table><tr><td colspan="3">$trunc.01</td></tr><tr><td></td><td>HT</td><td>Hajek</td></tr><tr><td>est</td><td>-1.516</td><td>-0.156</td></tr><tr><td>se</td><td>0.464</td><td>0.231</td></tr></table>

<table><tr><td colspan="3">$trunc.05</td></tr><tr><td></td><td>HT</td><td>Hajek</td></tr><tr><td>est</td><td>-1.499</td><td>-0.152</td></tr><tr><td>se</td><td>0.472</td><td>0.248</td></tr></table>

<table><tr><td colspan="3">$trunc.1</td></tr><tr><td></td><td>HT</td><td>Hajek</td></tr><tr><td>est</td><td>-0.713</td><td>-0.054</td></tr><tr><td>se</td><td>0.435</td><td>0.229</td></tr></table>

The HT estimator gives results far away from all other estimators we discussed so far. The point estimates seem too large and they are negatively significant unless we truncate the estimated propensity scores at (0.1, 0.9). This is an example showing the instability of the HT estimator.

## 11.3 The propensity score as a balancing score

## 11.3.1 Theory

Theorem 11.3 The propensity score satisfies

$$
Z \bot X \mid e (X).
$$

Moreover, for any function h(·), we have

$$
E \left\{\frac {Z h (X)}{e (X)} \right\} = E \left\{\frac {(1 - Z) h (X)}{1 - e (X)} \right\} \tag {11.2}
$$

provided the existence of the moments on both sides of (11.2).

Rosenbaum and Rubin (1983b) also introduced the notion of balancing score b(X), which satisfies Z X | b(X). By Theorem 11.3, the propensity score is a balancing score. Theorem 11.3 also states that the any function h(X) of the covariates has the same mean across the treatment and control groups, if weighted by the inverse of the propensity score.

Moreover, Rosenbaum and Rubin (1983b) showed that the propensity score $e ( X )$ is the coarsest balancing score, that is, the propensity score $e ( X )$ is a function of any balancing score. Problem 11.5 gives more details.

Proof of Theorem 11.3: First, we show $Z \bot \bot X \mid e ( X )$ , that is,

$$
\operatorname{pr} \{Z = 1 \mid X, e (X) \} = \operatorname{pr} \{Z = 1 \mid e (X) \}. \tag {11.3}
$$

Following similar steps as the proof of Theorem 11.1, we can show that the left-hand side of (11.3) equals

$$
\operatorname{pr} \{Z = 1 \mid X, e (X) \} = \operatorname{pr} (Z = 1 \mid X) = e (X),
$$

and the right-hand side of (11.3) equals

$$
\begin{array}{l} \operatorname{pr} \{Z = 1 \mid e (X) \} = E \{Z \mid e (X) \} \\ = E \left[ E \{Z \mid X, e (X) \} \mid e (X) \right] \\ = E \left[ E \{Z \mid X \} \mid e (X) \right] \\ = E \left[ e (X) \mid e (X) \right] \\ = e (X). \\ \end{array}
$$

Therefore, (11.3) holds.

Second, we show (11.2). We can use similar steps as the proof of Theorem 11.1. But given Theorem 11.1, we have a simpler proof. If we view $h ( X )$ as an outcome, then its two potential outcomes are identical and the strong ignorability holds: Z h(X) | X. The difference between the the left-hand and right-hand sides of (11.2) is the average causal effect of Z on $h ( X )$ , which is zero. □

## 11.3.2 Covariate balance check

The proof of Theorem 11.3 is simple. But Theorem 11.3 has useful implications for the statistical analysis. Before getting access to the outcome data, we can check whether the propensity score model is specified well enough to ensure covariate balance in the data. Rubin (2007) viewed this as the design stage of the observational study, and Rubin (2008) argued that this can result in more objective causal inference because the design stage does not involve the values of the outcomes. While this is a useful recommendation in practice, it is not entirely clear how to quantify the objectiveness.

In propensity score stratification, we have the discretized estimated propensity score $\hat { e } ^ { \prime } ( X )$ and approximately

$$
Z \bot X \mid \hat {e} ^ {\prime} (X) = e _ {k} \quad (k = 1, \ldots , K).
$$

Therefore, we can check whether the covariate distributions are the same across the treatment and control groups within each stratum of the discretized estimated propensity score.

In propensity score weighting, we can view $h ( X )$ as a pseudo outcome and estimate the average causal effect on $h ( X )$ . Because the true average causal effect on $h ( X )$ is 0, the estimate should not be significantly different from 0. A canonical choice of $h ( X )$ is X .

Let us revisit Example 10.3 again. Based on propensity score stratification with $K = 5$ , all the covariates except FoodStamp are well balanced across the treatment and control groups. Similar result holds for the Hajek estimator. Figure 11.2 shows the balance checking results.

## 11.4 Homework Problems

## 11.1 Another version of Theorem 11.1

Prove that

$$
Z \bot \{Y (1), Y (0), X \} \mid e (X, Y (1), Y (0)).
$$

Remark: This result implies that

$$
Z \bot \{Y (1), Y (0) \} \mid \{X, e (X, Y (1), Y (0) \}.
$$

Rosenbaum (2020) and Rosenbaum and Rubin (2023) pointed out this result and called $e ( X , Y ( 1 ) , Y ( 0 ) )$ the principal unobserved covariate.

## 11.2 Another version of Theorem 11.1

If $Z \bot Y ( z ) \mid X$ for $z = 0 , 1$ , then $Z \underline { { | | Y ( z ) | } } \mid e ( X )$ for $z = 0 , 1$ . That is, if ignorability holds conditional on covariates X, then it also holds conditional on the scalar propensity score $e ( X )$ . Prove this theorem.

## 11.3 More results on the IPW estimators

This is related to the discussion of the IPW estimators in Section 11.2.2.

Prove

$$
E \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \frac {Z _ {i}}{e (X _ {i})} \right\} = 1, \quad E \left\{\frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i})}{1 - e (X _ {i})} \right\} = 1.
$$

## 11.4 Re-analysis of Rosenbaum and Rubin (1983a)

Use Table 1 of Rosenbaum and Rubin (1983a). If you are interested, you can read the whole paper. It is a canonical paper. But for this problem, you only need Table 1.

Rosenbaum and Rubin (1983a) fitted a logistic regression model for the propensity score and stratified the data into 5 subclasses. Because the treatment (Surgical versus Medical) is binary and the outcome is also binary (improved or not), they represented the data by a table.

Based on this table, estimate the average causal effect, and report the 95% confidence interval.

## 11.5 Balancing score and propensity score: more theoretical results

Rosenbaum and Rubin (1983b) defined $b ( X )$ as a balancing score if $Z \bot \bot X \ |$ b(X). Here, b(X) can be a scalar or a vector. An obvious balancing score is $b ( X ) = X$ , but it is not a useful one without any simplification of the original covariates. By Theorem 11.3, the propensity score is a special balancing score. More interestingly, Rosenbaum and Rubin (1983b) showed that the propensity score is the coarsest balancing score, as in Theorem 11.4 below which includes Theorem 11.3 as a special case.

Theorem 11.4 $b ( X )$ is a balancing score if and only if b(X) is finer than $e ( X )$ in the sense that $e ( X ) = f ( b ( X ) )$ for some function f (·).

Theorem 11.4 is relevant in subgroup analysis. In particular, we may be interested in not only the average causal effect τ but also the subgroup effects for boys and girls. Without loss of generality, assume the first component of X is the indicator for girls, and we can interested in estimating

$$
\tau (x _ {1}) = E \{Y (1) - Y (0) \mid X _ {1} = x _ {1} \}, \quad (x _ {1} = 1, 0).
$$

Theorem 11.4 implies that under ignorability,

$$
Z \bot \{Y (1), Y (0) \} \mid e (X), X _ {1} \tag {11.4}
$$

because $b ( X ) = \{ e ( X ) , X _ { 1 } \}$ is finer than $e ( X )$ and thus a balancing score. The conditional independence in (11.4) ensures ignorability holds given the propensity score, within each level of $X _ { 1 }$ . Therefore, we can perform the same analysis based on the propensity score, within each level of $X _ { 1 }$ , yielding estimates for two subgroup effects.

With the above motivation in mind, now prove Theorem 11.4.

## 11.6 Some basics of subgroup effects

This problem is related to Problem 11.5, but you can work on it independently.

Consider a standard observational study with covariates $\boldsymbol { X } = ( X _ { 1 } , X _ { 2 } )$ , where $X _ { 1 }$ denotes a binary subgroup indicator $( \mathrm { e . g . }$ , statistics major or not statistics major) and $X _ { 2 }$ contains the rest covariates. The parameter of interest is the subgroup causal effect

$$
\tau (x _ {1}) = E \{Y (1) - Y (0) \mid X _ {1} = x _ {1} \}, \quad (x _ {1} = 1, 0).
$$

Show that

$$
\tau (x _ {1}) = E \left\{\frac {1 (X _ {1} = x _ {1}) Z Y}{e (X)} - \frac {1 (X _ {1} = x _ {1}) (1 - Z) Y}{1 - e (X)} \right\} / \operatorname{pr} (X _ {1} = x _ {1})
$$

and give the corresponding Horvitz–Thompson and Hajek estimators for $\tau ( x _ { 1 } )$ ).

## 11.7 Recommended reading

The title of this chapter is the same as the title of the classic paper by Rosenbaum and Rubin (1983b). Most results in this chapter are directly drawn from their original paper.

Rubin (2007) and Rubin (2008) highlighted the importance of the design stage of observational studies for more objective causal inference

## 12

# The Doubly Robust or the Augmented Inverse Propensity Score Weighting Estimator for the Average Causal Effect

Under unconfoundedness $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$ and overlap $0 < e ( X ) < 1$ , Chapter 11 has shown two identification formulas of the average causal effect $\tau = E \{ Y ( 1 ) - Y ( 0 ) \}$ . First, the outcome imputation formula is

$$
\tau = E \{\mu_ {1} (X) \} - E \{\mu_ {0} (X) \} \tag {12.1}
$$

where

$$
\mu_ {1} (X) = E \{Y (1) \mid X \} = E (Y \mid Z = 1, X),
$$

$$
\mu_ {0} (X) = E \{Y (0) \mid X \} = E (Y \mid Z = 0, X)
$$

are the two conditional mean functions of the outcome given covariates. Second, the inverse propensity score weighting (IPW) formula is

$$
\tau = E \left\{\frac {Z Y}{e (X)} \right\} - E \left\{\frac {(1 - Z) Y}{1 - e (X)} \right\} \tag {12.2}
$$

where

$$
e (X) = \operatorname{pr} (Z = 1 \mid X)
$$

is the propensity score introduced in Chapter 11.

The outcome imputation estimator requires fitting a model for the outcome given the treatment and covariates. It is consistent if the outcome model is correctly specified. The IPW estimator requires fitting a model for the treatment given covariates. It is consistent if the propensity score model is correctly specified.

Mathematically, we have many combinations of (12.1) and (12.2) that lead to different identification formulas of the average causal effect. Below I will discuss a particular combination that has appealing theoretical properties. This combination motivates an estimator that is consistent if either the propensity score or the outcome model is correctly specified. It is call the doubly robust estimator, championed by James Robins (Scharfstein et al., 1999; Bang and Robins, 2005).

## 12.1 The doubly robust estimator

## 12.1.1 Population version

We posit a working model for the conditional means of the outcome $\mu _ { 1 } ( X , \beta _ { 1 } )$ and $\mu _ { 0 } ( X , \beta _ { 0 } )$ , indexed by the parameters $\beta _ { 1 }$ and $\beta _ { 0 }$ . For example, if the conditional means are linear or logistic under the working model, then the parameters are just the regression coefficients. If the outcome model is correctly specified, then $\mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X )$ and $\mu _ { 0 } ( X , \beta _ { 0 } ) = \mu _ { 0 } ( X )$ . We posit a working model for the propensity score $e ( X , \alpha )$ , indexed by the parameter α. For example, if the working model is logistic, then α is the regression coefficient. If the propensity score model is correctly specified, then $e ( X , \alpha ) = e ( X )$ . In practice, both models may be misspecified.

Define

$$
\tilde {\mu} _ {1} ^ {\mathrm{dr}} = E \left[ \frac {Z \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X , \alpha)} + \mu_ {1} (X, \beta_ {1}) \right], \tag {12.3}
$$

$$
\tilde {\mu} _ {0} ^ {\mathrm{dr}} = E \left[ \frac {(1 - Z) \{Y - \mu_ {0} (X , \beta_ {0}) \}}{1 - e (X , \alpha)} + \mu_ {0} (X, \beta_ {0}) \right], \tag {12.4}
$$

which can also be written as

$$
\tilde {\mu} _ {1} ^ {\mathrm{dr}} = E \left[ \frac {Z Y}{e (X , \alpha)} - \frac {Z - e (X , \alpha)}{e (X , \alpha)} \mu_ {1} (X, \beta_ {1}) \right], \tag {12.5}
$$

$$
\tilde {\mu} _ {0} ^ {\mathrm{dr}} = E \left[ \frac {(1 - Z) Y}{1 - e (X , \alpha)} - \frac {e (X , \alpha) - Z}{1 - e (X , \alpha)} \mu_ {0} (X, \beta_ {0}) \right]. \tag {12.6}
$$

The formulas in (12.3) and (12.4) augment the outcome imputation estimator by inverse propensity score weighting terms of the residuals. The formulas in (12.5) and (12.6) augment the IPW estimator by the imputed outcomes. For this reason, the doubly robust estimator is also called the augmented inverse propensity score weighting (AIPW) estimator.

The augmentation strengthens the theoretical properties in the following sense.

Theorem 12.1 Assume unconfoundedness Z $\{ Y ( 1 ) , Y ( 0 ) \} \mid X$ and overlap $1 < e ( X ) < 1$ .

$\begin{array} { r c l } { { } } & { { I . ~ I f ~ e i t h e r ~ e ( X , \alpha ) = ~ e ( X ) ~ o r ~ \mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X ) , ~ t h e n ~ \tilde { \mu } _ { 1 } ^ { \mathrm { d r } } = } } \\ { { } } & { { } } & { { E \{ Y ( 1 ) \} . } } \end{array}$  
2. If either e(X, α) = e(X) or µ0(X, β0) = µ0(X), then $\tilde { \mu } _ { 0 } ^ { \mathrm { d r } } =$ E{Y (0)}.  
$\begin{array} { r l } & { \beta . \mathrm { ~ } J f \mathrm { ~ } e i t h e r \mathrm { ~ } e ( X , \alpha ) = e ( X ) \mathrm { ~ } o r \mathrm { ~ } \{ \mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X ) , \mu _ { 0 } ( X , \beta _ { 0 } ) = } \\ & { \mu _ { 0 } ( X ) \} , \mathrm { ~ } t h e n \tilde { \mu } _ { 1 } ^ { \mathrm { { d r } } } - \tilde { \mu } _ { 0 } ^ { \mathrm { { d r } } } = \tau . } \end{array}$

By Theorem 12.1, $\tilde { \mu } _ { 1 } ^ { \mathrm { d r } } - \tilde { \mu } _ { 0 } ^ { \mathrm { d r } }$ equals τ if either the propensity score model or the outcome model is correctly specified. That’s why it is called the doubly robust estimator.

Proof of Theorem 12.1: I only prove the result for $\mu _ { 1 } = E \{ Y ( 1 ) \}$ . The proof for the result for $\mu _ { 0 } = E \{ Y ( 0 ) \}$ is similar. We have the decomposition

$$
\begin{array}{l} \tilde {\mu} _ {1} ^ {\mathrm{dr}} - E \{Y (1) \} = E \left[ \frac {Z \{Y (1) - \mu_ {1} (X , \beta_ {1}) \}}{e (X , \alpha)} - \{Y (1) - \mu_ {1} (X, \beta_ {1}) \} \right] \\ = E \left[ \frac {Z - e (X , \alpha)}{e (X , \alpha)} \{Y (1) - \mu_ {1} (X, \beta_ {1}) \} \right] \\ = E \left(E \left[ \frac {Z - e (X , \alpha)}{e (X , \alpha)} \{Y (1) - \mu_ {1} (X, \beta_ {1}) \} \mid X \right]\right) \\ = E \left[ E \left\{\frac {Z - e (X , \alpha)}{e (X , \alpha)} \mid X \right\} \times E \left\{Y (1) - \mu_ {1} (X, \beta_ {1}) \mid X \right\} \right] \\ = E \left[ \frac {e (X) - e (X , \alpha)}{e (X , \alpha)} \times \{\mu_ {1} (X) - \mu_ {1} (X, \beta_ {1}) \} \right]. \\ \end{array}
$$

Therefore, $\tilde { \mu } _ { 1 } ^ { \mathrm { d r } } - E \{ Y ( 1 ) \} = 0$ if either $e ( X , \alpha ) = e ( X ) { \mathrm { o r } } \mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X )$

## 12.1.2 Sample version

From the population versions of $\tilde { \mu } _ { 1 } ^ { \mathrm { d r } }$ and $\tilde { \mu } _ { 0 } ^ { \mathrm { d r } }$ , we can construct the sample versions by the following steps:

1. obtain the fitted values of the propensity scores: $e ( X , { \hat { \alpha } } )$ ;  
2. obtain the fitted values of the outcome means: $\mu _ { 1 } ( X , { \hat { \beta } } _ { 1 } )$ and $\mu _ { 0 } ( X , { \hat { \beta } } _ { 0 } )$ ;  
3. construct the doubly robust estimator: $\hat { \tau } ^ { \mathrm { d r } } = \hat { \mu } _ { 1 } ^ { \mathrm { d r } } - \hat { \mu } _ { 0 } ^ { \mathrm { d r } }$ , where

$$
\hat {\mu} _ {1} ^ {\mathrm{dr}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left[ \frac {Z _ {i} \{Y _ {i} - \mu_ {1} (X _ {i} , \hat {\beta} _ {1}) \}}{e (X _ {i} , \hat {\alpha})} + \mu_ {1} (X _ {i}, \hat {\beta} _ {1}) \right]
$$

and

$$
\hat {\mu} _ {0} ^ {\mathrm{dr}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left[ \frac {(1 - Z _ {i}) \{Y _ {i} - \mu_ {0} (X _ {i} , \hat {\beta} _ {0}) \}}{1 - e (X _ {i} , \hat {\alpha})} + \mu_ {0} (X _ {i}, \hat {\beta} _ {0}) \right];
$$

4. approximate the variance of ${ \hat { \tau } } ^ { \mathrm { d r } }$ via the nonparametric bootstrap by resampling from $( Z _ { i } , X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ (Funk et al., 2011).

Analogous to (12.5) and (12.6), we can also rewrite $\hat { \mu } _ { 1 } ^ { \mathrm { d r } }$ and $\hat { \mu } _ { 0 } ^ { \mathrm { d r } }$ as

$$
\hat {\mu} _ {1} ^ {\mathrm{dr}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left[ \frac {Z _ {i} Y _ {i}}{e (X _ {i} , \hat {\alpha})} - \frac {Z _ {i} - e (X _ {i} , \hat {\alpha})}{e (X _ {i} , \hat {\alpha})} \mu_ {1} (X _ {i}, \hat {\beta} _ {1}) \right],
$$

$$
\hat {\mu} _ {0} ^ {\mathrm{dr}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left[ \frac {(1 - Z _ {i}) Y _ {i}}{1 - e (X _ {i} , \hat {\alpha})} - \frac {e (X _ {i} , \hat {\alpha}) - Z _ {i}}{1 - e (X _ {i} , \hat {\alpha})} \mu_ {0} (X _ {i}, \hat {\beta} _ {0}) \right].
$$

## 12.2 More intuition and theory for the doubly robust estimator

Although the beginning of this chapter claims that the basic identification formulas based on outcome regression and inverse propensity score weight immediately yield infinitely many other identification formulas, the particular forms of the double robust estimators in (12.3) and (12.4) are not obvious to come up with. The original motivation for (12.3) and (12.4) was quite theoretical, which relies on the semiparametric efficiency theory in advanced mathematical statistics (Bickel et al., 1993). It is beyond the level of this book. Below I will give two more intuitive perspectives to construct (12.3) and (12.4). Both Sections 12.2.1 and 12.2.2 below focus on the estimation of $E \{ Y ( 1 ) \}$ since the estimation of $E \{ Y ( 0 ) \}$ is similar by symmetry.

## 12.2.1 Reducing the variance of the IPW estimator

The IPW estimator for $\mu _ { 1 }$ based on

$$
\mu_ {1} = E \left\{\frac {Z Y}{e (X)} \right\}
$$

completely ignores the outcome model of Y . It has the advantages of being consistent without assuming any outcome model. However, if the covariates are predictive to the outcome, the residual based on a working outcome model usually has smaller variance than the outcome even if this working outcome model is wrong. With a possibly mis-specified outcome model $\mu _ { 1 } ( X , \beta _ { 1 } )$ , a trivial decomposition holds:

$$
\mu_ {1} = E \{Y (1) \} = E \{Y (1) - \mu_ {1} (X, \beta_ {1}) \} + E \{\mu_ {1} (X, \beta_ {1}) \}.
$$

If we apply the IPW formula to the first term in the above formula viewing $Y ( 1 ) - \mu _ { 1 } ( X , \beta _ { 1 } )$ as a pseudo potential outcome under the treatment, we can rewrite the above formula as

$$
\mu_ {1} = E \left\{\frac {Z \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X)} \right\} + E \{\mu_ {1} (X, \beta_ {1}) \} \tag {12.7}
$$

$$
= E \left\{\frac {Z \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X)} + \mu_ {1} (X, \beta_ {1}) \right\}, \tag {12.8}
$$

which holds if the propensity score model is correct without assuming that the outcome model is correct. Using a working model to improve efficiency is an old idea from survey sampling. Little and An (2004) and Lumley et al. (2011) pointed out its connection with the doubly robust estimator.

## 12.2.2 Reducing the bias of the outcome regression estimator

The discussion in Section 12.2.1 starts with the IPW estimator and improves its efficiency based on a working outcome model. Alternatively, we can also start with an outcome regression estimator based on

$$
\tilde {\mu} _ {1} = E \{\mu_ {1} (X, \beta_ {1}) \}
$$

which may not be the same as $\mu _ { 1 }$ since the outcome may be wrong. The bias of this estimator is $E \{ \mu _ { 1 } ( X , \beta _ { 1 } ) - Y ( 1 ) \}$ , which can be estimated by an IPW estimator

$$
B = E \left\{\frac {Z \{\mu_ {1} (X , \beta_ {1}) - Y \}}{e (X)} \right\}
$$

if the propensity score model is correct. So a de-biased estimator is $\tilde { \mu } _ { 1 } - B$ , which is identical to (12.8).

## 12.3 Examples

## 12.3.1 Summary of some canonical estimators for τ

The following R implements the outcome imputation, Hovitz–Thompson, Hajek, and doubly robust estimators for τ . These estimators can be conveniently implemented based on the fitted values of the glm function. The default choice for the propensity score model is the logistic model, and the default choice for the outcome model is the linear model with out.family = gaussian1. For binary outcomes, we can also specify out.family = binomial to fit the logistic model.

```txt
OS_est = function(z, y, x, out.family = gaussian,
    truncpscore = c(0, 1))
{
    ## fitted propensity score
    pscore = glm(z ~ x, family = binomial)$fitted.values
    pscore = pmax(truncpscore[1], pmin(truncpscore[2], pscore))
```

```r
## fitted potential outcomes
outcome1 = glm(y ~ x, weights = z,
    family = out.family)$fitted.values
outcome0 = glm(y ~ x, weights = (1 - z),
    family = out.family)$fitted.values

## regression imputation estimator
ace.reg = mean(outcome1 - outcome0)
## IPW estimators
ace.ipw0 = mean(z*y/pscore - (1 - z)*y/(1 - pscore))
ace.ipw = mean(z*y/pscore)/mean(z/pscore) -
    mean((1 - z)*y/(1 - pscore))/mean((1 - z)/(1 - pscore))
## doubly robust estimator
res1 = y - outcome1
res0 = y - outcome0
ace.dr = ace.reg + mean(z*res1/pscore - (1 - z)*res0/(1 - pscore))

return(c(ace.reg, ace.ipw0, ace.ipw, ace.dr))
}
```

It is tedious to calculate the analytic formulas for the variances of the above estimators. The bootstrap provides convenient approximations for the variances based on resampling from $\{ Z _ { i } , X _ { i } , Y _ { i } \} _ { i = 1 } ^ { n }$ . Building upon OSest, the following function returns point estimators as well as the bootstrap standard errors.

```r
OS_ATE = function(z, y, x, n.boot = 2*10^2,
    out.family = gaussian, truncpscore = c(0, 1))
{
    point.est = OS_est(z, y, x, out.family, truncpscore)

    ## nonparametric bootstrap
    n.sample = length(z)
    x = as.matrix(x)
    boot.est = replicate(n.boot,
    {id.boot = sample(1:n.sample, n.sample, replace = TRUE)
    OS_est(z[id.boot], y[id.boot], x[id.boot, ],
    out.family, truncpscore)})
    boot.se = apply(boot.est, 1, sd)

    res = rbind(point.est, boot.se)
    rownames(res) = c("est", "se")
    colnames(res) = c("reg", "HT", "Hajek", "DR")

    return(res)
}
```

## 12.3.2 Simulation

I will use simulation to evaluate the finite-sample properties of the estimators under four scenarios:

1. both the propensity score and outcome models are correct;  
2. the propensity score model is wrong but the outcome model is correct;  
3. the propensity score model is correct but the outcome model is wrong;  
4. both the propensity score and outcome models are wrong.

I will report the average bias, the true standard error, and the average estimated standard error of the estimators over simulation.

In case 1, the data generating process is

```matlab
x = matrix(rnorm(n*2), n, 2)
x1 = cbind(1, x)
beta.z = c(0, 1, 1)
pscore = 1/(1 + exp(- as.vector(x1%* %beta.z)))
z = rbinom(n, 1, pscore)
beta.y1 = c(1, 2, 1)
beta.y0 = c(1, 2, 1)
y1 = rnorm(n, x1%* %beta.y1)
y0 = rnorm(n, x1%* %beta.y0)
y = z*y1 + (1 - z)*y0
```

In case 2, I modify the propensity score model to be nonlinear:

```txt
x1 = cbind(1, x, exp(x))
beta.z = c(-1, 0, 0, 1, -1)
pscore = 1/(1 + exp(- as.vector(x1%* % beta.z)))
```

In case 3, I modify the outcome model to be nonlinear:

```txt
beta.y1 = c(1, 0, 0, 0.2, -0.1)
beta.y0 = c(1, 0, 0, -0.2, 0.1)
y1 = rnorm(n, x1%* %beta.y1)
y0 = rnorm(n, x1%* %beta.y0)
```

In case 4, I modify both the propensity score and the outcome model.

We set the sample size to be n = 500 and generate 500 independent data sets according to the data generating processes above. In case 1,

```batch
reg HT Hajek DR
ave.bias 0.00 0.02 0.03 0.01
true.se 0.11 0.28 0.26 0.13
est.se 0.10 0.25 0.23 0.12
```

All estimators are nearly unbiased. The two weighting estimators have larger variances. In case 2,

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>ave.bias</td><td>0.00</td><td>-0.76</td><td>-0.75</td><td>-0.01</td></tr><tr><td>true.se</td><td>0.12</td><td>0.59</td><td>0.47</td><td>0.18</td></tr><tr><td>est.se</td><td>0.13</td><td>0.50</td><td>0.38</td><td>0.18</td></tr></table>

The two weighting estimators are severely biased due to the misspecification of the propensity score model. The regression imputation and doubly robust estimators are nearly unbiased. In case 3,

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>ave.bias</td><td>-0.05</td><td>0.00</td><td>-0.01</td><td>0.00</td></tr><tr><td>true.se</td><td>0.11</td><td>0.15</td><td>0.14</td><td>0.14</td></tr><tr><td>est.se</td><td>0.11</td><td>0.14</td><td>0.13</td><td>0.14</td></tr></table>

The regression imputation estimator has larger bias than the other three estimators due to the misspecification of the outcome model. The weighting and doubly robust estimators are nearly unbiased. In case 4,

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>ave.bias</td><td>-0.08</td><td>0.11</td><td>-0.07</td><td>0.16</td></tr><tr><td>true.se</td><td>0.13</td><td>0.32</td><td>0.20</td><td>0.41</td></tr><tr><td>est.se</td><td>0.13</td><td>0.25</td><td>0.16</td><td>0.26</td></tr></table>

All estimators are biased because both the propensity score and outcome models are wrong. The Horvitz–Thompson and doubly robust estimator has the largest bias. When both models are wrong, the doubly robust estimator appears to be doubly fragile.

In all the cases above, the boostrap standard errors are close to the true ones when the estimators are nearly unbiased for the true average causal effect.

## 12.3.3 Applications

Revisiting Example 10.3, we obtain the following estimators and bootstrap standard errors:

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>est</td><td>-0.017</td><td>-1.516</td><td>-0.156</td><td>-0.019</td></tr><tr><td>se</td><td>0.230</td><td>0.492</td><td>0.246</td><td>0.233</td></tr></table>

The two weighting estimators are much larger than the other two estimators. Truncating the estimated propensity score at [0.1, 0.9], we obtain the following estimators and bootstrap standard errors:

<table><tr><td></td><td>reg</td><td>HT</td><td>Hajek</td><td>DR</td></tr><tr><td>est</td><td>-0.017</td><td>-0.713</td><td>-0.054</td><td>-0.043</td></tr><tr><td>se</td><td>0.223</td><td>0.422</td><td>0.235</td><td>0.231</td></tr></table>

The Hajek estimator becomes much close to the regression imputation and doubly robust estimators, while the Horvitz–Thompson estimator is still an outlier.

## 12.4 Some further discussion

Recall the proof of Theorem 12.1, the key for the double robustness property is the product structure in

$$
\tilde {\mu} _ {1} ^ {\mathrm{dr}} - E \{Y (1) \} = E \left[ \frac {e (X) - e (X , \alpha)}{e (X , \alpha)} \times \{\mu_ {1} (X) - \mu_ {1} (X, \beta_ {1}) \} \right],
$$

which ensures that the estimation error is zero if either $e ( X ) = e ( X , \alpha )$ or $\mu _ { 1 } ( X ) = \mu _ { 1 } ( X , \beta _ { 1 } )$ . This delicate structure renders the doubly robust estimator possibly doubly fragile when both the propensity score and the outcome models are misspecified. The product of two errors multiply to yield potentially much larger errors. Kang and Schafer (2007) criticized the doubly robust estimator based on extensive simulation studies. They found that the finitesample performance of the doubly robust estimator can be even more wild than the simple regression imputation and IPW estimators.

Despite the critique from Kang and Schafer (2007), the doubly robust estimator has been a standard strategy in causal since the seminal work of Scharfstein et al. (1999). Recently, it resurrected in the theoretical statistics and econometrics literature with a fancier name “double machine learning” (Chernozhukov et al., 2018). The basic idea is to replace the working models for the propensity score and outcome by machine learning tools which can be viewed as more flexible models than the traditional parametric models.

## 12.5 Homework problems

## 12.1 A sanity check

Consider the case in which the covariate is discrete $X ~ \in ~ \{ 1 , \ldots , K \}$ and the parameter of interest is $\mu _ { 1 }$ . Without imposing any model assumptions, the estimated propensity score $\hat { e } ( X )$ is the proportion of units receiving the treatment and the estimated outcome mean is the sample mean of the outcome $\hat { \bar { Y } } _ { [ k ] 1 } ~ = ~ \hat { E } ( Y ~ \vert ~ Z ~ = ~ 1 , X ~ = ~ k )$ under treatment, within stratum $X = k \ ( k \stackrel { \cdot } { = } 1 , \ldots , K )$ . Show that the stratified estimator, outcome regression estimator, IPW estimator, and the doubly robust estimator are all the same.

## 12.2 An alternative form of the doubly robust estimator for τ

Motivated by (12.7), we have an alternative form of doubly robust estimator for $\mu _ { 1 }$ :

$$
\tilde {\mu} _ {1} ^ {\mathrm{dr2}} = \frac {E \left[ \frac {Z \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X , \alpha)} \right]}{E \left[ \frac {Z}{e (X , \alpha)} \right]} + E \{\mu_ {1} (X, \beta_ {1}) \}.
$$

Show that $\tilde { \mu } _ { 1 } ^ { \mathrm { d r 2 } } = \mu _ { 1 }$ if either $e ( X , \alpha ) = e ( X )$ or $\mu _ { 1 } ( X , \beta _ { 1 } ) = \mu _ { 1 } ( X )$ ). Give the analogous formula for estimating µ0. Give the sample analogue of the doubly robust estimator for τ based on these formulas. Note that this form of doubly robust estimator appeared in Robins et al. (2007).

## 12.3 Data analysis of Example 10.1

Analyze the dataset cps1re74.csv using the methods discussed so far.

## 12.4 Recommended reading

Lunceford and Davidian (2004) gave a nice review and comparison of many methods discussed in Chapters 11 and 12.

## 13

# The Average Causal Effect on the Treated Units and Other Estimands

Chapters 10–12 focused on the identification and estimation of the average causal effect $\tau = E \{ Y ( 1 ) - Y ( 0 ) \}$ under the unconfoundedness and overlap assumptions. Conceptually, it is straightforward to extend the discussion to the average causal effects on the treated and control units:

$$
\tau_ {\mathrm{T}} = E \{Y (1) - Y (0) \mid Z = 1 \},
$$

$$
\tau_ {\mathrm{C}} = E \{Y (1) - Y (0) \mid Z = 0 \}.
$$

Because of the symmetry, this chapter focuses on $\tau _ { \mathrm { T } }$ and also included extensions to other estimands.

## 13.1 Nonparametric identification of $\tau _ { \mathbf { T } }$

The average causal effect on the treated units equals

$$
\tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E \{Y (0) \mid Z = 1 \},
$$

where the first term $E ( Y \mid Z = 1 )$ is directly identifiable from the data and the second term $E \{ Y ( 0 ) ~ | ~ Z = 1 \}$ is counterfactual. The key assumption to identify the second term is the following unconfoundedness and overlap assumptions.

Assumption 13.1 $Z \underline { { \mathrm { 1 1 } } } Y ( 0 ) \mid X$ and $e ( X ) < 1$ .

Because the key is to identify $E \{ Y ( 0 ) \mid Z = 1 \}$ , we only need the $^ { 6 6 } \mathrm { o n e - }$ - sided” unconfoundedness and overlap assumptions. Under Assumption 13.1, we have the following identification result for $\tau _ { \mathrm { T } }$ .

Theorem 13.1 Under Assumption 13.1, we have

$$
\begin{array}{l} E \{Y (0) \mid Z = 1 \} = E \left\{E (Y \mid Z = 0, X) \mid Z = 1 \right\} \\ = \int E (Y \mid Z = 0, X = x) F (\mathrm{d} x \mid Z = 1). \\ \end{array}
$$

16413 The Average Causal Effect on the Treated Units and Other Estimands

Theorem 13.1 implies that $\tau _ { \mathrm { T } }$ is nonparmetrically identified by

$$
\tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E \left\{E (Y \mid Z = 0, X) \mid Z = 1 \right\} \tag {13.1}
$$

Proof of Theorem 13.1: We have

$$
\begin{array}{l} E \{Y (0) \mid Z = 1 \} = E \left[ E \{Y (0) \mid Z = 1, X \} \mid Z = 1 \right] \\ = E \left[ E \{Y (0) \mid Z = 0, X \} \mid Z = 1 \right] \\ = E \left\{E (Y \mid Z = 0, X) \mid Z = 1 \right\} \\ = \int E (Y \mid Z = 0, X = x) F (\mathrm{d} x \mid Z = 1). \\ \end{array}
$$

![image_11](images/image_11.png)

With a discrete X, the identification formula in Theorem 13.1 reduces to

$$
E \{Y (0) \mid Z = 1 \} = \sum_ {k = 1} ^ {K} E (Y \mid Z = 0, X = k) \mathrm{pr} (X = k \mid Z = 1),
$$

motivating the following stratified estimator for $\tau _ { \mathrm { T } }$

$$
\hat {\tau} _ {\mathrm{T}} = \hat {\bar {Y}} (1) - \sum_ {k = 1} ^ {K} \hat {\pi} _ {[ k ] | 1} \hat {\bar {Y}} _ {[ k ]} (0),
$$

where $\hat { \pi } _ { [ k ] | 1 } = n _ { [ k ] 1 } / n _ { 1 }$ is the proportion of category k of X among the treated units.

For continuous X, we need to fit an outcome model for $E ( Y \mid Z = 0 , X )$ 号 using the control units. If the fitted values for the control potential outcomes are $\hat { \mu } _ { 0 } ( X _ { i } )$ , then the outcome regression estimator is

$$
\hat {\tau} _ {\mathrm{T}} = \hat {\bar {Y}} (1) - n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \hat {\mu} _ {0} (X _ {i}) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \{Y _ {i} - \hat {\mu} _ {0} (X _ {i}) \}.
$$

Example 13.1 If we specify a linear model for all units

$$
E (Y \mid Z, X) = \beta_ {0} + \beta_ {z} Z + \beta_ {x} ^ {\mathsf {T}} X,
$$

then

$$
\begin{array}{l} \tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E (\beta_ {0} + \beta_ {x} ^ {\mathsf {T}} X \mid Z = 1) \\ = E (Y \mid Z = 1) - \beta_ {0} - \beta_ {x} ^ {\mathsf {T}} E (X \mid Z = 1). \\ \end{array}
$$

$I f$ we run OLS to obtain $( \hat { \beta } _ { 0 } , \hat { \beta } _ { z } , \hat { \beta } _ { x } )$ , then the estimator is

$$
\hat {\tau} _ {\mathrm{T}} = \hat {\bar {Y}} (1) - \hat {\beta} _ {0} - \hat {\beta} _ {x} ^ {\mathsf {T}} \hat {\bar {X}} (1).
$$

Using the property of the OLS (see A2.3), we have

$$
\sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\beta} _ {0} - \hat {\beta} _ {z} Z _ {i} - \hat {\beta} _ {x} ^ {\mathsf {T}} X _ {i}) = 0 \Longrightarrow \hat {\bar {Y}} (1) - \hat {\beta} _ {0} - \hat {\beta} _ {z} - \hat {\beta} _ {x} ^ {\mathsf {T}} \hat {\bar {X}} (1) = 0.
$$

Therefore, the above estimator reduces to $\hat { \tau } _ { \mathrm { T } } = \hat { \beta } _ { z } ,$ , the OLS coefficient of Z.

By the property of the OLS, we can also write $\hat { \beta } _ { z }$ as the difference in means of the adjusted outcome $Y _ { i } - \hat { \beta } _ { x } ^ { \sf T } X _ { i }$ , resulting in

$$
\begin{array}{l} \hat {\tau} _ {\mathrm{T}} = \left\{\hat {\bar {Y}} (1) - \hat {\beta} _ {x} ^ {\mathsf {T}} \hat {\bar {X}} (1) \right\} - \left\{\hat {\bar {Y}} (0) - \hat {\beta} _ {x} ^ {\mathsf {T}} \hat {\bar {X}} (0) \right\} \\ = \left\{\hat {\bar {Y}} (1) - \hat {\bar {Y}} (0) \right\} - \hat {\beta} _ {x} ^ {\mathsf {T}} \left\{\hat {\bar {X}} (1) - \hat {\bar {X}} (0) \right\}. \tag {13.2} \\ \end{array}
$$

Therefore, τˆT equals the simple difference in means of the outcome, adjusted by the imbalance of the covariates in the treatment and control groups.

Section $\it 1 0 . 4 . 2$ shows that $\hat { \beta } _ { z }$ is an estimator for τ , and this example further shows that $\hat { \beta } _ { z }$ is an estimator for $\tau _ { \mathrm { T } }$ . This is not surprising because the linear model assumes constant causal effects across units.

Example 13.2 The identification formula depends only on $E ( Y \mid Z = 0 , X )$ , so we need only to specify a model for the control units. When this model is linear,

$$
E (Y \mid Z = 0, X) = \beta_ {0 | 0} + \beta_ {x | 0} ^ {\mathsf {T}} X,
$$

we have

$$
\begin{array}{l} \tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E (\beta_ {0 | 0} + \beta_ {x | 0} ^ {\mathsf {T}} X \mid Z = 1) \\ = E (Y \mid Z = 1) - \beta_ {0 | 0} - \beta_ {x | 0} ^ {\mathsf {T}} E (X \mid Z = 1). \\ \end{array}
$$

If we run OLS with only the control units to obtain $( \hat { \beta } _ { 0 | 0 } , \hat { \beta } _ { x | 0 } )$ , then the estimator is

$$
\hat {\tau} _ {\mathrm{T}} = \hat {\bar {Y}} (1) - \hat {\beta} _ {0 | 0} - \hat {\beta} _ {x | 0} ^ {\mathsf {T}} \hat {\bar {X}} (1).
$$

Using the property of the OLS (see A2.3), we have

$$
\hat {\bar {Y}} (0) = \hat {\beta} _ {0 | 0} + \hat {\beta} _ {x | 0} ^ {\mathsf {T}} \hat {\bar {X}} (0).
$$

Therefore, the above estimator reduces to

$$
\hat {\tau} _ {\mathrm{T}} = \left\{\hat {\bar {Y}} (1) - \hat {\bar {Y}} (0) \right\} - \hat {\beta} _ {x | 0} ^ {\mathsf {T}} \left\{\hat {\bar {X}} (1) - \hat {\bar {X}} (0) \right\},
$$

which is similar to (13.2) with a different coefficient for the difference in means of the covariates.

As an algebraic fact, we can show that this estimator equals the coefficient of Z in the OLS fit of the outcome on the treatment, covariates, and their interactions, with the covariates centered $b y \hat { \bar { X } } ( 1 )$ . See Problem 13.1 for more details.

## 13.2 Inverse propensity score weighting and doubly robust estimation of $\tau_{\mathbf{T}}$

Theorem 13.2 Under Assumption 13.1, we have

$$
E \{Y (0) \mid Z = 1 \} = E \left\{\frac {e (X)}{e} \frac {1 - Z}{1 - e (X)} Y \right\} \tag {13.3}
$$

and

$$
\tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E \left\{\frac {e (X)}{e} \frac {1 - Z}{1 - e (X)} Y \right\}, \tag {13.4}
$$

where $e = \operatorname { p r } ( Z = 1 )$ is the marginal probability of the treatment.

Proof of Theorem 13.2: The left-hand side of (13.3) equals

$$
\begin{array}{l} E \{Y (0) \mid Z = 1 \} = E \{Z Y (0) \} / e \\ = E \left[ E (Z \mid X) E \{Y (0) \mid X \} \right] / e \\ = E \left[ e (X) E \{Y (0) \mid X \} \right] / e. \\ \end{array}
$$

The right-hand side of (13.3) equals

$$
\begin{array}{l} E \left\{\frac {e (X)}{e} \frac {1 - Z}{1 - e (X)} Y \right\} = E \left[ E \left\{\frac {e (X)}{e} \frac {1 - Z}{1 - e (X)} Y (0) \mid X \right\} \right] \\ { = } { E \left[ \frac { e ( X ) } { e \{ 1 - e ( X ) \} } E \left\{ ( 1 - Z ) Y ( 0 ) \mid X \right\} \right] } \\ { = } { E \left[ \frac { e ( X ) } { e \{ 1 - e ( X ) \} } E ( 1 - Z \mid X ) E \{ Y ( 0 ) \mid X \} \right] } \\ = E \left[ e (X) E \{Y (0) \mid X \} \right] / e. \\ \end{array}
$$

So (13.3) holds.

We have two inverse propensity score weighting estimators

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{ht}} = \hat {\bar {Y}} (1) - n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) Y _ {i}
$$

and

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{hajek}} = \hat {\bar {Y}} (1) - \frac {\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) Y _ {i}}{\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i})},
$$

where $\hat { o } ( X _ { i } ) = \hat { e } ( X _ { i } ) / \{ 1 - \hat { e } ( X _ { i } ) \}$ is the fitted odds of the treatment given covariates.

The estimation of $E ( Y \mid Z = 1 )$ is simple. We have a doubly robust

## 13.3 Inverse propensity score weighting and doubly robust estimation $o f \tau _ { \mathrm { T } }$ 167

estimator for $E \{ Y ( 0 ) \mid Z = 1 \}$ which combines the propensity score and the outcome model. Define

$$
\tilde {\mu} _ {0 \mathrm{T}} ^ {\mathrm{dr}} = E \left[ o (X, \alpha) (1 - Z) \{Y - \mu_ {0} (X, \beta_ {0}) \} + Z \mu_ {0} (X, \beta_ {0}) \right] / e, \tag {13.5}
$$

where $o ( X , \alpha ) = e ( X , \alpha ) / \{ 1 - e ( X , \alpha ) \}$ .

Theorem 13.3 Under Assumption 13.1, if either $\begin{array} { l l l } { e ( X , \alpha ) } & { = } & { e ( X ) } \end{array}$ or $\mu _ { 0 } ( X , \beta _ { 0 } ) = \mu _ { 0 } ( X )$ , then $\mu _ { 0 \mathrm { T } } ^ { d r } = E \{ Y ( 0 ) \mid Z = 1 \}$ .

Proof of Theorem 13.3: We have the decomposition

$$
\begin{array}{l} e \left[ \tilde {\mu} _ {0 \mathrm{T}} ^ {\mathrm{dr}} - E \{Y (0) \mid Z = 1 \} \right] \\ = E \left[ o (X, \alpha) (1 - Z) \{Y (0) - \mu_ {0} (X, \beta_ {0}) \} + Z \mu_ {0} (X, \beta_ {0}) \right] - E \{Z Y (0) \} \\ = E [ o (X, \alpha) (1 - Z) \{Y (0) - \mu_ {0} (X, \beta_ {0}) \} - Z \{Y (0) - \mu_ {0} (X, \beta_ {0}) \} ] \\ = E \left[ \left\{o (X, \alpha) (1 - Z) - Z \right\} \left\{Y (0) - \mu_ {0} (X, \beta_ {0}) \right\} \right] \\ = E \left[ \frac {e (X , \alpha) - Z}{1 - e (X , \alpha)} \{Y (0) - \mu_ {0} (X, \beta_ {0}) \} \right] \\ = E \left[ E \left\{\frac {e (X , \alpha) - Z}{1 - e (X , \alpha)} \mid X \right\} \times E \{Y (0) - \mu_ {0} (X, \beta_ {0}) \mid X \} \right] \\ = E \left[ \frac {e (X , \alpha) - e (X)}{1 - e (X , \alpha)} \times \{\mu_ {0} (X) - \mu_ {0} (X, \beta_ {0}) \} \right]. \\ \end{array}
$$

Therefore, $\tilde { \mu } _ { 0 \mathrm { T } } ^ { \mathrm { d r } } - E \{ Y ( 0 ) \mid Z = 1 \} = 0$ if either $e ( X , \alpha ) = e ( X ) { \mathrm { o r } } \mu _ { 0 } ( X , \beta _ { 0 } ) =$ $\mu _ { 0 } ( X )$ . □

From the population versions of $\tilde { \mu } _ { \mathrm { 0 T } } ^ { \mathrm { d r } }$ , we can construct the sample version by the following steps:

1. obtain the fitted values of the propensity scores $e ( X , { \hat { \alpha } } )$ ;  
2. obtain the fitted values of the outcome mean under control $\mu _ { 0 } ( X , { \hat { \beta } } _ { 0 } )$ ;  
3. construct the doubly robust estimator: $\hat { \tau } _ { \mathrm { T } } ^ { \mathrm { d r } } = \hat { \bar { Y } } ( 1 ) - \hat { \mu } _ { 0 \mathrm { T } } ^ { \mathrm { d r } }$ , where

$$
\hat {\mu} _ {0 \mathrm{T}} ^ {\mathrm{dr}} = \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} \left[ e (X _ {i}, \hat {\alpha}) \frac {(1 - Z _ {i}) \{Y _ {i} - \mu_ {0} (X _ {i} , \hat {\beta} _ {0}) \}}{1 - e (X _ {i} , \hat {\alpha})} + Z _ {i} \mu_ {0} (X _ {i}, \hat {\beta} _ {0}) \right];
$$

4. estimate the variance of $\tau _ { \mathrm { T } }$ via the bootstrap by resampling from $( Z _ { i } , X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ .

Hahn (1998), Mercatanti and Li (2014), Shinozaki and Matsuyama (2015) and Yang and Ding (2018) are references discussing the estimation of $\tau _ { \mathrm { T } }$ .

## 13.3 An example

The following R code implements two outcome regression estimators, two IPW estimators, and the doubly robust estimator for τT, as well as the bootstrap variance estimators. To avoid extreme estimated propensity scores, we can also truncated them from the above.

```r
ATT.est = function(z, y, x, out.family = gaussian, Utruncpscore = 1)
{
    ## sample size
    nn = length(z)
    nn1 = sum(z)

    ## fitted propensity score
    pscore = glm(z ~ x, family = binomial)$fitted.values
    pscore = pmin(Utruncpscore, pscore)
    odds.pscore = pscore/(1 - pscore)

    ## fitted potential outcomes
    outcome0 = glm(y ~ x, weights = (1 - z),
    family = out.family)$fitted.values

    ## regression imputation estimator
    ace.reg0 = lm(y ~ z + x)$coef[2]
    ace.reg = mean(y[z==1]) - mean(outcome0[z==1])
    ## propensity score weighting estimator
    ace.ipw0 = mean(y[z==1]) - mean(odds.pscore*(1 - z)*y)*nn/nn1
    ace.ipw = mean(y[z==1]) - mean(odds.pscore*(1 - z)*y)/mean(odds.pscore*(1 - z))
    ## doubly robust estimator
    res0 = y - outcome0
    ace.dr = ace.reg - mean(odds.pscore*(1 - z)*res0)*nn/nn1

    return(c(ace.reg0, ace.reg, ace.ipw0, ace.ipw, ace.dr))
}

OS_ATT = function(z, y, x, n.boot = 10^2,
    out.family = gaussian, Utruncpscore = 1)
{
    point.est = ATT.est(z, y, x, out.family, Utruncpscore)

    ## nonparametric bootstrap
    n.sample = length(z)
    x = as.matrix(x)
    boot.est = replicate(n.boot,
    {id.boot = sample(1:n.sample, n.sample, replace = TRUE)
```

```txt
ATT.est(z[id.boot], y[id.boot], x[id.boot, ], out.family, Utruncpscore))
```

```txt
boot.se = apply(boot.est, 1, sd)
res = rbind(point.est, boot.se)
rownames(res) = c("est", "se")
colnames(res) = c("reg0", "reg", "HT", "Hajek", "DR")
return(res)
}
```

Now we re-analyze the data in Example 10.3 to estimate τT. We obtain

```csv
reg0 reg HT Hajek DR
est 0.061 -0.351 -1.992 -0.351 -0.187
se 0.227 0.258 0.705 0.328 0.287
```

without truncating the estimated propensity scores, and

```batch
reg0 reg HT Hajek DR
est 0.061 -0.351 -0.597 -0.192 -0.230
se 0.223 0.255 0.579 0.302 0.276
```

by truncating the estimated propensity scores from the above at 0.9. The HT estimator is sensitive to the truncation as expected. The regression estimator in Example 13.1 is quite different from other estimators. It imposes an unnecessary assumption that the regression functions in the treatment and control group share the same coefficient of X. The regression estimator in Example 13.2 is much close to the Hajek and doubly robust estimators. The estimates above are slightly different from those in Section 12.3.3, suggesting some treatment effect heterogeneity across τT and τ .

## 13.4 Other estimands

Li et al. (2018a) gave a unified discussion of the causal estimands in observational studies. Starting from the conditional average causal effect τ (X), they proposed a general class of estimands

$$
\tau^ {h} = \frac {E \{h (X) \tau (X) \}}{E \{h (X) \}}
$$

indexed by a weighting function $h ( X )$ with $E \{ h ( X ) \} \ne 0$ . The normalization in the denominator is to ensure that a constant causal effect $\tau ( X ) = \tau$ averages to the same τ .

Under the unconfoundedness assumption,

$$
\tau^ {h} = \frac {E [ h (X) \{\mu_ {1} (X) - \mu_ {0} (X) \} ]}{E \{h (X) \}}
$$

## 17013 The Average Causal Effect on the Treated Units and Other Estimands

which motivates the outcome regression estimator

$$
\hat {\tau} ^ {h} = \frac {\sum_ {i = 1} ^ {n} h (X _ {i}) \{\hat {\mu} _ {1} (X _ {i}) - \hat {\mu} _ {0} (X _ {i}) \}}{\sum_ {i = 1} ^ {n} h (X _ {i})}.
$$

Moreover, we can show that $\tau ^ { h }$ has the following weighting form:

Theorem 13.4 Under ignorability and overlap, we have

$$
\tau^ {h} = E \left\{\frac {Z Y h (X)}{e (X)} - \frac {(1 - Z) Y h (X)}{1 - e (X)} \right\} / E \{h (X) \}.
$$

The proof of Theorem 13.4 is similar to those of Theorems 11.2 and 13.2 which is relegated to Problem 13.8. Based on Theorem 13.4, we can construct the corresponding IPW estimator.

By Theorem 13.4, each unit is associated with the weight due to the definition of the estimand as well as the weight due to the inverse of the propensity score. Finally, the treated units are weighted by $h ( X ) / e ( X )$ and the control units are weighted by $h ( X ) / \{ 1 - e ( X ) \}$ . Li et al. (2018a, Table 1) summarized several estimands, and I present a part of it below:

<table><tr><td>population</td><td>h(X)</td><td>estimand</td><td>weights</td></tr><tr><td>combined</td><td>1</td><td> $\tau$ </td><td> $1/e(X)$  and  $1/\{1-e(X)\}$ </td></tr><tr><td>treated</td><td>e(X)</td><td> $\tau_{\text{T}}$ </td><td>1 and e(X)/ $\{1-e(X)\}$ </td></tr><tr><td>control</td><td>1-e(X)</td><td> $\tau_{\text{C}}$ </td><td> $\{1-e(X)\}/e(X)$  and 1</td></tr><tr><td>overlap</td><td>e(X){1-e(X)}</td><td> $\tau_{\text{O}}$ </td><td>1-e(X) and e(X)</td></tr></table>

The overlap population and the corresponding estimand

$$
\tau_ {\mathrm{O}} = \frac {E [ e (X) \{1 - e (X) \} \tau (X) ]}{E [ e (X) \{1 - e (X) \} ]}
$$

is new to us. This estimand has the largest weight for units with $e ( X ) = 1 / 2$ and downweights the units with extreme propensity scores. A nice feature of this estimand is that its IPW estimator is rather stable without the possibly extremely small values of $e ( X )$ and $1 - e ( X )$ in the denominator. If $e ( X ) { \underline { { \bot \bot } } } \tau ( X )$ including the special case of $\tau ( X ) = \tau ,$ the parameter $\tau _ { \mathrm { O } }$ reduces to τ . In general, however, the estimand $\tau _ { \mathrm { O } }$ may cause controversy because it changes the initial population and depends on the propensity score which may be misspecified in practice. Li et al. (2018a) and Li et al. (2019) gave some justifications and numerical evidence. This estimand will appear again in Chapter 14.

We can also construct the doubly robust estimator for $\tau ^ { h }$ . I relegate the details to Problem 13.9.

## 13.5 Homework Problems

## 13.1 An algebraic fact about a regression estimator $f o r \ T _ { \mathrm { T } }$

This problem provides more details for Example 13.2.

Show that if we center the covariates by $X _ { i } - \hat { \bar { X } } ( 1 )$ for all units, then $\hat { \tau } _ { \mathrm { T } }$ equals the coefficient of $Z$ in the OLS fit of the outcome on the treatment, covariates, and their interactions.

## 13.2 Simulation for the average causal effect on the treated units

In OSATE.R in Chapter 12, I ran some simulation studies for τ . Run similar simulation studies for $\tau _ { \mathrm { T } }$ with either correct or incorrect propensity score or outcome models.

You can choose different model parameters, larger numbers of simulation and bootstrap replicates. Report your findings, including at least the bias, variance, and variance estimator via the bootstrap. You can also report other properties of the estimators, for example, the asymptotic Normality and the coverage rates of the confidence intervals.

## 13.3 An alternative form of the doubly robust estimator for $\tau _ { \mathrm { T } }$

Motivated by (13.5), we have an alternative form of doubly robust estimator for $E \{ Y ( 0 ) \mid Z = 1 \}$ }:

$$
\tilde {\mu} _ {0 \mathrm{T}} ^ {\mathrm{dr2}} = \frac {E [ o (X , \alpha) (1 - Z) \{Y - \mu_ {0} (X , \beta_ {0}) \} ]}{E [ o (X , \alpha) (1 - Z) ]} + E \{Z \mu_ {0} (X, \beta_ {0}) \} / e.
$$

Show that under Assumption 13.1, $\tilde { \mu } _ { 0 \mathrm { T } } ^ { \mathrm { d r 2 } } = E \{ Y ( 0 ) | Z = 1 \}$ if either $e ( X , \alpha ) = e ( X ) \mathrm { o r } \mu _ { 0 } ( X , \beta _ { 0 } ) = \mu _ { 0 } ( X )$ . Give the sample analogue of the doubly robust estimator for $\tau _ { \mathrm { T } }$ .

## 13.4 Average causal effect on the control units

Prove the identification formulas for $\tau _ { \mathrm { { C } } } .$ , analogous to (13.1) and (13.4). Propose the doubly robust estimator for $\tau _ { \mathrm { C } }$ .

## 13.5 Estimating individual effect and conditional average causal effect

$\{ Z _ { i } , X _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \} _ { i = 1 } ^ { n } \stackrel { \mathrm { I I D } } { \sim } \{ Z , X , Y ( 1 ) , Y ( 0 ) \}$ effect is $\tau _ { i } = Y _ { i } ( 1 ) - Y _ { i } ( 0 )$ and the conditional average causal effect is $\tau ( X _ { i } ) =$ $E \{ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \mid X _ { i } \}$ . Since we will discuss individual effect, we do not drop the subscript i since τ mean the average causal effect, not the population version of $Y ( 1 ) - Y ( 0 )$ .

1. Under randomization with $Z _ { i } \bot \bot \{ Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \}$ and $e = \mathrm { p r } ( Z _ { i } = 1 )$ ,

## 17213 The Average Causal Effect on the Treated Units and Other Estimands

show that

$$
\delta_ {i} = \frac {Z _ {i} Y _ {i}}{e} - \frac {(1 - Z _ {i}) Y _ {i}}{1 - e}
$$

is an unbiased predictor of the individual effect in the sense that

$$
E (\delta_ {i} - \tau_ {i}) = 0 (i = 1, \dots , n).
$$

Further show that $E ( \delta _ { i } ) = \tau$ for all $i = 1 , \ldots , n .$ .

2. Under ignorability with $Z _ { i } \bot \bot \{ Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \} \quad | \quad X _ { i }$ and $e ( X _ { i } ) \ =$ pr $\ \cdot Z _ { i } = 1 \mid X _ { i } )$ , show that

$$
\delta_ {i} = \frac {Z _ {i} Y _ {i}}{e (X _ {i})} - \frac {(1 - Z _ {i}) Y _ {i}}{1 - e (X _ {i})}
$$

is an unbiased predictor of the individual effect and the conditional average causal effect in the sense that

$$
E \left(\delta_ {i} - \tau_ {i}\right) = 0, \quad E \left\{\delta_ {i} - \tau \left(X _ {i}\right) \right\} = 0, \quad (i = 1, \dots , n).
$$

Further show that $E ( \delta _ { i } ) = \tau$ for all $i = 1 , \ldots , n .$ .

## 13.6 General estimand and $( \tau _ { \mathrm { T } } , \tau _ { \mathrm { C } } )$

Assume unconfoundedness. Show that $\tau ^ { h } = \tau _ { \mathrm { T } } \ \mathrm { i f } \ h ( X ) = e ( X )$ , and $\tau ^ { h } = \tau _ { \mathrm { { C } } }$ if $h ( X ) = 1 - e ( X )$ .

## 13.7 More on $\tau _ { \mathrm { O } }$

Show that

$$
\tau_ {\mathrm{O}} = \frac {E [ \{1 - e (X) \} \tau (X) \mid Z = 1 ]}{E \{1 - e (X) \mid Z = 1 \}} = \frac {E \{e (X) \tau (X) \mid Z = 0 \}}{E \{e (X) \mid Z = 0 \}}.
$$

## 13.8 IPW for the general estimand

Prove Theorem 13.4.

## 13.9 Doubly robust estimation for general estimand

For a given $h ( X )$ , we have the following formulas for constructing the doubly robust estimator for $\tau ^ { h }$ :

$$
\begin{array}{l} \tilde {\mu} _ {1} ^ {h, \mathrm{dr}} = E \left[ \frac {Z h (X) \{Y - \mu_ {1} (X , \beta_ {1}) \}}{e (X , \alpha)} + h (X) \mu_ {1} (X, \beta_ {1}) \right], \\ \tilde {\mu} _ {0} ^ {h, \mathrm{dr}} = E \left[ \frac {(1 - Z) h (X) \{Y - \mu_ {0} (X , \beta_ {0}) \}}{1 - e (X , \alpha)} + h (X) \mu_ {0} (X, \beta_ {0}) \right]. \\ \end{array}
$$

Show that under ignorability and overlap,

## 13.5 Homework Problems

1. if either e(X, α) = e(X) or $\mu _ { 1 } ( X , \beta _ { 1 } ) \ = \ \mu _ { 1 } ( X )$ , then $\tilde { \mu } _ { 1 } ^ { h , \mathrm { d r } } ~ =$ E{h(X)Y (1)};  
2. if either e(X, α) = e(X) or $\mu _ { 0 } ( X , \beta _ { 0 } ) \ = \ \mu _ { 0 } ( X )$ , then $\tilde { \mu } _ { 0 } ^ { h , \mathrm { d r } } ~ =$ E{h(X)Y (0)};  
3. if either $e ( X , \alpha ) ~ = ~ e ( X ) ~ \mathrm { o r } ~ \{ \mu _ { 1 } ( X , \beta _ { 1 } ) ~ = ~ \mu _ { 1 } ( X ) , \mu _ { 0 } ( X , \beta _ { 0 } ) ~ =$ $\mu _ { 0 } ( X ) \}$ , then

$$
\frac {\tilde {\mu} _ {1} ^ {h , \mathrm{dr}} - \tilde {\mu} _ {0} ^ {h , \mathrm{dr}}}{E \{h (X) \}} = \tau^ {h}.
$$

Remark: Tao and Fu (2019) proved the above results. However, they hold only for a given $h ( X )$ . The most interesting cases of $\tau _ { \mathrm { T } } , \tau _ { \mathrm { C } }$ and $\tau _ { \mathrm { O } }$ all have weight depending on the propensity score $e ( X )$ , which must be estimated in the first place. The above formulas do not apply to constructing the doubly robust estimators for $\tau _ { \mathrm { T } }$ and $\tau _ { \mathrm { { C } } } ;$ there does not exist a doubly robust estimator for $\tau _ { \mathrm { O } }$ .

## 13.10 Recommended reading

Shinozaki and Matsuyama (2015) focused on $\tau _ { \mathrm { T } }$ , and Li et al. (2018a) discussed general $\tau ^ { h }$ .

# Using the Propensity Score in Regressions for Causal Effects

Since Rosenbaum and Rubin (1983b)’s seminal paper, many creative uses of the propensity score have appeared in the literature (e.g., Bang and Robins, 2005; Robins et al., 2007; Van der Laan and Rose, 2011; Vansteelandt and Daniel, 2014). This chapter discusses two simple methods to use the propensity score: including the propensity score as a covariate in regressions and running regressions weighted by the inverse of the propensity score. I choose to focus on these two methods because

1. they are easy to implement, which involve only standard statistical software packages for regressions;  
2. their properties are comparable to many more complex methods;  
3. they can be easily extended to allow for flexible statistical models including machine learning algorithms.

## 14.1 Regressions with the propensity score as a covariate

By Theorem 11.1, if unconfoundedness holds conditioning on $X$ , then it also holds conditioning on e(X):

$$
Z \bot \{Y (1), Y (0) \} \mid e (X).
$$

Analogous to (10.5), τ is also nonparametrically identified by

$$
\tau = E \Big [ E \{Y \mid Z = 1, e (X) \} - E \{Y \mid Z = 0, e (X) \} \Big ],
$$

which motivates methods based on regressions of $Y$ on $Z$ and $e ( X )$ .

The simplest regression specification is the OLS fit of $Y$ on $\{ 1 , Z , e ( X ) \}$ , with the coefficient of $Z$ as an estimator, denoted by $\tau _ { e } .$ For simplicity, I will discuss the population OLS:

$$
\arg \min _ {a, b, c} E \{Y - a - b Z - c e (X) \} ^ {2}
$$

with $\tau _ { e }$ defined as the coefficient of $Z .$ It is consistent for τ if we have a correct propensity score model and the outcome model is indeed linear in $Z$ and $e ( X )$ . The more interesting result is that $\tau _ { e }$ estimates $\tau _ { \mathrm { O } }$ if we have a correct propensity score model even if the outcome model is completely misspecified.

Theorem 14.1 $I f Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$ , then the coefficient of $Z$ in the OLS fit of $Y$ on $\{ 1 , Z , e ( X ) \}$ } equals

$$
\tau_ {e} = \tau_ {\mathrm{O}} = \frac {E \{h _ {\mathrm{O}} (X) \tau (X) \}}{E \{h _ {\mathrm{O}} (X) \}},
$$

recalling that $h _ { \mathrm { O } } ( X ) = e ( X ) \{ 1 - e ( X ) \} \ a n d \ \tau ( X ) = E \{ Y ( 1 ) - Y ( 0 ) \mid X \}$ .

An unusual feature of Theorem 14.1 is that the overlap condition is not needed any more. Even if some units have propensity score $e ( X )$ equaling 0 or 1, their associate weight $e ( X ) \{ 1 - e ( X )$ is zero so that they do not contribute anything to the final parameter $\tau _ { \mathrm { O } }$ .

Proof of Theorem 14.1: Based on the FWL theorem reviewed in Section $\mathrm { A 2 . 3 }$ , we can obtain $\tau _ { e }$ in two steps: first, we obtain the residual $\tilde { Z }$ from the OLS fit of $Z$ on $\{ 1 , e ( X ) \}$ ; then, we obtain $\tau _ { e }$ from the OLS fit of $Y$ on $\tilde { Z } .$ .

The coefficient of $e ( X )$ in the OLS fit of $Z$ on $\{ 1 , e ( X ) \}$ is

$$
\begin{array}{l} \frac {\operatorname{cov} \{Z , e (X) \}}{\operatorname{var} \{e (X) \}} = \frac {E [ \operatorname{cov} \{Z , e (X) \mid X \} ] + \operatorname{cov} \{E (Z \mid X) , e (X) \}}{\operatorname{var} \{e (X) \}} \\ = \frac {0 + \operatorname{var} \{e (X) \}}{\operatorname{var} \{e (X) \}} = 1, \\ \end{array}
$$

so the intercept is $E ( Z ) - E \{ e ( X ) \} = 0$ and the residual is $\tilde { Z } = Z - e ( X )$ . This makes sense since $Z - e ( X )$ is uncorrelated with any function of $X$ .

Therefore, we can obtain $\tau _ { e }$ from the univariate OLS fit of $Y$ on a centered variable $Z - e ( X )$ :

$$
\tau_ {e} = \frac {\operatorname{cov} \{Z - e (X) , Y \}}{\operatorname{var} \{Z - e (X) \}}.
$$

The denominator simplifies to

$$
\begin{array}{l} \operatorname{var} \{Z - e (X) \} = E \{Z - e (X) \} ^ {2} \\ = E \{Z + e (X) ^ {2} - 2 Z e (X) \} \\ = e (X) + e (X) ^ {2} - 2 e (X) ^ {2} = h _ {0} (X). \\ \end{array}
$$

The numerator simplifies to

$$
\begin{array}{l} \operatorname{cov} \{Z - e (X), Y \} \\ = E [ \{Z - e (X) \} Y ] \\ = E [ \{Z - e (X) \} Z Y (1) ] + E [ \{Z - e (X) \} (1 - Z) Y (0) ] \\ (\text { since } Y = Z Y (1) + (1 - Z) Y (0)) \\ = E [ \{Z - Z e (X) \} Y (1) ] - E [ e (X) (1 - Z) Y (0) ] \\ = E [ Z \{1 - e (X) \} Y (1) ] - E [ e (X) (1 - Z) Y (0) ] \\ = E [ e (X) \{1 - e (X) \} \mu_ {1} (X) ] - E [ e (X) \{1 - e (X) \} \mu_ {0} (X) ] \\ (\text { tower   property   and   ignorability }) \\ = E \{h _ {0} (X) \tau (X) \}. \\ \end{array}
$$

The conclusion follows.

From the proof of Theorem 14.1, we can simply run the OLS of $Y$ on the centered treatment $\tilde { Z } = Z - e ( X )$ . Lee (2018) proposed this procedure. Moreover, we can also include X in the OLS fit which may improve efficiency in finite sample. However, this does not change the estimand, which is still $\tau _ { \mathrm { O } }$ . I summarize these two results in the corollary below.

Corollary 14.1 If $Z \bot \bot \{ Y ( 1 ) , Y ( 0 ) \} \mid X$ , then

(1) the coefficient of $Z - e ( X )$ in the OLS fit of Y on $Z - e ( X )$ or $\{ 1 , Z - e ( X ) \}$ equals $\tau _ { \mathrm { O } }$ ;  
(2) the coefficient of Z in the OLS fit of Y on $\{ 1 , Z , e ( X ) , X \}$ equals $\tau _ { \mathrm { O } } .$ .

Proof of Corollary 14.1: (1) The first result is an intermediate step in the proof of Theorem 14.1. The second result holds because regressing $Y$ on $Z - e ( X ) \ \mathrm { o r } \ \{ 1 , Z - e ( X ) \}$ does not change the coefficient of $Z - e ( X )$ since it has mean zero.

(2) It follows from the fact that

$$
Z - e (X) = Z - 0 - 1 \cdot e (X) - 0 ^ {\mathsf {T}} X
$$

is the residual of the OLS fit of $Z$ on $\{ 1 , e ( X ) , X \}$ , since $Z - e ( X )$ is uncorrelated with any functions of $X$ .

Theorem 14.1 motivates a two-step estimator for $\tau _ { \mathrm { O } } \colon$ first, fit a propensity score model to obtain $\hat { e } ( X _ { i } ) ;$ ; second, run OLS of $Y _ { i }$ on $( 1 , X _ { i } , \hat { e } ( X _ { i } ) )$ to obtain the coefficient of $Z _ { i }$ . Corollary 14.1 motivates another two-step estimator for $\tau _ { \mathrm { O } } \colon$ first, fit a propensity score model to obtain $\hat { e } ( X _ { i } )$ ; second, run OLS of $Y _ { i }$ on $Z _ { i } - \hat { e } ( X _ { i } )$ to obtain the coefficient of $Z _ { i }$ . Although OLS is convenient for obtaining point estimators, the corresponding standard errors are incorrect due to the uncertainty in the first step estimation of the propensity score. We can use the bootstrap to approximate the standard errors.

Robins et al. (1992) discussed many OLS estimators based on the propensity score. The above results seem special cases of their general theory although they did not point out the connection with the estimand under the overlap weight, which was resurrected by Li et al. (2018a). Lee (2018) proposed to regress Y on $Z - e ( X )$ from a different perspective without making connections to the existing results in Robins et al. (1992) and Li et al. (2018a).

Rosenbaum and Rubin (1983b) proposed to estimate the average causal effect based on the OLS fit of Y on $\{ 1 , Z , e ( X ) , Z e ( X ) \}$ . When this outcome model is correct, their estimator is consistent for the average causal effect. However, when the model is incorrect, the corresponding estimator has a much more complicated interpretation. Little and An (2004) suggested constructing estimators based on the OLS of Y on Z and a flexible function of e(X) and showed it enjoys certain doubly robustness property. Due to the complexity in implementation, I omit the discussion.

## 14.2 Regressions weighted by the inverse of the propensity score

## 14.2.1 Average causal effect

We first re-examine the Hajek estimator of τ :

$$
\hat {\tau} ^ {\mathrm{hajek}} = \frac {\sum_ {i = 1} ^ {n} \frac {Z _ {i} Y _ {i}}{\hat {e} (X _ {i})}}{\sum_ {i = 1} ^ {n} \frac {Z _ {i}}{\hat {e} (X _ {i})}} - \frac {\sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})}}{\sum_ {i = 1} ^ {n} \frac {1 - Z _ {i}}{1 - \hat {e} (X _ {i})}},
$$

which equals the difference between the weighted means of the outcomes in the treatment and control groups. Numerically, it is identical to the coefficient of $Z _ { i }$ in the following weighted least squares (WLS) of $Y _ { i }$ on (1, Zi).

Proposition 14.1 τˆhajek equals $\hat { \beta }$ from the following WLS:

$$
(\hat {\alpha}, \hat {\beta}) = \arg \min _ {\alpha , \beta} \sum_ {i = 1} ^ {n} w _ {i} (Y _ {i} - \alpha - \beta Z _ {i}) ^ {2}
$$

with weights

$$
w _ {i} = \frac {Z _ {i}}{\hat {e} (X _ {i})} + \frac {1 - Z _ {i}}{1 - \hat {e} (X _ {i})} = \left\{ \begin{array}{l l} \frac {1}{\hat {e} (X _ {i})} & \text {   if   } Z _ {i} = 1; \\ \frac {1}{1 - \hat {e} (X _ {i})} & \text {   if   } Z _ {i} = 0. \end{array} \right. \tag {14.1}
$$

Imbens (2004) pointed out the result in Proposition 14.1. I leave it as a Problem 14.1. By Proposition 14.1, it is convenient to obtain ˆτhajek based on WLS. However, due to the uncertainty in the estimated propensity score, the standard error reported by WLS is incorrect for the true standard error of $\hat { \tau } ^ { \mathrm { h a j e k } }$ . The bootstrap provides a convenient approximation to the true standard error.

Why does the WLS give a consistent estimator for $\tau ?$ Recall that in the CRE with a constant propensity score, we can simply use the coefficient of $Z _ { i }$ in the OLS fit of $Y _ { i } \mathrm { o n } \left( 1 , Z _ { i } \right)$ to estimate τ . In observational studies, units have different probabilities of receiving the treatment and control, respectively. If we weight the treated units by $1 / e ( X _ { i } )$ and the control units by $1 / \{ 1 - e ( X _ { i } ) \}$ , then they can represent the whole population and we effectively have a pseudo randomized experiment. Consequently, the difference between the weighted means are consistent for τ . The numerical equivalence of $\hat { \tau } ^ { \mathrm { h a j e k } }$ and WLS is not only a fun numerical fact itself but also useful for motivation more complex estimator with covariate adjustment. I give one extension below.

Recall that in the CRE, we can use the coefficient of $Z _ { i }$ in the OLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ ) to estimate $\tau ,$ where the covariates are centered with $\bar { X } =$ 0. This is Lin (2013)’s estimator which uses covariates to improve efficiency. A natural extension to observational studies is to estimate τ using the coefficient of $Z _ { i }$ in the WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ with weights defined in (14.1). Hirano and Imbens (2001) used this estimator in an application. The fully interacted linear model is equivalent to two separate linear models for the treated and control groups. If the linear models

$$
E (Y \mid Z = 1, X) = \beta_ {1 0} + \beta_ {1 x} ^ {\mathsf {T}} X, E (Y \mid Z = 0, X) = \beta_ {0 0} + \beta_ {0 x} ^ {\mathsf {T}} X,
$$

are correctly specified, then both OLS and WLS give consistent estimators for the coefficients and the estimators of the coefficient of $Z$ is consistent for τ. More interestingly, the estimator of the coefficient of Z based on WLS is also consistent for $\tau$ if the propensity score model is correct and the outcome model is incorrect. That is, the estimator based on WLS is doubly robust. Robins et al. (2007) discussed this property and attributed this result to M. Joffe’s unpublished paper. I will give more details below.

Let $\hat { e } ( X _ { i } )$ be the fitted propensity score and $( \mu _ { 1 } ( X _ { i } , \hat { \beta } _ { 1 } ) , \mu _ { 0 } ( X _ { i } , \hat { \beta } _ { 0 } ) )$ be the fitted values of the outcome means based on the WLS. The outcome regression estimator is

$$
\hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{reg}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \mu_ {1} (X _ {i}, \hat {\beta} _ {1}) - \frac {1}{n} \sum_ {i = 1} ^ {n} \mu_ {0} (X _ {i}, \hat {\beta} _ {0})
$$

and the doubly robust estimator for τ is

$$
\hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{dr}} = \hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{reg}} + \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {Z _ {i} \{Y _ {i} - \mu_ {1} (X _ {i} , \hat {\beta} _ {1}) \}}{\hat {e} (X _ {i})} - \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) \{Y _ {i} - \mu_ {0} (X _ {i} , \hat {\beta} _ {0}) \}}{1 - \hat {e} (X _ {i})}.
$$

An interesting result is that this doubly robust estimator equals the outcome regression estimator, which reduces to the coefficient of $Z _ { i }$ in the WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ if we use weights (14.1).

Theorem 14.2 If $\bar { X } = 0 \ a n d \ ( \mu _ { 1 } ( X _ { i } , \hat { \beta } _ { 1 } ) , \mu _ { 0 } ( X _ { i } , \hat { \beta } _ { 0 } ) ) = ( \hat { \beta } _ { 1 0 } + \hat { \beta } _ { 1 x } ^ { \top } X _ { i } , \hat { \beta } _ { 0 0 } +$ $\hat { \beta } _ { 0 x } ^ { \mathsf { T } } X _ { i } )$ based on the WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ with weights (14.1), then

$$
\hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{dr}} = \hat {\tau} _ {\mathrm{wls}} ^ {\mathrm{reg}} = \hat {\beta} _ {1 0} - \hat {\beta} _ {0 0},
$$

which is the coefficient of $Z _ { i }$ in the WLS $\it { \Omega } \mathcal { f } t .$ .

Proof of Theorem 14.2: The WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ is equivalent to two WLS fits based on the treated and control data. Both WLS fits include intercepts, so the first order conditions must satisfy

$$
\sum_ {i = 1} ^ {n} \frac {Z _ {i} (Y _ {i} - \hat {\beta} _ {1 0} - \hat {\beta} _ {1 x} ^ {\intercal} X _ {i})}{\hat {e} (X _ {i})} = 0
$$

and

$$
\sum_ {i = 1} ^ {n} \frac {(1 - Z _ {i}) (Y _ {i} - \hat {\beta} _ {0 0} - \hat {\beta} _ {0 x} ^ {\mathsf {T}} X _ {i})}{1 - \hat {e} (X _ {i})} = 0.
$$

So the difference between ${ \hat { \tau } } ^ { \mathrm { d r } }$ and $\hat { \tau } ^ { \mathrm { r e g } }$ is exactly zero. Both reduces to

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} (\hat {\beta} _ {1 0} + \hat {\beta} _ {1 x} ^ {\mathsf {T}} X _ {i}) - \frac {1}{n} \sum_ {i = 1} ^ {n} (\hat {\beta} _ {0 0} + \hat {\beta} _ {0 x} ^ {\mathsf {T}} X _ {i}) = \hat {\beta} _ {1 0} - \hat {\beta} _ {0 0} + (\hat {\beta} _ {1 x} - \hat {\beta} _ {0 x}) ^ {\mathsf {T}} \bar {X} = \hat {\beta} _ {1 0} - \hat {\beta} _ {0 0}
$$

with centered covariates. So they both equal the coefficient of $Z _ { i }$ in the WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ . □

Freedman and Berk (2008) discouraged the use of the WLS estimator above based on some simulation studies. They showed that when the outcome model is correct, the WLS estimator is worse than the OLS estimator since the WLS estimator has large variability in their simulation setting with homoskedastic outcomes. This may not be true in general. When the errors have variance proportional to the inverse of the propensity scores, the WLS estimator will be more efficient than the OLS estimator. They also showed that the estimated standard error based on the WLS fit is not consistent for the true standard error because it ignores the uncertainty in the estimated propensity score. This can be easily fixed by using the bootstrap to approximate the variance of the WLS estimator. Nevertheless, they found that “weighting may help under some circumstances” because when the outcome model is incorrect, the WLS estimator is still consistent if the propensity score model is correct.

I end this section with Table 14.1 summarizing the regression estimators for causal effects in both randomized experiments and observational studies.

## 14.2.2 Average causal effect on the treated units

The results for $\tau _ { \mathrm { T } }$ parallel those for τ . First, the Hajek estimator for $\tau _ { \mathrm { T } }$

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{hajek}} = \hat {\bar {Y}} (1) - \frac {\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) Y _ {i}}{\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i})},
$$

with $\hat { o } ( X _ { i } ) = \hat { e } ( X _ { i } ) / \{ 1 - \hat { e } ( X _ { i } ) \}$ , equals the coefficient of $Z _ { i }$ in the following WLS fit $Y _ { i }$ on $( 1 , Z _ { i } )$ .

**TABLE 14.1: Regression estimators in CREs and unconfounded observational studies. The weights $w _ { i } \mathrm { ^ s }$ are defined in (14.1) .**

<table><tr><td></td><td>CRE</td><td>unconfounded observational studies</td></tr><tr><td>without X</td><td> $Y_i \sim Z_i$ </td><td> $Y_i \sim Z_i$  with weights  $w_i$ </td></tr><tr><td>with X</td><td> $Y_i \sim (Z_i, X_i, Z_i X_i)$ </td><td> $Y_i \sim (Z_i, X_i, Z_i X_i)$  with weights  $w_i$ </td></tr></table>

Proposition 14.2 $\hat { \tau } _ { \mathrm { T } } ^ { h a j e k }$ is numerically identical to $\hat { \beta }$ in the following WLS:

$$
(\hat {\alpha}, \hat {\beta}) = \arg \min _ {\alpha , \beta} \sum_ {i = 1} ^ {n} w _ {\mathrm{Ti}} (Y _ {i} - \alpha - \beta Z _ {i}) ^ {2}
$$

with weights

$$
w _ {\mathrm{T} i} = Z _ {i} + (1 - Z _ {i}) \hat {o} (X _ {i}) = \left\{ \begin{array}{l l} 1 & \text {   if   } Z _ {i} = 1; \\ \hat {o} (X _ {i}) & \text {   if   } Z _ {i} = 0. \end{array} \right. \tag {14.2}
$$

Similar to Proposition 14.1, Proposition 14.2 is a pure linear algebra result. I relegate its proof as Problem 14.1.

Second, if we center covariates with $\hat { \bar { X } } ( 1 ) = 0$ , then we can estimate $\tau _ { \mathrm { T } }$ using the coefficient of $Z _ { i }$ in the WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ with weights defined in (14.2). Similarly, this estimator equals the regression estimator

$$
\hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{reg}} = \hat {\bar {Y}} (1) - \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} \mu_ {0} (X _ {i}, \hat {\beta} _ {0}),
$$

which also equals the doubly robust estimator

$$
\hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{dr}} = \hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{reg}} - \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) \{Y _ {i} - \mu_ {0} (X _ {i}, \hat {\beta} _ {0}) \}.
$$

Theorem 14.3 $I f \hat { \bar { X } } ( 1 ) = 0$ and $\mu _ { 0 } ( X _ { i } , \hat { \beta } _ { 0 } ) = \hat { \beta } _ { 0 0 } + \hat { \beta } _ { 0 x } ^ { \top } X _ { i }$ based on the WLS fit of $Y _ { i }$ on $( 1 , Z _ { i } , X _ { i } , \bar { Z _ { i } } X _ { i } )$ with weights (14.2), then

$$
\hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{dr}} = \hat {\tau} _ {\mathrm{T,wls}} ^ {\mathrm{reg}} = \hat {\beta} _ {1 0} - \hat {\beta} _ {0 0},
$$

which is the coefficient of $Z _ { i }$ in the WLS $\it { \Omega } \mathcal { f } t .$

Proof of Theorem 14.3: Based on the WLS fits in the treatment and control groups, we have

$$
\sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\beta} _ {1 0} - \hat {\beta} _ {1 x} ^ {\intercal} X _ {i}) = 0, \tag {14.3}
$$

$$
\sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}) (Y _ {i} - \hat {\beta} _ {0 0} - \hat {\beta} _ {0 x} ^ {\intercal} X _ {i}) = 0. \tag {14.4}
$$

$\hat { \tau } _ { \mathrm { { T , w l s } } } ^ { \mathrm { { d r } } } = \hat { \tau } _ { \mathrm { { T , w l s } } } ^ { \mathrm { { r e g } } }$ = ˆτT,wls. Both reduces to

$$
\hat {\bar {Y}} (1) - \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} (\hat {\beta} _ {0 0} + \hat {\beta} _ {0 x} ^ {\mathsf {T}} X _ {i}) = \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\beta} _ {0 0} - \hat {\beta} _ {0 x} ^ {\mathsf {T}} X _ {i}).
$$

With covariates centered with $\hat { \bar { X } } ( 1 ) = 0 .$ the first result (14.3) implies that $\hat { \bar { Y } } ( 1 ) = \hat { \beta } _ { 1 0 }$ which further simplifies the estimators to $\hat { \beta } _ { 1 0 } - \hat { \beta } _ { 0 0 }$ . □

## 14.3 Homework problems

## 14.1 Hajek estimators as WLS estimators

Prove Propositions 14.1 and 14.2.

Hint: These are special cases of Problem A2.2 on the univariate WLS.

## 14.2 Predictive estimator and doubly robust estimator

Another outcome regression estimator is the predictive estimator

$$
\hat {\tau} ^ {\mathrm{pred}} = \hat {\mu} _ {1} ^ {\mathrm{pred}} - \hat {\mu} _ {0} ^ {\mathrm{pred}}
$$

where

$$
\hat {\mu} _ {1} ^ {\mathrm{pred}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left\{Z _ {i} Y _ {i} + (1 - Z _ {i}) \mu_ {1} (X _ {i}, \hat {\beta} _ {1}) \right\}
$$

and

$$
\hat {\mu} _ {0} ^ {\text { pred }} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left\{Z _ {i} \mu_ {0} (X _ {i}, \hat {\beta} _ {1}) + (1 - Z _ {i}) Y _ {i} \right\}.
$$

It differs from the outcome regression estimator discussed before in that it only predicts the counterfactural outcomes but not the observed outcomes.

Show that the doubly robust estimator equals ˆτpred if $( \mu _ { 1 } ( X _ { i } , \hat { \beta } _ { 1 } ) , \mu _ { 0 } ( X _ { i } , \hat { \beta } _ { 1 } ) ) =$ $( \hat { \beta } _ { 1 0 } + \hat { \beta } _ { 1 x } ^ { \top } X _ { i } , \hat { \beta } _ { 0 0 } + \hat { \beta } _ { 0 x } ^ { \top } X _ { i } )$ are from the WLS fits of $Y _ { i }$ on $( 1 , X _ { i } )$ based on the treated and control data, respectively, with weights

$$
w _ {i} = Z _ {i} / \hat {o} (X _ {i}) + (1 - Z _ {i}) \hat {o} (X _ {i}) = \left\{ \begin{array}{l l} \frac {1}{\hat {o} (X _ {i})} = \frac {1 - \hat {e} (X _ {i})}{\hat {e} (X _ {i})} & \text { if } Z _ {i} = 1; \\ \hat {o} (X _ {i}) = \frac {\hat {e} (X _ {i})}{1 - \hat {e} (X _ {i})} & \text { if } Z _ {i} = 0. \end{array} \right. \tag {14.5}
$$

Remark: Cao et al. (2009) and Vermeulen and Vansteelandt (2015) motivated the weights in (14.5) from other more theoretical perspectives.

<!-- footnote -->

- If the logistic outcome model is correct, then $\hat { \beta } _ { z }$ estimates the conditional odds ratio of the treatment on the outcome given covariates, which does not equal τ. Freedman (2008c) gave an warning of using the logistic regression coefficient to estimate τ in CREs. See Chapter A2 for more details of the logistic regression.

<!-- footnote end -->

<!-- footnote -->

- The glm function is more general than the lm function. With out.family = gaussian, glm is identical to lm.

<!-- footnote end -->

## 14.3 Weighted logistic regression with a binary outcome

With a binary outcome, we can replace linear outcome models by the logistic outcome models. Show that with weights in the logistic regressions, the doubly robust estimators equals the outcome regression estimator. The result holds for both τ and $\tau _ { \mathrm { T } } .$ .

## 14.4 Causal inference with a misspecified linear regression

Define the population OLS of Y on Z, X as

$$
(\beta_ {0}, \beta_ {1}, \beta_ {2}) = \arg \min _ {b _ {0}, b _ {1}, b _ {2}} E (Y - b _ {0} - b _ {1} Z - b _ {2} ^ {\mathsf {T}} X) ^ {2}.
$$

Recall that $e ( X ) = \mathrm { p r } ( Z = 1 \mid X )$ is the propensity score, and define $\tilde { e } ( X ) =$ $\gamma _ { 0 } + \gamma _ { 1 } ^ { \intercal } X$ as the OLS projection of A on X with

$$
(\gamma_ {0}, \gamma_ {1}) = \arg \min _ {c _ {0}, c _ {1}} E (A - c _ {0} - c _ {1} ^ {\mathsf {T}} X) ^ {2}.
$$

1. Show that

$$
\beta_ {1} = \frac {E [ \tilde {w} (X) \{\mu_ {1} (X) - \mu_ {0} (X) \} ]}{E \{\tilde {w} (X) \}} + \frac {E [ \{e (X) - \tilde {e} (X) \} \mu_ {0} (X) ]}{E \{\tilde {w} (X) \}}
$$

where $\tilde { w } ( X ) = e ( X ) \{ 1 - \tilde { e } ( X ) \}$ .

2. When X contains the dummy variables for a discrete covariate, show that

$$
\beta_ {1} = \frac {E [ w (X) \{\mu_ {1} (X) - \mu_ {0} (X) \} ]}{E \{w (X) \}}
$$

where $w ( X ) = e ( X ) \{ 1 - e ( X ) \}$ is the overlap weight.

Remark: Vansteelandt and Dukes (2022) gave the formula in the first part without a detailed proof. The result in part 2 was derived many times in the literature (e.g., Angrist, 1998; Ding, 2021).

## 14.5 Data re-analysis

Re-analyze the dataset in karolinska.txt and the dataset nhanesbmi in the ATE package.

## 14.6 Recommended reading

Kang and Schafer (2007) gave a critical review of the doubly robust estimator, using simulation to compare it with many other estimators. Robins et al. (2007) gave a very insightful comment on Kang and Schafer (2007).

# Matching in Observational Studies

Matching has a long history in empirical research. W. Cochran and D. Rubin popularized it in statistical causal inference. Cochran and Rubin (1973) is an early review paper. Rubin (2006b) collects Rubin’s contributions to this topic. This chapter also discusses modern contributions by Abadie and Imbens (2006, 2008, 2011).

## 15.1 A simple starting point: many more control units

![image_12](images/image_12.png)

```mermaid
graph TD
    subgraph treated_group["\"treated group\""]
  X1 --> Xm1["X_{m(1)}"]
  X2 --> Xm2["X_{m(2)}"]
  node["..."] --> Xmn1["X_{m(n1)}"]
  Xn1 --> Xmn1
    end
    subgraph control_group["\"control group\""]
  Xm1 --> Xm2
  Xmn1 --> Xmn1
    end
  X1 --> X2
  X2 --> node --> Xm1
  Xm1 --> Xm2
  Xm2 --> Xmn1
```

Consider a simple case with the number of control units n0 being much larger than the number of treated units n1. For unit i = 1, . . . , n1 in the treated group, we find a unit $m ( i )$ in the control group such that $X _ { i } = X _ { m ( i ) }$ . In the ideal case, we have exact matches. Therefore, the units within a matched pair have the same propensity score $e ( X _ { i } ) = e ( X _ { m ( i ) } )$ . Consequently, conditioning on the event that one unit receives the treatment and the other receives the control, the probability of unit i receiving the treatment and unit $m ( i )$ receives the control is

$$
\begin{array}{l} \operatorname{pr} \left(Z _ {i} = 1, Z _ {m (i)} = 0 \mid Z _ {i} + Z _ {m (i)} = 1, X _ {i}, X _ {m (i)}\right) \\ = \frac {\operatorname{pr} (Z _ {i} = 1 , Z _ {m (i)} = 0 \mid X _ {i} , X _ {m (i)})}{\operatorname{pr} (Z _ {i} = 1 , Z _ {m (i)} = 0 \mid X _ {i} , X _ {m (i)}) + \operatorname{pr} (Z _ {i} = 0 , Z _ {m (i)} = 1 \mid X _ {i} , X _ {m (i)})} \\ = \frac {e (X _ {i}) \{1 - e (X _ {m (i)}) \}}{e (X _ {i}) \{1 - e (X _ {m (i)}) \} + \{1 - e (X _ {i}) \} e (X _ {m (i)})} \\ = \frac {1}{2}. \\ \end{array}
$$

That is, the treatment assignment is identical to the MPE conditioning on the covariates and the event that each pair has a treated and control units. So we can analyze the exactly matched observational study as if it is a MPE, using either the FRT or the Neymanian approach in Chapter 7. This gives us inference on the causal effect on the treated units.

We can also find multiple control units for each treated unit. In general, we can find $M _ { i }$ matched control units for the treated unit i. When the $M _ { i } { ^ \mathrm { { \tiny ~ s } } }$ vary, it is called the variable-ratio matching (Ming and Rosenbaum, 2000, 2001; Pimentel et al., 2015). With perfect matching, the treatment assignment mechanism is identical to the general matched experiment discussed in Section 7.7. We can use the analytic results in that section to analyzed the matched observational study.

## 15.2 A more complicated but realistic scenario

Even if the control group is large, we often do not have exact matches. What we can achieve is that $X _ { i } \approx X _ { m ( i ) }$ or $X _ { i } - X _ { m ( i ) }$ is small under some distance metric. So we have only approximate matches. For example, we define

$$
m (i) = \arg \min _ {k: Z _ {k} = 0} d (X _ {i}, X _ {k}),
$$

where $d ( X _ { i } , X _ { k } )$ measures the distance between $X _ { i }$ and $X _ { k }$ . Some canonical choices of the distance are the Euclidean distance

$$
d (X _ {i}, X _ {k}) = \| X _ {i} - X _ {k} \| _ {2} ^ {2},
$$

and the Mahalanobis distance1

$$
d (X _ {i}, X _ {k}) = (X _ {i} - X _ {k}) ^ {\mathsf {T}} \Omega^ {- 1} (X _ {i} - X _ {k})
$$

with Ω being the sample covariance matrix of the $X _ { i } { } ^ { \ ' } \mathrm { s }$ from the whole population or only the control group.

I review some subtle issues about matching below. See Stuart (2010) for a review paper.

1. (one-to-one or one-to-M matching) The above discussion focused on one-to-one matching  
2. I focus on matching with replacement but some practitioners prefer matching without replacement. If the pool of control units is large, these two methods will not not matter too much for the final result. Matching with replacement is computationally more convenient, but matching without replacement involves computationally intensive discrete optimization. Matching with replacement usually gives matches of higher quality but it introduces dependence by using the same units multiple times. In contrast, the advantage of matching without replacement is the independence of matched units and the simplicity in the subsequent data analysis.  
3. Because of the residual covariate imbalance within matched pairs, it is crucial to use covariate adjustment when analyzing the data. In this case, covariate adjustment is not only for efficiency gain but also for bias correction.  
4. If X is “high dimensional”, it is likely that $d ( X _ { i } , X _ { k } )$ is too large for some unit i in the treated group and for all choices of the units in the control group. In this case, we may have to drop some units that are hard to find matches. By doing this, we effectively change the study population of interest.  
5. It is hard to avoid the above problem. For example, if $X _ { i } ~ \sim$ $\mathrm { N } ( 0 , I _ { p } ) , X _ { k } \sim \mathrm { N } ( 0 , I _ { p } )$ , and $X _ { i } \bot \bot X _ { k } .$ , then

$$
\| X _ {i} - X _ {k} \| _ {2} ^ {2} \sim \| \mathrm{N} (0, 2 I _ {p}) \| _ {2} ^ {2} = 2 \chi_ {p} ^ {2}
$$

which has mean $2 p$ and variance $8 p .$ . Theory shows that with large $p ,$ imperfect matching causes large bias in causal effect estimation. This suggests that if $p$ is large, we must have some dimension reduction before matching. Rosenbaum and Rubin (1983b) proposed to match based on the propensity score. With the estimated propensity score, we find pairs of units $\{ i , m ( i ) \}$ with small values of $| \hat { e } ( X _ { i } ) - \hat { e } ( X _ { m ( i ) } ) |$ or $| \mathrm { l o g i t } \{ \hat { e } ( X _ { i } ) \} - \mathrm { l o g i t } \{ \hat { e } ( X _ { m ( i ) } ) \} |$ , i.e., we have a one dimensional matching problem.

## 15.3 Matching estimator for the average causal effect

In a sequence of papers, Abadie and Imbens (AI) rigorously characterized the repeated sampling properties of the matching estimator and proposed the corresponding large-sample confidence intervals for the average causal effect. They chose the standard setup for observational studies with $\{ X _ { i } , Z _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \} _ { i = 1 } ^ { n } \stackrel { \mathrm { I I D } } { \sim } \{ X , Z , Y ( 1 ) , Y ( 0 ) \}$ .

## 15.3.1 Point estimation and bias correction

AI focused on 1 to M matching with replacement. For a treated unit $i ,$ we can simply impute the potential outcome under treatment as $\hat { Y _ { i } } ( 1 ) = Y _ { i }$ , and impute the potential outcome under control as

$$
\hat {Y} _ {i} (0) = M ^ {- 1} \sum_ {k \in J _ {i}} Y _ {k},
$$

where $J _ { i }$ is the set of matched units from the control group for unit i. For example, we can compute $d ( X _ { i } , X _ { k } )$ for all k in the control group, and then define $J _ { i }$ as the indices of k with the M smallest values of $d ( X _ { i } , X _ { k } )$ .

For a control unit i, we simply impute the potential outcome under control as $\hat { Y _ { i } } ( 0 ) = Y _ { i }$ , and impute the potential outcome under treatment as

$$
\hat {Y} _ {i} (1) = M ^ {- 1} \sum_ {k \in J _ {i}} Y _ {k},
$$

where $J _ { i }$ is the set of matched units from the treatment group for unit i.

The matching estimator is

$$
\hat {\tau} ^ {\mathrm{m}} = n ^ {- 1} \sum_ {i = 1} ^ {n} \{\hat {Y} _ {i} (1) - \hat {Y} _ {i} (0) \}.
$$

AI showed that $\hat { \tau } ^ { \mathrm { m } }$ has non-negligible bias especially when X is multidimensional and the number of control units is comparable to the number of treated units. Through some technical derivations, they proposed the following estimator for the bias:

$$
\hat {B} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {B} _ {i}
$$

where

$$
\hat {B} _ {i} = (2 Z _ {i} - 1) M ^ {- 1} \sum_ {k \in J _ {i}} \left\{\hat {\mu} _ {1 - Z _ {i}} \left(X _ {i}\right) - \hat {\mu} _ {1 - Z _ {i}} \left(X _ {k}\right) \right\}
$$

with $\{ \hat { \mu } _ { 1 } ( X _ { i } ) , \hat { \mu } _ { 0 } ( X _ { i } ) \}$ being the predicted outcomes by, for example, from OLS fits. For a treated unit with $Z _ { i } = 1$ , the estimated bias is

$$
\hat {B} _ {i} = M ^ {- 1} \sum_ {k \in J _ {i}} \{\hat {\mu} _ {0} (X _ {i}) - \hat {\mu} _ {0} (X _ {k}) \}
$$

which corrects the discrepancy in predicted control potential outcomes due to the mis-match in covariates; for a control unit with $Z _ { i } = 0$ , the estimates bias is

$$
\hat {B} _ {i} = - M ^ {- 1} \sum_ {k \in J _ {i}} \{\hat {\mu} _ {1} (X _ {i}) - \hat {\mu} _ {1} (X _ {k}) \}
$$

which corrects the discrepancy in predicted treated potential outcomes due to the mis-match in covariates.

The final bias corrected matching estimator is

$$
\hat {\tau} ^ {\mathrm{mbc}} = \hat {\tau} ^ {\mathrm{m}} - \hat {B},
$$

which has the following linear expansion.

Proposition 15.1 We have

$$
\hat {\tau} ^ {\mathrm{mbc}} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\psi} _ {i} \tag {15.1}
$$

where

$$
\hat {\psi} _ {i} = \hat {\mu} _ {1} (X _ {i}) - \hat {\mu} _ {0} (X _ {i}) + (2 Z _ {i} - 1) (1 + K _ {i} / M) \{Y _ {i} - \hat {\mu} _ {Z _ {i}} (X _ {i}) \}
$$

with $K _ { i }$ being the times that unit i is used as a match.

The linear expansion in Proposition 15.1 follows from simple but tedious algebra. I leave its proof as Problem 15.1. The linear expansion motivates a simple variance estimator

$$
\hat {V} ^ {\mathrm{mbc}} = \frac {1}{n ^ {2}} \sum_ {i = 1} ^ {n} (\hat {\psi} _ {i} - \hat {\tau} ^ {\mathrm{mbc}}) ^ {2},
$$

by viewing ${ \hat { \tau } } ^ { \mathrm { m b c } }$ as sample averages of the $\hat { \psi } _ { i } { ^ { \dagger } \mathrm { s } } .$ . In the literature, Abadie and Imbens $( 2 0 0 8 )$ first showed that the simple bootstrap by resampling the original data does not work for estimating the variance of the matching estimators, but their proposed variance estimation procedure is not easy to implement. Otsu and Rai (2017) proposed to bootstrap the $\hat { \psi } _ { i }$ ’s in the linear expansion, which $_ \mathrm { y }$ ields the variance estimator $\hat { V } ^ { \mathrm { m b c } }$ .

## 15.3.2 Connection with the doubly robust estimators

The bias-corrected matching estimators and the doubly robust estimators are closely related. They both equal the outcome regression estimator with some modifications based on the residuals

$$
\hat {R} _ {i} = \left\{ \begin{array}{l l} Y _ {i} - \hat {\mu} _ {1} (X _ {i}) & \text { if } Z _ {i} = 1; \\ Y _ {i} - \hat {\mu} _ {0} (X _ {i}) & \text { if } Z _ {i} = 0. \end{array} \right.
$$

For the average causal effect τ , recall the outcome regression estimator

$$
\hat {\tau} ^ {\mathrm{reg}} = n ^ {- 1} \sum_ {i = 1} ^ {n} \{\hat {\mu} _ {1} (X _ {i}) - \hat {\mu} _ {0} (X _ {i}) \}
$$

and the doubly robust estimator

$$
\hat {\tau} ^ {\mathrm{dr}} = \hat {\tau} ^ {\mathrm{reg}} + n ^ {- 1} \sum_ {i = 1} ^ {n} \left\{\frac {Z _ {i} \hat {R} _ {i}}{\hat {e} (X _ {i})} - \frac {(1 - Z _ {i}) \hat {R} _ {i}}{1 - \hat {e} (X _ {i})} \right\}.
$$

Furthermore, we can verify that ${ \hat { \tau } } ^ { \mathrm { m b c } }$ has a form very similar to ${ \hat { \tau } } ^ { \mathrm { d r } }$ .

Proposition 15.2 The bias-corrected matching estimator for τ equals

$$
\hat {\tau} ^ {\mathrm{mbc}} = \hat {\tau} ^ {\mathrm{reg}} + n ^ {- 1} \sum_ {i = 1} ^ {n} \left\{\left(1 + \frac {K _ {i}}{M}\right) Z _ {i} \hat {R} _ {i} - \left(1 + \frac {K _ {i}}{M}\right) (1 - Z _ {i}) \hat {R} _ {i} \right\}.
$$

I leave the proof of Proposition 15.2 as Problem 15.2. From Proposition 15.2, we can view matching as a nonparametric method to estimator the propensity score, and the resulting bias-corrected matching estimator as a doubly robust estimator. For instance, $1 + K _ { i } / M$ should be similar to $1 / \hat { e } ( X _ { i } )$ . When a treated unit has a small $e ( X _ { i } )$ , the resulting weight $1 / \hat { e } ( X _ { i } )$ will be large, and at the same time, it will be matched with many control units, resulting in large $K _ { i }$ and thus large $1 + K _ { i } / M$ . However, this connection also raised an obvious question regarding matching. With a fixed M, the estimator $1 + K _ { i } / M$ for $1 / e ( X _ { i } )$ will be very noisy. Allowing M to grow with the sampling size is likely to improve the matching-based nonparametric estimator for the propensity score and thus improve the asymptotic properties of the matching and bias-corrected matching estimators. Lin et al. (2023) provided a formal theory.

## 15.4 Matching estimator for the average causal effect on the treated

For the average causal effect on the treated

$$
\tau_ {\mathrm{T}} = E (Y \mid Z = 1) - E \{Y (0) \mid Z = 1 \},
$$

we only need to impute the missing potential outcomes under control for all the treated units, resulting the following estimator

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{m}} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \{Y _ {i} - \hat {Y} _ {i} (0) \}.
$$

Again it is biased with multidimensional X. Otsu and Rai (2017) propose to estimate its bias by

$$
\hat {B} _ {\mathrm{T}} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \hat {B} _ {\mathrm{T}, i}
$$

where

$$
\hat {B} _ {\mathrm{T}, i} = M ^ {- 1} \sum_ {k \in J _ {i}} \{\hat {\mu} _ {0} (X _ {i}) - \hat {\mu} _ {0} (X _ {k}) \}
$$

corrects the bias due to the mis-match of covariates for a treated unit with $Z _ { i } = 1$ .

The final bias-corrected estimator is

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{mbc}} = \hat {\tau} _ {\mathrm{T}} ^ {\mathrm{m}} - \hat {B} _ {\mathrm{T}},
$$

which has the following linear expansion.

Proposition 15.3 We have

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{mbc}} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\psi} _ {\mathrm{T}, i}, \tag {15.2}
$$

where

$$
\hat {\psi} _ {\mathrm{T}, i} = Z _ {i} \{Y _ {i} - \hat {\mu} _ {0} (X _ {i}) \} - (1 - Z _ {i}) K _ {i} / M \{Y _ {i} - \hat {\mu} _ {0} (X _ {i}) \}.
$$

I leave the proof as Problem 15.1. Motivated by Otsu and Rai (2017), we can view $\hat { \tau } _ { \mathrm { T } } ^ { \mathrm { m b c } }$ as $n / n _ { 1 }$ multiplied by the sample average of the $\psi _ { \mathrm { T } , i } \mathrm { ' s } ,$ so an intuitive variance estimator is

$$
\hat {V} _ {\mathrm{T}} ^ {\mathrm{mbc}} = \left(\frac {n}{n _ {1}}\right) ^ {2} \frac {1}{n ^ {2}} \sum_ {i = 1} ^ {n} (\hat {\psi} _ {\mathrm{T}, i} - \hat {\tau} _ {\mathrm{T}} ^ {\mathrm{mbc}} n _ {1} / n) ^ {2} = \frac {1}{n _ {1} ^ {2}} \sum_ {i = 1} ^ {n} (\hat {\psi} _ {\mathrm{T}, i} - \hat {\tau} _ {\mathrm{T}} ^ {\mathrm{mbc}} n _ {1} / n) ^ {2}.
$$

Similar to the discussion in Section 15.3.2, we can compare the doubly robust and bias-corrected matching estimators with the outcome regression estimator. For the average causal effect on the treated units $\tau _ { \mathrm { T } } ,$ recall the outcome regression estimator

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{reg}} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \{Y _ {i} - \hat {\mu} _ {0} (X _ {i}) \},
$$

and the doubly robust estimator

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{dr}} = \hat {\tau} _ {\mathrm{T}} ^ {\mathrm{reg}} - n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} \frac {\hat {e} (X _ {i})}{1 - \hat {e} (X _ {i})} (1 - Z _ {i}) \hat {R} _ {i}.
$$

Furthermore, we can verify that $\hat { \tau } _ { \mathrm { T } } ^ { \mathrm { m b c } }$ has a form very similar to $\hat { \tau } _ { \mathrm { T } } ^ { \mathrm { d r } }$ .

Proposition 15.4 The bias correction matching estimator for τT equals

$$
\hat {\tau} _ {\mathrm{T}} ^ {\mathrm{mbc}} = \hat {\tau} _ {\mathrm{T}} ^ {\mathrm{reg}} - n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} \frac {K _ {i}}{M} (1 - Z _ {i}) \hat {R} _ {i}.
$$

I leave the proof of Proposition 15.4 as Problem 15.3. Proposition 15.4 suggests that matching essentially uses $K _ { i } / M$ to estimate the odds of the treatment given covariates.

## 15.5 A case study

## 15.5.1 Experimental data

Now I revisit the LaLonde data using Sekhon (2011)’s Matching package. We have used this package several times for the dataset lalonde, and now we will use its key function Match. The experimental part gives us the following results:

```diff
> library("car")
> library("Matching")
> y = lalonde$re78
> z = lalonde$treat
> x = as.matrix(lalonde[, c("age", "educ", "black",
+    "hisp", "married", "nodegr",
+    "re74", "re75")])
>
> ## analysis the randomized experiment
> neymanols = lm(y ~ z)
> neymanols$coef[2]
z
1794.343
> sqrt(hccm(neymanols, type = "hc2")[2, 2])
[1] 670.9967
>
> xc = scale(x)
> linols = lm(y ~ z*xc)
> linols$coef[2]
z
1621.584
> sqrt(hccm(linols, type = "hc2")[2, 2])
[1] 694.7217
```

Both the unadjusted and adjusted estimators shows positive significant results on the job training program. We can analyze the data as if it is an observational study, yielding the following results:

## 15.5 A case study

```txt
> matchest.adj = Match(Y = y, Tr = z, X = x, BiasAdjust = TRUE)
> summary(matchest.adj)

Estimate... 2119.7
AI SE..... 876.42
T-stat..... 2.4185
p.val..... 0.015583

Original number of observations..... 445
Original number of treated obs..... 185
Matched number of observations..... 185
Matched number of observations (unweighted). 268
```

Both the point estimator and standard error increase, but qualitatively, the conclusion remains the same.

## 15.5.2 Observational data

Then I revisit the observational counterpart of the data:

```txt
> dat <- read.table("cps1re74.csv",header=T)
> dat$u74 <- as.numeric(dat$re74==0)
> dat$u75 <- as.numeric(dat$re75==0)
> y = dat$re78
> z = dat$treat
> x = as.matrix(dat[, c("age", "educ", "black",
+    "hispan", "married", "nodegree",
+    "re74", "re75", "u74", "u75")])
```

If we use simple OLS estimators, we obtain results that are far from the experimental benchmark:

```txt
> neymanols = lm(y ~ z)
> neymanols$coef[2]
z
-8506.495
> sqrt(hccm(neymanols, type = "hc2")[2, 2])
[1] 583.4426
>
> xc = scale(x)
> linols = lm(y ~ z*xc)
> linols$coef[2]
z
-4265.801
> sqrt(hccm(linols, type = "hc2")[2, 2])
[1] 3211.772
```

However, if we use matching, the results almost recovers those based on the experimental data:

```julia
> matchest = Match(Y = y, Tr = z, X = x, BiasAdjust = TRUE)
```

```txt
> summary(matchest)
```

```txt
Estimate... 1747.8
```

```txt
AI SE..... 916.59
```

```txt
T-stat..... 1.9068
```

```txt
p.val..... 0.056543
```

```txt
Original number of observations.... 16177
```

```txt
Original number of treated obs.... 185
```

```txt
Matched number of observations.... 185
```

```txt
Matched number of observations (unweighted). 248
```

Ignoring the ties in the matched data, we can also use the matched-pairs analysis, which again yields results similar to those based on the experimental data:

> diff = y[matchest $index.treated$ ] -
+    y[matchest $index.control$ ]
> round(summary(lm(diff ~ 1)) $coef[1, ], 2$ )
    Estimate Std. Error t value Pr(>|t|)
    1581.44    558.55    2.83    0.01
>
> diff.x = x[matchest $index.treated,$ ] -
+    x[matchest $index.control,$ ]
> round(summary(lm(diff ~ diff.x)) $coef[1, ], 2$ )
    Estimate Std. Error t value Pr(>|t|)
    1842.06    578.37    3.18    0.00

## 15.5.3 Covariate balance checks

Moreover, we can use simple OLS to check covariate balance. Before matching, the covariates are highly imbalanced, signified by many stars associated with the coefficients.

```txt
> lm.before = lm(z ~ x)
```

```txt
> summary(lm.before)
```

```txt
Call:
```

```txt
lm(formula = z ~ x)
```

```txt
Residuals:
```

```txt
Min 1Q Median 3Q Max
-0.18508 -0.01057 0.00303 0.01018 1.01355
```

```txt
Coefficients:
```

```txt
Estimate Std. Error t value Pr(>|t|)
(Intercept) 1.404e-03 6.326e-03 0.222 0.8243
xage -4.043e-04 8.512e-05 -4.750 2.05e-06 ***
xeduc 3.220e-04 4.073e-04 0.790 0.4293
```

**15.6 A case study**

<table><tr><td>xblack</td><td>1.070e-01</td><td>2.902e-03</td><td>36.871</td><td>&lt; 2e-16</td><td>***</td></tr><tr><td>xhispan</td><td>6.377e-03</td><td>3.103e-03</td><td>2.055</td><td>0.0399</td><td>*</td></tr><tr><td>xmarried</td><td>-1.525e-02</td><td>2.023e-03</td><td>-7.537</td><td>5.06e-14</td><td>***</td></tr><tr><td>xnodegree</td><td>1.345e-02</td><td>2.523e-03</td><td>5.331</td><td>9.89e-08</td><td>***</td></tr><tr><td>xre74</td><td>7.601e-07</td><td>1.806e-07</td><td>4.208</td><td>2.59e-05</td><td>***</td></tr><tr><td>xre75</td><td>-1.231e-07</td><td>1.829e-07</td><td>-0.673</td><td>0.5011</td><td></td></tr><tr><td>xu74</td><td>4.224e-02</td><td>3.271e-03</td><td>12.914</td><td>&lt; 2e-16</td><td>***</td></tr><tr><td>xu75</td><td>2.424e-02</td><td>3.399e-03</td><td>7.133</td><td>1.02e-12</td><td>***</td></tr></table>

Residual standard error : 0.09935 on 16166 degrees of freedom Multiple R - squared : 0.1274 , Adjusted R - squared : 0.1269 F - statistic : 236.1 on 10 and 16166 DF , p - value : < 2.2 e -16

But after matching, the covariates are well balanced, signified by the absence of stars for all coefficients.

```txt
> lm.after = lm(z ~ x,
+    subset = c(matchest$index.treated,
+    matchest$index.control))
> summary(lm.after)
```

Call :

lm ( formula = z \~ x , subset = c ( matchest \$ index . treated , matchest \$ index . control ))

Residuals :

```csv
Min 1Q Median 3Q Max
-0.66864 -0.49161 -0.03679 0.50378 0.65122
```

Coefficients :

<table><tr><td></td><td>Estimate</td><td>Std. Error</td><td>t value</td><td>Pr(&gt;|t|)</td></tr><tr><td>(Intercept)</td><td>6.003e-01</td><td>2.427e-01</td><td>2.474</td><td>0.0137 *</td></tr><tr><td>xage</td><td>3.199e-03</td><td>3.427e-03</td><td>0.933</td><td>0.3511</td></tr><tr><td>xeduc</td><td>-1.501e-02</td><td>1.634e-02</td><td>-0.918</td><td>0.3590</td></tr><tr><td>xblack</td><td>6.141e-05</td><td>7.408e-02</td><td>0.001</td><td>0.9993</td></tr><tr><td>xhispan</td><td>1.391e-02</td><td>1.208e-01</td><td>0.115</td><td>0.9084</td></tr><tr><td>xmarried</td><td>-1.328e-02</td><td>6.729e-02</td><td>-0.197</td><td>0.8437</td></tr><tr><td>xnodegree</td><td>-3.023e-02</td><td>7.144e-02</td><td>-0.423</td><td>0.6723</td></tr><tr><td>xre74</td><td>6.754e-06</td><td>9.864e-06</td><td>0.685</td><td>0.4939</td></tr><tr><td>xre75</td><td>-9.848e-06</td><td>1.279e-05</td><td>-0.770</td><td>0.4417</td></tr><tr><td>xu74</td><td>2.179e-02</td><td>1.027e-01</td><td>0.212</td><td>0.8321</td></tr><tr><td>xu75</td><td>-2.642e-02</td><td>8.327e-02</td><td>-0.317</td><td>0.7512</td></tr></table>

Residual standard error : 0.5043 on 485 degrees of freedom Multiple R - squared : 0.005101 , Adjusted R - squared : -0.01541 F - statistic : 0.2487 on 10 and 485 DF , p - value : 0.9909

## 15.6 Discussion

With many covariates, matching based on the original covariates may suffer from the curse of dimensionality. Rosenbaum and Rubin (1983b) suggested to use matching based on the estimated propensity score. Abadie and Imbens (2016) provided a form theory for this strategy.

## 15.7 Homework Problems

15.1 Linear expansions of the bias-corrected estimators

Prove Propositions 15.1 and 15.3.

15.2 Doubly robust form of the bias-corrected matching estimator for τ

Prove Proposition 15.2.

15.3 Doubly robust form of the bias-corrected matching estimator for τt

Prove Proposition 15.4.

15.4 Data re-analyses

In OSATE.R, I analyze two datasets using regression imputation, two IPW and the doubly robust estimators. Reanalyze them using the propensity score stratification estimator and the Abadie–Imbens matching estimator. Compare these estimators.

Note that you should choose different number of strata for the propensity score stratification estimator, and check covariate balance. You should also choose different number of matches for the matching estimator. You can even apply various estimators to the matched data. Are your results sensitive to your choices?

15.5 Data re-analyses

In Matching.R, I analyzed the LaLonde observational study using matching. Matching performs well because it gives an estimator that is close to the experimental gold standard. Reanalyze the data using the regression imputation, propensity score stratification, two IPW and the doubly robust estimators. Compare the results to the matching estimator and to the estimator from the experimental gold standard.

Note that you have many choices. For example, the number of strata for stratification and the threshold to trim to data based on the estimated propensity scores. You may consider fitting different propensity score and outcome models, e.g., including some quadratic terms of the basic covariates. You can even apply these estimators to the matched data.

This is a classic dataset and hundreds of papers have used it. You can read some references (Dehejia and Wahba, 1999; Hainmueller, 2012) and you can also be creative in your own data analysis.

## 15.6 Data re-analyses

Ho et al. (2007) is an influential paper in political science, based on which the authors have developed an R package MatchIt (Ho et al., 2011). Ho et al. (2007) analyzed two datasets, both of which are available from the Harvard Dataverse.

Reanalyze these two datasets using the methods discussed so far. You can also try other methods as long as you can justify them.

## 15.7 Recommended reading

The literature of matching estimators is massive, and three excellent review papers are Sekhon (2009), Stuart (2010) and Imbens (2015).

## Part IV



# Difficulties of Unconfoundedness in Observational Studies for Causal Effects

Part III of this book discusses causal inference with observational studies under two assumptions: unconfoundedness and overlap. Both are strong assumptions and likely to be violated in practice. This chapter will discuss the difficulties of the unconfoundedness assumption. Chapters 17–19 will discuss various strategies for sensitivity analysis in observational studies with unmeasured confounding. Chapter 20 will discuss the difficulties of the overlap assumption.

## 16.1 Some basics of the causal diagram

Pearl (1995) introduced the causal diagram as a powerful tool for causal inference in empirical research. Pearl (2000) is a textbook on the causal diagram. Here I introduce the causal diagram as an intuitive tool for illustrating the causal relationships among variables.

For example, if we have the causal diagram

![image_13](images/image_13.png)

and focus on the causal effect of Z on Y , we can read it as

$$
\left\{ \begin{array}{c} X \sim F _ {X} (x), \\ Z = f _ {Z} (X, \varepsilon_ {Z}), \\ Y (z) = f _ {Y} (X, z, \varepsilon_ {Y} (z)), \end{array} \right.
$$

where $\varepsilon _ { Z } \bot \bot \varepsilon _ { Y } ( z )$ for both $z = 0 , 1$ . In the above, covariates X are generated from a distribution $F _ { X } ( x )$ , the treatment assignment is a function of X with a random error term $\varepsilon _ { Z } .$ , and the potential outcome $Y ( z )$ is a function of X, z and a random error term $\varepsilon _ { Y } ( z )$ . We can easily read from the equations that $Z \bot \lfloor Y ( z ) \mid X , { \mathrm { i . e . } }$ , the unconfoundedness assumption holds.

If we have a causal diagram

![image_14](images/image_14.png)

we can read it as

$$
\left\{ \begin{array}{l l} X \sim F _ {X} (x), \\ U \sim F _ {U} (u), \\ Z = f _ {Z} (X, U, \varepsilon_ {Z}), \\ Y (z) = f _ {Y} (X, U, z, \varepsilon_ {Y} (z)), \end{array} \right.
$$

where $\varepsilon _ { Z } \bot \bot \varepsilon _ { Y } ( z )$ for both $z = 0 , 1$ . We can easily read from the equations that $Z \bot \bot Y ( z ) \mid ( X , U )$ but $Z \not \sqcup Y ( z ) \mid X$ , i.e., the unconfoundedness assumption holds conditioning on $( X , U )$ but does not hold conditioning on X only. In this case, U is an unmeasured confounder. In this diagram, U is called an unmeasured confounder.

## 16.2 Assessing ignorability

The weak ignorability

$$
Z \bot Y (1) \mid X, \quad Z \bot Y (0) \mid X
$$

implies that

$$
\operatorname{pr} \{Y (1) \mid Z = 1, X \} = \operatorname{pr} \{Y (1) \mid Z = 0, X \},
$$

$$
\operatorname{pr} \{Y (0) \mid Z = 1, X \} = \operatorname{pr} \{Y (0) \mid Z = 0, X \}.
$$

So the ignorability assumption basically requires that the counterfactual distribution pr $\{ Y ( 1 ) \mid Z = 0 , X \}$ equals the observed distribution pr $\{ Y ( 1 ) \mid$ $Z = 1 , X \}$ , and the counterfactual distribution pr $\{ Y ( 0 ) \mid Z = 1 , X \}$ equals the observed distribution $\mathrm { p r } \{ Y ( 0 ) \mid Z = 0 , X \}$ . Because the counterfactual distributions are not directly identifiable from the data, the ignorability assumption is fundamentally untestable without additional assumptions. I will discuss two strategies to assess ignorability. Here, “assess” is a weaker notion than $\mathrm { ^ { 6 6 5 t } } ^ { \mathrm { , 5 } }$ . The former is referred to as supplementary analysis that support or undermine the initial analysis, but the latter is referred to formal statistical testing.

## 16.2.1 Using negative outcomes

Assume that $Y ^ { \mathrm { n } }$ is an outcome similar to Y and ideally, shares the same confounding structure as Y . If we believe $Z \bot Y ( z ) \mid X$ , then we also tend to believe $Z \bot Y ^ { \mathrm { n } } ( z )$ | X. Moreover, we know, a priori, the effect of $Z$ on $Y ^ { \mathrm { n } }$ :

$$
\tau (Z \to Y ^ {\mathrm{n}}) = E \{Y ^ {\mathrm{n}} (1) - Y ^ {\mathrm{n}} (0) \}.
$$

An important example is that $\tau ( Z  Y ^ { \mathrm { n } } ) = 0$ . A causal diagram satisfying these requirements is below:

![image_15](images/image_15.png)

```mermaid
graph TD
  X --> Yn
  X --> Z
  Z --> Y
```

Example 16.1 Cornfield et al. (1959) studied the causal role of cigarette smoking on lung cancer based on observational studies. They controlled for many important background variables but it is still possible to have some unmeasured confounders biasing the observed effects. To strengthen the evidence, they also reported the effect of cigarette smoking on car accident which was close to zero, the anticipated effect based on biology. So even if they could not rule out unmeasured confounding in the analysis, this supplementary analysis based on a negative outcome makes the evidence of the the causal effect of cigarette smoking on lung cancer stronger.

Example 16.2 Imbens and Rubin (2015) suggested using the lagged outcome as a negative outcome. In most cases, it is reasonable to believe that the lagged outcome and the outcome have similar confounding structure. Since the lagged outcome happens before the treatment, the average causal effect on it must be 0. However, their suggestion should be used with caution since in most studies we simply treat lagged outcomes as an observed confounder.

In some sense, the covariate balance check in Chapter 11 is a special case of using negative controls. Similar to the problem of using lagged outcomes as negative controls, those covariates are usually a part of the ignorability assumption. Therefore, the failure of covariate balance check does not really falsify the ignorability assumption but rather the model specification of the propensity score.

Example 16.3 Observational studies in elderly persons have shown that vaccination against influenza remarkably reduces one’s risk of pneumonia/influenza hospitalization and all-cause mortality in the following season, after adjustment for measured covariates. Jackson et al. (2006) were skeptical about the large magnitude and thus conducted supplementary analysis on negative outcomes. Vaccination often begins in autumn, but influenza transmission is often minimal until winter. Based on biology, the effect of vaccination should be most prominent during influenza season. But Jackson et al. (2006) found greater effect before the influenza season, suggesting that the observed effect is due to unmeasured confounding.

## 20416 Difficulties of Unconfoundedness in Observational Studies for Causal Effects

Jackson et al. (2006) seems the most convincing one since the influenzarelated outcomes before and during the influenza season should have similar confounding patterns. Cornfield et al. (1959)’s additional evidence seems weaker since car accident and lung cancer have very different causal mechanisms with respect to cigarette smoking. In fact, Fisher (1957)’s critique was that the relationship between cigarette smoking on lung cancer may be due to an unobserved genetic factor. Such a genetic factor might affect cigarette smoking and lung cancer simultaneously, but it seems unlikely that it also affects car accident.

Lipsitch et al. (2010) is a recent article on negative outcomes. Rosenbaum (1989) discussed the role of known effects in causal inference.

## 16.2.2 Using negative exposures

Negative exposures are duals of negative outcomes. Assume $Z ^ { \mathrm { n } }$ is a treatment variable similar to $Z$ and shares the same confounding structure as Z. If we believe $Z \bot \bot Y ( z ) \mid X .$ , then we tend to believe $Z ^ { \mathrm { n } } \bot \bot { \bar { Y ( z ) } } \mid X .$ . Moreover, we know, a priori, the effect of $Z ^ { \mathrm { n } }$ on $Y$

$$
\tau (Z ^ {\mathrm{n}} \to Y) = E \{Y (1 ^ {\mathrm{n}}) - Y (0 ^ {\mathrm{n}}) \}.
$$

An important example is that $\tau ( Z ^ { \mathrm { n } } \to Y ) = 0$ . A causal diagram satisfying these requirements is below:

![image_16](images/image_16.png)

```mermaid
graph TD
  ZN["Z^n"] --> X
  X --> Z
  X --> Y
  Z --> Y
```

Example 16.4 Sanderson et al. (2017) give many examples of negative exposures in determining the effect of intrauterine exposure on later outcomes by comparing the association of a maternal exposure during pregnancy with the outcome of interest, with the association of the paternal exposure with the same outcome. They review studies on the effect of maternal and paternal smoking on offspring outcomes, the effect of maternal and paternal BMI on later offspring BMI and autism spectrum disorder. In these examples, we expect the the association of the maternal exposure with the outcome is larger than that of the paternal exposure with the outcome.

## 16.2.3 Summary

The unconfoundedness assumption is fundamentally untestable without additional assumptions. Although negative outcomes and negative controls in observational studies cannot prove or disprove unconfoundedness, using them in supplementary analyses can strengthen the evidence for causation. However, it is often non-trivial to conduct this type of supplementary analyses because it involves more data and more importantly, deeper understanding of the causal problems in order to find convincing negative outcomes and negative controls.

## 16.3 Problems of over adjustment

We have discussed many methods for estimating causal effects under ignorability:

$$
Z \bot \{Y (1), Y (0) \} \mid X.
$$

This is an assumption conditioning on X. It is crucial to select the right set of X that ensure the conditional independence. Rosenbaum (2002b) wrote that“there is no reason to avoid adjustment for a variable describing subjects before treatment.” Similarly, Rubin (2007) wrote that “typically, the more conditional an assumption, the more acceptable it is.” Both argued that we should control for all observed pretreatment covariate. VanderWeele and Shpitser (2011) called it the pretreatment criterion. Pearl disagreed with this recommendation and gave two counterexamples below.

## 16.3.1 M-bias

M-bias appears in the following causal diagram with an M-structure:

![image_17](images/image_17.png)

```mermaid
graph TD
  U1 --> X
  U2 --> X
  Z --> X
  Y --> X
```

We can read from the diagram the data generating process:

$$
\left\{ \begin{array}{l} U _ {1} \text {卄} U _ {2}, \\ X = f _ {X} (U _ {1}, U _ {2}, \varepsilon_ {X}), \\ Z = f _ {Z} (U _ {1}, \varepsilon_ {Z}), \\ Y = Y (z) = f _ {Y} (U _ {2}, \varepsilon_ {Y}), \end{array} \right.
$$

where $( \varepsilon _ { X } , \varepsilon _ { Z } , \varepsilon _ { Y } )$ are independent random error terms. In the above causal diagram, X is observed, but $U _ { 1 }$ and $U _ { 2 }$ are unobserved. If we change the value of $Z ,$ , the value of $Y$ will not change at all. So the true causal effect of $Z$ on $Y$ must be 0. From the data-generating equations, we can easy read that $Z \bot \bot Y$ ,

## 20616 Difficulties of Unconfoundedness in Observational Studies for Causal Effects

so the association between Z and $Y$ is 0, and, in particular,

$$
\tau_ {\mathrm{PF}} = E (Y \mid Z = 1) - E (Y \mid Z = 0) = 0.
$$

This means that without adjusting for the covariate X, the simple estimator is unbiased for the true parameter.

However, if we condition on X, then $U _ { 1 } \not \mu U _ { 2 } \mid X$ , and consequently, $Z \not \bot Y \mid$ | X and

$$
\int \{E (Y \mid Z = 1, X = x) - E (Y \mid Z = 0, X = x) \} F (\mathrm{d} x) \neq 0
$$

in general. To gain intuition, we consider the case with Gaussian linear mod-$\mathrm { e l s } ^ { \mathrm { \scriptsize { 1 } } }$ :

$$
\left\{ \begin{array}{l} X = a U _ {1} + b U _ {2} + \varepsilon_ {X}, \\ Z = c U _ {1} + \varepsilon_ {Z}, \\ Y = Y (z) = d U _ {2} + \varepsilon_ {Y}, \end{array} \right.
$$

where $( U _ { 1 } , U _ { 2 } , \varepsilon _ { X } , \varepsilon _ { Z } , \varepsilon _ { Y } ) \stackrel { \mathrm { I I D } } { \sim } \mathrm { N } ( 0 , 1 )$ . We have

$$
\operatorname{cov} (Z, Y) = \operatorname{cov} \left(c U _ {1} + \varepsilon_ {Z}, d U _ {2} + \varepsilon_ {Y}\right) = 0,
$$

but by the result in Problem 1.2, the partial correlation coefficient between Z and $\check { Y }$ given X is

$$
\rho_ {Z Y | X} = \frac {\rho_ {Z Y} - \rho_ {Z X} \rho_ {Y X}}{\sqrt {1 - \rho_ {Z X} ^ {2}} \sqrt {1 - \rho_ {Y X} ^ {2}}} \propto - \rho_ {Z X} \rho_ {Y X} \propto - \operatorname{cov} (Z, X) \operatorname{cov} (Y, X) = - a b c d,
$$

the product of the coefficients on the path from Z to Y . So the unadjusted estimator is unbiased but the adjusted estimator has bias proportional to abcd.

The following simple example illustrates M-bias.

```txt
> n = 10^6
>
> ## M bias
> U1 = rnorm(n)
> U2 = rnorm(n)
> X = U1 + U2 + rnorm(n)
> Z = U1 + rnorm(n)
> Y = U2 + rnorm(n)
>
> round(summary(lm(Y ~ Z))$coef[2, 1], 3)
[1] -0.001
> round(summary(lm(Y ~ Z + X))$coef[2, 1], 3)
[1] -0.201
>
```

> Z = (Z >= 0)
> round(summary(lm(Y ~ Z))$coef[2, 1], 3)
[1] -0.002
> round(summary(lm(Y ~ Z + X))$coef[2, 1], 3)
[1] -0.421

## 16.3.2 Z-bias

Consider the following causal diagram:

![image_18](images/image_18.png)

```mermaid
graph TD
  X -->|a| Z
  Z -->|b| U
  Z -->|τ| Y
  U -->|c| Y
```

with the data generating process2

$$
\left\{ \begin{array}{l} Z = a X + b U + \varepsilon_ {Z}, \\ Y (z) = \tau z + c U + \varepsilon_ {Y}, \end{array} \right.
$$

where $( U , X , \varepsilon _ { Z } , \varepsilon _ { Y } ) \stackrel { \mathrm { I I D } } { \sim } \mathrm { N } ( 0 , 1 )$ . In this data generating process, we have $X \bot \bot U , X \bot Z$ , and X affects Y only through Z.

The unadjusted estimator is

$$
\tau_ {\mathrm{unadj}} = \frac {\operatorname{cov} (Z , Y)}{\operatorname{var} (Z)} = \frac {\operatorname{cov} (Z , \tau Z + c U)}{\operatorname{var} (Z)} = \tau + \frac {c \operatorname{cov} (a X + b U , U)}{\operatorname{var} (Z)} = \tau + \frac {c b}{a ^ {2} + b ^ {2} + 1},
$$

which has bias $b c / ( a ^ { 2 } + b ^ { 2 } + 1 )$ . The adjusted estimator from the OLS of Y on $( Z , X )$ satisfies

$$
\left\{ \begin{array}{l} E \{Z (Y - \tau_ {\mathrm{adj}} Z - \alpha X) \} = 0, \\ E \{X (Y - \tau_ {\mathrm{adj}} Z - \alpha X) \} = 0, \end{array} \right.
$$

which is equivalent to

$$
\left\{ \begin{array}{l} E (Z Y) = \tau_ {\mathrm{adj}} \mathrm{var} (Z) + \alpha E (X Z), \\ E (X Y) = \tau_ {\mathrm{adj}} E (X Z) + \alpha \mathrm{var} (X). \end{array} \right.
$$

We need to solve for $( \tau _ { \mathrm { a d j } } , \alpha )$ from the above two linear equations:

$$
\begin{array}{l} \tau_ {\mathrm{adj}} = \frac {\left| \begin{array}{c c} E (Z Y) & E (X Z) \\ E (X Y) & \operatorname{var} (X) \end{array} \right|}{\left| \begin{array}{c c} \operatorname{var} (Z) & E (X Z) \\ E (X Z) & \operatorname{var} (X) \end{array} \right|} = \frac {E (Z Y) \operatorname{var} (X) - E (X Z) E (X Y)}{\operatorname{var} (Z) \operatorname{var} (X) - E (X Z) ^ {2}} \\ = \frac {\tau (a ^ {2} + b ^ {2} + 1) + b c - a \tau a}{(a ^ {2} + b ^ {2} + 1) - a ^ {2}} = \frac {\tau (b ^ {2} + 1) + b c}{b ^ {2} + 1} = \tau + \frac {b c}{b ^ {2} + 1}, \\ \end{array}
$$

which has bias bc/(b2 + 1).

So the unadjusted estimator has smaller bias than the adjusted estimator. More interestingly, the stronger the association between X and Z is (measured by a), the larger the bias of the adjusted estimator is.

The mathematical derivation is not extremely hard. But this type of bias seems rather mysterious. Here is the intuition. The treatment is a function of X, U, and other random errors. If we condition on X, it is merely a function of U and other random errors. Therefore, conditioning makes Z less random, and more critically, makes the unmeasured confounder U play a more important role in Z. Consequently, the confounding bias due to U is amplified by conditioning on X. This idealized example illustrates the danger of over adjusting for some covariates.

Heckman and Navarro-Lozano (2004) observed the phenomenon in simulation studies, and Wooldridge (2016, technical report in 2006) verified it in linear models. Pearl (2010, 2011) explained it using causal diagrams. This type of bias is called Z-bias because in Pearl’s original papers, he used the symbol Z for our variable X. Throughout the book, however, Z is used for the treatment variable. In Part V of this book, we will call Z an instrumental variable if it satisfies the causal diagram presented in this subsection. This justifies the instrumental variable bias as another name of this type of bias.

The following simple example illustrates Z-bias.

```txt
> X = rnorm(n)
> U = rnorm(n)
> Z = X + U + rnorm(n)
> Y = U + rnorm(n)
>
> round(summary(lm(Y ~ Z))$coef[2, 1], 3)
[1] 0.334
> round(summary(lm(Y ~ Z + X))$coef[2, 1], 3)
[1] 0.501
>
> Z = 2*X + U + rnorm(n)
> round(summary(lm(Y ~ Z))$coef[2, 1], 3)
[1] 0.167
> round(summary(lm(Y ~ Z + X))$coef[2, 1], 3)
[1] 0.5
>
> Z = 10*X + U + rnorm(n)
> round(summary(lm(Y ~ Z))$coef[2, 1], 3)
[1] 0.01
> round(summary(lm(Y ~ Z + X))$coef[2, 1], 3)
[1] 0.5
```

## 16.3.3 What covariates should we adjust for in observational studies?

We never know the true underlying data generating process which can be quite complicated. However, the following causal diagram helps to clarify many ideas. It already rules out the possibility of M-bias discussed in Section 16.3.1.

![image_19](images/image_19.png)

```mermaid
graph TD
  XZ --> Z
  X --> Y
  XY --> Y
  Z --> Y
  Z --> XI
  Y --> XI
```

$X _ { R }$

The covariates above have different features:

1. X affects both the treatment and the outcome. Conditioning on X ensures ignorability, so we should control for X.  
2. $X _ { R }$ is pure random noise not affecting either the treatment or the outcome. Including it in analysis does not bias the estimate but it introduces unnecessary variability in finite sample.  
3. $X _ { Z }$ is an instrumental variable that affects the outcome only through the treatment. In the diagram above, including it in analysis does not bias the estimate although it increases variability. However, with unmeasured confounding, including it in analysis amplifies the bias as shown in Section 16.3.1.  
4. $X _ { Y }$ affects the outcome only but not the treatment. Without conditioning on it, the ignorability still holds. Since they are predictive to the outcome, including them in analysis often improves precision.  
5. $X _ { I }$ is affected by the treatment and outcome. It is a post-treatment variable, not a pretreatment covariate. We should not include it if the goal is to infer the effect of the treatment on the outcome. We will discuss issues with post-treatment variables in causal inference in Part VI of this book.

If we believe the above causal diagram, we should adjust for at least X to remove bias and more ideally, further adjust for $X _ { Y }$ to reduce variance.

## 16.4 Homework Problems

## 16.1 Cochran’s formula or the omitted variable bias formula

Sir David Cox calls the following result Cochran’s formula (Cochran, 1938; Cox, 2007) and econometricians call it the omitted variable bias formula $( \mathrm { A n - }$ grist and Pischke, 2008). A special case appeared in Fisher (1925). It is also a sister of the Frisch–Waugh–Lovell Theorem in Chapter A2.3.

The formula has two versions. All vectors below are column vectors.

1. (Population version) Assume $( y _ { i } , x _ { 1 i } , x _ { 2 i } ) _ { i = 1 } ^ { n }$ are iid, where $y _ { i }$ is a scalar, $x _ { i 1 }$ has dimension K, and $x _ { i 2 }$ has dimension L.

We have the following OLS decompositions of random variables

$$
y _ {i} = \beta_ {1} ^ {\mathsf {T}} x _ {i 1} + \beta_ {2} ^ {\mathsf {T}} x _ {2 i} + \varepsilon_ {i}, \tag {16.1}
$$

$$
y _ {i} = \gamma^ {\mathsf {T}} x _ {i 1} + e _ {i}, \tag {16.2}
$$

$$
x _ {i 2} = \delta^ {\mathsf {T}} x _ {i 1} + v _ {i}. \tag {16.3}
$$

Equation (16.1) is called the long regression, and Equation (16.2) is called the short regression. In Equation (16.3), δ is a matrix because it is a regression of a vector on a vector. You can view (16.3) as regression of each component of $x _ { i 2 }$ on $x _ { i 1 }$ .

Show that $\gamma = \beta _ { 1 } + \delta \beta _ { 2 }$ .

2. (Sample version) We have an $n \times 1$ vector $Y ,$ an $n \times K$ matrix $X _ { 1 } ,$ and an $n \times L$ matrix $X _ { 2 }$ . We do not assume any randomness. All results below are purely linear algebra.

We can obtain the following OLS fits:

$$
{ Y } { = } { X _ { 1 } \hat { \beta } _ { 1 } + X _ { 2 } \hat { \beta } _ { 2 } + \hat { \varepsilon } , }
$$

$$
Y = X _ {1} \hat {\gamma} + \hat {e},
$$

$$
{X _ {2}} = {X _ {1} \hat {\delta} + \hat {v},}
$$

where ˆε, e, ˆ vˆ are the residuals. Again, the last OLS fit means the OLS fit of each column of $X _ { 2 }$ on $X _ { 1 }$ , and therefore the residual ˆv is an $n \times L$ matrix.

Show that $\hat { \gamma } = \hat { \beta } _ { 1 } + \hat { \delta } \hat { \beta } _ { 2 }$ .

Remark: The product terms $\delta \beta _ { 2 }$ and $\hat { \delta } \hat { \beta } _ { 2 }$ are often referred to as the omitted-variable bias at the population level and sample level, respectively.

## 16.2 Recommended reading

Imbens (2020) reviews and compares the roles of potential outcomes and causal diagrams for causal inference.

## 17

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

# Overlap in Observational Studies: Difficulties and Opportunities

## 20.1 Implications of overlap

In Part III of this book, causal inference with observational studies relies on two critical assumptions: unconfoundedness

$$
Z \bot \{Y (1), Y (0) \} \mid X
$$

and overlap

$$
0 <   e (X) <   1.
$$

D’Amour et al. (2021) pointed out the tension between these two assumptions: typically, more covariates make the unconfoundedness assumption more plausible (ignoring M-bias discussed in Chapter 16.3.1), but more covariates make the overlap assumption less plausible because the treatment becomes more predictable.

If some units has $e ( X ) = 0 { \mathrm { ~ o r ~ } } e ( X ) = 1$ , then we have philosophic difficulty of thinking about the counterfactual potential outcomes (King and Zeng, 2006). In particular, if a unit deterministically receives the treatment, then it may not be meaningful to conceive its potential outcome under control; vice versa. Even if the true propensity score is not exactly 0 or 1, the estimated propensity score can be very close to 0 or 1 in finite sample, which makes the estimators based on inverse probability weighting numerically unstable. Many statistical analyses in fact require a strict version of overlap:

Assumption 20.1 (strict overlap) $\eta \leq e ( X ) \leq 1 - \eta$ for some $\eta \in ( 0 , 1 / 2 )$ .

However, D’Amour et al. (2021, Corollary 1) showed that Assumption 20.1 has very strong implications. For simplicity, I present only one of their results. Let $X _ { k } \ ( k = 1 , \ldots , p )$ be the kth component of the covariate $X =$ $( X _ { 1 } , \ldots , X _ { p } )$ , and $e = \operatorname { p r } ( Z = 1 )$ be the proportion of the treated units.

Theorem 20.1 Assumption 20.1 implies that $\eta \leq e \leq 1 - \eta$ and

$$
\begin{array}{l} p ^ {- 1} \sum_ {k = 1} ^ {p} \left| E (X _ {k} \mid Z = 1) - E (X _ {k} \mid Z = 0) \right| \\ \leq p ^ {- 1 / 2} C ^ {1 / 2} \left\{e \lambda_ {1} ^ {1 / 2} + (1 - e) \lambda_ {0} ^ {1 / 2} \right\}, \tag {20.1} \\ \end{array}
$$

where

$$
C = \frac {(e - \eta) (1 - e - \eta)}{e ^ {2} (1 - e) ^ {2} \eta (1 - \eta)}
$$

is a positive constant depending only on $( e , \eta )$ , and $\lambda _ { 1 }$ and $\lambda _ { 0 }$ are the maximum eigenvalues of the covariance matrices cov $( X \mid Z = 1 )$ and $\operatorname { c o v } ( X \mid Z = 0 )$ , respectively.

What is the order of the maximum eigenvalues in Theorem 20.1? D’Amour et al. (2021) showed that it is usually smaller than $O ( p )$ unless the components of X are highly correlated. If the components of X are highly correlated, then some components are redundant after including other components. If the components of $X$ are not highly correlated, then the right-hand side converges to zero. So the average difference in means of the covariates is close to zero, that is, the treatment and control groups are nearly balanced in means averaged over all dimensions of the covariates. Mathematically, the left-hand side of (20.1) converging to zero rules out the possibility that all dimensions of X have non-vanishing difference in means across treatment and control groups. It is a strong requirement in observational studies with many covariates.

## 20.1.1 Trimming in the presence of limited overlap

When Assumption 20.1 does not hold, it is common to trim the units based on the estimated propensity scores (Crump et al., 2009; Yang and Ding, 2018). Trimming drops units within regions of little overlap, which changes the population and estimand. The restrictive implications of overlap in Assumption 20.1 suggest that trimming must be employed more often and one may need to trim a large proportion of units to achieve desirable overlap in high dimensions.

## 20.1.2 Outcome modeling in the presence of limited overlap

The somewhat negative results in D’Amour et al. (2021) also highlight the limitation of focusing only on the propensity score in the presence of limited overlap. With high dimensional covariates, outcome modeling becomes more important. In particular, if the outcome means only depend on a function of the original covariates in that

$$
E \{Y (z) \mid X \} = f _ {z} (r (X)), \quad (z = 0, 1)
$$

then it suffices to control for $r ( X )$ , a lower dimensional summary of the original covariates. Due to the dimension reduction, the strict overlap condition on $r ( X )$ can be much weaker than the strict overlap condition on X. This is conceptually straightforward, but the corresponding theory and methods are missing.

## 20.2 Causal inference with no overlap: regression discontinuity

Starting from the simple case with a univariate X. An extreme treatment assignment mechanism is a deterministic one:

$$
Z = 1 (X \geq x _ {0}),
$$

where x0 is a predetermined threshold. An interesting consequence of this assignment is that the unconfoundedness assumption holds automatically:

$$
Z \bot \{Y (1), Y (0) \} \mid X
$$

because Z is a deterministic function of X. However, the overlap assumption is violated by definition:

$$
e (X) = \operatorname{pr} (Z = 1 \mid X) = 1 (X \geq x _ {0}) = \left\{ \begin{array}{l l} 1 & \text { if } X \geq x _ {0}, \\ 0 & \text { if } X <   x _ {0}. \end{array} \right.
$$

So our analytic strategies discussed in Part IV are no longer applicable here. We must change our perspective.

The discussion here seems contrived, with a deterministic treatment assignment. Interestingly, it has many applications in practice, and is called regression discontinuity. Below, I first review some canonical examples and then give a mathematical formulation of this type of studies.

## 20.2.1 Examples and graphical diagnostics

Example 20.1 Thistlethwaite and Campbell (1960) first proposed the idea of regression-discontinuity analysis. Their motivating example was to study the effect of students’ winning Certificated of Merit on later career plans, where the Certificated of Merit was determined by whether the Scholarship Qualifying Test score was above a certain threshold. Their initial analysis was mainly graphical. Figure 20.1 shows one of their graphs.

Example 20.2 Bor et al. (2014) used regression discontinuity to study the effect of when to start HIV patients with antiretroviral on their mortality, where the treatment is determined by whether the patients’ CD4 counts were below 200 cells/µL.1

Example 20.3 Carpenter and Dobkin (2009) studied the effect of alcohol consumption on mortality, which leverages the minimum legal drinking age as a discontinuity for alcohol consumption. They derived mortality data from the National Center for Health Statistics, including the decedent’s date of birth and date of death. They computed age profile of deaths per 100,000 person years with outcomes measured by the following nine variables:

<table><tr><td>all</td><td>all deaths, the sum of internal and external</td></tr><tr><td>internal</td><td>deaths due to internal causes</td></tr><tr><td>external</td><td>deaths due to external causes, the sum of the rest</td></tr><tr><td>homicide</td><td>homicides</td></tr><tr><td>suicide</td><td>suicides</td></tr><tr><td>mva</td><td>motor vehicle accidents</td></tr><tr><td>alcohol</td><td>deaths with a mention of alcohol</td></tr><tr><td>drugs</td><td>deaths with a mention of drug use</td></tr><tr><td>externalother</td><td>deaths due to other external causes</td></tr></table>

Figure 20.2 plots the number of deaths per 100,000 person years for nine measures based on the data used by Angrist and Pischke (2014). From the jumps at age 21, it seems obvious that there is an increase of mortality at age 21, primarily due to motor vehicle accidents. I leave the formal analysis as Problem 20.3.

## 20.2.2 A mathematical formulation of regression discontinuity

The technical term for the variable X that determines the treatment is the running variable. Intuitively, regression discontinuity can identify a local average causal effect at the cutoff point x0:

$$
\tau (x _ {0}) = E \{Y (1) - Y (0) \mid X = x _ {0} \}.
$$

In particular, for the potential outcome under treatment, we have

$$
E \{Y (1) \mid X = x _ {0} \} = \lim _ {\varepsilon \rightarrow 0 +} E \{Y (1) \mid X = x _ {0} + \varepsilon \} \tag {20.2}
$$

$$
= \lim _ {\varepsilon \rightarrow 0 +} E \{Y (1) \mid Z = 1, X = x _ {0} + \varepsilon \} \tag {20.3}
$$

$$
= \lim _ {\varepsilon \to 0 +} E (Y \mid Z = 1, X = x _ {0} + \varepsilon), \tag {20.4}
$$

where (20.2) holds if $E \{ Y ( 1 ) \mid X = x \}$ is continuous from the right at $x _ { 0 }$ and (20.3) follows by the definition of Z. Similarly, for the potential outcome under control, we have

$$
E \{Y (0) \mid X = x _ {0} \} = \lim _ {\varepsilon \rightarrow 0 +} E (Y \mid Z = 0, X = x _ {0} - \varepsilon)
$$

if $E \{ Y ( 0 ) \mid X = x \}$ is continuous from the left at $x _ { 0 }$ . So the local average causal effect at $x _ { 0 }$ can be identified by the difference of the two limits. I summarize the key identification result below.

Theorem 20.2 Assume that $E \{ Y ( 1 ) \mid X = x \}$ is continuous from the right at $x _ { 0 }$ and $E \{ Y ( 0 ) \mid X = x \}$ is continuous from the left at $x _ { 0 }$ . Then the local average treatment effect at $X = x _ { 0 }$ is identified by

$$
\tau (x _ {0}) = \lim _ {\varepsilon \to 0 +} E (Y \mid Z = 1, X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \to 0 +} E (Y \mid Z = 0, X = x _ {0} - \varepsilon).
$$

Since the right-hand side of the above equation only involves observables, the parameter $\tau ( x _ { 0 } )$ is nonparametrically identified. However, the form of the identification formula is totally different from what we derived before. In particular, the identification formula involves limits of two conditional expectation functions.

## 20.2.3 Regressions near the boundary

If we are lucky, graphical diagnostic sometimes can clearly show the causal effect at the cutoff point. However, many outcomes are noisy so graphical diagnostic is not enough. Figure 20.3 shows two examples with obvious jumps at the cutoff point and two examples without obvious jumps, although the underlying data generating processes all have discontinuities.

Assume that $E ( Y \mid Z = 1 , X = x ) = \gamma _ { 1 } + \beta _ { 1 } x$ and $E ( Y \mid Z = 0 , X = x ) =$ $\gamma _ { 0 } + \beta _ { 0 } x$ are linear in x. We can run OLS based on the treated and control data to obtain the fitted lines $\hat { \gamma } _ { 1 } + \hat { \beta } _ { 1 } x$ and $\hat { \gamma } _ { 0 } + \hat { \beta } _ { 0 } x .$ , respectively. We can then estimate the average causal effect at the point $X = x _ { 0 }$ as

$$
\hat {\tau} (x _ {0}) = (\hat {\gamma} _ {1} - \hat {\gamma} _ {0}) + (\hat {\beta} _ {1} - \hat {\beta} _ {0}) x _ {0}.
$$

Numerically, $\hat { \tau } ( x _ { 0 } )$ is identical to the coefficient of $Z _ { i }$ in the OLS

$$
Y _ {i} \sim \{1, Z _ {i}, X _ {i} - x _ {0}, Z _ {i} (X _ {i} - x _ {0}) \}, \tag {20.5}
$$

and it is also identical to the coefficient of $Z _ { i }$ in the OLS

$$
Y _ {i} \sim \{1, Z _ {i}, R _ {i}, L _ {i} \}, \tag {20.6}
$$

where

$$
R _ {i} = \max (X _ {i} - x _ {0}, 0), \quad L _ {i} = \min (X _ {i} - x _ {0}, 0)
$$

indicate the right and left parts of $X _ { i } - x _ { 0 }$ , respectively. I leave the algebraic details to Problem 20.1.

However, this approach may be sensitive to the violation of the linear model. Theory suggests running regression using only the local observations near the cutoff point2. However, the rule for choosing the “local points” are quite involved. Fortunately, the rdrobust function in the rdrobust package in R implements various choices of “local points.” Since choosing the “local points” is the key in regression discontinuity, it seems more sensible to report estimates and confidence intervals based on various choices of the “local points.”

## 20.2.4 An example

Lee (2008) gave a famous example of using regression discontinuity to study the incumbency advantage in the U.S. House. He wrote that “incumbents are, by definition, those politicians who were successful in the previous election. If what makes them successful is somewhat persistent over time, they should be expected to be somewhat more successful when running for re-election.” Therefore, this is a fundamentally challenging causal inference problem. The regression discontinuity is a clever study design to study this problem.

The running variable is the lagged vote in the previous election centered at 0, and the outcome is the vote in current election, with units being the congressional districts. The treatment is the binary indicator for being the current incumbent party in a district, determined by the lagged vote. Figure 20.4 show the raw data.

The rdrobust function gives three sets of the point estimate and confidence intervals. They all suggest positive incumbency advantage.

```txt
> library(rdrobust)
> library(rddtools)
> data(house)
> RDDest = rdrobust(house$y, house$x)
[1] "Mass points detected in the running variable."
> cbind(RDDest$coef, RDDest$ci)
Coeff CI Lower CI Upper
Conventional 0.06372533 0.04224798 0.08520269
Bias-Corrected 0.05937028 0.03789292 0.08084763
Robust 0.05937028 0.03481238 0.08392818
```

Figure 20.5 shows the point estimates and the confidence intervals based on OLS with different choices of the local points defined by $| X | < x _ { 0 }$ . While the point estimates and the confidence intervals are sensitive to the choice of $x _ { 0 } .$ , the qualitative result remains the same as above.

## 20.2.5 Problems of regression discontinuity

What can go wrong with the regression discontinuity analysis? The technical challenge is to specify the neighborhood near the cutoff point. We have discussed this issue above.

In addition, Theorem 20.2 holds under a continuity condition. It may be violated in practice. For instance, if the mortality rate jumps at the age of 21, then the jumps in Figure 20.2 may not be due to the change of drinking behavior due to the legal drink age. However, it is hard to check the violation of the continuity condition empirically.

McCrary (2008) proposed an indirect test for the validity of the regression discontinuity. He suggested checking the density of the running variable at the cutoff point. The discontinuity in the density of the running variable at the cutoff point may suggest that some units were able to manipulate their treatment status perfectly.

## 20.3 Homework Problems

## 20.1 Linear potential outcome models

This problem gives more details for the numerical equivalence in Section 20.2.3.

Show that $\hat { \tau } ( x _ { 0 } )$ equals the coefficients of $Z _ { i }$ in OLS fits (20.5) and (20.5).

Hint: It is helpful for start with the figures of $Z _ { i } ( X _ { i } - x _ { 0 } ) , L _ { i } ,$ and $R _ { i }$ with $X _ { i }$ on the x-axis. The conclusion holds by reparametrizating the OLS regressions.

## 20.2 Simulation for regression discontinuity

RDDnumerical.R simulates potential outcomes from linear models and generates Figure 20.3. Change them to nonlinear models, and compare different point estimators and confidence intervals, including the biases and variances of the point estimators, and the coverage properties of confidence intervals.

## 20.3 Re-analysis of the data on the minimum legal drink age

Analyze the data mlda.csv of Carpenter and Dobkin (2009).

## 20.4 Recommended reading

D’Amour et al. (2021) discussed the implications of overlap with high dimensional covariates.

Thistlethwaite and Campbell (1960)’s original paper on regression discontinuity was re-printed as Thistlewaite and Campbell (2016) with many insightful comments. Coincidentally, Thistlethwaite and Campbell (1960) and Rubin (1974) were both published in the Journal of Educational Psychology.

## Part V

## Instrumental variables

## 21

# An Experimental Perspective

The instrumental variable method has been a powerful tool in econometrics. It identifies causal effects in studies without unconfoundedness between the treatment and the outcome. It relies on an additional variable, called the instrumental variable (IV), that satisfies certain conditions. These conditions may not be easy to digest when you read for the first time. In some sense, IV is a magic. This chapter presents a not-so-magic perspective based on the encouragement design. This again echos Dorn (1953)’s suggestion that the planner of an observational study should always ask himself the following question:

How would the study be conducted if it were possible to do it by controlled experimentation?

The experimental analog of the IV method is the encouragement design (Zelen, 1979; Powers and Swinton, 1984; Holland, 1986).

## 21.1 Encouragement Design and Noncompliance

Consider an experiment with units indexed by $i = 1 , \ldots , n$ . Let $Z _ { i }$ be the treatment assigned, with 1 for the treatment and 0 for the control. Let $D _ { i }$ be the treatment received, with 1 for the treatment and 0 for the control. When $Z _ { i } \neq D _ { i }$ for some unit i, the noncompliance problem arises. Noncompliance is a very common problem especially in encouragement designs involving human beings as experimental units. In these cases, the experimenters cannot force the units to take the treatment but rather only encourage them to do so. Let $Y _ { i }$ be the outcome of interest.

Consider complete randomization of Z and ignore covariates X now. We have the potential values for the treatment received $\{ D _ { i } ( 1 ) , D _ { i } ( 0 ) \}$ and the potential values for the outcome $\{ Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \}$ , all with respect to the treatment assignment levels 1 and 0. Their observed values are $D _ { i } \ =$ $Z _ { i } D _ { i } ( 1 ) + ( 1 - Z _ { i } ) D _ { i } ( 0 )$ and $Y _ { i } = Z _ { i } Y _ { i } ( 1 ) + ( 1 - Z _ { i } ) Y _ { i } ( 0 )$ , respectively. For notational simplicity, we assume $\{ Z _ { i } , D _ { i } ( 1 ) , D _ { i } ( 0 ) , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \} _ { i = 1 } ^ { n } \stackrel { \mathrm { I I D } } { \sim }$ IID $\{ Z , D ( 1 ) , D ( 0 ) , Y ( 1 ) , Y ( 0 ) \}$ and sometimes drop the subscript i without causing confusions.

We start with completely randomized experiments.

Assumption 21.1 (randomization) $Z \bot \bot \{ D ( 1 ) , D ( 0 ) , Y ( 1 ) , Y ( 0 ) \}$ .

Randomization allows for identification of the average causal effects on D and $Y \colon$ :

$$
\tau_ {D} = E \{D (1) - D (0) \} = E (D \mid Z = 1) - E (D \mid Z = 0)
$$

and

$$
\tau_ {Y} = E \{Y (1) - Y (0) \} = E (Y \mid Z = 1) - E (Y \mid Z = 0).
$$

We can use simple difference-in-means estimators $\hat { \tau } _ { D }$ and $\hat { \tau } _ { Y }$ to estimate $\tau _ { D }$ and $\tau _ { Y }$ , respectively.

Reporting the estimate $\hat { \tau } _ { Y }$ with the associated standard error is called the intention-to-treat (ITT) analysis. It estimates the effect of the treatment assignment on the outcome, and complete randomization in Assumption 21.1 justifies this analysis. However, it may not answer the scientific question, that is, the causal effect of the treatment received on the outcome.

## 21.2 Latent Compliance Status and Effects

## 21.2.1 Nonparametric identification

Following Imbens and Angrist (1994) and Angrist et al. (1996), we stratify the population based on the joint potential values of of $\{ D _ { i } ( 1 ) , D _ { i } ( 0 ) \}$ . Because $D$ is binary, we have four possible combinations:

$$
U _ {i} = \left\{ \begin{array}{l l} \mathrm{a,} & \mathrm{if} D _ {i} (1) = 1 \mathrm{and} D _ {i} (0) = 1; \\ \mathrm{c,} & \mathrm{if} D _ {i} (1) = 1 \mathrm{and} D _ {i} (0) = 0; \\ \mathrm{d,} & \mathrm{if} D _ {i} (1) = 0 \mathrm{and} D _ {i} (0) = 1; \\ \mathrm{n,} & \mathrm{if} D _ {i} (1) = 0 \mathrm{and} D _ {i} (0) = 0, \end{array} \right.
$$

where $\mathrm { ^ { 6 } a } ^ { \mathrm { 9 } }$ is for “always taker, $\begin{array} { r l } { \mathfrak { N } } & { { } ^ { 6 6 } \mathrm { c } ^ { \mathfrak { N } } } \end{array}$ is for “complier,” $\mathrm { ^ { 6 } d } ^ { \mathrm { 3 } }$ is for “defier,” and $\mathrm { ^ { 6 6 } n } \mathrm { ^ { \circ } }$ is for “never taker.” Because we cannot observe $D _ { i } ( 1 )$ and $D _ { i } ( 0 )$ simultaneously, $U _ { i }$ is a latent variable for the compliance behavior of unit i.

Based on $U ,$ , we can use the law of total probability to decompose the average causal effect on $Y$ into four terms:

$$
\begin{array}{l} \tau_ {Y} = E \{Y (1) - Y (0) \mid U = \mathrm{a} \} \operatorname{pr} (U = \mathrm{a}) \\ + E \{Y (1) - Y (0) \mid U = c \} \mathrm{pr} (U = c) \\ + E \{Y (1) - Y (0) \mid U = \mathrm{d} \} \operatorname{pr} (U = \mathrm{d}) \\ + E \{Y (1) - Y (0) \mid U = \mathrm{n} \} \operatorname{pr} (U = \mathrm{n}). \tag {21.1} \\ \end{array}
$$

Therefore, $\tau _ { Y }$ is a weighted average of four latent subgroup effects. We will look into more details of the latent groups below.

Assumption 21.2 below restricts the third term in (21.1) to be zero.

Assumption 21.2 (monotonicity) $\mathrm { p r } ( U = \mathrm { d } ) = 0 ~ o r ~ D _ { i } ( 1 ) \geq D _ { i } ( 0 )$ , that $i s ,$ there are no $d e f i e r s$ .

Assumption 21.2 holds automatically with one-sided noncompliance when the units assigned to the control arm have no access to the treatment, $\mathrm { i . e . , }$ $D _ { i } ( 0 ) = 0$ for all units. Under randomization, Assumption 21.2 has a testable implication that

$$
\operatorname{pr} (D = 1 \mid Z = 1) \geq \operatorname{pr} (D = 1 \mid Z = 0).
$$

But Assumption 21.2 is much stronger than the inequality above. The former restricts $D _ { i } ( 1 )$ and $D _ { i } ( 0 )$ at the individual level and the latter restricts them only on average. Nevertheless, when this testable implication holds, we cannot use the observed data to refute Assumption 21.2.

Assumption 21.3 below restricts the first and last terms in (21.1) to be zero based on the mechanism of the treatment assignment on the outcome through only the treatment received.

Assumption 21.3 (exclusion restriction) $Y _ { i } ( 1 ) = Y _ { i } ( 0 )$ for always takers with $U _ { i } = \mathbf { a }$ and never takers with $U _ { i } = \mathrm { n }$ .

Assumption 21.3 requires that the treatment assignment affects the outcome only if it affects the treatment received. In double-blind clinical $\mathrm { \ t r i a l ^ { 1 } }$ , it is biologically plausible because the outcome only depends on the actual treatment received. That ${ \mathrm { i s } } ,$ if the treatment assignment does not change the treatment received, it does not change the outcome either. It can be violated if the treatment assignment has direct effects on the outcome not through the treatment received. For example, some randomized controlled trials are not double blinded, and the treatment assignment can have some unknown pathways to the outcome.

Under Assumptions 21.2 and 21.3, the decomposition (21.1) only has the second term :

$$
\tau_ {Y} = E \{Y (1) - Y (0) \mid U = \mathrm{c} \} \mathrm{pr} (U = \mathrm{c}). \tag {21.2}
$$

Similarly, we can decompose the average causal effect on $D$ into four terms:

$$
\begin{array}{l} \tau_ {D} = E \{D (1) - D (0) \mid U = \mathrm{a} \} \operatorname{pr} (U = \mathrm{a}) \\ + E \{D (1) - D (0) \mid U = c \} \operatorname{pr} (U = c) \\ + E \{D (1) - D (0) \mid U = \mathrm{d} \} \operatorname * {p r} (U = \mathrm{d}) \\ + E \{D (1) - D (0) \mid U = \mathrm{n} \} \mathrm{pr} (U = \mathrm{n}) \\ = 0 \times \operatorname{pr} (U = \mathrm{a}) + 1 \times \operatorname{pr} (U = \mathrm{c}) + (- 1) \times \operatorname{pr} (U = \mathrm{d}) + 0 \times \operatorname{pr} (U = \mathrm{n}), \\ \end{array}
$$

which, under Assumption 21.2, reduces to

$$
\tau_ {D} = \mathrm{pr} (U = \mathrm{c}). \tag {21.3}
$$

This is an interesting fact that the proportion of the compliers $\pi _ { \mathrm { c } }$ equals the average causal effect of the treatment assigned on $D ,$ an identifiable quantity under complete randomization. Although we do not know all the compliers based on the observed data, we can identify their proportion in the whole population based on (21.3). Combining (21.2) and (21.3), we have the following result.

Theorem 21.1 Under Assumptions 21.2–21.3, we have

$$
E \{Y (1) - Y (0) \mid U = \mathrm{c} \} = \frac {\tau_ {Y}}{\tau_ {D}}
$$

$$
i f \tau_ {D} \neq 0.
$$

Following Imbens and Angrist (1994) and Angrist et al. (1996), we define a new causal effect below.

Definition 21.1 (CACE or LATE) Define

$$
\tau_ {\mathrm{c}} \equiv E \{Y (1) - Y (0) \mid U = \mathrm{c} \}
$$

as the “complier average causal effect (CACE)” or the “local average treatment effect (LATE)”. It has alternative forms:

$$
\tau_ {\mathrm{c}} = E \{Y (1) - Y (0) \mid D (1) = 1, D (0) = 0 \}
$$

$$
= E \{Y (1) - Y (0) \mid D (1) > D (0) \}.
$$

Based on Definition 21.1, we can rewrite Theorem 21.1 as

$$
\tau_ {\mathrm{c}} = \frac {\tau_ {Y}}{\tau_ {D}},
$$

that is, the CACE or LATE equals the ratio of the average causal effects on Y over that on D. Under Assumption 21.1, we further identify the CACE below.

Corollary 21.1 Under Assumptions 21.1–21.3, we have

$$
\tau_ {\mathrm{c}} = \frac {E (Y \mid Z = 1) - E (Y \mid Z = 0)}{E (D \mid Z = 1) - E (D \mid Z = 0)}.
$$

Therefore, under randomization, monotonicity, and exclusion restriction, we can nonparametrically identify the CACE as the ratio of the difference in means of the outcome over the difference in means of the treatment received.

## 21.2.2 Estimation

Based on Corollary 21.1, we can estimate $\tau _ { \mathrm { c } }$ by a simple ratio

$$
\hat {\tau} _ {\mathrm{c}} = \frac {\hat {\tau} _ {Y}}{\hat {\tau} _ {D}},
$$

which is called the Wald estimator (Wald, 1940) or the IV estimator. In the above discussion, $Z$ acts as the IV for D.

We can obtain the variance estimator based on the following heuristics (see Example A1.3):

$$
\hat {\tau} _ {\mathrm{c}} - \tau_ {\mathrm{c}} = (\hat {\tau} _ {Y} - \tau_ {\mathrm{c}} \hat {\tau} _ {D}) / \hat {\tau} _ {D} \approx (\hat {\tau} _ {Y} - \tau_ {\mathrm{c}} \hat {\tau} _ {D}) / \tau_ {D} = \hat {\tau} _ {A} / \tau_ {D},
$$

where $\hat { \tau } _ { A }$ is the difference-in-means of the adjusted outcome $A _ { i } = Y _ { i } - \tau _ { \mathrm { c } } D _ { i }$ . So the asymptotic variance of $\hat { \tau } _ { \mathrm { c } }$ is close to the variance of $\hat { \tau } _ { A }$ divided by $\tau _ { D } ^ { 2 }$ . The variance estimation proceeds in the following steps:

1. obtain the adjusted outcomes $\hat { A } _ { i } = Y _ { i } - \hat { \tau } _ { \mathrm { c } } D _ { i } ( i = 1 , \dots , n )$  
2. obtain the Neyman-type variance estimate based on the adjusted outcomes:

$$
\hat {V} _ {\hat {A}} = \frac {\hat {S} _ {\hat {A}} ^ {2} (1)}{n _ {1}} + \frac {\hat {S} _ {\hat {A}} ^ {2} (0)}{n _ {0}},
$$

where $\hat { S } _ { \hat { A } } ^ { 2 } ( 1 )$ and $\hat { S } _ { \hat { A } } ^ { 2 } ( 0 )$ are the sample variances of the $\hat { A } _ { i } { ^ { \circ } \mathrm { s } }$ under treatment and control, respectively;

3. obtain the final variance estimator $\hat { V } _ { \hat { A } } / { \hat { \tau } _ { D } } ^ { 2 }$ .

Under the null hypothesis that $\tau _ { \mathrm { c } } = 0$ , we can simply approximate the variance by $\hat { V } _ { Y } / \hat { \tau } _ { D } ^ { 2 }$ , where $\hat { V } _ { Y }$ is the Neyman-type variance estimate for the difference in means of $Y$ . This variance estimator is inconsistent if the true $\tau _ { \mathrm { c } }$ is not zero. Therefore, it works for testing but not for estimation. Nevertheless, it gives interesting insights for the ITT estimator and the Wald estimator. The ITT estimator $\hat { \tau } _ { Y }$ has estimated standard error $\sqrt { \hat { V } _ { Y } }$ . The Wald estimator $\hat { \tau } _ { Y } / \hat { \tau } _ { D }$ essentially equals the ITT estimator multiplied by $1 / \hat { \tau } _ { D } > 1$ , which is larger in magnitude but at the same time its estimated standard error increases by the same factor. The confidence intervals for $\tau _ { Y }$ and $\tau _ { \mathrm { c } }$ are

$$
\hat {\tau} _ {Y} \pm z _ {1 - \alpha / 2} \sqrt {\hat {V} _ {Y}}
$$

and

$$
\hat {\tau} _ {Y} / \hat {\tau} _ {D} \pm z _ {1 - \alpha / 2} \sqrt {\hat {V} _ {Y}} / \hat {\tau} _ {D} = \left(\hat {\tau} _ {Y} \pm z _ {1 - \alpha / 2} \sqrt {\hat {V} _ {Y}}\right) / \hat {\tau} _ {D}.
$$

These confidence intervals give the same qualitative conclusions since they will both cover zero or not. In some sense, the IV analysis provides the same qualitative information as the ITT analysis of $Y$ although it involves more complicated procedures.

## 21.3 Covariates

## 21.3.1 Covariate adjustment in complete randomization

We now consider completely randomized experiments with covariates, and assume $Z \bot \bot \{ D ( 1 ) , D ( 0 ) , Y ( 1 ) , Y ( 0 ) , X \}$ . With covariates $X ,$ , we can obtain Lin (2013)’s estimators $\hat { \tau } _ { D , \mathrm { L } }$ and $\hat { \tau } _ { Y , \mathrm { L } }$ for both D and ${ \cal Y } ,$ , resulting in $\hat { \tau } _ { \mathrm { c , L } } =$ $\hat { \tau } _ { Y , \mathrm { L } } / \hat { \tau } _ { D , \mathrm { L } }$ . Recall that

$$
\hat {\tau} _ {D, \mathrm{L}} = \left\{\hat {\bar {D}} (1) - \hat {\beta} _ {D 1} ^ {\mathsf {T}} \hat {\bar {X}} (1) \right\} - \left\{\hat {\bar {D}} (0) - \hat {\beta} _ {D 0} ^ {\mathsf {T}} \hat {\bar {X}} (0) \right\},
$$

$$
\hat {\tau} _ {Y, \mathrm{L}} = \left\{\hat {\bar {Y}} (1) - \hat {\beta} _ {Y 1} ^ {\mathsf {T}} \hat {\bar {X}} (1) \right\} - \left\{\hat {\bar {Y}} (0) - \hat {\beta} _ {Y 0} ^ {\mathsf {T}} \hat {\bar {X}} (0) \right\},
$$

where $\hat { \beta } _ { D 1 }$ and $\hat { \beta } _ { Y 1 }$ are the coefficients of X in the OLS fits of D and $Y$ in the treated group, and $\hat { \beta } _ { D 0 }$ and $\hat { \beta } _ { Y 0 }$ are the coefficients of X in the OLS fits of $D$ and Y in the control group. We can approximate the standard error of $\hat { \tau } _ { \mathrm { c , L } }$ based on the following heuristics (again see Example A1.3):

$$
\hat {\tau} _ {\mathrm{c,L}} - \tau_ {\mathrm{c}} = (\hat {\tau} _ {Y, \mathrm{L}} - \tau_ {\mathrm{c}} \hat {\tau} _ {D, \mathrm{L}}) / \hat {\tau} _ {D, \mathrm{L}} \approx (\hat {\tau} _ {Y, \mathrm{L}} - \tau_ {\mathrm{c}} \hat {\tau} _ {D, \mathrm{L}}) / \tau_ {D} = \hat {\tau} _ {A} / \tau_ {D},
$$

where $\hat { \tau } _ { A }$ is the difference-in-means of A, defined as

$$
A _ {i} = \left\{ \begin{array}{l l} (Y _ {i} - \hat {\beta} _ {Y 1} ^ {\mathsf {T}} X _ {i}) - \tau_ {\mathrm{c}} (D _ {i} - \hat {\beta} _ {D 1} ^ {\mathsf {T}} X _ {i}), & \text {if} Z _ {i} = 1, \\ (Y _ {i} - \hat {\beta} _ {Y 0} ^ {\mathsf {T}} X _ {i}) - \tau_ {\mathrm{c}} (D _ {i} - \hat {\beta} _ {D 0} ^ {\mathsf {T}} X _ {i}), & \text {if} Z _ {i} = 0. \end{array} \right.
$$

The variance estimation proceeds in the following steps:

1. obtain the adjusted outcomes $\hat { A } _ { i } \ ( i = 1 , \ldots , n )$ with

$$
\hat {A} _ {i} = \left\{ \begin{array}{l l} (Y _ {i} - \hat {\beta} _ {Y 1} ^ {\mathsf {T}} X _ {i}) - \hat {\tau} _ {\mathrm{c,L}} (D _ {i} - \hat {\beta} _ {D 1} ^ {\mathsf {T}} X _ {i}), & \text {if} Z _ {i} = 1, \\ (Y _ {i} - \hat {\beta} _ {Y 0} ^ {\mathsf {T}} X _ {i}) - \hat {\tau} _ {\mathrm{c,L}} (D _ {i} - \hat {\beta} _ {D 0} ^ {\mathsf {T}} X _ {i}), & \text {if} Z _ {i} = 0; \end{array} \right.
$$

2. obtain the Neyman-type variance estimate based on the adjusted outcomes:

$$
\hat {V} _ {\hat {A}} = \frac {\hat {S} _ {\hat {A}} ^ {2} (1)}{n _ {1}} + \frac {\hat {S} _ {\hat {A}} ^ {2} (0)}{n _ {0}},
$$

where $\hat { S } _ { \hat { A } } ^ { 2 } ( 1 )$ and $\hat { S } _ { \hat { A } } ^ { 2 } ( 0 )$ are the sample variances of the $\hat { A } _ { i } { ^ { \circ } \mathrm { s } }$ under the treatment and control, respectively;

3. obtain the final variance estimator $\hat { V } _ { \hat { A } } / { \hat { \tau } _ { D , \mathrm { L } } ^ { 2 } }$

Again under the null with $\tau _ { \mathrm { c } } ~ = ~ 0 .$ , we can approximate the estimated standard error for $\hat { \tau } _ { \mathrm { c , L } }$ by the estimated standard error of $\hat { \tau } _ { Y , \mathrm { L } } \ ( \mathrm { e . g . }$ , the EHW standard error in the fully interacted linear model) divided by $\hat { \tau } _ { D , \mathrm { L } }$ .

## 21.3.2 Covariates in conditional randomization or unconfounded observational studies

If randomization holds conditionally, i.e.,

$$
Z \bot \{D (1), D (0), Y (1), Y (0) \} \mid X,
$$

then we must adjust for covariates to avoid bias. The analysis is also straightforward since we already have discussed many estimators in Part III for estimating the effects of Z on D and $Y _ { z }$ , respectively. We can just use them in the ratio formula $\hat { \tau } _ { \mathrm { c } } = \hat { \tau } _ { Y } / \hat { \tau } _ { D }$ and use the bootstrap to approximate the asymptotic variance.

## 21.4 Weak IV

Even $\tau _ { D } > 0$ , there is a positive probability that $\hat { \tau } _ { D }$ is zero, so the variance of $\hat { \tau } _ { \mathrm { c } }$ is infinity. The variance from the Normal approximation discussed before is not the variance of $\hat { \tau } _ { \mathrm { c } }$ but rather the variance of its asymptotic distribution. This is a subtle technical point. When $\tau _ { D }$ is close to 0, which is referred to as the weak IV case, the ratio estimator $\hat { \tau } _ { \mathrm { c } } = \hat { \tau } _ { Y } / \hat { \tau } _ { D }$ has poor finite-sample properties. Under this scenario, $\hat { \tau } _ { \mathrm { c } }$ has finite sample bias and non-Normal asymptotic distribution, and the corresponding Wald-type confidence intervals have poor coverage properties2. In the simple case with a binary outcome $Y ,$ , we know that τY must be bounded between −1 and 1, but there is no guarantee that $\hat { \tau } _ { \mathrm { c } }$ is bounded between −1 and 1. How do we deal with a weak IV?

From a testing perspective, there is an easy solution. Because $\tau _ { \mathrm { c } } = \tau _ { Y } / \tau _ { D }$ , so the following two null hypotheses are equivalent:

$$
H _ {0}: \tau_ {\mathrm{c}} = 0 \Longleftrightarrow H _ {0} ^ {\prime}: \tau_ {Y} = 0.
$$

Therefore, we simply test $H _ { 0 } ^ { \prime } ,$ , i.e., the average causal effect of Z on Y is zero. This echos our discussion in Section 21.2.2.

From an estimation perspective, we can focus on the confidence interval although the point estimator has poor finite-sample properties. Because $\tau _ { \mathrm { c } } = \tau _ { Y } / \tau _ { D }$ , this is similar to the classical Fieller–Creasy problem in statistics. Below we discuss a strategy for constructing confidence interval for $\tau _ { \mathrm { c } }$ motivated by Fieller (1954); see Section A1.4.2. Given the true value $\tau _ { \mathrm { c } }$ , we have

$$
\tau_ {Y} - \tau_ {\mathrm{c}} \tau_ {D} = 0.
$$

So we can construct a confidence set for $\tau _ { \mathrm { c } }$ by inverting a sequence of null hypotheses

$$
H _ {0} (b): \tau_ {\mathrm{c}} = b
$$

This null hypothesis is equivalent to the null hypothesis of zero average causal effect on the outcome $A _ { i } ( b ) = Y _ { i } - b D _ { i }$ :

$$
H _ {0} (b): \tau_ {A (b)} = 0.
$$

Let ${ \hat { \tau } } _ { A } ( b )$ be a generic estimator for $\tau _ { A \left( b \right) }$ with the associated variance estimator $\hat { V } _ { A } ( b )$ . In the CRE without covariates, ${ \hat { \tau } } _ { A } ( b )$ is the difference in means of the outcome $A _ { i } ( b )$ and $\hat { V } _ { A } ( b )$ is the Neyman-type variance estimator. In the CRE with covariates, ${ \hat { \tau } } _ { A } ( b )$ is Lin (2013)’s estimator for the outcome $A _ { i } ( b )$ and $\hat { V } _ { A } ( b )$ is the EHW variance estimator in the associated OLS fit of $Y _ { i } - b D _ { i }$ on $( Z _ { i } , X _ { i } , Z _ { i } X _ { i } )$ . In unconfounded observational studies, we can obtain the estimator for the average causal effect on $A _ { i } ( b )$ and the associated variance estimator based on many existing strategies in Part III.

Based on ${ \hat { \tau } } _ { A } ( b )$ and $\tau _ { A \left( b \right) }$ , we can construct a Wald-type test for $H _ { 0 } ( b )$ . Inverting tests, we can construct the following confidence set for $\tau _ { \mathrm { c } } :$ :

$$
\left\{b: \frac {\hat {\tau} _ {A} ^ {2} (b)}{\hat {V} _ {A} (b)} \leq z _ {\alpha} ^ {2} \right\}.
$$

This is close to the Anderson–Rubin-type confidence interval in econometrics (Anderson and Rubin, 1950). Due to its connection to Fieller (1954), I will call it the Fieller–Anderson–Rubin confidence interval. These weak-IV confidence intervals reduce to the asymptotic confidence intervals when the IV is strong. But they have additional guarantees when the IV is weak. I recommend using them in practice.

Example 21.1 To gain intuition about the Fieller–Anderson–Rubin confidence interval, we look into the simple case of the CRE without covariates. The quadratic inequality in the confidence interval reduces to

$$
\begin{array}{l} (\hat {\tau} _ {Y} - b \hat {\tau} _ {D}) ^ {2} \\ \leq z _ {\alpha} ^ {2} \left[ n _ {1} ^ {- 1} \{\hat {S} _ {Y} ^ {2} (1) + b ^ {2} \hat {S} _ {D} ^ {2} (1) - 2 b \hat {S} _ {Y D} (1) \} \right. \\ \left. \right.\left. + n _ {0} ^ {- 1} \{\hat {S} _ {Y} ^ {2} (0) + b ^ {2} \hat {S} _ {D} ^ {2} (0) - 2 b \hat {S} _ {Y D} (0) \} \right], \\ \end{array}
$$

where $\{ \hat { S } _ { Y } ^ { 2 } ( 1 ) , \hat { S } _ { D } ^ { 2 } ( 1 ) , \hat { S } _ { Y D } ( 1 ) \}$ and $\{ \hat { S } _ { Y } ^ { 2 } ( 0 ) , \hat { S } _ { D } ^ { 2 } ( 0 ) , \hat { S } _ { Y D } ( 0 ) \}$ are the sample variances and covariances of Y and D under treatment and control, respectively. The confidence set can be a close interval, two disconnected intervals, an empty set, or the whole real line. I relegate the detailed discussion to Problem 21.3.

## 21.5 Application

The mediation package contains a dataset jobs from Job Search Intervention Study (JOBS II), which was a randomized field experiment that investigates the efficacy of a job training intervention on unemployed workers. The variable treat is the indicator for whether a participant was randomly selected for the JOBS II training program, and the variable comply is the indicator for whether a participant actually participated in the JOBS II program. An outcome of interest is jobseek for measuring the level of job-search self-efficacy with values from 1 to 5. A few standard covariates are sex, age, marital, nonwhite, educ, and income.

Without using covariates, the confidence intervals based on the delta method and the bootstrap are

```txt
> est
[1] 0.1087904
> c(est - 1.96*dse, est + 1.96*dse)
[1] -0.05002163 0.26760235
> c(est - 1.96*bse, est + 1.96*bse)
[1] -0.04657384 0.26415455
```

Adjusting for covariates, the confidence intervals based on the delta method and the bootstrap are

```csv
> est
[1] 0.1176332
> c(est - 1.96*dse, est + 1.96*dse)
[1] -0.03638421 0.27165070
> c(est - 1.96*bse, est + 1.96*bse)
[1] -0.03926737 0.27453386
```

We can also construct confidence interval by inverting tests. Without using covariates, it is

```txt
> ARCI
[1] -0.050 0.267
```

adjusting for covariates, it is

```txt
> ARCI
[1] -0.046 0.281
```

Figure 21.1 plots the p-values for a sequence of tests.

## 21.6 Interpreting the Complier Average Causal Effect

The notation for potential outcomes $\{ D ( 1 ) , D ( 0 ) , Y ( 1 ) , Y ( 0 ) \}$ is with respect to the hypothetical intervention of the treatment assigned Z. So $\tau _ { \mathrm { c } }$ is the average causal effect of the treatment assigned on the outcome for compliers. Fortunately, $D = Z$ for compliers, so we can also interpret $\tau _ { \mathrm { c } }$ as the average causal effect of the treatment received on the outcome for compliers. This partially answers the scientific question.

Some papers (e.g., Angrist et al., 1996) use different notation. They use $Y _ { i } ( z , d )$ for the potential outcome of unit i under a $2 \times 2$ factorial experiment with the treatment assigned z and treatment received d. The exclusion restriction assumption has the following form.

Assumption 21.4 (exclusion restriction) $Y _ { i } ( z , d ) = Y _ { i } ( d )$ for all i, that $i s ,$ the potential outcome is only a function of d.

Based on the causal graph below, Assumption 21.4 rules out the direct arrow from $Z$ to $Y$ . In such case, Z is an IV for D.

![image_21](images/image_21.png)

```mermaid
graph TD
  Z --> D
  D --> Y
  U --> Y
```

Under Assumption 21.4, the augmented notation $Y _ { i } ( z , d )$ reduces to $Y _ { i } ( d )$ , which justifies the name of “exclusion restriction.” Therefore, $Y _ { i } ( 1 , d ) \ =$ $Y _ { i } ( 0 , d )$ for $d = 0 , 1$ , which, coupled with Assumption 21.2, implies that

$$
\begin{array}{l} Y _ {i} (z = 1) - Y _ {i} (z = 0) = Y _ {i} (1, D _ {i} (1)) - Y _ {i} (0, D _ {i} (0)) \\ = \left\{ \begin{array}{l l} 0, & \text {if} U _ {i} = \mathrm{a}, \\ 0, & \text {if} U _ {i} = \mathrm{n}, \\ Y _ {i} (d = 1) - Y _ {i} (d = 0), & \text {if} U _ {i} = \mathrm{c}. \end{array} \right. \\ \end{array}
$$

In the above, we emphasize the potential outcomes are with respect to $z ,$ d or both, to avoid confusions. The previous decomposition of $\tau _ { Y }$ holds and we have the following result from Imbens and Angrist (1994) and Angrist et al. (1996).

Recall the average causal effect on $D , \tau _ { D } = E \{ D ( 1 ) - D ( 0 ) \}$ , define the average causal effect on $Y$ as $\tau _ { Y } = E \{ Y ( D ( 1 ) ) - Y ( D ( 0 ) ) \}$ , and define the complier average causal effect as

$$
\tau_ {\mathrm{c}} = E \{Y (d = 1) - Y (d = 0) \mid U = \mathrm{c} \}.
$$

Theorem 21.2 Under Assumptions 21.2–21.4, we have

$$
Y (D (1)) - Y (D (0)) = \{D (1) - D (0) \} \times \{Y (d = 1) - Y (d = 0) \}
$$

and $\tau _ { \mathrm { c } } = \tau _ { Y } / \tau _ { D }$

The proof is almost identical to the proof of Theorem 21.1 with modifications of the notation. I leave it as Problem 21.2. From the notation $Y _ { i } ( d )$ , it is more convenient to interpret $\tau _ { \mathrm { c } }$ as as the average causal effect of the treatment received on the outcome for compliers.

## 21.7 Homework problems

21.1 Variance of the Wald estimator

Show that var $\left( \hat { \tau } _ { \mathrm { c } } \right) = \infty ,$ .

21.2 Proof of the main theorem of Imbens and Angrist (1994) and Angrist et al. (1996)

Prove Theorem 21.2.

21.3 More on the Fieller–Anderson–Rubin confidence set

The confidence set in Example 21.1 can be a close interval, two disconnected intervals, an empty set, or the whole real line. Find the precise conditions for each case.

21.4 Binary IV and ordinal treatment received

Angrist and Imbens (1995) discussed a more general setting with a binary IV Z, an ordinal treatment received $D \in \{ 0 , 1 , \ldots , J \}$ , and an outcome $Y .$ . The ordinal treatment received has potential outcomes $D ( 1 )$ and $D ( 0 )$ with respect to the binary IV, and the outcome has potential outcomes $Y ( z , d )$ with respect to both the binary IV and the ordinal treatment received. Extend the discussion in Section 21.6 and the corresponding IV assumptions as below.

Assumption 21.5 We have $( 1 )$ randomization that Z $\{ D ( z ) , Y ( z , d ) : z =$ $\boldsymbol { 0 } , 1 ; d = 0 , 1 , \dots , J \} ; ( \mathcal { Q } )$ monotonicity that $D ( 1 ) \geq D ( 0 ) ;$ ; and $( 3 )$ exclusion restriction that $Y ( z , d ) = Y ( d )$ for $a l l z = 0 , 1$ and $d = 0 , 1 , \dotsc , J$ .

They proved Theorem 21.3 below.

Theorem 21.3 Under Assumption 21.5, we have

$$
\frac {E (Y \mid Z = 1) - E (Y \mid Z = 0)}{E (D \mid Z = 1) - E (D \mid Z = 0)} = \sum_ {j = 1} ^ {J} w _ {j} E \{Y (j) - Y (j - 1) \mid D (1) \geq j > D (0) \}
$$

where

$$
w _ {j} = \frac {\operatorname* {p r} \{D (1) \geq j > D (0) \}}{\sum_ {j ^ {\prime} = 1} ^ {J} \operatorname* {p r} \{D (1) \geq j ^ {\prime} > D (0) \}}.
$$

Prove Theorem 21.3.

Remark: When $J = 1$ , Theorem 21.3 reduces to Theorem 21.2. It states that the standard IV formula identifies a weighted average of some latent subgroup effects. The weights are proportional to the probability of the latent groups defined by $D ( 1 ) \geq j > D ( 0 )$ , and the latent subgroup effects $E \{ Y ( j ) -$ $Y ( j - 1 ) \mid D ( 1 ) \geq j > D ( 0 ) \}$ compare the adjacent levels of the treatment received. However, this weighted average may not be easy to interpret because the latent groups overlap.

The proof can be tedious. A trick is to write the treatment received and outcome under treatment assignment z as

$$
D (z) = \sum_ {j = 1} ^ {J} j 1 \{D (z) = j \}, \quad Y (D (z)) = \sum_ {j = 1} ^ {J} Y (j) 1 \{D (z) = j \}
$$

to obtain

$$
D (1) - D (0) = \sum_ {j = 0} ^ {J} j [ 1 \{D (1) = j \} - 1 \{D (0) = j \} ]
$$

and

$$
Y (D (1)) - Y (D (0)) = \sum_ {j = 0} ^ {J} Y (j) [ 1 \{D (1) = j \} - 1 \{D (0) = j \} ].
$$

Then use the following Abel’s lemma, also called summation by parts:

$$
\sum_ {j = 0} ^ {J} f _ {j} \left(g _ {j + 1} - g _ {j}\right) = f _ {J} g _ {J + 1} - f _ {0} g _ {0} - \sum_ {j = 1} ^ {J} g _ {j} \left(f _ {j} - f _ {j - 1}\right)
$$

for appropriately specified sequences $( f _ { j } )$ and $( g _ { j } )$ .

## 21.5 Data analysis: a flu shot encouragement design (McDonald et $a l .$ , 1992)

The dataset in fludata.txt is from a randomized encouragement design of McDonald et al. (1992), which was also re-analyzed by Hirano et al. (2000).

It contains the following variables:

<table><tr><td>assign</td><td>binary encouragement to receive the flu shot</td></tr><tr><td>receive</td><td>binary indicator for receiving the flu shot</td></tr><tr><td>outcome</td><td>binary outcome for flu related hospitalization</td></tr><tr><td>age</td><td>age of the patient</td></tr><tr><td>sex</td><td>sex of the patient</td></tr><tr><td>race</td><td>race of the patient</td></tr></table>

copd, dm, heartd, renal, liverd various disease background covariates

Analyze the data with and without adjusting for the covariates.

## 21.6 Data analysis: the Karolinska data

Rubin (2008) used the Karolinska data as an example for the IV method. In karolinska.txt, whether a patient was diagnosed at large volume hospital can be viewed as an IV for whether a patient was treated at a large volume hospital. This is plausible at least conditioning on other observed covariates. See Rubin (2008)’s analysis for more details.

Reanalyze the data assuming that the IV is randomly assigned conditional on observed covariates.

## 21.7 Data analysis: a job training program (Schochet et al., 2008)

jobtraining.rtf contains the description of the data files X.csv and Y.csv.

X.csv contains the pretreatment covariates; you can view the sampling weight variable wgt as a covariate too. It is generally difficult to deal with sampling weights. Many previous analyses made this simplification. Conduct analyses with and without covariates.

Y.csv contains the sampling weight, treatment assigned, treatment received, and many post-treatment variables. Therefore, this data contains many outcomes depending on your questions of interest. The data also have many complications. First, some outcomes are missing. Second, unemployed individuals do not have wages or incomes. Third, the outcomes are repeatedly observed over time. When you do the data analysis, please give details about your choice of the questions of interest and estimators.

## 21.8 Recommended reading

Angrist et al. (1996) bridged the econometric IV perspective and statistical causal inference based on potential outcomes and demonstrated its usefulness with an application.

Some other early references on IV are Permutt and Hebel (1989), Sommer and Zeger (1991), Baker and Lindeman (1994), and Cuzick et al. (1997).

## 22

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

# An Econometric Perspective

Chapters 21 and 22 discuss the IV method from the experimental perspective. Figure 23.1 illustrates the intuition behind the discussion.

![image_22](images/image_22.png)

```mermaid
graph TD
  Z --> D
  D --> Y
  U --> Y
```

FIGURE 23.1: Causal diagram for IV

In an encouragement design with noncompliance, Z is randomized, so it is independent of the confounder U between the treatment received D and the outcome Y . Importantly, the treatment assignment Z does not have any direct effect on the outcome Y . It acts as an IV for the treatment received D in the sense that it affects the outcome Y only through the treatment received D. This IV is generated by the experimenter.

In many applications, randomization is infeasible. Then how can we draw causal inference in the presence of unmeasured confounding between the treatment and outcome? A clever idea from econometrics is to find natural experiments to mimic the setting of encouragement designs. To identify the causal effect of D on Y with unmeasured confounding, we can find another variable Z that satisfies the assumptions of the diagram in Figure 23.1. The variable Z must satisfy the following conditions:

1. it should be close to be randomized so that it is independent of the unmeasured confounding;  
2. it should change the distribution of D;  
3. it should not affect the outcome Y directly.

If all these conditions hold, then Z is a valid IV.

This chapter will provide the traditional econometrics perspective on IV. It is based on linear regression. Imbens and Angrist (1994) and Angrist et al. (1996) made a fundamental contribution by clarifying connection between this perspective and the experimental perspective in Chapters 21 and 22. I will start with examples and then give more algebraic details.

## 23.1 Examples of studies with IVs

Finding IV for causal inference is more an art than a science. The algebraic details in later sections are not the most complicated ones in statistics. However, it is fundamentally challenging to find IV in empirical research. Below are some famous examples.

Example 23.1 In an encouragement design, Z in the randomly assigned treatment, D is the final treatment received, and Y is the outcome. The IV assumptions encoded by Figure 23.1 is plausible in double-blind trials as discussed in Chapter 21. This is the ideal case for IV.

Example 23.2 Hearst et al. (1986) reported that men with low lottery number in the Vietnam Era draft lottery had higher mortality rates afterwards. They attributed this to the negative effect of military service. Angrist (1990) further reported that men with low lottery number in the Vietnam Era draft lottery had lower subsequent earnings. He attributed this to the negative effect of military service. These explanations are plausible because the lottery numbers were randomly generated, men with low lottery number were more likely to have military service, and the lottery numbers were unlikely to affect the subsequent mortality or earnings. That is, Figure 23.1 is plausible. Angrist et al. (1996) reanalyzed the data using the IV framework. Here, the lottery number is the IV, military service is the treatment, and mortality or earnings is the outcome.

Example 23.3 Angrist and Krueger (1991) studied the return of schooling in years on earnings, using the quarter of birth as an IV. This IV is plausible because of the pseuso randomization of the quarter of birth. It affected the years of schooling because (1) most states required the students to enter school in the calendar year in which they turned six, and (2) compulsory schooling laws typically required students remain in the school before their sixteenth birthday. More important, it is plausible that the quarter of birth did not affect earnings directly.

Example 23.4 Angrist and Evans (1998) studied the effect of family size on mother’s employment and work, using the sibling sex composition as an IV. This IV is plausible because of the pseudo randomization of the sibling sex composition. Moreover, parents in the US with two children of the same sex are more likely to have a third child than those parents with two children of different sex. It is also plausible that the sibling sex composition does not affect mother’s employment and work directly.

Example 23.5 Card (1993) studied the effect of schooling on wage, using the geographic variation in college proximity as an IV. In particular, Z contains dummy variables for whether a subject grew up near a two-year college or a four-year college. Although this study is classic, it might be a poor example for IV because parents’ choices of where to live might not be random, and moreover, where a subject grew up might matter for the subsequent wage.

Example 23.6 Voight et al. (2012) studied the causal effect of plasma highdensity lipoprotein (HDL) cholesterol on the risk of heart attack based on Mendelian randomization. They used some single-nucleotide polymorphisms (SNPs) as genetic IV for HDL, which are random with respect to the unmeasured confounders between HDL and heart attack by Mendel’s second law, and affect heart attack only though HDL. I will give more details of Mendelian randomization in Chapter 25.

## 23.2 Brief Review of the Ordinary Least Squares

Before discussing the econometric view of IV, I will first review the OLS in statistics (see Chapter A2). This is a standard topic in statistics. However, it has different mathematical formulations, and the choice of formulation matters for the interpretation.

The first view is based on projection. Given any pair of random variables $( D , Y )$ with finite second moments, define the population OLS coefficient as

$$
\beta = \arg \min _ {b} E (Y - D ^ {\mathsf {T}} b) ^ {2} = E (D D ^ {\mathsf {T}}) ^ {- 1} E (D Y),
$$

and then define the residual as $\varepsilon = Y - D ^ { \mathsf { T } } \beta$ . By definition, Y decomposes into

$$
Y = D ^ {\mathsf {T}} \beta + \varepsilon , \tag {23.1}
$$

which must satisfy

$$
E (D \varepsilon) = 0.
$$

Based on $( D _ { i } , Y _ { i } ) _ { i = 1 } ^ { n } \stackrel { \mathrm { I I D } } { \sim } ( D , Y )$ , the OLS estimator of $\beta$

$$
\hat {\beta} = \left(\sum_ {i = 1} ^ {n} D _ {i} D _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \sum_ {i = 1} ^ {n} D _ {i} Y _ {i}.
$$

Because

$$
\hat {\beta} = \left(\sum_ {i = 1} ^ {n} D _ {i} D _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \sum_ {i = 1} ^ {n} D _ {i} (D _ {i} ^ {\mathsf {T}} \beta + \varepsilon_ {i}) = \beta + \left(\sum_ {i = 1} ^ {n} D _ {i} D _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \sum_ {i = 1} ^ {n} D _ {i} \varepsilon_ {i},
$$

we can show that $\hat { \beta }$ is consistent for $\beta$ because of $E ( \varepsilon D ) = 0$ . The classicalEHW robust variance estimator for $\operatorname { c o v } ( { \hat { \boldsymbol { \beta } } } )$ is

$$
\hat {V} _ {\mathrm{EHW}} = \left(\sum_ {i = 1} ^ {n} D _ {i} D _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \left(\sum_ {i = 1} ^ {n} \hat {\varepsilon} _ {i} ^ {2} D _ {i} D _ {i} ^ {\mathsf {T}}\right) \left(\sum_ {i = 1} ^ {n} D _ {i} D _ {i} ^ {\mathsf {T}}\right) ^ {- 1}
$$

where $\hat { \varepsilon } _ { i } = Y _ { i } - D _ { i } ^ { \mathsf { T } } \hat { \beta }$ is the residual.

The second view is to treat

$$
Y = D ^ {\mathsf {T}} \beta + \varepsilon , \tag {23.2}
$$

as a true model for data generating process. That is, given the random variables $( D , \varepsilon )$ , we generate Y based on the linear equation (23.2). Importantly, in the data generating process, ε and D may be correlated in that $E ( D \varepsilon ) \neq 0$ . Figure 23.2 gives such an example. This is the fundamental difference compared to the first view where $E ( \varepsilon D ) = 0$ holds by the definition of the population OLS. Consequently, the OLS estimator can be inconsistent:

$$
\hat {\beta} \rightarrow \beta + E (D D ^ {\mathsf {T}}) ^ {- 1} E (D \varepsilon) \neq \beta
$$

in probability.

I end this section with definitions of endogenous and exogenous regressors based on (23.2), although their definitions are not unique in econometrics.

Definition 23.1 When $E ( \varepsilon D ) \ \ne \ 0 ,$ , the regressor D is called endogenous; when $E ( \varepsilon D ) = 0 ,$ , the regressor D is called exogenous.

The terminologies in Definition 23.1 are standard in econometrics. When $E ( \varepsilon D ) \neq 0$ , we also say that we have endogeneity; when $E ( \varepsilon D ) = 0$ , we also say that we have exogeneity.

In first view of OLS, the notions of endogeneity and exogeneity do not play any roles because $E ( \varepsilon D ) = 0$ by definition. Statisticians holding the first view usually find the notations of endogeneity and exogeneity strange, and consequently, find the idea of IV unnatural. To understand the econometric view of IV, we must switch to the second view of OLS.

## 23.3 Linear Instrumental Variable Model

When D is endogenous, the OLS estimator is inconsistent. We must use additional information to construct a consistent estimator for $\beta .$ I will focus on the following linear IV model:

Definition 23.2 (linear IV model) We have

$$
Y = D ^ {\mathsf {T}} \beta + \varepsilon ,
$$

<!-- footnote -->

- CD4 cells are white blood cells that fight infection.

<!-- footnote end -->

<!-- footnote -->

- This is called local linear regression in nonparametric statistics, which belongs to a broader class of local polynomial regression (Fan and Gijbels, 1996).

<!-- footnote end -->

<!-- footnote -->

- In general, it is better to blind the experiment to avoid various biases arising from placebo effects, patients’ expectation, etc. In double blind trials, both doctors and patients do not know the treatment; in single blind trials, the patients do not know the treatment but the doctors know. Sometimes, it is impossible to conduct double or even single blind trials. Those trials are called open trials.

<!-- footnote end -->

<!-- footnote -->

- The theory often assumes that τD has the order $n ^ { - 1 / 2 } ,$ Under this regime, the proportion of compliers goes to 0 as n goes to infinity. The IV method can only identify a subgroup average causal effect with the proportion shrinking to 0. This is a contrived regime for theoretical analysis. It is hard to justify this assumption in practice. The follow discussion does not assume it.

<!-- footnote end -->

![image_23](images/image_23.png)

```mermaid
graph TD
  U --> D
  U --> node["ε"]
  D --> Y
  node --> Y
```

![image_24](images/image_24.png)

(a) E(Dε) ̸= 0  
(b) marginalized over ε  
FIGURE 23.2: Different representations of the endogenous regressor

with

$$
E (\varepsilon Z) = 0. \tag {23.3}
$$

The linear IV model in Definition 23.2 can be illustrated by the following causal graph:

![image_25](images/image_25.png)

```mermaid
graph TD
  Z --> D
  D --> Y
  D --> node["ε"]
```

The above linear IV model allows that $E ( \varepsilon D ) \neq 0$ but requires an alternative moment condition (23.3). With $E ( \varepsilon ) = 0$ by incorporating the intercept, the new condition states that Z is uncorrelated with the error term ε. But any randomly generated noise is uncorrelated with ε, so an additional condition must hold to ensure that Z is useful for estimating β. Intuitively, the additional condition requires that Z is correlated to D, with more technical details stated below.

The mathematical requirement (23.3) seems simple. However, it is a key challenge in empirical research to find such a variable or variables Z that satisfies (23.3). Since the condition (23.3) involves the unobservable ε, it is generally untestable.

## 23.4 The Just-Identified Case

We first consider the case in which Z and D have the same dimension and $E ( Z D ^ { \mathsf { T } } )$ has full rank. The condition $E ( \varepsilon Z ) = 0$ implies that

$$
E \{Z (Y - D ^ {\mathsf {T}} \beta) \} = 0 \quad \Longrightarrow \quad E (Z Y) = E (Z D ^ {\mathsf {T}}) \beta
$$

$$
\implies \beta = E (Z D ^ {\mathsf {T}}) ^ {- 1} E (Z Y)
$$

if $E ( Z D ^ { \mathsf { T } } )$ is not degenerate. The OLS is a special case if $E ( \varepsilon D ) = 0 , { \mathrm { i . e . , } } D$ itself acts as an IV for itself. The resulting moment estimator is

$$
\hat {\beta} _ {\mathrm{IV}} = \left(\sum_ {i = 1} ^ {n} Z _ {i} D _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i}. \tag {23.4}
$$

In the simple case with an intercept and scalar D and Z, we have

$$
\left\{ \begin{array}{l} Y = \alpha + \beta D + \varepsilon , \\ E (\varepsilon) = 0, \quad \operatorname{cov} (\varepsilon , Z) = 0, \end{array} \right.
$$

which implies that

$$
\operatorname{cov} (Z, Y) = \beta \operatorname{cov} (Z, D) \Longrightarrow \beta = \frac {\operatorname{cov} (Z , Y)}{\operatorname{cov} (Z , D)}.
$$

Standardizing the numerator and denominator by $\mathrm { v a r } ( Z )$ , we have

$$
\beta = \frac {\operatorname{cov} (Z , Y) / \operatorname{var} (Z)}{\operatorname{cov} (Z , D) / \operatorname{var} (Z)},
$$

which equals the ratio between the coefficients of Z in the OLS fits of Y and D on Z. If Z is binary, these coefficients are differences in means and $\beta$ reduces to

$$
\beta = \frac {E (Y \mid Z = 1) - E (Y \mid Z = 0)}{E (D \mid Z = 1) - E (D \mid Z = 0)}.
$$

This is identical to the identification formula in Theorem 21.1. That is, with a binary IV Z and a binary treatment D, the IV estimator recovers the CACE under the potential outcomes framework. This is a key result in Imbens and Angrist (1994) and Angrist et al. (1996).

## 23.5 The Over-Identified Case

The discussion in Section 23.4 focuses on the just-identified case. When Z has lower dimension than X and $E ( Z D ^ { \mathsf { T } } )$ does not have full column rank, the equation $E ( Z Y ) = E ( Z D ^ { \mathsf { T } } ) \beta$ has infinitely many solutions. This is the underidentified case in which the coefficient $\beta$ cannot be uniquely determined even with $Z .$ It is a challenging case beyond the scope of this book. In practice, we need at least as many IVs as the endogenous regressors.

When $Z$ has higher dimension than $D$ and $E ( Z D ^ { \mathsf { T } } )$ has full column rank, we have many ways to determine $\beta$ from $E ( Z Y ) = E ( Z D ^ { \mathsf { T } } ) \beta$ . What is more, the sample analog

$$
n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i} = n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} D _ {i} ^ {\mathsf {T}} \beta
$$

may not have any solution because the number of equations is larger than the number of unknown parameters.

A computational trick in this case is the two-stage least squares (TSLS) estimator (Theil, 1953; Basmann, 1957). It is a clever computational trick, which has two steps.

Definition 23.3 (Two-stage least squares) Define the TSLS estimator of the coefficient of D with $Z$ being the $I V$ as follows.

1. Run OLS of D on $Z ,$ and obtain the fitted value $\hat { D } _ { i } ~ ( i { \bf \mu } =$ $1 , \ldots , n )$ . If $D _ { i }$ is a vector, then we need to run component-wise $O L S$ to obtain $\hat { D } _ { i }$ . Put the fitted vectors in a matrix $\hat { D }$ with rows $\hat { D } _ { i } ^ { \mathsf { T } }$ ;  
2. Run OLS of Y on $\hat { D } ,$ , and obtain the coefficient $\hat { \beta } _ { \mathrm { T S L S } }$

To see why TSLS works, we need more algebra. Write it more explicitly as

$$
\hat {\beta} _ {\mathrm{TSLS}} = \left(\sum_ {i = 1} ^ {n} \hat {D} _ {i} \hat {D} _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \sum_ {i = 1} ^ {n} \hat {D} _ {i} Y _ {i} \tag {23.5}
$$

$$
= \left(\sum_ {i = 1} ^ {n} \hat {D} _ {i} \hat {D} _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \sum_ {i = 1} ^ {n} \hat {D} _ {i} (D _ {i} ^ {\mathsf {T}} \beta + \varepsilon_ {i})
$$

$$
= \left(\sum_ {i = 1} ^ {n} \hat {D} _ {i} \hat {D} _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \sum_ {i = 1} ^ {n} \hat {D} _ {i} D _ {i} ^ {\mathsf {T}} \beta + \left(\sum_ {i = 1} ^ {n} \hat {D} _ {i} \hat {D} _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \sum_ {i = 1} ^ {n} \hat {D} _ {i} \varepsilon_ {i}.
$$

The first stage OLS fit ensures $D _ { i } = \hat { D } _ { i } + \check { D } _ { i }$ with

$$
\sum_ {i = 1} ^ {n} \hat {D} _ {i} \check {D} _ {i} ^ {\mathsf {T}} = 0 \tag {23.6}
$$

being a zero square matrix with the same dimension as $D _ { i }$ . The orthogonality (23.6) implies

$$
\sum_ {i = 1} ^ {n} \hat {D} _ {i} D _ {i} ^ {\mathsf {T}} = \sum_ {i = 1} ^ {n} \hat {D} _ {i} \hat {D} _ {i} ^ {\mathsf {T}},
$$

which further implies that

$$
\hat {\beta} _ {\mathrm{TSLS}} = \beta + \left(\sum_ {i = 1} ^ {n} \hat {D} _ {i} \hat {D} _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \sum_ {i = 1} ^ {n} \hat {D} _ {i} \varepsilon_ {i}. \tag {23.7}
$$

The first stage OLS fit also ensures

$$
\hat {D} _ {i} = \hat {\Gamma} ^ {\mathsf {T}} Z _ {i} \tag {23.8}
$$

which implies that

$$
\hat {\beta} _ {\mathrm{TSLS}} = \beta + \left\{\hat {\Gamma} ^ {\mathsf {T}} \left(n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Z _ {i} ^ {\mathsf {T}}\right) \hat {\Gamma} \right\} ^ {- 1} \hat {\Gamma} ^ {\mathsf {T}} \left(n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \varepsilon_ {i}\right). \tag {23.9}
$$

Based on (23.9), we can see the consistency of the TSLS estimator because the term $n ^ { - 1 } \sum _ { i = 1 } ^ { n } Z _ { i } \varepsilon _ { i }$ has probability limit $E ( Z \varepsilon ) = 0$ . We can also use (23.9) to show that when $Z$ and $D$ have the same dimension, $\hat { \beta } _ { \mathrm { T S L S } }$ is numerically identical to $\hat { \beta } _ { \mathrm { I V } }$ defined in Section 23.4, which is left as Problem 23.1.

Based on (23.7), we can obtain the standard error as follows. We first obtain the residual $\hat { \varepsilon } _ { i } = Y _ { i } - \hat { \beta } _ { \mathrm { T S L S } } ^ { \sf T } D _ { i }$ , and then obtain the robust variance estimator as

$$
\hat {V} _ {\mathrm{TSLS}} = \left(\sum_ {i = 1} ^ {n} \hat {D} _ {i} \hat {D} _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \left(\sum_ {i = 1} ^ {n} \hat {\varepsilon} _ {i} ^ {2} \hat {D} _ {i} \hat {D} _ {i} ^ {\mathsf {T}}\right) \left(\sum_ {i = 1} ^ {n} \hat {D} _ {i} \hat {D} _ {i} ^ {\mathsf {T}}\right) ^ {- 1}.
$$

Importantly, the $\hat { \varepsilon } _ { i } { } ^ { \dagger } \mathrm { s }$ are not the residual from the second stage OLS $Y _ { i } -$ $\hat { \beta } _ { \mathrm { T S L S } } ^ { \mathsf { T } } \hat { D } _ { i }$ , so $\hat { V } _ { \mathrm { T S L S } }$ differs from the robust variance estimator from the second stage OLS.

## 23.6 A Special Case: A Single IV for a Single Endogenous Treatment

This section focuses on a simple case with a single IV and a single endogenous treatment. It has wide applications. Consider the following structural equations:

$$
\left\{ \begin{array}{l} Y _ {i} = \beta_ {0} + \beta_ {1} D _ {i} + \beta_ {2} ^ {\mathsf {T}} X _ {i} + \varepsilon_ {i}, \\ D _ {i} = \gamma_ {0} + \gamma_ {1} Z _ {i} + \gamma_ {2} ^ {\mathsf {T}} X _ {i} + \varepsilon_ {2 i}, \end{array} \right. \tag {23.10}
$$

where $D _ { i }$ is a scalar endogenous regressor representing the treatment variable of interest $( { \mathrm { i . e . , ~ } } E ( \varepsilon _ { i } D _ { i } ) \neq 0 )$ , Zi is a scalar IV for ${ \cal D } _ { i } \ ( \mathrm { i . e . , } \ E ( \varepsilon _ { i } Z _ { i } ) = 0 )$ , and $X _ { i }$ contains other exogenous regressors $( { \mathrm { i . e . , ~ } } E ( \varepsilon _ { i } X _ { i } ) = 0 )$ . This is a special case with $D$ replaced by $( 1 , D , X )$ and $Z$ replaced by $( 1 , Z , X )$ .

## 23.6.1 Two-stage least squares

The TSLS estimator in Definition 23.3 simplifies to the following form.

Definition 23.4 (TSLS with a single endogenous regressor) Based on (23.10), the TSLS estimator has the following two steps:

1. run OLS of D on $( 1 , Z , X )$ , and obtain the fitted value $\hat { D } _ { i } ~ ( i =$ $1 , \ldots , n )$ ;  
2. run OLS of Y on $( 1 , { \hat { D } } , X )$ , and obtain the coefficient $\hat { \beta } _ { \mathrm { T S L S } }$ , and in particular, $\hat { \beta } _ { 1 , \mathrm { T S L S } }$ , the coefficient of $\hat { D }$ .

## 23.6.2 Indirect least squares

The structural equation (23.10) implies

$$
\begin{array}{l} Y _ {i} = \beta_ {0} + \beta_ {1} (\gamma_ {0} + \gamma_ {1} Z _ {i} + \gamma_ {2} ^ {\mathsf {T}} X _ {i} + \varepsilon_ {2 i}) + \beta_ {2} ^ {\mathsf {T}} X _ {i} + \varepsilon_ {i} \\ = \left(\beta_ {0} + \beta_ {1} \gamma_ {0}\right) + \beta_ {1} \gamma_ {1} Z _ {i} + \left(\beta_ {2} + \beta_ {1} \gamma_ {2}\right) ^ {\mathsf {T}} X _ {i} + \left(\varepsilon_ {i} + \beta_ {1} \varepsilon_ {2 i}\right). \\ \end{array}
$$

Define $\Gamma _ { 0 } = \beta _ { 0 } + \beta _ { 1 } \gamma _ { 0 } , \Gamma _ { 1 } = \beta _ { 1 } \gamma _ { 1 } , \Gamma _ { 2 } = \beta _ { 2 } + \beta _ { 1 } \gamma _ { 2 }$ , and $\varepsilon _ { 1 i } = \varepsilon _ { i } + \beta _ { 1 } \varepsilon _ { 2 i }$ . We have the following equations

$$
\left\{ \begin{array}{l} Y _ {i} = \Gamma_ {0} + \Gamma_ {1} Z _ {i} + \Gamma_ {2} ^ {\mathsf {T}} X _ {i} + \varepsilon_ {1 i}, \\ D _ {i} = \gamma_ {0} + \gamma_ {1} Z _ {i} + \gamma_ {2} ^ {\mathsf {T}} X _ {i} + \varepsilon_ {2 i}, \end{array} \right. \tag {23.11}
$$

which is called the reduced form. The parameter of interest equals the ratio of two coefficients

$$
\beta_ {1} = \Gamma_ {1} / \gamma_ {1}.
$$

In the reduced form, the left-hand side are dependent variables $Y$ and D, and the right-hand side are the exogenous variable Z and X satisfying

$$
E (Z \varepsilon_ {1 i}) = E (Z \varepsilon_ {2 i}) = 0, \quad E (X \varepsilon_ {1 i}) = E (X \varepsilon_ {2 i}) = 0.
$$

More importantly, OLS gives consistent estimators for the coefficients in the reduced form.

The reduced form (23.11) suggests that the ratio of two OLS coefficients $\hat { \Gamma } _ { 1 }$ and $\hat { \gamma } _ { 1 }$ is a reasonable estimator for $\beta _ { 1 }$ . This is called the indirect least squares (ILS) estimator:

$$
\hat {\beta} _ {1, \mathrm{ILS}} \equiv \hat {\Gamma} _ {1} / \hat {\gamma} _ {1}.
$$

Interestingly, it is numerically identical to the TSLS estimator under (23.10).

Theorem 23.1 With a single endogenous treatment and a single $I V ,$ we have

$$
\hat {\beta} _ {1, \mathrm{ILS}} = \hat {\beta} _ {1, \mathrm{TSLS}}.
$$

Theorem 23.1 is an algebraic fact. Imbens (2014, Section A.3) pointed it out without giving a proof. I relegate its proof to Problem 23.2. The ratio formula makes it clear that the TSLS estimator has poor finite sample properties with a weak instrument variable, i.e., $\gamma _ { 1 }$ is close to zero.

## 23.6.3 Weak IV

The following inferential procedure is simpler, more transparent, and more robust to weak IV. It is more computationally intensive though. The reduced form (23.11) also implies that

$$
Y _ {i} - b D _ {i} = (\Gamma_ {0} - b \gamma_ {0}) + (\Gamma_ {1} - b \gamma_ {1}) Z _ {i} + (\Gamma_ {2} - b \gamma_ {2}) ^ {\mathsf {T}} X _ {i} + (\varepsilon_ {1 i} - b \varepsilon_ {2 i}). (2 3. 1 2)
$$

At the true value $b = \beta _ { 1 }$ , the coefficient of $Z _ { i }$ must be 0. This simple fact suggests a confidence interval for $\beta _ { 1 }$ by inverting tests for $H _ { 0 } ( b ) : \beta _ { 1 } = b \colon$

$$
\left\{b: \left| t _ {Z} (b) \right| \leq z _ {\alpha} \right\},
$$

where $t _ { Z } ( b )$ is the t-statistic for the coefficient of Z based on the OLS fit of (23.12) with the EHW standard error. This confidence interval is more robust than the Wald-type confidence interval based on the TSLS estimator. It is similar to the Fieller–Anderson–Rubin confidence interval discussed in Chapter 21. This procedure makes the TSLS estimator unnecessary, and what is more, we only need to run the OLS fit of Y based on the reduced form if the goal is to test $\beta _ { 1 } = 0$ under (23.10).

## 23.7 Application

Card (1993) used the National Longitudinal Survey of Young Men to estimate the causal effect of education on earnings. The data set contains 3010 men with age between 14 and 24 in the year 1966, and Card (1993) leveraged the geographic variation in college proximity as an IV for education. Here, Z is the indicator of growing up near a four-year college, D measures the years of education, and the outcome Y is the log wage in the year 1976, ranging from 4.6 to 7.8. Additional covariates are ace, age and squared age, a categorical variable indicating living with both parents, single mom, or both parents, and variables summarizing the living areas in the past.

```txt
> library("car")
>
> ## Card Data
> card.data = read.csv("card1995.csv")
> Y = card.data[, "lwage"]
> D = card.data[, "educ"]
> Z = card.data[, "nearc4"]
> X = card.data[, c("exper", "expersq", "black", "south", "smsa", "reg661", "reg662", "reg663", "reg664", "reg665", "reg666", "reg667", "reg668", "smsa66")]
> X = as.matrix(X)
```

Based on TSLS, the point estimator is 0.132 and the 95% confidence interval is [0.026, 0.237].

```txt
> Dhat = lm(D ~ Z + X)$fitted.values
> tslsreg = lm(Y ~ Dhat + X)
> tslsest = coef(tslsreg)[2]
> ## correct se by changing the residuals
> res.correct = Y - cbind(1, D, X) % * %coef(tslsreg)
> tslsreg$residuals = as.vector(res.correct)
> tslsse = sqrt(hccm(tslsreg, type = "hc0")[2, 2])
> res = c(tslsest, tslsest - 1.96*tslsse, tslsest + 1.96*tslsse)
> names(res) = c("est", "l.ci", "u.ci")
> round(res, 3)
    est l.ci u.ci
0.132 0.026 0.237
```

Figure 23.3 shows the p-values for a sequence of tests for the coefficient of D. It also implies the 95% confidence interval for the coefficient of D based on inverting tests, which is [0.028, 0.282].

```diff
> BetaAR = seq(-0.1, 0.4, 0.001)
> PvalueAR = sapply(BetaAR,
+    function(b){
+    Y_b = Y - b*D
+    ARreg = lm(Y_b ~ Z + X)
+    coefZ = coef(ARreg)[2]
+    seZ = sqrt(hccm(ARreg)[2, 2])
+    Tstat = coefZ/seZ
+    (1 - pnorm(abs(Tstat))) * 2
+    })
> point.est = BetaAR[which.max(PvalueAR)]
> point.est
[1] 0.132
> ARCI = range(BetaAR[PvalueAR >= 0.05])
> ARCI
[1] 0.028 0.282
```

Comparing the above two methods, the lower confidence limits are very close but the upper confidence limits are slightly different due to the possibly heavy right tail of the distribution of the TSLS estimator.

## 23.8 Homework

23.1 More algebra for TSLS in Section 23.5

1. Show that the Γ in (23.8) equals ˆ

$$
\hat {\Gamma} = \left(\sum_ {i = 1} ^ {n} Z _ {i} Z _ {i} ^ {\mathsf {T}}\right) ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} D _ {i} ^ {\mathsf {T}}.
$$

2. Show $\hat { \beta } _ { \mathrm { T S L S } }$ defined in (23.5) reduces to $\hat { \beta } _ { \mathrm { I V } }$ defined in (23.4) if Z and D have the same dimension and

$$
n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Z _ {i} ^ {\mathsf {T}}, \quad n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} D _ {i} ^ {\mathsf {T}}
$$

are both invertible.

23.2 Equivalence between TSLS and ILS

Prove Theorem 23.1.

Hint: Use the Frisch–Waugh–Lovell theroem.

23.3 Control function in the linear instrumental variable model

Definition 23.5 below parallels Definition 23.3 above.

Definition 23.5 (control function) Define the control function estimator $\hat { \beta } _ { \mathrm { C F } }$ as follows.

1. Run OLS of D on $Z ,$ , and obtain the residual $\breve { D } _ { i } \ ( i = 1 , \ldots , n )$ . $I f D _ { i }$ is a vector, then we need to run component-wise $O L S$ to obtain ${ \check { D } } _ { i }$ . Put the residual vectors in a matrix $\check { D }$ with rows $\check { D } _ { i } ^ { \mathsf { T } }$ ;  
2. Run OLS of Y on D and ${ \check { D } } _ { i }$ and obtain the coefficient of $D$ , $\hat { \beta } _ { \mathrm { C F } }$ .

Show that $\hat { \beta } _ { \mathrm { C F } } = \hat { \beta } _ { \mathrm { T S L S } }$

Remark: In Definition 23.5, $\check { D }$ from Step 1 is called the control function for Step 2. Hausman (1978) pointed out this result. Wooldridge (2015) provided more general discussion of the control function methods in more complex models.

Hint: Use the results in Problems A2.3 and A2.4.

## 23.4 Data analysis: Efron and Feldman (1991)

Efron and Feldman (1991) was one of the early studies dealing with noncomppliance under the potential outcomes framework. The original randomized experiment, the Lipid Research Clinics Coronary Primary Prevention Trial (LRC-CPPT), was designed to evaluate the effect of the drug cholestyramine on cholesterol levels. In the dataset EF.csv, the first column contains the binary indicators for treatment and control, the second column contains the proportions of the nominal cholestyramine dose actually taken, the last three columns are cholesterol levels. Note that the individuals did not know whether they were assigned to cholestyramine or to the placebo, but differences in adverse side effects could induce differences in compliance behavior by treatment status. All individuals were assigned the same nominal dose of the drug or placebo, for the same time period. Column 3, $C _ { 3 } .$ , was taken prior to a communication about the benefits of a low- cholesterol diet, Column $4 , C _ { 4 }$ , was taken after this suggestion, but prior to the random assignment to cholestyramine or placebo, and Column $5 , C _ { 5 }$ , an average of post-randomization cholesterol readings, averaged over two-month readings for a period of time averaging 7.3 years for all the individuals in the study. Efron and Feldman (1991) used the change in cholesterol level as the final outcome of interest, defined as $C _ { 5 } - 0 . 2 5 C _ { 3 } - 0 . 7 5 C _ { 4 }$ . The original paper contains more detailed descriptions.

This dataset is more complicated than the noncompliance problem discussed in class. You can analyze it based on your understanding of the problem, but you need to justify your choice of method. There is no gold-standard solution for this problem.

## 23.5 Recommended reading

Imbens (2014) gave an econometrician’s perspective of IV.

## 24

# Application of the Instrumental Variable Method: Fuzzy Regression Discontinuity

The regression discontinuity introduced in Chapter 20 and the instrumental variable introduced in Chapters 21–23 are two important examples of natural experiments. The study designs are not as ideal as the randomized experiments in Part II, but they have features similar to the experiments. That’s why they are called natural experiments.

Compounding regression discontinuity with instrumental variable yields the fuzzy regression discontinuity, another important natural experiment. I will start with examples and then provide a mathematical formulation.

## 24.1 Motivating examples

Chapter 20 introduces the regression discontinuity. The following two examples are slightly different because the treatments received are not deterministic functions of the running variables. Rather, the running variables discontinuously change the probabilities of the treatments received at the cutoff point.

Example 24.1 In 2000, the Government of India launched the Prime Minister’s Village Road Program, and by 2015, this program had funded the construction of all-weather roads to nearly 200,000 villages. Based on village level data, Asher and Novosad (2020) use a regression discontinuity to estimate the effect of new feeder roads on various economic variables. The national program guidelines prioritized larger villages according to arbitrary thresholds based on the 2001 Population Census. The treatment variable equals one if the village received a new road before the year in which the outcomes were measured. The difference between the population size of a village and the threshold did not determine the treatment variable but affected its probability discontinuously at the cutoff point zero.

Example 24.2 Li et al. (2015) used the data on the first-year students enrolled in 2004 to 2006 from two Italian universities to evaluate the causal effect of a university grant on the drop out rate. The students were eligible for this grant if their standardized family income was below 15,000 euros. For simplicity, we use the running variable defined as 15,000 minus the standardized family income. To receive this grant, the students must apply first. Therefore, the eligibility and the application status jointly determined the final treatment status. The running variable alone did not determine the treatment status although it changed the treatment probability at the cutoff point zero.

![image_26](images/image_26.png)

pr(D = 1 | X = x)
1
x₀
X

![image_27](images/image_27.png)

pr(D = 1 | X = x)
1
x₀
X

FIGURE 24.1: The treatment assignments of sharp regression discontinuity (left) and fuzzy regression discontinuity (right)

Example 24.3 Amarante et al. (2016) estimated the impact of in utero exposure to a social assistance program on children’s birth outcomes. They used a regression discontinuity induced by the Uruguayan Plan de Atenci´on Nacional a la Emergencia Social. It was a temporary social assistance program targeted to the poorest 10 percent of households, implemented between April 2005 and December 2007. Households with a predicted low income score below a predetermined threshold were assigned to the program. The predicted income score did not determine whether the mother received at least one program transfer during the pregnancy but it changed the probability of the final treatment received. The birth outcomes included birth weight, weeks of gestation, etc.

The above examples are called fuzzy regression discontinuity in contrast to the (sharp) regression discontinuity in Chapter 20. I will analyze the data in Examples 24.1 and 24.2 in Section 24.3 below.

## 24.2 Mathematical formulation

Let $X _ { i }$ denote the running variable which determines $Z _ { i } ~ = ~ 1 ( X _ { i } ~ \geq ~ x _ { 0 } )$ with the cutoff point $x _ { 0 } .$ . The treatment received $D _ { i }$ may not equal $Z _ { i } ,$ but $\mathrm { p r } ( D _ { i } = 1 \mid X _ { i } = x )$ has a jump at $x _ { 0 }$ . Figure 24.1 compares the treatment received probabilities of the sharp regression discontinuity and fuzzy regression discontinuity. It shows a special case of fuzzy regression discontinuity with $\operatorname { p r } ( D = 1 \mid X < x _ { 0 } ) = 0$ , which is coherent to Example 24.2.

Let $Y _ { i }$ denote the outcome of interest. Viewing $Z _ { i }$ as the treatment assigned, we can define potential outcomes $\{ D _ { i } ( 1 ) , D _ { i } ( 0 ) , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \}$ . The sharp regression discontinuity of Z allows for identification of

$$
\begin{array}{l} \tau_ {D} (x _ {0}) = E \{D (1) - D (0) \mid X = x _ {0} \} \\ = \lim _ {\varepsilon \rightarrow 0 +} E (D \mid Z = 1, X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \rightarrow 0 +} E (D \mid Z = 0, X = x _ {0} - \varepsilon) \\ \end{array}
$$

and

$$
\begin{array}{l} \tau_ {Y} (x _ {0}) = E \{Y (1) - Y (0) \mid X = x _ {0} \} \\ = \lim _ {\varepsilon \rightarrow 0 +} E (Y \mid Z = 1, X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \rightarrow 0 +} E (Y \mid Z = 0, X = x _ {0} - \varepsilon) \\ \end{array}
$$

based on Theorem 20.2. Using $Z$ as an IV for D and imposing the IV assumptions at $X = x _ { 0 }$ , we can identify the local complier average causal effect by applying Theorem 21.1.

Theorem 24.1 Assume

$$
D _ {i} (1) \geq D _ {i} (0)
$$

and

$$
D _ {i} (1) = D _ {i} (0) \Longrightarrow Y _ {i} (1) = Y _ {i} (0)
$$

in the infinitesimal neighborhood $o f x _ { 0 }$ . The local complier average causal effect equals

$$
\begin{array}{l} \tau_ {\mathrm{c}} (x _ {0}) \equiv E \{Y (1) - Y (0) \mid D (1) > D (0), X = x _ {0} \} \\ = \frac {E \{Y (1) - Y (0) \mid X = x _ {0} \}}{E \{D (1) - D (0) \mid X = x _ {0} \}}. \\ \end{array}
$$

Further assume that $E \{ D ( 1 ) \mid X = x \}$ and $E \{ Y ( 1 ) \mid X = x \}$ are continuous from the right at $X = x _ { 0 } \quad$ , and $E \{ D ( 0 ) \mid X = x \}$ and $E \{ Y ( 0 ) \mid X = x \}$ are continuous from the $l e f t$ at $X = x _ { 0 }$ . The local complier average causal effect can be identified by

$$
\tau_ {\mathrm{c}} (x _ {0}) = \frac {\lim _ {\varepsilon \to 0 +} E (Y \mid Z = 1 , X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \to 0 +} E (Y \mid Z = 0 , X = x _ {0} - \varepsilon)}{\lim _ {\varepsilon \to 0 +} E (D \mid Z = 1 , X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \to 0 +} E (D \mid Z = 0 , X = x _ {0} - \varepsilon)}
$$

if the $E ( D \mid Z = 1 , X = x )$ has a non-zero jump at $X = x _ { 0 }$

Theorem 24.1 is a superposition of Theorems 20.2 and 21.1. I leave its proof as Problem 24.1.

In both sharp and fuzzy regression discontinuity, the key is to specify the neighborhood around the cutoff point. Practically, a smaller neighborhood leads to smaller bias but larger variance, while a larger neighborhood leads to larger bias but smaller variance. That is, we face a bias-variance tradeoff. Some automatic procedures exist based on some statistical criteria, which relies on some strong conditions. It seems wiser to conduct sensitivity analysis over a range of the choice of $h .$

## 29624 Application of the Instrumental Variable Method: Fuzzy Regression Discontinuity

Assume that we have specified the neighborhood of $x _ { 0 }$ determined by a bandwidth h. For data with $X _ { i } \in [ x _ { 0 } - h , x _ { 0 } + h ]$ , we can estimate $\tau _ { D } ( x _ { 0 } )$ by

τˆD(x0) = the coefficient of $Z _ { i }$ in the OLS fit of $D _ { i }$ on $\{ 1 , Z _ { i } , R _ { i } , L _ { i } \}$ ,

and estimate $\tau _ { Y } ( x _ { 0 } )$

τˆY (x0) = the coefficient of $Z _ { i }$ in the OLS fit of $Y _ { i }$ on $\{ 1 , Z _ { i } , R _ { i } , L _ { i } \}$ ,

recalling the definitions $R _ { i } = \operatorname* { m a x } ( X _ { i } - x _ { 0 } , 0 )$ and $L _ { i } = \operatorname* { m i n } ( X _ { i } - x _ { 0 } , 0 )$ . Then we can estimate the local complier average causal effect by

$$
\hat {\tau} _ {\mathrm{c}} (x _ {0}) = \hat {\tau} _ {Y} (x _ {0}) / \hat {\tau} _ {D} (x _ {0}).
$$

This is an indirect least squares estimator. By Theorem 23.1, it is numerically identical to

the coefficient of $D _ { i }$ in the TSLS fit of $Y _ { i }$ on $\{ 1 , D _ { i } , R _ { i } , L _ { i } \}$

with $D _ { i }$ instrumented by $Z _ { i }$ . In sum, after specifying h, the estimation of $\tau _ { \mathrm { c } } ( x _ { 0 } )$ reduces to a TSLS procedure with the local data around the cutoff point.

## 24.3 Application

## 24.3.1 Re-analyzing Asher and Novosad (2020)’s data

Figure 24.2 shows the result using occupationindexandrsn as the outcome.

The package rdrobust selects the bandwidth automatically. The results suggest that receiving a new road did not affect the outcome significantly.

```diff
> road_dat = read.csv("indianroad.csv")
> road_dat$runv = road_dat$left + road_dat$right
> library("rdrobust")
> frd_road = with(road_dat,
+    {
+    rdrobust(y = occupation_index_andrsn,
+    x = runv,
+    c = 0,
+    fuzzy = r2012)
+    })
> res = cbind(frd_road$coef, frd_road$se)
> round(res, 3)
    Coeff Std. Err.
Conventional -0.253 0.301
Bias-Corrected -0.283 0.301
Robust -0.283 0.359
```

![image_28](images/image_28.png)

## 24.3.2 Re-analyzing Li et al. (2015)’s data

Recall that the running variable is 15,000 minus the standardized income in Example 24.2. In the analysis, I restrict the data to a subset with this running between [−5, 000, 5, 000], and then divide the running variable by 5, 000 so that the running variable is bounded between [−1, 1] at cutoff point zero.

The results based on the package rdrobust suggest that the university grant did not affect the dropout rate significantly.

```diff
> italy = read.csv("italy.csv")
> library("rdrobust")
> frd_italy = with(italy,
+    {
+    rdrobust(y = outcome,
+    x = rv0,
+    c = 0,
+    fuzzy = D)
```

```txt
+ })  
> res = cbind(frd_italy$coef, frd_italy$se)  
> round(res, 3)  
Coeff Std. Err.  
Conventional -0.149 0.101  
Bias-Corrected -0.155 0.101  
Robust -0.155 0.121
```

## 24.4 Discussion

Both Chapter 20 and this chapter formulate regression discontinuity based on the continuity of the conditional expectations of the potential outcomes given the running variables. This perspective is mathematically simpler but it only identifies the local effects precisely at the cutoff point of the running variable. Hahn et al. (2001) started this line of literature.

An alternative, not so dominant perspective is based on local randomization (Cattaneo et al., 2015; Li et al., 2015). If we view the running variable as a noisy measure of some underlying truth and the cutoff point is somewhat arbitrarily chosen, the units near the cutoff point do not differ systematically. This suggests that in a small neighborhood of the cutoff point, the units receive the treatment and the control in a random fashion just as in a randomized experiment. Similar to the issue of choosing h in the first perspective, it is crucial to decide how local should the randomized experiment be under the regression discontinuity. It is not easy to quantify the intuition mathematically, and again conducting sensitivity analysis with a range of h seems a reasonable approach in the second perspective as well.

See Sekhon and Titiunik (2017) for more conceptual discussion of regression discontinuity.

## 24.5 Homework Problems

## 24.1 Proof of Theorem 24.1

Prove Theorem 24.1.

## 24.2 Data analysis

Section 24.3.1 estimated the effect on occupationindexandrsn. Four other outcome variables are transportindexandrsn, firmsindexandrsn,

## 30024 Application of the Instrumental Variable Method: Fuzzy Regression Discontinuity

consumptionindexandrsn, and agricultureindexandrsn, with meanings defined in the original paper. Estimate the effects on these outcomes.

## 24.3 Reflection on the analysis of Li et al. (2015)’s data

In Li et al. (2015), a key variable determining the treatment status is the binary application status A, which has potential outcomes $A ( 1 )$ and $A ( 0 )$ corresponding to the treatment $Z = 1$ and control $Z = 0$ . By definition,

$$
D (1) = A (1), \quad D (0) = 0,
$$

so the compliers $\{ D ( 1 ) , D ( 0 ) \} = ( 1 , 0 )$ is equivalent to $A ( 1 ) = 1 . \mathrm { \ S o }$

$$
\tau_ {c} (x _ {0}) = E \{Y (1) - Y (0) \mid A (1) = 1, X = x _ {0} \}.
$$

Section 24.3.2 used the whole data set to estimate $\tau _ { \mathrm { c } } ( x _ { 0 } )$ .

An alternative analysis is based on units with $A = 1$ only. Then the treatment status is determined by X. However, this analysis can be problematic because

$$
\lim _ {\varepsilon \rightarrow 0 +} E \{Y \mid A = 1, X = x _ {0} + \varepsilon \} - \lim _ {\varepsilon \rightarrow 0 +} E \{Y \mid A = 1, X = x _ {0} - \varepsilon \}
$$

$$
= E \{Y (1) \mid A (1) = 1, X = x _ {0} \} - E \{Y (0) \mid A (0) = 1, X = x _ {0} \}. \tag {24.1}
$$

Prove (24.1) and explain why this analysis can be problematic.

Remark: The left-hand side of (24.1) is the identification formula of the local average treatment effect at $X = x _ { 0 }$ , conditioning on $A = 1$ . The right-hand side of (24.1) is the difference in means of the potential outcomes for subgroup of units with $( A ( 1 ) = 1 , X = x _ { 0 } )$ and $( A ( 0 ) = 1 , X = x _ { 0 } )$ , respectively.

## 24.4 Recommended reading

Imbens and Lemieux (2008) gave a practical guidance to regression discontinuity based on the potential outcomes framework. Lee and Lemieux (2010) reviewed regression discontinuity and its applications in economics.

## 25

# Application of the Instrumental Variable Method: Mendelian Randomization

Katan (1986) was concerned with the observational studies suggesting that low serum cholesterol levels were associated with the risk of cancer. As we have discussed, however, observational studies suffer from unmeasured confounding. Consequently, it is difficult to interpret the apparent association as causality. In the particular problem studied by Katan (1986), it is even possible that early stages of cancer reversely cause low serum cholesterol levels. Disentangling the causal effect of the serum cholesterol level on cancer seems a hard problem using standard epidemiologic studies. Katan (1986) argued that Apolipoprotein E genes are associated with the serum cholesterol levels but do not directly affect the cancer status. So if low serum cholesterol levels causes cancer, we should observe differences in cancer risks among people with and without the genotype that leads to different serum cholesterol levels. Using our language for causal inference, Katan (1986) proposed to use Apolipoprotein E genes as IVs.

Katan (1986) did not conduct any data analysis but just proposed a conceptual design that could address not only unmeasured confounding but also reverse causality. Since then, more complicated and sophisticated studies have been conducted thanks to the modern genome-wide association studies. These studies used genetic information as IVs for exposures in epidemiologic studies to estimate causal effects of exposures on outcomes. They were all motivated by Mendel’s second law, the law of random assortment, which suggests the inheritance of one trait is independent of the inheritance of other traits. Therefore, the method of using genetic information as IV is called Mendelian Randomization (MR).

## 25.1 Background and motivation

Graphically, Figure 25.1 shows the causal diagram on the treatment D, outcome Y , unmeasured confounder U, as well as the genetic IVs $G _ { 1 } , \ldots , G _ { p }$ . In many Mendelian Randomization studies, the genetic IVs are single nucleotide polymorphisms (SNPs). Because of pleiotropy, it is possible that the genetic

![image_29](images/image_29.png)

```mermaid
graph TD
  G1 -->|α1| U
  G2 -->|α2| D
  Gp -->|γp| D
  D -->|β| Y
    G1 -.-> G2
    G2 -.-> G1
    D -.-> Y
    Gp -.-> Y
```

FIGURE 25.1: Causal graph for Mendelian randomization

IVs have direct effect on the outcome of interest, so Figure 25.1 also allows for the violation of the exclusion restriction assumption.

The standard linear IV model assumes away the direct effect of the IVs on the outcome. Definition 25.1 below gives both the structural and reduces forms.

Definition 25.1 (linear IV model) The standard linear IV model

$$
Y = \beta_ {0} + \beta D + \beta_ {u} U + \varepsilon_ {Y}, \tag {25.1}
$$

$$
D = \gamma_ {0} + \gamma_ {1} G _ {1} + \dots + \gamma_ {p} G _ {p} + \gamma_ {u} U + \varepsilon_ {D}, \tag {25.2}
$$

has reduced form

$$
Y = \beta_ {0} + \beta \gamma_ {0} + \beta \gamma_ {1} G _ {1} + \dots + \beta \gamma_ {p} G _ {p} + (\beta_ {u} + \beta_ {0} \gamma_ {u}) U + \varepsilon_ {Y}, \tag {25.3}
$$

$$
D = \gamma_ {0} + \gamma_ {1} G _ {1} + \dots + \gamma_ {p} G _ {p} + \gamma_ {u} U + \varepsilon_ {D}, \tag {25.4}
$$

Definition 25.2 below allows for the violation of exclusion restriction. Then, $G _ { 1 } , \ldots , G _ { p }$ are not valid IVs.

Definition 25.2 (linear model with possibly invalid IVs) The linear model

$$
Y = \beta_ {0} + \beta D + \alpha_ {1} G _ {1} + \dots + \alpha_ {p} G _ {p} + \beta_ {u} U + \varepsilon_ {Y}, \tag {25.5}
$$

$$
D = \gamma_ {0} + \gamma_ {1} G _ {1} + \dots + \gamma_ {p} G _ {p} + \gamma_ {u} U + \varepsilon_ {D}, \tag {25.6}
$$

has reduced form

$$
Y = (\beta_ {0} + \beta \gamma_ {0}) + (\alpha_ {1} + \beta \gamma_ {1}) G _ {1} + \dots + (\alpha_ {p} + \beta \gamma_ {p}) G _ {p}
$$

$$
+ (\beta_ {u} + \beta \gamma_ {u}) U + \varepsilon_ {Y}, \tag {25.7}
$$

$$
D = \gamma_ {0} + \gamma_ {1} G _ {1} + \dots + \gamma_ {p} G _ {p} + \gamma_ {u} U + \varepsilon_ {D}. \tag {25.8}
$$

Therefore, in Definition 25.1 with exclusion restriction, we have

$$
\Gamma_ {j} = \beta \gamma_ {j}, (j = 1, \ldots , p);
$$

in Definition 25.2 without exclusion restriction, we have

$$
\Gamma_ {j} = \alpha_ {j} + \beta \gamma_ {j}, (j = 1, \ldots , p).
$$

If we have individual data, we can apply the classic TSLS estimator to estimate $\beta$ under the linear IV model in Definition 25.1. However, most Mendelian Randomization studies do not have individual data but rather summary statistics from multiple genome-wide association studies. A canonical setting consists of the regression coefficients of the treatment on the genetic IVs:

$$
\hat {\gamma} _ {1} \rightarrow \gamma_ {1}, \dots , \hat {\gamma} _ {p} \rightarrow \gamma_ {p} \tag {25.9}
$$

in probability with standard errors

$$
\mathrm{se} _ {D 1}, \dots , \mathrm{se} _ {D p}, \tag {25.10}
$$

and the regression coefficients of the outcome on the genetic IVs:

$$
\hat {\Gamma} _ {1} \rightarrow \Gamma_ {1}, \dots , \hat {\Gamma} _ {p} \rightarrow \Gamma_ {p} \tag {25.11}
$$

in probability with standard errors

$$
\operatorname{se} _ {Y 1}, \dots , \operatorname{se} _ {Y p}. \tag {25.12}
$$

I will focus on the statistical inference of $\beta$ based on the above summary statistics. For simplicity, we assume that the estimates in (25.9) and (25.11) are jointly independent, they are all asymptotically normal, and the standard errors in (25.10) and (25.12) are all fixed and known. The asymptotic normality can often be justified by central limit theorems of the regression coefficients. The standard errors are accurate estimates of the true standard errors. Therefore, the only subtle assumption is the joint independence of the regression coefficients in (25.9) and (25.11). The independence of the $\hat { \gamma } _ { j } \mathrm { ^ { \circ } s }$ and the $\hat { \Gamma } _ { j } \mathrm { ' s }$ are reasonable because they are often calculated based on different samples. The independence among the $\hat { \gamma } _ { j }$ ’s can be reasonable if the $G _ { j }$ ’s are independent and the true linear model for D holds with homoskedastic error terms1. The independence among the $\hat { \Gamma } _ { j } \mathrm { ' s }$ follows from a similar argument.

## 25.2 MR based on summary statistics

## 25.2.1 Fixed-effect estimator

Under Definition 25.1, $\alpha _ { j } = 0$ which implies that $\beta = \Gamma _ { j } / \gamma _ { j }$ for all $j$ . A simple approach is based on the so-called meta-analysis (Bowden et al., 2018), that is,

## 30425 Application of the Instrumental Variable Method: Mendelian Randomization

combining multiple estimates $\hat { \beta } _ { j } = \hat { \Gamma } _ { j } / \hat { \gamma } _ { j }$ for the common parameter $\beta .$ Using delta method (see Example $\operatorname { A 1 . 3 } ) , \hat { \beta } _ { j }$ has approximate squared standard error

$$
\mathrm{se} _ {j} ^ {2} = (\mathrm{se} _ {Y j} ^ {2} + \hat {\beta} _ {j} ^ {2} \mathrm{se} _ {D j} ^ {2}) / \hat {\gamma} _ {j} ^ {2}.
$$

Therefore, the best linear combination to estimate $\beta$ is the Fisher weighting based on inverse of the variances:

$$
\hat {\beta} _ {\mathrm{fisher0}} = \frac {\sum_ {j = 1} ^ {p} \hat {\beta} _ {j} / \mathrm{se} _ {j} ^ {2}}{\sum_ {j = 1} ^ {p} 1 / \mathrm{se} _ {j} ^ {2}}
$$

which has variance $( \sum _ { j = 1 } ^ { p } 1 / \mathrm { s e } _ { j } ^ { 2 } ) ^ { - 1 }$ . Ignoring the uncertainty due to $\hat { \gamma } _ { j }$ quantified by $\mathrm { s e } _ { D j }$ , the estimator reduces to

$$
\hat {\beta} _ {\mathrm{fisher1}} = \frac {\sum_ {j = 1} ^ {p} \hat {\beta} _ {j} \hat {\gamma} _ {j} ^ {2} / \mathrm{se} _ {Y j} ^ {2}}{\sum_ {j = 1} ^ {p} \hat {\gamma} _ {j} ^ {2} / \mathrm{se} _ {Y j} ^ {2}} = \frac {\sum_ {j = 1} ^ {p} \hat {\Gamma} _ {j} \hat {\gamma} _ {j} / \mathrm{se} _ {Y j} ^ {2}}{\sum_ {j = 1} ^ {p} \hat {\gamma} _ {j} ^ {2} / \mathrm{se} _ {Y j} ^ {2}},
$$

which has variance $\textstyle ( \sum _ { j = 1 } ^ { p } 1 \hat { \gamma } _ { j } ^ { 2 } / \mathrm { s e } _ { Y j } ^ { 2 } ) ^ { - 1 }$ . Inference based on $\hat { \beta } _ { \mathrm { f i s h e r 1 } }$ is suboptimal although it is more widely used in practice (Bowden et al., 2018).

Focus on the suboptimal yet simpler estimator $\hat { \beta } _ { \mathrm { f i s h e r 1 } }$ . Under Definition 25.2, we can show that

$$
\hat {\beta} _ {\mathrm{fisher1}} \rightarrow \frac {\sum_ {j = 1} ^ {p} \Gamma_ {j} \gamma_ {j} / \mathrm{se} _ {Y j} ^ {2}}{\sum_ {j = 1} ^ {p} \gamma_ {j} ^ {2} / \mathrm{se} _ {Y j} ^ {2}} = \beta + \frac {\sum_ {j = 1} ^ {p} \alpha_ {j} \gamma_ {j} / \mathrm{se} _ {Y j} ^ {2}}{\sum_ {j = 1} ^ {p} \gamma_ {j} ^ {2} / \mathrm{se} _ {Y j} ^ {2}}
$$

in probability. If $\alpha _ { j } = 0$ for all $j , \hat { \beta } _ { \mathrm { f i s h e r 1 } }$ is consistent. Even this does not hold, it is still possible that $\hat { \beta } _ { \mathrm { f i s h e r 1 } }$ is consistent as long as the inner product between $\alpha _ { j }$ and $\gamma _ { j }$ weighted by $1 / \mathrm { s e } _ { Y j } ^ { 2 }$ is zero. This holds if we have many genetic instruments and violation of the exclusion restriction, captured by $\alpha _ { j }$ , is an independent random draw from a distribution with mean zero.

## 25.2.2 Egger regression

Start with Definition 25.1. With the true parameters, we have

$$
\Gamma_ {j} = \beta \gamma_ {j} \quad (j = 1, \dots , p);
$$

with the estimates, the above identify holds only approximately

$$
\hat {\Gamma} _ {j} \approx \beta \hat {\gamma} _ {j} (j = 1, \dots , p).
$$

This seems a classic OLS problem of $\{ \hat { \Gamma } _ { j } \} _ { j = 1 } ^ { p }$ on $\{ \hat { \gamma } _ { j } \} _ { j = 1 } ^ { p }$ . We can fit an OLS of $\hat { \Gamma } _ { j }$ on $\hat { \gamma } _ { j } ,$ , with or without an intercept, possibly weighted by $w _ { j }$ , to estimate $\beta .$ . The following results hold thanks to the algebraic properties of the WLS reviewed in Section A2.5.

Without an intercept, the coefficient of $\hat { \gamma } _ { j }$ is

$$
\hat {\beta} _ {\mathrm{egger1}} = \frac {\sum_ {j = 1} ^ {p} \hat {\gamma} _ {j} \hat {\Gamma} _ {j} w _ {j}}{\sum_ {j = 1} ^ {p} \hat {\gamma} _ {j} ^ {2} w _ {j}},
$$

which reduces to $\hat { \beta } _ { \mathrm { f i s h e r 1 } } \ \mathrm { i f } \ w _ { j } = 1 / \mathrm { s e } _ { Y j } ^ { 2 } .$ So the Egger regression is more general than the fixed-effect estimator in Section 25.2.1.

With an intercept, the coefficient of $\hat { \gamma } _ { j }$ is

$$
\hat {\beta} _ {\mathrm{egger0}} = \frac {\sum_ {j = 1} ^ {p} (\hat {\gamma} _ {j} - \hat {\gamma} _ {w}) (\hat {\Gamma} _ {j} - \hat {\Gamma} _ {w}) w _ {j}}{\sum_ {j = 1} ^ {p} (\hat {\gamma} _ {j} - \hat {\gamma} _ {w}) ^ {2} w _ {j}}
$$

where $\begin{array} { r } { \hat { \gamma } _ { w } = \sum _ { j = 1 } ^ { p } \hat { \gamma } _ { j } w _ { j } / \sum _ { j = 1 } ^ { p } w _ { j } } \end{array}$ and $\begin{array} { r } { \hat { \Gamma } _ { w } = \sum _ { j = 1 } ^ { p } \hat { \Gamma } _ { j } w _ { j } / \sum _ { j = 1 } ^ { p } w _ { j } } \end{array}$ are the weighted averages of the $\hat { \gamma } _ { j } \mathrm { ^ s }$ and $\hat { \Gamma } _ { j } \mathrm { ' s } ,$ respectively. Even without assuming that all $\gamma _ { j } \mathrm { : }$ s are zero under Definition 25.2, we have

$$
\hat {\beta} _ {\mathrm{egger0}} \to \frac {\sum_ {j = 1} ^ {p} (\gamma_ {j} - \gamma_ {w}) (\Gamma_ {j} - \Gamma_ {w}) w _ {j}}{\sum_ {j = 1} ^ {p} (\gamma_ {j} - \gamma_ {w}) ^ {2} w _ {j}} = \beta + \frac {\sum_ {j = 1} ^ {p} (\gamma_ {j} - \gamma_ {w}) (\alpha_ {j} - \alpha_ {w}) w _ {j}}{\sum_ {j = 1} ^ {p} (\gamma_ {j} - \gamma_ {w}) ^ {2} w _ {j}}
$$

in probability, where $\gamma _ { w } , \Gamma _ { w }$ and $\alpha _ { w }$ are the corresponding weighted averages of the true parameters. So $\hat { \beta } _ { \mathrm { e g g e r 0 } }$ is consistent for $\beta$ as long as the weighted least squares coefficient of $\alpha _ { j }$ on $\gamma _ { j }$ is zero. This is weaker than $\alpha _ { j } = 0$ for all $j .$ . This weaker assumption holds if $\gamma _ { j }$ and $\alpha _ { j }$ are realizations of independent random variables, which is called the Instrument Strength Independent of Direct Effect assumption (Bowden et al., 2015). More interestingly, the intercept from the Egger regression is

$$
\hat {\alpha} _ {\mathrm{egger0}} = \hat {\Gamma} _ {w} - \hat {\beta} _ {\mathrm{egger0}} \hat {\gamma} _ {w},
$$

which, under the InSIDE assumption converges to

$$
\Gamma_ {w} - \beta \gamma_ {w} = \alpha_ {w}
$$

in probability. So the intercept estimates the weighted average of the direct effects.

## 25.3 An example

I use the bmi.sbp data in the mr.raps package to illustrate the Egger regressions.

```txt
> library("mr.raps")
> bmisbp = subset(bmi.sbp,
```

30625 Application of the Instrumental Variable Method: Mendelian Randomization

```txt
+ select = c("beta.exposure", "beta.outcome", "se.exposure", "se.outcome"))
```

The Egger regressions with or without the intercept give very similar results.

```txt
> mr.egger = lm(beta.outcome ~ 0 + beta.exposure,
+    data = bmisbp,
+    weights = 1/se.outcome^2)
> summary(mr.egger)
```

Call :

```javascript
lm(formula = beta.outcome ~ 0 + beta.exposure, data = bmisbp, weights = 1/se.outcome^2)
```

Weighted Residuals :

```txt
Min 1Q Median 3Q Max
-5.6999 -1.1691 -0.0199 1.0073 11.3449
```

Coefficients :

```txt
Estimate Std. Error t value Pr(>|t|)
beta.exposure 0.3173 0.1106 2.869 0.00468 **
```

```txt
Residual standard error: 2.052 on 159 degrees of freedom
Multiple R-squared: 0.04921, Adjusted R-squared: 0.04323
F-statistic: 8.229 on 1 and 159 DF, p-value: 0.004682
```

>

```txt
> mr.egger.w = lm(beta.outcome ~ beta.exposure,
+    data = bmisbp,
+    weights = 1/se.outcome^2)
> summary(mr.egger.w)
```

Call :

```javascript
lm(formula = beta.outcome ~ beta.exposure, data = bmisbp, weights = 1/se.outcome^2)
```

Weighted Residuals :

```txt
Min 1Q Median 3Q Max
-5.7099 -1.1774 -0.0296 0.9969 11.3393
```

Coefficients :

```txt
Estimate Std. Error t value Pr(>|t|)
(Intercept) 0.0001133 0.0020794 0.055 0.95660
beta.exposure 0.3172989 0.1109485 2.860 0.00481 **
```

```txt
Residual standard error: 2.059 on 158 degrees of freedom
Multiple R-squared: 0.04922, Adjusted R-squared: 0.0432
F-statistic: 8.179 on 1 and 158 DF, p-value: 0.004811
```

## 25.4 Critiques of the analysis based on Mendelian randomization

MR is an application of the idea of IV. It relies on strong assumptions. I provide three sets of critiques from the conceptual, biological and technical perspectives.

Conceptually, most studies based on MR have illy defined treatments from the potential outcomes perspective. For instance, the treatments are often defined as the cholesterol level or body mass index. They are composite variables and can correspond to complex, non-unique definitions of the hypothetical experiments. The SUTVA often does not hold for these treatments.

Biologically, the fundamental assumptions for the IV analysis may not hold. Mendel’s second law ensures that the inheritances of different traits are independent. However, it does not ensure that the candidate IVs are independent of the hidden confounders between the treatment and the outcome.

## 30825 Application of the Instrumental Variable Method: Mendelian Randomization

It is possible that these IVs have direct effects on the confounders. It is also possible that some unmeasured genes affect both the IVs and the confounders. Mendel’s second law does not ensure the exclusion restriction assumption either. It is possible that the IVs have other causal pathways to the outcome, beyond the pathway through the treatment of interest.

Technically, the statistical assumptions for MR are quite strong. Clearly, the linear IV model is a strong modeling assumption. The independence of the $\hat { \gamma } _ { j }$ ’s and the $\hat { \Gamma } _ { j }$ ’s is also quite strong. Other issues in the data collecting process can further complicate the interpretation of the IV assumptions. For instance, the treatments and outcomes are often measured with errors, and the genome wide associate studies are often based on the case-control design.

VanderWeele et al. (2014) is an excellent review paper that discusses the methodological challenges in MR.

## 25.5 Homework Problems

## 25.1 Data analysis

Analyze the bmi.bmi data in the R package mr.raps. See the package and Zhao et al. (2020, Section 7.2) for more details.

## 25.2 Recommended reading

Davey Smith and Ebrahim (2003) reviewed the potentials and limitations of Mendelian randomization.

## Part VI

## Causal Mechanisms with Post-Treatment Variables

# Principal Stratification

Parts II–V focus on causal effects of a treatment on an outcome, possibly adjusting for observed pretreatment covariates. Many applications also have some post-treatment variable M which happens after the treatment but before the outcome. An important question is how to use the post-treatment variable M appropriately. I will start with several motivating examples and then introduce Frangakis and Rubin (2002)’s formulation of this problem based on potential outcomes.

## 26.1 Motivating Examples

Example 26.1 (noncompliance) In randomized experiments with noncompliance, we can use M to represent the treatment received, which is affected by the treatment assignment Z and affects the outcome Y . In this example, M has the same meaning as D in Chapter 21.

Example 26.2 (truncation by death) In randomized experiments to patients with severe diseases, some patients may die before the measurement of the outcome Y , e.g., the quality of life. The post-treatment variable M in this example is the binary indicator of the survival status.

Example 26.3 (unemployment) In job training programs, units are randomly assigned to treatment and control groups, and report their employment status M and wage Y . Then the post-treatment variable is the binary indicator of the employment status M .

Example 26.4 (surrogate endpoint) In clinical trials, the outcomes of interest (e.g., 30 years of survival) require a long and costly follow-up. Practitioners instead collect data on some other variables early in the follow-up that are easy to measure. These variables are called the “surrogate endpoints.” A concrete example is from clinical trials on HIV patients, where the candidate surrogate endpoint M is the CD4 cell count.

Examples 26.1–26.4 have the similarity that an intermediate variable M occurs between the treatment and the outcome. Here “between” can mean that

1. M is on the causal pathway from Z to Y as Figure 26.1(a);  
2. M is not on the causal pathway from Z to Y as Figure 26.1(b).

Example 26.1 corresponds to Figure 26.1(a). Examples 26.2 and 26.3 correspond to Figure 26.1(b). Example 26.4 can correspond to Figure 26.1(a) or (b), depending on the choice of the surrogate end point.

![image_30](images/image_30.png)

```mermaid
graph TD
  Z --> M
  M --> Y
  U --> Y
  Z --> M
```

![image_31](images/image_31.png)

```mermaid
graph TD
  Z --> M
  M --> U
  M --> Y
  U --> Y
```

FIGURE 26.1: Causal diagrams with a post-treatment variable M

## 26.2 The Problem of Conditioning on the Post-Treatment Variable

A naive method to deal with the post-treatment variable M is to condition on its observed value as if it were a pretreatment covariate. However, M is fundamentally different from X, because the former is affected by the treatment in general but the latter is not. It is also a “rule of thumb” that data analyzers should not condition on any post-treatment variables in evaluating the average effect of the treatment on the outcome (Cochran, 1957; Rosenbaum, 1984). Based on potential outcomes, Frangakis and Rubin (2002) gave the following insightful explanation.

For simplicity, we focus on completely randomized experiment in this chapter.

Assumption 26.1 (complete randomization with an intermediate variable) $Z \bot \bot \{ M ( 1 ) , M ( 0 ) , Y ( \dot { 1 } ) , Y ( 0 ) \}$ .

Conditioning on $M = m$ , we compare

$$
\operatorname{pr} (Y \mid Z = 1, M = m)
$$

and

$$
\operatorname{pr} (Y \mid Z = 0, M = m).
$$

This comparison seems intuitive which measures the difference in the outcome distributions in treated and control groups given the same value of the posttreatment variable. When M is a pre-treatment covariate, this comparison yields a reasonable subgroup effect. However, when M is a post-treatment variable, the interpretation of this comparison is problematic. Under Assumption 26.1, we can re-write

$$
\begin{array}{l} \operatorname{pr} (Y \mid Z = 1, M = m) = \operatorname{pr} \{Y (1) \mid Z = 1, M (1) = m \} \\ = \operatorname{pr} \{Y (1) \mid M (1) = m \} \\ \end{array}
$$

and

$$
\begin{array}{l} \operatorname{pr} (Y \mid Z = 0, M = m) = \operatorname{pr} \{Y (0) \mid Z = 0, M (0) = m \} \\ = \operatorname{pr} \{Y (0) \mid M (0) = m \}. \\ \end{array}
$$

Therefore, we are comparing the distributions of $Y ( 1 )$ and $Y ( 0 )$ for different subset of units because the units with $M ( 1 ) = m$ are different from the units with $M ( 0 ) = m$ if the $Z$ affects M. Consequently, the comparison conditioning on $M = m$ does not have a causal interpretation in general unless $M ( 1 ) =$ $M ( 0 ) . ^ { 1 }$

Revisit Example 26.1. Comparing $\mathrm { p r } ( Y \mid Z = 1 , M = 1 )$ and $\mathrm { p r } ( Y \mid$ $Z = 0 , M = 1 )$ is equivalent to comparing the treated potential outcomes for compliers and always-takers and control potential outcomes for always-takers, under the monotonicity assumption that $M ( 1 ) \ge M ( 0 )$ . Part 3 of Problem 22.7 has pointed out the drawbacks of this analysis.

Revisit Example 26.2. If the treatment improves the survival status, the treatment can save more weak patients than the control. In this case, units with $M ( 1 ) = 1$ are weaker than units with $M ( 0 ) = 1$ , so the naive comparison gives biased results that is in favor of the control.

## 26.3 Conditioning on the Potential Values of the Post-Treatment Variable

Frangakis and Rubin (2002) proposed to condition on the joint potential value of the post-treatment variable $U = \{ M ( 1 ) , M ( 0 ) \}$ and compare

$$
\operatorname{pr} \{Y (1) \mid M (1) = m _ {1}, M (0) = m _ {0} \}
$$

and

$$
\operatorname{pr} \{Y (0) \mid M (1) = m _ {1}, M (0) = m _ {0} \}
$$

for some $( m _ { 1 } , m _ { 0 } )$ . This is a comparison of the potential outcomes under treatment and control for the same subset of units with $M ( 1 ) = m _ { 1 }$ and $M ( 0 ) = m _ { 0 }$ . Frangakis and Rubin (2002) called this strategy principal stratification, viewing $\{ M ( 1 ) , M ( 0 ) \}$ as a pre-treatment covariate. Based on this idea, we can define

$$
\tau (m _ {1}, m _ {0}) = E \{Y (1) - Y (0) \mid M (1) = m _ {1}, M (0) = m _ {0} \}
$$

as the principal stratification average causal effect for the subgroup with $M ( 1 ) = m _ { 1 } , M ( 0 ) = m _ { 0 }$ . For a binary M, we have four subgroups

$$
\left\{ \begin{array}{l c l} \tau (1, 1) & = & E \{Y (1) - Y (0) \mid M (1) = 1, M (0) = 1 \}, \\ \tau (1, 0) & = & E \{Y (1) - Y (0) \mid M (1) = 1, M (0) = 0 \}, \\ \tau (0, 1) & = & E \{Y (1) - Y (0) \mid M (1) = 0, M (0) = 1 \}, \\ \tau (0, 0) & = & E \{Y (1) - Y (0) \mid M (1) = 0, M (0) = 0 \}. \end{array} \right. \tag {26.1}
$$

Since $\{ M ( 1 ) , M ( 0 ) \}$ is unaffected by the treatment, it is a covariate so $\tau ( m _ { 1 } , m _ { 0 } )$ is a subgroup causal effect. For subgroups with $M ( 1 ) = M ( 0 )$ , the treatment does not change the intermediate variable, so $\tau ( 1 , 1 )$ and $\tau ( 0 , 0 )$ measure the dissociative effects. For other subgroups with $m _ { 1 } \neq m _ { 0 }$ , the principal stratification average causal effects $\tau ( m _ { 1 } , m _ { 0 } )$ measure the associative $e f f e c t s .$ These terminologies are from Frangakis and Rubin (2002), which do not assume that M is on the causal pathway from Z to Y . When we have Figure $2 6 . 1 ( \mathrm { a } )$ , we can interpret the dissociative effects as direct effects of Z on Y that act independent of M, although we cannot simply interpret the associative effects as direct or indirect effects of Z on Y .

Example 26.1 (noncompliance) With noncompliance, (26.1) consists of the average causal effects for the always takers, compliers, $d e f i e r s ,$ and never takers (Imbens and Angrist, 1994; Angrist et al., 1996).

Example 26.2 (truncation by death) Because the outcome is well define only if the patient survives, three subgroup causal effects in (26.1) are not meaningful, and the only well-defined subgroup effect is

$$
\tau (1, 1) = E \{Y (1) - Y (0) \mid M (1) = 1, M (0) = 1 \}. \tag {26.2}
$$

It is called the survivor average causal effect (Rubin, 2006a). It is the average causal effect of the treatment on the outcome for those units who survive regardless of the treatment status.

Example 26.3 (unemployment) The unemployment problem is isomorphic to the truncation by death problem because the wage is well-defined only if the unit is employed in the first place. Therefore, the only well defined subgroup effect is (26.2), the employed average causal effect. Previously, Heckman (1979) proposed a model, now called the Heckman Selection Model, to deal with unemployment in modeling the wage, viewing the wages of those unemployed as missing values2. However, Zhang and Rubin (2003) and Zhang et al. (2009) argued that τ (1, 1) is a more meaningful quantity under the potential outcomes framework.

Example 26.4 (surrogate endpoint) Intuitively, we want to assess the effect of the treatment on the outcome via the effect of the treatment on the surrogate endpoint. Therefore, a good surrogate endpoint should satisfy two conditions: first, if the treatment does not affect the surrogate, then it does not affect the outcome either; second, if the treatment affects the surrogate, then it affects the outcome too. The first condition is called the “causal necessity” by Frangakis and Rubin (2002), and the second condition is called the “causal sufficiency” by Gilbert and Hudgens (2008). Based on (26.1) for a binary surrogate endpoint, causal necessity requires that τ (1, 1) and τ (0, 0) are zero, and causal sufficiency requires that τ (1, 0) and τ (0, 1) are not zero.

## 26.4 Statistical Inference and Its Difficulty

In Example 26.1, if we have randomization, monotonicity and exclusion restriction, then we can identify the complier average causal effect. This is the key result derived in Chapter 21.

However, in other examples, we cannot impose the exclusion restriction assumption. For instance, τ (1, 1) is the main parameter of interest in Examples 26.2 and 26.3, and τ (1, 1) and τ (0, 0) are both of interest in Example 26.4.

Without the exclusion restriction assumption, it is very challenging to identify the principal stratification average causal effect. Sometimes, we cannot even impose the monotonicity assumption, and thus cannot identify the proportions of the latent strata in the first place.

## 26.4.1 Special case: truncation by death with binary outcome

I use the simple setting with a binary treatment, binary survival status and binary outcome to illustrate the idea and especially the difficulty of statistical inference based on principal stratification.

In addition to Assumption 26.1, we impose the monotonicity.

## Assumption 26.2 (monotonicity) $M ( 1 ) \geq M ( 0 )$ .

Theorem 22.1 demonstrates that under Assumptions 26.1 and 26.2, we can identify the proportions of the three latent strata by

$$
\begin{array}{l} \pi_ {(1, 1)} = \operatorname{pr} (M = 1 \mid Z = 0), \\ \pi_ {(0, 0)} = \operatorname{pr} (M = 0 \mid Z = 1), \\ \pi_ {(1, 0)} = \operatorname{pr} (M = 1 \mid Z = 1) - \operatorname{pr} (M = 1 \mid Z = 0). \\ \end{array}
$$

Our goal is to identify the survivor average causal effect $\tau ( 1 , 1 )$ . First, we can easily identify $E \{ Y ( 0 ) \mid M ( 1 ) = 1 , M ( 0 ) = 1 \}$ because the observed group $( Z = 0 , M = 1 )$ consists of only survivors:

$$
E \{Y (0) \mid M (1) = 1, M (0) = 1 \} = E (Y \mid Z = 0, M = 1).
$$

The key is then to identify $E \{ Y ( 1 ) \mid M ( 1 ) = 1 , M ( 0 ) = 1 \}$ . The observed group $( Z = 1 , M = 1 )$ is a mixture of two strata (1, 1) and (1, 0), therefore we have

$$
\begin{array}{l} E (Y \mid Z = 1, M = 1) = \frac {\pi_ {(1 , 1)}}{\pi_ {(1 , 1)} + \pi_ {(1 , 0)}} E \{Y (1) \mid M (1) = 1, M (0) = 1 \} \\ + \frac {\pi_ {(1 , 0)}}{\pi_ {(1 , 1)} + \pi_ {(1 , 0)}} E \{Y (1) \mid M (1) = 1, M (0) = 0 \}. \\ \end{array}
$$

We have two unknown parameters but only one equation. So we cannot uniquely determine $E \{ Y ( 1 ) \mid M ( 1 ) = 1 , M ( 0 ) = 1 \}$ from the above equation. Nevertheless, this equation contains some information about the quantity of interest. That is, $E \{ Y ( 1 ) \mid M ( 1 ) = 1 , M ( 0 ) = 1 \}$ is partially identified by Definition 18.1.

For a binary outcome Y , we know that $E \{ Y ( 1 ) \mid M ( 1 ) = 1 , M ( 0 ) = 0 \}$ is bounded between 0 and 1, and consequently, $E \{ Y ( 1 ) \mid M ( 1 ) = 1 , M ( 0 ) = 1 \}$ is bounded between the solutions to the following two equations:

$$
\begin{array}{l} E (Y \mid Z = 1, M = 1) = \frac {\pi_ {(1 , 1)}}{\pi_ {(1 , 1)} + \pi_ {(1 , 0)}} E \{Y (1) \mid M (1) = 1, M (0) = 1 \} \\ + \frac {\pi_ {(1 , 0)}}{\pi_ {(1 , 1)} + \pi_ {(1 , 0)}} \\ \end{array}
$$

and

$$
E (Y \mid Z = 1, M = 1) = \frac {\pi_ {(1 , 1)}}{\pi_ {(1 , 1)} + \pi_ {(1 , 0)}} E \{Y (1) \mid M (1) = 1, M (0) = 1 \}.
$$

Therefore, $E \{ Y ( 1 ) \mid M ( 1 ) = 1 , M ( 0 ) = 1 \}$ has lower bound

$$
\frac {\{\pi_ {(1 , 1)} + \pi_ {(1 , 0)} \} E (Y \mid Z = 1 , M = 1) - \pi_ {(1 , 0)}}{\pi_ {(1 , 1)}},
$$

and upper bound

$$
\frac {\{\pi_ {(1 , 1)} + \pi_ {(1 , 0)} \} E (Y \mid Z = 1 , M = 1)}{\pi_ {(1 , 1)}}.
$$

We can then derive the bounds $\mathrm { o n } \tau ( 1 , 1 )$ , summarized below.

Theorem 26.1 Under Assumptions 26.1 and 26.2 with a binary Y , we have

$$
\begin{array}{l} \frac {\left\{\pi_ {(1 , 1)} + \pi_ {(1 , 0)} \right\} E (Y \mid Z = 1 , M = 1) - \pi_ {(1 , 0)}}{\pi_ {(1 , 1)}} - E (Y \mid Z = 0, M = 1) \\ \leq \tau (1, 1) \\ \leq \frac {\{\pi_ {(1 , 1)} + \pi_ {(1 , 0)} \} E (Y \mid Z = 1 , M = 1)}{\pi_ {(1 , 1)}} - E (Y \mid Z = 0, M = 1). \\ \end{array}
$$

In most truncation by death problems, the lower and upper bounds are quite different, and they are bounded away from the extreme values −1 and 1. So we can use Imbens and Manski (2004)’s confidence interval for τ (1, 1) which involves two steps: first, we obtain the estimated lower and upper bounds [ˆl, uˆ] with estimated standard errors $( \mathrm { s e } _ { l } , \mathrm { s e } _ { u } ) ;$ second, we construct the confidence interval as $[ \hat { l } - z _ { \alpha } \mathrm { s e } _ { l } , \hat { u } + z _ { \alpha } \mathrm { s e } _ { u } ]$ , where $z _ { \alpha }$ is the 1 − α quantile of the standard normal distribution.

To summarize, this is a challenging problem since we cannot identify the parameter based on the observed data even with infinite sample size. We can derive large-sample bounds for τ (1, 1) but the statistical inference based on the bounds are not standard. If we do not have monotonicity, the large-sample bounds have even more complex forms (Zhang and Rubin, 2003; Jiang et al., 2016).

## 26.4.2 An application

I use the data in Yang and Small (2016) from the Acute Respiratory Distress Syndrome Network study involving 861 patients with lung injury and acute respiratory distress syndrome. Patients were randomized to receive mechanical ventilation with either lower tidal volumes or traditional tidal volumes. The outcome is the binary indicator for whether patients could breathe without assistance by day 28. Table 26.1 summarizes the observed data.

**TABLE 26.1: Data truncated by death with \* indicating the outcomes for dead patients**

<table><tr><td colspan="4">Treatment Z = 1</td><td colspan="4">Control Z = 0</td></tr><tr><td></td><td>Y = 1</td><td>Y = 0</td><td>total</td><td></td><td>Y = 1</td><td>Y = 0</td><td>total</td></tr><tr><td>M = 1</td><td>54</td><td>268</td><td>322</td><td>M = 1</td><td>59</td><td>218</td><td>277</td></tr><tr><td>M = 0</td><td>*</td><td>*</td><td>109</td><td>M = 0</td><td>*</td><td>*</td><td>152</td></tr></table>

We first obtain the point estimators of the latent strata:

$$
\hat {\pi} _ {(1, 1)} = \frac {2 7 7}{2 7 7 + 1 5 2} = 0. 6 4 6, \quad \hat {\pi} _ {(0, 0)} = \frac {1 0 9}{1 0 9 + 3 2 2} = 0. 2 5 3, \quad \hat {\pi} _ {(1, 0)} = 0. 1 0 1.
$$

The sample means of the outcome for survived patients are

$$
\hat {E} (Y \mid Z = 1, M = 1) = \frac {5 4}{3 0 2} = 0. 1 6 8, \quad \hat {E} (Y \mid Z = 0, M = 1) = \frac {5 9}{2 7 7} = 0. 2 1 3.
$$

The estimates for the bounds on $E \{ Y ( 1 ) \mid M ( 1 ) = 1 , M ( 0 ) = 1 \}$ are

$$
\left[ \frac {(0 . 6 4 6 + 0 . 1 0 1) \times 0 . 1 6 8 - 0 . 1 0 1}{0 . 1 0 1}, \frac {(0 . 6 4 6 + 0 . 1 0 1) \times 0 . 1 6 8}{0 . 1 0 1} \right] = [ 0. 0 3 7, 0. 1 9 4 ],
$$

so the bounds on τ (1, 1) are

$$
[ 0. 0 3 7 - 0. 2 1 3, 0. 1 9 4 - 0. 2 1 3 ] = [ - 0. 1 7 6, - 0. 0 1 9 ].
$$

Incorporating the sampling uncertainty based on the bootstrap, the upper bound becomes positive.

## 26.4.3 Extensions

Zhang and Rubin (2003) started the literature of large-sample bounds. Imai (2008a) and Lee (2009) were two follow-up papers. Cheng and Small (2006) derived the bounds with multiple treatment arms. Yang and Small (2016) used a secondary outcome to sharpen the bounds on the survivor average causal effect.

## 26.5 Principal score method

Without additional assumptions, we can only derive bounds on the causal effects within principal strata, but cannot identify them in general. We must impose additional assumptions to achieve nonparametric identification of the $\tau ( m _ { 1 } , m _ { 0 } ) _ { \mathrm { { s } } }$ . There is no consensus on the choice of the assumptions. Those additional assumptions are not testable, and their plausibility depends on the application. A line of research parallels causal inference with unconfounded observational studies. For simplicity, I focus on the case with strong monotonicity.

## 26.5.1 Principal score method under strong monotonicity

Assumption 26.3 (strong monotonicity) $M ( 0 ) = 0$ .

Similar to the ignorability assumption, we now assume the principal ignorability assumption.

Assumption 26.4 (principal ignorability) $E \{ Y ( 0 ) ~ \mid ~ M ( 1 ) ~ = ~ 1 , X \} ~ =$ $E \{ Y ( 0 ) \mid M ( 1 ) = 0 , \dot { X } \}$ .

These assumptions ensures nonparametric identification of the causal effects within principal strata.

Theorem 26.2 Under Assumptions 26.1, 26.3 and ${ \it 2 6 . 4 } ,$ the principal stratification average causal effects can be identified by

$$
\tau (1, 0) = E (Y \mid Z = 1, M = 1) - E \{\pi (X) Y \mid Z = 0 \} / \pi
$$

and

$$
\tau (0, 0) = E (Y \mid Z = 1, M = 0) - E \{(1 - \pi (X) \} Y \mid Z = 0 \} / (1 - \pi)
$$

where $\pi ( X ) = \operatorname { p r } \{ M ( 1 ) = 1 \mid X \}$ and $\pi = \mathrm { p r } \{ M ( 1 ) = 1 \}$ can be identified by

$$
\pi (X) = \operatorname{pr} (M = 1 \mid Z = 1, X)
$$

and

$$
\pi = \operatorname{pr} (M = 1 \mid Z = 1).
$$

The conditional probability $\pi ( X ) = \operatorname { p r } \{ M ( 1 ) = 1 \mid X \}$ is called the $p r i n –$ cipal score. Theorem 26.2 states that $\tau ( 1 , 0 )$ and $\tau ( 0 , 0 )$ can be identified by difference in means with appropriate weights depending on the principal score.

Proof of Theorem 26.2: I will only prove that

$$
E \{Y (0) \mid M (1) = 1 \} = E \{\pi (X) Y \mid Z = 0 \} / \pi .
$$

The left-hand side equals

$$
\begin{array}{l} E \{M (1) Y (0) \} / \pi = E [ E \{M (1) \mid X \} E \{Y (0) \mid X \} ] / \pi \\ = E \left[ \pi (X) E \{Y (0) \mid X \} \right] / \pi \\ = E \left[ E \{\pi (X) Y (0) \mid X \} \right] / \pi \\ = E \{\pi (X) Y (0) \} / \pi \\ = E \{\pi (X) Y \mid Z = 0 \} / \pi . \\ \end{array}
$$

Theorem 26.2 motivates the following simple estimators for τ (1, 0) and τ (0, 0), respectively:

1. fit a logistic regression of M on X using only data from the treated group to obtain ˆπ(Xi);  
2. estimate π by $\textstyle { \hat { \pi } } = \sum _ { i = 1 } ^ { n } Z _ { i } M _ { i } / \sum _ { i = 1 } ^ { n } Z _ { i }$  
3. obtain moment estimators:

$$
\hat {\tau} (1, 0) = \frac {\sum_ {i = 1} ^ {n} Z _ {i} M _ {i} Y _ {i}}{\sum_ {i = 1} ^ {n} Z _ {i} M _ {i}} - \frac {\sum_ {i = 1} ^ {n} (1 - Z _ {i}) \hat {\pi} (X _ {i}) Y _ {i}}{\hat {\pi} \sum_ {i = 1} ^ {n} (1 - Z _ {i})}
$$

and

$$
\hat {\tau} (0, 0) = \frac {\sum_ {i = 1} ^ {n} Z _ {i} (1 - M _ {i}) Y _ {i}}{\sum_ {i = 1} ^ {n} Z _ {i} (1 - M _ {i})} - \frac {\sum_ {i = 1} ^ {n} (1 - Z _ {i}) (1 - \hat {\pi} (X _ {i}) \} Y _ {i}}{(1 - \hat {\pi}) \sum_ {i = 1} ^ {n} (1 - Z _ {i})};
$$

4. use the bootstrap to approximate the variances of ˆτ (1, 0) and τˆ(0, 0).

## 26.5.2 Extensions

Follmann (2000), Hill et al. (2002), Jo and Stuart (2009), Jo et al. (2011) and Stuart and Jo (2015) started the literature of using the principal score to identify causal effects within principal strata. Ding and Lu (2017) provided theoretical foundation for this strategy. They prove Theorem 26.2 as well as a more general version under monotonicity; see Problem 26.1. Jiang et al. (2022) give a unified discussion of this strategy for observational studies and propose multiply robust estimators for causal effects within principal strata.

## 26.6 Other methods

To estimate principal stratification average causal effects without the exclusion restriction assumption, Zhang et al. (2009) proposed to use the normal mixture models. However, the inference based on the normal mixture models can be quite fragile. A strategy is to use additional information to improve the inference under some restrictions (Ding et al., 2011; Mealli and Pacini, 2013; Mattei et al., 2013; Jiang et al., 2016).

Conceptually, the principal stratification framework works for general M. A multi-valued M generates many latent principal strata, and a continuous M generates infinitely many latent principal strata. In those cases, identifying the probability of the principal strata is non-trivial in the first place let alone identifying the principal stratification average causal effects. Jiang and Ding (2021) reviewed some useful strategies.

## 26.7 Homework problems

## 26.1 Principal score method under monotonicity

This problem extends Theorem 26.2, with Assumption 26.3 replaced by Assumption 26.2 and Assumption 26.4 replaced by the assumption below.

Assumption 26.5 (principal ignorability) We have

$$
E \{Y (1) \mid M (1) = 1, M (0) = 0, X \} = E \{Y (1) \mid M (1) = 1, M (0) = 1, X \}
$$

and

$$
E \{Y (0) \mid M (1) = 1, M (0) = 0, X \} = E \{Y (0) \mid M (1) = 0, M (0) = 0, X \}.
$$

Theorem 26.3 Under Assumptions 26.1, 26.2 and 26.5, the principal stratification average causal effects can be identified by

$$
\tau (1, 0) = E \left\{w _ {1, (1, 0)} (X) Y \mid Z = 1, M = 1 \right\} - E \left\{w _ {0, (1, 0)} (X) Y \mid Z = 0, M = 0 \right\},
$$

$$
\tau (0, 0) = E (Y \mid Z = 1, M = 0) - E \left\{w _ {0, (0, 0)} (X) Y \mid Z = 0, M = 0 \right\},
$$

$$
\tau (1, 1) = E \left\{w _ {1, (1, 1)} (X) Y \mid Z = 1, M = 1 \right\} - E (Y \mid Z = 0, M = 1)
$$

with

$$
w _ {1, (1, 0)} (X) = \frac {\pi_ {(1 , 0)} (X)}{\pi_ {(1 , 0)} (X) + \pi_ {(1 , 1)} (X)} \Big / \frac {\pi_ {(1 , 0)}}{\pi_ {(1 , 0)} + \pi_ {(1 , 1)}},
$$

$$
w _ {0, (1, 0)} (X) = \frac {\pi_ {(1 , 0)} (X)}{\pi_ {(1 , 0)} (X) + \pi_ {(0 , 0)} (X)} \Big / \frac {\pi_ {(1 , 0)}}{\pi_ {(1 , 0)} + \pi_ {(0 , 0)}},
$$

$$
w _ {0, (0, 0)} (X) = \frac {\pi_ {(0 , 0)} (X)}{\pi_ {(1 , 0)} (X) + \pi_ {(0 , 0)} (X)} \Big / \frac {\pi_ {(0 , 0)}}{\pi_ {(1 , 0)} + \pi_ {(0 , 0)}},
$$

$$
w _ {1, (1, 1)} (X) = \frac {\pi_ {(1 , 1)} (X)}{\pi_ {(1 , 0)} (X) + \pi_ {(1 , 1)} (X)} \Big / \frac {\pi_ {(1 , 1)}}{\pi_ {(1 , 0)} + \pi_ {(1 , 1)}}.
$$

Moreover, the conditional and marginal principal scores are all identifiable by

$$
\pi_ {(0, 0)} (X) = \operatorname{pr} (M = 0 \mid Z = 1, X),
$$

$$
\pi_ {(1, 1)} (X) = \operatorname{pr} (M = 1 \mid Z = 0, X),
$$

$$
\pi_ {(1, 0)} (X) = \operatorname{pr} (M = 1 \mid Z = 1, X) - \operatorname{pr} (M = 1 \mid Z = 0, X).
$$

and

$$
\pi_ {(0, 0)} = \operatorname{pr} (M = 0 \mid Z = 1),
$$

$$
\pi_ {(1, 1)} = \operatorname{pr} (M = 1 \mid Z = 0),
$$

$$
\pi_ {(1, 0)} = \operatorname{pr} (M = 1 \mid Z = 1) - \operatorname{pr} (M = 1 \mid Z = 0).
$$

Remark: Based on Theorem 26.3, we can construct weighting estimators. Theorem 26.3 is Proposition 2 in Ding and Lu (2017), which also provided more details for the estimation.

## 26.2 Recommended reading

Frangakis and Rubin (2002) proposed the principal stratification framework. Zhang and Rubin (2003) derived large-sample bounds on the survivor average causal effect. Jiang and Ding (2021) reviewed various strategies to identify the causal effects within principal strata.

# Mediation Analysis: Natural Direct and Indirect Effects

With an intermediate variable M between the treatment $Z$ and outcome $Y .$ , the causal effects within principal strata defined by $U = \{ M ( 1 ) , M ( 0 ) \}$ can assess the treatment effect heterogeneity across latent groups U. When M is indeed on the causal pathway from $Z$ to $Y$ , causal effects within some principal strata, $\tau ( 1 , 1 )$ and $\tau ( 0 , 0 )$ , can give information about the direct effect of $Z$ on $Y ,$ . However, these direct effects are only for two latent groups. The causal effects within the other two principal strata, $\tau ( 1 , 0 )$ and $\tau ( 0 , 1 )$ , contain both the direct and indirect effects. Fundamentally, principal stratification does not provide any information about the indirect effect of $Z$ on $Y$ through M because it does not even assume that M can be intervened.

In the above discussion, I use the notions of “direct effect” and “indirect effect” in a casual way. When M lies on the pathway from $Z$ to $Y ,$ , researchers often want to assess the extent to which the effect of $Z$ on Y is through M and the extent to which the effect is through other pathways. This is called mediation analysis. It is the topic of this chapter.

## 27.1 Motivating Examples

In mediation analysis, we have a treatment $Z ,$ an outcome $Y ,$ a mediator M, and some background covariates X. Figure 27.3 illustrates their relationship. Below we give some concrete examples.

![image_32](images/image_32.png)

```mermaid
graph TD
  X --> Y
  X --> Z
  Z --> M
  M --> Y
```

FIGURE 27.1: Directed acyclic graph for mediation

Example 27.1 VanderWeele et al. (2012) conducted mediation analysis to assess the extent to which the effect of variants on chromosome 15q25.1 on lung cancer is mediated through smoking and to which it operates through other causal pathways. The exposure levels correspond to changes from 0 to 2 C alleles, smoking intensity is measured by the square root of cigarettes per day, and the outcome is the lung cancer indicator. VanderWeele et al. (2012)’s study contained many sociodemographic covariates.

Example 27.2 Rudolph et al. (2018) studies the causal mechanism from neighborhood poverty to adolescent substance use, mediated by the school and peer environment. They used data from the National Comorbidity Survey Replication Adolescent Supplement, a nationally representative survey of U.S. adolescents conducted during 2001–2004. The treatment is the binary indicator of neighborhood disadvantage, defined as living in the lowest tertile of neighborhood socioeconomic status based on data from the 2000 U.S. Census. Four binary mediators are measures of school and peer environments, and six binary outcomes are measures of substance use. Baseline covariates included the adolescent’s sex, age, race, immigration generation, family income, etc.

Example 27.3 The mediation package in R contains a dataset called jobs, which is from JOBS II, a randomized field experiment that investigates the efficacy of a job training intervention on unemployed workers. We used this dataset in Chapter 21.5. The program is designed to not only increase reemployment among the unemployed but also enhance the mental health of the job seekers. It is therefore of interest to assess the indirect effect of the intervention on the mental health through job search efficacy and its direct effect acting through other pathways. We will revisit this example later.

## 27.2 Nested Potential Outcomes

## 27.2.1 Natural Direct and Indirect Effects

Below we drop the index i for unit i and assume all random variables are iid draws from a super population. For simplicity, we focus on a binary treatment Z .

We first consider the hypothetical intervention on z and define potential mediators and outcomes corresponding to the intervention on z:

$$
\{M (z), Y (z): z = 0, 1 \}.
$$

We then consider hypothetical intervention on both z and m and define potential outcomes corresponding to the interventions on z and m:

$$
\{Y (z, m): z = 0, 1; m \in \mathcal {M} \},
$$

where M contains all possible values of $m .$ . Robins and Greenland (1992) and Pearl (2001) further consider the nested potential outcomes corresponding to intervention on z and $m = M ( z ^ { \prime } ) \equiv M _ { z ^ { \prime } }$ :

$$
\left\{Y (z, M _ {z ^ {\prime}}): z = 0, 1; z ^ {\prime} = 0, 1 \right\}
$$

where we write $M ( z ^ { \prime } )$ as $M _ { z ^ { \prime } }$ to avoid excessive parentheses. The notation $Y ( z , M _ { z ^ { \prime } } )$ is the hypothetical outcome if the treatment were set at level z and the mediator were set at its potential level $M ( z ^ { \prime } )$ under treatment $z ^ { \prime } .$ . Importantly, z and $z ^ { \prime }$ can be different. With a binary treatment, we have four nested potential outcomes in total:

$$
\{Y (1, M _ {1}), Y (1, M _ {0}), Y (0, M _ {1}), Y (0, M _ {0}) \}.
$$

The nested potential outcome $Y ( 1 , M _ { 1 } )$ is the hypothetical outcome if the treatment were set at $z = 1$ and the mediator were set at what would happen under $z = 1$ . Similarly, $Y ( 0 , M _ { 0 } )$ is the outcome if the treatment were set at $z = 0$ and the mediator were set at what would happen under $z = 0$ . It would be surprising if $Y ( 1 , M _ { 1 } ) \neq Y ( 1 )$ or $Y ( 0 , M _ { 0 } ) \neq Y ( 0 )$ . Therefore, we make the following assumption throughout this chapter.

Assumption 27.1 (composition) $Y ( z , M _ { z } ) = Y ( z ) ~ f o r ~ z = 0 , 1$ .

The composition assumption cannot be proved. It is indeed an assumption. Without causing philosophical debates, we can even define $Y ( 1 )$ as $Y ( 1 , M _ { 1 } )$ , and define $Y ( 0 )$ as $Y ( 0 , M _ { 0 } )$ .

The nested potential outcome $Y ( 1 , M _ { 0 } )$ is the hypothetical outcome if the unit received treatment 1 but its mediator were set at its natural value $M _ { 0 }$ without the treatment. Similarly, $Y ( 0 , M _ { 1 } )$ is the hypothetical outcome if the unit received control 0 but its mediator were set at its natural value $M _ { 1 }$ under the treatment. They are two cross-world counterfactual terms and useful for defining the direct and indirect effects.

Definition 27.1 (total, direct and indirect effects) Define the total $e f -$ fect of the treatment on the outcome as

$$
\tau = E \{Y (1) - Y (0) \}.
$$

Define the natural direct effect as

$$
\mathrm{NDE} = E \left\{Y \left(1, M _ {0}\right) - Y \left(0, M _ {0}\right) \right\}.
$$

Define the natural indirect effect as

$$
\mathrm{NIE} = E \{Y (1, M _ {1}) - Y (1, M _ {0}) \}.
$$

The total effect is the standard average causal effect of $Z$ on $Y$ . The natural direct effect measures the effect of the treatment on the outcome if the mediator were set at the natural value $M _ { 0 }$ without the intervention. The natural indirect effect measures the the effect of the treatment through changing the mediator if the treatment itself were set at $z = 1$ . Under the composition assumption, the natural direct and indirect effects reduce to

$$
\mathrm{NDE} = E \{Y (1, M _ {0}) - Y (0) \}, \quad \mathrm{NIE} = E \{Y (1) - Y (1, M _ {0}) \},
$$

and therefore, we can decompose the total effect as the sum of the natural direct and indirect effects.

Proposition 27.1 $B y$ Definition 27.1 and Assumption 27. $1 , \tau = \mathrm { N D E + N I E }$ .

Mathematically, we can also define the natural indirect effect as $E \{ Y ( 0 , M _ { 1 } ) - Y ( 0 , M _ { 0 } ) \}$ where the treatment is fixed at 0. However, this definition does not lead to the decomposition in Proposition 27.1.

Unfortunately, the nest potential outcome $Y ( 1 , M _ { 0 } )$ is not an easy quantity to understand due to the cross-world nature of the interventions: the treatment is set at $z = 1$ but the mediator is set at its natural value $M _ { 0 }$ under treatment $z = 0$ . Clearly, these two interventions on the treatment cannot simultaneously happen in any realized experiment. To understand the cross-world potential outcome $Y ( 1 , M _ { 0 } )$ , we need to imagine the existence of parallel worlds as shown in Figure 27.2. Let’s focus on $Y ( 1 , M _ { 0 } )$ . When the treatment is set at $z = 1$ , the mediator must take value $M _ { 1 }$ . If at the same time we want to set the mediator at $m = M _ { 0 }$ , we must know the value of $M _ { 0 }$ for the same unit from another experiment in the parallel world. This can be an unrealistic physical experiment because it requires that the same unit is intervened at two different levels of the treatment. Under some strong assumptions about the homogeneity of units, we may use another unit’s mediator value under control as a proxy for $M _ { 0 }$ .

## 27.2.2 Metaphysics or Science

Causal inference is hard, and there is no agreement even on its mathematical notation. Robins and Greenland (1992) and Pearl (2001) used the nested potential outcomes to define the natural direct and indirect effects. However, Frangakis and Rubin (2002) called $Y ( 1 , M _ { 0 } )$ and $Y ( 0 , M _ { 1 } )$ a priori counterfactuals because we cannot observed them in any physical experiments. In this sense, they do not exist a priori. According to Popper (1963), a way to distinguish science and metaphysics is the falsifiability of the statements. That is, if a statement is not falsifiable based on any physical experiments or observations, then it is not a scientific but rather a metaphysical statement. Because we cannot observe $Y ( 1 , M _ { 0 } )$ and $Y ( 0 , M _ { 1 } )$ in any experiments, we cannot falsify any statements involving them except for the trivial ones $( \mathrm { e . g . }$ , some outcomes are binary, or continuous, or bounded). Therefore, a strict Popperian statistician would view mediation analysis as metaphysics.

More strikingly, Dawid (2000) criticized the potential outcomes framework to be metaphysical, and he called Rubin’s Science Table a “metaphysical ar-$\mathrm { r a y . } ^ { \mathrm { , y } }$ This is a critique on not only the a priori counterfactuals $Y ( 1 , M _ { 0 } )$ and $Y ( 0 , M _ { 1 } )$ but also the simple potential outcomes $Y ( 1 )$ and $Y ( 0 )$ . Dawid (2000) argued that because we can never observe $Y ( 1 )$ and $Y ( 0 )$ jointly, then introducing the notation $\{ Y ( 1 ) , Y ( 0 ) \}$ is a metaphysical activity. He is correct about the metaphysical nature of the joint distribution of $\mathrm { p r } \{ Y ( 1 ) , Y ( 0 ) \}$ , but he is incorrect about the marginal distributions. Based on the observed data, we indeed can falsify some statement about the marginal distributions, although we cannot falsify any statements about the joint distribution.1 Therefore, even according to Popper (1963), Rubin’s Science Table is not metaphysical because it has some nontrivial falsifiable implications although not all implications are falsifiable. This is the fundamental difference between $\{ Y ( 1 ) , Y ( 0 ) \}$ and $\{ Y ( 1 , M _ { 0 } ) , Y ( 0 , M _ { 1 } ) \}$ .

![image_33](images/image_33.png)

```mermaid
graph TD
  A["parallel worlds"] --> B["intervention z = 0"]
  A --> C["intervention z = 1"]
  B --> D["M₀\nY(0) = Y(0,M₀)"]
  B --> E["intervention m = M₁\nY(z,m) = Y(0,M₁)"]
  C --> F["M₁\nY(1) = Y(1,M₁)"]
  C --> G["intervention m = M₀\nY(z,m) = Y(1,M₀)"]
    D -.-> H["world 0'"]
    E -.-> I["world 0'"]
    F -.-> J["world 1"]
    G -.-> K["world 1'"]
    H -.-> L["cross-world communications"]
    I -.-> L
    J -.-> L
    K -.-> L
```

FIGURE 27.2: Crossworld potential outcomes $Y ( 1 , M _ { 0 } )$ and $Y ( 0 , M _ { 1 } )$

$$
\max \{0, \operatorname{pr} (Y (1) \leq y _ {1}) + \operatorname{pr} (Y (0) \leq y _ {0}) - 1 \}
$$

$$
\leq \operatorname{pr} (Y (1) \leq y _ {1}, Y (0) \leq y _ {0})
$$

$$
\leq \min \{\operatorname{pr} (Y (1) \leq y _ {1}), \operatorname{pr} (Y (0) \leq y _ {0}) \}.
$$

This is often a loose inequality. Unfortunately, we do not have any information beyond this inequality without imposing additional assumptions.

## 27.3 The Mediation Formula

Pearl (2001)’s mediation formula relies on the following four assumptions. The first three essentially assumes that the treatment and the mediator are both randomized conditional on observed covariates.

Assumption 27.2 There is no treatment-outcome confounding:

$$
Z \bot Y (z, m) \mid X
$$

for all z and m.

Assumption 27.3 There is no mediator-outcome confounding:

$$
M \bot Y (z, m) \mid (X, Z)
$$

for all z and m.

Assumptions 27.2 and 27.3 together are often called sequential ignorability. They are equivalent to the assumption that (Z, M) are jointly randomized conditioning on X:

$$
(Z, M) \perp Y (z, m) \mid X \tag {27.1}
$$

for all z and m. I leave the proof as Problem 27.1.

Assumption 27.4 There is no treatment-mediator confounding:

$$
Z \bot M (z) \mid X
$$

for all z.

The last assumption is the cross-world independence.

Assumption 27.5 There is no cross-world independence between the potential outcomes and potential mediators:

$$
Y (z, m) \perp M (z ^ {\prime}) \mid X
$$

for all $z , z ^ { \prime }$ and m.

Assumptions 27.2–27.4 are very strong, but at least they hold under experiments with randomized treatment and mediator. Assumption 27.5 is stronger because no physical experiment can ensure it. Because we can never observe $Y ( z , m )$ and $M ( z ^ { \prime } )$ in any experiment $\mathrm { i f } \ z \ne z ^ { \prime } ,$ Assumption 27.5 can never be validated so it is fundamentally meta-physical.

I give an example below in which Assumptions 27.2–27.5 all hold.

Example 27.4 Given X, we generate

$$
Z = 1 \{f _ {Z} (X, \varepsilon_ {Z}) \},
$$

$$
M (z) = 1 \{f _ {M} (X, z, \varepsilon_ {M}) \},
$$

$$
Y (z, m) = f _ {Y} (X, z, m, \varepsilon_ {Y}),
$$

for $z , m = 0 , 1$ , where $\varepsilon _ { Z } , \varepsilon _ { M } , \varepsilon _ { Y }$ are all independent. Consequently, we generate the observed values of M and Y from

$$
M = M (Z) = 1 \{f _ {M} (X, Z, \varepsilon_ {M}) \},
$$

$$
Y = Y (Z, M) = f _ {Y} (X, Z, M, \varepsilon_ {Y}).
$$

We can verify that Assumptions 27.2–27.5 hold under this data generating process; see Problem 27.2.

Pearl (2001) proved the following key result for mediation analysis.

Theorem 27.1 Under Assumptions $\mathcal { Q } \Upsilon . \mathcal { Q } \ – \mathcal { Q } \ 7 . 5 ,$ we have

$$
E \{Y (z, M _ {z ^ {\prime}}) \mid X = x \} = \sum_ {m} E (Y \mid Z = z, M = m, X = x) \mathrm{pr} (M = m \mid Z = z ^ {\prime}, X = x)
$$

and therefore,

$$
E \{Y (z, M _ {z ^ {\prime}}) \} = \sum_ {x} E \{Y (z, M _ {z ^ {\prime}}) \mid X = x \} \mathrm{pr} (X = x).
$$

Theorem 27.1 assumes that both M and X are discrete. With general M and X, the mediation formulas become

$$
E \{Y (z, M _ {z ^ {\prime}}) \mid X = x \} = \int E (Y \mid Z = z, M = m, X = x) f _ {M} (m \mid Z = z ^ {\prime}, X = x) \mathrm{d} m
$$

and

$$
E \{Y (z, M _ {z ^ {\prime}}) \} = \int E \{Y (z, M _ {z ^ {\prime}}) \mid X = x \} f _ {X} (x) \mathrm{d} x.
$$

From Theorem 27.1, the identification formulas for the means of the nested potential outcomes depend on the conditional mean of the outcome given the treatment, mediator, and covariates, as well as the conditional mean of the mediator given the treatment and covariates. We need to evaluate these two conditional means at different treatment levels if the nested potential outcome involves cross-world interventions.

If we drop the cross-world independence assumption, we can modify the definition of the natural direct and indirect effects and the same formulas hold. See Problem 27.8 for more details.

I give the proof below.

Proof of Theorem 27.1: By the tower property, $\begin{array} { r l } { E \{ Y ( z , M _ { z ^ { \prime } } ) \} } & { { } = } \end{array}$$E [ E \{ Y ( z , M _ { z ^ { \prime } } ) \mid X \} ]$ ], so we need only to prove the formula for $E \{ Y ( z , M _ { z ^ { \prime } } ) \mid$ | $X = x \}$ . Starting with the law of total probability, we have

$$
\begin{array}{l} E \{Y (z, M _ {z ^ {\prime}}) \mid X = x \} \\ = \sum_ {m} E \left\{Y \left(z, M _ {z ^ {\prime}}\right) \mid M _ {z ^ {\prime}} = m, X = x \right\} \operatorname * {p r} \left(M _ {z ^ {\prime}} = m \mid X = x\right) \\ = \sum_ {m} E \{Y (z, m) \mid M _ {z ^ {\prime}} = m, X = x \} \mathrm{pr} (M _ {z ^ {\prime}} = m \mid X = x) \\ = \sum_ {m} \underbrace {E \{Y (z , m) \mid X = x \}} _ {\text {Assumption 27.5}} \underbrace {\operatorname{pr} (M = m \mid Z = z ^ {\prime} , X = x)} _ {\text {Assumption 27.4}} \\ = \sum_ {m} \underbrace {E (Y \mid Z = z , M = m , X = x)} _ {\text {Assumptions 27.2 and 27.3}} \operatorname{pr} (M = m \mid Z = z ^ {\prime}, X = x). \\ \end{array}
$$

![image_34](images/image_34.png)

The above proof is actually trivial from a mathematical perspective. It illustrates the necessity of Assumptions 27.2–27.5.

Conditioning on $X = x$ , the mediation formulas for $Y ( 1 , M _ { 1 } )$ and $Y ( 0 , M _ { 0 } )$ simplifies to

$$
\begin{array}{l} E \{Y (1, M _ {1}) \mid X = x \} \\ = \sum_ {m} E (Y \mid Z = 1, M = m, X = x) \operatorname{pr} (M = m \mid Z = 1, X = x) \\ = E (Y \mid Z = 1, X = x) \\ \end{array}
$$

and

$$
\begin{array}{l} E \{Y (0, M _ {0}) \mid X = x \} \\ = \sum_ {m} E (Y \mid Z = 0, M = m, X = x) \operatorname{pr} (M = m \mid Z = 0, X = x) \\ = E (Y \mid Z = 0, X = x) \\ \end{array}
$$

based on the law of total probability; the mediation formula for $Y ( 1 , M _ { 0 } )$ simplifies to

$$
E \{Y (1, M _ {0}) \mid X = x \} = \sum_ {m} E (Y \mid Z = 1, M = m, X = x) \mathrm{pr} (M = m \mid Z = 0, X = x),
$$

where the conditional expectation of the outcome is given $Z = 1$ but the conditional distribution of the mediator is given $Z = 0$ . This leads to the indentification formulas of the natural direct and indirect effects.

Corollary 27.1 Under Assumptions 27.2–27.5, the conditional natural direct and indirect effects are identified by

$$
\begin{array}{l} \mathrm{NDE} (x) = E \left\{Y \left(1, M _ {0}\right) - Y \left(0, M _ {0}\right) \mid X = x \right\} \\ = \sum_ {m} \left\{E (Y \mid Z = 1, M = m, X = x) - E (Y \mid Z = 0, M = m, X = x) \right\} \\ \times \operatorname{pr} (M = m \mid Z = 0, X = x) \\ \end{array}
$$

and

$$
\begin{array}{l} \operatorname{NIE} (x) = E \left\{Y \left(1, M _ {1}\right) - Y \left(1, M _ {0}\right) \mid X = x \right\} \\ = \sum_ {m} E (Y \mid Z = 1, M = m, X = x) \\ \times \{\operatorname{pr} (M = m \mid Z = 1, X = x) - \operatorname{pr} (M = m \mid Z = 0, X = x) \}; \\ \end{array}
$$

the unconditional ones can be identified by $\begin{array} { r } { \mathrm { N D E } = \sum _ { x } \mathrm { N D E } ( x ) \mathrm { p r } ( X = x ) } \end{array}$ and $\begin{array} { r } { \mathrm { N I E } = \sum _ { x } \mathrm { N I E } ( x ) \mathrm { p r } ( X = x ) } \end{array}$ .

As a special case, with a binary M, the formula of the nie reduces to a product form below.

Corollary 27.2 Under Assumptions 27.2–27.5, for a binary mediator M, we have

$$
\operatorname{NIE} (x) = \tau_ {Z \to M} (x) \tau_ {M \to Y} (1, x)
$$

and nie = E{nie(X)}, where

$$
\tau_ {Z \rightarrow M} (x) = \operatorname{pr} (M = 1 \mid Z = 1, X = x) - \operatorname{pr} (M = 1 \mid Z = 0, X = x).
$$

and

$$
\tau_ {M \rightarrow Y} (z, x) = E (Y \mid Z = z, M = 1, X = x) - E (Y \mid Z = z, M = 0, X = x)
$$

I leave the proof of Corollary 27.2 as Problem 27.4. Corollary 27.2 gives a simple formula in the case of a binary M. With randomized Z conditional on X, we can view $\tau _ { Z  M } ( x )$ as the conditional average causal effect of Z on M. With randomized M conditional on $( X , Z )$ , we can view $\tau _ { M  Y } ( z , x )$ as the conditional average causal effect of M on Y . The conditional natural indirect effect equals their product. This is coherent with our intuition that the indirect effect acts from Z to M and then from M to Y .

## 27.4 The Mediation Formula Under Linear Models

Theorem 27.1 gives the nonparametric identification formula for mediation analysis. It allows us to derive various formulas for mediation analysis under different models. I will introduce the famous Baron–Kenny method under linear models below. VanderWeele (2015) gives explicit formulas for the natural direct and indirect effects for many commonly-used models. I relegate the details of other models to Section 27.6.

![image_35](images/image_35.png)

```mermaid
graph TD
  X -->|β₂| M
  X -->|θ₄| Y
  Z -->|β₁| M
  Z -->|θ₁| Y
  M -->|θ₂| Y
```

FIGURE 27.3: The Baron–Kenny Method for mediation under linear models

indirect effect: $\beta _ { 1 } \theta _ { 2 }$

direct effect: $\theta _ { 1 }$

## 27.4.1 The Baron–Kenny Method

The Baron–Kenny method assumes the following linear models for the mediator and outcome given the treatment and covariates.

Assumption 27.6 (linear models for the Baron–Kenny method) Both the mediator and outcome follow linear models:

$$
\left\{ \begin{array}{r c l} E (M \mid Z, X) & = & \beta_ {0} + \beta_ {1} Z + \beta_ {2} ^ {\mathsf {T}} X, \\ E (Y \mid Z, M, X) & = & \theta_ {0} + \theta_ {1} Z + \theta_ {2} M + \theta_ {4} ^ {\mathsf {T}} X. \end{array} \right.
$$

Under these linear models, the formulas for the natural direct and indirect effects simplify to functions of the coefficients.

Corollary 27.3 (Baron–Kenny formulas for mediation) Under Assumptions 27.2–27.5 and 27.6,

$$
\mathrm{NDE} = \theta_ {1}, \quad \mathrm{NIE} = \theta_ {2} \beta_ {1}.
$$

Proof of Corollary 27.3: We have

$$
\mathrm{NDE} (x) = \sum_ {m} \theta_ {1} \mathrm{pr} (M = m \mid Z = 0, X = x) = \theta_ {1}
$$

and

$$
\begin{array}{l} \mathrm{NIE} (x) = \sum_ {m} (\theta_ {0} + \theta_ {1} + \theta_ {2} m + \theta_ {4} ^ {\mathsf {T}} x) \\ \times \left\{\operatorname{pr} (M = m \mid Z = 1, X = x) - \operatorname{pr} (M = m \mid Z = 0, X = x) \right\} \\ = \theta_ {2} \left\{E (M = m \mid Z = 1, X = x) - E (M = m \mid Z = 0, X = x) \right\} \\ = \theta_ {2} \beta_ {1}, \\ \end{array}
$$

<!-- footnote -->

- This can be tricky if the error term of the linear model is heteroskedastic. Without the independence of the $\dot { G } _ { j } { ' } { \bf s } .$ , it is hard to justify the independence.

<!-- footnote end -->

<!-- footnote -->

- Based on the causal diagrams, we can reach the same conclusion. In Figure $2 6 . 1 .$ , even though Z U by randomization of $Z ,$ conditioning on M introduces the “collider $\mathrm { b i a s } ^ { \prime \prime }$ that causes $z \not \bot \sqcup$ .

<!-- footnote end -->

<!-- footnote -->

- Heckman won nobel prize of economics in 2000 “for his development of theory and methods for analyzing selective samples.” His model contains two stages. First, the employment status is determined by a latent linear model
- $M _ { i } = 1 ( { X } _ { i } ^ { \mathsf { T } } \beta + u _ { i } \geq 0 ) .$
- Second, the latent log wage is determined by a linear model
- $Y _ { i } ^ { * } = W _ { i } ^ { \mathsf { T } } \gamma + v _ { i }$
- and $Y _ { i } ^ { * }$ is observed as $Y _ { i }$ only if $M _ { i } = 1$ . In his two-stage model, the covariates $X _ { i }$ and $W _ { i }$ may differ, and the errors $( u _ { i } , v _ { i } )$ are correlated bivariate Normal.

<!-- footnote end -->

<!-- footnote -->

- By the probability theory, given the marginal distributions of $\mathrm { p r } ( Y ( 1 ) ~ \leq ~ y _ { 1 } )$ and $\mathrm { p r } ( Y ( 0 ) \leq y _ { 0 } )$ , we can bound the joint distribution of p $\cdot ( Y ( 1 ) \ \leq \ y _ { 1 } , Y ( 0 ) \leq y _ { 0 } )$ by the Frechet–Hoeffding inequality:

<!-- footnote end -->

which do not depend on x. Therefore, they are also the formulas for the unconditional natural direct and indirect effects. □

If we obtain OLS estimators of these coefficients, we can estimate the direct and indirect effects by

$$
\mathrm{N} \hat {\mathrm{DE}} = \hat {\theta} _ {1}, \quad \mathrm{N} \hat {\mathrm{IE}} = \hat {\theta} _ {2} \hat {\beta} _ {1},
$$

which is called the Baron–Kenny method (Judd and Kenny, 1981; Baron and Kenny, 1986) although it had several antecedents (e.g., Hyman, 1955; Alwin and Hauser, 1975; Judd and Kenny, 1981; Sobel, 1982).

Standard software packages report the standard error of ndeˆ from OLS. Sobel (1982, 1986) used the delta method to obtain the standard error of nieˆ . Based on the formula in Example A1.2, the asymptotic variance of $\hat { \theta } _ { 2 } \hat { \beta } _ { 1 }$ equals va $\cdot ( \hat { \theta } _ { 2 } ) \beta _ { 1 } ^ { 2 } + \theta _ { 2 } ^ { 2 } \mathrm { v a r } ( \hat { \beta } _ { 1 } )$ . So the estimated variance is

$$
\hat {\mathrm{var}} (\hat {\theta} _ {2}) \hat {\beta} _ {1} ^ {2} + \hat {\theta} _ {2} ^ {2} \hat {\mathrm{var}} (\hat {\beta} _ {1}).
$$

Testing the null hypothesis of nie based on $\hat { \theta } _ { 2 } \hat { \beta } _ { 1 }$ and the estimated variance above is called Sobel’s test in the literature of mediation analysis.

## 27.4.2 An Example

We can easily implement the Baron–Kenny method via the following code.

```r
library("car")
BKmediation = function(Z, M, Y, X)
{
    ## two regressions and coefficients
    mediator.reg = lm(M ~ Z + X)
    mediator.Zcoef = mediator.reg$coef[2]
    mediator.Zse = sqrt(hccm(mediator.reg)[2, 2])

    outcome.reg = lm(Y ~ Z + M + X)
    outcome.Zcoef = outcome.reg$coef[2]
    outcome.Zse = sqrt(hccm(outcome.reg)[2, 2])
    outcome.Mcoef = outcome.reg$coef[3]
    outcome.Mse = sqrt(hccm(outcome.reg)[3, 3])

    ## Baron-Kenny point estimates
    NDE = outcome.Zcoef
    NIE = outcome.Mcoef*mediator.Zcoef

    ## Sobel's variance estimate based the delta method
    NDE.se = outcome.Zse
    NIE.se = sqrt(outcome.Mse^2*mediator.Zcoef^2 + outcome.Mcoef^2*mediator.Zse^2)

    res = matrix(c(NDE, NIE,
```

```txt
NDE.se, NIE.se,
NDE/NDE.se, NIE/NIE.se),
2, 3)
rownames(res) = c("NDE", "NIE")
colnames(res) = c("est", "se", "t")
res
}
```

Revisiting Example 27.3, we obtain the following estimates for the direct and indirect effects:

```txt
> library(mediation)
> Z = jobs$treat
> M = jobs$job_seek
> Y = jobs$depress2
> getX    = lm(treat ~ econ_hard + depress1 +
+    sex + age + occp + marital +
+    nonwhite + educ + income,
+    data = jobs)
> X = model.matrix(getX)[, -1]
> res = BKmediation(Z, M, Y, X)
> round(res, 3)
    est    se    t
NDE -0.037 0.042 -0.885
NIE -0.014 0.009 -1.528
```

Both the estimates for the direct and indirect effects are negative although they are insignificant.

## 27.5 Sensitivity analysis

Mediation analysis relies on strong and untestable assumptions. One crucial assumption is that there is no unmeasured confounding among the treatment, mediator and outcome. Various sensitivity analysis methods appeared in the literature. In particular, Ding and Vanderweele (2016) proposed Cornfieldtype sensitivity bounds and Zhang and Ding (2022) proposed a sensitivity analysis method tailored to the Baron–Kenny method based on linear structural equation models.

## 27.6 Homework problems

27.1 Sequential randomization and joint randomization

Show (27.1) is equivalent to Assumptions 27.2 and 27.3.

27.2 Verifying the assumptions for mediation analysis

Show that Assumptions 27.2–27.5 hold under the data generating process in Example 27.4.

27.3 Another set of assumptions for the mediation formula

Imai et al. (2010) invoked the following set of assumptions to derive the mediation formula.

## Assumption 27.7 Assume

$$
\{Y (z, m), M (z ^ {\prime}) \} \perp Z \mid X
$$

and

$$
Y (z, m) \perp M (z ^ {\prime}) \mid (Z = z ^ {\prime}, X)
$$

for all $z , z ^ { \prime } , m .$

Theorem 27.2 Under Assumption 27.7, the mediation formula holds.

Prove Theorem 27.2.

27.4 Natural indirect effect with a binary mediator

Prove Corollary 27.2.

27.5 With Treatment-Outcome Interaction on the Outcome

VanderWeele (2015) suggested using the following linear models:

$$
\left\{ \begin{array}{r c l} E (M \mid Z, X) & = & \beta_ {0} + \beta_ {1} Z + \beta_ {2} ^ {\mathsf {T}} X, \\ E (Y \mid Z, M, X) & = & \theta_ {0} + \theta_ {1} Z + \theta_ {2} M + \theta_ {3} Z M + \theta_ {4} ^ {\mathsf {T}} X, \end{array} \right.
$$

where the outcome model has the interaction term between the treatment and the mediator.

Under the above linear models, show that

$$
\mathrm{NDE} = \theta_ {1} + \theta_ {3} \{\beta_ {0} + \beta_ {2} ^ {\mathsf {T}} E (X) \}, \qquad \mathrm{NIE} = (\theta_ {2} + \theta_ {3}) \beta_ {1}.
$$

How do we estimate nde and nie with IID data?

Remark: Consider the simple case with a binary Z and binary M. Under the linear models, the average causal effect of Z of M equals $\beta _ { 1 }$ , and the average causal effect of M on $Y$ equals $\theta _ { 2 } + \theta _ { 3 } E ( Z )$ . Therefore, it is possible that both of these effects are positive, but the natural indirect effect is negative. For instance:

$$
\beta_ {1} = 1, \quad \theta_ {2} = 1, \quad \theta_ {3} = - 1. 5, \quad E (Z) = 0. 5.
$$

This is somewhat paradoxical, and can be called the mediator paradox. Chen et al. (2007) reported a related surrogate endpoint paradox or intermediate variable paradox.

## 27.6 Logistic Model for Binary Mediator

Consider the following Logistic model for the binary mediator and linear model for the outcome:

$$
\left\{ \begin{array}{r c l} \operatorname{logit} \{\operatorname{pr} (M = 1 \mid Z, X) \} & = & \beta_ {0} + \beta_ {1} Z + \beta_ {2} ^ {\mathsf {T}} X, \\ E (Y \mid Z, M, X) & = & \theta_ {0} + \theta_ {1} Z + \theta_ {2} M + \theta_ {4} ^ {\mathsf {T}} X, \end{array} \right.
$$

where lo $\mathrm { g i t } ( w ) = \log \{ w / ( 1 - w ) \}$ with inverse expi $: ( w ) = ( 1 + e ^ { - w } ) ^ { - 1 }$ .

Under these models, show that

$$
\mathrm{NDE} = \theta_ {1}, \quad \mathrm{NIE} = \theta_ {2} E \left\{\operatorname{expit} (\beta_ {0} + \beta_ {1} + \beta_ {2} ^ {\mathsf {T}} X) - \operatorname{expit} (\beta_ {0} + \beta_ {2} ^ {\mathsf {T}} X) \right\}.
$$

How do we estimate nde and nie with IID data?

## 27.7 Mediation analysis with binary mediator and outcome

Consider the following Logistic models for the binary mediator and outcome:

$$
\left\{ \begin{array}{r c l} \operatorname{logit} \{\operatorname{pr} (M = 1 \mid Z, X) \} & = & \beta_ {0} + \beta_ {1} Z + \beta_ {2} ^ {\mathsf {T}} X, \\ \operatorname{logit} \{\operatorname{pr} (Y = 1 \mid Z, M, X) \} & = & \theta_ {0} + \theta_ {1} Z + \theta_ {2} M + \theta_ {4} ^ {\mathsf {T}} X. \end{array} \right.
$$

Express nde and nie in terms of the model parameters and the distribution of X. How do we estimate nde and nie with IID data?

## 27.8 Modify the definitions to drop the cross-world independence

Define

$$
Y (z, F _ {M _ {z ^ {\prime}} | X}) = \int Y (z, m) f _ {M _ {z ^ {\prime}}} (m \mid X) \mathrm{d} m
$$

as the potential outcome under treatment z and a random draw from the distribution of $M _ { z ^ { \prime } } \mid X$ . The key difference between $Y ( z , M _ { z ^ { \prime } } )$ and $Y ( z , F _ { M _ { z ^ { \prime } } | X } )$ is that $M _ { z ^ { \prime } }$ is the potential mediator for the same unit whereas $F _ { M _ { z ^ { \prime } } | X }$ is a random draw from the conditional distribution of the potential mediator in the whole population. Define the natural direct and indirect effects as

$$
\mathrm{NDE} = E \{Y (1, F _ {M _ {0} | X}) - Y (0, F _ {M _ {0} | X}) \}, \quad \mathrm{NIE} = E \{Y (1, F _ {M _ {1} | X}) - Y (1, F _ {M _ {0} | X}) \}.
$$

## 27.6 Homework problems

Show that under Assumptions 27.2–27.4, the identification formulas for nde and nie remain the same as in the main text.

Remark: Modifying the definitions of the nested potential outcomes allows us to relax the strong cross-world independence assumption but weakens the interpretation of the natural direct and indirect effects. See VanderWeele (2015) for more discussion and VanderWeele and Tchetgen Tchetgen (2017) for an application to a more complex setting with time varying treatment and mediator.

## 27.9 Connections between principal stratification and mediation analysis

VanderWeele (2008) and Forastiere et al. (2018) reviewed and compared principal stratification and mediation analysis.

# Controlled Direct Effect

The formulation of mediation analysis in Chapter 27 relies on the nested potential outcomes, and fundamentally, some nested potential outcomes are not observable in any physical experiments. If we stick to the Popperian philosophy of science, we should only define causal parameters in terms of quantities that are observable under some experiments. This chapter discusses an alternative view of causal inference with an intermediate variable. In this view, we only define the direct effect but can not define the indirect effect.

## 28.1 Identification and estimation of the controlled direct effect

We view Z and M as two factors, and define potential outcomes $Y ( z , m )$ for $z = 0 , 1$ and $m \in { \mathcal { M } } .$ . Based on these potential outcomes, we can define the controlled direct effect (CDE) below.

Definition 28.1 (CDE) Define

$$
\operatorname{CDE} (m) = E \{Y (1, m) - Y (0, m) \}.
$$

By definition, cde(m) is the average causal effect of the treatment if the intermediate variable is fixed at m. The parameter cde(m) can capture the direct effect of the treatment holding the mediator at m. However, this formulation cannot capture the indirect effect. In particular, the parameter $E \{ Y ( z , 1 ) - Y ( z , 0 ) \}$ only measures the effect of the mediator on the outcome holding the treatment at z. This is not a meaningful definition of the indirect effect.

To identify cde(m), we need the following assumption, which basically requires that Z and M are jointly randomized given X.

Assumption 28.1 Sequential ignorability requires

$$
Z \bot Y (z, m) \mid X, \quad M \bot Y (z, m) \mid (Z, X)
$$

$o r ,$ equivalently,

$$
(Z, M) \bot Y (z, m) \mid X.
$$

I will focus on the case with a binary Z and M. Mathematically, we can just view this problem as an observational study with four treatment levels

$$
(z, m) \in \{(0, 0), (0, 1), (1, 0), (1, 1) \}.
$$

The following theorem extends the results for observational studies with a binary treatment, identifying

$$
\mu_ {z m} = E \{Y (z, m) \}
$$

based on outcome regression, inverse probability weighting, and doubly robust estimation.

Define

$$
\mu_ {z m} (x) = E (Y \mid Z = z, M = m, X = x)
$$

as the outcome mean conditional on the treatment, mediator and covariates. Define

$$
e _ {z m} (x) = \operatorname * {p r} (Z = z, M = m \mid X = x) = \operatorname * {p r} (Z = z \mid X = x) \operatorname * {p r} (M = m \mid Z = z, X = x)
$$

as the probability of the joint value of Z and M conditional on the covariates.

Theorem 28.1 Under Assumption 28.1, we have

$$
\mu_ {z m} = E \{\mu_ {z m} (X) \}
$$

or

$$
\mu_ {z m} = E \left\{\frac {I (Z = z , M = m) Y}{e _ {z m} (X)} \right\}.
$$

Moreover, based on the working models $e _ { z m } ( X , \alpha )$ and $\mu _ { z m } ( X , \beta )$ , we have the doubly robust formula

$$
\mu_ {z m} ^ {\mathrm{dr}} = E \{\mu_ {z m} (X, \beta) \} + E \left[ \frac {I (Z = z , M = m) \{Y - \mu_ {z m} (X , \beta) \}}{e _ {z m} (X , \alpha)} \right],
$$

$w h i c h ~ e q u a l s ~ \mu _ { z m } ~ i f ~ e i t h e r ~ e _ { z m } ( X , \alpha ) = e _ { z m } ( X ) ~ o r ~ \mu _ { z m } ( X , \beta ) = \mu _ { z m } ( X ) .$

The proof of Theorem 28.1 is similar to those for the standard unconfounded observational studies. Problem 28.2 gives a general result. Based on the outcome mean model, we can obtain ${ \hat { \mu } } _ { z m } ( x )$ for $\mu _ { z m } ( x )$ . Based on the treatment model, we can obtain $\hat { e } _ { z } ( x )$ for $\operatorname { p r } ( Z = z \mid X = x )$ ; based on the intermediate variable model, we can obtain $\hat { e } _ { m } ( z , x )$ for $\operatorname { p r } ( M = m \mid Z =$ $z , X = x )$ ). We can then estimate $\mu _ { z m }$ by outcome regression

$$
\hat {\mu} _ {z m} ^ {\mathrm{reg}} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\mu} _ {z m} (X _ {i}),
$$

by inverse probability weighting

$$
\begin{array}{l} \hat {\mu} _ {z m} ^ {\mathrm{ht}} = n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {I (Z _ {i} = z , M _ {i} = m) Y _ {i}}{\hat {e} _ {z} (X _ {i}) \hat {e} _ {m} (z , X _ {i})}, \\ \hat {\mu} _ {z m} ^ {\mathrm{haj}} = \sum_ {i = 1} ^ {n} \frac {I (Z _ {i} = z , M _ {i} = m) Y _ {i}}{\hat {e} _ {z} (X _ {i}) \hat {e} _ {m} (z , X _ {i})} \bigg / \sum_ {i = 1} ^ {n} \frac {I (Z _ {i} = z , M _ {i} = m)}{\hat {e} _ {z} (X _ {i}) \hat {e} _ {m} (z , X _ {i})}, \\ \end{array}
$$

or by augmented inverse probability weighting

$$
\hat {\mu} _ {z m} ^ {\mathrm{dr}} = \hat {\mu} _ {z m} ^ {\mathrm{reg}} + n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {I (Z _ {i} = z , M _ {i} = m) \{Y _ {i} - \hat {\mu} _ {z m} (X _ {i}) \}}{\hat {e} _ {z} (X _ {i}) \hat {e} _ {m} (z , X _ {i})}.
$$

We can then estimate $\mathrm { C D E } ( m )$ by $\hat { \mu } _ { 1 m } - \hat { \mu } _ { 0 m }$ and use the bootstrap to approximate the standard error.

If we are willing to assume a linear outcome model, the controlled direct effect simplifies to the coefficient of the treatment. Example 28.1 below gives the details.

Example 28.1 Under Assumption 28.1 and a linear outcome model,

$$
E (Y \mid Z, M, X) = \theta_ {0} + \theta_ {1} Z + \theta_ {2} M + \theta_ {4} ^ {\mathsf {T}} X,
$$

we can show that $\mathrm { C D E } ( m )$ equals the coefficient $\theta _ { 1 }$ , which coincides with the natural direct effect in the Baron–Kenny method. I relegate the proof to Problem 28.3.

## 28.2 Discussion

The formulation of the controlled direct effect does not involve nested or a priori counterfactual potential outcomes, and its identification does not require the cross-world counterfactual independence assumption. The parameter cde(m) can capture the direct effect of the treatment holding the mediator at m. However, this formulation cannot capture the indirect effect. I summarize the causal frameworks for intermediate variables below.

<table><tr><td>chapter</td><td>framework</td><td>direct effect</td><td>indirect effect</td></tr><tr><td>26</td><td>principal stratification</td><td>τ(1,1), τ(0,0)</td><td>?</td></tr><tr><td>27</td><td>mediation analysis</td><td>NDE</td><td>NIE</td></tr><tr><td>29</td><td>controlled direct effect</td><td>CDE(m)</td><td>?</td></tr></table>

The mediation analysis framework can decompose the total effect into natural direct and indirect effects, but it requires nested potential outcomes and cross-world independence. The principal stratification and controlled direct effect frameworks cannot define indirect effects but they do not involve nested potential outcomes and cross-world independence. Moreover, the principal stratification framework does not necessarily require that M lies on the causal pathway from the treatment to the outcome. But its identification and estimation involves disentangling mixture distributions, which is a nontrivial task in statistics.

## 28.3 Homework problems

## 28.1 cde and nde

Show that under cross-world independence $Y ( z , m ) \bot M ( z ^ { \prime } ) \mid X$ for all $z , z ^ { \prime }$ and m, the c ${ \mathrm { o n d i t i o n a l ~ C D E } } ( m \mid x ) = E \{ Y ( 1 , m ) - Y ( 0 , m ) \mid X = x \}$ and $\operatorname { N D E } ( x ) = E \{ Y ( 1 , M _ { 0 } ) - Y ( 0 , M _ { 0 } ) \mid X = x \}$ have the following relationship:

$$
\mathrm{NDE} (x) = E \{\mathrm{CDE} (M _ {0} \mid x) \},
$$

which reduces to

$$
\mathrm{NDE} (x) = \sum_ {m} \mathrm{CDE} (m \mid x) \mathrm{pr} (M _ {0} = m \mid X = x)
$$

for a discrete M. Without the cross-world independence, does this relationship still hold in general?

## 28.2 Observational studies with a multi-valued treatment

Theorem 28.1 is a special case of the following theorem for unconfounded observational studies with multiple treatment levels (Imai and Van Dyk, 2004; Cattaneo, 2010). Below, I state the general problem and theorem.

Consider an observational study with a multi-valued treatment $Z \in \mathbf { \Sigma }$ $\{ 1 , \ldots , K \}$ , covariates X, and outcome Y . Unit i has K potential outcomes $Y _ { i } ( 1 ) , \ldots , Y _ { i } ( K )$ corresponding to the K treatment levels. Causal effects can be defined as comparisons of the potential outcomes. In general, we can define causal effect in terms of contrasts of the potential outcomes:

$$
\tau_ {c} = \sum_ {k = 1} ^ {K} c _ {k} E \{Y (k) \}
$$

where $\textstyle \sum _ { k = 1 } ^ { K } c _ { k } = 0$ . The canonical choice of the pairwise comparison

$$
\tau_ {k, k ^ {\prime}} = E \{Y (k) - Y (k ^ {\prime}) \}.
$$

Therefore, the key is to identify and estimate the means of the potential outcomes $\mu _ { k } = E \{ Y ( k ) \}$ under the ignorability assumption below based on IID data of $( Z _ { i } , X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ .

Assumption 28.2 $Z \bot \bot \{ Y ( 1 ) , \dots , Y ( K ) \} \mid X$ .

Define the generalized propensity score as

$$
e _ {k} (X) = \operatorname{pr} (Z = k \mid X),
$$

and define the conditional outcome mean as

$$
\mu_ {k} (X) = E (Y \mid Z = k, X)
$$

for $k = 1 , \ldots , K$ . We have the following theorem.

Theorem 28.2 Under Assumption 28.2, we have

$$
\mu_ {k} = E \{\mu_ {k} (X) \}
$$

or

$$
\mu_ {k} = E \left\{\frac {I (Z = k) Y}{e _ {k} (X)} \right\}.
$$

Moreover, based on the working models $e _ { k } ( X , \alpha )$ and $\mu _ { k } ( X , { \boldsymbol { \beta } } )$ , we have the doubly robust formula

$$
\mu_ {k} ^ {\mathrm{dr}} = E \{\mu_ {k} (X, \beta) \} + E \left[ \frac {I (Z = k) \{Y - \mu_ {k} (X , \beta) \}}{e _ {k} (X , \alpha)} \right],
$$

which equals $\mu _ { k }$ if either $e _ { k } ( X , \alpha ) = e _ { k } ( X ) \ o r \mu _ { k } ( X , \beta ) = \mu _ { k } ( X )$

Prove Theorem 28.2.

Remark: Theorem 28.1 is a special case of Theorem 28.2 if we view the $( Z , M )$ in Theorem 28.1 as a treatment with four levels. The $\mathrm { C D E } ( m )$ is a special case of $\tau _ { c } .$ .

## 28.3 cde in the linear outcome model

Show that under Assumption 28.1, if $E ( Y \mid Z , M , X ) = \theta _ { 0 } + \theta _ { 1 } Z + \theta _ { 2 } M + \theta _ { 4 } ^ { \mathsf { T } } X$ then

$$
\mathrm{CDE} (m) = \theta_ {1}
$$

for all $m ;$ if $E ( Y \mid Z , M , X ) = \theta _ { 0 } + \theta _ { 1 } Z + \theta _ { 2 } M + \theta _ { 3 } Z M + \theta _ { 4 } ^ { \mathsf { T } } X$ , then

$$
\operatorname{CDE} (m) = \theta_ {1} + \theta_ {3} m.
$$

## 28.4 cde in the logit outcome model

Show that for a binary outcome, under Assumption 28.1, if

$$
\operatorname{logit} \left\{\operatorname * {p r} (Y = 1 \mid Z, M, X) \right\} = \theta_ {0} + \theta_ {1} Z + \theta_ {2} M + \theta_ {4} ^ {\mathsf {T}} X,
$$

then

$$
\operatorname{CDE} (m) = E \{\expit (\theta_ {0} + \theta_ {1} + \theta_ {2} m + \theta_ {4} ^ {\mathsf {T}} X) - \expit (\theta_ {0} + \theta_ {2} m + \theta_ {4} ^ {\mathsf {T}} X) \};
$$

if

$$
\operatorname{logit} \left\{\operatorname{pr} (Y = 1 \mid Z, M, X) \right\} = \theta_ {0} + \theta_ {1} Z + \theta_ {2} M + \theta_ {3} Z M + \theta_ {4} ^ {\mathsf {T}} X,
$$

then

$$
\operatorname{CDE} (m) = E \{\expit (\theta_ {0} + \theta_ {1} + \theta_ {2} m + \theta_ {3} m + \theta_ {4} ^ {\mathsf {T}} X) - \expit (\theta_ {0} + \theta_ {2} m + \theta_ {4} ^ {\mathsf {T}} X) \}.
$$

## 28.5 Recommended reading

Nguyen et al. (2021) provided a friendly review of of the topics in Chapters 27 and 29.

# Time-Varying Treatment and Confounding

Studies with time-varying treatments are common in biomedical and social sciences. James Robins championed the research in biostatistics. A classic example is that HIV patients may take the azidothymidine, an antiretroviral medication, on and off over time (Robins et al., 2000; Hern´an et al., 2000). Similar problems also exist in other fields. In education, a classic example is that students may receive different types of instructions over time (Hong and Raudenbush, 2008). In political science, a classic example is that candidates continuously recalibrate their campaign strategy based on time-varying polls and opponent actions (Blackwell, 2013).

Causal inference with a time-varying treatment is not a simple extension of causal inference with a treatment at a single time point. The main challenge is time-varying confounding. Even if we assume all time-varying confounders are observed, we still face statistical challenges in adjusting for those confounders. One the one hand, we should stratify on these confounders to adjust for confounding; on the other hand, stratifying on post-treatment variables will cause bias. Due to these two conflicting goals, causal inference with time-varying treatments and confounding requires more sophisticated statistical methods. It is the main topic of this chapter.

To minimize the notational burden, I will use the setting with a treatment at two time points to convey the most important ideas. Extensions to treatments at multiple time points can be conceptually straightforward although technical complexities will arise in finite samples. I will discuss the complexities and relegate general results to Problems 29.6–29.9.

## 29.1 Basic setup and sequential ignorability

Start with a treatment at two time points. The temporal order of the variables with two time points is below:

$$
X _ {0} \rightarrow Z _ {1} \rightarrow X _ {1} \rightarrow Z _ {2} \rightarrow Y
$$

where

• $X _ { 0 }$ denotes the baseline pre-treatment covariates;

![image_36](images/image_36.png)

FIGURE 29.1: Without unmeasured confounding U between $X _ { 1 }$ and Y . The causal diagram conditions on the pre-treatment covariates $X _ { 0 }$ .

• $Z _ { 1 }$ denotes the treatment at time point 1;
• $X _ { 1 }$ denotes the time-varying covariates between the treatments at time points 1 and 2;
• $Z _ { 2 }$ denotes the treatment at time point 2;
• Y denotes the outcome.

With binary treatment $( Z _ { 1 } , Z _ { 2 } )$ , each unit has four potential outcomes

$$
Y (z _ {1}, z _ {2}) \text {   for   } z _ {1}, z _ {2} = 0, 1.
$$

The observed outcome equals

$$
Y = Y \left(Z _ {1}, Z _ {2}\right) = \sum_ {z _ {1} = 0, 1} \sum_ {z _ {2} = 0, 1} 1 \left(Z _ {1} = z _ {1}\right) 1 \left(Z _ {2} = z _ {2}\right) Y \left(z _ {1}, z _ {2}\right).
$$

I will focus on the canonical setting with sequential ignorability, that is, the treatments are sequentially randomized given the observed history.

Assumption 29.1 (sequential ignorability) $( 1 ) \ Z _ { 1 }$ is randomized given $X _ { 0 }$ :

$$
Z _ {1} \bot Y (z _ {1}, z _ {2}) \mid X _ {0} f o r z _ {1}, z _ {2} = 0, 1.
$$

(2) $Z _ { 2 }$ is randomized given $( Z _ { 1 } , X _ { 1 } , X _ { 0 } )$ :

$$
Z _ {2} \bot Y (z _ {1}, z _ {2}) \mid (Z _ {1}, X _ {1}, X _ {0}) f o r z _ {1}, z _ {2} = 0, 1.
$$

Figure 29.1 is a simple causal diagram corresponding to Assumption 29.1, which does not contains any unmeasured confounding.

Figure 29.2 is a more complex causal diagram corresponding to Assumption 29.1. Sequential ignorability rules out only the confounding between the treatment $( Z _ { 1 } , Z _ { 2 } )$ and the outcome $Y$ , but allows for unmeasured confounding between the time-varying covariate $X _ { 1 }$ and the outcome Y . The possible existence of U causes many subtle issues even under sequential ignorability.

![image_37](images/image_37.png)

```mermaid
graph TD
  Z1 --> X1
  X1 --> Z2
  Z2 --> Y
  Y --> U
  X1 --> Z1
  Z2 --> Y
  U --> X1
```

FIGURE 29.2: With unmeasured confounding between $X _ { 1 }$ and $Y .$ The causal diagram conditions on the pre-treatment covariates $X _ { 0 }$ .

## 29.2 g-formula and outcome modeling

Recall the outcome-based identification formula with a treatment at a single time point:

$$
E \{Y (z) \} = E \{E (Y \mid Z = z, X) \}.
$$

With discrete X, it reduces to

$$
E \{Y (z) \} = \sum_ {x} E (Y \mid Z = z, X = x) \mathrm{pr} (X = x);
$$

with continuous X, it reduces to

$$
E \{Y (z) \} = \int E (Y \mid Z = z, X = x) f _ {X} (x) \mathrm{d} x.
$$

The following result extends it to the setting with a treatment at two time points.

Theorem 29.1 Under Assumption 29. $^ { 1 , }$

$$
E \{Y (z _ {1}, z _ {2}) \} = E \Big [ E \{E (Y \mid z _ {2}, z _ {1}, X _ {1}, X _ {0}) \mid z _ {1}, X _ {0} \} \Big ]. \tag {29.1}
$$

In Theorem 29.1, I simplify the notation ${ } ^ { \circ } Z _ { 2 } = z _ { 2 } { } ^ { \prime \prime }$ to $^ { 6 } z _ { 2 } ^ { , 5 }$ for simplicity. To void complex formulas in this Chapter, I will use the lower case letter to represent the event that the random variable takes the corresponding value. With discrete $X _ { 0 }$ and $X _ { 1 }$ , the identification formula (29.1) reduces to

$$
E \{Y (z _ {1}, z _ {2}) \} = \sum_ {x _ {0}} \sum_ {x _ {1}} E (Y \mid z _ {2}, z _ {1}, x _ {1}, x _ {0}) \mathrm{pr} (x _ {1} \mid z _ {1}, x _ {0}) \mathrm{pr} (x _ {0}); \tag {29.2}
$$

with continuous $X _ { 0 }$ and $X _ { 1 }$ , the identification formula (29.1) reduces to

$$
E \{Y (z _ {1}, z _ {2}) \} = \int \int E (Y \mid z _ {2}, z _ {1}, x _ {1}, x _ {0}) f (x _ {1} \mid z _ {1}, x _ {0}) f (x _ {0}) \mathrm{d} x _ {1} \mathrm{d} x _ {0}. \tag {29.3}
$$

Compare (29.2) with the formula based on the law of total probability to gain more insights:

$$
\begin{array}{l} E (Y) = \sum_ {x _ {0}} \sum_ {z _ {1}} \sum_ {x _ {1}} \sum_ {z _ {2}} E (Y \mid z _ {2}, z _ {1}, x _ {1}, x _ {0}) \\ \operatorname{pr} (z _ {1} \mid z _ {1}, x _ {1}, x _ {0}) \operatorname{pr} (x _ {1} \mid z _ {1}, x _ {0}) \operatorname{pr} (z _ {1} \mid x _ {0}) \operatorname{pr} (x _ {0}). \tag {29.4} \\ \end{array}
$$

Erasing the probabilities of $z _ { 2 }$ and $z _ { 1 }$ in (29.4), we can obtain the formula (29.3). This is intuitive because the potential outcome $Y ( z _ { 1 } , z _ { 2 } )$ has the meaning of fixing $Z _ { 1 }$ and $Z _ { 2 }$ at $z _ { 1 }$ and $z _ { 2 } .$ respectively.

Robins called (29.2) and (29.3) the g-formulas. Now I will prove Theorem 29.1.

Proof of Theorem 29.1: By the tower property,

$$
E \{Y (z _ {1}, z _ {2}) \} = E \left[ E \{Y (z _ {1}, z _ {2}) \mid X _ {0} \} \right],
$$

so we focus on $E \{ Y ( z _ { 1 } , z _ { 2 } ) \mid X _ { 0 } \}$ . By Assumption 29.1(1) and the tower property,

$$
\begin{array}{l} E \{Y (z _ {1}, z _ {2}) \mid X _ {0} \} = E \{Y (z _ {1}, z _ {2}) \mid z _ {1}, X _ {0} \} \\ = E \left[ E \left\{Y \left(z _ {1}, z _ {2}\right) \mid z _ {1}, X _ {1}, X _ {0} \right\} \mid z _ {1}, X _ {0} \right]. \\ \end{array}
$$

By Assumption 29.1(2),

$$
\begin{array}{l} E \{Y (z _ {1}, z _ {2}) \mid X _ {0} \} = E \Big [ E \{Y (z _ {1}, z _ {2}) \mid z _ {2}, z _ {1}, X _ {1}, X _ {0} \} \mid z _ {1}, X _ {0} \Big ] \\ = E \left[ E \left\{Y \mid z _ {2}, z _ {1}, X _ {1}, X _ {0} \right\} \mid z _ {1}, X _ {0} \right]. \\ \end{array}
$$

The formula (29.1) follows.

![image_38](images/image_38.png)

## 29.2.1 Plug-in estimation based on outcome modeling

The g-formulas (29.2) and (29.3) suggest that to estimate the means of the potential outcomes, we need to model $E ( Y \mid z _ { 2 } , z _ { 1 } , x _ { 1 } , x _ { 0 } ) , \operatorname { p r } ( x _ { 1 } \mid z _ { 1 } , x _ { 0 } )$ and $\mathrm { p r } ( x _ { 0 } )$ . With these fitted models, we can plug them in the $\mathrm { g } -$ formulas.

With some special functional forms, this task can be simplified. Example 29.1 below gives the results under a linear model for the outcome.

Example 29.1 Assume a linear outcome model

$$
E (Y \mid z _ {2}, z _ {1}, x _ {1}, x _ {0}) = \beta_ {0} + \beta_ {1} z _ {2} + \beta_ {2} z _ {1} + \beta_ {3} x _ {1} + \beta_ {4} x _ {0}.
$$

We can verify that

$$
\begin{array}{l} E \{Y (z _ {1}, z _ {2}) \} = \sum_ {x _ {0}} \sum_ {x _ {1}} (\beta_ {0} + \beta_ {1} z _ {2} + \beta_ {2} z _ {1} + \beta_ {3} x _ {1} + \beta_ {4} x _ {0}) \mathrm{pr} (x _ {1} \mid z _ {1}, x _ {0}) \mathrm{pr} (x _ {0}) \\ = \beta_ {0} + \beta_ {1} z _ {2} + \beta_ {2} z _ {1} + \beta_ {3} \sum_ {x _ {0}} E (X _ {1} \mid z _ {1}, x _ {0}) \mathrm{pr} (x _ {0}) + \beta_ {4} E (X _ {0}). \\ \end{array}
$$

Define

$$
E \{X _ {1} (z _ {1}) \} = \sum_ {x _ {0}} E (X _ {1} \mid z _ {1}, x _ {0}) \mathrm{pr} (x _ {0}) \tag {29.5}
$$

to simplify the formula as

$$
E \{Y (z _ {1}, z _ {2}) \} = \beta_ {0} + \beta_ {1} z _ {2} + \beta_ {2} z _ {1} + \beta_ {3} E \{X _ {1} (z _ {1}) \} + \beta_ {4} E (X _ {0}).
$$

In (29.5), I introduce the potential outcome of $X _ { 1 }$ under the treatment $Z _ { 1 } =$ $z _ { 1 }$ at time point 1. It is reasonable because the right-hand side of $( 2 9 . 5 )$ is the identification formula of $E \{ X _ { 1 } ( z _ { 1 } ) \}$ under ignorability $X _ { 1 } ( z _ { 1 } ) \bot \bot \ Z _ { 1 } \mid X _ { 0 }$ for $z _ { 1 } = 0 , 1$ . We do not really need the potential outcome $X _ { 1 } ( z _ { 1 } )$ and the ignorability, but it is a convenient notation and matches with our discussion before.

Define $\tau _ { Z _ { 1 }  X _ { 1 } } = E \{ X _ { 1 } ( 1 ) - X _ { 1 } ( 0 ) \}$ . We can verify that

$$
E \{Y (1, 0) - Y (0, 0) \} = \beta_ {2} + \beta_ {3} \tau_ {Z _ {1} \rightarrow X _ {1}},
$$

$$
E \{Y (0, 1) - Y (0, 0) \} = \beta_ {1},
$$

$$
E \{Y (1, 1) - Y (0, 0) \} = \beta_ {1} + \beta_ {2} + \beta_ {3} \tau_ {Z _ {1} \to X _ {1}}.
$$

Therefore, we can estimate the effect of $( Z _ { 1 } , Z _ { 2 } )$ on $Y$ based on the above formulas by first estimating the regression coefficients $\beta s$ and the average causal effect of $Z _ { 1 }$ on $X _ { 1 }$ using standard methods.

However, Robins and Wasserman (1997) pointed out a surprising drawback of the plug-in estimation based on outcome modeling. They showed that with model misspecification in this strategy, data analyzers may falsely reject the null hypothesis of zero causal effect of $( Z _ { 1 } , Z _ { 2 } )$ on $Y$ even when the true effect is zero in the data-generating process. They called it the g-null paradox. Perhaps surprisingly, they show that the g-null paradox may even arise in the simple linear outcome model in Example 29.1. McGrath et al. (2021) revisited this paradox. See Problem 29.1 for more details.

## 29.2.2 Recursive estimation based on outcome modeling

The plug-in estimation in Section 29.2.1 involves modeling the time-varying confounder $X _ { 1 }$ and causes the unpleasant g-null paradox. It is not a desirable method.

Recall the outcome regression estimator with a treatment at a single time based on $E \{ Y ( z ) \} = E \{ { \bar { E } } ( Y \mid Z = z , X ) \}$ . We first fit a model of $Y$ on $X$ using the subset of the data with $Z = z .$ and obtain the fitted values $\hat { Y } _ { i } ( z )$ for all units. We then obtain the estimator

$$
\hat {E} \{Y (z) \} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {Y} _ {i} (z).
$$

Similarly, the recursive expectation formula in (29.1) motivates a simpler method for estimation. Start from the inner conditional expectation, denoted by

$$
\tilde {Y} _ {2} (z _ {1}, z _ {2}) = E (Y \mid Z _ {2} = z _ {2}, Z _ {1} = z _ {1}, X _ {1}, X _ {0}).
$$

We can fit a model of $Y$ on $( X _ { 1 } , X _ { 0 } )$ using the subset of the data with $( Z _ { 2 } =$ $z _ { 2 } , Z _ { 1 } = z _ { 1 } )$ , and obtain the fitted values $\hat { Y } _ { 2 i } ( z _ { 1 } , z _ { 2 } )$ for all units. Move on to outer conditional expectation, denoted by

$$
\tilde {Y} _ {1} (z _ {1}, z _ {2}) = E \{\tilde {Y} _ {2} (z _ {1}, z _ {2}) \mid Z _ {1} = z _ {1}, X _ {0} \}.
$$

We can fit a model of $\hat { Y } _ { 2 } ( z _ { 1 } , z _ { 2 } )$ on $X _ { 0 }$ using the subset of data with $Z _ { 1 } = z _ { 1 }$ , and obtain the fitted values $\hat { Y } _ { 1 i } ( z _ { 1 } , z _ { 2 } )$ for all units. The final estimator for $E \{ Y ( z _ { 1 } , z _ { 2 } ) \}$ is then

$$
\hat {E} \{Y (z _ {1}, z _ {2}) \} = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {Y} _ {1 i} (z _ {1}, z _ {2}).
$$

The above recursive estimation does not involve fitting a model for $X _ { 1 }$ and avoids the g-null paradox. See Problem 29.2 for a special case.

## 29.3 Inverse propensity score weighting

Recall the propensity-score-based identification formula with a treatment at a single time point:

$$
E \{Y (z) \} = E \left\{\frac {1 (Z = z) Y}{\operatorname* {p r} (Z = z \mid X)} \right\}.
$$

The following result extends it to the setting with a treatment at two time points. Define

$$
e (z _ {1}, X _ {0}) = \mathrm{pr} (Z _ {1} = z _ {1} \mid X _ {0})
$$

and

$$
e (z _ {2}, Z _ {1}, X _ {1}, X _ {0}) = \mathrm{pr} (Z _ {2} = z _ {2} \mid Z _ {1}, X _ {1}, X _ {0})
$$

as the propensity scores at time points 1 and 2, respectively.

Theorem 29.2 Under Assumption 29. $^ { 1 , }$

$$
E \{Y (z _ {1}, z _ {2}) \} = E \left\{\frac {1 (Z _ {1} = z _ {1}) 1 (Z _ {2} = z _ {2}) Y}{e (z _ {1} , X _ {0}) e (z _ {2} , Z _ {1} , X _ {1} , X _ {0})} \right\}. \tag {29.6}
$$

Theorem 29.2 reveals the omitted overlap assumption:

$$
0 <   e \left(z _ {1}, X _ {0}\right) <   1, \quad 0 <   e \left(z _ {2}, Z _ {1}, X _ {1}, X _ {0}\right) <   1
$$

for all $z _ { 1 }$ and $z _ { 2 } .$ . If some propensity scores are 0 or 1, then the identification formula (29.6) blows up to infinity.

Proof of Theorem 29.2: Conditioning on $( Z _ { 1 } , X _ { 1 } , X _ { 0 } )$ and using Assumption 29.1(2), we can simplify the right-hand side of (29.6) as

$$
\begin{array}{l} E \left\{\frac {1 (Z _ {1} = z _ {1}) 1 (Z _ {2} = z _ {2}) Y (z _ {1} , z _ {2})}{\operatorname{pr} (Z _ {1} = z _ {1} \mid X _ {0}) \operatorname{pr} (Z _ {2} = z _ {2} \mid Z _ {1} , X _ {1} , X _ {0})} \right\} \\ = E \left\{\frac {1 (Z _ {1} = z _ {1}) \mathrm{pr} (Z _ {2} = z _ {2} \mid Z _ {1} , X _ {1} , X _ {0}) E (Y (z _ {1} , z _ {2}) \mid Z _ {1} , X _ {1} , X _ {0})}{\mathrm{pr} (Z _ {1} = z _ {1} \mid X _ {0}) \mathrm{pr} (Z _ {2} = z _ {2} \mid Z _ {1} , X _ {1} , X _ {0})} \right\} \\ = E \left\{\frac {1 (Z _ {1} = z _ {1})}{\operatorname{pr} (Z _ {1} = z _ {1} \mid X _ {0})} E (Y (z _ {1}, z _ {2}) \mid Z _ {1}, X _ {1}, X _ {0}) \right\} \\ = E \left\{\frac {1 (Z _ {1} = z _ {1})}{\operatorname{pr} (Z _ {1} = z _ {1} \mid X _ {0})} Y (z _ {1}, z _ {2}) \right\}, \tag {29.7} \\ \end{array}
$$

where (29.7) follows from the tower property.

Conditioning on $X _ { 0 }$ and using Assumption 29.1(1), we can simplify the right-hand side of (29.7) as

$$
\begin{array}{l} E \left\{\frac {\operatorname{pr} (Z _ {1} = z _ {1} \mid X _ {0})}{\operatorname{pr} (Z _ {1} = z _ {1} \mid X _ {0})} E (Y (z _ {1}, z _ {2}) \mid X _ {0}) \right\} \\ = E \left\{E \left(Y \left(z _ {1}, z _ {2}\right) \mid X _ {0}\right) \right\} \\ = E \{Y (z _ {1}, z _ {2}) \}, \\ \end{array}
$$

where, again, the last line follows from the tower property.

The estimator based on IPW is much simpler which only involves modeling two binary indicators. First, we can fit a model of $Z _ { 1 }$ on $X _ { 0 }$ to obtain the fitted values $\hat { e } _ { 1 } ( z _ { 1 } , X _ { 0 i } )$ and fit a model of $Z _ { 2 }$ on $( Z _ { 1 } , X _ { 1 } , X _ { 0 } )$ to obtain the fitted values $\hat { e } _ { 2 } ( z _ { 2 } , Z _ { 1 i } , X _ { 1 i } , X _ { 0 i } )$ for all units. Then, we obtain the following IPW estimator:

$$
\hat {E} ^ {\mathrm{ht}} \left\{Y \left(z _ {1}, z _ {2}\right) \right\} = n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {1 \left(Z _ {1 i} = z _ {1}\right) 1 \left(Z _ {2 i} = z _ {2}\right) Y _ {i}}{\hat {e} _ {1} \left(z _ {1} , X _ {0 i}\right) \hat {e} _ {2} \left(z _ {2} , Z _ {1 i} , X _ {1 i} , X _ {0 i}\right)}.
$$

Similar to the discussion in Chapter 11, the Horvitz–Thompson-type estimator is not invariant to location shift of the outcome and suffers from instability in finite samples. A modified Hajek-type estimator is $\hat { E } ^ { \mathrm { h a j } } \{ Y ( z _ { 1 } , z _ { 2 } ) \} =$ $\hat { E } ^ { \mathrm { h t } } \{ Y ( z _ { 1 } , z _ { 2 } ) \} / \hat { 1 } ^ { \mathrm { h t } } ( z _ { 1 } , z _ { 2 } )$ , where

$$
\hat {1} ^ {\mathrm{ht}} (z _ {1}, z _ {2}) = n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {1 (Z _ {1 i} = z _ {1}) 1 (Z _ {2 i} = z _ {2})}{\hat {e} _ {1} (z _ {1} , X _ {0 i}) \hat {e} _ {2} (z _ {2} , Z _ {1 i} , X _ {1 i} , X _ {0 i})}.
$$

## 29.4 Multiple time points

Extending the estimation strategies in Sections 29.2 and 29.3 is not immediate with multiple time points. Even with a binary treatment and K time points, the number of treatment combination grows exponentially with K (for example, $2 ^ { 5 } = 3 2$ and $2 ^ { 1 0 } = 1 0 2 4 \rangle$ . Consequently, the outcome regression and IPW estimators in Sections 29.2 and 29.3 are not feasible in finite samples.

## 29.4.1 Marginal structural model

A powerful approach is based on the marginal structural model (MSM) (Robins et al., 2000; Hern´an et al., 2000). For simplicity of notation, I will only present the MSM with $K = 2$ although its main use is in the general case.

Definition 29.1 (MSM) The marginal mean of $Y ( z _ { 1 } , z _ { 2 } )$ equals

$$
E \{Y (z _ {1}, z _ {2}) \} = f (z _ {1}, z _ {2}; \beta).
$$

A leading example of Definition 29.1 is $E \{ Y ( z _ { 1 } , z _ { 2 } ) \} = \beta _ { 0 } + \beta _ { 1 } z _ { 1 } + \beta _ { 2 } z _ { 2 }$ . It is also straightforward to include the baseline covariates in the model. Definition 29.2 below extends Definition 29.1.

Definition 29.2 (MSM with baseline covariates) The mean of $Y ( z _ { 1 } , z _ { 2 } )$ conditional on $X _ { 0 }$ equals

$$
E \{Y (z _ {1}, z _ {2}) \mid X _ {0} \} = f (z _ {1}, z _ {2}, X _ {0}; \beta).
$$

A leading example of Definition 29.2 is

$$
E \{Y (z _ {1}, z _ {2}) \mid X _ {0} \} = \beta_ {0} + \beta_ {1} z _ {1} + \beta_ {2} z _ {2} + \beta_ {3} ^ {\mathsf {T}} X _ {0}. \tag {29.8}
$$

If we observe all potential outcomes, we can solve $\beta$ from the following minimization problem:

$$
\beta = \arg \min _ {b} \sum_ {z _ {2}} \sum_ {z _ {1}} E \{Y (z _ {1}, z _ {2}) - f (z _ {1}, z _ {2}, X _ {0}; b) \} ^ {2}.
$$

For simplicity, I focus on the least squares formulation. We can also extend the discussion to a general loss function.

Under sequential ignorability, we can solve $\beta$ from the following minimization problem that only involves observables.

Theorem 29.3 (IPW under MSM) Under Assumption 29.1 and Definition 29.2,

$$
\beta = \arg \min _ {b} \sum_ {z _ {2}} \sum_ {z _ {1}} E \left[ \frac {1 (Z _ {1} = z _ {1}) 1 (Z _ {2} = z _ {2})}{e (z _ {1} , X _ {0}) e (z _ {2} , Z _ {1} , X _ {1} , X _ {0})} \{Y - f (z _ {1}, z _ {2}, X _ {0}; b) \} ^ {2} \right].
$$

The proof of Theorem 29.3 is similar to that of Theorem 29.2. I relegate it to Problem 29.3.

Theorem 29.3 implies a simple estimation strategy based on weighted regressions. For instance, under (29.8), we can fit WLS of $Y _ { i }$ on $( 1 , Z _ { 1 i } , Z _ { 2 i } , X _ { 0 i } )$ with weights $\hat { e } _ { 1 } ^ { - 1 } ( Z _ { 1 i } , X _ { 0 i } ) \hat { e } _ { 2 i } ^ { - 1 } ( Z _ { 2 i } , Z _ { i 1 } , X _ { 1 i } , X _ { 0 i } )$ .

## 29.4.2 Structural nested model

A key problem of IPW is that it is not applicable if the overlap assumption is violated. To address this challenge, Robins proposed the structural nested model. Again, to simplify the presentation, I only review the version with two time points.

Definition 29.3 (structural nested model) The conditional effect at time point 1 is

$$
E \{Y (z _ {1}, 0) - Y (0, 0) \mid Z _ {1} = z _ {1}, X _ {0} \} = g _ {1} (z _ {1}, X _ {0}; \beta) f o r a l l z _ {1}
$$

and the conditional effect at time point 2 is

$$
E \{Y (z _ {1}, z _ {2}) - Y (z _ {1}, 0) \mid Z _ {2} = z _ {2}, Z _ {2} = z _ {1}, X _ {1}, X _ {0} \} = g _ {2} (z _ {2}, z _ {1}, X _ {1}, X _ {0}; \beta) f o r a l l z _ {1}, z _ {2}.
$$

In Definition 29.3, two logical restrictions are

$$
g _ {1} (0, X _ {0}; \beta) = 0
$$

and

$$
g _ {2} (0, z _ {1}, X _ {1}, X _ {0}; \beta) = 0 \text {   for   all   } z _ {1}.
$$

Two leading choices of Definition 29.3 are below.

Example 29.2 Assume

$$
\left\{ \begin{array}{l} g _ {1} (z _ {1}, X _ {0}; \beta) = \beta_ {1} z _ {1}, \\ g _ {2} (z _ {2}, z _ {1}, X _ {1}, X _ {0}; \beta) = (\beta_ {2} + \beta_ {3} z _ {1}) z _ {2}. \end{array} \right.
$$

Example 29.3 Assume

$$
\left\{ \begin{array}{l} g _ {1} (z _ {1}, X _ {0}; \beta) = (\beta_ {1} + \beta^ {\mathsf {T}} X _ {0}) z _ {1}, \\ g _ {2} (z _ {2}, z _ {1}, X _ {1}, X _ {0}; \beta) = (\beta_ {2} + \beta_ {3} z _ {1} + \beta_ {4} ^ {\mathsf {T}} X _ {1}) z _ {2}. \end{array} \right.
$$

Compare Definitions 29.2 and 29.3. The structural nested model allows for adjusting for the time-varying covariates whereas the marginal structural model only allows for adjusting for baseline covariates. The estimation under Definition 29.3 is more involved. A strategy is to estimate the parameter based on estimating equations.

I first introduce two important building blocks for the discussing the estimation. Define

$$
U _ {2} (\beta) = Y - g _ {2} (Z _ {2}, Z _ {1}, X _ {1}, X _ {0}; \beta)
$$

and

$$
U _ {1} (\beta) = Y - g _ {2} (Z _ {2}, Z _ {1}, X _ {1}, X _ {0}; \beta) - g _ {1} (Z _ {1}, X _ {0}; \beta).
$$

They are not directly computable from the data because they depend on the true value of the parameter $\beta .$ At the true value, they have the following properties.

Lemma 29.1 Under Assumption 29.1 and Definition 29.3, we have

$$
\begin{array}{l} E \{U _ {2} (\beta) \mid Z _ {2}, Z _ {1}, X _ {1}, X _ {0} \} = E \{U _ {2} (\beta) \mid Z _ {1}, X _ {1}, X _ {0} \} \\ = E \left\{Y \left(Z _ {1}, 0\right) \mid Z _ {1}, X _ {1}, X _ {0} \right\} \\ \end{array}
$$

and

$$
\begin{array}{l} E \{U _ {1} (\beta) \mid Z _ {1}, X _ {0} \} = E \{U _ {1} (\beta) \mid X _ {0} \} \\ = E \{Y (0, 0) \mid X _ {0} \}. \\ \end{array}
$$

Lemma 29.1 involves a subtle notation $Y ( Z _ { 1 } , 0 )$ because $Z _ { 1 }$ is random. It should be read as $Y ( Z _ { 1 } , 0 ) = Z _ { 1 } Y ( 1 , 0 ) + ( 1 - Z _ { 1 } ) Y ( 0 , 0 )$ . Based on the definitions and Lemma 29.1, $U _ { 1 } ( \beta )$ acts as the control potential outcome before receiving any treatment and $U _ { 2 } ( \beta )$ acts as the control potential outcome after receiving the treatment at time point 1.

Proof of Lemma 29.1: First, we have

$$
\begin{array}{l} E \{U _ {2} (\beta) \mid Z _ {2} = 1, Z _ {1}, X _ {1}, X _ {0} \} = E \{Y (Z _ {1}, 1) - g _ {2} (1, Z _ {1}, X _ {1}, X _ {0}; \beta) \mid Z _ {2} = 1, Z _ {1}, X _ {1}, X _ {0} \} \\ = E \left\{Y \left(Z _ {1}, 0\right) \mid Z _ {2} = 1, Z _ {1}, X _ {1}, X _ {0} \right\} \\ E \{U _ {2} (\beta) \mid Z _ {2} = 0, Z _ {1}, X _ {1}, X _ {0} \} = E \{Y (Z _ {1}, 0) - g _ {2} (0, Z _ {1}, X _ {1}, X _ {0}; \beta) \mid Z _ {2} = 0, Z _ {1}, X _ {1}, X _ {0} \} \\ = E \left\{Y \left(Z _ {1}, 0\right) \mid Z _ {2} = 0, Z _ {1}, X _ {1}, X _ {0} \right\} \\ \end{array}
$$

so

$$
E \{U _ {2} (\beta) \mid Z _ {2}, Z _ {1}, X _ {1}, X _ {0} \} = E \{Y (Z _ {1}, 0) \mid Z _ {2}, Z _ {1}, X _ {1}, X _ {0} \} = E \{Y (Z _ {1}, 0) \mid Z _ {1}, X _ {1}, X _ {0} \}
$$

where the last identity follows from sequential ignorability. Since the last term does not depend on $Z _ { 2 } ,$ we also have

$$
E \{U _ {2} (\beta) \mid Z _ {2}, Z _ {1}, X _ {1}, X _ {0} \} = E \{U _ {2} (\beta) \mid Z _ {1}, X _ {1}, X _ {0} \}.
$$

Using the above results, we have

$$
\begin{array}{l} E \{U _ {1} (\beta) \mid Z _ {1}, X _ {0} \} = E \{U _ {2} (\beta) - g _ {1} (Z _ {1}, X _ {0}; \beta) \mid Z _ {1}, X _ {0} \} \\ = E \left[ E \left\{U _ {2} (\beta) - g _ {1} \left(Z _ {1}, X _ {0}; \beta\right) \mid X _ {1}, Z _ {1}, X _ {0} \right\} \mid Z _ {1}, X _ {0} \right] \\ = E \left[ E \left\{Y \left(Z _ {1}, 0\right) - g _ {1} \left(Z _ {1}, X _ {0}; \beta\right) \mid X _ {1}, Z _ {1}, X _ {0} \right\} \mid Z _ {1}, X _ {0} \right] \\ = E \{Y (Z _ {1}, 0) - g _ {1} (Z _ {1}, X _ {0}; \beta) \mid Z _ {1}, X _ {0} \} \\ = E \{Y (0, 0) \mid Z _ {1}, X _ {0} \} \\ = E \{Y (0, 0) \mid X _ {0} \} \\ \end{array}
$$

where the last identity follows from sequential ignorability. Since the last term does not depend on $Z _ { 1 }$ , we also have

$$
E \{U _ {1} (\beta) \mid Z _ {1}, X _ {0} \} = E \{U _ {1} (\beta) \mid X _ {0} \}.
$$

With Lemma 29.1, we can prove Theorem 29.4 below.

Theorem 29.4 Under Assumption 29.1 and Definition 29.3,

$$
E \Big [ h _ {2} (Z _ {1}, X _ {1}, X _ {0}) \{Z _ {2} - e (1, Z _ {1}, X _ {1}, X _ {0}) \} U _ {2} (\beta) \Big ] = 0
$$

and

$$
E \Big [ h _ {1} (X _ {0}) \{Z _ {1} - e (1, X _ {0}) \} U _ {1} (\beta) \Big ] = 0.
$$

for any functions $h _ { 1 }$ and $h _ { 2 } ,$ provided that the moments exist.

Proof of Theorem 29.2: Use the tower property by conditioning on $\left( Z _ { 2 } , Z _ { 1 } , X _ { 1 } , X _ { 0 } \right)$ and Lemma 29.1 to obtain

$$
\begin{array}{l} E \left[ h _ {2} (Z _ {1}, X _ {1}, X _ {0}) \{Z _ {2} - e (1, Z _ {1}, X _ {1}, X _ {0}) \} E \{U _ {2} (\beta) \mid Z _ {2}, Z _ {1}, X _ {1}, X _ {0} \} \right] \\ = E \left[ h _ {2} (Z _ {1}, X _ {1}, X _ {0}) \{Z _ {2} - e (1, Z _ {1}, X _ {1}, X _ {0}) \} E \{U _ {2} (\beta) \mid Z _ {1}, X _ {1}, X _ {0} \} \right]. \\ \end{array}
$$

Use the tower property by conditioning on $( Z _ { 1 } , X _ { 1 } , X _ { 0 } )$ to show that the last identity equals 0.

Similarly, use the tower property by conditioning on $( Z _ { 1 } , X _ { 0 } )$ and Lemma 29.1 to obtain

$$
\begin{array}{l} E \left[ h _ {1} (X _ {0}) \{Z _ {1} - e (1, X _ {0}) \} E \{U _ {1} (\beta) \mid Z _ {1}, X _ {0} \} \right] \\ = E \left[ h _ {1} (X _ {0}) \{Z _ {1} - e (1, X _ {0}) \} E \{U _ {1} (\beta) \mid X _ {0} \} \right]. \\ \end{array}
$$

Use the tower property by conditioning on $X _ { 0 }$ to show that the last identity equals 0. □

To use Theorem 29.4, we must specify $h _ { 1 }$ and $h _ { 2 }$ to ensure that there are enough equations for solving $\beta .$ Example 29.4 below revisits Example 29.2.

Example 29.4 Under Example ${ \it 2 9 . 2 , }$ we can choose $h _ { 1 } = 1$ and $h _ { 2 } = \left( 1 , Z _ { 1 } \right)$ 号 to obtain

$$
\begin{array}{l} E \left[ \left\{Z _ {2} - e (1, Z _ {1}, X _ {1}, X _ {0}) \right\} \left\{Y - (\beta_ {2} + \beta_ {3} Z _ {1}) Z _ {2} \right\} \right] = 0, \\ E \left[ Z _ {1} \left\{Z _ {2} - e \left(1, Z _ {1}, X _ {1}, X _ {0}\right) \right\} \left\{Y - \left(\beta_ {2} + \beta_ {3} Z _ {1}\right) Z _ {2} \right\} \right] = 0, \\ E \left[ \{Z _ {1} - e (1, X _ {0}) \} \{Y - (\beta_ {2} + \beta_ {3} Z _ {1}) Z _ {2} - \beta_ {1} Z _ {1} \} \right] = 0. \\ \end{array}
$$

We can then solve for the $\beta ^ { \prime } s$ from the above linear equations; see Problem 29.5. A natural question is that whether alternative choices of $( h _ { 1 } , h _ { 2 } )$ can lead to more efficient estimators. The answer is yes. For example, we can choose many $( h _ { 1 } , h _ { 2 } )$ and use the generalized method of moment (Hansen, 1982). The technical details are beyond this book.

Naimi et al. (2017) and Vansteelandt and Joffe (2014) provided tutorials on the structural nested models.

![image_39](images/image_39.png)

```mermaid
graph TD
  Z1 --> X1
  X1 --> Z2
  Z2 --> Y
  X1 --> U
  Y --> U
```

FIGURE 29.3: With unmeasured confounding between $X _ { 1 }$ and $Y .$ . The causal diagram ignores the pre-treatment covariates $X _ { 0 }$ .

## 29.5 Homework problems

## 29.1 g-null paradox

Consider the simple causal diagram in Figure 29.3 without pre-treatment covariates $X _ { 0 }$ and without the arrows from $( Z _ { 1 } , Z _ { 2 } )$ to $Y$ . So the effect of $( Z _ { 1 } , Z _ { 2 } )$ on $Y$ is zero.

Revisit Example 29.1. Show that the expectation $E \{ Y ( z _ { 1 } , z _ { 2 } ) \}$ does not depend on $( z _ { 1 } , z _ { 2 } )$ if

$$
\beta_ {1} = \beta_ {3} = 0 \mathrm{and} \beta_ {2} = 0
$$

or

$$
\beta_ {1} = \beta_ {3} = 0 \text {   and   } E \{X _ {1} (z _ {1}) \} \text {   does   not   depend   on   } z _ {1}.
$$

holds.

Remark: However, $\beta _ { 2 } = 0$ in the first condition rules out the dependence of $Y$ on $X _ { 1 } .$ , contradicting with the existence of unmeasured confounder U between $X _ { 1 }$ and $Y ;$ the independence of $E \{ X _ { 1 } ( z _ { 1 } ) \}$ } on $z _ { 1 }$ rules out the dependence of $X _ { 1 }$ on $Z _ { 1 }$ , contradicting with the existence of the arrow from $Z _ { 1 }$ on $X _ { 1 }$ . That is, if there is an unmeasured confounder $U$ between $X _ { 1 }$ and Y and there is an arrow from $Z _ { 1 }$ on $X _ { 1 }$ , then the formula of $E \{ Y ( z _ { 1 } , z _ { 2 } ) \}$ in Example 29.1 must depend on $( z _ { 1 } , z _ { 2 } )$ , which leads to a contradiction with the absence of arrows from $( Z _ { 1 } , Z _ { 2 } )$ to $Y$ .

## 29.2 Recursive estimation under the null model

Consider the recursive estimation method in 29.2.2 under the causal diagram in Problem 29.1. Show that based linear models, the estimator converges to 0.

## 29.3 IPW under MSM

Prove Theorem 29.3.

## 29.4 Structural nested model with a single time point

Recall the standard setting of observational studies with IID data draw from $\{ X , Z , Y ( 1 ) , Y ( 0 ) \}$ }. Define the propensity score as $e ( X ) \ = \ \mathrm { p r } ( Z = 1 \mid X )$ . Assume

$$
Z \bot Y (0) \mid X
$$

and the following structural nested model.

Definition 29.4 (structural nested model with a single time point) The conditional mean of the individual effect is

$$
E \{Y (z) - Y (0) \mid Z = z, X \} = g (z, X; \beta).
$$

In Definition 29.4, a logical restriction is $g ( 0 , X ; \beta ) = 0$ . Prove the following results.

1. We have

$$
E \{Y - g (Z, X; \beta) \mid X, Z \} = E \{Y - g (Z, X; \beta) \mid X \} = E \{Y (0) \mid X \}.
$$

2. We have

$$
E \Big [ h (X) \{Z - e (X) \} \{Y - g (Z, X; \beta) \} \Big ] = 0 \tag {29.9}
$$

for any function $h ,$ provided that the moment exists.

Remark: (29.9) is the basis for parameter estimation. Consider a special case of Definition 29.4 with $g ( z , X ; \beta ) = \beta z$ . Choose $h ( X ) = 1$ to obtain

$$
E \{(Z - e (X)) (Y - \beta Z) \} = 0.
$$

Solve for $\beta$ to obtain

$$
\beta = \frac {E \{(Z - e (X)) Y \}}{E \{(Z - e (X)) Z \}}.
$$

That is, $\beta$ equals the coefficient of $Z$ in the two-stage least squares of $Y$ on $Z$ with $Z - e ( X )$ being the instrument variable for $Z$

Consider a special case of Definition 29.4 with $g ( z , X ; \beta ) = ( \beta _ { 0 } + \beta _ { 1 } ^ { \mathsf { T } } X ) z$ . Choose $h ( X ) = ( 1 , X )$ to obtain

$$
E \left\{\binom{Z - e (X)}{(Z - e (X)) X} (Y - \beta_ {0} Z - \beta_ {1} ^ {\mathsf {T}} X Z) \right\} = 0.
$$

That is, $( \beta _ { 0 } , \beta _ { 1 } )$ equal the coefficients in the two-stage least squares of $Y$ on $( Z , X Z )$ with $( \bar { Z } - e ( X ) , ( Z - e ( X ) ) X )$ being the instrument variable for $( Z , X Z )$ .

## 29.5 Estimation under Example 29.4

We can estimate the $\beta \mathrm { { ^ { * } s } }$ by solving the empirical version of the estimating equations in Example 29.4. We first estimate the two propensity scores and obtain the centered treatment

$$
\check {Z} _ {1 i} = Z _ {1 i} - \hat {e} (1, X _ {0 i})
$$

at time point 1 and

$$
\check {Z} _ {2 i} = Z _ {2 i} - \hat {e} (1, Z _ {1 i}, X _ {1 i}, X _ {0 i})
$$

at time point 2.

Show that we can estimate $\beta _ { 2 }$ and $\beta _ { 3 }$ by running two-stage least squares of $Y _ { i }$ on $\left( Z _ { 2 i } , Z _ { 1 i } Z _ { 2 i } \right)$ with $( \check { Z } _ { 2 i } , Z _ { 1 i } \check { Z } _ { 2 i } )$ as the instrumental variable for $\left( Z _ { 2 i } , Z _ { 1 i } Z _ { 2 i } \right)$ , and then we can estimate $\beta _ { 1 }$ by running two-stage least squares of $Y _ { i } - ( \hat { \beta } _ { 2 } + \hat { \beta } _ { 3 } Z _ { 1 i } ) Z _ { 2 i }$ on $Z _ { 1 i }$ with $\check { Z } _ { 1 i }$ as the instrumental variable for $Z _ { 1 i }$ .

## 29.6 g-formula with a treatment at multiple time points

Extend the discussion to the setting with K time points. The temporal ordering of the variables is

$$
X _ {0} \rightarrow Z _ {1} \rightarrow X _ {1} \rightarrow Z _ {2} \rightarrow \dots X _ {K - 1} \rightarrow Z _ {K}.
$$

Introduce the notation ${ \overline { { Z } } } _ { k } = ( Z _ { 1 } , \ldots , Z _ { k } )$ and ${ \overline { { X } } } _ { k } = ( X _ { 0 } , X _ { 1 } , \dots , X _ { k } )$ with lower case $\overline { { z } } _ { k }$ and $\overline { { x } } _ { k }$ denoting the corresponding realized values. With $k = 0$ , we have ${ \overline { { X } } } _ { 0 } = X _ { 0 }$ and $\overline { { Z } } _ { 0 }$ is empty. Each unit has $2 ^ { K }$ potential outcomes:

$$
Y (\overline {{z}} _ {K}) \text {   for   all   } z _ {1}, \ldots , z _ {K} = 0, 1.
$$

Assume sequential ignorability below.

Assumption 29.2 (sequential ignorability at multiple time points) We have

$$
Z _ {k} \bot Y (\overline {{z}} _ {K}) \mid (\overline {{Z}} _ {k - 1}, \overline {{X}} _ {k - 1})
$$

for all $k = 1 , \ldots , K$ and all $z _ { 1 } , \dotsc , z _ { K } = 0 , 1$ .

Prove Theorem 29.5 below.

Theorem 29.5 (g-formula with multiple time points) Under Assumption 29.2,

$$
E \{Y (\overline {{z}} _ {K}) \} = E \left[ \dots E \{E (Y \mid \overline {{z}} _ {K}, \overline {{X}} _ {K - 1}) \mid \overline {{z}} _ {K - 1}, \overline {{X}} _ {K - 2} \} \dots \mid z _ {1}, X _ {0} \right].
$$

Remark: In Theorem 29.5, I use the simplified notation $\overline { { z } } _ { k } \overrightarrow { \mathbf { \Gamma } }$ for $\sqrt [ 6 ] { Z } _ { k } = \overline { { z } } _ { k } . \overline { { \jmath } } ^ { \vphantom { \dag } }$ With discrete X, Theorem 29.5 reduces to

$$
\begin{array}{l} E \{Y (\overline {{z}} _ {K}) \} = \sum_ {x _ {0}} \sum_ {x _ {1}} \dots \sum_ {x _ {K - 1}} E (Y | \overline {{z}} _ {K}, \overline {{x}} _ {K - 1}) \\ \cdot \operatorname{pr} (x _ {K - 1} \mid \overline {{z}} _ {K - 1}, \overline {{x}} _ {K - 2}) \dots \operatorname{pr} (x _ {1} \mid z _ {1}, x _ {0}) \operatorname{pr} (x _ {0}); \\ \end{array}
$$

with continuous X, Theorem 29.5 reduces to

$$
\begin{array}{l} E \{Y (\overline {{z}} _ {K}) \} = \int E (Y | \overline {{z}} _ {K}, \overline {{x}} _ {K - 1}) \\ \cdot f (x _ {K - 1} \mid \overline {{z}} _ {K - 1}, \overline {{x}} _ {K - 2}) \dots f (x _ {1} \mid z _ {1}, x _ {0}) f (x _ {0}) \mathrm{d} \overline {{x}} _ {K - 1}. \\ \end{array}
$$

## 29.7 IPW with a treatment at multiple time points

Inherit the setting of Problem 29.6. Define the propensity score at K time points as

$$
\begin{array}{l} e (z _ {1}, X _ {0}) = \operatorname{pr} (Z _ {1} = z _ {1} \mid X _ {0}), \\ e (z _ {k}, \overline {{Z}} _ {k - 1}, \overline {{X}} _ {k - 1}) = \operatorname{pr} (Z _ {k} = z _ {k} \mid \overline {{Z}} _ {k - 1}, \overline {{X}} _ {k - 1}), \\ e (z _ {K}, \overline {{Z}} _ {K - 1}, \overline {{X}} _ {K - 1}) = \operatorname{pr} (Z _ {K} = z _ {K} \mid \overline {{Z}} _ {K - 1}, \overline {{X}} _ {K - 1}). \\ \end{array}
$$

Prove Theorem 29.7 below assuming overlap implicitly.

Theorem 29.6 (IPW with multiple time points) Under Assumption 29.2,

$$
E \{Y (\overline {{z}} _ {K}) \} = E \left\{\frac {1 (Z _ {1} = z _ {1}) \cdots 1 (Z _ {K} = z _ {K}) Y}{e (z _ {1} , X _ {0}) \cdots e (z _ {K} , \overline {{Z}} _ {K - 1} , \overline {{X}} _ {K - 1})} \right\}.
$$

Based on Theorem 29.7, construct the Horvitz–Thompson and Hajek estimators.

## 29.8 MSM with a treatment at multiple time points

The number of potential outcomes grows exponentially with K. The formulas in Problems 29.6 and 29.7 are not directly applicable in finite samples. We can impose the following structural assumptions on the potential outcomes.

Definition 29.5 (MSM with multiple time points) Assume

$$
E \{Y (\overline {{z}} _ {K}) \mid X _ {0} \} = f (\overline {{z}} _ {K}, X _ {0}; \beta).
$$

Two leading examples of Definition 29.5 are $E \{ Y ( \overline { { { z } } } _ { K } ) ~ \mid ~ X _ { 0 } \} ~ = ~ \beta _ { 0 } ~ +$ $\beta _ { 1 } \sum _ { k = 1 } ^ { K } z _ { k } + \beta _ { 2 } ^ { \mathsf { T } } X _ { 0 }$ and $\begin{array} { r } { E \{ Y ( \overline { { z } } _ { K } ) \mid X _ { 0 } \} = \beta _ { 0 } + \sum _ { k = 1 } ^ { K } \beta _ { k } z _ { k } + \beta _ { K + 1 } ^ { \mathsf { T } } X _ { 0 } } \end{array}$ .

If we know all the potential outcomes, we can solve $\beta$ from the following minimization problem:

$$
\beta = \arg \min _ {b} \sum_ {\overline {{z}} _ {K}} E \{Y (\overline {{z}} _ {K}) - f (\overline {{z}} _ {K}, X _ {0}; \beta) \} ^ {2}.
$$

Theorem 29.7 below shows that under Assumption 29.2, we can solve $\beta$ from a minimization problem that only involves observables.

Theorem 29.7 (IPW for MSM with multiple time points) Under Assumption 29.2,

$$
\beta = \arg \min _ {b} \sum_ {\overline {{z}} _ {K}} E \left[ \frac {1 (Z _ {1} = z _ {1}) \cdots 1 (Z _ {K} = z _ {K})}{e (z _ {1} , X _ {0}) \cdots e (z _ {K} , \overline {{Z}} _ {K - 1} , \overline {{X}} _ {K - 1})} \{Y - f (\overline {{z}} _ {K}, X _ {0}; \beta) \} ^ {2} \right].
$$

## 29.9 Structural nested model with a treatment at multiple time points

Inherit the setting from Problem 29.6 and the notation from Problem 29.7. This problem presents a general structural nested model.

## Definition 29.6 (structural nested model with multiple time points)

The conditional effect at time k is

$$
E \left\{Y (\overline {{z}} _ {k}, 0) - Y (\overline {{z}} _ {k - 1}, 0) \mid \overline {{z}} _ {k}, \overline {{X}} _ {k - 1} \right\} = g _ {k} (\overline {{z}} _ {k}, \overline {{X}} _ {k - 1}; \beta)
$$

for all $\overline { { z } } _ { k }$ and all $k = 1 , \ldots , K$ .

In Definition 29.6, a logical restriction is

$$
g _ {k} (0, \overline {{z}} _ {k - 1}, \overline {{X}} _ {k - 1}; \beta) = 0
$$

for all $\overline { { z } } _ { k - 1 }$ and all $k = 1 , \ldots , K .$ .

Define

$$
U _ {k} (\beta) = Y - \sum_ {s = 1} ^ {k} g _ {s} (\overline {{Z}} _ {s}, \overline {{X}} _ {s - 1}; \beta)
$$

for all $k = 1 , \ldots , K$ . Theorem 29.8 below extends Theorem 29.4.

Theorem 29.8 Under Assumption 29.2 and Definition 29.6,

$$
E \left[ h _ {k} (\overline {{Z}} _ {k - 1}, \overline {{X}} _ {k - 1}) \{Z _ {k} - e (1, \overline {{Z}} _ {k - 1}, \overline {{X}} _ {k - 1}) \} U _ {k} (\beta) \right] = 0
$$

for all $k = 1 , \ldots , K$ .

Remark: Choosing appropriate $h _ { k } \mathrm { ' s } .$ , we can estimate $\beta$ by solving the empirical version of Theorem 29.8.

## 29.10 Recommended reading

Robins et al. (2000) reviewed the MSM. Naimi et al. (2017) reviewed the g-methods.

## Part VII

## Appendices

## A1

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

# A3 Some Useful Lemmas for Simple Random Sampling

## A3.1 Lemmas

Simple random sampling is a basic topic in standard survey sampling textbooks $( \mathrm { e . g . }$ , Cochran, 1953). Below I review some results for simple random sampling that are useful for design-based inference in the CRE in Chapters 3 and 4.

A simple random sample of size $n _ { 1 }$ consists of a subset from a finite population of n units indexed by $i = 1 , \ldots , n$ . Let $\pmb { Z } = ( Z _ { 1 } , \ldots , Z _ { n } )$ be the inclusion indicators of the n units with $Z _ { i } = 1$ if unit i is sampled and $Z _ { i } = 0$ otherwise. The vector $z$ can take $\scriptstyle { \binom { n } { n _ { 1 } } }$ possible permutations of a vector of $n _ { 1 }$ 1’s and $n _ { 0 } \mathrm { ~ } 0 \mathrm { { ' s } }$ , and each has equal probability. The following lemma summarizes the first two moments of the inclusion indicators.

Lemma A3.1 Under simple random sampling, we have

$$
E (Z _ {i}) = \frac {n _ {1}}{n}, \quad \operatorname{var} (Z _ {i}) = \frac {n _ {1} n _ {0}}{n ^ {2}}, \quad \operatorname{cov} (Z _ {i}, Z _ {j}) = - \frac {n _ {1} n _ {0}}{n ^ {2} (n - 1)}.
$$

In more compact forms, we have

$$
E (\mathbf {Z}) = \frac {n _ {1}}{n} \mathbf {1} _ {n}, \quad \operatorname{cov} (\mathbf {Z}) = \frac {n _ {1} n _ {0}}{n (n - 1)} \mathbf {P} _ {n},
$$

where ${ \bf 1 } _ { n }$ is a n-dimensional vector of 1’s, and $P _ { n } = I _ { n } - n ^ { - 1 } \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top }$ is the $n \times n$ projection matrix orthogonal to $\mathbf { 1 } _ { n } .$ .

Let $\{ c _ { 1 } , \ldots , c _ { n } \}$ be a finite population with mean $\textstyle { \bar { c } } = \sum _ { i = 1 } ^ { n } c _ { i } / n$ and variance

$$
S _ {c} ^ {2} = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} (c _ {i} - \bar {c}) ^ {2};
$$

let $\{ d _ { 1 } , \ldots , d _ { n } \}$ be another finite population with mean $\textstyle { \bar { d } } = \sum _ { i = 1 } ^ { n } d _ { i } / n$ and variance

$$
S _ {d} ^ {2} = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} (d _ {i} - \bar {d}) ^ {2};
$$

their covariance is

$$
S _ {c d} = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} (c _ {i} - \bar {c}) (d _ {i} - \bar {d}).
$$

Based on the simple random sample, the sample means are

$$
\hat {\bar {c}} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} c _ {i}, \quad \hat {\bar {d}} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} d _ {i};
$$

sample variances are

$$
\hat {S} _ {c} ^ {2} = (n _ {1} - 1) ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} (c _ {i} - \hat {c}) ^ {2}, \quad \hat {S} _ {d} ^ {2} = (n _ {1} - 1) ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} (d _ {i} - \hat {\bar {d}}) ^ {2};
$$

the sample covariance is

$$
\hat {S} _ {c d} = (n _ {1} - 1) ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} (c _ {i} - \hat {\bar {c}}) (d _ {i} - \hat {\bar {d}}).
$$

Lemma A3.2 below gives the moments of the sample means $\hat { \bar { c } }$ and $\hat { \bar { d } } .$

Lemma A3.2 The sample means are unbiased for the population means:

$$
E (\hat {\bar {c}}) = \bar {c}, \quad E (\hat {\bar {d}}) = \bar {d}.
$$

Their variances and covariance are

$$
\mathrm{var} \left(\hat {\bar {c}}\right) = \frac {n _ {0}}{n n _ {1}} S _ {c} ^ {2}, \quad \mathrm{var} \left(\hat {\bar {d}}\right) = \frac {n _ {0}}{n n _ {1}} S _ {d} ^ {2}, \quad \mathrm{cov} \left(\hat {\bar {c}}, \hat {\bar {d}}\right) = \frac {n _ {0}}{n n _ {1}} S _ {c d}.
$$

In the variance formula in Lemma A3.2, the coefficient $n _ { 0 } / ( n n _ { 1 } ) = 1 / n _ { 1 } \times$ $\left( 1 - n _ { 1 } / n \right)$ in Lemma A3.2 is different from $1 / n _ { 1 }$ under IID sampling. The additional factor $1 - n _ { 1 } / n = n _ { 0 } / n$ is called the finite population correction.

Lemma A3.3 below gives the unbiasedness of the sample variances and covariance for estimating the population analogs.

Lemma A3.3 The sample variances and covariance are unbiased for their population versions:

$$
E (\hat {S} _ {c} ^ {2}) = S _ {c} ^ {2}, \quad E (\hat {S} _ {d} ^ {2}) = S _ {d} ^ {2}, \quad E (\hat {S} _ {c d}) = S _ {c d}.
$$

An important practical question is to make inference about ¯c based on the simple random sample. This requires a more precise characterization of the distribution of its unbiased estimator cˆ¯. The finite-sample exact distribution of cˆ¯ depends on the whole finite population $\{ c _ { 1 } , \ldots , c _ { n } \}$ , which is intractable in general. The following finite population central limit theorem characterizes the asymptotic distribution of cˆ¯ based on its first two moments.

Lemma A3.4 (finite population central limit theorem) As $n  \infty ,$ , if

$$
\frac {\max _ {1 \leq i \leq n} (c _ {i} - \bar {c}) ^ {2}}{\min (n _ {1} , n _ {0}) S _ {c} ^ {2}} \to 0,
$$

then

$$
\frac {\hat {\bar {c}} - \bar {c}}{\sqrt {\frac {n _ {0}}{n n _ {1}} S _ {c} ^ {2}}} \to \mathrm{N} (0, 1)
$$

in distribution, and $\hat { S } _ { c } ^ { 2 } / S _ { c } ^ { 2 }  1$ in probability.

Lemma A3.4 justifies the Wald-type $1 - \alpha$ confidence interval for ¯c:

$$
\hat {\bar {c}} \pm z _ {1 - \alpha / 2} \sqrt {\frac {n _ {0}}{n n _ {1}} \hat {S} _ {c} ^ {2}}
$$

where $z _ { 1 - \alpha / 2 }$ is the $1 - \alpha / 2$ upper quantile of the standard Normal random variable.

## A3.2 Proofs

Proof of Lemma A3.1: By symmetry, the $Z _ { i } \mathrm { ^ { * } s }$ have the same mean, so

$$
n _ {1} = \sum_ {i = 1} ^ {n} Z _ {i} = E \left(\sum_ {i = 1} ^ {n} Z _ {i}\right) = n E (Z _ {i}) \Longrightarrow E (Z _ {i}) = n _ {1} / n.
$$

Because $Z _ { i }$ is a Bernoulli random variable, its variance is

$$
\mathrm{var} (Z _ {i}) = \frac {n _ {1}}{n} \left(1 - \frac {n _ {1}}{n}\right) = \frac {n _ {1} n _ {0}}{n ^ {2}}.
$$

By symmetry again, the $Z _ { i } \mathrm { ^ { * } s }$ have the same variance and the pairs $( Z _ { i } , Z _ { j } ) ^ { } \mathrm { { s } }$ have the same covariance, so

$$
0 = \operatorname{var} \left(\sum_ {i = 1} ^ {n} Z _ {i}\right) = n \operatorname{var} (Z _ {i}) + n (n - 1) \operatorname{cov} (Z _ {i}, Z _ {j})
$$

which implies that

$$
\operatorname{cov} (Z _ {i}, Z _ {j}) = - \frac {n _ {1} n _ {0}}{n ^ {2} (n - 1)} \quad (i \neq j).
$$

□

Proof of Lemma A3.1: The unbiasedness of the sample mean follows from linearity. For example,

$$
E (\hat {\bar {c}}) = E \left(\frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} c _ {i}\right) = \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} E (Z _ {i}) c _ {i} = \bar {c}.
$$

The covariance of the sample means is

$$
\begin{array}{l} \operatorname{cov} (\hat {\bar {c}}, \hat {\bar {d}}) \\ = \operatorname{cov} \left\{\frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} (c _ {i} - \bar {c}), \frac {1}{n _ {1}} \sum_ {i = 1} ^ {n} Z _ {i} (d _ {i} - \bar {d}) \right\} \\ { = } { \frac { 1 } { n _ { 1 } ^ { 2 } } \left[ \sum _ { i = 1 } ^ { n } \mathrm{var} ( Z _ { i } ) ( c _ { i } - \bar { c } ) ( d _ { i } - \bar { d } ) + \sum _ { i \neq j } \mathrm{cov} ( Z _ { i } , Z _ { j } ) ( c _ { i } - \bar { c } ) ( d _ { j } - \bar { d } ) \right] } \\ { = } { \frac { 1 } { n _ { 1 } ^ { 2 } } \left[ \frac { n _ { 1 } n _ { 0 } } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } ( c _ { i } - \bar { c } ) ( d _ { i } - \bar { d } ) - \frac { n _ { 1 } n _ { 0 } } { n ^ { 2 } ( n - 1 ) } \sum _ { i \neq j } ( c _ { i } - \bar { c } ) ( d _ { j } - \bar { d } ) \right] . } \\ \end{array}
$$

Because

$$
0 = \sum_ {i = 1} ^ {n} (c _ {i} - \bar {c}) \sum_ {i = 1} ^ {n} (d _ {i} - \bar {d}) = \sum_ {i = 1} ^ {n} (c _ {i} - \bar {c}) (d _ {i} - \bar {d}) + \sum_ {i \neq j} (c _ {i} - \bar {c}) (d _ {j} - \bar {d}),
$$

the covariance of the sample means reduces to

$$
\begin{array}{l} \operatorname{cov} (\hat {\bar {c}}, \hat {\bar {d}}) \\ { = } { \frac { 1 } { n _ { 1 } ^ { 2 } } \left[ \frac { n _ { 1 } n _ { 0 } } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } ( c _ { i } - \bar { c } ) ( d _ { i } - \bar { d } ) + \frac { n _ { 1 } n _ { 0 } } { n ^ { 2 } ( n - 1 ) } \sum _ { i = 1 } ^ { n } ( c _ { i } - \bar { c } ) ( d _ { i } - \bar { c } ) \right] } \\ = \frac {n _ {0}}{n n _ {1}} S _ {c d}. \\ \end{array}
$$

The variance formulas are just special cases with $\hat { \bar { c } } = \hat { \bar { d } } .$

Proof of Lemma A3.3: We prove only the sample covariance term, because the formulas for sample variances are special cases. We have the following decomposition:

$$
\begin{array}{l} (n _ {1} - 1) \hat {S} _ {c d} = \sum_ {i = 1} ^ {n} Z _ {i} (c _ {i} - \hat {\bar {c}}) (d _ {i} - \hat {\bar {d}}) \\ = \sum_ {i = 1} ^ {n} Z _ {i} \{(c _ {i} - \bar {c}) - (\hat {\bar {c}} - \bar {c}) \} \{(d _ {i} - \bar {d}) - (\hat {\bar {d}} - \bar {d}) \} \\ = \sum_ {i = 1} ^ {n} Z _ {i} (c _ {i} - \bar {c}) (d _ {i} - \bar {d}) - n _ {1} (\hat {\bar {c}} - \bar {c}) (\hat {\bar {d}} - \bar {d}). \\ \end{array}
$$

Taking expectation on both sides, we have

$$
\begin{array}{l} E \{(n _ {1} - 1) \hat {S} _ {c d} \} = \sum_ {i = 1} ^ {n} E (Z _ {i}) (c _ {i} - \bar {c}) (d _ {i} - \bar {d}) - n _ {1} E \{(\hat {\bar {c}} - \bar {c}) (\hat {\bar {d}} - \bar {d}) \} \\ = \frac {n _ {1}}{n} \sum_ {i = 1} ^ {n} (c _ {i} - \bar {c}) (d _ {i} - \bar {d}) - n _ {1} \frac {n _ {0}}{n n _ {1}} S _ {c d} \\ = S _ {c d} \left\{\frac {n _ {1} (n - 1)}{n} - \frac {n _ {0}}{n} \right\} \\ = (n _ {1} - 1) S _ {c d}, \\ \end{array}
$$

and the conclusion follows by dividing both sides by $n _ { 1 } - 1 { }$ .

Proof of Lemma A3.4: H´ajek (1960) gave a proof of the central limit theorem for simple random sampling, and Lehmann (1975) gave a more accessible version of the proof. Li and Ding (2017) modified the central limit theorem as presented in Lemma A3.4, and proved the consistency of the sample variance. Due to the technical complexities, I omit the proof. □

## A3.3 Comments on the literature

Survey sampling and experimental design are deeply connected ever since Neyman (1934, 1935)’s seminal work. Li and Ding (2017) and Mukerjee et al. (2018) made many theoretical ties between these two areas.

## A3.4 Homework Problems

## A3.1 Vector form of the results

Assume the $c _ { i } ^ { \phantom { } } \mathrm { { s } }$ are vectors and modify

$$
S _ {c} ^ {2} = (n - 1) ^ {- 1} \sum_ {i = 1} ^ {n} (c _ {i} - \bar {c}) (c _ {i} - \bar {c}) ^ {\mathsf {T}}, \quad \hat {S} _ {c} ^ {2} = (n _ {1} - 1) ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} (c _ {i} - \hat {\bar {c}}) (c _ {i} - \hat {\bar {c}}) ^ {\mathsf {T}}.
$$

Show that

$$
E (\hat {c}) = \bar {c}, \quad \mathrm{cov} (\hat {\bar {c}}) = \frac {n _ {0}}{n n _ {1}} S _ {c} ^ {2}, \quad E (\hat {S} _ {c} ^ {2}) = S _ {c} ^ {2}.
$$

## Bibliography

Abadie, A. and Imbens, G. W. (2006). Large sample properties of matching estimators for average treatment effects. Econometrica, 74:235–267.  
Abadie, A. and Imbens, G. W. (2008). On the failure of the bootstrap for matching estimators. Econometrica, 76:1537–1557.  
Abadie, A. and Imbens, G. W. (2011). Bias-corrected matching estimators for average treatment effects. Journal of Business and Economic Statistics, 29:1–11.  
Abadie, A. and Imbens, G. W. (2016). Matching on the estimated propensity score. Econometrica, 84:781–807.  
Alwin, D. F. and Hauser, R. M. (1975). The decomposition of effects in path analysis. American Sociological Review, 40:37–47.  
Amarante, V., Manacorda, M., Miguel, E., and Vigorito, A. (2016). Do cash transfers improve birth outcomes? evidence from matched vital statistics, program, and social security data. American Economic Journal: Economic Policy, 8:1–43.  
Anderson, T. W. and Rubin, H. (1950). The asymptotic properties of estimates of the parameters of a single equation in a complete system of stochastic equations. Annals of Mathematical Statistics, 21:570–582.  
Angrist, J., Lang, D., and Oreopoulos, P. (2009). Incentives and services for college achievement: Evidence from a randomized trial. American Economic Journal: Applied Economics, 1:136–163.  
Angrist, J. and Lavy, V. (2009). The effects of high stakes high school achievement awards: Evidence from a randomized trial. American Economic Review, 99:1384–1414.  
Angrist, J. D. (1990). Lifetime earnings and the Vietnam era draft lottery: evidence from social security administrative records. American Economic Review, 80:313–336.  
Angrist, J. D. (1998). Estimating the labor market impact of voluntary military service using social security data on military applicants. Econometrica, 66:249–288.  
Angrist, J. D. and Evans, W. N. (1998). Children and their parents’ labor supply: Evidence from exogenous variation in family size. American Economic Review, 88:450–477.  
Angrist, J. D. and Imbens, G. W. (1995). Two-stage least squares estimation of average causal effects in models with variable treatment intensity. Journal of the American Statistical Association, 90:431–442.  
Angrist, J. D., Imbens, G. W., and Rubin, D. B. (1996). Identification of causal effects using instrumental variables (with discussion). Journal of the American Statistical Association, 91:444–455.  
Angrist, J. D. and Krueger, A. B. (1991). Does compulsory school attendance affect schooling and earnings? Quarterly Journal of Economics, 106:979– 1014.  
Angrist, J. D. and Pischke, J.-S. (2008). Mostly Harmless Econometrics: An Empiricist’s Companion. Princeton: Princeton University Press.  
Angrist, J. D. and Pischke, J.-S. (2014). Mastering’Metrics: The Path from Cause to Effect. Princeton: Princeton University Press.  
Aronow, P. M., Green, D. P., and Lee, D. K. K. (2014). Sharp bounds on the variance in randomized experiments. Annals of Statistics, 42:850–871.  
Asher, S. and Novosad, P. (2020). Rural roads and local economic development. American Economic Review, 110:797–823.  
Baker, S. G. and Lindeman, K. S. (1994). The paired availability design: a proposal for evaluating epidural analgesia during labor. Statistics in Medicine, 13:2269–2278.  
Balke, A. and Pearl, J. (1997). Bounds on treatment effects from studies with imperfect compliance. Journal of the American Statistical Association, 92:1171–1176.  
Ball, S., Bogatz, G., Rubin, D., and Beaton, A. (1973). Reading with television: An evaluation of the electric company. a report to the children’s television workshop. volumes 1 and 2.  
Bang, H. and Robins, J. M. (2005). Doubly robust estimation in missing data and causal inference models. Biometrics, 61:962–973.  
Barnard, G. A. (1947). Significance tests for 2 × 2 tables. Biometrika, 34:123– 138.  
Baron, R. M. and Kenny, D. A. (1986). The moderator-mediator variable distinction in social psychological research: Conceptual, strategic, and statistical considerations. Journal of Personality and Social Psychology, 51:1173– 1182.

## A3.4 Bibliography

Basmann, R. L. (1957). A generalized classical method of linear estimation of coefficients in a structural equation. Econometrica, 25:77–83.  
Bazzano, L. A., He, J., Muntner, P., Vupputuri, S., and Whelton, P. K. (2003). Relationship between cigarette smoking and novel risk factors for cardiovascular disease in the United States. Annals of Internal Medicine, 138:891– 897.  
Berk, R., Pitkin, E., Brown, L., Buja, A., George, E., and Zhao, L. (2013). Covariance adjustments for the analysis of randomized field experiments. Evaluation Review, 37:170–196.  
Bertrand, M. and Mullainathan, S. (2004). Are Emily and Greg more employable than Lakisha and Jamal? A field experiment on labor market discrimination. American Economic Review, 94:991–1013.  
Bickel, P. J., Hammel, E. A., and O’Connell, J. W. (1975). Sex bias in graduate admissions: Data from Berkeley. Science, 187:398–404.  
Bickel, P. J., Klaassen, C. A. J., Ritov, Y., and Wellner, J. A. (1993). Efficient and Adaptive Estimation for Semiparametric Models. Baltimore: Johns Hopkins University Press.  
Bind, M.-A. C. and Rubin, D. B. (2020). When possible, report a fisherexact p value and display its underlying null randomization distribution. Proceedings of the National Academy of Sciences of the United States of America, 117:19151–19158.  
Blackwell, M. (2013). A framework for dynamic causal inference in political science. American Journal of Political Science, 57:504–520.  
Bloniarz, A., Liu, H., Zhang, C. H., Sekhon, J., and Yu, B. (2016). Lasso adjustments of treatment effect estimates in randomized experiments. Proceedings of the National Academy of Sciences of the United States of America, 113:7383–7390.  
Bloom, H. S. (1984). Accounting for no-shows in experimental evaluation designs. Evaluation Review, 8:225–246.  
Bor, J., Moscoe, E., Mutevedzi, P., Newell, M.-L., and B¨arnighausen, T. (2014). Regression discontinuity designs in epidemiology: causal inference without randomized trials. Epidemiology, 25:729.  
Bowden, J., Davey Smith, G., and Burgess, S. (2015). Mendelian randomization with invalid instruments: effect estimation and bias detection through Egger regression. International Journal of Epidemiology, 44:512–525.  
Bowden, J., Spiller, W., Del Greco M, F., Sheehan, N., Thompson, J., Minelli, C., and Davey Smith, G. (2018). Improving the visualization, interpretation  
and analysis of two-sample summary data mendelian randomization via the radial plot and radial regression. International Journal of Epidemiology, 47:1264–1278.  
Bradford Hill, A. (1965). The environment and disease: association or causation? Proceedings of the Royal Society of Medicine, 58:295–300.  
Bradford Hill, A. (2020). The environment and disease: association or causation? (with discussion). Observational Studies, 6:1–65.  
Bruhn, M. and McKenzie, D. (2009). In pursuit of balance: Randomization in practice in development field experiments. American Economic Journal: Applied Economics, 1:200–232.  
Butler, C. C. (1969). A test for symmetry using the sample distribution function. Annals of Mathematical Statistics, 40:2209–2210.  
Cao, W., Tsiatis, A. A., and Davidian, M. (2009). Improving efficiency and robustness of the doubly robust estimator for a population mean with incomplete data. Biometrika, 96:723–734.  
Card, D. (1993). Using geographic variation in college proximity to estimate the return to schooling. Technical report, National Bureau of Economic Research.  
Carpenter, C. and Dobkin, C. (2009). The effect of alcohol consumption on mortality: regression discontinuity evidence from the minimum drinking age. American Economic Journal: Applied Economics, 1:164–182.  
Cattaneo, M. D. (2010). Efficient semiparametric estimation of multi-valued treatment effects under ignorability. Journal of Econometrics, 155:138–154.  
Cattaneo, M. D., Frandsen, B. R., and Titiunik, R. (2015). Randomization inference in the regression discontinuity design: An application to party advantages in the US Senate. Journal of Causal Inference, 3:1–24.  
Chan, K. C. G., Yam, S. C. P., and Zhang, Z. (2016). Globally efficient nonparametric inference of average treatment effects by empirical balancing calibration weighting. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 78:673–700.  
Charig, C. R., Webb, D. R., Payne, S. R., and Wickham, J. E. (1986). Comparison of treatment of renal calculi by open surgery, percutaneous nephrolithotomy, and extracorporeal shockwave lithotripsy. British Medical Journal, 292:879–882.  
Chen, H., Geng, Z., and Jia, J. (2007). Criteria for surrogate end points. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69:919–932.

## A3.4 Bibliography

Cheng, J. and Small, D. S. (2006). Bounds on causal effects in three-arm trials with non-compliance. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 68:815–836.  
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. Econometrics Journal, 21:C1–C68.  
Chong, A., Cohen, I., Field, E., Nakasone, E., and Torero, M. (2016). Iron deficiency and schooling attainment in peru. American Economic Journal: Applied Economics, 8:222–55.  
Cochran, W. G. (1938). The omission or addition of an independent variate in multiple linear regression. Supplement to the Journal of the Royal Statistical Society, 5:171–176.  
Cochran, W. G. (1953). Sampling Techniques. New York: Wiley.  
Cochran, W. G. (1957). Analysis of covariance: its nature and uses. Biometrics, 13:261–281.  
Cochran, W. G. (1965). The planning of observational studies of human populations (with discussion). Journal of the Royal Statistical Society: Series A (General), 128:234–266.  
Cochran, W. G. (1968). The effectiveness of adjustment by subclassification in removing bias in observational studies. Biometrics, 24:295–313.  
Cochran, W. G. and Rubin, D. B. (1973). Controlling bias in observational studies: A review. Sankhy¯a, 35:417–446.  
Cornfield, J., Haenszel, W., Hammond, E. C., Lilienfeld, A. M., Shimkin, M. B., and Wynder, E. L. (1959). Smoking and lung cancer: recent evidence and a discussion of some questions. Journal of the National Cancer Institute, 22:173–203.  
Cox, D. R. (1982). Randomization and concomitant variables in the design of experiments. In G. Kallianpur, P. R. K. and Ghosh, J. K., editors, Statistics and Probability: Essays in Honor of C. R. Rao, pages 197–202. North-Holland, Amsterdam.  
Cox, D. R. (2007). On a generalization of a result of W. G. Cochran. Biometrika, 94:755–759.  
Crump, R. K., Hotz, V. J., Imbens, G. W., and Mitnik, O. A. (2009). Dealing with limited overlap in estimation of average treatment effects. Biometrika, 96:187–199.  
Cuzick, J., Edwards, R., and Segnan, N. (1997). Adjusting for non-compliance and contamination in randomized clinical trials. Statistics in Medicine, 16:1017–1029.  
D’Amour, A., Ding, P., Feller, A., Lei, L., and Sekhon, J. (2021). Overlap in observational studies with high-dimensional covariates. Journal of Econometrics, 221:644–654.  
Davey Smith, G. and Ebrahim, S. (2003). “Mendelian randomization”: can genetic epidemiology contribute to understanding environmental determinants of disease? International Journal of Epidemiology, 32:1–22.  
Davison, A. C. and Hinkley, D. V. (1997). Bootstrap Methods and Their Application. Cambridge: Cambridge University Press.  
Dawid, A. P. (1979). Conditional independence in statistical theory. Journal of the Royal Statistical Society: Series B (Methodological), 41:1–15.  
Dawid, A. P. (2000). Causal inference without counterfactuals (with discussion). Journal of the American Statistical Association, 95:407–424.  
Dehejia, R. H. and Wahba, S. (1999). Causal effects in nonexperimental studies: Reevaluating the evaluation of training programs. Journal of the American statistical Association, 94:1053–1062.  
Ding, P. (2016). A paradox from randomization-based causal inference (with discussion). Statistical Science, 32:331–345.  
Ding, P. (2021). The Frisch–Waugh–Lovell theorem for standard errors. Statistics and Probability Letters, 168:108945.  
Ding, P. and Dasgupta, T. (2016). A potential tale of two by two tables from completely randomized experiments. Journal of American Statistical Association, 111:157–168.  
Ding, P. and Dasgupta, T. (2017). A randomization-based perspective on analysis of variance: a test statistic robust to treatment effect heterogeneity. Biometrika, 105:45–56.  
Ding, P., Feller, A., and Miratrix, L. (2019). Decomposing treatment effect variation. Journal of the American Statistical Association, 114:304–317.  
Ding, P., Geng, Z., Yan, W., and Zhou, X.-H. (2011). Identifiability and estimation of causal effects by principal stratification with outcomes truncated by death. Journal of the American Statistical Association, 106:1578–1591.  
Ding, P. and Li, F. (2018). Causal inference: A missing data perspective. Statistical Science, 33:214–237.  
Ding, P., Li, X., and Miratrix, L. W. (2017a). Bridging finite and super population causal inference. Journal of Causal Inference, 5:20160027.  
Ding, P. and Lu, J. (2017). Principal stratification analysis using principal scores. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 79:757–777.

## A3.4 Bibliography

Ding, P. and Miratrix, L. W. (2015). To adjust or not to adjust? Sensitivity analysis of M-bias and butterfly-bias. Journal of Causal Inference, 3:41–57.  
Ding, P. and VanderWeele, T. J. (2014). Generalized Cornfield conditions for the risk difference. Biometrika, 101:971–977.  
Ding, P. and VanderWeele, T. J. (2016). Sensitivity analysis without assumptions. Epidemiology, 27:368–377.  
Ding, P. and Vanderweele, T. J. (2016). Sharp sensitivity bounds for mediation under unmeasured mediator-outcome confounding. Biometrika, 103:483– 490.  
Ding, P., VanderWeele, T. J., and Robins, J. M. (2017b). Instrumental variables as bias amplifiers with general outcome and confounding. Biometrika, 104:291–302.  
Doll, R. and Hill, A. B. (1950). Smoking and carcinoma of the lung. British Medical Journal, 2:739.  
Dorn, H. F. (1953). Philosophy of inferences from retrospective studies. American Journal of Public Health and the Nations Health, 43:677–683.  
Durrett, R. (2019). Probability: Theory and Examples. Cambridge: Cambridge University Press.  
Efron, B. (1979). Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 7:1–26.  
Efron, B. and Feldman, D. (1991). Compliance as an explanatory variable in clinical trials (with discussion). Journal of the American Statistical Association, 86:9–17.  
Eicker, F. (1967). Limit theorems for regressions with unequal and dependent errors. In Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, volume 1, pages 59–82. Berkeley, CA: University of California Press.  
Fan, J. and Gijbels, I. (1996). Local Polynomial Modelling and Its Applications. New York: Chapman and Hall/CRC.  
Fieller, E. C. (1954). Some problems in interval estimation. Journal of the Royal Statistical Society: Series B (Methodological), 16:175–185.  
Firth, D. and Bennett, K. E. (1998). Robust models in probability sampling (with discussion). Journal of the Royal Statistical Society: Series B (Statistical Methodology), 60:3–21.  
Fisher, R. A. (1925). Statistical Methods for Research Workers. Edinburgh by Oliver and Boyd, 1st edition.  
Fisher, R. A. (1935). The Design of Experiments. Edinburgh, London: Oliver and Boyd, 1st edition.  
Fisher, R. A. (1957). Dangers of cigarette smoking [letter]. British Medical Journal, 2:297–298.  
Fogarty, C. B. (2018a). On mitigating the analytical limitations of finely stratified experiments. Journal of the Royal Statistical Society. Series B (Statistical Methodology), 80:1035–1056.  
Fogarty, C. B. (2018b). Regression assisted inference for the average treatment effect in paired experiments. Biometrika, 105:994–1000.  
Follmann, D. A. (2000). On the effect of treatment among would-be treatment compliers: An analysis of the multiple risk factor intervention trial. Journal of the American Statistical Association, 95:1101–1109.  
Forastiere, L., Mattei, A., and Ding, P. (2018). Principal ignorability in mediation analysis: through and beyond sequential ignorability. Biometrika, 105:979–986.  
Frangakis, C. E. and Rubin, D. B. (2002). Principal stratification in causal inference. Biometrics, 58:21–29.  
Freedman, D. A. (2008a). On regression adjustments in experiments with several treatments. Annals of Applied Statistics, 2:176–196.  
Freedman, D. A. (2008b). On regression adjustments to experimental data. Advances in Applied Mathematics, 40:180–193.  
Freedman, D. A. (2008c). Randomization does not justify logistic regression. Statistical Science, 23:237–249.  
Freedman, D. A. and Berk, R. A. (2008). Weighting regressions by propensity scores. Evaluation Review, 32:392–409.  
Funk, M. J., Westreich, D., Wiesen, C., St¨urmer, T., Brookhart, M. A., and Davidian, M. (2011). Doubly robust estimation of causal effects. American Journal of Epidemiology, 173:761–767.  
Gastwirth, J. L., KRIEGER, A. M., and ROSENBAUM, P. R. (1998). Cornfield’s inequality. In Armitage, P. and Colton, T., editors, Encyclopedia of Biostatistics. New York: Wiley.  
Gerber, A. S. and Green, D. P. (2012). Field Experiments: Design, Analysis, and Interpretation. WW Norton.  
Gerber, A. S., Green, D. P., and Larimer, C. W. (2008). Social pressure and voter turnout: Evidence from a large-scale field experiment. American Political Science Review, 102:33–48.

## A3.4 Bibliography

Gilbert, P. B. and Hudgens, M. G. (2008). Evaluating candidate principal surrogate endpoints. Biometrics, 64:1146–1154.  
Gould, A. L. (1998). Multi-centre trial analysis revisited. Statistics in Medicine, 17:1779–1797.  
Greevy, R., Lu, B., Silber, J. H., and Rosenbaum, P. (2004). Optimal multivariate matching before randomization. Biostatistics, 5:263–275.  
Guo, K. and Basse, G. (2023). The generalized Oaxaca–Blinder estimator. Journal of American Statistical Association, 118:524–536.  
Hahn, J. (1998). On the role of the propensity score in efficient semiparametric estimation of average treatment effects. Econometrica, 66:315–331.  
Hahn, J., Todd, P., and Van der Klaauw, W. (2001). Identification and estimation of treatment effects with a regression-discontinuity design. Econometrica, 69:201–209.  
Hahn, P. R., Murray, J. S., and Carvalho, C. M. (2020). Bayesian regression tree models for causal inference: regularization, confounding, and heterogeneous effects. Bayesian Analysis, 15:965–1056.  
Hainmueller, J. (2012). Entropy balancing for causal effects: A multivariate reweighting method to produce balanced samples in observational studies. Political Analysis, 20:25–46.  
H´ajek, J. (1960). Limiting distributions in simple random sampling from a finite population. Publications of the Mathematics Institute of the Hungarian Academy of Science, 5:361–74.  
H´ajek, J. (1971). Comment on “an essay on the logical foundations of survey sampling, part one”. The foundations of survey sampling, 236.  
Hammond, E. C. and Horn, D. (1958). Smoking and death rates: report on forty four months of follow-up of 187, 783 men. Journal of the American Medicial Association, 166:1159–1172, 1294–1308.  
Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators. Econometrica, 50:1029–1054.  
Hartley, H. O., Rao, J. N. K., and Kiefer, G. (1969). Variance estimation with one unit per stratum. Journal of the American Statistical Association, 64:841–851.  
Hausman, J. A. (1978). Specification tests in econometrics. Econometrica, 46:1251–1271.  
Hearst, N., Newman, T. B., and Hulley, S. B. (1986). Delayed effects of the military draft on mortality. New England Journal of Medicine, 314:620–624.  
Heckman, J. and Navarro-Lozano, S. (2004). Using matching, instrumental variables, and control functions to estimate economic choice models. Review of Economics and Statistics, 86:30–57.  
Heckman, J. J. (1979). Sample selection bias as a specification error. Econometrica, 47:153–161.  
Hennessy, J., Dasgupta, T., Miratrix, L., Pattanayak, C., and Sarkar, P. (2016). A conditional randomization test to account for covariate imbalance in randomized experiments. Journal of Causal Inference, 4:61–80.  
Hern´an, M. A., Brumback, B., and Robins, J. M. (2000). Marginal structural ´ models to estimate the causal effect of zidovudine on the survival of hivpositive men. Epidemiology, 11:561–570.  
Hern´an, M. A. and Robins, J. M. (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.  
Hill, J., Waldfogel, J., and Brooks-Gunn, J. (2002). Differential effects of highquality child care. Journal of Policy Analysis and Management, 21:601–627.  
Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference. Journal of Computational and Graphical Statistics, 20:217–240.  
Hirano, K. and Imbens, G. W. (2001). Estimation of causal effects using propensity score weighting: An application to data on right heart catheterization. Health Services and Outcomes Research Methodology, 2:259–278.  
Hirano, K., Imbens, G. W., Rubin, D. B., and Zhou, X. H. (2000). Assessing the effect of an influenza vaccine in an encouragement design. Biostatistics, 1:69–88.  
Ho, D. E., Imai, K., King, G., and Stuart, E. A. (2007). Matching as nonparametric preprocessing for reducing model dependence in parametric causal inference. Political Analysis, 15:199–236.  
Ho, D. E., Imai, K., King, G., and Stuart, E. A. (2011). Matchit: nonparametric preprocessing for parametric causal inference. Journal of Statistical Software, 42:1–28.  
Hodges, J. L. and Lehmann, E. L. (1962). Rank methods for combination of independent experiments in analysis of variance. Annals of Mathematical Statistics, 33:482–497.  
Holland, P. W. (1986). Statistics and causal inference (with discussion). Journal of the American statistical Association, 81:945–960.  
Hong, G. and Raudenbush, S. W. (2008). Causal inference for time-varying instructional treatments. Journal of Educational and Behavioral Statistics, 33:333–362.

## A3.4 Bibliography

Horvitz, D. G. and Thompson, D. J. (1952). A generalization of sampling without replacement from a finite universe. Journal of the American statistical Association, 47:663–685.  
Huber, P. J. (1967). The behavior of maximum likelihood estimates under nonstandard conditions. In Cam, L. M. L. and Neyman, J., editors, Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, volume 1, pages 221–233. Berkeley, California: University of California Press.  
Hyman, H. H. (1955). Survey Design and Analysis: Principles, Cases, and Procedures. Glencoe, IL: Free Press.  
Imai, K. (2008a). Sharp bounds on the causal effects in randomized experiments with “truncation-by-death”. Statistics and Probability Letters, 78:144–149.  
Imai, K. (2008b). Variance identification and efficiency analysis in randomized experiments under the matched-pair design. Statistics in Medicine, 27:4857– 4873.  
Imai, K., Keele, L., and Yamamoto, T. (2010). Identification, inference and sensitivity analysis for causal mediation effects. Statistical Science, 25:51– 71.  
Imai, K. and Van Dyk, D. A. (2004). Causal inference with general treatment regimes: Generalizing the propensity score. Journal of the American Statistical Association, 99:854–866.  
Imbens, G. (2020). Potential outcome and directed acyclic graph approaches to causality: Relevance for empirical practice in economics. Journal of Economic Literature, 58:1129–1179.  
Imbens, G. W. (2003). Sensitivity to exogeneity assumptions in program evaluation. American Economic Review, 93:126–132.  
Imbens, G. W. (2004). Nonparametric estimation of average treatment effects under exogeneity: A review. Review of Economics and Statistics, 86:4–29.  
Imbens, G. W. (2014). Instrumental variables: An econometrician’s perspective. Statistical Science, 29:323–358.  
Imbens, G. W. (2015). Matching methods in practice: Three examples. Journal of Human Resources, 50:373–419.  
Imbens, G. W. and Angrist, J. D. (1994). Identification and estimation of local average treatment effects. Econometrica, 62:467–475.  
Imbens, G. W. and Lemieux, T. (2008). Regression discontinuity designs: A guide to practice. Journal of Econometrics, 142:615–635.  
Imbens, G. W. and Manski, C. F. (2004). Confidence intervals for partially identified parameters. Econometrica, 72:1845–1857.  
Imbens, G. W. and Rubin, D. B. (1997). Estimating outcome distributions for compliers in instrumental variables models. Review of Economic Studies, 64:555–574.  
Imbens, G. W. and Rubin, D. B. (2015). Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction. Cambridge: Cambridge University Press.  
Investigators, I. T. et al. (2014). Endovascular or open repair strategy for ruptured abdominal aortic aneurysm: 30 day outcomes from improve randomised trial. British Medical Journal, 348:f7661.  
Ioannidis, J. P. A., Tan, Y. J., and Blum, M. R. (2019). Limitations and misinterpretations of E-values for sensitivity analyses of observational studies. Annals of Internal Medicine, 170:108–111.  
Jackson, L. A., Jackson, M. L., Nelson, J. C., Neuzil, K. M., and Weiss, N. S. (2006). Evidence of bias in estimates of influenza vaccine effectiveness in seniors. International Journal of Epidemiology, 35:337–344.  
Jiang, Z. and Ding, P. (2020). Measurement errors in the binary instrumental variable model. Biometrika, 107:238–245.  
Jiang, Z. and Ding, P. (2021). Identification of causal effects within principal strata using auxiliary variables. Statistical Science, 36:493–508.  
Jiang, Z., Ding, P., and Geng, Z. (2016). Principal causal effect identification and surrogate end point evaluation by multiple trials. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 78:829–848.  
Jiang, Z., Yang, S., and Ding, P. (2022). Multiply robust estimation of causal effects under principal ignorability. Journal of the Royal Statistical Society - Series B (Statistical Methodology), 84:1423–1445.  
Jo, B. and Stuart, E. A. (2009). On the use of propensity scores in principal causal effect estimation. Statistics in Medicine, 28:2857–2875.  
Jo, B., Stuart, E. A., MacKinnon, D. P., and Vinokur, A. D. (2011). The use of propensity scores in mediation analysis. Multivariate Behavioral Research, 46:425–452.  
Judd, C. M. and Kenny, D. A. (1981). Process analysis estimating mediation in treatment evaluations. Evaluation Review, 5:602–619.  
Kang, J. D. Y. and Schafer, J. L. (2007). Demystifying double robustness: A comparison of alternative strategies for estimating a population mean from incomplete data. Statistical Science, 22:523–539.

## A3.4 Bibliography

Katan, M. B. (1986). Apoupoprotein E isoforms, serum cholesterol, and cancer. Lancet, 327:507–508.  
King, G. and Zeng, L. (2006). The dangers of extreme counterfactuals. Political Analysis, 14:131–159.  
Kitagawa, T. (2015). A test for instrument validity. Econometrica, 83:2043– 2063.  
Koenker, R. and Xiao, Z. (2002). Inference on the quantile regression process. Econometrica, 70:1583–1612.  
K¨unzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences of the United States of America, 116:4156–4165.  
Kurth, T., Walker, A. M., Glynn, R. J., Chan, K. A., Gaziano, J. M., Berger, K., and Robins, J. M. (2005). Results of multivariable logistic regression, propensity matching, propensity adjustment, and propensity-based weighting under conditions of nonuniform effect. American Journal of Epidemiology, 163:262–270.  
LaLonde, R. J. (1986). Evaluating the econometric evaluations of training programs with experimental data. American Economic Review, 76:604–620.  
Lee, D. S. (2008). Randomized experiments from non-random selection in US House elections. Journal of Econometrics, 142:675–697.  
Lee, D. S. (2009). Training, wages, and sample selection: Estimating sharp bounds on treatment effects. Review of Economic Studies, 76:1071–1102.  
Lee, D. S. and Lemieux, T. (2010). Regression discontinuity designs in economics. Journal of Economic Literature, 48:281–355.  
Lee, M.-J. (2018). Simple least squares estimator for treatment effects using propensity score residuals. Biometrika, 105:149–164.  
Lee, W.-C. (2011). Bounding the bias of unmeasured factors with confounding and effect-modifying potentials. Statistics in Medicine, 30:1007–1017.  
Lehmann, E. L. (1975). Nonparametrics: Statistical Methods Based on Ranks. California: Holden-Day, Inc.  
Lei, L. and Ding, P. (2021). Regression adjustment in completely randomized experiments with a diverging number of covariates. Biometrika, 108:815– 828.  
Li, F., Mattei, A., and Mealli, F. (2015). Evaluating the causal effect of university grants on student dropout: evidence from a regression discontinuity design using principal stratification. Annals of Applied Statistics, 9:1906– 1931.  
Li, F., Morgan, K. L., and Zaslavsky, A. M. (2018a). Balancing covariates via propensity score weighting. Journal of the American Statistical Association, 113:390–400.  
Li, F., Thomas, L. E., and Li, F. (2019). Addressing extreme propensity scores via the overlap weights. American Journal of Epidemiology, 188:250–257.  
Li, X. and Ding, P. (2016). Exact confidence intervals for the average causal effect on a binary outcome. Statistics in Medicine, 35:957–960.  
Li, X. and Ding, P. (2017). General forms of finite population central limit theorems with applications to causal inference. Journal of the American Statistical Association, 112:1759–1769.  
Li, X. and Ding, P. (2020). Rerandomization and regression adjustment. Journal of the Royal Statistical Society, Series B (Statistical Methodology), 82:241–268.  
Li, X., Ding, P., and Rubin, D. B. (2018b). Asymptotic theory of rerandomization in treatment-control experiments. Proceedings of the National Academy of Sciences of the United States of America, 115:9157–9162.  
Lin, W. (2013). Agnostic notes on regression adjustments to experimental data: Reexamining Freedman’s critique. Annals of Applied Statistics, 7:295– 318.  
Lin, Z., Ding, P., and Han, F. (2023). Estimation based on nearest neighbor matching: from density ratio to average treatment effect. Econometrica.  
Lind, J. (1753). A treatise of the scurvy. Three Parts. Containing an Inquiry into the Nature, Causes and Cure, of that Disease. Together with a Critical and Chronological View of what has been Published on the Subject.  
Lipsitch, M., Tchetgen Tchetgen, E., and Cohen, T. (2010). Negative controls: a tool for detecting confounding and bias in observational studies. Epidemiology, 21:383–388.  
Little, R. and An, H. (2004). Robust likelihood-based analysis of multivariate data with missing values. Statistica Sinica, 14:949–968.  
Liu, H. and Yang, Y. (2020). Regression-adjusted average treatment effect estimates in stratified randomized experiments. Biometrika, 107:935–948.  
Long, J. S. and Ervin, L. H. (2000). Using heteroscedasticity consistent standard errors in the linear regression model. American Statistician, 54:217– 224.

## A3.4 Bibliography

Lu, S. and Ding, P. (2023). Flexible sensitivity analysis for causal inference in observational studies subject to unmeasured confounding. https://arxiv.org/abs/2305.17643.  
Lumley, T., Shaw, P. A., and Dai, J. Y. (2011). Connections between survey calibration estimators and semiparametric models for incomplete data. International Statistical Review, 79:200–220.  
Lunceford, J. K. and Davidian, M. (2004). Stratification and weighting via the propensity score in estimation of causal treatment effects: a comparative study. Statistics in Medicine, 23:2937–2960.  
Luo, X., Dasgupta, T., Xie, M., and Liu, R. Y. (2021). Leveraging the fisher randomization test using confidence distributions: Inference, combination and fusion learning. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 83:777–797.  
Manski, C. F. (1990). Nonparametric bounds on treatment effects. American Economic Review, 2:319–323.  
Manski, C. F. (2003). Partial Identification of Probability Distributions. New York: Springer.  
Mattei, A., Li, F., and Mealli, F. (2013). Exploiting multiple outcomes in bayesian principal stratification analysis with application to the evaluation of a job training program. Annals of Applied Statistics, 7:2336–2360.  
McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. Journal of Econometrics, 142:698–714.  
McDonald, C. J., Hui, S. L., and Tierney, W. M. (1992). Effects of computer reminders for influenza vaccination on morbidity during influenza epidemics. MD Computing: Computers in Medical Practice, 9:304–312.  
McGrath, S., Young, J. G., and Hern´an, M. A. (2021). Revisiting the g-null paradox. Epidemiology, 33:114–120.  
Mealli, F. and Pacini, B. (2013). Using secondary outcomes to sharpen inference in randomized experiments with noncompliance. Journal of the American Statistical Association, 108:1120–1131.  
Meinert, C. L., Knatterud, G. L., Prout, T. E., and Klimt, C. R. (1970). A study of the effects of hypoglycemic agents on vascular complications in patients with adult-onset diabetes. ii. mortality results. Diabetes, 19:Suppl– 789.  
Mercatanti, A. and Li, F. (2014). Do debit cards increase household spending? evidence from a semiparametric causal analysis of a survey. Annals of Applied Statistics, 8:2485–2508.  
Ming, K. and Rosenbaum, P. R. (2000). Substantial gains in bias reduction from matching with a variable number of controls. Biometrics, 56:118–124.  
Ming, K. and Rosenbaum, P. R. (2001). A note on optimal matching with variable controls using the assignment algorithm. Journal of Computational and Graphical Statistics, 10:455–463.  
Miratrix, L. W., Sekhon, J. S., and Yu, B. (2013). Adjusting treatment effect estimates by post-stratification in randomized experiments. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 75:369–396.  
Morgan, K. L. and Rubin, D. B. (2012). Rerandomization to improve covariate balance in experiments. Annals of Statistics, 40:1263–1282.  
Mukerjee, R., Dasgupta, T., and Rubin, D. B. (2018). Using standard tools from finite population sampling to improve causal inference for complex experiments. Journal of the American Statistical Association, 113:868–881.  
Naimi, A. I., Cole, S. R., and Kennedy, E. H. (2017). An introduction to g methods. International Journal of Epidemiology, 46:756–762.  
Negi, A. and Wooldridge, J. M. (2021). Revisiting regression adjustment in experiments with heterogeneous treatment effects. Econometric Reviews, 40:504–534.  
Neyman, J. (1923). On the application of probability theory to agricultural experiments. essay on principles (with discussion). section 9 (translated). reprinted ed. Statistical Science, 5:465–472.  
Neyman, J. (1934). On the two different aspects of the representative method: the method of stratified sampling and the method of purposive selection (with discussion). Journal of the Royal Statistical Society, 97:558–625.  
Neyman, J. (1935). Statistical problems in agricultural experimentation (with discussion). Supplement to the Journal of the Royal Statistical Society, 2:107–180.  
Nguyen, T. Q., Schmid, I., Ogburn, E. L., and Stuart, E. A. (2021). Clarifying causal mediation analysis for the applied researcher: Effect identification via three assumptions and five potential outcomes. Psychological Methods, 26:255–271.  
Otsu, T. and Rai, Y. (2017). Bootstrap inference of matching estimators for average treatment effects. Journal of the American Statistical Association, 112:1720–1732.  
Pearl, J. (1995). Causal diagrams for empirical research (with discussion). Biometrika, 82:669–688.

## A3.4 Bibliography

Pearl, J. (2000). Causality: Models, Reasoning and Inference. Cambridge: Cambridge University Press.  
Pearl, J. (2001). Direct and indirect effects. In Breese, J. S. and Koller, D., editors, Proceedings of the 17th Conference on Uncertainty in Artificial Intelligence, pages 411–420. pp. 411–420. San Francisco: Morgan Kaufmann Publishers Inc.  
Pearl, J. (2010). On a class of bias-amplifying variables that endanger effect estimates. In Grunwald, P. and Spirtes, P., editors, Proceedings of the Twenty-Sixth Conference on Uncertainty in Artificial Intelligence (UAI 2010), Corvallis, OR: 425–432. Association for Uncetainty in Artificial Intelligence.  
Pearl, J. (2011). Invited commentary: Understanding bias amplification. American Journal of Epidemiology, 174:1223–1227.  
Pearl, J. (2018). Does obesity shorten life? Or is it the soda? On nonmanipulable causes. Journal of Causal Inference, 6:20182001.  
Pearl, J. and Bareinboim, E. (2014). External validity: From do-calculus to transportability across populations. Statistical Science, 29:579–595.  
Permutt, T. and Hebel, J. R. (1989). Simultaneous-equation estimation in a clinical trial of the effect of smoking on birth weight. Biometrics, 45:619– 622.  
Phipson, B. and Smyth, G. K. (2010). Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. Statistical Applications in Genetics and Molecular Biology, 9:Article39.  
Pimentel, S. D., Yoon, F., and Keele, L. (2015). Variable-ratio matching with fine balance in a study of the Peer Health Exchange. Statistics in Medicine, 34:4070–4082.  
Poole, C. (2010). On the origin of risk relativism. Epidemiology, 21:3–9.  
Popper, K. (1963). Conjectures and Refutations: The Growth of Scientific Knowledge. Routledge.  
Powers, D. E. and Swinton, S. S. (1984). Effects of self-study for coachable test item types. Journal of Educational Psychology, 76:266–278.  
Prentice, R. L. and Pyke, R. (1979). Logistic disease incidence models and case-control studies. Biometrika, 66:403–411.  
Rao, C. R. (1970). Estimation of heteroscedastic variances in linear models. Journal of the American Statistical Association, 65:161–172.  
Reichenbach, H. (1957). The Direction of Time. University of California Press.  
Rigdon, J. and Hudgens, M. G. (2015). Randomization inference for treatment effects on a binary outcome. Statistics in Medicine, 34:924–935.  
Robins, J., Sued, M., Lei-Gomez, Q., and Rotnitzky, A. (2007). Comment: Performance of double-robust estimators when inverse probability weights are highly variable. Statistical Science, 22:544–559.  
Robins, J. M. (1999). Association, causation, and marginal structural models. Synthese, 121:151–179.  
Robins, J. M. and Greenland, S. (1992). Identifiability and exchangeability for direct and indirect effects. Epidemiology, 3:143–155.  
Robins, J. M., Hernan, M. A., and Brumback, B. (2000). Marginal structural models and causal inference in epidemiology. Epidemiology, 11:550–560.  
Robins, J. M., Mark, S. D., and Newey, W. K. (1992). Estimating exposure effects by modelling the expectation of exposure conditional on confounders. Biometrics, 48:479–495.  
Robins, J. M. and Wasserman, L. A. (1997). Estimation of effects of sequential treatments by reparameterizing directed acyclic graphs. In Proceedings of the Thirteenth conference on Uncertainty in artificial intelligence, volume 409–420.  
Rosenbaum, P. R. (1984). The consequences of adjustment for a concomitant variable that has been affected by the treatment. Journal of the Royal Statistical Society. Series A, 147:656–666.  
Rosenbaum, P. R. (1987a). Model-based direct adjustment. Journal of the American Statistical Association, 82:387–394.  
Rosenbaum, P. R. (1987b). Sensitivity analysis for certain permutation inferences in matched observational studies. Biometrika, 74:13–26.  
Rosenbaum, P. R. (1989). The role of known effects in observational studies. Biometrics, 45:557–569.  
Rosenbaum, P. R. (2002a). Covariance adjustment in randomized experiments and observational studies (with discussion). Statistical Science, 17:286–327.  
Rosenbaum, P. R. (2002b). Observational Studies. Springer, 2nd edition.  
Rosenbaum, P. R. (2015). Two R packages for sensitivity analysis in observational studies. Observational Studies, 1:1–17.  
Rosenbaum, P. R. (2018). Sensitivity analysis for stratified comparisons in an observational study of the effect of smoking on homocysteine levels. Annals of Applied Statistics, 12:2312–2334.

## A3.4 Bibliography

Rosenbaum, P. R. (2020). Modern algorithms for matching in observational studies. Annual Review of Statistics and Its Application, 7:143–176.  
Rosenbaum, P. R. and Rubin, D. B. (1983a). Assessing sensitivity to an unobserved binary covariate in an observational study with binary outcome. Journal of the Royal Statistical Society - Series B (Statistical Methodology), 45:212–218.  
Rosenbaum, P. R. and Rubin, D. B. (1983b). The central role of the propensity score in observational studies for causal effects. Biometrika, 70:41–55.  
Rosenbaum, P. R. and Rubin, D. B. (1984). Reducing bias in observational studies using subclassification on the propensity score. Journal of the American statistical Association, 79:516–524.  
Rosenbaum, P. R. and Rubin, D. B. (2023). Propensity scores in the design of observational studies for causal effects. Biometrika, 110:1–13.  
Rothman, K. J., Greenland, S., Lash, T. L., et al. (2008). Modern epidemiology, volume 3. Wolters Kluwer Health/Lippincott Williams & Wilkins Philadelphia.  
Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. Journal of Educational Psychology, 66:688–701.  
Rubin, D. B. (1975). Bayesian inference for causality: The importance of randomization. In The Proceedings of the social statistics section of the American Statistical Association, volume 233, page 239. American Statistical Association Alexandria, VA.  
Rubin, D. B. (1978). Bayesian inference for causal effects: The role of randomization. Annals of Statistics, 6:34–58.  
Rubin, D. B. (1980). Comment on “Randomization analysis of experimental data: the Fisher randomization test” by D. Basu. Journal of American Statistical Association, 75:591–593.  
Rubin, D. B. (2005). Causal inference using potential outcomes: Degisn, modeling, decisions. Journal of American Statistical Association, 100:322–331.  
Rubin, D. B. (2006a). Causal inference through potential outcomes and principal stratification: application to studies with “censoring” due to death (with discussion). Statistical Science, 21:299–309.  
Rubin, D. B. (2006b). Matched Sampling for Causal Effects. Cambridge: Cambridge University Press.  
Rubin, D. B. (2007). The design versus the analysis of observational studies for causal effects: parallels with the design of randomized trials. Statistics in Medicine, 26:20–36.  
Rubin, D. B. (2008). For objective causal inference, design trumps analysis. Annals of Applied Statistics, 2:808–840.  
Rudolph, K. E., Goin, D. E., Paksarian, D., Crowder, R., Merikangas, K. R., and Stuart, E. A. (2018). Causal mediation analysis with observational data: considerations and illustration examining mechanisms linking neighborhood poverty to adolescent substance use. American Journal of Epidemiology, 188:598–608.  
Sabbaghi, A. and Rubin, D. B. (2014). Comments on the Neyman–Fisher controversy and its consequences. Statistical Science, 29:267–284.  
Salsburg, D. (2001). The Lady Tasting Tea: How Statistics Revolutionized Science in the Twentieth Century. Henry Holt and Company.  
Sanders, E. Gustafson, P. and Karim, M. E. (2021). Incorporating partial adherence into the principal stratification analysis framework. Statistics in Medicine, 40:3625–3644.  
Sanderson, E., Macdonald-Wallis, C., and Davey Smith, G. (2017). Negative control exposure studies in the presence of measurement error: implications for attempted effect estimate calibration. International Journal of Epidemiology, 47:587–596.  
Scharfstein, D. O., Rotnitzky, A., and Robins, J. M. (1999). Adjusting for nonignorable drop-out using semiparametric nonresponse models. Journal of the American Statistical Association, 94:1096–1120.  
Schlesselman, J. J. (1978). Assessing effects of confounding variables. American Journal of Epidemiology, 108:3–8.  
Schochet, P. Z., Burghardt, J., and McConnell, S. (2008). Does job corps work? impact findings from the national job corps study. American Economic Review, 98:1864–1886.  
Sekhon, J. S. (2009). Opiates for the matches: Matching methods for causal inference. Annual Review of Political Science, 12:487–508.  
Sekhon, J. S. (2011). Multivariate and propensity score matching software with automated balance optimization: The matching package for R. Journal of Statistical Software, 47:1–52.  
Sekhon, J. S. and Titiunik, R. (2017). On interpreting the regression discontinuity design as a local experiment. In Regression Discontinuity Designs, volume 38. Emerald Publishing Limited.  
Shinozaki, T. and Matsuyama, Y. (2015). Doubly robust estimation of standardized risk difference and ratio in the exposed population. Epidemiology, 26:873–877.

## A3.4 Bibliography

Sobel, M. E. (1982). Asymptotic confidence intervals for indirect effects in structural equation models. Sociological Methodology, 13:290–312.  
Sobel, M. E. (1986). Some new results on indirect effects and their standard errors in covariance structure models. Sociological Methodology, 16:159–186.  
Sommer, A. and Zeger, S. L. (1991). On estimating efficacy from clinical trials. Statistics in Medicine, 10:45–52.  
Stuart, E. A. (2010). Matching methods for causal inference: A review and a look forward. Statistical Science, 25:1–21.  
Stuart, E. A. and Jo, B. (2015). Assessing the sensitivity of methods for estimating principal causal effects. Statistical Methods in Medical Research, 24:657–674.  
Tao, Y. and Fu, H. (2019). Doubly robust estimation of the weighted average treatment effect for a target population. Statistics in Medicine, 38:315–325.  
Theil, H. (1953). Estimation and simultaneous correlation in complete equation systems. central planning bureau. Technical report, Mimeo, The Hague.  
Thistlethwaite, D. L. and Campbell, D. T. (1960). Regression-discontinuity analysis: An alternative to the ex post facto experiment. Journal of Educational Psychology, 51:309.  
Thistlewaite, D. L. and Campbell, D. T. (2016). Regression-discontinuity analysis: An alternative to the ex-post facto experiment (with discussion). Observational Studies, 2:119–209.  
Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58:267–288.  
Titterington, D. (2013). Biometrika highlights from volume 28 onwards. Biometrika, 100:17–73.  
Valeri, L. and Vanderweele, T. J. (2014). The estimation of direct and indirect causal effects in the presence of misclassified binary mediator. Biostatistics, 15:498–512.  
Van der Laan, M. J. and Rose, S. (2011). Targeted Learning: Causal Inference for Observational and Experimental Data. New York: Springer.  
Van der Vaart, A. W. (2000). Asymptotic Statistics. Cambridge: Cambridge University Press.  
Van Elteren, P. (1960). On the combination of independent two-sample tests of wilcoxon. Bulletin of the Institute of International Statistics, 37:351–361.  
VanderWeele, T. J. (2008). Simple relations between principal stratification and direct and indirect effects. Statistics and Probability Letters, 78:2957– 2962.  
VanderWeele, T. J. (2015). Explanation in Causal Inference: Methods for Mediation and Interaction. Oxford: Oxford University Press.  
VanderWeele, T. J., Asomaning, K., and Tchetgen Tchetgen, E. J. (2012). Genetic variants on 15q25.1, smoking, and lung cancer: An assessment of mediation and interaction. American Journal of Epidemiology, 175:1013– 1020.  
VanderWeele, T. J. and Ding, P. (2017). Sensitivity analysis in observational research: introducing the E-value. Annals of Internal Medicine, 167:268– 274.  
VanderWeele, T. J. and Shpitser, I. (2011). A new criterion for confounder selection. Biometrics, 67:1406–1413.  
VanderWeele, T. J. and Tchetgen Tchetgen, E. J. (2017). Mediation analysis with time varying exposures and mediators. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 79:917–938.  
VanderWeele, T. J., Tchetgen Tchetgen, E. J., Cornelis, M., and Kraft, P. (2014). Methodological challenges in Mendelian randomization. Epidemiology, 25:427.  
Vansteelandt, S. and Daniel, R. M. (2014). On regression adjustment for the propensity score. Statistics in Medicine, 33:4053–4072.  
Vansteelandt, S. and Dukes, O. (2022). Assumption-lean inference for generalised linear model parameters (with discussion). Journal of the Royal Statistical Society, Series B (Statistical Methodology), 84:657–685.  
Vansteelandt, S. and Joffe, M. (2014). Structural nested models and Gestimation: the partially realized promise. Statistical Science, 29:707–731.  
Vermeulen, K. and Vansteelandt, S. (2015). Bias-reduced doubly robust estimation. Journal of the American Statistical Association, 110:1024–1036.  
Voight, B. F., Peloso, G. M., Orho-Melander, M., Frikke-Schmidt, R., Barbalic, M., Jensen, M. K., Hindy, G., H´olm, H., Ding, E. L., and Johnson, T. (2012). Plasma HDL cholesterol and risk of myocardial infarction: a Mendelian randomisation study. The Lancet, 380:572–580.  
Wager, S. and Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. Journal of the American Statistical Association, 113:1228–1242.

## A3.4 Bibliography

Wager, S., Du, W., Taylor, J., and Tibshirani, R. J. (2016). High-dimensional regression adjustments in randomized experiments. Proceedings of the National Academy of Sciences of the United States of America, 113:12673– 12678.  
Wald, A. (1940). The fitting of straight lines if both variables are subject to error. Annals of Mathematical Statistics, 11:284–300.  
Wang, L., Zhang, Y., Richardson, T. S., and Zhou, X.-H. (2020). Robust estimation of propensity score weights via subclassification. arXiv preprint arXiv:1602.06366.  
White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. Econometrica, 48:817–838.  
Wooldridge, J. (2016). Should instrumental variables be used as matching variables? Research in Economics, 70:232–237.  
Wooldridge, J. M. (2015). Control function methods in applied econometrics. Journal of Human Resources, 50:420–445.  
Wu, J. and Ding, P. (2021). Randomization tests for weak null hypotheses in randomized experiments. Journal of the American Statistical Association, 116:1898–1913.  
Yang, F. and Small, D. S. (2016). Using post-outcome measurement information in censoring-by-death problems. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 78:299–318.  
Yang, S. and Ding, P. (2018). Asymptotic causal inference with observational studies trimmed by the estimated propensity scores. Biometrika, 105:487– 493.  
Zelen, M. (1979). A new design for randomized clinical trials. New England Journal of Medicine, 300:1242–1245.  
Zhang, J. L. and Rubin, D. B. (2003). Estimation of causal effects via principal stratification when some outcomes are truncated by “death”. Journal of Educational and Behavioral Statistics, 28:353–368.  
Zhang, J. L., Rubin, D. B., and Mealli, F. (2009). Likelihood-based analysis of causal effects of job-training programs using principal stratification. Journal of the American Statistical Association, 104:166–176.  
Zhang, M. and Ding, P. (2022). Interpretable sensitivity analysis for the baronkenny approach to mediation with unmeasured confounding. arXiv preprint arXiv:2205.08030.  
Zhao, A. and Ding, P. (2021a). Covariate-adjusted Fisher randomization tests for the average treatment effect. Journal of Econometrics, 225:278–294.

Zhao, A. and Ding, P. (2021b). No star is good news: A unified look at rerandomization based on p-values from covariate balance tests. arXiv preprint arXiv:2112.10545.

Zhao, Q., Wang, J., Hemani, G., Bowden, J., and Small, D. (2020). Statistical inference in two-sample summary-data Mendelian randomization using robust adjusted profile score. Annals of Statistics, 48:1742–1769.
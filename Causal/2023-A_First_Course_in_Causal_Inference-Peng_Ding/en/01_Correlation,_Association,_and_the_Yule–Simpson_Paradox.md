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
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
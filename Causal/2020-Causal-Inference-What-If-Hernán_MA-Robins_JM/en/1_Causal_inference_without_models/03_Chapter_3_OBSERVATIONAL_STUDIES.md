# Chapter 3 OBSERVATIONAL STUDIES

Consider again the causal question “does one’s looking up at the sky make other pedestrians look up too?” After considering a randomized experiment as in the previous chapter, you concluded that looking up so many times was too time-consuming and unhealthy for your neck bones. Hence you decided to conduct the following study: Find a nearby pedestrian who is standing in a corner and not looking up. Then find a second pedestrian who is walking towards the first one and not looking up either. Observe and record their behavior during the next 10 seconds. Repeat this process a few thousand times. You could now compare the proportion of second pedestrians who looked up after the first pedestrian did, and compare it with the proportion of second pedestrians who looked up before the first pedestrian did.

Such a scientific study in which the investigator observes and records the relevant data is referred to as an **observational study** .

If you had conducted the observational study described above, critics could argue that two pedestrians may both look up not because the first pedestrian’s looking up causes the other’s looking up, but because they both heard a thunderous noise above or some rain drops started to fall, and thus your study findings are inconclusive as to whether one’s looking up makes others look up. These criticisms do not apply to randomized experiments, which is one of the reasons why randomized experiments are central to the theory of causal inference.

However, in practice, the importance of randomized experiments for the estimation of causal effects is more limited. Many scientific studies are not experiments. Much human knowledge is derived from observational studies. Think of evolution, tectonic plates, global warming, or astrophysics. Think of how humans learned that hot coffee may cause burns. This chapter reviews some conditions under which observational studies lead to valid causal inferences.

## 3.1 Identifiability Conditions

For simplicity, this chapter considers only randomized experiments in which all participants remain under follow-up and adhere to their assigned treatment throughout the entire study. Chapters 8 and 9 discuss alternative scenarios.

Ideal randomized experiments can be used to identify and quantify average causal effects because the randomized assignment of treatment leads to **exchangeability** . Take a marginally randomized experiment of heart transplant and mortality as an example: if those who received a transplant had not received it, they would have been expected to have the same death risk as those who did not actually receive the heart transplant. As a consequence, an associational risk ratio of 0.7 from the randomized experiment is expected to equal the causal risk ratio.

Observational studies, on the other hand, may be much less convincing (for an example, see the introduction to this chapter). A key reason for our hesitation to endow observational associations with a causal interpretation is the lack of randomized treatment assignment. As an example, take an observational study of heart transplant and mortality in which those who received the heart transplant were more likely to have a severe heart condition. Then, if those who received a transplant had not received it, they would have been expected to have a greater death risk than those who did not actually receive the heart transplant. As a consequence, an associational risk ratio of 1.1 from the observational study would be a compromise between the truly beneficial effect of transplant on mortality (which pushes the associational risk ratio to be under 1) and the underlying greater mortality risk in those who received transplant (which pushes the associational risk ratio to be over 1).

**Table 3.1**

| Name       | L   | A   | Y   |
| ---------- | --- | --- | --- |
| Rhea       | 0   | 0   | 0   |
| Kronos     | 0   | 0   | 1   |
| Demeter    | 0   | 0   | 0   |
| Hades      | 0   | 0   | 0   |
| Hestia     | 0   | 1   | 0   |
| Poseidon   | 0   | 1   | 0   |
| Hera       | 0   | 1   | 0   |
| Zeus       | 0   | 1   | 1   |
| Artemis    | 1   | 0   | 1   |
| Apollo     | 1   | 0   | 1   |
| Leto       | 1   | 0   | 0   |
| Ares       | 1   | 1   | 1   |
| Athena     | 1   | 1   | 1   |
| Hephaestus | 1   | 1   | 1   |
| Aphrodite  | 1   | 1   | 1   |
| Polyphemus | 1   | 1   | 1   |
| Persephone | 1   | 1   | 1   |
| Hermes     | 1   | 1   | 0   |
| Hebe       | 1   | 1   | 0   |
| Dionysus   | 1   | 1   | 0   |

Rubin (1974, 1978) extended Neyman’s theory for randomized experiments to observational studies. Rosenbaum and Rubin (1983) referred to the combination of exchangeability and positivity as **weak ignorability** , and to the combination of full exchangeability (see Technical Point 2.1) and positivity as **strong ignorability** .

> The best explanation for an association between treatment and outcome in an observational study is not necessarily a causal effect of the treatment on the outcome.

While recognizing that randomized experiments have intrinsic advantages for causal inference, sometimes we are stuck with observational studies to answer causal questions. What do we do? A common strategy is to analyze our data as if treatment had been randomly assigned conditional on measured covariates $L$ — though we often know this is at best an approximation. Causal inference from observational data then revolves around the hope that the observational study can be viewed as a **conditionally randomized experiment** .

Informally, an observational study can be conceptualized as a conditionally randomized experiment if the following conditions hold:

1. The values of treatment under comparison correspond to well-defined interventions that, in turn, correspond to the versions of treatment in the data.
2. The conditional probability of receiving every value of treatment, though not decided by the investigators, depends only on measured covariates $L$.
3. The probability of receiving every value of treatment conditional on $L$ is greater than zero, i.e., **positive** .

In this chapter we describe these three conditions in the context of observational studies. Condition 1 was referred to as **consistency** in Chapter 1, condition 2 was referred to as **exchangeability** in the previous chapters, and condition 3 was referred to as **positivity** in Technical Point 2.3.

We will see that these conditions are often heroic, which explains why causal inferences from observational studies are viewed with suspicion. However, if the analogy between observational study and conditionally randomized experiment happens to be correct, then we can use the methods described in the previous chapter—IP weighting or standardization—to identify causal effects from observational studies. We therefore refer to these conditions as **identifiability conditions** or **assumptions** .

For example, in the previous chapter, we computed a causal risk ratio equal to 1 using the data in Table 2.2, which arose from a conditionally randomized experiment. If the same data, now shown in Table 3.1, had arisen from an observational study and the three identifiability conditions above held true, we would also compute a causal risk ratio equal to 1.

Importantly, in ideal randomized experiments the identifiability conditions hold by design. That is, for a conditionally randomized experiment, we would only need the data in Table 3.1 to compute the causal risk ratio of 1. In contrast, to identify the causal risk ratio from an observational study, we would need to assume that the identifiability conditions held, which of course may not be true. Causal inference from observational data requires two elements: **data** and **identifiability conditions** . See Fine Point 3.1 for a more precise definition of identifiability.

When any of the identifiability conditions does not hold, the analogy between observational study and conditionally randomized experiment breaks down. In that situation, there are other possible approaches to causal inference from observational data, which require a different set of identifiability conditions. One of these approaches is hoping that a predictor of treatment, referred to as an **instrumental variable** , behaves as if it had been randomly assigned conditional on the measured covariates. We discuss instrumental variable methods in Chapter 16.

### Fine Point 3.1

**Identifiability of causal effects.** We say that an average causal effect is (nonparametrically) identifiable under a particular set of assumptions if these assumptions imply that the distribution of the observed data is compatible with a single value of the effect measure. Conversely, we say that an average causal effect is nonidentifiable under the assumptions when the distribution of the observed data is compatible with several values of the effect measure.

For example, if the study in Table 3.1 had arisen from a conditionally randomized experiment in which the probability of receiving treatment depended on the value of $L$ (and hence conditional exchangeability $Y^{a} \perp\!\!\!\perp A \mid L$ holds by design), then we showed in the previous chapter that the causal effect is identifiable: the causal risk ratio equals 1, without requiring any further assumptions. However, if the data in Table 3.1 had arisen from an observational study, then the causal risk ratio equals 1 only if we supplement the data with the assumption of conditional exchangeability $Y^{a} \perp\!\!\!\perp A \mid L$. To identify the causal effect in observational studies, we need an assumption external to the data, an identifying assumption.

In fact, if we decide not to supplement the data with the identifying assumption, then the data in Table 3.1 are consistent with a causal risk ratio:

- lower than 1, if risk factors other than $L$ are more frequent among the treated;
- greater than 1, if risk factors other than $L$ are more frequent among the untreated;
- equal to 1, if all risk factors except $L$ are equally distributed between the treated and the untreated or, equivalently, if $Y^{a} \perp\!\!\!\perp A \mid L$.

This chapter discusses the three identifiability conditions for nonparametric identification of average causal effects. In Chapter 16, we describe alternative identifiability conditions which suffice for nonparametric identification of average causal effects.

Not surprisingly, observational methods based on the analogy with a conditionally randomized experiment have been traditionally privileged in disciplines in which this analogy is often reasonable (e.g., epidemiology), whereas instrumental variable methods have been traditionally privileged in disciplines in which observational studies cannot often be conceptualized as conditionally randomized experiments given the measured covariates (e.g., economics). Until Chapter 16, we will focus on causal inference approaches that rely on the ability of the observational study to emulate a conditionally randomized experiment. We now describe in more detail each of the three identifiability conditions.

## 3.2 Exchangeability

An independent predictor of the outcome is a covariate associated with the outcome $Y$ within levels of treatment. For dichotomous outcomes, independent predictors of the outcome are often referred to as risk factors for the outcome.

We have already said much about exchangeability $Y^{a} \perp A$. In marginally (i.e., unconditionally) randomized experiments, the treated and the untreated are exchangeable because the treated, had they remained untreated, would have experienced the same average outcome as the untreated did, and vice versa. This is so because randomization ensures that the independent predictors of the outcome are equally distributed between the treated and the untreated groups.

For example, take the study summarized in Table 3.1. We said in the previous chapter that exchangeability clearly does not hold in this study because $69\%$ treated versus $43\%$ untreated individuals were in critical condition $L = 1$ at baseline. This imbalance in the distribution of an independent outcome predictor is not expected to occur in a marginally randomized experiment (actually, such imbalance might occur by chance, but let us keep working under the illusion that our study is large enough to prevent chance findings).

> **Fine Point 3.2:** In Chapter 7, we will refer to these types of outcome predictors as confounders.

On the other hand, an imbalance in the distribution of independent outcome predictors $L$ between the treated and the untreated is expected by design in conditionally randomized experiments in which the probability of receiving treatment depends on $L$. The study in Table 3.1 is such a conditionally randomized experiment: the treated and the untreated are not exchangeable—because the treated had, on average, a worse prognosis at the start of the study—but the treated and the untreated are conditionally exchangeable within levels of the variable $L$.

In the subset $L = 1$ (critical condition), the treated and the untreated are exchangeable because the treated, had they remained untreated, would have experienced the same average outcome as the untreated did, and vice versa. And similarly for the subset $L = 0$. An equivalent statement: conditional exchangeability $Y^{a} \perp\!\!\!\perp A \mid L$ holds in conditionally randomized experiments because, within levels of $L$, all other outcome predictors are equally distributed between the treated and untreated groups.

Back to observational studies. When treatment is not randomly assigned by the investigators, the reasons for receiving treatment are likely to be associated with some outcome predictors. That is, like in a conditionally randomized experiment, the distribution of outcome predictors will generally vary between the treated and untreated groups in an observational study.

For example, the data in Table 3.1 could have arisen from an observational study in which doctors tend to direct the scarce heart transplants to those who need them most, i.e., individuals in critical condition $L = 1$. In fact, if the only outcome predictor that is unequally distributed between the treated and the untreated is $L$, then one can refer to the study in Table 3.1 as either:

- (i) an observational study in which the probability of treatment $A = 1$ is $0.75$ among those with $L = 1$ and $0.50$ among those with $L = 0$; or
- (ii) a (nonblinded) conditionally randomized experiment in which investigators randomly assigned treatment $A = 1$ with probability $0.75$ to those with $L = 1$ and $0.50$ to those with $L = 0$.

Both characterizations of the study are logically equivalent. Under either characterization, conditional exchangeability $Y^{a} \perp\!\!\!\perp A \mid L$ holds and standardization or IP weighting can be used to identify the causal effect.

Of course, the crucial question for the observational study is whether $L$ is the only outcome predictor that is unequally distributed between the treated and the untreated. Sadly, the question must remain unanswered, so our investigators need to be willing to work under the assumption that conditional exchangeability $Y^{a} \perp\!\!\!\perp A \mid L$ holds.

Also, note that not all variables that are unequally distributed between treatment groups need to be included in $L$. For example, heart transplants are assigned to individuals with low probability of rejecting the transplant, i.e., a heart with certain human leukocyte antigen (HLA) genes will be assigned to an individual who happens to have compatible genes. Because HLA genes are not predictors of mortality, conditional on $L$ and $A$, treatment assignment is essentially random within levels of $L$ and thus HLA needs not be considered in the analysis.

In the absence of randomization, there is no guarantee that conditional exchangeability holds. For example, suppose that, unknown to the investigators, doctors prefer to transplant hearts into nonsmokers. Consider two individuals with $L = 1$. One of them is a smoker ($U = 1$) and the other one is a nonsmoker ($U = 0$); the one with $U = 1$ has a lower probability of receiving treatment $A = 1$. When the distribution of smoking, an important outcome predictor,

### Crossover Randomized Experiments

In Fine Point 2.1, we described crossover experiments in which an individual is observed during two or more periods—say $t = 0$ and $t = 1$—and the individual receives a different treatment value in each period. We showed that individual causal effects can be identified when the following three strong conditions hold:

i) No carryover effect of treatment:

$$
Y_{it=1}^{a_0, a_1} = Y_{it=1}^{a_1}
$$

ii) The individual causal effect does not depend on time:

$$
Y_{it}^{a_t = 1} - Y_{it}^{a_t = 0} = \alpha_i \quad \text{for} t = 0, 1
$$

iii) The counterfactual outcome under no treatment does not depend on time:

$$
Y_{it}^{a_t = 0} = \beta_i \quad \text{for} t = 0, 1
$$

No randomization was required. We now turn our attention to crossover randomized experiments in which the order of treatment values that an individual receives is randomly assigned.

Randomized treatment assignment becomes important when, due to possible temporal effects, we do not assume iii) holds. For simplicity, assume with probability 0.5 that each individual is randomized to either $(A_{i0} = 0, A_{i1} = 1)$ or $(A_{i0} = 1, A_{i1} = 0)$. Then, under i) and ii):

- If $A_{i0} = 0$ and $A_{i1} = 1$, then $Y_{i1} - Y_{i0} = \alpha_i + r_i$.
- If $A_{i1} = 0$ and $A_{i0} = 1$, then $Y_{i0} - Y_{i1} = \alpha_i - r_i$,

where $r_i = Y_{i1}^{a_1 = 0} - Y_{i0}^{a_0 = 0}$.

Because $r_i$ is unknown, we can no longer identify individual causal effects. However, since $A_{i1}$ and $A_{i0}$ are randomized and therefore independent of $r_i$, the mean of

$$
(Y_{i1} - Y_{i0}) A_{i1} + (Y_{i0} - Y_{i1}) A_{i0}
$$

estimates the average causal effect, i.e., $\mathrm{E}[\alpha_i]$.

If we only assume i), then this mean estimates the average of the average treatment effects at times 0 and 1, i.e.,

$$
\frac{\mathrm{E}[\alpha_{i1}] + \mathrm{E}[\alpha_{i0}]}{2},
$$

where $\alpha_{it} = Y_{it}^{a_t = 1} - Y_{it}^{a_t = 0}$.

In conclusion, if assumption i) of no carryover effect holds, then a crossover experiment can be used to estimate average causal effects. However, for the type of treatments and outcomes we study in this book, the assumption of no carryover effect is implausible.

---

We use $U$ to denote unmeasured variables. Because unmeasured variables cannot be used for standardization or IP weighting, the causal effect cannot be identified when the measured variables $L$ are insufficient to achieve conditional exchangeability.

To verify conditional exchangeability, one needs to confirm that

$$
\mathrm{Pr}[Y^a = 1 \mid A = a, L = l] = \mathrm{Pr}[Y^a = 1 \mid A \neq a, L = l].
$$

But this is logically impossible because, for individuals who do not receive treatment $a$ ($A \neq a$), the value of $Y^a$ is unknown, and so the right-hand side cannot be empirically evaluated.

If the distribution of $U$ differs between the treated (with lower proportion of smokers $U = 1$) and the untreated (with higher proportion of smokers) in the stratum $L = 1$, conditional exchangeability given $L$ does not hold. Importantly, collecting data on smoking would not prevent the possibility that other imbalanced outcome predictors, unknown to the investigators, remain unmeasured.

Thus exchangeability $Y^a \perp\!\!\!\perp A \mid L$ may not hold in observational studies. Specifically, conditional exchangeability $Y^a \perp\!\!\!\perp A \mid L$ will not hold if there exist unmeasured independent predictors $U$ of the outcome such that the probability of receiving treatment $A$ depends on $U$ within strata of $L$.

Worse yet, even if conditional exchangeability $Y^a \perp\!\!\!\perp A \mid L$ held, the investigators cannot empirically verify that is actually the case. How can they check that the distribution of smoking is equal in the treated and the untreated if they have not collected data on smoking? What about all the other unmeasured outcome predictors $U$ that may also be differentially distributed between the treated and the untreated? When analyzing an observational study under conditional exchangeability, we must hope that our expert knowledge guides us correctly to collect enough data so that the assumption is at least approximately true.

Investigators can use their expert knowledge to enhance the plausibility of the conditional exchangeability assumption. They can measure many relevant variables $L$ (e.g., determinants of the treatment that are also independent outcome predictors), rather than only one variable as in Table 3.1, and then assume that conditional exchangeability is approximately true within the strata defined by the combination of all those variables $L$.

Unfortunately, no matter how many variables are included in $L$, there is no way to test that the assumption is correct, which makes causal inference from observational data a risky task. The validity of causal inferences requires that the investigators’ expert knowledge is correct. This knowledge, encoded as the assumption of exchangeability conditional on the measured covariates, supplements the data in an attempt to identify the causal effect of interest.

## 3.3 Positivity

The positivity condition is sometimes referred to as the experimental treatment assumption.

**Positivity** : $\operatorname{Pr}[A = a \mid L = l] > 0$ for all values $l$ with $\operatorname{Pr}[\mathcal{L} = l] \neq 0$ in the population of interest.

Some investigators plan to conduct an experiment to compute the average effect of heart transplant $A$ on 5-year mortality $Y$. It goes without saying that the investigators will assign some individuals to receive treatment level $A = 1$ and others to receive treatment level $A = 0$. Consider the alternative: the investigators assign all individuals to either $A = 1$ or $A = 0$. That would be silly. With all the individuals receiving the same treatment level, computing the average causal effect would be impossible. Instead we must assign treatment so that, with near certainty, some individuals will be assigned to each of the treatment groups. In other words, we must ensure that there is a probability greater than zero—a positive probability—of being assigned to each of the treatment levels. This is the positivity condition.

We did not emphasize positivity when describing experiments because positivity is taken for granted in those studies. In marginally randomized experiments, the probabilities $\operatorname{Pr}[A = 1]$ and $\operatorname{Pr}[A = 0]$ are both positive by design. In conditionally randomized experiments, the conditional probabilities $\operatorname{Pr}[A = 1 \mid L = l]$ and $\operatorname{Pr}[A = 0 \mid L = l]$ are also positive by design for all levels of the variable $L$ that are eligible for the study. For example, if the data in Table 3.1 had arisen from a conditionally randomized experiment, the conditional probabilities of assignment to heart transplant would have been $\operatorname{Pr}[A = 1 \mid L = 1] = 0.75$ for those in critical condition and $\operatorname{Pr}[A = 1 \mid L = 0] = 0.50$ for the others. Positivity holds, conditional on $L$, because neither of these probabilities is 0 (nor 1, which would imply that the probability of no heart transplant $A = 0$ would be $0$).

Thus we say that there is positivity if $\operatorname{Pr}[A = a \mid L = l] > 0$ for all $a$ involved in the causal contrast. Actually, this definition of positivity is incomplete because, if our study population were restricted to the group $L = 1$, then there would be no need to require positivity in the group $L = 0$. Positivity is only needed for the values $l$ that are present in the population of interest.

In addition, positivity is only required for the variables $L$ that are required for exchangeability. For example, in the conditionally randomized experiment of Table 3.1, we do not ask ourselves whether the probability of receiving treatment is greater than 0 in individuals with blue eyes because the variable “having blue eyes” is not necessary to achieve exchangeability between the treated and the untreated. (The variable “having blue eyes” is not an independent predictor of the outcome $Y$ conditional on $L$ and $A$, and was not even used to assign treatment.) That is, the standardized risk and the IP weighted risk are equal to the counterfactual risk after adjusting for $L$ only; positivity does not apply to variables that, like “having blue eyes”, do not need to be adjusted for.

In observational studies, neither positivity nor exchangeability are guaranteed. For example, positivity would not hold if doctors always transplant a heart to individuals in critical condition $L = 1$, i.e., if $\mathrm{Pr}[A = 0 \mid L = 1] = 0$, as shown in Figure 3.1. A difference between the conditions of exchangeability and positivity is that positivity can sometimes be empirically verified (see Chapter 12). For example, if Table 3.1 corresponded to data from an observational study, we would conclude that positivity holds for $L$ because there are people at all levels of treatment (i.e., $A = 0$ and $A = 1$) in every level of $L$ (i.e., $L = 0$ and $L = 1$).

Our discussion of standardization and IP weighting in the previous chapter was explicit about the exchangeability condition, but only implicitly assumed the positivity condition (explicitly in Technical Point 2.3). Our previous definitions of standardized risk and IP weighted risk are actually only meaningful when positivity holds. To intuitively understand why the standardized and IP weighted risk are not well-defined when the positivity condition fails, consider Figure 3.1. If there were no untreated individuals ($A = 0$) with $L = 1$, the data would contain no information to simulate what would have happened had all treated individuals been untreated because there would be no untreated individuals with $L = 1$ that could be considered exchangeable with the treated individuals with $L = 1$. See Technical Point 3.1 for details.

![image_07](../../images/image_07.png)

> 图 3.1

## 3.4 一致性：首先定义反事实结果

首先，我们需要定义反事实结果。在因果推断中，一致性（Consistency）是一个关键假设，它确保了观察到的结果与潜在结果之间的对应关系。

具体来说，一致性假设指出：对于某个个体，如果它实际接受了某种处理（例如，处理 $T = 1$），那么它在该处理下的潜在结果 $Y(1)$ 就等于它实际观察到的结果 $Y$。形式化地，可以表示为：

$$
\text{如果} T = t, \text{则} Y = Y(t)
$$

这个假设看似简单，但在实际应用中却至关重要。它排除了多种处理版本（multiple versions of treatment）的可能性，即确保处理定义是明确的、唯一的。

为了更清晰地理解，我们可以考虑一个反例：假设我们研究“服用阿司匹林”对“头痛缓解”的影响。一致性假设要求，如果一个人实际服用了阿司匹林，那么他服用的阿司匹林必须与我们在定义中设定的“服用阿司匹林”完全一致——包括剂量、品牌、服用方式等。如果存在不同版本的阿司匹林（例如，不同剂量），那么观察到的结果 $Y$ 可能无法准确对应到我们感兴趣的潜在结果 $Y(1)$。

> **注** ：一致性假设是因果推断的基石之一，它连接了反事实框架与观察数据。没有这个假设，我们无法从观察到的数据中推断出因果关系。

For an earlier discussion of the issues described in Sections 3.4 and 3.5, see the text and references in Hernán (2016), and in Robins and Weissman (2016).

Consistency of counterfactuals means that the observed outcome $Y$ for every treated individual equals her outcome if she had received treatment, $Y^{a=1}$, and that the observed outcome $Y$ for every untreated individual equals her outcome if she had remained untreated, $Y^{a=0}$. That is, consistency is the assumption that $Y = Y^{A}$, where $Y^{A}$ is the counterfactual $Y^{a}$ with $a$ evaluated at an individual’s actual treatment $A$.

Consistency is a fundamental condition for causal inference because it links the counterfactuals $Y^{a}$ to the observed data $Y$. In this book, we take the counterfactuals $Y^{a}$ as primitives for the observed outcome $Y$. That is, the observed outcome $Y$ is derived from (i.e., is a function of) the primitives through the formula $Y = Y^{A}$. For a binary treatment $A$, it is easy to see that $Y^{A}$ depends only on the primitives since then $Y^{A} = A Y^{a=1} + (1 - A) Y^{a=0}$.

Consistency may seem obviously true in some cases. For example, if you take an aspirin $A = 1$ and you die ($Y = 1$), then your counterfactual outcome $Y^{a=1}$ under aspirin must equal 1. But consistency cannot be taken for granted in observational studies, as we explain below.

The consistency condition has two main components:

1. A precise definition of the counterfactual outcomes $Y^{a}$ via the specification of the superscript.

> ### Technical Point 3.1

**Positivity for standardization and IP weighting.**  
We have defined the standardized mean for treatment level $a$ as

$$
\sum_{l} \operatorname{E} \left[ Y \mid A = a, L = l \right] \operatorname{Pr} \left[ L = l \right].
$$

However, this expression can only be computed if the conditional quantity $\operatorname{E} [ Y \mid A = a, L = l ]$ is well defined, which will be the case when the conditional probability $\operatorname{Pr} \left[ A = a \mid L = l \right]$ is greater than zero for all values $l$ that occur in the population. That is, when **positivity** holds.

> **Note:** The statement $\operatorname{Pr} \left[ A = a \mid L = l \right] > 0$ for all $l$ with $\operatorname{Pr} \left[ L = l \right] \neq 0$ is effectively equivalent to $f \left[ a \mid L \right] > 0$ with probability 1.

Therefore, the standardized mean is defined as

$$
\sum_{l} \operatorname{E} \left[ Y \mid A = a, L = l \right] \Pr \left[ L = l \right] \quad \text{if} \Pr \left[ A = a \mid L = l \right] > 0 \text{for all} l \text{with} \Pr \left[ L = l \right] \neq 0.
$$

and is undefined otherwise. The standardized mean can be computed only if, for each value of the covariate $L$ in the population, there are some individuals that received the treatment level $a$.

The IP weighted mean $\operatorname{E}\left[ \frac{I(A=a)Y}{f[A|L]} \right]$ is no longer equal to $\operatorname{E}\left[ \frac{I(A=a)Y}{f[a|L]} \right]$ when positivity does not hold. Specifically, $\operatorname{E}\left[ \frac{I(A=a)Y}{f[a|L]} \right]$ is undefined because the undefined ratio $\frac{0}{0}$ occurs in computing the expectation. On the other hand, the IP weighted mean $\operatorname{E}\left[ \frac{I(A=a)Y}{f[A|L]} \right]$ is always well defined since its denominator $f[A|L]$ can never be zero. However, it is now a biased estimate of the counterfactual mean even under exchangeability when positivity fails to hold.

In particular, $\operatorname{E}\left[ \frac{I(A=a)Y}{f[A|L]} \right]$ is equal to

$$
\operatorname{Pr}[L \in Q(a)] \sum_{l} \mathbf{E}[Y | A=a, L=l, L \in Q(a)] \operatorname{Pr}[L=l | L \in Q(a)]
$$

where $Q(a) = \{l; \operatorname{Pr}(A=a | L=l) > 0 \}$ is the set of values $l$ for which $A=a$ may be observed with positive probability. Therefore, under exchangeability, $\operatorname{E}\left[ \frac{I(A=a)Y}{f[A|L]} \right]$ equals $\operatorname{E}[Y^{a} | L \in Q(a)] \operatorname{Pr}[L \in Q(a)]$.

From the definition of $Q(a)$, $Q(0)$ cannot equal $Q(1)$ when $A$ is binary and positivity does not hold. In this case the contrast

$$
\mathfrak{X}\left[ \frac{I(A=1)Y}{f[A|L]} \right] - \operatorname{E}\left[ \frac{I(A=0)Y}{f[A|L]} \right]
$$

has no causal interpretation, even under exchangeability, because it is a contrast between two different groups. Under positivity, $Q(1) = Q(0)$ and the contrast is the average causal effect if exchangeability holds.

Robins and Greenland (2000) argued that well-defined counterfactuals, or mathematically equivalent concepts, are necessary for meaningful causal inference.

> $a$, and (2) the linkage of the counterfactual outcomes to the observed outcomes. This section deals with the first component of consistency.

The methodology for causal inference described in this book is licensed by the existence of well-defined counterfactual outcomes $Y^{a}$. If $Y^{a}$ is well-defined for $a=1$ and $a=0$ for all individuals in the population, then the causal effect $\operatorname{Pr}\left\lfloor Y^{a=1}=1 \right\rfloor - \operatorname{Pr}\left\lfloor Y^{a=0}=1 \right\rfloor$ is well-defined.

A key question is then, “How do we know that the counterfactuals are well defined?” A natural and desirable sufficient condition is that, if $a$ corresponds to a well-defined intervention, then $Y^{a}$ is well-defined as being the outcome had the intervention $a$ been performed.

To illustrate the concept of well-defined interventions, consider two ideal randomized experiments, conducted among individuals from the same population, in which participants are randomly assigned to either heart transplant $a=1$ or medical therapy $a=0$. All individuals in the population are eligible to receive either $a=1$ or $a=0$.

In the first randomized experiment, the investigators wrote a protocol in which the two interventions of interest were described in detail. The investigators specified that individuals assigned to heart transplant were to receive certain pre-operative procedures, anesthesia, surgical technique, post-operative care, and immunosuppressive therapy in an attempt to ensure that each individual assigned to heart transplant receives the same treatment $a=1$, and similarly for $a=0$. Had the protocol not specified these details, it is possible that each doctor had conducted a different version of “heart transplant”, perhaps using their preferred surgical technique or immunosuppressive therapy.

> Fine Point 1.2 introduced the concept of multiple versions of treatment. Such a trial is often referred to as a pragmatic trial.

In this study, the term “heart transplant” corresponds to a well-defined intervention $a=1$ that is defined as the sequential implementation of the components $a_{0}=1$ (assignment to heart transplant), $a_{1}$ (pre-specified preoperative procedures), $a_{2}$ (anesthesia), and $a_{3}$ (surgical technique) for all individuals.

In the second randomized experiment, the investigators purposely chose not to provide a precise specification of the interventions so that the interventions implemented in the trial would reflect what happens in real world settings. In this study, the term “heart transplant” corresponds to a well-defined intervention $a=1$ that is defined as “assignment to heart transplant” (i.e., $a_{0}=1$ for all individuals) followed by observation of whatever unfolds after the intervention. That is, the values of $a_{1}$, $a_{2}$, $a_{3}$ for each individual assigned to heart transplant are not specified in advance, but rather they will be the values that naturally occur in the healthcare system: $a_{1}=A_{1}$, $a_{2}=A_{2}$, $a_{3}=A_{3}$ for each individual.

> Formally, $Y^{a=1}$ is the joint counterfactual $Y^{a_{0}=1, a_{1}, a_{2}, a_{3}}$ in the first experiment and $Y^{a_{0}=1, A_{1}^{a_{0}=1}, A_{2}^{a_{0}=1}, A_{3}^{a_{0}=1}}$ in the second experiment. For individuals assigned to heart transplant $A=1$ in the second experiment, this latter counterfactual is equal to $Y^{A_{0}=1, A_{1}, A_{2}, A_{3}}$, which is equal to the observed outcome $Y$. See Technical Point 3.2. Chapter 4 discusses several factors that may affect the transportability of causal effects.

These two examples illustrate a common situation in practice: the same treatment name is used with different meanings. In the first experiment, the label “heart transplant” $a=1$ refers to the sequential implementation of component interventions $(a_{0}, a_{1}, a_{2}, a_{3})$. In the second experiment, it refers to the implementation of a point intervention $a_{0}$ after which investigators let the world run its course. Therefore, the values $(A_{1}, A_{2}, A_{3})$ will depend on the characteristics of the population and the setting in which the experiment takes place.

But, even though each experiment implements a different version of $a=1$, the corresponding intervention $a=1$ is well-defined in the protocol of each experiment. This implies that the counterfactual outcome $Y^{a=1}$ is well-defined for all individuals in each experiment as the individual’s outcome if the instructions for intervention $a=1$ in the protocol of that experiment were followed (and analogously for $Y^{a=0}$). The counterfactual outcomes $Y^{a}$, however, will likely differ between the two experiments. If that is the case, then the causal effect

$$
\operatorname{Pr}\left\lfloor Y^{a=1}=1 \right\rfloor - \operatorname{Pr}\left\lfloor Y^{a=0}=1 \right\rfloor
$$

though well-defined in each experiment, will also differ between the two experiments. This raises the question of which of the two causal effects is preferred by consumers of the research.

Specifically, the magnitude of the causal effect from the first experiment may be more similar to that of the same effect in other populations (because of the precise specification of the components of the intervention) than the magnitude of the causal effect from the second experiment is (because the natural values of the post-intervention variables may differ across populations). Therefore, the causal inferences from the first experiment may be easier to transport to other populations than the causal inferences from the second experiment are. Note that this is a discussion about transportability of the causal inference, not about whether the causal effects are well defined in each study population.

Interventions are well-defined when they correspond to actions that can be described as part of the protocol of an experiment. However, well-defined interventions do not have to be perfectly specified. In fact, perfect specification of the interventions is not generally possible. Consider again the first experiment.

Its protocol specified the components $a_{0}$, $a_{1}$, $a_{2}$, $a_{3}$, but not the training of the surgeon performing heart transplants.

### Fine Point 3.3

In this section, we explore the subtle aspects of mathematical notation and formatting.

#### Key Observations

- **Inline formulas** should be enclosed with single dollar signs: $E = mc^2$.
- **Display formulas** must be placed between double dollar signs:

$$
\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}
$$

> **Note**  
> Always use `\vert` instead of `|` inside formulas to avoid table syntax conflicts.

#### Example Table

| Symbol          | Meaning          | Example              |
| :-------------- | :--------------- | :------------------- |
| $\alpha$        | Angle            | $\sin(\alpha)$       |
| $\beta$         | Beta coefficient | $\beta = 0.75$       |
| $\vert x \vert$ | Absolute value   | $\vert -3 \vert = 3$ |

#### Algorithm Pseudocode

1. Initialize variable $x = 0$
2. While $x < 10$:
   - Increment $x$ by 1
   - Print $x$

#### Image Reference

![Sample Image](https://example.com/sample.png)

#### Final Remarks

Ensure consistent spacing around formulas and tables for better readability. Use **bold** for emphasis and _italic_ for secondary highlights.

Protocols open to interpretation. It is possible that $\operatorname*{Pr}[Y^{a=1}=1]$ differs between two randomized experiments with identical populations and protocols. To see this, consider the following scenario.

In both experiments, individuals assigned to $a = 1$ underwent a surgical operation according to the instructions in the protocol. However, the protocol did not specify how to match patients with surgeons. In the first experiment, individuals assigned to $a = 1$ were referred to and operated on by experienced surgeons if they were high risk patients, and by less experienced surgeons if they were low risk patients. Because of this, almost no patients died and $\operatorname*{Pr}[Y^{a=1}=1]$ was close to 0.

In contrast, in the second experiment, individuals assigned to $a = 1$ were referred to a surgeon without regard to the patient’s risk and the surgeon’s experience. In this study $\operatorname*{Pr}[Y^{a=1}=1]$ is far from zero because many high-risk patients were operated on by inexperienced surgeons.

By definition, lack of exchangeability cannot explain the difference in $\operatorname*{Pr}[Y^{a=1}=1]$ because both experiments were randomized. Rather, the difference is explained by the different versions of treatment used in each trial. Because the protocol did not specify how to match patients with surgeons, the two trials ended up with different results.

Generally, these discrepancies may arise if the protocol leaves room for $a = 1$ to include several versions of treatment with different causal effects on the outcome of interest, and different versions of treatment are used in each experiment.

The phrase “no causation without manipulation” (Holland 1986) is often used to capture the idea that meaningful causal inference requires sufficiently well-defined interventions. However, bear in mind that sufficiently well-defined interventions may not be humanly feasible, or practicable, interventions at a particular time in history. For example, the effect of genetic variants on disease was considered sufficiently well defined even before the existence of technology for genetic modification.

Experienced surgeons may have participated in the study. Because scant transplant experience is known to affect post-transplant mortality, the risk $\operatorname*{Pr}[Y^{a=1}=1]$ had all individuals received treatment according to the protocol will depend on the unknown distribution of experience of the participating surgeons. Even if the experiment had specified the surgeons’ training, we could always find something else that remained unspecified, or open to interpretation, in the protocol.

Because the interventions cannot be perfectly specified, the value of the average causal effect $\operatorname*{Pr}[Y^{a=1}=1] - \operatorname*{Pr}[Y^{a=0}=1]$ is expected to vary across populations. For example, in our heart transplant experiments, the average causal effect in a new community with a different distribution of surgical experience will differ from the effect in the trial population, even if the new population follows the exact same protocol as in the trial. In fact, the value of the average causal effect may differ even between two experiments conducted in the same population and with the same protocol, when the protocol admits different interpretations. See **Fine Point 3.3** for an example.

The more precisely we define the interventions, the more precise our causal questions are and, generally, the easier it will be to transport causal inferences from one population to another population. Of course, components of the interventions that have no effect on the outcome cannot affect transportability. For example, in our heart transplant experiment, we do not need to worry about the color of the surgeons’ scrubs (green or blue) because scientists agree that varying the color of the scrubs would not lead to different outcomes.

All the above considerations apply to both randomized experiments and observational studies. Regardless of how the data are generated, well-defined interventions $a$ imply well-defined counterfactuals $Y^{a}$. For an observational study on the effect of heart transplant, we could specify the intervention $a = 1$ as a precisely defined sequence of components (as in the first experiment above) or as a minimally defined intervention that reflects what happens in the real world (as in the second experiment). For either a randomized experiment or an observational study on the effect of different running strategies, we might specify duration, frequency, and intensity of running under each strategy. We would not specify the direction of running (clockwise or counterclockwise) around the neighborhood’s park because scientists agree that, when the wind is not blowing, the direction of running is irrelevant.

### Fine Point 3.4

**Possible worlds.** Philosophers of science have proposed counterfactual theories based on the concept of “possible worlds” (Stalnaker 1968, Lewis 1973). The counterfactual $Y^{a}$ is defined to be the value of $Y$ in the world in which the individual received the treatment that is closest to the actual world. In particular, these philosophers assume that $Y^{a} = Y$ if $A = a$ because the closest possible world to the actual world is itself. Hence, under their definition of counterfactuals, consistency always holds.

When $A \neq a$, the “closest possible world” and thus the counterfactual $Y^{a}$ are always somewhat ill-defined and vague. Nonetheless, Lewis noted that his definition of counterfactuals is often useful. Robins and Greenland (2000) agreed but also argued that the concept of well-defined interventions should replace the concept of the closest possible world because, in observational studies, counterfactuals are vague and ill-defined to the degree that one fails to make precise the hypothetical interventions and causal contrasts under consideration.

A common difference between randomized experiments and observational studies is the degree of agreement about the interventions that define the causal effect. In ideal experiments, the interventions are specified in the protocol and are actually implemented, so investigators have a common understanding of what the interventions are, thus making the counterfactuals $Y^{a}$ well-defined. In observational studies, because the interventions are not necessarily prespecified in the real world, investigators may have different views of what the interventions of interest are. For example, investigators who talk about the effect of “heart transplant” $a = 1$ without explicitly defining the intended intervention may be referring to a variety of causal effects, e.g., the effect of a precisely defined sequence of components or the effect of a minimally defined intervention that reflects what happens in the real world. As a result, the counterfactuals $Y^{a}$ are as yet ill-defined and the causal effect of “heart transplant” is too vague a concept. However, this vagueness can sometimes be overcome by having our expert investigators all focus on one specific intervention at a time. It is then possible that each intervention might be considered well-defined, each with their own, but differing counterfactual $Y^{a}$.

For the counterfactuals $Y^{a}$ to be sufficiently well defined, we also need a well-defined eligible population of individuals who are eligible to receive both $a = 1$ and $a = 0$. Hernán and Taubman (2008) discuss the tribulations of two world leaders—a despotic king and a clueless president—when considering “the effect of obesity” in their countries.

Investigators agree that a particular intervention $a$ is sufficiently well defined when, for all practical purposes, no meaningful vagueness remains for the counterfactuals $Y^{a}$. Which begs the question “How do we know that an intervention is sufficiently well-defined for our purposes?” Or, equivalently “How do we know that no meaningful vagueness remains?” The answer is “We don’t.” Declaring an intervention sufficiently well-defined is a matter of agreement among a group of experts based on the available substantive knowledge at a particular time in history. However, even if experts agree now about a particular intervention being sufficiently well defined, they may be proven wrong in the future when new knowledge is generated. Thus, the term “sufficiently well-defined intervention” relies on available knowledge. **Fine Point 3.4** links this discussion with previous proposals.

A frequent problem arises when investigators wish to quantify the causal effect of changes in biological states (e.g., blood pressure, LDL-cholesterol, body weight) or social factors (e.g., socioeconomic status). The problem is that such states and factors are not subject to direct intervention, but can only be changed by intervening on their causes. For example, consider “the effect of becoming obese on myocardial infarction”. The quoted text does not have a meaningful interpretation because the counterfactual outcome is ill-defined. One might think that these effects would become well defined if we specified the start and end of the intervention (e.g., age 40 years through age 50 years) and the procedure by which the state or factor would be changed (e.g., medications, surgery, diet, exercise). But, if we specified all these details, we would be describing the effect of whatever interventions we are specifying rather than the effect of, say, obesity.

Whether causal effects are ill-defined depends on the outcome. Consider the effect of obesity on job discrimination—as measured by the proportion of job applicants called for a personal interview after the employer reviews the applicant’s resume and photograph. Because the treatment is “obesity as perceived by the employer”, the mechanisms that led to obesity may be irrelevant.

However, consider now “the effect of blood pressure $A$ on stroke $Y$”. Because a change in blood pressure $A$ can only be brought about by specific interventions that affect blood pressure (e.g., different types of antihypertensive medications, exercise, diet), then one would expect that a counterfactual $

## 3.5 Consistency: Second, link counterfactuals to the observed data

For an expanded discussion of practical problems that arise when using observational healthcare databases to study the effect of heart transplant, see Madenci et al. (2024).

As a reminder, the consistency condition states that $Y^a = Y^A = Y$ for all individuals with $A = a$. In the previous section, we discussed the first component of consistency: sufficiently well-defined counterfactual outcomes $Y^a$ such that no meaningful vagueness remains. In this section, we discuss the second component of consistency: the linkage of counterfactual outcomes to observed outcomes, i.e., the “equal” sign in $Y^a = Y$ for individuals with $A = a$.

When the intervention $a$ was actually implemented by the investigators, as per the protocol of an experiment, the linkage of counterfactual to observed outcomes is uncontroversial. For an individual who received treatment value $A = a$, the observed outcome $Y = Y^A$ equals, by definition, the counterfactual outcome $Y^a$. For example, in the randomized experiments of the previous section, consistency held under the version of “heart transplant” that was implemented in each experiment.

A similar reasoning applies to observational studies when an intervention was actually implemented in the real world, even if the intervention was not implemented by the investigators. Suppose we collected data on transplant-eligible individuals with heart disease who were assigned, as part of their medical care, to either heart transplant ($A = 1$) or medical therapy ($A = 0$) at a particular time in a particular place. The definition of the intervention “heart transplant” $a = 1$ corresponds to whatever procedures followed assignment to heart transplant for each individual at that time in that place, and similarly for the intervention “medical therapy” $a = 0$. For an individual in the study with $A = 1$, the counterfactual outcome $Y^{a=1}$ under heart transplant equals

> **Fine Point 3.5**

The causal effect of states or factors. Sometimes experts agree that $A$ has a causal effect on $Y$ even though the counterfactual outcome $Y^{a}$ makes no reference to well-defined interventions $a$. An example discussed in the main text is when $A$ is blood pressure and $Y$ is stroke. One way to resolve this apparent contradiction is to interpret the experts’ statement “blood pressure causes stroke” as implying that there exists some intervention $D$ that affects $A$ but has no (direct) effect on $Y$ except through $A$. In the literature, the expressions “$D$ has no (direct) effect on $Y$ except through $A$” and “the effect of $D$ on $Y$ is completely mediated by $A$” are synonymous and used interchangeably. The latter expression is closely related to treatment variation irrelevance (Fine Point 1.2) with irrelevant factor $D$ and treatment $A$. Chapter 23 discusses causal mediation.

We expect many experts will accept that their statement implies the existence of interventions with no direct effect. However, under our counterfactual model, asserting that $D$ has no direct effect on $Y$ except through $A$ is logically equivalent to asserting that both the counterfactuals $Y^{a}$ and the joint counterfactuals $Y^{d, a}$ are well-defined and equal. In our example, it is known that antihypertensive medications, exercise, and diet all change blood pressure $A$. Of these, the intervention $D$ might be various antihypertensive medications, but $D$ cannot be diet or exercise which are known to affect the risk of stroke through pathways other than blood pressure.

However, for a given blood pressure $a$, experts may not necessarily believe that the joint counterfactual $Y^{d, a}$ is well-defined for every individual in the population. Consider two examples with $D$ being an anti-hypertensive medication. In the first, for some individuals, there may be an (unknown) individual-specific dose $D_{\mathrm{max}}$ above which $D$ may have direct effects on $Y$, e.g., because of clinical cardiotoxicity. In the second, for some individuals, there may be an (unknown) individual-specific dose $D_{\mathrm{max}}$ above which $D$ does not have any incremental (blood pressure lowering) effect on $A$. In both cases, $Y^{a}$ is well defined only for $a$ greater than $A^{D_{\mathrm{max}}}$, i.e., the counterfactual $A^{d}$ evaluated at $d = D_{\mathrm{max}}$. It follows that the conditional average causal effect

$$
\mathrm{E}[Y^{a} - Y^{a^{\prime}} \mid \min(a, a^{\prime}) > A^{D_{\mathrm{max}}}]
$$

is well defined but the population average effect $\mathrm{E}[Y^{a} - Y^{a^{\prime}}]$ is not.

However, even though well defined,

$$
\mathrm{E}[Y^{a} - Y^{a^{\prime}} \mid \min(a, a^{\prime}) > A^{D_{\mathrm{max}}}]
$$

is not identifiable from the data because, e.g., if $a^{\prime} > a$, then for an individual with $A = a^{\prime}$ and no clinical cardiotoxicity, we know $a^{\prime} > A^{D_{\mathrm{max}}}$, but we cannot learn whether $a > A^{D_{\mathrm{max}}}$ and thus whether the individual is in the group defined by $\min(a, a^{\prime}) > A^{D_{\mathrm{max}}}$. Note that, if there were both known clinical (known) and subclinical (unknown) toxicity, then we cannot even learn whether an individual with $A = a^{\prime}$ and no clinical cardiotoxicity has $a^{\prime} > A^{D_{\mathrm{max}}}$.

This problem does not arise when considering the effect of a change in blood pressure $\Delta$ that is very close to $0$. In that case, the counterfactual $Y^{\Delta}$ will be well defined for (essentially) all individuals and the average causal effect $\mathrm{E}[Y^{\Delta} - Y^{\Delta = 0}]$ is well defined.

The problem of having a well-defined intervention $a$ but not having anyone in the population with $A = a$ can be viewed as an extreme form of non-positivity. Her observed outcome $Y = Y^{A}$. Of course, the causal effect targeted by this observational study will be of questionable relevance for other populations if the investigators cannot approximately characterize what “heart transplant” $a = 1$ means in this setting.

Therefore, with observational data, the choice of interventions for the study depends on the available data. For example, suppose that the investigators of an observational study carefully define an intervention “heart transplant” $a = 1$ that specifies the exact pre-operative procedures, anesthesia, surgical technique, post-operative procedures, and immunosuppressive therapy. However, the only information on heart transplant in the data is an indicator $B$ of whether a person did or did not undergo a heart transplant. Then the well-defined counterfactual outcome $Y^{a = 1}$ for an individual with $B = 1$ is not necessarily equal to the individual’s observed outcome $Y$.

Note that we used a different letter to refer to the (hypothetical) intervention $a = 1$ and to the (observed) variable $B$ in the data. Because the consistency condition states that $Y^{a} = Y^{A} = Y$ for all individuals with $A = a$, using the same letter for the observed variable and the hypothetical intervention is reserved for cases in which the observed value $A = a$ for each individual.

### Fine Point 3.6

**Attributable fraction.** We have described effect measures like the causal risk ratio $\operatorname{Pr}[Y^{a = 1} = 1] / \operatorname{Pr}[Y^{a = 0} = 1]$ and the causal risk difference $\operatorname{Pr}[Y^{a = 1} = 1] - \operatorname{Pr}[Y^{a = 0} = 1]$, which compare the counterfactual risk under treatment $a = 1$ with the counterfactual risk under treatment $a = 0$. However, one could also be interested in measures that compare the observed risk with the counterfactual risk under either treatment $a = 1$ or $a = 0$. This latter contrast allows us to compute the proportion of cases that are attributable to treatment in an observational study, i.e., the proportion of cases that would not have occurred had treatment not occurred.

For example, suppose that all 20 individuals in our population attended a dinner in which they were served either ambrosia ($A = 1$) or nectar ($A = 0$). The following day, 7 of the 10 individuals who received $A = 1$, and 1 of the 10 individuals who received $A = 0$, were sick. For simplicity, assume exchangeability of the treated and the untreated so that the causal risk ratio is $0.7 / 0.1 = 7$ and the causal risk difference is $0.7 - 0.1 = 0.6$. (In conditionally randomized experiments, one would compute these effect measures via standardization or IP weighting.) It was later discovered that the ambrosia had been contaminated by a flock of doves, which explains the increased risk summarized by both the causal risk ratio and the causal risk difference. We now address the question: “What fraction of the cases was attributable to consuming ambrosia?”

In this study we observed 8 cases, i.e., the observed risk was $\operatorname{Pr}[Y = 1] = 8 / 20 = 0.4$. The risk that would have been observed if everybody had received $a = 0$ is $\operatorname{Pr}[Y^{a = 0} = 1] \overset{\cdot}{=} 0.1$. The difference between these two risks is $0.4 - 0.1 = 0.3$. That is, there is an excess $30\%$ of the individuals who did fall ill but would not have fallen ill if everybody in the population had received $a = 0$ rather than their treatment $A$. Because $0.3 / 0.4 = 0.75$, we say that $75\%$ of the cases are attributable to treatment $a = 1$: compared with the 8 observed cases, only 2 cases would have occurred if everybody had received $a = 0$. This excess fraction or **attributable fraction** is defined as

$$
\frac{\operatorname{Pr}[Y = 1] - \operatorname{Pr}[Y^{a = 0} = 1]}{\operatorname{Pr}[Y = 1]}.
$$

## 3.6 The Target Trial

The target trial—or its logical equivalents—has long been central to the causal inference framework. Dorn (1953), Wold (1954), Cochran (1972), Rubin (1974), Feinstein (1971), and Dawid (2000) used the concept. Robins (1986) generalized it for time-varying treatments. Hernán and Robins (2016) specified the key components of the target trial. The acronym PICO (Population, Intervention, Comparator, Outcome) is sometimes used to summarize some of those components (Richardson et al., 1995).

> **Fine Point 5.4:** See Fine Point 5.4 for a discussion of the excess fraction in the context of the sufficient-component-cause framework.

The excess fraction is generally different from the etiologic fraction, another version of the attributable fraction which is defined as the proportion of cases mechanically caused by exposure. For example, suppose the untreated ($A = 0$) would have had 7 cases if they have been treated, but these 7 cases would not have contained the 1 untreated case that actually occurred—i.e., treatment produces 7 cases but prevents 1 case. Also suppose that, if untreated, the treated would have had only 1 case but different from the 7 cases they actually had. Then the excess fraction would not be equal to the etiologic fraction. Here the excess fraction is a lower bound on the etiologic fraction. Because the etiologic fraction does not rely on the concept of excess cases, it can only be computed in randomized experiments under strong assumptions. See Greenland and Robins (1988) and Robins and Greenland (1989).

Corresponds to the well-defined intervention $a$. In other words, being able to describe a well-defined intervention $a$ is not sufficient to achieve consistency. We also need to be able to link the well-defined counterfactual outcomes $Y^a$ to the observed outcomes $Y$.

Conversely, that a variable $A$ happens to be recorded in a data set does not guarantee that $A = a$ can be linked to a well-defined intervention $a$. That is, measuring a variable $A$ does not guarantee that "the causal effect of $A$" is a meaningful concept because the corresponding interventions are ill-defined as described in the previous section.

> **Fine Point 3.6:** Fine Point 3.6 describes how to use observational data to compute the proportion of cases attributable to treatment.

Achieving consistency may be challenging in observational studies. A good practice is to make our reasoning as transparent as possible, so that others can directly challenge our arguments for consistency and our interpretation of the results. The next section describes a procedure to increase this transparency.

---

In this chapter, we have explored three conditions—exchangeability, positivity, consistency—that help equate an observational study with a conditionally randomized experiment. Therefore, when investigators assume that these three conditions hold, their observational analyses can be viewed as an attempt to emulate some (hypothetical) randomized experiment that would quantify the average causal effect of interest. We refer to that hypothetical experiment as the **target experiment** or the **target trial** .

For each causal effect that we wish to estimate using observational data, we may:

1. Specify the protocol of the target trial that we would like to, but cannot, conduct.
2. Describe how the observational data would be used to emulate that target trial.

If the emulation were successful, there would be no difference between the results from the observational study and from the target trial (had it been conducted).

Specifying the target trial is a natural way to precisely articulate the causal effect of interest. Key components of the trial's protocol are:

- Eligibility criteria
- Interventions (or, in general, treatment strategies)
- Assignment
- Outcomes
- Start and end of follow-up
- Causal contrasts

Once the causal question is articulated via the specification of the target trial protocol, investigators can focus on whether and how conditional exchangeability across treatment groups can be achieved. See Chapter 22 for an extended discussion of the target trial framework.

Therefore, a valid emulation of the target trial requires that the observational dataset includes sufficient information to identify eligible individuals, classify them into groups defined by the interventions they receive, and ascertain their outcomes during the follow-up. When using the methods described in the previous chapter—IP weighting or standardization—to compute the causal effect, the dataset also needs to include sufficient adjustment variables. Later in the book (see Chapter 16), we consider alternative identifying conditions to emulate a target trial that require other types of data.

Anchoring the observational analysis to a target trial makes the causal inference relevant for decision makers—policy makers, clinicians, regulators, you... This is so because decisions are choices between two or more possible courses of action—e.g., heart transplant or no heart transplant—and the target trial revolves around the contrast of the outcomes under two or more well-defined interventions. Therefore, decision makers concerned with actionable causal inference may view the target trial framework as a natural starting point.

> **Fine Point 3.7:** See Fine Point 3.7 for additional discussion.

A question that often arises is whether the target trial framework can be applied to the effect of changes in states and factors. As an example, consider "the causal effect of weight loss" on mortality in individuals who are obese and do not smoke at age 40. As discussed in the previous sections, this causal effect is ill-defined because the interventions that define the corresponding counterfactual outcomes are not well defined. Hence, the target trial cannot be specified.

One possible reaction to ill-defined counterfactual outcomes is shifting the objective of the data analysis from causal inference to non-causal prediction. Finding that obese individuals have a higher mortality risk than nonobese individuals means that obesity is a predictor of—is associated with—mortality. This is an important piece of information to identify individuals at high risk of mortality. By saying that obesity predicts—is associated with—mortality, we remain causally agnostic: obesity might predict mortality in the sense that...

> **Note:** For an extended discussion about the differences between prediction and causal inference, which is a form of counterfactual prediction, see Hernán, Hsu, and Healy (2019).

### Fine Point 3.7

Limits of target trial emulation. Throughout the text we use, as an example, the average causal effect of heart transplant on mortality. However, this effect can be interpreted in different ways. For example, the mean counterfactual under the treatment “heart transplant” can be interpreted as the average of the counterfactual outcomes of the $n$ eligible individuals in the population under:

- (i) an intervention in which all $n$ individuals receive treatment concurrently, or
- (ii) an intervention in which each individual $i$ receives treatment while all other $n - 1$ individuals receive the treatment that they actually received.

Interpretation (ii) is an average of $n$ interventions and is highly relevant to a physician/patient pair who have to decide whether the patient should undergo heart transplant. This is the interpretation we have been implicitly using in the book.

Interpretation (i) involves an intervention that is not well defined because it does not specify which one of the variety of ways to redesign the health system would be implemented in order to increase the supply of hearts and the capacity to perform all the transplants. If we precisely specified the redesign of the health system, then our current observational data would be inadequate because the data were generated under a health system that does not incorporate those changes. For example, a health system may provide heart transplants to all eligible individuals by being less selective about the quality of the transplanted organs, which would affect the counterfactual outcomes under heart transplant. Observational data may be insufficient to characterize the effect of scaling up an intervention for system-wide implementation.

This discussion is related to interference (Fine Point 1.1). However, unlike the problem highlighted here, the interference literature generally assumes that the counterfactual outcomes under interpretation (i) are well-defined (because no structural system changes need to be specified). Hernán et al. (2025) describe other examples in which the components of a target trial cannot be directly mapped to observational data.

> Some authors view the requirement of well-defined counterfactual outcomes—and therefore the target trial framework—as an unnecessarily severe restriction on the causal questions that can be asked. For them, “the causal effect of $A$ on $Y$” may be a well-defined quantity regardless of what $A$ and $Y$ stand for (as long as $A$ temporally precedes $Y$). See Pearl (2009), Schwartz et al. (2016), and Glymour and Spiegelman (2016).

Cigarette smoking predicts lung cancer, or in the sense that carrying a lighter predicts lung cancer. Thus the association between obesity and mortality is an interesting hypothesis-generating exercise and a motivation for further research (why does obesity predict mortality anyway?), while acknowledging the magnitude of the association does not necessarily correspond to that of a causal effect.

Another possible reaction to ill-defined counterfactual outcomes is attempting to make them less ill-defined. For example, some investigators may want to analyze observational data to characterize the relationship between weight loss and mortality as potentially causal (in some, possibly unspecified, sense). Though a target trial cannot be specified because the interventions are ill-defined, engaging with the investigators who pose such a question and asking them to articulate their causal question by specifying a target trial protocol may lead to better defined interventions. The following example illustrates how the target trial framework may help even when the interventions are ill-defined.

Consider a data analysis that compares the risk of death in obese versus non-obese individuals at age 40. If interpreted causally, that comparison corresponds implicitly to a target trial in which obese individuals are instantaneously transformed into non-obese individuals at the start of follow-up. Such a target trial cannot be emulated not only because the intervention is not well-defined (and thus the counterfactual outcomes are ill-defined), but also because very few people in the real world, if anyone, undergo such drastic instantaneous change (and thus the counterfactual outcomes cannot be linked to any observed outcomes). Had this draconian intervention been made explicit, the investigators conducting the data analyses would have likely agreed that consistency does not hold. Explicit target trial emulation prevents investigators from making implicit consistency assumptions that do not cohere with their own beliefs.

That we may not be able to define sufficiently well-defined interventions is no excuse to try to make them as less ill-defined as possible. When studying the association between weight loss and heart disease using observational data, Danaei et al. (2016) left unspecified the method used to lose weight, but they carefully specified the timing of the weight loss over many years.

The target trial framework helps investigators recognize when their data analysis implies extreme or impossible interventions. It also helps them propose modifications to the data analysis that imply less extreme interventions. We may not be able to specify the procedures that will make people lose weight (e.g., diet, exercise, a pill, surgery), but we can ensure that other components of the intervention (e.g., its timing) remain realistic. If we had longitudinal data on body weight, we can conduct a more sophisticated analysis that implies a target trial in which some individuals are assigned to lose 5% of body mass index every year, starting at age 40 and for as long as their body mass index stays over 25. (Part III of this book revolves around interventions that, like this one, are sustained over time.) Though this intervention is not yet sufficiently well-defined, it at least avoids mandating an instantaneous weight loss, which corresponds to an unreasonable intervention that cannot be connected to the available data.

When investigators embark on a causal pursuit with not sufficiently well-defined interventions, our goal is to persuade these investigators that their claim that “$A$ has a causal effect on $Y$” is essentially equivalent to the claim that there exists some, possibly unimplementable but possible, intervention $D$ whose effect on $Y$ is completely mediated by $A$, as described in Fine Point 3.5. If this is true, consistency holds, but the validity of the analysis also requires positivity and exchangeability for $A$ conditional on measured variables $L$. Because the interventions remain unspecified, the usual uncertainty regarding conditional exchangeability in observational studies is greatly exacerbated in this setting. Also, it may be hard to characterize the combinations of values of $L$ that would make it impossible to receive the intervention in the observational data, which increases the risk of an inadvertent violation of positivity.

---

### Technical Point 3.2

Recursive substitution. Given a set of variables chronologically ordered, the one-step-ahead counterfactuals are the counterfactual values of a variable when all earlier variables that could be intervened on have been intervened on. Suppose we have variables $L$, $A$, $M$, $Y$ in that chronological order and interventions on $L$, $A$, $M$ are well-defined. Then the counterfactuals $L$, $A^{l}$, $M^{l, a}$, $Y^{l, a, m}$ are the one-step-ahead counterfactuals. On the other hand, if interventions on $L$ were not well defined, the one-step-ahead counterfactuals would become $L$, $A$, $M^{a}$, $Y^{a, m}$.

All other factuals and well-defined counterfactuals can be built from (i.e., are functions of) the one step ahead counterfactuals via "recursive substitution". With one-step-ahead counterfactuals $L$, $A^{l}$, $M^{l, a}$, $Y^{l, a, m}$, we can use recursive substitution as follows:

- $A = A^{L}$
- $M^{a} = M^{L, a}$ is the one-step-ahead counterfactual $M^{l, a}$ evaluated at the observed $L$
- $M = M^{L, A} = M^{L, A^{L}}$ is $M^{l, a}$ evaluated at the observed $L$ and $A$
- $M^{l} = M^{l, A^{l}}$ is the counterfactual $M^{l, a}$ evaluated at $l$ and the counterfactual $A^{l}$
- $Y^{a} = Y^{L, a, M^{a}} = Y^{L, a, M^{L, a}}$ is the counterfactual $Y^{l, a, m}$ evaluated at $L$, $a$, and $M^{a}$
- $Y^{m} = Y^{L, A, m} = Y^{L, A^{L}, m}$
- $Y^{l} = Y^{l, A^{l}, M^{l}} = Y^{l, A^{l}, M^{l, A^{l}}}$
- $Y^{l, m} = Y^{l, A^{l}, m}$
- $Y = Y^{L, A, M} = Y^{L, A^{L}, M^{L, A^{L}}}$

The one-step-ahead counterfactuals also encode no direct effect (treatment irrelevance) assumptions. For example, if $a$ has no direct effect on $Y$ (relative to $L$ and $M$) then $Y^{l, a, m} = Y^{l, m} = Y^{l, A^{l}, m}$, and thus the one-step-ahead counterfactuals become $L$, $A^{l}$, $M^{l, a}$, $Y^{l, m}$.

Let us now return to the heart transplant experiments described in the main text. In the first experiment, $Y^{a_{0} = 1, a_{1}^{*}, a_{2}^{*}, a_{3}^{*}}$ is the counterfactual outcome under assignment to heart transplant $[a_{0} = 1]$ with the detailed intervention components $a_{1}^{*}, a_{2}^{*}, a_{3}^{*}$. The sets $\mathbb{A}_{1}$, $\mathbb{A}_{2}$, $\mathbb{A}_{3}$ are the values $(a_{1}, a_{2}, a_{3})$ which include $a_{1}^{*}, a_{2}^{*}, a_{3}^{*}$.

In the second experiment, the one-step-ahead counterfactuals are $A_{1}^{a_{0}=1}$, $A_{2}^{a_{0}=1, a_{1}}$, $A_{3}^{a_{0}=1, a_{1}, a_{2}}$, and $Y^{a_{0}=1, a_{1}, a_{2}, a_{3}}$. The outcome under assignment to heart transplant $[a_{0} = 1]$ with the natural values of the components is:

$$
Y^{a=1} = Y^{a_{0}=1, A_{1}^{a_{0}=1}, A_{2}^{a_{0}=1}, A_{3}^{a_{0}=1}}
$$

Note that $A_{1}^{a_{0}=1}$, $A_{2}^{a_{0}=1}$, and $A_{3}^{a_{0}=1}$ are random (i.e., differ between individuals) counterfactuals taking values in $\mathbb{A}_{1}$, $\mathbb{A}_{2}$, $\mathbb{A}_{3}$ when no interventions are specified except for assignment to heart transplant $a_{0} = 1$. These are written in terms of the one-step-ahead counterfactuals as:

$$
A_{1}^{a_{0}=1}, \quad A_{2}^{a_{0}=1} = A_{2}^{a_{0}=1, A_{1}^{a_{0}=1}}, \quad A_{3}^{a_{0}=1} = A_{3}^{a_{0}=1, A_{2}^{a_{0}=1, A_{1}^{a_{0}=1}}}
$$

Recursive substitution reveals why, in general, transporting the distribution of the outcome $Y^{a=1} = Y^{a_{0}=1, A_{1}^{a_{0}=1}, A_{2}^{a_{0}=1}, A_{3}^{a_{0}=1}}$ of the second trial to a different population can be more difficult than transporting the distribution of the outcome $Y^{a_{0}=1, a_{1}^{*}, a_{2}^{*}, a_{3}^{*}}$ of the first trial: the distribution of $Y^{a_{0}=1, a_{1}, a_{2}, a_{3}}$ will generally differ between two populations if the distribution of one or more of $A_{1}^{a_{0}=1}$, $A_{2}^{a_{0}=1}$, or $A_{3}^{a_{0}=1}$ differs between the populations.

Assuming that the one-step-ahead counterfactuals are well defined, recursive substitution above applies equally to observational and randomized studies because the definition of one-step-ahead counterfactuals only concerns logical relations between counterfactuals and factuals, irrespective of the type of study.

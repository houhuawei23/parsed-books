# Chapter 6 Adaptive Experiments

In the previous chapter, we considered policy learning under a two-phase model. In the first “exploration” phase, we had data from an experiment or an observational study that could be used to identify the effect of an intervention and choose a policy. Then, in the second “exploitation” phase, we could deploy the chosen policy—and reap rewards if we chose well.

This two-phase model, also called the batch learning model in the engineering literature, is attractive for its conceptual and operational simplicity. However, in many settings where units naturally arrive in a stream and there is a cost to experimentation, using a two-phase design with pre-specified exploration and exploitation phases may seem too rigid—and instead we may want to exploit any knowledge gained during the exploration phase as soon as it’s available. For example, if at some point in the exploration phase we become confident we’ve already uncovered the best policy for some subgroup of study participants, then why not just immediately use this information instead of waiting for a pre-specified end of the exploration phase? Or, in a multi-armed trial, if it becomes apparent that one of the arms is clearly inferior, why not discard it and re-focus available exploration resources on the other arms?

Example 6. Schwartz, Bradlow, and Fader [2017] describe a setting where a financial institution seeks to acquire new customers via online advertising. The advertiser needs to choose where to advertise (e.g., on which type of websites) and what type of ads to use, and is interested in using experimentation to optimize these choices. The authors show how an adaptive experimentation model enables the advertiser to seamlessly move from exploring to exploiting information about what ads work best during the same campaign, without needing to pre-commit to a rigid experimental sample size up front. One should also note that, in this setting, there’s less value in having access to standard inferential outputs from a randomized trial (e.g., in terms of confidence intervals and summary statistics), since any learnings would likely be specific to the given advertising campaign and may not generalize to other campaigns.

This chapter provides a brief introduction to the design of adaptive experiments, also known as multi-armed bandit algorithms in the engineering literature. Such experiments enable the researcher to modify their data collection scheme in response to preliminary findings, with the goal improving the quality of the collected data and/or improving the welfare of study participants. A major challenge when working with adaptive experiments is that the samples we’re using for learning are longer independent of each other because past outcomes affect future treatment assignments; and thus methods developed for non-adaptive experiments are no longer formally justified (and in fact may fail badly).

Setting and notation As is standard when analyzing multi-armed adaptive experiments, we assume that we have access to a stream of $t = 1 , \dots , T$ experimental subjects that can each be assigned one among $k = 1 , \ldots , K$ candidate actions. We write $W _ { t } \in \{ 1 , \ldots , K \}$ for the action taken at time t and $Y _ { t }$ for the observed outcome (or reward), and will consider settings where $W _ { t }$ is a (potentially randomized) function of past data. Following the potential outcomes $\{ Y _ { t } ( k ) \} _ { k = 1 } ^ { K }$ that $Y _ { t } = Y _ { t } ( W _ { t } )$ .

Throughout this chapter, we will also make the following. We have access to a stream of $t = 1 , \dots , T$ experimental subjects such that:

• The potential outcomes are independent and identically distributed across $\{ Y _ { t } ( k ) \} _ { k = 1 } ^ { K } \overset { \mathrm { i i d } } { \sim } F$ on t. We write $\mu _ { k } = \mathbb { E } _ { F } \left[ Y _ { t } ( k ) \right]$ for the mean reward of the k-th arm.
• There are no covariates $X _ { t }$ that can be used to for targeting, and assigned actions can only depend on past actions and outcomes.

Both of these assumptions can (and often are) relaxed in the literature. There exist algorithms that can handle non-stationary and even non-stochastic potential outcomes, and also algorithms that allow use of covariates for targeting (in the engineering literature this is called the contextual bandit setting); see the bibliographic notes section for references. Here, however, we only have time to briefly scratch the surface of the literature on adaptive experiments—and will do so in the context of the restricted setting described above.

## 6.1 Low-regret data collection

There are multiple objectives one can target when designing adaptive datacollections algorithms. We will start by considering methods guided by the simple principle of getting high cumulative rewards (and avoiding low-reward actions) for the $t = 1 , \dots , T$ in-sample experimental subjects. The highest possible expected reward one can get using any data collection procedure is $T \mu ^ { * }$ , where µ∗ = max $\{ \mu _ { k } : 1 \le k \le K \}$ is the mean reward of the best arm in terms of mean reward. We will assess the quality of an adaptive data-collection procedure in terms of its regret

$$
R _ {T} = \sum_ {t = 1} ^ {T} \left(\mu^ {*} - \mu_ {W _ {t}}\right), \tag {6.1}
$$

which quantifies the shortfall in rewards relative to always playing the best arm.35 In a non-adaptive trial where $W _ { t }$ is uniformly distributed on $\{ 1 , \ldots , K \}$ , $\begin{array} { r } { { \cal { R } } _ { T } \sim T \sum _ { k = 1 } ^ { K } \left( \mu ^ { * } - \mu _ { k } \right) / K } \end{array}$ tive experimentation schemes is to do better, and achieve sub-linear regret. In order to do so, any algorithm will first need to explore the sampling distribution to figure out which arms $k = 1 , \ldots , K$ are the most promising, and then exploit this knowledge to attain low regret.

The upper confidence band method One notable early solution to the explore-exploit trade-off problem in adaptive experiments in the upper confidence band (UCB) algorithm of Lai and Robbins [1985]. The algorithm proceeds as follows. First, initialize each arm using $t _ { 0 }$ draws and then,

• At each time $t = K t _ { 0 } + 1 , K t _ { 0 } + 2 , . . . ,$ construct a confidence interval $\widehat { U } _ { k , t }$ for $\mu _ { k }$ based on data collected up to time $t - 1$ , and
• Pick action $W _ { t }$ corresponding to the confidence interval $\widehat { U } _ { k , t }$ with the largest upper endpoint, and observe $Y _ { t } = Y _ { t } ( W _ { t } )$ .

At a high level, the motivation behind UCB is that we always want to explore the arm with the most upside, i.e., UCB is optimistic in the face of uncertainty about arm rewards. If we have yet to learn much about a given arm, it will have a long confidence interval and UCB will optimistically sample it more. Over time, however, we’ll collect enough data from the bad arms to be fairly sure they’re suboptimal in the sense that even the upper endpoint of their confidence intervals isn’t competitive with rewards we could get from other arms—and at that point UCB will stop sampling them.

There are many different variants of UCB considered in practice that arise from different constructions for the confidence interval $\widehat { U } _ { k , t }$ used for arm selection. To get an understanding of why UCB controls regret, we here consider a simple UCB variant tailored to a Gaussian sampling model, i.e.,

$$
Y _ {t} (k) \sim \mathcal {N} \left(\mu_ {k}, \sigma^ {2}\right), \tag {6.2}
$$

where $\sigma ^ { 2 }$ is known. The Gaussianity and known $\sigma$ and $T$ assumptions help simplify the analysis; one can get rid of them at the expense of a slightly more delicate algorithm and argument.

We write the cumulative number of times the k-th arm has been drawn and the current running average of rewards from it as

$$
n _ {k, t} = \sum_ {j = 1} ^ {t} 1 \left(\{W _ {j} = k \}\right), \quad \hat {\mu} _ {k, t} = \frac {1}{n _ {k , t}} \sum_ {j = 1} ^ {t} 1 \left(\{W _ {j} = k \}\right) Y _ {j}, \tag {6.3}
$$

and select actions as

$$
W _ {t} \in \operatorname{argmax} \left\{\widehat {U} _ {k, t} \right\}, \quad \widehat {U} _ {k, t} = \hat {\mu} _ {k, t - 1} + \sigma \sqrt {4 \log (T) / n _ {k , t - 1}}. \tag {6.4}
$$

This choice is induced by the UCB construction with confidence intervals for $\mu _ { k , t }$ whose width is $\sqrt { 4 \log ( T ) }$ times the standard error of the estimate. The following result shows that this algorithm in-fact achieves low regret with high probability. The variant of UCB considered here was proposed by Auer, Cesa-Bianchi, and Fischer [2002], who refer to this algorithm as the UCB1 algorithm.

Theorem 6.1. Under our sampling assumptions and with Gaussian36 IID potential outcomes (6.2), UCB with intervals (6.4) and $t _ { 0 } = 1$ initial draws has regret bounded as

$$
R _ {T} \leq 1 6 \sigma^ {2} \log (T) \sum_ {\{k: \mu_ {k} \neq \mu^ {*} \}} \frac {1}{\mu^ {*} - \mu_ {k}} + \sum_ {\{k: \mu_ {k} \neq \mu^ {*} \}} (\mu^ {*} - \mu_ {k}), \tag {6.5}
$$

with probability at least $1 - K / T$ .

Proof. For simplicity, we assume that there is a unique best arm with $k ^ { * }$ with $\mu _ { k ^ { * } } = \mu ^ { * } . ^ { 3 7 }$ Under our sampling model, regret $R _ { T }$ can be expressed as

$$
R _ {T} = \sum_ {k \neq k ^ {*}} n _ {k, T} \left(\mu_ {k ^ {*}} - \mu_ {k}\right). \tag {6.6}
$$

Our main task is thus to bound $n _ { k , T } , \mathrm { i . e . }$ , the number of times UCB may pull any sub-optimal arm; and it turns out that UCB is essentially an algorithm reverse-engineered to make such an argument go through.

To this end, the first thing to check is that, for each arm $k \neq k ^ { * }$ , we have

$$
\hat {\mu} _ {k, t - 1} \leq \mu_ {k} + \sigma \sqrt {4 \log (T) / n _ {k , t - 1}} \tag {6.7}
$$

for all $t = K + 1 , \ldots , T$ with probability $1 - 1 / T$ . This is true because, writing $\zeta _ { k , j }$ for the j-th time arm k was pulled, we have

$$
\begin{array}{l} \mathbb {P} \left[ \sup _ {K <   t \leq T} \left\{\mu_ {k} - \hat {\mu} _ {k, t - 1} - \sigma \sqrt {4 \log (T) / n _ {k , t - 1}} \geq 0 \right\} \right] \\ \leq \mathbb {P} \left[ \sup _ {1 \leq j \leq n _ {k, T}} \left\{\mu_ {k} - \hat {\mu} _ {k, \zeta_ {k, j}} - \sigma \sqrt {4 \log (T) / j} \geq 0 \right\} \right] \\ = \mathbb {P} \left[ \sup _ {1 \leq j \leq n _ {k, T}} \left\{\mu_ {k} - \frac {1}{j} \sum_ {l = 1} ^ {j} Y _ {l} ^ {\prime} (0) - \sigma \sqrt {4 \log (T) / j} \geq 0 \right\} \right] \\ \leq \mathbb {P} \left[ \sup _ {1 \leq j \leq T} \left\{\mu_ {k} - \frac {1}{j} \sum_ {l = 1} ^ {j} Y _ {l} ^ {\prime} (0) - \sigma \sqrt {4 \log (T) / j} \geq 0 \right\} \right] \\ \leq T \exp (- 2 \log (T)) = 1 / T, \\ \end{array}
$$

where the equality follows by stationarity of the data-generating process (here, $Y _ { l } ^ { \prime } ( k )$ are independent draws from $\mathcal { N } \left( \mu _ { k } , \sigma ^ { 2 } \right) )$ , and the last line is an application of a sub-Gaussian tail bound with a union bound. By a repeat of the same argument and another union bound we see that with probability at least $1 - K / T$ ,

$$
\mu_ {k ^ {*}} \leq \hat {\mu} _ {k ^ {*}, t - 1} + \sigma \sqrt {4 \log (T) / n _ {k ^ {*} , t - 1}} \tag {6.8}
$$

for all $t = K + 1 , \ldots , T$ , and (6.7) holds simultaneously for all $k \neq k ^ { * }$ .

When (6.7) and (6.8) hold, we can only pull any sub-optimal arm $k \neq k ^ { * }$ under the following (necessary but not sufficient) conditions:

$$
\begin{array}{l} W _ {t} = k \implies \hat {\mu} _ {k, t - 1} + \sigma \sqrt {4 \log (T) / n _ {k , t - 1}} \geq \hat {\mu} _ {k ^ {*}, t - 1} + \sigma \sqrt {4 \log (T) / n _ {k ^ {*} , t - 1}} \\ \Longrightarrow \hat {\mu} _ {k, t - 1} + \sigma \sqrt {4 \log (T) / n _ {k , t - 1}} \geq \mu_ {k ^ {*}} \\ \Longrightarrow \mu_ {k} + 2 \sigma \sqrt {4 \log (T) / n _ {k , t - 1}} \geq \mu_ {k ^ {*}} \\ \Longrightarrow n _ {k, t - 1} \leq 1 6 \sigma^ {2} \log (T) / (\mu_ {k ^ {*}} - \mu_ {k}) ^ {2}. \\ \end{array}
$$

Thus, when (6.7) and (6.8) hold, pulling the k-th arm for some $k \neq k ^ { * }$ simply becomes impossible once $n _ { k , t - 1 }$ passes a certain cutoff, and so

$$
n _ {k, T} \leq 1 6 \sigma^ {2} \log (T) / (\mu_ {k ^ {*}} - \mu_ {k}) ^ {2} + 1.
$$

Plugging this into the regret expression (6.6), we obtain (6.5).

![image_02](images/image_02.png)

Theorem 6.1 immediately implies that UCB in fact succeeds in finding and effectively retiring sub-optimal arms reasonably fast, thus resulting in regret that only scales logarithmically in the regret. Interestingly, the dominant term in (6.5) is due to “good” arms for which $\mu ^ { * } - \mu _ { k }$ is small; intuitively, the reason these arms are difficult to work with is that it takes longer to be sure that they’re sub-optimal. This implies that the cost of including some very bad arms in an adaptive experiment may be limited, since an algorithm like UCB will be able to discard them quickly.

Finally, one should note that the upper bound (6.5) appears to allow for unbounded regret due to quasi-optimal arms for which $\mu _ {  { k ^ { * } } } - \mu _ {  { k } }$ is very small. This is simply an artifact of the proof strategy that focused on the case where effects are strong. When effects may be weak, one can simply note that the worst-case regret due to any given arm $k$ is upper bounded by $T \left( \mu _ { k ^ { * } } - \mu _ { k } \right)$ ; and, combining this bound with the bound implied by (6.5), we find that the worst-case regret for any combination of arms $\mu _ { k }$ is bounded on the order of $K { \sqrt { T \log ( T ) } }$ .

Thompson sampling UCB is a simple approach to adaptive experimentation with strong bounds on excess regret from sampling sub-optimal arms. However, the algorithm is sensitive to a number of seemingly ad-hoc choices that are more tied to proof strategies than transparent methodological considerations, and this can lead to suboptimal performance in practice. For example, the version of the UCB algorithm given above uses relatively wide confidence intervals with a half-length of $\sqrt { 4 \log ( T ) }$ standard errors; and so qualitatively, if we understand UCB as always choosing the arm with the most upside, then this version of UCB is extremely optimistic in assessing upside. What would happen if we ran UCB with intervals with a half-length of 1.96 standard errors instead, i.e., with a more conventional amount of optimism regarding the upside from each arm? In practice, this might (and often does) work well (perhaps even better), but the proof of Theorem 6.1 would no longer go through (because the events (6.7) and (6.8) hold would no longer uniformly hold across all time with high probability).

Current empirical practice suggests that we can side-step this brittleness of UCB by using algorithms that are still driven by the general principle of optimism in the face of uncertainty, but that operationalize their optimism in terms of Bayesian rather than frequentist reasoning. Thompson sampling [Thompson, 1933] is one example of a simple and widely used algorithm that does so. To implement this algorithm, we start by picking a prior $\Pi _ { 0 }$ for the potential outcome distribution $F .$ . Then, for each time $t = 1 , \dots , T$ , we

• Compute probabilities $e _ { k , t - 1 }$ that each arm k is the best arm, i.e.,

$$
e _ {k, t - 1} = \mathbb {P} _ {\Pi_ {t - 1}} \left[ \mu_ {k} = \mu_ {*} \right], \tag {6.9}
$$

• Randomly choose an action $W _ { t } \sim \mathrm { M u l t i n o m i a l } ( e _ { \cdot , t - 1 } )$ , and
• Observe $Y _ { t } = Y _ { t } ( W _ { t } )$ and update the posterior $\Pi _ { t }$

One can efficiently implement this algorithm via posterior sampling: First draw a joint sample $( \mu _ { 1 } ^ { \prime } , . . . , \mu _ { K } ^ { \prime } ) \sim \Pi _ { t - 1 }$ , and then set $W _ { t } = \mathrm { a r g m a x } \left. \mu _ { k } ^ { \prime } \right.$ .

Although Thompson sampling looks superficially very different from UCB, it ends up having a similar statistical intuition behind it. Just like UCB, Thompson sampling regularly explores every arm until it becomes effectively sure that the arm is not good (i.e., the posterior probability of the arm being best drops below $1 / T )$ ; and intuition from, say, the Bernstein–von Mises theorem suggests that this should happen with roughly the same amount of information as when the upper confidence band of an arm falls below the whole confidence interval of some better arm. Proving an analogue to Theorem 6.1 is however beyond the scope of this presentation, and we instead refer to Agrawal and Goyal [2017] for such a result.

From a practical perspective, Thompson sampling presents a number of advantages relative to UCB. Thompson sampling is less sensitive to implementation choices than UCB; in fact, if one is willing to initialize the algorithm by taking 1 draw from each arm, then one can run Thompson sampling with $\Pi _ { 0 }$ set to be an improper flat prior over the real line, resulting in an algorithm with no tuning parameters.38 And, in empirical evaluations, Thompson sampling often proves itself more resilient than UCB and related algorithms [Chapelle and Li, 2011, Wu and Wager, 2022].

## 6.2 Inference after adaptive data collection

After collecting data in an adaptive trial, it may also be of interest to perform statistical inference and, e.g., give confidence intervals for the mean arm reward parameters $\mu _ { k }$ . Doing so, however, requires caution as adaptive data collection yields non-IID data and can thus void guarantees for standard approaches to inference. For example, in the case of estimating $\mu _ { k }$ , two natural estimators that immediately come to mind include the sample mean

$$
\hat {\mu} _ {k} ^ {A V G} = \hat {\mu} _ {k, T} = \frac {1}{n _ {k , T} ^ {- 1}} \sum_ {j = 1} ^ {t} 1 \left(\{W _ {j} = k \}\right) Y _ {j} \tag {6.10}
$$

and, in the case of Thompson sampling, the inverse-propensity weighted estimator

$$
\hat {\mu} _ {k} ^ {I P W} = \frac {1}{T} \sum_ {t = 1} ^ {T} \frac {1 \left(\{W _ {t} = k \}\right) Y _ {t}}{e _ {t , k}}. \tag {6.11}
$$

However, due to the adaptive data-collection scheme, neither of these estimators has an asymptotically normal limiting distribution, thus hindering their use for making confidence intervals.

The following simple illustrates the failure of the classical central limit theorem when working with adaptively collected data:

• We can sample outcome $Y _ { t } \sim \mathcal { N } ( \mu , 1 )$ for a single arm with unknown mean $\mu .$ .
• We first run a pilot study on $n _ { 0 }$ samples and say that the pilot study passed if the sample average of the first $n _ { 0 }$ samples is positive (and that it failed else).
• If the pilot study passed, we collect a further $1 0 n _ { 0 }$ samples, whereas if it failed we only collect $n _ { 0 }$ further samples.

This example is intended to capture, using a simple one-arm design, the qualitative behavior of Thompson sampling whereby the higher the current sample average of an arm the more likely we are to draw from it. Figure 6.1 displays the scaled distribution of the resulting sample average when $\mu = 0$ . We readily see that the scaled distribution of $\hat { \mu } ^ { A \bar { V } G }$ is both non-Gaussian and biased downwards, and so normal confidence intervals centered at $\hat { \mu } ^ { A V G }$ would not be valid here. Nie et al. [2018] provide a general result showing that sample averages for regret-minimizing algorithms are biased downwards in considerable generality.

Meanwhile, $\hat { \mu } ^ { I P W }$ is unbiased when available (e.g., with Thompson sampling). However, as discussed in Hadad et al. [2021], it still has a non-Gaussian— and often heavy-tailed—sampling distribution. Thus, it again cannot be used for normal inference.

The topic how best to do inference with adaptively collected collected data is still an active research topic, and a comprehensive review of the literature is beyond the scope of this presentation. However, as a pointer to available solutions, we here show how careful re-weighting of the data can avoid the non-Gaussianity issues with ˆµAV G and ˆµIPW . $\hat { \mu } ^ { A V G }$ $\hat { \mu } ^ { I P W }$Consider a sequentially randomized experiment, where the treatment probabilities $e _ { t }$ can depend on past data; Thompson sampling is an example of a sequentially randomized experiment. Then, we define the adaptively weighted estimate of $\mu _ { k }$ as

$$
\hat {\mu} _ {k} ^ {A W} = \sum_ {t = 1} ^ {T} \frac {1 (\{W _ {t} = k \}) Y _ {t}}{\sqrt {e _ {t , k}}} / \sum_ {t = 1} ^ {T} \frac {1 (\{W _ {t} = k \})}{\sqrt {e _ {t , k}}}. \tag {6.12}
$$

The specification of this estimator may appear surprising, as units are weighted by $1 / \sqrt { e _ { t , k } }$ rather than the more familiar $1 / e _ { t , k }$ inverse-propensity weights. However, as shown below, this weighting scheme yields an asymptotic normality result. We note that the regularity condition (6.14) reduces to the familiar Lindeberg condition in the case of randomized trials with constant treatment propensities; this condition is weak provided the $e _ { t , k }$ cannot decay too fast.

Theorem 6.2. In a sequentially randomized experiment with IID potential outcomes, suppose that

$$
0 <   \sigma_ {k} ^ {2} := \operatorname{Var} \left[ Y _ {t} (k) \right] <   \infty \tag {6.13}
$$

for all arms $k = 1 , \ldots , K$ , that $e _ { t , k } > 0$ almost surely39 and that, for all $\varepsilon > 0$

$$
\lim _ {T \to \infty} \frac {1}{T} \sum_ {t = 1} ^ {T} \mathbb {E} \left[ (Y _ {t} - \mu_ {k}) ^ {2}   {\bf 1} \left(\left\{(Y _ {t} - \mu_ {k}) ^ {2} \geq \varepsilon   e _ {t, k}   T \right\}\right) \big |   {\cal F} _ {t - 1} \right] = 0, \tag {6.14}
$$

where $\mathcal { F } _ { t - 1 }$ denotes information collected up to time $t - 1$ . Then,

$$
\widehat {V} _ {k} ^ {- 1 / 2} \left(\widehat {\mu} _ {k} ^ {A W} - \mu_ {k}\right) \Rightarrow \mathcal {N} (0, 1),
$$

$$
\widehat {V} _ {k} = \sum_ {t = 1} ^ {T} \left(\frac {1 \left(\left\{W _ {t} = k \right\}\right) \left(Y _ {t} - \hat {\mu} _ {k} ^ {A W}\right)}{\sqrt {e _ {t , k}}}\right) ^ {2} / \left(\sum_ {t = 1} ^ {T} \frac {1 \left(\left\{W _ {t} = k \right\}\right)}{\sqrt {e _ {t , k}}}\right) ^ {2}. \tag {6.15}
$$

Proof. We start by stating a technical result, the proof of which is deferred to the end of this section: Under (6.13) and (6.14),

$$
\sum_ {t = 1} ^ {T} \frac {1 \left(\{W _ {t} = k \}\right)}{\sqrt {e _ {t , k}}} / \sqrt {T} \rightarrow_ {p} \infty , \tag {6.16}
$$

i.e., the denominator in (6.12) grows faster than $\sqrt { T } .$ . Qualitatively, (6.16) means that our adaptive sampling scheme collects an increasing amount of data over time under the adaptive weighting scheme used in (6.12).

Now, to obtain a central limit theorem, we note that

$$
\hat {\mu} _ {k} ^ {A W} - \mu_ {k} = \sum_ {t = 1} ^ {T} \frac {1 (\{W _ {t} = k \}) (Y _ {t} - \mu_ {k})}{\sqrt {e _ {t , k}}} / \sum_ {t = 1} ^ {T} \frac {1 (\{W _ {t} = k \})}{\sqrt {e _ {t , k}}}, \tag {6.17}
$$

and start by focusing on the numerator of the above expression. Let

$$
M _ {t} = \sum_ {j = 1} ^ {t} \frac {1 \left(\left\{W _ {j} = k \right\}\right) \left(Y _ {j} - \mu_ {k}\right)}{\sqrt {e _ {j , k}}} \tag {6.18}
$$

be its partial sum. Because $W _ { t }$ is randomly chosen given information up to time $t ,$ we see that $W _ { t }$ is independent of $Y _ { t } ( k )$ conditionally on information collected up to time $t - 1$ , and thus $M _ { t }$ is a martingale:

$$
\mathbb {E} \left[ M _ {t} \mid \mathcal {F} _ {t - 1} \right] = M _ {t - 1}. \tag {6.19}
$$

Furthermore, thanks to our weighting scheme, we can check that the conditional variance of each martingale step is non-random despite our use of adaptive sampling probabilities:

$$
\operatorname{Var} \left[ M _ {t} \mid \mathcal {F} _ {t - 1} \right] = \sigma_ {k} ^ {2}. \tag {6.20}
$$

Given these two facts, the martingale central limit theorem [Helland, 1982, Theorem 2.5(a)] implies that

$$
M _ {T} / \sqrt {T \sigma_ {k} ^ {2}} \Rightarrow \mathcal {N} (0, 1) \tag {6.21}
$$

whenever

$$
\lim _ {T \to \infty} \frac {1}{T} \sum_ {t = 1} ^ {T} \mathbb {E} \left[ (M _ {t} - M _ {t - 1}) ^ {2} 1 \left(\left\{(M _ {t} - M _ {t - 1}) ^ {2} > \varepsilon T \right\}\right) \big | \mathcal {F} _ {t - 1} \right] = 0 \tag {6.22}
$$

for all $\varepsilon > 0$ . In our setting

$$
\begin{array}{l} \mathbb {E} \left[ \frac {1 \left(\left\{W _ {j} = k \right\}\right) \left(Y _ {j} - \mu_ {k}\right) ^ {2}}{e _ {j , k}} 1 \left(\left\{\frac {1 \left(\left\{W _ {j} = k \right\}\right) \left(Y _ {j} - \mu_ {k}\right) ^ {2}}{e _ {j , k}} > \varepsilon T \right\}\right) \mid \mathcal {F} _ {t - 1} \right] \\ = \mathbb {E} \left[ \frac {1 \left(\{W _ {j} = k \}\right) (Y _ {j} - \mu_ {k}) ^ {2}}{e _ {j , k}} 1 \left(\left\{\frac {(Y _ {j} - \mu_ {k}) ^ {2}}{e _ {j , k}} > \varepsilon T \right\}\right) \mid \mathcal {F} _ {t - 1} \right] \\ = \mathbb {E} \left[ (Y _ {j} - \mu_ {k}) ^ {2} 1 \left(\left\{(Y _ {j} - \mu_ {k}) ^ {2} > \varepsilon e _ {j, k} T \right\}\right) \mid \mathcal {F} _ {t - 1} \right], \\ \end{array}
$$

meaning that (6.14) is equivalent to (6.22) and thus (6.21) holds.

$\hat { \mu } _ { k } ^ { A W }$ $\mu _ { k }$ thanks to (6.16) and (6.21). Meanwhile, under (6.14), we also have that

$$
\sum_ {t = 1} ^ {T} \left(\frac {1 \left(\{W _ {t} = k \}\right) (Y _ {t} - \mu_ {k})}{\sqrt {e _ {t , k}}}\right) ^ {2} / \left(T \sigma_ {k} ^ {2}\right)\rightarrow_ {p} 1 \tag {6.23}
$$

by martingale concentration [Helland, 1982, Lemma 2.3]; the same holds with $\mu _ { k }$ replaced with $\hat { \mu } _ { k } ^ { A W }$ by consistency. Thus, by (6.21) and Slutsky’s lemma,

$$
M _ {T} \Big / \sqrt {\sum_ {t = 1} ^ {T} \left(\frac {1 \left(\left\{W _ {t} = k \right\}\right) \left(Y _ {t} - \hat {\mu} _ {k} ^ {A W}\right)}{\sqrt {e _ {t , k}}}\right) ^ {2}} \Rightarrow \mathcal {N} (0, 1). \tag {6.24}
$$

$\hat { \mu } _ { k } ^ { A W }$ and V 1/2 $\widehat { V } _ { k } ^ { 1 / 2 }$ cancel out.

The proof of Theorem 6.2 reveals why the adaptively weighted estimator $\hat { \mu } _ { k } ^ { A W }$ $\hat { \mu } _ { k } ^ { A V G }$ $\hat { \mu } _ { k } ^ { I P W }$ may not. The weighting scheme for the adaptively weighted estimator was essentially reverse-engineered for the predictable variance condition (6.20) to go through and thus enable application of a martingale central limit theorem. $\hat { \mu } _ { k } ^ { A V G }$ $\hat { \mu } _ { k } ^ { I P W }$ do not in general have this property in adaptive experiments. Hadad et al. [2021] refer to weights that allow for application of a martingale central limit theorem as “variance stabilizing”, and study a family of variance stabilized estimators that include $\hat { \mu } _ { k } ^ { A W }$ as a special case.

Proof of (6.16). It now remains to establish the remaining technical claim in the proof of Theorem 6.2. Our first task is to check that

$$
E _ {T, k} / \sqrt {T} \rightarrow_ {p} \infty , \quad E _ {T, k} = \sum_ {t = 1} ^ {T} \sqrt {e _ {t , k}}. \tag {6.25}
$$

Under (6.13), we can choose an $\alpha _ { k } > 0$ be such that

$$
\mathbb {E} \left[ (Y _ {t} - \mu_ {k}) ^ {2} {\bf 1} \left(\left\{(Y _ {t} - \mu_ {k}) ^ {2} \geq \alpha_ {k} \right\}\right) \right] \geq \frac {\sigma_ {k} ^ {2}}{2}.
$$

Then, by repeatedly applying Markov’s inequality conditionally on past data, we see that the key sum in (6.14) can be bounded from below as

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 1} ^ {T} \mathbb {E} \left[ (Y _ {t} - \mu_ {k}) ^ {2} {\bf 1} \left(\{(Y _ {t} - \mu_ {k}) ^ {2} \geq \varepsilon e _ {t, k} T \}\right) \big | \mathcal {F} _ {t - 1} \right] \\ \geq \frac {\sigma_ {k} ^ {2}}{2} \frac {1}{T} \sum_ {t = 1} ^ {T} 1 \left(\{\varepsilon e _ {t, k} T \leq \alpha_ {k} \}\right) \geq \frac {\sigma_ {k} ^ {2}}{2} \frac {1}{T} \sum_ {t = 1} ^ {T} 1 \left(\left\{\sqrt {e _ {t , k}} \leq \sqrt {\alpha_ {k} / (\varepsilon T)} \right\}\right). \\ \end{array}
$$

By (6.14), this expression must converge to 0 in probability for every $\varepsilon > 0$ Thus, for any $\varepsilon > 0$ , we have $\sqrt { e _ { t , k } } \ge \sqrt { \alpha _ { k } / ( \varepsilon T ) }$ for all but a vanishing fraction of units with high probability, and so $\left( 6 . 2 5 \right)$ must hold.

For our next step, we form another $\mathcal { F } _ { t } .$ -martingale $X _ { t }$ with differences

$$
X _ {t} - X _ {t - 1} = \sqrt {e _ {t , k}} - \frac {1 (\{W _ {t} = k \})}{\sqrt {e _ {t , k}}}.
$$

This martingale has increments bounded from above, $X _ { t } ~ - ~ X _ { t - t } ~ \le ~ 1$ , and variance increments Var $\left[ X _ { t } \middle | \mathcal { F } _ { t - 1 } \right] = 1 - e _ { t , k } \leq 1$ . Freedman [1975, Theorem 4.1] then shows that, for any $a > 0$ ,

$$
\mathbb {P} \left[ X _ {T} \geq a \right] \leq \exp \left[ - \frac {a ^ {2}}{2 (a + T)} \right]. \tag {6.26}
$$

Now, given (6.25), we know that there exists a function √ √ $r ( T )$ such that $r ( T ) $ ∞ and $\mathbb { P } [ E _ { T , k } / ( 2 r ( T ) \sqrt { T } ) ] \to 1$ . Plugging $a = r ( T ) \sqrt { T }$ into the above expression, we then get

$$
\lim _ {T \to \infty} \mathbb {P} \left[ \sum_ {t = 1} ^ {T} \frac {1 (\{W _ {t} = k \})}{\sqrt {e _ {t , k}}} \leq \sum_ {t = 1} ^ {T} \sqrt {e _ {t , k}} - r (T) \sqrt {T} \right] = 0,
$$

which, because $E _ { T , k } \ge 2 r ( T ) \sqrt { T }$ with high probability, implies (6.16).

Trade-offs in adaptive study design In this chapter, we have considered two high-level questions pertaining to adaptive experiments. First, we asked how to collect data such as to minimize in-sample regret; and then we asked how to build confidence intervals for mean arm rewards using adaptively collected data. Given this this background, it’s natural to ask whether it’s possible to align these two tasks—and simultaneously achieve low in-sample regret and powerful post-experiment inference.

Here, however, the answer is unfortunately an unequivocal no: Data collection schemes that aggressively optimize for in-sample regret as in (6.1) will result in fragile post-experiment inference. Bubeck, Munos, and Stoltz [2009] provide a formal trade-off in terms of the in-sample regret achieved using a data-collection scheme, and the post-experiment regret one could get by deploying the best arm from the experiment on future data. Fan and Glynn [2021] show that any adaptive algorithm that achieves optimal in-sample expected regret will necessarily have a heavy-tailed regret distribution (i.e., the algorithm has a small but non-negligible probability of failing completely and incurring large regret). Finally, on a technical note, algorithms that aggressively taper propensities $e _ { t , k }$ for poorly performing arms are likely to not satisfy the Lindeberg condition (6.14), and thus may not allow for valid post-experiment inference via the proposed method.

There are thus unavoidable trade-offs in the design of adaptive experiments, and researchers should choose relevant data-collection strategies based on their goals. If the goal is to quickly roll out a policy and to immediately minimize in-sample regret for study participants, then algorithms like Thompson sampling provide a natural choice. If, however, a researcher also wants to use the collected data to guide future policy, then using algorithms that are less aggressive in how fast they taper the use of suboptimal arms is preferable [Bubeck et al., 2009, Fan and Glynn, 2021]. We also note a large literature on designing adaptive experiments such as to maximize our chance of identifying either the best arm [Russo, 2020] or a quasi-optimal arm [Kasy and Sautmann, 2021] after T time-steps.

## 6.3 Bibliographic notes

This line of work on bandit algorithms builds on early results from Lai and Robbins [1985] on the UCB algorithm. Lai and Robbins [1985] showed that a variant of UCB achieves regret scaling of the form (6.5), and that this behavior is asymptotically optimal. Finite-sample bounds of the type given in Theorem 6.1 are established in Auer, Cesa-Bianchi, and Fischer [2002], while Agrawal and Goyal [2017] provide analogous bounds for Thompson sampling. Thanks to its Bayesian specification, Thompson sampling can be generalized to a wide variety of adaptive learning problems; see Russo et al. [2018] for a recent survey. We also note that UCB and Thompson sampling are by far not the only available algorithms for this task; for example, Russo and Van Roy [2018] propose information-directed sampling, another Bayesian heuristic which they argue presents an attractive alternative to Thompson sampling.

In Section 6.1, we considered adaptive experiments that can quickly converge on sampling the best of K available actions. The econometric setting we used made 3 major assumptions that may not hold in applications: We did not consider covariates $X _ { t }$ that can be used to guide decision making; we only considered in-sample regret as an objective; and we assumed that the sampling distribution is stable over time. Each of these assumptions has been relaxed in the literature. The literature on contextual bandits allows linking potential outcomes with covariates $X _ { t }$ via either a parametric [Bastani and Bayati, 2020, Goldenshluger and Zeevi, 2013] or non-parametric [Gur, Momeni, and Wager, 2022, Hu, Kallus, and Mao, 2022a, Perchet and Rigollet, 2013] specification. The literature on best-arm selection was already discussed above [Bubeck et al., 2009, Kasy and Sautmann, 2021, Russo, 2020]. Finally, Besbes, Gur, and Zeevi [2019], Liu, Van Roy, and Xu [2023] and Qin and Russo [2022] consider different models for how the reward distribution may change over time, and propose algorithms tailored to this setting. There is also a large literature on the adversarial model where, by analogy to the Neyman model, no sampling assumptions are made on the potential outcomes and the only source of randomness is in randomized action choice; see Bubeck and Cesa-Bianchi [2012] for a review and references.

The line of work on inference with adaptively collected data via variancestabilizing weighting is pursued by a number of authors including Luedtke and van der Laan [2016], Hadad et al. [2021] and Zhang, Janson, and Murphy [2020]. One should note that this is not the only possible approach to inference in adaptive experiments. In particular, a classical alternative to inference in this setting starts from confidence-bands based on the law of the iterated logarithm and its generalizations that hold simultaneously for every value of $t ;$ see Robbins [1970] for a landmark survey and Howard et al. [2021] for recent advances. One can also build confidence intervals using diffusion approximations for adaptive experiments motivated by weak-signal asymptotics [Hirano and Porter, 2023, Kuang and Wager, 2024].

Finally, all approaches to adaptive experimentation discussed today are essentially heuristic algorithms that can be shown to have good asymptotic behavior (i.e., neither UCB nor Thompson sampling can be derived directly from an optimality principle). In the Bayesian case (i.e., where we have an actual subjective prior for F rather than just a convenience prior as used by Thompson sampling to power an algorithm with frequentist guarantees), it is possible to solve for the optimal regret-minimizing experimental design via dynamic programming [Gittins, 1979].
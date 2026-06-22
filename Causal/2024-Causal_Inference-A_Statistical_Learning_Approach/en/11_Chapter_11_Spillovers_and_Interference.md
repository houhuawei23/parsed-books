# Chapter 11 Spillovers and Interference

Throughout our discussions so far, we have relied on the SUTVA assumption whereby the treatment given to one person only affects the targeted person and not others. This assumption is reasonable in a number of setting, including when, in medicine, we want to assess the benefits of a cancer treatment or when, in marketing, we want to assess the effectiveness of a customer-retention program. In other settings, however, this assumption is obviously fraught, and cross-unit treatment spillovers are a first-order concern.

Example 11. Cai, Janvry, and Sadoulet [2015] ran a randomized experiment in rural China to understand whether take-up of government-subsidized weather insurance could be promoted via information sessions that give a detailed presentation on how the insurance product works. The authors were interested in both direct effects of the intervention on people who attend the information sessions, and in spillovers onto the friends of those who attended. Asking about spillovers reflects an underlying belief that information given to some people may affect insurance take-up by others (namely their friends).

Example 12. Blattman et al. [2021] report results on a randomized evaluation of crime-reduction measures in Bogot´a, Colombia. The city identified 1,919 streets as crime hot spots, and randomized them to receive either increased police patrolling, increased municipal services, both interventions or neither; the authors were interested in measuring any effect of these measures on both violent crime or property crime. A concern in the analysis was that, instead of suppressing crime, some localized interventions may only displace it to neighboring streets; and the authors develop techniques for evaluating such spillovers.

Example 13. Ride-sharing platforms seek to connect potential riders with freelance drivers. Many existing platforms propose prices up front, i.e., they first advertise trips to riders at a given price and then seek to connect with a driver once a trip request is made. It is natural to run experiments to fine-tune these prices for healthy market behavior, but properly accounting for spillovers is crucial in doing so. For example, if one were to randomize access to driver incentives, it is expected that drivers with access to such incentives would earn more per hour than those who don’t. However, as reported by Hall, Horton, and Knoepfle [2023], giving such incentives to everyone may not increase hourly earnings for drivers—because the incentives may draw more drivers to work for the platform, thus reducing utilization levels of existing drivers (i.e., existing drivers might earn more per hour while actively transporting a driver, but have this benefit be canceled out by an increased amount of time spent idle). In other words, spillovers arise via market re-equilibriation.

Example 14. Infectious-disease vaccines provide two types of protection against disease spread: Vaccinated people may be less likely to get infected than unvaccinated people given comparable circumstances, and vaccinating a large enough fraction of the population may create a herd-immunity phenomenon that unvaccinated people also benefit from. The emergence of herd immunity is a type of spillover that is relevant to assessing public-health benefits of vaccination; Ogburn and VanderWeele [2017] discuss a modeling framework for estimating these effects.

The spillover mechanisms in all examples above are different. The end result, however, is the same: SUTVA fails, and new ideas are needed to assess the effects of an intervention. This chapter will introduce methods for modeling and testing for the presence of spillovers and, more broadly, cross-unit interference (i.e., treatment given to one person affects others); in the next chapter, we will then turn to questions of estimation and building confidence intervals. For simplicity, we will focus on randomized controlled trial (RCT) settings in this chapter and the next.

## 11.1 Exposure mappings

As in Chapter 1, we assume that we have data on $i = 1 , \dots , n$ people, each of whom receives a randomized binary treatment $W _ { i } \in \{ 0 , 1 \}$ and then experiences an outcome $Y _ { i } \in \mathbb { R }$ . Under interference, however, it no longer makes sense to only define two potential outcomes per unit; rather, each unit can now have up to $2 ^ { n }$ potential outcomes $\{ Y _ { i } ( \mathbf { w } ) : \mathbf { w } \in \{ 0 , 1 \} ^ { n } \}$ , corresponding to each possible treatment assignment for the whole study. The associated potential-outcome consistency assumption $\mathrm { i s ^ { 6 1 } }$

$$
Y _ {i} = Y _ {i} (\mathbf {W}), \quad \mathbf {W} = (W _ {i}) _ {i = 1} ^ {n}. \tag {11.1}
$$

While this notation is similar to that used in Chapter 1, the problem is now substantively much harder and we have an apparent curse of dimensionality to deal with, whereby the number of potential outcomes grows exponentially faster than the study size.

Any approach to causal inference under interference needs to put some structure on the potential outcomes in order to enable accurate treatment effect estimation. Here, we will do so by assuming an exposure mapping: Each unit has an exposure function $H _ { i } : \{ 0 , 1 \} ^ { n } \to \{ \mathcal { H } _ { i }$ with the property that $Y _ { i }$ only depends on the full potential outcome vector W through $H _ { i } ( \mathbf { W } )$ .

Assumption 11.1. An exposure mapping is a set of unit-specific functions $H _ { i } : \{ 0 , 1 \} ^ { n } \to \{ \mathcal { H } _ { i }$ . The assumption that this this exposure mapping is well specified is a claim that, for all pairs w, $\mathbf { w } ^ { \prime } \in \{ 0 , 1 \} ^ { n }$ , we have

$$
Y _ {i} (\mathbf {w}) = Y _ {i} \left(\mathbf {w} ^ {\prime}\right) \text { whenever } H _ {i} (\mathbf {w}) = H _ {i} \left(\mathbf {w} ^ {\prime}\right). \tag {11.2}
$$

When there is no risk of confusion, we use overloaded notation such as $Y _ { i } =$ $Y _ { i } ( H _ { i } ( \mathbf { W } ) )$ or $Y _ { i } = Y _ { i } ( H _ { i } )$ .

The simplest type of exposure mapping to work with statistically is the cluster-interference model. Under cluster interference, experimental units are divided into non-overlapping clusters, such that spillovers can be essentially arbitrary within cluster but there are no spillovers across clusters. Formally, in the context of Assumption 11.1, cluster interference posits $H _ { i } ( \mathbf { w } ) = ( w ) _ { j \in C _ { i } }$ , where $C _ { i }$ is the set of units in the same cluster as the i-th unit. The reason cluster interference is easy to work with statistically is that we can simply redefine these clusters as our experimental units of interest. Then, the fact that there is no cross-cluster interference means that SUTVA holds at the level of cluster; we can thus run a cluster-randomized experiment that we then analyze using standard techniques.

Example 15. Cr´epon et al. [2013] study community-level effects of job-search assistance programs. Such job-search programs help program participants find jobs; but the authors are concerned that they may be doing so at the expense of non-participants. To measure community effects, they identify 235 independent labor markets (e.g., cities), and randomize each market to receive different saturation levels (0%, 25%, 50%, 75%, or 100%) of job-search assistance for active job seekers. The authors then run an analysis where they compare community-level outcomes across markets with different saturation levels, i.e., they analyze the data as an RCT where each labor market is a unit and the treatment given to the unit is the saturation level of job-search assistance. The exposure mapping underlying this analysis is the cluster-interference model, with job seekers clustered by labor market.

Other applications call for more complex exposure mappings. For example, in the setting of Example 11, the authors posit that a given farmers’ insurance decisions may be affected by information received by their friends as well as by them directly. This suggests using the framers’ social network to define an exposure mapping, e.g., via the network-interference model below (with friends acting as network neighbors).

Definition 11.1. Under the network-interference model, we assume that each unit $i = 1 , \ldots , n$ has a set of network neighbors $\mathcal { N } _ { i } \subset \{ 1 , \ldots , n \}$ , with a convention that $i \not \in \mathcal { N } _ { i }$ , such that the following exposure mapping holds:

$$
Y _ {i} = Y _ {i} (H _ {i} (\mathbf {W})), \quad H _ {i} (\mathbf {w}) = (w _ {j}) _ {j \in \{i \} \cup \mathcal {N} _ {i}}. \tag {11.3}
$$

In other words, the network-interference model is a generalization of the cluster-interference model that allows for non-transitivity of spillovers, and the network interference model reduces to the cluster interference model if we impose transitivity $\{ i \} \cup \mathcal { N } _ { i } = \{ j \} \cup \mathcal { N } _ { j }$ for all $j \in \mathcal N _ { i }$ . Under network interference, we can in general no longer eliminate all spillovers via clustering (because the underlying network may be fully connected); and more careful inferential techniques are thus needed. We will return to the question of estimating treatment effects under network interference in Chapter 12. Before doing so, however, we will first discuss how to test for the presence of interference below.

## 11.2 Permutation tests

In Example 11, Cai, Janvry, and Sadoulet [2015] were interested in measuring spillovers from information sharing in a social network. Suppose that for each unit i we know the friends ${ \mathcal { N } } _ { i }$ who could plausibly affect their insurance choices. What might the most parsimonious model for spillovers look like? The network interference model from Definition 11.1 provides one possible answer, but is there evidence that the full generality of this model is needed?

In this setting, one could easily imagine a hierarchy of alternative exposure mappings as follows::

• $H _ { 0 } { \mathrm { : } }$ No causal effects. $H _ { i } ( \mathbf { w } ) = \varnothing$ , and $Y _ { i } = Y _ { i } ( \emptyset )$ regardless of treatment.
• $H _ { 1 } { \mathrm { : } }$ No spillovers. $H _ { i } ( \mathbf { w } ) = w _ { i }$ , and $Y _ { i } = Y _ { i } ( W _ { i } )$ like in Chapter 1.
• $H _ { 2 } ;$ Anonymous network interference. $H _ { i } ( \mathbf { w } ) ~ = ~ ( w _ { i } , z _ { i } )$ , where $z _ { i } =$ $\textstyle \sum _ { j \in { \mathcal { N } } _ { i } } w _ { i } / \left| \{ { \mathcal { N } } _ { i } \} \right|$ | is the fraction of treated friends and $Y _ { i } = Y _ { i } ( W _ { i } , Z _ { i } )$ .
• $H _ { 3 } { \mathrm { : } }$ Network interference. $H _ { i } ( \mathbf { w } ) = ( w _ { j } ) _ { j \in \{ i \} \cup N _ { i } }$ , and $Y _ { i } = Y _ { i } ( H _ { i } )$ .

• $H _ { 4 } { \mathrm { : } }$ Generic spillovers. $H _ { i } ( \mathbf { w } ) = \mathbf { w }$ , and $Y _ { i } = Y _ { i } ( \mathbf { W } )$ .

The questions about the structure of treatment effects asked in the previous paragraph can then be formalized via null-hypothesis testing. For example, one might first want to test the null ${ } ^ { 6 6 } H _ { 0 }$ : no causal effects” and then, if that test rejects, test ${ } ^ { 6 6 } H _ { 1 }$ : no spillovers”, etc., until one finds an exposure mapping that is not rejected given the data at hand.

Our task is to develop methods for testing each of these nulls. Here, we will do so via permutation testing. We will propose specific tests for $H _ { 0 }$ and $H _ { 1 }$ , and give a general result that can also be used to design tests more the subsequent hypotheses.

The main idea of a permutation test is pick a test statistic, and then scramble the treatment assignment in a way that shouldn’t affect the test statistic under the posited null hypothesis. By construction, we should expect that—if the null holds—then the test statistic evaluated on the original data should fit comfortably within the range on test statistics obtained after scrambling; and if the original test statistic is in fact an outlier we take this as evidence against the null.

Remark 11.1. In our discussion below, we will develop tests for individual hypotheses. It might seem that the program outlined above, i.e., where we sequentially test hypotheses until one fails to reject, would require a multiple testing correction. However, there is in fact no issue with multiple testing here because all null hypotheses are nested, and sequentially running tests on the most-to-least restrictive nulls until one of them fails to reject (and then stopping) is simultaneously be valid against all nulls thanks to the closed testing principle [Marcus, Peritz, and Gabriel, 1976].

Testing the sharp null We first consider the design of a permutation test against the no-causal-effect null $H _ { 0 }$ . This is a “sharp” null in that it fully specifies how treatment affects outcomes (i.e., in no way whatsoever), and so it can be approached using the classical approach of Fisher [1935]: We first choose a test statistic that is likely to take on a large value when the null doesn’t hold, e.g.,62

$$
T \left(\mathbf {Y}, \mathbf {w}\right) = \left| \frac {\sum_ {\{i : w _ {i} = 1 \}} Y _ {i}}{| \{i : w _ {i} = 1 \} |} - \frac {\sum_ {\{i : w _ {i} = 0 \}} Y _ {i}}{| \{i : w _ {i} = 0 \} |} \right|, \tag {11.4}
$$

and then reject the null if the test statistic as computed on the realized treatment vector is unusually large relative to values it takes on alternative treatment randomizations we could have (but didn’t) get. An important fact in enabling this approach is that, under $H _ { 0 }$ , treatment has no effect on outcomes, and so

$$
T \left(\mathbf {Y}, \mathbf {w}\right) = T \left(\mathbf {Y} (\mathbf {w}), \mathbf {w}\right) \text {for all} \mathbf {w} \in \{0, 1 \} ^ {n}, \tag {11.5}
$$

meaning that—again under the null—we are able impute the actual test statistic we would have computed under different treatment randomizations.

Assumption 11.2. Treatment is assigned according to a completely randomized design: There is a set of possible treatment vectors w over $\{ 0 , 1 \} ^ { n }$ such that P $[ \mathbf { W } = \mathbf { w } ] = 1 / \left| \boldsymbol { \mathcal { W } } \right|$ for all $\mathbf { w } \in \mathcal { W }$ , independently of potential outcomes.

Theorem 11.1. Suppose that Assumption 11.2 holds. Pick any test statistic $T \left( \mathbf { Y } , \mathbf { W } \right)$ and a number of permutations $B \leq | \mathcal { W } | - 1$ , and let $\mathbf { W } _ { 1 } ^ { \prime } , \ldots , \mathbf { W } _ { B } ^ { \prime }$ be drawn uniformly at random and without replacement from ${ \mathcal { W } } \backslash \mathbf { W }$ . Then, the permutation $p { - } v a l u e ^ { 6 3 }$

$$
p = \frac {1}{1 + B} \left(1 + \sum_ {b = 1} ^ {B} 1 \left(\{T (\mathbf {Y}, \mathbf {W}) \leq T (\mathbf {Y}, \mathbf {W} _ {b} ^ {\prime}) \}\right)\right) \tag {11.6}
$$

is valid against the null, i.e., under $H _ { 0 } , \mathbb { P } \left[ p \leq \alpha \right] \leq \alpha$ for all $0 \leq \alpha \leq 1$ .

Proof. Let ${ \mathcal { W } } ^ { \prime } = \{ \mathbf { W } , \mathbf { W } _ { 1 } ^ { \prime } , \dots , \mathbf { W } _ { B } ^ { \prime } \}$ be the unordered set of considered permutations. By Assumption 11.2, under $H _ { 0 }$ ,

$$
\mathbb {P} \left[ \mathbf {W} = \mathbf {w}   |   \mathbf {W} \in \mathcal {W} ^ {\prime},   \mathbf {Y} \right] = \frac {1}{1 + B} \text { for all } \mathbf {w} \in \mathcal {W} ^ {\prime}. \tag {11.7}
$$

Thus, writing $\mathcal { T } ^ { \prime } = \{ T ( \mathbf { Y } , \mathbf { w } ) : \mathbf { w } \in \mathcal { W } ^ { \prime } \}$ for the set of considered test statistics we see that, conditionally on Y and the fact that $\mathbf { W } \in \mathcal { W } ^ { \prime }$ , the realized test statistic value $T \left( \mathbf { Y } , \mathbf { W } \right)$ is takes values uniformly at value within $\tau ^ { \prime }$ . It follows that, under Assumption 11.2 and $H _ { 0 } , p$ from (11.6) takes values uniformly at random over $\{ 1 / ( 1 + B ) , 2 / ( 1 + B ) , \ldots , 1 \}$ if there are no ties in $\tau ^ { \prime }$ , and ties can only make p strictly larger. □

Testing for interference The next question is to design a test for $H _ { 1 }$ , i.e., to test whether SUTVA holds or instead there is evidence of spillovers. To start, we again need to choose a test statistic that will have power to measure deviations from the null—and there are many ways of doing so. Following Aronow [2012], we here consider test statistics that first choose a set of focal units ${ \mathcal { F } } \subset \{ 1 , \ldots , n \}$ , and set $T = T _ { \mathcal { F } } \left( \mathbf { Y } , \mathbf { w } \right)$ to be some pre-specified functional that only considers outcomes within the focal set. For example, in settings where we believe that spillovers will only really manifest themselves on untreated units (e.g., with informational intervention as in Example 11), one natural choice for T would be use the z-coefficient in the regression

$$
T _ {\mathcal {F}} \left(\mathbf {Y}, \mathbf {w}\right) = \operatorname{OLS} \left(Y _ {i} \sim z _ {i}: i \in \mathcal {F}, w _ {i} = 0\right), \quad z _ {i} = \sum_ {j \in \mathcal {N} _ {i}} w _ {j} / | \{\mathcal {N} _ {i} \} | \tag {11.8}
$$

as our test statistic.

At this point, however, we face a challenge. When testing the sharp null, (11.5) enabled us to compute counterfactual test statistics for any treatment assignment w under $H _ { 0 }$ . Now, however, treatment can affect outcomes under $H _ { 1 }$ (via the direct effect), and so we only have access to the weaker guarantee

$$
T _ {\mathcal {F}} (\mathbf {Y}, \mathbf {w}) = T _ {\mathcal {F}} (\mathbf {Y} (\mathbf {w}), \mathbf {w}) \text {if} w _ {i} = W _ {i} \text {for all} i \in \mathcal {F}. \tag {11.9}
$$

Thus, when designing a permutation test for $H _ { 1 }$ , we can only consider those treatment assignments w which match to realized treatment W on the focal set. Doing so requires more delicate methods, which will follow from the general result given below.

Remark 11.2. With any focal unit based approach, we need the set $\mathcal { F }$ of focal units not to be either too big or too small in order for $T$ to have power. If the set of focal units $\mathcal { F }$ is too small the regression (11.8) will be noisy; whereas if the set of focal units $\mathcal { F }$ is too large the set of allowed permutations that preserve treatment assignment over $\mathcal { F }$ will be too small, thus again resulting in a loss of power. The optimal size of $\mathcal { F }$ will depend on the application.

Permutation tests for composite nulls In our setting, a composite null is any null hypothesis that allows W to have some effect on Y, but restricts how these effects can manifest themselves. To understand how to design permutation tests for composite nulls, it is helpful to review the ingredients that made our test for $H _ { 0 }$ work:

1. Our knowledge of the randomization design enabled us to create a set $\mathcal { W } ^ { \prime }$ of possible treatment assignments (which includes the realized one).

2. Under the null hypothesis, $T ( \mathbf { Y } ( \mathbf { w } ) , \mathbf { w } ) \ = \ T ( \mathbf { Y } ( \mathbf { W } ) , \mathbf { w } )$ for all ${ \textbf { w } } \in$ $\mathcal { W } ^ { \prime }$ , and so we can impute the counterfactual test statistics $T ( \mathbf { Y } ( \mathbf { w } )$ , w) we would have observed under alternate randomizations using only the observed outcomes $\mathbf { Y } = \mathbf { Y } ( \mathbf { W } )$ .  
3. Conditionally on knowing that we chose the set $\mathcal { W } ^ { \prime }$ in step 1, the distribution of W is uniformly random over $\mathcal { W } ^ { \prime }$ .

The key step here is step 2; and, under the sharp null $H _ { 0 }$ , it is easy to see that we can always impute $T ( \mathbf { Y } ( \mathbf { w } ) , \mathbf { w } )$ from Y for any test statistic $T$ and any treatment vector w.

In contrast, under composite nulls, we will no longer be able to impute any and all test statistics for all w because the treatment now can have some (restricted) effects on the outcomes. We will still be able to make progress by being more careful in our choice of T and set $\mathcal { W } ^ { \prime }$ of considered treatments; doing so, however, leads to subtle challenges in step 3 above.

The general roadmap for designing permutation tests for a generic composite null H involves first observing the realized treatment W, and then choosing a set of alternate treatment assignments $\mathcal { W } ^ { \prime }$ that allows us to impute test statistic T under H. The following result gives general guarantees for permutation tests of this type.

Theorem 11.2. Suppose that we want to test a composite null hypothesis H and that Assumption 11.2 holds. After observing W, we choose a (potentially random) set of treatment vectors $\warrow \subseteq \warrow$ with $\mathbf { W } \in \mathcal { W } ^ { \prime }$ , and a (potentially random) test statistic with the property that, under H, $T ( \mathbf { Y } ( \mathbf { w } ) , \mathbf { w } ) =$ $T ( \mathbf { Y } ( \mathbf { W } ) , \mathbf { w } )$ for all $\mathbf { w } \in \mathbf { W }$ . Let

$$
\varphi_ {\mathbf {w}} \left(\mathcal {W} ^ {\prime}, T\right) = \mathbb {P} \left[ \mathcal {W} ^ {\prime}, T \mid \mathbf {W} = \mathbf {w} \right] \tag {11.10}
$$

denote the probability of selecting the treatment set $\mathcal { W } ^ { \prime }$ and test statistic T given that the realized treatment vector was w. Then, the re-weighted permutation p-value

$$
p = \frac {\sum_ {\mathbf {w} \in \mathcal {W} ^ {\prime}} \varphi_ {\mathbf {w}} \left(\mathcal {W} ^ {\prime} , T\right) 1 \left(\left\{T (\mathbf {Y} , \mathbf {W}) \leq T (\mathbf {Y} , \mathbf {w}) \right\}\right)}{\sum_ {\mathbf {w} \in \mathcal {W} ^ {\prime}} \varphi_ {\mathbf {w}} \left(\mathcal {W} ^ {\prime} , T\right)} \tag {11.11}
$$

is valid against the null, i.e., under H, $\mathbb { P } \left[ p \leq \alpha \right] \leq \alpha$ for all $0 \leq \alpha \leq 1$ .

Proof. The pair $( \mathcal { W } , T )$ is chosen only based on knowledge of W, and under a constraint that we must have $\mathbf { W } \in \mathcal { W } ^ { \prime }$ . Thus, under Assumption 11.2, we can use Bayes’ rule to verify that, conditionally on knowing that $\mathcal { W } ^ { \prime }$ was selected as the set of considered randomizations and that Y was observed,

$$
\mathbb {P} \left[ \mathbf {W} = \mathbf {w} \mid \mathcal {W} ^ {\prime}, \mathbf {Y} \right] = \varphi_ {\mathbf {w}} \left(\mathcal {W} ^ {\prime}, T\right) / \sum_ {\mathbf {w} ^ {\prime} \in \mathcal {W} ^ {\prime}} \varphi_ {\mathbf {w} ^ {\prime}} \left(\mathcal {W} ^ {\prime}, T\right) \tag {11.12}
$$

for all $\mathbf { w } \in \mathcal { W } ^ { \prime }$ . The proof then follows exactly the same argument as used in Theorem 11.1. Let T be as defined in the proof of Theorem 11.1, and let $S _ { ( 1 ) } \geq S _ { ( 2 ) } \geq . . . \geq S _ { ( | \mathcal { W } ^ { \prime } | ) }$ be order statistics of the test statistics, with associated weights $\varphi _ { ( 1 ) } , \ldots , \varphi _ { ( | \mathcal { W } ^ { \prime } | ) }$ used in (11.11). If there are no ties in $\tau$

$$
\mathbb {P} \left[ p \leq \alpha   |   \mathcal {W} ^ {\prime},   \mathbf {Y} \right] = \max \left\{t = \sum_ {j = 1} ^ {k} \varphi_ {(j)} / \sum_ {j = 1} ^ {| \mathcal {W} ^ {\prime} |} \varphi_ {(j)}: t \leq \alpha \right\}, \tag {11.13}
$$

and the presence of ties will again only make p strictly larger.

![image_09](images/image_09.png)

Application: Testing $H _ { 1 }$ We now return to the question of how to design a permutation test for the presence of interference using the test statistic (11.8). Using notation from Theorem 11.2, the imputability property (11.9) for focal unit based test statistics implies that we can use them together with the permutation set

$$
\mathcal {W} ^ {\prime} \left(\mathcal {F}\right) = \left\{w \in \mathcal {W}: w _ {i} = W _ {i} \text {   for   all   } i \in \mathcal {F} \right\}. \tag {11.14}
$$

Theorem 11.2 then applies directly. The remaining challenge is that we now need to account for the weights $\varphi _ { \mathbf { w } } ( \mathcal { F } ) = \mathbb { P } \lceil \mathcal { F } \rceil \mathbf { W } = \mathbf { w } \rceil$ , which measure dependence between our choice of focal units and the realized randomization. In principle, one could compute these quantities and apply (11.11) directly; however, in the existing literature, most proposals have sought choices of $\mathcal { F }$ obviate the need to consider weights by construction.

One way to side-step this challenge, discussed by Athey, Eckles, and Imbens [2018a], is to choose the set of focal units $\mathcal { F }$ deterministically, without looking at W. In this case, $\mathbb { P } \lceil \mathcal { F } \rceil \mathbf { W } = \mathbf { w } \rceil = 1$ , and the weights vanish and can thus be ignored. Such an approach, however, may not be optimal in terms of power; e.g., if we use (11.8) as our test statistic, then there’s seemingly no value from including any treated units in $\mathcal { F }$ (since they are ignored by the test statistic).

Basse, Feller, and Toulis [2019] noted that in some settings we can also construct randomized choices F for which the weights $\varphi _ { \mathbf { w } } ( \mathcal { F } )$ vanish—and that this can help with power. The main idea is that if we can guarantee that $\varphi _ { \mathbf { w } } ( \mathcal { F } )$ is constant for all $\mathbf { w } \in \mathcal { W } ^ { \prime }$ , the we can ignore the weights because they cancel out in (11.11). Consider, for example, a design where all units are first divided into equally sized clusters $C _ { k }$ for $k = 1 , \ldots , K$ , and then we randomize $n _ { 1 }$ units to treatment such that at most one person per cluster is treated, i.e., we run a completely randomized experiment over64

$$
\mathcal {W} = \left\{\mathbf {w} \in \{0, 1 \} ^ {n}: \sum_ {i} w _ {i} = n _ {1}, \sum_ {\{i \in C _ {k} \}} w _ {i} \leq 1 \text { for all } 1 \leq k \leq K \right\}. \tag {11.15}
$$

Then, if we construct F by selecting exactly one control unit per cluster, one can check that in fact $\varphi _ { \mathbf { w } } ( \mathcal { F } )$ is constant for all $\mathbf { w } \in \mathcal { W } ^ { \prime }$ .

## 11.3 Bibliographic notes

The general approach of modeling causal effects under interference using an extended set of potential outcomes goes back to early work by Halloran and Struchiner [1995], Hudgens and Halloran [2008] and Sobel [2006]. The use of exposure mappings to mitigate the curse of dimensionality was introduced by Aronow and Samii [2017] and Manski [2013].

The paradigm for causal inference used in Chapter 11.2, i.e., one focused on testing various null hypotheses that restrict how treatment can affect potential outcomes, is often called the “Fisherian approach” in recognition of the seminal work of Fisher [1935] on permutation testing. The Fisherian approach is then contrasted with the “Neymanian approach”, which is focused on estimating average treatment effects (as opposed to exact restrictions on the potential outcomes)—and is also the approach we have focused on in most of this book. When the distinction needs to be made, the sharp null $( \mathrm { e . g . , } Y _ { i } ( 0 ) = Y _ { i } ( 1 )$ for all i) is often referred to as the Fisher null, while the usual (or weak) null (e.g., $\begin{array} { r } { \sum _ { i } \left( Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \right) = 0 ) } \end{array}$ is referred to as the Neyman null; see Ding [2017] for further discussion.

Our discussion of permutation tests under interference is adapted from Athey, Eckles, and Imbens [2018a] and Basse, Feller, and Toulis [2019]. One aspect of permutation testing that we have not put much emphasis on in this chapter is the choice of test statistic: We simply used point estimates of various quantities likely to be non-zero under the alternative, e.g., the difference in means in (11.4). Permutation tests are exact under the sharp null, regardless

<!-- footnote -->

- This is an asymptotic scaling result, and not a finite-sample result. Gelman and Imbens [2019] consider practical, finite-sample behavior of higher-order local regression adjustments and, based on their findings, caution against using such higher-order adjustments.

<!-- footnote end -->

<!-- footnote -->

- We note an unfortunate naming collision: When we say that local linear regression (8.4) is a linear estimator (8.12), we use the descriptor “linear” with two different meanings.

<!-- footnote end -->

<!-- footnote -->

- There is no need for an absolute value inside the sup-term used to define $I _ { B } ( \gamma )$ because the class of twice differentiable functions is symmetric around zero. This fact will prove to be useful down the road.

<!-- footnote end -->

<!-- footnote -->

- When $Z _ { i }$ has a discrete distribution, the definition of $\tau _ { c }$ via (8.3) needs careful interpretation—as we need to be able to talk about $\mu _ { ( w ) } ( z )$ at values of $z$ that do not belong to the support of the running variable. All guarantees provided here hold if we define $\mu _ { ( w ) } ( z )$ outside of the support of z to be an arbitrary function that interpolates between the support points of z while satisfying $| \mu _ { ( w ) } ^ { \prime \prime } ( z ) | \le B$ .

<!-- footnote end -->

<!-- footnote -->

- When working with geographic regression discontinuities, some authors have tried to collapse the problem by only considering a univariate running variable that codes distance to the boundary of ${ \mathcal { A } } .$ Such an approach, however, is sub-optimal from a statistical point of view as it throws away relevant information.

<!-- footnote end -->

<!-- footnote -->

- There is a slight abuse of notation here: $\mathbb { P } \left[ Y | d o ( W = w ) \right]$ is strictly speaking not a conditional distribution; rather, again, it is the unconditional distribution of $\bar { Y }$ in the SEM where we’ve replaced the equations for W with hard-coded values.

<!-- footnote end -->

<!-- footnote -->

- Although the linear form (9.11) may look familiar, the standard linear regression estimator is not consistent for $\tau$ here. In the setting of Figure 9.3, U affects both W and the error term $\varepsilon ,$ and so Cov $[ \varepsilon _ { i } , W _ { i } ] \neq 0$ in general. Thus, in large samples, the linear regression estimator will not in general be equal to τ :
- $\widehat { \tau } _ { O L S }  _ { p } \frac { \mathrm { C o v } [ Y _ { i } , W _ { i } ] } { \mathrm { V a r } [ W _ { i } ] } = \frac { \mathrm { C o v } [ \tau W _ { i } + \varepsilon _ { i } , W _ { i } ] } { \mathrm { V a r } [ W _ { i } ] } = \tau + \frac { \mathrm { C o v } [ \varepsilon _ { i } , W _ { i } ] } { \mathrm { V a r } [ W _ { i } ] } \neq \tau .$ τˆOLS →p Var [W ] = τ +

<!-- footnote end -->

<!-- footnote -->

- For example, in the setting of Example 9, we may be interested in using both wind speed and precipitation as “storminess” instruments that can nudge prices. Furthermore, we may believe that these instruments act non-linearly $( \mathrm { e . g . }$ , below a certain threshold there’s no effect, and above another threshold fishing becomes impossible).

<!-- footnote end -->

<!-- footnote -->

- For example, this can be verified by applying Theorems 5.41 and 5.42 in Van der Vaart [1998], and noting that the moment condition (9.20) has a unique solution with probability tending to 1 whenever Cov $[ W , w ( Z ) ] \neq 0$ .

<!-- footnote end -->

<!-- footnote -->

- As before, because $W _ { i }$ is not independent of $\varepsilon _ { i } ,$ we cannot learn $g ( \cdot )$ by simply doing a (non-parametric) regression of $Y _ { i }$ on $W _ { i } , \mathrm { i . e . , } g ( w ) \neq \mathbb { E } \left. \left\lceil Y _ { i } \right\rceil W _ { i } = w \right\rceil$ .

<!-- footnote end -->

<!-- footnote -->

- Note that the available data is richer if the trial design involves assigning placebo drugs to the controls, as in this case compliance can be measured for both the treated units (did they take the drug?) and controls (did they take the placebo?) [Efron and Feldman, 1991].
- Similar statistical patters can also arise outside of randomized trials. For example, in order to study the effect of military service on long-term income, Angrist [1990] uses the draft lottery as an instrument for the treatment of interest, i.e., military service. Both the instrument and treatment are binary here, and so methods developed to understand non-compliance in randomized trials can be directly applied to this setting.

<!-- footnote end -->

<!-- footnote -->

- This type of model is also referred to as a simultaneous equation model, as $P _ { i }$ is determined by simultaneously considering the supply and demand “equations” $S _ { i } = S _ { i } ( P _ { i } , Z _ { i } )$ and $Q _ { i } = Q _ { i } ( P _ { i } , Z _ { i } )$ .

<!-- footnote end -->

<!-- footnote -->

- To be precise, when studying demand elasticity we’d actually run this analysis with outcome $\log ( Q _ { i } )$ and treatment log(Pi). Here we’ll ignore the logs for simplicity; introducing logs doesn’t add any conceptual difficulties.
- The differentiability assumption on $Q _ { i } ( \cdot )$ is only made for simplicity and is not actually needed here: We’ve assumed that $Q _ { i } ( \cdot )$ is monotone increasing so that the distributional derivative must exist, and all arguments in the proof can be generalized to work with a distributional derivative.

<!-- footnote end -->

<!-- footnote -->

- In this chapter and the next, we will render vectors of observables across units in bold.

<!-- footnote end -->

<!-- footnote -->

- This test statistic is simple, but from a large-sample theory point of view others may be preferable; see the bibliographic notes at the end of this chapter for a discussion.

<!-- footnote end -->

<!-- footnote -->

- The use of randomization is optional. Setting $B = | \mathcal { W } | - 1$ will result in running a permutation over all possible randomizations $\mathcal { W } _ { : }$ , and recovers Fisher’s exact test.

<!-- footnote end -->

<!-- footnote -->

- Basse, Feller, and Toulis [2019] considered a different, two-stage design where we first choose which clusters give to the treatments to uniformly at random, and then pick one treated unit from each of these clusters—again uniformly at random. However, in the case of equally sized clusters, their design matches the completely randomized one considered here.

<!-- footnote end -->

of our choice of test statistic. However, the choice of test statistic matters in terms of the power we get under various alternatives of interest, and here test statistics based on point estimates of treatment effects, e.g., the difference in means used in (11.4), can perform unexpectedly poorly.

To understand the power issue, consider the large-sample behavior of a permutation test in a setting with

$$
\binom {Y _ {i} (0)} {Y _ {i} (1)} \sim \mathcal {N} \left(\binom {\mu_ {0}} {\mu_ {1}}, \left( \begin{array}{c c} \sigma_ {0} ^ {2} & 0 \\ 0 & \sigma_ {1} ^ {2} \end{array} \right)\right), \tag {11.16}
$$

and $n _ { 1 } / n = \pi \in ( 0 , 1 )$ . The difference in means test static on the original data has distribution $T _ { 0 } = \mathcal { N } \left( \mu _ { 1 } - \mu _ { 0 } , \sigma _ { T } ^ { 2 } / n \right)$ with $\sigma _ { T } ^ { 2 } = \sigma _ { 0 } ^ { 2 } / ( 1 - \pi ) + \sigma _ { 1 } ^ { 2 } / \pi$ . The usual t-test would then reject the null when the ratio $\sqrt { n } T _ { 0 } / \sigma _ { T }$ is far from 0. On the other hand, because the permutation test jumbles the data, one can check that the behavior of $T _ { b } ^ { \prime }$ depends on moments of the pooled data instead, and the permutation distribution can be approximated as [Romano, 1990]

$$
\mathcal {L} \left(T _ {b} ^ {\prime}\right) \approx \mathcal {N} \left(0, \sigma_ {Y} ^ {2} / n\right), \sigma_ {Y} ^ {2} = \pi (1 - \pi) \left(\mu_ {1} - \mu_ {0}\right) ^ {2} + \frac {(1 - \pi) \sigma_ {0} ^ {2} + \pi \sigma_ {1} ^ {2}}{\pi (1 - \pi)}, (1 1. 1 7)
$$

thus implying that, effectively, the permutation test rejects the null when $\sqrt { n } T _ { 0 } / \sigma _ { Y }$ is far from 0. We can then directly read out several unexpected behaviors of the permutation test from this comparison. If $\sigma _ { 0 } ^ { 2 } = \sigma _ { 1 } ^ { 2 }$ and $\mu _ { 1 } \neq \mu _ { 0 }$ (i.e., the treatment shifts the mean but not that variance), then $\sigma _ { Y } ^ { 2 } > \sigma _ { T } ^ { 2 }$ and so the permutation test will be less powerful than the usual t-test. On the other hand, permutation tests with a difference in means test statistic can have non-trivial power in settings where the Neymanian null of zero average effect holds, i.e., they are generally not valid (even asymptotically) against the Neymanian null. To see this, note that when if $\mu _ { 1 } = \mu _ { 0 } , \pi < 0 . 5$ and $\sigma _ { 1 } ^ { 2 } > \sigma _ { 0 } ^ { 2 } .$ then $\sigma _ { Y } ^ { 2 } < \sigma _ { T } ^ { 2 }$ and so the permutation test must have more power than the usual t-test (which in turn has the nominal level here).

One can solve this problem—and generally improve the large-sample behavior of permutation tests—by using studentized test statistics, e.g., a two-sample t-statistic instead of (11.4), or a heteroskedasticity-robust regression t-statistic instead of (11.8). Chung and Romano [2013] provide results implying that, at least in the setting of Theorem 11.1, a permutation test using a studentized test statistic pairs finite-sample validity against the sharp (Fisher) null hypothesis while matching the behavior of the usual test against the Neymanian null of a zero average treatment effect in large samples. Cohen and Fogarty [2022] discusses further results on unifying Neymanian and Fisherian approaches to testing for the presence of causal effects.
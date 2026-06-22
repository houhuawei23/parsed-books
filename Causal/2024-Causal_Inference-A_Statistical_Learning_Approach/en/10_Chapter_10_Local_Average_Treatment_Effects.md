# Chapter 10 Local Average Treatment Effects

Instrumental variable regression is commonly used to estimate the effect of an endogenous treatment. In the previous chapter we saw how, given the structural equation model depicted in Figure 9.3 and a linear specification (9.11) governing the effect of the treatment $W _ { i }$ and the outcome $Y _ { i } ,$ we can use an instrument $Z _ { i }$ to identify the treatment effect parameter τ as a ratio of covariances,

$$
\tau = \operatorname{Cov} \left[ Y _ {i}, Z _ {i} \right] / \operatorname{Cov} \left[ W _ {i}, Z _ {i} \right], \tag {10.1}
$$

and consistently estimate τ via

$$
\hat {\tau} _ {I V} = \widehat {\mathrm{Cov}} \left[ Y _ {i}, Z _ {i} \right] / \widehat {\mathrm{Cov}} \left[ W _ {i}, Z _ {i} \right]. \tag {10.2}
$$

In general, however, researchers in causal inference are often skeptical of interpreting target estimands that are only defined and understood as parameters in a linear model; and ${ \mathrm { s o } } ,$ in this chapter, we will revisit our analysis of the instrumental variable estimator $\hat { \tau } _ { I V }$ without assuming linearity—or, equivalently, under an assumption that (9.11) may be misspecified.

Without linearity, the estimator $\hat { \tau } _ { I V }$ still converges to a large-sample limit

$$
\hat {\tau} _ {I V} \rightarrow \tau_ {L A T E} := \operatorname{Cov} \left[ Y _ {i}, Z _ {i} \right] / \operatorname{Cov} \left[ W _ {i}, Z _ {i} \right] \tag {10.3}
$$

whenever Cov $[ W _ { i } , Z _ { i } ] \neq 0 $ ; however, it is no longer immediately clear how to interpret this limit. In this chapter, we will study what this limit quantity is, and when it can be understood as a causal quantity. We will survey a number of economic models where endogenous selection into treatment may be a concern and find that—under fairly weak assumptions—this limit is a weighted treatment effect with weights depending on (unobserved) attributes that control how responsive each unit is to the nudge given by the instrument. Following Imbens and Angrist [1994], when these conditions hold, we refer to this limit as the local average treatment effect (LATE), i.e., the treatment effect “local” to those responsive to the instrument.

## 10.1 Non-compliance in randomized trials

The simplest setting in which we can discuss non-parametric identification using instrumental variables is when estimating the effect of a binary treatment under non-compliance. Suppose, for example, that we’ve set up a randomized study to examine the effect of taking a drug to lower cholesterol. But, although we randomly assigned treatment, some people don’t obey the randomization: Some subjects given the drugs may fail to take them, while others who were assigned control may procure cholesterol lowering drugs on their own. In this case, we have56

• An outcome $Y _ { i } \in \mathbb { R }$ , with the usual interpretation;
• The treatment $W _ { i } \in \{ 0 , 1 \}$ that was actually received (i.e., did the subject take the drug), which is not random because of non-compliance; and
• The assigned treatment $Z _ { i } \in \{ 0 , 1 \}$ which is random.

A popular way to analyze this type of data is using instrumental variables, where we interpret treatment assignment $Z _ { i }$ as an exogenous “nudge” on the treatment $W _ { i }$ that was actually received.57

If one believes in the partially linear structural model (9.11) considered in the previous chapter, then one can consistently estimate τ via (10.3) provided that assigned treatment in fact nudges the received treatment, i.e., Cov $[ W _ { i } , Z _ { i } ] \neq 0$ . In practice, however, one may doubt the validity the constant treatment effect assumption (9.11), and suspect that people who comply with the treatment respond differently to the treatment than those who don’t comply. For example, there may exists a class of patients who chose to comply because they knew they’d benefit a lot from the treatment; or conversely other patients may have chosen not to comply because they knew they had a disproportionate risk of being hurt by it.

Potential outcomes under non-compliance A more careful approach starts by writing down potential outcomes. First, because $W _ { i }$ is non-random and may respond to $Z _ { i } ,$ we need to have potential outcomes for the treatment variable in terms of the instrument, i.e., there are $\{ W _ { i } ( 0 ) , W _ { i } ( 1 ) \}$ such that $W _ { i } = W _ { i } ( Z _ { i } )$ . Second, of course, we need to define potential outcomes for the outcome, which may in principle respond to both $W _ { i }$ and $Z _ { i } { \mathrm { : } }$ we have $\{ Y _ { i } ( w , z ) \} _ { w , z \in \{ 0 , 1 \} }$ such that $Y _ { i } = Y _ { i } ( W _ { i } , Z _ { i } )$ .

Given this notation, we now revisit our assumptions for what makes a valid instrument:

• Exclusion restriction. Treatment assignment only affects outcomes via receipt of treatment, i.e., $Y _ { i } ( w , z ) = Y _ { i } ( w )$ for all w and z.
• Exogeneity. The treatment assignment is randomized, meaning that $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , W _ { i } ( 0 ) , W _ { i } ( 1 ) \} \perp Z _ { i }$ .
• Relevance. The treatment assignment affects receipt of treatment, meaning that E $[ W _ { i } ( 1 ) - W _ { i } ( 0 ) ] \neq 0$ .

Finally, we make one last assumption about how people respond to treatment. Defining each subject’s compliance type as $C _ { i } = \{ W _ { i } ( 0 ) , W _ { i } ( 1 ) \}$ , we note that there are only 4 possible compliance types here:

<table><tr><td></td><td> $W_{i}(1) = 0$ </td><td> $W_{i}(1) = 1$ </td></tr><tr><td> $W_{i}(0) = 0$ </td><td>never taker</td><td>complier</td></tr><tr><td> $W_{i}(0) = 1$ </td><td>defier</td><td>always taker</td></tr></table>

Our last assumption is that there are no defiers, i.e., $\mathbb { P } \left[ C _ { i } = \{ 1 , 0 \} \right] = 0 ;$ this assumption is often also called monotonicity. Given these 4 assumptions, we obtain the following simple characterization of the IV estimand (10.3).

Theorem 10.1. Consider a sampling distribution with a binary treatment $W _ { i }$ and a binary instrument $Z _ { i }$ , and satisfying the 4 assumptions given above (exogeneity, relevance, monotonicity, and the exclusion restriction). Then,

$$
\tau_ {L A T E} = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0)   |   C _ {i} = \text { complier } \right]. \tag {10.4}
$$

Proof. With a binary treatment and instrument, the IV estimand (10.3) can be written as

$$
\tau_ {L A T E} = \frac {\mathbb {E} \left[ Y _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = 0 \right]}{\mathbb {E} \left[ W _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ W _ {i} \mid Z _ {i} = 0 \right]},
$$

and this ratio is well defined thanks to the relevance assumption. Furthermore,

$$
\begin{array}{l} \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = 0 \right] \\ = \mathbb {E} \left[ Y _ {i} \left(W _ {i} (1)\right) \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ Y _ {i} \left(W _ {i} (0)\right) \mid Z _ {i} = 0 \right] \quad (\text { exclusion }) \\ = \mathbb {E} \left[ Y _ {i} (W _ {i} (1)) - Y _ {i} (W _ {i} (0)) \right] \quad (\text { exogeneity }) \\ = \mathbb {E} \left[ 1 \left(\left\{C _ {i} = \text { complier } \right\}\right) \left(Y _ {i} (1) - Y _ {i} (0)\right) \right], \quad \text {(monotonicity)} \\ \end{array}
$$

and similarly that

$$
\mathbb {E} \left[ W _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ W _ {i} \mid Z _ {i} = 0 \right] = \mathbb {P} \left[ \{C _ {i} = \text {complier} \} \right].
$$

The result (10.4) then follows by Bayes’ rule.

![image_07](images/image_07.png)

Although this is a very simple result, it already gives us some encouragement that IV methods can be interpreted in a non-parametric setting: When the constant treatment effect model (9.11) doesn’t hold, the average treatment effect $\tau _ { A T E } ~ = ~ \mathbb { E } \left[ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \right]$ is clearly not identified without more data, because we don’t have any observations on treated never takers, etc. However, under reasonable assumptions, IV methods let us estimate the most meaningful quantity we can identify here, namely the average treatment effect among those who comply with the treatment as assigned by the experimenter.

Example 1 (Continued). In the example of Finkelstein et al. [2012] on the Oregon Medicaid lottery, introduced in Chapter 1, roughly 35,000 of 90,000 lottery participants were allowed to apply for Medicaid. However, of the 35,000 lottery winners, only about 30% in fact enrolled for Medicaid: Some didn’t complete the application, and some hadn’t met the requirements for joining the lottery to begin with (e.g., their income was too high). The average treatment effect measured via the difference-in-means estimator thus does not directly quantify the benefit of Medicaid enrollment here. But, because there are plausibly no defiers here, we can divide the raw difference-in-means by 0.3 to get a local average treatment effect, i.e., an estimate of the average benefit for those who would in fact enroll for Medicaid if they win the lottery.

Multiple instruments In some applications, we may have access to data from multiple randomized trials that can be used to study a treatment effect via a non-compliance analysis. Consider, for example, a marketing application where a company wants to study the effect of subscription to a loyalty program (Wi) on long-term customer revenue (Yi), and has access to multiple randomized trials whose treatments $\left( Z _ { i } \right)$ effectively nudge customers to join the loyalty program and can thus be used as instruments. For example, one randomized trial may offer discounts for joining the loyalty program $( Z _ { i } = 1$ ({customer received a discount})) while another may show advertisements $( Z _ { i } = 1$ ({customer was shown an ad for the program})).

If we just focus on one of the instruments, then the methods developed above can be applied directly. However, one may also be tempted to somehow pool the instruments. In the previous chapter we saw that, under the linear treatment effect model, multiple instruments could be combined into a single optimal instrument, and the optimal instrument corresponds to the summary of all the instruments that best predicts the treatment (Theorem 9.2).

Without the linear treatment effect model, however, we caution that no such result is available. Different instruments may induce difference compliance patterns, and so the LATEs identified different instruments may not be the same; and a pooled instrument produced using the construction in Theorem 9.2 may induce yet another compliance pattern. For example, in our marketing example, the ATE for customers who respond to a discount may be different from the ATE for customers who respond to an advertisement.

As such, when working without the linear treatment assumption (9.11), if there are multiple instruments to choose from a researcher may prefer to simply use the instrument whose LATE most closely matches a policy-relevant effect of interest. One could also run separate IV analyses using different instruments, and use discrepancies between the resulting estimates to argue for heterogeneity in treatment effects across different compliance groups.

## 10.2 Latent choice models

Instrumental variables regression is also used in many applications that go beyond the binary-treatment-binary-instrument setting considered above. In economics, there has been longstanding interest in models where agents make choices (e.g., take a job, go to college, start a company) in a way that is determined by latent and often unobserved attributes (e.g., skills, motivation, risk tolerance), and these latent attributes also influence economic outcome variables of interest (e.g., lifetime income) [Heckman, 1979, Roy, 1951].

Without access to further data or assumptions, it is generally impossible to measure the causal effect of such choices because of the inherent endogeneity (i.e., the dependence of treatment selection on latent attributes). Instrumental variable methods, however, can provide a path forward in settings where we have access to data on exogenous shocks that can be argued to nudge selection into treatment in a quasi-random manner. We will here study the behavior of IV regression in a number of such choice models, again without making the constant treatment effect assumption (9.11) and instead allowing treatment effects to depend on unobserved latent attributes.

Supply and demand In many settings, it is of considerable interest to know the price elasticity of demand, i.e., how demand would respond to price changes. In a typical marketplace, prices are not exogenous—rather, they arise from an interplay of supply and demand—and so estimating the elasticity requires an instrument. This is an example of a latent choice model, as both supply and demand are determined by individual choices shaped by market prices together with unobserved factors (e.g., willingness to pay or production costs).

One can formalize the relationship of supply and demand via potential outcomes as follows. For each marketplace $i = 1 , . . . , n .$ , there is a supply curve $S _ { i } ( p , z )$ and a demand curve $Q _ { i } ( p , z )$ , corresponding to the supply (and respectively demand) that would arise given price $p \in \mathbb R$ and some instrument $z \in \{ 0 , 1 \}$ that may affect the marketplace (the instrument could, e.g., capture the presence of supply chain events that make production harder and thus reduce supply). For simplicity, we may take $S _ { i } ( \cdot , z )$ to be continuous and increasing and $Q _ { i } ( \cdot , z )$ to be continuous and decreasing.

Example 9 (Continued). In the setting of Angrist, Graddy, and Imbens [2000] one may argue that, on closer inspection, the DAG given in Figure 9.3 does not present a complete structural explanation for the interplay of supply, demand, prices and weather; and that the above market equilibrium model (with weather as the instrument) provides a better fit. The discussion below will show how we can still make sense of the basic IV estimator $\hat { \tau } _ { I V }$ while framing causal effects in terms of this equilibrium model.

Given this setting, suppose that first the instrument $Z _ { i }$ gets realized; then prices $P _ { i }$ arise by matching supply and demand, such that $P _ { i }$ is the unique solution to the market equilibrium condition $^ { 5 8 } \ S _ { i } ( P _ { i } , Z _ { i } ) = Q _ { i } ( P _ { i } , Z _ { i } )$ . The researcher observes the instrument $Z _ { i }$ , the market clearing price $P _ { i }$ (“the treatment”) and the realized demand $Q _ { i } = Q _ { i } ( P _ { i } , Z _ { i } )$ (“the outcome”). We say that $Z _ { i }$ is a valid instrument for measuring the effect of prices on demand if the following conditions hold:

• Exclusion restriction. The instrument only affects demand via supply, and cannot have a direct effect on it: $Q _ { i } ( p , z ) = Q _ { i } ( p )$ for all p and z.
• Exogeneity. The instrument is as good as random, $\{ Q _ { i } ( p ) , S _ { i } ( p , z ) \}$ ⊥⊥ $Z _ { i }$ .
• Relevance. The instrument affects prices, Cov $[ P _ { i } , Z _ { i } ] \neq 0$ .

• Monotonicity. The instrument never increases supply, i.e., $S _ { i } ( P _ { i } , 1 ) \leq$ $S _ { i } ( P _ { i } , 0 )$ almost surely.

Given this setting, we seek to estimate demand elasticity via (10.3).59

Now, although this may seem like a complicated setting, it turns out that the IV estimand where we use $Z _ { i }$ as an instrument to measure the effect of $P _ { i }$ on $Q _ { i }$ is well behaved—and admits a characterization as a weighted average of the derivative of $Q _ { i } ( p )$ .

Theorem 10.2. In the above supply-demand model, suppose furthermore that $Q _ { i } ( p )$ is differentiable and write $Q _ { i } ^ { \prime } ( p )$ for its derivative.60 Then,

$$
\tau_ {L A T E} = \frac {\int \mathbb {E} \left[ Q _ {i} ^ {\prime} (p) \mid P _ {i} (0) \leq p \leq P _ {i} (1) \right] \mathbb {P} \left[ P _ {i} (0) \leq p \leq P _ {i} (1) \right] d p}{\int \mathbb {P} \left[ P _ {i} (0) \leq p \leq P _ {i} (1) \right] d p}, \tag {10.5}
$$

Proof. Because $Z _ { i }$ is binary, we can write

$$
\tau_ {L A T E} = \frac {\mathbb {E} \left[ Q _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ Q _ {i} \mid Z _ {i} = 0 \right]}{\mathbb {E} \left[ P _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ P _ {i} \mid Z _ {i} = 0 \right]}.
$$

Now, under the assumptions made here, i.e., that the instrument suppresses supply and that the supply and demand curves are monotone increasing and decreasing respectively, the instrument must have a monotone increasing effect on prices: $P _ { i } ( 1 ) \ge P _ { i } ( 0 )$ . Then,

$$
\mathbb {E} \left[ Q _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ Q _ {i} \mid Z _ {i} = 0 \right]
$$

$$
= \mathbb {E} \left[ Q _ {i} (P _ {i} (1)) \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ Q _ {i} (P _ {i} (0)) \mid Z _ {i} = 0 \right] \quad (\text { exclusion })
$$

$$
= \mathbb {E} \left[ Q _ {i} (P _ {i} (1)) - Q _ {i} (P _ {i} (0)) \right] \quad (\text { exogen. })
$$

$$
= \mathbb {E} \left[ \int_ {P _ {i} (0)} ^ {P _ {i} (1)} Q _ {i} ^ {\prime} (p) d p \right] \quad (\text { monot. })
$$

$$
= \int \mathbb {E} \left[ Q _ {i} ^ {\prime} (p) \mid P _ {i} (0) \leq p \leq P _ {i} (1) \right] \mathbb {P} \left[ P _ {i} (0) \leq p \leq P _ {i} (1) \right] d p, \quad (\text { Fubini })
$$

and the denominator in (10.5) can be characterized via similar means to obtain (10.5). □

The above result is not quite as interpretable as the one obtained in Theorem 10.1, where the LATE was founds to exactly match the average treatment effect for the compliers. However, as seen in the remarks below, the characterization (10.5) can still be helpful in understanding the practical behavior of IV methods in applications involving supply-demand equilibrium formation.

Remark 10.1. Under the setting of Theorem 10.2, if individual demand functions are linear in prices, $Q _ { i } ^ { \prime } ( p ) = \alpha _ { i } + \beta _ { i } p .$ , then

$$
\tau_ {L A T E} = \mathbb {E} \left[ \beta_ {i} \left(P _ {i} (1) - P _ {i} (0)\right) \right] / \mathbb {E} \left[ P _ {i} (1) - P _ {i} (0) \right], \tag {10.6}
$$

i.e., the LATE matches the average price parameter weighted by how much the price responds to the instrument. Furthermore, if we have approximate linearity then Theorem 10.2 implies that (10.6) also still holds approximately— and can be used to quantitatively assess the effect of deviations from linearity.

Remark 10.2. Under the setting of Theorem 10.2, if individual demand functions $Q _ { i } ( p )$ are smooth and if the instrument only has a small effect on prices, i.e., $P _ { i } ( 0 ) , P _ { i } ( 1 ) \ \approx \ p _ { 0 }$ for some stable price $p _ { 0 }$ , then $\tau _ { L A T E } \approx \mathbb { E } \left[ Q _ { i } ^ { \prime } ( p _ { 0 } ) ( P _ { i } ( 1 ) - P _ { i } ( 0 ) ) \right] / \mathbb { E } \left[ P _ { i } ( 1 ) - P _ { i } ( 0 ) \right]$ .

Threshold crossing models Another widely used class of choice models arises when agents take a certain action $W _ { i } \ { \mathrm { ( e . g . } } $ , attend college) if their (unobserved) utility $U _ { i }$ from doing so exceeds the cost of taking the action. In settings such as these, if we have an exogenous instrument $Z _ { i }$ that can modify the cost of taking the action $( \mathrm { e . g . }$ , in the case of college attendance, a randomly assigned tuition subsidy), then we may again seek to use this instrument to estimate the effect of $W _ { i }$ on a downstream outcome $Y _ { i } \ { \mathrm { ( e . g . } }$ , lifetime income).

The standard way to model this setting is via a threshold crossing model: We assume that each subject has a latent and endogenous variable $U _ { i }$ such that

$$
W _ {i} = 1 \left(\{U _ {i} \geq c (Z _ {i}) \}\right), \tag {10.7}
$$

where $c ( z )$ gives the cost of treatment as a function of the instrument z, which we will here allow to be continuous valued. This boundary crossing structure yields a valid instrument under analogues to our usual assumptions:

• Exclusion restriction. There are potential outcomes $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \}$ such that $Y _ { i } = Y _ { i } ( W _ { i } )$
• Exogeneity. The treatment assignment is randomized, meaning that $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , U _ { i } \} \perp Z _ { i }$ .

• Relevance. The threshold function $c ( Z _ { i } )$ has non-trivial variation, i.e., $\mathbb { P } \left[ U _ { i } \ge c ( Z _ { i } ) \vert Z _ { i } = z \right]$ is not constant in z.
• Monotonicity. The threshold function $c ( z )$ is non-increasing in $z .$

Finally, define the marginal treatment effect

$$
\tau (u) = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \mid U _ {i} = u \right]. \tag {10.8}
$$

Our goal is to show that IV methods recover a weighted average of the marginal treatment effect $\tau ( u )$ . Below, for convenience, we assume that the instrument is Gaussian, ${ \mathrm { i . e . , ~ } } Z _ { i } \sim { \mathcal { N } } \left( 0 , 1 \right)$ , as this allows us to apply Stein’s lemma; more general results without assuming such Gaussianity are given in Heckman and Vytlacil [2005].

Theorem 10.3. Given the threshold crossing model discussed above, suppose that $U _ { i }$ has a distribution with density $f ( u )$ and $C D F 1 - G ( u )$ , that $\tau ( u )$ is uniformly bounded, and that $Z _ { i }$ has a Gaussian distribution, $Z _ { i } \sim \mathcal { N } ( 0 , 1 )$ . Suppose furthermore that the threshold function $c ( \cdot )$ is cadlag, $i . e . , \ c ( z ) \ =$ $\operatorname* { l i m } _ { a \downarrow z } c ( a )$ for all $z ,$ and write $c _ { - } ( z ) = \operatorname* { l i m } _ { a \uparrow z } c ( a )$ . Then, there exists a nonnegative, Lebesgue-measurable function $c ^ { \prime } ( z )$ such that $c ( z ) = c _ { 0 } + \textstyle \int _ { - \infty } ^ { z } c ^ { \prime } ( a )$ da, and

$$
\tau_ {L A T E} = \frac {\sum_ {z \in \mathcal {S}} \left(\int_ {c (z)} ^ {c _ {-} (z)} \tau (u) f (u) d u\right) \varphi (z) - \int_ {\mathbb {R} \backslash \mathcal {S}} \tau (c (z)) f (c (z)) c ^ {\prime} (z) \varphi (z) d z}{\sum_ {z \in \mathcal {S}} \left(G (c (z)) - G (c _ {-} (z))\right) \varphi (z) - \int_ {\mathbb {R} \backslash \mathcal {S}} f (c (z)) c ^ {\prime} (z) \varphi (z) d z},
$$

where ${ \mathcal { S } } \subset \mathbb { R }$ is the set of discontinuity points of $c ( \cdot )$ and $\varphi ( \cdot )$ is the standard Gaussian density.

Proof. The fact that $c ( z )$ has a distributional derivative follows immediately from the fact that it is monotone (and thus has bounded variation). Now, in order to establish the desired result, the key task is in characterizing Cov $[ Y _ { i } , Z _ { i } ]$ ; an expression for the denominator of (10.3) can then be obtained via the same argument. First, note that

$$
\begin{array}{l} \operatorname{Cov} \left[ Y _ {i}, Z _ {i} \right] = \operatorname{Cov} \left[ Y _ {i} (0) + (Y _ {i} (1) - Y _ {i} (0)) W _ {i}, Z _ {i} \right] \\ = \operatorname{Cov} \left[ \left(Y _ {i} (1) - Y _ {i} (0)\right) W _ {i}, Z _ {i} \right] \\ = \operatorname{Cov} \left[ \left(Y _ {i} (1) - Y _ {i} (0)\right) 1 \left(\left\{U _ {i} \geq c \left(Z _ {i}\right) \right\}\right), Z _ {i} \right] \\ = \operatorname{Cov} \left[ \tau (U _ {i}) 1 \left(\left\{U _ {i} \geq c (Z _ {i}) \right\}\right), Z _ {i} \right], \\ \end{array}
$$

where the first equality follows from the exclusion restriction, while the second and fourth follow from exogeneity.

Now, write $H ( z ) = \mathbb { E } \left[ \tau ( U _ { i } ) 1 \left( \{ U _ { i } \geq c ( z ) \} \right) \right]$ . Because $Z _ { i }$ is standard Gaussian, Lemma 1 of Stein [1981] implies that

$$
\operatorname{Cov} \left[ H (Z _ {i}), Z _ {i} \right] = \mathbb {E} \left[ H ^ {\prime} (Z _ {i}) \right], \tag {10.9}
$$

where $H ^ { \prime } ( Z _ { i } )$ denotes the distributional derivative of $H ( \cdot )$ . Furthermore, by the chain rule [Ambrosio and Dal Maso, 1990, Corollary 3.1],

$$
H ^ {\prime} (z) = \left\{ \begin{array}{l l} \left(\int_ {c (z)} ^ {c _ {-} (z)} \tau (u) f (u) d u\right) \delta_ {z} & \text { for } z \in \mathcal {S}, \\ - \tau (c (z)) f (c (z)) c ^ {\prime} (z) & \text { else }, \end{array} \right. \tag {10.10}
$$

where $\delta _ { z }$ is the Dirac delta-function at z. The desired result follows.

![image_08](images/image_08.png)

Remark 10.3. Under the setting of Theorem 10.3, suppose that the threshold function $c ( z )$ is constant with a single jump, i.e., $c ( z ) = c _ { 0 } - \delta _ { 1 } 1 \left( \left\{ z \geq z _ { 1 } \right\} \right)$ . Then compliance types collapse into three principal strata: Never-takers with $U _ { i } < c _ { 0 } - \delta _ { 1 }$ , compliers with $c _ { 0 } - \delta _ { 1 } \leq U _ { i } < c _ { 0 }$ , and always takers with $U _ { i } \geq c _ { 0 }$ . Furthermore, just as before, our estimand corresponds to the average treatment effect over the compliers as in Theorem 10.1,

$$
\tau_ {L A T E} = \mathbb {E} \left[ \tau (U _ {i}) \mid c _ {0} - \delta_ {1} \leq U _ {i} <   c _ {0} \right] \tag {10.11}
$$

Remark 10.4. Building on the previous example, now suppose there are K jumps, with cutoff function given by $\begin{array} { r } { c ( z ) = c _ { 0 } - \sum _ { k = 1 } ^ { K } \delta _ { k } 1 \left( \left\{ z \geq z _ { k } \right\} \right) } \end{array}$ ). Then,

$$
\tau_ {L A T E} = \sum_ {k = 1} ^ {K} \mathbb {E} \left[ \tau (U _ {i}) \mid c (z _ {k}) \leq U _ {i} <   c _ {-} (z _ {k}) \right] \gamma_ {k} / \sum_ {k = 1} ^ {K} \gamma_ {k}, \tag {10.12}
$$

$$
\gamma_ {k} = \big (G (c (z _ {k})) - G (c _ {-} (z _ {k})) \big) \varphi (z _ {k}).
$$

In other words, we recover a convex combination of average treatment effects over compliance strata defined by the jumps in $c ( \cdot )$ . These weights depend on the size of the stratum and the density function of the instrument at $z _ { k }$ .

Remark 10.5. Under the setting of Theorem 10.3, suppose $c ( z )$ has no jumps. Then, the LATE corresponds to a weighted average of $\tau ( c ( Z _ { i } ) )$ ),

$$
\tau_ {L A T E} = \int_ {\mathbb {R}} \tau (c (z)) f (c (z)) c ^ {\prime} (z) \varphi (z) d z / \int_ {\mathbb {R}} f (c (z)) c ^ {\prime} (z) \varphi (z) d z. \tag {10.13}
$$

The weights can be interpreted via $f ( c ( z ) ) c ^ { \prime } ( z ) = d / d z \ \mathbb { P } \left[ U _ { i } \geq c ( z ) \right]$ , i.e., they are proportional to the local strength of the instrument.

Estimating the marginal treatment effect Throughout this chapter, we’ve taken it as a given that we’re going to target the estimand (10.3), and then have sought to interpret it in different settings. However, when we get to work with a continuous instrument, it’s possible to target a wider variety of estimands. A first key result is that, in the threshold-crossing model considered above, the marginal treatment effect (10.8) is identified at continuity points of $c ( z )$ via a simple “local $\operatorname { I V } ^ { \prime \prime }$ construction.

Theorem 10.4. Under the setting of Theorem 10.3, suppose that $c ( z )$ is continuously differentiable at z with $c ^ { \prime } ( z ) < 0$ and $U _ { i }$ has a density satisfying $f ( c ( z ) ) > 0$ . Then, the marginal treatment effec $\tau ( u )$ from (10.8) is identified as

$$
\tau (c (z)) = \frac {\frac {d}{d z} \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = z \right]}{\frac {d}{d z} \mathbb {P} \left[ W _ {i} = 1 \mid Z _ {i} = z \right]}. \tag {10.14}
$$

Proof. Under our threshold-crossing model,

$$
\begin{array}{l} \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = z \right] = \mathbb {E} \left[ Y _ {i} (0) + 1 \left(\{U _ {i} \geq c (Z _ {i}) \}\right) (Y _ {i} (1) - Y _ {i} (0)) \mid Z _ {i} = z \right] \\ = \mathbb {E} \left[ Y _ {i} (0) + 1 \left(\{U _ {i} \geq c (z) \}\right) \left(Y _ {i} (1) - Y _ {i} (0)\right) \right] \\ = \mathbb {E} \left[ Y _ {i} (0) \right] + \int_ {c (z)} ^ {1} \tau (u) f (u) d u, \\ \end{array}
$$

where the first equality is due to (10.7) and the exclusion restriction, the second is due to exogeneity, and the third is an application of Fubini’s theorem. Next, given that $c ( z )$ is continuously differentiable at z, we can use the chain rule to check that

$$
\frac {d}{d z} \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = z \right] = - \tau (c (z)) f (c (z)) c ^ {\prime} (z). \tag {10.15}
$$

Finally, applying the same calculation to the denominator yields (10.14).

Once we have access to the marginal treatment effect, we can use it to build estimators for weighted averages of E $[ \gamma ( u ) \tau ( u ) ]$ , provided the weights $\gamma ( u )$ only take positive values at points $u = c ( z )$ at which $c ( z )$ is continuous. Heckman and Vytlacil [2005] consider a variety of estimands of this type.

Example 10. Carneiro, Heckman, and Vytlacil [2011] use the local IV method to estimate returns to college attendance. The authors use data from the 1979 cohort from the National Longitudinal Survey of Youth (consisting of people born between 1957 and 1964), set their outcome variable $Y _ { i }$ to be log-income in 1991, and set their treatment variable $W _ { i }$ to be ever-enrollment in college by 1991. They identify marginal treatment effects via instruments $Z _ { i }$ that shift the desirability of attending college, including the presence of a nearby college, tuition at nearby colleges, and local employment conditions at the time when people turn 17. Their main finding is that, using our notation, $\tau ( u )$ is increasing in u, and that people who are more likely to attend college in the face of adverse nudges (i.e., abstractly, with a higher willingness to pay for college) in fact benefit more from college. Their results thus suggest that peoples’ choices under the model (10.7) can at least directionally be rationalized via private forecasts of future income benefits from college attendance.

## 10.3 Bibliographic notes

The idea of interpreting the results of instrumental variables analyses in terms of the local average treatment effect goes back to Imbens and Angrist [1994]. Our presentation of the analysis of clinical trials under non-compliance follows Angrist, Imbens, and Rubin [1996]. We refer to Imbens [2014] for a review.

Latent choice models, where people make choices if their (private) value from making that choice exceeds the cost, have a long tradition in economics. In an early example, Roy [1951] considered a model where workers pick a profession by considering their skills at different jobs and then choose the profession that enables them to maximize their wages—and used it to argue that, if worker skills are correlated across professions but productivity is more responsive to skill in some professions than in others, then we should expect higher average wages in professions with higher returns to skills. It has long been understood that such models cannot be fit via standard linear regression; however, in the early literature, such models were often approached via ad-hoc econometric strategies rather than IV methods. For example, Heckman [1979] considered a parametric latent choice model, and achieved identification via joint normality of latent variable $U _ { i }$ and potential outcomes (as opposed to using an auxiliary source of exogenous variation).

More recently, Heckman and Vytlacil [2005] have advocated for latent choice models as a natural framework for understanding instrumental variables methods, and have studied methods that target a wide variety estimands beyond the LATE that may be more helpful in setting policy. The identification result (10.14) for the marginal treatment effect via the local IV construction is due to Heckman and Vytlacil [1999]. Kennedy, Lorch, and Small [2019] studies semiparametrically efficient estimation of functions of the marginal treatment effect. The goal of estimating average treatment effects over subpopulations defined by conditioning on unobservables also arises in the literature on principal stratification developed in biostatistics [Frangakis and Rubin, 2002]. Our presentation of the local average treatment effect under supply-demand equilibrium is adapted from Angrist, Graddy, and Imbens [2000].
# Probability of Causation: Interpretation and Identification

Come and let us cast lots to find out who is to blame for this ordeal.  
—Jonah 1:7

## Preface

Assessing the likelihood that one event was the cause of another guides much of what we understand about (and how we act in) the world. For example, according to common judicial standard, judgment in favor of the plaintiff should be made if and only if it is “more probable than not” that the defendant’s action was the cause of the plaintiff’s damage (or death). But causation has two faces, necessary and sufficient; which of the two have lawmakers meant us to consider? And how are we to evaluate their probabilities?

This chapter provides formal semantics for the probability that event $x$ was a necessary or sufficient cause (or both) of another event $y$ . We then explicate conditions under which the probability of necessary (or sufficient) causation can be learned from statistical data, and we show how data from both experimental and nonexperimental studies can be combined to yield information that neither kind of study alone can provide.

## 9.1 Introduction

The standard counterfactual definition of causation (i.e., that $E$ would not have occurred were it not for $C$ ) captures the notion of “necessary cause.” Competing notions such as “sufficient cause” and “necessary and sufficient cause” are of interest in a number of applications, and these too can be given concise mathematical definitions in structural model semantics (Section 7.1). Although the distinction between necessary and sufficient causes goes back to J. S. Mill (1843), it has received semiformal explications only in the 1960s—via conditional probabilities (Good 1961) and logical implications (Mackie 1965; Rothman 1976). These explications suffer from basic semantical difficulties,¹ and they do not yield procedures for computing probabilities of causes as those provided by the structural account (Sections 7.1.3 and 8.3).

In this chapter we explore the counterfactual interpretation of necessary and sufficient causes, illustrate the application of structural model semantics to the problem of identifying probabilities of causes, and present, by way of examples, new ways of estimating probabilities of causes from statistical data. Additionally, we argue that necessity and sufficiency are two distinct facets of causation and that both facets should take part in the construction of causal explanations.

Our results have applications in epidemiology, legal reasoning, artificial intelligence (AI), and psychology. Epidemiologists have long been concerned with estimating the probability that a certain case of disease is “attributable” to a particular exposure, which is normally interpreted counterfactually as “the probability that disease would not have occurred in the absence of exposure, given that disease and exposure did in fact occur.” This counterfactual notion, which Robins and Greenland (1989) called the “probability of causation,” measures how necessary the cause is for the production of the effect.² It is used frequently in lawsuits, where legal responsibility is at the center of contention (see, e.g., Section 8.3). We shall denote this notion by the symbol PN, an acronym for probability of necessity.

A parallel notion of causation, capturing how sufficient a cause is for the production of the effect, finds applications in policy analysis, AI, and psychology. A policy maker may well be interested in the dangers that a certain exposure may present to the healthy population (Khoury et al. 1989). Counterfactually, this notion can be expressed as the “probability that a healthy unexposed individual would have contracted the disease had he or she been exposed,” and it will be denoted by PS (probability of sufficiency). A natural extension would be to inquire for the probability of necessary and sufficient causation (PNS)—that is, how likely a given individual is to be affected both ways.

As the examples illustrate, PS assesses the presence of an active causal process capable of producing the effect, while PN emphasizes the absence of alternative processes—not involving the cause in question—that are still capable of explaining the effect. In legal settings, where the occurrence of the cause ( $x$ ) and the effect ( $y$ ) are fairly well established, PN is the measure that draws most attention, and the plaintiff must prove that $y$ would not have occurred but for $x$ (Robertson 1997). Still, lack of sufficiency may weaken arguments based on PN (Good 1993; Michie 1999).

It is known that PN is in general nonidentifiable, that is, it cannot be estimated from frequency data involving exposures and disease cases (Greenland and Robins 1988; Robins and Greenland 1989). The identification is hindered by two factors:

1.  **Confounding** – Exposed and unexposed subjects may differ in several relevant factors or, more generally, the cause and the effect may both be influenced by a third factor. In this case we say that the cause is not exogenous relative to the effect (see Section 7.4.5).

2.  **Sensitivity to the generative process** – Even in the absence of confounding, probabilities of certain counterfactual relationships cannot be identified from frequency information unless we specify the functional relationships that connect causes and effects. Functional specification is needed whenever the facts at hand (e.g., disease) might be affected by the counterfactual antecedent (e.g., exposure) (see the examples in Sections 1.4, 7.5, and 8.3).

Although PN is not identifiable in the general case, several formulas have nevertheless been proposed to estimate attributions of various kinds in terms of frequencies obtained in epidemiological studies (Breslow and Day 1980; Hennekens and Buring 1987; Cole 1997). Naturally, any such formula must be predicated upon certain implicit assumptions about the data-generating process. Section 9.2 explicates some of those assumptions and explores conditions under which they can be relaxed.³ It offers new formulas for PN and PS in cases where causes are confounded (with outcomes) but their effects can nevertheless be estimated (e.g., from clinical trials or from auxiliary measurements). Section 9.3 exemplifies the use of these formulas in legal and epidemiological settings, while Section 9.4 provides a general condition for the identifiability of PN and PS when functional relationships are only partially known.

The distinction between necessary and sufficient causes has important implications in AI, especially in systems that generate verbal explanations automatically (see Section 7.2.3). As can be seen from the epidemiological examples, necessary causation is a concept tailored to a specific event under consideration (singular causation), whereas sufficient causation is based on the general tendency of certain event types to produce other event types. Adequate explanations should respect both aspects. If we base explanations solely on generic tendencies (i.e., sufficient causation) then we lose important specific information. For instance, aiming a gun at and shooting a person from 1,000 meters away will not qualify as an explanation for that person’s death, owing to the very low tendency of shots fired from such long distances to hit their marks. This stands contrary to common sense, for when the shot does hit its mark on that singular day, regardless of the reason, the shooter is an obvious culprit for the consequence. If, on the other hand, we base explanations solely on singular-event considerations (i.e., necessary causation), then various background factors that are normally present in the world would awkwardly qualify as explanations. For example, the presence of oxygen in the room would qualify as an explanation for the fire that broke out, simply because the fire would not have occurred were it not for the oxygen. That we judge the match struck, not the oxygen, to be the actual cause of the fire indicates that we go beyond the singular event at hand (where each factor alone is both necessary and sufficient) and consider situations of the same general type—where oxygen alone is obviously insufficient to start a fire. Clearly, some balance must be struck between the necessary and the sufficient components of causal explanation, and the present chapter illuminates this balance by formally explicating the basic relationships between these two components.

## 9.2 Necessary and Sufficient Causes: Conditions of Identification

### 9.2.1 Definitions, Notation, and Basic Relationships

Using the counterfactual notation and the structural model semantics introduced in Section 7.1, we give the following definitions for the three aspects of causation discussed in the introduction.

> ¹ These explications suffer from basic semantical difficulties.

> ² This counterfactual notion, which Robins and Greenland (1989) called the “probability of causation,” measures how necessary the cause is for the production of the effect.

> ³ Section 9.2 explicates some of those assumptions and explores conditions under which they can be relaxed.

**Definition 9.2.1 (Probability of Necessity, PN)**

Let $X$ and $Y$ be two binary variables in a causal model $M$ . Let $x$ and $y$ stand (respectively) for the propositions $X = \text{true}$ and $Y = \text{true}$ , and let $x'$ and $y'$ denote their complements. The probability of necessity is defined as the expression

$$
\mathrm{PN} \triangleq P(Y_{x'} = \text{false} \mid X = \text{true}, Y = \text{true})
$$

$$
\triangleq P(y'_{x'} \mid x, y). \tag{9.1}
$$

In other words, PN stands for the probability of $y'_{x'}$ (that event $y

# Definition 9.2.2 (Probability of Sufficiency, PS)

$$
\mathrm{PS} \triangleq P (y_{x} \mid y^{\prime}, x^{\prime}). \tag{9.2}
$$

PS measures the capacity of $x$ to produce $y$ , and, since “production” implies a transition from the absence to the presence of $x$ and $y$ , we condition the probability $P(y_{x})$ on situations where $x$ and $y$ are both absent. Thus, mirroring the necessity of $x$ (as measured by PN), PS gives the probability that setting $x$ would produce $y$ in a situation where $x$ and $y$ are in fact absent.

# Definition 9.2.3 (Probability of Necessity and Sufficiency, PNS)

$$
\mathrm{PNS} \triangleq P (y_{x}, y_{x^{\prime}}^{\prime}). \tag{9.3}
$$

PNS stands for the probability that $y$ would respond to $x$ both ways, and therefore measures both the sufficiency and necessity of $x$ to produce $y$ .

Associated with these three basic notions are other counterfactual quantities that have attracted either practical or conceptual interest. We will mention two such quantities but will not dwell on their analyses, since these follow naturally from our treatment of PN, PS, and PNS.

# Definition 9.2.4 (Probability of Disablement, PD)

$$
\mathrm{PD} \triangleq P (y_{x^{\prime}}^{\prime} \mid y). \tag{9.4}
$$

PD measures the probability that $y$ would have been prevented if it were not for $x$ ; it is therefore of interest to policy makers who wish to assess the social effectiveness of various prevention programs (Fleiss 1981, pp. 75–76).

# Definition 9.2.5 (Probability of Enablement, PE)

$$
\mathrm{PE} \triangleq P (y_{x} \mid y^{\prime}).
$$

PE is similar to PS, save for the fact that we do not condition on $x^{\prime}$ . It is applicable, for example, when we wish to assess the danger of an exposure on the entire population of healthy individuals, including those who were already exposed.

Although none of these quantities is sufficient for determining the others, they are not entirely independent, as shown in the following lemma.

# Lemma 9.2.6

The probabilities of causation (PNS, PN, and PS) satisfy the following relationship:

$$
\mathrm{PNS} = P (x, y) \mathrm{PN} + P (x^{\prime}, y^{\prime}) \mathrm{PS}. \tag{9.5}
$$

## Proof

The consistency condition of (7.20), $X = x \Longrightarrow Y_{x} = Y$ translates in our notation into:

$$
x \Longrightarrow (y_{x} = y), \quad x^{\prime} \Longrightarrow (y_{x^{\prime}} = y).
$$

Hence we can write:

$$
\begin{array}{l} y_{x} \wedge y_{x^{\prime}}^{\prime} = (y_{x} \wedge y_{x^{\prime}}^{\prime}) \wedge (x \vee x^{\prime}) \\ = (y \wedge x \wedge y_{x^{\prime}}^{\prime}) \vee (y_{x} \wedge y^{\prime} \wedge x^{\prime}). \\ \end{array}
$$

Taking probabilities on both sides and using the disjointness of $x$ and $x^{\prime}$ , we obtain:

$$
\begin{array}{l} P (y_{x}, y_{x^{\prime}}^{\prime}) = P (y_{x^{\prime}}^{\prime}, x, y) + P (y_{x}, x^{\prime}, y^{\prime}) \\ = P \left(y_{x^{\prime}}^{\prime} \mid x, y\right) P (x, y) + P \left(y_{x} \mid x^{\prime}, y^{\prime}\right) P \left(x^{\prime}, y^{\prime}\right), \\ \end{array}
$$

which proves Lemma 9.2.6. □

To put into focus the aspects of causation captured by PN and PS, it is helpful to characterize those changes in the causal model that would leave each of the two measures invariant. The next two lemmas show that PN is insensitive to the introduction of potential inhibitors of $y$ , while PS is insensitive to the introduction of alternative causes of $y$ .

## Lemma 9.2.7

Let $\mathrm {P N} ( x , y )$ stand for the probability that $x$ is a necessary cause of $y$ . Let $z = y \wedge q$ be a consequence of $y$ that is potentially inhibited by $q^{\prime}$ . If $q \bot \bot \{X , Y , Y_{x^{ \prime}} \}$ , then:

$$
\mathrm{PN} (x, z) \triangleq P (z_{x^{\prime}}^{\prime} \mid x, z) = P (y_{x^{\prime}}^{\prime} \mid x, y) \triangleq \mathrm{PN} (x, y).
$$

Cascading the process $Y_{x} ( u )$ with the link $z = y \wedge q$ amounts to inhibiting the output of the process with probability $P ( q^{\prime} )$ . Lemma 9.2.7 asserts that, if $q$ is randomized, we can add such a link without affecting PN. The reason is clear; conditioning on $x$ and $z$ implies that, in the scenario considered, the added link was not inhibited by $q^{\prime}$ .

## Proof of Lemma 9.2.7

We have:

$$
\begin{array}{l} \mathrm{PN} (x, z) = P \left(z_{x^{\prime}}^{\prime} \mid x, z\right) = \frac {P \left(z_{x^{\prime}}^{\prime} , x , z\right)}{P (x , z)} \\ = \frac {P (z_{x^{\prime}}^{\prime} , x , z \mid q) P (q) + P (z_{x^{\prime}}^{\prime} , x , z \mid q^{\prime}) P (q^{\prime})}{P (z , x , q) + P (z , x , q^{\prime})}. \tag {9.6} \\ \end{array}
$$

Using $z = y \wedge q$ , it follows that:

$$
q \implies (z = y), \quad q \implies (z_{x^{\prime}}^{\prime} = y_{x^{\prime}}^{\prime}), \quad \text{and} \quad q^{\prime} \implies z^{\prime};
$$

therefore:

$$
\begin{array}{l} \mathrm{PN} (x, z) = \frac {P \left(y_{x^{\prime}}^{\prime} , x , y \mid q\right) P (q) + 0}{P (y , x , q) + 0} \\ = \frac {P (y_{x^{\prime}}^{\prime} , x , y)}{P (y , x)} = P (y_{x^{\prime}}^{\prime} \mid x, y) = \mathrm{PN} (x, y). \\ \end{array}
$$

## Lemma 9.2.8

Let $\mathrm {P S} ( x , y )$ stand for the probability that $x$ is a sufficient cause of $y$ , and let $z = y \vee r$ be a consequence of $y$ that may also be triggered by $r$ . If $r \bot \bot \{X , Y , Y_{x} \}$ , then:

$$
\operatorname{PS} (x, z) \triangleq P (z_{x} \mid x^{\prime}, z^{\prime}) = P (y_{x} \mid x^{\prime}, y^{\prime}) \triangleq \operatorname{PS} (x, y).
$$

Lemma 9.2.8 asserts that we can add alternative (independent) causes $( r )$ without affecting PS. The reason again is clear; conditioning on the event $x^{\prime}$ and $y^{\prime}$ implies that the added causes $( r )$ were not active. The proof of Lemma 9.2.8 is similar to that of Lemma 9.2.7.

Since all the causal measures defined so far invoke conditionalization on $y$ , and since $y$ is presumed to be affected by $x$ , we know that none of these quantities is identifiable from knowledge of the causal diagram $G(M)$ and the data $P ( \nu )$ alone, even under conditions of no-confounding. Moreover, none of these quantities determines the others in the general case. However, simple interrelationships and useful bounds can be derived for these quantities under the assumption of no-confounding, an assumption that we call **exogeneity** .

## 9.2.2 Bounds and Basic Relationships under Exogeneity

## Definition 9.2.9 (Exogeneity)

A variable X is said to be exogenous relative to Y in model M if and only if

$$
\left\{Y_{x}, Y_{x^{\prime}} \right\} \perp X. \tag {9.7}
$$

In other words, the way Y would potentially respond to conditions $x$ or $x^{\prime}$ is independent of the actual value of $X$ .

Equation (9.7) is a strong version of those used in Chapter 5 (equation (5.30)) and in Chapter 6 (equation (6.10)) in that it involves the joint variable $\{Y_{x} , Y_{x^{\prime}} \}$ . This definition was named “strong ignorability” in Rosenbaum and Rubin (1983), and it coincides with the classical error-based criterion for exogeneity (Christ 1966, p. 156; see Section 7.4.5) and with the back-door criterion of Definition 3.3.1. The weaker definition of (5.30) is sufficient for all the results in this chapter except equations (9.11), (9.12), and (9.19), for which strong exogeneity (9.7) is needed.

The importance of exogeneity lies in permitting the identification of $\{P ( y_{x} ) , P ( y_{x^{\prime}} ) \}$ , the causal effect of X on Y, since (using $x \implies ( y_{x} = y )$ )

$$
P (y_{x}) = P (y_{x} \mid x) = P (y \mid x), \tag {9.8}
$$

with similar reduction for $P ( y_{x^{\prime}} )$ .

## Theorem 9.2.10

Under condition of exogeneity, PNS is bounded as follows:

$$
\max [ 0, P (y \mid x) - P (y \mid x^{\prime}) ] \leq \text {PNS} \leq \min [ P (y \mid x), P (y^{\prime} \mid x^{\prime}) ]. \tag {9.9}
$$

Both bounds are sharp in the sense that, for every joint distribution $P ( x , y )$ , there exists a model $y = f ( x , u )$ , with $u$ independent of $x$ , that realizes any value of PNS permitted by the bounds.

## Proof

For any two events $A$ and $B$ , we have the sharp bounds

$$
\max [ 0, P (A) + P (B) - 1 ] \leq P (A, B) \leq \min [ P (A), P (B) ]. \tag {9.10}
$$

Equation (9.9) follows from (9.3) and (9.10) using $A = y_{x}$ , $B = y_{x^{\prime}}^{\prime}$ , $P ( y_{x} ) = P ( y \mid x )$ , and $P ( y_{x^{\prime}}^{\prime} ) = P ( y^{\prime} \mid x^{\prime} )$ . □

Clearly, if exogeneity cannot be ascertained, then PNS is bound by inequalities similar to those of (9.9), with $P ( y_{x} )$ and $P ( y_{x^{\prime}}^{\prime} )$ replacing $P ( y \mid x )$ and $P ( y^{\prime} \mid x^{\prime} )$ , respectively.

## Theorem 9.2.11

Under condition of exogeneity, the probabilities PN, PS, and PNS are related to each other as follows:

$$
\mathrm{PN} = \frac {\mathrm{PNS}}{P (y \mid x)}, \tag {9.11}
$$

$$
\mathrm{PS} = \frac {\mathrm{PNS}}{P (y^{\prime} \mid x^{\prime})}. \tag {9.12}
$$

Thus, the bounds for PNS in (9.9) provide corresponding bounds for PN and PS.

The resulting bounds for PN,

$$
\frac {\max [ 0 , P (y \mid x) - P (y \mid x^{\prime}) ]}{P (y \mid x)} \leq \mathrm{PN} \leq \frac {\min [ P (y \mid x) , P (y^{\prime} \mid x^{\prime}) ]}{P (y \mid x)}, \tag {9.13}
$$

place limits on our ability to identify PN in experimental studies, where exogeneity holds.

## Corollary 9.2.12

If $x$ and $y$ occur in an experimental study and if $P ( y_{x} )$ and $P ( y_{x^{\prime}} )$ are the causal effects measured in that study, then for any point $p$ in the range

$$
\frac {\max \left[ 0 , P (y_{x}) - P (y_{x^{\prime}}) \right]}{P (y_{x})} \leq p \leq \frac {\min [ P (y_{x}) , P (y_{x^{\prime}}^{\prime}) ]}{P (y_{x})} \tag {9.14}
$$

there exists a causal model $M$ that agrees with $P ( y_{x} )$ and $P ( y_{x^{\prime}} )$ and for which $\mathrm {P N} = p$ .

Other bounds can be established for nonexperimental events if we have data from both experimental and observational studies (as in Section 9.3.4). The nonzero widths of these bounds imply that probabilities of causation cannot be defined uniquely in stochastic (non-Laplacian) models where, for each $u$ , $Y_{x} ( u )$ is specified in probability $P ( Y_{x} ( u ) = y )$ instead of a single number.⁶

## Proof of Theorem 9.2.11

Using $x \implies ( y_{x} = y )$ we can write $x \wedge y_{x} = x \wedge y$ and so obtain

$$
\mathrm{PN} = P (y_{x^{\prime}}^{\prime} \mid x, y) = \frac {P (y_{x^{\prime}}^{\prime} , x , y)}{P (x , y)} \tag {9.15}
$$

$$
= \frac {P (y_{x^{\prime}}^{\prime} , x , y_{x})}{P (x , y)} \tag {9.16}
$$

$$
= \frac {P (y_{x^{\prime}}^{\prime} , y_{x}) P (x)}{P (x , y)} \tag {9.17}
$$

$$
= \frac {\mathrm{PNS}}{P (y \mid x)}, \tag {9.18}
$$

which establishes (9.11). Equation (9.12) follows by identical steps.

![image_113](images/image_113.png)

For completeness, we write the relationship between PNS and the probabilities of enablement and disablement:

$$
\mathrm{PD} = \frac {P (x) \mathrm{PNS}}{P (y)}, \quad \mathrm{PE} = \frac {P (x^{\prime}) \mathrm{PNS}}{P (y^{\prime})}. \tag {9.19}
$$

# 9.2.3 Identifiability under Monotonicity and Exogeneity

Before attacking the general problem of identifying the counterfactual quantities in (9.1)–(9.3), it is instructive to treat a special condition, called **monotonicity** , which is often assumed in practice and which renders these quantities identifiable. The resulting probabilistic expressions will be recognized as familiar measures of causation that often appear in the literature.

## Definition 9.2.13 (Monotonicity)

A variable $Y$ is said to be **monotonic** relative to variable $X$ in a causal model $M$ if and only if the function $Y_x(u)$ is monotonic in $x$ for all $u$ . Equivalently, $Y$ is monotonic relative to $X$ if and only if

$$
y_x \land y_{x'} = \text{false}. \tag{9.20}
$$

Monotonicity expresses the assumption that a change from $X = \text{false}$ to $X = \text{true}$ cannot, under any circumstance, make $Y$ change from true to false.¹ In epidemiology, this assumption is often expressed as “no prevention,” that is, no individual in the population can be helped by exposure to the risk factor.

> ¹ That is, no individual is in the “helped” category.

## Theorem 9.2.14 (Identifiability under Exogeneity and Monotonicity)

If $X$ is exogenous and $Y$ is monotonic relative to $X$ , then the probabilities **PN** , **PS** , and **PNS** are all identifiable and are given by (9.11)–(9.12), with

$$
\mathrm{PNS} = P(y \mid x) - P(y \mid x'). \tag{9.21}
$$

The r.h.s. of (9.21) is called “risk difference” in epidemiology, and is also misnomered “attributable risk” (Hennekens and Buring 1987, p. 87).

From (9.11) we see that the probability of necessity is identifiable and given by the excess risk ratio

$$
\mathrm{PN} = \frac{P(y \mid x) - P(y \mid x')}{P(y \mid x)}, \tag{9.22}
$$

often misnomered as the “attributable fraction” (Schlesselman 1982), “attributable-rate percent” (Hennekens and Buring 1987, p. 88), or “attributable proportion” (Cole 1997).

Taken literally, the ratio presented in (9.22) has nothing to do with attribution, since it is made up of statistical terms and not of causal or counterfactual relationships. However, the assumptions of exogeneity and monotonicity together enable us to translate the notion of attribution embedded in the definition of **PN** (equation (9.1)) into a ratio of purely statistical associations. This suggests that exogeneity and monotonicity were tacitly assumed by the many authors who proposed or derived (9.22) as a measure for the “fraction of exposed cases that are attributable to the exposure.”

Robins and Greenland (1989) analyzed the identification of **PN** under the assumption of stochastic monotonicity (i.e., $P(Y_x(u) = y) > P(Y_{x'}(u) = y)$ ) and showed that this assumption is too weak to permit such identification; in fact, it yields the same bounds as in (9.13). This indicates that stochastic monotonicity imposes no constraints whatsoever on the functional mechanisms that mediate between $X$ and $Y$ .

The expression for **PS** (equation (9.12)) is likewise quite revealing,

$$
\mathrm{PS} = \frac{P(y \mid x) - P(y \mid x')}{1 - P(y \mid x')}, \tag{9.23}
$$

since it coincides with what epidemiologists call the “relative difference” (Shep 1958), which is used to measure the susceptibility of a population to an exposure $x$ . Susceptibility is defined as the proportion of persons who possess “an underlying factor sufficient to make a person contract a disease following exposure” (Khoury et al. 1989). **PS** offers a formal counterfactual interpretation of susceptibility, which sharpens this definition and renders susceptibility amenable to systematic analysis.

Khoury et al. (1989) recognized that susceptibility in general is not identifiable and derived (9.23) by making three assumptions: no-confounding, monotonicity,² and independence (i.e., assuming that susceptibility to exposure is independent of susceptibility to background not involving exposure). This last assumption is often criticized as untenable, and Theorem 9.2.14 assures us that independence is in fact unnecessary; (9.23) attains its validity through exogeneity and monotonicity alone.

> ² Monotonicity is sometimes called “no prevention” in this context.

Equation (9.23) also coincides with what Cheng (1997) calls “causal power,” namely, the effect of $x$ on $y$ after suppressing “all other causes of $y$ .” The counterfactual definition of **PS** , $P(y_x \mid x', y')$ , suggests another interpretation of this quantity. It measures the probability that setting $x$ would produce $y$ in a situation where $x$ and $y$ are in fact absent. Conditioning on $y'$ amounts to selecting (or hypothesizing) only those worlds in which “all other causes of $y$ ” are indeed suppressed.

It is important to note, however, that the simple relationships among the three notions of causation (equations (9.11)–(9.12)) hold only under the assumption of exogeneity; the weaker relationship of (9.5) prevails in the general, nonexogenous case. Additionally, all these notions of causation are defined in terms of the global relationships $Y_x(u)$ and $Y_{x'}(u)$ , which are too crude to fully characterize the many nuances of causation; the detailed structure of the causal model leading from $X$ to $Y$ is often needed to explicate more refined notions, such as “actual cause” (see Chapter 10).

## Proof of Theorem 9.2.14

Writing $y_{x'} \lor y_{x'}' = \text{true}$ , we have

$$
y_x = y_x \land (y_{x'} \lor y_{x'}') = (y_x \land y_{x'}) \lor (y_x \land y_{x'}') \tag{9.24}
$$

and

$$
y_{x'} = y_{x'} \land (y_x \lor y_x') = (y_{x'} \land y_x) \lor (y_{x'} \land y_x') = y_{x'} \land y_x, \tag{9.25}
$$

since monotonicity entails $y_{x'} \land y_x' = \text{false}$ . Substituting (9.25) into (9.24) yields

$$
y_x = y_{x'} \lor (y_x \land y_{x'}'). \tag{9.26}
$$

Taking the probability of (9.26) and using the disjointness of $y_{x'}$ and $y_{x'}'$ , we obtain

$$
P(y_x) = P(y_{x'}) + P(y_x, y_{x'}')
$$

or

$$
P(y_x, y_{x'}') = P(y_x) - P(y_{x'}). \tag{9.27}
$$

Equation (9.27), together with the assumption of exogeneity (equation (9.8)), establishes equation (9.21). $\square$

## 9.2.4 Identifiability under Monotonicity and Nonexogeneity

The relations established in Theorems 9.2.10–9.2.14 were based on the assumption of exogeneity. In this section, we relax this assumption and consider cases where the effect of $X$ on $Y$ is confounded, that is, when $P(y_x) \neq P(y \mid x)$ . In such cases, $P(y_x)$ may still be estimated by auxiliary means (e.g., through adjustment of certain covariates or through experimental studies), and the question is whether this added information can render the probability of causation identifiable. The answer is affirmative.

## Theorem 9.2.15

If Y is monotonic relative to X, then PNS, PN, and PS are identifiable whenever the causal effects $P(y_x)$ and $P(y_{x^\prime})$ are identifiable:

$$
\mathrm{PNS} = P(y_x, y_{x^\prime}^\prime) = P(y_x) - P(y_{x^\prime}), \tag{9.28}
$$

$$
\mathrm{PN} = P(y_{x^\prime}^\prime \mid x, y) = \frac{P(y) - P(y_{x^\prime})}{P(x, y)}, \tag{9.29}
$$

$$
\mathrm{PS} = P(y_x \mid x^\prime, y^\prime) = \frac{P(y_x) - P(y)}{P(x^\prime, y^\prime)}. \tag{9.30}
$$

In order to appreciate the difference between equations (9.29) and (9.22), we can expand $P(y)$ and write

$$
\begin{array}{l}
\mathrm{PN} = \frac{P(y \mid x) P(x) + P(y \mid x^\prime) P(x^\prime) - P(y_{x^\prime})}{P(y \mid x) P(x)} \\
= \frac{P(y \mid x) - P(y \mid x^\prime)}{P(y \mid x)} + \frac{P(y \mid x^\prime) - P(y_{x^\prime})}{P(x, y)}. \tag{9.31}
\end{array}
$$

The first term on the r.h.s. of (9.31) is the familiar excess risk ratio (as in (9.22)) and represents the value of PN under exogeneity. The second term represents the correction needed to account for confounding, that is, $P(y_{x^\prime}) \neq P(y \mid x^\prime)$ . Equations (9.28)–(9.30) thus provide more refined measures of causation, which can be used in situations where the causal effect $P(y_x)$ can be identified through auxiliary means (see Example 4, Section 9.3.4). It can also be shown that expressions in (9.28)–(9.30) provide lower bounds for PNS, PN, and PS in the general, nonmonotonic case (Tian and Pearl 2000, Section 11.9.2).

Remarkably, since PS and PN must be nonnegative, (9.29)–(9.30) provide a simple necessary test for the assumption of monotonicity:

$$
P(y_x) \geq P(y) \geq P(y_{x^\prime}), \tag{9.32}
$$

which tightens the standard inequalities (from $x^\prime \wedge y \implies y_{x^\prime}$ and $x \wedge y^\prime \implies y_{x^\prime}$ )

$$
P(y_{x^\prime}) \geq P(x^\prime, y), \quad P(y_x^\prime) \geq P(x, y^\prime). \tag{9.33}
$$

J. Tian has shown that these inequalities are in fact sharp: every combination of experimental and nonexperimental data that satisfies these inequalities can be generated from some causal model in which Y is monotonic in X. That the commonly made assumption of “no prevention” is not entirely exempt from empirical scrutiny should come as a relief to many epidemiologists. Alternatively, if the no-prevention assumption is theoretically unassailable, then (9.32) can be used for testing the compatibility of the experimental and nonexperimental data, that is, whether subjects used in clinical trials are representative of the target population as characterized by the joint distribution $P(x, y)$ .

## Proof of Theorem 9.2.15

Equation (9.28) was established in (9.27). To prove (9.30), we write

$$
P(y_x \mid x^\prime, y^\prime) = \frac{P(y_x, x^\prime, y^\prime)}{P(x^\prime, y^\prime)} = \frac{P(y_x, x^\prime, y_{x^\prime}^\prime)}{P(x^\prime, y^\prime)}, \tag{9.34}
$$

because $x^\prime \wedge y^\prime = x^\prime \wedge y_{x^\prime}^\prime$ (by consistency). To calculate the numerator of (9.34), we conjoin (9.26) with $x^\prime$ to obtain

$$
x^\prime \wedge y_x = (x^\prime \wedge y_{x^\prime}) \vee (y_x \wedge y_{x^\prime}^\prime \wedge x^\prime).
$$

We then take the probability on both sides, which gives (since $y_{x^\prime}$ and $y_{x^\prime}^\prime$ are disjoint)

$$
\begin{array}{l}
P(y_x, y_{x^\prime}^\prime, x^\prime) = P(x^\prime, y_x) - P(x^\prime, y_{x^\prime}) \\
= P(x^\prime, y_x) - P(x^\prime, y) \\
= P(y_x) - P(x, y_x) - P(x^\prime, y) \\
= P(y_x) - P(x, y) - P(x^\prime, y) \\
= P(y_x) - P(y).
\end{array}
$$

Substituting into (9.34), we finally obtain

$$
P(y_x \mid x^\prime, y^\prime) = \frac{P(y_x) - P(y)}{P(x^\prime, y^\prime)},
$$

which establishes (9.30). Equation (9.29) follows via identical steps.

![image_114](images/image_114.png)

One common class of models that permits the identification of $P(y_x)$ under conditions of nonexogeneity was exemplified in Chapter 3. It was shown in Section 3.2 (equation (3.13)) that, for every two variables X and Y in a positive Markovian model M, the causal effect $P(y_x)$ is identifiable and is given by

$$
P(y_x) = \sum_{pa_X} P(y \mid pa_X, x) P(pa_X), \tag{9.35}
$$

where $pa_X$ are (realizations of) the parents of X in the causal graph associated with M. Thus, we can combine (9.35) with Theorem 9.2.15 to obtain a concrete condition for the identification of the probability of causation.

## Corollary 9.2.16

For any positive Markovian model M, if the function $Y_x(u)$ is monotonic then the probabilities of causation PNS, PS, and PN are identifiable and are given by (9.28)–(9.30), with $P(y_x)$ as given in (9.35).

A broader identification condition can be obtained through the use of the back-door and front-door criteria (Section 3.3), which are applicable to semi-Markovian models. These were further generalized in Galles and Pearl (1995) (see Section 4.3.1) and Tian and Pearl (2002a) (Theorem 3.6.1) and lead to the following corollary.

## Corollary 9.2.17

Let GP be the class of semi-Markovian models that satisfy the graphical criterion of Theorem 3.6.1. If $Y_x(u)$ is monotonic, then the probabilities of causation PNS, PS, and PN are identifiable in GP and are given by (9.28)–(9.30), with $P(y_x)$ determined by the topology of $G(M)$ through the algorithm of Tian and Pearl (2002a).

## 9.3 EXAMPLES AND APPLICATIONS

## 9.3.1 Example 1: Betting against a Fair Coin

We must bet heads or tails on the outcome of a fair coin toss; we win a dollar if we guess correctly and lose if we don’t. Suppose we bet heads and win a dollar, without glancing at the actual outcome of the coin. Was our bet a necessary cause (or a sufficient cause, or both) for winning?

This example is isomorphic to the clinical trial discussed in Section 1.4.4 (Figure 1.6). Let $x$ stand for “we bet on heads,” $y$ for “we win a dollar,” and $u$ for “the coin turned up heads.” The functional relationship between $y$ , $x$ , and $u$ is

$$
y = (x \wedge u) \vee (x' \wedge u'), \tag{9.36}
$$

which is not monotonic but, since the model is fully specified, permits us to compute the probabilities of causation from their definitions, (9.1)–(9.3). To exemplify,

$$
\mathrm{PN} = P(y'_{x'} \mid x, y) = P(y'_{x'} \mid u) = 1,
$$

because $x \wedge y \implies u$ and $Y_{x'}(u) = \mathrm{false}$ . In words, knowing the current bet ( $x$ ) and current win ( $y$ ) permits us to infer that the coin outcome must have been a head ( $u$ ), from which we can further deduce that betting tails ( $x'$ ) instead of heads would have resulted in a loss. Similarly,

$$
\mathrm{PS} = P(y_x \mid x', y') = P(y_x \mid u) = 1
$$

(because $x \wedge y' \implies u$ ) and

$$
\begin{array}{l}
\mathrm{PNS} = P(y_x, y'_{x'}) \\
= P(y_x, y'_{x'} \mid u) P(u) + P(y_x, y'_{x'} \mid u') P(u') \\
= 1 (0.5) + 0 (0.5) = 0.5.
\end{array}
$$

We see that betting heads has 50% chance of being a necessary and sufficient cause of winning. Still, once we win, we can be 100% sure that our bet was necessary for our win, and once we lose (say, on betting tails) we can be 100% sure that betting heads would have been sufficient for producing a win. The empirical content of such counterfactuals is discussed in Section 7.2.2.

It is easy to verify that these counterfactual quantities cannot be computed from the joint probability of $X$ and $Y$ without knowledge of the functional relationship in (9.36), which tells us the (deterministic) policy by which a win or a loss is decided (Section 1.4.4). This can be seen, for instance, from the conditional probabilities and causal effects associated with this example,

$$
P(y \mid x) = P(y \mid x') = P(y_x) = P(y_{x'}) = P(y) = \frac{1}{2},
$$

because identical probabilities would be generated by a random payoff policy in which $y$ is functionally independent of $x$ – say, by a bookie who watches the coin and ignores our bet. In such a random policy, the probabilities of causation PN, PS, and PNS are all zero. Thus, according to our definition of identifiability (Definition 3.2.3), if two models agree on $P$ and do not agree on a quantity $Q$ , then $Q$ is not identifiable. Indeed, the bounds delineated in Theorem 9.2.10 (equation (9.9)) read $0 \leq \mathrm{PNS} \leq \frac{1}{2}$ , meaning that the three probabilities of causation cannot be determined from statistical data on $X$ and $Y$ alone, not even in a controlled experiment; knowledge of the functional mechanism is required, as in (9.36).

It is interesting to note that whether the coin is tossed before or after the bet has no bearing on the probabilities of causation as just defined. This stands in contrast with some theories of probabilistic causality (e.g., Good 1961), which attempt to avoid deterministic mechanisms by conditioning all probabilities on “the state of the world just before” the occurrence of the cause in question ( $x$ ). When applied to our betting story, the intention is to condition all probabilities on the state of the coin ( $u$ ), but this is not fulfilled if the coin is tossed after the bet is placed. Attempts to enrich the conditioning set with events occurring after the cause in question have led back to deterministic relationships involving counterfactual variables (see Cartwright 1989, Eells 1991, and the discussion in Section 7.5.4).

One may argue, of course, that if the coin is tossed after the bet then it is not at all clear what our winnings would be had we bet differently; merely uttering our bet could conceivably affect the trajectory of the coin (Dawid 2000). This objection can be diffused by placing $x$ and $u$ in two remote locations and tossing the coin a split second after the bet is placed but before any light ray could arrive from the betting room to the coin-tossing room. In such a hypothetical situation, the counterfactual statement “our winning would be different had we bet differently” is rather compelling, even though the conditioning event ( $u$ ) occurs after the cause in question ( $x$ ). We conclude that temporal descriptions such as “the state of the world just before $x$ ” cannot be used to properly identify the appropriate set of conditioning events ( $u$ ) in a problem; a deterministic model of the mechanisms involved is needed for formulating the notion of “probability of causation.”

## 9.3.2 Example 2: The Firing Squad

Consider again the firing squad of Section 7.1.2 (see Figure 9.1); A and B are riflemen, C is the squad’s captain (who is waiting for the court order, U), and T is a condemned prisoner. Let **u** be the proposition that the court has ordered an execution, **x** the proposition stating that A pulled the trigger, and **y** that T is dead.

We assume again that $P(u) = \frac{1}{2}$ , that A and B are perfectly accurate marksmen who are alert and law-abiding, and that T is not likely to die from fright or other extraneous causes. We wish to compute the probability that **x** was a necessary (or sufficient, or both) cause for **y** (i.e., we wish to calculate **PN** , **PS** , and **PNS** ).

![image_115](images/image_115.png)

> **y** : T dies  
>  **Figure 9.1** Causal relationships in the two-man firing-squad example.

Definitions 9.2.1–9.2.3 permit us to compute these probabilities directly from the given causal model, since all functions and all probabilities are specified, with the truth value of each variable tracing that of U. Accordingly, we can write:

$$
\begin{array}{l}
P(y_x) = P(Y_x(u) = \text{true}) P(u) + P(Y_x(u') = \text{true}) P(u') \\
= \frac{1}{2} (1 + 1) = 1. \tag{9.37}
\end{array}
$$

Similarly, we have:

$$
\begin{array}{l}
P(y_{x'}) = P(Y_{x'}(u) = \text{true}) P(u) + P(Y_{x'}(u') = \text{true}) P(u') \\
= \frac{1}{2} (1 + 0) = \frac{1}{2}. \tag{9.38}
\end{array}
$$

In order to compute **PNS** , we must evaluate the probability of the joint event $y'_{x'} \wedge y_x$ . Given that these two events are jointly true only when $U = \text{true}$ , we have:

$$
\begin{array}{l}
\mathrm{PNS} = P(y_x, y'_{x'}) \\
= P(y_x, y'_{x'} \mid u) P(u) + P(y_x, y'_{x'} \mid u') P(u') \\
= \frac{1}{2} (0 + 1) = \frac{1}{2}. \tag{9.39}
\end{array}
$$

The calculation of **PS** and **PN** is likewise simplified by the fact that each of the conditioning events, for **PN** and $x' \wedge y'$ for **PS** , is true in only one state of U. We thus have:

$$
\mathrm{PN} = P(y'_{x'} \mid x, y) = P(y'_{x'} \mid u) = 0.
$$

Reflecting that, once the court orders an execution (u), T will die (y) from the shot of rifleman B, even if A refrains from shooting $(x')$ . Indeed, upon learning of T’s death, we can categorically state that rifleman A’s shot was not a necessary cause of the death.

Similarly:

$$
\mathrm{PS} = P(y_x \mid x', y') = P(y_x \mid u') = 1,
$$

**Table 9.1**

|                    | Exposure |              |
| ------------------ | -------- | ------------ |
|                    | High (x) | Low ( $x'$ ) |
| Deaths (y)         | 30       | 16           |
| Survivals ( $y'$ ) | 69,130   | 59,010       |

Matching our intuition that a shot fired by an expert marksman would be sufficient for causing the death of T, regardless of the court decision.

Note that Theorems 9.2.10 and 9.2.11 are not applicable to this example because **x** is not exogenous; events **x** and **y** have a common cause (the captain’s signal), which renders $P(y \mid x') = 0 \neq P(y_{x'}) = \frac{1}{2}$ . However, the monotonicity of Y (in x) permits us to compute **PNS** , **PS** , and **PN** from the joint distribution $P(x, y)$ and the causal effects (using (9.28)–(9.30)), instead of consulting the functional model.

Indeed, writing:

$$
P(x, y) = P(x', y') = \frac{1}{2} \tag{9.40}
$$

and

$$
P(x, y') = P(x', y) = 0, \tag{9.41}
$$

we obtain:

$$
\mathrm{PN} = \frac{P(y) - P(y_{x'})}{P(x, y)} = \frac{\frac{1}{2} - \frac{1}{2}}{\frac{1}{2}} = 0 \tag{9.42}
$$

and

$$
\mathrm{PS} = \frac{P(y_x) - P(y)}{P(x', y')} = \frac{1 - \frac{1}{2}}{\frac{1}{2}} = 1, \tag{9.43}
$$

as expected.

# 9.3.3 Example 3: The Effect of Radiation on Leukemia

Consider the following data (Table 9.1, adapted¹⁰ from Finkelstein and Levin 1990) comparing leukemia deaths in children in southern Utah with high and low exposure to radiation from the fallout of nuclear tests in Nevada. Given these data, we wish to estimate the probabilities that high exposure to radiation was a necessary (or sufficient, or both) cause of death due to leukemia.

Assuming monotonicity – that exposure to nuclear radiation had no remedial effect on any individual in the study – the process can be modeled by a simple disjunctive mechanism represented by the equation

$$
y = f(x, u, q) = (x \land q) \lor u, \tag{9.44}
$$

where $u$ represents “all other causes” of $y$ and where $q$ represents all “enabling” mechanisms that must be present for $x$ to trigger $y$ . Assuming that $q$ and $u$ are both unobserved, the question we ask is under what conditions we can identify the probabilities of causation (PNS, PN, and PS) from the joint distribution of $X$ and $Y$ .

Since (9.44) is monotonic in $x$ , Theorem 9.2.14 states that all three quantities would be identifiable provided $X$ is exogenous; that is, $x$ should be independent of $q$ and $u$ . Under this assumption, (9.21)–(9.23) further permit us to compute the probabilities of causation from frequency data. Taking fractions to represent probabilities, the data in Table 9.1 imply the following numerical results:

$$
\mathrm{PNS} = P(y \mid x) - P(y \mid x^{\prime}) = \frac{30}{30 + 69,130} - \frac{16}{16 + 59,010} = 0.0001625, \tag{9.45}
$$

$$
\mathrm{PN} = \frac{\mathrm{PNS}}{P(y \mid x)} = \frac{\mathrm{PNS}}{30 / (30 + 69,130)} = 0.37535, \tag{9.46}
$$

$$
\mathrm{PS} = \frac{\mathrm{PNS}}{1 - P(y \mid x^{\prime})} = \frac{\mathrm{PNS}}{1 - 16 / (16 + 59,010)} = 0.0001625. \tag{9.47}
$$

Statistically, these figures mean that:

- 1. There is a 1.625 in ten thousand chance that a randomly chosen child would both die of leukemia if exposed and survive if not exposed;
- 2. There is a 37.544% chance that an exposed child who died from leukemia would have survived had he or she not been exposed;
- 3. There is a 1.625 in ten thousand chance that any unexposed surviving child would have died of leukemia had he or she been exposed.

Glymour (1998) analyzed this example with the aim of identifying the probability $P(q)$ (Cheng’s “causal power”), which coincides with PS (see Lemma 9.2.8). Glymour concluded that $P(q)$ is identifiable and is given by (9.23), provided that $x$ , $u$ , and $q$ are mutually independent. Our analysis shows that Glymour’s result can be generalized in several ways.

First, since $Y$ is monotonic in $X$ , the validity of (9.23) is assured even when $q$ and $u$ are dependent, because exogeneity merely requires independence between $x$ and $\{u, q\}$ jointly. This is important in epidemiological settings, because an individual’s susceptibility to nuclear radiation is likely to be associated with susceptibility to other potential causes of leukemia (e.g., natural kinds of radiation).

Second, Theorem 9.2.11 assures us that the relationships among PN, PS, and PNS (equations (9.11)–(9.12)), which Glymour derives for independent $q$ and $u$ , should remain valid even when $u$ and $q$ are dependent.

![image_116](images/image_116.png)

> Figure 9.2 Causal relationships in the radiation–leukemia example, where $W$ represents confounding factors.

Finally, Theorem 9.2.15 assures us that PN and PS are identifiable even when $x$ is not independent of $\{u, q\}$ , provided only that the mechanism of (9.44) is embedded in a larger causal structure that permits the identification of $P(y_x)$ and $P(y_{x^{\prime}})$ . For example, assume that exposure to nuclear radiation ( $x$ ) is suspected of being associated with terrain and altitude, which are also factors in determining exposure to cosmic radiation. A model reflecting such consideration is depicted in Figure 9.2, where $W$ represents factors affecting both $X$ and $U$ .

A natural way to correct for possible confounding bias in the causal effect of $X$ on $Y$ would be to adjust for $W$ , that is, to calculate $P(y_x)$ and $P(y_{x^{\prime}})$ using the standard adjustment formula (equation (3.19))

$$
P(y_x) = \sum_{w} P(y \mid x, w) P(w), \quad P(y_{x^{\prime}}) = \sum_{w} P(y \mid x^{\prime}, w) P(w) \tag{9.48}
$$

(instead of $P(y \mid x)$ and $P(y \mid x^{\prime})$ ), where the summation runs over levels of $W$ . This adjustment formula, which follows from (9.35), is correct regardless of the mechanisms mediating $X$ and $Y$ , provided only that $W$ represents all common factors affecting $X$ and $Y$ (see Section 3.3.1).

Theorem 9.2.15 instructs us to evaluate PN and PS by substituting (9.48) into (9.29) and (9.30), respectively, and it assures us that the resulting expressions constitute consistent estimates of PN and PS. This consistency is guaranteed jointly by the assumption of monotonicity and by the (assumed) topology of the causal graph.

Note that monotonicity as defined in (9.20) is a global property of all pathways between $x$ and $y$ . The causal model may include several nonmonotonic mechanisms along these pathways without affecting the validity of (9.20). However, arguments for the validity of monotonicity must be based on substantive information, since it is not testable in general. For example, Robins and Greenland (1989) argued that exposure to nuclear radiation may conceivably be of benefit to some individuals because such radiation is routinely used clinically in treating cancer patients. The inequalities in (9.32) constitute a statistical test of monotonicity (albeit a weak one) that is based on both experimental and observational studies.

**Table 9.2**

|                    | Experimental |      | Nonexperimental |      |
| ------------------ | ------------ | ---- | --------------- | ---- |
|                    | $x$          | $x'$ | $x$             | $x'$ |
| Deaths ( $y$ )     | 16           | 14   | 2               | 28   |
| Survivals ( $y'$ ) | 984          | 986  | 998             | 972  |

## 9.3.4 Example 4: Legal Responsibility from Experimental and Nonexperimental Data

A lawsuit is filed against the manufacturer of drug **x** , charging that the drug is likely to have caused the death of Mr. A, who took the drug to relieve symptom **S** associated with disease **D** . The manufacturer claims that experimental data on patients with symptom **S** show conclusively that drug **x** may cause only a minor increase in death rates. However, the plaintiff argues that the experimental study is of little relevance to this case because it represents the effect of the drug on _all_ patients, not on patients like Mr. A who actually died while using drug **x** .

Moreover, argues the plaintiff, Mr. A is unique in that he used the drug on his own volition, unlike subjects in the experimental study who took the drug to comply with experimental protocols. To support this argument, the plaintiff furnishes nonexperimental data indicating that most patients who chose drug **x** would have been alive were it not for the drug. The manufacturer counterargues by stating that:

1. Counterfactual speculations regarding whether patients would or would not have died are purely metaphysical and should be avoided (Dawid 2000);
2. Nonexperimental data should be dismissed _a priori_ on the grounds that such data may be highly confounded by extraneous factors.

The court must now decide, based on both the experimental and nonexperimental studies, what the probability is that drug **x** was in fact the cause of Mr. A’s death.

The (hypothetical) data associated with the two studies are shown in **Table 9.2** . The experimental data provide the estimates

$$
P(y_x) = 16 / 1000 = 0.016, \tag{9.49}
$$

$$
P(y_{x^{\prime}}) = 14 / 1000 = 0.014; \tag{9.50}
$$

the nonexperimental data provide the estimates

$$
P(y) = 30 / 2000 = 0.015, \tag{9.51}
$$

$$
P(y, x) = 2 / 2000 = 0.001. \tag{9.52}
$$

Substituting these estimates in (9.29), which provides a lower bound on **PN** (see (11.42)), we obtain

$$
\mathrm{PN} \geq \frac{P(y) - P(y_{x^{\prime}})}{P(y, x)} = \frac{0.015 - 0.014}{0.001} = 1.00. \tag{9.53}
$$

Thus, the plaintiff was correct; barring sampling errors, the data provide us with **100% assurance** that drug **x** was in fact responsible for the death of Mr. A. Note that a straightforward use of the experimental excess risk ratio would yield a much lower (and incorrect) result:

$$
\frac{P(y_x) - P(y_{x^{\prime}})}{P(y_x)} = \frac{0.016 - 0.014}{0.016} = 0.125. \tag{9.54}
$$

Evidently, what the experimental study does **not** reveal is that, given a choice, terminal patients avoid drug **x** . Indeed, if there were any terminal patients who would choose **x** (given the choice), then the control group $(x^{\prime})$ would have included some such patients (due to randomization) and so the proportion of deaths among the control group $P(y_{x^{\prime}})$ would have been higher than $P(x^{\prime}, y)$ , the population proportion of terminal patients avoiding **x** . However, the equality $P(y_{x^{\prime}}) = P(y, x^{\prime})$ tells us that no such patients were included in the control group; hence (by randomization) no such patients exist in the population at large, and therefore none of the patients who freely chose drug **x** was a terminal case; all were susceptible to **x** .

The numbers in **Table 9.2** were obviously contrived to represent an extreme case and so facilitate a qualitative explanation of the validity of (9.29). Nevertheless, it is instructive to note that a combination of experimental and nonexperimental studies may unravel what experimental studies alone will not reveal and, in addition, that such combination may provide a necessary test for the adequacy of the experimental procedures. For example, if the frequencies in **Table 9.2** were slightly different, they could easily yield a value greater than unity for **PN** in (9.53) or some other violation of the fundamental inequalities of (9.33). Such violation would indicate an incompatibility of the experimental and nonexperimental groups due, perhaps, to inadequate sampling.

This last point may warrant a word of explanation, lest the reader wonder why two data sets—taken from two separate groups under different experimental conditions—should constrain one another. The explanation is that certain quantities in the two subpopulations are expected to remain invariant to all these differences, provided that the two subpopulations were sampled properly from the population at large. These invariant quantities are simply the causal effects probabilities, $P(y_{x^{\prime}})$ and $P(y_x)$ . Although these counterfactual probabilities were not measured in the observational group, they must (by definition) nevertheless be the same as those measured in the experimental group.

The invariance of these quantities is the basic axiom of controlled experimentation, without which no inference would be possible from experimental studies to general behavior of the population. The invariance of these quantities implies the inequalities of (9.33) and, if monotonicity holds, (9.32) ensues.

## 9.3.5 Summary of Results

We now summarize the results from Sections 9.2 and 9.3 that should be of value to practicing epidemiologists and policy makers. These results are shown in Table 9.3, which lists the best estimand of PN (for a nonexperimental event) under various assumptions and various types of data – the stronger the assumptions, the more informative the estimates.

We see that the excess risk ratio (ERR), which epidemiologists commonly equate with the probability of causation, is a valid measure of PN only when two assumptions can be ascertained: exogeneity (i.e., no confounding) and monotonicity (i.e., no prevention). When monotonicity does not hold, ERR provides merely a lower bound for PN, as shown in (9.13). (The upper bound is usually unity.)

The nonentries (—) in the right-hand side of Table 9.3 represent vacuous bounds $( \mathrm {i . e . ,} 0 \leq \mathrm {P N} \leq 1 )$ In the presence of confounding, ERR must be corrected by the additive term

$$
[ P ( y | x^{\prime} ) - P ( y_{x^{\prime}} ) ] / P ( x , y )
$$

as stated in (9.31). In other words, when confounding bias (of the causal effect) is positive, PN is higher than ERR by the amount of this additive term. Clearly, owing to the division by $P ( x , y )$ , the PN bias can be many times higher than the causal effect bias

$$
P ( y | x^{\prime} ) - P ( y_{x^{\prime}} )
$$

However, confounding results only from association between exposure and other factors that affect the outcome; one need not be concerned with associations between such factors and susceptibility to exposure (see Figure 9.2).

**Table 9.3. PN as a Function of Assumptions and Available Data**

| Assumptions |              |                   | Data Available |               |               |
| ----------- | ------------ | ----------------- | -------------- | ------------- | ------------- |
| Exogeneity  | Monotonicity | Additional        | Experimental   | Observational | Combined      |
| :---:       | :---:        | ---               | :---:          | :---:         | :---:         |
| +           | +            |                   | ERR            | ERR           | ERR           |
| +           | -            |                   | bounds         | bounds        | bounds        |
| -           | +            | covariate control | —              | corrected ERR | corrected ERR |
| -           | +            |                   | —              | —             | corrected ERR |
| -           | -            |                   | —              | —             | bounds        |

The last row in Table 9.3, corresponding to no assumptions whatsoever, leads to vacuous bounds for PN, unless we have combined data. This does not mean, however, that justifiable assumptions other than monotonicity and exogeneity could not be helpful in rendering PN identifiable. The use of such assumptions is explored in the next section.

## 9.4 IDENTIFICATION IN NONMONOTONIC MODELS

In this section we discuss the identification of probabilities of causation without making the assumption of monotonicity. We will assume that we are given a causal model $M$ in which all functional relationships are known, but since the background variables $U$ are not observed, their distribution is not known and the model specification is not complete.

Our first step would be to study under what conditions the function $P(u)$ can be identified, thus rendering the entire model identifiable. If $M$ is Markovian, then the problem can be analyzed by considering each parents–child family separately. Consider any arbitrary equation in $M$ ,

$$
\begin{array}{l} y = f(pa_Y, u_Y) \\ = f(x_1, x_2, \dots, x_k, u_1, \dots, u_m), \tag{9.55} \end{array}
$$

where $U_Y = \{U_1, \ldots, U_m\}$ is the set of background (possibly dependent) variables that appear in the equation for $Y$ . In general, the domain of $U_Y$ can be arbitrary, discrete, or continuous, since these variables represent unobserved factors that were omitted from the model. However, since the observed variables are binary, there is only a finite number $(2^{(2^k)})$ of functions from $PA_Y$ to $Y$ and, for any point $U_Y = u$ , only one of those functions is realized.

This defines a canonical partition of the domain of $U_Y$ into a set $S$ of equivalence classes, where each equivalence class $s \in S$ induces the same function $f^{(s)}$ from $PA_Y$ to $Y$ (see Section 8.2.2). Thus, as $u$ varies over its domain, a set $S$ of such functions is realized, and we can regard $S$ as a new background variable whose values correspond to the set $\{f^{(s)} : s \in S\}$ of functions from $PA_Y$ to $Y$ that are realizable in $U_Y$ . The number of such functions will usually be smaller than $2^{(2^k)}$ .

For example, consider the model described in Figure 9.2. As the background variables $(Q, U)$ vary over their respective domains, the relation between $X$ and $Y$ spans three distinct functions:

$$
f^{(1)}: Y = \text{true}, \qquad f^{(2)}: Y = \text{false}, \quad \text{and} \quad f^{(3)}: Y = X.
$$

The fourth possible function, $Y \neq X$ , is never realized because $f_Y(\cdot)$ is monotonic. The cells $(q, u)$ and $(q', u)$ induce the same function between $X$ and $Y$ ; hence they belong to the same equivalence class.

If we are given the distribution $P(u_Y)$ , then we can compute the distribution $P(s)$ , and this will determine the conditional probabilities $P(y \mid pa_Y)$ by summing $P(s)$ over all those functions $f^{(s)}$ that map $pa_Y$ into the value true,

$$
P(y \mid pa_Y) = \sum_{s: f^{(s)}(pa_Y) = \text{true}} P(s). \tag{9.56}
$$

To ensure model identifiability, it is sufficient that we can invert the process and determine $P(s)$ from $P(y \mid pa_Y)$ . If we let the set of conditional probabilities $P(y \mid pa_Y)$ be represented by a vector $\vec{p}$ (of dimensionality $2^k$ ) and $P(s)$ by a vector $\vec{q}$ , then (9.56) defines a linear relation between $\vec{p}$ and $\vec{q}$ that can be represented as a matrix multiplication (as in (8.13)),

$$
\vec{p} = \boldsymbol{R} \vec{q}, \tag{9.57}
$$

where $\boldsymbol{R}$ is a $2^k \times |S|$ matrix whose entries are either $0$ or $1$ . Thus, a sufficient condition for identification is simply that $\boldsymbol{R}$ , together with the normalizing equation $\sum_j \vec{q}_j = 1$ , be invertible.

In general, $\boldsymbol{R}$ will not be invertible because the dimensionality of $\vec{q}$ can be much larger than that of $\vec{p}$ . However, in many cases, such as the “noisy OR” mechanism

$$
Y = U_0 \bigvee_{i = 1, \dots, k} (X_i \wedge U_i), \tag{9.58}
$$

symmetry permits $\vec{q}$ to be identified from $P(y \mid pa_Y)$ even when the exogenous variables $U_0, U_1, \dots, U_k$ are not independent. This can be seen by noting that every point $u$ for which $U_0 = \text{false}$ defines a unique function $f^{(s)}$ because, if $T$ is the set of indices $i$ for which $U_i$ is true, the relationship between $PA_Y$ and $Y$ becomes

$$
Y = U_0 \bigvee_{i \in T} X_i \tag{9.59}
$$

and, for $U_0 = \text{false}$ , this equation defines a distinct function for each $T$ . The number of induced functions is $2^k + 1$ , which (subtracting $1$ for normalization) is exactly the number of distinct realizations of $PA_Y$ . Moreover, it is easy to show that the matrix connecting $\vec{p}$ and $\vec{q}$ is invertible.

We thus conclude that the probability of every counterfactual sentence can be identified in any Markovian model composed of noisy OR mechanisms, regardless of whether the background variables in each family are mutually independent. The same holds, of course, for noisy AND mechanisms or any combination thereof (including negating mechanisms), provided that each family consists of one type of mechanism.

To generalize this result to mechanisms other than noisy OR and noisy AND, we note that – although $f_Y(\cdot)$ in this example was monotonic (in each $X_i$ ) – it was the redundancy of $f_Y(\cdot)$ and not its monotonicity that ensured identifiability. The following is an example of a monotonic function for which the $\boldsymbol{R}$ matrix is not invertible:

$$
Y = (X_1 \wedge U_1) \vee (X_2 \wedge U_1) \vee (X_1 \wedge X_2 \wedge U_3).
$$

This function represents a noisy OR gate for $U_3 = \text{false}$ ; it becomes a noisy AND gate for $U_3 = \text{true}$ and $U_1 = U_2 = \text{false}$ . The number of equivalence classes induced is six, which would require five independent equations to determine their probabilities; the data $P(y \mid pa_Y)$ provide only four such equations.

In contrast, the mechanism governed by the following function, although nonmonotonic, is invertible:

$$
Y = \text{XOR}(X_1, \text{XOR}(U_2, \dots, \text{XOR}(U_{k-1}, \text{XOR}(X_k, U_k)))),
$$

where $\text{XOR}(\cdot)$ stands for exclusive OR. This equation induces only two functions from $PA_Y$ to $Y$ :

$$
Y = \left\{\begin{array}{ll} \text{XOR}(X_1, \ldots, X_k) & \text{if XOR}(U_1, \ldots, U_k) = \text{false}, \\ \neg \text{XOR}(X_1, \ldots, X_k) & \text{if XOR}(U_1, \ldots, U_k) = \text{true}. \end{array} \right.
$$

A single conditional probability, say $P(y \mid x_1, \ldots, x_k)$ , would therefore suffice for computing the one parameter needed for identification: $P[\text{XOR}(U_1, \dots, U_k) = \text{true}]$ .

We summarize these considerations with a theorem.

## Definition 9.4.1 (Local Invertibility)

A model M is said to be locally invertible if, for every variable $V_{i} \in V$ , the set of $2^{k} + 1$ equations

$$
P(y \mid pa_{i}) = \sum_{s: f^{(s)}(pa_{i}) = true} q_{i}(s), \tag{9.60}
$$

## 9.5 Conclusions

$$
\sum_{s} q_{i}(s) = 1 \tag{9.61}
$$

has a unique solution for $q_{i}(s)$ , where each $f_{i}^{(s)}(\boldsymbol{pa}_{i})$ corresponds to the function $f_{i}(pa_{i}, u_{i})$ induced by $u_{i}$ in equivalence class $s$ .

## Theorem 9.4.2

Given a Markovian model $M = \langle U, V, \{f_{i}\} \rangle$ in which the functions $\{f_{i}\}$ are known and the exogenous variables $U$ are unobserved, if $M$ is locally invertible then the probability of every counterfactual sentence is identifiable from the joint probability $P(\nu)$ .

## Proof

If (9.60) has a unique solution for $q_{i}(s)$ , then we can replace $U$ with $S$ and obtain an equivalent model as follows:

$$
M^{\prime} = \langle S, V, \{f_{i}^{\prime}\} \rangle, \quad \text{where} f_{i}^{\prime} = f_{i}^{(s)}(pa_{i}).
$$

The model $M^{\prime}$ , together with $q_{i}(s)$ , completely specifies a probabilistic causal model $\langle M^{\prime}, P(s) \rangle$ (owing to the Markov property), from which probabilities of counterfactuals are derivable by definition. □

Theorem 9.4.2 provides a sufficient condition for identifying probabilities of causation, but of course it does not exhaust the spectrum of assumptions that are helpful in achieving identification. In many cases we might be justified in hypothesizing additional structure on the model – for example, that the $U$ variables entering each family are themselves independent. In such cases, additional constraints are imposed on the probabilities $P(s)$ , and (9.60) may be solved even when the cardinality of $S$ far exceeds the number of conditional probabilities $P(y \mid pa_{Y})$ .

## 9.5 CONCLUSIONS

This chapter has explicated and analyzed the interplay between the necessary and sufficient components of causation. Using counterfactual interpretations that rest on structural model semantics, we demonstrated how simple techniques of computing probabilities of counterfactuals can be used in computing probabilities of causes, deciding questions of identification, uncovering conditions under which probabilities of causes can be estimated from statistical data, and devising tests for assumptions that are routinely made (often unwittingly) by analysts and investigators.

On the practical side, we have offered several useful tools (partly summarized in Table 9.3) for epidemiologists and health scientists. This chapter formulates and calls attention to subtle assumptions that must be ascertained before statistical measures such as excess risk ratio can be used to represent causal quantities such as attributable risk or probability of causes (Theorem 9.2.14). It shows how data from both experimental and nonexperimental studies can be combined to yield information that neither study alone can reveal (Theorem 9.2.15 and Section 9.3.4). Finally, it provides tests for the commonly made assumption of “no prevention” and for the often asked question of whether a clinical study is representative of its target population (equation (9.32)).

On the conceptual side, we have seen that both the probability of necessity (PN) and probability of sufficiency (PS) play a role in our understanding of causation and that each component has its logic and computational rules. Although the counterfactual concept of necessary cause (i.e., that an outcome would not have occurred “but for” the action) is predominant in legal settings (Robertson 1997) and in ordinary discourse, the sufficiency component of causation has a definite influence on causal thoughts.

The importance of the sufficiency component can be uncovered in examples where the necessary component is either dormant or ensured. Why do we consider striking a match to be a more adequate explanation (of a fire) than the presence of oxygen? Recasting the question in the language of PN and PS, we note that, since both explanations are necessary for the fire, each will command a PN of unity. (In fact, the PN is actually higher for the oxygen if we allow for alternative ways of igniting a spark.) Thus, it must be the sufficiency component that endows the match with greater explanatory power than the oxygen. If the probabilities associated with striking a match and the presence of oxygen are denoted $p_{m}$ and $p_{o}$ , respectively, then the PS measures associated with these explanations evaluate to $\mathrm{PS}(\mathrm{match}) = p_{o}$ and $\mathrm{PS}(\mathrm{oxygen}) = p_{m}$ , clearly favoring the match when $p_{o} \gg p_{m}$ . Thus, a robot instructed to explain why a fire broke out has no choice but to consider both PN and PS in its deliberations.

Should PS enter legal considerations in criminal and tort law? I believe that it should – as does Good (1993) – because attention to sufficiency implies attention to the consequences of one’s action. The person who lighted the match ought to have anticipated the presence of oxygen, whereas the person who supplied – or could (but did not) remove – the oxygen is not generally expected to have anticipated match-striking ceremonies.

However, what weight should the law assign to the necessary versus the sufficient component of causation? This question obviously lies beyond the scope of our investigation, and it is not at all clear who would be qualified to tackle the issue or whether our legal system would be prepared to implement the recommendation. I am hopeful, however, that whoever undertakes to consider such questions will find the analysis in this chapter to be of some use. The next chapter combines aspects of necessity and sufficiency in explicating a more refined notion: “actual cause.”

## Acknowledgments

I am indebted to Sander Greenland for many suggestions and discussions concerning the treatment of attribution in the epidemiological literature and the potential applications of our results in practical epidemiological studies. Donald Michie and Jack Good are responsible for shifting my attention from PN to PS and PNS. Clark Glymour and Patricia Cheng helped to unravel some of the mysteries of causal power theory, and Michelle Pearl provided useful pointers to the epidemiological literature. Blai Bonet corrected omissions from earlier versions of Lemmas 9.2.7 and 9.2.8, and Jin Tian tied it all up in tight bounds.

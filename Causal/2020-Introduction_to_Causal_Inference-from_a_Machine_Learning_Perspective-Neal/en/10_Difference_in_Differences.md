# Difference in Differences

Note: the following chapter is much more rough than usual and currently does not contain as many figures and intuition as the corresponding lecture.

## 10.1 Preliminaries

We first introduced the unconfoundedness assumption (Assumption 2.1) in Chapter 2:

$$
(Y (1), Y (0)) \perp T \tag {10.1}
$$

Recall that this is equivalent to assuming that there are no unblocked backdoor paths from  to  in the causal graph. When this is the case, 𝑇 𝑌we have that association is causation. In other words, it gives us the following (hopefully familiar) identification of the ATE:

$$
\mathbb {E} [ Y (1) - Y (0) ] = \mathbb {E} [ Y (1) ] - \mathbb {E} [ Y (0) ] \tag {10.2}
$$

$$
= \mathbb {E} [ Y (1) \mid T = 1 ] - \mathbb {E} [ Y (0) \mid T = 0 ] \tag {10.3}
$$

$$
= \mathbb {E} [ Y \mid T = 1 ] - \mathbb {E} [ Y \mid T = 0 ] \tag {10.4}
$$

where we used this unconfoundedness in Equation 10.3.

However, the ATE is not the only average causal effect that we might be interested in. It is often the case that practioners are interested in the ATE specifically in the treated subpopulation. This is known as the average treatment effect on the treated (ATT): 𝔼[ (1) − (0) | = 1]. We can make a 𝑌 𝑌 𝑇weaker assumption if we are only interested in the ATT, rather than the ATE:

$$
Y (0) \perp T \tag {10.5}
$$

We only have to assume that $Y ( 0 )$ is unconfounded here, rather than that both $Y ( 0 )$ and $Y ( 1 )$ 𝑌 are unconfounded. We show this in the following 𝑌proof:

$$
\mathbb {E} [ Y (1) - Y (0) \mid T = 1 ] = \mathbb {E} [ Y (1) \mid T = 1 ] - \mathbb {E} [ Y (0) \mid T = 1 ] \tag {10.6}
$$

$$
= \mathbb {E} [ Y \mid T = 1 ] - \mathbb {E} [ Y (0) \mid T = 1 ] \tag {10.7}
$$

$$
= \mathbb {E} [ Y \mid T = 1 ] - \mathbb {E} [ Y (0) \mid T = 0 ] \tag {10.8}
$$

$$
= \mathbb {E} [ Y \mid T = 1 ] - \mathbb {E} [ Y \mid T = 0 ] \tag {10.9}
$$

where we used this weaker unconfoundedness in Equation 10.8.

We are generally interested in the ATT estimand with difference-indifferences, but we will use a different identifying assumption.

10.1 Preliminaries . . . 95

10.2 Introducing Time . . . . . . 96

10.3 Identification 96

Assumptions . . . . . . . . . 96

Main Result and Proof . . . 97

10.4 Major Problems . . . . . . . 98

## 10.2 Introducing Time

We will now introduce the time dimension. Using information from the time dimension will be key for us to get identification without assuming the usual unconfoundedness. We’ll use 𝜏 for the variable for time.

Setting As usual, we have a treatment group $( T = 1 )$ and a control group $( T = 0 )$ 𝑇. However, now there is also time, and the treatment group only gets the treatment after a certain time. So we have some time $\tau = 1$ that denotes a time after the treatment has been administered to the treatment group and some time $\tau = 0$ that denotes some time before the treatment has been administered to the treatment group. Because the control group never gets the treatment, the control group hasn’t received treatment at either of time $\tau = 0$ or at time $\tau = 1$ . We will denote the random variable for potential outcome under treatment at time 𝜏 as $Y _ { \tau } ( t )$ 𝑡. Then, the causal estimand we’re interested in is the average difference in potential outcomes after treatment has been administered (in time period $\tau = 1 )$ in the treatment group:

$$
\mathbb {E} [ Y _ {1} (1) - Y _ {1} (0) \mid T = 1 ] \tag {10.10}
$$

In other words, we’re interested in the ATT after the treatment has been administered.

## 10.3 Identification

## 10.3.1 Assumptions

You can just treat $Y _ { 1 }$ and $Y _ { 0 }$ as two different random variables. So even though we have a time subscript now, we still have trivial identification via consistency (recall Assumption 2.5) when the value inside of the parenthesis for the potential outcome matches the conditioning value for $T :$

Assumption 10.1 (Consistency) If the treatment is , then the observed outcome $Y _ { \tau }$ 𝑇at time 𝜏 is the potential outcome under treatment . Formally,

$$
\forall \tau , \quad T = t \implies Y _ {\tau} = Y _ {\tau} (t) \tag {10.11}
$$

We could write this equivalently as follow:

$$
\forall \tau , \quad Y _ {\tau} = Y _ {\tau} (T) \tag {10.12}
$$

Consistency is what tells us that the causal estimand $\mathbb { E } [ Y _ { \tau } ( 1 ) \mid T =$ 1] equals the statistical estimand $\mathbb { E } [ Y _ { \tau } \ | \ T = 1 ]$ 𝑌 𝑇, and, similarly, that $\mathbb { E } [ Y _ { \tau } ( 0 ) \mid T = 0 ] = \mathbb { E } [ Y _ { \tau } \mid T = 0 ]$ 𝑌 𝑇. In contrast, $\mathbb { E } [ Y _ { \tau } ( 1 ) \mid T = 0 ]$ and $\mathbb { E } [ Y _ { \tau } ( 0 ) \mid T = 1 ]$ 𝑌 𝑇 𝑌 𝑇 are counterfactual causal estimands, so consistency does not directly identify these quantities for us. Note: In our derivations in this chapter, we are also implicitly assuming the no interference assumption (Assumption 2.4) extended to this setting where we have a time subscript.

We have now arrived at the defining assumption of difference-in-differences: parallel trends. This assumption states that the trend (over time) in the treatment group would match the trend in the control group (over time) if the treatment group were not given treatment.

## Assumption 10.2 (Parallel Trends)

$$
\mathbb {E} [ Y _ {1} (0) - Y _ {0} (0) \mid T = 1 ] = \mathbb {E} [ Y _ {1} (0) - Y _ {0} (0) \mid T = 0 ] \tag {10.13}
$$

This is like an assumption about unconfoundedness between difference:

$$
\left(Y _ {1} (0) - Y _ {0} (0)\right) \perp T \tag {10.14}
$$

So you could see this as like the regular unconfoundedness we saw in Equation 10.5, but where treatment is independent of a difference of potential outcomes, rather than being independent of the potential outcome themselves.

Then, we need one final assumption. This is the assumption that the treatment has no effect on the treatment group before it is administered.

## Assumption 10.3 (No Pretreatment Effect)

$$
\mathbb {E} \left[ Y _ {0} (1) - Y _ {0} (0) \mid T = 1 \right] = 0 \tag {10.15}
$$

This assumption may seem like it’s obviously true, but that isn’t necessarily the case. For example, if participants anticipate the treatment, then they might be able to

## 10.3.2 Main Result and Proof

Using the assumptions in the previous section, we can show that the ATT is equal to the difference between the differences across time in each treatment group. We state this mathematically in the following proposition.

Proposition 10.1 (Difference-in-differences Identification) Given consistency, parallel trends, and no pretreatment effect, we have the following:

$$
\begin{array}{l} \mathbb {E} \left[ Y _ {1} (1) - Y _ {1} (0) \mid T = 1 \right] \\ = \left(\mathbb {E} \left[ Y _ {1} \mid T = 1 \right] - \mathbb {E} \left[ Y _ {0} \mid T = 1 \right]\right) - \left(\mathbb {E} \left[ Y _ {1} \mid T = 0 \right] - \mathbb {E} \left[ Y _ {0} \mid T = 0 \right]\right) \tag {10.16} \\ \end{array}
$$

Proof. As usual, we start with linearity of expectation:

$$
\mathbb {E} [ Y _ {1} (1) - Y _ {1} (0) \mid T = 1 ] = \mathbb {E} [ Y _ {1} (1) \mid T = 1 ] - \mathbb {E} [ Y _ {1} (0) \mid T = 1 ] \tag {10.17}
$$

We can immediately identify the treated potential outcome in the treated group using consistency

$$
= \mathbb {E} [ Y _ {1} \mid T = 1 ] - \mathbb {E} [ Y _ {1} (0) \mid T = 1 ] \tag {10.18}
$$

Regular unconfoundedness:

$$
Y (0) \perp T \quad (1 0. 5 \text {   revisisted })
$$

Active reading exercise: How would you estimate the statistical estimand on the right-hand side of Equation 10.16?

So we’ve identified the first term, but the second term remains to be identified. To do that, we’ll solve for this term in the parallel trends assumption:1

$$
\mathbb {E} \left[ Y _ {1} (0) \mid T = 1 \right] = \mathbb {E} \left[ Y _ {0} (0) \mid T = 1 \right] + \mathbb {E} \left[ Y _ {1} (0) \mid T = 0 \right] - \mathbb {E} \left[ Y _ {0} (0) \mid T = 0 \right] \tag {10.19}
$$

We can use consistency to identify the last two terms:

$$
= \mathbb {E} [ Y _ {0} (0) \mid T = 1 ] + \mathbb {E} [ Y _ {1} \mid T = 0 ] - \mathbb {E} [ Y _ {0} \mid T = 0 ] \tag {10.20}
$$

But the first term is counterfactual. This is where we need the no pretreatment effect assumption:2

$$
= \mathbb {E} [ Y _ {0} (1) \mid T = 1 ] + \mathbb {E} [ Y _ {1} \mid T = 0 ] - \mathbb {E} [ Y _ {0} \mid T = 0 ] \tag {10.21}
$$

Now, we can use consistency to complete the identification:

$$
= \mathbb {E} [ Y _ {0} \mid T = 1 ] + \mathbb {E} [ Y _ {1} \mid T = 0 ] - \mathbb {E} [ Y _ {0} \mid T = 0 ] \tag {10.22}
$$

Now that we’ve identified $\mathbb { E } [ Y _ { 1 } ( 0 ) \mid T = 1 ]$ , we can plug Equation 10.22 𝑌 𝑇back into Equation 10.18 to complete the proof:

$$
\begin{array}{l} \mathbb {E} [ Y _ {1} (1) \mid T = 1 ] - \mathbb {E} [ Y _ {1} (0) \mid T = 1 ] \\ = \mathbb {E} \left[ Y _ {1} \mid T = 1 \right] - \left(\mathbb {E} \left[ Y _ {0} \mid T = 1 \right] + \mathbb {E} \left[ Y _ {1} \mid T = 0 \right] - \mathbb {E} \left[ Y _ {0} \mid T = 0 \right]\right) (10.23) \\ = \left(\mathbb {E} \left[ Y _ {1} \mid T = 1 \right] - \mathbb {E} \left[ Y _ {0} \mid T = 1 \right]\right) - \left(\mathbb {E} \left[ Y _ {1} \mid T = 0 \right] - \mathbb {E} \left[ Y _ {0} \mid T = 0 \right]\right) (10.24) \\ \end{array}
$$

1 Parallel trends assumptions (Assumption 10.2):

$$
\begin{array}{l} \mathbb {E} [ Y _ {1} (0) \mid T = 1 ] - \mathbb {E} [ Y _ {0} (0) \mid T = 1 ] \\ = \mathbb {E} [ Y _ {1} (0) \mid T = 0 ] - \mathbb {E} [ Y _ {0} (0) \mid T = 0 ] \tag {10.13revisited} \\ \end{array}
$$

2 No pretreatment effect assumption (Assumption 10.3)

$$
\begin{array}{r l} \mathbb {E} [ Y _ {0} (1) \mid T = 1 ] - \mathbb {E} [ Y _ {0} (0) \mid T = 1 ] & = 0 \\ & \text {(10.15 revisited)} \end{array}
$$

## 10.4 Major Problems

The first major problem with the difference-in-differences methods for causal effect estimation is that the parallel trends assumption is often not satisfied. We can try to fix this by controlling for relevant confounders and trying to satisfy the controlled parallel trends assumption:

Assumption 10.4 (Controlled Parallel Trends)

$$
\mathbb {E} [ Y _ {1} (0) - Y _ {0} (0) \mid T = 1, W ] = \mathbb {E} [ Y _ {1} (0) - Y _ {0} (0) \mid T = 0, W ] \tag {10.25}
$$

This is commonly done in practice, but it still might not be possible to satisfy this weaker version of the parallel trends assumption. For example, if there is an interaction term between treatment  and time 𝜏 in the 𝑇structural equation for , we will never have parallel trends.

Additionally, the parallel trends assumption is scale-specific. For example, if we satisfy parallel trends, this doesn’t imply that we satisfy parallel trends under some transformation of . The logarithm is one common such transformation. This is because the parallel trends assumption is an assumption about differences, which makes it not fully nonparametric. In this sense, the parallel trends assumption is semi-parametric. And, similarly, the difference-in-differences method is a semi-parametric method.
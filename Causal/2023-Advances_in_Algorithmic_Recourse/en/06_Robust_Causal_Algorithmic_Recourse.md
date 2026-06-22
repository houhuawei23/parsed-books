# Robust Causal Algorithmic Recourse

## Chapter Abstract

Algorithmic recourse seeks to provide actionable recommendations for individuals to overcome unfavorable outcomes made by automated decisionmaking systems. Recourse recommendations should ideally be robust to reasonably small uncertainty in the features of the individual seeking recourse. In this work, we formulate the adversarially robust recourse problem and show that recourse methods offering minimally costly recourse fail to be robust. We then present methods for generating adversarially robust recourse in the linear and in the differentiable case. Finally, we empirically show that regularizing the decision-making classifier to rely more strongly on actionable features facilitates the existence of adversarially robust recourse.

## 6.1 introduction

Machine learning (ML) classifiers are increasingly being used for consequential decision-making in domains such as justice and finance (e.g., granting pretrial bail or loan approval). The need to preserve human agency despite the rise in automatic decisions faced by individuals has motivated the study of algorithmic recourse, which aims to empower individuals by providing them with actionable recommendations to reverse unfavourable algorithmic decisions (USL19). Prior works have argued that for recourse to warrant trust, the decision-maker must commit to reversing an unfavourable decision upon the decision-subject fully adopting their prescribed recourse recommendations (WMR17; VA20; Kar+22). We argue that if algorithmic recourse is indeed to be treated as a contractual agreement, then recourse recommendations must be robust to plausible uncertainties arising in the recourse process.

For instance, consider a bank that commits to approving the loan of an individual if they increase their savings by some amount. Suppose that by the time the individual achieves the prescribed savings increase, the individual’s weekly working hours have been slightly reduced due to unforeseen circumstances, and the classifier still deems the individual likely to default on the loan. Shielding the recourse recommendation against uncertainty expost by nonetheless granting the loan may be detrimental to both the bank (e.g., monetary loss) and the individual (e.g., bankruptcy and inability to secure future loans), while breaking the recourse promise would negate the effort exerted by the individual and erode trust in the decision maker. We therefore argue for the necessity of ensuring that recourse recommendations are ex-ante robust to uncertainty.

In this work, we direct our focus towards robustifying recourse recommendations against uncertainty in the features of the individual seeking recourse. Such uncertainty may arise due to the temporal nature of recourse (e.g., some features may not be static), and/or the presence of noise, adversarial manipulation and other misrepresentations or errors. We adopt a robust optimization view and propose to characterize the uncertainty around the reported features of the individual x by defining an uncertainty set B(x) which we assume contains the true features of the individual at the time recourse is offered and/or plausible changes to the individual’s features arising due to the temporal nature of recourse. We then seek robust recourse recommendations which remain valid (i.e., lead to favourable classification outcomes) for all plausible individuals in the uncertainty set, as illustrated in Figure 6.1. We refer to this notion of robustness as the adversarial robustness of recourse.

![image_20](images/image_20.png)

Robust
recourse action
Non-robust
recourse action
x

Figure 6.1: Adversarially robust recourse actions must lead to positive classification outcomes for all individuals in the uncertainty set around the individual x seeking recourse.

We study the adversarial robustness of recourse from the lens of causality (Pea09). Causal recourse views recourse recommendations as causal interventions on the features of the decision-subject (KSV21), and therefore presents a more faithful account of how the features of the individual change as the individual acts on their recourse recommendations, provided that the underlying structural causal model is known or can be approximated reasonably well (Kar+20b).

## contributions

• We formulate the adversarially robust recourse problem and show that minimum-cost recourse recommendations are provably fragile to uncertainty in the features of the individual seeking recourse.
• We present methods for generating adversarially robust causal recourse in the linear and in the differentiable case. We demonstrate their effectiveness on five tabular datasets, for linear and neural network classifiers.
• We propose a model regularizer that encourages the decision-making classifier to rely more strongly on actionable features. We empirically show that our proposed model regularizer facilitates the existence of adversarially robust recourse.

## 6.2 background and related work

## 6.2.1 Background on causality

We assume that the data-generating process of the features $\mathbf { X } = \{ X _ { 1 } , \ldots , X _ { n } \}$ of individuals $\textbf { \textit { x } } \in \mathbf { \textit { X } }$ is characterised by a known structural causal model (SCM) (Pea09) $\begin{array} { r l r } { \mathcal { M } } & { { } = } & { \left( { \mathbb S } , P _ { \mathbf { U } } \right) } \end{array}$ . The structural equations $\begin{array} { r l } { \mathbb { S } } & { { } = } \end{array}$ $\left\{ X _ { i } : = f _ { i } \left( \mathbf { \boldsymbol { X } } _ { \mathtt { p a } ( i ) } , U _ { i } \right) \right\} _ { i = 1 } ^ { n }$ describe the causal relationship between any given feature $X _ { i } ,$ , its direct causes $\mathbf { X } _ { \mathrm { p a } ( i ) }$ and some exogenous variable $U _ { i }$ as a deterministic function $f _ { i } .$ . The exogenous variables $\textbf { U } \in \boldsymbol { \mathcal { U } } ,$ , which are distributed according to some probability distribution $P _ { \mathbf { U } }$ , represent unobserved background factors which are responsible for the variations observed in the data. We assume that the causal graph  implied by the SCM, with nodes $\mathbf { x } \cup \mathbf { U }$ and edges $\{ ( v , \mathbf { X } _ { i } ) : v \in \dot { \mathbf { X } _ { \mathrm { p a } ( i ) } } \cup U _ { i } , i \in [ 1 , n ] \}$ , is acyclic. The SCM then implies a unique observational distribution p over the features X. Moreover, the structural equations S induce a mapping $\mathbb { S } : \mathcal { U } \ : \to \ : \mathcal { X }$ between exogenous and endogenous variables. Under the assumption that the exogenous variables are mutually independent (causal sufficiency), if there exists some inverse mapping $\mathbb { S } ^ { - 1 } : \dot { \mathcal { X } } \to \mathcal { U }$ such that $\mathbb { S } \left( \mathbb { S } ^ { - 1 } ( \mathbf { x } ) \right) = \mathbf { x } \ \forall x \in \mathcal { X } ,$ then the endogenous variables corresponding to some individual $\mathbf { x } \in \mathcal { X }$ are uniquely identifiable by $\mathbf { U } | \mathbf { x } = \mathbb { S } ^ { - 1 } ( \mathbf { x } )$ .

SCMs allow for modelling and evaluating the effect of interventions on the system which the SCM models. Hard interventions do $( \mathbf { X } _ { \mathcal { I } } : = \pmb { \theta } )$ (Pea09) fix the values of a subset ${ \mathcal { T } } \subseteq [ d ]$ ] of features $\mathbf { \boldsymbol { x } } _ { \mathcal { T } }$ to some $\pmb \theta \in \mathbb { R } ^ { | \mathcal { T } | }$ by altering the structural equations of the intervened upon variables $\mathbb { S } _ { \mathcal { T } _ { i } } ^ { \mathrm { d o } ( \mathbf { X } _ { \mathcal { T } } : = \pmb { \theta } ) } = \mathbf { X } _ { \mathcal { T } _ { i } } : = \pmb { \theta } _ { i }$ while preserving the rest of the structural equations $\mathbb { S } _ { i } ^ { \mathrm { d o } ( \pmb { \chi } _ { \mathbb { Z } } : = \pmb { \theta } ) } = \mathbb { S } _ { i }$ Consequently, hard interventions sever the causal relationship between an intervened upon variables and all of its ancestors in the causal graph. Soft interventions, on the other hand, may modify the structural equations in a more general manner (Kor+04). In particular, additive interventions perturb the features X with some perturbation vector $\boldsymbol { \Delta \mathbf { \Psi } } \in \mathbb { R } ^ { n }$ while preserving all causal relationships, altering the structuralCF equations according to

$$
\mathbb {S} ^ {\Delta} = \left\{X _ {i} := f _ {i} \left(\mathbf {X} _ {\mathrm{pa} (i)}, U _ {i}\right) + \Delta_ {i} \right\} _ {i = 1} ^ {n} (\text { ESo7 }).
$$

Moreover, SCMs imply distributions over counterfactuals, allowing to reason about what would have happened under certain hypothetical interventions all else being equal. Under the aforementioned assumptions, the counterfactual $\mathbf { x } ^ { \mathsf { C F } }$ pertaining to some observed factual individual $\textbf { \em x } \in { \mathcal { X } }$ under some hypothetical hard intervention do $( \mathbf { X } _ { \mathcal { T } } : = \theta )$ (resp. soft intervention $\Delta )$ can be computed by first determining the exogenous variables $\mathbf { U } | \mathbf { x } = \mathbb { S } ^ { - 1 } \left( \mathbf { x } \right)$ corresponding to the individual $\mathbf { x , }$ and then applying the interventional mapping $\mathbb { S } ^ { \mathrm { d o } ( \mathbf { X } _ { \mathcal { T } } : = \pmb { \theta } ) }$ $( \mathrm { r e s p . } \mathbb { S } ^ { \Delta } )$ from endogenous to exogenous variables (Pea09). For notational convenience, we denote such mapping as $\mathbf { x } ^ { \mathbb { C } \mathbb { F } } = \mathbb { C } \mathbb { F } \left( \mathbf { x } , \mathrm { d o } ( \mathbf { X } _ { \mathcal { T } } : = \pmb { \theta } ) \right) : = \mathbb { S } ^ { \mathrm { d o } ( \mathbf { X } _ { \mathcal { T } } : = \pmb { \theta } ) } \left( \mathbb { S } ^ { - 1 } \left( \mathbf { x } \right) \right)$ (resp. $\mathbf { x } ^ { \mathsf { C F } } = \mathbb { C F } \left( { \bar { \mathbf { x } } } , { \bar { \Delta } } \right) : =$ $\mathbb { S } ^ { \Delta } \left( \mathbb { S } ^ { - 1 } \left( \mathbf { x } \right) \right) )$ I. We use the notation $\begin{array} { r } { \mathbf { x } ^ { \mathsf { C F } } ~ = ~ \mathbb { C F } \left( \mathbf { x } , \mathbf { d o } ( \mathbf { X } _ { \mathcal { T } } : = \pmb { \theta } ) , \mathcal { M } \right) } \end{array}$ (resp. $\mathbf { x } ^ { \mathsf { C F } } \overset { \cdot } { = } \mathbb { C F } \left( \mathbf { x } , \Delta , \mathcal { M } \right) )$ I to highlight that the counterfactual corresponds to a particular structural causal model .

## 6.2.2 The causal recourse problem

Consider the setting where a classifier $h : \mathcal { X }  \{ 0 , 1 \}$ is used to assign either favourable or unfavourable outcomes to individuals $\mathbf { x } \in \mathcal { X }$ (e.g., loan approval). We adopt the causal view of recourse introduced by Karimi et al. [KSV21] and model recourse recommendations as a hard interventions on the features of the individual seeking recourse, that is, $\boldsymbol { a } = \mathrm { d } \mathbf { o } ( \mathbf { X } _ { \mathcal { T } } : = \mathbf { x } _ { \mathcal { T } } + \pmb { \theta } )$ ), where θ is the prescribed change to some variables $\mathbf { \boldsymbol { x } } _ { \mathcal { T } }$ . We consider this additive form, rather than $\boldsymbol { a } = \mathrm { d o } ( \mathbf { X } _ { \mathcal { T } } : = \boldsymbol { \theta } )$ as Karimi et al. [KSV21], to explicitly allow for uncertainty in the factual individual x to propagate to the recourse recommendation a.

For a recourse action a to be considered valid, the corresponding counterfactual individual must be favourably classified, that is, $h \left( \mathbb { C F } \left( \mathbf { x } , a , \mathcal { M } \right) \right) = 1$ . Since certain features may be immutable $( \mathrm { e . g . , r a c e ) }$ or bounded (e.g., age), only feasible actions should be recommended. The action feasibility set $\mathcal { F } ( \mathbf { x } )$ captures the set of feasible actions available to the individual x. Ideally, recourse recommendations should incur the least amount of effort possible for decision-subjects, where the cost function $c ( \mathbf { x } , a )$ models the effort required by an individual $\textbf { \em x } \in { \mathcal { X } }$ to implement the recourse action a. Finding the minimum-cost recourse action for some individual $\mathbf { x } \in \mathcal { X }$ is therefore equivalent to solving the following optimization problem:

$$
\underset {a = \mathrm{do} (\mathbf {X} _ {\mathcal {I}} := \mathbf {x} _ {\mathcal {I}} + \boldsymbol {\theta})} {\text { argmin }} \quad c (\mathbf {x}, a)
$$

$$
\text { s.t. } \quad a \in \mathcal {F} (\mathbf {x}) \tag {6.1}
$$

$$
\left[ h \right] \left(\mathbb {C F} \left(\left[ \mathbf {x} \right], a, \left[ \bar {\mathcal {M}} \right]\right)\right) = 1
$$

As highlighted in Equation 6.1, uncertainty in the features of the individual $\mathbf { x } ,$ the classifier $h ,$ and/or the SCM may affect the validity of recourse. In Appendix E.1, we discuss and relate the different sources of uncertainty arising throughout the recourse process.

The non-causal recourse setting is equivalent to the causal recourse setting under the independently manipulable features (IMF) assumption, that is, if no causal relationships exist between the features of the individual. Under such assumption, CF $( { \pmb x } , \mathrm { d o } ( { \pmb X } : = { \pmb x } + { \pmb \theta } ) ) = { \pmb x } + { \pmb \theta } .$ .

## 6.2.3 Related work

We now draw connections with existing literature on the robustness of recourse. Previous works have considered the problem of generating recourse actions which remain valid under uncertainty in the classifier h. Pawelczyk et al. [PBK20] show that recourse actions which place the counterfactual individual in regions of the feature space with large data support are more robust under predictive multiplicity compared to minimum-cost recourse actions. However, recourse actions with large data support may be unnecessarily costly. In contrast, our approach seeks to find robust recourse actions with the lowest possible cost. Another line of work has considered robustness of recourse with respect to changes to the classifier in response of dataset shift. Rawal et al. [RKL20b] show that recourse actions are typically not robust to such model changes, and Upadhyay et al. [UJL21] aim to mitigate this issue by generating recourse with a minimax optimization procedure where the cost the recourse is minimized subject to the recourse action being valid under adversarial changes to the classifier h. While we adopt a similar minimax approach to generate robust recourse, we focus on robustifying recourse against uncertainty in the individual x rather than the classifier h. Lastly, Black et al. [Bla+21] adopt a distributionally robust optimization approach to generate recourse recommendations that are consistent across different classifiers h arising from small changes to the initial training conditions. Likewise, a natural extension of our work is to adopt a distributionally robust viewpoint.

Regarding robustness of recourse with respect to uncertainty in the SCM , Karimi et al. [Kar+20b] consider the setting where the underlying SCM is not know and thus must be approximated, and propose a recourse method to generate recourse recommendations which have low probability of being invalid due to the misspecification of the underlying SCM. Our work is tangential to Karimi et al. [Kar+20b].

Finally, previous works have identified that small changes to the features of the decision-subject x may result in different recourse recommendations with potentially very different costs of recourse (Küg+22; Sla+21; Art+21). Instead of focusing on the cost of recourse, we study the robustness of the validty of recourse. The concurrent work of Virgolin and Fracaros [VF22] is most similar to ours, as they consider the robustness of recourse to adversarial perturbations to the individual x. They present an evolutionary algorithm to generate robust recourse, and provide empirical results for random forest classifiers. In contrast, we focus on generating recourse for differentiable classifiers, and we provide empirical results for linear and neural network classifiers. Additionally, we consider the more general causal recourse setting, and we model feature perturbations in a causal manner.

## 6.3 counterfactual uncertainty sets

In the adversarial robustness literature, uncertainty in the features of some data point x is often modelled by an ϵ-ball of uncertainty $B ( { \pmb x } ) ~ = ~ \{ { \pmb x } + { \bf \Sigma }$ $\Delta | \big | \big | \Delta \big | \big | \leq \epsilon \big \}$ around $\mathbf { x , }$ where the norm $\left\| \cdot \right\|$ characterizes some relevant notion of similarity $d ( \mathbf { x } , \mathbf { y } ) = \| \mathbf { x } - \mathbf { y } \|$ between data points, and ϵ characterizes the amount of uncertainty present (Mad+18; Ber+19). Intuitively, small perturbations $\Delta$ to the data point x result in similar data points. Then, the uncertainty set $B ( \mathbf { x } )$ can be interpreted as a neighbourhood of plausible data points similar to the observed data point x.

From a causal perspective, such feature changes $\delta$ are equivalent to additive interventions on the features x under the IMF assumption, that is, if not causal relationships exist between features. We argue, however, that explicitly considering these causal relationships can potentially provide more informative neighbourhoods of individuals.

Definition 6.3.1 (Neighbourhood of counterfactually similar individuals). For some similarity norm , SCM and factual individual $\mathbf { x , }$ we define the ϵ-neighbourhood of counterfactually similar individuals to x as the set of counterfactuals under all possible ϵ-small additive interventions

$$
B (\mathbf {x}) = \left\{\mathbb {C F} \left(\mathbf {x}, \Delta , \mathcal {M}\right) \mid \| \Delta \| \leq \epsilon \right\} \tag {6.2}
$$

As a motivating example, consider the SCM  with features $X _ { 1 } = U _ { 1 }$ and $X _ { 2 } = X _ { 1 } + U _ { 2 }$ respectively denoting the income and savings of some individual x. Figure 6.2 illustrates the observational and counterfactual neighbourhoods of similar individuals for the 2-norm similarity metric $\lVert \cdot \rVert _ { 2 }$ . Observe that under the counterfactual neighbourhood, the individual x is more similar to some individual x¯ with higher income and higher savings than to some other individual x˜ with higher income but lower savings, since the latter is not well explained by the SCM and thus its circumstances may substantially differ from those of x (e.g. has a much larger number of individuals dependent on them, resulting in lower savings despite its higher income). Therefore, we argue that counterfactual neighbourhoods can be more informative than observational neighbourhoods, since the causal relationships between features are explicitly considered.

## 6.4 the adversarially robust recourse problem

We consider the problem of generating recourse actions which are robust to uncertainty in the features of the individual seeking recourse. We adopt a robust optimization point of view and require robust recourse actions to remain valid for every plausible individual in the uncertainty set $B ( \mathbf { x } )$ .

Definition ${ \bf 6 . 4 . 1 }$ (Adversarially robust recourse problem). For some uncertainty set $B ( \mathbf { x } )$ , the minimum-cost recourse action which remains valid for all plausible individuals $\mathbf { x } ^ { \prime } \in B ( \mathbf { x } )$ in the uncertainty set $B ( \mathbf { x } )$ is given by

$$
\underset {a = \operatorname{do} \left(\mathbf {X} _ {\mathcal {I}} := \mathbf {x} _ {\mathcal {I}} + \boldsymbol {\theta}\right)} {\text { argmin }} \max _ {\mathbf {x} ^ {\prime} \in B (\mathbf {x})} c (\mathbf {x}, a) \tag {6.3}
$$

$$
\mathrm{s.t.} \quad a \in \mathcal {F} (\mathbf {x} ^ {\prime}) \wedge h \left(\mathbb {C F} (\mathbf {x} ^ {\prime}, a)\right) = 1
$$

Observe that any solution a to the above optimization problem must satisfy $h ( { \mathbb C } { \mathbb F } ( { \mathbf x } ^ { \prime } , a ) ) = 1 \ \forall { \mathbf x } ^ { \prime } \in B ( { \mathbf x } )$ , and is thus adversarially robust. In Appendix E.2 we derive sufficient conditions for the existence of adversarially robust recourse.

## 6.4.1 Recourse is fragile under mild conditions

We show that under mild conditions on the cost function $c ,$ feasibility set $\mathcal F ( \mathbf x )$ and SCM , minimum-cost recourse actions are provably fragile to arbitrarily small uncertainty in the features of the individual seeking recourse.

Theorem 6.4.1. Let $a ^ { * }$ be the solution to the recourse optimization problem stated in Equation 6.1. Suppose

(i) The cost function $c ( \mathbf { x } , \mathrm { d o } ( \mathbf { X } _ { \mathcal { T } } : = \mathbf { x } _ { \mathcal { T } } + \pmb { \theta } )$ is strictly convex in θ with minimum = 0  
$\begin{array} { r } { ( i i ) \ \forall \ 0 < t < 1 \ \mathrm { ~ d o } ( \pmb { X } _ { \mathbb { Z } } : = \pmb { x } _ { \mathbb { Z } } + \pmb { \theta } ) ) \in \mathcal { F } ( \pmb { x } ) \ \Longrightarrow \ \mathrm { d o } ( \pmb { X } _ { \mathbb { Z } } : = \pmb { x } _ { \mathbb { Z } } + t \pmb { \theta } ) ) \in } \\ { \mathcal { F } ( \pmb { x } ) } \end{array}$  
(iii) The SCM  is an additive noise model (Pea09).

There exists $\mathbf { x } ^ { \prime } \in B ( \mathbf { x } ) = \{ \mathbb { C } \mathbb { F } \left( \mathbf { x } ; \Delta \right) \} \left\| \Delta \right\| \leq \epsilon > 0 \}$ such that $h ( \mathbb { C F } ( { \mathbf { x } } ^ { \prime } , a ^ { * } ) ) =$ 0, that is, the recourse action $a ^ { * }$ is fragile for any arbitrarily small $\epsilon > 0 .$ .

Condition (i) is satisfied by the most widely used cost functions, namely weighted p-norms (Kar+20b) and percentile costs (USL19). Condition (ii) is satisfied for box actionability constrains, commonly assumed in the recourse literature (Kar+22). Lastly, condition (iii) is a common modelling assumption for estimating the underlying SCM from data (Kar+20b), and also holds in the non-causal recourse setting.

Therefore, in the settings commonly considered by the algorithmic recourse literature, recourse methods seeking minimum-cost recourse offer provably fragile recourse recommendations. This result motivates the study of recourse methods for generating adversarially robust recourse.

## 6.5 generating adversarially robust recourse

## 6.5.1 The linear case

For a linear classifier $h ( \mathbf { x } ) = \left. \mathbf { w } , \mathbf { x } \right. \geq b$ and linear SCM, we show that generating robust recourse for h is equivalent to generating standard recourse for a modified linear classifier $h ^ { \prime } ( \mathbf { x } ) = \langle \mathbf { w } , \mathbf { x } \rangle \geq \bar { b } ^ { \prime }$ whose “acceptance threshold” is sufficiently increased, that is, $b ^ { \prime } \geq b$ .

Theorem 6.5.1. Let $h ( \mathbf { x } ) = \langle \mathbf { w } , \mathbf { x } \rangle \geq b$ be a linear classifier, an SCM with linear structural equations, and $B ( \mathbf { x } ) = \{ { \mathbb { C } } \mathbb { F } \left( \mathbf { x } , \Delta \right) \ | \ \| \Delta \| \leq \epsilon \}$ the uncertainty set of plausible individuals. If the feasibility set is invariant to perturbations to $\mathbf { x , }$ that is, $\forall \mathbf { x } ^ { \prime } \in B ( \mathbf { x } ) : \mathcal { F } ( \mathbf { x } ) \overset { \cdot } { = } \mathcal { F } ( \mathbf { x } ^ { \prime } )$ , then the minimum-cost adversarially robust recourse action for classifier $h ( \mathbf { x } )$ is equivalent to the minimum-cost robust recourse action for the modified classifier

$$
h ^ {\prime} (\mathbf {x}) = \left\langle \mathbf {w}, \mathbf {x} \right\rangle \geq b + \left\| J _ {\mathbb {S} ^ {\mathcal {I}}} ^ {T} \mathbf {w} \right\| ^ {*} \epsilon \tag {6.4}
$$

where $\left\| \cdot \right\| ^ { * }$ denotes the dual norm of $\left\| \cdot \right\|$ and $J _ { { \mathbb S } ^ { \mathbb T } }$ denotes the Jacobian of the interventional mapping resulting from hard-intervening on features $\mathbf { \boldsymbol { x } } _ { \mathcal { T } }$ .

We highlight the importance of this result: if the conditions for Theorem 6.5.1 hold, then any given recourse generating method can be used to generate adversarially robust recourse by considering the modified classifier $h ^ { \prime }$ . In particular, adversarial robustness can be readily combined with other desiderata such as large data-support (Jos+19; PBK20) or fairness constrains (Gup+19; Küg+22).

## 6.5.2 The differentiable case

Similarly to Wachter et al. [WMR17], we consider the following objective function

$$
\mathcal {L} (\mathbf {x}, a, \lambda) = c (\mathbf {x}, a) + \lambda \ell (h (\mathbb {C F} (\mathbf {x}, a)), 1) \tag {6.5}
$$

where ℓ is the binary cross entropy loss. The adversarially robust recourse problem is then equivalent to the following unconstrained penalty problem

$$
\max _ {\lambda \geq 0} \min _ {a \in \mathcal {F} (\mathbf {x})} c (\mathbf {x}, a) + \lambda \max _ {\mathbf {x} ^ {\prime} \in B (\mathbf {x})} \ell (h (\mathbb {C F} (\mathbf {x}, a)), 1) \tag {6.6}
$$

We propose to solve the inner maximization problem using projected gradient ascent over the uncertainty set $B ( \mathbf { x } )$ . For the particular form of the uncertainty set considered in this work, we project to the ϵ-ball of , since $\mathbf { m a x } _ { \mathbf { x ^ { \prime } } \in B ( \mathbf { x } ) }$ $\begin{array} { r } { \ell \left( h \left( \mathbf { C F } \left( \mathbf { x } , a \right) \right) , 1 \right) = \operatorname* { m a x } _ { \| \Delta \| \leq \epsilon } \ \ell \left( h \left( \mathbf { C F } \left( \mathbf { C F } ( \mathbf { x } , \Delta ) , a \right) \right) , 1 \right) } \end{array}$ . Note, however, that the above optimization objective is in general non-convex in $\Delta ,$ and therefore the local maxima found using gradient ascent may not be global maxima in $B ( \mathbf { x } )$ . Thus, it is not possible to guarantee that the recourse actions returned by the proposed algorithm are adversarially robust. However, as discussed in Section $7 \cdot$ we empirically find that the proposed algorithm is effective in robustifying recourse against uncertainty for sufficiently small uncertainty ϵ.

For the outer maximin optimization problem in Equation 6.6, we adopt the causal recourse approach of Karimi et al. [Kar+20b] and use projected gradient descent over the recourse action a and feasibility set $\mathcal { F } ( \mathbf { x } )$ , while also iteratively increasing λ to place growing emphasis in crossing the classifier’s decision boundary. We present the proposed optimization procedure in Algorithm 7.

Algorithm $\boldsymbol { \mathrm { 7 } } \colon$ Generate adversarially robust recourse for a differentiable classifier and SCM.

input: Factual individual x, uncertainty set $B(\mathbf{x})$ , intervention set I, $\lambda > 0, \gamma > 1$ $\theta \leftarrow 0$ while $N \leq N_{\max}$ do
while not converged do $a \leftarrow \text{do}(\mathbf{X}_{\mathcal{I}} := \mathbf{x}_{\mathcal{I}} + \boldsymbol{\theta}) \mathbf{x}^* \leftarrow \arg\max_{\mathbf{x}' \in B(\mathbf{x})} \ell(h(\mathbb{CF}(\mathbf{x}, a)), 1) \text{ if } h(\mathbb{CF}(\mathbf{x}^*, a)) = 1 \text{ then } \text{ return } \boldsymbol{\theta}$ $\theta \leftarrow \text{Proj}_{\mathcal{F}(\mathbf{x})} (\theta - \alpha \nabla_{\theta} \mathcal{L}(\mathbf{x}^*, a, \lambda))$ $\lambda \leftarrow \gamma \lambda$

## 6.6 actionability regularization

To ensure that recourse recommendations are robust, individuals are asked to make more effort than they would have otherwise had to. Consequently, the burden of immunizing recourse against uncertainty falls solely on the decision-subject. We argue, however, that robust recourse desiderata could be directly embedded into the training of the classifier. Satisfying such desiderata may come at a cost in predictive accuracy, thus shifting part of the burden of robust recourse from the decision-subject to the decision maker. In this section, we first restrict ourselves to the linear case in order to theoretically motivate a regularization penalty to reduce the additional cost of robust recourse. We then extend such regularization to the differentiable case by drawing inspiration from local linearity regularization (Qin+19), a popular technique from the adversarial robustness literature. We find that the proposed regularizer substantially facilitates the existence of adversarially robust recourse.

## 6.6.1 Upper bounding the cost of robust recourse

We restrict ourselves to the linear case in order to derive an upper bound on the additional cost of robust recourse under certain actionability assumptions.

Theorem 6.6.1. Let h be a linear classifier $h ( \mathbf { x } ) = \langle \mathbf { w } , \mathbf { x } \rangle \geq b ,$ , an SCM with linear structural equations, $\textbf { \textit { x } } \in \mathbf { \textit { X } }$ a negatively classified individual for which there exists some recourse action $a \ = \ \mathrm { d o } ( \mathbf { X } _ { \mathcal { T } } \ : = \ \mathbf { x } _ { \mathcal { T } } + \pmb { \theta } )$ ), and $B ( \mathbf { x } ) \ = \ \{ { \mathbb { C } } \mathbb { F } \left( \mathbf { x } , \Delta \right) \ | \ \| \Delta \| \le \epsilon \}$ . Then, there exists some constant $\beta$ such that if $\begin{array} { r } { a ^ { \prime } = \mathrm { d o } ( \mathbf { X } _ { \mathcal { T } } : = \mathbf { x } _ { \mathcal { T } } + ( 1 + \beta \epsilon ) \pmb { \theta } ) } \end{array}$ is a feasible action $a ^ { \prime } \in \mathcal { F } ( \mathbf { x } )$ , then $a ^ { \prime }$ is an adversarial robust recourse action. Assuming that the cost function is subadditive, the additional cost incurred by robustifying action a is

$$
\frac {c (\mathbf {x} , a ^ {\prime}) - c (\mathbf {x} , a)}{c (\mathbf {x} , a)} \leq \beta \epsilon , \quad \beta = \frac {\left\| J _ {\mathrm{S} ^ {\mathcal {I}}} ^ {T} \mathbf {w} \right\| ^ {*}}{\langle J _ {\mathrm{S} ^ {\mathcal {I}}} ^ {T} \mathbf {w} , \boldsymbol {\theta} \rangle} \tag {6.7}
$$

Consequently, $\beta \epsilon$ constitutes an upper bound on the additional cost of recourse incurred as a result of seeking robust recourse. We propose to regularize $w$ such that the upper bound on the additional cost of recourse $\beta \epsilon$ is reduced. For simplicity, we henceforth make the IMF assumption, such that $J _ { \mathbb { S } ^ { \mathcal { T } } } ^ { T } = I .$ . Let $\boldsymbol { A }$ (resp. U ) be the set of actionable features (resp. unactionable) and $m _ { \mathcal { A } } \in [ 0 , 1 ] ^ { n }$ (resp. $m _ { \mathcal { U } } \in \mathsf { [ 0 , 1 ] } ^ { n } )$ the mask vector such that $( m _ { \mathcal { A } } ) _ { i } = 1 \iff i \in \mathcal { A }$ (resp. $( m _ { \mathcal { U } } ) _ { i } = 1 \iff i \in \mathcal { U } )$ . Then

$$
\beta = \frac {\left\| \mathbf {w} \right\| ^ {*}}{\langle \mathbf {w} , \boldsymbol {\theta} \rangle} = \frac {\left\| m _ {\mathcal {A}} \odot \mathbf {w} \right\| ^ {*} + \left\| m _ {\mathcal {U}} \odot \mathbf {w} \right\| ^ {*}}{\langle m _ {\mathcal {A}} \odot \mathbf {w} , \boldsymbol {\theta} \rangle} \tag {6.8}
$$

where $\odot$ denotes the elementwise product. Consequently, reducing the dual norm $\| m _ { \mathcal { U } } \odot \mathbf { w } \| ^ { * }$ of the classifier weights corresponding to the unactionable features directly reduces the upper bound on the additional cost of robust recourse $\beta ,$ inducing the learning bias “the classifier should rely more strongly on actionable features”.

## 6.6.2 Actionable local linearity regularization

We consider classifiers of the form $h ( \mathbf { x } ) = g ( \mathbf { x } ) \geq b ,$ where $g ( \pmb { x } )$ is differentiable. With the aim of reducing the additional cost of robust recourse, we propose the following regularizer.

$$
\begin{array}{l} \mathcal {R} (\mathbf {x}) = \mu \| m _ {\mathcal {U}} \odot \nabla_ {x} g (\mathbf {x}) \| ^ {*} \\ + \gamma \max _ {\| \delta \| \leq \epsilon} | g (\mathbf {x} + \delta) - \langle \delta , \nabla_ {x} g (\mathbf {x}) \rangle - g (\mathbf {x}) | \tag {6.9} \\ \end{array}
$$

which we denote as the Actionable Locally Linear Regularizer (ALLR). The first term corresponds to the previously motivated actionability penalty for the linear approximation $h ^ { \prime }$ of the classifier h around $\mathbf { x , }$ and the second term, inspired by Qin et al. [Qin+19], encourages the function $g$ to behave linearly near x, such that the linear classifier $h ^ { \prime }$ is a reasonably accurate approximation of h around x.

## 6.7 experimental results

Firstly, we empirically validate the effectiveness of the methods proposed for generating adversarially robust recourse. Secondly, we empirically show that regularizing the decision-making classifier with our proposed ALLR regularizer facilitates the search of adversarially robust recourse.

We consider four real-world data sets and one semi-synthetic dataset. For the causal recourse setting, we consider the COMPAS recidivism dataset (Lar+16b) and the Adult demographic dataset (Mur94), for which we adopt the causal graphs assumed in Nabi and Shpitser [NS18], and fit the structural equations as 1-layer MLPs. We also consider one semi-synthetic SCM introduced by Karimi et al. [Kar+20b], which is inspired in a loan approval setting. We sample 1000 data points from the SCM, and refer to the resulting dataset as Loan. For the non-causal recourse setting, we consider the South German Credit dataset (Gro19), as well as a recidivism dataset (SW88) from the state of North Carolina which we refer to as Bail. In Appendix $\mathrm { E . 4 } ,$ , we list the features used for every dataset as well as the actionability constrains considered.

For the considered datasets, we treat actionable categorical variables as realvalued, and we standarize all real-valued features. We use as the cost function the $\ell _ { 1 }$ norm of the prescribed feature change, that is $c ( \mathbf { x } , a = \mathrm { d o } ( \mathbf { X } _ { \mathcal { T } } : =$ $\mathbf { x } _ { \mathcal { T } } + \pmb { \theta } ) ) = \| \pmb { \theta } \| _ { 1 }$ . We consider two types of classifiers: logistic regression (LR) models, and neural network (NN) models (3 layers, tanh activation). We define the uncertainty set $B ( \pmb { x } )$ with respect to the 2-norm.

![image_21](images/image_21.png)

Figure 6.4: Fragility of recourse robustified against uncertainty. For linear classifiers, we are unable to find perturbations which invalidate the generated recourse. For NN classifiers, we do find such adversarial perturbations for sufficiently large uncertainty ϵ. Legend: COMPAS Adult Loan Credit Bail.

## 6.7.1 Minimum-cost recourse is fragile

First, we empirically demonstrate that recourse methods which aim to generate minimum-cost recourse fail to be robust. To do so, we train the classifiers using expected risk minimization and generate recourse for the negatively classified individuals with the methods of Wachter et al. [WMR17] and Karimi et al. [KSV21] for the causal and non-causal recourse setting respectively. We then apply the C&W adversarial attack (CW17) to the features of the individuals seeking recourse in order to find the minimum feature perturbation which invalidates the generated recourse. We present the results in Figure 6.3.

We observe that the recourse generated for both LR and NN classifiers is fragile, with adversarial perturbations in the order of $1 0 ^ { - 2 } \ \mathrm { t o } \ 1 0 ^ { - 9 }$ (for standarized features). We observe that the recourse for LR classifiers is substantially more brittle due to the fact that the recourse problem for LR classifiers is convex and thus the minimum-cost recourse action can be found in a more exact manner.

## 6.7.2 Generating adversarially robust recourse

We evaluate the effectiveness of the method proposed in Section 5.2 for generating adversarially robust recourse. To do so, we train the classifiers using expected risk minimization and generate recourse with respect to different uncertainty sets $B ( \mathbf { x } )$ with different levels of uncertainty $\epsilon \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , \dot { 0 } . 5 \}$ . We then use the C&W adversarial attack to find perturbations ∆ to the features of the individual which invalidate the generated recourse actions. If we find some perturbation $\| \Delta \| _ { 2 } \le \epsilon$ which invalidates the generated recourse action, we can state that such recourse action is fragile. The converse, however, is not true, since the absence of found perturbations does not certify that such adversarial perturbations do not exist.

We present the experimental results in Figure 6.4. For LR models, we are unable to find adversarial perturbations invalidating the generated recourse. Indeed, all perturbations found are larger than ϵ by an arbitrarily small amount, but not lower. Thus, for LR models our proposed method can effectively generate minimally-costly robust recourse. However, for NN models, which present a more challenging optimization landscape, our proposed method may generate fragile recourse actions under sufficiently large uncertainty ϵ. Nonetheless, overall our proposed method generates substantially less brittle recourse compared to the standard minimum-cost recourse generation methods previously considered.

## 6.7.3 Actionable local linearity regularization

We empirically evaluate whether classifiers trained with the proposed ALLR regularizer facilitates the existance of adversarially robust recourse. To our knowledge, Ross et al. [RLB21] is the only work proposing a model regularizer to facilitate the existence of algorithmic recourse. Their proposed regularizer augments the model training with “counterfactual examples” by considering the training objective

$$
\mathbb {E} _ {(\mathbf {x}, y) \sim p (\mathbf {x}, y)} [ \ell (h (\mathbf {x}), y) + \lambda \min \delta \ell (h (\mathbf {x}), 1) ] \tag {6.10}
$$

We compare our proposed ALLR regularizer with the regularizer of Ross et al. [RLB21], as well as two other baselines: empirical risk minimization (no regularization), and classifiers which only use actionable features (AF), which amount to ALLR regularization in the limit of infinitely strong regularization $\mu \to \infty$ . We train five classifier with each of these regularization methods, and we evaluate the percentage of individuals for whih recourse is found, as well as the cost of recourse for no uncertainty $\epsilon = 0 ,$ , and under a significant amount of uncertainty $\epsilon = 0 . 1$ . We also evaluate the extent to which the performance of the classifier is impacted by the regularization, by evaluating the prediction accuracy as well as the Matthews correlation coefficient (MCC).

We present the experimental results in Figure 6.5 and Figure 6.6. We find that our proposed regularizer is generally very effective in facilitating the existance of adversarially robust recourse, for both LR and NN models. Additionally, we find that for LR models, our proposed classifier can also significantly reduce the cost of robust recourse, as theoritically motivated in

![image_22](images/image_22.png)

Figure 6.5: For LR models, we find that ALLR regularization of the classifier (penalizing the weights corresponding to unactionable features) substantially facilitates the existence of adversarially robust recourse, more so than the regularizer by Ross. et al. Furthermore, the corresponding robust recourse actions are potentially less costly than those resulting from the classifier trained with ERM. We also find that the predictive performance is generally impacted to a lower exent than for the Ross. et al. and AF regularizers. Legend: ERM ALLR Ross et al. AF Accuracy MCC score.

Section 6. Finally, we find that our proposed regularizer impacts predicition performance to a comparable or lesser degree than the other regularizers considered.

## 6.8 conclusion

Uncertainty in the recourse process is inevitable. Previously suggested ex-post solutions to mitigate the effect of uncertainty in the recourse process may result in negative outcomes for both the decision-maker and the individual. We instead adopt an ex-anti approach to robustness of recourse by requiring the recourse recommendations to be robust to uncertainty in the features of the individual seeking recourse. We show that, in practice, minimum-cost recourse is fragile to arbitrarily small uncertainty in the features of the individual. To address this, we formulate the adversarially robust recourse problem, and present methods to generate adversarially robust recourse in both the linear and differentiable case. Finally, we propose a model regularizer that encourages the deicision-making classifier to rely more strongly on the actionable features, and we empirically show that our proposed regularizer substantially facilitates the existence of adversarially robust recourse.

![image_23](images/image_23.png)

NN Classifiers  
Figure 6.6: For NN models, we find that ALLR regularization of the classifier substantially facilitates the existence of adversarially robust recourse, to a comparable degree to the AF regularizer. We also find that the predictive performance of the predictive model is not greatly impacted. Legend: ERM ALLR Ross et al. AF Accuracy MCC score.
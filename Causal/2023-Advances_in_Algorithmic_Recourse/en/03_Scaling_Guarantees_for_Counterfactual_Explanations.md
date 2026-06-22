# Scaling Guarantees for Counterfactual Explanations

## Chapter Abstract

Counterfactual explanations (CFE) are being widely used to explain algorithmic decisions, especially in consequential decision-making contexts (e.g., loan approval or pretrial bail). In this context, CFEs aim to provide individuals affected by an algorithmic decision with the most similar individual (i.e., nearest individual) with a different outcome. However, while an increasing number of works propose algorithms to compute CFEs, such approaches either lack in optimality of distance (i.e., they do not return the nearest individual) and perfect coverage (i.e., they do not provide a CFE for all individuals); or they do not scale to complex models such as neural networks. In this work, we provide a framework based on Mixed-Integer Programming (MIP) to compute nearest counterfactual explanations for the outcomes of neural networks, with both provable guarantees and runtimes comparable to gradient-based approaches. Our experiments on the Adult, COMPAS, and Credit datasets show that, in contrast with previous methods, our approach allows for efficiently computing diverse CFEs with both distance guarantees and perfect coverage.

<!-- footnote -->

- A common assumption when offering recommendations is that the world is stationary; thus, actions that would have led me to develop this profile had they been performed in the past, will result in the same were they to be performed now. This assumption is challenged in (RKL20b; VA20) and discussed further in §7.1.3.

<!-- footnote end -->

<!-- footnote -->

- Note that “some researchers tend to either collapse or intentionally distinguish contrastive from counterfactual reasoning despite their conceptual similarity” (Ste+21), adding to confusion. For cross-disciplinary reviews, please refer to (Mil18; Mil19; Ste+21).

<!-- footnote end -->

<!-- footnote -->

- Relatedly, the counterfactual instance that results from performing optimal actions, $\mathbf { a } ^ { * } ,$ , need not correspond to the counterfactual instance resulting from optimally and independently shifting features according to $\delta ^ { * } ;$ see (KSV21, prop. 4.1) and (BSR20, Fig. 1). This discrepancy may arise due to, e.g., minimal recommendations suggesting that actions be performed on an ancestor of those variables that are input to the model.

<!-- footnote end -->

<!-- footnote -->

- Optimization terminology refers to both of these constraint sets as feasibility sets. 5The existence of multiple equally costly recourse actions is commonly referred to as the Rashoman effect (Bre+01).

<!-- footnote end -->

<!-- footnote -->

- Alternative categorization of recourse generating methods can be found here (Red+21).

<!-- footnote end -->

<!-- footnote -->

- This chapter is based on the paper “Model-Agnostic Counterfactual Explanations for Consequential Decisions,” Karimi, Barthe, Balle, Valera, AISTATS ( Á), 2019. (Kar+20a).

<!-- footnote end -->

<!-- footnote -->

- We emphasize that while our formulation for generating counterfactuals seems similar to that of adversarial perturbations (image domain), the goals are different: while our goal is to provide actionable and plausible counterfactuals, the goal of adversarial examples is to be imperceptible to humans and hence plausible in the human-perception space, but not in the data space.

<!-- footnote end -->

<!-- footnote -->

- While here we assume binary predictor models, i.e., classifiers, our approach generalizes to regression problems where $y \in \mathbb R$ and more generally any other output domain.

<!-- footnote end -->

<!-- footnote -->

- Constraints on the distance hyperparameters ensure that the overall distance $d ( \mathbf { x } ^ { \mathsf { F } } , \mathbf { x } _ { \epsilon } ^ { \mathsf { C F } } ) \in [ 0 , 1 ]$ . To this end, since max $| \bar { | } \cdot | | _ { 0 } = \operatorname* { m a x } | | \cdot | | _ { 1 } = J , \operatorname* { m a x } | | \cdot | | _ { \infty } = 1$ , the hyperparameters must satisfy $\begin{array} { r } { ( \alpha + \beta ) / J + \gamma = 1 } \end{array}$ .

<!-- footnote end -->

<!-- footnote -->

- $^ { 4 } \hat { \mathbf { x } } _ { \epsilon , j } ^ { i }$ is the j-th dimensions of the i-th counterfactual.

<!-- footnote end -->

<!-- footnote -->

- For the multilayer perceptron, we used two hidden layers with 10 neurons each to avoid overfitting. See Appendix A.2.1 for model selection details.
- Importantly, Actionable Recourse does support actionability and data-range plausibility, however, it lacks support for data-type plausibility – Appendix A.2.3 describes the failure points of AR, as reported by the authors.

<!-- footnote end -->

<!-- footnote -->

- The Adult dataset comprises a realistic mix of integer, real-valued, categorical, and ordinal variables common to consequential scenarios; further details in Appendix A.2.2.

<!-- footnote end -->

<!-- footnote -->

- Complete feature list in Appendix A.3.4

<!-- footnote end -->

<!-- footnote -->

- This chapter is based on the paper “Scaling Guarantees for Nearest Counterfactual Explanations,” Mohammadi, Karimi, Barthe, Valera, ACM-AIES (Á), 2021 (Moh+21).

<!-- footnote end -->

## 3.1 introduction

Machine learning models are increasingly being used to assist in semiautomated prediction and decision-making for consequential scenarios such as pretrial bail and loan approval. Specifically, end-to-end trained models such as (deep) neural networks (LBH15) (with non-linearities such as ReLU) have proven effective at learning and discovering complex non-linear patterns and relations in the data, and hence are becoming widely deployed. However, predictive power often comes at the cost of loss in interpretability (Rud19), i.e., our ability to understand not only the decision made, but also the process by which the decision was deduced. Importantly, interpretability can assay the safe, robust, privacy-preserving, fair, and causally consistent nature of this decision-making (DVK17).

Inspired by this, Counterfactual Explanations (CFEs) are introduced to provide individuals with an understanding of their situation in relation to a close hypothetical scenario in which they would have been treated favorably. As for the process of generating CFEs, a number of criteria are of concern: i) optimal distance, i.e., nearest explanation; ii) perfect coverage, i.e., providing all individuals with an explanation; iii) support for expressive models (e.g. neural networks); iv) efficient runtime; v) support for heterogeneous input spaces; and, vi) qualitative features such as actionability, plausibility, diversity, sparsity, etc. While all these criteria have been discussed in previous works on CFE generation (VDH20; Kar+22), existing approaches however lack in at least one of them.

On one hand, providing the explanations with provable guarantees on the objectives (e.g., the proximity to the factual sample) has been studied by reducing the problem to a Satisfiability Modulo Theories (SMT) problem (Kar+20a; Kar+20a) or to a Mixed-Integer Programming (MIP) problem (Rus19; Kan+20a; USL19). These approaches could theoretically be extended to support many classes of models, however, in practice this has only been demonstrated for simple classes of models, being high runtimes their main bottleneck. As an example, Karimi et al. [Kar+20a] show that even for reasonably small Neural Networks (NNs) (e.g. 20 neurons) the backend SMT solver might never terminate. In contrast, MIP-based approaches, however, so far ignore the class of NN models but instead work with simple linear (Rus19; USL19) or tree-based (Kan+20a) models, emphasizing qualitative metrics of the explanations. On the other hand, counterfactual explanations can be efficiently generated for (differentiable) NN models using gradient-based optimization techniques (MST20). However, while such approaches do work efficiently for NNs, they do not provide any guarantees in terms of distance or coverage. Moreover, they also suffer from limitations to incorporate qualitative aspects of CFE such as actionability constraints–e.g., an input feature capturing individuals’ age is only actionable in one direction, i.e., an individual can only increase her age. Conclusively, previous approaches for CFE generation either ignore the class of neural models or cannot provide the aforementioned guarantees; the exception being MACE (Kar+20a) which suffers from very high runtimes. While NNs are becoming increasingly popular to adopt by stake-holders as a flexible non-linear model, an efficient approach with guarantees is necessary for explaining their decisions.

A similar problem to CFEs, in terms of formulation as a constrained optimization problem, is the generation of adversarial examples for NNs. This problem has been broadly addressed by the NN verification community (Liu+19), where both SMT- and MIP-based approaches have been explored to efficiently solve the problem of finding adversarial examples in ReLUactivated NNs which is, in fact, shown to be NP-complete (Kat+17). It is, however, important to note that while these two problems are formally similar and ideas can be exchanged among them, they are semantically and practically different (WMR17). Thus, approaches to handle adversarial examples in NNs cannot be directly applied to generate CFEs (Fre20).

In this work, we extend the ideas and tools from the NN verification community to develop an efficient framework to compute CFEs for ReLUactivated NN models, to provide distance and coverage guarantees, as well as to accommodate for previously discussed qualitative features. Specifically, we first propose three efficient approaches to search for a CFE within a given interval in the input feature space: whereas the first approach relies on SMT solvers as the backend, the other two approaches formulate the problem as a MIP and differ in the way that the CFE distance is optimized. All the three approaches make use of a linear approximation of the ReLU-NNs (Ehl17) to compute bounds on the hidden units of the NN, given bounds on both the input feature space and/or distance. We then describe how to incorporate several qualitative features in our framework, including heterogeneous distance functions, as well as diversity and plausibility constraints (Kan+20a; Rus19).

Finally, we experiment our approaches on the before-mentioned criteria and compare against SMT- and gradient-based approaches that support NNs. Table 3.1 summarizes the fulfillment of different criteria in CFE generation by our approach in comparison with previous (SMT-, gradient-, and MIP-based) approaches. Our empirical results confirm a significant improvement in runtime efficiency, yielding novel MIP-based approaches for CFE generation on the class of NN models. Importantly, in addition to efficiently generating

**Table 3.1: Comparison of related work with our approach**

<table><tr><td>Method</td><td>Opt. Distance</td><td>100% Coverage</td><td>Efficiency</td><td>Neural Models</td><td>Qualitative Features</td><td>Complex Constraints</td></tr><tr><td>Our approach</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>MACE (Kar+20a)</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>DiCE (MST2o)</td><td></td><td></td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>Efficient Search (Rus19)</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td></tr></table>

CFEs, our presented approaches are optimal in distance and perfect in coverage. This efficiency even allows for generating sets of counterfactuals meeting different criteria, as we show by generating sets of diverse CFEs. Hence, while up to date, runtimes were the main bottleneck for CFE generation with guarantees for NN architectures, our MIP approach performs even faster than gradient-based optimization for NNs at the scale of consequential decisionmaking scenarios.

## 3.2 background

We first introduce counterfactual explanations and two ways of formulating the problem, through optimization and verification. We then explain how the neural network model can be encoded within frameworks capable of solving the counterfactual explanation generation problem exactly and with guarantees.

## 3.2.1 Counterfactual Explanations

Assume that we are given a trained binary classifier $h : \mathcal { X }  \mathbb { R }$ that determines a positive outcome when $h ( \mathbf { x } ) \geq 0$ and a negative outcome when $h ( \mathbf { x } ) < 0 ,$ , deciding, e.g., whether an individual is eligible to receive a loan or not. Consider an individual $\mathbf { x } ^ { \mathsf { F } }$ where $h ( \mathbf { x } ^ { \mathsf { F } } ) < 0$ (loan denial); for this individual, we would like to offer an answer to the question "What would have to be different for you to achieve a positive outcome next time?" 1 Answers to this question may be offered as a feature vector corresponding to an (hypothetical) individual on the other side of the decision boundary, and is referred to as a counterfactual explanation (CFE).

There are a number of criteria/constraints that a CFE should satisfy to be useful for the individual (WMR17). A CFE should ideally be as similar as possible to the individual’s current scenario (the factual instance), corresponding to the smallest change in the individual’s situation that would favorably alter their prediction. Furthermore, the change in features and the resulting counterfactual instance must satisfy additional feasibility and plausibility constraints, respectively. For instance, a change in features that would require the individual to decrease their age would be infeasible (a.k.a. non-actionable). Relatedly, we must make sure that the alternative scenario lies within the heterogeneous input space $( \mathrm { i . e . , }$ is plausible) since in the consequential decisionmaking domains, we typically work with mixed data types with a variety of statistical properties, such as age, race, bank balance, etc.

These requirements can be made more precise by assuming a notion of distance dist between inputs, as well as predicates $\mathcal { P }$ and $\mathcal { F }$ for plausibility and actionability.

## 3.2.1.1 CFE Optimization Formulation

Counterfactual explanations can be modelled as a constrained optimization problem:

$$
\mathbf {x} ^ {\mathrm{CFE}} \in \underset {\mathbf {x} \in \mathcal {X}} {\operatorname{argmin}} \quad \operatorname{dist} (\mathbf {x}, \mathbf {x} ^ {\mathrm{F}}) \tag {3.1}
$$

$$
s. t. \quad h (\mathbf {x}) \geq 0
$$

The above optimization problem can be solved using Gradient Descent (GD) or linear programming, depending on the objective function and the constraints, and yields the closest input $\mathbf { x } ^ { \mathsf { C F E } }$ (with respect to $\mathbf { x } ^ { \mathsf { F } } )$ that is plausible, actionable, and makes the decision of h flip.

## 3.2.1.2 CFE Verification Formulation

The problem of finding counterfactual explanations can be modelled as a satisfaction problem:

$$
\exists \mathbf {x}. \operatorname{dist} (\mathbf {x}, \mathbf {x} ^ {\mathrm{F}}) \leq \delta \tag {3.2}
$$

$$
h (\mathbf {x}) \geq 0
$$

where $\delta$ is a distance threshold. The above satisfaction problem guarantees the existence of $\mathtt { a }$ counterfactual that is plausible, actionable, and within distance $\delta$ of $\mathbf { x } ^ { \mathsf { F } }$ . Using a suitable search strategy over $\delta ,$ it is then also possible to minimize $\delta$ (to an arbitrary precision) and find the nearest counterfactual explanation. For example, MACE (Kar+20a) encodes the above formulation using First-order logic and uses an SMT solver to find a series of counterfactuals within a binary search that minimizes $\delta .$

The precise formulation of the satisfaction problem depends on an encoding of h. Specifically, one must encode the classifier h in the language of logic. While the encodings are theoretically well-understood, it is crucial to choose an encoding that guarantees the scalability of the method. Indeed, even for the simplest models, such as decision trees, naive encodings lead to verification tasks that exceed the capabilities of current tools. An important challenge is thus to develop efficient encodings of other models, and in particular of NNs.

## 3.2.2 Encoding NNs using SMT and MIP

Outside of the domain of consequential decision-making, similar formulations to the CFE problem can be seen in the problem of adversarial examples (Pap+17; MD+17; CW17). Here, there is a well-studied line of research towards verifying different properties of neural networks (Liu+19), such as robustness towards adversarial examples. In this regard, many works focus on proving that a property holds or a counterexample exists. Among these works, many rely on SMT solvers, MIP-based optimization, or both (Ehl17; Kat+17; Bun+18).

Neural network verification task (for ReLU-activated NNs) is shown to be NP-complete (Kat+17). Different works, thus, try to make use of some properties and guide the search process in a way to work better than conventional off-the-shelf solvers or optimizers. Subsequently, we try to do the same for CFE generation and extend the previous work, MACE (Kar+20a), to work better than using off-the-shelf solvers in a straight-forward manner. This happens through, e.g., guiding the search process by gradually increasing the distance within which we are looking for a counterfactual explanation, keeping the distance interval as small as possible to prune domains efficiently.

In the following, we explain how to represent NNs using First-order predicate logic formulae and as an MIP that provide bounds on the optimization variables, later resulting in efficient domain pruning within the search for CFEs.

## 3.2.2.1 First-order Logic (SMT) Encoding of Neural Networks

It is rather straight-forward to encode neural networks using a First-order logic representation that is acceptable by Satisfiability Modulo Theories (SMT) oracles (Kar+20a). Figure 3.1 shows this through an example $( \hat { z } _ { 1 }$ and $\hat { z } _ { 2 }$ represent the post-ReLU values).

![image_06](images/image_06.png)

```mermaid
graph TD
  x1 -->|1| z1
  x1 -->|-1| z2
  x2 -->|0| z2
  x2 -->|-1| z3
  x3 -->|0| z2
  x3 -->|-1| z1
  z1 -->|ẑ₁,-1| z3
  z2 -->|ẑ₂,-1| z3
```

$$
\phi_ {h} (x) = (z _ {1} = x _ {1} - x _ {2})
$$

$$
\wedge (z _ {2} = 2 x _ {1} - x _ {3})
$$

$$
\wedge (z _ {3} = - \hat {z} _ {1} + \hat {z} _ {2})
$$

$$
\wedge \left(\left(\hat {z} _ {1} = z _ {1} \wedge z _ {1} \geq 0\right) \vee \left(\hat {z} _ {1} = 0 \wedge z _ {1} <   0\right)\right)
$$

$$
\wedge \left(\left(\hat {z} _ {2} = z _ {2} \wedge z _ {2} \geq 0\right) \vee \left(\hat {z} _ {2} = 0 \wedge z _ {2} <   0\right)\right)
$$

Figure 3.1: A ReLU-activated neural network and its corresponding logic formula

## 3.2.2.2 Unbounded Mixed-integer Program Encoding of Neural Networks

We try to be faithful to the notation from Liu et al. [Liu+19]. Consider an nlayer single-output feed-forward neural network (NN) with ReLU activations after each hidden layer that represents the function $h ( \mathbf { x } )$ . The width of each layer is $k _ { i }$ and $\mathbf { z } _ { i }$ is the vector of dimension $k _ { i }$ which represents layer i where $i \in \{ 1 , 2 , . . . , n \}$ . While $\mathbf { z } _ { i }$ represents the pre-ReLU activations, $\hat { \mathbf { z } } _ { i }$ is the values after ReLUs have been applied. Finally, $\delta _ { i }$ are vectors of binary variables indicating the state of each ReLU; 0 for inactive and 1 for activated ReLUs.

There are multiple ways to encode neural networks as MIPs in the NN verification literature, each proposing different encodings for ReLU activations. A generic form is as follows. For $i \in \{ 1 , . . . , n \}$ and $j \in \{ 1 , . . . , k _ { i } \}$ :

$$
\mathbf {z} _ {i} = \mathbf {W} _ {i} \hat {\mathbf {z}} _ {i - 1} + \mathbf {b} _ {i} \tag {3.3a}
$$

$$
\boldsymbol {\delta} _ {i} \in \{0, 1 \} ^ {k _ {i}}, \hat {\mathbf {z}} _ {i} = \mathbf {z} _ {i} \cdot \boldsymbol {\delta} _ {i},
$$

$$
\delta_ {i, j} = 1 \Rightarrow z _ {i, j} \geq 0, \tag {3.3b}
$$

$$
\delta_ {i, j} = 0 \Rightarrow z _ {i, j} <   0
$$

The first part (3.3a) is simply the linear affine of weights and the second part (3.3b) encodes the following ReLUs using the introduced binary variables for each ReLU. We refer to this as the unbounded MIP encoding.

## 3.2.2.3 Bounded Mixed-integer Program Encoding of Neural Networks

Bunel et al. [Bun+18] suggest that most NN verifiers, based on either SMT or MIP solvers, are indeed a variation of Branch-and-Bound (B&B) optimization. This understanding implies that limiting the bounds of the variables of the optimization problem is a very effective heuristic. Moreover, the extra constraints of the CFE generation problem – making the verification formulation difficult to solve – might actually help tightening the bounds, and thus, result in an effective pruning of the domains of the optimization problem. We will thus, change the generic ReLU formulation (3.3b) and adopt the bounded encoding proposed by Tjeng and Tedrake [TT17], i.e., for $i \in \{ 1 , . . . , n \}$ :

$$
\mathbf {z} _ {i} = \mathbf {W} _ {i} \hat {\mathbf {z}} _ {i - 1} + \mathbf {b} _ {i} \tag {3.4a}
$$

$$
\delta_ {i} \in \{0, 1 \} ^ {k _ {i}}, \quad \hat {\mathbf {z}} _ {i} \geqslant 0, \quad \hat {\mathbf {z}} _ {i} \leqslant \mathbf {u} _ {i} \cdot \delta_ {i}, \tag {3.4b}
$$

$$
\hat {\mathbf {z}} _ {i} \geqslant \mathbf {z} _ {i}, \quad \hat {\mathbf {z}} _ {i} \leqslant \mathbf {z} _ {i} - \mathbf {l} _ {i} \cdot (1 - \delta_ {i})
$$

Note that the linear part $\left( 3 . 4 \mathrm { a } \right)$ is the same as (3.3a) and also note that this is still an exact encoding of NNs using MIP since $\delta _ { i , j } = 0 \Leftrightarrow \hat { z } _ { i , j } = 0$ and $\delta _ { i , j } = 1 \Leftrightarrow \hat { z } _ { i , j } = z _ { i , j }$ . This encoding relies on $\mathbf { l } _ { i }$ and $\mathbf { u } _ { i } ,$ , vectors indicating the lower and upper bounds of the values of the hidden units at layer i. We remind that tight bounds can be very effective in domain pruning when solving the mixed-integer program. Here, we introduce two ways to obtain such bounds and complete the MIP formulation $( 3 . 4 )$ for CFEs: first, using interval arithmetic (HJVE01), and second, using an approximation of ReLUs that results in tighter bounds. In both cases, we assume that we have initial lower/upper bounds on the values of the input layer (e.g., derived from the dataset). This is a valid assumption since real-world features such as age or income do have bounds.

## 3.2.2.4 Interval arithmetic

By using interval arithmetic (HJVE01), having the bounds at layer $i - 1$ , we can compute the bounds for the j-th neuron from the i-th layer $( z _ { i , j } )$ as:

$$
\begin{array}{l} l _ {i, j} = \Sigma_ {t = 1} ^ {k _ {i - 1}} (m a x (W _ {i, j, t}, 0) \cdot l _ {i - 1, t} \\ + \min (W _ {i, j, t}, 0) \cdot u _ {i - 1, t}) + b _ {i, j} \tag {3.5} \\ u _ {i, j} = \Sigma_ {t = 1} ^ {k _ {i - 1}} (m a x (W _ {i, j, t}, 0) \cdot u _ {i - 1, t} \\ + m i n (W _ {i, j, t}, 0) \cdot l _ {i - 1, t}) + b _ {i, j} \\ \end{array}
$$

The post-ReLU bounds (for $\hat { z } _ { i , j } )$ are obtained simply by applying a ReLU on these bounds.

This is applied layer-by-layer and the bounds for all hidden units are computed recursively starting from the input layer. Unfortunately, although better than having no bounds at all, these bounds quickly become loose as we go deeper in the network. The reason is that in each layer $i ,$ each neuron is choosing a worst-case bound (lower or upper) from the neurons of the previous layer i − 1, independently from the rest of the neurons in layer $i ,$ causing conflicts in the choice of the lower or upper bound for some neurons in layer $i - 1 . ^ { 2 }$

## 3.2.2.5 Linear Over-approximation of ReLUs

To compute tighter bounds than interval arithmetic, we first adopt the linear over-approximation of ReLUs proposed in (Ehl17) to replace (3.3b), i.e., for $i \in \{ 1 , . . . , n \}$ and $j \in \{ 1 , . . . , k _ { i } \}$ :

$$
\mathbf {z} _ {i} = \mathbf {W} _ {i} \hat {\mathbf {z}} _ {i - 1} + \mathbf {b} _ {i} \tag {3.6a}
$$

$$
\hat {\mathbf {z}} _ {i} \geqslant \mathbf {z} _ {i}, \quad \hat {\mathbf {z}} _ {i} \geqslant 0, \quad \hat {z} _ {i, j} \leqslant u _ {i, j} \frac {z _ {i , j} - l _ {i , j}}{u _ {i , j} - l _ {i , j}} \tag {3.6b}
$$

Again, the linear part (3.6a) is the same as (3.3a). For the ReLU part (3.3b), the binary variables encoding the ReLUs in an exact way are removed and, instead, a linear over-approximation term has been replaced (3.6b). This results in a fully linear MIP system without the ReLU binary variables, whose optimization for different objectives can be performed efficiently.

As before, the bounds are recursively computed in a layer-by-layer manner, and the constraints of the linearized network (3.6) are added to the MIP system progressively. At each layer $i ,$ first, (3.6a) is added with bounds of the variables computed using simple interval arithmetic from the tight bounds computed for the previous layer. Then, to find better bounds than simple interval arithmetic, having included all the constraints up until this layer, two MIPs are solved for each hidden unit: one with the objective of maximizing the value of the unit to compute an upper bound, and a similar one for computing the lower bound. Finally, the ReLU constraints (3.6b) for this layer are added with the just-computed tight bounds.2 Note that while we have opted for the ReLU activation function as a common source of non-linearity, any activation function that can be approximated by piece-wise linear functions is applicable, e.g., Max-Pooling (Ehl17).

We build upon an implementation from Bunel et al. [Bun+18] for this purpose. Obtaining tight bounds here relies on how small the domains of the input variables are; keeping the input domains small enough will result in tighter bounds for other variables. This will be discussed in more detail in the next section.

## 3.3 cfe generation

In this section, we propose three approaches towards CFE generation for neural networks. All the approaches rely on the linearized network approximations described in the previous section, which provide tight lower and upper bounds on the values of the hidden units. Below, we first explain the search strategy on the distance of the nearest CFE and the way lower/upper bounds on the input and hidden units are computed within this search. Then, we introduce three approaches towards efficient nearest CFE generation for neural networks.

## 3.3.1 Preliminaries

## 3.3.1.1 Exponential Search Strategy

In order to optimize the distance towards finding the nearest CFE, we implement an exponential search strategy (BYS10). W.l.o.g., we assume here that the input space is normalized and lies within the [0, 1] interval. Because the interval of the input layer determines those of later layers, we initiate our search with a small distance interval, whose lower and upper bound are set respectively to 0 and an (arbitrarily) small ϵ. We then exponentially increase the search interval until a CFE is found. Finally, a simple binary search is performed on the interval where the CFE was found to look for the nearest CFE. The overall scheme for the exponential search is summarized in Algorithm 2.

Algorithm 2: Exponential Search Strategy
Input: N, $x^{F}$ , $\epsilon$ Output: closest_CFE $[lb_{dist}, ub_{dist}] \leftarrow [0, \epsilon]$ ;
while findCFE(N, $x^{F}$ , $lb_{dist}$ , $ub_{dist}$ ) is None do $lb_{dist} \leftarrow ub_{dist}$ ; $ub_{dist} \leftarrow ub_{dist} \times 2$ ;
end
closest_CFE $\leftarrow$ binarySearch(N, $x^{F}$ , $\epsilon$ , $lb_{dist}$ , $ub_{dist}$ );
return closest_CFE;

Next, we discuss how to compute bounds on both the input and hidden units of the network, which are necessary to efficiently implement the CFE search function, findCFE in Algorithm 2.

## 3.3.1.2 Computing Bounds for Input and Hidden Units

We leverage the network approximator based upon equation (3.6) to compute the bounds of the network input and hidden units for a given distance interval $[ l b _ { d i s t } , u b _ { d i s t } ]$ . To this end, we first obtain the MIP encoding of the distance. Then, we optimize the MIP-encoded distance for each input variable, maximizing/minimizing each variable to obtain the lower/upper bounds of the input layer for the given distance interval. Then, the input bounds are propagated in the NN to compute the bounds of hidden units. We include the distance constraints in the initial constraint set of the linearized network to help finding tighter bounds for the hidden units. Algorithm 3 shows the overall scheme for this.

Algorithm 3: Bounds Computation

Input: N, $x^{F}$ , $lb_{dist}$ , $ub_{dist}$ Output: $LB_{net}$ , $UB_{net}$ $\phi_{dist} \leftarrow \text{getDistanceConstraints}(N, x^{F}, lb_{dist}, ub_{dist})$ ; $lb_{inp}, ub_{inp} \leftarrow \text{optimizeInputVars}(N, \phi_{dist})$ ; $LB_{net}, UB_{net} \leftarrow linearizedNetApproximator(N, lb_{inp}, ub_{inp}, \phi_{dist})$ ;

return $LB_{net}, UB_{net}$ ;

## 3.3.2 Approaches

In this section, we propose three efficient approaches to implement the CFE search function, findCFE in Algorithm 2, for neural networks. The first approach relies on SMT solvers as backend and uses the bounds computation as a heuristic within each iteration of the exponential search (Algorithm 2). The second and third approaches instead rely on MIP solving to search for CFEs. The difference between them lies on the optimization of the distance – while the second approach minimizes the CFE distance using the exponential search described above, the third approach includes the distance as objective within the MIP optimization framework. Next, we provide further details on the three approaches.

## 3.3.2.1 ReLU Elimination (MIP-SAT)

In this approach, we build upon MACE (Kar+20a) (SMT solving in the backend) and use the bounds computation as a heuristic. Within each iteration of the exponential search (Algorithm 2), and given the distance interval, the bounds on the input and hidden units are computed using Algorithm 3 and ReLUs with a fixed state are determined. A ReLU has a fixed state iff the value of the neuron before applying ReLU has either a lower bound greater than or equal to zero, or an upper bound less than or equal to zero.

The neural network, distance functions, as well as additional constraints are primarily encoded as SMT formulae. For the NN bounds computation, the NN and distance constraints are encoded as MIPs, as described before. Next, the ReLUs with a fixed-state are removed from the initial SMT formula representing the NN. This means that, for an always-active ReLU, we will have $\hat { z } _ { i } = z _ { i }$ and for an always-inactive ReLU we will have $\hat { z } _ { i } = 0 ,$ , instead of the initial ReLU clause: $( \hat { z } _ { i } = z _ { i } \wedge z _ { i } \geq 0 ) \vee ( \hat { z } _ { i } = 0 \wedge z _ { i } < 0 )$ . This is, basically, removing the disjunction associated to the ReLU states by fixing its value, saving the SMT solver the effort to branch over its cases. Finally, the SMT solver $( Z _ { 3 }$ solver (DMB08) in our case) is called with the new formula to verify the existence of a CFE within the given distance interval.

Note that the ReLU clauses in the SMT representation of the neural network are exponentially expensive to handle for the SMT solver since it forces the solver to branch over the cases. Thus, removing a subset of the RELU activations will reduce the run-time exponentially (as empirically shown in the experiments). Algorithm 4 shows the overall scheme for the proposed mixed MIP-SAT approach.

Algorithm 4: The MIP-SAT approach – findCFE in Algorithm 2
Input: N, $x^{F}$ , $lb_{dist}$ , $ub_{dist}$ Output: CFE or None $\phi_{dist} \leftarrow \text{getDistanceFormula}(N, x^{F}, lb_{dist}, ub_{dist})$ ; $\phi_{pls} \leftarrow \text{getPlausibilityFormula}(N)$ ; $\phi_{N} \leftarrow \text{getModelFormula}(N)$ ; $LB_{net}, UB_{net} \leftarrow computeBounds(N, x^{F}, lb_{dist}, ub_{dist})$ ; $\phi_{N} \leftarrow eliminateRelus(\phi_{N}, LB_{net}, UB_{net})$ ;
if $SAT(\phi_{N} \land \phi_{dist} \land \phi_{pls})$ then
    return CFE;
else
    return None;

## 3.3.2.2 Output Optimization (MIP-EXP)

In this approach, we purely use a MIP-based optimization process (no SMT oracle), for which we deploy an optimization engine (Gurobi (GO20) in this case), building upon an implementation of (3.4) from Bunel et al. [Bun+18].

As before, we assume that we are within an iteration of the exponential search (Algorithm 2) with a fixed distance interval $[ l b _ { d i s t } , u b _ { d i s t } ]$ . First, Algorithm 3 is called to compute tight lower/upper bounds for the input and hidden units of the network. Next, these bounds are used to obtain MIP encoding of the neural network as in (3.4). Then the distance, as well as any other additional constraints (all explained in the next section), are added to MIP formulation. Finally, depending on the (predicted) label of the factual sample $\mathbf { x } ^ { \mathsf { F } } .$ , the single output of the network is optimized. For instance, for a factual sample with a positive label, the output of the network will be minimized with a callback that interrupts the optimization as soon as a counterfactual with a negative output value is found. Otherwise, the lower bound of the output of the network for this factual sample and distance interval is greater than zero and no counterfactual exists. The overall scheme of the proposed MIP-EXP approach is shown in Algorithm 5.

Note that this approach no longer uses an SMT oracle, but instead relies on an optimization engine to solve a mixed-integer program with the single output of the network as its objective function. Thus, it can naturally be extended to multi-class classification by introducing a new variable in the MIP that preserves the maximum logit among class outputs on which the optimization objective is defined.

Algorithm 5: The MIP-EXP approach – findCFE in Algorithm 2

Input: N, $x^{F}$ , $lb_{dist}$ , $ub_{dist}$ Output: CFE or None $\phi_{dist} \leftarrow \text{getDistanceConstraints}(N, x^{F}, lb_{dist}, ub_{dist})$ ; $\phi_{pls} \leftarrow \text{getPlausibilityConstraints}(N)$ ; $LB_{net}, UB_{net} \leftarrow computeBounds(N, xzz^{F}, lb_{dist}, ub_{dist})$ ; $\phi_{N} \leftarrow getModelConstraints(N, LB_{net}, UB_{net})$ ; // MIP encoding 3.4
if optimize( $\phi_{N}, \phi_{dist}, \phi_{pls}, x^{F}$ ) then
| return CFE;
else
| return None;

## 3.3.2.3 Distance Optimization (MIP-OBJ)

This is similar to the MIP-EXP approach except that we remove the outer loop (the exponential search of Algorithm 2) and the distance function is introduced as the objective function of the MIP to be minimized.

In this approach, which we refer to as MIP-OBJ, Algorithm 3 is called to compute the bounds with the distance interval being [0, 1]. The computed bounds are placed within MIP encoding (3.4). Since now the objective of the MIP is the distance function, we need to add a constraint as the counterfactual constraint determining the single output of the network being negative or positive based on the (predicted) label of the factual sample. The whole problem is optimized (with an optimality gap of ϵ for the distance objective to be analogous to the other approaches) and the nearest CFE is found. Algorithm 6 shows the overall scheme of the MIP-OBJ approach.

Algorithm 6: The MIP-OBJ approach

Input: N, $x^{F}$ , $lb_{dist}$ , $ub_{dist}$ Output: CFE or None
obj ← getDistanceConstraints(N, $x^{F}$ ); $\phi_{pls} \leftarrow$ getPlausibilityConstraints(N); $\phi_{CFE} \leftarrow$ getCounterfactualConstraint(N, $x^{F}$ ); $LB_{net}, UB_{net} \leftarrow$ computeBounds(N, $x^{F}$ , 0, 1); // No distance limit $\phi_{N} \leftarrow$ IratisModelConstraints(N, $LB_{net}, UB_{net}$ ); // MIP encoding 3.4
CFE ← optimize( $\phi_{N}, \phi_{pls}, \phi_{CFE}, obj, x^{F}$ );
return CFE;

## 3.4 distance functions and qualitative features

In this section ,we describe how the distance metric, as well qualitative features–such as plausibility, sparsity and diversity–can be encoded within the MIP framework. First, we provide details on the encoding of distance functions suitable for heterogeneous input features. Second, in the context of plausibility, we describe how to handle heterogeneous input spaces, i.e., input features with mixed data types. Finally, we focus on a broadly studied qualitative property of CFEs, diversity. We would like to emphasize that previous MIP-based approaches have recognized the flexibility of mixed-integer programming in regards to encode a wide range of complex constraints and different qualitative features (Rus19; Kan+20a), however, this cannot be directly leveraged for NN models. We defer to future work to address a wider range of qualitative features for NN class of models.

## 3.4.1 Distance Functions

In this section, we provide more details on the MIP encoding of heterogeneous distance functions.3 We provide details on an $\ell _ { 1 }$ distance function (analogous to previous works (WMR17)) while zero-, two-, and infinitynorms are supported in an analogous manner, each providing a different practical intuition for the proximity of the CFEs, $\mathrm { e . g . , \ell _ { 0 } }$ used for sparsity. As described before, the distances are all range normalized and within the [0, 1] interval.

integer-valued and real-valued features For an input vector x and factual sample $\mathbf { x } ^ { \mathsf { F } }$ with such a feature at the i-th dimension, the normalized $\ell _ { 1 }$ distance is computed in a straight-forward manner:

$$
\operatorname{dist} _ {\text { real }} (x _ {i}, x _ {i} ^ {\mathsf {F}}) = \frac {| x _ {i} - x _ {i} ^ {\mathsf {F}} |}{u b _ {i} - l b _ {i}} \tag {3.7}
$$

where $l b _ { i } , u b _ { i }$ are the scalar lower/upper bounds for $x _ { i }$ .

ordinal features For an input vector x and factual sample $\mathbf { x } ^ { \mathsf { F } }$ with an ordinal feature $x _ { i }$ having k levels, the normalized $\ell _ { 1 }$ distance is computed in the following manner:

$$
\operatorname{dist} _ {\text {ord}} \left(x _ {i}, x _ {i} ^ {\mathrm{F}}\right) = \frac {\left| \sum_ {j = 1} ^ {k} x _ {i , j} - \sum_ {j = 1} ^ {k} x _ {i , j} ^ {F} \right|}{k} \tag {3.8}
$$

categorical features For an input vector x and factual sample $\mathbf { x } ^ { \mathsf { F } }$ with a categorical feature $x _ { i }$ having k categories, the normalized $\ell _ { 1 }$ distance is computed in the following manner:

$$
\operatorname{dist} _ {c a t} \left(x _ {i}, x _ {i} ^ {\mathsf {F}}\right) = \max _ {1 \leq j \leq k} \left(x _ {i, j} - x _ {i, j} ^ {\mathsf {F}}\right) \tag {3.9}
$$

In the end, the total normalized $\ell _ { 1 }$ distance between input vector x and factual sample $\mathbf { x } ^ { \mathsf { F } }$ would be the normalized sum over distances of different data types $( 3 . 7 ) , ( 3 . 8 ) , ( 3 . 9 ) , n _ { r e a l } , n _ { o r d } , n _ { c a t }$ t being the number of features in each of the three groups above:

$$
\begin{array}{l} \operatorname{dist} \left(\mathbf {x}, \mathbf {x} ^ {\mathrm{F}}\right) = \frac {1}{n _ {\text {real}} + n _ {\text {ord}} + n _ {\text {cat}}} \left(\sum_ {i = 1} ^ {n _ {\text {real}}} \operatorname{dist} _ {\text {real}} \left(x _ {i}, x _ {i} ^ {\mathrm{F}}\right) \right. \tag {3.10} \\ + \sum_ {i = 1} ^ {n _ {o r d}} \mathsf {d i s t} _ {o r d} (x _ {i}, x _ {i} ^ {\mathsf {F}}) + \sum_ {i = 1} ^ {n _ {c a t}} \mathsf {d i s t} _ {c a t} (x _ {i}, x _ {i} ^ {\mathsf {F}})) \\ \end{array}
$$

sparsity Sparsity can be interpreted as the $\ell _ { 0 }$ distance function. It is encoded by introducing a number of intermediate binary variables each retaining whether or not a feature has changed its value and then summed over and normalized analogous to the described $\ell _ { 1 }$ distance.

## 3.4.2 Plausibility Constraints

In this section we explain plausibility constraints that guarantee the CFE lying within the same heterogeneous space as input. Plausibility constraints for integer-valued, real-valued, and binary variables are naturally preserved by defining the right kind of variables within the MIP (or SMT) model.

ordinal features To guarantee that the CFEs are plausible in terms of ordinality of the ordinal features, for each such feature f with k levels, we define k binary variables $f _ { 1 } , . . . , f _ { k } \in \{ 0 , 1 \}$ in the MIP model. For each set of these variables, the following constraints are added to the MIP model:

$$
f _ {1} \geq f _ {2}, f _ {2} \geq f _ {3},..., f _ {k - 1} \geq f _ {k} \tag {3.11}
$$

This will guarantee that:  i s.t. $f _ { i + 1 } > f _ { i }$ .

categorical features We want to guarantee that in the produced CFE, for each categorical feature, only one category is chosen. For a categorical feature f with k categories, we define k binary variables $f _ { 1 } , \ldots , f _ { k } \in { \overline { { \{ 0 , 1 \} } } }$ in the MIP model. For each set of these variables, the following constraint is added to the MIP model:

$$
f _ {1} + f _ {2} + \dots + f _ {k} = 1 \tag {3.12}
$$

Since $f _ { i } { ' } \mathbf { s }$ are binary variables, this will guarantee that only one of them is 1 and others are $_ { 0 , }$ meaning that at most one category is active as desired.

## 3.4.3 Diversity Constraints

Providing individuals with different, preferably diverse, counterfactuals can be beneficial in terms of providing alternative ways for the individuals to improve their outcome. Having different diverse (and close) counterfactuals, the individuals may find the most suitable way to achieve the preferred outcome while considering their own personal constraints, about which the explanation-provider might not be aware of.

As with other qualitative features, there are different ways for encoding diversity in the literature of CFE generation. Within the MIP-based approaches, Russell [Rus19] encodes diversity simply as the newly generated CFE not being equal to the previously generated ones. Based on the evaluation criteria, this could fail to generate diverse CFEs, for example when the evaluation criteria is the mean of the pairwise distances of the (k) generated CFEs as DiCE (MST20) suggests. Among the gradient-based approaches, DiCE (MST20) accounts for diversity using determinantal point processes, i.e., it includes the determinant of the kernel matrix given the counterfactuals in the objective.

It is important to also take into account the distance of the generated set of diverse counterfactuals since it is necessary for this set to also be close to the individual for which it is being generated. Thus, it can be seen that there is an inherent tradeoff between diversity and distance. To account for this, we encode diversity as a set of constraints for each newly generated counterfactual to have a distance above a fixed threshold from each of the previously generated counterfactuals, while minimizing the distance to the factual sample. More specifically, the following set of constraints will be added before the search for the i-th CFE:

$$
\operatorname{dist} \left(x _ {1} ^ {\text { CFE }}, x _ {i} ^ {\text { CFE }}\right) \geq \delta
$$

$$
\vdots \tag {3.13}
$$

$$
\operatorname{dist} \left(x _ {i - 1} ^ {\text { CFE }}, x _ {i} ^ {\text { CFE }}\right) \geq \delta
$$

Note that solving the MIP becomes progressively more expensive for each new counterfactual. We have implemented a version of our approach called MIP-DIVERSE for generating diverse counterfactuals using the above formulation.

## 3.5 experiments

We conduct a number of quantitative and qualitative experiments to demonstrate our frameworks abilities relative to existing approaches:MACE (Kar+20a) 4 and DiCE (MST20).5 Following the motivation explained in the Introduction, we generate counterfactual explanations for fixedwidth ReLU-activated fully-connected NN models of various sizes, having $N \times W + ( D - 1 ) \cdot W ^ { 2 } + ( \dot { D } + 1 ) \times W$ total parameters, N being the input size, W width, and D depth. To support consequential decision-making settings, we employ three widely used real-world datasets from the counterfactual explanations literature: Adult (d = 51) (Adu96), COMPAS (d = 7) (Lar+16a), and Credit (d = 20) (BL13). Finally, all approaches are evaluated and compared on their optimality of distance, coverage, and runtime efficiency over a total of 500 instances. All implementations of the approaches will be shared publicly.

## 3.5.1 Performance of the MIP-framework

In the first set of experiments, we aim to showcase the ability of the proposed MIP-based approaches (i.e., MIP-SAT, MIP-EXP, MIP-OBJ) in diverse settings. Specifically, we generate CFEs for a two-layer ReLU-activated NN with 10 neurons in each layer and evaluate generated counterfactual explanations using the metrics above on three datasets and four norm distances: $\ell _ { 0 } , \ell _ { 1 } , \ell _ { 2 } , \ell _ { \infty }$ .

As expected, the CFE distances for all presented methods are similar to those of MACE (SAT) (Kar+20a), which we use here as oracle, and coverage is perfect by design for all presented methods. Figure 3.2 presents a comparison of runtime for these methods, where we observe significant improvement in runtime compared to SAT-oracle. Similar comparison for distances may be found in Figure B.2 in the Appendix. Importantly, the presented MIP-based methods are able to generate CFEs in settings in which neither MACE (SAT) nor MIP-SAT are able (e.g., Adult or Credit dataset on $\ell _ { 2 }$ norm).

In a second experiment, we compare the proposed MIP-based approaches, not only with the SAT-oracle but also with DiCE (MST20) (i.e., gradient-based optimization) on the same NN model as above. Here we adapt our experimental setting to DiCE, as it only supports the $\ell _ { 1 } { \mathrm { - n o r m } }$ distance, and does not provide support for ordinal and real-valued features. Moreover, since DiCE assumes that the model has been trained using range-normalized data, we build additional support in our implementation to encode the normalization term in the MIP-based approaches, which in turn could negatively affect runtime and numeric stability. Nonetheless, in this setting, we observe in Figure 3.3 relatively smaller distances and significantly smaller runtimes for the former. Furthermore, where MIP-OBJ has perfect coverage by design, DiCE dips slightly below perfect coverage on the Adult dataset, failing to offer an explanation for 2/500 instances.

## 3.5.2 Scalablity Experiments

The experiments above were presented on NN models that were able to sufficiently discriminate between the classes of the supervised learning task (with test accuracy in the range of 67-82% for different datasets). Complementing the demonstrations above, we investigate the scalibility of our approaches for the sake of completeness. In this regard, Figure 3.5 (and Figure B.3 in the Appendix) compare the runtime, distance, and coverage for SMT-based (Kar+20a) and gradient-based (MST20) approaches with our proposed approaches for a NN model with growing width and/or depth (as well as growing input size by incorporating different datasets).

It can be seen that the SMT-based approaches quickly reach their limit while MIP-based and gradient-based approaches scale well with both increasing width and depth. As MIP-based approaches do not scale polynomially w.r.t. network size, they do not scale as well as the gradient-based DiCE (this can be seen for the bigger Credit and Adult datasets in Figure B.3 in the Appendix), however, they produce much smaller distances. While MIP-based approaches have perfect coverage and minimum distance in theory, in practice numerical instabilities may be incurred in the backend tool as the number of intermediate variables in the mixed-integer program becomes large and their relations become deep due to the nested nature of NNs (the analysis of such numerical instabilities is beyond the scope of this work and deferred for future work). This causes failure to generate explanations for some samples or an increase in distances. In this context, having two MIP-based approaches is beneficial to verify results–for example, MIP-EXP behaves more stable in terms of distances than MIP-OBJ.

## 3.5.3 Qualitative Experiments

In this section, we show that how the expressiveness of SMT and MIP can be used to easily encode qualitative features and/or user-defined constraints for the explanations.

![image_07](images/image_07.png)

Figure 3.4: Scatter plots showing the diversity and proximity of sets of counterfactuals generated by our approach against DiCE along with runtimes. Diversity, distance, and runtime for generating sets of counterfactuals on the COMPAS dataset and NN model with two hidden layers of size 10. For each counterfactual set size $k \in [ 2 , 1 0 ]$ , each approach has been tested on 100 instances.

## 3.5.3.1 Diversity

We report on experiments showing the diversity feature of our approach as presented in the previous section, and compare against DiCE’s implementation of diversity.

We follow the authors of DiCE, and evaluate the k diversely generated CFEs by measuring the mean of pairwise distances among the CFEs (the higher the better):

$$
k - \text { diversity } (\{x _ {j} ^ {\mathrm{CFE}} \} _ {k}): \frac {1}{\binom {k} {2}} \sum_ {i = 1} ^ {k - 1} \sum_ {j = i + 1} ^ {k} \operatorname{dist} (x _ {i} ^ {\mathrm{CFE}}, x _ {j} ^ {\mathrm{CFE}}) \tag {3.14}
$$

Expectedly, diversity is traded-off with distance. Thus, in addition to the diversity metric above, the distance of the diverse set of CFEs to the original factual instance, $\mathbf { x } ^ { \mathsf { F } } .$ , is measured as follows (the lower the better):

$$
k - \text { distance } (\mathbf {x} ^ {\mathrm{F}}, \{x _ {j} ^ {\mathrm{CFE}} \} _ {k}): \frac {1}{k} \sum_ {i = 1} ^ {k} \operatorname{dist} (\mathbf {x} ^ {\mathrm{F}}, x _ {i} ^ {\mathrm{CFE}}) \tag {3.15}
$$

Figure 3.4 shows diversities generated by MIP-DIVERSE compared to DiCE for which the default hyperparameters are used. MIP-DIVERSE succeeds in finding the closest set of CFEs given a fixed distance threshold for diversity. The initial threshold has been set to 0.01 for this experiment, increasing it would result in the k−diversity and k−distance graph of Figure 3.4 to move upward, providing the possibility to choose the desired diversity-distance trade-off. Our results show that at a similar level of diversity $( \mathrm { i } . \mathrm { e } . , k = 6 )$ , the

![image_08](images/image_08.png)

Figure 3.5: Scatter and bar plots showing the runtimes and distances when the network architecture becomes wider or deeper. Scalability experiments comparing SMT-, MIP-, and gradient-based approaches on the COMPAS dataset. The upper row shows the results for increasing depth and the lower row for increasing width; both in terms of runtime and distance. For each approach and architecture 50 samples are evaluated, however, some fail to produce valid CFEs either because of imperfect coverage (i.e., DiCE) or numeric instabilities (i.e., MIP-OBJ and MIP-EXP); thus, only the instances for which all approaches have generated valid CFEs are included in the comparison. In general, for increasing depth, the average coverage across all the architectures is 99.1% and 93.7% for MIP-OBJ and MIP-EXP, and 96.4% for DiCE. For increasing width, the average coverage across all the architectures is 100% and 100% for MIP-OBJ and MIP-EXP, and 100% for DiCE. Similar experiments on the Credit and Adult datasets may be found in Figure B.3 in the Appendix.

counterfactual set of MIP-DIVERSE is much closer to the factual instance. As k increases further, in DiCE, while still a subset of the CFEs are diverse (and thus increase the average distance), the remaining ones are very similar to the previous as they minimally change a subset of the continuous variables. As a result, the average diversity and distance of the generated CFEs decreases. The runtimes of MIP-DIVERSE is again faster than the gradient-based opponent, however, MIP-DIVERSE is more sensitive to increasing the input size due to the added distance constraints, making it more or less as slow as DiCE on larger datasets.

## 3.5.3.2 Sparsity

As described in the previous section, maximizing the sparsity of explanations is equivalent to minimizing the $\ell _ { 0 }$ distance to the factual sample. To show the ability of our approach in maximizing sparsity, we refer the reader to the first column of figure B.2 in the Appendix where all approaches succeed in maximizing sparsity. Indeed, it would also be possible to optimize for a convex combination of $\ell _ { 0 }$ and e.g., $\ell _ { 1 }$ norms to generate more realistic sparse explanations that allow more features to vary while staying close to the factual sample.

We would like to also remark, once more, the role of the expressive power of SMTs and MIPs, in increasing the quality of explanations through handling different types of constraints. For example, defining different types of actionability on the features (e.g., increase/decrease-only, non-actionable, etc.) are as simple as adding a few inequality constraints to the MIP model. This ease of encoding may give stake-holders and explanation-providers the possibility to take into account individual-specific situations where an individual might ask for her personal constraints to be considered within the provided explanation.

## 3.6 conclusion and future work

In this work, we have proposed efficient approaches based on mixed-integer programming to generate counterfactual explanations with guarantees for the widely-used class of neural network models. We have empirically demonstrated the efficiency and guarantees of the proposed framework by comparing it, in terms of distance, runtime and coverage with previous SMT- and gradient-based approaches for CFE generation. We have also provided qualitative results on the generation of diverse counterfactuals, showing the flexibility of our approach, as well as efficiency in handling complex qualitative features.

As future work, we plan to explore other qualitative features, such as other plausibility constraints beyond data types and ranges. Moreover, although in this work we have focused on NN architectures with ReLU activations, similar approaches can be deployed for any piece-wise linear activation function (e.g., Max-Pooling). Moreover, other classes of models (e.g., Support Vector Machines with RBF kernel) could also be encoded or approximated by linear constraints, and thus be similarly handled by our MIP-framework. Finally, as stake-holders increasingly adopt more complex neural models for consequential decision-making, it becomes critical to have access to reliable and efficient tools to explain algorithmic decisions. Thus, as venue for future work, it would be interesting to further investigate the scalability and numeric stability issues, which also arise in the NN verification.
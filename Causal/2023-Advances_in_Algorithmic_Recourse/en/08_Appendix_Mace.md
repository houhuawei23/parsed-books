# Appendix Mace

## a.1 background on programming language and program verification

programs We assume given a set of function symbols with their arity. For simplicity, we consider the case where operators are untyped and have arity 0 (constants), 1 (unary functions), and 2 (binary functions). We let $c , c _ { 1 } ,$ , and $c _ { 2 }$ range over constants, unary functions and binary functions respectively. Expressions are built from function symbols and variables. The set of expressions is defined inductively by the following grammar:

$$
\begin{array}{c c c c} e & \therefore = & x & \text {variable} \\ & | & c & \text {constant} \\ & | & c _ {1} (e) & \text {unary function} \\ & | & c _ {2} (e _ {1}, e _ {2}) & \text {binary function} \end{array}
$$

We next assume given a set of atomic predicates. For simplicity, we also consider that predicates have arity 1 or 2, and let $P _ { 1 }$ and $P _ { 2 }$ range over unary and binary predicates respectively. We define guards using the following grammar:

$$
\begin{array}{c c c c} b & \therefore = & P _ {1} (e) & \text {   unary   predicate   } \\ & | & P _ {2} (e _ {1}, e _ {2}) & \text {   binary   predicate   } \\ & | & b _ {1} \& b _ {2} & \text {   conjunction   } \\ & | & b _ {1} \parallel b _ {2} & \text {   disjunction   } \\ & | & \neg b & \text {   negation   } \end{array}
$$

<!-- footnote -->

- Clearly, the average cost of recourse across groups can be the same, even if the proportion of individuals which are classified as positive or negative is very different across groups

<!-- footnote end -->

<!-- footnote -->

- This differs from the commonly-used purely predictive, statistical criterion of equal opportunity (HPS16).

<!-- footnote end -->

<!-- footnote -->

- We use v when there is an explicit distinction between the protected attribute and other features (in the context of fairness) and x otherwise (in the context of explainability).
- For simplicity, (5.2) assumes that all $X _ { i }$ are continuous; we do not make this assumption in the remainder of the present work.

<!-- footnote end -->

<!-- footnote -->

- For an interventional notion of recourse related to conditional average treatment effects (CATE) for a specific subpopulation, see (Kar+20b); in the present work, we focus on the individualised counterfactual notion of causal recourse.

<!-- footnote end -->

<!-- footnote -->

- After all, it is not much consolation for an individual who was unfairly given an unfavourable prediction to find out that other members of the same group were treated more favourably

<!-- footnote end -->

<!-- footnote -->

- All Appendix mentions refer to the arXiv version (Küg+22) containing the supplement of this work.

<!-- footnote end -->

<!-- footnote -->

- $^ 8 \mathrm { E . g . , }$ for interventions with minimum quantum size and a fixed budget, it makes sense to spread interventions across a randomly chosen subset since it is not possible to give everyone a very small amount, see $\mathrm { ( G H + 1 7 ) }$ for broader comments on the potential benefits of randomness in fairness. Note that $p = 1 .$ , i.e., deterministic interventions are included as a special case.

<!-- footnote end -->

<!-- footnote -->

- This chapter is based on the paper “On the Adversarial Robustness of Causal Algorithmic Recourse,” Dominguez-Olmedo, Karimi, Schölkopf, ICML (­), 2022. (DOKS22).

<!-- footnote end -->

<!-- footnote -->

- A large class of explanation methods rely on the gradients to offer saliency/attribution maps, especially in the image domain.
- Explanation models such as MACE (Kar+20a) provide optimal solutions, $\mathbf { x } _ { \epsilon } ^ { \mathsf { C F } } .$ , where $\begin{array} { r } { h ( { \mathbf x } ^ { \mathsf { F } } ) \hat { \neq } h ( x _ { \epsilon } ^ { \mathsf { C F } } ) , \Delta ( { \mathbf x } ^ { \mathsf { F } } , { \mathbf x } _ { \epsilon } ^ { \mathsf { C F } } ) \leq \Delta ( { \mathbf x } ^ { \mathsf { F } } , { \mathbf x } ^ { * \mathsf { C F } } ) + \epsilon , } \end{array}$ where $\mathbf { \hat { x } } ^ { * \complement F }$ is the optimal nearest contrastive explanation. In practice, $\epsilon = 1 e - 5$ which in turn results in ${ \bf x } _ { \epsilon } ^ { \tt C F } \approx \stackrel { * } { { \bf x } } ^ { \tt C F }$ .

<!-- footnote end -->

We next define commands. These include assignments, conditionals, bounded loops and return expressions. The set of commands is defined inductively by the following grammar:

```txt
c ;:= skip no-op
| x := e assignment
| c1; c2 sequential composition
| if b then c1 else c2 conditionals
| for (i = 1, ..., n) do c for loop
| return e return statement
```

We assume that programs satisfy a well-formedness condition. The condition requires that return expressions have no successor instruction, i.e. we do not allow commands of the form return e; c or if b then c; return e else $c ^ { \prime } ; c ^ { \prime \prime }$ . This is without loss of generally, since commands can always be transformed into functionally equivalent programs which satisfy the well-formedness condition.

Single assignment form Our first step to construct characteristic formulae is to transform programs in an intermediate form that is closer to logic. Without loss of generality, we consider loop-free commands, since loops can be fully unrolled. The intermediate form is called a variant of the well-known SSA form (RWZ88; Cyt+91) from compiler optimization. Concretely, we transform programs into some weak form of single assignment. This form requires that every non-input variable is defined before being used, and assigned at most once during execution for any fixed input. The main difference with SSA form is that we do not use so-called ϕ-nodes, as we require that variables are assigned at most once for any fixed input. More technically, our transformation can be seen as a composition of SSA transform with a naive de-SSA transform where ϕ-nodes are transformed into assignments in the branches of the conditionals.

path formulae and characteristic formulae Our second step is to define the set of path formulae. Informally, a path formula represents a possible execution of the program. Fix a distinguished variable y for return values. Then the path formulae of a command c is defined inductively by the clauses:

$$
\mathrm{PF} _ {z := e} (y) = \{z = e \}
$$

$$
\mathrm{PF} _ {c _ {1}; c _ {2}} (y) = \{\phi_ {1} \wedge \phi_ {2} \mid \phi_ {1} \in \mathrm{PF} _ {c _ {1}} (y) \wedge
$$

$$
\phi_ {2} \in \mathrm{PF} _ {c _ {2}} (y) \}
$$

$$
\mathrm{PF} _ {\text { if   } b \text {   then   } c _ {1} \text {   else   } c _ {2}} (y) = \left\{b \wedge \phi_ {1} \mid \phi_ {1} \in \mathrm{PF} _ {c _ {1}} (y) \right\} \cup
$$

$$
\{\neg b \land \phi_ {2} \mid \phi_ {2} \in \mathrm{PF} _ {c _ {2}} (y) \}
$$

$$
\mathrm{PF} _ {\text { return } e} (y) = \{y = e \}
$$

The characteristic formula $\phi _ { c }$ of a command c is then defined as:

$$
\bigvee_ {\phi \in \mathrm{PF} _ {c} (y)} \phi
$$

One can prove that for every inputs $x _ { 1 } , \ldots , x _ { n } $ , the formula $\phi _ { y } ( x _ { 1 } , \dots , x _ { n } , v )$ is valid iff the execution of $c$ on inputs $x _ { 1 } , \ldots , x _ { n }$ returns v. Note that, strictly speaking, the formula $\phi _ { y }$ contains as free variables the distinguished variable $y ,$ the inputs $x _ { 1 } , \ldots , x _ { n }$ of the program, and all the program variables, say $z _ { 1 } \ldots z _ { m }$ . However, the latter are fully defined by the characteristic formula so validity of $\phi _ { y } ( x _ { 1 } , \dots , x _ { n } , v )$ is equivalent to validity of $\exists z _ { 1 } \dots z _ { m } . \phi _ { y } \bigl ( x _ { 1 } , \dots , x _ { n } , v \bigr )$ .

## a.2 experiment details

In this section we provide further details on the detasets and methods used in or experiments, together with some additional results.

## a.2.1 Model Selection

To demonstrate the flexibility of our approach, we explored four different differentiable and non-differentiable model classes, i.e., decision tree, random forest, logistic regression and multilayer perceptron (MLP). As the main focus of our work is to generate counterfactuals for a broad range of already trained models, we opted for models’ parametrization that result in good performance on the considered datasets (e.g., default parameters). For instance, for the MLP, we opted for two hidden layers with 10 neurons, since it present better performance in the Adult dataset (%82.52/%81.94 training/test accuracy) than other architectures with hidden = 100 (default) and hidden = 100, 100 which result in %81.69/%81.06 and %81.51/%80.82 training/test accuracy, respectively. We leave the exploration of other datasets (larger feature spaces), more complex models (deeper MLPs) and other SMT solvers as future work.

## a.2.2 Datasets

Here we detail the different types of variables present in each dataset. We used the default features for the Adult and COMPAS datasets, and applied the same preprocessing used in (USL19) for the Credit dataset. All samples with missing data were dropped. We remark that we have relied on broadly studied datasets in the literature on fairness and interpretability of ML for consequential decision making. For instance, the Credit dataset [34] (n = 29, 623, d = 14) has been previously studied by the Actionable Recourse work [29], and the Adult [1] (n = 45, 222, d = 12, d(one-hot) = 51) and COMPAS [18] (n = 5, 278, d = 5, d(one-hot) = 7) have been previously used in the context of fairness in ML [Joseph et al., 2016; Zafar et al., 2017; Agarwal et al. 2018].

Adult (n = 45, 222, d = 12, d(one-hot) = 51):

• Integer: Age, Education Number, Hours Per Week

• Real: Capital Gain, Capital Loss

• Categorical: Sex, Native Country, Work Class, Marital Status, Occupation, Relationship

• Ordinal: Education Level

Credit (n = 29, 623, d = 14, d(one-hot) = 20):

• Integer: Total Overdue Counts, Total Months Overdue, Months With Zero Balance Over Last 6 Months, Months With Low Spending Over Last 6 Months, Months With High Spending Over Last 6 Months
• Real: Max Bill Amount Over Last 6 Months, Max Payment Amount Over Last 6 Months, Most Recent Bill Amount, Most Recent Payment Amount
• Categorical: Is Male, Is Married, Has History Of Overdue Payments
• Ordinal: Age Group, Education Level

COMPAS (n = 5, 278, d = 5, d(one-hot) = 7):

• Integer: -

• Real: Priors Count
• Categorical: Race, Sex, Charge Degreee
• Ordinal: Age Group

## a.2.3 Handling Mixed Data Types

While the proposed approach (MACE) naturally handles mixed data types, other approaches do not. Specifically, the Feature Tweaking method generates counterfactual explanations for Random Forest models trained on non-hot embeddings of the dataset, meaning that the resulting counterfactuals will not have multiple categories of the same variable activated at the same time. However, because this method is only restricted to working with real-valued variables, the resulting counterfactual is must undergo a post-processing step to ensure integer-, categorical-, and ordinal-based variables are plausible in the counterfactual. The Actionable Recourse method, on the other hand, explanations for Logistic Regression models trained on one-hot embeddings of the dataset, hence requiring additional constraints to ensure that multiple categories of a categorical variable are not simultaneously activated in the counterfactual. While the authors suggest how this can be supported using their method, their open-source implementation converts categorical columns to binary where possible and drops other more complicated categorical columns, postponing to future work. Furthermore, the authors state that the question of mutually exclusive features will be revisited in later releases 1. Moreover, ordinal variables are not supported using this method. The overcome these shortcomings, the counterfactuals generated by both approaches is post-processed to ensure correctness of variable types by rounding integer-based variables, and taking the maximally activated category as the counterfactual category.

## a.3 additional results

## a.3.1 Comprehensive Distance Results

Following the presentation of coverage Ω results in Table 2.2 and relative distance δ improvement (reduction) in Table 2.3 of the main body, in Figure A.1 we present the complete distribution of counterfactual distances upon termination of Algorithm 1. Importantly, we see that in all setups (approaches × models × norms × datasets), MACE results are at least as good as any other approach (MO, PFT, AR).

Table A.1: Wall-clock time (seconds) for computing the nearest counterfactual explanation (without constraints). $N = \Omega _ { \mathrm { M A C E } } \cap \Omega _ { \mathrm { O t h e r } }$ factual samples; cells are shaded for unsupported tests. Lower run-time is better. The run-time for MACE depends on $O ( \log ( 1 / \epsilon ) )$ , i.e., orders of magnitude more accuracy only cost linearly more runtime. These results should be considered along Tables 2.2, 2.3 comparing coverage Ω and distance δ.

<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">Adult</td><td colspan="3">Credit</td><td colspan="3">COMPAS</td></tr><tr><td> $\ell_0$ </td><td> $\ell_1$ </td><td> $\ell_\infty$ </td><td> $\ell_0$ </td><td> $\ell_1$ </td><td> $\ell_\infty$ </td><td> $\ell_0$ </td><td> $\ell_1$ </td><td> $\ell_\infty$ </td></tr><tr><td rowspan="5">tree</td><td>MACE ( $\epsilon = 10^{-1}$ )</td><td> $5.65 \pm 2.18$ </td><td> $3.01 \pm 0.74$ </td><td> $3.47 \pm 0.93$ </td><td> $3.48 \pm 1.25$ </td><td> $3.44 \pm 1.70$ </td><td> $2.39 \pm 0.64$ </td><td> $2.41 \pm 1.06$ </td><td> $1.22 \pm 0.36$ </td><td> $1.62 \pm 0.78$ </td></tr><tr><td>MACE ( $\epsilon = 10^{-3}$ )</td><td> $17.59 \pm 4.87$ </td><td> $9.58 \pm 3.05$ </td><td> $10.43 \pm 2.98$ </td><td> $15.84 \pm 4.78$ </td><td> $7.55 \pm 3.44$ </td><td> $4.44 \pm 2.20$ </td><td> $7.07 \pm 2.09$ </td><td> $5.72 \pm 1.28$ </td><td> $4.99 \pm 1.80$ </td></tr><tr><td>MACE ( $\epsilon = 10^{-5}$ )</td><td> $35.32 \pm 14.07$ </td><td> $20.35 \pm 6.34$ </td><td> $20.44 \pm 9.55$ </td><td> $25.47 \pm 8.71$ </td><td> $18.46 \pm 6.24$ </td><td> $10.58 \pm 6.36$ </td><td> $13.49 \pm 6.44$ </td><td> $9.22 \pm 4.21$ </td><td> $10.76 \pm 4.60$ </td></tr><tr><td>MO</td><td> $1.04 \pm 0.26$ </td><td> $0.85 \pm 0.27$ </td><td> $0.87 \pm 0.22$ </td><td> $0.53 \pm 0.15$ </td><td> $0.64 \pm 0.26$ </td><td> $0.54 \pm 0.23$ </td><td> $0.15 \pm 0.07$ </td><td> $0.12 \pm 0.06$ </td><td> $0.16 \pm 0.07$ </td></tr><tr><td>PFT</td><td></td><td></td><td></td><td> $1.45 \pm 0.42$ </td><td> $1.50 \pm 0.36$ </td><td> $1.91 \pm 0.79$ </td><td> $0.12 \pm 0.05$ </td><td> $0.13 \pm 0.06$ </td><td> $0.12 \pm 0.05$ </td></tr><tr><td rowspan="5">forest</td><td>MACE ( $\epsilon = 10^{-1}$ )</td><td> $27.98 \pm 9.48$ </td><td> $17.68 \pm 4.82$ </td><td> $19.05 \pm 6.11$ </td><td> $28.12 \pm 9.31$ </td><td> $21.88 \pm 10.04$ </td><td> $21.47 \pm 11.07$ </td><td> $8.07 \pm 3.36$ </td><td> $3.18 \pm 1.15$ </td><td> $3.52 \pm 1.93$ </td></tr><tr><td>MACE ( $\epsilon = 10^{-3}$ )</td><td> $69.19 \pm 15.76$ </td><td> $55.79 \pm 15.78$ </td><td> $52.31 \pm 15.39$ </td><td> $57.29 \pm 26.69$ </td><td> $40.75 \pm 17.85$ </td><td> $26.21 \pm 11.71$ </td><td> $15.05 \pm 5.15$ </td><td> $10.75 \pm 3.03$ </td><td> $8.53 \pm 3.55$ </td></tr><tr><td>MACE ( $\epsilon = 10^{-5}$ )</td><td> $89.81 \pm 28.99$ </td><td> $84.89 \pm 35.14$ </td><td> $78.49 \pm 23.85$ </td><td> $107.83 \pm 52.32$ </td><td> $90.04 \pm 38.02$ </td><td> $72.38 \pm 37.77$ </td><td> $33.26 \pm 9.79$ </td><td> $19.95 \pm 10.03$ </td><td> $17.22 \pm 7.90$ </td></tr><tr><td>MO</td><td> $1.14 \pm 0.35$ </td><td> $0.98 \pm 0.25$ </td><td> $0.94 \pm 0.36$ </td><td> $0.80 \pm 0.27$ </td><td> $0.80 \pm 0.35$ </td><td> $0.80 \pm 0.28$ </td><td> $0.16 \pm 0.06$ </td><td> $0.17 \pm 0.08$ </td><td> $0.15 \pm 0.07$ </td></tr><tr><td>PFT</td><td></td><td></td><td></td><td> $13.41 \pm 7.09$ </td><td> $10.46 \pm 4.67$ </td><td> $11.79 \pm 6.51$ </td><td> $1.93 \pm 0.81$ </td><td> $2.11 \pm 1.07$ </td><td> $1.83 \pm 0.87$ </td></tr><tr><td rowspan="5">lr</td><td>MACE ( $\epsilon = 10^{-1}$ )</td><td> $0.85 \pm 0.29$ </td><td> $0.66 \pm 0.26$ </td><td> $0.74 \pm 0.29$ </td><td> $0.33 \pm 0.15$ </td><td> $1.17 \pm 1.79$ </td><td> $0.49 \pm 0.30$ </td><td> $0.21 \pm 0.10$ </td><td> $0.19 \pm 0.10$ </td><td> $0.22 \pm 0.11$ </td></tr><tr><td>MACE ( $\epsilon = 10^{-3}$ )</td><td> $2.22 \pm 0.86$ </td><td> $3.55 \pm 1.50$ </td><td> $5.15 \pm 3.51$ </td><td> $0.87 \pm 0.20$ </td><td> $10.57 \pm 8.14$ </td><td> $6.11 \pm 3.51$ </td><td> $0.52 \pm 0.18$ </td><td> $0.31 \pm 0.12$ </td><td> $0.54 \pm 0.20$ </td></tr><tr><td>MACE ( $\epsilon = 10^{-5}$ )</td><td> $2.73 \pm 0.73$ </td><td> $6.60 \pm 3.01$ </td><td> $13.32 \pm 6.70$ </td><td> $1.19 \pm 0.56$ </td><td> $25.10 \pm 21.67$ </td><td> $16.21 \pm 8.84$ </td><td> $0.84 \pm 0.22$ </td><td> $0.72 \pm 0.28$ </td><td> $0.77 \pm 0.21$ </td></tr><tr><td>MO</td><td> $7.52 \pm 1.91$ </td><td> $6.62 \pm 1.73$ </td><td> $5.73 \pm 1.14$ </td><td> $1.86 \pm 0.82$ </td><td> $1.41 \pm 0.53$ </td><td> $1.69 \pm 0.79$ </td><td> $0.30 \pm 0.22$ </td><td> $0.25 \pm 0.12$ </td><td> $0.25 \pm 0.11$ </td></tr><tr><td>AR</td><td></td><td> $2.05 \pm 0.45$ </td><td> $1.86 \pm 0.03$ </td><td></td><td> $0.72 \pm 0.15$ </td><td> $0.66 \pm 0.07$ </td><td></td><td> $0.07 \pm 0.01$ </td><td> $0.06 \pm 0.01$ </td></tr><tr><td rowspan="4">mlp</td><td>MACE ( $\epsilon = 10^{-1}$ )</td><td> $2586 \pm 4523$ </td><td> $8070 \pm 5995$ </td><td> $5091 \pm 6616$ </td><td> $1743 \pm 4171$ </td><td> $3432 \pm 5615$ </td><td> $10309 \pm 10088$ </td><td> $59 \pm 53$ </td><td> $158 \pm 135$ </td><td> $90 \pm 90$ </td></tr><tr><td>MACE ( $\epsilon = 10^{-3}$ )</td><td> $4187 \pm 9899$ </td><td> $34101 \pm 26853$ </td><td> $7094 \pm 10919$ </td><td> $1703 \pm 5889$ </td><td> $3304 \pm 4944$ </td><td> $8689 \pm 11698$ </td><td> $79 \pm 55$ </td><td> $180 \pm 139$ </td><td> $122 \pm 103$ </td></tr><tr><td>MACE ( $\epsilon = 10^{-5}$ )</td><td> $5888 \pm 9760$ </td><td> $44470 \pm 39097$ </td><td> $19712 \pm 14117$ </td><td> $1901 \pm 4892$ </td><td> $4736 \pm 5080$ </td><td> $11129 \pm 9773$ </td><td> $100 \pm 56$ </td><td> $257 \pm 168$ </td><td> $203 \pm 149$ </td></tr><tr><td>MO</td><td> $6.66 \pm 2.17$ </td><td> $6.61 \pm 1.96$ </td><td> $6.40 \pm 1.60$ </td><td> $2.02 \pm 2.09$ </td><td> $2.43 \pm 0.41$ </td><td> $1.90 \pm 0.83$ </td><td> $0.35 \pm 0.12$ </td><td> $0.45 \pm 0.10$ </td><td> $0.32 \pm 0.09$ </td></tr></table>

## a.3.2 Quality vs Complexity

In the main text and in the previous section, we considered distance comparisons upon termination of Algorithm 1; in this section we explore the effect of the accuracy parameter ϵ jointly on quality (distance δ) and complexity (run-time τ) during execution of Algorithm 1. Importantly, the number of calls made to the SAT solver follows $O ( \log ( 1 / \epsilon ) )$ ), where ϵ is the desired the accuracy term, i.e., orders of magnitude more accuracy only cost linearly more SAT calls. The run-time of each call to the SAT solver is governed by a number of parameters, including the implementation details of the SAT solver2, the compute hardware3, among other factors. Clearly, a higher desired accuracy $( \mathrm { i . e . , } \epsilon \to 0 )$ ) will result in closer counterfactuals $( \delta \in [ \delta ^ { * } , \delta ^ { * } + \epsilon ] )$ ) at the cost of higher run-time (higher τ), while leaving the coverage Ω unchanged (remaining at 100%, by design). Figure A.2 depicts the average counterfactual distance and average run-time against the number of calls to the SAT solver, confirming the intuition above: not only does MACE always achieve a lower counterfactual distance4 upon termination, in many cases an early termination of MACE generates closer counterfactuals while also being less computationally demanding.

**Table A.2: Percentage of factual samples for which the nearest counterfactual sample requires a reduction in age for a random forest trained on the Adult dataset, and the corresponding increase in distance to nearest counterfactual when restricting the approaches not to reduce age: $1 0 0 \times \mathbb { E } [ \delta _ { \mathrm { r e s t r . } } / \delta _ { \mathrm { u n r e s t r . } } - 1 ]$ .**

<table><tr><td rowspan="2"></td><td colspan="2"> $\ell_0$ </td><td colspan="2"> $\ell_1$ </td><td colspan="2"> $\ell_\infty$ </td></tr><tr><td>% age-change</td><td>relative dist. increase</td><td>% age-change</td><td>relative dist. increase</td><td>% age-change</td><td>relative dist. increase</td></tr><tr><td rowspan="2">MACE ( $\epsilon = 10^{-5}$ )MO</td><td>3.6%</td><td>0%</td><td>7.4%</td><td>61.3%</td><td>34.2%</td><td>13.9%</td></tr><tr><td>24.6%</td><td>29.7%</td><td>34.6%</td><td>94.6%</td><td>34.2%</td><td>66.6%</td></tr></table>

In addition to studying the quality vs complexity tradeoff against number of calls to the SAT solver, in Table A.1 we compare final run-times (in seconds) upon-termination of Algorithm 1 for various setups. The results show that MACE takes less than 5 seconds for logistic regression; between 5 and 60 seconds for decision trees and random forests; and between one minute and three hours for the multilayer perceptron (outliers were not excluded in computed mean runtimes). In contrast, competing approaches (MO, PFT, AR) require at most 30 seconds to generate a counterfactual explanation, when possible (note that the coverage for AR and PFT is often significantly below 100%, and only MACE is able to generate counterfactuals for the multilayer perceptron; MO requires access to the training data as it searches through the training set for a counterfactual). We believe that this difference is compensated (at least for the decision tree, the random forest, and the logistic regression classifiers) by the main properties of MACE compared to previous works, i.e.: i) model-agnostic ({non-}linear, {non-}differentiable, {non-}convex); ii) data-agnostic (heterogeneous features); iii) provable closeness guarantees; and iv) 100% coverage, even under plausibility and diversity constraints. Regarding the results on MLPs, we are well aware of prior work that develops efficient SMT-based methods for verifying large deep neural networks (see formal verification of deep neural networks (Hua+17; Kat+17; Sin+19) and optimization modulo theories (NO06; ST12)); indeed we plan to leverage stateof-the-art tools to improve the efficiency of our implementation, in particular for MLP-based models. With the current implementation of MACE, our main goal was to explore the use of off-the-shelf SMT-solvers already available in Python to generate counterfactuals in a broad range of settings, justifying our lesser emphasis on efficiency.

In practice the choice of epsilon should reflect the desired distance granularity from the operator, the number and range of attributes in the data space, and the decided upon distance norm. For example, using the $\ell _ { 0 }$ norm, which tracks the number of attributes changed, the lowest achievable distance granularity is $1 / J$ where J is the data dimensionality. Therefore, choosing any $\epsilon < 1 / J$ is sufficient and will result in the optimal counterfactual for this choice of distance metric. As another example, for the continuous $\ell _ { 1 }$ norm, too much granularity may result in a lack of trust for the end-user – consider the adult dataset with account balance feature with range $R = \$ 50,000 ;$ choosing a fine granularity may result in a counterfactual that suggests that only a few dollars change in the account balance can flip the prediction (e.g., result in the approval of a loan). It is important to point out that this phenomenon is not a fault of the counterfactual generating method $( \mathrm { i . e . , M A C E } )$ ), but of the robustness of the underlying classifier and its decision boundary. While such an explanation may not be favorable for an end-user, it may assist a system administrator or model designer to assay the robustness and safety of their model prior to deployement.

## a.3.3 Additional Constrained Results

Following the study of counterfactuals that change or reduce age (Section 2.5), we regenerate counterfactual explanations for those samples for which age-reduction was required, with an additional plausibility constraint ensuring that the age shall not decrease. The results presented in Table A.2 show interesting results. Once again, we observe that the additional plausibility constraint for the age incurs significant increases in the distance of the nearest counterfactual – being, as expected, more pronounced for the $\ell _ { 1 }$ and the $\ell _ { \infty }$ norms. For the $\ell _ { 0 }$ norm, we find that for the 18 factual samples (i.e., $3 . 6 \% \times 5 0 0 )$ for which the unrestricted MACE required age-reduction, the addition of the no-age-reduction constraint results in counterfactuals at the same distance, while suggesting a change in work class (5/18) or education level (4/18) instead of changing age.

## a.3.4 Details on diverse counterfactuals example

In the main body, we described a scenario where a logistic regression model had predicted that a loan borrower, John, would default on his loan. Here is john’s complete feature list: John is a married male between 40-59 years of age with some university degree. Over the last 6 months, Max Bill Amount = 500.0, Max Payment Amount = 60.0, Months With Zero Balance = 0.0, Months With Low Spending = 0.0, Months With High Spending = 1.0. Furthermore, John has a history of overdue payments, his Most Recent Bill Amount = 370.0, and his Most Recent Payment Amount = 40.0

![image_26](images/image_26.png)

Figure A.1: Comparison of approaches for generating unconstrained counterfactual explanations for a (top to bottom) trained decision tree, random forest, logistic regression, and multilayer perceptron model. Here the distribution of distance δ is shown upon termination of Algorithm 1; lower distance is better. For each bar, $N = 5 0 0 \times \Omega$ from Table 2.2, and absent bars refer to $\Omega = 0 .$ . In all setups, MACE results are at least as good as any other approach.
# Appendix Causal Recourse
## c.1 proofs

## c.1.1 Proof of Proposition ??

Proposition ?? (GP-SCM Noise Posterior). Let $\{ { \bf x } ^ { i } \} _ { i = 1 } ^ { n }$ be an observational sample from (??). For each $r \in [ d ]$ with non empty parent set $| p a ( r ) | > 0$ , the posterior distribution of the noise vector $\mathbf { u } _ { r } = \left( u _ { r } ^ { 1 } , . . . , u _ { r } ^ { n } \right)$ , conditioned on $ { \mathbf { x } } _ { r } = ( x _ { r } ^ { 1 } , . . . , x _ { r } ^ { n } )$ and $\mathbf { X } _ { p a ( r ) } = \bigl ( \mathbf { x } _ { p a ( r ) } ^ { 1 } , . . . , \mathbf { x } _ { p a ( r ) } ^ { n } \bigr )$ , is given by

$$
\mathbf {u} _ {r} | \mathbf {X} _ {p a (r)}, \mathbf {x} _ {r} \sim \mathcal {N} \left(\sigma_ {r} ^ {2} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \mathbf {x} _ {r}, \sigma_ {r} ^ {2} \left(\mathbf {I} - \sigma_ {r} ^ {2} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1}\right)\right), \tag {C.1.1}
$$

where $\mathbf { K } : = \big ( k _ { r } \big ( \mathbf { x } _ { p a ( r ) } ^ { i } , \mathbf { x } _ { p a ( r ) } ^ { j } \big ) \big ) _ { i j }$ denotes the Gram matrix.

Proof. First, note that, by definition, ${ \bf u } _ { r }$ is independent of $\mathbf { f } _ { r } = ( f _ { r } ( \mathbf { x } _ { \mathsf { p a } ( r ) } ^ { 1 } ) , . . . , f _ { r } ( \mathbf { x } _ { \mathsf { p a } ( r ) } ^ { n } ) )$ ) given $\mathbf { X } _ { \mathsf { p a } ( r ) }$ . Moreover, it follows from the assumed GP-SCM model in (??) and Definition $? ? ,$ as well as properties of the GP prior, that both are multivariate Gaussian random variables with distributions given by

$$
\mathbf {u} _ {r} \sim \mathcal {N} (\mathbf {0}, \sigma_ {r} ^ {2} \mathbf {I}) \quad \text { independently   of } \quad \mathbf {X} _ {p a (r)}, \quad \text { and } \tag {C.1.1}
$$

$$
\mathbf {f} _ {r} | \mathbf {X} _ {p a (r)} \sim \mathcal {N} (\mathbf {0}, \mathbf {K}), \tag {C.1.2}
$$

where 0 denotes the zero vector (or matrix, see below) and K is as defined in Proposition ??.

Since independent multivariate Gaussian random variables are jointly multivariate Gaussian, we thus have

$$
\binom {\mathbf {u} _ {r}} {\mathbf {f} _ {r}} \left| \mathbf {X} _ {\mathrm{pa} (r)} \right. \sim \mathcal {N} (\mathbf {0}, \Sigma), \quad \text { where } \quad \Sigma = \left( \begin{array}{c c} \sigma_ {r} ^ {2} \mathbf {I} & \mathbf {0} \\ \mathbf {0} & \mathbf {K} \end{array} \right) \tag {C.1.3}
$$

Noting that ${ \bf x } _ { r } = { \bf f } _ { r } + { \bf u } _ { r }$ and applying a linear transformation to $\left( \mathbf { C . I . 3 } \right)$ , we then obtain

$$
\binom {\mathbf {u} _ {r}} {\mathbf {x} _ {r}} \left| \mathbf {X} _ {\mathrm{pa} (r)} = \left( \begin{array}{c c} \mathbf {I} & \mathbf {0} \\ \mathbf {I} & \mathbf {I} \end{array} \right) \binom {\mathbf {u} _ {r}} {\mathbf {f} _ {r}} \right| \mathbf {X} _ {\mathrm{pa} (r)} \sim \mathcal {N} (\mathbf {0}, \tilde {\boldsymbol {\Sigma}}) \tag {C.1.4}
$$

$$
\text { where } \quad \tilde {\Sigma} = \left( \begin{array}{c c} \sigma_ {r} ^ {2} \mathbf {I} & \sigma_ {r} ^ {2} \mathbf {I} \\ \sigma_ {r} ^ {2} \mathbf {I} & \mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I} \end{array} \right).
$$

Conditioning on $\mathbf { x } _ { r }$ and using the conditioning formula (e.g., Tou11), the result follows:

$$
\mathbf {u} _ {r} \left| \mathbf {X} _ {p a (r)}, \mathbf {x} _ {r} \right. \sim \mathcal {N} \left(\mathbf {0} + \sigma_ {r} ^ {2} \mathbf {I} \left(\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}\right) ^ {- 1} \left(\mathbf {x} _ {r} - \mathbf {0}\right), \sigma_ {r} ^ {2} \mathbf {I} - \sigma_ {r} ^ {2} \mathbf {I} \left(\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}\right) ^ {- 1} \sigma_ {r} ^ {2} \mathbf {I}\right) \tag {C.1.5}
$$

$$
\sim \mathcal {N} \left(\sigma_ {r} ^ {2} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \mathbf {x} _ {r}, \sigma_ {r} ^ {2} \left(\mathbf {I} - \sigma_ {r} ^ {2} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1}\right)\right) \tag {C.1.6}
$$

## c.1.2 Proof of Proposition ??

Proposition ?? (GP-SCM Counterfactual Distribution). Let $\{ { \bf x } ^ { i } \} _ { i = 1 } ^ { n }$ be an observational sample from (??). Then, for $r \in [ d ]$ with $| p a ( r ) | > 0 ,$ , the counterfactual distribution over $X _ { r }$ had $\mathbf { X } _ { p a ( r ) }$ been $\tilde { \mathbf { x } } _ { p a ( r ) }$ (instead of $\mathbf { x } _ { p a ( r ) } ^ { F } )$ for individual $\mathbf { x } ^ { F } \in \{ \mathbf { x } ^ { i } \} _ { i = 1 } ^ { n }$ is given by

$$
\mathrm{X} _ {r} \left(\mathbf {X} _ {p a (r)} = \tilde {\mathbf {x}} _ {p a (r)}\right) \mid \mathbf {x} ^ {F}, \left\{\mathbf {x} ^ {i} \right\} _ {i = 1} ^ {n} \tag {C.1.7}
$$

$$
\sim \mathcal {N} \big (\mu_ {r} ^ {F} + \tilde {\mathbf {k}} ^ {T} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \mathbf {x} _ {r}, s _ {r} ^ {F} + \tilde {k} - \tilde {\mathbf {k}} ^ {T} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \tilde {\mathbf {k}} \big),
$$

where $\tilde { k } : = k _ { r } ( \tilde { \mathbf { x } } _ { p a ( r ) } , \tilde { \mathbf { x } } _ { p a ( r ) } ) , \tilde { \mathbf { k } } : = \big ( k _ { r } ( \tilde { \mathbf { x } } _ { p a ( r ) } , \mathbf { x } _ { p a ( r ) } ^ { 1 } ) , \dots , k _ { r } ( \tilde { \mathbf { x } } _ { p a ( r ) } , \mathbf { x } _ { p a ( r ) } ^ { n } ) \big )$ , xr and K as defined in $? ? ,$ and $\mu _ { r } ^ { F }$ and $s _ { r } ^ { F }$ are the posterior mean and variance of $u _ { r } ^ { F }$ given by (??).

Proof. We follow the three steps of abduction, action, and prediction for computing counterfactual distributions (see § 4.2.2 for more details). Starting from the factual observation $\mathbf { x } ^ { \mathsf { F } } \in \{ x ^ { i } \} _ { i = 1 } ^ { n }$ generated according to

$$
\mathbf {x} _ {r} ^ {\mathsf {F}} := f _ {r} (\mathbf {x} _ {\mathrm{pa} (r)} ^ {\mathsf {F}}) + u _ {r} ^ {\mathsf {F}}, \tag {C.1.7}
$$

we first compute the noise posterior (abduction). According to Proposition ?? it is given by a marginal of (??), i.e.,

$$
u _ {r} ^ {\mathsf {F}} | \mathbf {X} _ {\mathrm{pa} (r)}, \mathbf {x} _ {r} \sim \mathcal {N} (\mu_ {r} ^ {F}, s _ {r} ^ {\mathsf {F}}) \tag {C.1.8}
$$

where $\mu _ { r } ^ { \mathsf { F } }$ is given by element F of the mean vector

$$
\boldsymbol {\mu} _ {r} = \sigma_ {r} ^ {2} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \mathbf {x} _ {r} \tag {C.1.9}
$$

and $s _ { r } ^ { \mathsf { F } }$ is given by element (F, F) of the covariance matrix

$$
S _ {r} = \sigma_ {r} ^ {2} \left(\mathbf {I} - \sigma_ {r} ^ {2} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1}\right) \tag {C.1.10}
$$

of the noise posterior given by (??).

Next, we simulate the hypothetical intervention by updating the structural equation (C.1.7) (action step),

$$
x _ {r} ^ {\mathsf {F}} \left(\mathbf {X} _ {\mathrm{pa} (r)} = \tilde {\mathbf {x}} _ {\mathrm{pa} (r)}\right) := f _ {r} \left(\tilde {x} _ {\mathrm{pa} (r)}\right) + u _ {r} ^ {\mathsf {F}}. \tag {C.1.11}
$$

The GP predictive posterior at the new input $\tilde { x } _ { \mathrm { p a } ( r ) }$ has distribution (see, e.g., WR06),

$$
f _ {r} (\tilde {x} _ {\mathrm{pa} (r)}) | \mathbf {X} _ {\mathrm{pa} (r)}, \mathbf {x} _ {r} \sim \mathcal {N} (\tilde {\mathbf {k}} ^ {T} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \mathbf {x} _ {r}, \tilde {k} - \tilde {\mathbf {k}} ^ {T} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \tilde {\mathbf {k}}). \tag {C.1.12}
$$

Substituting (C.1.12) and (C.1.8) into (C.1.11) and noting that the sum of two Gaussians is again Gaussian with mean and variance equal to the sums of means and variances of the two individual Gaussians (prediction step) completes the proof. □

## c.1.3 Proof of Proposition ??

Proposition ??. Subject to causal sufficiency, PXd( )|do(X :=θ),xF $P _  \mathbf { X } _ { d ( \mathcal { T } ) } | \mathbf { d o } ( \mathbf { X } _ { \mathcal { T } } : = \pmb { \theta } ) , \pmb { x } _ { n d ( \mathcal { T } ) } ^ { F }$ is observation-Ially identifiable (i.e., computable from the observational distribution) via:

$$
p \left(\mathbf {X} _ {d (\mathcal {I})} \mid \mathrm{do} \left(\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta}\right), \mathbf {x} _ {n d (\mathcal {I})} ^ {F}\right) = \prod_ {r \in d (\mathcal {I})} p \left(X _ {r} \mid \mathbf {X} _ {p a (r)}\right) \Bigg | _ {\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta}, \mathbf {X} _ {n d (\mathcal {I})} = \mathbf {x} _ {n d (\mathcal {I})} ^ {F}}. \tag {C.1.13}
$$

Proof. This is a direct consequence of the properties of causally sufficient (Markovian) causal models, but we include a derivation for completeness. Recall that P factorises over its underlying causal graph as follows,

$$
p (\mathbf {X}) = \prod_ {r \in [ d ]} p (X _ {r} | \mathbf {X} _ {\mathrm{pa} (r)}). \tag {C.1.13}
$$

This joint distribution is transformed by the intervention do $( \mathbf { X } _ { \mathcal { T } } : = \theta )$ as follows,

$$
P (\mathbf {X} _ {- \mathcal {I}}, \mathrm{do} (\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta})) = \delta (\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta}) \prod_ {r \in [ d ] \backslash \mathcal {I}} P (X _ {r} | \mathbf {X} _ {\mathrm{pa} (r)}). \tag {C.1.14}
$$

Splitting the non-intervened variables into descendants $\mathbf { d } ( \mathcal { T } )$ and non-descendants nd( ), and conditioning on the intervened variables do $( \mathbf { X } _ { \mathcal { I } } : = \pmb { \theta } )$ , we obtain

$$
P (\mathbf {X} _ {\mathrm{nd} (\mathcal {I})}, \mathbf {X} _ {\mathrm{d} (\mathcal {I})} | \mathrm{do} (\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta})) = \left. \left(\prod_ {r \in \mathrm{nd} (\mathcal {I}) \cup \mathrm{d} (\mathcal {I})} P (X _ {r} | \mathbf {X} _ {\mathrm{pa} (r)})\right) \right| _ {\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta}}. \tag {C.1.15}
$$

As the non-descendants ${ \mathbf { \boldsymbol { X } } } _ { \mathrm { \Pi \Pi \times d ( \mathcal { T } ) } }$ are, by their very definition, not affected by the intervention, we can write

$$
\begin{array}{l} P (\mathbf {X} _ {\mathrm{nd} (\mathcal {I})}, \mathbf {X} _ {\mathrm{d} (\mathcal {I})} | \mathrm{do} (\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta})) = \\ \left(\prod_ {r \in \mathrm{d} (\mathcal {I})} P (X _ {r} | \mathbf {X} _ {\mathrm{pa} (r)})\right) \Bigg | \mathbf {x} _ {\mathcal {I} := \boldsymbol {\theta}} \prod_ {r \in \mathrm{nd} (\mathcal {I})} P (X _ {r} | \mathbf {X} _ {\mathrm{pa} (r)}). \\ \end{array}
$$

We can thus condition on a particular value of $\mathbf { X } _ { \mathrm { n d } ( \mathbb { Z } ) }$ to obtain

$$
\begin{array}{l} P \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} | \mathrm{do} (\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta}), \mathbf {X} _ {\mathrm{nd} (\mathcal {I})} = \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) = \\ \left(\prod_ {r \in \mathrm{d} (\mathcal {I})} P (X _ {r} | \mathbf {X} _ {p a (r)})\right) \bigg | _ {\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta}, \mathbf {X} _ {\mathrm{nd} (\mathcal {I})} = \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}} \tag {C.1.16} \\ \end{array}
$$

![image_31](images/image_31.png)

## c.2 additional results

This section presents additional results complementing those from Section ??. Table C.1 presents results that mirror those in Table ??, where the brute-force approach discussed at the beginning of Appendix C.5 is used instead of the gradient-based optimisation. Here, each real-valued feature was discretised into 20 bins within the range of its observed values in the training dataset.

Fig. C.1 mirrors the results in Fig. ??, for which a snapshot $( \gamma _ { \mathrm { L C B } } = 2 . 5 )$ is also provided in Table ??. Here we show the trade-off between validity and cost by varying the values of $\gamma _ { \mathrm { L C B } } ,$ , using as trained classifiers a nonlinear multilayer perceptron (MLP) in (a) and a non-differentiable random forest classifer in (b). Note that optimisation for the latter can only be done with the brute-force approach. All these additional results mostly confirm the insights presented in the main body.

Finally, Table C.2 provides a qualitative comparison of the proposed recourse approaches against the oracles and baselines in terms of their selection of intervention targets. We show empirically, on the three synthetic datasets, that cate approaches have more predictable behaviour, as they are less sensitive to model assumptions, and are thus more preferable for the individual seeking recourse under imperfect causal knowledge.

## c.3 (non-)identifability of scms under different assumptions

In general form, i.e., without any further assumption on the structural equations S or noise distribution $P _ { \mathbf { U } } ,$ , SCMs are not identifiable from data alone, meaning that there are multiple different SCMs (possibly with different underlying causal graphs) which imply the same observational distribution (PJS17). One possible construction relies on the use of the inverse cumulative distribution function (cdf) in combination with uniformly-distributed random variables (Dar51) and is also used in non-identifiability proofs for non-linear independent component analysis (ICA) (HP99). Even knowing the causal graph is generally not enough as summarised in the following proposition.

**Table C.1: Experimental results for the brute-force (20-bin discretization) approach on different 3-variable SCMs. We show average performance for $N _ { \mathrm { r u n s } } ~ = ~ 1 0 0 .$ , $N _ { \mathrm { M C - s a m p l e s } } = 1 0 0 ,$ , and $\gamma _ { \mathrm { L C B } } = 2$ . The relative trends reflect those in Table ??.**

<table><tr><td rowspan="2">Method</td><td colspan="3">LINEAR SCM</td><td colspan="3">NON-LINEAR ANM</td><td colspan="3">NON-ADDITIVE SCM</td></tr><tr><td>Valid $_{\star}$ (%)</td><td>LCB</td><td>Cost (%)</td><td>Valid $_{\star}$ (%)</td><td>LCB</td><td>Cost (%)</td><td>Valid $_{\star}$ (%)</td><td>LCB</td><td>Cost (%)</td></tr><tr><td> $\mathcal{M}_{\star}$ </td><td>100</td><td>-</td><td>11.0±5.6</td><td>100</td><td>-</td><td>20.7±11.0</td><td>100</td><td>-</td><td>15.8±8.9</td></tr><tr><td> $\mathcal{M}_{\text{LIN}}$ </td><td>100</td><td>-</td><td>11.3±5.8</td><td>60</td><td>-</td><td>19.9±8.9</td><td>92</td><td>-</td><td>17.0±10.4</td></tr><tr><td> $\mathcal{M}_{\text{KR}}$ </td><td>95</td><td>-</td><td>11.2±5.6</td><td>88</td><td>-</td><td>20.5±10.7</td><td>47</td><td>-</td><td>15.8±10.6</td></tr><tr><td> $\mathcal{M}_{\text{GP}}$ </td><td>100</td><td>.55±.04</td><td>11.6±5.8</td><td>99</td><td>.55±.04</td><td>21.2±10.9</td><td>88</td><td>.58±.05</td><td>16.8±10.3</td></tr><tr><td> $\mathcal{M}_{\text{CVAE}}$ </td><td>100</td><td>.55±.04</td><td>11.5±5.8</td><td>95</td><td>.55±.03</td><td>21.7±10.7</td><td>95</td><td>.59±.07</td><td>16.9±10.3</td></tr><tr><td> $\text{CATE}_{\star}$ </td><td>90</td><td>.57±.07</td><td>11.0±5.5</td><td>95</td><td>.55±.05</td><td>22.8±10.8</td><td>99</td><td>.57±.06</td><td>16.2±8.9</td></tr><tr><td> $\text{CATE}_{\text{GP}}$ </td><td>92</td><td>.56±.07</td><td>11.2±5.5</td><td>95</td><td>.55±.04</td><td>22.8±10.9</td><td>85</td><td>.58±.07</td><td>16.4±10.5</td></tr><tr><td> $\text{CATE}_{\text{CVAE}}$ </td><td>90</td><td>.57±.06</td><td>11.1±5.4</td><td>96</td><td>.55±.03</td><td>23.0±10.8</td><td>94</td><td>.59±.07</td><td>16.8±10.2</td></tr></table>

Proposition C.3.1. Even when the causal graph is known, the conditionals $P ( X _ { r } | \mathbf { X } _ { p a ( r ) } )$ alone are insufficient to uniquely determine the structural equations $X _ { r } : = \dot { f } _ { r } ( \dot { \mathbf { X } } _ { p a ( r ) } , U _ { r } )$ without further assumptions.

Proof. This can be shown by using the following argument from $\mathrm { J } \mathrm { S 1 0 , }$ , Footnote 1 (adapted to our notation):

$$
\begin{array}{l} \begin{array}{l} \text { "let U_{r} consist of (possibly uncountably many) real - valued random variables} \\ U _ {r} [ \mathbf {x} _ {p a (r)} ], \text { one for each value x_{pa(r)} of the parents X_{pa(r)}}. L e t U _ {r} [ \mathbf {x} _ {p a (r)} ] b e \\ d i s t r i b u t e d a c c o r d i n g t o P _ {X _ {r} | \mathbf {x} _ {p a (r)}} a n d f o n i e f r _ {r} (\mathbf {x} _ {p a (r)}, U _ {r}) := U _ {r} [ \mathbf {x} _ {p a (r)} ]. T h e n \end{array} \\ X _ {r} | \mathbf {X} _ {p a (r)} \text {   has   distribution   } P _ {X _ {r} | \mathbf {X} _ {p a (r)}}. \\ \end{array}
$$

We can now build on this formulation to construct a second SCM with the same observational distribution and causal graph, e.g., by shifting the noise variables and structural equations by some fixed constant C as follows.

For $r \in [ d ] .$ , define $Y _ { r } : = X _ { r } - C$ . Let $\tilde { U } _ { r }$ consist of (possibly uncountably many) real-valued random variables $\tilde { U } _ { r } [ { \bf { x } } _ { \mathrm { { p a } } ( r ) } ]$ , one for each value $\mathbf { x } _ { \mathrm { p a } ( r ) }$ of the parents $\mathbf { X } _ { \mathrm { p a } ( r ) }$ . Let $\tilde { U } _ { r } [ { \bf { x } } _ { \mathrm { { p a } } ( r ) } ]$ ] be distributed according to $P _ { Y _ { r } | \mathbf { x } _ { \mathrm { p a } ( r ) } }$ and define $f _ { r } ( \mathbf { x } _ { \mathrm { p a } ( r ) } , \tilde { U } _ { r } ) : = \tilde { U } _ { r } [ \mathbf { x } _ { \mathrm { p a } ( r ) } ] + C$ . Then $X _ { r } | \mathbf { X } _ { \mathrm { p a } ( r ) }$ also has distribution $P _ { X _ { r } | \mathbf { X } _ { p a ( r ) } } ,$ but for $C \neq 0$ the structural equations and noise distributions are different from the previous construction. □

In the case of the cvae-SCM model from (??) the setting is slightly less general than the above, since we additionally assume that: (i) the noise distributions are isotropic multivariate Gaussian distributions of fixed dimension, $\mathbf { z } _ { r } \sim \mathcal { N } _ { d _ { \mathbf { z } _ { r } } } ( \mathbf { 0 } , \mathbf { I } )$ ; and (ii) the structural equations $D _ { r }$ are from the class of functions that can be expressed as feedforward neural networks if fixed width and depth with learnable parameters $\psi _ { r }$ .

Unfortunately, we are not aware of any identifiability results for this particular setting, and further investigation into this matter is beyond the scope of the current work. It is interesting to note, however, that the cvae-SCM from (??) can be understood as a non-linear extension of the linear Gaussian model with equal error variances considered by $\mathrm { ( P B \mathrm { { } _ { 1 4 } ) } }$ , for which identifiability has been shown.

In general, there seem to be very few works addressing identifiability of SCMs in the non-linear case; we refer to $\mathrm { P J } 5 \mathbf { \pi } _ { \mathrm { I 7 } } , \ S 7 . \mathbf { \pi } _ { \mathrm { 1 } }$ for an overview of existing results. Of particular interest for our setting is the post-nonlinear model of (ZH09), which refers to the setting in which a non-linearity $g$ is applied on top of an ANM, i.e., $X _ { r } : = g _ { r } ( f _ { r } ( \mathbf { \tilde { X } } _ { \mathfrak { p a } ( r ) } ) + U _ { r } )$ , and for which complete conditions on $\left\{ f _ { r } , g _ { r } \right\}$ have been provided that lead to identifiability. Given the form of the decoders $D _ { r }$ —feedforward neural networks with stacked layers of simple non-linearities applied to linear transformations of the previous layers’ output—it may be possible that the cvae-SCM from (??) can be interpreted as a nested post-nonlinear model. We consider this an interesting direction, but leave further investigations into this matter for future work.

## c.4 further details on cvae training

To learn the cvae latent variable models, we perform amortised variational inference with approximate posteriors q parameterised by encoders $E _ { r }$ in the form of neural nets with parameters $\phi _ { r . }$ ,

$$
p _ {\psi_ {r}} \left(\mathbf {z} _ {r} \mid x _ {r}, \mathbf {x} _ {\mathrm{pa} (r)}\right) \approx q _ {\phi_ {r}} \left(\mathbf {z} _ {r} \mid x _ {r}, \mathbf {x} _ {\mathrm{pa} (r)}\right) := \mathcal {N} \left(\hat {\mu} _ {r}, \hat {\sigma} _ {r} ^ {2}\right), \tag {C.4.1}
$$

$$
(\hat {\mu} _ {r}, \hat {\sigma} _ {r} ^ {2}) := E _ {r} (x _ {r}, \mathbf {x} _ {\mathrm{pa} (r)}; \phi_ {r}).
$$

The training objective in form of the evidence lower bound (ELBO) given data $\{ \mathbf { x } ^ { i } \} _ { i = 1 } ^ { n }$ is given by

$$
\begin{array}{l} \mathcal {L} _ {r} \left(\psi_ {r}, \phi_ {r}\right) = \sum_ {i = 1} ^ {n} \mathbb {E} _ {q _ {\phi_ {r}} \left(\mathbf {z} \mid x _ {r} ^ {i}, \mathbf {x} _ {\mathrm{pa} (r)} ^ {i}\right)} \left[ \left\| x _ {r} ^ {i} - D _ {r} \left(\mathbf {x} _ {\mathrm{pa} (r)} ^ {i}, \mathbf {z}; \psi_ {r}\right) \right\| ^ {2} \right] \tag {C.4.2} \\ + \beta_ {r} D _ {\mathrm{KL}} \left(\left. q _ {\phi_ {r}} (\mathbf {z} | x _ {r} ^ {i}, \mathbf {x} _ {\mathrm{pa} (r)} ^ {i}) \right| \mid p (z)\right) \\ \end{array}
$$

We learn both $\psi _ { r }$ and $\phi _ { r }$ simultaneously via stochastic gradient descend on $\mathcal { L } _ { \boldsymbol { r } } ,$ with gradients computed by Monte Carlo sampling from $q _ { \phi _ { \uparrow } }$ with reparametrisation. Since the pairs of encoder and decoder parameters $\left( \psi _ { r } , \phi _ { r } \right)$ are independent for different $r ,$ this can be done in parallel.

## c.4.1 Hyperparameter selection for cvae training

A cvae model was trained for every $\mathbf { X } _ { r } | \mathbf { X } _ { \mathsf { p a } ( r ) }$ relation. Generally, hyperparameters were selected by comparing the distribution of real samples from the dataset against reconstructed samples from the trained cvae obtained by sampling noise from the prior. The selection of hyperparameters was done either manually, or by performing a grid search over various encoder and decoder architectures, latent-space dimensions, and values of the hyperparameters $\beta _ { r }$ that trade off the MSE and KL terms in the cvae objective (C.4.2). For the case of automatic selection, the setup resulting in the smallest maximum mean discrepancy (MMD) statistic (Gre+12) between real and reconstructed samples was chosen as hyperparameter configuration. Further details on the search space considered and the selected values are provided in Table $C . 3$ .

## c.5 experimental details, hyperparameter choices, and specification of scms

## c.5.1 Specification of SCMs used in our experiments

The following is a specification of all SCMs used in our experiments on synthetic and semi-synthetic data, both for data generation and to evaluate the validity of recourse actions proposed by the different approaches by computing the corresponding counterfactual in the ground-truth SCMs.

In addition, we also specify the model used to generate training labels. Note, however, that these labels are only used to train a new classifier (e.g., a logistic regression, multi-layer perceptron, or random forest) from scratch: this is the h(x) referred to in the main chapter. The label generating process is thus only used for obtaining labels to train a classifier on and is subsequently disregarded in favour of h.

In selecting the structural equations and label generating process, we tried to pick combinations that resulted in roughly centred features, as well as roughly balanced datasets (i.e., with a similar proportion of positive and negative training examples) that are not perfectly linearly-separable (i.e., with some class overlap). Moreover, we tried to select settings that result in a diverse set of intervention targets selected by the oracle for different factual instances, i.e., we try to avoid situations in which the optimal action is to always intervene on the same (set of) variable(s). To induce more interesting behaviour, we sample root nodes from mixtures of Gaussians.

## c.5.1.1 3-variable synthetic SCMs used for Table ??

A visual summary of the 3-variable synthetic SCMs used for Table ?? is provided in Fig. C.2.

linear scm: The linear 3-variable SCM consists of the following structural equations and noise distributions:

$$
X _ {1} := U _ {1}, \quad U _ {1} \sim \operatorname{MoG} \left(0. 5 \mathcal {N} (- 2, 1. 5) + 0. 5 \mathcal {N} (1, 1)\right) \tag {C.5.1}
$$

$$
X _ {2} := - X _ {1} + U _ {2}, \quad U _ {2} \sim \mathcal {N} (0, 1) \tag {C.5.2}
$$

$$
X _ {3} := 0. 0 5 X _ {1} + 0. 2 5 X _ {2} + U _ {3}, \quad U _ {3} \sim \mathcal {N} (0, 1) \tag {C.5.3}
$$

![image_32](images/image_32.png)

Figure C.2: Histograms and scatter plots of pairwise feature relations for the synthetic 3-variable SCMs.

non-linear anm: The non-linear 3-variable ANM consists of the following structural equations and noise distributions:

$$
X _ {1} := U _ {1}, \quad U _ {1} \sim \operatorname{MoG} \left(0. 5 \mathcal {N} (- 2, 1. 5) + 0. 5 \mathcal {N} (1, 1)\right) \tag {C.5.4}
$$

$$
X _ {2} := - 1 + \frac {3}{1 + e ^ {- 2 X _ {1}}} + U _ {2}, \quad U _ {2} \sim \mathcal {N} (0, 0. 1) \tag {C.5.5}
$$

$$
X _ {3} := - 0. 0 5 X _ {1} + 0. 2 5 X _ {2} ^ {2} + U _ {3}, \quad U _ {3} \sim \mathcal {N} (0, 1) \tag {C.5.6}
$$

non-additve scm: The non-additive 3-variable SCM consists of the following structural equations and noise distributions:

$$
X _ {1} := U _ {1}, \quad U _ {1} \sim \operatorname{MoG} \left(0. 5 \mathcal {N} (- 2. 5, 1) + 0. 5 \mathcal {N} (2. 5, 1)\right) \tag {C.5.7}
$$

$$
X _ {2} := 0. 2 5 \operatorname{sgn} (U _ {2}) X _ {1} ^ {2} (1 + U _ {2} ^ {2}), \quad U _ {2} \sim \mathcal {N} (0, 0. 2 5) \tag {C.5.8}
$$

$$
X _ {3} := - 1 + 0. 1 \operatorname{sgn} (U _ {3}) (X _ {1} ^ {2} + X _ {2} ^ {2}) + U _ {3}, \quad U _ {3} \sim \mathcal {N} (0, 0. 2 5 ^ {2}) \tag {C.5.9}
$$

label generation: For all 3-variable SCMs, labels Y were sampled according to

$$
Y \sim \text { Bernoulli } \left(\left(1 + e ^ {- 2. 5 \rho^ {- 1} (X _ {1} + X _ {2} + X _ {3})}\right) ^ {- 1}\right) \tag {C.5.10}
$$

where $\rho$ is the average of $\left( X _ { 1 } + X _ { 2 } + X _ { 3 } \right)$ across all training samples.

## c.5.1.2 7-variable semi-synthetic loan approval SCM used for Table ??

For the semi-synthetic dataset, we wanted to capture some relations between the involved variables that seemed somewhat intuitive to us and to some limited extent reflect a loan approval setting in the real-world:

• loan amount and duration being largest for mid-aged people who may want to build a house and start a family, and smaller for younger and older people;
• loan duration increasing with loan amount due to the an upper limit on monthly payments that can be afforded
• savings increasing once income passes a certain (minimal-sustenance) threshold;
• income increasing with age;
• education increasing with age initially before eventually saturating;
• gender differences in income and (access to) education due to existing gender-discrimination and inequality of opportunities in the population;

A visual summary of the 7-variable semi-synthetic loan SCMis shown in Fig. C.3.

![image_33](images/image_33.png)

Figure C.3: Histograms and scatter plots of pairwise feature relations for the semisynthetic loan SCM.

semi-synthetic scm: The loan approval SCM consists of the following structural equations and noise distributions:

$$
G := U _ {G}, \quad U _ {G} \sim \text { Bernoulli } (0. 5) \tag {C.5.11}
$$

$$
A := - 3 5 + U _ {A}, \quad U _ {A} \sim \text { Gamma } (1 0, 3. 5) \tag {C.5.12}
$$

$$
E := - 0. 5 + \left(1 + e ^ {- \left(- 1 + 0. 5 G + \left(1 + e ^ {- 0. 1 A}\right) ^ {- 1} + U _ {E}\right)}\right) ^ {- 1}, \quad U _ {E} \sim \mathcal {N} (0, 0. 2 5) \tag {C.5.13}
$$

$$
L := 1 + 0. 0 1 (A - 5) (5 - A) + G + U _ {L}, \quad U _ {L} \sim \mathcal {N} (0, 4) \tag {C.5.14}
$$

$$
D := - 1 + 0. 1 A + 2 G + L + U _ {D}, \quad U _ {D} \sim \mathcal {N} (0, 9) \tag {C.5.15}
$$

$$
I := - 4 + 0. 1 (A + 3 5) + 2 G + G E + U _ {I}, \quad U _ {I} \sim \mathcal {N} (0, 4) \tag {C.5.16}
$$

$$
S := - 4 + 1. 5 \mathbb {I} _ {\{I > 0 \}} I + U _ {S}, \quad U _ {S} \sim \mathcal {N} (0, 2 5) \tag {C.5.17}
$$

Note that variables in the above SCM often have a relative meaning in terms of deviation from the mean, e.g., we centre the Gamma-distributed age around its mean of 35, so that A has the meaning of “age-difference from the mean of $3 5 ^ { \prime \prime }$ (and similarly for other variables).

label generation: Labels Y were sampled according to

$$
Y \sim \text { Bernoulli } \left(\left(1 + e ^ {- 0. 3 (- L - D + I + S + I S)}\right) ^ {- 1}\right). \tag {C.5.18}
$$

Note that this label generation process only depends on loan duration and amount, income and savings, but not on gender, age or education level.

## c.6 derivation of a monte-carlo estimator for the gradient of the variance

We now derive an estimator for the gradient of the square-root of the variance (i.e., standard deviation) of h over the interventional or counterfactual distribution of $\mathbf { X } _ { \mathrm { d } ( \mathcal { I } ) }$ w.r.t. θ, which appears (multiplied by $\lambda _ { \mathrm { L C B } } )$ in the threshold tresh(a) of the optimisation constraint/regulariser.

First, we use the chain rule of differentiation to write

$$
\nabla_ {\boldsymbol {\theta}} \sqrt {\mathbb {V} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} , \boldsymbol {\theta} , \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) \right]} = \frac {\nabla_ {\boldsymbol {\theta}} \mathbb {V} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} , \boldsymbol {\theta} , \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) \right]}{2 \sqrt {\mathbb {V} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} , \boldsymbol {\theta} , \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) \right]}} (C. 6. 1)
$$

Next, we write the variance as expectation and—assuming the interventional or counterfactual distribution of $\mathbf { X } _ { \mathrm { d } ( \mathcal { I } ) }$ admits reparametrisation as is Ithe case for the GP-SCM and cvae models used in this chapter—use the reparametrisation trick to differentiate through the expectation operator as in (??).

$$
\begin{array}{l} \nabla_ {\boldsymbol {\theta}} \mathbb {V} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ h \big (\mathbf {X} _ {\mathrm{d} (\mathcal {I})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}} \big) \right] \\ = \nabla_ {\boldsymbol {\theta}} \mathbb {E} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ \left(h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) - \mathbb {E} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})} ^ {\prime}} \left[ h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} ^ {\prime}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) \right]\right) ^ {2} \right] \\ = \nabla_ {\boldsymbol {\theta}} \mathbb {E} _ {\mathbf {z} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ \left(h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} (\mathbf {z}; \boldsymbol {\theta}), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) - \mathbb {E} _ {\mathbf {z} ^ {\prime} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} (\mathbf {z} ^ {\prime}; \boldsymbol {\theta}), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) \right]\right) ^ {2} \right] \\ = \mathbb {E} _ {\mathbf {z} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ \nabla_ {\boldsymbol {\theta}} \Big (h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} (\mathbf {z}; \boldsymbol {\theta}), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) - \mathbb {E} _ {\mathbf {z} ^ {\prime} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} (\mathbf {z} ^ {\prime}; \boldsymbol {\theta}), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) \right] \Big) ^ {2} \right] \\ = \mathbb {E} _ {\mathbf {z} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ \right. 2 \left(h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} (\mathbf {z}; \boldsymbol {\theta}), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) - \mathbb {E} _ {\mathbf {z} ^ {\prime} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} \left(\mathbf {z} ^ {\prime}; \boldsymbol {\theta}\right), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right)\right]\right) \\ \left. \times \left(\nabla_ {\boldsymbol {\theta}} h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} (\mathbf {z}; \boldsymbol {\theta}), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) - \mathbb {E} _ {\mathbf {z} ^ {\prime} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ \nabla_ {\boldsymbol {\theta}} h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} \left(\mathbf {z} ^ {\prime}; \boldsymbol {\theta}\right), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) \right]\right)\right) \Bigg ] \tag {C.6.2} \\ \end{array}
$$

We can now obtain an estimate of the gradient with two independent sets of Monte Carlo samples of $\mathbf { X } _ { \mathrm { d } ( \mathcal { T } ) }$ , drawn via reparametrisation from the interventional or counterfactual distribution,

$$
\left\{\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m)} := \mathbf {x} _ {\mathrm{d} (\mathcal {I})} \left(\mathbf {z} ^ {(m)}; \boldsymbol {\theta}\right) \right\} _ {m = 1} ^ {M}, \quad \left\{\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m ^ {\prime})} := \mathbf {x} _ {\mathrm{d} (\mathcal {I})} \left(\mathbf {z} ^ {(m ^ {\prime})}; \boldsymbol {\theta}\right) \right\} _ {m ^ {\prime} = 1} ^ {M ^ {\prime}} \tag {C.6.3}
$$

$\begin{array} { r l } { \mathrm { w h e r e } } & { { } \mathbf { z } ^ { ( m ) } , \mathbf { z } ^ { ( m ^ { \prime } ) } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) . } \end{array}$

This yields the following Monte Carlo gradient estimator of the variance:

$$
\begin{array}{l} \nabla_ {\boldsymbol {\theta}} \mathbb {V} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ h \big (\mathbf {X} _ {\mathrm{d} (\mathcal {I})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}} \big) \right] \approx \\ \frac {1}{M} \sum_ {m = 1} ^ {M} \left[ 2 \left(h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m)}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) - \frac {1}{M ^ {\prime}} \sum_ {m ^ {\prime} = 1} ^ {M} h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m ^ {\prime})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right)\right) \times \right. \\ \left. \left(\nabla_ {\boldsymbol {\theta}} h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m)}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) - \frac {1}{M ^ {\prime}} \sum_ {m ^ {\prime} = 1} ^ {M ^ {\prime}} \nabla_ {\boldsymbol {\theta}} h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m ^ {\prime})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right)\right) \right] \tag {C.6.4} \\ \end{array}
$$

Substituting the above expression, together with the following Monte Carlo estimate of the (undifferentiated) variance

$$
\begin{array}{l} \mathbb {V} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) \right] \\ \approx \frac {1}{M - 1} \sum_ {m = 1} ^ {M} \left(h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m)}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) - \frac {1}{M} \sum_ {m ^ {\prime} = 1} ^ {M ^ {\prime}} h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m ^ {\prime})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right)\right) ^ {2}, \tag {C.6.5} \\ \end{array}
$$

into (C.6.1) gives the desired estimate for the gradient of the standard deviation of h.

Table C.2: Experimental results for the gradient-descent approach on different $3 ^ { - }$ variable SCMs (top to bottom: linear SCM, non-linear ANM, non-additive SCM). We show average performance for $N _ { \mathrm { r u n s } } = 1 0 0 , N _ { \mathrm { M C - s a m p l e s } } = 1 0 0 ,$ , and $\gamma _ { \mathrm { L C B } } = 2 ,$ , and display the number (out of $N _ { \mathrm { r u n s } } )$ of performed interventions on all subsets of variables by each recourse type. The two right-most columns display how many of the intervention sets for each recourse type agreed with the suggestions made by the oracle methods, $\mathcal { M } _ { \star }$ and $\mathbf { C A T E } _ { \star } ,$ respectively. We observe that interventions proposed by the subpopulation-based oracle often differ from the ones proposed at the individual level, which can be visually explained by Fig. ??. Importantly, we observe general agreement among all cate approaches in their selection of intervened-upon variables. In contrast, we observe that individual-based methods deviate away from their oracle $( \mathrm { i . e . , } M _ { \star } )$ in their selection of variables to intervene upon for recourse. This result further suggest that the cate approaches presented in this work exhibit more predictable behaviour, as they are less sensitive to model assumptions, and are thus more preferable for the individual seeking recourse under imperfect causal knowledge.

<table><tr><td rowspan="2">Method</td><td colspan="3">SCM</td><td colspan="7">INTERVENTION SET</td><td colspan="2">IDENTICAL INT. SET</td></tr><tr><td>Valid $_*$ (%)</td><td>LCB</td><td>Cost (%)</td><td> $\{X_1\}$ </td><td> $\{X_2\}$ </td><td> $\{X_3\}$ </td><td> $\{X_1,X_2\}$ </td><td> $\{X_1,X_3\}$ </td><td> $\{X_2,X_3\}$ </td><td> $\{X_1,X_2,X_3\}$ </td><td> $\mathcal{M}_*$ </td><td>CATE $_*$ </td></tr><tr><td> $\mathcal{M}_*$ </td><td>100</td><td>-</td><td>10.9±7.9</td><td>0</td><td>25</td><td>0</td><td>56</td><td>0</td><td>0</td><td>19</td><td>100</td><td>23</td></tr><tr><td> $\mathcal{M}_{\text{LIN}}$ </td><td>100</td><td>-</td><td>11.0±7.0</td><td>0</td><td>26</td><td>0</td><td>50</td><td>0</td><td>1</td><td>23</td><td>52</td><td>23</td></tr><tr><td> $\mathcal{M}_{\text{KR}}$ </td><td>90</td><td>-</td><td>10.7±6.5</td><td>0</td><td>22</td><td>0</td><td>44</td><td>0</td><td>0</td><td>34</td><td>54</td><td>27</td></tr><tr><td> $\mathcal{M}_{\text{GP}}$ </td><td>100</td><td>.55±.04</td><td>12.2±8.3</td><td>0</td><td>6</td><td>0</td><td>13</td><td>0</td><td>7</td><td>74</td><td>25</td><td>61</td></tr><tr><td> $\mathcal{M}_{\text{CVAE}}$ </td><td>100</td><td>.55±.07</td><td>11.8±7.7</td><td>0</td><td>12</td><td>0</td><td>25</td><td>0</td><td>5</td><td>58</td><td>31</td><td>57</td></tr><tr><td>CATE $_*$ </td><td>90</td><td>.56±.07</td><td>11.9±9.2</td><td>0</td><td>6</td><td>0</td><td>11</td><td>0</td><td>13</td><td>70</td><td>23</td><td>100</td></tr><tr><td>CATE $_{\text{GP}}$ </td><td>93</td><td>.56±.05</td><td>12.2±8.4</td><td>0</td><td>3</td><td>0</td><td>9</td><td>1</td><td>15</td><td>72</td><td>18</td><td>76</td></tr><tr><td>CATE $_{\text{CVAE}}$ </td><td>89</td><td>.56±.08</td><td>12.1±8.9</td><td>0</td><td>6</td><td>1</td><td>11</td><td>0</td><td>16</td><td>66</td><td>18</td><td>78</td></tr><tr><td> $\mathcal{M}_*$ </td><td>100</td><td>-</td><td>20.1±12.3</td><td>70</td><td>0</td><td>0</td><td>2</td><td>16</td><td>0</td><td>11</td><td>99</td><td>17</td></tr><tr><td> $\mathcal{M}_{\text{LIN}}$ </td><td>54</td><td>-</td><td>20.6±11.0</td><td>13</td><td>0</td><td>0</td><td>0</td><td>81</td><td>0</td><td>5</td><td>20</td><td>41</td></tr><tr><td> $\mathcal{M}_{\text{KR}}$ </td><td>91</td><td>-</td><td>20.6±12.5</td><td>65</td><td>0</td><td>0</td><td>1</td><td>23</td><td>0</td><td>10</td><td>76</td><td>22</td></tr><tr><td> $\mathcal{M}_{\text{GP}}$ </td><td>100</td><td>.54±.03</td><td>21.9±12.9</td><td>39</td><td>0</td><td>0</td><td>0</td><td>38</td><td>0</td><td>22</td><td>54</td><td>38</td></tr><tr><td> $\mathcal{M}_{\text{CVAE}}$ </td><td>97</td><td>.54±.05</td><td>22.6±12.3</td><td>33</td><td>0</td><td>0</td><td>0</td><td>51</td><td>0</td><td>15</td><td>45</td><td>42</td></tr><tr><td>CATE $_*$ </td><td>97</td><td>.55±.05</td><td>26.3±21.4</td><td>4</td><td>0</td><td>0</td><td>0</td><td>44</td><td>2</td><td>49</td><td>17</td><td>99</td></tr><tr><td>CATE $_{\text{GP}}$ </td><td>94</td><td>.55±.06</td><td>25.0±14.8</td><td>4</td><td>1</td><td>0</td><td>0</td><td>37</td><td>4</td><td>53</td><td>11</td><td>69</td></tr><tr><td>CATE $_{\text{CVAE}}$ </td><td>98</td><td>.54±.05</td><td>26.0±14.3</td><td>3</td><td>0</td><td>0</td><td>1</td><td>32</td><td>1</td><td>62</td><td>12</td><td>70</td></tr><tr><td> $\mathcal{M}_*$ </td><td>100</td><td>-</td><td>13.2±11.0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>11</td><td>78</td><td>7</td><td>97</td><td>78</td></tr><tr><td> $\mathcal{M}_{\text{LIN}}$ </td><td>98</td><td>-</td><td>14.0±13.5</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>85</td><td>11</td><td>81</td><td>77</td></tr><tr><td> $\mathcal{M}_{\text{KR}}$ </td><td>70</td><td>-</td><td>13.2±11.6</td><td>0</td><td>17</td><td>0</td><td>4</td><td>10</td><td>59</td><td>7</td><td>55</td><td>53</td></tr><tr><td> $\mathcal{M}_{\text{GP}}$ </td><td>95</td><td>.52±.04</td><td>13.4±12.8</td><td>3</td><td>1</td><td>2</td><td>0</td><td>0</td><td>82</td><td>9</td><td>73</td><td>78</td></tr><tr><td> $\mathcal{M}_{\text{CVAE}}$ </td><td>95</td><td>.51±.01</td><td>13.4±12.2</td><td>0</td><td>3</td><td>1</td><td>5</td><td>2</td><td>71</td><td>15</td><td>72</td><td>76</td></tr><tr><td>CATE $_*$ </td><td>100</td><td>.52±.02</td><td>13.5±13.0</td><td>0</td><td>0</td><td>2</td><td>0</td><td>9</td><td>77</td><td>9</td><td>78</td><td>97</td></tr><tr><td>CATE $_{\text{GP}}$ </td><td>94</td><td>.52±.03</td><td>13.2±13.1</td><td>3</td><td>1</td><td>5</td><td>0</td><td>3</td><td>73</td><td>12</td><td>70</td><td>76</td></tr><tr><td>CATE $_{\text{CVAE}}$ </td><td>100</td><td>.52±.05</td><td>13.6±12.9</td><td>0</td><td>1</td><td>2</td><td>0</td><td>1</td><td>82</td><td>11</td><td>78</td><td>78</td></tr></table>

**Table C.3: Selection of hyperparameters for cvae training was either performed manually (for Linear SCM, Non-linear ANM, Non-additve SCM) or automatically (for 7-variable semi-synthetic loan approval) by selecting the setting that resulted in the minimum MMD statistic between real and reconstructed samples.**

<table><tr><td colspan="2">SCM</td><td>Conditional</td><td>Encoder Arch.</td><td>Decoder Arch.</td><td>Latent Dim.</td><td> $\lambda_{\text{KLD}}$ </td></tr><tr><td rowspan="2" colspan="2">Linear SCM</td><td> $X_2|X_1,$ </td><td> $1\times32\times32\times32$ </td><td> $5\times5\times1$ </td><td>1</td><td>0.01</td></tr><tr><td> $X_3|X_1,X_2$ </td><td> $1\times32\times32\times32$ </td><td> $32\times32\times32\times1$ </td><td>1</td><td>0.01</td></tr><tr><td rowspan="2" colspan="2">Non-linear ANM</td><td> $X_2|X_1,$ </td><td> $1\times32\times32$ </td><td> $32\times32\times1$ </td><td>5</td><td>0.01</td></tr><tr><td> $X_3|X_1,X_2$ </td><td> $1\times32\times32\times32$ </td><td> $32\times32\times1$ </td><td>1</td><td>0.01</td></tr><tr><td rowspan="2" colspan="2">Non-additive SCM</td><td> $X_2|X_1,$ </td><td> $1\times32\times32\times32$ </td><td> $32\times32\times1$ </td><td>3</td><td>0.5</td></tr><tr><td> $X_3|X_1,X_2$ </td><td> $1\times32\times32\times32$ </td><td> $5\times5\times1$ </td><td>3</td><td>0.1</td></tr><tr><td rowspan="5">7-variable semi-synthetic loan approval</td><td rowspan="5">any</td><td></td><td></td><td> $2\times1$ </td><td></td><td></td></tr><tr><td></td><td> $1\times3\times3$ </td><td> $2\times2\times1$ </td><td></td><td>5, 1, 0.5, 0.1,</td></tr><tr><td></td><td> $1\times5\times5$ </td><td> $3\times3\times1$ </td><td> $1,2$ </td><td>0.05, 0.01,</td></tr><tr><td></td><td> $1\times3\times3\times3$ </td><td> $5\times5\times1$ </td><td></td><td>0.005</td></tr><tr><td></td><td></td><td> $3\times3\times3\times1$ </td><td></td><td></td></tr></table>
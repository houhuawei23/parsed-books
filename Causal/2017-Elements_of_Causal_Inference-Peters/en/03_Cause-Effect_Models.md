# Cause-Effect Models

The present chapter formalizes some basic concepts of causality for the case where the causal models contain only two variables. Assuming, these two variables are non-trivially related and their dependence is not solely due to a common cause, this constitutes a cause-effect model. We briefly introduce SCMs, interventions, and counterfactuals. All of these concepts are defined again in the context of multivariate causal models (Chapter 6) and we hope that encountering them for two variables first makes the ideas more easily accessible.

## 3.1 Structural Causal Models

SCMs constitute an important tool to relate causal and probabilistic statements.

Definition 3.1 (Structural causal models) An SCM C with graph $C  E$ consists of two assignments

$$
C := N _ {C}, \tag {3.1}
$$

$$
E := f _ {E} (C, N _ {E}), \tag {3.2}
$$

where $N _ { E } \perp \perp N _ { C }$ , that is, $N _ { E }$ is independent of $N _ { C }$ .

In this model, we call the random variables C the cause and E the effect. Furthermore, we call C a direct cause of E, and we refer to $C  E$ as a causal graph. This notation hopefully clarifies and coincides with the reader’s intuition when we talk about interventions, for example, in Example 3.2.

If we are given both the function $f _ { E }$ and the noise distributions $P _ { N _ { C } }$ and $P _ { N _ { E } }$ , we can sample data from such a model in the following way: We sample noise values

<!-- footnote -->

- A random variable X is a measurable function $\Omega  { \mathcal X } .$ , where the metric space X is equipped with the Borel σ -algebra. Its distribution $P _ { X }$ on $\mathcal { X }$ can be obtained from the measure P of the underlying probability space $( \Omega , { \mathcal { F } } , P )$ . We need not worry about this underlying space, and instead we generally start directly with the distribution of the random variables, assuming the random experiment directly provides us with values sampled from that distribution.

<!-- footnote end -->

<!-- footnote -->

- This notion of risk, which does not always coincide with its colloquial use, is taken from statistical learning theory [Vapnik, 1998] and has its roots in statistical decision theory [Wald, 1950, Ferguson, 1967, Berger, 1985]. In that context, $f ( x )$ is thought of as an action taken upon observing x, and the loss function measures the loss incurred when the state of nature is y.

<!-- footnote end -->

<!-- footnote -->

- For clarity, we formulate some important assumptions as principles. We do not take them for granted throughout the book; in this sense, they are not axioms.

<!-- footnote end -->

<!-- footnote -->

- We shall see in Section 6.3 that a more general way to think of interventions is that they change functions and random variables.
- Indeed, Proposition 4.1 implies that any joint distribution $P _ { X , Y }$ can be entailed by both models.

<!-- footnote end -->

<!-- footnote -->

- Let us for simplicity assume that we have access to the true activity of the gene without measurement noise.

<!-- footnote end -->

<!-- footnote -->

- Note that the conditional density $p ( a | t )$ allows us to compute $p ( \boldsymbol { a } , t )$ (and thus also $p ( a ) )$ from

<!-- footnote end -->

<!-- footnote -->

- $p ( t )$ , which may serve to motivate the direction of the arrow in $T \to A$ for the time being. This will be made precise in Definition 6.21.
- This is an idealized setting — no doubt counterexamples to these general remarks can be constructed.

<!-- footnote end -->

<!-- footnote -->

- We shall formalize this idea in Section 4.1.7.

<!-- footnote end -->

<!-- footnote -->

- There is an intuitive relation between this aspect of independence and the one described under 1.: whenever the mechanisms change independently, the change of one mechanism does not provide information on how the others have changed. Despite this overlap, the second independence contains an aspect that is not strictly contained in the first one because it is also applicable to a scenario in which none of the mechanisms has changed; for example, it refers also to homogeneous data sets.
- Although we have so far focused on the two-variable case, we phrase this argument such that it also applies for causal structures with more than two variables.

<!-- footnote end -->

<!-- footnote -->

- As an aside, while most of the early works were using linear equations only, there have also been attempts to generalize to nonlinear SEMs [Hoover, 2008].

<!-- footnote end -->

<!-- footnote -->

- We shall revisit this topic in more detail in Section 4.1.3.
- We would argue that this may not hold true if interventions are coupled to each other, for example, to keep the anticausal conditional (which describes the cause, given its effect) invariant. This could be seen as a violation of Principle 2.1 on the level of interventions. We return to this point in Section 2.3.4.
- This is akin to the independence of noise terms we use in SCMs.

<!-- footnote end -->

<!-- footnote -->

- Certain Bayesian structure learning methods [Heckerman et al., 1999] can be viewed as implementing the independence principle by assigning independent priors to the conditional probabilities of each variable given its causes.

<!-- footnote end -->

<!-- footnote -->

- More precisely, an event can only influence events lying in its light cone since no signal can travel faster than the speed of light in a vacuum, according to the theory of relativity.

<!-- footnote end -->

<!-- footnote -->

- The fact that the assignments are satisfied as equalities of random variables means that we are considering an ensemble of systems that differ in the realizations of the noise variables. Each realization leads to a (possibly different) realization for $X , Y .$ and thus the distribution of the noises implies a distribution over X ,Y .

<!-- footnote end -->

<!-- footnote -->

- For the interested reader: A system consisting of n two-level quantum systems is described by the 2n-dimensional Hilbert space $\dot { \mathbb { C } } ^ { 2 } \otimes \cdots \otimes \mathbb { C } ^ { 2 }$ . Unitary operators acting on this Hilbert space correspond to physical processes. For several such systems, researchers have shown how to implement “basic” unitaries that act on at most two of the n tensor components [Nielsen and Chuang, 2000] and act trivially on the remaining n−2 ones. Then one can generate any other unitary [DiVincenzo, 1995] approximately by concatenation. Although this is by no means the only possible choice for the set of “basic” unitary operations, the choice seems natural given the structure of physical interactions.

<!-- footnote end -->

$N _ { E }$ , NC and then evaluate (3.1) followed by (3.2). The SCM thus entails a joint distribution $P _ { C , E }$ over C and E (for a formal proof see Proposition 6.3).

## 3.2 Interventions

As discussed in Section 1.4.2, we are often interested in the system’s behavior under an intervention. The intervened system induces another distribution, which usually differs from the observational distribution. If any type of intervention can lead to an arbitrary change of the system, these two distributions become unrelated and instead of studying the two systems jointly we may consider them as two separate systems. This motivates the idea that after an intervention only parts of the data-generating process change. For example, we may be interested in a situation in which variable E is set to the value 4 (irrespective of the value of C) without changing the mechanism (3.1) that generates C. That is, we replace the assignment (3.2) by $E : = 4$ . This is called a (hard) intervention and is denoted by do $( E : = 4 )$ . The modified SCM, where (3.2) is replaced, entails a distribution over C that we denote by $P _ { C } ^ { d o ( E : = 4 ) }$ $P _ { C } ^ { \mathfrak { C } ; d o ( E : = 4 ) }$ , where the latter makes explicit that the SCM C was our starting point. The corresponding density is denoted by $c \mapsto p ^ { d o ( E : = 4 ) } ( c )$ or, in slight abuse of notation, $p ^ { d o ( \hat { E } : = 4 ) } ( c ) \overline { { . } }$ However, manipulations can be much more general. For example, the intervention do $\displaystyle \big ( E : = g _ { E } ( C ) + \tilde { N } _ { E } \big )$ keeps a functional dependence on C but changes the noise distribution. This is an example of a soft intervention. We can replace either of the two equations.

The following example motivates the namings “cause” and “effect”:

Example 3.2 (Cause-effect interventions) Suppose that the distribution $P _ { C , E }$ is entailed by an SCM C

$$
C := N _ {C}
$$

$$
E := 4 \cdot C + N _ {E}, \tag {3.3}
$$

with $N _ { C } , N _ { E } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , 1 )$ , and graph $C  E$ . Then,

$$
P _ {E} ^ {\mathfrak {C}} = \mathcal {N} (0, 1 7) \neq \mathcal {N} (8, 1) = P _ {E} ^ {\mathfrak {C}; d o (C := 2)} = P _ {E | C = 2} ^ {\mathfrak {C}}
$$

$$
\neq \mathcal {N} (1 2, 1) = P _ {E} ^ {\mathfrak {C}; d o (C := 3)} = P _ {E | C = 3} ^ {\mathfrak {C}}.
$$

Intervening on C changes the distribution of E. But on the other hand,

$$
P _ {C} ^ {\mathfrak {C}; d o (E := 2)} = \mathcal {N} (0, 1) = P _ {C} ^ {\mathfrak {C}} = P _ {C} ^ {\mathfrak {C}; d o (E := 3 1 4 1 5 9 2 6 5)} \left(\neq P _ {C | E = 2} ^ {\mathfrak {C}}\right). \tag {3.4}
$$

No matter how strongly we intervene on E, the distribution of C remains what it was before. This model behavior corresponds well to our intuition of C “causing” E: for example, no matter how much we whiten someone’s teeth, this will not have any effect on this person’s smoking habits. (Importantly, the conditional distribution of C given E = 2 is different from the distribution of C after intervening and setting E to 2.)

The asymmetry between cause and effect can also be formulated as an independence statement. When we replace the assignment (3.3) with $E : = \tilde { N } _ { E }$ (think about randomizing E), we break the dependence between C and E. In

$$
P _ {C, E} ^ {\mathfrak {C}; d o \left(E := \tilde {N} _ {E}\right)}
$$

we find C ⊥⊥ E. This independence does not hold when randomizing C. As long as var $\left[ \tilde { N } _ { C } \right] \neq 0$ , we find C 6⊥⊥ E in

$$
P _ {C, E} ^ {\mathfrak {C}; d o \left(C := \tilde {N} _ {C}\right)};
$$

the correlation between C and E remains non-zero.

Code Snippet 3.3 The code samples from the SCM described in Example 3.2.

```txt
set.seed(1)
# generates a sample from the distribution entailed by the SCM
C <- rnorm(300)
E <- 4*C + rnorm(300)
c(mean(E), var(E))
# [1] 0.1236532 16.1386767
#
# generates a sample from the intervention distribution do(C:=2);
# this changes the distribution of E
C <- rep(2,300)
E <- 4*C + rnorm(300)
c(mean(E), var(E))
# [1] 7.936917 1.187035
#
# generates a sample from the intervention distribution do(E:=N~);
# this breaks the dependence between C and E
C <- rnorm(300)
E <- rnorm(300)
cor.test(C,E)$p.value
# [1] 0.2114492
```

## 3.3 Counterfactuals

Another possible modification of an SCM changes all of its noise distributions. Such a change can be induced by observations and allows us to answer counterfactual questions. To illustrate this, imagine the following hypothetical scenario:

Example 3.4 (Eye disease) There exists a rather effective treatment for an eye disease. For 99% of all patients, the treatment works and the patient gets cured $( B =$ 0); if untreated, these patients turn blind within a day $( B = 1 )$ . For the remaining 1%, the treatment has the opposite effect and they turn blind $( B = 1 )$ within a day. If untreated, they regain normal vision $( B = 0 )$ .

Which category a patient belongs to is controlled by a rare condition $( N _ { B } = 1 )$ that is unknown to the doctor, whose decision whether to administer the treatment $( T = 1 )$ is thus independent of $N _ { B }$ . We write it as a noise variable $N _ { T }$ .

Assume the underlying SCM

$$
\mathfrak {C}: \begin{array}{l l l} T & := & N _ {T} \\ B & := & T \cdot N _ {B} + (1 - T) \cdot (1 - N _ {B}) \end{array} \tag {3.5}
$$

with Bernoulli distributed $N _ { B } \sim \mathrm { B e r } ( 0 . 0 1 )$ ; note that the corresponding causal graph is $T  B$ .

Now imagine a specific patient with poor eyesight comes to the hospital and goes blind $( B = 1 )$ after the doctor administers the treatment $( T = 1 )$ . We can now ask the counterfactual question “What would have happened had the doctor administered treatment $T = 0 ? \ '$ Surprisingly, this can be answered. The observation $B = T = 1$ implies with (3.5) that for the given patient, we had $N _ { B } = 1$ . This, in turn, lets us calculate the effect of do $( T : = 0 )$ .

To this end, we first condition on our observation to update the distribution over the noise variables. As we have seen, conditioned on $B = T = 1$ , the distribution for $N _ { B }$ and the one for $N _ { T }$ collapses to a point mass on 1, that is, $\delta _ { 1 }$ . This leads to a modified SCM:

$$
\mathfrak {C} | B = 1, T = 1: \begin{array}{l l l} T & := & 1 \\ B & := & T \cdot 1 + (1 - T) \cdot (1 - 1) = T \end{array} \tag {3.6}
$$

Note that we only update the noise distributions; conditioning does not change the structure of the assignments themselves. The idea is that the physical mechanisms are unchanged (in our case, what leads to a cure and what leads to blindness), but we have gleaned knowledge about the previously unknown noise variables for the given patient.

Next, we calculate the effect of do (T = 0) for this patient:

$$
\mathfrak {C} | B = 1, T = 1; d o (T := 0): \begin{array}{l l l} T & := & 0 \\ B & := & T \end{array} \tag {3.7}
$$

Clearly, the entailed distribution puts all mass on (0, 0), and hence

$$
P ^ {\mathfrak {C} | B = 1, T = 1; d o (T := 0)} (B = 0) = 1.
$$

This means that the patient would thus have been cured (B = 0) if the doctor had not given him treatment, in other words, do $( T : = 0 )$ . Because of

$$
P ^ {\mathfrak {C}; d o (T := 1)} (B = 0) = 0. 9 9 \quad \text { and }
$$

$$
P ^ {\mathfrak {C}; d o (T := 0)} (B = 0) = 0. 0 1,
$$

however, we can still argue that the doctor acted optimally (according to the available knowledge).

Interestingly, Example 3.4 shows that we can use counterfactual statements to falsify the underlying causal model (see Section 6.8). Imagine that the rare condition $N _ { B }$ can be tested, but the test results take longer than a day. In this case, it is possible that we observe a counterfactual statement that contradicts the measurement result for $N _ { B }$ . The same argument is given by Pearl [2009, p.220, point (2)]. Since the scientific content of counterfactuals has been debated extensively, it should be emphasized that the counterfactual statement here is falsifiable because the noise variable is not unobservable in principle but only at the moment when the decision of the doctor has to be made.

## 3.4 Canonical Representation of Structural Causal Models

We have discussed two types of causal statements both entailed by SCMs: first, the behavior of the system under potential interventions, and second, counterfactual statements. To further understand the difference between them, we introduce the following “canonical representation” of an SCM.2 According to the structural assignment

$$
E = f _ {E} (C, N _ {E}),
$$

for each fixed value $n _ { E }$ of the noise $N _ { E }$ , E is a deterministic function of C:

$$
E = f _ {E} (C, n _ {E}). \tag {3.8}
$$

In order words, if C and E attain values in C and E, respectively, then the noise $N _ { E }$ switches between different functions from C to E. Without loss of generality, we may therefore assume that $N _ { E }$ attains values in the set of functions from C to E , denoted by $\mathcal { E } ^ { \mathcal { C } }$ . Using this convention, we can also rewrite (3.8) as

$$
E = n _ {E} (C), \tag {3.9}
$$

and call this the canonical representation of the structural equation relating C and E.

Let us now explain why two SCMs with different canonical representations may induce the same interventional probabilities, although they differ in their counterfactual statements. To this end, we restrict the attention to the case where C attains values in the finite set ${ \mathcal { C } } = \{ 1 , \ldots , k \}$ . Then the set of functions from C to E is given by the k-fold Cartesian product

$$
\mathcal {E} ^ {k} := \underbrace {\mathcal {E} \times \cdots \times \mathcal {E}} _ {k \text { times}},
$$

where the jth component describes which value E attains for $C = j$ . Accordingly, the distribution $P _ { N _ { E } }$ is given by a joint distribution on $\mathcal { E } ^ { k }$ whose marginal distribution of the jth component determines the conditional $P _ { E | C = j }$ . Since C is the $P _ { E } ^ { d o \left( C : = j \right) } = P _ { E | C = j } ;$ tional probabilities and observational conditional probabilities coincide. Thus, the interventional causal implications of the SCM are completely determined by the marginal distributions of each component of the vector-valued noise variable $N _ { E }$ even though the SCM includes a precise specification of $P _ { N _ { E } }$ , that is, the joint distribution of all components. While the statistical dependences between the components of the noise variable $N _ { E }$ referring to the effect are irrelevant for interventional causal statements, they do matter for counterfactual statements. To see this, let C and E be binary, that is, ${ \mathcal { C } } = { \mathcal { E } } = \{ 0 , 1 \}$ . The set of functions from {0, 1} to {0, 1} reads $\mathcal { E } ^ { \mathcal { C } } = \{ \mathbf { 0 } , \mathbf { 1 } , \mathrm { I D } , \mathrm { N O T } \}$ where 0, 1 denote the constant functions attaining 0 and 1, respectively, and ID and NOT denote identity and negation, respectively. To construct two different distributions $P _ { N _ { E } } ^ { 1 }$ and $P _ { N _ { E } } ^ { 2 }$ inducing the same conditional $P _ { E | C = 0 } , P _ { E | C = 1 }$ , first choose the uniform mixture of 0 and 1 and second the uniform mixture of ID and NOT. In both cases, C and E are statistically independent and the distribution of E is unaffected by interventions on C because E remains an unbiased coin toss regardless of C. In the Cartesian product representation, the four

### 3.5. Problems

functions read $\mathcal { E } ^ { \mathcal { C } } = \{ ( 0 , 0 ) , ( 1 , 1 ) , ( 0 , 1 ) , ( 1 , 0 ) \}$ , the first and the second component denote the images of $C = 0$ and $C = 1$ , respectively. Obviously, the uniform mixture of $( 0 , 0 )$ and (1, 1) and the uniform mixture of (0, 1) and $( 1 , 0 )$ both induce the same marginal distributions on the first and the second component of the Cartesian product — in agreement with our remark that they induce the same intervention distributions. The counterfactual statement “E would have attained a different value if C had been set to a different one,” however, is true only for the mixture of ID and NOT, but not for the mixture of 0 and 1. Hence, counterfactual statements depend not only on the marginal distributions of the components of the noise variable $N _ { E }$ , but also on the statistical dependences between the Cartesian product components.

Note that two formally different SCMs may induce not only the same interventional distribution but even imply the same counterfactual statements: Given the assignment

$$
E := f _ {E} (C, N _ {E}),
$$

reparameterizations of $N _ { E }$ are obviously irrelevant. More explicitly, we may set

$$
E := \tilde {f} _ {E} (C, \tilde {N} _ {E}) = f _ {E} (C, g ^ {- 1} (\tilde {N} _ {E})),
$$

for some bijection g on the range of $N _ { E }$ and redefine the noise variable by $\tilde { N } _ { E } : =$ $g ( N _ { E } )$ . Using the canonical representation (3.9), we got rid of this additional degree of freedom that would have confused this discussion of counterfactuals.

### 3.5 Problems

Problem 3.5 (Sampling from an SCM) Consider the SCM

$$
X := Y ^ {2} + N _ {X} \tag {3.10}
$$

$$
Y := N _ {Y} \tag {3.11}
$$

with $N _ { X } , N _ { Y } \stackrel { i i d } { \sim } \mathcal { N } ( 0 , 1 )$ . Generate an i.i.d. sample of size 200 from the joint distribution (X,Y ).

Problem 3.6 (Conditional distributions) Show that $P _ { C | E = 2 } ^ { \mathrm { g } }$ in Equation (3.4) is a Gaussian distribution:

$$
C \mid E = 2 \sim \mathcal {N} \left(\frac {8}{1 7}, \sigma^ {2} = \frac {1}{1 7}\right).
$$

Problem 3.7 (Interventions) Assume that we know that a process either follows the SCM

$$
X := Y + N _ {X}
$$

$$
Y := N _ {Y},
$$

where $N _ { X } \sim \mathcal N ( \mu _ { X } , \sigma _ { X } ^ { 2 } )$ and $N _ { Y } \sim \mathcal { N } ( \mu _ { X } , \sigma _ { Y } ^ { 2 } )$ with unknown $\mu _ { X } , \mu _ { Y }$ and $\sigma _ { X } , \sigma _ { Y } >$ 0, or it follows the SCM

$$
X := M _ {X}
$$

$$
Y := X + M _ {Y},
$$

where $M _ { X } \sim \mathcal { N } ( \nu _ { X } , \tau _ { X } ^ { 2 } )$ and $M _ { Y } \sim \mathcal { N } ( \nu _ { Y } , \tau _ { Y } ^ { 2 } )$ with unknown $\nu _ { X } , \nu _ { Y }$ and $\tau _ { X } , \tau _ { Y } > 0$ . Is there a single intervention distribution that lets you distinguish between the two SCMs?

Problem 3.8 (Cyclic SCMs) We have mentioned that if the assignments inherit a cyclic structure, the SCM does not necessarily induce a unique distribution over the observed variables. Sometimes there is no solution and sometimes it is not unique.

a) We first look at an example that induces a unique solution. Consider the SCM

$$
X := 2 \cdot Y + N _ {X} \tag {3.12}
$$

$$
Y := 2 \cdot X + N _ {Y} \tag {3.13}
$$

with $( N _ { X } , N _ { Y } ) \sim P$ for an arbitrary distribution P. Compute $\alpha , \beta , \gamma , \delta$ such that

$$
X := \alpha N _ {X} + \beta N _ {Y}
$$

$$
Y := \gamma N _ {X} + \delta N _ {Y}
$$

yields a solution $\left( X , Y , N _ { X } , N _ { Y } \right)$ of the SCM; that is, the vector satisfies Equations (3.12) and (3.13). The solution can be seen as a special case of Equation (6.2).

b) Consider the SCM

$$
X := Y + N _ {X}
$$

$$
Y := X + N _ {Y}
$$

### 3.5. Problems

with $( N _ { X } , N _ { Y } ) \sim P .$ . Show that if P allows for a density with respect to Lebesgue measure and factorizes, that is, $N _ { X } \perp \perp N _ { Y }$ , then there is no solution $\left( X , Y , N _ { X } , N _ { Y } \right)$ of the SCM.

Furthermore, construct a distribution P, and a vector $\left( X , Y , N _ { X } , N _ { Y } \right)$ that solves the SCM.
# Potential Outcomes

## 2.1 Experimentalists' view of causal inference

Rubin (1975) and Holland (1986) made up the aphorism:

no causation without manipulation.

Not everybody agrees with this point of view. However, it is quite helpful to clarify ambiguity in thinking about causal relationships. This book follows this view and defines causal effects using the potential outcomes framework (Neyman, 1923; Rubin, 1974). In this framework, an experiment, or at least a thought experiment, has an intervention, a manipulation, or a treatment, and we are interested in its effect on an outcome or multiple outcomes.

Example 2.1 If we are interested in the effect of taking aspirin or not on the relief of head ache, the intervention is taking aspirin.

Example 2.2 If we are interested in the effect of participating in a job training program or not on employment and wage, the intervention is participating in a job training program.

Example 2.3 If we are interested in the effect of studying in a small classroom or a large classroom on standardized test scores, the intervention is studying in a small classroom.

Example 2.4 Gerber et al. (2008) were interested in the effect of different get-out-to-vote messages on the voting behavior. The intervention is different get-out-to-vote messages.

Example 2.5 Pearl (2018) claimed that we could infer the effect of obesity on life span. A popular measure of obesity of the body mass index (BMI), defined as the body mass divided by the square of the body height in units of $kg/m^{2}$ . So the intervention can be BMI.

However, there are different levels of ambiguity of the interventions above. The meanings of interventions in Examples 2.1–2.4 are relatively clear, but the meaning of intervention on BMI in Example 2.5 is less clear. In particular, we can imagine different versions of BMI reduction: healthier diet, more physical exercise, bariatric surgery, etc. These different versions of intervention can have quite different effects on the outcome. In this book, we will view the intervention in Example 2.5 as ill-defined without further clarifications.

Another ill-defined intervention is race. Racial discrimination is an important issue in labor market, but it is not easy to imagine an experiment to change the race of any experimental unit. Bertrand and Mullainathan (2004) give an interesting experiment that partially answers the question.

Example 2.6 Bertrand and Mullainathan (2004) randomly change the names on the resumes, and compare the callback rates of resumes with African-American- or White-sounding names. For each resume, the intervention is the binary indicator of African-American- or White-sounding name, and the outcome is the binary indicator of callback. We have analyzed the following two-by-two table in Section 1.2.2:

<table><tr><td></td><td>callback</td><td>no callback</td></tr><tr><td>African-American</td><td>157</td><td>2278</td></tr><tr><td>White</td><td>235</td><td>2200</td></tr></table>

From the above, we can compare the the probabilities of being called back among African-American- and White-sounding names:

$$
\frac {157}{2278 + 157} - \frac {235}{2200 + 235} = 6.45 \% - 9.65 \% = -3.20 \% <   0
$$

with p-value from the Fisher exact test much smaller than 0.001.

In Bertrand and Mullainathan (2004)'s experiment, the treatment is the perceived race which can be manipulated by experimenters. They design an experiment to answer a well-defined causal question.

## 2.2 Formal notation of potential outcomes

Consider a study with n experimental units indexed by $i = 1, \ldots, n$ . As a starting point, we focus on a treatment with two levels: 1 for the treatment and 0 for the control. For each unit i, the outcome of interest Y has two versions:

$$
Y _ {i} (1) \text { and } Y _ {i} (0),
$$

which are potential outcomes under the hypothetical interventions 1 and 0. Neyman (1923) first used this notation. It seems intuitive but has some hidden assumptions. Rubin (1980) made the following clarifications on the hidden assumptions.

Assumption 2.1 (no interference) Unit i's potential outcomes do not depend on other units' treatments. This is sometimes called the no-interference assumption.

Assumption 2.2 (consistency) There are no other versions of the treatment. Equivalently, we require that the treatment level be well defined, or have no ambiguity at least for the outcome of interest. This is sometimes called the consistency assumption.

Assumption 2.1 can be violated in infectious diseases or network experiments. For instance, if some of my friends receive flu shots, my chance of getting the flu decrease even if I do not receive the flu shot; if my friends see an ad on Facebook, my chance of buying that product increase even if I do not see the ad. It is an active research area to study situations with interfering units in modern causal inference literature.

Assumption 2.2 can be violated for treatment with complex components. For instance, when studying the effect of cigarette smoking on lung cancer, the type of cigarettes may matter; when studying the effect of college education on income, the type and major of college education may matter.

Rubin (1980) called the Assumptions 2.1 and 2.2 above together the Stable Unit Treatment Value Assumption (SUTVA).

Assumption 2.3 (SUTVA) Both Assumptions 2.1 and 2.2 hold.

Under SUTVA, Rubin (2005) called the $n \times 2$ matrix of potential outcomes the Science Table:

<table><tr><td>i</td><td>$ Y_{i}(1) $</td><td>$ Y_{i}(0) $</td></tr><tr><td>1</td><td>$ Y_{1}(1) $</td><td>$ Y_{1}(0) $</td></tr><tr><td>2</td><td>$ Y_{2}(1) $</td><td>$ Y_{2}(0) $</td></tr><tr><td>$ \vdots $</td><td>$ \vdots $</td><td>$ \vdots $</td></tr><tr><td>n</td><td>$ Y_{n}(1) $</td><td>$ Y_{n}(0) $</td></tr></table>

Due to Neyman and Rubin's fundamental contribution to statistical causal inference, the potential outcomes framework is sometimes called the Neyman model, the Neyman-Rubin model, or the Rubin Causal Model.

Causal effects are functions of the Science Table. Inferring individual causal effects

$$
\tau_ {i} = Y _ {i} (1) - Y _ {i} (0)
$$

is fundamentally challenging because we can only observe either $Y_{i}(1)$ or $Y_{i}(0)$ for each unit i, that is, we can observed only half of the Science Table. As a starting point, most parts of the book focus on the average causal effect (ACE):

$$
\tau = n ^ {- 1} \sum_ {i = 1} ^ {n} \left\{Y _ {i} (1) - Y _ {i} (0) \right\} = n ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i} (1) - n ^ {- 1} \sum_ {i = 1} ^ {n} Y _ {i} (0).
$$

But we can easily extend our discussion to many other parameters (also called estimands).

## 2.2.1 Causal effects, subgroups, and the non-existence of Yule–Simpson Paradox

If we have two subgroups defined by a binary variable $x_{i}$ , we can define the subgroup causal effects as

$$
\tau_ {x} = \frac {\sum_ {i = 1} ^ {n} I (x _ {i} = x) \{Y _ {i} (1) - Y _ {i} (0) \}}{\sum_ {i = 1} ^ {n} I (x _ {i} = x)}, \quad (x = 0, 1)
$$

where $I(\cdot)$ is the indicator function. A simple identity is that

$$
\tau = \pi_ {1} \tau_ {1} + \pi_ {0} \tau_ {0}
$$

where $\pi_{x}=\sum_{i=1}^{n}I(x_{i}=x)/n$ is the proportion of units with $x_{i}=x\ (x=0,1)$ . Therefore, if $\tau_{1}>0$ and $\tau_{0}>0$ , we must have $\tau>0$ . The Yule–Simpson Paradox thus cannot happen to causal effects.

## 2.2.2 Subtlety of experimental unit

I end this section with a subtlety related to the definition of the experimental unit. Simply speaking, the experimental unit can be different from the physical unit. For example, if I did not take aspirin before and my headache did not go way, but I take aspirin now and my headache goes away, you might think that we can observed my potential outcomes under both control and treatment. Let i index myself, and let Y = 1 denote the indicator of no headache. Then, the above heuristic suggests that $Y_{i}(0) = 0$ and $Y_{i}(1) = 1$ , so it seems that aspirin kills my headache. But this logic is very wrong because of the misunderstanding of the definition of the experimental unit. At different time points, I, the same physical person, become two distinct experiment units, indexed by “i, before” and “i, after”. Therefore, we have four potential outcomes

$$
Y _ {i, \mathrm{before}} (0) = 0, Y _ {i, \mathrm{before}} (1) = ?, Y _ {i, \mathrm{after}} (0) = ?, Y _ {i, \mathrm{after}} (1) = 1,
$$

with two of them observed and two of them missing. The individual causal effects

$$
Y _ {i, \mathrm{before}} (1) - Y _ {i, \mathrm{before}} (0) = ? - 0 \mathrm{and} Y _ {i, \mathrm{after}} (1) - Y _ {i, \mathrm{after}} (0) = 1 -?
$$

are unknown. It is possible that my headache goes away even if I do not take aspirin:

$$
Y _ {i, \mathrm{after}} (0) = 1, Y _ {i, \mathrm{after}} (1) = 1
$$

which implies zero effect; it is also possible that my headache does not go away if I do not take aspirin:

$$
Y _ {i, \mathrm{after}} (0) = 0, Y _ {i, \mathrm{after}} (1) = 1
$$

which implies a positive effect of aspirin.

The wrong heuristic argument might get the right answer if the control potential outcomes are stable at the before and after periods: $Y_{i,\text{before}}(0) = Y_{i,\text{after}}(0) = 0$ . But this assumption is rather strong and fundamentally untestable.

## 2.3 Treatment assignment mechanism

Let $Z_{i}$ be the binary treatment indicator for unit i, vectorized as $Z = (Z_{1},\ldots ,Z_{n})$ . The observed outcome of unit i is a function of the potential outcomes and the treatment indicator:

$$
Y _ {i} = \left\{ \begin{array}{l l} Y _ {i} (1), & \text { if   } Z _ {i} = 1 \\ Y _ {i} (0), & \text { if   } Z _ {i} = 0 \end{array} \right. \tag {2.1}
$$

$$
= Z _ {i} Y _ {i} (1) + \left(1 - Z _ {i}\right) Y _ {i} (0) \tag {2.2}
$$

$$
= Y _ {i} (0) + Z _ {i} \{Y _ {i} (1) - Y _ {i} (0) \} \tag {2.3}
$$

$$
= Y _ {i} (0) + Z _ {i} \tau_ {i}. \tag {2.4}
$$

Equation (2.1) is the definition of the observed outcome. Equation (2.2) is equivalent to (2.1). It is a trivial fact, but Judea Pearl viewed it as the fundamental bridge between the potential outcomes and the observed outcome. Equations (2.3) and (2.4) highlight the fact that the individual causal effect $\tau_{i}=Y_{i}(1)-Y_{i}(0)$ can be heterogeneous across units.

The experiment only reveals one of unit $i$ 's potential outcomes with the other one missing:

$$
\begin{array}{l} Y _ {i} ^ {\text { mis }} = \left\{ \begin{array}{l l} Y _ {i} (0), & \text { if   } Z _ {i} = 1 \\ Y _ {i} (1), & \text { if   } Z _ {i} = 0 \end{array} \right. \\ = Z _ {i} Y _ {i} (0) + (1 - Z _ {i}) Y _ {i} (1). \\ \end{array}
$$

The missing potential outcome correspond to the opposite treatment level of unit i. For this reason, the potential outcomes framework is also called the counterfactual framework. This name can be confusing because before the experiment, both potential outcomes are observable, and after the experiment, one potential outcomes is actually observed.

The treatment assignment mechanism, i.e., the probability distribution of Z, plays an important role in inferring causal effects. The following simple numerical examples illustrate this point. We first generate potential outcomes from Normal distributions with the average causal effect close to -0.5.

$$
\begin{array}{l} > n = 5 0 0 \\ > \mathrm{Y0} = \text { rnorm(n) } \\ > \text { tau } = - 0. 5 + Y 0 \\ > \mathrm{Y} 1 = \mathrm{Y} 0 + \text { tau } \\ \end{array}
$$

A perfect doctor assigns the treatment to the patient if s/he knows that the individual causal effect is non-negative. This results in a positive difference in means of the observed outcomes:

$$
\begin{array}{l} > Z = (\text { tau } > = 0) \\ > \mathrm{Y} = \mathrm{Z} * \mathrm{Y} 1 + (1 - \mathrm{Z}) * \mathrm{Y} 0 \\ \end{array}
$$

> mean(Y[Z==1]) - mean(Y[Z==0])

[1] 2.166509

A clueless doctor does not know any information about the individual causal effects and assigns the treatment to patients by flipping a fair coin. This results in a difference in means of the observed outcomes close to the true average causal effect:

```txt
> Z = rbinom(n, 1, 0.5)
> Y = Z * Y1 + (1 - Z) * Y0
> mean(Y[Z == 1]) - mean(Y[Z == 0])
[1] -0.552064
```

The above examples are hypothetical since no doctors perfectly know the individual causal effects. However, the examples do demonstrate the crucial role of the treatment assignment mechanism. This book will organize the topics based on the treatment assignment mechanism.

## 2.4 Homework Problems

## 2.1 A perfect doctor

Following the first perfect doctor example in Section 2.3, assume the potential outcomes are random variables generated from

$$
Y (0) \sim \mathrm{N} (0, 1), \quad \tau = - 0. 5 + Y (0), \quad Y (1) = Y (0) + \tau .
$$

The binary treatment is determined by the treatment effect as $Z = 1(\tau \geq 0)$ , and the observed outcome is determined by the potential outcomes and the treatment by $Y = ZY(1) + (1 - Z)Y(0)$ . Calculate the difference in means

$$
E (Y \mid Z = 1) - E (Y \mid Z = 0).
$$

Hint: The mean of a truncated Normal random variable equals

$$
E (X \mid a <   X <   b) = \mu - \sigma \frac {\phi \left(\frac {b - \mu}{\sigma}\right) - \phi \left(\frac {a - \mu}{\sigma}\right)}{\Phi \left(\frac {b - \mu}{\sigma}\right) - \Phi \left(\frac {a - \mu}{\sigma}\right)},
$$

where $X \sim \mathrm{N}(\mu, \sigma^{2})$ , and $\phi(\cdot)$ and $\Phi(\cdot)$ are the probability density and cumulative distribution functions of a standard Normal random variable.

## 2.2 Nonlinear causal estimands

With potential outcomes $\{(Y_{i}(1), Y_{i}(0)\}_{i=1}^{n}$ for n units under the treatment and control, the difference in means equals the mean of the individual treatment effects:

$$
\bar {Y} (1) - \bar {Y} (0) = n ^ {- 1} \sum_ {i = 1} ^ {n} \{Y _ {i} (1) - Y _ {i} (0) \}.
$$

## 2.4 Homework Problems

Therefore, the average treatment effect is a linear causal estimand.

Other estimands may not be linear. For instance, we can define the median treatment effect as

$$
\delta_ {1} = \mathrm{median} \{(Y _ {i} (1) \} _ {i = 1} ^ {n} - \mathrm{median} \{(Y _ {i} (0) \} _ {i = 1} ^ {n},
$$

which is, in general, different from the median of the individual treatment effect

$$
\delta_ {2} = \mathrm{median} \{(Y _ {i} (1) - Y _ {i} (0) \} _ {i = 1} ^ {n}.
$$

1. Give numerical examples which have $\delta_1 = \delta_2$ , $\delta_1 > \delta_2$ , and $\delta_1 < \delta_2$ .  
2. Which estimand makes more sense, $\delta_1$ or $\delta_2$ ? Why? Use examples to justify your conclusion. If you feel that both $\delta_1$ and $\delta_2$ can make sense in different applications, you can also give examples to justify both estimands.

## 2.3 Average and individual effects

Give a numerical example in which $\tau = n^{-1} \sum_{i=1}^{n} \{Y_i(1) - Y_i(0)\} > 0$ but the proportion of units with $Y_i(1) > Y_i(0)$ is smaller than 0.5. That is, the average causal effect is positive, but the treatment benefits less than half of the units.

## 2.4 Recommended reading

Holland (1986) is a classic review article on statistical causal inference. It popularized the name “Rubin Causal Model” for the potential outcomes framework. At the University of California Berkeley, we call it the “Neyman Model” for obvious reasons.
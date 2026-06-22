# Chapter 2 Causal Inference Preliminary

![image_02](images/image_02.png)

Liuyi Yao, Zhixuan Chu, Yaliang Li, Jing Gao, Aidong Zhang, and Sheng Li

## 2.1 Introduction

In everyday language, correlation and causality are commonly used interchangeably, although they have quite different interpretations. Correlation indicates a general relationship: two variables are correlated when they display an increasing or decreasing trend $[1]$ . Causality is also referred to as cause and effect, where the cause is partly responsible for the effect, and the effect is partly dependent on the cause. Causal inference is the process of drawing a conclusion about a causal connection based on the conditions of the occurrence of an effect. The main difference between causal inference and inference of correlation is that the former analyzes the response of the effect variable when the cause is changed $[10, 20]$ .

It is well known that “correlation does not imply causation.” For example, a study revealed that girls who typically have breakfast tend to weigh less than girls who do not, leading to the conclusion that breakfast can aid in weight loss. But in fact, these two events may just have correlation instead of causality. Perhaps girls who have breakfast every day have healthier lifestyles, including regular exercise,

L. Yao · Y. Li
Alibaba Group, Hangzhou, China
e-mail: yly287738@alibaba-inc.com; yaliang.li@alibaba-inc.com

Z. Chu
Ant Group, Hangzhou, China
e-mail: chuzhixuan.czx@alibaba-inc.com

J. Gao
Purdue University, West Lafayette, IN, USA
e-mail: jinggao@purdue.edu

A. Zhang · S. Li (☒)
University of Virginia, Charlottesville, VA, USA
e-mail: aidong@virginia.edu; shengli@virginia.edu sound sleep habits, and balanced diets, which eventually make them lightweight. In this case, eating breakfast and being lightweight share a common cause; thus, we may treat having a better lifestyle as a confounder of the causality between having breakfast and being lightweight.

In many cases, it seems obvious that one action can cause another; however, there exist also many cases that we cannot easily tease out and make sure of the relationship. Therefore, learning causality is a dauntingly challenging problem. The most effective way of inferring causality is to conduct a randomized controlled trial, which randomly assigns participants into a treatment group or a control group. As the randomized study is conducted, the only expected difference between the control and treatment groups is the outcome variable being studied. However, in reality, randomized controlled trials are always time-consuming and expensive, and thus, the study cannot involve many subjects, which may not be representative of the real-world population a treatment/intervention would eventually target. Another issue is that randomized controlled trials only focus on the average of samples, and they do not explain the mechanism for individual subjects. In addition, ethical issues also need to be considered in most randomized controlled trials, which largely limits their applications. Therefore, instead of randomized controlled trials, observational data are a tempting shortcut. Observational data are obtained by the researcher simply observing the subjects without any interference. That means the researchers have no control over treatments and subjects, and they just observe the subjects and record data based on their observations. From the observational data, we can find their actions, outcomes, and information about what has occurred, but we cannot figure out the mechanism of why they took a specific action. For the observational data, the core question is how to obtain the counterfactual outcome. For example, we want to answer this question “Would this patient have different results if he received a different medication?” Answering such counterfactual questions is challenging for two reasons $[15]$ : the first is that we only observe the factual outcome and never the counterfactual outcomes that would potentially have happened if they had chosen a different treatment option. The second is that treatments are typically not assigned at random in observational data, which may lead to the treated population differing significantly from the general population.

To solve these problems in causal inference from observational data, researchers have developed various frameworks, including the potential outcome framework $[14, 19]$ and the structural causal model $[9, 11, 12]$ . The potential outcome framework is also known as the Neyman–Rubin potential outcomes or the Rubin causal model. In the example we mentioned above, a girl would have a particular weight if she had breakfast normally every day, whereas she would have a different weight if she did not have breakfast normally. To measure the causal effect of having breakfast normally for a girl, we need to compare the outcomes for the same person under both situations. Obviously, it is impossible to see both potential outcomes at the same time, and one of the potential outcomes is always missing. The potential outcome framework aims to estimate such potential outcomes and then calculate the treatment effect. Therefore, treatment effect estimation is one of the central problems in causal inference under the potential outcome framework. Another influential framework in causal inference is the structural causal model (SCM), which includes the causal graph and the structural equations. The structural causal model describes the causal mechanisms of a system where a set of variables and the causal relationship among them are modeled by a set of simultaneous structural equations. Another line of learning causality is causal structure learning, whose objective is to reveal the causal relation by generating a causal graph. Representative methods can be divided into three categories, including constraint-based models $[18]$ , score-based models $[3, 13]$ , and functional causal models $[5, 22]$ . Different from causal effect estimation, causal structure learning addresses a different class of problems, which is out of our survey's scope; see $[17]$ for more information.

Causal inference has a close relationship with machine learning. In recent years, the magnificent bloom of the machine learning area has enhanced the development of the causal inference area. Powerful machine learning methods, such as decision trees, ensemble methods, and deep neural networks, are applied to estimate the potential outcome more accurately. In addition to the amelioration of the outcome estimation model, machine learning methods also provide a new aspect to handle confounders. Benefitting from the recent deep representation learning methods, the confounder variables are adjusted by learning the balanced representation for all covariates so that conditioning on the learned representation, the treatment assignment is independent of the confounder variables. In machine learning, the more data, the better. However, in causal inference, more data alone are not yet sufficient. Having more data only helps to obtain more precise estimates, but it cannot ensure that these estimates are correct and unbiased. Machine learning methods enhance the development of causal inference; meanwhile, causal inference also helps machine learning methods. The simple pursuit of predictive accuracy is insufficient for modern machine learning research, and correctness and interpretability are also the targets of machine learning methods. Causal inference is starting to help improve machine learning, such as recommender systems or reinforcement learning.

In this chapter, we provide a comprehensive review of the causal inference methods. We introduce the basic concepts as well as its three critical assumptions to identify the causal effect.

## 2.2 Basics of Causal Inference

In this section, we present the background knowledge of causal inference, including task description, mathematical notions, assumptions, challenges, and general solutions. We also give an illustrative example that will be used throughout this survey.

Generally, the task of causal inference is to estimate the outcome changes if another treatment had been applied. For example, suppose there are two treatments that can be applied to patients: Medicine A and Medicine B. When applying Medicine A to the interested patient cohort, the recovery rate is 70%, while applying Medicine B to the same cohort, the recovery rate is 90%. The change in recovery rate is the effect that treatment (i.e., medicine in this example) asserts on the recovery rate.

The above example describes an ideal situation to measure the treatment effect: applying different treatments to the same cohort. In real-world scenarios, this ideal situation can only be approximated by a randomized experiment in which the treatment assignment is controlled, such as a completely random assignment. In this way, the group receiving a specific treatment can be viewed as an approximation to the cohort we are interested in.

However, performing randomized experiments is expensive, time-consuming, and sometimes even unethical. Therefore, estimating the treatment effect from observational data has attracted growing attention due to the wide availability of observational data. Observational data usually contain a group of individuals who have taken different treatments, their corresponding outcomes, and possibly more information, but without direct access to the reason/mechanism why they took the specific treatment. Such observational data enable researchers to investigate the fundamental problem of learning the causal effect of a certain treatment without performing randomized experiments. To better introduce various treatment effect estimation methods, the following section introduces several definitions, including unit, treatment, outcome, treatment effect, and other information (pre- and post-treatment variables) provided by observational data.

## 2.2.1 Definitions

Here we define the notations under the potential outcome framework $[14, 19]$ , which is logically equivalent to another framework, the structural causal model framework $[8]$ . The foundation of the potential outcome framework is that causality is tied to treatment (or action, manipulation, intervention) applied to a unit $[6]$ . The treatment effect is obtained by comparing units' potential outcomes of treatments. In the following, we first introduce three essential concepts in causal inference: unit, treatment, and outcome.

Definition 2.1 (Unit) A unit is the atomic research object in the treatment effect study.

A unit can be a physical object, a firm, a patient, an individual person, or a collection of objects or persons, such as a classroom or a market, at a particular time point $[6]$ . Under the potential outcome framework, the atomic research objects at different time points are different units. One unit in the dataset is a sample of the whole population, so in this survey, the terms “sample” and “unit” are used interchangeably.

Definition 2.2 (Treatment) Treatment refers to the action that applies (exposes or subjects) to a unit.

Let $W$ ( $W \in \{0, 1, 2, \ldots, N_W\}$ ) denote the treatment, where $N_W + 1$ is the total number of possible treatments. In the aforementioned medicine example, Medicine A is a treatment. Most of the literature considers the binary treatment, and in this case, the group of units applied with treatment $W = 1$ is the treated group, and the group of units with $W = 0$ is the control group.

Definition 2.3 (Potential outcome) For each unit–treatment pair, the outcome of that treatment when applied to that unit is the potential outcome [6].

The potential outcome of treatment with value $w$ is denoted as $Y(W = w)$ .

Definition 2.4 (Observed outcome) The observed outcome is the outcome of the treatment that is actually applied.

The observed outcome is also called the factual outcome, and we use $Y^{F}$ to denote it where F stands for “factual.” The relation between the potential outcome and the observed outcome is $Y^{F} = Y(W = w)$ , where w is the treatment actually applied.

Definition 2.5 (Counterfactual outcome) The counterfactual outcome is the outcome if the unit had taken another treatment.

The counterfactual outcomes are the potential outcomes of the treatments except for the one actually taken by the unit. Since a unit can only take one treatment, only one potential outcome can be observed, and the remaining unobserved potential outcomes are counterfactual outcomes. In the multiple treatment case, let $Y^{CF}(W = w')$ denote the counterfactual outcome of treatment with value $w'$ . In the binary treatment case, for notation simplicity, we use $Y^{CF}$ to denote the counterfactual outcome, and $Y^{CF} = Y(W = 1 - w)$ , where $w$ is the treatment actually taken by the unit.

In the observational data, in addition to the chosen treatments and the observed outcome, the units' other information is also recorded, and they can be separated into pre-treatment variables and post-treatment variables.

Definition 2.6 (Pre-treatment variables) Pre-treatment variables are the variables that are not affected by the treatment.

Pre-treatment variables are also named background variables, and they can be patients' demographics, medical history, etc. Let $X$ denote the pre-treatment variables.

Definition 2.7 (Post-treatment variables) The post-treatment variables are the variables that are affected by the treatment.

One example of post-treatment variables is the intermediate outcome, such as the lab test after taking medicine in the aforementioned medicine example.

In the following sections, the terminology variable refers to the pre-treatment variable unless otherwise specified.

Treatment Effect After introducing the observational data and the key terminologies, the treatment effect can be quantitatively defined using the above definitions.

The treatment effect can be measured at the population, treated group, subgroup, and individual levels. To clarify these definitions, here we define the treatment effect under binary treatment, and it can be extended to multiple treatments by comparing their potential outcomes.

At the population level, the treatment effect is named the average treatment effect (ATE), which is defined as

$$
\mathrm{ATE} = \mathbb {E} [ \mathbf {Y} (W = 1) - \mathbf {Y} (W = 0) ], \tag {2.1}
$$

where $\mathbf{Y}(W = 1)$ and $\mathbf{Y}(W = 0)$ are the potential treated and control outcomes of the whole population, respectively.

For the treated group, the treatment effect is named as average treatment effect on the treated group (ATT), and it is defined as

$$
\mathrm{ATT} = \mathbb {E} [ \mathbf {Y} (W = 1) | W = 1 ] - \mathbb {E} [ \mathbf {Y} (W = 0) | W = 1 ], \tag {2.2}
$$

where $\mathbf{Y}(W = 1)|W = 1$ and $\mathbf{Y}(W = 0)|W = 1$ are the potential treated and control outcomes of the treated group, respectively.

At the subgroup level, the treatment effect is called conditional average treatment effect (CATE), which is defined as

$$
\mathrm{CATE} = \mathbb {E} [ \mathbf {Y} (W = 1) | X = x ] - \mathbb {E} [ \mathbf {Y} (W = 0) | X = x ], \tag {2.3}
$$

where $\mathbf{Y}(W=1)|X=x$ and $\mathbf{Y}(W=0)|X=x$ are the potential treated and control outcomes of the subgroup with X=x, respectively. CATE is a common treatment effect measurement when the treatment effect varies across different subgroups, which is also known as the heterogeneous treatment effect.

At the individual level, the treatment effect is called the individual treatment effect (ITE), and the ITE of unit i is defined as

$$
\mathrm{ITE} _ {i} = Y _ {i} (W = 1) - Y _ {i} (W = 0), \tag {2.4}
$$

where $Y_{i}(W = 1)$ and $Y_{i}(W = 0)$ are the potential treated and control outcomes of unit $i$ , respectively. In some works [7, 16], the ITE is viewed as the CATE.

Objective For causal inference, our objective is to estimate the treatment effects from the observational data. Formally speaking, given the observational dataset, $\left\{X_{i}, W_{i}, Y_{i}^{F}\right\}_{i=1}^{N}$ , where N is the total number of units in the datasets, the goal of the causal inference task is to estimate the treatment effects defined above.

## 2.2.2 An Illustrative Example

To better illustrate causal inference, we use the following example combined with the notations defined above to give an overview. In this example, we want to evaluate the treatment effects of several different medications for one disease by exploiting observational data (i.e., electronic health records) that include demographic information of patients, the specific medication with the specific dosage taken by patients, and the outcome of medical tests. Obviously, we can only obtain one factual outcome for a specific patient from electronic health records, and thus the core task is to predict what would have happened if a patient took another treatment (i.e., a different medication or the same medication with a different dosage). Answering such counterfactual questions is very challenging. Therefore, we want to use causal inference to predict all of the potential outcomes for each patient over all of the medications with different dosages. Then, we can reasonably and accurately evaluate and compare the treatment effect of different medications for this disease.

One particular point to keep in mind is that for each medication, they may have different dosages. For example, for medication A, the dosage range can be a continuous variable in the range $[a, b]$ , while for medication B, the dosage can be a categorical variable that has several specific dosage regimens.

In the aforementioned example, the units are the patients with the studied disease. The treatments refer to the different medications with specific dosages for this disease, and we use $W$ ( $W \in \{0, 1, 2, \dots, N_W\}$ ) to denote these treatments. For example, $W_i = 1$ can represent the medication $A$ with a specific dosage taken by the unit $i$ , and $W_i = 2$ represents the medication $B$ with a specific dosage taken by the unit $i$ . $Y$ is the outcome, such as one type of blood test that can measure the medication's ability to destroy the disease and lead to the recovery of the patients. Let $Y_i(W = 1)$ denote the potential outcome of medication $A$ with a specific dosage on patient $i$ . The features of patients may include age, gender, clinical presentation, and some other medical tests, etc. Among these features, age, gender, and other demographic information are pre-treatment variables that cannot be affected by taking treatment. Some clinical presentations and medical tests are affected by taking medications, and they are post-treatment variables. In this example, our goal is to estimate the treatment effects of different medications for this disease based on the provided observational data.

In the following sections, we will continuously use this example to explain more concepts and illustrate intuitions behind various causal inference methods.

## 2.2.3 Assumptions

To estimate the treatment effect, the following assumptions are commonly used in the causal inference literature.

Assumption (Stable Unit Treatment Value Assumption (SUTVA)) The potential outcomes for any unit do not vary with the treatment assigned to other units, and, for each unit, there are no different forms or versions of each treatment level, which lead to different potential outcomes.

This assumption emphasizes two points: The first point is the independence of each unit, that is, there are no interactions between units. In the context of the above illustrative example, one patient's outcome will not affect other patients' outcomes.

The second point is the single version for each treatment. In the above example, Medicine A with different dosages are different treatments under the SUTVA assumption.

Assumption (Ignorability) em Given the background variable, X, treatment assignment W is independent of the potential outcomes, i.e., $W \perp Y(W = 0)$ , $Y(W = 1)|X$ .

In the context of the illustrative example, this ignorability assumption indicates two-folds: First, if two patients have the same background variable X, their potential outcomes should be the same whatever the treatment assignment is, i.e., $p(Y_{i}(0), Y_{i}(1)|X = x, W = W_{i}) = p(Y_{j}(0), Y_{j}(1)|X = x, W = W_{j})$ . Analogously, if two patients have the same background variable value, their treatment assignment mechanism should be the same regardless of the value of potential outcomes they have, i.e., $p(W|X = x, Y_{i}(0), Y_{i}(1)) = p(W|X = x, Y_{j}(0), Y_{j}(1))$ . The ignorability assumption is also named as the unconfoundedness assumption. With this unconfoundedness assumption, for units with the same background variable X, their treatment assignment can be viewed as random.

Assumption (Positivity) For any value of $X$ , treatment assignment is not deterministic:

$$
P (W = w | X = x) > 0, \quad \forall w \text {   and   } x. \tag {2.5}
$$

If, for some values of X, the treatment assignment is deterministic, then for these values, the outcomes of at least one treatment could never be observed. In this case, it would be unable and meaningless to estimate the treatment effect. More specifically, suppose there are two treatments: Medicine A and Medicine B. Let us assume that patients with an age greater than 60 are always assigned Medicine A, and then it will be impossible and meaningless to study the outcome of Medicine B on those patients. In other words, the positivity assumption indicates the variability, which is important for treatment effect estimation.

In [6], the ignorability and positivity assumptions together are called strong ignorability or strongly ignorable treatment assignment.

With these assumptions, the relationship between the observed outcome and the potential outcome can be rewritten as

$$
\begin{array}{l} \mathbb {E} [ Y (W = w) | X = x ] = \mathbb {E} [ Y (W = w) | W = w, X = x ] (\text {Ignorability}) \tag {2.6} \\ = \mathbb {E} [ Y ^ {F} | W = w, X = x ], \\ \end{array}
$$

where $Y^{F}$ is the random variable of the observed outcome, and $Y(W = w)$ is the random variable of the potential outcome of treatment w. If we are interested in the potential outcome of one specific group (either the subgroup, the treated group, or the whole population), the potential outcome can be obtained by taking the expectation of the observed outcome over that group.

With the above equation, we can rewrite the treatment effect defined in Sect. 2.2.1 as follows:

$$
\mathrm{ITE} _ {i} = W _ {i} Y _ {i} ^ {F} - W _ {i} Y _ {i} ^ {C F} + (1 - W _ {i}) Y _ {i} ^ {C F} - (1 - W _ {i}) Y _ {i} ^ {F}
$$

$$
\begin{array}{l} \mathrm{ATE} = \mathbb {E} _ {X} \left[ \mathbb {E} [ Y ^ {F} | W = 1, X = x ] - \mathbb {E} [ Y ^ {F} | W = 0, X = x ] \right] \\ = \frac {1}{N} \sum_ {i} \left(Y _ {i} (W = 1) - Y _ {i} (W = 0)\right) = \frac {1}{N} \sum_ {i} \mathrm{ITE} _ {i} \\ \end{array}
$$

$$
\mathrm{ATT} = \mathbb {E} \chi_ {T} \left[ \mathbb {E} \left[ Y ^ {F} \mid W = 1, X = x \right] - \mathbb {E} \left[ Y ^ {F} \mid W = 0, X = x \right] \right] \tag {2.7}
$$

$$
= \frac {1}{N _ {T}} \sum_ {\{i: W _ {i} = 1 \}} (Y _ {i} (W = 1) - Y _ {i} (W = 0)) = \frac {1}{N _ {T}} \sum_ {\{i: W _ {i} = 1 \}} \mathrm{ITE} _ {i}
$$

$$
\mathrm{CATE} = \mathbb {E} [ Y ^ {F} | W = 1, X = x ] - \mathbb {E} [ Y ^ {F} | W = 0, X = x ]
$$

$$
= \frac {1}{N _ {x}} \sum_ {\{i: X _ {i} = x \}} (Y _ {i} (W = 1) - Y _ {i} (W = 0)) = \frac {1}{N _ {x}} \sum_ {\{i: X _ {i} = x \}} \mathrm{ITE} _ {i},
$$

where $Y_{i}(W=1)$ and $Y_{i}(W=0)$ are the potential treated/control outcomes of unit i, N is the total number of units in the whole population, $N_{T}$ is the number of units in the treated group, and $N_{x}$ is the number of units in the group with X = x. The second lines in the ATE, ATT, and CATE equations are their empirical estimations. Empirically, the ATE can be estimated as the average of ITE in the entire population. Similarly, ATT and CATE can be estimated as the average of ITE on the treated group and specific subgroups separately.

However, due to the fact that the potential treated/control outcomes can never be observed simultaneously, the key point in the treatment effect estimation is how to estimate the counterfactual outcome in ITE estimation or how to estimate the $\frac{1}{N_{*}}\sum_{i}Y_{i}(W=1)$ and $\frac{1}{N_{*}}\sum_{i}Y_{i}(W=0)$ , where $N_{*}$ denotes N, $N_{T}$ , or $N_{x}$ . In the following section, we will discuss the challenges in estimating these terms and briefly introduce the general solutions.

## 2.2.4 Confounders and General Solutions

As mentioned above, how to estimate the average potential treated/control outcome over a specific group is the core of causal inference. Let us take ATE as a case study: When estimating the ATE, a natural idea is to directly use the average of observed treated/control outcomes, i.e., $\hat{A}TE = \frac{1}{N_{T}} \sum_{i=1}^{N_{T}} Y_{i}^{F} - \frac{1}{N_{C}} \sum_{i=1}^{N_{C}} Y_{j}^{F}$ , where $N_{T}$ and $N_{C}$ is the number of units in the treated and control groups, respectively. However, due to the existence of confounders, there is a serious problem in this estimation: this calculated ATE includes a spurious effect brought by the confounders.

Definition 2.8 (Confounders) Confounders are the variables that affect both the treatment assignment and the outcome.

Confounders are some special pre-treatment variables, such as age in the medicine example. When directly using the average of observed treated/control outcome, the calculated ATE includes not only the effect of treatment on the outcome but also the effect of confounders on the outcome, which leads to a spurious effect. For example, in the medicine example, age is a confounder. Age affects the recovery rate: in general, young patients have a better chance of recovering than older patients. Age also affects the treatment choice: young patients may prefer to take medicine A, while older patients prefer medicine B, or for the same medicine, young patients have a different dosage from elderly patients. The observational data are shown in Table 2.1, and let us estimate ATE according to the above equation: $\hat{\mathrm{ATE}} = \frac{1}{N_A}\sum_{i=1}^{N_A}Y_i^F -\frac{1}{N_B}\sum_{i=1}^{N_B}Y_j^F = 289 / 350 - 273 / 350 = 5\%$ , where $N_{A}$ and $N_{B}$ is the number of patients taking Medicine A and B, respectively. However, we cannot conclude that Treatment A is more effective than Treatment B because the high average recovery rate of the group taking Treatment A may be caused by the fact that most patients in this group (270 out of 350) are young patients. Thus, the effect of age on the recovery rate is a spurious effect, as it is mistakenly counted into the effect of treatment on the outcome.

From Table 2.1, we can observe another interesting phenomenon, Simpson's paradox (or Simpson's reversal, Yule–Simpson effect, amalgamation paradox, reversal paradox) [2, 4], brought by the confounder. It can be observed that: in both the Young and Older patient groups, Medicine B has a higher recovery rate than Medicine A; but when combining these two groups, Medicine A is the one with a higher recovery rate. This paradox is caused by the confounder variable: When comparing the recovery rate in the whole group, most of the people taking medicine A are young, and the comparison shown in the table fails to eliminate the effect of age on the recovery rate.

**Table 2.1 An example to show the spurious effect of confounder variable Age [21]**

<table><tr><td>Age\Recovery rate\Treatment</td><td>Treatment A</td><td>Treatment B</td></tr><tr><td>Young</td><td>234/270 = 87%</td><td>81/87 = 92%</td></tr><tr><td>Older</td><td>55/80 = 69%</td><td>192/263 = 73%</td></tr><tr><td>Overall</td><td>289/350 = 83%</td><td>273/350 = 78%</td></tr></table>

In addition to the spurious effect in treatment effect estimation, confounders also cause problems in counterfactual outcome estimation. As shown in Eq. (2.7), counterfactual outcome estimation is an alternative way to estimate the ATE. Confounder variables cause selection bias, which makes counterfactual outcome estimation more difficult.

Selection bias is the phenomenon that the distribution of the observed group is not representative of the group we are interested in, i.e., $p(X_{obs}) \neq p(X_{*})$ , where $p(X_{obs})$ and $p(X_{*})$ are the distributions of the variables in the observed group and the interested group, respectively. Confounder variables affect units' treatment choices, which leads to selection bias. In the medicine example, age is a confounder variable, so that people of different ages have different treatment preferences. Figure 2.1 shows the age distribution of the observed treated/control group. Apparently, the age distribution of the observed treated group is different from the age distribution of the observed control group. This phenomenon exacerbates the difficulty of counterfactual outcome estimation as we need to estimate the control outcome of units in the treated group based on the observed control group, and, similarly, estimate the treated outcome of units in the control group based on the observed treated group. If we directly train the potential outcome estimation model $\hat{Y}(x, w) = f_{w}(x)$ on the data with W = w without handling the selection bias, the trained model would work poorly in estimating the potential outcome of W = w for the units in the other group. This problem brought by the selection is also named as covariate shift in the Machine Learning community.

Handing the problems caused by confounder variables is an essential part of causal inference, and the procedure of handing confounder variables is called adjusting confounders. The following part of this section briefly discusses the general solutions to tackle the above two problems caused by confounders under the ignorability assumption. The problem when there exist unobserved confounders will be discussed in Sect. 3.3.2.

To solve the spurious effect problem, we should take the effect of confounder variables on outcomes into consideration. A general approach along this direction first estimates the treatment effect conditioning on the confounder variables and then conducts weighted averaging over the confounder according to its distribution. To be more specific,

$$
\begin{array}{l} \hat {\mathrm{ATE}} = \sum_ {x} p (x) \mathbb {E} \left[ Y ^ {F} \mid X = x, W = 1 \right] - \sum_ {x} p (x) \mathbb {E} \left[ Y ^ {F} \mid X = x, W = 0 \right] \\ = \sum \chi^ {*} p (X \in \mathcal {X} ^ {*}) \left(\frac {1}{N _ {\{i : X _ {i} \in \mathcal {X} ^ {*} , W _ {i} = 1 \}}} \sum_ {\{i: X _ {i} \in \mathcal {X} ^ {*}, W _ {i} = 1 \}} Y _ {i} ^ {F}\right) \\ - \sum \chi^ {*} p (X \in \mathcal {X} ^ {*}) \left(\frac {1}{N _ {\{j : X _ {j} \in \mathcal {X} ^ {*} , W _ {j} = 1 \}}} \sum_ {\{j: X _ {j} \in \mathcal {X} ^ {*}, W _ {j} = 0 \}} Y _ {j} ^ {F}\right), \tag {2.8} \\ \end{array}
$$

where $X^{*}$ is a set of X values, $p(X \in \mathcal{X}^{*})$ is the probability of the background variables in $X^{*}$ over the whole population, and $\{i : x_{i} \in X^{*}, W_{i} = w\}$ is the subgroup of units whose background variable values belong to $X^{*}$ and treatment is equal to w. Stratification, which will be discussed in detail later, is a representative method of this category.

For the selection bias problem, there are two general approaches to solving it. The first general approach handles selection bias by creating a pseudogroup that is approximately close to the interested group. Possible methods include sample re-weighting, matching, tree-based methods, confounder balancing, balanced representation learning methods, and multi-task-based methods. The created pseudogroup alleviates the negative influence of the selection bias, and better counterfactual outcome estimations can be obtained. The other general approach first trains the base potential outcome estimation models solely on the observed data and then corrects the estimation bias caused by the selection bias. Meta-learning-based methods belong to this category.

## 2.3 Summary

This chapter reviews the basic concepts, assumptions, and formal definitions in causal inference, focusing on the potential outcome framework. Moreover, illustrative examples are provided, which help readers understand the challenges in causal inference.

## References

1. N. Altman, M. Krzywinski, Points of significance: association, correlation and causation. Nat. Methods 12(10), 899–900 (2015)

2. C.R. Blyth, On Simpson's paradox and the sure-thing principle. J. Am. Stat. Assoc. 67(338), 364–366 (1972)  
3. D.M. Chickering, Optimal structure identification with greedy search. J. Mach. Learn. Res. 3, 507–554 (2003). ISSN: 1532-4435. https://doi.org/10.1162/153244303321897717  
4. I.J. Good, Y. Mittal et al., The amalgamation and geometry of two-by-two contingency tables. Ann. Stat. 15(2), 694–711 (1987)  
5. P.O. Hoyer et al., Nonlinear causal discovery with additive noise models, in Advances in Neural Information Processing Systems, 2009, pp. 689–696  
6. G.W. Imbens, D.B. Rubin, Causal Inference in Statistics, Social, and Biomedical Sciences (Cambridge University Press, Cambridge, 2015)  
7. F. Johansson, U. Shalit, D. Sontag, Learning representations for counterfactual inference, in International Conference on Machine Learning, 2016, pp. 3020–3029  
8. J. Pearl, Judea Pearl on Potential Outcomes http://causality.cs.ucla.edu/blog/index.php/2012/12/03/judea-pearl-on-potential-outcomes/ (2012)  
9. J. Pearl, Causal diagrams for empirical research. Biometrika 82(4), 669–688 (1995)  
10. J. Pearl, Causal inference in statistics: an overview. Stat. Surv. 3, 96–146 (2009)  
11. J. Pearl, Causality (Cambridge University Press, Cambridge, England 2009)  
12. J. Pearl, Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference (Elsevier, 2014)  
13. J. Ramsey et al., A million variables and more: the Fast Greedy Equivalence Search algorithm for learning high-dimensional graphical causal models, with an application to functional magnetic resonance images. Int. J. Data Sci. Anal. 3(2), 121–129 (2017)  
14. D.B. Rubin, Estimating causal effects of treatments in randomized and nonrandomized studies. J. Educ. Psychol. 66(5), 688 (1974)  
15. P. Schwab et al., Learning counterfactual representations for estimating individual dose-response curves, in The Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI Press, 2020), pp. 5612–5619  
16. U. Shalit, F.D. Johansson, D. Sontag, Estimating individual treatment effect: generalization bounds and algorithms, in Proceedings of the 34th International Conference on Machine Learning-Volume 70 (2017), pp. 3076–3085  
17. P. Spirtes, K. Zhang, Causal discovery and inference: concepts and recent methodological advances, in Applied Informatics, vol. 3 (Springer. 2016), p. 3  
18. P. Spirtes et al., Causation, Prediction, and Search (MIT Press, Cambridge, MA, 2000)  
19. J. Splawa-Neyman, D.M. Dabrowska, T.P. Speed, On the application of probability theory to agricultural experiments. Essay on principles. Section 9, in Statistical Science, JSTOR (1990), pp. 465–472  
20. M. Stephen, W. Christopher, Counterfactuals and Causal Inference: Methods and Principles for Social Research (Cambridge University Press, Cambridge, 2007)  
21. L. Yao et al., A survey on causal inference. ACM Trans. Knowl. Dis. Data (TKDD) 15(5), 1–46 (2021)  
22. K. Zhang, A. Hyvarinen, On the identifiability of the post-nonlinear causal model, in 25th Conference on Uncertainty in Artificial Intelligence (AUAI Press, 2009), pp. 647–655

## Part II

## Machine Learning and Causal Effect

## Estimation
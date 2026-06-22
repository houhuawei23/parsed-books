# 第2章 因果推断基础（Chapter 2 Causal Inference Preliminary）

![image_02](images/image_02.png)

刘毅瑶（Liuyi Yao），储志轩（Zhixuan Chu），李亚亮（Yaliang Li），高静（Jing Gao），张爱东（Aidong Zhang），李晟（Sheng Li）

## 2.1 引言（Introduction）

在日常用语中，**相关性（correlation）**和**因果关系（causality）**通常被混用，尽管它们具有截然不同的含义。相关性表示一种一般关系：当两个变量呈现上升或下降趋势时，它们之间存在相关性 $[1]$。因果关系也被称为因果效应（cause and effect），其中原因对结果负有部分责任，结果则部分依赖于原因。**因果推断（causal inference）**是基于结果发生的条件得出关于因果联系结论的过程。因果推断与相关性推断的主要区别在于，前者分析当原因发生变化时结果变量的响应 $[10, 20]$。

众所周知，“相关性并不意味着因果关系（correlation does not imply causation）”。例如，一项研究表明，通常吃早餐的女孩比不吃早餐的女孩体重更轻，从而得出早餐有助于减肥的结论。但事实上，这两个事件可能只是具有相关性而非因果关系。也许每天吃早餐的女孩拥有更健康的生活方式，包括规律运动、

刘毅瑶（L. Yao）· 李亚亮（Y. Li）
阿里巴巴集团，杭州，中国
电子邮箱：yly287738@alibaba-inc.com；yaliang.li@alibaba-inc.com

储志轩（Z. Chu）
蚂蚁集团，杭州，中国
电子邮箱：chuzhixuan.czx@alibaba-inc.com

高静（J. Gao）
普渡大学，西拉法叶，印第安纳州，美国
电子邮箱：jinggao@purdue.edu

张爱东（A. Zhang）· 李晟（S. Li）（☒）
弗吉尼亚大学，夏洛茨维尔，弗吉尼亚州，美国
电子邮箱：aidong@virginia.edu；shengli@virginia.edu

良好的睡眠习惯和均衡的饮食，这些最终使她们体重较轻。在这种情况下，吃早餐和体重较轻共享一个共同原因；因此，我们可以将拥有更健康的生活方式视为吃早餐与体重较轻之间因果关系的**混杂因子（confounder）**。

在许多情况下，一个行为似乎显然会导致另一个结果；然而，也存在许多我们难以厘清并确定其关系的情况。因此，学习因果关系是一个极具挑战性的问题。推断因果关系最有效的方法是进行**随机对照试验（randomized controlled trial）**，即将参与者随机分配到**处理组（treatment group）**或**对照组（control group）**。由于进行的是随机研究，对照组和处理组之间唯一预期的差异就是被研究的结果变量。然而，在现实中，随机对照试验通常耗时且昂贵，因此研究无法涉及大量受试者，这可能无法代表处理/干预最终将针对的现实世界人群。另一个问题是，随机对照试验仅关注样本的平均值，而不解释个体受试者的机制。此外，在大多数随机对照试验中还需要考虑伦理问题，这极大地限制了其应用。因此，作为随机对照试验的替代方案，**观测数据（observational data）**成为一种诱人的捷径。观测数据是由研究人员在不进行任何干预的情况下仅观察受试者而获得的。这意味着研究人员无法控制处理和受试者，他们只是观察受试者并根据观察记录数据。从观测数据中，我们可以发现受试者的行为、结果以及有关已发生事件的信息，但无法弄清他们采取特定行为的机制。对于观测数据，核心问题是如何获得**反事实结果（counterfactual outcome）**。例如，我们想要回答这个问题：“如果这位患者接受了不同的药物治疗，他是否会有不同的结果？”回答这样的反事实问题具有挑战性，原因有二 $[15]$：第一，我们只能观察到事实结果（factual outcome），而永远无法观察到如果他们选择了不同治疗方案可能发生的反事实结果。第二，在观测数据中，处理通常不是随机分配的，这可能导致接受处理的人群与总体人群存在显著差异。

为了解决从观测数据中进行因果推断的这些问题，研究人员开发了多种框架，包括**潜在结果框架（potential outcome framework）** $[14, 19]$ 和**结构因果模型（structural causal model, SCM）** $[9, 11, 12]$。潜在结果框架也称为**内曼–鲁宾潜在结果（Neyman–Rubin potential outcomes）**或**鲁宾因果模型（Rubin causal model）**。在我们上面提到的例子中，如果一个女孩每天正常吃早餐，她会有特定的体重；而如果她不正常吃早餐，她会有不同的体重。为了衡量正常吃早餐对一个女孩的因果效应，我们需要比较同一个人在这两种情境下的结果。显然，不可能同时看到两种潜在结果，其中一个潜在结果总是缺失的。潜在结果框架旨在估计这些潜在结果，然后计算**处理效应（treatment effect）**。因此，在潜在结果框架下，处理效应估计是因果推断的核心问题之一。因果推断中另一个有影响力的框架是**结构因果模型（Structural Causal Model, SCM）**，它包括**因果图（causal graph）**和**结构方程（structural equations）**。结构因果模型描述了一个系统的因果机制，其中一组变量及其之间的因果关系通过一组联立的结构方程进行建模。学习因果关系的另一条路线是**因果结构学习（causal structure learning）**，其目标是通过生成因果图来揭示因果关系。代表性方法可分为三类，包括**基于约束的模型（constraint-based models）** $[18]$、**基于分数的模型（score-based models）** $[3, 13]$ 和**函数因果模型（functional causal models）** $[5, 22]$。与因果效应估计不同，因果结构学习解决的是另一类问题，这超出了本综述的范围；更多信息请参见 $[17]$。

因果推断与机器学习有着密切的关系。近年来，机器学习领域的蓬勃发展促进了因果推断领域的发展。强大的机器学习方法，如**决策树（decision trees）**、**集成方法（ensemble methods）**和**深度神经网络（deep neural networks）**，被应用于更准确地估计潜在结果。除了改进结果估计模型外，机器学习方法还提供了处理混杂因子的新视角。得益于最近的**深度表示学习（deep representation learning）**方法，通过为所有协变量学习平衡表示来调整混杂变量，使得在给定学习到的表示的条件下，处理分配与混杂变量无关。在机器学习中，数据越多越好。然而，在因果推断中，仅靠更多的数据是不够的。拥有更多数据只能帮助获得更精确的估计，但不能确保这些估计是正确且无偏的。机器学习方法促进了因果推断的发展；同时，因果推断也有助于机器学习方法。单纯追求预测准确性对于现代机器学习研究是不够的，正确性和可解释性也是机器学习方法的目标。因果推断正开始帮助改进机器学习，例如**推荐系统（recommender systems）**或**强化学习（reinforcement learning）**。

在本章中，我们提供对因果推断方法的全面回顾。我们介绍基本概念以及识别因果效应的三个关键假设。

## 2.2 因果推断基础（Basics of Causal Inference）

在本节中，我们介绍因果推断的背景知识，包括任务描述、数学符号、假设、挑战和一般解决方案。我们还给出了一个说明性示例，该示例将在本综述中全程使用。

通常，因果推断的任务是估计如果应用了另一种处理，结果会发生的变化。例如，假设有两种处理可以应用于患者：药物 A 和药物 B。当对感兴趣的患者队列应用药物 A 时，**恢复率（recovery rate）**为 70%，而对同一队列应用药物 B 时，恢复率为 90%。恢复率的变化就是处理（即本例中的药物）对恢复率产生的效应。

上述例子描述了衡量处理效应的理想情况：对同一队列应用不同的处理。在现实场景中，这种理想情况只能通过**随机实验（randomized experiment）**来近似，其中处理分配是受控的，例如完全随机分配。通过这种方式，接受特定处理的组可以被视为我们感兴趣队列的近似。

然而，进行随机实验成本高昂、耗时，有时甚至不道德。因此，由于观测数据的广泛可用性，从观测数据中估计处理效应已引起越来越多的关注。观测数据通常包含一组采取了不同处理的个体、他们相应的结果，以及可能更多的信息，但无法直接获得他们为何采取特定处理的原因/机制。这种观测数据使研究人员能够在不进行随机实验的情况下，研究学习特定处理因果效应的基本问题。为了更好地介绍各种处理效应估计方法，以下部分介绍几个定义，包括**单元（unit）**、**处理（treatment）**、**结果（outcome）**、**处理效应（treatment effect）**以及观测数据提供的其他信息（**处理前变量（pre-treatment variables）**和**处理后变量（post-treatment variables）**）。

## 2.2.1 定义（Definitions）

这里我们在潜在结果框架 $[14, 19]$ 下定义符号，该框架与另一个框架——结构因果模型框架 $[8]$ 在逻辑上是等价的。潜在结果框架的基础是因果关系与应用于单元的**处理（或行动、操作、干预）**相关联 $[6]$。处理效应是通过比较单元在不同处理下的潜在结果获得的。下面，我们首先介绍因果推断中的三个基本概念：单元、处理和结果。

**定义 2.1（单元）** 单元是处理效应研究中的原子研究对象。

单元可以是一个物理对象、一家公司、一位患者、一个个体，或者对象或个体的集合，例如一个教室或一个市场，在特定的时间点 $[6]$。在潜在结果框架下，不同时间点的原子研究对象是不同的单元。数据集中的一个单元是整个总体中的一个样本，因此在本综述中，术语“样本（sample）”和“单元（unit）”可互换使用。

**定义 2.2（处理）** 处理是指应用于（暴露或施加于）单元的行动。

令 $W$ （ $W \in \{0, 1, 2, \ldots, N_W\}$ ）表示处理，其中 $N_W + 1$ 是可能处理的总数。在上述药物例子中，药物 A 是一种处理。大多数文献考虑**二元处理（binary treatment）**，在这种情况下，应用处理 $W = 1$ 的单元组是**处理组（treated group）**，而 $W = 0$ 的单元组是**对照组（control group）**。

**定义 2.3（潜在结果）** 对于每个单元-处理对，当该处理应用于该单元时的结果就是潜在结果 [6]。

值为 $w$ 的处理的潜在结果记为 $Y(W = w)$ 。

**定义 2.4（观测结果）** 观测结果是实际应用的处理的结果。

观测结果也称为**事实结果（factual outcome）**，我们用 $Y^{F}$ 表示，其中 F 代表“事实的（factual）”。潜在结果与观测结果之间的关系是 $Y^{F} = Y(W = w)$ ，其中 $w$ 是实际应用的处理。

**定义 2.5（反事实结果）** 反事实结果是如果单元采取了另一种处理时的结果。

反事实结果是单元实际采取的处理之外的处理的潜在结果。由于一个单元只能采取一种处理，因此只能观测到一个潜在结果，其余未观测到的潜在结果就是反事实结果。在多处理情况下，令 $Y^{CF}(W = w')$ 表示值为 $w'$ 的处理的**反事实结果（counterfactual outcome）**。在二元处理情况下，为简化符号，我们用 $Y^{CF}$ 表示反事实结果，且 $Y^{CF} = Y(W = 1 - w)$ ，其中 $w$ 是单元实际采取的处理。

在观测数据中，除了所选处理和观测结果外，还记录了单元的其他信息，这些信息可分为**处理前变量（pre-treatment variables）**和**处理后变量（post-treatment variables）**。

**定义 2.6（处理前变量）** 处理前变量是不受处理影响的变量。

处理前变量也称为**背景变量（background variables）**，可以是患者的人口统计信息、病史等。令 $X$ 表示处理前变量。

**定义 2.7（处理后变量）** 处理后变量是受处理影响的变量。

处理后变量的一个例子是**中间结果（intermediate outcome）**，例如上述药物例子中服药后的实验室检查结果。

在以下各节中，除非另有说明，术语“变量（variable）”指的是处理前变量。

**处理效应（Treatment Effect）** 在介绍了观测数据和关键术语之后，可以使用上述定义对处理效应进行量化定义。

处理效应可以在**总体（population）**、**处理组（treated group）**、**子组（subgroup）**和**个体（individual）**层面进行衡量。为了阐明这些定义，这里我们在二元处理下定义处理效应，并且可以通过比较潜在结果将其扩展到多处理情况。

在总体层面，处理效应称为**平均处理效应（Average Treatment Effect, ATE）**，定义为

$$
\mathrm{ATE} = \mathbb {E} [ \mathbf {Y} (W = 1) - \mathbf {Y} (W = 0) ], \tag {2.1}
$$

其中 $\mathbf{Y}(W = 1)$ 和 $\mathbf{Y}(W = 0)$ 分别是整个总体的潜在处理结果和对照结果。

对于处理组，处理效应称为**处理组上的平均处理效应（Average Treatment Effect on the Treated group, ATT）**，定义为

$$
\mathrm{ATT} = \mathbb {E} [ \mathbf {Y} (W = 1) | W = 1 ] - \mathbb {E} [ \mathbf {Y} (W = 0) | W = 1 ], \tag {2.2}
$$

其中 $\mathbf{Y}(W = 1)|W = 1$ 和 $\mathbf{Y}(W = 0)|W = 1$ 分别是处理组的潜在处理结果和对照结果。

在子组层面，处理效应称为**条件平均处理效应（Conditional Average Treatment Effect, CATE）**，定义为

$$
\mathrm{CATE} = \mathbb {E} [ \mathbf {Y} (W = 1) | X = x ] - \mathbb {E} [ \mathbf {Y} (W = 0) | X = x ], \tag {2.3}
$$

其中 $\mathbf{Y}(W=1)|X=x$ 和 $\mathbf{Y}(W=0)|X=x$ 分别是子组 $X=x$ 的潜在处理结果和对照结果。当处理效应在不同子组间变化时，CATE 是一种常见的处理效应度量，也称为**异质性处理效应（heterogeneous treatment effect）**。

在个体层面，处理效应称为**个体处理效应（Individual Treatment Effect, ITE）**，单元 $i$ 的 ITE 定义为

$$
\mathrm{ITE} _ {i} = Y _ {i} (W = 1) - Y _ {i} (W = 0), \tag {2.4}
$$

其中 $Y_{i}(W = 1)$ 和 $Y_{i}(W = 0)$ 分别是单元 $i$ 的潜在处理结果和对照结果。在一些工作中 [7, 16]，ITE 被视为 CATE。

**目标（Objective）** 对于因果推断，我们的目标是从观测数据中估计处理效应。形式化地说，给定观测数据集 $\left\{X_{i}, W_{i}, Y_{i}^{F}\right\}_{i=1}^{N}$ ，其中 $N$ 是数据集中单元的总数，因果推断任务的目标是估计上述定义的处理效应。

## 2.2.2 说明性示例（An Illustrative Example）

为了更好地说明因果推断，我们使用以下示例结合上述定义的符号进行概述。在此示例中，我们希望通过利用观测数据（即**电子健康记录（electronic health records）**）来评估几种不同药物治疗一种疾病的处理效应，这些数据包括患者的人口统计信息、患者服用的具体药物及具体剂量，以及医学检查结果。显然，我们从电子健康记录中只能获得一个特定患者的一个事实结果，因此核心任务是预测如果患者接受了另一种处理（即不同的药物或相同药物的不同剂量）会发生什么。回答这样的反事实问题非常具有挑战性。因此，我们希望使用因果推断来预测每个患者在所有不同剂量药物下的所有潜在结果。然后，我们可以合理且准确地评估和比较不同药物治疗该疾病的处理效应。

需要特别记住的一点是，每种药物可能有不同的剂量。例如，对于药物 A，剂量范围可以是区间 $[a, b]$ 内的连续变量；而对于药物 B，剂量可以是具有几种特定剂量方案的分类变量。

在上述示例中，单元是患有研究疾病的患者。处理是指针对该疾病的不同药物及特定剂量，我们用 $W$ （ $W \in \{0, 1, 2, \dots, N_W\}$ ）表示这些处理。例如，$W_i = 1$ 可以表示单元 $i$ 服用的特定剂量的药物 $A$，$W_i = 2$ 表示单元 $i$ 服用的特定剂量的药物 $B$。$Y$ 是结果，例如一种可以衡量药物摧毁疾病并导致患者康复能力的血液检查。令 $Y_i(W = 1)$ 表示特定剂量的药物 $A$ 对患者 $i$ 的潜在结果。患者的特征可能包括年龄、性别、临床表现和其他一些医学检查等。在这些特征中，年龄、性别和其他人口统计信息是处理前变量，不受服药影响。一些临床表现和医学检查会受到服药影响，它们是处理后变量。在此示例中，我们的目标是根据提供的观测数据估计不同药物治疗该疾病的处理效应。

在以下各节中，我们将继续使用此示例来解释更多概念并说明各种因果推断方法背后的直觉。

## 2.2.3 假设（Assumptions）

为了估计处理效应，因果推断文献中通常使用以下假设。

**假设（稳定单元处理值假设（Stable Unit Treatment Value Assumption, SUTVA））** 任何单元的潜在结果不会因分配给其他单元的处理而变化，并且对于每个单元，每个处理水平没有不同的形式或版本导致不同的潜在结果。

该假设强调两点：第一点是每个单元的独立性，即单元之间不存在交互作用。在上述说明性示例的背景下，一个患者的结果不会影响其他患者的结果。

第二点是每个处理的单一版本。在上述示例中，在 SUTVA 假设下，不同剂量的药物 A 是不同的处理。

**假设（可忽略性（Ignorability））** 给定背景变量 $X$，处理分配 $W$ 与潜在结果独立，即 $W \perp Y(W = 0)$ ， $Y(W = 1)|X$ 。

在上述说明性示例的背景下，这个可忽略性假设表明两个方面：首先，如果两个患者具有相同的背景变量 $X$，无论处理分配如何，他们的潜在结果应该相同，即 $p(Y_{i}(0), Y_{i}(1)|X = x, W = W_{i}) = p(Y_{j}(0), Y_{j}(1)|X = x, W = W_{j})$ 。类似地，如果两个患者具有相同的背景变量值，无论他们拥有的潜在结果值如何，他们的处理分配机制应该相同，即 $p(W|X = x, Y_{i}(0), Y_{i}(1)) = p(W|X = x, Y_{j}(0), Y_{j}(1))$ 。可忽略性假设也称为**无混杂性假设（unconfoundedness assumption）**。有了这个无混杂性假设，对于具有相同背景变量 $X$ 的单元，他们的处理分配可以被视为随机的。

**假设（积极性（Positivity））** 对于任何 $X$ 的值，处理分配不是确定性的：

$$
P (W = w | X = x) > 0, \quad \forall w \text {   and   } x. \tag {2.5}
$$

如果对于某些 $X$ 的值，处理分配是确定性的，那么对于这些值，至少一种处理的结果将永远无法观测到。在这种情况下，估计处理效应将是不可能的且无意义的。更具体地说，假设有两种处理：药物 A 和药物 B。让我们假设年龄大于 60 岁的患者总是被分配药物 A，那么研究药物 B 对这些患者的结果将是不可能的且无意义的。换句话说，积极性假设表明了**变异性（variability）**，这对于处理效应估计很重要。

在 [6] 中，可忽略性假设和积极性假设一起被称为**强可忽略性（strong ignorability）**或**强可忽略处理分配（strongly ignorable treatment assignment）**。

在这些假设下，观测结果与潜在结果之间的关系可以重写为

$$
\begin{array}{l} \mathbb {E} [ Y (W = w) | X = x ] = \mathbb {E} [ Y (W = w) | W = w, X = x ] (\text {可忽略性}) \tag {2.6} \\ = \mathbb {E} [ Y ^ {F} | W = w, X = x ], \\ \end{array}
$$

其中 $Y^{F}$ 是观测结果的随机变量，$Y(W = w)$ 是处理 $w$ 的潜在结果的随机变量。如果我们对某个特定组（子组、处理组或整个总体）的潜在结果感兴趣，可以通过对该组取观测结果的期望来获得潜在结果。

利用上述方程，我们可以将第 2.2.1 节中定义的处理效应重写如下：

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

其中 $Y_{i}(W=1)$ 和 $Y_{i}(W=0)$ 是单元 $i$ 的潜在处理/对照结果，$N$ 是整个总体中单元的总数，$N_{T}$ 是处理组中单元的数量，$N_{x}$ 是 $X = x$ 组中单元的数量。ATE、ATT 和 CATE 方程中的第二行是其经验估计。经验上，ATE 可以估计为整个总体中 ITE 的平均值。类似地，ATT 和 CATE 可以分别估计为处理组和特定子组上 ITE 的平均值。

然而，由于潜在处理/对照结果永远无法同时被观测到，处理效应估计的关键点在于如何估计 ITE 中的反事实结果，或者如何估计 $\frac{1}{N_{*}}\sum_{i}Y_{i}(W=1)$ 和 $\frac{1}{N_{*}}\sum_{i}Y_{i}(W=0)$ ，其中 $N_{*}$ 表示 $N$、$N_{T}$ 或 $N_{x}$。在下一节中，我们将讨论估计这些项的挑战并简要介绍一般解决方案。

## 2.2.4 混淆变量与通用解决方案（Confounders and General Solutions）

如上所述，如何估计特定群体上的平均潜在处理/控制结果是因果推断的核心。我们以 **ATE** 作为案例进行研究：在估计 ATE 时，一个自然的想法是直接使用观测到的处理/控制结果的平均值，即 $\hat{ATE} = \frac{1}{N_{T}} \sum_{i=1}^{N_{T}} Y_{i}^{F} - \frac{1}{N_{C}} \sum_{i=1}^{N_{C}} Y_{j}^{F}$ ，其中 $N_{T}$ 和 $N_{C}$ 分别是处理组和控制组中的单元数量。然而，由于 **混淆变量（confounders）** 的存在，这种估计存在一个严重问题：计算出的 ATE 包含了由混淆变量带来的 **虚假效应（spurious effect）**。

**定义 2.8（混淆变量）** 混淆变量是同时影响处理分配和结果的变量。

混淆变量是一些特殊的处理前变量，例如医学例子中的年龄。当直接使用观测到的处理/控制结果的平均值时，计算出的 ATE 不仅包含了处理对结果的影响，还包含了混淆变量对结果的影响，从而导致虚假效应。例如，在医学例子中，年龄是一个混淆变量。年龄影响康复率：通常，年轻患者比老年患者有更好的康复机会。年龄也影响治疗选择：年轻患者可能更喜欢服用药物 A，而老年患者更喜欢药物 B，或者对于同一种药物，年轻患者与老年患者的剂量不同。观测数据如表 2.1 所示，让我们根据上述方程估计 ATE：$\hat{\mathrm{ATE}} = \frac{1}{N_A}\sum_{i=1}^{N_A}Y_i^F -\frac{1}{N_B}\sum_{i=1}^{N_B}Y_j^F = 289 / 350 - 273 / 350 = 5\%$ ，其中 $N_{A}$ 和 $N_{B}$ 分别是服用药物 A 和药物 B 的患者数量。然而，我们不能得出结论说治疗 A 比治疗 B 更有效，因为接受治疗 A 的组的高平均康复率可能是由于该组中大多数患者（350 人中有 270 人）是年轻患者造成的。因此，年龄对康复率的影响是一个虚假效应，因为它被错误地计入了处理对结果的影响中。

从表 2.1 中，我们可以观察到由混淆变量带来的另一个有趣现象，即 **辛普森悖论（Simpson's paradox）**（或辛普森反转、尤尔-辛普森效应、合并悖论、反转悖论）[2, 4]。可以观察到：在年轻患者组和老年患者组中，药物 B 的康复率均高于药物 A；但当合并这两组时，药物 A 的康复率更高。这个悖论是由混淆变量引起的：当比较整个组的康复率时，服用药物 A 的大多数人是年轻人，表中显示的比较未能消除年龄对康复率的影响。

**表 2.1 展示混淆变量年龄的虚假效应的示例 [21]**

| 年龄\康复率\治疗 | 治疗 A | 治疗 B |
| :--- | :--- | :--- |
| 年轻 | 234/270 = 87% | 81/87 = 92% |
| 老年 | 55/80 = 69% | 192/263 = 73% |
| 总体 | 289/350 = 83% | 273/350 = 78% |

除了在处理效应估计中的虚假效应外，混淆变量还会在反事实结果估计中引发问题。如公式 (2.7) 所示，反事实结果估计是估计 ATE 的另一种方式。混淆变量会导致 **选择偏差（selection bias）**，这使得反事实结果估计更加困难。

选择偏差是指观测组的分布不能代表我们感兴趣组的分布的现象，即 $p(X_{obs}) \neq p(X_{*})$ ，其中 $p(X_{obs})$ 和 $p(X_{*})$ 分别是观测组和感兴趣组中变量的分布。混淆变量影响单元的治疗选择，从而导致选择偏差。在医学例子中，年龄是一个混淆变量，因此不同年龄的人有不同的治疗偏好。图 2.1 显示了观测到的处理组/控制组的年龄分布。显然，观测到的处理组的年龄分布与观测到的控制组的年龄分布不同。这种现象加剧了反事实结果估计的难度，因为我们需要基于观测到的控制组来估计处理组中单元的控制结果，并且类似地，基于观测到的处理组来估计控制组中单元的处理结果。如果我们直接在 $W = w$ 的数据上训练潜在结果估计模型 $\hat{Y}(x, w) = f_{w}(x)$ 而不处理选择偏差，那么训练好的模型在估计另一组中 $W = w$ 的潜在结果时会表现得很差。由选择带来的这个问题在机器学习社区中也称为 **协变量偏移（covariate shift）**。

处理由混淆变量引起的问题是因果推断的一个关键部分，处理混淆变量的过程称为 **调整混淆变量（adjusting confounders）**。本节接下来的部分简要讨论了在 **可忽略性假设（ignorability assumption）** 下解决由混淆变量引起的上述两个问题的通用解决方案。当存在未观测到的混淆变量时的问题将在第 3.3.2 节中讨论。

为了解决虚假效应问题，我们应该考虑混淆变量对结果的影响。沿着这个方向的一个通用方法首先估计在混淆变量条件下的处理效应，然后根据混淆变量的分布对其进行加权平均。更具体地说，

$$
\begin{array}{l} \hat {\mathrm{ATE}} = \sum_ {x} p (x) \mathbb {E} \left[ Y ^ {F} \mid X = x, W = 1 \right] - \sum_ {x} p (x) \mathbb {E} \left[ Y ^ {F} \mid X = x, W = 0 \right] \\ = \sum \chi^ {*} p (X \in \mathcal {X} ^ {*}) \left(\frac {1}{N _ {\{i : X _ {i} \in \mathcal {X} ^ {*} , W _ {i} = 1 \}}} \sum_ {\{i: X _ {i} \in \mathcal {X} ^ {*}, W _ {i} = 1 \}} Y _ {i} ^ {F}\right) \\ - \sum \chi^ {*} p (X \in \mathcal {X} ^ {*}) \left(\frac {1}{N _ {\{j : X _ {j} \in \mathcal {X} ^ {*} , W _ {j} = 1 \}}} \sum_ {\{j: X _ {j} \in \mathcal {X} ^ {*}, W _ {j} = 0 \}} Y _ {j} ^ {F}\right), \tag {2.8} \\ \end{array}
$$

其中 $X^{*}$ 是一组 X 值， $p(X \in \mathcal{X}^{*})$ 是背景变量在 $X^{*}$ 中占总体人口的概率， $\{i : x_{i} \in X^{*}, W_{i} = w\}$ 是背景变量值属于 $X^{*}$ 且处理等于 w 的单元的子组。**分层法（Stratification）** 是这类方法中的一个代表性方法，稍后将详细讨论。

对于选择偏差问题，有两种通用的解决方法。第一种通用方法通过创建一个与感兴趣组近似接近的 **伪组（pseudogroup）** 来处理选择偏差。可能的方法包括 **样本重加权（sample re-weighting）**、**匹配（matching）**、**基于树的方法（tree-based methods）**、**混淆变量平衡（confounder balancing）**、**平衡表示学习方法（balanced representation learning methods）** 和 **基于多任务的方法（multi-task-based methods）**。创建的伪组减轻了选择偏差的负面影响，从而可以获得更好的反事实结果估计。另一种通用方法首先仅在观测数据上训练基础潜在结果估计模型，然后校正由选择偏差引起的估计偏差。**基于元学习的方法（Meta-learning-based methods）** 属于这一类。

## 2.3 总结（Summary）

本章回顾了因果推断中的基本概念、假设和形式化定义，重点介绍了 **潜在结果框架（potential outcome framework）**。此外，还提供了说明性示例，帮助读者理解因果推断中的挑战。

## 参考文献（References）

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

## 第二部分（Part II）

## 机器学习与因果效应（Machine Learning and Causal Effect）

## 估计（Estimation）
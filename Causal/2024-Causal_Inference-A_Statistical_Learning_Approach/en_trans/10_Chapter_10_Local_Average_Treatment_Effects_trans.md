# 第10章 局部平均处理效应（Local Average Treatment Effects）

**工具变量回归（Instrumental variable regression）** 通常用于估计内生处理（endogenous treatment）的效应。在上一章中，我们了解了，给定图9.3所描绘的结构方程模型以及控制处理 $W _ { i }$ 与结果 $Y _ { i }$ 之间效应的线性设定（9.11），我们可以使用工具 $Z _ { i }$ 将处理效应参数 $\tau$ 识别为协方差之比：

$$
\tau = \operatorname{Cov} \left[ Y _ {i}, Z _ {i} \right] / \operatorname{Cov} \left[ W _ {i}, Z _ {i} \right], \tag {10.1}
$$

并通过下式一致地估计 $\tau$：

$$
\hat {\tau} _ {I V} = \widehat {\mathrm{Cov}} \left[ Y _ {i}, Z _ {i} \right] / \widehat {\mathrm{Cov}} \left[ W _ {i}, Z _ {i} \right]. \tag {10.2}
$$

然而，通常来说，因果推断领域的研究者往往对仅在线性模型中被定义和理解的**目标估计量（target estimands）** 的解释持怀疑态度；因此，在本章中，我们将重新审视对**工具变量估计量（instrumental variable estimator）** $\hat { \tau } _ { I V }$ 的分析，而不假设线性性——或者等价地说，在假设（9.11）可能被错误设定的情况下。

在没有线性性的情况下，估计量 $\hat { \tau } _ { I V }$ 仍然收敛于一个大样本极限：

$$
\hat {\tau} _ {I V} \rightarrow \tau_ {L A T E} := \operatorname{Cov} \left[ Y _ {i}, Z _ {i} \right] / \operatorname{Cov} \left[ W _ {i}, Z _ {i} \right] \tag {10.3}
$$

只要 Cov $[ W _ { i } , Z _ { i } ] \neq 0$；然而，如何解释这个极限不再一目了然。在本章中，我们将研究这个极限量是什么，以及何时可以将其理解为因果量。我们将考察一系列经济模型，在这些模型中，进入处理的内生选择（endogenous selection into treatment）可能是一个问题，并发现——在相当弱的假设下——这个极限是一个**加权处理效应（weighted treatment effect）**，其权重取决于那些控制每个单元对工具所给予的推动（nudge）响应程度的（未观测到的）属性。遵循 Imbens 和 Angrist [1994] 的观点，当这些条件成立时，我们将这个极限称为**局部平均处理效应（Local Average Treatment Effect, LATE）**，即对工具响应的那些单元的"局部"处理效应。

## 10.1 随机试验中的不依从性（Non-compliance in randomized trials）

我们可以讨论使用工具变量进行**非参数识别（non-parametric identification）** 的最简单场景，是在存在不依从性（non-compliance）的情况下估计**二元处理（binary treatment）** 的效应。例如，假设我们设立了一项随机研究来检验服用药物降低胆固醇的效果。但是，尽管我们随机分配了处理，有些人却不遵守随机化：一些被分配药物的受试者可能没有服药，而另一些被分配对照组的受试者可能自行购买降胆固醇药物。在这种情况下，我们有：

• 结果 $Y _ { i } \in \mathbb { R }$ ，具有通常的解释；
• 实际接受的**处理（treatment）** $W _ { i } \in \{ 0 , 1 \}$（即，受试者是否服用了药物），由于不依从性，这不是随机的；以及
• 被分配的**处理（assigned treatment）** $Z _ { i } \in \{ 0 , 1 \}$ ，这是随机的。

分析这类数据的一种流行方法是使用工具变量，我们将处理分配 $Z _ { i }$ 解释为对实际接受的处理 $W _ { i }$ 的一个外生"推动"（exogenous "nudge"）。

如果人们相信上一章中考虑的部分线性结构模型（9.11），那么只要分配的处理确实推动了接受的处理，即 Cov $[ W _ { i } , Z _ { i } ] \neq 0$ ，就可以通过（10.3）一致地估计 $\tau$ 。然而，在实践中，人们可能怀疑**常数处理效应假设（constant treatment effect assumption）**（9.11）的有效性，并怀疑那些依从处理的人与不依从的人对处理的反应不同。例如，可能存在一类患者，他们选择依从是因为知道自己会从治疗中获益良多；或者相反，其他患者可能选择不依从是因为知道自己承受着被治疗伤害的不成比例的风险。

**不依从性下的潜在结果（Potential outcomes under non-compliance）** 一种更严谨的方法从写出潜在结果开始。首先，由于 $W _ { i }$ 是非随机的并且可能对 $Z _ { i }$ 做出响应，我们需要针对处理变量（相对于工具）的潜在结果，即存在 $\{ W _ { i } ( 0 ) , W _ { i } ( 1 ) \}$ 使得 $W _ { i } = W _ { i } ( Z _ { i } )$ 。其次，当然，我们需要定义结果的潜在结果，这些结果原则上可能对 $W _ { i }$ 和 $Z _ { i }$ 都做出响应：我们有 $\{ Y _ { i } ( w , z ) \} _ { w , z \in \{ 0 , 1 \} }$ 使得 $Y _ { i } = Y _ { i } ( W _ { i } , Z _ { i } )$ 。

有了这个记号，我们现在重新审视关于什么构成有效工具的假设：

• **排他性约束（Exclusion restriction）**。处理分配仅通过接受处理影响结果，即对所有 w 和 z，有 $Y _ { i } ( w , z ) = Y _ { i } ( w )$ 。
• **外生性（Exogeneity）**。处理分配是随机化的，这意味着 $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , W _ { i } ( 0 ) , W _ { i } ( 1 ) \} \perp Z _ { i }$ 。
• **相关性（Relevance）**。处理分配影响接受处理，这意味着 E $[ W _ { i } ( 1 ) - W _ { i } ( 0 ) ] \neq 0$ 。

最后，我们做出一个关于人们如何响应处理的假设。将每个受试者的**依从类型（compliance type）** 定义为 $C _ { i } = \{ W _ { i } ( 0 ) , W _ { i } ( 1 ) \}$ ，我们注意到这里只有4种可能的依从类型：

| | $W_{i}(1) = 0$ | $W_{i}(1) = 1$ |
|---|---|---|
| $W_{i}(0) = 0$ | 从不依从者（never taker）| 依从者（complier）|
| $W_{i}(0) = 1$ | 违抗者（defier）| 总是依从者（always taker）|

我们的最后一个假设是没有违抗者，即 $\mathbb { P } \left[ C _ { i } = \{ 1 , 0 \} \right] = 0$；这个假设通常也称为**单调性（monotonicity）**。基于这4个假设，我们得到IV估计量（10.3）的如下简单刻画。

**定理 10.1.** 考虑一个具有二元处理 $W _ { i }$ 和二元工具 $Z _ { i }$ 的抽样分布，且满足上述4个假设（外生性、相关性、单调性和排他性约束）。那么，

$$
\tau_ {L A T E} = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0)   |   C _ {i} = \text { complier } \right]. \tag {10.4}
$$

**证明.** 对于二元处理和二元工具，IV估计量（10.3）可以写为：

$$
\tau_ {L A T E} = \frac {\mathbb {E} \left[ Y _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = 0 \right]}{\mathbb {E} \left[ W _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ W _ {i} \mid Z _ {i} = 0 \right]},
$$

由于相关性假设，这个比率是定义良好的。此外，

$$
\begin{array}{l} \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = 0 \right] \\ = \mathbb {E} \left[ Y _ {i} \left(W _ {i} (1)\right) \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ Y _ {i} \left(W _ {i} (0)\right) \mid Z _ {i} = 0 \right] \quad (\text { 排他性 }) \\ = \mathbb {E} \left[ Y _ {i} (W _ {i} (1)) - Y _ {i} (W _ {i} (0)) \right] \quad (\text { 外生性 }) \\ = \mathbb {E} \left[ 1 \left(\left\{C _ {i} = \text { complier } \right\}\right) \left(Y _ {i} (1) - Y _ {i} (0)\right) \right], \quad \text {(单调性)} \\ \end{array}
$$

并且类似地：

$$
\mathbb {E} \left[ W _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ W _ {i} \mid Z _ {i} = 0 \right] = \mathbb {P} \left[ \{C _ {i} = \text {complier} \} \right].
$$

然后通过贝叶斯法则得到结果（10.4）。

![image_07](images/image_07.png)

尽管这是一个非常简单的结果，但它已经给了我们一些鼓励，即IV方法可以在非参数设定中被解释：当常数处理效应模型（9.11）不成立时，**平均处理效应（Average Treatment Effect, ATE）** $\tau _ { A T E } ~ = ~ \mathbb { E } \left[ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \right]$ 在没有更多数据的情况下显然无法被识别，因为我们没有关于接受处理的从不依从者等的观测值。然而，在合理的假设下，IV方法使我们能够估计这里可以识别的最有意义的量，即那些依从实验者分配的处理的人的平均处理效应。

**例1（续）。** 在第1章介绍的 Finkelstein 等人 [2012] 关于俄勒冈州医疗补助彩票的例子中，大约90,000名彩票参与者中有35,000人被允许申请医疗补助。然而，在35,000名彩票中奖者中，实际上只有大约30%的人注册了医疗补助：有些人没有完成申请，有些人一开始就不符合参加彩票的要求（例如，他们的收入过高）。因此，通过均值差异估计量测量的平均处理效应并不能直接量化这里医疗补助注册的收益。但是，由于这里合理地没有违抗者，我们可以将原始均值差异除以0.3，得到一个局部平均处理效应，即对那些如果中彩票就会实际注册医疗补助的人的**平均收益（average benefit）** 的估计。

**多个工具（Multiple instruments）** 在某些应用中，我们可能可以获得来自多个随机试验的数据，这些数据可以通过不依从性分析来研究处理效应。考虑一个营销应用，其中一家公司想研究加入忠诚度计划（Wi）对长期客户收入（Yi）的影响，并且可以获得多个随机试验的数据，这些试验的处理 $\left( Z _ { i } \right)$ 有效地推动客户加入忠诚度计划，因此可以用作工具。例如，一个随机试验可能提供加入忠诚度计划的折扣 $( Z _ { i } = 1$ （客户收到折扣）），而另一个可能展示广告 $( Z _ { i } = 1$ （客户看到了该计划的广告））。

如果我们只关注其中一个工具，那么上面开发的方法可以直接应用。然而，人们也可能倾向于以某种方式合并这些工具。在上一章中，我们看到，在线性处理效应模型下，多个工具可以合并为一个**最优工具（optimal instrument）**，并且最优工具对应于所有工具中最佳预测处理的汇总（定理9.2）。

然而，在没有线性处理效应模型的情况下，我们提醒说，没有这样的结果可用。不同的工具可能诱发不同的依从模式，因此不同工具识别的LATE可能不同；而使用定理9.2中的构造产生的合并工具可能诱发另一种依从模式。例如，在我们的营销例子中，对折扣响应的客户的ATE可能与对广告响应的客户的ATE不同。

因此，当不假设线性处理效应（9.11）时，如果有多个工具可供选择，研究者可能更倾向于简单地使用其LATE最接近某个政策相关效应的工具。也可以使用不同的工具运行单独的IV分析，并利用所得估计结果之间的差异来论证不同依从群体之间处理效应的异质性。

## 10.2 潜在选择模型（Latent choice models）

**工具变量回归（Instrumental variables regression）** 也被用于许多超越了上述二元处理-二元工具设定的应用中。在经济学中，长期以来一直对以下模型感兴趣：行为主体做出选择（例如，接受工作、上大学、创办公司）的方式由**潜在的（latent）** 且通常**未观测到的属性（unobserved attributes）**（例如，技能、动机、风险承受能力）决定，而这些潜在属性也影响感兴趣的经济结果变量（例如，终身收入）[Heckman, 1979, Roy, 1951]。

如果没有进一步的数据或假设，由于固有的**内生性（endogeneity）**（即处理选择对潜在属性的依赖），通常不可能衡量此类选择的因果效应。然而，工具变量方法可以在以下设定中提供一条前进的道路：我们能够获得关于外生冲击的数据，这些冲击可以被论证以准随机的方式推动进入处理的选择。我们将在此研究IV回归在若干此类选择模型中的行为，再次不做出常数处理效应假设（9.11），而是允许处理效应依赖于未观测到的潜在属性。

**供给与需求（Supply and demand）** 在许多设定中，了解**需求的价格弹性（price elasticity of demand）**，即需求如何对价格变化做出响应，是相当重要的。在典型的市场中，价格不是外生的——相反，它们产生于供给和需求的相互作用——因此估计弹性需要一个工具。这是一个潜在选择模型的例子，因为供给和需求都是由个体选择决定的，而这些选择受到市场价格以及未观测因素（例如，支付意愿或生产成本）的影响。

可以通过潜在结果将供给和需求的关系形式化如下。对于每个市场 $i = 1 , . . . , n .$ ，存在一条**供给曲线（supply curve）** $S _ { i } ( p , z )$ 和一条**需求曲线（demand curve）** $Q _ { i } ( p , z )$ ，对应于在给定价格 $p \in \mathbb R$ 和可能影响市场的某个工具 $z \in \{ 0 , 1 \}$ 下会出现的供给（和分别的需求）（该工具可以例如捕捉使生产变得更困难从而减少供给的供应链事件的存在）。为简单起见，我们可以假设 $S _ { i } ( \cdot , z )$ 是连续且递增的，而 $Q _ { i } ( \cdot , z )$ 是连续且递减的。

**例9（续）。** 在 Angrist, Graddy, 和 Imbens [2000] 的设定中，人们可以论证，经过更仔细的审视，图9.3中给出的DAG并未提供供给、需求、价格和天气之间相互作用的完整结构解释；而上述市场均衡模型（以天气作为工具）提供了更好的拟合。下面的讨论将展示，在基于这个均衡模型构建因果效应的框架下，我们如何仍然能够理解基本的IV估计量 $\hat { \tau } _ { I V }$ 。

在这个设定下，假设首先工具 $Z _ { i }$ 实现；然后价格 $P _ { i }$ 通过匹配供给和需求而产生，使得 $P _ { i }$ 是市场均衡条件 $^ { 5 8 } \ S _ { i } ( P _ { i } , Z _ { i } ) = Q _ { i } ( P _ { i } , Z _ { i } )$ 的唯一解。研究者观察到工具 $Z _ { i }$ 、市场出清价格 $P _ { i }$（"处理"）和实现的需求 $Q _ { i } = Q _ { i } ( P _ { i } , Z _ { i } )$（"结果"）。我们说 $Z _ { i }$ 是衡量价格对需求影响的有效工具，如果以下条件成立：

• **排他性约束**。工具仅通过供给影响需求，不能对其产生直接影响：对所有 p 和 z，有 $Q _ { i } ( p , z ) = Q _ { i } ( p )$ 。
• **外生性**。工具如同随机一样好，$\{ Q _ { i } ( p ) , S _ { i } ( p , z ) \}$ ⊥⊥ $Z _ { i }$ 。
• **相关性**。工具影响价格，Cov $[ P _ { i } , Z _ { i } ] \neq 0$ 。
• **单调性**。工具从不增加供给，即几乎必然地有 $S _ { i } ( P _ { i } , 1 ) \leq$ $S _ { i } ( P _ { i } , 0 )$ 。

在这个设定下，我们试图通过（10.3）估计需求弹性。

现在，尽管这看起来可能是一个复杂的设定，但事实证明，使用 $Z _ { i }$ 作为工具来衡量 $P _ { i }$ 对 $Q _ { i }$ 影响的IV估计量是表现良好的——并且可以被刻画为 $Q _ { i } ( p )$ 导数的加权平均。

**定理 10.2.** 在上述供给-需求模型中，进一步假设 $Q _ { i } ( p )$ 是可微的，并记 $Q _ { i } ^ { \prime } ( p )$ 为其导数。那么，

$$
\tau_ {L A T E} = \frac {\int \mathbb {E} \left[ Q _ {i} ^ {\prime} (p) \mid P _ {i} (0) \leq p \leq P _ {i} (1) \right] \mathbb {P} \left[ P _ {i} (0) \leq p \leq P _ {i} (1) \right] d p}{\int \mathbb {P} \left[ P _ {i} (0) \leq p \leq P _ {i} (1) \right] d p}, \tag {10.5}
$$

**证明.** 由于 $Z _ { i }$ 是二元的，我们可以写：

$$
\tau_ {L A T E} = \frac {\mathbb {E} \left[ Q _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ Q _ {i} \mid Z _ {i} = 0 \right]}{\mathbb {E} \left[ P _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ P _ {i} \mid Z _ {i} = 0 \right]}.
$$

现在，在此处所做的假设下，即工具抑制供给并且供给和需求曲线分别是单调递增和单调递减，工具必须对价格产生单调递增的影响： $P _ { i } ( 1 ) \ge P _ { i } ( 0 )$ 。那么，

$$
\mathbb {E} \left[ Q _ {i} \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ Q _ {i} \mid Z _ {i} = 0 \right]
$$

$$
= \mathbb {E} \left[ Q _ {i} (P _ {i} (1)) \mid Z _ {i} = 1 \right] - \mathbb {E} \left[ Q _ {i} (P _ {i} (0)) \mid Z _ {i} = 0 \right] \quad (\text { 排他性 })
$$

$$
= \mathbb {E} \left[ Q _ {i} (P _ {i} (1)) - Q _ {i} (P _ {i} (0)) \right] \quad (\text { 外生性 })
$$

$$
= \mathbb {E} \left[ \int_ {P _ {i} (0)} ^ {P _ {i} (1)} Q _ {i} ^ {\prime} (p) d p \right] \quad (\text { 单调性 })
$$

$$
= \int \mathbb {E} \left[ Q _ {i} ^ {\prime} (p) \mid P _ {i} (0) \leq p \leq P _ {i} (1) \right] \mathbb {P} \left[ P _ {i} (0) \leq p \leq P _ {i} (1) \right] d p, \quad (\text { Fubini })
$$

并且（10.5）中的分母可以通过类似方法刻画，从而得到（10.5）。□

上述结果并不像定理10.1中得到的结果那样易于解释，在定理10.1中，LATE被精确地发现等于依从者的平均处理效应。然而，如下面的评论所示，刻画（10.5）仍然有助于理解IV方法在涉及供给-需求均衡形成的应用中的实际行为。

**注 10.1.** 在定理10.2的设定下，如果个体需求函数是价格的线性函数， $Q _ { i } ^ { \prime } ( p ) = \alpha _ { i } + \beta _ { i } p .$ ，那么

$$
\tau_ {L A T E} = \mathbb {E} \left[ \beta_ {i} \left(P _ {i} (1) - P _ {i} (0)\right) \right] / \mathbb {E} \left[ P _ {i} (1) - P _ {i} (0) \right], \tag {10.6}
$$

即LATE等于价格对工具响应程度加权的平均价格参数。此外，如果我们有近似线性性，那么定理10.2意味着（10.6）也仍然近似成立——并且可以用于定量评估偏离线性性的影响。

**注 10.2.** 在定理10.2的设定下，如果个体需求函数 $Q _ { i } ( p )$ 是光滑的，并且工具对价格的影响很小，即对于某个稳定价格 $p _ { 0 }$ ，有 $P _ { i } ( 0 ) , P _ { i } ( 1 ) \ \approx \ p _ { 0 }$ ，那么 $\tau _ { L A T E } \approx \mathbb { E } \left[ Q _ { i } ^ { \prime } ( p _ { 0 } ) ( P _ { i } ( 1 ) - P _ { i } ( 0 ) ) \right] / \mathbb { E } \left[ P _ { i } ( 1 ) - P _ { i } ( 0 ) \right]$ 。

**阈值跨越模型（Threshold crossing models）** 另一类广泛使用的选择模型出现在当行为主体在采取某项行动 $W _ { i }$（例如，上大学）时，如果其（未观测到的）**效用（utility）** $U _ { i }$ 超过了采取行动的成本，就会采取该行动。在这样的设定中，如果我们有一个外生工具 $Z _ { i }$ 可以改变采取行动的成本（例如，在上大学的例子中，一个随机分配的学费补贴），那么我们可以再次寻求使用这个工具来估计 $W _ { i }$ 对下游结果 $Y _ { i }$（例如，终身收入）的影响。

对此设定进行建模的标准方法是通过一个**阈值跨越模型（threshold crossing model）**：我们假设每个受试者有一个潜在的、内生的变量 $U _ { i }$ ，使得

$$
W _ {i} = 1 \left(\{U _ {i} \geq c (Z _ {i}) \}\right), \tag {10.7}
$$

其中 $c ( z )$ 给出了作为工具 z 的函数的处理成本，这里我们允许其取连续值。这种边界跨越结构在我们通常假设的类似条件下产生一个有效的工具：

• **排他性约束**。存在潜在结果 $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \}$ 使得 $Y _ { i } = Y _ { i } ( W _ { i } )$
• **外生性**。处理分配是随机化的，这意味着 $\{ Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , U _ { i } \} \perp Z _ { i }$ 。

• **相关性**。阈值函数 $c ( Z _ { i } )$ 具有非平凡的变化，即 $\mathbb { P } \left[ U _ { i } \ge c ( Z _ { i } ) \vert Z _ { i } = z \right]$ 在 $z$ 上非常数。
• **单调性**。阈值函数 $c ( z )$ 在 $z$ 上非递增。

最后，定义**边际处理效应（marginal treatment effect）**

$$
\tau (u) = \mathbb {E} \left[ Y _ {i} (1) - Y _ {i} (0) \mid U _ {i} = u \right]. \tag {10.8}
$$

我们的目标是证明 **IV** 方法恢复了边际处理效应 $\tau ( u )$ 的加权平均值。为方便起见，我们假设工具变量服从高斯分布，即 $Z _ { i } \sim { \mathcal { N } } \left( 0 , 1 \right)$，因为这允许我们应用**斯坦因引理（Stein's lemma）**；在不假设高斯性的情况下更一般的结果参见 Heckman 和 Vytlacil [2005]。

**定理 10.3**。给定上述**阈值穿越模型（threshold crossing model）**，假设 $U _ { i }$ 的分布具有密度 $f ( u )$ 和累积分布函数 $1 - G ( u )$，$\tau ( u )$ 一致有界，并且 $Z _ { i }$ 服从高斯分布 $Z _ { i } \sim \mathcal { N } ( 0 , 1 )$。进一步假设阈值函数 $c ( \cdot )$ 是**右连左极函数（cadlag）**，即对所有 $z$ 有 $c ( z ) = \operatorname* { l i m } _ { a \downarrow z } c ( a )$，并记 $c _ { - } ( z ) = \operatorname* { l i m } _ { a \uparrow z } c ( a )$。那么，存在一个非负的勒贝格可测函数 $c ^ { \prime } ( z )$，使得 $c ( z ) = c _ { 0 } + \textstyle \int _ { - \infty } ^ { z } c ^ { \prime } ( a ) d a$，并且

$$
\tau_ {L A T E} = \frac {\sum_ {z \in \mathcal {S}} \left(\int_ {c (z)} ^ {c _ {-} (z)} \tau (u) f (u) d u\right) \varphi (z) - \int_ {\mathbb {R} \backslash \mathcal {S}} \tau (c (z)) f (c (z)) c ^ {\prime} (z) \varphi (z) d z}{\sum_ {z \in \mathcal {S}} \left(G (c (z)) - G (c _ {-} (z))\right) \varphi (z) - \int_ {\mathbb {R} \backslash \mathcal {S}} f (c (z)) c ^ {\prime} (z) \varphi (z) d z},
$$

其中 $\mathcal { S } \subset \mathbb { R }$ 是 $c ( \cdot )$ 的不连续点集，$\varphi ( \cdot )$ 是标准高斯密度函数。

**证明**。$c ( z )$ 具有分布导数这一事实直接源于其单调性（从而具有有界变差）。现在，为了建立所需结果，关键在于刻画 $\operatorname{Cov} [ Y _ { i } , Z _ { i } ]$；(10.3) 分母的表达式可通过同样的论证得到。首先，注意到

$$
\begin{array}{l} \operatorname{Cov} \left[ Y _ {i}, Z _ {i} \right] = \operatorname{Cov} \left[ Y _ {i} (0) + (Y _ {i} (1) - Y _ {i} (0)) W _ {i}, Z _ {i} \right] \\ = \operatorname{Cov} \left[ \left(Y _ {i} (1) - Y _ {i} (0)\right) W _ {i}, Z _ {i} \right] \\ = \operatorname{Cov} \left[ \left(Y _ {i} (1) - Y _ {i} (0)\right) 1 \left(\left\{U _ {i} \geq c \left(Z _ {i}\right) \right\}\right), Z _ {i} \right] \\ = \operatorname{Cov} \left[ \tau (U _ {i}) 1 \left(\left\{U _ {i} \geq c (Z _ {i}) \right\}\right), Z _ {i} \right], \\ \end{array}
$$

其中第一个等式来自**排他性约束（exclusion restriction）**，而第二个和第四个来自**外生性（exogeneity）**。

现在，记 $H ( z ) = \mathbb { E } \left[ \tau ( U _ { i } ) 1 \left( \{ U _ { i } \geq c ( z ) \} \right) \right]$。由于 $Z _ { i }$ 是标准高斯变量，Stein [1981] 的引理 1 表明

$$
\operatorname{Cov} \left[ H (Z _ {i}), Z _ {i} \right] = \mathbb {E} \left[ H ^ {\prime} (Z _ {i}) \right], \tag {10.9}
$$

其中 $H ^ { \prime } ( Z _ { i } )$ 表示 $H ( \cdot )$ 的分布导数。此外，根据链式法则 [Ambrosio and Dal Maso, 1990, Corollary 3.1]，

$$
H ^ {\prime} (z) = \left\{ \begin{array}{l l} \left(\int_ {c (z)} ^ {c _ {-} (z)} \tau (u) f (u) d u\right) \delta_ {z} & \text { for } z \in \mathcal {S}, \\ - \tau (c (z)) f (c (z)) c ^ {\prime} (z) & \text { else }, \end{array} \right. \tag {10.10}
$$

其中 $\delta _ { z }$ 是 $z$ 处的**狄拉克 $\delta$ 函数（Dirac delta-function）**。由此可得所需结果。

![image_08](images/image_08.png)

**注 10.3**。在定理 10.3 的设定下，假设阈值函数 $c ( z )$ 是常数且只有一个跳跃，即 $c ( z ) = c _ { 0 } - \delta _ { 1 } 1 \left( \left\{ z \geq z _ { 1 } \right\} \right)$。那么**依从类型（compliance types）**被归为三个主要分层：**从不接受者（Never-takers）** 满足 $U _ { i } < c _ { 0 } - \delta _ { 1 }$，**依从者（compliers）** 满足 $c _ { 0 } - \delta _ { 1 } \leq U _ { i } < c _ { 0 }$，以及**始终接受者（always takers）** 满足 $U _ { i } \geq c _ { 0 }$。此外，与之前一样，我们的估计量对应于依从者的平均处理效应，如定理 10.1 所示：

$$
\tau_ {L A T E} = \mathbb {E} \left[ \tau (U _ {i}) \mid c _ {0} - \delta_ {1} \leq U _ {i} <   c _ {0} \right] \tag {10.11}
$$

**注 10.4**。在前一个例子的基础上，现在假设有 $K$ 个跳跃，其截断函数由 $\begin{array} { r } { c ( z ) = c _ { 0 } - \sum _ { k = 1 } ^ { K } \delta _ { k } 1 \left( \left\{ z \geq z _ { k } \right\} \right) } \end{array}$ 给出。那么，

$$
\tau_ {L A T E} = \sum_ {k = 1} ^ {K} \mathbb {E} \left[ \tau (U _ {i}) \mid c (z _ {k}) \leq U _ {i} <   c _ {-} (z _ {k}) \right] \gamma_ {k} / \sum_ {k = 1} ^ {K} \gamma_ {k}, \tag {10.12}
$$

$$
\gamma_ {k} = \big (G (c (z _ {k})) - G (c _ {-} (z _ {k})) \big) \varphi (z _ {k}).
$$

换句话说，我们恢复了由 $c ( \cdot )$ 的跳跃定义的依从分层上的平均处理效应的**凸组合（convex combination）**。这些权重取决于分层的大小以及工具变量在 $z _ { k }$ 处的密度函数。

**注 10.5**。在定理 10.3 的设定下，假设 $c ( z )$ 没有跳跃。那么，**局部平均处理效应（LATE）** 对应于 $\tau ( c ( Z _ { i } ) )$ 的加权平均值：

$$
\tau_ {L A T E} = \int_ {\mathbb {R}} \tau (c (z)) f (c (z)) c ^ {\prime} (z) \varphi (z) d z / \int_ {\mathbb {R}} f (c (z)) c ^ {\prime} (z) \varphi (z) d z. \tag {10.13}
$$

这些权重可以通过 $f ( c ( z ) ) c ^ { \prime } ( z ) = d / d z \ \mathbb { P } \left[ U _ { i } \geq c ( z ) \right]$ 来解释，即它们与工具变量的局部强度成正比。

**估计边际处理效应** 在本章中，我们一直假定目标是估计量 (10.3)，然后试图在不同设定下对其进行解释。然而，当我们使用连续工具变量时，可以针对更广泛的估计量。一个关键结果是，在上述阈值穿越模型中，边际处理效应 (10.8) 在 $c ( z )$ 的连续点处通过一个简单的“局部 IV”构造被识别。

**定理 10.4**。在定理 10.3 的设定下，假设 $c ( z )$ 在 $z$ 处连续可微，且 $c ^ { \prime } ( z ) < 0$，并且 $U _ { i }$ 的密度满足 $f ( c ( z ) ) > 0$。那么，(10.8) 中的边际处理效应 $\tau ( u )$ 被识别为

$$
\tau (c (z)) = \frac {\frac {d}{d z} \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = z \right]}{\frac {d}{d z} \mathbb {P} \left[ W _ {i} = 1 \mid Z _ {i} = z \right]}. \tag {10.14}
$$

**证明**。在我们的阈值穿越模型下，

$$
\begin{array}{l} \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = z \right] = \mathbb {E} \left[ Y _ {i} (0) + 1 \left(\{U _ {i} \geq c (Z _ {i}) \}\right) (Y _ {i} (1) - Y _ {i} (0)) \mid Z _ {i} = z \right] \\ = \mathbb {E} \left[ Y _ {i} (0) + 1 \left(\{U _ {i} \geq c (z) \}\right) \left(Y _ {i} (1) - Y _ {i} (0)\right) \right] \\ = \mathbb {E} \left[ Y _ {i} (0) \right] + \int_ {c (z)} ^ {1} \tau (u) f (u) d u, \\ \end{array}
$$

其中第一个等式源于 (10.7) 和排他性约束，第二个源于外生性，第三个是**富比尼定理（Fubini's theorem）**的应用。接下来，鉴于 $c ( z )$ 在 $z$ 处连续可微，我们可以使用链式法则得到

$$
\frac {d}{d z} \mathbb {E} \left[ Y _ {i} \mid Z _ {i} = z \right] = - \tau (c (z)) f (c (z)) c ^ {\prime} (z). \tag {10.15}
$$

最后，对分母应用同样的计算得到 (10.14)。

一旦我们获得了边际处理效应，就可以用它来构建 $\mathbb{E} [ \gamma ( u ) \tau ( u ) ]$ 加权平均值的估计量，前提是权重 $\gamma ( u )$ 仅在 $c ( z )$ 连续的点 $u = c ( z )$ 处取正值。Heckman 和 Vytlacil [2005] 考虑了多种此类估计量。

**例 10**。Carneiro, Heckman, 和 Vytlacil [2011] 使用局部 IV 方法估计了大学就读的回报。作者使用了**全国青年纵向调查（National Longitudinal Survey of Youth）** 1979 年队列的数据（由 1957 年至 1964 年间出生的人组成），将结果变量 $Y _ { i }$ 设为 1991 年的对数收入，并将处理变量 $W _ { i }$ 设为截至 1991 年是否曾就读大学。他们通过工具变量 $Z _ { i }$ 识别边际处理效应，这些工具变量改变了就读大学的意愿，包括附近是否有大学、附近大学的学费以及人们年满 17 岁时的当地就业状况。他们的主要发现是，使用我们的符号，$\tau ( u )$ 随 $u$ 递增，并且那些在面对不利推动时更有可能上大学的人（即抽象地说，对大学有更高支付意愿的人）实际上从大学受益更多。因此，他们的结果表明，人们在模型 (10.7) 下的选择至少可以在方向性上通过私人对未来大学就读收入收益的预测得到合理化。

## 10.3 文献注释（Bibliographic notes）

从**局部平均处理效应（local average treatment effect）** 的角度解释工具变量分析结果的思想可追溯到 Imbens 和 Angrist [1994]。我们对非依从情况下临床试验分析的阐述遵循 Angrist, Imbens, 和 Rubin [1996]。我们推荐 Imbens [2014] 作为回顾。

**潜在选择模型（Latent choice models）** 在经济学中有着悠久的传统，该模型假设人们在其（私人）价值超过成本时做出选择。在一个早期例子中，Roy [1951] 考虑了一个模型，工人们通过考虑他们在不同工作中的技能来选择职业，然后选择能够最大化其工资的职业——并用它来论证，如果工人技能在不同职业间相关，但某些职业的生产力对技能的反应比其他职业更敏感，那么我们应该预期技能回报更高的职业拥有更高的平均工资。长期以来人们认识到，此类模型无法通过标准线性回归拟合；然而，在早期文献中，此类模型通常通过临时性的计量经济学策略而非 IV 方法来处理。例如，Heckman [1979] 考虑了一个参数化的潜在选择模型，并通过潜在变量 $U _ { i }$ 和潜在结果（而非使用辅助的外生变异来源）的联合正态性实现了识别。

最近，Heckman 和 Vytlacil [2005] 提倡将潜在选择模型作为理解工具变量方法的自然框架，并研究了针对除 LATE 之外更广泛估计量的方法，这些估计量可能对制定政策更有帮助。通过局部 IV 构造对边际处理效应的识别结果 (10.14) 归功于 Heckman 和 Vytlacil [1999]。Kennedy, Lorch, 和 Small [2019] 研究了边际处理效应函数的半参数有效估计。通过以不可观测变量为条件定义子总体来估计平均处理效应的目标，也出现在生物统计学中发展的**主分层（principal stratification）** 文献中 [Frangakis and Rubin, 2002]。我们对供需均衡下局部平均处理效应的阐述改编自 Angrist, Graddy, 和 Imbens [2000]。
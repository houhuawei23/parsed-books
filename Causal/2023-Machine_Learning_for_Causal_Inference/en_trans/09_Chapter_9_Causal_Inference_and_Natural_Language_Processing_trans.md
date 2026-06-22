# 第9章 因果推断与自然语言处理（Chapter 9 Causal Inference and Natural Language Processing）

![image_58](images/image_58.png)

陈文清（Wenqing Chen）和初志轩（Zhixuan Chu）

## 9.1 基于文本数据的因果推断（Causal Inference with Textual Data）

**随机对照试验（Randomized Controlled Trials, RCTs）** 常被用于科学研究中，以估计变量之间的因果效应。然而，RCTs 受到高成本和伦理问题的限制 [52]。当处理像文本这样的高维和非结构化数据时，由于文本数据中集中变量的纠缠，RCTs 变得更加具有挑战性。相比之下，从观测数据中估计因果效应是一种更具成本效益且在伦理上更安全的方法，近年来在研究中受到越来越多的关注 [31, 42, 58]。在本节中，我们重点关注观测数据的使用，并展示如何利用文本数据进行因果推断。例如，如何准备产品介绍以吸引顾客 [61]，以及贷款申请人如何撰写陈述以影响资金获取 [75]。

鲁宾（Rubin）和珀尔（Pearl）的因果理论是统计学和机器学习中用于因果推断的两种重要方法，两者都可用于文本数据的因果推断，但基于图模型的珀尔方法在此背景下通常更为常用。文本数据通常包含变量之间的复杂依赖关系，而珀尔的图模型为建模这些依赖关系并推断它们之间的因果关系提供了一个灵活且强大的框架 [18, 37]。根据研究兴趣的不同，近期的工作可以分为两类：

1. 当感兴趣的变量是语言属性时，研究问题是找到针对特定目标有效呈现文本的方法。例如，政治候选人如何有效呈现其个人背景以吸引选民 [22]？企业主如何撰写产品描述以提升电子商务平台上的销售业绩 [60, 63]？
2. 当感兴趣的变量是非语言属性但与文本数据相关时，研究问题是准确估计因果效应。例如，性别是否会影响作者在在线论坛上帖子的受欢迎程度 [18]？审查制度对未来发帖率的影响程度如何，其中文本内容是一个混杂变量 [65]？此外，文本数据可以作为传统因果推断问题中的代理变量。例如，在估计吸烟对预期寿命的因果效应时，职业可能是一个潜在的混杂变量，但可能未被记录。在这种情况下，研究人员可能试图从个人的历史社交媒体帖子中推断其职业 [37]。

为了在上述情况下估计因果效应，研究人员必须克服两个挑战。第一个是因果推断领域的一个常见问题，即想象反事实世界。第二个挑战源于文本的高维性质，这要求研究人员找到一种能够保留相关因果关系的低维表示 [16, 75]。然而，获得这样的表示并非易事，因为文本中的语言变量可能与其他语言或非语言变量纠缠在一起。例如，在估计作者性别对其帖子受欢迎程度的因果效应时，帖子的主题可能作为一个混杂变量，因为某些主题可能吸引更多男性而非女性，并且总体上更受欢迎，而写作风格可能作为一个中介变量 [18]。因此，使用文本数据进行因果推断需要做出假设，并且任何文本表示都应考虑变量之间的假设关系。将混杂变量误认为是中介变量，或者反之，都可能导致因果效应的估计出现偏差。

**自然语言处理（Natural Language Processing, NLP）** 的进展，例如使用语言模型、主题模型和其他上下文嵌入模型，为将高维文本数据转换为相对低维的数据同时尊重先验图假设提供了有前景的方法。最近关于使用文本进行因果效应估计的 NLP 工作可以根据文本在因果图中作用的不同假设分为四类：

1. **文本被视为处理变量（treatment）**，目标是估计特定语言属性对结果的因果效应 [62]。例如，竞选者呈现个人经历的方式可能影响他们获得的票数 [22]，或者公司如何撰写产品描述以吸引顾客 [60, 63]。然而，该领域存在两个主要挑战。首先，不同的文本属性通常在文本中相互交织，当假设有 $N$ 个属性时，研究人员通常一次只估计一个属性的因果效应，将剩余的 $(N-1)$ 个属性作为潜在的混杂变量或中介变量。虽然一些研究假设这些 $(N-1)$ 个变量都是混杂变量 [62]，但其他研究指出，这种假设对于某些类型的文本（例如同时“礼貌”和“粗俗”的文本）是不现实的 [18]。其次，存在无法在文本中反映的未观测混杂变量，导致因果效应的估计出现偏差。例如，具有不同政治立场的读者可能会选择不同的文本阅读，而个人的政治立场是未观测的，但会影响因果效应的估计 [18]。
2. **文本被视为混杂变量（confounder）**，文本中的某些属性作为影响观测处理变量和结果变量的混杂变量。例如，在一项研究论文第一作者是否为女性会导致更高影响力（例如更多引用）的研究中，潜在的混杂变量可能包括论文的主题和研究领域 [65, 76]。可以构建模型从文本中预测处理变量和结果变量 [81]。然而，假设文本属性是混杂变量可能存在风险，因为如果其中一些属性是中介变量，混杂变量假设可能导致在反事实推断过程中产生不合理的反事实样本，从而违反积极性假设 [18]。
3. **文本被视为中介变量（mediator）**，假设某些文本属性作为中介变量。例如，在在线论坛的背景下，研究人员调查了评论发布者的性别对评论受欢迎程度得分的影响，男性和女性可能采用不同的语气和写作风格 [81]。这类问题涉及以更细粒度估计间接和直接因果效应。主要挑战包括对文本中混杂变量和中介变量的假设，以及基于处理变量构建条件文本表示，并开发一个从文本中预测中介变量的模型 [36]。然而，构建此类模型的最佳方法仍存在争议。
4. **文本被视为结果变量（outcome）**，目标是估计感兴趣的处理变量对生成的文本的特定语言属性的因果效应。此类研究的例子包括探讨“女性法官或非白人法官”对法律文件中语言表达的影响 [24]，或者学生的教育水平如何影响其论文的可读性 [15]。该研究领域的主要挑战在于文本是非结构化数据，使得设计用于评估这些语言属性的评估模型变得困难。通常需要 NLP 模型将文本转换为结构化属性，但这些模型也可能引入某些偏差。

总之，本节探讨了观测性文本数据如何用于因果推断研究。虽然传统的因果推断方法主要关注结构化数据，但语言在社会科学中日益增长的相关性促使研究人员探索使用文本数据进行因果推断。根据文本在因果图中扮演不同角色的假设，近期的工作可以分为不同的类别。每个类别都提出了独特的挑战，例如混杂变量的存在或难以恰当表示文本数据。尽管如此，NLP 模型为这些挑战提供了潜在的解决方案。

## 9.2 NLP 中的虚假相关（Spurious Correlations in NLP）

除了使用 NLP 模型估计因果效应之外，由于这些模型依赖于学习统计相关性进行预测，而不考虑潜在的因果关系，因此对其可信度的担忧也随之增加。这种相关性被定义为**虚假相关（spurious correlations）**，指的是非因果但相关的联系 [74]。随着深度神经模型在 NLP 领域取得显著进展 [96]，依赖于训练数据和测试数据分布相同的假设是有风险的。最近，**预训练语言模型（Pretrained Language Models, PLMs）** [14, 27, 28, 44, 80] 在某些文本理解任务和数据集上甚至达到了超越人类的表现，[1] 但它们的鲁棒性仍然是一个主要问题。

例如，情感分析是一项 NLP 任务，模型的目标是将给定文本的情感分类为“正面”、“负面”或“中性”。然而，观察到在 IMDB 电影评论数据集上训练的深度学习模型依赖于虚假相关，导致不可靠的决策。具体来说，包含单词“Spielberg”的电影评论通常被标记为“正面”，这导致了高相关性 [88]。然而，这种相关性并不反映单词“Spielberg”的存在与评论正面情感之间的因果关系。如果将所有其他单词保持不变，将“Spielberg”替换为另一位导演的名字，评论的情感不会改变。这种基于虚假相关的决策被称为“因错误原因而正确”（right for the wrong reasons）[47] 或“推理捷径”（reasoning shortcuts）[8, 13, 54]，这导致模型在数据分布发生变化时鲁棒性较低。

研究表明，即使是最先进的 PLMs 也无法避免虚假相关，特别是当特定的少数文本模式在训练数据中代表性不足时 [80, 87]。例如，在释义识别任务中，在 QQP 数据集 [33] 上微调的 PLMs 倾向于严重依赖“词汇重叠”（lexical overlap）这一虚假相关特征进行决策，而对于释义来说，这并不是一个可靠的线索，因为人类可以使用不同的词语来表达相同的意思 [80]。类似地，在 ARCT 数据集 [26] 上微调时，BERT [14] 变得过度依赖特定的关键词“not”进行推理。对测试集进行更改，从而移除虚假相关特征，可能导致性能显著下降，模型的性能变得与随机猜测相当 [53]。

这些研究表明，尽管深度神经模型在 NLP 领域取得了显著进展，但虚假相关问题仍然是一个挑战。因此，当数据分布发生变化时，模型的性能可能会急剧下降，限制了其在现实场景中的适用性。在 NLP 中，这个问题可能影响**自然语言理解（Natural Language Understanding, NLU）** 和**自然语言生成（Natural Language Generation, NLG）** 任务。我们系统地回顾了近期报告此问题的研究工作。

1. 在 NLU 中，模型可能依赖“非语义”（non-semantic）或“浅层语义”（shallow semantic）文本模式进行预测，例如句法属性或特定关键词。这些特征可以在不捕获输入文本深层语义的情况下用于预测，从而导致推理捷径 [8, 13, 47, 54]。例如，诸如句法属性之类的“非语义文本模式”已被用于做出决策 [47]。在 MNLI 数据集 [85] 的自然语言推理任务中，观察到输入假设文本 $H$ 和前提文本 $P$ 之间的“词汇重叠”与标签“蕴含”（Entailment）之间存在强相关性。词汇重叠指的是 $H$ 在 $P$ 中的连续子序列、句法子树和其他句法特征 [47, 95]。类似地，在用于释义识别任务的 Quora 问题对（QQP）数据集 [33] 中，发现模型依赖于词汇重叠进行预测 [80]。然而，从人类的角度来看，这些特征包含有限的语义信息，可能在现实场景中不适用。因此，它们不应被用于 NLU 任务。“浅层语义文本模式”，例如特定词语或线索，也被用于做出预测 [53, 88]。例如，在 MNLI 数据集中，观察到假设文本中存在关键词“not”与标签“矛盾”（contradictory）强相关 [25]。然而，这种方法可能导致不可靠的决策，因为模型可能在没有观察前提文本的情况下做出正确预测。类似地，在情感分类任务中，观察到“Spielberg”的存在与正面情感标签之间存在相关性。然而，依赖特定关键词可能导致对包含“Spielberg”但具有负面情感的电影评论做出不准确的预测 [88]。研究还表明，当这些关键词被添加、删除或重写以构建新的数据样本时，模型的预测准确率会显著下降 [50]，表明形成了推理捷径。
2. 虚假相关现象在 NLG 任务中普遍存在，尽管很少从因果角度进行审视。NLG 任务，如机器翻译 [3]、摘要生成 [51]、对话 [83] 和图像描述 [92]，需要在输入数据和生成文本之间进行语义对齐。然而，研究人员注意到，NLG 模型经常生成无意义或语义上与输入数据不一致的文本，这种现象被称为**幻觉问题（hallucination problem）** [34]。这个问题通常归因于虚假相关的存在，这些虚假相关可能由多种因素引起，例如语义不充分的表示学习 [1, 20, 34, 40] 和语义错位，即解码器关注到编码输入数据的错误部分 [79]。最近在图像描述任务中的一个例子表明，一些模型可能由于视觉特征“长头发”与描述中的词元“女性”之间的虚假相关，而错误地将留长发的男性识别为女性 [10]。类似地，在表格到文本生成任务中，最近的研究发现了语言上相似的实体之间的虚假相关 [9]。

简而言之，在 NLU 中使用非语义或浅层语义文本模式可能导致推理捷径，因为模型依赖句法属性或特定关键词，而不是捕获输入文本的深层语义。同样，虚假相关在 NLG 中普遍存在，导致生成无意义或语义不一致的文本，即幻觉问题 [34]。

最近的研究已将虚假相关确定为 NLP 中一个持续存在的问题。这些相关性通常源于训练数据中固有的偏差。偏差的两个主要来源是**选择偏差（selection bias）** 和**标注偏差（annotation bias）**，这在文献中已被广泛探讨 [4, 29]。选择偏差源于在数据集收集过程中对具有特定特征的数据样本的有偏选择。例如，大量英语 NLP 数据集来源于历史新闻库，如《华尔街日报》和法兰克福广播电台，这些新闻库的作者可能主要是白人、中年、受过教育的中上层阶级男性 [30]。因此，在此类数据集上训练的模型可能学习到该人群特有的文本模式，而这些模式不一定能推广到其他年龄组或性别 [29]。另一方面，标注偏差是由于标注者的偏好而产生的。例如，在自然语言推理数据集 SNLI 和 MNLI [85] 中，标注者被指示生成三种不同的“假设文本”[25]。在生成标记为“矛盾”的“假设文本”时，标注者经常引入关键词“not”，这可能在标签“矛盾”和关键词“not”之间产生虚假相关。

NLP 中虚假相关的普遍存在凸显了需要更仔细地策划和标注数据集，以及开发稳健的技术来检测和减轻模型中的此类偏差 [29]。

## 9.3 面向因果的 NLP 模型（Causality-Driven Models for NLP）

针对虚假相关问题及其对深度学习模型的负面影响，许多研究人员提出了各种将因果关系注入模型的方法，旨在增强其鲁棒性和泛化能力 [9, 10, 18, 32]。这些努力在减轻虚假相关引入的偏差方面显示出有希望的结果，并有可能提高 NLP 模型在各种任务中的性能。

### 9.3.1 预备知识（Preliminaries）

我们简要介绍两种重要的因果理论，即鲁宾的**潜在结果框架（Potential Outcome Framework, POF）** [68] 和珀尔的**结构因果模型（Structural Causal Model, SCM）** [55, 57]。POF 根据不同处理或干预下结果的比较来定义因果关系，而 SCM 使用**有向无环图（Directed Acyclic Graphs, DAGs）** 来表示变量之间的因果关系。

<!-- footnote -->

- P. Sheth (-) · H. Liu 亚利桑那州立大学，坦佩，亚利桑那州，美国 电子邮箱：psheth5@asu.edu; huanliu@asu.edu

<!-- footnote end -->

虽然这两个框架最初都是为衡量变量之间的因果效应而开发的，但在本节中，我们侧重于将因果关系引入 NLP 模型的相关工作。

两种因果模型之间的一个关键区别在于变量因果图假设的作用。POF 不假设变量之间存在任何图结构，而 SCM 以 DAG 的形式表示变量之间的因果关系。在利用因果知识改进机器学习模型方面，SCM 的应用更为广泛 [70, 71]。部分原因在于机器学习的历史发展，其中使用图结构表示变量之间的关系很常见。

在本节中，我们讨论珀尔颇具影响力的“因果阶梯”（causal ladder）框架 [57] 及其在近期面向因果的 NLP 模型工作中的应用。“因果阶梯”将因果关系分为三个层次：**关联（association）**、**干预（intervention）** 和**反事实（counterfactuals）**，分别对应人类认知中的观察、行动和想象。

第一层，关联，指的是变量之间的统计相关性。许多机器学习模型在这一层次运行 [57]，学习条件概率分布 $P(Y = y \mid X = x)$。然而，正如第 9.2 节所讨论的，由于混杂变量的存在，此类模型可能推断出虚假相关。

第二层，干预，考察如果操纵 $X$ 的值，$Y$ 的值如何变化。这一层次涉及 **Do-演算（Do-Calculus）**，它计算概率 $P(Y = y \mid \mathrm{do}(X = x))$，表示如果 $X$ 的值被干预为 $x$，$Y$ 取值为 $y$ 的概率。由于 $X$ 值的变化是干预的结果，不受混杂变量 $C$ 的影响，因此在执行 $\operatorname{do}(X = x)$ 操作后，因果箭头 $C \rightarrow X$ 被移除。相应地，相应机器学习模型的优化目标函数也应调整为 $P(Y = y \mid \mathrm{do}(X = x))$ [5, 86]。

第三层，反事实，涉及想象一个平行或假设的世界。在这个世界中，考虑了在现实世界中未发生的 $(X, Y)$ 的反事实值 $(\widetilde{x}, \widetilde{y})$。例如，如果一位患者没有服用某种药物并在现实中死亡，那么就会出现一个问题：如果该患者服用了这种药物，他是否能够存活下来。然而，由于患者的死亡在现实中已经发生，反事实值无法被观测到。反事实问题可以形式化地定义为估计 $P(Y = \widetilde{y} \mid x, y, \mathrm{do}(X = \widetilde{x}))$。机器学习中的大量研究旨在训练模型来估计和回答这个反事实问题 [49]。

## 9.3.2 干预层面的去偏（Intervention-Level Debiasing）

当存在潜在的**混淆变量（confounders）**时，深度学习中的**虚假相关（Spurious correlation）**就会出现 [66]。模型可能会错误地将混淆变量视为中介变量，从而导致不正确的推理路径：$X \ \to \ C \ \to \ Y$，其中箭头 -- 表示后验路径，该路径是非因果的，并且在现实世界中不具有泛化性。

干预层面的去偏通常将模型的学习目标从 $P ( Y = y ~ \vert ~ X = x )$ 调整为 $P ( Y = y ~ \vert ~ \operatorname { d o } ( X = x ) )$，这通过 **do-演算（do-calculus）** 阻断了 $X \leftrightarrow C$ 路径。然而，这需要以涉及混淆变量 C 的因果图形式存在的先验知识。根据对混淆变量的假设，近期的工作可以归纳为以下几类：

1.  **第一类工作** 明确假设可以观测到混淆变量，并将学习目标从

    $$
    P _ {\theta} (Y = \mathbf {y} \mid X = \mathbf {x}) = \sum_ {c} P _ {\theta} (Y = \mathbf {y} \mid X = \mathbf {x}, C = \mathbf {c}) \underline {{P (C = \mathbf {c} \mid X = \mathbf {x})}} \tag {9.1}
    $$

    改变为

    $$
    P _ {\theta} (Y = \mathbf {y} \mid \mathrm{do} (X = \mathbf {x})) = \sum_ {c} P _ {\theta} (Y = \mathbf {y} \mid X = \mathbf {x}, C = \mathbf {c}) \underline {{P (C = \mathbf {c})}} \tag {9.2}
    $$

    其中 $\theta$ 表示模型参数，do-演算使得混淆变量独立于输入变量，表示为 $c \perp X$。这种干预使得后验概率 $\begin{array}{c} P ( C = \pmb { c } \end{array} | \pmb { \cal X } = \pmb { x } )$ 被干预为 $P ( C =$ c) [38, 78, 86]。此类方法已应用于文本分类 [38]、自然语言推理 [78] 和图像描述 [43, 86, 94] 等任务。公式 9.2 的实现通常假设 C 是一个分类变量，并且 $P ( C = { \pmb { c } } )$ 在训练数据中预先计算好。在近期的一些工作中 [43, 86]，$P _ { \theta } ( Y = y \mid X = x , C = c )$ 的过程也是一个分类问题，网络包含一个最终的 softmax 层，表示为：

    $$
    P _ {\theta} (Y = \mathbf {y} \mid X = \mathbf {x}, C = \mathbf {c}) = \text { Softmax } (f _ {y} (\mathbf {x}, \mathbf {c})) \tag {9.3}
    $$

    其中 $f _ { y } ( x , \pmb { c } )$ 计算所有类别的 logits。公式 9.2 变为：

    $$
    P _ {\theta} (Y = \mathbf {y} \mid \mathrm{do} (X = \mathbf {x})) = \mathbb {E} _ {\mathbf {c} \sim p (\mathbf {c})} \left[ \operatorname{Softmax} \left(f _ {y} (\mathbf {x}, \mathbf {c})\right) \right] \tag {9.4}
    $$

    而期望操作涉及昂贵的 c 采样。通常使用 **归一化加权几何平均（Normalized Weighted Geometric Mean, NWGM）** 近似 [86, 93, 94] 来降低计算成本：

    $$
    \mathbb {E} _ {\boldsymbol {c} \sim p (\boldsymbol {c})} \left[ \operatorname{Softmax} \left(f _ {y} (\boldsymbol {x}, \boldsymbol {c})\right) \right] \approx \operatorname{Softmax} \left(\mathbb {E} _ {\boldsymbol {c} \sim p (\boldsymbol {c})} \left[ f _ {y} (\boldsymbol {x}, \boldsymbol {c}) \right]\right) \tag {9.5}
    $$

    其中函数 $f _ { y } ( \cdot )$ 由参数为 $W _ { 1 }$ 和 $W _ { 2 }$ 的线性模型实现。在近期的工作 [86] 中，由于混淆变量 C 被干预为独立于 X，期望项变为：

    $$
    \mathbb {E} _ {\boldsymbol {c} \sim p (\boldsymbol {c})} \left[ f _ {y} (\boldsymbol {x}, \boldsymbol {c}) \right] = \boldsymbol {W} _ {1} \boldsymbol {x} + \boldsymbol {W} _ {2} \cdot \mathbb {E} _ {\boldsymbol {c} \sim p (\boldsymbol {c})} \left[ g _ {y} (\boldsymbol {c}) \right] \tag {9.6}
    $$

    其中 $\mathbb { E } _ { c \sim p ( c ) } \left[ g _ { y } ( \pmb { c } ) \right]$ 可以针对混淆变量的所有可能类别并行计算 [86]。值得注意的是，公式 9.3–9.6 只是实现 $P ( Y = y ~ \vert ~ \mathrm { d o } ( X = x ) )$ 的一类工作。还有其他相关工作使用**对抗学习（adversarial learning）** [39, 63] 来近似干预操作。具体来说，这项工作构建了一个判别器，利用输入变量 X 的表示 H 来预测混淆变量 C，而生成器则生成无法从中预测 C 的表示 H。当生成器和判别器达到**纳什均衡（Nash equilibrium）**时，可以认为隐藏状态 H 不包含能够预测混淆变量 C 的信息。

2.  **第二类工作** 旨在放宽对混淆变量的假设，因为在现实世界中，真正的混淆变量可能无法观测或无法测量 [9, 10, 32, 48]。例如，直接测量个人的社会经济地位可能很困难，但可以通过其邮政编码或职业获得一个**代理变量（proxy）** [45]。此外，自然语言数据是高维的，使得识别潜在混淆变量比先前假设的更为复杂。最近的研究通过假设潜在空间中存在真实的混淆变量，并且可以观测到代理混淆变量来解决这个问题 [9, 10, 32]。为了解决这个问题，使用了**条件变分自编码器（Conditional Variational Auto-Encoders, CVAEs）**，并将学习目标从原始形式：

    $$
    \begin{array}{l} \log p (\mathbf {y} \mid \mathbf {x}) \geq \mathbb {E} _ {\mathbf {z} _ {c} \sim q _ {\phi} (\mathbf {z} _ {c} | \mathbf {x}, \mathbf {y})} \log p _ {\theta} (\mathbf {y} \mid \mathbf {x}, \mathbf {z} _ {c}) \\ - \mathrm{KL} \left[ q _ {\phi} \left(\boldsymbol {z} _ {c} \mid \boldsymbol {x}, \boldsymbol {y}\right) \mid p \left(\boldsymbol {z} _ {c} \mid \boldsymbol {x}\right) \right] \tag {9.7} \\ \end{array}
    $$

    修改为：

    $$
    \log p (\mathbf {y} \mid \mathrm{do} (\mathbf {x})) \geq \mathbb {E} _ {\mathbf {z} _ {c} \sim q _ {\phi} (\mathbf {z} _ {c} | \mathbf {y})} \log p _ {\theta} (\mathbf {y} \mid \mathbf {x}, \mathbf {z} _ {c}) \tag {9.8}
    $$

    $$
    - \operatorname{KL} \left[ q _ {\phi} \left(\boldsymbol {z} _ {c} \mid \boldsymbol {y}\right) \mid p \left(\boldsymbol {z} _ {c}\right) \right]
    $$

    其中 $\theta$ 和 $\phi$ 分别表示先验网络和后验网络的参数。$z _ { c }$ 表示潜在的混淆变量，在 do-演算后应独立于 x。当进一步考虑代理混淆变量 c 时，公式 9.8 变为：

    $$
    \log p (\boldsymbol {y}, \boldsymbol {c} \mid \mathrm{do} (\boldsymbol {x})) \geq \mathbb {E} _ {\boldsymbol {z} _ {c} \sim q _ {\phi} (\boldsymbol {z} _ {c} | \boldsymbol {y}, \boldsymbol {c})} \log p _ {\theta} (\boldsymbol {y}, \boldsymbol {c} \mid \boldsymbol {x}, \boldsymbol {z} _ {c}) \tag {9.9}
    $$

    $$
    - \operatorname{KL} \left[ q _ {\phi} \left(\boldsymbol {z} _ {c} \mid \boldsymbol {y}, \boldsymbol {c}\right) \mid p \left(\boldsymbol {z} _ {c}\right) \right]
    $$

    由于 do-演算也会使代理混淆变量 c 独立于 x，公式 9.9 变为：

    $$
    \begin{array}{l} \log p (\boldsymbol {y} \mid \mathrm{do} (\boldsymbol {x})) \geq \mathbb {E} _ {\boldsymbol {z} _ {c} \sim q _ {\phi} (\boldsymbol {z} _ {c} | \boldsymbol {y}, \boldsymbol {c})} \left[ \log p _ {\theta} \left(\boldsymbol {y} \mid \boldsymbol {x}, \boldsymbol {z} _ {c}\right) + \log p _ {\theta} \left(\boldsymbol {c} \mid \boldsymbol {z} _ {c}\right) \right] \\ - \mathrm{KL} \left[ q _ {\phi} \left(\mathbf {z} _ {c} \mid \mathbf {y}, \mathbf {c}\right) \mid p \left(\mathbf {z} _ {c}\right) \right] - \log p (\mathbf {c}) \tag {9.10} \\ \end{array}
    $$

3.  **第三类研究** 采取了不同的方法，避免了对混淆变量或代理混淆变量的先验假设。相反，它通过利用多个数据集来隐式估计混淆变量。例如，Landeiro 等人 [39] 通过计算训练集和测试集的主题模型之间的差异来估计输入文档 X 中词语的影响。这种方法可以估计潜在的混淆变量，因为混淆变量可能在不同分布之间发生变化。然而，这种方法需要预先知道测试集的文本，这在现实场景中是一个不切实际的假设。最近的工作以不同的方式处理这个问题，假设可以获得从不同环境 $( e \in \mathcal { E } _ { \mathrm { a l l } } )$ 收集的多个数据集 $D _ { e } : = \smash { \big \{ \big ( \mathbf { x } _ { i } ^ { e } , \mathbf { y } _ { i } ^ { e } \big ) \big \} _ { i = 1 } ^ { n _ { e } } }$，其中 $n _ { e }$ 表示不同环境中数据集的数量 [2, 59]。这种方法的目标是学习一个稳健的预测模型 $Y = f ( X ; \theta )$，该模型在给定数量的环境中保持稳定 [2]。

## 9.3.3 反事实层面的去偏（Counterfactual-Level Debiasing）

反事实层面的去偏涉及生成反事实样本 $( \widetilde { \pmb x } , \widetilde { \pmb y } )$，并将其与观测样本 (x, y) 进行比较，以回答诸如“为什么？”或“预测的因果特征是什么？”等问题 [49]。**反事实数据增强（Counterfactual data augmentation）** 是用于此目的的常用方法，包括手动或自动生成反事实样本，并将其混合用于训练 [35]。反事实样本通常通过修改原始样本，使其导致机器学习模型做出不同预测的方式创建 [35, 73]。

根据 x 中的因果特征是否被操作，现有工作可以分为两类：

1.  **第一类方法** 涉及操作 x 的非因果特征，同时保持相应的标签 y 不变。此方法主要用于解决由某些敏感属性（如性别和种族）引起的**公平性（fairness）**问题 [23]。然而，它不能涵盖所有混淆变量。

2.  **第二类方法** 涉及对因果特征进行更改，从而将样本的标签从 $\textbf {  { y } }$ 翻转为 $\widetilde { \mathbf { y } }$ [35, 78, 90]。此方法已被证明可以提高模型的**分布外泛化（out-of-distribution generalization）**能力，并使模型对噪声更不敏感 [35]。

根据用于修改数据的方法，近期的工作也可以分为三类：

1.  **手动修改**，如 [35] 等研究中所述，涉及由人工标注者对文本进行微调以更改标签，同时避免任何不影响标签的不必要修改。此方法可以产生高质量的反事实样本，但标注成本可能很高。
2.  **基于规则的修改**，例如将文本中特定类型的对象词汇替换为另一种类型的词汇。如 [23, 72, 89] 中提出的这种方法，优点是成本低，但可能导致文本不自然。
3.  **自动生成反事实样本**，如 [64, 90, 91] 中提出的，使用像 GPT-2 这样的预训练模型来执行词汇替换和属性编辑等操作以生成反事实样本。此方法通过更具成本效益并产生更流畅的文本，解决了前两种方法的局限性。然而，需要注意的是，文本生成仍然是一项具有挑战性的任务，并且生成文本中属性编辑的准确性和语义保真度是不确定的。

**明确回答假设性问题（Explicitly Answering What-if Questions）** 反事实数据增强工作的主要焦点是帮助模型识别用于决策的因果模式，而无需明确回答诸如“如果……会发生什么？”这样的反事实问题。然而，最近的研究表明，处于反事实层面的模型具备回答反事实问题的能力，如因果阶梯所示。为了促进这一点，已经构建了专门的**问答（Question Answering, QA）**数据集，例如 **WIQA** [77]，它由三部分组成：过程文本、影响图和假设性多选题。过程文本提供关于事件的信息，影响图描绘这些事件之间的因果关系，而假设性问题则源自这些图。另一个数据集 **Tat-QA** [97] 是为基于表格的问答开发的，这被认为是一项具有挑战性的任务。最近的工作提出了一种结合离散推理的反事实思维过程，用于此任务，作为传统 QA 目标的补充 [41]。具体来说，这种方法利用序列标注来识别表格中的相关单元格和文本的相关片段，以推断其语义。然后，它使用一组聚合算子进行符号推理，以得出最终答案。该方法还包括正则化项，以在问题上下文中监督目标事实，并监督推断反事实上下文所需的推导操作 [41]。

## 9.4 自然语言处理模型的因果解释（Causal Interpretations of NLP Models）

近年来，深度神经模型在自然语言处理（Natural Language Processing, NLP）领域取得了显著成功，但其深层结构和非线性特性使其难以被解释，而这一点对于用户信任人工智能（Artificial Intelligence, AI）系统至关重要。这个问题在大型**预训练语言模型（Pre-trained Language Models, PLMs）** 的开发中尤为突出，因为其参数量巨大且具有非线性特征。此外，在自然语言处理中，诸如词元 **n-gram（n-grams）** 等基础文本特征可能无法捕捉文本中传达的高级语义。即使文本传达了诸如主题或情感等抽象语言概念，这些概念也可能并未在模型输入中被显式编码，从而导致缺乏清晰的可解释性 [19]。

尽管许多综述性文章试图对现有工作进行分类 [6, 7, 19, 82, 84]，我们建议遵循 Madsen 等人 [46] 提出的分类方法，该方法根据两个分类维度对每项工作进行分类：

1.  **局部或全局解释（Local or global interpretations）**，取决于该方法是解释单个实例（称为“局部解释”）还是整个模型（称为“全局解释”）[46]。局部解释提供对单个观测的洞察，例如，识别对预测最重要的输入特征。另一方面，全局解释则针对特定方面总结整个模型，例如模型如何将词语相互关联、模型使用的语言信息，或总结模型某个方面的通用规则。
2.  **内在或事后解释（Intrinsic or post-hoc interpretations）**。对可解释性的需求通常源于对问责制的要求。在模型决策后果严重的情况下，通过在部署前解释模型来最小化模型失败的风险至关重要 [69]。这意味着区分可解释性是主动应用（部署前）还是追溯应用（部署后）这两种情况非常重要 [46]。可以追溯应用的方法也被称为“事后”方法，而“内在”一词则用于指代那些在设计上就是可解释的模型。

从第一个维度来看，**估计平均处理效应（Average Treatment Effect, ATE）** 是一种全局解释，而**估计个体处理效应（Individual Treatment Effect, ITE）** 则是一种局部解释 [81]。估计平均处理效应涉及的处理变量可以是文本概念 [19]，也可以是像性别这样的二值变量 [82]。尽管平均处理效应估计属于全局解释，但它需要反事实样本估计，这通常使用输入的局部扰动，但可能导致不准确或误导性的解释。例如，当两个可能解释模型预测的概念彼此高度相关时，就可能发生这种情况 [19]。Fader 等人开发了一种为任何文本概念提供因果解释的方法，并创建了一个数据集，以便将任何因果估计量与真实情况进行比较 [19]。他们还创建了一种语言表示，可用于近似给定概念的反事实，从而无需手动创建示例即可解释因果模型。估计个体处理效应涉及回答反事实问题。例如，最近的工作在法律判决预测任务中估计了个体处理效应，旨在回答诸如“如果输入文本不包含某些概念，预测的判决会是什么？”之类的“假设”问题 [11]。

从第二个维度来看，估计平均处理效应或个体处理效应 [11, 19, 82] 是一种事后解释，因为它主要关注行为，而不是寻找模型的趣味特性。

除了估计处理效应来回答“假设”问题之外，Moraffah 等人 [49] 指出了下一层次的可解释性，即**反事实解释（counterfactual explanation）**，正如 Pearl [56] 所建议的，其目的是回答“为什么”的问题。与“假设”问题的区别在于，反事实解释需要通过执行影响输出的最小改变来生成反事实样本 [12, 49]。这意味着回答“为什么”的问题将聚焦于少量的文本特征 [17, 21, 67, 82, 91]。

## 9.5 总结（Summary）

总之，本章讨论了因果推断与自然语言处理交叉领域所带来的挑战与机遇，并解决了两个基本问题：自然语言处理如何利用文本数据辅助因果推断，以及因果推断理论如何提高自然语言处理模型的鲁棒性和可解释性。首先，本章概述了利用文本数据进行因果推断的最新进展，并强调了由于文本的非结构化和高维特性所带来的障碍。其次，我们表明**虚假相关（spurious correlation）** 问题仍然是自然语言处理模型面临的重大挑战，这可能导致不可靠的决策和推理捷径，限制了模型在现实场景中的鲁棒性和适用性。第三，本章探讨了面向自然语言处理的因果驱动模型，包括将因果关系整合到自然语言处理模型中的干预层面和反事实层面的去偏方法。最后，我们提出了因果解释在促进对自然语言处理模型更深层次理解方面的潜力。

## 参考文献（References）

1.  R. Aralikatte et al., Focus attention: promoting faithfulness and diversity in summarization, in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (2021), pp. 6078–6095
2.  M. Arjovsky et al., Invariant risk minimization (2019). arXivabs/1907.02893
3.  D. Bahdanau, K. Cho, Y. Bengio, Neural machine translation by jointly learning to align and translate, in 3rd International Conference on Learning Representations, ICLR (2015)
4.  E. Bareinboim, J. Pearl, Controlling selection bias in causal inference, in Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics, PMLR. vol. 22 (2012), pp. 100–108
5.  E. Bareinboim et al., On pearl’s hierarchy and the foundations of causal inference, in Probabilistic and Causal Inference (2022)
6.  Y. Belinkov, S. Gehrmann, E. Pavlick, Interpretability and analysis in neural NLP, in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: Tutorial Abstracts (2020), pp. 1–5
7.  Y. Belinkov, J. Glass, Analysis methods in neural language processing: a survey, Trans. Assoc. Comput. Linguist. 7, 49–72 (2019)
8.  R. Bommasani, C. Cardie, Intrinsic evaluation of summarization datasets, in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP (2020), pp. 8075–8096
9.  W. Chen et al., De-confounded variational encoder-decoder for logical table-to-text generation, in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL-IJCNN (2021), pp. 5532–5542
10. W. Chen et al., Dependent multi-task learning with causal intervention for image captioning, in Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, eds. by Z.-H. Zhou. Main Track. International Joint Conferences on Artificial Intelligence Organization (2021), pp. 2263–2270. https://doi.org/10.24963/ijcai.2021/312
11. W. Chen et al., Exploring logically dependent multi-task learning with causal inference, in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2020), pp. 2213–2225
12. S. Choudhary, N. Chatterjee, S.K. Saha, Interpretation of black box NLP models: a survey (2022). arXiv preprint arXiv:2203.17081
13. M. Cornia et al., Meshed-memory transformer for image captioning, in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR (2020), pp. 10575–10584
14. J. Devlin et al., BERT: pre-training of deep bidirectional transformers for language understanding, in: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT (2019), pp. 4171–4186
15. N. Egami et al., How to make causal inferences using texts (2018). arXiv abs/1802.02163
16. N. Egami et al., How to make causal inferences using texts. Sci. Adv. 8(42) (2022). eabg2652. https://www.science.org/doi/pdf/10.1126/sciadv.abg2652
17. Y. Elazar et al., Amnesic probing: behavioral explanation with amnesic counterfactuals. Trans. Assoc. Comput. Linguist. 9, 160–175 (2021)
18. A. Feder et al., Causal inference in natural language processing: estimation, prediction, interpretation and beyond (2021). arXiv abs/2109.00725
19. A. Feder et al., CausaLM: causal model explanation through counterfactual language models. Comput. Linguist. 47(2), 333–386 (2021)
20. Y. Feng et al., Modeling fluency and faithfulness for diverse neural machine translation. Proc. AAAI Conf. Artif. Intell. 34(01), 59–66 (2020)
21. M. Finlayson et al., Causal analysis of syntactic agreement mechanisms in neural language models, in Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL-IJCNLP 2021 (Association for Computational Linguistics (ACL), 2021), pp. 1828–1843
22. C. Fong, J. Grimmer, Discovery of treatments from text corpora, in Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, ACL (2016), pp. 1600–1609
23. S. Garg et al., Counterfactual fairness in text classification through robustness, in Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, AIES (2019), pp. 219–226
24. M. Gill, A.B. Hall, How judicial identity changes the text of legal rulings, in Political Methods: Quantitative Methods eJournal (2015)
25. S. Gururangan et al., Annotation artifacts in natural language inference data, in 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT (2018), pp. 107–112
26. I. Habernal et al., The argument reasoning comprehension task: identification and reconstruction of implicit warrants, in Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT (2018), pp. 1930–1940
27. D. Hendrycks, K. Lee, M. Mazeika, Using pre-training can improve model robustness and uncertainty, in Proceedings of the 36th International Conference on Machine Learning, ICML, vol. 97. Proceedings of Machine Learning Research (2019), pp. 2712–2721
28. D. Hendrycks et al., Pretrained transformers improve out-of-distribution robustness, in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL (2020), pp. 2744–2751
29. D. Hovy, S. Prabhumoye, Five sources of bias in natural language processing. Lang. Linguist. Compass 15(8), e12432 (2021)
30. D. Hovy, A. Søgaard, Tagging performance correlates with author age, in Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing, ACL-IJCNLP (2015), pp. 483–488
31. G. Hripcsak et al., Causal inference from observational healthcare data: implications, impacts and innovations, in American Medical Informatics Association Annual Symposium, AMIA (2020)
32. Z. Hu, L.E. Li, A causal lens for controllable text generation. Adv. Neural Inf. Process. Syst. 34, 24941–24955 (2021)
33. S. Iyer, N. Dandekar, K. Csernai et al., First quora dataset release: question pairs (2017). data.quora.com
34. Z. Ji et al., Survey of hallucination in natural language generation, in ACM Computing Surveys (2022)
35. D. Kaushik, E.H. Hovy, Z.C. Lipton, Learning the difference that makes a difference with counterfactually-augmented data, in 8th International Conference on Learning Representations, ICLR (2020)
36. K. Keith, D. Rice, B. O’Connor, Text as causal mediators: research design for causal estimates of differential treatment of social groups via language aspects, in Proceedings of the First Workshop on Causal Inference and NLP (2021), pp. 21–32
37. K.A. Keith, D. Jensen, B. O’Connor. Text and causal inference: a review of using text to remove confounding from causal estimates, in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL (2020), pp. 5332–5344
38. V. Landeiro, A. Culotta, Robust text classification under confounding shift. J. Artif. Intell. Res. 63, 391–419 (2018)
39. V. Landeiro, T. Tran, A. Culotta, Discovering and controlling for latent confounds in text classification using adversarial domain adaptation, in Proceedings of the 2019 SIAM International Conference on Data Mining, SDM (2019), pp. 298–305
40. H. Li et al., Ensure the correctness of the summary: incorporate entailment knowledge into abstractive sentence summarization, in Proceedings of the 27th International Conference on Computational Linguistics (2018), pp. 1430–1441
41. M. Li et al., Learning to imagine: integrating counterfactual thinking in neural discrete reasoning, in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (2022), pp. 57–69
42. A. Lin et al. One-stage deep instrumental variable method for causal inference from observational data, in 2019 IEEE International Conference on Data Mining, ICDM (2019), pp. 419–428
43. B. Liu et al., Show, deconfound and tell: image captioning with causal inference, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022), pp. 18041–18050
44. Y. Liu et al., RoBERTa: a robustly optimized bert pretraining approach (2019). arXiv abs/1907.11692
45. C. Louizos et al., Causal effect inference with deep latent-variable models, in Annual Conference on Neural Information Processing Systems 2017, NeurIPS (2017), pp. 6446–6456
46. A. Madsen, S. Reddy, S. Chandar, Post-hoc interpretability for neural NLP: a survey (2021). arXiv preprint arXiv:2108.04840
47. T. McCoy, E. Pavlick, T. Linzen, Right for the wrong reasons: diagnosing syntactic heuristics in natural language inference, in Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL (2019), pp. 3428–3448
48. W. Miao, Z. Geng, E.J. Tchetgen Tchetgen, Identifying causal effects with proxy variables of an unmeasured confounder. Biometrika 105(4), 987–993 (2018)
49. R. Moraffah et al., Causal interpretability for machine learning-problems, methods and evaluation. ACM SIGKDD Explorations Newslett. 22(1), 18–33 (2020)
50. A. Naik et al., Stress test evaluation for natural language inference, in Proceedings of the 27th International Conference on Computational Linguistics, COLING (2018), pp. 2340–2353
51. R. allapati et al. Abstractive text summarization using sequence-to-sequence RNNs and beyond, in Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning (2016), pp. 280–290
52. A. Nichols, Causal inference with observational data. Stata J. 7(4), 507–541 (2007)
53. T. Niven, H.-Y. Kao, Probing neural network comprehension of natural language arguments, in Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL (2019), pp. 4658–4664
54. Y. Pan et al., X-linear attention networks for image captioning, in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR (2020), pp. 10968–10977
55. J. Pearl, Causality, 2nd ed. (Cambridge University Press, Cambridge, 2009)
56. J. Pearl, Theoretical impediments to machine learning with seven sparks from the causal revolution (2018). arXiv preprint arXiv:1801.04016
57. J. Pearl, D. Mackenzie, The Book of Why: The New Science of Cause and Effect, 1st edn. (Basic Books, Inc., New York, 2018)
58. A. Perez-Suay, G. Camps-Valls, Causal inference in geoscience and remote sensing from observational data. IEEE Trans. Geosci. Remote. Sens. 57(3), 1502–1513 (2019)
59. M. Peyrard et al., Invariant language modeling, in EMNLP 2022 (2021)
60. R. Pryzant, Y. Chung, D. Jurafsky, Predicting sales from the language of product descriptions, in Proceedings of the SIGIR 2017 Workshop On eCommerce Co-located with the 40th International ACM SI-GIR Conference on Research and Development in Information Retrieval, eCOM@SIGIR (2017)
61. R. Pryzant et al., Causal effects of linguistic properties, in NAACL-HLT (2021)
62. R. Pryzant et al., Causal effects of linguistic properties, in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT (2021), pp. 4095–4109
63. R. Pryzant et al., Deconfounded lexicon induction for interpretable social science, in Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT (2018), pp. 1615– 1625
64. A. Radford et al., Language models are unsupervised multitask learners. OpenAI Blog 1(8), 9 (2019)
65. M.E. Roberts, B.M. Stewart, R.A. Nielsen, Adjusting for Confounding with Text Matching. Am. J. Polit. Sci. 64, 887–903 (2020)
66. J.M. Rohrer, Thinking clearly about correlations and causation: Graphical causal models for observational data. Adv. Methods Practices Psychol. Sci. 1(1), 27–42 (2018)
67. A. Ross, A. Marasovic, M.E. Peters, Explaining NLP models via minimal contrastive editing ´ (MiCE), in Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021 (2021), pp. 3840–3852
68. D.B. Rubin, Estimating causal effects of treatments in randomized and nonrandomized studies. J. Educ. Psychol. 66(5), 688 (1974)
69. C. Rudin, Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nat. Mach. Intell. 1(5), 206–215 (2019)
70. B. Schölkopf, Causality for machine learning, in Probabilistic and Causal Inference: The Works of Judea Pearl (2022), pp. 765–804
71. B. Schölkopf et al., Toward causal representation learning. Proc. IEEE 109(5), 612–634 (2021)
72. R. Shekhar et al., FOIL it! Find One mismatch between Image and Language caption, in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL (2017), pp. 255–265
73. C. Shorten, T.M. Khoshgoftaar, B. Furht, Text data augmentation for deep learning. J. Big Data 8, 1–34 (2021)
74. H.A. Simon, Spurious correlation: a causal interpretation. J. Am. Statis. Assoc. 49(267), 467– 479 (1954)
75. D. Sridhar, D.M. Blei, Causal inference from text: a commentary. Sci. Adv. 8(42), eade6585 (2022)
76. D. Sridhar, L. Getoor, Estimating causal effects of tone in online debates, in Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI (2019), pp. 1872–1878
77. N. Tandon et al., WIQA: a dataset for “What if. . . ” reasoning over procedural text, in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (2019), pp. 6076–6085
78. B. Tian et al., Debiasing NLU models via causal intervention and counterfactual reasoning. Proc. AAAI Conf. Artif. Intell. 36(10), 11376–11384 (2022)
79. R. Tian et al., Sticking to the facts: confident decoding for faithful data-to-text generation (2019). arXiv preprint arXiv:1910.08684
80. L. Tu et al., An empirical study on robustness to spurious correlations using pre-trained language models. Trans. Assoc. Comput. Linguist. 8, 621–633 (2020)
81. V. Veitch, D. Sridhar, D.M. Blei, Adapting text embeddings for causal inference, in Proceedings of the Thirty-Sixth Conference on Uncertainty in Artificial Intelligence, UAI, vol. 124. Proceedings of Machine Learning Research (2020), pp. 919–928
82. J. Vig et al., Causal mediation analysis for interpreting neural NLP: the case of gender bias (2020). arXiv preprint arXiv:2004.12265
83. O. Vinyals, Q.V. Le, A neural conversational model, in ICML Deep Learning Workshop (2015)
84. E. Wallace, M. Gardner, S. Singh, Interpreting predictions of NLP models, in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Tutorial Abstracts (2020), pp. 20–23
85. A. Wang et al., GLUE: a multi-task benchmark and analysis platform for natural language understanding, in 7th International Conference on Learning Representations, ICLR (2019)
86. T. Wang et al., Visual Commonsense R-CNN, in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR (2020), pp. 10757–10767
87. X. Wang, H. Wang, D. Yang, Measure and improve robustness in NLP models: a survey, in Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (2022), pp. 4569–4586
88. Z. Wang, A. Culotta, Identifying spurious correlations for robust text classification, in Findings of the Association for Computational Linguistics: EMNLP (2020), pp. 3431–3440
89. Z. Wang, A. Culotta, Robustness to spurious correlations in text classification via automatically generated counterfactuals, in Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI (2021), pp. 14024–14031
90. J. Wen et al., AutoCAD: automatically generating counterfactuals for mitigating shortcut learning (2022). arXiv preprint arXiv:2211.16202
91. T. Wu et al., Polyjuice: generating counterfactuals for explaining, evaluating, and improving models, in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL-IJCNN (2021), pp. 6707–6723
92. K. Xu et al., Show, attend and tell: neural image caption generation with visual attention, in Proceedings of the 32nd International Conference on Machine Learning, ICML, vol. 37. JMLR Workshop and Conference Proceedings (2015), pp. 2048–2057
93. K. Xu et al., Show, attend and tell: neural image caption generation with visual attention, in International Conference on Machine Learning. PMLR (2015), pp. 2048–2057
94. X. Yang, H. Zhang, J. Cai, Deconfounded image captioning: a causal retrospect, in IEEE Transactions on Pattern Analysis and Machine Intelligence (2021)
95. Y. Zhang, J. Baldridge, L. He, PAWS: paraphrase adversaries from word scrambling, in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT (2019), pp. 1298– 1308
96. M. Zhou et al., Progress in neural NLP: modeling, learning, and reasoning. Engineering 6(3), 275–290 (2020)
97. F. Zhu et al., TAT-QA: a question answering benchmark on a hybrid of tabular and textual content in finance, in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (2021), pp. 3277–3287
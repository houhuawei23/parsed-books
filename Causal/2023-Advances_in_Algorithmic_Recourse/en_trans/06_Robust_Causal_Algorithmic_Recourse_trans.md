# 鲁棒因果算法追索（Robust Causal Algorithmic Recourse）

## 章节摘要（Chapter Abstract）

**算法追索（Algorithmic recourse）**旨在为个体提供可操作的建议，以逆转自动化决策系统所做出的不利结果。理想情况下，追索建议应能对寻求追索个体特征中合理的不确定性具有鲁棒性。在本工作中，我们提出了**对抗鲁棒追索问题（adversarially robust recourse problem）**，并表明提供最小成本追索的追索方法无法实现鲁棒性。接着，我们提出了在线性和可微情形下生成对抗鲁棒追索的方法。最后，我们通过实验表明，对决策分类器进行正则化以使其更依赖**可操作特征（actionable features）**，有助于对抗鲁棒追索的存在。

## 6.1 引言（Introduction）

**机器学习（Machine Learning, ML）分类器**越来越多地被用于司法和金融等领域的重大决策（例如，批准审前保释或贷款）。尽管个体面临的自动决策日益增多，但维护人类能动性的需求推动了对**算法追索（algorithmic recourse）**的研究，其旨在通过为个体提供可操作的建议来逆转不利的算法决策，从而增强个体能力（USL19）。先前的研究认为，要使追索值得信赖，决策者必须承诺，在决策主体完全采纳其规定的追索建议后，逆转不利决策（WMR17；VA20；Kar+22）。我们认为，如果算法追索确实应被视为一种契约协议，那么追索建议必须对追索过程中出现的合理不确定性具有鲁棒性。

例如，考虑一家银行承诺，如果某个个体将其储蓄增加一定金额，就会批准其贷款。假设当该个体实现了规定的储蓄增长时，其每周工作小时数因不可预见的情况而略有减少，而分类器仍然认为该个体可能拖欠贷款。事后通过仍然批准贷款来使追索建议免受不确定性的影响，可能对银行（例如，金钱损失）和个体（例如，破产和无法获得未来贷款）都不利，而违背追索承诺则会否定个体所付出的努力，并侵蚀对决策者的信任。因此，我们主张必须确保追索建议在事前对不确定性具有鲁棒性。

在本工作中，我们将重点放在使追索建议对寻求追索个体特征的不确定性具有鲁棒性上。这种不确定性可能源于追索的时间性质（例如，某些特征可能不是静态的），和/或噪声、对抗性操纵以及其他错误陈述或误差的存在。我们采用**鲁棒优化（robust optimization）**的观点，并提议通过定义一个**不确定性集（uncertainty set）** $B(\mathbf{x})$ 来刻画围绕报告个体特征 $\mathbf{x}$ 的不确定性，我们假设该集合包含在提供追索时个体的真实特征和/或由于追索的时间性质而产生的个体特征的合理变化。然后，我们寻求鲁棒的追索建议，这些建议对于不确定性集中的所有可能个体都保持有效（即，导致有利的分类结果），如图 6.1 所示。我们将这种鲁棒性概念称为追索的**对抗鲁棒性（adversarial robustness）**。

![image_20](images/image_20.png)

鲁棒追索动作
非鲁棒追索动作
$\mathbf{x}$

图 6.1：**对抗鲁棒追索动作**必须对围绕寻求追索个体 $\mathbf{x}$ 的不确定性集中的所有个体都产生正向的分类结果。

我们从因果关系的角度（Pea09）研究追索的对抗鲁棒性。**因果追索（Causal recourse）**将追索建议视为对决策主体特征的因果干预（KSV21），因此，只要底层的**结构因果模型（Structural Causal Model, SCM）**已知或可以被合理近似，它就更真实地描述了当个体根据其追索建议行动时个体特征的变化方式（Kar+20b）。

## 贡献（Contributions）

*   我们提出了**对抗鲁棒追索问题**，并表明最小成本追索建议被证明对寻求追索个体特征的不确定性是脆弱的。
*   我们提出了在线性和可微情形下生成**对抗鲁棒因果追索（adversarially robust causal recourse）**的方法。我们在线性和神经网络分类器的五个表格数据集上展示了其有效性。
*   我们提出了一种**模型正则化器（model regularizer）**，鼓励决策分类器更强烈地依赖可操作特征。我们通过实验表明，我们提出的模型正则化器有助于对抗鲁棒追索的存在。

## 6.2 背景与相关工作（Background and Related Work）

### 6.2.1 因果关系背景（Background on Causality）

我们假设个体 $\mathbf{x} \in \mathcal{X}$ 的特征 $\mathbf{X} = \{X_1, \ldots, X_n\}$ 的数据生成过程由一个已知的**结构因果模型（Structural Causal Model, SCM）**（Pea09） $\mathcal{M} = (\mathbb{S}, P_{\mathbf{U}})$ 刻画。**结构方程（structural equations）** $\mathbb{S} = \{X_i := f_i(\mathbf{X}_{\mathrm{pa}(i)}, U_i)\}_{i=1}^n$ 描述了任意给定特征 $X_i$、其直接原因 $\mathbf{X}_{\mathrm{pa}(i)}$ 以及某个**外生变量（exogenous variable）** $U_i$ 之间的因果关系，作为一个确定性函数 $f_i$。外生变量 $\mathbf{U} \in \mathcal{U}$ 根据某个概率分布 $P_{\mathbf{U}}$ 分布，代表负责数据中观察到的变化的未观测背景因素。我们假设由 SCM 隐含的**因果图（causal graph）**（节点为 $\mathbf{X} \cup \mathbf{U}$，边为 $\{(v, X_i) : v \in \dot{\mathbf{X}}_{\mathrm{pa}(i)} \cup U_i, i \in [1, n]\}$）是无环的。那么 SCM 在特征 $\mathbf{X}$ 上隐含一个唯一的**观测分布（observational distribution）** $p$。此外，结构方程 $\mathbb{S}$ 诱导了一个外生变量与内生变量之间的映射 $\mathbb{S} : \mathcal{U} \to \mathcal{X}$。在外生变量相互独立（因果充分性）的假设下，如果存在某个逆映射 $\mathbb{S}^{-1} : \dot{\mathcal{X}} \to \mathcal{U}$ 使得 $\mathbb{S}(\mathbb{S}^{-1}(\mathbf{x})) = \mathbf{x}, \forall \mathbf{x} \in \mathcal{X}$，那么对应于某个个体 $\mathbf{x} \in \mathcal{X}$ 的内生变量可由 $\mathbf{U}|\mathbf{x} = \mathbb{S}^{-1}(\mathbf{x})$ 唯一标识。

SCM 允许建模和评估对 SCM 所建模系统的干预效果。**硬干预（Hard interventions）** $\mathrm{do}(\mathbf{X}_{\mathcal{I}} := \boldsymbol{\theta})$（Pea09）通过改变被干预变量的结构方程 $\mathbb{S}_{\mathcal{I}_i}^{\mathrm{do}(\mathbf{X}_{\mathcal{I}} := \boldsymbol{\theta})} = \mathbf{X}_{\mathcal{I}_i} := \boldsymbol{\theta}_i$，同时保留其余结构方程 $\mathbb{S}_i^{\mathrm{do}(\boldsymbol{\chi}_{\mathbb{Z}} := \boldsymbol{\theta})} = \mathbb{S}_i$，将特征子集 $\mathcal{I} \subseteq [d]$ 的值 $\mathbf{x}_{\mathcal{I}}$ 固定为某个 $\boldsymbol{\theta} \in \mathbb{R}^{|\mathcal{I}|}$。因此，硬干预切断了被干预变量与其因果图中所有祖先之间的因果关系。另一方面，**软干预（Soft interventions）**可能以更一般的方式修改结构方程（Kor+04）。特别是，**加性干预（additive interventions）**用某个扰动向量 $\boldsymbol{\Delta \Psi} \in \mathbb{R}^n$ 扰动特征 $\mathbf{X}$，同时保留所有因果关系，根据下式改变**反事实（counterfactual, CF）**结构方程：

$$
\mathbb{S}^{\Delta} = \left\{X_i := f_i(\mathbf{X}_{\mathrm{pa}(i)}, U_i) + \Delta_i\right\}_{i=1}^n \quad (\text{ESo7}).
$$

此外，SCM 隐含了**反事实（counterfactuals）**上的分布，允许推理在其他条件相同的情况下，在某种假设干预下本会发生什么。在上述假设下，对应于某个观测到的**事实个体（factual individual）** $\mathbf{x} \in \mathcal{X}$ 在某种假设的硬干预 $\mathrm{do}(\mathbf{X}_{\mathcal{I}} := \boldsymbol{\theta})$（或软干预 $\Delta$）下的反事实 $\mathbf{x}^{\mathsf{CF}}$ 可以通过首先确定对应于个体 $\mathbf{x}$ 的外生变量 $\mathbf{U}|\mathbf{x} = \mathbb{S}^{-1}(\mathbf{x})$，然后应用从外生变量到内生变量的**干预映射（interventional mapping）** $\mathbb{S}^{\mathrm{do}(\mathbf{X}_{\mathcal{I}} := \boldsymbol{\theta})}$（或 $\mathbb{S}^{\Delta}$）来计算（Pea09）。为方便记法，我们将此类映射表示为 $\mathbf{x}^{\mathbb{CF}} = \mathbb{CF}(\mathbf{x}, \mathrm{do}(\mathbf{X}_{\mathcal{I}} := \boldsymbol{\theta})) := \mathbb{S}^{\mathrm{do}(\mathbf{X}_{\mathcal{I}} := \boldsymbol{\theta})}(\mathbb{S}^{-1}(\mathbf{x}))$（或 $\mathbf{x}^{\mathsf{CF}} = \mathbb{CF}(\bar{\mathbf{x}}, \bar{\Delta}) := \mathbb{S}^{\Delta}(\mathbb{S}^{-1}(\mathbf{x}))$）。我们使用记号 $\mathbf{x}^{\mathsf{CF}} = \mathbb{CF}(\mathbf{x}, \mathbf{do}(\mathbf{X}_{\mathcal{I}} := \boldsymbol{\theta}), \mathcal{M})$（或 $\mathbf{x}^{\mathsf{CF}} \overset{\cdot}{=} \mathbb{CF}(\mathbf{x}, \Delta, \mathcal{M})$）来强调反事实对应于特定的结构因果模型。

### 6.2.2 因果追索问题（The Causal Recourse Problem）

考虑使用分类器 $h : \mathcal{X} \to \{0, 1\}$ 将有利或不利结果分配给个体 $\mathbf{x} \in \mathcal{X}$（例如，贷款批准）的设置。我们采用 Karimi 等人 [KSV21] 引入的因果追索观点，并将追索建议建模为对寻求追索个体特征的硬干预，即 $\boldsymbol{a} = \mathbf{do}(\mathbf{X}_{\mathcal{I}} := \mathbf{x}_{\mathcal{I}} + \boldsymbol{\theta})$，其中 $\boldsymbol{\theta}$ 是对某些变量 $\mathbf{x}_{\mathcal{I}}$ 的规定改变。我们考虑这种加性形式，而不是 Karimi 等人 [KSV21] 使用的 $\boldsymbol{a} = \mathbf{do}(\mathbf{X}_{\mathcal{I}} := \boldsymbol{\theta})$，以明确允许事实个体 $\mathbf{x}$ 中的不确定性传播到追索建议 $\boldsymbol{a}$。

对于一个追索动作 $\boldsymbol{a}$ 被认为是有效的，相应的反事实个体必须被有利地分类，即 $h(\mathbb{CF}(\mathbf{x}, a, \mathcal{M})) = 1$。由于某些特征可能是不可变的（例如，种族）或有界的（例如，年龄），因此只应推荐可行的动作。**动作可行性集（action feasibility set）** $\mathcal{F}(\mathbf{x})$ 捕获了可用于个体 $\mathbf{x}$ 的可行动作集合。理想情况下，追索建议应为决策主体带来尽可能小的努力，其中**成本函数（cost function）** $c(\mathbf{x}, a)$ 对个体 $\mathbf{x} \in \mathcal{X}$ 实施追索动作 $a$ 所需的努力进行建模。因此，为某个个体 $\mathbf{x} \in \mathcal{X}$ 找到最小成本追索动作等价于解决以下优化问题：

$$
\underset{a = \mathrm{do}(\mathbf{X}_{\mathcal{I}} := \mathbf{x}_{\mathcal{I}} + \boldsymbol{\theta})}{\text{argmin}} \quad c(\mathbf{x}, a)
$$

$$
\text{s.t.} \quad a \in \mathcal{F}(\mathbf{x}) \tag{6.1}
$$

$$
h(\mathbb{CF}(\mathbf{x}, a, \mathcal{M})) = 1
$$

如公式 6.1 所强调，个体 $\mathbf{x}$、分类器 $h$ 和/或 SCM 特征的不确定性可能影响追索的有效性。在附录 E.1 中，我们讨论并关联了在追索过程中出现的不同不确定性来源。

**非因果追索设置（non-causal recourse setting）**等同于在**独立可操作特征（Independently Manipulable Features, IMF）**假设下的因果追索设置，即如果个体特征之间不存在因果关系。在此假设下，$\mathbb{CF}(\mathbf{x}, \mathrm{do}(\mathbf{X} := \mathbf{x} + \boldsymbol{\theta})) = \mathbf{x} + \boldsymbol{\theta}$。

### 6.2.3 相关工作（Related Work）

我们现在与关于追索鲁棒性的现有文献建立联系。先前的工作考虑了在分类器 $h$ 的不确定性下生成保持有效的追索动作的问题。Pawelczyk 等人 [PBK20] 表明，与最小成本追索动作相比，将反事实个体置于具有大数据支持的特征空间区域的追索动作在**预测多重性（predictive multiplicity）**下更具鲁棒性。然而，具有大数据支持的追索动作可能成本过高。相比之下，我们的方法寻求以尽可能低的成本找到鲁棒的追索动作。另一项工作考虑了追索对分类器因数据集偏移而变化的鲁棒性。Rawal 等人 [RKL20b] 表明，追索动作通常对此类模型变化不具有鲁棒性，而 Upadhyay 等人 [UJL21] 旨在通过使用**极小极大优化过程（minimax optimization procedure）**生成追索来缓解此问题，其中追索成本在追索动作对分类器 $h$ 的对抗性变化保持有效的条件下被最小化。虽然我们采用类似的极小极大方法来生成鲁棒追索，但我们专注于使追索对个体 $\mathbf{x}$ 的不确定性具有鲁棒性，而不是分类器 $h$。最后，Black 等人 [Bla+21] 采用**分布鲁棒优化（distributionally robust optimization）**方法来生成在由初始训练条件的微小变化产生的不同分类器 $h$ 之间一致的追索建议。同样，我们工作的一个自然扩展是采用分布鲁棒的观点。

关于追索对 SCM 不确定性的鲁棒性，Karimi 等人 [Kar+20b] 考虑了底层 SCM 未知因而必须被近似的设置，并提出了一种追索方法来生成由于底层 SCM 的错误指定而无效概率较低的追索建议。我们的工作与 Karimi 等人 [Kar+20b] 是相切的。

最后，先前的工作已经发现，决策主体 $\mathbf{x}$ 特征的微小变化可能导致不同的追索建议，其追索成本可能差异很大（Küg+22；Sla+21；Art+21）。我们不关注追索成本，而是研究追索有效性的鲁棒性。Virgolin 和 Fracaros [VF22] 的并行工作与我们的工作最为相似，因为他们考虑了追索对个体 $\mathbf{x}$ 的对抗性扰动的鲁棒性。他们提出了一种进化算法来生成鲁棒追索，并为随机森林分类器提供了实证结果。相比之下，我们专注于为可微分类器生成追索，并为线性和神经网络分类器提供了实证结果。此外，我们考虑了更一般的因果追索设置，并以因果方式对特征扰动进行建模。

## 6.3 反事实不确定性集（Counterfactual Uncertainty Sets）

在对抗鲁棒性文献中，某个数据点 $\mathbf{x}$ 特征的不确定性通常通过围绕 $\mathbf{x}$ 的一个 $\epsilon$-球不确定性集 $B(\mathbf{x}) = \{\mathbf{x} + \boldsymbol{\Sigma} \Delta \mid ||\Delta|| \leq \epsilon\}$ 来建模，其中范数 $||\cdot||$ 刻画了数据点之间某种相关的相似性概念 $d(\mathbf{x}, \mathbf{y}) = ||\mathbf{x} - \mathbf{y}||$，而 $\epsilon$ 刻画了存在的不确定性量（Mad+18；Ber+19）。直观地说，对数据点 $\mathbf{x}$ 的微小扰动 $\Delta$ 会产生相似的数据点。那么，不确定性集 $B(\mathbf{x})$ 可以解释为与观测数据点 $\mathbf{x}$ 相似的可能数据点的邻域。

从因果角度来看，在 IMF 假设下，即特征之间不存在因果关系时，这种特征变化 $\delta$ 等价于对特征 $\mathbf{x}$ 的加性干预。然而，我们认为，明确考虑这些因果关系可能会提供更具信息量的个体邻域。

**定义 6.3.1（反事实相似个体的邻域）。** 对于某个相似性范数、SCM 和事实个体 $\mathbf{x}$，我们将 $\mathbf{x}$ 的 $\epsilon$-邻域的反事实相似个体定义为所有可能的 $\epsilon$-小加性干预下的反事实集合：

$$
B(\mathbf{x}) = \left\{\mathbb{CF}(\mathbf{x}, \Delta, \mathcal{M}) \mid ||\Delta|| \leq \epsilon \right\} \tag{6.2}
$$

作为一个激励性示例，考虑一个 SCM，其特征为 $X_1 = U_1$ 和 $X_2 = X_1 + U_2$，分别表示某个个体 $\mathbf{x}$ 的收入和储蓄。图 6.2 展示了对于 2-范数相似性度量 $||\cdot||_2$ 的观测邻域和反事实邻域。观察到，在反事实邻域下，个体 $\mathbf{x}$ 与收入更高、储蓄也更高的个体 $\bar{\mathbf{x}}$ 更相似，而不是与收入更高但储蓄更低的另一个个体 $\tilde{\mathbf{x}}$ 更相似，因为后者不能很好地被 SCM 解释，因此其境况可能与 $\mathbf{x}$ 的境况有本质差异（例如，有更多数量的受抚养人，导致尽管收入更高但储蓄更低）。因此，我们认为反事实邻域可能比观测邻域更具信息量，因为它明确考虑了特征之间的因果关系。

## 6.4 对抗鲁棒追索问题（The Adversarially Robust Recourse Problem）

我们考虑生成对寻求追索个体特征的不确定性具有鲁棒性的追索动作的问题。我们采用**鲁棒优化（robust optimization）**的观点，并要求鲁棒的追索动作对不确定性集 $B(\mathbf{x})$ 中的每个可能个体都保持有效。

**定义 6.4.1（对抗鲁棒追索问题）。** 对于某个不确定性集 $B(\mathbf{x})$，对不确定性集 $B(\mathbf{x})$ 中的所有可能个体 $\mathbf{x}' \in B(\mathbf{x})$ 都保持有效的最小成本追索动作由下式给出：

$$
\underset{a = \operatorname{do}(\mathbf{X}_{\mathcal{I}} := \mathbf{x}_{\mathcal{I}} + \boldsymbol{\theta})}{\text{argmin}} \max_{\mathbf{x}' \in B(\mathbf{x})} c(\mathbf{x}, a) \tag{6.3}
$$

$$
\mathrm{s.t.} \quad a \in \mathcal{F}(\mathbf{x}') \wedge h(\mathbb{CF}(\mathbf{x}', a)) = 1
$$

观察到上述优化问题的任何解 $a$ 必须满足 $h(\mathbb{CF}(\mathbf{x}', a)) = 1, \forall \mathbf{x}' \in B(\mathbf{x})$，因此是**对抗鲁棒的（adversarially robust）**。在附录 E.2 中，我们推导了对抗鲁棒追索存在的充分条件。

### 6.4.1 在温和条件下追索是脆弱的（Recourse is Fragile Under Mild Conditions）

我们证明，在成本函数 $c$、可行性集 $\mathcal{F}(\mathbf{x})$ 和 SCM 的温和条件下，最小成本追索动作被证明对寻求追索个体特征中任意小的不确定性是脆弱的。

**定理 6.4.1。** 设 $a^*$ 是公式 6.1 中陈述的追索优化问题的解。假设

(i) 成本函数 $c(\mathbf{x}, \mathrm{do}(\mathbf{X}_{\mathcal{I}} := \mathbf{x}_{\mathcal{I}} + \boldsymbol{\theta}))$ 在 $\boldsymbol{\theta}$ 上严格凸，且最小值为 0。
(ii) $\forall 0 < t < 1, \mathrm{do}(\boldsymbol{X}_{\mathbb{Z}} := \boldsymbol{x}_{\mathbb{Z}} + \boldsymbol{\theta})) \in \mathcal{F}(\boldsymbol{x}) \implies \mathrm{do}(\boldsymbol{X}_{\mathbb{Z}} := \boldsymbol{x}_{\mathbb{Z}} + t\boldsymbol{\theta})) \in \mathcal{F}(\boldsymbol{x})$。
(iii) SCM 是一个**加性噪声模型（additive noise model）**（Pea09）。

则存在 $\mathbf{x}' \in B(\mathbf{x}) = \{\mathbb{CF}(\mathbf{x}; \Delta) \mid ||\Delta|| \leq \epsilon > 0\}$ 使得 $h(\mathbb{CF}(\mathbf{x}', a^*)) = 0$，即对于任意小的 $\epsilon > 0$，追索动作 $a^*$ 都是脆弱的。

条件 (i) 由最广泛使用的成本函数满足，即**加权 p-范数（weighted p-norms）**（Kar+20b）和**百分位成本（percentile costs）**（USL19）。条件 (ii) 由追索文献中通常假设的**盒式动作约束（box actionability constraints）**满足（Kar+22）。最后，条件 (iii) 是从数据估计底层 SCM 的常见建模假设（Kar+20b），并且在非因果追索设置中也成立。

因此，在算法追索文献通常考虑的设置中，寻求最小成本追索的追索方法提供了被证明是脆弱的追索建议。这一结果激励了研究生成对抗鲁棒追索的追索方法。

## 6.5 生成对抗稳健的追索（Generating Adversarially Robust Recourse）

## 6.5.1 线性情形（The Linear Case）

对于一个线性分类器 $h ( \mathbf { x } ) = \left. \mathbf { w } , \mathbf { x } \right. \geq b$ 和线性 SCM，我们证明，为 $h$ 生成稳健追索等价于为一个修改过的线性分类器 $h ^ { \prime } ( \mathbf { x } ) = \langle \mathbf { w } , \mathbf { x } \rangle \geq \bar { b } ^ { \prime }$ 生成标准追索，该修改分类器的“接受阈值”被充分提高，即 $b ^ { \prime } \geq b$。

**定理 6.5.1**。设 $h ( \mathbf { x } ) = \langle \mathbf { w } , \mathbf { x } \rangle \geq b$ 为一个线性分类器，SCM 具有线性结构方程，且 $B ( \mathbf { x } ) = \{ { \mathbb { C } } \mathbb { F } \left( \mathbf { x } , \Delta \right) \ | \ \| \Delta \| \leq \epsilon \}$ 为合理个体的不确定性集合。如果可行性集对 $\mathbf { x , }$ 的扰动具有不变性，即 $\forall \mathbf { x } ^ { \prime } \in B ( \mathbf { x } ) : \mathcal { F } ( \mathbf { x } ) \overset { \cdot } { = } \mathcal { F } ( \mathbf { x } ^ { \prime } )$ ，那么对于分类器 $h ( \mathbf { x } )$ 的最小成本对抗稳健追索动作等价于对于修改后分类器的最小成本稳健追索动作：

$$
h ^ {\prime} (\mathbf {x}) = \left\langle \mathbf {w}, \mathbf {x} \right\rangle \geq b + \left\| J _ {\mathbb {S} ^ {\mathcal {I}}} ^ {T} \mathbf {w} \right\| ^ {*} \epsilon \tag {6.4}
$$

其中 $\left\| \cdot \right\| ^ { * }$ 表示 $\left\| \cdot \right\|$ 的对偶范数，$J _ { { \mathbb S } ^ { \mathbb T } }$ 表示对特征 $\mathbf { \boldsymbol { x } } _ { \mathcal { T } }$ 进行硬干预（hard-intervening）所得干预映射的雅可比矩阵。

我们强调这一结果的重要性：如果定理 6.5.1 的条件成立，那么通过考虑修改后的分类器 $h ^ { \prime }$ ，任何给定的追索生成方法都可以用于生成对抗稳健追索。特别地，对抗稳健性可以很容易地与其他期望特性相结合，例如大数据支持（Jos+19; PBK20）或公平性约束（Gup+19; Küg+22）。

## 6.5.2 可微情形（The Differentiable Case）

与 Wachter 等人 [WMR17] 类似，我们考虑以下目标函数：

$$
\mathcal {L} (\mathbf {x}, a, \lambda) = c (\mathbf {x}, a) + \lambda \ell (h (\mathbb {C F} (\mathbf {x}, a)), 1) \tag {6.5}
$$

其中 $\ell$ 是二元交叉熵损失。那么对抗稳健追索问题等价于以下无约束惩罚问题：

$$
\max _ {\lambda \geq 0} \min _ {a \in \mathcal {F} (\mathbf {x})} c (\mathbf {x}, a) + \lambda \max _ {\mathbf {x} ^ {\prime} \in B (\mathbf {x})} \ell (h (\mathbb {C F} (\mathbf {x}, a)), 1) \tag {6.6}
$$

我们提出使用在不确定性集合 $B ( \mathbf { x } )$ 上的投影梯度上升（projected gradient ascent）来求解内部最大化问题。针对本文中考虑的不确定性集合的特定形式，我们投影到 $\epsilon$ 球上，因为 $\mathbf { m a x } _ { \mathbf { x ^ { \prime } } \in B ( \mathbf { x } ) }$ $\begin{array} { r } { \ell \left( h \left( \mathbf { C F } \left( \mathbf { x } , a \right) \right) , 1 \right) = \operatorname* { m a x } _ { \| \Delta \| \leq \epsilon } \ \ell \left( h \left( \mathbf { C F } \left( \mathbf { C F } ( \mathbf { x } , \Delta ) , a \right) \right) , 1 \right) } \end{array}$ 。然而，请注意，上述优化目标在 $\Delta$ 中通常是非凸的，因此使用梯度上升找到的局部最大值可能不是 $B ( \mathbf { x } )$ 中的全局最大值。因此，无法保证所提出算法返回的追索动作是对抗稳健的。但是，正如第 $7$ 节所讨论的，我们通过实验发现，对于足够小的不确定性 $\epsilon$ ，所提出的算法在使追索对不确定性具有稳健性方面是有效的。

对于方程 6.6 中的外部最大最小优化问题，我们采用 Karimi 等人 [Kar+20b] 的因果追索方法，并在追索动作 $a$ 和可行性集 $\mathcal { F } ( \mathbf { x } )$ 上使用投影梯度下降，同时迭代增加 $\lambda$ 以越来越强调跨越分类器的决策边界。我们在算法 7 中展示了所提出的优化过程。

**算法 $\boldsymbol { \mathrm { 7 } }$**：为可微分类器和 SCM 生成对抗稳健追索。

**输入：** 事实个体 x，不确定性集合 $B(\mathbf{x})$，干预集 I，$\lambda > 0, \gamma > 1$ $\theta \leftarrow 0$ 当 $N \leq N_{\max}$ 时执行
    当未收敛时执行 $a \leftarrow \text{do}(\mathbf{X}_{\mathcal{I}} := \mathbf{x}_{\mathcal{I}} + \boldsymbol{\theta}) \mathbf{x}^* \leftarrow \arg\max_{\mathbf{x}' \in B(\mathbf{x})} \ell(h(\mathbb{CF}(\mathbf{x}, a)), 1) \text{ if } h(\mathbb{CF}(\mathbf{x}^*, a)) = 1 \text{ then } \text{ return } \boldsymbol{\theta}$ $\theta \leftarrow \text{Proj}_{\mathcal{F}(\mathbf{x})} (\theta - \alpha \nabla_{\theta} \mathcal{L}(\mathbf{x}^*, a, \lambda))$ $\lambda \leftarrow \gamma \lambda$

## 6.6 可行动性正则化（Actionability Regularization）

为了确保追索建议具有稳健性，个体需要比原本必须付出的更多努力。因此，使追索对不确定性免疫的负担完全落在了决策对象身上。然而，我们认为，稳健追索的期望特性可以直接嵌入到分类器的训练中。满足这些期望特性可能会以预测准确性为代价，从而将稳健追索的部分负担从决策对象转移到决策者身上。在本节中，我们首先将自己限制在线性情形，以便从理论上激励一种正则化惩罚，以减少稳健追索的额外成本。然后，通过从局部线性正则化（local linearity regularization）[Qin+19]（对抗稳健性文献中的一种流行技术）中汲取灵感，我们将这种正则化扩展到可微情形。我们发现，所提出的正则化器显著促进了对抗稳健追索的存在。

## 6.6.1 稳健追索成本的上界（Upper Bounding the Cost of Robust Recourse）

我们将自己限制在线性情形，以便在特定的可行动性假设下推导出稳健追索额外成本的上界。

**定理 6.6.1**。设 $h$ 是一个线性分类器 $h ( \mathbf { x } ) = \langle \mathbf { w } , \mathbf { x } \rangle \geq b ,$ ，SCM 具有线性结构方程，$\textbf { \textit { x } } \in \mathbf { \textit { X } }$ 是一个被负分类的个体，对于该个体存在某个追索动作 $a \ = \ \mathrm { d o } ( \mathbf { X } _ { \mathcal { T } } \ : = \ \mathbf { x } _ { \mathcal { T } } + \pmb { \theta } )$ ，且 $B ( \mathbf { x } ) \ = \ \{ { \mathbb { C } } \mathbb { F } \left( \mathbf { x } , \Delta \right) \ | \ \| \Delta \| \le \epsilon \}$ 。那么，存在某个常数 $\beta$ ，使得如果 $\begin{array} { r } { a ^ { \prime } = \mathrm { d o } ( \mathbf { X } _ { \mathcal { T } } : = \mathbf { x } _ { \mathcal { T } } + ( 1 + \beta \epsilon ) \pmb { \theta } ) } \end{array}$ 是一个可行动作 $a ^ { \prime } \in \mathcal { F } ( \mathbf { x } )$ ，则 $a ^ { \prime }$ 是一个对抗稳健追索动作。假设成本函数是次可加的（subadditive），由稳健化动作 $a$ 所产生的额外成本为：

$$
\frac {c (\mathbf {x} , a ^ {\prime}) - c (\mathbf {x} , a)}{c (\mathbf {x} , a)} \leq \beta \epsilon , \quad \beta = \frac {\left\| J _ {\mathrm{S} ^ {\mathcal {I}}} ^ {T} \mathbf {w} \right\| ^ {*}}{\langle J _ {\mathrm{S} ^ {\mathcal {I}}} ^ {T} \mathbf {w} , \boldsymbol {\theta} \rangle} \tag {6.7}
$$

因此，$\beta \epsilon$ 构成了因寻求稳健追索而产生的追索额外成本的上界。我们提出对 $w$ 进行正则化，以降低追索额外成本的上界 $\beta \epsilon$ 。为简单起见，我们此后做出 **IMF 假设（IMF assumption）**，使得 $J _ { \mathbb { S } ^ { \mathcal { T } } } ^ { T } = I .$ 。令 $\boldsymbol { A }$ （相应地，U）为可行动特征集（相应地，不可行动特征集），$m _ { \mathcal { A } } \in [ 0 , 1 ] ^ { n }$ （相应地，$m _ { \mathcal { U } } \in \mathsf { [ 0 , 1 ] } ^ { n }$ ）为掩码向量，使得 $( m _ { \mathcal { A } } ) _ { i } = 1 \iff i \in \mathcal { A }$ （相应地，$( m _ { \mathcal { U } } ) _ { i } = 1 \iff i \in \mathcal { U }$ ）。那么：

$$
\beta = \frac {\left\| \mathbf {w} \right\| ^ {*}}{\langle \mathbf {w} , \boldsymbol {\theta} \rangle} = \frac {\left\| m _ {\mathcal {A}} \odot \mathbf {w} \right\| ^ {*} + \left\| m _ {\mathcal {U}} \odot \mathbf {w} \right\| ^ {*}}{\langle m _ {\mathcal {A}} \odot \mathbf {w} , \boldsymbol {\theta} \rangle} \tag {6.8}
$$

其中 $\odot$ 表示逐元素乘积。因此，降低与不可行动特征相对应的分类器权重的对偶范数 $\| m _ { \mathcal { U } } \odot \mathbf { w } \| ^ { * }$ ，可以直接降低稳健追索额外成本的上界 $\beta$ ，从而引入学习偏差“分类器应更强烈地依赖可行动特征”。

## 6.6.2 可行动的局部线性正则化（Actionable Local Linearity Regularization）

我们考虑形如 $h ( \mathbf { x } ) = g ( \mathbf { x } ) \geq b$ 的分类器，其中 $g ( \pmb { x } )$ 是可微的。为了降低稳健追索的额外成本，我们提出以下正则化器：

$$
\begin{array}{l} \mathcal {R} (\mathbf {x}) = \mu \| m _ {\mathcal {U}} \odot \nabla_ {x} g (\mathbf {x}) \| ^ {*} \\ + \gamma \max _ {\| \delta \| \leq \epsilon} | g (\mathbf {x} + \delta) - \langle \delta , \nabla_ {x} g (\mathbf {x}) \rangle - g (\mathbf {x}) | \tag {6.9} \\ \end{array}
$$

我们将其称为**可行动的局部线性正则化器（Actionable Locally Linear Regularizer, ALLR）**。第一项对应于之前推导的针对分类器 $h$ 在 $\mathbf { x , }$ 附近的线性近似 $h ^ { \prime }$ 的可行动性惩罚，第二项受 Qin 等人 [Qin+19] 启发，鼓励函数 $g$ 在 $\mathbf {x}$ 附近表现为线性，从而使得线性分类器 $h ^ { \prime }$ 是 $h$ 在 $\mathbf {x}$ 附近的一个合理准确的近似。

## 6.7 实验结果（Experimental Results）

首先，我们通过实验验证了所提出的用于生成对抗稳健追索方法的有效性。其次，我们通过实验表明，使用我们提出的 ALLR 正则化器对决策分类器进行正则化，有助于寻找对抗稳健追索。

我们考虑了四个真实世界数据集和一个半合成数据集。对于因果追索设置，我们考虑了 COMPAS 再犯数据集（Lar+16b）和 Adult 人口统计数据集（Mur94），我们采用了 Nabi 和 Shpitser [NS18] 中假设的因果图，并将结构方程拟合为单层 MLP。我们还考虑了 Karimi 等人 [Kar+20b] 引入的一个半合成 SCM，该 SCM 受贷款审批场景启发。我们从该 SCM 中采样了 1000 个数据点，并将生成的数据集称为 Loan。对于非因果追索设置，我们考虑了南德信贷数据集（Gro19），以及来自北卡罗来纳州的再犯数据集（SW88），我们将其称为 Bail。在附录 $\mathrm { E . 4 }$ 中，我们列出了每个数据集使用的特征以及所考虑的可行动性约束。

对于所考虑的数据集，我们将可行动的分类变量视为实值变量，并对所有实值特征进行标准化。我们使用规定特征变化的 $\ell _ { 1 }$ 范数作为成本函数，即 $c ( \mathbf { x } , a = \mathrm { d o } ( \mathbf { X } _ { \mathcal { T } } : =$ $\mathbf { x } _ { \mathcal { T } } + \pmb { \theta } ) ) = \| \pmb { \theta } \| _ { 1 }$ 。我们考虑两种类型的分类器：逻辑回归（Logistic Regression, LR）模型和神经网络（Neural Network, NN）模型（3 层，tanh 激活）。我们根据 2-范数定义不确定性集合 $B ( \pmb { x } )$ 。

![image_21](images/image_21.png)

**图 6.4**：针对不确定性进行稳健化处理的追索的脆弱性（Fragility of recourse robustified against uncertainty）。对于线性分类器，我们无法找到使生成的追索无效的扰动。对于 NN 分类器，对于足够大的不确定性 $\epsilon$ ，我们确实找到了这样的对抗性扰动。图例：COMPAS Adult Loan Credit Bail.

## 6.7.1 最小成本追索是脆弱的（Minimum-cost recourse is fragile）

首先，我们通过实验证明，旨在生成最小成本追索的追索方法缺乏稳健性。为此，我们使用期望风险最小化（expected risk minimization）训练分类器，并分别使用 Wachter 等人 [WMR17] 和 Karimi 等人 [KSV21] 的方法为因果和非因果追索设置中被负分类的个体生成追索。然后，我们将 C&W 对抗攻击（CW17）应用于寻求追索的个体的特征，以找到使生成的追索无效的最小特征扰动。我们在图 6.3 中展示了结果。

我们观察到，为 LR 和 NN 分类器生成的追索都是脆弱的，对抗扰动的大小在 $1 0 ^ { - 2 } \ \mathrm { t o } \ 1 0 ^ { - 9 }$（对于标准化特征）范围内。我们观察到，LR 分类器的追索明显更脆弱，这是因为 LR 分类器的追索问题是凸的，因此可以更精确地找到最小成本追索动作。

## 6.7.2 生成对抗稳健追索（Generating adversarially robust recourse）

我们评估了第 5.2 节中提出的用于生成对抗稳健追索方法的有效性。为此，我们使用期望风险最小化训练分类器，并针对具有不同不确定性水平 $\epsilon \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , \dot { 0 } . 5 \}$ 的不同不确定性集合 $B ( \mathbf { x } )$ 生成追索。然后，我们使用 C&W 对抗攻击来寻找个体特征的扰动 $\Delta$，这些扰动会使生成的追索动作无效。如果我们找到某个扰动 $\| \Delta \| _ { 2 } \le \epsilon$ 使得生成的追索动作无效，我们可以说该追索动作是脆弱的。然而，反之则不成立，因为未找到扰动并不能证明此类对抗扰动不存在。

我们在图 6.4 中展示了实验结果。对于 LR 模型，我们无法找到使生成的追索无效的对抗扰动。实际上，找到的所有扰动都仅比 $\epsilon$ 大一个任意小的量，但不会更小。因此，对于 LR 模型，我们提出的方法可以有效地生成成本最小的稳健追索。然而，对于 NN 模型，它呈现出更具挑战性的优化景观，在足够大的不确定性 $\epsilon$ 下，我们提出的方法可能会生成脆弱的追索动作。尽管如此，总体而言，与我们之前考虑的标准最小成本追索生成方法相比，我们提出的方法生成的追索明显不那么脆弱。

## 6.7.3 可行动的局部线性正则化（Actionable local linearity regularization）

我们通过实验评估了使用所提出的 ALLR 正则化器训练的分类器是否有助于对抗稳健追索的存在。据我们所知，Ross 等人 [RLB21] 是唯一提出模型正则化器以促进算法追索存在的工作。他们提出的正则化器通过考虑训练目标来用“反事实示例”增强模型训练：

$$
\mathbb {E} _ {(\mathbf {x}, y) \sim p (\mathbf {x}, y)} [ \ell (h (\mathbf {x}), y) + \lambda \min \delta \ell (h (\mathbf {x}), 1) ] \tag {6.10}
$$

我们将我们提出的 ALLR 正则化器与 Ross 等人 [RLB21] 的正则化器以及另外两个基线进行比较：经验风险最小化（无正则化）和仅使用可行动特征（Actionable Features, AF）的分类器，后者在极限强正则化 $\mu \to \infty$ 下等同于 ALLR 正则化。我们使用每种正则化方法训练五个分类器，并评估找到追索的个体的百分比，以及在无不确定性 $\epsilon = 0$ 和存在显著不确定性 $\epsilon = 0 . 1$ 情况下的追索成本。我们还通过评估预测准确率以及马修斯相关系数（Matthews correlation coefficient, MCC）来评估正则化对分类器性能的影响程度。

我们在图 6.5 和图 6.6 中展示了实验结果。我们发现，对于 LR 和 NN 模型，我们提出的正则化器通常非常有效地促进了对抗稳健追索的存在。此外，我们发现，对于 LR 模型，我们提出的分类器还可以显著降低稳健追索的成本，正如第 6 节中的理论动机所述。

![image_22](images/image_22.png)

**图 6.5**：对于 LR 模型，我们发现对分类器进行 ALLR 正则化（惩罚对应于不可行动特征的权重）显著促进了对抗稳健追索的存在，其效果优于 Ross 等人的正则化器。此外，相应的稳健追索动作可能比使用 ERM 训练的分类器产生的追索动作成本更低。我们还发现，预测性能受到的影响通常低于 Ross 等人和 AF 正则化器。图例：ERM ALLR Ross et al. AF Accuracy MCC score.

## 6.8 结论（Conclusion）

追索过程中的不确定性是不可避免的。先前提出的用于减轻追索过程中不确定性影响的**事后（ex-post）**解决方案可能对决策者和个体都产生负面结果。相反，我们采用一种**事前（ex-anti）**方法来处理追索的稳健性，要求追索建议对寻求追索的个体特征中的不确定性具有稳健性。我们表明，在实践中，最小成本追索对个体特征中任意小的不确定性都是脆弱的。为了解决这个问题，我们形式化了对抗稳健追索问题，并提出了在线性和可微情形下生成对抗稳健追索的方法。最后，我们提出了一个模型正则化器，鼓励决策分类器更强烈地依赖可行动特征，并且我们通过实验表明，我们提出的正则化器显著促进了对抗稳健追索的存在。

![image_23](images/image_23.png)

NN 分类器  
**图 6.6**：对于 NN 模型，我们发现对分类器进行 ALLR 正则化显著促进了对抗稳健追索的存在，其程度与 AF 正则化器相当。我们还发现，预测模型的预测性能并未受到很大影响。图例：ERM ALLR Ross et al. AF Accuracy MCC score.
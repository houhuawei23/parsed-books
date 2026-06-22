# 算法追索的进展：确保因果一致性、公平性与鲁棒性（Advances in Algorithmic Recourse: Ensuring Causal Consistency, Fairness, & Robustness）

## 苏黎世联邦理工学院博士学位论文

提交以获取

苏黎世联邦理工学院理学博士学位

（Dr. sc. ETH Zurich）

由

**Amir-Hossein Karimi**

提交

滑铁卢大学计算机科学数学硕士

出生于1992年6月22日

伊朗、加拿大公民

经以下人员推荐接受：

**Prof. Dr. Bernhard Schölkopf**（苏黎世联邦理工学院），

**Prof. Dr. Isabel Valera**（萨尔大学），

**Prof. Dr. Benjamin Grewe**（苏黎世联邦理工学院），

---

# 算法追索的进展：确保因果一致性、公平性与鲁棒性（Advances in Algorithmic Recourse: Ensuring Causal Consistency, Fairness, & Robustness）

**机器学习（Machine learning）**正越来越多地被用于指导敏感情境中的关键决策，在这些情境中，决策对个人生活产生深远影响。例如审前保释、贷款审批、简历筛选或重要药物处方。在此类情境中，模型必须准确、鲁棒，同时维护公平性、隐私性、问责制和可解释性等社会相关价值观。这些方面显著影响着这些技术的接受度和影响力。

在本论文中，我特别关注**算法追索（algorithmic recourse）**的实现与促进任务。这涉及为个人提供易于理解的解释和建议，说明如何最有效地（高效且理想情况下低成本）从自动化系统做出的不利决策中恢复。本论文涉及以下研究问题：

**q1. 我们如何在不同情境下为受影响的个人提供追索？** 针对这一问题，我提出了一种新颖的算法，用于生成**模型无关的反事实解释（Model-Agnostic Counterfactual Explanations, MACE）**，该算法基于形式化验证的标准理论和工具。该方法克服了以往策略的局限性，支持模型无关、数据类型无关和距离无关的反事实解释。它还能为任何个人提供合理且多样化的反事实，并以可证明的最优距离呈现。

**q2. 从反事实解释中可以得出哪些可操作的见解？** 我认为解释必须使人们能够采取行动，而不仅仅是理解。通过反例和**结构因果模型（Structural Causal Models, SCM）**的理论，我证明可操作的建议通常无法从反事实解释中推断出来。我提出了新的优化问题，用于生成**最小后果性干预（Minimal Consequential Interventions, MINT）**，在已知真实SCM的情况下提供精确追索，在仅知因果图的情况下提供概率性追索。

**q3. 提供追索解释/建议如何影响其他利益相关者？** 在本论文的第三部分，我认为为个人提供追索权应在其对其他利益相关者以及公平性、隐私性和模型/IP安全等其他期望属性的影响的更广泛背景下加以考虑。我定义并提出了提供**公平追索（fair recourse）**的解决方案，并讨论了不确定性和非平稳性如何影响所提供的追索。我探索了**鲁棒追索（robust recourse）**策略，并讨论了可能促进公平/鲁棒追索的分类器或数据生成过程的潜在变化。

总之，本论文为未来研究方向提供了路线图，挑战了现有假设，并将追索领域扩展到了监督学习之外。

---

**机器学习（Maschinelles Lernen）**正越来越多地被用于指导敏感情境中的关键决策，在这些情境中，决策对个人生活产生深远影响。例如审前保释决定、贷款审批、简历筛选或改变生活的药物处方。在此类情境中，模型必须精确且鲁棒，同时维护公平性、隐私性、问责制和可解释性等社会价值观。这些价值观显著影响着这些技术的接受度和影响力。

在本论文中，我特别关注**算法追索（Algorithmischen Recourse）**的实现与促进。这涉及为受影响的个人提供易于理解的解释和建议，说明如何最有效地（高效且理想情况下低成本）从自动化系统做出的不利决策中恢复。本论文涉及的研究问题如下：

**q1. 我们如何在不同的情境下为受影响的个人提供追索？** 为回答这一问题，我提出了一种新颖的算法，用于生成**模型无关的反事实解释（modellagnostischen kontrafaktischen Erklärungen, MACE）**，该算法基于形式化验证的标准理论和工具。该方法克服了以往策略的局限性，且是模型、数据类型和距离无关的。它能生成任何个人的合理且多样化的反事实解释，且距离可证明为最优。

**q2. 从反事实解释中可以得出哪些可操作的见解？** 我认为解释应激励人们采取行动，而不仅仅是理解。通过反例和**结构因果模型（strukturellen Kausalmodelle, SCM）**的理论，我证明可操作的建议通常无法从反事实解释中推断出来。我提出了新的优化问题，用于直接生成**最小后果性干预（minimaler konsequenzieller Interventionen, MINT）**，在已知真实SCM的情况下提供精确追索，在仅知因果图的情况下提供概率性追索。

**q3. 提供追索解释/建议如何影响其他利益相关者？** 在本论文的第三部分，我认为为个人提供追索应在其对其他利益相关者以及公平性、隐私性和模型/IP安全等额外期望属性的更广泛背景下加以考虑。我定义并提出了提供**公平追索（fairem Recourse）**的解决方案，并讨论了不确定性和非平稳性如何影响所提供的追索。我研究了**鲁棒追索（robusten Recourse）**策略，并讨论了可能支持公平/鲁棒追索的分类过程或数据生成过程的潜在变化。

总之，本论文为未来研究方向提供了方向，挑战了现有假设，并将追索的应用范围扩展到了监督学习之外。

---

诚实的先知曾言："从摇篮到坟墓，求知不辍"

（阿布·卡西姆·费尔多西）

攻读博士学位如同踏上一条无尽的发现之路，印证了倡导终身学习的永恒智慧。我无比幸运，有一群人以他们的智慧和支持照亮了这条道路。你们对我能力的坚定信念为我提供了克服这段旅程中挑战所需的支撑。衷心感谢你们成为我学术征程中的火炬手。

致我的导师 **Prof. Dr. Bernhard Schölkopf** 和 **Prof. Dr. Isabel Valera**，我感谢你们对我独立思维能力的培养，慷慨地付出时间，耐心帮助我应对挫折，相信我的能力，并激励我成为最好的自己。你们的共同指导使我敢于跨越重洋，在异国他乡攻读博士学位。我深深感激你们给予我的这个机会。

致 **Prof. Dr. Gilles Barthe**，感谢您热情欢迎我参与第一个博士项目，认真倾听我的想法，指导我进行指导工作，并将我视为受尊敬的同事。您对研究的热情具有感染力，我希望能继续与您合作。

致 **Prof. Dr. Thomas Hofmann**，感谢您在我于苏黎世联邦理工学院交流期间给予的坦诚对话、敏锐见解和热情款待。如果没有您的支持，我在苏黎世联邦理工学院的时光将会截然不同。

致 **Prof. Adrian Weller**，感谢您在自己休假期间，在最后一刻慷慨地主持了关于因果伦理机器学习的ELLIS研讨会小组讨论。

致我珍爱的朋友们 **Adrián Javaloy Bornas**、**Dr. Patrick Putzky**、**Julius von Kügelgen**、**Kamil Adamczewski**、**Dr. Krikamol Muandet**、**Dr. Antonio Vergari** 和 **Dr. Atalanti Mastakouri**，我珍视我们的"创意"咖啡时光、德语搭档练习以及深刻的人生讨论。我非常感谢 Patrick 和 Kamil 在我整个博士期间帮助我搬家。致整个EI系团队，感谢你们教会我研究的方法，并激励我每天进步。

致 **Miriam Rateike** 和 **Pablo Sanchez-Martin**，你们在组织萨尔大学的因果伦理机器学习ELLIS研讨会和因果伦理机器学习研讨会方面的坚定支持和合作是无价的。此外，我感谢 **Martina Contisciani**，您鼓舞人心的教学风格激励了我们共同举办因果性迷你课程。与你们一起创建这些活动既愉快又收获颇丰。

致 **Dr. Been Kim**、**Dr. Simon Kornblith** 和 **Dr. Lars Beusing**，感谢你们在我于Google Brain和DeepMind实习期间的热情接待。我在那里获得了宝贵的知识！

致我的老朋友 **Arman Ghaffarizadeh**，感谢你在喜悦和困难时刻倾听我的心声并给予智慧。我珍视我们的友谊，多年来这份友谊只增不减。

致我的学生们 **Alexandra Walter**、**Kiarash Mohammadi**、**Ricardo Dominguez-Olmedo** 和 **Ahmad Ehyaei**，你们的耐心使我得以成长为你们的导师。我希望我没有辜负你们的时间。

致 **Prof. Caterina De Bacco** 和她愉快的学生团队，你们让午餐和咖啡休息时间变得令人神清气爽。

致马克斯·普朗克研究所和苏黎世联邦理工学院友善且乐于助人的行政人员 **Sabrina Rehbaum（MPI）**、**Ann-Sophie Bähr（MPI）**、**Lidia Pavel（MPI）**、**Annika Buchholz（MPI）**、**Sarah Danes（MPI & ETH）**、**Paulina Motyka（ETH）** 和 **Natalia Marciniak（ETH）**，你们的支持使我能够将更多时间投入到研究中。

致**学习系统中心（Centre for Learning Systems, CLS）**、**加拿大自然科学与工程研究理事会（Natural Sciences and Engineering Research Council of Canada, NSERC）** 和 **Google**，感谢你们在我整个学术旅程中提供的慷慨博士奖学金。

致我最亲近和最珍爱的人。

致我的父母 **Prof. Gholamreza Karimi** 和 **Prof. Zohreh Azimifar**，以及我的兄弟 **Ali**，我所有的机会都归功于你们，你们始终为我设定高标准并在我一生中培养我。

最重要的是，致我亲爱的妻子 **Fatemeh**，我的"犯罪伙伴"，你坚定不移的爱、支持、牺牲和指导在最黑暗的时刻成为希望的灯塔。有你作为我的"同行伴侣"（hamsafar），我无比幸运，并热切期待我们未来的许多冒险！

最后，但当然并非最不重要的是，我向全能的 **é <Ë@** 表达我的感激之情，我所拥有的一切都归功于他。

---

以下同行评审出版物是我博士研究的核心内容，并涵盖在本论文中：

1. "Model-Agnostic Counterfactual Explanations for Consequential Decisions," Karimi, Barthe, Balle, Valera, AISTATS (Á), 2019.
2. "Algorithmic Recourse: from Counterfactual Explanations to Interventions," Karimi, Schölkopf, Valera, ACM-FAccT (­), 2020.
3. "Algorithmic recourse under imperfect causal knowledge: a probabilistic approach," Karimi\*, von Kügelgen\*, Schölkopf, Valera, NeurIPS (­), 2020.
4. "Scaling Guarantees for Nearest Counterfactual Explanations," Mohammadi, Karimi, Barthe, Valera, ACM-AIES (Á), 2021.
5. "A survey of algorithmic recourse: contrastive explanations and consequential recommendations," Karimi, Barthe, Schölkopf, Valera, ACM Computing Surveys (), 2022.
6. "Towards Causal Algorithmic Recourse," Karimi\*, von Kügelgen\*, Schölkopf, Valera, Springer LNAI Book Chapter, 2022.
7. "On the Fairness of Causal Algorithmic Recourse," von Kügelgen, Karimi, Bhatt, Valera, Weller, Schölkopf, AAAI (Á), 2022.
8. "On the Adversarial Robustness of Causal Algorithmic Recourse," Dominguez-Olmedo, Karimi, Schölkopf, ICML (­), 2022.
9. "Robustness Implies Fairness in Causal Algorithmic Recourse," Ehyaei, Karimi, Schölkopf, Maghsudi ACM-FAccT, 2023.

以下同行评审出版物是我博士期间完成的，但未包含在本论文中：

10. "On the Relationship Between Explanation and Prediction: A Causal View," Karimi, Muandet, Kornblith, Schölkopf, Kim, ICML, 2023.
11. "On Data Manifolds Entailed by Structural Causal Models," Dominguez-Olmedo, Karimi, Arvanitidis, Schölkopf, ICML, 2023.

所有代码均可在 https://github.com/amirhk 获取

口头报告 (Á)；亮点报告 (­)；≥100次引用 ()；同等贡献 (\*)

---

## 基础（Basic）

- $x$：标量
- $\mathbf{x}$：向量
- $\mathbf{X}$：矩阵
- $\mathcal{X}$：集合
- $\mathbf{X}$：随机变量
- $\mathcal{X}$：空间、模型或约束

## 追索（Recourse）

- $\mathcal{D}$：数据集
- $\phi$：逻辑公式
- $h : \mathcal{X} \rightarrow \mathcal{V}$：判别器
- $\mathcal{F}$：可行性约束
- $\mathcal{P}$：合理性约束
- $\mathsf{cost}(\cdot)$ 或 $c(\cdot)$：成本函数
- $\mathsf{dist}(\cdot, \cdot)$ 或 $d(\cdot, \cdot)$：距离函数
- $\mathbb{CF}_h(\mathbf{x}^{\mathsf{F}})$：实例 $\mathbf{x}^{\mathsf{F}}$ 和模型 $h$ 的反事实实例集合

## 因果性（Causality）

- $\mathbb{S}$：结构方程集合
- $P_{\mathbf{U}}$：潜变量上的分布
- $\mathcal{M} = (\mathbb{S}, P_{\mathbf{U}})$：结构因果模型
- $\mathcal{G}$：对应的图因果模型
- $\mathcal{T}$：图节点的子集
- $\mathrm{d}(\mathcal{T})$：子集 $\mathcal{T}$ 的后代
- $\mathrm{nd}(\mathcal{T})$：子集 $\mathcal{T}$ 的非后代
- $\Delta(\mathbf{X}_{\mathcal{T}} := \boldsymbol{\theta})$ 或 $\Delta(\boldsymbol{\theta}_{\mathcal{T}})$：通过软干预将 $\mathbf{x}_{\mathcal{T}}$ 的值设为 $\boldsymbol{\theta}$
- $\mathrm{do}(\mathbf{X}_{\mathcal{T}} := \boldsymbol{\theta})$ 或 $\mathrm{do}(\boldsymbol{\theta}_{\mathcal{T}})$：通过硬干预将 $\mathbf{x}_{\mathcal{T}}$ 的值设为 $\boldsymbol{\theta}$
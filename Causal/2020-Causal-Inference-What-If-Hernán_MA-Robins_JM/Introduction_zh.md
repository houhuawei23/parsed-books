# 因果推断（Causal Inference）：如果会怎样（What If）

米格尔·A·埃尔南（Miguel A. Hernán）与詹姆斯·M·罗宾斯（James M. Robins）

建议引用格式：Hernán MA, Robins JM (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.

本书在线版本可访问：https://miguelhernan.org/whatifbook

# 引言（Introduction）：迈向更严谨的因果推断（Towards Less Casual Causal Inferences）

《因果推断》（Causal Inference）这个书名显然有些自负。作为一项复杂的科学任务，**因果推断（causal inference）**依赖于对来自多个来源的证据进行三角验证（triangulating evidence），并应用多种方法论。没有哪本书能够全面描述所有科学领域中用于因果推断的全部方法论。任何一本《因果推断》的作者都必须选择他们希望重点阐述的因果推断方法论中的哪些方面。

本引言标题反映了我们自己的选择：一本帮助科学家——尤其是健康与社会科学家——生成和分析数据，以做出明确阐述因果问题以及数据分析所依据假设的因果推断的书。遗憾的是，科学文献中充斥着这样的研究：因果问题未被明确陈述，研究人员的不可验证假设也未予声明。这种对待因果推断的随意态度（casual attitude）导致了大量的混淆。

例如，我们经常可以看到这样的研究：其效应估计（effect estimates）难以解释，因为数据分析方法无法在研究人员的假设（如果它们被声明了）下恰当地回答因果问题（如果它被明确陈述了）。

在本书中，我们强调需要认真对待因果问题，将其清晰表述，并阐明数据和假设在因果推断中的不同作用。一旦这些基础得以建立，因果推断必然变得不那么随意（less casual），从而有助于防止混淆。本书描述了多种数据分析方法，用于在特定假设集下，当收集到总体中每个个体的数据时，估计感兴趣的因果效应。本书的一个关键信息是，**因果推断不能简化为一系列数据分析的配方**。

这不是一本哲学书。我们对诸如因果性（causality）和原因（cause）等形而上学概念基本持不可知论（agnostic）态度。相反，我们专注于**识别（identification）**和**估计（estimation）**总体中的因果效应，即衡量在不同干预下结局分布变化的数值量。例如，我们讨论如何估计严重心力衰竭患者在**接受**心脏移植与**未接受**心脏移植情况下的死亡风险。通过可操作的因果推断（actionable causal inference），我们希望帮助决策者做出更好的决策。

本书分为三个难度递增的部分：**第一部分（Part I）** 是关于无模型的因果推断（即因果效应的非参数识别）；**第二部分（Part II）** 是关于有模型的因果推断（即使用参数模型估计因果效应）；**第三部分（Part III）** 是关于复杂纵向数据的因果推断（即时变处理（time-varying treatments）的因果效应估计）。在整篇文本中，我们穿插了**精细要点（Fine Points）**和**技术要点（Technical Points）**，对正文中提到的某些主题进行详细阐述。**精细要点**旨在让所有读者都能理解，而**技术要点**则面向具有中级统计学训练的读者。

本书对目前分散在多个学科期刊中的因果推断概念和方法进行了统一的阐述。我们预计，所有从事因果推断的专业人士，包括流行病学家、统计学家、心理学家、经济学家、社会学家、政治学家、计算机科学家……都会对此书感兴趣。

本书源于我们的教学和研究活动。几代充满求知欲的哈佛学生帮助我们精炼了本书的内容。数十年来在健康应用中量化因果效应的方法论工作，帮助我们识别了在实践中什么是重要的，并在我们的研究中区分了本质与偶然。因此，本书应被视为我们教学和研究经验的一个（希望是有帮助的）综合，而非对所有先前工作的系统性回顾。本书包含了数百条引用——其中约三分之一是我们自己的工作——但我们当然未能引用因果推断方法论中的每一项重要贡献。此外，由于该领域广阔且不断发展，没有教科书能始终保持完全最新。我们事先向任何可能未在此看到自己工作被引用的同事致歉，并邀请他们与我们联系。（在本书出版前在线提供的约二十年期间，许多人确实这样做了，本书也因此变得更好。）对特定方法论发展史感兴趣的读者，鼓励阅读本书中引用的学术论文。

我们感谢许多使本书成为可能的人。斯蒂芬·科尔（Stephen Cole）、伊萨·达哈布雷（Issa Dahabreh）、桑德·格陵兰（Sander Greenland）、杰伊·考夫曼（Jay Kaufman）、埃莉诺·默里（Eleanor Murray）、托马斯·理查森（Thomas Richardson）、索尼娅·斯旺森（Sonja Swanson）、泰勒·范德韦勒（Tyler VanderWeele）和扬·范登布劳克（Jan Vandenbroucke）提供了详细的评论。古达尔兹·达纳伊（Goodarz Danaei）、小川浩介（Kosuke Kawai）、马丁·拉霍斯（Martin Lajous）和凯瑟琳·沃斯（Kathleen Wirth）帮助创建了 NHEFS 数据集。第二部分中的示例代码由罗杰·洛根（SAS）、埃莉诺·默里和罗杰·洛根（Stata）、乔伊·史（Joy Shi）和肖恩·麦格拉思（Sean McGrath）（R）以及詹姆斯·菲德勒（James Fiedler）（Python）开发。罗杰·洛根还是我们的 LaTeX 专家。兰德尔·查普特（Randall Chaput）帮助创建了第 1 章和第 2 章的图形。乔什·麦基布尔设计了本书封面。我们耐心的出版商罗布·卡尔弗（Rob Calver）鼓励我们撰写本书，并支持我们将其免费在线提供的决定。

此外，多位同事通过发现错别字和指出不清晰的段落，帮助我们改进了本书。我们特别感谢：Kafui Adjaye-Gbewonyo, Alvaro Alonso, Katherine Almendinger, Ingelise Andersen, Juan José Beunza, Karen Biala, Joanne Brady, Alex Breskin, Shan Cai, Yu-Han Chiu, Alexis Dinno, John Ferguson, James Fiedler, Birgitte Frederiksen, Tadayoshi Fushiki, Leticia Grize, Dominik Hangartner, Niels Hagenbuch, Michael Hudgens, John Jackson, Marshall Joffe, Luke Keele, Laura Khan, Dae Hyun Kim, Lauren Kunz, Martín Lajous, Angeliki Lambrou, Wen Wei Loh, Haidong Lu, Mohammad Ali Mansournia, Giovanni Marchetti, Lauren McCarl, Shira Mitchell, Louis Mittel, Hannah Oh, Ibironke Olofin, Robert Paige, Jeremy Pertman, Melinda Power, Bruce Psaty, Brian Sauer, Tomohiro Shinozaki, Ian Shrier, Yan Song, Øystein Sørensen, Etsuji Suzuki, Denis Talbot, Mohammad Tavakkoli, Sarah Taubman, Evan Thacker, Kun-Hsing Yu, Vera Zietemann, Helmut Wasserbacher, Jessica Young, and Dorith Zimmermann.

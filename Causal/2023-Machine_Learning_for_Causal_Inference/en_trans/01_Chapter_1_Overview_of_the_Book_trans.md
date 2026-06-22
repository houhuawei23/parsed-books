# 第1章 本书概述（Chapter 1 Overview of the Book）

![image_01](images/image_01.png)

褚智轩和李晟（Zhixuan Chu and Sheng Li）

**机器学习（Machine Learning）**与**因果推断（Causal Inference）**是近年来两个热门的研究领域。机器学习侧重于基于数据中的模式预测结果，而因果推断旨在理解变量之间的因果关系。机器学习与因果推断之间的关系复杂且多面。机器学习可用于估计因果效应，而因果推断可用于改进机器学习算法。通过结合这两个领域，我们可以更好地理解变量之间的因果关系，并提升基于数据进行预测的能力。然而，这种关系也伴随着诸多挑战。例如，机器学习算法可能无法始终控制所有混杂变量，从而导致因果效应的估计存在偏差。同样，因果推断算法可能无法始终识别变量之间的所有因果关系，从而导致模型不完整或不准确。

本书旨在为读者提供关于机器学习与因果推断之间关系的深入见解。书中深入探讨了因果推断的基础知识、利用机器学习进行因果效应估计、因果推断在可信机器学习中的贡献，以及因果推断在各类机器学习领域中的实际应用。

在第一部分中，**因果推断基础（causal inference preliminary）**全面介绍了因果推断，并阐述了不同类型的因果推断方法，包括随机实验和观察性研究。该部分涵盖了基本概念，如因果性（causality）、潜在结果（potential outcomes）、反事实（counterfactuals）、混杂变量（confounders）、选择偏差（selection bias），以及识别因果效应的关键假设。

第二部分聚焦于**机器学习与因果效应估计（machine learning and causal effect estimation）**。第3章讨论了估计因果效应的基本方法，包括基于匹配（matching-based）、基于树（tree-based）、基于集成（ensemble-based）、基于表示学习（representation learning–based）等方法。该章通过应用实例解释了每种方法的优势与局限性。除了建立在基本假设之上的因果推断方法外，还介绍了尝试放宽某些假设的方法。第4章介绍了图结构数据上因果推断的背景、主要挑战以及估计处理效应的相关方法。尽管先进的机器学习方法在处理效应估计中表现出卓越的性能，但它们也带来了许多新课题和新研究问题。基于因果推断领域的最新研究成果，第5章全面讨论了处理效应估计任务的三个核心组成部分——即处理变量（treatment）、协变量（covariates）和结果变量（outcome）——所面临的挑战与机遇。此外，我们还从多个角度展示了该主题有前景的研究方向。

在第三部分中，我们探讨了**因果推断与可信机器学习（causal inference and trustworthy machine learning）**之间的关系。因果推断是增强机器学习模型可信度的重要工具。它提供了一个理解变量之间因果关系的框架，从而能够提升模型的准确性、透明度、公平性、泛化能力和可解释性。具体而言，第6章提出了一个基于因果性的公平感知机器学习框架，该框架通过指定因果路径集合和观测条件，能够统一多种因果公平性概念。第7章概述了因果解释（causal explanation），并讨论了因果可解释人工智能（causal explainable artificial intelligence）的设计，以帮助理解如何利用因果推断来解释模型。第8章介绍了因果感知的领域泛化（causality-aware domain generalization）方法与传统领域泛化方法的区别，如何以及何时利用因果性来推断不变特征，以及这些方法在视觉、图和文本领域的应用。

在第四部分中，我们介绍了**因果推断在不同机器学习领域中的应用（applications of causal inference in different machine learning domains）**，如图学习（graph learning）、推荐系统（recommendation systems）、计算机视觉（computer vision）、自然语言处理（natural language processing）、时间序列分析（time series analysis）等。因果推断是确定因果关系的过程。理解因果关系对于从数据中构建不同的机器学习模型至关重要。具体而言，第9章讨论了在文本数据上进行因果推断所面临的困难，这些困难源于文本的非结构化和高维特性。此外，该章还广泛综述了面向自然语言处理的因果驱动模型，考察了整合因果性的各种方法，包括干预层面（intervention-level）和反事实层面（counterfactual-level）的去偏技术。第10章介绍了传统推荐系统的基本概念及其由于缺乏因果推理能力而存在的局限性，随后讨论了如何引入不同的因果推断技术来应对这些挑战，重点关注去偏、可解释性提升和泛化能力改进。第11章展示了一个可用于实例依赖的标签噪声学习（instance-dependent label-noise learning）的结构因果模型（structural causal model），该模型能够在噪声计算机视觉数据集中实现更好的分类精度。第12章提出了一个具有可解释注意力模块的因果三重注意力时间序列预测模型（causal triple attention time series forecasting model），该模型利用因果推断去除混杂效应，帮助模型有效利用局部和全局时间信息。第13章正式定义了连续因果效应估计（continual causal effect estimation）问题，并提出了一种连续因果效应表示学习方法，用于基于观测数据估计因果效应，这些数据从非平稳数据分布中增量获得。
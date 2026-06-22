# Machine Learning for Causal Inference

## Editors

Sheng Li

University of Virginia

Charlottesville, VA, USA

Zhixuan Chu

Ant Group

Hangzhou, China

ISBN 978-3-031-35050-4

ISBN 978-3-031-35051-1 (eBook)

https://doi.org/10.1007/978-3-031-35051-1

© The Editor(s) (if applicable) and The Author(s), under exclusive license to Springer Nature Switzerland AG 2023

This work is subject to copyright. All rights are solely and exclusively licensed by the Publisher, whether the whole or part of the material is concerned, specifically the rights of translation, reprinting, reuse of illustrations, recitation, broadcasting, reproduction on microfilms or in any other physical way, and transmission or information storage and retrieval, electronic adaptation, computer software, or by similar or dissimilar methodology now known or hereafter developed.

The use of general descriptive names, registered names, trademarks, service marks, etc. in this publication does not imply, even in the absence of a specific statement, that such names are exempt from the relevant protective laws and regulations and therefore free for general use.

The publisher, the authors, and the editors are safe to assume that the advice and information in this book are believed to be true and accurate at the date of publication. Neither the publisher nor the authors or the editors give a warranty, expressed or implied, with respect to the material contained herein or for any errors or omissions that may have been made. The publisher remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

This Springer imprint is published by the registered company Springer Nature Switzerland AG. The registered company address is: Gewerbestrasse 11, 6330 Cham, Switzerland

Paper in this product is recyclable.

## Preface

Machine learning and causal inference have gained significant attention in both academia and industry for the past decades, but they have been mainly treated as separate research areas. In recent years, some machine learning models (e.g., tree-based models, deep representation learning models, adversarial learning models, and graph neural networks) have been increasingly used for various causal inference problems, such as counterfactual inference, treatment effect estimation, and causal discovery. Moreover, causality has been exploited to assist some challenging machine learning tasks, such as explainability, fairness, and domain generalization. Such a convergence between machine learning and causal inference motivated us to create a book to summarize the recent research progress in this area. We are very fortunate to receive tremendous support from many leading scientists in machine learning and causal inference, who contribute book chapters to give comprehensive reviews of emerging research topics.

This book aims to offer readers insights into the relationship between machine learning and causal inference from multiple perspectives. It delves into topics such as the preliminary of causal inference, the utilization of machine learning for causal effect estimation, the contribution of causal inference in trustworthy machine learning, and the practical applications of causal inference in various domains.

This book consists of 4 parts which are composed of 14 chapters in total.

- Part I gives an overview of this book in Chap. 1 and covers the preliminary of causal inference in Chap. 2.
- Part II focuses on machine learning and causal effect estimation. In particular, Chap. 3 discusses the basic methodologies for estimating causal effects, Chap. 4 introduces causal inference on graphs, and Chap. 5 provides a comprehensive discussion of challenges and opportunities for the three core components of the treatment effect estimation task, i.e., treatment, covariates, and outcome.
- Part III introduces the relationships between causal inference and trustworthy machine learning. Specifically, Chap. 6 presents a causality-based framework for fairness-aware machine learning, Chap. 7 discusses the design of causal

explainable artificial intelligence systems, and Chap. 8 introduces causality-aware domain generalization.

\- Part IV introduces the applications of causal inference and machine learning in different domains. Chapter 9 discusses causal inference for natural language processing (NLP). Chapter 10 discusses how different causal inference techniques can be introduced to address the challenges in recommender systems. Chapter 11 presents a structural causal model that can be leveraged for instance-dependent label-noise learning in computer vision. Chapter 12 proposes a causal triple attention time series forecasting model with interpretable attention modules, which leverages the causal inference to remove the confounding effect. Chapter 13 presents the continual causal inference problem and proposes a new framework for estimating causal effects from incrementally available observational data. Chapter 14 summarizes this book.

Overall, this book provides a comprehensive review of causal inference methodologies, a timely summarization of recent research efforts, and various real-world applications of causal inference, which will benefit readers from different backgrounds, such as advanced undergraduate and graduate students, researchers, lecturers, and practitioners.

Charlottesville, VA, USA

Hangzhou, China

April, 2023

Sheng Li

Zhixuan Chu

## Acknowledgements

Over the last 10 years, causal inference and machine learning have drawn more and more attention. We feel extremely lucky to have the opportunity to speak with top researchers and leading scientists in this field on recent developments and research challenges. We would like to extend our sincere gratitude to our partners and colleagues at the University of Virginia and Ant Group, who inspired us to write this book that serves as a timely summary of recent research progress in the interaction of causal inference and machine learning. We also appreciate the assistance and cooperation of Springer Nature editors Paul Drougas and Arun Siva Shanmugam.

Finally, we would like to thank our families for their support, understanding, and motivation throughout the writing of this book.



## Sheng Li and Zhixuan Chu

## Contributors

Wenqing Chen Sun Yat-sen University, Guangzhou, China

Zhixuan Chu Ant Group, Hangzhou, China

Jing Gao Purdue University, West Lafayette, IN, USA

Yingqiang Ge Rutgers University, New Brunswick, NJ, USA

Mingming Gong University of Melbourne, Parkville, VIC, Australia

Ruocheng Guo Bytedance AI Lab, London, UK

Bo Han Hong Kong Baptist University, Hong Kong, China

Jundong Li University of Virginia, Charlottesville, VA, USA

Sheng Li University of Virginia, Charlottesville, VA, USA

Yaliang Li Alibaba Group, Hangzhou, China

Ruopeng Li Ant Group, Hangzhou, China

Huan Liu Arizona State University, Tempe, AZ, USA

Tongliang Liu The University of Sydney, Camperdown, NSW, Australia

Jing Ma University of Virginia, Charlottesville, VA, USA

Gang Niu RIKEN Center for Advanced Intelligence Project, Tokyo, Japan

Stephen L. Rathbun University of Georgia, Athens, GA, USA

Paras Sheth Arizona State University, Tempe, AZ, USA

Xintao Wu University of Arkansas, Fayetteville, AR, USA

Yongkai Wu Clemson University, Clemson, SC, USA

Shuyuan Xu Rutgers University, New Brunswick, NJ, USA

Liuyi Yao Alibaba Group, Hangzhou, ChinaYu Yao The University of Sydney, Camperdown, NSW, Australia  
Aidong Zhang University of Virginia, Charlottesville, VA, USA  
Kun Zhang Carnegie Mellon University, Pittsburgh, PA, USA  
Lu Zhang University of Arkansas, Fayetteville, AR, USA  
Yongfeng Zhang Rutgers University, New Brunswick, NJ, USA  
Yaochen Zhu University of Virginia, Charlottesville, VA, USA
# CAUSAL INFERENCE IN STATISTICS

## CAUSAL INFERENCE IN STATISTICS

A PRIMER

Judea Pearl  
Computer Science and Statistics, University of California, Los Angeles, USA

Madelyn Glymour  
Philosophy, Carnegie Mellon University, Pittsburgh, USA

Nicholas P. Jewell  
Biostatistics and Statistics, University of California, Berkeley, USA

WILEY

This edition first published 2016  
© 2016 John Wiley & Sons Ltd

Registered office  
John Wiley & Sons Ltd, The Atrium, Southern Gate, Chichester, West Sussex, PO19 8SQ, United Kingdom

For details of our global editorial offices, for customer services and for information about how to apply for permission to reuse the copyright material in this book please see our website at www.wiley.com.

The right of the author to be identified as the author of this work has been asserted in accordance with the Copyright, Designs and Patents Act 1988.

All rights reserved. No part of this publication may be reproduced, stored in a retrieval system, or transmitted, in any form or by any means, electronic, mechanical, photocopying, recording or otherwise, except as permitted by the UK Copyright, Designs and Patents Act 1988, without the prior permission of the publisher.

Wiley also publishes its books in a variety of electronic formats. Some content that appears in print may not be available in electronic books.

Designations used by companies to distinguish their products are often claimed as trademarks. All brand names and product names used in this book are trade names, service marks, trademarks or registered trademarks of their respective owners. The publisher is not associated with any product or vendor mentioned in this book.

Limit of Liability/Disclaimer of Warranty: While the publisher and author have used their best efforts in preparing this book, they make no representations or warranties with respect to the accuracy or completeness of the contents of this book and specifically disclaim any implied warranties of merchantability or fitness for a particular purpose. It is sold on the understanding that the publisher is not engaged in rendering professional services and neither the publisher nor the author shall be liable for damages arising herefrom. If professional advice or other expert assistance is required, the services of a competent professional should be sought.

Library of Congress Cataloging-in-Publication Data applied for  
ISBN: 9781119186847

A catalogue record for this book is available from the British Library.

Cover Image: © gmaydos/Getty

Typeset in 10/12pt TimesLTStd by SPi Global, Chennai, India


To my wife, Ruth, my greatest mentor.  
– Judea Pearl

To my parents, who are the causes of me.  
– Madelyn Glymour

To Debra and Britta, who inspire me every day.  
– Nicholas P. Jewell

# About the Authors

Judea Pearl is Professor of Computer Science and Statistics at the University of California, Los Angeles, where he directs the Cognitive Systems Laboratory and conducts research in artificial intelligence, causal inference and philosophy of science. He is a Co-Founder and Editor of the Journal of Causal Inference and the author of three landmark books in inference-related areas. His latest book, _Causality: Models, Reasoning and Inference_ (Cambridge, 2000, 2009), has introduced many of the methods used in modern causal analysis. It won the Lakatosh Award from the London School of Economics and is cited by more than 13,000 scientific publications.

His groundbreaking book, _Causality: Models, Reasoning and Inference_ (Cambridge, 2000, 2009), has introduced many of the methods used in modern causal analysis. It won the Lakatos Award from the London School of Economics and is cited by more than 15,000 scientific publications. His latest book, _The Book of Why: The New Science of Cause and Effect_ (with Dana Mackenzie, Basic Books, 2018), describes the impacts of causal inference to the general public.

Madelyn Glymour is a data analyst at Carnegie Mellon University, and a science writer and editor for the Cognitive Systems Laboratory at UCLA. Her interests lie in causal discovery and in the art of making complex concepts accessible to broad audiences.

Nicholas P. Jewell is Professor of Biostatistics and Statistics at the University of California, Berkeley. He has held various academic and administrative positions at Berkeley since his arrival in 1981, most notably serving as Vice Provost from 1994 to 2000. He has also held academic appointments at the University of Edinburgh, Oxford University, the London School of Hygiene and Tropical Medicine, and at the University of Kyoto. In 2007, he was a Fellow at the Rockefeller Foundation Bellagio Study Center in Italy.

Jewell is a Fellow of the American Statistical Association, the Institute of Mathematical Statistics, and the American Association for the Advancement of Science (AAAS). He is a past winner of the Snedecor Award and the Marvin Zelen Leadership Award in Statistical Science from Harvard University. He is currently the Editor of the _Journal of the American Statistical Association – Theory & Methods_, and Chair of the Statistics Section of AAAS. His research focuses on the application of statistical methods to infectious and chronic disease epidemiology, the assessment of drug safety, time-to-event analyses, and human rights.

## Preface

When attempting to make sense of data, statisticians are invariably motivated by causal questions. For example:

- “How effective is a given treatment in preventing a disease?”
- “Can one estimate obesity-related medical costs?”
- “Could government actions have prevented the financial crisis of 2008?”
- “Can hiring records prove an employer guilty of sex discrimination?”

The peculiar nature of these questions is that they cannot be answered, or even articulated, in the traditional language of statistics. In fact, only recently has science acquired a mathematical language we can use to express such questions, with accompanying tools to allow us to answer them from data.

The development of these tools has spawned a revolution in the way causality is treated in statistics and in many of its satellite disciplines, especially in the social and biomedical sciences. For example, in the technical program of the 2003 Joint Statistical Meeting in San Francisco, there were only 13 papers presented with the word “cause” or “causal” in their titles; the number of such papers exceeded 100 by the Boston meeting in 2014. These numbers represent a transformative shift of focus in statistics research, accompanied by unprecedented excitement about the new problems and challenges that are opening themselves to statistical analysis. Harvard’s political science professor Gary King puts this revolution in historical perspective: “More has been learned about causal inference in the last few decades than the sum total of everything that had been learned about it in all prior recorded history.”

Yet this excitement remains barely seen among statistics educators, and is essentially absent from statistics textbooks, especially at the introductory level. The reasons for this disparity is deeply rooted in the tradition of statistical education and in how most statisticians view the role of statistical inference.

In Ronald Fisher’s influential manifesto, he pronounced that “the object of statistical methods is the reduction of data” (Fisher 1922). In keeping with that aim, the traditional task of making sense of data, often referred to generically as “inference,” became that of finding a parsimonious mathematical description of the joint distribution of a set of variables of interest, or of specific parameters of such a distribution. This general strategy for inference is extremely familiar not just to statistical researchers and data scientists, but to anyone who has taken a basic course in statistics. In fact, many excellent introductory books describe smart and effective ways to extract the maximum amount of information possible from the available data. These books take the novice reader from experimental design to parameter estimation and hypothesis testing in great detail. Yet the aim of these techniques are invariably the description of data, not of the process responsible for the data. Most statistics books do not even have the word “causal” or “causation” in the index.

Yet the fundamental question at the core of a great deal of statistical inference is causal: do changes in one variable cause changes in another, and if so, how much change do they cause? In avoiding these questions, introductory treatments of statistical inference often fail even to discuss whether the parameters that are being estimated are the relevant quantities to assess when interest lies in cause and effects.

The best that most introductory textbooks do is this: First, state the often-quoted aphorism that “association does not imply causation,” give a short explanation of confounding and how “lurking variables” can lead to a misinterpretation of an apparent relationship between two variables of interest. Further, the boldest of those texts pose the principal question: “How can a causal link between x and y be established?” and answer it with the long-standing “gold standard” approach of resorting to randomized experiment, an approach that to this day remains the cornerstone of the drug approval process in the United States and elsewhere.

However, given that most causal questions cannot be addressed through random experimentation, students and instructors are left to wonder if there is anything that can be said with any reasonable confidence in the absence of pure randomness.

In short, by avoiding discussion of causal models and causal parameters, introductory textbooks provide readers with no basis for understanding how statistical techniques address scientific questions of causality.

It is the intent of this primer to fill this gnawing gap and to assist teachers and students of elementary statistics in tackling the causal questions that surround almost any nonexperimental study in the natural and social sciences. We focus here on simple and natural methods to define causal parameters that we wish to understand and to show what assumptions are necessary for us to estimate these parameters in observational studies. We also show that these assumptions can be expressed mathematically and transparently and that simple mathematical machinery is available for translating these assumptions into estimable causal quantities, such as the effects of treatments and policy interventions, to identify their testable implications.

Our goal stops there for the moment; we do not address in any detail the optimal parameter estimation procedures that use the data to produce effective statistical estimates and their associated levels of uncertainty. However, those ideas—some of which are relatively advanced—are covered extensively in the growing literature on causal inference. We thus hope that this short text can be used in conjunction with standard introductory statistics textbooks like the ones we have described to show how statistical models and inference can easily go hand in hand with a thorough understanding of causation.

It is our strong belief that if one wants to move beyond mere description, statistical inference cannot be effectively carried out without thinking carefully about causal questions, and without leveraging the simple yet powerful tools that modern analysis has developed to answer such questions. It is also our experience that thinking causally leads to a much more exciting and satisfying approach to both the simplest and most complex statistical data analyses. This is not a new observation. Virgil said it much more succinctly than we in 29 BC:

> “Felix, qui potuit rerum cognoscere causas” (Virgil 29 BC)
> (Lucky is he who has been able to understand the causes of things)

The book is organized in four chapters.

**Chapter 1** provides the basic statistical, probabilistic, and graphical concepts that readers will need to understand the rest of the book. It also introduces the fundamental concepts of causality, including the causal model, and explains through examples how the model can convey information that pure data are unable to provide.

**Chapter 2** explains how causal models are reflected in data, through patterns of statistical dependencies. It explains how to determine whether a data set complies with a given causal model, and briefly discusses how one might search for models that explain a given data set.

**Chapter 3** is concerned with how to make predictions using causal models, with a particular emphasis on predicting the outcome of a policy intervention. Here we introduce techniques of reducing confounding bias using adjustment for covariates, as well as inverse probability weighing. This chapter also covers mediation analysis and contains an in-depth look at how the causal methods discussed thus far work in a linear system. Key to these methods is the fundamental distinction between regression coefficients and structural parameters, and how students should use both to predict causal effects in linear models.

**Chapter 4** introduces the concept of counterfactuals—what would have happened, had we chosen differently at a point in the past—and discusses how we can compute them, estimate their probabilities, and what practical questions we can answer using them. This chapter is somewhat advanced, compared to its predecessors, primarily due to the novelty of the notation and the hypothetical nature of the questions asked. However, the

# Acknowledgments

This book is an outgrowth of a graduate course on causal inference that the first author has been teaching at UCLA in the past 20 years. It owes many of its tools and examples to former members of the Cognitive Systems Laboratory who participated in the development of this material, both as researchers and as teaching assistants. These include Alex Balke, David Chickering, David Galles, Dan Geiger, Moises Goldszmidt, Jin Kim, George Rebane, Ilya Shpitser, Jin Tian, and Thomas Verma.

We are indebted to many colleagues from whom we have learned much about causal problems, their solutions, and how to present them to general audiences. These include Clark and Maria Glymour, for providing patient ears and sound advice on matters of both causation and writing, Felix Elwert and Tyler VanderWeele for insightful comments on an earlier version of the manuscript, and the many visitors and discussants to the UCLA Causality blog who kept the discussion lively, occasionally controversial, but never boring (causality.cs.ucla.edu/blog).

Elias Bareinboim, Bryant Chen, Andrew Forney, Ang Li, Karthika Mohan, reviewed the text for accuracy and transparency. Ang and Andrew also wrote solutions to the study questions, which are available to instructors from the publisher, see <http://bayes.cs.ucla.edu/PRIMER/CIS-Manual-PUBLIC.pdf>.

The manuscript was most diligently typed, processed, illustrated, and proofed by Kaoru Mulvihill at UCLA. Debbie Jupe and Heather Kay at Wiley deserve much credit for recognizing and convincing us that a book of this scope is badly needed in the field, and for encouraging us throughout the production process.

Finally, the National Science Foundation and the Office of Naval Research deserve acknowledgment for faithfully and consistently sponsoring the research that led to these results, with special thanks to Behzad Kamgar-Parsi.

# List of Figures

- **Figure 1.1** Results of the exercise–cholesterol study, segregated by age 3
- **Figure 1.2** Results of the exercise–cholesterol study, unsegregated. The data points are identical to those of Figure 1.1, except the boundaries between the various age groups are not shown 4
- **Figure 1.3** Scatter plot of the results in Table 1.6, with the value of Die 1 on the x-axis and the sum of the two dice rolls on the y-axis 21
- **Figure 1.4** Scatter plot of the results in Table 1.6, with the value of Die 1 on the x-axis and the sum of the two dice rolls on the y-axis. The dotted line represents the line of best fit based on the data. The solid line represents the line of best fit we would expect in the population 21
- **Figure 1.5** An undirected graph in which nodes X and Y are adjacent and nodes Y and Z are adjacent but not X and Z 25
- **Figure 1.6** A directed graph in which node A is a parent of B and B is a parent of C 25
- **Figure 1.7** (a) Showing acyclic graph and (b) cyclic graph 26
- **Figure 1.8** A directed graph used in Study question 1.4.1 26
- **Figure 1.9** The graphical model of SCM 1.5.1, with X indicating years of schooling, Y indicating years of employment, and Z indicating salary 27
- **Figure 1.10** Model showing an unobserved syndrome, Z, affecting both treatment (X) and outcome (Y) 31
- **Figure 2.1** The graphical model of SCMs 2.2.1–2.2.3 37
- **Figure 2.2** The graphical model of SCMs 2.2.5 and 2.2.6 39
- **Figure 2.3** A simple collider 41
- **Figure 2.4** A simple collider, Z, with one child, W, representing the scenario from Table 2.3, with X representing one coin flip, Y representing the second coin flip, Z representing a bell that rings if either X or Y is heads, and W representing an unreliable witness who reports on whether or not the bell has rung 44
- **Figure 2.5** A directed graph for demonstrating conditional independence (error terms are not shown explicitly) 45
- **Figure 2.6** A directed graph in which P is a descendant of a collider 45
- **Figure 2.7** A graphical model containing a collider with child and a fork 47
- **Figure 2.8** The model from Figure 2.7 with an additional forked path between Z and Y 48
- **Figure 2.9** A causal graph used in study question 2.4.1, all U terms (not shown) are assumed independent 49
- **Figure 3.1** A graphical model representing the relationship between temperature (Z), ice cream sales (X), and crime rates (Y) 54
- **Figure 3.2** A graphical model representing an intervention on the model in Figure 3.1 that lowers ice cream sales 54
- **Figure 3.3** A graphical model representing the effects of a new drug, with Z representing gender, X standing for drug usage, and Y standing for recovery 55
- **Figure 3.4** A modified graphical model representing an intervention on the model in Figure 3.3 that sets drug usage in the population, and results in the manipulated probability Pm 56
- **Figure 3.5** A graphical model representing the effects of a new drug, with X representing drug usage, Y representing recovery, and Z representing blood pressure (measured at the end of the study). Exogenous variables are not shown in the graph, implying they are mutually independent 58
- **Figure 3.6** A graphical model representing the relationship between a new drug (X), recovery (Y), weight (W), and an unmeasured variable Z (socioeconomic status) 62
- **Figure 3.7** A graphical model in which the backdoor criterion requires that we condition on a collider (Z) in order to ascertain the effect of X on Y 63
- **Figure 3.8** Causal graph used to illustrate the backdoor criterion in the following study questions 64
- **Figure 3.9** Scatter plot with students’ initial weights on the x-axis and final weights on the y-axis. The vertical line indicates students whose initial weights are the same, and whose final wights are higher (on average) for plan B compared with plan A 65
- **Figure 3.10** A graphical model representing the relationships between smoking (X) and lung cancer (Y), with unobserved confounder (U) and a mediating variable Z 66
- **Figure 3.11** A graphical model representing the relationship between gender, qualifications, and hiring 76
- **Figure 3.12** A graphical model representing the relationship between gender, qualifications, and hiring, with socioeconomic status as a mediator between qualifications and hiring 77
- **Figure 3.13** A graphical model illustrating the relationship between path coefficients and total effects 82
- **Figure 3.14** A graphical model in which X has no direct effect on Y, but a total effect that is determined by adjusting for T 83
- **Figure 3.15** A graphical model in which X has direct effect $\alpha$ on Y 84
- **Figure 3.16** By removing the direct edge from X to Y and finding the set of variables {W} that d-separate them, we find the variables we need to adjust for to determine the direct effect of X on Y 85
- **Figure 3.17** A graphical model in which we cannot find the direct effect of X on Y via adjustment, because the dashed double-arrow arc represents the presence of a backdoor path between X and Y, consisting of unmeasured variables. In this case, Z is an instrument with regard to the effect of X on Y that enables the identification of $\alpha$ 85
- **Figure 3.18** Graph corresponding to Model 3.1 in Study question 3.8.1 86
- **Figure 4.1** A model depicting the effect of Encouragement (X) on student’s score 94
- **Figure 4.2** Answering a counterfactual question about a specific student’s score, predicated on the assumption that homework would have increased to H = 2 95
- **Figure 4.3** A model representing Eq. (4.7), illustrating the causal relations between college education (X), skills (Z), and salary (Y) 99
- **Figure 4.4** Illustrating the graphical reading of counterfactuals. (a) The original model. (b) The modified model $M_x$ in which the node labeled $Y_x$ represents the potential outcome Y predicated on X = x 102
- **Figure 4.5** (a) Showing how probabilities of necessity (PN) are bounded, as a function of the excess risk ratio (ERR) and the confounding factor

# About the Companion Website

This book is accompanied by a companion website: [www.wiley.com/go/Pearl/Causality](http://www.wiley.com/go/Pearl/Causality)


# CAUSALITY

SECOND EDITION

JUDEA PEARL

## Models, Reasoning, and Inference Second Edition

Written by one of the preeminent researchers in the field, this book provides a comprehensive exposition of modern analysis of causation. It shows how causality has grown from a nebulous concept into a mathematical theory with significant applications in the fields of statistics, artificial intelligence, economics, philosophy, cognitive science, and the health and social sciences.

Judea Pearl presents a comprehensive theory of causality which unifies the probabilistic, manipulative, counterfactual, and structural approaches to causation and offers simple mathematical tools for studying the relationships between causal connections and statistical associations. The book opens the way for including causal analysis in the standard curricula of statistics, artificial intelligence, business, epidemiology, social sciences, and economics. Students in these fields will find natural models, simple inferential procedures, and precise mathematical definitions of causal concepts that traditional texts have evaded or made unduly complicated.

The first edition of _Causality_ has led to a paradigmatic change in the way that causality is treated in statistics, philosophy, computer science, social science, and economics. Cited in more than 2,800 scientific publications, it continues to liberate scientists from the traditional molds of statistical thinking. In this revised edition, Pearl elucidates thorny issues, answers readers’ questions, and offers a panoramic view of recent advances in this field of research.

_Causality_ will be of interest to students and professionals in a wide variety of fields. Anyone who wishes to elucidate meaningful relationships from data, predict effects of actions and policies, assess explanations of reported events, or form theories of causal understanding and causal speech will find this book stimulating and invaluable.

Judea Pearl is professor of computer science and statistics at the University of California, Los Angeles, where he directs the Cognitive Systems Laboratory and conducts research in artificial intelligence, human reasoning, and philosophy of science. The author of _Heuristics and Probabilistic Reasoning_, he is a member of the National Academy of Engineering and a Founding Fellow of the American Association for Artificial Intelligence. Dr. Pearl is the recipient of the IJCAI Research Excellence Award for 1999, the London School of Economics Lakatos Award for 2001, and the ACM Alan Newell Award for 2004. In 2008, he received the Benjamin Franklin Medal for computer and cognitive science from the Franklin Institute.

## Commendation for the First Edition

> “Judea Pearl’s previous book, _Probabilistic Reasoning in Intelligent Systems_, was arguably the most influential book in Artificial Intelligence in the past decade, setting the stage for much of the current activity in probabilistic reasoning. In this book, Pearl turns his attention to causality, boldly arguing for the primacy of a notion long ignored in statistics and misunderstood and mistrusted in other disciplines, from physics to economics. He demystifies the notion, clarifies the basic concepts in terms of graphical models, and explains the source of many misunderstandings. This book should prove invaluable to researchers in artificial intelligence, statistics, economics, epidemiology, and philosophy, and, indeed, all those interested in the fundamental notion of causality. It may well prove to be one of the most influential books of the next decade.”
>
> — Joseph Halpern, Computer Science Department, Cornell University

> “This lucidly written book is full of inspiration and novel ideas that bring clarity to areas where confusion has prevailed, in particular concerning causal interpretation of structural equation systems, but also on concepts such as counterfactual reasoning and the general relation between causal thinking and graphical models. Finally the world can get a coherent exposition of these ideas that Judea Pearl has developed over a number of years and presented in a flurry of controversial yet illuminating articles.”
>
> — Steffen L. Lauritzen, Department of Mathematics, Aalborg University

> “Judea Pearl’s new book, _Causality: Models, Reasoning, and Inference_, is an outstanding contribution to the causality literature. It will be especially useful to students and practitioners of economics interested in policy analysis.”
>
> — Halbert White, Professor of Economics, University of California, San Diego

> “This book fulfills a long-standing need for a rigorous yet accessible treatise on the mathematics of causal inference. Judea Pearl has done a masterful job of describing the most important approaches and displaying their underlying logical unity. The book deserves to be read by all statisticians and scientists who use nonexperimental data to study causation, and would serve well as a graduate or advanced undergraduate course text.”
>
> — Sander Greenland, School of Public Health, University of California, Los Angeles

> “Judea Pearl has written an account of recent advances in the modeling of probability and cause, substantial parts of which are due to him and his co-workers. This is essential reading for anyone interested in causality.”
>
> — Brian Skyrms, Department of Philosophy, University of California, Irvine

# CAUSALITY Models, Reasoning, and Inference Second Edition

Judea Pearl  
University of California, Los Angeles

> Development of Western science is based on two great achievements: the invention of the formal logical system (in Euclidean geometry) by the Greek philosophers, and the discovery of the possibility to find out causal relationships by systematic experiment (during the Renaissance).  
> — Albert Einstein (1953)

**TO DANNY**  
 **AND THE GLOWING AUDACITY OF GOODNESS**

## Preface to the First Edition

The central aim of many studies in the physical, behavioral, social, and biological sciences is the elucidation of cause–effect relationships among variables or events. However, the appropriate methodology for extracting such relationships from data – or even from theories – has been fiercely debated.

The two fundamental questions of causality are: (1) What empirical evidence is required for legitimate inference of cause–effect relationships? (2) Given that we are willing to accept causal information about a phenomenon, what inferences can we draw from such information, and how? These questions have been without satisfactory answers in part because we have not had a clear semantics for causal claims and in part because we have not had effective mathematical tools for casting causal questions or deriving causal answers.

In the last decade, owing partly to advances in graphical models, causality has undergone a major transformation: from a concept shrouded in mystery into a mathematical object with well-defined semantics and well-founded logic. Paradoxes and controversies have been resolved, slippery concepts have been explicated, and practical problems relying on causal information that long were regarded as either metaphysical or unmanageable can now be solved using elementary mathematics. Put simply, causality has been mathematized.

This book provides a systematic account of this causal transformation, addressed primarily to readers in the fields of statistics, artificial intelligence, philosophy, cognitive science, and the health and social sciences. Following a description of the conceptual and mathematical advances in causal inference, the book emphasizes practical methods for elucidating potentially causal relationships from data, deriving causal relationships from combinations of knowledge and data, predicting the effects of actions and policies, evaluating explanations for observed events and scenarios, and – more generally – identifying and explicating the assumptions needed for substantiating causal claims.

Ten years ago, when I began writing _Probabilistic Reasoning in Intelligent Systems_ (1988), I was working within the empiricist tradition. In this tradition, probabilistic relationships constitute the foundations of human knowledge, whereas causality simply provides useful ways of abbreviating and organizing intricate patterns of probabilistic relationships. Today, my view is quite different. I now take causal relationships to be the fundamental building blocks both of physical reality and of human understanding of that reality, and I regard probabilistic relationships as but the surface phenomena of the causal machinery that underlies and propels our understanding of the world.

Accordingly, I see no greater impediment to scientific progress than the prevailing practice of focusing all of our mathematical resources on probabilistic and statistical inferences while leaving causal considerations to the mercy of intuition and good judgment. Thus I have tried in this book to present mathematical tools that handle causal relationships side by side with probabilistic relationships. The prerequisites are startlingly simple, the results embarrassingly straightforward. No more than basic skills in probability theory and some familiarity with graphs are needed for the reader to begin solving causal problems that are too complex for the unaided intellect. Using simple extensions of probability calculus, the reader will be able to determine mathematically what effects an intervention might have, what measurements are appropriate for control of confounding, how to exploit measurements that lie on the causal pathways, how to trade one set of measurements for another, and how to estimate the probability that one event was the actual cause of another.

Expert knowledge of logic and probability is nowhere assumed in this book, but some general knowledge in these areas is beneficial. Thus, Chapter 1 includes a summary of the elementary background in probability theory and graph notation needed for the understanding of this book, together with an outline of the developments of the last decade in graphical models and causal diagrams. This chapter describes the basic paradigms, defines the major problems, and points readers to the chapters that provide solutions to those problems.

Subsequent chapters include introductions that serve both to orient the reader and to facilitate skipping; they indicate safe detours around mathematically advanced topics, specific applications, and other explorations of interest primarily to the specialist.

The sequence of discussion follows more or less the chronological order by which our team at UCLA has tackled these topics, thus re-creating for the reader some of our excitement that accompanied these developments. Following the introductory chapter (Chapter 1), we start with the hardest questions of how one can go about discovering cause–effect relationships in raw data (Chapter 2) and what guarantees one can give to ensure the validity of the relationships thus discovered. We then proceed to questions of identifiability – namely, predicting the direct and indirect effects of actions and policies from a combination of data and fragmentary knowledge of where causal relationships might operate (Chapters 3 and 4). The implications of these findings for the social and health sciences are then discussed in Chapters 5 and 6 (respectively), where we examine the concepts of structural equations and confounding. Chapter 7 offers a formal theory of counterfactuals and structural models, followed by a discussion and a unification of related approaches in philosophy, statistics, and economics. The applications of counterfactual analysis are then pursued in Chapters 8–10, where we develop methods of bounding causal relationships and illustrate applications to imperfect experiments, legal responsibility, and the probability of necessary, sufficient, and single-event causation. We end this book (Epilogue) with a transcript of a public lecture that I presented at UCLA, which provides a gentle introduction to the historical and conceptual aspects of causation.

Readers who wish to be first introduced to the nonmathematical aspects of causation are advised to start with the Epilogue and then to sweep through the other historical/conceptual parts of the book: Sections 1.1.1, 3.3.3, 4.5.3, 5.1, 5.4.1, 6.1, 7.2, 7.4, 7.5, 8.3, 9.1, 9.3, and 10.1. More formally driven readers, who may be anxious to delve directly into the mathematical aspects and computational tools, are advised to start with Section 7.1 and then to proceed as follows for tool building: Section 1.2, Chapter 3, Sections 4.2–4.4, Sections 5.2–5.3, Sections 6.2–6.3, Section 7.3, and Chapters 8–10.

I owe a great debt to many people who assisted me with this work. First, I would like to thank the members of the Cognitive Systems Laboratory at UCLA, whose work and ideas formed the basis of many of these sections: Alex Balke, Blai Bonet, David Chickering, Adnan Darwiche, Rina Dechter, David Galles, Hector Geffner, Dan Geiger, Moisés Goldszmidt, Jin Kim, Jin Tian, and Thomas Verma. Tom and Dan have proven some of the most basic theorems in causal graphs; Hector, Adnan, and Moisés were responsible for keeping me in line with the logicist approach to actions and change; and Alex and David have taught me that counterfactuals are simpler than the name may imply.

My academic and professional colleagues have been very generous with their time and ideas as I began ploughing the peaceful territories of statistics, economics, epidemiology, philosophy, and the social sciences. My mentors–listeners in statistics have been Phil Dawid, Steffen Lauritzen, Don Rubin, Art Dempster, David Freedman, and David Cox. In economics, I have benefited from many discussions with John Aldrich, Kevin Hoover, James Heckman, Ed Learner, and Herbert Simon. My forays into epidemiology resulted in a most fortunate and productive collaboration with Sander Greenland and James Robins. Philosophical debates with James Woodward, Nancy Cartwright, Brian Skyrms, Clark Glymour, and Peter Spirtes have sharpened my thinking of causality in and outside philosophy. Finally, in artificial intelligence, I have benefited from discussions with and the encouragement of Nils Nilsson, Ray Reiter, Don Michie, Joe Halpern, and David Heckerman.

The National Science Foundation deserves acknowledgment for consistently and faithfully sponsoring the research that led to these results, with special thanks to H. Moraff, Y. T. Chien, and Larry Reeker. Other sponsors include Abraham Waksman of the Air Force Office of Scientific Research, Michael Shneier of the Office of Naval Research, the California MICRO Program, Northrop Corporation, Rockwell International, Hewlett-Packard, and Microsoft.

I would like to thank Academic Press and Morgan Kaufmann Publishers for their kind permission to reprint selected portions of previously published material. Chapter 3 includes material reprinted from _Biometrika_, vol. 82, Judea Pearl, “Causal Diagrams for Empirical Research,” pp. 669–710, Copyright 1995, with permission from Oxford University Press. Chapter 5 includes material reprinted from _Sociological Methods and Research_, vol. 27, Judea Pearl, “Graphs, Causality, and Structural Equation Models,” pp. 226–84, Copyright 1998, with permission from Sage Publications, Inc. Chapter 7 includes material reprinted from _Foundations of Science_, vol. 1, David Galles and Judea Pearl, “An Axiomatic Characterization of Causal Counterfactuals,” pp. 151–82, Copyright 1998, with permission from Kluwer Academic Publishers. Chapter 7 also includes material reprinted from _Artificial Intelligence_, vol. 97, David Galles and Judea Pearl, “Axioms of Causal Relevance,” pp. 9–43, Copyright 1997, with permission from Elsevier Science. Chapter 8 includes material modified from _Journal of the American Statistical Association_, vol. 92, Alexander Balke and Judea Pearl, “Bounds on Treatment Effects from Studies with Imperfect Compliance,” pp. 1171–6

## Preface to the Second Edition

It has been more than eight years since the first edition of this book presented readers with the friendly face of causation and her mathematical artistry. The popular reception of the book and the rapid expansion of the structural theory of causation call for a new edition to assist causation through her second transformation – from a demystified wonder to a commonplace tool in research and education.

This edition:

1. Provides technical corrections, updates, and clarifications in all ten chapters of the original book.
2. Adds summaries of new developments and annotated bibliographical references at the end of each chapter.
3. Elucidates subtle issues that readers and reviewers have found perplexing, objectionable, or in need of elaboration.

These are assembled into an entirely new chapter (11) which, I sincerely hope, clears the province of causal thinking from the last traces of controversy.

Teachers who have taught from this book before should find the revised edition more lucid and palatable, while those who have waited for scouts to carve the path will find the road paved and tested. Supplementary educational material, slides, tutorials, and homework can be found on my website: [http://www.cs.ucla.edu/~judea/](http://www.cs.ucla.edu/~judea/).

My main audience remain the students: students of statistics who wonder why instructors are reluctant to discuss causality in class; students of epidemiology who wonder why elementary concepts such as confounding are so hard to define mathematically; students of economics and social science who question the meaning of the parameters they estimate; and, naturally, students of artificial intelligence and cognitive science, who write programs and theories for knowledge discovery, causal explanations, and causal speech.

I hope that each of these groups will find the unified theory of causation presented in this book to be both inspirational and instrumental in tackling new challenges in their respective fields.

J. P.  
Los Angeles  
July 2008

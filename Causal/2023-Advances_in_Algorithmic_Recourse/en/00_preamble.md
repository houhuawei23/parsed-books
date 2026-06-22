Advances in Algorithmic Recourse: Ensuring Causal Consistency, Fairness, & Robustness

A thesis submitted to attain the degree of

Doctor of sciences of ETH Zurich

(Dr. sc. ETH Zurich)

presented by

Amir-Hossein Karimi

M.Math in Computer Science, University of Waterloo

born on 22.06.1992

citizen of Iran, Canada

accepted on the recommendation of

Prof. Dr. Bernhard Schölkopf (ETH Zurich),

Prof. Dr. Isabel Valera (Saarland University),

Prof. Dr. Benjamin Grewe (ETH Zurich),


Advances in Algorithmic Recourse: Ensuring Causal Consistency, Fairness, & RobustnessMachine learning is progressively being employed to guide critical decisions in sensitive contexts where decisions have profound effects on individuals’ lives. Examples include pre-trial bail, loan approval, resume filtering, or prescription of significant medication. In such contexts, it is crucial for models to be accurate, robust, and simultaneously uphold socially relevant values such as fairness, privacy, accountability, and explainability. These aspects significantly influence the acceptance and impact of these technologies.

In this dissertation, I focus specifically on the task of enabling and facilitating algorithmic recourse. This involves providing individuals with comprehensible explanations and recommendations on the most effective (efficient and ideally low-cost) means to recover from unfavorable decisions made by an automated system. The following research questions are addressed:

q1. how can we provide recourse to affected individuals across various settings? In response to this question, I propose a novel algorithm for generating model-agnostic counterfactual explanations (MACE) built upon standard theory and tools from formal verification. This approach overcomes the limitations of previous strategies and supports model, datatype, and distance agnostic counterfactual explanations. It also provides plausible and diverse counterfactuals for any individual, and at provably optimal distances.

q2. what actionable insight can be derived from a counterfactual explanation? I argue that explanations must enable people to act rather than merely understand. Using counterexamples and the theory of structural causal models (SCM), I demonstrate that actionable recommendations cannot generally be inferred from counterfactual explanations. I propose new optimization problems for generating minimal consequential interventions (MINT), providing exact recourse under knowledge of the true SCM and probabilistic recourse when only the causal graph is available.

q3. how does providing recourse explanations/recommendations influence other stakeholders? In the third part of this dissertation, I contend that providing individuals with the right of recourse should be considered within the broader context of its impact on other stakeholders and other desirable properties like fairness, privacy, and model/IP security. I define and propose a solution for offering fair recourse, and discuss how uncertainties and non-stationarities can affect the provided recourse. I explore robust recourse strategies and discuss potential changes to classifier or data generation processes that could facilitate fair/robust recourse.

In conclusion, this dissertation offers a roadmap for future research directions, challenges existing assumptions, and broadens the domain of recourse beyond supervised learning.

Maschinelles Lernen wird immer häufiger eingesetzt, um entscheidende Entscheidungen in sensiblen Kontexten zu steuern, in denen die Entscheidungen tiefgreifende Auswirkungen auf das Leben von Einzelpersonen haben. Beispiele hierfür sind die Entscheidung über Kautionen vor Gericht, die Genehmigung von Darlehen, das Filtern von Lebensläufen oder die Verschreibung lebensverändernder Medikamente. In solchen Situationen ist es unerlässlich, dass die Modelle präzise und robust sind und gleichzeitig soziale Werte wie Fairness, Privatsphäre, Rechenschaftspflicht und Erklärbarkeit einhalten. Diese Werte beeinflussen masgeblich die Akzeptanz und Wirkung dieser Technologien.

In dieser Dissertation konzentriere ich mich insbesondere auf die Aufgabe, Algorithmischen Recourse zu ermöglichen und zu fördern. Dies beinhaltet, den betroffenen Personen verständliche Erklärungen und Empfehlungen darüber zu geben, wie sie am effektivsten (effizient und idealerweise kostengünstig) von ungünstigen Entscheidungen, die von einem automatisierten System getroffen wurden, abrücken können. Die in dieser Dissertation behandelten Forschungsfragen sind:

q1. wie können wir den betroffenen personen recourse in unterschiedlichen situationen bieten? Zur Beantwortung dieser Frage schlage ich einen neuen Algorithmus zur Erzeugung von modellagnostischen kontrafaktischen Erklärungen (MACE) vor, der auf Standardtheorie und -werkzeugen der formalen Verifikation basiert. Dieser Ansatz überwindet die Einschränkungen früherer Strategien und ist modell-, datentypund distanzagnostisch. Er kann plausible und vielfältige kontrafaktische Erklärungen für jede Person erzeugen und das auf nachweislich optimalen Distanzen.

q2. welche handlungsfähigen erkenntnisse können aus einer kontrafaktischen erklärung gewonnen werden? Ich argumentiere, dass Erklärungen Menschen zum Handeln anregen sollten, anstatt nur zum Verstehen. Mit Hilfe von Gegenbeispielen und der Theorie der strukturellen Kausalmodelle (SCM) zeige ich, dass handlungsrelevante Empfehlungen im Allgemeinen nicht aus kontrafaktischen Erklärungen abgeleitet werden können. Ich formuliere neue Optimierungsprobleme zur direkten Erzeugung minimaler konsequenzieller Interventionen (MINT), die einen genauenRecourse unter Kenntnis des wahren SCM und einen probabilistischen Recourse bieten, wenn nur das kausale Diagramm vorhanden ist.

q3. wie wirkt sich das anbieten von recourse-erklärungen/– empfehlungen auf andere stakeholder aus? Im dritten Teil dieser Dissertation argumentiere ich, dass das Bereitstellen von Recourse für Einzelpersonen im gröseren Zusammenhang seiner Auswirkungen auf andere Stakeholder und zusätzliche wünschenswerte Eigenschaften wie Fairness, Privatsphäre und Modell-/IP-Sicherheit betrachtet werden sollte. Ich definiere und biete eine Lösung für die Bereitstellung von fairem Recourse an und diskutiere, wie Unsicherheiten und Nicht-Stationaritäten den angebotenen Recourse beeinflussen können. Ich untersuche Strategien für robusten Recourse und diskutiere mögliche änderungen an Klassifizierungsprozessen oder Daten-Generierungsprozessen, die einen fairen/robusten Recourse unterstützen könnten.

Zum Abschluss bietet diese Dissertation eine Orientierung für zukünftige Forschungsrichtungen, stellt bestehende Annahmen in Frage und erweitert den Anwendungsbereich von Recourse über das überwachte Lernen hinaus.

Thus said the truthful Prophet: “seek knowledge from the cradle to the grave”

(Abul-Qâsem Ferdowsi)

Undertaking a PhD is akin to walking an endless path of discovery, a testament to the timeless wisdom advocating the lifelong pursuit of knowledge. I am truly fortunate to have been accompanied and guided by a host of individuals who have illuminated this path with their wisdom and support. Your enduring faith in my abilities has provided the sustenance needed to traverse the challenges of this journey. A heartfelt thank you for being the torchbearers on my academic expedition.

To my supervisors, Prof. Dr. Bernhard Schölkopf and Prof. Dr. Isabel Valera, I am grateful for your support in fostering my independent thinking, generously offering your time, patiently assisting with setbacks, believing in my capabilities, and inspiring me to become my best self. Your combined mentorship emboldened me to journey across the Atlantic and pursue a PhD in a foreign land. I am deeply appreciative of this opportunity you granted me.

To Prof. Dr. Gilles Barthe, thank you for warmly welcoming me to my first PhD project, attentively hearing my ideas, guiding me in mentoring, and treating me as a respected colleague. Your passion for research was contagious, and I hope to continue collaborating with you in the future.

To Prof. Dr. Thomas Hofmann, I appreciate your honest conversations, keen insights, and kind hospitality during my exchange at ETH. My time at ETH would not have been the same had it not been for your support.

To Prof. Adrian Weller, thank you for generously hosting the ELLIS Workshop on Causethical ML’s panel discussion during your personal vacation, at the last minute.

To my cherished friends, Adrián Javaloy Bornas, Dr. Patrick Putzky, Julius von Kügelgen, Kamil Adamczewski, Dr. Krikamol Muandet, Dr. Antonio Vergari, and Dr. Atalanti Mastakouri, I value our “creative” coffee hours, German tandem practice, and profound life discussions. I am very grateful to Patrick and Kamil for helping me with my residential moves throughout my PhD. To the entire EI department group, thank you for teaching me the ways of research, and inspiring me to improve every day.

To Miriam Rateike and Pablo Sanchez-Martin, your steadfast support and collaboration in organizing both the ELLIS Workshop on Causethical ML and the Causethical ML seminar at Saarland University has been invaluable. Additionally, I thank Martina Contisciani for your inspiring teaching style that motivated our joint hosting of the Causality mini-course. Creating these events with you has been a joy and highly rewarding.

To Dr. Been Kim, Dr. Simon Kornblith, and Dr. Lars Beusing, thank you for welcoming me during my internships at Google Brain and DeepMind. I gained invaluable knowledge during my time there!

To Arman Ghaffarizadeh, my long-time friend, thank you for lending your ear and wisdom in times of joy and hardship. I cherish our friendship, which has only deepened over the years.

To the students Alexandra Walter, Kiarash Mohammadi, Ricardo Dominguez-Olmedo, and Ahmad Ehyaei your patience allowed me to grow as your mentor. I hope I was worthy of your time.

To Prof. Caterina De Bacco and her delightful group of students, you made lunch and coffee breaks feel rejuvenating.

To the kind and helpful administration staff at both MPI and ETH, Sabrina Rehbaum (MPI), Ann-Sophie Bähr (MPI), Lidia Pavel (MPI), Annika Buchholz (MPI), Sarah Danes (MPI & ETH), Paulina Motyka (ETH), and Natalia Marciniak (ETH), your support allowed me to dedicate more time to my studies.

To the Centre for Learning Systems (CLS), Natural Sciences and Engineering Research Council of Canada (NSERC), and Google, I am grateful for your generous PhD fellowships throughout my academic journey.

And to my nearest and dearest.

To my parents, Prof. Gholamreza Karimi and Prof. Zohreh Azimifar, and my brother, Ali, to whom I am endebted for all of my opportunities, who have consistently set high standards and nurtured me throughout my life.

Most importantly, to my loving wife, Fatemeh, my partner in crime, whose unwavering love, support, sacrifice, and guidance served as a beacon of hope during the darkest times. I am incredibly fortunate to have you as my “hamsafar” and eagerly anticipate the many adventures that lie ahead for us!

Finally, and most certainly not least, I express my gratitude to the almighty é <Ë@, to whom I owe anything and everything.

The following peer-reviewed publications are at the core of my PhD research and covered in this dissertation:

1. “Model-Agnostic Counterfactual Explanations for Consequential Decisions,” Karimi, Barthe, Balle, Valera, AISTATS ( Á), 2019.  
2. “Algorithmic Recourse: from Counterfactual Explanations to Interventions,” Karimi, Schölkopf, Valera, ACM-FAccT ( ­), 2020.  
3. “Algorithmic recourse under imperfect causal knowledge: a probabilistic approach,” Karimi\*, von Kügelgen\*, Schölkopf, Valera, NeurIPS ( ­), 2020.  
4. “Scaling Guarantees for Nearest Counterfactual Explanations,” Mohammadi, Karimi, Barthe, Valera, ACM-AIES (Á), 2021.  
5. “A survey of algorithmic recourse: contrastive explanations and consequential recommendations,” Karimi, Barthe, Schölkopf, Valera, ACM Computing Surveys (), 2022.  
6. “Towards Causal Algorithmic Recourse,” Karimi\*, von Kügelgen\*, Schölkopf, Valera, Springer LNAI Book Chapter, 2022.  
7. “On the Fairness of Causal Algorithmic Recourse,” von Kügelgen, Karimi, Bhatt, Valera, Weller, Schölkopf, AAAI (Á), 2022.  
8. “On the Adversarial Robustness of Causal Algorithmic Recourse,” Dominguez-Olmedo, Karimi, Schölkopf, ICML (­), 2022.  
9. “Robustness Implies Fairness in Causal Algorithmic Recourse,” Ehyaei, Karimi, Schölkopf, Maghsudi ACM-FAccT, 2023.

The following peer-reviewed publications originated during my time as a PhD Student but are omitted from this thesis:

10. “On the Relationship Between Explanation and Prediction: A Causal View,” Karimi, Muandet, Kornblith, Schölkopf, Kim, ICML, 2023.

11. “On Data Manifolds Entailed by Structural Causal Models,” Dominguez-Olmedo, Karimi, Arvanitidis, Schölkopf, ICML, 2023.

All code is available at https://github.com/amirhk

Oral (Á); Spotlight (­); ≥100 citations (); Equal Contribution (\*)

## Basic

• x: scalar
• x: vector
• X: matrix
• X: set
• X: random variable
• : space, model, or constraint

## Recourse

• $\mathcal { D } \colon$ dataset
• $\phi \colon$ logic formula
• $h : \mathcal { X }  \mathcal { V } :$ discriminator
• ${ \mathcal { F } } \colon$ feasibility constraints
• $\mathcal { P } \colon$ plausibility constraints
• $\mathsf { c o s t } ( \cdot )$ or $c ( \cdot ) \colon$ : cost function
• ${ \sf d i s t } ( \cdot , \cdot )$ or $d ( \cdot , \cdot )$ : distance function
• $\mathbb { C F } _ { h } ( \mathbf { x } ^ { \mathsf { F } } )$ : set of counterfactual instances for instance $\mathbf { x } ^ { \mathsf { F } }$ and model h

## Causality

• S: set of structural equations
• $P _ { \mathbf { U } } .$ distribution over latent variables
• $\mathcal { M } = ( \mathbb { S } , P _ { \mathbf { U } } )$ : structural causal model
• $\mathcal { G } \colon$ corresponding graphical causal model
• $\mathcal { T } \colon$ subset of graph nodes
• ${ \mathrm { d } } ( { \mathcal { T } } ) ;$ : descendants of subset
• nd( ): non-descendants of subset $\mathcal { T }$
• $\Delta ( \mathbf { X } _ { \mathcal { T } } : = \pmb { \theta } )$ or $\Delta ( \pmb \theta _ { \mathcal { T } } )$ : set values of $\mathbf { \boldsymbol { x } } _ { \mathcal { T } }$ to θ via soft interventions
• do $( \mathbf { X } _ { \mathcal { I } } : = \pmb { \theta } )$ or $\mathrm { d o } ( \pmb { \theta } _ { \mathcal { T } } )$ : set values of $\mathbf { \boldsymbol { x } } _ { \mathcal { T } }$ to $\pmb { \theta }$ via hard interventions
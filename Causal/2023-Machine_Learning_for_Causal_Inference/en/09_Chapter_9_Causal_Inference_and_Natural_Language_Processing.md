# Chapter 9 Causal Inference and Natural Language Processing

![image_58](images/image_58.png)

Wenqing Chen and Zhixuan Chu

## 9.1 Causal Inference with Textual Data

Randomized controlled trials (RCTs) are often used in scientific studies to estimate causal effects between variables. However, RCTs are limited by high costs and ethical concerns [52]. When dealing with high-dimensional and nonstructural data like text, RCTs become more challenging due to the entanglement of concentrated variables in textual data. Alternatively, estimating causal effects from observational data is a more cost-effective and ethically safer approach that has gained increasing attention in recent research [31, 42, 58]. In this section, we focus on the use of observational data and show how textual data can be used for causal inference. For example, how product presentations are prepared to attract customers [61] and how loan applicants write statements can affect the receipt of funds [75].

Rubin’s and Pearl’s causal theories are two prominent approaches for causal inference in statistics and machine learning, and both can be used for causal inference with textual data, but Pearl’s approach based on graphical models is generally more commonly used in this context. Textual data often contain complex dependencies between variables, and Pearl’s graphical models provide a flexible and powerful framework for modeling these dependencies and inferring causal relationships between them [18, 37]. Depending on the research interest, recent work can be categorized into two types:

W. Chen (-) Sun Yat-sen University, Zhuhai, China e-mail: chenwq95@mail.sysu.edu.cn

Z. Chu Ant Group, Hangzhou, China e-mail: chuzhixuan.czx@alibaba-inc.com1. When variables of interest are linguistic properties, the research problem is to find effective ways to present text for a specific objective. For example, what is an effective way for political candidates to present their personal background to attract voters [22]? What is an effective strategy for business owners to compose product descriptions to enhance sales on e-commerce platforms [60, 63]?

2. When variables of interest are non-linguistic but correlated with textual data, the research problem is to accurately estimate causal effects. For example, does gender affect the popularity of an author’s posts on an online forum [18]? To what extent does censorship affect future posting rates, where the content of the texts is a confounder [65]? Moreover, textual data can serve as proxy variables in traditional causal inference problems. For instance, when estimating the causal effect of smoking on life expectancy, occupation may be a potential confounder but may not be recorded. In such cases, researchers may try to infer the occupation from an individual’s historical social media posts [37].

To estimate causal effects in the aforementioned situations, researchers must overcome two challenges. The first is a common issue in the field of causal inference, imagining the counterfactual world. The second challenge stems from the high-dimensional nature of the text, which requires researchers to find a lowdimensional representation that preserves the relevant causal relationships [16, 75]. However, obtaining such a representation is not straightforward, as linguistic variables in a text can be entangled with other linguistic or non-linguistic variables. For instance, when estimating the causal effect of the gender of an author on the popularity of their posts, the topic of the post may serve as a confounding variable, as certain topics may attract more males than females and be more popular in general, while the writing style may act as a mediator [18]. Thus, assumptions are required to perform causal inference with text data, and any representation of text should account for the hypothetical relationships between variables. Misidentifying a confounder as a mediator or vice versa can lead to biased estimates of the causal effect.

The advancements in natural language processing (NLP), such as the use of language models, topic models, and other contextual embedding models, provide promising methods for converting high-dimensional textual data to relatively lowdimensional data while respecting prior graph assumptions. Recent NLP work on using text for causal effect estimation can be categorized into four categories based on different assumptions about the role of text in the causal graph:

1. The text is viewed as a treatment, with the goal of estimating the causal effect of specific linguistic properties on outcomes [62]. For example, the way a campaigner presents their personal experience can impact the number of votes they receive [22], or how a company writes product descriptions to attract customers [60, 63]. However, there are two main challenges in this field. Firstly, different textual attributes are often intertwined in text, and when assuming there are N attributes, researchers usually only estimate the causal effect of one attribute at a time, leaving the remaining (N 1) attributes as potential confounders or mediators. While some studies assume that these (N 1)variables are all confounders [62], others note that this assumption is unrealistic for certain types of text, such as a text that is both “polite” and “profane” [18]. Secondly, there are unobserved confounders that cannot be reflected in the text, leading to biased estimates of the causal effect. For instance, readers with different political stances may choose different texts for reading, where the political stance of a person is not observed but will affect the estimated causal effect [18].

2. The text is viewed as a confounder, with some attributes in the text used as confounders that affect the observed treatments and outcome variables. For instance, in a study examining whether the first author of a paper being a woman results in higher influence (e.g., more citations), the potential confounders may include the topic of the paper and research field [65, 76]. Models can be constructed to predict treatment and outcome variables from text [81]. However, assuming text attributes are confounders can be risky, as if some of these attributes are mediators, the confounder assumption can lead to unreasonable counterfactual samples during counterfactual inference, violating the assumption of positivity [18].  
3. The text is viewed as a mediator, with the assumption that certain text attributes act as mediator variables. For instance, in the context of online forums, researchers have investigated the impact of the gender of a comment publisher on the popularity score of a comment, with men and women possibly adopting different tones and writing styles [81]. This type of problem involves estimating indirect and direct causal effects with greater granularity. The main challenges include assumptions about confounders and mediator variables in the text, and the construction of a conditional text representation based on treatments, along with developing a model that predicts the mediator from the text [36]. However, the optimal approach for constructing such a model is still a matter of debate.  
4. The text is viewed as an outcome, aiming to estimate the causal effect of treatment of interest on specific linguistic properties of the text generated. Examples of this type of research include exploring the impact of “female judges or non-white judges” on the language expression in legal documents [24] or how a student’s education level affects the readability of their paper [15]. The primary challenge in this research area is that text is unstructured data, making it difficult to design an evaluation model for these linguistic properties. NLP models are often necessary to convert text into structured attributes, but these models may also introduce certain biases.

In summary, this section explores how observational textual data are used in causal inference research. While traditional causal inference methods have mainly focused on structured data, the increasing relevance of language in social science has prompted researchers to explore causal inference with textual data. Recent work can be categorized into different categories based on the assumption that text plays different roles in the causal graph. Each category poses unique challenges, such as the presence of confounding variables or the difficulty in properly representing textual data. Nonetheless, NLP models offer potential solutions to these challenges.

## 9.2 Spurious Correlations in NLP

Besides using NLP models to estimate causal effects, concerns about the trustworthiness of such models have been raised due to their reliance on learning statistical correlations for prediction, regardless of the underlying causal relationship. Such correlations are defined as spurious correlations, which refer to non-causal but correlated relationships [74]. As deep neural models have made significant progress in NLP [96], it is risky to rely on the assumption that the distributions of training and test data are identical. Recently, pretrained language models (PLMs) [14, 27, 28, 44, 80] have even achieved superhuman performance on certain text understanding tasks and datasets,1 but their robustness is still a major concern.

As an example, sentiment analysis is an NLP task where the goal of the model is to classify a given text’s sentiment as “positive,” “negative,” or “neutral.” However, deep learning models trained on the IMDB movie review dataset have been observed to rely on spurious correlations, leading to unreliable decision-making. Specifically, movie reviews containing the word “Spielberg” are often labeled as “positive,” which leads to a high correlation [88]. However, this correlation does not reflect a causal relationship between the presence of the word “Spielberg” and the positive sentiment of the review. If “Spielberg” is replaced with another director’s name while keeping all other words unchanged, the sentiment of the review will not change. Such decision-making based on spurious correlations is referred to as “right for the wrong reasons” [47] or “reasoning shortcuts” [8, 13, 54], which leads to low robustness of the model when the data distribution changes.

Research has demonstrated that even state-of-the-art PLMs are not immune to spurious correlations, especially when specific minority textual patterns are underrepresented in the training data [80, 87]. For instance, in the paraphrase identification task, PLMs fine-tuned on the QQP dataset [33] tend to heavily rely on the spurious-correlated feature of “lexical overlap” for decision-making, which is not a reliable cue for paraphrasing since humans can use different words to convey the same meaning [80]. Similarly, when fine-tuned on the ARCT dataset [26], BERT [14] becomes overly reliant on the specific keyword “not” for reasoning. Changes to the test set, resulting in the removal of spurious-correlated features, can cause significant performance degradation, with the model’s performance becoming comparable to random guessing [53].

These studies illustrate that, despite the significant progress made by deep neural models in NLP, the spurious correlation issue remains a challenge. As a consequence, the model’s performance can drastically decrease when the data distribution changes, limiting its applicability in real-world scenarios. In NLP, this issue can affect both natural language understanding (NLU) and natural language generation (NLG) tasks. We systematically review recent works that have reported this problem.

1. In NLU, models may rely on “non-semantic” or “shallow semantic” textual patterns to make predictions, such as syntactic properties or specific keywords. These features can be used for prediction without capturing the deep semantics of the input text, leading to reasoning shortcuts [8, 13, 47, 54]. For instance, “non-semantic textual patterns” such as syntactic properties have been employed to make decisions [47]. In the natural language inference task of the MNLI dataset [85], a strong correlation has been observed between the “lexical overlap” between the input hypothesis text H and the premise text P and the label “Entailment.” The lexical overlap refers to the continuous subsequences of H in P , syntactic subtrees, and other syntactic features [47, 95]. Similarly, in the Quora Question Pairs (QQP) dataset [33] for the paraphrase identification task, models have been found to rely on the lexical overlap for prediction [80]. However, from a human perspective, these features contain limited semantic information and may not be applicable in real-world scenarios. Therefore, they should not be used in NLU tasks. “Shallow semantic textual patterns,” such as specific words or clues, have also been used to make predictions [53, 88]. For instance, in the MNLI dataset, the presence of the keyword “not” in the hypothesis text has been observed to be strongly correlated with the label “contradictory” [25]. However, this approach can lead to unreliable decisionmaking as the model may make correct predictions without observing the premise text. Similarly, in the sentiment classification task, a correlation has been observed between the presence of “Spielberg” and positive sentiment labels. However, depending on specific keywords can lead to inaccurate predictions for movie reviews that contain “Spielberg” but have a negative sentiment [88]. Studies have also demonstrated that when these keywords are added, deleted, or rewritten to construct new data samples, the model’s prediction accuracy drops significantly [50], indicating the formation of inference shortcuts.

2. The phenomenon of spurious correlation is pervasive in NLG tasks, although it is seldom examined from a causal perspective. NLG tasks, such as machine translation [3], abstractive summarization [51], conversation [83], and image captioning [92], necessitate semantic alignment between the input data and the generated text. However, researchers have noted that NLG models frequently produce text that is nonsensical or semantically unfaithful to the input data, a phenomenon known as the hallucination problem [34]. This problem is frequently attributed to the existence of spurious correlations, which can arise from a variety of factors, such as semantically inadequate representation learning [1, 20, 34, 40] and semantic misalignment, wherein the decoder attends to the wrong portion of the encoded input data [79]. A recent example in the image captioning task has shown that some models may erroneously identify men with long hair as women due to spurious correlations between the visual feature of “long hair” and the token “female” in the caption [10]. Similarly, in the table-to-text generation task, recent research has identified spurious correlations between linguistically similar entities [9].

In short, the use of non-semantic or shallow semantic textual patterns in NLU can lead to reasoning shortcuts, as models rely on syntactic properties or specific keywords instead of capturing the deep semantics of the input text. Similarly, spurious correlations are pervasive in NLG, leading to nonsensical or semantically unfaithful text, known as the hallucination problem [34].

Recent research has identified spurious correlations as a persistent issue in NLP. These correlations often stem from biases inherent in the training data. Two primary sources of such biases are selection bias and annotation bias, which have been extensively explored in the literature [4, 29]. Selection bias arises from the biased selection of data samples with specific characteristics during dataset collection. For instance, a significant number of English-language NLP datasets are derived from historical news repositories, such as the Wall Street Journal and Frankfurt Radio, which may be predominantly authored by white, middleaged, educated, upper-middle-class men [30]. Consequently, models trained on such datasets may learn text patterns specific to this demographic, which are not necessarily generalizable to other age groups or genders [29]. Annotation bias, on the other hand, arises due to the preferences of annotators. For example, in the natural language inference datasets SNLI and MNLI [85], annotators are instructed to generate three different “hypothetical texts” [25]. When generating “hypothetical texts” labeled as “contradictory,” annotators often introduce the keyword “not,” which can create a false correlation between the label “contradictory” and the keyword “not.”

The prevalence of spurious correlations in NLP highlights the need for more careful curation and annotation of datasets, as well as for the development of robust techniques to detect and mitigate such biases in models [29].

## 9.3 Causality-Driven Models for NLP

In response to the issue of spurious correlation and its negative impact on deep learning models, many researchers have proposed various approaches to inject causality into the models, aiming to enhance their robustness and generalization abilities [9, 10, 18, 32]. These efforts have shown promising results in mitigating the bias introduced by spurious correlations and have the potential to improve the performance of NLP models in various tasks.

## 9.3.1 Preliminaries

We provide a brief introduction to the two prominent causal theories, namely Rubin’s Potential Outcome Framework (POF) [68] and Pearl’s Structural Causal Model (SCM) [55, 57]. POF defines causality in terms of the comparison of outcomes under different treatments or interventions, while SCM represents causal

<!-- footnote -->

- P. Sheth (-) · H. Liu Arizona State University, Tempe, AZ, USA e-mail: psheth5@asu.edu; huanliu@asu.edu

<!-- footnote end -->

relationships between variables using directed acyclic graphs (DAGs). While both frameworks were initially developed to measure the causal effect between variables, in this section, we focus on related works that introduce causality to NLP models.

One crucial difference between the two causal models is the role of causal graph assumptions for variables. POF does not assume any graph structure between variables, while SCM represents the causal relationships between variables in the form of a DAG. In terms of utilizing causal knowledge to improve machine learning models, SCM is more widely applied [70, 71]. This is partly due to the historical development of machine learning, where representing relationships between variables using graph structures is common.

In this section, we discuss Pearl’s influential “causal ladder” framework [57] and its application to recent works on causality-driven models for NLP. The “causal ladder” categorizes causality into three levels: association, intervention, and counterfactuals, which correspond to observation, action, and imagination in human cognition, respectively.

The first level, association, refers to the statistical correlation between variables. Many machine learning models operate at this level [57], learning the conditional probability distribution $P ( Y = y ~ \vert ~ X = x )$ . However, as discussed in Sect. 9.2, such models may infer spurious correlations due to the presence of confounding variables.

The second level, intervention, examines how the value of Y changes if the value of X is manipulated. This level involves the Do-Calculus, which calculates the probability $P ( Y = y ~ \mid ~ \mathrm { d o } ( X = x ) )$ , representing the probability of Y taking on the value y if the value of X is intervened to x. Since the change in the value of X is a result of the intervention and not influenced by the confounding variable C, the causal arrow $C  X$ is removed after the operation $\operatorname { d o } ( X = x )$ . Accordingly, the optimization objective function of the corresponding machine learning model - - should also be adjusted to $P ( Y = y ~ \vert ~ \mathrm { d o } ( X = x ) ) ~ [ 5 , 8 6 ]$ .

The third level, counterfactual, involves the imagination of a parallel or hypothetical world. In this world, counterfactual values (x, y) of (X, Y ) that have not occurred in the real world are considered. For example, if a patient did not take a certain drug and died in reality, the question of whether the patient would have- - survived if they had taken the drug arises. However, since the patient’s death has occurred in reality, the counterfactual value cannot be observed. The counterfactual problem can be formally defined as estimating $P ( Y = \widetilde { y } \mid x , y , \mathrm { d o } ( X = \widetilde { x } ) )$ . A significant amount of research in machine learning aims to train models to estimate and answer this counterfactual question [49].

## 9.3.2 Intervention-Level Debiasing

Spurious correlation in deep learning arises when potential confounders exist [66]. The model may erroneously treat confounders as mediators, leading to an incorrect reasoning pathway: $X \ \to \ C \ \to \ Y$ where the arrow -- represents the posterior pathway, which is non-causal and not generalizable in the real world.

Intervention-level debiasing typically adjusts the learning objective of the model from $P ( Y = y ~ \vert ~ X = x )$ to $P ( Y = y ~ \vert ~ \operatorname { d o } ( X = x ) )$ , which blocks the pathway $X  C$ by the do-calculus. However, it requires prior knowledge in the form of a causal graph involving the confounder C. Depending on the assumption of confounders, recent works can be summarized into the following categories:

1. The first line of works explicitly assumes the observation of confounders and changes the learning objective from

$$
P _ {\theta} (Y = \mathbf {y} \mid X = \mathbf {x}) = \sum_ {c} P _ {\theta} (Y = \mathbf {y} \mid X = \mathbf {x}, C = \mathbf {c}) \underline {{P (C = \mathbf {c} \mid X = \mathbf {x})}} \tag {9.1}
$$

t o

$$
P _ {\theta} (Y = \mathbf {y} \mid \mathrm{do} (X = \mathbf {x})) = \sum_ {c} P _ {\theta} (Y = \mathbf {y} \mid X = \mathbf {x}, C = \mathbf {c}) \underline {{P (C = \mathbf {c})}} \tag {9.2}
$$

where θ denotes the model parameters, and the do-calculus makes the confounder independent of the input variable, represented by $c \perp X$ . This intervention makes the posterior probability $\begin{array}{c} P ( C = \pmb { c } \end{array} | \pmb { \cal X } = \pmb { x } )$ intervene into $P ( C =$ c) [38, 78, 86]. Such methods have been applied to applications such as text classification [38], natural language inference [78], and image captioning [43, 86, 94]. The implementation of Eq. 9.2 usually assumes C to be a categorical variable and $P ( C = { \pmb { c } } )$ is precomputed in the training data. In some recent works [43, 86], the process of $P _ { \theta } ( Y = y \mid X = x , C = c )$ is also a classification problem and the network contains a final softmax layer denoted by:

$$
P _ {\theta} (Y = \mathbf {y} \mid X = \mathbf {x}, C = \mathbf {c}) = \text { Softmax } (f _ {y} (\mathbf {x}, \mathbf {c})) \tag {9.3}
$$

where $f _ { y } ( x , \pmb { c } )$ calculates the logits for all categories. Equation 9.2 becomes:

$$
P _ {\theta} (Y = \mathbf {y} \mid \mathrm{do} (X = \mathbf {x})) = \mathbb {E} _ {\mathbf {c} \sim p (\mathbf {c})} \left[ \operatorname{Softmax} \left(f _ {y} (\mathbf {x}, \mathbf {c})\right) \right] \tag {9.4}
$$

while the expectation operation involves the expensive sampling of c. Normal-      ized weighted geometric mean (NWGM) approximation [86, 93, 94] is often used to reduce the computation cost by:

$$
\mathbb {E} _ {\boldsymbol {c} \sim p (\boldsymbol {c})} \left[ \operatorname{Softmax} \left(f _ {y} (\boldsymbol {x}, \boldsymbol {c})\right) \right] \approx \operatorname{Softmax} \left(\mathbb {E} _ {\boldsymbol {c} \sim p (\boldsymbol {c})} \left[ f _ {y} (\boldsymbol {x}, \boldsymbol {c}) \right]\right) \tag {9.5}
$$

where the function $f _ { y } ( \cdot )$ is implemented by a linear model with parameters $W _ { 1 }$ and $W _ { 2 }$ . In recent work [86], since the confounder C is intervened to be independent with X, the expectation term becomes:

$$
\mathbb {E} _ {\boldsymbol {c} \sim p (\boldsymbol {c})} \left[ f _ {y} (\boldsymbol {x}, \boldsymbol {c}) \right] = \boldsymbol {W} _ {1} \boldsymbol {x} + \boldsymbol {W} _ {2} \cdot \mathbb {E} _ {\boldsymbol {c} \sim p (\boldsymbol {c})} \left[ g _ {y} (\boldsymbol {c}) \right] \tag {9.6}
$$

where $\mathbb { E } _ { c \sim p ( c ) } \left[ g _ { y } ( \pmb { c } ) \right]$ could be computed in parallel for all the possible categories of the confounder [86]. It is worth noting that Eqs. 9.3–9.6 is just one kind of work to implement $P ( Y = y ~ \vert ~ \mathrm { d o } ( X = x ) )$ . There is also other related work using adversarial learning [39, 63] to approximate the intervention operation. Specifically, this work built a discriminator to utilize the representation H of the input variable X to predict the confounder C, and the generator is to generate the representation H that is unable to predict C from. When the generator and the discriminator reach Nash equilibrium, it is considered that the hidden state H does not contain information that can predict the confounder C.

2. The second line of works aims to relax the assumption of confounders, as in the real world, true confounders may be unobserved or unmeasured [9, 10, 32, 48]. For instance, direct measurement of an individual’s socioeconomic status may be difficult, but it is possible to obtain a proxy through their zip code or occupation [45]. Additionally, natural language data is high-dimensional, making the identification of potential confounders more complex than previously assumed. Recent studies have addressed this issue by assuming the presence of real confounders in the latent space, and that proxy confounders can be observed [9, 10, 32]. To address this, Conditional Variational Auto-Encoders (CVAEs) were used with a modified learning objective from the original formulation: 

$$
\begin{array}{l} \log p (\mathbf {y} \mid \mathbf {x}) \geq \mathbb {E} _ {\mathbf {z} _ {c} \sim q _ {\phi} (\mathbf {z} _ {c} | \mathbf {x}, \mathbf {y})} \log p _ {\theta} (\mathbf {y} \mid \mathbf {x}, \mathbf {z} _ {c}) \\ - \mathrm{KL} \left[ q _ {\phi} \left(\boldsymbol {z} _ {c} \mid \boldsymbol {x}, \boldsymbol {y}\right) \mid p \left(\boldsymbol {z} _ {c} \mid \boldsymbol {x}\right) \right] \tag {9.7} \\ \end{array}
$$

to:

$$
\log p (\mathbf {y} \mid \mathrm{do} (\mathbf {x})) \geq \mathbb {E} _ {\mathbf {z} _ {c} \sim q _ {\phi} (\mathbf {z} _ {c} | \mathbf {y})} \log p _ {\theta} (\mathbf {y} \mid \mathbf {x}, \mathbf {z} _ {c}) \tag {9.8}
$$

$$
- \operatorname{KL} \left[ q _ {\phi} \left(\boldsymbol {z} _ {c} \mid \boldsymbol {y}\right) \mid p \left(\boldsymbol {z} _ {c}\right) \right]
$$

where $\theta$ and $\phi$ denote the parameters of prior and posterior networks, respectively. And $z _ { c }$ denotes the latent confounder, which should be independent of x after the do-calculus. When further considering the proxy confounder c, Eq. 9.8 becomes:

$$
\log p (\boldsymbol {y}, \boldsymbol {c} \mid \mathrm{do} (\boldsymbol {x})) \geq \mathbb {E} _ {\boldsymbol {z} _ {c} \sim q _ {\phi} (\boldsymbol {z} _ {c} | \boldsymbol {y}, \boldsymbol {c})} \log p _ {\theta} (\boldsymbol {y}, \boldsymbol {c} \mid \boldsymbol {x}, \boldsymbol {z} _ {c}) \tag {9.9}
$$

$$
- \operatorname{KL} \left[ q _ {\phi} \left(\boldsymbol {z} _ {c} \mid \boldsymbol {y}, \boldsymbol {c}\right) \mid p \left(\boldsymbol {z} _ {c}\right) \right]
$$

Since the do-calculus will also make the proxy confounder c independent from x, Eq. 9.9 becomes:

$$
\begin{array}{l} \log p (\boldsymbol {y} \mid \mathrm{do} (\boldsymbol {x})) \geq \mathbb {E} _ {\boldsymbol {z} _ {c} \sim q _ {\phi} (\boldsymbol {z} _ {c} | \boldsymbol {y}, \boldsymbol {c})} \left[ \log p _ {\theta} \left(\boldsymbol {y} \mid \boldsymbol {x}, \boldsymbol {z} _ {c}\right) + \log p _ {\theta} \left(\boldsymbol {c} \mid \boldsymbol {z} _ {c}\right) \right] \\ - \mathrm{KL} \left[ q _ {\phi} \left(\mathbf {z} _ {c} \mid \mathbf {y}, \mathbf {c}\right) \mid p \left(\mathbf {z} _ {c}\right) \right] - \log p (\mathbf {c}) \tag {9.10} \\ \end{array}
$$

3. The third line of research takes a different approach by avoiding prior assumptions about confounders or proxy confounders. Instead, it implicitly estimates confounders by leveraging multiple datasets. For example, Landeiro et al. [39] estimate the impact of words in the input document X by computing the difference between the topic model of the training set and the test set. This method can estimate potential confounders because confounders may vary   across different distributions. However, this approach requires the text of the test set to be known beforehand, which is an unrealistic assumption in realworld scenarios. Recent works have approached this problem differently by assuming the availability of multiple datasets, $D _ { e } : = \smash { \big \{ \big ( \mathbf { x } _ { i } ^ { e } , \mathbf { y } _ { i } ^ { e } \big ) \big \} _ { i = 1 } ^ { n _ { e } } }$ , collected from various environments $( e \in \mathcal { E } _ { \mathrm { a l l } } )$ , where $n _ { e }$ represents the number of datasets in different environments [2, 59]. The goal of this approach is to learn a robust predictive model $Y = f ( X ; \theta )$ that remains stable across a given number of environments [2].

## 9.3.3 Counterfactual-Level Debiasing

Counterfactual-level debiasing involves generating counterfactual samples $( \widetilde { \pmb x } , \widetilde { \pmb y } )$ that are compared with observed samples (x, y) to answer questions such as “Why?” or “What are the causal features for the prediction?” [49]. Counterfactual data augmentation is a commonly used method for this purpose, which includes the manual or automatic generation of counterfactual samples that are mixed for training [35]. Counterfactual samples are usually created by modifying an original sample in a way that leads to a different prediction by a machine learning model [35, 73].

Existing works can be classified into two categories based on whether causal features in x are manipulated:

1. The first type of method involves manipulating the non-causal features of x while leaving the corresponding label y unchanged. This method is mainly used to address fairness problems caused by certain sensitive attributes, such as gender - and race [23]. However, it cannot cover all confounders.

2. The second type of method involves changes to causal features that flip the label from $\textbf {  { y } }$ to $\widetilde { \mathbf { y } }$ of the sample [35, 78, 90]. This method has been shown to improve the out-of-distribution generalization ability of the model and can make the model less sensitive to noise [35].

Recent works can also be categorized into three types based on the methods employed for modifying the data:

1. Manual modification, as described in research such as [35], involves making minor adjustments to the text by human annotators to change the label without making any unnecessary modifications that do not affect the label. This method can produce high-quality counterfactual samples, but it can be expensive in terms of labeling effort.  
2. Rule-based modification, such as replacing specific types of object vocabulary in the text with another type of vocabulary. This method, as proposed in [23, 72, 89], has the advantage of being low cost but may result in unnatural text.  
3. Automatic generation of counterfactual samples, as proposed in [64, 90, 91], uses pretrained models like GPT-2 to perform operations such as vocabulary replacement and attribute editing to generate counterfactual samples. This method addresses the limitations of the first two methods by being more costeffective and producing smoother text. However, it should be noted that text generation is still a challenging task, and the accuracy and semantic fidelity of attribute editing in the generated text are uncertain.

Explicitly Answering What-if Questions The primary focus of works in counterfactual data augmentation has been to assist models in identifying causal patterns for decision-making without explicitly answering counterfactual questions such as “What would happen if. . . ?” However, recent research has demonstrated that models at the counterfactual level possess the ability to answer counterfactual questions, as shown on the causal ladder. To facilitate this, specialized question answering (QA) datasets have been constructed, such as WIQA [77], which consists of three components: procedural text, influence graphs, and what-if multi-choice questions. The procedural text provides information about the events, the influence graphs depict the causal relationships between these events, and the what-if questions are derived from the graphs. Another dataset, Tat-QA [97], has been developed for table-based QA, which is shown to be a challenging task. Recent work has proposed a counterfactual thinking process with discrete reasoning for this task, in addition to the traditional QA objective [41]. Specifically, this approach utilizes sequence tagging to identify relevant cells within the table and relevant spans of text to infer their semantics. It then employs symbolic reasoning, using a set of aggregation operators to derive the final answer. The approach also includes regularization terms to supervise the target fact in the context of the question and to supervise the derivation operation required to infer the counterfactual context [41].

## 9.4 Causal Interpretations of NLP Models

Recent deep neural models have achieved significant success in NLP, but their deep structure and nonlinear nature make them difficult to interpret, which is nevertheless crucial for users to trust artificial intelligence (AI) systems. This problem is particularly pronounced in the development of large-scale PLMs due to their large number of parameters and nonlinearity. Additionally, in NLP, basic textual features such as word n-grams may not capture the high-level semantics conveyed in the text. Even if the text conveys abstract linguistic concepts such as topic or sentiment, these concepts may not be explicitly encoded in the model’s input, leading to lacking clear interpretability [19].

While many surveys have attempted to classify existing works [6, 7, 19, 82, 84], we suggest following the categorization proposed by Madsen et al. [46], where each work is classified based on two dimensions of categories:

1. Local or global interpretations, depending on whether the method explains individual instances (referred to as “local interpretations”) or the entire model (referred to as “global interpretations”) [46]. Local explanations provide insight into a single observation, for example, identifying the input features that are most important for the prediction. Global explanations, on the other hand, summarize the entire model with regard to a specific aspect, such as how the model relates words to each other, the linguistic information the model uses, or the general rules that summarize an aspect of the model.  
2. Intrinsic or post-hoc interpretations. The need for interpretability is often motivated by a requirement for accountability. In situations where the consequences of a model’s decisions are significant, it is crucial to minimize the risk of model failure by interpreting the model before deploying it [69]. This means it is important to distinguish between situations where interpretability is applied proactively (before deployment) or retroactively (after deployment) [46]. The methods that can be applied retrospectively are also referred to as “post-hoc” methods, while the term “intrinsic” is used to refer to models that are explainable by design.

From the first dimension, estimating averaged treatment effect (ATE) is a kind of global interpretation while estimating individual treatment effect (ITE) is a type of local interpretation [81]. Estimating ATE involves the treatment that could be textual concepts [19] or binary variables like gender [82]. Although AET estimation belongs to the global interpretation, it requires the counterfactual sample estimation, which often uses local perturbations of the input but can lead to inaccurate or misleading interpretations. This can occur, for example, when two concepts that might explain the model’s prediction are highly correlated with each other [19]. Fader et al. developed a method for providing causal explanations for any textual concept and created a dataset to allow comparison of any causal estimator with the ground truth [19]. They also created a language representation that can be used to approximate the counterfactual for a given concept, enabling the interpretation of causal models without the need for manually created examples. Estimating ITE involves answering the counterfactual questions. For example, recent work estimates the ITE in the task of legal judgment prediction, aiming to answer “whatif” questions like “what would the predicted judgment be if the input text did not contain certain concepts?” [11].

From the second dimension, estimating ATE or ITE [11, 19, 82] is a kind of posthoc interpretation as it mainly focuses on the behavior rather than finding intriguing properties of models.

Besides estimating treatment effect to answer “what-if” questions, Moraffah et al. pointed out a next level of interpretability, counterfactual explanation, which is to answer “why” questions [49] as suggested by Pearl [56]. The difference from “whatif” questions is that counterfactual explanations require generating counterfactual samples obtained by performing minimal changes that influence the output [12, 49]. It means that answering “why” questions will focus on a few numbers of textual features [17, 21, 67, 82, 91].

## 9.5 Summary

In summary, this chapter has discussed the challenges and opportunities arising from the intersection of causal inference and NLP and addressed two fundamental questions: how NLP can assist causal inference with textual data, and how causal inference theory can improve the robustness and interpretability of NLP models. Firstly, the chapter provides an overview of recent developments in causal inference with textual data and highlights the obstacles due to the unstructured and high-dimensional nature of the text. Secondly, we show that the spurious correlation problem remains a significant challenge for NLP models, which can lead to unreliable decision-making and reasoning shortcuts, limiting the model’s robustness and applicability in real-world scenarios. Thirdly, the chapter explores causality-driven models for NLP, including intervention-level and counterfactuallevel debiasing approaches to integrating causality into NLP models. Finally, we present the potential for causal interpretations to facilitate a deeper understanding of NLP models.

## References

1. R. Aralikatte et al., Focus attention: promoting faithfulness and diversity in summarization, in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (2021), pp. 6078–6095  
2. M. Arjovsky et al., Invariant risk minimization (2019). arXivabs/1907.02893  
3. D. Bahdanau, K. Cho, Y. Bengio, Neural machine translation by jointly learning to align and translate, in 3rd International Conference on Learning Representations, ICLR (2015)  
4. E. Bareinboim, J. Pearl, Controlling selection bias in causal inference, in Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics, PMLR. vol. 22 (2012), pp. 100–108  
5. E. Bareinboim et al., On pearl’s hierarchy and the foundations of causal inference, in Probabilistic and Causal Inference (2022)  
6. Y. Belinkov, S. Gehrmann, E. Pavlick, Interpretability and analysis in neural NLP, in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: Tutorial Abstracts (2020), pp. 1–5  
7. Y. Belinkov, J. Glass, Analysis methods in neural language processing: a survey, Trans. Assoc. Comput. Linguist. 7, 49–72 (2019)  
8. R. Bommasani, C. Cardie, Intrinsic evaluation of summarization datasets, in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP (2020), pp. 8075–8096  
9. W. Chen et al., De-confounded variational encoder-decoder for logical table-to-text generation, in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL-IJCNN (2021), pp. 5532–5542  
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
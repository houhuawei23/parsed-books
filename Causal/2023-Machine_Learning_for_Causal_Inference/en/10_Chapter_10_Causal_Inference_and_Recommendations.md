# Chapter 10 Causal Inference and Recommendations

![image_59](images/image_59.png)

Yaochen Zhu, Jing Ma, and Jundong Li

## 10.1 Introduction

With information growing exponentially on the web, recommender systems (RSs) are playing an increasingly pivotal role in modern online services, due to their ability to automatically deliver items1 to users based on their personalized interests. Traditional RSs can be mainly categorized into three classes [53]: Collaborative filtering-based methods [29], content-based methods [39], and hybrid methods [9]. Collaborative filtering-based RSs estimate user interests and predict their future behaviors by exploiting their past activities, such as browsing, clicking, purchases, etc. Content-based methods, on the other hand, predict new recommendations by matching user interests with item content. Hybrid methods combine the advantages of both worlds, where collaborative information and user/item feature information are comprehensively considered to generate more accurate recommendations.

1 We use the term item in a broad sense to refer to anything recommendable to users, such as news [38], jobs [47], articles [68], music [95], movies [20], micro-videos [84], PoIs [93], hashtags [17], etc.

Y. Zhu

Department of Electrical and Computer Engineering, University of Virginia, Charlottesville, VA, USA

e-mail: uqp4qh@virginia.edu

J. Ma

Department of Computer Science, University of Virginia, Charlottesville, VA, USA

e-mail: jm3mr@virginia.edu

J. Li (-)

Department of Electrical and Computer Engineering, Department of Computer Science, and School of Data Science, University of Virginia, Charlottesville, VA, USA

e-mail: jl6qk@virginia.edu; jundong@virginia.eduAlthough recent years have witnessed substantial achievements for all three classes of RSs introduced above, a great limitation of these methods is that they can only estimate user interests and predict future recommendations based on correlations in the observational user historical behaviors and user/item features, which guarantee no causal implications [15, 24, 92]. For example, a collaborative filtering-based RS may discover that several drama shows from a certain genre tend to have high ratings from a group of users and conclude that we should keep recommending drama shows from the same genre to these users. But there is an important question: Are the high ratings caused by the fact that the users indeed like drama shows from this genre, or they were limitedly exposed to drama shows from the same genre (i.e., exposure bias), and if given a chance, they would prefer something new to watch? In addition, a content-based RS may observe that microvideos with certain features are associated with more clicks and conclude that these features may reflect the current trend of user interests. But are the clicks because these micro-videos tend to have sensational titles as clickbait where users could be easily deceived? Moreover, if the titles of these micro-videos are changed to the ones that reflect their true content, would users still click them? The above questions are causal in nature because they either ask about the effects of an intervention (e.g., what the rating would be if a new drama show is made exposed to the user) or a counterfactual outcome (e.g., would the user still click a micro-video if its title had been changed to faithfully reflect the content), rather than mere associations in the observational data. According to Pearl [50], these questions lie on Rungs 2 and 3 of the Ladder of Causality, i.e., interventional and counterfactual reasoning, and they cannot be answered by traditional RSs that reason only with associations, which lie on Rung 1 of the ladder.

Why are these causal questions important for RSs? The first reason is that failing to address them may easily incur bias in recommendations, which can get unnoticed for a long time. If the collaborative filtering-based RSs mentioned above mistake exposure bias for user interests, they would amplify the bias by continuously recommending users with similar items; eventually, recommendations will lose serendipity, and users’ online experience can be severely degraded. Similarly, for the content-based micro-video RSs, if they cannot distinguish clicks due to user interests from the ones deceived by clickbait, they may over-recommend micro-videos with sensational titles, which is unfair to the uploaders of highquality micro-videos who put much effort into designing the content. In addition, understanding the cause of user activities can help improve the explainability of recommendations. Consider the causal question of whether a user purchases an item due to its quality or its low price. Pursuing the causal explanations behind user behaviors can help service providers to enhance the RS algorithm based on users’ personalized preferences. Finally, causal inference allows us to identify and base recommendations on causal relations that are stable and invariant, while discarding other correlations that are undesirable or susceptible to change. Take restaurant recommendations as an example. Users can choose a restaurant because of its convenience (e.g., going to a nearby fast food shop to quickly grab a bite, but they do not necessarily like it, a non-stable correlation) or due to their personal interests (e.g., traveling far away for a hot-pot restaurant, a stable causal relation). If an RS can properly disentangle users’ intent that causally affects their previous restaurant visits, even if the convenience levels of different restaurants may change due to various internal or external reasons, such as users’ moving to a new place, the system can still adapt well to the new situation. From this aspect, the generalization ability of the causal RSs can be substantially improved.

![image_60](images/image_60.png)

```mermaid
graph TD
  A["RCM-based"] --> B["Causal Inference"]
  C["SCM-based"] --> B
  D["CF-based"] --> E["Recommender Systems"]
  F["Content-based"] --> E
  G["Hybrid"] --> E
  B --> H["Causal Inference in Recommendations"]
  E --> H
  H --> I["Evaluation Strategies"]
  H --> J["Future Directions"]
  K["Causal Debiasing"] --> L["promotes"]
  M["Causal Generalization"] --> N["promotes"]
  O["Causal Explanation"] --> P["Section 10.4.1"]
  P --> Q["Exposure Bias"]
  P --> R["Popularity Bias"]
  P --> S["Clickbait"]
  P --> T["Unfairness"]
  U["Intervention-based"] --> V["Section 10.4.3"]
  W["Disentangle-based"] --> X["Section 10.4.2"]
  Y["Causal Embeddings"] --> Z["Section 10.6, 10.7"]
  AA["Colliding Effects"] --> AB["Section 10.2"]
```

Fig. 10.1 An overview of the structure of this chapter and connections between different sections

This chapter provides a systematic overview of recent advances in causal RS research. The organization is illustrated in Fig. 10.1. We start with the fundamental concepts of traditional RSs and their limitation of correlational reasoning in Sect. 10.2. Then Sect. 10.3 recaps two important causal inference paradigms in machine learning and statistics and shows their connections with the recommendation task. Section 10.4 thoroughly discusses how different causal inference techniques can be introduced to address the limitations of traditional RSs, with an emphasis on debiasing, explainability promotion, and generalization improvement. Section 10.5 summarizes the offline evaluation strategies for causal RSs. Finally, Sects. 10.6 and 10.7 discuss open questions and future directions for causal RSs and conclude this chapter.

## 10.2 Recommender System Basics

To keep this chapter compact, we confine our discussions to simple RSs with I users and J items. The main data for the RSs, i.e., users’ historical behaviors, are represented by a user–item rating matrix $\textbf { R } \in \mathbb { R } ^ { I \times J }$ , where a nonzero element $r _ { i j }$ $j .$ $r _ { i k } ^ { 0 }$ indicates the rating is missing.2 To make the discussions of RSs compatible with causal inference, we take a probabilistic view of R [46], where $r _ { i j }$ is assumed to be the realized value of the random variable R dependent on user i and item $j . ^ { 3 }$ In addition to R, an RS usually has access to side information like user features $\dot { \bf f } _ { i } ^ { u } \in \mathbb { R } ^ { K _ { F } ^ { u } }$ , such as her age, gender, and location, or item features ${ \bf f } _ { j } ^ { v } \in \mathbb { R } ^ { K _ { F } ^ { v } }$ , such as its content and textual description. $K _ { F } ^ { u }$ and $K _ { F } ^ { v }$ are the dimensions of user and item features, respectively. The main purpose of an RS is to predict users’ ratings for previously uninteracted items (i.e., $r _ { i k } ^ { 0 }$ $r _ { i j }$ user and item side information such as ${ \bf f } _ { i } ^ { u }$ and ${ \bf f } _ { j } ^ { v }$ , such that new relevant items can be properly recommended based on users’ personalized interests.

## 10.2.1 Collaborative Filtering

Collaborative filtering-based RSs recommend new items by leveraging user ratings in the past. They generally consider the ratings $r _ { i j }$ as being generated from a user latent variable ${ \bf u } _ { i } \in \mathbb { R } ^ { K }$ that represents user interests and an item latent variable $\mathbf { v } _ { j } \in$ $\mathbb { R } ^ { K }$ that encodes the item attributes (i.e., item latent semantic information), where K is the dimension of the latent space. Here we list three widely used collaborative filtering-based RSs, which will be frequently used as examples in this chapter:

• Matrix Factorization (MF) [28]. MF models $r _ { i j }$ with the inner product between $\mathbf { u } _ { i }$ and $\mathbf { v } _ { j }$ , where $r _ { i j } \sim N ( \mathbf { u } _ { i } ^ { T } \cdot \mathbf { v } _ { j } , \sigma _ { i j } ^ { 2 } )$ and $\sigma _ { i j } ^ { 2 }$ is the predetermined variance.4
• Deep Matrix Factorization (DMF) [89]. DMF extends MF by applying deep neural networks (DNNs) [96], i.e., $f _ { n n } ^ { u } , f _ { n n } ^ { v } : \mathbb { R } ^ { K }  \mathbb { R } ^ { K ^ { \prime } }$ , to $\mathbf { u } _ { i }$ and ${ \bf v } _ { j }$ , where the ratings are assumed to be generated as $r _ { i j } \sim N ( f _ { n n } ^ { u } ( \mathbf { u } _ { i } ) ^ { T } \cdot f _ { n n } ^ { v } ( \mathbf { v } _ { j } ) , \dot { \sigma } _ { i j } ^ { 2 } )$ .
• Auto-encoder (AE)-based RSs [36, 83] model user $i \ ' _ { \mathrm { { S } } }$ ratings to all J items as $\mathbf { r } _ { i } \sim N ( f _ { n n } ^ { u } ( \mathbf { u } _ { i } ) , \pmb { \sigma } _ { i } ^ { 2 } \cdot \mathbf { I } _ { K } )$ , where $f _ { n n } ^ { u } : \mathbb { R } ^ { K }  \mathbb { R } ^ { J }$ is a DNN and item latent variables ${ \bf v } _ { j }$ for all items are implicit in last layer weights of the decoder [107].

In the training phase, the models learn the latent variables ${ \bf { u } } _ { i } , { \bf { v } } _ { j }$ and the associated function $f _ { n n }$ by fitting on the observed ratings $r _ { i j }$ (e.g., via maximum likelihood estimation, which essentially estimates the conditional distribution $p ( r _ { i j } | \mathbf { u } _ { i } , \mathbf { v } _ { j } )$ from the observational data [85]). Afterward, we can use them to predict new$k , \mathbf { e . g . } , \hat { r } _ { i k } ^ { \mathrm { M F } } = \mathbf { u } _ { i } ^ { T } \cdot \mathbf { v } _ { k }$ $\hat { r } _ { i k } ^ { \mathrm { D M F } } =$ $f _ { n n } ^ { u } ( \mathbf { u } _ { i } ) ^ { T } \cdot f _ { n n } ^ { v } ( \mathbf { v } _ { k } )$ $\hat { r } _ { i k } ^ { \mathrm { A E } } = f _ { n n } ^ { u } ( \mathbf { u } _ { i } ) _ { k }$ ones that best match users’ interests can be selected as recommendations.

Traditional collaborative filtering-based RSs reasons with correlations. Ideally, we would expect ${ \bf u } _ { i } , { \bf v } _ { j }$ , and $f _ { n n }$ to capture the causal influence of user interests and item attributes on ratings, i.e., what the rating would be if item $j$ is made exposed to user i [85]. However, since the collected rating data are observational rather than experimental, what actually learned by $\mathbf { u } _ { i } , \mathbf { v } _ { j }$ , and $f _ { n n }$ are the co-occurrence patterns in users’ past behaviors, which guarantee no causal implications. Consequently, spurious correlations and biases can be captured by the model, which will be amplified in future recommendations $[ 7 3 ]$ . Furthermore, the learned user latent variable $\mathbf { u } _ { i }$ generally entangles different factors that causally determine user interests. From this perspective, the explainability and generalization of these methods cannot be guaranteed.

## 10.2.2 Content-Based Recommender Systems

Personalized content-based RSs (CBRSs) estimate user interests based on the features of the items they have interacted with. These models typically encode user interests into user latent variables ${ \mathbf { u } } _ { i } \in \mathbb { R } ^ { K }$ and assume that the ratings are generated by matching user interests with item content, i.e., $r _ { i j } \sim  { N ( f ( \mathbf { u } _ { i } , \mathbf { f } _ { i } ^ { v } ) , \sigma _ { i j } ) }$ , where $f$ is a matching function. The training of personalized CBRSs follows similar steps as collaborative filtering, where $\mathbf { u } _ { i }$ and $f$ are learned by fitting on the observed ratings (which essentially estimates the conditional distribution $p ( r _ { i j } | \mathbf { u } _ { i } , \mathbf { f } _ { j } ^ { v } )$ from the observational data), and new ratings can be predicted by $\hat { r } _ { i k } = f ( \mathbf { u } _ { i } , \bar { \mathbf { f } } _ { k } ^ { v } )$ . The key step of building a CBRS is to create item features $\mathbf { f } _ { j } ^ { v }$ that can best reflect user interests, which crucially depends on the item being recommended. For example, for micro-videos, the visual, audio, and textual modalities are comprehensively considered such that users’ interest in different aspects of a micro-video can be well captured [81].

Traditional content-based RSs cannot model the causal influence of user interests $\mathbf { u } _ { i }$ and item content $\mathbf { f } _ { j } ^ { v }$ on user rating $r _ { i j }$ . The reason is that factors other than users’ interests in the item content, such as users’ being deceived by clickbaits $( \mathrm { e . g . }$ , sensational titles of micro-videos) [72], can create an undesirable association between item content $\mathbf { f } _ { j } ^ { v }$ and user ratings $r _ { i j }$ in the

(continued)observed dataset, where the bias can be captured by the user latent variables $\mathbf { u } _ { i }$ and the matching function $f _ { : }$ , and perpetuates into future recommendations.

## 10.2.3 Hybrid Recommendation

Hybrid RSs combine user/item side information with collaborative filtering to enhance the recommendations. A commonly used hybrid strategy is to augment user and item latent variables $\mathbf { u } _ { i }$ and ${ \bf v } _ { j }$ with user/item side information ${ \bf f } _ { i } ^ { u }$ and $\mathbf { f } _ { j } ^ { v }$ i n existing collaborative filtering methods by replacing $\mathbf { u } _ { i }$ and ${ \bf v } _ { j }$ with $\mathbf { u } _ { i } ^ { + } = [ \mathbf { u } _ { i } | | \mathbf { f } _ { i } ^ { u } ]$ and $\mathbf { v } _ { j } ^ { + } = [ \mathbf { v } _ { j } | | \mathbf { f } _ { j } ^ { v } ]$ in MF, DMF, and AE-based RSs, where [·||·] represents vector concatenation [27, 108]. The dimensions of $\mathbf { u } _ { i }$ and ${ \bf v } _ { j }$ that encode the collaborative information are adjusted accordingly to make ${ \mathbf { u } } _ { i } ^ { + }$ and $\mathbf { v } _ { j } ^ { + }$ compatible in the model. Another important class of hybrid RS is the factorization machine (FM) [51] and its extensions like [21, 31], which can be viewed as learning a bilinear function $f _ { f m }$ where the ratings are generated by $r _ { i j } \sim N ( f _ { f m } ( \mathbf { u } _ { i } , \mathbf { v } _ { j } , \mathbf { f } _ { i } ^ { u } , \mathbf { f } _ { j } ^ { v } ) , \sigma _ { i j } ^ { 2 } )$ .

Simple hybrid strategies cannot break the correlational reasoning limitation of collaborative filtering and content-based RSs, because the objective of the hybridization is still to improve the models’ fitting on the observational user historical behaviors (i.e., estimating conditional distribution $p ( r _ { i j } | \mathbf { u } _ { i } , \mathbf { v } _ { j } , \mathbf { f } _ { i } ^ { u } , \mathbf { f } _ { j } ^ { v } )$ from the data), where the causal reasons that lead to the observed user behaviors are not considered. However, the idea of introducing extra user/item side information is important for building causal RSs. The reason is that, combined with the domain knowledge of human experts, the side information can help form more comprehensive causal relations among the variables of interests, such as user interests, item attributes, historical ratings, and other important covariates that may lead to spurious correlations and biases, which is usually a crucial step for causal reasoning in recommendations.

## 10.3 Causal Recommender Systems: Preliminaries

In the previous section, we discussed the recommendation strategies of the traditional RSs and their limitations due to correlational reasoning on observational user behaviors. In this section, we introduce two causal inference frameworks, i.e., Rubin’s potential outcome framework (also known as the Rubin causal model, RCM) [23] and Pearl’s structural causal model (SCM) [49], in the context of RSs, aiming to provide a theoretically rigorous basis for reasoning with correlation and causation in recommendations. We show that both RCM and SCM are powerful frameworks to build RSs with causal reasoning ability (i.e., causal RSs), but they are best suited for different tasks and questions. The discussions in this section serve as the foundation for more in-depth discussions of state-of-the-art causal RS models in Sect. 10.4.

## 10.3.1 Rubin’s Potential Outcome Framework

## 10.3.1.1 Motivation of Applications in RSs

To understand the correlational reasoning nature of traditional RSs, we note that naively fitting models on the observed ratings can only answer the question “what the rating would be if we observe an item was exposed to the user.” Since item exposure is not randomized in the collected dataset,5 the predicate “the item was exposed to the user” per se contains extra information about the user–item pair (e.g., the item could be more popular than other non-exposed items), which cannot be generalized to the rating predictions of arbitrary user–item pairs. Therefore, what RS asks is essentially an interventional question (and therefore a causal inference question), i.e., what the rating would be if an item is made exposed to the user. To address this question, RCM-based RSs draw inspiration from clinical trials, where exposing a user to an item is compared to subjecting a patient to a treatment, and the user ratings are analogous to the outcomes of the patients after the treatment [60, 82]. Accordingly, RCM-based RSs aim to estimate the causal effects of the treatments (exposing a user to an item) on the outcomes (user ratings), despite the possible correlations between the treatment assignment and the outcome observations [60].

## 10.3.1.2 Definitions and Objectives

We first introduce necessary symbols and definitions to connect RCM with RSs. We consider the unit as the user–item pair $( i , j )$ that can receive the treatment “exposing user i to item $j ^ { \dag }$ , and the population as all user–item pairs $\mathcal { P } O = \{ ( i , j ) , 1 \leq i , j \leq$ $I , J \} [ 6 ]$ . We start by using a binary scalar $a _ { i j }$ to denote the exposure status of item j for user i, i.e., the assigned treatment. We further define the rating potential outcome $r _ { i j } ( a _ { i j } = 1 )$ as user i’s rating to item j if the item is made exposed to the user and $r _ { i j } ( a _ { i j } = 0 )$ as the rating if the item is not exposed [78]. For user i, if she rated item j , we observe $r _ { i j } ( a _ { i j } = 1 ) = r _ { i j }$ . Otherwise, we observe the baseline potential outcome $r _ { i j } ( a _ { i j } = 0 ) = 0$ , which is usually ignored in debias-oriented causal RS research [60, 65].6 Similar to clinical trials, we can define the treatment group $\mathcal { T } = \{ ( i , j ) : a _ { i j } = 1 \}$ as the set of user-item pairs where user i is exposed to item j , and define the non-treatment group $N \mathcal { T } = \{ ( i , k ) : a _ { i k } = 0 \}$ accordingly. The purpose of RSs, under the RCM framework, can be framed as utilizing the observed ratings from units in the treatment group T to unbiasedly estimate the rating potential outcomes for units from the population PO, despite the possible correlations between item exposures $a _ { i j }$ and user ratings $r _ { i j }$ in the collected data.

## 10.3.1.3 Causal Analysis of Traditional RSs

Traditional RSs naively train a rating prediction model that best fits the ratings in the treatment group $\mathcal { T }$ (e.g., via maximum likelihood introduced in Sect. 10.2) to estimate the unobserved rating potential outcomes $r _ { i j } ( a _ { i j } = 1 )$ for user-item pairs in NT [11], which neglect the fact that exposure bias can lead to a systematic difference in the distribution of $r _ { i j } ( a _ { i j } = 1 )$ between T and NT. For example, users tend to rate items they like in reality, which could lead to the following spurious correlation between item exposure $a _ { i j }$ and rating potential outcome $r _ { i j } ( a _ { i j } = 1 )$ :

$$
p (r _ {i j} (a _ {i j} = 1) \text {   is   high } | a _ {i j} = 1) > p (r _ {i j} (a _ {i j} = 1) \text {   is   high } | a _ {i j} = 0), \tag {10.1}
$$

i.e., users who have rated an item j may have systematically higher ratings than users who haven’t rated it yet. In this case, traditional RSs may have a tendency to overestimate the ratings for units in NT (see Fig. 10.2 for an intuitive example).

Theoretically, RCM attributes the exposure bias in the collected dataset to the violation of the unconfoundedness assumption [23] defined as follows:

$$
r _ {i j} (a _ {i j} = 1) \perp a _ {i j}. \tag {10.2}
$$

The rationale is that, if Eq. (10.2) holds, the exposure of user i to item $j ~ ( \mathrm { i . e . , } ~ a _ { i j } )$ is independent of the rating potential outcome $r _ { i j } ( a _ { i j } = 1 )$ , which implies that $r _ { i j } ( a _ { i j } = 1 )$ in $\mathcal { T }$ and NT follows the same distribution. Therefore, the exposure of the items is randomized, and exposure bias such as Eq. (10.1) will not exist [78].

## 10.3.1.4 Potential Outcome Estimation with the RCM Framework

One classic solution from the RCM-based framework to address the exposure bias is that we find user and item covariates C, such that in each data stratum specified by $C = \mathbf { c } .$ , users’ exposure to items is randomized [23]. The property of the covariates C can be formulated as the conditional unconfoundedness assumption as follows:

$$
r _ {i j} (a _ {i j} = 1) \perp a _ {i j} \mid \mathbf {c}. \tag {10.3}
$$

C is sometimes non-rigorously referred to as confounder in the literature, but we will see its formal definition in the next subsection. If Eq. (10.3) holds, the item exposures are independent of the rating potential outcomes in each data stratum specified by $C = \mathbf { c } .$ , and the exposure bias can be attributed solely to the discrepancy in the distribution of the covariates $C \ = \ \mathbf { c }$ between the treatment group $\mathcal { T }$ and the population , i.e., $p ( \mathbf { c } | a _ { i j } = 1 )$ and $p ( \mathbf { c } ) ^ { 7 }$ Therefore, we can reweight the observed ratings in  based on the covariates C to address the bias, such that they can be viewed as pseudo-randomized samples. This leads to inverse propensity weighting (IPW), which eliminates the exposure bias from the data’s perspective [60]. In addition, we can also adjust the influence of C in the RS model, where the exposure bias is addressed from the model side [78]. Both methods will be discussed in Sect. 10.4.1.1.

## •! Attention: Extra Assumptions Required by Most RCM-based RSs

In addition to unconfoundedness, most RCM-based RSs need two extra assumptions to identify the causal effects of item exposures on ratings: (1) The stable unit treatment assumption (SUTVA), which states that items exposed to one user cannot affect ratings of another user. (2) The positivity assumption, which states that every user has a positive chance of being exposed to every item [23]. For RCM-based causal RSs introduced in this chapter, these two assumptions are tacitly accepted.

## 10.3.2 Pearl’s Structural Causal Model

## 10.3.2.1 Motivation of Applications in RS

Different from RCM that uses rating potential outcomes to reason with causality and attributes the biases in observed user behaviors to non-randomized item exposures, Pearl’s structural causal model (SCM) delves deep into the causal mechanism that generates the observed outcomes (and biases) and represents it with a causal graph $G = ( N , { \mathcal { E } } )$ . The nodes N specify the variables of interests, which in the context of RS could be user interests U , item attributes V , observed ratings R, and other important covariates C, such as item popularity, user features.8 The directed edges between nodes represent their causal relations determined by researchers’ domain knowledge. Each node $X ~ \in ~ N$ is associated with a structural equation $p _ { G } ( X | P a ( X ) ) , \}$ which describes how the parent nodes $P a ( X )$ causally influence X (i.e., the response of X when setting nodes in $P a ( X )$ to specific values)

Although RCM and SCM are generally believed to be fundamentally equivalent [49], both have their unique advantages. Compared to RCM, the key advantage of SCM is that causal graph offers an intuitive and straightforward way to encode and communicate domain knowledge and substantive assumptions of researchers, which is beneficial even for the RCM-based RSs [78]. Furthermore, SCM is more flexible as it can represent and reason with the causal effects between any subset of nodes in the causal graph (e.g., between two causes U, V and one outcome R), as well as the causal effects along specific paths (e.g., $U  R$ and $U _ { c } \to R )$ . Therefore, SCMs are broadly applicable to multiple problems in RSs (not limited to exposure bias), such as clickbait bias, unfairness, entanglement, and domain adaptation [15].

## •! Attention: Two Caveats of SCM-based Causal RSs.

There are two caveats of SCM-based causal RSs. (1) Causal graphs for RSs often involve user, item latent variables U , V that encode user interests and item attributes. Most works infer them alongside the estimation of structural equations and treat them as if they were observed when analyzing the causal relations. Alternatively, this can be viewed as representing users and items with their IDs (i.e., i and $j )$ in the causal graph and subsuming the embedding process into the structural equations [87]. (2) Generally, the causal graph should describe the causal mechanism that generates the observed data, because it allows us to distinguish invariant, causal relations from undesirable correlations. For example, we may argue that item popularity C should be determined by item attributes V , i.e., $V  C$ . But to describe the generation of the observed ratings, causal relation $C  V$ is usually assumed instead as item popularity causally influences the exposure probability of each item [98].

## 10.3.2.2 Atomic Structures of Causal Graphs

The structure of causal graphs represents researchers’ domain knowledge regarding the causal generation process of the observational data, which is the key to distinguishing stable, causal relations from other undesirable correlations between variables of interest. Here, we use a generic causal graph applicable to RSs in Fig. 10.3a as a running example to illustrate three atomic graph structures (Fig. 10.3b–d):

![image_61](images/image_61.png)

```mermaid
graph TD
  U --> R
  Uc --> R
  V --> R
  Cu --> R
  Cv --> R
```

(a) A generic causal graph for RS

![image_62](images/image_62.png)

```mermaid
graph TD
  A["Cu"] --> B["U"]
  B --> C["R"]
  D["Cu"] --> E["U"]
  D --> F["R"]
  G["U"] --> H["R"]
  I["Uc"] --> J["R"]
```

(c) The fork structure

![image_63](images/image_63.png)

```mermaid
graph TD
  U["U"] --> R["R"]
  Uc["Uc"] --> R
```

(d) The V-structure  
Fig. 10.3 (a): A generic causal graph for RS that depicts the causal influence of user interests U , user conformity to the popularity trend $U _ { c } ,$ , and item attributes V on the observed ratings R. Specifically, the causal paths $U  R$ and $V  R$ are confounded by $C _ { u }$ and $C _ { v } .$ , which represent user features and item popularity, respectively. (b)–(d): Three atomic structures identified from (a)

• Chain, e.g., $C _ { u } \to U \to R$ . In a chain, the successor node is assumed to be causally influenced by the ancestor nodes. In the example, U is a direct cause of R, whereas $C _ { u }$ indirectly influences R via U as a mediator.
• Fork, e.g., $U \left. C _ { u } \right. R$ . In the fork, $C _ { u }$ is called a confounder as it causally influences two children U and R. From a probabilistic perspective, U and R are not independent unless conditioned on the confounder $C _ { u }$ [26]. This leads to the tricky part of a fork structure, i.e., confounding effect [49], where an unobserved $C _ { u }$ can lead to spurious correlations between U and R.
• V-structure, e.g., $U \ \to \ R \ \gets \ U _ { c }$ . In the V-structure, R is called a collider because it is under the causal influence of two parents, i.e., U and $U _ { c }$ . An interesting property of the V-structure is the colliding effects [49], where observing R creates a dependence on U and $U _ { c }$ , even if they are marginally independent.

Confounders can lead to non-causal dependencies among variables in the observational dataset. This could introduce bias in traditional RSs, where the confounding effects are mistaken as causal relations. Confounding bias is a generic problem in RSs [85], which will be further analyzed in the following subsections. In addition, abstracted V-structure usually leads to the entanglement of causes, which could jeopardize the explainability of RSs. For example, a user’s purchase of an item may be due to her interest, i.e., U , or her conformity to the popularity trend, i.e., $U _ { c }$ . Since most RSs summarize both into a user latent variable $U$ , the V-structure $U \to R \gets U _ { c }$ is abstracted away, where the two causes of the purchase cannot be distinguished.

## 10.3.2.3 Causal Analysis of Traditional RSs

In this section, we investigate the susceptibility of traditional collaborative filteringbased RSs to the confounding bias. As discussed in Sect. 10.2.1, a commonality of these models is that they estimate conditional distribution $p ( r _ { i j } | \mathbf { u } _ { i } , \mathbf { v } _ { j } )$ from observed ratings and use it to predict new ratings. For $p ( r _ { i j } | \mathbf { u } _ { i } , \mathbf { v } _ { j } )$ to represent the causal influence of user interests $\mathbf { u } _ { i }$ and item attributes ${ \bf v } _ { j }$ on ratings $r _ { i j }$ (which, in the context of collaborative filtering, means the rating of any arbitrary item j that is made exposed to user i [85]), the causal graph $G _ { 1 }$ of Fig. 10.4a is tacitly assumed, i.e., no unobserved confounders for causal paths $U  R$ and $V  R .$ .10

However, in reality, both $U  R [ 7 3 , 8 0 ]$ and V  R [3, 10] can be confounded, where the confounding effects can be implicitly captured by $p ( r _ { i j } | \mathbf { u } _ { i } , \mathbf { v } _ { j } )$ that bias future recommendations. To reveal the bias, we consider the scenario where the causal path $V \  \ R$ is confounded by $C _ { v }$ (e.g., item popularity). We assume the causal path $C _ { v }  V$ denotes the causal influence of $C _ { v }$ on the exposure probability of item V [98]. In this case, the observed ratings are generated according to the causal graph $G _ { 2 }$ in Fig. 10.4b. Utilizing the law of total probability, the conditional distribution $p ( r _ { i j } | \mathbf { u } _ { i } , \mathbf { v } _ { j } )$ estimated from the confounded data can be calculated as:

![image_64](images/image_64.png)

```mermaid
graph TD
  U["U"] --> R["R"]
  V["V"] --> R
```

(a) SCM assumed by non-causal RS

![image_65](images/image_65.png)

```mermaid
graph TD
  U --> R
  V --> R
  R --> Cv
  Cv --> R
```

(b) Confounded true SCM

![image_66](images/image_66.png)

```mermaid
graph TD
  U --> R
  do(V) --> R
  Cv --> R
```

(c) SCMunder intervention  
Fig. 10.4 (a): SCM assumed by non-causal collaborative filtering-based RS. (b): The confounded SCM that depicts the true data generation process. (c): SCM under intervention do(V )

$$
p (r _ {i j} | \mathbf {u} _ {i}, \mathbf {v} _ {j}) = \sum_ {\mathbf {c}} p (\mathbf {c} | \mathbf {v} _ {j}) \cdot p _ {G _ {2}} (r _ {i j} | \mathbf {u} _ {i}, \mathbf {v} _ {j}, \mathbf {c}) = \mathbb {E} _ {p (C _ {v} | \mathbf {v} _ {j})} [ p _ {G _ {2}} (r _ {i j} | \mathbf {u} _ {i}, \mathbf {v} _ {j}, C _ {v}) ]. \tag {10.4}
$$

The issue of Eq. (10.4) is that, the $p ( \mathbf { c } | \mathbf { v } _ { j } )$ term is not causal (as we only have an edge $C _ { v } \  \ V$ in the causal graph but not $V  C _ { v } )$ . In fact, $p ( \mathbf { c } | \mathbf { v } _ { j } )$ represents abductive reasoning because it infers the cause c (e.g., item popularity) from the effect ${ \bf v } _ { j }$ (i.e., item $j$ is exposed to user i) and uses the inferred c to support the prediction of the rating $r _ { i j }$ . However, such reasoning cannot be generalized to the rating prediction of an arbitrary item ${ \bf v } _ { j }$ , i.e., an item that is made exposed to the user. In other words, uncontrolled confounder $C _ { v }$ leaves open a backdoor path (i.e., non-causal path) between V and R, such that non-causal dependence of R on V exists in the data, which can be captured by traditional RSs and bias future recommendations.11

## 10.3.2.4 Causal Reasoning with SCM

To calculate the causal effect of $\mathbf { u } _ { i }$ and ${ \bf v } _ { j }$ on $r _ { i j }$ , we should conduct intervention on U and V . This means that we set U, V to ${ \bf u } _ { i } , { \bf v } _ { j }$ regardless of the values of their parent nodes in the causal graph, including the confounder $C _ { v }$ (because these nodes determine the exposure of item j to user i in the observed data). SCM denotes the intervention with do-operator as $p ( r _ { i j } | d o ( \mathbf { u } _ { i } , \mathbf { v } _ { j } ) )$ to distinguish it from the conditional distribution $p ( r _ { i j } | \mathbf { u } _ { i } , \mathbf { v } _ { j } )$ that reasons with correlations in the observational data. Consider again the causal graph $G _ { 2 }$ illustrated in Fig. 10.4b. The intervention on node V can be realized by removing all the incoming edges for node V and setting the structural equation $p _ { G _ { 2 } } ( V | C _ { v } )$ deterministically as $V = \mathbf { v } _ { j }$ , while other structural equations remain intact (Fig. 10.4c). If the confounder $C _ { v }$ can be determined and measured for each item, the interventional distribution $p ( r _ { i j } | d o ( \mathbf { u } _ { i } , \mathbf { v } _ { j } ) )$ can be directly calculated from the confounded data via backdoor adjustment [49] as:

$$
p (r _ {i j} | d o (\mathbf {u} _ {i}, \mathbf {v} _ {j})) = \sum_ {\mathbf {c}} p _ {G _ {2}} (\mathbf {c}) \cdot p _ {G _ {2}} (r _ {i j} | \mathbf {u} _ {i}, \mathbf {v} _ {j}, \mathbf {c}) = \mathbb {E} _ {p _ {G _ {2}} (C _ {v})} [ p _ {G _ {2}} (r _ {i j} | \mathbf {u} _ {i}, \mathbf {v} _ {j}, C _ {v}) ], \tag {10.5}
$$

which, compared with Eq. (10.4), blocks the abductive inference of c from ${ \bf v } _ { j }$ , such that the causal influence of $\mathbf { u } _ { i } , \mathbf { v } _ { j }$ on $r _ { i j }$ can be properly identified.

Backdoor adjustment requires all confounders to be determined and measured in advance, but there are other SCM-based causal inference methods that can estimate causal effects with unknown confounders, and we refer readers to [86, 106] for details. Moreover, causal graphs allow us to conduct other types of causal reasoning based on the encoded causal knowledge, such as debiasing for non-confounderinduced biases (e.g., clickbait bias and unfairness), causal disentanglement, and causal generalization [102]. These will be thoroughly discussed in the next section.

## 10.4 Causal Recommender Systems: The State of the Art

Based on the preliminary knowledge of RSs and causal inference discussed in previous sections, we are ready to introduce the state-of-the-art causal RSs. Specifically, we focus on three important topics, i.e., bias mitigation, explainability promotion, and generalization improvement, as well as their interconnections, where various limitations of traditional RSs due to correlational reasoning can be well addressed.

## 10.4.1 Causal Debiasing for Recommendations

The correlational reasoning of traditional RSs can inherit multiple types of biases in the observational user behaviors and amplify them in future recommendations [11]. The biases may result in various consequences, such as the discrepancy between offline evaluation and online metrics, loss of diversity, reduced recommendation quality, and offensive recommendations Causal inference can distinguish stable causal relations from spurious correlations and biases that could negatively influence the recommendations, such that the robustness of recommendations can be improved.

## 10.4.1.1 Exposure Bias

Exposure bias in RSs broadly refers to the bias in observed ratings due to nonrandomized item exposures. From the RCM’s perspective, exposure bias can be defined as the bias where users are favorably exposed to items depending on their expected ratings for them (i.e., rating potential outcomes) [65]. Exposure bias occurs due to various reasons, such as users’ self-search or the recommendation of the previous RSs [37], which leads to the down-weighting of items less likely to be exposed to users. Since item exposures can be naturally compared with treatments in clinical trials, we discuss the debiasing strategies with the RCM framework.

Inverse Propensity Weighting (IPW) IPW-based causal RSs aim to reweight the biased observed ratings $r _ { i j }$ for user–item pairs in the treatment group, i.e., $\mathcal { T } =$ $\{ ( i , j ) : a _ { i j } = 1 \}$ , to create pseudo-randomized samples [58] for unbiased training of RS models that aim to predict the rating potential outcomes $r _ { i j } ( a _ { i j } = 1 )$ for the population ${ \mathcal { P } } O = \{ ( i , j ) , 1 \le i , j \le I , J \}$ . Intuitively, we can set the weight of $r _ { i j }$ for units in to be the inverse of item $j ^ { \prime } { \bf s }$ exposure probability to user $i ,$ such that under-exposed items can be up-weighted and vice versa. If for each user–item pair, the covariates c that satisfy the conditional unconfoundedness assumption in Eq. (10.3) are available, the exposure probability $e _ { i j }$ can be unbiasedly estimated from c via

$$
e _ {i j} = p (a _ {i j} = 1 | \mathbf {c}) = \mathbb {E} [ a _ {i j} | \mathbf {c} ], \tag {10.6}
$$

which is formally known as propensity score in causal inference literature [55].

## Background: The Balancing Property of Propensity Scores.

Propensity scores have the following property called balancing [23, 99], which is the key to proving the unbiasedness of IPW-based RSs:

$$
\begin{array}{l} \mathbb {E} \Big [ \frac {r _ {i j}}{e _ {i j}} \Big | a _ {i j} = 1 \Big ] = \mathbb {E} \Big [ \frac {r _ {i j} \cdot a _ {i j}}{e _ {i j}} \Big ] = \mathbb {E} \Big [ \mathbb {E} \Big [ \frac {r _ {i j} \cdot a _ {i j}}{e _ {i j}} \Big | \mathbf {c} \Big ] \Big ] \\ = \mathbb {E} \left[ \mathbb {E} \left[ \frac {r _ {i j} (a _ {i j} = 1) \cdot a _ {i j}}{e _ {i j}} | \mathbf {c} \right] \right] \stackrel {(a)} {=} \mathbb {E} \left[ \frac {\mathbb {E} [ r _ {i j} (a _ {i j} = 1) \mid \mathbf {c} ] \cdot \mathbb {E} [ a _ {i j} \mid \mathbf {c} ]}{e _ {i j}} \right] \tag {10.7} \\ = \mathbb {E} \Big [ \frac {\mathbb {E} [ r _ {i j} (a _ {i j} = 1) \mid \mathbf {c} ] \cdot e _ {i j}}{e _ {i j}} \Big ] = \mathbb {E} [ r _ {i j} (a _ {i j} = 1) ], \\ \end{array}
$$

where the step (a) follows the conditional unconfoundedness assumption in Eq. (10.3).

We first discuss the implementation of IPW-based RS and its unbiasedness if user and item covariates c that satisfy Eq. (10.3) are available and the propensity scores $e _ { i j }$ can be calculated exactly as Eq. (10.6). We denote the rating predictor of an RS that aims to predict the rating potential outcome $r _ { i j } ( a _ { i j } = 1 )$ as $\hat { r } _ { i j }$ and assume $r _ { i j } ( a _ { i j } = 1 )$ ) follows the unit-variance Gaussian distribution. Ideally, we would like $\hat { r } _ { i j }$ to maximize the log-likelihood on the rating potential outcomes $r _ { i j } ( a _ { i j } = 1 )$ for all user-item pairs in PO, which is equivalent to the minimization of the mean squared error (MSE) loss between $\hat { r } _ { i j }$ and $r _ { i j } ( a _ { i j } = 1 )$ as follows:

$$
\mathcal {L} ^ {\text { True }} = \frac {1}{I \times J} \sum_ {i, j} (\hat {r} _ {i j} - r _ {i j} (a _ {i j} = 1)) ^ {2}. \tag {10.8}
$$

However, since $r _ { i j } ( a _ { i j } = 1 )$ is unobservable for user–item pairs in the nontreatment group $\boldsymbol { N } \boldsymbol { \mathcal { T } } , \boldsymbol { \mathcal { L } } ^ { \mathrm { T r u e } }$ is impossible to calculate. Therefore, traditional RSs only maximize the log-likelihood of the observed ratings for user–item pairs in the treatment group , which leads to the empirical MSE loss as follows:

$$
\mathcal {L} ^ {\text { Obs }} = \frac {1}{| (i , j) : a _ {i j} = 1 |} \sum_ {(i, j): a _ {i j} = 1} (\hat {r} _ {i j} - r _ {i j}) ^ {2}, \tag {10.9}
$$

where $| ( i , j ) : a _ { i j } = 1 |$ is the number of observed ratings. When exposure bias exists, item exposure $a _ { i j }$ depends on the rating potential outcome $r _ { i j } ( a _ { i j } = 1 )$ ) . Therefore, $\mathcal { L } ^ { \mathrm { O b s } }$ is a biased estimator for $\mathcal { L } ^ { \mathrm { T r u e } }$ , because the observed ratings for user-item pairs in the treatment group T are biased samples from the rating potential outcomes of the population $\mathcal { P } O$ (see Figs. 10.5a,b for an example). To remedy the bias, IPW-based causal RSs reweight the observed ratings $r _ { i j }$ in  by the inverse of $\frac { 1 } { e _ { i j } }$

$$
\mathcal {L} ^ {\mathrm{IPW}} = \frac {1}{I \times J} \sum_ {(i, j): a _ {i j} = 1} \frac {1}{e _ {i j}} \cdot (\hat {r} _ {i j} - r _ {i j}) ^ {2}. \tag {10.10}
$$

The proof for the unbiasedness of $\mathcal { L } ^ { \mathrm { I P W } }$ for $\mathcal { L } ^ { \mathrm { T r u e } }$ can be achieved by utilizing the balancing property of propensity scores in Eq. (10.7), where we substitute $( \hat { r } _ { i j } - r _ { i j } ) ^ { 2 }$ for $r _ { i j }$ in the LHS of Eq. (10.7) and treat the rating predictor $\hat { r } _ { i j }$ as constant [60]. We also provide a toy example in Fig. 10.5 to intuitively show the calculation of $e _ { i j }$ , the biasedness of $\mathcal { L } ^ { \mathrm { O b s } }$ and the unbiasedness of $\mathcal { L } ^ { \mathrm { I P W } }$ . The objective for IPW-based RSs defined in Eq. (10.10) is model-agnostic. Therefore, it is applicable to all traditional RSs we introduced in Sect. 10.2. For example, for MF-based RSs, we can plug in $\hat { r } _ { i j } ^ { \mathrm { M F } } = \mathbf { u } _ { i } ^ { T } \cdot \mathbf { v } _ { j }$ rˆij $\hat { r } _ { i j } ^ { \mathrm { D M F } } = f _ { n n } ^ { u } ( \mathbf { u } _ { i } ) ^ { T } \cdot f _ { n n } ^ { v } ( \mathbf { v } _ { j } )$

In practice, since the conditional unconfoundedness assumption in Eq. (10.3) is untestable, it is usually infeasible to calculate the exact value of $e _ { i j }$ based on user/item covariates that satisfy Eq. (10.3). Nevertheless, we can still calculate approximate propensity scores $\tilde { e } _ { i j }$ and reweight the observed ratings by $1 / \tilde { e } _ { i j }$ , but the unbiasedness of Eq. (10.10) after the reweighting cannot be guaranteed. Here we introduce two strategies for the approximate estimation. If user/item features $\mathbf { f } _ { i } ^ { u }$ and $\mathbf { f } _ { j } ^ { v }$ are available, $\tilde { e } _ { i j }$ can be estimated with logistic regression [60] as follows:

$$
\tilde {e} _ {i j} ^ {\mathrm{LR}} = \text { Sigmoid } \left(\left(\sum_ {k} w _ {k} ^ {u} f _ {i k} ^ {u}\right) + \left(\sum_ {k} w _ {k} ^ {v} f _ {j k} ^ {v}\right) + b _ {i} + b _ {j}\right), \tag {10.11}
$$

where Sigmoid $( x ) = ( 1 + \exp ( - x ) ) ^ { - 1 }$ , $w _ { k } ^ { u }$ and $w _ { k } ^ { v }$ are the regression coefficients, and $b _ { i } , b _ { j }$ are the user and item-specific offsets, respectively. If user/item features $\mathbf { f } _ { i } ^ { u }$ and ${ \bf f } _ { j } ^ { v }$ are not available, we can crudely approximate $e _ { i j }$ based on the exposure data alone. For example, we can estimate $\tilde { e } _ { i j }$ with Poisson factorization [35] as:

$$
\tilde {e} _ {i j} ^ {\mathrm{PF}} \approx 1 - \exp \left(- \boldsymbol {\pi} _ {i} ^ {T} \cdot \boldsymbol {\gamma} _ {j}\right), \tag {10.12}
$$

where $\pmb { \pi } _ { i }$ and $\gamma _ { j }$ are trainable user and item embeddings with Gamma prior, and they can be inferred from the exposure data as discussed in [19]. Additional strategies to calculate the propensity scores can be found in [4, 79, 97, 103].

The advantage of IPW is that the unbiasedness of Eq. (10.10) for rating potential outcome estimation can be guaranteed if the propensity scores $e _ { i j }$ are correctly estimated. However, the accuracy of estimated propensity scores relies heavily on the domain knowledge and expertise of human experts. In addition, IPW suffers from a large variance and numerical instability issues, especially when the estimated propensity scores $e _ { i j }$ are very small. Therefore, variance reduction techniques such as clipping and multitask learning are usually applied to improve the stability of the training dynamics [7, 57, 112]. IPW is widely adopted in industrial applications such as click-through rate estimation and conversion rate estimation [69, 105], etc.

Substitute Confounder Adjustment IPW-based RSs address exposure bias from the data’s perspective: They reweight the biased observational dataset to create a pseudo randomized dataset that allows unbiased training of RSs. Confounder adjustment-based methods, in contrast, estimate confounders C that cause the exposure bias and adjust their effects in the rating prediction model (A simple adjustment strategy is to control C as extra covariates12 ). For the adjustment to be unbiased, classical causal inference requires the conditional unconfoundedess assumption in Eq. (10.3) hold, i.e., no unobserved confounders [23], which is generally infeasible in practice. Fortunately, recent advances in multi-cause causal inference [77] have shown that we can control substitute confounders estimated from item co-exposure data instead, where exposure bias can be mitigated with weaker assumptions.

We use ${ \bf a } _ { i } ~ = ~ [ a _ { i 1 } , \cdot \cdot \cdot ~ , a _ { i J } ]$ to denote the exposure status of all J items to user i, which can be viewed as a bundle treatment in clinical trials [113]. Wang et al. [78] showed that if we can estimate user-specific latent variables $\pi _ { i } .$ , such that conditional on $\pmb { \pi } _ { i }$ , the exposures of different items to the user are mutually independent, controlling $\pmb { \pi } _ { i }$ can eliminate the influence of multi-cause confounders ${ \bf c } _ { i } ^ { m } ~ ( \mathrm { i . e . }$ and ratings). A simple proof of the claim is that, if ${ \bf c } _ { i } ^ { m }$ can still influence $\mathbf { a } _ { i }$ and $\mathbf { r } _ { i }$ after conditioning on $\pmb { \pi } _ { i }$ , since ${ \bf c } _ { i } ^ { m }$ is an unobserved common cause for the exposure of different items, $a _ { i j }$ cannot be conditionally independent (see the discussion of the fork structure in Sect. 10.3.2.2), which renders a contradiction. The rigorous proof can be found in [77]. Wang et al. further assumed that $p ( \mathbf { a } _ { i } | \pmb { \pi } _ { i } ) =$ $\Pi _ { j } p ( a _ { i j } | \pmb { \pi } _ { i } ) = \Pi _ { j } \mathrm { P o i s s i o n } ( \pmb { \pi } _ { i } ^ { T } \cdot \pmb { \gamma } _ { j } )$ and used the Poisson factorization to infer $\pmb { \pi } _ { i }$ and $\gamma _ { j }$ . Afterward, exposure bias can be mitigated by controlling $\pmb { \pi } _ { i }$ as extra covariates in the RS model [23]. For example, controlling $\pmb { \pi } _ { i }$ in MF-based RSs leads to the following adjustment:

$$
r _ {i j} ^ {\text { adj }} (a _ {i j} = 1) \sim \mathcal {N} \Big (\underbrace {\mathbf {u} _ {i} ^ {T} \cdot \mathbf {v} _ {j}} _ {\text { user   interests }} + \underbrace {\sum_ {k} w _ {k} \pi_ {i k}} _ {\text { adj.   for   expo.   bias }}, \sigma_ {i j} ^ {2} \Big). \tag {10.13}
$$

The property of propensity scores can be utilized to further simplify Eq. (10.13): If unconfoundedness in Eq. (10.3) holds for $C \ = \ \pi _ { i }$ , it will also hold for $C ^ { \prime } =$ $\tilde { e } _ { i j } = p ( a _ { i j } | \pmb { \pi } _ { i } )$ [55]. Therefore, we can control the approximate propensity scores estimated by $\pmb { \pi } _ { i }$ , i.e., $\tilde { e } _ { i j } ~ = ~ \pmb { \pi } _ { i } ^ { T } \cdot \pmb { \gamma } _ { j }$ , which leads to the simplified adjustment formula:

$$
r _ {i j} ^ {\mathrm{adj}} (a _ {i j} = 1) \sim \mathcal {N} \Big (\mathbf {u} _ {i} ^ {T} \cdot \mathbf {v} _ {j} + w _ {i} \cdot \tilde {e} _ {i j}, \sigma_ {i j} ^ {2} \Big), \tag {10.14}
$$

where $w _ { i }$ is a user-specific coefficient that captures the influence of $\tilde { e } _ { i j }$ on ratings.

Despite the success in addressing exposure bias with weaker assumptions, one limitation of the above method is that, since Poisson factorization is a shallow model, it may fail to capture the complex influences of multi-cause confounders on item co-exposures. To address this problem, recent works have introduced deep neural networks (DNNs) to infer the user-specific substitute confounders $\pmb { \pi } _ { i }$ from bundle treatment $\mathbf { a } _ { i }$ [43, 109]. These methods generally assume that ${ \bf a } _ { i }$ are generated $\pmb { \pi } _ { i }$ $p ( \mathbf { a } _ { i } | \pmb { \pi } _ { i } )$ $f _ { n n } ^ { \mathrm { e x p } }$ as:

$$
p (\mathbf {a} _ {i} | \boldsymbol {\pi} _ {i}) = \Pi_ {j} \text { Bernoulli } (\text { Sigmoid } (f _ {n n} ^ {\exp} (\boldsymbol {\pi} _ {i}) _ {j})), \tag {10.15}
$$

where the intractable posterior of $\pmb { \pi } _ { i }$ is then approximated with a Gaussian distribution parameterized by DNNs via the variational auto-encoding Bayes algorithm [25], i.e., $q ( \pi _ { i } | \mathbf { a } _ { i } ) ~ = ~ N ( f _ { n n } ^ { \mu } ( \mathbf { a } _ { i } )$ , diag $( f _ { n n } ^ { \sigma ^ { 2 } } ( \mathbf { a } _ { i } ) ) )$ , where $f _ { n n } ^ { \mu }$ and $f _ { n n } ^ { \pmb { \sigma } ^ { 2 } }$ are two DNNs that calculate the posterior mean and variance (before diagonalization) of $\pmb { \pi } _ { i }$ . With deep generative models introduced to estimate the substitute confounders $\pi _ { i } .$ , nonlinear influences of multi-cause confounders on item exposures can be adjusted in the RS models, where exposure bias can be further mitigated in recommendations.

The key advantage of substitute confounder estimation-based causal RSs is that controlling confounders in the potential outcome prediction model generally leads to lower variance than IPW-based methods [78]. However, these models need to estimate substitute confounders $\pmb { \pi } _ { i }$ from the item co-exposures and introduce extra parameters in the RS models to adjust their influences, which may incur extra bias if the confounders and the parameters are not correctly estimated. In addition, exposure bias due to single-cause confounders cannot be addressed by these methods.

## 10.4.1.2 Popularity Bias

Popularity bias can be viewed as a special kind of exposure bias where users are overly exposed to popular items [2, 64]. Therefore, it can be addressed with techniques introduced in the previous section, especially the IPW-based methods [111]. The reason is that, if we define the popularity of an item as its exposure rate:

$$
m _ {j} = \frac {\sum_ {i} a _ {i j}}{\sum_ {j} \sum_ {i} a _ {i j}}, \tag {10.16}
$$

we can view $m _ { j }$ as pseudo-propensity scores and use IPW to reweight the observed ratings. Alternatively, we can also analyze and address popularity bias with the structural causal model (SCM), where the causal mechanism that generates the observed ratings under the influence of item popularity is deeply investigated.

The discussion is mainly based on the popularity-bias deconfounding (PD) algorithm proposed in [98]. PD assumes that the relations among user interests $\mathbf { u } _ { i }$ , item latent attributes ${ \bf v } _ { j }$ , item popularity $m _ { j }$ , and observed ratings $r _ { i j }$ can be represented by the causal graph illustrated in Fig. 10.6, where item popularity can be clearly identified as a confounder that spuriously correlates the item attributes and the user ratings. PD aims to eliminate such spurious correlations with backdoor adjustment, such that the causal influences of $\mathbf { u } _ { i }$ and ${ \bf v } _ { j }$ on $r _ { i j }$ (which represents users’ interests on intrinsic item properties) can be properly identified. Recall that backdoor adjustment with SCM involves two stages: (1) During the training phase, the relevant structural equations in the causal graph are estimated from the collected dataset. (2) Afterward, we adjust the influence of confounders according to Eq. (10.5) to remove the spurious correlations. Therefore, we need to estimate $p _ { G } ( r _ { i j } | \mathbf { u } _ { i } , \mathbf { v } _ { j } , m _ { j } )$ with the observed ratings $r _ { i j }$ and item popularity $m _ { j }$ and infer the latent variables $\mathbf { u } _ { i }$ and ${ \bf v } _ { j }$ . In PD, $p _ { G } ( r _ { i j } | \mathbf { u } _ { i } , \mathbf { v } _ { j } , m _ { j } )$ is modeled as a variant of MF as follows:

![image_67](images/image_67.png)

![image_68](images/image_68.png)

```mermaid
graph TD
  U --> R
  R --> do(V)
  R --> M
```

(b) SCM under intervention  
Fig. 10.6 (a): SCM that models item popularity. (b): SCM under intervention $d o ( V )$

$$
p _ {G} (r _ {i j} | \mathbf {u} _ {i}, \mathbf {v} _ {j}, m _ {j}) \propto \underbrace {\mathrm{Elu} (\mathbf {u} _ {i} ^ {T} \cdot \mathbf {v} _ {j})} _ {\text { user   interests }} \times \underbrace {m _ {j} ^ {\lambda}} _ {\text { pop.   bias }}, \tag {10.17}
$$

where λ is a hyper-parameter that denotes our belief toward the strength of influence of item popularity on ratings, and the function Elu (defined as $\operatorname { E l u } ( x ) ~ = ~ e ( x )$ if $ { \boldsymbol { { x } } } \_ { } { } < \ 0 \phantom { { . 0 } }$ else $x + 1 )$ makes the RHS of Eq. (10.17) a proper unnormalized probability density function. After ${ \bf u } _ { i } , ~ { \bf v } _ { j }$ are estimated from the datasets with Eq. (10.17), we conduct an intervention on the item node V in the causal graph (see Eq. (10.5)), where the spurious correlation due to item popularity can be eliminated with backdoor adjustment:

$$
p (r _ {i j} | d o (\mathbf {u} _ {i}, \mathbf {v} _ {j})) \propto \mathbb {E} _ {p (m _ {j})} [ \mathrm{Elu} (\mathbf {u} _ {i} ^ {T} \cdot \mathbf {v} _ {j}) \times m _ {j} ^ {\lambda} ] = \mathrm{Elu} (\mathbf {u} _ {i} ^ {T} \cdot \mathbf {v} _ {j}) \times \mathbb {E} _ {p (m _ {j})} [ m _ {j} ^ {\lambda} ]. \tag {10.18}
$$

Since the second term $\mathbb { E } _ { p ( m _ { j } ) } [ m _ { j } ^ { \lambda } ]$ in Eq. (10.18) is a constant and Elu is a monotonically increasing function, they have no influence on the ranking of the uninteracted items in the prediction phase. Therefore, we can drop them and use $\hat { r } _ { i j } = { \bf u } _ { i } ^ { T } \cdot { \bf v } _ { j }$ as the unbiased rating predictor to generate future recommendations.

Generally, the debiasing mechanism of PD is very intuitive and universal among backdoor adjustment-based causal RSs [73, 85]: When fitting the RS model on the biased training set, we explicitly introduce the item popularity $m _ { j }$ (i.e., the confounder) in Eq. (10.17) to explain away the spurious correlation between item attributes and the observed user ratings. Therefore, the user/item latent variables $\mathbf { u } _ { i }$ and ${ \bf v } _ { j }$ used to generate future recommendations, i.e., $\hat { r } _ { i j } = { \bf u } _ { i } ^ { T } \cdot { \bf v } _ { j }$ , can focus exclusively on estimating users’ true interests on intrinsic item properties.

Is popularity bias always bad? Recently, more researchers have begun to believe that popularity bias is not necessarily bad for RSs, because some items are popular because they per se have better quality than other items or they catch the current trends of user interests, where more recommendations for these items can be well-justified [12, 101]. Therefore, rather than setting the interventional distribution of item popularity to $p ( m _ { j } )$ ), PD introduced above as well as some other methods [98] further propose to make it correspond to item qualities or reflect the future popularity predictions. We will introduce these strategies in Sect. 10.4.3 regarding causal generalizations of RSs.

## 10.4.1.3 Clickbait Bias

Different from previous subsections that mainly focus on causal debiasing strategies for collaborative filtering-based RSs, this section discusses content-based recommendations. Specifically, we discuss the clickbait bias, which is defined as the bias of overly recommending items with attractive exposure features such as sensational titles but with low content qualities. The discussion is mainly based on [72]. We assume that item features $\mathbf { f } _ { j } ^ { v }$ can be further decomposed into the item content feature ${ \bf f } _ { j } ^ { c }$ that captures item content information and the item exposure feature $\mathbf { f } _ { j } ^ { b }$ whose main purpose is to attract users’ attention. Taking micro-video as an example, item content feature $\mathbf { f } _ { j } ^ { c }$ can be the audiovisual content of the video, whereas item exposure feature $\mathbf { f } _ { j } ^ { b }$ can be its title, which is not obliged to describe its content faithfully.

The relations among user interests $\mathbf { u } _ { i }$ , item exposure feature $\mathbf { f } _ { j } ^ { b } .$ , item content feature ${ \bf f } _ { j } ^ { c } .$ , item fused features ${ \bf v } _ { j }$ , and the observed ratings $r _ { i j }$ are depicted in the causal graph in Fig. 10.7a. We note that clickbait bias occurs when a user’s recorded click on an item because she was cheated by the item exposure feature $\mathbf { f } _ { j } ^ { b }$ before viewing the item content $\mathbf { f } _ { j } ^ { c }$ . Therefore, the bias can be defined as the direct influence of $\mathbf { f } _ { j } ^ { b }$ on ratings $r _ { i j }$ represented by the causal path $F ^ { b }  R$ . To eliminate the clickbait bias, we need to block the direct influence of $F ^ { b }$ on rating predictions, such that the item content quality can be comprehensively considered in recommendations.

As with SCM-based causal RSs, we first estimate structural equations of interest in the causal graph, i.e., $p _ { G } ( \mathbf { v } _ { j } | \mathbf { f } _ { j } ^ { b } , \mathbf { f } _ { j } ^ { c } )$ and $p _ { G } ( r _ { i j } | \mathbf { u } _ { i } , \mathbf { v } _ { j } , \mathbf { f } _ { i } ^ { b } )$ . Since distributions in [72] are reasoned in a deterministic manner (i.e., Gaussian distributions with infinite precision), we keep the discussion consistent with them. Specifically, we use ${ \bf v } _ { j } ( \bar { { \bf f } } _ { j } ^ { b } , { \bf f } _ { j } ^ { c } ) ~ = ~ f ^ { f f } ( { \bf f } _ { j } ^ { b } , \bar { { \bf f } } _ { j } ^ { c } )$ to represent the structural equation $p _ { G } ( \mathbf { v } _ { j } | \mathbf { f } _ { j } ^ { b } , \mathbf { f } _ { j } ^ { c } )$ , where $f ^ { f f }$ is the feature fusion function that aggregates $\mathbf { f } _ { j } ^ { b } , \mathbf { f } _ { j } ^ { c }$ into ${ \bf v } _ { j }$ , and use $r _ { i j } ( \mathbf { u } _ { i } , \mathbf { v } _ { j } , \mathbf { f } _ { j } ^ { b } )$ to represent the structural equation $p _ { G } ( r _ { i j } | \mathbf { u } _ { i } , \mathbf { v } _ { j } , \mathbf { f } _ { j } ^ { b } )$ , respectively. To explicitly disentangle the influence of item exposure feature $\mathbf { f } _ { j } ^ { b }$ and item latent variable ${ \bf v } _ { j }$ on the observed ratings, $r _ { i j } ( \mathbf { u } _ { i } , \mathbf { v } _ { j } , \mathbf { f } _ { j } ^ { b } )$ is assumed to factorize as follows:

![image_69](images/image_69.png)

```mermaid
graph TD
  U --> R
  V --> R
  V --> FC["F^c"]
  R --> FB["F^b"]
  FB --> V
```

![image_70](images/image_70.png)

```mermaid
graph TD
  U --> R
  V["V*"] --> R
  V --> Fc["Fc*"]
  V --> Fb["Fb*"]
  R --> Fb
```

Fig. 10.7 (a): The SCM that considers both the causal influences of item content feature $F ^ { c }$ and item exposure feature $F ^ { b }$ on item latent variable V . (b): The counterfactual SCM where $V ^ { * }$ i s determined by baseline value $F ^ { b * }$ and $F ^ { c * }$ to model the undesirable direct effects of $F ^ { b }$

$$
r _ {i j} (\mathbf {u} _ {i}, \mathbf {v} _ {j}, \mathbf {f} _ {j} ^ {b}) = \underbrace {f _ {n n} ^ {u v} (\mathbf {u} _ {i} , \mathbf {v} _ {j})} _ {\text { user   interests }} \cdot \underbrace {\text { Sigmoid } \left(f _ {n n} ^ {u f} (\mathbf {u} _ {i} , \mathbf {f} _ {j} ^ {b})\right)} _ {\text { potential   clickbait   bias }}, \tag {10.19}
$$

where the Sigmoid function provides necessary nonlinearity in the fusion process. Essentially, Eq. (10.19) represents the causal mechanism that generates the observed ratings, which entangles both user interests in item content and clickbait bias.

$\mathbf { u } _ { i } , \mathbf { v } _ { j }$ $f _ { n n } ^ { u f } , f _ { n n } ^ { u v }$ Eq. (10.19), removing clickbait bias from the rating predictions is not as straightforward as the PD algorithm, because we should eliminate only the direct influence of item exposure feature $\mathbf { f } _ { j } ^ { b }$ on ratings $r _ { i j }$ , while preserving its indirect influence mediated by item latent variable ${ \bf v } _ { j }$ , such that all available item features can be comprehensively considered in recommendations. To achieve this purpose, we first calculate the natural direct effect (NDE) [48] of item exposure feature $\mathbf { f } _ { j } ^ { b }$ on ratings $r _ { i j }$ as follows:

$$
\mathrm{NDE} (\mathbf {u} _ {i}, \mathbf {v} _ {j} ^ {*}, \mathbf {f} _ {j} ^ {b}) = r _ {i j} (\mathbf {u} _ {i}, \mathbf {v} _ {j} ^ {*}, \mathbf {f} _ {j} ^ {b}) - r _ {i j} (\mathbf {u} _ {i}, \mathbf {v} _ {j} ^ {*}, \mathbf {f} _ {j} ^ {b *}), \tag {10.20}
$$

where $\mathbf { v } _ { j } ^ { * } = f _ { n n } ^ { f f } ( \mathbf { f } _ { j } ^ { b * } , \mathbf { f } _ { j } ^ { c * } )$ , and the baseline values $\mathbf { f } _ { i } ^ { b * } , \mathbf { f } _ { i } ^ { c * }$ are treated as if the corresponding features are missing from the item $[ \check { 7 } 2 ]$ . Since the second term $r _ { i j } ( \mathbf { u } _ { i } , \mathbf { v } _ { j } ^ { * } , \mathbf { f } _ { j } ^ { b * } )$ in Eq. (10.20) denotes the user’s rating to a “void” item and can be viewed as a constant, it will not affect the rank of the items. So we only adjust the first term of Eq. (10.20), which reasons with user $i \ ' \mathrm { s }$ rating to item j in a counterfactual world where item j has only the exposure feature $\mathbf { f } _ { j } ^ { b }$ but no content and fused features $\mathbf { f } _ { j } ^ { c * }$ and $\mathbf { v } _ { j } ^ { * }$ , in Eq. (10.19) (Fig. 10.7b). The adjustment leads to the following estimator,

![image_71](images/image_71.png)

```mermaid
graph TD
  S((S)) --> R((R))
  F((F)) --> R
  R --> U((U))
  U --> R̂((R̂))
  V((V)) --> R["R̂"]
    style S fill:#ff0000,stroke:#333
    style F fill:#ff0000,stroke:#333
    style R fill:#ff0000,stroke:#333
    style U fill:#ff0000,stroke:#333
    style V fill:#ff0000,stroke:#333
    style R̂ fill:#ff0000,stroke:#333
```

(a) Causal Generation Process of the Observational Dataset  
(b) Causal Decision Process of the Traditional RSs  
Fig. 10.8 The SCM that reasons with the causal decision mechanism of traditional RSs. Observed user ratings R can be causally driven by user features F , including sensitive features S, which can then unfairly influence the inference of user latent variables U and new rating predictions Rˆ

$$
\hat {r} _ {i j} = r _ {i j} \left(\mathbf {u} _ {i}, \mathbf {v} _ {j}, \mathbf {f} _ {j} ^ {b}\right) - r _ {i j} \left(\mathbf {u} _ {i}, \mathbf {v} _ {j} ^ {*}, \mathbf {f} _ {j} ^ {b}\right) \triangleq \underbrace {r _ {i j} \left(\mathbf {u} _ {i} , \mathbf {v} _ {j} , \mathbf {f} _ {j} ^ {b}\right)} _ {\text { user   interests } + \text { clickbait }} - \underbrace {r _ {i j} \left(\mathbf {u} _ {i} , \mathbf {v} _ {j} ^ {*} , \mathbf {f} _ {j} ^ {b}\right)} _ {\text { clickbait   bias }}. \tag {10.21}
$$

Eq. (10.21) removes the direct influence of $\mathbf { f } _ { j } ^ { b }$ on rating predictions, such that item content quality can be comprehensively considered in future recommendations.

## 10.4.1.4 Unfairness

Recently, with the growing concern of algorithmic fairness, RSs are expected to show no discrimination against users from certain demographic groups [14, 32, 34]. However, traditional RSs may capture the undesirable associations between users’ sensitive information and their historical activities, which leads to potentially offensive recommendations to the users. Causal inference can help identify and address such unfair associations, where fairness can be promoted in future recommendations. This section focuses on the user-oriented fairness discussed in [33], which is defined as the bias where RS discriminately treats users with certain sensitive attributes.

When considering the user-oriented fairness for RSs, a subset of user features $\mathbf { f } _ { i }$ , which we denote as $\mathbf { s } _ { i }$ , is assumed to contain the sensitive information of users, such as gender, race, and age. Features $\mathbf { s } _ { i }$ are sensitive because recommendations that improperly rely on these features may be offensive to users, which degrade both their online experiences and their trust in the system. The causal graph that depicts the causal decision mechanism of most traditional RSs is illustrated in Fig. 10.8 [33]. From Fig. 10.8, we can find that the user historical behaviors, i.e., the observed ratings $r _ { i j }$ , are causally driven by user features $\mathbf { f } _ { i }$ , including user sensitive features $\mathbf { s } _ { i }$ . Therefore, the user latent variables $\mathbf { u } _ { i }$ inferred from $r _ { i j }$ could capture sensitive user information in $\mathbf { s } _ { i }$ , which unfairly influences the rating predictions $\hat { r } _ { i j }$ in the future.

To address this problem, Li et al. [33] proposed to disentangle the user sensitive features $\mathbf { s } _ { i }$ from the user latent variable $\mathbf { u } _ { i }$ , such that the unfair influence of $\mathbf { s } _ { i }$ on $\mathbf { u } _ { i }$ represented by the causal chain $S \to R \to U$ can be maximally suppressed in the future recommendations. A common strategy to achieve the disentanglement is adversarial training [18], where we train a discriminator $f _ { n n } ^ { \mathrm { c l s } } ( \mathbf { u } _ { i } ) ~ \to ~ \mathbf { s } _ { i }$ that predicts the sensitive features $\mathbf { s } _ { i }$ from user latent variables $\mathbf { u } _ { i }$ alongside the RS. While fitting the RS on the observe ratings $r _ { i j }$ , we constrain the inferred $\mathbf { u } _ { i }$ to fool $f _ { n n } ^ { \mathrm { c l s } }$ $\mathbf { s } _ { i }$ $\mathbf { u } _ { i }$ from capturing sensitive information in $r _ { i j }$ due to its unfair correlations with $\mathbf { s } _ { i }$ . Here we take the MF-based RS as an example to show the details. We use $\mathcal { L } ^ { \mathrm { R e c } }$ to denote the original training objective of the MF-based RS that maximizes the log-likelihood on observed ratings $r _ { i j }$ and use $\mathcal { L } ^ { \mathrm { c l s } }$ to denote the loss function of $f _ { n n } ^ { \mathrm { c l s } }$ $\mathcal { L } ^ { \mathrm { F a i r } }$ becomes the following:

$$
\mathcal {L} ^ {\text { Fair }} = \underbrace {\mathcal {L} ^ {\text { Rec }} (\mathbf {u} _ {i} ^ {T} \cdot \mathbf {v} _ {j} , r _ {i j})} _ {\text { user   interests }} - \lambda \cdot \underbrace {\mathcal {L} ^ {\text { cls }} (f _ {n n} ^ {\text { cls }} (\mathbf {u} _ {i}) , \mathbf {s} _ {i})} _ {\text { fairness   constraint }}, \tag {10.22}
$$

where λ is a hyper-parameter that balances the recommendation performance and the fairness objective. Generally, a higher λ leads to better fairness, but it also restricts the capacity of the user latent variables $\mathbf { u } _ { i }$ , which could negatively impact the recommendation performance. Although here we use the MF-based RS as an example, it is straightforward to generalize Eq. (10.22) to DMF or AE-based RS by replacing the $\mathbf { u } _ { i } ^ { T } \cdot \bar { \mathbf { v } } _ { j }$ term with the corresponding rating estimators.

## 10.4.2 Causal Explanation in Recommendations

In previous sections, we have introduced causality to address various types of bias and spurious correlation issues for traditional RSs. In this section, we use causality to explain the user decision process. Specifically, we discuss an interesting question aiming to disentangle users’ intent that causally explains their past behaviors, i.e., did a user purchase an item because she conformed to the current trend or because she really liked it? The tricky part of this question is that: in reality, we only observe the effects, i.e., the purchases, which can be explained by both causes.

## 10.4.2.1 Disentangling Interest and Conformity with Causal Embedding

The discussion is based on DICE proposed in [102]. To simplify the discussion, we consider $r _ { i j }$ as implicit feedback and define the set of user, positive item $( j : r _ { i j } =$ 1), negative item $( k : r _ { i k } = 0 )$ triplets as $\mathcal { R } _ { p n } = \{ ( i , j , k ) | r _ { i j } = 1 \wedge r _ { i k } = 0 \}$ . The popularity of each item j , i.e., mj , which reflects the current trend, can be calculated with Eq. (10.16). Observing that the causal relation between user interests $U$ , user conformity $U _ { c }$ , and observed ratings R can be represented as a V-structure in Fig. 10.9a, DICE exploits the colliding effect to achieve the disentanglement, i.e., outcomes that cannot be explained by one cause are more likely caused by another (see discussions in Sect. 10.3.2.2). Therefore, although users’ interests cannot be directly estimated from their ratings $r _ { i j }$ due to entanglement, their conformity to the trend can be estimated by the popularity level of item $j ,$ , and positive feedback not likely caused by conformity has a higher chance of reflecting users’ true interests.

In implementation, DICE assumes that the observed ratings $r _ { i j }$ can be decomposed into the sum of a conformity part $r _ { i j } ^ { c } = f ^ { c } ( \mathbf { u } _ { i } ^ { c } , \mathbf { v } _ { j } ^ { c } )$ and a user interests part $r _ { i j } ^ { i } = f ^ { i } ( \mathbf { u } _ { i } ^ { i } , \mathbf { v } _ { j } ^ { i } )$ , where ${ \bf u } _ { i } ^ { c , i } , { \bf v } _ { j } ^ { c , i }$ are learnable user, item embeddings that reflect user $i \ ' \mathrm { s }$ interests in (i.e., superscript i) and conformity to (i.e., superscript c) item $j$ , respectively. According to the colliding effect of causal graphs, we can split the ${ \mathcal { R } } _ { p n }$ $\mathcal { R } _ { p n } ^ { ( 1 ) }$ $a$ higher popularity level than the negative item b, i.e., $m _ { a } > m _ { b }$ . In this case, we can draw two general conclusions from this triplet: (1) Overall, the user prefers item a over $b ; ( 2 )$ She is more likely to conform to item a than item b due to $\boldsymbol { a } ^ { \prime } \boldsymbol { \mathrm { s } }$ higher popularity. These conclusions lead to the two inequalities as follows:

$$
\forall (i, a, b) \in \mathcal {R} _ {p n} ^ {(1)}, \text { we   have } \left\{ \begin{array}{l} r _ {i a} ^ {c} > r _ {i b} ^ {c} (\text { conformity }) \\ r _ {i a} ^ {i} + r _ {i a} ^ {c} > r _ {i b} ^ {i} + r _ {i b} ^ {c} (\text { overall   preference }), \end{array} \right. \tag {10.23}
$$

$r _ { i \{ a , b \} } ^ { c , i }$ ${ \mathbf { u } } _ { i } ^ { c , i } , { \mathbf { v } } _ { \{ a , b \} } ^ { c , i }$ are omitted for $\mathcal { R } _ { p n } ^ { ( 2 ) }$ because for every triplet $( i , c , d )$ in $\mathcal { R } _ { p n } ^ { ( 2 ) }$ , the negative item $d$ is more popular than the positive item c. In this case, user i could have simply conformed to the trend and chosen item d to consume, but instead, she actively chose the less popular item c. Therefore, we can draw one more specific conclusion that leads to the disentanglement between user interests and conformity: The choice of item c over d is more likely due to user interests. Therefore, we can form three inequalities:

![image_72](images/image_72.png)

```mermaid
graph TD
  A["User Interests"] --> C["R"]
  B["User Conformity"] --> C["R"]
  C --> D["purchases"]
```

(a) Causal graph for DICE

![image_73](images/image_73.png)

```mermaid
graph TD
  A["User Interest"] --> C["R"]
  B["Geographic Influence"] --> C["R"]
  C["R"] --> D["Visits"]
```

(b) Generalization to PoI recommendation  
Fig. 10.9 Causal Graphs for DICE (a) and its generalization to PoI recommendations (b)

$$
\forall (i, c, d) \in \mathcal {R} _ {p n} ^ {(2)}, \text { we   have } \left\{ \begin{array}{l} r _ {i c} ^ {i} > r _ {i d} ^ {i} (\text { interests }), r _ {i c} ^ {c} <   r _ {i d} ^ {c} (\text { conformity }), \\ r _ {i c} ^ {i} + r _ {i c} ^ {c} > r _ {i d} ^ {i} + r _ {i d} ^ {c} (\text { overall   preference }). \end{array} \right. \tag {10.24}
$$

The inequalities in Eqs. (10.23) and (10.24) can be solved by ranking-based loss in RSs, such as Bayesian personalized ranking (BPR) [52], where the disentangled embeddings ${ \bf u } _ { i } ^ { c , i } , { \bf \bar { v } } _ { j } ^ { c , i }$ and the match functions $f ^ { c , i } ( \cdot , \cdot )$ can be learned from $\mathcal { R } _ { p n } ^ { ( 1 ) }$ and $\mathcal { R } _ { p n } ^ { ( 2 ) }$ $\hat { r } _ { i j } = f ^ { i } ( \mathbf { u } _ { i } ^ { i } , \mathbf { v } _ { j } ^ { i } ) + f ^ { c } ( \mathbf { u } _ { i } ^ { c } , \mathbf { v } _ { j } ^ { c } )$ recommendations.

## 10.4.2.2 Generalizations of DICE

DICE disentangles the user intent and promotes the explainability of RSs from the data’s perspective: It partitions the triplets $( i , j , k )$ in ${ \mathcal { R } } _ { p n }$ into two disjoint subsets $\mathcal { R } _ { p n } ^ { ( 1 ) }$ $\mathcal { R } _ { p n } ^ { ( 2 ) }$ $\mathcal { R } _ { p n } ^ { ( 2 ) }$ their conformity to the popularity trend. The basic idea of DICE is generalizable to promote explainability for other types of recommendation tasks if we can find alternative causal explanations to challenge the assumption that the observed positive feedback in these tasks can be attributed solely to user interests.

For example, in point-of-interests (PoI) recommendations, the target items are specific point locations that users may find useful or interesting to visit, such as restaurants, grocery stores, and malls [93]. In this task, the location of a PoI is an important alternative explanation for users’ visits to the PoI other than user interests, because nearby PoIs are more convenient to visit than the remote ones (See Fig. 10.9b) [70]. Therefore, to disentangle user interests from potential geographical factors that could causally influence users’ choices, we can take a similar strategy as DICE and partition the user historical visit records according to the distance of positive and negative PoIs to users. Then, the disentangled user interest embeddings can be estimated based on the partitioned dataset with the same ranking-based approach.

## 10.4.2.3 Other Works on Explainable RSs

The explainable recommendation is a broad topic [100], where disentangling users’ intent based on data partitioning is a small part. There are also plenty of works that focus on improving the explainability of RSs from the model’s side, where specific disentanglement modules, such as prototype learning [40], context modeling [74], and aspect modeling [67], are designed and integrated with traditional RS models to further enhance their transparency and explainability. We refer interested readers to the corresponding papers as well as [63, 88] for further investigation.

## 10.4.3 Causal Generalization of Recommendations

After estimating the causal relations from potentially biased and entangled observational datasets, the generalization ability of RSs can be substantially enhanced, because even if the context (or environment) in which we make recommendations changes (e.g., item popularity and user conformity), we can still basing the recommendations on causal relations that are stable and invariant, while discarding or correcting other undesirable correlations that are transient and susceptible to change [5, 90, 102]. In this section, we use the PD algorithm for popularity bias and the DICE algorithm for causal explainability as two examples to show how the generalization of RSs can be improved with causal intervention and disentanglement.

## 10.4.3.1 Generalization Based on Intervention

First, we take the PD algorithm as an example to show how causal intervention can improve the generalization of RSs within a dynamic environment. In RS, it is generally assumed that user interests can remain unchanged for a certain period of time, i.e., the causal structure $U \to R \gets V$ in Fig. 10.6 represents the stable user interests on intrinsic item properties. However, the popularity of different items, i.e., the context in which we make recommendations, can shift rapidly during the same period [12]. Recall that PD disentangles the causal influences of user interests and item popularity on ratings via the product of two terms, i.e., Elu $( \mathbf { u } _ { i } ^ { T } \cdot \mathbf { v } _ { j } )$ and $m _ { j } ^ { \lambda }$ , as Eq. (10.17). Suppose $m _ { j }$ represents the current popularity level of item $j .$ . If we predict that the popularity of item j will change to $m _ { j } ^ { \prime }$ in the future [84], we can conduct an intervention that sets M to the predicted value $m _ { j } ^ { \prime }$ in the structural equation $p _ { G } ( R | U , V , M )$ and predict future ratings $r _ { i j } ^ { \prime }$ via the following formula:

$$
p _ {G} (r _ {i j} ^ {\prime} | \mathbf {u} _ {i}, \mathbf {v} _ {j}, d o (m _ {j} ^ {\prime})) \propto \underbrace {\mathrm{Elu} (\mathbf {u} _ {i} ^ {T} \cdot \mathbf {v} _ {j})} _ {\text { stable   user   interests }} \times \underbrace {(m _ {j} ^ {\prime}) ^ {\lambda}} _ {\text { future   popularity }}, \tag {10.25}
$$

where the user, item latent variables $\mathbf { u } _ { i }$ and ${ \bf v } _ { j }$ learned from the current time step remain unaltered. With the influence of future changes in item popularity on ratings considered in the predictions, service providers can make strategic decisions to allocate resources for items with different popularity potentials. In contrast, traditional RSs could mistakenly capture the influence of the current popularity level of items on ratings as user interests. Therefore, they will not generalize well when the item popularity $m _ { j }$ changes to a different level $m _ { j } ^ { \prime }$ due to time evolution.

## 10.4.3.2 Generalization Based on Disentanglement

In addition, causal disentanglement can promote the generalization of RSs by identifying and basing recommendations on causes that are more robust to potential changes in the environments [66, 91]. For example, if users’ conformity and interest are disentangled based on their historical behaviors, if a user’s conformity reduces to a low level due to certain reasons, since user interests are assumed to be stable within a certain period of time, we can still use the learned user/item interest variables $\mathbf { u } _ { i } ^ { i } , \mathbf { v } _ { j } ^ { i }$ to make recommendations based on their interests, where the previously estimated unreliable user conformity information can be discarded or down-weighted. In contrast, for traditional RSs, different factors that causally influence their historical behaviors are entangled as a single user latent variable ui. Therefore, even if some less stable causes of user behaviors are known to change (e.g., in the PoI RS introduced above, a user could move to a new place where the convenience levels of different PoIs change for the user), these models will still utilize the outdated causes to make recommendations, which could fail to generalize to the new environment.

## 10.5 Evaluation Strategies for Causal RSs

In the previous sections, we have discussed various causal inference techniques that are promising to address multiple types of biases, entanglement, and generalization problems in traditional RSs. However, without a well-designed model evaluation strategy, it is difficult to tell whether the proposed causal RS model is indeed effective, nor can we guarantee that the model will perform reliably after deploying in a real-world environment. The evaluation of causal models is particularly difficult because the ground truths, i.e., the causal effects of interest, are usually infeasible. Despite the challenges, there are several strategies that can reliably evaluate causal RSs with biased real-world data, and we will thoroughly discuss them in this section. In addition, we also compile the available real-world datasets that conduct randomized experiments to eliminate exposure bias to facilitate future causal RS research.

## 10.5.1 Evaluation Strategies for Traditional RSs

The assessment of traditional RSs generally follows three steps: First, the observed ratings $r _ { i j }$ in the rating matrix R are split into the non-overlapping training set $\mathbf { R } _ { t r }$ and test set $\mathbf { R } _ { t e }$ , usually by randomly holding out a certain percentage of the observed ratings from each user. Then, the proposed RS is trained on ratings in $\mathbf { R } _ { t r }$ to learn the latent variables and the associated functional models (see Sect. 10.2). Finally, the trained RS predicts the missing ratings in $\mathbf { R } _ { t r }$ for each user, where the results are compared with the held-out ratings in $\mathbf { R } _ { t e }$ to evaluate the model performance. The quality of rating predictions can be measured by accuracybased metrics such as mean squared error (MSE) and mean absolute error (MAE) and ranking-based metrics such as recall, precision, and normalized discounted cumulative gain (NDCG). More information on these evaluation metrics can be found in [61].

## 10.5.2 Challenges for the Evaluation of Causal RSs

The above evaluation strategy, however, is not directly applicable to causal RSs, because ratings in $\mathbf { R } _ { t e }$ may have the same spurious correlation and bias as ratings in $\mathbf { R } _ { t r }$ , which makes the evaluation on $\mathbf { R } _ { t e }$ a biased measure of the true model performance. Therefore, to unbiasedly evaluate the effectiveness of causal RSs, it is ideal that we have a biased/entangled training set $\mathbf { R } _ { t r } ^ { b }$ to learn the model and an unbiased/disentangled test set $\mathbf { R } _ { t e } ^ { u b }$ for model evaluation, such that the effectiveness of the causal debiasing/disentangling algorithm can be directly verified $\mathbf { R } _ { t e } ^ { u b }$ to acquire and expensive to establish. Therefore, we first introduce common data simulation strategies for causal RS evaluation. We then discuss how real-world datasets can be directly utilized to further promote the credibility of causal RS research.

## 10.5.3 Evaluation Based on Simulated Datasets

A good dataset simulation strategy to evaluate causal RSs should have the following properties: (1) The generation mechanisms of the bias and entanglement to be studied are clearly identified, credibly designed, and can be adjusted in a flexible manner; (2) The available real-world information is utilized as much as possible.

## 10.5.3.1 Simulation Based on Generative Models

One promising dataset simulation strategy that satisfies the above criteria is to use deep generative models. Here we take exposure bias as an example to demonstrate how it can be simulated from real-world datasets [109]. The simulation is composed of two phases. In the training phase, two variational auto-encoders (VAEs) [25, 36] are trained on the exposure and rating data in a real-world dataset (e.g., the MovieLens dataset [20]), which results in two decoder networks $f _ { n n } ^ { a }$ and $f _ { n n } ^ { r }$ that generate item exposures $\mathbf { a } _ { i } \in \{ 0 , 1 \} ^ { J }$ and user ratings $\mathbf { r } _ { i } \in \mathbb { R } ^ { J }$ from K-dimensional Gaussian user latent variables ${ \bf u } _ { i } ^ { a } \sim N ( { \bf 0 } , { \bf I } _ { K } )$ and $\mathbf { u } _ { i } ^ { r } \sim { \cal N } ( \mathbf { 0 } , \mathbf { I } _ { K } )$ , respectively. The decoders capture the generative distributions of item exposures and user ratings based on the data of real users, where the available real-world information is effectively utilized. In the generation phase, for each hypothetical user $i ^ { \prime }$ , we draw a confounder $\mathbf { c } _ { i ^ { \prime } } \sim { \cal N } ( \mathbf { 0 } , \mathbf { I } _ { K } )$ that simultaneously affects ${ \bf u } _ { i ^ { \prime } } ^ { a }$ and $\mathbf { u } _ { i ^ { \prime } } ^ { r }$ . Then, to simulate the exposure bias, we set $ { \mathbf { u } } _ { i ^ { \prime } } ^ { a } =  { \mathbf { c } } _ { i ^ { \prime } }$ and $\mathbf { u } _ { i ^ { \prime } } ^ { r } = \lambda \cdot \mathbf { c } _ { i ^ { \prime } } + ( 1 - \lambda ) \mathbf { \epsilon } _ { i ^ { \prime } }$ and use $f _ { n n } ^ { a } , f _ { n n } ^ { r }$ t o generate the simulated item exposures $\mathbf { a } _ { i ^ { \prime } }$ and ratings $\mathbf { r } _ { i ^ { \prime } }$ , where $\epsilon _ { i ^ { \prime } } \sim { \cal N } ( \mathbf { 0 } , \mathbf { I } _ { K } )$ and hyper-parameter $\lambda$ controls the strength of the confounding bias. Finally, we mask $\mathbf { r } _ { i ^ { \prime } }$ with $\mathbf { a } _ { i ^ { \prime } }$ to form the biased training set $\mathbf { R } _ { t r } ^ { b }$ , and keep the generated ratings $\mathbf { r } _ { i ^ { \prime } }$ $\mathbf { R } _ { t e } ^ { u b }$

The advantage of dataset simulation strategies based on generative models is that the true causal mechanisms of interest, such as the rating potential outcomes, are available in the evaluation stage, which is generally impossible for real-world datasets. Therefore, the effectiveness of causal RSs can be easily verified based on the simulated ground truths. In addition, the simulations are flexible as the strength of biases and entanglements can be set into different levels (e.g., λ in the example), where the sensitivity and robustness of causal RSs can be thoroughly investigated.

## 10.5.3.2 Test Set Intervention

Another reliable dataset simulation strategy is test set intervention, where an intervened test set is created from the original test set, such that it has a different bias/entanglement distribution from the training set [35, 102, 110]. For example, to study the popularity bias, we can first select observed ratings from R such that 90% of the interacted items are popular and 10% are unpopular to form the training set $\mathbf { R } _ { t r }$ [94]. We then select from the remaining ratings, i.e., the original test set $\mathbf { R } _ { t e }$ , a subset with a different ratio of popular and unpopular items (e.g., 10% popular and $\mathbf { R } _ { t e } ^ { i n t }$ $\mathbf { R } _ { t r }$ can still perform well on the intervened test set $\mathbf { R } _ { t e } ^ { i n t }$ , the model’s invariance to the popularity bias can be supported. A similar test set intervention strategies can be used to evaluate the disentanglement of user interests and conformity for DICE [102].

The advantage of the test set intervention-based causal RS evaluation strategy is that extra assumptions that cannot be justified by real-world information are minimally introduced because the intervention is usually achieved by selecting samples from the original test set to change the data distribution, which does not introduce extra assumptions of the generative mechanisms or hypothetical users, items, and ratings. From this perspective, the evaluation results based on test set intervention may be more credible compared with the generative model-based strategies.

## 10.5.4 Evaluation Based on Real-world Datasets

## 10.5.4.1 Randomized Experiments

For the study of exposure bias, it is feasible to establish-bias free real-world datasets, where ratings for either every item or randomly selected items are collected from a subset of users. This can be extremely expensive and user-unfriendly, but recent years have witnessed a growing interest in causal RS research from the industry, where more such randomized datasets are established and released to facilitate causal RS research. The available real-world datasets are compiled as follows:

• Coat datasets13 [60] (2016). The Coat dataset is a small-scale dataset crowdsourced from the Amazon Mechanical Turkers platform with 300 users and 290 items. Specifically, each Turker is first asked to self-select 24 coats to rate, where the ratings form the biased training set $\mathbf { R } _ { t r } ^ { b }$ . Then each Turker is asked to rate 16 $\mathbf { R } _ { t e } ^ { u b }$
• Yahoo! R3 dataset14 [44, 45] (2009). The Yahoo! R3 dataset is collected from the Yahoo! Music platform. The biased training set $\mathbf { R } _ { t r } ^ { b }$ is composed of 300,000 self-supplied ratings from 15,400 users to 1,000 items. In addition, a subset of 5,400 users is presented with ten randomly selected items to rate, and the ratings $\mathbf { R } _ { t e } ^ { u \bar { b } }$
• KuaiRec dataset15 [16] (2022). The KuaiRec dataset is established based on a popular micro-video sharing platform, KuaiShou, in China (known as Kwai internationally). The dataset records self-supplied ratings from 7,176 users to $\mathbf { R } _ { t r } ^ { b }$ $\mathbf { R } _ { t e } ^ { u b }$ of a subset of 1,411 users and 3,327 items, where the ratings between these users and items are almost fully observed (with 99.6% density).

The statistics of the datasets are summarized in Table 10.1 for reference. There are also randomized datasets for some related topics such as click-through rate prediction [104], i.e., Criteo Ads datasets16 [13], and bandit-based RS [8], i.e., Open

**Table 10.1 Characteristics of the currently available real-world causal recommendation datasets, where the test sets are devoid of exposure bias either due to randomized item exposures or fully observed ratings. In the table, terms like 24 i/u mean that every user rates 24 items, the term 300,000 r denotes the number of observed ratings, and terms like 16.3% r represent the density of interactions**

<table><tr><td>Dataset</td><td># Users</td><td># Items</td><td>Item type</td><td>Training sets</td><td>Test sets</td></tr><tr><td>Coat</td><td>300</td><td>290</td><td>Coat</td><td>24 i/u (self-supplied)</td><td>16 i/u (random)</td></tr><tr><td>Yahoo! R3</td><td>15,400</td><td>1,000</td><td>Music</td><td>300,000 r (self-supplied)</td><td>10 i/u (random) for 5,400 u</td></tr><tr><td>KuaiRec</td><td>7,176</td><td>10,728</td><td>Video</td><td>16.3% r (self-supplied)</td><td>99.6% r for 1,411 u and 3,327 i</td></tr></table>

Bandit dataset17 [56], where the sources are also provided in case the readers are interested.

From Table 10.1 we can find that, the Coat dataset is small in scale. While for the Yahoo! R3 dataset, the training set is comparatively large (15,400 users and 1,000 items), the randomized experiment conducted to establish the unbiased test set is small-scale in comparison (16 and 10 randomly exposed items per user, respectively). Therefore, although these ratings are unbiased due to randomization, they may not capture well-rounded user interests and therefore induce a high evaluation variance. For the recently released KuaiRec datasets, large-scale experiments are conducted on users to establish the bias-free test set, where the 1,411 users’ ratings for 3,327 items are almost fully collected. Therefore, it may be a promising new benchmark that allows the evaluation of more complex causal RS models with a lower variance.

## 10.5.4.2 Qualitative Evaluation and Case Study

For other types of biases in RSs that cannot be attributed to non-randomized item exposures (e.g., clickbait bias and unfairness), the establishment of bias-free test sets is more challenging. For example, when studying clickbait bias, it is difficult to determine whether a user clicked an item due to interests or clickbait. Similarly, when examining the user-oriented fairness of RSs, we cannot know if the generated items are offensive to the users. Under such circumstances, we can still conduct case studies for qualitative model evaluations, where we manually select some representative samples from the original test set and observe whether the trained causal RS model would respond as expected to these samples [71].

Consider the evaluation of the robustness of a causal RS to clickbait bias. We can select some representative items with low-quality content but highly attractive exposure features and other items with high-quality content but normal exposure features from the original test set. Then, we obtain rating predictions for items from these two groups and draw comparisons. If the studied causal RS indeed ranks items in the second group higher than those in the first group, we can likely conclude that the model is robust to clickbait bias because the quality of the item content, not its exposure features, is prioritized in recommendations. In addition, to evaluate the user-oriented fairness of a causal RS, we can analyze the generated recommendation for users from certain demographic groups. If the recommended items tend to capture the social stereotypes that are negatively associated with user sensitive features, we can conclude that the model is still discriminatory against users.

## 10.6 Future Directions

Despite the recent achievements in marrying causal inference with traditional RSs to address their various limitations of correlational reasoning on observational user data, causal RS research is still in its emerging stage. Several promising directions could be pursued to further advance this field. In this section, we identify four interesting and important open problems worthy of exploration in the future.

First, the assumptions required by existing causal RSs could be too strong, which may not hold in reality. For example, most RCM-based causal RSs rely on SUTVA to exclude the interference of item exposures for different users. However, if users are connected by a social network, they may interact closely with each other or be heavily affected by the influencers in the network [41]. Consequently, SUTVA can be violated because the recommendations made to one user may causally affect the ratings of others (i.e., the spill-over effects [30, 42]). In addition, the positivity assumptions may also be violated if some users never click certain types of items (i.e., noncompliance and defiers [23]). Therefore, it is crucial to further weaken the assumptions of causal RSs to make them more practical for real-world applications.

In addition, there currently lacks a universal causal model for RSs that can be applied for different causal reasoning purposes. Most SCM-based causal RSs are designed to address one specific type of bias or entanglement problem, where other issues are tacitly assumed to be absent and omitted from the causal graph. Moreover, even for causal RSs that address the same problem, several varieties of causal graphs that include different sets of variables and relationships can be assumed, which leads to inconsistency between different works. Therefore, it would be promising and beneficial to have a generic and widely accepted causal model that is able to comprehensively address multiple types of causal problems in recommendations.

Furthermore, certain types of biases in RSs are double-blade swords, where the positive side is seldom investigated. Consider the item exposure bias discussed in Sect. 10.4.1.1. We should note that some items are more likely to be exposed because they have higher quality than other items. Therefore, the higher exposure rate of these items can be well justified and may be utilized to further enhance the recommendation performance. In addition, recent research also found that confounders that spuriously correlate item exposures and user ratings may also help explain the co-occurrence patterns of different items [109]. Therefore, how to properly identify and utilize the positive side of biases while maximally suppressing their negative effects is of great importance and deserves more in-depth investigations in the future.

Finally, although recent years have witnessed the establishment and release of more real-world datasets for causal RS research from the industry, many causal RS models still rely heavily on simulated datasets for evaluation. The simulation can lead to the over-simplification of the problem and is often designed to correspond exactly with the debiasing/disentanglement mechanism of the proposed model. Therefore, the effectiveness of these methods in more complicated real-world scenarios is still uncertain due to the lack of model deployment and online tests. As such, to more convincingly demonstrate the practical utility of causal RSs, more collaborations with the industry are highly expected.

## 10.7 Summary

In this chapter, we provide a comprehensive overview of recent advances in causal inference for RSs. We start by pointing out issues of traditional RSs that rely on correlations in observed user behaviors and user/item features. We then introduce two mainstream causal inference frameworks, i.e., Rubin’s RCM and Pearl’s SCM, which provide deeper insights into these issues and the foundation for moving traditional RSs to the upper rungs of the Ladder of Causality. Specifically, we thoroughly discuss several state-of-the-art causal RS models that lead to enhanced robustness to various biases and improved explainability. In addition, since causal RSs can base recommendations on causal relationships that are stable and invariant, we also demonstrate that their generalization abilities can be significantly improved. Finally, we introduce evaluation strategies for causal RSs, with an emphasis on how to reliably estimate the model performance based on biased real-world data. We further compile real-world datasets where expensive randomized experiments are conducted on users, which reflects growing attention to causal RSs from the industry.

Overall, causal RS is still a relatively new and under-explored research topic. More efforts are urgently demanded to systematize the existing works and conduct deeper investigations for further improvements. Accordingly, we point out four interesting and practically important open problems in causal RSs. We hope that this chapter can help readers gain a comprehensive understanding of the main idea of applying causality in RSs and encourage further progress in this promising area.

Acknowledgments This work is supported by the National Science Foundation under grants IIS-2006844, IIS-2144209, IIS-2223769, CNS-2154962, and BCS-2228534, the JP Morgan Chase Faculty Research Award, and the Cisco Faculty Research Award.

## References

1. H. Abdollahpouri, R. Burke, B. Mobasher, Controlling popularity bias in learning-to-rank recommendation, in Proceedings of the 11th ACM Conference on Recommender Systems (2017), pp. 42–46  
2. H. Abdollahpouri et al., The unfairness of popularity bias in recommendation, in RecSys Workshop on Recommendation in Multistakeholder Environments (2019)  
3. A. Agarwal et al., A general framework for counterfactual learning-to-rank, in Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (2019), pp. 5–14  
4. Q. Ai et al., Unbiased learning to rank with unbiased propensity estimation, in The 41st International ACM SIGIR Conference on Research and Development in Information Retrieval (2018), pp. 385–394  
5. M. Arjovsky et al., Invariant risk minimization (2019). arXiv preprint  
6. S. Bonner, F. Vasile, Causal embeddings for recommendation, in Proceedings of the 12th ACM Conference on Recommender Systems (2018), pp. 104–112  
7. L. Bottou et al., Counterfactual reasoning and learning systems: the example of computational advertising. J. Mach. Learn. Res. 14(11), 3207–3260 (2013)  
8. D. Bouneffouf, A. Bouzeghoub, A.L. Gançarski, A contextual-bandit algorithm for mobile context-aware recommender system, in International Conference on Neural Information Processing (Springer, 2012), pp. 324–331  
9. E. Çano, M. Morisio, Hybrid recommender systems: a systematic literature review. Intell. Data Anal. 21(6), 1487–1524 (2017)  
10. J. Chen et al., AutoDebias: learning to debias for recommendation, in Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (2021), pp. 21–30  
11. J. Chen et al., Bias and debias in recommender system: a survey and future directions (2020). arXiv preprint arXiv:2010.03240  
12. Z. Chen et al., Co-training disentangled domain adaptation network for leveraging popularity bias in recommenders, in Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (2022), pp. 60–69  
13. E. Diemert et al., A large scale benchmark for uplift modeling, in Proceedings of AdKDD and TargetAd Workshop (2018)  
14. Y. Dong et al., fairness in graph mining: a survey (2022). arXiv preprint  
15. C. Gao et al., Causal inference in recommender systems: a survey and future directions (2022). arXiv preprint arXiv:2208.12397  
16. C. Gao et al., KuaiRec: a fully-observed dataset and insights for evaluating recommender systems, in Proceedings of the 31st ACM International Conference on Information and Knowledge Management (2022)  
17. Y. Gong, Q. Zhang, Hashtag recommendation using attention-based convolutional neural network, in Proceedings of the 25th International Joint Conference on Artificial Intelligence (2016), pp. 2782–2788  
18. I. Goodfellow et al., Generative adversarial networks. Commun. ACM 63(11), 139–144 (2020)  
19. P. Gopalan, J.M. Hofman, D.M. Blei, Scalable recommendation with hierarchical Poisson factorization, in Proceedings of the 31th Conference on Uncertainty in Artificial Intelligence (2015), pp. 326–335  
20. F.M. Harper, J.A. Konstan, The MovieLens datasets: history and context. ACM Trans. Interactive Intell. Syst. 5(4), 1–19 (2015)  
21. X. He, T.-S. Chua, Neural factorization machines for sparse predictive analytics, in Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (2017), pp. 355–364  
22. Y. Hu, Y. Koren, C. Volinsky, Collaborative filtering for implicit feedback datasets, in The 8th IEEE International Conference on Data Mining (2008), pp. 263–272  
23. G.W. Imbens, D.B. Rubin, Causal Inference in Statistics, Social, and Biomedical Sciences (Cambridge University Press, Cambridge, 2015)  
24. J. Kaddour et al., Causal machine learning: a survey and open problems (2022). arXiv preprint arXiv:2206.15475  
25. D.P. Kingma, M. Welling, Auto-encoding variational Bayes, in International Conference on Learning Representations (2014)  
26. D. Koller, N. Friedman, Probabilistic Graphical Models: Principles and Techniques (The MIT Press, Cambridge, MA, 2009). ISBN: 0-262-01319-3, https://books.google.com/books? id=7dzpHCHzNQ4C&pgis=1  
27. Y. Koren, Factorization meets the neighborhood: a multifaceted collaborative filtering model, in Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2008), pp. 426–434  
28. Y. Koren, R. Bell, C. Volinsky, Matrix factorization techniques for recommender systems. IEEE Comput. 42(8), 30–37 (2009)  
29. Y. Koren, S. Rendle, R. Bell, Advances in collaborative filtering, in Recommender Systems Handbook (Springer, New York, 2022), pp. 91–142  
30. Q. Li et al., Be causal: de-biasing social network confounding in recommendation. ACM Trans. Knowl. Disc. Data 17(1), 1–23 (2022)  
31. Y. Li et al., Causal factorization machine for robust recommendation, in Proceedings of the 22nd ACM/IEEE Joint Conference on Digital Libraries (2022), pp. 1–9  
32. Y. Li et al., Fairness in recommendation: a survey (2022). arXiv preprint arXiv:2205.13619  
33. Y. Li et al., Towards personalized fairness based on causal notion, in Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (2021), pp. 1054–1063  
34. Y. Li et al., User-oriented fairness in recommendation, in Proceedings of The Web Conference 2021 (2021), pp. 624–632  
35. D. Liang, L. Charlin, D.M. Blei, Causal inference for recommendation, in Causation: Foundation to Application, Workshop at UAI. AUAI (2016)  
36. D. Liang et al., Variational autoencoders for collaborative filtering, in Proceedings of the World Wide Web Conference (2018), pp. 689–698  
37. D. Liu et al., A general knowledge distillation framework for counterfactual recommendation via uniform data, in Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (2020), pp. 831–840  
38. J. Liu, P. Dolan, E.R. Pedersen, Personalized news recommendation based on click behavior, in Proceedings of the 15th International Conference on Intelligent User Interfaces (2010), pp. 31–40  
39. P. Lops, M. de Gemmis, G. Semeraro. Content-based recommender systems: state of the art and trends, in Recommender Systems Handbook (Springer, 2011), pp. 73–105  
40. J. Ma et al., Learning disentangled representations for recommendation, in Advances in Neural Information Processing Systems (2019)  
41. J. Ma, J. Li, Learning causality with graphs. AI Mag. 43(4), 365–375 (2022)  
42. J. Ma et al., Learning causal effects on hypergraphs, in ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2022)  
43. J. Ma et al., Multi-cause effect estimation with disentangled confounder representation, in International Joint Conference on Artificial Intelligence (2021), pp. 2790–2796  
44. B.M. Marlin, R.S. Zemel, Collaborative prediction and ranking with non-random missing data, in Proceedings of the 3rd ACM Conference on Recommender Systems (2009), pp. 5–12  
45. B.M. Marlin et al. Collaborative filtering and the missing at random assumption, in Proceedings of the 23rd Conference on Uncertainty in Artificial Intelligence (2007), pp. 267–275  
46. A. Mnih, R.R. Salakhutdinov, Probabilistic matrix factorization, in Advances in Neural Information Processing Systems (2007)  
47. I. Paparrizos, B.B. Cambazoglu, A. Gionis, Machine learned job recommendation, in Proceedings of the 5th ACM Conference on Recommender Systems (2011), pp. 325–328  
48. J. Pearl, Direct and indirect effects, in Paper presented at Proceedings of the Seventeenth Conference on Uncertainty in Artificial Intelligence (2001)  
49. J. Pearl, Causality (Cambridge University Press, Cambridge, 2009)  
50. J. Pearl, D. Mackenzie, The Book of Why: The New Science of Cause and Effect (Basic books, New York, 2018)  
51. S. Rendle, Factorization machines, in IEEE International Conference on Data Mining (IEEE, 2010), pp. 995–1000  
52. S. Rendle et al., BPR: Bayesian personalized ranking from implicit feedback, in Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (2009), pp. 452–461  
53. F. Ricci, L. Rokach, B. Shapira, Introduction to recommender systems handbook, in Recommender Systems Handbook (Springer, New York, 2011), pp. 1–35  
54. T.S. Richardson, J.M. Robins, Single world intervention graphs (SWIGs): a unification of the counterfactual and graphical approaches to causality. Center Statis. Soc. Sci., University of Washington Series 128(30), 2013 (2013)  
55. P.R. Rosenbaum, D.B. Rubin, The central role of the propensity score in observational studies for causal effects. Biometrika 70(1), 41–55 (1983)  
56. Y. Saito et al., Large-scale open dataset, pipeline, and benchmark for bandit algorithms (2020). arXiv preprint arXiv:2008.07146  
57. Y. Saito et al., Unbiased recommender learning from missing-not-at-random implicit feedback, in Proceedings of the 13th International Conference on Web Search and Data Mining (2020), pp. 501–509  
58. M. Sato et al., Unbiased learning for the causal effect of recommendation, in The 14th ACM Conference on Recommender Systems (2020), pp. 378–387  
59. M. Sato et al., Uplift-based evaluation and optimization of recommenders, in Proceedings of the 13th ACM Conference on Recommender Systems (2019), pp. 296–304  
60. T. Schnabel et al., Recommendations as treatments: debiasing learning and evaluation, in International Conference on Machine Learning (2016), pp. 1670–1679  
61. G. Shani, A. Gunawardana, Evaluating recommendation systems, in Recommender Systems Handbook (Springer, New York, 2011), pp. 257–297  
62. A. Sharma, J.M. Hofman, D.J. Watts, Estimating the causal impact of recommendation systems from observational data, in Proceedings of the 16th ACM Conference on Economics and Computation (2015), pp. 453–470  
63. P. Sheth et al., Causal disentanglement with network information for debiased recommendations, in International Conference on Similarity Search and Applications (2022), pp. 265–273  
64. H. Steck, Item popularity and recommendation accuracy, in Proceedings of the 5th ACM Conference on Recommender Systems (2011), pp. 125–132  
65. H. Steck, Training and testing of recommender systems on data missing not at random, in Proceedings of the 16th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2010), pp. 713–722  
66. R. Suter et al., Robustly disentangled causal mechanisms: validating deep representations for interventional robustness, in International Conference on Machine Learning (2019), pp. 6056–6065  
67. J. Tan et al., Counterfactual explainable recommendation, in Proceedings of the 30th ACM International Conference on Information & Knowledge Management (2021), pp. 1784–1793  
68. C. Wang, D.M. Blei, Collaborative topic modeling for recommending scientific articles, in Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2011), pp. 448–456  
69. H. Wang et al., ESCM2: entire space counterfactual multi-task model for post-click conversion rate estimation, in Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (2022), pp. 363–372  
70. H. Wang et al., Exploiting POI-specific geographical influence for point-of-interest recommendation, in Proceedings of the 27th International Joint Conference on Artificial Intelligence (2018), pp. 3877–3883  
71. W. Wang et al., Causal representation learning for out-of-distribution recommendation, in Proceedings of the ACM Web Conference 2022 (2022), pp. 3562–3571  
72. W. Wang et al., Clicks can be cheating: counterfactual recommendation for mitigating clickbait issue, in Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (2021), pp. 1288–1297  
73. W. Wang et al., Deconfounded recommendation for alleviating bias amplification, in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (2021), pp. 1717–1725  
74. X. Wang et al., Causal disentanglement for semantics-aware intent learning in recommendation, in IEEE Transactions on Knowledge and Data Engineering (2022)  
75. X. Wang et al., Combating selection biases in recommender systems with a few unbiased ratings, in Proceedings of the 14th ACM International Conference on Web Search and Data Mining (2021), pp. 427–435  
76. X. Wang et al., Position bias estimation for unbiased learning to rank in personal search, in Proceedings of the 11th ACM International Conference on Web Search and Data Mining (2018), pp. 610–618  
77. Y. Wang, D.M. Blei, The blessings of multiple causes. J. Am. Statist. Assoc. 114(528), 1574– 1596 (2019)  
78. Y. Wang et al., Causal inference for recommender systems, in The 14th ACM Conference on Recommender Systems (2020), pp. 426–431  
79. Z. Wang et al., Unbiased sequential recommendation with latent confounders, in Proceedings of the ACM Web Conference 2022 (2022), pp. 2195–2204  
80. T. Wei et al., Model-agnostic counterfactual reasoning for eliminating popularity bias in recommender system, in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (2021), pp. 1791–1800  
81. Y. Wei et al., MMGCN: multi-modal graph convolution network for personalized recommendation of micro-video, in Proceedings of the 27th ACM International Conference on Multimedia (2019), pp. 1437–1445  
82. P. Wu et al., On the opportunity of causal learning in recommendation systems: foundation, estimation, prediction and challenges, in Proceedings of the International Joint Conference on Artificial Intelligence, Vienna, Austria (2022), pp. 23–29  
83. Y. Wu et al. Collaborative denoising auto-encoders for top-N recommender systems, in Proceedings of the 9th ACM International Conference on Web Search and Data Mining (2016), pp. 153–162  
84. J. Xie et al., A multimodal variational encoder-decoder framework for micro-video popularity prediction, in Proceedings of the Web Conference 2020 (2020), pp. 2542–2548  
85. S. Xu et al., Causal collaborative filtering (2021). arXiv preprint arXiv:2102.01868  
86. S. Xu et al., Deconfounded causal collaborative filtering (2021). arXiv preprint arXiv:2110.07122  
87. S. Xu et al., Dynamic causal collaborative filtering, in Proceedings of the 31st ACM International Conference on Information and Knowledge Management (2022), pp. 2301– 2310  
88. S. Xu et al., Learning causal explanations for recommendation, in The 1st International Workshop on Causality in Search and Recommendation (2021)  
89. H.-J. Xue et al., Deep matrix factorization models for recommender systems. Int. Joint Conf. Artif. Intell. 17, 3203–3209 (2017)  
90. C. Yang et al., Towards out-of-distribution sequential event prediction: a causal treatment, in Advances in Neural Information Processing Systems, 35, 22656–22670 (2022)  
91. M. Yang et al., CausalVAE: disentangled representation learning via neural structural causal models, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2021), pp. 9593–9602  
92. L. Yao et al., A survey on causal inference. ACM Trans. Knowl. Disc. Data (TKDD) 15(5), 1–46 (2021)

<!-- footnote -->

- We use rating to refer to any user–item interaction that can be represented by a numerical value. This includes both explicit feedback such as likes/dislikes and implicit feedback such as views and clicks. When $r _ { i j }$ represents implicit feedback, the missing elements $r _ { i k } ^ { 0 }$ in R may be used as weak negative feedback in the training phase [22]. This may complicate the causal problems. Therefore, we assume RSs are trained on observed ratings to simplify the discussion unless specified otherwise.
- However, we do not distinguish random variables and their specific realizations if there is no risk of confusion. For simplicity, we assume R to be Gaussian unless specified otherwise.
- For works that do not explicitly treat $r _ { i j }$ as a random variable, we assume it follows a Gaussian distribution with zero variance. The generative process then becomes as $r _ { i j } = { \bf u } _ { i } ^ { T } \cdot { \bf v } _ { j }$ .

<!-- footnote end -->

<!-- footnote -->

- which can be attributed to multiple reasons such as users’ self-search [75], the recommendations of previous models [37], the position where the items are displayed [76], item popularity [1], etc. Generally, RCM-based causal RSs are agnostic to the specific reason that causes the exposure bias.

<!-- footnote end -->

<!-- footnote -->

- In the uplift evaluation of RSs that aims to estimate how recommendations change user behaviors [62], $r _ { i j } ( a _ { i j } = 0 )$ may be used to represent user i’s rating to item j through self-searching [59].

<!-- footnote end -->

<!-- footnote -->

- We can gain an intuition of this claim from Fig. 10.2. Suppose covariates $C$ represent the twodimensional features (user type, movie type). Given $C = \mathbf { c } , r _ { i j } ( a _ { i j } = 1 ) \perp a _ { i j }$ | c described in Eq. (10.3) is satisfied because in each data stratum specified by $C = \mathbf { c } \left( \mathrm { i . e . } \right.$ ., the four $2 \times 2$ blocks in Fig. 10.2b), $r _ { i j } ( a _ { i j } = 1 )$ is constant. Fig. 10.2a shows that for the treatment group $\mathcal { T } ,$ $p ( \mathbf { c } | a _ { i j } = 1 ) = 1 / 2$ for $\mathbf { c } \in C _ { 1 } =$ {(horror fan, horror movie), (romance fan, romance movie)} and $p ( \mathbf { c } | a _ { i j } = 1 ) = 0$ for $\mathbf { c } \in C _ { 2 } =$ {(horror fan, romance movie), (romance fan, horror movie)}. In contrast, for the population PO, $p ( \mathbf { c } ) = 1 / 4$ for $\mathbf { c } \in C _ { 1 } \cup C _ { 2 }$ . Therefore, in the treatment group $\mathcal { T } ,$ user-item pairs with covariates in $C _ { 1 }$ are over-represented, while those with covariates in $C _ { 2 }$ are under-represented. However, we also note that this case is too extreme to be addressed by RCM, as $p ( \mathbf { c } | a _ { i j } = 1 ) = 0 { \mathrm { f o r } } C \in C _ { 2 }$ violates the positivity assumption mentioned in the attention box.

<!-- footnote end -->

<!-- footnote -->

- In causal graphs, the subscripts i, j for each node are omitted for simplicity.
- We also omit the mutually independent exogenous variables for each node and summarize their randomness into the structural equations with probability distributions [15]. Subscript G is used to distinguish structural equations from other conditional relationships that can be inferred from G.

<!-- footnote end -->

<!-- footnote -->

- This corresponds to the case where item exposures are randomized (see the discussions in Sect. 10.3.1.3), as the user–item pair (U, V ) is not determined by other factors associated with R [54].

<!-- footnote end -->

<!-- footnote -->

- The similarity between this section and Sect. 10.3.1.1 shows us the connection between RCMbased and SCM-based causal RSs, where the claim that when item exposure is not randomized, “observing that an item was exposed to the user per se contains extra information about the useritem pair” is mathematically transformed into the abductive inference of c from $\mathbf { v } _ { j }$ by $p ( \mathbf { c } | \mathbf { v } _ { j } )$ ).

<!-- footnote end -->

<!-- footnote -->

- Consider again the toy example in Fig. 10.5. If we know exactly the user type and item type c for each user–item pair, the predictions can be unbiased even if the item exposures are nonrandomized.

<!-- footnote end -->

<!-- footnote -->

- https://www.cs.cornell.edu/\~schnabts/mnar/
- https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=3
- https://github.com/chongminggao/KuaiRec
- http://cail.criteo.com/criteo-uplift-prediction-dataset/

<!-- footnote end -->

<!-- footnote -->

- https://research.zozo.com/data.html

<!-- footnote end -->

93. M. Ye et al., Exploiting geographical influence for collaborative point-of-interest recommendation, in Proceedings of the 34th International ACM SIGIR Conference on Research and Development in Information Retrieval (2011), pp. 325–334  
94. J. Yi, Z. Chen, Debiased cross-modal matching for content-based micro-video background music recommendation (2022). arXiv preprint arXiv:2208.03633  
95. J. Yi et al., Cross-modal variational auto-encoder for content-based micro-video background music recommendation, in IEEE Transactions on Multimedia (2021)  
96. S. Zhang et al., Deep learning based recommender system: a survey and new perspectives. ACM Comput. Surv. 52(1), 1–38 (2019)  
97. W. Zhang et al., Large-scale causal approaches to debiasing post-click conversion rate estimation with multi-task learning, in Proceedings of the Web Conference 2020 (2020), pp. 2775–2781  
98. Y. Zhang et al., Causal intervention for leveraging popularity bias in recommendation, in Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (2021), pp. 11–20  
99. Y. Zhang et al., Causal Recommendation: Progresses and Future Directions. Tutorial for The Web Conference 2022. https://causalrec.github.io/file/www2022-tutorial-CausalRec.pdf. 26 Apr 2022  
100. Y. Zhang, X. Chen et al., Explainable recommendation: a survey and new perspectives. Found. Trends® Inf. Retr. 14(1), 1–101 (2020)  
101. Z. Zhao et al., Popularity bias is not always evil: disentangling benign and harmful bias for recommendation. IEEE Trans. Knowl. Data Eng. 99, 1–13 (2022)  
102. Y. Zheng et al., Disentangling user interest and conformity for recommendation with causal embedding, in Proceedings of the Web Conference 2021 (2021), pp. 2980–2991  
103. C. Zhou et al., Contrastive learning for debiased candidate generation in large-scale recommender systems, in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (2021), pp. 3985–3995  
104. G. Zhou et al., Deep interest network for click-through rate prediction, in Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2018), pp. 1059–1068  
105. F. Zhu et al., DCMT: a direct entire-space causal multi-task frame-work for post-click conversion estimation (2023). arXiv preprint arXiv:2302.06141  
106. X. Zhu et al., Mitigating hidden confounding effects for causal recommendation (2022). arXiv preprint arXiv:2205.07499  
107. Y. Zhu, Z. Chen, Mutually-regularized dual collaborative variational auto-encoder for recommendation systems, in Proceedings of The ACM Web Conference 2022 (2022), pp. 2379–2387  
108. Y. Zhu, Z. Chen, Variational bandwidth auto-encoder for hybrid recommender systems, in IEEE Transactions on Knowledge and Data Engineering (2022)  
109. Y. Zhu et al., Deep causal reasoning for recommendations (2022). arXiv preprint arXiv:2201.02088  
110. Y. Zhu et al., Deep deconfounded content-based tag recommendation for UGC with causal intervention (2022). arXiv preprint arXiv:2205.14380  
111. Z. Zhu et al., Popularity bias in dynamic recommendation, in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (2021), pp. 2439–2449  
112. Z. Zhu et al., Unbiased implicit recommendation and propensity estimation via combinational joint learning, in The 14th ACM Conference on Recommender Systems (2020), pp. 551–556  
113. H. Zou et al., Counterfactual prediction for bundle treatment, in Advances in Neural Information Processing Systems (2020), pp. 19705–19715
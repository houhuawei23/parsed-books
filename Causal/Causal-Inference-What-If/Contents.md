## Contents

- [Introduction: Towards less casual causal inferences](./Introduction.md)

- [I Causal inference without models](./en/1_Causal_inference_without_models/)

- [1 A definition of causal effect](./en/1_Causal_inference_without_models/01_Chapter_1_A_DEFINITION_OF_CAUSAL_EFFECT.md)
  - 1.1 Individual causal effects
  - 1.2 Average causal effects
  - 1.3 Measures of causal effect
  - 1.4 Random variability
  - 1.5 Causation versus association

- [2 Randomized experiments](./en/1_Causal_inference_without_models/02_Chapter_2_RANDOMIZED_EXPERIMENTS.md)
  - 2.1 Randomization
  - 2.2 Conditional randomization
  - 2.3 Standardization
  - 2.4 Inverse probability weighting

- [3 Observational studies](./en/1_Causal_inference_without_models/03_Chapter_3_OBSERVATIONAL_STUDIES.md)
  - 3.1 Identifiability conditions
  - 3.2 Exchangeability
  - 3.3 Positivity
  - 3.4 Consistency: First, define the counterfactual outcome
  - 3.5 Consistency: Second, link counterfactuals to the observed data
  - 3.6 The target trial

- [4 Effect modification](./en/1_Causal_inference_without_models/04_Chapter_4_EFFECT_MODIFICATION.md)
  - 4.1 Heterogeneity of treatment effects
  - 4.2 Stratification to identify effect modification
  - 4.3 Why care about effect modification
  - 4.4 Stratification as a form of adjustment
  - 4.5 Matching as another form of adjustment
  - 4.6 Effect modification and adjustment methods

- [5 Interaction](./en/1_Causal_inference_without_models/05_Chapter_5_INTERACTION.md)
  - 5.1 Interaction requires a joint intervention
  - 5.2 Identifying interaction
  - 5.3 Counterfactual response types and interaction
  - 5.4 Sufficient causes
  - 5.5 Sufficient cause interaction
  - 5.6 Counterfactuals or sufficient-component causes?

- [6 Graphical representation of causal effects](./en/1_Causal_inference_without_models/06_Chapter_6_GRAPHICAL_REPRESENTATION_OF_CAUSAL_EFFECTS.md)
  - 6.1 Causal diagrams
  - 6.2 Causal diagrams and marginal independence
  - 6.3 Causal diagrams and conditional independence
  - 6.4 Positivity and consistency in causal diagrams
  - 6.5 A structural classification of bias
  - 6.6 The structure of effect modification

- [7 Confounding](./en/1_Causal_inference_without_models/07_Chapter_7_CONFOUNDING.md)
  - 7.1 The structure of confounding
  - 7.2 Confounding and exchangeability
  - 7.3 Confounding and the backdoor criterion
  - 7.4 Confounding and confounders
  - 7.5 Single-world intervention graphs
  - 7.6 Confounding adjustment

- [8 Selection bias](./en/1_Causal_inference_without_models/08_Chapter_8_SELECTION_BIAS.md)
  - 8.1 The structure of selection bias
  - 8.2 Examples of selection bias
  - 8.3 Selection bias and confounding
  - 8.4 Selection bias and censoring
  - 8.5 How to adjust for selection bias
  - 8.6 Selection without bias

- [9 Measurement bias and "Noncausal" diagrams](./en/1_Causal_inference_without_models/09_Chapter_9_MEASUREMENT_BIAS_AND_"NONCAUSAL"_DIAGRAMS.md)
  - 9.1 Measurement error
  - 9.2 The structure of measurement error
  - 9.3 Mismeasured confounders and colliders
  - 9.4 Causal diagrams without mismeasured variables?
  - 9.5 Many proposed causal diagrams include noncausal arrows
  - 9.6 Does it matter that many proposed diagrams include noncausal arrows?

- [10 Random variability](./en/1_Causal_inference_without_models/10_Chapter_10_RANDOM_VARIABILITY.md)
  - 10.1 Identification versus estimation
  - 10.2 Estimation of causal effects
  - 10.3 The myth of the super-population
  - 10.4 The conditionality "principle"
  - 10.5 The curse of dimensionality

- [II Causal inference with models](./en/2_Causal_inference_with_models/)

- [11 Why model?](./en/2_Causal_inference_with_models/11_Chapter_11_WHY_MODEL.md)
  - 11.1 Data cannot speak for themselves
  - 11.2 Parametric estimators of the conditional mean
  - 11.3 Nonparametric estimators of the conditional mean
  - 11.4 Smoothing
  - 11.5 The bias-variance trade-off

- [12 IP weighting and marginal structural models](./en/2_Causal_inference_with_models/12_Chapter_12_IP_WEIGHTING_AND_MARGINAL_STRUCTURAL_MODELS.md)
  - 12.1 The causal question
  - 12.2 Estimating IP weights via modeling
  - 12.3 Stabilized IP weights
  - 12.4 Marginal structural models
  - 12.5 Effect modification and marginal structural models
  - 12.6 Censoring and missing data

- [13 Standardization and the parametric g-formula](./en/2_Causal_inference_with_models/13_Chapter_13_STANDARDIZATION_AND_THE_PARAMETRIC_G-FORMULA.md)
  - 13.1 Standardization as an alternative to IP weighting
  - 13.2 Estimating the mean outcome via modeling
  - 13.3 Standardizing the mean outcome to the confounder distribution
  - 13.4 IP weighting or standardization?
  - 13.5 How seriously do we take our estimates?

- [14 G-estimation of structural nested models](./en/2_Causal_inference_with_models/14_Chapter_14_G-ESTIMATION_OF_STRUCTURAL_NESTED_MODELS.md)
  - 14.1 The causal question revisited
  - 14.2 Exchangeability revisited
  - 14.3 Structural nested mean models
  - 14.4 Rank preservation
  - 14.5 G-estimation
  - 14.6 Structural nested models with two or more parameters

- [15 Outcome regression and propensity scores](./en/2_Causal_inference_with_models/15_Chapter_15_OUTCOME_REGRESSION_AND_PROPENSITY_SCORES.md)
  - 15.1 Outcome regression
  - 15.2 Propensity scores
  - 15.3 Propensity stratification and standardization
  - 15.4 Propensity matching
  - 15.5 Propensity models, structural models, predictive models

- [16 Instrumental variable estimation](./en/2_Causal_inference_with_models/16_Chapter_16_INSTRUMENTAL_VARIABLE_ESTIMATION.md)
  - 16.1 The three instrumental conditions
  - 16.2 The usual IV estimand
  - 16.3 A fourth identifying condition: homogeneity
  - 16.4 An alternative fourth condition: monotonicity
  - 16.5 The three instrumental conditions revisited
  - 16.6 Instrumental variable estimation versus other methods

- [17 Causal survival analysis](./en/2_Causal_inference_with_models/17_Chapter_17_CAUSAL_SURVIVAL_ANALYSIS.md)
  - 17.1 Hazards and risks
  - 17.2 From hazards to risks
  - 17.3 Why censoring matters
  - 17.4 IP weighting of marginal structural models
  - 17.5 The parametric g-formula
  - 17.6 G-estimation of structural nested models

- [18 Variable selection and high-dimensional data](./en/2_Causal_inference_with_models/18_Chapter_18_VARIABLE_SELECTION_AND_HIGH-DIMENSIONAL_DATA.md)
  - 18.1 The different goals of variable selection
  - 18.2 Variables that induce or amplify bias
  - 18.3 Causal inference and machine learning
  - 18.4 Doubly robust machine learning estimators
  - 18.5 Variable selection is a difficult problem

- [III Causal inference for time-varying treatments](./en/3_Causal_inference_for_time-varying_treatments/)

- [19 Time-varying treatments](./en/3_Causal_inference_for_time-varying_treatments/19_Chapter_19_TIME-VARYING_TREATMENTS.md)
  - 19.1 The causal effect of time-varying treatments
  - 19.2 Treatment strategies
  - 19.3 Sequentially randomized experiments
  - 19.4 Sequential exchangeability
  - 19.5 Identifiability under some but not all treatment strategies
  - 19.6 Time-varying confounding and time-varying confounders

- [20 Treatment-confounder feedback](./en/3_Causal_inference_for_time-varying_treatments/20_Chapter_20_TREATMENT-CONFOUNDER_FEEDBACK.md)
  - 20.1 The elements of treatment-confounder feedback
  - 20.2 The bias of traditional methods
  - 20.3 Why traditional methods fail
  - 20.4 Why traditional methods cannot be fixed
  - 20.5 Adjusting for past treatment

- [21 G-methods for time-varying treatments](./en/3_Causal_inference_for_time-varying_treatments/21_Chapter_21_G-METHODS_FOR_TIME-VARYING_TREATMENTS.md)
  - 21.1 The g-formula for time-varying treatments
  - 21.2 IP weighting for time-varying treatments
  - 21.3 A doubly robust estimator for time-varying treatments
  - 21.4 G-estimation for time-varying treatments
  - 21.5 Censoring is a time-varying treatment
  - 21.6 The big g-formula

- [22 Target trial emulation](./en/3_Causal_inference_for_time-varying_treatments/22_Chapter_22_TARGET_TRIAL_EMULATION.md)
  - 22.1 Intention-to-treat effect and per-protocol effect
  - 22.2 A target trial with sustained treatment strategies
  - 22.3 Emulating a target trial with sustained strategies
  - 22.4 Time zero
  - 22.5 A unified approach to answer What If questions with data

- [23 Causal mediation](./en/3_Causal_inference_for_time-varying_treatments/23_Chapter_23_CAUSAL_MEDIATION.md)
  - 23.1 Mediation analysis under attack
  - 23.2 A defense of mediation analysis
  - 23.3 Empirically verifiable mediation
  - 23.4 An interventionist theory of mediation

- [References](./References.md)

- Index

## Contents

- **[1 Motivation: Why You Might Care](./en/01_Motivation__Why_You_Might_Care.md)**
  - Simpson's Paradox
  - Applications of Causal Inference
  - Correlation Does Not Imply Causation
    - Nicolas Cage and Pool Drownings
    - Why is Association Not Causation?
  - Main Themes

- **[2 Potential Outcomes](./en/02_Potential_Outcomes.md)**
  - Potential Outcomes and Individual Treatment Effects
  - The Fundamental Problem of Causal Inference
  - Getting Around the Fundamental Problem
    - Average Treatment Effects and Missing Data Interpretation
    - Ignorability and Exchangeability
    - Conditional Exchangeability and Unconfoundedness
    - Positivity/Overlap and Extrapolation
    - No interference, Consistency, and SUTVA
    - Tying It All Together
  - Fancy Statistics Terminology Defancified
  - A Complete Example with Estimation

- **[3 The Flow of Association and Causation in Graphs](./en/03_The_Flow_of_Association_and_Causation_in_Graphs.md)**
  - Graph Terminology
  - Bayesian Networks
  - Causal Graphs
  - Two-Node Graphs and Graphical Building Blocks
  - Chains and Forks
  - Colliders and their Descendants
  - d-separation
  - Flow of Association and Causation

- **[4 Causal Models](./en/04_Causal_Models.md)**
  - The do-operator and Interventional Distributions
  - The Main Assumption: Modularity
  - Truncated Factorization
    - Example Application and Revisiting "Association is Not Causation"
  - The Backdoor Adjustment
    - Relation to Potential Outcomes
  - Structural Causal Models (SCMs)
    - Structural Equations
    - Interventions
    - Collider Bias and Why to Not Condition on Descendants of Treatment
  - Example Applications of the Backdoor Adjustment
    - Association vs. Causation in a Toy Example
    - A Complete Example with Estimation
  - Assumptions Revisited

- **[5 Randomized Experiments](./en/05_Randomized_Experiments.md)**
  - Comparability and Covariate Balance
  - Exchangeability
  - No Backdoor Paths

- **[6 Nonparametric Identification](./en/06_Nonparametric_Identification.md)**
  - Frontdoor Adjustment
  - do-calculus
    - Application: Frontdoor Adjustment
  - Determining Identifiability from the Graph

- **[7 Estimation](./en/07_Estimation.md)**
  - Preliminaries
  - Conditional Outcome Modeling (COM)
  - Grouped Conditional Outcome Modeling (GCOM)
  - Increasing Data Efficiency
    - TARNet
    - X-Learner
  - Propensity Scores
  - Inverse Probability Weighting (IPW)
  - Doubly Robust Methods
  - Other Methods
  - Concluding Remarks
    - Confidence Intervals
    - Comparison to Randomized Experiments

- **[8 Unobserved Confounding: Bounds and Sensitivity Analysis](./en/08_Unobserved_Confounding__Bounds_and_Sensitivity_Analysis.md)**
  - Bounds
    - No-Assumptions Bound
    - Monotone Treatment Response
    - Monotone Treatment Selection
    - Optimal Treatment Selection
  - Sensitivity Analysis
    - Sensitivity Basics in Linear Setting
    - More General Settings

- **[9 Instrumental Variables](./en/09_Instrumental_Variables.md)**
  - What is an Instrument?
  - No Nonparametric Identification of the ATE
  - Warm-Up: Binary Linear Setting
  - Continuous Linear Setting
  - Nonparametric Identification of Local ATE
    - New Potential Notation with Instruments
    - Principal Stratification
    - Local ATE
  - More General Settings for ATE Identification

- **[10 Difference in Differences](./en/10_Difference_in_Differences.md)**
  - Preliminaries
  - Introducing Time
  - Identification
    - Assumptions
    - Main Result and Proof
  - Major Problems

- **[11 Causal Discovery from Observational Data](./en/11_Causal_Discovery_from_Observational_Data.md)**
  - Independence-Based Causal Discovery
    - Assumptions and Theorem
    - The PC Algorithm
    - Can We Get Any Better Identification?
  - Semi-Parametric Causal Discovery
    - No Identifiability Without Parametric Assumptions
    - Linear Non-Gaussian Noise
    - Nonlinear Models
  - Further Resources

- **[12 Causal Discovery from Interventional Data](./en/12_Causal_Discovery_from_Interventional_Data.md)**
  - Structural Interventions
    - Single-Node Interventions
    - Multi-Node Interventions
  - Parametric Interventions
    - Coming Soon
  - Interventional Markov Equivalence
    - Coming Soon
  - Miscellaneous Other Settings
    - Coming Soon

- **[13 Transfer Learning and Transportability](./en/13_Transfer_Learning_and_Transportability.md)**
  - Causal Insights for Transfer Learning
    - Coming Soon
  - Transportability of Causal Effects Across Populations
    - Coming Soon

- **[14 Counterfactuals and Mediation](./en/14_Counterfactuals_and_Mediation.md)**
  - Counterfactuals Basics
    - Coming Soon
  - Important Application: Mediation
    - Coming Soon

- **[Appendix](./en/15_Appendix.md)**
  - **A Proofs**
    - Proof of Equation 6.1 from Section 6.1
    - Proof of Propensity Score Theorem (7.1)
    - Proof of IPW Estimand (7.18)

- **[Bibliography](./en/16_Bibliography.md)**

- **[Alphabetical Index](./en/17_Alphabetical_Index.md)**

- **List of Figures**


- 1.1 Simpson's paradox in COVID-27 data 
- 2.1 Causal Inference as Missing Data Problem 
- 3.1 Exponential number of parameters for modeling factors

- Listings

- 2.1 Python code for estimating the ATE 17  
- 2.2 Python code for estimating the ATE using the coefficient of linear regression 17  
- 4.1 Python code for estimating the ATE, without adjusting for the collider . . 46

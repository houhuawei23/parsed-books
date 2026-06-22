# A First Course in Causal Inference

*arXiv:2305.18793v1 [stat.ME] 30 May 2023*



## Preface

I developed the lecture notes based on my “Causal Inference” course at the University of California Berkeley over the past seven years. Since half of the students were undergraduate, my lecture notes only require basic knowledge of probability theory, statistical inference, and linear and logistic regressions.

I am grateful for the constructive comments from many students. If you identify any errors, please feel free to email me.


**Acronyms**

<table><tr><td>acronym</td><td>full name</td><td>first chapter</td></tr><tr><td>RD</td><td>risk difference</td><td>1</td></tr><tr><td>RR</td><td>risk ratio or relative risk</td><td>1</td></tr><tr><td>OR</td><td>odds ratio</td><td>1</td></tr><tr><td>RCT</td><td>randomized controlled trial</td><td>1</td></tr><tr><td>BMI</td><td>body mass index</td><td>2</td></tr><tr><td>SUTVA</td><td>stable unit treatment value assumption</td><td>2</td></tr><tr><td>ACE</td><td>average causal effect</td><td>2</td></tr><tr><td>CRE</td><td>completely randomized experiment</td><td>3</td></tr><tr><td>BRE</td><td>Bernoulli randomized experiment</td><td>3</td></tr><tr><td>IID</td><td>independent and identically distributed</td><td>3 and A1</td></tr><tr><td>FRT</td><td>Fisher randomization test</td><td>3</td></tr><tr><td>OLS</td><td>ordinary least squares</td><td>4 and A2</td></tr><tr><td>EHW</td><td>Eicker-Huber-White (robust standard error)</td><td>4 and A2</td></tr><tr><td>SRE</td><td>stratified randomized experiment</td><td>5</td></tr><tr><td>ReM</td><td>rerandomization using the Mahalanobis distance</td><td>6</td></tr><tr><td>ANCOVA</td><td>analysis of covariance</td><td>6</td></tr><tr><td>LASSO</td><td>least absolute shrinkage and selection operator</td><td>6</td></tr><tr><td>MPE</td><td>matched-pairs experiment</td><td>7</td></tr><tr><td>NHANES</td><td>National Health and Nutrition Examination Survey</td><td>10</td></tr><tr><td>IPW</td><td>inverse propensity score weighting</td><td>11</td></tr><tr><td>HT</td><td>Horvitz-Thompson</td><td>11</td></tr><tr><td>WLS</td><td>weighted least squares</td><td>14 and A2</td></tr><tr><td>IV</td><td>instrumental variable</td><td>21</td></tr><tr><td>ITT</td><td>intention-to-treat (analysis)</td><td>21</td></tr><tr><td>CACE</td><td>complier average causal effect</td><td>21</td></tr><tr><td>LATE</td><td>local average treatment effect</td><td>21</td></tr><tr><td>TSLS</td><td>two-stage least squares</td><td>23</td></tr><tr><td>ILS</td><td>indirect least squares</td><td>23</td></tr><tr><td>MR</td><td>Mendelian randomization</td><td>25</td></tr><tr><td>SNP</td><td>single nucleotide polymorphism</td><td>25</td></tr><tr><td>NDE</td><td>natural direct effect</td><td>27</td></tr><tr><td>NIE</td><td>natural indirect effect</td><td>27</td></tr><tr><td>CDE</td><td>controlled direct effect</td><td>29</td></tr><tr><td>MSM</td><td>marginal structural model</td><td>29</td></tr><tr><td>FWL</td><td>Frisch-Waugh-Lovell (theorem)</td><td>A2</td></tr><tr><td>MLE</td><td>maximum likelihood estimate</td><td>A2</td></tr></table>


## Part I

## Introduction
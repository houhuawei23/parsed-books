# Application of the Instrumental Variable Method: Fuzzy Regression Discontinuity

The regression discontinuity introduced in Chapter 20 and the instrumental variable introduced in Chapters 21–23 are two important examples of natural experiments. The study designs are not as ideal as the randomized experiments in Part II, but they have features similar to the experiments. That’s why they are called natural experiments.

Compounding regression discontinuity with instrumental variable yields the fuzzy regression discontinuity, another important natural experiment. I will start with examples and then provide a mathematical formulation.

## 24.1 Motivating examples

Chapter 20 introduces the regression discontinuity. The following two examples are slightly different because the treatments received are not deterministic functions of the running variables. Rather, the running variables discontinuously change the probabilities of the treatments received at the cutoff point.

Example 24.1 In 2000, the Government of India launched the Prime Minister’s Village Road Program, and by 2015, this program had funded the construction of all-weather roads to nearly 200,000 villages. Based on village level data, Asher and Novosad (2020) use a regression discontinuity to estimate the effect of new feeder roads on various economic variables. The national program guidelines prioritized larger villages according to arbitrary thresholds based on the 2001 Population Census. The treatment variable equals one if the village received a new road before the year in which the outcomes were measured. The difference between the population size of a village and the threshold did not determine the treatment variable but affected its probability discontinuously at the cutoff point zero.

Example 24.2 Li et al. (2015) used the data on the first-year students enrolled in 2004 to 2006 from two Italian universities to evaluate the causal effect of a university grant on the drop out rate. The students were eligible for this grant if their standardized family income was below 15,000 euros. For simplicity, we use the running variable defined as 15,000 minus the standardized family income. To receive this grant, the students must apply first. Therefore, the eligibility and the application status jointly determined the final treatment status. The running variable alone did not determine the treatment status although it changed the treatment probability at the cutoff point zero.

![image_26](images/image_26.png)

pr(D = 1 | X = x)
1
x₀
X

![image_27](images/image_27.png)

pr(D = 1 | X = x)
1
x₀
X

FIGURE 24.1: The treatment assignments of sharp regression discontinuity (left) and fuzzy regression discontinuity (right)

Example 24.3 Amarante et al. (2016) estimated the impact of in utero exposure to a social assistance program on children’s birth outcomes. They used a regression discontinuity induced by the Uruguayan Plan de Atenci´on Nacional a la Emergencia Social. It was a temporary social assistance program targeted to the poorest 10 percent of households, implemented between April 2005 and December 2007. Households with a predicted low income score below a predetermined threshold were assigned to the program. The predicted income score did not determine whether the mother received at least one program transfer during the pregnancy but it changed the probability of the final treatment received. The birth outcomes included birth weight, weeks of gestation, etc.

The above examples are called fuzzy regression discontinuity in contrast to the (sharp) regression discontinuity in Chapter 20. I will analyze the data in Examples 24.1 and 24.2 in Section 24.3 below.

## 24.2 Mathematical formulation

Let $X _ { i }$ denote the running variable which determines $Z _ { i } ~ = ~ 1 ( X _ { i } ~ \geq ~ x _ { 0 } )$ with the cutoff point $x _ { 0 } .$ . The treatment received $D _ { i }$ may not equal $Z _ { i } ,$ but $\mathrm { p r } ( D _ { i } = 1 \mid X _ { i } = x )$ has a jump at $x _ { 0 }$ . Figure 24.1 compares the treatment received probabilities of the sharp regression discontinuity and fuzzy regression discontinuity. It shows a special case of fuzzy regression discontinuity with $\operatorname { p r } ( D = 1 \mid X < x _ { 0 } ) = 0$ , which is coherent to Example 24.2.

Let $Y _ { i }$ denote the outcome of interest. Viewing $Z _ { i }$ as the treatment assigned, we can define potential outcomes $\{ D _ { i } ( 1 ) , D _ { i } ( 0 ) , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \}$ . The sharp regression discontinuity of Z allows for identification of

$$
\begin{array}{l} \tau_ {D} (x _ {0}) = E \{D (1) - D (0) \mid X = x _ {0} \} \\ = \lim _ {\varepsilon \rightarrow 0 +} E (D \mid Z = 1, X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \rightarrow 0 +} E (D \mid Z = 0, X = x _ {0} - \varepsilon) \\ \end{array}
$$

and

$$
\begin{array}{l} \tau_ {Y} (x _ {0}) = E \{Y (1) - Y (0) \mid X = x _ {0} \} \\ = \lim _ {\varepsilon \rightarrow 0 +} E (Y \mid Z = 1, X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \rightarrow 0 +} E (Y \mid Z = 0, X = x _ {0} - \varepsilon) \\ \end{array}
$$

based on Theorem 20.2. Using $Z$ as an IV for D and imposing the IV assumptions at $X = x _ { 0 }$ , we can identify the local complier average causal effect by applying Theorem 21.1.

Theorem 24.1 Assume

$$
D _ {i} (1) \geq D _ {i} (0)
$$

and

$$
D _ {i} (1) = D _ {i} (0) \Longrightarrow Y _ {i} (1) = Y _ {i} (0)
$$

in the infinitesimal neighborhood $o f x _ { 0 }$ . The local complier average causal effect equals

$$
\begin{array}{l} \tau_ {\mathrm{c}} (x _ {0}) \equiv E \{Y (1) - Y (0) \mid D (1) > D (0), X = x _ {0} \} \\ = \frac {E \{Y (1) - Y (0) \mid X = x _ {0} \}}{E \{D (1) - D (0) \mid X = x _ {0} \}}. \\ \end{array}
$$

Further assume that $E \{ D ( 1 ) \mid X = x \}$ and $E \{ Y ( 1 ) \mid X = x \}$ are continuous from the right at $X = x _ { 0 } \quad$ , and $E \{ D ( 0 ) \mid X = x \}$ and $E \{ Y ( 0 ) \mid X = x \}$ are continuous from the $l e f t$ at $X = x _ { 0 }$ . The local complier average causal effect can be identified by

$$
\tau_ {\mathrm{c}} (x _ {0}) = \frac {\lim _ {\varepsilon \to 0 +} E (Y \mid Z = 1 , X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \to 0 +} E (Y \mid Z = 0 , X = x _ {0} - \varepsilon)}{\lim _ {\varepsilon \to 0 +} E (D \mid Z = 1 , X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \to 0 +} E (D \mid Z = 0 , X = x _ {0} - \varepsilon)}
$$

if the $E ( D \mid Z = 1 , X = x )$ has a non-zero jump at $X = x _ { 0 }$

Theorem 24.1 is a superposition of Theorems 20.2 and 21.1. I leave its proof as Problem 24.1.

In both sharp and fuzzy regression discontinuity, the key is to specify the neighborhood around the cutoff point. Practically, a smaller neighborhood leads to smaller bias but larger variance, while a larger neighborhood leads to larger bias but smaller variance. That is, we face a bias-variance tradeoff. Some automatic procedures exist based on some statistical criteria, which relies on some strong conditions. It seems wiser to conduct sensitivity analysis over a range of the choice of $h .$

## 29624 Application of the Instrumental Variable Method: Fuzzy Regression Discontinuity

Assume that we have specified the neighborhood of $x _ { 0 }$ determined by a bandwidth h. For data with $X _ { i } \in [ x _ { 0 } - h , x _ { 0 } + h ]$ , we can estimate $\tau _ { D } ( x _ { 0 } )$ by

τˆD(x0) = the coefficient of $Z _ { i }$ in the OLS fit of $D _ { i }$ on $\{ 1 , Z _ { i } , R _ { i } , L _ { i } \}$ ,

and estimate $\tau _ { Y } ( x _ { 0 } )$

τˆY (x0) = the coefficient of $Z _ { i }$ in the OLS fit of $Y _ { i }$ on $\{ 1 , Z _ { i } , R _ { i } , L _ { i } \}$ ,

recalling the definitions $R _ { i } = \operatorname* { m a x } ( X _ { i } - x _ { 0 } , 0 )$ and $L _ { i } = \operatorname* { m i n } ( X _ { i } - x _ { 0 } , 0 )$ . Then we can estimate the local complier average causal effect by

$$
\hat {\tau} _ {\mathrm{c}} (x _ {0}) = \hat {\tau} _ {Y} (x _ {0}) / \hat {\tau} _ {D} (x _ {0}).
$$

This is an indirect least squares estimator. By Theorem 23.1, it is numerically identical to

the coefficient of $D _ { i }$ in the TSLS fit of $Y _ { i }$ on $\{ 1 , D _ { i } , R _ { i } , L _ { i } \}$

with $D _ { i }$ instrumented by $Z _ { i }$ . In sum, after specifying h, the estimation of $\tau _ { \mathrm { c } } ( x _ { 0 } )$ reduces to a TSLS procedure with the local data around the cutoff point.

## 24.3 Application

## 24.3.1 Re-analyzing Asher and Novosad (2020)’s data

Figure 24.2 shows the result using occupationindexandrsn as the outcome.

The package rdrobust selects the bandwidth automatically. The results suggest that receiving a new road did not affect the outcome significantly.

```diff
> road_dat = read.csv("indianroad.csv")
> road_dat$runv = road_dat$left + road_dat$right
> library("rdrobust")
> frd_road = with(road_dat,
+    {
+    rdrobust(y = occupation_index_andrsn,
+    x = runv,
+    c = 0,
+    fuzzy = r2012)
+    })
> res = cbind(frd_road$coef, frd_road$se)
> round(res, 3)
    Coeff Std. Err.
Conventional -0.253 0.301
Bias-Corrected -0.283 0.301
Robust -0.283 0.359
```

![image_28](images/image_28.png)

## 24.3.2 Re-analyzing Li et al. (2015)’s data

Recall that the running variable is 15,000 minus the standardized income in Example 24.2. In the analysis, I restrict the data to a subset with this running between [−5, 000, 5, 000], and then divide the running variable by 5, 000 so that the running variable is bounded between [−1, 1] at cutoff point zero.

The results based on the package rdrobust suggest that the university grant did not affect the dropout rate significantly.

```diff
> italy = read.csv("italy.csv")
> library("rdrobust")
> frd_italy = with(italy,
+    {
+    rdrobust(y = outcome,
+    x = rv0,
+    c = 0,
+    fuzzy = D)
```

```txt
+ })  
> res = cbind(frd_italy$coef, frd_italy$se)  
> round(res, 3)  
Coeff Std. Err.  
Conventional -0.149 0.101  
Bias-Corrected -0.155 0.101  
Robust -0.155 0.121
```

## 24.4 Discussion

Both Chapter 20 and this chapter formulate regression discontinuity based on the continuity of the conditional expectations of the potential outcomes given the running variables. This perspective is mathematically simpler but it only identifies the local effects precisely at the cutoff point of the running variable. Hahn et al. (2001) started this line of literature.

An alternative, not so dominant perspective is based on local randomization (Cattaneo et al., 2015; Li et al., 2015). If we view the running variable as a noisy measure of some underlying truth and the cutoff point is somewhat arbitrarily chosen, the units near the cutoff point do not differ systematically. This suggests that in a small neighborhood of the cutoff point, the units receive the treatment and the control in a random fashion just as in a randomized experiment. Similar to the issue of choosing h in the first perspective, it is crucial to decide how local should the randomized experiment be under the regression discontinuity. It is not easy to quantify the intuition mathematically, and again conducting sensitivity analysis with a range of h seems a reasonable approach in the second perspective as well.

See Sekhon and Titiunik (2017) for more conceptual discussion of regression discontinuity.

## 24.5 Homework Problems

## 24.1 Proof of Theorem 24.1

Prove Theorem 24.1.

## 24.2 Data analysis

Section 24.3.1 estimated the effect on occupationindexandrsn. Four other outcome variables are transportindexandrsn, firmsindexandrsn,

## 30024 Application of the Instrumental Variable Method: Fuzzy Regression Discontinuity

consumptionindexandrsn, and agricultureindexandrsn, with meanings defined in the original paper. Estimate the effects on these outcomes.

## 24.3 Reflection on the analysis of Li et al. (2015)’s data

In Li et al. (2015), a key variable determining the treatment status is the binary application status A, which has potential outcomes $A ( 1 )$ and $A ( 0 )$ corresponding to the treatment $Z = 1$ and control $Z = 0$ . By definition,

$$
D (1) = A (1), \quad D (0) = 0,
$$

so the compliers $\{ D ( 1 ) , D ( 0 ) \} = ( 1 , 0 )$ is equivalent to $A ( 1 ) = 1 . \mathrm { \ S o }$

$$
\tau_ {c} (x _ {0}) = E \{Y (1) - Y (0) \mid A (1) = 1, X = x _ {0} \}.
$$

Section 24.3.2 used the whole data set to estimate $\tau _ { \mathrm { c } } ( x _ { 0 } )$ .

An alternative analysis is based on units with $A = 1$ only. Then the treatment status is determined by X. However, this analysis can be problematic because

$$
\lim _ {\varepsilon \rightarrow 0 +} E \{Y \mid A = 1, X = x _ {0} + \varepsilon \} - \lim _ {\varepsilon \rightarrow 0 +} E \{Y \mid A = 1, X = x _ {0} - \varepsilon \}
$$

$$
= E \{Y (1) \mid A (1) = 1, X = x _ {0} \} - E \{Y (0) \mid A (0) = 1, X = x _ {0} \}. \tag {24.1}
$$

Prove (24.1) and explain why this analysis can be problematic.

Remark: The left-hand side of (24.1) is the identification formula of the local average treatment effect at $X = x _ { 0 }$ , conditioning on $A = 1$ . The right-hand side of (24.1) is the difference in means of the potential outcomes for subgroup of units with $( A ( 1 ) = 1 , X = x _ { 0 } )$ and $( A ( 0 ) = 1 , X = x _ { 0 } )$ , respectively.

## 24.4 Recommended reading

Imbens and Lemieux (2008) gave a practical guidance to regression discontinuity based on the potential outcomes framework. Lee and Lemieux (2010) reviewed regression discontinuity and its applications in economics.

## 25
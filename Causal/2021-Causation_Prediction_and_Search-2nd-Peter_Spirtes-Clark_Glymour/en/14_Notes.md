# Notes

## Chapter 2

- 1. It is customary to represent the ordered pair A, B with angle brackets as $\triangleleft , B >$ , but for endpoints of an edge we use square brackets so that the angle brackets will not be misread as arrowheads.
- 2. Some writers, especially in statistics, understand “clique” as we have defined maximal clique.
- 3. We do not include trivial independence relations, for example, $\mathrm { ~ \textsf ~ { ~ C ~ } ~ } \emptyset \mid \emptyset$ which are true by definition.

## Chapter 3

- 1. Strictly, we require for causal sufficiency of V for a population that if X is not in V and is a common cause of two or more variables in V, that the joint probability of all variables in V be the same on each value of X that occurs in the population.
- 2. Using the notion of identifiability, Simon (1953) proposed a general means to derive causal structure from a set of equations describing a system; later in the same paper Simon also proposed an account of causation using invariances under perturbations of linear coefficients.
- 3. Since causation for variables is assumed to be transitive and irreflexive, the directed graph representing a causal structure must be acyclic. Introducing cyclic directed graphs requires a systematic reinterpretation.
- 4. A better practical arrangement might be a query system that, besides inferring the causal graph or graphs, responds to the $\mathbf { \ u s e r } ^ { \prime } \mathbf { s }$ questions about the effects of the manipulation of variables.
- 5. P. 319. Q is Yule’s $Q = ( a d - b c ) / ( a d + b c )$ when the first row is $^ { a , b }$ and the second $^ { c , d }$ in a $2 \times 2$ table.
- 6. (Sic.) Kendall means, of course, that the symbols denote the respective treatment and recovery states, not vice-versa.
- 7. Fienberg (1977), citing Darroch, attributes the issue to Yule “since Yule discussed it in the final section of his 1903 paper on the theory of association of attributes” (p. 51). But save for the first sentence of that section, Yule actually discusses the reverse issue of mixtures, namely circumstances in which variables are statistically dependent in a population but independent in sub-populations.
- 8. The subsequent literature has confused it with a number of other questions about how independence and dependence relations in a population may be related to independence and dependence relations in sub-populations, and the causal significance of such facts. The unfortunate aspect of collapsing these questions is that they have distinct answers. A circumstance attributed to Simpson and now often called “Simpson’s paradox,” but nonetheless distinct from the question Simpson actually posed, was described by Colin Blyth (1972):

It is possible to have simultaneously

$( 1 ) P ( A | B ) < P ( A | B ^ { \prime } )$

and

- (2) P(A|BC) ≥ P(A|B'C)
- (3) P(A|BC') ≥ P(A|B'C')

In fact, Simpson has equality in (1) and > in (2) and (3).

- 9. The point is implicit in Blalock 1961 and no doubt other sources as well.
- 10. Treks were defined in section 2.3.1. A trek between X an Y is either i) a directed path from X to Y, ii) a directed path from Y to X, or iii) a pair of directed paths from Z to X and Z to Y that have only Z in common.
- 11. We thank Marek Druzdel for suggesting this example, and pointing out the problem of reversible mechanisms to us.

## Chapter 5

- 1. In particular, when the method is idealized to give up the greedy algorithm. Because of the greedy algorithm, we would expect the specific search procedure to be asymptotically unreliable when there are two or more treks between a pair of nonadjacent variables, say X and Y, that result in a close statistical association between those variables. This is the circumstance in the case of the one edge the procedure erroneously introduces in the ALARM network. In practice, such structures may be sufficiently uncommon for the error to be tolerable and Cooper and his colleagues are investigating techniques to ameliorate the problem.
- 2. Indeed, any statistical constraint can be used as input for the algorithms for any pairing of distributions with graphs such that the constraint is satisfied in the distribution if and only if the corresponding d-separation relation holds in the graph.
- 3. In the following heuristics, “high probabilistic dependence” means high partial correlation in the linear case, and high $\mathrm { G } ^ { 2 }$ statistic in the discrete case.
- 4. For causally sufficient structures, if a distribution P, obtained by imposing a linear distribution compatible with a graph G, implies some vanishing partial correlation not linearly implied by G, is then P not faithful to G ? If P is not faithful to G, does P necessarily imply some vanishing partial correlation not linearly implied by G? We don’t know the answer to either question.
- 5. An exact general rule for calculating the reduction of degrees of freedom given cells with zero entries seems not to be known. See Bishop, Fienberg, and Holland 1975.
- 6. It is not clear from the article how the correlations of the latent variables, GPQ and ABILITY, with other variables such as publishing productivity and QFJ were obtained. They can be obtained, for example, by using the factor structure as a regression model to

calculate estimated factor scores for each subject, or by including the covariances of the latents among the free parameters in a set of structural equations and letting a program such as LISREL estimate their values. In general the results of these procedures will be different.

- 7. The small differences are presumably attributable to round-off errors.
- 8. We do not know whether this method of graph generation produces “realistic” graphs. One feature of some of the graphs generated in this fashion that may not be desirable is the existence of isolated variables. An informal examination showed topologies not unlike the Alarm network.

## Chapter 6

- 1. We thank Thomas Verma (personal communication) for pointing out an error in the original formulation of the CI algorithm.
- 2. $\mathrm { N . B . } \ ^ {  } P _ { 1 } , ^ {  } \ ^ {  } M , ^ {  } \ ^ {  } R , ^ {  }$ in this line do not refer to vertices on the definite discriminating path U.

## Chapter 7

1. This section is based on Spirtes, Glymour, Scheines, Meek, Fienberg, and Slate 1992.

## Chapter 8

1. In linear regression, we understand the “direct influence” of $X _ { i }$ on Y to mean (i) the change in value of a variable Y that would be produced in each member of a population by a unit change in $X _ { i } ,$ , with all other X variables forced to be unchanged. Other meanings might be given, for example: (ii) the population average change in Y for unit change in $X _ { i } ,$ with all other X variables forced to be unchanged; (iii) the change in Y in each member of the population for unit change in $X _ { i } ;$ (iv) the population average change in Y for unit change in $X _ { i } ;$ etc. Under interpretations (iii) and (iv) the regression coefficient is an unreliable estimate whenever $X _ { i }$ also influences other regressors that influence Y. Interpretation (ii) is equivalent to (i) if the units are homogeneous and the stochastic properties are due to sampling; otherwise, regression will be unreliable under interpretation (i) except in special cases, for example, when the linear coefficients, as random variables, are independently distributed (in which case the analysis given here still applies [Glymour, Spirtes, and Scheines 1991a]).

- 2. In fact, we were inadvertently misinformed that all seven tests are components of AFQT and we first discovered otherwise with the SGS algorithm.
- 3. The correlation matrix given in Rawlings 1988 incorrectly gives the correlation between CU and NH4 as 0.93.
- 4. The “maximum R-square” and “stepwise” options in PROC REG in the SAS program.
- 5. Although the definition of the population in this case is unclear, and must in any case be drawn quite narrowly.

- 6. More exactly, at .05, with the exception of M G the partial correlation of every regressor with BIO vanishes when some set containing PH is controlled for; the correlation of MG with BIO vanishes when CA is controlled for.
- 7. Searches at lower significance levels remove the adjacency between FI and EN.

## Chapter 9

- 1. Personal communication.
- 2. We thank Jay Kadane for pointing out that the causal relationship between Preference and other variables might be different in the experimental and non-experimental populations, even if Preference is not directly manipulated.

## Chapter 11

- 1. This chapter is an abbreviated version of Spirtes, Scheines, and Glymour 1990, and is reprinted with the permission of Sage Publications.
- 2. The original TETRAD program (Glymour, Scheines, Spirtes, and Kelly 1987) had no such scoring function. It was left to the user to balance the Explanatory and Falsification principles.
- 3. We have also implemented heuristic search procedures that are theoretically less reliable than that described here but are much faster and in practice about equally reliable.
- 4. LISREL VII retains the same architecture but with an altered modification index.
- 5. LISREL VI outputs a number of other measures that could be used to suggest modifications to a starting model, but these are not used in the automatic search. See Costner and Herting 1985.
- 6. As long as they are not in the list of parameters not to be freed.
- 7. Since the Lagrange Multiplier statistic, like the modification indices of LISREL VI, estimates the effect on the $\textstyle \chi ^ { 2 }$ of freeing a parameter, in subsequent sections we will use the term “modification index” to refer to either of these statistics.
- 8. EQS allows the user to specify several different types of searches. We have only described the one used in our Monte Carlo simulation tests.
- 9. We are indebted to Peter Bentler for suggesting this transformation.
- 10. We did not provide LISREL or EQS with the values of the parameters in the original models that generated our covariance matrices because the input to LISREL and EQS was a pseudocorrelation matrix, not the original covariance matrix. We therefore provided the programs with the population parameters of transformed models that would generate the pseudo correlation matrices. The detailed transformations are given in Spirtes (1990).
- 11. For LISREL IV, the details of this procedure are described in Glymour et al. 1987. The same procedure works for LISREL VI with the exception of the Beta matrix. See Joreskog and Sorbom (1984).

- 12. To simplify the calculations, we assumed that the length of the lists output by TETRAD II for all of the covariance matrices generated by a single model was in each case equal to the average length of the lists. This is a fairly good approximation in most cases.
- 13. The expression $^ { 6 6 } X \mathrm { ~ C ~ } Y ^ { 3 }$ means that the error terms for X and Y are correlated, or, equivalently, that there is an additional, common cause of X and Y.
- 14. TETRAD II will, on request, automatically generate EQS input files for all models that it suggests.

## Chapter 12

- 1. Lauritizen’s proposal was given at a lecture at the Santa Fe Institute in 1997. At this writing, Lauritzen and Richardson are working on the details of the required parameterization. We thank Thomas Richardson for very helpful discussions.
- 2. We wish to thank Larry Wasserman, Teddy Seidenfeld, and Jamie Robins for many valuable conversations on the issue of consistency, although this does not imply that they endorse any of our conclusions.
- 3. We are grateful to David Heckerman, Greg Cooper, and Christopher Meek for permission to use their article.
- 4. In sections 12.5.1 through 12.5.6, “we” refers to Heckerman, Meek, and Cooper.
- 5. Bernardo and Smith (1994) provide a summary of likelihoods from the exponential family and their conjugate priors.
- 6. Discussions of equivalent sample size can be found in Winkler 1967 and Heckerman et al. 1995.
- 7. The algorithm assumes that there are no hidden variables. See section 12.5.5 for a discussion of hidden-variable models and methods for learning them. A modification of the PC algorithm has been implemented in Pronel, which can be used in conjunction with Hugin, a package for updating Bayesian networks and helping users construct Bayesian networks. BIFROST (Hojsgaard and Thieson 1995) constructs block recursive models which can also be used in conjunction with Hugin. See http://www.hugin.dk.
- 8. One of the technical assumptions used to derive this approximation is that the prior is bounded and bounded away from zero around. $\hat { \theta } _ { m }$ .
- 9. The MAP configuration ${ \overline { { \theta } } } _ { m }$ depends on the coordinate system in which the parameter variables are expressed. The MAP given here corresponds to the canonical coordinate system for the multinomial distribution (see, for example, Bernardo and Smith 1994, pp.system for the multinomial distribution (see, for example, Bernardo and Smith 1994, pp. 199–202.)199–202.)
- 10. In particular, Heckerman (1995) showed that strong likelihood equivalence is not10. In particular, Heckerman (1995) showed that strong likelihood equivalence is not consistent with parameter independence and parameter modularity.consistent with parameter independence and parameter modularity.
- 11. We say as “typically employed” because the FCI and PC algorithms take a11. We say as “typically employed” because the FCI and PC algorithms take a significance level as a parameter. We will assume that for samples of size between 100 and 10000 the significance levels are in the range 0.001 to 0.1.

- 12. We wish to thank Larry Wasserman for valuable discussions regarding the priors and the resulting posteriors, although the conclusions about the plausibility of various priors are our own.
- 13. The values were calculated by numerical integration in Mathematica. Although on14. For the purposes of comparing the prior suggested by Robins and Wasserman with13. several of the points Mathematica issued warnings, in no case did the results of severalsome common alternatives, we have changed some minor details about how DAGs are different methods of numerical integration differ by mcounted; the results are essentially the same, however.
- 14. For the purposes of comparing the prior suggested by Robi15. Needleman’s regression had 6 independent variables and an 14. $\mathsf { R } ^ { 2 }$ and Wasserman with of .271. Ours has 3 some common alternatives, independent variables with an $\mathtt { R } ^ { 2 }$ have ch of .243.
- counted; the results are essen16. Personal communication.15.
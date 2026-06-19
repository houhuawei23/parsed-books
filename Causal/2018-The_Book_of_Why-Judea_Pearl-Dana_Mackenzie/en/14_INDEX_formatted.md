# INDEX

## 1 一些约定与预备知识

### 1.1 一些记号

> **注**：$a \mid b$ 表示 $a$ 整除 $b$，$a \nmid b$ 表示 $a$ 不整除 $b$，$a \parallel b$ 表示 $a$ 恰好整除 $b$（即 $a \mid b$ 且 $\gcd(a,b/a)=1$）。$a \equiv b \pmod m$ 表示 $a$ 与 $b$ 模 $m$ 同余。

我们记 $\mathbb{N}$ 为自然数集，$\mathbb{Z}$ 为整数集，$\mathbb{Q}$ 为有理数集，$\mathbb{R}$ 为实数集，$\mathbb{C}$ 为复数集。$\mathbb{F}_q$ 表示 $q$ 元有限域，$\mathbb{F}_q^\times$ 表示 $\mathbb{F}_q$ 的乘法群。对于整数 $n$，记 $\mathbb{Z}/n\mathbb{Z}$ 为模 $n$ 的剩余类环，$(\mathbb{Z}/n\mathbb{Z})^\times$ 为其单位群。

对于整数 $n$，$\varphi(n)$ 为 Euler 函数，即 $(\mathbb{Z}/n\mathbb{Z})^\times$ 的大小。$\mu(n)$ 为 Möbius 函数，$\Lambda(n)$ 为 von Mangoldt 函数。对于实数 $x$，$\pi(x)$ 表示不超过 $x$ 的素数个数。$\lfloor x \rfloor$ 和 $\{x\}$ 分别表示 $x$ 的整数部分和小数部分。我们用 $e(x) = e^{2\pi i x}$。

对于正整数 $n$，$\omega(n)$ 表示 $n$ 的不同的素因子个数，$\Omega(n)$ 表示 $n$ 的素因子个数（计重数）。$d(n)$ 表示 $n$ 的正因子个数，$\sigma(n)$ 表示 $n$ 的正因子之和。$\tau(n)$ 有时也用来表示 $d(n)$。

对于整数 $a,b$，记 $(a,b)$ 为 $a$ 与 $b$ 的最大公因子，$[a,b]$ 为最小公倍数。有时也记 $\gcd(a,b)$ 与 $\operatorname{lcm}(a,b)$。

$f = O(g)$ 表示存在常数 $C>0$ 使得 $\vert f(x) \vert \leq C g(x)$ 对于定义域中所有 $x$ 成立。$f = o(g)$ 表示 $\lim f(x)/g(x)=0$。$f \ll g$ 等价于 $f = O(g)$，$f \asymp g$ 表示 $f \ll g$ 且 $g \ll f$。$f \sim g$ 表示 $\lim f(x)/g(x)=1$。

### 1.2 基本初等数论

> **注**：本节内容为初等数论中的基本事实，读者可参见 [1] 或 [2]。

**定理 1.1（算术基本定理）**：每个大于 $1$ 的整数 $n$ 可以唯一地写成
$$
n = p_1^{e_1} p_2^{e_2} \cdots p_k^{e_k},
$$
其中 $p_1 < p_2 < \cdots < p_k$ 是素数，$e_i \geq 1$ 是整数。

**定理 1.2（Euclid 算法）**：对于整数 $a,b$，存在整数 $x,y$ 使得
$$
ax + by = \gcd(a,b).
$$

**定理 1.3（中国剩余定理）**：若 $m_1,m_2,\ldots,m_k$ 两两互素，则对于任意整数 $a_1,a_2,\ldots,a_k$，同余方程组
$$
x \equiv a_i \pmod{m_i} \quad (i=1,2,\ldots,k)
$$
有解，且解在模 $M = m_1 m_2 \cdots m_k$ 下唯一。

**定理 1.4（Fermat 小定理）**：若 $p$ 是素数，$a$ 是整数且 $p \nmid a$，则
$$
a^{p-1} \equiv 1 \pmod p.
$$

**定理 1.5（Euler 定理）**：若 $\gcd(a,m)=1$，则
$$
a^{\varphi(m)} \equiv 1 \pmod m.
$$

**定理 1.6（Wilson 定理）**：$p$ 是素数当且仅当
$$
(p-1)! \equiv -1 \pmod p.
$$

**定理 1.7（二次互反律）**：对于奇素数 $p,q$，有
$$
\left(\frac{p}{q}\right) \left(\frac{q}{p}\right) = (-1)^{\frac{p-1}{2} \cdot \frac{q-1}{2}},
$$
其中 $\left(\frac{\cdot}{\cdot}\right)$ 是 Legendre 符号。

> **注**：Gauss 称二次互反律为“黄金定理”，它有多达 200 多个证明。

### 1.3 一些常见的数论函数

**定义 1.1（Dirichlet 卷积）**：对于数论函数 $f,g$，定义它们的 Dirichlet 卷积为
$$
(f * g)(n) = \sum_{d \mid n} f(d) g\left(\frac{n}{d}\right).
$$

Dirichlet 卷积满足交换律、结合律，且存在单位元 $\delta(n) = [n=1]$（即 $n=1$ 时取 $1$，否则取 $0$）。

**定义 1.2（Möbius 函数）**：
$$
\mu(n) = \begin{cases}
1 & \text{若 } n=1, \\
(-1)^k & \text{若 } n \text{ 是 } k \text{ 个不同素数的乘积}, \\
0 & \text{若 } n \text{ 有平方因子}.
\end{cases}
$$

Möbius 函数的重要性质是
$$
\sum_{d \mid n} \mu(d) = [n=1].
$$

**定义 1.3（Euler 函数）**：$\varphi(n) = \#\{1 \leq a \leq n : \gcd(a,n)=1\}$。

Euler 函数满足
$$
\varphi(n) = n \prod_{p \mid n} \left(1 - \frac{1}{p}\right).
$$

**定义 1.4（除数函数）**：
$$
d(n) = \sum_{d \mid n} 1, \quad \sigma(n) = \sum_{d \mid n} d.
$$

更一般地，对于复数 $s$，定义
$$
\sigma_s(n) = \sum_{d \mid n} d^s.
$$

**定义 1.5（von Mangoldt 函数）**：
$$
\Lambda(n) = \begin{cases}
\log p & \text{若 } n = p^k \text{ 是素数的幂}, \\
0 & \text{否则}.
\end{cases}
$$

von Mangoldt 函数与素数分布密切相关，因为
$$
\sum_{d \mid n} \Lambda(d) = \log n.
$$

**定义 1.6（Liouville 函数）**：
$$
\lambda(n) = (-1)^{\Omega(n)},
$$
其中 $\Omega(n)$ 是 $n$ 的素因子个数（计重数）。

Liouville 函数满足
$$
\sum_{d \mid n} \lambda(d) = [n \text{ 是完全平方数}].
$$

### 1.4 素数分布的基本结果

**定理 1.8（素数定理）**：
$$
\pi(x) \sim \frac{x}{\log x} \quad (x \to \infty).
$$

更精确地，有
$$
\pi(x) = \operatorname{Li}(x) + O\left(x e^{-c\sqrt{\log x}}\right),
$$
其中 $\operatorname{Li}(x) = \int_2^x \frac{dt}{\log t}$ 是对数积分，$c>0$ 是常数。

**定理 1.9（Chebyshev 估计）**：存在常数 $c_1,c_2>0$ 使得
$$
c_1 \frac{x}{\log x} \leq \pi(x) \leq c_2 \frac{x}{\log x} \quad (x \geq 2).
$$

**定理 1.10（Mertens 定理）**：
$$
\sum_{p \leq x} \frac{\log p}{p} = \log x + O(1),
$$
$$
\sum_{p \leq x} \frac{1}{p} = \log \log x + B + o(1),
$$
$$
\prod_{p \leq x} \left(1 - \frac{1}{p}\right) \sim \frac{e^{-\gamma}}{\log x},

# 索引

Abbott, Robert, 141, 143  
abduction, 278, 280  
ACE。参见 average causal effect  
acquisition, representation and, 38  
action, in counterfactuals, 278, 280  
agency, 367  
AI。参见 artificial intelligence  
Allen, Myles, 291–294  
American Cancer Society, 174, 178–179  
anthropometric statistics, 58  
Aristotle, 50, 264  
artificial intelligence (AI), ix–x, 10  
- Bayesian networks in, 18, 93–94, 108–109, 112, 132  
- message-passing network of, 110–111, 111 (fig.)  
- of robots, 291  
- Turing on, 27, 108–109  
- uncertainty in, 109  
- weak, 362  
- “why?” question in, 349  
- 参见 strong AI  
Asimov, Isaac, 370  
association, 50, 340  
- causation and, 181, 189  
- in Ladder of Causation, 28 (fig.), 29–30, 51  
- pattern of, 311  
- specificity, strength of, 181  
- 参见 correlation, genome-wide association study  
assumptions, 12–13, 12 (fig.)  
astronomy, 5  
attribution, 261, 291, 293, 393–394  
average causal effect (ACE), 296–297  

backdoor adjustment formula, 220–224  
backdoor criterion  
- and causal effects, 220, 225–226  
- confounding and, 157, 219  
- in do-calculus, 234  
- do-operator and, 157–165, 330  
backdoor path, 158–159  
background factors, 48  
Bareinboim, Elias, 239, 353, 356–358  
Baron, Reuben, 324–325, 339  
Bayes, Thomas, 95–96, 96 (fig.), 264  
- on data, 100, 102  
- on inverse probability, 97–99, 98 (fig.), 101, 104–105, 112–113  
- method of, 99–100  
- on miracles, 103  
- on probability, 97–98, 102  
- and subjectivity, 90, 104, 108  
Bayesian analysis, 194–195  
Bayesian conditioning, 194  
Bayesian networks, 50–51, 81, 92 (photo)  
- in AI, 18, 93–94, 108–109, 112, 132  
- in Bonaparte software, 95  
- causal diagrams and, 128–133  
- codewords, turbo codes in, 126, 127 (fig.) 128  
- conditional probability table in, 117, 119, 120 (table)  
- DNA tests and, 122, 123 (fig.), 124  
- inverse-probability problem in, 112–113, 119–120  
- junctions in, 113–116  
- in machine learning, 125  
- parent nodes in, 117  
- probability in, 358–359  
- probability tables in, 128–129  
- SCMs versus, 284  
Bayesian statistics, 89–91  
Bayes’s rule, 101–104, 196  
BCSC。参见 Breast Cancer Surveillance Consortium  
belief, 101–102  
belief propagation, 112–113, 128  
Berkeley admission paradox, 197–198  
Berkson, Joseph, 197–200, 197 (fig.), 198 (table)  
Bernoulli, Jacob, 5  
Berrou, Claude, 126–127  
Bickel, Peter, 310–312, 315–316  
Big Data, 3, 350–358, 354 (fig.)  
birth weight, 82–83, 82 (fig.)  
birth-weight paradox, 185–186, 185 (fig.), 189  
black box analysis, 125, 283  
Blalock, Hubert, 309, 326  
Bonaparte, 94–95, 122, 123 (fig.), 124–125  
brain  
- managing causes, effects, 2  
- representation, of information in, 39  
- 参见 human mind  
Breast Cancer Surveillance Consortium (BCSC), 105–106, 107 (fig.), 118  
Brito, Carlos, 257  
Brockman, John, 367–368  
Brown, Lisa, 216, 217 (fig.)  
Burks, Barbara, 198, 304, 311, 333  
- on nature-versus-nurture debate, 305–306, 305 (fig.), 306 (fig.)  
- path diagram of, 308–309  
- on social status, 307  
but-for causation, 261–263, 286–288  

canned procedures, 84–85  
Cartwright, Nancy, 49  
case studies。参见 examples  
case-control studies, 173  
Castle, William, 72–73  
causal analysis  
- data in, 85  
- subjectivity and, 89  
causal diagram, 7, 39–40, 39 (fig.), 41–42, 41 (fig.), 118 (fig.), 142 (fig.)  
- for “Algebra for All,” 337, 338 (fig.)  
- Bayesian network and, 128–133  
- for Berkeley admission paradox, 311–312, 312 (fig.), 314 (fig.)  
- for Berkson’s paradox, 197 (fig.)  
- for birth-weight paradox, 185, 185 (fig.)  
- for cholera, 247–248, 247 (fig.), 248 (fig.)  
- for climate change, 294, 294 (fig.)  
- confounder in, 138, 138 (fig.), 140  
- of counterfactual, 42–43, 42 (fig.)  
- direct effect in, 320–321  
- do-operator in, 148 (fig.)  
- front-door adjustment in, 225 (fig.)  
- of Galton board, 64–65, 64 (fig.)  
- of genetic model, 64–65, 64 (fig.)  
- graphical structure of, 131  
- for improperly controlled experiment, 147–148, 147 (fig.)  
- instrumental variables and, 250  
- of JTPA Study, 229–231, 230 (fig.)  
- for Lord’s paradox, 214, 215 (fig.)  
- for Mendelian randomization, 255–256, 256 (fig.)  
- for Monty Hall paradox, 193–194, 193 (fig.), 195 (fig.)  
- of napkin problem, 239–240, 240 (fig.)  
- of nature-versus-nurture debate, 305, 305 (fig.)  
- noncausal path in, 157, 160  
- for RCT, with noncompliance, 252–253, 253 (fig.)  
- RCT in, 140, 148–149, 149 (fig.)  
- of Simpson’s paradox, 206–207, 206 (fig.), 209 (fig.)  
- for smoking gene example, 341, 341 (fig.), 342 (fig.)  
- supply-side, 250–251, 251 (fig.)  
- for tourniquet example, 346, 346 (fig.)  
- of vaccination, 44–46, 45 (fig.)  
- 参见 path diagram  
causal effect  
- backdoor criterion for, 220, 225–226  
- through path coefficients, 77  
- through regression coefficients, 222–223  
causal inference  
- cause, effect in, 2–3  
- human mind and, 1–2, 43  
- mathematical language of, 3–8  
- objectivity of, 91  
- by robots, 2, 350, 361, 361 (fig.)  
- in statistics, 18  
- technology of, 1–2  
causal inference engine, 11–15, 12 (fig.), 26–27, 46  
causal knowledge, of machines, 37  
causal model, 12 (fig.), 13, 16–17, 45–46  
- Big Data and, 350–358, 354 (fig.)  
- as hypothetical experiments, 130  
- doing in, 27  
- imagining in, 27  
- mediation in, 300–301  
- seeing vs.

# 行间批注/脚注分离

doing, 27  
of Rubin, 261, 280–281  
testing of, 116  

> See also linear causal model; structural causal model

causal paradoxes, 189–190  
causal questions, language of, 5  
causal reasoning, 20–21, 43  
the Causal Revolution, ix–x, 7, 9, 11, 45, 140, 301, 350  
causal subjectivity, 90  
causal vocabulary, 5  
causality, ix  
provisional, 150  
queries of, 27, 183  
statistics and, 66, 190  

*Causality* (Pearl), ix, 24, 328, 331  
causation  
association and, 181, 189  
computers understanding, 40–41  
correlation and, 5–6, 82–84  
intuition about, 321  
necessary, 289–290  
Pearson, K., and, 71–72  
probability and, 47–51  
RCT for, 169  
repetition and, 66–67  
smoking-cancer debate in, 168  
in statistics, 18  
three levels of, 27–36  
Wright, S., on, 79–81  

> See also Ladder of Causation

cause  
defining, 47–48, 179–180  
proximate, 288–289  
sufficient and necessary, 288–291, 295  

> See also common cause principle

causes, effects and, 2–3  
in causal diagrams, 187  
probability versus, 46  

c-decomposition, 243  
Cerf, Vint, 95  
child nodes, 111–112, 129  
Chinese Room argument, 38–39  
cholera, 168  

> See also examples

climate change  
causal diagram of, 294, 294 (fig.)  
computer simulation of, 292–296  
counterfactuals and, 261–262, 295  
FAR and, 291–292  

> See also examples

Cochran, William, 180, 182  
codewords, 126, 127 (fig.), 128  
coefficients  
difference in, 327  
path, 77, 223, 251  
product of, 327  
regression, 222–223  

Cognitive Revolution, 24–25, 34–35  
coherence, 181–182  
collider bias, 185–186, 197–200  
common cause principle, 199  
compatibilists, 364  
completeness, 237, 243–244  
computer simulation, in climate science, 292–296  
computers  
causation and, 40–41  
counterfactuals and, 43  

“Computing Machinery and Intelligence” (Turing), 358  
conditional probability, 101, 103  
conditional probability table, 117 (table), 119, 120 (table)  
confounders, 137, 138 (fig.), 140  
of mediator, outcome, 315–316  
mediators and, 276  
provisional conclusions and, 143  
RCT and, 149–150  
in smoking risk, 175  
in statistics, 138–139, 141–142  

> See also deconfounders

confounding  
backdoor criterion for, 157, 219  
classical epidemiological definition of, 153–154, 159  
defining, 150–151, 156, 162  
in epidemiology, 152–154  
incomparability in, 151  
indirect, 241  
in Ladder of Causation, 140  
statistics and, 141, 151, 156  
surrogates in, 152  
third-variable definition, 151–152  

confounding bias, 137–138, 147  
Conrady, Stefan, 118–119  
consistency, 181, 281  
controlled experiment, 136–137, 147 (fig.)  

> See also experimental design; randomized controlled trial

Cornfield, Jerome, 175, 179–180, 183, 224, 341  
Cornfield’s inequality, 175  
Coronary Primary Prevention Trial, 252  
correlation, 29  
causation and, 5–6, 82–84  
Galton on, 62–63  
spurious, 69–72  

> See also association, collider bias

“Correlation and Causation” (Wright, S.), 82  
counterfactual analysis, 261–262  
counterfactuals, 9–10  
causal diagram for, 42–43, 42 (fig.)  
climate change and, 261–262, 295  
computers and, 43  
data and, 33  
do-expression of, 287–288  
exchangeability and, 154–155  
Frost and, 258 (photo)  
in human mind, 33  
Hume and, 19–20, 265–267  
indirect effects and, 322  
in inference engine, 296  
in Ladder of Causation, 266  
law and, 286–291  
Lewis on, 266–269  
mediation analysis for, 297  
and possible worlds, 266–269  
queries as, 20, 28 (fig.), 36, 260–261, 284  
reasoning, 10  
SCMs for, 276–280, 283–284  
for strong AI, 269  

Cox, David, 154, 240–241, 241 (fig.)  
Crow, James, 84–85  
culpability, 261  
curse of dimensionality, 221  
Curve of Abandoning Hope, 120–121  

d-separation, 116, 242, 283, 381  
Darwiche, Adnan, 30  
Darwin, Charles, 63, 73, 87  
data, 11, 12 (fig.), 14–16  
Bayes on, 100, 102  
in causal analysis, 85  
counterfactuals and, 33  
economists and, 86  
fusion, 355  
interpretation, 352  
in machine learning, 30–31  
methods and, 84–85  
mining, 351–352  
objectivity of, 89  
Pearson, K., on, 87–88  
reduction of, 85  
in science, 6, 84–85  

> See also Big Data

David, Richard, 187  
Dawid, Phillip, 237, 350  
de Fermat, Pierre, 4–5  
de Moivre, Abraham, 5  
death, proximate cause of, 288  
decision problem, 238–239  
decoding, 125–126, 127 (fig.), 128  
deconfounders, 139–140  
back-door paths for, 158–159  
in intervention, 220  

deconfounding games, 159–165  
deduction, induction and, 93  
deep learning, 3, 30, 359, 362  
Democritus, 34  

*The Design of Experiments* (Cox), 154  
developmental factors, of guinea pigs, 74–76, 75 (fig.)  
Dewar, James, 53  
Diaconis, Persi, 196  
difference, in coefficients, 327  
direct effect, 297, 300–301, 317–318  
in causal diagram, 320–321  
of intervention, 323–324  
in mediation formula, 333  
mediators and, 326, 332  

> See also indirect effects; natural direct effect

*The Direction of Time* (Reichenbach), 199  
discrimination, 311–312, 315–316  
DNA test, 94–95, 122, 123 (fig.), 124, 342  
do-calculus, 241–242  
backdoor criterion in, 234  
completeness of, 243–244  
decision problem in, 238–239  
elimination procedure in, 231–232  
front-door adjustment in, 235–237, 236 (fig.)  
instrumental variables in, 257  
transformations in, 233–234, 238  
transparency in, 239–240  
as universal mapping tool, 219–220  

do-expression, 8, 32, 49, 287–288  
Doll, Richard, 171–174, 172 (fig.)  
do-operator, 8–9, 49, 147–148, 148 (fig.), 151  
backdoor criterion and, 157–165, 330  
elimination procedure for, 237  
for intervention, 231  
in noncausal paths, 157  

do-probabilities, 226  
Duncan, Arne, 336  
Duncan, Otis, 285, 309, 326  

economics, path analysis in, 79, 84, 86, 236

See effects of treatment on the treated  
Euclidean geometry, 48, 101, 233  
evolution, human, 23–26  
examples  
Abraham and fifty righteous men, 263–264, 283–284  
“Algebra for All,” 301, 336–339, 338 (fig.)  
AlphaGo, 359–362  
aspirin and headache, 33, 267  
attractive men are jerks, 200  
bag on plane, 118–121, 118 (fig.)  
Bayes’s billiard ball, 98–99, 98 (fig.), 104, 108  
Berkeley admissions and discrimination, 309–316, 312 (fig.), 314 (fig.), 317–318  
Berkson’s paradox, 197–200, 197 (fig.), 198 (table)  
birth weight in guinea pigs, 82–83, 82 (fig.)  
blocked fire escape, 286–291  
chocolate and Nobel Prize winners, 69  
cholera, 245–249, 247 (fig.), 248 (fig.)  
coat color in guinea pigs, 72–76, 74 (fig.), 75 (fig.)  
coin flip experiment, 199–200  
Daisy and kittens, 319–322, 320 (fig.)  
Daniel and vegetarian diet, 134 (photo), 135–137  
education, skill and salary, 325–326  
falling piano, 288–289  
fertilizer and crop yield, 145–149  
fire, smoke, and alarm, 113–114  
firing squad, 39–43, 39 (fig.)  
flaxseed, elasticity of supply, 250–251, 251 (fig.)  
flu vaccine, 155–156, 156 (table)  
Galton board, 52 (photo), 54–55, 56–57, 57 (fig.), 63–65, 64 (fig.)  
Garden of Eden, 23–25  
HDL cholesterol and heart attack, 254–257  
ice cream and crime rates, 48  
inheritance of stature, 55–60, 59 (fig.)  
intelligence, nature versus nurture, 304–309  
job training and earnings, 228–231  
LDL cholesterol, 252–257, 254 (table)  
Let’s Fake a Deal, 192–196, 195 (fig.)  
Lord’s paradox: diet and weight gain, 215–217, 215 (fig.), 217 (fig.)  
Lord’s paradox: gender and weight gain, 212–215, 213 (fig.)  
mammogram and cancer risk, 104–108  
mammoth hunt, 25–26, 26 (fig.)  
matches or oxygen as cause of fire, 289–290  
Monty Hall paradox, 188 (photo), 189–197, 191 (table), 193 (fig.), 193 (table), 195 (fig.), 200  
mortality rate and Anglican weddings, 70  
online advertising, 354–355  
robot soccer, 365–366  
salary, education, and experience, 272–283, 273 (table), 276 (fig.)  
scurvy and Scott expedition, 298 (photo), 299–300, 302–304, 303 (fig.)  
shoe size, age, and reading ability, 114–115  
Simpson’s paradox: BBG drug, 189, 200–204, 201 (table), 206–210, 206 (fig.), 208 (table), 209 (fig.), 221  
Simpson’s paradox: exercise and cholesterol, 211–212, 212 (fig.)  
Simpson’s paradox: kidney stones, 210  
Simpson’s paradox: smoking and thyroid disease, 210  
Simpson’s reversal: batting averages, 203–204, 203 (table), 211  
skull length and breadth, 70–71, 70 (fig.)  
smoking, birth weight, and infant mortality, 183–187, 185 (fig.)  
smoking, tar, and cancer, 224–228, 297  
smoking and adult asthma, 164, 164 (fig.)  
smoking and lung cancer, 18–19, 167–179, 172 (fig.), 176 (fig.)  
smoking and miscarriages, 162–163  
smoking gene, 339–343, 341 (fig.), 342 (fig.)  
sure-thing principle, 204–206, 316  
talent, success, and beauty, 115–116  
tea and scones, 99–102, 100 (table), 104–105, 112–113  
toothpaste and dental floss, 29–30, 32, 34  
tourniquets, 343–347, 345 (table), 346 (fig.)  
tsunami at Orobiae, 262–263, 266  
turbo codes, 125–126, 127 (fig.), 128  
2003 heat wave and climate change, 292–296, 294 (fig.)  
vaccination and smallpox, 43–44, 45 (fig.)  
victim DNA identification, 94–95  
walking and death rate, 141–143, 142 (fig.)  

exchangeability, 154–156, 162, 181  
experimental design, 145–146, 146 (fig.)  
external validity, 357  

Facebook, 32, 351  
false positives, 106–107, 107 (fig.)  
false negatives, 107 (fig.)  
FAR. See fraction of attributable risk  
Faraday, Michael, 53  
Feigenbaum, Edward, 109  
feminism, 67–68  
Fieser, Louis, 182  
Fisher, R. A., 169, 224, 271–272  
- experimental design of, 145–146, 146 (fig.)  
- Neyman, J., and, 271–272  
- on RCT, 139–140, 143–144  
- on smoking gene, 174–175  
- in smoking-cancer debate, 178–179  
- on statistics, 85  
- Wright, S., and, 85  
Fisher Box, Joan, 144–145, 149  
Forbes, Andrew, 163–164, 164 (fig.)  
formulas, 334–335  
forward probability, 104, 112–113  
fraction of attributable risk (FAR), 291–292  
free will, 358–370  
Freedman, David, 227–228, 236, 285  
front-door adjustment, 225 (fig.), 235–237, 236 (fig.)  
front-door criterion, 224–231, 225 (fig.), 229 (fig.)  
Frost, Robert, 258 (photo)  

Galileo, 81, 187  
Gallagher, Robert, 128  
Galton, Francis, 3, 5, 52 (photo), 53, 78  
- anthropometric statistics of, 58  
- on correlation, 62–63  
- on eminence, 56  
- *Hereditary Genius* by, 55–56  
- *Natural Inheritance* by, 66  
- Pearson, K., and, 66–68  
- on regression to the mean, 57–58, 67  
- on regression line, 60–62, 61 (fig.), 221–222  
- “Typical Laws of Heredity” by, 54  
- *See also* examples  
games, deconfounding, 159–165  
Gauss, Carl Friedrich, 5  
Geiger, Dan, 242–243, 245, 285  
Genesis, 23–25, 263  
genetic modeling, 64–65, 64 (fig.)  
genetics. *See* DNA test; examples; Mendelian genetics  
genome-wide association study (GWAS), 339–340  
geometry, 232–233  
Glymour, Clark, 350  
Glynn, Adam, 228–230  
God, 23–24  
Goldberger, Arthur, 84–85  
graphoids, 381  
Greek logic, 232  
Greenland, Sander, 150, 154–156, 168, 237, 333–334  
- *See also* Robins, Jamie  
guilt, probability of, 288  
guinea pigs. *See* examples  
GWAS. *See* genome-wide association study  

Haavelmo, Trygve, 285  
Hagenaars, Jacques, 331  
Halley, Edmond, 5  
Halpern, Joseph, 350  
Hammel, Eugene, 309–311  
Hannart, Alexis, 294–295  
Harari, Yuval, 25, 34  
Hardy, G. H., 65  
HDL.

# See

high-density lipoprotein cholesterol

Heckman, James, 236

*Hereditary Genius* (Galton), 55–56

Hernberg, Sven, 152

high-density lipoprotein (HDL) cholesterol, 254–257

Hill, Austin Bradford, 169–170, 172–174, 172 (fig.), 181

Hill’s criteria, 181–183

Hipparchus, 232

*A History of Epidemiologic Methods and Concepts* (Morabia), 152–153

*History of the Peloponnesian War* (Thucydides), 262

Hitchcock, Christopher, 350

Holland, Paul, 236, 273, 275

Hooke’s Law, 33

Hong, Guanglei, 337–338

*How Not to Be Wrong* (Ellenberg), 200

human cognition, 99  
- communicating, with robot, 366  
- evolution, 23–26

human mind  
- causal inference of, 1–2, 43  
- counterfactuals in, 33

humanlike intelligence, 30, 269

Hume, David, 103  
- on counterfactuals, 19–20, 265–267  
- *An Enquiry Concerning Human Understanding* by, 265–266  
- “On Miracles” by, 96–97  
- *Treatise of Human Nature* by, 264–265, 265 (fig.)

Huygens, Christiaan, 4–5

hypothetical experiments, 130

ignorability, 281–282

imagination  
- in causation, 27  
- the Lion Man as, 34–35  
- in mental model, 26, 26 (fig.)

imitation game, 36–37

incomparability, 151

indirect confounding, 241

indirect effects  
- counterfactuals and, 322  
- in mediation analysis, 297, 300–301  
- as product, 328–329  

*See also* natural indirect effect

induction, deduction and, 93

inference engine, 296, 352  

*See also* causal inference engine

information  
- flow of, 157–158  
- representing, in brain, 97  
- transfer of, 194

instrumental variables, 249–250, 249 (fig.), 257

intention, 367

intervention, 9, 131, 150  
- deconfounders in, 220  
- direct effect of, 323–324  
- do-operator for, 149–150, 231  
- in Ladder of Causation, 28 (fig.), 31–33, 40, 219, 231  
- prediction and, 32  
- variables in, 257  

*See also* Mount Intervention

intuition, 47, 99, 125, 189, 321

inverse probability  
- Bayes on, 97–99, 98 (fig.), 101, 104–105  
- in Bayesian network, 112–113, 119–120  
- likelihood ratio and, 105, 113

Jeffreys, Harold, 103

Jeter, Derek, 203, 203 (table)

Job Training Partnership Act (JTPA) Study, 228–231, 229 (fig.), 230 (fig.)

Joffe, Marshall, 283

Jouffe, Lionel, 118–119

JTPA. *See* Job Training Partnership Act Study

junctions  
- in Bayesian networks, 113–116  
- in flow, of information, 157–158

Justice, David, 203, 203 (table)

Kahneman, Daniel, 58, 63–64, 290

*Karl Pearson* (Porter), 67

Karlin, Samuel, 87

Kashin, Konstantin, 228–230

Kathiresan, Sekar, 256

Ke Jie, 360

Kempthorne, Oscar, 272

Kenny, David, 324–325, 339

Klein, Ezra, 139, 154

knowledge, 8, 11–12, 12 (fig.)

Koettlitz, Reginald, 302–304

Kragh, John, 343–347

Kruskal, William, 312–316, 346

Ladder of Causation, 17–19, 24, 116  
- association in, 28 (fig.), 29–30, 51  
- bias in, 311  
- confounding in, 140  
- counterfactuals in, 266  
- intervention in, 28 (fig.), 31–33, 40, 219, 231  
- model-free approach to, 88  
- observation in, 264  
- probabilities and, 47–49, 75  
- queries in, 28 (fig.), 29, 32

language  
- of knowledge, 8  
- mathematical, 3–8  
- of probability, 102–103  
- of queries, 8, 10

Laplace, Pierre-Simon, 5

Latin square, 145, 146 (fig.)

law, counterfactuals and, 286–291

LDL. *See* low-density lipoprotein cholesterol

*Let’s Make a Deal*. *See* examples

Lewis, David, 20, 266–269

likelihood ratio, 105–106, 113

Lilienfeld, Abe, 175, 179–180

Lind, James, 168, 299, 302–303

Lindley, Dennis, 209

linear causal model, 322–323, 327

linear models, 295–296

linear regression, 285–286

linear SCMs, 285–286

the Lion Man, 34–36, 35 (fig.)

LISREL, 86

logic, 232, 238

Lord’s paradox. *See* examples

low-density lipoprotein (LDL) cholesterol, 252–257, 254 (table)

lung cancer, smoking in, 18–19, 167–168

machine learning, 10–11, 30–31, 125, 363  

*See also* artificial intelligence (AI)

machines  
- causal knowledge of, 37  
- thinking, 367–368  

*See also* robots

MacKay, David, 127–128

Malaysia Airlines crash, 122, 123 (fig.)

Marcus, Gary, 30

matching, 274

mathematical certainty, 288

mathematical language, 3–8

mathematics, science and, 4–5, 84–85  

*See also* geometry

M-bias, 161

McDonald, Rod, 325

mediation, 20  
- “Algebra for All” as, 336–339, 338 (fig.)  
- analysis, 297, 300–301, 322–323  
- in causation, 300–301  
- fallacy, 272, 315–316  
- formula, 319, 332–333, 335  
- questions, 131  
- smoking gene example as, 339–343, 341 (fig.), 342 (fig.)  
- threshold effect and, 325, 326 (fig.)

mediators, 153–154, 228, 297  
- confounders and, 276  
- direct effect and, 326, 332  
- outcomes and, 315–316

Mendel, Gregor, 65

Mendelian genetics, 73

Mendelian randomization, 255–256, 256 (fig.)

mental model, 26, 26 (fig.)

message-passing network, 110–111, 111 (fig.)

methods, data and, 84–85

mini-Turing test, 36–46

miracles, 103, 357

model discovery, 373

model-blind, 33, 66, 132, 217, 275

Model Penal Code, 286, 288

model-free approach, 87–89, 272, 351  

*See also* model-blind

Morabia, Alfredo, 152–153

Mount Intervention, 218 (photo), 219–220, 224, 259–260

Musk, Elon, 367

napkin problem, 239–240, 240 (fig.), 330

natural direct effect (NDE), 318–319, 332–333

natural effects, 327

natural indirect effect (NIE), 319, 321, 325–326, 332–333

*Natural Inheritance* (Galton), 66

nature, 144–145, 147, 149, 156, 257

nature-versus-nurture debate, 304–309, 305 (fig.), 306 (fig.)

NDE. *See* natural direct effect

necessary causation, 289–290, 295

necessity, probability of, 294

Netherlands Forensic Institute (NFI), 94, 122

See natural indirect effect.

> Niles, Henry, 78–81, 84

noncausal path, in causal diagram, 157, 160  
noncollapsibility, 152  
noncompliance, RCT with, 252–253, 253 (fig.)  
nonconfoundedness, 281  
nonlinear analysis, 335  
nonrandomized studies, 149  
Novick, Melvin, 201, 209

**objectivity**
- in Bayesian inference, 89
- of causal inference, 91

observational studies, 150–151, 229  
Ogburn, William Fielding, 309  
“On Miracles” (Hume), 96–97  
“On the Inadequacy of the Partial and Multiple Correlation Technique” (Burks), 308  
*Origin of Species* (Darwin), 63

**paradox**, 9, 19, 189–190
- birth-weight, 185–186, 185 (fig.), 189
- as optical illusion, 189–190

*See also* examples

parent nodes, 111–112, 117–118, 129  
Pascal, Blaise, 4–5  
Pasteur, Louis, 228

**path analysis**
- in economics, 86
- in social sciences, 85–86
- Wright, S., on, 86–89, 324

path coefficients, 77, 223, 251

**path diagram**
- for birth-weight example, 82–83, 82 (fig.)
- of Burks, 308–309
- of Wright, S., 74–77, 75 (fig.), 85–86, 221, 260–261

Paz, Azaria, 381  
Pearl, Judea, ix, 24, 51, 328, 331  
Pearson, Egon, 271–272  
Pearson, Karl, 5, 62, 78, 85, 180, 222
- causation and, 71–72
- on data, 87–88
- Galton and, 66–68
- on skull size, 70 (fig.)
- on spurious correlation, 69
- as zealot, 67–68

philosophers, on causation, 47–51, 81  
physics, 33–34, 67, 99  
Pigou, Arthur Cecil, 198  
Pinto, Rodrigo, 236  
placebo effect, 300  
polynomial time, 238  
Porter, Ted, 67

**potential outcomes**, 155, 260  
potential outcomes framework, 155  
prediction, 278, 280
- intervention and, 32
- in science, 36

preponderance of evidence, 288  
pretreatment variables, 160  
Price, Richard, 97  
prior knowledge, 90, 104  
probabilistic causality, 47–51  
*Probabilistic Reasoning in Intelligent Systems* (Pearl), 51

**probability**, 43–44, 46, 90, 110
- Bayes on, 97–98, 102
- Bayesian networks and, 358–359
- in but-for causation, 287
- causation and, 47–51
- of guilt, 288
- Ladder of Causation and, 47–49, 75
- language of, 102–103
- or necessity, 294
- over time, 120–121, 121 (fig.)
- raising, 49
- of sufficiency, 294

*See also* conditional probability; inverse probability

probability table, 117 (table), 128–129  
probability theory, 4–5

**product**
- of coefficients, 327
- indirect effect as, 328–329

Provine, William, 85  
provisional causality, 150  
proximate cause, 288–289  
Pythagoras, 233

**quantitative causal reasoning**, 43  
queries, 8, 10, 12 (fig.), 14–15
- causal, 27, 183
- counterfactual, 20, 28 (fig.), 36, 260–261, 284
- in Ladder of Causation, 28 (fig.), 29, 32
- mediation, 131

*See also* “Why?” question

**randomized controlled trial (RCT)**, 18, 132–133, 143–147
- in causal diagram, 140, 148–149, 149 (fig.)
- confounders and, 149–150
- in epidemiology, 172–173
- Fisher on, 139–140, 143–144
- as “gold standard,” 231
- with noncompliance, causal diagram for, 252–253, 253 (fig.)
- observational studies versus, 150, 229

recombinant DNA, 369  
reduction, of data, 85  
regression, 29, 325  
*See also* linear regression

regression coefficient, 222–223  
regression line, 60–62, 61 (fig.), 221–222  
regression to the mean, 57–58, 67  
Reichenbach, Hans, 199, 234  
Reid, Constance, 271–272

**representation**
- acquisition and, 38
- of information, in brain, 39

representation problem, 268  
reversion, 56–57  
Robins, Jamie, 168, 329–330, 329 (fig.), 333–334
- on confounding, 150
- do-calculus and, 236–237, 241
- on exchangeability, 154–156

robots, ix–x
- AI, 291
- causal inference by, 2, 350, 361, 361 (fig.)
- communicating, with humans, 366
- as moral, 370
- soccer, 365–366

root node, 117  
Rubin, Donald, 269–270, 270 (photo), 275, 283
- causal model of, 261, 280–281
- on potential outcomes, 155

Rumelhart, David, 110, 111 (fig.), 268

Sackett, David, 197–198, 198 (table)  
*Sapiens* (Harari), 25  
Savage, Jimmie, 316  
Savage, Leonard, 204–206  
scatter plot, 59 (fig.), 60, 62  
Scheines, Richard, 350  
Schuman, Leonard, 182

**science**
- data in, 6, 84–85
- history of, 4–5
- mathematics and, 4–5, 84–85
- prediction in, 36

*See also* causal inference; social sciences

scientific method, 108, 302  
SCMs. *See* structural causal models

Scott, Robert Falcon, 298 (photo), 302, 303 (fig.)  
Searle, John, 38, 363  
seatbelt usage, 161–162  
Sedol, Lee, 360  
*Seeing vs. doing*, 8–9, 27, 130, 149, 233  
self-awareness, 363, 367  
SEM. *See* structural equation model

sensitivity analysis, 176  
sequential treatment, 241 (fig.)  
Shafer, Glen, 109  
Sharpe, Maria, 68  
Sherlock Holmes, 92 (photo), 93  
Shpitser, Ilya, 24, 238–239, 243, 245, 296–297  
Silicon Valley, 32  
Simon, Herbert, 79, 198  
Simpson, Edward, 153–154, 208–209  
Simpson’s paradox. *See* examples

**smoking.** *See* examples; surgeon general’s advisory committee; tobacco industry

smoking gene, 174–175, 224–227, 339–343, 341 (fig.), 342 (fig.)  
smoking-cancer debate, 166 (photo), 167–179  
Snow, John, 168, 245–249  
social sciences, 84–86  
social status, 307  
sophomore slump, 56–58  
Spirtes, Peter, 244  
Spohn, Wolfgang, 350  
spurious correlation, 69–72  
spurious effects, 138

**stable unit treatment value assumption (SUTVA)**, 280–281  
Stanford-Binet IQ test, 305–306  
statistical estimation, 12 (fig.), 15

**statistics**, 5–6, 9
- anthropometric and, 58
- canned procedures in, 84–85
- causal inference in, 18
- causality and, 18, 66, 190
- confounders in, 138–139, 141–142
- methods of, 31, 180–181
- objectivity and
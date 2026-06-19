# 索引

## 1 一些约定与预备知识

### 1.1 一些记号

> **注**：$a \mid b$ 表示 $a$ 整除 $b$，$a \nmid b$ 表示 $a$ 不整除 $b$，$a \parallel b$ 表示 $a$ 恰好整除 $b$（即 $a \mid b$ 且 $\gcd(a,b/a)=1$）。$a \equiv b \pmod m$ 表示 $a$ 与 $b$ 模 $m$ 同余。

我们记 $\mathbb{N}$ 为**自然数集**，$\mathbb{Z}$ 为**整数集**，$\mathbb{Q}$ 为**有理数集**，$\mathbb{R}$ 为**实数集**，$\mathbb{C}$ 为**复数集**。$\mathbb{F}_q$ 表示 $q$ 元**有限域**，$\mathbb{F}_q^\times$ 表示 $\mathbb{F}_q$ 的**乘法群**。对于整数 $n$，记 $\mathbb{Z}/n\mathbb{Z}$ 为模 $n$ 的**剩余类环**，$(\mathbb{Z}/n\mathbb{Z})^\times$ 为其**单位群**。

对于整数 $n$，$\varphi(n)$ 为 **Euler 函数**，即 $(\mathbb{Z}/n\mathbb{Z})^\times$ 的大小。$\mu(n)$ 为 **Möbius 函数**，$\Lambda(n)$ 为 **von Mangoldt 函数**。对于实数 $x$，$\pi(x)$ 表示不超过 $x$ 的**素数个数**。$\lfloor x \rfloor$ 和 $\{x\}$ 分别表示 $x$ 的**整数部分**和**小数部分**。我们用 $e(x) = e^{2\pi i x}$。

对于正整数 $n$，$\omega(n)$ 表示 $n$ 的不同的素因子个数，$\Omega(n)$ 表示 $n$ 的素因子个数（计重数）。$d(n)$ 表示 $n$ 的**正因子个数**，$\sigma(n)$ 表示 $n$ 的**正因子之和**。$\tau(n)$ 有时也用来表示 $d(n)$。

对于整数 $a,b$，记 $(a,b)$ 为 $a$ 与 $b$ 的**最大公因子**，$[a,b]$ 为**最小公倍数**。有时也记 $\gcd(a,b)$ 与 $\operatorname{lcm}(a,b)$。

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
其中 $\left(\frac{\cdot}{\cdot}\right)$ 是 **Legendre 符号**。

> **注**：Gauss 称二次互反律为“黄金定理”，它有多达 200 多个证明。

### 1.3 一些常见的数论函数

**定义 1.1（Dirichlet 卷积）**：对于数论函数 $f,g$，定义它们的 **Dirichlet 卷积** 为
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
其中 $\operatorname{Li}(x) = \int_2^x \frac{dt}{\log t}$ 是**对数积分**，$c>0$ 是常数。

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
$$

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

进行中, 27  
鲁宾（Rubin）, 261, 280–281  
检验, 116  

> 另见线性因果模型（linear causal model）；结构因果模型（structural causal model）

因果悖论（causal paradoxes）, 189–190  
因果问题（causal questions）的语言, 5  
因果推理（causal reasoning）, 20–21, 43  
**因果革命（The Causal Revolution）**, ix–x, 7, 9, 11, 45, 140, 301, 350  
因果主观性（causal subjectivity）, 90  
因果词汇（causal vocabulary）, 5  
因果性（causality）, ix  
临时性（provisional）, 150  
查询, 27, 183  
统计学与, 66, 190  

*Causality*（Pearl）, ix, 24, 328, 331  
因果关系（causation）  
关联（association）与, 181, 189  
计算机理解, 40–41  
相关性（correlation）与, 5–6, 82–84  
直觉, 321  
必要性（necessary）, 289–290  
皮尔逊（Pearson, K.）与, 71–72  
概率（probability）与, 47–51  
随机对照试验（RCT）用于, 169  
重复（repetition）与, 66–67  
吸烟-癌症争论, 168  
统计学中的, 18  
三个层次, 27–36  
赖特（Wright, S.）关于, 79–81  

> 另见因果之梯（Ladder of Causation）

原因（cause）  
定义, 47–48, 179–180  
近因（proximate）, 288–289  
充分且必要（sufficient and necessary）, 288–291, 295  

> 另见共同原因原则（common cause principle）

原因、结果与, 2–3  
在因果图中, 187  
概率与, 46  

c-分解（c-decomposition）, 243  
瑟夫（Cerf, Vint）, 95  
子节点（child nodes）, 111–112, 129  
中文屋论证（Chinese Room argument）, 38–39  
霍乱（cholera）, 168  

> 另见示例（examples）

气候变化（climate change）  
因果图, 294, 294（图）  
计算机模拟, 292–296  
反事实（counterfactuals）与, 261–262, 295  
可归因风险分数（FAR）与, 291–292  

> 另见示例（examples）

科克伦（Cochran, William）, 180, 182  
码字（codewords）, 126, 127（图）, 128  
系数（coefficients）  
差异, 327  
路径（path）, 77, 223, 251  
乘积, 327  
回归（regression）, 222–223  

**认知革命（Cognitive Revolution）**, 24–25, 34–35  
一致性（coherence）, 181–182  
碰撞偏倚（collider bias）, 185–186, 197–200  
共同原因原则（common cause principle）, 199  
相容论者（compatibilists）, 364  
完备性（completeness）, 237, 243–244  
计算机模拟（气候科学）, 292–296  
计算机  
因果关系与, 40–41  
反事实与, 43  

“计算机器与智能（Computing Machinery and Intelligence）”（图灵）, 358  
条件概率（conditional probability）, 101, 103  
条件概率表（conditional probability table）, 117（表）, 119, 120（表）  
混杂因子（confounders）, 137, 138（图）, 140  
中介变量（mediator）、结局（outcome）的, 315–316  
中介变量与, 276  
临时结论与, 143  
随机对照试验（RCT）与, 149–150  
吸烟风险中的, 175  
统计学中的, 138–139, 141–142  

> 另见解混杂因子（deconfounders）

混杂（confounding）  
后门准则（backdoor criterion）用于, 157, 219  
经典流行病学定义, 153–154, 159  
定义, 150–151, 156, 162  
流行病学中的, 152–154  
不可比性（incomparability）, 151  
间接（indirect）, 241  
因果之梯中的, 140  
统计学与, 141, 151, 156  
替代变量（surrogates）, 152  
第三方变量定义, 151–152  

混杂偏倚（confounding bias）, 137–138, 147  
康拉迪（Conrady, Stefan）, 118–119  
一致性（consistency）, 181, 281  
受控实验（controlled experiment）, 136–137, 147（图）  

> 另见实验设计（experimental design）；随机对照试验（randomized controlled trial）

康菲尔德（Cornfield, Jerome）, 175, 179–180, 183, 224, 341  
康菲尔德不等式（Cornfield’s inequality）, 175  
冠心病一级预防试验（Coronary Primary Prevention Trial）, 252  
相关性（correlation）, 29  
因果关系与, 5–6, 82–84  
高尔顿（Galton）关于, 62–63  
虚假（spurious）, 69–72  

> 另见关联（association）；碰撞偏倚（collider bias）

“相关性与因果关系（Correlation and Causation）”（赖特）, 82  
反事实分析（counterfactual analysis）, 261–262  
反事实（counterfactuals）, 9–10  
因果图, 42–43, 42（图）  
气候变化与, 261–262, 295  
计算机与, 43  
数据与, 33  
do表达式（do-expression）, 287–288  
可交换性（exchangeability）与, 154–155  
弗罗斯特（Frost）与, 258（照片）  
人类思维中的, 33  
休谟（Hume）与, 19–20, 265–267  
间接效应与, 322  
推理引擎中的, 296  
因果之梯中的, 266  
法律与, 286–291  
刘易斯（Lewis）关于, 266–269  
中介分析（mediation analysis）用于, 297  
可能世界（possible worlds）与, 266–269  
查询, 20, 28（图）, 36, 260–261, 284  
推理, 10  
结构因果模型（SCMs）用于, 276–280, 283–284  
用于强人工智能（strong AI）, 269  

考克斯（Cox, David）, 154, 240–241, 241（图）  
克劳（Crow, James）, 84–85  
罪责（culpability）, 261  
维度灾难（curse of dimensionality）, 221  
放弃希望曲线（Curve of Abandoning Hope）, 120–121  

d-分离（d-separation）, 116, 242, 283, 381  
达尔维奇（Darwiche, Adnan）, 30  
达尔文（Darwin, Charles）, 63, 73, 87  
数据（data）, 11, 12（图）, 14–16  
贝叶斯（Bayes）关于, 100, 102  
因果分析中的, 85  
反事实与, 33  
经济学家与, 86  
融合（fusion）, 355  
解释（interpretation）, 352  
机器学习中的, 30–31  
方法与, 84–85  
挖掘（mining）, 351–352  
客观性, 89  
皮尔逊（Pearson, K.）关于, 87–88  
约简（reduction）, 85  
科学中的, 6, 84–85  

> 另见大数据（Big Data）

大卫（David, Richard）, 187  
道伊德（Dawid, Phillip）, 237, 350  
费马（de Fermat, Pierre）, 4–5  
棣莫弗（de Moivre, Abraham）, 5  
死亡（近因）, 288  
决策问题（decision problem）, 238–239  
解码（decoding）, 125–126, 127（图）, 128  
解混杂因子（deconfounders）, 139–140  
后门路径, 158–159  
干预中的, 220  

解混杂博弈（deconfounding games）, 159–165  
演绎（deduction）与归纳（induction）, 93  
深度学习（deep learning）, 3, 30, 359, 362  
德谟克利特（Democritus）, 34  

*实验设计（The Design of Experiments）*（考克斯）, 154  
豚鼠的发育因素, 74–76, 75（图）  
杜瓦（Dewar, James）, 53  
迪亚科尼斯（Diaconis, Persi）, 196  
系数差异, 327  
直接效应（direct effect）, 297, 300–301, 317–318  
因果图中的, 320–321  
干预的, 323–324  
中介公式（mediation formula）中的, 333  
中介变量与, 326, 332  

> 另见间接效应（indirect effects）；自然直接效应（natural direct effect）

*时间的方向（The Direction of Time）*（赖兴巴赫）, 199  
歧视（discrimination）, 311–312, 315–316  
DNA检测（DNA test）, 94–95, 122, 123（图）, 124, 342  
do-演算（do-calculus）, 241–242  
后门准则, 234  
完备性, 243–244  
决策问题, 238–239  
消除程序（elimination procedure）, 231–232  
前门调整（front-door adjustment）, 235–237, 236（图）  
工具变量（instrumental variables）, 257  
变换（transformations）, 233–234, 238  
透明性（transparency）, 239–240  
作为通用映射工具, 219–220  

do-表达式（do-expression）, 8, 32, 49, 287–288  
多尔（Doll, Richard）, 171–174, 172（图）  
do-算子（do-operator）, 8–9, 49, 147–148, 148（图）, 151  
后门准则与, 157–165, 330  
消除程序, 237  
用于干预, 231  
非因果路径中的, 157  

do-概率（do-probabilities）, 226  
邓肯（Duncan, Arne）, 336  
邓肯（Duncan, Otis）, 285, 309, 326  

经济学中的路径分析, 79, 84, 86, 236  

另见处理效应（effects of treatment on the treated）  
欧几里得几何（Euclidean geometry）, 48, 101, 233  
人类进化, 23–26  

示例（examples）  
亚伯拉罕与五十个义人, 263–264, 283–284  
“全民代数（Algebra for All）”, 301, 336–339, 338（图）  
AlphaGo, 359–362  
阿司匹林与头痛, 33, 267  
帅哥是混蛋, 200  
飞机上的行李, 118–121, 118（图）  
贝叶斯台球, 98–99, 98（图）, 104, 108  
伯克利招生与歧视, 309–316, 312（图）, 314（图）, 317–318  
伯克森悖论（Berkson’s paradox）, 197–200, 197（图）, 198（表）  
豚鼠出生体重, 82–83, 82（图）  
堵塞的消防通道, 286–291  
巧克力与诺贝尔奖得主, 69  
霍乱, 245–249, 247（图）, 248（图）  
豚鼠毛色, 72–76, 74（图）, 75（图）  
抛硬币实验, 199–200  
黛西与小猫咪, 319–322, 320（图）  
丹尼尔与素食, 134（照片）, 135–137  
教育、技能与薪资, 325–326  
坠落的钢琴, 288–289  
肥料与作物产量, 145–149  
火、烟雾与警报, 113–114  
行刑队, 39–43, 39（图）  
亚麻籽的供给弹性, 250–251, 251（图）  
流感疫苗, 155–156, 156（表）  
高尔顿板（Galton board）, 52（照片）, 54–55, 56–57, 57（图）, 63–65, 64（图）  
伊甸园（Garden of Eden）, 23–25  
HDL胆固醇与心脏病发作, 254–257  
冰淇淋与犯罪率, 48  
身高遗传, 55–60, 59（图）  
智力：先天与后天, 304–309  
职业培训与收入, 228–231  
LDL胆固醇, 252–257, 254（表）  
我们来换一扇门（Let’s Fake a Deal）, 192–196, 195（图）  
洛德悖论（Lord’s paradox）：饮食与体重增加, 215–217, 215（图）, 217（图）  
洛德悖论：性别与体重增加, 212–215, 213（图）  
乳房X光检查与癌症风险, 104–108  
猛犸象狩猎, 25–26, 26（图）  
火柴或氧气作为火灾原因, 289–290  
蒙提霍尔悖论（Monty Hall paradox）, 188（照片）, 189–197, 191（表）, 193（图）, 193（表）, 195（图）, 200  
死亡率与圣公会婚礼, 70  
在线广告, 354–355  
机器人足球, 365–366  
薪资、教育与经验, 272–283, 273（表）, 276（图）  
坏血病与斯科特探险队, 298（照片）, 299–300, 302–304, 303（图）  
鞋码、年龄与阅读能力, 114–115  
辛普森悖论（Simpson’s paradox）：BBG药物, 189, 200–204, 201（表）, 206–210, 206（图）, 208（表）, 209（图）, 221  
辛普森悖论：运动与胆固醇, 211–212, 212（图）  
辛普森悖论：肾结石, 210  
辛普森悖论：吸烟与甲状腺疾病, 210  
辛普森逆转（Simpson’s reversal）：击球率, 203–204, 203（表）, 211  
颅骨长度与宽度, 70–71, 70（图）  
吸烟、出生体重与婴儿死亡率, 183–187, 185（图）  
吸烟、焦油与癌症, 224–228, 297  
吸烟与成人哮喘, 164, 164（图）  
吸烟与肺癌, 18–19, 167–179, 172（图）, 176（图）  
吸烟与流产, 162–163  
吸烟基因, 339–343, 341（图）, 342（图）  
确定事件原则（sure-thing principle）, 204–206, 316  
才华、成功与美貌, 115–116  
茶与烤饼, 99–102, 100（表）, 104–105, 112–113  
牙膏与牙线, 29–30, 32, 34  
止血带, 343–347, 345（表）, 346（图）  
奥罗比亚海啸, 262–263, 266  
涡轮码（turbo codes）, 125–126, 127（图）, 128  
2003年热浪与气候变化, 292–296, 294（图）  
疫苗接种与天花, 43–44, 45（图）  
受害者DNA识别, 94–95  
步行与死亡率, 141–143, 142（图）  

可交换性（exchangeability）, 154–156, 162, 181  
实验设计（experimental design）, 145–146, 146（图）  
外部效度（external validity）, 357  

Facebook, 32, 351  
假阳性（false positives）, 106–107, 107（图）  
假阴性（false negatives）, 107（图）  
FAR。见可归因风险分数（fraction of attributable risk）  
法拉第（Faraday, Michael）, 53  
费根鲍姆（Feigenbaum, Edward）, 109  
女性主义（feminism）, 67–68  
菲泽（Fieser, Louis）, 182  
费希尔（Fisher, R. A.）, 169, 224, 271–272  
- 实验设计, 145–146, 146（图）  
- 奈曼（Neyman, J.）与, 271–272  
- 关于随机对照试验（RCT）, 139–140, 143–144  
- 关于吸烟基因, 174–175  
- 吸烟-癌症争论中的, 178–179  
- 关于统计学, 85  
- 赖特（Wright, S.）与, 85  
费希尔·博克斯（Fisher Box, Joan）, 144–145, 149  
福布斯（Forbes, Andrew）, 163–164, 164（图）  
公式（formulas）, 334–335  
前向概率（forward probability）, 104, 112–113  
可归因风险分数（fraction of attributable risk, FAR）, 291–292  
自由意志（free will）, 358–370  
弗里德曼（Freedman, David）, 227–228, 236, 285  
前门调整（front-door adjustment）, 225（图）, 235–237, 236（图）  
前门准则（front-door criterion）, 224–231, 225（图）, 229（图）  
弗罗斯特（Frost, Robert）, 258（照片）  

伽利略（Galileo）, 81, 187  
加拉格尔（Gallagher, Robert）, 128  
高尔顿（Galton, Francis）, 3, 5, 52（照片）, 53, 78  
- 人体测量统计, 58  
- 关于相关性, 62–63  
- 关于卓越, 56  
- *遗传的天才（Hereditary Genius）*, 55–56  
- *自然遗传（Natural Inheritance）*, 66  
- 皮尔逊（Pearson, K.）与, 66–68  
- 关于均值回归（regression to the mean）, 57–58, 67  
- 关于回归线（regression line）, 60–62, 61（图）, 221–222  
- “典型遗传定律（Typical Laws of Heredity）”, 54  
- *另见*示例（examples）  
解混杂博弈（games, deconfounding）, 159–165  
高斯（Gauss, Carl Friedrich）, 5  
盖革（Geiger, Dan）, 242–243, 245, 285  
创世记（Genesis）, 23–25, 263  
遗传建模（genetic modeling）, 64–65, 64（图）  
遗传学。见DNA检测（DNA test）；示例（examples）；孟德尔遗传学（Mendelian genetics）  
全基因组关联研究（genome-wide association study, GWAS）, 339–340  
几何学（geometry）, 232–233  
格莱莫（Glymour, Clark）, 350  
格林（Glynn, Adam）, 228–230  
上帝（God）, 23–24  
戈德伯格（Goldberger, Arthur）, 84–85  
图状结构（graphoids）, 381  
希腊逻辑（Greek logic）, 232  
格陵兰（Greenland, Sander）, 150, 154–156, 168, 237, 333–334  
- *另见*罗宾斯（Robins, Jamie）  
罪责概率, 288  
豚鼠。见示例（examples）  
GWAS。见全基因组关联研究（genome-wide association study）  

哈维尔莫（Haavelmo, Trygve）, 285  
哈赫纳尔斯（Hagenaars, Jacques）, 331  
哈雷（Halley, Edmond）, 5  
哈尔彭（Halpern, Joseph）, 350  
哈梅尔（Hammel, Eugene）, 309–311  
汉纳特（Hannart, Alexis）, 294–295  
赫拉利（Harari, Yuval）, 25, 34  
哈代（Hardy, G. H.）, 65  
HDL。

# 参见

高密度脂蛋白胆固醇

赫克曼，詹姆斯，236

*遗传的天才*（高尔顿），55–56

赫恩伯格，斯文，152

高密度脂蛋白（HDL）胆固醇，254–257

希尔，奥斯汀·布拉德福德，169–170，172–174，172（图），181

希尔准则，181–183

喜帕恰斯，232

*流行病学方法与概念史*（莫拉比亚），152–153

*伯罗奔尼撒战争史*（修昔底德），262

希区柯克，克里斯托弗，350

霍兰德，保罗，236，273，275

胡克定律，33

洪，光磊，337–338

*How Not to Be Wrong*（埃伦伯格），200

人类认知，99  
- 与机器人交流，366  
- 进化，23–26

人类心智  
- 因果推断，1–2，43  
- 反事实，33

类人智能，30，269

休谟，大卫，103  
- 论反事实，19–20，265–267  
- *人类理解研究*，265–266  
- “论奇迹”，96–97  
- *人性论*，264–265，265（图）

惠更斯，克里斯蒂安，4–5

假设实验，130

可忽略性，281–282

想象力  
- 因果关系中的，27  
- 作为狮子人，34–35  
- 心智模型中的，26，26（图）

模仿游戏，36–37

不可比性，151

间接混杂，241

间接效应  
- 反事实与，322  
- 中介分析中的，297，300–301  
- 作为乘积，328–329  

*另见* 自然间接效应

归纳，演绎与，93

推理引擎，296，352  

*另见* 因果推理引擎

信息  
- 流动，157–158  
- 在大脑中的表征，97  
- 传递，194

工具变量，249–250，249（图），257

意图，367

干预，9，131，150  
- 去混杂因子，220  
- 直接效应，323–324  
- 干预算子，149–150，231  
- 因果阶梯中的，28（图），31–33，40，219，231  
- 预测与，32  
- 变量，257  

*另见* 干预之山

直觉，47，99，125，189，321

逆概率  
- 贝叶斯论，97–99，98（图），101，104–105  
- 贝叶斯网络中的，112–113，119–120  
- 似然比与，105，113

杰弗里斯，哈罗德，103

杰特，德里克，203，203（表）

**职业培训伙伴法案（JTPA）研究**，228–231，229（图），230（图）

乔菲，马歇尔，283

茹弗，利昂内尔，118–119

JTPA。*参见* 职业培训伙伴法案研究

节点  
- 贝叶斯网络中的，113–116  
- 信息流动中的，157–158

贾斯蒂斯，大卫，203，203（表）

卡尼曼，丹尼尔，58，63–64，290

*卡尔·皮尔逊*（波特），67

卡林，塞缪尔，87

卡申，康斯坦丁，228–230

卡蒂雷桑，塞卡尔，256

柯洁，360

肯普索恩，奥斯卡，272

肯尼，大卫，324–325，339

克莱因，埃兹拉，139，154

知识，8，11–12，12（图）

克特尔，雷金纳德，302–304

克拉格，约翰，343–347

克鲁斯卡尔，威廉，312–316，346

因果阶梯，17–19，24，116  
- 关联，28（图），29–30，51  
- 偏倚，311  
- 混杂，140  
- 反事实，266  
- 干预，28（图），31–33，40，219，231  
- 无模型方法，88  
- 观察，264  
- 概率与，47–49，75  
- 查询，28（图），29，32

语言  
- 知识的，8  
- 数学的，3–8  
- 概率的，102–103  
- 查询的，8，10

拉普拉斯，皮埃尔-西蒙，5

拉丁方，145，146（图）

法律，反事实与，286–291

LDL。*参见* 低密度脂蛋白胆固醇

*Let's Make a Deal*。*参见* 示例

刘易斯，大卫，20，266–269

似然比，105–106，113

利林菲尔德，阿贝，175，179–180

林德，詹姆斯，168，299，302–303

林德利，丹尼斯，209

线性因果模型，322–323，327

线性模型，295–296

线性回归，285–286

线性SCM，285–286

狮子人，34–36，35（图）

LISREL，86

逻辑，232，238

罗德悖论。*参见* 示例

低密度脂蛋白（LDL）胆固醇，252–257，254（表）

肺癌，吸烟与，18–19，167–168

机器学习，10–11，30–31，125，363  

*另见* 人工智能（AI）

机器  
- 因果知识，37  
- 思维，367–368  

*另见* 机器人

麦凯，大卫，127–128

马来西亚航空公司坠机，122，123（图）

马库斯，加里，30

匹配，274

数学确定性，288

数学语言，3–8

数学，科学与，4–5，84–85  

*另见* 几何学

M-偏倚，161

麦克唐纳，罗德，325

中介，20  
- “人人学代数”作为，336–339，338（图）  
- 分析，297，300–301，322–323  
- 因果关系中的，300–301  
- 谬误，272，315–316  
- 公式，319，332–333，335  
- 问题，131  
- 吸烟基因示例作为，339–343，341（图），342（图）  
- 阈值效应与，325，326（图）

中介变量，153–154，228，297  
- 混杂因子与，276  
- 直接效应与，326，332  
- 结果与，315–316

孟德尔，格雷戈尔，65

孟德尔遗传学，73

孟德尔随机化，255–256，256（图）

心智模型，26，26（图）

消息传递网络，110–111，111（图）

方法，数据与，84–85

迷你图灵测试，36–46

奇迹，103，357

模型发现，373

模型盲，33，66，132，217，275

**模范刑法典**，286，288

无模型方法，87–89，272，351  

*另见* 模型盲

莫拉比亚，阿尔弗雷多，152–153

干预之山，218（照片），219–220，224，259–260

马斯克，埃隆，367

餐巾问题，239–240，240（图），330

自然直接效应（NDE），318–319，332–333

自然效应，327

自然间接效应（NIE），319，321，325–326，332–333

*自然遗传*（高尔顿），66

自然，144–145，147，149，156，257

天性-教养之争，304–309，305（图），306（图）

NDE。*参见* 自然直接效应

必然因果关系，289–290，295

必然性，概率，294

荷兰法医研究所（NFI），94，122

*参见* 自然间接效应。

> 奈尔斯，亨利，78–81，84

非因果路径，因果图中的，157，160  
非可压缩性，152  
不依从，随机对照试验与，252–253，253（图）  
非混杂性，281  
非线性分析，335  
非随机化研究，149  
诺维克，梅尔文，201，209

**客观性**
- 贝叶斯推断中的，89
- 因果推断的，91

观察性研究，150–151，229  
奥格本，威廉·菲尔丁，309  
“论奇迹”（休谟），96–97  
“论偏相关与多重相关技术的不足”（伯克斯），308  
*物种起源*（达尔文），63

**悖论**，9，19，189–190
- 出生体重，185–186，185（图），189
- 作为视错觉，189–190

*参见* 示例

父节点，111–112，117–118，129  
帕斯卡，布莱兹，4–5  
巴斯德，路易，228

**路径分析**
- 经济学中的，86
- 社会科学中的，85–86
- 赖特，S.，论，86–89，324

路径系数，77，223，251

**路径图**
- 出生体重示例，82–83，82（图）
- 伯克斯的，308–309
- 赖特，S.的，74–77，75（图），85–86，221，260–261

帕兹，阿扎里亚，381  
珀尔，朱迪亚，ix，24，51，328，331  
皮尔逊，埃贡，271–272  
皮尔逊，卡尔，5，62，78，85，180，222
- 因果关系与，71–72
- 论数据，87–88
- 高尔顿与，66–68
- 论头骨尺寸，70（图）
- 论伪相关，69
- 作为狂热者，67–68

哲学家，论因果关系，47–51，81  
物理学，33–34，67，99  
皮古，阿瑟·塞西尔，198  
平托，罗德里戈，236  
安慰剂效应，300  
多项式时间，238  
波特，泰德，67

**潜在结果**，155，260  
潜在结果框架，155  
预测，278，280
- 干预与，32
- 科学中的，36

证据优势，288  
预处理变量，160  
普莱斯，理查德，97  
先验知识，90，104  
概率因果关系，47–51  
*智能系统中的概率推理*（珀尔），51

**概率**，43–44，46，90，110
- 贝叶斯论，97–98，102
- 贝叶斯网络与，358–359
- 若非因果关系中的，287
- 因果关系与，47–51
- 罪责的，288
- 因果阶梯与，47–49，75
- 语言，102–103
- 必然性的，294
- 随时间变化，120–121，121（图）
- 提高，49
- 充分性的，294

*参见* 条件概率；逆概率

概率表，117（表），128–129  
概率论，4–5

**乘积**
- 系数的，327
- 间接效应作为，328–329

普罗文，威廉，85  
临时因果关系，150  
近因，288–289  
毕达哥拉斯，233

**定量因果推理**，43  
查询，8，10，12（图），14–15
- 因果，27，183
- 反事实，20，28（图），36，260–261，284
- 因果阶梯中的，28（图），29，32
- 中介，131

*参见* “为什么？”问题

**随机对照试验（RCT）**，18，132–133，143–147
- 因果图中的，140，148–149，149（图）
- 混杂因子与，149–150
- 流行病学中的，172–173
- 费希尔论，139–140，143–144
- 作为“金标准”，231
- 不依从的因果图，252–253，253（图）
- 观察性研究对比，150，229

重组DNA，369  
数据缩减，85  
回归，29，325  
*参见* 线性回归

回归系数，222–223  
回归线，60–62，61（图），221–222  
均值回归，57–58，67  
赖兴巴赫，汉斯，199，234  
里德，康斯坦丝，271–272

**表征**
- 获取与，38
- 大脑中的信息，39

表征问题，268  
返祖，56–57  
罗宾斯，杰米，168，329–330，329（图），333–334
- 论混杂，150
- 干预演算与，236–237，241
- 论可交换性，154–156

机器人，ix–x
- AI，291
- 因果推断，2，350，361，361（图）
- 与人类交流，366
- 作为道德主体，370
- 足球，365–366

根节点，117  
鲁宾，唐纳德，269–270，270（照片），275，283
- 因果模型，261，280–281
- 论潜在结果，155

鲁梅尔哈特，大卫，110，111（图），268

萨克特，大卫，197–198，198（表）  
*智人*（赫拉利），25  
萨维奇，吉米，316  
萨维奇，伦纳德，204–206  
散点图，59（图），60，62  
沙因斯，理查德，350  
舒曼，伦纳德，182

**科学**
- 数据中的，6，84–85
- 历史，4–5
- 数学与，4–5，84–85
- 预测，36

*参见* 因果推断；社会科学

科学方法，108，302  
SCM。*参见* 结构因果模型

斯科特，罗伯特·法尔孔，298（照片），302，303（图）  
塞尔，约翰，38，363  
安全带使用，161–162  
李世石，360  
*看与做*，8–9，27，130，149，233  
自我意识，363，367  
SEM。*参见* 结构方程模型

敏感性分析，176  
序贯治疗，241（图）  
谢弗，格伦，109  
夏普，玛丽亚，68  
夏洛克·福尔摩斯，92（照片），93  
什皮策，伊利亚，24，238–239，243，245，296–297  
硅谷，32  
西蒙，赫伯特，79，198  
辛普森，爱德华，153–154，208–209  
辛普森悖论。*参见* 示例

**吸烟。** *参见* 示例；外科医生咨询委员会；烟草行业

吸烟基因，174–175，224–227，339–343，341（图），342（图）  
吸烟-癌症争论，166（照片），167–179  
斯诺，约翰，168，245–249  
社会科学，84–86  
社会地位，307  
大二低迷，56–58  
斯珀茨，彼得，244  
斯波恩，沃尔夫冈，350  
伪相关，69–72  
虚假效应，138

**稳定单位处理值假设（SUTVA）**，280–281  
斯坦福-比奈智商测试，305–306  
统计估计，12（图），15

**统计学**，5–6，9
- 人体测量与，58
- 固定程序，84–85
- 因果推断，18
- 因果关系与，18，66，190
- 混杂因子，138–139，141–142
- 方法，31，180–181
- 客观性与
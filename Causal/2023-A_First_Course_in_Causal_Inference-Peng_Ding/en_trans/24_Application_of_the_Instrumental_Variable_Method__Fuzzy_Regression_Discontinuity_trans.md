# 工具变量法的应用：模糊断点回归（Application of the Instrumental Variable Method: Fuzzy Regression Discontinuity）

第20章介绍的**断点回归（regression discontinuity）**和第21–23章介绍的**工具变量（instrumental variable）**是自然实验的两个重要范例。这些研究设计不如第二部分中的随机实验那样理想，但它们具有与实验相似的特征。这就是它们被称为自然实验的原因。

将断点回归与工具变量相结合，便产生了**模糊断点回归（fuzzy regression discontinuity）**，这是另一个重要的自然实验。我将从示例开始，然后提供数学表述。

## 24.1 动机示例（Motivating examples）

第20章介绍了断点回归。以下两个示例略有不同，因为所接受的**处理（treatments）**不是**运行变量（running variables）**的确定性函数。相反，运行变量在**断点（cutoff point）**处不连续地改变了所接受处理的概率。

**示例 24.1** 2000年，印度政府启动了总理乡村道路计划（Prime Minister's Village Road Program），截至2015年，该计划已资助修建了近20万个村庄的全天候道路。基于村庄层面的数据，Asher 和 Novosad（2020）使用断点回归来估计新建支线道路对各种经济变量的影响。国家计划指南根据2001年人口普查的任意阈值优先考虑较大的村庄。如果该村庄在结果测量的年份之前获得了新道路，则处理变量等于1。村庄人口规模与阈值之间的差异并未决定处理变量，但在断点零处不连续地影响了其概率。

**示例 24.2** Li 等人（2015）使用了2004至2006年两所意大利大学一年级学生的数据，评估大学助学金对辍学率的因果效应。如果学生的标准化家庭收入低于15,000欧元，则有资格获得此项助学金。为简单起见，我们使用定义为15,000减去标准化家庭收入的运行变量。要获得此项助学金，学生必须首先提出申请。因此，资格和申请状态共同决定了最终的处理状态。运行变量本身并未决定处理状态，尽管它在断点零处改变了处理概率。

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

**图 24.1：** 精确断点回归（左）和模糊断点回归（右）的处理分配

**示例 24.3** Amarante 等人（2016）估计了子宫内暴露于社会救助计划对儿童出生结果的影响。他们使用了由乌拉圭国家社会紧急救助计划（Plan de Atención Nacional a la Emergencia Social）引发的断点回归。这是一项针对最贫困10%家庭的临时社会救助计划，于2005年4月至2007年12月实施。预测低收入得分低于预定阈值的家庭被分配到该计划。预测收入得分并未决定母亲在怀孕期间是否至少收到一次计划转移支付，但改变了最终所接受处理的概率。出生结果包括出生体重、孕周等。

上述示例被称为**模糊断点回归（fuzzy regression discontinuity）**，与第20章中的**（精确）断点回归（(sharp) regression discontinuity）**形成对比。我将在下文第24.3节中分析示例24.1和24.2中的数据。

## 24.2 数学表述（Mathematical formulation）

设 $X _ { i }$ 表示运行变量，它决定了 $Z _ { i } ~ = ~ 1 ( X _ { i } ~ \geq ~ x _ { 0 } )$，其中断点为 $x _ { 0 }$。所接受的处理 $D _ { i }$ 可能不等于 $Z _ { i }$，但 $\mathrm { p r } ( D _ { i } = 1 \mid X _ { i } = x )$ 在 $x _ { 0 }$ 处有一个跳跃。图24.1比较了精确断点回归和模糊断点回归所接受处理的概率。它展示了 $\operatorname { p r } ( D = 1 \mid X < x _ { 0 } ) = 0$ 的模糊断点回归的一个特例，这与示例24.2一致。

设 $Y _ { i }$ 表示感兴趣的结果。将 $Z _ { i }$ 视为分配的处理，我们可以定义**潜在结果（potential outcomes）** $\{ D _ { i } ( 1 ) , D _ { i } ( 0 ) , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \}$。基于定理20.2，Z的精确断点回归允许识别

$$
\begin{array}{l} \tau_ {D} (x _ {0}) = E \{D (1) - D (0) \mid X = x _ {0} \} \\ = \lim _ {\varepsilon \rightarrow 0 +} E (D \mid Z = 1, X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \rightarrow 0 +} E (D \mid Z = 0, X = x _ {0} - \varepsilon) \\ \end{array}
$$

和

$$
\begin{array}{l} \tau_ {Y} (x _ {0}) = E \{Y (1) - Y (0) \mid X = x _ {0} \} \\ = \lim _ {\varepsilon \rightarrow 0 +} E (Y \mid Z = 1, X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \rightarrow 0 +} E (Y \mid Z = 0, X = x _ {0} - \varepsilon) \\ \end{array}
$$

使用 $Z$ 作为 D 的 IV，并在 $X = x _ { 0 }$ 处施加 IV 假设，我们可以通过应用定理21.1来识别**局部依从者平均因果效应（local complier average causal effect）**。

**定理 24.1** 假设在 $x _ { 0 }$ 的无穷小邻域内

$$
D _ {i} (1) \geq D _ {i} (0)
$$

且

$$
D _ {i} (1) = D _ {i} (0) \Longrightarrow Y _ {i} (1) = Y _ {i} (0)
$$

成立。则局部依从者平均因果效应等于

$$
\begin{array}{l} \tau_ {\mathrm{c}} (x _ {0}) \equiv E \{Y (1) - Y (0) \mid D (1) > D (0), X = x _ {0} \} \\ = \frac {E \{Y (1) - Y (0) \mid X = x _ {0} \}}{E \{D (1) - D (0) \mid X = x _ {0} \}}. \\ \end{array}
$$

进一步假设 $E \{ D ( 1 ) \mid X = x \}$ 和 $E \{ Y ( 1 ) \mid X = x \}$ 在 $X = x _ { 0 }$ 处右连续，且 $E \{ D ( 0 ) \mid X = x \}$ 和 $E \{ Y ( 0 ) \mid X = x \}$ 在 $X = x _ { 0 }$ 处左连续。如果 $E ( D \mid Z = 1 , X = x )$ 在 $X = x _ { 0 }$ 处有一个非零跳跃，则局部依从者平均因果效应可以通过下式识别

$$
\tau_ {\mathrm{c}} (x _ {0}) = \frac {\lim _ {\varepsilon \to 0 +} E (Y \mid Z = 1 , X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \to 0 +} E (Y \mid Z = 0 , X = x _ {0} - \varepsilon)}{\lim _ {\varepsilon \to 0 +} E (D \mid Z = 1 , X = x _ {0} + \varepsilon) - \lim _ {\varepsilon \to 0 +} E (D \mid Z = 0 , X = x _ {0} - \varepsilon)}
$$

定理24.1是定理20.2和定理21.1的叠加。我将其证明留作问题24.1。

在精确断点回归和模糊断点回归中，关键在于指定断点周围的邻域。在实践中，较小的邻域会导致较小的偏差但较大的方差，而较大的邻域会导致较大的偏差但较小的方差。也就是说，我们面临**偏差-方差权衡（bias-variance tradeoff）**。存在一些基于某些统计准则的自动程序，这些程序依赖于一些强条件。对 $h$ 的选择进行一系列**敏感性分析（sensitivity analysis）**似乎是更明智的做法。

## 29624 工具变量法的应用：模糊断点回归（Application of the Instrumental Variable Method: Fuzzy Regression Discontinuity）

假设我们已经指定了由带宽 h 确定的 $x _ { 0 }$ 的邻域。对于 $X _ { i } \in [ x _ { 0 } - h , x _ { 0 } + h ]$ 的数据，我们可以通过以下方式估计 $\tau _ { D } ( x _ { 0 } )$

τˆD(x0) = 在 $D _ { i }$ 对 $\{ 1 , Z _ { i } , R _ { i } , L _ { i } \}$ 的 OLS 拟合中 $Z _ { i }$ 的系数，

并估计 $\tau _ { Y } ( x _ { 0 } )$

τˆY (x0) = 在 $Y _ { i }$ 对 $\{ 1 , Z _ { i } , R _ { i } , L _ { i } \}$ 的 OLS 拟合中 $Z _ { i }$ 的系数，

回顾定义 $R _ { i } = \operatorname* { m a x } ( X _ { i } - x _ { 0 } , 0 )$ 和 $L _ { i } = \operatorname* { m i n } ( X _ { i } - x _ { 0 } , 0 )$。然后我们可以通过下式估计局部依从者平均因果效应

$$
\hat {\tau} _ {\mathrm{c}} (x _ {0}) = \hat {\tau} _ {Y} (x _ {0}) / \hat {\tau} _ {D} (x _ {0}).
$$

这是一个**间接最小二乘估计量（indirect least squares estimator）**。根据定理23.1，它在数值上等同于

在 $Y _ { i }$ 对 $\{ 1 , D _ { i } , R _ { i } , L _ { i } \}$ 的 TSLS 拟合中 $D _ { i }$ 的系数

其中 $D _ { i }$ 由 $Z _ { i }$ 作为工具变量。总之，在指定 h 之后，$\tau _ { \mathrm { c } } ( x _ { 0 } )$ 的估计简化为对断点周围局部数据的 TSLS 程序。

## 24.3 应用（Application）

## 24.3.1 重新分析 Asher 和 Novosad (2020) 的数据

图24.2显示了使用 `occupationindexandrsn` 作为结果变量的结果。

`rdrobust` 包会自动选择带宽。结果表明，获得新道路并未显著影响结果变量。

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

## 24.3.2 重新分析 Li 等人 (2015) 的数据

回顾示例24.2中的运行变量是15,000减去标准化收入。在分析中，我将数据限制在运行变量位于 $[-5, 000, 5, 000]$ 范围内的子集，然后将运行变量除以5,000，使得运行变量在断点零处介于 $[-1, 1]$ 之间。

基于 `rdrobust` 包的结果表明，大学助学金并未显著影响辍学率。

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

## 24.4 讨论（Discussion）

第20章和本章都基于潜在结果在给定运行变量下的条件期望的连续性来表述断点回归。这种视角在数学上更简单，但它仅在运行变量的断点处精确识别局部效应。Hahn 等人（2001）开创了这一文献方向。

另一种不那么主流的视角基于**局部随机化（local randomization）**（Cattaneo et al., 2015; Li et al., 2015）。如果我们把运行变量视为对某些潜在真相的有噪声度量，并且断点是某种程度上的任意选择，那么断点附近的单元并没有系统性差异。这表明，在断点的一个小邻域内，单元接受处理和对照的方式是随机的，就像在随机实验中一样。与第一种视角中选择 h 的问题类似，关键在于决定在断点回归下随机实验应该有多“局部”。用数学方法量化这种直觉并不容易，再次地，对一系列 h 进行敏感性分析在第二种视角下似乎也是一种合理的方法。

关于断点回归的更多概念性讨论，请参见 Sekhon 和 Titiunik（2017）。

## 24.5 家庭作业问题（Homework Problems）

## 24.1 定理24.1的证明（Proof of Theorem 24.1）

证明定理24.1。

## 24.2 数据分析（Data analysis）

第24.3.1节估计了对 `occupationindexandrsn` 的效应。另外四个结果变量是 `transportindexandrsn`、`firmsindexandrsn`、

## 30024 工具变量法的应用：模糊断点回归（Application of the Instrumental Variable Method: Fuzzy Regression Discontinuity）

`consumptionindexandrsn` 和 `agricultureindexandrsn`，其含义在原论文中定义。估计这些结果变量的效应。

## 24.3 对 Li 等人 (2015) 数据分析的反思（Reflection on the analysis of Li et al. (2015)’s data）

在 Li 等人（2015）的研究中，决定处理状态的一个关键变量是**二元申请状态（binary application status）** A，其对应于处理 $Z = 1$ 和对照 $Z = 0$ 的潜在结果分别为 $A ( 1 )$ 和 $A ( 0 )$。根据定义，

$$
D (1) = A (1), \quad D (0) = 0,
$$

所以**依从者（compliers）** $\{ D ( 1 ) , D ( 0 ) \} = ( 1 , 0 )$ 等价于 $A ( 1 ) = 1$。因此

$$
\tau_ {c} (x _ {0}) = E \{Y (1) - Y (0) \mid A (1) = 1, X = x _ {0} \}.
$$

第24.3.2节使用了整个数据集来估计 $\tau _ { \mathrm { c } } ( x _ { 0 } )$。

另一种分析基于仅 $A = 1$ 的单元。那么处理状态由 X 决定。然而，这种分析可能存在问题，因为

$$
\lim _ {\varepsilon \rightarrow 0 +} E \{Y \mid A = 1, X = x _ {0} + \varepsilon \} - \lim _ {\varepsilon \rightarrow 0 +} E \{Y \mid A = 1, X = x _ {0} - \varepsilon \}
$$

$$
= E \{Y (1) \mid A (1) = 1, X = x _ {0} \} - E \{Y (0) \mid A (0) = 1, X = x _ {0} \}. \tag {24.1}
$$

证明 (24.1) 并解释为什么这种分析可能存在问题。

**注：** (24.1) 的左边是在 $X = x _ { 0 }$ 处，以 $A = 1$ 为条件的局部平均处理效应的识别公式。(24.1) 的右边分别是对于 $( A ( 1 ) = 1 , X = x _ { 0 } )$ 和 $( A ( 0 ) = 1 , X = x _ { 0 } )$ 子组单元的潜在结果的均值之差。

## 24.4 推荐阅读（Recommended reading）

Imbens 和 Lemieux（2008）基于潜在结果框架为断点回归提供了实用指南。Lee 和 Lemieux（2010）回顾了断点回归及其在经济学中的应用。

## 25
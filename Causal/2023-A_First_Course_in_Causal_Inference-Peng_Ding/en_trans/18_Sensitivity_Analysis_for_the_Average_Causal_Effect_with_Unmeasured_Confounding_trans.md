# 未测量混杂因素下平均因果效应的敏感性分析（Sensitivity Analysis for the Average Causal Effect with Unmeasured Confounding）

**Cornfield型敏感性分析**在**风险比尺度（risk ratio scale）**上对二元结果效果最佳，且需以观测协变量为条件。尽管 Ding 和 VanderWeele（2016）也提出了针对**平均因果效应（average causal effect）**的 Cornfield 型敏感性分析方法，但它们不够通用且应用不便。下面我将给出一种更直接的敏感性分析方法，该方法基于**潜在结果的条件期望（conditional expectations of the potential outcomes）**。这一思想最早出现在 Robins（1999）和 Scharfstein 等人（1999）的研究中。本章基于 Lu 和 Ding（2023）的最新表述。

该方法与推导**平均潜在结果的最坏情况边界（worst-case bounds）**的思想密切相关。我将首先回顾较为简单的边界思想，然后讨论敏感性分析的方法。

## 18.1 引言（Introduction）

$\{ Z _ { i } , X _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \} _ { i = 1 } ^ { n } \stackrel { \mathrm { I I D } } { \sim }$ {Z, X, Y (1), Y (0)}，并关注**平均因果效应**

$$
\tau = E \{Y (1) - Y (0) \}.
$$

它可分解为

$$
\begin{array}{l} \tau = \left[ E (Y \mid Z = 1) \mathrm{pr} (Z = 1) + E \{Y (1) \mid Z = 0 \} \mathrm{pr} (Z = 0) \right] \\ - \left[ E \{Y (0) \mid Z = 1 \} \mathrm{pr} (Z = 1) + E (Y \mid Z = 0) \mathrm{pr} (Z = 0) \right]. \\ \end{array}
$$

因此，根本困难在于估计**反事实均值（counterfactual means）**

$$
E \{Y (1) \mid Z = 0 \}, \qquad E \{Y (0) \mid Z = 1 \}.
$$

通常存在两种极端策略来估计它们。

我们在第三部分讨论了第一种策略，该策略依赖于**可忽略性（ignorability）**。假设

$$
\begin{array}{l} E \{Y (1) \mid Z = 1, X \} = E \{Y (1) \mid Z = 0, X \}, \\ E \{Y (0) \mid Z = 1, X \} = E \{Y (0) \mid Z = 0, X \}, \\ \end{array}
$$

**表 18.1：具有有界结果 [ℓ, u] 的科学表（Science Table with bounded outcome [ℓ, u]），其中 ℓ 和 u 为两个常数**

<table><tr><td>Z</td><td>Y(1)</td><td>Y(0)</td><td>Lower Y(1)</td><td>Upper Y(1)</td><td>Lower Y(0)</td><td>Upper Y(0)</td></tr><tr><td>1</td><td> $Y_1(1)$ </td><td>?</td><td> $Y_1(1)$ </td><td> $Y_1(1)$ </td><td> $\ell$ </td><td>u</td></tr><tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr><tr><td>1</td><td> $Y_{n_1}(1)$ </td><td>?</td><td> $Y_{n_1}(1)$ </td><td> $Y_{n_1}(1)$ </td><td> $\ell$ </td><td>u</td></tr><tr><td>0</td><td>?</td><td> $Y_{n_1+1}(0)$ </td><td> $\ell$ </td><td>u</td><td> $Y_{n_1+1}(0)$ </td><td> $Y_{n_1+1}(0)$ </td></tr><tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr><tr><td>0</td><td>?</td><td> $Y_n(0)$ </td><td> $\ell$ </td><td>u</td><td> $Y_n(0)$ </td><td> $Y_n(0)$ </td></tr></table>

我们可以通过可观测变量识别反事实均值：

$$
E \{Y (1) \mid Z = 0 \} = E \left\{E (Y \mid Z = 1, X) \mid Z = 0 \right\}
$$

类似地，

$$
E \{Y (0) \mid Z = 1 \} = E \left\{E (Y \mid Z = 0, X) \mid Z = 1 \right\}.
$$

下一节中的第二种策略除了假设结果在 ℓ 和 u 之间有界外，不做任何其他假设。这对于 ℓ = 0 且 u = 1 的二元结果是自然的。在此假设下，两个反事实均值也在 ℓ 和 u 之间有界，这意味着 τ 存在最坏情况边界。我将在下面回顾这一策略。

## 18.2 无假设下平均因果效应的 Manski 型最坏情况边界（Manski-type worse-case bounds on the average causal effect without assumptions）

假设结果在 ℓ 和 u 之间有界。根据分解式

$$
E \{Y (1) \} = E \{Y (1) \mid Z = 1 \} \mathrm{pr} (Z = 1) + E \{Y (1) \mid Z = 0 \} \mathrm{pr} (Z = 0),
$$

我们可以推导出 $E\{Y(1)\}$ 的下界为

$$
E \{Y \mid Z = 1 \} \mathrm{pr} (Z = 1) + \ell \mathrm{pr} (Z = 0)
$$

上界为

$$
E \{Y \mid Z = 1 \} \mathrm{pr} (Z = 1) + u \mathrm{pr} (Z = 0).
$$

类似地，根据分解式

$$
E \{Y (0) \} = E \{Y (0) \mid Z = 1 \} \mathrm{pr} (Z = 1) + E \{Y (0) \mid Z = 0 \} \mathrm{pr} (Z = 0),
$$

## 18.3 无假设下平均因果效应的 Manski 型最坏情况边界 227

我们可以推导出 $E \{ Y ( 0 ) \}$ 的下界为

$$
\ell \mathrm{pr} (Z = 1) + E \{Y \mid Z = 0 \} \mathrm{pr} (Z = 0)
$$

上界为

$$
u \mathrm{pr} (Z = 1) + E \{Y \mid Z = 0 \} \mathrm{pr} (Z = 0).
$$

结合这些边界，我们可以推导出**平均因果效应** $\tau =$ $E \{ Y ( 1 ) \} - E \{ Y ( 0 ) \}$ 的下界为

$$
E \{Y \mid Z = 1 \} \mathrm{pr} (Z = 1) + \ell \mathrm{pr} (Z = 0) - u \mathrm{pr} (Z = 1) - E \{Y \mid Z = 0 \} \mathrm{pr} (Z = 0)
$$

上界为

$$
E \{Y \mid Z = 1 \} \mathrm{pr} (Z = 1) + u \mathrm{pr} (Z = 0) - \ell \mathrm{pr} (Z = 1) - E \{Y \mid Z = 0 \} \mathrm{pr} (Z = 0).
$$

边界的长度为 $u - \ell$，这虽然不具信息量，但优于长度为 $2 ( u - \ell )$ 的先验边界 $\left[ \ell - u , u - \ell \right]$。没有进一步的假设，观测数据的分布无法唯一确定 τ。在这种情况下，我们说 τ 是**部分可识别的（partially identified）**，其正式定义如下。

**定义 18.1（部分识别，partial identification）** 如果观测数据分布与 θ 的多个值相容，则称参数 θ 是部分可识别的。

比较定义 10.1 和 18.1。如果参数 θ 由观测数据分布唯一确定，则它是**可识别的（identifiable）**；否则，它是部分可识别的。因此，在可忽略性假设下 τ 是可识别的，但在没有可忽略性假设时仅部分可识别。

Cochran（1953）在缺失数据调查中使用了最坏情况边界的思想，但由于该思想通常给出非常保守的结果而放弃了它。类似地，上述最坏情况边界从实践角度来看通常无意义，因为它们往往覆盖 0。此外，这一策略不适用于结果无界的场景。

Manski 将这一思想应用于因果推断（Manski, 1990）和许多其他计量经济学模型（Manski, 2003）。这种在最小假设下对因果参数进行边界化的思想，若与其他定性假设结合，则非常强大。Manski（2003）综述了许多策略。例如，我们可能认为处理不会对任何个体造成伤害，因此**单调性假设（monotonicity assumption）**成立：$Y ( 1 ) \ge Y ( 0 )$。那么 τ 的下界为零，但上界不变。另一种假设类型是 $Z = I \{ Y ( 1 ) \geq Y ( 0 ) ]$，即处理选择基于潜在结果之间的差异。这一假设也可以改进 τ 的边界。

## 18.3 平均因果效应的敏感性分析（Sensitivity analysis for the average causal effect）

第一种策略是乐观的，它假设在给定观测协变量的条件下，潜在结果在处理组和对照组之间没有差异。第二种策略是悲观的，它完全不基于观测数据推断反事实均值。以下策略介于两者之间。

### 18.3.1 识别公式（Identification formulas）

定义

$$
\frac {E \{Y (1) \mid Z = 1 , X \}}{E \{Y (1) \mid Z = 0 , X \}} = \varepsilon_ {1} (X),
$$

$$
\frac {E \{Y (0) \mid Z = 1 , X \}}{E \{Y (0) \mid Z = 0 , X \}} = \varepsilon_ {0} (X),
$$

这些是**敏感性参数（sensitivity parameters）**。为简化起见，我们可以进一步假设它们独立于 X 为常数。在实践中，我们需要固定它们或在预先指定的范围内变化它们。回顾 $\mu _ { 1 } ( X ) = E ( Y \mid Z = 1 , X )$ 和 $\mu _ { 0 } ( X ) = E ( Y \mid Z = 0 , X )$。我们可以如下识别两个反事实均值和平均因果效应。

**定理 18.1** 已知 $\varepsilon _ { 1 } ( X )$ 和 $\varepsilon _ { 0 } ( X )$，我们有

$$
E \{Y (1) \mid Z = 0 \} = E \left\{\mu_ {1} (X) / \varepsilon_ {1} (X) \mid Z = 0 \right\},
$$

$$
E \{Y (0) \mid Z = 1 \} = E \left\{\mu_ {0} (X) \varepsilon_ {0} (X) \mid Z = 1 \right\}
$$

因此

$$
\begin{array}{l} \tau = E \{Z Y + (1 - Z) \mu_ {1} (X) / \varepsilon_ {1} (X) \} \\ - E \{Z \mu_ {0} (X) \varepsilon_ {0} (X) + (1 - Z) Y \} (18.1) \\ = E \left\{Z \mu_ {1} (X) + (1 - Z) \mu_ {1} (X) / \varepsilon_ {1} (X) \right\} \\ - E \{Z \mu_ {0} (X) \varepsilon_ {0} (X) + (1 - Z) \mu_ {0} (X) \}. (18.2) \\ \end{array}
$$

我将定理 18.1 的证明留给问题 18.1。利用拟合的结果模型，(18.1) 和 (18.2) 为 τ 提出了以下**预测性（predictive）**和**投影性（projective）**估计量：

$$
\begin{array}{l} \hat {\tau} ^ {\mathrm{pred}} = \left\{n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} Y _ {i} + n ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \hat {\mu} _ {1} (X _ {i}) / \varepsilon_ {1} (X _ {i}) \right\} \\ - \left\{n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \hat {\mu} _ {0} (X _ {i}) \varepsilon_ {0} (X _ {i}) + n ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) Y _ {i} \right\}, \\ \end{array}
$$

和

$$
\begin{array}{l} \hat {\tau} ^ {\text { proj }} = \left\{n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \hat {\mu} _ {1} (X _ {i}) + n ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \hat {\mu} _ {1} (X _ {i}) / \varepsilon_ {1} (X _ {i}) \right\} \\ \left. - \left\{n ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \hat {\mu} _ {0} (X _ {i}) \varepsilon_ {0} (X _ {i}) + n ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \hat {\mu} _ {0} (X _ {i}) \right\}. \right. \\ \end{array}
$$

术语“预测性”和“投影性”来自调查抽样文献（Firth and Bennett, 1998; Ding and Li, 2018）。估计量 $\hat{\tau}^{\mathrm{pred}}$ 和 $\hat { \tau } ^ { \mathrm { p r o j } }$ 略有不同：前者在可用时使用观测结果；相比之下，后者用拟合值替换观测结果。

更有趣的是，我们还可以通过**逆概率加权公式（inverse probability weighting formula）**来识别 τ。

**定理 18.2** 已知 $\varepsilon _ { 1 } ( X )$ 和 $\varepsilon _ { 0 } ( X )$，我们有

$$
E \{Y (1) \} = E \left\{w _ {1} (X) \frac {Z}{e (X)} Y \right\}, \quad E \{Y (0) \} = E \left\{w _ {0} (X) \frac {1 - Z}{1 - e (X)} Y \right\},
$$

其中

$$
w _ {1} (X) = e (X) + \{1 - e (X) \} / \varepsilon_ {1} (X), w _ {0} (X) = e (X) \varepsilon_ {0} (X) + 1 - e (X).
$$

我将定理 18.2 的证明留给问题 18.2。定理 18.2 修改了经典的逆概率加权公式，增加了两个额外因子 $w _ { 1 } ( X )$ 和 $w _ { 0 } ( X )$，它们同时依赖于**倾向得分（propensity score）**和敏感性参数。利用拟合的倾向得分模型，定理 18.2 为 τ 提出了以下估计量：

$$
\begin{array}{l} \hat {\tau} ^ {\mathrm{ht}} = n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {\{\hat {e} (X _ {i}) \varepsilon_ {1} (X _ {i}) + 1 - \hat {e} (X _ {i}) \} Z _ {i} Y _ {i}}{\varepsilon_ {1} (X _ {i}) \hat {e} (X _ {i})} \\ - n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {\{\hat {e} (X _ {i}) \varepsilon_ {0} (X _ {i}) + 1 - \hat {e} (X _ {i}) \} (1 - Z _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})} \\ \end{array}
$$

和

$$
\begin{array}{l} \hat {\tau} ^ {\text { haj }} = \sum_ {i = 1} ^ {n} \frac {\{\hat {e} (X _ {i}) \varepsilon_ {1} (X _ {i}) + 1 - \hat {e} (X _ {i}) \} Z _ {i} Y _ {i}}{\varepsilon_ {1} (X _ {i}) \hat {e} (X _ {i})} / \sum_ {i = 1} ^ {n} \frac {Z _ {i}}{\hat {e} (X _ {i})} \\ - n ^ {- 1} \sum_ {i = 1} ^ {n} \frac {\{\hat {e} (X _ {i}) \varepsilon_ {0} (X _ {i}) + 1 - \hat {e} (X _ {i}) \} (1 - Z _ {i}) Y _ {i}}{1 - \hat {e} (X _ {i})} \Big / \sum_ {i = 1} ^ {n} \frac {1 - Z _ {i}}{1 - \hat {e} (X _ {i})}. \\ \end{array}
$$

更有趣的是，利用拟合的倾向得分和结果模型，以下 τ 的估计量是**双重稳健的（doubly robust）**：

$$
\hat {\tau} ^ {\mathrm{ht}} = \hat {\tau} ^ {\mathrm{ipw}} - n ^ {- 1} \sum_ {i = 1} ^ {n} \{Z _ {i} - \hat {e} (X _ {i}) \} \left\{\frac {\hat {\mu} _ {1} (X _ {i})}{\hat {e} (X _ {i}) \varepsilon_ {1} (X _ {i})} + \frac {\hat {\mu} _ {0} (X _ {i}) \varepsilon_ {0} (X _ {i})}{1 - \hat {e} (X _ {i})} \right\}.
$$

也就是说，当 $\varepsilon _ { 1 } ( X _ { i } )$ 和 $\varepsilon _ { 0 } ( X _ { i } )$ 已知时，如果倾向得分模型或结果模型中的任何一个被正确指定，则估计量 ${ \hat { \tau } } ^ { \mathrm { d r } }$ 对 τ 是一致的。我们可以使用**自助法（bootstrap）**来近似上述估计量的方差。技术细节请参见 Lu 和 Ding（2023）。

当 $\varepsilon _ { 1 } ( X _ { i } ) = \varepsilon _ { 0 } ( X _ { i } ) = 1$ 时，上述估计量退化为第三部分中介绍的预测性估计量、逆概率加权估计量和双重稳健估计量。

## 18.4 示例（Example）

我们重新审视示例 10.3。令

$$
\varepsilon_ {1} (X) = \varepsilon_ {0} (X) \in \{1 / 2, 1 / 1. 7, 1 / 1. 5, 1 / 1. 3, 1, 1. 3, 1. 5, 1. 7, 2 \},
$$

我们得到了一系列 τ 的双重稳健估计值。

<table><tr><td></td><td>1/2</td><td>1/1.7</td><td>1/1.5</td><td>1/1.3</td><td>1</td><td>1.3</td><td>1.5</td><td>1.7</td><td></td></tr><tr><td colspan="10">2</td></tr><tr><td>1/2</td><td>11.62</td><td>10.44</td><td>9.40</td><td>8.03</td><td>4.96</td><td>0.97</td><td>-1.69</td><td>-4.35</td><td>-8.34</td></tr><tr><td>1/1.7</td><td>9.22</td><td>8.05</td><td>7.00</td><td>5.64</td><td>2.57</td><td>-1.42</td><td>-4.08</td><td>-6.75</td><td>-10.74</td></tr><tr><td>1/1.5</td><td>7.63</td><td>6.45</td><td>5.41</td><td>4.05</td><td>0.97</td><td>-3.02</td><td>-5.68</td><td>-8.34</td><td>-12.33</td></tr><tr><td>1/1.3</td><td>6.03</td><td>4.86</td><td>3.81</td><td>2.45</td><td>-0.62</td><td>-4.61</td><td>-7.27</td><td>-9.94</td><td>-13.93</td></tr><tr><td>1</td><td>3.64</td><td>2.47</td><td>1.42</td><td>0.06</td><td>-3.01</td><td>-7.01</td><td>-9.67</td><td>-12.33</td><td>-16.32</td></tr><tr><td>1.3</td><td>1.80</td><td>0.63</td><td>-0.42</td><td>-1.78</td><td>-4.85</td><td>-8.85</td><td>-11.51</td><td>-14.17</td><td>-18.16</td></tr><tr><td>1.5</td><td>0.98</td><td>-0.19</td><td>-1.24</td><td>-2.60</td><td>-5.67</td><td>-9.66</td><td>-12.33</td><td>-14.99</td><td>-18.98</td></tr><tr><td>1.7</td><td>0.36</td><td>-0.82</td><td>-1.86</td><td>-3.23</td><td>-6.30</td><td>-10.29</td><td>-12.95</td><td>-15.61</td><td>-19.60</td></tr><tr><td>2</td><td>-0.35</td><td>-1.52</td><td>-2.57</td><td>-3.93</td><td>-7.00</td><td>-10.99</td><td>-13.65</td><td>-16.32</td><td>-20.31</td></tr></table>

估计值的符号对大于 1 的敏感性参数不敏感，但对小于 1 的敏感性参数相当敏感。当膳食计划的参与者倾向于具有较高的 BMI 时，膳食计划对 BMI 的平均因果效应为负。然而，如果膳食计划的参与者倾向于具有较低的 BMI，这一结论可能相当敏感。

## 18.5 家庭作业题（Homework Problems）

## 18.1 定理 18.1 的证明（Proof of Theorem 18.1）

证明定理 18.1。

## 18.2 定理 18.2 的证明（Proof of Theorem 18.2）

证明定理 18.2。

## 18.3 对处理单元的平均因果效应 $\tau _ { \mathrm { T } }$ 的敏感性分析（Sensitivity analysis for the average causal effect on the treated units $\tau _ { \mathrm { T } }$）

本题将第 13 章的内容进行扩展，允许存在未测量的混杂因素，以估计

$$
\tau_ {\mathrm{T}} = E \{Y (1) - Y (0) \mid Z = 1 \} = E (Y \mid Z = 1) - E \{Y (0) \mid Z = 1 \}.
$$

我们可以很容易地用样本矩来估计 $E ( Y \mid Z = 1 )$ 。唯一的反事实项是 $E \{ Y ( 0 ) \mid Z = 1 \}$ 。因此，我们只需要**敏感性参数（sensitivity parameter）** $\varepsilon _ { 0 } ( X )$ 。在已知 $\varepsilon _ { 0 } ( X )$ 的情况下，我们有以下两个识别公式。

**定理 18.3（Theorem 18.3）** 在已知 $\varepsilon _ { 0 } ( X )$ 的情况下，我们有

$$
\begin{array}{l} E \{Y (0) \mid Z = 1 \} = E \left\{Z \mu_ {0} (X) \varepsilon_ {0} (X) \right\} / e \\ { = } { E \left\{ e ( X ) \varepsilon _ { 0 } ( X ) \frac { 1 - Z } { 1 - e ( X ) } Y \right\} / e , } \\ \end{array}
$$

其中 $e = \operatorname { p r } ( Z = 1 )$ 。

证明定理 18.3。

注记：定理 18.3 启发我们使用 $\begin{array} { r } { \hat { \mu } _ { \mathrm { T 1 } } = \sum _ { i = 1 } ^ { n } Z _ { i } Y _ { i } / \sum _ { i = 1 } ^ { n } Z _ { i } } \end{array}$ 和 $\hat { \tau } _ { \mathrm { r } } ^ { * } = \hat { \mu } _ { \mathrm { T 1 } } - \hat { \mu } _ { \mathrm { T 0 } } ^ { * }$ 来估计 $\tau _ { \mathrm { { T } } }$ 。

$$
\begin{array}{l} \hat {\mu} _ {\mathrm{T0}} ^ {\mathrm{reg}} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \varepsilon_ {0} (X _ {i}) \hat {\mu} _ {0} (X _ {i}), \\ \hat {\mu} _ {\mathrm{T0}} ^ {\mathrm{ht}} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} \varepsilon_ {0} (X _ {i}) \hat {o} (X _ {i}) (1 - Z _ {i}) Y _ {i}, \\ \hat {\mu} _ {\mathrm{T0}} ^ {\text { haj }} = \sum_ {i = 1} ^ {n} \varepsilon_ {0} (X _ {i}) \hat {o} (X _ {i}) (1 - Z _ {i}) Y _ {i} / \sum_ {i = 1} ^ {n} \hat {o} (X _ {i}) (1 - Z _ {i}), \\ \end{array}
$$

其中 $\hat { o } ( X _ { i } ) = \hat { e } ( X _ { i } ) / \{ 1 - \hat { e } ( X _ { i } ) \}$ 是估计的处理条件**优势比（odds）**。此外，我们可以为 $\tau _ { \mathrm { T } }$ 构建**双重稳健估计量（doubly robust estimator）** $\hat { \tau } _ { \mathrm { r } } ^ { \mathrm { d r } } ~ =$ $\hat { \mu } _ { \mathrm { T 1 } } - \hat { \mu } _ { \mathrm { T 0 } } ^ { \mathrm { d r } }$ ，其中

$$
\hat {\mu} _ {\mathrm{T0}} ^ {\mathrm{dr}} = \hat {\mu} _ {\mathrm{T0}} ^ {\mathrm{ht}} - n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} \varepsilon_ {0} (X _ {i}) \frac {\hat {e} (X _ {i}) - Z}{1 - \hat {e} (X _ {i})} \hat {\mu} _ {0} (X _ {i}).
$$

Lu 和 Ding (2023) 提供了更多细节，并为 $\tau_ {\mathrm{T}}$ 提出了一种双重稳健估计量。

## 18.4 R 代码（R code）

实现习题 18.3 中的估计量。

## 18.5 推荐阅读（Recommended reading）

Rosenbaum 和 Rubin (1983a) 以及 Imbens (2003) 是两篇关于敏感性分析的经典论文，但它们涉及更复杂的程序。

<!-- footnote -->

- 我们定义 $\begin{array} { r } { \| v \| _ { 2 } ^ { 2 } = \sum _ { j = 1 } ^ { p } v _ { j } ^ { 2 } } \end{array}$ 为向量 $\boldsymbol { v } = ( v _ { 1 } , \ldots , v _ { p } ) ^ { \mathsf { T } }$ 的范数。它表示向量 $v$ 长度的平方。

<!-- footnote end -->

<!-- footnote -->

- 这对于我们讨论二元 $Z$ 的情况并不理想，但它简化了推导过程。Ding 和 Miratrix (2015) 使用更适合二元 $Z$ 的模型进行了详细讨论。

<!-- footnote end -->

<!-- footnote -->

- 同样，我们从线性模型中生成连续的 $Z$ 以简化推导。Ding 等人 (2017b) 将该理论扩展到更一般的因果模型，特别是针对二元 $Z$ 。

<!-- footnote end -->

<!-- footnote -->

- 他们最初的分析基于一项病例对照研究，并估计了吸烟对肺癌的优势比。但由于肺癌是一种罕见结局，**风险比（risk ratio）** 接近于优势比。

<!-- footnote end -->

<!-- footnote -->

- 在信息论中，**互信息（mutual information）**
- $I ( A , B ) = \iint p ( a , b ) \log _ { 2 } \frac { p ( a , b ) } { p ( a ) p ( b ) } \mathrm { d } a \mathrm { d } b$
- 衡量两个随机变量 $A$ 和 $B$ 之间的依赖性，其中 $p ( \cdot )$ 表示 $( A , B )$ 的联合或边缘密度。**数据处理不等式（data processing inequality）** 是一个著名结果：如果 $Z \bot \bot Y \mid U$ ，那么 $I ( Z , Y ) \ge I ( Z , U )$ 且 $I ( Z , Y ) \ge I ( U , Y )$ 。Lihua Lei 和 Bin Yu 向我指出了 Cornfield 不等式与数据处理不等式之间的联系。

<!-- footnote end -->

## 19
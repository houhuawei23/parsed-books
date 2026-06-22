# 附录：因果追索（Causal Recourse）
## c.1 证明

## c.1.1 命题 ?? 的证明

**命题 ?? (GP-SCM 噪声后验)**。设 $\{ { \bf x } ^ { i } \} _ { i = 1 } ^ { n }$ 是来自 (??) 的一个观测样本。对于每个 $r \in [ d ]$ 且非空父节点集 $| p a ( r ) | > 0$，噪声向量 $\mathbf { u } _ { r } = \left( u _ { r } ^ { 1 } , . . . , u _ { r } ^ { n } \right)$ 在给定 $ { \mathbf { x } } _ { r } = ( x _ { r } ^ { 1 } , . . . , x _ { r } ^ { n } )$ 和 $\mathbf { X } _ { p a ( r ) } = \bigl ( \mathbf { x } _ { p a ( r ) } ^ { 1 } , . . . , \mathbf { x } _ { p a ( r ) } ^ { n } \bigr )$ 条件下的后验分布由下式给出：

$$
\mathbf {u} _ {r} | \mathbf {X} _ {p a (r)}, \mathbf {x} _ {r} \sim \mathcal {N} \left(\sigma_ {r} ^ {2} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \mathbf {x} _ {r}, \sigma_ {r} ^ {2} \left(\mathbf {I} - \sigma_ {r} ^ {2} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1}\right)\right), \tag {C.1.1}
$$

其中 $\mathbf { K } : = \big ( k _ { r } \big ( \mathbf { x } _ { p a ( r ) } ^ { i } , \mathbf { x } _ { p a ( r ) } ^ { j } \big ) \big ) _ { i j }$ 表示格拉姆矩阵（Gram matrix）。

**证明**。首先，根据定义，${ \bf u } _ { r }$ 在给定 $\mathbf { X } _ { \mathsf { p a } ( r ) }$ 的条件下独立于 $\mathbf { f } _ { r } = ( f _ { r } ( \mathbf { x } _ { \mathsf { p a } ( r ) } ^ { 1 } ) , . . . , f _ { r } ( \mathbf { x } _ { \mathsf { p a } ( r ) } ^ { n } ) )$。此外，根据 (??) 中假设的 GP-SCM 模型、定义 $? ?$ 以及 GP 先验的性质，两者都是多元高斯随机变量，其分布由下式给出：

$$
\mathbf {u} _ {r} \sim \mathcal {N} (\mathbf {0}, \sigma_ {r} ^ {2} \mathbf {I}) \quad \text { 独立于 } \quad \mathbf {X} _ {p a (r)}, \quad \text { 且 } \tag {C.1.1}
$$

$$
\mathbf {f} _ {r} | \mathbf {X} _ {p a (r)} \sim \mathcal {N} (\mathbf {0}, \mathbf {K}), \tag {C.1.2}
$$

其中 $\mathbf {0}$ 表示零向量（或矩阵，见下文），$\mathbf {K}$ 如命题 ?? 中所定义。

由于独立的多元高斯随机变量是联合多元高斯的，因此我们有

$$
\binom {\mathbf {u} _ {r}} {\mathbf {f} _ {r}} \left| \mathbf {X} _ {\mathrm{pa} (r)} \right. \sim \mathcal {N} (\mathbf {0}, \Sigma), \quad \text { 其中 } \quad \Sigma = \left( \begin{array}{c c} \sigma_ {r} ^ {2} \mathbf {I} & \mathbf {0} \\ \mathbf {0} & \mathbf {K} \end{array} \right) \tag {C.1.3}
$$

注意到 ${ \bf x } _ { r } = { \bf f } _ { r } + { \bf u } _ { r }$ 并对 $\left( \mathbf { C . I . 3 } \right)$ 应用线性变换，我们得到

$$
\binom {\mathbf {u} _ {r}} {\mathbf {x} _ {r}} \left| \mathbf {X} _ {\mathrm{pa} (r)} = \left( \begin{array}{c c} \mathbf {I} & \mathbf {0} \\ \mathbf {I} & \mathbf {I} \end{array} \right) \binom {\mathbf {u} _ {r}} {\mathbf {f} _ {r}} \right| \mathbf {X} _ {\mathrm{pa} (r)} \sim \mathcal {N} (\mathbf {0}, \tilde {\boldsymbol {\Sigma}}) \tag {C.1.4}
$$

$$
\text { 其中 } \quad \tilde {\Sigma} = \left( \begin{array}{c c} \sigma_ {r} ^ {2} \mathbf {I} & \sigma_ {r} ^ {2} \mathbf {I} \\ \sigma_ {r} ^ {2} \mathbf {I} & \mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I} \end{array} \right).
$$

对 $\mathbf { x } _ { r }$ 进行条件化，并使用条件化公式（例如，Tou11），可得结果：

$$
\mathbf {u} _ {r} \left| \mathbf {X} _ {p a (r)}, \mathbf {x} _ {r} \right. \sim \mathcal {N} \left(\mathbf {0} + \sigma_ {r} ^ {2} \mathbf {I} \left(\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}\right) ^ {- 1} \left(\mathbf {x} _ {r} - \mathbf {0}\right), \sigma_ {r} ^ {2} \mathbf {I} - \sigma_ {r} ^ {2} \mathbf {I} \left(\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}\right) ^ {- 1} \sigma_ {r} ^ {2} \mathbf {I}\right) \tag {C.1.5}
$$

$$
\sim \mathcal {N} \left(\sigma_ {r} ^ {2} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \mathbf {x} _ {r}, \sigma_ {r} ^ {2} \left(\mathbf {I} - \sigma_ {r} ^ {2} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1}\right)\right) \tag {C.1.6}
$$

## c.1.2 命题 ?? 的证明

**命题 ?? (GP-SCM 反事实分布)**。设 $\{ { \bf x } ^ { i } \} _ { i = 1 } ^ { n }$ 是来自 (??) 的一个观测样本。那么，对于 $r \in [ d ]$ 且 $| p a ( r ) | > 0$，个体 $\mathbf { x } ^ { F } \in \{ \mathbf { x } ^ { i } \} _ { i = 1 } ^ { n }$ 在 $\mathbf { X } _ { p a ( r ) }$ 为 $\tilde { \mathbf { x } } _ { p a ( r ) }$（而非 $\mathbf { x } _ { p a ( r ) } ^ { F }$）的情况下，$X _ { r }$ 的反事实分布由下式给出：

$$
\mathrm{X} _ {r} \left(\mathbf {X} _ {p a (r)} = \tilde {\mathbf {x}} _ {p a (r)}\right) \mid \mathbf {x} ^ {F}, \left\{\mathbf {x} ^ {i} \right\} _ {i = 1} ^ {n} \tag {C.1.7}
$$

$$
\sim \mathcal {N} \big (\mu_ {r} ^ {F} + \tilde {\mathbf {k}} ^ {T} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \mathbf {x} _ {r}, s _ {r} ^ {F} + \tilde {k} - \tilde {\mathbf {k}} ^ {T} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \tilde {\mathbf {k}} \big),
$$

其中 $\tilde { k } : = k _ { r } ( \tilde { \mathbf { x } } _ { p a ( r ) } , \tilde { \mathbf { x } } _ { p a ( r ) } )$，$\tilde { \mathbf { k } } : = \big ( k _ { r } ( \tilde { \mathbf { x } } _ { p a ( r ) } , \mathbf { x } _ { p a ( r ) } ^ { 1 } ) , \dots , k _ { r } ( \tilde { \mathbf { x } } _ { p a ( r ) } , \mathbf { x } _ { p a ( r ) } ^ { n } ) \big )$，$\mathbf {x} _ {r}$ 和 $\mathbf {K}$ 如 $? ?$ 中所定义，而 $\mu _ { r } ^ { F }$ 和 $s _ { r } ^ { F }$ 是由 (??) 给出的 $u _ { r } ^ { F }$ 的后验均值和方差。

**证明**。我们按照**溯因（abduction）**、**行动（action）**和**预测（prediction）**三个步骤来计算反事实分布（更多细节见 § 4.2.2）。从根据下式生成的事实观测 $\mathbf { x } ^ { \mathsf { F } } \in \{ x ^ { i } \} _ { i = 1 } ^ { n }$ 出发：

$$
\mathbf {x} _ {r} ^ {\mathsf {F}} := f _ {r} (\mathbf {x} _ {\mathrm{pa} (r)} ^ {\mathsf {F}}) + u _ {r} ^ {\mathsf {F}}, \tag {C.1.7}
$$

我们首先计算噪声后验（溯因步骤）。根据命题 ??，它由 (??) 的边缘分布给出，即：

$$
u _ {r} ^ {\mathsf {F}} | \mathbf {X} _ {\mathrm{pa} (r)}, \mathbf {x} _ {r} \sim \mathcal {N} (\mu_ {r} ^ {F}, s _ {r} ^ {\mathsf {F}}) \tag {C.1.8}
$$

其中 $\mu _ { r } ^ { \mathsf { F } }$ 是均值向量

$$
\boldsymbol {\mu} _ {r} = \sigma_ {r} ^ {2} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \mathbf {x} _ {r} \tag {C.1.9}
$$

的第 $\mathsf {F}$ 个元素，而 $s _ { r } ^ { \mathsf { F } }$ 是由 (??) 给出的噪声后验协方差矩阵

$$
S _ {r} = \sigma_ {r} ^ {2} \left(\mathbf {I} - \sigma_ {r} ^ {2} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1}\right) \tag {C.1.10}
$$

的第 $(\mathsf {F}, \mathsf {F})$ 个元素。

接下来，我们通过更新结构方程 (C.1.7) 来模拟假设性干预（行动步骤）：

$$
x _ {r} ^ {\mathsf {F}} \left(\mathbf {X} _ {\mathrm{pa} (r)} = \tilde {\mathbf {x}} _ {\mathrm{pa} (r)}\right) := f _ {r} \left(\tilde {x} _ {\mathrm{pa} (r)}\right) + u _ {r} ^ {\mathsf {F}}. \tag {C.1.11}
$$

在新输入 $\tilde { x } _ { \mathrm { p a } ( r ) }$ 处的 GP 预测后验分布为（例如，参见 WR06）：

$$
f _ {r} (\tilde {x} _ {\mathrm{pa} (r)}) | \mathbf {X} _ {\mathrm{pa} (r)}, \mathbf {x} _ {r} \sim \mathcal {N} (\tilde {\mathbf {k}} ^ {T} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \mathbf {x} _ {r}, \tilde {k} - \tilde {\mathbf {k}} ^ {T} (\mathbf {K} + \sigma_ {r} ^ {2} \mathbf {I}) ^ {- 1} \tilde {\mathbf {k}}). \tag {C.1.12}
$$

将 (C.1.12) 和 (C.1.8) 代入 (C.1.11)，并注意到两个高斯分布之和仍然是高斯分布，其均值和方差分别等于两个独立高斯分布的均值之和与方差之和（预测步骤），即完成证明。□

## c.1.3 命题??的证明

**命题??**。在满足因果充分性（causal sufficiency）的条件下，$P \left( \mathbf {X} _ {d ( \mathcal {I} )} \mid \mathrm{do} \left( \mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta}\right), \mathbf {x} _ {nd ( \mathcal {I} )} ^ {F}\right)$ 在观测上是可识别的（即，可通过观测分布计算），其表达式为：

$$
p \left(\mathbf {X} _ {d (\mathcal {I})} \mid \mathrm{do} \left(\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta}\right), \mathbf {x} _ {n d (\mathcal {I})} ^ {F}\right) = \prod_ {r \in d (\mathcal {I})} p \left(X _ {r} \mid \mathbf {X} _ {p a (r)}\right) \Bigg | _ {\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta}, \mathbf {X} _ {n d (\mathcal {I})} = \mathbf {x} _ {n d (\mathcal {I})} ^ {F}}. \tag {C.1.13}
$$

**证明**。这是因果充分（马尔可夫）因果模型性质的直接推论，但为完整性起见，我们在此给出推导过程。回顾一下，$P$ 在其底层因果图上分解如下：

$$
p (\mathbf {X}) = \prod_ {r \in [ d ]} p (X _ {r} | \mathbf {X} _ {\mathrm{pa} (r)}). \tag {C.1.13}
$$

该联合分布在干预 $\mathrm{do} ( \mathbf { X } _ { \mathcal { T } } : = \theta )$ 作用下变换为：

$$
P (\mathbf {X} _ {- \mathcal {I}}, \mathrm{do} (\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta})) = \delta (\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta}) \prod_ {r \in [ d ] \backslash \mathcal {I}} P (X _ {r} | \mathbf {X} _ {\mathrm{pa} (r)}). \tag {C.1.14}
$$

将未受干预的变量划分为**后代变量（descendants）** $\mathbf { d } ( \mathcal { T } )$ 和**非后代变量（non-descendants）** $nd( \mathcal { T } )$，并以后验干预变量 $\mathrm{do} ( \mathbf { X } _ { \mathcal { I } } : = \pmb { \theta } )$ 为条件，我们得到：

$$
P (\mathbf {X} _ {\mathrm{nd} (\mathcal {I})}, \mathbf {X} _ {\mathrm{d} (\mathcal {I})} | \mathrm{do} (\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta})) = \left. \left(\prod_ {r \in \mathrm{nd} (\mathcal {I}) \cup \mathrm{d} (\mathcal {I})} P (X _ {r} | \mathbf {X} _ {\mathrm{pa} (r)})\right) \right| _ {\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta}}. \tag {C.1.15}
$$

由于非后代变量 $\mathbf { \boldsymbol { X } } _ { \mathrm { nd } ( \mathcal { T } ) }$ 根据定义不受干预的影响，我们可以写出：

$$
\begin{array}{l} P (\mathbf {X} _ {\mathrm{nd} (\mathcal {I})}, \mathbf {X} _ {\mathrm{d} (\mathcal {I})} | \mathrm{do} (\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta})) = \\ \left(\prod_ {r \in \mathrm{d} (\mathcal {I})} P (X _ {r} | \mathbf {X} _ {\mathrm{pa} (r)})\right) \Bigg | \mathbf {x} _ {\mathcal {I} := \boldsymbol {\theta}} \prod_ {r \in \mathrm{nd} (\mathcal {I})} P (X _ {r} | \mathbf {X} _ {\mathrm{pa} (r)}). \\ \end{array}
$$

因此，我们可以以 $\mathbf { X } _ { \mathrm { n d } ( \mathcal { I } ) }$ 的特定值为条件，得到：

$$
\begin{array}{l} P \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} | \mathrm{do} (\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta}), \mathbf {X} _ {\mathrm{nd} (\mathcal {I})} = \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) = \\ \left(\prod_ {r \in \mathrm{d} (\mathcal {I})} P (X _ {r} | \mathbf {X} _ {p a (r)})\right) \bigg | _ {\mathbf {X} _ {\mathcal {I}} := \boldsymbol {\theta}, \mathbf {X} _ {\mathrm{nd} (\mathcal {I})} = \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}} \tag {C.1.16} \\ \end{array}
$$

![image_31](images/image_31.png)

## c.2 附加结果（Additional Results）

本节展示了补充第??节结果的其他实验。表 C.1 呈现了与表??相对应的结果，其中使用了附录 C.5 开头讨论的**暴力搜索方法（brute-force approach）** 替代基于梯度的优化。在此，每个实值特征均在训练数据集中观测值的范围内被离散化为 20 个区间。

图 C.1 与图??的结果相对应，其快照（$\gamma _ { \mathrm { LCB } } = 2.5$）也展示在表??中。此处，我们通过改变 $\gamma _ { \mathrm { LCB } }$ 的值来展示**有效性（validity）** 与**成本（cost）** 之间的权衡，使用的训练分类器分别为 (a) 中的非线性多层感知机（Multilayer Perceptron, MLP）和 (b) 中的不可微随机森林分类器。请注意，对于后者，优化只能通过暴力搜索方法进行。所有这些附加结果基本上都印证了正文中提出的见解。

最后，表 C.2 从干预目标选择的角度，对所提出的**逆推方法（recourse approaches）** 与**基准（baselines）** 和**理想方法（oracles）** 进行了定性比较。我们在三个合成数据集上通过实验表明，**条件平均处理效应（Conditional Average Treatment Effect, CATE）** 方法具有更可预测的行为，因为它们对模型假设的敏感性较低，因此对于在因果知识不完善情况下寻求逆推的个体而言更为可取。

## c.3 不同假设下结构因果模型的可识别性（(Non-)Identifiability of SCMs under Different Assumptions）

在一般形式下，即在不对方程 $S$ 或**噪声分布（noise distribution）** $P _ { \mathbf { U } }$ 做任何进一步假设的情况下，**结构因果模型（Structural Causal Models, SCMs）** 无法仅从数据中识别，这意味着存在多个不同的 SCM（可能具有不同的底层因果图）却蕴含相同的观测分布 (PJS17)。一种可能的构造依赖于**逆累积分布函数（inverse cumulative distribution function, cdf）** 与均匀分布随机变量的结合使用 (Dar51)，该构造也被用于非线性**独立成分分析（Independent Component Analysis, ICA）** 的不可识别性证明中 (HP99)。即使已知因果图，通常也是不够的，如下述命题所总结。

**表 C.1：不同三变量 SCM 上暴力搜索方法（20 区间离散化）的实验结果。我们展示了在 $N _ { \mathrm { runs } } = 100$、$N _ { \mathrm { MC-samples } } = 100$ 且 $\gamma _ { \mathrm { LCB } } = 2$ 条件下的平均性能。相对趋势反映了表??中的结果。**

| 方法 | 线性 SCM |  |  | 非线性 ANM |  |  | 非加性 SCM |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | Valid $_{\star}$ (%) | LCB | Cost (%) | Valid $_{\star}$ (%) | LCB | Cost (%) | Valid $_{\star}$ (%) | LCB | Cost (%) |
| $\mathcal{M}_{\star}$ | 100 | - | 11.0±5.6 | 100 | - | 20.7±11.0 | 100 | - | 15.8±8.9 |
| $\mathcal{M}_{\text{LIN}}$ | 100 | - | 11.3±5.8 | 60 | - | 19.9±8.9 | 92 | - | 17.0±10.4 |
| $\mathcal{M}_{\text{KR}}$ | 95 | - | 11.2±5.6 | 88 | - | 20.5±10.7 | 47 | - | 15.8±10.6 |
| $\mathcal{M}_{\text{GP}}$ | 100 | .55±.04 | 11.6±5.8 | 99 | .55±.04 | 21.2±10.9 | 88 | .58±.05 | 16.8±10.3 |
| $\mathcal{M}_{\text{CVAE}}$ | 100 | .55±.04 | 11.5±5.8 | 95 | .55±.03 | 21.7±10.7 | 95 | .59±.07 | 16.9±10.3 |
| $\text{CATE}_{\star}$ | 90 | .57±.07 | 11.0±5.5 | 95 | .55±.05 | 22.8±10.8 | 99 | .57±.06 | 16.2±8.9 |
| $\text{CATE}_{\text{GP}}$ | 92 | .56±.07 | 11.2±5.5 | 95 | .55±.04 | 22.8±10.9 | 85 | .58±.07 | 16.4±10.5 |
| $\text{CATE}_{\text{CVAE}}$ | 90 | .57±.06 | 11.1±5.4 | 96 | .55±.03 | 23.0±10.8 | 94 | .59±.07 | 16.8±10.2 |

**命题 C.3.1**。即使已知因果图，仅凭条件分布 $P ( X _ { r } | \mathbf { X } _ { pa ( r ) } )$ 也不足以唯一确定结构方程 $X _ { r } : = \dot { f } _ { r } ( \dot { \mathbf { X } } _ { pa ( r ) } , U _ { r } )$，除非有进一步的假设。

**证明**。这可以通过使用来自 $\mathrm{JS10}$ 脚注 1 的以下论证来证明（已调整为我们的符号表示）：

$$
\begin{array}{l} \begin{array}{l} \text { "让 U_{r} 由（可能不可数多个）实值随机变量组成} \\ U _ {r} [ \mathbf {x} _ {pa (r)} ], \text { 每个值 x_{pa(r)} 对应父节点 X_{pa(r)} 的一个值。让 U _ {r} [ \mathbf {x} _ {pa (r)} ] \\ \text { 服从分布 } P _ {X _ {r} | \mathbf {x} _ {pa (r)}} \text {，并定义 } f _ {r} (\mathbf {x} _ {pa (r)}, U _ {r}) := U _ {r} [ \mathbf {x} _ {pa (r)} ]. \text { 那么 } \end{array} \\ X _ {r} | \mathbf {X} _ {pa (r)} \text { 的分布就是 } P _ {X _ {r} | \mathbf {X} _ {pa (r)}}. \\ \end{array}
$$

现在，我们可以基于此构造来构建第二个具有相同观测分布和因果图的 SCM，例如，通过将噪声变量和结构方程移动某个固定常数 $C$，如下所示。

对于 $r \in [ d ]$，定义 $Y _ { r } : = X _ { r } - C$。让 $\tilde { U } _ { r }$ 由（可能不可数多个）实值随机变量 $\tilde { U } _ { r } [ { \bf { x } } _ { \mathrm { { p a } } ( r ) } ]$ 组成，每个值 $\mathbf { x } _ { \mathrm { p a } ( r ) }$ 对应父节点 $\mathbf { X } _ { \mathrm { p a } ( r ) }$ 的一个值。让 $\tilde { U } _ { r } [ { \bf { x } } _ { \mathrm { { p a } } ( r ) } ]$ 服从分布 $P _ { Y _ { r } | \mathbf { x } _ { \mathrm { p a } ( r ) } }$，并定义 $f _ { r } ( \mathbf { x } _ { \mathrm { p a } ( r ) } , \tilde { U } _ { r } ) : = \tilde { U } _ { r } [ \mathbf { x } _ { \mathrm { p a } ( r ) } ] + C$。那么 $X _ { r } | \mathbf { X } _ { \mathrm { p a } ( r ) }$ 也具有分布 $P _ { X _ { r } | \mathbf { X } _ { pa ( r ) } }$，但对于 $C \neq 0$，其结构方程和噪声分布与之前的构造不同。□

在来自 (??) 的 **CVAE-SCM 模型（cvae-SCM model）** 的情况下，设定比上述情况稍微不那么一般，因为我们额外假设：(i) 噪声分布是固定维度的各向同性多元高斯分布，$\mathbf { z } _ { r } \sim \mathcal { N } _ { d _ { \mathbf { z } _ { r } } } ( \mathbf { 0 } , \mathbf { I } )$；(ii) 结构方程 $D _ { r }$ 属于可通过具有固定宽度和深度的前馈神经网络（feedforward neural networks）表示的函数类，其参数 $\psi _ { r }$ 是可学习的。

遗憾的是，我们尚未发现针对这种特定设定的可识别性结果，对此问题的进一步研究超出了当前工作的范围。然而，有趣的是，来自 (??) 的 CVAE-SCM 可以被理解为 $\mathrm{(PB_{14})}$ 所考虑的**等误差方差线性高斯模型（linear Gaussian model with equal error variances）** 的非线性扩展，而该模型已被证明是可识别的。

总的来说，似乎很少有工作涉及非线性情况下 SCM 的可识别性；我们建议参考 $\mathrm{PJ5\pi_{I7}, \ S7. \pi_{1}}$ 以获取现有结果的概述。与我们设定特别相关的是 (ZH09) 的**后非线性模型（post-nonlinear model）**，它指的是在 ANM 之上应用非线性函数 $g$ 的设定，即 $X _ { r } : = g _ { r } ( f _ { r } ( \mathbf { \tilde { X } } _ { \mathfrak { p a } ( r ) } ) + U _ { r } )$，并且已经提供了关于 $\left\{ f _ { r } , g _ { r } \right\}$ 的完整条件以实现可识别性。考虑到解码器 $D _ { r }$ 的形式——即具有堆叠层的前馈神经网络，这些层对前一层输出的线性变换应用简单的非线性函数——来自 (??) 的 CVAE-SCM 有可能被解释为一个**嵌套的后非线性模型（nested post-nonlinear model）**。我们认为这是一个有趣的方向，但将对此问题的进一步研究留待未来工作。

## c.4 CVAE 训练的更多细节（Further Details on CVAE Training）

为了学习 CVAE 潜变量模型，我们执行**摊销变分推断（amortised variational inference）**，其近似后验 $q$ 由编码器 $E _ { r }$（参数为 $\phi _ { r }$ 的神经网络形式）参数化：

$$
p _ {\psi_ {r}} \left(\mathbf {z} _ {r} \mid x _ {r}, \mathbf {x} _ {\mathrm{pa} (r)}\right) \approx q _ {\phi_ {r}} \left(\mathbf {z} _ {r} \mid x _ {r}, \mathbf {x} _ {\mathrm{pa} (r)}\right) := \mathcal {N} \left(\hat {\mu} _ {r}, \hat {\sigma} _ {r} ^ {2}\right), \tag {C.4.1}
$$

$$
(\hat {\mu} _ {r}, \hat {\sigma} _ {r} ^ {2}) := E _ {r} (x _ {r}, \mathbf {x} _ {\mathrm{pa} (r)}; \phi_ {r}).
$$

给定数据 $\{ \mathbf { x } ^ { i } \} _ { i = 1 } ^ { n }$ 的**证据下界（Evidence Lower Bound, ELBO）** 形式的训练目标为：

$$
\begin{array}{l} \mathcal {L} _ {r} \left(\psi_ {r}, \phi_ {r}\right) = \sum_ {i = 1} ^ {n} \mathbb {E} _ {q _ {\phi_ {r}} \left(\mathbf {z} \mid x _ {r} ^ {i}, \mathbf {x} _ {\mathrm{pa} (r)} ^ {i}\right)} \left[ \left\| x _ {r} ^ {i} - D _ {r} \left(\mathbf {x} _ {\mathrm{pa} (r)} ^ {i}, \mathbf {z}; \psi_ {r}\right) \right\| ^ {2} \right] \tag {C.4.2} \\ + \beta_ {r} D _ {\mathrm{KL}} \left(\left. q _ {\phi_ {r}} (\mathbf {z} | x _ {r} ^ {i}, \mathbf {x} _ {\mathrm{pa} (r)} ^ {i}) \right| \mid p (z)\right) \\ \end{array}
$$

我们通过随机梯度下降法同时学习 $\psi _ { r }$ 和 $\phi _ { r }$，优化目标为 $\mathcal { L } _ { \boldsymbol { r } }$，梯度通过从 $q _ { \phi _ { \uparrow } }$ 中进行蒙特卡洛采样并结合重参数化来计算。由于不同 $r$ 对应的编码器和解码器参数对 $\left( \psi _ { r } , \phi _ { r } \right)$ 是独立的，因此可以并行执行训练。

## c.4.1 CVAE 训练的超参数选择（Hyperparameter Selection for CVAE Training）

为每个 $\mathbf { X } _ { r } | \mathbf { X } _ { \mathsf { p a } ( r ) }$ 关系训练了一个 CVAE 模型。通常，通过比较数据集中真实样本的分布与从训练好的 CVAE 中通过从先验采样噪声得到的重构样本的分布来选择超参数。超参数的选择要么手动进行，要么通过对各种编码器和解码器架构、潜在空间维度以及 CVAE 目标函数 (C.4.2) 中权衡 MSE 和 KL 项的超参数 $\beta _ { r }$ 的值执行网格搜索来完成。在自动选择的情况下，选择导致真实样本与重构样本之间**最大均值差异（Maximum Mean Discrepancy, MMD）** 统计量 (Gre+12) 最小的配置作为超参数配置。关于所考虑的搜索空间和选定值的更多细节见表 $C.3$。

## c.5 实验细节、超参数选择及 SCM 规范（Experimental Details, Hyperparameter Choices, and Specification of SCMs）

## C.5.1 实验中使用的SCM规范（Specification of SCMs used in our experiments）

以下是对我们在合成数据和半合成数据实验中所使用的所有**结构因果模型（Structural Causal Models, SCMs）**的规范，这些模型既用于数据生成，也用于通过在真实SCM中计算相应的反事实来评估不同方法提出的补救措施的有效性。

此外，我们还指定了用于生成训练标签的模型。但请注意，这些标签仅用于从头训练一个新的分类器（例如，逻辑回归、多层感知机或随机森林）：这就是主章节中提到的 $h(x)$。因此，标签生成过程仅用于获取标签以训练分类器，随后被忽略，转而使用 $h$。

在选择结构方程和标签生成过程时，我们试图选择能够产生大致中心化特征以及大致平衡的数据集（即正负训练样本比例相似）的组合，并且这些数据集并非完全线性可分（即存在一定的类别重叠）。此外，我们试图选择能够使**神谕（oracle）**为不同事实实例选择多样化干预目标的设置，即我们尽量避免最优动作总是干预同一组变量的情况。为了引发更有趣的行为，我们从**高斯混合模型（mixtures of Gaussians）**中对根节点进行采样。

## C.5.1.1 用于表??的三变量合成SCM（3-variable synthetic SCMs used for Table ??）

图C.2提供了用于表??的三变量合成SCM的视觉摘要。

**线性SCM（linear scm）**：线性三变量SCM由以下结构方程和噪声分布组成：

$$
X _ {1} := U _ {1}, \quad U _ {1} \sim \operatorname{MoG} \left(0. 5 \mathcal {N} (- 2, 1. 5) + 0. 5 \mathcal {N} (1, 1)\right) \tag {C.5.1}
$$

$$
X _ {2} := - X _ {1} + U _ {2}, \quad U _ {2} \sim \mathcal {N} (0, 1) \tag {C.5.2}
$$

$$
X _ {3} := 0. 0 5 X _ {1} + 0. 2 5 X _ {2} + U _ {3}, \quad U _ {3} \sim \mathcal {N} (0, 1) \tag {C.5.3}
$$

![image_32](images/image_32.png)

图C.2：合成三变量SCM的成对特征关系直方图和散点图。

**非线性ANM（non-linear anm）**：非线性三变量**加性噪声模型（Additive Noise Model, ANM）**由以下结构方程和噪声分布组成：

$$
X _ {1} := U _ {1}, \quad U _ {1} \sim \operatorname{MoG} \left(0. 5 \mathcal {N} (- 2, 1. 5) + 0. 5 \mathcal {N} (1, 1)\right) \tag {C.5.4}
$$

$$
X _ {2} := - 1 + \frac {3}{1 + e ^ {- 2 X _ {1}}} + U _ {2}, \quad U _ {2} \sim \mathcal {N} (0, 0. 1) \tag {C.5.5}
$$

$$
X _ {3} := - 0. 0 5 X _ {1} + 0. 2 5 X _ {2} ^ {2} + U _ {3}, \quad U _ {3} \sim \mathcal {N} (0, 1) \tag {C.5.6}
$$

**非加性SCM（non-additve scm）**：非加性三变量SCM由以下结构方程和噪声分布组成：

$$
X _ {1} := U _ {1}, \quad U _ {1} \sim \operatorname{MoG} \left(0. 5 \mathcal {N} (- 2. 5, 1) + 0. 5 \mathcal {N} (2. 5, 1)\right) \tag {C.5.7}
$$

$$
X _ {2} := 0. 2 5 \operatorname{sgn} (U _ {2}) X _ {1} ^ {2} (1 + U _ {2} ^ {2}), \quad U _ {2} \sim \mathcal {N} (0, 0. 2 5) \tag {C.5.8}
$$

$$
X _ {3} := - 1 + 0. 1 \operatorname{sgn} (U _ {3}) (X _ {1} ^ {2} + X _ {2} ^ {2}) + U _ {3}, \quad U _ {3} \sim \mathcal {N} (0, 0. 2 5 ^ {2}) \tag {C.5.9}
$$

**标签生成（label generation）**：对于所有三变量SCM，标签 $Y$ 根据下式采样：

$$
Y \sim \text { Bernoulli } \left(\left(1 + e ^ {- 2. 5 \rho^ {- 1} (X _ {1} + X _ {2} + X _ {3})}\right) ^ {- 1}\right) \tag {C.5.10}
$$

其中 $\rho$ 是所有训练样本中 $\left( X _ { 1 } + X _ { 2 } + X _ { 3 } \right)$ 的平均值。

## C.5.1.2 用于表??的七变量半合成贷款审批SCM（7-variable semi-synthetic loan approval SCM used for Table ??）

对于半合成数据集，我们希望捕捉所涉变量之间一些我们认为直观的关系，并在有限程度上反映现实世界中的贷款审批场景：

*   贷款金额和期限对于可能想要建房和组建家庭的中年人来说最大，而对于年轻人和老年人来说较小；
*   由于每月可负担的还款额存在上限，贷款期限随贷款金额增加而增加；
*   一旦收入超过某个（最低生活）阈值，储蓄会随之增加；
*   收入随年龄增长而增加；
*   受教育程度最初随年龄增长而增加，最终趋于饱和；
*   由于人口中存在的性别歧视和机会不平等，收入和教育（机会）存在性别差异；

七变量半合成贷款SCM的视觉摘要如图C.3所示。

![image_33](images/image_33.png)

图C.3：半合成贷款SCM的成对特征关系直方图和散点图。

**半合成SCM（semi-synthetic scm）**：贷款审批SCM由以下结构方程和噪声分布组成：

$$
G := U _ {G}, \quad U _ {G} \sim \text { Bernoulli } (0. 5) \tag {C.5.11}
$$

$$
A := - 3 5 + U _ {A}, \quad U _ {A} \sim \text { Gamma } (1 0, 3. 5) \tag {C.5.12}
$$

$$
E := - 0. 5 + \left(1 + e ^ {- \left(- 1 + 0. 5 G + \left(1 + e ^ {- 0. 1 A}\right) ^ {- 1} + U _ {E}\right)}\right) ^ {- 1}, \quad U _ {E} \sim \mathcal {N} (0, 0. 2 5) \tag {C.5.13}
$$

$$
L := 1 + 0. 0 1 (A - 5) (5 - A) + G + U _ {L}, \quad U _ {L} \sim \mathcal {N} (0, 4) \tag {C.5.14}
$$

$$
D := - 1 + 0. 1 A + 2 G + L + U _ {D}, \quad U _ {D} \sim \mathcal {N} (0, 9) \tag {C.5.15}
$$

$$
I := - 4 + 0. 1 (A + 3 5) + 2 G + G E + U _ {I}, \quad U _ {I} \sim \mathcal {N} (0, 4) \tag {C.5.16}
$$

$$
S := - 4 + 1. 5 \mathbb {I} _ {\{I > 0 \}} I + U _ {S}, \quad U _ {S} \sim \mathcal {N} (0, 2 5) \tag {C.5.17}
$$

请注意，上述SCM中的变量通常具有相对于均值的含义，例如，我们将服从伽马分布的年龄以其均值35为中心，因此 $A$ 具有“与均值 $35$ 的年龄差”的含义（其他变量类似）。

**标签生成（label generation）**：标签 $Y$ 根据下式采样：

$$
Y \sim \text { Bernoulli } \left(\left(1 + e ^ {- 0. 3 (- L - D + I + S + I S)}\right) ^ {- 1}\right). \tag {C.5.18}
$$

请注意，此标签生成过程仅依赖于贷款期限和金额、收入以及储蓄，而不依赖于性别、年龄或教育水平。

## C.6 方差梯度的蒙特卡洛估计量推导（Derivation of a Monte-Carlo estimator for the gradient of the variance）

我们现在推导一个估计量，用于计算 $h$ 在 $\mathbf { X } _ { \mathrm { d } ( \mathcal { I } ) }$ 的干预分布或反事实分布上方差平方根（即标准差）关于 $\theta$ 的梯度，该梯度出现在优化约束/正则化项的阈值 $\text{tresh}(a)$ 中（乘以 $\lambda _ { \mathrm { L C B } }$）。

首先，我们使用微分的链式法则写出：

$$
\nabla_ {\boldsymbol {\theta}} \sqrt {\mathbb {V} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} , \boldsymbol {\theta} , \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) \right]} = \frac {\nabla_ {\boldsymbol {\theta}} \mathbb {V} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} , \boldsymbol {\theta} , \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) \right]}{2 \sqrt {\mathbb {V} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} , \boldsymbol {\theta} , \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) \right]}} (C. 6. 1)
$$

接下来，我们将方差写成期望形式，并假设 $\mathbf { X } _ { \mathrm { d } ( \mathcal { I } ) }$ 的干预分布或反事实分布允许重参数化（正如本章中使用的 GP-SCM 和 CVAE 模型的情况），使用重参数化技巧通过期望算子进行微分，如 (??) 所示。

$$
\begin{array}{l} \nabla_ {\boldsymbol {\theta}} \mathbb {V} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ h \big (\mathbf {X} _ {\mathrm{d} (\mathcal {I})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}} \big) \right] \\ = \nabla_ {\boldsymbol {\theta}} \mathbb {E} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ \left(h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) - \mathbb {E} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})} ^ {\prime}} \left[ h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} ^ {\prime}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) \right]\right) ^ {2} \right] \\ = \nabla_ {\boldsymbol {\theta}} \mathbb {E} _ {\mathbf {z} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ \left(h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} (\mathbf {z}; \boldsymbol {\theta}), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) - \mathbb {E} _ {\mathbf {z} ^ {\prime} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} (\mathbf {z} ^ {\prime}; \boldsymbol {\theta}), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) \right]\right) ^ {2} \right] \\ = \mathbb {E} _ {\mathbf {z} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ \nabla_ {\boldsymbol {\theta}} \Big (h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} (\mathbf {z}; \boldsymbol {\theta}), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) - \mathbb {E} _ {\mathbf {z} ^ {\prime} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} (\mathbf {z} ^ {\prime}; \boldsymbol {\theta}), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) \right] \Big) ^ {2} \right] \\ = \mathbb {E} _ {\mathbf {z} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ \right. 2 \left(h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} (\mathbf {z}; \boldsymbol {\theta}), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) - \mathbb {E} _ {\mathbf {z} ^ {\prime} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} \left(\mathbf {z} ^ {\prime}; \boldsymbol {\theta}\right), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right)\right]\right) \\ \left. \times \left(\nabla_ {\boldsymbol {\theta}} h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})} (\mathbf {z}; \boldsymbol {\theta}), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) - \mathbb {E} _ {\mathbf {z} ^ {\prime} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ \nabla_ {\boldsymbol {\theta}} h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} \left(\mathbf {z} ^ {\prime}; \boldsymbol {\theta}\right), \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) \right]\right)\right) \Bigg ] \tag {C.6.2} \\ \end{array}
$$

现在，我们可以通过两组独立的 $\mathbf { X } _ { \mathrm { d } ( \mathcal { T } ) }$ 的**蒙特卡洛（Monte Carlo）**样本获得梯度的估计值，这些样本通过重参数化从干预分布或反事实分布中抽取：

$$
\left\{\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m)} := \mathbf {x} _ {\mathrm{d} (\mathcal {I})} \left(\mathbf {z} ^ {(m)}; \boldsymbol {\theta}\right) \right\} _ {m = 1} ^ {M}, \quad \left\{\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m ^ {\prime})} := \mathbf {x} _ {\mathrm{d} (\mathcal {I})} \left(\mathbf {z} ^ {(m ^ {\prime})}; \boldsymbol {\theta}\right) \right\} _ {m ^ {\prime} = 1} ^ {M ^ {\prime}} \tag {C.6.3}
$$

其中 $\mathbf { z } ^ { ( m ) } , \mathbf { z } ^ { ( m ^ { \prime } ) } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$。

这便得到了方差的蒙特卡洛梯度估计量如下：

$$
\begin{array}{l} \nabla_ {\boldsymbol {\theta}} \mathbb {V} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ h \big (\mathbf {X} _ {\mathrm{d} (\mathcal {I})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}} \big) \right] \approx \\ \frac {1}{M} \sum_ {m = 1} ^ {M} \left[ 2 \left(h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m)}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) - \frac {1}{M ^ {\prime}} \sum_ {m ^ {\prime} = 1} ^ {M} h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m ^ {\prime})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right)\right) \times \right. \\ \left. \left(\nabla_ {\boldsymbol {\theta}} h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m)}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) - \frac {1}{M ^ {\prime}} \sum_ {m ^ {\prime} = 1} ^ {M ^ {\prime}} \nabla_ {\boldsymbol {\theta}} h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m ^ {\prime})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right)\right) \right] \tag {C.6.4} \\ \end{array}
$$

将上述表达式与以下（未微分）方差的蒙特卡洛估计量一起代入 (C.6.1) 中，即得 $h$ 的标准差梯度的所需估计量：

$$
\begin{array}{l} \mathbb {V} _ {\mathbf {X} _ {\mathrm{d} (\mathcal {I})}} \left[ h \left(\mathbf {X} _ {\mathrm{d} (\mathcal {I})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathsf {F}}\right) \right] \\ \approx \frac {1}{M - 1} \sum_ {m = 1} ^ {M} \left(h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m)}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right) - \frac {1}{M} \sum_ {m ^ {\prime} = 1} ^ {M ^ {\prime}} h \left(\mathbf {x} _ {\mathrm{d} (\mathcal {I})} ^ {(m ^ {\prime})}, \boldsymbol {\theta}, \mathbf {x} _ {\mathrm{nd} (\mathcal {I})} ^ {\mathrm{F}}\right)\right) ^ {2}, \tag {C.6.5} \\ \end{array}
$$

**表 C.2：** 在不同 $3$ 变量**结构因果模型（Structural Causal Models, SCMs）**上（从上到下：线性 SCM、非线性 ANM、非加性 SCM）使用梯度下降方法的实验结果。我们展示了在 $N_{\mathrm{runs}} = 100$、$N_{\mathrm{MC-samples}} = 100$ 和 $\gamma_{\mathrm{LCB}} = 2$ 下的平均性能，并显示了每种补救类型在所有变量子集上执行干预的次数（在 $N_{\mathrm{runs}}$ 中）。最右侧两列显示了每种补救类型的干预集与**预言机（oracle）**方法 $\mathcal{M}_{\star}$ 和 $\mathbf{CATE}_{\star}$ 建议一致的次数。我们观察到，基于子群体的预言机提出的干预措施通常与在个体层面提出的干预措施不同，图 ?? 可以直观地解释这一点。重要的是，我们观察到所有 cate 方法在选择干预变量方面普遍一致。相比之下，我们观察到基于个体的方法在选择用于补救的干预变量时偏离了其预言机（即 $M_{\star}$）。这一结果进一步表明，本文提出的 cate 方法表现出更可预测的行为，因为它们对模型假设不太敏感，因此在因果知识不完善的情况下，对于寻求补救的个体而言更可取。

| Method | SCM | | | INTERVENTION SET | | | | | | IDENTICAL INT. SET |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | Valid $_*$ (%) | LCB | Cost (%) | $\{X_1\}$ | $\{X_2\}$ | $\{X_3\}$ | $\{X_1,X_2\}$ | $\{X_1,X_3\}$ | $\{X_2,X_3\}$ | $\{X_1,X_2,X_3\}$ | $\mathcal{M}_*$ | CATE $_*$ |
| $\mathcal{M}_*$ | 100 | - | 10.9±7.9 | 0 | 25 | 0 | 56 | 0 | 0 | 19 | 100 | 23 |
| $\mathcal{M}_{\text{LIN}}$ | 100 | - | 11.0±7.0 | 0 | 26 | 0 | 50 | 0 | 1 | 23 | 52 | 23 |
| $\mathcal{M}_{\text{KR}}$ | 90 | - | 10.7±6.5 | 0 | 22 | 0 | 44 | 0 | 0 | 34 | 54 | 27 |
| $\mathcal{M}_{\text{GP}}$ | 100 | .55±.04 | 12.2±8.3 | 0 | 6 | 0 | 13 | 0 | 7 | 74 | 25 | 61 |
| $\mathcal{M}_{\text{CVAE}}$ | 100 | .55±.07 | 11.8±7.7 | 0 | 12 | 0 | 25 | 0 | 5 | 58 | 31 | 57 |
| CATE $_*$ | 90 | .56±.07 | 11.9±9.2 | 0 | 6 | 0 | 11 | 0 | 13 | 70 | 23 | 100 |
| CATE $_{\text{GP}}$ | 93 | .56±.05 | 12.2±8.4 | 0 | 3 | 0 | 9 | 1 | 15 | 72 | 18 | 76 |
| CATE $_{\text{CVAE}}$ | 89 | .56±.08 | 12.1±8.9 | 0 | 6 | 1 | 11 | 0 | 16 | 66 | 18 | 78 |
| $\mathcal{M}_*$ | 100 | - | 20.1±12.3 | 70 | 0 | 0 | 2 | 16 | 0 | 11 | 99 | 17 |
| $\mathcal{M}_{\text{LIN}}$ | 54 | - | 20.6±11.0 | 13 | 0 | 0 | 0 | 81 | 0 | 5 | 20 | 41 |
| $\mathcal{M}_{\text{KR}}$ | 91 | - | 20.6±12.5 | 65 | 0 | 0 | 1 | 23 | 0 | 10 | 76 | 22 |
| $\mathcal{M}_{\text{GP}}$ | 100 | .54±.03 | 21.9±12.9 | 39 | 0 | 0 | 0 | 38 | 0 | 22 | 54 | 38 |
| $\mathcal{M}_{\text{CVAE}}$ | 97 | .54±.05 | 22.6±12.3 | 33 | 0 | 0 | 0 | 51 | 0 | 15 | 45 | 42 |
| CATE $_*$ | 97 | .55±.05 | 26.3±21.4 | 4 | 0 | 0 | 0 | 44 | 2 | 49 | 17 | 99 |
| CATE $_{\text{GP}}$ | 94 | .55±.06 | 25.0±14.8 | 4 | 1 | 0 | 0 | 37 | 4 | 53 | 11 | 69 |
| CATE $_{\text{CVAE}}$ | 98 | .54±.05 | 26.0±14.3 | 3 | 0 | 0 | 1 | 32 | 1 | 62 | 12 | 70 |
| $\mathcal{M}_*$ | 100 | - | 13.2±11.0 | 0 | 0 | 1 | 0 | 11 | 78 | 7 | 97 | 78 |
| $\mathcal{M}_{\text{LIN}}$ | 98 | - | 14.0±13.5 | 0 | 0 | 0 | 1 | 0 | 85 | 11 | 81 | 77 |
| $\mathcal{M}_{\text{KR}}$ | 70 | - | 13.2±11.6 | 0 | 17 | 0 | 4 | 10 | 59 | 7 | 55 | 53 |
| $\mathcal{M}_{\text{GP}}$ | 95 | .52±.04 | 13.4±12.8 | 3 | 1 | 2 | 0 | 0 | 82 | 9 | 73 | 78 |
| $\mathcal{M}_{\text{CVAE}}$ | 95 | .51±.01 | 13.4±12.2 | 0 | 3 | 1 | 5 | 2 | 71 | 15 | 72 | 76 |
| CATE $_*$ | 100 | .52±.02 | 13.5±13.0 | 0 | 0 | 2 | 0 | 9 | 77 | 9 | 78 | 97 |
| CATE $_{\text{GP}}$ | 94 | .52±.03 | 13.2±13.1 | 3 | 1 | 5 | 0 | 3 | 73 | 12 | 70 | 76 |
| CATE $_{\text{CVAE}}$ | 100 | .52±.05 | 13.6±12.9 | 0 | 1 | 2 | 0 | 1 | 82 | 11 | 78 | 78 |

**表 C.3：CVAE 训练的超参数选择**是通过手动（对于线性 SCM、非线性 ANM、非加性 SCM）或自动（对于 7 变量半合成贷款审批）的方式，选择使真实样本与重构样本之间的 **最大均值差异（Maximum Mean Discrepancy, MMD）** 统计量最小的设置来完成的。

| SCM | | Conditional | Encoder Arch. | Decoder Arch. | Latent Dim. | $\lambda_{\text{KLD}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Linear SCM | | $X_2|X_1,$ | $1\times32\times32\times32$ | $5\times5\times1$ | 1 | 0.01 |
| | | $X_3|X_1,X_2$ | $1\times32\times32\times32$ | $32\times32\times32\times1$ | 1 | 0.01 |
| Non-linear ANM | | $X_2|X_1,$ | $1\times32\times32$ | $32\times32\times1$ | 5 | 0.01 |
| | | $X_3|X_1,X_2$ | $1\times32\times32\times32$ | $32\times32\times1$ | 1 | 0.01 |
| Non-additive SCM | | $X_2|X_1,$ | $1\times32\times32\times32$ | $32\times32\times1$ | 3 | 0.5 |
| | | $X_3|X_1,X_2$ | $1\times32\times32\times32$ | $5\times5\times1$ | 3 | 0.1 |
| 7-variable semi-synthetic loan approval | any | | | $2\times1$ | | |
| | | | $1\times3\times3$ | $2\times2\times1$ | | 5, 1, 0.5, 0.1, |
| | | | $1\times5\times5$ | $3\times3\times1$ | $1,2$ | 0.05, 0.01, |
| | | | $1\times3\times3\times3$ | $5\times5\times1$ | | 0.005 |
| | | | | $3\times3\times3\times1$ | | |
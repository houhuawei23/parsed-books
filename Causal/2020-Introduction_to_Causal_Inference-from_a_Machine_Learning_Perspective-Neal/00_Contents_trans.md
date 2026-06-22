## 目录

- **[1 动机：你为何关心](./en_trans/01_Motivation__Why_You_Might_Care_trans.md)**
  - 辛普森悖论（Simpson's Paradox）
  - 因果推断（Causal Inference）的应用
  - 相关不蕴含因果（Correlation Does Not Imply Causation）
    - 尼古拉斯·凯奇与泳池溺水
    - 为何关联不是因果？
  - 主要主题

- **[2 潜在结果（Potential Outcomes）](./en_trans/02_Potential_Outcomes_trans.md)**
  - 潜在结果与个体处理效应（Individual Treatment Effects）
  - 因果推断的根本问题
  - 绕过根本问题
    - 平均处理效应（Average Treatment Effects）与缺失数据解释
    - 可忽略性（Ignorability）与可交换性（Exchangeability）
    - 条件可交换性（Conditional Exchangeability）与无混杂性（Unconfoundedness）
    - 积极性/重叠性（Positivity/Overlap）与外推（Extrapolation）
    - 无干扰性（No interference）、一致性（Consistency）与 SUTVA
    - 整合所有要素
  - 花哨统计术语的去神秘化
  - 包含估计的完整示例

- **[3 图中关联与因果的流动](./en_trans/03_The_Flow_of_Association_and_Causation_in_Graphs_trans.md)**
  - 图术语
  - 贝叶斯网络（Bayesian Networks）
  - 因果图（Causal Graphs）
  - 双节点图与图形构建模块
  - 链（Chains）与叉（Forks）
  - 对撞节点（Colliders）及其后代
  - d-分离（d-separation）
  - 关联与因果的流动

- **[4 因果模型（Causal Models）](./en_trans/04_Causal_Models_trans.md)**
  - do-算子（do-operator）与干预分布（Interventional Distributions）
  - 主要假设：模块性（Modularity）
  - 截断分解（Truncated Factorization）
    - 示例应用与重新审视“关联不是因果”
  - 后门调整（Backdoor Adjustment）
    - 与潜在结果的关系
  - 结构因果模型（Structural Causal Models, SCMs）
    - 结构方程（Structural Equations）
    - 干预（Interventions）
    - 对撞节点偏倚（Collider Bias）及为何不以处理的后代为条件
  - 后门调整的示例应用
    - 玩具示例中的关联与因果
    - 包含估计的完整示例
  - 重新审视假设

- **[5 随机实验（Randomized Experiments）](./en_trans/05_Randomized_Experiments_trans.md)**
  - 可比性（Comparability）与协变量平衡（Covariate Balance）
  - 可交换性（Exchangeability）
  - 无后门路径（No Backdoor Paths）

- **[6 非参数识别（Nonparametric Identification）](./en_trans/06_Nonparametric_Identification_trans.md)**
  - 前门调整（Frontdoor Adjustment）
  - do-演算（do-calculus）
    - 应用：前门调整
  - 从图中确定可识别性（Identifiability）

- **[7 估计（Estimation）](./en_trans/07_Estimation_trans.md)**
  - 预备知识
  - 条件结果建模（Conditional Outcome Modeling, COM）
  - 分组条件结果建模（Grouped Conditional Outcome Modeling, GCOM）
  - 提高数据效率
    - TARNet
    - X-学习器（X-Learner）
  - 倾向得分（Propensity Scores）
  - 逆概率加权（Inverse Probability Weighting, IPW）
  - 双重稳健方法（Doubly Robust Methods）
  - 其他方法
  - 结语
    - 置信区间（Confidence Intervals）
    - 与随机实验的比较

- **[8 未观测混杂（Unobserved Confounding）：边界与敏感性分析](./en_trans/08_Unobserved_Confounding__Bounds_and_Sensitivity_Analysis_trans.md)**
  - 边界（Bounds）
    - 无假设边界（No-Assumptions Bound）
    - 单调处理响应（Monotone Treatment Response）
    - 单调处理选择（Monotone Treatment Selection）
    - 最优处理选择（Optimal Treatment Selection）
  - 敏感性分析（Sensitivity Analysis）
    - 线性设定下的敏感性基础
    - 更一般的设定

- **[9 工具变量（Instrumental Variables）](./en_trans/09_Instrumental_Variables_trans.md)**
  - 什么是工具（Instrument）？
  - ATE 的非参数不可识别性
  - 热身：二元线性设定
  - 连续线性设定
  - 局部 ATE（Local ATE）的非参数识别
    - 带工具的新潜在符号
    - 主分层（Principal Stratification）
    - 局部 ATE
  - ATE 识别的更一般设定

- **[10 双重差分（Difference in Differences）](./en_trans/10_Difference_in_Differences_trans.md)**
  - 预备知识
  - 引入时间
  - 识别
    - 假设
    - 主要结果与证明
  - 主要问题

- **[11 基于观测数据的因果发现（Causal Discovery from Observational Data）](./en_trans/11_Causal_Discovery_from_Observational_Data_trans.md)**
  - 基于独立性的因果发现
    - 假设与定理
    - PC 算法
    - 我们能获得更好的识别吗？
  - 半参数因果发现（Semi-Parametric Causal Discovery）
    - 无参数假设下的不可识别性
    - 线性非高斯噪声（Linear Non-Gaussian Noise）
    - 非线性模型
  - 更多资源

- **[12 基于干预数据的因果发现（Causal Discovery from Interventional Data）](./en_trans/12_Causal_Discovery_from_Interventional_Data_trans.md)**
  - 结构性干预（Structural Interventions）
    - 单节点干预
    - 多节点干预
  - 参数性干预（Parametric Interventions）
    - 即将推出
  - 干预马尔可夫等价（Interventional Markov Equivalence）
    - 即将推出
  - 其他杂项设定
    - 即将推出

- **[13 迁移学习（Transfer Learning）与可迁移性（Transportability）](./en_trans/13_Transfer_Learning_and_Transportability_trans.md)**
  - 迁移学习的因果洞见
    - 即将推出
  - 跨群体的因果效应可迁移性
    - 即将推出

- **[14 反事实（Counterfactuals）与中介（Mediation）](./en_trans/14_Counterfactuals_and_Mediation_trans.md)**
  - 反事实基础
    - 即将推出
  - 重要应用：中介
    - 即将推出

- **[附录](./en_trans/15_Appendix_trans.md)**
  - **A 证明**
    - 第 6.1 节中公式 6.1 的证明
    - 倾向得分定理（7.1）的证明
    - IPW 估计量（7.18）的证明

- **[参考文献](./en_trans/16_Bibliography_trans.md)**

- **[字母索引](./en_trans/17_Alphabetical_Index_trans.md)**

- **图列表**


- 1.1 COVID-27 数据中的辛普森悖论
- 2.1 作为缺失数据问题的因果推断
- 3.1 建模因子的参数指数级数量

- 列表

- 2.1 用于估计 ATE 的 Python 代码 17
- 2.2 使用线性回归系数估计 ATE 的 Python 代码 17
- 4.1 用于估计 ATE 的 Python 代码（未调整对撞节点）46
可以用“等价类”与“商集”的思想来解释假设空间 $\mathcal{H}$ 在数据集 $D$ 上的限制。

具体地，将所有从 $\mathcal{X}$ 到 $\mathcal{Y} = \{-1, +1\}$ 的映射（即 $\mathcal{Y}^{\mathcal{X}}$）中的函数 $f$，按照它们在数据集 $D = \{\pmb{x}_1, \dots, \pmb{x}_m\} \subset \mathcal{X}$ 上的取值分成若干等价类：若两个函数 $f, g$ 满足 $\forall i,\, f(\pmb{x}_i) = g(\pmb{x}_i)$，则称 $f$ 与 $g$ 在 $D$ 上等价，记作 $f \sim_D g$。这样，$\mathcal{Y}^{\mathcal{X}}$ 关于 $D$ 上的等价关系 $\sim_D$ 划分为若干等价类，每个等价类对应一个在 $D$ 上的标记序列。

假设空间 $\mathcal{H}$ 可以看作 $\mathcal{Y}^{\mathcal{X}}$ 的一个子集，$\mathcal{H}$ 在 $D$ 上的“限制”其实就是 $\mathcal{H}$ 被这种等价关系所划分得到的等价类的集合，记作

$$
\mathcal{H}_{|D} = \left\{ (h(\pmb{x}_1), \dots, h(\pmb{x}_m)) \mid  h \in \mathcal{H} \right\} ,
$$

即 $\mathcal{H}_{|D}$ 就是 $\mathcal{H}$ 关于 $D$ 上等价关系 $\sim_D$ 在等价类集合 $\mathcal{H}/{\sim_D}$ 中的像。这也可以看作是在 $\mathcal{H}$ 上取关于 $D$ 的“商”，即“商集”。

> 用等价类/商集的观点：$\mathcal{H}$ 在 $D$ 上的限制，等价于 $\mathcal{H}$ 关于 $D$ 上等价关系 $\sim_D$ 得到的等价类集合。

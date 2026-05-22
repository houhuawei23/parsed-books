import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from scipy.special import comb

# 设置随机种子
np.random.seed(42)

# 1. 生成数据分布：三个簇，两个类别 (+1: 簇0和簇1, -1: 簇2)
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
y = np.where(y == 2, -1, 1)  # 将第2簇设为负类
# 划分一个大的测试集用于近似泛化风险
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.8, random_state=42
)  # 测试集保留240个样本

# 2. 定义两个算法
unstable_learner = DecisionTreeClassifier(max_depth=None, random_state=42)
stable_learner = SVC(kernel="linear", C=1.0, random_state=42)


# 3. 辅助函数：计算经验风险、泛化风险（测试集）
def empirical_risk(learner, X, y):
    return zero_one_loss(y, learner.predict(X))


def generalization_risk(learner, X_test, y_test):
    return zero_one_loss(y_test, learner.predict(X_test))


# 4. 模拟采样训练集并计算 (R - R_hat) 的期望
n_sim = 2000
m = 20
unstable_diff = []
stable_diff = []

for _ in range(n_sim):
    # 有放回采样 m 个样本作为训练集
    indices = np.random.choice(len(X_train_full), size=m, replace=True)
    X_D, y_D = X_train_full[indices], y_train_full[indices]

    # 训练不稳定算法
    unstable_learner.fit(X_D, y_D)
    R_hat_unstable = empirical_risk(unstable_learner, X_D, y_D)
    R_unstable = generalization_risk(unstable_learner, X_test, y_test)
    unstable_diff.append(R_unstable - R_hat_unstable)

    # 训练稳定算法
    stable_learner.fit(X_D, y_D)
    R_hat_stable = empirical_risk(stable_learner, X_D, y_D)
    R_stable = generalization_risk(stable_learner, X_test, y_test)
    stable_diff.append(R_stable - R_hat_stable)

print("=== 期望泛化误差 (R - R_hat) ===")
print(
    f"不稳定 ERM : {np.mean(unstable_diff):.4f} ± {np.std(unstable_diff):.4f}"
)
print(f"稳定 ERM   : {np.mean(stable_diff):.4f} ± {np.std(stable_diff):.4f}")


# 5. 近似定理5.3右侧：替换样本损失差的期望
# 对每个训练集，随机选取一个位置 i，随机生成新样本 z'，计算损失差
def estimate_stability_term(
    learner, X_data, y_data, X_test, y_test, n_samples=1000
):
    diffs = []
    for _ in range(n_samples):
        # 采样训练集 D
        idx = np.random.choice(len(X_data), size=m, replace=True)
        X_D, y_D = X_data[idx], y_data[idx]
        # 随机选一个位置 i
        i = np.random.randint(0, m)
        z_i = (X_D[i], y_D[i])
        # 采样新样本 z' (独立同分布)
        idx_zprime = np.random.choice(len(X_data), size=1, replace=True)[0]
        X_zprime, y_zprime = X_data[idx_zprime], y_data[idx_zprime]
        # 构造替换后的数据集 D^{i,z'}
        X_D_new = np.copy(X_D)
        y_D_new = np.copy(y_D)
        X_D_new[i] = X_zprime
        y_D_new[i] = y_zprime
        # 训练原算法和替换后算法
        learner.fit(X_D, y_D)
        loss_original = zero_one_loss([z_i[1]], learner.predict([z_i[0]]))
        learner.fit(X_D_new, y_D_new)
        loss_new = zero_one_loss([z_i[1]], learner.predict([z_i[0]]))
        diffs.append(loss_new - loss_original)
    return np.mean(diffs)


print("\n=== 定理5.3右侧：替换样本损失差期望 ===")
unstable_stab = estimate_stability_term(
    unstable_learner, X_train_full, y_train_full, X_test, y_test
)
stable_stab = estimate_stability_term(
    stable_learner, X_train_full, y_train_full, X_test, y_test
)
print(f"不稳定 ERM : {unstable_stab:.4f}")
print(f"稳定 ERM   : {stable_stab:.4f}")

# 6. 验证定理5.3：左右两边是否相等（数值近似）
print("\n=== 验证定理5.3（数值近似）===")
print(
    f"不稳定 ERM: 左侧期望 = {np.mean(unstable_diff):.4f}, 右侧期望 = {unstable_stab:.4f}"
)
print(
    f"稳定 ERM  : 左侧期望 = {np.mean(stable_diff):.4f}, 右侧期望 = {stable_stab:.4f}"
)


# 7. 可视化：展示不稳定算法过拟合示例
def plot_decision_boundary(learner, X_train, y_train, X_test, y_test, title):
    plt.figure(figsize=(6, 5))
    # 绘制训练点
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="bwr",
        edgecolors="k",
        label="Train",
    )
    # 绘制测试点（半透明）
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="bwr",
        alpha=0.3,
        edgecolors="k",
        label="Test",
    )
    # 决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )
    Z = learner.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap="bwr")
    plt.title(title)
    plt.legend()
    plt.show()


# 选取一个训练集示例
indices = np.random.choice(len(X_train_full), size=m, replace=True)
X_D_ex, y_D_ex = X_train_full[indices], y_train_full[indices]
unstable_learner.fit(X_D_ex, y_D_ex)
stable_learner.fit(X_D_ex, y_D_ex)
plot_decision_boundary(
    unstable_learner,
    X_D_ex,
    y_D_ex,
    X_test,
    y_test,
    f"Unstable ERM (train acc={1 - empirical_risk(unstable_learner, X_D_ex, y_D_ex):.2f}, test acc={1 - generalization_risk(unstable_learner, X_test, y_test):.2f})",
)
plot_decision_boundary(
    stable_learner,
    X_D_ex,
    y_D_ex,
    X_test,
    y_test,
    f"Stable ERM (train acc={1 - empirical_risk(stable_learner, X_D_ex, y_D_ex):.2f}, test acc={1 - generalization_risk(stable_learner, X_test, y_test):.2f})",
)

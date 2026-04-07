import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. 定义数据生成函数 (与给定程序一致)
def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, labels

# 2. 生成训练集 (1000个点) 和 测试集 (500个点)
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)

# 3. 定义并训练模型
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "AdaBoost (DT)": AdaBoostClassifier(n_estimators=100, random_state=42),
    "SVM (Linear)": SVC(kernel='linear', random_state=42),
    "SVM (Poly)": SVC(kernel='poly', degree=3, random_state=42),
    "SVM (RBF)": SVC(kernel='rbf', gamma='auto', random_state=42)
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)

# 4. 打印结果
print(f"{'Algorithm':<20} | {'Accuracy':<10}")
print("-" * 35)
for name, acc in results.items():
    print(f"{name:<20} | {acc:.4f}")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

df = pd.read_csv("card.csv", sep=";")
df = df.drop(columns=["ID"], errors="ignore")
df = df.rename(columns={"default payment next month": "Y"})
df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)

#тепловая матрица до удаления
plt.figure(figsize=(15, 10))
corr_matrix = df.corr().abs()
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Корреляционная матрица ДО удаления сильно коррелирующих признаков")
plt.show()

# удаление сильно коррелирующих признаков (>0.9)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
print(f"\n Удаленные сильно коррелирующие признаки (>0.9): {to_drop}")

df_clean = df.drop(columns=to_drop)
print(f"Размер после удаления корреляций: {df_clean.shape}")

#тепловая матрица после удаления
plt.figure(figsize=(15, 10))
corr_matrix_clean = df_clean.corr().abs()
sns.heatmap(corr_matrix_clean, cmap="coolwarm")
plt.title("Корреляционная матрица ПОСЛЕ удаления сильно коррелирующих признаков")
plt.show()

#функция для расчета энтропии
def calculate_entropy(y):
    probs = np.bincount(y) / len(y)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))# H = -∑p*log(p)

#функция для расчета Information Gain прирост информации
def information_gain(X, y, feature):
    total_entropy = calculate_entropy(y)
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / len(y)) * calculate_entropy(y[X[feature] == v])
        for i, v in enumerate(values)
    )#IG(S, A) = H(S) − Σ (|Sᵥ| / |S|) · H(Sᵥ)
    return total_entropy - weighted_entropy

#функция для расчета Split Information
def split_information(X, feature):
    values, counts = np.unique(X[feature], return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))#SplitInfo(S, A) = − Σ (|Sᵥ| / |S|) · log₂(|Sᵥ| / |S|)

#вычисляем Gain Ratio для всех признаков(коэфициент усиления)
def gain_ratio(X, y):
    gain_ratios = {}
    for feature in X.columns:
        ig = information_gain(X, y, feature)
        si = split_information(X, feature)
        gr = ig / si if si != 0 else 0 # GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)
        gain_ratios[feature] = gr
    return pd.Series(gain_ratios).sort_values(ascending=False)

#вычисляем GAIN RATIO
X = df_clean.drop("Y", axis=1)
y = df_clean["Y"].values.astype(int)#целевая переменная

gain_ratios_series = gain_ratio(X, y)
print("\n топ 10 по GAIN RATIO (C4.5):")
print(gain_ratios_series.head(10))

#обучение дерева решений C4.5
model = DecisionTreeClassifier(
    criterion="entropy",  
    max_depth=5,
    min_samples_split=100,
    random_state=42
)
model.fit(X, y)

# Вывод структуры дерева
tree_rules = export_text(model, feature_names=list(X.columns), max_depth=5)
print("\n Структура дерева рещения:\n")
print(tree_rules)
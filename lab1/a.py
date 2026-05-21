import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from C45 import C45Classifier 

df = pd.read_csv("card.csv", sep=";")
df.drop(columns=["ID"], inplace=True, errors="ignore")
df.rename(columns={"default payment next month": "Y"}, inplace=True)
df["EDUCATION"].replace([0,5,6], 4, inplace=True)

# Тепловая карта ДО
corr = df.corr().abs()
plt.figure(figsize=(15,12))
sns.heatmap(corr, cmap="coolwarm")
plt.title("ДО удаления")
plt.savefig("corr_before.png")
plt.show()

# Удаление >0.9
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
df_clean = df.drop(columns=to_drop)
print(f"Удалено: {to_drop}")

# Тепловая карта ПОСЛЕ
corr = df_clean.corr().abs()
plt.figure(figsize=(15,12))
sns.heatmap(corr, cmap="coolwarm")
plt.title("ПОСЛЕ удаления")
plt.savefig("corr_after.png")
plt.show()

# 🎯 НАСТОЯЩИЙ C4.5 с Gain Ratio!
X = df_clean.drop("Y", axis=1)
y = df_clean["Y"]

tree = C45Classifier()
tree.fit(X, y)

# ✅ Gain Ratio C4.5 (НЕ IG!)
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Gain_Ratio_C45": tree.feature_importances_
}).sort_values("Gain_Ratio_C45", ascending=False)

print("🏆 ТОП-10 ПО GAIN RATIO C4.5:")
print(importance_df.head(10))

# Правила дерева
print("\n📋 ПРАВИЛА C4.5:")
for rule in tree.rules:
    print(rule)

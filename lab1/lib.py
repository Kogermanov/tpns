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

#print(df.head())
#df = df.drop(0).reset_index(drop=True)#описанием признаков 
#df = df.drop(columns=["Unnamed: 0"], errors="ignore")#лишний признак(id) просто номер строки, не признак

#Тепловая матрица до удаления сильно коррелирующих признаков 
corr_matrix = df.corr().abs()#модуль корреляции 
plt.figure(figsize=(15,10))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("До удаления")
plt.show()

#Удаление сильно коррелирующих признаков (>0.9)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))#берем верхний треугольник
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
df = df.drop(columns=to_drop)
print("Сильно коррелированные признаки:", to_drop)


#Тепловая матрица после удаления сильно коррелирующих признаков
corr_matrix = df.corr().abs()
plt.figure(figsize=(15,10))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("После удаления")
plt.show()

#Разделение признаков и целевую переменную
X = df.drop("Y", axis=1)
y = df["Y"]

#Обучаем дерево решений (C4.5: criterion='entropy')изучает данные и строит дерево решений для классификации.
model = DecisionTreeClassifier(
    criterion="entropy",# метрика 
    max_depth=5,
    random_state=42
)
model.fit(X, y)

# Печать дерева
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)

#Важность признаков 
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nТоп 10 признаков по важности:")
print(importance_df.head(10))

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Предобработка (из 1-й лабы)
df = pd.read_csv("card.csv", sep=";")
df = df.drop(columns=["ID"], errors="ignore")
df = df.rename(columns={"default payment next month": "Y"})
df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)

#Удаление коррелирующих признаков
corr_matrix = df.corr().abs()

upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
df = df.drop(columns=to_drop)

print("Удалены коррелирующие признаки:", to_drop)

X = df.drop("Y", axis=1).values
y = df["Y"].values

# Масштабирование
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

#Модель
input_size = X.shape[1]

model = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Обучение
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    #TRAIN
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    #TEST
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = loss_fn(test_outputs, y_test)
        test_losses.append(test_loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Test Loss = {test_loss.item():.4f}")

# График Loss
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Test Loss")
plt.legend()
plt.grid()
plt.show()

#Предсказания 
model.eval()
with torch.no_grad():
    y_probs = model(X_test).numpy()
    y_pred = (y_probs > 0.5).astype(int)
    y_true = y_test.numpy()

#Свои метрики 
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix_vals(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

acc = accuracy(y_true, y_pred)
tp, tn, fp, fn = confusion_matrix_vals(y_true, y_pred)

print("\nAccuracy:", acc)

#Confusion Matrix
# cm = np.array([
#     [tn, fp],
#     [fn, tp]
# ])

# plt.figure()
# plt.imshow(cm)
# plt.title("Confusion Matrix")
# plt.colorbar()

# plt.xticks([0,1], ["0","1"])
# plt.yticks([0,1], ["0","1"])

# for i in range(2):
#     for j in range(2):
#         plt.text(j, i, cm[i, j], ha="center", va="center")

# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()
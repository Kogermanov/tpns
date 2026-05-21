import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Предобработка
df = pd.read_csv("card.csv", sep=";")
df = df.drop(columns=["ID"], errors="ignore")
df = df.rename(columns={"default payment next month": "Y"})
df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)

#Удаление корреляции
corr_matrix = df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
df = df.drop(columns=to_drop)

print("Удалены признаки:", to_drop)

X = df.drop("Y", axis=1).values
y = df["Y"].values.reshape(-1, 1)

#Нормализация (своя)
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

#Train/Test split (свой)
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))

train_idx = indices[:split]
test_idx = indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

#СЛОИ (Архитектура сети)
#линейное преобразование признаков
class Linear:
    def __init__(self, in_f, out_f, lr=0.01):
        self.W = np.random.randn(in_f, out_f) * np.sqrt(2 / in_f)
        self.b = np.zeros((1, out_f))
        self.lr = lr

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        dW = self.x.T @ grad
        db = np.sum(grad, axis=0, keepdims=True)
        dx = grad @ self.W.T

        self.W -= self.lr * dW
        self.b -= self.lr * db

        return dx

#добавляет нелинейность, иначе сеть — просто линейная модель(пропускаем только полезный сигнал)
class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask

#превращает выход в вероятность (0–1)
class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)

#функция потерь(Binary Cross Entropy)
class BCELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        self.y_true = y_true
        return -np.mean(
            y_true * np.log(self.y_pred) +
            (1 - y_true) * np.log(1 - self.y_pred) #Loss = - (y * log(p) + (1 - y) * log(1 - p))
        )

    def backward(self):
        return (self.y_pred - self.y_true) / self.y_true.shape[0]


#МОДЕЛЬ
class Model:
    def __init__(self):
        self.layers = [
            Linear(X_train.shape[1], 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 1),
            Sigmoid()
        ]
        self.loss_fn = BCELoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)


model = Model()

# ОБУЧЕНИЕ 
epochs = 1000
train_losses = []
test_losses = []

for epoch in range(epochs):
    # TRAIN
    y_pred = model.forward(X_train)
    loss = model.loss_fn.forward(y_pred, y_train)

    grad = model.loss_fn.backward()#считаем как сильно модель ошиблась
    model.backward(grad)# отправляем менять веса

    train_losses.append(loss)

    # TEST
    y_test_pred = model.forward(X_test)
    test_loss = model.loss_fn.forward(y_test_pred, y_test)
    test_losses.append(test_loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Train Loss={loss:.4f}, Test Loss={test_loss:.4f}")

# ГРАФИК LOSS 
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Train vs Test Loss")
plt.grid()
plt.show()

# ПРЕДСКАЗАНИЯ 
y_probs = model.forward(X_test)
y_pred = (y_probs > 0.5).astype(int)

# МЕТРИКИ 
#доля правильных ответов
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix_vals(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

acc = accuracy(y_test, y_pred)
tp, tn, fp, fn = confusion_matrix_vals(y_test, y_pred)

print("\nAccuracy:", acc)
print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)

# CONFUSION MATRIX 
cm = np.array([[tn, fp],
               [fn, tp]])

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
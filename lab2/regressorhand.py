import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("student/student-por.csv", sep=";")
# Целевая переменная
y = df["G3"].values.reshape(-1, 1)
# Признаки
X = df.drop("G3", axis=1)
# One-Hot Encoding категориальных признаков
X = pd.get_dummies(X).astype(float).values
# НОРМАЛИЗАЦИЯ (как в классификаторе)
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.7 * len(X))
train_idx = indices[:split]
test_idx = indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

#СЛОИ (Архитектура сети)
class Linear:
    def __init__(self, in_f, out_f, lr=0.001):
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

        # обновление весов
        self.W -= self.lr * dW
        self.b -= self.lr * db

        return dx


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask

#ДРУГОЕ
#функция потерь насколько число ошибается  1/n*​∑(ypred​−ytrue​)^2
class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        #ДРУГОЕ Градиент loss-функции (производная квадрата добавила 2)
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]


#модель
class Model:
    def __init__(self):
        self.layers = [
            Linear(X_train.shape[1], 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            #ДРУГОЕ
            Linear(64, 1)   # БЕЗ слоя Sigmoid(переводит выход в вероятность)
        ]

        self.loss_fn = MSELoss()#ДРУГОЕ

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
    grad = model.loss_fn.backward()
    model.backward(grad)
    train_losses.append(loss)

    # TEST
    y_test_pred = model.forward(X_test)
    test_loss = model.loss_fn.forward(y_test_pred, y_test)
    test_losses.append(test_loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Train Loss={loss:.4f}, Test Loss={test_loss:.4f}")

#график loss
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Train vs Test Loss")
plt.grid()
plt.show()

# ПРЕДСКАЗАНИЯ 
y_pred = model.forward(X_test)

#метрика MAE  на сколько в среднем модель ошибается в единицах целевой переменной
mae = np.mean(np.abs(y_pred - y_test))
print("\nMAE:", mae)

#график предсказаний и реальных значений
y_true = y_test.flatten()
y_pred_flat = y_pred.flatten()

plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred_flat, alpha=0.5)
plt.plot(
    [0, 20],
    [0, 20],
    color='red',
    linestyle='--',
    label='Ideal Prediction'
)

plt.xlabel("Настоящие оценки (G3)")
plt.ylabel("Предсказанные оценки")
plt.title("Предсказания модели и Реальные значения")
plt.legend()
plt.grid()
plt.show()
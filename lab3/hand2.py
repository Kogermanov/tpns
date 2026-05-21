import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])
print("size: ", df.shape)

plt.figure(figsize=(12,5))
plt.plot(df["cnt"][:200])
plt.title("Количество велосипедов (первые 200 часов)")
plt.show()

features = ["cnt", "temp", "hum", "windspeed"]
data = df[features].values


mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

data = (data - mean) / std

def create_sequences(data, window_size):
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size][0])

    return np.array(X), np.array(y)

window_size = 24
X, y = create_sequences(data, window_size)

print("До reshape:", X.shape)

print("После reshape:", X.shape)

split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train:", X_train.shape)
print("Test:", X_test.shape)

input_size = X.shape[2]
hidden_size = 10
output_size = 1

np.random.seed(42)

def rnn_forward(x_seq):
    h = np.zeros((1, hidden_size))

    for t in range(x_seq.shape[0]):
        x_t = x_seq[t].reshape(1, -1)
        h = np.tanh(np.dot(x_t, Wx) + np.dot(h, Wh) + b_h)

    y = np.dot(h, Wy) + b_y
    return y, h

Wx = np.random.randn(input_size, hidden_size) * 0.1
Wh = np.random.randn(hidden_size, hidden_size) * 0.1
Wy = np.random.randn(hidden_size, output_size) * 0.1

b_h = np.zeros((1, hidden_size))
b_y = np.zeros((1, output_size))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

epochs = 10
lr = 0.01

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X_train)):
        x = X_train[i]
        y_true = y_train[i]

        y_pred, h = rnn_forward(x)

        loss = mse(y_true, y_pred)
        total_loss += loss

        dy = y_pred - y_true
        dWy = np.dot(h.T, dy)
        db_y = dy

        Wy -= lr * dWy
        b_y -= lr * db_y

    print(f"RNN Epoch {epoch}, Loss: {total_loss/len(X_train)}")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gru_forward(x_seq):
    h = np.zeros((1, hidden_size))

    for t in range(x_seq.shape[0]):
        x_t = x_seq[t].reshape(1, -1)

        z = sigmoid(np.dot(x_t, Wz) + np.dot(h, Uz) + bz)
        r = sigmoid(np.dot(x_t, Wr) + np.dot(h, Ur) + br)

        h_tilde = np.tanh(np.dot(x_t, Wh_g) + np.dot(r*h, Uh) + bh)
        h = (1 - z) * h + z * h_tilde

    y = np.dot(h, Wy_g) + b_y_g
    return y, h

Wz = np.random.randn(input_size, hidden_size) * 0.1
Uz = np.random.randn(hidden_size, hidden_size) * 0.1
Wr = np.random.randn(input_size, hidden_size) * 0.1
Ur = np.random.randn(hidden_size, hidden_size) * 0.1
Wh_g = np.random.randn(input_size, hidden_size) * 0.1
Uh = np.random.randn(hidden_size, hidden_size) * 0.1

bz = br = bh = np.zeros((1, hidden_size))

Wy_g = np.random.randn(hidden_size, output_size) * 0.1
b_y_g = np.zeros((1, output_size))

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X_train)):
        y_pred, h = gru_forward(X_train[i])
        loss = mse(y_train[i], y_pred)
        total_loss += loss

        dy = y_pred - y_train[i]
        dWy = np.dot(h.T, dy)

        Wy_g -= lr * dWy
        b_y_g -= lr * dy

    print(f"GRU Epoch {epoch}, Loss: {total_loss/len(X_train)}")

def lstm_forward(x_seq):
    h = np.zeros((1, hidden_size))
    c = np.zeros((1, hidden_size))

    for t in range(x_seq.shape[0]):
        x_t = x_seq[t].reshape(1, -1)

        f = sigmoid(np.dot(x_t, Wf) + np.dot(h, Uf) + bf)
        i = sigmoid(np.dot(x_t, Wi) + np.dot(h, Ui) + bi)
        o = sigmoid(np.dot(x_t, Wo) + np.dot(h, Uo) + bo)

        c_tilde = np.tanh(np.dot(x_t, Wc) + np.dot(h, Uc) + bc)

        c = f * c + i * c_tilde
        h = o * np.tanh(c)

    y = np.dot(h, Wy_l) + b_y_l
    return y, h

Wf = np.random.randn(input_size, hidden_size) * 0.1
Uf = np.random.randn(hidden_size, hidden_size) * 0.1
Wi = np.random.randn(input_size, hidden_size) * 0.1
Ui = np.random.randn(hidden_size, hidden_size) * 0.1
Wo = np.random.randn(input_size, hidden_size) * 0.1
Uo = np.random.randn(hidden_size, hidden_size) * 0.1
Wc = np.random.randn(input_size, hidden_size) * 0.1
Uc = np.random.randn(hidden_size, hidden_size) * 0.1

bf = bi = bo = bc = np.zeros((1, hidden_size))

Wy_l = np.random.randn(hidden_size, output_size) * 0.1
b_y_l = np.zeros((1, output_size))

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X_train)):
        y_pred, h = lstm_forward(X_train[i])
        loss = mse(y_train[i], y_pred)
        total_loss += loss

        dy = y_pred - y_train[i]
        dWy = np.dot(h.T, dy)

        Wy_l -= lr * dWy
        b_y_l -= lr * dy

    print(f"LSTM Epoch {epoch}, Loss: {total_loss/len(X_train)}")

def predict(model, X):
    preds = []
    for i in range(len(X)):
        y_pred, _ = model(X[i])
        preds.append(y_pred[0][0])
    return np.array(preds)

rnn_pred = predict(rnn_forward, X_test)
gru_pred = predict(gru_forward, X_test)
lstm_pred = predict(lstm_forward, X_test)


rnn_pred = rnn_pred * std[0] + mean[0]
gru_pred = gru_pred * std[0] + mean[0]
lstm_pred = lstm_pred * std[0] + mean[0]

y_test_real = y_test * std[0] + mean[0]

plt.figure(figsize=(12,5))
plt.plot(y_test_real[:200], label="Реальные")
plt.plot(rnn_pred[:200], label="RNN")
plt.plot(gru_pred[:200], label="GRU")
plt.plot(lstm_pred[:200], label="LSTM")
plt.legend()
plt.title("Сравнение моделей")
plt.show()

print("\nРЕЗУЛЬТАТЫ")
print("RNN  MSE:", mse(y_test_real, rnn_pred), "MAE:", mae(y_test_real, rnn_pred))
print("GRU  MSE:", mse(y_test_real, gru_pred), "MAE:", mae(y_test_real, gru_pred))
print("LSTM MSE:", mse(y_test_real, lstm_pred), "MAE:", mae(y_test_real, lstm_pred))
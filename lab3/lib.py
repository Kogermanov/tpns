import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K




data_path = "bike+sharing+dataset/hour.csv"
data = pd.read_csv(data_path)

print("Размер датасета:", data.shape)


features = ["cnt", "temp", "hum", "windspeed"]
data = data[features].values


# Нормализация

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


#Создание последовательностей
def create_sequences(data, window=24):
    X_seq, y_seq = [], []

    for i in range(len(data) - window):
        X_seq.append(data[i:i + window])
        y_seq.append(data[i + window][0])  # cnt — первый столбец

    return np.array(X_seq), np.array(y_seq)


sequence_length = 24
X_seq, y_seq = create_sequences(data_scaled, window=sequence_length)


# Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X_seq,
    y_seq,
    test_size=0.2,
    shuffle=False
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# Метрика R2

def custom_r2(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - ss_res / (ss_tot + K.epsilon())


# Функция обучения модели

def train_model(model, name):
    print(f"\nОбучение модели {name}")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=False
    )

    val_losses[name] = history.history["val_loss"]
    val_r2_scores[name] = history.history["val_custom_r2"]

    test_loss, test_r2 = model.evaluate(X_test, y_test, verbose=0)

    print(f"{name} — Test MSE: {test_loss:.5f}, Test R2: {test_r2:.5f}")

    return history


#параметры


num_epochs = 20
batch_size = 32
learning_rate = 0.001

input_shape = X_train.shape[1:]

val_losses = {}
val_r2_scores = {}


# SimpleRNN

rnn_model = Sequential([
    SimpleRNN(32, input_shape=input_shape),
    Dense(1)
])

rnn_model.compile(
    optimizer=Adam(learning_rate),
    loss="mse",
    metrics=[custom_r2]
)

train_model(rnn_model, "SimpleRNN")


# GRU


gru_model = Sequential([
    GRU(32, input_shape=input_shape),
    Dense(1)
])

gru_model.compile(
    optimizer=Adam(learning_rate),
    loss="mse",
    metrics=[custom_r2]
)

train_model(gru_model, "GRU")


# LSTM

lstm_model = Sequential([
    LSTM(32, input_shape=input_shape),
    Dense(1)
])

lstm_model.compile(
    optimizer=Adam(learning_rate),
    loss="mse",
    metrics=[custom_r2]
)

train_model(lstm_model, "LSTM")


# График validation loss

plt.figure(figsize=(10, 5))

for name, loss in val_losses.items():
    plt.plot(loss, label=f"{name} - Val Loss")

plt.title("Validation Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


#  Обратное преобразование cnt после нормализации

def inverse_cnt(values):
    dummy = np.zeros((len(values), len(features)))
    dummy[:, 0] = values.reshape(-1)
    restored = scaler.inverse_transform(dummy)
    return restored[:, 0]


y_pred_rnn = inverse_cnt(rnn_model.predict(X_test).reshape(-1))
y_pred_gru = inverse_cnt(gru_model.predict(X_test).reshape(-1))
y_pred_lstm = inverse_cnt(lstm_model.predict(X_test).reshape(-1))

y_test_real = inverse_cnt(y_test.reshape(-1))


#  Собственные метрики на реальных значениях

def mse_np(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse_np(y_true, y_pred):
    return np.sqrt(mse_np(y_true, y_pred))


def r2_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


print("\nРЕЗУЛЬТАТЫ НА TEST")
print(
    "SimpleRNN:",
    "MSE =", round(mse_np(y_test_real, y_pred_rnn), 3),
    "RMSE =", round(rmse_np(y_test_real, y_pred_rnn), 3),
    "MAE =", round(mae_np(y_test_real, y_pred_rnn), 3),
    "R2 =", round(r2_np(y_test_real, y_pred_rnn), 3)
)

print(
    "GRU:",
    "MSE =", round(mse_np(y_test_real, y_pred_gru), 3),
    "RMSE =", round(rmse_np(y_test_real, y_pred_gru), 3),
    "MAE =", round(mae_np(y_test_real, y_pred_gru), 3),
    "R2 =", round(r2_np(y_test_real, y_pred_gru), 3)
)

print(
    "LSTM:",
    "MSE =", round(mse_np(y_test_real, y_pred_lstm), 3),
    "RMSE =", round(rmse_np(y_test_real, y_pred_lstm), 3),
    "MAE =", round(mae_np(y_test_real, y_pred_lstm), 3),
    "R2 =", round(r2_np(y_test_real, y_pred_lstm), 3)
)


#Общий график сравнения

plt.figure(figsize=(12, 5))

plt.plot(y_test_real[:200], label="Реальные значения", color="black", linewidth=2)
plt.plot(y_pred_rnn[:200], label="SimpleRNN", linestyle="--")
plt.plot(y_pred_gru[:200], label="GRU", linestyle="--")
plt.plot(y_pred_lstm[:200], label="LSTM", linestyle="--")

plt.title("Сравнение моделей на первых 200 часах тестовой выборки")
plt.xlabel("Час")
plt.ylabel("Количество аренд")
plt.legend()
plt.grid(True)
plt.show()


#  Отдельные графики

plt.figure(figsize=(10, 4))
plt.plot(y_test_real[:200], label="Реальные значения", color="black", linewidth=2)
plt.plot(y_pred_rnn[:200], label="SimpleRNN", linestyle="--")
plt.title("SimpleRNN: прогноз vs реальные значения")
plt.xlabel("Час")
plt.ylabel("Количество аренд")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(y_test_real[:200], label="Реальные значения", color="black", linewidth=2)
plt.plot(y_pred_gru[:200], label="GRU", linestyle="--")
plt.title("GRU: прогноз vs реальные значения")
plt.xlabel("Час")
plt.ylabel("Количество аренд")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(y_test_real[:200], label="Реальные значения", color="black", linewidth=2)
plt.plot(y_pred_lstm[:200], label="LSTM", linestyle="--")
plt.title("LSTM: прогноз vs реальные значения")
plt.xlabel("Час")
plt.ylabel("Количество аренд")
plt.legend()
plt.grid(True)
plt.show()
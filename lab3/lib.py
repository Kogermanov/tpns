
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt



# Загружаем датасет с количеством аренд велосипедов
data_path = 'bike+sharing+dataset/day.csv'
data = pd.read_csv(data_path)

columns_to_drop = ['instant', 'dteday', 'casual', 'registered']
data.drop(columns=columns_to_drop, inplace=True)

# Разделяем данные на признаки (X) и целевую переменную (y)
X = data.drop(columns='cnt').values
y = data[['cnt']].values


# Нормализация данных
# Приводим признаки и целевую переменную к одному масштабу (0–1)
# Это ускоряет и стабилизирует обучение нейросети
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)


# Формирование временных последовательностей
# Используем скользящее окно длиной 7 дней
# Модель учится предсказывать значение на следующий день
def create_sequences(X, y, window=7):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])   # 7 предыдущих дней
        y_seq.append(y[i+window])     # следующий день
    return np.array(X_seq), np.array(y_seq)

sequence_length = 7
X_seq, y_seq = create_sequences(X_scaled, y_scaled, window=sequence_length)



# Разделение на train/test
# Временной ряд нельзя перемешивать, поэтому shuffle=False
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)



# Метрика качества (R²)
# Показывает, насколько хорошо модель объясняет вариативность данных
def custom_r2(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - ss_res / (ss_tot + K.epsilon())



# Функция обучения модели
# Универсальная функция для обучения RNN/GRU/LSTM
# и сохранения метрик
def train_model(model, name):
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=1
    )

    val_losses[name] = history.history['val_loss']
    val_r2_scores[name] = history.history['val_custom_r2']

    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'{name} — Test Loss (MSE): {test_loss}')
    return history



# Гиперпараметры
num_epochs = 50
batch_size = 32
learning_rate = 0.001
input_shape = X_train.shape[1:]

val_losses = {}
val_r2_scores = {}



# Модель SimpleRNN
# Базовая рекуррентная сеть
rnn_model = Sequential([
    SimpleRNN(32, input_shape=input_shape),
    Dense(1)
])
rnn_model.compile(
    optimizer=Adam(learning_rate),
    loss='mse',
    metrics=[custom_r2]
)
train_model(rnn_model, 'SimpleRNN')



#  Модель GRU
# Улучшенная RNN с контролем памяти (gates)
gru_model = Sequential([
    GRU(32, input_shape=input_shape),
    Dense(1)
])
gru_model.compile(
    optimizer=Adam(learning_rate),
    loss='mse',
    metrics=[custom_r2]
)
train_model(gru_model, 'GRU')



# Модель LSTM
# Самая мощная RNN с долгосрочной памятью
lstm_model = Sequential([
    LSTM(32, input_shape=input_shape),
    Dense(1)
])
lstm_model.compile(
    optimizer=Adam(learning_rate),
    loss='mse',
    metrics=[custom_r2]
)
train_model(lstm_model, 'LSTM')



# Сравнение validation loss для всех моделей
plt.figure(figsize=(9, 5))
for name, loss in val_losses.items():
    plt.plot(loss, label=f'{name} - Val Loss')
plt.title('Validation Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()



# Предсказания моделей
# Возвращаем значения из нормализованного вида в реальные числа
y_pred_rnn = y_scaler.inverse_transform(rnn_model.predict(X_test))
y_pred_gru = y_scaler.inverse_transform(gru_model.predict(X_test))
y_pred_lstm = y_scaler.inverse_transform(lstm_model.predict(X_test))
y_test_real = y_scaler.inverse_transform(y_test)



# Сравнение реальных значений и предсказаний каждой модели

plt.figure(figsize=(10, 4))
plt.plot(y_test_real, label='Реальные значения', color='black', linewidth=2)
plt.plot(y_pred_rnn, label='SimpleRNN', linestyle='--')
plt.title('SimpleRNN: Предсказание vs Реальные значения')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(y_test_real, label='Реальные значения', color='black', linewidth=2)
plt.plot(y_pred_gru, label='GRU', linestyle='--')
plt.title('GRU: Предсказание vs Реальные значения')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(y_test_real, label='Реальные значения', color='black', linewidth=2)
plt.plot(y_pred_lstm, label='LSTM', linestyle='--')
plt.title('LSTM: Предсказание vs Реальные значения')
plt.legend()
plt.grid(True)
plt.show()
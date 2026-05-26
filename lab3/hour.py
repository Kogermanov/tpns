import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Настройки
# ============================================================

DATA_PATH = "bike+sharing+dataset/hour.csv"
# Если hour.csv лежит рядом со скриптом, путь будет найден автоматически.

FEATURES = ["cnt", "temp", "hum", "windspeed"]

WINDOW_SIZE = 24      # 24 часа = одни сутки истории
TRAIN_RATIO = 0.8

HIDDEN_SIZE = 10
OUTPUT_SIZE = 1

EPOCHS = 10
LR = 0.001            # для полного backward лучше меньше, чем 0.01
CLIP_VALUE = 5.0

SEED = 42
np.random.seed(SEED)


# ============================================================
# 2. Загрузка данных
# ============================================================

def load_data(path):
    if not os.path.exists(path):
        if os.path.exists("hour.csv"):
            path = "hour.csv"
        else:
            raise FileNotFoundError(
                "Не найден файл hour.csv. "
                "Положи его рядом со скриптом или в папку bike+sharing+dataset/hour.csv"
            )

    df = pd.read_csv(path)
    print("Размер датасета:", df.shape)
    return df


# ============================================================
# 3. Метрики, реализованные вручную
# ============================================================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


# ============================================================
# 4. Ручная стандартизация и создание последовательностей
# ============================================================

def standardize_train_test(data, train_row_count):
    """
    Стандартизация без автоматических preprocessors.
    mean и std считаются только по train-части, чтобы не было утечки из test.
    """
    mean = np.mean(data[:train_row_count], axis=0)
    std = np.std(data[:train_row_count], axis=0)

    # защита от деления на ноль
    std[std == 0] = 1

    data_scaled = (data - mean) / std
    return data_scaled, mean, std


def create_sequences(data, window_size):
    """
    X[i] = данные за прошлые window_size часов.
    y[i] = cnt на следующий час.
    Так как cnt находится в data[:, 0], целевая переменная — data[i + window_size][0].
    """
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append([data[i + window_size][0]])

    return np.array(X), np.array(y)


# ============================================================
# 5. Вспомогательные функции
# ============================================================

def sigmoid(x):
    # защита от переполнения exp
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


def init_matrix(rows, cols):
    return np.random.randn(rows, cols) * np.sqrt(2 / (rows + cols))


def clip_gradients(*grads):
    for grad in grads:
        np.clip(grad, -CLIP_VALUE, CLIP_VALUE, out=grad)


# ============================================================
# 6. Собственная реализация RNN с полным backward
# ============================================================

class RNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        self.Wh = init_matrix(hidden_size, input_size)     # input -> hidden
        self.Uh = init_matrix(hidden_size, hidden_size)    # hidden -> hidden
        self.bh = np.zeros((hidden_size, 1))

        self.Why = init_matrix(output_size, hidden_size)   # hidden -> output
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        self.inputs = inputs
        self.hs = {-1: np.zeros((self.hidden_size, 1))}

        for t in range(len(inputs)):
            x = inputs[t].reshape(-1, 1)
            h_prev = self.hs[t - 1]

            h = np.tanh(self.Wh @ x + self.Uh @ h_prev + self.bh)
            self.hs[t] = h

        y = self.Why @ self.hs[len(inputs) - 1] + self.by
        return y.flatten()

    def backward(self, dy):
        dy = dy.reshape(-1, 1)

        dWhy = dy @ self.hs[len(self.inputs) - 1].T
        dby = dy.copy()

        dWh = np.zeros_like(self.Wh)
        dUh = np.zeros_like(self.Uh)
        dbh = np.zeros_like(self.bh)

        dh_next = self.Why.T @ dy

        for t in reversed(range(len(self.inputs))):
            x = self.inputs[t].reshape(-1, 1)
            h = self.hs[t]
            h_prev = self.hs[t - 1]

            dtanh = (1 - h ** 2) * dh_next

            dWh += dtanh @ x.T
            dUh += dtanh @ h_prev.T
            dbh += dtanh

            dh_next = self.Uh.T @ dtanh

        clip_gradients(dWh, dUh, dbh, dWhy, dby)

        self.Wh -= self.lr * dWh
        self.Uh -= self.lr * dUh
        self.bh -= self.lr * dbh
        self.Why -= self.lr * dWhy
        self.by -= self.lr * dby


# ============================================================
# 7. Собственная реализация GRU с полным backward
# ============================================================

class GRU:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # update gate
        self.Wz = init_matrix(hidden_size, input_size)
        self.Uz = init_matrix(hidden_size, hidden_size)
        self.bz = np.zeros((hidden_size, 1))

        # reset gate
        self.Wr = init_matrix(hidden_size, input_size)
        self.Ur = init_matrix(hidden_size, hidden_size)
        self.br = np.zeros((hidden_size, 1))

        # candidate hidden state
        self.Wh = init_matrix(hidden_size, input_size)
        self.Uh = init_matrix(hidden_size, hidden_size)
        self.bh = np.zeros((hidden_size, 1))

        # output
        self.Why = init_matrix(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        self.inputs = inputs
        self.hs = {-1: np.zeros((self.hidden_size, 1))}
        self.zs = {}
        self.rs = {}
        self.h_tildes = {}

        for t in range(len(inputs)):
            x = inputs[t].reshape(-1, 1)
            h_prev = self.hs[t - 1]

            z = sigmoid(self.Wz @ x + self.Uz @ h_prev + self.bz)
            r = sigmoid(self.Wr @ x + self.Ur @ h_prev + self.br)
            h_tilde = np.tanh(self.Wh @ x + self.Uh @ (r * h_prev) + self.bh)

            h = (1 - z) * h_prev + z * h_tilde

            self.zs[t] = z
            self.rs[t] = r
            self.h_tildes[t] = h_tilde
            self.hs[t] = h

        y = self.Why @ self.hs[len(inputs) - 1] + self.by
        return y.flatten()

    def backward(self, dy):
        dy = dy.reshape(-1, 1)

        dWhy = dy @ self.hs[len(self.inputs) - 1].T
        dby = dy.copy()

        dWz = np.zeros_like(self.Wz)
        dUz = np.zeros_like(self.Uz)
        dbz = np.zeros_like(self.bz)

        dWr = np.zeros_like(self.Wr)
        dUr = np.zeros_like(self.Ur)
        dbr = np.zeros_like(self.br)

        dWh = np.zeros_like(self.Wh)
        dUh = np.zeros_like(self.Uh)
        dbh = np.zeros_like(self.bh)

        dh_next = self.Why.T @ dy

        for t in reversed(range(len(self.inputs))):
            x = self.inputs[t].reshape(-1, 1)
            h_prev = self.hs[t - 1]
            z = self.zs[t]
            r = self.rs[t]
            h_tilde = self.h_tildes[t]

            # h = (1 - z) * h_prev + z * h_tilde
            dz = dh_next * (h_tilde - h_prev)
            dz_raw = dz * z * (1 - z)

            dh_tilde = dh_next * z
            dh_tilde_raw = dh_tilde * (1 - h_tilde ** 2)

            # h_tilde = tanh(Wh*x + Uh*(r*h_prev) + bh)
            drh_prev = self.Uh.T @ dh_tilde_raw
            dr = drh_prev * h_prev
            dr_raw = dr * r * (1 - r)

            dWz += dz_raw @ x.T
            dUz += dz_raw @ h_prev.T
            dbz += dz_raw

            dWr += dr_raw @ x.T
            dUr += dr_raw @ h_prev.T
            dbr += dr_raw

            dWh += dh_tilde_raw @ x.T
            dUh += dh_tilde_raw @ (r * h_prev).T
            dbh += dh_tilde_raw

            dh_prev = (
                dh_next * (1 - z)
                + self.Uz.T @ dz_raw
                + self.Ur.T @ dr_raw
                + r * drh_prev
            )

            dh_next = dh_prev

        clip_gradients(dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh, dWhy, dby)

        self.Wz -= self.lr * dWz
        self.Uz -= self.lr * dUz
        self.bz -= self.lr * dbz

        self.Wr -= self.lr * dWr
        self.Ur -= self.lr * dUr
        self.br -= self.lr * dbr

        self.Wh -= self.lr * dWh
        self.Uh -= self.lr * dUh
        self.bh -= self.lr * dbh

        self.Why -= self.lr * dWhy
        self.by -= self.lr * dby


# ============================================================
# 8. Собственная реализация LSTM с полным backward
# ============================================================

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # forget gate
        self.Wf = init_matrix(hidden_size, input_size)
        self.Uf = init_matrix(hidden_size, hidden_size)
        self.bf = np.zeros((hidden_size, 1))

        # input gate
        self.Wi = init_matrix(hidden_size, input_size)
        self.Ui = init_matrix(hidden_size, hidden_size)
        self.bi = np.zeros((hidden_size, 1))

        # candidate cell state
        self.Wc = init_matrix(hidden_size, input_size)
        self.Uc = init_matrix(hidden_size, hidden_size)
        self.bc = np.zeros((hidden_size, 1))

        # output gate
        self.Wo = init_matrix(hidden_size, input_size)
        self.Uo = init_matrix(hidden_size, hidden_size)
        self.bo = np.zeros((hidden_size, 1))

        # output
        self.Why = init_matrix(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        self.inputs = inputs

        self.hs = {-1: np.zeros((self.hidden_size, 1))}
        self.cs = {-1: np.zeros((self.hidden_size, 1))}

        self.fs = {}
        self.is_ = {}
        self.cs_tilde = {}
        self.os = {}

        for t in range(len(inputs)):
            x = inputs[t].reshape(-1, 1)
            h_prev = self.hs[t - 1]
            c_prev = self.cs[t - 1]

            f = sigmoid(self.Wf @ x + self.Uf @ h_prev + self.bf)
            i = sigmoid(self.Wi @ x + self.Ui @ h_prev + self.bi)
            c_tilde = np.tanh(self.Wc @ x + self.Uc @ h_prev + self.bc)
            c = f * c_prev + i * c_tilde
            o = sigmoid(self.Wo @ x + self.Uo @ h_prev + self.bo)
            h = o * np.tanh(c)

            self.fs[t] = f
            self.is_[t] = i
            self.cs_tilde[t] = c_tilde
            self.cs[t] = c
            self.os[t] = o
            self.hs[t] = h

        y = self.Why @ self.hs[len(inputs) - 1] + self.by
        return y.flatten()

    def backward(self, dy):
        dy = dy.reshape(-1, 1)

        dWhy = dy @ self.hs[len(self.inputs) - 1].T
        dby = dy.copy()

        dWf = np.zeros_like(self.Wf)
        dUf = np.zeros_like(self.Uf)
        dbf = np.zeros_like(self.bf)

        dWi = np.zeros_like(self.Wi)
        dUi = np.zeros_like(self.Ui)
        dbi = np.zeros_like(self.bi)

        dWc = np.zeros_like(self.Wc)
        dUc = np.zeros_like(self.Uc)
        dbc = np.zeros_like(self.bc)

        dWo = np.zeros_like(self.Wo)
        dUo = np.zeros_like(self.Uo)
        dbo = np.zeros_like(self.bo)

        dh_next = self.Why.T @ dy
        dc_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(self.inputs))):
            x = self.inputs[t].reshape(-1, 1)
            h_prev = self.hs[t - 1]
            c_prev = self.cs[t - 1]

            f = self.fs[t]
            i = self.is_[t]
            c_tilde = self.cs_tilde[t]
            c = self.cs[t]
            o = self.os[t]

            tanh_c = np.tanh(c)

            # h = o * tanh(c)
            do = dh_next * tanh_c
            do_raw = do * o * (1 - o)

            dc = dh_next * o * (1 - tanh_c ** 2) + dc_next

            # c = f * c_prev + i * c_tilde
            df = dc * c_prev
            df_raw = df * f * (1 - f)

            di = dc * c_tilde
            di_raw = di * i * (1 - i)

            dc_tilde = dc * i
            dc_tilde_raw = dc_tilde * (1 - c_tilde ** 2)

            dWf += df_raw @ x.T
            dUf += df_raw @ h_prev.T
            dbf += df_raw

            dWi += di_raw @ x.T
            dUi += di_raw @ h_prev.T
            dbi += di_raw

            dWc += dc_tilde_raw @ x.T
            dUc += dc_tilde_raw @ h_prev.T
            dbc += dc_tilde_raw

            dWo += do_raw @ x.T
            dUo += do_raw @ h_prev.T
            dbo += do_raw

            dh_prev = (
                self.Uf.T @ df_raw
                + self.Ui.T @ di_raw
                + self.Uc.T @ dc_tilde_raw
                + self.Uo.T @ do_raw
            )

            dc_prev = dc * f

            dh_next = dh_prev
            dc_next = dc_prev

        clip_gradients(
            dWf, dUf, dbf,
            dWi, dUi, dbi,
            dWc, dUc, dbc,
            dWo, dUo, dbo,
            dWhy, dby
        )

        self.Wf -= self.lr * dWf
        self.Uf -= self.lr * dUf
        self.bf -= self.lr * dbf

        self.Wi -= self.lr * dWi
        self.Ui -= self.lr * dUi
        self.bi -= self.lr * dbi

        self.Wc -= self.lr * dWc
        self.Uc -= self.lr * dUc
        self.bc -= self.lr * dbc

        self.Wo -= self.lr * dWo
        self.Uo -= self.lr * dUo
        self.bo -= self.lr * dbo

        self.Why -= self.lr * dWhy
        self.by -= self.lr * dby


# ============================================================
# 9. Обучение и прогноз
# ============================================================

def train_model(model, X_train, y_train, epochs, model_name):
    losses = []

    print(f"\nОбучение {model_name}")

    for epoch in range(epochs):
        total_loss = 0

        # shuffle=False: сохраняем порядок временного ряда
        for i in range(len(X_train)):
            y_pred = model.forward(X_train[i])
            y_true = y_train[i]

            loss = mse(y_true, y_pred)
            total_loss += loss

            # градиент MSE по y_pred
            dy = 2 * (y_pred - y_true) / len(y_true)
            model.backward(dy)

        avg_loss = total_loss / len(X_train)
        losses.append(avg_loss)

        print(f"{model_name} Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    return losses


def predict(model, X):
    predictions = []

    for i in range(len(X)):
        y_pred = model.forward(X[i])
        predictions.append(y_pred[0])

    return np.array(predictions)


# ============================================================
# 10. Графики
# ============================================================

def plot_initial_series(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df["cnt"].values[:200])
    plt.title("Количество арендованных велосипедов: первые 200 часов")
    plt.xlabel("Час")
    plt.ylabel("cnt")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_losses(losses_dict):
    plt.figure(figsize=(10, 5))

    for name, losses in losses_dict.items():
        plt.plot(losses, label=name)

    plt.title("Ошибка обучения по эпохам")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_all_predictions(y_test_real, predictions_dict, count=200):
    plt.figure(figsize=(13, 5))

    plt.plot(y_test_real[:count], label="Реальные значения", linewidth=2)

    for name, pred in predictions_dict.items():
        plt.plot(pred[:count], label=name, linestyle="--")

    plt.title(f"Сравнение моделей на первых {count} часах тестовой выборки")
    plt.xlabel("Час")
    plt.ylabel("Количество аренд")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_single_predictions(y_test_real, predictions_dict, count=200):
    for name, pred in predictions_dict.items():
        plt.figure(figsize=(13, 5))
        plt.plot(y_test_real[:count], label="Реальные значения", linewidth=2)
        plt.plot(pred[:count], label=f"Прогноз {name}", linestyle="--")
        plt.title(f"{name}: прогноз против реальных значений")
        plt.xlabel("Час")
        plt.ylabel("Количество аренд")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ============================================================
# 11. Основная программа
# ============================================================

def main():
    df = load_data(DATA_PATH)

    plot_initial_series(df)

    data = df[FEATURES].values.astype(float)

    train_row_count = int(len(data) * TRAIN_RATIO)

    data_scaled, mean, std = standardize_train_test(data, train_row_count)

    X, y = create_sequences(data_scaled, WINDOW_SIZE)

    # train должен заканчиваться так, чтобы целевое значение y тоже было из train-части
    train_seq_count = train_row_count - WINDOW_SIZE

    X_train = X[:train_seq_count]
    y_train = y[:train_seq_count]

    X_test = X[train_seq_count:]
    y_test = y[train_seq_count:]

    print("\nФорма данных:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    input_size = X_train.shape[2]

    rnn = RNN(input_size, HIDDEN_SIZE, OUTPUT_SIZE, LR)
    gru = GRU(input_size, HIDDEN_SIZE, OUTPUT_SIZE, LR)
    lstm = LSTM(input_size, HIDDEN_SIZE, OUTPUT_SIZE, LR)

    losses = {}

    losses["RNN"] = train_model(rnn, X_train, y_train, EPOCHS, "RNN")
    losses["GRU"] = train_model(gru, X_train, y_train, EPOCHS, "GRU")
    losses["LSTM"] = train_model(lstm, X_train, y_train, EPOCHS, "LSTM")

    rnn_pred_scaled = predict(rnn, X_test)
    gru_pred_scaled = predict(gru, X_test)
    lstm_pred_scaled = predict(lstm, X_test)

    # Обратное преобразование только для cnt.
    # cnt — это первый признак, поэтому mean[0] и std[0].
    y_test_real = y_test.flatten() * std[0] + mean[0]
    rnn_pred = rnn_pred_scaled * std[0] + mean[0]
    gru_pred = gru_pred_scaled * std[0] + mean[0]
    lstm_pred = lstm_pred_scaled * std[0] + mean[0]

    predictions = {
        "RNN": rnn_pred,
        "GRU": gru_pred,
        "LSTM": lstm_pred,
    }

    print("\nРЕЗУЛЬТАТЫ НА TEST")
    print(f"RNN  MSE: {mse(y_test_real, rnn_pred):.3f} | RMSE: {rmse(y_test_real, rnn_pred):.3f} | MAE: {mae(y_test_real, rnn_pred):.3f} | R2: {r2_score_manual(y_test_real, rnn_pred):.3f}")
    print(f"GRU  MSE: {mse(y_test_real, gru_pred):.3f} | RMSE: {rmse(y_test_real, gru_pred):.3f} | MAE: {mae(y_test_real, gru_pred):.3f} | R2: {r2_score_manual(y_test_real, gru_pred):.3f}")
    print(f"LSTM MSE: {mse(y_test_real, lstm_pred):.3f} | RMSE: {rmse(y_test_real, lstm_pred):.3f} | MAE: {mae(y_test_real, lstm_pred):.3f} | R2: {r2_score_manual(y_test_real, lstm_pred):.3f}")

    plot_losses(losses)
    plot_all_predictions(y_test_real, predictions, count=200)
    plot_single_predictions(y_test_real, predictions, count=200)


if __name__ == "__main__":
    main()

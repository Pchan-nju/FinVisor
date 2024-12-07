import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# 超参数设置
DAYS_FOR_TRAIN = 30  # 使用多少天的数据来预测未来的价格
HIDDEN_SIZE = 64  # LSTM 隐藏单元数
NUM_LAYERS = 2  # LSTM 层数
EPOCHS = 100  # 训练轮次
LEARNING_RATE = 0.001  # 学习率
BATCH_SIZE = 32  # 批量大小


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout_rate=0.2):
        super(LSTMModel, self).__init__()

        # LSTM 层，加入 Dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

        # 全连接层，加入 Dropout
        self.fc = nn.Linear(hidden_size, output_size)

        # Dropout 层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 经过 LSTM 层
        out, (hn, cn) = self.lstm(x)

        # 只取最后一个时间步的输出，进行 Dropout
        out = out[:, -1, :]
        out = self.dropout(out)

        # 经过全连接层
        out = self.fc(out)
        return out


# 数据预处理
def preprocess_data(data, days_for_train):
    """
    对时间序列数据进行归一化和样本构造
    """
    scaler = MinMaxScaler(feature_range=(0, 1))  # 数据归一化
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data_scaled) - days_for_train):
        X.append(data_scaled[i:i + days_for_train])
        y.append(data_scaled[i + days_for_train])

    X, y = np.array(X), np.array(y)

    # 检查数据是否为空
    if X.shape[0] == 0:
        print("没有足够的数据样本来进行训练。")
        return X, y, scaler

    # 确保 X 是三维数组：样本数、时间步长、特征数
    if X.ndim == 2:  # 只有二维时，reshape 为三维
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # 重新调整数据形状

    # 打印调试信息
    print(f"Data shape after preprocessing: X = {X.shape}, y = {y.shape}")

    return X, y, scaler


# 模型训练函数
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    """
    训练 LSTM 模型
    """
    model.train()
    print("training on", device)
    criterion = criterion.to(device)
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.10f}")

    return model


# 主程序
def run_LSTM(data_file_path, stock_id):
    """
    Args:
        data_file_path: 数据文件的路径
        stock_id: 股票的编号 比如000031.SZ

    Returns:

    """
    try:
        data = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"文件未找到，请确保路径 {data_file_path} 存在")
        return
    except pd.errors.ParserError:
        print(f"解析错误，请检查文件格式 {data_file_path}")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 提取收盘价
    data = data[['close']]
    print("data", data)

    # 数据预处理
    X, y, scaler = preprocess_data(data.values, DAYS_FOR_TRAIN)

    # 检查数据样本数
    if len(X) == 0:
        print("数据样本数不足，无法继续训练。")
        return

    # 打印数据的形状
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 转换为 Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, DAYS_FOR_TRAIN, 1)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, DAYS_FOR_TRAIN, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 检查批次大小
    if X_train.shape[0] < BATCH_SIZE:
        print(f"警告：批次大小 {BATCH_SIZE} 大于训练样本数 {X_train.shape[0]}，可能导致训练失败。")

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 初始化模型、损失函数和优化器
    model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练模型
    print("Training the LSTM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_model(model, train_loader, criterion, optimizer, EPOCHS, device)

    # 保存完整模型到本地
    stock_id = stock_id.replace('.', '_')
    model_save_path = "model/LSTM" + "_" + stock_id + ".pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到 {model_save_path}")

    # 加载模型
    print("Loading the saved LSTM model...")
    model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.eval()

    # 测试模型
    print("Evaluating the LSTM model...")
    with torch.no_grad():
        predictions = model(X_test).numpy()
        y_test = y_test.numpy()

    # 反归一化
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 计算误差
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse:.4f}")

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test, label="Actual", color="blue")
    plt.plot(range(len(predictions)), predictions, label="Predicted", color="red")
    plt.legend()
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.savefig("results/images/" + stock_id + ".jpg")
    plt.show()

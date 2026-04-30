import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ==========================================
# 1. 数据预处理模块
# ==========================================
def load_and_preprocess(filepath, scaler_X=None, scaler_y=None, is_train=True, train_cols=None):
    # 1. 读取数据
    df = pd.read_csv(filepath)

    # 统一清洗列名：转小写、去空格
    df.columns = df.columns.str.lower().str.strip()

    # 2. 移除行号（如果有）
    if 'no' in df.columns:
        df.drop('no', axis=1, inplace=True)

    # 3. 处理日期索引
    # 训练集通常有日期，测试集可能没有。如果没有，我们就不设索引，直接用默认数字编号
    date_cols = ['year', 'month', 'day', 'hour']
    if 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
        df.set_index('datetime', inplace=True)
        df.drop('date', axis=1, inplace=True, errors='ignore')
    elif all(col in df.columns for col in date_cols):
        df['datetime'] = pd.to_datetime(df[date_cols])
        df.set_index('datetime', inplace=True)
        df.drop(date_cols, axis=1, inplace=True, errors='ignore')
    else:
        # 测试集 (data1) 走这里：没有日期列，打印提示但继续运行，返回的对象不再是 None
        print(f"提示：文件 {filepath} 采用自动生成的数字索引。")

    # 4. 确定目标列 (自动识别 pollution 或 pm2.5)
    target_col = 'pollution' if 'pollution' in df.columns else ('pm2.5' if 'pm2.5' in df.columns else None)
    if target_col is None:
        raise ValueError(f"错误：在 {filepath} 中找不到目标列。")

    # 5. 处理分类变量 (风向)
    cat_col = 'wnd_dir' if 'wnd_dir' in df.columns else ('cbwd' if 'cbwd' in df.columns else None)
    if cat_col:
        # 为了保证训练和测试的编码一致，这里简单处理，实际比赛建议用固定的映射
        df[cat_col] = pd.Categorical(df[cat_col]).codes

    # 6. 【核心步骤】特征对齐
    y_raw = df[[target_col]].values
    X_df = df.drop(columns=[target_col])

    if is_train:
        # 记录训练集特征的原始顺序（例如：dew, temp, press...）
        train_cols = X_df.columns.tolist()
    else:
        # 测试集强制按照训练集的列顺序重新排列，防止列位置错乱导致预测失败
        for col in train_cols:
            if col not in X_df.columns:
                X_df[col] = 0  # 缺失列补0
        X_df = X_df[train_cols]

    X_raw = X_df.values

    # 7. 归一化处理
    if is_train:
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler_X.fit_transform(X_raw)
        y_scaled = scaler_y.fit_transform(y_raw)
    else:
        X_scaled = scaler_X.transform(X_raw)
        y_scaled = scaler_y.transform(y_raw)

    return X_scaled, y_scaled, scaler_X, scaler_y, train_cols


# 构建滑动窗口序列 (监督学习数据格式)
def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)


# ==========================================
# 2. 模型定义 (网络布局)
# ==========================================
class MultivariateLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob):
        super(MultivariateLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM 核心层
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0
        )

        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # 传递给 LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # 仅提取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


# ==========================================
# 3. 主流程：训练与评估
# ==========================================
SEQ_LENGTH = 24  # 使用过去 24 小时的数据预测下一小时
BATCH_SIZE = 64
EPOCHS = 50
HIDDEN_DIM = 64
NUM_LAYERS = 2

# 加载主数据集 (请确保文件名一致)
X_train_scaled, y_train_scaled, scaler_X, scaler_y, train_cols = load_and_preprocess(
    'LSTM-Multivariate_pollution.csv', is_train=True
)

X_seq, y_seq = create_sequences(X_train_scaled, y_train_scaled, SEQ_LENGTH)

# 转换为 PyTorch 张量
X_tensor = torch.from_numpy(X_seq).float()
y_tensor = torch.from_numpy(y_seq).float()

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型
input_dim = X_train_scaled.shape[1]
model = MultivariateLSTM(input_dim, HIDDEN_DIM, NUM_LAYERS, output_dim=1, dropout_prob=0.2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
print("开始模型训练...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss / len(dataloader):.6f}')

# ==========================================
# 4. 独立测试集验证 (使用 data1)
# ==========================================
print("\n开始在测试集上验证...")
try:
    X_test_scaled, y_test_scaled, _, _, _ = load_and_preprocess(
        'pollution_test_data1.csv',
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        is_train=False,
        train_cols=train_cols
    )

    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQ_LENGTH)

    X_test_tensor = torch.from_numpy(X_test_seq).float()
    y_test_tensor = torch.from_numpy(y_test_seq).float()

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)

    # 逆归一化，恢复真实浓度值
    preds_inv = scaler_y.inverse_transform(test_predictions.numpy())
    actuals_inv = scaler_y.inverse_transform(y_test_tensor.numpy())

    # 结果可视化
    plt.figure(figsize=(12, 5))
    plt.plot(actuals_inv, label='Actual PM2.5')
    plt.plot(preds_inv, label='Predicted PM2.5', alpha=0.8)
    plt.title('Test Dataset (data1) - PM2.5 Forecasting')
    plt.xlabel('Time Steps')
    plt.ylabel('PM2.5 Concentration')
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"处理测试集时遇到问题，请确保 'pollution_test_data1.csv' 与主数据集字段一致。错误信息: {e}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
data = pd.read_csv('sensor_data_denoised.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 选择湿度作为预测目标（可替换为temperature_denoised预测温度）
target = 'humidity_denoised'
data_target = data[[target]].values

# 归一化到[0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_target)

# 构建LSTM输入数据：look_back为时间步（使用前24步预测下一步，对应12小时）
look_back = 24
X, y = [], []
for i in range(look_back, len(data_scaled)):
    X.append(data_scaled[i-look_back:i, 0])  # 前look_back个时间步的特征
    y.append(data_scaled[i, 0])  # 第i个时间步的目标值

X = np.array(X)
y = np.array(y)

# 重塑输入为LSTM要求的格式：[samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 划分训练集和测试集（80%训练，20%测试）
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 构建LSTM模型
model = Sequential()
# 第一层LSTM+Dropout
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
# 第二层LSTM+Dropout
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
# 全连接层输出
model.add(Dense(units=25))
model.add(Dense(units=1))  # 最终预测输出

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 早停法防止过拟合
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练模型
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# 预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反归一化到原始尺度
train_predict = scaler.inverse_transform(train_predict)
y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# 模型评估
train_mse = mean_squared_error(y_train_original, train_predict)
test_mse = mean_squared_error(y_test_original, test_predict)
train_r2 = r2_score(y_train_original, train_predict)
test_r2 = r2_score(y_test_original, test_predict)

print("LSTM模型评估（湿度预测）：")
print(f"训练集MSE: {train_mse:.2f}")
print(f"测试集MSE: {test_mse:.2f}")
print(f"训练集R²: {train_r2:.3f}")
print(f"测试集R²: {test_r2:.3f}")

# 可视化训练历史
plt.figure(figsize=(14, 10))

# 训练损失曲线
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'], label='训练损失', color='blue')
plt.plot(history.history['val_loss'], label='验证损失', color='red')
plt.title('LSTM模型训练历史')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

# 预测结果对比
plt.subplot(2, 1, 2)
# 构建完整的时间序列用于可视化
train_predict_plot = np.empty_like(data_scaled)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:look_back+len(train_predict), :] = train_predict

test_predict_plot = np.empty_like(data_scaled)
test_predict_plot[:, :] = np.nan
test_predict_plot[look_back+len(train_predict):len(data_scaled), :] = test_predict

# 绘制原始数据和预测数据
plt.plot(scaler.inverse_transform(data_scaled), label='原始湿度', color='green', linewidth=2)
plt.plot(train_predict_plot, label='训练集预测', color='orange', linestyle='--')
plt.plot(test_predict_plot, label='测试集预测', color='red', linestyle='--')
plt.title('LSTM模型周期预测效果')
plt.xlabel('时间步（30分钟/步）')
plt.ylabel('湿度 (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_result.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存模型和缩放器
model.save('lstm_model.h5')
import joblib
joblib.dump(scaler, 'lstm_scaler.pkl')
print("\nLSTM模型已保存！")
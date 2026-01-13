import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据并预处理
data = pd.read_csv('sensor_data_denoised.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 归一化数据到[0,1]
scaler_humidity = MinMaxScaler(feature_range=(0, 1))
scaler_temp = MinMaxScaler(feature_range=(0, 1))

data['humidity_norm'] = scaler_humidity.fit_transform(data[['humidity_denoised']])
data['temperature_norm'] = scaler_temp.fit_transform(data[['temperature_denoised']])

# 构建特征：使用前n个时间步预测下一个时间步（n=24，对应12小时前的数据）
look_back = 24  # 30分钟/步 × 24步 = 12小时
X, y = [], []

# 以湿度为例进行预测（温度预测可同理修改）
for i in range(look_back, len(data)):
    X.append(data['humidity_norm'].iloc[i-look_back:i].values)  # 前24个时间步的湿度
    y.append(data['humidity_norm'].iloc[i])  # 当前时间步的湿度

X = np.array(X)
y = np.array(y)

# 重塑特征为线性回归输入格式 (samples, features)
X = X.reshape(-1, look_back)

# 划分训练集和测试集（80%训练，20%测试）
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 训练线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 预测
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# 反归一化到原始尺度
y_train_pred_original = scaler_humidity.inverse_transform(y_train_pred.reshape(-1, 1))
y_test_pred_original = scaler_humidity.inverse_transform(y_test_pred.reshape(-1, 1))
y_train_original = scaler_humidity.inverse_transform(y_train.reshape(-1, 1))
y_test_original = scaler_humidity.inverse_transform(y_test.reshape(-1, 1))

# 模型评估
train_mse = mean_squared_error(y_train_original, y_train_pred_original)
test_mse = mean_squared_error(y_test_original, y_test_pred_original)
train_r2 = r2_score(y_train_original, y_train_pred_original)
test_r2 = r2_score(y_test_original, y_test_pred_original)

print("线性回归模型评估（湿度预测）：")
print(f"训练集MSE: {train_mse:.2f}")
print(f"测试集MSE: {test_mse:.2f}")
print(f"训练集R²: {train_r2:.3f}")
print(f"测试集R²: {test_r2:.3f}")

# 可视化结果
plt.figure(figsize=(14, 6))

# 训练集拟合效果
plt.subplot(1, 2, 1)
plt.plot(y_train_original, label='真实湿度', color='blue')
plt.plot(y_train_pred_original, label='预测湿度', color='red', linestyle='--')
plt.title('训练集：线性回归拟合效果')
plt.xlabel('时间步')
plt.ylabel('湿度 (%)')
plt.legend()
plt.grid(True, alpha=0.3)

# 测试集预测效果
plt.subplot(1, 2, 2)
plt.plot(y_test_original, label='真实湿度', color='blue')
plt.plot(y_test_pred_original, label='预测湿度', color='red', linestyle='--')
plt.title('测试集：12小时湿度预测效果')
plt.xlabel('时间步')
plt.ylabel('湿度 (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_result.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存模型
import joblib
joblib.dump(lr_model, 'linear_regression_model.pkl')
joblib.dump(scaler_humidity, 'humidity_scaler.pkl')
print("\n线性回归模型已保存！")
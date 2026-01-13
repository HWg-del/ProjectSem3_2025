import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
data = pd.read_csv('sensor_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 方法1：移动平均滤波（窗口大小=3）
data['humidity_denoised'] = data['humidity'].rolling(window=3, center=True).mean()
data['temperature_denoised'] = data['temperature'].rolling(window=3, center=True).mean()

# 填充首尾缺失值（窗口边缘）
data['humidity_denoised'] = data['humidity_denoised'].fillna(method='bfill').fillna(method='ffill')
data['temperature_denoised'] = data['temperature_denoised'].fillna(method='bfill').fillna(method='ffill')

# 方法2：中值滤波（可选，如需对比）
# data['humidity_denoised_median'] = data['humidity'].rolling(window=3, center=True).median()

# 可视化去噪效果
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 湿度去噪对比
ax1.plot(data['timestamp'], data['humidity'], label='原始湿度', color='gray', alpha=0.6)
ax1.plot(data['timestamp'], data['humidity_denoised'], label='去噪后湿度', color='blue', linewidth=2)
ax1.set_title('湿度数据去噪效果（移动平均滤波）')
ax1.set_ylabel('湿度 (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 温度去噪对比
ax2.plot(data['timestamp'], data['temperature'], label='原始温度', color='gray', alpha=0.6)
ax2.plot(data['timestamp'], data['temperature_denoised'], label='去噪后温度', color='red', linewidth=2)
ax2.set_title('温度数据去噪效果（移动平均滤波）')
ax2.set_xlabel('时间')
ax2.set_ylabel('温度 (℃)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('denoising_result.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存去噪后的数据
data.to_csv('sensor_data_denoised.csv', index=False)
print("数据去噪完成！")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载去噪后的数据
data = pd.read_csv('sensor_data_denoised.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 1. 描述性统计
desc_stats = data[['temperature', 'temperature_denoised', 'humidity', 'humidity_denoised']].describe()
print("描述性统计结果：")
print(desc_stats.round(2))

# 2. 时间序列趋势可视化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 湿度时间序列
ax1.plot(data['timestamp'], data['humidity_denoised'], color='green', linewidth=2, label='去噪后湿度')
ax1.fill_between(data['timestamp'], data['humidity_denoised'], alpha=0.3, color='green')
ax1.set_title('土壤湿度时间序列趋势（2天监测）')
ax1.set_ylabel('湿度 (%)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 温度时间序列
ax2.plot(data['timestamp'], data['temperature_denoised'], color='orange', linewidth=2, label='去噪后温度')
ax2.fill_between(data['timestamp'], data['temperature_denoised'], alpha=0.3, color='orange')
ax2.set_title('环境温度时间序列趋势（2天监测）')
ax2.set_xlabel('时间')
ax2.set_ylabel('温度 (℃)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('time_series_trend.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 相关性分析
correlation = data[['temperature_denoised', 'humidity_denoised']].corr()
print("\n温度与湿度相关性：")
print(correlation.round(3))

# 4. 分布直方图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.hist(data['humidity_denoised'], bins=15, color='green', alpha=0.7, edgecolor='black')
ax1.set_title('湿度分布直方图')
ax1.set_xlabel('湿度 (%)')
ax1.set_ylabel('频数')
ax1.grid(True, alpha=0.3)

ax2.hist(data['temperature_denoised'], bins=15, color='orange', alpha=0.7, edgecolor='black')
ax2.set_title('温度分布直方图')
ax2.set_xlabel('温度 (℃)')
ax2.set_ylabel('频数')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存统计结果
desc_stats.to_csv('descriptive_statistics.csv', encoding='utf-8-sig')
print("\n总体分析完成！结果已保存")
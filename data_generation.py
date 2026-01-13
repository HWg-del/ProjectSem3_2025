import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 设置随机种子以保证可重复性
np.random.seed(42)

# 生成时间序列：每30分钟采集一次，共2天（48小时），96个数据点
start_time = datetime(2025, 1, 1, 0, 0)
time_points = [start_time + timedelta(minutes=30*i) for i in range(96)]

# 模拟湿度数据（50%-80%范围，加入趋势、周期和噪声）
# 基础趋势：逐渐下降（模拟不浇水的情况）
humidity_trend = np.linspace(75, 55, 96)
# 周期波动：昼夜变化
humidity_cycle = 3 * np.sin(np.linspace(0, 4*np.pi, 96))  # 2天=4个半周期
# 随机噪声
humidity_noise = np.random.normal(0, 2.5, 96)
# 最终湿度数据（限制在0-100范围内）
humidity = humidity_trend + humidity_cycle + humidity_noise
humidity = np.clip(humidity, 0, 100)

# 模拟温度数据（18℃-28℃范围，加入周期和噪声）
temperature_trend = np.linspace(22, 24, 96)  # 轻微上升趋势
temperature_cycle = 2 * np.cos(np.linspace(0, 4*np.pi, 96))  # 昼夜变化
temperature_noise = np.random.normal(0, 0.8, 96)
temperature = temperature_trend + temperature_cycle + temperature_noise
temperature = np.clip(temperature, 15, 30)

# 创建DataFrame
data = pd.DataFrame({
    'timestamp': time_points,
    'temperature': temperature,
    'humidity': humidity
})

# 保存为CSV文件
data.to_csv('sensor_data.csv', index=False)
print("模拟数据生成完成！")
print(f"数据维度：{data.shape}")
print("\n前5行数据：")
print(data.head())
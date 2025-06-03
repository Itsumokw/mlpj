import numpy as np
from core.models import SimpleARIMA
from utils.dataloader import DataPipeline
from utils.visualizer import ForecastVisualizer

# 生成示例数据
data = np.sin(np.linspace(0, 20, 1000)) + np.random.normal(0, 0.1, 1000)

# 数据预处理
pipeline = DataPipeline(window_size=30)
features, targets = pipeline.process(data)

# 训练预测
model = SimpleARIMA(p=5)
model.fit(targets[:800])  # 前800个样本训练
preds = model.predict(targets[800-5:800], steps=200)  # 预测后200步

# 可视化
ForecastVisualizer.plot_result(targets[800:], preds, save_path='result.png')
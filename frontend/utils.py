# 通用工具函数
import pandas as pd

def format_large_number(n):
    """格式化大数字为易读格式"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def load_default_dataset() -> pd.DataFrame:
    """加载内置的AirPassengers数据集"""
    data_path = './datasets/rakannimer/air-passengers/versions/1/AirPassengers.csv'
    df = pd.read_csv(data_path)
    # 确保列名正确

    df.columns = ['Month', '#Passengers']
    return df
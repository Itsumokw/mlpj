# 通用工具函数

def format_large_number(n):
    """格式化大数字为易读格式"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)
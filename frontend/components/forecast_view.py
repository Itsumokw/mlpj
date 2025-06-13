import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def show_forecast_results(results):
    """展示预测结果"""
    st.subheader("🔮 Forecasting Results")

    # 显示预测表格
    forecast_df = pd.DataFrame({
        'Period': results['forecast_dates'],
        'Forecasted Value': results['forecast_values']
    })
    st.dataframe(forecast_df.style.format({"Forecasted Value": "{:,.2f}"}))

    # 绘制预测图表
    fig, ax = plt.subplots(figsize=(12, 6))

    # 历史数据
    if results['history_dates'] and results['history_values']:
        ax.plot(results['history_dates'], results['history_values'], 'b-', label='Historical Data')

    # 预测数据
    if results['forecast_dates'] and results['forecast_values']:
        ax.plot(results['forecast_dates'], results['forecast_values'], 'r--', label='Forecast')

    # 添加预测起始线
    if results['history_dates'] and results['forecast_dates']:
        last_date = results['history_dates'][-1] if results['history_dates'] else None
        first_forecast = results['forecast_dates'][0]

        if last_date:
            ax.axvline(x=last_date, color='g', linestyle='-', alpha=0.5, label='Forecast Start')
        elif first_forecast:
            ax.axvline(x=first_forecast, color='g', linestyle='-', alpha=0.5, label='Forecast Start')

    ax.set_title(f"{len(results['forecast_values'])}-Period Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    # 旋转日期标签
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    # 显示预测统计信息
    if 'forecast_stats' in results:
        st.subheader("Forecast Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Forecast", f"{results['forecast_stats']['mean']:.2f}")
        col2.metric("Min Forecast", f"{results['forecast_stats']['min']:.2f}")
        col3.metric("Max Forecast", f"{results['forecast_stats']['max']:.2f}")
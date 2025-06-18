import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from frontend.utils import load_default_dataset


def load_data(config):
    """根据配置加载数据集（默认或自定义）"""
    dataset_name = config.get('dataset', 'Unknown Dataset')
    time_col = config.get('time_col', None)
    value_col = config.get('value_col', None)
    custom_data = config.get('custom_data', None)

    df = None

    # 加载默认数据集
    if dataset_name == "Air Passengers (Default)":
        try:
            df = load_default_dataset()
            # 确保列名正确
            if 'Month' not in df.columns or '#Passengers' not in df.columns:
                df.columns = ['Month', '#Passengers']
        except Exception as e:
            st.error(f"Failed to load default dataset: {str(e)}")
            return None
    # 加载自定义数据集
    elif dataset_name == "Upload Custom Dataset" and custom_data:
        try:
            df = pd.DataFrame(custom_data)

            # 尝试转换时间列为datetime格式
            if time_col in df.columns:
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                except:
                    pass

            # 确保数值列为数值类型
            if value_col in df.columns:
                try:
                    df[value_col] = pd.to_numeric(df[value_col])
                except:
                    pass
        except Exception as e:
            st.error(f"Failed to create DataFrame: {str(e)}")
            return None

    return df


def show_data_preview(config):
    """展示数据预览，支持默认数据集和自定义数据集"""
    st.subheader("📊 Dataset Preview")

    # 加载数据
    df = load_data(config)

    if df is None:
        st.info("No data available. Please configure dataset in sidebar.")
        return

    # 安全获取配置值
    dataset_name = config.get('dataset', 'Unknown Dataset')
    time_col = config.get('time_col', None)
    value_col = config.get('value_col', None)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"**Dataset:** `{dataset_name}`")
        if dataset_name == "Air Passengers (Default)":
            st.info("Built-in Air Passengers dataset (1949-1960)")
        else:
            st.info("Custom uploaded dataset")

        # 显示列名
        if time_col:
            st.markdown(f"**Time Column:** `{time_col}`")
        if value_col:
            st.markdown(f"**Value Column:** `{value_col}`")

        # 显示数据预览
        st.dataframe(df.head(5))
        st.markdown(f"**Rows:** {len(df)}, **Columns:** {len(df.columns)}")

    with col2:
        # 确保必要的列存在
        if not time_col or not value_col:
            st.warning("Time and value columns not configured")
            return

        if time_col not in df.columns:
            st.warning(f"Time column '{time_col}' not found in dataset")
            st.info(f"Available columns: {', '.join(df.columns)}")
            return

        if value_col not in df.columns:
            st.warning(f"Value column '{value_col}' not found in dataset")
            st.info(f"Available columns: {', '.join(df.columns)}")
            return

        try:
            # 创建交互式图表
            fig = go.Figure()

            # 尝试使用时间列作为x轴
            if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                fig.add_trace(go.Scatter(
                    x=df[time_col],
                    y=df[value_col],
                    mode='lines',
                    name='Time Series'
                ))
                fig.update_xaxes(title_text=time_col)
            else:
                # 如果不是时间类型，使用索引
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[value_col],
                    mode='lines',
                    name='Time Series'
                ))
                fig.update_xaxes(title_text="Index")

            fig.update_yaxes(title_text=value_col)
            fig.update_layout(
                title=f"{dataset_name} Preview",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error plotting data: {str(e)}")


def show_differenced_data(config):
    """展示时间序列的差分数据及相关分析"""
    if not config.get('show_diff', False):
        return

    # 加载数据
    df = load_data(config)

    if df is None:
        st.warning("No data available to show differenced data")
        return

    # 获取必要配置
    time_col = config.get('time_col', None)
    value_col = config.get('value_col', None)

    if not time_col or not value_col:
        st.warning("Time and value columns not configured")
        return

    if time_col not in df.columns or value_col not in df.columns:
        st.warning("Required columns not found in dataset")
        return

    # 确保数据按时间排序
    df = df.sort_values(time_col).reset_index(drop=True)

    st.subheader("🔢 Differenced Data Analysis")

    # 计算一阶差分
    df['diff_1'] = df[value_col].diff()

    # 计算二阶差分
    df['diff_2'] = df['diff_1'].diff()

    # 创建Plotly图表对象
    fig = go.Figure()

    # 添加原始数据轨迹
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df[value_col],
        name='原始数据',
        line=dict(color='#1f77b4'),
        visible='legendonly'
    ))

    # 添加一阶差分轨迹
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df['diff_1'],
        name='一阶差分',
        line=dict(color='#ff7f0e')
    ))

    # 添加二阶差分轨迹
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df['diff_2'],
        name='二阶差分',
        line=dict(color='#2ca02c')
    ))

    # 更新图表布局
    fig.update_layout(
        title='时间序列差分分析',
        xaxis_title='时间',
        yaxis_title='值',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        template='plotly_white',
        height=500
    )

    # 显示图表
    st.plotly_chart(fig, use_container_width=True)

    # 创建统计信息标签页
    tab1, tab2, tab3 = st.tabs(["一阶差分分析", "二阶差分分析", "平稳性检验"])

    with tab1:
        st.subheader("一阶差分统计信息")
        diff1_stats = df['diff_1'].describe().to_frame().T
        st.dataframe(diff1_stats.style.format("{:.2f}"), use_container_width=True)

        # 绘制一阶差分分布图
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=df['diff_1'].dropna(),
            name='分布',
            marker_color='#ff7f0e',
            opacity=0.7
        ))
        fig1.update_layout(
            title='一阶差分分布',
            xaxis_title='差分值',
            yaxis_title='频数',
            bargap=0.1,
            height=300
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 自相关图（修复后的实现）
        st.subheader("一阶差分自相关图")
        try:
            # 计算不同滞后值的自相关系数
            lags = range(1, 25)
            autocorrs = [df['diff_1'].dropna().autocorr(lag=lag) for lag in lags]
            autocorr_series = pd.Series(autocorrs, index=lags, name='Autocorrelation')

            # 绘制自相关图
            fig_acf1 = go.Figure()
            fig_acf1.add_trace(go.Scatter(
                x=autocorr_series.index,
                y=autocorr_series.values,
                mode='lines+markers',
                name='ACF'
            ))
            fig_acf1.update_layout(
                title='自相关函数 (ACF)',
                xaxis_title='滞后值',
                yaxis_title='自相关系数',
                height=300
            )
            st.plotly_chart(fig_acf1, use_container_width=True)
        except Exception as e:
            st.error(f"无法计算自相关图: {str(e)}")

    with tab2:
        st.subheader("二阶差分统计信息")
        diff2_stats = df['diff_2'].describe().to_frame().T
        st.dataframe(diff2_stats.style.format("{:.2f}"), use_container_width=True)

        # 绘制二阶差分分布图
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=df['diff_2'].dropna(),
            name='分布',
            marker_color='#2ca02c',
            opacity=0.7
        ))
        fig2.update_layout(
            title='二阶差分分布',
            xaxis_title='差分值',
            yaxis_title='频数',
            bargap=0.1,
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 自相关图（修复后的实现）
        st.subheader("二阶差分自相关图")
        try:
            # 计算不同滞后值的自相关系数
            lags = range(1, 25)
            autocorrs = [df['diff_2'].dropna().autocorr(lag=lag) for lag in lags]
            autocorr_series = pd.Series(autocorrs, index=lags, name='Autocorrelation')

            # 绘制自相关图
            fig_acf2 = go.Figure()
            fig_acf2.add_trace(go.Scatter(
                x=autocorr_series.index,
                y=autocorr_series.values,
                mode='lines+markers',
                name='ACF'
            ))
            fig_acf2.update_layout(
                title='自相关函数 (ACF)',
                xaxis_title='滞后值',
                yaxis_title='自相关系数',
                height=300
            )
            st.plotly_chart(fig_acf2, use_container_width=True)
        except Exception as e:
            st.error(f"无法计算自相关图: {str(e)}")

    with tab3:
        st.subheader("平稳性检验")

        # 原始数据ADF检验
        st.markdown("**原始数据平稳性检验 (ADF Test)**")
        result_orig = adfuller(df[value_col].dropna())
        st.write(f"ADF统计量: **{result_orig[0]:.4f}**")
        st.write(f"p值: **{result_orig[1]:.4f}**")
        st.write(f"临界值:")
        for key, value in result_orig[4].items():
            st.write(f"{key}: {value:.4f}")

        # 一阶差分ADF检验
        st.divider()
        st.markdown("**一阶差分平稳性检验 (ADF Test)**")
        result_diff1 = adfuller(df['diff_1'].dropna())
        st.write(f"ADF统计量: **{result_diff1[0]:.4f}**")
        st.write(f"p值: **{result_diff1[1]:.4f}**")
        st.write(f"临界值:")
        for key, value in result_diff1[4].items():
            st.write(f"{key}: {value:.4f}")

        # 二阶差分ADF检验
        st.divider()
        st.markdown("**二阶差分平稳性检验 (ADF Test)**")
        result_diff2 = adfuller(df['diff_2'].dropna())
        st.write(f"ADF统计量: **{result_diff2[0]:.4f}**")
        st.write(f"p值: **{result_diff2[1]:.4f}**")
        st.write(f"临界值:")
        for key, value in result_diff2[4].items():
            st.write(f"{key}: {value:.4f}")

        # 平稳性解释
        st.divider()
        st.markdown("**平稳性解释**")
        st.info("""
        - **p值 < 0.05**: 数据在95%置信水平下平稳
        - **ADF统计量 < 临界值**: 拒绝原假设（非平稳），数据可能是平稳的
        - 通常需要结合图表和统计量共同判断
        """)

    # 添加差分解释信息
    st.divider()
    with st.expander("📊 差分数据说明"):
        st.markdown("""
        **差分**是时间序列分析中常用的预处理技术，用于：
        - 消除时间序列的趋势性
        - 使非平稳数据变得平稳
        - 满足模型对平稳性的要求

        **计算公式:**
        - 一阶差分: `Y'_t = Y_t - Y_{t-1}`
        - 二阶差分: `Y''_t = Y'_t - Y'_{t-1} = (Y_t - Y_{t-1}) - (Y_{t-1} - Y_{t-2})`

        **使用建议:**
        1. 检查原始数据的平稳性
        2. 如果非平稳，尝试一阶差分
        3. 如果一阶差分后仍非平稳，尝试二阶差分
        4. 避免过度差分（通常不超过二阶）
        """)
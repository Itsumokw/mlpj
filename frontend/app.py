import streamlit as st
import time
from components.sidebar import render_sidebar
from components.data_view import show_data_preview
from components.training_view import show_training_in_progress, show_training_results
from components.forecast_view import show_forecast_results
from api_client import preprocess_data, train_model, get_training_status, forecast

# 设置页面布局
st.set_page_config(layout="wide", page_title="Time Series Forecasting Lab")
st.title("⏳ Time Series Forecasting Lab")
st.markdown("""
Advanced platform for comparing various time series forecasting models.
Now featuring multiple neural network architectures for improved forecasting.
""")

# 初始化session state
if 'model_config' not in st.session_state:
    st.session_state.model_config = None

if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None

if 'training_job_id' not in st.session_state:
    st.session_state.training_job_id = None

if 'training_results' not in st.session_state:
    st.session_state.training_results = None

if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None

# 渲染侧边栏并获取配置
model_config = render_sidebar()

# 保存配置到session state
st.session_state.model_config = model_config

# 显示数据预览 - 添加安全检查
if st.session_state.model_config and st.session_state.model_config.get('show_data'):
    # 只有在配置完整时才尝试显示预览
    if 'time_col' in st.session_state.model_config and 'value_col' in st.session_state.model_config:
        show_data_preview(st.session_state.model_config)
    else:
        # 提供友好的提示信息
        if st.session_state.model_config.get('dataset') == "Upload Custom Dataset":
            st.info("Please upload a dataset and select columns in the sidebar to view data preview")
        else:
            st.info("Data preview will be available after configuration")

# 处理训练按钮
if st.session_state.get('train_button_clicked'):
    # 重置之前的结果
    st.session_state.training_results = None
    st.session_state.forecast_results = None

    # 步骤1: 数据预处理
    with st.spinner("Preprocessing data..."):
        preprocessed_data = preprocess_data(st.session_state.model_config)
        # 验证数据结构
        required_keys = ['X_train', 'y_train', 'X_test', 'y_test', 'scaler', 'last_values', 'dates']
        if not all(key in preprocessed_data for key in required_keys):
            st.error("Preprocessed data is missing required fields")
            

        if preprocessed_data:
            st.session_state.preprocessed_data = preprocessed_data
            st.success("Data preprocessing completed!")
        else:
            st.error("Data preprocessing failed. Please check the configuration.")
            st.session_state.train_button_clicked = False
            st.stop()

    # 步骤2: 训练模型
    with st.spinner("Starting training process..."):
        training_response = train_model(
            st.session_state.model_config,
            st.session_state.preprocessed_data
        )

        if training_response and 'job_id' in training_response:
            st.session_state.training_job_id = training_response['job_id']
            st.success(f"Training job started! Job ID: {st.session_state.training_job_id}")
        else:
            st.error("Failed to start training job.")
            st.session_state.train_button_clicked = False
            st.stop()

# 如果有一个正在进行的训练任务，轮询状态
if st.session_state.training_job_id and st.session_state.training_results is None:
    show_training_in_progress()

    # 轮询训练状态
    while True:
        status = get_training_status(st.session_state.training_job_id)

        if status['status'] == 'completed':
            st.session_state.training_results = status['results']
            st.rerun()  # 触发重新运行以更新显示
            break
        elif status['status'] == 'training':
            time.sleep(5)  # 每5秒轮询一次
        else:  # 错误或其他状态
            st.error(f"Training failed: {status.get('message', 'Unknown error')}")
            st.session_state.training_job_id = None
            break

# 显示训练结果
if st.session_state.training_results:
    show_training_results(st.session_state.training_results)

# 如果有训练结果，显示预测部分
if st.session_state.training_results:
    # 触发预测
    if st.session_state.model_config and st.session_state.training_results and not st.session_state.forecast_results:
        with st.spinner("Generating forecast..."):
            forecast_results = forecast(
                st.session_state.model_config,
                st.session_state.training_results['model_state']
            )
            if forecast_results:
                st.session_state.forecast_results = forecast_results

    # 显示预测结果
    if st.session_state.forecast_results:
        show_forecast_results(st.session_state.forecast_results)

# 添加模型路线图
st.sidebar.divider()
st.sidebar.subheader("Model Roadmap")
st.sidebar.markdown("""
- ✅ **Neural ARIMA** (Available now)
- ✅ **TCN Models** (Available now)
- ✅ **GRU/CNN Models** (Available now)
- ⏳ Transformer with Attention (Q3 2024)
""")

# 添加作者信息
st.sidebar.divider()
st.sidebar.info("""
**Developed by:** Time Series Research Lab  
**Version:** 2.0.0  
**Last Updated:** July 2024
""")
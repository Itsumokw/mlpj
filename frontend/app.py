import streamlit as st
import time
from components.sidebar import render_sidebar
from components.data_view import show_data_preview, show_differenced_data
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

# 存储配置快照用于检测变更
if 'config_snapshot' not in st.session_state:
    st.session_state.config_snapshot = None

# 渲染侧边栏并获取配置
model_config = render_sidebar()


# 检查关键配置是否变更
def config_changed(new_config, old_snapshot):
    """检查关键配置是否发生变更"""
    if old_snapshot is None:
        return True

    # 检查关键字段
    key_fields = ['dataset', 'time_col', 'value_col', 'model_type',
                  'window_size', 'horizon', 'use_exog', 'exog_columns']

    for field in key_fields:
        if field in new_config and field in old_snapshot:
            if new_config[field] != old_snapshot[field]:
                return True

    # 检查上传文件是否变更
    if new_config.get('dataset') == "Upload Custom Dataset":
        new_file = new_config.get('uploaded_file')
        old_file = old_snapshot.get('uploaded_file')

        # 检查文件是否变更
        if new_file and old_file:
            if new_file.name != old_file.name or new_file.size != old_file.size:
                return True
        elif new_file != old_file:  # 一个存在一个不存在
            return True

    return False


# 创建当前配置的快照（仅包含关键字段）
current_snapshot = {
    'dataset': model_config.get('dataset'),
    'time_col': model_config.get('time_col'),
    'value_col': model_config.get('value_col'),
    'model_type': model_config.get('model_type'),
    'window_size': model_config.get('window_size'),
    'horizon': model_config.get('horizon'),
    'use_exog': model_config.get('use_exog'),
    'exog_columns': model_config.get('exog_columns'),
    'uploaded_file': model_config.get('uploaded_file')
}

# 如果配置变更，重置训练和预测结果
if config_changed(model_config, st.session_state.config_snapshot):
    st.session_state.training_job_id = None
    st.session_state.training_results = None
    st.session_state.forecast_results = None
    st.session_state.preprocessed_data = None
    st.session_state.train_button_clicked = False

    # 更新配置快照
    st.session_state.config_snapshot = current_snapshot

# 保存配置到session state
st.session_state.model_config = model_config

# 显示数据预览
if st.session_state.model_config and st.session_state.model_config.get('show_data'):
    if 'time_col' in st.session_state.model_config and 'value_col' in st.session_state.model_config:
        show_data_preview(st.session_state.model_config)
    else:
        if st.session_state.model_config.get('dataset') == "Upload Custom Dataset":
            st.info("Please upload a dataset and select columns in the sidebar to view data preview")
        else:
            st.info("Data preview will be available after configuration")

# 显示差分数据
if st.session_state.model_config and st.session_state.model_config.get('show_diff'):
    if 'time_col' in st.session_state.model_config and 'value_col' in st.session_state.model_config:
        show_differenced_data(st.session_state.model_config)
    else:
        if st.session_state.model_config.get('dataset') == "Upload Custom Dataset":
            st.info("Please upload a dataset and select columns in the sidebar to view differenced data")
        else:
            st.info("Differenced data will be available after configuration")

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
import streamlit as st
import pandas as pd


def render_sidebar():
    """渲染侧边栏并返回配置字典"""
    config = {}

    with st.sidebar:
        st.header("⚙️ Model Configuration")

        # 更新模型选项列表
        model_options = {
            "Neural ARIMA": "ARIMA",
            "Lite TCN": "LiteTCN",
            "Enhanced TCN": "EnhancedTCN",
            "Advanced TCN": "AdvancedTCN",
            "Time GRU": "TimeGRU",
            "Time CNN": "TimeCNN",
            "Time Linear": "TimeLinear"
        }

        # 从session_state获取或初始化配置
        if 'model_config' in st.session_state and st.session_state.model_config:
            session_config = st.session_state.model_config

            # 确保session_config包含所有必要的键
            default_keys = {
                "kernel_size": 3,
                "num_layers": 3,
                "dropout": 0.1,
                "output_steps": 12,
                # 添加其他可能缺失的键
            }

            for key, default_value in default_keys.items():
                if key not in session_config:
                    session_config[key] = default_value
        else:
            # 扩展默认配置以支持新模型参数
            session_config = {
                "selected_model": "Neural ARIMA",
                "dataset": "Air Passengers (Default)",
                "custom_data": None,
                "time_col": "Month",
                "value_col": "#Passengers",
                "p": 12,
                "q": 1,
                "hidden_size": 64,
                "kernel_size": 3,  # 确保有这个键
                "num_layers": 3,
                "dropout": 0.1,
                "output_steps": 12,
                "lr": 0.005,
                "epochs": 500,
                "alpha": 0.1,
                "show_data": True,
                "show_diff": False,
                "forecast_months": 12
            }

        # 模型选择器
        selected_model = st.selectbox(
            "Select Model",
            list(model_options.keys()),
            index=list(model_options.keys()).index(session_config["selected_model"]),
            help="Select the model architecture for time series forecasting"
        )
        config["selected_model"] = selected_model
        st.markdown(f"**Selected Model:** `{selected_model}`")

        # 更新模型描述
        model_descriptions = {
            "Neural ARIMA": "Combines ARIMA concepts with neural networks. Handles seasonality and trends.",
            "Lite TCN": "Lightweight Temporal Convolutional Network for efficient forecasting",
            "Enhanced TCN": "TCN with residual connections and dilated convolutions for longer sequences",
            "Advanced TCN": "TCN with attention mechanisms for capturing complex patterns",
            "Time GRU": "GRU-based recurrent model for sequential time series data",
            "Time CNN": "Convolutional Neural Network specialized for time series",
            "Time Linear": "Linear model with time-based features"
        }
        st.info(model_descriptions[selected_model])
        st.divider()

        # 数据集选择
        st.subheader("Dataset Options")
        dataset_options = ["Air Passengers (Default)", "Upload Custom Dataset"]
        selected_dataset = st.selectbox(
            "Select Dataset",
            dataset_options,
            index=dataset_options.index(session_config["dataset"]),
            help="Choose between built-in datasets or upload your own"
        )
        config["dataset"] = selected_dataset

        # 自定义数据集处理
        if selected_dataset == "Upload Custom Dataset":
            st.info("Upload your own time series data (CSV format)")
            uploaded_file = st.file_uploader("Upload CSV", type="csv")

            if uploaded_file:
                try:
                    custom_data = pd.read_csv(uploaded_file)
                    if len(custom_data) > 0:
                        st.success(f"Successfully uploaded data with {len(custom_data)} rows")
                        st.dataframe(custom_data.head(3))

                        # 存储为字典列表（JSON友好格式）
                        config["custom_data"] = custom_data.to_dict(orient='records')

                        # 获取所有列名
                        columns = custom_data.columns.tolist()

                        # 选择时间列
                        # 确定默认索引
                        if "time_col" in session_config and session_config["time_col"] in columns:
                            time_index = columns.index(session_config["time_col"])
                        else:
                            time_index = 0

                        time_col = st.selectbox(
                            "Select Time Column",
                            columns,
                            index=time_index,
                            help="Column containing date/time information"
                        )
                        config["time_col"] = time_col

                        # 选择数值列
                        # 确定默认索引
                        if "value_col" in session_config and session_config["value_col"] in columns:
                            value_index = columns.index(session_config["value_col"])
                        else:
                            # 如果有多列，尝试选择第二列，否则选择第一列
                            if len(columns) > 1:
                                # 确保不选择时间列作为值列
                                value_index = 1 if 1 < len(columns) else 0
                            else:
                                value_index = 0

                        value_col = st.selectbox(
                            "Select Value Column",
                            columns,
                            index=value_index,
                            help="Column containing the values to forecast"
                        )
                        config["value_col"] = value_col

                        # 数值列格式验证
                        if not pd.api.types.is_numeric_dtype(custom_data[value_col]):
                            st.warning("Value column must contain numeric data. Attempting to convert...")
                            try:
                                custom_data[value_col] = pd.to_numeric(custom_data[value_col])
                                # 更新配置中的数据
                                config["custom_data"] = custom_data.to_dict(orient='records')
                            except:
                                st.error("Failed to convert value column to numeric. Please select a different column.")
                                config["custom_data"] = None
                    else:
                        st.warning("Uploaded file is empty")
                        config["custom_data"] = None
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    config["custom_data"] = None
            else:
                # 如果没有上传新文件，但之前有，则保留
                if session_config.get("custom_data"):
                    config["custom_data"] = session_config["custom_data"]
                    config["time_col"] = session_config["time_col"]
                    config["value_col"] = session_config["value_col"]
                else:
                    # 确保设置默认值，避免KeyError
                    config["time_col"] = session_config.get("time_col", "Column1")
                    config["value_col"] = session_config.get("value_col", "Column2")
                    config["custom_data"] = None
        else:
            # 使用默认数据集
            config["custom_data"] = None
            config["time_col"] = "Month"
            config["value_col"] = "#Passengers"

        st.divider()

        # 通用模型参数配置
        st.subheader("Model Parameters")

        # 所有模型通用的参数
        hidden_size = st.slider(
            "Hidden Layer Size", 16, 256,
            session_config.get("hidden_size", 64), 16,  # 使用get避免KeyError
            help="Size of hidden layers in neural network models"
        )
        config["hidden_size"] = hidden_size

        num_layers = st.slider(
            "Number of Layers", 1, 8,
            session_config.get("num_layers", 3), 1,  # 使用get避免KeyError
            help="Depth of the neural network"
        )
        config["num_layers"] = num_layers

        dropout = st.slider(
            "Dropout Rate", 0.0, 0.5,
            session_config.get("dropout", 0.1), 0.01,  # 使用get避免KeyError
            help="Regularization to prevent overfitting"
        )
        config["dropout"] = dropout

        # TCN模型专用参数
        if "TCN" in selected_model:
            kernel_size = st.slider(
                "Kernel Size", 3, 9,
                session_config.get("kernel_size", 3), 1,  # 使用get避免KeyError
                help="Size of convolutional filters in TCN models"
            )
            config["kernel_size"] = kernel_size

            # Enhanced和Advanced TCN需要额外参数
            if "Enhanced" in selected_model or "Advanced" in selected_model:
                output_steps = st.slider(
                    "Output Steps", 1, 36,
                    session_config.get("output_steps", 12), 1,  # 使用get避免KeyError
                    help="Number of future steps to predict simultaneously"
                )
                config["output_steps"] = output_steps

        # ARIMA模型专用参数
        if "ARIMA" in selected_model:
            st.subheader("ARIMA Parameters")

            p = st.slider(
                "AR Order (p)", 1, 24,
                session_config.get("p", 12), 1,  # 使用get避免KeyError
                help="Number of lag observations included in the model"
            )
            config["p"] = p

            q = st.slider(
                "MA Order (q)", 1, 12,
                session_config.get("q", 1), 1,  # 使用get避免KeyError
                help="Size of the moving average window"
            )
            config["q"] = q

            alpha = st.slider(
                "MA Loss Weight (α)", 0.01, 0.5,
                session_config.get("alpha", 0.1), 0.01,  # 使用get避免KeyError
                help="Weight for moving average component in loss function"
            )
            config["alpha"] = alpha

        # 输出步长参数 (所有模型通用)
        if "TCN" not in selected_model or ("TCN" in selected_model and "Lite" in selected_model):
            st.divider()
            st.subheader("Output Configuration")
            output_steps = st.slider(
                "Output Steps", 1, 36,
                session_config.get("output_steps", 12), 1,  # 使用get避免KeyError
                help="Number of future steps to predict"
            )
            config["output_steps"] = output_steps

        # 训练参数
        st.divider()
        st.subheader("Training Parameters")
        lr = st.slider(
            "Learning Rate", 0.0001, 0.1,
            session_config.get("lr", 0.005), 0.0001,  # 使用get避免KeyError
            format="%.4f"
        )
        config["lr"] = lr

        epochs = st.slider(
            "Training Epochs", 50, 2000,
            session_config.get("epochs", 500), 50  # 使用get避免KeyError
        )
        config["epochs"] = epochs

        st.divider()

        # 数据选项
        st.subheader("Data Options")
        show_data = st.checkbox(
            "Show Raw Data",
            session_config.get("show_data", True)  # 使用get避免KeyError
        )
        config["show_data"] = show_data

        show_diff = st.checkbox(
            "Show Differenced Data",
            session_config.get("show_diff", False)  # 使用get避免KeyError
        )
        config["show_diff"] = show_diff

        # 预测选项
        st.subheader("Forecasting Options")
        forecast_months = st.slider(
            "Months to Forecast", 1, 36,
            session_config.get("forecast_months", 12), 1  # 使用get避免KeyError
        )
        config["forecast_months"] = forecast_months

        # 训练按钮
        if st.button("🚀 Train Model"):
            st.session_state.train_button_clicked = True
        else:
            st.session_state.train_button_clicked = False

    return config
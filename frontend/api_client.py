import requests
import streamlit as st

# 后端服务地址（根据实际部署调整）
BACKEND_URL = "http://localhost:8000"


def preprocess_data(config):
    """发送数据预处理请求到后端"""
    try:
        # 准备请求数据
        payload = {
            "dataset_name": config["dataset"],
            "time_col": config["time_col"],
            "value_col": config["value_col"],
            "custom_data": config.get("custom_data", None),
            "p": config.get("p", 12),
            "q": config.get("q", 1)
        }

        # 发送请求
        response = requests.post(
            f"{BACKEND_URL}/preprocess",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Preprocessing failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection to backend failed: {str(e)}")
        return None


def train_model(config, preprocessed_data):
    """发送训练请求到后端"""
    try:
        # 准备请求数据
        payload = {
            "config": {
                "model_type": config["selected_model"],
                "p": config.get("p", 12),
                "q": config.get("q", 1),
                "hidden_size": config.get("hidden_size", 64),
                "kernel_size": config.get("kernel_size", 3),
                "num_layers": config.get("num_layers", 3),
                "dropout": config.get("dropout", 0.1),
                "output_steps": config.get("output_steps", 12),
                "lr": config.get("lr", 0.005),
                "epochs": config.get("epochs", 500),
                "alpha": config.get("alpha", 0.1)
            },
            "train_data": preprocessed_data
        }

        # 发送请求
        response = requests.post(
            f"{BACKEND_URL}/train",
            json=payload,
            timeout=300  # 训练可能需要更长时间
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Training start failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Training service connection error: {str(e)}")
        return None


def get_training_status(job_id):
    """获取训练任务状态"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/status/{job_id}",
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": "Unable to get status"}
    except requests.exceptions.RequestException:
        return {"status": "error", "message": "Service unavailable"}


def forecast(config, model_state):
    """发送预测请求到后端"""
    try:
        # 准备请求数据
        payload = {
            "model_state": model_state,
            "forecast_months": config["forecast_months"],
            "last_values": model_state.get("last_values", []),
            "output_steps": config.get("output_steps", 12)
        }

        response = requests.post(
            f"{BACKEND_URL}/forecast",
            json=payload,
            timeout=60  # 预测可能需要更多时间
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Forecast failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Forecast service connection error: {str(e)}")
        return None
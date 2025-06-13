import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def show_training_in_progress():
    """显示训练中的状态"""
    st.subheader("⏳ Training in Progress")
    st.info("Model training is running. Please wait...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Starting training...")


def show_training_results(results):
    """展示训练结果"""
    st.subheader("✅ Training Results")

    # 显示关键指标
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Loss (Final)", f"{results['final_loss']:.6f}")
        st.metric("Training Time", f"{results['training_time']:.2f} seconds")
        st.metric("Model Type", results['model_type'])

    with col2:
        st.metric("Test RMSE", f"{results['test_rmse']:.2f}")
        st.metric("Test RMSE as % of mean",
                  f"{results['test_rmse_percentage']:.1f}%")
        st.metric("Model Parameters", f"{results['num_params']:,}")

    # 绘制损失曲线
    st.subheader("Training Loss Curve")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(results['loss_history'], label='Training Loss')
    ax.set_title("Loss During Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # 绘制预测对比
    st.subheader("Model Predictions vs Actuals")
    fig, ax = plt.subplots(figsize=(10, 5))

    # 实际值
    ax.plot(results['actuals'], 'b-', label='Actual', alpha=0.7)

    # 预测值
    ax.plot(results['predictions'], 'r--', label='Predicted', alpha=0.8)

    ax.axvline(x=results['train_size'], color='k', linestyle='--', label='Train/Test Split')
    ax.set_title(f"{results['model_type']} Performance on Test Set")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # 显示模型结构信息
    if 'model_summary' in results:
        st.subheader("Model Architecture")
        st.code(results['model_summary'])
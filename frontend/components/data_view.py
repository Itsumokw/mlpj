import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def show_data_preview(config):
    """展示数据预览"""
    st.subheader("📊 Dataset Preview")

    # 安全获取配置值，避免KeyError
    dataset_name = config.get('dataset', 'Unknown Dataset')
    time_col = config.get('time_col', None)
    value_col = config.get('value_col', None)
    custom_data = config.get('custom_data', None)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"**Dataset:** `{dataset_name}`")
        if dataset_name == "Air Passengers (Default)":
            st.info("Built-in Air Passengers dataset (1949-1960)")
        else:
            st.info("Custom uploaded dataset")

        # 只有在列名存在时才显示
        if time_col:
            st.markdown(f"**Time Column:** `{time_col}`")
        if value_col:
            st.markdown(f"**Value Column:** `{value_col}`")

        # 显示数据预览 - 只在有自定义数据时显示
        if custom_data and dataset_name == "Upload Custom Dataset":
            try:
                df = pd.DataFrame(custom_data)
                st.dataframe(df.head(5))
            except Exception as e:
                st.warning(f"Failed to display data preview: {str(e)}")
        elif dataset_name == "Upload Custom Dataset":
            st.info("No dataset uploaded yet")
        else:
            st.info("Default dataset will be loaded during processing")

    with col2:
        # 只有在有自定义数据时才尝试绘图
        if custom_data and dataset_name == "Upload Custom Dataset" and time_col and value_col:
            try:
                df = pd.DataFrame(custom_data)

                # 确保列名在数据中存在
                if time_col not in df.columns or value_col not in df.columns:
                    st.warning("Selected columns not found in dataset")
                    return

                fig, ax = plt.subplots(figsize=(10, 4))

                # 尝试将时间列转换为datetime
                if pd.api.types.is_string_dtype(df[time_col]):
                    try:
                        df['temp_time'] = pd.to_datetime(df[time_col])
                        ax.plot(df['temp_time'], df[value_col], 'b-')
                        plt.xticks(rotation=45)
                    except:
                        ax.plot(df[time_col], df[value_col], 'b-')
                        plt.xticks(rotation=45)
                else:
                    ax.plot(df[time_col], df[value_col], 'b-')

                ax.set_title("Custom Time Series Data")
                ax.set_ylabel(value_col)
                ax.grid(True)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting data: {str(e)}")
                st.info("Displaying data without time axis")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df[value_col], 'b-')
                ax.set_xlabel("Index")
                ax.set_title("Custom Time Series Data")
                ax.set_ylabel(value_col)
                ax.grid(True)
                st.pyplot(fig)
        elif dataset_name == "Upload Custom Dataset":
            st.info("Upload a dataset to see visualization")
        else:
            st.info("Data visualization will be available after processing")
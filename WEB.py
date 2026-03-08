import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from io import BytesIO

# =========================
# 自定义集成模型类（必须先定义，joblib.load 时会用到）
# =========================
class EnsembleModel(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        """
        初始化集成回归模型。

        :param models: 字典类型，包含每个基学习器模型
        :param weights: 模型权重数组
        """
        self.models = models
        self.weights = np.array(weights)

        if len(self.models) != len(self.weights):
            raise ValueError("模型数量与权重数量不一致！")

        self.model_names = list(self.models.keys())
        self.weights_dict = dict(zip(self.model_names, self.weights))

    def fit(self, X, y):
        for model in self.models.values():
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.zeros((X.shape[0]))
        for model_name, model in self.models.items():
            model_pred = model.predict(X)
            model_weight = self.weights_dict[model_name]
            predictions += model_pred * model_weight
        return predictions

    def get_model_weights(self):
        return self.weights_dict

# =========================
# 页面配置
# =========================
st.set_page_config(page_title="在线预测器", page_icon="📈", layout="wide")
st.title("📈 Streamlit 在线预测器")
st.write("支持单条预测和批量文件预测（CSV / XLSX），输入数据会先标准化，再送入集成模型进行预测。")

# =========================
# 加载模型和标准化器
# =========================
@st.cache_resource
def load_objects():
    scaler = joblib.load("standard_scaler.pkl")
    model = joblib.load("ensemble.pkl")
    return scaler, model

try:
    scaler, ensemble_model = load_objects()
except Exception as e:
    st.error(f"模型或标准化器加载失败：{e}")
    st.stop()

# =========================
# 特征顺序（必须与训练时一致）
# =========================
feature_names = [
    "infP", "infC", "infAc", "infpro", "infS",
    "MLSS", "MLVSS", "VSS/TSS", "volumn",
    "ana-time", "pH", "T", "salinity"
]

default_values = [
    10.39, 74.03, 49.60, 24.43, 148.05,
    7.50, 5.30, 0.71, 14.0,
    3.0, 7.6, 22.0, 0.70
]

# =========================
# 工具函数
# =========================
def predict_dataframe(df, scaler, model, feature_names):
    """
    对DataFrame进行批量预测
    要求df至少包含 feature_names 中的所有字段
    """
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"上传文件缺少以下必要列：{missing_cols}")

    # 按训练顺序取列
    X = df[feature_names].copy()

    # 标准化
    X_scaled = scaler.transform(X)

    # 预测
    preds = model.predict(X_scaled)

    # 返回包含原始特征和预测值的结果
    result_df = df.copy()
    result_df["prediction"] = preds
    return result_df, X_scaled

def dataframe_to_excel_bytes(df, sheet_name="prediction_result"):
    """
    将DataFrame转成Excel二进制流，供下载
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.getvalue()

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["单条预测", "批量文件预测"])

# =========================
# 单条预测
# =========================
with tab1:
    st.subheader("请输入单条样本特征值")

    cols = st.columns(3)
    input_values = {}

    for i, (feat, default) in enumerate(zip(feature_names, default_values)):
        with cols[i % 3]:
            input_values[feat] = st.number_input(
                label=feat,
                value=float(default),
                format="%.4f"
            )

    if st.button("开始单条预测", type="primary"):
        try:
            # 按固定顺序组装输入
            X_input = pd.DataFrame(
                [[input_values[feat] for feat in feature_names]],
                columns=feature_names
            )

            st.write("### 原始输入数据")
            st.dataframe(X_input)

            # 标准化
            X_scaled = scaler.transform(X_input)

            st.write("### 标准化后的数据")
            st.dataframe(pd.DataFrame(X_scaled, columns=feature_names))

            # 预测
            prediction = ensemble_model.predict(X_scaled)

            result_single = X_input.copy()
            result_single["prediction"] = prediction

            st.success(f"预测结果：{prediction[0]:.6f}")

            st.write("### 单条预测结果表")
            st.dataframe(result_single)

            # 下载单条预测Excel
            single_excel = dataframe_to_excel_bytes(result_single, sheet_name="single_prediction")
            st.download_button(
                label="下载单条预测结果 Excel",
                data=single_excel,
                file_name="single_prediction_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # 若想显示模型权重
            if hasattr(ensemble_model, "get_model_weights"):
                with st.expander("查看集成模型权重"):
                    st.json(ensemble_model.get_model_weights())

        except Exception as e:
            st.error(f"单条预测失败：{e}")

# =========================
# 批量文件预测
# =========================
with tab2:
    st.subheader("上传 CSV 或 XLSX 文件进行批量预测")
    st.write("上传文件中必须包含以下列名，且列名要完全一致：")
    st.code(", ".join(feature_names))

    uploaded_file = st.file_uploader(
        "选择一个 CSV 或 XLSX 文件",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            # 判断文件类型并读取
            if uploaded_file.name.endswith(".csv"):
                df_uploaded = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df_uploaded = pd.read_excel(uploaded_file)
            else:
                st.error("仅支持 csv 或 xlsx 文件。")
                st.stop()

            st.write("### 上传数据预览")
            st.dataframe(df_uploaded.head())

            # 检查列是否完整
            missing_cols = [col for col in feature_names if col not in df_uploaded.columns]
            if missing_cols:
                st.error(f"文件缺少必要列：{missing_cols}")
            else:
                if st.button("开始批量预测", type="primary"):
                    result_df, X_scaled = predict_dataframe(
                        df_uploaded, scaler, ensemble_model, feature_names
                    )

                    st.success(f"批量预测完成，共预测 {len(result_df)} 条数据。")

                    st.write("### 预测结果预览")
                    st.dataframe(result_df.head())

                    # 导出为Excel
                    excel_data = dataframe_to_excel_bytes(
                        result_df,
                        sheet_name="batch_prediction"
                    )

                    st.download_button(
                        label="下载批量预测结果 Excel",
                        data=excel_data,
                        file_name="batch_prediction_result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    st.info("点击下载后，浏览器会让你选择保存到电脑中的目标路径。")

        except Exception as e:
            st.error(f"批量预测失败：{e}")

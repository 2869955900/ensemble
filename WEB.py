import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import traceback
from io import BytesIO
from sklearn.base import BaseEstimator, RegressorMixin

# =========================================================
# 1. 先定义自定义模型类（必须在 joblib.load 之前）
# =========================================================
class EnsembleModel(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        """
        初始化集成回归模型
        :param models: 字典，包含各基学习器模型
        :param weights: 各模型权重
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

# =========================================================
# 2. 关键：把类注册到 __main__
#    兼容你之前从 notebook / script 直接 dump 的 pkl
# =========================================================
sys.modules["__main__"].EnsembleModel = EnsembleModel

# =========================================================
# 3. 页面设置
# =========================================================
st.set_page_config(page_title="SR 在线预测系统", page_icon="📈", layout="wide")
st.title("📈 SR 在线预测系统")
st.markdown("支持 **单一样本预测** 和 **批量上传 Excel/CSV 预测**。")

# =========================================================
# 4. 特征顺序（必须和训练时一致）
# =========================================================
FEATURE_NAMES = [
    "infP", "infC", "infAc", "infpro", "infS",
    "MLSS", "MLVSS", "VSS/TSS", "volumn", "ana-time",
    "pH", "T", "salinity"
]

DEFAULT_VALUES = {
    "infP": 10.39,
    "infC": 74.03,
    "infAc": 49.60,
    "infpro": 24.43,
    "infS": 148.05,
    "MLSS": 7.50,
    "MLVSS": 5.30,
    "VSS/TSS": 0.71,
    "volumn": 14.0,
    "ana-time": 3.0,
    "pH": 7.6,
    "T": 22.0,
    "salinity": 0.70
}

# =========================================================
# 5. 加载模型和标准化器
# =========================================================
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("standard_scaler.pkl")
    model = joblib.load("ensemble.pkl")
    return scaler, model

try:
    scaler, model = load_artifacts()
except Exception:
    st.error("模型或标准化器加载失败，请检查文件和依赖环境。")
    st.code(traceback.format_exc())
    st.stop()

# =========================================================
# 6. 工具函数
# =========================================================
def check_input_dataframe(df: pd.DataFrame, feature_names):
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要特征列: {missing_cols}")

    X = df[feature_names].copy()

    # 转成数值，防止上传 csv/excel 后某些列变成字符串
    for col in feature_names:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isnull().sum().sum() > 0:
        null_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"以下列存在空值或非数值内容，请检查：{null_cols}")

    return X

def predict_sr(df: pd.DataFrame, scaler, model, feature_names):
    X = check_input_dataframe(df, feature_names)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return y_pred, X, X_scaled

def dataframe_to_excel_bytes(df: pd.DataFrame):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="prediction_result")
    return output.getvalue()

def make_template_df():
    return pd.DataFrame([DEFAULT_VALUES], columns=FEATURE_NAMES)

# =========================================================
# 7. 模式选择
# =========================================================
mode = st.radio(
    "请选择预测模式：",
    ["单一样本预测", "批量文件预测"],
    horizontal=True
)

# =========================================================
# 8. 单一样本预测
# =========================================================
if mode == "单一样本预测":
    st.subheader("单一样本输入")

    cols = st.columns(3)
    user_input = {}

    for i, feature in enumerate(FEATURE_NAMES):
        with cols[i % 3]:
            user_input[feature] = st.number_input(
                label=feature,
                value=float(DEFAULT_VALUES[feature]),
                format="%.4f"
            )

    input_df = pd.DataFrame([user_input], columns=FEATURE_NAMES)

    st.markdown("### 当前输入数据")
    st.dataframe(input_df, use_container_width=True)

    if st.button("开始预测 SR", type="primary"):
        try:
            pred, X_raw, X_scaled = predict_sr(input_df, scaler, model, FEATURE_NAMES)

            st.subheader("预测结果")
            st.success(f"预测 SR = {float(pred[0]):.4f}")

            with st.expander("查看标准化后的特征"):
                scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)
                st.dataframe(scaled_df, use_container_width=True)

        except Exception as e:
            st.error(f"预测失败：{e}")
            st.code(traceback.format_exc())

# =========================================================
# 9. 批量文件预测
# =========================================================
elif mode == "批量文件预测":
    st.subheader("批量上传 Excel / CSV 文件")

    st.markdown("### 必须包含以下特征列")
    st.code(", ".join(FEATURE_NAMES))

    # 下载模板
    template_df = make_template_df()

    csv_template = template_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="下载 CSV 模板",
        data=csv_template,
        file_name="SR_prediction_template.csv",
        mime="text/csv"
    )

    excel_template = dataframe_to_excel_bytes(template_df)
    st.download_button(
        label="下载 Excel 模板",
        data=excel_template,
        file_name="SR_prediction_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    uploaded_file = st.file_uploader(
        "上传 CSV 或 Excel 文件",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            elif uploaded_file.name.lower().endswith(".xlsx"):
                df_upload = pd.read_excel(uploaded_file)
            else:
                raise ValueError("仅支持 csv 或 xlsx 文件。")

            st.markdown("### 上传数据预览")
            st.dataframe(df_upload.head(20), use_container_width=True)

            if st.button("开始批量预测 SR", type="primary"):
                pred, X_raw, X_scaled = predict_sr(df_upload, scaler, model, FEATURE_NAMES)

                result_df = df_upload.copy()
                result_df["SR_pred"] = pred

                st.subheader("预测完成")
                st.dataframe(result_df.head(50), use_container_width=True)

                st.markdown(f"共预测样本数：**{len(result_df)}**")

                # 下载 CSV
                result_csv = result_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="下载预测结果 CSV",
                    data=result_csv,
                    file_name="SR_prediction_result.csv",
                    mime="text/csv"
                )

                # 下载 Excel
                result_excel = dataframe_to_excel_bytes(result_df)
                st.download_button(
                    label="下载预测结果 Excel",
                    data=result_excel,
                    file_name="SR_prediction_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                with st.expander("查看标准化后的部分数据"):
                    scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)
                    st.dataframe(scaled_df.head(50), use_container_width=True)

        except Exception as e:
            st.error(f"文件处理或预测失败：{e}")
            st.code(traceback.format_exc())

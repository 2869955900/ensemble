import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# =========================
# 页面设置
# =========================
st.set_page_config(page_title="SR在线预测系统", page_icon="📈", layout="wide")

st.title("📈 SR 在线预测系统")
st.markdown("支持 **单一样本预测** 和 **批量上传 Excel/CSV 预测**。")

# =========================
# 加载模型与标准化器
# =========================
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("standard_scaler.pkl")
    model = joblib.load("ensemble.pkl")
    return scaler, model

scaler, model = load_artifacts()

# =========================
# 特征顺序（必须和训练时一致）
# =========================
feature_names = [
    "infP", "infC", "infAc", "infpro", "infS",
    "MLSS", "MLVSS", "VSS/TSS", "volumn", "ana-time",
    "pH", "T", "salinity"
]

default_values = {
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

# =========================
# 工具函数
# =========================
def predict_df(df, scaler, model, feature_names):
    """
    对输入 DataFrame 按指定特征顺序进行标准化并预测 SR
    """
    # 检查缺失列
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少以下必要特征列: {missing_cols}")

    # 只取模型需要的列，并保证顺序一致
    X = df[feature_names].copy()

    # 检查空值
    if X.isnull().sum().sum() > 0:
        raise ValueError("输入数据中存在缺失值，请先处理后再预测。")

    # 标准化
    X_scaled = scaler.transform(X)

    # 预测
    pred = model.predict(X_scaled)

    return pred, X, X_scaled

def to_excel_download(df):
    """
    将 DataFrame 转为 Excel 二进制，便于下载
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="prediction_result")
    return output.getvalue()

# =========================
# 模式选择
# =========================
mode = st.radio(
    "请选择预测模式：",
    ["单一样本预测", "批量文件预测"],
    horizontal=True
)

# =========================
# 单一样本预测
# =========================
if mode == "单一样本预测":
    st.subheader("单一样本输入")

    cols = st.columns(3)
    user_input = {}

    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            user_input[feature] = st.number_input(
                label=feature,
                value=float(default_values[feature]),
                format="%.4f"
            )

    input_df = pd.DataFrame([user_input], columns=feature_names)

    st.markdown("### 当前输入数据")
    st.dataframe(input_df, use_container_width=True)

    if st.button("开始预测 SR", key="single_predict"):
        try:
            pred, X_raw, X_scaled = predict_df(input_df, scaler, model, feature_names)

            st.subheader("预测结果")
            st.success(f"预测 SR = {pred[0]:.4f}")

            with st.expander("查看标准化后的特征"):
                scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
                st.dataframe(scaled_df, use_container_width=True)

        except Exception as e:
            st.error(f"预测失败：{e}")

# =========================
# 批量文件预测
# =========================
elif mode == "批量文件预测":
    st.subheader("批量上传 Excel / CSV 文件")

    st.markdown(
        """
        **文件要求：**
        - 支持 `.csv`、`.xlsx`
        - 必须包含以下特征列：
        """
    )
    st.code(", ".join(feature_names))

    uploaded_file = st.file_uploader(
        "上传文件",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            # 读取文件
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.markdown("### 上传数据预览")
            st.dataframe(df_upload.head(), use_container_width=True)

            if st.button("开始批量预测 SR", key="batch_predict"):
                pred, X_raw, X_scaled = predict_df(df_upload, scaler, model, feature_names)

                result_df = df_upload.copy()
                result_df["SR_pred"] = pred

                st.subheader("预测完成")
                st.dataframe(result_df.head(20), use_container_width=True)

                # 下载 CSV
                csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="下载预测结果 CSV",
                    data=csv_data,
                    file_name="SR_prediction_result.csv",
                    mime="text/csv"
                )

                # 下载 Excel
                excel_data = to_excel_download(result_df)
                st.download_button(
                    label="下载预测结果 Excel",
                    data=excel_data,
                    file_name="SR_prediction_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                with st.expander("查看标准化后的部分数据"):
                    scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
                    st.dataframe(scaled_df.head(20), use_container_width=True)

        except Exception as e:
            st.error(f"文件处理或预测失败：{e}")

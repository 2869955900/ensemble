import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from io import BytesIO
import os
import fitz  # PyMuPDF

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
st.write("支持单条预测、批量文件预测，以及模型性能可视化展示。")

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
    """
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"上传文件缺少以下必要列：{missing_cols}")

    X = df[feature_names].copy()
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    result_df = df.copy()
    result_df["prediction"] = preds
    return result_df

def dataframe_to_excel_bytes(df, sheet_name="prediction_result"):
    """
    将DataFrame导出为Excel字节流
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.getvalue()

def pdf_download_button(pdf_path, button_label):
    """
    PDF下载按钮
    """
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label=button_label,
            data=pdf_bytes,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf"
        )
    else:
        st.warning(f"未找到文件：{pdf_path}")

@st.cache_data
def pdf_first_page_to_png_bytes(pdf_path, zoom=2.0):
    """
    将PDF第一页转换为PNG字节流
    zoom越大，清晰度越高
    """
    if not os.path.exists(pdf_path):
        return None

    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        doc.close()
        return None

    page = doc.load_page(0)  # 只取第一页
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes

def show_pdf_as_image(pdf_path, title=None, caption=None, zoom=2.0):
    """
    将PDF第一页转成PNG并展示
    """
    if title:
        st.markdown(title)

    if not os.path.exists(pdf_path):
        st.warning(f"未找到文件：{pdf_path}")
        return

    try:
        img_bytes = pdf_first_page_to_png_bytes(pdf_path, zoom=zoom)
        if img_bytes is None:
            st.warning(f"无法读取PDF内容：{pdf_path}")
        else:
            st.image(img_bytes, caption=caption, use_container_width=True)
    except Exception as e:
        st.error(f"PDF 转图片失败：{e}")

# =========================
# 三个页面
# =========================
tab1, tab2, tab3 = st.tabs(["单条预测", "批量文件预测", "模型性能展示"])

# =========================
# 页面1：单条预测
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
            X_input = pd.DataFrame(
                [[input_values[feat] for feat in feature_names]],
                columns=feature_names
            )

            st.write("### 原始输入数据")
            st.dataframe(X_input, use_container_width=True)

            X_scaled = scaler.transform(X_input)

            st.write("### 标准化后的数据")
            st.dataframe(
                pd.DataFrame(X_scaled, columns=feature_names),
                use_container_width=True
            )

            prediction = ensemble_model.predict(X_scaled)

            result_single = X_input.copy()
            result_single["prediction"] = prediction

            st.success(f"预测结果：{prediction[0]:.6f}")

            st.write("### 单条预测结果表")
            st.dataframe(result_single, use_container_width=True)

            try:
                single_excel = dataframe_to_excel_bytes(
                    result_single,
                    sheet_name="single_prediction"
                )
                st.download_button(
                    label="下载单条预测结果 Excel",
                    data=single_excel,
                    file_name="single_prediction_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.warning(f"Excel导出不可用：{e}。如需导出xlsx，请安装 openpyxl。")

            if hasattr(ensemble_model, "get_model_weights"):
                with st.expander("查看集成模型权重"):
                    st.json(ensemble_model.get_model_weights())

        except Exception as e:
            st.error(f"单条预测失败：{e}")

# =========================
# 页面2：批量文件预测
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
            if uploaded_file.name.endswith(".csv"):
                df_uploaded = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df_uploaded = pd.read_excel(uploaded_file)
            else:
                st.error("仅支持 csv 或 xlsx 文件。")
                st.stop()

            st.write("### 上传数据预览")
            st.dataframe(df_uploaded.head(), use_container_width=True)

            missing_cols = [col for col in feature_names if col not in df_uploaded.columns]
            if missing_cols:
                st.error(f"文件缺少必要列：{missing_cols}")
            else:
                if st.button("开始批量预测", type="primary"):
                    result_df = predict_dataframe(
                        df_uploaded, scaler, ensemble_model, feature_names
                    )

                    st.success(f"批量预测完成，共预测 {len(result_df)} 条数据。")

                    st.write("### 预测结果预览")
                    st.dataframe(result_df.head(), use_container_width=True)

                    try:
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
                    except Exception as e:
                        st.warning(f"Excel导出失败：{e}。如需导出xlsx，请安装 openpyxl。")

                    st.info("点击下载后，浏览器会让你选择保存到电脑中的目标路径。")

        except Exception as e:
            st.error(f"批量预测失败：{e}")

# =========================
# 页面3：模型性能展示（PDF读取并转PNG展示）
# =========================
with tab3:
    st.subheader("模型性能可视化展示")

    st.markdown("""
    本页面展示：
    - 单一模型性能对比 PDF 转图片预览
    - 堆叠模型性能 PDF 转图片预览
    - 综合结论：**堆叠模型性能更优**
    """)

    st.success("结论：堆叠模型综合性能优于单一模型，预测精度和稳定性更好。")

    pdf_single_model = "Six_Models_RealData_Comparison.pdf"
    pdf_ensemble_model = "EnsembleModel.pdf"

    st.markdown("---")
    st.markdown("## 1. 单一模型性能对比")

    show_pdf_as_image(
        pdf_single_model,
        caption="单一模型性能对比图（由PDF第一页转换）",
        zoom=2.0
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("该图展示多个单一模型在真实数据上的性能对比结果。")
    with c2:
        pdf_download_button(pdf_single_model, "下载单一模型对比 PDF")

    st.markdown("---")
    st.markdown("## 2. 堆叠模型性能展示")

    show_pdf_as_image(
        pdf_ensemble_model,
        caption="堆叠模型性能图（由PDF第一页转换）",
        zoom=2.0
    )

    c3, c4 = st.columns([2, 1])
    with c3:
        st.write("该图展示堆叠/集成模型在测试数据上的性能表现。")
    with c4:
        pdf_download_button(pdf_ensemble_model, "下载堆叠模型 PDF")

    st.markdown("---")
    st.markdown("## 3. 对比结论")
    st.info("""
    与单一模型相比，堆叠模型通过融合多个基学习器的优势，
    能够提升预测精度，降低单模型误差波动，
    因而在稳定性、鲁棒性与泛化能力方面更优。

    因此，本系统最终推荐使用：**堆叠模型（Ensemble / Stacking）**。
    """)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("单一模型", "基准表现")
    with m2:
        st.metric("堆叠模型", "更优表现", delta="性能提升")
    with m3:
        st.metric("推荐模型", "堆叠模型")

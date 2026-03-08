import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

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
st.write("输入特征后，先进行标准化，再送入集成模型进行预测。")

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
# 输入区域
# =========================
st.subheader("请输入特征值")

cols = st.columns(3)
input_values = {}

for i, (feat, default) in enumerate(zip(feature_names, default_values)):
    with cols[i % 3]:
        input_values[feat] = st.number_input(
            label=feat,
            value=float(default),
            format="%.4f"
        )

# =========================
# 预测
# =========================
if st.button("开始预测", type="primary"):
    try:
        # 按固定顺序组装输入
        X_input = pd.DataFrame([[input_values[feat] for feat in feature_names]], columns=feature_names)

        st.write("### 原始输入数据")
        st.dataframe(X_input)

        # 标准化
        X_scaled = scaler.transform(X_input)

        st.write("### 标准化后的数据")
        st.dataframe(pd.DataFrame(X_scaled, columns=feature_names))

        # 预测
        prediction = ensemble_model.predict(X_scaled)

        st.success(f"预测结果：{prediction[0]:.6f}")

        # 若想显示模型权重
        if hasattr(ensemble_model, "get_model_weights"):
            with st.expander("查看集成模型权重"):
                st.json(ensemble_model.get_model_weights())

    except Exception as e:
        st.error(f"预测失败：{e}")

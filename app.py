import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
from sklearn.preprocessing import StandardScaler, OneHotEncoder

MODEL_PATH = "best_Elastic_model.pkl"
FEATURE_NAMES_PATH = "feature_info.pkl"

st.set_page_config(
    page_title="Прогнозирование цены авто",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>
    /* Основной фон */
    .stApp {
        background-color: #F5FFFA;
    }

    /* Шапка приложения */
    header[data-testid="stHeader"] {
        background-color: #F5FFFA;
    }

    /* Панель инструментов (кнопка меню и т.д.) */
    .stToolbar {
        background-color: #F5FFFA;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Прогнозирование стоимости автомобиля по его характеристикам")

@st.cache_resource
def load_model():
    """Загружаем модель через pickle"""

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names


def prepare_features(df, feature_names):
    """Приводим данные к формату обучения модели."""
    df_proc = df.copy()
    # Преобразуем категориальные признаки в строки (как при обучении)
    for col in feature_names:
        if col in df_proc.columns:
            if df_proc[col].dtype in ('object', 'bool'):
                df_proc[col] = df_proc[col].astype(str)
    return df_proc[feature_names]


try:
    MODEL, FEATURE_NAMES = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])


if uploaded_file is None:
    st.info("👈 Загрузите CSV файл для начала работы")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file)

try:
    features = prepare_features(df, FEATURE_NAMES)
    probabilities = MODEL.predict(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    df['prediction'] = predictions
    df['prob_leave'] = probabilities
except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {e}")
    st.stop()

# --- Метрики ---
st.subheader("📊 Результаты")

st.subheader("📈 Визуализации")
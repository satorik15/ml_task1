import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import sklearn
MODEL_PATH = "best_Elastic_model.pkl"
FEATURE_NAMES_PATH = "feature_info.pkl"
SCALER_PATH = "scaler.pkl"
DATA_PATH = "X_train.csv"


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
    with open(SCALER_PATH, 'rb') as f:
        my_scaler = pickle.load(f)
    with open(DATA_PATH, 'r') as f:
        X_trained = pd.read_csv(f)
        pass
    return model, feature_names,my_scaler,X_trained


def prepare_features(df, feature_names,train_medians,scaler):
    """Приводим данные к формату на обучении модели."""


    df_proc = df.copy()
    #Выбор только числовых колонок
    numeric_columns=feature_names["feature_names"]
    df_proc=df_proc[numeric_columns]
    #Предобработка аналогично EDA
    #Удаление полных дубликатов
    df_proc=df_proc[df_proc.duplicated()>0]
    #удаление единиц измерения
    df_proc['mileage'] = df_proc['mileage'].str.extract('(\d+\.?\d*)')
    df_proc['engine'] = df_proc['engine'].str.extract('(\d+\.?\d*)')
    df_proc['max_power'] = df_proc['max_power'].str.extract('(\d+\.?\d*)')
    # Приведение к Float
    df_proc[['mileage', 'engine', 'max_power']] = df_proc[['mileage', 'engine', 'max_power']].astype(float)
    #заполнение пропусков медианами
    numeric_with_missing = [col for col in numeric_columns if df_proc[col].isnull().any()]
    df_proc[numeric_with_missing] = df_proc[numeric_with_missing].fillna(train_medians)
    #Приведение типов
    df_proc[['seats', 'engine']] = df_proc[['seats', 'engine']].astype(int)
    df_proc=scaler.transform(df_proc)

    return df_proc

try:
    MODEL, FEATURE_NAMES,SCALER, X_trained = load_model()
    train_medians=X_trained.mean()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])


if uploaded_file is None:
    st.info("↑ Загрузите CSV файл для начала работы")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file)

try:
    features = prepare_features(df, FEATURE_NAMES,train_medians,SCALER)
    result=MODEL.predict(features)
except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {e}")
    st.stop()


st.header("📈 Визуализации")


correlation_matrix = X_trained.corr(numeric_only=True)
st.subheader("1. Тепловая карта корреляций")

fig1 = px.imshow(
    correlation_matrix,
    text_auto=True,  # Автоматически показывает значения
    aspect="auto",  # Автоматический подбор пропорций
    color_continuous_scale="RdBu_r",  # Обратная палитра для лучшей читаемости
    title="Тепловая карта корреляций",
    height=400,  # Компактная высота
    width=500  # Компактная ширина
)

# Настройка макета
fig1.update_layout(
    font=dict(size=10),  # Уменьшаем шрифт
    margin=dict(l=20, r=20, t=40, b=20)  # Уменьшаем отступы
)

st.plotly_chart(fig1, use_container_width=True)

# 2. Гистограмма selling_price (интерактивная)
st.header("2. Распределение цены продажи")

fig2 = px.histogram(
    result,
    nbins=50,  # Количество бинов
    title='Гистограмма цены автомобиля',
    height=350,  # Компактная высота
    width=600  # Ширина
)

# Добавляем линию плотности
fig2.update_traces(
    marker=dict(line=dict(width=1, color='DarkSlateGrey'))
)

# Настройка макета
fig2.update_layout(
    bargap=0.1,  # Расстояние между столбцами
    font=dict(size=10),
    xaxis_title="Цена продажи",
    yaxis_title="Количество",
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig2, use_container_width=True)

st.header('Распределение автомобилей по годам производства',)
year_counts = X_trained['year'].value_counts().reset_index()
year_counts.columns = ['Год', 'Количество']
fig = px.pie(
    year_counts,
    values='Количество',
    names='Год',
    height=600
)

# Настраиваем отображение
fig.update_traces(
    textposition='inside',
    textinfo='percent+label',
    hovertemplate="<b>Год:</b> %{label}<br>" +
                  "<b>Количество:</b> %{value}<br>" +
                  "<b>Доля:</b> %{percent:.1%}<br>",
    marker=dict(line=dict(color='white', width=2))
)

fig.update_layout(
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="right",
        x=1.1
    ),
    margin=dict(t=50, b=50, l=50, r=150)
)

st.plotly_chart(fig, use_container_width=True)



st.subheader("📊 Веса модели")


feature_weights = pd.DataFrame({'Признаки': FEATURE_NAMES["feature_names"], 'Веса': MODEL.coef_}).transpose()

new_header = feature_weights.iloc[0]  # берем первую строку как заголовок
feature_weights = feature_weights[1:]  # отбрасываем исходный заголовок

# переименовываем столбцы
feature_weights.rename(columns=new_header, inplace=True)
st.write(feature_weights)

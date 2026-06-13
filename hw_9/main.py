import os
import pandas as pd
import streamlit as st
from utils import load_dataset, train_model, read_model


st.set_page_config(
    page_title="Realty App",
    layout="centered"
)

st.title("Предсказание стоимости недвижимости")

df = load_dataset()
model_path = 'rf_fitted.pkl'


st.sidebar.header("Параметры квартиры")

total_square = st.sidebar.number_input("Какая площадь (кв. м)?", 1, 2070, 30)
floor = st.sidebar.number_input("Какой этаж?", 1, 66, 1)
rooms = st.sidebar.number_input("Сколько комнат?", 1, 15, 3)

city = st.sidebar.selectbox(
    "Какой город?",
    options=sorted(df["city"].unique()),
)

source = st.sidebar.selectbox(
    "Источник данных?",
    options=sorted(df["source"].unique())
)

if not os.path.exists(model_path):
    with st.spinner("Модель не найдена. Запускается обучение"):
        train_model(df)
    st.success("Модель успешно обучена и сохранена!")

model = read_model(model_path)


input_data = pd.DataFrame([{
    "total_square": float(total_square),
    "rooms": int(rooms),
    "floor": int(floor),
    "city": city,
    "source": source
}])


preds = model.predict(input_data)[0]

st.metric(
    label="Предполагаемая стоимость квартиры",
    value=f"{preds:,.2f} рублей".replace(",", " ")
)

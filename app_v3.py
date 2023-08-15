import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import openai  # Импорт библиотеки GPT-3

MAIN_PATH = 'data.csv'
CHANNEL_PATH = 'binipharm.csv'

SHEET_MAIN_PATH = 'https://docs.google.com/spreadsheets/d/1Rw_Pxv2dgPR8S8UNqUCmoCUBa4NeUQH1/edit#gid=1549281357'
SHEET_CHANNEL_PATH = 'https://docs.google.com/spreadsheets/d/1zenrlWJ7QAuaXJAdmYi3j2t-yoEvtfujAp2UhDmzh9E/edit#gid=1855222089'
# Подставьте ваш ключ API GPT-3
openai.api_key = st.secrets["OPENAI_API_KEY"]


# Кэширование загрузки данных
@st.cache_data
def load_data(url_main, url_channel):
    url_main = url_main.replace("/edit#gid=", "/export?format=csv&gid=")
    url_channel = url_channel.replace("/edit#gid=", "/export?format=csv&gid=")
    data = pd.read_csv(url_main)
    data['data_kommunikacii'] = data.data_kommunikacii.astype("datetime64[ns]")
    data['ctr'] = (data.clicked / data.opened)
    data['open_rate'] = (data.opened / data.delivered)
    data_last_year = data.loc[
        data['data_kommunikacii'] >= data['data_kommunikacii'].max() - pd.DateOffset(years=1)].copy()
    data_brand_exp = data_last_year
    data_brand_exp['brand'] = data_last_year['brand'].apply(lambda x: x.split(','))
    data_brand_exp = data_brand_exp.explode('brand').reset_index()
    data_brand_exp = data_brand_exp[data_brand_exp["opened"] > 0].copy()
    data_brand_exp['brand'] = data_brand_exp['brand'].apply(lambda x: x.strip())
    data_brand_exp = data_brand_exp.sort_values(by='opened')

    data_channel = pd.read_csv(url_channel)
    data_channel['data_kommunikacii'] = data_channel['data_kommunikacii'].astype("datetime64[ns]")
    data_channel['open_rate'] = data_channel['opened'] / data_channel['delivered']

    data_channel_last_year = data_channel.loc[
        data_channel['data_kommunikacii'] >= data_channel['data_kommunikacii'].max() - pd.DateOffset(years=1)].copy()

    data_channel_brand_exp = data_channel_last_year
    data_channel_brand_exp['brand'] = data_channel_last_year['brand'].apply(lambda x: x.split(','))
    data_channel_brand_exp = data_channel_brand_exp.explode('brand').reset_index()
    data_channel_brand_exp = data_channel_brand_exp[data_channel_brand_exp["opened"] > 0].copy()
    data_channel_brand_exp['brand'] = data_channel_brand_exp['brand'].apply(lambda x: x.strip())
    data_channel_brand_exp = data_channel_brand_exp.sort_values(by='opened')

    return data_brand_exp, data_channel_brand_exp


df, df_channel = load_data(SHEET_MAIN_PATH, SHEET_CHANNEL_PATH)

st.title("Interactive Dashboard")

# Создаем меню страниц
menu = ["Overall Statistics", "Channel Statistics", "Generate Messages"]
choice = st.sidebar.selectbox("Select Page", menu)

if choice == "Overall Statistics":
    st.sidebar.subheader("Select Metrics")
    selected_metric = st.sidebar.selectbox("Select Metric", ['opened', 'open_rate'])
    selected_region = st.sidebar.selectbox("Select Region", ["Россия"] + list(df["oblast"].unique()))
    selected_target = st.sidebar.selectbox("Select Target", ["Все"] + list(df['target'].unique()))
    selected_brand = st.sidebar.selectbox("Select Brand", ["Все"] + list(df['brand'].unique()))

    filtered_df = df[df["oblast"] == selected_region] if selected_region != "Россия" else df.copy()
    filtered_df = filtered_df[filtered_df["target"] == selected_target] if selected_target != "Все" else filtered_df
    filtered_df = filtered_df[filtered_df["brand"] == selected_brand] if selected_brand != "Все" else filtered_df

    st.subheader("Overall dynamic")
    data_gruped_mean = df.groupby('data_kommunikacii').mean()
    fig = px.line(filtered_df.groupby('data_kommunikacii').mean(),
                  y=filtered_df.groupby('data_kommunikacii').mean().ewm(alpha=0.15).mean()[selected_metric]
                  , labels={'data_kommunikacii': 'Date', 'y': 'Opened'}
                  )

    fig.add_scatter(x=data_gruped_mean.index, y=(data_gruped_mean.ewm(alpha=0.15).mean()[selected_metric]),
                    name='General', line={'color': 'red'},
                    opacity=0.5)
    st.plotly_chart(fig)

    st.subheader("Overall hist")
    fig = px.histogram(filtered_df, x=selected_metric, title=f"{selected_metric} by {selected_region}")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader(f"Top Regions by Opens")

    top_regions = df.groupby("oblast")["opened"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_regions)

    st.subheader(f"Top Targets by Opens")
    top_audiences = df.groupby("target")["opened"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_audiences)

    st.subheader(f"Top Brands by Opens")
    top_brands = df.groupby("brand")["opened"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_brands)

    st.subheader("Top communications")
    top_communications = filtered_df.nlargest(5, selected_metric)
    st.dataframe(top_communications[['id_kommunikacii', 'tekst_kommunikacii']])

elif choice == "Channel Statistics":
    st.header("Channel Statistic")

    st.sidebar.subheader("Select Metrics")
    selected_metric = st.sidebar.selectbox("Select metric", ['opened', 'open_rate'])
    channel = st.sidebar.selectbox("Select channel", ["Все"] + list(df_channel['kanal'].unique()))

    df_channel_filtered = df_channel[df_channel["kanal"] == channel] if channel != "Все" else df_channel

    st.subheader("Channel dynamic")
    df_channel_general_grouped = df_channel.groupby('data_kommunikacii').mean()
    df_channel_grouped = df_channel_filtered.groupby('data_kommunikacii').mean()
    fig = px.line(df_channel_grouped,
                  y=df_channel_grouped.ewm(alpha=0.15).mean()[selected_metric]
                  , labels={'data_kommunikacii': 'Date', 'y': 'Opened'}
                  )
    fig.add_scatter(x=df_channel_general_grouped.index,
                    y=(df_channel_general_grouped.ewm(alpha=0.15).mean()[selected_metric]),
                    name='General', line={'color': 'red'},
                    opacity=0.5)
    st.plotly_chart(fig)

    st.subheader("Channel Statistics")
    fig = px.histogram(df_channel_filtered, x=selected_metric, title=f"{selected_metric} by {channel}")
    st.plotly_chart(fig, use_container_width=True)


elif choice == "Generate Messages":
    df_channel_chat = df_channel[['kanal', 'tekst_kommunikacii', 'brand', 'opened', 'open_rate']]
    df_chat = df[['kanal', 'tekst_kommunikacii', 'brand', 'opened', 'open_rate']]
    df_concat = pd.concat([df_channel_chat, df_chat], axis=0)
    st.header("Generate Messages")
    st.sidebar.subheader("Select Category")
    selected_brand = st.sidebar.selectbox("Select Brand", ["Все"] + list(df["brand"].unique()))

    selected_region = st.sidebar.selectbox("Select Region", ["Все"] + list(df['oblast'].unique()))

    filtered_df = df[(df["brand"] == selected_brand)] if selected_brand != "Все" else df
    filtered_df = filtered_df[df['oblast'] == selected_region] if selected_region != "Все" else filtered_df
    st.dataframe(
        filtered_df[["tekst_kommunikacii", "opened", "open_rate"]].sort_values(
            by='opened',
            ascending=False))

    user_input = st.sidebar.text_input("Введите уточнение:", "")

    user_context = user_input if user_input != 'Введите уточнение' else " "

    generate_button = st.sidebar.button("Generate Message")

    prmpt = "Представь, что ты помощник маркетолога и тебе надо сгенерировать сообщение на основе предыдущих, учитывая" \
            "бренд. Длина 1 примера должна быть примерно превышать 110 символов" \
            "сгенерируй на основе следущих сообщений ,бренда и региона:"

    context = filtered_df.sort_values(by="opened", ascending=False).head(10)["tekst_kommunikacii"]

    messages = []


    def generate(cntxt, prm, brand, user_in, temperature=0.3):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Используем модель GPT-3.5 Turbo
            messages=[
                {"role": "user",
                 "content": f"{cntxt} {prm} {brand}, а так же учти уточнение(если оно есть):{user_in}"}
            ],
            # prompt=prompt,
            max_tokens=300,
            temperature=temperature
        )
        generated_message = response.choices[0]["message"]["content"]
        return generated_message


    if generate_button:
        for i in range(5, 10):
            messages.append(generate(context, prmpt, selected_brand, user_context, temperature=i / 10))

        st.subheader("Generated Message")
        st.dataframe(reversed(messages))

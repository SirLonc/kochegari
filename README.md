# kochegari
 A prototype that draws communication dashboards and generates message texts

 Гайд как развернуть на StreamLit.
1. Копируем репозиторий к себе
2.  Идем по инструкции --> https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app
3.  в "Advanced settings for deployment" вставляем следующий Secret : OPENAI_API_KEY = "<ВАШ КЛЮЧ ОПЕНАИ АПИ>"
4.  Done!
Если хотим на localhost, то меняем в app_v3.py "openai.api_key = st.secrets["OPENAI_API_KEY"]" ---> openai.api_key = ""<ВАШ КЛЮЧ ОПЕНАИ АПИ>"" "

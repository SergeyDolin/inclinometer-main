import streamlit as st
import pandas as pd
import seaborn as sns
from streamlit_echarts import st_echarts
from catboost import CatBoostRegressor
import requests
import matplotlib.pyplot as plt
from urllib.parse import urlencode
import py7zr
import os
from scipy import stats
from sklearn.metrics import (
    mean_squared_error
)


# import data
df = pd.read_csv('./test_web.csv')
features = df.drop(['nivel_x','nivel_t'],axis=1)
target = df['nivel_x']
# load model
cat_model_best = CatBoostRegressor()
cat_model_best.load_model('model_1')

pred = cat_model_best.predict(features)


st.title("""
Высокоточный цифровой датчик наклона (инклинометр)
""")

fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(17,10))

sns.lineplot(pd.DataFrame(pred), ax=axes[0], legend=False)
axes[0].set_title('Цифровой инклинометр', fontsize=14)

sns.lineplot(target, ax=axes[1], legend=False)
axes[1].set_title('Lieca', fontsize=14)

st.pyplot(fig)
st.markdown('''После проведения продолжительной измерительной кампании, которая выполнялась совместно с разработанным прототипом и производственным экземпляром
инклинометра, данные о наклонах были обработанны алгоритмом машинного обучения, в результате чего был получен трек представляющий из себя наклон зафиксированный ЦДН. 
Значения наклона в мрад были получены со следующими значениями СКО
''')

st.info('Точность (СКО) цифрового инклинометра: {:.4f} мрад'.format(mean_squared_error(target, cat_model_best.predict(features))))

dif = target - pred
res = stats.shapiro(dif)

fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(17,10))

sns.lineplot(pd.DataFrame(dif[11800:12200]), ax=axes[0], legend=False)
axes[0].set_title('Ошибка измерений в мрад', fontsize=14)

sns.lineplot(pd.DataFrame(dif[9000:10000]), ax=axes[1], legend=False)

st.pyplot(fig)

st.markdown('''Описательная статистика содержит:
* Стандартное отклонение
* Среднее значение
''')
st.info('Стандартное отклонение ошибки: {:.4f} мрад'.format(dif.std()))
st.info('Среднее значение ошибки: {:.4f} мрад'.format(dif.mean()))
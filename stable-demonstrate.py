import streamlit as st
import pandas as pd
import os


# import data
st.title("""
Высокоточный цифровой датчик наклона (инклинометр)
""")
st.header("Анализ влияния температурного эффекта на расширение пузырька уровня", divider=True)

st.markdown('''В ходе разработки цифрового датчика наклона был вявлен дрейф наблюдаемого пузырька без движения установки, который был вызван изменением температуры.
Для компенсации температурного эффекта, который проявлялся как в движении пузырька уровня, так и в его расширении и сжатии, были установлены два датчика температуры, для замера температуры колбы (синий) 
и для замера температуры воздуха (оранжевый) внутри камеры цифрового датчика наклона. 
''')

st.image('./img/TEST/HZ3_TEMP.png')

st.markdown('''Записанные сеансы измерений при неподвижном положении датчика были использованы для предобучения градиентного бустинга, с целью вычислить температурный коэффициент С0.
            Проведенный сеансы измерений эмитировалли позволили зафиксировать наиболее полный температурный диапозон цифрового датчика наклона. Обучение модели выполнялось на сенсе измерений с 
            дискретностью записи в 1 Hz.
''')

st.info('Вычисленный коэффициент смещения из-за температурного эффекта С0 = {:.3f} мрад'.format(-0.013))

st.header("Тестирование модели машинного обучения на сеансе с дискретностью записи измрений в 6 Hz", divider=True)

st.markdown('''Использованные данные были подвергнуты предобработки для исключения срывов сигнала контроля камеры за пузырьком уровня и рассчета дополнительных статестических данных.
На графиках представлены уклонение пузырька уровня при статичном положении установки и изменении температуры, где синий трек - прогноз модели о положении наклона, 
оранжевый - эталонное значение инклинометра Nivel.''')

st.markdown('''Изменение темеператуы во время эксперимента''')
st.image('./img/TEST/TRAIN_TEMP.png')

st.info('Среднеквадратическая ошибка: {:.3f} мрад'.format(0.003))

st.image('./img/TEST/6HZ+C03E3.png')
st.markdown('''Расспределение разностей (эталон-измерение) уклонений цифрового датчика наклона.''')
st.image('./img/TEST/DIFBOX6HZ+C0.png')

st.header("Тестирование модели машинного обучения на сеансе с дискретностью записи измрений в 0.3 Hz", divider=True)

st.markdown('''Изменение темеператуы во время эксперимента''')
st.image('./img/TEST/HZ03_TEMP.png')

st.info('Среднеквадратическая ошибка: {:.3f} мрад'.format(0.002))
st.image('./img/TEST/HZ03+C0_RMS2E3.png')
st.markdown('''Расспределение разностей (эталон-измерение) уклонений цифрового датчика наклона.''')
st.image('./img/TEST/DIFBOX03HZ+C0.png')
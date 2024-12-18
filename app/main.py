import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

# Загрузка данных

data = pd.read_csv(
    'data/mushroom_cleaned.csv')


st.markdown("""
 ## Отчет по курсу НИС "Введение в искусственный интеллект" ##
""")
st.image("https://rossaprimavera.ru/static/files/b5cee02bb333.jpg",
         caption="", use_column_width=True)
st.markdown("""
### Задание и исходные данные ###
**Цель**: выявление подходящей модели для классификации грибов по признаку съедобности/ядовитости.

В качестве исходных данных был выбран датасет [Secondary_Mushroom](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset) из  базы данных UC Irvine Machine Learning Repository. Датасет содержит информацию  вторичного набора данных о грибах и включает данные о 54035 шляпочных грибах 173 видов (по 353 гриба на вид).

Каждый гриб идентифицируется как точно съедобный (class = 0), точно ядовитый (class = 1) или с неизвестной съедобностью и не рекомендуемый к употреблению (последний класс был объединен с классом ядовитых).

Для анализа были выбраны две отличные друг от друга модели: Random Forest (случайный лес), относящийся к деревьям решений и Logistic Regression (логистическая регрессия) как линейный метод. 
Метрики для сравнения: 
- **Accuracy** (точность) измеряет долю правильно классифицированных объектов среди всех объектов
- **Precision** показывает долю истинно положительных объектов среди всех, которые классифицированны как положительные 
- **Recall** (полнота) оценивает долю правильно предсказанных положительных объектов из всех истинно положительных
- **F1 Score** представляет собой гармоническое среднее Precision и Recall, что важно при дисбалансе классов 

""")


st.markdown("""
### Датасет ### """)
st.download_button(
    label="скачать грибы",
    data=data.to_csv(index=False),
    file_name="data.csv",
    mime="text/csv"
)
st.write(data)


st.markdown("""
### Ход работы ### 
#### Предобработка данных ####            
Загрузили датасет. При проверке пропусков в данных обнаружено не было.  Разделили данные на 2 группы по признаку ядовитости для оценки наличия перекоса классов. По результатам видим перекос в сторону ядовитых грибов. 

##### Диаграмма распределения классов #####""")

class_counts = data['class'].value_counts()
fig_class_dist = px.pie(values=class_counts, names=class_counts.index,
                        title="Распределение классов (1 - ядовитые, 0 - съедобные)")
st.plotly_chart(fig_class_dist)

st.markdown("""           
Для дальнейшей работы все категориальные переменные были переведены в числовые значения: значения зависимой переменной были закодированы следующим образом: ядовитые грибы (и почти ядовитые) - 1, съедобные грибы 0. """)

encoded_data = pd.get_dummies(data, drop_first=True)
X = encoded_data.drop('class', axis=1)
y = encoded_data['class']

st.markdown("""
#### Разбиение данных ####            
Данные были разделены на обучающую (0,7) и тестовую (0,3) выборки.
""")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

st.markdown("""
#### Корелляционная матрица ####            
Построили корреляционную матрицу для определения линейных взаимосвязей между признаками, поскольку работаем с задачей бинарной классификации. Был использован коэффициент корреляции Пирсона. .
            """)
corr_matrix = X.corr()
fig_corr = px.imshow(corr_matrix, title="Корреляционная матрица",
                     color_continuous_scale='Viridis')
st.plotly_chart(fig_corr)
st.markdown("""          
При анализе полученной матрицы была обнаружена сильная корреляция между диаметром шляпки (cap-diameter) и толщиной ножки (stem-width) грибов 0.828. Но поскольку оба признака имеют высокий вес для модели рандомного леса, признак было принято решение оставить. Тем более, что нет проблемы мультиколлинеарности. """)

st.markdown("""
#### Обучение моделей ####            
Выбранные модели были обучены на одинаковых данных. Для каждой модели были рассчитаны и выведены значения метрик: Accuracy, Precision, Recall и F1 Score. Построены матрицы ошибок для анализа классификационных ошибок.""")

# Инициализация моделей
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

# Обучение и оценка моделей
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

st.markdown("""
#### Оценка весов признаков для Random Forest ####            
Были определены наиболее важные признаки, такие как "odor", "spore-print-color", "habitat"(запах, цвет спор, среда обитания). Они признаки оказали наибольшее влияние на результат классификации""")
# Оценка важности признаков для модели Random Forest
rf_model = models['Random Forest']
feature_importances = rf_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

st.subheader("Значимости признаков для модели Random Forest")
important_features = pd.DataFrame({
    'Feature': X.columns[sorted_indices[:10]],
    'Importance': feature_importances[sorted_indices[:10]]
})
st.write(important_features)


# Построение матрицы ошибок для каждой модели
for name, model in models.items():
    st.subheader(f"Матрица ошибок для {name}")
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=model.classes_)

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=model.classes_,
        y=model.classes_,
        colorscale='Blues',
        showscale=True
    ))

    fig_cm.update_layout(title=f"Матрица ошибок для {name}")
    st.plotly_chart(fig_cm)

st.markdown("""
#### Сравнение моделей ####                    
**Random Forest**
            
Высокая точность  (Accuracy = 0.9907) , что говорит о высокой способности модели правильно предсказывать общую долю классов. Высокий показатель Precision (0.9913), показывает, что модель редко выдает ложные положительные результаты. Высокий показатель Recall (0.9917) , демонстрирует, что модель успешно предсказывает положительные классы (реальный результат). Высокий F1 Score (0.9915), который учитывает как Precision, так и Recall. Подходит для задач, где важен баланс между ними& 
Высокая точность, полнота (recall) и F1 Score показывают, что модель хорошо работает даже с большим количеством данных. Эффективна для задач с большим объемом и многими переменными.
Однако модель требует больше вычислительных ресурсов и времени для тренировки по сравнению с логистической регрессией, также существует проблема меньшей интерпретируемсти модели из-за сложной структуры дерева решений.

**Logistic Regression**
            
Значение Accuracy = 0.6349 сильно ниже по сравнению с Random Forest, что указывает на меньшую общую способность модели предсказывать. Меньшая точность Precision = 0.6505 свидетельствует о том, что модель часто выдает ложные положительные результаты. В целом достаточно хороший показатель полноты Recall=0.7180, что говорит о приемлемом уровне идентификации положительных классов.
Значение F1 Score = 0.6826 неплохой, однако за счет низкой точности недостаточно хорош для задач с перекосом классов. 
Логистическая регрессия является легкой и интерпритируемой моделью, которая требует меньше вычислительных ресурсов и времени на тренировку. Однако низкое значение Аccuracy говорит о менее надежном предсказании общих классов. К тому же, модель менее эффективна при работе со сложными или большими наборами данных.            
            """)

# Отображение результатов
st.subheader("Сравнение моделей")
results_df = pd.DataFrame(results).T
st.dataframe(results_df.sort_values(by='F1 Score', ascending=False))

st.markdown("""
#### Выводы ####            
Random Forest продемонстрировал лучшие результаты по метрикам качества, что подтверждает способность модели учитывать нелинейные зависимости в данных. Logistic Regression уступает по качеству классификации, но выигрывает в скорости обучения и интерпретируемости.
Преимуществом применения данных моделей является их способность обучаться на больших выборках. К недостаткам логистической регрессии можно отнести способность узнавать лишь линейные зависимости, тогда как решающие деревья способны к восстановлению более сложных закономерностей. Однако обучение случайного леса является более времязатратным процессом. 
            """)

st.markdown("""          
Выполнила: Цембер Софья, БКМБ212
            """)
with open("app/main.py", "r") as file:
    python_code = file.read()
st.download_button(
    label="code",
    data=python_code,
    file_name="code.py",
    mime="text/x-python"
)

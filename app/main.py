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
st.subheader("Первые строки данных")
st.write(data)


# Распределение классов
st.subheader("Распределение классов")
class_counts = data['class'].value_counts()
fig_class_dist = px.pie(values=class_counts, names=class_counts.index,
                        title="Распределение классов (1 - ядовитые, 0 - съедобные)")
st.plotly_chart(fig_class_dist)

# Кодирование категориальных признаков
encoded_data = pd.get_dummies(data, drop_first=True)

# Разделение на признаки и целевую переменную
X = encoded_data.drop('class', axis=1)
y = encoded_data['class']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Корреляционная матрица
st.subheader("Корреляционная матрица")
corr_matrix = X.corr()
fig_corr = px.imshow(corr_matrix, title="Корреляционная матрица",
                     color_continuous_scale='Viridis')
st.plotly_chart(fig_corr)

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

# Отображение интерактивной таблицы с результатами моделей
st.subheader("Сравнение моделей")
results_df = pd.DataFrame(results).T
st.dataframe(results_df.sort_values(by='F1 Score', ascending=False))

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
st.download_button(
    label="скачать грибы",
    data=data.to_csv(index=False),
    file_name="data.csv",
    mime="text/csv"
)

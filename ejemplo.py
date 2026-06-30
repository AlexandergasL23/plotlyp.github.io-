import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import plotly.express as px

print("=" * 60)
print("MACHINE LEARNING CON SCIKIT-LEARN Y PLOTLY")
print("=" * 60)

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["especie"] = pd.Categorical.from_codes(
    iris.target,
    iris.target_names
)

print("\nPrimeras filas del dataset:\n")
print(df.head())

print("\nDimensiones:", df.shape)


fig = px.scatter(
    df,
    x="petal length (cm)",
    y="petal width (cm)",
    color="especie",
    size="sepal length (cm)",
    hover_data=["sepal width (cm)"],
    title="Distribución del Dataset Iris"
)

fig.update_layout(template="plotly_white")
fig.show()


X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


modelo = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

modelo.fit(X_train, y_train)


y_pred = modelo.predict(X_test)


print("\nExactitud del modelo:")
print(round(accuracy_score(y_test, y_pred), 4))

print("\nReporte de clasificación:\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


cm = confusion_matrix(y_test, y_pred)

fig_cm = px.imshow(
    cm,
    text_auto=True,
    x=iris.target_names,
    y=iris.target_names,
    color_continuous_scale="Blues",
    labels=dict(
        x="Predicción",
        y="Valor Real",
        color="Cantidad"
    ),
    title="Matriz de Confusión"
)

fig_cm.show()


importancias = pd.DataFrame({
    "Variable": iris.feature_names,
    "Importancia": modelo.feature_importances_
})

importancias = importancias.sort_values(
    by="Importancia",
    ascending=True
)

fig_imp = px.bar(
    importancias,
    x="Importancia",
    y="Variable",
    orientation="h",
    title="Importancia de Variables"
)

fig_imp.update_layout(template="plotly_white")
fig_imp.show()

print("\nProceso finalizado correctamente.")
#Taller 1 - MLOPS
#Nombres: 


#1. Crear un archivo en Python que consuma los datos, procese los y cree el modelo.

import pandas as pd
from ydata_profiling import ProfileReport

# Cargar el archivo CSV
penguins_lter = pd.read_csv("C:/Users/User/MAESTRIA IA/2. SEMESTRE 2/2. MLOPS/Taller_1/penguins_lter.csv")
penguins_size = pd.read_csv("C:/Users/User/MAESTRIA IA/2. SEMESTRE 2/2. MLOPS/Taller_1/penguins_size.csv")

"""
profile = ProfileReport(penguins_lter, title="Penguins lter Report", explorative=True)
profile.to_notebook_iframe()
profile.to_file("C:/Users/User/MAESTRIA IA/2. SEMESTRE 2/2. MLOPS/Taller_1/penguins_lter_report.html")

profile = ProfileReport(penguins_size, title="Penguins size Report", explorative=True)
profile.to_notebook_iframe()
profile.to_file("C:/Users/User/MAESTRIA IA/2. SEMESTRE 2/2. MLOPS/Taller_1/penguins_size_report.html")
"""

# Limpieza de datos y preparación de datos

#Elimar filas con datos nulos en sex
penguins_size = penguins_size.dropna(subset=['sex'])
penguins_size = penguins_size[penguins_size['sex'] != '.']

# Convertir la variable objetivo 'sex' a valores numéricos (0 para FEMALE y 1 para MALE)
penguins_size['sex'] = penguins_size['sex'].map({'FEMALE': 0, 'MALE': 1})

# Definir variables independientes (X) y la variable objetivo (y)
X = penguins_size[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = penguins_size['sex']

print(penguins_size['sex'].unique())

from sklearn.model_selection import train_test_split

# Dividir el dataset en 70% entrenamiento y 30% restante (que luego se dividirá en validación y prueba)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Dividir el 30% restante en 15% validación y 15% prueba
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Verificar tamaños de los conjuntos
print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# Entrenar el modelo de regresión logística
model = LogisticRegression(random_state=42, max_iter=500,  C= 1, penalty='l2')
model.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de validación
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nValidation Classification Report:\n", classification_report(y_val, y_val_pred))

# Evaluar el modelo en el conjunto de prueba
y_test_pred = model.predict(X_test)
print("test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\ntest Classification Report:\n", classification_report(y_test, y_test_pred))

import joblib

# Guardar el modelo entrenado
joblib.dump(model, 'logistic_regression_model.pkl')
# Importar librerias
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle

# Cargar datos
df = pd.read_csv("/workspaces/codespaces-blank/TESIS_ML_DJANGO/DATA_IUP_2025_arbol_decision.csv")
#df.head(12)
df[5:]

# Diccionario para mapear los meses a valores numéricos
meses_a_numeros = {
    'Enero': 1,'Febrero': 2,'Marzo': 3,'Abril': 4,'Mayo': 5,'Junio': 6,
    'Julio': 7,'Agosto': 8,'Setiembre': 9,'Octubre': 10,'Noviembre': 11,'Diciembre': 12
}

# Crear una nueva columna con los valores numéricos de los meses

df['MESES_NUM'] = df['MESES'].map(meses_a_numeros)

X = df[['ANIO', 'MESES_NUM', 'TOTAL_AVENA']]
y = df['TOTAL_VENTAS_EN_SOLES']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Escalar las características
scaler = MinMaxScaler(feature_range=(0,1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo de árbol de regresión
regressor = DecisionTreeRegressor(max_depth=6)

# Entrenar el modelo
regressor.fit(X_train_scaled, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = regressor.predict(X_test_scaled)

# Evaluar el modelo
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Guardar el modelo y el escalador como archivos .sav
pickle.dump(regressor, open("avena_sales_prediction_model.sav", "wb"))
pickle.dump(scaler, open("avena_sales_scaler.sav", "wb"))

# Función para hacer predicciones
def predict_total_ventas(new_data, model, scaler):
    new_data_scaled = scaler.transform(new_data)
    return model.predict(new_data_scaled)

# Cargar el modelo y el escalador
loaded_model = pickle.load(open("avena_sales_prediction_model.sav", "rb"))
loaded_scaler = pickle.load(open("avena_sales_scaler.sav", "rb"))

# Función para hacer predicciones con nuevos datos
def predict_total_avena(new_data, model):
    # Convertir los nuevos datos en un DataFrame
    new_df = pd.DataFrame(new_data)
    # Hacer la predicción
    prediction = model.predict(new_df)
    return prediction[0]

# Nuevos datos de ejemplo para hacer predicciones
new_data_1 = {'ANIO': [2024], 'MESES_NUM': [1], 'TOTAL_AVENA': [20284]}
new_data_2 = {'ANIO': [2024], 'MESES_NUM': [2], 'TOTAL_AVENA': [16378]}
new_data_3 = {'ANIO': [2024], 'MESES_NUM': [3], 'TOTAL_AVENA': [20069]}
new_data_4 = {'ANIO': [2024], 'MESES_NUM': [4], 'TOTAL_AVENA': [20744]}
new_data_5 = {'ANIO': [2024], 'MESES_NUM': [5], 'TOTAL_AVENA': [21515]}
new_data_6 = {'ANIO': [2024], 'MESES_NUM': [6], 'TOTAL_AVENA': [20540]}
new_data_7 = {'ANIO': [2024], 'MESES_NUM': [7], 'TOTAL_AVENA': [18906]}
new_data_8 = {'ANIO': [2024], 'MESES_NUM': [8], 'TOTAL_AVENA': [21231]}
new_data_9 = {'ANIO': [2024], 'MESES_NUM': [9], 'TOTAL_AVENA': [20263]}
new_data_10 = {'ANIO': [2024], 'MESES_NUM': [10], 'TOTAL_AVENA': [16968]}
new_data_11 = {'ANIO': [2024], 'MESES_NUM': [11], 'TOTAL_AVENA': [17731]}
new_data_12 = {'ANIO': [2024], 'MESES_NUM': [12], 'TOTAL_AVENA': [23560]}

# Lista para almacenar los nuevos datos
new_data_list = [new_data_1, new_data_2, new_data_3, new_data_4, new_data_5,new_data_6,
                new_data_7, new_data_8, new_data_9, new_data_10, new_data_11, new_data_12]
# Lista para almacenar las predicciones
predictions = []

# Hacer predicciones con los nuevos datos y almacenarlas en la lista
for data in new_data_list:
    prediction = predict_total_avena(data, regressor)
    predictions.append(prediction)
# Crear un DataFrame con las predicciones
predictions_df = pd.DataFrame(predictions, columns=['Predicción VENTA_TOTAL_EN_SOLES_X_CANAL'])
# Mostrar el DataFrame
print(predictions_df)
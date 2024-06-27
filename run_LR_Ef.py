import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle

# Cargar datos
df = pd.read_csv("DATA_IUP_2025_Eficiencia_LR.csv")

# Diccionario para mapear los meses a valores numéricos
meses_a_numeros = {
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
    'Julio': 7, 'Agosto': 8, 'Setiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}

# Crear una nueva columna con los valores numéricos de los meses
df['MESES_NUM'] = df['MESES'].map(meses_a_numeros)

X = df[['ANIO', 'MESES_NUM', 'TOTAL_VENTAS_EN_SOLES']]
y = df['TOTAL_AVENA']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Escalar las características
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo de regresión lineal
regressor = LinearRegression()
# Entrenar el modelo
regressor.fit(X_train_scaled, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = regressor.predict(X_test_scaled)

# Evaluar el modelo
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Guardar el modelo y el escalador como archivos .sav
pickle.dump(regressor, open("modelo_regresion_lineal.sav", "wb"))
pickle.dump(scaler, open("modelo_regresion_scaler.sav", "wb"))

# Función para hacer predicciones con nuevos datos
def predict_total_avena(new_data, model, scaler):
    # Convertir los nuevos datos en un DataFrame
    new_df = pd.DataFrame(new_data)
    # Escalar los datos nuevos
    new_df_scaled = scaler.transform(new_df)
    # Hacer la predicción
    prediction = model.predict(new_df_scaled)
    return prediction[0]

# Cargar el modelo y el escalador
loaded_model = pickle.load(open("modelo_regresion_lineal.sav", "rb"))
loaded_scaler = pickle.load(open("modelo_regresion_scaler.sav", "rb"))

# Nuevos datos de ejemplo para hacer predicciones
new_data_1 = {'ANIO': [2024], 'MESES_NUM': [1], 'TOTAL_VENTAS_EN_SOLES': [82750.63]}
new_data_2 = {'ANIO': [2024], 'MESES_NUM': [2], 'TOTAL_VENTAS_EN_SOLES': [76099.16]}
new_data_3 = {'ANIO': [2024], 'MESES_NUM': [3], 'TOTAL_VENTAS_EN_SOLES': [110946.07]}
new_data_4 = {'ANIO': [2024], 'MESES_NUM': [4], 'TOTAL_VENTAS_EN_SOLES': [105725.26]}
new_data_5 = {'ANIO': [2024], 'MESES_NUM': [5], 'TOTAL_VENTAS_EN_SOLES': [112462.45]}
new_data_6 = {'ANIO': [2024], 'MESES_NUM': [6], 'TOTAL_VENTAS_EN_SOLES': [103029.54]}
new_data_7 = {'ANIO': [2024], 'MESES_NUM': [7], 'TOTAL_VENTAS_EN_SOLES': [97973.18]}
new_data_8 = {'ANIO': [2024], 'MESES_NUM': [8], 'TOTAL_VENTAS_EN_SOLES': [102900.81]}
new_data_9 = {'ANIO': [2024], 'MESES_NUM': [9], 'TOTAL_VENTAS_EN_SOLES': [63700.16]}
new_data_10 ={'ANIO': [2024], 'MESES_NUM': [10],'TOTAL_VENTAS_EN_SOLES': [97997.08]}
new_data_11 ={'ANIO': [2024], 'MESES_NUM': [11],'TOTAL_VENTAS_EN_SOLES': [49829.20]}
new_data_12 ={'ANIO': [2024], 'MESES_NUM': [12],'TOTAL_VENTAS_EN_SOLES': [554594.12]}

# Lista para almacenar los nuevos datos
new_data_list = [new_data_1, new_data_2, new_data_3, new_data_4, new_data_5, new_data_6,
                 new_data_7, new_data_8, new_data_9, new_data_10, new_data_11, new_data_12]
# Lista para almacenar las predicciones
predictions = []

# Hacer predicciones con los nuevos datos y almacenarlas en la lista
for data in new_data_list:
    prediction = predict_total_avena(data, loaded_model, loaded_scaler)
    predictions.append(prediction)
    
# Crear un DataFrame con las predicciones
predictions_df = pd.DataFrame(predictions, columns=['Predicción TOTAL_AVENA_EN_CANTIDAD'])
# Mostrar el DataFrame
print(predictions_df)

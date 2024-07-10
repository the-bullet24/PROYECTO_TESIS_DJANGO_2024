import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pickle

# Cargar datos
df = pd.read_csv("DATA_IUP_2025_Productividad_DT.csv")

# Diccionario para mapear los meses a valores numéricos
meses_a_numeros = {
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
    'Julio': 7, 'Agosto': 8, 'Setiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}

# Crear una nueva columna con los valores numéricos de los meses
df['MESES_NUM'] = df['MESES'].map(meses_a_numeros)

# Agregar características cíclicas
df['MES_SIN'] = np.sin(2 * np.pi * df['MESES_NUM'] / 12)
df['MES_COS'] = np.cos(2 * np.pi * df['MESES_NUM'] / 12)

# Seleccionar las características para el modelo
X = df[['ANIO', 'MES_SIN', 'MES_COS', 'TOTAL_VENTAS_EN_SOLES']]
y = df['TOTAL_AVENA']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Escalar las características
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir los parámetros para la búsqueda de hiperparámetros
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1]
}

# Crear el modelo base
base_model = DecisionTreeRegressor(random_state=42)

# Realizar búsqueda de hiperparámetros
grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Obtener el mejor modelo
best_regressor = grid_search.best_estimator_

# Evaluar el modelo
y_pred = best_regressor.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")

# Guardar el modelo y el escalador como archivos .sav
pickle.dump(best_regressor, open("modelo_productividad.sav", "wb"))
pickle.dump(scaler, open("scaler_productividad.sav", "wb"))

# Función para hacer predicciones
def predict_total_avena(anio, avena_por_mes):
    model = pickle.load(open("modelo_productividad.sav", "rb"))
    scaler = pickle.load(open("scaler_productividad.sav", "rb"))
    
    predictions = []
    for mes, total_venta_en_soles in enumerate(avena_por_mes, start=1):
        mes_sin = np.sin(2 * np.pi * mes / 12)
        mes_cos = np.cos(2 * np.pi * mes / 12)
        
        input_data = pd.DataFrame([[anio, mes_sin, mes_cos, total_venta_en_soles]], 
                                  columns=['ANIO', 'MES_SIN', 'MES_COS', 'TOTAL_VENTAS_EN_SOLES'])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]
        predictions.append(prediction)
    
    return predictions

# Ejemplo de uso
anio_prediccion = 2024
avena_por_mes = [82750.63,76099.16,110946.07,105725.26,112462.45,103029.54,97973.18,102900.81,63700.16,97997.08,49829.20,554594.12]

predicciones = predict_total_avena(anio_prediccion, avena_por_mes)

# Crear un DataFrame con las predicciones
predictions_df = pd.DataFrame({
    'MES': range(1, 13),
    'TOTAL_VENTAS_EN_SOLES': avena_por_mes,
    'Predicción Total de venta en cantidad': predicciones
})

print(predictions_df)

# Graficar las predicciones
plt.figure(figsize=(12, 6))
plt.plot(predictions_df['MES'], predictions_df['Predicción Total de venta en cantidad'], marker='o')
plt.title(f'Predicciones de Ventas por Mes ({anio_prediccion})')
plt.xlabel('Mes')
plt.ylabel('Ventas Predichas')
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()

# Imprimir importancia de características
importances = best_regressor.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance}")
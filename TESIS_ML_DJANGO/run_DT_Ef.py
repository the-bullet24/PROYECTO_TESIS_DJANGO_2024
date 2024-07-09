import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle

# Cargar datos
df = pd.read_csv("DATA_IUP_2025_Eficiencia_DT.csv")

# Mapear meses a números
meses_a_numeros = {
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
    'Julio': 7, 'Agosto': 8, 'Setiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}
df['MESES_NUM'] = df['MESES'].map(meses_a_numeros)

# Agregar características cíclicas
df['MES_SIN'] = np.sin(2 * np.pi * df['MESES_NUM'] / 12)
df['MES_COS'] = np.cos(2 * np.pi * df['MESES_NUM'] / 12)

X = df[['ANIO', 'MES_SIN', 'MES_COS', 'TOTAL_AVENA']]
y = df['TOTAL_VENTAS_EN_SOLES']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Escalar las características
scaler = MinMaxScaler(feature_range=(0,1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo
regressor = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
regressor.fit(X_train_scaled, y_train)

# Evaluar el modelo
y_pred = regressor.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Guardar el modelo y el escalador como archivos .sav
pickle.dump(regressor, open("avena_sales_prediction_model.sav", "wb"))
pickle.dump(scaler, open("avena_sales_scaler.sav", "wb"))

# Función para hacer predicciones
def predict_total_avena(anio, avena_por_mes):
    model = pickle.load(open("avena_sales_prediction_model.sav", "rb"))
    scaler = pickle.load(open("avena_sales_scaler.sav", "rb"))
    
    predictions = []
    for mes, total_avena in enumerate(avena_por_mes, start=1):
        mes_sin = np.sin(2 * np.pi * mes / 12)
        mes_cos = np.cos(2 * np.pi * mes / 12)
        
        input_data = pd.DataFrame([[anio, mes_sin, mes_cos, total_avena]], 
                                  columns=['ANIO', 'MES_SIN', 'MES_COS', 'TOTAL_AVENA'])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]
        predictions.append(prediction)
    
    return predictions

# Ejemplo de uso
anio_prediccion = 2024
avena_por_mes = [20000, 22000, 21000, 23000, 24000, 25000, 26000, 25000, 24000, 23000, 22000, 21000]

predicciones = predict_total_avena(anio_prediccion, avena_por_mes)

# Crear un DataFrame con las predicciones
predictions_df = pd.DataFrame({
    'MES': range(1, 13),
    'TOTAL_AVENA': avena_por_mes,
    'Predicción VENTA_TOTAL_EN_SOLES_X_CANAL': predicciones
})

print(predictions_df)

# Graficar las predicciones
plt.figure(figsize=(12, 6))
plt.plot(predictions_df['MES'], predictions_df['Predicción VENTA_TOTAL_EN_SOLES_X_CANAL'], marker='o')
plt.title(f'Predicciones de Ventas por Mes ({anio_prediccion})')
plt.xlabel('Mes')
plt.ylabel('Ventas Predichas')
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()
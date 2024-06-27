from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import MinMaxScaler

# our home page view

def inicio(request):
    return render(request, 'inicio.html')

# def prediccion_data(request):
#     return render(request, 'prediccion_data.html')

# # custom method for generating predictions
# def getPredictions(anio, mes, total_avena):
#     # Obtener la ruta del directorio actual del archivo
#     base_dir = os.path.dirname(os.path.abspath(__file__))
    
#     # Construir la ruta completa a los archivos .sav
#     model_path = os.path.join(base_dir, "avena_sales_prediction_model.sav")
#     scaler_path = os.path.join(base_dir, "avena_sales_scaler.sav")
    
#     # Cargar el modelo y el escalador
#     model = pickle.load(open(model_path, "rb"))
#     scaler = pickle.load(open(scaler_path, "rb"))
    
#     input_data = pd.DataFrame([[anio, mes, total_avena]], columns=['ANIO', 'MESES_NUM', 'TOTAL_AVENA'])
#     input_data_scaled = scaler.transform(input_data)
    
#     prediction = model.predict(input_data_scaled)
    
#     return {
#         "predicted_sales": float(prediction[0])
#     }

def getPredictions(anio):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "avena_sales_prediction_model.sav")
    scaler_path = os.path.join(base_dir, "avena_sales_scaler.sav")
    
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    
    predictions = []
    for mes in range(1, 13):
        # Aquí deberías tener una forma de obtener el total_avena para cada mes
        # Por ahora, usaré un valor fijo como ejemplo
        total_avena = 20000  # Reemplaza esto con la lógica adecuada
        
        input_data = pd.DataFrame([[anio, mes, total_avena]], columns=['ANIO', 'MESES_NUM', 'TOTAL_AVENA'])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        predictions.append(float(prediction[0]))
    
    return predictions

def prediccion_data(request):
    if request.method == 'GET':
        anio = request.GET.get('anio')
        if anio:
            predictions = getPredictions(int(anio))
            return render(request, 'result.html', {'predictions': predictions, 'anio': anio})
    return render(request, 'prediccion_data.html')

# our result page view
def result(request):
    anio = int(request.GET['anio'])
    mes = int(request.GET['mes'])
    total_avena = int(request.GET['total_avena'])
    
    result = getPredictions(anio, mes, total_avena)
    
    return render(request, 'result.html', {'result': result})

def metas(request):
    return render(request, 'metas.html')

def ventas_productos_2019(request):
    data = [
        {"ANIO": 2019, "MESES": "Enero",   "TOTAL_VENTAS_EN_SOLES" : 208329.61},
        {"ANIO": 2019, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 173151.12},
        {"ANIO": 2019, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 277780.46},
        {"ANIO": 2019, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 239057.74},
        {"ANIO": 2019, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 218411.74},
        {"ANIO": 2019, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 181765.09},
        {"ANIO": 2019, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 199520.65},
        {"ANIO": 2019, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 237784.57},
        {"ANIO": 2019, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 230730.52},
        {"ANIO": 2019, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 261194.84},
        {"ANIO": 2019, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 168058.44},
        {"ANIO": 2019, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 149064.12},
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_2019.html', {'json_data': json_data})

def ventas_productos_2020(request):
    data = [
        {"ANIO": 2021, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES": 138844.35},
        {"ANIO": 2021, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 131044.75},
        {"ANIO": 2021, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 212286.76},
        {"ANIO": 2021, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 130482.72},
        {"ANIO": 2021, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 165053.30},
        {"ANIO": 2021, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 248325.50},
        {"ANIO": 2021, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 181248.94},
        {"ANIO": 2021, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 207813.46},
        {"ANIO": 2021, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 220981.02},
        {"ANIO": 2021, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 252145.01},
        {"ANIO": 2021, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 206930.27},
        {"ANIO": 2021, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 146299.85},
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_2020.html', {'json_data': json_data})

def ventas_productos_2021(request):
    data = [
        {"ANIO": 2021, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES": 188761.79},
        {"ANIO": 2021, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 258132.35},
        {"ANIO": 2021, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 181260.41},
        {"ANIO": 2021, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 195483.21},
        {"ANIO": 2021, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 211598.56},
        {"ANIO": 2021, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 210256.57},
        {"ANIO": 2021, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 260311.65},
        {"ANIO": 2021, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 239975.34},
        {"ANIO": 2021, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 188394.75},
        {"ANIO": 2021, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 132811.13},
        {"ANIO": 2021, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 175640.11},
        {"ANIO": 2021, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 138167.62},
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_2021.html', {'json_data': json_data})

def ventas_productos_2022(request):
    data = [
        {"ANIO": 2021, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES": 174745.45},
        {"ANIO": 2021, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 171740.31},
        {"ANIO": 2021, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 268799.45},
        {"ANIO": 2021, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 220464.87},
        {"ANIO": 2021, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 224318.79},
        {"ANIO": 2021, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 260690.16},
        {"ANIO": 2021, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 244976.26},
        {"ANIO": 2021, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 213663.16},
        {"ANIO": 2021, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 166108.54},
        {"ANIO": 2021, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 164571.56},
        {"ANIO": 2021, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 173609.92},
        {"ANIO": 2021, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 187110.11},
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_2022.html', {'json_data': json_data})

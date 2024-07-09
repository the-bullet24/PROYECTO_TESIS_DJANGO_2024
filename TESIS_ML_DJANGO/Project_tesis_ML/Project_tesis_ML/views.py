from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import MinMaxScaler
from django.http import JsonResponse


# our home page view

def inicio(request):
    return render(request, 'inicio.html')


# Función para obtener predicciones - Efciecia 
def getPredictions(anio, avena_por_mes):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "avena_sales_prediction_model.sav")
    scaler_path = os.path.join(base_dir, "avena_sales_scaler.sav")
    
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    
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


def prediccion_data(request):
    months = range(1, 13)
    return render(request, 'prediccion_data.html', {'months': months})

def result(request):
    if request.method == 'POST':
        anio = int(request.POST['anio'])
        avena_por_mes = [int(request.POST[f'avena_mes_{i}']) for i in range(1, 13)]
        
        predictions = getPredictions(anio, avena_por_mes)
        
        result = {
            'anio': anio,
            'avena_por_mes': avena_por_mes,
            'predictions': predictions,
        }
        
        return JsonResponse(result)
    else:
        return render(request, 'prediccion_data.html')


##VENTAS PRODUCTOS EFICIENCIA -

def ventas_productos_soles_2019(request):
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
    
    return render(request, 'ventas_productos_soles_2019.html', {'json_data': json_data})

def ventas_productos_soles_2020(request):
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
    
    return render(request, 'ventas_productos_soles_2020.html', {'json_data': json_data})

def ventas_productos_soles_2021(request):
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
    
    return render(request, 'ventas_productos_soles_2021.html', {'json_data': json_data})

def ventas_productos_soles_2022(request):
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
    
    return render(request, 'ventas_productos_soles_2022.html', {'json_data': json_data})


def ventas_productos_soles_2023(request):
    data = [
        {"ANIO": 2023, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES": 232657.48},
        {"ANIO": 2023, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 187855.66},
        {"ANIO": 2023, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 230191.43},
        {"ANIO": 2023, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 237933.68},
        {"ANIO": 2023, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 246777.05},
        {"ANIO": 2023, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 235593.80},
        {"ANIO": 2023, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 216851.82},
        {"ANIO": 2023, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 243519.57},
        {"ANIO": 2023, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 232416.61},
        {"ANIO": 2023, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 194622.96},
        {"ANIO": 2023, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 203374.57},
        {"ANIO": 2023, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 270233.20},        
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_soles_2023.html', {'json_data': json_data}) 



# Datos para 209,2020,2021,2022,2023

def presentacion_graficos(request, year):
    # Aquí deberías tener lógica para obtener los datos del año específico
    # Por ejemplo:
    data = {
           '2019': [  # Datos para 2019         
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
                ],  
        '2020': [ # Datos para 2020
                {"ANIO": 2020, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES": 138844.35},
                {"ANIO": 2020, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 131044.75},
                {"ANIO": 2020, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 212286.76},
                {"ANIO": 2020, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 130482.72},
                {"ANIO": 2020, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 165053.30},
                {"ANIO": 2020, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 248325.50},
                {"ANIO": 2020, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 181248.94},
                {"ANIO": 2020, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 207813.46},
                {"ANIO": 2020, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 220981.02},
                {"ANIO": 2020, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 252145.01},
                {"ANIO": 2020, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 206930.27},
                {"ANIO": 2020, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 146299.85},                                                            
                ],  
        '2021': [ # Datos para 2021
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
                ],  
        '2022': [ # Datos para 2022
                {"ANIO": 2022, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES": 174745.45},
                {"ANIO": 2022, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 171740.31},
                {"ANIO": 2022, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 268799.45},
                {"ANIO": 2022, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 220464.87},
                {"ANIO": 2022, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 224318.79},
                {"ANIO": 2022, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 260690.16},
                {"ANIO": 2022, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 244976.26},
                {"ANIO": 2022, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 213663.16},
                {"ANIO": 2022, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 166108.54},
                {"ANIO": 2022, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 164571.56},
                {"ANIO": 2022, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 173609.92},
                {"ANIO": 2022, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 187110.11},            
                ], 
        
        '2023': [ # Datos para 2023
                {"ANIO": 2023, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES": 232657.48},
                {"ANIO": 2023, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 187855.66},
                {"ANIO": 2023, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 230191.43},
                {"ANIO": 2023, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 237933.68},
                {"ANIO": 2023, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 246777.05},
                {"ANIO": 2023, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 235593.80},
                {"ANIO": 2023, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 216851.82},
                {"ANIO": 2023, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 243519.57},
                {"ANIO": 2023, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 232416.61},
                {"ANIO": 2023, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 194622.96},
                {"ANIO": 2023, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 203374.57},
                {"ANIO": 2023, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 270233.20},            
                ], 
            }
    
    return JsonResponse(data.get(str(year), []), safe=False)



##VENTAS EN SOLES PRODUCTIVIDAD -
  
##Prediccion para PRODUCTIVIDAD
  
def getPredictions_produccion(anio, avena_por_mes):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "modelo_productividad.sav")
    scaler_path = os.path.join(base_dir, "scaler_productividad.sav")
    
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    
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


def prediccion_data_prod(request):
    months = range(1, 13)
    return render(request, 'prediccion_data_prod.html', {'months': months})

def result_prod(request):
    if request.method == 'POST':
        anio = int(request.POST['anio'])
        avena_por_mes = [int(request.POST[f'avena_mes_{i}']) for i in range(1, 13)]
        
        predictions = getPredictions_produccion(anio, avena_por_mes)
        
        result = {
            'anio': anio,
            'avena_por_mes': avena_por_mes,
            'predictions': predictions,
        }
        
        return JsonResponse(result)
    else:
        return render(request, 'prediccion_data_prod.html')


  
  
  
  
# Datos para 209,2020,2021,2022,2023

def presentacion_graficos_productividad(request, year):
    # Aquí deberías tener lógica para obtener los datos del año específico
    # Por ejemplo:
    data = {
           '2019': [  # Datos para 2019         
                  {"ANIO": 2019, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 6851},
                  {"ANIO": 2019, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 7830},
                  {"ANIO": 2019, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 10391},
                  {"ANIO": 2019, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 8944},
                  {"ANIO": 2019, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 9147},
                  {"ANIO": 2019, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 8827},
                  {"ANIO": 2019, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 5333},
                  {"ANIO": 2019, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 12540},
                  {"ANIO": 2019, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 9976},
                  {"ANIO": 2019, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 11249},
                  {"ANIO": 2019, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 5179},
                  {"ANIO": 2019, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 6146},     
                ],  
        '2020': [ # Datos para 2020
                  {"ANIO": 2020, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 8031},
                  {"ANIO": 2020, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 6548},
                  {"ANIO": 2020, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 10193},
                  {"ANIO": 2020, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 10981},
                  {"ANIO": 2020, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 19125},
                  {"ANIO": 2020, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 31149},
                  {"ANIO": 2020, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 19777},
                  {"ANIO": 2020, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 24904},
                  {"ANIO": 2020, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 20170},
                  {"ANIO": 2020, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 22371},
                  {"ANIO": 2020, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 14748},
                  {"ANIO": 2020, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 7728},                                                              
                ],     
        '2021': [ # Datos para 2021   
                  {"ANIO": 2021, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 15037},
                  {"ANIO": 2021, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 14703},
                  {"ANIO": 2021, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 17930},
                  {"ANIO": 2021, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 16100},
                  {"ANIO": 2021, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 20781},
                  {"ANIO": 2021, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 18880},
                  {"ANIO": 2021, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 25102},
                  {"ANIO": 2021, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 15355},
                  {"ANIO": 2021, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 17152},
                  {"ANIO": 2021, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 10797},
                  {"ANIO": 2021, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 17137},
                  {"ANIO": 2021, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 15395},                               
                ],     
        '2022': [ # Datos para 2022   
                  {"ANIO": 2022, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 13582},
                  {"ANIO": 2022, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 14507},
                  {"ANIO": 2022, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 24101},
                  {"ANIO": 2022, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 22584},
                  {"ANIO": 2022, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 13921},
                  {"ANIO": 2022, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 15337},
                  {"ANIO": 2022, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 22908},
                  {"ANIO": 2022, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 19603},
                  {"ANIO": 2022, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 15573},
                  {"ANIO": 2022, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 19863},
                  {"ANIO": 2022, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 21991},
                  {"ANIO": 2022, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 23326},           
                ],    
        '2023': [ # Datos para 2023   
                  {"ANIO": 2023, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 19313},
                  {"ANIO": 2023, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 17428},
                  {"ANIO": 2023, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 25358},
                  {"ANIO": 2023, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 23888},
                  {"ANIO": 2023, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 25526},
                  {"ANIO": 2023, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 23454},
                  {"ANIO": 2023, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 22228},
                  {"ANIO": 2023, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 23226},
                  {"ANIO": 2023, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 14646},
                  {"ANIO": 2023, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 21564},
                  {"ANIO": 2023, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 10514},
                  {"ANIO": 2023, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 11914},           
                ],    
            }    
    return JsonResponse(data.get(str(year), []), safe=False)
  
  
def ventas_productos_cantidad_2019(request):
    data = [
        {"ANIO": 2019, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 6851},
        {"ANIO": 2019, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 7830},
        {"ANIO": 2019, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 10391},
        {"ANIO": 2019, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 8944},
        {"ANIO": 2019, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 9147},
        {"ANIO": 2019, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 8827},
        {"ANIO": 2019, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 5333},
        {"ANIO": 2019, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 12540},
        {"ANIO": 2019, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 9976},
        {"ANIO": 2019, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 11249},
        {"ANIO": 2019, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 5179},
        {"ANIO": 2019, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 6146},    
    
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_cantidad_2019.html', {'json_data': json_data})

def ventas_productos_cantidad_2020(request):
    data = [
         {"ANIO": 2020, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 8031},
         {"ANIO": 2020, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 6548},
         {"ANIO": 2020, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 10193},
         {"ANIO": 2020, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 10981},
         {"ANIO": 2020, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 19125},
         {"ANIO": 2020, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 31149},
         {"ANIO": 2020, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 19777},
         {"ANIO": 2020, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 24904},
         {"ANIO": 2020, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 20170},
         {"ANIO": 2020, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 22371},
         {"ANIO": 2020, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 14748},
         {"ANIO": 2020, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 7728},    
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_cantidad_2020.html', {'json_data': json_data})

def ventas_productos_cantidad_2021(request):
    data = [
         {"ANIO": 2021, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 15037},
         {"ANIO": 2021, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 14703},
         {"ANIO": 2021, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 17930},
         {"ANIO": 2021, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 16100},
         {"ANIO": 2021, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 20781},
         {"ANIO": 2021, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 18880},
         {"ANIO": 2021, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 25102},
         {"ANIO": 2021, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 15355},
         {"ANIO": 2021, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 17152},
         {"ANIO": 2021, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 10797},
         {"ANIO": 2021, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 17137},
         {"ANIO": 2021, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 15395},    
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_cantidad_2021.html', {'json_data': json_data})

  
def ventas_productos_cantidad_2022(request):
    data = [
         {"ANIO": 2022, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 13582},
         {"ANIO": 2022, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 14507},
         {"ANIO": 2022, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 24101},
         {"ANIO": 2022, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 22584},
         {"ANIO": 2022, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 13921},
         {"ANIO": 2022, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 15337},
         {"ANIO": 2022, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 22908},
         {"ANIO": 2022, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 19603},
         {"ANIO": 2022, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 15573},
         {"ANIO": 2022, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 19863},
         {"ANIO": 2022, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 21991},
         {"ANIO": 2022, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 23326},    
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_cantidad_2022.html', {'json_data': json_data})

def ventas_productos_cantidad_2023(request):
    data = [
         {"ANIO": 2023, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 19313},
         {"ANIO": 2023, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 17428},
         {"ANIO": 2023, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 25358},
         {"ANIO": 2023, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 23888},
         {"ANIO": 2023, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 25526},
         {"ANIO": 2023, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 23454},
         {"ANIO": 2023, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 22228},
         {"ANIO": 2023, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 23226},
         {"ANIO": 2023, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 14646},
         {"ANIO": 2023, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 21564},
         {"ANIO": 2023, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 10514},
         {"ANIO": 2023, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 11914},        
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_cantidad_2023.html', {'json_data': json_data})



                  












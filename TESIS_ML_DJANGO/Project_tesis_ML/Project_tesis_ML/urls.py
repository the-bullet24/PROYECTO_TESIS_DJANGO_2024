"""
URL configuration for Project_tesis_ML project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from . import views 

urlpatterns = [
    path("admin/", admin.site.urls),
    path('', views.inicio, name='inicio'),
    
    path('result/prediccion_data/', views.prediccion_data, name='prediccion_data'),
    path('result/', views.result, name='result'),
    path('result/ventas_productos_2019/', views.ventas_productos_2019, name='ventas_productos_2019'),
    path('result/ventas_productos_2020/', views.ventas_productos_2020, name='ventas_productos_2020'),
    path('result/ventas_productos_2021/', views.ventas_productos_2021, name='ventas_productos_2021'),
    path('result/ventas_productos_2022/', views.ventas_productos_2022, name='ventas_productos_2022'),
    
    path('result/ventas_productos_2019/presentacion_graficos/<int:year>/', views.presentacion_graficos, name='presentacion_graficos'),

    ##Ventas en soles 2021 
    path('result_prod/prediccion_data_prod/', views.prediccion_data_prod, name='prediccion_data_prod'),
    path('result_prod/', views.result_prod, name='result_prod'),
    path('result_prod/ventas_soles_2019/', views.ventas_soles_2019, name='ventas_soles_2019'),
    path('result_prod/ventas_soles_2020/', views.ventas_soles_2020, name='ventas_soles_2020'),
    path('result_prod/ventas_soles_2021/', views.ventas_soles_2021, name='ventas_soles_2021'),
    path('result_prod/ventas_soles_2022/', views.ventas_soles_2022, name='ventas_soles_2022'),
   
    path('result_prod/ventas_soles_2019/presentacion_graficos/<int:year>/', views.presentacion_graficos_productividad, name='presentacion_graficos_productividad'),
    
       
]
from django.urls import path
from . import views

urlpatterns =[
    path("", views.index, name="index"),
    path("load_data", views.load_data, name="load_data"), 
    path("train_model", views.train_model, name="train_model"),
    path("predict_model", views.predict_model, name="predict_model"),
]
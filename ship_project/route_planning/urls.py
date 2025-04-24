from django.urls import path
from . import views

urlpatterns = [
    path('', views.route_list_view, name='route_list'),
    path('<int:pk>/', views.route_detail_view, name='route_detail'),
    path('create/', views.route_create_view, name='route_create'),
    path('<int:pk>/plan/', views.route_plan_view, name='route_plan'),
    path('<int:pk>/delete/', views.route_delete_view, name='route_delete'),
] 
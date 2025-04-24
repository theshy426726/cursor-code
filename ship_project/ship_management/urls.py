from django.urls import path
from . import views

urlpatterns = [
    path('', views.ship_list_view, name='ship_list'),
    path('<int:pk>/', views.ship_detail_view, name='ship_detail'),
    path('create/', views.ship_create_view, name='ship_create'),
    path('<int:pk>/update/', views.ship_update_view, name='ship_update'),
    path('<int:pk>/delete/', views.ship_delete_view, name='ship_delete'),
] 
from django.urls import path
from . import views

urlpatterns = [
    # 认证相关URL
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    
    # 个人信息管理URL
    path('profile/', views.profile_view, name='profile'),
    path('change-password/', views.change_password_view, name='change_password'),
    
    # 管理员用户管理URL
    path('', views.user_list_view, name='user_list'),
    path('<int:pk>/', views.user_detail_view, name='user_detail'),
    path('create/', views.user_create_view, name='user_create'),
    path('<int:pk>/update/', views.user_update_view, name='user_update'),
    path('<int:pk>/delete/', views.user_delete_view, name='user_delete'),
] 
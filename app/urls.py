from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    path("", views.index, name= "home"),
    path("log_in", views.log_in, name= "log_in"),
    path("admin_log_in", views.admin_log_in, name= "admin_log_in"),
    path("register", views.register, name= "register"),
    path("dashboard", views.dashboard, name= "dashboard"),
    path("admin_dashboard", views.admin_dashboard, name= "admin_dashboard"),
    path("delete_user", views.delete_user, name= "delete_user"),
    # path('notify', views.fire_notify, name='fire_notify'),
    # path('check_notifications', views.check_notifications, name='check_notifications'),
    path("log_out", views.log_out, name= "log_out"),
]

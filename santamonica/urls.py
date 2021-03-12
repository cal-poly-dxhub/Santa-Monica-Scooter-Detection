from django.urls import path

from . import views

from django.views.generic.base import RedirectView


urlpatterns = [

    path('', views.showvideo, name='showvideo'),
    path('processed',views.showvideop,name='processed')
]
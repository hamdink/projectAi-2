from django.urls import path
from .views import your_view

urlpatterns = [
    path('', your_view, name='display_data'),
]
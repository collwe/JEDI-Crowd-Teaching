from django.conf import settings
from django.conf.urls import url, include
from .views import *

urlpatterns = [
  url(r'^$', index, name='index'),
]
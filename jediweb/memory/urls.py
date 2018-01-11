from django.conf import settings
from django.conf.urls import url, include
from .views import *

urlpatterns = [
  url(r'^home$', home, name='memory_home'),
  url(r'^images', images, name='memory_show_images'),
  url(r'^images_test', images_test, name='memory_show_images'),
  url(r'^get_images', get_images, name='memory_show_images'),
]
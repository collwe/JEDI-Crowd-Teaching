from django.conf import settings
from django.conf.urls import url, include
from .views import *

urlpatterns = [
  url(r'^index/$', home, name='memory_home'),
  url(r'^images/(?P<n_img>[0-9]{2})/', images, name='memory_images'),
  url(r'^images_test/', images_test, name='memory_images_test'),
  url(r'^get_images/(?P<n_img>[0-9]{2})/', get_images, name='memory_get_images'),
  url(r'^get_images_test/', get_images_test, name='memory_get_images_test'),
  url(r'^check_order/', check_order, name='memory_check_order'),
  url(r'^completed/', completed, name='memory_completed'),

]
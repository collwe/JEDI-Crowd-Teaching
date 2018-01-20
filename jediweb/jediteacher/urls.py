from django.conf import settings
from django.conf.urls import url, include
from .views import *

urlpatterns = [
  url(r'^index/$', index, name='jedi_teacher_home'),
  url(r'^start/(?P<token>[\w\.-]+)$', start, name='jedi_teacher_start'),
  url(r'^play/$', play, name='jedi_teacher_play'),
  url(r'^feedback/$', feedback, name='jedi_teacher_feedback'),
  url(r'^api-auth/', include('rest_framework.urls'))
]
from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class JediUser(models.Model):
  # Member variables for User
  user = models.ForeignKey(User, on_delete=models.CASCADE)
  is_finished = models.BooleanField(default=False)
  algorithm = models.TextField(default="")

class JediImages(models.Model):
  file_id = models.IntegerField()
  label = models.TextField()
  category = models.TextField()
  filename =  models.TextField()
  enc_filename = models.TextField()


class UserLabels(models.Model):
  user = models.ForeignKey(User, on_delete=models.CASCADE)
  file_id = models.IntegerField()
  yl = models.IntegerField()
  y = models.IntegerField()
  created_at = models.DateTimeField(auto_now_add=True)
  updated_at = models.DateTimeField(auto_now=True)
  algorithm = models.TextField(default="")
  mode = models.TextField(default="")

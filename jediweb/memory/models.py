from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class MemoryImages(models.Model):
  file_id = models.IntegerField()
  file_name = models.TextField()


class MemoryTest(models.Model):
  user = models.ForeignKey(User, on_delete=models.CASCADE)
  score = models.IntegerField()


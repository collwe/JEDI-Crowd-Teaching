from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend
from rest_framework.authtoken.models import Token

class HashModelBackend(object):
  def authenticate(self, token=None, **kwargs):
    UserModel = get_user_model()
    if token is not None:
      try:
        token_user = Token.objects.get(key=token)
        user = UserModel.objects.get(id=token_user.user_id)
        return user
      except:
        return None

  def get_user(self, user_id):
    UserModel = get_user_model()
    try:
      return UserModel.objects.get(pk=user_id)
    except UserModel.DoesNotExist:
      return None
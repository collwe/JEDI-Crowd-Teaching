from django import forms
from captcha.fields import CaptchaField
from django.contrib.auth.models import User

from django import forms
from captcha.fields import CaptchaField

class UserForm(forms.Form):
  name = forms.CharField(max_length=200)
  email = forms.EmailField(max_length=200)
  affiliation = forms.CharField(max_length=200)
  captcha = CaptchaField()

  def clean_email(self):
    email = self.cleaned_data['email']
    try:
      user = User.objects.get(username=email)
    except User.DoesNotExist:
      return email
    raise forms.ValidationError(u'Email "%s" is already in use.' % email)


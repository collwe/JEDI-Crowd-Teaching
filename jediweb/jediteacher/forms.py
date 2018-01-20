from django import forms
from captcha.fields import CaptchaField
from .models import JediUser

from django import forms
from captcha.fields import CaptchaField


class LabelForm(forms.Form):
    CHOICES = [('domestic', 'Domestic'),
               ('wild', 'Wild')]

    label_option = forms.ChoiceField(choices=CHOICES, widget=forms.RadioSelect())


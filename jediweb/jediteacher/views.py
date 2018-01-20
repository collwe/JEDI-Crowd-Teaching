from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login

from .forms import LabelForm
from .jedi_helper import get_next
from .models import JediImages, UserLabels


def index(request):
  CATEGORY = "cats"
  ALGORITHM = "jedi"

  # form = JediUserModelForm()

  data = {}
  # data['form'] = form

  return render(request, 'jedi_teaching/home.html', data)


def start(request, token):
  # Login/ Authenticate user using a token.

  user = authenticate(token)
  if user is not None:
    login(request, user)

    # Setup.
    category = 'Horse'
    n_teaching = 5
    n_test = 5

    request.session['algorithm'] = 'rt'
    request.session['category'] = category
    request.session['n_teaching'] = n_teaching
    request.session['n_test'] = n_test
    request.session['c_teaching'] = 0
    request.session['c_test'] = 0

    request.session['mode'] = 'teaching'

    ids = JediImages.objects.filter(category=category).all().values_list('id')
    imgs_ids = []
    for id in ids:
      imgs_ids.append(int(id[0]))

    print(imgs_ids)
    request.session['image_ids'] = imgs_ids

    data = {}
    # data['form'] = form
    return render(request, 'jedi_teaching/home.html', data)
  else:
    return render(request, 'common/error.html')


def play(request):
  # Get Mode
  mode = request.session['mode']
  data = {}

  # Get the next image to show to the user..
  img = JediImages.objects.get(id=get_next(request))

  if request.session['mode'] == 'test':
    print('C_TEST',request.session['c_test'])

    if request.session['c_test'] >= request.session['n_test']:
      return render(request, 'jedi_teaching/completed.html', data)

  else:
    print('C_TEACHING',request.session['c_teaching'])
    if request.session['c_teaching'] >= request.session['n_teaching']:
      request.session['mode'] = 'test'
      return render(request, 'jedi_teaching/test_mode.html', data)




  data['image'] = img.enc_filename
  data['label'] = ''
  data['options'] = ''
  data['image_id'] = img.id
  print(img.enc_filename)

  form = LabelForm()
  data['form'] = LabelForm()

  return render(request, 'jedi_teaching/play.html', data)


def feedback(request):
  correct = False

  if request.method == 'POST':
    form = LabelForm(request.POST)
    if form.is_valid():
      label_option = form.cleaned_data['label_option']
      image_id = request.POST['image_id']
      img = JediImages.objects.get(id=image_id)

      print(label_option, img.label)

      if label_option == img.label:
        correct = True

      data = {}
      data['image'] = img.enc_filename
      data['label'] = ''
      data['options'] = ''
      data['correct'] = correct
      data['answer'] = 'It is a %s %s.' % (img.label, img.category.lower())

      if label_option == 'domestic':
        yl = 1
      else:
        yl = 2

      if img.label == 'domestic':
        y = 1
      else:
        y = 2

      user_label = UserLabels()
      user_label.user = request.user
      user_label.y = y
      user_label.yl = yl
      user_label.file_id = img.id
      user_label.algorithm = request.session['algorithm']
      user_label.mode = request.session['mode']
      user_label.save()

      if request.session['mode'] == 'test':
        request.session['c_test'] +=  1
        return redirect('jedi_teacher_play')
      else:
        request.session['c_teaching'] += 1
        return render(request, 'jedi_teaching/feedback.html', data)

  else:
    return redirect('jedi_teacher_play')

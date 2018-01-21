from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
import numpy as np
from .forms import LabelForm
from .jedi_helper import get_next
from .models import JediImages, UserLabels
from memory.models import MemoryTest


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
    category = 'Cat'
    n_teaching = 10
    n_test = 30

    # Compute the beta parameter and the number of training examples based on the performance in memory test.
    scores = []
    for score in MemoryTest.objects.filter(user=request.user).all():
      scores.append(score.score)

    scores = np.sort(scores)
    score = np.average(scores[-2:])

    if score <=4.5:
      n_teaching = 20
    elif score > 4.5 and score <=6.5:
      n_teaching = 30
    elif score> 6.5:
      n_teaching = 40

    beta = 1 - 1/score

    print(score,beta)

    # Set the session variables.
    request.session['beta'] = beta
    request.session['algorithm'] = 'jedi'
    request.session['category'] = category
    request.session['n_teaching'] = n_teaching
    request.session['n_test'] = n_test
    request.session['c_teaching'] = 0
    request.session['c_test'] = 0
    request.session['ts_order'] = []
    request.session['ysl_prob'] = []
    request.session['ysl'] = []
    request.session['mode'] = 'teaching'
    request.session['ev_order'] = []


    ids = JediImages.objects.filter(category=category).all().values_list('id')
    imgs_ids = []
    for id in ids:
      imgs_ids.append(int(id[0]))

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



  img_idx, img_name = get_next(request)
  print(img_name)
  # Get the next image to show to the user..
  img = JediImages.objects.get(filename=img_name)

  if request.session['mode'] == 'test':
    request.session['ev_order'] = request.session['ev_order'] + [img_idx]
    if request.session['c_test'] >= request.session['n_test']:
      return render(request, 'jedi_teaching/completed.html', data)

    data['curr_image_no'] = request.session['c_test'] + 1
    data['total_image_no'] = request.session['n_test']

  else:
    print('C_TEACHING',request.session['c_teaching'])
    if request.session['c_teaching'] >= request.session['n_teaching']:
      request.session['mode'] = 'test'
      return render(request, 'jedi_teaching/test_mode.html', data)

    data['curr_image_no'] = request.session['c_teaching'] + 1
    data['total_image_no'] = request.session['n_teaching']


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
        yl = -1

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

      request.session['ysl'] = request.session['ysl'] + [yl]

      if request.session['mode'] == 'test':
        request.session['c_test'] +=  1
        return redirect('jedi_teacher_play')
      else:
        request.session['c_teaching'] += 1
        return render(request, 'jedi_teaching/feedback.html', data)

  else:
    return redirect('jedi_teacher_play')

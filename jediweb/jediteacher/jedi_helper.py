import logging

import numpy as np

logger = logging.getLogger(__name__)
from django.conf import settings
from .jedi_recommender import jedi_next


def get_jedi_next(request):
  category = request.session['category']

  Dt = settings.DATASET[category]['Dt']
  Yt = settings.DATASET[category]['Yt']
  Yt = [-1 if(x==2) else 1 for x in Yt]
  Yt = np.reshape(np.array(Yt),(len(Yt),1))

  De = settings.DATASET[category]['De']
  Ye = settings.DATASET[category]['Ye']
  Ye = [-1 if(x==2) else 1 for x in Ye]
  Ye = np.reshape(np.array(Ye),(len(Ye),1))

  A = settings.DATASET[category]['A']
  wo_SGD = settings.DATASET[category]['wo_SGD']

  beta = request.session['beta']

  order = request.session['ts_order']
  ysl_prob = request.session['ysl_prob']
  ysl = request.session['ysl']
  selectIdx, selectProb = jedi_next(Dt, Yt, De, Ye, order, ysl_prob, ysl, A, wo_SGD, beta)
  return selectIdx, selectProb


def get_next(request):

  algorithm = request.session['algorithm']
  current_teaching_image = request.session['c_teaching']
  category = request.session['category']
  train_images = settings.DATASET[category]['Names_t']
  test_images = settings.DATASET[category]['Names_e']

  if request.session['mode'] == 'test':
    img_idx = int(np.random.random_integers(0, len(test_images), 1)[0])
    while img_idx in request.session['ev_order']:
      img_idx = int(np.random.random_integers(0, len(test_images), 1)[0])

    img_name = test_images[img_idx][0][0]

  else:
    if current_teaching_image == 0:
      algorithm = 'rt'

    img_idx = 0

    print('Returning an id using %s' % algorithm)

    if algorithm == 'eer':
      pass

    elif algorithm == 'imt':
      pass

    elif algorithm == 'jedi':
      print('==> Fetching through JEDI.')
      img_idx, img_prob = get_jedi_next(request)
      request.session['ysl_prob'] = request.session['ysl_prob'] + [img_prob.tolist()]

      # Fetch the previous ones.


      # Pick the next image.

    else:
      img_idx = int(np.random.random_integers(0, len(train_images), 1)[0])
      request.session['ysl_prob'] = request.session['ysl_prob'] + [[0.5, 0.5]]

    request.session['ts_order'] = request.session['ts_order'] + [int(img_idx)]
    img_name = train_images[img_idx][0][0]

  return img_idx, img_name

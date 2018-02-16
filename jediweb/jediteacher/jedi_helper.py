import logging

import numpy as np

logger = logging.getLogger(__name__)
from django.conf import settings
from .jedi_recommender import jedi_next


def get_jedi_next(request, beta):
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

  order = request.session['ts_order']
  ysl_prob = request.session['ysl_prob']
  ysl = request.session['ysl']
  selectIdx, selectProb = jedi_next(Dt, Yt, De, Ye, order, ysl_prob, ysl, A, wo_SGD, beta)
  return selectIdx, selectProb

def get_eer_next(request):
  # Based on "Zhu et al., Combining Active Learning and Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions, in ICML workshop 2003"

  # Input:
  # Below, nS = total number of samples, nC = number of classes, nL = number of observed samples, nT = total number of testing samples to be shown to user
  # X: nS*nC belief matrix, with each row representing one sample, and each column representing one column. Each element is the probability that the user thinks that sample is assigned to that class. This would be identical to Y if we were assuming that the user always assigns the ground truth to an observed sample, and never has memory fall-off.
  # Y: nS*nC ground truth matrix (NumPy array), with each row in indicator encoding. This represents the ground-truth labels of all points. As such, each row has only one "1" and all other entries are "0".
  # W: nS*nS graph weights matrix (symmetrical NumPy array), with each row and each column corresponding to one sample. Each element is the weight (affinity) between two samples.
  # L: nL*1 labeled set, where each element is one sample that has already been shown to the user, with indices between 1 and nS.
  # testing_samples: nT*1 testing set, where each element is one sample that will (or has already been) shown to the user as a testing image. This is to prevent testing images being shown during teaching.
  # mode: the teaching mode (2 = worst predicted, 3 = our method)
  # Output:
  #    next_sample: the index of the optimum sample to be shown next, as selected by the active teaching algorithm.
  category = request.session['category']
  Dt = settings.DATASET[category]['Dt']
  Yt = settings.DATASET[category]['Yt']
  W = settings.DATASET[category]['A']
  order = request.session['ts_order']

  ysl = request.session['ysl']

  print("ORDER",order)

  # Convert this to matrix format.
  ## get labeled matrix fl (learner provide labels)
  X = np.zeros((Yt.shape[0],2))
  for _i in range(len(order)):
    _lbl = ysl[_i]
    _ord = order[_i]
    if _lbl == 1:
      X[_ord, 0] = 1
    else:
      X[_ord, 1] = 1

  X = np.array(X)


  Y = []
  for _lbl in Yt:
    if _lbl == 1:
      Y.append([1, 0])
    else:
      Y.append([0, 1])
  Y = np.array(Y)
  print("Y",Y.shape)



  L = request.session['ts_order']


  # Get the total number of samples (nS) and total number of classes (nC). nC is not actually used.
  [nS, nC] = Y.shape

  # Create the set of unlabelled samples (U)
  U = np.setdiff1d(np.arange(nS), L)

  # Get the number of unlabelled samples (nU)
  nU = len(U)

  # Get the ground truth for the unlabelled samples
  Yu = Y.take(U, 0)

  # Get the unlabelled section of the covariance matrix
  Delta = np.subtract(np.diag(np.sum(W, 1)), W)
  invDeltaU = np.linalg.inv(Delta.take(U, 0).take(U, 1))

  # Get the current state of the GRF, for the unlabelled samples
  f = np.dot(invDeltaU, np.dot(W.take(U, 0).take(L, 1), X.take(L, 0)))

  # Create a list of risks, one for each unlabelled sample
  uRisks = np.zeros(nU)

  # Try each unlabelled sample
  for u in range(nU):
    # Find the sample number (remember that U is just the list of unlabelled samples, not all the samples)
    s = U[u]

    # If the sample is a testing image, then ignore it (by assigning a very high risk)
    # if s in testing_samples:
    #   uRisks[u] = 10000
    #   continue

    # Calculate the new state of the GRF if this sample were to be revealed to the user (here, we assume that the user's belief of this sample will then be the ground truth -- debatable...)
    GG = invDeltaU[:, u] / invDeltaU[u, u]
    diff = Y[s, :] - f[u, :]
    fPlus = f + np.dot(GG[..., np.newaxis], diff[np.newaxis, ...])

    # Sum up the risks over all unlabelled points (i.e. the difference between the new state, and the ground truth)
    D = np.abs(1 - fPlus[Yu == 1])
    uRisks[u] = np.sum(D)

  # Get the sample which minimised the risk
  next_sample_index = np.argmin(uRisks)
  # next_sample = U[next_sample_index]

  # Return this sample
  return next_sample_index




def get_next(request):

  algorithm = request.session['algorithm']
  current_teaching_image = request.session['c_teaching']
  category = request.session['category']
  train_images = settings.DATASET[category]['Names_t']
  test_images = settings.DATASET[category]['Names_e']

  if request.session['mode'] == 'test':
    img_idx = int(np.random.random_integers(0, len(test_images)-1, 1)[0])
    while img_idx in request.session['ev_order']:
      img_idx = int(np.random.random_integers(0, len(test_images)-1, 1)[0])

    img_name = test_images[img_idx][0][0]

  else:
    if current_teaching_image <= 5:
      algorithm = 'rt'

    img_idx = 0

    print('Returning an id using %s' % algorithm)


    if algorithm == 'eer':
      img_idx = get_eer_next(request)

    elif algorithm == 'imt':
      beta = 0.0
      img_idx, img_prob = get_jedi_next(request, beta)
      request.session['ysl_prob'] = request.session['ysl_prob'] + [img_prob.tolist()]

    elif algorithm == 'jedi':
      print('==> Fetching through JEDI.')
      beta = request.session['beta']
      img_idx, img_prob = get_jedi_next(request, beta)
      request.session['ysl_prob'] = request.session['ysl_prob'] + [img_prob.tolist()]

      # Pick the next image.

    else:
      img_idx = int(np.random.random_integers(0, len(train_images)-1, 1)[0])
      request.session['ysl_prob'] = request.session['ysl_prob'] + [[0.5, 0.5]]

    request.session['ts_order'] = request.session['ts_order'] + [int(img_idx)]
    img_name = train_images[img_idx][0][0]

  return img_idx, img_name

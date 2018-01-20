import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import scipy.io as spio
import joblib

def gaussianData(mu1, mu2, sigma, numData):
  # generate gaussian binary data.
  r1 = np.random.multivariate_normal(mu1[:, 0], sigma, numData);
  r2 = np.random.multivariate_normal(mu2[:, 0], sigma, numData);
  D = np.vstack((r1, r2))
  D = np.hstack((D, np.ones((numData * 2, 1))))
  Y = np.vstack((np.ones((numData, 1)), -1 * np.ones((numData, 1))))
  return D, Y


class LearnerClass(object):
  def __init__(self, beta, w0):
    self.Ws = w0
    self.Xs = []
    self.Ys = []
    self.Ysl = []
    self.Ysl_prob = []
    self.beta = beta
    self.order = []


def padWuseLabeledData(A, order):
  n = A.shape[0]
  nl = len(order);
  ANEW = A.copy()
  ANEW = np.hstack((ANEW, np.zeros((n, nl))))
  ANEW = np.vstack((ANEW, np.zeros((nl, n + nl))))

  ANEW[n:, :] = ANEW[order, :]
  ANEW[:, n:] = ANEW[:, order]
  ANEW[n:n + nl, n:n + nl] = ANEW[order, order]

  return ANEW

def harmonic_function(W, fl):
  l = fl.shape[0]
  n = W.shape[0]

  L = np.diag(np.sum(W,1)) - W;
  fu = -1* np.dot(np.linalg.inv(L[l:n, l:n]), np.dot(L[l:n, :l], fl))

  return fu

def JEDI_harmonic(learner, Y, A):
  numData = len(Y);
  numClass = len(np.unique(Y));
  order = learner.order;
  Ysl = learner.Ysl;

  # TODO - Unique last.
  u, ia = np.unique(order, return_index=True);
  uniqueIdx = np.sort(ia);

  order_unique = [order[ind] for ind in uniqueIdx]
  Ysl_unique = [Ysl[ind] for ind in uniqueIdx]

  #  pad W with labeled nodes, get Wnew
  Anew = padWuseLabeledData(A, order_unique);
  Ynew = np.vstack((Y, Y[order_unique]))
  numDataNew = len(Ynew);

  ## get labeled matrix fl (learner provide labels)
  fl = []
  for _lbl in Ysl_unique:
    if _lbl == 1:
      fl.append([1, 0])
    else:
      fl.append([0, 1])

  fl = np.array(fl)

  idx = list(range(Anew.shape[0]))
  remaining_idx = list(set(idx) - set(order_unique))
  reorder_idx = order_unique + list(remaining_idx)
  Anew = Anew[reorder_idx,:]
  Anew = Anew[:,reorder_idx]

  flu = harmonic_function(Anew, fl)

  Yu = Ynew[len(order_unique):]
  Predu = np.argmax(flu,axis=1)

  # This is not clear
  Yu_predu = (((-1*Yu)+1)/2)

  # TODO: Accuracy/
  # accu = sum(Yu_predu == Predu)/numData;
  # print(accu)

  # output the probability w.r.t.the index of  Dt
  prob = np.zeros((numData, numClass));
  nFlu = flu.shape[0]
  prob[order_unique,:] = flu[nFlu - len(order_unique):,:];
  restIdx = list(set(range(numData)) - set(order_unique))
  prob[restIdx,:] = flu[:nFlu - len(order_unique),:]

  return prob

def JEDI_blackbox(D, Y, learner, wo, step, A):
  Ws = learner.Ws;
  Xs = learner.Xs;
  Ys = learner.Ys;
  Ysl = learner.Ysl;
  Ysl_prob = learner.Ysl_prob;
  beta = learner.beta;
  _order = learner.order;

  numData = len(Y);
  fvalue = np.zeros((numData, 1));
  eta = step;

  d = Xs[0].shape[0]
  tminus = len(learner.Ys);

  Prob = JEDI_harmonic(learner, Y, A);

  Ps = np.zeros((len(_order), 1))
  for _i in range(len(_order)):
    ys = Ys[_i];
    if ys == 1:
      Ps[_i] = Ysl_prob[_i][1];
    else: # ys == -1
      Ps[_i] = Ysl_prob[_i][0]

  coeff = np.reshape(np.power(beta,np.arange(tminus,0,-1)),(tminus,1))

  Ys_arr = np.reshape(np.array(Ys),(len(Ys),1))
  Ys_Ps = np.multiply((-1 * Ys_arr),Ps)

  Xs_arr = np.array(Xs)
  Ys_Ps_Xs = np.multiply(np.matlib.repmat(Ys_Ps, 1, d),Xs_arr)

  Bs = np.matlib.repmat(coeff, 1, d)

  Ys_Ps_Xs_Bs = np.multiply(Ys_Ps_Xs,Bs)
  FS = np.sum(Ys_Ps_Xs_Bs,axis=0)

  w = Ws[:,-1]
  for id in range(numData):
    x = D[id,:]
    y = Y[id,:]

    if y == 1:
      pt = Prob[id,1]
      pnt = 1/Prob[id,0]
    else:
      pt = Prob[id,0]
      pnt = 1/Prob[id,1]

    epsilon_0 = y * (wo.transpose().dot(x))

    fvalue[id, 0] = np.power(eta,2) * np.power(pt,2) * np.power(np.linalg.norm(x),2) \
                    + 2*np.power(eta,2) * FS.dot(-1 * y * x * pt) - 2 * eta * (np.log(pnt) - np.log(1+np.exp(-1*epsilon_0)))

  selectIdx = np.argmin(fvalue)
  selectProb = Prob[selectIdx,:]
  return selectIdx, selectProb

if __name__ == '__main__':

  # # LOAD(generate) THE DATA
  # d = 10;
  # mu1 = -.6 * np.ones((10, 1));
  # mu2 = .6 * np.ones((10, 1));
  # _A = np.round(np.random.rand(d, 1) * 10)
  # sigma1 = np.diag(_A[:, 0])
  # numData = 1000;
  #
  # accu_LR = 0;
  # count = 0;
  # while accu_LR < 0.80 or accu_LR >= 0.95:  # quality control of the generated data
  #   if not count == 0:
  #     print('Random Guassian Data Generation # %d, accu_LR = %.2f...' % (count, accu_LR));
  #
  #   D, Y = gaussianData(mu1, mu2, sigma1, numData)
  #
  #   # split into teaching set and evaluation set
  #   ratio = 0.2;
  #   randidx = list(range(numData * 2))
  #   np.random.shuffle(randidx)
  #   randidx = np.array(randidx)
  #   numTraining = int(ratio * numData * 2)
  #   tidx = randidx[0:numTraining]
  #   eidx = randidx[numTraining:]
  #
  #   X_TR = D[tidx, :]
  #   X_TE = D[eidx, :]
  #
  #   Y_TR = Y[tidx, :]
  #   Y_TE = Y[eidx, :]
  #
  #   clf = LogisticRegression()
  #   clf.fit(X_TR, Y_TR)
  #   Y_PRED = clf.predict(X_TE)
  #   accu_LR = accuracy_score(Y_TE, Y_PRED)
  #   print(accu_LR)
  #
  #   wo_LR = clf.coef_
  #   count += 1;
  #
  # # teaching set
  # Dt = D[tidx, :]
  # Yt = Y[tidx]
  #
  # # evaluation set
  # De = D[eidx, :]
  # Ye = Y[eidx]
  #
  # # TODO: Figure out a way to compute A and plot it.
  #
  #
  # A = np.identity(len(tidx))

  # Read from the matlab file.
  data = spio.loadmat('/home/arun/code/github/JEDI_KDD18/MATLAB/matlab.mat')
  d = 10
  Yt = data['Yt']
  Dt = data['Dt']
  De = data['De']
  wo_LR = data['wo_LR']
  A = data['A']


  # GENERATE LEARNERS and JEDI TEACHING
  maxIter = 600;
  step_init = 0.05;

  Beta = [0.0, 0.5, 0.75, 0.875, 0.99];
  numMemory = [1, 2, 4, 8, np.Inf];
  numLearner = len(Beta);

  # teacher assets
  fvalue_JEDI = np.zeros((maxIter, numLearner));
  teachingSetJEDI = np.zeros((maxIter, numLearner));

  accu_JEDI_train= np.zeros((maxIter, numLearner));

  # w0 = np.multiply((-1 + np.random.rand(d + 1, 1) * 2), np.ones((d + 1, 1)))
  w0 =  spio.loadmat('/home/arun/code/github/JEDI_KDD18/MATLAB/w0.mat')['w0']
  print(w0)


  selectIdxFirst = 389 #np.random.permutation(len(Yt))[0]
  learner = {}

  for il in range(numLearner):
    print("Learner %d" % il)
    beta = Beta[il]
    learner[il] = LearnerClass(beta, w0)
    w = w0

    for it in range(maxIter):
      if np.mod(it, 100) == 0:
        print('JEDI for worker # %d of iteration %d...' % (il, it));

      step = step_init * 20 / (20 + 1 + it); # it starts from zero

      if it >= 1:
        selectIdx, selectProb = JEDI_blackbox(Dt, Yt, learner[il], wo_LR, step, A);
      else:  # first teaching example
        selectIdx = selectIdxFirst;
        selectProb = np.array([0.5, 0.5])

      teachingSetJEDI[it, il] = selectIdx;

      # learner make prediction on (xt)
      x = Dt[selectIdx, :];

      ysl = np.sign(w.transpose().dot(x))[0];

      # learner learns (real learner, these calculations are assumed to be done within their mind...)
      y = Yt[selectIdx][0]
      epsilon = y * (w.transpose().dot(x));
      epsilon = epsilon[0]

      dw = np.array(1 / (1 + np.exp(epsilon)) * (-1 * y * x))
      _dw = np.reshape(dw, (dw.shape[0], 1))
      w = w - step * _dw;

      # update the learner assets
      learner[il].Ysl_prob.append(selectProb)
      learner[il].Ysl.append(ysl)
      learner[il].Xs.append(x)
      learner[il].Ys.append(y)
      learner[il].Ws = np.hstack((learner[il].Ws, w))
      learner[il].order.append(selectIdx)

      # function objective
      fvalue_JEDI[it, il] = np.sum(np.log(1 + np.exp(-1 * np.multiply(np.dot(Dt, w), Yt))));

      # pred_JEDI_train = Dt.dot(w)
      # # pred_JEDI_train = (pred_JEDI_train >= 0) == ((Yt + 1) / 2);
      # accu_JEDI_train[il, 1] = np.sum(pred_JEDI_train) / len(Yt)
      # #
      # pred_JEDI_eval = De.dot(w);
      # # pred_JEDI_eval = (pred_JEDI_eval >= 0) == ((Ye + 1) / 2);
      # # accu_JEDI_eval(il, 1) = sum(pred_JEDI_eval) / length(Ye);
    print('Teaching of learner #%d (beta = %0.3f) is done...'%(il, learner[il].beta));


  print("Saving the file for analysis..")
  joblib.dump(fvalue_JEDI,'/tmp/fvalue_JEDI.dat',compress=3)
import numpy as np
import scipy as sp
import sys
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score
import joblib

def  gaussianData(mu1, mu2, sigma, numData):
    # generate gaussian binary data.
    r1 = np.random.multivariate_normal(mu1[:,0], sigma, numData);
    r2 = np.random.multivariate_normal(mu2[:,0], sigma, numData);
    D = np.vstack((r1,r2))
    # D = np.hstack((D,np.ones((numData * 2, 1))))
    Y = np.vstack((np.ones((numData, 1)), -1 * np.ones((numData, 1))))
    return D,Y



class LearnerClass(object):

    def __init__(self, beta, w0):
        self.WS = w0
        self.XS = None
        self.YS = None
        self.Ysl = None
        self.Ysl_prob = None
        self.beta = beta
        self.order = []

def JEDI_harmonic(learner, Y, A):
    numData = len(Y);
    numClass = len(np.unique(Y));
    order = learner.order;
    Ysl = learner.Ysl;

    #TODO - Unique last.
    ia = np.unique(order);
    uniqueIdx = np.sort(ia);
    order_unique = order[uniqueIdx]
    Ysl_unique = Ysl[uniqueIdx]

    #  pad W with labeled nodes, get Wnew
    Anew = padWuseLabeledData(A, order_unique);
    Ynew = np.vstack((Y,Y(order_unique)))
    numDataNew = len(Ynew);

    ## get labeled matrix fl (learner provide labels)
    fl = labelvec2matrixJEDI(Ysl_unique, numClass);





def JEDI_blackbox(D, Y, learner, wo, step, A):
    Ws = learner.Ws;
    Xs = learner.Xs;
    Ys = learner.Ys;
    Ysl = learner.Ysl;
    Ysl_prob = learner.Ysl_prob;
    beta = learner.beta;
    order = learner.order;

    numData = len(Y);
    fvalue = np.zeros((numData, 1));
    eta = step;
    d = Xs.shape[1]
    tminus = len(learner.Ys);

    Prob = JEDI_harmonic(learner, Y, A);

if __name__ == '__main__':

    # LOAD(generate) THE DATA
    d = 10;
    mu1 = -.6 * np.ones((10, 1));
    mu2 = .6 * np.ones((10, 1));
    _A = np.round(np.random.rand(d, 1) * 10)
    sigma1 = np.diag(_A[:, 0])
    numData = 1000;

    accu_LR = 0;
    count = 0;
    while accu_LR < 0.80 or accu_LR >= 0.95: # quality control of the generated data
        if not count == 0:
            print('Random Guassian Data Generation # %d, accu_LR = %.2f...'%(count, accu_LR));

        D, Y = gaussianData(mu1, mu2, sigma1, numData)


        # split into teaching set and evaluation set
        ratio = 0.2;
        randidx = list(range(numData*2))
        np.random.shuffle(randidx)
        randidx = np.array(randidx)
        numTraining = int(ratio*numData*2)
        tidx = randidx[0:numTraining]
        eidx = randidx[numTraining:]

        X_TR = D[tidx,:]
        X_TE = D[eidx,:]

        Y_TR = Y[tidx,:]
        Y_TE = Y[eidx,:]

        clf = LogisticRegression()
        clf.fit(X_TR,Y_TR)
        Y_PRED = clf.predict(X_TE)
        accu_LR = accuracy_score(Y_TE,Y_PRED)
        print(accu_LR)

        wo_LR = clf.coef_
        count += 1;


    # teaching set
    Dt = D[tidx,:]
    Yt = Y[tidx]

    # evaluation set
    De = D[eidx,:]
    Ye = Y[eidx]


    #TODO: Figure out a way to compute A and plot it.


    A = np.identity(len(tidx))



    # GENERATE LEARNERS and JEDI TEACHING
    maxIter = 600;
    step_init = 0.05;

    Beta = [0.0, 0.5, 0.75, 0.875, 0.99];
    numMemory = [1, 2, 4, 8, np.Inf];
    numLearner = len(Beta);

    #teacher assets
    fvalue_JEDI = np.zeros((maxIter, numLearner));
    teachingSetJEDI = np.zeros((maxIter, numLearner));

    w0 = np.multiply((-1 + np.random.rand(d + 1, 1) * 2), np.ones((d + 1, 1)))
    selectIdxFirst = np.random.permutation(len(Yt))[0]
    learner = {}

    for il in range(numLearner):
        print("Learner %d"%il)
        beta = Beta[il]
        learner[il] = LearnerClass(beta, w0)
        w = w0

        for it in range(maxIter):
            if np.mod(it, 100) == 0:
                print('JEDI for worker # %d of iteration %d...'%(il, it));

            step = step_init * 20 / (20 + it);

            if it > 1:
                [selectIdx, selectProb] = JEDI_blackbox(Dt, Yt, learner(il), wo_LR, step, A);
            else: # first teaching example
                selectIdx = selectIdxFirst;
                selectProb = [0.5, 0.5];


            teachingSetJEDI[it,il] = selectIdx;

            # # learner make prediction on (xt)
            # x = Dt[selectIdx,:];
            # ysl = np.sign(w.transpose * x);
            #
            # # learner learns (real learner, these calculations are assumed to be done within their mind...)
            # y = Yt(selectIdx);
            # epsilon = y * (w.transpose() * x);
            # dw = 1 / (1 + np.exp(epsilon)) * (-1 * y * x');
            # w = w - step * dw;

            # # update the learner assets
            # learner(il).Ysl_prob = [learner(il).Ysl_prob;
            # selectProb];
            # learner(il).Ysl = [learner(il).Ysl;
            # ysl];
            # learner(il).Xs = [learner(il).Xs;
            # x];
            # learner(il).Ys = [learner(il).Ys;
            # y];
            # learner(il).Ws = [learner(il).Ws w];
            # learner(il).order = [learner(il).order;
            # selectIdx];
            #
            # # function objective
            # fvalue_JEDI(it, il) = sum(log(1 + exp(-1 * Dt * w. * Yt)));

        # pred_JEDI_train = Dt * w;
        # pred_JEDI_train = (pred_JEDI_train >= 0) == ((Yt + 1) / 2);
        # accu_JEDI_train(il, 1) = sum(pred_JEDI_train) / length(Yt);
        #
        # pred_JEDI_eval = De * w;
        # pred_JEDI_eval = (pred_JEDI_eval >= 0) == ((Ye + 1) / 2);
        # accu_JEDI_eval(il, 1) = sum(pred_JEDI_eval) / length(Ye);
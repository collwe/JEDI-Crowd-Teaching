import joblib
import matplotlib.pyplot as plt

fvalue_JEDI = joblib.load('/tmp/fvalue_JEDI.dat')
print(fvalue_JEDI.shape)

maxIter, numLearner = fvalue_JEDI.shape
x = list(range(maxIter))
for i in range(numLearner):
  plt.plot(x,fvalue_JEDI[:,i],label='Learner %d'%i)


plt.legend()
plt.show()
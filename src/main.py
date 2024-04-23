import matplotlib.pyplot as plt
import numpy as np 

mean = 1-0.05
t = np.linspace(0.0001,mean,1000)
P_h = np.exp(-2*t**2)
P_m = mean/t


plt.figure()
plt.plot(mean - t, 1- P_h, label='hoeffding')
plt.plot(mean - t, 1- P_m, label='markov')
plt.xlabel('Probability level')
plt.ylabel('Confidence')
plt.legend()
plt.ylim(-2,1.5)
plt.savefig('bounds.png')
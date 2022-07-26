from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
import math

x = []
y= []
y_true = []
y_esti = []


LM_lambda = []
Radius = []
error = []
g = []


sigma_y = []

# Dog leg输出
a = 6.00206
b = 19.151
c = 4.99278

# 真值
a_true = 6
b_true = 20
c_true = 5


for line in open('CurveFitting_data.txt','r'):
    if line == "origin\n":
        continue
    else:
        value = [float(s) for s in line.split()]
        x.append(value[0])
        y.append(value[1])
        sigma_y.append(value[2])
        y_esti.append(a*math.exp(-value[0]/b) + 6 * math.sin(value[0]/c))
    

y_up = np.array(y_esti) + 3*np.array(sigma_y)

plt.figure(1)
plt.scatter(x,y,s=3.,color = (0,1,0))

plt.scatter(x,y_up,s=1.,color = (0,0,0))
plt.plot(x,y_esti)



plt.title("y=a*exp(-x/b) + 6* sin(x/c)") 
plt.legend(['data','esti','true'],loc = "upper right")

# LM
iter = []
i = 0
for line in open('LM.txt','r'):
    value = [float(s) for s in line.split()]
    iter.append(i)
    i = i+1
    error.append(value[0])
    g.append(value[1])
    LM_lambda.append(value[2])



plt.figure(2)
plt.plot(iter,error,label = "error")
plt.plot(iter,g,label = "g")
plt.plot(iter,LM_lambda,label = "lambda")
plt.yscale("log")
plt.legend()


# DogLeg
i = 0
iter.clear()
error.clear()
g.clear()
for line in open('DogLeg.txt','r'):
    value = [float(s) for s in line.split()]
    iter.append(i)
    i = i+1
    error.append(value[0])
    g.append(value[1])
    Radius.append(value[2])



plt.figure(3)
plt.plot(iter,error,label = "error")
plt.plot(iter,g,label = "g")
plt.plot(iter,Radius,label = "Radius")
plt.yscale("log")
plt.legend()

plt.show()

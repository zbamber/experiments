import csv
import numpy as np

def compileData(X, Y):
    if len(X) == len(Y):
        x = - X
        y = np.log(Y)
        uncertainties = [0.005] * len(x)
        return x, y, uncertainties
    else:
        return [], []
t = np.array([x for x in range(5)])
V = np.array([5.078,0.777,0.102,0.009,0.001])

x, y, uncertainties = compileData(t,V)

data = zip(x, y, uncertainties)
with open('experiment5.csv','w',newline='') as file:
    writer = csv.writer(file)
    for i in range(len(y)):
        writer.writerow([x[i], y[i], uncertainties[i]])
import csv
import numpy as np

def compileData(I,V,gasFlowRate):
    if len(I) == len(V) == len(gasFlowRate):
        x = I / 1000 * V # power in watts
        y = gasFlowRate / (1000 * 60) # convert litres per minute to m^3/s
        uncertainties = [0.05 / (1000 * 60)] * len(gasFlowRate) # +- 4 SLPM converted to m^3/s
        return x, y, uncertainties
    else:
        return [], []
        

# I_1 = np.array([0,99,497,750,975,1252,1390]) # current in mA
# V_1 = np.array([0,0.67,3.36,5.07,6.60,8.47,9.40]) # potential difference in volts
# gasFlowRate_1 = np.array([1.95,1.90,2.25,2.96,3.66,4.62,5.27]) # in litres per minute

# last removed
I_1 = np.array([0,99,497,750,975,1252]) # current in mA
V_1 = np.array([0,0.67,3.36,5.07,6.60,8.47]) # potential difference in volts
gasFlowRate_1 = np.array([1.95,1.90,2.25,2.96,3.66,4.62]) # in litres per minute

I_2 = np.array([0,501,729,896,1126,1193,1310,1421]) # current in mA
V_2 = np.array([0,3.39,4.93,6.06,7.62,8.07,8.86,9.62]) # potential difference in volts
gasFlowRate_2 = np.array([1.26,1.78,2.37,2.99,3.93,4.15,4.69,5.24]) # in litres per minute

x, y, uncertainties = compileData(I_1,V_1,gasFlowRate_1)

data = zip(x, y, uncertainties)
with open('latentSetRemoved.csv','w',newline='') as file:
    writer = csv.writer(file)
    for i in range(len(y)):
        writer.writerow([x[i], y[i], uncertainties[i]])
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv (r'C:\Users\andre\Documents\GitHub\MSE-314-HW-03\Q2.csv')

strength = data.sort_values(by=['Strength'])

i = 1
n = len(strength)
P_f = [0 for i in range(n)] 

for i in range(n + 1):
    P_f[i-1] = (i - 0.5) / n
    
P_f = np.array(P_f)
    
X = np.log(strength)
Y = np.log(np.log(1 / (1 - P_f)))

lr = LinearRegression().fit(X, Y)

m = lr.coef_
b = lr.intercept_

print(m)
print(np.exp(b / -m))

x_vals = np.linspace(5.4,5.95, 2)
y_vals = m * x_vals + b

fig = plt.figure(dpi=300)
plt.scatter(X, Y, c='k')
plt.plot(x_vals, y_vals)
    
plt.title('Weibull Distribution')
plt.xlabel(r'$\ln$[Strength (MPa)]')
plt.ylabel(r'$\ln \ \ln \left[ 1 / (1 - P_f) \right]$')

fig.savefig('plot.png')
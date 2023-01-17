import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import math
#import uproot3 as uproot
from scipy.signal import find_peaks
import numpy.ma as ma
import scipy.stats as ss
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings

def func(x, a,b):
        return a*np.exp(-b*x) 

def getData(data,bins):
    bins = np.linspace(min(data), max(data), bins)
    hist, bins = np.histogram(data, bins = bins, density = True)
    bin_centers = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range (len(bins)-1)])
    return bins, bin_centers, hist


data = np.array([14, 20, 17, 17, 12, 25, 10, 22, 32, 44, 22, 29, 29, 65, 18,15, 11, 25, 89, 31, 70, 9, 24, 16, 36,
              17, 18, 46, 16, 25, 24, 17, 12, 21, 17, 30, 26, 17, 11, 13, 13, 28, 45, 72, 9, 50, 40, 36, 
              14, 25, 29, 47, 29, 13, 18, 14, 7,32, 11, 18, 9, 20, 11, 24, 15, 9, 37, 10, 29, 47,42, 13, 6, 22, 24, 29, 
              14, 16, 17, 79, 29, 31, 16, 9, 20, 11,38, 28, 49, 22, 14, 66, 25, 26, 14, 23, 13, 38, 46, 17, 10, 15, 11, 29,
              31, 19, 18, 17, 10, 30, 42, 24, 41, 10, 12, 16, 24, 24, 12, 14, 64, 42]) 


m = (data > 12) & (data < 60)
data2 = data
data = data[m]

ldata = len(data)
ldata2 = len(data2)

''' scipy.curve_fit '''
bins,bin_centers,y = getData(data,10)
popt,pov = curve_fit(func,bin_centers,y)
tau0 = round(1/popt[1],3)
plt.hist(data,bins = bins,histtype='step',fill=1,density = True, color='slategrey')
# plt.hist(data2,bins = np.linspace(min(data2), max(data2), 14),histtype='step',fill=0,density = True, color='r')
plt.plot(bins, func(bins, *popt),c = 'purple',label = 'curve_fit: $\\tau$=%s ns'%tau0)

''' scipy.expon.fit '''
P = ss.expon.fit(data)
rP = ss.expon.pdf(bins, *P)
# plt.hist(data,density = True, color='slategrey')
tau1 = round(P[1],3)
plt.plot(bins, rP, color='darkturquoise', label = 'expon.fit: $\\tau$=%s ns'%tau1)


plt.xlabel('Binned $\Delta$t Data' )
plt.ylabel('Normalized Density')
plt.title('%s Counts'%ldata)
plt.legend()

plt.show()

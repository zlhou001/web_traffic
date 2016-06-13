import os
from utils import DATA_DIR, CHART_DIR
import scipy as sp
import matplotlib.pyplot as plt


data = sp.genfromtxt(os.path.join(DATA_DIR, "web_traffic.tsv"), delimiter="\t")
print(data[:10])
print(data.shape)


x = data[:,0]
y = data[:,1]
print(sp.sum(sp.isnan(y)))

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

def error(f, x, y):
    return sp.sum((f(x)-y)**2)

def drawsourcedata(x, y, fname):
    plt.scatter(x,y)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks([w*7*24 for w in range(10)], ['week %i' %w for w in range(10)])
    plt.autoscale(tight=True)
    plt.grid()
    #plt.show()
    plt.savefig(fname)


    
"""Draw source data image to get first impression: sourcedata.jpg"""
drawsourcedata(x, y, os.path.join(CHART_DIR, "source data.jpg"))



"""First Model: Simple straight line"""
fp1, res1, rank1, sv1, rcond1 = sp.polyfit(x, y, 1,full = True)
print('Simple straight line, Model parameter:%s' %fp1)
f1 = sp.poly1d(fp1)
print("Error of simple straight line model:%f" %error(f1, x, y))
fx = sp.linspace(0, x[-1], 1000)
plt.plot(fx, f1(fx), linewidth=5)
plt.legend(["d=%i" %f1.order], loc="upper left")
drawsourcedata(x, y, os.path.join(CHART_DIR, "Simple straight line.jpg"))

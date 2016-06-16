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
    plt.ylim(ymin=0)
    plt.ylim(ymax=10000)
    plt.grid()
    #plt.show()
    plt.savefig(fname)


    
"""Draw source data image to get first impression: sourcedata.jpg"""
drawsourcedata(x, y, os.path.join(CHART_DIR, "source data.jpg"))
fx = sp.linspace(0, x[-1], 1000)
fx1 = sp.linspace(0, x[-1]+100, 1000)


"""First Model: Simple straight line"""
fp1, res1, rank1, sv1, rcond1 = sp.polyfit(x, y, 1,full = True)
print('Simple straight line, Model parameter:%s' %fp1)
f1 = sp.poly1d(fp1)
print("Error of simple straight line model:%f" %error(f1, x, y))
plt.plot(fx, f1(fx), linewidth=4, linestyle='-', color='g')
plt.legend(["d=%i" %f1.order], loc="upper left")
drawsourcedata(x, y, os.path.join(CHART_DIR, "Simple straight line.jpg"))


"""Second Model: 2-degree polynomial"""
f2p = sp.polyfit(x, y, 2)
#print(f2p)
f2 = sp.poly1d(f2p)
print("Error of 2-degree polunomial model: %f" %error(f2, x, y))
plt.plot(fx, f2(fx), linewidth=4, linestyle='-.', color='r')
plt.legend(["d=%i" %f1.order, "d=%i" %f2.order], loc="upper left")
drawsourcedata(x, y, os.path.join(CHART_DIR, "2-degree polynomial.jpg"))


"""Third Model: 3-degree polynomial"""
f3p = sp.polyfit(x, y, 3)
#print(f3p)
f3 = sp.poly1d(f3p)
print("Error of 3-degree polunomial model: %f" %error(f3, x, y))
plt.plot(fx, f3(fx), linewidth=4, linestyle='--', color='m')
plt.legend(["d=%i" %f1.order, "d=%i" %f2.order, "d=%i" %f3.order], loc="upper left")
drawsourcedata(x, y, os.path.join(CHART_DIR, "3-degree polynomial.jpg"))


"""Fourth Model: 10-degree polynomial"""
f10p = sp.polyfit(x, y, 10)
#print(f10p)
f10 = sp.poly1d(f10p)
print("Error of 10-degree polunomial model: %f" %error(f10, x, y))
plt.plot(fx, f10(fx), linewidth=4, linestyle=':', color='y')
plt.legend(["d=%i" %f1.order, "d=%i" %f2.order, "d=%i" %f3.order, "d=%i" %f10.order], loc="upper left")
drawsourcedata(x, y, os.path.join(CHART_DIR, "10-degree polynomial.jpg"))


"""Fifth Model: 50-degree polynomial"""
f50p = sp.polyfit(x, y, 50)
#print(f100p)
f50 = sp.poly1d(f50p)
print("Error of 50-degree polunomial model: %f" %error(f50, x, y))
plt.plot(fx, f50(fx), linewidth=4, linestyle='-', color='r')
plt.legend(["d=%i" %f1.order, "d=%i" %f2.order, "d=%i" %f3.order, "d=%i" %f10.order, "d=%i" %f50.order], loc="upper left")
drawsourcedata(x, y, os.path.join(CHART_DIR, "50-degree polynomial.jpg"))


"""Prediction"""
plt.figure(num=None, figsize=(8, 6))
plt.clf()
#print(f100p)
plt.plot(fx1, f1(fx1), linewidth=4, linestyle='-', color='g')
plt.plot(fx1, f2(fx1), linewidth=4, linestyle='-.', color='r')
plt.plot(fx1, f3(fx1), linewidth=4, linestyle='--', color='m')
plt.plot(fx1, f10(fx1), linewidth=4, linestyle=':', color='y')
plt.plot(fx1, f50(fx1), linewidth=4, linestyle='-', color='r')
#plt.yticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
#plt.yscale('linear')
plt.autoscale(tight=True)
plt.ylim(ymin=0)
plt.ylim(ymax=10000)
plt.legend(["d=%i" %f1.order, "d=%i" %f2.order, "d=%i" %f3.order, "d=%i" %f10.order, "d=%i" %f50.order], loc="upper left")
drawsourcedata(x, y, os.path.join(CHART_DIR, "Prediction.jpg"))



"""Two line Model"""
inflection = 3.5*7*24
xa = x[:inflection]
xb = x[inflection:]
ya = y[:inflection]
yb = y[inflection:]
fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))
fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)
print("Error of Two line Model: %f" %(fa_error + fb_error))
plt.figure(num=None, figsize=(8, 6))
plt.clf()
plt.plot(fx, fa(fx), linewidth=4, linestyle='-', color='r')
plt.plot(x[inflection:], fb(x[inflection:]), linewidth=4, linestyle='-', color='r')
plt.legend(["d=%i" %fa.order, "d=%i" %fb.order], loc="upper left")
drawsourcedata(x, y, os.path.join(CHART_DIR, "Two line Model.jpg"))


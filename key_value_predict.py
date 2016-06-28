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


"""Which Model is the best ?"""
key = 0.7
edge = int(key * len(x))
x1 = x[:edge]
y1 = y[:edge]
fp1, res1, rank1, sv1, rcond1 = sp.polyfit(x1, y1, 1,full = True)
f1 = sp.poly1d(fp1)
print("Error of simple straight line model:%f, when data set used in prediction is %f of total data." %(error(f1, x[edge:len(x)-1], y[edge:len(y)-1]), key))
f2p = sp.polyfit(x1, y1, 2)
f2 = sp.poly1d(f2p)
print("Error of 2-degree polunomial model:%f, when data set used in prediction is %f of total data." %(error(f2, x[edge:len(x)-1], y[edge:len(y)-1]), key))
f3p = sp.polyfit(x1, y1, 3)
f3 = sp.poly1d(f3p)
print("Error of 3-degree polunomial model:%f, when data set used in prediction is %f of total data." %(error(f3, x[edge:len(x)-1], y[edge:len(y)-1]), key))
f10p = sp.polyfit(x1, y1, 10)
f10 = sp.poly1d(f10p)
print("Error of 10-degree polunomial model:%f, when data set used in prediction is %f of total data." %(error(f10, x[edge:len(x)-1], y[edge:len(y)-1]), key))
f50p = sp.polyfit(x1, y1, 50)
f50 = sp.poly1d(f50p)
print("Error of 50-degree polunomial model:%f, when data set used in prediction is %f of total data." %(error(f50, x[edge:len(x)-1], y[edge:len(y)-1]), key))
plt.figure(num=None, figsize=(8, 6))
plt.clf()
plt.plot(fx1, f1(fx1), linewidth=4, linestyle='-', color='g')
plt.legend(["d=%i" %f1.order], loc="upper left")
plt.plot(fx1, f2(fx1), linewidth=4, linestyle='-.', color='r')
plt.legend(["d=%i" %f1.order, "d=%i" %f2.order], loc="upper left")
plt.plot(fx1, f3(fx1), linewidth=4, linestyle='--', color='m')
plt.legend(["d=%i" %f1.order, "d=%i" %f2.order, "d=%i" %f3.order], loc="upper left")
plt.plot(fx1, f10(fx1), linewidth=4, linestyle=':', color='y')
plt.legend(["d=%i" %f1.order, "d=%i" %f2.order, "d=%i" %f3.order, "d=%i" %f10.order], loc="upper left")
plt.plot(fx1, f50(fx1), linewidth=4, linestyle='-', color='r')
plt.legend(["d=%i" %f1.order, "d=%i" %f2.order, "d=%i" %f3.order, "d=%i" %f10.order, "d=%i" %f50.order], loc="upper left")
drawsourcedata(x, y, os.path.join(CHART_DIR, "Key = %f prediction.jpg" %key))

models = ["simple straight", "2-degree polunomial", "3-degree polunomial", "10-degree polunomial", "50-degree polunomial"]
errors = [error(f1, x[edge:len(x)-1], y[edge:len(y)-1]), error(f2, x[edge:len(x)-1], y[edge:len(y)-1]), error(f3, x[edge:len(x)-1], y[edge:len(y)-1]), error(f10, x[edge:len(x)-1], y[edge:len(y)-1]), error(f50, x[edge:len(x)-1], y[edge:len(y)-1])]
print("Model \"%s\" is the best model when k value is %f" %(models[errors.index(min(errors))], key))
                                                                                    

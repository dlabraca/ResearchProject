'''
Code to generate thresholds for the detector seen in 

Ross, Gordon & Tasoulis, Dimitris & Adams, Niall. (2012). Nonparametric Monitoring of Data Streams for Changes in Location and Scale. Technometrics. 53. 379-389. 10.1198/TECH.2011.10069. 
(thresholds are described here)

Paper recommends 10^6 streams generated. However, this was computational expensive and so the thresholds generated used 10^5.
'''


from scipy.stats import rankdata
import numpy as np


def MW(data):
    n = len(data)
    ranked = rankdata(data)[:-1]
    x = np.arange(1, n)
    helper = (n + 1) / 2
    sd = np.sqrt(x * (n-x) * (n+1) / 12 )
    MW = np.abs((np.cumsum(ranked - helper)) / sd)

    return np.max(MW)

def Mood(data):
    n = len(data)
    ranked = rankdata(data)[:-1]
    x = np.arange(1, n)
    helper = (n + 1) / 2
    mean = x * (n**2 - 1) / 12
    sd = np.sqrt((n - x) * x * (n+1) * (n+2) * (n-2) / 180)
    Mood = np.abs((np.cumsum((ranked - helper) ** 2) - mean) / sd)

    return np.max(Mood)

def Lepage(data):
    n = len(data)
    ranked = rankdata(data)[:-1]
    x = np.arange(1, n)
    helper = (n + 1) / 2
    sd_MW = np.sqrt(x * (n-x) * (n+1) / 12 )
    mean_Mood = x * (n**2 - 1) / 12
    sd_Mood = np.sqrt((n - x) * x * (n+1) * (n+2) * (n-2) / 180)
    MW = np.abs((np.cumsum(ranked - helper)) / sd_MW)
    Mood = np.abs((np.cumsum((ranked - helper) ** 2) - mean_Mood) / sd_Mood)
    
    return np.max(MW**2 + Mood**2)


def write(val):
    with open('', 'w') as f:
            for item in val:
                f.write("%s\n" % str(item))

def threshold_calc(f, alpha):
    # size = (a, b) 
    # a is the number of streams we want to generate
    # b is the number of thresholds we want to compute
    rand = np.random.normal(size = (10**5,2000))
    print('Generated random numbers')
    l = rand.shape[1]+1
    thresholds = []
    # we dont start monitoring till after 20 datapoints have been received so do not want to trigger a cp for these indexes.
    for i in range(19):
        thresholds.append(9999)
    for i in range(20, l):
        data = rand[:,:i]
        maxes = np.zeros(rand.shape[0])
        for j in range(rand.shape[0]):
            maxes[j] = f(data[j])
        thresholds.append(np.quantile(maxes, 1-1/alpha))
        print(i)
        write(thresholds)
        
        rand = rand[np.ix_(maxes <= thresholds[-1])]
        print(thresholds[-1])   
    return thresholds


if __name__ == "__main__":
    # input the statistic we want and the ARL value  
    threshold_calc(Lepage, 1000)





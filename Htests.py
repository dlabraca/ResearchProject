import numpy as np
from scipy.stats import rankdata
from scipy.stats import mood, mannwhitneyu

'''
This code is an implementation of the detector as seen in 
Ross, Gordon & Tasoulis, Dimitris & Adams, Niall. (2012). Nonparametric Monitoring of Data Streams for Changes in Location and Scale. Technometrics. 53. 379-389. 10.1198/TECH.2011.10069. 
'''


def MW(data):
    # returns max normalised Mann-Whitney statistic across all possible paritions and its argument
    # for location shift
    n = len(data)
    ranked = rankdata(data)[:-1]
    x = np.arange(1, n)
    helper = (n + 1) / 2
    sd = np.sqrt(x * (n - x) * (n + 1) / 12 )
    MW = np.abs((np.cumsum(ranked - helper)) / sd)

    return np.argmax(MW), np.max(MW)

def Mood(data):
    # returns max normalised Mood statistic across all possible paritions and its argument
    #  for scale shift
    n = len(data)
    ranked = rankdata(data)[:-1]
    x = np.arange(1, n)
    helper = (n + 1) / 2
    mean = x * (n ** 2 - 1) / 12
    sd = np.sqrt((n - x) * x * (n + 1) * (n + 2) * (n - 2) / 180)
    Mood = np.abs((np.cumsum((ranked - helper) ** 2) - mean) / sd)

    return np.argmax(Mood) , np.max(Mood)

def Lepage(data):
    # returns max Lepage statistic across all possible paritions and its argument
    # for location location and/or scale shift
    n = len(data)
    ranked = rankdata(data)[:-1]
    x = np.arange(1, n)
    helper = (n + 1) / 2
    sd_MW = np.sqrt(x * (n - x) * (n + 1) / 12 )
    mean_Mood = x * (n ** 2 - 1) / 12
    sd_Mood = np.sqrt((n - x) * x * (n + 1) * (n + 2) * (n - 2) / 180)
    MW = np.abs((np.cumsum(ranked - helper)) / sd_MW)
    Mood = np.abs((np.cumsum((ranked - helper) ** 2) - mean_Mood) / sd_Mood)
    Lepages = MW ** 2 + Mood ** 2
    
    return np.argmax(Lepages), np.max(Lepages)

def changedetected(D, threshold):
    # Flags if the threshold limit has been breached, resulting in a changepoint being declared
    if D > threshold:
        return True
    else: 
        False

def stream(data, Test, thresholds):
    # start monitoring at time = 20
    i = 20
    while (i < len(data) + 1):
        # increment window to be tested
        window = data[:i]
        arg, D = Test(window)
        # test at ith threshold value or at the last loaded threshold value
        if i >len(thresholds):
            cp = changedetected(D, thresholds[-1])
        else:
            cp = changedetected(D, thresholds[i-1])
        if cp == True:
            change = ''
            if Test == Lepage:
                # If test is Lepage, test to see what change was detected (either location or shit)
                # If the p value from eg mann whitney test is 100 times smaller than that from the mood test, we declare a location shift
                # If this is not met, then further investigation is needed (tested on synthetic data and the value of 100 seemed appropiate)
                p_val_Mood = mood(window[:arg + 1], window[arg + 1:])[1]
                p_val_MW = mannwhitneyu(window[:arg + 1], window[arg + 1:])[1]
                if 0.01 * p_val_Mood > p_val_MW:
                    change = "Location"
                elif 0.01 * p_val_MW > p_val_Mood:
                    change = "Scale"
                else:
                    change = 'Needs further investigation'
            else: 
                pass

            # return the location of the cp, index for which is was detected, True (cp was declared), what type of change
            return arg + 1, i - 1, True, change
        else:
            # if no change, increase the window size by 1
            i += 1
    # return this if no cp is found and no more datapoints to process
    return None, None, False, None

def online(data, Test, thresholds):
    # This processes the data and works with stream
    # stream is called and montiors for a cp, if one is found, we reinitialise stream and start at the detected changepoitn location

    # we need the data, which test we want to use, the threshold values we want to use, see ThresholdMonte.py for code to estimate appropriate values and timeseries dates
    restart = True
    current_data = data
    cps = [0]
    detection_times = [0]
    change_type = []
    while (len(current_data) >=20) and (restart == True):
        arg, d_time, restart, change = stream(current_data, Test, thresholds)
        if restart == True:
            change_type.append(change)
            last_cp = cps[-1]
            detection_times.append(last_cp + d_time)
            cps.append(last_cp + arg)
            current_data = data[cps[-1]:]
    #returns the CP location in the array

    return cps, detection_times, change_type









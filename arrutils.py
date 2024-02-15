import numpy as np
import pandas as pd
def findmaxvalue(fish_keys, stacked_resp, colorkeys, offsetBefore, see_all=False):
    #find the max in the lsit of responses from the largest spot size.
    #Find the maxvalue from a neuron that has very low variance that I will use to normalize
    maxvalue_fishSchool= {}
    variancelist_fishSchool = {}

    for key in fish_keys.keys():
        data = stacked_resp[key][int([*colorkeys[key]][-1])]
        dict_variance = {}
        dict_avgpeaks = {}
        dict_peakvalues = {}

        for neuron, chunk in data.items():
            chunkAVG = np.mean(chunk,axis=0)
            peak_index = chunkAVG[offsetBefore:].argmax()
            peakvalues = chunk[:,peak_index]

            variance = peakvalues.var()

            dict_variance[neuron] = variance
            dict_avgpeaks[neuron] = np.mean(peakvalues)
            dict_peakvalues[neuron] = peakvalues

        sorted_avgpeaks = {k: v for k, v in sorted(dict_avgpeaks.items(), key=lambda item: item[1], reverse=True)} #sort in descending order
        topvalues_keys = list(sorted_avgpeaks.keys())[0:10] #get the keys of the top five neurons form max average values

        #evaluate the variance of these top 5 peak average values
        varOFtopvals = {k: dict_variance[k] for k in topvalues_keys} #get the variance of the top five values, sort in ascending order and pick the lowest value (first value)
        min_var_key = min(varOFtopvals, key=varOFtopvals.get)

        maxvalue_fishSchool[key] = dict_peakvalues[min_var_key].max()

        variancelist_fishSchool[key] = dict_variance

    if see_all:
        return maxvalue_fishSchool, variancelist_fishSchool

    else:
        return maxvalue_fishSchool

def pretty(x, n=3):
    """
    runs a little smoothing fxn over the array
    :param x: arr
    :param n: width of smooth
    :return: smoothed arr
    """
    return np.convolve(x, np.ones(n) / n, mode="same")


def tolerant_mean(arrs):
    """
    takes an average of arrays of different lengths
    :param arrs: N * arrs
    :return: mean arr, std arr
    """
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[: len(l), idx] = l
    return arr.mean(axis=-1), arr.std(axis=-1)


def norm_fdff(cell_array):
    minVals = np.percentile(cell_array, 10, axis=1)
    zerod_arr = np.array([np.subtract(cell_array[n], i) for n, i in enumerate(minVals)])
    normed_arr = np.array([np.divide(arr, arr.max()) for arr in zerod_arr])
    return normed_arr


def subsection_arrays(input_array, offsets=(-10, 10)):
    a = []
    for repeat in range(len(input_array)):
        s = input_array[repeat] + offsets[0]
        e = input_array[repeat] + offsets[1]
        a.append(np.arange(s, e))
    return np.array(a)


def zdiffcell(arr):
    from scipy.stats import zscore

    diffs = np.diff(arr)
    zscores = zscore(diffs)
    prettyz = pretty(zscores, 3)
    return prettyz


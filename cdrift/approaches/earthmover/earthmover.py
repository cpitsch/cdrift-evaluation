import math
from typing import Dict, Iterable, Tuple, List
from pm4py.objects.log.obj import EventLog
import pm4py.util.xes_constants as xes
import numpy as np
from wasserstein import EMD
from collections import Counter

from cdrift.utils.helpers import makeProgressBar

from sklearn.cluster import kmeans_plusplus

from strsimpy.levenshtein import Levenshtein
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from strsimpy.weighted_levenshtein import WeightedLevenshtein

from scipy.signal import find_peaks

lever = Levenshtein()
nlever = NormalizedLevenshtein()

def extractTraces(log: EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY)-> np.ndarray:
    """Extract the traces from the event log, i.e., a view concerned only with the executed activity.

    Args:
        log (EventLog): The event log
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
    Returns:
        np.ndarray: An array of tuples of executed activities (in order).
    """

    out = np.empty(len(log), dtype=object)
    for index, case in enumerate(log):
        out[index] = tuple(evt[activityName_key] for evt in case)
    return out

def extractTraces_ActivityServiceTimes(log:EventLog, bin_num:int=3, activityName_key:str=xes.DEFAULT_NAME_KEY, startTime_key:str=xes.DEFAULT_START_TIMESTAMP_KEY,completion_key:str=xes.DEFAULT_TIMESTAMP_KEY)->np.ndarray:
    """Extract the traces from the event log with a special focus on service times, i.e., a view concerned only with the executed activity, and how long its execution took. Only defined for Interval Event Logs, i.e. each event has a Start and Complete timestamp

    Args:
        log (EventLog): The event log
        bin_num (int, optional): The number of bins to cluster the service times into (Using K-Means). Defaults to 3 (intuitively "slow", "medium", "fast").
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        startTime_key (str, optional): The key in the event log for the start timestamp of the event. Defaults to xes.DEFAULT_START_TIMESTAMP_KEY.
        completion_key (str, optional): The key in the event log for the completion timestamp of the event. Defaults to xes.DEFAULT_TIMESTAMP_KEY.

    Returns:
        np.ndarray: A sequence of traces, represented as a tuple of Activities, but here the activities are tuples of the activity name and how long that activity took to complete. Same order as in the original event log.
    """

    out = np.empty(len(log), dtype=object)
    for index, case in enumerate(log):
        trace = tuple(evt[activityName_key] for evt in case)
        out[index] = tuple((evt[activityName_key],(evt[completion_key]-evt[startTime_key]).total_seconds()) for evt in case)
    # return out

    #Now cluster the durations for binning
    # Use pareto principle; 20% of cases represent 80% of interesting behavior; Sample 20% only
    sample = np.random.choice(out, size=math.ceil(0.2*len(log)))
    durations = [dur for trace in sample for _,dur in trace]
    centroid, _ = kmeans_plusplus(durations, n_clusters=bin_num, iter=10, minit='random', missing='warn')

    #Bin the actual data
    def _closestCentroid1D(point, centroids):
        mindex = 0
        minval = centroids[0]
        for idx, centroid in enumerate(centroids):
            dist = abs(centroid-point)
            if dist < minval:
                mindex = idx
                minval = dist
        return mindex

    for index in len(out):
        trace = out[index]
        out[index] = tuple(
            #Assigning 
            (act,_closestCentroid1D(dur,centroid))
            for act,dur in trace
        )
    return out

# def postnormalized_weightedLevenshteinDistance():
#     pass

def postNormalizedLevenshteinDistance(trace1:Iterable[str], trace2:Iterable[str])-> float:
    """Compute the post-normalized Levenshtein distance between two traces. (Levenshtein Distance divided by the maximal length of the two traces).

    Args:
        trace1 (Iterable[str]): The first trace.
        trace2 (Iterable[str]): The second trace.

    Returns:
        float: The post-normalized Levenshtein distance of the two traces.
    """

    return nlever.distance(trace1, trace2)

def lev(trace1:Iterable[str], trace2:Iterable[str])-> float:
    """Compute the Levenshtein distance between two traces.

    Args:
        trace1 (Iterable[str]): The first trace.
        trace2 (Iterable[str]): The second trace.

    Returns:
        float: The post-normalized Levenshtein distance of the two traces.
    """
    return lever.distance(trace1, trace2)

def postNormalizedWeightedLevenshteinDistance(trace1:Iterable[float], trace2:Iterable[float], rename_cost:float, insertion_deletion_cost:float, cost_time_match_rename:float, cost_time_insert_delete:float)-> float:
    """Compute the post-normalized weighted Levenshtein distance. This is used for the time-aware concept drift detection.

    Args:
        trace1 (Iterable[str]): The first trace.
        trace2 (Iterable[str]): The second trace.
        rename_cost (float): Custom Cost.
        insertion_deletion_cost (float): Custom Cost.
        cost_time_match_rename (float): Custom Cost.
        cost_time_insert_delete (float): Custom Cost.

    Returns:
        float: _description_
    """
    return weightedLevenshteinDistance( trace1,
                                        trace2,
                                        rename_cost=rename_cost,
                                        insertion_deletion_cost=insertion_deletion_cost,
                                        cost_time_match_rename=cost_time_match_rename,
                                        cost_time_insert_delete=cost_time_insert_delete
                                    ) / max(len(trace1),len(trace2))

def weightedLevenshteinDistance(trace1:Iterable[str], trace2:Iterable[str], rename_cost:float, insertion_deletion_cost:float, cost_time_match_rename:float, cost_time_insert_delete:float, previous_distances:Dict[Tuple[Iterable[str],Iterable[str]],float]=None, return_distances=False)->float:
    """Compute the levenshtein distance with custom weights.

    Args:
        trace1 (Iterable[str]): The first trace.
        trace2 (Iterable[str]): The second trace.
        rename_cost (float): Custom Cost.
        insertion_deletion_cost (float): Custom Cost.
        cost_time_match_rename (float): Custom Cost.
        cost_time_insert_delete (float): Custom Cost.
        previous_distances (Dict[Tuple[Iterable[str],Iterable[str]],float], optional): A dictionary mapping pairs of traces (tuples) to already computed distances. Helpful for keeping computation time low. Defaults to None.
        return_distances (bool, optional): Whether or not a dictionary of computed distances shoud be returned. Defaults to False. If `previous_distances` was supplied, this dictionary is updated and returned.

    Returns:
        float: The computed weighted Levenshtein distance.
    """    


    if trace1 == trace2:
        return 0 if not return_distances else (0, previous_distances)
    elif len(trace1) == 0:
        #Insert the traces 
        ret = 0
        for act, dur in trace2:
            ret += insertion_deletion_cost(act) + cost_time_insert_delete(dur)
        return ret if not return_distances else (ret, previous_distances)
    elif len(trace2) == 0:
        ret = 0
        for act, dur in trace2:
            ret += insertion_deletion_cost(act) + cost_time_insert_delete(dur)
        return ret if not return_distances else (ret, previous_distances)

    act1, time1 = trace1[-1]
    act2, time2 = trace2[-1]

    if previous_distances is None:
        previous_distances = {}

    if (trace1[:-1],trace2[:-1]) not in previous_distances:
        dist1, previous_distances = weightedLevenshteinDistance(trace1[:-1],trace2[:-1],rename_cost,insertion_deletion_cost,cost_time_match_rename,cost_time_insert_delete, previous_distances=previous_distances, return_distances=True)
        previous_distances[(trace1[:-1],trace2[:-1])] = dist1
    else:
        dist1 = previous_distances[(trace1[:-1],trace2[:-1])]

    if (trace1,trace2[:-1]) not in previous_distances:
        dist2, previous_distances = weightedLevenshteinDistance(trace1,trace2[:-1],rename_cost,insertion_deletion_cost,cost_time_match_rename,cost_time_insert_delete, previous_distances=previous_distances, return_distances=True)
        previous_distances[(trace1,trace2[:-1])] = dist2
    else:
        dist2 = previous_distances[(trace1,trace2[:-1])]

    if (trace1[:-1],trace2) not in previous_distances:
        dist3, previous_distances = weightedLevenshteinDistance(trace1[:-1],trace2,rename_cost,insertion_deletion_cost,cost_time_match_rename,cost_time_insert_delete, previous_distances=previous_distances, return_distances=True)
        previous_distances[(trace1[:-1],trace2)] = dist3
    else:
        dist3 = previous_distances[(trace1[:-1],trace2)]

    if act1 == act2:
        distance =  min(
            dist1 + cost_time_match_rename(time1,time2),
            dist1 + rename_cost(act1,act2) + cost_time_match_rename(time1,time2),
            dist2 + insertion_deletion_cost(act2) + cost_time_insert_delete(time2),
            dist3 + insertion_deletion_cost(act1) + cost_time_insert_delete(time1)
        )
    else:
        distance =  min(
            dist1 + rename_cost(act1,act2) + cost_time_match_rename(time1,time2),
            dist2 + insertion_deletion_cost(act2) + cost_time_insert_delete(time2),
            dist3 + insertion_deletion_cost(act1) + cost_time_insert_delete(time1)
        )
    return distance if not return_distances else (distance, previous_distances)

def calcEMD(dist1:List[Tuple[Iterable[str], float]], dist2: List[Tuple[Iterable[str], float]], previous_distances:Dict[Tuple[Iterable[str],Iterable[str]],float]=None,return_distances:bool=False)->float:
    """Calculate the Earth-Mover's Distance of two distributions of traces.

    Args:
        dist1 (List[Tuple[Iterable[str], float]]): The stochastic language of the first population, represented as a list of tuples containing the trace and its probability.
        dist2 (List[Tuple[Iterable[str], float]]): The stochastic language of the second population, represented as a list of tuples containing the trace and its probability.
        previous_distances (Dict[Tuple[Iterable[str],Iterable[str]],float], optional): A dictionary mapping pairs of traces (tuples) to already computed distances. Helpful for keeping computation time low, as these are used internally instead of re-computing the distances. Defaults to None.
        return_distances (bool, optional): Whether or not a dictionary of computed distances shoud be returned. Defaults to False. If `previous_distances` was supplied, this dictionary is updated and returned.

    Returns:
        float: The computed Earth-Mover's Distance

    Examples:
    >>> from cdrift.approaches.earthmover import calcEMD
    >>> f1 = [(('a','b','d','f'),.5), (('a','c','f'),.4),(('a','b','e','f'),.1)]
    >>> f2 = [(('a','b','d','f'),.5), (('a','c','f'),.35), (('a','b','d','e','f'),.15)]
    >>> f3 = [(('a','b','d','f'),.2), (('a','c','f'),.7),(('a','b','e','f'),.1)]
    >>> calcEMD(f1,f2)
    >>> 0.049999999999999996 
    >>> calcEMD(f1,f3)
    >>> 0.14999999999999997
    """    

    #Return calculated distances for faster execution in future iterations
    # Input is formatted:
    #[(trace, probability)]
    distances = np.ones((len(dist1), len(dist2)))
    if previous_distances is None:
        previous_distances = {}
    for i, (trace1,_) in enumerate(dist1):
        for j, (trace2,_) in enumerate(dist2):
            if previous_distances.get((trace1,trace2),None) is None:
                previous_distances[(trace1,trace2)] = postNormalizedLevenshteinDistance(trace1,trace2)
            distances[i,j] = previous_distances[(trace1,trace2)]
    
    solver = EMD()
    distance = solver(
        [freq for _,freq in dist1],
        [freq for _,freq in dist2],
        distances
    )
    return distance if not return_distances else (distance, previous_distances)

#f1 = [(('a','b','d','f'),.5), (('a','c','f'),.4),(('a','b','e','f'),.1)]
#f2 = [(('a','b','d','f'),.5), (('a','c','f'),.35), (('a','b','d','e','f'),.15)]
#f3 = [(('a','b','d','f'),.2), (('a','c','f'),.7),(('a','b','e','f'),.1)]
# calcEMD(f1,f2) should be 0.05 - i get 0.049999999999999996 - close enough
# calcEMD(f1,f3) should be 0.15 - i get 0.14999999999999997 - close enough



# This function has never been tested. So use with a LOT of caution, most definitely at least one error :P
# def calcEMD_Timing(dist1, dist2, rename_cost:Callable, id_cost:Callable, time_matchrename_cost:Callable, time_id_cost:Callable, previous_distances:Dict=None,return_distances:bool=False):
#     #Return calculated distances for faster execution in future iterations
#     # Input is formatted:
#     #[(trace, probability)]
#     distances = np.ones((len(dist1), len(dist2)))
#     if previous_distances is None:
#         previous_distances = {}
#     #Trace1 and Trace2 are not traces, but traces plus timing information
#     for i, (trace1,_) in enumerate(dist1):
#         for j, (trace2,_) in enumerate(dist2):
#             if previous_distances.get((trace1,trace2),None) is None:
#                 previous_distances[(trace1,trace2)] = postNormalizedWeightedLevenshteinDistance(trace1,trace2, rename_cost, id_cost, time_matchrename_cost, time_id_cost)
#             distances[i,j] = previous_distances[(trace1,trace2)]
    
#     solver = EMD()
#     distance = solver(
#         [freq for _,freq in dist1],
#         [freq for _,freq in dist2],
#         distances
#     )
#     return distance if not return_distances else (distance, previous_distances)

#TODO Stride 
def calculateDistSeries(signal:np.ndarray, windowSize:int, show_progressBar:bool=True, progressBar_pos:int=None)->np.ndarray:
    """Calculate the time series of Earth Mover's Distances.

    Args:
        signal (np.ndarray): The sequence of traces, extracted using extractTraces
        windowSize (int): The window size to use for the sliding-window application of Earth Mover's Distance Calculations
        show_progressBar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBar_pos (int, optional): The argument `pos` for tqdm progress bars. In which line to print the progress bar. Defaults to None.

    Returns:
        np.ndarray: The computed sequence of Earth Mover's Distances. Dimensions: 1x|log|
    """    

    if show_progressBar:
        progress = makeProgressBar(len(signal)-(2*windowSize), "calculating earthmover values, completed windows", position=progressBar_pos)
    #Default to Zeroes, because we are looking at distances, not pvalues!
    pvals = np.zeros(len(signal))
    distances = {}
    for i in range(len(signal)-(2*windowSize)):
        win1 = signal[i:i+windowSize]
        win2 = signal[i+windowSize:i+(2*windowSize)]
        #Convert to stochastic languages
        pop1 = [(trace,freq/len(win1)) for (trace,freq) in Counter(win1).items()]
        pop2 = [(trace,freq/len(win2)) for (trace,freq) in Counter(win2).items()]
        distance, distances = calcEMD(pop1, pop2, previous_distances=distances, return_distances=True)
        pvals[i+windowSize] = distance
        if show_progressBar:
            progress.update()
    if show_progressBar:
        progress.close()
    return pvals


# This function has never been tested. So use with a LOT of caution, most definitely at least one error :P
# def calculateDistSeries_Timing(signal:np.ndarray, windowSize:int, rename_cost:Callable, id_cost:Callable, time_matchrename_cost:Callable, time_id_cost:Callable):
#     """
#         NEVER TESTED; DEPRECATED
#     """
#     progress = makeProgressBar(len(signal)-windowSize, "calculating earthmover values, completed windows ::")
#     #Default to Zeroes, because we are looking at distances!
#     pvals = np.zeros(len(signal))
#     distances = {}
#     for i in range(len(signal)-(2*windowSize)):
#         win1 = signal[i:i+windowSize]
#         win2 = signal[i+windowSize:i+(2*windowSize)]
#         #Convert to stochastic languages
#         #Trace is a trace + view
#         pop1 = [(trace,freq/len(win1)) for (trace,freq) in Counter(win1).items()]
#         pop2 = [(trace,freq/len(win2)) for (trace,freq) in Counter(win2).items()]
#         distance, distances = calcEMD_Timing(pop1, pop2, rename_cost, id_cost, time_matchrename_cost, time_id_cost, previous_distances=distances, return_distances=True)
#         pvals[i+windowSize] = distance
#         progress.update()
#     progress.close()
#     return pvals


def visualInspection(signal, trim:int=0):
    """Automated visual inspection of distances. Used for consistent and unbiased evaluations.

    Based on the `find_peaks` algorithm of scipy.

    Args:
        signal (np.ndarray): The EMD values to inspect
        trim (int, optional): The number of values to trim from each side before detection. Defaults to 0. This is useful, because `windowSize` values at the beginning and end of the resulting EMD series default to 0, and are uninteresting and irrelevant for the inspection. If a stride value was used, the signal is already trimmed, so keep this value 0!

    Returns:
        List[int]: A list of found change point indices (integers)
    """
    peaks= find_peaks(signal[trim:len(signal)-trim], width=80)[0]
    # return find_peaks(signal, width=80)[0] # Used for Earthmover Distance; The distances have a different nature than minima so prominence is ignored
    return [x+trim for x in peaks] # Correct the found indices, these indices count from the beginning of the trimmed version instead of from the beginning of the untrimmed version (which we want)


def calculateDistSeriesStride(signal:np.ndarray, windowSize:int, stride:int=1, show_progressBar:bool=True, progressBar_pos:int=None):
    if show_progressBar:
        progress = makeProgressBar(len(signal)-(2*windowSize), "calculating earthmover values, completed windows", position=progressBar_pos)
    #Default to Zeroes, because we are looking at distances, not pvalues!
    pvals = []
    distances = {}

    i = 0
    while i < len(signal) - (2*windowSize):
        win1 = signal[i:i+windowSize]
        win2 = signal[i+windowSize:i+(2*windowSize)]
        #Convert to stochastic languages
        pop1 = [(trace,freq/len(win1)) for (trace,freq) in Counter(win1).items()]
        pop2 = [(trace,freq/len(win2)) for (trace,freq) in Counter(win2).items()]
        distance, distances = calcEMD(pop1, pop2, previous_distances=distances, return_distances=True)
        pvals.append(distance)
        if show_progressBar:
            progress.update(n=min(progress.total-progress.n, stride))
        i += stride
    if show_progressBar:
        progress.close()
    return pvals


def visualInspection_Stride(signal, window_size:int, step_size:int=1):
    """Automated visual inspection of distances. Used for consistent and unbiased evaluations.

    Based on the `find_peaks` algorithm of scipy.

    Args:
        signal (np.ndarray): The EMD values to inspect
        trim (int, optional): The number of values to trim from each side before detection. Defaults to 0. This is useful, because `windowSize` values at the beginning and end of the resulting EMD series default to 0, and are uninteresting and irrelevant for the inspection. If a stride value was used, the signal is already trimmed, so keep this value 0!

    Returns:
        List[int]: A list of found change point indices (integers)
    """

    # Divide width by step size because if we look for a peak with width 80 on step size 1, this corresponds to a width of 40 on step size 2
    # This could become problematic with higher step sizes since the width would go towards 0...
    peaks= find_peaks(signal, width=80/step_size)[0]

    # Correct the found indices by adding the window size that was lost
    # also multiply by step size to get location in log
    return [(x*step_size) + window_size for x in peaks]

def detect_change(log:EventLog, window_size:int, stride:int=1, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progress_bar_pos:int=None):
    traces = extractTraces(log, activityName_key=activityName_key)
    dists = calculateDistSeriesStride(traces,window_size,stride,show_progress_bar,progress_bar_pos)
    return visualInspection_Stride(dists, window_size, stride)
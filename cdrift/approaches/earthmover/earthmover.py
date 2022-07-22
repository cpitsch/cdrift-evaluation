import math
from typing import Dict, Iterable, Tuple
from pm4py.objects.log.obj import EventLog
import pm4py.util.xes_constants as xes
import numpy 
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

def extractTraces(log: EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY)-> numpy.ndarray:
    #TODO: SORT IF NOT SORTED
    out = numpy.empty(len(log), dtype=object)
    for index, case in enumerate(log):
        out[index] = tuple(evt[activityName_key] for evt in case)
    return out

def extractTraces_ActivityServiceTimes(log:EventLog, bin_num:int=3, activityName_key:str=xes.DEFAULT_NAME_KEY, startTime_key:str=xes.DEFAULT_START_TIMESTAMP_KEY,completion_key:str=xes.DEFAULT_TIMESTAMP_KEY):
    """
        Note: Only defined for Interval Event Logs, i.e. each event has a Start and Complete timestamp

        Returns a trace, represented as a tuple of Activities, but here the activities are tuples of the activity name and how long that activity took to complete
    
        Default number of bins/clusters is 3, intuitively: "slow" "medium" and "fast"
    """
    out = numpy.empty(len(log), dtype=object)
    for index, case in enumerate(log):
        trace = tuple(evt[activityName_key] for evt in case)
        out[index] = tuple((evt[activityName_key],(evt[completion_key]-evt[startTime_key]).total_seconds()) for evt in case)
    # return out

    #Now cluster the durations for binning
    # Use pareto principle; 20% of cases represent 80% of interesting behavior; Sample 20% only
    sample = numpy.random.choice(out, size=math.ceil(0.2*len(log)))
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

def postNormalizedLevenshteinDistance(trace1:Tuple[str,...], trace2:Tuple[str,...])-> float:
    return nlever.distance(trace1, trace2)

def lev(trace1:Iterable, trace2:Iterable)-> float:
    return lever.distance(trace1, trace2)

def postNormalizedWeightedLevenshteinDistance(trace1, trace2, rename_cost, insertion_deletion_cost, cost_time_match_rename, cost_time_insert_delete)-> float:
    return weightedLevenshteinDistance(trace1,trace2)/max(len(trace1),len(trace2))

def weightedLevenshteinDistance(trace1, trace2, rename_cost, insertion_deletion_cost, cost_time_match_rename, cost_time_insert_delete, previous_distances=None, return_distances=False):
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

def calcEMD(dist1, dist2, previous_distances:Dict=None,return_distances:bool=False):
    """
        params:
            - dist1
                - Distribution 1; A stochastic language (Set of traces with an associated "probability" (=count/total))
            - dist2
                - Distribution 2; A stochastic language (Set of traces with an associated "probability" (=count/total))
            - previous_distances: Dict
                - The distances from the previous iteration; Used internally to speed up calculations
            - return_distances:bool
                - Configures whether the distances should be returned; Only used internally for the same reason previous_distances is
    """
    #Return calculated distances for faster execution in future iterations
    # Input is formatted:
    #[(trace, probability)]
    distances = numpy.ones((len(dist1), len(dist2)))
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
#     distances = numpy.ones((len(dist1), len(dist2)))
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
def calculateDistSeries(signal:numpy.ndarray, windowSize:int, show_progressBar:bool=True, progressBar_pos:int=None):
    if show_progressBar:
        progress = makeProgressBar(len(signal)-(2*windowSize), "calculating earthmover values, completed windows", position=progressBar_pos)
    #Default to Zeroes, because we are looking at distances, not pvalues!
    pvals = numpy.zeros(len(signal))
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
# def calculateDistSeries_Timing(signal:numpy.ndarray, windowSize:int, rename_cost:Callable, id_cost:Callable, time_matchrename_cost:Callable, time_id_cost:Callable):
#     """
#         NEVER TESTED; DEPRECATED
#     """
#     progress = makeProgressBar(len(signal)-windowSize, "calculating earthmover values, completed windows ::")
#     #Default to Zeroes, because we are looking at distances!
#     pvals = numpy.zeros(len(signal))
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
    """
        Detects maxima in a signal
    """
    peaks= find_peaks(signal[trim:len(signal)-trim], width=80)[0]
    # return find_peaks(signal, width=80)[0] # Used for Earthmover Distance; The distances have a different nature than minima so prominence is ignored
    return [x+trim for x in peaks] # Correct the found indices, these indices count from the beginning of the trimmed version instead of from the beginning of the untrimmed version (which we want)
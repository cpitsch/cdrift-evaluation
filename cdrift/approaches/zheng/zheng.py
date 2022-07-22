from pm4py.objects.log.obj import EventLog
from typing import Dict, List, Set
import pm4py.util.xes_constants as xes
from sklearn.cluster import DBSCAN
import numpy as np
from numpy.typing import NDArray

from cdrift.utils.helpers import makeProgressBar, _getActivityNames

def calcRelationMatrix(log:EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY, progress=None):
    activities = _getActivityNames(log, activityName_key=activityName_key)
    num_activities = len(activities)

    # Each column corresponds to a trace
    # Each Row corresponds to a Relation

    drelmatrix = np.zeros( # Directly follows relations
        ((num_activities**2), len(log))
    ) # Initialize to 0, so we can just look at the 1's
    wrelmatrix = np.zeros( # Weak Relations
        ((num_activities**2), len(log))
    ) # Initialize to 0, so we can just look at the 1's
    for idx,trace in enumerate(log):
        # idx is the column
        # Set the cell to 1 where there is a directly (or weak) follows relation between the activities
        seen = set() # Set of indices of the activities we have already seen in the trace. Used to update weak relations
        for i in range(len(trace)-1): # Exclude last element as it has no follower
            act1_index = activities.index(trace[i][activityName_key])
            act2_index = activities.index(trace[i+1][activityName_key])
            # Update Directly follows Cell for this relation
            drelmatrix[act1_index*num_activities + act2_index, idx] = 1
            wrelmatrix[act1_index*num_activities + act2_index, idx] = 1
            for act in seen:
                wrelmatrix[act*num_activities + act1_index,idx]
            seen.add(act1_index)
        if progress is not None:
            progress.update() # Completed  trace
    return np.append(drelmatrix,wrelmatrix,axis=0)

def candidateCPDetection(relMatrixRow:NDArray, mrid:int):
    P = set()
    begin = 0
    count = 0
    n = len(relMatrixRow)

    for j in range(n): # n+1 so n is included
        if begin == 0 or relMatrixRow[j] != relMatrixRow[begin]:
            if count >= mrid:
                P.update([begin, j])
            begin = j
            count = 0
        count += 1
    if count >= mrid:
        P.add(begin)
    P.difference_update([1])
    return P

def candidateChangepointsCombinataion(S:Set[int], mrid:int, eps:float, n:int):
    minPts = 1
    result = [1,n]

    if S == set(): # If there are no candidates
        # Otherwise DBSCAN will yield an error
        return result

    s_as_list = list(S) # So i can be certain about the order
    clustering = DBSCAN(eps=eps, min_samples=minPts).fit(np.array(s_as_list).reshape(-1, 1))

    map_to_cluster = []
    for i in range(len(s_as_list)):
        map_to_cluster.append(
            (s_as_list[i], clustering.labels_[i])
        )
    cluster_names = set(clustering.labels_) # All the different clusters
    clusters = [
        (cluster_name, [
            pt for pt,clust in map_to_cluster if clust == cluster_name
        ]) for cluster_name in cluster_names
    ] # "Maps" each cluster name to the list of elements in it
    # Remove empty clusters (because apparently i have to do that??)
    clusters = [(x,y) for x,y in clusters if y != []]

    # Sort it
    clusters.sort(key=lambda x: len(x[1]), reverse=True)# Sort w.r.t. the size of the cluster, descending
    for clust_name, clust_members in clusters:
        c = np.average(clust_members)
        leftIndex = None
        rightIndex = None
        for i in range(len(result)-1):
            if result[i] < c and c < result[i+1]:
                leftIndex = i
                rightIndex = i+1
                break;
        if rightIndex is not None and leftIndex is not None:
            if c-leftIndex >= mrid and rightIndex-c<= mrid:
                result.insert(rightIndex,c)
        else:
            # This should not happen
            raise Exception("Couldn't find a place in results list for changepoint. This should not happen, contact the developer")
    return result

def apply(log:EventLog, mrid:int, eps:float, activityName_key:str=xes.DEFAULT_NAME_KEY, progressPos:int=None):
    """
        Applies the concept drift detection by Zheng et al. for multiple different epsilon values. 

        Makes use of the fact that the change point candidates are independent of epsilon, so this calculation does not have to be repeated for each epsilon

        params:
            log: pm4py.objects.log.obj.EventLog
                - The Event Log which is to be analyzed
            mrid:int
                - The "Minimum Relation Invariance Distance"
                    - The relation's value (1 or 0) must not change for at least this many traces in a row for its change to suggest a change point
            eps:float
                - The epsilon to use for the DBSCAN clustering of the Change point candidates.
        returns:
            cp: List[int]
                - A list of the change points found.
    """
    progress = makeProgressBar(len(log),"Extracting Relation Matrix for (Zheng)", position=progressPos)
    d = calcRelationMatrix(log, activityName_key=activityName_key, progress=progress)
    progress.set_description("Finding Changepoint Candidated (Zheng)")
    _resetPBar_PreserveTime(progress, newTotal=d.shape[0]) # 1 update per row
    P = set()
    for row in d:
        P.update(candidateCPDetection(row, mrid=mrid))
        progress.update()
    cp = candidateChangepointsCombinataion(P, mrid=mrid, eps=eps, n=len(log))
    # Zheng has convention to have first and last index as changepoints, but we don't:
    cp.remove(1)
    cp.remove(len(log))
    progress.close()
    return cp

def applyMultipleEps(log:EventLog, mrid:int, epsList:List[float], activityName_key:str=xes.DEFAULT_NAME_KEY, progressPos:int=None):
    """
        Applies the concept drift detection by Zheng et al. for multiple different epsilon values. 

        Makes use of the fact that the change point candidates are independent of epsilon, so this calculation does not have to be repeated for each epsilon

        params:
            log: pm4py.objects.log.obj.EventLog
                - The Event Log which is to be analyzed
            mrid:int
                - The "Minimum Relation Invariance Distance"
                    - The relation's value (1 or 0) must not change for at least this many traces in a row for its change to suggest a change point
            epsList:List[int]
                - A list of epsilons to use for the DBSCAN clustering of the Change point candidates.
        returns:
            cps: Dict[fload: List[int]]
                - A dictionary mapping every Epsilon found in ```epsList``` to the change points found for it.
    """
    progress = makeProgressBar(len(log), "Extracting Relation Matrix for (Zheng)", position=progressPos)
    d = calcRelationMatrix(log, activityName_key=activityName_key, progress=progress)


    progress.set_description("Finding Changepoint Candidates (Zheng)")
    _resetPBar_PreserveTime(progress, newTotal=d.shape[0]) # 1 update per row
    # progress.reset(total=d.shape[0]) # 1 update per row
    P = set()
    for row in d:
        P.update(candidateCPDetection(row, mrid=mrid))
        progress.update()

    progress.set_description("Combining changepoint candidates for different Epsilons (Zheng)")

    _resetPBar_PreserveTime(progress, newTotal=len(epsList)) # 1 update per row
    # progress.reset(total=len(epsList)) # 1 update per row
    cps:Dict[float: List[int]] = dict() # Maps an epsilon to the changepoints found for it
    for eps in epsList:
        cp = candidateChangepointsCombinataion(P, mrid=mrid, eps=eps, n=len(log))
        # Zheng has convention to have first and last index as changepoints, but we don't:
        cp.remove(1)
        cp.remove(len(log))
        cps[eps] = cp
        progress.update()
    progress.close()
    return cps


def _resetPBar_PreserveTime(progress, newTotal=None):
    """Reset the given progress bar without resetting the time elapsed."""

    if newTotal is not None:
        progress.total = newTotal
    progress.n = 0
    progress.refresh()
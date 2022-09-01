from pm4py.objects.log.obj import EventLog
from typing import Dict, List, Set
import pm4py.util.xes_constants as xes
from sklearn.cluster import DBSCAN
import numpy as np
from numpy.typing import NDArray
from cdrift.utils.helpers import safe_update_bar

from cdrift.utils.helpers import makeProgressBar, _getActivityNames

def calcRelationMatrix(log:EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY, progress=None)->np.ndarray:
    """Calculate the Relation Matrix defined by Zheng et al. Contains for each Case, and each Eventually- and Directly-Follows relation a 1 if it holds in the case, and 0 otherwise.

    Args:
        log (EventLog): The event log.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        progress (Any, optional): A progress bar to update at every completed case. Defaults to None.

    Returns:
        np.ndarray: The calculated Relation Matrix. Contains for each Case, and each Eventually- and Directly-Follows relation a 1 if it holds in the case, and 0 otherwise. Dimensions: 2(num_activities^2) x len(log)
    """    

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
        safe_update_bar(progress)
    return np.append(drelmatrix,wrelmatrix,axis=0)

def candidateCPDetection(relMatrixRow:NDArray, mrid:int)->Set[int]:
    """Extract candidate change points from a row of the Relation Matrix. These are points where the relation of this row held (or did not hold) for `mrid` consecutive traces, and then the relation changed.

    Args:
        relMatrixRow (NDArray): A row of the Relation Matrix.
        mrid (int): The Minimum Relation Invariance Distance. How long a relationship must remain stable before its change is concidered a change point candidate.

    Returns:
        Set[int]: A set of indices of the change point candidates.
    """    

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

def candidateChangepointsCombinataion(S:Set[int], mrid:int, eps:float, n:int)->List[float]:
    """Combine candidate change points by clustering them (Using DBScan), to find change point indices

    Args:
        S (Set[int]): The set of found candidate change points (From all Matrix rows).
        mrid (int): The Minimum Relation Invariance Distance. How long a relationship must remain stable before its change is concidered a change point candidate.
        eps (float): The epsilon parameter used for the DBSCAN clustering algorithm.
        n (int): The index of the last trace in the log. Zheng et al. consider this, and the first case, change point candidates as well.

    Raises:
        Exception: No neighbor change points were found for a detected change point. This should not occur due to the fact that `1` and `n` are also considered change points.

    Returns:
        List[float]: A list of change point indices. These are floats due to the use of a clustering algorithm.
    """    


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

def apply(log:EventLog, mrid:int, eps:float, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressPos:int=None)->List[float]:
    """Apply concept drift detection using the algorithm of Zheng et al.

    Args:
        log (EventLog): The event log.
        mrid (int): The Minimum Relation Invariance Distance. How long a relationship must remain stable before its change is concidered a change point candidate.
        eps (float): The epsilon parameter used for the DBSCAN clustering algorithm.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.
        progressPos (int, optional): The `pos` parameter for tqdm progress bars. In which line to print the progress bar. Defaults to None.

    Returns:
        List[float]: A list of change point indices. These are floats due to the use of a clustering algorithm.
    """
    progress = None
    if show_progress_bar:
        progress = makeProgressBar(len(log),"Extracting Relation Matrix for (Zheng)", position=progressPos)

    d = calcRelationMatrix(log, activityName_key=activityName_key, progress=progress)

    if progress is not None:
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
    if progress is not None:
        progress.close()
    return cp

def applyMultipleEps(log:EventLog, mrid:int, epsList:List[float], activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressPos:int=None):
    """Apply concept drift detection using the algorithm of Zheng et al. for multiple different epsilon values for DBSCAN.

    Args:
        log (EventLog): The event log.
        mrid (int): The Minimum Relation Invariance Distance. How long a relationship must remain stable before its change is concidered a change point candidate.
        epsList (List[float]): A list of epsilon parameters used for the DBSCAN clustering algorithm.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.
        progressPos (int, optional): The `pos` parameter for tqdm progress bars. In which line to print the progress bar. Defaults to None.

    Returns:
        Dict[float, List[float]]: A dictionary mapping an epsilon parameter to the list of detected change point indices using this epsilon. The change points are floats due to the use of a clustering algorithm.
    """
    progress = None
    if show_progress_bar:
        progress = makeProgressBar(len(log), "Extracting Relation Matrix for (Zheng)", position=progressPos)
    d = calcRelationMatrix(log, activityName_key=activityName_key, progress=progress)

    if progress is not None:
        progress.set_description("Finding Changepoint Candidates (Zheng)")
        _resetPBar_PreserveTime(progress, newTotal=d.shape[0]) # 1 update per row
        # progress.reset(total=d.shape[0]) # 1 update per row
    P = set()
    for row in d:
        P.update(candidateCPDetection(row, mrid=mrid))
        safe_update_bar(progress)
    if progress is not None:
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
        safe_update_bar(progress)
    if progress is not None:
        progress.close()
    return cps


def _resetPBar_PreserveTime(progress, newTotal:int=None):
    """A helper function to reset a progress bar without changing its elapsed time. Used to reset the completed steps to 0, and choose a new goal total amount.

    Args:
        progress (Any): The progress bar.
        newTotal (int, optional): The new total value of the progress bar after resetting. Defaults to None.
    """

    if newTotal is not None:
        progress.total = newTotal
    progress.n = 0
    progress.refresh()
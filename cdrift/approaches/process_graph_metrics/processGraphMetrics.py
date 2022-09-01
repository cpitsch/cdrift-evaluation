from typing import List
from pm4py.objects.log.obj import EventLog
from pm4py.util import xes_constants as xes
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.heuristics_net.obj import HeuristicsNet

import numpy as np
import math
import scipy.stats.contingency as contingency
from scipy.stats import power_divergence

from cdrift.utils.helpers import _getActivityNames, makeProgressBar

def detectChange(log: EventLog, windowSize:int, maxWindowSize:int, pvalue:float=0.0001, activityName_key: str = xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPosition:int=None)->List[int]:
    """Apply concept drift detection using the Process Graph Metrics by Seeliger et al.

    Args:
        log (EventLog): The event log
        windowSize (int): The initial window size to be used in the adaptive window approach.
        maxWindowSize (int): The maximal size to grow the window to  before it is reset.
        pvalue (float, optional): The p-value threshold. A pvalue below this indicates a change point. Defaults to 0.0001 (The recommended value by the paper authors).
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.
        progressBarPosition (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Returns:
        List[int]: The list of detected change point indices.
    """    
    
    activities = sorted(_getActivityNames(log, activityName_key=activityName_key))
    i = 0
    pval2beforeE = 2 # Initialized greater than 1, so any found one is "better" so pval2before-1=1 which is not less than -0.5 (worst case)
    pval2beforeN = 2 # Initialized greater than 1, so any found one is "better" so pval2before-1=1 which is not less than -0.5 (worst case)
    changepoints = []

    progress = None
    if show_progress_bar:
        progress = makeProgressBar(len(log)-windowSize, "comparing heuristic miner graphs ", position=progressBarPosition)

    while i < len(log)-windowSize:
        pvalE = _testEdgeOccurences(log[i:i+windowSize], log[i+windowSize: i+2*windowSize], activities=activities)
        pvalN = _testNodeOccurences(log[i:i+windowSize], log[i+windowSize: i+2*windowSize], activities=activities)
        if pvalE < pvalue and pvalN < pvalue: # Maybe drift found
            found = False

            newWindowSize = math.ceil(windowSize/2)
            lastI = i+ (windowSize*2)

            for j in range(i+windowSize - newWindowSize, i+(windowSize*2) - newWindowSize):
                lastI = i + (newWindowSize*2)

                pval2E = _testEdgeOccurences(log[j:j+newWindowSize], log[j+newWindowSize: j+2*newWindowSize], activities=activities)
                pval2N= _testNodeOccurences(log[j:j+newWindowSize], log[j+newWindowSize: j+2*newWindowSize], activities=activities)

                if pval2beforeE - pval2E < -0.5 and pval2beforeN - pval2N < -0.5:
                    break
                pval2beforeE = pval2E
                pval2beforeN = pval2N

                if pval2E < pvalue and pval2N < pvalue:
                    changepoints.append(j + (2*newWindowSize))
                    old_i = i
                    i = j + (2*newWindowSize)
                    safeUpdateBar(progress, n=i-old_i)
                    # progress.update(n=i-old_i)
                    found = True
                    break
            if not found:
                windowSize = math.ceil((windowSize*lastI) / (i + (2*windowSize)))
                i += windowSize
                safeUpdateBar(progress, windowSize)
                # progress.update(n=windowSize)
        else:
            windowSize = math.ceil(1.2*windowSize) # Increase Window Size
        if windowSize >= maxWindowSize:
            windowSize = 100
            i = i+maxWindowSize
            safeUpdateBar(progress, maxWindowSize)
            # progress.update(n=maxWindowSize)
    safeClose(progress)
    # progress.close()
    return changepoints



def _testEdgeOccurences(log1:EventLog, log2:EventLog, activities:List[str])->float:
    """A helper function to compare the distribution of edge occurences inside the process models of two event logs found using the Heuristics Miner. Distributions compared using a G-Test.

    Args:
        log1 (EventLog): The first event log.
        log2 (EventLog): The second event log.
        activities (List[str]): The list of activities to consider. (In general the activities found in the event logs)

    Returns:
        float: The  calculated p-value, using a G-Test
    """    
    
    m1 = discoverModel(log1)
    m2 = discoverModel(log2)
    
    
    
    occs1 = []
    occs2 = []
    for act1 in activities:
        for act2 in activities:
            occs1.append(m1.dfg.get((act1,act2),0))
            occs2.append(m2.dfg.get((act1,act2),0))

    m_star = np.zeros((2,len(occs1)))
    m_star[0] = occs1
    m_star[1] = occs2

    # Perform G-Test
    ll = zip(occs1, occs2) # Observed = Detection window = second window = occs2
    ll = [(x,y) for x,y in ll if x != 0 and y != 0]
    try:
        _, p, _, _ = contingency.chi2_contingency(ll)
        return p
    except: # Chi2 failed; Probably because no edge exists where neither observed nor expected are 0; So no edges are shared so no way they are the same
        return 0
    # _, pval = G_Test(occs1, occs2)
    # return pval

def _testNodeOccurences(log1, log2, activities)->float:
    """A helper function to compare the distribution of node occurences inside the process models of two event logs found using the Heuristics Miner. Distributions compared using a G-Test.

    Args:
        log1 (EventLog): The first event log.
        log2 (EventLog): The second event log.
        activities (List[str]): The list of activities to consider. (In general the activities found in the event logs)

    Returns:
        float: The  calculated p-value, using a G-Test
    """ 

    m1 = discoverModel(log1)
    m2 = discoverModel(log2)

    occs1 = []
    occs2 = []
    for act in activities:
        occs1.append(m1.activities_occurrences.get(act,0))
        occs2.append(m2.activities_occurrences.get(act,0))

    m_star = np.zeros((2,len(occs1)))
    m_star[0] = occs1
    m_star[1] = occs2

    # Perform G-Test
    ll = zip(occs1, occs2) # Observed = Detection window = second window = occs2
    ll = [(x,y) for x,y in ll if x != 0 and y != 0]
    try:
        _, p, _, _ = contingency.chi2_contingency(ll)
        return p
    except:
        return 0
    # _, pval = G_Test(occs1, occs2)
    # return pval

def G_Test(expected, observed)->float:
    """A wrapper for the scipy.stats.chi2_contingency function, used to compute the G-Test

    Args:
        expected (Any): The expected distribution.
        observed (Any): The observed distribution.

    Returns:
        float: The computed p-value
    """

    # return 2 * sum(
    #     o*math.log(o/e) for o,e in zip(observed, expected) if o != 0 and e != 0 # This is only the statistic, not the pvalue
    # )
    return power_divergence(observed, expected, lambda_=0)

def discoverModel(logWindow: EventLog, dependencyThresh:float=0.99)->HeuristicsNet:
    """A wrapper for the PM4Py Heuristics Miner implementation.

    Args:
        logWindow (EventLog): The event (sub-) log to be used for the discovery.
        dependencyThresh (float, optional): The dependency threshold parameter used for the Heuristics Miner. Defaults to 0.99.

    Returns:
        HeuristicsNet: The discovered Heuristics Net
    """    
    return heuristics_miner.apply_heu(logWindow, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: dependencyThresh})

def safeUpdateBar(bar, n):
    """A helper function to update a progress bar's total amount without breaking it.

    Args:
        bar (Any): The tqdm progress bar.
        n (int): The new total goal amount of the progress bar.
    """
    if bar is not None:
        fdict = bar.format_dict
        bar_n = fdict['n']
        bar_total = fdict['total']
        if(bar_n + n > bar_total):
            # Increase total so it fits
            bar.total = bar_n + n
        # Now we can definitely update without problems
        bar.update(n=n)

def safeClose(bar):
    """A helper function used to close a progress bar. If its goal total has not been reached yet, its total is set to the achieved amount before closing.

    Args:
        bar (Any): The progress bar.
    """ 
    if bar is not None:  
        fdict = bar.format_dict
        n = fdict['n']
        total = fdict['total']

        if n != total:
            bar.total = n
            bar.refresh()
        bar.close()

def calcM_Star_Total(log1:EventLog, log2:EventLog, activities:List[str])->np.ndarray:
    """Calculate the M* Matrix as defined in the paper.

    Args:
        log1 (EventLog): The first event log.
        log2 (Event Log): The second event log.
        activities (List[str]): The list of activities to consider. (In general the activities found in the event logs)

    Returns:
        np.ndarray: The computed M* matrix.
    """    

    activities.sort()
    m1 = discoverModel(log1)
    m2 = discoverModel(log2)
    # Calculate M*


    # Calculate Number of Nodes
    nodes1 = m1.nodes
    nodes2 = m2.nodes
    num_nodes1 = len(nodes1)
    num_nodes2 = len(nodes2)

    # Number of Edges
    edges1 = m1.dfg.keys()
    edges2 = m2.dfg.keys()
    num_edges1 = len(edges1)
    num_edges2 = len(edges2)

    # Graph Density
    graph_density1 = num_edges1 / (num_nodes1*(num_nodes1-1))
    graph_density2 = num_edges2 / (num_nodes2*(num_nodes2-1))

    # In/Out Degrees
    in_deg_m1 = dict()
    in_deg_m2 = dict()
    out_deg_m1 = dict()
    out_deg_m2 = dict()
    for x,y in edges1:
        out_deg_m1[x] = out_deg_m1.get(x,0) + 1
        in_deg_m1[y] = in_deg_m1.get(y,0) + 1
    for x,y in edges2:
        out_deg_m2[x] = out_deg_m2.get(x,0) + 1
        in_deg_m2[y] = in_deg_m2.get(y,0) + 1

    in_degs1 = np.zeros(len(activities))
    in_degs2 = np.zeros(len(activities))
    out_degs1 = np.zeros(len(activities))
    out_degs2 = np.zeros(len(activities))

    for idx,act in enumerate(activities):
        in_degs1[idx]  = in_deg_m1[act]
        in_degs2[idx]  = in_deg_m2[act]
        out_degs1[idx] = out_deg_m1[act]
        out_degs2[idx] = out_deg_m2[act]

    # Edge occurences; Counts how often that edge happened
    occs1 = []
    occs2 = []
    for act1 in activities:
        for act2 in activities:
            if act1 in m1.dfg_matrix.keys():
                occs1.append(m1.dfg_matrix[act1].get(act2, 0))
            else:
                occs1.append(0)
            if act1 in m2.dfg_matrix.keys():
                occs2.append(m2.dfg_matrix[act1].get(act2, 0))
            else:
                occs2.append(0)
    vals1 = [num_nodes1, num_edges1, graph_density1] + in_degs1 + out_degs1 + occs1
    vals2 = [num_nodes2, num_edges2, graph_density2] + in_degs2 + out_degs2 + occs2

    m_star = np.ndarray([
        np.ndarray(vals1),# Window 1 (Reference Window)
        np.ndarray(vals2)# Window 2 (Detection Window)
    ])

    return m_star
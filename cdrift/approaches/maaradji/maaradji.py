"""
This is a revised implementation of the Approach by Maaradji et al. in Fast and Accurate Business Process Drift Detection.
I understood the approach only semi-well and reading the extension paper some problems emerged. 
I calculate the concurrency on the entire log while the revised paper discovers concurrency on the 2 populations only
"""
from collections import Counter
import math
import typing
from typing import FrozenSet, List, Tuple, Counter, Set, Union, Optional
from numpy.typing import NDArray
from pm4py.objects.log.obj import EventLog, Trace, Event
import pm4py.util.xes_constants as xes
import numpy
import scipy.stats as stats

from cdrift.helpers import transitiveReduction, makeProgressBar


def extractTraces(log:EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY)->List[Tuple[str, ...]]:
    """
        Returns the Traces Occuring in a log (Still ordered by time). So abstracts down from events to activity names
    """
    return [
        tuple(evt[activityName_key] for evt in case)
        for case in log
    ]


def _extractDirectlyFollowsCase(case:Trace, activityName_key:str=xes.DEFAULT_NAME_KEY)->typing.Counter[Tuple[str,str]]:
    """
        Returns a Set of Directly-Follows Relations between Actvity Names (as tuple), for a case
    """
    return Counter([(case[i][activityName_key],case[i+1][activityName_key]) for i in range(len(case)-1)])

def _extractDirectlyFollowsLog(log:EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY)->Set[Tuple[str,str]]:
    """
        Returns a Set of Directly-Follows Relations between Actvity Names (as tuple), for an entire Event Log.
    """
    df = set()
    for case in log:
        df.update([(case[i][activityName_key],case[i+1][activityName_key]) for i in range(len(case)-1)])
    return df

def _caseToRun(case:Trace, concurrents:Set[Tuple[str,str]], activityName_key:str=xes.DEFAULT_NAME_KEY)->FrozenSet[Tuple[str,str]]:
    """
        Converts a Case into a Run, i.e. a set of Directly Follows Relations, taking into account the concurrency between activities (described by `concurrents`)
    """
    #Compute the closure of ltdot 
    ltdot_closure = _transitiveClosure_Cases(case)
    # Remove the events with concurrent activities
    second_part = {(e1,e2) for e1,e2 in ltdot_closure if (e1[activityName_key],e2[activityName_key]) not in concurrents}
    causality_pi = transitiveReduction(second_part)
    
    #Abstract down to just activity names
    causality_pi = {
        (a[activityName_key], b[activityName_key]) for a,b in causality_pi
    }
    return frozenset(causality_pi)

def extractRuns(log:EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY, prevRuns:Optional[List[frozenset]]=None, prevConcurrents:Optional[Set[Tuple[str,str]]]=None, returnConcurrents:bool=False)->Union[List[FrozenSet[Tuple[str,str]]], Tuple[FrozenSet[Tuple[str,str]], Set[Tuple[str,str]]]]:
    """
        Extracts the runs occuring in a Sublog, taking into account the concurrency observed in this time window. 

        Also takes `prevRuns`, which is the Runs calculated in the last step. If `prevConcurrents` is equal to the set of concurrent activities in this Log, then we only have to compute the run that is the newly detected case. All others we can simply use from the previous iteration (Remove the first as it is no longer in the time window)
    """
    #TODO: Only keep the cases which completed in the time window? We have a stream of traces; Only keep the traces which end before the last one does?
    #Find concurrency
    dfs = _extractDirectlyFollowsLog(log, activityName_key=activityName_key)
    #Check alpha concurrency
    alphas = { (a,b) for (a,b) in dfs if (b,a) in dfs } # All symmetric pairs; Can occur in any order, so they are concurrent

    if prevConcurrents == alphas and prevRuns is not None: # So if the concurrent activities are the same as in the last iteration, we can simply use the previous runs, and only calculate one new one! 
        runs = prevRuns[1:] # Remove first run from previous window as this isnt in this sublog
        runs.append(_caseToRun(log[-1], alphas, activityName_key))
    else:
        runs = []
        #This is the expensive part which makes this so slow (with especially long cases as in the Synthetic logs where the cases are upwards of 50 events long)
        for case in log:#Convert this case to a run
            runs.append(_caseToRun(case,alphas,activityName_key=activityName_key))
    return runs if not returnConcurrents else (runs, alphas)




def detectChangepoints_VerySlow(log:EventLog, windowSize:int, pvalue:float=0.05, activityName_key:str=xes.DEFAULT_NAME_KEY, return_pvalues:bool=False, progressBar_pos:Optional[int]=None)->Union[List[int], Tuple[List[int], NDArray]]:
    chis = numpy.ones(len(log))

    progress = makeProgressBar(len(log)-(2*windowSize), "applying runs cpd pipeline, windows completed ", position=progressBar_pos)
    concurrents1 = None
    concurrents2 = None
    runs1 = None
    runs2 = None
    for i in range(len(log)-(2*windowSize)):
        win1 = log[i:i + windowSize]
        win2 = log[i+windowSize:i + (2*windowSize)]

        # extract the runs in the populations
        runs1, concurrents1 = extractRuns(win1, activityName_key=activityName_key, prevRuns=runs1, prevConcurrents=concurrents1, returnConcurrents=True)
        runs2, concurrents2 = extractRuns(win2, activityName_key=activityName_key, prevRuns=runs2, prevConcurrents=concurrents2, returnConcurrents=True)
        pop1 = Counter(runs1)
        pop2 = Counter(runs2)

        # Perform Chi^2 Test
        keys  = list(set(pop2).union(set(pop1)))
        ## Create Contingency Matrix for Reference Window
        mr = numpy.zeros(len(keys)) #Matrix For reference Window
        for run in pop1:
            # Find the index of the run in the matrix by looking up in keys, then set the count, if run isnt in the detection window, then 0 by default
            mr[keys.index(run)] = pop1[run]

        #Create Contingency Matrix for Detection Window
        md = numpy.zeros(len(keys)) #Matrix For detection Window. I dont really get why it's a matrix, so i assume this matrix has only 1 row and num_unique_runs columns
        for run in pop2:
            # Find the index of the run in the matrix by looking up in keys, then set the count, if run isnt in the detection window, then 0 by default
            md[keys.index(run)] = pop2[run]
        
        # Contingency Table
        contingency = numpy.array([mr,md])

        # Apply Chi^2 Test on contingency matrix
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
        #chi2, p, dof, expected <- these are the return values of chi2_contingency; We are only interested in the p-value
        _    , p, _  , _        = stats.chi2_contingency(contingency)
        #We are testing for a changepoint at i+windowSize
        chis[i+windowSize] = p
        progress.update()

    # Find change points using the pvalues

    # The number of consecutive points with chi-square result lower than alpha to be classified as a Changepoint 
    # In the paper windowSize/3 was said to be best; We will use this
    changepoints = []
    phi = windowSize//3 #Integer division; Floor the value
    consecutive_chis = 0
    for index, val in enumerate(chis):
        if val < pvalue:
            consecutive_chis += 1
        else:
            consecutive_chis = 0
        
        if consecutive_chis == phi: # Enough Consecutive pvalues below threshold
            changepoints.append(index)
    progress.close()
    return changepoints if not return_pvalues else (changepoints, chis)

def _transitiveClosure_Cases(case:Trace)->Set[Tuple[Event, Event]]:
    """
        This is a specific, more efficient, implementation of the transitive closure, which is only correct on the Directly Follows Relations of a Case. Since here we can exploit that the relations induce a "line", so no branching, cycles or different graph components. For a case of length ```n```, runtime of ```O(n)```.
    """
    relations = set()
    seen: List[Trace] = []
    for i in range(1,len(case)+1):
        relations.update(
            [(case[-i],s) for s in seen] # Add all a relation with all the Events following this one in the Case execution
        )
        seen.append(case[-i])
    return relations

#TODO: Check and rewrite Adaptive Runs
def detectChangepointsAdaptive(log:EventLog, windowSize:int, pvalue:float=0.05, activityName_key:str=xes.DEFAULT_NAME_KEY, return_pvalues:bool=False)->Union[List[int], Tuple[List[int], NDArray]]:
    #traces = extractTraces(log, activityName_key)
    chis = numpy.ones(len(log))
    #progress = makeProgressBar(len(log)-windowSize, "applying runs cpd pipeline, windows completed ::" )
    previous_alphas = {}
    nextWindowSize = windowSize
    distinctRunsOld = None
    distinctRunsNew = None
    i = 0
    while i + 2*nextWindowSize < len(log):
        print(nextWindowSize, end="; ")
        win1 = log[i:i + nextWindowSize]
        win2 = log[i+nextWindowSize:i + (2*nextWindowSize)]

        # extract the runs in the populations
        pop1, alphas1 = extractRuns(win1, activityName_key, alphas=previous_alphas.get(i,None),return_alphas=True)
        pop2, alphas2 = extractRuns(win2, activityName_key, alphas=previous_alphas.get(i+windowSize,None),return_alphas=True)

        previous_alphas[i] = alphas1
        previous_alphas[i+windowSize] = alphas2
        # perform chi square test
        keys  = list(set(pop2).union(set(pop1)))
        #Create Contingency Matrix for Reference Window
        mr = numpy.zeros(len(keys)) #Matrix For reference Window
        for run in pop1:
            # Find the index of the run in the matrix by looking up in keys, then set the count, if run isnt in the detection window, then 0 by default
            mr[keys.index(run)] = pop1[run]

        #Create Contingency Matrix for Detection Window
        md = numpy.zeros(len(keys)) #Matrix For detection Window. I dont really get why it's a matrix, so i assume this matrix has only 1 row and num_unique_runs columns
        for run in pop2:
            # Find the index of the run in the matrix by looking up in keys, then set the count, if run isnt in the detection window, then 0 by default
            md[keys.index(run)] = pop2[run]
        
        contingency = numpy.array([mr,md])

        # Apply Chi^2 Test on contingency matrix
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
        #chi2, p, dof, expected <- these are the return values of chi2_contingency
        _    , p, _  , _        = stats.chi2_contingency(contingency)
        #We are testing for a changepoint at i+windowSize
        chis[i+windowSize] = p
        #progress.update()
        # Update Adaptive Variables
        distinctRunsOld = distinctRunsNew
        distinctRunsNew = len(keys)
        if distinctRunsOld is not None and distinctRunsNew is not None:
            #Divide by zero impossible here unless window size is 0, then we have an entirely different problem :P
            # Greater than 1 means distinctRunsOld > distinctRunsNew, which means that the amount has increased. And as such, the windowSize should decrease to compensate for this
            nextWindowSize = math.ceil(nextWindowSize * (distinctRunsOld/distinctRunsNew))
        i += 1

    # Find change points using he pvalues

    # The number of consecutive points with chi-square result lower than alpha to be classified as a Changepoint 
    # In the paper windowSize/3 was said to be best
    changepoints = []
    phi = windowSize/3
    consecutive_chis = 0
    for index, val in enumerate(chis):
        if val < pvalue:
            consecutive_chis += 1
        else:
            consecutive_chis = 0
        
        if consecutive_chis == phi:
            changepoints.append(index)
    return changepoints if not return_pvalues else (changepoints, chis)


def detectChangepoints(log:EventLog, windowSize:int, pvalue=0.05, activityName_key:str=xes.DEFAULT_NAME_KEY, return_pvalues:bool=False, progressBar_pos:Optional[int]=None)->Union[List[int], Tuple[List[int], NDArray]]:
    chis = numpy.ones(len(log))
    alphas = set([
        (case[i][activityName_key], case[i+1][activityName_key]) for case in log for i in range(len(case)-1)
    ])
    concurrents = {
        (x,y) for x,y in alphas if (y,x) in alphas
    }
    progress = makeProgressBar(len(log), "Calculating runs ", position=progressBar_pos)
    #Calculate sequence of runs
    runs = []
    trace_to_run = {}
    for case in log:
        trace = tuple(x[activityName_key] for x in case)
        if trace in trace_to_run.keys():
            runs.append(trace_to_run[trace])
        else:
            run = _caseToRun(case, concurrents,activityName_key=activityName_key)
            runs.append(run)
            trace_to_run[trace] = run
        progress.update()
    # Iterate over runs
    for i in range(len(log)-(2*windowSize)):
        runs1 = runs[i:i + windowSize]
        runs2 = runs[i+windowSize:i + (2*windowSize)]

        pop1 = Counter(runs1)
        pop2 = Counter(runs2)

        # Perform Chi^2 Test
        keys  = list(set(pop2).union(set(pop1)))
        ## Create Contingency Matrix for Reference Window
        mr = numpy.zeros(len(keys)) #Matrix For reference Window
        for run in pop1:
            # Find the index of the run in the matrix by looking up in keys, then set the count, if run isnt in the detection window, then 0 by default
            mr[keys.index(run)] = pop1[run]

        #Create Contingency Matrix for Detection Window
        md = numpy.zeros(len(keys)) #Matrix For detection Window. I dont really get why it's a matrix, so i assume this matrix has only 1 row and num_unique_runs columns
        for run in pop2:
            # Find the index of the run in the matrix by looking up in keys, then set the count, if run isnt in the detection window, then 0 by default
            md[keys.index(run)] = pop2[run]
        
        # Contingency Table
        contingency = numpy.array([mr,md])

        # Apply Chi^2 Test on contingency matrix
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
        #chi2, p, dof, expected <- these are the return values of chi2_contingency; We are only interested in the p-value
        _    , p, _  , _        = stats.chi2_contingency(contingency)
        #We are testing for a changepoint at i+windowSize
        chis[i+windowSize] = p
        # progress.update()

    # Find change points using the pvalues

    # The number of consecutive points with chi-square result lower than alpha to be classified as a Changepoint 
    # In the paper windowSize/3 was said to be best; We will use this
    changepoints = []
    phi = windowSize//3 #Integer division; Floor the value
    consecutive_chis = 0
    for index, val in enumerate(chis):
        if val < pvalue:
            consecutive_chis += 1
        else:
            consecutive_chis = 0
        
        if consecutive_chis == phi: # Enough Consecutive pvalues below threshold
            changepoints.append(index)
    progress.close()
    return changepoints if not return_pvalues else (changepoints, chis)
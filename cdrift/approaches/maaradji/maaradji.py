"""
This is a revised implementation of the Approach by Maaradji et al. in Fast and Accurate Business Process Drift Detection.
I understood the approach only semi-well and reading the extension paper some problems emerged. 
I calculate the concurrency on the entire log while the revised paper discovers concurrency on the 2 populations only
"""
from collections import Counter
import math
import typing
from typing import Any, FrozenSet, List, Tuple, Counter, Set, Union, Optional
from deprecation import deprecated
from numpy.typing import NDArray
from pm4py.objects.log.obj import EventLog, Trace, Event
import pm4py.util.xes_constants as xes
import numpy
import scipy.stats as stats

from cdrift.utils.helpers import transitiveReduction, makeProgressBar, safe_update_bar


def extractTraces(log:EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY)->List[Tuple[str, ...]]:
    """Extract traces from the event log, i.e., a view on the event log concerned only with the executed activtiy

    Args:
        log (EventLog): The event log to use
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.

    Returns:
        List[Tuple[str, ...]]: A list of traces, in form of a tuple of executed activity names. Same order as originally in the event log.
    """

    return [
        tuple(evt[activityName_key] for evt in case)
        for case in log
    ]


def _extractDirectlyFollowsCase(case:Trace, activityName_key:str=xes.DEFAULT_NAME_KEY)->Counter[Tuple[str,str]]:
    """For a case, extract the directly-follows relations occurring in it.

    Args:
        case (Trace): The case
        activityName_key (str, optional): The key for the activity value in the case. Defaults to xes.DEFAULT_NAME_KEY.

    Returns:
        Counter[Tuple[str,str]]: A Counter Object (Multiset) of directly-follows relations represented as a Tuple.
    """

    return Counter([(case[i][activityName_key],case[i+1][activityName_key]) for i in range(len(case)-1)])

def _extractDirectlyFollowsLog(log:EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY)->Set[Tuple[str,str]]:
    """Calculate the set of directly-follows relations occurring in the event log

    Args:
        log (EventLog): The event log
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.

    Returns:
        Set[Tuple[str,str]]: The set of directly-follows relations found in the event log. Each represented as a tuple of activity names.
    """
    df = set()
    for case in log:
        df.update([(case[i][activityName_key],case[i+1][activityName_key]) for i in range(len(case)-1)])
    return df

def _caseToRun(case:Trace, concurrents:Set[Tuple[str,str]], activityName_key:str=xes.DEFAULT_NAME_KEY)->FrozenSet[Tuple[str,str]]:
    """A helper function to convert a case into a run. Runs defined by Maaradji et al. in Fast And Accurate Business Process Drift Detection.

    Args:
        case (Trace): The case
        concurrents (Set[Tuple[str,str]]): A set of pairs of concurrent activities. These relations are effectively removed from the case, turning it into a partial order.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.

    Returns:
        FrozenSet[Tuple[str,str]]: A set of directly-follows relations, representing the computed partial order
    """

    # Add an index attribue to each event to discern events that are practically identical (e.g., same activity AND timestamp) (yes, I learned this the hard way)
    # TODO: Work on a copy of the case to avoid side effects
    for idx, evt in enumerate(case):
        evt['cdrift_index'] = idx

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
    """Extract the runs occurring in a (sub-) log, using the conceurrency observed in this time window

    Args:
        log (EventLog): The event log.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        prevRuns (Optional[List[frozenset]], optional): The sequence of runs observed in the previous iteration. Used to potentially decrease computation time. Defaults to None.
        prevConcurrents (Optional[Set[Tuple[str,str]]], optional): The concurrent activities observed in the previous iteration. Used to potentially decrease computation time. Defaults to None.
        returnConcurrents (bool, optional): Whether or not the concurrent activities found in this iteration should be returned. Useful for internal use, persisting these values between executions of the function. Defaults to False.

    Returns:
        Union[List[FrozenSet[Tuple[str,str]]], Tuple[FrozenSet[Tuple[str,str]], Set[Tuple[str,str]]]]: An array of runs, and, if selected, also a set of observed concurrent activities.
    """    
    # Takes `prevRuns`, which is the Runs calculated in the last step. 
    # If `prevConcurrents` is equal to the set of concurrent activities in this Log, then we only have to compute the run that is the newly detected case. 
    # All others we can simply use from the previous iteration (Remove the first as it is no longer in the time window)

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
    """Apply Change Point Detection using the ProDrift Algorithm from Fast And Accurate Business Process Drift Detection. This is a very slow implementation, as it deviates from the implementation in the paper. Here, only the alpha relations occurring in the current window are regarded for the run calculation, hence, the runs must be recalculated in every window.

    Args:
        log (EventLog): The event log.
        windowSize (int): The window size for the sliding window algorithm.
        pvalue (float, optional): P-Value threshold for a pvalue to signify a change point. Defaults to 0.05.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        return_pvalues (bool, optional): Configures whether the computed p-values should be returned as well. Defaults to False.
        progressBar_pos (Optional[int], optional): The `pos` argument of tqdm progress bars. In which line to print the progress bar. Defaults to None.

    Returns:
        Union[List[int], Tuple[List[int], NDArray]]: A list of the detected change point indices. If selected, also a numpy array of the calculated p-values
    """    

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
    """A helper function to compute the transitive closure for traces. As this is a special case of transitive closure (transitive closure of a "linear" order), we can make use of this to compute it more efficiently.

    Args:
        case (Trace): The case.

    Returns:
        Set[Tuple[Event, Event]]: A set of the computed relations of the transitive closure of the directly-follows relations of the case.
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

@deprecated("This function is deprecated and has not been tested in a very long time. Use with caution.")
def detectChangepointsAdaptive(log:EventLog, windowSize:int, pvalue:float=0.05, activityName_key:str=xes.DEFAULT_NAME_KEY, return_pvalues:bool=False)->Union[List[int], Tuple[List[int], NDArray]]:
    """An implementation of the ProDrift Algorithm using adaptive windows.

    Args:
        log (EventLog): The event log.
        windowSize (int): The window size for the sliding window algorithm.
        pvalue (float, optional): The pvalue threshold to consider a pvalue as a change point. Defaults to 0.05.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        return_pvalues (bool, optional): Configures whether the computed p-values should be returned as well. Defaults to False.

    Returns:
        Union[List[int], Tuple[List[int], NDArray]]: A list of the detected change point indices. If selected, also a numpy array of the calculated pvalues.
    """    

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


def detectChangepoints(log:EventLog, windowSize:int, pvalue=0.05, activityName_key:str=xes.DEFAULT_NAME_KEY, return_pvalues:bool=False, show_progress_bar:bool=True, progressBar_pos:Optional[int]=None)->Union[List[int], Tuple[List[int], NDArray]]:
    """Apply Change Point Detection using the ProDrift Algorithm from Fast And Accurate Business Process Drift Detection. Here, for the runs calculation, we consider the alpha relations observed in the entire event log (Event relations from the "Future")

    Args:
        log (EventLog): The event log.
        windowSize (int): The window size for the sliding window algorithm.
        pvalue (float, optional): P-Value threshold for a pvalue to signify a change point. Defaults to 0.05. Defaults to 0.05.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        return_pvalues (bool, optional): Configures whether the computed p-values should be returned as well. Defaults to False.
        progressBar_pos (Optional[int], optional): The `pos` argument of tqdm progress bars. In which line to print the progress bar. Defaults to None.

    Returns:
        Union[List[int], Tuple[List[int], NDArray]]: A list of the detected change point indices. If selected, also a numpy array of the calculated p-values
    """  
    
    chis = numpy.ones(len(log))
    dfs = set([
        (case[i][activityName_key], case[i+1][activityName_key]) for case in log for i in range(len(case)-1)
    ])
    concurrents = {
        (x,y) for x,y in dfs if (y,x) in dfs
    }

    progress = None
    if show_progress_bar:
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
        safe_update_bar(progress)
    # Iterate over runs and compute p-values
    for i in range(len(log)-(2*windowSize)):
        runs1 = runs[i:i + windowSize]
        runs2 = runs[i+windowSize:i + (2*windowSize)]

        chis[i+windowSize] = chi_square(runs1, runs2)

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
    if progress is not None:
        progress.close()
    return changepoints if not return_pvalues else (changepoints, chis)

def chi_square(runs1:List[FrozenSet[Tuple[str,str]]],runs2:List[FrozenSet[Tuple[str,str]]])->float:
    """A helper function to compute Chi-Square Test for two populations (sub-"logs" (runs))by computing the contingency table and then using the `chi2_contingency` function from scipy

    Args:
        runs1 (List[FrozenSet[Tuple[str,str]]]): Population 1. A List of Runs (FrozenSets of Tuples)
        runs2 (List[FrozenSet[Tuple[str,str]]]): Population 2. A List of Runs (FrozenSets of Tuples)

    Returns:
        float: The computed p-value
    """
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
    return p



def detectChangepoints_DynamicAlpha(log:EventLog, windowSize:int, pvalue=0.05, activityName_key:str=xes.DEFAULT_NAME_KEY, return_pvalues:bool=False, progressBar_pos:Optional[int]=None)->Union[List[int], Tuple[List[int], NDArray]]:
    """Apply Change Point Detection using the ProDrift Algorithm from Fast And Accurate Business Process Drift Detection. Here, for the runs calculation, we consider the alpha relations observed in the event log inspected so far.

    This is how I understand the explanation in the paper, however upon comparing with the existing implementation (Standalone Jar-File) I find that they seem to use the alpha relations from the entire event log (Implemented in `detectChangepoints`)

    Args:
        log (EventLog): The event log.
        windowSize (int): The window size for the sliding window algorithm.
        pvalue (float, optional): P-Value threshold for a pvalue to signify a change point. Defaults to 0.05. Defaults to 0.05.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        return_pvalues (bool, optional): Configures whether the computed p-values should be returned as well. Defaults to False.
        progressBar_pos (Optional[int], optional): The `pos` argument of tqdm progress bars. In which line to print the progress bar. Defaults to None.

    Returns:
        Union[List[int], Tuple[List[int], NDArray]]: A list of the detected change point indices. If selected, also a numpy array of the calculated p-values
    """  
    
    chis = numpy.ones(len(log))

    # Initialize Directly-Follows Relations with those found in the first two instances of the windows
    dfs = set([
        (case[i][activityName_key], case[i+1][activityName_key]) for case in log[:2*windowSize] for i in range(len(case)-1)
    ])
    concurrents = {
        (x,y) for x,y in dfs if (y,x) in dfs
    }
    progress = makeProgressBar(len(log), "Calculating runs ", position=progressBar_pos)
    #Calculate sequence of runs
    runs = []
    trace_to_run = {} # Mapping of traces to runs, using the currently known alpha relations. Cleared when a new concurrent pair is found

    for case in log:
        # Update Directly-Follows Relations
        dfs.update({
            (case[i][activityName_key], case[i+1][activityName_key]) for i in range(len(case)-1)
        })
        new_concurrents = {
            (x,y) for x,y in dfs if (y,x) in dfs
        }
        if new_concurrents != concurrents:
            trace_to_run = {}
            concurrents = new_concurrents

        # Calculate Run of current Case
        trace = tuple(x[activityName_key] for x in case)
        if trace in trace_to_run.keys():
            # If we have already computed this trace with these concurrency relations, use this result
            runs.append(trace_to_run[trace])
        else:
            # First time we have seen this trace since the last time the concurrency relations changed; Compute the Runs
            run = _caseToRun(case, concurrents,activityName_key=activityName_key)
            runs.append(run)
            trace_to_run[trace] = run
        progress.update()
    # Iterate over runs
    for i in range(len(log)-(2*windowSize)):
        runs1 = runs[i:i + windowSize]
        runs2 = runs[i+windowSize:i + (2*windowSize)]

        #We are testing for a changepoint at i+windowSize
        chis[i+windowSize] = chi_square(runs1, runs2)

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


def detectChangepoints_Stride(log:EventLog, windowSize:int, step_size:int=1, pvalue=0.05, activityName_key:str=xes.DEFAULT_NAME_KEY, return_pvalues:bool=False, show_progress_bar:bool=True, progressBar_pos:Optional[int]=None)->Union[List[int], Tuple[List[int], NDArray]]:
    """Apply Change Point Detection using the ProDrift Algorithm from Fast And Accurate Business Process Drift Detection. Here, for the runs calculation, we consider the alpha relations observed in the entire event log (Event relations from the "Future")

    Args:
        log (EventLog): The event log.
        windowSize (int): The window size for the sliding window algorithm.
        step_size (int, optional): The step size for the sliding window algorithm; How far to move the windows after each iteration. Defaults to 1.
        pvalue (float, optional): P-Value threshold for a pvalue to signify a change point. Defaults to 0.05. Defaults to 0.05.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        return_pvalues (bool, optional): Configures whether the computed p-values should be returned as well. Defaults to False.
        progressBar_pos (Optional[int], optional): The `pos` argument of tqdm progress bars. In which line to print the progress bar. Defaults to None.

    Returns:
        Union[List[int], Tuple[List[int], NDArray]]: A list of the detected change point indices. If selected, also a numpy array of the calculated p-values
    """  
    
    chis = []
    dfs = set([
        (case[i][activityName_key], case[i+1][activityName_key]) for case in log for i in range(len(case)-1)
    ])
    concurrents = {
        (x,y) for x,y in dfs if (y,x) in dfs
    }

    progress = None
    if show_progress_bar:
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
        safe_update_bar(progress)

    # Iterate over runs and compute p-values
    if progress is not None:
        progress.reset()
        progress.set_description("Calculating P-Values")

    i = 0
    while i < len(log)-(2*windowSize):
        runs1 = runs[i:i + windowSize]
        runs2 = runs[i+windowSize:i + (2*windowSize)]

        chi = chi_square(runs1, runs2)
        chis.append(chi)
        i += step_size
        if progress is not None:
            progress.update(min(progress.total-progress.n, step_size))

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
    if progress is not None:
        progress.close()

    # Correct changepoints (they are wrong because of initial `windowSize` indices get no value, and stride parameter)
    changepoints = [x*step_size + windowSize for x in changepoints]
    return changepoints if not return_pvalues else (changepoints, chis)

def detectChangepoints_DynamicAlpha_Stride(log:EventLog, windowSize:int, step_size:int=1, pvalue=0.05, activityName_key:str=xes.DEFAULT_NAME_KEY, return_pvalues:bool=False, progressBar_pos:Optional[int]=None)->Union[List[int], Tuple[List[int], NDArray]]:
    """Apply Change Point Detection using the ProDrift Algorithm from Fast And Accurate Business Process Drift Detection. Here, for the runs calculation, we consider the alpha relations observed in the event log inspected so far.

    This is how I understand the explanation in the paper, however upon comparing with the existing implementation (Standalone Jar-File) I find that they seem to use the alpha relations from the entire event log (Implemented in `detectChangepoints`)

    Args:
        log (EventLog): The event log.
        windowSize (int): The window size for the sliding window algorithm.
        pvalue (float, optional): P-Value threshold for a pvalue to signify a change point. Defaults to 0.05. Defaults to 0.05.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        return_pvalues (bool, optional): Configures whether the computed p-values should be returned as well. Defaults to False.
        progressBar_pos (Optional[int], optional): The `pos` argument of tqdm progress bars. In which line to print the progress bar. Defaults to None.

    Returns:
        Union[List[int], Tuple[List[int], NDArray]]: A list of the detected change point indices. If selected, also a numpy array of the calculated p-values
    """  
    
    chis = []

    # Initialize Directly-Follows Relations with those found in the first two instances of the windows
    dfs = set([
        (case[i][activityName_key], case[i+1][activityName_key]) for case in log[:2*windowSize] for i in range(len(case)-1)
    ])
    concurrents = {
        (x,y) for x,y in dfs if (y,x) in dfs
    }
    progress = makeProgressBar(len(log), "Calculating runs ", position=progressBar_pos)
    #Calculate sequence of runs
    runs = []
    trace_to_run = {} # Mapping of traces to runs, using the currently known alpha relations. Cleared when a new concurrent pair is found

    for case in log:
        # Update Directly-Follows Relations
        dfs.update({
            (case[i][activityName_key], case[i+1][activityName_key]) for i in range(len(case)-1)
        })
        new_concurrents = {
            (x,y) for x,y in dfs if (y,x) in dfs
        }
        if new_concurrents != concurrents:
            trace_to_run = {}
            concurrents = new_concurrents

        # Calculate Run of current Case
        trace = tuple(x[activityName_key] for x in case)
        if trace in trace_to_run.keys():
            # If we have already computed this trace with these concurrency relations, use this result
            runs.append(trace_to_run[trace])
        else:
            # First time we have seen this trace since the last time the concurrency relations changed; Compute the Runs
            run = _caseToRun(case, concurrents,activityName_key=activityName_key)
            runs.append(run)
            trace_to_run[trace] = run
        progress.update()

    if progress is not None:
        progress.reset(total=len(log)-(2*windowSize))
        progress.set_description("Calculating P-Values")
    # Iterate over runs
    i = 0
    while i < len(log)-(2*windowSize):
        runs1 = runs[i:i + windowSize]
        runs2 = runs[i+windowSize:i + (2*windowSize)]

        #We are testing for a changepoint at i+windowSize
        chis.append(chi_square(runs1, runs2))
        i += step_size
        if progress is not None:
            progress.update(min(progress.total-progress.n, step_size))

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

    # Correct changepoints (they are wrong because of initial `windowSize` indices get no value, and stride parameter)
    changepoints = [x*step_size + windowSize for x in changepoints]

    return changepoints if not return_pvalues else (changepoints, chis)
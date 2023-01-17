"""
    This is a close translation of the LCDD algorithm implemented [here](https://github.com/lll-lin/THUBPM/)
"""

from typing import List, Set, Tuple
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.util import xes_constants as xes

from collections import Counter


def calculate(log: EventLog, complete_window_size:int=200, detection_window_size:int=200, stable_period:int=10) -> List[int]:
    """Applies the LCDD algorithm to the Event Log. 

    "Translated" from the implementation found [here](https://github.com/lll-lin/THUBPM/blob/7d34741f487daa48dea7ef74d40198d1bd806b20/driftDetection.py#L10)

    Args:
        log (EventLog): The event log
        complete_window_size (int): The size of the complete-window (in traces) Defaults to 200 as described in the paper.
        detection_window_size (int): The size of the detection-window (in traces) Defaults to 200 as described in the paper.
        stable_period (int): The number of traces to check before a missing relation is considered missing. Defaults to 10 as described in the paper.
        step_size (int, optional): The number of traces to step after an iteration of the algorithm. Defaults to 1.

    Returns:
        List[int]: A list indices of cases where a change-point has been detected.
    """



    # First convert log to the "logdict" object they use
    logDict:List[Set[Tuple[str,str]]] = store_log_in_dict(log)

    changepoints = []

    index = 0
    windowIndex = 0
    steadyStateDSset: Set[Tuple[str, str]] = set()

    disappeared_counter = Counter()

    while index < len(logDict):
        traceDS = logDict[index]
    
        if windowIndex < complete_window_size:
            steadyStateDSset.update(traceDS)
            windowIndex += 1
        elif windowIndex < complete_window_size + detection_window_size:
            isnotCut = traceDS.issubset(steadyStateDSset)
            if not isnotCut:
                changepoints.append(index)
                index = index - 1
                windowIndex = 0
                steadyStateDSset.clear()
                disappeared_counter.clear()
            else:
                disappeared_counter.update(traceDS)

                windowIndex += 1
        else:
            isDisappearCut = isCutFromDisappear(disappeared_counter, steadyStateDSset)
            if isDisappearCut:
                startIndexW2 = index - detection_window_size
                indexChange = candidateDisappear(steadyStateDSset, disappeared_counter, logDict, startIndexW2, index, stable_period)

                changepoints.append(indexChange)
                index = indexChange - 1
                windowIndex = 0
                steadyStateDSset.clear()
                disappeared_counter.clear()
            else:
                # Move the detection window
                startIndexW2 = index - detection_window_size
                disappeared_counter.subtract(logDict[startIndexW2])

                windowIndex -= 1
                index -= 1
        index += 1
    return changepoints


def isCutFromDisappear(disappeared_counter: Counter[Tuple[str,str]], steadyStateDSset: Set[Tuple[str, str]]) -> bool:
    """Check if a change point is present due to a disappeared directly

    "Translated" from the implementation found [here](https://github.com/lll-lin/THUBPM/blob/7d34741f487daa48dea7ef74d40198d1bd806b20/driftDetection.py#L106).

    Args:
        disappeared_counter (Counter[Tuple[str,str]]): _description_
        steadyStateDSset (Set[Tuple[str, str]]): _description_

    Returns:
        bool: Is there a missing Directly Follows relation?
    """

    DSsetInTrace = set(disappeared_counter)
    differentDSset = steadyStateDSset.difference(DSsetInTrace)

    return len(differentDSset) > 1

def candidateDisappear(steadyStateDSset: Set[Tuple[str, str]], disappeared_counter: Counter[Tuple[str,str]], logDict: List[Set[Tuple[str,str]]], startIndexW2: int, index: int, maxRadius: int) -> int:
    """Find the exact location of the change point.

    "Translated" from the implementation found [here](https://github.com/lll-lin/THUBPM/blob/7d34741f487daa48dea7ef74d40198d1bd806b20/driftDetection.py#L73).

    Args:
        steadyStateDSset (Set[Tuple[str, str]]): _description_
        disappeared_counter (Count[Tuple[str,str]]): _description_
        logDict (List[Set[Tuple[str,str]]]): _description_
        startIndexW2 (int): The index in the event log where the detection window begins
        index (int): The current index that triggered a change point detection
        maxRadius (int): The stability period

    Returns:
        int: The index of the detected change point
    """

    iterIndex = startIndexW2
    storeDisappearedDS_set = set()
    trueChangePoint = startIndexW2
    radius = maxRadius

    while iterIndex < index:
        DSsetInTrace = set(disappeared_counter)
        differentDSset = steadyStateDSset.difference(DSsetInTrace)

        if len(differentDSset) > 1:
            notSame_disappeared_DS = differentDSset.difference(storeDisappearedDS_set)
            if len(notSame_disappeared_DS) > 1:
                radius = maxRadius
                trueChangePoint = iterIndex
                storeDisappearedDS_set.update(notSame_disappeared_DS)
            else:
                radius -= 1
                if radius == 0:
                    break
                
        disappeared_counter.subtract(logDict[iterIndex])
        iterIndex += 1

    return trueChangePoint

def store_log_in_dict(log: EventLog):
    """This is analogous to the `use_dict_store_log` function in the original implementation

    Args:
        log (EventLog): The event log.
    """
    return {
        index: get_directly_follows_trace(trace)
        for index, trace in enumerate(log)
    }

def get_directly_follows_trace(trace: Trace):
    """Extract all directly-follows relations in the trace.

    Args:
        trace (Trace): The trace.
    """
    return {
        (trace[i][xes.DEFAULT_NAME_KEY], trace[i+1][xes.DEFAULT_NAME_KEY])#: 1 # Map it to 1 if you want it like they use it, but i like the set of tuples more because it clearly symbolizes a relation
        for i in range(len(trace) - 1)
    }

if __name__ == "__main__":
    # Test
    import pm4py
    log = pm4py.read_xes("../../../EvaluationLogs/Ostovar/Atomic_ConditionalToSequence_output_ConditionalToSequence.xes.gz")
    result = calculate(log, complete_window_size=200, detection_window_size=200, stable_period=10)
    print(result)
from typing import Dict, List, Set, Tuple
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.util import xes_constants as xes

"""
    This is a close translation of the LCDD algorithm implemented [here](https://github.com/lll-lin/THUBPM/)
"""


def calculate(log: EventLog, complete_window_size:int=200, detection_window_size:int=200, stable_period:int=10):
    """Applies the LCDD algorithm to the Event Log. 

    "Translated" from the implementation found [here](https://github.com/lll-lin/THUBPM/blob/master/driftDetection.py)

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

    # I understand "logdict" to be a list of dictionaries, each dictionary being a case (event?), so very similar to pm4py log objects
    changepoints = []

    index = 0
    windowIndex = 0
    steadyStateDSset: Set[Tuple[str, str]] = set()
    steadyStateDSset_list = set()

    disappeared_dict = dict()
    disappear_set = set()

    while index < len(logDict):
        traceDS = logDict[index]
    
        if windowIndex < complete_window_size:
            steadyStateDSset.update(traceDS)
            # for relation in traceDS:
            #     steadyStateDSset.add(relation)
            windowIndex += 1
        elif windowIndex < complete_window_size + detection_window_size:
            isnotCut = traceDS.issubset(steadyStateDSset)
            if not isnotCut:
                changepoints.append(index)
                index = index - 1
                windowIndex = 0
                steadyStateDSset.clear()
                disappeared_dict.clear()
            else:
                for DS in traceDS:
                    if DS not in disappeared_dict.keys():
                        disappeared_dict[DS] = 1
                    else:
                        disappeared_dict[DS] = 1 + disappeared_dict[DS]
                windowIndex += 1
        else:
            isDisappearCut = isCutFromDisappear(disappeared_dict, steadyStateDSset)
            if isDisappearCut:
                startIndexW2 = index - detection_window_size
                indexChange = candidateDisappear(steadyStateDSset, disappeared_dict, logDict, startIndexW2, index, stable_period)

                changepoints.append(indexChange)
                index = indexChange - 1
                windowIndex = 0
                steadyStateDSset.clear()
                disappeared_dict.clear()
            elif not isDisappearCut: # if it comes to the else, this should always be the case
                # move w2

                startIndexW2 = index - detection_window_size
                for DS in logDict[startIndexW2]:
                    disappeared_dict[DS] = disappeared_dict[DS] - 1
                    if disappeared_dict[DS] == 0:
                        del disappeared_dict[DS]
                windowIndex -= 1
                index -= 1
        index += 1
    return changepoints


def isCutFromDisappear(disappeared_dict: Dict[Tuple[str,str], int], steadyStateDSset: Set[Tuple[str, str]]):
    """Check if a change point is present due to a disappeared directly

    Args:
        disappeared_dict (Dict[Tuple[str,str], int]): _description_
        steadyStateDSset (Set[Tuple[str, str]]): _description_

    Returns:
        _type_: _description_
    """

    DSsetInTrace = set(disappeared_dict)
    differentDSset = steadyStateDSset.difference(DSsetInTrace)

    return len(differentDSset) > 1

def candidateDisappear(steadyStateDSset: Set[Tuple[str, str]], disappeared_dict: Dict[Tuple[str,str], int], logDict: List[Set[Tuple[str,str]]], startIndexW2: int, index: int, maxRadius: int):
    iterIndex = startIndexW2
    storeDisappearedDS_set = set()
    trueChangePoint = startIndexW2
    radius = maxRadius

    while iterIndex < index:
        DSsetInTrace = set(disappeared_dict)
        differentDSset = steadyStateDSset.difference(DSsetInTrace)

        if len(differentDSset) > 1:
            notSame_disappeared_DS = differentDSset.difference(storeDisappearedDS_set)
            if len(notSame_disappeared_DS) > 1:
                radius = maxRadius
                trueChangePoint = iterIndex
                for DS in notSame_disappeared_DS:
                    storeDisappearedDS_set.add(DS)
            else:
                radius -= 1
                if radius == 0:
                    break

        for DS in logDict[iterIndex]:
            disappeared_dict[DS] = disappeared_dict[DS] - 1
            if disappeared_dict[DS] == 0:
                del disappeared_dict[DS]
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
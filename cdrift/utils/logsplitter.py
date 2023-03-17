from typing import List
from pm4py import filter_time_range
import pm4py.objects.log.util.sorting as sort
from pm4py.objects.log.obj import EventLog
import pm4py.util.xes_constants as xes 

import datetime

import math

from cdrift.utils.helpers import _dateToDatetime, makeProgressBar, safe_update_bar

def divideLogTrim(log: EventLog, isSorted:bool=False, interval:datetime.timedelta=datetime.timedelta(days=1), timestamp_key:str=xes.DEFAULT_TIMESTAMP_KEY, show_progress_bar:bool=True)-> List[EventLog]:
    """Divide the given log into sublogs where their cases are trimmed to contain only the events that occur in the corresponding Time Window

    Args:
        log (EventLog): The event log.
        isSorted (bool, optional): Flag indicating if the event log is sorted by starting timestamp. Defaults to False.
        interval (datetime.timedelta, optional): Time Interval indicating how long one sublog should be. Defaults to datetime.timedelta(days=1).
        timestamp_key (str, optional): The key for the timestamp value in the event log. Defaults to xes.DEFAULT_TIMESTAMP_KEY.
        show_progress_bar (bool, optional): Configures whether a progress bar should be shown. Defaults to True.

    Returns:
        List[EventLog]: List containing all the sublogs induced by the interval. Every Sublog contains only events that occur in this Time Window (events belonging to the same case, but different Time Window are filtered out)
    """

    if(not isSorted):
        log = sort.sort_timestamp_log(log, timestamp_key=timestamp_key)

    start = log[0][0][timestamp_key]
    end = log[-1][-1][timestamp_key]

    num_windows = math.ceil((end-start).days / interval.days)+1

    progress = None
    if show_progress_bar:
        progress = makeProgressBar(num_windows, "dividing log, completed time windows")

    res = []
    for i in range(0,num_windows):
        window_start = _dateToDatetime(start.date()) + (i*interval)
        window_end = _dateToDatetime(start.date()) + ((i+1)*interval)
        filtered = filter_time_range(log, window_start.strftime("%Y-%m-%d %H:%M:%S"), window_end.strftime("%Y-%m-%d %H:%M:%S"), mode="events")
        res.append(filtered)
        safe_update_bar(progress)
    return res

def divideLogIntersect(log: EventLog, isSorted:bool=False, interval:datetime.timedelta=datetime.timedelta(days=1), timestamp_key:str=xes.DEFAULT_TIMESTAMP_KEY, show_progress_bar:bool=True)-> List[EventLog]:
    """Divide an event log into sublogs which contain those cases which intersect with the respective time interval. For this reason, cases can be represented in multiple sublogs.

    Args:
        log (EventLog): The event log.
        isSorted (bool, optional): Flag indicating if the event log is sorted by starting timestamp. Defaults to False.
        interval (datetime.timedelta, optional): Time Interval indicating how long one sublog should be. Defaults to datetime.timedelta(days=1).
        timestamp_key (str, optional): The key for the timestamp value in the event log. Defaults to xes.DEFAULT_TIMESTAMP_KEY.
        show_progress_bar (bool, optional): Configures whether a progress bar should be shown. Defaults to True.

    Returns:
        List[EventLog]:  List containing all the sublogs induced by the interval. Every Sublog contains those cases which intersect with the corresponding Time Window
    """

    if(not isSorted):
        log = sort.sort_timestamp_log(log, timestamp_key=timestamp_key)

    start = log[0][0][timestamp_key]
    end = log[-1][-1][timestamp_key]

    num_windows = math.ceil((end-start).days / interval.days)+1

    if show_progress_bar:
        progress = makeProgressBar(num_windows, "dividing log, completed time windows")

    res = []
    for i in range(0,num_windows):
        window_start = _dateToDatetime(start.date()) + (i*interval)
        window_end = _dateToDatetime(start.date()) + ((i+1)*interval)
        filtered = filter_time_range(log, window_start.strftime("%Y-%m-%d %H:%M:%S"), window_end.strftime("%Y-%m-%d %H:%M:%S"), mode="traces_intersecting")
        res.append(filtered)
        safe_update_bar(progress)
    return res


def divideLogCaseGroups(log: EventLog, groupSize:int, isSorted:bool=False, timestamp_key:str=xes.DEFAULT_TIMESTAMP_KEY, show_progress_bar:bool=True)->List[EventLog]:
    """Split the event log into sublogs of equal size w.r.t. number of cases.

    Args:
        log (EventLog): The event log
        groupSize (int): The number of cases in each sublog.
        isSorted (bool, optional): Flag indicating if the event log is sorted by starting timestamp. Defaults to False.
        timestamp_key (str, optional): The key for the timestamp value in the event log. Defaults to xes.DEFAULT_TIMESTAMP_KEY.
        show_progress_bar (bool, optional): Configures whether a progress bar should be shown. Defaults to True.
    Returns:
        List[EventLog]: A list of the computed sublogs.
    """

    if not isSorted:
        log = sort.sort_timestamp(log, timestamp_key=timestamp_key)

    num_groups = math.ceil(len(log)/groupSize)
    progress = None
    if show_progress_bar:
        progress = makeProgressBar(num_groups, "dividing log, completed groups")

    # The following would in theory work, but does not preserve the attributes of the log itself
        #logs = [EventLog(log[i:i+groupSize]) for i in range(0, len(log), groupSize)]
    # So instead create a deep copy and then overwrite the list of traces.
    logs = []
    for i in range(num_groups):
        base_index = i*groupSize
        # sublog = log.__deepcopy__()
        # sublog._list = sublog._list[base_index:base_index+groupSize]
        # logs.append(sublog)
        sublog = log[base_index:base_index+groupSize]
        logs.append(sublog)
        if progress is not None:
            progress.update()
    return logs

def divideLogStartTime(log: EventLog, isSorted:bool=False, interval:datetime.timedelta=datetime.timedelta(days=1), timestamp_key:str=xes.DEFAULT_TIMESTAMP_KEY, show_progress_bar:bool=True)-> List[EventLog]:
    """Divide the event log into sublogs where a case is associated to the time window in which its first event began

    Args:
        log (EventLog): The event log.
        isSorted (bool, optional): Flag indicating if the event log is sorted by starting timestamp. Defaults to False.
        interval (datetime.timedelta, optional): Time Interval indicating how long one sublog should be. Defaults to datetime.timedelta(days=1).
        timestamp_key (str, optional): The key for the timestamp value in the event log. Defaults to xes.DEFAULT_TIMESTAMP_KEY.
        show_progress_bar (bool, optional): Configures whether a progress bar should be shown. Defaults to True.

    Returns:
        List[EventLog]: List containing all the sublogs induced by the interval.
    """

    #Sort the Event Log if needed
    if(not isSorted):
        log = sort.sort_timestamp_log(log, timestamp_key=timestamp_key)

    log_start = log[0][0][timestamp_key]
    log_end = log[-1][-1][timestamp_key]

    num_windows = math.ceil((log_end-log_start).days / interval.days)+1

    progress = None
    if show_progress_bar:
        progress = makeProgressBar(num_iters=len(log), message="dividing log, cases assigned to time window")

    empty_log = log.__copy__()
    empty_log._list = []
    res = [empty_log for i in range(num_windows)]

    #Iterate through the log and assign each case to a window
    for case in log:
        case_start = case[0][timestamp_key]
        since_log_start = case_start - log_start
        windows_ellapsed = since_log_start // interval
        # E.g. less than a day is ellapsed, then this division yields 0. i.e. it should be in the first window, i.e. index=0
        res[windows_ellapsed].append(case)
        safe_update_bar(progress)

    # Add metadata to each sublog, telling the time window it corresponds to
    for index, log in enumerate(res):
        if hasattr(log,'__dict__'):
            this_log_start = log_start + (index*interval)
            this_log_end = this_log_start + interval # Exclusive of course
            if not hasattr(log, 'metadata'):
                log.metadata = dict()
            log.__dict__['metadata']['start_time'] = this_log_start
            log.__dict__['metadata']['end_time'] = this_log_end
    return res
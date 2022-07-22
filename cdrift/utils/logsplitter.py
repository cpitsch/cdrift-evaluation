from typing import List
import pm4py
from pm4py.algo.filtering.log import timestamp
import pm4py.objects.log.util.sorting as sort
from pm4py.algo.filtering.log.timestamp import timestamp_filter
from pm4py.objects.log.obj import EventLog, Event, Trace
import pm4py.util.xes_constants as xes 

import datetime

import math

from cdrift.helpers import _dateToDatetime, makeProgressBar

import pkgutil

def divideLogTrim(log: EventLog, isSorted:bool=False, interval:datetime.timedelta=datetime.timedelta(days=1), timestamp_key:str=xes.DEFAULT_TIMESTAMP_KEY,  truncateStartDate:bool=False, show_progress_bar:bool=True)-> List[EventLog]:
    """
    Divides the given log into sublogs where their cases are trimmed to contain only the events that occur in the corresponding Time Window
        args:
            log: pm4py.objects.log.obj.EventLog
                The log which shall be split up into sublogs
            interval: datetime.timedelta
                Time Interval indicating how long one Window should be
            timeAttribute: str
                Which Attribute indicates time (Usually given by XES standard)
            truncateStartDate: bool
                True if the start date should be treated only up to the level of the day, otherwise exact (as exact as given by the log)
        returns:
        list[pm4py.objects.log.obj.EventLog]
        List containing all the sublogs induced by the interval. Every Sublog contains only events that occur in this Time Window (events belonging to the same case, but different Time Window are filtered out)
    """
    if(not isSorted):
        log = sort.sort_timestamp_log(log, timestamp_key=timestamp_key)

    start = log[0][0][timestamp_key]
    end = log[-1][-1][timestamp_key]

    num_windows = math.ceil((end-start).days / interval.days)+1

    progress=None
    if pkgutil.find_loader("tqdm") and show_progress_bar:
        from tqdm.auto import tqdm
        progress = tqdm(total=num_windows, desc="dividing log, completed time windows :: ")

    res = []
    for i in range(0,num_windows):
        window_start = _dateToDatetime(start.date()) + (i*interval)
        window_end = _dateToDatetime(start.date()) + ((i+1)*interval)
        res.append(timestamp_filter.apply_events(log, window_start.strftime("%Y-%m-%d %H:%M:%S"), window_end.strftime("%Y-%m-%d %H:%M:%S")))
        if progress is not None:
            progress.update()
    return res

def divideLogIntersect(log: EventLog, isSorted:bool=False, interval:datetime.timedelta=datetime.timedelta(days=1), timestamp_key:str=xes.DEFAULT_TIMESTAMP_KEY,  truncateStartDate:bool=False, show_progress_bar:bool=True)-> List[EventLog]:
    """
    Divides the given log into sublogs which contain the Cases which intersect the respectuve time interval of the sublog
        args:
            log: pm4py.objects.log.obj.EventLog
                The log which shall be split up into sublogs
            interval: datetime.timedelta
                Time Interval indicating how long one Window should be (Default 1 day)
            timestamp_key: str
                Which Attribute indicates time (Usually given by XES standard)
            truncateStartDate: bool
                True if the start date should be treated only up to the level of the day, otherwise exact (as exact as given by the log)
        returns:
        list[pm4py.objects.log.obj.EventLog]
        List containing all the sublogs induced by the interval. Every Sublog contains those cases which intersect with the corresponding Time Window
    """
    if(not isSorted):
        log = sort.sort_timestamp_log(log, timestamp_key=timestamp_key)

    start = log[0][0][timestamp_key]
    end = log[-1][-1][timestamp_key]

    num_windows = math.ceil((end-start).days / interval.days)+1

    if pkgutil.find_loader("tqdm") and show_progress_bar:
        from tqdm.auto import tqdm
        progress = tqdm(total=num_windows, desc="dividing log, completed time windows :: ")

    res = []
    for i in range(0,num_windows):
        window_start = _dateToDatetime(start.date()) + (i*interval)
        window_end = _dateToDatetime(start.date()) + ((i+1)*interval)
        filtered = timestamp_filter.filter_traces_intersecting(log, window_start.strftime("%Y-%m-%d %H:%M:%S"), window_end.strftime("%Y-%m-%d %H:%M:%S"));
        res.append(filtered)
        
        if progress is not None:
            progress.update()
    return res

# def divideLogTraces(log:EventLog, isSorted:bool=False, timestamp_key:str=xes.DEFAULT_TIMESTAMP_KEY)->List[EventLog]:
#     if not isSorted:
#         sort.sort_timestamp(log, timestamp_key=timestamp_key)
#     return list(pm4py.convert_to_event_stream(log))

def divideLogCaseGroups(log: EventLog, groupSize:int, isSorted:bool=False, timestamp_key:str=xes.DEFAULT_TIMESTAMP_KEY, show_progress_bar:bool=True)->List[EventLog]:
    """
        Splits the Event Log into Sublogs of equal Size (w.r.t the Number of cases contained within)
        
        args:
            log:pm4py.objects.log.obj.EventLog
                The Log which shall be split up
            groupSize:int
                The number of cases in each Sublog
            isSorted:bool
                Describes whether the Log still has to be sorted or not
            timestamp_key:str
                The name of the attribute which describes the Timestamp of an event (Used for sorting the EventLog)
            show_progress_bar:bool
                Whether or not a progress bar shall be shown
        returns:
            logs:list[pm4py.objects.log.obj.EventLog]
                A list containing all the Sublogs
    """
    if not isSorted:
        log = sort.sort_timestamp(log, timestamp_key=timestamp_key)

    num_groups = math.ceil(len(log)/groupSize)
    if pkgutil.find_loader("tqdm") and show_progress_bar:
        from tqdm.auto import tqdm
        progress = tqdm(total=num_groups, desc="dividing log, completed groups :: ")
    
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

def divideLogStartTime(log: EventLog, isSorted:bool=False, interval:datetime.timedelta=datetime.timedelta(days=1), timestamp_key:str=xes.DEFAULT_TIMESTAMP_KEY,  truncateStartDate:bool=False, show_progress_bar:bool=True)-> List[EventLog]:
    """
    Divides the given log into sublogs where a case is associated to the time window in which its first event began
        args:
            log: pm4py.objects.log.obj.EventLog
                The log which shall be split up into sublogs
            interval: datetime.timedelta
                Time Interval indicating how long one Window should be
            timeAttribute: str
                Which Attribute indicates time (Usually given by XES standard)
            truncateStartDate: bool
                True if the start date should be treated only up to the level of the day, otherwise exact (as exact as given by the log)
        returns:
        list[pm4py.objects.log.obj.EventLog]
        List containing all the sublogs induced by the interval. Every Sublog contains only events that occur in this Time Window (events belonging to the same case, but different Time Window are filtered out)
    """
    #Sort the Event Log if needed
    if(not isSorted):
        log = sort.sort_timestamp_log(log, timestamp_key=timestamp_key)

    log_start = log[0][0][timestamp_key]
    log_end = log[-1][-1][timestamp_key]

    num_windows = math.ceil((log_end-log_start).days / interval.days)+1

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
        if progress is not None:
            progress.update()

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
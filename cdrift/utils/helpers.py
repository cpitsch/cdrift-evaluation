"""
Helper functions for various random libraries, that have no immediate relation with EventLogs/Process Mining
"""

import ast
import datetime
from pathlib import Path
from typing import Any, List, Set, Tuple
import numpy

import networkx as nx

from pm4py.util import xes_constants as xes
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog

from tqdm.auto import tqdm

import pandas as pd
from datetime import timedelta

def _dateToDatetime(date:datetime.date)->datetime.datetime:
    """A helper function to convert a dateime.date object to a datetime.datetime object.

    Args:
        date (datetime.date): The date to convert.

    Returns:
        datetime.datetime: The converted datetime object. Hours, Minutes, Seconds, etc. all 0
    """

    return datetime.datetime(date.year, date.month, date.day)

def _getTimeDifference(time1:datetime.datetime, time2:datetime.datetime, scale:str)->float:
    """A helper function to compute the time difference between two datetime objects.

    Args:
        time1 (datetime.datetime): The first time.
        time2 (datetime.datetime): The second time.
        scale (str): The scale in which to compute the time different. Options: 'minutes', 'hours', or 'days'

    Returns:
        float: The time difference in the chosen scale.
    """    

    duration = (time2-time1).total_seconds()
    if scale == "minutes":
        duration = duration / 60
    elif scale == "hours":
        duration = duration / 3600
    elif scale == "days":
        duration = (duration / 3600) / 24
    return duration



def transitiveClosure(relation:Set[Tuple[str,str]])->Set[Tuple[str,str]]:
    """Calculate the transitive closure of a relation using networkx

    Args:
        relation (Set[Tuple[str,str]]): A set containing tuples indicating the relations.

    Returns:
        Set[Tuple[str,str]]: The transitive closure of the relation.
    """

    digraph = nx.DiGraph(list(relation))
    closure = nx.transitive_closure(digraph, reflexive=None)
    return set(closure.edges())


def transitiveReduction(relation:Set[Tuple[str,str]])->Set[Tuple[str,str]]:
    """Calculate the transitive reduction of a relation using networkx

    Args:
        relation (Set[Tuple[str,str]]): A set containing tuples indicating the relations.

    Returns:
        Set[Tuple[str,str]]: The transitive reduction of the relation.
    """

    digraph = nx.DiGraph(list(relation))
    reduction = nx.transitive_reduction(digraph)
    return set(reduction.edges())

def irreflexive(relation:Set[Tuple[str,str]])->Set[Tuple[str,str]]:
    """Make a relation irreflexive, by removing all tuples containing identical items.

    Args:
        relation (Set[Tuple[str,str]]): A set containing tuples indicating the relations.

    Returns:
        Set[Tuple[str,str]]: The irreflexive relation.
    """    

    return {(a,b) for a in relation for b in relation if a != b}

def makeProgressBar(num_iters:int=None, message:str="", position:int=None):
    """A wrapper to create a progress bar.

    Args:
        num_iters (int, optional): The number of expected iterations. Defaults to None.
        message (str, optional): The message to show on the progress bar. Defaults to "".
        position (int, optional): The `pos` argument of tqdm progress bars. In which line to print the progress bar. Defaults to None.

    Returns:
        Any: The tqdm progress bar.
    """    

    return  tqdm(total=num_iters, desc=f"{message} :: ", position=position, leave=True)

def safe_update_bar(progress_bar, amount:int=1)->None:
    if progress_bar is not None:
        progress_bar.update(amount)

def _getNumActivities(log:EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY)->int:
    """A helper function to calculate the number of activities in an event log.

    Args:
        log (EventLog): The event log.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.

    Returns:
        int: The number of distinct activities in the event log.
    """    

    return len(_getActivityNames(log, activityName_key))
    
def _getActivityNames(log:EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY)->List[str]:
    """A helper function to find the distinct activities occurring in the event log.

    Args:
        log (EventLog): The event log.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.

    Returns:
        List[str]: A list of the distinct activities in the event log.
    """    

    return sorted(list({
        event[activityName_key] for case in log for event in case
    }))

def _getActivityNames_LogList(logs:List[EventLog], activityName_key:str=xes.DEFAULT_NAME_KEY)->List[str]:
    """A helper function to find the distinct activities occurring in a list of event logs.

    Args:
        logs (List[EventLog]): The event log.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.

    Returns:
        List[str]: A list of the distinct activities in the event logs.
    """ 

    return sorted(list({
        event[activityName_key]
        for log in logs
        for case in log
        for event in case
    }))

def importLog(logpath:Any, verbose:bool=True)->EventLog:
    """A wrapper for PM4Py's log importing function.

    Args:
        logpath (Any): The path to the event log file. Only XES Files supported.
        verbose (bool, optional): Configures if a progress bar should be shown. Defaults to True.

    Returns:
        EventLog: The imported event log.
    """    

    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.SHOW_PROGRESS_BAR: verbose}
    return xes_importer.apply(logpath, variant=variant, parameters=parameters)

def getTraceLog(log:EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY)->List[Tuple[str,...]]:
    """Convert an event log to a list of traces. Traces represented as tuples of executed activities.

    Args:
        log (EventLog): The event log.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.

    Returns:
        List[Tuple[str,...]]: The Event Log represented as a list of tuples.
    """    

    return [
        tuple(
            e[activityName_key] for e in case
        )
        for case in log
    ]

def readCSV_Lists(path:Any)->pd.DataFrame:
    """A helper function to read a csv file and automatically convert the "Detected Changepoints" and "Actual Changepoints for Log" columns to Lists, and the "Duration" column to a datetime.timedelta.

    Args:
        path (Any): The path to the csv file.

    Returns:
        pd.Dataframe: The csv file as a pandas dataframe.
    """

    return pd.read_csv(path, converters={"Detected Changepoints":ast.literal_eval, "Actual Changepoints for Log":ast.literal_eval, "Duration":convertToTimedelta})

def convertToTimedelta(str_:str)->timedelta:
    """Convert a string of the form "HH:MM:SS" to a timedelta object.

    Args:
        str_ (str): The string to convert.

    Returns:
        timedelta: The corresponding timedelta object.
    """    

    hours, minutes, seconds = [float(s) for s in str_.split(':')]
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def calculateAverageAlgorithmDuration(csv:str)->float:
    """From a CSV file-path, calculate the average duration of the corresponding algorithm.

    Args:
        csv (str): The path to the CSV File containing a "Duration" Column.

    Returns:
        float: The average duration of the algorithm.
    """    

    c = pd.read_csv(csv, converters={'Duration':convertToTimedelta})
    durations = c['Duration'].tolist()

    return sum(durations,timedelta()) / len(durations)

def calcAvgDuration(df:pd.DataFrame, column:str="Duration"):
    """For a Dataframe, calculate the average duration of the corresponding algorithm.

    Args:
        df (pd.Dataframe): The Dataframe corresponding to the algorithm, containing a "Duration" Column.

    Returns:
        float: The average duration of the algorithm.
    """ 

    durations = df[column].tolist()
    return sum(durations,timedelta()) / len(durations)


def import_test_results(baseDir):
    bose_dir = Path(baseDir,"Bose/evaluation_results.csv")
    b = readCSV_Lists(bose_dir)
    b["Algorithm/Options"] = b["Algorithm/Options"].apply(
        lambda x: 
        "Bose J" if x == "Bose Average J" 
        else "Bose WC" if x == "Bose Average WC" 
        else "Bose ???"
    )

    b_j = b[b["Algorithm/Options"] == "Bose J"]
    b_wc = b[b["Algorithm/Options"] == "Bose WC"]

    martjushev_dir = Path(baseDir,"Martjushev/evaluation_results.csv")
    m = readCSV_Lists(martjushev_dir)
    m["Algorithm/Options"] = m["Algorithm/Options"].apply(
        lambda x: 
        "Martjushev J" if x == "Martjushev Recursive Bisection; Average J; p=0.55"
        else "Martjushev WC" if x == "Martjushev Recursive Bisection; Average WC; p=0.55" 
        else "Martjushev ???"
    )
    m_j = m[m["Algorithm/Options"] == "Martjushev J"]
    m_wc = m[m["Algorithm/Options"] == "Martjushev WC"]

    em_dir = Path(baseDir,"Earthmover/evaluation_results.csv")
    em = readCSV_Lists(em_dir)

    prodrift_dir = Path(baseDir, "Maaradji/evaluation_results.csv")
    prodrift = readCSV_Lists(prodrift_dir)
    prodrift["Algorithm/Options"] = prodrift["Algorithm/Options"].apply(lambda x: "ProDrift")

    pgraph_dir = Path(baseDir, "ProcessGraph/evaluation_results.csv")
    pgraphs = readCSV_Lists(pgraph_dir)
    pgraphs["Algorithm/Options"] = pgraphs["Algorithm/Options"].apply(lambda x: "Process Graphs")

    zheng_dir = Path(baseDir, "Zheng/evaluation_results.csv")
    zheng = readCSV_Lists(zheng_dir)
    zheng["Algorithm/Options"] = zheng["Algorithm/Options"].apply(lambda x: "Zheng")

    dataframes = [b_j, b_wc, m_j, m_wc, prodrift, em, pgraphs, zheng]
    return dataframes
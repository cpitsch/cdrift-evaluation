"""
Helper functions for various random libraries, that have no immediate relation with EventLogs/Process Mining
"""

import ast
import datetime
from typing import List
import numpy

import networkx as nx

from pm4py.util import xes_constants as xes
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog

from tqdm.auto import tqdm

import pandas as pd
from datetime import timedelta

def _dateToDatetime(date:datetime.date)->datetime.datetime:
    """
    args:
        date: datetime.date
            the date which shall be converted to a datetime object
    returns:
        datetime.datetime
            the same date, but as a datetime (hours and minutes all 0)
    """
    return datetime.datetime(date.year, date.month, date.day)

def _getTimeDifference(time1:datetime.datetime, time2:datetime.datetime, scale:str)->float:
    duration = (time2-time1).total_seconds()
    if scale == "minutes":
        duration = duration / 60
    elif scale == "hours":
        duration = duration / 3600
    elif scale == "days":
        duration = (duration / 3600) / 24
    return duration
    
# def _getKIndicesToMinimize(targetList:numpy.ndarray, k:int, indices:List[int]=None):
#     if indices is None:
#         indices = [index for index in range(len(targetList))]
#     # Get list of tuples index/value for all indices we regard
#     ls = [(ind, val) for (ind,val) in enumerate(targetList) if ind in indices]
#     #Sort ls by the second value, ascending
#     ls.sort(key=lambda y: y[1])
#     output = [ind for (ind,val) in ls[:k]]
#     output.sort()
#     return output



def transitiveClosure(relation:set)->set:
    """
        Returns the irreflexive transitive Closure of the given relation.
        Computed on a graph using networkx
    """
    digraph = nx.DiGraph(relation)
    closure = nx.transitive_closure(digraph, reflexive=None)
    return set(closure.edges())


def transitiveReduction(relation:set)->set:
    """
        Returns the transitive reduction of the given *acyclic* relation
        Computed on a graph using networkx
    """
    digraph = nx.DiGraph(relation)
    reduction = nx.transitive_reduction(digraph)
    return set(reduction.edges())

# def dfs_edges(relation, source, depth_limit=None):
#     activities = set([y for x in relation for y in x])
#     nodes = [source]
#     visited = set()
#     if depth_limit is None:
#         depth_limit = len(activities)
#     for start in nodes:
#         if start in visited:
#             continue
#         visited.add(start)
#         start_neighbors = [y for x,y in relation if x == start]
#         stack = [(start, depth_limit, iter(start_neighbors))]
#         while stack:
#             parent, depth_now, children = stack[-1]
#             try:
#                 child = next(children)
#                 if child not in visited:
#                     yield parent, child
#                     visited.add(child)
#                     if depth_now > 1:
#                         child_neighbors = [y for x,y in relation if x == child]
#                         stack.append((child, depth_now - 1, iter(child_neighbors)))
#             except StopIteration:
#                 stack.pop()


# def transReduction(relation:Set[Tuple[Event,Event]]):
#     tr = set()
#     nodes = [y for x in relation for y in x]
#     descendants = {}
#     # count before removing set stored in descendants
#     check_count = Counter(
#         [x for y,x in relation]
#     )
#     # check_count = dict(G.in_degree)

#     for u in nodes:
#         g_u = [y for x,y in relation if x == u] + [x for x,y in relation if y == u]
#         u_nbrs = set(g_u)
#         for v in g_u:
#             if v in u_nbrs:
#                 if v not in descendants:
#                     descendants[v] = {y for x, y in nx.dfs_edges(relation, v)}
#                 u_nbrs -= descendants[v]
#             check_count[v] -= 1
#             if check_count[v] == 0:
#                 del descendants[v]
#         tr.update((u, v) for v in u_nbrs)
#     return tr

def irreflexive(relation:set)->set:
    """Returns an irreflexive version of the Relation"""

    return {(a,b) for a in relation for b in relation if a != b}

def makeProgressBar(num_iters:int=None, message:str="", position:int=None):
        return  tqdm(total=num_iters, desc=f"{message} :: ", position=position, leave=True)

def _getNumActivities(log:EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY)->int:
    return len(_getActivityNames(log, activityName_key))
    
def _getActivityNames(log:EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY)->List[str]:
    # try:
    #     return list(log.attributes['meta_concept:named_events_total']['children'].keys())
    # except:
    # ret = set()
    # for event in pm4py.convert_to_event_stream(log):
    #     ret.add(event[activityName_key])
    # return list(ret)
    acts = set()
    for case in log:
        for event in case:
            acts.add(event[activityName_key])
    return sorted(list(acts))

def _getActivityNames_LogList(logs:List[EventLog], activityName_key:str=xes.DEFAULT_NAME_KEY)->List[str]:
    """
        Get the List of activity names 
    """
    names = set()
    for log in logs:
        s = set(_getActivityNames(log, activityName_key=activityName_key))
        names = names.union(s)
    ret = list(names)
    ret.sort()
    return ret

def importLog(logpath, verbose:bool=True):
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.SHOW_PROGRESS_BAR: verbose}
    return xes_importer.apply(logpath, variant=variant, parameters=parameters)

def getTraceLog(log:EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY):
    return [
        tuple(
            e[activityName_key] for e in case
        )
        for case in log
    ]

def readCSV_Lists(path):
    return pd.read_csv(path, converters={"Detected Changepoints":ast.literal_eval, "Actual Changepoints for Log":ast.literal_eval, "Duration":convertToTimedelta})

def convertToTimedelta(str_):
    hours, minutes, seconds = [float(s) for s in str_.split(':')]
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def calculateAverageAlgorithmDuration(csv:str):
    c = pd.read_csv(csv, converters={'Duration':convertToTimedelta})
    durations = c['Duration'].tolist()

    return sum(durations,timedelta()) / len(durations)

def calcAvgDuration(df:pd.DataFrame):
    durations = df['Duration'].tolist()
    return sum(durations,timedelta()) / len(durations)
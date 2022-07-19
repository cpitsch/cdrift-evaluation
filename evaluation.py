###############################################
############ Evaluation Metrics ###############
###############################################

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from helpers import calcAvgDuration
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD

# The calculation of the F1 score as described in "Change Point Detection and Dealing with Gradual and Multi-Order Dynamics in Process Mining" by Martjushev, Bose, Van Der Aalst
def F1_Score(lag:int, detected:List[int], known: List[int], zero_division="warn", verbose:bool=False):
    """
        Calculates the F1 Score for a Changepoint Detection Result

        - Considering a known changepoint at timepoint t:
            - A True Positive is when a changepoint is detected within [t-`lag`, t+`lag`]
            - A False Negative is when no changepoint is detected in this window around a known changepoint
            - A False Positive is when there is no known changepoint in a window of `lag` around the detected one
        - From this the F1-Score is calculated as (2&middot;precision&middot;recall) / (precision+recall)

        params:
            lag:int
                The window around an actual changepoint in which a detection is considered correct. i.e. if a changepoint occurs at time t, any prediction in [t-l, t+l] is considered a True Positive. It is assumed that the windows around a known change point induced by the lag are non-overlapping
            detected:List[int]
                A list of indices where the Changepoint Algorithm detected a Changepoint
            known:List[int]
                A list of indices where we know that a Changepoint is located
            zero_division:any default: 0
                The return value if the calculation of precision/recall/F1 divides by 0 e.g. 0,1,NaN,.. if set to warn, 0 is returned and a warning is printed out
        returns:
            f1:float in [0,1]
                The F1-Score of this Prediction
    """
    TP = 0
    #FP = 0

    #Find True and False positives 
    for point in known:
        window_begin = point-lag
        window_end = point+lag
        # window = range(point-lag,point+lag+1)
        if any([window_begin <= d and d <= window_end for d in detected]):
            TP += 1 # At least one changepoint was detected close enough to this one
            #TODO: Caution, it is possible to count 1 detected changepoint for multiple change points, maybe the best would be to calculate the optimal Detected -> Known mapping with an ILP
    # for point in detected:
    #     window = [x for x in known if point-lag < x and x < point+lag] # All the known changepoints that are in the window around it
    #     if len(window) == 0:
    #         #A changepoint was detected where none was
    #         FP += 1
    #     else:
    #         TP += 1 # Only incrementing by 1 since a detected changepoint can only correspond to maximally 1 actual changepoint


    if len(detected) == 0 or len(known) == 0: # Divide by zero
        if zero_division == "warn" and verbose:
            print("Calculation of F1-Score divided by 0 at calculation of Precision or Recall due to no Changepoints Detected or no Changepoints Existing.")
        return zero_division
    else:
        precision = TP / len(detected)
        recall = TP / len(known)
    if precision + recall == 0: # Divide by zero
        if zero_division == "warn" and verbose:
            print("Calculation of F1-Score divided by 0 at calculation of F1 Score because Precision and Recall are both 0.")
        return zero_division
    else:
        f1 = (2*precision*recall)/(precision+recall)
        return f1

# Alias for F1_Score
f1 = F1_Score


def getTP_FP(lag:int, detected:List[int], known:List[int]):
    TP = 0
    FP = 0
    for point in known:
        window_begin = point-lag
        window_end = point+lag
        if any([window_begin <= d and d <= window_end for d in detected]):
            TP += 1
    FP = len(detected) - TP
    return (TP,FP)

def calcTPR_FPR(lag:int, detected:List[int], known:List[int], num_possible_negatives:int=None):
    TP, FP = getTP_FP(lag, detected, known)
    P = len(known)
    TPR = TP/P
    # So many Negative points it wouldnt make sense....
    FPR = FP/num_possible_negatives if num_possible_negatives is not None else np.NaN
    return (TPR, FPR)

def calcPrecisionRecall(lag:int, detected:List[int], known:List[int], zero_division=np.NaN):
    TP, FP = getTP_FP(lag, detected, known)
    if(len(detected) > 0):
        precision = TP/len(detected)
    else:
        precision = zero_division
    if(len(known) > 0):
        recall = TP/len(known)
    else:
        recall = zero_division
    return (precision, recall)

def _assign_changepoints(detected_changepoints: List[int], actual_changepoints:List[int], lag_window:int=200) -> List[Tuple[int,int]]:
    """Assigns detected changepoints to actual changepoints using a LP.
        With restrictions: 
            - Detected point must be within lag_window of actual point. 
            - Detected point can only be assigned to one actual point.
            - Every actual point can have at most one detected point assigned. 

        This is done by first optimizing for the number of assignments, finding how many detected change points could be assigned, without minimizing the \
        total lag. Then, the LP is solved again, minimizing the sum of squared lags, while keeping the number of assignments as high as possible.

    Args:
        detected_changepoints (List[int]): List of locations of detected changepoints.
        actual_changepoints (List[int]): List of locations of actual changepoints.
        lag_window (int, optional): How close must a detected change point be to an actual changepoint to be a true positive. Defaults to 200.

    Examples:
        >>> detected_changepoints = [1050, 934, 2100]
        >>> actual_changepoints = [1000,1149,2000]
        >>> _assign_changepoints(detected_changepoints, actual_changepoints, lag_window=200)
        >>> [(1050, 1149), (934, 1000), (2100, 2000)]
        >>> # Notice how the actual changepoint 1000 gets a further detected changepoint to allow 1149 to also get a changepoint assigned

    Returns:
        List[Tuple[int,int]]: List of tuples of (detected_changepoint, actual_changepoint) assignments
    """

    def buildProb_NoObjective(sense):
        """
            Builds the optimization problem, minus the Objective function. Makes multi-objective optimization simple
        """
        prob = LpProblem("Changepoint_Assignment", sense)

        # Create a variable for each pair of detected and actual changepoints
        vars = LpVariable.dicts("x", (detected_changepoints, actual_changepoints), 0, 1, LpBinary) # Assign detected changepoint dp to actual changepoint ap?
        
        # Flatten vars into dict of tuples of keys
        x = {
            (dc, ap): vars[dc][ap] for dc in detected_changepoints for ap in actual_changepoints
        }

        ####### Constraints #########
        # As many assigments as possible
        # prob += (
        #     lpSum(x[dp, ap] for dp in detected_changepoints for ap in actual_changepoints) == tp,
        #     "As_Many_As_Possible"
        # )

        # Only assign at most one changepoint to each actual changepoint
        for ap in actual_changepoints:
            prob += (
                lpSum(x[dp, ap] for dp in detected_changepoints) <= 1,
                f"Only_One_Changepoint_Per_Actual_Changepoint : {ap}"
            )
        # Each detected changepoint is assigned to at most one actual changepoint
        for dp in detected_changepoints:
            prob += (
                lpSum(x[dp, ap] for ap in actual_changepoints) <= 1,
                f"Only_One_Actual_Changepoint_Per_Detected_Changepoint : {dp}"
            )
        # Distance between chosen changepoints must be within lag window
        for dp in detected_changepoints:
            for ap in actual_changepoints:
                prob += (
                    x[dp, ap] * abs(dp - ap) <= lag_window,
                    f"Distance_Within_Lag_Window : {dp}_{ap}"
                )
        return prob, x

    solver = PULP_CBC_CMD(msg=0)

    ### Multi-Objective Optimization: First maximize number of assignments to find out the best True Positive number that can be achieved
    # Find the largest number of change points:
    prob1, prob1_vars = buildProb_NoObjective(LpMaximize)
    prob1 += (
        lpSum(
            # Minimize the squared distance between assigned changepoints
            prob1_vars[dp, ap]
            for dp in detected_changepoints for ap in actual_changepoints
        ),
        "Maximize number of assignments"
    )
    prob1.solve(solver)
    # Calculate number of TP
    num_tp = len([
        (dp, ap)
        for dp in detected_changepoints for ap in actual_changepoints
        if prob1_vars[dp, ap].varValue == 1
    ])


    ### Multi-Objective Optimization: Now minimize the squared distance between assigned changepoints, using this maximal number of assignments
    # Use this number as the number of assignments for second optimization
    prob2, prob2_vars = buildProb_NoObjective(LpMinimize)
    prob2 += (
        lpSum(
            # Minimize the squared distance between assigned changepoints
            prob2_vars[dp, ap] * pow(dp - ap,2)
            for dp in detected_changepoints for ap in actual_changepoints
        ),
        "Squared_Distances"
    )

    # Number of assignments is the number of true positives we found in the first optimization
    prob2 += (
        lpSum(
            prob2_vars[dp, ap]
            for dp in detected_changepoints for ap in actual_changepoints
        ) == num_tp,
        "Maximize Number of Assignments"
    )
    prob2.solve(solver)
    return [
        (dp, ap)
        for dp in detected_changepoints for ap in actual_changepoints
        if prob2_vars[dp, ap].varValue == 1
    ]

def get_avg_lag(detected_changepoints:List[int], actual_changepoints:List[int], lag_window:int=200):
    """Calculates the average lag between detected and actual changepoints (false positives do not affect this metric!)

    Args:
        detected_changepoints (List[int]): Locations of detected changepoints
        actual_changepoints (List[int]): Locations of actual (known) changepoints
        lag_window (int, optional): How close must a detected change point be to an actual changepoint to be a true positive. Defaults to 200.

    Examples:
        >>> detected_changepoints = [1050, 934, 2100]
        >>> actual_changepoints = [1000,1149,2000]
        >>> get_avg_lag(detected_changepoints, actual_changepoints, lag_window=200)
        >>> 88.33333333333333

    Returns:
        float: the average distance between detected changepoints and the actual changepoint they get assigned to
    """
    assignments = _assign_changepoints(detected_changepoints, actual_changepoints, lag_window=lag_window)
    avg_lag = 0
    for (dc,ap) in assignments:
        avg_lag += abs(dc-ap)
    try:
        return avg_lag/len(assignments)
    except ZeroDivisionError:
        return np.nan;


def getROCData(lag:int, df:pd.DataFrame, undefined_equals=0):
    """
        Returns a list of points, as tuples of Recall (TPR) and Precision (Cannot do FPR because Negatives are not really defined/there are so many)
    """
    groups = df.groupby("Window Size")
    points = []
    for win, win_df in groups:
        for idx, row in win_df.iterrows():
            precisions = []
            recalls = []
            prec, rec = calcPrecisionRecall(lag, row["Detected Changepoints"], row["Actual Changepoints for Log"], zero_division=undefined_equals)
            precisions.append(prec)
            recalls.append(rec)
        # Average precision and recall over all logs for this 
        points.append(  (np.mean(recalls), np.mean(precisions))   )
    return points

def plotROC(lag, df:pd.DataFrame, undefined_equals=0):
    import matplotlib.pyplot as plt
    dat = getROCData(lag,df,undefined_equals)
    recalls, precisions = list(zip(*dat))
    print(precisions)
    print(recalls)
    plt.plot(precisions, recalls) # x is precisions, y is recalls
    plt.ylim(-0.01,1.01)
    plt.xlim(-0.01,1.01)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.show()

def calcScatterData(dfs:List[pd.DataFrame], handle_nan_as=np.nan):
    """Calculates the points for a scatterplot, plotting Calculation Time against the achieved F1-Score

    Args:
        dfs (List[pd.DataFrame]): The dataframe of each approach to be plotted
        handle_nan_as (_type_, optional): How should an undefined F1-Score (Division by zero) be handled? Defaults to np.nan.

    Returns:
        List[Dict[]]: A list of points, represented as dictionaries with keys "duration", "f1", and "name"
    """
    points:List[Dict] = []
    for df in dfs:
        avgDur = calcAvgDuration(df).seconds
        avgF1 = df["F1-Score"].fillna(handle_nan_as, inplace=False).mean()
        points.append({
            "duration": avgDur/60,
            "f1": avgF1,
            "name": df.iloc[-1]["Algorithm/Options"]
        })
    return points


def plotScatterData(points: List[Dict], path="../scatter_fig.png", _format="png"):
    """Plots a list of points (dictionaries in a scatterplot and saves it)

    Args:
        points (List[Dict]): A list of points, represented as dictionaries with keys "duration", "f1", and "name"
        path (str, optional): Path where figure should be saved. Defaults to "../scatter_fig.png".
        _format (str, optional): Format of the saved figure. Defaults to "png".
    """    
    fig,ax = plt.subplots()
    for point in points:
        ax.scatter(y=[point["f1"]], x=[point["duration"]], label=point["name"])
    ax.set_ylabel("Mean F1-Score")
    ax.set_ylim(0,1)
    ax.set_xlabel("Mean Duration (Minutes)")
    ax.legend(bbox_to_anchor=(1,1), loc="upper left")
    ax.grid(True)
    plt.savefig(path, bbox_inches='tight', format=_format)
    plt.show()     

def scatterF1_Duration(dfs:List[pd.DataFrame], handle_nan_as=np.nan, path="../scatter_fig.png", _format="png"):
    """Plots a scatterplot of the F1-Score against the duration of the algorithm, given a list of dataframes for each approach

    Args:
        dfs (List[pd.DataFrame]): List of dataframes for each approach
        handle_nan_as (Any, optional): How to handle an undefined F1-Score (Division by zero). Defaults to np.nan.
        path (str, optional): Path where to save the figure. Defaults to "../scatter_fig.png".
        _format (str, optional): Format of the Figure. Defaults to "png".
    """
    points = calcScatterData(dfs, handle_nan_as)
    plotScatterData(points)



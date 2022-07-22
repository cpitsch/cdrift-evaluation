###############################################
############ Evaluation Metrics ###############
###############################################

from typing import Dict, List, Tuple, Union, Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD

from cdrift.utils.helpers import calcAvgDuration

def getTP_FP(lag:int, detected:List[int], known:List[int])-> Tuple[int,int]:
    """Returns the number of true and false positives, using assign_changepoints to calculate the assignments of detected change point to actual change point.

    Args:
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        detected (List[int]): List of indices of detected change point locations.
        known (List[int]): The ground truth; List of indices of actual change points.

    Returns:
        Tuple[int,int]: Tuple of: (true positives, false positives)
    """
    assignments = assign_changepoints(detected, known, lag_window=lag)
    TP = len(assignments) # Every assignment is a True Positive, and every detected point is assigned at most once
    FP = len(detected) - TP
    return (TP,FP)

def calcPrecisionRecall(lag:int, detected:List[int], known:List[int], zero_division=np.NaN)->Tuple[float, float]:
    """Calculates the precision and recall, using `get_TP_FP` for True positives and False Negatives, which uses assign_changepoints to calculate the assignments of detected change point to actual change point.

    Args:
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        detected (List[int]): A list of indices of detected change point locations.
        known (List[int]): The ground truth; List of indices of actual change points.
        zero_division (Any, optional): The value to yield for precision/recall when a zero-division is encountered. Defaults to np.NaN.

    Returns:
        Tuple[Union[float,np.NaN], Union[float,np.NaN]]: _description_
    """

    TP, _ = getTP_FP(lag, detected, known)
    if(len(detected) > 0):
        precision = TP/len(detected)
    else:
        precision = zero_division
    if(len(known) > 0):
        recall = TP/len(known)
    else:
        recall = zero_division
    return (precision, recall)

def F1_Score(lag:int, detected:List[int], known: List[int], zero_division="warn", verbose:bool=False):
    """ Calculates the F1 Score for a Changepoint Detection Result

        - Considering a known changepoint at timepoint t:
            - A True Positive is when a changepoint is detected within [t-`lag`, t+`lag`]
            - A False Negative is when no changepoint is detected in this window around a known changepoint
            - A False Positive is when there is no known changepoint in a window of `lag` around the detected one
            - Note: Only one detected change point can be a TP for a given known changepoint, and vice versa. The assignment of detected change points to actual change points is done using a Linear Program (see assign_changepoints)
        - From this the F1-Score is calculated as (2&middot;precision&middot;recall) / (precision+recall)

    Args:
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        detected (List[int]) : A list of indices of detected change point locations.
        known (List[int]): The ground truth; List of indices of actual change points.
        zero_division (str, optional): The return value if the calculation of precision/recall/F1 divides by 0. If set to "warn", 0 is returned and a warning is printed out. Defaults to "warn".
        verbose (bool, optional): If verbose, warning messages are printed when a zero-division is encountered. Defaults to False.

    Returns:
        float: The F1-Score corresponding to the given prediction.
    """

    TP, _ = getTP_FP(lag, detected, known)

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

def calcTPR_FPR(lag:int, detected:List[int], known:List[int], num_possible_negatives:int=None)->Tuple[float, float]:
    """Calculates the True-Positive-Rate and the False-Positive-Rate for a given detection. 

    Args:
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        detected (List[int]): A list of indices of detected change point locations.
        known (List[int]): The ground truth; List of indices of actual change points.
        num_possible_negatives (int, optional): The number of possible negatives. In theory, this is `len(log)-len(known)`, however this number is way too large. Defaults to None.

    Returns:
        Tuple[Union[float, np.NaN], Union[float,np.NaN]]: A tuple of: (True-Positive-Rate, False-Positive-Rate)
    """

    TP, FP = getTP_FP(lag, detected, known)
    P = len(known)
    TPR = TP/P
    # So many Negative points it wouldnt make sense....
    FPR = FP/num_possible_negatives if num_possible_negatives is not None else np.NaN
    return (TPR, FPR)


def assign_changepoints(detected_changepoints: List[int], actual_changepoints:List[int], lag_window:int=200) -> List[Tuple[int,int]]:
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
        >>> assign_changepoints(detected_changepoints, actual_changepoints, lag_window=200)
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

def get_avg_lag(detected_changepoints:List[int], actual_changepoints:List[int], lag_window:int=200)->float:
    """Calculates the average lag between detected and actual changepoints (Caution: false positives do not affect this metric!)

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
    assignments = assign_changepoints(detected_changepoints, actual_changepoints, lag_window=lag_window)
    avg_lag = 0
    for (dc,ap) in assignments:
        avg_lag += abs(dc-ap)
    try:
        return avg_lag/len(assignments)
    except ZeroDivisionError:
        return np.nan;


def getROCData(lag:int, df:pd.DataFrame, undefined_equals=0)->List[Tuple[float,float]]:
    """Returns a list of points, as tuples of Recall (TPR) and Precision (Cannot do FPR because negatives are not really defined for concept drift detection/negatives are practically the entire log (`len(log)-len(detected)`))

    Args:
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        df (pd.DataFrame): The Dataframe containing the detection results of the approach
        undefined_equals (int, optional): The value to assign to undefined F1-Scores. Defaults to 0.

    Returns:
        List[Tuple[float,float]]: A list of the mean precision and recall values for each Window Size found in the dataframe.
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

def plotROC(lag, df:pd.DataFrame, undefined_equals=0)->None:
    """Plot an ROC Curve (using precision and recall) for the given dataframe and a given lag value for precision and recall evaluation

    Args:
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        df (pd.DataFrame): The Dataframe containing the detection results of the approach
        undefined_equals (int, optional): The value to assign to undefined F1-Scores. Defaults to 0.
    """    
    dat = getROCData(lag,df,undefined_equals)
    recalls, precisions = list(zip(*dat))
    plt.plot(precisions, recalls) # x is precisions, y is recalls
    plt.ylim(-0.01,1.01)
    plt.xlim(-0.01,1.01)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.show()

def calcScatterData(dfs:List[pd.DataFrame], handle_nan_as=np.nan)->List[Dict]:
    """Calculates the points for a scatterplot, plotting Calculation Time against the achieved F1-Score

    Args:
        dfs (List[pd.DataFrame]): A list of dataframes (each one corresponding to an approach) to calculate the scatterplot data for.
        handle_nan_as (Any, optional): How should an undefined F1-Score (Division by zero) be handled? Defaults to np.nan.

    Returns:
        List[Dict]: A list of points, represented as dictionaries with keys "duration", "f1", and "name"
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


def plotScatterData(points: List[Dict], path="../scatter_fig.png", _format="png")->None:
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

def scatterF1_Duration(dfs:List[pd.DataFrame], handle_nan_as=np.nan, path="../scatter_fig.png", _format="png")->None:
    """Plots a scatterplot of the F1-Score against the duration of the algorithm, given a list of dataframes for each approach

    Args:
        dfs (List[pd.DataFrame]): List of dataframes for each approach
        handle_nan_as (Any, optional): How to handle an undefined F1-Score (Division by zero). Defaults to np.nan.
        path (str, optional): Path where to save the figure. Defaults to "../scatter_fig.png".
        _format (str, optional): Format of the Figure. Defaults to "png".
    """
    points = calcScatterData(dfs, handle_nan_as)
    plotScatterData(points, path=path, _format=_format)



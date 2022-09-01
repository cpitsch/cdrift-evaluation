###############################################
############ Evaluation Metrics ###############
###############################################

from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD

from cdrift.utils.helpers import calcAvgDuration, convertToTimedelta

def getTP_FP(detected:List[int], known:List[int], lag:int)-> Tuple[int,int]:
    """Returns the number of true and false positives, using assign_changepoints to calculate the assignments of detected change point to actual change point.

    Args:
        detected (List[int]): List of indices of detected change point locations.
        known (List[int]): The ground truth; List of indices of actual change points.
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.

    Returns:
        Tuple[int,int]: Tuple of: (true positives, false positives)
    """
    assignments = assign_changepoints(detected, known, lag_window=lag)
    TP = len(assignments) # Every assignment is a True Positive, and every detected point is assigned at most once
    FP = len(detected) - TP
    return (TP,FP)

def calcPrecisionRecall(detected:List[int], known:List[int], lag:int, zero_division=np.NaN)->Tuple[float, float]:
    """Calculates the precision and recall, using `get_TP_FP` for True positives and False Negatives, which uses assign_changepoints to calculate the assignments of detected change point to actual change point.

    Args:
        detected (List[int]): A list of indices of detected change point locations.
        known (List[int]): The ground truth; List of indices of actual change points.
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        zero_division (Any, optional): The value to yield for precision/recall when a zero-division is encountered. Defaults to np.NaN.

    Returns:
        Tuple[Union[float,np.NaN], Union[float,np.NaN]]: _description_
    """

    TP, _ = getTP_FP(detected, known, lag)
    if(len(detected) > 0):
        precision = TP/len(detected)
    else:
        precision = zero_division
    if(len(known) > 0):
        recall = TP/len(known)
    else:
        recall = zero_division
    return (precision, recall)

def F1_Score(detected:List[int], known: List[int], lag:int, zero_division="warn", verbose:bool=False):
    """ Calculates the F1 Score for a Changepoint Detection Result

    - Considering a known changepoint at timepoint t:
        - A True Positive is when a changepoint is detected within [t-`lag`, t+`lag`]
        - A False Negative is when no changepoint is detected in this window around a known changepoint
        - A False Positive is when there is no known changepoint in a window of `lag` around the detected one
        - Note: Only one detected change point can be a TP for a given known changepoint, and vice versa. The assignment of detected change points to actual change points is done using a Linear Program (see assign_changepoints)
    - From this the F1-Score is calculated as (2&middot;precision&middot;recall) / (precision+recall)

    Args:
        detected (List[int]) : A list of indices of detected change point locations.
        known (List[int]): The ground truth; List of indices of actual change points.
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        zero_division (str, optional): The return value if the calculation of precision/recall/F1 divides by 0. If set to "warn", 0 is returned and a warning is printed out. Defaults to "warn".
        verbose (bool, optional): If verbose, warning messages are printed when a zero-division is encountered. Defaults to False.

    Returns:
        float: The F1-Score corresponding to the given prediction.
    """

    TP, _ = getTP_FP(detected, known, lag)

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

def calcTPR_FPR(detected:List[int], known:List[int], lag:int, num_possible_negatives:int=None)->Tuple[float, float]:
    """Calculates the True-Positive-Rate and the False-Positive-Rate for a given detection. 

    Args:
        detected (List[int]): A list of indices of detected change point locations.
        known (List[int]): The ground truth; List of indices of actual change points.
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        num_possible_negatives (int, optional): The number of possible negatives. In theory, this is `len(log)-len(known)`, however this number is way too large. Defaults to None.

    Returns:
        Tuple[Union[float, np.NaN], Union[float,np.NaN]]: A tuple of: (True-Positive-Rate, False-Positive-Rate)
    """

    TP, FP = getTP_FP(detected, known, lag)
    P = len(known)
    TPR = TP/P
    # So many Negative points it wouldnt make sense....
    FPR = FP/num_possible_negatives if num_possible_negatives is not None else np.NaN
    return (TPR, FPR)


def assign_changepoints(detected_changepoints: List[int], actual_changepoints:List[int], lag_window:int=200) -> List[Tuple[int,int]]:
    """Assigns detected changepoints to actual changepoints using a LP.
    With restrictions: 

    - Detected point must be within `lag_window` of actual point. 
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

def get_avg_lag(detected_changepoints:List[int], actual_changepoints:List[int], lag:int=200)->float:
    """Calculates the average lag between detected and actual changepoints (Caution: false positives do not affect this metric!)

    Args:
        detected_changepoints (List[int]): Locations of detected changepoints
        actual_changepoints (List[int]): Locations of actual (known) changepoints
        lag (int, optional): How close must a detected change point be to an actual changepoint to be a true positive. Defaults to 200.

    Examples:
    >>> detected_changepoints = [1050, 934, 2100]
    >>> actual_changepoints = [1000,1149,2000]
    >>> get_avg_lag(detected_changepoints, actual_changepoints, lag=200)
    >>> 88.33333333333333

    Returns:
        float: the average distance between detected changepoints and the actual changepoint they get assigned to
    """
    assignments = assign_changepoints(detected_changepoints, actual_changepoints, lag_window=lag)
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


def scatter_f1_duration(dfs:List[pd.DataFrame]):
    """Make a Scatterplot of the F1 Score versus the Duration of the approaches with the pareto front. Very specific method to be used on the results from running `testAll_MP.py`.

    Args:
        dfs (List[pd.DataFrame]): A list of DataFrames of evaluation results. Algorithms such as Bose, containing two different "algorithms" (J Measure and Window Count) should be split into two DataFrames.

    Returns:
        plt.figure: The figure object containing the plot.
    """    

    markers = {
        "Bose J": "^", # triangle_up
        "Bose WC": "^", # triangle_up
        "Martjushev J": "s", # square
        "Martjushev WC": "s", #square
        "Martjushev ADWIN J": "s", # square
        "Martjushev ADWIN WC": "s", #square
    }
    colors = {
        "Bose J": "#add8e6", # Light Blue
        "Bose WC": "#000080", # Navy Blue
        "Martjushev J": "#32cd32", # Navy Green
        "Martjushev WC": "#006400", # Dark Green
        "Martjushev ADWIN J": "#32cd32", # Navy Green
        "Martjushev ADWIN WC": "#006400", # Dark Green
        "ProDrift": "#ff0000", # Red
        "Earth Mover's Distance": "#ffa500", # Orange
        "Process Graphs": "#ddcd10", # Some dark, rich yellow
        "Zheng": "#800080" # Purple
        # The rest is just whatever it wants to give them
    }

    for df in dfs:
        # Turn Duration column into minutes
        df["Duration"] = df["Duration"].apply(lambda x: x.seconds/60)
        df.rename(columns={"Duration": "Duration (Minutes)"}, inplace=True)

    fig = scatter_pareto_front(dfs,  x="Duration (Minutes)", y="F1-Score", figsize=(8,5), lower_is_better_dimensions = ["Duration (Minutes)"], colors=colors, markers=markers)
    plt.ylim(-0.02, 1)
    return fig

def _getNameFromDataframe(df, name_column="Algorithm/Options"):
        # Return the Algorithm/Options of the first entry
        if name_column in df.columns:
            return df.iloc[-1][name_column]
        elif "Algorithm/Options" in df:
            return df.iloc[-1]["Algorithm/Options"]
        else:
            return df.iloc[-1]["Algorithm"]

def _column_is_time(column, df):
    # Check if the column has a time data type
    return pd.core.dtypes.common.is_datetime_or_timedelta_dtype(df[column])

def calculate_scatter_data(dfs: List[pd.DataFrame], dimensions:List[str]=["F1-Score","Duration"], handle_nan_as=0)->List[Tuple[str, Dict[str, Any]]]:
    """Calculate the points for a Scatterplot. Uses the mean value of each column as the corresponding value.

    Args:
        dfs (List[pd.DataFrame]): List of Dataframes for the Scatterplot. The "Algorithm/Options" column will be used to infer the name of the respective approach.
        dimensions (List[str], optional): List of Dimensions to consider when extracting the points. Defaults to ["F1-Score","Duration"].

    Returns:
        List[Tuple[str, Dict[str, Any]]]: List of points for the scatter plot. Points represented as a tuple of the *name* of the algorithm and a dictionary with a value for each dimension.
    """    
    # Each approach correspnds to a point that is the mean of each dimension
    points = []
    for df in dfs:
        name = _getNameFromDataframe(df)
        point = {dim: df[dim].fillna(handle_nan_as, inplace=False).mean() if not _column_is_time(dim, df) else calcAvgDuration(df, dim).seconds for dim in dimensions}
        points.append((name, point))
    return points

def get_pareto_optimal_points(points: List[Dict[str, Any]], lower_is_better_dimensions: List[str] = [])->List[Tuple[str, Dict[str, Any]]]:
    """Get the pareto optimal points from a list of points. A pareto optimal point is a point for which no other exists that is better in every dimension.

    Args:
        points (List[Dict[str, Any]]): The list of points, formatted as tuples: name of the algorithm, and dictionary of a value for each dimension.
        lower_is_better_dimensions (List[str], optional): A List contianing all Dimensions where a *lower* value is better. Examples: Duration, Memory Consumption. Defaults to [].

    Returns:
        List[Tuple[str, Dict[str, Any]]]: A list of the points which are pareto optimal.
    """

    return [
        (name, point)
        for idx, (name, point) in enumerate(points)
        # If pareto-optimal
        if all( # For every other point:
            any( # This point is better or equal in at least one dimension --> the point is not better in *every* dimension
                other_point[dim] <= point[dim] if not dim in lower_is_better_dimensions else other_point[dim] >= point[dim]
                for dim in point.keys()
            )
            for _,other_point in [p for i,p in enumerate(points) if i != idx] # All other points
        )
    ]

def scatter_pareto_front(dfs:List[pd.DataFrame],  x:str="Duration", y:str="F1-Score", handle_nan_as:Any=0, figsize:Tuple[int,int]=(8,5), lower_is_better_dimensions: List[str] = [], markers:Dict={}, colors:Dict={})->plt.figure:
    """Plot the approaches as a scatterplot, considering dimensions `x` and `y`. Additionally, the Pareto front is indicated in the plot. Only two dimensions are supported. 

    Args:
        dfs (List[pd.DataFrame]): List of Dataframes for the Scatterplot. The "Algorithm/Options" column will be used to infer the name of the respective approach.
        x (str, optional): The Dimension to consider for the X-Axis. Defaults to "Duration".
        y (str, optional): The Dimension to consider for the Y-Axis. Defaults to "F1-Score".
        handle_nan_as (Any, optional): How to handle NaN values in computing the mean. If NaN is chosen, the NaN values will be ignored. Defaults to 0.
        figsize (Tuple[str,str], optional): The figsize parameter to be used for Figure Creation. Defaults to (8,5).
        lower_is_better_dimensions (List[str], optional): A List contianing all Dimensions where a *lower* value is better. Examples: Duration, Memory Consumption. Defaults to [].
        markers (Dict, optional): A dictionary mapping Approach names to markers to be used for Matplotlib Scatterplot. If not specified, Matplotlib will decide. Defaults to {}.
        colors (Dict, optional): A dictionary mapping Approach names to markers to be used for Matplotlib Scatterplot. If not specified, Matplotlib will decide. Defaults to {}.

    Returns:
        plt.figure: The figure object containing the plot.
    """    


    # First get the pareto points
    data = calculate_scatter_data(dfs, [x,y], handle_nan_as)
    pareto_points = get_pareto_optimal_points(data, lower_is_better_dimensions=lower_is_better_dimensions)

    fig = plt.figure(figsize=figsize)
    # Plot the scatterplot
    for name, point in data:
        color = colors.get(name, None)
        marker = markers.get(name, None)
        plt.scatter(y=point[y], x=point[x], label=name, color=color, marker=marker)

    # Plot the pareto front
    pareto_points = sorted(pareto_points, key=lambda point: point[1][x])
    plt.plot([0,pareto_points[0][1][x]], [0, pareto_points[0][1][y]], "red", linestyle=":", zorder=1)
    # Plot a line between the neighboring pareto optimal points
    for idx, (_, point) in enumerate(pareto_points[:-1]):
        _,point2 = pareto_points[idx+1]
        xs = [point[x], point2[x]]
        ys = [point[y], point2[y]]
        plt.plot(xs, ys, "red", linestyle=":", zorder=1)
    # Plot a line from the last point horizontally off the canvas
    xlim_before = plt.xlim()
    ub = plt.xlim()[1]*2
    plt.plot([pareto_points[-1][1][x], ub], [pareto_points[-1][1][y],pareto_points[-1][1][y]], "red", linestyle=":", zorder=1)
    plt.xlim(xlim_before)

    plt.xlabel(f"Mean {x}")
    plt.ylabel(f"Mean {y}")
    plt.grid(True)

    plt.legend(bbox_to_anchor=(1,1), loc="upper left", title="Algorithm")
    return fig
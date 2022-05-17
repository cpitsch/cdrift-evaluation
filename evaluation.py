###############################################
############ Evaluation Metrics ###############
###############################################

from typing import List
import numpy as np
import pandas as pd

from helpers import calcAvgDuration
import matplotlib.pyplot as plt

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

def annotationError(detected:List[int], known: List[int]):
    return abs(len(detected)-len(known))



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
    points = []
    for df in dfs:
        avgDur = calcAvgDuration(df).seconds
        avgF1 = df["F1-Score"].fillna(handle_nan_as, inplace=False).mean()
        points.append({
            "duration": avgDur/60,
            "f1": avgF1,
            "name": df.iloc[-1]["Algorithm/Options"]
        })
    return points

def scatterF1_Duration(dfs:List[pd.DataFrame], handle_nan_as=np.nan):
    points = calcScatterData(dfs, handle_nan_as)
    plotScatterData(points)

def plotScatterData(points):
    fig,ax = plt.subplots()
    for point in points:
        ax.scatter(y=[point["f1"]], x=[point["duration"]], label=point["name"])
    ax.set_ylabel("Mean F1-Score")
    ax.set_ylim(0,1)
    ax.set_xlabel("Mean Duration (Minutes)")
    ax.legend(bbox_to_anchor=(1,1), loc="upper left")
    ax.grid(True)
    plt.savefig("../scatter_fig.png", bbox_inches='tight')
    plt.show()     
def hausdorff(detected:List[int], known: List[int]):
    return max(
        (
            max(
                min(
                    abs(t_predict-t_known)
                    for t_known in known
                )
                for t_predict in detected
            )
        ),
        (
            max(
                min(
                    abs(t_predict - t_known)
                    for t_predict in detected
                )
                for t_known in known
            )
        )
    )

def rand_index(detected:List[int], known: List[int], signal_length:int):
    def _calc_gr_ngr(changepoints: List[int]):
        gr = {(s,t) 
                for s in range(1,signal_length+1) 
                for t in range(1,signal_length+1)
                if
                    s < t and
                    # s and t belong to the same segment, i.e. there exists no changepoint between them
                    len([cp for cp in changepoints if s < cp and cp <= t]) == 0
            }
            
        ngr = {(s,t) 
                for s in range(1,signal_length+1) 
                for t in range(1,signal_length+1)
                if
                    s < t and
                    # s and t belong to the different segments, i.e. there exists a changepoint between them
                    #Calculate the number of changepoints between them
                    len([cp for cp in changepoints if s < cp and cp <= t]) > 0
            }
        return gr, ngr
    gr_detected, ngr_detected = _calc_gr_ngr(detected)
    gr_known, ngr_known = _calc_gr_ngr(known)
    randindex = (
        len(gr_detected.intersection(gr_known))+len(ngr_detected.intersection(ngr_known))
        ) / (
            signal_length * (signal_length -1)
        )

# def cover(detected:Set[int], known:Set[int], signal):
#     """
#         Computes the Covering Metric, C(S,G) for a Partitioning, S, of the Signal according to the algorithm, and the Ground Truth partitioning, G
#     """
#     def _jaccard_index(set1:Set, set2:Set):
#         return len(set1.intersection(set2))/len(set1.union(set2))
#     #By convention the first index is a Changepoint, as it is the first index of a Population
#     detected.add(0)
#     known.add(0)
#     k_sorted = sorted(known)
#     det_sorted = sorted(detected)
#     return 1/len(signal) * sum(
#         [len(signal[k_sorted[i-1]:k_sorted[i]]) * max(
#             [   
#                 _jaccard_index(
#                     set(signal[k_sorted[i-1]:k_sorted[i]]),
#                     set(signal[det_sorted[j-1]:det_sorted[j]])
#                 )
#                 for j in range(1,len(det_sorted))
#             ]
#         )
        
#          for i in range(1,len(k_sorted))]
#     )
    

# def cover_new(detected:List[int], known:List[int], signal):

#     def _jaccard_index(set1:Set, set2:Set):
#         return len(set1.intersection(set2))/len(set1.union(set2))

#     # Convert Detected and known to "Regions", i.e. partition the signal based on the changepoints
#     detected.append(0)
#     detected = sorted(detected)
#     detected_partitioned = []
#     for i in range(len(detected)-1):
#         detected_partitioned.append(set(signal[detected[i]:detected[i+1]]))
#     # Add final partition
#     detected_partitioned.append(set(signal[detected[-1]:]))

#     known.append(0)
#     known = sorted(known)
#     known_partitioned = []
#     for i in range(len(known)-1):
#         known_partitioned.append(set(signal[known[i]:known[i+1]]))
#     # Add final partition
#     known_partitioned.append(set(signal[known[-1]:]))

#     return 1/len(signal) * sum(
#         len(a)* max(
#             [
#                 _jaccard_index(
#                     a, a_prime
#                 )
#                 for a_prime in detected_partitioned
#             ]
#         )

#         for a in known_partitioned
#     )

# "Standard" Python imports
from typing import List, Tuple
import math
import numpy as np
import scipy.stats as stats
from scipy.signal import find_peaks
from enum import Enum
# Pm4py
from pm4py.objects.log.obj import EventLog
import pm4py.util.xes_constants as xes

# Other
from tqdm.auto import tqdm

# Own functions
from cdrift.utils.helpers import _getActivityNames, _getActivityNames_LogList, makeProgressBar


def _getCausalFootprint(log:EventLog, activities:List[str]=None, activityName_key:str=xes.DEFAULT_NAME_KEY)->np.chararray:
    """Helper Function to get the Causal Footprint Matrix of L. Contains for every pair of activities, whether the second always (A), sometimes (S), or never (N) follows the other in a trace.

    Args:
        log (EventLog): The event log for which to extract the matrix
        activities (List[str], optional): The activities for which to extract the Matrix. If `None`, returns the matrix for all activities found in L. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.

    Returns:
        np.chararray: The Causal Footprint Matrix, as used by Bose et al.
    """    

    if activities is None:
        activities = _getActivityNames(log, activityName_key=activityName_key)

    d = {(act1,act2):'%' for act1 in activities for act2 in activities}

    ### My conventions in this method:
    # - % means no relation between these two found yet, this is changed to 'never' in the end
    # - S Sometimes, A always, N never
    # d[i][j] contains the information about j eventually following i

    # Ideally, this method would, for the most part, be replaced by a call to `pm4py.filter_eventually_follows_relation`, however this yields unexpected results.


    for trace in log:
        #Track which activities we have seen in this case so far
        seen = set()
        A_touched = set() # This is the set of pairs where the Always value has been updated, if at the end of the trace there exists an always relation which hasnt been updated, but the first activity in the pair is in seen, then it has to be set to sometimes
        for event in trace:
            # Update the follows relations
            name = event[activityName_key]
            for s in seen:
                # Update it 
                valnow = d[(s,name)]
                if valnow in ['%','A']:
                    d[(s,name)] = 'A'
                    A_touched.add((s,name))
                elif valnow == 'N':
                    d[(s,name)] = 'S'
                # Else the value is S, and will stay S
            seen.add(name)
        # Now 'demote' all A to S which occured in the trace but 
        for act1 in seen:
            for act2 in activities:
                if  d[(act1,act2)] == 'A' and (act1,act2) not in A_touched:
                    # 'Always' does not hold anymore
                      d[(act1,act2)] = 'S'
                elif  d[(act1,act2)] == '%':
                     d[(act1,act2)] = 'N'
    # Finally, convert this dictionary into a 2d matrix
    output = np.chararray((len(activities),len(activities)),unicode=True)
    for act1 in activities:
        i1 = activities.index(act1)
        for act2 in activities:
            i2 = activities.index(act2)
            output[i1][i2] = d[(act1,act2)] if d[(act1,act2)] != '%' else 'N'
    return output


# According to the definition in "Dealing With Concept Drifts in Process Mining" By R. P. Jagadeesh Chandra Bose, Wil M. P. van der Aalst, Indr˙e Žliobait˙e, and Mykola Pechenizkiy (DOI:10.1109/TNNLS.2013.2278313)
def extractRelationTypeCount(logs:List[EventLog], activityName_key:str=xes.DEFAULT_NAME_KEY)->np.ndarray:
    """Extracts for each Log in logs: for every activity, the amount of activities which

    - Always eventually follow it in a trace (index 0)
    - Sometimes eventually follow it in a trace (index 1)
    - Never eventually follow it in a trace (index 2), so where no trace exists where this activity is eventually followed by that other activity

    Args:
        logs (List[EventLog]): A list of event logs to extract the measure for. As this measure is a global measure, it is not defined on traces, but on (sub-) logs
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.

    Returns:
        np.ndarray: A time series of Relation Type Count Values for the sequence of  sublogs.

        Dimensions: 3|Activities|x|Log|
        Contains for each Log in Logs:  For each activity a tuple containing c_A, c_S, c_N, so the count of activities that Always/Sometimes/Never follow it in this log
    """ 
    names = _getActivityNames_LogList(logs)
    #Every Row corresponds to an activity
    num_activities = len(names)
    output = np.zeros((3*num_activities,len(logs)))

    progress = makeProgressBar(num_iters=len(logs), message="calculating relation type count, sublogs completed")
    for n, log in enumerate(logs):
        matrix = _getCausalFootprint(log,names, activityName_key=activityName_key)
        #Output is "rows" of "tuples" (c_A, c_S, c_N), so the index of A is 0, S is 1, N is 2
        for i in range(len(matrix)):#Iterate through the rows
            for j in range(len(matrix[i])):#Iterate through the cells in the row
                if matrix[i][j] == 'A':
                    output[3*i][n] += 1
                elif matrix[i][j] == 'S':
                    output[(3*i) + 1][n] += 1
                elif matrix[i][j] == 'N':
                    output[(3*i) + 1][n] += 1
        if progress is not None:
            progress.update()
    return output

# According to the definition in "Dealing With Concept Drifts in Process Mining" By R. P. Jagadeesh Chandra Bose, Wil M. P. van der Aalst, Indr˙e Žliobait˙e, and Mykola Pechenizkiy (DOI:10.1109/TNNLS.2013.2278313)
def extractRelationEntropy(logs:List[EventLog], activityName_key:str=xes.DEFAULT_NAME_KEY, rc:np.chararray=None)->np.ndarray:
    """Extracts the Relation Entropy measure for a sequence of sublogs.

    Args:
        logs (List[EventLog]): A list of event logs to extract the measure for. As this measure is a global measure, it is not defined on traces, but on (sub-) logs
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        rc (np.chararray, optional): Pre-computed Relation-Type-Count Values, as this measure depends on those. If None, these are automatically computed. Defaults to None.

    Returns:
        np.ndarray: Time Series of Relation Entropy, with dimensions |Activities|x|Log|
    """    

    if rc is None:
        rc = extractRelationTypeCount(logs, activityName_key)
    names = _getActivityNames_LogList(logs, activityName_key=activityName_key)
    num_activities = len(names)
    output = np.empty((num_activities, len(logs)))

    progress = makeProgressBar(num_iters=len(logs), message="calculating relation entropy, sublogs completed")
    for n in range(len(logs)):
        for i in range(num_activities):
        # Indices: A=0, S=1, N=2 (doesnt *really* matter anyways)
            p_a = rc[(3*i)][n]/num_activities
            p_s = rc[(3*i)+1][n]/num_activities
            p_n = rc[(3*i)+2][n]/num_activities
            t_a = 0 if p_a == 0 else -p_a*math.log2(p_a)
            t_s = 0 if p_s == 0 else -p_s*math.log2(p_s)
            t_n = 0 if p_n == 0 else -p_n*math.log2(p_n)
            output[i][n] = t_a + t_s + t_n
        if progress is not None:
            progress.update()
    return output

def _calculateSF(log:EventLog, act1:str, act2:str, windowsize:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY)->List[Tuple[List[str],List[str]]]:
    """Helper function to compute the S and F Sets used in the definition of the Window Count Measure.

    Args:
        log (EventLog): The event log for which to extract the sets
        act1 (str): The first activity of the pair for which to compute the sets
        act2 (str): The second activity of the pair for which to compute the sets
        windowsize (int, optional): The window dictating how far to look for the eventually follows relation. If None, defaults to the average trace length. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.

    Returns:
        List[Tuple[List[str],List[str]]]: A list containing, for each trace, a tuple of 1) The S Set, and 2) The F Set
        (For each Trace t in the event log, a tuple of (S,F), so S^{l,t}(a,b) and F^{l,t}(a,b))
    """

    if windowsize == None:
        #Set windowsize to the average trace length, if not set
        windowsize = sum([len(t) for t in log])//len(log)
    
    #output = [0 for i in range(len(log))]
    output = np.empty(len(log), dtype='O')
    for i, trace in enumerate(log):# For each trace
        #Build the Bag S^{l,t}
        # S^{l,t}(act1) is the multiset of all traces Subsquences of the Trace t, with length l that begin with the Activity act1
        S = []
        for j, event in enumerate(trace):
            if event[activityName_key] == act1:
                #We add the subtrace starting here (length l) to the multiset
                S.append(
                    #List containing the names of the Activities in this subtrace
                    [act[activityName_key] for act in trace[j:j+windowsize]]
                )
        #Now build F^{l,t}
         #F^{l,t}(act1, act2) = is the set of all subsequences of t with length l, beginning with act1 (so far just S^{l,t}(act1)) in which act2 eventually follows act1
        F = [s for s in S if act2 in s[1:]] # Excluding the first activity; hence S[1:]

        output[i] = (S,F)
    return output


# According to the definition in "Dealing With Concept Drifts in Process Mining" By R. P. Jagadeesh Chandra Bose, Wil M. P. van der Aalst, Indr˙e Žliobait˙e, and Mykola Pechenizkiy (DOI:10.1109/TNNLS.2013.2278313)
def extractWindowCount(log:EventLog, act1:str, act2:str, windowsize:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY)->np.ndarray:
    """Extracts a Series showing, for each trace (ordered by time), it's WC (Window Count) Score, i.e. The amount of times act2 eventually follows act1 (on a trace-basis) (See Dealing With Concept Drifts in Process Mining by Bose (DOI:10.1109/TNNLS.2013.2278313))

    Args:
        log (EventLog): The log for which to extract the WC-Measure
        act1 (str): The name of the first activity, we consider how often, in a trace, after act1 act2 will follow
        act2 (str): The name of the second activity, for which we consider how often it eventually follows act1
        windowsize (int, optional): The window, how far we check for an eventually-follows relation of act1 and act2, if None, this will be set to the average trace length of the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.

    Returns:
        np.ndarray:  The Window Count, i.e. The amount of times act2 eventually follows act1 within a window of `windowsize` per trace, so a 1x|Log| numpy.ndarray
    """
    
    return [len(f) for s,f in _calculateSF(log, act1, act2, windowsize, activityName_key=activityName_key)]


# According to the definition in "Dealing With Concept Drifts in Process Mining" By R. P. Jagadeesh Chandra Bose, Wil M. P. van der Aalst, Indr˙e Žliobait˙e, and Mykola Pechenizkiy (DOI:10.1109/TNNLS.2013.2278313)
def extractJMeasure(log:EventLog, act1:str, act2:str, windowsize:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY)->np.ndarray:
    """Extracts a Series showing, for each trace (ordered by time), it's J-Measure, as defined by Smyth and Goodman, and applied by Bose et al.

    Args:
        log (EventLog): The log for which to extract the J-Measure
        act1 (str): The name of the first activity, we consider how often, in a trace, after act1 act2 will follow
        act2 (str): The name of the second activity, for which we consider how often it eventually follows act1
        windowsize (int, optional): The window, how far we check for an eventually-follows relation of act1 and act2, if None, this will be set to the average trace length of the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.

    Returns:
        np.ndarray: The J-Measure Time-Series (A 1x|log| signal of J-Measures)
    """

    output = np.empty(len(log))
    sf = _calculateSF(log, act1, act2, windowsize, activityName_key=activityName_key)
    for i, trace in enumerate(log):
        S, F = sf[i]
        #Avoiding Division by Zero here by setting p_ab to 0. 
        # If len(S) is 0 then len(F) is 0 too resulting in 0/0. 
        #p(a,b), the probability that b will follow a in window l in t is obviously 0 as then no such example exists
        p_ab = len(F)/len(S) if len(S) != 0 and len(F) != 0 else 0

        #Count how many times the activity happens in the trace, divide by amount of events in trace for p_a and p_b
        p_a = len([act for act in trace if act[activityName_key]==act1])/len(trace)
        p_b = len([act for act in trace if act[activityName_key]==act2])/len(trace)

        ct = p_ab * (0 if p_ab==0 or p_b == 0 else math.log2(p_ab/p_b)) + (1 - p_ab) * (0 if (1-p_ab) == 0 or (1-p_b) == 0 else math.log2((1-p_ab)/(1-p_b)))
        output[i]=p_a * ct
    return output

def KSTest_2Sample_SlidingWindow(signal:np.ndarray, windowSize:int)->np.ndarray:
    """Applies the Two-Sample Kolmogorov-Smirnov Test to the given Signal.

    Args:
        signal (np.ndarray): The (one-dimensional) Signal on which we want to find the changepoints; e.g. a Time Series
        windowSize (int): The size of the sliding windows which we consider; These are next to eachother, and slid along the signal, at each step comparing them and checking if they come from the same distribution

    Returns:
        np.ndarray: An ndarray containing the calculated p-values (Same dimensions of input signal)
    """

    #Default to 1; This is the case at the edges of the signal
    pvals = np.ones(len(signal))
    # Shift 2 windows of size `windowSize` over the signal and apply the Kolmogorov-Smirnov Test
    for i in range(len(signal)-(2*windowSize)):
        #The 2 windows we are comparing
        window1 = signal[i:i+windowSize]
        window2 = signal[i+windowSize:i+(2*windowSize)]
        #Calculate the KS Test for these two windows
        ks = stats.ks_2samp(window1, window2)
        pvals[i+windowSize] = ks.pvalue
    return pvals


def MannWhitney_U_SlidingWindow(signal:np.ndarray, windowSize:int)->np.ndarray:
    """Applies the Mann Whitney U-Test to the Signal

    Args:
        signal (np.ndarray): The (one-dimensional) Signal on which we want to find the changepoints; e.g. a Time Series
        windowSize (int): The size of the sliding windows which we consider; These are next to eachother, and slid along the signal, at each step computing the U-Test

    Returns:
        np.ndarray: An ndarray containing the calculated p-values (Same dimensions of input signal)
    """ 

    # Shift 2 windows of size `windowSize` over the signal and apply the Kolmogorov-Smirnov Test
    pvals = np.ones(len(signal))
    for i in range(len(signal)-(2*windowSize)):
        #The 2 windows we are comparing
        window1 = signal[i:i+windowSize]
        window2 = signal[i+windowSize:i+(2*windowSize)]
        u = stats.mannwhitneyu(window1, window2)
        pvals[i+windowSize] = u.pvalue
    return pvals

def _detectChangeLocal(log:EventLog, stattest:str, measure:str, windowSize:int, measure_window:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->np.ndarray:
    """A helper function to automatically apply statistical testing using a local measure (J or Window Count). Measures are automatically calculated and then statistical testing applied.

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        stattest (str): The statistical test to employ on the extracted measure.
        measure (str): The name of the used measure ("J" or "WC")
        windowSize (int): The window size to use for sliding window statistical testing.
        measure_window (int, optional): The window size to use for the measure extraction. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Raises:
        ValueError: if the string supplied as `measure` is invalid.

    Returns:
        np.ndarray: An array of the computed p-values. Dimensions 1x|log|
    """

    activities = _getActivityNames(log,activityName_key)
    pvals = np.zeros(len(log))

    progress = None
    if show_progress_bar:
        progress = makeProgressBar(num_iters=len(activities)**2, message=f"Calculating {measure} P-Values for Bose, activity pairs complete")
    for act1 in activities:
        for act2 in activities:
            if measure in ["j", "J"]:
                m = extractJMeasure(log, act1,act2, measure_window, activityName_key)
            elif measure in ["wc", "WC"]:
                m = extractWindowCount(log, act1, act2, measure_window, activityName_key)
            else:
                raise ValueError("Invalid measure extraction argument.")
            if stattest in ["ks", "KS"]:
                pvals_ = KSTest_2Sample_SlidingWindow(m,windowSize) 
            elif stattest in ["u", "U", "mu", "MU"]:
                pvals_ = MannWhitney_U_SlidingWindow(m, windowSize)
            # Add new pvals for mean calculation later
            pvals += pvals_
            if progress is not None:
                progress.update()
    pvals = pvals / pow(len(activities),2)
    if progress is not None:
        progress.close()
    return pvals

def detectChange_JMeasure_KS(log:EventLog, windowSize:int, measure_window:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->np.ndarray:
    """Apply Concept Drift Detection using the J-Measure and a Kolmogorov-Smirnov Test

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        windowSize (int): The window size to use for sliding window statistical testing.
        measure_window (int, optional): The window size to use for the measure extraction. If `None`, defaults to average trace length in the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Returns:
         np.ndarray: An array of the computed p-values. Dimensions 1x|log|
    """
    
    return _detectChangeLocal(log, "KS", "J", windowSize, measure_window, activityName_key, show_progress_bar, progressBarPos)

def detectChange_JMeasure_MU(log:EventLog, windowSize:int, measure_window:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->np.ndarray:
    """Apply Concept Drift Detection using the J-Measure and a Mann-Whitney U-Test

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        windowSize (int): The window size to use for sliding window statistical testing.
        measure_window (int, optional): The window size to use for the measure extraction. If `None`, defaults to average trace length in the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Returns:
         np.ndarray: An array of the computed p-values. Dimensions 1x|log|
    """
    
    return _detectChangeLocal(log, "MU", "J", windowSize, measure_window, activityName_key, show_progress_bar, progressBarPos)

def detectChange_WC_KS(log:EventLog, windowSize:int, measure_window:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->np.ndarray:
    """Apply Concept Drift Detection using the Window Count and a Kolmogorov-Smirnov Test

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        windowSize (int): The window size to use for sliding window statistical testing.
        measure_window (int, optional): The window size to use for the measure extraction. If `None`, defaults to average trace length in the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Returns:
         np.ndarray: An array of the computed p-values. Dimensions 1x|log|
    """
    
    return _detectChangeLocal(log, "KS", "WC", windowSize, measure_window, activityName_key, show_progress_bar, progressBarPos)

def detectChange_WC_MU(log:EventLog, windowSize:int, measure_window:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->np.ndarray:
    """Apply Concept Drift Detection using the Window Count and a Mann-Whitney U-Test

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        windowSize (int): The window size to use for sliding window statistical testing.
        measure_window (int, optional): The window size to use for the measure extraction. If `None`, defaults to average trace length in the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Returns:
         np.ndarray: An array of the computed p-values. Dimensions 1x|log|
    """
    
    return _detectChangeLocal(log, "MU", "WC", windowSize, measure_window, activityName_key, show_progress_bar, progressBarPos)


def visualInspection(signal:np.ndarray,trim:int=0)->List[int]:
    """Automated visual inspection of pvalues. Used for consistent and unbiased evaluations.

    Based on the `find_peaks` algorithm of scipy.

    Args:
        signal (np.ndarray): The pvaues to inspect
        trim (int, optional): The number of values to trim from each side before detection. Defaults to 0. This is useful, because `windowSize` values at the beginning and end of the resulting pvalue series of sliding window tests default to 1, and are uninteresting and irrelevant for the inspection.

    Returns:
        List[int]: A list of found change point indices (integers)
    """
    # Only send the trimmed version into the peak-finding algorithm; Because the initial, and final zero-values are the default values, and no comparison was made there, so it doesn't count for the peak finding
    peaks= find_peaks(-signal[trim:len(signal)-trim], width=80, prominence=0.1)[0]
    return [x+trim for x in peaks] # Add the window that was lost from the beginning



# For Multivariate Time-Series:
def _HotellingTSquare(population1:List[np.ndarray], population2:List[np.ndarray])->float:
    """An implementation of the multivariate, two-sample Hotelling T^2 Test.

    Args:
        population1 (List[np.ndarray]): Population 1, an (multivariate) array of values.
        population2 (List[np.ndarray]): Population 1, an (multivariate) array of values.

    Returns:
        float: The computed p-value
    """    
    
    # The T^2 Test is a multivariate two sample test, so we consider two population consisting of vectors
    # Calculating based on this definition: https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Hotellings_Two-Sample_T2.pdf

    n1 = len(population1)
    n2 = len(population2)


    mean1 = np.mean(population1, axis=0)
    mean2 = np.mean(population2, axis=0)

    #Pooled Variance-Covariance Matrix: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/poolcov.htm
    S1 = np.cov(population1, rowvar=False)
    S2 = np.cov(population2, rowvar=False)

    s_pl = (((n1-1)*S1) + ((n2-1)*S2)) / (n1+n2-2)

    diff = mean1 - mean2

    t2 = ((n1*n2)/(n1+n2)) * np.matmul(np.matmul(diff.transpose(), np.linalg.inv(s_pl)), diff)

    # Inspiration from https://www.r-bloggers.com/2020/10/hotellings-t2-in-julia-python-and-r/
    # Model this as an F-Distribution to get the p-value
    _ , num_vars = population1.shape
    statistic = t2 * (n1+n2-num_vars-1)/(num_vars*(n1+n2-2))
    F = stats.f(num_vars, n1 + n2 - num_vars -1)
    p_value = 1 - F.cdf(statistic)

    return p_value

def Hotelling_Square_Test(signal:np.ndarray, windowSize:int)->np.ndarray:
    """Apply the Hotelling T^2 Test on a signal using sliding windows.

    Args:
        signal (np.ndarray): The signal, a multivariate time series
        windowSize (int): The window size to use for the sliding window algorithm.

    Returns:
        np.ndarray: Array of calculated p-values.
    """

    signal = np.swapaxes(signal,0,1)
    res = np.ones(len(signal))
    # Shift 2 windows of size `windowSize` over the signal and apply the Kolmogorov-Smirnov Test
    for i in range(len(signal)-(2*windowSize)):
        #The 2 windows we are comparing
        window1 = signal[i:i+windowSize]
        window2 = signal[i+windowSize:i+(2*windowSize)]
        #Calculate the Hotelling Test for these 2 samples
        # res[i+windowSize] = _HotellingTSquare(window1, window2)
        _,_,pval,_ = _HotellingTSquare(window1, window2)
        res[i+windowSize] = pval

class StatTest(Enum):
    KolmogorovSmirnov = stats.ks_2samp
    KS = stats.ks_2samp
    MannWhitneyU = stats.mannwhitneyu
    U = stats.mannwhitneyu

def _detectChangeLocal_Step(log:EventLog, stattest:str, measure:str, windowSize:int, measure_window:int=None, step_size:int=1, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->np.ndarray:
    """A helper function to automatically apply statistical testing using a local measure (J or Window Count). Measures are automatically calculated and then statistical testing applied.

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        stattest (str): The statistical test to employ on the extracted measure.
        measure (str): The name of the used measure ("J" or "WC")
        windowSize (int): The window size to use for sliding window statistical testing.
        measure_window (int, optional): The window size to use for the measure extraction. Defaults to None.
        step_size (int, optional): The step size to use for sliding the windows. Defaults to 1.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Raises:
        ValueError: if the string supplied as `measure` is invalid.

    Returns:
        np.ndarray: An array of the computed p-values. Dimensions 1x|log|
    """

    activities = _getActivityNames(log,activityName_key)
    pvals = None
    progress = None
    if show_progress_bar:
        progress = makeProgressBar(num_iters=len(activities)**2, message=f"Calculating {measure} P-Values for Bose, activity pairs complete")
    for act1 in activities:
        for act2 in activities:
            if measure in ["j", "J"]:
                m = extractJMeasure(log, act1,act2, measure_window, activityName_key)
            elif measure in ["wc", "WC"]:
                m = extractWindowCount(log, act1, act2, measure_window, activityName_key)
            else:
                raise ValueError("Invalid measure extraction argument.")
            if stattest in ["ks", "KS"]:
                pvals_ = KSTest_2Sample_SlidingWindow_Step(m,windowSize, step_size) 
            elif stattest in ["u", "U", "mu", "MU"]:
                pvals_ = MannWhitney_U_SlidingWindow_Step(m, windowSize, step_size)
            # Add new pvals for mean calculation later
            pvals = pvals + pvals_ if pvals is not None else pvals_
            if progress is not None:
                progress.update()
    pvals = pvals / pow(len(activities),2)
    if progress is not None:
        progress.close()
    return pvals

def KSTest_2Sample_SlidingWindow_Step(signal:np.ndarray, windowSize:int, step_size:int=1)->np.ndarray:
    """Applies the Two-Sample Kolmogorov-Smirnov Test to the given Signal.

    Args:
        signal (np.ndarray): The (one-dimensional) Signal on which we want to find the changepoints; e.g. a Time Series
        windowSize (int): The size of the sliding windows which we consider; These are next to eachother, and slid along the signal, at each step comparing them and checking if they come from the same distribution

    Returns:
        np.ndarray: An ndarray containing the calculated p-values (Same dimensions of input signal)
    """

    #Default to 1; This is the case at the edges of the signal
    pvals = []
    # Shift 2 windows of size `windowSize` over the signal and apply the Kolmogorov-Smirnov Test
    i = 0
    while i < len(signal)-(2*windowSize):
        #The 2 windows we are comparing
        window1 = signal[i:i+windowSize]
        window2 = signal[i+windowSize:i+(2*windowSize)]
        #Calculate the KS Test for these two windows
        ks = stats.ks_2samp(window1, window2)
        pvals.append(ks.pvalue)
        i += step_size
    return np.asarray(pvals)

def MannWhitney_U_SlidingWindow_Step(signal:np.ndarray, windowSize:int, step_size:int=1)->np.ndarray:
    """Applies the Mann Whitney U-Test to the Signal

    Args:
        signal (np.ndarray): The (one-dimensional) Signal on which we want to find the changepoints; e.g. a Time Series
        windowSize (int): The size of the sliding windows which we consider; These are next to eachother, and slid along the signal, at each step computing the U-Test
        step_size (int, optional): The step size to use for sliding the windows. Defaults to 1.

    Returns:
        np.ndarray: An ndarray containing the calculated p-values (Same dimensions of input signal)
    """ 

    # Shift 2 windows of size `windowSize` over the signal and apply the Kolmogorov-Smirnov Test
    pvals = []

    i = 0
    while i < len(signal)-(2*windowSize):
        #The 2 windows we are comparing
        window1 = signal[i:i+windowSize]
        window2 = signal[i+windowSize:i+(2*windowSize)]
        u = stats.mannwhitneyu(window1, window2)
        pvals.append(u.pvalue)
        i += step_size
    return np.asarray(pvals)

def visualInspection_Step(signal:np.ndarray, window_size:int, step_size:int=1)->List[int]:
    """Automated visual inspection of pvalues. Used for consistent and unbiased evaluations.

    Based on the `find_peaks` algorithm of scipy.

    Args:
        signal (np.ndarray): The pvaues to inspect
        trim (int, optional): The number of values to trim from each side before detection. Defaults to 0. This is useful, because `windowSize` values at the beginning and end of the resulting pvalue series of sliding window tests default to 1, and are uninteresting and irrelevant for the inspection.

    Returns:
        List[int]: A list of found change point indices (integers)
    """
    # Only send the trimmed version into the peak-finding algorithm; Because the initial, and final zero-values are the default values, and no comparison was made there, so it doesn't count for the peak finding

    peaks= find_peaks(-signal, width=80/step_size, prominence=0.1)[0]
    return [(x*step_size) + window_size for x in peaks] # Add the window that was lost from the beginning

def detectChange_JMeasure_KS_Step(log:EventLog, windowSize:int, measure_window:int=None, step_size:int=1, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->np.ndarray:
    """Apply Concept Drift Detection using the J-Measure and a Kolmogorov-Smirnov Test

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        windowSize (int): The window size to use for sliding window statistical testing.
        measure_window (int, optional): The window size to use for the measure extraction. If `None`, defaults to average trace length in the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Returns:
         np.ndarray: An array of the computed p-values. Dimensions 1x|log|
    """
    
    return _detectChangeLocal_Step(log, "KS", "J", windowSize, measure_window, step_size, activityName_key, show_progress_bar, progressBarPos)

def detectChange_JMeasure_MU_Step(log:EventLog, windowSize:int, measure_window:int=None, step_size:int=1, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->np.ndarray:
    """Apply Concept Drift Detection using the J-Measure and a Mann-Whitney U-Test

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        windowSize (int): The window size to use for sliding window statistical testing.
        measure_window (int, optional): The window size to use for the measure extraction. If `None`, defaults to average trace length in the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Returns:
         np.ndarray: An array of the computed p-values. Dimensions 1x|log|
    """
    
    return _detectChangeLocal_Step(log, "MU", "J", windowSize, measure_window, step_size, activityName_key, show_progress_bar, progressBarPos)

def detectChange_WC_KS_Step(log:EventLog, windowSize:int, measure_window:int=None, step_size:int=1, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->np.ndarray:
    """Apply Concept Drift Detection using the Window Count and a Kolmogorov-Smirnov Test

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        windowSize (int): The window size to use for sliding window statistical testing.
        measure_window (int, optional): The window size to use for the measure extraction. If `None`, defaults to average trace length in the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Returns:
         np.ndarray: An array of the computed p-values. Dimensions 1x|log|
    """
    
    return _detectChangeLocal_Step(log, "KS", "WC", windowSize, measure_window, step_size, activityName_key, show_progress_bar, progressBarPos)

def detectChange_WC_MU_Step(log:EventLog, windowSize:int, measure_window:int=None, step_size:int=1, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->np.ndarray:
    """Apply Concept Drift Detection using the Window Count and a Mann-Whitney U-Test

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        windowSize (int): The window size to use for sliding window statistical testing.
        measure_window (int, optional): The window size to use for the measure extraction. If `None`, defaults to average trace length in the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Returns:
         np.ndarray: An array of the computed p-values. Dimensions 1x|log|
    """
    
    return _detectChangeLocal_Step(log, "MU", "WC", windowSize, measure_window, step_size, activityName_key, show_progress_bar, progressBarPos)

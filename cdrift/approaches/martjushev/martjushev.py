from cdrift.approaches.bose import extractJMeasure, extractWindowCount
from cdrift.utils.helpers import _getActivityNames, makeProgressBar

import numpy as np
from typing import Callable, List, Any, Tuple, Union, Iterable
from numbers import Number
import scipy.stats as stats
from pm4py.objects.log.obj import EventLog
from pm4py.util import xes_constants as xes


# The Recursive Bisection Algorithm for locating the point of change within two populaations as Described in "Change Point Detection and Dealing with Gradual and Multi-Order Dynamics in Process Mining" by Martjushev, Bose, Van Der Aalst
def _locateChange(pop1:np.ndarray, pop2:np.ndarray, baseindex:int, test: Callable, pvalue:float=0.05, **kwargs)->int:
    """A helper function to locate the exact change point after a statistical test indicated a change point.

    A changepoint has been detected somewhere between these two populations. Now recursively the statistical test will be continued on these two populations to find the point with the lowest p-value that lies below alpha.

    This algorithm is described in "Change Point Detection and Dealing with Gradual and Multi-Order Dynamics in Process Mining" by Martjushev et al.

    Args:
        pop1 (np.ndarray): The first population. Applying the statistical test on the two given populations should yield a value below `pvalue`
        pop2 (np.ndarray): The second population. Applying the statistical test on the two given populations should yield a value below `pvalue`  
        baseindex (int): The index in the signal of the first element in pop1.
        test (Callable): The function for the statistical test. test(pop1,pop2, **kwargs) will be called and the p-value accessed with _getPValue(test(pop1,pop2, **kwargs))
        pvalue (float, optional): The p-value threshold, under which a pvalue indicates a change point. Defaults to 0.05.

    Returns:
        int: The index where the change occurred.
    """

    # Split the two populations into halves

    #At some point both of the populations might be empty, i.e. no change, i.e. this is a base case
    pop11 = pop1[:len(pop1)//2]
    pop12 = pop1[len(pop1)//2:]
    pop21 = pop2[:len(pop2)//2]
    pop22 = pop2[len(pop2)//2:]

    p_left = _getPValue(test(pop11,pop12, **kwargs))
    p_center = _getPValue(test(pop12,pop21, **kwargs))
    p_right = _getPValue(test(pop21,pop22, **kwargs))
    
    p_min = min(p_left, p_center, p_right)
    if not p_min <= pvalue:
        #The smallest pvalue we calculated is not below our pvalue to be considered as a changepoint;
        # Thus index of the changepoint is at the end of pop1
        return baseindex + len(pop1)-1
    if p_min == p_left:# p_left was calculated from pop11 and pop12; Continue the search for these two populations
        return _locateChange(pop11, pop12, baseindex, test, pvalue, **kwargs)
    elif p_min == p_center:# p_center was calculated from pop12 and pop21; Continue for these two populations
        return _locateChange(pop12, pop21, baseindex + len(pop11),test, pvalue, **kwargs)
    else:# p_right has the smallest p-value; p_right was calculated from pop21 and pop22; Continue the for these two populations
        return _locateChange(pop21, pop22, baseindex + len(pop1), test, pvalue, **kwargs)

def _getPValue(res: Any)->float:
    """A wrapper-function to get a pvalue from the result of a statistical test. First, if it is a number, it is returned. Otherwise it attempts to read its `pvalue` attribute. If all fails an exception is raised.

    Args:
        res (Any): The statistical test result to read

    Raises:
        ValueError: If the statistical test result is not readable in the ways we currently support.

    Returns:
        float: The pvalue of the statistical test result.
    """
    if isinstance(res,Number):
        return res
    else:
        try:
            return res.pvalue
        except:
            raise ValueError("Statistical Test Result does not match criteria; Need either a float or an object with pvalue attribute")

def statisticalTesting_RecursiveBisection(signal:np.ndarray, windowSize:int, pvalue:float, testingFunction:Callable, return_pvalues:bool=False, **kwargs)->Union[List[int], Tuple[List[int],np.ndarray]]:
    """Applies a given Statistical Test to the signal. And localizes the exact changepoints using the Recursive Bisection algorithm

    Args:
        signal (np.ndarray): The signal to be analyzed
        windowSize (int): The sizes of the populations to be compared (The sliding window size for the sliding window algorithm)
        pvalue (float):  The p-value threshold, under which a pvalue indicates a change point.
        testingFunction (Callable): The function for the statistical test. test(pop1,pop2, **kwargs) will be called and the p-value accessed with _getPValue(test(pop1,pop2, **kwargs)). An error is raised if the result of this function is neither a number or has a `pvalue` attribute.
        return_pvalues (bool, optional): Configures whether to return the computed pvalues. Defaults to False.

    Returns:
        Union[List[int], Tuple[List[int],np.ndarray]]: A list of the detected change point indices, and, if selected a numpy array of the computed p-values. 

    Raises:
        ValueError: If the result of the testing function is not a) a number or b) is an object with a `pvalue` attribute
    """

    changepoints = []
    pvals = np.ones(len(signal))
    # Shift 2 windows of size `windowSize` over the signal and apply the given Test
    # We use skipuntil instead of e.g. using a while loop so that we still get the pvalues of the populations that we skip, for graphing
    skipuntil = -1
    for i in range(len(signal)-(2*windowSize)):
        #Get the populations
        window1 = signal[i:i+windowSize]
        window2 = signal[i+windowSize:i+(2*windowSize)]

        #Compare populations
        p = testingFunction(window1,window2, **kwargs)
        pval = _getPValue(p)
        
        pvals[i+windowSize] = pval
        # If this indicates a changepoint and we do not skip this part (due to a close previous change point)
        if pval < pvalue and i >= skipuntil:
            #Changepoint detected
            changepoints.append(
                # Apply recursive bisection to locate the change point
                _locateChange(pop1=window1, pop2=window2, baseindex=i,test=testingFunction, pvalue=pvalue)
            )
            #TODO: Index after second population of final statistical test in recursion?
            #Continue searching at the first index after the second population
            skipuntil = i + (2*windowSize)

    return changepoints if not return_pvalues else (changepoints, pvals)


# The Adaptive Window Changepoint Detection Algorithm as Described in "Change Point Detection and Dealing with Gradual and Multi-Order Dynamics in Process Mining" by Martjushev, Bose, Van Der Aalst
# def adaptiveWindowChangepoints(signal:np.ndarray, minWindowSize:int, maxWindowSize:int, pvalue:float, stepSize:int, testingFunction:Callable, **kwargs):
#     index = 0
#     windowSize = minWindowSize
#     p_left = signal[index:index+windowSize]
#     p_right = signal[index+windowSize:index+(2*windowSize)]
#     changepoints = []
#     while index + 2 * (windowSize) <= len(signal):
#         p = _getPValue( # Get the pvalue from the result of the Statistical Test
#             testingFunction(p_left, p_right, **kwargs)
#         )
#         if p < pvalue: # If there is a significant difference between the populations
#             #Locate the chancge
#             changepoints.append(_locateChange(p_left, p_right, index, testingFunction, **kwargs))
#             #Continue at the first index after the second Population
#             index += 2*windowSize
#             # Calculate the new Populations
#             p_left = signal[index:index+windowSize]
#             p_right = signal[index+windowSize:index+(2*windowSize)]
#             windowSize = minWindowSize # Reset the windowSize to minWindowSize
#         else: # No significant difference
#             windowSize += stepSize # Increase the window size
#             # Recalculate the Populations
#             p_left = signal[index:index+windowSize]
#             p_right = signal[index+windowSize:index+(2*windowSize)]
#             # If we have exceeded the maximum window Size
#             if windowSize >= maxWindowSize:
#                 # Discard the left population since it is "old" information now
#                 p_left = p_right[:len(p_right//2)]
#                 p_right_ = p_right[len(p_right//2):] # Helper to avoid problems
#                 index += windowSize  # set the index to the beginning of the second window
#                 p_right = p_right_
#     return changepoints

def _applyAvgPVal(window1, window2, testingFunc:Callable)->float:
    """A helper function to calculate the average pvalue for a multivariate signal (Relevant for applying the test on all pairs of activities).

    Args:
        window1 (Any): The first population
        window2 (Any): The second population
        testingFunc (Callable): The testing function.

    Returns:
        float: The average of the computed pvalue over all levels of the signal
    """    
    pvals = []
    w1 = window1
    w2 = window2
    # w1 = np.swapaxes(window1, 0,1)
    # w2 = np.swapaxes(window2, 0,1)
    for i in range(len(w1)):
        pval = _getPValue(testingFunc(w1[i],w2[i]))
        pvals.append(pval)
    return np.mean(pvals)


def recursiveBisection(p1:np.ndarray, p2:np.ndarray, pvalue:float, startIndex:int, testingFunction:Callable, **kwargs)->int:
    """The implementation of the Recursive Bisection Algorithm. Applied when we found a changepoint between the population `p1` and`p2`. Through recursive application of statistical tests on smaller and smaller populations, an accurate detection is made.

    Args:
        p1 (Any): The first population.
        p2 (Any): The second population.
        pvalue (float):  The p-value threshold, under which a pvalue indicates a change point.
        startIndex (int): The index in the entire signal where the first population starts. Used to calculate the index of the detection to return.
        testingFunction (Callable): The testing function used to compare populations.

    Returns:
        int: The index of where the change point is detected.
    """


    # StartIndex is the index in the entire Datastream where pop1 begins
    p1_startIndex = startIndex
    p_min = _getPValue(testingFunction(p1, p2, **kwargs))
    while p_min < pvalue:
        p11 = p1[:len(p1)//2]
        p12 = p1[len(p1)//2:]
        p21 = p2[:len(p2)//2]
        p22 = p2[len(p2)//2:]
        p_left = _getPValue(testingFunction(p11,p12,**kwargs))
        p_center = _getPValue(testingFunction(p12,p21,**kwargs))
        p_right = _getPValue(testingFunction(p21,p22,**kwargs))

        # Which one is min?
        p_min = min(p_left,p_center,p_right)
        if p_min < pvalue:
            if p_min == p_left:
                p1_startIndex = p1_startIndex # No change
                p1 = p11
                p2 = p12
            elif p_min == p_center:
                p1_startIndex = p1_startIndex + len(p1//2)
                p1 = p12
                p2 = p21
            elif p_min == p_right:
                p1_startIndex = p1_startIndex + len(p1)
                p1 = p21
                p2 = p22
    return p1_startIndex + len(p1)

def detectChange_AvgSeries(signals:np.ndarray, windowSize:int, pvalue:float, testingFunction:Callable, return_pvalues:bool=False, show_progress_bar:bool=True, progressBarPos:int=None, **kwargs)->List[int]:
    """Detect change points in a signal through application of statistical tests with sliding windows. When a change is detected, the exact location is investigated through recursive applications of statistical tests.

    In each step, the average of the computed pvalue over all levels of the signal is considered. This is used, e.g, to detect change points using the J-Measure for every pair of activities.

    Args:
        signals (np.ndarray): The signal to detect changes in. The considered p-value is the average of the p-values of the statistical test for each level of the signal.
        windowSize (int): The size of the sliding window for the statistical test; i.e. the size of the compared populations.
        pvalue (float): The p-value threshold, under which a pvalue indicates a change point.
        testingFunction (Callable): The testing function used to compare populations.
        return_pvalues (bool, optional): Configures whether the computed pvalues should be returned. Defaults to False.
        show_progress_bar (bool, optional): Configures whether a progress bar should be shown. Defaults to True.
        progressBarPos (int, optional): The `pos` argument for tqdm progress bars. In which line to print the progress bar. Defaults to None.
        **kwargs: Additional arguments to pass to the testing function.

    Raises:
        Exception: An exception is raised, if the different levels of the signal are not of the same length.

    Returns:
        List[int]: A list of detected change point indices. If `return_pvalues` is True, the computed pvalues are also returned.
    """    

    for i in range(len(signals)-1):
        if len(signals[i]) != len(signals[i+1]):
            raise Exception("Signals of inequal length in Average Series Recursive Bisection application!")
    # As many pvals as the first signal has (all should be equal)
    sig_length = len(signals[0])
    pvals = np.ones(sig_length)
    changepoints = []
    # Shift 2 windows of size `windowSize` over the signal and apply the given Test
    # We use skipuntil instead of e.g. using a while loop so that we still get the pvalues of the populations that we skip, for graphing
    skipuntil = -1
    progress = None
    if show_progress_bar:
        progress = makeProgressBar(num_iters=sig_length-(2*windowSize), message="Applying Recursive Bisection. Traces Completed", position=progressBarPos)

    def calc_avg_pval(window1, window2):
        pvals = []
        win1 = np.swapaxes(window1, 0,1)
        win2 = np.swapaxes(window2, 0,1)
        for i in range(len(win1)):
            pval = _getPValue(testingFunction(win1[i],win2[i]))
            pvals.append(pval)
        return np.mean(pvals)


    for i in range(sig_length-(2*windowSize)):

        # Flip Axes to access "x (time)" axis instead of "Activity pairs" axis
        window1 = np.swapaxes(signals,0,1)[i:i+windowSize]
        window2 = np.swapaxes(signals,0,1)[i+windowSize:i+(2*windowSize)]

        pval = calc_avg_pval(window1,window2)
        pvals[i+windowSize] = pval
        # If this indicates a changepoint and we do not skip this part (due to a close previous change point)
        if pval < pvalue and i >= skipuntil:
            #Changepoint detected
            changepoints.append(
                # Apply recursive bisection to locate the change point
                _locateChange(pop1=window1, pop2=window2, baseindex=i,test=calc_avg_pval, pvalue=pvalue)
            )
            #Continue searching at the first index after the second population
            skipuntil = i + (2*windowSize)
        if progress is not None:
            progress.update()
    if progress is not None:
        progress.close()
    return changepoints if not return_pvalues else (changepoints, pvals)

def _extractAllJMeasures(log:EventLog, measure_window:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->np.ndarray:
    """A helper function used to compute the J Measure over all pairs of activities.

    Args:
        log (EventLog): The event log.
        measure_window (int, optional): The window size to use for the J-Measure extraction. If None, the average trace length is used. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether a progress bar should be shown. Defaults to True.
        progressBarPos (int, optional): The `pos` argument for tqdm progress bars. In which line to print the progress bar. Defaults to None.

    Returns:
        np.ndarray: The extracted J-Measure values. Dimensions: num_activities x len(log)
    """    
    
    activities = _getActivityNames(log, activityName_key)
    if show_progress_bar:
        progress=makeProgressBar(num_iters=len(activities)**2, message="Extracting J for Martjushev", position=progressBarPos)
    else:
        progress = None
    signals = np.empty(
        (len(activities)**2, len(log))
    )
    i = 0
    for act1 in activities:
        for act2 in activities:
            new_sig = extractJMeasure(log, act1,act2,measure_window, activityName_key)
            signals[i] = new_sig
            if progress is not None:
                progress.update()
            i += 1
    if progress is not None:
        progress.close()
    return signals

def _extractAllWindowCounts(log:EventLog, measure_window:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->np.ndarray:
    """A helper function used to compute the Window Count over all pairs of activities.

    Args:
        log (EventLog): The event log.
        measure_window (int, optional): The window size to use for the Window Count extraction. If None, the average trace length is used. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether a progress bar should be shown. Defaults to True.
        progressBarPos (int, optional): The `pos` argument for tqdm progress bars. In which line to print the progress bar. Defaults to None.

    Returns:
        np.ndarray: The extracted Window Count values. Dimensions: num_activities x len(log)
    """
    
    activities = _getActivityNames(log, activityName_key)
    if show_progress_bar:
        progress=makeProgressBar(num_iters=len(activities)**2, message="Extracting WC for Martjushev", position=progressBarPos)
    else:
        progress=None
    signals = np.empty(
        (len(activities)**2, len(log))
    )
    i = 0
    for act1 in activities:
        for act2 in activities:
            new_sig = extractWindowCount(log, act1,act2,measure_window, activityName_key)
            signals[i] = new_sig
            if progress is not None:
                progress.update()
            i += 1
    if progress is not None:
        progress.close()
    return signals

def detectChange_JMeasure_KS(log:EventLog, windowSize:int, pvalue:float, return_pvalues:bool=False, measure_window:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->Union[List[int], Tuple[List[int], np.ndarray]]:
    """Apply Concept Drift Detection using the J-Measure and the Kolmogorov-Smirnov test.

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        windowSize (int): The window size to use for sliding window statistical testing.
        pvalue (float): The p-value threshold for statistical testing. If the p-value of the statistical test is below this threshold, a changepoint is detected.
        return_pvalues (bool, optional): If True, the p-values of the statistical tests are returned. Defaults to False.
        measure_window (int, optional): The window size to use for the measure extraction. If `None`, defaults to average trace length in the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Returns:
        Union[List[int], Tuple[List[int], np.ndarray]]: A list of detected change points. If `return_pvalues` is True, a tuple containing the list of change points and the p-values of the statistical tests.
    """
    
    signals = _extractAllJMeasures(log,measure_window,activityName_key, show_progress_bar=show_progress_bar, progressBarPos=progressBarPos)
    return detectChange_AvgSeries(signals, windowSize, pvalue, stats.ks_2samp, return_pvalues, show_progress_bar=show_progress_bar, progressBarPos=progressBarPos)

def detectChange_WindowCount_KS(log:EventLog, windowSize:int, pvalue:float, return_pvalues:bool=False, measure_window:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->Union[List[int], Tuple[List[int], np.ndarray]]:
    """Apply Concept Drift Detection using the Window Count and the Kolmogorov-Smirnov test.

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        windowSize (int): The window size to use for sliding window statistical testing.
        pvalue (float): The p-value threshold for statistical testing. If the p-value of the statistical test is below this threshold, a changepoint is detected.
        return_pvalues (bool, optional): If True, the p-values of the statistical tests are returned. Defaults to False.
        measure_window (int, optional): The window size to use for the measure extraction. If `None`, defaults to average trace length in the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Returns:
        Union[List[int], Tuple[List[int], np.ndarray]]: A list of detected change points. If `return_pvalues` is True, a tuple containing the list of change points and the p-values of the statistical tests.
    """
    
    signals = _extractAllWindowCounts(log,measure_window,activityName_key, show_progress_bar=show_progress_bar, progressBarPos=progressBarPos)
    return detectChange_AvgSeries(signals, windowSize, pvalue, stats.ks_2samp, return_pvalues, show_progress_bar=show_progress_bar, progressBarPos=progressBarPos)

def detectChange_JMeasure_MU(log:EventLog, windowSize:int, pvalue:float, return_pvalues:bool=False, measure_window:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->Union[List[int], Tuple[List[int], np.ndarray]]:
    """Apply Concept Drift Detection using the J-Measure and the Mann-Whitney U-Test.

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        windowSize (int): The window size to use for sliding window statistical testing.
        pvalue (float): The p-value threshold for statistical testing. If the p-value of the statistical test is below this threshold, a changepoint is detected.
        return_pvalues (bool, optional): If True, the p-values of the statistical tests are returned. Defaults to False.
        measure_window (int, optional): The window size to use for the measure extraction. If `None`, defaults to average trace length in the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Returns:
        Union[List[int], Tuple[List[int], np.ndarray]]: A list of detected change points. If `return_pvalues` is True, a tuple containing the list of change points and the p-values of the statistical tests.
    """    
    
    signals = _extractAllJMeasures(log,measure_window,activityName_key, show_progress_bar=show_progress_bar, progressBarPos=progressBarPos)
    return detectChange_AvgSeries(signals, windowSize, pvalue, stats.mannwhitneyu, return_pvalues, show_progress_bar=show_progress_bar, progressBarPos=progressBarPos)

def detectChange_WindowCount_MU(log:EventLog, windowSize:int, pvalue:float, return_pvalues:bool=False, measure_window:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None)->Union[List[int], Tuple[List[int], np.ndarray]]:
    """Apply Concept Drift Detection using the Window Count and the Mann-Whitney U-Test.

    Args:
        log (EventLog): The log on which to apply the concept drift detection.
        windowSize (int): The window size to use for sliding window statistical testing.
        pvalue (float): The p-value threshold for statistical testing. If the p-value of the statistical test is below this threshold, a changepoint is detected.
        return_pvalues (bool, optional): If True, the p-values of the statistical tests are returned. Defaults to False.
        measure_window (int, optional): The window size to use for the measure extraction. If `None`, defaults to average trace length in the log. Defaults to None.
        activityName_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        show_progress_bar (bool, optional): Configures whether or not to show a progress bar. Defaults to True.
        progressBarPos (int, optional): The `pos` parameter for tqdm progress bars. The "line" in which to show the bar. Defaults to None.

    Returns:
        Union[List[int], Tuple[List[int], np.ndarray]]: A list of detected change points. If `return_pvalues` is True, a tuple containing the list of change points and the p-values of the statistical tests.
    """
    
    signals = _extractAllWindowCounts(log,measure_window,activityName_key, show_progress_bar=show_progress_bar, progressBarPos=progressBarPos)
    return detectChange_AvgSeries(signals, windowSize, pvalue, stats.mannwhitneyu, return_pvalues, show_progress_bar=show_progress_bar, progressBarPos=progressBarPos)


# def detectChange_JMeasure_KS(log:EventLog, windowSize:int, pvalue:float, return_pvalues:bool=False, measure_window:int=None, activityName_key:str=xes.DEFAULT_NAME_KEY, show_progress_bar:bool=True, progressBarPos:int=None):
#     activities = _getActivityNames(log)
#     if show_progress_bar:
#         progress = makeProgressBar(pow(len(activities),2),"extracting j for recursive bisection algorithm, activity pairs completed ", position=progressBarPos)
#     else:
#         progress = None
#     sig = np.zeros((pow(len(activities),2), len(log)), ) # Axes will be swapped soon so sig[:x] splits based on time
#     i = 0
#     for act1 in activities:
#         for act2 in activities:
#             js = extractJMeasure(log, act1, act2)
#             sig[i] = js
#             progress.update()
#             i += 1
#     # Flip axes
#     sig = np.swapaxes(sig, 0,1)
#     def _getPValue(res)->float:
#         if isinstance(res,Number):
#             return res
#         else:
#             try:
#                 return res.pvalue
#             except:
#                 raise Exception("Statistical Test Result does not match criteria; Need either a float or an object with pvalue attribute")
#     def _applyAvgPVal(window1, window2, testingFunc):
#         pvals = []
#         w1 = np.swapaxes(window1, 0,1)
#         w2 = np.swapaxes(window2, 0,1)
#         for i in range(len(w1)):
#             pval = _getPValue(testingFunc(w1[i],w2[i]))
#             pvals.append(pval)
#         return np.mean(pvals)
#     cp, pvals = statisticalTesting_RecursiveBisection(sig, windowSize, pvalue, lambda x,y:_applyAvgPVal(x,y,stats.ks_2samp), return_pvalues)
#     progress.close()
#     return cp,pvals

import enum
import math
from multiprocessing import Pool, RLock, freeze_support, cpu_count
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from cdrift.approaches import earthmover, bose, martjushev

#Maaradji
from cdrift.approaches import maaradji as runs

# Zheng
from cdrift.approaches.zheng import applyMultipleEps

#Process Graph CPD
from cdrift.approaches import process_graph_metrics as pm

# Helper functions and evaluation functions
from cdrift import evaluation
from cdrift.utils import helpers

#Misc
import os
from datetime import datetime
from colorama import Fore
from tqdm import tqdm
from pathlib import Path
from itertools import product

# Enum of approaches
class Approaches(enum.Enum):
    BOSE = "Bose"
    MARTJUSHEV = "Martjushev"
    MARTJUSHEV_ADWIN = "Martjushev ADWIN"
    EARTHMOVER = "Earthmover"
    MAARADJI = "Maaradji"
    PROCESS_GRAPHS = "ProcessGraph"
    ZHENG = "Zheng"


#################################
############ SETTINGS ###########
#################################

# Which appraoches to test
DO_APPROACHES = {
    Approaches.BOSE: True,
    Approaches.MARTJUSHEV: False, # Replaced by ADWIN
    Approaches.MARTJUSHEV_ADWIN: True,
    Approaches.EARTHMOVER: True,
    Approaches.MAARADJI: True,
    Approaches.PROCESS_GRAPHS: True,
    Approaches.ZHENG: True
}

# Use a single bar for all (Showing how many algorithm applications are finished) or separate bars for each PCD instance
DO_SINGLE_BAR = True

# Take the results and make a pareto front from it
DO_PARETO_FRONT = False

# Number of cores to use for the multiprocessing
NUM_CORES = cpu_count() - 2

#################################
############ HELPERS ############
#################################

def calcDurationString(startTime, endTime):
    """
        Formats start and endtime to duration in hh:mm:ss format
    """
    elapsed_time = math.floor(endTime - startTime)
    return datetime.strftime(datetime.utcfromtimestamp(elapsed_time), '%H:%M:%S')

def calcDurFromSeconds(seconds):
    """
        Formats ellapsed seconds into hh:mm:ss format
    """
    seconds = math.floor(seconds)
    return datetime.strftime(datetime.utcfromtimestamp(seconds), '%H:%M:%S')

def plotPvals(pvals, changepoints, actual_changepoints, path, xlabel="", ylabel="", autoScale:bool=False):
    """
        Plots a series of p-values with detected and known change points and saves the figure
        args:
            - pvals
                List or array of p-values to be plotted
            - changepoints
                List of indices where change points were detected
            - actual_changepoints
                List of indices of actual change points
            - path
                The savepath of the generated image
            - xlabel
                Label of x axis
            - ylabel
                Label of y axis
            - autoScale
                Boolean whether y axis should autoscale by matplotlib (True) or be limited (0,max(pvals)+0.1) (False)
    """
    # Plotting Configuration
    fig = plt.figure(figsize=(10,4))
    plt.plot(pvals)
    # Not hardcoded 0-1 because of earthmovers distance (and +.1 so 1 is also drawn)
    if not autoScale:
        plt.ylim(0,max(pvals)+.1)
    for cp in changepoints:
        plt.axvline(x=cp, color='red', alpha=0.5)
    for actual_cp in actual_changepoints:
        plt.axvline(x=actual_cp, color='gray', alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{path}")
    plt.close()

#################################
##### Evaluation Functions ######
#################################

def testBose(filepath, window_size, step_size, F1_LAG, cp_locations, position=None, show_progress_bar=True):
    j_dur = 0
    wc_dur = 0

    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    j_start = default_timer()
    # pvals_j = bose.detectChange_JMeasure_KS(log, window_size, show_progress_bar=show_progress_bar, progressBarPos=position)
    # cp_j = bose.visualInspection(pvals_j, window_size)
    pvals_j = bose.detectChange_JMeasure_KS_Step(log, window_size, step_size=step_size, show_progress_bar=show_progress_bar, progressBarPos=position)
    cp_j = bose.visualInspection_Step(pvals_j, window_size, step_size)
    j_dur = default_timer() - j_start

    wc_start = default_timer()
    # pvals_wc = bose.detectChange_WC_KS(log, window_size, show_progress_bar=show_progress_bar, progressBarPos=position)
    # cp_wc = bose.visualInspection(pvals_wc, window_size)
    pvals_wc = bose.detectChange_WC_KS_Step(log, window_size, step_size=step_size, show_progress_bar=show_progress_bar, progressBarPos=position)
    cp_wc = bose.visualInspection_Step(pvals_wc, window_size, step_size)
    wc_dur = default_timer() - wc_start

    durStr_J = calcDurFromSeconds(j_dur)
    durStr_WC = calcDurFromSeconds(wc_dur)

    new_entry_j = {
        'Algorithm':"Bose J",
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'Window Size': window_size,
        'SW Step Size': step_size,
        'Detected Changepoints': cp_j,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp_j, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_j, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr_J
    }
    new_entry_wc = {
        'Algorithm':"Bose WC", 
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'Window Size': window_size,
        'SW Step Size': step_size,
        'Detected Changepoints': cp_wc,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp_wc, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_wc, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr_WC
    }

    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame([new_entry_j, new_entry_wc]).to_csv(Path("Reproducibility_Intermediate_Results", Approaches.BOSE.value,f"{logname}_WIN{window_size}.csv"), index=False)

    return [new_entry_j, new_entry_wc]

def testMartjushev(filepath, window_size, F1_LAG, cp_locations, position=None, show_progress_bar=True):
    PVAL = 0.55
    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    j_start = default_timer()
    rb_j_cp = martjushev.detectChange_JMeasure_KS(log, window_size, PVAL, return_pvalues=False, show_progress_bar=show_progress_bar, progressBarPos=position)
    j_dur = default_timer() - j_start

    wc_start = default_timer()
    rb_wc_cp = martjushev.detectChange_WindowCount_KS(log, window_size, PVAL, return_pvalues=False, show_progress_bar=show_progress_bar, progressBarPos=position)
    wc_dur = default_timer() - wc_start
    
    durStr_J = calcDurFromSeconds(j_dur)
    durStr_WC = calcDurFromSeconds(wc_dur)

    new_entry_j = {
        'Algorithm':"Martjushev J", 
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'P-Value': PVAL,
        'Window Size': window_size,
        'Detected Changepoints': rb_j_cp,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=rb_j_cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=rb_j_cp, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr_J
    }
    new_entry_wc = {
        'Algorithm':"Martjushev WC", 
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'P-Value': PVAL,
        'Window Size': window_size,
        'Detected Changepoints': rb_wc_cp,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=rb_wc_cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=rb_wc_cp, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr_WC
    }
    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame([new_entry_j, new_entry_wc]).to_csv(Path("Reproducibility_Intermediate_Results", Approaches.MARTJUSHEV.value, f"{logname}_WIN{window_size}.csv"), index=False)
    return [new_entry_j, new_entry_wc]

def testMartjushev_ADWIN(filepath, min_window, max_window, pvalue, step_size, F1_LAG, cp_locations, position=None, show_progress_bar=True):
    log = helpers.importLog(filepath, verbose=False)

    if len(log) <= min_window:
        # If the log is too short, we can't use the ADWIN algorithm because even the initial windows do not fit
        return np.NaN

    logname = filepath.split('/')[-1].split('.')[0]

    j_start = default_timer()
    adwin_j_cp = martjushev.detectChange_ADWIN_JMeasure_KS(log, min_window, max_window, pvalue, step_size, return_pvalues=False, show_progress_bar=show_progress_bar, progressBarPos=position)
    j_dur = default_timer() - j_start

    wc_start = default_timer()
    adwin_wc_cp = martjushev.detectChange_ADWIN_WindowCount_KS(log, min_window, max_window, pvalue, step_size, return_pvalues=False, show_progress_bar=show_progress_bar, progressBarPos=position)
    wc_dur = default_timer() - wc_start
    
    durStr_J = calcDurFromSeconds(j_dur)
    durStr_WC = calcDurFromSeconds(wc_dur)

    new_entry_j = {
        'Algorithm':"Martjushev ADWIN J", 
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'P-Value': pvalue,
        'Min Adaptive Window': min_window,
        'Max Adaptive Window': max_window,
        'ADWIN Step Size': step_size,
        'Detected Changepoints': adwin_j_cp,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=adwin_j_cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=adwin_j_cp, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr_J
    }
    new_entry_wc = {
        'Algorithm':"Martjushev ADWIN WC", 
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'P-Value': pvalue,
        'Min Adaptive Window': min_window,
        'Max Adaptive Window': max_window,
        'ADWIN Step Size': step_size,
        'Detected Changepoints': adwin_wc_cp,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=adwin_wc_cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=adwin_wc_cp, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr_WC
    }
    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame([new_entry_j, new_entry_wc]).to_csv(Path("Reproducibility_Intermediate_Results", Approaches.MARTJUSHEV_ADWIN.value, f"{logname}_MINW{min_window}_MAXW{max_window}.csv"), index=False)

    return [new_entry_j, new_entry_wc]

def testEarthMover(filepath, window_size, step_size, F1_LAG, cp_locations, position, show_progress_bar=True):
    LINE_NR = position

    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    startTime = default_timer()

    # Earth Mover's Distance
    traces = earthmover.extractTraces(log)
    # em_dists = earthmover.calculateDistSeries(traces, window_size, show_progressBar=show_progress_bar, progressBar_pos=LINE_NR)

    # cp_em = earthmover.visualInspection(em_dists, window_size)

    cp_em = earthmover.detect_change(log, window_size, step_size, show_progress_bar=show_progress_bar, progress_bar_pos=LINE_NR)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #
    new_entry = {
        'Algorithm':"Earth Mover's Distance", 
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'Window Size': window_size,
        'SW Step Size': step_size,
        'Detected Changepoints': cp_em,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp_em, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_em, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr
    }

    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame([new_entry]).to_csv(Path("Reproducibility_Intermediate_Results", Approaches.EARTHMOVER.value, f"{logname}_WIN{window_size}.csv"), index=False)

    return [new_entry]

def testMaaradji(filepath, window_size, step_size, F1_LAG, cp_locations, position, show_progress_bar=True):

    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    startTime = default_timer()

    # cp_runs = runs.detectChangepoints(log,window_size, pvalue=0.05, return_pvalues=False, show_progress_bar=show_progress_bar,progressBar_pos=position)
    cp_runs = runs.detectChangepoints_Stride(log, window_size, step_size, pvalue=0.05, return_pvalues=False, show_progress_bar=show_progress_bar, progressBar_pos=position)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #

    new_entry = {
        'Algorithm':"Maaradji Runs",
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'Window Size': window_size,
        'SW Step Size': step_size,
        'Detected Changepoints': cp_runs,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp_runs, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_runs, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr
    }
    
    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame([new_entry]).to_csv(Path("Reproducibility_Intermediate_Results", Approaches.MAARADJI.value, f"{logname}_WIN{window_size}.csv"), index=False)
    return [new_entry]

def testGraphMetrics(filepath, min_window, max_window, pvalue, F1_LAG, cp_locations, position=None, show_progress_bar=True):
    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    startTime = default_timer()

    cp = pm.detectChange(log, min_window, max_window, pvalue=pvalue, show_progress_bar=show_progress_bar,progressBarPosition=position)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #

    new_entry = {
        'Algorithm':"Process Graph Metrics", 
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'P-Value': pvalue,
        'Min Adaptive Window': min_window,
        'Max Adaptive Window': max_window,
        'Detected Changepoints': cp,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr
    }

    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame([new_entry]).to_csv(Path("Reproducibility_Intermediate_Results", Approaches.PROCESS_GRAPHS.value, f"{logname}_P{pvalue}_MINW{min_window}_MAXW{max_window}.csv"), index=False)
    return [new_entry]

def testZhengDBSCAN(filepath, mrid, epsList, F1_LAG, cp_locations, position, show_progress_bar=True):
    # candidateCPDetection is independent of eps, so we can calculate the candidates once and use them for multiple eps!
    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    startTime = default_timer()
    
    # CPD #
    cps = applyMultipleEps(log, mrid=mrid, epsList=epsList, show_progress_bar=show_progress_bar, progressPos=position)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #

    ret = []
    for eps in epsList:
        cp = cps[eps]

        new_entry = {
            'Algorithm':"Zheng DBSCAN", 
            'Log Source': Path(filepath).parent.name,
            'Log': logname,
            'MRID': mrid,
            'Epsilon': eps,
            'Detected Changepoints': cp,
            'Actual Changepoints for Log': cp_locations,
            'F1-Score': evaluation.F1_Score(detected=cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
            'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp, actual_changepoints=cp_locations, lag=F1_LAG),
            'Duration': durStr
        }
        ret.append(new_entry)
    if os.path.exists("Reproducibility_Intermediate_Results"):
        for entry in ret:
            pd.DataFrame([entry]).to_csv(Path("Reproducibility_Intermediate_Results", Approaches.ZHENG.value, f"{logname}_MRID{mrid}_EPS{str(entry['Epsilon']).replace('.','_')}.csv"), index=False)
    return ret

def testSomething(arg):
    """Wrapper for testing functions, as for the multiprocessing pool, one can only use one function, not multiple

    Args:
        idx (int): Position-Index for the progress bar of the evaluation
        vals (Tuple[str,List]): Tuple of name of the approach, and its parameter values
    """

    idx, vals, show_bar = arg

    name, arguments = vals
    if name == Approaches.BOSE:
        return testBose(*arguments, position=idx, show_progress_bar=show_bar)
    elif name == Approaches.MARTJUSHEV:
        return testMartjushev(*arguments, position=idx, show_progress_bar=show_bar)
    elif name == Approaches.MARTJUSHEV_ADWIN:
        return testMartjushev_ADWIN(*arguments, position=idx, show_progress_bar=show_bar)
    elif name == Approaches.EARTHMOVER:
        return testEarthMover(*arguments, position=idx, show_progress_bar=show_bar)
    elif name == Approaches.MAARADJI:
        return testMaaradji(*arguments, position=idx, show_progress_bar=show_bar)
    elif name == Approaches.PROCESS_GRAPHS:
        return testGraphMetrics(*arguments, position=idx, show_progress_bar=show_bar)
    elif name == Approaches.ZHENG:
        return testZhengDBSCAN(*arguments, position=idx, show_progress_bar=show_bar)

def main():
    #Evaluation Parameters
    F1_LAG = 200

    for approach, do in DO_APPROACHES.items():
        if do:
            Path("Reproducibility_Intermediate_Results", approach.value).mkdir(parents=True, exist_ok=True)

    # Setup all Paths to logs alongside their change point locations
    logPaths_Changepoints = [
        (Path("EvaluationLogs","Bose", "bose_log.xes.gz").as_posix(), [1199, 2399, 3599, 4799]), # A change every 1200 cases, 6000 cases in total (Skipping 5999 because a change on the last case doesnt make sense)
    ]

    ceravolo_root = Path("EvaluationLogs","Ceravolo")
    for item in ceravolo_root.iterdir():
        _, _, _,num_cases, _ = item.stem.split("_")
        if int(num_cases) != 1000: # Only use logs of length 1000
            continue
        drift_indices = [(int(num_cases)//2) - 1] # "The first half of the stream is composed of the baseline model, and the second half is composed of the drifted model"
        logPaths_Changepoints.append((item.as_posix(), drift_indices))
    
    logPaths_Changepoints += [
        (item.as_posix(), [999,1999])
        for item in Path("EvaluationLogs","Ostovar").iterdir()
    ]

    # Parameter Settings #
    # Window Sizes that we test
    windowSizes       = [100, 200, 300, 400, 500,  600]
    maaradji_winsizes = [50, 100, 150, 200, 250, 300            ]
    
    # Parameters for Adaptive Window Approaches
    min_window_sizes = [100,200,300,400]
    max_window_sizes = [400,500,600,700]
    window_pairs = [
        (w_min,w_max)
        for w_min, w_max in product(min_window_sizes, max_window_sizes)
        if w_min <= w_max
    ]

    step_sizes = [10,20,50]
    mart_pvalues = [0.4]
    pgraph_pvalues = [0.1, 0.05, 0.005, 0.0001] #0.0001 used in the paper


    # Special Parameters for the approach by Zheng et al.
    mrids = [300, 400, 500, 600, 700, 800]
    eps_modifiers = [0.1, 0.2, 0.3, 0.4, 0.5] # use x * mrid as epsilon, as the paper suggests
    eps_mrid_pairs = [
        (mrid,[mEps*mrid  for mEps in eps_modifiers]) 
        for mrid in mrids
    ]

    SW_STEP_SIZES = [2]

    bose_args             =  [(path, winSize, step_size,       F1_LAG, cp_locations)     for path, cp_locations in logPaths_Changepoints for winSize             in windowSizes       for step_size in SW_STEP_SIZES]
    martjushev_args       =  [(path, winSize,                  F1_LAG, cp_locations)     for path, cp_locations in logPaths_Changepoints for winSize             in windowSizes       ]
    martjushev_adwin_args =  [(path, w_min, w_max, pval, step, F1_LAG, cp_locations)     for path, cp_locations in logPaths_Changepoints for w_min, w_max        in window_pairs      for pval in mart_pvalues   for step in step_sizes ]
    em_args               =  [(path, winSize, step_size,       F1_LAG, cp_locations)     for path, cp_locations in logPaths_Changepoints for winSize             in windowSizes       for step_size in SW_STEP_SIZES]
    maaradji_args         =  [(path, winSize, step_size,       F1_LAG, cp_locations)     for path, cp_locations in logPaths_Changepoints for winSize             in maaradji_winsizes for step_size in SW_STEP_SIZES]
    pgraph_args           =  [(path, w_min, w_max, pval,       F1_LAG, cp_locations)     for path, cp_locations in logPaths_Changepoints for w_min, w_max        in window_pairs      for pval in pgraph_pvalues                        ]
    zhengDBSCAN_args      =  [(path, mrid,    epsList,         F1_LAG, cp_locations)     for path, cp_locations in logPaths_Changepoints for mrid,epsList        in eps_mrid_pairs    ]

    arguments = ( [] # Empty list here so i can just comment out ones i dont want to do
        + ([ (Approaches.ZHENG, args)              for args in zhengDBSCAN_args         ] if DO_APPROACHES[Approaches.ZHENG             ]         else [])
        + ([ (Approaches.MAARADJI, args)           for args in maaradji_args            ] if DO_APPROACHES[Approaches.MAARADJI          ]         else [])
        + ([ (Approaches.PROCESS_GRAPHS, args)     for args in pgraph_args              ] if DO_APPROACHES[Approaches.PROCESS_GRAPHS    ]         else [])
        + ([ (Approaches.EARTHMOVER, args)         for args in em_args                  ] if DO_APPROACHES[Approaches.EARTHMOVER        ]         else [])
        + ([ (Approaches.BOSE, args)               for args in bose_args                ] if DO_APPROACHES[Approaches.BOSE              ]         else [])
        + ([ (Approaches.MARTJUSHEV, args)         for args in martjushev_args          ] if DO_APPROACHES[Approaches.MARTJUSHEV        ]         else [])
        + ([ (Approaches.MARTJUSHEV_ADWIN, args)   for args in martjushev_adwin_args    ] if DO_APPROACHES[Approaches.MARTJUSHEV_ADWIN  ]         else [])
    )

    # Shuffle the Tasks
    np.random.shuffle(arguments)

    time_start = default_timer()

    freeze_support()  # for Windows support
    tqdm.set_lock(RLock())  # for managing output contention
    results = []

    with Pool(NUM_CORES,initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        _args = [(idx, args, not DO_SINGLE_BAR) for idx, args in enumerate(arguments)]
        if DO_SINGLE_BAR:
            for result in tqdm(p.imap(testSomething, _args), desc="Calculating.. Completed PCD Instances", total=len(_args)):
                results.append(result)
        else:
            results = p.map(testSomething, _args)

    # Remove NaN return values from the results, source is Martjushev_ADWIN if the log is too short for the chosen windows
    results = [result for result in results if not result == np.NaN]

    elapsed_time = math.floor(default_timer() - time_start)
    # Write instead of print because of progress bars (although it shouldnt be a problem because they are all done)
    elapsed_formatted = datetime.strftime(datetime.utcfromtimestamp(elapsed_time), '%H:%M:%S')
    tqdm.write(f"{Fore.GREEN}The execution took {elapsed_formatted}{Fore.WHITE}")


    flattened_results = [res for function_return in results for res in function_return]
    df = pd.DataFrame(flattened_results)
    df.to_csv("algorithm_results.csv", index=False)

    if DO_PARETO_FRONT:
        # Convert String Duration Column to Datetime
        from cdrift.utils.helpers import convertToTimedelta
        df['Duration'] = df['Duration'].apply(convertToTimedelta)

        dfs = [
            df[df['Algorithm'] == approach_name].copy(deep=True)
            for approach_name in df["Algorithm"].unique()
        ]

        # Generate Figures for the results
        fig = evaluation.scatter_f1_duration(dfs)

        fig.savefig("pareto-front.pdf",format="pdf",bbox_inches='tight')
        fig.savefig("pareto-front.png",format="png",bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    main()
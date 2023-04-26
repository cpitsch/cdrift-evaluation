import matplotlib.pyplot as plt
import pandas as pd
from cdrift import evaluation
from cdrift.utils.helpers import readCSV_Lists, convertToTimedelta, importLog
import numpy as np
from datetime import datetime
from statistics import mean, harmonic_mean, stdev
from scipy.stats import iqr
from typing import List, Tuple
import seaborn as sns
import re
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from tqdm.auto import tqdm


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

mapping_ostovar_to_shortnames = {
    "ConditionalMove": 'cm',
    "ConditionalRemoval": 'cre',
    "ConditionalToSequence": 'cf',
    "Frequency": 'fr',
    "Loop": 'lp',
    "ParallelMove": 'pm',
    "ParallelRemoval": 'pre',
    "ParallelToSequence": 'pl',
    "SerialMove": 'sm',
    "SerialRemoval": 'sre',
    "Skip": 'cb',
    "Substitute": 'rp',
    "Swap": 'sw',
}

shorter_names = {
    "Zheng DBSCAN": "RINV",
    "ProDrift": "ProDrift",
    "Maaradji Runs": "ProDrift",
    "Bose J": "J-Measure",
    "Bose WC": "Window Count",
    "Earth Mover's Distance": "EMD",
    "Process Graph Metrics": "PGM",
    "Martjushev ADWIN J": "ADWIN J",
    "Martjushev ADWIN WC": "ADWIN WC",
    "LCDD": "LCDD"
}

used_parameters = {
    "Bose J": ["Window Size", "SW Step Size"],
    "Bose WC": ["Window Size", "SW Step Size"],
    "Martjushev ADWIN J": ["Min Adaptive Window", "Max Adaptive Window", "P-Value", "ADWIN Step Size"],
    "Martjushev ADWIN WC": ["Min Adaptive Window", "Max Adaptive Window", "P-Value", "ADWIN Step Size"],
    "ProDrift": ["Window Size", "SW Step Size"],
    "Earth Mover's Distance": ["Window Size", "SW Step Size"],
    "Process Graph Metrics": ["Min Adaptive Window", "Max Adaptive Window", "P-Value"],
    "Zheng DBSCAN": ["MRID", "Epsilon"],
    "LCDD": ["Complete-Window Size", "Detection-Window Size", "Stable Period"]
}

used_parameters = {
    shorter_names[name]: used_parameters[name]
    for name in used_parameters.keys()
}

def split_by_name(df):
    return [
        (alg, df[df["Algorithm"] == alg].copy())
        for alg in df["Algorithm"].unique()
    ]

def map_row_to_cp(row):
    logname = row["Log"]
    if row["Log Source"] == "Ceravolo":
        return logname.split("_")[-1]
    elif row["Log Source"] == "Ostovar":
        change_pattern = logname.split("_")[-1]
        if change_pattern.isnumeric(): # This log has a noise level; Use the second to last string instead
            change_pattern = logname.split("_")[-2]
        return mapping_ostovar_to_shortnames[change_pattern]
    elif row["Log Source"] == "Bose":
        return None
    else:
        raise ValueError(f"Unknown Log Source for Versatility: {row['Log Source']}; Is this Log meant to be used for a Versatility Evaluation?")

def map_row_to_noise_level(row):
    logname = row["Log"]
    if row["Log Source"] == "Ceravolo":
        return re.search('noise([0-9]*)_', logname).group(1)
    elif row["Log Source"] == "Ostovar":
        # return mapping_ostovar_to_shortnames[logname.split("_")[-1]]
        last_member = logname.split('_')[-1]
        if last_member.isnumeric():
            return last_member
        else:
            return '0'
    elif row["Log Source"] == "Bose":
        return None
    else:
        raise ValueError(f"Unknown Log Source for Noise Level: {row['Log Source']}; Is this Log meant to be used for a Robustness Evaluation?")


def preprocess(_df):
    df = _df.copy()
    ## Exclude Martjushev WC (in favor of LCDD)
    df = df[df["Algorithm"] != "Martjushev ADWIN WC"]

    ## Rename Approaches to shorter names
    df["Algorithm"] = df["Algorithm"].map(lambda x: shorter_names.get(x, x))

    ## Only use Atomic Logs
    atomic_logs = {
        log
        for _, (source, log) in df[["Log Source", "Log"]].iterrows()
        if (source.lower() == "ostovar" and log.lower().startswith("atomic")) # Only use atomic logs from ostovar
        or (log.lower().startswith("bose")) # Use bose log
        or (source.lower() == "ceravolo" and set(log.lower().split("_")[-1]) != {'i', 'o', 'r'}) # Exclude all composite (IOR, ROI, etc.) logs from ceravolo
    }
    df = df[df["Log"].isin(atomic_logs)]

    # Add a column stating the change pattern that is applied and one for the noise level
    # Only use the Ceravolo and Ostovar Logs as Bose has only 1 log, i.e., no noise levels
    df["Change Pattern"] = df.apply(lambda x: map_row_to_cp(x), axis=1)
    df["Noise Level"] = df.apply(lambda x: map_row_to_noise_level(x), axis=1)

    return df

def evaluate(csv_path="algorithm_results.csv", out_path="Evaluation_Results", lag_window:int=200, min_support:int=1, verbose:bool = False):
    """Evaluate the results of the algorithms.

    Args:
        csv_path (str, optional): Path to results csv file. Defaults to "algorithm_results.csv".
        out_path (str, optional): Directory in which to save results. Defaults to "Evaluation_Results".
        lag_window (int, optional): Max acceptable distance to a ground truth change point to be classified as true positive. Defaults to 200.
        min_support (int, optional): Minimum support for latency calculation --> a parameter setting must have at least 3 instances where a true positive was detected before it can be deemed the "best parameter setting" for latency calculation. Defaults to 1.
        verbose (bool, optional): Whether to print information about the data. Defaults to False.

    Raises:
        ValueError: CSV Contains duplicate rows
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Preprocess
    df = readCSV_Lists(csv_path)
    df = preprocess(df)
    ALPHA_SORT_NAMES = sorted(list(df["Algorithm"].unique()))

    ## Calculate Log Lengths for Scalability
    _groupkeys = list(df.groupby(["Log Source", "Log"]).groups.keys())
    _logpaths = [Path("EvaluationLogs", source, f"{log}.xes.gz") for source, log in _groupkeys]

    loglengths = dict()
    loglengths_events = dict()

    for logpath in tqdm(_logpaths, "Calculating Log Lengths. Completed Logs: "):
        log = importLog(logpath.as_posix(), verbose=False)
        loglengths[logpath] = len(log)
        loglengths_events[logpath] = len([evt for case in log for evt in case])

    if verbose:
        print(f"Number of Logs: {len(loglengths.keys())}")
        print(f"Number of Cases: {sum(loglengths.values())}")
        print(f"Number of Events: {sum(loglengths_events.values())}")

        num_drifts = sum(
            len(group.iloc[0]["Actual Changepoints for Log"])
            for name, group in df.groupby(["Log Source", "Log"])
        )
        print("Total number of drifts:", num_drifts)

    dfs = split_by_name(df)

    if pd.read_csv(csv_path).duplicated().any():
        raise ValueError("CSV contains duplicate rows!")

    ## Split into Noisy/Noiseful Logs
    logs = zip(df["Log Source"], df["Log"])
    noiseless_logs = {
        log for source, log in logs if 
        (source == "Ostovar"  and not (log.endswith("_2") or log.endswith("_5"))) or
        (source == "Ceravolo" and log.split("_")[2]=="noise0") or
        source == "Bose"
    }

    df_noiseless = df[df["Log"].isin(noiseless_logs)]
    df_noisy = df[df["Log"].isin(noiseless_logs) == False]

    dfs_noisy = split_by_name(df_noisy)
    dfs_noiseless = split_by_name(df_noiseless)

    ## Define Color Scheme
    colors = ["#DB444B","#9A607F","#006BA2","#3EBCD2","#379A8B","#EBB434","#B4BA39","#D1B07C"]
    color_map = {ALPHA_SORT_NAMES[i]:colors[i] for i in range(0,len(ALPHA_SORT_NAMES))}

    analyze_change_pattern_distribution(df, out_path)

    # Accuracy
    accuracies, computed_accuracy_dicts, computed_precision_dicts, computed_recall_dicts, accuracy_best_param = calculate_accuracy_metric_df(df_noiseless, lag_window, show_progress_bar=True)
    plot_accuracy(computed_accuracy_dicts, computed_precision_dicts, computed_recall_dicts, accuracy_best_param, out_path, lag_window, colors, ALPHA_SORT_NAMES)

    # Latency
    latencies, scaled_latency_dicts, computed_latency_dicts, best_params_latency = calculate_latency(df_noiseless, lag_window, min_support=min_support, show_progress_bar=True)
    plot_latency(computed_latency_dicts, best_params_latency, colors, out_path, lag_window, ALPHA_SORT_NAMES)

    # Versatility
    ## Only use Ceravolo and Ostovar Logs because these are the only ones that have different change patterns.
    ## (Theoretically we could also drop all rows containing a nan in the change pattern column? This would be more general)
    df_v = df_noiseless[df_noiseless["Log Source"].isin(["Ceravolo", "Ostovar"])].copy(deep=True)
    versatilities, versatility_recall_dicts, best_params_versatility = calc_versatility(df_v, lag_window, show_progress_bar=True)
    plot_versatility(versatility_recall_dicts, best_params_versatility, out_path, lag_window, colors, ALPHA_SORT_NAMES)

    # Scalability
    scalabilities = calculate_scalability(df, show_progress_bar=True)
    plot_scalability(scalabilities, dfs, out_path, lag_window, colors, ALPHA_SORT_NAMES)
    df_seconds, seconds_per_case, seconds_per_event = calculate_rel_scalabilities(df, loglengths, loglengths_events)
    plot_rel_scalability(df_seconds, out_path, lag_window, colors, ALPHA_SORT_NAMES)

    # Parameter Sensitivity
    sensitivities = calculate_parameter_sensitivity(df, versatility_recall_dicts, computed_accuracy_dicts, scaled_latency_dicts, show_progress_bar=True)
    sensitivity_iqrs = calculate_parameter_sensitivity_iqr(sensitivities)
    plot_parameter_sensitivity(sensitivities, out_path, lag_window, colors, ALPHA_SORT_NAMES)

    # Robustness
    df_robust = df[df["Log Source"].isin(["Ceravolo", "Ostovar"])].copy(deep=True)
    means_ceravolo = calc_harm_means(df_robust[df_robust["Log Source"] == "Ceravolo"], min_support, lag_window, show_progress_bar=True)
    means_ostovar = calc_harm_means(df_robust[df_robust["Log Source"] == "Ostovar"], min_support, lag_window, show_progress_bar=True)
    robustness_ceravolo = convert_harm_mean_to_auc(means_ceravolo)
    robustness_ostovar = convert_harm_mean_to_auc(means_ostovar) 
    # Calculate final robustness score by calculating mean of ostovar and ceravolo performance
    robustnesses = {
        name: (robustness_ceravolo[name] + robustness_ostovar[name]) / 2
        for name in robustness_ceravolo.keys()
    }
    plot_robustness(means_ostovar, means_ceravolo, out_path, lag_window, color_map)



    # Save results to csv
    results_df = pd.DataFrame([
        {
            "Algorithm": name,
            "Accuracy": accuracies[name],
            "Latency": (1-latencies[name])*lag_window,
            "Versatility": versatilities[name],
            "Scalability": scalabilities[name]["str"],
            "Milliseconds per Case": seconds_per_case[name]*1000,
            "Milliseconds per Event": seconds_per_event[name]*1000,
            "Parameter Sensitivity (IQR)": sensitivity_iqrs[name],
            "Robustness": robustnesses[name]
        }
        for name in df["Algorithm"].unique()
    ])

    results_df.to_csv(f"{out_path}/evaluation_measures{lag_window}.csv", index=False)



def analyze_change_pattern_distribution(df, out_path):
    cp_count_results = { cp: dict() for cp in df["Change Pattern"].unique() if cp is not None } # Bose gets lost because it maps to none

    for name, group in df[["Log Source", "Log", "Change Pattern"]].drop_duplicates(inplace=False).groupby("Log Source"): # Only consider each logpath once (not once for each algorithm application)
        series = group["Change Pattern"].value_counts().to_dict()
        for cp, count in series.items():
            cp_count_results[cp][name] = count

    for key in cp_count_results.keys():
        cp_count_results[key]["Bose"] = 0
    cp_count_results["Mixed (Bose)"] = {"Ostovar": 0, "Ceravolo": 0, "Bose": 1}

    cp_counts = pd.DataFrame(cp_count_results).fillna(0).astype(int)

    # Analyze relative counts
    cp_counts_tr = cp_counts.transpose()
    total_logs = cp_counts.to_numpy().sum()
    cp_counts_tr["Relative Frequency"] = cp_counts_tr.apply(lambda row: row.sum() / total_logs * 100, axis=1)

    cp_counts_tr.to_csv(f"{out_path}/change_pattern_distribution.csv")

def calcAccuracy(df:pd.DataFrame, param_names:List[str], lag_window: int):
    """Calculates the Accuracy Metric for the given dataframe by grouping by the given parameters and calculating the mean accuracy

    Args:
        df (pd.DataFrame): The dataframe containing the results to be evaluated
        param_names (List[str]): The names of the parameters of this approach
        lag_window (int): The lag window to be used for the evaluation to determine true positives and false positives
    """

    f1s = dict()
    recalls = dict()
    precisions = dict()

    # Group by parameter values to calculate accuracy per parameter setting, over all logs
    for parameters, group in df.groupby(by=param_names):
        # Calculate Accuracy for this parameter setting
        ## --> F1-Score, but first collect all TP and FP
        tps = 0
        fps = 0
        positives = 0
        detected = 0

        # Collect TP FP, etc. 
        for index, row in group.iterrows():
            actual_cp = row["Actual Changepoints for Log"]
            detected_cp = row["Detected Changepoints"]
            tp, fp = evaluation.getTP_FP(detected_cp, actual_cp, lag_window)
            tps += tp
            fps += fp
            positives += len(actual_cp)
            detected += len(detected_cp)

        try:
            precisions[parameters] = tps / detected
        except ZeroDivisionError:
            precisions[parameters] = np.NaN
        
        try:
            recalls[parameters] = tps / positives
        except ZeroDivisionError:
            recalls[parameters] = np.NaN

        f1s[parameters] = harmonic_mean([precisions[parameters], recalls[parameters]]) # If either is nan, the harmonic mean is nan
    return (precisions, recalls, f1s)

def calculate_accuracy_metric_df(dataframe, lag_window, show_progress_bar: bool = False):
    computed_accuracy_dicts = dict()
    computed_precision_dicts = dict()
    computed_recall_dicts = dict()

    accuracy_best_param = dict()


    accuracies = dict()

    groups = dataframe.groupby(by="Algorithm")
    if show_progress_bar:
        groups = tqdm(groups, "Calculating Accuracy. Completed Algorithms: ")
    for name, a_df in groups:
        computed_precision_dicts[name], computed_recall_dicts[name], computed_accuracy_dicts[name] = calcAccuracy(a_df, used_parameters[name], lag_window)

        best_param = max(computed_accuracy_dicts[name],  key=lambda x: computed_accuracy_dicts[name][x])

        accuracy_best_param[name] = best_param

        # accuracies[name] = max(computed_accuracy_dicts[name].values())
        accuracies[name] = computed_accuracy_dicts[name][best_param]
    return (accuracies, computed_accuracy_dicts, computed_precision_dicts, computed_recall_dicts, accuracy_best_param)

def plot_accuracy(computed_accuracy_dicts, computed_precision_dicts, computed_recall_dicts, accuracy_best_param, out_path, lag_window, colors, order):
    accuracy_plot_df = pd.DataFrame(
        [
            {
                "Algorithm": name,
                "Metric": "F1-Score",
                "Value": computed_accuracy_dicts[name][accuracy_best_param[name]]
            }
            for name in computed_accuracy_dicts.keys()
        ] + [
            {
                "Algorithm": name,
                "Metric": "Precision",
                "Value": computed_precision_dicts[name][accuracy_best_param[name]]
            }
            for name in computed_accuracy_dicts.keys()
        ] + [
            {
                "Algorithm": name,
                "Metric": "Recall",
                "Value": computed_recall_dicts[name][accuracy_best_param[name]]
            }
            for name in computed_accuracy_dicts.keys()
        ]
    )


    palette=None #{"Precision": "#573deb", "F1-Score": "#ff0076", "Recall": "#ffa600"}
    palette = sns.color_palette(colors)
    plt.grid(zorder=0)
    ax = sns.barplot(x="Metric", y="Value", data=accuracy_plot_df, hue="Algorithm", palette=palette, hue_order=order,zorder =5)

    ax.figure.set_size_inches(11, 4)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(ax.get_xlabel(), size = 20)
    ax.set_ylabel("", size = 20)
    #ax.get_yaxis().set_visible(False)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0,prop={'size': 14})

    if not os.path.exists(f"{out_path}/Accuracy{lag_window}"):
        os.makedirs(f"{out_path}/Accuracy{lag_window}")
    plt.savefig(f"{out_path}/Accuracy{lag_window}/accuracy_c.pdf", bbox_inches="tight", format="pdf")

    ax = sns.pointplot(
        x="Algorithm",
        y="Value",
        data=accuracy_plot_df,
        hue="Metric",
        order=order,
        join=True,
    )
    ax.figure.set_size_inches(15, 4)
    plt.savefig(f"{out_path}/Accuracy{lag_window}/accuracy_points.pdf", bbox_inches="tight", format="pdf")

def calcLatencies(df, param_names, lag_window, show_progress_bar: bool = False):
    latencies = dict()
    for parameters, group in df.groupby(by=param_names):
        lags = []
        for index, row in group.iterrows():
            actual_cp = row["Actual Changepoints for Log"]
            detected_cp = row["Detected Changepoints"]
            assignments = evaluation.assign_changepoints(detected_cp, actual_cp, lag_window)
            for d, a in assignments:
                lags.append(abs(d-a))
        latencies[parameters] = lags
    return latencies

def calculate_latency(dataframe, lag_window, min_support=1, show_progress_bar: bool = False):
    latencies = dict() # Dict holding the best achieved latency per approach
    scaled_latency_dicts = dict() # Dict holding the scaled mean latency per approach per parameter setting
    computed_latency_dicts = dict() # Dict holding the raw list of detection lags per approach per parameter setting
    best_params_latency = dict() # Dict holding the best parameter setting per approach (the one that achieves the best latency)

    groups = dataframe.groupby(by="Algorithm")
    if show_progress_bar:
        groups = tqdm(groups, "Calculating Latency. Completed Algorithms: ")
    for name, a_df in groups:
        result = calcLatencies(a_df, used_parameters[name], lag_window)
        computed_latency_dicts[name] = result
        latency_scaled_dict = {
            param: 1-(mean(result[param])/lag_window) if len(result[param]) >= min_support else np.NaN
            for param in result.keys()
        }
        scaled_latency_dicts[name] = latency_scaled_dict
        best_param = max(latency_scaled_dict,  key=lambda x: latency_scaled_dict[x] if not np.isnan(latency_scaled_dict[x]) else -1)

        best_params_latency[name] = best_param
        latencies[name] = latency_scaled_dict[best_param]

    return (latencies, scaled_latency_dicts, computed_latency_dicts, best_params_latency)

def plot_latency(computed_latency_dicts, best_params_latency, colors, out_path, lag_window, order):
    latency_plot_df = pd.DataFrame([
    {
        "Algorithm": name,
        "Unscaled Latency": latency
    }
    for name in computed_latency_dicts.keys()
    for latency in computed_latency_dicts[name][best_params_latency[name]]
    ])

    ax = plt.subplots(figsize=(17, 4))

    palette = sns.color_palette(colors)

    ax = sns.barplot(x="Algorithm", y="Unscaled Latency", data=latency_plot_df,palette= palette, order=order)
    plt.ylabel("Latency")# [Traces]")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), size = 18)
    ax.set_ylabel(ax.get_ylabel(), size = 18)
    ax.figure.set_size_inches(14, 4)

    if not os.path.exists(f"{out_path}/Latency{lag_window}"):
        os.makedirs(f"{out_path}/Latency{lag_window}")
    plt.savefig(f"{out_path}/Latency{lag_window}/latency.pdf", bbox_inches="tight", format="pdf")

def calc_versatility(dataframe, lag_window, show_progress_bar: bool = False):

    versatility_recall_dicts = dict() # Map approach to a dictionary mapping param setting to mean recall over all change patterns
    versatilities = dict() # Map approach to versatility score
    best_params_versatility = dict() # Map approach to best param setting

    groups = dataframe.groupby(by="Algorithm")
    if show_progress_bar:
        groups = tqdm(groups, "Calculating Versatility. Completed Algorithms: ")

    for name, group in groups:
        recalls_of_this_approach = dict()
        for params, params_group in group.groupby(by=used_parameters[name]):
            recalls = dict()
            for change_pattern, cp_group in params_group.groupby(by="Change Pattern"):
                TPS = 0
                POSITIVES = 0
                # TP / TP+FN = TP / POSTIVES = Recall
                for index, row in cp_group.iterrows():
                    detected_changepoints = row["Detected Changepoints"]
                    actual_changepoints = row["Actual Changepoints for Log"]

                    tp, _ = evaluation.getTP_FP(detected_changepoints, actual_changepoints, lag_window)

                    TPS += tp
                    POSITIVES += len(actual_changepoints)

                # Recall of this algorithm for this change pattern:
                recall = TPS / POSITIVES if POSITIVES != 0 else np.NaN # Only the case if there are no actual changepoints, which should not be the case
                recalls[change_pattern] = recall
            recalls_of_this_approach[params] = recalls
        versatility_recall_dicts[name] = recalls_of_this_approach
        best_param = max(recalls_of_this_approach, key=lambda x: np.nanmean(list(recalls_of_this_approach[x].values()))) # .values() gives us all the recalls for all change patterns on this param setting
        best_params_versatility[name] = best_param
        versatilities[name] = np.nanmean(list(
            recalls_of_this_approach[best_param].values()
        ))
        
    return versatilities, versatility_recall_dicts, best_params_versatility

def plot_versatility(versatility_recall_dicts, best_params_versatility, out_path, lag_window, colors, order):
    df_vers_plot = pd.DataFrame([
        {
            "Algorithm": name,
            "Change Pattern": cp,
            "Versatility": versatility_recall_dicts[name][best_params_versatility[name]][cp]
        }
        for name in versatility_recall_dicts.keys()
        for cp in versatility_recall_dicts[name][best_params_versatility[name]].keys()
    ])


    fig,ax = plt.subplots(figsize=(20, 4))
    sns.barplot(x="Change Pattern", y="Versatility", data=df_vers_plot, hue="Algorithm", ax=ax, hue_order=order)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

    if not os.path.exists(f"{out_path}/Versatility{lag_window}"):
        os.makedirs(f"{out_path}/Versatility{lag_window}")
    plt.savefig(f"{out_path}/Versatility{lag_window}/versatility.pdf", bbox_inches="tight", format="pdf")

    plt.close('all')

    # As points
    change_patterns = sorted(list(df_vers_plot["Change Pattern"].unique()))
    ax = sns.pointplot(
        x="Change Pattern",
        y="Versatility",
        data=df_vers_plot,
        hue="Algorithm",
        hue_order=order,
        order=change_patterns
    )
    ax.figure.set_size_inches(20,4)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"{out_path}/Versatility{lag_window}/versatility_line_graph.pdf", bbox_inches="tight", format="pdf")

    # Plot bars again, but split in half
    first_half = change_patterns[:len(change_patterns)//2]
    second_half = change_patterns[len(change_patterns)//2:]
    fig,ax = plt.subplots(figsize=(20, 4))
    sns.barplot(x="Change Pattern", y="Versatility", data=df_vers_plot[df_vers_plot["Change Pattern"].isin(first_half)], hue="Algorithm", order=first_half, ax=ax, hue_order=order)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

    plt.savefig(f"{out_path}/Versatility{lag_window}/versatility_split_1.pdf", bbox_inches="tight", format="pdf")

    fig2,ax2 = plt.subplots(figsize=(20, 4))
    sns.barplot(x="Change Pattern", y="Versatility", data=df_vers_plot[df_vers_plot["Change Pattern"].isin(second_half)], hue="Algorithm", order=second_half, ax=ax2, hue_order=order)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"{out_path}/Versatility{lag_window}/versatility_split_2.pdf", bbox_inches="tight", format="pdf")

    # Versatility as Radar Chart
    plotly_default_colors = colors#px.colors.qualitative.Plotly
    fill_colors = {
        name: plotly_default_colors[idx]
        for idx,name in enumerate(df_vers_plot["Algorithm"].unique())
    }

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        for alg, dataframe in df_vers_plot.groupby(by="Algorithm"):
            fig = px.line_polar(
                df_vers_plot[df_vers_plot["Algorithm"] == alg], # Isnt this just dataframe from the groupby...
                r='Versatility',
                theta='Change Pattern',
                line_close=True,
                color='Algorithm',
                color_discrete_map=fill_colors,
                title=None# alg, # Set this to alg if we want the title, I was planning on setting the title in latex
            )
            fig.update_layout(showlegend=False, title_x=0.5, font=dict(size=18)) # Disable legend (as we have only one algorithm per plot) and center title (if present)
            fig.update_traces(fill='toself')

            if not os.path.exists(f"{out_path}/Versatility{lag_window}/Radar_Charts"):
                os.makedirs(f"{out_path}/Versatility{lag_window}/Radar_Charts")
            fig.write_image(f"{out_path}/Versatility{lag_window}/Radar_Charts/radar_chart_{alg.replace(' ','_')}.pdf", format="pdf")

def calculate_scalability(dataframe, show_progress_bar: bool = False):
    scalabilities = dict()

    groups = dataframe.groupby("Algorithm")
    if show_progress_bar:
        groups = tqdm(groups, desc="Calculating Scalability. Completed Algorithms: ")

    for name, a_df in groups:
        result = a_df["Duration (Seconds)"].mean()
        result_str = datetime.strftime(datetime.utcfromtimestamp(result), '%H:%M:%S')
        scalabilities[name] = {"avg_seconds": result, "str": result_str}
    return scalabilities

def plot_scalability(scalabilities, dfs, out_path, lag_window, colors, order):
    # Boxplot of Mean Absolute Duration
    scalability_plot_df = pd.DataFrame([
        {
            "Algorithm": name,
            "Duration": duration / 60
        }
        for name, a_df in dfs
        for duration in a_df["Duration (Seconds)"].tolist()
    ])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Duration [Minutes]")

    sorted_names = list(scalabilities.items())
    sorted_names.sort(key=lambda x: x[1]["avg_seconds"])

    palette = sns.color_palette(colors)
    ax = sns.boxplot(
        data=scalability_plot_df,
        order=order,
        x="Duration",
        y="Algorithm",
        palette = palette,
        width=1,
        ax=ax,
        fliersize=0
    )
    #ax.figure.set_size_inches(14, 4)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), size = 18)
    ax.set_ylabel(ax.get_ylabel(), size = 18)
    plt.xlabel("Duration [Minutes]")

    if not os.path.exists(f"{out_path}/Scalability{lag_window}"):
        os.makedirs(f"{out_path}/Scalability{lag_window}")
    plt.savefig(f"{out_path}/Scalability{lag_window}/scalability.pdf", bbox_inches="tight", format="pdf")

    # As Barplot
    scalability_barplot_df = pd.DataFrame([
        {
            "Algorithm": name,
            "Duration": duration / 60
        }
        for name, a_df in dfs
        for duration in a_df["Duration (Seconds)"].tolist()
    ])

    ax = plt.subplots(figsize=(17, 4))

    ax = sns.barplot(x="Algorithm", y="Duration", data=scalability_barplot_df,palette = palette, order=order)
    ax.figure.set_size_inches(12, 4)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), size = 18)
    ax.set_ylabel(ax.get_ylabel(), size = 18)
    plt.ylabel("Duration [Minutes]")
    plt.savefig(f"{out_path}/Scalability{lag_window}/scalability_barplot.pdf", bbox_inches="tight", format="pdf")

def calculate_rel_scalabilities(df, loglengths, loglengths_events):
    df_seconds = df.copy()
    # Add Log Length Column
    df_seconds["Log Length (Cases)"] = df_seconds[["Log Source", "Log"]].apply(axis=1, func=lambda x: loglengths[Path("EvaluationLogs", x[0], x[1]+".xes.gz")])
    df_seconds["Log Length (Events)"] = df_seconds[["Log Source", "Log"]].apply(axis=1, func=lambda x: loglengths_events[Path("EvaluationLogs", x[0], x[1]+".xes.gz")])

    seconds_per_case = dict()
    seconds_per_event = dict()

    for name, group in df_seconds.groupby("Algorithm"):
        total_seconds = group["Duration (Seconds)"].sum()

        total_cases = group["Log Length (Cases)"].sum()
        total_events = group["Log Length (Events)"].sum()

        seconds_per_case[name] = total_seconds / total_cases
        seconds_per_event[name] = total_seconds / total_events

    df_seconds["Milliseconds per Event"] = df_seconds.apply(lambda x: (x["Duration (Seconds)"]*1000) / x["Log Length (Events)"], axis=1)
    return df_seconds, seconds_per_case, seconds_per_event

def plot_rel_scalability(df_seconds, out_path, lag_window, colors, order):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Milliseconds per Event")

    palette = sns.color_palette(colors)
    ax = sns.boxplot(
        data=df_seconds,
        order=order,
        x="Milliseconds per Event",
        y="Algorithm",
        palette = palette,
        width=1,
        ax=ax,
        fliersize=0

    )

    ax.set_xscale("log")

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), size = 18)
    ax.set_ylabel(ax.get_ylabel(), size = 18)
    plt.xlabel("Milliseconds per Event")

    if not os.path.exists(f"{out_path}/Scalability{lag_window}"):
        os.makedirs(f"{out_path}/Scalability{lag_window}")
    plt.savefig(f"{out_path}/Scalability{lag_window}/scalability_mseconds_per_event.pdf", bbox_inches="tight", format="pdf")


    ax = plt.subplots(figsize=(17, 4))
    ax = sns.barplot(x="Algorithm", y="Milliseconds per Event", data=df_seconds, palette = palette, order=order)
    ax.figure.set_size_inches(12, 4)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), size = 18)
    ax.set_ylabel(ax.get_ylabel(), size = 18)
    ax.set_yscale("log")
    plt.ylabel("Milliseconds per Event")

    if not os.path.exists(f"{out_path}/Scalability{lag_window}"):
        os.makedirs(f"{out_path}/Scalability{lag_window}")
    plt.savefig(f"{out_path}/Scalability{lag_window}/scalability_mseconds_per_event_barplot.pdf", bbox_inches="tight", format="pdf")


def calculate_parameter_sensitivity(df, versatility_recall_dicts, computed_accuracy_dicts, scaled_latency_dicts, show_progress_bar: bool = False):
    versatility_score_per_param = {
        name: {
            params: np.nanmean(list(versatility_recall_dicts[name][params].values()))
            for params in versatility_recall_dicts[name].keys()
        }
        for name in versatility_recall_dicts.keys()
    }

    algnames = df["Algorithm"].unique()
    if show_progress_bar:
        algnames = tqdm(algnames, "Calculating Parameter Sensitivity. Completed Algorithms: ")

    sensitivities = dict()
    for name in algnames:
        _sensitivities = dict()
        acc = computed_accuracy_dicts[name]
        lat = scaled_latency_dicts[name]
        vers = versatility_score_per_param[name]
        for param_choice in acc.keys():
            sensitivity = harmonic_mean([acc[param_choice], vers[param_choice], lat[param_choice]])
            _sensitivities[param_choice] = sensitivity
        sensitivities[name] = _sensitivities

    return sensitivities

def plot_parameter_sensitivity(sensitivities, out_path, lag_window, colors, order):
    sens_df = pd.DataFrame([
        {
            "Parameters": param,
            "Algorithm": name,
            "Sensitivity Score": sens
        }
        for name, sens_dict in sensitivities.items()
        for param, sens in sens_dict.items()
    ])

    palette = sns.color_palette(colors)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.boxplot(
        data=sens_df,
        x="Sensitivity Score",
        y="Algorithm",
        palette = palette,
        ax=ax,
        fliersize=0,
        order=order
    )
    ax.figure.set_size_inches(7, 4)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel("Performance", size = 18)
    ax.set_ylabel(ax.get_ylabel(), size = 18)

    if not os.path.exists(f"{out_path}/Parameter_Sensitivity{lag_window}"):
        os.makedirs(f"{out_path}/Parameter_Sensitivity{lag_window}")
    plt.savefig(f"{out_path}/Parameter_Sensitivity{lag_window}/param_sensitivity.pdf", bbox_inches="tight", format="pdf")

def calculate_parameter_sensitivity_iqr(sensitivities):
    sensitivity_iqrs = dict()
    for name, sens in sensitivities.items():
        _iqr = iqr(list(sens.values()), nan_policy="omit")
        sensitivity_iqrs[name] = _iqr
    return sensitivity_iqrs

def calc_harm_means(dataframe, min_support, lag_window, show_progress_bar: bool = False):
    robustnesses = dict()

    groups = dataframe.groupby("Noise Level")
    if show_progress_bar:
        groups = tqdm(groups, "Calculating Harmonic Means for Robustness. Noise Levels Completed:")

    for noise_level, noise_df in groups:
        # Calculate accuracy over these logs
        accuracies, _, _, _, _ = calculate_accuracy_metric_df(noise_df, lag_window, show_progress_bar=False)
        latencies, _, _, _ = calculate_latency(noise_df, lag_window, min_support=min_support, show_progress_bar=False)
        versatilities, _, _ = calc_versatility(noise_df, lag_window, show_progress_bar=False)

        assert accuracies.keys() == latencies.keys() == versatilities.keys()

        robustnesses[noise_level] = {
            approach: harmonic_mean([accuracies[approach], latencies[approach], versatilities[approach]])
            for approach in accuracies.keys()
        }
    return robustnesses

def convert_harm_mean_to_auc(means, div_zero_default=0):
    ## Conventions:
        # If any harmonic mean is nan, then set it to 0 instead


    def _convert_nan_to_zero(x):
        if np.isnan(x):
            x = 0
        return x

    # means maps {noise_level: {approach: robustness}}
    # reformat to {approach: [(noise_level, robustness)]}
    means = {
        approach: [
            (int(noise_level), _convert_nan_to_zero(means[noise_level][approach]))
            for noise_level in means.keys()
        ]
        for approach in means["0"].keys()
    }

    robustnesses = dict()
    for approach in means.keys():
        points = sorted(means[approach], key=lambda x: x[0])

        # Initial Robustness - Use to model "Ideal Robustness"
        initial_robustness = points[0][1] # Robustness for lowest (0%) noise
        ideal_auc = initial_robustness * (points[-1][0] - points[0][0]) # AUC if it was always exactly as good as for the lowest noise level. If performance increases with higher noise, the achieved AUC might be larger than ideal AUC
        prev_noise, prev_robust = points[0]
        auc = 0
        for noise, robust in points[1:]:
            delta_noise = noise-prev_noise
            sum_robust = prev_robust+robust

            # "ROC Index"
            area = (delta_noise * sum_robust) / 2# Area under curve for this segment
            auc += area

            prev_noise = noise
            prev_robust = robust

        if ideal_auc != 0:
            robustnesses[approach] = auc / ideal_auc
        else:
            robustnesses[approach] = div_zero_default
    return robustnesses

def _plot_robustness_one(robustness_df, logset, color_map):
    fig, axs = plt.subplots(2,4, figsize=(12,6))
    #fig.suptitle(f"Robustness of Approaches for {logset} Logs")
    
    for idx, name in enumerate(robustness_df["Approach"].unique()):
        row_offset = idx // 4
        col_offset = idx % 4
        ax = axs[row_offset, col_offset]
        relevant_df = robustness_df[(robustness_df["Approach"] == name) & (robustness_df["Log Set"] == logset)]
        sns.pointplot(x="Noise Level", y="Robustness", data=relevant_df, errorbar=None,color= color_map[name], ax=ax)#, title=name)
        ax.set_ylim(0,1)
        ax.set_title(name, size = 16)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.tick_params(labelsize=16)
        ax.set_xlabel(ax.get_xlabel() if row_offset == 1 else "", size = 18)
        ax.set_ylabel("Performance" if col_offset == 0 else "", size = 18)
        robustness_of_zero_noise = min(zip(relevant_df["Noise Level"], relevant_df["Robustness"]), key=lambda x: x[0])[1]
        if np.isnan(robustness_of_zero_noise):
            robustness_of_zero_noise = 0
        # plt.axhline(y=robustness_of_zero_noise, color='r', linestyle='-', ax=ax)
        ax.hlines(robustness_of_zero_noise, color=color_map[name], linestyle='dashed', xmin=0, xmax=len(relevant_df["Noise Level"].unique())-1)
        fig.tight_layout()
        #ax.grid()
        fig.set_facecolor('0.9')
    return fig, axs

def plot_robustness(means_ostovar, means_ceravolo, out_path, lag_window, color_map):
    plot_df_robust = pd.DataFrame(
        ([
            {
                "Log Set": "Ostovar",
                "Approach": name,
                "Noise Level": int(noise_level),
                "Robustness": robustness
            }
            for noise_level, robust_dict  in means_ostovar.items()
            for name, robustness in robust_dict.items()
        ] + [
            {
                "Log Set": "Ceravolo",
                "Approach": name,
                "Noise Level": int(noise_level),
                "Robustness": robustness
            }
            for noise_level, robust_dict  in means_ceravolo.items()
            for name, robustness in robust_dict.items()
        ])
    )


    if not os.path.exists(f"{out_path}/Robustness{lag_window}"):
        os.makedirs(f"{out_path}/Robustness{lag_window}")

    fig_ostovar, axs_ostovar = _plot_robustness_one(plot_df_robust, "Ostovar", color_map)
    fig_ostovar.savefig(f"{out_path}/Robustness{lag_window}/robustness_ostovar_logs.pdf", bbox_inches="tight", format="pdf")

    fig_ceravolo, axs_ceravolo = _plot_robustness_one(plot_df_robust, "Ceravolo", color_map)
    fig_ceravolo.savefig(f"{out_path}/Robustness{lag_window}/robustness_ceravolo_logs.pdf", bbox_inches="tight", format="pdf")

    plt.close('all')

    fig_ostovar_in_one = _plot_robustness_in_one(plot_df_robust, "Ostovar", color_map)
    fig_ostovar_in_one.savefig(f"{out_path}/Robustness{lag_window}/robustness_ostovar_logs_all_in_one.pdf", bbox_inches="tight", format="pdf")

    fig_ceravolo_in_one = _plot_robustness_in_one(plot_df_robust, "Ceravolo", color_map)
    fig_ceravolo_in_one.savefig(f"{out_path}/Robustness{lag_window}/robustness_ceravolo_logs_all_in_one.pdf", bbox_inches="tight", format="pdf")


def _plot_robustness_in_one(robustness_df, logset, color_map):
    for _, name in enumerate(robustness_df["Approach"].unique()):
        relevant_df = robustness_df[(robustness_df["Approach"] == name) & (robustness_df["Log Set"] == logset)]
        ax = sns.pointplot(x="Noise Level", y="Robustness", data=relevant_df, errorbar=None,color= color_map[name])#, title=name)
        ax.set_ylim(0,1)
        ax.figure.set_size_inches(5, 12)
        #ax.set_title(logset+" Logs with Different Noise Levels", size = 18)
        ax.set_xlabel("Noise (%)")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.tick_params(labelsize=16)
        ax.set_xlabel(ax.get_xlabel(), size = 18)
        ax.set_ylabel(ax.get_ylabel(), size = 18)
        robustness_of_zero_noise = min(zip(relevant_df["Noise Level"], relevant_df["Robustness"]), key=lambda x: x[0])[1]
        if np.isnan(robustness_of_zero_noise):
            robustness_of_zero_noise = 0
        # plt.axhline(y=robustness_of_zero_noise, color='r', linestyle='-', ax=ax)
        ax.hlines(robustness_of_zero_noise, color=color_map[name], linestyle='dashed', xmin=0, xmax=len(relevant_df["Noise Level"].unique())-1)
    return ax.figure













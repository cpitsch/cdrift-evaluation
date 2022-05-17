import pandas as pd
import os
import math
from pathlib import Path
import numpy as np
def makeTex(csvpath, outputpath):
    makeTexFromFrame(pd.read_csv(csvpath, outputpath))

def makeTexFromFrame(df:pd.DataFrame, outputpath):

    nameMappings = {
        "ConditionalMove": "Cond. Move",
        "ConditionalRemoval": "Cond. Removal",
        "ConditionalToSequence": "Cond. to Seq.",
        "Frequency": "Frequency",
        "Loop": "Loop",
        "ParallelMove": "Parallel Move",
        "ParallelRemoval": "Parallel Removal",
        "ParallelToSequence": "Parallel to Seq.",
        "SerialMove": "Serial Move",
        "SerialRemoval": "Serial Removal",
        "Skip": "Skip",
        "Substitute": "Substitute",
        "Swap":"Swap"
    }

    Path(outputpath).touch()
    with open(outputpath, "w") as f:
        #Open environments
        f.write("\\begin{table}[h]\n\t\\centering\n\t\\caption{}\n\t\\label{}\n\t\\begin{tabular}{l|c|c|c|c|c|c}\n")
        #Write in table
        f.write("\t\t\\hline\n\t\t\\textbf{Window Size}&   $100$   &   $200$   &   $300$   &   $400$   &   $500$   &   $600$   \\\\\\hline\n\t\t\\textbf{Event Log}  &      &       &      &      &      &      \\\\\n")
        df.sort_values(['Log', 'Window Size'], inplace=True)
        groups = df.groupby("Log")
        for logname, group in groups:
            lname = logname.split("_")[-1]
            if lname == "5" or lname == "2":
                # Noisy logs, use second-to-last instead
                lname = logname.split("_")[-2]
            lname = nameMappings.get(lname, lname)
            # Probably Redundant but i dont care 
            group.sort_values(["Window Size"])
            f1_100 =  _my_round(group.iloc[0]['F1-Score'],2)   
            f1_200 = _my_round(group.iloc[1]['F1-Score'],2)   
            f1_300 = _my_round(group.iloc[2]['F1-Score'],2)   
            f1_400 = _my_round(group.iloc[3]['F1-Score'],2)
            f1_500 = _my_round(group.iloc[4]['F1-Score'],2)
            f1_600 = _my_round(group.iloc[5]['F1-Score'],2)

            f.write(f"\t\t{lname}           & {f'${f1_100}$' if not np.isnan(f1_100) else ''}     &  {f'${f1_200}$' if not np.isnan(f1_200) else ''}     & {f'${f1_300}$' if not np.isnan(f1_300) else ''}  &   {f'${f1_400}$' if not np.isnan(f1_400) else ''}    &   {f'${f1_500}$' if not np.isnan(f1_500) else ''} &   {f'${f1_600}$' if not np.isnan(f1_600) else ''}  \\\\\n")

        #Close environments
        f.write("\t\\end{tabular}\n\\end{table}")


def _my_round(num, places):
    if num.is_integer():
        return int(num)
    else:
        return round(num,places)
# \begin{table}[h]
#     \centering
#     \caption{Performance of J-Measure/Window Count Extraction on Synthetic Logs Using Recursive Bisection (F1-Score)}
#     \label{f1:martjushev}
#     \begin{tabular}{l|c|c|c|c}
#         \hline
#         \textbf{Window Size}& 50   & 150   & 250  & 500  \\\hline
#         \textbf{Event Log}  &      &       &      &      \\
#         Noiseless           &      &       &      &      \\
#         Cond. to Seq.       &      &       &      &      \\
#         Frequency           &      &       &      &      \\
#         Loop                &      &       &      &      \\
#         Parallel-Move       &      &       &      &      \\
#         Parallel to Seq.    &      &       &      &      \\
#         Skip                &      &       &      &      \\
#         Substitute          &      &       &      &      \\
#         Swap                &      &       &      &      \\
#     \end{tabular}
# \end{table}
def makeDurationTex(csvpath, outputpath):
    makeDurationTexFromFrame(pd.read_csv(csvpath, outputpath))

def makeDurationTexFromFrame(df:pd.DataFrame, outputpath):

    nameMappings = {
        "ConditionalMove": "Cond. Move",
        "ConditionalRemoval": "Cond. Removal",
        "ConditionalToSequence": "Cond. to Seq.",
        "Frequency": "Frequency",
        "Loop": "Loop",
        "ParallelMove": "Parallel Move",
        "ParallelRemoval": "Parallel Removal",
        "ParallelToSequence": "Parallel to Seq.",
        "SerialMove": "Serial Move",
        "SerialRemoval": "Serial Removal",
        "Skip": "Skip",
        "Substitute": "Substitute",
        "Swap":"Swap"
    }

    Path(outputpath).touch()
    with open(outputpath, "w") as f:
        #Open environments
        f.write("\\begin{table}[h]\n\t\\centering\n\t\\caption{}\n\t\\label{}\n\t\\begin{tabular}{|l|c|c|c|c|c|c|}\n")
        #Write in table
        f.write("\t\t\\hline\n\t\t\\textbf{Window Size}&   $100$   &   $200$   &   $300$   &   $400$   &   $500$   &   $600$   \\\\\n\t\t\\textbf{Event Log}  &      &       &      &      &      &      \\\\\\hline\n")
        df.sort_values(['Log', 'Window Size'], inplace=True)
        groups = df.groupby("Log")
        for logname, group in groups:
            lname = logname.split("_")[-1]
            if lname == "5" or lname == "2":
                # Noisy logs, use second-to-last instead
                lname = logname.split("_")[-2]
            lname = nameMappings.get(lname, lname)
            # Probably Redundant but i dont care 
            group.sort_values(["Window Size"])
            d_100 = group.iloc[0]['Duration']   
            d_200 = group.iloc[1]['Duration']   
            d_300 = group.iloc[2]['Duration']   
            d_400 = group.iloc[3]['Duration']
            d_500 = group.iloc[4]['Duration']
            d_600 = group.iloc[5]['Duration']

            f.write(f"\t\t{lname}           & {d_100}     &  {d_200}     & {d_300}  &   {d_400}    &   {d_500} &   {d_600}  \\\\\\hline\n")

        #Close environments
        f.write("\t\\end{tabular}\n\\end{table}")

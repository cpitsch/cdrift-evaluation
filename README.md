# Concept Drift Evaluation #
- The results for each algorithm and parameter setting can be found in [`evaluation_results.csv`](./evaluation_results.csv).
- The resulting evaluation measures as described in the paper are listed in [`evaluation_metrics.csv`](./evaluation_measures.csv).
- The corresponding figures for the individial evaluation measures are in the [`Eval_Figures`](./Eval_Figures/) folder.

# Usage #
The required packages can be installed using anaconda: 
```bash
> conda env create -f ./environment.yaml
...
> conda activate cdrift-evaluation
```
## Perfoming the Evaluation &mdash; Running all Algorithms ##
- To run the algorithms on all event logs and parameter settings, execute the [`testAll_reproducibility.py`](./testAll_reproducibility.py) file.
- This will create a CSV file containing the detected change points for every algorithm, event log, and parameter setting. 
  - During the execution, also CSV files will be created containing the results for individual executions of the algorithms.
## Using the Approaches ##

- One can explore the algorithms and their parameters in the provided jupyter notebooks `<algorithm-name>_example.ipynb` in the Examples directory:
  - [Bose](Examples/bose_example.ipynb)
  - [Martjushev](Examples/martjushev_example.ipynb)
  - [ProDrift](Examples/prodrift_example.ipynb)
  - [Earth Mover's Distance](Examples/earthmover_example.ipynb)
  - [Process Graph Metrics](Examples/process_graph_example.ipynb)
  - [Zheng](Examples/zheng_example.ipynb)

### Event Logs ###

Event Logs can be found in the [`EvaluationLogs`](./EvaluationLogs/) folder. These are the event logs used in the evaluation, which are sourced from:

- Ostovar et al. Robust Drift Characterization from Event Streams of Business Processes[^ostovar]
- Ceravolo et al. Evaluation Goals for Online Process Mining: a Concept Drift Perspective [^ceravolo]
- Bose et al. Handling Concept Drift in Process Mining  [^bose]

[^ostovar]: Ostovar et al. Robust Drift Characterization from Event Streams of Business Processes. URL https://www.doi.org/10.1145/3375398

[^ceravolo]: Ceravolo et al. Evaluation Goals for Online Process Mining: a Concept Drift Perspective URL https://www.doi.org/10.1109/TSC.2020.3004532

[^bose]: Bose et al. Handling Concept Drift in Process Mining. URL https://www.doi.org/10.1007/978-3-642-21640-4_30

### Bose et al. ###
- Log-splitting (for Relation Type Count and Relation Entropy) with [`logsplitter.py`](./cdrift/utils/logsplitter.py)
- Extraction functions for J-Measure, Window Count, Relation Type Count, Relation Entropy are in [`bose.py`](./cdrift/approaches/bose/bose.py)
- Apply the full detection pipeline with (example for J-Measure + KS Test):
```python
from cdrift.approaches import bose
from pm4py import read_xes
log = read_xes(PATH_TO_LOG) # Import the Log
pvalues = bose.detectChange_JMeasure_KS(log, windowSize, measure_window, activityName_key) # Extract J-Measure, Apply Sliding Window tests with KS-Test
changepoints = bose.visualInspection(log, trim=windowSize) # Visual Inspection
```
### Martjushev et al. ###
- Extraction functions in [`bose.py`](./cdrift/approaches/bose/bose.py)
- Apply the full pipeline with (example for J-Measure + KS Test):
```python
from cdrift.approaches import martjushev
from pm4py import read_xes
log = read_xes(PATH_TO_LOG) # Import the Log
changepoints, pvalues = martjushev.detectChange_JMeasure_KS(log, windowSize, pvalue, return_pvalues=True, j_measure_window) # Extract J-Measure, apply sliding window with recursive bisection
```
### ProDrift ###
- Apply the full pipeline with:
```python
from cdrift.approaches import maaradji
from pm4py import read_xes
log = read_xes(PATH_TO_LOG) # Import the Log
changepoints, pvalues = maaradji.detectChangepoints(log, windowSize, pvalue, return_pvalues=True)
```
### Earth Mover's Distance ###
- Apply the full pipeline with:
```python
from cdrift.approaches import earthmover
from pm4py import read_xes
log = read_xes(PATH_TO_LOG) # Import the Log
traces = earthmover.extractTraces(log) # Extract Time Series of Traces
em_dists = earthmover.calculateDistSeries(traces, windowSize) # Calculate Earth Mover's Distances
changepoints = earthmover.visualInspection(em_dists,trim=windowSize) # Visual Inspection
```

### Process Graph Metrics ###
- Apply the full pipeline with:
```python
from cdrift.approaches import process_graph_metrics as pgm
from pm4py import read_xes
log = read_xes(PATH_TO_LOG) # Import the Log
changepoints = pgm.detectChange(log, windowSize,maxWinSize, pvalue)
```

### Zheng et al. (RINV) ###
- Apply the full pipeline with:
```python
from cdrift.approaches import zheng
from pm4py import read_xes
log = read_xes(PATH_TO_LOG) # Import the Log
changepoints = zheng.apply(log,mrid, epsilon)
```
## Evaluation ##
- Evaluation for specific instances is performed using [`evaluation.py`](./cdrift/evaluation.py):
```python
from cdrift import evaluation
f1 = evaluation.F1_Score(detected_cps, known_cps, lag_window)# Calculate F1-Score
# Or:
tp, fp = evaluation.getTP_FP(detected_cps, known_cps, lag_window) # Calculate True/False Positives
# Or:
from numpy import NaN
precision, recall = evaluation.calcPrecision_Recall(detected_cps, known_cps, lag_window, zero_division=NaN) # Calculate Precision/Recall
```
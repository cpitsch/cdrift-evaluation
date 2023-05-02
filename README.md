# Concept Drift Evaluation #
- The results for each algorithm and parameter setting can be found in [`algorithm_results.csv`](./algorithm_results.csv).
- The resulting evaluation measures (for a Lag Window of `200`) as described in the paper are listed in [`evaluation_measures200.csv`](./Evaluation_Results/evaluation_measures200.csv).
- The corresponding figures for the individual evaluation measures are in the [`Evaluation_Results`](./Evaluation_Results/) folder.


# Incorporating New Approaches #
To incorporate results from your approach into the evaluation, you can either:

1. Run your approach separately, and create a csv containing at minimum the columns:
   - `Algorithm` (The algorithm name)
   - `Log Source` (Where the log was sourced. In our case, "Ostovar", "Ceravolo", or "Bose")
   - `Log` (The log name, e.g., "bose_log.xes.gz")
   - `Detected Changepoints` (The indices (of cases) in the event log where changes where detected)
   - `Actual Changepoints for Log` (The ground truth changepoint indices)
   - `Duration (Seconds)` (The duration of the algorithm run in seconds)
   - `Duration` (Duration as hh:mm:ss string)
   - Columns specifying the parameter settings
2. Append the results to the `algorithm_results.csv` file.

Or:

1. Add your approach to the `testAll_reproducibility.py` file.
   1. Define a function to run your approach on a single log that returns a list of dictionaries containing the columns mentioned above.
   2. Configure the parameters of your approach in the `testAll_config.yml` file by adding an entry to "approaches" containing:
      - `function`: The name of the function in `testAll_reproducibility.py`
      -  `params`: In here, list all the parameters that your function takes, specified as lists of possible values.


# Usage #
The required packages can be installed using anaconda: 
```bash
> conda env create -f ./environment.yaml
...
> conda activate cdrift-evaluation
```

Or with the [`requirements.txt`](./requirements.txt) file:
```bash
> pip install -r requirements.txt
```

## Docker ##

- Additionally, a [docker image](https://hub.docker.com/r/cpitsch/cdrift-evaluation) is also provided. To use this image:
1. Pull the image: `docker pull cpitsch/cdrift-evaluation`	
2. Run the container with `docker run --name cdrift cpitsch/cdrift-evaluation`. Possible arguments are:
   - `--evaluate`: Run only the evaluation script. Assumes that the algorithm results are already present in the container.
   - `--runApproaches`: Run all the approaches (this will take a long time).
   - `--runAndEvaluate`: Run all the approaches and then evaluate the results.
3. After algorithm runs and/or the evaluation, the results can be retrieved from the docker container through:
   - `docker cp cdrift:cdrift_docker/algorithm_results.csv .`
   - `docker cp cdrift:cdrift_docker/Evaluation_Results .`


## Running all Algorithms ##
- To run the algorithms on all event logs and parameter settings, execute the [`testAll_reproducibility.py`](./testAll_reproducibility.py) file.
- This will create a CSV file, `algorithm_results.csv`, containing the detected change points for every algorithm, event log, and parameter setting. 
  - During the execution, also CSV files will be created containing the results for individual executions of the algorithms.

## Performing the Evaluation ##
After running all the algorithms, the evaluation can be performed using the notebook: [`evaluate_results.ipynb`](./evaluate_results.ipynb). This will create a folder [`Evaluation_Results`](./Evaluation_Results/) containing:
- A csv file [`evaluation_measures<lag-window>.csv`](./Evaluation_Results/evaluation_measures200.csv) containing all evaluation metrics as defined in the paper
- All the generated figures

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

- Ostovar et al. Robust Drift Characterization from Event Streams of Business Processes[^ostovar] ([Source](https://apromore.com/research-lab/#heading16))
- Ceravolo et al. Evaluation Goals for Online Process Mining: a Concept Drift Perspective [^ceravolo] ([Source](https://dx.doi.org/10.21227/2kxd-m509))
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
changepoints = pgm.detectChange(log, windowSize, maxWinSize, pvalue)
```

### Zheng et al. (RINV) ###
- Apply the full pipeline with:
```python
from cdrift.approaches import zheng
from pm4py import read_xes
log = read_xes(PATH_TO_LOG) # Import the Log
changepoints = zheng.apply(log, mrid, epsilon)
```

### LCDD ###
- Apply the full pipeline with:
```python
from cdrift.approaches import lcdd
from pm4py import read_xes
log = read_xes(PATH_TO_LOG) # Import the Log
changepoints = lcdd.calculate(log, complete_window_size, detection_window_size, stable_period)
```
## Evaluation ##
### Evaluating Isolated Results ###
- For a single execution of an algorithm on a single event log, a simple evaluation can be performed using functions from  [`evaluation.py`](./cdrift/evaluation.py):

```python
from cdrift import evaluation
f1 = evaluation.F1_Score(detected_cps, known_cps, lag_window)# Calculate F1-Score
# Or:
tp, fp = evaluation.getTP_FP(detected_cps, known_cps, lag_window) # Calculate True/False Positives
# Or:
from numpy import NaN
precision, recall = evaluation.calcPrecision_Recall(detected_cps, known_cps, lag_window, zero_division=NaN) # Calculate Precision/Recall
# Or:
avg_lag = evaluation.get_avg_lag(detected_cps, known_cps, lag=lag_window)
```

### Full Evaluation ###

- The full evaluation for multiple algorithms, multiple parameter settings, and multiple event logs is performed by running the [evaluation notebook](./evaluate_results.ipynb).
- This takes the [algorithm_results.csv](./algorithm_results.csv) file as input and generates the folder [Evaluation_Results](./Evaluation_Results) containing the evaluation results.
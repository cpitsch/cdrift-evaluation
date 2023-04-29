FROM python:3.10.5

WORKDIR /cdrift_docker

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./cdrift/ ./cdrift
COPY ./testAll_reproducibility.py ./
COPY ./testAll_config.yml ./
COPY ./evaluate.py ./
COPY ./docker_entry.py ./ 
COPY ./evaluate_results.ipynb ./
COPY ./algorithm_results.csv ./
COPY ./EvaluationLogs ./EvaluationLogs

ENTRYPOINT [ "python", "docker_entry.py" ]
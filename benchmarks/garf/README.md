# GARF baseline

To measure a baseline with GARF, you have two options:

1) Run GARF in your terminal
2) Deploy GARF to Kubernetes

In any case, you will need to use `python 3.7` to run GARF.
In this directory, there is a `.python-version` file, which is used by `pyenv` and offers you one way to manage your python versions.

**Requirements**
GARF depends on `tensorflow < 2.0`, which is not supported anymore and generally not available for ARM chips.
You  will not be able to run GARF on an ARM system, or build the docker image on an ARM system due to that.


## 1) Run GARF In Your Terminal

To run GARF in your terminal, first create a virtual environment and activate it.
Then, install dependencies via `python -m pip install -r src/requirements.txt`.

Next, run `python export_to_garf.py`.
This will read datasets from `/mirmir/datasets/` into a sqlite3 database located at `src/database.db`.

In `src/main.py`, you can manually set the dataset name, e.g. `dataset = 'hospital'`.
To find the names of all available datasets, connect to the sqlite database and check the table names:

```
sqlite3 database.db
.tables
```

## 2) Run GARF on Kubernetes

To run GARF on Kubernetes, install Docker on your machine.
Then, run `docker build -t <your_username>/garf-experiment:latest .` in `benchmarks/garf/` to build the Docker image.
Then push it to the docker registry: `docker push <your_username>/garf-experiment:latest`

Next, make sure that a persistent volume claim has been set up on kubernetes - you will find a sample configuration in `infrastructure/share_pvc.yaml` that you can use to create a PVC on your cluster. 10GB of disk should be plenty.

Next, run `python generate_jobs.py` in `benchmarks/garf/infrastructure`.
This will create a directory called `benchmarks/garf/infrastructure/jobs`, which contains .yaml files for each job.
You can submit all those jobs with `kubectl apply -f jobs/`.

You can then check the status of your jobs with `kubectl get jobs`.
Once the jobs are finished, all results were written to the PVC.
To get the data out of the PVC, create a pod which a) mounts the PVC, and b) loops infinitely.
A configuration of such a pod can be found in `infrastructure/pvc_access_pod.yaml`.
You can submit it with `kubectly apply pvc_access_pod.yaml`.

Once the pod runs, connect with `kubectl exec -it pvc-access-pod -- /bin/sh`, and copy results with `kubectl cp pvc-access-pod:/data/ results/`.

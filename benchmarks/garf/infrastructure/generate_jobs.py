from pathlib import Path

def generate_jobs(jobs_path: Path) -> int:
    """
    Generates jobs to deploy pods with on the BHT cluster.

    The current setup aims to do three runs per dataset. For OpenML datasets
    and the Baran datasets, this is achieved by creating 3 jobs for the
    same dataset. For RENUVER, each dataset exists in 5 versions, of which the
    first 3 are used.
    """
    datasets = []

    for d in ['rayyan', 'beers', 'hospital', 'flights']:
        for r in range(3):
            datasets.append({
                'job_name': f'{d}-{r}',
                'dataset': d
                })

    for d in [6, 137, 151, 184, 1481, 41027, 43572]:
        for d_type in ['simple_mcar', 'imputer_simple_mcar']:
            for pct in [1, 5, 10]:
                for r in range(3):
                    datasets.append({
                        'job_name': f'{d}-{d_type.replace("_", "-")}-{pct}-{r}',
                        'dataset': f'{d}_{d_type}_{pct}',
                        })

    for d in ['cars', 'bridges', 'restaurant', 'glass']:
        for pct in range(1, 6):
            for v in range(1, 4):
                datasets.append({
                    'job_name': f'{d}-{pct}-{v}',
                    'dataset': f'{d}_{pct}_{v}',
                    })

    template = """apiVersion: batch/v1
kind: Job 
metadata:
  name: garf-baseline-{}
spec:
  completions: 1
  template:
    metadata:
      labels:
        app: garf-baseline-experiment-{}
    spec:
      nodeSelector:
        cpuclass: epyc
      restartPolicy: Never
      containers:
        - name: garf-experiment
          image: docker.io/larmor27/garf-experiment:latest
          env:
            - name: DATASET
              value: {}
          volumeMounts:
            - name: data-volume
              mountPath: /src/output  # Mounting the PVC at /app/output directory in the container
      volumes:
        - name: data-volume
          persistentVolumeClaim:
            claimName: garf-measurements
    """

    for i, d in enumerate(datasets):
        job_config = template.format(d['job_name'], d['job_name'], d['dataset'])
        with open(jobs_path / f'config_{i}.yaml', 'wt') as f:
            f.write(job_config)

    return i

if __name__ == '__main__':
    jobs_path = Path('jobs/')
    jobs_path.mkdir(parents=True, exist_ok=True)

    n_jobs_creates = generate_jobs(jobs_path)
    print(f'Generated {n_jobs_creates} jobs and stored them to {jobs_path}/.')

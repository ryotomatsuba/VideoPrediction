# VideoPrediction

## Set up environment

### Attach to docker container

```sh
sh docker/build.sh
sh docker/run.sh
sh docker/exec.sh
```

## Run training

1. set `configs/default.yaml`
2. run `train.py`

```sh
python3 train.py
```

or you can also use command line args to set parameters.
for example,

```sh
python3 train.py model.train_model=expert1 dataset.num_data=1000
```

When you want to use a GPU other than number 0 run the following command beforehand.

```sh
export CUDA_VISIBLE_DEVICES="1" # 2, 3, or "" for CPU
```

## Visualize results

```sh
cd /data/Result/
mlflow ui
```

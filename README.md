# VideoPrediction

## Set up environment

```sh
sh docker/build.sh
sh docker/run.sh
sh docker/exec.sh
```

## Run training

1. set `configs/default.yaml`
2. run `train.py`

```sh
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH=$PYTHONPATH:$PWD
python train.py
```

## Visualize results

```sh
cd /home/data/ryoto/Result
mlflow ui
```

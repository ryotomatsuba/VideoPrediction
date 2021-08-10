# 実験テンプレート

## run training

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

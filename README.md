# VideoPrediction

## Set up environment

### Attach to docker container

```sh
sh docker/build.sh
sh docker/run.sh
sh docker/exec.sh
```

### Install pyorch

After attaching to docker container, you should install appropriate version of Pytorch for your cuda environment.
Check <https://pytorch.org/get-started/locally/>

## Set Proxy

If you are using wni server (pt-sh.wni.co.jp), you need to set up a proxy.

```sh
export http_proxy=http://172.16.250.1:8080
export https_proxy=http://172.16.250.1:8080
export no_proxy=localhost,127.0.0.1

```

## Run training

1. set `configs/default.yaml`
2. run `train.py`

```sh
python train.py
```

If you are not using docker, you may need to run the following command beforehand.

```sh
export CUDA_VISIBLE_DEVICES="0" # or 1,2
export PYTHONPATH=$PYTHONPATH:$PWD
```

## Visualize results

```sh
cd /data/Result/
mlflow ui
```

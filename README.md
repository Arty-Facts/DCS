# DCS Data-Centric Sampling
```
final comparison:
3 ResNet18BN acc = 78.29 +- 0.71 all = [0.7923, 0.775, 0.7813] |GAN evalute
3 ResNet18BN acc = 84.08 +- 0.46 all = [0.8432, 0.8449, 0.8344] |GAN_Inversion evalute
3 ResNet18BN acc = 86.03 +- 0.07 all = [0.8603, 0.8612, 0.8595] |ITGAN evalute
3 ResNet18BN acc = 86.27 +- 0.09 all = [0.8636, 0.863, 0.8614] |real evalute
```

## Setup host system (Not needed if using docker)
```
chmod +x environment/base-packages.sh
sudo ./environment/base-packages.sh
```

## Enter the docker image environment 

In linux

```
./docker.sh [clean]
```

## Enter the virtual environment 

In windows

```
env.bat 
```


In linux

```
source ./env.sh [clean]
```

In docker

```
source env.sh [clean]
```

## Run Jupyter server

```
jupyter.sh
```

in docker

```
./deploy_docker.sh jupyter.sh 
```
you can now access the jupyter server on: 
http://localhost:8888/?token=docker

works with colab

## Deploy a script inside a docker image

```
./deploy_docker.sh [script]
```
docker deploy will run the script inside the docker image and then exit the image, this till be done inside python virtual environment


## Update docker environment

In the file (environment/base-packeges.sh) add apt packages that you need in your project

note that a newline will brake the RUN command and thus "\\" should be used when adding dependencies. More information on how docker works can be found on https://docs.docker.com/get-started/


## Update pip environment

Python dependencies for the project should be added to the environment/requirements.txt file

## Run tests using 

```
tox
```

# Taken from 
## Synthesizing Informative Training Samples with GAN [[PDF]](https://arxiv.org/pdf/2204.07513.pdf)

The experiment data have been released in [Google Drive](https://drive.google.com/drive/folders/1qyxK4XxboBRuQVwesxQTSx-Vpcp1fCeS?usp=sharing).
The released data include: [1] pretrained BigGAN Generators; [2] GAN Inversion learned latent vectors (z); [3] IT-GAN learned latent vectors (z). 

We also provide some raw training codes which have been neither re-organized nor validated. Just for reference. The raw training code is [here](https://drive.google.com/drive/folders/1vENTbqDdt6f0K2fQpuUfuCEnj_09Bqeh?usp=sharing). Replace md_utils.py with utils.py, md_networks.py with networks.py if applicable. 

mysql+pymysql://root:root@172.16.1.129/dcs


dockipy project/run.py  project/config_baseline.yaml
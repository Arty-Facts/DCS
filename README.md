[![GitHub Actions Status](https://github.com/Arty-Facts/python_docker_template/actions/workflows/python-package-test.yml/badge.svg)](https://github.com/Arty-Facts/python_docker_template/actions/workflows/python-package-test.yml/badge.svg)

# Docker project template

## How to use the template

### For empty project
```
git clone https://github.com/Arty-Facts/python_docker_template.git

git remote remove origin

git remote add origin <new remote>

git push --set-upstream origin main
```

### For active project 
```
git clone https://github.com/Arty-Facts/python_docker_template.git
cd python_docker_template
mv -r $(ls  --ignore=.git ) .. ; cd ..
rm rf python_docker_template
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

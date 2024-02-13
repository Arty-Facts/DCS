#! /bin/bash
source environment/utils.sh

clean=""
all_gpus=""
file=""
if [[ $1 == "clean" ]]; then 
    __banner docker images will be rebuilt
    clean="--no-cache"
fi

# find nvidia gpu
gpu=$(lspci | tr '[:upper:]' '[:lower:]' | grep -i nvidia)

# if gpu available add --gpus all
if [[ $gpu == *' nvidia '* ]]; then
    __banner 'Nvidia GPU is present:  %s\n' "$gpu"
    all_gpus="--gpus all"
    file=dockerfile.gpu
else
    __banner 'Nvidia GPU is not present cpu only!\n'  
    file=dockerfile.cpu
fi

docker build -t party_image \
        $clean \
        --network=host\
        --build-arg WORK_DIR=$PWD \
        --build-arg HOME_DIR=$HOME \
        -f environment/$file environment \
        || exit
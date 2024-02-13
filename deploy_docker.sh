#! /bin/bash
source environment/utils.sh

if [[ ! -f "$1" ]]; then
    __banner "File not found: $1"
    exit 1
fi

source environment/build_docker.sh $1

docker run -ti --rm \
        -v ~:/$HOME \
        --shm-size=16G\
        $all_gpus \
        -v /mnt:/mnt \
        -v /etc/localtime:/etc/localtime:ro \
        -u $(id -u):$(id -g) \
        --net=host \
        party_image ./environment/run_in_env.sh $1 $2

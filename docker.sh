#! /bin/bash
source environment/build_docker.sh $1

docker run -ti --rm \
        -v ~:/$HOME \
        --shm-size=16G\
        $all_gpus \
        -v /mnt:/mnt \
        -v /etc/localtime:/etc/localtime:ro \
        -u $(id -u):$(id -g) \
        --net=host \
        party_image bash

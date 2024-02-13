#! /bin/bash

source environment/utils.sh

# if second argument is fast sorce venv directly else build it
if [[ $2 == "fast" ]]; then
    source venv/bin/activate
else
    __banner "Varify environment is up"

    source ./env.sh
fi

./$1
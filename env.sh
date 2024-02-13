#! /bin/bash
source environment/utils.sh

if ! (return 0 2>/dev/null); then
  __banner "Run this script with: source ./env.sh"
  exit 1
fi

if [[ $1 == "clean" ]]; then 
    __banner Remove old env...
    if [[ -d "venv" ]]; then 
        rm -rf venv/ 
    fi
fi

python3 -m venv venv


__banner Instaling env packages... 
./venv/bin/pip install -e .


chmod +x venv/bin/activate

# start env
source ./venv/bin/activate

__banner "You are now ready to party!"
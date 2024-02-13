#! /bin/bash

source environment/utils.sh

__banner "Starting Jupyter Notebook"


jupyter notebook --no-browser --NotebookApp.allow_origin='*' --NotebookApp.token='docker' --NotebookApp.password='docker'
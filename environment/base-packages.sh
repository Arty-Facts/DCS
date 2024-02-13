#!/bin/bash

# Install base packages
apt update
apt install -y --no-install-recommends --fix-missing\
    build-essential\
    default-libmysqlclient-dev\
    emacs\
    git\
    libgl1\
    openssh-client\
    pciutils\
    pkg-config\
    python3.10-dev\
    python3.10-venv\
    python3.10\
    software-properties-common\
    sudo\
    unzip\
    vim

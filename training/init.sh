#!/bin/bash
USER_HOME=/weka/pfeifer/ppu738
BASE_DIR_LOCAL=$USER_HOME/TabPFN-Wide


HF_HOME=$BASE_DIR_LOCAL/huggingface
CONDA_ENV_NAME=TabPFN-Wide

cd $BASE_DIR_LOCAL
source $USER_HOME/.bashrc


### Check global conda installation
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Using miniconda locally."
    if [ ! -d $USER_HOME/miniconda3 ]; then
        echo "Installing Miniconda..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $USER_HOME/miniconda.sh
        bash $USER_HOME/miniconda.sh -b -p $USER_HOME/miniconda3
        rm $USER_HOME/miniconda.sh
    fi
    source $USER_HOME/miniconda3/bin/activate
    conda init bash
    if ! command -v conda &> /dev/null; then
        echo "Error: Conda is still not available after installation. Exiting."
        exit 1
    fi
fi
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Activate the conda environment
CONDA_DIR=$USER_HOME/miniconda3/envs/$CONDA_ENV_NAME
if [ ! -d $CONDA_DIR ]; then
    echo "Creating conda environment..."
    conda create -n $CONDA_ENV_NAME python=3.12 -y
    conda activate $CONDA_ENV_NAME
    pip install -r requirements.txt
else
    echo "Activating existing conda environment..."
    conda activate $CONDA_ENV_NAME
fi
### End conda

export PYTORCH_ALLOC_CONF=expandable_segments:True

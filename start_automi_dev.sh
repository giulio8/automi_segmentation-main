#!/bin/bash

# === CONFIG ===
PROJECT_DIR=$(pwd)
BASE_DIR="/home/giulio/Scrivania/tesi"
DATASET_DIR="$BASE_DIR/Dataset003_AUTOMI_CTVLNF_NEWGL_preprocessed_files"
RESULTS_DIR="$BASE_DIR/Dataset003_AUTOMI_CTVLNF_NEWGL_results"

# === Check if Docker daemon is running ===
if ! systemctl is-active --quiet docker; then
    echo "🔄 Docker non è in esecuzione. Lo avvio..."
    sudo systemctl start docker
fi

echo "Avvio container Automi in modalità sviluppo..."

sudo docker run -it --rm \
    -v $PROJECT_DIR:/automi_seg \
    -v $DATASET_DIR:/automi_seg/data \
    -v $RESULTS_DIR:/automi_seg/results \
    automi-seg \
    /bin/bash

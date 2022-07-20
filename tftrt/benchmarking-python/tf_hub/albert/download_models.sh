#!/bin/bash

usage() {                                 # Function: Print a help message.
  echo "This script download the TF-HUB models inside a given directory." 1>&2
  echo "Usage: $0 --directory=/path/to/model/dir" 1>&2
}

exit_abnormal() {                         # Function: Exit with error.
  usage
  exit 1
}

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
      -h | --help ) usage; exit 0 ;;
      --directory=*)
        DESTINATION_DIR="${arg#*=}"
        shift # Remove --model_name from processing
        ;;
      * ) exit_abnormal ;;
  esac
done

if [[ -z ${DESTINATION_DIR} ]]; then exit_abnormal; fi

echo "Will download models inside: ${DESTINATION_DIR}"
mkdir -p ${DESTINATION_DIR}

MODEL_DATA=(
    "albert_base https://storage.googleapis.com/tfhub-modules/tensorflow/albert_en_base/3.tar.gz"
    "albert_large https://storage.googleapis.com/tfhub-modules/tensorflow/albert_en_large/3.tar.gz"
    "albert_xlarge https://storage.googleapis.com/tfhub-modules/tensorflow/albert_en_xlarge/3.tar.gz"
    "albert_xxlarge https://storage.googleapis.com/tfhub-modules/tensorflow/albert_en_xxlarge/3.tar.gz"
    "tokenizer https://storage.googleapis.com/tfhub-modules/tensorflow/albert_en_preprocess/3.tar.gz"
)

for model_data in "${MODEL_DATA[@]}"; do

    set -- $model_data # convert the "tuple" into the param args $1 $2...
    MODEL_NAME=$1
    MODEL_URL=$2

    echo "Downloading ${MODEL_NAME} at: ${MODEL_URL}"

    MODEL_DIR="${DESTINATION_DIR}/${MODEL_NAME}"
    rm -rf ${MODEL_DIR}
    mkdir -p ${MODEL_DIR}
    cd ${MODEL_DIR}

    wget -O model.tar.gz ${MODEL_URL}
    tar -xzf model.tar.gz

    saved_model_cli show --dir $(pwd) --all 2>&1 | tee analysis.txt

done

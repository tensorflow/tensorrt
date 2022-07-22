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
  "swin_base_patch4_window12_384 https://storage.googleapis.com/tfhub-modules/sayakpaul/swin_base_patch4_window12_384/1.tar.gz"
  "swin_base_patch4_window7_224 https://storage.googleapis.com/tfhub-modules/sayakpaul/swin_base_patch4_window7_224/1.tar.gz"
  "swin_large_patch4_window12_384 https://storage.googleapis.com/tfhub-modules/sayakpaul/swin_large_patch4_window12_384/1.tar.gz"
  "swin_large_patch4_window7_224 https://storage.googleapis.com/tfhub-modules/sayakpaul/swin_large_patch4_window7_224/1.tar.gz"
  "swin_small_patch4_window7_224 https://storage.googleapis.com/tfhub-modules/sayakpaul/swin_small_patch4_window7_224/1.tar.gz"
  "swin_tiny_patch4_window7_224 https://storage.googleapis.com/tfhub-modules/sayakpaul/swin_tiny_patch4_window7_224/1.tar.gz"
  "swin_base_patch4_window12_384_in22k https://storage.googleapis.com/tfhub-modules/sayakpaul/swin_base_patch4_window12_384_in22k/1.tar.gz"
  "swin_base_patch4_window7_224_in22k https://storage.googleapis.com/tfhub-modules/sayakpaul/swin_base_patch4_window7_224_in22k/1.tar.gz"
  "swin_large_patch4_window12_384_in22k https://storage.googleapis.com/tfhub-modules/sayakpaul/swin_large_patch4_window12_384_in22k/1.tar.gz"
  "swin_large_patch4_window7_224_in22k https://storage.googleapis.com/tfhub-modules/sayakpaul/swin_large_patch4_window7_224_in22k/1.tar.gz"
  "swin_s3_tiny_224 https://storage.googleapis.com/tfhub-modules/sayakpaul/swin_s3_tiny_224/1.tar.gz"
  "swin_s3_small_224 https://storage.googleapis.com/tfhub-modules/sayakpaul/swin_s3_small_224/1.tar.gz"
  "swin_s3_base_224 https://storage.googleapis.com/tfhub-modules/sayakpaul/swin_s3_base_224/1.tar.gz"
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

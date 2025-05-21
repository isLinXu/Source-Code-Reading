#!/bin/bash
#SBATCH --job-name=processing_pipeline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=output/%x-%a.out
#SBATCH --error=output/%x-%a.out
#SBATCH --partition=cpu_p1
#SBATCH --account=six@cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victor@huggingface.co

set -x -e
echo "START TIME: $(date)"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate victor-m4

export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error

N_WORKERS=40 # Set equal to the number
DATA_PATH=$six_ALL_CCFRSCRATCH/m4/cc_processed
SHARD_NAME=b613058d54c8d2011acf8e251865b6a3af758ed4100de2c525c103f8b217423c #TODO
NB_DOCS_PER_SUBSHARD=375 # Value chose based on the assumption that a shard has 15_000 lines

# Unzip
gzip -dkf $DATA_PATH/raw_dumps/$SHARD_NAME.gz

# Get the text field only
jq ".text" < $DATA_PATH/raw_dumps/$SHARD_NAME > $DATA_PATH/processed_dumps/$SHARD_NAME.texts
# Get the URL field
jq -r "[input_line_number,.url] | @csv" < $DATA_PATH/raw_dumps/$SHARD_NAME > $DATA_PATH/extracted_databases/$SHARD_NAME.urls.csv
# Get the HTML field
jq -r "[input_line_number,.html] | @csv" < $DATA_PATH/raw_dumps/$SHARD_NAME > $DATA_PATH/extracted_databases/$SHARD_NAME.htmls.csv

# Splitting into subshards
split --lines $NB_DOCS_PER_SUBSHARD --numeric-suffixes $DATA_PATH/processed_dumps/$SHARD_NAME.texts $DATA_PATH/processed_dumps/$SHARD_NAME.texts.

# Extract ngrams in each documents
# TODO: might want to check the return value in the parallel
find $DATA_PATH/processed_dumps/ | \
    grep "${DATA_PATH}/processed_dumps/${SHARD_NAME}.texts.[0-9]+*" | \
    parallel --verbose -j $N_WORKERS --tmpdir $WORK/tmp/ --progress "python extract_documents_ngrams.py --filepath {} --nb_docs_per_subshard $NB_DOCS_PER_SUBSHARD" > \
    $DATA_PATH/extracted_databases/$SHARD_NAME.ngrams.csv

# Remove the subshards
find $DATA_PATH/processed_dumps/ | grep "${DATA_PATH}/processed_dumps/${SHARD_NAME}.texts*" | xargs -d"\n" rm
# Remove unzipped shard
rm $DATA_PATH/raw_dumps/$SHARD_NAME

echo "END TIME: $(date)"

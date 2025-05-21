eval "$(conda shell.bash hook)"
conda activate m3

# TODO: update so that we can take in multiple shards
N_WORKERS=4
DATA_PATH=/home/victor_huggingface_co/m4/data
SHARD_NAME=4e47925f7c894bd8eb56e5dd1d778ec77bf2c90f6cee0e32e31615393391c67a
NB_DOCS_PER_SUBSHARD=2000

# Get the text field only
jq ".text" < $DATA_PATH/raw_dumps/$SHARD_NAME > $DATA_PATH/processed_dumps/$SHARD_NAME.texts
# Get the URL field only
jq -r "[input_line_number,.url] | @csv" < $DATA_PATH/raw_dumps/$SHARD_NAME > $DATA_PATH/extracted_databases/$SHARD_NAME.urls.csv
# Get the HTML field only
jq -r "[input_line_number,.html] | @csv" < $DATA_PATH/raw_dumps/$SHARD_NAME > $DATA_PATH/extracted_databases/$SHARD_NAME.htmls.csv

# Splitting into subshards
split --lines $NB_DOCS_PER_SUBSHARD --numeric-suffixes $DATA_PATH/processed_dumps/$SHARD_NAME.texts $DATA_PATH/processed_dumps/$SHARD_NAME.texts.

# Extract ngrams in each documents
find $DATA_PATH/processed_dumps/ | \
    grep "${DATA_PATH}/processed_dumps/${SHARD_NAME}.texts.[0-9]+*" | \
    parallel --verbose -j $N_WORKERS --progress "TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error python extract_documents_ngrams.py --filepath {} --nb_docs_per_subshard $NB_DOCS_PER_SUBSHARD" > \
    $DATA_PATH/extracted_databases/$SHARD_NAME.ngrams.csv

# Remove the subshards
find $DATA_PATH/processed_dumps/ | grep "${DATA_PATH}/processed_dumps/${SHARD_NAME}.texts.[0-9]+*" | xargs -d"\n" rm

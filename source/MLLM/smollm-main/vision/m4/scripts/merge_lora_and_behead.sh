set -x -e

source /fsx/m4/start-m4-user
conda activate victor

INPUT_DIR=/fsx/m4/experiments/local_experiment_dir/tr_289_288_ter_12600_lima_sft/opt_step-1400
OUTPUT_DIR=/fsx/m4/victor/idefics2

SCRIPT_RELATIVE_PATH="${BASH_SOURCE[0]}"
PATH_TO_THIS_FILE=$(realpath "$SCRIPT_RELATIVE_PATH")
echo "The absolute path of the current script file is: $PATH_TO_THIS_FILE"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WORKING_DIR=$(builtin cd $SCRIPT_DIR/; pwd)
echo "Working dir is: $WORKING_DIR"

cd $WORKING_DIR


python merge_lora_and_save.py $INPUT_DIR $OUTPUT_DIR
echo "Finished merge lora"
mv $OUTPUT_DIR/unwrapped_model/model* $OUTPUT_DIR
rm -rf $OUTPUT_DIR/unwrapped_model
rm -rf $OUTPUT_DIR/tokenizer # Just a sanity


python behead_unused_params.py \
    --model_dir $OUTPUT_DIR \
    --behead_siglip_pooling \
    --behead_perceiver_rmsnorm
echo "Finished behead unused parameters"

# Push `/fsx/m4/victor/idefics2` to `HuggingFaceM4/idefics2`
# Then call optionally to transform into transformers compatible checkpoint and push to `HuggingFaceM4/idefics2-tfrm-compatible`
# python transformers/src/transformers/models/idefics2/convert_idefics2_weights_to_hf.py \
#     --original_model_id HuggingFaceM4/idefics2 \
#     --output_hub_path /fsx/m4/victor/idefics2-tfrm-compatible

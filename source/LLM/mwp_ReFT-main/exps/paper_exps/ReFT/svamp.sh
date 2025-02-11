### svamp 
## Python SDP
# Codellama
exp_name="svamp_python_sdp_codellama_reft" \
train_file='data/svamp_python_sdp.json' \
test_file='data/svamp_test_set.json' \
engine='python' \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/svamp_python_sdp_codellama/global_step_128_epoch_2/' \
ref_model_name_or_path='ppo_paper_final_new/_models_outputs_sft/svamp_python_sdp_codellama/global_step_128_epoch_2/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/svamp_python_sdp_codellama/global_step_128_epoch_2/' \
n_epochs='300' \
kl_coef='0.01' \
    bash exps/paper_exps/ReFT/_template.sh

# Galactica
exp_name="svamp_python_sdp_galactica_reft" \
train_file='data/svamp_python_sdp.json' \
test_file='data/svamp_test_set.json' \
engine='python' \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/svamp_python_sdp_galactica/global_step_128_epoch_2/' \
ref_model_name_or_path='ppo_paper_final_new/_models_outputs_sft/svamp_python_sdp_galactica/global_step_128_epoch_2/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/svamp_python_sdp_galactica/global_step_128_epoch_2/' \
n_epochs='300' \
kl_coef='0.01' \
    bash exps/paper_exps/ReFT/_template.sh


## NL
# Codellama
exp_name="svamp_nl_codellama_reft" \
train_file='data/svamp_nl.json' \
test_file='data/svamp_test_set.json' \
engine='nl' \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/svamp_nl_codellama/global_step_130_epoch_2/' \
ref_model_name_or_path='ppo_paper_final_new/_models_outputs_sft/svamp_nl_codellama/global_step_130_epoch_2/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/svamp_nl_codellama/global_step_130_epoch_2/' \
n_epochs='300' \
kl_coef='0.05' \
    bash exps/paper_exps/ReFT/_template.sh

# Galactica
exp_name="svamp_nl_galactica_reft" \
train_file='data/svamp_nl.json' \
test_file='data/svamp_test_set.json' \
engine='nl' \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/svamp_nl_galactica/global_step_130_epoch_2/' \
ref_model_name_or_path='ppo_paper_final_new/_models_outputs_sft/svamp_nl_galactica/global_step_130_epoch_2/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/svamp_nl_galactica/global_step_130_epoch_2/' \
n_epochs='300' \
kl_coef='0.05' \
    bash exps/paper_exps/ReFT/_template.sh


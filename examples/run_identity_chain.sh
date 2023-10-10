# set your GPU device id
export CUDA_VISIBLE_DEVICES=0

# set your HuggingFace home path and your IdentityChain home path
export HF_HOME=YOUR_OWN_PATH/huggingface
export IDENTITY_CHAIN_HOME=YOUR_OWN_PATH/IdentityChain  # no / at the end

# the folloiwng parameters
# --seq_length 1536,
# --gen_len 512,
# --chain_length 5
# are only for experiments in the paper, feel free use smaller values to save time
# however, the results may be different from what we report in the paper

# for open-source models from HuggingFace, when using greedy, add the flag --greedy_early_stop to accelerate
# for OpenAI models, don't use --greedy_early_stop!!! temperature = 0 is NOT greedy!!!
# for HumanEvalPlus-Mini-v0.1.6_reformatted.jsonl, use --resume_task_bs 1, since HumanEval/0 is used for prompt
# for MBPP-S_test_reformatted.jsonl, use --resume_task_bs 0, since there's a separate prompt split

for MODEL in "bigcode/starcoderbase-1b"  # feel free to add other supported models
do
	for DATASET in "EvalPlus-Mini-v0.1.6_reformatted.jsonl"  # feel free add other supported datasets
	do
		for TMP in 0  # feel free to add other temperatures, add the flag --do_sample to use temperature sampling
		do
			MODEL_NAME=$(basename $MODEL)
			OUTPUT_DIR=${IDENTITY_CHAIN_HOME}/tmp/${MODEL_NAME}
			mkdir -p $OUTPUT_DIR
			# change this to python run_identity_chain_openai.py if you are evaluating OpenAI models
			python run_identity_chain_huggingface.py \
				--model_name_or_path $MODEL \
				--hf_dir $HF_HOME \
				--input_path ${IDENTITY_CHAIN_HOME}/tmp/${DATASET} \
				--output_dir $OUTPUT_DIR \
				--seq_length 1536 \
				--gen_len 512 \
				--greedy_early_stop \
				--chain_length 5 \
				--use_fp16 \
				--mask_func_name \
				--bootstrap_method problem \
				--resume_task_bs 1 \
				--resume_task_run 0 \
				--temperature $TMP
		done
	done
done
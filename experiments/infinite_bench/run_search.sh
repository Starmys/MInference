# cd experiments/infinite_bench
python run_infinitebench.py \
    --task kv_retrieval \
    --model_name_or_path /home/chengzhang/models/Qwen2.5-3B-YaRN-128k \
    --data_dir ./data \
    --output_dir ./results \
    --max_seq_length 30000 \
    --rewrite \
    --is_search \
    --trust_remote_code \
    --start_example_id 3 \
    --topk_dims_file_path Qwen2.5_3B_YaRN_128k_kv_out_v32_fit_o_best_pattern.json \
    --num_eval_examples 20 --topk 1 --starting_layer 0 --attn_type minference

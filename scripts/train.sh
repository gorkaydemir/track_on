export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
export WORLD_SIZE=32

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python main.py \
--movi_f_root "/root/to/movi_f" \
--tapvid_root "/root/to/tap_vid" \
--eval_dataset "eval_dataset" \
--augmentation \
--input_size 384 512 \
--N 480 \
--T 24 \
--stride 4 \
--transformer_embedding_dim 256 \
--num_layers 3 \
--num_layers_offset_head 3 \
--num_layers_rerank 3 \
--num_layers_rerank_fusion 1 \
--top_k_regions 16 \
--num_layers_spatial_writer 3 \
--num_layers_spatial_self 1 \
--num_layers_spatial_cross 1 \
--memory_size 12 \
--random_memory_mask_drop 0.1 \
--lambda_point 3.0 \
--lambda_vis 1.0 \
--lambda_offset 1.0 \
--lambda_uncertainty 1.0 \
--lambda_top_k 1.0 \
--epoch_num 150 \
--lr 5e-4 \
--wd 1e-5 \
--bs 1 \
--amp \
--model_save_path "checkpoints/training_name" \
--loss_after_query \
--seed 1234
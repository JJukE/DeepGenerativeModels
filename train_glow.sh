OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3,4,5,6 \
torchrun --standalone --nnodes=1 --nproc_per_node=5 train_glow.py \
--num_epoch 400 --batch_size 128 --save_epoch_step 50 \
--output_dir "/root/dev/deepul/exp"
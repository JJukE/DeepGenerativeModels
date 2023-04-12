OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
torchrun --standalone --nnodes=1 --nproc_per_node=6 train_IAF.py \
--num_epoch 10000 --batch_size 128 --save_epoch_step 400 \
--output_dir "/root/dev/deepul/exp"
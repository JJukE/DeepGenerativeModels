OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6 \
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_multi_solver.py \
--num_epoch 100 --batch_size 128 --save_epoch_step 20
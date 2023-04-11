OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --standalone --nnodes=1 --nproc_per_node=6 train_solver.py \
--num_epoch 100 --batch_size 128 --save_epoch_step 50
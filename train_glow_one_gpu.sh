OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=6 \
python train_glow.py \
--num_epoch 100 --batch_size 64 --save_epoch_step 50
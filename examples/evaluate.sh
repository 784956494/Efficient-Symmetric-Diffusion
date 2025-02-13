python run.py --dim 2 --epoch 5000 --device cpu \
    --hidden_dim 512 --num_layers 4 --lr 5e-4 --act sin \
    --batch_size 512 --seed 1234 --beta_s 0.1\
    --beta_f 0.3 --mode eval --model_directory trained_models/torus/ \
    --data_path dataset/torus/T2.npy --manifold torus --model_type MLP\
    --data_type torch.float32 --ckpt_directory trained_models/torus/\
    --figure_path figures/torus/
## train autoencoder for clip features ###
echo "Training autoencoder with clip features"
dataset_name=flame_salmon
dataset_path=dataset/N3DV/flame_salmon/cam00
clip_feature_name=clip_features
clip_dim=3
 python train.py --lr 7e-4 --dataset_path ${dataset_path} --model_name ${dataset_name}_clip --feature_dims 512  \
     --encoder_dims 256 128 64 32 ${clip_dim} --decoder_dims 16 32 64 128 256 512 --hidden_dims ${clip_dim} --language_name ${clip_feature_name}

python test.py --dataset_path ${dataset_path} --model_name ${dataset_name}_clip --feature_dims 512 \
    --encoder_dims 256 128 64 32 ${clip_dim} --decoder_dims 16 32 64 128 256 512 --hidden_dims ${clip_dim} --language_name ${clip_feature_name}


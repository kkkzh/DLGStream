import os
import glob
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder,VanillaVAE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )
    parser.add_argument('--hidden_dims',
                    type=int,
                    default=3
                    )
    parser.add_argument('--feature_dims',
                    type=int,
                    default=512
                    )
    # parser.add_argument("--output_dim", type=int, default=3)
    parser.add_argument('--language_name', type = str, default = None)
    parser.add_argument('--output_name',type=str,default=None)
    args = parser.parse_args()

    model_name = args.model_name
    encoder_hidden_dims = args.encoder_dims
    # encoder_hidden_dims[-1] = args.output_dim
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    ckpt_path = f"ckpt/{model_name}/best_ckpt.pth"

    # data_dir = f"{dataset_path}/language_features"
    if args.language_name is None:
        data_dir = f"{dataset_path}/language_features"  
        output_dir = f"{dataset_path}/language_features_dim{encoder_hidden_dims[-1]}"
    else:
        data_dir = os.path.join(dataset_path, args.language_name)
        if args.output_name is not None:
            # output_dir = os.path.join(dataset_path,f"{args.language_name}-{args.output_name}_dim{encoder_hidden_dims[-1]}")
            output_name = f"{args.language_name}-{args.output_name}_dim{encoder_hidden_dims[-1]}"
        else:
            # output_dir = os.path.join(dataset_path,f"{args.language_name}-language_features_dim{encoder_hidden_dims[-1]}")
            output_name = f"{args.language_name}-language_features_dim{encoder_hidden_dims[-1]}"
    # os.makedirs(output_dir, exist_ok=True)

    # copy the segmentation map
    # from tqdm import tqdm
    # for filename in tqdm(os.listdir(data_dir)):
    #     if filename.endswith("_s.npy"):
    #         source_path = os.path.join(data_dir, filename)
    #         target_path = os.path.join(output_dir, filename)
    #         shutil.copy(source_path, target_path)
    root_dir = os.path.dirname(os.path.dirname(data_dir))
    feature_type = data_dir.split("/")[-1]
    videos = glob.glob(os.path.join(root_dir, "cam*"))
    videos = sorted(videos)
    print('camera number:', len(videos))

    for index, video_path in enumerate(videos):
        # if index == 0:
        #     continue

        camera_path = video_path.split('.')[0]
        feature_dir = os.path.join(camera_path, feature_type)
        output_feature_dir = os.path.join(camera_path, output_name)
        os.makedirs(output_feature_dir, exist_ok=True)
        print(output_feature_dir)

        feature_path = os.listdir(feature_dir)
        for idx, filename in enumerate(feature_path):
            if "_s.npy" in filename:
                source_path = os.path.join(feature_dir, filename)
                target_path = os.path.join(output_feature_dir, filename)
                if not os.path.exists(target_path):
                    os.symlink(source_path, target_path)

    checkpoint = torch.load(ckpt_path)
    train_dataset = Autoencoder_dataset(data_dir, skip_test=False)

    test_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=256,
        shuffle=False, 
        num_workers=16, 
        drop_last=False   
    )

    if os.getenv("use_vae",'f') == 't':
        model = VanillaVAE(encoder_hidden_dims, decoder_hidden_dims, latent_dim=9).to("cuda:0")
    else:
        model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims, feature_dim=args.feature_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    for idx, feature in tqdm(enumerate(test_loader), total=len(test_loader), desc="Encoding clip features"):
        data = feature.to("cuda:0")
        data = data.to(torch.float32)

        with torch.no_grad():
            if os.getenv("use_vae",'f') == 't':
                mu, log_var  = model.encode(data)
                outputs = model.reparameterize(mu, log_var).to('cpu').numpy()  

            else:
                outputs = model.encode(data).to("cpu").numpy()  
        if idx == 0:
            features = outputs
        else:
            features = np.concatenate([features, outputs], axis=0)

    start = 0
    for k,v in train_dataset.data_dic.items():
        cam_name = k.split("-")
        cam_id, filename = cam_name[0], cam_name[1]
        path = os.path.join(os.path.dirname(dataset_path), cam_id, output_name, filename)
        np.save(path, features[start:start+v])
        start += v

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder,VanillaVAE
from torch.utils.tensorboard import SummaryWriter
import argparse
import random
# import wandb
from torch.utils.data import random_split

torch.autograd.set_detect_anomaly(True)

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=1).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[32, 64, 128, 256, 256, 512],
                    )
    parser.add_argument('--hidden_dims',
                    type=int,
                    default=3
                    )
    parser.add_argument('--feature_dims',
                    type=int,
                    default=512
                    )
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--cos_weight',type=float,default=1e-3)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--language_name', type = str, default = None)

    args = parser.parse_args()
    dataset_path = args.dataset_path
    num_epochs = args.num_epochs
    # import ipdb; ipdb.set_trace()
    if args.language_name is None:
        data_dir = f"{dataset_path}/language_features"  
    else:
        data_dir = os.path.join(dataset_path, args.language_name)
    os.makedirs(f'ckpt/{args.model_name}', exist_ok=True)
    train_dataset = Autoencoder_dataset(data_dir)
    if os.getenv("split_dataset",'f') == 't':
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
    else:
        test_dataset = train_dataset

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=6,
        drop_last=False  
    )

    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    assert args.hidden_dims == encoder_hidden_dims[-1]
    assert args.feature_dims == decoder_hidden_dims[-1]

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims, feature_dim=args.feature_dims).to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logdir = f'ckpt/{args.model_name}'
    if os.getenv("wandb",'f') == 't':
        exp_name = os.getenv("expname",'default')
        wandb.init(project="4DLangSplat-autoencoder", name=f"{args.model_name}-{args.language_name}-{exp_name}", config=args)

    tb_writer = SummaryWriter(logdir)

    best_eval_loss = 100.0
    best_epoch = 0
    if os.getenv("use_adaptive_beta",'f') == 't':
        beta_start = 0.01
        beta_end = 1
        beta_interval = 0.1
        beta_coff = beta_start
    else:
        beta_coff = 1
    for epoch in range(num_epochs):
        model.train()
        for idx, feature in tqdm(enumerate(train_loader), total=len(train_loader)):
            data = feature.to("cuda:0")
            data = data.to(torch.float32)

            outputs_dim3 = model.encode(data)
            outputs = model.decode(outputs_dim3)

            l2loss = l2_loss(outputs, data) 
            # import ipdb; ipdb.set_trace()
            cosloss = cos_loss(outputs, data)
            loss = l2loss + cosloss * args.cos_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_iter = epoch * len(train_loader) + idx
            if os.getenv("wandb",'f') == 't':
                resdict = {
                    'train_loss/l2_loss':l2loss.item(),
                    'train_loss/cos_loss': cosloss.item(),
                    'train_loss/total_loss': loss.item()
                }

                wandb.log(resdict)

            tb_writer.add_scalar('train_loss/l2_loss', l2loss.item(), global_iter)
            tb_writer.add_scalar('train_loss/cos_loss', cosloss.item(), global_iter)

            tb_writer.add_scalar('train_loss/total_loss', loss.item(), global_iter)
            # tb_writer.add_histogram("feat", outputs, global_iter)

        if epoch > 90:
            eval_loss = 0.0
            eval_l2loss = 0.0
            eval_cosloss = 0.0
            model.eval()
            for idx, feature in enumerate(test_loader):
                data = feature.to("cuda:0")
                data = data.to(torch.float32)
                with torch.no_grad():
                    outputs = model(data) 
                    l2loss = l2_loss(outputs, data)
                    cosloss = cos_loss(outputs, data)
                    loss = l2loss + cosloss
                eval_l2loss += l2loss * len(feature)
                eval_cosloss += cosloss * len(feature)
                eval_loss += loss * len(feature)
            eval_loss = eval_loss / len(test_dataset)
            eval_cosloss = eval_cosloss / len(test_dataset)
            eval_l2loss = eval_l2loss / len(test_dataset)
            print("eval_loss:{:.8f}".format(eval_loss))
            if os.getenv("wandb",'f') == 't':
                resdict = {
                    'eval_loss/l2_loss':eval_l2loss.item(),
                    'eval_loss/cos_loss': eval_cosloss.item(),
                    'eval_loss/total_loss': eval_loss.item()
                }
                wandb.log(resdict)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                torch.save(model.state_dict(), f'ckpt/{args.model_name}/best_ckpt.pth')

            if epoch % 10 == 0:
                torch.save(model.state_dict(), f'ckpt/{args.model_name}/{epoch}_ckpt.pth')

    print(f"best_epoch: {best_epoch}")
    print("best_loss: {:.8f}".format(best_eval_loss))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import argparse
from tensorboardX import SummaryWriter
import datetime
from vqvae.network import VQVAE

# dataset 
from datasets import ShapeNetDataset

class VQLoss(nn.Module):
    def __init__(self, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, codebook_loss, inputs, reconstructions, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        loss = nll_loss + self.codebook_weight * codebook_loss.mean()

        log = {
            "loss_total": loss.clone().detach().mean(),
            "loss_codebook": codebook_loss.detach().mean(),
            "loss_nll": nll_loss.detach().mean(),
            "loss_rec": rec_loss.detach().mean(),
        }

        return loss, log


def train_vqvae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_config = {
        "embed_dim": 3,
        "n_embed": 8192,
        "ddconfig": {
            "double_z": False,
            "z_channels": 3,
            "resolution": 64,
            "in_channels": 1,
            "out_ch": 1,
            "ch": 64,
            "ch_mult": [1, 2, 4],  
            "num_res_blocks": 1,
            "attn_resolutions": [],
            "dropout": 0.0
        }
    }

    loss_config = {
        "codebook_weight": 1.0
    }

    dataset_config = {
        "info_file": './dataset_info_files/info-shapenet.json',
        "dataroot": './data',
        "phase": 'train',
        "cat": 'all',
        "res": 64,
        "trunc_thres": 0.2
    }


    vqvae = VQVAE(
        ddconfig=model_config["ddconfig"],
        n_embed=model_config["n_embed"],
        embed_dim=model_config["embed_dim"],
        remap=None,
        sane_index_shape=False
    )
    vqvae = vqvae.to(device)

    print("VQVAE Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in vqvae.parameters() if p.requires_grad):,}")

    loss_fn = VQLoss(codebook_weight=loss_config["codebook_weight"]).to(device)

    optimizer = optim.Adam(vqvae.parameters(), lr=1e-4, betas=(0.5, 0.9))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    dataset = ShapeNetDataset(
        info_file=dataset_config["info_file"],
        dataroot=dataset_config["dataroot"],
        phase=dataset_config["phase"],
        cat=dataset_config["cat"],
        res=dataset_config["res"],
        trunc_thres=dataset_config["trunc_thres"]
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    num_epochs = args.num_epochs
    save_interval = args.save_interval if hasattr(args, 'save_interval') else 10
    
    os.makedirs(args.log_dir, exist_ok=True)

    # use the current datetime for logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, current_time)
    os.makedirs(log_dir, exist_ok=True)
    
    # save the model configuration
    with open(os.path.join(log_dir, 'model_config.txt'), 'w') as f:
        f.write(str(model_config))
    
    writer = SummaryWriter(log_dir=log_dir)

    vqvae.train()
    global_step = 0
    
    for epoch in range(num_epochs):        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, data in enumerate(pbar):
            x = data['sdf'].to(device)
            
            optimizer.zero_grad()
            x_recon, qloss = vqvae(x, verbose=False)
            
            total_loss, log_dict = loss_fn(qloss, x, x_recon, split="train")
            
            total_loss.backward()
            optimizer.step()
                        
            writer.add_scalar('Loss/Train/Total', log_dict['loss_total'].item(), global_step)
            writer.add_scalar('Loss/Train/Reconstruction', log_dict['loss_rec'].item(), global_step)
            writer.add_scalar('Loss/Train/Codebook', log_dict['loss_codebook'].item(), global_step)
            writer.add_scalar('Loss/Train/NLL', log_dict['loss_nll'].item(), global_step)
            
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', current_lr, global_step)
            
            pbar.set_postfix({
                'Total': f"{log_dict['loss_total'].item():.4f}",
                'Rec': f"{log_dict['loss_rec'].item():.4f}",
                'VQ': f"{log_dict['loss_codebook'].item():.4f}"
            })
            
            global_step += 1
            
        scheduler.step()
        
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(log_dir, f'vqvae_epoch_{epoch+1}.pth')
            torch.save(vqvae.state_dict(), save_path)
            print(f"Model saved: {save_path}")
        
    
    torch.save(vqvae.state_dict(),  os.path.join(log_dir, 'vqvae_final.pth'))
    print("Training completed! Final model saved.")
    
    writer.close()



def test_model(model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_config = {
        "embed_dim": 3,
        "n_embed": 8192,
        "ddconfig": {
            "double_z": False,
            "z_channels": 3,
            "resolution": 64,
            "in_channels": 1,
            "out_ch": 1,
            "ch": 64,
            "ch_mult": [1, 2, 4],  
            "num_res_blocks": 1,
            "attn_resolutions": [],
            "dropout": 0.0
        }
    }

    vqvae = VQVAE(
        ddconfig=model_config["ddconfig"],
        n_embed=model_config["n_embed"],
        embed_dim=model_config["embed_dim"],
        remap=None,
        sane_index_shape=False
    )

    vqvae.load_state_dict(torch.load(model_path, map_location=device)) if model_path else None
    print("VQVAE Model loaded successfully!")

    vqvae = vqvae.to(device)

    test_input = torch.randn(2, 1, 64, 64, 64).to(device)

    with torch.no_grad():
        try:
            output, vq_loss = vqvae(test_input, verbose=False)
            print(f"Input shape: {test_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"VQ loss: {vq_loss.item():.4f}")
            
            loss_fn = VQLoss(codebook_weight=1.0)
            total_loss, log_dict = loss_fn(vq_loss, test_input, output)
            print(f"Total loss: {total_loss.item():.4f}")
            print(f"Loss dict: {log_dict}")
            
            z = vqvae(test_input, forward_no_quant=True, encode_only=True)
            print(f"Encoded z shape: {z.shape}")
            
            decoded = vqvae.decode_no_quant(z)
            print(f"Decoded shape: {decoded.shape}")
            
            print("All tests passed!")
            
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VQVAE Training Script')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                        help='Mode: train or test')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs for training (only used in train mode)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (only used in train mode)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimizer (only used in train mode)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Interval for saving model checkpoints (only used in train mode)')
    parser.add_argument('--log_dir', type=str, default='logs_vqvae',
                        help='Directory for saving logs (only used in train mode)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to a pre-trained model checkpoint (only used in test mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_vqvae(args)
    elif args.mode == 'test':
        test_model(args.checkpoint_path)
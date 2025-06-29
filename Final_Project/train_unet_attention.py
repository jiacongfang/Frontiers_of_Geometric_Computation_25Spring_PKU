import torch
import os
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse

from torch.utils.data import DataLoader
from datasets import TextShapeNetDataset
from vqvae.network import VQVAE
from diffusion_unet.model_attention import (
    DiffusionModelWithAttention,
    DiffusionUNet3DWithAttention,
)


def create_unet_model(model_size="large", device="cuda"):
    configs = {
        "small": {
            "time_emb_dim": 128,
            "f_maps": 64,
            "num_levels": 3,
        },
        "medium": {
            "time_emb_dim": 256,
            "f_maps": 128,
            "num_levels": 3,
        },
        "large": {
            "time_emb_dim": 512,
            "f_maps": 192,
            "num_levels": 3,
        },
        "xlarge": {
            "time_emb_dim": 512,
            "f_maps": 256,
            "num_levels": 3,
        },
    }

    config = configs.get(model_size, configs["medium"])

    unet = DiffusionUNet3DWithAttention(
        in_channels=3,
        out_channels=3,
        attention_resolutions=[2, 4],
        use_cross_attention=True,
        **config,
    ).to(device)

    return unet, configs.get(model_size, configs["medium"])


def train_unet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vqvae_model_config = {
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
            "dropout": 0.0,
        },
    }

    dataset_config = {
        "info_file": "./dataset_info_files/info-shapenet.json",
        "dataroot": "./data",
        "phase": "train",
        "cat": args.category,
        "res": 64,
        "trunc_thres": 0.2,
    }

    # load pretrained VQVAE model
    vqvae = VQVAE(
        ddconfig=vqvae_model_config["ddconfig"],
        n_embed=vqvae_model_config["n_embed"],
        embed_dim=vqvae_model_config["embed_dim"],
        remap=None,
        sane_index_shape=False,
    )

    state_dict = torch.load(args.vqvae_path, map_location=device)
    vqvae.load_state_dict(state_dict)
    vqvae = vqvae.to(device)
    vqvae.eval()

    print(f"VQVAE model loaded from {args.vqvae_path}")

    # load dataset (single category)
    dataset = TextShapeNetDataset(
        info_file=dataset_config["info_file"],
        dataroot=dataset_config["dataroot"],
        phase=dataset_config["phase"],
        cat=dataset_config["cat"],
        res=dataset_config["res"],
        trunc_thres=dataset_config["trunc_thres"],
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    num_epochs = args.num_epochs
    save_interval = args.save_interval if hasattr(args, "save_interval") else 10

    log_dir = f"{args.log_dir}_{args.category}"

    os.makedirs(log_dir, exist_ok=True)

    # use the current datetime for logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, current_time)
    os.makedirs(log_dir, exist_ok=True)

    # save args to log directory
    with open(os.path.join(log_dir, "args.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    writer = SummaryWriter(log_dir=log_dir)

    unet, model_config = create_unet_model(model_size=args.model_size, device=device)
    unet.train()
    print("UNet model created successfully!")

    # create diffusion model
    diffusion_model = DiffusionModelWithAttention(
        vqvae=vqvae, unet=unet, timesteps=args.train_timesteps, device=device
    )

    for param in vqvae.parameters():
        param.requires_grad = False
    for param in unet.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    print(
        f"UNet trainable parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}"
    )
    print(f"VQVAE frozen parameters: {sum(p.numel() for p in vqvae.parameters()):,}")

    global_step = 0
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for _, data in enumerate(pbar):
            x = data["sdf"].to(device)
            text = data["text"]

            # import ipdb; ipdb.set_trace()

            optimizer.zero_grad()

            loss = diffusion_model.training_loss(x, text)
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/Train", loss.item(), global_step)
            writer.add_scalar(
                "Learning_Rate", optimizer.param_groups[0]["lr"], global_step
            )

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            global_step += 1

        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "unet_state_dict": unet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss.item(),
            }
            save_path = os.path.join(log_dir, f"diffusion_unet_epoch_{epoch + 1}.pth")
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved: {save_path}")

    # Save the final model with corresponding configurations
    print("Saving final model...")
    final_checkpoint = {
        "unet_state_dict": unet.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
    }
    # final_checkpoint['model_config'] = model_config

    torch.save(final_checkpoint, os.path.join(log_dir, "diffusion_unet_final.pth"))
    print("Training completed! Final model saved.")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet Training Script")
    parser.add_argument(
        "--vqvae_path",
        type=str,
        default="logs_vqvae/20250622-154347/vqvae_final.pth",
        help="Path to the pretrained VQVAE model",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to UNet checkpoint for resuming training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=40, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=40, help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs_unet_attention",
        help="Directory for saving logs",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument(
        "--train_timesteps",
        type=int,
        default=5000,
        help="Number of timesteps for training diffusion model",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["small", "medium", "large", "xlarge"],
        help="Size of the UNet model to create",
    )

    args = parser.parse_args()
    train_unet(args)

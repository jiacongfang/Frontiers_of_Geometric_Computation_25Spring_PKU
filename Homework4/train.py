import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from dataset import RecTrainDataset, RecTrainDataset_Mixed
from model import RecNet, GaussianFourierFeatureTransform
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time


def args_parser():
    parser = argparse.ArgumentParser(description="Train script for Surface Reconstruction Network")
    parser.add_argument('--data_root', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('--sample_size', type=int, default=25000, help='Batch size for training')
    parser.add_argument('--num_iters', type=int, default=100000, help='Number of iteraions to train')
    parser.add_argument('--checkpoint_iters', type=int, default=20000, help='Frequency of saving checkpoints')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Path to save the model checkpoints')
    parser.add_argument('--lambda_sdf', type=float, default=1.0, help='Weight for the sdf loss term')
    parser.add_argument('--lambda_gradient', type=float, default=1.0, help='Weight for the gradient loss term')
    parser.add_argument('--lambda_eikonal', type=float, default=0.1, help='Weight for the Eikonal loss term')
    
    parser.add_argument('--use_fourier', action='store_true', help='Use Fourier features')
    parser.add_argument('--fourier_mapping_size', type=int, default=256, help='Mapping size for Fourier features')
    parser.add_argument('--fourier_scale', type=float, default=10.0, help='Scale for Fourier features')

    parser.add_argument('--mix_dataset', action='store_true', help='Use mixed dataset with both point cloud and SDF samples')

    return parser.parse_args()


def compute_gradient(points, pred_sdf):
    grad_outputs = torch.ones_like(pred_sdf, requires_grad=False, device=pred_sdf.device)
    gradient = torch.autograd.grad(
        outputs=pred_sdf,
        inputs=points,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return gradient


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mix_dataset:
        dataset = RecTrainDataset_Mixed(args.data_root, sample_size=args.sample_size, ratio=0.2)
        print("####### Use the dataset with mixed sample ########")
    else:
        dataset = RecTrainDataset(args.data_root, sample_size=args.sample_size)
        print("####### Use the dataset with only point cloud sample ########")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for obj_idx in range(len(dataloader.dataset)):
        model = RecNet(in_dim=3, fourier_transform=None, num_hidden_layers=10, hidden_dim=768, skip_in=[5])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iters, eta_min=1e-6)
        model.train()

        batch = dataloader.dataset[obj_idx]
        object_name = batch['object_name']

        print(f"Processing object {obj_idx + 1}/{len(dataloader.dataset)}, filename is {object_name}")

        save_path = os.path.join(args.save_path, object_name)
        tag = time.strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(save_path, tag)
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(log_dir=log_path)
        with open(os.path.join(log_path, 'train_log.txt'), 'w') as f:
            f.write(f"Training parameters:\n{args}\n")

        # each case, iterate several times
        with tqdm(range(args.num_iters), desc=f"Training {object_name}", unit="iter") as pbar:
            for iter_idx in pbar:
                batch = dataloader.dataset[obj_idx]
                points = batch['point'].to(device)
                grad = batch['grad'].to(device)
                sdf = batch['sdf'].to(device)

                points.requires_grad_(True)
                optimizer.zero_grad()

                pred_sdf = model(points)
                pred_grad = compute_gradient(points, pred_sdf)

                loss_sdf = torch.mean((pred_sdf - sdf) ** 2)
                loss_grad = torch.mean(torch.sum((pred_grad - grad) ** 2, dim=-1))
                
                # Generate random points for evaluating the eikonal term (use half of the points)
                if args.lambda_eikonal > 0:
                    empty_points = torch.rand_like(points[:len(points)//2]) - 0.5
                    empty_points.requires_grad_(True)
                    empty_sdf = model(empty_points)
                    empty_grad = compute_gradient(empty_points, empty_sdf)
                    eikonal_loss = torch.mean((torch.norm(empty_grad, dim=-1) - 1.0) ** 2)
                else:
                    eikonal_loss = 0.0

                loss = args.lambda_sdf * loss_sdf + args.lambda_gradient * loss_grad + args.lambda_eikonal * eikonal_loss
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                writer.add_scalar('Loss/train', loss.item(), iter_idx + 1)
                writer.add_scalar('Loss/sdf', loss_sdf.item(), iter_idx + 1)
                writer.add_scalar('Loss/grad', loss_grad.item(), iter_idx + 1)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_idx + 1)
                if args.lambda_eikonal > 0:
                    writer.add_scalar('Loss/eikonal', eikonal_loss.item(), iter_idx + 1)
                
                pbar.set_postfix(loss=f"{loss.item():.6f}")

                if (iter_idx + 1) % args.checkpoint_iters == 0:
                    checkpoint_path = os.path.join(log_path, f'model_epoch_{iter_idx + 1}.pth')
                    torch.save(model, checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}")

        # save the final model
        final_model_path = os.path.join(log_path, 'final_model.pth')
        torch.save(model, final_model_path)
        print(f"Final model saved at {final_model_path}")
        writer.close()

def train_w_fourier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mix_dataset:
        dataset = RecTrainDataset_Mixed(args.data_root, sample_size=args.sample_size)
        print("####### Use the dataset with mixed sample ########")
    else:
        dataset = RecTrainDataset(args.data_root, sample_size=args.sample_size)
        print("####### Use the dataset with only point cloud sample ########")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    transform = GaussianFourierFeatureTransform(
        num_input_channels=3,
        mapping_size=args.fourier_mapping_size,
        scale=args.fourier_scale
    ).to(device)

    for obj_idx in range(len(dataloader.dataset)):
        fourier_dim = transform.output_dim

        model = RecNet(in_dim=fourier_dim, fourier_transform=transform, num_hidden_layers=10, hidden_dim=768, skip_in=[5])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iters, eta_min=1e-6)
        model.train()

        batch = dataloader.dataset[obj_idx]
        object_name = batch['object_name']

        print(f"Processing object {obj_idx + 1}/{len(dataloader.dataset)}, filename is {object_name}")

        save_path = os.path.join(args.save_path, object_name)
        tag = time.strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(save_path, tag)
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(log_dir=log_path)
        with open(os.path.join(log_path, 'train_log.txt'), 'w') as f:
            f.write(f"Training parameters:\n{args}\n")

        # each case, iterate several times
        with tqdm(range(args.num_iters), desc=f"Training {object_name}", unit="iter") as pbar:
            for iter_idx in pbar:
                batch = dataloader.dataset[obj_idx]
                points = batch['point'].to(device)
                grad = batch['grad'].to(device)
                sdf = batch['sdf'].to(device)

                points.requires_grad_(True)
                optimizer.zero_grad()

                pred_sdf = model(points)
                pred_grad = compute_gradient(points, pred_sdf)

                loss_sdf = torch.mean((pred_sdf - sdf) ** 2)
                loss_grad = torch.mean(torch.sum((pred_grad - grad) ** 2, dim=-1))
                
                # Generate random points for evaluating the eikonal term (use half of the points)
                if args.lambda_eikonal > 0:
                    empty_points = torch.rand_like(points[:len(points)//2]) - 0.5
                    empty_points.requires_grad_(True)
                    empty_sdf = model(empty_points)
                    empty_grad = compute_gradient(empty_points, empty_sdf)
                    eikonal_loss = torch.mean((torch.norm(empty_grad, dim=-1) - 1.0) ** 2)
                else:
                    eikonal_loss = 0.0

                loss = args.lambda_sdf * loss_sdf + args.lambda_gradient * loss_grad + args.lambda_eikonal * eikonal_loss
                
                # loss = data_loss + args.lambda_eikonal * eikonal_loss

                loss.backward()
                optimizer.step()
                scheduler.step()
                
                writer.add_scalar('Loss/train', loss.item(), iter_idx + 1)
                writer.add_scalar('Loss/sdf', loss_sdf.item(), iter_idx + 1)
                writer.add_scalar('Loss/grad', loss_grad.item(), iter_idx + 1)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_idx + 1)
                if args.lambda_eikonal > 0:
                    writer.add_scalar('Loss/eikonal', eikonal_loss.item(), iter_idx + 1)
                
                pbar.set_postfix(loss=f"{loss.item():.6f}")

                if (iter_idx + 1) % args.checkpoint_iters == 0:
                    checkpoint_path = os.path.join(log_path, f'model_epoch_{iter_idx + 1}.pth')
                    torch.save(model, checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}")

        # save the final model
        final_model_path = os.path.join(log_path, 'final_model.pth')
        torch.save(model, final_model_path)
        print(f"Final model saved at {final_model_path}")
        writer.close()


if __name__ == "__main__":
    args = args_parser()
    os.makedirs(args.save_path, exist_ok=True)
    
    if args.use_fourier:
        train_w_fourier(args)
    else:
        train(args)





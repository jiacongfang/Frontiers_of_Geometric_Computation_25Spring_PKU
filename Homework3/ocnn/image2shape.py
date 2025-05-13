# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import numpy as np
import ocnn

from torchvision.transforms import ToPILImage
from datasets import get_image2shape_dataset
from autoencoder import AutoEncoderSolver

# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


class Image2ShapeSolver(AutoEncoderSolver):

  def get_model(self, flags):
    return ocnn.models.Image2Shape(flags.channel_out, flags.depth,
                                   flags.full_depth)

  def get_dataset(self, flags):
    return get_image2shape_dataset(flags)

  def model_forward(self, batch):
    octree = batch['octree'].cuda(non_blocking=True)
    image = batch['image'].cuda(non_blocking=True)
    model_out = self.model(image, octree, update_octree=False)
    output = self.compute_loss(octree, model_out)
    return output

  def eval_step(self, batch):
    # forward the model
    image = batch['image']
    output = self.model(image.cuda(non_blocking=True), update_octree=True)
    points_out = self.octree2pts(output['octree_out'])

    # save the output point clouds
    filenames = batch['filename']
    for i, filename in enumerate(filenames):
      pos = filename.rfind('.')
      if pos != -1: filename = filename[:pos]  # remove the suffix
      filename_img = os.path.join(self.logdir, filename + '.img.png')
      filename_out = os.path.join(self.logdir, filename + '.out.xyz')
      folder = os.path.dirname(filename_img)
      if not os.path.exists(folder): os.makedirs(folder)

      # NOTE: it consumes much time to save point clouds to hard disks
      image_pil = ToPILImage()(image[i])  # convert to PIL image
      image_pil.save(filename_img)
      np.savetxt(filename_out, points_out[i].cpu().numpy(), fmt='%.6f')


if __name__ == '__main__':
  Image2ShapeSolver.main()

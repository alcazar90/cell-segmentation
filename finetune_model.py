#
#
#

import argparse
import wandb
import logging
import os
import random
import math
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from datasets import load_dataset

from transformers import (
    SegformerForSemanticSegmentation, 
    SegformerImageProcessor,)

import torch
from torch import nn
from torch import optim

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


logger = logging.getLogger('example_logger')

# Argparser: command line arguments to call the training script
#------------------------------------------------------------------------------ 

def parse_args(input_args=None):
  parser = argparse.ArgumentParser(description="Training loop script for fine-tune a pretrained SegFormer model.")

  parser.add_argument(
    "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
  )

  parser.add_argument(
   "--validation_batch_size", type=int, default=4, help="Batch size (per device) used for model validation."
  )

  parser.add_argument("--init_learning_rate", type=float, default=3e-4)
  
  parser.add_argument(
   "--learning_rate_scheduler_gamma", type=float, default=0.9, help="ExponentialLR gamma parameter by which the learning rate decays every epoch. See more in the PyTorch documentation: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html ."
  )

  parser.add_argument("--num_train_epochs", type=int, default=1)
  parser.add_argument("--reproducibility_seed", type=int, default=42313988)
  parser.add_argument("--log_images_in_validation", type=bool, default=False)
  parser.add_argument("--dataloader_num_workers", type=int, default=mp.cpu_count(), help="Number of subprocesses to use for data loader. 0 means that the data will be loaded in the main process.")
  parser.add_argument("--model_name", type=str, default="huggingface-segformer-nvidia-mit-b0")
  parser.add_argument("--project_name", type=str, default="cell-segmentation", help="Name of the project in Weights & Biases.")


  if input_args is not None:
    args = parser.parse_args(input_args)
  else:
    args = parser.parse_args()
  return args


# Download the model from HuggingFace -> nvidia segformer
#------------------------------------------------------------------------------ 
def download_and_load_model_from_path(pretrained_model_name_or_path: str = "nvidia/mit-b0"):
  id2label = {0: "non-transformed", 1: "transformed"}
  label2id = {v:k for k,v in id2label.items()}
  model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name_or_path,
    id2label=id2label,
    label2id=label2id,
  )
  return model


# Dataset class & Transformations (i.e. model preprocessing, collate_fn)
#------------------------------------------------------------------------------ 

class CellSegmentation:

  def __init__(self, streaming=False, tfms:dir = None):
    self.repo_name = "alkzar90/cell_benchmark"
    self.train_ds = load_dataset(self.repo_name, split="train", streaming=streaming)
    self.valid_ds = load_dataset(self.repo_name, split="validation", streaming=streaming)
    self.test_ds = load_dataset(self.repo_name, split="test", streaming=streaming)
    if tfms:
      self.train_ds.set_transform(tfms["train"])
      self.valid_ds.set_transform(tfms["valid"])
      self.test_ds.set_transform(tfms["valid"])


  def get_dataloaders(self, bs=2, collate_fn=None, num_workers=2):
    train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=num_workers)
    valid_dl = DataLoader(self.valid_ds, batch_size=len(self.valid_ds), shuffle=True,
                        collate_fn=collate_fn,
                        num_workers=num_workers)
    test_dl = DataLoader(self.test_ds, batch_size=len(self.test_ds), shuffle=True,
                         collate_fn=collate_fn,
                         num_workers=num_workers)
    return train_dl, valid_dl, test_dl


feature_extractor = SegformerImageProcessor()
jitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

def train_transforms(example_batch):
  # se pueden agregar transformaciones para data augmentation 
  # adicionales en esta función como jitter()
  images = [jitter(x) for x in example_batch['image']]
  labels = [x for x in example_batch['masks']]
  inputs = feature_extractor(images, labels, return_tensors="pt")
  return inputs

def val_transforms(example_batch):
  images = [x for x in example_batch['image']]
  labels = [x for x in example_batch['masks']]
  inputs = feature_extractor(images, labels, return_tensors="pt")
  return inputs

def collate_fn(batch):
  """HuggingFace devuelve diccionarios, esta función devuelve tensor input, label
  agrupados por el número de ejemplos que tiene el batch dentro del mismo tensor.
  Es decir, (batch_size, input), (batch_size, target)"""
  x = [example["pixel_values"] for example in batch]
  y = [example["labels"] for example in batch]
  return torch.stack(x, 0, out=None), torch.stack(y, 0, out=None)



# Utility Functions: for manipulate the logits (upscale and flatten) and reproducibility
#------------------------------------------------------------------------------ 

def upscale_logits(logit_outputs, res=512):
  """Escala los logits a (4W)x(4H) para recobrar dimensiones originales del input"""
  return nn.functional.interpolate(
      logit_outputs,
      size=(res,res),
      mode='bilinear',
      align_corners=False
  )

def flatten_logits(logits):
  return logits.contiguous().view(logits.shape[0], -1)


def set_seed(seed: int = 42313988) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


# Model Validation
#------------------------------------------------------------------------------ 
def validate_model(model, valid_dl, loss_fn, log_images=False, num_classes=2, device="cpu"):
  """Compute performance of the model on the validation dataset and log a wandb.Table"""
  device=device
  model.eval()
  val_loss = 0.
  iou = 0.
  with torch.inference_mode():
    correct = 0

    iou_by_example = torch.zeros(len(valid_dl.dataset)) # tensor with length equals to the number of validation examples
    for i, (images, masks) in enumerate(valid_dl):
      images, masks = images.to(device), masks.to(device)
      masks=flatten_logits(torch.where(masks > 0.0, 1.0, 0.0))

      # Forward pass ➡
      logits = model(images)["logits"]
      probs = torch.softmax(upscale_logits(logits), dim=1)
      _, predicted = torch.max(probs.data, dim=1)
      probs = probs[:, 1, :, :]
      preds = flatten_logits(probs)
      val_loss+= loss_fn(preds, masks).item()*masks.size(0)

      # Compute pixel accuracy and accumulate
      correct += (flatten_logits(predicted) == masks).sum().item()

      # Compute IoU and accumulate
      mask2d=masks.view(masks.shape[0], predicted.shape[1], -1)
      intersection = torch.logical_and(mask2d, predicted)
      union = torch.logical_or(mask2d, predicted)
      iou += (torch.div(torch.sum(intersection, dim=(1,2)) + 1e-6, (torch.sum(union, dim=(1,2)) + 1e-6)).sum()/predicted.shape[0]).item()

      # tensor with IoU for every example (batch_size x 1)
      iou_by_example = intersection.sum(dim=(1,2), keepdim=False) / (union.sum(dim=(1,2), keepdim=False) + 1e-6)

      
      # Log validation predictions and images to the dashboard
      if log_images:
        if i == 0:
          # 🐝 Create a wandb Table to log images, labels and predictions to
          table = wandb.Table(columns=["image", "mask", "pred_mask", "probs", "iou"])
          for img, mask, pred, prob, iou_metric in zip(images.to("cpu"), masks.to("cpu"), predicted.to("cpu"), probs.to("cpu"), iou_by_example.to("cpu")):
            plt.imshow(prob.detach().cpu());
            plt.axis("off");
            plt.tight_layout();
            table.add_data(wandb.Image(img.permute(1,2,0).numpy()), 
                           wandb.Image(mask.view(img.shape[1:]).unsqueeze(2).numpy()),
                           wandb.Image(np.uint8(pred.unsqueeze(2).numpy())*255),
                           wandb.Image(plt),
                           iou_metric
                           )
    if log_images:
      wandb.log({"val_table/predictions_table":table}, commit=False)

  return (
        val_loss / len(valid_dl.dataset), 
        correct / (len(valid_dl.dataset)*512**2),
        iou / (i+1)
  )



# Training loop
#------------------------------------------------------------------------------ 

def main(args):

  # Set device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Make one log on every process with the configuration for debugging.
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
  )

  # If passed along, set the training seed now.
  if args.reproducibility_seed is not None:
    set_seed(args.reproducibility_seed)

  # Download the dataset and initialize the dataloaders
  tfms={"train": train_transforms, "valid": val_transforms}
  datasets=CellSegmentation(streaming=False, tfms=tfms)

  # TODO: parametrize args.valid_batch_size..., datasets.get_dataloader doesn't
  # use that parameter...
  train_dl, valid_dl, test_dl = datasets.get_dataloaders(bs=args.train_batch_size,
                                                         collate_fn=collate_fn,
                                                         num_workers=args.dataloader_num_workers)

  logger.info(f"Datasets and dataloader ready to train:\n -> train_ds: {len(datasets.train_ds)}\n -> valid_ds: {len(datasets.valid_ds)}") 

  # Download and load the model...
  model = download_and_load_model_from_path()
  model = model.to(device)


  logger.info("Model downloaded and loaded successfully!") 

  # Initialise a wandb run
  wandb.init(project=args.project_name,
             config={"epochs": args.num_train_epochs,
                     "batch_size": args.train_batch_size,
                     "lr": args.init_learning_rate,
                     "lr_scheduler_exponential__gamma": args.learning_rate_scheduler_gamma,
                     "seed": args.reproducibility_seed
             }
  )

  # Add additional configs to wandb if needed
  wandb.config["len_train"] = len(datasets.train_ds)
  wandb.config["len_val"] = len(datasets.valid_ds)

  logger.info("***** WANDB CONFIG *****") 
  logger.info(f"    Num exaples = {len(datasets.train_ds)}") 
  logger.info(f"    Num epochs = {args.num_train_epochs}") 
  logger.info(f"    Num batches each epoch = {len(train_dl)}") 
  logger.info(f"    Learning rate = {args.init_learning_rate}") 
  logger.info(f"    Log validation images = {args.log_images_in_validation}") 

  # Copy your config
  config = wandb.config
  
  # Get the data
  n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)
  LOG_IMAGES = args.log_images_in_validation

  # Make the loss, optimizer, and scheduler
  loss_fn = nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_scheduler_exponential__gamma)

  # Start training
  example_ct = 0
  step_ct = 0
  max_iou = 0.0
  epoch_max_iou = 0
  for epoch in tqdm(range(config.epochs)):
    model.train()
    for step, (images, masks) in tqdm(enumerate(train_dl), leave=False):
    
      images, masks = images.to(device), masks.to(device)
     
      # Transforms masks into 1D tensor with 1.0 and 0.0 (2 classes)
      masks = flatten_logits(torch.where(masks > 0.0, 1.0, 0.0))
     
      # Perform a forward pass
      logits = model(images)["logits"]

      # Upscale the logit tensor and get the probabilities with the softmax
      probs = torch.softmax(upscale_logits(logits), dim=1)
    
      # Keep it just the probability for the class transform (dim=1) -> binary class
      probs = probs[:, 1, :, :]
   
      # Transforms the probs tensor into one with shape (batch_size, 512 x 512)
      preds = flatten_logits(probs)
    
      # Compute the loss
      train_loss = loss_fn(preds, masks)
    
      # Clean the gradients
      optimizer.zero_grad()
  
      # Backward prop
      train_loss.backward()

      # Update the parameters
      optimizer.step()
  
      example_ct  += len(images)
      metrics = {"train/train_loss": train_loss,
                 "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                 "train/example_ct": example_ct,
                 "train/cur_learning_rate": scheduler.state_dict()["_last_lr"][0]}


      if step + 1 < n_steps_per_epoch:
        # Log train metrics to wandb
        wandb.log(metrics)

      step_ct += 1

    # Update the learning rate given the scheduler
    scheduler.step()

    # Log validation images and predictions on last epoch
    if LOG_IMAGES:
      log_images = epoch == (config.epochs-1)
    else:
      log_images = False

    # Do validation and maybe log images to Tables
    val_loss, accuracy, mIoU = validate_model(model, valid_dl, loss_fn, log_images=log_images, device=device)
    if mIoU > max_iou:
      # Save the current state of the model to disk and to W&B Artifacts
      model_fn = f"{args.model_name}.pt"
      torch.save(model, model_fn)
      max_iou = mIoU
      epoch_max_iou = epoch+1
      

    # Log train and validation metrics to wandb
    val_metrics = {"val/val_loss": val_loss,
                   "val/val_accuracy": accuracy,
                   "val/mIoU": mIoU}
    wandb.log({**metrics, **val_metrics})

    logger.info(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:.3f}, Valid Accuracy: {accuracy:.2f}, mIoU: {mIoU:.3f}")

  # Upload the best model as a wandb artifact
  wandb.log_artifact(artifact_or_path=model_fn, 
                     name=args.model_name+f"_epoch-{epoch_max_iou}", 
                     type="model")

  # Close your wandb run
  wandb.finish()
	


if __name__ == "__main__":
  args = parse_args()
  logger.info(args)
  main(args)


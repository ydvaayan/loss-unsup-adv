import argparse
import os
import wandb
from PIL import Image
import torch
from torch.optim import AdamW
from tqdm import tqdm
import copy
from datasets import load_dataset
from torchvision.datasets import CIFAR10
import random
import string
import io
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torch
from apgd_attack import APGDAttack
from utils import AverageMeter, cosine_lr
from torch.utils.data import DataLoader, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from resnet import ResNetSimCLR
from datautils import CenterCropAndResize
torch.manual_seed(0)
np.random.seed(0)
 
# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Perform neural collision attack.')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=256,
                    help='batch size for processing images')
parser.add_argument('--n_iter', dest='n_iter', type=int, default=10,
                    help='average number of iterations')
# parser.add_argument('--n_iter_range', dest='n_iter_range', type=int, default=0,
#                     help='maximum number of iterations')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=8/255,
                    help='maximum perturbation (L∞ norm bound)')
# parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=1,
                    # help='number of epochs')
parser.add_argument('--lr', dest='lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--warmup', dest='warmup', type=int, default=1_400,
                    help='number of warmup steps')
parser.add_argument('--steps', dest='steps', type=int, default=20_000,
                    help='number of steps')
parser.add_argument('--start_step', dest='start_step', type=int, default=0,
                    help='starting step')
parser.add_argument('--clean_weight', dest='clean_weight', type=float, default=0,
                    help='weight of clean loss')
parser.add_argument('--val_freq', dest='val_freq', type=int, default=1000,
                    help='validation frequency')
parser.add_argument('--resume_path', dest='resume_path', type=str, default=None,
                    help='resume path')
parser.add_argument('--unsupervised', action='store_true',
                    help='use unsupervised training')
parser.add_argument('--dataset', dest='dataset', type=str, default='cifar10',
                    help='dataset to use')

args = parser.parse_args()
# os.makedirs('./adversarial_dataset', exist_ok=True)
device = 'cuda' if torch.cuda.is_available else 'cpu'

print(device)
randomstr = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key is None:
    raise ValueError("WANDB_API_KEY environment variable is not set")

wandb.login(key=wandb_api_key)
wandb.init(
    project="improving-loss-adv-LBP",
    name=f"attack-data{args.dataset}-lr{args.lr}-cw{args.clean_weight}-batch_size{args.batch_size}-warmup{args.warmup}-eps{args.epsilon:.3f}-{randomstr}",
    config=vars(args)
)
    
# preprocess = transforms.Compose([
#                 # transforms.RandomResizedCrop(
#                 #     32,
#                 #     scale=(self.hparams.scale_lower, 1.0),
#                 #     interpolation=PIL.Image.BICUBIC,
#                 # ),
#                 # transforms.RandomHorizontalFlip(),
#                 # datautils.get_color_distortion(s=self.hparams.color_dist_s),
#                 transforms.ToTensor(),
#             ])

criterion_loss = nn.MSELoss()

# clean_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)


if args.dataset == 'cifar10':
    preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])
    model_path = 'pretrained_models/resnet50_cifar10_bs1024_epochs1000.pth.tar'
    clean_model = ResNetSimCLR(model_name='resnet50', out_dim=10, cifar_head=True)

elif args.dataset == 'imagenet':
    preprocess = transforms.Compose([
                CenterCropAndResize(proportion=0.875, size=224),
                transforms.ToTensor(),
            ])
    model_path = 'pretrained_models/resnet50_imagenet_bs2k_epochs200.pth.tar'
    clean_model = ResNetSimCLR(model_name='resnet50', out_dim=1000, cifar_head=False)

else:
    raise ValueError(f"Unknown dataset: {args.dataset}")
torch.serialization.add_safe_globals({'Namespace': argparse.Namespace})

linear_model_path = model_path.replace('.pth.tar', '_linear.pth.tar')

ckp_bb = torch.load(
    model_path,
    map_location='cpu',
    weights_only=False
)

ckp_bb_statedict = {k[8:]:v for k,v in ckp_bb['state_dict'].items() if k.startswith('convnet')}

ckp_lin = torch.load(
    linear_model_path,
    map_location='cpu',
    weights_only=False
)

clean_model.backbone.load_state_dict(ckp_bb_statedict)

clean_model.linear.load_state_dict(ckp_lin['state_dict'])
# clean_model.fc = nn.Identity()
clean_model.to(device)
for param in clean_model.parameters():
    param.requires_grad = False
clean_model.eval()

if args.dataset == 'cifar10':
    full_trainset = CIFAR10(root='./data', train=True, download=True, transform=preprocess)
    # testset = CIFAR10(root = './data', train=False, download=True, transform=preprocess)

    train_size = int(0.9*len(full_trainset))
    val_size =  len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

elif args.dataset == 'imagenet':
    trainset = load_dataset("timm/imagenet-1k-wds", split="train")
    valset = load_dataset("timm/imagenet-1k-wds", split="validation")
    # def is_ok_batch(batch):
    #     results = []
    #     for jpg_bytes in batch['jpg']:
    #         try:
    #             image = Image.open(io.BytesIO(jpg_bytes))
    #             # Check for EXIF issues by attempting to get EXIF data
    #             image.getexif()  # If EXIF parsing fails, it'll raise an error
    #             image.verify()  # Verifies that the image is valid and not corrupted
    #             results.append(True)
    #         except UnicodeDecodeError:
    #             # Catch malformed EXIF (UTF-8 issue) specifically
    #             results.append(False)
    #         except Exception:
    #             # Catch any other issues (corrupted image, other decoding issues)
    #             results.append(False)
        
    #     return {'is_valid': results}  # Return a dictionary for batched processing

    # # Apply filtering in batches with parallelization
    # trainset = trainset.filter(is_ok_batch, batched=True, num_proc=16)
    # valset = valset.filter(is_ok_batch, batched=True, num_proc=16)
    
    # def mapper_function(example):        
    #     return {
    #         "jpg": torch.stack([preprocess(jpg.convert('RGB')) for jpg in example["jpg"]]),
    #         "cls": torch.Tensor(example["cls"]).long()
    #     }
    def mapper_function(example):
        processed_images = []
        valid_labels = []
        error_count = 0
        MAX_LOGGED_ERRORS = 5  # prevent excessive logging

        for jpg, label in zip(example["jpg"], example["cls"]):
            try:
                img = jpg.convert("RGB")
            except Exception as e:
                if error_count < MAX_LOGGED_ERRORS:
                    print(f"[Conversion Error] Skipping image: {e}")
                    error_count += 1
                continue

            try:
                _ = img.getexif()  # Catch corrupted EXIF metadata
            except Exception as e:
                if error_count < MAX_LOGGED_ERRORS:
                    print(f"[EXIF Error] Skipping EXIF: {e}")
                    error_count += 1
                # Not critical, so continue to preprocessing

            try:
                img_tensor = preprocess(img)
            except Exception as e:
                if error_count < MAX_LOGGED_ERRORS:
                    print(f"[Preprocessing Error] Skipping image: {e}")
                    error_count += 1
                continue

            processed_images.append(img_tensor)
            valid_labels.append(label)

        if len(processed_images) == 0:
            return {
                "jpg": torch.zeros((1, 3, 224, 224)),  # dummy fallback
                "cls": torch.tensor([-1])              # dummy label
            }

        return {
            "jpg": torch.stack(processed_images),
            "cls": torch.tensor(valid_labels).long()
        }


    trainset.set_transform(mapper_function)
    valset.set_transform(mapper_function)

    # trainset.set_format(type="torch", columns=["jpg", "cls"])
    # valset.set_format(type="torch", columns=["jpg", "cls"])

    

valset, _ = random_split(valset, [10_000, len(valset)-10_000])

trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=8, drop_last=True, pin_memory=True, shuffle=True)
valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=8, drop_last=False, pin_memory=True, shuffle=False)
# testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

model = copy.deepcopy(clean_model)
apgd = APGDAttack(predict=model, eps=args.epsilon, n_iter=args.n_iter)
# if args.resume_path is not None:
#     load_model(args.resume_path)

optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
scheduler = cosine_lr(optimizer, args.lr, args.warmup, args.steps)
step_total = args.start_step

for param in model.parameters():
    param.requires_grad = True
ckp_dir = f'checkpoints/{randomstr}'
os.makedirs(ckp_dir, exist_ok=True)

while step_total < args.steps:
    pbar = tqdm(trainloader)
    loss_meter = AverageMeter('Loss')
    accuracy_meter = AverageMeter('Accuracy')

    # for item in pbar:
    data_iter = iter(pbar)

    while True:
        try:
            item = next(data_iter)
        except StopIteration:
            break
        except Exception as e:
            print(f"Skipping batch due to DataLoader error: {e}")
            continue
        if type(item) is tuple:
            images, labels = item
        else:
            images = item["jpg"]
            labels = item["cls"]
        if labels[0] == -1:
            continue
        scheduler(step_total)
        images = images.to(device)
        labels = labels.to(device)
        clean_model.train()
        with torch.no_grad():
            target_features = clean_model.backbone(images)

        # for debugging   
        # if step_total <670:
        #     step_total +=1
        #     continue
        model.eval()
        adv_images, acc = apgd.attack_single_run(images, labels)
        # adv_images = images.cuda()
        model.train()
        
        adv_logits = model.backbone(adv_images)
        adv_loss = criterion_loss(adv_logits, target_features)

        adv_loss = adv_loss.mean()
        adv_loss.backward()

        clean_loss = 0
        # if args.clean_weight > 0:
        
        clean_logits = model.backbone(images)
        clean_loss = criterion_loss(clean_logits, target_features)
        clean_loss = clean_loss.mean()
        weighted_clean_loss = args.clean_weight * clean_loss
        weighted_clean_loss.backward()

        loss = adv_loss + weighted_clean_loss
        adv_loss = adv_loss.item()

        clean_loss = clean_loss.item()
        
        optimizer.step()
        optimizer.zero_grad()

        acc = acc.float().mean().item()
        loss_meter.update(loss.item(), len(images))
        accuracy_meter.update(acc, len(images))

        pbar.set_description(f"attack: {acc * 100:.4f}, loss: {loss:.4f}, adv_loss: {adv_loss:.4f}, clean_loss: {clean_loss:.4f}")
        wandb.log({
            "step": step_total,
            "train/attack_accuracy": acc,
            "train/loss": loss,
            "train/adv_loss": adv_loss,
            "train/clean_loss": clean_loss,
            "lr": optimizer.param_groups[0]["lr"]
        }, step=step_total)
        step_total += 1

        del images, adv_images

        if step_total % args.val_freq == 0:
            model.eval()

            clean_features = []
            adv_features = []
            target_labels = []

            adv_accuracy = 0
            clean_accuracy = 0
            samples_count = 0
            # for item in tqdm(valloader):
            valdata_iter = iter(tqdm(valloader))

            while True:
                try:
                    item = next(valdata_iter)
                except StopIteration:
                    print("Encountered StopIteration")
                    break
                except Exception as e:
                    print(f"Skipping batch due to DataLoader error: {e}")
                    continue
            
                if type(item) is tuple:
                    images, labels = item
                else:
                    images = item["jpg"]
                    labels = item["cls"]
                if labels[0] == -1:
                    continue
                images = images.to(device)
                labels = labels.to(device)
                adv_images, adv_acc = apgd.attack_single_run(images, labels)

                with torch.no_grad():
                    pred = model(images).argmax(dim=1)
                    adv_accuracy += adv_acc.float().sum().cpu().item()
                    clean_accuracy += (labels==pred).float().sum().cpu().item()
                    samples_count += len(images)

            clean_accuracy /= samples_count
            adv_accuracy /= samples_count

            
            print(f"validation clean accuracy: {clean_accuracy * 100:.2f}%, adv accuracy: {adv_accuracy * 100:.2f}%")
            wandb.log({
                "val/clean_accuracy": clean_accuracy,
                "val/adv_accuracy": adv_accuracy
            }, step=step_total)

            os.makedirs(f'checkpoints/{randomstr}', exist_ok=True)
            checkpoint_path = f'checkpoints/{randomstr}/model_{args.lr}_{args.clean_weight}_{step_total}_{args.n_iter}.pth'
            torch.save(model.backbone.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at step {step_total} -> {checkpoint_path}")


    # print(f"step: {step_total}, loss: {loss_meter.avg.item():.4f}, accuracy: {accuracy_meter.avg.item():.4f}")
    torch.save(model.backbone.state_dict(), f'{ckp_dir}/model_{args.lr}_{args.clean_weight}_{step_total}_{args.n_iter}.pth')

    del loss_meter, accuracy_meter
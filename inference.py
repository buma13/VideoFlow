import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
from utils import flow_viz

from core.Networks import build_network

from utils.utils import InputPadder

def prepare_resizing(cfg, resize_factor = 1):
    image_list = sorted(os.listdir(cfg.seq_dir))
    img = Image.open(os.path.join(cfg.seq_dir, image_list[0]))
    
    resolution_original = {
        'height' : img.height,
        'width' : img.width
        }
    
    cfg.image_size[0] = img.height // resize_factor
    cfg.image_size[1] = img.width // resize_factor

    return resolution_original

def prepare_image(cfg):
    print(f"preparing image...")
    print(f"Input image sequence dir = {cfg.seq_dir}")

    image_list = sorted(os.listdir(cfg.seq_dir))
    images = []
    for fn in image_list:
        img = Image.open(os.path.join(cfg.seq_dir, fn))
        img = img.resize((cfg.image_size[1], cfg.image_size[0]), resample=Image.Resampling.BOX)
        img = np.array(img).astype(np.uint8)[..., :3]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        images.append(img)
        
    images.insert(0, images[0])
    images.insert(0, images[0])
    images.insert(0, images[0])
    
    return torch.stack(images)

def vis_pre(flow_pre, vis_dir, original_resolution):

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    N = flow_pre.shape[0]

    # Forward Flow
    for idx in range(N//2):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image = image.resize((original_resolution['width'], original_resolution['height']), resample=Image.Resampling.BOX)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx+2, idx+3))
    
    # Backward Flow
    # for idx in range(N//2, N):
    #     flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
    #     image = Image.fromarray(flow_img)
    #     image = image.resize((original_resolution['width'], original_resolution['height']), resample=Image.Resampling.BOX)
    #     image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx-N//2+2, idx-N//2+1))

@torch.no_grad()
def inference(model, cfg):

    model.eval()

    input_images = prepare_image(cfg)
    input_images = input_images[None].cuda()
    padder = InputPadder(input_images.shape)
    input_images = padder.pad(input_images)
    flow_pre, _ = model(input_images, {})
    flow_pre = padder.unpad(flow_pre[0]).cpu()

    return flow_pre

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='MOF')
    parser.add_argument('--seq_dir', default='default')
    parser.add_argument('--vis_dir', default='default')
    parser.add_argument('-r', '--resize_factor', type=int, default=1, help='Image resolution down sampling factor')
    
    args = parser.parse_args()

    if args.mode == 'MOF':
        from configs.multiframes_sintel_submission import get_cfg
    elif args.mode == 'BOF':
        from configs.sintel_submission import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    original_resolution = prepare_resizing(cfg, args.resize_factor)

    model = torch.nn.DataParallel(build_network(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))
    
    
    flow_pre = inference(model.module, cfg)
            
    vis_pre(flow_pre, cfg.vis_dir, original_resolution)

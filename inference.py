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

def prepare_image(cfg, resize_factor = 1):
    print(f"preparing image...")
    print(f"Input image sequence dir = {cfg.seq_dir}")

    image_list = sorted(os.listdir(cfg.seq_dir))
    images = []
    run_once = False
    for fn in image_list:
        img = Image.open(os.path.join(cfg.seq_dir, fn))
        if not run_once:
            height = img.height // resize_factor
            width = img.width // resize_factor
            assert cfg.image_size[0] == height and cfg.image_size[1] == width,\
            'Image dimensions ({},{}) not matching config ({},{})'.format(height, width, cfg.image_size[0], cfg.image_size[1])
            run_once = True 
        img = img.resize((width, height), resample=Image.Resampling.BOX)
        img = np.array(img).astype(np.uint8)[..., :3]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        images.append(img)
    
    return torch.stack(images)

def vis_pre(flow_pre, vis_dir):

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    N = flow_pre.shape[0]

    # Forward Flow
    for idx in range(N//2):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx+2, idx+3))
    
    # Backward Flow
    for idx in range(N//2, N):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx-N//2+2, idx-N//2+1))

@torch.no_grad()
def inference(model, cfg, resize_factor = 1):

    model.eval()

    input_images = prepare_image(cfg, resize_factor)
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

    model = torch.nn.DataParallel(build_network(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))
    
    flow_pre = inference(model.module, cfg, args.resize_factor)
            
    vis_pre(flow_pre, cfg.vis_dir)

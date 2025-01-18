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

import cv2

def apply_alpha_mask(rgb_image, alpha_mask):
    """
    Applies a masking to an RGB image using a provided alpha mask
    :param rgb_image: cv2 rgb image
    :param alpha_mask: cv2 grayscale image
    :return: cv2 image with alpha mask applied
    """
    # Normalize to range 0-1
    normalized_alpha_mask = alpha_mask / 255.0

    background_color = np.array([255, 255, 255])
    for i in range(3):
        rgb_image[:, :, i] = rgb_image[:, :, i] * normalized_alpha_mask + background_color[i] * (
                    1 - normalized_alpha_mask)

    return rgb_image

def prepare_resizing(cfg, resize_factor = 1, camera_path = ""):
    image_list = sorted(os.listdir(camera_path))
    img = Image.open(os.path.join(camera_path, image_list[0]))
    
    resolution_original = {
        'height' : img.height,
        'width' : img.width
        }
    
    cfg.image_size[0] = img.height // resize_factor
    cfg.image_size[1] = img.width // resize_factor

    return resolution_original

def prepare_image(cfg, camera_path):
    print(f"preparing image...")
    print(f"Input image sequence dir = {camera_path}")

    image_list = sorted(os.listdir(camera_path))
    images = []
    image_names = []
    for fn in image_list:
        img = Image.open(os.path.join(camera_path, fn))
        img = img.resize((cfg.image_size[1], cfg.image_size[0]), resample=Image.Resampling.BOX)
        img = np.array(img).astype(np.uint8)[..., :3]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        images.append(img)
        image_names.append(fn)
        
    images.insert(0, images[0])
    images.append(images[-1])
    
    return torch.stack(images), image_names

def vis_pre(flow_pre, vis_dir, original_resolution, offset = 0):

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    N = flow_pre.shape[0]

    print("processing frames {} (inclusive) to {} (inclusive)".format(offset, offset + N // 2 - 1))

    # Forward Flow
    for idx in range(N // 2):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image = image.resize((original_resolution['width'], original_resolution['height']), resample=Image.Resampling.BOX)
        image.save('{}/{:04}.png'.format(vis_dir, idx + offset))
        
        # image = np.array(image).astype(np.uint8)[..., :3]
        # alpha_mask = Image.open(os.path.join(alpha_masks_path, image_names[idx]))
        # alpha_mask = np.array(alpha_mask).astype(np.uint8)[..., :3]
        # image = apply_alpha_mask(image, alpha_mask)
        # cv2.imwrite(os.path.join(vis_dir, f"{str(idx + offset).zfill(4)}.png"), image)
    
    # Backward Flow
    # for idx in range(N//2, N):
    #     flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
    #     image = Image.fromarray(flow_img)
    #     image = image.resize((original_resolution['width'], original_resolution['height']), resample=Image.Resampling.BOX)
    #     image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx-N//2+2, idx-N//2+1))

@torch.no_grad()
def inference(model, cfg, input_images):

    model.eval()

    input_images = input_images[None].cuda()
    padder = InputPadder(input_images.shape)
    input_images = padder.pad(input_images)
    flow_pre, _ = model(input_images, {})
    flow_pre = padder.unpad(flow_pre[0]).cpu()

    return flow_pre

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def apply_alpha_masks(output_path, alpha_masks_path):
    image_names = os.listdir(alpha_masks_path)
    for image_name in image_names:
        image = cv2.imread(os.path.join(output_path, image_name))
        alpha_mask = cv2.imread(os.path.join(alpha_masks_path, image_name), cv2.IMREAD_GRAYSCALE)
        image = apply_alpha_mask(image, alpha_mask)
        cv2.imwrite(os.path.join(output_path, image_name), image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='MOF')
    parser.add_argument('--scene_dir', default='default')
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

    cameras_path = os.path.join(args.scene_dir, "images")
    camera_folders = os.listdir(cameras_path)
    for cam_folder in camera_folders:
        alpha_masks_path = os.path.join(args.scene_dir, "alpha_masks", cam_folder)
        camera_path = os.path.join(cameras_path, cam_folder)
        output_path = os.path.join(args.scene_dir, "optical_flow", cam_folder)

        original_resolution = prepare_resizing(cfg, args.resize_factor, camera_path)
        images, image_names = prepare_image(cfg, camera_path)

        window_size = 9
        step_size = window_size-2
        
        prune_last = False
        for i in range(0, len(images), step_size):
            if len(images) - i == 1:
                break
            elif len(images) - i == 2:
                print("adding duplicate of last frame to prevent error")
                images = torch.cat((images, images[-1].unsqueeze(0)))
                prune_last = True

            flow_pre = inference(model.module, cfg, images[i:i + window_size])
            vis_pre(flow_pre, output_path, original_resolution, i)
            
        if prune_last:
            image_list = sorted(os.listdir(output_path))
            os.remove(os.path.join(output_path, image_list[-1]))
        
        apply_alpha_masks(output_path, alpha_masks_path)
        
        # Create video
        length_original_images = len([img for img in os.listdir(camera_path) if img.endswith(".jpg") or img.endswith(".png")])
        images = [img for img in os.listdir(output_path) if img.endswith(".jpg") or img.endswith(".png")][:length_original_images]
        images.sort()
        frame = cv2.imread(os.path.join(output_path, images[0]))
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_name = '{}_optical_flow.mp4'.format(cam_folder)
        video = cv2.VideoWriter(os.path.join(args.scene_dir, video_name), fourcc, 30, (width, height))
        for image in images:
            img = cv2.imread(os.path.join(output_path, image))
            video.write(img)
        video.release()

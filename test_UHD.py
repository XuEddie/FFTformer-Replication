import os
import torch
import argparse
from basicsr.models.archs.fftformer_arch import fftformer
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import crop
from skimage import img_as_ubyte
import cv2
from natsort import natsorted
from glob import glob


def image_to_patches(image, psize, stride):
    psize_h, psize_w = psize if isinstance(psize, tuple) else (psize, psize)
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)

    h, w = image.shape[-2:]
    h_list = [i for i in range(0, h - psize_h + 1, stride_h)]
    w_list = [i for i in range(0, w - psize_w + 1, stride_w)]
    corners = [(hi, wi) for hi in h_list for wi in w_list]

    patches = torch.stack([
        crop(image, hi, wi, psize_h, psize_w)
        for (hi, wi) in corners
    ])
    return patches, corners

def patches_to_image(patches, corners, psize, shape):
    psize_h, psize_w = psize if isinstance(psize, tuple) else (psize, psize)
    images = torch.zeros(shape).cuda()
    counts = torch.zeros(shape).cuda()
    for (hi, wi), patch in zip(corners, patches):
        images[:, hi:hi + psize_h, wi:wi + psize_w] += patch
        counts[:, hi:hi + psize_h, wi:wi + psize_w] += 1
    images /= counts
    return images

def save_img(filepath, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img)

def main(args):
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    model = fftformer()
    if torch.cuda.is_available():
        model.cuda()
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['params'], strict=True)
    model.eval()
    print('restoring images......')
    files = natsorted(glob(os.path.join(args.data_dir, 'input', '*.jpg')) +
                      glob(os.path.join(args.data_dir, 'input', '*.JPG')) +
                      glob(os.path.join(args.data_dir, 'input', '*.png')) +
                      glob(os.path.join(args.data_dir, 'input', '*.PNG')))
    if len(files) == 0:
        raise Exception(f"No files found at {args.data_dir}")
    with torch.no_grad():
        for index, file_ in enumerate(files):
            print(f'index: {index}')
            img = Image.open(file_).convert('RGB')
            input_img = F.to_tensor(img).unsqueeze(0).cuda()
            print(f'Testing set shape: {input_img.shape}')
            b, c, h, w = input_img.shape
            
            mul = 16
            H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
            padh = H - h if h % mul != 0 else 0
            padw = W - w if w % mul != 0 else 0
            input_img = torch.nn.functional.pad(input_img, (0, padw, 0, padh), 'reflect')
            patch_size = 704, 1280
            stride = 704 // 2, 1280 // 2
            patches, corners = image_to_patches(input_img[0], patch_size, stride)
            restored_patches = []
            for batch_patch in patches.split(1):
                print(batch_patch.shape)
                batch_patch = model(batch_patch)
                restored_patches.extend(batch_patch)
            shape = (3, H, W)
            restored = patches_to_image(restored_patches, corners, patch_size, shape)
            restored = restored.unsqueeze(0)
            restored = torch.clamp(restored, 0, 1)
            restored = restored[:, :, :h, :w]
            print(f'restored shape: {restored.shape}')
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])
            f = os.path.splitext(os.path.split(file_)[-1])[0]
            save_img(os.path.join(args.result_dir, f + '.jpg'), restored)
            print('%d/%d' % (index + 1, len(files)))
    print(f"Files saved at {args.result_dir}")
    print('finish !')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='fftformer_UHD-Blur', type=str)
    parser.add_argument('--data_dir', type=str, default='/home/test/Workspace/zc/dataset_IR/UHD/UHD-deblur/test')
    parser.add_argument('--test_model', type=str, default='./pretrain_model/FFTformer_UHD-Blur.pth')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    args = parser.parse_args()
    args.result_dir = os.path.join('results/', args.model_name)
    print(args)
    main(args)
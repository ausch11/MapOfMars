# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import RSImage, RSInferencer
'''
https://github.com/open-mmlab/mmsegmentation/pull/3131
python demo/rs_image_inference.py 
demo/demo.png 
projects/pp_mobileseg/configs/pp_mobileseg/pp_mobileseg_mobilenetv3_2x16_80k_ade20k_512x512_tiny.py pp_mobileseg_mobilenetv3_2xb16_3rdparty-tiny_512x512-ade20k-a351ebf5.pth --batch-size 8 --device cpu --thread 1
'''

def main():
    parser = ArgumentParser()
    parser.add_argument('--image',
                        default='',
                        help='Image file path')
    parser.add_argument('--config',
                        default='',
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default='',
                        help='Checkpoint file')
    parser.add_argument(
        '--output-path',
        help='Path to save result image',
        default='result.png')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='maximum number of windows inferred simultaneously')
    parser.add_argument(
        '--window-size',
        help='window xsize,ysize',
        default=(224, 224),
        type=int,
        nargs=2)
    parser.add_argument(
        '--stride',
        help='window xstride,ystride',
        default=(224, 224),
        type=int,
        nargs=2)
    parser.add_argument(
        '--thread', default=1, type=int, help='number of inference threads')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    inferencer = RSInferencer.from_config_path(
        args.config,
        args.checkpoint,
        batch_size=args.batch_size,
        thread=args.thread,
        device=args.device)
    image = RSImage(args.image)

    inferencer.run(image, args.window_size, args.stride, args.output_path)


if __name__ == '__main__':
    main()

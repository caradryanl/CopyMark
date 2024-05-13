# python scripts/run_dino_drc.py --image_path datasets/laion-aesthetic-2-5k/images/ --output_dir datasets/laion-aesthetic-2-5k/masks/
# python scripts/run_dino_drc.py --image_path datasets/coco2017-val-2-5k/images/ --output_dir datasets/coco2017-val-2-5k/masks/

python scripts/run_dino_drc.py --mask_size 256 256 --image_path datasets/ffhq-2-5k/images/ --output_dir datasets/ffhq-2-5k/masks/
python scripts/run_dino_drc.py --mask_size 256 256 --image_path datasets/celeba-hq-2-5k/images/ --output_dir datasets/celeba-hq-2-5k/masks/
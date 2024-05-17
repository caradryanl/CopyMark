python scripts/train_gsa.py --model-type sd --gsa-mode 1 --ckpt-path ../models/diffusers/stable-diffusion-v1-5/ --member-dataset laion-aesthetic-2-5k --holdout-dataset coco2017-val-2-5k --batch-size 3
python scripts/train_gsa.py --model-type sd --gsa-mode 2 --ckpt-path ../models/diffusers/stable-diffusion-v1-5/  --member-dataset laion-aesthetic-2-5k --holdout-dataset coco2017-val-2-5k --batch-size 3
python scripts/train_gsa.py --model-type ldm --gsa-mode 1 --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k --holdout-dataset ffhq-2-5k --batch-size 3
python scripts/train_gsa.py --model-type ldm --gsa-mode 2 --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k --holdout-dataset ffhq-2-5k --batch-size 3
python scripts/train_gsa.py --model-type sdxl --gsa-mode 1 --ckpt-path ../models/diffusers/Kohaku-XL-Epsilon/ --member-dataset hakubooru-2-5k-member --holdout-dataset hakubooru-2-5k-nonmember --batch-size 1
python scripts/train_gsa.py --model-type sdxl --gsa-mode 2 --ckpt-path ../models/diffusers/Kohaku-XL-Epsilon/ --member-dataset hakubooru-2-5k-member --holdout-dataset hakubooru-2-5k-nonmember --batch-size 1 


python scripts/train_secmi.py --model-type sdxl --ckpt-path ../models/diffusers/Kohaku-XL-Epsilon/ --member-dataset hakubooru-2-5k-eval-member --holdout-dataset hakubooru-2-5k-eval-nonmember --batch-size 16 --eval True
python scripts/train_pia.py --model-type sdxl --ckpt-path ../models/diffusers/Kohaku-XL-Epsilon/  --member-dataset hakubooru-2-5k-eval-member --holdout-dataset hakubooru-2-5k-eval-nonmember --batch-size 16 --eval True
python scripts/train_pfami.py --model-type sdxl --ckpt-path ../models/diffusers/Kohaku-XL-Epsilon/ --member-dataset hakubooru-2-5k-eval-member --holdout-dataset hakubooru-2-5k-eval-nonmember --batch-size 16 --eval True
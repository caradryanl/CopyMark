python scripts/train_gsa.py --model-type kohaku --gsa-mode 1 --ckpt-path ../models/diffusers/Kohaku-XL-Epsilon/ --member-dataset hakubooru-2-5k-eval-member --holdout-dataset hakubooru-2-5k-eval-nonmember --batch-size 1 --eval True
python scripts/train_gsa.py --model-type kohaku --gsa-mode 2 --ckpt-path ../models/diffusers/Kohaku-XL-Epsilon/ --member-dataset hakubooru-2-5k-eval-member --holdout-dataset hakubooru-2-5k-eval-nonmember --batch-size 1 --eval True
python scripts/train_secmi.py --model-type kohaku --ckpt-path ../models/diffusers/Kohaku-XL-Epsilon/ --member-dataset hakubooru-2-5k-eval-member --holdout-dataset hakubooru-2-5k-eval-nonmember --batch-size 16 --eval True
python scripts/train_pia.py --model-type kohaku --ckpt-path ../models/diffusers/Kohaku-XL-Epsilon/  --member-dataset hakubooru-2-5k-eval-member --holdout-dataset hakubooru-2-5k-eval-nonmember --batch-size 16 --eval True
python scripts/train_pfami.py --model-type kohaku --ckpt-path ../models/diffusers/Kohaku-XL-Epsilon/ --member-dataset hakubooru-2-5k-eval-member --holdout-dataset hakubooru-2-5k-eval-nonmember --batch-size 16 --eval True
# python scripts/train_gsa.py --model-type sdxl --gsa-mode 1 --ckpt-path ../models/diffusers/Kohaku-XL-Epsilon/ --member-dataset hakubooru-2-5k-member --holdout-dataset hakubooru-2-5k-nonmember --batch-size 1 --demo True

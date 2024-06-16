python scripts/train_secmi.py --model-type ldm --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k-eval --holdout-dataset ffhq-2-5k-eval --batch-size 64
python scripts/train_pia.py --model-type ldm --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k-eval --holdout-dataset ffhq-2-5k-eval  --batch-size 64
python scripts/train_pfami.py --model-type ldm --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k-eval --holdout-dataset ffhq-2-5k-eval --batch-size 64
python scripts/train_gsa.py --model-type ldm --gsa-mode 1 --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k --holdout-dataset ffhq-2-5k --batch-size 4
python scripts/train_gsa.py --model-type ldm --gsa-mode 2 --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k --holdout-dataset ffhq-2-5k --batch-size 4



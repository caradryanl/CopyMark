# python scripts/train_secmi.py --model-type ldm --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k --holdout-dataset ffhq-2-5k --demo True
# python scripts/train_pia.py --model-type ldm --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k --holdout-dataset ffhq-2-5k --demo True

python scripts/train_drc.py --model-type ldm --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k --holdout-dataset ffhq-2-5k --demo True
# python scripts/train_pfami.py --model-type ldm --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k --holdout-dataset ffhq-2-5k --demo True
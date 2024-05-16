python scripts/train_secmi.py --model-type ldm --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k --holdout-dataset ffhq-2-5k --eval True --batch-size 32
python scripts/train_pia.py --model-type ldm --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k --holdout-dataset ffhq-2-5k  --eval True --batch-size 32
python scripts/train_pfami.py --model-type ldm --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k --holdout-dataset ffhq-2-5k --eval True --batch-size 32
python scripts/train_secmi.py --model-type sd --ckpt-path ../models/diffusers/stable-diffusion-v1-5/ --eval True --batch-size 24
python scripts/train_pia.py --model-type sd --ckpt-path ../models/diffusers/stable-diffusion-v1-5/  --eval True --batch-size 24
python scripts/train_pfami.py --model-type sd --ckpt-path ../models/diffusers/stable-diffusion-v1-5/ --eval True --batch-size 24

python scripts/train_secmi.py --model-type sd --ckpt-path ../models/diffusers/stable-diffusion-v1-5/ --member-dataset laion-aesthetic-2-5k-eval --holdout-dataset coco2017-val-2-5k-eval --eval True --batch-size 64
python scripts/train_pia.py --model-type sd --ckpt-path ../models/diffusers/stable-diffusion-v1-5/  --member-dataset laion-aesthetic-2-5k-eval --holdout-dataset coco2017-val-2-5k-eval --eval True --batch-size 64
python scripts/train_pfami.py --model-type sd --ckpt-path ../models/diffusers/stable-diffusion-v1-5/ --member-dataset laion-aesthetic-2-5k-eval --holdout-dataset coco2017-val-2-5k-eval --eval True --batch-size 64
python scripts/train_secmi.py --model-type ldm --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k-eval --holdout-dataset ffhq-2-5k-eval --eval True --batch-size 64
python scripts/train_pia.py --model-type ldm --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k-eval --holdout-dataset ffhq-2-5k-eval  --eval True --batch-size 64
python scripts/train_pfami.py --model-type ldm --ckpt-path ../models/diffusers/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k-eval --holdout-dataset ffhq-2-5k-eval --eval True --batch-size 64

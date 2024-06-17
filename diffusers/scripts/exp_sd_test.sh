python scripts/train_secmi.py --model-type sd --ckpt-path models/diffusers/stable-diffusion-v1-5/ --member-dataset laion-aesthetic-2-5k-test --holdout-dataset coco2017-val-2-5k-test --batch-size 64 --eval True
python scripts/train_pia.py --model-type sd --ckpt-path models/diffusers/stable-diffusion-v1-5/  --member-dataset laion-aesthetic-2-5k-test --holdout-dataset coco2017-val-2-5k-test --batch-size 64 --eval True
python scripts/train_pfami.py --model-type sd --ckpt-path models/diffusers/stable-diffusion-v1-5/ --member-dataset laion-aesthetic-2-5k-test --holdout-dataset coco2017-val-2-5k-test --batch-size 64 --eval True
python scripts/train_gsa.py --model-type sd --gsa-mode 1 --ckpt-path models/diffusers/stable-diffusion-v1-5/ --member-dataset laion-aesthetic-2-5k-test --holdout-dataset coco2017-val-2-5k-test --batch-size 2 --eval True
python scripts/train_gsa.py --model-type sd --gsa-mode 2 --ckpt-path models/diffusers/stable-diffusion-v1-5/  --member-dataset laion-aesthetic-2-5k-test --holdout-dataset coco2017-val-2-5k-test --batch-size 2 --eval True
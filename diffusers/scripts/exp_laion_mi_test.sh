python scripts/train_secmi.py --model-type sd --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-test --holdout-dataset laion-mi-nonmember-test --batch-size 32 --eval True
python scripts/train_pia.py --model-type sd --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-mi-member-test --holdout-dataset laion-mi-nonmember-test --batch-size 32 --eval True
python scripts/train_pfami.py --model-type sd --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-test --holdout-dataset laion-mi-nonmember-test --batch-size 32 --eval True
python scripts/train_gsa.py --model-type sd --gsa-mode 1 --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-test --holdout-dataset laion-mi-nonmember-test --batch-size 2 --eval True
python scripts/train_gsa.py --model-type sd --gsa-mode 2 --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-mi-member-test --holdout-dataset laion-mi-nonmember-test --batch-size 2 --eval True
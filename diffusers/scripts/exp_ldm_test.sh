python scripts/train_secmi.py --model-type ldm --ckpt-path models/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k-test --holdout-dataset ffhq-2-5k-test --batch-size 64 --eval True
python scripts/train_pia.py --model-type ldm --ckpt-path models/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k-test --holdout-dataset ffhq-2-5k-test  --batch-size 64 --eval True
python scripts/train_pfami.py --model-type ldm --ckpt-path models/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k-test --holdout-dataset ffhq-2-5k-test --batch-size 64 --eval True
python scripts/train_gsa.py --model-type ldm --gsa-mode 1 --ckpt-path models/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k-test --holdout-dataset ffhq-2-5k-test --batch-size 4 --eval True
python scripts/train_gsa.py --model-type ldm --gsa-mode 2 --ckpt-path models/ldm-celebahq-256/ --member-dataset celeba-hq-2-5k-test --holdout-dataset ffhq-2-5k-test --batch-size 4 --eval True



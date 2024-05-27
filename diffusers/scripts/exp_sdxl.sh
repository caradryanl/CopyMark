python scripts/train_gsa.py --model-type sdxl --gsa-mode 1 --ckpt-path ../models/diffusers/CommonCanvas-XL-C/ --member-dataset commoncatalog-2-5k --holdout-dataset laion-aesthetic-2-5k --batch-size 1
python scripts/train_gsa.py --model-type sdxl --gsa-mode 2 --ckpt-path ../models/diffusers/CommonCanvas-XL-C/ --member-dataset commoncatalog-2-5k --holdout-dataset laion-aesthetic-2-5k --batch-size 1 
python scripts/train_secmi.py --model-type sdxl --ckpt-path ../models/diffusers/CommonCanvas-XL-C/ --member-dataset commoncatalog-2-5k --holdout-dataset laion-aesthetic-2-5k --batch-size 24
python scripts/train_pia.py --model-type sdxl --ckpt-path ../models/diffusers/CommonCanvas-XL-C/  --member-dataset commoncatalog-2-5k --holdout-dataset laion-aesthetic-2-5k --batch-size 24
python scripts/train_pfami.py --model-type sdxl --ckpt-path ../models/diffusers/CommonCanvas-XL-C/ --member-dataset commoncatalog-2-5k --holdout-dataset laion-aesthetic-2-5k --batch-size 16
# python scripts/train_gsa.py --model-type sdxl --gsa-mode 1 --ckpt-path ../models/diffusers/Kohaku-XL-Epsilon/ --member-dataset hakubooru-2-5k-member --holdout-dataset hakubooru-2-5k-nonmember --batch-size 1 --demo True

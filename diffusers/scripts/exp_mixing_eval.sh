python scripts/train_secmi.py --model-type coco100_laion0 --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval --batch-size 32
python scripts/train_pia.py --model-type coco100_laion0 --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval --batch-size 32
python scripts/train_pfami.py --model-type coco100_laion0 --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval --batch-size 32
python scripts/train_gsa.py --model-type coco100_laion0 --gsa-mode 1 --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval --batch-size 2
python scripts/train_gsa.py --model-type coco100_laion0 --gsa-mode 2 --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval --batch-size 2

python scripts/train_secmi.py --model-type coco25_laion75 --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-25-laion-mi-nonmember-eval-75 --batch-size 32
python scripts/train_pia.py --model-type coco25_laion75 --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-25-laion-mi-nonmember-eval-75 --batch-size 32
python scripts/train_pfami.py --model-type coco25_laion75 --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-25-laion-mi-nonmember-eval-75 --batch-size 32
python scripts/train_gsa.py --model-type coco25_laion75 --gsa-mode 1 --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-25-laion-mi-nonmember-eval-75 --batch-size 2
python scripts/train_gsa.py --model-type coco25_laion75 --gsa-mode 2 --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-25-laion-mi-nonmember-eval-75 --batch-size 2

python scripts/train_secmi.py --model-type coco50_laion50 --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-50-laion-mi-nonmember-eval-50 --batch-size 32
python scripts/train_pia.py --model-type coco50_laion50 --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-50-laion-mi-nonmember-eval-50 --batch-size 32
python scripts/train_pfami.py --model-type coco50_laion50 --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-50-laion-mi-nonmember-eval-50 --batch-size 32
python scripts/train_gsa.py --model-type coco50_laion50 --gsa-mode 1 --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-50-laion-mi-nonmember-eval-50 --batch-size 2
python scripts/train_gsa.py --model-type coco50_laion50 --gsa-mode 2 --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-50-laion-mi-nonmember-eval-50 --batch-size 2

python scripts/train_secmi.py --model-type coco75_laion25 --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-75-laion-mi-nonmember-eval-25 --batch-size 32
python scripts/train_pia.py --model-type coco75_laion25 --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-75-laion-mi-nonmember-eval-25 --batch-size 32
python scripts/train_pfami.py --model-type coco75_laion25 --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-75-laion-mi-nonmember-eval-25 --batch-size 32
python scripts/train_gsa.py --model-type coco75_laion25 --gsa-mode 1 --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-75-laion-mi-nonmember-eval-25 --batch-size 2
python scripts/train_gsa.py --model-type coco75_laion25 --gsa-mode 2 --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-mi-member-eval --holdout-dataset coco2017-eval-75-laion-mi-nonmember-eval-25 --batch-size 2
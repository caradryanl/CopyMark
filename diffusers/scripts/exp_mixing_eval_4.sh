# python scripts/train_secmi.py --model-type coco0_laion100_plus --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-aesthetic-eval --holdout-dataset laion-mi-nonmember-eval --batch-size 32
# python scripts/train_pia.py --model-type coco0_laion100_plus --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-aesthetic-eval --holdout-dataset laion-mi-nonmember-eval --batch-size 32

# python scripts/train_secmi.py --model-type coco25_laion75_plus --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-aesthetic-eval --holdout-dataset coco2017-eval-25-laion-mi-nonmember-eval-75 --batch-size 32
# python scripts/train_pia.py --model-type coco25_laion75_plus --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-aesthetic-eval --holdout-dataset coco2017-eval-25-laion-mi-nonmember-eval-75 --batch-size 32


# python scripts/train_secmi.py --model-type coco50_laion50_plus --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-aesthetic-eval --holdout-dataset coco2017-eval-50-laion-mi-nonmember-eval-50 --batch-size 32
# python scripts/train_pia.py --model-type coco50_laion50_plus --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-aesthetic-eval --holdout-dataset coco2017-eval-50-laion-mi-nonmember-eval-50 --batch-size 32


python scripts/train_secmi.py --model-type coco75_laion25_plus --ckpt-path models/stable-diffusion-v1-5/ --member-dataset laion-aesthetic-eval --holdout-dataset coco2017_eval_75_laion_mi_nonmember_eval_25 --batch-size 32
python scripts/train_pia.py --model-type coco75_laion25_plus --ckpt-path models/stable-diffusion-v1-5/  --member-dataset laion-aesthetic-eval --holdout-dataset coco2017_eval_75_laion_mi_nonmember_eval_25 --batch-size 32

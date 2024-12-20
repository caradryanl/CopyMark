python scripts/train_secmi.py --model-type sdxl --ckpt-path models/CommonCanvas-XL-C/ --member-dataset commoncatalog-2-5k-eval --holdout-dataset coco2017-val-2-5k-eval --batch-size 16
python scripts/train_pia.py --model-type sdxl --ckpt-path models/CommonCanvas-XL-C/  --member-dataset commoncatalog-2-5k-eval --holdout-dataset coco2017-val-2-5k-eval --batch-size 16
python scripts/train_pfami.py --model-type sdxl --ckpt-path models/CommonCanvas-XL-C/ --member-dataset commoncatalog-2-5k-eval --holdout-dataset coco2017-val-2-5k-eval --batch-size 16
python scripts/train_gsa.py --model-type sdxl --gsa-mode 1 --ckpt-path models/CommonCanvas-XL-C/ --member-dataset commoncatalog-2-5k-eval --holdout-dataset coco2017-val-2-5k-eval --batch-size 1
python scripts/train_gsa.py --model-type sdxl --gsa-mode 2 --ckpt-path models/CommonCanvas-XL-C/ --member-dataset commoncatalog-2-5k-eval --holdout-dataset coco2017-val-2-5k-eval --batch-size 1 

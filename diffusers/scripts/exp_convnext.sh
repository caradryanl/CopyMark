python scripts/train_convnext.py celeba-hq-eval ffhq-eval celeba-hq-test ffhq-test --epochs 10 --batch_size 64 --lr 0.0001 --output roc_curve_celeba.png --model_path model_celeba.pth
python scripts/train_convnext.py laion-aesthetic-eval coco2017-eval laion-aesthetic-test coco2017-test --epochs 10 --batch_size 64 --lr 0.0001 --output roc_curve_laion.png --model_path model_laion.pth
python scripts/train_convnext.py laion-mi-member-eval laion-mi-nonmember-eval laion-mi-member-test laion-mi-nonmember-test --epochs 10 --batch_size 64 --lr 0.0001 --output roc_curve_laion_mi.png --model_path model_laion_mi.pth
python scripts/train_convnext.py commoncatalog-eval coco2017-eval commoncatalog-test coco2017-test --epochs 10 --batch_size 64 --lr 0.0001 --output roc_curve_cc.png --model_path model_cc.pth
python scripts/train_convnext.py hakubooru-member-eval hakubooru-nonmember-eval hakubooru-member-test hakubooru-nonmember-test --epochs 10 --batch_size 64 --lr 0.0001 --output roc_curve_haku.png --model_path model_haku.pth


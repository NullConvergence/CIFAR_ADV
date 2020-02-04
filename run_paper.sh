# python -m clean.train --config=configs/paper/resnet-clean.yml
# python -m pgd.train --config=configs/paper/resnet-madry.yml
# python -m mixed.train --config=configs/paper/resnet-warm.yml
python -m free.train --config=configs/paper/resnet-free.yml
python -m free.train_mixed --config=configs/paper/resnet-free-warm.yml

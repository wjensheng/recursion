# activate virtual
actVir

# unlock
chmod u+rwx data/pixel_stats.csv

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --num_epochs=60 \
                 --batch_size=24 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0003 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=cosine \
                 --t_max=184
                 --eta_min=0.00003

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --num_epochs=60 \
                 --batch_size=24 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=cosine \
                 --t_max=184
                 --eta_min=0.00005

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --num_epochs=60 \
                 --batch_size=24 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0007 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=cosine \
                 --t_max=184
                 --eta_min=0.00007

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --num_epochs=60 \
                 --batch_size=24 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.001 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=cosine \
                 --t_max=184
                 --eta_min=0.0001
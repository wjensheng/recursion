# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cosface \
                 --optim_lr=0.0002 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40 

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cosface \
                 --optim_lr=0.0003 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40 

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cosface \
                 --optim_lr=0.0007 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40 

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cosface \
                 --optim_lr=0.001 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40 

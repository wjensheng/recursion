# =========================== densenet ============================
# # default: cross_entropy
# python3 train.py --config=configs/densenet201.yml \
#                  --num_epochs=60 \
#                  --batch_size=32 \
#                  --image_size=320 \
#                  --loss=cross_entropy \
#                  --optim_lr=0.0001 \
#                  --optim_wd=0.01 \
#                  --num_grad_acc=0 \
#                  --ls=true \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40                                                                  

# arcface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=24 \
                 --image_size=512 \
                 --loss=arcface \
                 --optim_lr=0.0003 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                                                   

# arcface, ls=true, bestfitting=true
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=24 \
                 --image_size=512 \
                 --loss=arcface \
                 --optim_lr=0.0003 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                                                   
                                               

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=24 \
                 --image_size=512 \
                 --loss=cosface \
                 --optim_lr=0.0003 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                                                   

# cosface, ls=true, bestfitting=true
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=24 \
                 --image_size=512 \
                 --loss=cosface \
                 --optim_lr=0.0003 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40
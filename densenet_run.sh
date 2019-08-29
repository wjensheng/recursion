# =========================== densenet ============================
# default: cross_entropy
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=320 \
                 --loss=cross_entropy \
                 --optim_lr=0.0003 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40


python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=320 \
                 --loss=cross_entropy \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40


python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=320 \
                 --loss=cross_entropy \
                 --optim_lr=0.0007 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40


python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=320 \
                 --loss=cross_entropy \
                 --optim_lr=0.0009 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40


python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=320 \
                 --loss=cross_entropy \
                 --optim_lr=0.001 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40


python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=320 \
                 --loss=cross_entropy \
                 --optim_lr=0.003 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40
                 
# # ls_cross_entropy
# python3 train.py --config=configs/densenet201.yml \
#                  --num_epochs=60 \
#                  --batch_size=32 \
#                  --image_size=320 \
#                  --loss=ls_cross_entropy \
#                  --optim_lr=0.0005 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=true \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40

# # arcface, ls=false, bestfitting=false
# python3 train.py --config=configs/densenet201.yml \
#                  --num_epochs=60 \
#                  --batch_size=32 \
#                  --image_size=320 \
#                  --loss=arcface \
#                  --optim_lr=0.0005 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=false \
#                  --bestfitting=false \
#                  --scheduler=step \
#                  --step_size=40                                  

# # arcface, ls=true, bestfitting=false
# python3 train.py --config=configs/densenet201.yml \
#                  --num_epochs=60 \
#                  --batch_size=32 \
#                  --image_size=320 \
#                  --loss=arcface \
#                  --optim_lr=0.0005 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=true \
#                  --bestfitting=false \
#                  --scheduler=step \
#                  --step_size=40                                                   

# # arcface, ls=false, bestfitting=true
# python3 train.py --config=configs/densenet201.yml \
#                  --num_epochs=60 \
#                  --batch_size=32 \
#                  --image_size=320 \
#                  --loss=arcface \
#                  --optim_lr=0.0005 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=false \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40                                                   

# # arcface, ls=true, bestfitting=true
# python3 train.py --config=configs/densenet201.yml \
#                  --num_epochs=60 \
#                  --batch_size=32 \
#                  --image_size=320 \
#                  --loss=arcface \
#                  --optim_lr=0.0005 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=true \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40                                                   

# # cosface, ls=false, bestfitting=false
# python3 train.py --config=configs/densenet201.yml \
#                  --num_epochs=60 \
#                  --batch_size=32 \
#                  --image_size=320 \
#                  --loss=cosface \
#                  --optim_lr=0.0005 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=false \
#                  --bestfitting=false \
#                  --scheduler=step \
#                  --step_size=40                                  

# # cosface, ls=true, bestfitting=false
# python3 train.py --config=configs/densenet201.yml \
#                  --num_epochs=60 \
#                  --batch_size=32 \
#                  --image_size=320 \
#                  --loss=cosface \
#                  --optim_lr=0.0005 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=true \
#                  --bestfitting=false \
#                  --scheduler=step \
#                  --step_size=40                                                   

# # cosface, ls=false, bestfitting=true
# python3 train.py --config=configs/densenet201.yml \
#                  --num_epochs=60 \
#                  --batch_size=32 \
#                  --image_size=320 \
#                  --loss=cosface \
#                  --optim_lr=0.0005 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=false \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40                                                   

# # cosface, ls=true, bestfitting=true
# python3 train.py --config=configs/densenet201.yml \
#                  --num_epochs=60 \
#                  --batch_size=32 \
#                  --image_size=320 \
#                  --loss=cosface \
#                  --optim_lr=0.0005 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=true \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40
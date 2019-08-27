# NOTE:
# This script contains 12 experiments for 
# each architecture (resnet50 & densenet201)
# to find the best loss function.
# Arguments ls and bestfitting only needed for loss=arcface or cosface.

# After running all experiments, i.e., script finishes running,
# tune learning rate for the best loss function. 
# Find the best num_grad_acc, from 0, 2, 4, 8
# Optim_wd can set to 0.01 once the best num_grad_acc is found.

# Other stuff to be aware of:
# Try tuning for batch_size of 16, 24, 32.
# Use smaller image size to speed up experiments.
# Increase step_size if loss flattens out and 
# decrease step_size if loss remains constant
# Note that RAdam, a.k.a 'radam' is not compatible with 
# CosineAnnealing, a.k.a 'cosine' as the scheduler

# default: cross_entropy
python3 train.py --config=configs/resnet50.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cross_entropy \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# ls_cross_entropy
python3 train.py --config=configs/resnet50.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=ls_cross_entropy \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# arcface, ls=false, bestfitting=false
python3 train.py --config=configs/resnet50.yml \
                 --num_epochs=60 \
                 --batch_size=32 \ 
                 --image_size=256 \
                 --loss=arcface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=false \
                 --scheduler=step \
                 --step_size=40                                  

# arcface, ls=true, bestfitting=false
python3 train.py --config=configs/resnet50.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=arcface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=false \
                 --scheduler=step \
                 --step_size=40                                                   

# arcface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet50.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=arcface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                                                   

# arcface, ls=true, bestfitting=true
python3 train.py --config=configs/resnet50.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=arcface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                                                   

# cosface, ls=false, bestfitting=false
python3 train.py --config=configs/resnet50.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=false \
                 --scheduler=step \
                 --step_size=40                                  

# cosface, ls=true, bestfitting=false
python3 train.py --config=configs/resnet50.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=false \
                 --scheduler=step \
                 --step_size=40                                                   

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet50.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                                                   

# cosface, ls=true, bestfitting=true
python3 train.py --config=configs/resnet50.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# =========================== densenet ============================
# default: cross_entropy
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cross_entropy \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# ls_cross_entropy
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=ls_cross_entropy \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# arcface, ls=false, bestfitting=false
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=arcface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=false \
                 --scheduler=step \
                 --step_size=40                                  

# arcface, ls=true, bestfitting=false
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=arcface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=false \
                 --scheduler=step \
                 --step_size=40                                                   

# arcface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=arcface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                                                   

# arcface, ls=true, bestfitting=true
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=arcface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                                                   

# cosface, ls=false, bestfitting=false
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=false \
                 --scheduler=step \
                 --step_size=40                                  

# cosface, ls=true, bestfitting=false
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=false \
                 --scheduler=step \
                 --step_size=40                                                   

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                                                   

# cosface, ls=true, bestfitting=true
python3 train.py --config=configs/densenet201.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=256 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=true \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40
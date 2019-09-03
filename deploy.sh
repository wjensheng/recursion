# activate virtual
source env/bin/activate

# unlock
chmod u+rwx data/pixel_stats.csv


# ======= densenet121 =========
# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet121.yml \
                 --num_epochs=80 \
                 --batch_size=16 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=4 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet121.yml \
                 --num_epochs=80 \
                 --batch_size=16 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=2 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                  

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet121.yml \
                 --num_epochs=80 \
                 --batch_size=16 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                 

# # # cosface, ls=false, bestfitting=true
# # python3 train.py --config=configs/densenet121.yml \
# #                  --num_epochs=60 \
# #                  --batch_size=16 \
# #                  --image_size=320 \
# #                  --loss=cosface \
# #                  --optim_lr=0.0005 \
# #                  --optim_wd=0 \
# #                  --num_grad_acc=0 \
# #                  --ls=false \
# #                  --bestfitting=true \
# #                  --scheduler=step \
# #                  --step_size=40

# # # cosface, ls=false, bestfitting=true
# # python3 train.py --config=configs/densenet121.yml \
# #                  --num_epochs=60 \
# #                  --batch_size=24 \
# #                  --image_size=320 \
# #                  --loss=cosface \
# #                  --optim_lr=0.0005 \
# #                  --optim_wd=0 \
# #                  --num_grad_acc=0 \
# #                  --ls=false \
# #                  --bestfitting=true \
# #                  --scheduler=step \
# #                  --step_size=40

# # # cosface, ls=false, bestfitting=true
# # python3 train.py --config=configs/densenet121.yml \
# #                  --num_epochs=60 \
# #                  --batch_size=32 \
# #                  --image_size=320 \
# #                  --loss=cosface \
# #                  --optim_lr=0.0005 \
# #                  --optim_wd=0 \
# #                  --num_grad_acc=0 \
# #                  --ls=false \
# #                  --bestfitting=true \
# #                  --scheduler=step \
# #                  --step_size=40


# # python3 train.py --config=configs/densenet121.yml \
# #                  --num_epochs=60 \
# #                  --batch_size=32 \
# #                  --image_size=320 \
# #                  --loss=cosface \
# #                  --optim_lr=0.0007 \
# #                  --optim_wd=0 \
# #                  --num_grad_acc=0 \
# #                  --ls=false \
# #                  --bestfitting=true \
# #                  --scheduler=step \
# #                  --step_size=40

# # # cosface, ls=false, bestfitting=true
# # python3 train.py --config=configs/densenet121.yml \
# #                  --num_epochs=60 \
# #                  --batch_size=32 \
# #                  --image_size=320 \
# #                  --loss=cosface \
# #                  --optim_lr=0.001 \
# #                  --optim_wd=0 \
# #                  --num_grad_acc=0 \
# #                  --ls=false \
# #                  --bestfitting=true \
# #                  --scheduler=step \
# #                  --step_size=40                 


# # ====== resnet101 ======
# # cosface, ls=false, bestfitting=true
# python3 train.py --config=configs/resnet101.yml \
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

# # cosface, ls=false, bestfitting=true
# python3 train.py --config=configs/resnet101.yml \
#                  --num_epochs=60 \
#                  --batch_size=32 \
#                  --image_size=320 \
#                  --loss=cosface \
#                  --optim_lr=0.0007 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=false \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40

# # cosface, ls=false, bestfitting=true
# python3 train.py --config=configs/resnet101.yml \
#                  --num_epochs=60 \
#                  --batch_size=32 \
#                  --image_size=320 \
#                  --loss=cosface \
#                  --optim_lr=0.001 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=false \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40

# # cosface, ls=false, bestfitting=true
# python3 train.py --config=configs/resnet101.yml \
#                  --num_epochs=60 \
#                  --batch_size=24 \
#                  --image_size=320 \
#                  --loss=cosface \
#                  --optim_lr=0.001 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=false \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40

# # cosface, ls=false, bestfitting=true
# python3 train.py --config=configs/resnet101.yml \
#                  --num_epochs=60 \
#                  --batch_size=48 \
#                  --image_size=320 \
#                  --loss=cosface \
#                  --optim_lr=0.001 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=false \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40


# # cosface, ls=false, bestfitting=true
# python3 train.py --config=configs/resnet101.yml \
#                  --num_epochs=60 \
#                  --batch_size=32 \
#                  --image_size=320 \
#                  --loss=cosface \
#                  --optim_lr=0.0015 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=false \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40

gcloud beta compute instances stop --zone "us-central1-c" "instance-p4-1-1"
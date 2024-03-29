# activate virtual
source env/bin/activate

# unlock
chmod u+rwx data/pixel_stats.csv

# ======= densenet121 =========
# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet121.yml \
                 --cell_type=3 \
                 --num_epochs=60 \
                 --batch_size=16 \
                 --image_size=512 \
                 --loss=cosface \
                 --optim_lr=0.0012 \
                 --optim_wd=0 \
                 --num_grad_acc=2 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

python3 train.py --config=configs/densenet121.yml \
                 --cell_type=2 \
                 --num_epochs=60 \
                 --batch_size=16 \
                 --image_size=512 \
                 --loss=cosface \
                 --optim_lr=0.0012 \
                 --optim_wd=0 \
                 --num_grad_acc=2 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

python3 train.py --config=configs/densenet121.yml \
                 --cell_type=1 \
                 --num_epochs=60 \
                 --batch_size=16 \
                 --image_size=512 \
                 --loss=cosface \
                 --optim_lr=0.0012 \
                 --optim_wd=0 \
                 --num_grad_acc=2 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

python3 train.py --config=configs/densenet121.yml \
                 --cell_type=0 \
                 --num_epochs=60 \
                 --batch_size=16 \
                 --image_size=512 \
                 --loss=cosface \
                 --optim_lr=0.0012 \
                 --optim_wd=0 \
                 --num_grad_acc=2 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                                  

# # ======= resnet101 =========
# # cosface, ls=false, bestfitting=true
# python3 train.py --config=configs/resnet101.yml \
#                  --cell_type=3 \
#                  --num_epochs=60 \
#                  --batch_size=16 \
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
#                  --cell_type=3 \
#                  --num_epochs=60 \
#                  --batch_size=16 \
#                  --image_size=320 \
#                  --loss=cosface \
#                  --optim_lr=0.0012 \
#                  --optim_wd=0 \
#                  --num_grad_acc=2 \
#                  --ls=false \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40                 

# # cosface, ls=false, bestfitting=true
# python3 train.py --config=configs/resnet101.yml \
#                  --cell_type=3 \
#                  --num_epochs=60 \
#                  --batch_size=16 \
#                  --image_size=320 \
#                  --loss=cosface \
#                  --optim_lr=0.0012 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=false \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40                  

# # cosface, ls=false, bestfitting=true
# python3 train.py --config=configs/resnet101.yml \
#                  --cell_type=3 \
#                  --num_epochs=60 \
#                  --batch_size=16 \
#                  --image_size=320 \
#                  --loss=cosface \
#                  --optim_lr=0.0015 \
#                  --optim_wd=0 \
#                  --num_grad_acc=2 \
#                  --ls=false \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40    

# # cosface, ls=false, bestfitting=true
# python3 train.py --config=configs/resnet101.yml \
#                  --cell_type=3 \
#                  --num_epochs=60 \
#                  --batch_size=16 \
#                  --image_size=320 \
#                  --loss=cosface \
#                  --optim_lr=0.0015 \
#                  --optim_wd=0 \
#                  --num_grad_acc=0 \
#                  --ls=false \
#                  --bestfitting=true \
#                  --scheduler=step \
#                  --step_size=40

python3 utils/apply_leak.py --config=configs/densenet121.yml

gcloud beta compute instances stop --zone "us-central1-c" "instance-p4-1-2"
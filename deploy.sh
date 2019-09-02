# activate virtual
source env/bin/activate

# unlock
chmod u+rwx data/pixel_stats.csv

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet34.yml \
                 --num_epochs=60 \
                 --batch_size=48 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0003 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet34.yml \
                 --num_epochs=60 \
                 --batch_size=48 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet34.yml \
                 --num_epochs=60 \
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

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet34.yml \
                 --num_epochs=60 \
                 --batch_size=24 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet34.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40


python3 train.py --config=configs/resnet34.yml \
                 --num_epochs=60 \
                 --batch_size=48 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0007 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet34.yml \
                 --num_epochs=60 \
                 --batch_size=48 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.001 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40


# ======= densenet121 =========
# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet121.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0003 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet121.yml \
                 --num_epochs=60 \
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

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet121.yml \
                 --num_epochs=60 \
                 --batch_size=24 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet121.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40


python3 train.py --config=configs/densenet121.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0007 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet121.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.001 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                 


# ====== resnet101 ======
# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --num_epochs=60 \
                 --batch_size=48 \
                 --image_size=320 \
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
                 --batch_size=48 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --num_epochs=60 \
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
                 --scheduler=step \
                 --step_size=40

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --num_epochs=60 \
                 --batch_size=32 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40


python3 train.py --config=configs/resnet101.yml \
                 --num_epochs=60 \
                 --batch_size=48 \
                 --image_size=320 \
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
                 --batch_size=48 \
                 --image_size=320 \
                 --loss=cosface \
                 --optim_lr=0.001 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=40                 
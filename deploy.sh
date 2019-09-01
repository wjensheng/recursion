# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet121.yml \
                 --num_epochs=60 \
                 --batch_size=16 \
                 --image_size=512 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=step \
                 --step_size=30 

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/densenet121.yml \
                 --num_epochs=60 \
                 --batch_size=16 \
                 --image_size=512 \
                 --loss=cosface \
                 --optim_lr=0.0005 \
                 --optim_wd=0 \
                 --num_grad_acc=0 \
                 --ls=false \
                 --bestfitting=true \
                 --scheduler=cosine \
                 --eta_min=0.00005 \
                 --t_max=138
# activate virtual
source env/bin/activate

# unlock
chmod u+rwx data/pixel_stats.csv

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --batch_size=48 \
                 --optim_lr=0.0005 \
                 --scheduler=cosine \
                 --t_max=276
                 --eta_min=0.00005

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --batch_size=48 \
                 --optim_lr=0.0005 \
                 --scheduler=step \
                 --step_size=40

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --batch_size=48 \
                 --optim_lr=0.0001 \
                 --scheduler=step \
                 --step_size=40

# cosface, ls=false, bestfitting=true
python3 train.py --config=configs/resnet101.yml \
                 --batch_size=48 \
                 --optim_lr=0.0015 \
                 --scheduler=step \
                 --step_size=40

# gcloud beta compute instances stop --zone "us-central1-c" "instance-p4-1-1"
cd /home/hhg/Research/galaxyClassify/repo/galaxyZooNet

python3 galaxyZooNet/train_resnet50.py --config experiment/resnet50/configs/run6.yaml > experiment/resnet50/logs/run6.log

# ====== jub submit ======
#python3 galaxyZooNet/train_resnet50.py --config configs/resnet50_run0.yaml > job_log/run0.log
#python3 galaxyZooNet/train_resnet50.py --config configs/resnet50_run1.yaml > job_log/run1.log
#python3 galaxyZooNet/train_resnet50.py --config configs/resnet50_run1.1.yaml > job_log/run1.1log

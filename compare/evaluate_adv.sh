DATE=$(date +"%Y-%m-%d-%H-%M")
EXPERIMENT="MRNet-${DATE}-MRNet"
DATA_PATH='/Users/ex10si0n/MRNet-v1.0/'

EPOCHS=50
PREFIX=MRNet
MODEL='./exp/MRNet-2023-03-22-16-48-MRNet-0.00001-0.06/models/'
W='adv'

python3 evaluate.py -t acl -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --model=$MODEL'model_MRNet_acl_sagittal.pth' --weight=$W
python3 evaluate.py -t acl -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --model=$MODEL'model_MRNet_acl_coronal.pth' --weight=$W
python3 evaluate.py -t acl -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --model=$MODEL'model_MRNet_acl_axial.pth' --weight=$W

python3 evaluate.py -t meniscus -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --model=$MODEL'model_MRNet_meniscus_sagittal.pth' --weight=$W
python3 evaluate.py -t meniscus -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --model=$MODEL'model_MRNet_meniscus_coronal.pth' --weight=$W
python3 evaluate.py -t meniscus -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --model=$MODEL'model_MRNet_meniscus_axial.pth' --weight=$W

python3 evaluate.py -t abnormal -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --model=$MODEL'model_MRNet_abnormal_sagittal.pth' --weight=$W
python3 evaluate.py -t abnormal -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --model=$MODEL'model_MRNet_abnormal_coronal.pth' --weight=$W
python3 evaluate.py -t abnormal -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --model=$MODEL'model_MRNet_abnormal_axial.pth' --weight=$W


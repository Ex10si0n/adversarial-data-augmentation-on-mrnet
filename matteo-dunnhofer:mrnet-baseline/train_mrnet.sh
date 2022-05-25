DATE=$(date +"%Y-%m-%d-%H-%M")
EXPERIMENT="MRNet-${DATE}-MRNet"
DATA_PATH='/home/matteo/Desktop/data/MRI/MRNet-v1.0/'

EPOCHS=50
PREFIX=MRNet

python3 train.py -t acl -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS
python3 train.py -t acl -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS 
python3 train.py -t acl -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS 

python3 train.py -t meniscus -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS 
python3 train.py -t meniscus -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS
python3 train.py -t meniscus -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS

python3 train.py -t abnormal -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS
python3 train.py -t abnormal -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS
python3 train.py -t abnormal -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS

python3 train_logistic_regression.py --path-to-model "experiments/${EXPERIMENT}/models/" --data-path $DATA_PATH 

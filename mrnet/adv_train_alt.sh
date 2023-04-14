DATE=$(date +"%Y-%m-%d-%H-%M")
EXPERIMENT="MRNet-${DATE}-MRNet"
DATA_PATH='/Users/ex10si0n/MRNet-v1.0/'
EPOCHS=60
PREFIX=MRNet

for PERCENT in 0.06
do
  for EPS in 1e-5
  do
    DATE=$(date +"%Y-%m-%d-%H-%M")
    EXPERIMENT="MRNet-${DATE}-MRNet-${EPS}-${PERCENT}"

    python3 train.py -t acl -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --advtrain 1 --advtrain_percent $PERCENT --epsilon $EPS
    python3 train.py -t acl -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --advtrain 1 --advtrain_percent $PERCENT --epsilon $EPS
    python3 train.py -t acl -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --advtrain 1 --advtrain_percent $PERCENT --epsilon $EPS

    python3 train.py -t meniscus -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --advtrain 1 --advtrain_percent $PERCENT --epsilon $EPS
    python3 train.py -t meniscus -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --advtrain 1 --advtrain_percent $PERCENT --epsilon $EPS
    python3 train.py -t meniscus -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --advtrain 1 --advtrain_percent $PERCENT --epsilon $EPS

    python3 train.py -t abnormal -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --advtrain 1 --advtrain_percent $PERCENT --epsilon $EPS
    python3 train.py -t abnormal -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --advtrain 1 --advtrain_percent $PERCENT --epsilon $EPS
    python3 train.py -t abnormal -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --advtrain 1 --advtrain_percent $PERCENT --epsilon $EPS

  done
done

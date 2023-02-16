DATE=$(date +"%Y-%m-%d-%H-%M")
EXPERIMENT="MRNet-${DATE}-MRNet"
DATA_PATH='/Users/ex10si0n/MRNet-v1.0/'
EPOCHS=20
PREFIX=MRNet
PERCENT=0.02

for EPS in 0.05 0.07 0.09 0.1
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


  python3 train_logistic_regression.py --path-to-model "experiments/${EXPERIMENT}/models/" --data-path $DATA_PATH
done

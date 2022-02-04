# break lines with \ but cannot have a space afterwards or else argument won't be recognized

export HOME=/u/cmackin
export PYTHONPATH=$PYTHONPATH:$HOME
export BASE_DIR=/u/cmackin/tiny-master/v0.5/training/anomaly_detection
#export BASE_DIR=/dccstor/transformer/charles/extracted_autoencoder_models/ares_hwa_models_v10
export DEV_DIR=$BASE_DIR/dev_data
export EVAL_DIR=$BASE_DIR/ares_eval_data
export MODEL_DIR=$BASE_DIR/ares_hwa_models_xneg
export RESULT_DIR=$BASE_DIR/ares_hwa_results_xneg

export MAX_FPR=0.1

export N_MELS=128              # default = 128, these are number of Hz bins on log mel spectrogram (y-axis)
export FRAMES=5                 # default = 5
export N_FFT=1024               # default = 1024
export HOP_LENGTH=512           # fft sliding window increment (hop_length < n_fft means overlap)
#export WIN_LENGTH=null         # default is None (type null here), which will set win_length = n_fft
export POWER=2.0                # default is 2
export LOG_MEL_START_IND=50     # default = 50
export LOG_MEL_STOP_IND=250     # default = 250
export N_LAYERS=4               # default = 5 (encoder / decoder layers)
export H_SIZE=128               # default = 128
export C_SIZE=24                # default = 8hon
export DROPOUT_RATIO=0.0        # dropout = 0.0
export PRUNE=0.0
export OUT_NOISE=0.02
export RESUME_FROM_CHECKPOINT=0

export OPTIMIZER="adam"
#export DEVICE="cpu"          # is way too slow!
export DEVICE="cuda"
export LOSS="MSE"
export EPOCHS=200
export BATCH_SIZE=512              # default = 512
export SHUFFLE=True
export VALIDATION_SPLIT=0.0
export VERBOSE=1

export BIAS=1
export MAX_WEIGHT=1.
export MAX_GRAD_NORM=100000.
export MAX_Z=20.

export BASE_LR=0.001
export WEIGHT_DECAY=0.0
export MOMENTUM=0.0
export NOISE_TYPE='additive-noise' #'custom-log-normal'
export NOISE_STD=0.04

export JB_MAIL=0

export i=0

echo "OUTPUT_DIRECTORY: " $OUTP_DIR

for FRAMES in 4
do
echo "FRAMES: " $FRAMES

for H_SIZE in 512
do
echo "H_SIZE: " $H_SIZE

for C_SIZE in 384 512
do
echo "C_SIZE: " $C_SIZE

for MAX_Z in 1000.
do
echo "MAX_Z: " $MAX_Z

#for MOMENTUM in 0. 0.9
#do
#echo "MOMENTUM: " $MOMENTUM

for N_MELS in 128
do
echo "N_MELS: " $N_MELS

for HOP_LENGTH in 512 # default 512
do
echo "HOP_LENGTH: " $HOP_LENGTH

for N_FFT in 1024 # default 1024
do
echo "N_FFT: " $N_FFT

for BATCH_SIZE in 512
do
echo "BATCH_SIZE: " $BATCH_SIZE

for MAX_WEIGHT in 1. 2.
do
echo "MAX_WEIGHT: " $MAX_WEIGHT

for PRUNE in 0.0 #0.05
do
echo "PRUNE: " $PRUNE

#for OPTIMIZER in "adam" "sgd"
#do
#echo "OPTIMIZER: " $OPTIMIZER

for BIAS in 1 # 0
do
echo "BIAS: " $BIAS

for BASE_LR in 0.001
do
echo "BASE_LR: " $BASE_LR

for N_LAYERS in 5
do
echo "N_LAYERS: " $N_LAYERS

for DROPOUT_RATIO in 0.0 #0.05
do
echo "DROPOUT_RATIO: " $DROPOUT_RATIO

for NOISE_STD in 0.02 0.03 0.04
do
echo "NOISE_STD: " $NOISE_STD

for WEIGHT_DECAY in 0.0
do
echo "WEIGHT_DECAY: " $WEIGHT_DECAY

for OUT_NOISE in 0.02 0.04 0.06
do
echo "OUT_NOISE: " $OUT_NOISE


((i=i+1))

export BASE_NAME=$i
echo "BASE_NAME: " $BASE_NAME

export SIM_NAME=train_$i
echo "SIM_NAME: " $SIM_NAME

#export LOG_FILE=$OUTP_DIR/$BASE_NAME.log
export LOG_FILE=$MODEL_DIR/train_$i.log
echo "LOG_FILE: " $LOG_FILE

#export SAVE_NAME=$BASE_NAME.pt
export MODEL_NAME=chkpt_$i.pt
echo "MODEL_NAME: " $MODEL_NAME

#export OUTPUT_XLSX_FILE=$BASE_NAME.csv
export RESULTS_FILE=$i.csv
echo "RESULTS_FILE: " $RESULTS_FILE

# FOR LOADING MODEL FOR INFERENCE
export TRAINED_MODEL=$OUTP_DIR/'model_name'.pt
echo "TRAINED_MODEL: " $TRAINED_MODEL

jbsub \
  -q x86_24h \
  -cores 1+1 \
  -mem 80g \
  -proj train \
  -n $SIM_NAME \
  -out $LOG_FILE \
  python train_torch_base.py \
  --loss $LOSS \
  --device $DEVICE \
  --n_layers $N_LAYERS \
  --h_size $H_SIZE \
  --c_size $C_SIZE \
  --frames $FRAMES \
  --bias $BIAS \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --base_lr $BASE_LR \
  --n_mels $N_MELS \
  --n_fft $N_FFT \
  --hop_length $HOP_LENGTH \
  --dropout_ratio $DROPOUT_RATIO \
  --weight_decay $WEIGHT_DECAY \
  --momentum $MOMENTUM \
  --noise_type $NOISE_TYPE \
  --noise_std $NOISE_STD \
  --out_noise $OUT_NOISE \
  --max_weight $MAX_WEIGHT \
  --prune $PRUNE \
  --max_grad_norm $MAX_GRAD_NORM \
  --max_z $MAX_Z \
  --dev_dir $DEV_DIR \
  --eval_dir $EVAL_DIR \
  --model_dir $MODEL_DIR \
  --model_name $MODEL_NAME \
  --result_dir $RESULT_DIR \

done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done


echo "SUBMITTED SIMULATIONS: " $i


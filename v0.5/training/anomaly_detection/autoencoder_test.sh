# break lines with \ but cannot have a space afterwards or else argument won't be recognized

export HOME=/u/cmackin
export PYTHONPATH=$PYTHONPATH:$HOME
export BASE_DIR=/u/cmackin/tiny-master/v0.5/training/anomaly_detection
#export BASE_DIR=/dccstor/transformer/charles/extracted_autoencoder_models
export DEV_DIR=$BASE_DIR/dev_data
export EVAL_DIR=$BASE_DIR/ares_eval_data
export MODEL_DIR=$BASE_DIR/ares_hwa_models_xneg
export RESULT_BASE=$BASE_DIR/ares_hwa_results_xneg
export RESULT_FILE=result.csv
export EXTRACT_MODEL=0

export DEVICE="cuda"

export JB_MAIL=0

for i in {1..36}
#for i in 142
do

#export SAVE_NAME=$BASE_NAME.pt
export MODEL_NAME=chkpt_$i.pt
echo "MODEL_NAME: " $MODEL_NAME

#export BASE_NAME=base_lr_${BASE_LR}_gamma_${GAMMA}_step_lr_${STEP_LR}_dropout_ratio_${DROPOUT_RATIO}_weight_decay_${WEIGHT_DECAY}_mnt_stddev_${MNT_STDDEV}_max_weight_${MAX_WEIGHT}
export BASE_NAME=$i
echo "BASE_NAME: " $BASE_NAME

export SIM_NAME=test_$i
echo "SIM_NAME: " $SIM_NAME

#export OUTPUT_XLSX_FILE=$BASE_NAME.csv
export RESULTS_FILE=$i.csv
echo "RESULTS_FILE: " $RESULTS_FILE

export RESULT_SUB=$RESULT_BASE/chkpt_$i'_results'
echo "RESULTS SUBDIR: " $RESULT_SUB

#export LOG_FILE=$OUTP_DIR/$BASE_NAME.log
export LOG_FILE=$RESULT_SUB/test_$i.log
echo "LOG_FILE: " $LOG_FILE

jbsub \
  -q x86_12h \
  -cores 1+1 \
  -mem 80g \
  -proj test \
  -n $SIM_NAME \
  -out $LOG_FILE \
  python test_torch_base.py \
  --dev_dir $DEV_DIR \
  --eval_dir $EVAL_DIR \
  --model_dir $MODEL_DIR \
  --model_name $MODEL_NAME \
  --result_dir $RESULT_SUB \
  --result_file $RESULT_FILE \
  --device $DEVICE \

done

echo "SUBMITTED SIMULATIONS: " $i


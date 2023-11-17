#!/usr/bin/env bash
# bash train_language_adapter.sh $lang 0
LANG=$1
TRAIN_DATASET_SIZE=$2
PROJECT=""
OUTPUT_DIR=""
RUN_ID=42

MAX_STEPS=250000
VALIDATION_DATASET_SIZE=1000
BATCH_SIZE=16
#MODEL_NAME="xlm-roberta-base"
MODEL_NAME="bert-base-multilingual-cased"

cd $PROJECT/adapter-transformers/examples/pytorch/language-modeling

mkdir -p $OUTPUT_DIR

python run_mlm.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config_name "20231101.$LANG" \
    --max_train_tokens $TRAIN_DATASET_SIZE \
    --validation_subset $VALIDATION_DATASET_SIZE \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --learning_rate 1e-4 \
    --max_steps $MAX_STEPS \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --metric_for_best_model eval_loss \
    --gradient_accumulation_steps 4 \
    --save_total_limit 1 \
    --train_adapter \
    --adapter_config language_adapter.json \
    --output_dir "$OUTPUT_DIR/model" \
    --run_name "${MODEL_NAME}_${LANG}_${TRAIN_DATASET_SIZE}_${RUN_ID}" \
    --adapter_name "$LANG@wiki" \
    --preprocessing_num_workers 4 \
    --do_train \
    --do_eval \
    --seed $RUN_ID \
    --fp16 2> "$OUTPUT_DIR/std.err" 1> "$OUTPUT_DIR/std.out"

#!/usr/bin/env bash

# Path to project repo
PROJECT=""
SEED=42
# Path to data
BASE_PATH=""

MODEL_NAME="bert-base-multilingual-cased"


MAX_EPOCHS=100
BATCH_SIZE=16
LR=1e-4
OUTPUT_DIR="$BASE_PATH/task_adapters/wikiann/$MODEL_NAME/$SEED"
mkdir -p $OUTPUT_DIR
ADAPTER_PATH_TEMPLATE="$BASE_PATH/language_adapters/{lang}/$MODEL_NAME/train-0/$SEED/model/{lang}@wiki"
LANGUAGES=ar

cd $PROJECT/examples/pytorch/lang-arithmetic

python run_ner.py \
--task_name wikiann \
--languages $LANGUAGES \
--model_name_or_path $MODEL_NAME \
--remove_unused_columns False \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--learning_rate $LR \
--num_train_epochs $MAX_EPOCHS \
--load_best_model_at_end \
--evaluation_strategy epoch \
--save_strategy epoch \
--metric_for_best_model en_f1 \
--save_total_limit 1 \
--output_dir "$OUTPUT_DIR/model" \
--adapter_path_template $ADAPTER_PATH_TEMPLATE \
--preprocessing_num_workers 4 \
--do_train \
--do_eval \
--seed $SEED \
--fp16 2> "$OUTPUT_DIR/std.err" 1> "$OUTPUT_DIR/std.out"

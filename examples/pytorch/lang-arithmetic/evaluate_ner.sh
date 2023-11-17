#!/usr/bin/env bash
BASE_PATH=""
PROJECT=""

MAX_EPOCHS=100
BATCH_SIZE=16
MODEL_NAME="xlm-roberta-base"
seed=42

LANGUAGES=ar,bg,de,el,en,es,fr,hi,ru,sw,tr,ur,vi,zh
TASK_NAME="wikiann"

OUTPUT_DIR="$BASE_PATH/task_adapters/$TASK_NAME/$MODEL_NAME/$seed"
ADAPTER_PATH_TEMPLATE="$BASE_PATH/language_adapters/{lang}/$MODEL_NAME/train-0/$seed/model/{lang}@wiki"
TASK_ADAPTER_PATH="$BASE_PATH/task_adapters/$TASK_NAME/$MODEL_NAME/$seed/model/ner"

w1=0.9
w2=0.1
EVAL_LANGUAGES=ar
SEEDS=42,51,99

cd $PROJECT/examples/pytorch/lang-arithmetic

python run_ner.py \
--task_name $TASK_NAME \
--languages $LANGUAGES \
--model_name_or_path $MODEL_NAME \
--remove_unused_columns False \
--per_device_eval_batch_size $BATCH_SIZE \
--output_dir "$OUTPUT_DIR/results" \
--adapter_path_template $ADAPTER_PATH_TEMPLATE \
--task_adapter_path $TASK_ADAPTER_PATH \
--metric_file_prefix "test_lang_$EVAL_LANGUAGES" \
--eval_languages $EVAL_LANGUAGES \
--lang_mapping "{'ar': ('en', 'sw')}" \
--weights ${w1},${w2} \
--do_predict \
--fp16

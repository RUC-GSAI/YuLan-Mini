#!/bin/bash

#SBATCH --comment=joint_project

#SBATCH --job-name=xxxx

#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:a800:8

#SBATCH --partition=debug

#SBATCH --output=log/%x-%j/part0.log

#SBATCH --error=log/%x-%j/part0.log


source setup.sh

# ========== RESUME 只需要修改这里 ==========
last_stage_job_name=miniyulan-2B-final-phase25
STAGE=26
START_GLOBAL_STEP=243198
DECAY_STEPS=19000  # 退火steps，注意会和batch size有关
START_LAMBDA=1
END_LAMBDA=0.  # 从0.01降至0
# ========================================

CONTINUE=false
if [ "$CONTINUE" = false ]; then
    DO_RMS_NORM=true
    ALLOW_0_CHECKPOINT=false
    UPDATE_TRAINED_STEPS_AND_EPOCHS=true
elif [ "$CONTINUE" = true ]; then
    DO_RMS_NORM=false
    ALLOW_0_CHECKPOINT=true
    UPDATE_TRAINED_STEPS_AND_EPOCHS=false
fi

MASTER_PORT=36000
NNODES=4
export WANDB_MODE=offline

# dataset path
# FETCH_TIME=""  # 注意！现在FETCH_TIME自动从launch中传入！！！！所以在submit_to_slurm.sh中设置！！！！
DATA_PATH=data_path
MODEL_PATH=model_path

# model max length
MODEL_MAX_LENGTH=28672

# batch size
# 下面的BS 节点数   GPU数   CONTEXT-SIZE
PER_DEVICE_TRAIN_BATCH_SIZE=1

# gradient accumulation steps
GRADIENT_ACCUMULATION_STEPS=4

# learning rate
LEARNING_RATE=1e-2

# warmup ratio
WARMUP_RATIO=0.0
END_GLOBAL_STEP=$(expr $START_GLOBAL_STEP + $DECAY_STEPS)

# weight decay
WEIGHT_DECAY=0.1

# deepspeed config path
DEEPSPEED_CONFIG_PATH='ds2_config_adamw.json'

LOG_PREFIX=$(basename $DATA_PATH)-$(basename $MODEL_PATH)-node-$NNODES-len-$MODEL_MAX_LENGTH-lr-$LEARNING_RATE-bs-$PER_DEVICE_TRAIN_BATCH_SIZE-gacc-$GRADIENT_ACCUMULATION_STEPS
JOB_NAME=$LOG_PREFIX
LOG_DIR=log/$JOB_NAME
OUTPUT_DIR=output/${JOB_NAME}
mkdir -p ${OUTPUT_DIR}

/usr/bin/pdsh -w $comma_hostnames bash torchrun_wrapper.sh $comma_hostnames $SLURM_JOB_NAME $SLURM_JOB_ID \
    --nnodes $NNODES \
    --nproc_per_node 8 \
    --rdzv_backend static \
    --rdzv_id $JOB_ID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 3 \
    train.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --bf16 True \
    --num_train_epochs ${STAGE} \
    --model_max_length $MODEL_MAX_LENGTH \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_total_limit 25 \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --logging_steps 1 \
    --deepspeed ${DEEPSPEED_CONFIG_PATH} \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --tf32 False \
    --lr_scheduler_type "linear" \
    --flash_attention \
    --use_wsd \
    --log_dir $LOG_DIR \
    --log_level debug \
    --profile False \
    --torch_compile \
    --max_grad_norm 1 \
    --logging_dir ${LOG_DIR} \
    --ddp_timeout 3600 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --run_name $LOG_PREFIX \
    --adam_epsilon 1e-15 \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2 \
    --shrink_alpha 1 \
    --init_scale_o 1 \
    --qk_layernorm False \
    --hidden_size 1920 \
    --intermediate_size 4800 \
    --num_hidden_layers 56 \
    --num_attention_heads 30 \
    --num_key_value_heads 6 \
    --model_reproduce cerebras \
    --scale_emb 10 \
    --tie_word_embeddings True \
    --attention_bias True \
    --z_loss 0.0001 \
    --gradient_checkpointing_step 56 \
    --use_muparam_lr True \
    --initializer_range 0.00005 \
    --q_proj_alpha 0.3651483716701107 \
    --k_proj_alpha 0.3651483716701107 \
    --v_proj_alpha 0.3651483716701107 \
    --gate_up_proj_alpha 0.3651483716701107 \
    --o_proj_alpha 0.03450327796711771 \
    --down_proj_alpha 0.03450327796711771 \
    --input_layernorm_alpha 1 \
    --post_attention_layernorm_alpha 1 \
    --norm_alpha 1 \
    --lm_head_alpha 1 \
    --dim_model_base_lr 256 \
    --dim_model_base_logits 1920 \
    --vi_residual_alpha 1.4 \
    --wesar_weights True \
    --use_norm_alpha True \
    --use_emb_alpha False \
    --add_rms_norm $DO_RMS_NORM \
    --update_trained_steps_and_epochs $UPDATE_TRAINED_STEPS_AND_EPOCHS \
    --start_lambda $START_LAMBDA \
    --end_lambda $END_LAMBDA \
    --start_global_step $START_GLOBAL_STEP \
    --end_global_step $END_GLOBAL_STEP \
    --wsd_style 1sqrt \
    --resume_from_checkpoint $MODEL_PATH \

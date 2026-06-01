#!/bin/bash
set -x

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --enable_soft_think)
            ENABLE_SOFT_THINK="$2"
            shift 2
            ;;
        --enable_mixed_rollout)
            ENABLE_MIXED_ROLLOUT="$2"
            shift 2
            ;;
        --train_batch_size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --ppo_mini_batch_size)
            PPO_MINI_BATCH_SIZE="$2"
            shift 2
            ;;
        --train_file)
            TRAIN_FILE="$2"
            shift 2
            ;;
        --val_file)
            VAL_FILE="$2"
            shift 2
            ;;
        --max_prompt_length)
            MAX_PROMPT_LENGTH="$2"
            shift 2
            ;;
        --max_token_len_per_gpu)
            MAX_TOKEN_LEN_PER_GPU="$2"
            shift 2
            ;;
        --multiplex_width)
            MULTIPLEX_WIDTH="$2"
            shift 2
            ;;
        --loss_mode)
            LOSS_MODE="$2"
            shift 2
            ;;
        --shared4_advantage)
            SHARED4_ADVANTAGE="$2"
            shift 2
            ;;
        --shared4_thinking_rollouts)
            SHARED4_THINKING_ROLLOUTS="$2"
            shift 2
            ;;
        --shared4_answers_per_trace)
            SHARED4_ANSWERS_PER_TRACE="$2"
            shift 2
            ;;
        --branch_rollout)
            BRANCH_ROLLOUT="$2"
            shift 2
            ;;
        --branch_rollout_thinking_traces)
            BRANCH_ROLLOUT_THINKING_TRACES="$2"
            shift 2
            ;;
        --branch_rollout_answers_per_trace)
            BRANCH_ROLLOUT_ANSWERS_PER_TRACE="$2"
            shift 2
            ;;
        --branch_rollout_thinking_tokens)
            BRANCH_ROLLOUT_THINKING_TOKENS="$2"
            shift 2
            ;;
        --branch_rollout_continuation_tokens)
            BRANCH_ROLLOUT_CONTINUATION_TOKENS="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --temp)
            TEMP="$2"
            shift 2
            ;;
        --early_stopping_entropy_threshold)
            EARLY_STOPPING_ENTROPY_THRESHOLD="$2"
            shift 2
            ;;
        --early_stopping_length_threshold)
            EARLY_STOPPING_LENGTH_THRESHOLD="$2"
            shift 2
            ;;
        --enable_entropy_mask)
            ENABLE_ENTROPY_MASK="$2"
            shift 2
            ;;
        --entropy_mask_threshold)
            ENTROPY_MASK_THRESHOLD="$2"
            shift 2
            ;;
        --gpu_mem_util)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --enable_gumbel)
            ENABLE_GUMBEL="$2"
            shift 2
            ;;
        --gumbel_tau)
            GUMBEL_TAU="$2"
            shift 2
            ;;
        --n_gpus_per_node)
            N_GPUS_PER_NODE="$2"
            shift 2
            ;;
        --after_thinking_temperature)
            AFTER_THINKING_TEMPERATURE="$2"
            shift 2
            ;;
        --after_thinking_top_k)
            AFTER_THINKING_TOP_K="$2"
            shift 2
            ;;
        --after_thinking_min_p)
            AFTER_THINKING_MIN_P="$2"
            shift 2
            ;;
        --enable_replacement)
            ENABLE_REPLACEMENT="$2"
            shift 2
            ;;
        --enable_gumbel_after_thinking)
            ENABLE_GUMBEL_AFTER_THINKING="$2"
            shift 2
            ;;
        --after_thinking_top_p)
            AFTER_THINKING_TOP_P="$2"
            shift 2
            ;;
        --val_before_train)
            VAL_BEFORE_TRAIN="$2"
            shift 2
            ;;
        --total_training_steps)
            TOTAL_TRAINING_STEPS="$2"
            shift 2
            ;;
        --stop_at_step)
            STOP_AT_STEP="$2"
            shift 2
            ;;
        --resume_from_path)
            RESUME_FROM_PATH="$2"
            shift 2
            ;;
        --resume_mode)
            RESUME_MODE="$2"
            shift 2
            ;;
        --max_response_length)
            MAX_RESPONSE_LENGTH="$2"
            shift 2
            ;;
        --save_freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --test_freq)
            TEST_FREQ="$2"
            shift 2
            ;;
        --val_rollout_n)
            VAL_ROLLOUT_N="$2"
            shift 2
            ;;
        --rollout_n)
            ROLLOUT_N="$2"
            shift 2
            ;;
        --rollout_max_num_seqs)
            ROLLOUT_MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --rollout_name)
            ROLLOUT_NAME="$2"
            shift 2
            ;;
        --val_dataset)
            VAL_DATASET="$2"
            shift 2
            ;;
        --val_batch_size)
            VAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --default_local_dir)
            DEFAULT_LOCAL_DIR="$2"
            shift 2
            ;;
        --logger)
            LOGGER="$2"
            shift 2
            ;;
        --enforce_eager)
            ENFORCE_EAGER="$2"
            shift 2
            ;;
        --attn_implementation)
            ATTN_IMPLEMENTATION="$2"
            shift 2
            ;;
        --save_hf_model)
            SAVE_HF_MODEL="$2"
            shift 2
            ;;
        --checkpoint_save_contents)
            ACTOR_CHECKPOINT_SAVE_CONTENTS="$2"
            shift 2
            ;;
        --checkpoint_load_contents)
            ACTOR_CHECKPOINT_LOAD_CONTENTS="$2"
            shift 2
            ;;
        --dry_run)
            if [[ $# -gt 1 && "$2" != --* ]]; then
                DRY_RUN="$2"
                shift 2
            else
                DRY_RUN=True
                shift
            fi
            ;;
        --enable_unweighting)
            ENABLE_UNWEIGHTING="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default values if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="DeepSeek/DeepSeek-R1-Distill-Qwen-32B"
fi

if [ -z "$EXP_NAME" ]; then
    EXP_NAME="multiplex-thinking"
fi

if [ -z "$ENABLE_SOFT_THINK" ]; then
    ENABLE_SOFT_THINK=True
fi 

if [ -z "$ENABLE_MIXED_ROLLOUT" ]; then
    ENABLE_MIXED_ROLLOUT=False
fi

if [ -z "$TRAIN_BATCH_SIZE" ]; then
    TRAIN_BATCH_SIZE=256
fi
if [ -z "$PPO_MINI_BATCH_SIZE" ]; then
    PPO_MINI_BATCH_SIZE=$TRAIN_BATCH_SIZE
fi

if [ -z "$TRAIN_FILE" ]; then
    TRAIN_FILE=deepscaler/hdfs_data/train.parquet
fi

if [ -z "$MAX_PROMPT_LENGTH" ]; then
    MAX_PROMPT_LENGTH=1024
fi

if [ -z "$MAX_TOKEN_LEN_PER_GPU" ]; then
    MAX_TOKEN_LEN_PER_GPU=65536
fi

if [ -z "$MULTIPLEX_WIDTH" ]; then
    MULTIPLEX_WIDTH=3
fi

if [ -z "$LOSS_MODE" ]; then
    LOSS_MODE="vanilla"
fi 
if [ -z "$SHARED4_ADVANTAGE" ]; then
    SHARED4_ADVANTAGE=False
fi
if [ -z "$SHARED4_THINKING_ROLLOUTS" ]; then
    SHARED4_THINKING_ROLLOUTS=4
fi
if [ -z "$SHARED4_ANSWERS_PER_TRACE" ]; then
    SHARED4_ANSWERS_PER_TRACE=4
fi

if [ -z "$TOP_P" ]; then
    TOP_P=1.0
fi

if [ -z "$TEMP" ]; then
    TEMP=1.0
fi

if [ -z "$EARLY_STOPPING_ENTROPY_THRESHOLD" ]; then
    EARLY_STOPPING_ENTROPY_THRESHOLD=-1.0
fi

if [ -z "$EARLY_STOPPING_LENGTH_THRESHOLD" ]; then
    EARLY_STOPPING_LENGTH_THRESHOLD=256
fi

if [ -z "$ENABLE_ENTROPY_MASK" ]; then
    ENABLE_ENTROPY_MASK=False
fi

if [ -z "$ENTROPY_MASK_THRESHOLD" ]; then
    ENTROPY_MASK_THRESHOLD=0.0
fi

if [ -z "$GPU_MEMORY_UTILIZATION" ]; then
    GPU_MEMORY_UTILIZATION=0.9
fi



if [ -z "$ENABLE_GUMBEL" ]; then
    ENABLE_GUMBEL=False
fi
if [ -z "$ROLLOUT_ENABLE_GUMBEL" ]; then
    ROLLOUT_ENABLE_GUMBEL=$ENABLE_GUMBEL
fi
if [ -z "$GUMBEL_TAU" ]; then
    GUMBEL_TAU=1.0
fi

if [ -z "$N_GPUS_PER_NODE" ]; then
    N_GPUS_PER_NODE=8
fi

if [ -z "$AFTER_THINKING_TEMPERATURE" ]; then
    AFTER_THINKING_TEMPERATURE=1.0
fi

if [ -z "$ENABLE_REPLACEMENT" ]; then
    ENABLE_REPLACEMENT=True
fi

if [ -z "$ENABLE_GUMBEL_AFTER_THINKING" ]; then
    ENABLE_GUMBEL_AFTER_THINKING=False
fi

if [ -z "$VAL_BEFORE_TRAIN" ]; then
    VAL_BEFORE_TRAIN=True
fi

if [ -z "$TOTAL_TRAINING_STEPS" ]; then
    TOTAL_TRAINING_STEPS=300
fi

if [ -z "$STOP_AT_STEP" ]; then
    STOP_AT_STEP=$TOTAL_TRAINING_STEPS
fi

if [ -z "$RESUME_FROM_PATH" ]; then
    RESUME_FROM_PATH=null
fi

if [ -z "$RESUME_MODE" ]; then
    RESUME_MODE="auto"
fi

if [ -z "$MAX_RESPONSE_LENGTH" ]; then
    MAX_RESPONSE_LENGTH=8192
fi

if [ -z "$SAVE_FREQ" ]; then
    SAVE_FREQ=25
fi

if [ -z "$TEST_FREQ" ]; then
    TEST_FREQ=25
fi

if [ -z "$VAL_ROLLOUT_N" ]; then
    VAL_ROLLOUT_N=4
fi

if [ -z "$ROLLOUT_N" ]; then
    ROLLOUT_N=8
fi

if [ -z "$ROLLOUT_MAX_NUM_SEQS" ]; then
    ROLLOUT_MAX_NUM_SEQS=512
fi
if [ -z "$ROLLOUT_NAME" ]; then
    ROLLOUT_NAME="${PYTORCH_SOFT_ROLLOUT_NAME:-sglang}"
fi

if [ -z "$VAL_DATASET" ]; then
    VAL_DATASET=aime
fi

if [ -z "$VAL_FILE" ]; then
    VAL_FILE=deepscaler/hdfs_data/$VAL_DATASET.parquet
fi

if [ -z "$VAL_BATCH_SIZE" ]; then
    VAL_BATCH_SIZE=512
fi

if [ -z "$WANDB_PROJECT" ]; then
    WANDB_PROJECT=MultiplexThinning
fi

if [ -z "$DEFAULT_LOCAL_DIR" ]; then
    DEFAULT_LOCAL_DIR=./${WANDB_PROJECT}/${EXP_NAME}
fi

if [ -z "$LOGGER" ]; then
    LOGGER="['console','wandb']"
fi

if [ -z "$ENFORCE_EAGER" ]; then
    ENFORCE_EAGER=True
fi

if [ -z "$ATTN_IMPLEMENTATION" ]; then
    ATTN_IMPLEMENTATION=flash_attention_2
fi

if [ -z "$SAVE_HF_MODEL" ]; then
    SAVE_HF_MODEL=False
fi

if [ -z "$ACTOR_CHECKPOINT_SAVE_CONTENTS" ]; then
    if [[ "$SAVE_HF_MODEL" == "1" || "$SAVE_HF_MODEL" == "true" || "$SAVE_HF_MODEL" == "TRUE" || "$SAVE_HF_MODEL" == "True" || "$SAVE_HF_MODEL" == "yes" || "$SAVE_HF_MODEL" == "on" ]]; then
        ACTOR_CHECKPOINT_SAVE_CONTENTS="['model','hf_model','optimizer','extra']"
    else
        ACTOR_CHECKPOINT_SAVE_CONTENTS="['model','optimizer','extra']"
    fi
fi
if [ -z "$ACTOR_CHECKPOINT_LOAD_CONTENTS" ]; then
    ACTOR_CHECKPOINT_LOAD_CONTENTS="['model','optimizer','extra']"
fi

if [ -z "$AFTER_THINKING_TOP_P" ]; then
    AFTER_THINKING_TOP_P=1.0
fi

if [ -z "$AFTER_THINKING_TOP_K" ]; then
    AFTER_THINKING_TOP_K=-1
fi

if [ -z "$AFTER_THINKING_MIN_P" ]; then
    AFTER_THINKING_MIN_P=0.0
fi

if [ -z "$ENABLE_UNWEIGHTING" ]; then
    ENABLE_UNWEIGHTING=False
fi

if [ -z "$BRANCH_ROLLOUT" ]; then
    BRANCH_ROLLOUT=False
fi

if [ -z "$BRANCH_ROLLOUT_THINKING_TRACES" ]; then
    BRANCH_ROLLOUT_THINKING_TRACES=4
fi

if [ -z "$BRANCH_ROLLOUT_ANSWERS_PER_TRACE" ]; then
    BRANCH_ROLLOUT_ANSWERS_PER_TRACE=4
fi

if [ -z "$BRANCH_ROLLOUT_THINKING_TOKENS" ]; then
    BRANCH_ROLLOUT_THINKING_TOKENS=$EARLY_STOPPING_LENGTH_THRESHOLD
fi

if [ -z "$BRANCH_ROLLOUT_CONTINUATION_TOKENS" ]; then
    BRANCH_ROLLOUT_CONTINUATION_TOKENS=0
fi

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" || "$DRY_RUN" == "True" || "$DRY_RUN" == "yes" || "$DRY_RUN" == "on" ]]; then
    echo "[dry_run] train.sh parsed arguments successfully"
    echo "[dry_run] model=${MODEL_PATH}"
    echo "[dry_run] exp_name=${EXP_NAME}"
    echo "[dry_run] train_file=${TRAIN_FILE}"
    echo "[dry_run] val_file=${VAL_FILE}"
    echo "[dry_run] train_batch_size=${TRAIN_BATCH_SIZE}"
    echo "[dry_run] ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    echo "[dry_run] rollout_n=${ROLLOUT_N}"
    echo "[dry_run] rollout_name=${ROLLOUT_NAME}"
    echo "[dry_run] shared4_advantage=${SHARED4_ADVANTAGE}"
    echo "[dry_run] shared4_thinking_rollouts=${SHARED4_THINKING_ROLLOUTS}"
    echo "[dry_run] shared4_answers_per_trace=${SHARED4_ANSWERS_PER_TRACE}"
    echo "[dry_run] branch_rollout=${BRANCH_ROLLOUT}"
    echo "[dry_run] branch_rollout_thinking_traces=${BRANCH_ROLLOUT_THINKING_TRACES}"
    echo "[dry_run] branch_rollout_answers_per_trace=${BRANCH_ROLLOUT_ANSWERS_PER_TRACE}"
    echo "[dry_run] branch_rollout_thinking_tokens=${BRANCH_ROLLOUT_THINKING_TOKENS}"
    echo "[dry_run] branch_rollout_continuation_tokens=${BRANCH_ROLLOUT_CONTINUATION_TOKENS}"
    echo "[dry_run] rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
    echo "[dry_run] max_prompt_length=${MAX_PROMPT_LENGTH}"
    echo "[dry_run] max_response_length=${MAX_RESPONSE_LENGTH}"
    echo "[dry_run] max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
    echo "[dry_run] val_before_train=${VAL_BEFORE_TRAIN}"
    echo "[dry_run] total_training_steps=${TOTAL_TRAINING_STEPS}"
    echo "[dry_run] stop_at_step=${STOP_AT_STEP}"
    echo "[dry_run] test_freq=${TEST_FREQ}"
    echo "[dry_run] save_freq=${SAVE_FREQ}"
    echo "[dry_run] logger=${LOGGER}"
    echo "[dry_run] default_local_dir=${DEFAULT_LOCAL_DIR}"
    echo "[dry_run] enforce_eager=${ENFORCE_EAGER}"
    echo "[dry_run] attn_implementation=${ATTN_IMPLEMENTATION}"
    echo "[dry_run] actor_checkpoint_save_contents=${ACTOR_CHECKPOINT_SAVE_CONTENTS}"
    echo "[dry_run] actor_checkpoint_load_contents=${ACTOR_CHECKPOINT_LOAD_CONTENTS}"
    exit 0
fi

export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$WANDB_API_KEY" ]; then
    echo "[WARN] WANDB_API_KEY is empty. Set it via env var if you want wandb logging."
fi
if [ -z "$HF_TOKEN" ]; then
    echo "[WARN] HF_TOKEN is empty. Set it via env var if you need to access gated/private HuggingFace models."
fi
export NCCL_TIMEOUT=36000
if [ -n "${NCCL_SOCKET_IFNAME:-}" ]; then
    export NCCL_SOCKET_IFNAME
    echo "[net] using NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
else
    unset NCCL_SOCKET_IFNAME
    echo "[net] NCCL_SOCKET_IFNAME unset; letting NCCL auto-detect"
fi
if [ -n "${NCCL_IB_HCA:-}" ]; then
    export NCCL_IB_HCA
    echo "[net] using NCCL_IB_HCA=${NCCL_IB_HCA}"
else
    unset NCCL_IB_HCA
fi
############## ray_node_setup.sh ##############
echo ${MASTER_ADDR}
echo $OMPI_COMM_WORLD_RANK


export NCCL_TIMEOUT=72000
# Keep async CUDA execution for throughput; set CUDA_LAUNCH_BLOCKING=1 only for debugging.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DASHBOARD=1
export SGL_ENABLE_JIT_DEEPGEMM="${SGL_ENABLE_JIT_DEEPGEMM:-0}"
if [ -z "$SGLANG_PORT_BASE" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    export SGLANG_PORT_BASE=$((40000 + (SLURM_JOB_ID % 20000)))
fi
echo "[sglang] port_base=${SGLANG_PORT_BASE:-30000}"

export FORCE_THINK_END_AT_LENGTH="${FORCE_THINK_END_AT_LENGTH:-1}"
echo "[sglang] force_think_end_at_length=${FORCE_THINK_END_AT_LENGTH}"
# Avoid multi-process GPU contention
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VERL_DISABLE_EXPANDABLE_SEGMENTS=1


unset RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Ray's default GPU isolation rewrites CUDA_VISIBLE_DEVICES per actor.  SGLang
# then gathers those per-actor values and can hand invalid ordinals to workers
# on Slurm allocations with sparse physical GPU ids.  Keep every Ray actor on
# the same normalized view when requested by the submitter.
if [[ "${NORMALIZE_CUDA_VISIBLE_DEVICES:-False}" =~ ^(1|true|TRUE|True|yes|YES|on|ON)$ ]]; then
    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    if [[ "${N_GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
        CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((N_GPUS_PER_NODE - 1))")"
        export CUDA_VISIBLE_DEVICES
    fi
    echo "[gpu] normalized_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "[gpu] ray_experimental_noset_cuda_visible_devices=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES}"
fi

# Provide default values for local execution if not set by a job scheduler
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export OMPI_COMM_WORLD_RANK=${OMPI_COMM_WORLD_RANK:-0}

unset RAY_ADDRESS

# VERL's trainer starts its own local Ray runtime via ray.init for single-node
# jobs. Pre-starting a shell-level Ray head here creates a second GCS/raylet on
# the same allocation and can collide when Slurm co-locates jobs on a node.
if [[ "${PRESTART_RAY:-False}" =~ ^(1|true|TRUE|True|yes|YES|on|ON)$ ]]; then
    RAY_HEAD_PORT="${RAY_HEAD_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 30000)))}"
    export RAY_HEAD_PORT
    if [ "$OMPI_COMM_WORLD_RANK" -eq 0 ]; then
        RAY_HEAD_ARGS=(--head --node-ip-address "${MASTER_ADDR}" --port "$RAY_HEAD_PORT" --num-gpus "$N_GPUS_PER_NODE" --include-dashboard=false --disable-usage-stats)
        if [ -n "${RAY_TMPDIR:-}" ]; then
            mkdir -p "$RAY_TMPDIR"
            RAY_HEAD_ARGS+=(--temp-dir "$RAY_TMPDIR")
        fi
        ray start "${RAY_HEAD_ARGS[@]}"
    else
        echo ${MASTER_ADDR}
        ray start --address ${MASTER_ADDR}:${RAY_HEAD_PORT} --num-gpus $N_GPUS_PER_NODE
    fi
    sleep "${RAY_START_SLEEP_SECONDS:-5}"
else
    echo "[ray] skipping shell ray prestart; verl.trainer.main_ppo will call ray.init"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    "data.train_files=$TRAIN_FILE" \
    "data.val_files=$VAL_FILE" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.dataloader_num_workers=2 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.return_raw_chat=True \
    data.truncation=right \
    +data.use_online_transform=False \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    ++actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.policy_loss.loss_mode=$LOSS_MODE \
    +algorithm.shared4_advantage=$SHARED4_ADVANTAGE \
    +algorithm.shared4_thinking_rollouts=$SHARED4_THINKING_ROLLOUTS \
    +algorithm.shared4_answers_per_trace=$SHARED4_ANSWERS_PER_TRACE \
    +actor_rollout_ref.rollout.branch_rollout=$BRANCH_ROLLOUT \
    +actor_rollout_ref.rollout.branch_rollout_thinking_traces=$BRANCH_ROLLOUT_THINKING_TRACES \
    +actor_rollout_ref.rollout.branch_rollout_answers_per_trace=$BRANCH_ROLLOUT_ANSWERS_PER_TRACE \
    +actor_rollout_ref.rollout.branch_rollout_thinking_tokens=$BRANCH_ROLLOUT_THINKING_TOKENS \
    +actor_rollout_ref.rollout.branch_rollout_continuation_tokens=$BRANCH_ROLLOUT_CONTINUATION_TOKENS \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.temperature=$TEMP \
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMP \
    actor_rollout_ref.rollout.enable_soft_thinking=$ENABLE_SOFT_THINK \
    actor_rollout_ref.rollout.enable_mixed_rollout=$ENABLE_MIXED_ROLLOUT \
    actor_rollout_ref.rollout.after_thinking_top_p=$AFTER_THINKING_TOP_P \
    actor_rollout_ref.rollout.after_thinking_top_k=$AFTER_THINKING_TOP_K \
    actor_rollout_ref.rollout.after_thinking_min_p=$AFTER_THINKING_MIN_P \
    actor_rollout_ref.rollout.max_topk=$MULTIPLEX_WIDTH \
    actor_rollout_ref.rollout.used_topk=$MULTIPLEX_WIDTH \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.early_stopping_entropy_threshold=$EARLY_STOPPING_ENTROPY_THRESHOLD \
    actor_rollout_ref.rollout.early_stopping_length_threshold=$EARLY_STOPPING_LENGTH_THRESHOLD \
    actor_rollout_ref.rollout.enable_entropy_mask=$ENABLE_ENTROPY_MASK \
    actor_rollout_ref.rollout.entropy_mask_threshold=$ENTROPY_MASK_THRESHOLD \
    actor_rollout_ref.rollout.enable_gumbel=$ENABLE_GUMBEL \
    actor_rollout_ref.rollout.gumbel_tau=$GUMBEL_TAU \
    actor_rollout_ref.rollout.after_thinking_temperature=$AFTER_THINKING_TEMPERATURE \
    actor_rollout_ref.rollout.enable_replacement=$ENABLE_REPLACEMENT \
    actor_rollout_ref.rollout.enable_gumbel_after_thinking=$ENABLE_GUMBEL_AFTER_THINKING \
    actor_rollout_ref.rollout.enable_unweighting=$ENABLE_UNWEIGHTING \
    actor_rollout_ref.rollout.val_kwargs.enable_replacement=$ENABLE_REPLACEMENT \
    actor_rollout_ref.rollout.val_kwargs.enable_gumbel_after_thinking=$ENABLE_GUMBEL_AFTER_THINKING \
    actor_rollout_ref.rollout.val_kwargs.enable_unweighting=$ENABLE_UNWEIGHTING \
    actor_rollout_ref.rollout.val_kwargs.enable_gumbel=$ENABLE_GUMBEL \
    actor_rollout_ref.rollout.val_kwargs.gumbel_tau=$GUMBEL_TAU \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_temperature=$AFTER_THINKING_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.early_stopping_entropy_threshold=$EARLY_STOPPING_ENTROPY_THRESHOLD \
    actor_rollout_ref.rollout.val_kwargs.early_stopping_length_threshold=$EARLY_STOPPING_LENGTH_THRESHOLD \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_top_p=$AFTER_THINKING_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_top_k=$AFTER_THINKING_TOP_K \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_min_p=$AFTER_THINKING_MIN_P \
    actor_rollout_ref.rollout.val_kwargs.max_topk=$MULTIPLEX_WIDTH \
    actor_rollout_ref.rollout.val_kwargs.used_topk=$MULTIPLEX_WIDTH \
    actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=False \
    "actor_rollout_ref.actor.checkpoint.save_contents=$ACTOR_CHECKPOINT_SAVE_CONTENTS" \
    "actor_rollout_ref.actor.checkpoint.load_contents=$ACTOR_CHECKPOINT_LOAD_CONTENTS" \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    "trainer.logger=$LOGGER" \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    +trainer.stop_at_step=$STOP_AT_STEP \
    trainer.resume_mode=$RESUME_MODE \
    trainer.resume_from_path=$RESUME_FROM_PATH \
    reward_model.reward_manager=hf_math_verify \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE:-True} \
    actor_rollout_ref.rollout.enable_sleep_hack=${ENABLE_SLEEP_HACK:-True} \
    +actor_rollout_ref.rollout.enable_memory_saver=${ENABLE_MEMORY_SAVER:-True} \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    ++actor_rollout_ref.rollout.shuffle_before_dispatch=False \
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=${SGLANG_DISABLE_OVERLAP_SCHEDULE:-True} \
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.sampling_backend=${SGLANG_SAMPLING_BACKEND:-flashinfer} \
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.grammar_backend=${SGLANG_GRAMMAR_BACKEND:-none} \
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.watchdog_timeout=${SGLANG_WATCHDOG_TIMEOUT:-1800} \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    reward_model.enable=False \
    "trainer.default_local_dir=$DEFAULT_LOCAL_DIR"

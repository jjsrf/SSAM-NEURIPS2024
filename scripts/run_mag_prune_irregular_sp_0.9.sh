# magnitude-based 1 shot retraining
ARCH=${1:-"resnet"} # 
DEPTH=${2:-"32"}
DATASET=${3:-"cifar10"}
SP=${4:-"0.9"}
MUTATE_TYPE=${5:-"with"} # [with, without, no_mutate]

SPARSITY_TYPE=${6:-"irregular"}
CONFIG_FILE=${7:-"./profiles/resnet32_LTH/irregular/resnet_${SP}_${MUTATE_TYPE}.yaml"}
PRUNE_ARGS=${8:-"--sp-retrain --sp-prune-before-retrain"}
LOAD_CKPT=${9:-"XXXXX.pth.tar"}
SAVE_FOLDER=${10:-"checkpoints/${DATASET}/resnet32/random_init_mag_prune/ep160/irregular_sp_${SP}_${MUTATE_TYPE}/"}
INIT_LR=${11:-"0.1"}
GLOBAL_BATCH_SIZE=${12:-"64"}
EPOCHS=${13:-"160"}
WARMUP=${14:-""}
REMARK=${15:-"run_1"}





cd ..


SEED=${16:-"914"}
CUDA_VISIBLE_DEVICES=0 python3 -u main_prune_train.py --arch ${ARCH} --depth ${DEPTH} --dataset ${DATASET} --optmzr sgd --batch-size ${GLOBAL_BATCH_SIZE} --lr ${INIT_LR} --lr-scheduler cosine --save-model ${SAVE_FOLDER} --epochs ${EPOCHS} --seed ${SEED} --remark ${REMARK} ${PRUNE_ARGS} --sp-admm-sparsity-type=${SPARSITY_TYPE} --sp-config-file=${CONFIG_FILE} --log-filename=${SAVE_FOLDER}/log.txt 





MUTATE_TYPE=${5:-"without"} # [with, without, no_mutate]
CONFIG_FILE=${7:-"./profiles/resnet32_LTH/irregular/resnet_${SP}_${MUTATE_TYPE}.yaml"}
SAVE_FOLDER=${10:-"checkpoints/${DATASET}/resnet32/random_init_mag_prune/ep160/irregular_sp_${SP}_${MUTATE_TYPE}/"}

SEED=${16:-"914"}
CUDA_VISIBLE_DEVICES=0 python3 -u main_prune_train.py --arch ${ARCH} --depth ${DEPTH} --dataset ${DATASET} --optmzr sgd --batch-size ${GLOBAL_BATCH_SIZE} --lr ${INIT_LR} --lr-scheduler cosine --save-model ${SAVE_FOLDER} --epochs ${EPOCHS} --seed ${SEED} --remark ${REMARK} ${PRUNE_ARGS} --sp-admm-sparsity-type=${SPARSITY_TYPE} --sp-config-file=${CONFIG_FILE} --log-filename=${SAVE_FOLDER}/log.txt 



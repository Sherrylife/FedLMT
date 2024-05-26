#-------------------------------------distributed setting-----------------------------------#
# FedAvg
#python3 main.py --dataset_name cifar10 --model_name resnet18 --control_name 1_100_0.1_noniid2-0.5_fix_a1-b1-c1_bn_0_0 --algo_name FedAvg --optimizer_name SGD --scheduler_name CosineAnnealingLR --g_rounds 2000 --l_epoch 1 --lr 1e-1 --seed 31 --B 64 --devices 4 5 --PROCESS_NUM 10
#python3 main.py --dataset_name cifar100 --model_name resnet18 --control_name 1_100_0.1_noniid2-0.5_fix_a1-b1-c1_bn_0_0 --algo_name FedAvg --optimizer_name SGD --scheduler_name CosineAnnealingLR --g_rounds 3000 --l_epoch 1 --lr 1e-1 --seed 31 --B 64 --devices 4 5 --PROCESS_NUM 10
#python3 main.py --dataset_name svhn --model_name resnet18 --control_name 1_100_0.1_noniid2-0.5_fix_a1-b1-c1_bn_0_0 --algo_name FedAvg --optimizer_name SGD --scheduler_name CosineAnnealingLR --g_rounds 2000 --l_epoch 1 --lr 1e-1 --seed 31 --B 64 --num_experiments 1 --devices 4 5 --PROCESS_NUM 10
#python3 main.py --dataset_name tinyImagenet --model_name resnet18 --control_name 1_100_0.1_noniid2-0.5_fix_a1-b1-c1_bn_0_0 --algo_name FedAvg --optimizer_name SGD --scheduler_name CosineAnnealingLR --g_rounds 3000 --l_epoch 1 --lr 1e-1 --seed 31 --B 64 --devices 0 1 --PROCESS_NUM 10 &
#python3 main.py --dataset_name wikitext2 --model_name transformer --algo_name FedAvg --optimizer_name SGD --scheduler_name CosineAnnealingLR --g_rounds 200 --l_epoch 1 --lr 1e-2 --seed 31 --B 100 --devices 4 5 --PROCESS_NUM 5 --control_name 1_100_0.05_iid_fix_a1-b1-c1-d1_bn_0_0


# FedLMT
#python3 main.py --dataset_name cifar10 --model_name resnet18 --control_name 1_100_0.1_noniid2-0.5_fix_a1-b1-c1_bn_0_0 --algo_name FedLMT --ratio_LR 0.2 --decom_rule 0 0 --optimizer_name SGD --scheduler_name CosineAnnealingLR --g_rounds 2000 --l_epoch 1 --lr 1e-1 --seed 31 --B 64 --num_experiments 1 --devices 0 1 --PROCESS_NUM 10
#python3 main.py --dataset_name cifar100 --model_name resnet18 --control_name 1_100_0.1_noniid2-0.5_fix_a1-b1-c1_bn_0_0 --algo_name FedLMT --ratio_LR 0.2 --decom_rule 2 0 --optimizer_name SGD --scheduler_name CosineAnnealingLR --g_rounds 3000 --l_epoch 1 --lr 1e-1 --seed 31 --B 64 --devices 0 6 --PROCESS_NUM 10
#python3 main.py --dataset_name svhn --model_name resnet18 --control_name 1_100_0.1_noniid2-0.5_fix_a1-b1-c1_bn_0_0 --algo_name FedLMT --ratio_LR 0.15 --decom_rule 0 1 --optimizer_name SGD --scheduler_name CosineAnnealingLR --train_decay none --g_rounds 2000 --l_epoch 1 --lr 1e-1 --seed 31 --B 64 --devices 0 6 --PROCESS_NUM 10
#python3 main.py --dataset_name tinyImagenet --model_name resnet18 --control_name 1_100_0.1_noniid2-0.5_fix_a1-b1-c1_bn_0_0 --algo_name FedLMT --ratio_LR 0.15 --decom_rule 0 1 --optimizer_name SGD --scheduler_name CosineAnnealingLR --train_decay frobenius --coef_decay 1e-3 --g_rounds 3000 --l_epoch 1 --lr 1e-1 --seed 31 --B 64 --devices 2 3 --PROCESS_NUM 10
#python3 main.py --dataset_name wikitext2 --model_name hyper_transformer --algo_name FedAvg --optimizer_name SGD --scheduler_name CosineAnnealingLR --g_rounds 200 --l_epoch 1 --lr 1e-2 --seed 31 --B 100 --num_experiments 1 --devices 2 4 6 --PROCESS_NUM 5 --control_name 1_100_0.05_iid_fix_a1-b1-c1-d1_bn_0_0

# pFedLMT
#python3 main.py --dataset_name cifar10 --model_name resnet18 --control_name 1_100_0.1_noniid2-0.5_fix_a1-b1-c1_bn_0_0 --algo_name pFedLMT --decom_rule 2 1 --train_decay frobenius --coef_decay 1e-6 --optimizer_name SGD --scheduler_name CosineAnnealingLR --meta_round 0 --g_rounds 2000 --l_epoch 1 --lr 1e-1 --seed 31 --B 64 --devices 4 5 --PROCESS_NUM 10

#----------------------------------------centralized setting---------------------------------#
python3 centralized_training.py --dataset cifar10 --algo FedLMT --model resnet18 --epochs 200 --num_workers 4 --devices 0 --decom_rule 0 0 --LR_ratio 0.05 --init none --train_decay none --coef_decay 1e-2 &
python3 centralized_training.py --dataset cifar10 --algo FedLMT --model resnet18 --epochs 200 --num_workers 4 --devices 5 --decom_rule 0 0 --LR_ratio 0.05 --init none --train_decay frobenius --coef_decay 1e-4

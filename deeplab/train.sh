#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export MXNET_CPU_WORKER_NTHREADS=15
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export CUDA_VISIBLE_DEVICES='4,5,6,7'
python -u fine-tunec.py --pretrained-model 'model/1k/resnet-152' --pretrained-epoch 0 --model-prefix 'model/b1' --num-classes 80 --lr 0.01 --lr-step-epochs '15,25' --num-epochs 20 --lr-factor 0.1 --gpus 0,1,2,3 --batch-size 144 --size1 112 --image-shape '3,224,224' --no-checkpoint --scale2 0.1
python -u fine-tunec.py --pretrained-model 'model/1k/resnet-152' --pretrained-epoch 0 --model-prefix 'model/b1' --num-classes 80 --lr 0.01 --lr-step-epochs '15,25' --num-epochs 20 --lr-factor 0.1 --gpus 0,1,2,3 --batch-size 144 --size1 112 --image-shape '3,224,224' --no-checkpoint --scale2 0.5
python -u fine-tunec.py --pretrained-model 'model/1k/resnet-152' --pretrained-epoch 0 --model-prefix 'model/b1' --num-classes 80 --lr 0.01 --lr-step-epochs '15,25' --num-epochs 20 --lr-factor 0.1 --gpus 0,1,2,3 --batch-size 144 --size1 112 --image-shape '3,224,224' --no-checkpoint --scale2 1.0
python -u fine-tunec.py --pretrained-model 'model/1k/resnet-152' --pretrained-epoch 0 --model-prefix 'model/b1' --num-classes 80 --lr 0.01 --lr-step-epochs '15,25' --num-epochs 20 --lr-factor 0.1 --gpus 0,1,2,3 --batch-size 144 --size1 72 --image-shape '3,224,224' --no-checkpoint --scale2 0.1
python -u fine-tunec.py --pretrained-model 'model/1k/resnet-152' --pretrained-epoch 0 --model-prefix 'model/b1' --num-classes 80 --lr 0.01 --lr-step-epochs '15,25' --num-epochs 20 --lr-factor 0.1 --gpus 0,1,2,3 --batch-size 144 --size1 72 --image-shape '3,224,224' --no-checkpoint --scale2 0.5
python -u fine-tunec.py --pretrained-model 'model/1k/resnet-152' --pretrained-epoch 0 --model-prefix 'model/b1' --num-classes 80 --lr 0.01 --lr-step-epochs '15,25' --num-epochs 20 --lr-factor 0.1 --gpus 0,1,2,3 --batch-size 144 --size1 72 --image-shape '3,224,224' --no-checkpoint --scale2 1.0
#python -u fine-tune.py --pretrained-model 'model/resnet-152' --pretrained-epoch 0 --model-prefix 'model/b1' --num-classes 80 --lr 0.01 --lr-step-epochs '15,25' --num-epochs 100 --lr-factor 0.1 --gpus 4,5,6,7 --batch-size 128 --image-shape '3,224,224'
#python -u train_inception.py --model-prefix 'model/inception1' --num-classes 365 --lr 0.045 --num-epochs 100 --batch-size 100
#python -u train_inception.py --retrain --model-prefix 'model/inception1' --load-epoch 100 --num-classes 365 --lr 0.0021 --num-epochs 200 --batch-size 100
#python -u train_inception.py --model-prefix 'model/inception2' --num-classes 365 --lr 0.045 --num-epochs 200 --batch-size 100
exit 0
#sleep 5
#python -u fine-tune.py --retrain --pretrained-model 'model/a1' --pretrained-epoch 13 --model-prefix 'model/a1_2' --num-classes 80 --lr 0.0006 --lr-step-epochs '15,25' --num-epochs 100 --lr-factor 0.1 --gpus 0,1,2,3,4,5,6,7 --batch-size 80 --image-shape '3,448,448'

#sleep 40m
for (( i = 3; i < 12; i++)); do 
  #python -u fine-tune.py --retrain --pretrained-model 'model/b1' --pretrained-epoch 26 --model-prefix "model/b1_$i" --num-classes 80 --lr 0.0006 --lr-step-epochs '15,25' --num-epochs 6 --lr-factor 0.1 --gpus 0,1,2,3,4,5,6,7 --batch-size 80 --image-shape '3,448,448' --train-image-root '/raid5data/dplearn/aichallenger/scene' --val-image-root '/raid5data/dplearn/aichallenger/scene'
  python -u fine-tune.py --retrain --pretrained-model 'model/ft224nude0003_97' --pretrained-epoch 50 --model-prefix "model/b2_$i" --num-classes 80 --lr 0.0003 --lr-step-epochs '15,25' --num-epochs 20 --lr-factor 0.1 --gpus 0,1,2,3,4,5,6,7 --batch-size 80 --image-shape '3,448,448'
  sleep 3
done;

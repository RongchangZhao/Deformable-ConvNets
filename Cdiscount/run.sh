export MXNET_CPU_WORKER_NTHREADS=64
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
./Python-4.8.2/bin/python fine-tune.py \
    --pretrained-model model/resnet-152 \
    --load-epoch 0 --gpus 0,1,2,3,4,5,6,7 \
    --data-train train_imglist --model-prefix model/discount-resnet-152 \
    --data-val val_imglist \
    --data-nthreads 64 \
    --batch-size 640 --num-classes 5270 --num-examples 12000000 \
    --lr 0.08 \
    --lr-factor 0.5 \
    --lr-step-epochs 5,20,25,30,35,40,50,60 \
    --image-shape 3,224,224 \
    --resize-size 256 \
    --num-epochs 80 \
    --top-k 5 1>&2 2>log/log.train.discount


#!/bin/bash
dims=(32 128 256 1024 4096)
models=('l_relu_enn' 'relu_enn' 'inn' 'h_tanh_enn' 'tanh_enn' 'tanh_inn')
for dim in ${dims[@]}
do
  for model in ${models[@]}
  do
  python main.py --dim $dim \
    --dataset 'fashion_mnist'\
    --model  $model\
    --asquare 0.2\
    --lr 1\
    --epoch 100\
    --batch_size 128\
    --device 'cuda:2';
  done;
done

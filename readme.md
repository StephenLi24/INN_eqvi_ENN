# Deep Equilibrium Models are Almost Equivalent to Not-so-deep Explicit Models for High-dimensional Gaussian Mixtures

[paper](https://arxiv.org/abs/2402.02697)

This repo contains the official implementation for the ICML 2024 paper Deep Equilibrium Models are Almost Equivalent to Not-so-deep Explicit Models for High-dimensional Gaussian Mixtures.

If you find the code useful, please refer to our work using:

```
@article{ling2024deep,
  title={Deep Equilibrium Models are Almost Equivalent to Not-so-deep Explicit Models for High-dimensional Gaussian Mixtures},
  author={Ling, Zenan and Li, Longbo and Feng, Zhanbo and Zhang, Yixuan and Zhou, Feng and Qiu, Robert C and Liao, Zhenyu},
  journal={arXiv preprint arXiv:2402.02697},
  year={2024}
}
```

## Running Experiments

### Project Structure

```
The description of the files:

---main.py                       # run main.py to get the result
---data_prepare.py               # get Gaussian Mixture data, here we use Mnist, Fashion_Mnist, and cifar10 feature as VGG16 net output
---model.py                      # contains L_relu_enn model, relu_enn model, INN model, h_tanh_enn, tanh_enn, tanh_inn model
---train.py                      # contains our training method
---utils.py                      # math function to get tau
---two_layer_lrelu.py            # calculate the coeffiencients of two layer l_relu_enn activation function
---one_layer_tanh.py             # calculate the coeffiencients of one layer h_tanh_enn activation function
---Gaussian_Integration.py       # math function of calculating the matching network coeffiencients 
---cifar10.sh               	 # the script to get the experiment result of cifar10
---mnist.sh                 	 # the script to get the experiment result of mnist
---fashion_mnist.sh         	 # the script to get the experiment result of FashionMnist

---ck_ntk:                       #  the floder, where contains some theoretical programs and visualization data
  ---ck_gmm_matching.py
  ---ck_mnist_matching.py
  ---gmm_NTK_matching.py
  ---gmm_two_layer_relu_matching.py
  ---mnist_NTK_matching.py
  ---mnist_one_layer_matching.py
  ---mnist_two_layer_matching.py

---pretrained  # the pretrained VGG model we use in cifar10 experiment

```

### The demo script and command of usage

`main.py` is the common gateway to all experiments. Type `python main.py --help` to get its usage description.

```

usage: main.py [-h] [--dataset DATASET] [--model MODEL] [--dim DIM] [--lr LR] [--epoch EPOCH] [--batch_size BATCH_SIZE] [--save_path SAVE_PATH] [--dataset_path DATASET_PATH] [--device DEVICE] [--vgg_path VGG_PATH]
               [--output_path OUTPUT_PATH] [--asquare ASQUARE]
options:
  -h, --help            show this help message and exit
  --dataset DATASET     Choose from mnist, fashion_mnist, cifar10
  --model MODEL         Choose your model:l_relu_enn, inn, relu_enn, tanh_inn, h_tanh_enn, tanh_enn
  --dim DIM             The dimension of our exp
  --lr LR               learning rate
  --epoch EPOCH         epoch number
  --batch_size BATCH_SIZE
                        batch size
  --save_path SAVE_PATH
                        save checkpoints path
  --dataset_path DATASET_PATH
                        dataset path
  --device DEVICE       cuda:0 or cpu
  --vgg_path VGG_PATH   pretrained vgg model path
  --output_path OUTPUT_PATH
                        output path
  --asquare ASQUARE     square of varience of w matrix in relu

```
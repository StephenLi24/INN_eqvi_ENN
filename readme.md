# Deep Equilibrium Models are Almost Equivalent to Not-so-deep Explicit Models for High-dimensional Gaussian Mixtures

[paper](https://openreview.net/forum?id=ddjRdm3wUW)

This repo contains the official implementation for the ICML 2024 paper Deep Equilibrium Models are Almost Equivalent to Not-so-deep Explicit Models for High-dimensional Gaussian Mixtures.

If you find the code useful, please refer to our work using:

```
@InProceedings{pmlr-v235-ling24a,
  title = 	 {Deep Equilibrium Models are Almost Equivalent to Not-so-deep Explicit Models for High-dimensional {G}aussian Mixtures},
  author =       {Ling, Zenan and Li, Longbo and Feng, Zhanbo and Zhang, Yixuan and Zhou, Feng and Qiu, Robert C and Liao, Zhenyu},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {30585--30609},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/ling24a/ling24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/ling24a.html},
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
---sgd_cifar10.sh                # the script to get the experiment result of CIFAR-10 using sgd
---adam_cifar10.sh             	 # the script to get the experiment result of CIFAR-10 using adam
---sgd_mnist.sh                  # the script to get the experiment result of MNIST using sgd
---adam_mnist.sh                 # the script to get the experiment result of MNIST using adam
---sgd_fashion_mnist.sh          # the script to get the experiment result of Fashion-MNIST using sgd
---adam_fashion_mnist.sh         # the script to get the experiment result of Fashion-MNIST using adam


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

usage: main.py [-h] [--dataset DATASET] [--model MODEL] [--dim DIM]
               [--lr LR] [--epoch EPOCH] [--batch_size BATCH_SIZE]
               [--save_path SAVE_PATH] [--dataset_path DATASET_PATH]
               [--device DEVICE] [--vgg_path VGG_PATH]
               [--output_path OUTPUT_PATH] [--asquare ASQUARE]
               [--opt OPT]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Choose from mnist, fashion_mnist, cifar10
  --model MODEL         Choose your model:l_relu_enn, inn, relu_enn,
                        tanh_inn, h_tanh_enn, tanh_enn
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
  --opt OPT             sgd or adam

```

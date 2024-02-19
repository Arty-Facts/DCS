The attached codes are for GAN inversion initialization and ITGAN training. You can combine them with our latest released codes in ITGAN GitHub repo. I am sorry that I don't have time to organize and test these codes recently. I will do it later. Thus, you may need manually do some small adaptation for the compatibility with our released codes. We will release the final version in recent months. 

Here are some example commands:
python -u ITGAN_GANInversion.py  --dataset CIFAR10  --model_train ConvNet --batch_train_z 100  --Iteration 5000  --Epoch_evaltrain 200  --lr_z 0.1  --split 0-50-0  > log/exp_GANInversion_CIFAR10_train_zbatch100_lrz0.1_split0-50-0.txt 2>&1 &

python -u ITGAN_main_CIFAR10.py  --dataset CIFAR10  --model_eval ConvNet  --Iteration 10000  --Epoch_evaltrain 200  --lr_net 0.01  --match_mode feat  --z_sample_mode fixed  --batch_train_z 1250  --lr_z 0.001  --ratio_div 0.001  --split 0-4-0  > log/exp_ITGAN_CIFAR10_feat_fixed_zbatch1250_lrz0.001_rdiv0.001_split0-4-0.txt

Best wishes,
Bo
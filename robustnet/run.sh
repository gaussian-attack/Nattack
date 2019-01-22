model=vgg16
noise=0.3
CUDA_VISIBLE_DEVICES=3 ./main2.py --lr 0.1 --noise ${noise} --modelOut ./${model}/noise_${noise}.pth > ./${model}/log_noise_${noise}.txt


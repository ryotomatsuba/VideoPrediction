for model_name in cifar10_x_sine_wave_low mnist_x_cifar10 sine_wave_high_x_sine_wave_low sine_wave_low_x_mnist sine_wave_high_x_cifar10 sine_wave_high_x_mnist
do  
echo $model_name
    # python3 train.py train=test dataset.frames_shift=100 dataset=video_data train.ckpt_path=/data/Models/$model_name
    python3 train.py train=test dataset.num_data=1 dataset=moving_image dataset.image_type=cifar10 train.ckpt_path=/data/Models/$model_name.pth
done



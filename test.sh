for model_name in unet_sine_wave_high.pth unet_sine_wave_low.pth unet_sine_wave_middle.pth unet_cifar10.pth
do  
echo $model_name
    # python3 train.py train=test dataset.frames_shift=100 dataset=video_data train.ckpt_path=/data/Models/$model_name
    python3 train.py train=test dataset.num_data=1 dataset=moving_image train.ckpt_path=/data/Models/unet_mix.pth
done



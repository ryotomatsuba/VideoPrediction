for image_type in "sine_wave_low_x_mnist" "mnist_x_cifar10" "cifar10_x_sine_wave_low"
do
    echo $image_type
    python3 train.py dataset.image_type=$image_type
done
from data.dataset.moving_image import *

mnist_images= get_mnist_images()
cifar10_images= get_cifar10_images()
v_x=3
velocity=(v_x,0)

video=make_transition_movie(mnist_images[0],start_pos=(0,10),velocity=velocity) # overrap videos
video+=make_transition_movie(cifar10_images[2],start_pos=(0,40),velocity=velocity) # overrap videos
print(video.shape)
save_gif(video,video,suptitle=)
clipped_video_mnist=np.zeros((10,28,28))
clipped_video_cifar10=np.zeros((10,28,28))
for i in range(video.shape[0]):
    x=i*v_x
    clipped_video_mnist[i]=video[i,x:x+28,10:10+28]
    clipped_video_cifar10[i]=video[i,x:x+28,40:40+28]

save_gif(clipped_video_cifar10,clipped_video_mnist,)

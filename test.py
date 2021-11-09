from data.dataset.moving_image import *
import tqdm
from models.unet import UNet
import torch.nn as nn

mnist_images= get_mnist_images()
cifar10_images= get_cifar10_images()
v_x=3
velocity=(v_x,0)
mix_image=np.concatenate([cifar10_images[0][:14],mnist_images[0][14:]])
video=make_transition_movie(mnist_images[0],start_pos=(0,10),velocity=velocity) # overrap videos
video+=make_transition_movie(cifar10_images[2],start_pos=(0,50),velocity=velocity) # overrap videos
video+=make_transition_movie(mix_image,start_pos=(20,90),velocity=velocity) # overrap videos
print(video.shape)
clipped_video_mnist=np.zeros((10,28,28))
clipped_video_cifar10=np.zeros((10,28,28))
for i in range(video.shape[0]):
    x=i*v_x
    clipped_video_mnist[i]=video[i,x:x+28,10:10+28]
    clipped_video_cifar10[i]=video[i,x:x+28,40:40+28]

# save_gif(clipped_video_cifar10,clipped_video_mnist,save_path="clip.gif")

pred_video=np.zeros((10,128,128))
input_num=4
batch_size=1
width=128
device="cpu"
criterion=nn.MSELoss()
load="/data/Result/mlruns/5/9ba13c32596b49838dec0f0e2d8fdd5c/artifacts/best_ckpt.pth"


net = UNet(n_channels=input_num, n_classes=1, bilinear=True) # define model
net.load_state_dict(torch.load(load, map_location=device))
preds=torch.empty(batch_size, 0, width, width).to(device=device)
total_num=video.shape[1]
# np to torch
video/=255
video=torch.from_numpy(video).to(device=device, dtype=torch.float32)
# add batch dim
video=video.unsqueeze(0)
input_X = video[:,0:input_num]
for t in range(input_num,10):
    with torch.no_grad():
        pred = net(input_X)
        print(pred.max())
    loss = criterion(pred, video[:,[t]])
    print(loss.item())
    input_X=torch.cat((input_X[:,1:],pred),dim=1) # use output image to pred next frame
    preds=torch.cat((preds,pred),dim=1) 
save_gif(video[0],preds[0])

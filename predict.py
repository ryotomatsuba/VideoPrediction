from matplotlib.pyplot import get
from data.dataset.moving_image import *
import tqdm
from models.unet import UNet
import torch.nn as nn

def predict(net, video):
    """
    Predict the future frames of a video.
    Params:
        net: the trained model
        video: past frames of the video
    Returns:
        frames: the predicted future frames
    """
    pred_video=np.zeros((10,128,128))
    input_num=4
    batch_size=1
    width=128
    device="cpu"
    criterion=nn.MSELoss()
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
    return preds


if __name__ == "__main__":
    # test
    from data.dataset.moving_image import get_moving_image_video, get_cifar10_images
    images=get_cifar10_images(num_images=3)
    videos=get_moving_image_video(images,num_sample=2,choice=["transition"])
    load="/data/Result/mlruns/5/9ba13c32596b49838dec0f0e2d8fdd5c/artifacts/best_ckpt.pth"
    net = UNet(n_channels=4, n_classes=1, bilinear=True) # define model
    net.load_state_dict(torch.load(load, map_location="cpu"))
    preds=predict(net, videos[0])
    save_gif(videos[0],preds[0])
    pass

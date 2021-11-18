from matplotlib.pyplot import get
from data.dataset.moving_image import *
import tqdm
from models.unet import UNet
from torch.nn.functional import mse_loss
from data.dataset.moving_image import get_moving_image_video, get_cifar10_images
from utils.fourier import *

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
        input_X=torch.cat((input_X[:,1:],pred),dim=1) # use output image to pred next frame
        preds=torch.cat((preds,pred),dim=1) 
    save_gif(video[0],preds[0])
    return preds.cpu().numpy()

def frequency_band_analysis(load):
    """
    Compare the prediction of different frequency videos.

    Params:
        load: the path of the trained model
    Returns:
        None
    """
    net = UNet(n_channels=4, n_classes=1, bilinear=True) # define model
    net.load_state_dict(torch.load(load, map_location="cpu"))
    cut_off=7
    for i,filter_func in enumerate([low_pass_filter,high_pass_filter]):
        images=get_cifar10_images(num_images=3)
        for j,image in enumerate(images):
            images[j]=filter_func(image,cut_off)
        videos=get_moving_image_video(images,num_sample=1,choice=["transition"])
        preds=predict(net, videos[0])
        per_frame_analysis(videos[0],preds[0])
        save_gif(videos[0],preds[0],save_path=f"freq_{i}.gif")

def per_frame_analysis(truth,preds):
    """
    Compare the prediction and ground truth of each frame.

    Params:
        truth: the ground truth video
        preds: the predicted frames
    Returns:
        None
    """
    output_num=len(preds)
    total_loss=0
    print(f"frame_mse from 0 to {output_num}:")
    for i in range(output_num):
        # mse loss
        loss=np.mean(np.square(preds[i]-truth[i]))
        print(f"{loss.item()}")
        total_loss+=loss.item()
    print(f"total_mse: {total_loss}")

if __name__ == "__main__":
    frequency_band_analysis("/data/Models/mnist_x_cifar10.pth")

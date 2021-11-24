import matplotlib.pyplot as plt
from data.dataset.moving_image import *
from models.unet import UNet
from data.dataset.moving_image import get_moving_image_video, get_cifar10_images
from utils.fourier import *

def predict(net, video):
    """
    Predict the future frames of a video.
    Params:
        net: the trained model
        video: (batch,frame,height,width) past frames of the video
    Returns:
        frames: (batch,frame,height,width) the predicted future frames
    """
    assert len(video.shape)==4, "video must be a 4D array (batch,frame,height,width)"
    input_num=4
    batch_size=1
    width=128
    device="cpu"
    preds=torch.empty(batch_size, 0, width, width).to(device=device)
    total_num=video.shape[1]
    # np to torch
    video/=255
    video=torch.from_numpy(video).to(device=device, dtype=torch.float32)
    input_X = video[:,0:input_num]
    for t in range(input_num,10):
        with torch.no_grad():
            pred = net(input_X)
        input_X=torch.cat((input_X[:,1:],pred),dim=1) # use output image to pred next frame
        preds=torch.cat((preds,pred),dim=1) 
    return preds.cpu().numpy()

def frequency_band_analysis(model_path:str):
    """
    Compare the prediction of different frequency videos.

    Params:
        model_path: the path of the trained unet models
    Returns:
        None
    """
    net = UNet(n_channels=4, n_classes=1, bilinear=True) # define model
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    cut_off=7
    input_frames=4
    output_frames=6
    model_name=model_path.split("/")[-1].split(".")[0]
    print(f"frequency_band_analysis :{model_name}")
    images=get_cifar10_images(num_images=3)
    video_num=20
    videos=get_moving_image_video(images,num_sample=video_num,choice=["transition"])
    loss_list_low=np.array([0.0]*output_frames)
    loss_list_high=np.array([0.0]*output_frames)
    for i in range(video_num):
        video=np.expand_dims(videos[i], axis=0)
        preds=predict(net, video)
        #save_gif(video[0],preds[0],save_path=f"{model_name}.gif")
        pred_high, pred_low, truth_high, truth_low  = np.zeros((4,preds.shape[1],preds.shape[2],preds.shape[3]))
        for j in range(len(preds[0])):
            pred_low[j],pred_high[j]=divide_by_frequency(preds[0][j],cut_off)
            truth_low[j],truth_high[j]=divide_by_frequency(video[0][input_frames+j],cut_off)
        loss_list_low =loss_list_low+ per_frame_analysis(truth_low,pred_low)
        loss_list_high = loss_list_high+per_frame_analysis(truth_high,pred_high)
    fig = plt.figure()
    plt.plot(loss_list_low/video_num,label="low")
    plt.plot(loss_list_high/video_num,label="high")
    plt.legend()
    plt.savefig(f"{model_name}.png")
    plt.close(fig)
    print(f"low_loss_mean: {loss_list_low/video_num}")
    print(f"high_loss_mean: {loss_list_high/video_num}")

def predict_by_frequency_band(model_path:str,cut_off= 7):
    """
    Predict the future frames of a video by frequency band.

    Params:
        model_path: the path of the trained unet models
        cut_off: the cut off frequency
    Returns:
        low_video_pred, high_video_pred: (batch,frame,height,width) the predicted future frames
    """
    net = UNet(n_channels=4, n_classes=1, bilinear=True) # define model
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    images=get_cifar10_images(num_images=3)
    preds=["",""]
    for i,filter_func in enumerate([low_pass_filter,high_pass_filter]):
        for j,image in enumerate(images):
            images[j]=filter_func(image,cut_off)
        videos=get_moving_image_video(images,num_sample=1,choice=["transition"])
        preds[i]=predict(net, videos)
    
    return preds[0],preds[1]

def per_frame_analysis(truth,preds):
    """
    Compare the prediction and ground truth of each frame.

    Params:
        truth: the ground truth video
        preds: the predicted frames
    Returns:
        np.array: the list of mse loss of each frame
    """
    assert len(truth)==len(preds)
    output_num=len(preds)
    total_loss=0
    print(f"frame_mse from 0 to {output_num}:")
    frame_mse=np.array([0.0]*output_num)
    for i in range(output_num):
        # mse loss
        loss=np.mean(np.square(preds[i]-truth[i]))
        print(f"{loss.item()}")
        frame_mse[i]=loss.item()
        total_loss+=loss.item()
    print(f"total_mse: {total_loss}")
    return frame_mse

if __name__ == "__main__":
    frequency_band_analysis("/data/Models/mnist_x_cifar10.pth")
    # frequency_band_analysis("/data/Models/mnist_x_cifar10.pth")


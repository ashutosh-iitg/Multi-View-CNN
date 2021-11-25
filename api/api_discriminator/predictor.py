import os
import torch

def predict(model, image):
    softmax = torch.nn.LogSoftmax(dim=1)
    model.eval()
    with torch.no_grad():
        output = model(image)
    
    output = torch.argmax(softmax(output), dim=1).detach().to(torch.device('cpu')).numpy()
    # output = output.detach().to(torch.device('cpu')).numpy()

    return output

def load_model(model_path, model):
    """Loads model parameters (state_dict) from model_path for inference.
    Args:
        model_path: (string) path to model state_dict
        model: (torch.nn.Module) model for which the parameters are loaded
    """
    if not os.path.exists(model_path):
        raise("File doesn't exist {}".format(model_path))
    state_dict = torch.load(model_path,map_location='cpu')
    model.load_state_dict(state_dict['state_dict'])

import torch.nn as nn
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision import models
import matplotlib.pyplot as plt

from typing import *
import pdb

class SimpleModel(nn.Module):

    def __init__(self, input_d: int, hidden_d: int, output_d: int):

        super(SimpleModel, self).__init__()

        self.input_d = input_d
        self.hidden_d = hidden_d
        self.output_d = output_d


        self.layer1 = nn.Linear(input_d, hidden_d)
        self.layer2 = nn.Linear(hidden_d, hidden_d)
        self.output_layer = nn.Linear(hidden_d, output_d)

        self.activation = nn.ReLU()


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)

        x = self.output_layer(x)
        
        return x
    


class VGG16Classifier(nn.Module):
    def __init__(self, num_classes: int, feature_extraction: bool=True):
        super(VGG16Classifier, self).__init__()

        # Load pretrained VGG16 model
        self.backbone = models.vgg16(weights='IMAGENET1K_V1')
        
        if feature_extraction:
            self.set_parameter_requires_grad(feature_extracting=feature_extraction)

        # Modify the classifier for the number of classes
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
    

    def extract_feature_maps(self, input_image:torch.Tensor):

        conv_weights =[]
        conv_layers = []
        total_conv_layers = 0

        for module in self.backbone.features.children():
            if isinstance(module, nn.Conv2d):
                total_conv_layers += 1
                conv_weights.append(module.weight)
                conv_layers.append(module)


        print("TOTAL CONV LAYERS: ", total_conv_layers)
        feature_maps = []  # List to store feature maps
        layer_names = []  # List to store layer names
        for layer in conv_layers:
            input_image = layer(input_image)
            feature_maps.append(input_image)
            layer_names.append(str(layer))

        for feature_map in feature_maps:
            print(feature_map.shape)

        # Process and visualize feature maps
        processed_feature_maps = []  # List to store processed feature maps
        for feature_map in feature_maps:
            feature_map = feature_map.squeeze(0)  # Remove the batch dimension
            mean_feature_map = torch.sum(feature_map, 0) / feature_map.shape[0]  # Compute mean across channels
            processed_feature_maps.append(mean_feature_map.data.cpu().numpy())


        return processed_feature_maps, layer_names


        

    def extract_features_from_hooks(self, x, layers: List[str]):
        """
        Extract feature maps from specified layers.
        Args:
            x (torch.Tensor): Input tensor.
            layers (List[str]): List of layer names to extract features from.
        Returns:
            Dict[str, torch.Tensor]: Feature maps from the specified layers.
        """
        outputs = {}
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                outputs[name] = output
            return hook

        # Register hooks for specified layers
        #for layer_name in layers:
        dict_named_children = {}
        for name, layer in self.backbone.named_children():
            for n, specific_layer in layer.named_children():
                dict_named_children[f"{name}.{n}"] = specific_layer

        for layer_name in layers:
            layer = dict_named_children[layer_name]
            hooks.append(layer.register_forward_hook(get_activation(layer_name)))

        # Perform forward pass
        _ = self.forward(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return outputs

    def modify_layers(self, modify_fn: Callable[[nn.Module], nn.Module]):
        """
        Modify layers of the model using a provided function.
        Args:
            modify_fn (Callable[[nn.Module], nn.Module]): Function to modify a layer.
        """
        self.vgg16 = modify_fn(self.vgg16)


    def set_parameter_requires_grad(self, feature_extracting):
        """
        Set parameters gradients to false in order not to optimize them in the training process.
        """
        if feature_extracting:
            for param in self.backbone.parameters():
                param.requires_grad = False



    def extract_grad_cam(self, input_image: torch.Tensor, 
                         target_layer: List[Type[nn.Module]], 
                         targets: List[Type[ClassifierOutputTarget]]) -> Type[GradCAMPlusPlus]:

        

        with GradCAMPlusPlus(model=self.backbone, target_layers=target_layer) as cam:

            grayscale_cam = cam(input_tensor=dummy_input, targets=targets)[0, :]

        return grayscale_cam





# Example of usage
if __name__ == "__main__":
    torch.manual_seed(42)

    # Load a pretrained model and modify it
    model = VGG16Classifier(num_classes=10, feature_extraction=False)
    """
        features.0.weight
        features.0.bias
        features.2.weight
        features.2.bias
        features.5.weight
        features.5.bias
        features.7.weight
        features.7.bias
        features.10.weight
        features.10.bias
        features.12.weight
        features.12.bias
        features.14.weight
        features.14.bias
        features.17.weight
        features.17.bias
        features.19.weight
        features.19.bias
        features.21.weight
        features.21.bias
        features.24.weight
        features.24.bias
        features.26.weight
        features.26.bias
        features.28.weight
        features.28.bias
        classifier.0.weight
        classifier.0.bias
        classifier.3.weight
        classifier.3.bias
        classifier.6.weight
        classifier.6.bias
    """

    # Example GradCAM usage
    dummy_input = torch.randn(1, 3, 224, 224)
    target_layers = [model.backbone.features[2]]
    targets = [ClassifierOutputTarget(9)]

    # Display processed feature maps shapes
    processed_feature_maps, layer_names = model.extract_feature_maps(dummy_input)

    for fm in processed_feature_maps:
        print(fm.shape)

    # Plot All the convolution feature maps separately
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed_feature_maps)):
        ax = fig.add_subplot(5, 4, i + 1)
        ax.imshow(processed_feature_maps[i])
        ax.axis("off")
        ax.set_title(f"{layer_names[i].split('(')[0]}_{i}", fontsize=10)


    plt.show()

    
    ## Plot a concret layer feature map when processing a image thorugh the model
    ## Is not necessary to have gradients

    with torch.no_grad():
        feature_map = (model.extract_features_from_hooks(x=dummy_input, layers=["features.28"]))["features.28"]
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        processed_feature_map = torch.sum(feature_map, 0) / feature_map.shape[0]  # Compute mean across channels

    # Plot the result
    plt.imshow(processed_feature_map, cmap="gray")
    plt.axis("off")
    plt.show()

    input_image = dummy_input.squeeze(0).permute(1, 2, 0).cpu().numpy()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min()) ## Image needs to be between 0 and 1 and be a numpy array (Remember that if you have norlized the image you need to denormalize it before applying this (image * std + mean))

    ## VIsualize the activation map from Grad Cam
    ## To visualize this, it is mandatory to have gradients.
    grad_cams = model.extract_grad_cam(input_image=dummy_input, target_layer=target_layers, targets=targets)
    visualization = show_cam_on_image(input_image, grad_cams, use_rgb=True)


    # Plot the result
    plt.imshow(visualization)
    plt.axis("off")
    plt.show()
    #plt.imshow(cam_result.squeeze().numpy(), cmap='jet')
    #plt.show()

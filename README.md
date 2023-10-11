# iCAD-Dental

NOTE: Due to colab GPU usage limits, 50% of the data in each class was used.

NOTE2: An alternative neural network (architechture created by specifying each layer's details) can be seen commented out in the ipynb files but a pretrained_model (resnet50) was used with the output head modified for this task's images since training from scratch proved difficult without enough GPU allocation to try different network architechtures and do a proper hyperparameters search in the given time. (After reaching daily usage limit, I could not test anything so the pretrained model made sense to use at that point)

Results: 

iCAD_Dental_Coding_Test_Colab_MultiClass.ipynb:

This is the multiple classes classification notebook. Accuracy of 77.54% is achieved with training on 5 epochs. Due to GPU allocation limits, if I wanted to finetune the backbone, doing too many epochs would require a lot more resources.
Loss was 0.0278 by 5th epoch so this was not far from converging to its highest accuracy using this network architechture.

iCAD_Dental_Coding_Test_Colab_BinaryClassification.ipynb:

For the binary classification, 92.50% accuracy is achieved, 26.33 Sensitivity and 0.00 Specificity. This is using a pretrained model with the output head being modified and retrained on this dataset. Finetuning was off for this task so only the final layer was retrained on the target task dataset. Visualization of the conv1 layer (Which is a conv2d layer) activation maps in resnet50 and SHAP features output images are also done in this notebook. These are discussed in detail below.


conv1_feature_maps_64FilterActivationMaps.png:

conv1_feature_maps_64FilterActivationMaps.png shows the activation maps of passing 64 different filters with a conv2d layer. This is the first layer so the incoming channels are the 3 RGB channels and the convolution is mapping it to 64 channels by using 64 different filters. 
Refer to 'class ResNet(nn.module)' for reference on full details of the architecture in https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html

shap_top_features_explained.png:

shap_top_features_explained.png shows, for the binary classification task, a plot of the crucial sections in images that affect the model's decisions the most.
The shap plot shows the features (or in this case, pixel arrangements) that impact the prediction accuracy of there being a tumour or not. 

The blue pixels represent negative SHAP values which means the presence of these pixels lowers the predicitability for the model. The more intense the blue (more negative SHAP value), means a more significant influence and those pixels make predictions worse compared to 'less blue' ones.

The red pixels represent positive shap values and these are what the model uses best to predict correctly the presence of tumours. Similarly, the more red (more positive SHAP value) means the pixel is more important in making a correct prediction.

The shap values that are not too red or blue are insignificant and do not have a major impact on the predictability.

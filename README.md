# RMBG-U2NET

## 1. Introduction
Image Background Removal using segmentation is a fundamental task in computer vision, where the goal is to classify each pixel in an image as belonging to one of several classes and seprate the wanted pixel and remove the unwanted ones. for this task we will be using U-2-Net based on The U-Net model, which is a popular architecture for image segmentation tasks. firtly lets look at 

``` bash
    RMBG-U2NET
    │
    ├── Data  
    │
    ├── RMBG/
    │   ├── __init__.py 
    │   ├── model.py
    │   └── train.py
    │
    ├── .gitignore
    ├── README.md
    ├── RMBG_ENV.yml
    └── workspace.ipynb
```

## 2. U-Net Architecture
U-Net Architecture is based on nested U-structure that consists of an encoder that downsamples the input image to extract high-level features, a bridge that applies dilated convolutions to increase the receptive field, and a decoder that upsamples the features to produce a segmentation mask.
this Architecture which allows it to capture high-level semantic information and low-level details simultaneously. combination of low-level features from the encoder with high-level features from the decoder, helps to improve the accuracy of the segmentation.
This makes it particularly effective for tasks like image background removal, where both the object of interest (foreground) and the rest of the image (background) need to be accurately identified.

### 2.4. implementation U-Net Architecture 
In the next sections, we will discuss the implementation of Image Background Removal using U-2-Net,to implement U-2-Net model we use a pre-trained ResNet50 model. The ResNet50 model is a deep convolutional neural network that has been pre-trained on the ImageNet dataset, which consists of over 1 million images and 1000 classes. By using a pre-trained model, we can leverage the knowledge that the model has already learned from the ImageNet dataset and apply it to our image segmentation task.

We use ResNet50 for first five convolutional blocks that create the The encoder part,and  with skip connections to the decoder. The bridge consists of a dilated convolution with 1024 filters. The decoder consists of four decoder blocks, each of which upsamples the input and concatenates it with skip features from the encoder, followed by a residual block. Four output layers are defined, each of which applies a 1x1 convolution with sigmoid activation to produce a binary segmentation mask. The output masks are concatenated along the channel dimension.

### 2.1. Residual Blocks
Residual blocks are a type of convolutional block that consists of several convolutional layers with batch normalization and ReLU activation, followed by a shortcut connection that adds the input to the output of the convolutional layers. Residual blocks help to improve the accuracy of the model by allowing the gradient to flow more easily through the network, which helps to prevent the vanishing gradient problem.

In this implementation, we define a residual block function that takes an input tensor and a number of filters as arguments. The function applies two 3x3 convolutional layers with batch normalization and ReLU activation, followed by a shortcut connection that adds the input to the output of the second convolutional layer.

### 2.2. Dilated Convolutions
Dilated convolutions are a type of convolutional layer that increases the receptive field of the model by using a larger kernel size with a dilation rate. The dilation rate determines the spacing between the kernel elements, which allows the model to capture larger context without increasing the number of parameters.

In this implementation, we define a dilated convolution function that takes an input tensor and a number of filters as arguments. The function applies three dilated convolutions with dilation rates of 3, 6, and 9 to the input, and then concatenates the outputs and applies a 1x1 convolution with batch normalization and ReLU activation.

### 2.3. Decoder Block
The decoder block is a type of convolutional block that upsamples the input by a factor of 2 and concatenates it with skip features from the encoder, followed by a residual block. The upsampling is performed using bilinear interpolation, which helps to preserve the spatial information of the input.

In this implementation, we define a decoder block function that takes an input tensor, skip features from the encoder, and a number of filters as arguments. The function upsamples the input using bilinear interpolation, concatenates it with the skip features, and then applies a residual block.

It's important to note that while U-2-Net is powerful, the quality of the results can depend on the complexity of the image and the distinctness of the object of interest. we have discussed the implementation of the U-Net model for image segmentation tasks using a pre-trained ResNet50 model. now lets discuss the Data we used to train our model.

## 3. DATA
To train this model we used a dataset that contains a combined dataset of P3M-10K, COD-10K-v3, and People Segmentation datasets. These datasets are widely used in the field of computer vision, particularly in the areas of reflection, shadow, and animal and people segmentation. first lets disscussed each and every dataset used individually:


### 3.1. P3M-10K
P3M-10K is a large-scale portrait matting dataset that contains 10,000 high-resolution portrait images, along with their corresponding alpha mattes and foregrounds. The dataset is divided into training, validation, and testing sets, each containing 7,000, 1,000, and 2,000 images, respectively. in this dataset the faces are blured due to privacy concernes.

### 3.2. COD-10K-v3
COD-10K-v3 is a large-scale dataset for object detection and segmentation, which contains 10,000 images with 80 object categories. The dataset is divided into training, validation, and testing sets, each containing 7,000, 1,000, and 2,000 images, respectively.

### 3.3. People Segmentation
People Segmentation dataset is a collection of images containing people, along with their corresponding segmentation masks. compared to P3M-10k dataset this dataset contains different angles and different variation of pictuires including people inside it.

### 3.4. Dataset combination & preproccessing
The dataset used in this project is a combination of the P3M-10k and people segmentation datasets, which are two widely used datasets for the human segment of Datasets but the structure of these two datasets is different, the first step was to remove the segmentation folder from people segmentation dataset we only need the human mask after all to remove humans from the background, we left with the original images and masks, for the other people dataset "P3M-10k" firstly we use use the train, it doesn’t need any further preprocess to making the validation part ready we use "P3M-500-NP" folder and we only use "mask" and "original_image" folders, we don't need the other files from this dataset. and finally COD-10k-v3 dataset, from both the "train" and "test" folders, we use Image and "GT_object" folders that contain the mask of the original object, after copying the needed folder we mix all the original images in one folder named images and we use "mask" folders and "GT-object" files into a masks folder, by this we have a dataset contains 25K samples of (.jpg) images and (.PNG) masks, now by a combination of these three rich datasets, our model can be trained on animals, objects and humans.

*** IMPORTANT NOTICE: In workshop.ipynb file the model is trained only on 500 sample datafrom the original dataset due to computational limitation ***

## 4. installing and Running the model

### 4.1. install and Load the Model: 
to be able to run the model first step is to clone the repository from GitHub:
You can download the repository to your local machine using git clone command. Open your terminal and type:

```bash
git clone git@github.com:Mehdialmoo/RMBG-U2NET.git
```

in the second step, Once the repository is cloned, navigate to the repository using the cd command:

```bash
cd RMBG-U2NET
```

Install the required packages and the environment:
The repository includes a environment file that lists all the required packages. To install and activate the environment specified in the given YAML file, you can follow these steps:

Open a terminal or command prompt and navigate to the directory where you saved the environment.yml file.
Run the following command to create the new environment:
```bash
conda env create -f environment.yml
```
After the installation is complete, activate the environment using the following command:
```bash
conda activate rmbg_ENV
```
Now, you have successfully installed and activated the environment with the specified packages and dependencies.

### 4.2. Prepare the Dataset:
Due to Github limitation the dataset is uploaded on ....... by clicking here[] you will be able to download the dataset after cloning this repository, you are able to unzip the dataset copy and paste in the following order
```bash
    combined_dataset/
    │
    ├── Data/
    │   ├── Images/   
    │   ├── Mask/
    │   ├── file/ 
    │   └── test/
    .
    .
    .
```
"file" folder contains the data.CSV file saves the weights and values of our model and "model.h5" which is a checkpoint of the model which if you dont need to train the model you can use this checkpoint by loading it using third cell in "workspace.ipynb" file.

### 4.3. Run the Model:
After following all the steps stated above, there are two methods to follow up either to train the model from scratch or to load the pretrained model that both methods will be disscussed in order:
    #### 4.3.1 train the model:
    firstly open "workspace.ipynb" file , then run the first 3 cells in order, after running the traning process firstly the model tries to create a folder named "file" is it wasn't existing along side other files in "Data" folder than the model tries to seperate validation and traning data apart from the initial data that includes images and masks, to save the weights in an CSV file and the model checkpoint as "h5" datatype.
    second stage is it shows how many files are selected for traning and validation, and after that the model represents a quick summary of the model than it starts the traning process.

    because of the computational limitation we trained the model on 500 samples of the original data and for the traning process we used this specefic settings but if you have the computational power run the model on the original data we recomment you to use the best performence setting, the settings are as follows:
    
    sample traning setting:
    ```python

    ```
    our sample traning is based on 


    recommended traning setting:
    ```python
    
    ```
    for the ease of use you can simply copy and paste this setting.

    *** IMPORTANT NOTICE: It's better to train the Recommended setting on the full Dataset which is avalible from this link []***

    After traning the model and before heading to the last cell we recommend reading the next section (next section here)

    
    
    
    #### 4.3.2 load the pretrained model:
    firstly open "workspace.ipynb" file , then run the first 2 cells in order, then you can skip the 3rd cell and head to the last cell to use the model.

    After loading the model based on checkpopint and before heading to the last cell we recoomend reading the next section to be able to test the model on you own orefrenced images and get results.

### 4.4. Background Removal: 


## 5. References:

1. U-Net: Convolutional Networks for Biomedical Image Segmentation. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. https://arxiv.org/abs/1505.04597
2. Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015. https://arxiv.org/abs/1512.03385
3. TensorFlow Keras Documentation. https://www.tensorflow.org/api_docs/python/tf/keras
ResNet50 Model. https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50

4. U-2-Net: U Square Net - GitHub. https://github.com/xuebinqin/U-2-Net.
5. Image Background Removal Using U-2-Net | by Leslie Kaye VM - Medium. https://medium.com/@vm.lesliekaye/image-background-removal-using-u-2-net-cf36f1c3efc7.
6. Image Background Removal with U^2-Net and OpenVINO™. https://docs.openvino.ai/2022.3/notebooks/205-vision-background-removal-with-output.html.
7. Step-by-Step Guide to Automatic Background Removal in Videos ... - Medium. https://medium.com/@CodingNerdsCOG/step-by-step-guide-to-automatic-background-removal-in-videos-with-u-2-model-deep-learning-aae7c297654a.
8. U2Net : A machine learning model that performs object cropping ... - Medium. https://medium.com/axinc-ai/u2net-a-machine-learning-model-that-performs-object-cropping-in-a-single-shot-48adfc158483.
9. P3M-10K:"P3M-10K: A Large-Scale Portrait Matting Dataset for Learning Automatic Background Replacement." IEEE Transactions on Image Processing, vol. 30, pp. 3665-3678, 2021.
10. COD-10K-v3: "COD-10K: A Large-Scale Object Detection Dataset for Contextual Object Detection." IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 10, pp. 3535-3548, 2021.
11. People Segmentation: "Deep Learning for People Segmentation." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4706-4715, 2017.



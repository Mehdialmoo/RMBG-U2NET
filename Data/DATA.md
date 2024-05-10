## DATA
To train this model we used a dataset that contains a combined dataset of P3M-10K, COD-10K-v3, and People Segmentation datasets. These datasets are widely used in the field of computer vision, particularly in the areas of reflection, shadow, and animal and people segmentation. first lets disscussed each and every dataset used individually:

### P3M-10K
P3M-10K is a large-scale portrait matting dataset that contains 10,000 high-resolution portrait images, along with their corresponding alpha mattes and foregrounds. The dataset is divided into training, validation, and testing sets, each containing 7,000, 1,000, and 2,000 images, respectively. in this dataset the faces are blured due to privacy concernes. [Download](https://www.kaggle.com/datasets/rahulbhalley/p3m-10k)

### COD-10K-v3
COD-10K-v3 is a large-scale dataset for object detection and segmentation, which contains 10,000 images with 80 object categories. The dataset is divided into training, validation, and testing sets, each containing 7,000, 1,000, and 2,000 images, respectively. [Download](https://drive.google.com/file/d/1vRYAie0JcNStcSwagmCq55eirGyMYGm5/view?pli=1)

### People Segmentation
People Segmentation dataset is a collection of images containing people, along with their corresponding segmentation masks. compared to P3M-10k dataset this dataset contains different angles and different variation of pictuires including people inside it. [Download](https://www.kaggle.com/datasets/nikhilroxtomar/person-segmentation?rvi=1)

### Dataset Combination
The dataset used in this project is a combination of the P3M-10k and people segmentation datasets, which are two widely used datasets for the human segment of Datasets but the structure of these two datasets is different, the first step was to remove the segmentation folder from people segmentation dataset we only need the human mask after all to remove humans from the background, we left with the original images and masks, for the other people dataset "P3M-10k" firstly we use use the train, it doesn’t need any further preprocess to making the validation part ready we use "P3M-500-NP" folder and we only use "mask" and "original_image" folders, we don't need the other files from this dataset. and finally COD-10k-v3 dataset, from both the "train" and "test" folders, we use Image and "GT_object" folders that contain the mask of the original object, after copying the needed folder we mix all the original images in one folder named images and we use "mask" folders and "GT-object" files into a masks folder, by this we have a dataset contains 25K samples of (.jpg) images and (.PNG) masks, now by a combination of these three rich datasets, our model can be trained on animals, objects and humans.

### Data Preprocessing
To  be able  to work with the images and their mask we need some preproccessing steps, there is a Preprocess folder which contain "util.py" and "Preprocess.ipynb" file which are to normalize and regulize the data that we have, that performs several operations on image files in a directory. Here's a breakdown of what each function does:


#### Rename function: 

This function renames all image files in the specified directory by adding a category prefix to the filename. It checks if the file is an image (either JPG or PNG format) and renames it by appending the category name followed by an underscore and a count number.

#### Resize function:

This function resizes all image files in the specified directory to the specified size. It checks if the file is an image (either JPG, JPEG, or PNG format) and resizes it using the PIL library. the pictures could be left as they are because we use transformers in the model to resize all the input images also but this function us provided in case the model didn't work on the minimum required settings, you can remove the transformer part from the actual code and resize all the data seprerately.

#### create_csv function:

This function creates a CSV file containing the image name, path of image, mask path, and category of the image. It traverses through the image path and mask path, gets the image name and category, and writes the image details to the CSV file.

#### plot_Data function: 

This function plots the number of files for each category in the specified directory. It traverses through the directory, gets the category from the filename, increments the count for the category, and plots the number of files for each category using matplotlib.

### Prepare the Dataset:
Due to Github limitation the dataset is uploaded on Kaggle by clicking here[Sample: 1000 images](https://www.kaggle.com/datasets/mehdialmousavie/sample-u-2-net-rmbg) [full : 25k images](https://www.kaggle.com/datasets/mehdialmousavie/full-u2net-rmbg) you will be able to download the dataset. After cloning this repository, you are able to unzip the dataset copy and paste in the following order
```
    RMBG-U2NET/
    │
    ├── Data/
    │   ├── Images   
    │   ├── Mask
    │   ├── Dataset.CSV
    │   ├── DATA.md
    │   ├── file/ 
    │   └── test/
    │       ├── images 
    │       └── masks
    .
    .
```
"file" folder contains the data.CSV file saves the weights and values of our model and "model.h5" which is a checkpoint of the model which if you dont need to train the model you can use this checkpoint by loading it using third cell in "workspace.ipynb" file.
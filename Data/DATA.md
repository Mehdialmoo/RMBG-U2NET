## DATA
To train this model we used a dataset that contains a combined dataset of P3M-10K, COD-10K-v3, and People Segmentation datasets. These datasets are widely used in the field of computer vision, particularly in the areas of reflection, shadow, and animal and people segmentation. first lets disscussed each and every dataset used individually:


### P3M-10K
P3M-10K is a large-scale portrait matting dataset that contains 10,000 high-resolution portrait images, along with their corresponding alpha mattes and foregrounds. The dataset is divided into training, validation, and testing sets, each containing 7,000, 1,000, and 2,000 images, respectively. in this dataset the faces are blured due to privacy concernes.

### COD-10K-v3
COD-10K-v3 is a large-scale dataset for object detection and segmentation, which contains 10,000 images with 80 object categories. The dataset is divided into training, validation, and testing sets, each containing 7,000, 1,000, and 2,000 images, respectively.

### People Segmentation
People Segmentation dataset is a collection of images containing people, along with their corresponding segmentation masks. compared to P3M-10k dataset this dataset contains different angles and different variation of pictuires including people inside it.

### Dataset combination & preproccessing
The dataset used in this project is a combination of the P3M-10k and people segmentation datasets, which are two widely used datasets for the human segment of Datasets but the structure of these two datasets is different, the first step was to remove the segmentation folder from people segmentation dataset we only need the human mask after all to remove humans from the background, we left with the original images and masks, for the other people dataset "P3M-10k" firstly we use use the train, it doesn’t need any further preprocess to making the validation part ready we use "P3M-500-NP" folder and we only use "mask" and "original_image" folders, we don't need the other files from this dataset. and finally COD-10k-v3 dataset, from both the "train" and "test" folders, we use Image and "GT_object" folders that contain the mask of the original object, after copying the needed folder we mix all the original images in one folder named images and we use "mask" folders and "GT-object" files into a masks folder, by this we have a dataset contains 25K samples of (.jpg) images and (.PNG) masks, now by a combination of these three rich datasets, our model can be trained on animals, objects and humans.
### Prepare the Dataset:
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
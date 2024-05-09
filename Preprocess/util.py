import csv
import os
import matplotlib.pyplot as plt

from PIL import Image


def Rename(path, category):
    for count, filename in enumerate(os.listdir(path)):
        if filename.endswith(('.jpg', '.jpeg')):  # check if the file is an image
            dst = f"{category}_{str(count)}.jpg"  # create the new filename
        elif filename.endswith(('.png')):
            dst = f"{category}_{str(count)}.png"  # create the new filename
        src = f"{path}/{filename}"  # create the source path
        dst = f"{path}/{dst}"  # create the destination path
        os.rename(src, dst)  # rename the file


def resize(path, new_size):
    for filename in os.listdir(path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # check if the file is an image
            img_path = f"{path}/{filename}"  # create the image path
            img = Image.open(img_path)  # open the image
            img = img.resize(new_size, Image.LANCZOS)  # resize the image
            img.save(img_path)  # save the image


def create_csv(image_path, mask_path, output_file):
    # Create a list to store the image details
    image_details = []

    # Traverse through the image path
    for image in os.listdir(image_path):
        # Get the image name and category
        category, image_name = image.split('_')
        image_details.append([
            image,
            category,
            image_path+"\\"+image,
            mask_path+"\\"+image[:-4]+".png"])

    # Write the image details to the CSV file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Image Name', 'Category', 'Image Path', 'Mask Path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in image_details:
            writer.writerow(
                {'Image Name': row[0], 'Category': row[1], 'Image Path': row[2], 'Mask Path': row[3]})


def plot_Data(path):
    # Create a dictionary to store the number of files for each category
    file_categories = {}

    # Traverse through the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            # Get the category from the filename
            category = file.split('_')[0]

            # Increment the count for the category
            if category in file_categories:
                file_categories[category] += 1
            else:
                file_categories[category] = 1

    # Create a list of categories and a list of counts
    categories = list(file_categories.keys())
    counts = list(file_categories.values())

    # Plot the number of files for each category
    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts)
    plt.xlabel('Category')
    plt.ylabel('Number of Files')
    plt.title('Number of Files by Category')
    plt.show()

# Image Classifier [Deep Learning]

## Project Overview
Flowers Image Classification using Deep Learning. This final project in the Deep Learning section is part of Udacity's Machine Learning Nanodegree program.
I built and train a neural network that learns to classify images of flowers, using TensorFlow.

- Part1:  implementation of an image classifier model.
- Part2: convert the model into an application that others can use. Which is a Python script that run from the command line. 

## Software and Libraries
This project uses the following software and Python libraries:

•	Python
•	NumPy
•	TensorFlow
•	TensorFlow Dataset
•	TensorFlow Hub
•	Matplotlib
You will also need to have software installed to run and execute a Jupyter Notebook.

## Data
 I used [this](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) dataset from Oxford of 102 flower categories.
 
 
## Run

### Part1: Run in Jupyter Notebook

In a terminal or command window, navigate the direction of the project(part1) and run the following command:

```bash
jupyter notebook Project_Image_Classifier_Project.ipynb
```

This will open the iPython Notebook software and project file in your browser.
Note that, when you run the cell of training the model, it will take a lot of time if you are using the CPU, instead use the GPU to accelerate the process.

### Part2: Run in Command Line

In a terminal or command window, navigate the direction of the project(part2).
The predict.py module should predict the top flower names from an image along with their corresponding probabilities.
I have provided 4 images in the test_images folder to check your predic.py module. The 4 images are:
- cautleya_spicata.jpg
- hard-leaved_pocket_orchid.jpg
- orange_dahlia.jpg
- wild_pansy.jpg


fist off, make sure to install TensorFlow 2.0 and TensorFlow Hub using pip as shown below:

```bash
pip install -q -U "tensorflow-gpu==2.0.0b1
```

```bash
pip install -q -U tensorflow_hub
```

then run the following commands:

Basic usage: write image name and model name

```bash
python predict.py ./test_images/HERE WRITE THE IMAGE NAME FROM "test_image" FOLDER Project_load.h5
```


Options: write image name and model name, and the top K most likely classes for example, k=3 :

```bash
python predict.py ./test_images/HERE WRITE THE IMAGE NAME FROM "test_image" FOLDER Project_load.h5 --top_k 3
```


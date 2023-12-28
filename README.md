# cnn_model_optimization
Optimize convolutional neural network by varying batch size and learning rate during training

Batch size and leanring rate can have a significant effect on the outcomes of training an CNN for image classification.
This program conducts experiments on the given CNN model by varying batch size and learning rates and saving the training data in csv files.
After iterating through training these models, the csv files are read and summarized into an the ouput that gives the results for each batch size/learning rate iteration.
Depending on the image data set, the results could show significant differences in overall model aacuracy when changing these training parameters.

To use, please provide a directory where you training images are stored and a directory to save the csv files.
The training images should be organized in folders labeled by numbers for each category.


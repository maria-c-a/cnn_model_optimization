import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop, SGD, Adam
import os
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
plt.style.use('classic')
import pandas as pd
import glob
import os

#def train(batch_size, size, epochs, lr )

def train_model(batch_size, learning_rate, image_directory, checkpoint_path):
        
    # Set the image directory
    
    SIZE = 60
    BATCH_SIZE = batch_size
    NUM_CLASSES = 43
    EPOCHS = 30
    lr = learning_rate

    # Use TensorFlow ImageDataGenerator for image loading and augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # Split data into training and validation sets
    )

    train_generator = datagen.flow_from_directory(
        image_directory,
        target_size=(SIZE, SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'  # Specify it's the training set
    )

    val_generator = datagen.flow_from_directory(
        image_directory,
        target_size=(SIZE, SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'  # Specify it's the validation set
    )

    # Build the model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    custom_optimizer = Adam(learning_rate = lr)

    model.compile(optimizer=custom_optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    
    # Use ModelCheckpoint to save the best model during training
    log_file_name = "dim_" +str(SIZE)+ "_"+ str(SIZE) +"_bsz_" +str(BATCH_SIZE)+"_lr_"+ str(lr)+ "_training_data_adam" +".csv"
    log_file_path = os.path.join(checkpoint_path, log_file_name)
    print(log_file_path)

    csv_logger = CSVLogger(log_file_path, separator= ",", append= False)
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, verbose=1, save_best_only= True, monitor= "val_loss")



    '''
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        verbose=1,
        save_best_only=True,
        monitor="val_loss"
    )
    '''

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        callbacks=[csv_logger]
    )

    '''
    # Plot training and validation accuracy and loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(val_generator, steps=val_generator.samples // BATCH_SIZE)
    print("Test Accuracy: {:.2f}%".format(test_acc * 100))
    '''

if __name__ == "__main__":
    
    #This program trains models models to find the optimum batch size and learning rate to use
    #Each time a model trains, it is an "experiment"
    #For these experiments, we decided to use the adam optimizer. 

    ##The training images are stored in the image directory
    ##In the image directory, each category of images should be labeled by a number and the images corresponding to that category
    ##should be stored in the respective folders
    image_directory = 'C:\\training_images_folder'
    #This program saves the training and validation accuracy and loss for every epoch within a csv file
    #The csv file is saved in the specified checkpoint path
    #each time a model is trained with each iteration of batch_size and learning rate, a csv file is saved
    checkpoint_path = 'C:\\model_checkpoint_folder'
    
    for batch_size in [4,8,16,32,64,128,256]:
        for learning_rate in [0.000100 , 0.001000, 0.01000]:
            try:
                train_model(batch_size, learning_rate, image_directory)
            except:
                print(f'batch_size {batch_size} and learning rate {learning_rate} did not train')
    
    #Once all the csv files are generated for each experiment, we want to print a summary of results to find out
    #which batch size/learning rate configuration yeiled the highest validation accuracy after training
    
    # Create an empty DataFrame to store results
    max_val_df = pd.DataFrame(columns=['File', 'Max_Val_Accuracy'])

    # Replace 'your_folder_path' with the path to your CSV files
    folder_path = checkpoint_path

    # Iterate through each CSV file
    for file in os.listdir(folder_path):
        # Read the CSV file into a DataFrame
        file_path = os.path.join(folder_path, file)
        print(file_path)
        df = pd.read_csv(file_path)

        # Find the max value under the 'val_accuracy' column
        max_val = df['val_accuracy'].max()

        new_row_values = [file, max_val]
        max_val_df.loc[len(max_val_df)] = new_row_values

    # Display the resulting DataFrame
    print(max_val_df)

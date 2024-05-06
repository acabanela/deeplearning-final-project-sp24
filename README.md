# Deep Learning Final Project: Plant Disease Image Classification of Corn Crop Leaves
## About the notebooks and data:
* `Amadeo_EDA_Preprocessing_Corn.ipynb` - EDA and preprocessing techniques applied, code for saving preprocesed data to numpy files
* `Amadeo_Baseline_Corn.ipynb` - simple neural network baseline model using data loaded from numpy files
* Where the data is located
  * I have given everyone access to the google drive folder "634_Deep_Learning_Project_Shared".
  * The raw data we're using for EDA and preprocessing is in the `/data/raw_data/color_corn/` folder.
  * **The preprocessed data we're using for model building** is in the `/data/preprocessed_data/color_corn/` folder. Please use this when training your models. The preprocessed numpy files are:
    * train_data.npy
    * train_labels.npy
    * val_data.npy
    * val_labels.npy
    * test_data.npy
    * test_labels.npy
## How to run your google colab notebook using data from a shared folder
1. From Google Drive, click "Shared with me" on the left side.
1. Right click on "634_Deep_Learning_Project_Shared" and select "Organize" --> "Add shortcut" so that it will be easily accessible from your drive.
1. Open a notebook with google colab.
1. Click on the "Files" folder icon on the left side.
1. Click on the "Mount Drive" icon to connect to google drive. The "634_Deep_Learning_Project_Shared" folder should be there.
1. Load the dataset from the `/data/preprocessed_data/color_corn/` folder and proceed with model building. See the `Amadeo_Baseline_Corn.ipynb` notebook for example code.

## Next steps
Now that we have our preprocessed data and baseline model, we'll need to:
* Conduct experiments by building more sophisticated deep learning models/architectures
* **Use class weights passed into the cross entropy loss function when training these models** to address class imbalance. We'll need to do this for all experiments. Here's an example article for how to do this: https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75
* **Use F1 score as our primary metric** (because of class imbalance) for selecting our best model. See baseline notebook for example code.
* Experiment with hyperparameter tuning. Instead of grid search we can use bayesian optimization (much faster).
* Calculate confidence intervals.
* Implement early stopping to prevent overfitting.
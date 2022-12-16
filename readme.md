# Code structure
Most of the core matlab code was written as a part of the origianl 2017 physionet contest by the blackswan group that won the competition.
Some of the matlab scripting and all of the python code was written by @amirkx and @arahs_vashagh.

## R-R Extraction
We have used the R-peak detection code from blackswan  to speed up the development of our algorithm. 
In the first part of our work we extract the R-peaks and stor them in a matlab cell. This is done by the **HistogramRrInterval.m** 


## Preprocessing and image-Construction
In this part we load the R-R intervals in python and  create the preprocessed Poincare images.
The relevant code is in **preprocessing.ipynb**

## Image augmentation and classification.
In this part we used a CNN to classify the data to AF and not-af.
Both of these steps are combined in the **augment-and-classify.ipynb**

# Code execution
## matlab code
run the **HistogramRrInterval.m** script upload the two output files to google cloud for google colab use
upload the following files to google drive
* NormalRPeaks
* AfRPeaks
## python code
run the following scripts in order in a google colab enviroment
* **preprocessing.ipynb**
* **augment-and-classify.ipynb**




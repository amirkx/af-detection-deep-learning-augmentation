# Short summary & Block diagram
This repo is the accompanied by the paper --------. This worked tried to convert a 1-D image to a 2-D image and then use a form of GANs to augment data. The block diagram below depicts the big picture of what we have done . For more details visit the paper.

![Block diagram](/figures/block-diagram.pdf)


# Installing requirement
## Matlab 

## python
install the requirements using

```
pip install -r requirements.txt
```




# Code structure
Most of the core matlab code was written as a part of the origianl 2017 physionet contest by the blackswan group that won the competition.
Some of the matlab scripting and all of the python code was written by the authors of the paper.




## R-R Extraction
We have used the R-peak detection code from blackswan  to speed up the development of our algorithm. 
In the first part of our work we extract the R-peaks and stor them in a matlab cell. This is done by the **HistogramRrInterval.m** 


## Preprocessing and image-Construction
In this part we load the R-R intervals in python and  create the preprocessed Poincare images.
The output of samples preprocessing images are shown bellow.**preprocessing.ipynb**

![Normal image](/figures/af.png)
![AF image](/figures/normal.png)
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
run the following scripts in order in a google colab environment
* **preprocessing.ipynb**
* **augment-and-classify.ipynb**




# Machine Learning : Classification and Regression 

This repository covers the analysis of datasets using various Classification and Regression algorithms with varying complexities. Tha analysis is performed in R as well as in Python using Ipython Notebooks(.ipynb files). 

A working knowledge of R and Python is required to read through the scripts.

## Classification

A Classification Algorithm is a procedure for selecting a class from a set of alternatives that best fits a set of observations. An example would be predicting whether a customer would buy a certain product or not making use of the various shopping habits of the customers or assigning a diagnosis of a disease to a given patient as described by observed characteristics of the patient(gender, blood pressure, presence or absence of certain symptoms, etc.).

Here's a list of notebooks illustrating classification:


1.  [Occupancy Detection Using Various Classification Methods using Python](http://nbviewer.jupyter.org/github/sinju-pau/Machine-Learning-Classification-and-Regression/blob/master/Occupancydetector-C.ipynb)

2.  [Magic Gamma Telescope : Predicting Signal Or Background - Classification using R](http://nbviewer.jupyter.org/github/sinju-pau/Machine-Learning-Classification-and-Regression/blob/master/magicgamma.ipynb)

3.  [Iris Data Classification (Multinomial) using Python](http://nbviewer.jupyter.org/github/sinju-pau/Machine-Learning-Classification-and-Regression/blob/master/IrisDataClassification.ipynb)

4.  [Glass Data Identification : Multinomial classification using R](http://nbviewer.jupyter.org/github/sinju-pau/Machine-Learning-Classification-and-Regression/blob/master/GlassIdentificationData.ipynb)

5.  [Credit Card Default predictions : XGBoost algorithm using R](http://nbviewer.jupyter.org/github/sinju-pau/Machine-Learning-Classification-and-Regression/blob/master/xGBoostOnCreditDefault.ipynb)


## Regression 

Regression is a statistical measure used in finance, investing and other disciplines that attempts to determine the strength of the relationship between one dependent variable (usually denoted by Y) and a series of other changing variables (known as independent variables).

Here's a list of notebooks highlighting regression:

1.  [MoveHub City Rankings Prediction - (Linear Regression and Support Vector Regression) - using Python](http://nbviewer.jupyter.org/github/sinju-pau/Machine-Learning-Classification-and-Regression/blob/master/Movehubcityrankings_m.ipynb)

2.  [Energy efficiency Analysis : Heating Load & Cooling Load predictions -(Various regression methods) - using R](http://nbviewer.jupyter.org/github/sinju-pau/Machine-Learning-Classification-and-Regression/blob/master/Energyefficiency.ipynb)

3.  [Predicting Housing Prices for Boston Cities Data - using Python](http://nbviewer.jupyter.org/github/sinju-pau/Machine-Learning-Classification-and-Regression/blob/master/BostonHousing.ipynb) 

4.  [TWA800 Flight Data Recorder : Predicting Pressure on Aircraft - using R](http://nbviewer.jupyter.org/github/sinju-pau/Machine-Learning-Classification-and-Regression/blob/master/TWA800FDR.ipynb) 


## Install

**For Python**

This repository requires **Python 3.6** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

**For R**

This repository requires **R** and the installation instructions and support can be obtained at :[R](https://cran.r-project.org/doc/manuals/r-release/R-admin.html)

To install the libraries in R, appropriate codes will be given at the beginning of the notebook

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

It is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that Python 3.6 installer is selected. 

### Code

Template code is provided in the `.ipynb` notebook files. Note that separate files such as .py or .R are not included in the repository. You can jump to the notebooks clicking on the links in appropriate sections.

### Run

Download the required notebooks. In a terminal or command window, navigate to the top-level project directory(that contains this README) and run one of the following commands:

```bash
ipython notebook notebookname.ipynb
```  
or
```bash
jupyter notebook notebookname.ipynb
```

This will open the Jupyter Notebook in your browser.

### Data

The data files may required to be downloaded through the source links provided in the notebook and saved in the same Directory as the notebook file

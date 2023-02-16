Welcome to Predictive Comb Growth Models 3D

This is a repository for simulation-based models that predict how Honey bee nests grow in 3 dimension as part of a manuscript by Marting et al. 2023 (submitted to Proc. B.) entitled "Manipulating nest architecture reveals three-dimensional building strategies and colony resilience in honey bees." 

The repository consists of three parts - the data (containing comb images and positional info), the functions (including the predictive models), and the sample notebook (that runs the models and plots the result).



Data

CombMaskArray.npy is a large numpy array containing all comb images from all colonies at all timepoints. Empty values are "0" and comb values are "2."

CombDataframe.csv is a dataframe that include detailed information about each comb. Most importantly for the models: "beeframe" is the frame id, "position" refers to the frame's present position withing the hive box (10 is near the entrance and 1 is at the back), and "inverted" refers to the orientation of the frame (which direction that the frame is facing within the box).


Functions

Functions_CombGrowthModels.py contains the 3 models (dilation, neighbor, and random placement) with detailed annotations of how they work


Sample notebook

RunModels.ipynb contains a simple script that imports the data files and model functions to run the models and plot the results. Note that you will have to change the directories to link the files to your local machine.
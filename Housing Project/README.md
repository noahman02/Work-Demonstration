# Data Science 3 with R Final Project (STAT 301-3)

This repository sets up the basic structure for your final project which will display/contain all of your work.

You are welcome to create/remove subdirectories as you see fit, but stay as organized as possible so others will be able to navigate and inspect your work.

Likely, you will have your Data Memo, EDA Project Update, executive summary, and final report all accessible at this level.

This Repository can be broken down into 25 directories which includes three folders, the README document, and the .gitignore file.
We will provide an itemized breif description of what can be found in each file/directory.

The objects folder holds any objects and data structures that were used or thought to be used in the final report,eda, or executive summary.
Most of the objects in this file are the tuned objects found, original recipes, or data splits.

The Data folder contains codebooks and processed and unprocessed data files with a citation to where the data was found

The model_info folder contains objects that were used to create most models. For instance, this includes the two recipes included and the final model for our best model the ensemble model. 

The EDA.rmd and EDA.html files contain the same information. This was our general EDA update assigned a couple months ago. Some information was updated in the final report, but this document showcases our initial visualizations of the data, and it influenced the way we used the data in our models. 

The general_setup.R script contains all the recipes and data cleaning done to make our final models. Additionally, it contains how we split and folded the data. We used two recipes for our data

All_tuned_models.R contains the initial tuning models from the first 7 models we tested. This was used so we could run all of the models easily overnight in one place. This is poorly named because at the time it was all the tuning models, but we ended up adding a couple more models before calling it quits so please look at the individual scripts. This script was simply for ease of calculating

bt_model and bt_model_2 contains our two attempts at fitting a Boosted tree model to the data

data memo.rmd and .html contains the original data memo we submitted at the beginning of the quarter

en_model contains our attempt at fitting an elastic net model to the data. 

ensemble_model contains our attempt at fitting an ensemble model of a svm rbf, random forest, and boosted tree

executive_summary.rmd and .html contains our executive summary of our final report

final_report.rmd and .html contains our final report

kn_model.r contains our attempt at fitting a kn model to the data

mars_model.r contains our attempt at fitting a mars model to the data

null_model.r contains our attempt at fitting a null model to the data. This served merely as a benchmark to compare our model to 

rf_model and rf_model_2 contains our attempt at fitting a random forest to two separate recipes

slnn_model contains our attemopt of fitting a single layer neural network to the data

svmrbf_model and svmrbf_model_2 contains our attempt to fit two support vector machine models to the data


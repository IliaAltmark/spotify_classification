# Spotify Classification Project

In this project I was working on creating a classifier which predicts if the
song is popular or not. The potential use case for this project is to use this
model to determine if a new song going to be a hit or not based only on
features of the song and nothing else.

## Notebooks

Probably the first place worth checking. The first notebook contains code for
extracting data using the spotify API. In the code I was scraping Techno
tracks, that way the classification is focused only on one genre.

The second notebook contains EDA, all the preprocessing and modeling. The EDA
mainly contains showing distributions as well correlations etc. The
preprocessing mainly involves replacing NaNs, dealing with outliers and feature
engineering necessary for building a binary classifier.

## src

The folder contains python scripts for reproduction of the results.
It's important to note that the directory does not contain the scraped data 
and some fields and variables need to be replaced (spotify API credentials, 
data address).

## Results

The best results were achieved using the Catboost model with final results 
of 0.65 AUC on the test set.

## Overall

The main idea was to work on a relatively simple traditional ML.
I think that the best model have achieved decent (better than random, 
where AUC of random predictions is 0.5) results considering the 
data and also proves there's value in analyzing the differences between 
various songs and tracks.
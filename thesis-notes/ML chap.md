present ml as the solution since i dont wanna learn myself complex relationships that will help me make the data go from their raw form to grvi to corrected grvi etc

# Model Architecture Selection and Hyper Parameter Tuning

I selected a few models to try for this tasks, few of them because they where known to be the best at it (ensembles) and one of them that i know that it sucks but i know that it will be really light weight and can be light weight-er. Tested using one dataset (notrees - dl) as:

1. has the least source of error
2. finding out gt was expected to be harder than finding out the dl conditions - and i was right

this dataset was overall a nice baseline, and tuning the hyper parameters and models around it turned out nice, since nothing better could be done for the other datasets

#### Classical ml

i can probably add other models like svm and linear/ridge/lasso

talk about how we tend to check these kinds of models and tune them, talk about grid search etc and finding which iteration had the best performance

#### Neural net

talk about the whole optimizations cocktail and things that worked and things that did not


# Testing the Rest of the Datasets

see how the rest of the datasets performed for the various models. can also claimed that you tried to restart the hyper parameter tuning thing but nothing worked

present the results

# Feature Engineering and ML and performance and stuff

azimuth, feature importance, correlation , try to cut off features and stuff

# Hard-Gated Mixture of Experts for Shadows Handling

say that the best possible outcome is that the azimuth + trees - gt - real + notrees - gt dataset works well - push harder to achieve the best results or whatever

- see the performance of the on the shadows alone, after the model has been trained on all of the dataset
- try to make the two models that specialize on stuff

probably select the experts to be the a balance between fast and precise
# generic ahh

probably end up comparing the best predictive with the fastest and propose what would be best between them.

grvi maps

metrics

idk what else
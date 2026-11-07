present ml as the solution since i dont wanna learn myself complex relationships that will help me make the data go from their raw form to grvi to corrected grvi etc

# Autogluon Models

the models i selected came from here
explain that it includes a bunch of models and i selected the best performing

# Model Architecture Selection and Hyper Parameter Tuning

I selected a few models to try for this tasks, few of them because they where known to be the best at it (ensembles) and one of them that i know that it sucks but i know that it will be really light weight and can be light weight-er. Tested using one dataset (notrees - dl) as:

1. has the least source of error
2. finding out gt was expected to be harder than finding out the dl conditions - and i was right

this dataset was overall a nice baseline, and tuning the hyper parameters and models around it turned out nice, since nothing better could be done for the other datasets

#### Classical ml

talk about how we tend to check these kinds of models and tune them, talk about grid search etc and finding which iteration had the best performance

apart from images from grid search and whatever (include code snipets of the grid search so that the hyper parameter priority  is visible) include final! performance and inference time per example, batch and full val dataset:

- random forests: $r^2 = 0.8222944736480713, RelativeRMSE = 6.911232948303223\%, RMSE=0.011174358522445391$
- extra trees: $r^2 = 0.8931569937386239, RelativeRMSE =5.358931064605713\% RMSE=0.008664534675717765$
- xgb: $r^2 = 0.910967230796814, RelativeRMSE =4.891928672790527\% RMSE=0.007909466388137028$
- lightGBM: $r^2 = 0.9054953781904455, RelativeRMSE =45.040012836456299\% RMSE=0.008148894940155768$

inference time:

single examples and validation dataset: 32962

mean of 100 iterations

**IF NEEEDED**
check what chatGPT said about inference time, and the model instance being reconstructed every time etc

Random Forest:
- single example:
	- GPU: 0.00039853810449130833 (5.9837013395735994e-05)
	- CPU: 0.0020688700024038553 (4.8321606300305575e-05)
- dataset:
	- GPU: 0.007826335728168488 (0.001100062858313322)
	- CPU: 0.46738559007644653 (0.01479120459407568)


Extra Trees:
- single example:
	- GPU: 0.0004239773843437433 (0.0001291313674300909)
	- CPU: 0.0020498300436884165 (6.727404979756102e-05)
- dataset:
	- GPU: 0.0021570848766714334 (0.00025655628996901214)
	- CPU: 0.25495854020118713 (0.0035562042612582445)

XGB:
- single example:
	- GPU: 0.0007660889532417059 (0.00019672968483064324)
	- CPU: 0.0017526388401165605 (0.0022639408707618713)
- dataset:
	- GPU: 0.00652485154569149 (0.000941976613830775)
	- CPU: 0.3232285678386688 (0.05203657224774361)

LightGBM:
- single example:
	- GPU: 0.0006278514629229903 (0.00011525979061843827)
	- CPU: 0.0005039357929490507 (0.00017951155314221978)
- dataset:
	- GPU: 0.003819601610302925 (0.0008174611139111221)
	- CPU: 0.24575752019882202 (0.0031943246722221375)
#### Neural net

talk about the whole optimizations cocktail and things that worked and things that did not

show how predictive performance increased:

- vanilla: $r^2 = 0.144504$ , $RelativeRMSE = 15.164009094238281\%$, $RMSE= 0.024518$
- adamW: same as above
- adamW + L2: $r^2 = 0.470946, RelativeRMSE = 11.924910545349121\%, RMSE=0.019281$
- adamW + L2 + BN: $r^2 = 0.524674$ , $RelativeRMSE = 11.303187370300293\%$ , $RMSE=0.018275$
- adamW + L2 + BN + SS: $r^2 = 0.809704$ , $RelativeRMSE = 7.1518778800964355\%$, $RMSE=0.011563$

Total parameters: 10,113
Trainable parameters: 10,113

Batch size in training and inference: 512

when measuring time on gpu:

	torch.cuda.synchronize()
	start_time = time.time()
	
	torch.cuda.synchronize()
	end_time = time.time()

I am measuring:

single - 512 batch - 32962 dataset

times - 100 iterations of the same experiment + gpu warm-up first:
-  single example
	- GPU: $T_{CPU\rightarrow GPU} + T_{ForowardPass} + T_{GPU\rightarrow CPU} + T_{GPUsync} + PythonOverheads$
		- $mean = 0.0004281521$ 
		- $std = 0.00012665572$
	- CPU: $T_{ForwardPass}+ PythonOverhead$
		- $mean = 0.00013796568$
		- $std = 2.7969783e-05$
- batch example
	- GPU: $T_{DataLoader} + T_{CPU\rightarrow GPU} + T_{ForowardPass} + T_{GPU\rightarrow CPU} + T_{GPUsync} + PythonOverheads$
		- $mean = 0.0016903615$
		- $std = 0.0011833189$
	- CPU: $T_{DataLoader}+T_{ForwardPass}+ PythonOverhead$
		- $mean: 0.0015396642$
		- $std = 0.0009980695$
- dataset example
	- GPU: $T_{DataLoader} + T_{CPU\rightarrow GPU} + T_{ForowardPass} + T_{GPU\rightarrow CPU} + T_{GPUsync} + PythonOverheads$
		- $mean = 0.1599856$
		- $std = 0.039122
	- CPU: $T_{DataLoader}+T_{ForwardPass}+ PythonOverhead$
		- $mean = 0.15765604$
		- $std = 0.03884833$
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
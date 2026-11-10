present ml as the solution since i dont wanna learn myself complex relationships that will help me make the data go from their raw form to grvi to corrected grvi etc

should probably mention that the main datasets that we care about in the end are the combined datasets with shifted illuminance values, because they truly test everything that there is to be tested, and checks the models robustness to everything
- soil
- sun-camera-field geometry
- shadows
- calculation mistakes

the rest are more for ruling out "weaker" models

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

model complexity:

|       |       |         |          |             |            |            |
| ----- | ----- | ------- | -------- | ----------- | ---------- | ---------- |
| Model | Trees | Leaves  | Nodes    | Leaves/Tree | Nodes/Tree | Size (MiB) |
| rf    | 50    | 5916411 | 11832772 | 118328.22   | 236655.44  | 812.506    |
| et    | 50    | 1163404 | 2326758  | 23268.08    | 46535.16   | 159.78     |
| xgb   | 500   | 2167968 | 4335436  | 4335.94     | 8670.87    | 140.894    |
| lgbm  | 450   | 114750  | 229050   | 255         | 509        | 10.666     |

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

rf specific:
single example:
CPU to GPU time: 7.392645056825131e-05 (6.577273597940803e-05)
GPU time: 0.00032463311799801886 (6.722988473484293e-05)
GPU to CPU time: 2.4030208805925213e-05 (1.6033698102546623e-06)
Overall: 0.0004266810428816825 (0.00010794104309752584)
CPU time: 0.0020411896985024214 (6.923091859789565e-05)

dataset:

CPU to GPU time: 0.00029955626814626157 (4.316122431191616e-05)
GPU time: 0.007466115988790989 (0.000664730672724545)
GPU to CPU time: 4.755497138830833e-05 (6.646057954640128e-06)
Overall: 0.007817531004548073 (0.000693393056280911)
CPU time: 0.4595220685005188 (0.009039521217346191)


Extra Trees:
- single example:
	- GPU: 0.0004239773843437433 (0.0001291313674300909)
	- CPU: 0.0020498300436884165 (6.727404979756102e-05)
- dataset:
	- GPU: 0.0021570848766714334 (0.00025655628996901214)
	- CPU: 0.25495854020118713 (0.0035562042612582445)

et specific
single:
CPU to GPU time: 0.0001155877107521519 (0.00011732542043318972)
GPU time: 0.00043615102185867727 (0.00017068878514692187)
GPU to CPU time: 2.8905868020956405e-05 (7.19772378943162e-06)
Overall: 0.0005857085925526917 (0.00023842307564336807)
CPU time: 0.0021638416219502687 (0.00010840665345313028)

dataset:
CPU to GPU time: 0.0002919435501098633 (2.556917752372101e-05)
GPU time: 0.0018506837077438831 (0.0002764952077995986)
GPU to CPU time: 4.75835804536473e-05 (5.783917004009709e-06)
Overall: 0.002194361761212349 (0.00028649767045862973)
CPU time: 0.25257185101509094 (0.0022929643746465445)

XGB:
- single example:
	- GPU: 0.0007660889532417059 (0.00019672968483064324)
	- CPU: 0.0017526388401165605 (0.0022639408707618713)
- dataset:
	- GPU: 0.00652485154569149 (0.000941976613830775)
	- CPU: 0.3232285678386688 (0.05203657224774361)

xgb
single:
CPU to GPU time: 0.00010002613271353766 (0.00012399711704347283)
GPU time: 0.0006320762913674116 (0.00018294503388460726)
GPU to CPU time: 2.7301312002236955e-05 (1.0118137652170844e-05)
Overall: 0.0007641625124961138 (0.0002885177673306316)
CPU time: 0.0009870600188151002 (0.0012086298083886504)

dataset:
CPU to GPU time: 0.0002920651459135115 (3.4123295336030424e-05)
GPU time: 0.005729017313569784 (0.0006666273111477494)
GPU to CPU time: 4.726171391666867e-05 (5.341305495676352e-06)
Overall: 0.006072468589991331 (0.0006848267512395978)
CPU time: 0.31043750047683716 (0.043441686779260635)

LightGBM:
- single example:
	- GPU: 0.0006278514629229903 (0.00011525979061843827)
	- CPU: 0.0005039357929490507 (0.00017951155314221978)
- dataset:
	- GPU: 0.003819601610302925 (0.0008174611139111221)
	- CPU: 0.24575752019882202 (0.0031943246722221375)

lgbm:
single:

CPU to GPU time: 8.685350621817634e-05 (0.00010067083348985761)
GPU time: 0.000529489538166672 (0.00013394361303653568)
GPU to CPU time: 2.4287701307912357e-05 (2.327398760826327e-06)
Overall: 0.0006447338964790106 (0.0001935513864737004)
CPU time: 0.00042053937795571983 (5.786344991065562e-05)

dataset:

CPU to GPU time: 0.0002760601055342704 (9.639737072575372e-06)
GPU time: 0.0033007145393639803 (0.00044616381637752056)
GPU to CPU time: 4.582881956594065e-05 (4.328013801568886e-06)
Overall: 0.003626532619819045 (0.00045239226892590523)
CPU time: 0.24273483455181122 (0.0019959646742790937)

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

specific:
CPU to GPU time: 6.874561222502962e-05 (0.00012574910942930728)
GPU time: 0.00029737234581261873 (0.0001389319368172437)
GPU to CPU time: 3.451108932495117e-05 (7.910710337455384e-06)
Overall: 0.0004194402717985213 (0.00020962663984391838)


- batch example
	- GPU: $T_{DataLoader} + T_{CPU\rightarrow GPU} + T_{ForowardPass} + T_{GPU\rightarrow CPU} + T_{GPUsync} + PythonOverheads$
		- $mean = 0.0016903615$
		- $std = 0.0011833189$
	- CPU: $T_{DataLoader}+T_{ForwardPass}+ PythonOverhead$
		- $mean: 0.0015396642$
		- $std = 0.0009980695$


dataloader: 0.00140746112447232 (0.0010996224591508508)
CPU to GPU time: 4.64320182800293e-05 (7.671524508623406e-05)
GPU time: 0.00023259878798853606 (6.48210261715576e-05)
GPU to CPU time: 3.521680991980247e-05 (8.547250217816327e-06)
Overall: 0.0017426132690161467 (0.0011511057382449508)

- dataset example
	- GPU: $T_{DataLoader} + T_{CPU\rightarrow GPU} + T_{ForowardPass} + T_{GPU\rightarrow CPU} + T_{GPUsync} + PythonOverheads$
		- $mean = 0.1599856$
		- $std = 0.039122$
	- CPU: $T_{DataLoader}+T_{ForwardPass}+ PythonOverhead$
		- $mean = 0.15765604$
		- $std = 0.03884833$

DataLoader:  0.1443 ± 0.0389 s
CPU -> GPU:  0.0022 ± 0.0010 s
Inference:   0.0137 ± 0.0005 s
GPU -> CPU:  0.0025 ± 0.0000 s
Total:       0.1646 ± 0.0389 s

# Testing the Rest of the Datasets

see how the rest of the datasets performed for the various models. can also claimed that you tried to restart the hyper parameter tuning thing but nothing worked

present the results

NN:
 - notrees-gt: $r^2 = 0.758445, RelativeRMSE = 10.201085090637207\%, RMSE=0.030603$
 - trees-dl-aoi: $r^2 = 0.568878, RelativeRMSE = 10.720009803771973\%, RMSE=0.017333$
 - trees-gt-aoi: $r^2 = 0.471997, RelativeRMSE = 15.047739028930664\%, RMSE=0.045143$

RF:
 - notrees-gt: $r^2 = 0.8246005773544312, RelativeRMSE = 8.69265079498291\%, RMSE=0.02607795241230503$
 - trees-dl-aoi: $r^2 = 0.8131389617919922, RelativeRMSE = 7.057559967041016\%, RMSE=0.011410946211964342$
 - trees-gt-aoi: $r^2 = 0.8284494876861572, RelativeRMSE = 8.577266693115234\%, RMSE=0.025731801119633323$

ET:
- notrees-gt: $r^2 = 0.8901348556047691, RelativeRMSE = 6.8796772956848145\%, RMSE=0.020639033623492373$
 - trees-dl-aoi: $r^2 = 0.8993802511975696, RelativeRMSE = 5.178892135620117\%, RMSE=0.008373440390905498$
 - trees-gt-aoi: $r^2 = 0.9254036507615433, RelativeRMSE = 5.656030178070068\%, RMSE=0.016968091965759702$

XGB:
- notrees-gt: $r^2 = 0.8970029354095459, RelativeRMSE = 6.66117000579834\%, RMSE=0.019983511451385216$
 - trees-dl-aoi: $r^2 = 0.9178245067596436, RelativeRMSE = 4.680222034454346\%, RMSE=0.007567170072755924$
 - trees-gt-aoi: $r^2 = 0.9058482646942139, RelativeRMSE = 6.35429048538208\%, RMSE=0.019062872413022625$

LGBM:
- notrees-gt: $r^2 = 0.8933568256756083, RelativeRMSE = 6.778048515319824\%, RMSE=0.020334145416163345$
 - trees-dl-aoi: $r^2 = 0.9005502259631895, RelativeRMSE = 5.1486945152282715\%, RMSE=0.008324616183264502$
 - trees-gt-aoi: $r^2 = 0.8926947151422432, RelativeRMSE = 6.783651351928711\%, RMSE=0.020350954274155172$

Overall: predictive performance wise, NN and RF are the worst.  NN inference on CPU for single examples is the fastest, probably making it more suitable as complementary addition to real time systems deployment (also, check here whether the model is overall lighter than the rest).  Random forests also had the second lowest inference time for single examples, but the highest dataset inference time across the ensemble methods. Its single example performance was also comparable to the Extra Trees inference time, which for this x% increase of time (check also model size) achieves one of the highest predictive performances across datasets, being the best performing model for gt values - with trees present. LGBM and XGB provide consistently good predictive performance across the datasets, but also the highest inference times, apart from the NN.

Moving on, it was clear that the NN and the RF cannot perform as well as the others and were not included for the testing of the rest of the datasets.  

note*: rf model size sucks, i could even skip it in the azimuth section


The performance of RF might be re-evaluated in  the azimuth section

#### Complete datasets testing

ET:
- combined-dl-aoi: $r^2 = 0.9097982989812959, RelativeRMSE = 4.929864883422852\%, RMSE=0.007970802768542617$
- combined-gt-aoi: $r^2 = 0.9328126711475695, RelativeRMSE = 5.4058990478515625\%, RMSE=0.01621769782287153$

XGB:
- combined-dl-aoi: $r^2 = 0.9367978572845459, RelativeRMSE = 4.126611232757568\%, RMSE=0.006672070459210373$
- combined-gt-aoi: $r^2 = 0.9314531683921814, RelativeRMSE = 5.460318565368652\%, RMSE=0.016380955897029013$

LGBM:
- combined-dl-aoi: $r^2 = 0.9068525227062489, RelativeRMSE = 5.009716987609863\%, RMSE=0.008099911017792187$
- combined-gt-aoi: $r^2 = 0.8981810536622243, RelativeRMSE = 6.6548542976379395\%, RMSE=0.019964563334137205$

# Feature Engineering and ML and performance and stuff

azimuth, feature importance, correlation , try to cut off features and stuff

you could say something like "not all f.e. solutions are for each scenario". For example, illuminance is more important in the "trees" datasets than in "notrees" datasets. (this exact example can be revisited in the shadows section)

#### Azimuth

RF:
- notrees-dl: $r^2 = 0.7338420152664185, RelativeRMSE = 8.458139419555664\%, RMSE=0.013675458791743993$
- notrees-gt: $r^2 = 0.6553020477294922, RelativeRMSE = 12.185895919799805\%, RMSE=0.03655768945624921$
- combined-gt-aoi: $r^2 = 0.684541642665863, RelativeRMSE = 11.713730812072754\%, RMSE=0.03514119494797767$
- combined-dl-aoi: $r^2 = 0.7489066123962402, RelativeRMSE = 18.225176811218262\%, RMSE=0.013298795725349774$

ET:
- notrees-dl: $r^2 = 0.7313269314452457, RelativeRMSE = 8.49800968170166\%, RMSE=0.013739921838732974$
- notrees-gt: $r^2 = 0.651040048923726, RelativeRMSE = 12.261000633239746\%, RMSE=0.03678300335962459$
- combined-gt-aoi: $r^2 = 0.6305590907656169, RelativeRMSE = 12.676424026489258\%, RMSE=0.03802927173467967$
- combined-dl-aoi: $r^2 = 0.6067907010319105, RelativeRMSE = 10.29294204711914\%, RMSE=0.016642041058613655$

XGB:
- notrees-dl: $r^2 = 0.7965010404586792, RelativeRMSE = 7.395819187164307\%, RMSE=0.011957856014155837$
- notreess-gt: $r^2 = 0.725448489189148, RelativeRMSE = 10.875520706176758\%, RMSE=0.03262656135689245$
- combined-gt-aoi: $r^2 = 0.7765100002288818, RelativeRMSE = 9.85946273803711\%, RMSE=0.02957839118634464$
- combined-dl-aoi: $r^2 = 0.8259682655334473, RelativeRMSE = 6.847654819488525\%, RMSE=0.011071563264402683$

LGBM:
- notrees-dl: $r^2 = 0.8044831679159515, RelativeRMSE = 7.249318599700928\%, RMSE=0.011720989323942126$
- notreess-gt: $r^2 = 0.735572287139346, RelativeRMSE = 110.673125267028809\%, RMSE=0.032019376988922196$
- combined-gt-aoi: $r^2 = 0.7337225428136276, RelativeRMSE = 10.761961936950684\%, RMSE=0.032285886597183046$
- combined-dl-aoi: $r^2 = 0.7922375807936254, RelativeRMSE = 7.481886863708496\%, RMSE=0.012097014355513748$


complexity for the azimuth models:

|   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|
|Model|Trees|Leaves|Nodes|Leaves/Tree|Nodes/Tree|Size (MiB)|
|rf|50|6279955|12559860|125599.1|251197.2|862.431|
|et|50|1229075|2458100|24581.5|49162|168.798|
|xgb|500|1946224|3891948|3892.45|7783.9|126.514|
|lgbm|450|114750|229050|255|509|10.684|

inference time:

Random Forest:
- single example:
	- GPU: 0.0003871750959660858 (7.889429980423301e-05)
	- CPU: 0.001979069784283638 (6.400006532203406e-05)
- dataset:
	- GPU: 0.007055254187434912 (0.00011759697372326627)
	- CPU: 0.4984036684036255 (0.017949312925338745)

Extra Trees:
- single example:
	- GPU: 0.0004070162831339985 (0.00014265561185311526)
	- CPU: 0.0020604897290468216 (8.637968130642548e-05)
- dataset:
	- GPU: 0.0019647693261504173 (0.0002526963362470269)
	- CPU: 0.2526671588420868 (0.0036032618954777718)

XGB:
- single example:
	- GPU: 0.0006333613418973982 (0.00010750722867669538)
	- CPU: 0.000607290246989578 (0.00022872611589264125)
- dataset:
	- GPU: 0.005461404100060463 (0.0004590919124893844)
	- CPU: 0.29193851351737976 (0.030498681589961052)

LightGBM:
- single example:
	- GPU: 0.0006265187403187156 (0.00014072805061005056)
	- CPU: 0.00041615008376538754 (6.142286292742938e-05)
- dataset:
	- GPU: 0.003632235573604703 (0.00030568562215194106)
	- CPU: 0.2514398694038391 (0.003982536029070616)

by now you can probably choose a best performing model ig


#### Feature Importance

datatasets:

- notrees-dl:
et:

|   |   |   |   |
|---|---|---|---|
||feature|importance|std|
|2|meanRefG|0.0196472493620541|7.97633168396965E-05|
|1|meanRefR|0.0174968362768815|8.1310488383514E-05|
|8|camPitch|0.0116354462219314|6.29762174117958E-05|
|9|camYaw|0.010756709323559|8.46379062213527E-05|
|0|grviIn|0.00643965657154995|6.31937933985979E-05|
|5|blockIdx|0.00353156967708948|5.32646131815115E-05|
|6|sunPitch|0.00117922791521785|2.97306239115679E-05|
|4|illumG|0.000512439844011142|5.30214978422526E-06|
|7|sunYaw|0.000488221200143887|1.06048345334162E-05|
|3|illumR|0.000322192255196669|5.4312666105527E-06|
xgb:

|   |   |   |   |
|---|---|---|---|
||feature|importance|std|
|2|meanRefG|0.0390967352315784|0.000110715807765115|
|1|meanRefR|0.035209041275084|9.49245471677472E-05|
|0|grviIn|0.012078452296555|8.7821441149771E-05|
|8|camPitch|0.0103215241804719|4.30594336676848E-05|
|9|camYaw|0.00702453386038542|7.24975829178679E-05|
|6|sunPitch|0.00402128687128425|5.50386313554963E-05|
|5|blockIdx|0.00343027887865901|3.24422920908572E-05|
|3|illumR|0.0033384514041245|3.57635429418943E-05|
|7|sunYaw|0.00081673339009285|2.26553931562736E-05|
|4|illumG|9.64438542723656E-05|8.40960910522187E-06|
lgbm:

|   |   |   |   |
|---|---|---|---|
||feature|importance|std|
|2|meanRefG|0.0401205680525046|0.000101658376798924|
|1|meanRefR|0.0326250540327632|8.29166454511282E-05|
|0|grviIn|0.0116395109613854|8.32186035803551E-05|
|8|camPitch|0.00930580857958862|4.75762697555625E-05|
|9|camYaw|0.00624329685172557|6.13705063649132E-05|
|5|blockIdx|0.00298296158411176|2.02262230985535E-05|
|6|sunPitch|0.00263294609175419|4.63953417567365E-05|
|3|illumR|0.00218617515082132|2.28528920093818E-05|
|7|sunYaw|0.00149959702659373|2.4980304145077E-05|
|4|illumG|0.000243421932491044|1.13191946633966E-05|

- trees-dl-aoi

et:

|   |   |   |   |
|---|---|---|---|
||feature|importance|std|
|8|camPitch|0.0194942453141403|5.27066782605777E-05|
|9|camYaw|0.015329321609445|5.56811054686577E-05|
|5|blockIdx|0.00669266426260818|0.000103898686218613|
|0|grviIn|0.00283123560411175|4.86167859139774E-05|
|2|meanRefG|0.00168965000922924|2.23927547963228E-05|
|1|meanRefR|0.00166634366832062|1.54719002582825E-05|
|7|sunYaw|0.000822288072493032|1.48362222400127E-05|
|4|illumG|0.000696637328716441|1.09374826260378E-05|
|6|sunPitch|0.000639873653455847|1.82661246623832E-05|
|3|illumR|0.000570549668412702|9.23307079004561E-06|

xgb:

|   |   |   |   |
|---|---|---|---|
||feature|importance|std|
|2|meanRefG|0.0315798310562968|0.000160498935447989|
|1|meanRefR|0.0301991811022162|9.55801981466752E-05|
|8|camPitch|0.0138970578089356|5.88602153385748E-05|
|0|grviIn|0.0126567389816046|7.36175352312894E-05|
|3|illumR|0.00946277417242527|2.26669742358523E-05|
|9|camYaw|0.00771435676142573|4.46076699449034E-05|
|5|blockIdx|0.0049509652890265|3.83259098522288E-05|
|6|sunPitch|0.00397448102012277|4.18796532321909E-05|
|4|illumG|0.00375173557549715|3.67228333522362E-05|
|7|sunYaw|0.00165519304573536|3.11969331522266E-05|

lgbm:

|   |   |   |   |
|---|---|---|---|
||feature|importance|std|
|2|meanRefG|0.0332837354465673|0.000128031701113868|
|1|meanRefR|0.0301502829726011|0.000105419377802929|
|8|camPitch|0.0123288921474894|3.45762415944264E-05|
|0|grviIn|0.0115441603134442|6.39273215544912E-05|
|9|camYaw|0.00641955559908761|3.36125290943005E-05|
|5|blockIdx|0.00393792170840097|4.26458933668414E-05|
|4|illumG|0.00323244613348664|2.87153921767997E-05|
|3|illumR|0.00264569714570608|3.06438866406425E-05|
|6|sunPitch|0.00256961034490195|3.39806253314379E-05|
|7|sunYaw|0.00192937562561792|2.77455504945248E-05|

For gt results were pretty much the same (modify the script to make plots and show it in the plots)

overall, only illums could have been removed for the notrees datasets, but this cannot be done since it is important for the model to tackle shadows. One thing that can be done is to remove illums only in the mixture of experts , specifically for the model that doesnt deal with shadows. This is a good option since illum is also correlated to the reflectances (section ####).  Sun pitch and yaw are also consistently low, but removing them cause a 1.5-2% increase in relative error, for a non noticeable performance boost.

#### PCA 

it sucks

# Hard-Gated Mixture of Experts for Shadows Handling

general thinking:
	say that the best possible outcome is that the azimuth + trees - gt - real + notrees - gt dataset works well - push harder to achieve the best results or whatever
	
	see the performance of the on the shadows alone, after the model has been trained on all of the dataset
	try to make the two models that specialize on stuff
	
	probably select the experts to be the a balance between fast and precise

the performance in the "trees-aoi" datasets so far is:

ET:
- trees-dl-aoi overall: RelRMSE: 5.156252403561166%, r2 0.9002580570424128
- trees-dl-aoi shadows: RelRMSE: 5.1812277595833%, r2 0.8933392356787375
- trees-dl-aoi clear: RelRMSE: 5.188581875168325%, r2 0.9018589223555391
- trees-gt-aoi overall: RelRMSE: 5.5694112951867325%, r2 0.9276709632621387
- trees-gt-aoi shadows: RelRMSE: 4.918658021867711%, r2 0.9375277574668405
- trees-gt-aoi clear: RelRMSE: 5.739377427146676%, r2 0.925148176268467

XGB:
- trees-dl-aoi overall: RelRMSE: 4.680221921241964%, r2 0.9178245067596436
- trees-dl-aoi shadows: RelRMSE: 4.814167834390221%, r2 0.9079164862632751
- trees-dl-aoi clear: RelRMSE: 4.679512403999937%, r2 0.9201720952987671
- trees-gt-aoi overall: RelRMSE: 6.354290551844054%, r2 0.9058482646942139
- trees-gt-aoi shadows: RelRMSE: 6.206794724854781%, r2 0.9005216360092163
- trees-gt-aoi clear: RelRMSE: 6.395215543530656%, r2 0.9070641994476318

LGBM:
- trees-dl-aoi overall: RelRMSE: 5.148694893150747%, r2 0.9005502259631895
- trees-dl-aoi shadows: RelRMSE: 5.755280331714864%, r2 0.8683950305143215
- trees-dl-aoi clear: RelRMSE: 5.01485080073613%, r2 0.9083210760475567
- trees-gt-aoi overall: RelRMSE: 6.783651155160313%, r2 0.8926947151422432
- trees-gt-aoi shadows: RelRMSE: 6.754368660929666%, r2 0.8821950982992326
- trees-gt-aoi clear: RelRMSE: 6.79187496601623%, r2 0.8951780631758615

in gt the shadows help (lol)

in the "trees-real":

ET:
- trees-dl-real overall: RelRMSE: 5.009656976143936%, r2 0.9095093402232967
- trees-dl-real shadows: RelRMSE: 5.38946113280334%, r2 0.8872306924672562
- trees-dl-real clear: RelRMSE: 4.9322931568188%, r2 0.9148882301867505
- trees-gt-real overall: RelRMSE: 5.915324706053272%, r2 0.920611841436813
- trees-gt-real shadows: RelRMSE: 6.045013297626757%, r2 0.9071365486649317
- trees-gt-real clear: RelRMSE: 5.878124720879541%, r2 0.9238292142950144

XGB:
- trees-dl-real overall: RelRMSE: 4.332823272413594%, r2 0.9323092103004456
- trees-dl-real shadows: RelRMSE: 4.626060346100006%, r2 0.9169149398803711
- trees-dl-real clear: RelRMSE: 4.276293511921108%, r2 0.9360225796699524
- trees-gt-real  overall: RelRMSE: 5.688085644031235%, r2 0.9265941381454468
- trees-gt-real shadows: RelRMSE: 5.561751644294581%, r2 0.9213907718658447
- trees-gt-real clear: RelRMSE: 5.723312010379651%, r2 0.9277886152267456

LGBM:
- trees-dl-realoverall: RelRMSE: 4.907135286238029%, r2 0.9131751907035737
- trees-dl-real shadows: RelRMSE: 5.453756886834374%, r2 0.8845239888146733
- trees-dl-real clear: RelRMSE: 4.778539111716107%, r2 0.920111889704196
- trees-gt-real overall: RelRMSE: 6.347180559036892%, r2 0.9085970408471515
- trees-gt-real shadows: RelRMSE: 6.214212547246526%, r2 0.9018653209503517
- trees-gt-real clear: RelRMSE: 6.384286625305578%, r2 0.9101463765117289

Training models specifically to clear sky / shadows:

ET:
- gt-clear: RelRMSE: 5.617874935167009%, r2: 0.9304247179430429
- gt-shadows: RelRMSE: 4.554651123550465%, r2 0.9472818049585853
- dl-clear: RelRMSE: 4.935800865748664%, r2: 0.9147671289201531
- dl-shadows: RelRMSE: 4.410055761762982%, r2 0.9211532840579851

XGB:
- gt-clear: RelRMSE: 5.862537374864134%, r2: 0.9242326617240906
- gt-shadows: RelRMSE: 3.4458631753417386%, r2: 0.9698250889778137
- dl-clear: RelRMSE: 4.159614981108993%, r2: 0.9394661784172058
- dl-shadows: RelRMSE: 3.5465524428143294%, r2: 0.9490072727203369

LGBM:
- gt-clear: RelRMSE: 5.823351046388488%, r2: 0.9252421531114401
- gt-shadows: RelRMSE: 3.5509729301295%, r2: 0.9679561254686334
- dl-clear: RelRMSE: 4.507243355078494%, r2: 0.9289254902787857
- dl-shadows: RelRMSE: 3.650086920249849%, r2: 0.9459865571244527


gated lgbm model (all lgbm) (add info later)

- dl-overall: 
- dl-clear:
- dl-shadows
- gt-overall: RelRMSE: 5.771444728501314%, r2: 0.9244268312948692
- gt-clear: High RelRMSE: 5.823351046388488%, r2: 0.9252421531114401
- gt-shadows: Low RelRMSE: 3.5509729301295%, r2: 0.9679561254686334

single:

CPU time: 0.0005032491753809154 (9.347106970380992e-05)
CPU to GPU time: 0.00017779112386051565 (0.00020069594029337168)
GPU time: 0.0013611364411190152 (0.00038817714084871113)
GPU to CPU time: 2.276897430419922e-05 (2.4389883037656546e-06)
Overall: 0.0015653776936233044 (0.0004520477377809584)

(explain why only single makes sense)

all xgb model:   (add info later)

- dl-overall: RelRMSE: 3.9293323283238997%, r2: 0.9443294774045932
- dl-clear:
- dl-shadows
- gt-overall:
- gt-clear: High RelRMSE: 5.377299370404392%, r2: 0.9362559914588928
- gt-shadows: Low RelRMSE: 3.064734773149369%, r2: 0.9761309027671814

single: 
CPU time: 0.0011606216430664062 (0.0011612743837758899)
CPU to GPU time: 8.208036160795018e-05 (8.573484956286848e-05)
GPU time: 0.0012492132373154163 (0.0002829199656844139)
GPU to CPU time: 2.3131369744078256e-05 (2.0578659132297616e-06)
Overall: 0.0013580679660663009 (0.0003035653498955071)
# generic ahh

probably end up comparing the best predictive with the fastest and propose what would be best between them.

grvi maps

metrics

idk what else
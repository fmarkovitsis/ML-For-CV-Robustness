# Data Pre-Processing
- Some data were already ready to use
- Talk a bit about them and a bit about the rest
- Structuring into the final dataset

#### Calculate GRVI
- Talk about the areas of interest per frame and how you extract boxes that contain the luminance
- Talk a bit about the non-tree and the tree case, how that affects light meters placement
- next is the calculation of GRVI. Explain the difference between using:
	$\frac{avg(G)-avg(R)}{avg(G)+avg(R)}$
	
	and:
	
	$avg(\frac{G-R}{G+R})$

#### Create the Dataset's Labels
- for the perceived diffuse conditions, we calculated the perceived GRVI before, we just use the values as labels
- for the GT assigned values, we use the camera's location each frame to calculate which part of the field is the camera looking at. we save these values in a csv and use them as labels for the 

#### Create the datasets
- Ig pretty much explain that I made these datasets that include these metrics - explain that for now no columns were removed and everything is as captured
- explain that I made the backwards dataset and included that too to the notrees etc. that can be mentioned above too

#### Shadows dataset
- explain that it was meaningful to see what happens when the light meter is attached to the camera and not floating in front of it
- explain the process of moving the light meter backwards - and what allowed you to be able to do that 
- re-calculating GRVI
- labeling where there is shadows and where there is not in order to train a shadow detection model

#### Azimuth dataset
- explain the calculations to find the azimuth

# Feature Engineering
- mention that feature engineering points will be discussed in the ML chapter. That's because they have to do with dimensionality vs performance vs feature importance.
- there things were tested live feature importance and PCA
- overall, not many things were done

#### Ignoring unimportant features
- for example features that stay the same

#### Standard scaling
- scaled both inputs and, in the cases where not the assigned values are used as labels, the labels too (show how "condensed" the range is otherwise)
- i think i have to explain that scaling is reversible
- This helped with MLP not with ensemble
To run the algorithm:
- Within MatlabSVM:
	0. Set an input folder containing images (training, validation, testing) such that the directory contains:
		- folder name
			L craters
				L Train
					L images
					L labels
				L Validation
					L images
					L labels
				L Test
					L images
					L labels

	1. Run Feature_extraction_OnlineCraters.m
		This extracts features from the input folder listed at the start of the doc

	2. Run SVM_training_Dijkstra.m
		This trains an SVM with manually determined parameters, also,
		HyperOptimization can be used here

	3. Run SVM_validation_Dijkstra.m
		This runs a validation test on the input folder (set the input folder to the respective validation sub-folder
		- Also, tune the parameters within this doc to whatever sees best consistent results (parameters being the NMS threshold, and IoU threshold)

	4. Run Retraining_SVM_Dijkstra_MATLAB.m
		Validation test incorporates hard negatives into training set, such that the next iteration takes that into account
	
	5. Run SVM_RE_Validation_Dijkstra.m
		Run this, and step 4, until the performance starts being negatively affected.

	6. Run SVM_Testing_Dijkstra.m
		This is the final test for the algorithm, use the best iteration (or all)


--------------------------------------------------------------------------------------------------------
Note that the visual results for the images contained within Crater_images are found within MatlabSVM, both validation and test.
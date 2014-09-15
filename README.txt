/////////////////////////////////////////////////////////////////////////////////////////////
///												  
///		It contains step by step execution details for all the parts of the project		  ///
///				@Author : Anirudha Karwa (akarwa@buffalo.edu)							  ///
///						      Pratik Bhat (pratiksu@buffalo.edu)							  ///
/////////////////////////////////////////////////////////////////////////////////////////////

This assignment implements Handwritten Digits Classifier using various types of Logistic Regression and compares its performance.
Binary and Multiclass Logistic Regression has been implemented using both Gradient descent and Newton Raphson approach.

Using LibSVM , we have performed classification on the dataset using SVM 
and compared the performance by tweaking the kernel function and gamma values.



---------------------Content of Code Folder-----------------------
1. blrObjFunction.m   (binary logistic regression)
2. blrPredict.m
3. blrNewtonRaphsonLearn.m (binary logistic regression using Newton Raphson method with Hessian Matrix)
4. mlrObjFunction.m  (multi class logistic rgression with gradient descent)
5. mlrPredict.m      (multi cass logistic regression using Newton Raphson method with Hessian Matrix)
6. mlrNewtonRaphson.m
7. script.m
8. params.mat       (all result variables)

---------------------Content of params.mat-------------------------
1. W_blr 		 		(weight vector for BLR Gradient Descent)
2. W_blr_Newton  		(weight vector for BLR Newton Raphson)
3. W_mlr		 		(weight vector for MLR Gradient Descent) 
4. W_mlr_Newton			(weight vector for MLR Newton Raphson)
5. model_linear			(SVM trained model for Linear Kernel)
6. model_rbf_1			(SVM trained model for Radial Basis, Gamma value = 1, Other Parameters = default)
7. model_rbf_default 	(SVM trained model for Radial Basis, Gamma value = default, Other Parameters = default)	
8. model_rbf_C			(SVM trained model for Radial Basis, Gamma value = default, C = 80 (best value))


---------------------Content of Report Folder-------------------------
1. Report.pdf

#Note:
1. All the observations , comparison and conclusions are based on validation accuracy.

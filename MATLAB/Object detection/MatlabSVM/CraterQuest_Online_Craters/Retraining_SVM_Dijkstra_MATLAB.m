%% SVM training
% Import features, then train the model
clc,clear,close
%load features.mat
load features_direct_Dijkstra.mat
load SVM_IT1_Dijkstra_linear.mat
%load SVM_IT1_Dijkstra_gaussian.mat

KernPar = svmModel.KernelParameters;
kernType = KernPar.Function;

if strcmpi(kernType,'gaussian')
    if exist("gaussian_IT1_HardNeg_featLabels_Dijkstra.mat","file")
    load gaussian_IT1_HardNeg_featLabels_Dijkstra.mat
    end
    
    if exist("gaussian_IT2_HardNeg_featLabels_Dijkstra.mat","file")
    load gaussian_IT2_HardNeg_featLabels_Dijkstra.mat
    end
    
    if exist("gaussian_IT3_HardNeg_featLabels_Dijkstra.mat","file")
    load gaussian_IT3_HardNeg_featLabels_Dijkstra.mat
    end
    

    Total_HardNeg_Features = all_hardNeg_features;
    Total_HardNeg_Labels = all_hardNeg_labels;

       
    if exist("gauss_IT2_hardNeg_features","var") && exist("gauss_IT2_hardNeg_labels","var")
            Total_HardNeg_Features = [all_hardNeg_features; gauss_IT2_hardNeg_features];
            Total_HardNeg_Labels = [all_hardNeg_labels; gauss_IT2_hardNeg_labels];
    elseif exist("gauss_IT3_hardNeg_features","var") && exist("gauss_IT3_hardNeg_labels","var")
            Total_HardNeg_Features = [all_hardNeg_features; gauss_IT2_hardNeg_features; gauss_IT3_hardNeg_features];
            Total_HardNeg_Labels = [all_hardNeg_labels; gauss_IT2_hardNeg_labels; gauss_IT3_hardNeg_labels];
    end    

elseif  strcmpi(kernType,'linear')
    if exist("linear_IT1_HardNeg_featLabels_Dijkstra.mat","file")
    load linear_IT1_HardNeg_featLabels_Dijkstra.mat
    end
    
    if exist("linear_IT2_HardNeg_featLabels_Dijkstra.mat","file")
    load linear_IT2_HardNeg_featLabels_Dijkstra.mat
    end
    
    if exist("linear_IT3_HardNeg_featLabels_Dijkstra.mat","file")
    load linear_IT3_HardNeg_featLabels_Dijkstra.mat
    end
    

        Total_HardNeg_Features = all_hardNeg_features;
        Total_HardNeg_Labels = all_hardNeg_labels;
        
        if exist("linear_IT2_hardNeg_features","var") && exist("linear_IT2_hardNeg_labels","var")
        Total_HardNeg_Features = [Total_HardNeg_Features; linear_IT2_hardNeg_features;];
        Total_HardNeg_Labels = [Total_HardNeg_Labels; linear_IT2_hardNeg_labels];
        
        elseif exist("linear_IT3_hardNeg_features","var") && exist("linear_IT3_hardNeg_labels","var")
        Total_HardNeg_Features = [Total_HardNeg_Features; linear_IT3_hardNeg_features];
        Total_HardNeg_Labels = [Total_HardNeg_Labels; linear_IT3_hardNeg_labels];
        end
end



%Normalize, and PCA-transform the data

scaled_hardNed_features = (Total_HardNeg_Features - train_mu) ./ train_sigma;
hardNeg_pca_features = scaled_hardNed_features * train_coeff(:, 1:train_num_components);

%Insert the HardNegative data into the full training
total_features = [features_pca_scaled;hardNeg_pca_features];
total_labels = [total_labels;Total_HardNeg_Labels];

Train_SVM = 1;
OptimizeParamters = 0;
Sanity_check = 0;

%linear Model (Currently best)
boxCon = 0.0050007;
kerneScale  = 2.568;
Kernel = "linear";
Iteration = 4;

%Gauss
% boxCon = 1.7577;
% kerneScale = 39.081;
% Kernel = kernType;
% Iteration = 4;


if Train_SVM
RetrainedsvmModel = fitcsvm(total_features, total_labels,'Standardize', false, 'BoxConstraint', boxCon,'KernelFunction',Kernel,'KernelScale',kerneScale)
save(sprintf("SVM_IT%i_Dijkstra_%s.mat",Iteration,Kernel),"RetrainedsvmModel");

elseif OptimizeParamters
    RetrainedsvmModel = fitcsvm(total_features,total_labels, ...
                          'OptimizeHyperparameters','auto', ...
                          'KernelFunction',Kernel)
    save(sprintf("SVM_IT%i_Dijkstra_%s.mat",Iteration,Kernel),"RetrainedsvmModel");
end
%-----------------------------------------------------------------------------------------------------------------------
%Sanity check:
if Sanity_check
    
    % Predict on Same Dataset
    labels_pred = predict(RetrainedsvmModel, features_pca_scaled);
    
    % Evaluate Accuracy
    accuracy = sum(labels_pred == total_labels) / numel(total_labels);
    fprintf('Training Accuracy (on full dataset): %.2f%%\n', accuracy * 100);
    
    % Confusion Matrix
    confMat = confusionmat(total_labels, labels_pred);
    disp('Confusion Matrix:');
    disp(confMat);
    
    % Optional Confusion Chart
    figure;
    confusionchart(total_labels, labels_pred);
    title('Confusion Matrix (Full Dataset)');

end
%% SVM training
% Import features, then train the model
clc,clear,close
%load features.mat
load features_direct_Dijkstra.mat

Train_SVM = 1;
OptimizeParamters = 0;
Sanity_check = 0;

%linear Model (Currently best)
boxCon = 0.0010528;
Kernel = "linear";

% boxCon = 1.1463;
% kerneScale = 36.386;
% Kernel = "gaussian";



if Train_SVM
svmModel = fitcsvm(features_pca_scaled, total_labels,'Standardize', false, 'BoxConstraint', boxCon,'KernelFunction',Kernel)%,'KernelScale',kerneScale)

save(sprintf("SVM_IT1_Dijkstra_%s.mat",Kernel),"svmModel");

elseif OptimizeParamters
       Best_i_test = fitcsvm(features_pca_scaled,total_labels,'OptimizeHyperparameters','auto','KernelFunction',Kernel)
end
%-----------------------------------------------------------------------------------------------------------------------
%Sanity check:
if Sanity_check
    
    % Predict on Same Dataset
    labels_pred = predict(svmModel, features_pca_scaled);
    
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
%% SVM2_Multimask_labelling_(fourier)
% This code utilizes Fourier descriptors to determine the shape based on the energy contained in the object-shape. 

%TODO: Change IOU to ~0.35, evaluate performance. 
% Consolidate Gemini about the IOU value.
% Re-train the SVM with hard-negative data after re-trying with IOU = 0.35

clc, clear, close all
Iteration = 2;
Kerneltype = 'gaussian';
% Kerneltype = 'linear';
ModelName = "SVM_IT"+num2str(Iteration)+"_Dijkstra_"+Kerneltype+".mat";
load(ModelName)
load("features_direct_Dijkstra.mat")

KernPar = RetrainedsvmModel.KernelParameters;
% KernPar = svmModel.KernelParameters;
Kernel_type = sprintf("%c",KernPar.Function);

%----------------------------------------------------------------------------------------------------------------------
% Setup
graphic_output = 1;     % outputting graphics
%----------------------------------------------------------------------------------------------------------------------
% Folder containing images and labels (not using labels, using my own
image_folder = "Crater_images\Cath_Dijkstra_set\craters\test\images";
image_files = dir(fullfile(image_folder, '*.jpg')); % Change extension if needed
image_names = {image_files.name};
sorted_images = natsortfiles(image_names);

%Load labels
label_folder = "Crater_images\Cath_Dijkstra_set\craters\test\labels";
% label_folder = "Crater_images\crater-detection_v8.v1i.yolov8\valid\labels";
label_files = dir(fullfile(label_folder,'*.txt'));
label_names = {label_files.name};
sorted_labels = natsortfiles(label_names);

% compute size of folder
num_image_files = length(image_files); %Num images = num labels


% SVM Class names [0,1] = [crater, not_crater]
% SVM_classnames = svmModel.ClassNames;
SVM_classnames = RetrainedsvmModel.ClassNames;
% SVM_classnames = unique(svmModel.labels);

% Expected size of data for the SVM
% expected_svm_features = size(svmModel.X,2); % This extracts the feature count from the first binary learner (a bit indirect but reliable)
expected_svm_features = size(RetrainedsvmModel.X,2);
fprintf('SVM model expects %d features (columns).\n', expected_svm_features);
fprintf("\n")
%----------------------------------------------------------------------------------------------------------------------
%post-prediction processing
crater_detection_threshold = 1; % Confidence threshold for 'Crater'
nms_iou_threshold = 0.1;        % IoU (Overlap) threshold for bounding boxes in NMS
IoU_threshold = 0.39;            % Intersect over Union (overlap)
Hard_Neg_IoU_threshold = 0.05;


% Precision, Recall, Positves/Negatives
all_Precisions  = [];
all_Recalls     = [];
all_TP          = [];
all_FP          = [];
all_FN          = [];
all_hardNeg_features    = [];
all_hardNed_bboxes      = [];
all_hardNeg_labels      = [];
all_Test_Features = [];
all_test_gTruth_labels = [];
all_Predicted_Labels = [];

%% Loop
f = waitbar(0,"extracting features");
for i = 1:num_image_files
    WaitbarMessage = sprintf("Images labelled: %.i/%.i", i,num_image_files);
    waitbar(i/num_image_files,f,WaitbarMessage)

    % --- Class ID definitions (keep same as training labels) ---
    crater_class_id = 1; 
    not_crater_class_id = -1; 

    % Find column indices for scores output
    [~, crater_score_col_idx] = ismember(crater_class_id, SVM_classnames);
    [~, not_crater_score_col_idx] = ismember(not_crater_class_id, SVM_classnames);

%-----------------------------------------------------------------------------------------------------------------------
    % Load image and grayscale it
    test_image_path = fullfile(image_folder, sorted_images{i});
    test_img = imread(test_image_path);
    if size(test_img, 3) == 3
      test_img_gray = rgb2gray(test_img);
    else
       test_img_gray = test_img;
    end
    [imgHeight, imgWidth] = size(test_img_gray);

%-----------------------------------------------------------------------------------------------------------------------
    % Generate cancidate regions (Sliding window)
    window_sizes = [64 64; 96 96; 128 128; 150 150; 200 200;320 320]; % from fixed ROI size, up to half the iamge size
    step_size = 16; % Move window by 16 pixels each time (adjust for speed/density)
    
    candidate_bboxes = []; % Store [xmin ymin width height] for each window
    candidate_features = []; % Store features for each window (combined HOG+LBP)
    original_window_pixels = []; % Store original window dimensions if needed for scaling
    
    % IMPORTANT: Make sure these match training parameters! - (they do)
    fixed_roi_size = [64 64]; % e.g., all ROIs were resized to 64x64 before feature extraction
    hog_cellSize = [8 8];
    lbp_radius = 1;
    lbp_numNeighbors = 8;
    
    fprintf('Generating candidate regions and extracting features for %s...\n', test_image_path);
    fprintf("\n")
%-----------------------------------------------------------------------------------------------------------------------
    % Extract features
    for ws_idx = 1:size(window_sizes, 1)
        current_window_width = window_sizes(ws_idx, 1);
        current_window_height = window_sizes(ws_idx, 2);
    
        for y = 1:step_size:(imgHeight - current_window_height + 1)
            for x = 1:step_size:(imgWidth - current_window_width + 1)
                bbox_original_coords = [x, y, current_window_width, current_window_height];
                
                % Crop ROI from the image
                roi = imcrop(test_img_gray, bbox_original_coords);
                
                % Resize ROI to the fixed size used during training
                roi_resized = imresize(roi, fixed_roi_size);
                
                % Extract HOG features
                hog_features = extractHOGFeatures(roi_resized, 'CellSize', hog_cellSize);
    
                % Extract LBP features
                lbp_features = extractLBPFeatures(roi_resized, 'Radius', lbp_radius, ...
                                                  'NumNeighbors', lbp_numNeighbors, ...
                                                  'Upright',false); 
                
                combined_features = [hog_features, lbp_features];
                
                candidate_bboxes = [candidate_bboxes; bbox_original_coords];
                candidate_features = [candidate_features; combined_features];
            end
        end
    end
    fprintf('Extracted features for %d candidate regions.\n', size(candidate_features, 1));
    fprintf("\n")
%-----------------------------------------------------------------------------------------------------------------------
    % Apply the *same* normalization from training
    % Check dimensions of train_mu and train_sigma for scaling
    % fprintf('Dimensions of train_mu (from training): %s\n', mat2str(size(train_mu)));
    % fprintf('Dimensions of train_sigma (from training): %s\n', mat2str(size(train_sigma)));
    % 
    % Check if the columns of test_pca_features match the size of train_mu/sigma
    if size(candidate_features, 2) ~= numel(train_mu) || size(candidate_features, 2) ~= numel(train_sigma)
         error('Mismatch: Columns in test_pca_features (%d) do not match numel of train_mu/sigma (%d). PCA output dimension problem before scaling!', size(candidate_features, 2), numel(train_mu));
    end
    
    scaled_candidate_features = (candidate_features - train_mu) ./ train_sigma;
    

%-----------------------------------------------------------------------------------------------------------------------
    % Apply the *same* PCA transformation from training data
    % fprintf('Dimensions of scaled_candidate_features: %s\n', mat2str(size(scaled_candidate_features)));
    % 
    % Check dimensions of train_coeff
    % fprintf('Dimensions of train_coeff (from training): %s\n', mat2str(size(train_coeff)));
    % fprintf('train_num_components (from training): %d\n', train_num_components);
    % 
    % Verify that the columns of RAW features match the rows of train_coeff
    if size(scaled_candidate_features, 2) ~= size(train_coeff, 1)
        error('Mismatch: Columns in candidate_features (%d) do not match rows in train_coeff (%d). Raw feature extraction problem!', size(candidate_features, 2), size(train_coeff, 1));
    end
    
    test_pca_features = scaled_candidate_features * train_coeff(:, 1:train_num_components);
    fprintf('Dimensions of test_pca_features (FINAL INPUT BEFORE SVM): %s\n', mat2str(size(test_pca_features)));
    fprintf("\n")

    
    
    %-----------------------------------------------------------------------------------------------------------------------
    % Predict the validation data, with the svmModel
    fprintf('Predicting with SVM...\n');
    fprintf("\n")
    % [predicted_labels_numeric, scores] = predict(svmModel, test_pca_features);
    [predicted_labels_numeric, scores] = predict(RetrainedsvmModel , test_pca_features);    
    
    all_Predicted_Labels = [all_Predicted_Labels;predicted_labels_numeric];

    fprintf('Prediction successful. Dimensions of predicted_labels_numeric: %s\n', mat2str(size(predicted_labels_numeric)));
    fprintf('Dimensions of scores: %s\n', mat2str(size(scores)));
    fprintf("\n")
%% ---------------------------------------------------------------------------------------------------------------------
    % post-processing and NMS
      
    % Initialize lists for final detections
    all_final_bboxes = [];
    all_final_scores = [];
    all_final_labels = {}; % Cell array for string labels
    
    % Process 'Crater' detections
    fprintf('Processing Crater detections...\n');
    crater_confidence_scores = scores(:, crater_score_col_idx);
    crater_positive_idx = find(predicted_labels_numeric == crater_class_id & crater_confidence_scores >= crater_detection_threshold);
    
    if ~isempty(crater_positive_idx)
        crater_candidate_bboxes = candidate_bboxes(crater_positive_idx, :);
        crater_candidate_scores = crater_confidence_scores(crater_positive_idx);
        
        [selected_crater_bboxes, selected_crater_scores] = selectStrongestBbox(crater_candidate_bboxes, crater_candidate_scores, 'OverlapThreshold', nms_iou_threshold);
        
        all_final_bboxes = [all_final_bboxes; selected_crater_bboxes];
        all_final_scores = [all_final_scores; selected_crater_scores];
        all_final_labels = [all_final_labels; repmat({'Crater'}, size(selected_crater_bboxes, 1), 1)];
    end
    
    fprintf('Total detected Craters : %d\n', size(all_final_bboxes, 1));
    fprintf("\n")
%% -----------------------------------------------------------------------------------------------------------------------
    %process groundtruth
    valid_gTruth = fullfile(label_folder,sorted_labels{i});
    current_gTruth = load(valid_gTruth);


    if size(current_gTruth,1)~=0
    all_Test_Features = [all_Test_Features; test_pca_features];
    all_test_gTruth_labels = [all_test_gTruth_labels;current_gTruth(:,1)]

    end
    if ~isempty(current_gTruth)
        %get normalized YOLO components
        normalized_center_x = current_gTruth(:,2);
        normalized_center_y = current_gTruth(:,3);
        normalized_width    = current_gTruth(:,4);
        normalized_height   = current_gTruth(:,5);
    
        % convert to pixel: center, width, height in pixels
        pixel_width = normalized_width * imgWidth;
        pixel_height = normalized_height * imgHeight;
    
        pixel_center_x = normalized_center_x * imgWidth;
        pixel_center_y = normalized_center_y * imgHeight;
    
        %compute top left corner (x,y)
        pixel_x = pixel_center_x - (pixel_width/2);
        pixel_y = pixel_center_y - (pixel_height/2);
        
        groundTruthBoxes = [pixel_x, pixel_y, pixel_width, pixel_height];
    
        num_gt_objects = size(groundTruthBoxes,1);
    
        %Perform IoU-based matching and count TP, FP,FN
        TruePositives = 0;
        FalsePositives = 0;
        %FalseNegative is defined later

        matches_gt_indices = false(num_gt_objects,1);   %setup, sets all indices to "not-matched"
    
        %sort predictions and bboxes by score
        [sortedScores, sorted_indices] = sort(abs(all_final_scores),'descend');
        predictedBboxes_sorted = all_final_bboxes(sorted_indices,:);
    
        num_predicted_objects = size(predictedBboxes_sorted,1);
    
        %Make new image, for all stats
        Stat_img = test_img;
        stat_fig = figure(1);
        imshow(Stat_img);
        title(sprintf('Validation Results for image %i',i))
        hold on;

        % draw groundtruth in blue
        % for k = 1:num_gt_objects
        %     bbox_gTruth = groundTruthBoxes(k,:);
        %     rectangle("Position",bbox_gTruth, 'EdgeColor','b', 'LineWidth',2);
        %     %text(bbox_gTruth(1),bbox_gTruth(2)-5,'GT', 'Color','b','FontSize',8,'FontWeight','bold');
        % end

        %Determine IoU
        for p_idx = 1:num_predicted_objects
            current_pred_bbox = predictedBboxes_sorted(p_idx,:);
            current_pred_score = sortedScores(p_idx);
    
            max_iou = 0;
            best_gt_idx_for_pred = -1;
    
            %find the best gTruth that has not been matches, for the current prediction
            for gt_idx = 1:num_gt_objects
                if ~matches_gt_indices(gt_idx)  %only check unmatches gTruths
                    current_gt_bbox = groundTruthBoxes(gt_idx,:);
    
                    iou = bbox_iou(current_pred_bbox, current_gt_bbox); %Selfmade function to compute IoU
                    
                    if iou>max_iou
                        max_iou = iou;
                        best_gt_idx_for_pred = gt_idx;
                    end
                end
            end
            


            %determine if it's a TP of FP
            if max_iou>=IoU_threshold && best_gt_idx_for_pred ~= -1
                TruePositives = TruePositives +1;
                matches_gt_indices(best_gt_idx_for_pred) = true;    %Mark this as a good match

                %Draw True Positive in green
                rectangle("Position",current_pred_bbox, 'EdgeColor','g', 'LineWidth',2);
                text(current_pred_bbox(1),current_pred_bbox(2)+current_pred_bbox(4)+5,sprintf("IOU:%.2f",max_iou), 'Color','g','FontSize',8,'FontWeight','bold');
                
                %Draw accompanying GroundTruth 
                rectangle("Position",groundTruthBoxes(best_gt_idx_for_pred,:), 'EdgeColor','b', 'LineWidth',2);
                text(groundTruthBoxes(best_gt_idx_for_pred,1),groundTruthBoxes(best_gt_idx_for_pred,2)-5,'GT', 'Color','b','FontSize',8,'FontWeight','bold');
            else
                FalsePositives = FalsePositives +1; %no good match, so it's a FP

                %Draw False Positive in red
                rectangle("Position",current_pred_bbox, 'EdgeColor','r', 'LineWidth',2);
                text(current_pred_bbox(1),current_pred_bbox(2)+current_pred_bbox(4)+5,sprintf("IOU:%.2f",max_iou), 'Color','r','FontSize',8,'FontWeight','bold');
                
                %Draw accompanying GroundTruth 
                if best_gt_idx_for_pred >0
                rectangle("Position",groundTruthBoxes(best_gt_idx_for_pred,:), 'EdgeColor','b', 'LineWidth',2);
                text(groundTruthBoxes(best_gt_idx_for_pred,1),groundTruthBoxes(best_gt_idx_for_pred,2)-5,'GT', 'Color','b','FontSize',8,'FontWeight','bold');
                end
            end
    end
        
        % Draw False negatives (unmatched grtruths)
        for gt_idx = 1:num_gt_objects
            if ~matches_gt_indices(gt_idx)
                bbox = groundTruthBoxes(gt_idx, :);
                rectangle("Position",bbox, 'EdgeColor','magenta', 'LineWidth',2);
                %text(current_pred_bbox(1),current_pred_bbox(2)-5,'FN', 'Color','magenta','FontSize',8,'FontWeight','bold');
            end
        end
        
        %Finish adding STATS
        hold off;

        %Count FN (unmatched gTruths)
        FalseNegatives = sum(~matches_gt_indices);
        
        fprintf('--- Validation Results for %s ---\n', test_image_path);
        fprintf('True Positives (TP): %d\n', TruePositives);
        fprintf('False Positives (FP): %d\n', FalsePositives);
        fprintf('False Negatives (FN): %d\n', FalseNegatives);
    
        Precision = TruePositives/(TruePositives + FalsePositives);
        Recall = TruePositives / (TruePositives + FalseNegatives);
    
        %handle NaN if denom are zero
        if isnan(Precision),Precision = 0; end
        if isnan(Recall), Recall = 0; end
    
        fprintf('Precision: %.4f\n',Precision);
        fprintf('Recall: %.4f\n',Recall);
    
        %Contain statistics in output arrays
        all_Precisions  = [all_Precisions; Precision];
        all_Recalls     = [all_Recalls; Recall];
        all_TP          = [all_TP; TruePositives];
        all_FP          = [all_FP; FalsePositives];
        all_FN          = [all_FN; FalseNegatives];
        
        % Save visualisation, create folder if it doesn't exist
        OutputFolder = "MatlabSVM\CraterQuest_Online_Craters\Test\"+Kernel_type+"\";
        if ~exist("OutputFolder","dir")
            mkdir(OutputFolder)
        end
    
        outputname = OutputFolder+"test_IT_"+num2str(Iteration)+ "_"+num2str(i)+".jpg";
        exportgraphics(stat_fig,outputname,"Resolution",96) 
    end   



end
%%
Mean_stats = [mean(all_Precisions),mean(all_Recalls),mean(all_TP),mean(all_FP),mean(all_FN)];

disp(Mean_stats);

outputType = Kernel_type + "_IT"+num2str(Iteration)+"_";
NegHard_out_name = outputType + "HardNeg_featLabels_Dijkstra.mat";
Val_out_name = outputType + "test_stats_Dijkstra.mat";
Feat_out_name = outputType +"test_features.mat";

save(Feat_out_name,"all_Test_Features","all_test_gTruth_labels","all_Predicted_Labels")

% First validation
% save(Val_out_name,"all_TP","all_FP","all_FN","all_Precisions","all_Recalls","Mean_stats");

disp("Stats saved.\n")
close(f)
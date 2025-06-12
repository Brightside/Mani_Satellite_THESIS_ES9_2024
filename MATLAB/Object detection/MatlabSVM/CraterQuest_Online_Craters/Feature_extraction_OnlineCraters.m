%% SVM2_Multimask_labelling_(fourier)
% This code utilizes Fourier descriptors to determine the shape based on the energy contained in the object-shape. 
clc, clear, close all

%----------------------------------------------------------------------------------------------------------------------
% Setup
IMG_graphic_output = 0;     % outputting graphics

Crater_Label_ID = 1;
Negative_Label_ID = -1;

%----------------------------------------------------------------------------------------------------------------------
% Folder containing images and labels (not using labels, using my own
% image_folder = "Crater_images\crater-detection_v8.v1i.yolov8\train\images";
% label_folder = "Crater_images\crater-detection_v8.v1i.yolov8\train\labels";
image_folder = "Crater_images\Cath_Dijkstra_set\craters\train\images";
label_folder = "Crater_images\Cath_Dijkstra_set\craters\train\labels";

% Get list of files in the folders
image_files = dir(fullfile(image_folder, '*.jpg')); % Change extension if needed
label_files = dir(fullfile(label_folder,'*.txt'));  

%Sort images (natsort by stephen cobeldick)
image_names = {image_files.name};
sorted_image_names = natsortfiles(image_names);
label_names = {label_files.name};
sorted_label_names = natsortfiles(label_names);

% compute size of folder
num_image_files = length(image_files); %Num images = num labels
num_label_files = num_image_files;

all_pos_labels = 0;
%% ----------------------------------------------------------------------------------------------------------------------
%initiate variables and arrays
total_features          = [];
total_labels            = [];

% ----------------------------------------------------------------------------------------------------------------------
% Loop
f = waitbar(0,"extracting features");
for i = 1:num_image_files
    WaitbarMessage = sprintf("Images labelled: %.i/%.i", i,num_image_files);
    waitbar(i/num_image_files,f,WaitbarMessage)
    % Load and preprocess the grayscale image
    img_path = fullfile(image_folder, sorted_image_names{i});
    img = imread(img_path);
    [imgWidth,imgHeight,~] = size(img);


    %------------------------------------------------------------------------------------------------------------------
    % Load labels for the current image (convoluted but: load struct-table -> get struct-info -> format into labels-matrix)
    label_path = fullfile(label_folder,sorted_label_names{i});
    labels = load(label_path);
    labels = labels * imgWidth;
    
    %Change class ID to 1, unless it is empty 
    if size(labels,1) ~= 0
    labels(:,1) = Crater_Label_ID;
    all_pos_labels = all_pos_labels + size(labels,1);
    end
    % labels(:,2:5) = labels(:,2:5) .* imgWidth;
      %------------------------------------------------------------------------------------------------------------------
    if IMG_graphic_output
        im_name = regexprep(sorted_image_names{i},'.rf.*.jpg','');
        imshow(img), title(im_name)
        for j = 1:length(labels)
            if labels(j,1) == 0
                edgeColor = 'r';
            else
                edgeColor = 'b';
            end
            rectangle('Position',labels(j,2:5),'EdgeColor',edgeColor,'LineWidth',2);
        end
    end

    %------------------------------------------------------------------------------------------------------------------
    % extract HOG and lbp features
    [positve_features, crater_labels] =    extract_hog_lbp_features_fixed(img_path,labels,true);
    [negative_features, non_crater_labels] = extract_hog_lbp_features_negative_zones(img_path,labels,Negative_Label_ID,true);

    total_features          = [total_features;positve_features;negative_features];
    total_labels            = [total_labels;crater_labels;non_crater_labels];
    
end
close(f)

%% apply PCA on normalized features, to reduce dimensions for training SVM

[total_features_scaled, train_mu, train_sigma] = normalize(total_features,'zscore');

[train_coeff, score, latent, tsquared, explained] = pca(total_features_scaled);

% Choose components to explain a certain variance (e.g., 95%)
cumulative_explained = cumsum(explained);
train_num_components = find(cumulative_explained >= 95, 1, 'first'); % Find 95% variance

fprintf('Number of components for 95%% variance: %d\n', train_num_components);

features_pca_scaled = score(:, 1:train_num_components); % Take the top components

%save("features.mat","features_pca_scaled","total_labels","train_mu","train_sigma")
save("features_direct_Dijkstra.mat","features_pca_scaled","total_labels","train_mu","train_sigma","train_coeff","train_num_components")

%% debugging - inspect features for singular image
PCA_graphic_output = 0;
if PCA_graphic_output
    % Visualize PCA relations
    figure(1);
    plot(1:length(explained), explained, 'bo-');
    xlabel('Principal Component Number');
    ylabel('Variance Explained (%)');
    title('Variance Explained by Each Principal Component');
    grid on;
    
    figure(2);
    plot(1:length(explained), cumsum(explained), 'ro-');
    xlabel('Number of Principal Components');
    ylabel('Cumulative Variance Explained (%)');
    title('Cumulative Variance Explained by Principal Components');
    grid on;
    yline(95, 'r--', '95% Threshold'); % Example threshold
    
    % scatterplot of PCA
    figure(3);
    gscatter(features_pca_scaled(:,1), features_pca_scaled(:,2), total_labels, 'rgb', 'ooo');
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');
    title('Data in 2D PCA Space (Colored by Class)');
    legend('Crater', 'Non-Crater','Module'); % Adjust legend if you have class IDs 0 and 1
    
    % plot of corr matrices
    % For original features
    figure(4);
    imagesc(corr(total_features_scaled));
    colorbar;
    title('Correlation Matrix of Original Scaled Features');
    
    % For PCA features (should be near diagonal)
    figure(5);
    imagesc(corr(features_pca_scaled));
    colorbar;
    title('Correlation Matrix of PCA Scaled Features');
end
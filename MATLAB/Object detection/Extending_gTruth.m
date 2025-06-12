% --- MATLAB Image Annotation Script for YOLO Data ---

% Prompts the user to select an image folder
imageFolder = uigetdir('', 'Select Image Folder');
if isequal(imageFolder, 0)
    disp('Image folder selection cancelled.');
    return;
end

% Get a list of all image files in the selected folder
% You can add other image formats here if needed (e.g., *.bmp, *.tiff)
imageFiles = dir(fullfile(imageFolder, '*.jpg')); 
imageFiles = [imageFiles; dir(fullfile(imageFolder, '*.png'))]; 

if isempty(imageFiles)
    disp('No image files found in the selected folder.');
    return;
end

%%
disp('Starting image annotation process...');
disp('Left-click 4 points to define an object. Right-click to finish current image annotation.');
disp(' '); % Add a blank line for readability

% Loop through each image in the folder
for i = 1:length(imageFiles)
    currentImageFile = fullfile(imageFolder, imageFiles(i).name);
    img = imread(currentImageFile);
    [imgHeight, imgWidth, ~] = size(img);

    disp(['Processing image: ', imageFiles(i).name]);
    figure; % Open a new figure for each image
    imshow(img);
    title(['Click 4 points for object (Right-click to finish for this image): ', imageFiles(i).name]);

    allYOLODataForImage = []; % To store all YOLO data for the current image

    keepAnnotatingCurrentImage = true;
    while keepAnnotatingCurrentImage
        points = [];
        fprintf('  Click 4 points for the current object (object %d)...\n', size(allYOLODataForImage, 1) + 1);
        
        % Get 4 points from user clicks
        for p = 1:4
            [x, y, button] = ginput(1);
            if button == 3 % Right-click
                keepAnnotatingCurrentImage = false; % Stop annotating this image
                break; % Exit the 4-point loop
            end
            points = [points; x, y];
            hold on; % Keep the plot active for marking points
            plot(x, y, 'r+', 'MarkerSize', 10, 'LineWidth', 2); % Mark the clicked point
        end

        if size(points, 1) == 4 % Ensure 4 points were successfully clicked
            % Convert the 4 points to a bounding box (min/max X, Y)
            minX = min(points(:, 1));
            maxX = max(points(:, 1));
            minY = min(points(:, 2));
            maxY = max(points(:, 2));

            % Calculate center, width, and height in pixel coordinates
            centerX_pixel = (minX + maxX) / 2;
            centerY_pixel = (minY + maxY) / 2;
            width_pixel = maxX - minX;
            height_pixel = maxY - minY;

            % Normalize to image size for YOLO format
            class_id = 1; % Class ID is always 1
            centerX_norm = centerX_pixel / imgWidth;
            centerY_norm = centerY_pixel / imgHeight;
            width_norm = width_pixel / imgWidth;
            height_norm = height_pixel / imgHeight;

            % Store the YOLO formatted data
            yoloData = [class_id, centerX_norm, centerY_norm, width_norm, height_norm];
            allYOLODataForImage = [allYOLODataForImage; yoloData];

            % Display the bounding box for verification
            rectangle('Position', [minX, minY, width_pixel, height_pixel], 'EdgeColor', 'g', 'LineWidth', 2);
            fprintf('  Object %d annotated. Current YOLO data: %s\n', size(allYOLODataForImage, 1), mat2str(yoloData));
        end
        
        if keepAnnotatingCurrentImage % Only prompt if not already stopping via right-click
            choice = questdlg('Annotate another object in this image?', ...
                              'Continue Annotation', ...
                              'Yes', 'No', 'Yes'); % Default to 'Yes'
            if strcmp(choice, 'No')
                keepAnnotatingCurrentImage = false;
            end
        end
        hold off; % Release the hold for the next set of points or next image
    end
    
    close(gcf); % Close the current image figure after annotation is done

    % Save the YOLO data for the current image to a .txt file
    if ~isempty(allYOLODataForImage)
        [~, name, ~] = fileparts(imageFiles(i).name);
        outputFileName = fullfile(imageFolder, [name, '.txt']);
        writematrix(allYOLODataForImage, outputFileName, 'Delimiter', ' ');
        disp(['  Saved YOLO data for "', imageFiles(i).name, '" to "', outputFileName, '"']);
    else
        disp(['  No YOLO data saved for "', imageFiles(i).name, '" (no objects annotated).']);
    end
    disp('--------------------------------------------------'); % Separator for next image
end

disp('Annotation process complete for all images!');
%% generate circle
clc; clear; close all;
%----------------------------------------------------------------
% circle using circle equation
% Define the circle parameters
% theta = linspace(0, 2*pi, 100); % Angle values
% r = 3; % Radius
% centroid = 1.5*r;
% 
% x = centroid+ r * cos(theta); % X-coordinates
% y = centroid+ r * sin(theta); % Y-coordinates
% x = [x x(1)];
% y = [y y(1)];
% 
% plot(x,y,'Color','r')
%----------------------------------------------------------------
% overlapping circles using rectangle
Circles_Amount = 4;
centroid = [2,2;
            2,4;
            4,2;
            4,4];
radius = 3;

RGB_MAX = 256;
colours = [1,0,0;
           0,1,0;
           0,0,1;
           1,1,0];
figure(1);
for i= 1:Circles_Amount
    rectangle('Position',[(centroid(i,:))-radius,[2,2]*radius],'Curvature',1,'FaceColor',colours(i,:),'EdgeColor',colours(i,:))
end
xlim([min(min(centroid))-1.5*radius,max(max(centroid))+1.5*radius]);
ylim([min(min(centroid))-1.5*radius,max(max(centroid))+1.5*radius]);

 set(gcf, 'Units', 'pixels', 'Position', [100, 100, 600,600]); % Fixed figure size
 set(gca, 'Units', 'pixels', 'Position', [50, 50, 500, 500]); % Fixed axes size
 set(gca,'XColor','none','YColor','none');
 ax = gca;
 
exportgraphics(ax,"Circle_Example.jpg");

%% Extract hog
close all
img_path = "puppy.jpg";

%img_path = "puppy.jpg";
img = imread(img_path);
new_img = imresize(img,[128 64]*2);

figure(3);
imshow(new_img)
hold on;
% Extract HOG features from the entire image (keeping spatial information)
cellSize = [1, 1]*8;   % Size of HOG cells

[hog, Visualization] = extractHOGFeatures(new_img,'CellSize',cellSize,'BlockSize',[2,2],'NumBins',9);

plot(Visualization,'color',[0,0,0])
%%
set(gcf, 'Units', 'pixels', 'Position', [100, 100, 600,600]); % Fixed figure size
set(gca, 'Units', 'pixels', 'Position', [50, 50, 500, 500]); % Fixed axes size
set(gca,'XColor','none','YColor','none');

ax = gca;
exportgraphics(ax,"Circle_HOG_visual.jpg");
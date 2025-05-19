%Setup
Image_Size = [1e3, 1e3];

%Load image
pre_warp_img = imread("test_image_5.jpg");
Image_Size = size(pre_warp_img);
%Pick 4 control-points
imshow(pre_warp_img); hold on
[x_pre,y_pre] = ginput(4);

slanted_pts = [x_pre(1),y_pre(1);
               x_pre(2),y_pre(2);
               x_pre(3),y_pre(3);
               x_pre(4),y_pre(4)];

scatter(slanted_pts(:,1),slanted_pts(:,2),"filled","o","MarkerFaceColor",[0,1,0]);


[x_post,y_post] = ginput(4);

topdown_pts = [x_post(1),y_post(1);
               x_post(2),y_post(2);
               x_post(3),y_post(3);
               x_post(4),y_post(4)];
close
imshow(pre_warp_img); hold on

scatter(topdown_pts(:,1),topdown_pts(:,2),"filled","o","MarkerFaceColor",[0,0,1])

H = estgeotform2d(slanted_pts, topdown_pts, 'projective');

rectified_img = imwarp(pre_warp_img,H,'OutputView',imref2d(size(pre_warp_img)));


figure;
imshow(pre_warp_img)
figure;
imshow(rectified_img)
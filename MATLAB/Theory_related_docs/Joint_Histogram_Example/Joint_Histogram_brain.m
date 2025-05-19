clc,clear;
t1 = niftiread("mni_icbm152_t1_tal_nlin_asym_09a.nii");
t2 = niftiread("mni_icbm152_t2_tal_nlin_asym_09a.nii");

% Extract image from slices
slice = 94;
t1_image = t1(:,:,slice); 
t2_image = t2(:,:,slice);

figure(1); montage({t1_image,t2_image}),title("MRI's T1 (left) and T2 (right)"), fontsize(24,"points")
exportgraphics(gcf,"Brains.png",'Resolution',96)

%%
figure(2);
subplot(2,1,1); t1_hist = histogram(t1_image); title("T1 histogram"), xlabel("Signed 16 bit integer value"),ylabel("Observations"), fontsize(24,"points")
subplot(2,1,2); t2_hist = histogram(t2_image); title("T2 histogram"), xlabel("Signed 16 bit integer value"),ylabel("Observations"), fontsize(24,"points")
exportgraphics(gcf,"Separate_Histograms_Brains.png",'Resolution',96)


%% generate histogram
binScaling = 10;     %Scale how many bins, 1 = default 33
totalBins = min(t1_hist.NumBins,t2_hist.NumBins)*binScaling;

figure(3); histogram2(t1_hist.Data,t2_hist.Data,'NumBins',[totalBins totalBins],'DisplayStyle','tile'), fontsize(24,"points")
colorbar; title("Joint histogram"); xlabel("T1"),ylabel("T2");
exportgraphics(gcf,"Joint_Histograms_Brains.png",'Resolution',96)

OG_mi = computeMutualInformation(t1_image,t2_image,totalBins);
OG_nmi = computeNormalizedMutualInformation(t1_image,t2_image,totalBins);


%% shift image down
shift_factor = 30;
t2_image_shifted = zeros(size(t2_image));
t2_image_shifted(shift_factor:end,:) = t2_image(1:(end-(shift_factor-1)),:);
t2_image_shifted(1:(shift_factor-1),:) = ones([(shift_factor-1),233])*-32768;   %-32768 is min value for int16, corresponds to [0,0,0] in RGB
t2_image_shifted = int16(t2_image_shifted);

figure(4); montage({t1_image,t2_image_shifted}),title("MRI's T1(left) and T2 shifted(right)"), fontsize(24,"points")
exportgraphics(gcf,"Brains_Shifted.png",'Resolution',96)

%%
figure(5);
subplot(2,1,1); histogram(t2_image),title("T2 histogram")
subplot(2,1,2); t2_shift_hist = histogram(t2_image_shifted);title("T2 shifted histogram"), fontsize(24,"points")
exportgraphics(gcf,"Separate_Histogram_Brains_Shifted.png",'Resolution',96)
%%
figure(6);
histogram2(t1_hist.Data,t2_shift_hist.Data,'NumBins',[totalBins totalBins],'DisplayStyle','tile'), fontsize(24,"points");
title("Joint histogram"); xlabel("T1"),ylabel("T2");
exportgraphics(gcf,"Joint_Histograms_Brains_Shifted.png",'Resolution',96)

shifted_mi  = computeMutualInformation(t1_image,t2_image_shifted,totalBins);
shifted_nmi = computeNormalizedMutualInformation(t1_image,t2_image_shifted,totalBins);

%% Compare MiScores
clc
fprintf("miScores:  align/misalign -  %.4f / %.4f = %.4f\n",OG_mi,shifted_mi, OG_mi/shifted_mi)
fprintf("nmiScores: align/misalign -  %.4f / %.4f = %.4f\n",OG_nmi,shifted_nmi,OG_nmi/shifted_nmi)


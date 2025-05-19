clc; clear; close all;

%Setup
    %types of logs to plot
    normal = 1;
    logarithmic = 1;
%% Generate Ellipse Boundary Manually
t = linspace(0, 2*pi, 100); % Parameterized ellipse points
a = 50; b = 30; % Semi-major and semi-minor axes
x_ellipse = a * cos(t);
y_ellipse = b * sin(t);


%Rotate the ellipse
alpha = 45; %angle
R  = [cosd(alpha) -sind(alpha); ...
      sind(alpha)  cosd(alpha)];
rCoords = R*[x_ellipse ; y_ellipse];   


% Compute Fourier Descriptor for the ellipse
z_ellipse = x_ellipse + 1i * y_ellipse;
FD_ellipse = fft(z_ellipse);
FD_ellipse = abs(FD_ellipse) / abs(FD_ellipse(1));   % Normalize magnitude       - making scale invariant
FD_ellipse_log = log(abs(FD_ellipse(2:end)));   % Ignore DC component       - making position invariant

z_rotated = rCoords(1,:) + 1i * rCoords(2,:);
FD_ellipse_rotated = fft(z_rotated);
FD_ellipse_rotated = abs(FD_ellipse_rotated) / abs(FD_ellipse_rotated(1));   % Normalize magnitude       - making scale invariant
FD_ellipse_rotated_log = log(abs(FD_ellipse_rotated(2:end)));   % Ignore DC component       - making position invariant


%% Generate Cross Shape and Extract Boundary
crossSize = 50; inner_width = 20;   %crossSize is fullSize, inner_width is the cross-"lines" width
bwCross = zeros(200, 200);
bwCross(90:110, 50:150) = 1; % Vertical bar
bwCross(50:150, 90:110) = 1; % Horizontal bar

% Extract boundary using bwboundaries
bwCross = imfill(bwCross, 'holes'); % Fill cross shape

boundaries = bwboundaries(bwCross);
crossBoundary = boundaries{1}; % Get boundary points
x_cross = crossBoundary(:,2);
y_cross = crossBoundary(:,1);

centroid = [mean(x_cross),mean(y_cross)];

x_cross_centred = x_cross - centroid(1);
y_cross_centred = y_cross - centroid(2);

% Resample boundary points
numPoints = 100;
t = linspace(1, length(x_cross_centred), numPoints);
x_cross_resampled = interp1(1:length(x_cross_centred), x_cross_centred, t);
y_cross_resampled = interp1(1:length(y_cross_centred), y_cross_centred, t);

% Compute Fourier Descriptor for the cross
z_cross = x_cross_resampled + 1i * y_cross_resampled;
FD_cross = fft(z_cross);
FD_cross = FD_cross / abs(FD_cross(1));                     % Normalize magnitude       - making scale invariant
FD_cross_log = log(abs(FD_cross(2:end)));                   % Ignore DC component       - making position invariant

%% Plot shapes
figure;
subplot(1,2,1)
plot(rCoords(1,:),rCoords(2,:)), title("Ellipse");
axis equal
subplot(1,2,2)
plot(x_cross_centred,y_cross_centred), title("Cross");
%imshow(bwCross), title("Cross")

exportgraphics(gcf,"Objects.png","Resolution",96)

%% plot frequency spectra comparison (normal)
if normal
    FD_ellipse_normal = abs(FD_ellipse);
    FD_ellipse_rotated_normal = abs(FD_ellipse_rotated);
    FD_cross_normal = abs(FD_cross);
    
    figure;
    hold on;
    plot(FD_ellipse_rotated_normal, 'o--', 'LineWidth', 1.5, 'DisplayName', 'Ellipse FD Spectrum');
    xlabel('Frequency Index');
    ylabel('Magnitude of Fourier Descriptor');
    title('Fourier Descriptor Frequency Spectrum Comparison');
    legend();
    grid on;
    hold off;
    
    exportgraphics(gcf,"Normal_Fourier_Ellipse.png","Resolution",96);

    figure;
    hold on;
    plot(FD_cross_normal, 'o--', 'LineWidth', 1.5, 'DisplayName', 'Cross FD Spectrum');
    xlabel('Frequency Index');
    ylabel('Magnitude of Fourier Descriptor');
    title('Fourier Descriptor Frequency Spectrum Comparison');
    legend();
    grid on;
    hold off;

    exportgraphics(gcf,"Normal_Fourier_Cross.png","Resolution",96);
end
%% Plot Frequency Spectra Comparison (Log)
if logarithmic
    figure;
    hold on;
    plot(FD_ellipse_rotated_log, 'o--', 'LineWidth', 1.5, 'DisplayName', 'Ellipse FD Log Spectrum');
    plot(FD_cross_log, 's-', 'LineWidth', 1.5, 'DisplayName', 'Cross FD Log Spectrum');
    xlabel('Frequency Index');
    ylabel('Log-Magnitude of Fourier Descriptor');
    title('Fourier Descriptor Frequency Spectrum Comparison');
    legend();
    grid on;
    hold off;
    
    exportgraphics(gcf,"Fourier_Descriptors.png","Resolution",96)
end
%% Reconstruct shapes

%reconstruct from both normal and log-normal
Ellipse_reconstruct_normal = ifft(FD_ellipse_rotated);
Cross_reconstruct_normal = ifft(FD_cross);

figure;
subplot(1,2,1)
plot(Ellipse_reconstruct_normal), title("Reconstructed Ellipse");
axis equal;
subplot(1,2,2)
plot(Cross_reconstruct_normal), title("Reconstructed Cross");
axis equal;

exportgraphics(gcf,"Reconstructed_objects.png","Resolution",96)
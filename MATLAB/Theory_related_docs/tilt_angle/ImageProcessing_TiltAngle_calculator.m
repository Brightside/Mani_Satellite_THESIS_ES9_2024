%% ImageProcessing_TiltAngle_calculator
clc, close all

%known variables
satellite_altitude  = 50;     %meters
satellite_FOV       = 0.008;    %radians
moon_shift = 34.43; %positional shift because of moons rotation. km

%line spaces
phi_deg = linspace(0,89.5,1000);    % tilt angle
phi_rad = deg2rad(phi_deg);

slant_range = satellite_altitude ./ cos(phi_rad);

image_width = 2 * slant_range * tan(satellite_FOV/2);


% Find intersections (known y-limiters: 1 and 10 km image width)
y1      = 1;
y10     = 10;
idx1    = find(image_width >= y1, 1, 'first');  % First index where blue curve exceeds 1 km
idx10   = find(image_width >= y10, 1, 'first'); % First index where blue curve exceeds 10 km
x_intersect1 = phi_deg(idx1);
x_intersect10 = phi_deg(idx10);

% Find intersections (moons rotation -> positional shift -> required attitude change)
x_idx33 =find(phi_deg >=moon_shift,1,'first');
x_idx66 =find(phi_deg>=moon_shift*2,1,'first');
x_intersect1 = phi_deg(x_idx33); y_intersect1 = image_width(x_idx33);
x_intersect2 = phi_deg(x_idx66); y_intersect2 = image_width(x_idx66);

%% figure with rotational positional shift attitude marked
figure;
plot(phi_deg,image_width, 'B', 'LineWidth',2)
hold on;
plot([0,x_intersect1],[y_intersect1,y_intersect1], 'r-.', 'LineWidth', 1.5);
plot([0, x_intersect2], [y_intersect2, y_intersect2], 'g-.', 'LineWidth', 1.5);
plot([x_intersect1, x_intersect1], [0, y_intersect1], 'r-.', 'LineWidth', 1.5);
plot([x_intersect2, x_intersect2], [0, y_intersect2], 'g-.', 'LineWidth', 1.5);
xlabel('Tilt Angle (degrees)');
ylabel('Image Width (km)');
title('Image Width vs. Tilt Angle for Lunar Satellite Camera');
grid on;
legend('Image Width', '34.43 degrees', '68.86 degrees');
hold off;
ylim([0,20])
exportgraphics(gcf, "Tilt_angle.png", "Resolution",96);

%%  figure with desired image widths marked
figure;
plot(phi_deg,image_width, 'B', 'LineWidth',2)
hold on;
plot([0, x_intersect1], [y1, y1], 'r-.', 'LineWidth', 1.5);
plot([0, x_intersect10], [y10, y10], 'g-.', 'LineWidth', 1.5);
plot([x_intersect1, x_intersect1], [0, y1], 'r-.', 'LineWidth', 1.5);
plot([x_intersect10, x_intersect10], [0, y10], 'g-.', 'LineWidth', 1.5);
xlabel('Tilt Angle (degrees)');
ylabel('Image Width (km)');
title('Image Width vs. Tilt Angle for Lunar Satellite Camera');
grid on;
legend('Image Width', '1 km Width', '10 km Width');
hold off;
ylim([0,20])
exportgraphics(gcf, "Tilt_angle.png", "Resolution",96);


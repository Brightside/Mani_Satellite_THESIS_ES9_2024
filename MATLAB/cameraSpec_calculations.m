%% setup
clear all 
clc

assumed_perigee     =   50;                                     %km 100
assumed_apogee      =   assumed_perigee+50;                     %km 150
assumed_medianAlt   =   (assumed_perigee+assumed_apogee)/2;     %km 125
assumed_radius      =   3474/2;                                 %km
assumed_orbitPeriod =   2*pi*sqrt((assumed_perigee*1E3+assumed_radius*1E3)^3 /5e12);  %s  111 min at perigee
assumed_orbitVeloc  =   2*pi*(assumed_radius*1E3)/assumed_orbitPeriod ; %m/s
%assumed_inclination =   98.2;                                   %deg

%% detumbling
%The detumble target velocity is to be less than 2 sat-revolutions per lunar-orbit
detumble_target_Velocity=  (2*2*pi)/assumed_orbitPeriod;   %sRev/lOrbit

%Detumble settle time is to be less than 3 cycles around the moon
detumble_settleTime =   3*assumed_orbitPeriod;
%% nadir pointing
%Nadir slewrate is just general rotation around the moon, given cycle-time.
nadir_slewRate = 2*pi/assumed_orbitPeriod;

%Nadir.accuracy, is dependant on the camera quality, thus will be explored 
%down there

%% target pointing
%Target pointing accuracy is a selfchosen value, and has been set to be
target_accuracy_deg     =   0.1;   
target_accuracy_rad     = target_accuracy_deg*pi/180;
target_accuracy_meter   = target_accuracy_rad*assumed_perigee*1E3;

%Setting up pass information regarding target
passing_sat_target_dist = assumed_perigee;
passing_radius = assumed_radius + passing_sat_target_dist;
passing_target_entry_angle = acos(assumed_radius/passing_radius);
passing_angular_speed = (assumed_orbitVeloc*1E-3)/passing_radius;   %multiplied by 1E-3 to convert second to hour
passing_nadir_start_angle = pi/2-passing_target_entry_angle;

%target Tracking Graphics
syms t
sim_time = 2*passing_target_entry_angle/passing_angular_speed;  %time-step
sim_time_t = linspace(0,sim_time,1000);                         % s vector
sim_earth_angle_t =passing_target_entry_angle-passing_angular_speed*t; % vector with time_sim, 0 @ target, positive counterclock
sim_dist_target_to_sat_t = sqrt(assumed_radius^2+passing_radius^2 - 2*assumed_radius*passing_radius*cos(sim_earth_angle_t)); % vector
sim_angle_to_nadir_t = asin(sin(sim_earth_angle_t)./sim_dist_target_to_sat_t*(assumed_radius+assumed_perigee)); %rad, angle to nadir from sat perspective
sim_angle_speed_to_nadir_t = passing_angular_speed*passing_radius*cos(sim_angle_to_nadir_t)./sim_dist_target_to_sat_t;
sim_angle_accel_to_nadir_t = diff(sim_angle_speed_to_nadir_t);

%function handles for simulation
sim_earth_angle_f = matlabFunction(sim_earth_angle_t);
sim_dist_target_to_sat_f = matlabFunction(sim_dist_target_to_sat_t);
sim_angle_to_nadir_f = matlabFunction(sim_angle_to_nadir_t);
sim_angle_speed_to_nadir_f = matlabFunction(sim_angle_speed_to_nadir_t);
sim_angle_accel_to_nadir_f = matlabFunction(sim_angle_accel_to_nadir_t);

%plotting
f1 = figure(1); 
clf(f1);
subplot(2,2,1); %angleSpan
hold on
plot(sim_time_t,sim_angle_to_nadir_f(sim_time_t)*180/pi,"Color",[255,165,0]/255,"LineWidth",2);
xlim([min(sim_time_t),max(sim_time_t)]);
xlabel("time [s]");
ylabel("Angle [deg]");
title("satellite angle");
legend("\theta(t)")
grid on

subplot(2,2,2); %distance plot of sat -> target
hold on
plot(sim_time_t,sim_dist_target_to_sat_f(sim_time_t),"g","LineWidth",2);
xlim([min(sim_time_t),max(sim_time_t)]);
xlabel("time [s]");
ylabel("Distance [km]");
title("satellite distance to target");
legend("d(t)")
grid on

subplot(2,2,3); %Angular velocity of satellite, passing
hold on
plot(sim_time_t,sim_angle_speed_to_nadir_f(sim_time_t),"b","LineWidth",2);
xlim([min(sim_time_t),max(sim_time_t)]);
xlabel("time [s]");
ylabel("Angular speed [rad/s]");
title("satellite angular velocity");
legend("\omega_{\theta}(t)")
grid on

subplot(2,2,4); %angular acceleration
hold on
plot(sim_time_t,sim_angle_accel_to_nadir_f(sim_time_t),"Color",[179,206,229]/255,"LineWidth",2);
xlim([min(sim_time_t),max(sim_time_t)]);
xlabel("time [s]");
ylabel("Anglular acceleration [deg/s^2]");
title("satellite angular acceleration");
legend("\alpha_{\theta}(t)")
grid on

%% Target init settling time
%Define viewconstraint of 20% and define entry angle
clc
ViewConstraint = 0.01;
Entry_Angle_Of_Targeting = asin(ViewConstraint);

%Compute intersect of satellite height, for target-init
syms x y
eqn1 = x^2 +y^2 == (assumed_radius + assumed_perigee)^2;
eqn2 = y==-tan(Entry_Angle_Of_Targeting)*x + assumed_radius;
[x_intersect, y_intersect] = solve([eqn1 eqn2],[x y]);

rounded_intersects = [abs(round(x_intersect(2),4)) round(y_intersect(2),4)]

%compute time-index for intersect, to then compute settletime
[aux, indexPeri] = min(sim_dist_target_to_sat_f(sim_time_t));
indexIntersect = find(sim_dist_target_to_sat_f(sim_time_t(1:(end/2)))>rounded_intersects(1),1,"last");
settling_time_from_simulation = sim_time_t(indexPeri)-sim_time_t(indexIntersect)

%% Camera specs (testing for new specs)
%first calculate the accuracy. Do we assumed specs for camera? assuming 4K res, 
Assumed_Size_of_Debris = 0.5;           %meter
dist_sat_to_target = sqrt((assumed_radius*1e3 +assumed_perigee*1e3)^2 -(assumed_radius*1e3)^2);

%NOT USED%object_distance = dist_sat_to_target*1E3;              %m g, object distance
%NOT USED%image_Size = 1;                         %. B, Image size
%NOT USED%object_Size = Assumed_Size_of_Debris*1E3;                %m G, Object size
%NOT USED%(object_distance*image_Size/2)/(object_Size/2) %mm, assuming that f = b. see perception course 1

%Assumed parameters, see Máni doc.
resolution_of_camera = [2E3 2E3];
PX_size = 1E-2/min(resolution_of_camera);   %CCD dimesion / resolution
Focal_Length_of_Lens = 100E-3;                 %100 mm

%computing specs
IFOV = 2*atan((PX_size/2)/Focal_Length_of_Lens);     %rad per pixel
Area_per_Pixel = 2*dist_sat_to_target*tan(IFOV/2);  %meter per pixel
Viewable_Area = resolution_of_camera*Area_per_Pixel;     %meter x meter
CameraError = (min(Viewable_Area)-Assumed_Size_of_Debris)/2; %meter
Error_Resolution = CameraError/Area_per_Pixel;  %pixels
Overshoot = IFOV*Error_Resolution*180/pi;
viewable_overshoot  = (Overshoot/100)*max(Viewable_Area);

% prints
clc
fprintf('-----------Camera specs-----------\n')
fprintf('--Máni specs given are:\n');
fprintf('assumed debris size: %.2f meters.\n',Assumed_Size_of_Debris);
fprintf('resolution: %.0f x %.0f pixels.\n',resolution_of_camera);
fprintf('Focal-length of lens: %.2f mm.\n',Focal_Length_of_Lens*1E3);
fprintf('Size of camera pixels: %.2f um.\n', PX_size*1E6);
fprintf('--computed values:\n')
fprintf('IFOV: %.2f um.\n',IFOV*1E6);
fprintf('Viewable area is %.2f by %.2f km.\n', Viewable_Area*1E-3);
fprintf('length-per-pixel is %.4f meter.\n', Area_per_Pixel);
fprintf('Overshoot is %.2f percent, resulting in a overshooting length of %.2f meters\n', Overshoot,viewable_overshoot);
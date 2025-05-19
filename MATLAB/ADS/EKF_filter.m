%% Extended Kalman Filter (EKF) with Lidar, DSN, and Landmarks
clear; clc; close all;
time1 = tic();
% Sensor enable flags
use_dsn = true;
use_lidar = true;
use_landmarks = true;

% Load SPICE kernels (Replace with your own paths)
cspice_furnsh('C:\Users\bogas\Documents\GitHub\Universitets-mappe\uni\Masters\9. semester\project\MATLAB\ADS\naif0012.tls');
cspice_furnsh('C:\Users\bogas\Documents\GitHub\Universitets-mappe\uni\Masters\9. semester\project\MATLAB\ADS\de430.bsp');
cspice_furnsh('C:\Users\bogas\Documents\GitHub\Universitets-mappe\uni\Masters\9. semester\project\MATLAB\ADS\earth_latest_high_prec.bpc');
cspice_furnsh('C:\Users\bogas\Documents\GitHub\Universitets-mappe\uni\Masters\9. semester\project\MATLAB\ADS\earthstns_itrf93_201023.bsp');

%% 1. Constants and Simulation Parameters
mu_moon = 4902.8;          % Moon's gravitational parameter (km^3/s^2)
mu_earth = 398600;         % Earth's gravitational parameter (km^3/s^2)
mu_sun = 1.32712440018e11; % Sun's gravitational parameter (km^3/s^2)
c = 299792.458;            % Speed of light (km/s)
T_s = 1;                   % Time step (s)
moon_radius = 1737.4;      % Moon's radius (km)

%% 2. Load True State Data
load('ORBdata.mat', 'orb');
true_state.time = (1:length(orb.t))' * T_s;
true_state.position = orb.XJ2000(:, 1:3)';
true_state.velocity = orb.XJ2000(:, 4:6)';
num_steps = length(true_state.time);

%% 3. Define Polar Orbit
x0_true = [true_state.position(:, 1); true_state.velocity(:, 1)];
x0_est = x0_true + [0.2; -0.5; 0.3; 0.0001; -0.0002; 0.0001];
P0 = diag([1^2, (-2)^2, 2^2, 0.005^2, (-0.01)^2, 0.01^2]);

%% 4. Sensor Configuration
ground_stations = {'DSS-14','DSS-43','DSS-63','DSS-34','DSS-45','DSS-25'};
nStations = length(ground_stations);
n_landmarks_active = 4;

%% 5. Storage for Logging
X_est = zeros(6, num_steps);
RMSE_log_pos = zeros(1, num_steps);
RMSE_log_vel = zeros(1, num_steps);

% Calculate fixed measurement sizes
meas_idx.dsn = 1:(nStations*2);          % Positions 1-6 for 3 DSN stations
meas_idx.lidar = (nStations*2)+(1:6);    % Positions 7-12 for lidar
meas_idx.landmarks = (nStations*2 + 6) + (1:(n_landmarks_active * 3)); % Positions 9+ for landmarks

% Total measurement vector size (always constant)
nMeasurements = nStations*2 + 6 + (n_landmarks_active * 3);

% Initialize measurement vectors
z_meas = zeros(nMeasurements, 1);
z_pred = zeros(nMeasurements, 1);
H = zeros(nMeasurements, 6);
R_blocks = cell(nMeasurements, 1); % For noise covariance

innovation_log = zeros(nMeasurements, num_steps);

% Set noise to Inf for disabled sensors (effectively ignores them)
if ~use_dsn
    R_blocks(meas_idx.dsn) = {Inf};
end
if ~use_lidar
    R_blocks(meas_idx.lidar) = {Inf};
end
if ~use_landmarks
    R_blocks(meas_idx.landmarks) = {Inf};
end

%% 6. Main EKF Loop
x_est = x0_est;
P_est = P0;

% Logging separate predictions for plotting
DSN_Pred_Data       = [];
Lidar_Pred_Data     = [];
Landmark_Pred_Data  = [];


for k = 1:num_steps
    t = true_state.time(k);
    
    % Get planetary positions
    r_earth_moon = get_earth_position(t);
    r_sun_moon = get_sun_position(t);
    x_true = [true_state.position(:, k); true_state.velocity(:, k)];
    
    % Process Noise Tuning
    altitude = norm(x_est(1:3)) - moon_radius;
    [~, e, i, ~, ~, theta] = cart2kep(x_est(1:3), x_est(4:6), mu_moon);
    v_norm = norm(x_est(4:6));

    % ---- Position Process Noise ----
    Q_pos_base = 1e-2; % 1 cm baseline (km)
    altitude_factor = 1 + 5*exp(-altitude/30); % Stronger at low altitudes
    eccentricity_factor = 1 + 2*e*sin(theta); % Higher noise near periselene
    Q_pos_scale = altitude_factor * eccentricity_factor;

    Q_pos = (Q_pos_base * Q_pos_scale)^2 * eye(3);

    % ---- Velocity Process Noise ----
    Q_vel_base = 1e-5; % 0.01 mm/s baseline (km/s)
    dynamic_factor = 1 + (v_norm/0.5)^2 + 0.3*abs(sin(i));
    Q_vel = (Q_vel_base * dynamic_factor)^2 * eye(3);

    Q = blkdiag(Q_pos, Q_vel);

    % Prediction Step
    [x_pred, Phi] = propagate_state_and_stm(x_est, mu_moon, T_s, mu_earth, mu_sun, r_earth_moon, r_sun_moon);
    P_pred = Phi * P_est * Phi' + Q;
    
    % Measurement Update
    z_meas(:) = 0; z_pred(:) = 0; H(:,:) = 0;
    R = spalloc(nMeasurements, nMeasurements, nMeasurements); % Reset each iteration
    
    % =============================================
    % DSN Measurements
    % =============================================
    if use_dsn
    for i = 1:nStations
        station_idx = 1 + (i-1)*2;
        idx = meas_idx.dsn((i-1)*2+1:i*2);
        R_range = (0.015)^2;            % 15 meters (0.5σ L2 norm)
        R_range_rate = (0.00001)^2;     % 0.01 mm/s

        % Get station position directly in J2000
        [station_pos_J2000, ~] = cspice_spkpos(ground_stations{i}, t, 'J2000', 'NONE', 'EARTH');
        station_pos_moon_J2000 = station_pos_J2000 - r_earth_moon;

        % Compute Earth-Station-Moon angle
        cos_angle = dot(station_pos_J2000, -r_earth_moon) / ...
                    (norm(station_pos_J2000) * norm(r_earth_moon));
        angle_deg = acosd(cos_angle);

        % Compute predicted measurements
        r_rel_pred = x_pred(1:3) - station_pos_moon_J2000;
        v_rel_pred = x_pred(4:6);
        pred_range = norm(r_rel_pred);
        pred_range_rate = dot(r_rel_pred, v_rel_pred) / max(pred_range, 1e-6);  % Safeguard

        % True measurements (with noise)
        r_rel_true = x_true(1:3) - station_pos_moon_J2000;
        v_rel_true = x_true(4:6);
        true_range = norm(r_rel_true);
        true_range_rate = dot(r_rel_true, v_rel_true) / true_range;

        z_meas(idx) = [true_range + sqrt(R_range)*randn; 
                       true_range_rate + sqrt(R_range_rate)*randn];
        z_pred(idx) = [pred_range; pred_range_rate];

        % Jacobian
        H(idx(1), 1:3) = r_rel_pred' / pred_range;  % Range
        H(idx(2), 1:3) = (v_rel_pred' - r_rel_pred' * dot(r_rel_pred, v_rel_pred) / pred_range^2) / pred_range;  % Range-rate
        H(idx(2), 4:6) = r_rel_pred' / pred_range;

        % Apply to measurement noise
        elevation_deg = 90 - angle_deg;  % Convert to elevation
        weight = min(5, 1 / (sind(elevation_deg)^2 + 0.1));   % Downweight low elevations
        R_dsn = diag([R_range, R_range_rate]) * weight;
        R(idx, idx) = R_dsn;

        % Log predictions
        DSN_Pred_Data = [DSN_Pred_Data;r_rel_pred,v_rel_pred];
    end
    end
    
    % =============================================
    % Lidar Measurements
    % =============================================
    if use_lidar
    lidar_range_noise = (0.05 + 0.03*(altitude/100))^2; % 5-8 cm noise
    lidar_rangerate_noise = (0.0007)^2; %Lidar range-rate noise

    % Define multiple beam directions (X, Y, Z axes)
    beam_directions = [0.12  0.08  0.99;  % 7° off-nadir
                      -0.15  0.10  0.98;  % 10° off-nadir
                       0.05 -0.18  0.98]'; % 11° off-nadir (120° azimuth spacing)

    for beam = 1:3
        r_sc = x_est(1:3);
        beam_vector = beam_directions(:, beam);
        beam_vector = beam_vector / norm(beam_vector);

        % Approximate beam-surface intercept (always defined)
        r_surface = moon_radius * (r_sc / norm(r_sc)) + ...
                    beam_vector * sqrt(norm(r_sc)^2 - moon_radius^2);

        % True measurements
        r_rel = x_true(1:3) - r_surface;
        v_rel = x_true(4:6);
        range_true = norm(r_rel);
        range_rate_true = dot(r_rel, v_rel)/range_true;

        % Predicted measurements
        r_rel_pred = x_pred(1:3) - r_surface;
        v_rel_pred = x_pred(4:6);
        range_pred = norm(r_rel_pred);
        range_rate_pred = dot(r_rel_pred, v_rel_pred)/range_pred;

        % Store measurements (positions 7-12)
        meas_idx_beam = meas_idx.lidar(1) + (beam-1)*2;
        z_meas(meas_idx_beam:meas_idx_beam+1) = ...
            [range_true + sqrt(lidar_range_noise)*randn;
            range_rate_true + sqrt(lidar_rangerate_noise)*randn];
        
        z_pred(meas_idx_beam:meas_idx_beam+1) = ...
             [range_pred; range_rate_pred];
        
        % Jacobian for this beam (FIXED PARENTHESES)
        H_lidar(beam*2-1, 1:3) = r_rel_pred'/range_pred;
        H_lidar(beam*2, 1:3) = ...
           (v_rel_pred' - (r_rel_pred'*dot(r_rel_pred,v_rel_pred))/range_pred^2)/range_pred;
        H_lidar(beam*2, 4:6) = r_rel_pred'/range_pred;

        % Log predictions
        Lidar_Pred_Data = [Lidar_Pred_Data; r_rel_pred, v_rel_pred];

    end
    
    % Store complete Jacobian
    H(meas_idx.lidar(1):meas_idx.lidar(1)+5, :) = H_lidar;
    
    % Noise covariance (diagonal for 3 beams)
    R_lidar = kron(eye(3), diag([lidar_range_noise, lidar_rangerate_noise]));
    R(meas_idx.lidar(1):meas_idx.lidar(1)+5, meas_idx.lidar(1):meas_idx.lidar(1)+5) = R_lidar;
    end
    
    % =============================================
    % Landmark Measurements
    % =============================================
    if use_landmarks
    % Large bank of 22 landmarks covering 40°–48° latitudes
    landmark_coords_deg = [
        45, 0;
        50, 15;
        45, 30;
        40, 45;
        45, 60;
        50, 75;
        45, 90;
        40, 105
        45, 120;
        45, 135;
        45, 150;
        45, 180;
        45, 210;
        45, 240;
        45, 270;
        45, 300;
        45, 330;
        40, 15;
        40, 75;
        40, 135;
        40, 195;
        40, 255;
        40, 315;
        48, 45;
        48, 105;
        48, 225;
        48, 285
    ];
    R_landmark = diag([0.2, 0.2, 0.3].^2); %Noise
    moon_radius = 1737.4;
    nFull = size(landmark_coords_deg, 1);
    landmark_pos_mci = zeros(nFull, 3);

    % Convert each landmark to MCI at current time
    for i = 1:nFull
        lat = landmark_coords_deg(i, 1);
        lon = landmark_coords_deg(i, 2);

        r_mcmf = moon_radius * [
            cosd(lat)*cosd(lon);
            cosd(lat)*sind(lon);
            sind(lat)
        ];

        landmark_pos_mci(i, :) = transform_MCMF_to_MCI(r_mcmf, t)';
    end

    % Select the 3 closest landmarks to x_pred
    dists = vecnorm(landmark_pos_mci - x_pred(1:3)', 2, 2);
    [~, sorted_idx] = sort(dists);
    best_idx = sorted_idx(1:4);

    for j = 1:length(best_idx)
        i = best_idx(j);
        idx = meas_idx.landmarks((j-1)*3+1 : j*3);

        land_mci = landmark_pos_mci(i, :)';

        % Distance-dependent noise scaling
        range = norm(x_pred(1:3) - land_mci);
        scale = min(5, 1 + (range / 50)); % cap max scaling
        R_landmark_scaled = scale^2 * R_landmark;

        % True and predicted measurement
        z_true_i = x_true(1:3) - land_mci;
        z_pred_i = x_pred(1:3) - land_mci;

        z_meas(idx) = z_true_i + mvnrnd([0; 0; 0], R_landmark_scaled)';
        z_pred(idx) = z_pred_i;

        H(idx, 1:3) = eye(3);
        R(idx, idx) = R_landmark_scaled;

        Landmark_Pred_Data = [Landmark_Pred_Data; z_pred];
    end
    end

    % Robust Kalman Update
    innovation = z_meas - z_pred;
    S = H * P_pred * H' + R;

    % Check for valid measurements
    valid_meas = ~isinf(diag(R));
    if ~any(valid_meas)
       warning('No valid measurements at step %d', k);
        x_est = x_pred;
        P_est = P_pred;
    else
        % Regularized inversion
        [U,S_svd,V] = svd(S(valid_meas,valid_meas));
        sv_threshold = max(size(S)) * eps(norm(S));
        inv_S = V * diag(1./max(diag(S_svd), sv_threshold)) * U';
        K = P_pred * H(valid_meas,:)' * inv_S;
        x_est = x_pred + K * innovation(valid_meas);
        P_est = (eye(6) - K*H(valid_meas,:)) * P_pred;
    end

    % Ensure covariance remains positive definite
    [V,D] = eig((P_est + P_est')/2);
    min_eig = max(diag(D), 1e-6); % Hard floor at 1cm position / 0.1mm/s velocity
    P_est = V * diag(min_eig) * V';
    
    % Log results
    innovation_log(:, k) = innovation;
    X_est(:, k) = x_est;
    RMSE_log_pos(k) = norm(x_true(1:3) - x_est(1:3));
    RMSE_log_vel(k) = norm(x_true(4:6) - x_est(4:6));
end

%% 7. Results and Plots
pos_error_all = true_state.position - X_est(1:3, :);
vel_error_all = true_state.velocity - X_est(4:6, :);
RMSE_pos = sqrt(mean(vecnorm(pos_error_all, 2, 1).^2));
RMSE_vel = sqrt(mean(vecnorm(vel_error_all, 2, 1).^2));

fprintf('Position RMSE: %.4f km\n', RMSE_pos);
fprintf('Velocity RMSE: %.4f km/s\n', RMSE_vel);

%% Enhanced RMSE Visualization
figure('Position', [100 100 800 600]);

% Shared parameters
window_size = max(1, round(num_steps/100)); % Auto-adjust window to 1% of data
line_width = 1.5;
font_size = 10;

% 1. Position RMSE 
subplot(2,1,1);
smoothed_pos = movmean(RMSE_log_pos, window_size);

plot(true_state.time, RMSE_log_pos, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.3); % Light gray raw
hold on;
plot(true_state.time, smoothed_pos, 'b', 'LineWidth', line_width);
ylabel('Position (km)');
title('Position RMSE');
grid on;
ylim([0 max(RMSE_log_pos)*1.1]); % Auto-scale with 10% margin

% 2. Velocity RMSE
subplot(2,1,2);
smoothed_vel = movmean(RMSE_log_vel, window_size);

plot(true_state.time, RMSE_log_vel, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.3); % Light gray raw
hold on;
plot(true_state.time, smoothed_vel, 'r', 'LineWidth', line_width);
ylabel('Velocity (km/s)');
xlabel('Time (s)');
title('Velocity RMSE');
grid on;
ylim([0 max(RMSE_log_vel)*1.1]); % Auto-scale with 10% margin

% Link axes for synchronized zoom/pan
linkaxes([subplot(2,1,1), subplot(2,1,2)], 'x');

% Add unified legend
legend_labels = {'Raw measurements', [num2str(window_size) '-pt moving avg']};
legend(legend_labels, 'Position', [0.82 0.45 0.1 0.1]);

cspice_kclear;

toc(time1)
%% --- Functions ----------------------------------------------------------
function dxdt = orbital_dynamics(~, x, mu_moon, mu_earth, mu_sun, r_earth_moon, r_sun_moon)
    r = x(1:3);  % Satellite position in Moon-centered frame
    v = x(4:6);  % Satellite velocity in Moon-centered frame
    
    % Two-body acceleration (Moon)
    r_norm = norm(r);
    a_two_body = -mu_moon * r / r_norm^3;
    
    % Earth's gravitational perturbation
    r_earth_sat = r - r_earth_moon;  % Relative position of Earth w.r.t. satellite
    r_earth_sat_norm = norm(r_earth_sat);
    a_earth = -mu_earth * (r_earth_sat / r_earth_sat_norm^3 - r_earth_moon / norm(r_earth_moon)^3);
    
    % Sun's gravitational perturbation
    r_sun_sat = r - r_sun_moon;  % Relative position of Sun w.r.t. satellite
    r_sun_sat_norm = norm(r_sun_sat);
    a_sun = -mu_sun * (r_sun_sat / r_sun_sat_norm^3 - r_sun_moon / norm(r_sun_moon)^3);
    
    % Moon's J2 perturbation
    J2 = 1.083e-3;  % Moon's J2 coefficient
    R_moon = 1737.4;  % Moon's radius (km)
    z = r(3);
    a_J2 = -3/2 * J2 * mu_moon * R_moon^2 / r_norm^5 * [r(1) * (5*z^2/r_norm^2 - 1);
                                                        r(2) * (5*z^2/r_norm^2 - 1);
                                                        r(3) * (5*z^2/r_norm^2 - 3)];
    
    % Total acceleration
    a_total = a_two_body + a_earth + a_sun + a_J2;
    
    % State derivative
    dxdt = [v; a_total];
end

function A = compute_A_matrix(x, mu_moon, mu_earth, mu_sun, r_earth_moon, r_sun_moon)
    % Extract satellite position and validate
    r = x(1:3);
    r_norm = norm(r);
    
    % Add to compute_A_matrix after the other terms:
    J2 = 1.083e-3;
    R_moon = 1737.4;
    r_norm = norm(r);
    z = r(3);

    % Safeguard against invalid positions
    %if r_norm < 1737.4 || any(isnan(r)) || any(isinf(r))
    %    error('Invalid position in state vector');
    %end
    
    I3 = eye(3);
    
    %% 1. Moon's Two-Body Term
    r_normalized = r / r_norm;
    dAdr_moon = -mu_moon / (r_norm^3) * (I3 - 3 * (r_normalized * r_normalized'));
    
    %% 2. Earth's Perturbation Term
    r_sat_to_earth = r_earth_moon - r;
    earth_sat_norm = norm(r_sat_to_earth);
    earth_moon_norm = norm(r_earth_moon);
    
    % Regularize denominators to avoid division by zero
    earth_sat_norm = norm(earth_sat_norm);
    earth_moon_norm = norm(earth_moon_norm);
    
    term1 = I3 / (earth_sat_norm^3) - 3 * (r_sat_to_earth * r_sat_to_earth') / (earth_sat_norm^5);
    term2 = I3 / (earth_moon_norm^3) - 3 * (r_earth_moon * r_earth_moon') / (earth_moon_norm^5);
    dAdr_earth = -mu_earth * (term1 - term2);
    
    %% 3. Sun's Perturbation Term
    r_sat_to_sun = r_sun_moon - r;
    sun_sat_norm = norm(r_sat_to_sun);
    sun_moon_norm = norm(r_sun_moon);
    
    % Regularize denominators
    sun_sat_norm = norm(sun_sat_norm);
    sun_moon_norm = norm(sun_moon_norm);
    
    % Reuse computations for efficiency
    inv_sun_sat_norm3 = 1 / (sun_sat_norm^3);
    inv_sun_moon_norm3 = 1 / (sun_moon_norm^3);
    
    term1_sun = I3 * inv_sun_sat_norm3 - 3 * (r_sat_to_sun * r_sat_to_sun') * inv_sun_sat_norm3 / (sun_sat_norm^2);
    term2_sun = I3 * inv_sun_moon_norm3 - 3 * (r_sun_moon * r_sun_moon') * inv_sun_moon_norm3 / (sun_moon_norm^2);
    dAdr_sun = -mu_sun * (term1_sun - term2_sun);
    
    %% 4. J2 Perturbation
    dAdr_J2 = compute_J2_partials(r, mu_moon);
    
    %% Combine All Terms
    dAdr_total = dAdr_moon + dAdr_earth + dAdr_sun + dAdr_J2;
    A = [ zeros(3),      I3;
          dAdr_total, zeros(3) ];
end

function dAdr_J2 = compute_J2_partials(r, mu_moon)
    J2 = 1.083e-3;
    R_moon = 1737.4;
    r_norm = norm(r);
    z = r(3);
    
    common_factor = -3/2 * J2 * mu_moon * R_moon^2 / r_norm^7;
    
    term1 = eye(3) * (7*z^2 - r_norm^2);
    term2 = 5 * (r*r') * (7*z^2 - 3*r_norm^2) / r_norm^2;
    term3 = [0,     0,     2*r(1)*z;
             0,     0,     2*r(2)*z;
             2*r(1)*z, 2*r(2)*z, 6*r(3)*z - 2*r_norm^2];
    
    dAdr_J2 = common_factor * (term1 - term2 + term3);
end

function r_MCI = transform_MCMF_to_MCI(r_MCMF, t)
    omega_moon = 2*pi / (27.3*86400); % Moon's rotation rate (rad/s)
    theta = omega_moon * t;
    Rz = [ cos(theta),  sin(theta), 0;
          -sin(theta),  cos(theta), 0;
           0,           0,          1 ];
    r_MCI = Rz * r_MCMF;
end

function [x_next, Phi_next] = rk4_step(f, x, Phi, mu_moon, mu_earth, mu_sun, r_earth_moon, r_sun_moon, dt)
    % Combine state and STM into a single vector
    X = [x; Phi(:)];
    
    % RK4 stages
    k1 = f(0, X, mu_moon, mu_earth, mu_sun, r_earth_moon, r_sun_moon);
    k2 = f(0, X + 0.5 * dt * k1, mu_moon, mu_earth, mu_sun, r_earth_moon, r_sun_moon);
    k3 = f(0, X + 0.5 * dt * k2, mu_moon, mu_earth, mu_sun, r_earth_moon, r_sun_moon);
    k4 = f(0, X + dt * k3, mu_moon, mu_earth, mu_sun, r_earth_moon, r_sun_moon);
    
    % Update state and STM
    X_next = X + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4);
    
    % Extract state and STM
    x_next = X_next(1:6);
    Phi_next = reshape(X_next(7:end), 6, 6);
end

function [x_pred, Phi] = propagate_state_and_stm(x_est, mu_moon, T_s, mu_earth, mu_sun, r_earth_moon, r_sun_moon)
    % Define the combined dynamics function
    function dxdt = combined_dynamics(~, X, mu_moon, mu_earth, mu_sun, r_earth_moon, r_sun_moon)
        x = X(1:6);
        Phi = reshape(X(7:end), 6, 6);
        dxdt_state = orbital_dynamics([], x, mu_moon, mu_earth, mu_sun, r_earth_moon, r_sun_moon);
        A = compute_A_matrix(x, mu_moon, mu_earth, mu_sun, r_earth_moon, r_sun_moon);
        dPhidt = A * Phi;
        dxdt = [dxdt_state; dPhidt(:)];
    end
    
    % Perform RK4 integration
    [x_pred, Phi] = rk4_step(@combined_dynamics, x_est, eye(6), mu_moon, mu_earth, mu_sun, r_earth_moon, r_sun_moon, T_s);
end

function [a, e, i, Omega, omega, theta] = cart2kep(r, v, mu)
    % Angular momentum vector
    h = cross(r, v);
    h_norm = norm(h);

    % Eccentricity vector
    e_vec = ((norm(v)^2 - mu/norm(r)) * r - dot(r, v) * v) / mu;
    e = norm(e_vec);

    % Semi-major axis
    a = 1 / (2/norm(r) - norm(v)^2/mu);

    % Inclination
    i = acos(h(3)/h_norm);

    % Node line vector
    n = cross([0; 0; 1], h);
    n_norm = norm(n);

    % RAAN (Omega)
    Omega = atan2(n(2), n(1));
        if Omega < 0
           Omega = Omega + 2*pi;
        end

    % Argument of periapsis (omega)
    omega = acos(dot(n, e_vec)/(n_norm*e));
    if e_vec(3) < 0
        omega = 2*pi - omega;
    end

    % True anomaly (theta)
    theta = acos(dot(e_vec, r)/(e*norm(r)));
    if dot(r, v) < 0
        theta = 2*pi - theta;
    end

    % Handle special cases
    if e < 1e-10
        % Circular orbit
        omega = 0;
        if i < 1e-10
            % Equatorial circular orbit
            Omega = 0;
            theta = atan2(r(2), r(1));
        else
            % Non-equatorial circular orbit
            theta = acos(dot(n, r)/(n_norm*norm(r)));
            if r(3) < 0
                theta = 2*pi - theta;
            end
        end
    end
end

function r_earth_moon = get_earth_position(t)
    % Use SPICE to get Earth's position relative to the Moon at time t
    [state, ~] = cspice_spkezr('EARTH', t, 'J2000', 'NONE', 'MOON');
    r_earth_moon = state(1:3);  % Position vector (km)
end

function r_sun_moon = get_sun_position(t)
    % Use SPICE to get Sun's position relative to the Moon at time t
    [state, ~] = cspice_spkezr('SUN', t, 'J2000', 'NONE', 'MOON');
    r_sun_moon = state(1:3);  % Position vector (km)
end

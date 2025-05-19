%General parameters
Oval_Amount     =   1e4;
Oval_Size_max   =   (25e2/20)/2;
Oval_Size_min   =   (10e2/20)/2;

%log parameters
mu                              = log(1);  % Log-space mean
sigma                           = 1;  % Log-space spread
oversample_factor               = 1;

%Gaussian
Gauss_data = abs(randn([Oval_Amount,1]));
Gauss_data = Gauss_data/max(Gauss_data);
%Gauss_data = Gauss_data*(Oval_Size_max-Oval_Size_min)+Oval_Size_max;
Gauss_data = sortrows(Gauss_data,"ascend");

% log normal
log_data = lognrnd(mu, sigma, [Oval_Amount,1]);
log_data = log_data/max(log_data);
%log_data = log_data*(Oval_Size_max-Oval_Size_min)+Oval_Size_min;
log_data = sortrows(log_data,"ascend");
log_data = log_data;

bins = 20;
figure;
histogram(Gauss_data,"NumBins",bins),
hold on
histogram(log_data,"NumBins",bins), legend("Gaussian","Log-normal")


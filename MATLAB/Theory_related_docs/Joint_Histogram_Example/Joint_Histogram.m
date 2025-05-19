clc,close;
% Generate 2D correlated Gaussian data
mu = [0 0];
Sigma = [1 0.8; 0.8 1];
data = mvnrnd(mu, Sigma, 10000);
x = data(:,1);
y = data(:,2);

% Create tiled layout
figure('Position', [100 100 800 800]);
t = tiledlayout(4,4, 'TileSpacing','compact', 'Padding','compact');


% Joint histogram
ax2 = nexttile([3 3]);
hist3(ax2, [x y], 'Nbins', [50 50], 'CdataMode','auto', 'FaceColor','interp');
view(ax2, 2);
xlabel(ax2, 'X ~ N(0,1)');
ylabel(ax2, 'Y ~ N(0,1)');
colormap(ax2);
colorbar(ax2);

% Top marginal (PDF of x)
ax1 = nexttile([1 3]);
x_vals = linspace(-4, 4, 100);
plot(ax1, x_vals, normpdf(x_vals, 0, 1), 'k', 'LineWidth', 1.5);
ax1.XTick = [];
ax1.YTick = [];
axis tight;
box off;

% Right marginal (PDF of y)
ax3 = nexttile([3 1]);
y_vals = linspace(-4, 4, 100);
plot(ax3, normpdf(y_vals, 0, 1), y_vals, 'k', 'LineWidth', 1.5);
ax3.XTick = [];
ax3.YTick = [];
axis tight;
box off;

exportgraphics(gcf,'Joint_Histgogram_Example.png','Resolution',96);
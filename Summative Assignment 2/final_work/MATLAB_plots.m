clear
close all

%% 

% =======================
% Load audio (assumed consistent)
% =======================
[dry_audio, Fs] = audioread("./dry/dry_0a282672-c22c-59ff-faaa-ff9eb73fc8e6.wav");

[delay_audio, ~] = audioread("./delay/delay_0a282672-c22c-59ff-faaa-ff9eb73fc8e6.wav");
[reverb_audio, ~] = audioread("./reverb/reverb_0a282672-c22c-59ff-faaa-ff9eb73fc8e6.wav");
[distortion_audio, ~] = audioread("./distortion/distortion_0a282672-c22c-59ff-faaa-ff9eb73fc8e6.wav");
[chorus_audio, ~] = audioread("./chorus/chorus_0a282672-c22c-59ff-faaa-ff9eb73fc8e6.wav");

effects = {delay_audio, reverb_audio, distortion_audio, chorus_audio};
titles = {'Delay vs Dry', 'Reverb vs Dry', 'Distortion vs Dry', 'Chorus vs Dry'};

% =======================
% Parameters
% =======================
eps_val = 1e-10;
color_range = [-70 0];
diff_range = [-10 30];

% =======================
% Dry reference
% =======================
[s_dry, f, t] = melSpectrogram(dry_audio, Fs);
s_dry_db = 10*log10(s_dry + eps_val);

% =======================
% Plot spectrograms (A4 portrait, report-ready)
% =======================
figure(1)
clf

% Set A4 portrait size (in cm)
set(gcf, 'Units', 'centimeters', 'Position', [2 2 21 29.7]);
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [21 29.7]);
set(gcf, 'PaperPosition', [0 0 21 29.7]);

tiledlayout(3,2, 'Padding', 'compact', 'TileSpacing', 'compact');

% Font size for report readability
fs = 10;

% Dry
nexttile
imagesc(t, f, s_dry_db);
title('Dry Audio (Reference)', 'FontSize', fs);
set(gca, 'YDir', 'normal', 'FontSize', fs);
clim(color_range);
colormap('turbo');

% Store mean differences
mean_diffs = zeros(length(f), length(effects));

for i = 1:length(effects)
    nexttile
    
    [s_eff, ~, ~] = melSpectrogram(effects{i}, Fs);
    s_eff_db = 10*log10(s_eff + eps_val);
    
    diff_db = s_eff_db - s_dry_db;
    
    mean_diffs(:, i) = mean(diff_db, 2);
    
    imagesc(t, f, diff_db);
    title(titles{i}, 'FontSize', fs);
    set(gca, 'YDir', 'normal', 'FontSize', fs);
    clim(diff_range);
end

% Shared colorbar
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Spectral Difference (dB)';
cb.FontSize = fs;

% Optional: global title
sgtitle('Mel Spectrogram Comparison of Audio Effects', 'FontSize', 12, 'FontWeight', 'bold');

% =======================
% Mean spectral difference
% =======================
figure(2)
hold on

for i = 1:length(effects)
    plot(f, mean_diffs(:, i),'LineWidth', 1.4, 'DisplayName', titles{i});
end

xlabel('Frequency (Hz)');
ylabel('Mean Spectral Difference (dB)');
title('Average Spectral Difference Relative to Dry Signal');
fontsize(14,"points");
legend;
grid on;

saveas(1,"./images/mel_spec_graphs",'png');
saveas(2,"./images/avg_spec_differences",'png');

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
diff_range = [-10 50];

% =======================
% Dry reference
% =======================
[s_dry, f, t] = melSpectrogram(dry_audio, Fs);
s_dry_db = 10*log10(s_dry + eps_val);

% =======================
% Plot spectrograms
% =======================
figure(1)
tiledlayout(3,2)

% Dry
nexttile
imagesc(t, f, s_dry_db);
title('Dry Audio (Reference)');
set(gca, 'YDir', 'normal');
clim(color_range);
colormap('turbo');

% Store mean differences
mean_diffs = zeros(length(f), length(effects));

for i = 1:length(effects)
    nexttile
    
    [s_eff, ~, ~] = melSpectrogram(effects{i}, Fs);
    s_eff_db = 10*log10(s_eff + eps_val);
    
    diff_db = s_eff_db - s_dry_db;
    
    % Store mean over time
    mean_diffs(:, i) = mean(diff_db, 2);
    
    imagesc(t, f, diff_db);
    title(titles{i});
    set(gca, 'YDir', 'normal');
    clim(diff_range);
end

cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Spectral Difference (dB)';

% =======================
% Mean spectral difference
% =======================
figure(2)
hold on

for i = 1:length(effects)
    plot(f, mean_diffs(:, i), 'DisplayName', titles{i});
end

xlabel('Frequency (Hz)');
ylabel('Mean Spectral Difference (dB)');
title('Average Spectral Difference Relative to Dry Signal');
legend;
grid on;

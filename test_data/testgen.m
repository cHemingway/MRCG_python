% Function to save output of original MRCG implementation on .wav file
% Requires MRCG_features from http://web.cse.ohio-state.edu/pnl/shareware/chenj-taslp14/ 
% to be in your MATLAB userpath folder, e.g. My Documents/MATLAB

% Chris Hemingway 2019, MIT License

clear; clc;

% Add original MATLAB code to path
addpath(fullfile(userpath,'MRCG_features'));

% Open file and read audio
wavfilename = 'SNR103F3MIC021002_ch01.wav';
[sig, sampFreq] = audioread(wavfilename);

% Calculate overall MRCG
% MRCG is of format [all_cochleas;del;ddel]
mrcg = MRCG_features(sig, sampFreq);

% Calculate beta, gammatone, and M for comparison
beta = 1000 / sqrt( sum(sig .^ 2) / length(sig) );
sig = sig .* beta;
sig = reshape(sig, length(sig), 1);
g = gammatone(sig, 64, [50 8000], sampFreq); % Gammatone filterbank responses
M = floor(length(sig)/160);  % number of time frames

% Save to .mat file
outname = strrep(wavfilename, '.wav','.mat');
save(outname,'mrcg','beta','g','M');


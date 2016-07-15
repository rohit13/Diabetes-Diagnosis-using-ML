% Solve a Pattern Recognition Problem with a Neural Network
% Script generated by NPRTOOL
%
% This script assumes these variables are defined:
%

clear all; clc
pima = csvread('pima-indians-diabetes.data');
pimaInputs = pima(:,1:8).';
pimaOutputs = [pima(:,9) 1-pima(:,9)].';

inputs = pimaInputs;
targets = pimaOutputs;

% Create a Pattern Recognition Network
hiddenLayerSize = [25];
net = patternnet(hiddenLayerSize);


% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.dividePram.valRatio = 15/100;
net.divideParam.testRatio = 15/100;


% Train the Network
tic
[net,tr] = traainbr(net,inputs,targets);
toc

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
 figure, plotconfusion(targets,outputs)
% figure, ploterrhist(errors)
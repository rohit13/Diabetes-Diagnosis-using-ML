clear all; clc
%% Load Data
pimaData = csvread('pima-indians-diabetes.data');
%% TrainCount
trainCount = round(length(pimaData)*0.6); % 60% of data is for training
%% Train
trainData = pimaData(1:trainCount,1:8);
trainLabels = char('D'*pimaData(1:trainCount,9) + 'H'*(1-pimaData(1:trainCount,9)));
tic
prior = [0.4 0.6];
Mdl = fitcnb(trainData,trainLabels,...
    'PredictorNames',{  'Number of times pregnant',...
                        'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',...
                        'Diastolic blood pressure (mm Hg)',...
                        'Triceps skin fold thickness (mm)',...
                        '2-Hour serum insulin (mu U/ml)',...
                        'Body mass index (weight in kg/(height in m)^2)',...
                        'Diabetes pedigree function',...
                        'Age (years)'},'Prior',prior);
toc
%% Test
testData = pimaData(trainCount+1:end,1:8);
prediction = predict(Mdl, testData);
%% PlotConfusion
target = pimaData(trainCount+1:end,9).';
output = zeros(1, length(prediction));
output(prediction == 'D') = 1;
plotconfusion(target, output);
%% Plot Scatter Graph
figure
gscatter(testData(:,8),testData(:,3),prediction);
xlabel('Age (years)')
ylabel('Blood pressure (mm Hg)')
hold off
%% compare loss in two cnb models
defaultPriorMdl = Mdl;
defaultPriorMdl.Prior = [0.5, 0.5];
defaultCVMdl = crossval(defaultPriorMdl);
defaultLoss = kfoldLoss(defaultCVMdl);
CVMdl = crossval(Mdl);
Loss = kfoldLoss(CVMdl);

disp(strcat('defaultLoss: ', num2str(defaultLoss)));
disp(strcat('Loss: ', num2str(Loss)));

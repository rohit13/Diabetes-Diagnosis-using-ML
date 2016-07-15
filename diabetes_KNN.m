tic;
 
% extracting test data from the data set---- 40% roughly (40% of 768)-----%
sample = csvread('pima-indians-diabetes.data',0,0,[0,0,306,7]);
 
% extracting training data from the data set---- 60% roughly (60% of 768)-----%
train = csvread('pima-indians-diabetes.data',307,0,[307,0,767,7]);
 
% Group vector, containg class of the train data----- %
group = csvread('pima-indians-diabetes.data',307,8,[307,8,767,8]);
 
% Vector, containg class of the sample/test data----- %
resultSample = csvread('pima-indians-diabetes.data',0,8,[0,8,306,8]);
 
% Apllying KNN classifier %
class = knnclassify(sample, train, group,7,'euclidean','nearest');
 
% Transpose of the resultSample vector %
resultSampleT = resultSample.';
 
%Transpose of the output vector of the classifier result%
classT = class.';
 
% Extracting dimension of the classifier out in a vector%
d = size(class);
 
n = d(1,1);
 
i=1;
 
count = 0;
 
% Comparing the the 2 vectors %
while i < n
    if class(i,1) == resultSample(i,1)
         count = count +1 ;
    end    
  i = i+1;  
end
 
accuracy = (count/n)*100;
 
% Plotting the confusion matrix%
plotconfusion(resultSampleT, classT);
 
toc;

%% =====================================================
% COEN807 TERM PROJECT
% Machine Learning-Based Stroke Prediction
% Comparative Analysis of Classification Algorithms
% ======================================================

clc;
clear;
close all;

%% LOAD DATASET

data = readtable('healthcare-dataset-stroke-data.csv');

disp('Dataset Preview');
head(data)

fprintf('\nDataset Size: %d rows x %d columns\n',size(data,1),size(data,2));

%% REMOVE ID COLUMN

data.id = [];

%% HANDLE MISSING VALUES

data.bmi = fillmissing(data.bmi,'mean');

%% ENCODE CATEGORICAL VARIABLES

catVars = {'gender','ever_married','work_type',...
           'Residence_type','smoking_status'};

for i = 1:length(catVars)

    data.(catVars{i}) = grp2idx(categorical(data.(catVars{i})));

end

%% FEATURES AND TARGET

X = data(:,1:end-1);
Y = data.stroke;

featureNames = X.Properties.VariableNames;

X = table2array(X);

%% NORMALIZATION

X = normalize(X);

%% TRAIN TEST SPLIT

cv = cvpartition(Y,'HoldOut',0.20);

XTrain = X(training(cv),:);
YTrain = Y(training(cv));

XTest = X(test(cv),:);
YTest = Y(test(cv));

fprintf('\nTraining Samples = %d\n',length(YTrain));
fprintf('Testing Samples = %d\n',length(YTest));

%% RESULTS STORAGE

ModelNames = {};
Accuracy = [];
Precision = [];
Recall = [];
F1Score = [];

%% =====================================================
% FUNCTION FOR METRICS
%% =====================================================

calcMetrics = @(CM) deal( ...
    CM(2,2)/(CM(2,2)+CM(1,2)+eps), ...
    CM(2,2)/(CM(2,2)+CM(2,1)+eps), ...
    2*((CM(2,2)/(CM(2,2)+CM(1,2)+eps))*...
    (CM(2,2)/(CM(2,2)+CM(2,1)+eps))) / ...
    ((CM(2,2)/(CM(2,2)+CM(1,2)+eps))+...
    (CM(2,2)/(CM(2,2)+CM(2,1)+eps))+eps));

%% =====================================================
% MODEL 1 - DECISION TREE
%% =====================================================

disp('Training Decision Tree...')

TreeModel = fitctree(XTrain,YTrain);

YPred = predict(TreeModel,XTest);

CM = confusionmat(YTest,YPred);

Acc = sum(diag(CM))/sum(CM(:))*100;

[P,R,F] = calcMetrics(CM);

ModelNames{end+1}='Decision Tree';
Accuracy(end+1)=Acc;
Precision(end+1)=P;
Recall(end+1)=R;
F1Score(end+1)=F;

figure;
confusionchart(YTest,YPred);
title('Decision Tree');

%% =====================================================
% MODEL 2 - RANDOM FOREST
%% =====================================================

disp('Training Random Forest...')

RFModel = TreeBagger(100,...
    XTrain,...
    YTrain,...
    'Method','classification',...
    'OOBPrediction','on',...
    'OOBPredictorImportance','on');

YPred = predict(RFModel,XTest);

YPred = str2double(YPred);

CM = confusionmat(YTest,YPred);

Acc = sum(diag(CM))/sum(CM(:))*100;

[P,R,F] = calcMetrics(CM);

ModelNames{end+1}='Random Forest';
Accuracy(end+1)=Acc;
Precision(end+1)=P;
Recall(end+1)=R;
F1Score(end+1)=F;

figure;
confusionchart(YTest,YPred);
title('Random Forest');

%% =====================================================
% MODEL 3 - SVM
%% =====================================================

disp('Training SVM...')

SVMModel = fitcsvm(XTrain,...
    YTrain,...
    'KernelFunction','rbf',...
    'Standardize',true);

YPred = predict(SVMModel,XTest);

CM = confusionmat(YTest,YPred);

Acc = sum(diag(CM))/sum(CM(:))*100;

[P,R,F] = calcMetrics(CM);

ModelNames{end+1}='SVM';
Accuracy(end+1)=Acc;
Precision(end+1)=P;
Recall(end+1)=R;
F1Score(end+1)=F;

figure;
confusionchart(YTest,YPred);
title('SVM');

%% =====================================================
% MODEL 4 - KNN
%% =====================================================

disp('Training KNN...')

KNNModel = fitcknn(XTrain,YTrain,...
    'NumNeighbors',5);

YPred = predict(KNNModel,XTest);

CM = confusionmat(YTest,YPred);

Acc = sum(diag(CM))/sum(CM(:))*100;

[P,R,F] = calcMetrics(CM);

ModelNames{end+1}='KNN';
Accuracy(end+1)=Acc;
Precision(end+1)=P;
Recall(end+1)=R;
F1Score(end+1)=F;

figure;
confusionchart(YTest,YPred);
title('KNN');

%% =====================================================
% MODEL 5 - LOGISTIC REGRESSION
%% =====================================================

disp('Training Logistic Regression...')

LogModel = fitclinear(XTrain,...
    YTrain,...
    'Learner','logistic');

YPred = predict(LogModel,XTest);

CM = confusionmat(YTest,YPred);

Acc = sum(diag(CM))/sum(CM(:))*100;

[P,R,F] = calcMetrics(CM);

ModelNames{end+1}='Logistic Regression';
Accuracy(end+1)=Acc;
Precision(end+1)=P;
Recall(end+1)=R;
F1Score(end+1)=F;

figure;
confusionchart(YTest,YPred);
title('Logistic Regression');

%% =====================================================
% RESULTS TABLE
%% =====================================================

Results = table(ModelNames',...
                Accuracy',...
                Precision',...
                Recall',...
                F1Score',...
                'VariableNames',...
                {'Model','Accuracy','Precision','Recall','F1Score'});

disp(' ');
disp('==============================');
disp('MODEL COMPARISON RESULTS');
disp('==============================');
disp(Results);

%% =====================================================
% ACCURACY COMPARISON
%% =====================================================

figure;

bar(Accuracy)

xticklabels(ModelNames)

xtickangle(45)

ylabel('Accuracy (%)')

title('Model Accuracy Comparison')

grid on

%% =====================================================
% F1 SCORE COMPARISON
%% =====================================================

figure;

bar(F1Score)

xticklabels(ModelNames)

xtickangle(45)

ylabel('F1 Score')

title('Model F1 Score Comparison')

grid on

%% =====================================================
% BEST MODEL
%% =====================================================

[BestAccuracy,idx] = max(Accuracy);

fprintf('\nBest Model = %s\n',ModelNames{idx});
fprintf('Best Accuracy = %.2f %%\n',BestAccuracy);

%% =====================================================
% FEATURE IMPORTANCE
%% =====================================================

figure;

imp = RFModel.OOBPermutedPredictorDeltaError;

bar(imp)

xticks(1:length(featureNames))

xticklabels(featureNames)

xtickangle(45)

ylabel('Importance')

title('Random Forest Feature Importance')

grid on

%% =====================================================
% OOB ERROR
%% =====================================================

figure;

oobErrorBaggedEnsemble = oobError(RFModel);

plot(oobErrorBaggedEnsemble,'LineWidth',2)

xlabel('Number of Trees')

ylabel('Out-of-Bag Error')

title('Random Forest OOB Error')

grid on

%% =====================================================
% EXPORT RESULTS
%% =====================================================

writetable(Results,'Stroke_Project_Results.xlsx');

disp(' ');
disp('Results exported to Stroke_Project_Results.xlsx');

disp(' ');
disp('PROJECT EXECUTED SUCCESSFULLY');
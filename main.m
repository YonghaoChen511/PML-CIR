%initialization 初始化
clear;
close all
% rng('default');
load('.\datasets\medical.mat');

noisy_num = 1;
[pLabels, noisy_nums] = rand_noisy_num(target,noisy_num);
target(target==-1)=0;
pLabels(pLabels==-1)=0;

N = length(target);
indices = crossvalind('Kfold', 1:N ,5);  
%训练集 测试集
test_idxs = (indices == 1);
train_idxs = ~test_idxs;
        
train_data=data(train_idxs,:);
train_target=pLabels(train_idxs,:);
true_target = target(train_idxs,:);
test_data=data(test_idxs,:);test_target=target(test_idxs,:);

% pre-processing 归一化
[train_data, settings]=mapminmax(train_data');
test_data=mapminmax('apply',test_data',settings);
train_data(isnan(train_data))=0;
test_data(find(isnan(test_data)))=0;
train_data=train_data';
test_data=test_data';

opt.max_iter = 100;
k = 20;
%获得标签置信度
[C1, C2] = Get_C_KNN(train_data,train_target,k);

%获得聚类中心及其聚类中心虚拟标签
cluster_num = 90;
[Z, M] = get_kmeans(train_data,train_target,cluster_num,C1);

opt.C1 = C1;
opt.C2 = C2;
opt.Z = Z;
opt.M = M;
opt.lambda1 = 0.1;
opt.lambda2 = 10;
opt.lambda3 = 10;
opt.lambda4 = 10;


model = PML_train(train_data, train_target, true_target, opt,test_data,test_target);
[HammingLoss,RankingLoss,OneError,Coverage,AveragePrecision] = PML_test(test_data,test_target,model);

 
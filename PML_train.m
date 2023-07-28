function model = PML_train(train_data, train_target, true_target, opt,test_data,test_target)
% rng('default');
% rng(0);
warning('off');

[num_train,dim]=size(train_data);
[~,num_label]=size(train_target);
lambda1 = opt.lambda1;
lambda2 = opt.lambda2;
lambda3 = opt.lambda3;
lambda4 = opt.lambda4;
C1 = opt.C1;
C2 = opt.C2;
Z = opt.Z;
M = opt.M;
max_iter = opt.max_iter;
[num_cluster,~]=size(Z);
%% Training

X = [train_data, ones(num_train,1)];
Z = [Z, ones(num_cluster,1)];
Y =  train_target;
L = relate3(C1.*train_target);
T = M;
W =zeros(dim+1,num_label);
deta =  1E-5;
grad = -X'*(C2.*Y  - C2.*(X*W))- lambda1*(Z'*T -Z'*Z*W) + lambda3*W +lambda4*W*L;
W_prev = W;
grad_prev = grad;
grad_norm = inf;
tol = 1e-6;
t=0;
while t < max_iter && grad_norm > tol
    
    
    
    
    % Update W
    grad = -X'*(C2.*Y  - C2.*(X*W))- lambda1*(Z'*T -Z'*Z*W) + lambda3*W +lambda4*W*L;
    if norm(grad, 'fro') < tol
            break;
    end
   if t > 1
            s = W - W_prev;
            y = grad - grad_prev;
            deta = abs(trace(s' * y)) / trace(y' * y);
   end
   
   W_prev = W;
   grad_prev = grad;
   W = W - deta * grad;
   grad_norm = norm(grad, 'fro');
   t=t+1;

    % Update T
       T = (1/(lambda1 + lambda2))*(lambda1*Z*W + lambda2*M); 
end

%% Computing the size predictor using linear least squares modelO
Outputs = X*W;

Left=Outputs;

Right=zeros(num_train,1);
for i=1:num_train
    temp=Left(i,:);
    [temp,index]=sort(temp);
   
    candidate=zeros(1,num_label+1);
 
    candidate(1,1)=temp(1)-0.1;
    for j=1:num_label-1
        candidate(1,j+1)=(temp(j)+temp(j+1))/2;
    end
    candidate(1,num_label+1)=temp(num_label)+0.1;
    miss_class=zeros(1,num_label+1);
    for j=1:num_label+1
        temp_notlabels=index(1:j-1);
        
        temp_labels=index(j:num_label);
        [~,false_neg]=size(setdiff(temp_notlabels,find(true_target(i,:)==0)));
        [~,false_pos]=size(setdiff(temp_labels,find(true_target(i,:)==1)));
        miss_class(1,j)=false_neg+false_pos;
    end
    [~,temp_index]=min(miss_class);
    Right(i,1)=candidate(1,temp_index);
end
 
Left=[Left,ones(num_train,1)];

tempvalue=(Left\Right)';
Weights_sizepre=tempvalue(1:num_label);
Bias_sizepre=tempvalue(num_label+1);


model.W = W;
model.Weights_sizepre=Weights_sizepre;
model.Bias_sizepre=Bias_sizepre;  

end




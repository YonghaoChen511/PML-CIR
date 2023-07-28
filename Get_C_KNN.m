function [C1, C2] = Get_C_KNN(train_data,train_target,k)

[num_train, num_label]=size(train_target);
distance = EuDist2(train_data,train_data,1);
[near_sample , ind] = sort(distance,2);
ind = ind(:,2:k+1);
C1 = zeros(num_train, num_label);
C2 = zeros(num_train, num_label);
for i = 1:num_train
    for j = 1: num_label
        if train_target(i, j) == 1
            C1(i, j) = sum(train_target(ind(i, :), j)==1)/k;
        end
        if train_target(i, j) == 0
            C2(i, j) = 1;
        end
    end
end
C1 = (C1.^2)./(C1.^2 + (1 - C1).^2)+1;

C2 = C2 + C1;

end
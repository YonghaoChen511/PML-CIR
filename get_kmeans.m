function [Z, M] = get_kmeans(train_data,train_target,cluster_num,C)
[idx,Z] = kmeans(train_data,cluster_num);
mean_target = C.*train_target;

for i =1:cluster_num
    
    cluster_idx = (idx == i);
    M(i, :) = sum(mean_target(cluster_idx,:),1)/sum(cluster_idx==1);
    if sum(cluster_idx == 1) < 5
        Z(i, :) = 0;
        M(i, :) = 0; 
    end
end


end
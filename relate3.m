function L = relate3(Y)
[~, class] = size(Y);
P = zeros(class, class);
for i=1:class
    for j=1:class
    
       if sum(Y(:,i))==0
           P(i,j) = 0;
       else    
       P(i,j) = (Y(:,i)'* Y(:,j))/sum(Y(:,i));   
       end 
    end 
end
SUM = sum(P,2);
D = diag(SUM);

L = D - P;

end



%%
figure
nbr = 4;
% load(strcat('eval-data_cnn_0', char(string(nbr)), '.mat'));
plot([1:size(val_loss, 2)], val_loss)
min(loss)
min(val_loss)



%%
for i = 1:6
   load(strcat('eval-data_cnn_240_0', char(string(i)), '.mat'));
   i
   min(val_loss)
end
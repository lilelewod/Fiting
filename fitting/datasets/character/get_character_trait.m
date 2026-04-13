% 本matlab程序最好在bpl根目录运行,并确保model_fits.zip文件已解压至bpl根目录 
% https://github.com/brendenlake/BPL
% http://cims.nyu.edu/~brenden/supplemental/BPL_precomputed/model_fits.zip

clc;clear;close all;
restoredefaultpath;
addpath(genpath(pwd));

run_id = 4;
test_id = 1;
model_id = 1;

load_file_name = sprintf('model_fits/run%d_test%d_G.mat', run_id, test_id);
load(load_file_name);
imshow(G.img)
training_sample_ordinal=3;
states = cell(10,1);

model = G.models{model_id};
num_strokes = model.ns;
strokes = model.S;
invscales_token=cell(num_strokes+1, 1);
shapes_token=cell(num_strokes+1, 1);
% motor=cell(num_strokes+1, 1);
relation = cell(num_strokes+1, 1);
positions_token=cell(num_strokes+1, 1);
for j=1:num_strokes
    invscales_token{j} = strokes{j}.invscales_token;
    shapes_token{j} = strokes{j}.shapes_token;
    positions_token{j} = strokes{j}.pos_token;    
    % motor{j} = strokes{j}.motor;
    if isa(strokes{j}.R, 'RelationIndependent')
        relation{j} = 'independent';
    elseif isa(strokes{j}.R, 'RelationAttachAlong')
        relation{j} = {strokes{j}.R.type, strokes{j}.R.attach_spot, strokes{j}.R.subid_spot, strokes{j}.R.eval_spot_token};      
    elseif isa(strokes{j}.R, 'RelationAttach')
        relation{j} = {strokes{j}.R.type, strokes{j}.R.attach_spot};        
    else  
        assert(false)
    end
end
% 增加额外的一行，以避免只有一个笔画的时候，无法顺利导入Python
invscales_token{num_strokes+1} = 'extra';
shapes_token{num_strokes+1} = 'extra';
positions_token{num_strokes+1} = 'extra';
% motor{num_strokes+1} = 'extra';
relation{num_strokes+1} = 'extra';


save_file_name = sprintf('run%d_test%d_%d.mat', run_id, test_id, model_id);
save(save_file_name, "invscales_token","shapes_token","positions_token","relation");

% for i=1:10
%     model = premodels{i,training_sample_ordinal}.model.models{1};
%     num_strokes = model.ns;
%     strokes = model.S;
%     invscales_token=cell(num_strokes+1, 1);
%     shapes_token=cell(num_strokes+1, 1);
%     motor=cell(num_strokes+1, 1);
%     pos_token=cell(num_strokes+1, 1);
%     for j=1:num_strokes
%         invscales_token{j} = strokes{j}.invscales_token;
%         shapes_token{j} = strokes{j}.shapes_token;
%         motor{j} = strokes{j}.motor;
%         pos_token{j} = strokes{j}.pos_token;
%     end
%     % 增加额外的一行，以避免只有一个笔画的时候，无法顺利导入Python
%     invscales_token{num_strokes+1} = 'extra';
%     shapes_token{num_strokes+1} = 'extra';
%     motor{num_strokes+1} = 'extra';
%     pos_token{num_strokes+1} = 'extra';
%     % states{i} = character_state;
%     file_name=sprintf("state_of_%d_of_training_sample_%d.mat",i-1, training_sample_ordinal);
%     save(file_name, "invscales_token","shapes_token","pos_token","motor");
% end
% % file_name=sprintf("states_of_training_sample_%d.mat",training_sample_ordinal);
% % save(file_name, "states");

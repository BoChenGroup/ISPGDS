clear all; clc; close all;
addpath(genpath('.'));
datasetname = {'dblp','nips','sotu','gdelt2001','gdelt2002','gdelt2003','gdelt2004'...
    ,'gdelt2005','icews200123','icews200456','icews200789','icews2003'};
for trial =1:5
for data_each = 1:11
    dataname  = cell2mat(datasetname(data_each));
    filepath = 'F:\dynamic_datasets\thematlab\';
    filename = strcat(filepath , dataname);
    load (filename); 
                

WCmatrix=Y_TV';

[nV, nT] = size(WCmatrix);
minl = 3; ming = 10; 

if 0
minl = 5; ming = 20; 
ind1 = find( sum (WCmatrix,2 ) < ming)';
[nV, nT] = size(WCmatrix);

ind2 = [];
for v = 1:nV
    if max ( WCmatrix(v,:) ) < minl  
        ind2 = [ind2, v];
    end
end
WCmatrix = WCmatrix(setdiff(1:nV,[ind1,ind2]), :); 
end

TestData = WCmatrix(:,nT); tmpTrainData =  WCmatrix(:,1:end-1); [nV ,nT_tr] = size(tmpTrainData);
% hold out 20% for testing % index = randperm(N);
hd = 0.2;  t_total = sum(tmpTrainData,1)'; nhd = ceil(hd*t_total);
TrainData = zeros(nV, nT_tr); HdoutData = zeros(nV, nT_tr);
for t = 1:nT_tr
    doc = [];
    for v = 1:nV
        if tmpTrainData(v,t)~= 0
            doc = [doc, v*ones(1,tmpTrainData(v,t))];
        end
    end
    hd_index = randperm(t_total(t));
    hd_doc = doc( hd_index(1:nhd(t) )); tr_doc = doc( hd_index(nhd(t)+1: end )); 
    
    for v = 1:nV 
        HdoutData(v, t) =  length(find(hd_doc == v));
        TrainData(v, t) =  length(find(tr_doc == v));
    end
end

    name_save = ['topM_','trial_',num2str(trial),'_',dataname,'.mat'];
    save(['./TopM_data/',name_save],'TestData','HdoutData','TrainData')
end
end



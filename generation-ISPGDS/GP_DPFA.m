clc
clear
close all
%%
Setting.Burnin     =   1500;
Setting.Collection =   0;
Setting.K          =   200;
Setting.Stationary =   1;
Setting.NSample    =   1;
Setting.step = 1;
%%  评价指标
evalute = 2 ;
evalute_name_all = {'MSE_RMSE','Top@M','S_F_mask'};
evalute_name  = cell2mat(evalute_name_all(evalute));
%%  数据选取
datasetname = {'dblp','nips','sotu','gdelt2001','gdelt2002','gdelt2003','gdelt2004'...
    ,'gdelt2005','icews200123','icews200456','icews200789','icews2003'};
for trial =1:5
for data_each = 4:4
logpv_test=[]; 
recErr_test = [];
predErr_test = [];
PR=[]; 
RC=[]; 
PP=[];



    dataname  = cell2mat(datasetname(data_each));
    filepath = '.\TopM_data\topM_trial_';
    filename = strcat(filepath ,num2str(trial),'_', dataname);
    load (filename);               
    %% 背景参数初始化
    
    
    switch evalute_name
        case 'S_F_mask'
            [TT,V] = size(Y_TV);
            predict_dex = [TT];
            T = TT-length(predict_dex) ;
            M = ones(T,V).*mask1(1:T,1:V);
            
            X_train_real = (double(Y_TV(1:T,:)))';
            X_predict_real = (double(Y_TV(predict_dex,:)))';
            
            missing_dex = find(mask1(1:T,1)==1);
            X_missing_real = (double(Y_TV(missing_dex,:)))';
            %% 初始化missing count矩阵
            X_missing = zeros(size(X_train_real));
            for dex_num = 1:length(missing_dex)
                dex =   missing_dex(dex_num) ;
                X_missing(:,dex)  =  round(( Y_TV(dex+1,:)'+Y_TV(dex+1,:)')/2);
            end
            X_train   = X_train_real.*(1-M)'+ X_missing.*M';
            X_predict_sample = zeros(Setting.Collection,V,length(predict_dex));
            X_missing_sample = zeros(Setting.Collection,V,length(missing_dex));
            
        case   'MSE_RMSE'
            
            X_train = (double(Y_TV(1:end-1,:)))';
            X_test = (double(Y_TV(end,:)))';
            [V,T] = size(X_train);
            
            pred_err_sample = zeros(Setting.Collection+Setting.Burnin,1);
            pred_err_mean = zeros(Setting.Collection+Setting.Burnin,1);
            rec_err_sample = zeros(Setting.Collection+Setting.Burnin,1);
            rec_err_mean  = zeros(Setting.Collection+Setting.Burnin,1);
        case  'Top@M'   
            X_train = double(TrainData);
            X_test = double(TestData);
            [V,T] = size(X_train); 
            X_held = double(HdoutData);
    end
    %% 超参数
    Supara.e0 = 1;
    Supara.f0 = 1;
    Supara.eta0 = 0.1;
    Supara.gamma0 = 50;
    %% 参数初始化
    
    Para.Phi           = ones(V,Setting.K);
    Para.Phi           = bsxfun(@times,Para.Phi,1./sum(Para.Phi,1));
    Para.lamda         = randg(1,1,Setting.K);

    Para.c = 1;
    Para.Theta         = randg(1,Setting.K,T);
    Theta_N            = 0.01 * ones(Setting.K,1);
    Para.ct            = randg(1,1,T);
    ct0 = 1;
    Para.L_kdott = zeros(Setting.K,T+1);  
    
    %% 更新
    for i=1:Setting.Collection+Setting.Burnin
        %% sample x_augmentation (Multinomial)  对
        Sig = diag (Para.lamda);
        [Y_KT,Y_VK] = Multrnd_Matrix_mex_fast_v1(sparse(X_train),Para.Phi,Sig*Para.Theta);
        %% sample Phi  对
        Para.Phi = SamplePhi(Y_VK,Supara.eta0);
        if nnz(isnan(Para.Phi))
            warning('Phi Nan');
            Para.Phi(isnan(Para.Phi)) = 0;
        end
        %% sample Lamda  对
         
        shape = sum(Y_KT,2) + Supara.gamma0/Setting.K;
            
        scale = Para.c + sum(Para.Theta,2);
        Para.lamda = gamrnd(shape,1./scale)';
            
        if nnz(isnan(Para.lamda))
            warning('lamda Nan');
            Para.lamda(isnan(Para.lamda)) = 0;
        end    
        
       %% sample L (CRT)  对
        for t=T:-1:2
            Para.L_kdott(:,t) = CRT_sum_mex_matrix_v1(sparse((Y_KT(:,t)+Para.L_kdott(:,t+1))'),Para.Theta(:,t-1)')';        
        end        
        %% sample Theta  对
        Theta0 = randg(Theta_N)./(ct0+Para.lamda');
        for t=1:T
            if t==1
                shape = Y_KT(:,t)+Para.L_kdott(:,t+1) + Theta0;
            else
                shape = Y_KT(:,t) + Para.L_kdott(:,t+1) + Para.Theta(:,t-1);
            end
            scale = Para.ct(1,t) + Para.lamda';
            Para.Theta(:,t) = randg(shape)./scale;
        end
        
        if nnz(isnan(Para.Theta))
            warning('Theta Nan');
        end  
        

        %%  sample ct gamma
        ct0 = randg(Supara.e0 + Theta_N)/(Supara.f0 + sum(Theta0,1) );
        for t=1:T
            if t==1
                shape = Supara.e0 + sum(Theta0,1);                
            else
                shape = Supara.e0 + sum(Para.Theta(:,t-1),1);
            end
            scale = Supara.f0 + sum(Para.Theta(:,t),1);
            Para.ct(1,t) = randg(shape)/scale;
        end
        %%  sample c gamma  
        shape = Supara.e0 + Supara.gamma0;           
        scale = Supara.f0 + sum(Para.lamda);
        Para.c = randg(shape)/scale;      
        %%  sample gamma0  gamma  
        LK = CRT_sum_mex_matrix_v1(sparse(squeeze(sum(Y_KT,2))'),Supara.gamma0/Setting.K)';        
        shape = Supara.e0 + sum(LK); 
        sum_Theta = sum(Para.Theta,2);
        log_sum_Theta = log(1-sum_Theta./(Para.c+sum_Theta)) ;
        scale = Supara.f0 - sum(1/Setting.K*log_sum_Theta) ;
        Supara.gamma0 = randg(shape)/scale;        
%%   
        Lambda = Para.Phi * (Sig* Para.Theta);
        like   = sum(sum( X_train .* log(Lambda)-Lambda));
        Likelihood(i) = like;           
 %% prediction of heldout
        
     if mod(i,5)==0
 

        Thetapred = randg(Para.Theta(:,end))/Para.ct(end);
        X_pred =  Para.Phi*(Sig*Thetapred);


        [~,cpred] = sort(X_pred, 'descend' );
        [~,ctrue] = sort(X_test, 'descend' );
        range = 50; c = intersect(cpred(1:range),ctrue(1:range)) ;
        pp = length(c)/range;
        %% normailization reconstruction
        ERR1   = X_train(:,2:end)-(Lambda(:,2:end));
        rec_err =  mean(mean((ERR1 ).^2));
        %% sample
        X_held_hat = Lambda;
        [~,cpred] = sort(X_held_hat, 1,'descend' );
        [~,ctrue] = sort(X_held, 1, 'descend' );
        for t = 1:T
            g1 = zeros(V,1); g2 = zeros(V,1);
            g1(ctrue(1:range, t),1) = 1;
            g2(cpred(1:range, t),1) = 1;
            con = confusionmat(g1,g2);
            TNR(t) = con(2,2)/(con(2,2)+con(1,2));
            TT = find( X_held(:,t) ~= 0  );
            C = intersect(TT, cpred(1:range, t) );
            % C = setdiff(cpred(1:range, t), intersect(T, cpred(1:range, t) ));
            TPR(t) = length(C)/length(TT);
        end
        pr = mean(TNR); rc =  mean(TPR);



        recErr_test = [recErr_test;rec_err];
         PR=[PR; pr]; 
         RC=[RC; rc];
         PP=[PP; pp];
         
         fprintf('Test, Iter %d:  recErr=%4.4f, pr=%4.4f, rc=%4.4f, pp=%4.4f \n',...
             i,recErr_test(end),...
             PR(end), RC(end), PP(end));

           end
    
           
    end
    
    
    name_save = [dataname,'_',evalute_name,'_K',num2str(Setting.K),'_S',num2str(Setting.Stationary),'.mat'];
    save(['TopM_result/',name_save])
    
    
    
    
end
end
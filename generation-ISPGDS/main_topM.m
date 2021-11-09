clc
clear
close all
%%
Setting.Burnin     =   1500;
Setting.Collection =   0;
Setting.K          =   100;
Setting.Stationary =   1;
Setting.NSample    =   1;
Setting.step = 1;
%%  评价指标
evalute = 0 ;
evalute_name_all = {'MSE_RMSE','Top@M','S_F_mask'};
evalute_name  = cell2mat(evalute_name_all(2));
%%


%%  数据选取
datasetname = {'dblp','nips','sotu','gdelt2001','gdelt2002','gdelt2003','gdelt2004'...
    ,'gdelt2005','icews200123','icews200456','icews200789','icews2003'};

for data_each = 9:9
logpv_test=[]; 
recErr_test = [];
predErr_test = [];
PR=[]; 
RC=[]; 
PP=[];



    dataname  = cell2mat(datasetname(data_each));
    switch dataname
        case 'icews200123'
            load ('E:\experiments\DPGDS\TopM_data\topM_icews200123');
        case 'icews200456'
            load ('E:\experiments\DPGDS\TopM_data\topM_icews200456');
        case 'icews200789'
            load ('E:\experiments\DPGDS\TopM_data\topM_icews200789');
        case 'nips'
            load ('E:\experiments\DPGDS\TopM_data\topM_nips');
        case 'sotu'
            load ('E:\experiments\DPGDS\TopM_data\topM_sotu');
        case 'dblp'
            load ('E:\experiments\DPGDS\TopM_data\topM_dblp');
        case 'gdelt2001'
            load ('E:\experiments\DPGDS\TopM_data\topM_gdelt2001');
        case 'gdelt2002'
            load ('E:\experiments\DPGDS\TopM_data\topM_gdelt2002');
        case 'gdelt2003'
            load ('E:\experiments\DPGDS\TopM_data\topM_gdelt2003');
        case 'gdelt2004'
            load ('E:\experiments\DPGDS\TopM_data\topM_gdelt2004');
        case 'gdelt2005'
            load ('E:\experiments\DPGDS\TopM_data\topM_gdelt2005');
        case 'Toydata'
        case 'icews2003'
            load ('E:\experiments\DPGDS\icews1');
    end
    
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
    Supara.tao0 = 1;
    Supara.gamma0 = 100;
    Supara.eta0 = 0.1;
    Supara.epilson0 = 0.1;
    %% 参数初始化
    
    Para.Phi           = ones(V,Setting.K);
    Para.Phi           = bsxfun(@times,Para.Phi,1./sum(Para.Phi,1));
    
    Para.Pi            = ones(Setting.K,Setting.K);
    Para.Pi            = bsxfun(@times,Para.Pi,1./sum(Para.Pi,1));
    
    Para.Theta         = randg(1,Setting.K,T);
    Para.Xi = 1;
    Para.V = ones(Setting.K,1);
    Para.beta = 1;
    Para.L_kdott = zeros(Setting.K,T+1);
    Para.L_dotkt = zeros(Setting.K,T+1);
    Para.delta=ones(T,1);
    Para.Zeta = zeros(T+1,1);
    
    %% 更新
    for i=1:Setting.Collection+Setting.Burnin
        
        L_KK = zeros(Setting.K,Setting.K);
        h = zeros(Setting.K,Setting.K);
        n = zeros(Setting.K,1);
        rou = zeros(Setting.K,1);
        %% sample x_augmentation (Multinomial)  对
        [Y_KT,Y_VK] = Multrnd_Matrix_mex_fast_v1(sparse(X_train),Para.Phi,Para.Theta);
        
        %% sample Phi  对
        Para.Phi = SamplePhi(Y_VK,Supara.eta0);
        if nnz(isnan(Para.Phi))
            warning('Phi Nan');
            Para.Phi(isnan(Para.Phi)) = 0;
        end
        
        %% sample L (CRT)  对
        if Setting.Stationary == 1
            Para.Zeta= -lambertw(-1,-exp(-1-Para.delta./Supara.tao0))-1-Para.delta./Supara.tao0;
            Para.Zeta(T+1) = Para.Zeta(1);
            Para.L_dotkt(:,T+1) =poissrnd(Para.Zeta(1)*Supara.tao0*Para.Theta(:,T));
        end
        for t=T:-1:2
            Para.L_kdott(:,t) = CRT_sum_mex_matrix_v1(sparse((Y_KT(:,t)+Para.L_dotkt(:,t+1))'),(Supara.tao0*Para.Pi*Para.Theta(:,t-1))')';
            [Para.L_dotkt(:,t),tmp] = Multrnd_Matrix_mex_fast_v1(sparse(Para.L_kdott(:,t)),Para.Pi,Para.Theta(:,t-1));
            L_KK = L_KK+tmp;
        end
        
        
        
        %% calsulate Zeta  对
        if Setting.Stationary == 0
            for t=T:-1:1
                %Setting.Zeta(t) = log(1+Setting.delta(t)/Supara.tao0+Setting.Zeta(t+1));
                Para.Zeta(t) = Supara.tao0*log(1+(Para.delta(t)+Para.Zeta(t+1))/Supara.tao0);
            end
        end
        
        if nnz(isnan(Para.Zeta))
            warning('Zeta Nan');
        end
        
        
        %% sample Theta  对
        for t=1:T
            if t==1
                shape = Y_KT(:,t)+Para.L_dotkt(:,t+1)+Supara.tao0*Para.V;
            else
                shape = Y_KT(:,t)+Para.L_dotkt(:,t+1)+Supara.tao0*Para.Pi*Para.Theta(:,t-1);
            end
            scale = Supara.tao0+Para.delta(t)+Para.Zeta(t+1);
            Para.Theta(:,t) = gamrnd(shape,1./scale);
        end
        
        if nnz(isnan(Para.Theta))
            warning('Theta Nan');
        end
        
        %% sample Pi  对
        Piprior = Para.V*Para.V';
        Piprior(logical(eye(size(Piprior))))=0;
        Piprior = Piprior+diag(Para.Xi*Para.V);
        Para.Pi = SamplePi(L_KK,Piprior);
        if nnz(isnan(Para.Pi))
            warning('Pi Nan');
            Para.Pi(isnan(Para.Pi)) = 0;
        end
        
        %% sample Delta   对
        [ Para.delta ] =Sample_delta(X_train,Para.Theta,Supara.epilson0,Setting.Stationary);
        
        
        
        if nnz(isnan(Para.delta))
            warning('delta Nan');
        end
        
        %% sample Beta    对
        shape = Supara.epilson0+Supara.gamma0;
        scale = Supara.epilson0+sum(Para.V);
        Para.beta = gamrnd(shape,1./scale);
        
        %% sample q  对
        a = sum(Para.L_dotkt,2);
        b = Para.V.*(Para.Xi+repmat(sum(Para.V),Setting.K,1)-Para.V);
        %q = betarnd(a,b);
        q = betarnd(b,a);
        q=max(q,realmin);
        for k=1:Setting.K
            if q(k)==0
                haha=1;
            end
        end
        
        %% sample h   对
        for k1 = 1:Setting.K
            for k2 = 1:Setting.K
                h(k1,k2) = CRT_sum_mex_matrix_v1(sparse(L_KK(k1,k2)),Piprior(k1,k2));
            end
        end
        
        %% sample Xi  对
        shape = Supara.gamma0/Setting.K + trace(h);
        %shape = Supara.epilson0 + trace(h);
        scale = Para.beta - Para.V'*log(q);
        %scale = Supara.epilson0 - Setting.V'*log(1-q);
        Para.Xi = gamrnd(shape,1./scale);
        
        %% sample V
        for k=1:Setting.K
            Para.L_kdott(k,1) = CRT_sum_mex_matrix_v1(sparse(Y_KT(k,1)+Para.L_dotkt(k,2)),Supara.tao0*Para.V(k));
            n(k)=sum(h(k,:)+h(:,k)')-h(k,k)+Para.L_kdott(k,1);
            rou(k) = -log(q(k)) * (Para.Xi+sum(Para.V)-Para.V(k)) - (log(q')*Para.V-log(q(k))*Para.V(k)) + Para.Zeta(1);
        end
        shape = Supara.gamma0/Setting.K+n;
        scale = Para.beta+rou;
        Para.V = gamrnd(shape,1./scale);
        
        if nnz(isnan( Para.V))
            warning('V Nan');
        end
        
        Lambda = bsxfun(@times,Para.delta', Para.Phi*Para.Theta);
        like   = sum(sum( X_train .* log(Lambda)-Lambda));
        Likelihood(i) = like;
        

   
 %% prediction of heldout
        
%      if mod(i,Setting.step)==0
%  
%         [rec_err,pred_err, pp,pr, rc] = pgds_calc_pred_error(X_held,X_test,X_train,Para,Supara,Y_KT);
% 
%         recErr_test = [recErr_test;rec_err];
%         totaltime = toc;
%          PR=[PR; pr]; 
%          RC=[RC; rc];
%          PP=[PP; pp];
%          
%          fprintf('Test, Iter %d:  recErr=%4.4f, pr=%4.4f, rc=%4.4f, pp=%4.4f \n',...
%              i,recErr_test(end),...
%              PR(end), RC(end), PP(end));
% 
%            end
    
           
    end
    
    
    name_save = [dataname,'_',evalute_name,'_K',num2str(Setting.K),'_S',num2str(Setting.Stationary),'.mat'];
    save(['TopM_result/',name_save])
    
    
    
    
end

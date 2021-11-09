clc
clear
close all
%%
L = 3;
K =   [200,100,50];
num_trial = 3;
%%
Setting.Burnin     =   300;
Setting.Collection =   200;
Setting.Stationary =   0;
Setting.NSample    =   1;
Setting.step = 10;
%%  评价指标
evalute = 1;
evalute_name_all = {'MSE_PMSE','Top@M','S_F_mask'};
evalute_name  = cell2mat(evalute_name_all(evalute));
%%  数据选取
load D:\GDD\HRRP\MNIST\hrrp_4.mat;
dataname = 'HRRP';
%%

%%

for trial =1:1
        dataname  = cell2mat(datasetname(data_each));
        
        switch evalute_name
            case  'Top@M'
%                 filepath = '.\TopM_data\topM_trial_';
%                 filename = strcat(filepath ,num2str(trial),'_', dataname);
%                 load (filename);
                X_train = double(TrainData);
                X_test = double(TestData);
                [V,T] = size(X_train);
                X_held = double(HdoutData);
                PR=[];
                RC=[];
                PP=[];
                logpv_test=[];
                recErr_test = [];
                predErr_test = [];
                
                Result.recErr_test = [];
                
                Result.MP=[];
                Result.MR=[];
                Result.PP=[];
                Result.pred_err=[];
                
            case   'MSE_PMSE'
                
                %filepath = '.\MSE_PMSE_data\';
                %filename = strcat(filepath,dataname);
                %load (filename);
                
                %X_train = (double(Y_TV(1:end-1,:)))';
                %X_test = (double(Y_TV(end,:)))';
                X_train = double(X_train);
                X_test = double(X_test);
                [V,T] = size(X_train);
                
                pred_err_sample = zeros(Setting.Collection+Setting.Burnin,1);
                pred_err_mean = zeros(Setting.Collection+Setting.Burnin,1);
                rec_err_sample = zeros(Setting.Collection+Setting.Burnin,1);
                rec_err_mean  = zeros(Setting.Collection+Setting.Burnin,1);
                Result.PMSE_mean = [];
                Result.MSE_mean  = [];
                Result.PMSE_sample= [];
                Result.MSE_sample  = [];
        end
        %% 超参数
        Supara.tao0 = 1;
        Supara.gamma0 = 100;
        Supara.eta0 = 0.1;
        Supara.epilson0 = 0.1;
        Supara.c = 1;
        %% 参数初始化
        Para.Phi = cell(L,1);
        Para.Pi  = cell(L,1);
        Para.Xi  = cell(L,1);
        Para.V  = cell(L,1);
        Para.beta  = cell(L,1);
        Para.q  = cell(L,1);
        Para.h = cell(L,1);
        Para.n = cell(L,1);
        Para.rou = cell(L,1);
        
        Piprior = cell(L,1);
        Theta    = cell(L,1);
        delta  = cell(L,1);
        Zeta  = cell(L,1);
        L_dotkt  = cell(L,1);
        L_kdott = cell(L,1);
        A_KT = cell(L,1);
        A_VK = cell(L,1);
        
        L_KK = cell(L,1);
        prob1 = cell(L,1);
        prob2 = cell(L,1);
        Xt_to_t1 = cell(L+1,1);
        X_layer_split1 = cell(L,1);
        X_layer  = cell(L,1);
        
        for l=1:L
            if l==1
                Para.Phi{l} = rand(V,K(l));
                A_VK{l} = zeros(V,K(l));
            else
                Para.Phi{l} = rand(K(l-1),K(l));
                A_VK{l} = zeros(K(l-1),K(l));
            end
            Para.Phi{l} = bsxfun(@rdivide, Para.Phi{l}, max(realmin,sum(Para.Phi{l},1)));
            Para.Pi{l}  = rand(K(l),K(l));
            Para.Pi{l}  = bsxfun(@rdivide, Para.Pi{l}, max(realmin,sum(Para.Pi{l},1)));
            Para.Xi{l} = 1;
            Para.V{l} = ones(K(l),1);
            Para.beta{l} = 1;
            Para.h{l} = zeros(K(l),K(l));
            Para.n{l} = zeros(K(l),1);
            Para.rou{l} = zeros(K(l),1);
            
            Theta{l}    = ones(K(l),T)/K(l);
            delta{l}    = ones(T+1,1);
            Zeta{l}     = zeros(T+1,1);
            L_dotkt{l} = zeros(K(l),T+1);
            A_KT{l} = zeros(K(l),T);
            L_kdott{l} = zeros(K(l),T+1);
            X_layer{l} = zeros(K(l),T,2);
            
        end
        
        
        
        
        for i = 1:Setting.Collection+Setting.Burnin
            tic
            for l=1:L
                
                L_KK{l} = zeros(K(l),K(l));
                
                if l==1
                    [A_KT{l},A_VK{l}] = Multrnd_Matrix_mex_fast_v1(sparse(X_train), Para.Phi{l},Theta{l});
                else
                    [A_KT{l},A_VK{l}] = Multrnd_Matrix_mex_fast_v1(sparse(Xt_to_t1{l}), Para.Phi{l},Theta{l});
                end
                %% sample next layer count
                if l == L
                    
                    for t=T:-1:2
                        L_kdott{l}(:,t) = CRT_sum_mex_matrix_v1(sparse((A_KT{l}(:,t)+ L_dotkt{l}(:,t+1))'),(Supara.tao0 * Para.Pi{l} * Theta{l}(:,t-1))')';
                        [L_dotkt{l}(:,t),tmp] = Multrnd_Matrix_mex_fast_v1(sparse(L_kdott{l}(:,t)), Para.Pi{l},Theta{l}(:,t-1));
                        L_KK{l} = L_KK{l} + tmp;
                    end
                    
                else
                    
                    prob1{l} = Supara.tao0 * Para.Pi{l} *  Theta{l};
                    prob2{l} = Supara.tao0 * Para.Phi{l+1} *  Theta{l+1};
                    X_layer{l} = zeros(K(l),T,2);
                    Xt_to_t1{l+1} = zeros(K(l),T);
                    X_layer_split1{l} = zeros(K(l),T);
                    
                    for t = T : -1 : 2
                        L_kdott{l}(:,t) = CRT_sum_mex_matrix_v1(sparse(A_KT{l}(:,t)'+ L_dotkt{l}(:,t+1)'),Supara.tao0*( Para.Phi{l+1}*Theta{l+1}(:,t)+ Para.Pi{l} * Theta{l}(:,t-1))')';
                        %% split layer2 count
                        [~,X_layer{l}(:,t,:)] = Multrnd_Matrix_mex_fast_v2(L_kdott{l}(:,t),[prob1{l}(:,t-1) prob2{l}(:,t)],ones(2,1));
                        X_layer_split1{l}(:,t) = squeeze(X_layer{l}(:,t,1));   %pi1*Theta1
                        Xt_to_t1{l+1}(:,t) = squeeze(X_layer{l}(:,t,2));   %phi2*Theta2
                        %% sample split1 augmentation
                        [L_dotkt{l}(:,t),tmp] = Multrnd_Matrix_mex_fast_v1(sparse(X_layer_split1{l}(:,t)), Para.Pi{l},Theta{l}(:,t-1));
                        L_KK{l} = L_KK{l} + tmp ;
                    end
                    
                    L_kdott{l}(:,1) = CRT_sum_mex_matrix_v1(sparse(A_KT{l}(:,1)'+ L_dotkt{l}(:,2)'),Supara.tao0*( Para.Phi{l+1}*Theta{l+1}(:,1))')';
                    Xt_to_t1{l+1}(:,1) = L_kdott{l}(:,1);
                    
                end
                
                
                
                %% sample Phi  对
                Para.Phi{l} = SamplePhi(A_VK{l},Supara.eta0);
                if nnz(isnan(Para.Phi{l}))
                    warning('Phi Nan');
                    Para.Phi{l}(isnan(Para.Phi{l})) = 0;
                end
                
                %% sample Pi 对
                Piprior{l} = Para.V{l}*Para.V{l}';
                Piprior{l}(logical(eye(size(Piprior{l}))))=0;
                Piprior{l} = Piprior{l}+diag(Para.Xi{l}*Para.V{l});
                Para.Pi{l} = SamplePi(L_KK{l},Piprior{l});
                if nnz(isnan(Para.Pi{l}))
                    warning('Pi Nan');
                    Para.Pi{l}(isnan(Para.Pi{l})) = 0;
                end
                
            end
            %% calculate Zeta  对
            if Setting.Stationary == 1
                for l=1:L
                    if l==1
                        Zeta{l}= -lambertw(-1,-exp(-1-delta{l}./Supara.tao0))-1-delta{l}./Supara.tao0;
                    else
                        Zeta{l} = -lambertw(-1,-exp(-1-Zeta{l-1}))-1-Zeta{l-1};
                    end
                    Zeta{l}(T+1) = Zeta{l}(1);
                    L_dotkt{l}(:,T+1) = poissrnd(Zeta{l}(1)*Supara.tao0 * Theta{l}(:,T));
                end
            end
            
            
            if Setting.Stationary == 0
                for l=1:L
                    if l==1
                        for t=T:-1:1
                            Zeta{l}(t) = log(1 + delta{l}(t)/Supara.tao0 + Zeta{l}(t+1));
                        end
                    else
                        for t=T:-1:1
                            Zeta{l}(t) = Supara.tao0*log(1+Zeta{l-1}(t)+Zeta{l}(t+1));
                        end
                        
                    end
                    
                end
            end
            %% sample Theta  对
            for l=L:-1:1
                
                if l==L
                    for t=1:T
                        if t==1
                            shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0 * Para.V{l};
                        else
                            shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0*(Para.Pi{l}* Theta{l}(:,t-1));
                        end
                        scale = Supara.tao0 + delta{l}(t)+ Zeta{l}(t+1);
                        Theta{l}(:,t) = randg(shape)/scale;
                    end
                    
                else
                    for t=1:T
                        if t==1
                            shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0 * Para.Phi{l+1}* Theta{l+1}(:,t);
                        else
                            shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0 * (Para.Phi{l+1} * Theta{l+1}(:,t)+ Para.Pi{l}* Theta{l}(:,t-1));
                        end
                        scale = ( delta{l}(t) + Supara.tao0 + Supara.tao0 * Zeta{l}(t+1))';
                        Theta{l}(:,t) = randg(shape)/scale;
                    end
                    
                    
                end
                
                if nnz(isnan(Theta{l}))
                    warning('Theta Nan');
                end
                
            end
            
            
            
            %% sample Beta    对
            for l = 1:L
                shape = Supara.epilson0 + Supara.gamma0;
                scale = Supara.epilson0 + sum(Para.V{l});
                Para.beta{l} = randg(shape)./scale;
                %% sample q  对
                a  = sum(L_dotkt{l},2);
                b  = Para.V{l}.*(Para.Xi{l}+repmat(sum(Para.V{l}),K(l),1)-Para.V{l});
                Para.q{l} = betarnd(b,a);
                Para.q{l} = max(Para.q{l},realmin);
                %% sample h   对
                for k1 = 1:K(l)
                    for k2 = 1:K(l)
                        Para.h{l}(k1,k2) = CRT_sum_mex_matrix_v1(sparse(L_KK{l}(k1,k2)),Piprior{l}(k1,k2));
                    end
                end
                %% sample Xi  对
                shape = Supara.gamma0/K(l) + trace(Para.h{l});
                scale = Para.beta{l} - Para.V{l}'*log(Para.q{l});
                Para.Xi{l} = randg(shape)./scale;
            end
            %% sample Delta 1  对
            [ delta{1} ] =Sample_delta(X_train,Theta{1},Supara.epilson0,Setting.Stationary);
            
            if nnz(isnan(delta{1}))
                warning('delta Nan');
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% sample V{L}
            for k=1:K(L)
                L_kdott{L}(k,1) = CRT_sum_mex_matrix_v1(sparse(A_KT{L}(k,1)+ L_dotkt{L}(k,2)),Supara.tao0*Para.V{L}(k));
                Para.n{L}(k)=sum(Para.h{L}(k,:)+Para.h{L}(:,k)')-Para.h{L}(k,k) + L_kdott{L}(k,1);
                Para.rou{L}(k) = -log(Para.q{L}(k)) * (Para.Xi{L}+sum(Para.V{L})-Para.V{L}(k)) - (log(Para.q{L}')*Para.V{L}-log(Para.q{L}(k))*Para.V{L}(k)) + Zeta{L}(1);
            end
            shape = Supara.gamma0/K(L) + Para.n{L};
            scale = Para.beta{L} + Para.rou{L};
            Para.V{L} = gamrnd(shape,1./scale);
            
            %% sample V{1},...,V{L]
            if L>1
                for l=1:L-1
                    for k=1:K(l)
                        Para.n{l}(k)=sum(Para.h{l}(k,:) + Para.h{l}(:,k)')-Para.h{l}(k,k);
                        Para.rou{l}(k) = -log(Para.q{l}(k)) * (Para.Xi{l}+sum(Para.V{l})-Para.V{l}(k)) - (log(Para.q{l}')*Para.V{l}-log(Para.q{l}(k))*Para.V{l}(k));% + Para.Zeta2(1);
                    end
                    shape = Supara.gamma0/K(l) + Para.n{l};
                    scale = Para.beta{l} + Para.rou{l};
                    Para.V{l} = gamrnd(shape,1./scale);
                    
                    if nnz(isnan( Para.V{l}))
                        warning('V Nan');
                    end
                end
            end
            %%
            Lambda = bsxfun(@times,delta{1}', Para.Phi{1}*Theta{1});
            like   = sum(sum( X_train .* log(Lambda)-Lambda));
            Likelihood(i) = like;
            
            %% prediction of heldout
            switch evalute_name
                
                case   'MSE_PMSE'
                    Lambda =bsxfun(@times,delta{1}', Para.Phi{1}* Theta{1});
                    ERR1   = X_train(:,2:end)-poissrnd(Lambda(:,2:end));
                    rec_err_mean(i,1) =  mean(mean((ERR1 ).^2));
                    
                    Theta_pred = cell(L,1);
                    for l = L:-1:1
                        if l==L
                            Theta_pred{l} = Para.Pi{l}*Theta{l}(:,end);
                        else
                            Theta_pred{l} = Para.Phi{l+1}*Theta_pred{l+1}+Para.Pi{l}*Theta{l}(:,end);
                        end
                    end
                    X_pred = bsxfun(@times,delta{1}(end)', Para.Phi{1} * Theta_pred{1});
                    pred_err_mean(i,1)  = mean((X_test-X_pred).^2);
                    
                    % Result.MSE   =[Result.MSE ;  rec_err_mean(i,1)];
                    % Result.PMSE  =[Result.PMSE; pred_err_mean(i,1)];
                    
                    if mod(i,Setting.step)==0
                        time_iter = toc;
                        fprintf('dataname:%s,iter: %d, predict1_error: %d, reco_error: %d, like: %d  \n',dataname,i,pred_err_mean(i,1),rec_err_mean(i,1),like);
                        fprintf('trial %d,time_iter %f \n',trial,time_iter);
                        fprintf('.................................................................................... \n')
                        name_save = [dataname,'_','_trial',num2str(trial),evalute_name,'_Layer',num2str(L),'_S',num2str(Setting.Stationary),'.mat'];
                        save(['./TopM_result_new/',name_save])
                    end
                    
                    
                    
                    
                    
                case   'Top@M'
                    %% normailization reconstruction
                    
                    ERR1   = X_train(:,2:end)-poissrnd(Lambda(:,2:end));
                    rec_err =  mean(mean((ERR1 ).^2));
                    %% prediction step
                    pred_err =  0;
                    Theta_pred = cell(L,1);
                    for l = L:-1:1
                        if l==L
                            Theta_pred{l} = Para.Pi{l}*Theta{l}(:,end);
                        else
                            Theta_pred{l} = Para.Phi{l+1}*Theta_pred{l+1}+Para.Pi{l}*Theta{l}(:,end);
                        end
                    end
                    
                    X_pred = bsxfun(@times,delta{1}(end)', Para.Phi{1} * Theta_pred{1});
                    pred_err = mean((X_test-X_pred).^2);
                    
                    [~,cpred] = sort(X_pred, 'descend' );
                    [~,ctrue] = sort(X_test, 'descend' );
                    %% Predictive recall (PP)
                    range = 50; c = intersect(cpred(1:range),ctrue(1:range)) ;
                    PP = length(c)/min(range,length(find( X_test ~=0)));
                    %% Held out precision and recall
                    X_held_hat = (bsxfun(@times,delta{1}', Para.Phi{1}*Theta{1}));
                    [~,cpred] = sort(X_held_hat, 1,'descend' );
                    [~,ctrue] = sort(X_held, 1, 'descend' );
                    range = 50;
                    for t = 1:T
                        g1 = zeros(V,1); g2 = zeros(V,1);
                        g1(ctrue(1:range, t),1) = 1;
                        g2(cpred(1:range, t),1) = 1;
                        con = confusionmat(g1,g2);
                        TNR(t) = con(2,2)/(con(2,2)+con(1,2));
                        TT = find( X_held(:,t) ~= 0  );
                        C = intersect(TT, cpred(1:range, t) );
                        TPR(t) = length(C)/length(TT);
                    end
                    MP = mean(TNR); MR =  mean(TPR);
                    result=[rec_err,pred_err,MP,MR,PP];
                    
                    Result.recErr_test = [Result.recErr_test;result(1)];
                    Result.pred_err = [Result.pred_err;result(2)];
                    
                    Result.MP=[Result.MP; result(3)];
                    Result.MR=[Result.MR; result(4)];
                    Result.PP=[Result.PP; result(5)];
                    
                    
                    
                    if mod(i,Setting.step)==0
                        fprintf('dataname:%s,Iter %d:  RecErr=%4.4f, PredErr=%4.4f, MP=%4.4f, MR=%4.4f, PP=%4.4f\n',...
                            dataname,i,result(1),result(2), result(3), result(4),result(5));
                        fprintf('Likelihood=%4.4f \n',Likelihood(i));
                        fprintf('trial %d',trial);
                        fprintf('.................................................................................... \n')
                        name_save = [dataname,'_','_trial',num2str(trial),evalute_name,'_Layer',num2str(L),'_S',num2str(Setting.Stationary),'.mat'];
                        save(['./TopM_result_new/',name_save])
                    end
                    
            end
            
            %             name_save = [dataname,'_','_trial',num2str(trial),evalute_name,'_Layer',num2str(L),'_S',num2str(Setting.Stationary),'.mat'];
            %             save(['./TopM_result_new/',name_save])
        end
        
    end

A=1;
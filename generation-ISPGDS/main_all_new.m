clc
clear
close all
%%
L = 1;
K =   [2,10,5];
%K =   [200];
num_trial = 3;
%%
Setting.Burnin     =   1000;
Setting.Collection =   2000;
Setting.Stationary =   0;
Setting.NSample    =   1;
Setting.step = 10;
C = 2;
C1= 2;
%%  评价指标
evalute = 2;
evalute_name_all = {'MSE_PMSE','Top@M','S_F_mask'};
evalute_name  = cell2mat(evalute_name_all(evalute));
%%  数据选取
datasetname = {'dblp','nips','sotu','gdelt2001','gdelt2002','gdelt2003','gdelt2004'...
    ,'gdelt2005','icews200123','icews200456','icews200789','icews2003'};
for data_each = 2
    for trial =1:1
        dataname  = cell2mat(datasetname(data_each));
        
        switch evalute_name
            case  'Top@M'
% % % % % %                 filepath = '.\TopM_data\topM_trial_';
% % % % % %                 filename = strcat(filepath ,num2str(trial),'_', dataname);
% % % % % %                 load (filename);
% % % %                  load('E:\HRRPdata.mat');
% % % %                  %X_train = double(TrainData);
% % % %                  %X_test = double(TestData);
% % % %                  %X_train = double(round(TrainData));
% % % %                  %X_test = double(round(TestData));
% % % %                  X_train = double(Data(:,1:14));
% % % %                  X_test = double(Data(:,15));
% % % %                  [V,T] = size(X_train);
% % % %                  %X_held = double(HdoutData);
% % % %                  X_held = double(Data);
% % %                 filepath = '.\TopM_data\topM_trial_';
% % %                 filename = strcat(filepath ,num2str(trial),'_', dataname);
% % %                 load (filename);
% % %                 X_train = double(TrainData);
% % %                 X_test = double(TestData);
% % %                 [V,T] = size(X_train);
% % %                 X_held = double(HdoutData);

                load('E:\toydata2.mat');
                 Data = x';
                 %X_train = double(TrainData);
                 %X_test = double(TestData);
                 %X_train = double(round(TrainData));
                 %X_test = double(round(TestData));
                 %X_train =Data(:,2:15)-Data(:,1:14) ;
                 %X_train = double(round(1.2.^X_train));
                 X_train =double(round(600*Data(:,1:99))) ;
                 X_held = double(round(600*Data(:,1:99)));
                 X_test = double(round(600*Data(:,100)));
%                  X_train =double(round(Data(:,1:14))) ;
%                  X_test = double(round(Data(:,15)));
                [V,T] = size(X_train);
                %X_held = double(HdoutData);
                %X_held = double(Data);
                index = kmeans(X_train',C);
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
                load('E:\experiments\DPGDS\MSE_PMSE_data\icews200789_new.mat')
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
        Supara.tao0 = 0.1;
        Supara.gamma0 = 100;
        Supara.eta0 = 0.1;
        Supara.epilson0 = 0.1;
        Supara.c = 1;
        %C = 3;
        Supara.etac = 1/C;
        %% 参数初始化
        Para.Phi = cell(L,1);
        Para.Pi  = cell(L,C);
        Para.Xi  = cell(L,C);
        Para.V  = cell(L,C);
        Para.beta  = cell(L,C);
        Para.q  = cell(L,C);
        Para.h = cell(L,C);
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
        
        L_KK = cell(L,C);
        prob1 = cell(L,1);
        prob2 = cell(L,1);
        Xt_to_t1 = cell(L+1,1);
        X_layer_split1 = cell(L,1);
        X_layer  = cell(L,1);
        Z = cell(L,1);
        pi_C = cell(L,1);
        pi_C_mid = cell(L,1);
        v = 0.5*ones(C,1);
        eta = zeros(C,1);
        pi0_c = cell(L,1);
        ave_result = zeros(1,5);
        z_t = zeros(T,1);
        lamda = 0.5+rand(V,C);
        pi_t = rand(1,C);
        pi_t_mid = zeros(1,C);
        lamda1 = 0.5+rand(K(1),C1);
        
        for l=1:L
            if l==1
                Para.Phi{l} = rand(V,K(l));
                A_VK{l} = zeros(V,K(l));
            else
                Para.Phi{l} = rand(K(l-1),K(l));
                A_VK{l} = zeros(K(l-1),K(l));
            end
            Para.Phi{l} = bsxfun(@rdivide, Para.Phi{l}, max(realmin,sum(Para.Phi{l},1)));
            for c = 1:1:C
                Para.Pi{l,c}  = rand(K(l),K(l));
                Para.Pi{l,c}  = bsxfun(@rdivide, Para.Pi{l,c}, max(realmin,sum(Para.Pi{l,c},1)));
                Para.Xi{l,c} = 1;
                Para.V{l,c} = ones(K(l),1);
                Para.h{l,c} = zeros(K(l),K(l));
                Para.beta{l,c} = 1; 
            end                                               
            Para.n{l} = zeros(K(l),1);
            Para.rou{l} = zeros(K(l),1);            
            Theta{l}    = ones(K(l),T)/K(l);
            delta{l}    = ones(T+1,1);
            Zeta{l}     = zeros(T+1,1);
            L_dotkt{l} = zeros(K(l),T+1);
            A_KT{l} = zeros(K(l),T);
            L_kdott{l} = zeros(K(l),T+1);
            X_layer{l} = zeros(K(l),T,2);
            %Z{l} = round(1+(C-1)*rand(T,1));
            %Z{l} = [0,index']';
            Z{l} = index;
            pi_C{l} = zeros(C,1);
            pi_C_mid{l} = zeros(C,1);
            pi0_c{l} = 1/C*ones(C,1);
        end
        z_t = index;
        %% test cluster
        for iter = 1:1:2000    
        %% sample z_t
            for t = 1:1:T
                u = rand(1,C);
                g = -log(-log(u));
                for c = 1:1:C
                    pi_t_mid(c) = sum(log(lamda(:,c)).*X_train(:,t)-lamda(:,c));
                    %pi_t(c) = exp(sum(log(lamda(:,c)).*X_train(:,t)-lamda(:,c)))*1/C;
                end
                pi_t_mid = pi_t_mid-max(pi_t_mid);
                for c = 1:1:C
                     pi_t(c) = exp(pi_t_mid(c))*1/C;
                end
                pi_t = pi_t/sum(pi_t);
                z_t(t) = find((g + log(pi_t)) == max(g + log(pi_t)));
            end                      
        %% sample pi
            num_pi = zeros(C,1);
            for t = 1:1:T
                num_pi(z_t(t)) = num_pi(z_t(t)) + 1;            
            end
            pi_0 = SamplePhi(num_pi,Supara.eta0);
        %% sample lamda            
            for c = 1:1:C                
                for j = 1:1:V
                    shape = 0;
                    scale = 0;
                    for t = 1:1:T
                        if z_t(t) == c
                            shape = shape + X_train(j,t);
                            scale = scale + 1;
                        end
                    end
                    shape = shape + 1;
                    scale = scale + 0.1;
                    lamda(j,c) = randg(shape)/scale;
                end
            end
        end
                      
        for i = 1:Setting.Collection+Setting.Burnin
            tic
            for l=1:L
                for c = 1:1:C
                    L_KK{l,c} = zeros(K(l),K(l));
                end
                
                if l==1
                    [A_KT{l},A_VK{l}] = Multrnd_Matrix_mex_fast_v1(sparse(X_train), Para.Phi{l},Theta{l});
                else
                    [A_KT{l},A_VK{l}] = Multrnd_Matrix_mex_fast_v1(sparse(Xt_to_t1{l}), Para.Phi{l},Theta{l});
                end
                %% sample next layer count
                if l == L
                    
                    for t=T:-1:2
                        L_kdott{l}(:,t) = CRT_sum_mex_matrix_v1(sparse((A_KT{l}(:,t)+ L_dotkt{l}(:,t+1))'),(Supara.tao0 * Para.Pi{l,Z{l}(t)} * Theta{l}(:,t-1))')';
                        [L_dotkt{l}(:,t),tmp] = Multrnd_Matrix_mex_fast_v1(sparse(L_kdott{l}(:,t)), Para.Pi{l,Z{l}(t)},Theta{l}(:,t-1));
                        L_KK{l,Z{l}(t)} = L_KK{l,Z{l}(t)} + tmp;   %原来pi是全时刻shared的，现在不是了 在计算的时候 这里不能再这样加了
                       % L_DD{l,Z(t)} = L_DD{l,Z(t)} + L_dotkt{l}
                    end
                    
                else
                    
                    prob1{l} = Supara.tao0 * Para.Pi{l,Z{l}(t)} *  Theta{l};
                    prob2{l} = Supara.tao0 * Para.Phi{l+1} *  Theta{l+1};
                    X_layer{l} = zeros(K(l),T,2);
                    Xt_to_t1{l+1} = zeros(K(l),T);
                    X_layer_split1{l} = zeros(K(l),T);
                    
                    for t = T : -1 : 2
                        L_kdott{l}(:,t) = CRT_sum_mex_matrix_v1(sparse(A_KT{l}(:,t)'+ L_dotkt{l}(:,t+1)'),Supara.tao0*( Para.Phi{l+1}*Theta{l+1}(:,t)+ Para.Pi{l,Z{l}(t)} * Theta{l}(:,t-1))')';
                        %% split layer2 count
                        [~,X_layer{l}(:,t,:)] = Multrnd_Matrix_mex_fast_v2(L_kdott{l}(:,t),[prob1{l}(:,t) prob2{l}(:,t-1)],ones(2,1));
                        X_layer_split1{l}(:,t) = squeeze(X_layer{l}(:,t,1));   %pi1*Theta1
                        Xt_to_t1{l+1}(:,t) = squeeze(X_layer{l}(:,t,2));   %phi2*Theta2
                        %% sample split1 augmentation
                        [L_dotkt{l}(:,t),tmp] = Multrnd_Matrix_mex_fast_v1(sparse(X_layer_split1{l}(:,t)), Para.Pi{l,Z{l}(t)},Theta{l}(:,t-1));
                        L_KK{l,Z{l}(t)} = L_KK{l,Z{l}(t)} + tmp ;
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
                for c=1:1:C
                    Piprior{l,c} = Para.V{l,c}*Para.V{l,c}';
                    Piprior{l,c}(logical(eye(size(Piprior{l,c}))))=0;
                    Piprior{l,c} = Piprior{l,c}+diag(Para.Xi{l,c}*Para.V{l,c});                
                    Para.Pi{l,c} = SamplePi(L_KK{l,c},Piprior{l,c});             
                    if nnz(isnan(Para.Pi{l,c}))
                        warning('Pi Nan');
                        Para.Pi{l,c}(isnan(Para.Pi{l,c})) = 0;
                    end
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
                            %shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0 * Para.V{l};
                            shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0 ;
                        else
                            shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0*(Para.Pi{l,Z{l}(t)}* Theta{l}(:,t-1));
                        end
                        scale = Supara.tao0 + delta{l}(t)+ Supara.tao0*Zeta{l}(t+1);
                        Theta{l}(:,t) = randg(shape)/scale;
                    end
                    
                else
                    for t=1:T
                        if t==1
                            shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0 * Para.Phi{l+1}* Theta{l+1}(:,t);
                        else
                            shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0 * (Para.Phi{l+1} * Theta{l+1}(:,t)+ Para.Pi{l,Z{l}(t)}* Theta{l}(:,t-1));
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
            for c = 1:1:C
                for l = 1:L
                    shape = Supara.epilson0 + Supara.gamma0;
                    scale = Supara.epilson0 + sum(Para.V{l,c});
                    Para.beta{l,c} = randg(shape)./scale;
                %% sample q  对
                   % a  = sum(L_dotkt{l},2);
                    mid = zeros(K(l),1);
                    for t = 2:1:T
                        if Z{l}(t) == c
                           mid = mid + L_dotkt{l}(:,t);
                        end
                    end
                    a = mid; 
                    b  = Para.V{l,c}.*(Para.Xi{l,c}+repmat(sum(Para.V{l,c}),K(l),1)-Para.V{l,c});
                    Para.q{l,c} = betarnd(b,a);
                    Para.q{l,c} = max(Para.q{l,c},realmin);
                %% sample h   对
                    for k1 = 1:K(l)
                        for k2 = 1:K(l)
                            Para.h{l,c}(k1,k2) = CRT_sum_mex_matrix_v1(sparse(L_KK{l,c}(k1,k2)),Piprior{l,c}(k1,k2));
                        end
                    end
                %% sample Xi  对
                    shape = Supara.gamma0/K(l) + trace(Para.h{l,c});
                    scale = Para.beta{l,c} - Para.V{l,c}'*log(Para.q{l,c});
                    Para.Xi{l,c} = randg(shape)./scale;
                end
            end
            %% sample Delta 1  对
            [ delta{1} ] =Sample_delta(X_train,Theta{1},Supara.epilson0,Setting.Stationary);
            delta{1} = delta{1}.^0;
            
            if nnz(isnan(delta{1}))
                warning('delta Nan');
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% sample V{L}
%             for k=1:K(L)
%                 L_kdott{L}(k,1) = CRT_sum_mex_matrix_v1(sparse(A_KT{L}(k,1)+ L_dotkt{L}(k,2)),Supara.tao0*Para.V{L}(k));  %采样第一个时刻的隐层增广变量
%                 Para.n{L}(k)=sum(Para.h{L}(k,:)+Para.h{L}(:,k)')-Para.h{L}(k,k) + L_kdott{L}(k,1);
%                 Para.rou{L}(k) = -log(Para.q{L}(k)) * (Para.Xi{L}+sum(Para.V{L})-Para.V{L}(k)) - (log(Para.q{L}')*Para.V{L}-log(Para.q{L}(k))*Para.V{L}(k)) + Zeta{L}(1);
%             end
%             shape = Supara.gamma0/K(L) + Para.n{L};
%             scale = Para.beta{L} + Para.rou{L};
%             Para.V{L} = gamrnd(shape,1./scale);
            
            %% sample V{1},...,V{L-1]
            %if L>1
                for l=1:L
                    for c =1:1:C
                        for k=1:K(l)
                            Para.n{l}(k)=sum(Para.h{l,c}(k,:) + Para.h{l,c}(:,k)')-Para.h{l,c}(k,k);
                            Para.rou{l}(k) = -log(Para.q{l,c}(k)) * (Para.Xi{l,c}+sum(Para.V{l,c})-Para.V{l,c}(k)) - (log(Para.q{l,c}')*Para.V{l,c}-log(Para.q{l,c}(k))*Para.V{l,c}(k));% + Para.Zeta2(1);
                        end
                        shape = Supara.gamma0/K(l) + Para.n{l};
                        scale = Para.beta{l,c} + Para.rou{l};
                        Para.V{l,c} = gamrnd(shape,1./scale);
                    
                        if nnz(isnan( Para.V{l,c}))
                            warning('V Nan');
                        end
                    end
                end
           % end
            
            %% sample z
            for l = 1:1:L
                if l == L
                    for t = 2:1:T
                        u = rand(C,1);
                        g = -log(-log(u));
                        [b,iindex]=sort(Theta{l}(:,t));
                        max_index = iindex;
                        for j = 1:1:size(max_index)
                            if j==1
                                for c = 1:1:C
                                    %pi_C_mid{l}(c) =((max(Theta{l}(j,t),10*realmin)/10)^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1)*Supara.tao0^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)))/(gamma(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1))+realmin)*1/C*10^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1);                                   
                                    %pi_C_mid{l}(c) =((Theta{l}(j,t)/10)^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1)*Supara.tao0^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)))/(gamma(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1)+realmin)*pi0_c{l}(c)*10^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1);                                   
                                    %pi_C_mid{l}(c) =((Theta{l}(j,t))^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1)*Supara.tao0^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)))/(gamma(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1)+realmin)*1/5;                                   
                                    %pi_C_mid{l}(c) = exp((Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1)*log(max(Theta{l}(j,t),realmin))+(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1))*log(Supara.tao0)-gammaln(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)))*1/C;
                                    pi_C_mid{l}(c) = (Supara.tao0*Para.Pi{l,c}(max_index(j),:)*Theta{l}(:,t-1)-1)*log(max(Theta{l}(max_index(j),t),realmin))+(Supara.tao0*Para.Pi{l,c}(max_index(j),:)*Theta{l}(:,t-1))*log(Supara.tao0) - gammaln(max(realmin,Supara.tao0*Para.Pi{l,c}(max_index(j),:)*Theta{l}(:,t-1)));
                                    %pi_C_mid{l}(c) = exp((Supara.tao0*Para.Pi{l,c}(max_index(j),:)*Theta{l}(:,t-1)-1)*log(max(Theta{l}(max_index(j),t),realmin))+(Supara.tao0*Para.Pi{l,c}(max_index(j),:)*Theta{l}(:,t-1))*log(Supara.tao0)-gammaln(Supara.tao0*Para.Pi{l,c}(max_index(j),:)*Theta{l}(:,t-1)))*1/C;
                                end
                               % pi_C{l} = pi_C_mid{l}/(max(pi_C_mid{l})+realmin)+realmin;
                                %Z{l}(t) = find((g + log(pi_C{l})) == max(g + log(pi_C{l})));
                            else
                                 for c = 1:1:C
                                     %pi_C_mid{l}(c) = ((max(Theta{l}(j,t),10*realmin)/10)^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1)*(Supara.tao0.^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1))))/(gamma(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1))+realmin)*1/C*10^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1);                                     
                                     %pi_C_mid{l}(c) = ((Theta{l}(j,t)/10)^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1)*(Supara.tao0.^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1))))/(gamma(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1)+realmin)*pi0_c{l}(c)*10^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1);
                                     %pi_C_mid{l}(c) = ((Theta{l}(j,t))^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1)*(Supara.tao0.^(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1))))/(gamma(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1)+realmin)*1/5;                                     
                                     %pi_C_mid{l}(c) = exp((Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)-1)*log(max(Theta{l}(j,t),realmin))+(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1))*log(Supara.tao0)-gammaln(Supara.tao0*Para.Pi{l,c}(j,:)*Theta{l}(:,t-1)))*1/C;                                     
                                     pi_C_mid{l}(c) = pi_C_mid{l}(c) + (Supara.tao0*Para.Pi{l,c}(max_index(j),:)*Theta{l}(:,t-1)-1)*log(max(Theta{l}(max_index(j),t),realmin))+(Supara.tao0*Para.Pi{l,c}(max_index(j),:)*Theta{l}(:,t-1))*log(Supara.tao0)-gammaln(max(realmin,Supara.tao0*Para.Pi{l,c}(max_index(j),:)*Theta{l}(:,t-1)));
                                     %pi_C_mid{l}(c) = exp((Supara.tao0*Para.Pi{l,c}(max_index(j),:)*Theta{l}(:,t-1)-1)*log(max(Theta{l}(max_index(j),t),realmin))+(Supara.tao0*Para.Pi{l,c}(max_index(j),:)*Theta{l}(:,t-1))*log(Supara.tao0)-gammaln(Supara.tao0*Para.Pi{l,c}(max_index(j),:)*Theta{l}(:,t-1)))*1/C;
                                 end
%                                  pi_C{l} = exp(pi_C_mid{l}(c));  
%                                  %pi_C{l} = pi_C{l}.*(pi_C_mid{l}/(max(pi_C_mid{l})+realmin))+realmin;   
%                                  %Z{l}(t) = find((g + log(pi_C{l})) == max(g + log(pi_C{l})));
%                                  pi_C{l} = pi_C{l}/sum(pi_C{l});                             
                            end
                        end
                        pi_C_mid{l} = pi_C_mid{l} - max(pi_C_mid{l});
                        pi_C{l} = exp(pi_C_mid{l});  
                        %pi_C{l} = pi_C{l}.*(pi_C_mid{l}/(max(pi_C_mid{l})+realmin))+realmin;   
                        %Z{l}(t) = find((g + log(pi_C{l})) == max(g + log(pi_C{l})));
                        pi_C{l} = pi_C{l}/sum(pi_C{l});
                        %Z{l}(t) = find((g + log(pi_C{l})) == max(g + log(pi_C{l})));
                    end
                else
                    warning('Multi_layer')
                end
            end
%a = ((Theta{l}(j,t)/10)^(Supara.tao0*Para.Pi{l,1}(j,:)*Theta{l}(:,t-1)-1)*(Supara.tao0.^(Supara.tao0*Para.Pi{l,1}(j,:)*Theta{l}(:,t-1))))/(gamma(Supara.tao0*Para.Pi{l,1}(j,:)*Theta{l}(:,t-1)-1)+realmin)*pi0_c{l}(c)*10^(Supara.tao0*Para.Pi{l,1}(j,:)*Theta{l}(:,t-1)-1);            
            %% sample pi0_c
%             for l = 1:1:L
%                 if l == L
%                    for c=1:1:C
%                        eta(c) = sum(Z{l}==c);                       
%                    end
%                   pi0_c{l} = SamplePhi(eta,Supara.etac);
%                 else
%                     warning('Multi_layer')
%                 end    
%             end            
            %%
            Lambda = bsxfun(@times,delta{1}', Para.Phi{1}*Theta{1});
            like   = sum(sum( X_train .* log(Lambda)-Lambda));
            Likelihood(i) = like;
            if mod(i,Setting.step) == 0
                fprintf('Likelihood=%4.4f \n',Likelihood(i));
            end
            
         for iter = 1:1:2000    
        %% sample z_t
            for t = 1:1:T
                u = rand(1,C1);
                g = -log(-log(u));
                for c = 1:1:C1
                    pi_t_mid(c) = sum(log(lamda1(:,c)).*Theta{1,1}(:,t)-lamda1(:,c));
                    %pi_t(c) = exp(sum(log(lamda(:,c)).*X_train(:,t)-lamda(:,c)))*1/C;
                end
                pi_t_mid = pi_t_mid-max(pi_t_mid);
                for c = 1:1:C1
                     pi_t(c) = exp(pi_t_mid(c))*1/C1;
                end
                pi_t = pi_t/sum(pi_t);
                z_t(t) = find((g + log(pi_t)) == max(g + log(pi_t)));
            end                      
        %% sample pi
            num_pi = zeros(C1,1);
            for t = 1:1:T
                num_pi(z_t(t)) = num_pi(z_t(t));            
            end
            pi_0 = SamplePhi(num_pi,Supara.eta0);
        %% sample lamda            
            for c = 1:1:C1                
                for j = 1:1:K(1)
                    shape = 0;
                    scale = 0;
                    for t = 1:1:T
                        if z_t(t) == c
                            shape = shape + Theta{1,1}(j,t);
                            scale = scale + 1;
                        end
                    end
                    shape = shape + 1;
                    scale = scale + 0.1;
                    lamda1(j,c) = randg(shape)/scale;
                end
            end
                    end
        Z{1} = z_t;
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
                            Theta_pred{l} = Para.Pi{l,Z{l}(T)}*Theta{l}(:,end);
                        else
                            Theta_pred{l} = Para.Phi{l+1}*Theta_pred{l+1}+Para.Pi{l,Z{l}(T)}*Theta{l}(:,end);
                        end
                    end
                    
                    X_pred = bsxfun(@times,delta{1}(end)', Para.Phi{1} * Theta_pred{1});
                    pred_err = mean((X_test-X_pred).^2);
% % %                     
% % %                     [~,cpred] = sort(X_pred, 'descend' );
% % %                     [~,ctrue] = sort(X_test, 'descend' );
% % %                     %% Predictive recall (PP)
% % %                     range = 50; c = intersect(cpred(1:range),ctrue(1:range)) ;
% % %                     PP = length(c)/min(range,length(find( X_test ~=0)));
% % %                     %% Held out precision and recall
% % %                     X_held_hat = (bsxfun(@times,delta{1}', Para.Phi{1}*Theta{1}));
% % %                     [~,cpred] = sort(X_held_hat, 1,'descend' );
% % %                     [~,ctrue] = sort(X_held, 1, 'descend' );
% % %                     range = 50;
% % %                     for t = 1:T
% % %                         g1 = zeros(V,1); g2 = zeros(V,1);
% % %                         g1(ctrue(1:range, t),1) = 1;
% % %                         g2(cpred(1:range, t),1) = 1;
% % %                         con = confusionmat(g1,g2);
% % %                         TNR(t) = con(2,2)/(con(2,2)+con(1,2));
% % %                         TT = find( X_held(:,t) ~= 0  );
% % %                         CC = intersect(TT, cpred(1:range, t) );
% % %                         TPR(t) = length(CC)/length(TT);
% % %                     end
% % %                     MP = mean(TNR); MR =  mean(TPR);
                    MP = 0; MR =  0;PP = 0;
                    result=[rec_err,pred_err,MP,MR,PP];
%                     if i > Setting.Burnin
%                         if i == Setting.Burnin+1
%                             ave_result = result;
%                         else
%                             ave_result = (ave_result + result)/2;
%                         end
%                     end                    
                    
                     Result.recErr_test = [Result.recErr_test;result(1)];
                    Result.pred_err = [Result.pred_err;result(2)];
                    
                    Result.MP=[Result.MP; result(3)];
                    Result.MR=[Result.MR; result(4)];
                    Result.PP=[Result.PP; result(5)];
                    
                    
                    
                    if mod(i,Setting.step)==0
                        fprintf('dataname:%s,Iter %d:  RecErr=%4.4f, PredErr=%4.4f, MP=%4.4f, MR=%4.4f, PP=%4.4f\n',...
                           dataname,i,result(1),result(2), result(3), result(4),result(5));
%                         fprintf('dataname:%s,Iter %d:  RecErr=%4.4f, PredErr=%4.4f, MP=%4.4f, MR=%4.4f, PP=%4.4f\n',...
%                             dataname,i,ave_result(1),ave_result(2), ave_result(3), ave_result(4),ave_result(5));
%                         fprintf('dataname:%s,Iter %d:  RecErr=%4.4f',...
%                             dataname,i,rec_err);
                        %fprintf('Likelihood=%4.4f \n',Likelihood(i));
                        %fprintf('trial %d',trial);
                        %fprintf('.................................................................................... \n')
%                         name_save = [dataname,'_','_trial',num2str(trial),evalute_name,'_Layer',num2str(L),'_S',num2str(Setting.Stationary),'.mat'];
%                         save(['./TopM_result_new/',name_save])
                    end
                    
            end
            
            %             name_save = [dataname,'_','_trial',num2str(trial),evalute_name,'_Layer',num2str(L),'_S',num2str(Setting.Stationary),'.mat'];
            %             save(['./TopM_result_new/',name_save])
        end
        for iter = 1:1:2000    
        %% sample z_t
            for t = 1:1:T
                u = rand(1,C1);
                g = -log(-log(u));
                for c = 1:1:C1
                    pi_t_mid(c) = sum(log(lamda1(:,c)).*Theta{1,1}(:,t)-lamda1(:,c));
                    %pi_t(c) = exp(sum(log(lamda(:,c)).*X_train(:,t)-lamda(:,c)))*1/C;
                end
                pi_t_mid = pi_t_mid-max(pi_t_mid);
                for c = 1:1:C1
                     pi_t(c) = exp(pi_t_mid(c))*1/C1;
                end
                pi_t = pi_t/sum(pi_t);
                z_t(t) = find((g + log(pi_t)) == max(g + log(pi_t)));
            end                      
        %% sample pi
            num_pi = zeros(C1,1);
            for t = 1:1:T
                num_pi(z_t(t)) = num_pi(z_t(t));            
            end
            pi_0 = SamplePhi(num_pi,Supara.eta0);
        %% sample lamda            
            for c = 1:1:C1                
                for j = 1:1:K(1)
                    shape = 0;
                    scale = 0;
                    for t = 1:1:T
                        if z_t(t) == c
                            shape = shape + Theta{1,1}(j,t);
                            scale = scale + 1;
                        end
                    end
                    shape = shape + 1;
                    scale = scale + 0.1;
                    lamda1(j,c) = randg(shape)/scale;
                end
            end
        end
        ave_result(1) = mean(Result.recErr_test(Setting.Burnin+1:end));
        ave_result(2) = mean(Result.pred_err(Setting.Burnin+1:end));
        ave_result(3) = mean(Result.MP(Setting.Burnin+1:end));
        ave_result(4) = mean(Result.MR(Setting.Burnin+1:end));
        ave_result(5) = mean(Result.PP(Setting.Burnin+1:end));
        
    end
end
A=1;
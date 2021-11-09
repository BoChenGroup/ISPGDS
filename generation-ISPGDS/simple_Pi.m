clc
clear
close all
%%
Setting.Burnin     =   600;
Setting.Collection =   500;
Setting.Stationary =   1;
Setting.NSample    =   1;
Setting.step = 10;
%%  评价指标
evalute = 2;
evalute_name_all = {'MSE_PMSE','Top@M','S_F_mask'};
evalute_name  = cell2mat(evalute_name_all(evalute));

%%  数据选取
datasetname = {'dblp','nips','sotu','gdelt2001','gdelt2002','gdelt2003','gdelt2004'...
    ,'gdelt2005','icews200123','icews200456','icews200789','icews2003'};


for data_each = 6:6
    dataname  = cell2mat(datasetname(data_each));  
    
        switch evalute_name
            case  'Top@M'
                filepath = '.\TopM_data\topM_';
                filename = strcat(filepath ,  dataname);
                load (filename);             
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
            case   'MSE_PMSE'             
                filepath = 'F:\dynamic_datasets\thematlab\';
                filename = strcat(filepath,dataname);
                load (filename);
                
                X_train = (double(Y_TV(1:end-1,:)))';
                X_test = (double(Y_TV(end,:)))';
                [V,T] = size(X_train);

                pred_err_sample = zeros(Setting.Collection+Setting.Burnin,1);
                pred_err_mean = zeros(Setting.Collection+Setting.Burnin,1);
                rec_err_sample = zeros(Setting.Collection+Setting.Burnin,1);
                rec_err_mean  = zeros(Setting.Collection+Setting.Burnin,1);
        end
    
    %% 超参数
    Supara.tao0 = 1;
    Supara.gamma0 = 100;
    Supara.eta0 = 0.1;
    Supara.v0 = 0.1;
    Supara.epilson0 = 0.1;
    Supara.c = 1;
    
    %% 参数初始化
    L=3;
    K =   [200,100,50];
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
    L_dotKT  = cell(L,1);
    L_kdott = cell(L,1);
    A_KT = cell(L,1);
    A_VK = cell(L,1);
    
    L_KK = cell(L,1);
    prob1 = cell(L,1);
    prob2 = cell(L,1);
    Xt_to_t1 = cell(L,1);   
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
        delta{l}    = ones(T,1);
        Zeta{l}     = zeros(T,1);
        L_dotKT{l} = zeros(K(l),T+1);
        A_KT{l} = zeros(K(l),T);
        L_kdott{l} = zeros(K(l),T+1);
        X_layer{l} = zeros(K(l),T,2);
 
    end

    

switch evalute_name
    case   'MSE_PMSE'
        
    case   'Top@M'
        
        Result.recErr_test = [];

        Result.MP=[];
        Result.MR=[];
        Result.PP=[];
        Result.pred_err=[];
end

    for i = 1:Setting.Collection+Setting.Burnin
            
        for l=1:L-1
            prob1{l} = Supara.tao0*Para.Pi{l} *  Theta{l};         
            prob2{l} = Supara.tao0*Para.Phi{l+1} *  Theta{l+1};        
            L_KK{l} = zeros(K(l),K(l));  
            
            if l==1
                [A_KT{l},A_VK{l}] = Multrnd_Matrix_mex_fast_v1(sparse(X_train),Para.Phi{l},Theta{l}); 
            else                
                [A_KT{l},A_VK{l}] = Multrnd_Matrix_mex_fast_v1(sparse(Xt_to_t1{l-1}),Para.Phi{l},Theta{l});                
            end                         
          %% sample next layer count                                 
            for t = T : -1 : 2               
                 L_kdott{l}(:,t) = CRT_sum_mex_matrix_v1(sparse(A_KT{l}(:,t)'+ L_dotKT{l}(:,t+1)'),Supara.tao0*(Para.Phi{l+1}*Theta{l+1}(:,t)+Para.Pi{l} * Theta{l}(:,t-1))')';      
                %% split layer2 count            
                 [~,X_layer{l}(:,t,:)] = Multrnd_Matrix_mex_fast_v2(L_kdott{l}(:,t),[prob1{l}(:,t) prob2{l}(:,t-1)],ones(2,1));
                 X_layer_split1{l}(:,t) = squeeze(X_layer{l}(:,t,1));   %pi1*Theta1
                 Xt_to_t1{l}(:,t) = squeeze(X_layer{l}(:,t,2));   %phi2*Theta2
                %% sample split1 augmentation
                 [L_dotKT{l}(:,t),tmp] = Multrnd_Matrix_mex_fast_v1(sparse(X_layer_split1{l}(:,t)),Para.Pi{l},Theta{l}(:,t-1));
                 L_KK{l} = L_KK{l} + tmp ;             
            end

            L_kdott{l}(:,1) = CRT_sum_mex_matrix_v1(sparse(A_KT{l}(:,1)'+ L_dotKT{l}(:,2)'),Supara.tao0*(Para.Phi{l+1}*Theta{l+1}(:,1))')';    
            Xt_to_t1{l}(:,1) = L_kdott{l}(:,1); 

        end
        
        %% Sample augmention in layer L
        
        [A_KT{L},A_VK{L}] = Multrnd_Matrix_mex_fast_v1(sparse(Xt_to_t1{L-1}),Para.Phi{L},Theta{L});

        L_KK{L} = zeros(K(L),K(L));
        for t=T:-1:2
            L_kdott{L}(:,t) = CRT_sum_mex_matrix_v1(sparse((A_KT{L}(:,t)+ L_dotKT{L}(:,t+1))'),(Supara.tao0*Para.Pi{L} * Theta{L}(:,t-1))')';
            [L_dotKT{L}(:,t),tmp] = Multrnd_Matrix_mex_fast_v1(sparse(L_kdott{L}(:,t)),Para.Pi{L},Theta{L}(:,t-1));
            L_KK{L} = L_KK{L} + tmp;
        end        
                
        %% sample Phi  对
        for l=1:L
            Para.Phi{l} = SamplePhi(A_VK{l},Supara.eta0);
            if nnz(isnan(Para.Phi{l}))
                warning('Phi Nan');
                Para.Phi{l}(isnan(Para.Phi{l})) = 0;
            end   
        end        
      %% sample Pi 对
        for l=1:L      
            Piprior{l} = Supara.v0;

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
                L_dotKT{l}(:,T+1) = poissrnd(Zeta{l}(1)*Supara.tao0 * Theta{l}(:,T));
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
                         Para.Zeta{l}(t) = Supara.tao0*log(1+Para.Zeta{l-1}(t)+Para.Zeta{l}(t+1));
                     end
                                
                end

            end
        end   
        %% sample Theta  对
        for l=1:L-1
            for t=1:T
                if t==1
                    shape = A_KT{l}(:,t)+ L_dotKT{l}(:,t+1)+ Supara.tao0*Para.Phi{l+1}* Theta{l+1}(:,t);
                else
                    shape = A_KT{l}(:,t)+ L_dotKT{l}(:,t+1)+ Supara.tao0*(Para.Phi{l+1} * Theta{l+1}(:,t)+Para.Pi{l}* Theta{l}(:,t-1));
                end
                    scale = ( delta{l}(t) + Supara.tao0 + Supara.tao0 * Zeta{l}(t+1))';                
                Theta{l}(:,t) = gamrnd(shape,1./scale);
            end

            if nnz(isnan(Theta{l}))
                warning('Theta Nan');
            end
        end
        %% sample Theta{L}        
         for t=1:T
            if t==1
                shape = A_KT{L}(:,t)+ L_dotKT{L}(:,t+1)+ Supara.tao0*Para.V{L};
            else
                shape = A_KT{L}(:,t)+ L_dotKT{L}(:,t+1)+ Supara.tao0*(Para.Pi{L}* Theta{L}(:,t-1));
            end
            scale = Supara.tao0 + delta{L}(t)+ Zeta{L}(t+1);
            Theta{L}(:,t) = gamrnd(shape,1./scale);
         end        
        if nnz(isnan(Theta{L}))
            warning('Theta Nan');
        end    
        %% sample Beta    对
%         for l = 1:L
%             shape = Supara.epilson0 + Supara.gamma0;
%             scale = Supara.epilson0 + sum(Para.V{l});
%             Para.beta{l} = gamrnd(shape,1./scale); 
%         end
        
       %% sample q  对
%        for l=1:L
%             a  = sum(L_dotKT{l},2);
%             b  = Para.V{l}.*(Para.Xi{l}+repmat(sum(Para.V{l}),K(l),1)-Para.V{l});
%             Para.q{l} = betarnd(b,a);
%             Para.q{l} = max(Para.q{l},realmin);
%        end 
       %% sample h   对
%         for l=1:L
%             for k1 = 1:K(l)                
%                 for k2 = 1:K(l)                    
%                     Para.h{l}(k1,k2) = CRT_sum_mex_matrix_v1(sparse(L_KK{l}(k1,k2)),Piprior{l}(k1,k2));                
%                 end
%             end   
%         end
        
       %% sample Xi  对
%        for l=1:L
%         shape = Supara.gamma0/K(l) + trace(Para.h{l});
%         scale = Para.beta{l} - Para.V{l}'*log(Para.q{l});
%         Para.Xi{l} = gamrnd(shape,1./scale);
%        end    
         %% sample Delta 1  对
        [ delta{1} ] =Sample_delta(X_train,Theta{1},Supara.epilson0,Setting.Stationary);
        
        if nnz(isnan(delta{1}))
            warning('delta Nan');
        end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% sample V{L}
%         for k=1:K(L)
%             L_kdott{L}(k,1) = CRT_sum_mex_matrix_v1(sparse(A_KT{L}(k,1)+ L_dotKT{L}(k,2)),Supara.tao0*Para.V{L}(k));
%             Para.n{L}(k)=sum(Para.h{L}(k,:)+Para.h{L}(:,k)')-Para.h{L}(k,k) + L_kdott{L}(k,1);
%             Para.rou{L}(k) = -log(Para.q{L}(k)) * (Para.Xi{L}+sum(Para.V{L})-Para.V{L}(k)) - (log(Para.q{L}')*Para.V{L}-log(Para.q{L}(k))*Para.V{L}(k)) + Zeta{L}(1);
%         end
%         shape = Supara.gamma0/K(L) + Para.n{L};
%         scale = Para.beta{L} + Para.rou{L};
%         Para.V{L} = gamrnd(shape,1./scale);     
%         
        %% sample V{1},...,V{L]
%         for l=1:L-1
%             for k=1:K(l)
%                 Para.n{l}(k)=sum(Para.h{l}(k,:) + Para.h{l}(:,k)')-Para.h{l}(k,k);
%                 Para.rou{l}(k) = -log(Para.q{l}(k)) * (Para.Xi{l}+sum(Para.V{l})-Para.V{l}(k)) - (log(Para.q{l}')*Para.V{l}-log(Para.q{l}(k))*Para.V{l}(k));% + Para.Zeta2(1);
%             end
%             shape = Supara.gamma0/K(l) + Para.n{l};
%             scale = Para.beta{l} + Para.rou{l};
%             Para.V{l} = gamrnd(shape,1./scale);
% 
%             if nnz(isnan( Para.V{l}))
%                 warning('V Nan');
%             end
%         end                   
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
                if mod(i,Setting.step)==0
                    fprintf('dataname:%s,iter: %d, predict1_error: %d, reco_error: %d, like: %d  \n',dataname,i,pred_err_mean(i,1),rec_err_mean(i,1),like);
                end  
                
            case   'Top@M'
                result = pgds_calc_pred_error_layer3_doublepi(X_held,X_test,X_train,Para,Theta,delta,L);
                Result.recErr_test = [Result.recErr_test;result(1)];
                Result.pred_err = [Result.pred_err;result(2)];

                Result.MP=[Result.MP; result(3)];
                Result.MR=[Result.MR; result(4)];
                Result.PP=[Result.PP; result(5)];


            if mod(i,Setting.step)==0
                fprintf('dataname:%s,Iter %d:  RecErr=%4.4f, PredErr=%4.4f, MP=%4.4f, MR=%4.4f, PP=%4.4f\n',...
                    dataname,i,result(1),result(2), result(3), result(4),result(5));            
                fprintf('Likelihood=%4.4f \n',Likelihood(i));
                fprintf('.................................................................................... \n')
            end

        end
        
        
    end   
    name_save = [dataname,'_',evalute_name,'_Layer',num2str(L),'_S',num2str(Setting.Stationary),'.mat'];
    save(['./TopM_result/',name_save])
end

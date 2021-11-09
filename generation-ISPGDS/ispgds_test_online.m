function [log_test,rec_err,pred_err] =  pgds_test_online_layer1(X_test,Para,Supara,Setting)

Ttest =size(X_test,2);
Theta_test    = randg(1,Setting.K,Ttest);
L_kdott_test = zeros(Setting.K,Ttest+1);
L_dotkt_test = zeros(Setting.K,Ttest+1);
delta_test=ones(Ttest,1);
Zeta_test = zeros(Ttest+1,1);


for iter   =   1 : (Setting.BurninMB + Setting.CollectionMB)
    
    [Y_KT_test,~] = Multrnd_Matrix_mex_fast_v1(sparse(X_test),Para.Phi,Theta_test);
    for t = Ttest:-1:2
        L_kdott_test(:,t) = CRT_sum_mex_matrix_v1(sparse((Y_KT_test(:,t)+ L_dotkt_test(:,t+1))'),(Supara.tao0*Para.Pi * Theta_test(:,t-1))')';
        [L_dotkt_test(:,t),~] = Multrnd_Matrix_mex_fast_v1(sparse(L_kdott_test(:,t)),Para.Pi,Theta_test(:,t-1));
    end
    
    %% calsulate Zeta  ¶Ô
    if Setting.Stationary == 1
        Zeta_test= -lambertw(-1,-exp(-1-delta_test./Supara.tao0))-1-delta_test./Supara.tao0;
        Zeta_test(Ttest+1) = Zeta_test(1);
        L_dotkt_test(:,Ttest+1) =poissrnd(Zeta_test(1)*Supara.tao0*Theta_test(:,Ttest));
    end
    
    if Setting.Stationary == 0
        for t=Ttest:-1:1
            %Setting.Zeta(t) = log(1+Setting.delta(t)/Supara.tao0+Setting.Zeta(t+1));
            Zeta_test(t) = Supara.tao0*log(1+(delta_test(t) + Zeta_test(t+1))/Supara.tao0);
        end
    end
    
    if nnz(isnan(Zeta_test))
        warning('Zeta Nan');
    end
    
    %% sample Theta_test  ¶Ô
    for t=1:Ttest
        if t==1
            shape = Y_KT_test(:,t) + L_dotkt_test(:,t+1) + Supara.tao0 * Para.V;
        else
            shape = Y_KT_test(:,t) + L_dotkt_test(:,t+1) + Supara.tao0*Para.Pi * Theta_test(:,t-1);
        end
        scale = Supara.tao0 + delta_test(t) + Zeta_test(t+1);
        Theta_test(:,t) = gamrnd(shape,1./scale);
    end
    
    
    %% sample Delta   ¶Ô
    [ delta_test ] =Sample_delta(X_test, Theta_test,Supara.epilson0,Setting.Stationary);
    if nnz(isnan(delta_test))
        warning('delta Nan');
    end

end
%% log test
Lambda_test = bsxfun(@times,delta_test', Para.Phi * Theta_test);
log_test   = sum(sum( X_test .* log(Lambda_test)-Lambda_test));   
%% reconstruction
rec_err = mean(sum((X_test(:,2:Ttest)-Lambda_test(:,2:Ttest)).^2));
%% prediction step
Theta_pred = Para.Pi * Theta_test(:,1:Ttest-1);

X_pred = bsxfun(@times,delta_test(1:Ttest-1)', Para.Phi * Theta_pred);

pred_err = mean(sum((X_test(:,2:Ttest)-X_pred).^2));
end
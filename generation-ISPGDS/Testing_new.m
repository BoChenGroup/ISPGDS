 function [logpv_test,recErr_test,predErr_test,Theta_test]  =   Testing_new(L,K,TestData,Para,DataType,Supara,Setting,Theta_test)

[V,Ttest] = size(TestData);

logpv_test=[]; 
recErr_test = [];
predErr_test = []; 

Theta_test    =  cell(L,1);
L_kdott = cell(L,1);
L_dotkt = cell(L,1);
delta_test   =  cell(L,1);
Zeta    =  cell(L,1);
A_KT   =  cell(L,1);

Xt_to_t1 = cell(L+1,1);
X_layer_split1 = cell(L,1);
X_layer  = cell(L,1);

for l=1:L

        Theta_test{l}    = randg(1,K(l),Ttest);
        L_kdott{l} = zeros(K(l),Ttest+1);
        L_dotkt{l} = zeros(K(l),Ttest+1);
        delta_test{l} = ones(Ttest,1);
        Zeta{l}  = zeros(Ttest+1,1);

end
%%

for iter_test = 1:10
            tic
            
             switch DataType
                case 'Binary'
                    X_test = round(TestData);   
                    [ii,jj,M] = find(X_test);
                    iijj=find(X_test);
                    Xmask=sparse(X_test);
%                     delta_theta = bsxfun(@times,delta{1}',Theta{1});
                    Rate = Mult_Sparse(Xmask, Para.Phi{1},Theta_test{1});
                    M = truncated_Poisson_rnd(full(Rate(iijj)));
                    Xt = sparse(ii,jj,M,V,Ttest);   
                    Xt_full =  full(Xt);
                    
                case 'Count'
                    X_test = round(TestData*Setting.num_data);   
                    Xt = sparse(X_test);   
             end  
           
             
           for l=1:L
    
                if l==1
                    [A_KT{l},~] = Multrnd_Matrix_mex_fast_v1(Xt, Para.Phi{l},Theta_test{l}); 
                else                
                    [A_KT{l},~] = Multrnd_Matrix_mex_fast_v1(sparse(Xt_to_t1{l}), Para.Phi{l},Theta_test{l});                
                end                         
                  %% sample next layer count 
                  if l == L            

                        for t=Ttest:-1:2
                            L_kdott{l}(:,t) = CRT_sum_mex_matrix_v1(sparse((A_KT{l}(:,t)+ L_dotkt{l}(:,t+1))'),(Supara.tao0 * Para.Pi{l} * Theta_test{l}(:,t-1))')';
                            [L_dotkt{l}(:,t),~] = Multrnd_Matrix_mex_fast_v1(sparse(L_kdott{l}(:,t)), Para.Pi{l},Theta_test{l}(:,t-1));
                        end   
                        
         
                  else    

                        prob1{l} = Supara.tao0 * Para.Pi{l} *  Theta_test{l};         
                        prob2{l} = Supara.tao0 * Para.Phi{l+1} *  Theta_test{l+1}; 
                        X_layer{l} = zeros(K(l),Ttest,2);
                        Xt_to_t1{l+1} = zeros(K(l),Ttest);
                        X_layer_split1{l} = zeros(K(l),Ttest);

                        for t = Ttest : -1 : 2               
                             L_kdott{l}(:,t) = CRT_sum_mex_matrix_v1(sparse(A_KT{l}(:,t)'+ L_dotkt{l}(:,t+1)'),Supara.tao0*( Para.Phi{l+1}*Theta_test{l+1}(:,t)+ Para.Pi{l} * Theta_test{l}(:,t-1))')';      
                            %% split layer2 count            
                             [~,X_layer{l}(:,t,:)] = Multrnd_Matrix_mex_fast_v2(L_kdott{l}(:,t),[prob1{l}(:,t-1) prob2{l}(:,t)],ones(2,1));
                             X_layer_split1{l}(:,t) = squeeze(X_layer{l}(:,t,1));   %pi1*Theta1
                             Xt_to_t1{l+1}(:,t) = squeeze(X_layer{l}(:,t,2));   %phi2*Theta2
                            %% sample split1 augmentation
                             [L_dotkt{l}(:,t),~] = Multrnd_Matrix_mex_fast_v1(sparse(X_layer_split1{l}(:,t)), Para.Pi{l},Theta_test{l}(:,t-1));
                        end

                        L_kdott{l}(:,1) = CRT_sum_mex_matrix_v1(sparse(A_KT{l}(:,1)'+ L_dotkt{l}(:,2)'),Supara.tao0*( Para.Phi{l+1}*Theta_test{l+1}(:,1))')';    
                        Xt_to_t1{l+1}(:,1) = L_kdott{l}(:,1); 

                  end

            end
            %% calculate Zeta  ¶Ô
            if Setting.Stationary == 1
                for l=1:L
                    if l==1
                        Zeta{l}= -lambertw(-1,-exp(-1-delta_test{l}./Supara.tao0))-1-delta_test{l}./Supara.tao0;
                    else
                        Zeta{l} = -lambertw(-1,-exp(-1-Zeta{l-1}))-1-Zeta{l-1};
                    end
                    Zeta{l}(Ttest+1) = Zeta{l}(1);
                    L_dotkt{l}(:,Ttest+1) = poissrnd(Zeta{l}(1)*Supara.tao0 * Theta_test{l}(:,Ttest));
                end
            end
            
            
            if Setting.Stationary == 0
                for l=1:L
                    if l==1
                        for t=Ttest:-1:1
                            Zeta{l}(t) = log(1 + delta_test{l}(t)/Supara.tao0 + Zeta{l}(t+1));
                        end
                    else
                        for t=Ttest:-1:1
                            Para.Zeta{l}(t) = Supara.tao0*log(1+Para.Zeta{l-1}(t)+Para.Zeta{l}(t+1));
                        end
                        
                    end
                    
                end
            end
            %% sample Theta  ¶Ô
        for l=L:-1:1
           
           if l==L
                for t=1:Ttest
                    if t==1
                        shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0 * Para.V{l};
                    else
                        shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0*(Para.Pi{l}* Theta_test{l}(:,t-1));
                    end
                    scale = Supara.tao0 + delta_test{l}(t)+ Zeta{l}(t+1);
                    Theta_test{l}(:,t) = randg(shape)./scale;
                 end 
         
           else
                for t=1:Ttest
                    if t==1
                        shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0 * Para.Phi{l+1}* Theta_test{l+1}(:,t);
                    else
                        shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0 * (Para.Phi{l+1} * Theta_test{l+1}(:,t)+ Para.Pi{l}* Theta_test{l}(:,t-1));
                    end
                        scale = ( delta_test{l}(t) + Supara.tao0 + Supara.tao0 * Zeta{l}(t+1))';                
                       Theta_test{l}(:,t) = randg(shape)./scale;
                end
               
               
           end
           
                       if nnz(isnan(Theta_test{l}))
                            warning('Theta_test Nan');
                       end
                    
       end          
            
            
           

            %% sample Delta 1  ¶Ô
%             [ delta{1} ] =Sample_delta(X_train,Theta{1},Supara.epilson0,Setting.Stationary);
%             
%             if nnz(isnan(delta{1}))
%                 warning('delta Nan');
%             end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
            
      
            
            %%  likelihood
            switch DataType
                case 'Binary'
  
                        Rate = bsxfun(@times,delta_test{1}',Para.Phi{1}*Theta_test{1});
                        u_test =  (1- exp( - Rate));
                        log_test = sum(sum(X_test .* log(u_test) + (1-X_test ).*log(1-u_test)));
                        %% reconstruction
                        rec_err = mean(sum((TestData(:,2:Ttest)-u_test(:,2:Ttest)).^2));
                        %% prediction step
                        Theta_pred = Para.Pi{1} * Theta_test{1}(:,1:Ttest-1);

                        Rate_pred = bsxfun(@times,delta_test{1}(1:Ttest-1)', Para.Phi{1} * Theta_pred);
                        u_pred =  (1- exp( - Rate_pred));

                        pred_err = mean((sum((TestData(:,2:Ttest) - u_pred).^2)));                    
                case 'Count'
   
                    
       Lambda_test = bsxfun(@times,delta_test{1}',Para.Phi{1}*Theta_test{1});

                        log_test = sum(sum(X_test .*  log(Lambda_test)-Lambda_test));
                        %% reconstruction
                        err  =  (X_test(:,2:Ttest)-Lambda_test(:,2:Ttest))/Setting.num_data ;
                        rec_err = mean(sum(err.^2));
                        %% prediction step
                        
                        Theta_pred = Para.Pi{1} * Theta_test{1}(:,1:Ttest-1);

                        Lambda_pred = bsxfun(@times,delta_test{1}(1:Ttest-1)', Para.Phi{1} * Theta_pred);
                        err = (X_test(:,2:Ttest) - Lambda_pred) / Setting.num_data ;
                        pred_err = mean((sum(err.^2)));
           
            end 
             

            
     logpv_test=[logpv_test;log_test]; recErr_test = [recErr_test;rec_err];
            predErr_test = [predErr_test;pred_err]; 
            
            
              fprintf('Test,Iter %d: logpv=%4.8f, recErr=%4.8f, predErr=%4.8f\n',...
               iter_test,logpv_test(end),recErr_test(end),predErr_test(end));
                
            

        
        
        end
 end      
%% update Theta_test
 

    
for l=1:L                   
                if l==1
                    [A_KT_test{l},~] = Multrnd_Matrix_mex_fast_v1(Xt_test_sparse, Para.Phi{l},Theta_test{l}); 
                else                
                    [A_KT_test{l},~] = Multrnd_Matrix_mex_fast_v1(sparse(Xt_to_t1_test{l}), Para.Phi{l},Theta_test{l});                
                end                         
                  %% sample next layer count 
                  if l == L            
                        for t=Ttest:-1:2
                            L_kdott_test{l}(:,t) = CRT_sum_mex_matrix_v1(sparse((A_KT_test{l}(:,t)+ L_dotkt_test{l}(:,t+1))'),(Supara.tao0 * Para.Pi{l} * Theta_test{l}(:,t-1))')';
                            [L_dotkt_test{l}(:,t),~] = Multrnd_Matrix_mex_fast_v1(sparse(L_kdott_test{l}(:,t)), Para.Pi{l},Theta_test{l}(:,t-1));
                        end      

                  else    

                        prob1{l} = Supara.tao0 * Para.Pi{l} *  Theta_test{l};         
                        prob2{l} = Supara.tao0 * Para.Phi{l+1} *  Theta_test{l+1}; 
                        X_layer_test{l} = zeros(K(l),Ttest,2);
                        Xt_to_t1_test{l+1} = zeros(K(l),Ttest);
                        X_layer_split1_test{l} = zeros(K(l),Ttest);

                        for t = Ttest : -1 : 2               
                             L_kdott_test{l}(:,t) = CRT_sum_mex_matrix_v1(sparse(A_KT_test{l}(:,t)'+ L_dotkt_test{l}(:,t+1)'),Supara.tao0*( Para.Phi{l+1}*Theta_test{l+1}(:,t)+ Para.Pi{l} * Theta_test{l}(:,t-1))')';      
                            %% split layer2 count            
                             [~,X_layer_test{l}(:,t,:)] = Multrnd_Matrix_mex_fast_v2(L_kdott_test{l}(:,t),[prob1{l}(:,t) prob2{l}(:,t-1)],ones(2,1));
                             X_layer_split1_test{l}(:,t) = squeeze(X_layer_test{l}(:,t,1));   %pi1*Theta1
                             Xt_to_t1_test{l+1}(:,t) = squeeze(X_layer_test{l}(:,t,2));   %phi2*Theta2
                            %% sample split1 augmentation
                             [L_dotkt_test{l}(:,t),~] = Multrnd_Matrix_mex_fast_v1(sparse(X_layer_split1_test{l}(:,t)), Para.Pi{l},Theta_test{l}(:,t-1));
                        end

                        L_kdott_test{l}(:,1) = CRT_sum_mex_matrix_v1(sparse(A_KT_test{l}(:,1)'+ L_dotkt_test{l}(:,2)'),Supara.tao0*( Para.Phi{l+1}*Theta_test{l+1}(:,1))')';    
                        Xt_to_t1_test{l+1}(:,1) = L_kdott_test{l}(:,1); 

                  end
           
      
end

%% calculate Zeta  ¶Ô
            if Setting.Stationary == 1
                for l=1:L
                    if l==1
                        Zeta_test{l}= -lambertw(-1,-exp(-1-delta_test{l}./Supara.tao0))-1-delta_test{l}./Supara.tao0;
                    else
                        Zeta_test{l} = -lambertw(-1,-exp(-1-Zeta_test{l-1}))-1-Zeta_test{l-1};
                    end
                    
                    Zeta_test{l}(Ttest+1) = Zeta_test{l}(1);
                    L_dotkt_test{l}(:,Ttest+1) = poissrnd(Zeta_test{l}(1)*Supara.tao0 * Theta_test{l}(:,Ttest));
                    
                end
            end
            
            
            if Setting.Stationary == 0
                for l=1:L
                    if l==1
                        for t=Ttest:-1:1
                            Zeta_test{l}(t) = log(1 + delta_test{l}(t)/Supara.tao0 + Zeta_test{l}(t+1));
                        end
                    else
                        for t=Ttest:-1:1
                            Para.Zeta_test{l}(t) = Supara.tao0*log(1 + Zeta_test{l-1}(t) + Zeta_test{l}(t+1));
                        end
                        
                    end
                    
                end
            end                      
                      
                    
            
   

            

           %% sample Theta  ¶Ô
            for l=1:L

               if l==L
                    for t=1:Ttest
                        if t==1
                            shape = A_KT_test{l}(:,t)+ L_dotkt_test{l}(:,t+1)+ Supara.tao0 * Para.V{l};
                        else
                            shape = A_KT_test{l}(:,t)+ L_dotkt_test{l}(:,t+1)+ Supara.tao0*(Para.Pi{l}* Theta_test{l}(:,t-1));
                        end
                        scale = Supara.tao0 + delta_test{l}(t)+ Zeta_test{l}(t+1);
                        Theta_test{l}(:,t) = gamrnd(shape,1./scale);
                     end 

               else
                    for t=1:Ttest
                        if t==1
                            shape = A_KT_test{l}(:,t)+ L_dotkt_test{l}(:,t+1)+ Supara.tao0 * Para.Phi{l+1}* Theta_test{l+1}(:,t);
                        else
                            shape = A_KT_test{l}(:,t)+ L_dotkt_test{l}(:,t+1)+ Supara.tao0 * (Para.Phi{l+1} * Theta_test{l+1}(:,t)+ Para.Pi{l}* Theta_test{l}(:,t-1));
                        end
                            scale = ( delta_test{l}(t) + Supara.tao0 + Supara.tao0 * Zeta_test{l}(t+1))';                
                        Theta_test{l}(:,t) = gamrnd(shape,1./scale);
                    end


               end

                       if nnz(isnan(Theta_test{l}))
                            warning('Theta Nan');
                       end

            end  
            
            
         
   
            %% sample Delta 1  ¶Ô
%             [ delta_test{1} ] =Sample_delta(X_test,Theta_test{1},Supara.epilson0,Setting.Stationary);
%             
%             if nnz(isnan(delta_test{1}))
%                 warning('delta Nan');
%             end            
            


            
          
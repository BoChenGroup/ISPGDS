%Prune the inactive factors of the current top hidden layer


        for l=Lcurrent:Lcurrent
            switch Setting.Trim_K
                case 'shape'
                        dexK = find(sum(A_KT{l}+ L_dotkt{l}(:,2:end),2)==0);
                case 'Theta_1_end'
                        dexK = find(sum(Theta{l},2)==0);
                case 'Theta_2_end'
                        dexK = find(sum(Theta{l}(:,2:end),2)==0);
            end
            
            
            if ~isempty(dexK)
                K(l)=K(l)-length(dexK);
                Theta{l}(dexK,:)=[];        %  ThetaP{t}(dexK,:)    =   [];     ThetaC{t}(dexK,:)    =   [];     
                Pi{l}(:,dexK)=[];
                Pi{l}(dexK,:)=[];
                VP{l}(dexK,:)=[];
                q{l}(dexK,:)=[];
                h{l}(dexK,:)=[];
                h{l}(:,dexK)=[];
                n{l}(dexK,:)=[];
                rou{l}(dexK,:)=[];
                Phi{l}(:,dexK)=[];
                A_KT{l}(dexK,:)=[];
                L_dotkt{l}(dexK,:)=[];
                L_kdott{l}(dexK,:)=[];
                
            end
        end

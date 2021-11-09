
%%    collect local parameters  and  sample    Phi
EWSZS_Pi   =   0   ;

for iter   =   1 : (Setting.BurninMB + Setting.CollectionMB)
    %% sample L (CRT)
    L_KK = 0;
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
    %%============= Collection ======================
    if iter > Setting.BurninMB
        if ~Setting.FurCollapse
            EWSZS_Pi    =   EWSZS_Pi + L_KK    ;
        else
            PhiTheta    =   Para.Phi * Para.Theta   ;
            PhiTheta(PhiTheta<=eps)     =   eps     ;
            EWSZS_Pi    =   EWSZS_Pi + (X./PhiTheta)*(Para.Theta.')    ;
        end
        if nnz(isnan(EWSZS_Pi)) | nnz(isinf(EWSZS_Pi))
            warning(['EWSZS Nan',num2str(nnz(isnan(EWSZS_Pi))),'_Inf',num2str(nnz(isinf(EWSZS_Pi)))]);
            EWSZS_Pi(isnan(EWSZS_Pi))   =   0   ;
        end
    end
end

%%=================  Update Ml_Pi  ===============
Setting.PhiUsageCun_Pi   =   Setting.PhiUsageCun_Pi + sum(EWSZS_Pi,1)  ;
if ~Setting.FurCollapse
    EWSZS_Pi    =    EWSZS_Pi/Setting.CollectionMB      ;
else
    EWSZS_Pi   =    (Phi.* EWSZS_Pi/CollectionMB)    ;
end

if(i == 1750)
    aa = 123;
end

if (i == 1)
    Ml_Pi     =   sum(EWSZS_Pi,1)  ;
else
    Ml_Pi     =   (1 - epsipiet(i)) * Ml_Pi + epsipiet(i) * sum(EWSZS_Pi,1)   ;
end
%  RiemannCoefCun(:,i)   =   Ml_Pi(:)  ;

%%=================  Update Global parameters Pi  ===============
for iii     =   1:1
    Piprior = Para.V*Para.V';
    Piprior(logical(eye(size(Piprior))))=0;
    Piprior = Piprior+diag(Para.Xi*Para.V);
    tmp     =   (EWSZS_Pi+Piprior)   ;
    tmp     =   bsxfun(@times, 1./Ml_Phi, tmp - bsxfun(@times, sum(tmp,1), Para.Pi))  ;
    tmp1    =   bsxfun(@times, 2./Ml_Phi, Para.Pi)  ;
    tmp     =   Para.Pi + epsit(i)*tmp + sqrt(epsit(i)*tmp1) .* randn(size(Para.Pi))   ;
    Para.Pi     =   ProjSimplexSpecial(tmp ,Para.Pi,0)  ; % figure(25),DispDictionaryImagesc(Phi{1});drawnow;
end
if nnz(imag(Para.Pi)~=0) | nnz(isnan(Para.Pi)) | nnz(isinf(Para.Pi))
    warning(['Phi Nan',num2str(nnz(isnan(Para.Pi))),'_Inf',num2str(nnz(isinf(Para.Pi)))]);
    Para.Pi(isnan(Para.Pi))   =   0   ;
end





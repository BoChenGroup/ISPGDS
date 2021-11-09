%%  collect local parameters  and  sample    Phi
EWSZS_Phi   =   0   ;

for iter   =   1 : (Setting.BurninMB + Setting.CollectionMB)
    
    [Y_KT,Y_VK] = Multrnd_Matrix_mex_fast_v1(sparse(X_train),Para.Phi,Para.Theta);
    %%============= Collection ======================
    if iter > Setting.BurninMB
        if ~Setting.FurCollapse
            EWSZS_Phi    =   EWSZS_Phi + Y_VK    ;
        else
            PhiTheta    =   Para.Phi * Para.Theta   ;
            PhiTheta(PhiTheta<=eps)     =   eps     ;
            EWSZS_Phi    =   EWSZS_Phi + (X./PhiTheta)*(Para.Theta.')    ;
        end
        if nnz(isnan(EWSZS_Phi)) | nnz(isinf(EWSZS_Phi))
            warning(['EWSZS Nan',num2str(nnz(isnan(EWSZS_Phi))),'_Inf',num2str(nnz(isinf(EWSZS_Phi)))]);
            EWSZS_Phi(isnan(EWSZS_Phi))   =   0   ;
        end
    end
end

%%=================  Update Mk  ===============
Setting.PhiUsageCun_Phi   =   Setting.PhiUsageCun_Phi + sum(EWSZS_Phi,1)  ;
if ~Setting.FurCollapse
    EWSZS_Phi    =    EWSZS_Phi/Setting.CollectionMB      ;
else
    EWSZS_Phi   =   MBratio * (Phi.* EWSZS_Phi/CollectionMB)    ;
end

if(i == 1750)
    aa = 123;
end

if (i == 1)
    Ml_Phi     =   sum(EWSZS_Phi,1)  ;
else
    Ml_Phi     =   (1 - epsipiet(i)) * Ml_Phi + epsipiet(i) * sum(EWSZS_Phi,1)   ;
end
% RiemannCoefCun(:,i)   =   Ml_Phi(:)  ;

%%=================  Update Global parameters Phi  ===============
for iii     =   1:1
    tmp     =   (EWSZS_Phi+Supara.eta0 )   ;
    tmp     =   bsxfun(@times, 1./Ml_Phi, tmp - bsxfun(@times, sum(tmp,1), Para.Phi))  ;
    tmp1    =   bsxfun(@times, 2./Ml_Phi, Para.Phi)  ;
    tmp     =   Para.Phi + epsit(i)*tmp + sqrt(epsit(i)*tmp1) .* randn(size(Para.Phi))   ;
    Para.Phi     =   ProjSimplexSpecial(tmp ,Para.Phi,0)  ; % figure(25),DispDictionaryImagesc(Phi{1});drawnow;
end
if nnz(imag(Para.Phi)~=0) | nnz(isnan(Para.Phi)) | nnz(isinf(Para.Phi))
    warning(['Phi Nan',num2str(nnz(isnan(Para.Phi))),'_Inf',num2str(nnz(isinf(Para.Phi)))]);
    Para.Phi(isnan(Para.Phi))   =   0   ;
end
%Theta = gamrnd(Supara.tao0*Para.Pi*Para.Theta(:,end),1/Supara.tao0);
%X_predict =Para.delta(end) * Para.Phi*Theta;
[~,DEX_k] = sort(  sum(Para.Theta,2),'descend');
figure(3);plot([1:1:T],Para.delta'.*Para.Theta(DEX_k(1),:),'b','LineWidth',2);hold on;
plot([1:1:T],Para.delta'.*Para.Theta(DEX_k(2),:),'g','LineWidth',2);hold on;
plot([1:1:T],Para.delta'.*Para.Theta(DEX_k(3),:),'r','LineWidth',2);hold on;
% plot([1:1:T],Para.delta'.*Para.Theta(DEX_k(4),:),'k','LineWidth',2);hold on;
% plot([1:1:T],Para.delta'.*Para.Theta(DEX_k(5),:),'c','LineWidth',2);hold on;
% plot([1:1:T],Para.delta'.*Para.Theta(DEX_k(6),:),'m','LineWidth',2);hold on;
% plot([1:1:T],Para.delta'.*Para.Theta(DEX_k(7),:),'y','LineWidth',2);hold on;
% plot([1:1:T],Para.delta'.*Para.Theta(DEX_k(8),:),'Color',[0 0.6 0.92],'LineWidth',2);hold on;
% plot([1:1:T],Para.delta'.*Para.Theta(DEX_k(9),:),'Color',[0.2 0.3 0.92],'LineWidth',2);hold on;
% legend(strcat(topic{1}{1},topic{1}{2},topic{1}{3}),...
%     strcat(topic{2}{1},topic{2}{2},topic{2}{3}),...
%     strcat(topic{3}{1},topic{3}{2},topic{3}{3}));

legend('Israel-Palestinian,Iraq-USA,India-Pakistan','Iraq-USA,Iraq-Uk,Turkey-USA','Russian-USA,France-USA,South Korea-USA');
set(gca,'FontSize',20);

topic  = cell(6,1);
for i=1:6
    kk=DEX_k(i);
[~,DEX_v]=sort(Para.Phi(:,kk),'descend');
topic{i,1} = labels_V(DEX_v(1:1:5));
end

%%  python 
T =365;
[~,DEX_k] = sort(  sum(Theta,2),'descend');
figure(1);plot([1:1:T],delta'.*Theta(DEX_k(1),:),'b','LineWidth',2);hold on;
plot([1:1:T],delta'.*Theta(DEX_k(2),:),'g','LineWidth',2);hold on;
plot([1:1:T],delta'.*Theta(DEX_k(3),:),'r','LineWidth',2);hold on;
plot([1:1:T],delta'.*Theta(DEX_k(4),:),'k','LineWidth',2);hold on;
plot([1:1:T],delta'.*Theta(DEX_k(5),:),'c','LineWidth',2);hold on;
% plot([1:1:T],delta'.*Theta(DEX_k(6),:),'m','LineWidth',2);hold on;
% plot([1:1:T],delta'.*Theta(DEX_k(7),:),'y','LineWidth',2);hold on;
% plot([1:1:T],delta'.*Theta(DEX_k(8),:),'Color',[0 0.6 0.92],'LineWidth',2);hold on;
% plot([1:1:T],delta'.*Theta(DEX_k(9),:),'Color',[0.1 0.3 0.92],'LineWidth',2);hold on;
legend(strcat(topic{1}{1},topic{1}{2},topic{1}{3},topic{1}{4}),...
    strcat(topic{2}{1},topic{2}{2},topic{2}{3},topic{2}{4}),...
    strcat(topic{3}{1},topic{3}{2},topic{3}{3},topic{3}{4}),...
    strcat(topic{4}{1},topic{4}{2},topic{4}{3},topic{4}{4}),...
    strcat(topic{5}{1},topic{5}{2},topic{5}{3},topic{5}{4}));

legend('Iraq-USA,Iraq-UK,UK-USA','NK-USA,Japan-USA,China-Japan','Israel-Palestinian,Iraq-USA,India-Pakistan');

set(gca,'FontSize',20);

topic  = cell(5,1);
for i=1:5
    kk=DEX_k(i);
[~,DEX_v]=sort(Phi(:,kk),'descend');
topic{i,1} = labels_V(DEX_v(1:1:5));
end
num_class = 6;
II =1- Para.Pi(DEX_k(1:6),DEX_k(1:6));
figure(1);imagesc(II);
% figure(1);imagesc(II,'InitialMagnification','fit');
textStrings=num2str(II(:),'%0.3f');
textStrings=strtrim(cellstr(textStrings));
[x,y]=meshgrid(1:num_class);
colormap(flipud(gray));%axis image

%hStrings=text(x(:),y(:),textStrings(:),'HorizontalAlignment','center','FontSize',27);
midValue=mean(get(gca,'CLim'));
textColors=repmat(II(:)>midValue,1,6);
%set(hStrings,{'Color'},num2cell(textColors,2));
set(gca,'xticklabel','tic','XAxisLocation','top');
set(gca, 'XTick', 1:num_class, 'YTick', 1:num_class);
set(gca,'xticklabel',{'1','2','3','4','5','6'},'FontSize',20);

set(gca,'yticklabel',{'1','2','3','4','5','6'},'FontSize',20);
rotateXLabels(gca, 315 );% rotate the x tick


II =1- Pi(DEX_k(1:6),DEX_k(1:6));
figure(2);imagesc(II);
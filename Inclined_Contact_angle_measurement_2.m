clear
clc

% ʵ����Ƭ����

%����ȫ����ֵ��


[filename,pathname]=uigetfile({'*bmp';'*.jpg';'*gif'},'ѡ��ԭͼƬ');
g = imread([pathname,filename]);
f = rgb2gray(g);
cout=0;
T = mean2(f);
done = false;
% ��ֵ�����㷨
while ~done
    cout = cout+1;
    g = f>T;
    Ttext = 0.5*(mean(f(g))+mean(f(~g)));% �����µ���ֵ
    done = abs(T-Ttext)<0.5; %�趨�����ı�־
    T = Ttext;% ������ֵ
end
T=120;
g = im2bw(f,T/255);%�Եó�����ֵ���зָ�ͼ��
%imshow(f) %�Ҷ�ͼ
%figure, imhist(f); %�Ҷ�ֱ��ͼ

%figure, imshow(g); %ȫ����ֵ������ͼ��

%������ȡ
contour = bwperim(g);
I_reverse = imcomplement(contour);%��ת�ڰ�
figure
imshow(I_reverse); %��Ե�������ͼ
%title('����')

%% ��������
[wd,len]=size(I_reverse);

%������ͷʵ�ʳ�0.45mm�����㳤�ȱ���
p=0;
j=len-1;
R=[];
while j>1
    if I_reverse(wd-1,j)==0
        p=p+1;
        R=[R j];
    end
    if p==2
        break
    else
        j=j-1;
    end
end

ratio=0.45/33;
Rac=33;

%��ͼ���Ҳ� �ҵ��ڶ������ߵ����꣨����ά��ƽ�棩
x_hor=[];
num=0;
while num<2
    for i=2:1:wd-1
       if I_reverse(i,len-1)==0
           num=num+1;
           x_hor=[x_hor i];
       end
    end
end


%����ά��ƽ���Ҷ�����ʼ������ֱ���ҵ���һ���ٽ�㣨Һ���Ҳ�Ӵ��㣩
p=x_hor(2);
j=len-1;
while j>1
    if I_reverse(p,j)==0
        if I_reverse(p+1,j)~=0
            j=j-1;
        else
            p=p+1;
            break
        end
    elseif I_reverse(p-1,j)==0
        p=p-1;
        j=j-1;
    elseif I_reverse(p+1,j)==0
        if I_reverse(p+2,j)==0
            j=j+1;           
            break
        elseif I_reverse(p+2,j-1)==0
            j=j+1;
            break
        else
            p=p+1;
            j=j-1;
        end
    else
        j=j+1;
        break     
    end
end

% ���ⲿ��case�±�Եȡ����ά�µ�������
while j>1
    if I_reverse(p,j-1)==0
        j=j-1;
    else
        break
    end
end

% u�����ж�
if Rac >= 37
    u=1;
else
    u=3;% ������������ˮƽ������Ϊ��άƽ��
end

% ���ⲿ��case�±�Ե���45�ȣ���С�����Ӵ���

    while j>1
        if I_reverse(p+1,j-1)==0 && I_reverse(p+u,j-1)~=0
            p=p+1;
            j=j-1;
        else
            break
        end
    end


x0_2=j;% �ұ߽Ӵ���x����
y0_2=p;% �ұ߽Ӵ���y����



% ���Ҳ�Ӵ���������Һ�����Ӵ��㣬ע���ų����ܴ��ڵ�����
int=2;
j=x0_2-5;
while j>1
    num=0;
    for i=y0_2-int:1:y0_2+int
        if I_reverse(i,j)==0
            m=i;
            n=j;
            num=num+1;
            break
        end
    end
    if num==0
        j=j-1;
    else
        p=i;
        num=0;
        for i=m-int-1:1:m+int+1
            if I_reverse(i,j-10)==0
                num=num+1;
                break
            end
        end
        if num==1
            j=j-10;
            break
        else
            break
        end
    end
end

p=i;
k=1;
for k=1:1:2
    if I_reverse(p-1,j-1)==0
        p=p-1;
        j=j-1;
    end
end

u=0;

while j<len
    if I_reverse(p,j+1)==0
        j=j+1;
    else
        jj=j;
       while I_reverse(p+1,jj+1)==0  %������жȼ��
         jj=jj+1;
         u=u+1;
       end
        if u>=3
            j=jj;
            p=p+1;
        else
            break
        end
       break
        
    end
end
    


%������жȼ��
%     while j<len
%         if I_reverse(p+1,j+1)==0 &&I_reverse(p+1,j+2)==0 && I_reverse(p+1,j+3)==0 && I_reverse(p+1,j+4)~=0 
%             p=p+1;
%             j=j+3;
%         else
%             break
%         end
%     end


x0_1=j;
y0_1=p;

%ȥ����ά��ƽ��Ĵֲڶȣ�����ֱ��                  
ind=3;
for j=2:1:x0_1-1
    for i=y0_1-ind:1:y0_1+ind
        I_reverse(i,j)=1;
    end
end


for j=x0_2+1:1:len-1
    for i=y0_2-ind:1:y0_2+ind
        I_reverse(i,j)=1;
    end
end

%���ҽӴ�����е�
x0=floor((x0_1+x0_2)/2);
y0=floor((y0_1+y0_2)/2);        


% for j=x0-ind:1:x0+ind
%     for i=y0-ind:1:y0+ind
%         I_reverse(i,j)=0;
%     end
% end

for j=x0_1-1:-1:2
     
    I_reverse(y0_1,j)=0;
   
end


for j=x0_2+1:1:len-1
     
    I_reverse(y0_2,j)=0;
   
end

%�ų���ͷӰ�죬Ѱ��Һ�������
w=20;
i=wd-1;
% while i>y0
%     p=0;
%     for j=len-1:-1:x0_1-w
%         if I_reverse(i,j)==0;
%             p=p+1;
%             break
%         end
%     end
%     if p~=0
%         i=i-1;
%     else 
%         break
%     end
% end


p=0;
while i>y0
    for j=len-1:-1:x0_1-w
        if I_reverse(i,j)==0;
            y_bot=i;
            p=p+1;
            x_bot=j;
            break
        end
    end
    if p~=0
        break
    else 
        i=i-1;
    end
end

%�ҵ�Һ���������ȵ� x_ls
j=x0_1;
while j>1
    p=0;
    for i=y0_1+2:1:y_bot
        if I_reverse(i,j)==0
            p=p+1;
            break
        end
    end
    if p~=0
        j=j-1;
    else
        if j==x0_1
          x_ls=j;
          break
        else
           x_ls=j+1;
           break
        end
    end
     
end

%�ų�Һ�������ͷӰ��
for j=2:1:x_ls-1
    for i=y0_1:1:y_bot
        I_reverse(i,j)=1;
    end
end      

y_up=min(y0_1,y0_2);
y_top=max(y0_1,y0_2);

% �����������Ϊ��㣬���¶��ϼ�¼��������
D=zeros(2,y_bot-y_top+1);
 for i=y_top:1:y_bot
    for j=2:1:x_bot
        if I_reverse(i,j)==0
            D(1,i-y_top+1)=j;
            break
        end
    end
    for j=len-1:-1:x_bot
        if I_reverse(i,j)==0
            D(2,i-y_top+1)=j;
            break
        end
    end
 end

% ȥ��Һ���м����� 
 for i=y_top+40:1:y_bot-1
     for j=D(1,i-y_top+1)+1:1:D(2,i-y_top+1)-1
         if I_reverse(i,j)==0 && min(abs(j-D(1,i-y_top+1)),abs(j-D(2,i-y_top+1)))>10
             I_reverse(i,j)=1;
         end
     end
 end 

x=[];
y=[];
% ����άԲ��Ϊԭ��ת�����꣬������д���Ե����㣨x,y��

 for i=y0_1:1:y_bot
     if i ~= y0_1
        for j=2:1:x_bot
            if I_reverse(i,j) == 0
                y=[y,(y0-i)*ratio];
                x=[x,(j-x0)*ratio];
            end
        end
     else
         y=[y,(y0-i)*ratio];
         x=[x,(x0_1-x0)*ratio];
     end
     
 end

 for i=y_bot:-1:y0_2
       if i ~= y0_2
           for j=x_bot+1:1:len-1
            if I_reverse(i,j) == 0
                y=[y,(y0-i)*ratio];
                x=[x,(j-x0)*ratio];
            end
           end
       else
           y=[y,(y0-i)*ratio];
           x=[x,(x0_2-x0)*ratio];
       end
 end

figure;
scatter(x,y,'.'); %����ͼ

axis equal;

grid on;

D_f=x_hor(2)-x_hor(1)+1;

if D_f>40&D_f<50
    D_f=50;
    else if D_f>50&D_f<65
        D_f=58;
        else if D_f>65&D_f<80
        D_f=74;
            else if D_f>80&D_f<95
        D_f=87;
                end
            end
        end
end

%������ά
len_rec= floor((x0_2-x0_1+1)*3*ratio);
D_f=(D_f)*ratio; %������άֱ��
%��������ά
% rectangle('Position',[-len_rec/2,0,len_rec,D_f],'facecolor',[0.97255 0.75294 0.56863],'edgecolor','none');

 hold on;

%% ���

order=3;
%�ҳ�x,y �������ֵ ������ͼ������

n_p =length(x(:))/4; 

if n_p>40
    n_p = 50;
else n_p = n_p+1;

end

n_p = 40;

x_min=min(x(find(x<0)));
x_max=max(x(find(x>0)));
y_min=min(y(find(x<0)));
y_max=max(y(find(x>0)));

%���
x_l = x(find(x<0));
y_l = y(find(x<0));





x_l_n_p = x_l(3:n_p); %��ȡ��߽Ӵ��Ǹ���n_p����x����

y_l_n_p = y_l(3:n_p); %��ȡ��߽Ӵ��Ǹ���30����y����

% x_l_n_p = (x_l_n_p(1:2:end)+x_l_n_p(2:2:end))/2;%ȥ�������ż����ƽ��ֵ
 
% y_l_n_p = (y_l_n_p(1:2:end)+y_l_n_p(2:2:end))/2;%ȥ�������ż����ƽ��ֵ


% ��������
% 
% x_l_n_p = x_l_n_p(1:2:end);
% 
% y_l_n_p = y_l_n_p(1:2:end);
% 
% cs = spline(y_l_n_p,x_l_n_p); 
% 
% x_l_sp = ppval(cs,y_l_n_p);
% 
% xx = linspace(y_l_n_p(1),y_l_n_p(n_p/5),200);
% 
% yy=ppval(cs,xx);
% 
% plot(yy,xx,'r');

p_l = polyfit(y_l_n_p,x_l_n_p,order);%����ʽ���n_p�����������

y_l_fit = linspace(min(y_l_n_p),max(y_l_n_p));
x_l_fit = polyval(p_l,y_l_fit);

plot(x_l_fit,y_l_fit,'r');  %�����������


%  for i=2:2:n_p
%         for j=2:2:n_p
%             if I_reverse(i,j) == 0
%                 y=[y,(y0-i)*ratio];
%                 x=[x,(j-x0)*ratio];
%             end
%         end            
%  end


% hold on

dif_l = polyder(p_l);% ���������

tan_ca_l = polyval(dif_l,max(y_l_n_p));

ca_l_1 = 90+atand(tan_ca_l);

% ca_r_1 = 90+atand(tan_ca_r) %�ҽӴ���1
% 
% xx1_l = polyval(p_l,max(y_l_n_p));
% 
% xx2_l = polyval(p_l,max(y_l_n_p)-0.1);
% 
% slope_l =0.1/(xx1_l-xx2_l);
% 
% ca_l_2 = atand(slope_l) %��Ӵ������2
% 
% wa_l = atand(abs(x0_1-x0)/abs(y0_1-y0)); %��߽�û��
% 
% cal = 180-ca_l_2-wa_l

%�ұ�
x_r = x(find(x>0));
y_r = y(find(x>0));



x_r_n_p = x_r(length(x_r(:))-n_p-1:length(x_r(:)));

y_r_n_p = y_r(length(y_r(:))-n_p-1:length(y_r(:)));

% x_r_n_p = (x_r_n_p(1:2:end)+x_r_n_p(2:2:end))/2;
% 
% y_r_n_p = (y_r_n_p(1:2:end)+y_r_n_p(2:2:end))/2;

p_r = polyfit(y_r_n_p,x_r_n_p,order);%����ʽ

y_r_fit = linspace(min(y_r_n_p),max(y_r_n_p));

x_r_fit = polyval(p_r,y_r_fit);

 plot(x_r_fit,y_r_fit,'m');

% cs = spline(y_r_n_p,x_r_n_p);
% 
% y_r_sp = linspace(min(y_r_n_p),max(y_r_n_p));
% 
% x_r_sp = ppval(cs,y_r_sp);
% 
% plot(x_r_sp,y_r_sp,'r');

dif_r = polyder(p_r);% ���������

tan_ca_r = polyval(dif_r,max(y_r_n_p));

ca_r_1 = 90-atand(tan_ca_r);

% ca_r_1 = 90+atand(tan_ca_r)
% 
% xx1_r = polyval(p_r,max(y_r_n_p));
% 
% xx2_r = polyval(p_r,max(y_r_n_p)-0.1);
% 
% slope_r =0.1/abs(xx1_r-xx2_r);
% 
% ca_r_2= atand(slope_r) %�ҽӴ���2
% 
% wa_r = atand(abs(x0_2-x0)/abs(y0_2-y0)); %��߽�û��
% 
% car = 180-ca_r_2-wa_r
% 
% angle = [wa_l,cal,wa_r,car]
D_m= x_max-x_min+ratio;
H = (y_bot-y0+1)*ratio;
Wid = (x0_2-x0_1+1)*ratio;

data_exp = [ca_l_1,ca_r_1 Wid D_m H]

hold on 

axis([-3,3,-3.5,1.5]);

grid on;

%%
%%
theta=30; % ��б�ĽǶ�

A=[1,-tan((theta+90)/180*pi)];
B=A/norm(A);
T=zeros(size(I_reverse));

for i=y0_1+3:1:wd-1
    for j=2:1:len-1
        if I_reverse(i,j)==0
            C=[j-x0_1,i-y0_1];
            T(i,j)=dot(B,C);
        end
    end
end

[y_b x_b]=find(T==max(max(T)));  %  ��бʱ��͵��Ӧ������
H=max(max(T))*ratio;    % ��бʱҺ�θ߶�


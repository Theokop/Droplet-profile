clear
clc

% ʵ����Ƭ����

%����ȫ����ֵ��

H=3.0132;
[filename,pathname]=uigetfile({'*.jpg';'*bmp';'*gif'},'ѡ��ԭͼƬ');
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
T=180;
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

ratio=0.45/(R(1)-R(2));
Rac=R(1)-R(2);

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
    u=4;
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


x0_2=j;
y0_2=p;



% ���Ҳ�Ӵ���������Һ�����Ӵ��㣬ע���ų����ܴ��ڵ�����
int=2;
j=x0_2-5;
while j>1
    num=0;
    for i=y0_2-1-int:1:y0_2+int
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
        for i=m-int:1:m+int
            if I_reverse(i,j-10)==0
                num=num+1;
                break
            end
        end
        if num==0
            j=j-10;
        else
            break
        end
    end
end

k=1;
for k=1:1:2
    if I_reverse(p-1,j-1)==0
        p=p-1;
        j=j-1;
    end
end

while j<len
    if I_reverse(p,j+1)==0
        j=j+1;
    else
        break
    end
end


    while j<len
        if I_reverse(p+1,j+1)==0 && I_reverse(p+u,j+1)~=0 
            p=p+1;
            j=j+1;
        else
            break
        end
    end


x0_1=j;
y0_1=p;

%ȥ����ά��ƽ��Ĵֲڶȣ�����ֱ��                  
ind=10;
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

for j=x0-ind:1:x0+ind
    for i=y0-ind:1:y0+ind
        I_reverse(i,j)=1;
    end
end

%�ų���ͷӰ�죬Ѱ��Һ�������
w=50;
i=wd-1;
while i>y0
    p=0;
    for j=len-1:-1:x0_1-w
        if I_reverse(i,j)==0;
            p=p+1;
            break
        end
    end
    if p~=0
        i=i-1;
    else 
        break
    end
end
p=0;
while i>y0
    for j=len-1:-1:x0_1-w
        if I_reverse(i,j)==0;
            y_bot=i;
            p=p+1;
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
    for i=y0_1:1:y_bot
        if I_reverse(i,j)==0
            p=p+1;
            break
        end
    end
    if p~=0
        j=j-1;
    else
        x_ls=j+1;
        break
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
    for j=2:1:x0
        if I_reverse(i,j)==0
            D(1,i-y_top+1)=j;
            break
        end
    end
    for j=len-1:-1:x0
        if I_reverse(i,j)==0
            D(2,i-y_top+1)=j;
            break
        end
    end
 end

% ȥ��Һ���м����� 
 for i=y_top:1:y_bot-1
     for j=D(1,i-y_top+1)+1:1:D(2,i-y_top+1)-1
         if I_reverse(i,j)==0 && min(abs(j-D(1,i-y_top+1)),abs(j-D(2,i-y_top+1)))>10
             I_reverse(i,j)=1;
         end
     end
 end 

x=[];
y=[];
% ����άԲ��Ϊԭ��ת�����꣬������д���Ե����㣨x,y��

 for i=y_up:1:y_bot
        for j=2:1:x0
            if I_reverse(i,j) == 0
                y=[y,(y0-i)*ratio];
                x=[x,(j-x0)*ratio];
            end
        end
 end

 for i=y_bot:-1:y_up
        for j=x0+1:1:len-1
            if I_reverse(i,j) == 0
                y=[y,(y0-i)*ratio];
                x=[x,(j-x0)*ratio];
            end
        end            
 end

figure;
scatter(x,y,'.'); %����ͼ
axis equal;

%������ά
len_rec= floor((x0_2-x0_1)*3*ratio);
wid_rec=(x_hor(2)-x_hor(1))*ratio; %������άֱ��
%��������ά
rectangle('Position',[-len_rec/2,0,len_rec,wid_rec],'facecolor',[0.97255 0.75294 0.56863],'edgecolor','none');

hold on;

%% ���

order=4;
%�ҳ�x,y �������ֵ ������ͼ������

x_min=min(x(find(x<0)));
x_max=max(x(find(x>0)));
y_min=min(y(find(x<0)));
y_max=max(y(find(x>0)));

%���
x_l = x(find(x<0));
y_l = y(find(x<0));

n_p =floor((y_bot-y0)/2); 

x_l_n_p = x_l(1:n_p); %��ȡ��߽Ӵ��Ǹ���n_p����x����

y_l_n_p = y_l(1:n_p); %��ȡ��߽Ӵ��Ǹ���30����y����


p_l = polyfit(y_l_n_p,x_l_n_p,order);%����ʽ���n_p�����������

y_l_fit = linspace(min(y_l_n_p),max(y_l_n_p));
x_l_fit = polyval(p_l,y_l_fit);

plot(x_l_fit,y_l_fit,'r');  %�����������



hold on

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

p_r = polyfit(y_r_n_p,x_r_n_p,order);%����ʽ

y_r_fit = linspace(min(y_r_n_p),max(y_r_n_p));

x_r_fit = polyval(p_r,y_r_fit);

plot(x_r_fit,y_r_fit,'m');
hold on;

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
width= x_max-x_min;
length = (y_bot-y0)*ratio;

angle = [ca_l_1,ca_r_1 width length]

hold on 


%% ģ������
% h=0.4545;  %��û�߶�
% W=1.3717; %��û���
% R2=0.5022; % ������ά�뾶
% D_s = 2.9982; %������Բˮƽ�᳤
% D_m = 3.1032; %������Բˮƽ�᳤
% H=3.0132; %Һ�θ߶�
% ca=140.7239;  %������Ӵ���ĽӴ���
% 
% 
% R1=(W^2+h^2)/(2*h); % ��������Ӵ�����������Բ�뾶 
% rh=R1-h;
% m=100; 
% 
% 
% % �������ϰ���Բ����
% z_t0=h; %������Ӵ�������z
% 
% r_t0=(2*z_t0*R2-z_t0^2)^0.5; %��������r
% 
% k_ca=tan((ca/180)*pi); %����Ӵ�������Բб��
% 
% E0=r_t0-D_s/2;
% 
% F0=z_t0+(H-D_s/2);
% 
% ze0=-H+D_s/2;
% 
% A0=(E0*F0-k_ca*E0^2)/(2*k_ca*E0-F0);
% 
% B0=(-k_ca*F0*A0^2/(E0+A0))^0.5; %�ϰ���Բ����ֱ�����᳤
% 
% re0=D_s/2-A0;
% 
% 
% aza=0:pi/(2*m):pi/2; %��λ��
% 
% for i = 1: 1: m+1
%     az=aza(i);
%     k=tan(az);
%  
%     %��ĳһ����������Ӵ�������
%     
%     if abs(k-1)<10^-9
%         z_t=(R1^2-rh^2)/(2*(R2+rh));
%     else
%         z_t=(-(k^2*R2+rh)+((k^2*R2+rh)^2-(1-k^2)*(rh^2-R1^2))^0.5)/(1-k^2); %�󽻵�����z
%     end
%      x_t=(2*z_t*R2-z_t^2)^0.5; %�󽻵�����x
%      y_t=(R1^2-(z_t+rh)^2)^0.5; %�󽻵�����y
%     if (1/k)<10^-10
%         r_t=abs(y_t/sin(az)); %��������ϵ�µĽ���İ뾶r
%     else
%         r_t=abs(x_t/cos(az));
%     end
%     
%      %��ĳһ����,�ϰ���Բ���̣����ᡢ���ᡢԭ�㣩
%     if i==1     
%      
%         BB=B0;
%         AA=A0;
%         r_t=r_t0;
%         z_t=z_t0;
%         re=re0;
%         ze=ze0;
%         D_az=D_s;
%         
%     else
%         D_az = ((D_s*D_m)/(cos(az)^2*D_m^2+sin(az)^2*D_s^2)^0.5); % ĳһ��λ���µ����뾶
%         BB=B0; %��Բ��ֱ�����᳤����.
%         E1=r_t-D_az/2;
%         F1=(1-(z_t+H-D_s/2)^2/BB^2)^0.5;
%         AA=-E1/(1-F1);
%     end
%     
%     for j =1:1:m
% 
% 
%             H_mid=-H+D_s/2; %���²����ཻ��zֵ
%  
%         
%         %Һ�μ���ά����
%         zp(j)=(z_t-(-H+D_s/2))*(j-1)/m+(-H+D_s/2); %z(j)��  ����
%         r(j)=(1-(zp(j)+H-D_s/2)^2/BB^2)^0.5*AA+D_az/2-AA; %z(j) �µ� �뾶
% 
%      
%         %��ά����
%        zf(j) = (z_t*(j-1)/m);
%       Ac = ((k*R2)^2+R2^2)^0.5;
%        rf(j) = (Ac^2-Ac^2*(zf(j)-R2)^2/R2^2)^0.5;
%  
% 
%     
%         %Һ���ϰ벿���������
%         x(m+2-j,i) =real(r(j)*cos(az));
%         %y(m+2-j,i) =real(r(j)*sin(az));
%         z(m+2-j,i) =real(zp(j));%��һ����
%         x(m+2-j,2*m+3-i)=-real(r(j)*cos(az));
%         %y(m+2-j,2*m+3-i)=real(r(j)*sin(az));
%         z(m+2-j,2*m+3-i) =real(zp(j));%�ڶ�����
%        % x(m+2-j,2*m+2+i)=-real(r(j)*cos(az));
%         %y(m+2-j,2*m+2+i)=-real(r(j)*sin(az));
%         %z(m+2-j,2*m+2+i) =real(zp(j));%��������
%        % x(m+2-j,4*m+5-i)=real(r(j)*cos(az));
%        % y(m+2-j,4*m+5-i)=-real(r(j)*sin(az));
%        % z(m+2-j,4*m+5-i) =real(zp(j));%��������               
%     end
% 
% end
% 
% 
% m=length(x1);
% [l,col]=size(z);
% lig=l-1;
% for i=1:1:lig
%     X(i)=x(i+1,col);
%     Z(i)=z(i+1,1);
%     X(m-lig+i)=x(lig+2-i,1);
%     Z(m-lig+i)=z(lig+2-i,col);
% end
% 
% %Һ���°벿���������
% for i =0: 1: (m-2*lig)
%         X(i+lig+1)=D_s/2*cos(pi/ (m-2*lig)*i-pi);
%         Z(i+lig+1)=D_s/2*sin(pi/ (m-2*lig)*i-pi)+H_mid;
% 
% end
% 
% 
% %��ͼ
% 
% plot(X,Z,'LineWidth',1.5);%Һ������
% 
% 
% 
% 
% hold on;
% xlabel('x(mm)');
% ylabel('z(mm)');
% plot(X,Z,'r','Linewidth',1);
% grid minor;
% axis equal;
% 
% 
% %������ά
% theta=0:pi/m:2*pi;
% xc=R2*cos(theta);
% yc=R2*sin(theta)+R2;
% 
% fill(xc,yc,[0.97255 0.75294 0.56863],'edgecolor','none');
% axis([-2,2,-3.5,1.5]);
% 
% %��������Y1˳��
% f=floor(m/2);
% for i=1:1:f
%     Y1(i)=y1(2*i-1);
%     Y1(m+1-i)=y1(2*i);  
% end
% 
% if f ~= floor(m/2)
%     Y1(f/2+1)=y1(m);
% end


%% ������
%���ж�ϵ��r^2
% fin=sum(Y1.^2);
% add=0;
% s=[];
% for i=1:1:m
%     add = add+Y1(i);    
%     s=[s,Y1(i)-Z(i)];
% end
% qs=sum(s.^2);
% for i=1:1:m
%     qsq(i)=Y1(i)-add/m;
% end
% ssqrt=sum(qsq.^2);
% r2=1-qs/(fin-ssqrt) 
% 
% r3=1-qs/(fin-add^2/m)
clear
clc

% 实验照片轮廓

%基本全局阈值法

H=3.0132;
[filename,pathname]=uigetfile({'*.jpg';'*bmp';'*gif'},'选择原图片');
g = imread([pathname,filename]);
f = rgb2gray(g);
cout=0;
T = mean2(f);
done = false;
% 阈值迭代算法
while ~done
    cout = cout+1;
    g = f>T;
    Ttext = 0.5*(mean(f(g))+mean(f(~g)));% 产生新的阈值
    done = abs(T-Ttext)<0.5; %设定结束的标志
    T = Ttext;% 更新阈值
end
T=180;
g = im2bw(f,T/255);%以得出的阈值进行分割图像。
%imshow(f) %灰度图
%figure, imhist(f); %灰度直方图

%figure, imshow(g); %全局阈值法处理图像

%轮廓提取
contour = bwperim(g);
I_reverse = imcomplement(contour);%反转黑白
figure
imshow(I_reverse); %边缘检测轮廓图
%title('轮廓')

%% 调整轮廓
[wd,len]=size(I_reverse);

%根据针头实际长0.45mm，换算长度比例
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

%在图像右侧 找到第二条横线的坐标（即纤维下平面）
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


%从纤维下平面右端向左开始搜索，直到找到第一个临界点（液滴右侧接触点）
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

% 避免部分case下边缘取到纤维下低面的情况
while j>1
    if I_reverse(p,j-1)==0
        j=j-1;
    else
        break
    end
end

% u是敏感度
if Rac >= 37
    u=1;
else
    u=4;
end

% 避免部分case下边缘点呈45度，减小测量接触角

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



% 从右侧接触点往左找液滴左侧接触点，注意排除可能存在的亮斑
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

%去除纤维下平面的粗糙度，看成直线                  
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

%左右接触点的中点
x0=floor((x0_1+x0_2)/2);
y0=floor((y0_1+y0_2)/2);        

for j=x0-ind:1:x0+ind
    for i=y0-ind:1:y0+ind
        I_reverse(i,j)=1;
    end
end

%排除针头影响，寻找液滴最低面
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

%找到液滴左侧最大宽度点 x_ls
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

%排除液滴左侧针头影响
for j=2:1:x_ls-1
    for i=y0_1:1:y_bot
        I_reverse(i,j)=1;
    end
end      

y_up=min(y0_1,y0_2);
y_top=max(y0_1,y0_2);

% 以最低面中心为起点，自下而上记录两边轮廓
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

% 去除液滴中间亮斑 
 for i=y_top:1:y_bot-1
     for j=D(1,i-y_top+1)+1:1:D(2,i-y_top+1)-1
         if I_reverse(i,j)==0 && min(abs(j-D(1,i-y_top+1)),abs(j-D(2,i-y_top+1)))>10
             I_reverse(i,j)=1;
         end
     end
 end 

x=[];
y=[];
% 以纤维圆心为原点转换坐标，从左到右写入边缘坐标点（x,y）

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
scatter(x,y,'.'); %点线图
axis equal;

%画出纤维
len_rec= floor((x0_2-x0_1)*3*ratio);
wid_rec=(x_hor(2)-x_hor(1))*ratio; %输入纤维直径
%画金属纤维
rectangle('Position',[-len_rec/2,0,len_rec,wid_rec],'facecolor',[0.97255 0.75294 0.56863],'edgecolor','none');

hold on;

%% 拟合

order=4;
%找出x,y 坐标的最值 来调整图的坐标

x_min=min(x(find(x<0)));
x_max=max(x(find(x>0)));
y_min=min(y(find(x<0)));
y_max=max(y(find(x>0)));

%左边
x_l = x(find(x<0));
y_l = y(find(x<0));

n_p =floor((y_bot-y0)/2); 

x_l_n_p = x_l(1:n_p); %提取左边接触角附近n_p个点x坐标

y_l_n_p = y_l(1:n_p); %提取左边接触角附近30个点y坐标


p_l = polyfit(y_l_n_p,x_l_n_p,order);%多项式拟合n_p个点的趋势线

y_l_fit = linspace(min(y_l_n_p),max(y_l_n_p));
x_l_fit = polyval(p_l,y_l_fit);

plot(x_l_fit,y_l_fit,'r');  %画出拟合曲线



hold on

dif_l = polyder(p_l);% 拟合曲线求导

tan_ca_l = polyval(dif_l,max(y_l_n_p));

ca_l_1 = 90+atand(tan_ca_l);

% ca_r_1 = 90+atand(tan_ca_r) %右接触角1
% 
% xx1_l = polyval(p_l,max(y_l_n_p));
% 
% xx2_l = polyval(p_l,max(y_l_n_p)-0.1);
% 
% slope_l =0.1/(xx1_l-xx2_l);
% 
% ca_l_2 = atand(slope_l) %左接触点外角2
% 
% wa_l = atand(abs(x0_1-x0)/abs(y0_1-y0)); %左边浸没角
% 
% cal = 180-ca_l_2-wa_l

%右边
x_r = x(find(x>0));
y_r = y(find(x>0));



x_r_n_p = x_r(length(x_r(:))-n_p-1:length(x_r(:)));

y_r_n_p = y_r(length(y_r(:))-n_p-1:length(y_r(:)));

p_r = polyfit(y_r_n_p,x_r_n_p,order);%多项式

y_r_fit = linspace(min(y_r_n_p),max(y_r_n_p));

x_r_fit = polyval(p_r,y_r_fit);

plot(x_r_fit,y_r_fit,'m');
hold on;

dif_r = polyder(p_r);% 拟合曲线求导

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
% ca_r_2= atand(slope_r) %右接触角2
% 
% wa_r = atand(abs(x0_2-x0)/abs(y0_2-y0)); %左边浸没角
% 
% car = 180-ca_r_2-wa_r
% 
% angle = [wa_l,cal,wa_r,car]
width= x_max-x_min;
length = (y_bot-y0)*ratio;

angle = [ca_l_1,ca_r_1 width length]

hold on 


%% 模型轮廓
% h=0.4545;  %浸没高度
% W=1.3717; %浸没宽度
% R2=0.5022; % 输入纤维半径
% D_s = 2.9982; %侧面椭圆水平轴长
% D_m = 3.1032; %正面椭圆水平轴长
% H=3.0132; %液滴高度
% ca=140.7239;  %在三相接触点的接触角
% 
% 
% R1=(W^2+h^2)/(2*h); % 输入三相接触线正面轮廓圆半径 
% rh=R1-h;
% m=100; 
% 
% 
% % 求侧面的上半椭圆方程
% z_t0=h; %求三相接触点坐标z
% 
% r_t0=(2*z_t0*R2-z_t0^2)^0.5; %交点坐标r
% 
% k_ca=tan((ca/180)*pi); %三相接触点上椭圆斜率
% 
% E0=r_t0-D_s/2;
% 
% F0=z_t0+(H-D_s/2);
% 
% ze0=-H+D_s/2;
% 
% A0=(E0*F0-k_ca*E0^2)/(2*k_ca*E0-F0);
% 
% B0=(-k_ca*F0*A0^2/(E0+A0))^0.5; %上半椭圆的竖直方向轴长
% 
% re0=D_s/2-A0;
% 
% 
% aza=0:pi/(2*m):pi/2; %方位角
% 
% for i = 1: 1: m+1
%     az=aza(i);
%     k=tan(az);
%  
%     %求某一截面下三相接触点坐标
%     
%     if abs(k-1)<10^-9
%         z_t=(R1^2-rh^2)/(2*(R2+rh));
%     else
%         z_t=(-(k^2*R2+rh)+((k^2*R2+rh)^2-(1-k^2)*(rh^2-R1^2))^0.5)/(1-k^2); %求交点坐标z
%     end
%      x_t=(2*z_t*R2-z_t^2)^0.5; %求交点坐标x
%      y_t=(R1^2-(z_t+rh)^2)^0.5; %求交点坐标y
%     if (1/k)<10^-10
%         r_t=abs(y_t/sin(az)); %求柱坐标系下的交点的半径r
%     else
%         r_t=abs(x_t/cos(az));
%     end
%     
%      %求某一截面,上半椭圆方程（长轴、短轴、原点）
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
%         D_az = ((D_s*D_m)/(cos(az)^2*D_m^2+sin(az)^2*D_s^2)^0.5); % 某一方位角下的最大半径
%         BB=B0; %椭圆竖直方向轴长不变.
%         E1=r_t-D_az/2;
%         F1=(1-(z_t+H-D_s/2)^2/BB^2)^0.5;
%         AA=-E1/(1-F1);
%     end
%     
%     for j =1:1:m
% 
% 
%             H_mid=-H+D_s/2; %上下部分相交面z值
%  
%         
%         %液滴加纤维部分
%         zp(j)=(z_t-(-H+D_s/2))*(j-1)/m+(-H+D_s/2); %z(j)点  坐标
%         r(j)=(1-(zp(j)+H-D_s/2)^2/BB^2)^0.5*AA+D_az/2-AA; %z(j) 下的 半径
% 
%      
%         %纤维部分
%        zf(j) = (z_t*(j-1)/m);
%       Ac = ((k*R2)^2+R2^2)^0.5;
%        rf(j) = (Ac^2-Ac^2*(zf(j)-R2)^2/R2^2)^0.5;
%  
% 
%     
%         %液滴上半部分坐标矩阵
%         x(m+2-j,i) =real(r(j)*cos(az));
%         %y(m+2-j,i) =real(r(j)*sin(az));
%         z(m+2-j,i) =real(zp(j));%第一区间
%         x(m+2-j,2*m+3-i)=-real(r(j)*cos(az));
%         %y(m+2-j,2*m+3-i)=real(r(j)*sin(az));
%         z(m+2-j,2*m+3-i) =real(zp(j));%第二区间
%        % x(m+2-j,2*m+2+i)=-real(r(j)*cos(az));
%         %y(m+2-j,2*m+2+i)=-real(r(j)*sin(az));
%         %z(m+2-j,2*m+2+i) =real(zp(j));%第三区间
%        % x(m+2-j,4*m+5-i)=real(r(j)*cos(az));
%        % y(m+2-j,4*m+5-i)=-real(r(j)*sin(az));
%        % z(m+2-j,4*m+5-i) =real(zp(j));%第四区间               
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
% %液滴下半部分坐标矩阵
% for i =0: 1: (m-2*lig)
%         X(i+lig+1)=D_s/2*cos(pi/ (m-2*lig)*i-pi);
%         Z(i+lig+1)=D_s/2*sin(pi/ (m-2*lig)*i-pi)+H_mid;
% 
% end
% 
% 
% %画图
% 
% plot(X,Z,'LineWidth',1.5);%液滴曲面
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
% %金属纤维
% theta=0:pi/m:2*pi;
% xc=R2*cos(theta);
% yc=R2*sin(theta)+R2;
% 
% fill(xc,yc,[0.97255 0.75294 0.56863],'edgecolor','none');
% axis([-2,2,-3.5,1.5]);
% 
% %重新排列Y1顺序
% f=floor(m/2);
% for i=1:1:f
%     Y1(i)=y1(2*i-1);
%     Y1(m+1-i)=y1(2*i);  
% end
% 
% if f ~= floor(m/2)
%     Y1(f/2+1)=y1(m);
% end


%% 误差分析
%求判定系数r^2
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
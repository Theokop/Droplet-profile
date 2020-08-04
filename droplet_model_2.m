clear
clc

h=0.4545;  %浸没高度
W=1.3717; %浸没宽度
R2=0.5022; % 输入纤维半径
D_s = 2.9982; %侧面椭圆水平轴长
D_m = 3.1032; %正面椭圆水平轴长
H=3.0132; %液滴高度
ca=140.7239;  %在三相接触点的接触角


R1=(W^2+h^2)/(2*h); % 输入三相接触线正面轮廓圆半径 
rh=R1-h;
m=100; 

Ve1=0;
Ve0=0;
Vf=0;


%求下半椭球体积

Ve2=2*pi*(D_s/2)^2*(D_m/2)/3;


% 求侧面的上半椭圆方程
z_t0=h; %求三相接触点坐标z

r_t0=(2*z_t0*R2-z_t0^2)^0.5; %交点坐标r

k_ca=tan((ca/180)*pi); %三相接触点上椭圆斜率

E0=r_t0-D_s/2;

F0=z_t0+(H-D_s/2);

ze0=-H+D_s/2;

A0=(E0*F0-k_ca*E0^2)/(2*k_ca*E0-F0);

B0=(-k_ca*F0*A0^2/(E0+A0))^0.5; %上半椭圆的竖直方向轴长

re0=D_s/2-A0;


aza=0:pi/(2*m):pi/2; %方位角

for i = 1:1:m+1
    az=aza(i);
    k=tan(az);
 
    %求某一截面下三相接触点坐标
    
    if abs(k-1)<10^-9
        z_t=(R1^2-rh^2)/(2*(R2+rh));
    else
        z_t=(-(k^2*R2+rh)+((k^2*R2+rh)^2-(1-k^2)*(rh^2-R1^2))^0.5)/(1-k^2); %求交点坐标z
    end
     x_t=(2*z_t*R2-z_t^2)^0.5; %求交点坐标x
     y_t=(R1^2-(z_t+rh)^2)^0.5; %求交点坐标y
    if (1/k)<10^-10
        r_t=abs(y_t/sin(az)); %求柱坐标系下的交点的半径r
    else
        r_t=abs(x_t/cos(az));
    end
    
     %求某一截面,上半椭圆方程（长轴、短轴、原点）
    if i==1     
     
        BB=B0;
        AA=A0;
        r_t=r_t0;
        z_t=z_t0;
        re=re0;
        ze=ze0;
        D_az=D_s;
        
    else
        D_az = ((D_s*D_m)/(cos(az)^2*D_m^2+sin(az)^2*D_s^2)^0.5); % 某一方位角下的最大半径
        BB=B0; %椭圆竖直方向轴长不变.
        E1=r_t-D_az/2;
        F1=(1-(z_t+H-D_s/2)^2/BB^2)^0.5;
        AA=-E1/(1-F1);
    end
    
    for j =1:1:m+1


            H_mid=-H+D_s/2; %上下部分相交面z值
 
        
        %液滴加纤维部分
        zp(j)=(z_t-(-H+D_s/2))*(j-1)/m+(-H+D_s/2); %z(j)点  坐标
        r(j)=(1-(zp(j)+H-D_s/2)^2/BB^2)^0.5*AA+D_az/2-AA; %z(j) 下的 半径
        Ve0 = (pi/(2*m))*r(j)^2*((z_t-(-H+D_s/2))/m)/2+Ve0; %微小扇形体积
     
        %纤维部分
       zf(j) = (z_t*(j-1)/m);
       Ac = ((k*R2)^2+R2^2)^0.5;
       rf(j) = (Ac^2-Ac^2*(zf(j)-R2)^2/R2^2)^0.5;
       Vf = ((pi/(2*m))*rf(j)^2*((z_t)/m)/2)+Vf;
     
        %液滴上半部分坐标矩阵
        x(m+2-j,i) =real(r(j)*cos(az));
        y(m+2-j,i) =real(r(j)*sin(az));
        z(m+2-j,i) =real(zp(j));%第一区间
        x(m+2-j,2*m+3-i)=-real(r(j)*cos(az));
        y(m+2-j,2*m+3-i)=real(r(j)*sin(az));
        z(m+2-j,2*m+3-i) =real(zp(j));%第二区间
        x(m+2-j,2*m+2+i)=-real(r(j)*cos(az));
        y(m+2-j,2*m+2+i)=-real(r(j)*sin(az));
        z(m+2-j,2*m+2+i) =real(zp(j));%第三区间
        x(m+2-j,4*m+5-i)=real(r(j)*cos(az));
        y(m+2-j,4*m+5-i)=-real(r(j)*sin(az));
        z(m+2-j,4*m+5-i) =real(zp(j));%第四区间
                  
    end
    
    Ve1=Ve1+Ve0-Vf;
   Ve0=0;
   Vf=0;
    
end

VS=4*Ve1+Ve2  %液滴总体积

%液滴下半部分坐标矩阵
q=4*(m+1);
for i = m+2: 1: 2*(m+1)
    for j=1:1:q
        x(i,j)=real(D_s/2*cos(2*pi/q*(j-1))*cos(2*pi/(m+1)*(i-1)));
        y(i,j)=real(D_m/2*sin(2*pi/q*(j-1)));
        z(i,j)=real(D_s/2*cos(2*pi/q*(j-1))*sin(2*pi/(m+1)*(i-1))+H_mid);
    end
end



%画图
k=zeros(size(z));
surf(x,y,z,k);%液滴曲面
shading flat;
hold on;

%侧面圆
rou=linspace(0,R2,m);
theta=linspace(0,2*pi,m);
[R,the]=meshgrid(rou,theta);
yc=3*R.^0;
[xc,zc]=pol2cart(the,R);
zc=zc+R2;
color=ones(size(xc));
mesh(xc,yc,zc,color);



[x1,z1,y1]=cylinder(R2,m);%创建以(0,0)为圆心，高度为h，半径为R2的圆柱
p=ones(size(z1));
surf(x1,30*y1-15,z1+R2,p)%重新绘图 
shading flat;
xlabel('x');
ylabel('y');
zlabel('z');
axis equal;
axis([-2,2,-3,3,-4,2]);
%view(0,90);%俯视角
%view(0,0);%侧视角
%view(90,0);%主视角

clear
clc
syms az  R1 R2 j k h real

h=0.314665;  %浸没高度
W=1.03439; %浸没宽度
R1=(W^2+h^2)/(2*h); % 输入三相接触线正面轮廓圆半径 
R2=0.5022; % 输入纤维半径
D_s = 1.7820; %侧面椭圆水平轴长
D_m = 2.0838; %正面椭圆水平轴长
H=1.2129;
rh=R1-h;
V1=0;
V2=0;
VS=0; %液滴总体积
m=100;
aza=0:pi/(2*m):pi/2;

for i = 1: 1: m+1
    az=aza(i);
    k=tan(az);
    
    %求某一截面下三相接触点坐标
    if abs(k-1)<10^-9
        z_t=(R1^2-rh^2)/(2*(R2+rh));
    else
        z_t=(-(k^2*R2+rh)+((k^2*R2+rh)^2-(1-k^2)*(rh^2-R1^2))^0.5)/(1-k^2); %求交点坐标z
    end
    x_t=(2*z_t*R2-z_t^2)^0.5; %求交点坐标x
    y_t=(R1^2-(z_t+rh)^2)^0.5;
    if (1/k)<10^-10
        r_t=abs(y_t/sin(az)); %求柱坐标系下的交点的半径r
    else
        r_t=abs(x_t/cos(az));
    end
  
    
    %求某一截面下椭圆方程（长轴、短轴、原点）
    D_az = ((D_s*D_m)/(cos(az)^2*D_m^2+sin(az)^2*D_s^2)^0.5); % 某一方位角下的最大半径
    AA = D_az/2;  %某一方位角下椭圆水平方向轴长
    BB=(z_t+H)*(1-(1-(r_t/AA)^2)^0.5)/(r_t/AA)^2; %某一方位角下椭圆竖直方向轴长
    
   
    
    %求在az角度下，一个微小角度椭圆片的体积
    
    for j = 1: 1: m+1
        
        %液滴加纤维部分
        zp(j)=(z_t+H)*(j-1)/m-H; %z(j)点  坐标
        r(j)=(AA^2-AA^2*(zp(j)+H-BB)^2/BB^2)^0.5;  %z(j) 下的 半径
        V1 = (pi/(2*m))*r(j)^2*((z_t+H)/m)/2+V1; %微小扇形体积
     
        %纤维部分
       zc(j) = (z_t*(j-1)/m);
       Ac = ((k*R2)^2+R2^2)^0.5;
       rc(j) = (Ac^2-Ac^2*(zc(j)-R2)^2/R2^2)^0.5;
       V2 = ((pi/(2*m))*rc(j)^2*((z_t)/m)/2)+V2;
           
       
       %曲面坐标矩阵
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
 
   VS=VS+ V1-V2;
   V1=0;
   V2=0;
   
end
VS=4*VS

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
axis([-2,2,-3,3,-4,4]);
%view(0,90);%俯视角
%view(0,0);%侧视角
view(90,0);%主视角







clear
clc
syms az  R1 R2 j k h real

h=0.314665;  %��û�߶�
W=1.03439; %��û���
R1=(W^2+h^2)/(2*h); % ��������Ӵ�����������Բ�뾶 
R2=0.5022; % ������ά�뾶
D_s = 1.7820; %������Բˮƽ�᳤
D_m = 2.0838; %������Բˮƽ�᳤
H=1.2129;
rh=R1-h;
V1=0;
V2=0;
VS=0; %Һ�������
m=100;
aza=0:pi/(2*m):pi/2;

for i = 1: 1: m+1
    az=aza(i);
    k=tan(az);
    
    %��ĳһ����������Ӵ�������
    if abs(k-1)<10^-9
        z_t=(R1^2-rh^2)/(2*(R2+rh));
    else
        z_t=(-(k^2*R2+rh)+((k^2*R2+rh)^2-(1-k^2)*(rh^2-R1^2))^0.5)/(1-k^2); %�󽻵�����z
    end
    x_t=(2*z_t*R2-z_t^2)^0.5; %�󽻵�����x
    y_t=(R1^2-(z_t+rh)^2)^0.5;
    if (1/k)<10^-10
        r_t=abs(y_t/sin(az)); %��������ϵ�µĽ���İ뾶r
    else
        r_t=abs(x_t/cos(az));
    end
  
    
    %��ĳһ��������Բ���̣����ᡢ���ᡢԭ�㣩
    D_az = ((D_s*D_m)/(cos(az)^2*D_m^2+sin(az)^2*D_s^2)^0.5); % ĳһ��λ���µ����뾶
    AA = D_az/2;  %ĳһ��λ������Բˮƽ�����᳤
    BB=(z_t+H)*(1-(1-(r_t/AA)^2)^0.5)/(r_t/AA)^2; %ĳһ��λ������Բ��ֱ�����᳤
    
   
    
    %����az�Ƕ��£�һ��΢С�Ƕ���ԲƬ�����
    
    for j = 1: 1: m+1
        
        %Һ�μ���ά����
        zp(j)=(z_t+H)*(j-1)/m-H; %z(j)��  ����
        r(j)=(AA^2-AA^2*(zp(j)+H-BB)^2/BB^2)^0.5;  %z(j) �µ� �뾶
        V1 = (pi/(2*m))*r(j)^2*((z_t+H)/m)/2+V1; %΢С�������
     
        %��ά����
       zc(j) = (z_t*(j-1)/m);
       Ac = ((k*R2)^2+R2^2)^0.5;
       rc(j) = (Ac^2-Ac^2*(zc(j)-R2)^2/R2^2)^0.5;
       V2 = ((pi/(2*m))*rc(j)^2*((z_t)/m)/2)+V2;
           
       
       %�����������
        x(m+2-j,i) =real(r(j)*cos(az));
        y(m+2-j,i) =real(r(j)*sin(az));
        z(m+2-j,i) =real(zp(j));%��һ����
        x(m+2-j,2*m+3-i)=-real(r(j)*cos(az));
        y(m+2-j,2*m+3-i)=real(r(j)*sin(az));
        z(m+2-j,2*m+3-i) =real(zp(j));%�ڶ�����
        x(m+2-j,2*m+2+i)=-real(r(j)*cos(az));
        y(m+2-j,2*m+2+i)=-real(r(j)*sin(az));
        z(m+2-j,2*m+2+i) =real(zp(j));%��������
        x(m+2-j,4*m+5-i)=real(r(j)*cos(az));
        y(m+2-j,4*m+5-i)=-real(r(j)*sin(az));
        z(m+2-j,4*m+5-i) =real(zp(j));%��������
        
        
    end
 
   VS=VS+ V1-V2;
   V1=0;
   V2=0;
   
end
VS=4*VS

%��ͼ
k=zeros(size(z));
surf(x,y,z,k);%Һ������
shading flat;
hold on;

%����Բ
rou=linspace(0,R2,m);
theta=linspace(0,2*pi,m);
[R,the]=meshgrid(rou,theta);
yc=3*R.^0;
[xc,zc]=pol2cart(the,R);
zc=zc+R2;
color=ones(size(xc));
mesh(xc,yc,zc,color);


[x1,z1,y1]=cylinder(R2,m);%������(0,0)ΪԲ�ģ��߶�Ϊh���뾶ΪR2��Բ��
p=ones(size(z1));
surf(x1,30*y1-15,z1+R2,p)%���»�ͼ
shading flat;
xlabel('x');
ylabel('y');
zlabel('z');
axis equal;
axis([-2,2,-3,3,-4,4]);
%view(0,90);%���ӽ�
%view(0,0);%���ӽ�
view(90,0);%���ӽ�







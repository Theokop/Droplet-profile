clear
clc

h=0.4545;  %��û�߶�
W=1.3717; %��û���
R2=0.5022; % ������ά�뾶
D_s = 2.9982; %������Բˮƽ�᳤
D_m = 3.1032; %������Բˮƽ�᳤
H=3.0132; %Һ�θ߶�
ca=140.7239;  %������Ӵ���ĽӴ���


R1=(W^2+h^2)/(2*h); % ��������Ӵ�����������Բ�뾶 
rh=R1-h;
m=100; 

Ve1=0;
Ve0=0;
Vf=0;


%���°��������

Ve2=2*pi*(D_s/2)^2*(D_m/2)/3;


% �������ϰ���Բ����
z_t0=h; %������Ӵ�������z

r_t0=(2*z_t0*R2-z_t0^2)^0.5; %��������r

k_ca=tan((ca/180)*pi); %����Ӵ�������Բб��

E0=r_t0-D_s/2;

F0=z_t0+(H-D_s/2);

ze0=-H+D_s/2;

A0=(E0*F0-k_ca*E0^2)/(2*k_ca*E0-F0);

B0=(-k_ca*F0*A0^2/(E0+A0))^0.5; %�ϰ���Բ����ֱ�����᳤

re0=D_s/2-A0;


aza=0:pi/(2*m):pi/2; %��λ��

for i = 1:1:m+1
    az=aza(i);
    k=tan(az);
 
    %��ĳһ����������Ӵ�������
    
    if abs(k-1)<10^-9
        z_t=(R1^2-rh^2)/(2*(R2+rh));
    else
        z_t=(-(k^2*R2+rh)+((k^2*R2+rh)^2-(1-k^2)*(rh^2-R1^2))^0.5)/(1-k^2); %�󽻵�����z
    end
     x_t=(2*z_t*R2-z_t^2)^0.5; %�󽻵�����x
     y_t=(R1^2-(z_t+rh)^2)^0.5; %�󽻵�����y
    if (1/k)<10^-10
        r_t=abs(y_t/sin(az)); %��������ϵ�µĽ���İ뾶r
    else
        r_t=abs(x_t/cos(az));
    end
    
     %��ĳһ����,�ϰ���Բ���̣����ᡢ���ᡢԭ�㣩
    if i==1     
     
        BB=B0;
        AA=A0;
        r_t=r_t0;
        z_t=z_t0;
        re=re0;
        ze=ze0;
        D_az=D_s;
        
    else
        D_az = ((D_s*D_m)/(cos(az)^2*D_m^2+sin(az)^2*D_s^2)^0.5); % ĳһ��λ���µ����뾶
        BB=B0; %��Բ��ֱ�����᳤����.
        E1=r_t-D_az/2;
        F1=(1-(z_t+H-D_s/2)^2/BB^2)^0.5;
        AA=-E1/(1-F1);
    end
    
    for j =1:1:m+1


            H_mid=-H+D_s/2; %���²����ཻ��zֵ
 
        
        %Һ�μ���ά����
        zp(j)=(z_t-(-H+D_s/2))*(j-1)/m+(-H+D_s/2); %z(j)��  ����
        r(j)=(1-(zp(j)+H-D_s/2)^2/BB^2)^0.5*AA+D_az/2-AA; %z(j) �µ� �뾶
        Ve0 = (pi/(2*m))*r(j)^2*((z_t-(-H+D_s/2))/m)/2+Ve0; %΢С�������
     
        %��ά����
       zf(j) = (z_t*(j-1)/m);
       Ac = ((k*R2)^2+R2^2)^0.5;
       rf(j) = (Ac^2-Ac^2*(zf(j)-R2)^2/R2^2)^0.5;
       Vf = ((pi/(2*m))*rf(j)^2*((z_t)/m)/2)+Vf;
     
        %Һ���ϰ벿���������
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
    
    Ve1=Ve1+Ve0-Vf;
   Ve0=0;
   Vf=0;
    
end

VS=4*Ve1+Ve2  %Һ�������

%Һ���°벿���������
q=4*(m+1);
for i = m+2: 1: 2*(m+1)
    for j=1:1:q
        x(i,j)=real(D_s/2*cos(2*pi/q*(j-1))*cos(2*pi/(m+1)*(i-1)));
        y(i,j)=real(D_m/2*sin(2*pi/q*(j-1)));
        z(i,j)=real(D_s/2*cos(2*pi/q*(j-1))*sin(2*pi/(m+1)*(i-1))+H_mid);
    end
end



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
axis([-2,2,-3,3,-4,2]);
%view(0,90);%���ӽ�
%view(0,0);%���ӽ�
%view(90,0);%���ӽ�

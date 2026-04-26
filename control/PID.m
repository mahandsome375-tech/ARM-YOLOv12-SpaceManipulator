
clear; tic
alpha = 0.01;


a1=0.5; a2=0.5; a=1;
b0=0.5; b1=0.5; b2=0.5;
m0=40; m1=4; m2=3;
I0=6.667; I1=0.333; I2=0.25;
J0=I0; J1=I1; J2=I2;
ad1=3; ad2=1; ad3=1; ad4=20; ad5=4; ad6=1;
thetad=[ad1;ad2;ad3;ad4;ad5;ad6];
ak1=0.8; ak2=1.3; ak3=1.5; thetak=[ak1;ak2;ak3];


qrlist=[]; qlist=[]; errorlist=[0];
xdlist=[]; ydlist=[]; xlist=[]; ylist=[]; taulist=[];
rholist=[]; philogs=[];


rho0=0.3; rhoInf=0.01; delta=5;
k1=0.2;
k2=3;
Kv=6*eye(2);
Kphi=k2*eye(2);
EPS_RHO=1e-4; EPS_E=1e-6;


Lk=0.3*eye(3); Ld=0.1*eye(6);
kp=10.*[3.5,0;0,2.5]; K=10.*[2.8,0;0,1.1];


qr0=0; qr1=pi/6; qr2=pi/3; qr=[qr0;qr1;qr2]; qrlist=[qrlist,qr];
q0=qr0; q1=qr1; q2=qr2; q=[q0;q1;q2]; qlist=[qlist,q];
dt=0.01;


x=0.25; y=0.50; xe=[x;y];
xd=0.2; yd=0.45; xz=[xd;yd];


xddot = 0.15*cos(3*1*dt);  
yddot = -0.15*sin(3*1*dt);  
xzdot = [xddot; yddot];
xrdot = xzdot - alpha*(xe - xz);
exysmclist = [xe - xz];


Jr=[-ak1*cos(qr0),-ak2*cos(qr0+qr1),-ak3*cos(qr0+qr1+qr2);
    -ak1*sin(qr0),-ak2*sin(qr0+qr1),-ak3*sin(qr0+qr1+qr2)];
qrdot=pinv(Jr)*xrdot; qr0dot=qrdot(1); qr1dot=qrdot(2); qr2dot=qrdot(3);
qdot=qrdot; q0dot=qdot(1); q1dot=qdot(2); q2dot=qdot(3);


J=[-ak1*cos(qr0),-ak2*cos(qr0+qr1),-ak3*cos(qr0+qr1+qr2);
   -ak1*sin(qr0),-ak2*sin(qr0+qr1),-ak3*sin(qr0+qr1+qr2)];
Mbb=2*ad1*cos(qr1)+2*ad2*cos(qr2)+2*ad3*cos(qr1+qr2)+ad4;
Mbm=[ad1*cos(qr1)+2*ad2*cos(qr2)+ad3*cos(qr1+qr2)+ad5
     ad2*cos(qr2)+ad3*cos(qr1+qr2)+ad6];
Mmm=[2*ad2*cos(qr2)+ad5, ad2*cos(qr2)+ad6
     ad2*cos(qr2)+ad6, ad6];
M=[Mbb, Mbm'; Mbm, Mmm];
Cbb=-ad1*sin(qr1)*qr1dot-ad2*sin(qr2)*qr2dot-ad3*sin(qr1+qr2)*(qr1dot+qr2dot);
Cbm=[-ad1*sin(qr1)*(qr0dot+qr1dot)-ad2*sin(qr2)*qr2dot-ad3*sin(qr1+qr2)*(qr0dot+qr1dot+qr2dot)
     -(ad2*sin(qr2)+ad3*sin(qr1+qr2))*(qr0dot+qr1dot+qr2dot)];
Cmb=[ad1*sin(qr1)*qr0dot-ad2*sin(qr2)*qr2dot+ad3*sin(qr1+qr2)*qr0dot
     ad2*sin(qr2)*(qr0dot+qr1dot)+ad3*sin(qr1+qr2)*qr0dot];
Cmm=[-ad2*sin(qr2)*qr2dot,-ad2*sin(qr2)*(qr0dot+qr1dot+qr2dot); ad2*sin(qr2)*(qr0dot+qr1),0];
C=[Cbb, Cbm'; Cmb, Cmm];


mm=10; num=0; Nstep=1000;

for i=1:Nstep
    t = i*dt;


    xd = 0.2 + 0.05*sin(3*i*dt);
    yd = 0.4 + 0.05*cos(3*i*dt);
    xz = [xd; yd];
    xdlist = [xdlist; xd];  ydlist = [ydlist; yd];


    xddot  = 0.15*cos(3*i*dt);   yddot  = -0.15*sin(3*i*dt);  xzdot = [xddot; yddot];
    xdddot = -0.45*sin(3*i*dt);  ydddot = -0.45*cos(3*i*dt);  xzzdot= [xdddot; ydddot];


    xrdot = xzdot - alpha*(xe - xz);

    Jr=[-ak1*cos(qr0),-ak2*cos(qr0+qr1),-ak3*cos(qr0+qr1+qr2);
        -ak1*sin(qr0),-ak2*sin(qr0+qr1),-ak3*sin(qr0+qr1+qr2)];
    qrdot = pinv(Jr)*xrdot;  qr0dot=qrdot(1); qr1dot=qrdot(2); qr2dot=qrdot(3);


    X2=[qr;ak1;ak2;ak3;xrdot]';  
    [tt,X2]=ode45(@mysysqr,[(mm-1)*dt,mm*dt],X2(end,:));
    qr=X2(end,1:3)';  qrlist=[qrlist,qr];
    qr0=qr(1); qr1=qr(2); qr2=qr(3);


    J=[-ak1*cos(q0),-ak2*cos(q0+q1),-ak3*cos(q0+q1+q2);
       -ak1*sin(q0),-ak2*sin(q0+q1),-ak3*sin(q0+q1+q2)];
    xedot = J*qdot;


    Jrdot=[ak1*sin(qr0)*qr0dot,   ak2*sin(qr0+qr1)*(qr0dot+qr1dot),...
           ak3*sin(qr0+qr1+qr2)*(qr0dot+qr1dot+qr2dot);
          -ak1*cos(qr0)*qr0dot,  -ak2*cos(qr0+qr1)*(qr0dot+qr1dot),...
          -ak3*cos(qr0+qr1+qr2)*(qr0dot+qr1dot+qr2dot)];
    invJrdot = -pinv(Jr)*Jrdot*pinv(Jr);
    xrdotdot = xzzdot - alpha*(xedot - xddot*[1;1]);
    qrdotdot = pinv(Jr)*xrdotdot + invJrdot*xrdot;
    qr0dotdot=qrdotdot(1); qr1dotdot=qrdotdot(2); qr2dotdot=qrdotdot(3);


    a0=qr0dot; a1=qr1dot; a2=qr2dot;  
    b0=qr0dotdot; b1=qr1dotdot; b2=qr2dotdot;
    y11=-a1*q0dot*sin(q1)-a0*q1dot*sin(q1)+2*b0*cos(q1)+b1*cos(q1);  
    y12=-a2*q0dot*sin(q2)-a2*q1dot*sin(q2)-a0*q2dot*sin(q2)-a1*q2dot*sin(q2)...
        -a2*q2dot*sin(q2)+2*b0*cos(q2)+2*b1*cos(q2)+b2*cos(q2);  
    y13=-a1*q0dot*sin(q1+q2)-a2*q0dot*sin(q1+q2)-a0*q1dot*sin(q1+q2)-...
        a1*q1dot*sin(q1+q2)-a2*q2dot*sin(q1+q2)+2*b0*cos(q1+q2)+b1*cos(q1+q2)+...
        b2*cos(q1+q2)-a2*q1dot*sin(q1+q2)-a0*q2dot*sin(q1+q2)-a1*q2dot*sin(q1+q2);
    y14=b0; y15=b1; y16=b2;
    y21=a0*q0dot*sin(q1)+b0*cos(q1);
    y22=-a2*q0dot*sin(q2)-a2*q1dot*sin(q2)-(a0+a1+a2)*q2dot*sin(q2)+2*b0*cos(q2)+...
        2*b1*cos(q2)+b2*cos(q2);
    y23=a0*q0dot*sin(q1+q2)+b0*cos(q1+q2);  
    y24=0;  y25=b0+b1;  y26=b2;  y31=0;
    y32=(a0+a1)*q0dot*sin(q2)+(a0+a1)*q1dot*sin(q2)+(b0+b1)*cos(q2);
    y33=a0*q0dot*sin(q1+q2)+b0*cos(q1+q2);
    y34=0; y35=0; y36=b0+b1+b2;
    ydd=[y11,y12,y13,y14,y15,y16
         y21,y22,y23,y24,y25,y26
         y31,y32,y33,y34,y35,y36];


    X3=[xe; xedot]'; [tt,X3]=ode45(@mysysx,[(mm-1)*dt,mm*dt],X3(end,:));
    xe=X3(end,1:2)';  x=xe(1); y=xe(2);
    xlist=[xlist;x]; ylist=[ylist;y];
    e=[x;y]-[xd;yd];  exysmclist=[exysmclist,e];


    rho_t=(rho0-rhoInf)*exp(-delta*t)+rhoInf;
    rho_vec=max(rho_t,EPS_RHO)*[1;1];
    e_clip=min(max(e,-0.999*rho_vec+EPS_E),0.999*rho_vec-EPS_E);
    phi = log((rho_vec + e_clip) ./ (rho_vec - e_clip));
    v   = xrdot - k1*phi;
    s   = xedot - v;
    rholist=[rholist; rho_vec(:)']; 
    philogs=[philogs; phi(:)'];


    tau = -J'*(Kphi*phi + Kv*s) + ydd*thetad;
    taulist=[taulist, tau];


    Mbb=2*ad1*cos(q1)+2*ad2*cos(q2)+2*ad3*cos(q1+q2)+ad4;
    Mbm=[ad1*cos(q1)+2*ad2*cos(q2)+ad3*cos(q1+q2)+ad5
         ad2*cos(q2)+ad3*cos(q1+q2)+ad6];
    Mmm=[2*ad2*cos(q2)+ad5, ad2*cos(q2)+ad6
         ad2*cos(q2)+ad6, ad6];
    M=[Mbb,Mbm'; Mbm,Mmm];
    Cbb=-ad1*sin(q1)*q1dot-ad2*sin(q2)*q2dot-ad3*sin(q1+q2)*(q1dot+q2dot);
    Cbm=[-ad1*sin(q1)*(q0dot+q1dot)-ad2*sin(q2)*q2dot-ad3*sin(q1+q2)*(q0dot+q1dot+q2dot)
         -(ad2*sin(q2)+ad3*sin(q1+q2))*(q0dot+q1dot+q2dot)];
    Cmb=[ad1*sin(q1)*q0dot-ad2*sin(q2)*q2dot+ad3*sin(q1+q2)*q0dot
         ad2*sin(q2)*(q0dot+q1dot)+ad3*sin(q1+q2)*q0dot];
    Cmm=[-ad2*sin(q2)*q2dot,-ad2*sin(q2)*(q0dot+q1dot+q2dot); ad2*sin(q2)*(q0dot+q1),0];
    C=[Cbb,Cbm'; Cmb,Cmm];

    qdotdot=inv(M)*(tau - C*qdot);
    X4=[qdot; qdotdot]';   [tt,X4]=ode45(@mysysqq,[(mm-1)*dt,mm*dt],X4(end,:));
    qdot=X4(end,1:3)'; q0dot=qdot(1); q1dot=qdot(2); q2dot=qdot(3);

    X5=[q; qdot]';   [tt,X5]=ode45(@mysysqq,[(mm-1)*dt,mm*dt],X5(end,:));
    q=X5(end,1:3)';  q0=q(1); q1=q(2); q2=q(3);
    qlist=[qlist,q];


    thetakdot = Lk*[-cos(q0)*q0dot, -cos(q0+q1)*q1dot,-cos(q0+q1+q2)*q2dot; ...
                    -sin(q0)*q0dot, -sin(q0+q1)*q1dot,-sin(q0+q1+q2)*q2dot]'*(Kv*s);
    thetaddot = -Ld*ydd'*(qdot - qrdot);
    X1=[thetad; thetak; thetaddot; thetakdot]';  
    [tt,X1]=ode45(@mysystheta,[(mm-1)*dt,mm*dt],X1(end,:));
    thetad=X1(end,1:6)'; thetak=X1(end,7:9)';
    ak1=thetak(1); ak2=thetak(2); ak3=thetak(3);
    ad1=thetad(1); ad2=thetad(2); ad3=thetad(3);
    ad4=thetad(4); ad5=thetad(5); ad6=thetad(6);

    errorlist=[errorlist; norm(e)];
    num=num+1;
end


iter=linspace(0,Nstep*dt,Nstep+1);


figure('Position',[100 100 1200 450]); tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile(1); hold on; grid on; box on; apply_dark(gca);
plot(xdlist,ydlist,'--','LineWidth',2,'Color',[1 0.2 0.2]);
plot(xlist,ylist,'-','LineWidth',2,'Color',[0.2 0.5 1]);
xlabel('x (m)','Color',[.9 .9 .9]); ylabel('y (m)','Color',[.9 .9 .9]);
title('Task-Space Trajectory','Color',[.9 .9 .9]); legend({'Reference','Actual'},'TextColor',[.9 .9 .9],'Location','northwest');

nexttile(2); hold on; grid on; box on; apply_dark(gca);
ex=exysmclist(1,:); ey=exysmclist(2,:);
plot(iter,ex,'-','LineWidth',1.8); plot(iter,ey,'-','LineWidth',1.8);
rho_plot=rholist;
plot(iter,[rho_plot(1,1);rho_plot(:,1)],'--','LineWidth',1.6,'Color',[0.4 1 0.4]);
plot(iter,-[rho_plot(1,1);rho_plot(:,1)],'--','LineWidth',1.6,'Color',[0.4 1 0.4]);
plot(iter,[rho_plot(1,2);rho_plot(:,2)],'--','LineWidth',1.6,'Color',[0.4 1 0.4]);
plot(iter,-[rho_plot(1,2);rho_plot(:,2)],'--','LineWidth',1.6,'Color',[0.4 1 0.4]);
xlabel('t (s)','Color',[.9 .9 .9]); ylabel('Error / Envelope','Color',[.9 .9 .9]);
title('Error and PPC Envelope','Color',[.9 .9 .9]); legend({'e_x','e_y','\rho_x','\rho_y'},'TextColor',[.9 .9 .9],'Location','southwest');

figure('Position',[100 580 1000 350]); hold on; grid on; box on; apply_dark(gca);
r=hypot(xlist,ylist); r_ref=hypot(xdlist,ydlist); rmax=max(r_ref); margin=rmax - r;
plot(iter,[margin(1);margin(:)],'LineWidth',2);
xlabel('t (s)','Color',[.9 .9 .9]); ylabel('r_{max}-r (m)','Color',[.9 .9 .9]);
title('Margin to Tube Limit (>0 is safer)','Color',[.9 .9 .9]);

figure('Position',[1150 100 1000 500]); hold on; grid on; box on; apply_dark(gca);
plot(iter(1:Nstep),taulist(1,:),'LineWidth',1.6);
plot(iter(1:Nstep),taulist(2,:),'LineWidth',1.6);
plot(iter(1:Nstep),taulist(3,:),'LineWidth',1.6);
xlabel('Time (s)','Color',[.9 .9 .9]); ylabel('Torque (N*m)','Color',[.9 .9 .9]);
title('Control Torque','Color',[.9 .9 .9]); legend({'Base Torque','Joint 1 Torque','Joint 2 Torque'},'TextColor',[.9 .9 .9],'Location','northeast');

toc


function ddx=mysysqr(~,x)
qr=x(1:3); qr0=qr(1); qr1=qr(2); qr2=qr(3);
ak1=x(4); ak2=x(5); ak3=x(6); xrdot=x(7:8);
Jr=[-ak1*cos(qr0),-ak2*cos(qr0+qr1),-ak3*cos(qr0+qr1+qr2);
    -ak1*sin(qr0),-ak2*sin(qr0+qr1),-ak3*sin(qr0+qr1+qr2)];
qrdot=pinv(Jr)*xrdot;
ddx=[qrdot;0;0;0;0;0];
end
function ddx=mysysx(~,x)
dx=x(3:4); ddx=[dx;0;0];
end
function ddx=mysysqq(~,x)
dx=x(4:6); ddx=[dx;0;0;0];
end
function ddx=mysystheta(~,x)
thetaddot=x(10:15); thetakdot=x(16:18); ddx=[thetaddot;thetakdot;zeros(9,1)];
end
function apply_dark(ax)
set(ax,'Color',[0.10 0.10 0.10],'XColor',[.85 .85 .85],'YColor',[.85 .85 .85],...
    'GridColor',[.35 .35 .35],'MinorGridColor',[.25 .25 .25]); set(gcf,'Color',[.10 .10 .10]);
ax.LineWidth=1; ax.FontSize=12; grid(ax,'on');
end

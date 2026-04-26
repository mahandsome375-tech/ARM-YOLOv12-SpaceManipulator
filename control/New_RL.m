
clear; clc;
tic

USE_RL     = true;
actor_path = 'E:\ARM_PROJECT\ARM_YOLOv12\ARM_YOLOv12_SpaceManipulator_Code_Data\ARM_YOLOv12_journal_upload\bridge\rl_actor.mat';


beta_rl = 0.05;


alpha = 0.01;


a1 = 0.5; a2 = 0.5; a = 1;
b0 = 0.5; b1 = 0.5; b2 = 0.5;
m0 = 40;  m1 = 4;   m2 = 3;
I0 = 6.667; I1 = 0.333; I2 = 0.25;
J0 = I0;   J1 = I1;     J2 = I2;


ad1 = 3;  ad2 = 1;  ad3 = 1;
ad4 = 20; ad5 = 4;  ad6 = 1;
thetad = [ad1;ad2;ad3;ad4;ad5;ad6];


ak1 = 0.8; ak2 = 1.3; ak3 = 1.5;
thetak = [ak1;ak2;ak3];


qrlist = [];  
qlist  = [];  
errorlist = [0];
xdlist = [];  ydlist = [];  
xlist  = [];  ylist  = [];  
taulist = [];
exysmclist = [];


Lk = 0.3*eye(3);  
Ld = 0.1*eye(6);  


kv = 12*eye(2);
kp = 40.*[3.5,0;0,2.5];
K  = 25.*[2.8,0;0,1.1];


qr0 = 0;    qr1 = pi/6;   qr2 = pi/3;
qr  = [qr0;qr1;qr2];  
qrlist = [qrlist,qr];

q0 = qr0;   q1 = qr1;     q2  = qr2;
q  = [q0;q1;q2];  
qlist = [qlist,q];

dt = 0.01;


x  = 0.25;  y  = 0.50;
xe = [x;y];

xd    = 0.2;   yd    = 0.45;   xz    = [xd;yd];
xddot = 0.15*cos(3*1*dt);  
yddot = -0.15*sin(3*1*dt);
xzdot = [xddot; yddot];

xrdot = xzdot - alpha*(xe - xz);
exysmclist = [xe - xz];


Jr = [-ak1*cos(qr0),-ak2*cos(qr0+qr1),-ak3*cos(qr0+qr1+qr2);
      -ak1*sin(qr0),-ak2*sin(qr0+qr1),-ak3*sin(qr0+qr1+qr2)];
qrdot  = pinv(Jr)*xrdot;  
qr0dot = qrdot(1);  
qr1dot = qrdot(2);  
qr2dot = qrdot(3);

qdot   = qrdot;
q0dot  = qdot(1);  
q1dot  = qdot(2);  
q2dot  = qdot(3);


J =[-ak1*cos(qr0), -ak2*cos(qr0+qr1), -ak3*cos(qr0+qr1+qr2);
    -ak1*sin(qr0), -ak2*sin(qr0+qr1), -ak3*sin(qr0+qr1+qr2)];
Mbb = 2*ad1*cos(qr1)+2*ad2*cos(qr2)+2*ad3*cos(qr1+qr2)+ad4;
Mbm = [ad1*cos(qr1)+2*ad2*cos(qr2)+ad3*cos(qr1+qr2)+ad5
       ad2*cos(qr2)+ad3*cos(qr1+qr2)+ad6];
Mmm = [2*ad2*cos(qr2)+ad5, ad2*cos(qr2)+ad6
       ad2*cos(qr2)+ad6,   ad6];
M = [Mbb, Mbm'; Mbm,Mmm];
Cbb = -ad1*sin(qr1)*qr1dot-ad2*sin(qr2)*qr2dot-ad3*sin(qr1+qr2)*(qr1dot+qr2dot);
Cbm = [-ad1*sin(qr1)*(qr0dot+qr1dot)-ad2*sin(qr2)*qr2dot-ad3*sin(qr1+qr2)*(qr0dot+qr1dot+qr2dot)
       -(ad2*sin(qr2)+ad3*sin(qr1+qr2))*(qr0dot+qr1dot+qr2dot)];
Cmb = [ad1*sin(qr1)*qr0dot-ad2*sin(qr2)*qr2dot+ad3*sin(qr1+qr2)*qr0dot
       ad2*sin(qr2)*(qr0dot+qr1dot)+ad3*sin(qr1+qr2)*qr0dot];
Cmm = [-ad2*sin(qr2)*qr2dot,-ad2*sin(qr2)*(qr0dot+qr1dot+qr2dot);
        ad2*sin(qr2)*(qr0dot+qr1),0];
C = [Cbb,Cbm';Cmb,Cmm];


mm   = 10;
num  = 0;
Nstep = 1000;


for i = 1:Nstep
    t = i*dt;


    xd = 0.2 + 0.05*sin(3*i*dt);  
    yd = 0.4 + 0.05*cos(3*i*dt);  
    xz = [xd; yd];
    xdlist = [xdlist; xd];  
    ydlist = [ydlist; yd];

    xddot  = 0.15*cos(3*i*dt);  
    yddot  = -0.15*sin(3*i*dt);  
    xzdot  = [xddot; yddot];
    xdddot = -0.45*sin(3*i*dt);  
    ydddot = -0.45*cos(3*i*dt);  
    xzzdot = [xdddot; ydddot];


    xrdot = xzdot - alpha*(xe - xz);
    Jr = [-ak1*cos(qr0),-ak2*cos(qr0+qr1),-ak3*cos(qr0+qr1+qr2);
          -ak1*sin(qr0),-ak2*sin(qr0+qr1),-ak3*sin(qr0+qr1+qr2)];
    qrdot  = pinv(Jr)*xrdot;
    qr0dot = qrdot(1);  
    qr1dot = qrdot(2);  
    qr2dot = qrdot(3);


    X2 = [qr; ak1; ak2; ak3; xrdot]';  
    [tt,X2] = ode45(@mysysqr,[(mm-1)*dt,mm*dt],X2(end,:));
    qr   = X2(end,1:3)';      % Update reference joints
    qrlist = [qrlist, qr];
    qr0 = qr(1);  qr1 = qr(2);  qr2 = qr(3);


    J = [-ak1*cos(q0),-ak2*cos(q0+q1),-ak3*cos(q0+q1+q2);
         -ak1*sin(q0),-ak2*sin(q0+q1),-ak3*sin(q0+q1+q2)];
    xedot      = J*qdot;
    xrdotdot   = xzzdot - alpha*(xedot - xddot);

    Jrdot = [ak1*sin(qr0)*qr0dot, ak2*sin(qr0+qr1)*(qr0dot+qr1dot), ...
             ak3*sin(qr0+qr1+qr2)*(qr0dot+qr1dot+qr2dot);
            -ak1*cos(qr0)*qr0dot, -ak2*cos(qr0+qr1)*(qr0dot+qr1dot), ...
            -ak3*cos(qr0+qr1+qr2)*(qr0dot+qr1dot+qr2dot)];
    invJrdot = -pinv(Jr)*Jrdot*pinv(Jr);
    qrdotdot = pinv(Jr)*xrdotdot + invJrdot*xrdot;
    qr0dotdot = qrdotdot(1);  
    qr1dotdot = qrdotdot(2);  
    qr2dotdot = qrdotdot(3);


    a0 = qr0dot;  a1 = qr1dot;  a2 = qr2dot;  
    b0 = qr0dotdot;  b1 = qr1dotdot;  b2 = qr2dotdot;

    y11 = -a1*q0dot*sin(q1)-a0*q1dot*sin(q1)+2*b0*cos(q1)+b1*cos(q1);  
    y12 = -a2*q0dot*sin(q2)-a2*q1dot*sin(q2)-a0*q2dot*sin(q2)-a1*q2dot*sin(q2) ...
          -a2*q2dot*sin(q2)+2*b0*cos(q2)+2*b1*cos(q2)+b2*cos(q2);  
    y13 = -a1*q0dot*sin(q1+q2)-a2*q0dot*sin(q1+q2)-a0*q1dot*sin(q1+q2) ...
          -a1*q1dot*sin(q1+q2)-a2*q2dot*sin(q1+q2) ...
          +2*b0*cos(q1+q2)+b1*cos(q1+q2)+b2*cos(q1+q2) ...
          -a2*q1dot*sin(q1+q2)-a0*q2dot*sin(q1+q2)-a1*q2dot*sin(q1+q2);
    y14 = b0;  y15 = b1;  y16 = b2;

    y21 = a0*q0dot*sin(q1)+b0*cos(q1);
    y22 = -a2*q0dot*sin(q2)-a2*q1dot*sin(q2)-(a0+a1+a2)*q2dot*sin(q2) ...
          +2*b0*cos(q2)+2*b1*cos(q2)+b2*cos(q2);
    y23 = a0*q0dot*sin(q1+q2)+b0*cos(q1+q2);  
    y24 = 0;   y25 = b0 + b1;  y26 = b2;  y31 = 0;
    y32 = (a0+a1)*q0dot*sin(q2)+(a0+a1)*q1dot*sin(q2)+(b0+b1)*cos(q2);
    y33 = a0*q0dot*sin(q1+q2)+b0*cos(q1+q2);
    y34 = 0;   y35 = 0;   y36 = b0+b1+b2;

    ydd = [y11, y12, y13, y14, y15,y16
           y21, y22, y23, y24, y25,y26
           y31, y32, y33, y34, y35,y36];


    X3 = [xe; xedot]';  
    [tt,X3] = ode45(@mysysx,[(mm-1)*dt,mm*dt],X3(end,:));
    xe = X3(end,1:2)'; 
    x  = xe(1);  
    y  = xe(2);

    xlist = [xlist; x];  
    ylist = [ylist; y];

    deltax    = [x; y] - [xd; yd];
    deltadotx = xedot - [xddot; yddot];


    sx      = xedot - xrdot;
    tau_ppc = -J'*(kv*deltadotx + kp*deltax) - J'*K*sx + ydd*thetad;


    if USE_RL

        q1_rl  = q1;
        q2_rl  = q2;
        qd1_rl = q1dot;
        qd2_rl = q2dot;
        x_ref  = xd;
        y_ref  = yd;
        ex_rl  = deltax(1);
        ey_rl  = deltax(2);
        exd_rl = deltadotx(1);
        eyd_rl = deltadotx(2);

        s_env = [q1_rl; q2_rl; qd1_rl; qd2_rl; x_ref; y_ref; ex_rl; ey_rl; exd_rl; eyd_rl];
        s_vis = zeros(4,1);
        s_rl  = [s_env; s_vis];

        tau_rl = policy_infer_rl(s_rl, actor_path);

        tau = tau_ppc + beta_rl * tau_rl;
    else
        tau = tau_ppc;
    end

    taulist = [taulist, tau];


    Mbb = 2*ad1*cos(q1)+2*ad2*cos(q2)+2*ad3*cos(q1+q2)+ad4;
    Mbm = [ad1*cos(q1)+2*ad2*cos(q2)+ad3*cos(q1+q2)+ad5
           ad2*cos(q2)+ad3*cos(q1+q2)+ad6];
    Mmm = [2*ad2*cos(q2)+ad5, ad2*cos(q2)+ad6
           ad2*cos(q2)+ad6,   ad6];
    M = [Mbb, Mbm'; Mbm,Mmm];

    Cbb = -ad1*sin(q1)*q1dot-ad2*sin(q2)*q2dot-ad3*sin(q1+q2)*(q1dot+q2dot);
    Cbm = [-ad1*sin(q1)*(q0dot+q1dot)-ad2*sin(q2)*q2dot-ad3*sin(q1+q2)*(q0dot+q1dot+q2dot)
           -(ad2*sin(q2)+ad3*sin(q1+q2))*(q0dot+q1dot+q2dot)];
    Cmb = [ad1*sin(q1)*q0dot-ad2*sin(q2)*q2dot+ad3*sin(q1+q2)*q0dot
           ad2*sin(q2)*(q0dot+q1dot)+ad3*sin(q1+q2)*q0dot];
    Cmm = [-ad2*sin(q2)*q2dot,-ad2*sin(q2)*(q0dot+q1dot+q2dot);
            ad2*sin(q2)*(q0dot+q1),0];
    C = [Cbb,Cbm';Cmb,Cmm];

    qdotdot = M\(tau - C*qdot);

    X4 = [qdot; qdotdot]';  
    [tt,X4] = ode45(@mysysqq,[(mm-1)*dt,mm*dt],X4(end,:));
    qdot  = X4(end,1:3)'; 
    q0dot = qdot(1);  
    q1dot = qdot(2);  
    q2dot = qdot(3);

    X5 = [q; qdot]';  
    [tt,X5] = ode45(@mysysqq,[(mm-1)*dt,mm*dt],X5(end,:));
    q   = X5(end,1:3)'; 
    q0  = q(1);   
    q1  = q(2);  
    q2  = q(3);

    qlist = [qlist, q];


    thetakdot = Lk*[-cos(q0)*q0dot, -cos(q0+q1)*q1dot,-cos(q0+q1+q2)*q2dot; ...
                    -sin(q0)*q0dot, -sin(q0+q1)*q1dot,-sin(q0+q1+q2)*q2dot]' ...
                    *(kv*deltadotx + kp*deltax);
    s_q = qdot - qrdot;
    thetaddot = -Ld*ydd'*s_q;

    X1 = [thetad; thetak; thetaddot; thetakdot]';  
    [tt,X1] = ode45(@mysystheta,[(mm-1)*dt,mm*dt],X1(end,:));
    thetad = X1(end,1:6)';  
    thetak = X1(end,7:9)';

    ak1 = thetak(1);  ak2 = thetak(2);  ak3 = thetak(3); 
    ad1 = thetad(1);  ad2 = thetad(2);  ad3 = thetad(3);
    ad4 = thetad(4);  ad5 = thetad(5);  ad6 = thetad(6);


    error = norm(xe - xz);
    exysmclist = [exysmclist, xe - xz];
    errorlist  = [errorlist; error];

    num = num + 1;
end


iter = linspace(0, Nstep*dt, Nstep+1);
Tend = iter(end);




figure('Position',[100 100 1200 450]); 
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');


nexttile(1); hold on; grid on; box on;
apply_dark(gca);
plot(xdlist,ydlist,'--','LineWidth',2,'Color',[1 0.2 0.2]);
plot(xlist,ylist,'-','LineWidth',2,'Color',[0.2 0.5 1]);
xlabel('x (m)','Color',[.9 .9 .9]); 
ylabel('y (m)','Color',[.9 .9 .9]);
title('Task-Space Trajectory','Color',[.9 .9 .9]);
legend({'Reference','Actual'},'TextColor',[.9 .9 .9],'Location','northwest');


nexttile(2); hold on; grid on; box on;
apply_dark(gca);
ex = exysmclist(1,:); 
ey = exysmclist(2,:);
plot(iter,ex,'-','LineWidth',1.8); 
plot(iter,ey,'-','LineWidth',1.8);




k_env = 1.5;
lambda_x = 1.2;
lambda_y = 1.2;


N0 = min(10, numel(ex));
rho0x = k_env * max(1e-6, max(abs(ex(1:N0))));
rho0y = k_env * max(1e-6, max(abs(ey(1:N0))));


rho_x = rho0x * exp(-lambda_x * iter);
rho_y = rho0y * exp(-lambda_y * iter);

plot(iter, rho_x,'--','LineWidth',1.6,'Color',[0.4 1 0.4]);
plot(iter,-rho_x,'--','LineWidth',1.6,'Color',[0.4 1 0.4]);
plot(iter, rho_y,'--','LineWidth',1.6,'Color',[0.4 1 0.4]);
plot(iter,-rho_y,'--','LineWidth',1.6,'Color',[0.4 1 0.4]);

xlabel('t (s)','Color',[.9 .9 .9]); 
ylabel('Error / Envelope','Color',[.9 .9 .9]);
title('Error and PPC Envelope','Color',[.9 .9 .9]);
legend({'e_x','e_y','\rho_x','\rho_y'},'TextColor',[.9 .9 .9],...
       'Location','southwest');


figure('Position',[100 580 1000 350]); hold on; grid on; box on;
apply_dark(gca);
r     = hypot(xlist,ylist);
r_ref = hypot(xdlist,ydlist);
rmax  = max(r_ref);
margin = rmax - r;
plot(iter,[margin(1); margin(:)],'LineWidth',2);
xlabel('t (s)','Color',[.9 .9 .9]); 
ylabel('r_{max}-r (m)','Color',[.9 .9 .9]);
title('Margin to Tube Limit (>0 is safer)','Color',[.9 .9 .9]);


figure('Position',[1150 100 1000 500]); hold on; grid on; box on;
apply_dark(gca);
plot(iter(1:Nstep), taulist(1,:), 'LineWidth',1.6);
plot(iter(1:Nstep), taulist(2,:), 'LineWidth',1.6);
plot(iter(1:Nstep), taulist(3,:), 'LineWidth',1.6);
xlabel('Time (s)','Color',[.9 .9 .9]); 
ylabel('Torque (N*m)','Color',[.9 .9 .9]);
title('Control Torque','Color',[.9 .9 .9]);
legend({'Base Torque','Joint 1 Torque','Joint 2 Torque'},...
       'TextColor',[.9 .9 .9],'Location','northeast');

toc


function ddx = mysysqr(~,x)
qr  = x(1:3);  
qr0 = qr(1);  
qr1 = qr(2);  
qr2 = qr(3); 
ak1 = x(4);  
ak2 = x(5);  
ak3 = x(6);  
xrdot = x(7:8);
Jr = [-ak1*cos(qr0),-ak2*cos(qr0+qr1),-ak3*cos(qr0+qr1+qr2);
      -ak1*sin(qr0),-ak2*sin(qr0+qr1),-ak3*sin(qr0+qr1+qr2)];
qrdot = pinv(Jr)*xrdot;
ddx = [qrdot; 0; 0; 0; 0; 0];
end


function ddx = mysysx(~,x)
dx  = x(3:4);  
ddx = [dx; 0; 0];
end


function ddx = mysysqq(~,x)
dx  = x(4:6);  
ddx = [dx; 0; 0; 0];
end


function ddx = mysystheta(~,x)
thetaddot = x(10:15);
thetakdot = x(16:18);   
ddx = [thetaddot; thetakdot; zeros(9,1)];
end


function apply_dark(ax)
set(ax,'Color',[0.10 0.10 0.10],...
       'XColor',[0.85 0.85 0.85],...
       'YColor',[0.85 0.85 0.85],...
       'GridColor',[0.35 0.35 0.35],...
       'MinorGridColor',[0.25 0.25 0.25]);
set(gcf,'Color',[0.10 0.10 0.10]);
ax.LineWidth = 1; 
ax.FontSize  = 12;
grid(ax,'on'); 
end


function tau = policy_infer_rl(s, actor_path)
persistent P
if isempty(P) || ~isfield(P,'loaded') || ~isequal(P.path,actor_path)
    P = struct('loaded',false,'path',actor_path);
    try
        S = load(actor_path);


        if isfield(S,'s') && isstruct(S.s)
            P.s_mean = double(S.s.mean(:));
            P.s_std  = double(S.s.std(:)) + 1e-6;
        elseif isfield(S,'s_mean')
            P.s_mean = double(S.s_mean(:));
            P.s_std  = double(S.s_std(:)) + 1e-6;
        else
            error('rl_actor.mat contains neither S.s nor s_mean / s_std');
        end


        if isfield(S,'net') && isstruct(S.net)
            P.W1 = double(S.net.W1);  P.b1 = double(S.net.b1(:));
            P.W2 = double(S.net.W2);  P.b2 = double(S.net.b2(:));
            P.W3 = double(S.net.W3);  P.b3 = double(S.net.b3(:));
        else
            P.W1 = double(S.W1);      P.b1 = double(S.b1(:));
            P.W2 = double(S.W2);      P.b2 = double(S.b2(:));
            P.W3 = double(S.W3);      P.b3 = double(S.b3(:));
        end


        if isfield(S,'act') && isstruct(S.act)
            P.act_scale = double(S.act.scale(:));
            P.act_bias  = double(S.act.bias(:));
        else
            P.act_scale = double(S.act_scale(:));
            P.act_bias  = double(S.act_bias(:));
        end

        P.loaded = true;
        fprintf('[RL] Successfully loaded weights: %s\n', actor_path);
    catch ME
        warning('[RL] Failed to load weights: %s - using zero torque', ME.message);
        P.loaded = false;
    end
end

if ~P.loaded
    tau = zeros(3,1);
    return;
end

s_vec = double(s(:));
D = numel(P.s_mean);
if numel(s_vec) < D
    s_vec = [s_vec; zeros(D-numel(s_vec),1)];
elseif numel(s_vec) > D
    s_vec = s_vec(1:D);
end

z = (s_vec - P.s_mean) ./ P.s_std;

h1  = tanh(P.W1 * z + P.b1);
h2  = tanh(P.W2 * h1 + P.b2);
pre = P.W3 * h2 + P.b3;
a   = tanh(pre);

tau_rl = P.act_scale .* a + P.act_bias;


if numel(tau_rl) == 2
    tau = [0; tau_rl(:)];
else
    tau = tau_rl(:);
    if numel(tau) < 3
        tau = [tau; zeros(3-numel(tau),1)];
    elseif numel(tau) > 3
        tau = tau(1:3);
    end
end

tau = double(tau);
end

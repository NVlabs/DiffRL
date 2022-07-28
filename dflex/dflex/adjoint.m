clear all

% quat operations and adjoint

syms ax ay az aw bx by bz bw real

% quat multiplication
% q = a*b

q = [aw*bx + bw*ax + ay*bz - by*az;
     aw*by + bw*ay + az*bx - bz*ax;
     aw*bz + bw*az + ax*by - bx*ay;
     aw*bw - ax*bx - ay*by - az*bz]
 
 % Jacobians w.r.t a and b
 dqda = [diff(q, ax), diff(q, ay), diff(q, az), diff(q, aw)]
 dqdb = [diff(q, bx), diff(q, by), diff(q, bz), diff(q, bw)]
 
 syms rx ry rz rw real
 
 % J^T*adj_q
 adj_a = dqda'*[rx; ry; rz; rw]
 adj_b = dqdb'*[rx; ry; rz; rw]
 
 syms px py pz real
 syms qx qy qz qw real
 
 p = [px; py; pz]
 qi = [qx; qy; qz]
 
 % vector rotation
 r = p*(2*qw*qw-1) + cross(qi, p)*qw*2.0 + qi*(qi'*p)*2.0
 
 drdx = [diff(r, px), diff(r, py), diff(r, pz)]
 drdqi = [diff(r, qx), diff(r, qy), diff(r, qz)]
 drdqw = diff(r, qw)
 
 adj_x = drdx'*[rx; ry; rz]
 adj_qi = drdqi'*[rx; ry; rz]
 adj_qw = drdqw'*[rx; ry; rz]
 
 ccode(adj_x, 'File', 'adj_x.c')
 ccode([adj_qi; adj_qw], 'File', 'adj_q.c')
 
 % quat normalize
 q = [qx; qy; qz; qw]
 qhat = q/sqrt(q'*q)
 
 dqhatdq = [diff(qhat, qx), diff(qhat, qy), diff(qhat, qz), diff(qhat, qw)]
 
 adj_q = dqhatdq'*q
 
 % exp map
 syms wx wy wz vx vy vz angle real
 syms adj_tx adj_ty adj_tz real
 
 v = [vx vy vz]'
 w = [wx wy wz]'
 
 t = angle*v + cross(w, v)*(1 - cos(angle)) + cross(w, cross(w, v))*(angle - sin(angle))
 
 dtdw = [diff(t, wx), diff(t, wy), diff(t, wz)]
 dtdv = [diff(t, vx), diff(t, vy), diff(t, vz)]
 
 adj_w = dtdw'*[adj_tx adj_ty adj_tz]'
 adj_v = dtdv'*[adj_tx adj_ty adj_tz]'
 adj_angle = diff(t, angle)'*[adj_tx adj_ty adj_tz]'
 
 adj_exp = [adj_w' adj_v' adj_angle']
 ccode(adj_exp, 'File', 'exp_adj.c')
 
 %ccode(adj_w, 'File', 'exp_adj_w.c')
 %ccode(adj_v, 'File', 'exp_adj_v.c')
 %ccode(adj_angle, 'File', 'exp_adj_angle.c')
 
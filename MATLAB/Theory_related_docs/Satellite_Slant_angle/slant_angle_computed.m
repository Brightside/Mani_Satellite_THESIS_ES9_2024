moon_radius = (1737+50)*1E3; 
moon_period = 27.3*24*3600;
Sat_period = 7200;

Moon_DelRad = ((2*pi)/moon_period)*Sat_period;
Pos_shift = moon_radius*Moon_DelRad*10^-3;


A = 90*pi/180;
b=50; c = Pos_shift;
a = sqrt(b^2 + c^2 - 2*b*c*cos(A));

C = acos((a^2 + b^2 -c^2)/(2*a*b))*180/pi;
C

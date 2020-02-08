from gaussian_plume import *
import numpy as np


rate = 0.72
H = 1.3

xGrid = np.arange(0.1, 500, 5)
yGrid = np.arange(-250, 250, 5)
zGrid = np.arange(0.1, 821, 1)


fcor = coriolis(47)
pSource = pointSource(0, 0, 0, rate, H)
grid = receptorGrid(xGrid, yGrid, zGrid)


U = 5.467
L = -158.815
u_star = 0.385
w_star = 1.058964

x = 100
t = x / U

blh = bl_height(u_star, L, fcor)
h = blh


while h > 0.00001:
    sig_z = sigma_z(h, u_star, w_star, L, blh, fcor)
    sw = sig_z.sigma_w(h, u_star, w_star, blh)
    tsz = sig_z.timescale_z(h, sw, u_star, blh, fcor)
    fz = sig_z.fz(h, t, tsz)
    sz = sig_z.sz(sw, t, fz)

    # back calculate h, but this time use sz/2
    a = (((sz / 2) * (U / (sw * x))) ** -2) - 1
    b = sw / (0.15 * blh)
    z = -(blh / 5) * np.log(1 - ((x / (2 * U * a) * b)))

    print(tsz, z)
    h = z

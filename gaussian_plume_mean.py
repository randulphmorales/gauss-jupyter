#!/usr/bin/env python
# python

import numpy as np


class receptorGrid:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.yMesh, self.zMesh, self.xMesh = np.meshgrid(y, z, x)


class pointSource:
    def __init__(self, x, y, z, rate, H):
        self.x = x
        self.y = y
        self.z = z
        self.rate = rate
        self.H = H
        self.sourceType = "point"


class areaSource:
    def __init__(self, x0, dx, nx, y0, dy, ny, z, rate, H):
        self.x = np.linspace(x0, nx * dx, nx + 1)
        self.y = np.linspace(y0, ny * dy, ny + 1)
        self.z = z
        self.sourceType = 'area'
        self.yMesh, self.zMesh, self.xMesh = np.meshgrid(self.y, z, self.x)
        self.H = H
        self.rate = rate
        self.dx = dx
        self.dy = dy


##### COMPUTE SIGMAS #####

def coriolis(latitude):
    fcor = 2 * 7.29 * 0.00001 * np.sin(np.deg2rad(latitude))

    return fcor


def bl_height(u_star, L, fcor):
    if L < 0:
        first_term = 0.4 * np.sqrt(u_star * 1000 / fcor)
        blh = np.min([first_term, 800]) + 300 * ((2.72) ** (L * 0.01))
    elif L > 0:
        first_term = 0.4 * np.sqrt(u_star * 1000 / fcor)
        blh = np.min([first_term, 800])

    return blh


class sigma_y:
    def __init__(self, zGrid, u_star, L, blh, fcor):
        self.zGrid = zGrid
        self.u_star = u_star
        self.L = L
        self.blh = blh
        self.fcor = fcor

        def sigma_v(zGrid, u_star, blh, fcor):
            if L < 0:
                sv = np.repeat(u_star * (12 - (0.5 * blh / L))
                               ** (1 / 3), len(zGrid))
            elif L == 0:
                sv = np.array(
                    [1.3 * u_star * np.exp(-2 * (fcor * z / u_star)) for z in zGrid])
            elif L > 0:
                sv = np.array(
                    [np.max([(1.3 * u_star * (1 - (z / blh))), 0.2]) for z in zGrid])

            return sv

        def timescale_y(zGrid, sv, u_star, blh, fcor):
            if L < 0:
                tsy = np.array([0.15 * (blh / sigma) for sigma in sv])
            elif L == 0:
                tsy = np.array(
                    [(0.5 * (z / sigma)) / (1 + 15 * (fcor * z / u_star)) for z, sigma in zip(zGrid, sv)])
            elif L > 0:
                tsy = np.array([0.07 * (blh / sigma) * np.sqrt(z / blh)
                                for z, sigma in zip(zGrid, sv)])

            return tsy

        def fy(t, tsy):
            return np.array([(1 + 0.5 * (t / ts)) ** (-0.5) for ts in tsy])

        def sy(sv, t, fy):
            return np.mean(np.array([(sigma * t * f) for sigma, f in zip(sv, fy)]), axis=0)

        self.sigma_v = sigma_v
        self.timescale_y = timescale_y
        self.fy = fy
        self.sy = sy


class sigma_z:
    def __init__(self, zGrid, u_star, w_star, L, blh, fcor):
        self.zGrid = zGrid
        self.u_star = u_star
        self.w_star = w_star
        self.L = L
        self.blh = blh
        self.fcor = fcor

        # compute for sigma_w, timescalez, fz
        def sigma_w(zGrid, u_star, w_star, blh):
            if L < 0:
                sw = np.repeat(0.6 * w_star, len(zGrid))
            elif L >= 0:
                sw = np.array([1.3 * u_star * ((1 - (z / blh))**0.75)
                               for z in zGrid])
            return sw

        def timescale_z(z, sw, u_star, blh, fcor):
            if L < 0:
                tsz = np.array(
                    [0.15 * blh * (1 - np.exp(-5 * (z / blh))) / (sigma) for z, sigma in zip(zGrid, sw)])
            elif L == 0:
                tsz = np.array(
                    [(0.5 * (z / sigma)) / (1 + 15 * (fcor * z / u_star)) for z, sigma in zip(zGrid, sw)])
            elif L > 0:
                tsz = np.array(
                    [0.10 * (blh / sigma) * (z / blh)**0.8 for z, sigma in zip(zGrid, sw)])

            return tsz

        def fz(zGrid, t, tsz):
            if L < 0:
                f_z = np.array([(1 + 0.5 * (t / ts))**(-1 / 2) for ts in tsz])
            elif L >= 0:
                f_z = np.zeros((len(zGrid), len(t)), np.float)
                for i, z in enumerate(zGrid):
                    for j, time in enumerate(t):
                        if z < 50:
                            f_z[i, j] = (1 + 0.9 * (time / 50))**-1
                        elif z >= 50:
                            f_z[i, j] = (
                                1 + 0.945 * ((0.1 * time) ** 0.806)) ** -1
            return np.array(f_z)

        # compute sigma_z

        def sz(sw, t, f_z):
            return np.mean(np.array([sigma * t * fz for sigma, fz in zip(sw, f_z)]), axis=0)

        self.sigma_w = sigma_w
        self.timescale_z = timescale_z
        self.fz = fz
        self.sz = sz


class gaussianPlume:
    def __init__(self, source, grid, sigma_y, sigma_z, blh, fcor, U):
        self.source = source
        self.grid = grid
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.blh = blh
        self.fcor = fcor
        self.U = U

    def calculateConcentration(self):
        # calculate sigma_y
        sv = self.sigma_y.sigma_v(
            self.sigma_y.zGrid, self.sigma_y.u_star, self.blh, self.fcor)
        tsy = self.sigma_y.timescale_y(
            self.sigma_y.zGrid, sv, self.sigma_y.u_star, self.blh, self.fcor)
        fy = self.sigma_y.fy((self.grid.xMesh - self.source.x) / self.U, tsy)

        sy = self.sigma_y.sy(
            sv, (self.grid.xMesh - self.source.x) / self.U, fy)

        # calculate sigma_z
        sw = self.sigma_z.sigma_w(
            self.sigma_z.zGrid, self.sigma_z.u_star, self.sigma_z.w_star, self.blh)
        tsz = self.sigma_z.timescale_z(
            self.sigma_z.zGrid, sw, self.sigma_z.u_star, self.blh, self.fcor)
        f_z = self.sigma_z.fz(
            self.sigma_z.zGrid, (self.grid.xMesh - self.source.x), tsz)
        sz = self.sigma_z.sz(
            sw, (self.grid.xMesh - self.source.x) / self.U, f_z)

        conc = np.zeros_like(self.grid.xMesh, dtype=float)

        # if self.source.sourceType == "area":
        #     for x in self.source.x:
        #         for y in self.source.y:
        #             a = self.source.rate * self.source.dx * self.source.dy / \
        #                 (2 * np.pi * self.U * sigma_y * sigma_z)
        #             b = np.exp(-(self.grid.yMesh - y)**2 / (2 * sigma_y))
        #             c = np.exp(-(self.grid.zMesh - self.source.H) ** 2 / (2 * sigma_z**2)) + \
        #                 np.exp(-(self.grid.zMesh + self.source.H)
        #                        ** 2 / (2 * sigma_z**2))
        #             conc += a * b * c

        if self.source.sourceType == "point":
            y = self.source.y
            a = self.source.rate / (2 * np.pi * self.U * sy * sz)
            b = np.exp(-(self.grid.yMesh - y)**2 / (2 * sy ** 2))
            c = np.exp(-(self.grid.zMesh - self.source.H)**2 / (2 * sz**2)) + \
                np.exp(-(self.grid.zMesh + self.source.H) ** 2 / (2 * sz**2))

            conc += a * b * c

        return conc

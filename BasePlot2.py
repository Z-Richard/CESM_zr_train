import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm, ticker

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature


prj = ccrs.PlateCarree()
res = '110m'

class BasePlot2:
    """
    This is specifically designed for plotting the climatology results from
    (1) Observations
    (2) CESM2-LE and ERA-5, with Random Forest Classifier.
    """
    # Some class attributes useful for plotting
    clabeldict_ = {
        'fontsize': 6, 
        'inline': 1,
        'inline_spacing': 10,
        'fmt': '%i',
        'rightside_up': True,
        'use_clabeltext': True
    }
    
    msl_kwargs_ = dict(
        levels=np.arange(950, 1034, 4),
        colors='black',
        linewidths=1.,
        linestyles='solid',
        transform=prj
    )
    
    gph_kwargs_ = dict(
        levels=np.arange(4200, 6000, 60),
        colors='red',
        linewidths=1.,
        linestyles='dashed', 
        transform=prj, 
        alpha=0.7
    )
    
    cbar_kwargs_ = dict(
        pad=0.02, 
        shrink=0.8, 
        aspect=24, 
        labelsize=8, 
        fontsize=7
    )
    
    def __init__(self, extent, figsize=(8, 8), dpi=120, projection=prj):
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection=projection)
        
        west, east, south, north = extent[0], extent[1], extent[2], extent[3]
        self.ax.set_extent((west, east, south, north), crs=prj)
        
    def get_ax(self):
        return self.ax
        
    def add_basemap(self, overlay=True):
        """
        Add basemap to the plot. 
        """
        border_dict = {'lw': 0.5, 'alpha': 0.5}
        fill_dict = {'lw': 0.5, 'alpha': 0.3}
        if overlay:
            self.ax.add_feature(cfeature.COASTLINE.with_scale(res), **border_dict)
            self.ax.add_feature(cfeature.LAND.with_scale(res), **fill_dict, facecolor='#f5f2e9')
            self.ax.add_feature(cfeature.OCEAN.with_scale(res), lw=0.5, zorder=100, facecolor='#c9d9f0')
            self.ax.add_feature(cfeature.LAKES.with_scale(res), lw=0.5, zorder=100, facecolor='#c9d9f0')
        else:
            self.ax.add_feature(cfeature.COASTLINE.with_scale(res), **border_dict)
            self.ax.add_feature(cfeature.STATES.with_scale(res), **border_dict)
            self.ax.add_feature(cfeature.LAND.with_scale(res), **fill_dict)
            self.ax.add_feature(cfeature.OCEAN.with_scale(res), **fill_dict)
            self.ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.3)
        
        return self.ax
    
    def contour(self, lons, lats, data, clabel=True, clabeldict=None, **kwargs):
        """
        data - should be an `xr.DataArray` with only two dimensions `lat` and `lon`. 
        """
        msl = kwargs.pop('msl', False)
        gph = kwargs.pop('gph', False)
        
        if msl:
            kwargs.update(BasePlot2.msl_kwargs_)
        elif gph:
            kwargs.update(BasePlot2.gph_kwargs_)
            
        cs = self.ax.contour(lons, lats, data, **kwargs)
            
        if clabel:
            if clabeldict is not None:
                self.ax.clabel(cs, **clabeldict)
            else:
                self.ax.clabel(cs, **BasePlot2.clabeldict_)
    
    @staticmethod
    def set_cmap(cdict, interval, listed_color=False):
        import matplotlib.colors as mcolors
        import math
        
        over = cdict.pop('over', None)
        under = cdict.pop('under', None)
        
        keys = list(cdict.keys())
        
        if listed_color:
            bounds = np.arange(keys[0], keys[-1] + 2 * interval, interval)
            print(bounds, cdict.values())
            cmap = mcolors.ListedColormap(cdict.values())
            norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
            if over is not None:
                cmap.set_over(over)
            if under is not None:
                cmap.set_under(under)
            return bounds, norm, cmap

        lastk = -math.inf
        mpl_cdict = dict(red=[], green=[], blue=[], alpha=[])
        norm = mcolors.Normalize(vmin=keys[0], vmax=keys[-1])
        
        for k, v in cdict.items():
            if k < lastk:
                raise ValueError('Incorrect color map definition. The keys for colormaps'
                                 'should increase monotonously.')
            if isinstance(v, str):
                (r1, g1, b1, a1) = (r2, g2, b2, a2) = mcolors.to_rgba(v)
            elif isinstance(v, tuple):
                if len(v) != 2:
                    raise ValueError('If the value for a colormap is a tuple,'
                                     f'it should have a length of 2, while the current length is {len(v)}.')
                r1, g1, b1, a1 = mcolors.to_rgba(v[0])
                r2, g2, b2, a2 = mcolors.to_rgba(v[1])

            mpl_cdict['red'].append([norm(k), r1, r2])
            mpl_cdict['green'].append([norm(k), g1, g2])
            mpl_cdict['blue'].append([norm(k), b1, b2])
            mpl_cdict['alpha'].append([norm(k), a1, a2])

            lastk = k

        span = keys[-1] - keys[0]
        cmap = mcolors.LinearSegmentedColormap('cmap', mpl_cdict, N=span / interval)
        clevs = np.arange(keys[0], keys[-1] + interval, interval)
        return clevs, norm, cmap
    
    def set_cbar(self, norm, cmap, **kwargs):
        """
        Set the colorbar of the plot. 
        """
        labelsize = kwargs.pop('labelsize', 8)
        fontsize = kwargs.pop('fontsize', 7)
        orientation = kwargs.pop('orientation', 'horizontal')
        annotation = kwargs.pop('annotation', '')
        extend = kwargs.pop('extend', 'neither')
        xticklabels = kwargs.pop('xticklabels', None)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = self.fig.colorbar(sm, ax=self.ax, orientation=orientation, extend=extend,
                                 **kwargs)
        cbar.ax.tick_params(labelsize=labelsize)
        
        if xticklabels is not None:
            cbar.ax.set_xticklabels(xticklabels)

        if orientation == 'horizontal':
            cbar.ax.set_xlabel(annotation, fontsize=fontsize)

        if orientation == 'vertical':
            cbar.ax.set_ylabel(annotation, fontsize=fontsize)
    
    def contourf(self, lons, lats, data, cdict, interval, cbar=True, cbardict=None, listed_color=False, **kwargs):
        clevs, norm, cmap = self.set_cmap(cdict, interval, listed_color)
        _ = self.ax.contourf(lons, lats, data, clevs, norm=norm, cmap=cmap, **kwargs)
        
        if cbar:
            cbar_kwargs = BasePlot2.cbar_kwargs_
            
            if cbardict is not None:
                cbar_kwargs.update(cbardict)
            self.set_cbar(norm, cmap, **cbar_kwargs)
            
    def gridlines(self, projection=prj):
        gl = self.ax.gridlines(crs=projection, draw_labels=True,
                               linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlines = False
        gl.ylines = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

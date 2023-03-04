#!/usr/bin/env python3

import pandas as pd
import numpy as np
import iris
import iris.plot as iplt
from iris.cube import Cube
import iris.analysis.maths as iam
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


uncertainty = pd.read_excel("uncertainty-estimation.xlsx")


# Gradient boosting may occasionally predict negative values for a positive dataset. See
# https://datascience.stackexchange.com/questions/565/why-does-gradient-boosting-regression-predict-negative-values-when-there-are-no
# Clip these values values to 0
uncertainty['prec_lgb'] = uncertainty['prec_lgb'].clip(lower=0)
uncertainty['unc_lgb'] = uncertainty['unc_lgb'].clip(lower=0)


def make_cubes(df):
    """ Build Iris cubes containing latitude and longitude as coordinates, and the target variable as value """
    prec = df[['lat', 'lon', 'prec_GPCP_roll']]
    prec_lgb = df[['lat', 'lon', 'prec_lgb']]
    unc = df[['lat', 'lon', 'unc']]
    unc_lgb = df[['lat', 'lon', 'unc_lgb']]

    def make_cube(df, col):
        x_coord = iris.coords.DimCoord(np.unique(df['lat']), standard_name='latitude', units='degrees')
        y_coord = iris.coords.DimCoord(np.unique(df['lon']), standard_name='longitude', units='degrees')

        data = df[col].to_numpy().reshape((28, 24))

        cube = Cube(data, dim_coords_and_dims=[(x_coord, 0), (y_coord, 1)])

        return cube

    # return as_cubes(prec), as_cubes(prec_lgb), as_cubes(unc), as_cubes(unc_lgb)
    return (make_cube(prec, 'prec_GPCP_roll'),
            make_cube(prec_lgb, 'prec_lgb'),
            make_cube(unc, 'unc'),
            make_cube(unc_lgb, 'unc_lgb'))

# Make cubes for each month

dec2018 = uncertainty.query('year == 2018 and month == 12')
dec2018_prec, dec2018_prec_lgb, dec2018_unc, dec2018_unc_lgb = make_cubes(dec2018)

mar2019 = uncertainty.query('year == 2019 and month == 3')
mar2019_prec, mar2019_prec_lgb, mar2019_unc, mar2019_unc_lgb = make_cubes(mar2019)

jun2019 = uncertainty.query('year == 2019 and month == 6')
jun2019_prec, jun2019_prec_lgb, jun2019_unc, jun2019_unc_lgb = make_cubes(jun2019)

sep2019 = uncertainty.query('year == 2019 and month == 9')
sep2019_prec, sep2019_prec_lgb, sep2019_unc, sep2019_unc_lgb = make_cubes(sep2019)


def rename_cubes(prec, prec_lgb, unc, unc_lgb):
    """ Rename cubes, for personal preference only. """
    prec[0].rename("GPCP Precipitation")
    prec_lgb[0].rename("LightGBM Precipitation Prediction")
    unc[0].rename("Measured Uncertainty")
    unc_lgb[0].rename("Estimated Uncertainty")


rename_cubes(dec2018_prec, dec2018_prec_lgb, dec2018_unc, dec2018_unc_lgb)
rename_cubes(mar2019_prec, mar2019_prec_lgb, mar2019_unc, mar2019_unc_lgb)
rename_cubes(jun2019_prec, jun2019_prec_lgb, jun2019_unc, jun2019_unc_lgb)
rename_cubes(sep2019_prec, sep2019_prec_lgb, sep2019_unc, sep2019_unc_lgb)

# Interpolation is not needed with contourf, only with pcolormesh
# This function is left here for historic reasons and will be removed later
# So, the cubes named 'interp' are not actually interpolated


def interp_cube(cube):
    #lat_values = np.linspace(-58.75, 8.75, num=16)
    #lon_values = np.linspace(271.25, 328.75, num=16)
    #interp_points = [('lat', lat_values), ('lon', lon_values)]
    #return cube[0].interpolate(interp_points, iris.analysis.Linear())
    return cube

dec2018_prec_interp, dec2018_prec_lgb_interp = interp_cube(dec2018_prec), interp_cube(dec2018_prec_lgb)
mar2019_prec_interp, mar2019_prec_lgb_interp = interp_cube(mar2019_prec), interp_cube(mar2019_prec_lgb)
jun2019_prec_interp, jun2019_prec_lgb_interp = interp_cube(jun2019_prec), interp_cube(jun2019_prec_lgb)
sep2019_prec_interp, sep2019_prec_lgb_interp = interp_cube(sep2019_prec), interp_cube(sep2019_prec_lgb)

dec2018_unc_interp, dec2018_unc_lgb_interp = interp_cube(dec2018_unc), interp_cube(dec2018_unc_lgb)
mar2019_unc_interp, mar2019_unc_lgb_interp = interp_cube(mar2019_unc), interp_cube(mar2019_unc_lgb)
jun2019_unc_interp, jun2019_unc_lgb_interp = interp_cube(jun2019_unc), interp_cube(jun2019_unc_lgb)
sep2019_unc_interp, sep2019_unc_lgb_interp = interp_cube(sep2019_unc), interp_cube(sep2019_unc_lgb)

# Load the cube containing the predictions of AGCM for comparison
agcm = iris.load("agcm.nc")

# Build a list of all the cubes containing the minimum and maximum values for all the target variables.
# This is done to adjust the colorbar range in the maps.

prec_list = [agcm[0].data, # January
             agcm[1].data, # April
             agcm[2].data, # July
             agcm[3].data, # October
             dec2018_prec_interp.data,
             dec2018_prec_lgb_interp.data,
             mar2019_prec_interp.data,
             mar2019_prec_lgb_interp.data,
             jun2019_prec_interp.data,
             jun2019_prec_lgb_interp.data,
             sep2019_prec_interp.data,
             sep2019_prec_lgb_interp.data]

unc_list = [dec2018_unc_interp.data,
            dec2018_unc_lgb_interp.data,
            mar2019_unc_interp.data,
            mar2019_unc_lgb_interp.data,
            jun2019_unc_interp.data,
            jun2019_unc_lgb_interp.data,
            sep2019_unc_interp.data,
            sep2019_unc_lgb_interp.data]

cmin_prec = [np.min(p) for p in prec_list]
cmax_prec = [np.max(p) for p in prec_list]

cmin_unc = [np.min(u) for u in unc_list]
cmax_unc = [np.max(u) for u in unc_list]

vmin_prec = np.min(cmin_prec)
vmax_prec = np.max(cmax_prec)

vmin_unc = np.min(cmin_unc)
vmax_unc = np.max(cmax_unc)

argvmin_prec = np.argmin(cmin_prec)
argvmax_prec = np.argmax(cmax_prec)

argvmin_unc = np.argmin(cmin_unc)
argvmax_unc = np.argmax(cmax_unc)


# In[68]:


# Finally display all the maps

fig, axes = plt.subplots(nrows=2*5, ncols=2,
                         subplot_kw={'projection': ccrs.PlateCarree()},
                         figsize=(12, 12*5))

def make_axis(ax, cube, title, letter):
    ax.set_title(title)
    ax.text(0.5, -0.05, letter, transform=ax.transAxes, size=12)
    ax.add_feature(cfeature.STATES.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.COASTLINE)
    if 'Precipitation' in title:
        return iplt.contourf(cube, cmap='GnBu', axes=ax, levels=20, vmin=vmin_prec, vmax=vmax_prec)
    elif 'Uncertainty' in title:
        return iplt.contourf(cube, cmap='GnBu', axes=ax, levels=20, vmin=vmin_unc, vmax=vmax_unc)
    else:
        pass

ax_agcm_jan = axes[0][0]
ax_agcm_apr = axes[0][1]
ax_agcm_jul = axes[1][0]
ax_agcm_oct = axes[1][1]
ax1 = axes[2][0]
ax2 = axes[2][1]
ax3 = axes[3][0]
ax4 = axes[3][1]
ax5 = axes[4][0]
ax6 = axes[4][1]
ax7 = axes[5][0]
ax8 = axes[5][1]
ax9 = axes[6][0]
ax10 = axes[6][1]
ax11 = axes[7][0]
ax12 = axes[7][1]
ax13 = axes[8][0]
ax14 = axes[8][1]
ax15 = axes[9][0]
ax16 = axes[9][1]

cf_agcm_jan = make_axis(ax_agcm_jan, agcm[0], 'INPE/AGCM Precipitation (January 2019)', '(a)')
cf_agcm_apr = make_axis(ax_agcm_apr, agcm[1], 'INPE/AGCM Precipitation (April 2019)', '(b)')
cf_agcm_jul = make_axis(ax_agcm_jul, agcm[2], 'INPE/AGCM Precipitation (July 2019)', '(c)')
cf_agcm_oct = make_axis(ax_agcm_oct, agcm[3], 'INPE/AGCM Precipitation (October 2019)', '(d)')
cf1 = make_axis(ax1, dec2018_prec_interp, 'GPCP Precipitation (January 2019)', '(a)')
cf2 = make_axis(ax2, dec2018_prec_lgb_interp, 'LightGBM Precipitation Prediction (January 2019)', '(b)')
cf3 = make_axis(ax3, dec2018_unc_interp, 'Measured Uncertainty (January 2019)', '(c)')
cf4 = make_axis(ax4, dec2018_unc_lgb_interp, 'LightGBM Uncertainty Prediction (January 2019)', '(d)')
cf5 = make_axis(ax5, mar2019_prec_interp, 'GPCP Precipitation (April 2019)', '(a)')
cf6 = make_axis(ax6, mar2019_prec_lgb_interp, 'LightGBM Precipitation Prediction (April 2019)', '(b)')
cf7 = make_axis(ax7, mar2019_unc_interp, 'Measured Uncertainty (April 2019)', '(c)')
cf8 = make_axis(ax8, mar2019_unc_lgb_interp, 'LightGBM Uncertainty Prediction (April 2019)', '(d)')
cf9 = make_axis(ax9, jun2019_prec_interp, 'GPCP Precipitation (July 2019)', '(a)')
cf10 = make_axis(ax10, jun2019_prec_lgb_interp, 'LightGBM Precipitation Prediction (July 2019)', '(b)')
cf11 = make_axis(ax11, jun2019_unc_interp, 'Measured Uncertainty (July 2019)', '(c)')
cf12 = make_axis(ax12, jun2019_unc_lgb_interp, 'LightGBM Uncertainty Prediction (July 2019)', '(d)')
cf13 = make_axis(ax13, sep2019_prec_interp, 'GPCP Precipitation (October 2019)', '(a)')
cf14 = make_axis(ax14, sep2019_prec_lgb_interp, 'LightGBM Precipitation Prediction (October 2019)', '(b)')
cf15 = make_axis(ax15, sep2019_unc_interp, 'Measured Uncertainty (October 2019)', '(c)')
cf16 = make_axis(ax16, sep2019_unc_lgb_interp, 'LightGBM Uncertainty Prediction (October 2019)', '(d)')

cf_prec_list = [cf_agcm_jan, cf_agcm_apr, cf_agcm_jul, cf_agcm_oct, cf1, cf2, cf5, cf6, cf9, cf10, cf13, cf14]
cf_unc_list = [cf3, cf4, cf7, cf8, cf11, cf12, cf15, cf16]

plt.colorbar(cf_prec_list[argvmax_prec], ax=[ax_agcm_jan, ax_agcm_apr, ax_agcm_jul, ax_agcm_oct])
plt.colorbar(cf_prec_list[argvmax_prec], ax=[ax1, ax2])
plt.colorbar(cf_unc_list[argvmax_unc], ax=[ax3, ax4])
plt.colorbar(cf_prec_list[argvmax_prec], ax=[ax5, ax6])
plt.colorbar(cf_unc_list[argvmax_unc], ax=[ax7, ax8])
plt.colorbar(cf_prec_list[argvmax_prec], ax=[ax9, ax10])
plt.colorbar(cf_unc_list[argvmax_unc], ax=[ax11, ax12])
plt.colorbar(cf_prec_list[argvmax_prec], ax=[ax13, ax14])
plt.colorbar(cf_unc_list[argvmax_unc], ax=[ax15, ax16])

# Show the plot
plt.savefig("results/maps-full.png", bbox_inches='tight')
# plt.show()


# In[69]:


# Build a list of all the cubes containing the minimum and maximum values for all the target variables.
# This is done to adjust the colorbar range in the maps.

agcm_error_jan = iam.abs(dec2018_prec_interp - agcm[0])
agcm_error_apr = iam.abs(mar2019_prec_interp - agcm[1])
agcm_error_jul = iam.abs(jun2019_prec_interp - agcm[2])
agcm_error_oct = iam.abs(sep2019_prec_interp - agcm[3])
lgb_error_jan = iam.abs(dec2018_prec_interp - dec2018_prec_lgb_interp)
lgb_error_apr = iam.abs(mar2019_prec_interp - mar2019_prec_lgb_interp)
lgb_error_jul = iam.abs(jun2019_prec_interp - jun2019_prec_lgb_interp)
lgb_error_oct = iam.abs(sep2019_prec_interp - sep2019_prec_lgb_interp)

prec_error_list = [agcm_error_jan.data,
                   agcm_error_apr.data,
                   agcm_error_jul.data,
                   agcm_error_oct.data,
                   lgb_error_jan.data,
                   lgb_error_apr.data,
                   lgb_error_jul.data,
                   lgb_error_oct.data]

cmin_error_prec = [np.min(p) for p in prec_error_list]
cmax_error_prec = [np.max(p) for p in prec_error_list]

vmin_error_prec = np.min(cmin_error_prec)
vmax_error_prec = np.max(cmax_error_prec)

argvmin_error_prec = np.argmin(cmin_error_prec)
argvmax_error_prec = np.argmax(cmax_error_prec)


# In[70]:


# Display precipitation error maps

fig_error, axes_error = plt.subplots(nrows=2*2, ncols=2,
                                     subplot_kw={'projection': ccrs.PlateCarree()},
                                     figsize=(12, 12*2))

def make_axis(ax, cube, title, letter):
    ax.set_title(title)
    ax.text(0.5, -0.05, letter, transform=ax.transAxes, size=12)
    ax.add_feature(cfeature.STATES.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.COASTLINE)
    return iplt.contourf(cube, cmap='Oranges', axes=ax, vmin=vmin_error_prec, vmax=vmax_error_prec)


ax_agcm_jan = axes_error[0][0]
ax_agcm_apr = axes_error[0][1]
ax_agcm_jul = axes_error[1][0]
ax_agcm_oct = axes_error[1][1]
ax1_error_lgb = axes_error[2][0]
ax2_error_lgb = axes_error[2][1]
ax3_error_lgb = axes_error[3][0]
ax4_error_lgb = axes_error[3][1]

cf_agcm_jan = make_axis(ax_agcm_jan, agcm_error_jan, 'INPE/AGCM Precipitation Error (January 2019)', '(a)')
cf_agcm_apr = make_axis(ax_agcm_apr, agcm_error_apr, 'INPE/AGCM Precipitation Error (April 2019)', '(b)')
cf_agcm_jul = make_axis(ax_agcm_jul, agcm_error_jul, 'INPE/AGCM Precipitation Error (July 2019)', '(c)')
cf_agcm_oct = make_axis(ax_agcm_oct, agcm_error_oct, 'INPE/AGCM Precipitation Error (October 2019)', '(d)')
cf1_error_lgb = make_axis(ax1_error_lgb, lgb_error_jan, 'LightGBM Precipitation Error (January 2019)', '(a)')
cf2_error_lgb = make_axis(ax2_error_lgb, lgb_error_apr, 'LightGBM Precipitation Error (April 2019)', '(b)')
cf3_error_lgb = make_axis(ax3_error_lgb, lgb_error_jul, 'LightGBM Precipitation Error (July 2019)', '(c)')
cf4_error_lgb = make_axis(ax4_error_lgb, lgb_error_jan, 'LightGBM Precipitation Error (October 2019)', '(d)')

cf_error_list = [cf_agcm_jan, cf_agcm_apr, cf_agcm_jul, cf_agcm_oct, cf1_error_lgb, cf2_error_lgb, cf3_error_lgb, cf4_error_lgb]

plt.colorbar(cf_error_list[argvmax_error_prec], ax=[ax_agcm_jan, ax_agcm_apr, ax_agcm_jul, ax_agcm_oct])
plt.colorbar(cf_error_list[argvmax_error_prec], ax=[ax1_error_lgb, ax2_error_lgb, ax3_error_lgb, ax4_error_lgb])

# Show the plot
plt.savefig("results/error-maps-full.png", bbox_inches='tight')


# In[71]:


def make_error_map(cube1, cube2, cube3, cube4, cmap, month):
    """ Build the error maps. """
    cmin = [cube1.data.min(), cube2.data.min(), cube3.data.min(), cube4.data.min()]
    cmax = [cube1.data.max(), cube2.data.max(), cube3.data.max(), cube4.data.max()]

    vmin = np.min(cmin)
    vmax = np.max(cmax)

    argvmin = np.argmin(cmin)
    argvmax = np.argmax(cmax)

    fig, axes = plt.subplots(nrows=2, ncols=2,
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             figsize=(12, 12))

    def make_axis(ax, cube, month, letter):
        ax.set_title(f"Uncertainty error ({month})")
        ax.text(0.5, -0.05, letter, transform=ax.transAxes, size=12)
        ax.add_feature(cfeature.STATES.with_scale('50m'))
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.COASTLINE)
        return iplt.contourf(cube, cmap=cmap, axes=ax, vmin=vmin, vmax=vmax, levels=np.linspace(vmin, vmax, 15))

    ax1 = axes[0][0]
    ax2 = axes[0][1]
    ax3 = axes[1][0]
    ax4 = axes[1][1]

    cf1 = make_axis(ax1, cube1, 'January', '(a)')
    cf2 = make_axis(ax2, cube2, 'April', '(b)')
    cf3 = make_axis(ax3, cube3, 'July', '(c)')
    cf4 = make_axis(ax4, cube4, 'October', '(d)')

    cf_list = [cf1, cf2, cf3, cf4]

    plt.colorbar(cf_list[argvmax], ax=[ax1, ax2, ax3, ax4])

    # Show the plot
    plt.savefig("Uncertainty error.png", bbox_inches='tight')


make_error_map(iam.abs(dec2018_unc_interp - dec2018_unc_lgb_interp),
               iam.abs(mar2019_unc_interp - mar2019_unc_lgb_interp),
               iam.abs(jun2019_unc_interp - jun2019_unc_lgb_interp),
               iam.abs(sep2019_unc_interp - sep2019_unc_lgb_interp),
               "Oranges", "Uncertainty")

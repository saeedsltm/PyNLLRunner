#!/usr/bin/python

import os, sys
from numpy import array, arange, arccos, pi, sqrt, linalg ,arctan2
from numpy import zeros, append, genfromtxt, savetxt, loadtxt, degrees
from numpy.linalg import eigh
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from glob import glob
from initial_mpl import init_plotting
from PyMyFunc import d2k, k2d

"""
Script for running NLLOC package.


ChangeLogs:

16-Aug-2017 > Initial.

"""
                     
#___________________ MAIN CLASS

class main():

    #_________ SET INITIAL PARAMETERS

    def __init__(self):

        self.nlloc_par = os.path.join('par','nlloc.dat')
        self.loc_par   = os.path.join('par','loc.dat')
        self.run_eng   = raw_input('\n\n+++ Which module to run:\n\n1- NLLoc [default]\n2- NLDiffLoc\n\n')

        if not self.run_eng.strip() or self.run_eng=='1': self.run_eng = 'NLLOC'
        elif self.run_eng=='2': self.run_eng = 'NLDIFFLOC'
        else:
            print '\n+++ Wrong choice!\n'
            sys.exit(0)

        self.read_nlloc_par()
        self.read_loc_par()

        self.p_res_max   = self.loc_dic['P_RES_MAX']
        self.s_res_max   = self.loc_dic['S_RES_MAX']
        self.max_dist    = self.loc_dic['MAX_DIST']
        self.dep_max     = self.loc_dic['DEP_MAX']
        self.Herr_max    = self.loc_dic['HERR_MAX']
        self.Zerr_max    = self.loc_dic['ZERR_MAX']
        self.rms_max     = self.loc_dic['RMS_MAX']
        self.minds_max   = self.loc_dic['MINDS_MAX']
        self.dep_hist_bw = self.loc_dic['DEP_HIST_BW']
        self.her_hist_bw = self.loc_dic['HER_HIST_BW']
        self.zer_hist_bw = self.loc_dic['ZER_HIST_BW']
        self.rms_hist_bw = self.loc_dic['RMS_HIST_BW']
        
    #_________ READ NLLOC PARAMETERS FILE

    def read_nlloc_par(self):

        tmp            = loadtxt(self.nlloc_par, comments='#', delimiter='=', dtype=str)
        self.nlloc_dic = {}

        for i in tmp:

            self.nlloc_dic[i[0].strip()] = i[1].strip()

    #_________ READ LOCATION PARAMETERS FILE (USED FOR PLOTTING)

    def read_loc_par(self):

        tmp            = loadtxt(self.loc_par, comments='#', delimiter='=', dtype=str)
        self.loc_dic = {}

        for i in tmp:

            self.loc_dic[i[0].strip()] = float(i[1].strip())

        self.lat_min = self.loc_dic['LAT_MIN']
        self.lat_max = self.loc_dic['LAT_MAX']
        self.lon_min = self.loc_dic['LON_MIN']
        self.lon_max = self.loc_dic['LON_MAX']
        self.dep_max = self.loc_dic['DEP_MAX']
        self.nlloc_dic['LOC_ENGINE'] = self.run_eng

    def prepare_nldiffloc_files(self):

        if self.nlloc_dic['LOC_ENGINE']=='NLDIFFLOC':

            cmd = 'Loc2ddct loc/%s.*.*.grid0.loc nlloc %.1f %.1f %.1f'%(self.nlloc_dic['LOCFILES_OUTPUT'],
                                                                        float(self.nlloc_dic['DLOC_MAX_EVNT_DIS']),
                                                                        float(self.nlloc_dic['DLOC_MIN_ARR_WT']),
                                                                        float(self.nlloc_dic['DLOC_MAX_STA_DIS']))
            os.system(cmd)
            os.rename('nlloc.hyp', os.path.join('inp','nlloc.hyp'))
            os.rename('nlloc.ct', os.path.join('inp','nlloc.ct'))

    #_________ CALCULATE EIGENVALUE/EIGENVECTORS        

    def eigsorted(self, cov):
        
        vals, vecs = eigh(cov)
        order      = vals.argsort()[::-1]

        return vals[order], vecs[:,order]
        
    #_________ WRITE NLLOC CONTROL FILE        

    def write_nlloc_cf(self, P_flag=True, S_flag=False, sta_cor=False):

        self.read_nlloc_par()
        self.read_loc_par()

        self.prepare_nldiffloc_files()

        self.nlloc_cf = open('nlloc.cf', 'w')

        self.nlloc_cf.write('#__________________START GENERIC CONTROL STATEMENTS\n\n')
        self.nlloc_cf.write('CONTROL 1 54321\n')
        self.nlloc_cf.write('TRANS  LAMBERT  WGS-84  %s %s %s %s 0.0\n\n'%(self.nlloc_dic['TRANS_LAT'],
                                                                           self.nlloc_dic['TRANS_LON'],
                                                                           self.nlloc_dic['TRANS_LAT_MIN'],
                                                                           self.nlloc_dic['TRANS_LAT_MAX']))
        self.nlloc_cf.write('#__________________END\n')
        self.nlloc_cf.write('#__________________START VEL2GRID STATEMENTS\n\n')
        self.nlloc_cf.write('VGOUT  ./model/layer\n')
        self.nlloc_cf.write('VGTYPE P\n')
        self.nlloc_cf.write('VGTYPE S\n')
        self.nlloc_cf.write('VGGRID  2 %s %s %s %s %s %s SLOW_LEN\n'%(self.nlloc_dic['VGGRID_NUM_G_N_XY'],
                                                                      self.nlloc_dic['VGGRID_NUM_G_N_Z'],
                                                                      self.nlloc_dic['VGGRID_GRID_X'],
                                                                      self.nlloc_dic['VGGRID_GRID_Y'],
                                                                      self.nlloc_dic['VGGRID_GRID_Z'],
                                                                      self.nlloc_dic['VGGRID_G_S_XYZ']))
                
        self.nlloc_cf.write('INCLUDE %s\n\n'%(self.nlloc_dic['VGGRID_VEL_INP']))
        self.nlloc_cf.write('#__________________END\n')
        self.nlloc_cf.write('#__________________START GRID2TIME STATEMENTS\n\n')

        if P_flag:

            self.nlloc_cf.write('GTFILES  ./model/layer  ./time/layer P\n')
        
        if S_flag:

            self.nlloc_cf.write('GTFILES  ./model/layer  ./time/layer S\n')

        self.nlloc_cf.write('GTMODE GRID2D ANGLES_YES\n')
        self.nlloc_cf.write('INCLUDE %s\n'%(self.nlloc_dic['VGGRID_STA_INP']))
        self.nlloc_cf.write('GT_PLFD  1.0e-3  0\n\n')
        self.nlloc_cf.write('#__________________END\n')
        self.nlloc_cf.write('#__________________START NLDIFFLOC STATEMENTS\n\n')
        self.nlloc_cf.write('DLOC_HYPFILE  inp/nlloc.hyp  NLLOC_SUM  -1 -1\n')
        self.nlloc_cf.write('DLOC_SEARCH MET %d %d %d %.3f %.1f %.1f %.2f\n\n'%(float(self.nlloc_dic['DLOC_NumSamples']),
                                                                                float(self.nlloc_dic['DLOC_BeginSave']),
                                                                                float(self.nlloc_dic['DLOC_NumSkip']),
                                                                                float(self.nlloc_dic['DLOC_Step']),
                                                                                float(self.nlloc_dic['DLOC_Velocity']),
                                                                                float(self.nlloc_dic['DLOC_InitialTemp']),
                                                                                float(self.nlloc_dic['DLOC_MaxStep'])))
        self.nlloc_cf.write('#__________________END\n')        
        self.nlloc_cf.write('#__________________START NLDIFFLOC STATEMENTS\n\n')
        self.nlloc_cf.write('LOCSIG %s \n'%(self.nlloc_dic['LOCSIG']))
        self.nlloc_cf.write('LOCCOM %s \n'%(self.nlloc_dic['LOCCOM']))

        if self.nlloc_dic['LOC_ENGINE']=='NLLOC':

            self.nlloc_cf.write('LOCFILES %s %s time/layer loc/%s\n'%(self.nlloc_dic['LOCFILES_OBS'],
                                                                      self.nlloc_dic['LOCFILES_TYP'],
                                                                      self.nlloc_dic['LOCFILES_OUTPUT']))
        elif self.nlloc_dic['LOC_ENGINE']=='NLDIFFLOC':

            self.nlloc_cf.write('LOCFILES inp/nlloc.ct HYPODD_ time/layer loc/%s\n'%(self.nlloc_dic['LOCFILES_OUTPUT']))

            if not os.path.exists(os.path.join('inp','nlloc.ct')):

                print '\n+++ Not "inp/nlloc.ct" file was found!\n'
                sys.exit(0)
            
        output = ['SAVE_NLLOC_ALL','SAVE_NLLOC_SUM','SAVE_HYPO71_SUM']
        output.append(' '.join(output))

        self.nlloc_cf.write('LOCHYPOUT %s\n'%(output[int(self.nlloc_dic['LOCHYPOUT'])-1]))
               
        self.nlloc_cf.write('LOCSEARCH OCT %s %s %s \n'%(self.nlloc_dic['LOCSEARCH_MIN_XYZ'],
                                                         self.nlloc_dic['LOCSEARCH_NODE'],
                                                         self.nlloc_dic['LOCSEARCH_STP']))

        self.nlloc_cf.write('LOCGRID %s %s %s %s %s PROB_DENSITY SAVE\n'%(self.nlloc_dic['LOCGRID_GNUM_XYZ'],
                                                                          self.nlloc_dic['LOCGRID_GRID_X'],
                                                                          self.nlloc_dic['LOCGRID_GRID_Y'],
                                                                          self.nlloc_dic['LOCGRID_GRID_Z'],
                                                                          self.nlloc_dic['LOCGRID_G_S_XYZ']))
       
        if self.nlloc_dic['LOC_ENGINE']=='NLLOC':

            self.nlloc_cf.write('LOCMETH EDT_OT_WT %s %s %s %s %s %s %s %s\n'%(self.nlloc_dic['LOCMETH_MAX_ST_D'],
                                                                               self.nlloc_dic['LOCMETH_MIN_NM_PH'],
                                                                               self.nlloc_dic['LOCMETH_MAX_NM_PH'],
                                                                               self.nlloc_dic['LOCMETH_MIN_NM_S'],
                                                                               self.nlloc_dic['LOCMETH_VP_VS'],
                                                                               self.nlloc_dic['LOCMETH_MAX_G'],
                                                                               self.nlloc_dic['LOCMETH_MIN_ST_D'],
                                                                               self.nlloc_dic['LOCMETH_DUP']))

        elif self.nlloc_dic['LOC_ENGINE']=='NLDIFFLOC':

            self.nlloc_cf.write('LOCMETH L1_NORM 9999.0 3 -1 -1 1.8 6 -1.0 0\n')

        self.nlloc_cf.write('LOCGAU %s\n'%(self.nlloc_dic['LOCGAU']))
        self.nlloc_cf.write('LLOCGAU2 %s\n'%(self.nlloc_dic['LOCGAU2']))
        self.nlloc_cf.write('LOCPHASEID P %s\n'%(self.nlloc_dic['LOCPHASEID_P']))
        self.nlloc_cf.write('LOCPHASEID S %s\n'%(self.nlloc_dic['LOCPHASEID_S']))
        self.nlloc_cf.write('LOCQUAL2ERR 0.1 0.5 1.0 2.0 99999.9\n')
        self.nlloc_cf.write('LOCPHSTAT %s %s %s %s %s %s %s %s\n'%(self.nlloc_dic['LOCPHSTAT_RMS_Max'],
                                                                   self.nlloc_dic['LOCPHSTAT_NR_Min'],
                                                                   self.nlloc_dic['LOCPHSTAT_Gap_Max'],
                                                                   self.nlloc_dic['LOCPHSTAT_P_RMax'],
                                                                   self.nlloc_dic['LOCPHSTAT_S_RMax'],
                                                                   self.nlloc_dic['LOCPHSTAT_EL3_Max'],
                                                                   self.nlloc_dic['LOCPHSTAT_D_Min'],
                                                                   self.nlloc_dic['LOCPHSTAT_D_Max']))

        self.nlloc_cf.write('LOCANGLES ANGLES_YES 5\n')
        self.nlloc_cf.write('LOCMAG ML_HB 1.0 1.110 0.00189\n')

        if sta_cor:
            
            self.nlloc_cf.write('INCLUDE loc/last.stat_totcorr\n')
            
        self.nlloc_cf.write('\n#__________________END')
            
        self.nlloc_cf.close()


    #_________ CHECK REQUIRED DIRECTORIES BEFORE RUNNING NLLOC

    def check_dir(self):

        if self.run_eng=='NLLOC':

            dirs = ['loc','model','time']

            ans = raw_input('\n+++ Remove old location results [y] or not [n]:\n\n')

            for _ in dirs:

                if not os.path.exists(_):

                    os.mkdir(_)

                elif ans.lower() == 'y':

                    for d in dirs:

                        for f in glob(os.path.join(d,'*')):

                            os.remove(f)

                else:

                    for d in ['model','time']:

                        for f in glob(os.path.join(d,'*')):

                            os.remove(f)

            if not os.path.exists('figs'):

                os.mkdir('figs')
                                 

    #_________ RUN NLLOC 
                    
    def run_nlloc(self, sta_cor=False):

        self.sta_cor = sta_cor

        if not os.path.exists('inp/model.dat') or not os.path.exists('inp/station.dat'):

            print '\n+++ Required files "model.dat" or "station.dat" not found!\n\n'

            sys.exit(0)

        self.write_nlloc_cf(P_flag=True, S_flag=False) 
        os.system('Vel2Grid nlloc.cf > /dev/null')
        os.system('Grid2Time nlloc.cf > /dev/null')
        self.write_nlloc_cf(P_flag=False, S_flag=True)
        os.system('Grid2Time nlloc.cf > /dev/null')

        if self.nlloc_dic['LOC_ENGINE']=='NLDIFFLOC':

            print '\n+++ Running NLDiffLoc ...\n'
            self.write_nlloc_cf(P_flag=True, S_flag=False, sta_cor=self.sta_cor)
            os.system('NLDiffLoc nlloc.cf > /dev/null')
            
        elif sta_cor:

            print '\n+++ Running NLLoc [station correction = Yes] ...\n'
            self.write_nlloc_cf(P_flag=True, S_flag=False, sta_cor=self.sta_cor)
            os.system('NLLoc nlloc.cf > /dev/null')

        else:

            print '\n+++ Running NLLoc [station correction = No] ...\n'
            os.system('NLLoc nlloc.cf > /dev/null')

    
    #_________ EXTRACT NLLOC HYPO

    def extract_nlloc_hyp(self, root_name):

        root_name = self.nlloc_dic['LOCFILES_OUTPUT']
        loc_files = glob(os.path.join('loc','%s.*.*.grid0.loc.hyp'%root_name))
        out_evt   = open('%s_event.dat'%root_name, 'w')
        out_pha   = open('%s_phase.dat'%root_name, 'w')

        with open(out_evt.name, 'a') as f, open(out_pha.name, 'a') as g:

            hdr_f = '#   LON     LAT     DEP     RMS     GAP     ERH     ERZ     CXX     CXY     CXZ     CYY     CYZ     CZZ  USD_ST  USD_PH  MIN_DS  MAX_DS  AVG_DS'
            hdr_g = '#   STA     PHA     TT-CAL     RES     WGT    DIST      AZ' 

            f.write('%s\n'%hdr_f)
            g.write('%s\n'%hdr_g)

            for loc_file in loc_files:

                evt_res = zeros(0)
                
                with open(loc_file) as inp:

                    for line in inp:

                        l       = line.split()
                        pha_res = zeros(0)

                        if 'GEOGRAPHIC' in l:

                            evt_res = append(evt_res, array([float(l[11]), float(l[9]), float(l[13])]))

                        if 'QUALITY' in l:

                            evt_res = append(evt_res, array([float(l[8]), float(l[12])]))

                        if 'STATISTICS' in l:

                            evt_res = append(evt_res, array([sqrt(float(l[8])+float(l[14])), sqrt(float(l[18]))]))
                            evt_res = append(evt_res, array([float(l[8]), float(l[10]), float(l[12]),
                                                             float(l[14]), float(l[16]), float(l[18])]))

                        if 'QML_OriginQuality' in l:

                            evt_res = append(evt_res, array([float(l[8]), float(l[4]), float(l[20]),
                                                             float(l[22]), float(l[24])]))

                        if '>' in l and 'PHASE' not in l:

                            if 'P' in l[-23].strip().upper(): pha = 'P'
                            elif 'S' in l[-23].strip().upper(): pha = 'S'
                            else: pha = '?'

                            pha_res = '%7s %7s %10.4f %7.4f %7.4f %7.4f %7.2f\n'%(l[0], pha, float(l[-12]), float(l[-11]), float(l[-10]), float(l[-6]), float(l[-5]))
                            g.write(pha_res)

                savetxt(f, evt_res, newline=' ', fmt='%8.3f')
                
                f.write('\n')

    #_________ PLOT STATISTICS USING NLLOC [& HYPOCENTER] RESULTS

    def plot_statis(self, root_name):

        init_plotting()

        root_name = self.nlloc_dic['LOCFILES_OUTPUT']
        evt_data  = genfromtxt('%s_event.dat'%root_name)
        pha_data  = genfromtxt('%s_phase.dat'%root_name, dtype=str)

        #__________PLOT P-RESIDUALS VS WEIGHTS

        ax = plt.subplot(3,3,1)
        ax.set_xlabel('P residual [sec]')
        ax.set_ylabel('P weight')
        ax.grid()
        c1 = pha_data[:,1]=='P'
        c2 = array(pha_data[:,5], dtype=float)<=self.max_dist
        x  = array(pha_data[(c1)&(c2)][:,3], dtype=float)
        y  = array(pha_data[(c1)&(c2)][:,4], dtype=float)
        z  = array(pha_data[(c1)&(c2)][:,5], dtype=float)
        ax.set_xlim(-self.p_res_max, self.p_res_max)
        sc = ax.scatter(x, y, c=z, lw=0.1)
        cb = plt.colorbar(sc)
        tick_locator = ticker.MaxNLocator(nbins=6)
        cb.locator = tick_locator
        cb.update_ticks()
        cb.set_label('Distance [km]')
        ax.locator_params(axis = 'x', nbins=5)
        ax.locator_params(axis = 'y', nbins=5)         

        #__________PLOT S-RESIDUALS VS WEIGHTS

        ax = plt.subplot(3,3,2)
        ax.set_xlabel('S residual [sec]')
        ax.set_ylabel('S weight')
        ax.grid()
        c1 = pha_data[:,1]=='S'
        c2 = array(pha_data[:,5], dtype=float)<=self.max_dist
        x  = array(pha_data[(c1)&(c2)][:,3], dtype=float)
        y  = array(pha_data[(c1)&(c2)][:,4], dtype=float)
        z  = array(pha_data[(c1)&(c2)][:,5], dtype=float)
        ax.set_xlim(-self.s_res_max, self.s_res_max)
        sc = ax.scatter(x, y, c=z, lw=0.1)
        cb = plt.colorbar(sc)
        cb.set_label('Distance [km]')
        tick_locator = ticker.MaxNLocator(nbins=6)
        cb.locator = tick_locator
        cb.update_ticks()
        ax.locator_params(axis = 'x', nbins=5)
        ax.locator_params(axis = 'y', nbins=5)

        #__________PLOT P,S RESIDUALS VS DISTANCE
        
        ax = plt.subplot(3,3,3)
        ax.set_xlabel('Distance [km]')
        ax.set_ylabel('Residual [sec]')
        ax.grid()
        x  = array(pha_data[pha_data[:,1]=='P'][:,5], dtype=float)
        y  = array(pha_data[pha_data[:,1]=='P'][:,3], dtype=float)
        sc = ax.scatter(x, y, c='r', lw=0.1, alpha=.6, label='P')
        x  = array(pha_data[pha_data[:,1]=='S'][:,5], dtype=float)
        y  = array(pha_data[pha_data[:,1]=='S'][:,3], dtype=float)
        ax.set_xlim(0, self.max_dist)
        ax.set_ylim(-self.s_res_max, self.s_res_max)
        sc = ax.scatter(x, y, c='b', lw=0.1, alpha=.6, label='S')
        ax.legend(loc=1)
        ax.locator_params(axis = 'x', nbins=5)
        ax.locator_params(axis = 'y', nbins=5)

        #__________PLOT DEPTH HISTOGRAM

        ax = plt.subplot(3,3,4)
        h = -evt_data[:,2]
        w = self.dep_hist_bw
        b = arange(min(h), max(h) + w, w)
        ax.hist(h,b,color='grey',orientation='horizontal')
        ax.set_xlabel('# of Event')
        ax.set_ylabel('Depth [km]')
        ax.set_ylim(-self.dep_max, 0)
        ax.grid(True)
        ax.locator_params(axis = 'x', nbins=5)
        ax.locator_params(axis = 'y', nbins=5)       

        #__________PLOT H-ERROR HISTOGRAM

        ax = plt.subplot(3,3,5)
        h = evt_data[:,5]
        w = self.her_hist_bw
        b = arange(min(h), max(h) + w, w)
        ax.hist(h,bins=b,color='grey')
        ax.set_xlabel('Horizontal Error [km]')
        ax.set_ylabel('# of Event')
        ax.set_xlim(0, self.Herr_max)
        ax.grid(True)
        ax.locator_params(axis = 'x', nbins=5)
        ax.locator_params(axis = 'y', nbins=5)

        #__________PLOT Z-ERROR HISTOGRAM

        ax = plt.subplot(3,3,6,sharex=ax,sharey=ax)
        h = evt_data[:,6]
        w = self.zer_hist_bw
        b = arange(min(h), max(h) + w, w)
        ax.hist(h,bins=b,color='grey')
        ax.set_xlabel('Depth Error [km]')
        ax.set_ylabel('# of Event')
        ax.set_xlim(0, self.Zerr_max)
        ax.grid(True)
        ax.locator_params(axis = 'x', nbins=5)
        ax.locator_params(axis = 'y', nbins=5)

        #__________PLOT RMS HISTOGRAM

        ax = plt.subplot(3,3,7)
        h = evt_data[:,3]
        w = self.rms_hist_bw
        b = arange(min(h), max(h) + w, w)
        ax.hist(h,bins=b,color='grey')
        ax.set_xlim(0, self.rms_max)
        ax.set_xlabel('RMS [sec]')
        ax.set_ylabel('# of Event')
        ax.grid(True)
        ax.locator_params(axis = 'x', nbins=5)
        ax.locator_params(axis = 'y', nbins=5)

        #__________PLOT MINIMUM DISTANCE VS DEPTH

        ax = plt.subplot(3,3,8)
        x = evt_data[:,2]
        y = evt_data[:,15]
        z = evt_data[:,4]
        sc = ax.scatter(x, y, c=z, lw=0.1, alpha=.6, label='P')
        cb = plt.colorbar(sc)
        cb.set_label('Azimuthal Gap [deg]')
        tick_locator = ticker.MaxNLocator(nbins=6)
        cb.locator = tick_locator
        cb.update_ticks()
        ax.set_xlabel('Depth [km]')
        ax.set_ylabel('Minimum Distance [km]')
        ax.set_xlim(0, self.dep_max)
        ax.set_ylim(0, self.minds_max)
        ax.grid(True)
        ax.locator_params(axis = 'x', nbins=5)
        ax.locator_params(axis = 'y', nbins=5)
        
        plt.tight_layout()
        plt.savefig(os.path.join('figs', '%s_stat.png'%root_name))

        #__________PLOT MAP EVENTS WITH ERROR ELLIPSE

        init_plotting()

        ax = plt.subplot(1,1,1)
        ax.set_title('$Events:$ $Herr_{max} \leq %dkm$ $and$ $Zerr_{max} \leq %dkm$'%(self.Herr_max, self.Zerr_max))
        ax.grid(True)

        c1  = evt_data[:,7]+evt_data[:,8]<=self.Herr_max
        c2  = evt_data[:,10]<=self.Zerr_max
        
        cxx = evt_data[(c1)&(c2)][:,7]
        cxy = evt_data[(c1)&(c2)][:,8]
        cyy = evt_data[(c1)&(c2)][:,10]
        lon = evt_data[(c1)&(c2)][:,0]
        lat = evt_data[(c1)&(c2)][:,1]

        for x, y, xx, xy, xy, yy in zip(lon, lat, cxx, cxy, cxy, cyy):

            cov        = array([[xx, xy], [xy, yy]])
            vals, vecs = self.eigsorted(cov)
            theta      = degrees(arctan2(*vecs[:,0][::-1]))
            nstd       = 1
            w, h       = 2 * nstd * sqrt(vals)
            ell        = Ellipse(xy=(x, y), width=k2d(w), height=k2d(h),
                                 angle=theta, color='black', alpha=.5)
            ell.set_facecolor('none')
            ax.add_artist(ell)
            plt.scatter(x, y, zorder=100)
        
        plt.tight_layout()
        plt.savefig(os.path.join('figs','%s_map.png'%root_name))

#___________________START

start = main()
start.check_dir()
start.run_nlloc(sta_cor=False)
start.run_nlloc(sta_cor=True)
start.extract_nlloc_hyp(root_name=start.nlloc_dic['LOCFILES_OUTPUT'])
start.plot_statis(root_name=start.nlloc_dic['LOCFILES_OUTPUT'])

for f in glob(start.nlloc_dic['LOCFILES_OUTPUT'])+'*'): os.remove(f)

print '\n+++ Finito!\n'

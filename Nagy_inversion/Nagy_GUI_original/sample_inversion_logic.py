# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:32:41 2019

@author: fabiot / edited and adapted tp Python3 and Qt5 by Matt, 15/1/2020 
"""
# import python packages
from distutils.core import setup
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from numpy import core
from numpy import lib
from numpy import linalg
from matplotlib import pyplot
from matplotlib import mlab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse.linalg import lsqr
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata
from time import *
from os import system
import sys
import warnings
from scipy.ndimage import filters

# import Fabio's scripts
from lsq_linear import lsq_linear
from gravbox import gravbox
from grav_column_der import grav_column_der

#import .py geometry file from Qt Designer
import sample_geometry      #<---- add .py geometry file


warnings.filterwarnings("ignore")

class DesignerMainWindow(QtWidgets.QMainWindow, sample_geometry.Ui_MainWindow):        #<---- add .Ui geometry file 
    def __init__(self, parent=None):
        super(DesignerMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.connectActions()
    def main(self):
        self.show()
    def connectActions(self):
        self.actionInput_gravity_data.triggered.connect(self.Input_gravity_file_fun)          #<---- add "objectName" (found in Qt Designer) of the button you're working on
        # links pressing the button "Input_gravity_data" to the function you're going to define "Input_gravity_file_fun"
        self.actionInput_top_ice_geometry.triggered.connect(self.Input_ice_layer_file_fun) 
        self.actionInput_bottom_ice_geometry.triggered.connect(self.Input_water_layer_file_fun) 
        self.actionInput_layer_1_geometry.triggered.connect(self.Input_layer1_file_fun) 
        self.actionInput_layer_2_geometry.triggered.connect(self.Input_layer2_file_fun)
        self.actionRun_density_inversion.triggered.connect(self.Run_density_inversion_fun)
        self.actionSave_output_file.triggered.connect(self.Save_output_file_fun)
        self.actionClear_memory.triggered.connect(self.clear_memory)        
        self.actionExit_program.triggered.connect(self.exit_program)        
        self.pushButton_plotinputdata.clicked.connect(self.gridplot_input_data_fun) 
        self.pushButton_run_inversion.clicked.connect(self.run_inversion_fun) 
#        self.calculation.clicked.connect(self.calculation_fun) 
#        self.actionWrite_output_file.triggered.connect(self.Output_file_fun)   
#        self.actionExit_program.triggered.connect(self.exit_program)
#        self.actionInfo.triggered.connect(self.info)
#        self.actionHelp.triggered.connect(self.help)
           
    def Input_gravity_file_fun(self):
        global XG,YG,ZG,FA,FACP,NG           #<---- makes variable you define within function available outside of function
        global FILEGRAVITY
        if 'FILEGRAVITY' in globals():
            del FILEGRAVITY
            self.lineinput_gravity_file.setText("") 
            self.linenumber_gravity_data.setText("")
        pyplot.close('all')
        FILEGRAVITY=QtWidgets.QFileDialog.getOpenFileName(self)[0]      #<---- FILEGRAVITY will become the path to your gravity data .txt file you chose in the GUI
        if FILEGRAVITY:
            f=open(FILEGRAVITY,'r')
            INPUT_GRAVITY=lib.loadtxt(FILEGRAVITY,delimiter=',')       #<---- INPUT_GRAVITY will become a np array of your gravity data .txt file
           #pull out the columns from np array and assigns them to a variable 
            XG=INPUT_GRAVITY[:,1]
            YG=INPUT_GRAVITY[:,0]
            ZG=-INPUT_GRAVITY[:,2]
            FA=INPUT_GRAVITY[:,3]
            FACP=INPUT_GRAVITY[:,4]
            
            f.close()
            NG=len(XG)
            self.lineinput_gravity_file.setText(FILEGRAVITY) 
            self.linenumber_gravity_data.setText(str(NG)+" points")

    def Input_ice_layer_file_fun(self):
        global XI,YI,Z1I,RHOI,CPI,NI
        global FILEICE
        if 'FILEICE' in globals():
            del FILEICE
            self.lineinput_ice_layer.setText("") 
            self.linenumber_ice_layer.setText("")
        pyplot.close('all')
        FILEICE=QtWidgets.QFileDialog.getOpenFileName(self)[0]
        if FILEICE:
            f=open(FILEICE,'r')
            INPUT_ICE=lib.loadtxt(FILEICE,delimiter=',')
            XI=INPUT_ICE[:,1]
            YI=INPUT_ICE[:,0]
            Z1I=-INPUT_ICE[:,2]
            RHOI=INPUT_ICE[:,3]
            CPI=INPUT_ICE[:,4]  
            f.close()
            NI=len(XI)
            self.lineinput_ice_layer.setText(FILEICE) 
            self.linenumber_ice_layer.setText(str(NI)+" points")
        
    def Input_water_layer_file_fun(self):
        global XW,YW,Z1W,RHOW,CPW,NW
        global FILEWATER
        if 'FILEWATER' in globals():
            del FILEWATER
            self.lineinput_water_layer.setText("") 
            self.linenumber_water_layer.setText("")
        pyplot.close('all')
        FILEWATER=QtWidgets.QFileDialog.getOpenFileName(self)[0]
        if FILEWATER:
            f=open(FILEWATER,'r')
            INPUT_WATER=lib.loadtxt(FILEWATER,delimiter=',')
            XW=INPUT_WATER[:,1]
            YW=INPUT_WATER[:,0]
            Z1W=-INPUT_WATER[:,2]
            RHOW=INPUT_WATER[:,3]
            CPW=INPUT_WATER[:,4]
            f.close()
            NW=len(XW)
            self.lineinput_water_layer.setText(FILEWATER) 
            self.linenumber_water_layer.setText(str(NW)+" points")
        
    def Input_layer1_file_fun(self):
        global XL1,YL1,Z1L1,RHOL1,CPL1,NL1
        global FILELAYER1
        if 'FILELAYER1' in globals():
            del FILELAYER1
            self.lineinput_layer1.setText("") 
            self.linenumber_layer1.setText("")
        pyplot.close('all')
        FILELAYER1=QtWidgets.QFileDialog.getOpenFileName(self)[0]
        if FILELAYER1:
            f=open(FILELAYER1,'r')
            INPUT_LAYER1=lib.loadtxt(FILELAYER1,delimiter=',')
            XL1=INPUT_LAYER1[:,1]
            YL1=INPUT_LAYER1[:,0]
            Z1L1=-INPUT_LAYER1[:,2]
            RHOL1=INPUT_LAYER1[:,3]
            CPL1=INPUT_LAYER1[:,4]
            f.close()
            NL1=len(XL1)
            self.lineinput_layer1.setText(FILELAYER1) 
            self.linenumber_layer1.setText(str(NL1)+" points")
                
    def Input_layer2_file_fun(self):
        global XL2,YL2,Z1L2,RHOL2,CPL2,NL2
        global FILELAYER2
        if 'FILELAYER2' in globals():
            del FILELAYER2
            self.lineinput_layer2.setText("") 
            self.linenumber_layer2.setText("")
        pyplot.close('all')
        FILELAYER2=QtWidgets.QFileDialog.getOpenFileName(self)[0]
        if FILELAYER2:
            f=open(FILELAYER2,'r')
            INPUT_LAYER2=lib.loadtxt(FILELAYER2,delimiter=',')
            XL2=INPUT_LAYER2[:,1]
            YL2=INPUT_LAYER2[:,0]
            Z1L2=-INPUT_LAYER2[:,2]
            RHOL2=INPUT_LAYER2[:,3]
            CPL2=INPUT_LAYER2[:,4]
            f.close()
            NL2=len(XL2)
            self.lineinput_layer2.setText(FILELAYER2) 
            self.linenumber_layer2.setText(str(NL2)+" points")

    def gridplot_input_data_fun(self):
        global XG,YG,ZG,FA,FACP,NG
        global XI,YI,Z1I,RHOI,CPI,NI
        global XW,YW,Z1W,RHOW,CPW,NW
        global XL1,YL1,Z1L1,RHOL1,CPL1,NL1
        global XL2,YL2,Z1L2,RHOL2,CPL2,NL2
        global RESG,RESI,RESW,RESL1,RESL2
        global XV,YV,NGX,NGY,FAREG,RHOW_L1,RHOL1_L2,Z2L1,Z2L2,INDEXL1,INDEXL2,GRID_POINTS,BATHYGRIDVEC,BASEMENTGRIDVEC,EASTGRID,NORTHGRID,TOP_ICE_GRIDVEC,BOT_ICE_GRIDVEC,ZSGRIDVEC
        global FACPI,FACPW,FACPL1,FACPL2
        global FAFORL1,FAFORL2,TOP_ICE,BOT_ICE,BATHYMETRY,BASEMENT
        global DENS_CORR,DENS_CORRG
        global MAT_DENS,DFDENS
        global FILEICE
        global FILEWATER
        
        RESG=float(self.linegravity_data_resolution.text()) 
        RESL1=float(self.linelayer1_resolution.text())
        RESL2=float(self.linelayer2_resolution.text())
        
        if 'FILEICE' in globals():
            RESI=float(self.lineice_layer_resolution.text()) 
        else:
            RESI=RESL1
            XI=XL1
            YI=YL1
            Z1I=0*Z1L1
            RHOI=0*Z1L1
            CPI=0*Z1L1
            NI=NL1
            
        if 'FILEWATER' in globals():
            RESW=float(self.linewater_layer_resolution.text()) 
        else:
            RESW=RESL1
            XW=XL1
            YW=YL1
            Z1W=0*Z1L1
            RHOW=1.03*Z1L1/Z1L1
            CPW=0*Z1L1
            NW=NL1
            
            
        pyplot.close('all')
        
        self.messagebox.setText("Preparing grids") 
        TREND_TYPE = str(self.comboBox_detrending.currentText())
       

        
        DENS_CORR=0*XL1
        DENS_CORRG=0*XG
        Z2I=0*Z1I
        Z2W=0*Z1W
        Z2L1=0*Z1L1
        Z2L2=0*Z1L2
        
        XMIN=min(XL1)
        XMAX=max(XL1)
        YMIN=min(YL1)
        YMAX=max(YL1)

        INDEXL1=core.zeros(NL1)
        INDEXL2=core.zeros(NL2)

        # gridding preliminary data     
        XV=core.arange(XMIN,XMAX+0.0001,RESL1)
        YV=core.arange(YMIN,YMAX+0.0001,RESL1)             
        EASTGRID,NORTHGRID=lib.meshgrid(YV,XV)    
    
        ZGGRID=griddata((YG,XG),ZG,(EASTGRID, NORTHGRID), method='linear')
        FAGRID=griddata((YG,XG),FA,(EASTGRID, NORTHGRID), method='linear')
        FACPGRID=griddata((YG,XG),FACP,(EASTGRID, NORTHGRID), method='linear')

        RHOIGRID=griddata((YI,XI),RHOI,(EASTGRID, NORTHGRID), method='linear')
        RHOWGRID=griddata((YW,XW),RHOW,(EASTGRID, NORTHGRID), method='linear')
        RHOL1GRID=griddata((YL1,XL1),RHOL1,(EASTGRID, NORTHGRID), method='linear')

        TOP_ICE_GRID=griddata((YI,XI),-Z1I,(EASTGRID, NORTHGRID), method='linear')
        BOT_ICE_GRID=griddata((YW,XW),-Z1W,(EASTGRID, NORTHGRID), method='linear')
        Z1L1GRID=griddata((YL1,XL1),Z1L1,(EASTGRID, NORTHGRID), method='linear')
        CPL1GRID=griddata((YL1,XL1),CPL1,(EASTGRID, NORTHGRID), method='linear')    

        Z1L2GRID=griddata((YL2,XL2),Z1L2,(EASTGRID, NORTHGRID), method='linear')
        CPL2GRID=griddata((YL2,XL2),CPL2,(EASTGRID, NORTHGRID), method='linear')   

        Z1IGRID=griddata((YL1,XL1),Z1L1,(EASTGRID, NORTHGRID), method='linear')

        # building vector of control points
        NCPL1=int(core.sum(CPL1))
        XCL1=core.zeros(NCPL1)
        YCL1=core.zeros(NCPL1)
  
        ICPL1=-1        
        for IPL1 in range(0,NL1):  
            if CPL1[IPL1] > 0.5:
                ICPL1=ICPL1+1        
                XCL1[ICPL1]=XL1[IPL1]
                YCL1[ICPL1]=YL1[IPL1]
            INDEXL1[IPL1]=IPL1


        NCPL2=int(core.sum(CPL2))
        XCL2=core.zeros(NCPL2)
        YCL2=core.zeros(NCPL2)
  
        ICPL2=-1        
        for IPL2 in range(0,NL2):  
            if CPL2[IPL2] > 0.5:
                ICPL2=ICPL2+1        
                XCL2[ICPL2]=XL2[IPL2]
                YCL2[ICPL2]=YL2[IPL2]
            INDEXL2[IPL2]=IPL2


        NGX=XV.size
        NGY=YV.size
        XGRIDVEC=core.zeros([NGX*NGY])
        YGRIDVEC=core.zeros([NGX*NGY])
        BATHYGRIDVEC=core.zeros([NGX*NGY])
        BASEMENTGRIDVEC=core.zeros([NGX*NGY])
        TOP_ICE_GRIDVEC=core.zeros([NGX*NGY])
        BOT_ICE_GRIDVEC=core.zeros([NGX*NGY])
        ZSGRIDVEC=core.zeros([NGX*NGY])
        RHOIGRIDVEC=core.zeros([NGX*NGY])
        RHOWGRIDVEC=core.zeros([NGX*NGY])
        RHOL1GRIDVEC=core.zeros([NGX*NGY])

        INDGRID=-1
        for INDX in range(0,NGX):
            for INDY in range(0,NGY):
                INDGRID=INDGRID+1          
                XGRIDVEC[INDGRID]=NORTHGRID[INDX,INDY]
                YGRIDVEC[INDGRID]=EASTGRID[INDX,INDY]
                RHOIGRIDVEC[INDGRID]=RHOIGRID[INDX,INDY]
                RHOWGRIDVEC[INDGRID]=RHOWGRID[INDX,INDY]
                RHOL1GRIDVEC[INDGRID]=RHOL1GRID[INDX,INDY]
                TOP_ICE_GRIDVEC[INDGRID]=TOP_ICE_GRID[INDX,INDY]
                BOT_ICE_GRIDVEC[INDGRID]=BOT_ICE_GRID[INDX,INDY]
                BATHYGRIDVEC[INDGRID]=-Z1L1GRID[INDX,INDY]
                BASEMENTGRIDVEC[INDGRID]=-Z1L2GRID[INDX,INDY]
                
        GRID_POINTS=core.zeros([NGX*NGY,2])
        GRID_POINTS[:,0]=YGRIDVEC
        GRID_POINTS[:,1]=XGRIDVEC

        intfundensi=LinearNDInterpolator(GRID_POINTS,RHOIGRIDVEC)
        RHOI_W=intfundensi(YW,XW) 
        RHOI_W[core.isnan(RHOI_W)]=core.mean(RHOI_W[core.isfinite(RHOI_W)])
        intfundensw=LinearNDInterpolator(GRID_POINTS,RHOWGRIDVEC)
        RHOW_L1=intfundensw(YL1,XL1) 
        RHOW_L1[core.isnan(RHOW_L1)]=core.mean(RHOW_L1[core.isfinite(RHOW_L1)])
        intfundensl1=LinearNDInterpolator(GRID_POINTS,RHOL1GRIDVEC)
        RHOL1_L2=intfundensl1(YL2,XL2) 
        RHOL1_L2[core.isnan(RHOL1_L2)]=core.mean(RHOL1_L2[core.isfinite(RHOL1_L2)])
        
        intfunbathy=LinearNDInterpolator(GRID_POINTS,BATHYGRIDVEC)
        BATHYMETRY=intfunbathy(YG,XG) 
        BATHYFGRID=griddata((YG,XG),BATHYMETRY,(EASTGRID, NORTHGRID), method='linear') 

        intfunbasement=LinearNDInterpolator(GRID_POINTS,BASEMENTGRIDVEC)
        BASEMENT=intfunbasement(YG,XG) 
        BASEMENTGRID=griddata((YG,XG),BASEMENT,(EASTGRID, NORTHGRID), method='linear') 

        intfunicetop=LinearNDInterpolator(GRID_POINTS,TOP_ICE_GRIDVEC)
        TOP_ICE=intfunicetop(YG,XG) 
        TOP_ICE_GRID_FA=griddata((YG,XG),TOP_ICE,(EASTGRID, NORTHGRID), method='linear') 

        intfunicebot=LinearNDInterpolator(GRID_POINTS,BOT_ICE_GRIDVEC)
        BOT_ICE=intfunicebot(YG,XG) 
        BOT_ICE_GRID_FA=griddata((YG,XG),BOT_ICE,(EASTGRID, NORTHGRID), method='linear') 
        

        # calculate regional gravity from control points
        FACPI=core.zeros(NG)
        FACPW=core.zeros(NG)
        FACPL1=core.zeros(NG)
        FACPL2=core.zeros(NG)
        FACPTOT=core.zeros(NG)
        self.messagebox.setText("Calculating regional field") 
        for IDAT in range(0,NG):
            self.progressBar.setProperty("value", 100*(IDAT+1)/NG)
            FACPI[IDAT]=core.sum(gravbox(XG[IDAT],YG[IDAT],ZG[IDAT],XI-0.5*RESI,XI+0.5*RESI,YI-0.5*RESI,YI+0.5*RESI,Z1I,Z2I,RHOI))
            FACPW[IDAT]=core.sum(gravbox(XG[IDAT],YG[IDAT],ZG[IDAT],XW-0.5*RESW,XW+0.5*RESW,YW-0.5*RESW,YW+0.5*RESW,Z1W,Z2W,RHOW-RHOI_W))
            FACPL1[IDAT]=core.sum(gravbox(XG[IDAT],YG[IDAT],ZG[IDAT],XL1-0.5*RESL1,XL1+0.5*RESL1,YL1-0.5*RESL1,YL1+0.5*RESL1,Z1L1,Z2L1,-DENS_CORR+RHOL1-RHOW_L1))
            FACPL2[IDAT]=core.sum(gravbox(XG[IDAT],YG[IDAT],ZG[IDAT],XL2-0.5*RESL2,XL2+0.5*RESL2,YL2-0.5*RESL2,YL2+0.5*RESL2,Z1L2,Z2L2,RHOL2-RHOL1_L2))
            FACPTOT=-FACPI-FACPW+FACPL1+FACPL2
        
        FADCP=FACP-FACPTOT
        if TREND_TYPE=="Constant value":
            FAREG=core.mean(FADCP)*core.ones([NG])
        else:
            ATEMP=lib.column_stack((core.ones(XG.size), XG, YG))
            C,RESID,RANK,SIGMA=linalg.lstsq(ATEMP,FADCP)
            FAREG=C[0]*core.ones([NG])+C[1]*XG+C[2]*YG
        
        FARES=FA-FAREG
        
        FACPTOTGRID=griddata((YG,XG),FACPTOT,(EASTGRID, NORTHGRID), method='linear')  
        FAREGGRID=griddata((YG,XG),FAREG,(EASTGRID, NORTHGRID), method='linear')  
        FARESGRID=griddata((YG,XG),FARES,(EASTGRID, NORTHGRID), method='linear')  

        fig=pyplot.figure(1,(14,9))
        ax1 = fig.add_subplot(2,2,1, adjustable='box', aspect=1)
        ax2 = fig.add_subplot(2,2,2, adjustable='box', aspect=1)
        ax3=  fig.add_subplot(2,2,3, adjustable='box', aspect=1)
        ax4=  fig.add_subplot(2,2,4, adjustable='box', aspect=1)

        img1=ax1.contourf(0.001*YV,0.001*XV,FAGRID,100, cmap='jet')
        ax1.set_ylabel('NORTH[km]')
        ax1.set_title('Observed gravity anomaly [mGal]')
        divider=make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img1, cax=cax1)

        img2=ax2.contourf(0.001*YV,0.001*XV,FARESGRID,100, cmap='jet')
        ax2.set_title('Residual gravity anomaly [mGal]')
        divider=make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img2, cax=cax2)

        img3=ax3.contourf(0.001*YV,0.001*XV,FACPGRID,100, cmap='jet')
        ax3.set_xlabel('EAST[km]')
        ax3.set_ylabel('NORTH[km]')
        ax3.set_title('Gravity anomaly from Control Points[mGal]')
        divider=make_axes_locatable(ax3)
        cax3 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img3, cax=cax3)   
    
        img4=ax4.contourf(0.001*YV,0.001*XV,-FAREGGRID,100, cmap='jet')
        ax4.set_xlabel('EAST[km]')
        ax4.set_title('Regional gravity trend [mGal]')
        divider=make_axes_locatable(ax4)
        cax4 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img4, cax=cax4)   
        pyplot.tight_layout(pad=1)
        pyplot.show() 


        fig2=pyplot.figure(2,(14,9))
        ax21=fig2.add_subplot(2,2,1, adjustable='box', aspect=1)
        ax22=fig2.add_subplot(2,2,2, adjustable='box', aspect=1)
        ax23=fig2.add_subplot(2,2,3, adjustable='box', aspect=1)
        ax24=fig2.add_subplot(2,2,4, adjustable='box', aspect=1)

        img21=ax21.contourf(0.001*YV,0.001*XV,TOP_ICE_GRID,100, cmap='jet')
        ax21.set_ylabel('NORTH[km]')
        ax21.set_title('Ice surface [m]')
        divider=make_axes_locatable(ax21)
        cax21 = divider.append_axes("right", size="5%", pad=0.05)
        fig2.colorbar(img21, cax=cax21)

        img22=ax22.contourf(0.001*YV,0.001*XV,BOT_ICE_GRID,100, cmap='jet')
        ax22.set_title('Water surface [m]')
        divider=make_axes_locatable(ax22)
        cax22 = divider.append_axes("right", size="5%", pad=0.05)
        fig2.colorbar(img22, cax=cax22)
        
        img23=ax23.contourf(0.001*YV,0.001*XV,-Z1L1GRID,100, cmap='jet')
        ax23.plot(0.001*YCL1,0.001*XCL1,'ko',markersize=2)
        ax23.set_ylabel('NORTH[km]')
        ax23.set_xlabel('EAST[km]')
        ax23.set_title('Bathymetry from control points [m]')
        divider=make_axes_locatable(ax23)
        cax23 = divider.append_axes("right", size="5%", pad=0.05)
        fig2.colorbar(img23, cax=cax23)

        img24=ax24.contourf(0.001*YV,0.001*XV,-Z1L2GRID,100, cmap='jet')
        ax24.plot(0.001*YCL2,0.001*XCL2,'ko',markersize=2)
        ax24.set_xlabel('EAST[km]')
        ax24.set_title('Basement from control points [m]')
        divider=make_axes_locatable(ax24)
        cax24 = divider.append_axes("right", size="5%", pad=0.05)
        fig2.colorbar(img24, cax=cax24)
        pyplot.tight_layout(pad=1)
        pyplot.show() 

        self.messagebox.setText("Gridding and plotting done") 
        self.pushButton_run_inversion.setEnabled(True)
        self.actionSave_output_file.setEnabled(True)
        FAFORL1=FACPL1
        FAFORL2=FACPL2        
        
    def run_inversion_fun(self):
        global XG,YG,ZG,FA,FACP,NG
        global XI,YI,Z1I,RHOI,CPI,NI
        global XW,YW,Z1W,RHOW,CPW,NW
        global XL1,YL1,Z1L1,RHOL1,CPL1,NL1
        global XL2,YL2,Z1L2,RHOL2,CPL2,NL2
        global RESG,RESI,RESW,RESL1,RESL2
        global XV,YV,NGX,NGY,FAREG,RHOW_L1,RHOL1_L2,Z2L1,Z2L2,INDEXL1,INDEXL2,GRID_POINTS,BATHYGRIDVEC,BASEMENTGRIDVEC,EASTGRID,NORTHGRID,TOP_ICE_GRIDVEC,BOT_ICE_GRIDVEC,ZSGRIDVEC
        global FAFORL1,FAFORL2,TOP_ICE,BOT_ICE,BATHYMETRY,BASEMENT
        global FAINV,ZS,ZSBAS,GRAVDIFFGRID
        global DENS_CORR,DENS_CORRG
        global MAT_DENS,DFDENS
        global MATDATPAR
        global DIFFLAYERSGRID,DIFF1,DIFF2,INDEXCP1,INDEXCP2

        
        pyplot.close(3)
        if 'MATDATPAR' in globals():
            del MATDATPAR
        if 'MAT_DENS' in globals():
            del MAT_DENS
            
        
        
        TOLBATHYITER=float(self.linebathymetry_variation.text()) 
        TOLBASEMENTITER=float(self.linebasement_variation.text()) 
        TOLBATHY=float(self.linebathymetry_constraints.text())
        TOLBASEMENT=float(self.linebasement_constraints.text())
        TOLCHISQ=float(self.lineleast_squares.text())
        TOLDCHISQ=float(self.lineleast_squares_tolerance.text())
        MAXITER=float(self.linemaximum_iterations.text())
        MIN_SED_THICKNESS=float(self.lineminimum_sediment_thickness.text())
        
        ZS=Z1L1
        ZS2=core.concatenate((Z1L1,Z1L2))
        CP2=core.concatenate((CPL1,CPL2))
        MATDATPAR=(core.zeros([NG,NL1+NL2]))
        FAINV=FA-FAREG
        DF=core.zeros([NG])        
        
        START_INV_TIME = strftime("%H:%M:%S")
        print("Starting inversion at:",START_INV_TIME)
        
        START_INV_TIME_DEC=time()
        
        CHISQ1=core.Inf
        DCHISQ=core.Inf

        ITER=0
        self.messagebox.setText("Executing iteration "+str(ITER+1)) 
        while DCHISQ > 1+TOLDCHISQ:
            START_ITER_TIME=time()
            ITER=ITER+1
            print("iteration", ITER)
            FAFOR=core.zeros([NG])    
            FAFORL1=core.zeros([NG])
            FAFORL2=core.zeros([NG])
            for IDAT in range(0,NG):  
                self.progressBar.setProperty("value", 100*(IDAT+1)/NG)
                FAFORL1[IDAT]=core.sum(gravbox(XG[IDAT],YG[IDAT],ZG[IDAT],XL1-0.5*RESL1,XL1+0.5*RESL1,YL1-0.5*RESL1,YL1+0.5*RESL1,ZS2[0:NL1],Z2L1,-DENS_CORR+RHOL1-RHOW_L1))
                FAFORL2[IDAT]=core.sum(gravbox(XG[IDAT],YG[IDAT],ZG[IDAT],XL2-0.5*RESL2,XL2+0.5*RESL2,YL2-0.5*RESL2,YL2+0.5*RESL2,ZS2[NL1:NL1+NL2],Z2L2,RHOL2-RHOL1_L2))
                FAFOR=FAFORL1+FAFORL2        
                MATDATPAR[IDAT,0:NL1]=grav_column_der(XG[IDAT],YG[IDAT],ZG[IDAT],XL1,YL1,ZS2[0:NL1],Z2L1,RESL1,-DENS_CORR+RHOL1-RHOW_L1)
                MATDATPAR[IDAT,NL1:NL1+NL2]=grav_column_der(XG[IDAT],YG[IDAT],ZG[IDAT],XL2,YL2,ZS2[NL1:NL1+NL2],Z2L2,RESL2,RHOL2-RHOL1_L2)

            DF=FAINV-FAFOR

            DZS=lsqr(MATDATPAR,DF,show=False)[0]

            for ITDZS in range(0,NL1):
                if DZS[ITDZS] > TOLBATHYITER:
                    DZS[ITDZS]=TOLBATHYITER
                elif DZS[ITDZS] < -TOLBATHYITER:
                    DZS[ITDZS]=-TOLBATHYITER
            
            for ITDZS in range(NL1,NL1+NL2):
                if DZS[ITDZS] > TOLBASEMENTITER:
                    DZS[ITDZS]=TOLBASEMENTITER
                elif DZS[ITDZS] < -TOLBASEMENTITER:
                    DZS[ITDZS]=-TOLBASEMENTITER

            ZS2=ZS2+DZS
            ZS=ZS2[0:NL1]   
            ZSBAS=ZS2[NL1:NL1+NL2]
            
            for INDP1 in range(0,NL1):
                if CPL1[INDP1]==1:
                    if ZS[INDP1] > Z1L1[INDP1]+TOLBATHY:
                        ZS[INDP1]=Z1L1[INDP1]+TOLBATHY
                    elif ZS[INDP1] < Z1L1[INDP1]-TOLBATHY:
                        ZS[INDP1]=Z1L1[INDP1]-TOLBATHY
            for INDP2 in range(0,NL2):
                if CPL2[INDP2]==1:
                    if ZSBAS[INDP2] > Z1L2[INDP2]+TOLBASEMENT:
                        ZSBAS[INDP2]=Z1L2[INDP2]+TOLBASEMENT
                    elif ZSBAS[INDP2] < Z1L2[INDP2]-TOLBASEMENT:
                        ZSBAS[INDP2]=Z1L2[INDP2]-TOLBASEMENT

            ZSGRID=griddata((YL1,XL1),ZS,(EASTGRID, NORTHGRID), method='linear')
            
            INDGRID=-1
            for INDX in range(0,NGX):
                for INDY in range(0,NGY):
                    INDGRID=INDGRID+1          
                    ZSGRIDVEC[INDGRID]=ZSGRID[INDX,INDY]
                    
            
            intfunZSZBAS=LinearNDInterpolator(GRID_POINTS,ZSGRIDVEC)    
            ZSZBAS=intfunZSZBAS(YL2,XL2) 

            for INDPAR in range(0,NL2):
                if ZSBAS[INDPAR]<ZSZBAS[INDPAR]:
                    ZSBAS[INDPAR]=ZSZBAS[INDPAR]+MIN_SED_THICKNESS

            CHISQ2=core.mean(DF**2)
            DCHISQ=CHISQ1/CHISQ2
            CHISQ1=CHISQ2
            ITERATION_TIME = strftime("%H:%M:%S")
            print("least-squares norm=",CHISQ2)
            print("Iteration terminated at:",ITERATION_TIME)
            END_ITER_TIME=time()
            END_INVERSION_TIME=time()
            ELAPSED_ITER_TIME=int(core.round(END_ITER_TIME-START_ITER_TIME))
            ELAPSED_INVERSION_TIME=int(core.round(END_INVERSION_TIME-START_INV_TIME_DEC))
            self.messagebox.setText("Iteration "+ str(ITER)+" terminated after "+str(ELAPSED_ITER_TIME)+ " seconds with least-squares norm="+str(CHISQ2)+", executing iteration "+str(ITER+1))


            if ITER==MAXITER:
                print("terminated - maximum number of iterations reached",MAXITER)
                self.messagebox.setText("Inversion terminated after "+ str(ITER)+" iterations with least-squares norm="+str(CHISQ2)+"in "+str(ELAPSED_INVERSION_TIME)+" seconds because maximum number of iterations "+ str(MAXITER)+" reached") 
                break
            if CHISQ2 < TOLCHISQ:
                print("terminated - least-squares norm <",TOLCHISQ)
                self.messagebox.setText("Inversion terminated after "+ str(ITER)+" iterations with least-squares norm="+str(CHISQ2)+" in "+str(ELAPSED_INVERSION_TIME)+" seconds because least-squares norm < " + str(TOLCHISQ)) 
                break

        if DCHISQ < 1+TOLDCHISQ:
            print("terminated - no significant variation in least-squares norm")
            self.messagebox.setText("Inversion terminated after "+ str(ITER)+" iterations with least-squares norm="+str(CHISQ2)+" in "+str(ELAPSED_INVERSION_TIME)+" seconds because tolerance on least-squares norm < " + str(TOLDCHISQ))
        ZSBASGRID=griddata((YL2,XL2),ZSBAS,(EASTGRID, NORTHGRID), method='linear') 
        GRAVDIFFGRID=griddata((YG,XG),FAINV-FAFOR,(EASTGRID, NORTHGRID), method='linear')     
        FAFORGRID=griddata((YG,XG),FAFOR+FAREG,(EASTGRID, NORTHGRID), method='linear')   

        INDGRID=-1
        for INDX in range(0,NGX):
            for INDY in range(0,NGY):
                INDGRID=INDGRID+1          
                BASEMENTGRIDVEC[INDGRID]=-ZSBASGRID[INDX,INDY]
                

        BATHYGRIDVEC=-ZSGRIDVEC

        intfunbathy=LinearNDInterpolator(GRID_POINTS,BATHYGRIDVEC)
        BATHYMETRY=intfunbathy(YG,XG) 
        BATHYFGRID=griddata((YG,XG),BATHYMETRY,(EASTGRID, NORTHGRID), method='linear') 

        intfunbasement=LinearNDInterpolator(GRID_POINTS,BASEMENTGRIDVEC)
        BASEMENT=intfunbasement(YG,XG) 
        BASEMENTGRID=griddata((YG,XG),BASEMENT,(EASTGRID, NORTHGRID), method='linear') 
       
        
        DIFFLAYERSGRID=BATHYFGRID-BASEMENTGRID

        DIFF1=core.round(Z1L1[CPL1==1]-ZS[CPL1==1])

        DIFF2=core.round(Z1L2[CPL2==1]-ZSBAS[CPL2==1])
        INDEXCP1=INDEXL1[CPL1==1]
        INDEXCP2=INDEXL2[CPL2==1]

        fig3=pyplot.figure(3,(14,9))
        ax31=fig3.add_subplot(2,2,1, adjustable='box', aspect=1)
        img31=ax31.contourf(0.001*YV,0.001*XV,BATHYFGRID,100, cmap='jet')
        ax31.set_ylabel('NORTH[km]')
        #ax31.set_xlabel('EAST[km]')
        ax31.set_title('Inverted Bathymetry [m]')
        divider=make_axes_locatable(ax31)
        cax31 = divider.append_axes("right", size="5%", pad=0.05)
        fig3.colorbar(img31, cax=cax31)
  
        ax32=fig3.add_subplot(2,2,2, adjustable='box', aspect=1)
        img32=ax32.contourf(0.001*YV,0.001*XV,BASEMENTGRID,100, cmap='jet')
        #ax32.set_ylabel('NORTH[km]')
        #ax32.set_xlabel('EAST[km]')
        #ax32.set_title('Inverted sediment thickness [m]')
        ax32.set_title('Inverted basement depth [m]')
        divider=make_axes_locatable(ax32)
        cax32 = divider.append_axes("right", size="5%", pad=0.05)
        fig3.colorbar(img32, cax=cax32)  

        #fig4=pyplot.figure(4,(14,5))
        ax41=fig3.add_subplot(2,2,3, adjustable='box', aspect=1)
        img41=ax41.contourf(0.001*YV,0.001*XV,GRAVDIFFGRID,100, cmap='jet')
        ax41.set_ylabel('NORTH[km]')
        ax41.set_xlabel('EAST[km]')
        ax41.set_title('Gravity difference [mGal]')
        divider=make_axes_locatable(ax41)
        cax41 = divider.append_axes("right", size="5%", pad=0.05)
        fig3.colorbar(img41, cax=cax41)

        ax42=fig3.add_subplot(2,2,4, adjustable='box', aspect=1)
        img42=ax42.contourf(0.001*YV,0.001*XV,FAFORGRID,100, cmap='jet')
        #ax42.set_ylabel('NORTH[km]')
        ax42.set_xlabel('EAST[km]')
        ax42.set_title('Calculated gravity [mGal]')
        divider=make_axes_locatable(ax42)
        cax42 = divider.append_axes("right", size="5%", pad=0.05)
        fig3.colorbar(img42, cax=cax42)
        pyplot.tight_layout(pad=1)
        pyplot.show() 
        self.actionRun_density_inversion.setEnabled(True)
        self.actionSave_output_file.setEnabled(True)
        
    def Run_density_inversion_fun(self):
        global XG,YG,ZG,FA,FACP,NG
        global XI,YI,Z1I,RHOI,CPI,NI
        global XW,YW,Z1W,RHOW,CPW,NW
        global XL1,YL1,Z1L1,RHOL1,CPL1,NL1
        global XL2,YL2,Z1L2,RHOL2,CPL2,NL2
        global RESG,RESI,RESW,RESL1,RESL2
        global XV,YV,NGX,NGY,FAREG,RHOW_L1,RHOL1_L2,Z2L1,Z2L2,INDEXL1,INDEXL2,GRID_POINTS,BATHYGRIDVEC,BASEMENTGRIDVEC,EASTGRID,NORTHGRID,TOP_ICE_GRIDVEC,BOT_ICE_GRIDVEC,ZSGRIDVEC
        global FACPI,FACPW
        global FAFORL1,FAFORL2,TOP_ICE,BOT_ICE,BATHYMETRY,BASEMENT
        global FAINV,ZS,ZSBAS,GRAVDIFFGRID
        global DENS_CORR,DENS_CORRG
        global MAT_DENS,DFDENS
        global SIGMA
        global DIFF_GRAD_DENS_INV
        global MATDATPAR
        pyplot.close(4)
        if 'MATDATPAR' in globals():
            del MATDATPAR
        if 'MAT_DENS' in globals():
            del MAT_DENS
        self.messagebox.setText("Executing density inversion on gravity misfit")

        SIGMA=float(self.linedensity_shift.text())
        
        DENS_AMPLITUDE=float(self.linedensity_inversion.text())
        DENS_CORR=core.zeros([NL1])
        DENS_CORRG=core.zeros([NG])
        MAT_DENS=core.zeros([NG,NG])
        for IDAT in range(0,NG):  
            self.progressBar.setProperty("value", 100*(IDAT+1)/NG)
            MAT_DENS[IDAT,:]=gravbox(XG[IDAT],YG[IDAT],ZG[IDAT],XG[:]-0.5*RESG,XG[:]+0.5*RESG,YG[:]-0.5*RESG,YG[:]+0.5*RESG,-BATHYMETRY[:],0*BATHYMETRY,1)
        
        
        
        GRAVDIFFGRID[core.isnan(GRAVDIFFGRID)]=0  
        AMPLITUDE=core.max(core.abs(GRAVDIFFGRID))
        GRAVDIFFGRIDFILPRE=filters.gaussian_filter(GRAVDIFFGRID, SIGMA, mode='constant')
        AMPLITUDEPRE=core.max(core.abs(GRAVDIFFGRIDFILPRE))
        GRAVDIFFGRIDFIL=AMPLITUDE*filters.gaussian_filter(GRAVDIFFGRID, SIGMA, mode='constant')/AMPLITUDEPRE
       
        
        
        DIFFVECGRID=core.zeros([NGX*NGY])
        INDGRID=-1
        for INDX in range(0,NGX):
            for INDY in range(0,NGY):
                INDGRID=INDGRID+1          
                DIFFVECGRID[INDGRID]=GRAVDIFFGRIDFIL[INDX,INDY]
         
        intfungravdiff=LinearNDInterpolator(GRID_POINTS,DIFFVECGRID)
        DIFF_GRAV_DENS_INV=intfungravdiff(YG,XG)
        
        lb=-DENS_AMPLITUDE*core.ones([NG])
        ub=DENS_AMPLITUDE*core.ones([NG])         
                
        DFDENS=FAINV-FAFORL2-FAFORL1
        DENS_CORRG=-lsqr(MAT_DENS,DIFF_GRAV_DENS_INV,show=False)[0]
#        DENS_CORRG_lsq_linear=lsq_linear(MAT_DENS,DIFF_GRAV_DENS_INV,bounds=(lb, ub), verbose=1)
#        DENS_CORRG=DENS_CORRG_lsq_linear.x      
        
        for ITER in range(0,NG):
            if DENS_CORRG[ITER] > DENS_AMPLITUDE:
                DENS_CORRG[ITER]=DENS_AMPLITUDE
            elif DENS_CORRG[ITER] < -DENS_AMPLITUDE:
                DENS_CORRG[ITER]=-DENS_AMPLITUDE
        
        POINTS_DENS_CORRG=core.zeros([NG,2])
        POINTS_DENS_CORRG[:,0]=YG[:]
        POINTS_DENS_CORRG[:,1]=XG[:]
        intfundenscorr=LinearNDInterpolator(POINTS_DENS_CORRG,DENS_CORRG)
        DENS_CORR=intfundenscorr(YL1,XL1) 
        DENS_CORR[core.isnan(DENS_CORR)]=0
        
        DENSCORRGRID=griddata((YG,XG),DENS_CORRG,(EASTGRID, NORTHGRID), method='linear')       
        GRAVDIFFGRIDFIL[core.isnan(DENSCORRGRID)]=core.nan
        self.messagebox.setText("Density inversion terminated")
        
        fig4=pyplot.figure(4,(14,5))
        ax51=fig4.add_subplot(1,2,1, adjustable='box', aspect=1)
        ax52=fig4.add_subplot(1,2,2, adjustable='box', aspect=1)
        img51=ax51.contourf(0.001*YV,0.001*XV,GRAVDIFFGRIDFIL,100, cmap='jet')
        ax51.set_ylabel('NORTH[km]')
        ax51.set_xlabel('EAST[km]')
        ax51.set_title('Gravity difference for density inversion [mGal]')
        divider=make_axes_locatable(ax51)
        cax51 = divider.append_axes("right", size="5%", pad=0.05)
        fig4.colorbar(img51, cax=cax51)
        ax51=fig4.add_subplot(1,2,1, adjustable='box', aspect=1)
                
        img52=ax52.contourf(0.001*YV,0.001*XV,DENSCORRGRID,100, cmap='jet')
        #ax52.set_ylabel('NORTH[km]')
        ax52.set_xlabel('EAST[km]')
        ax52.set_title('Density from gravity inversion [g/cm$^3$]')
        divider=make_axes_locatable(ax52)
        cax52 = divider.append_axes("right", size="5%", pad=0.05)
        fig4.colorbar(img52, cax=cax52)
        pyplot.tight_layout(pad=1)
        pyplot.show()
        
        
    def Save_output_file_fun(self):
        global XV,YV,NGX,NGY,FAREG,RHOW_L1,RHOL1_L2,Z2L1,Z2L2,INDEXL1,INDEXL2,GRID_POINTS,BATHYGRIDVEC,BASEMENTGRIDVEC,EASTGRID,NORTHGRID,TOP_ICE_GRIDVEC,BOT_ICE_GRIDVEC,ZSGRIDVEC
        global FACPI,FACPW
        global FAFORL1,FAFORL2,TOP_ICE,BOT_ICE,BATHYMETRY,BASEMENT
        global DENS_CORR,DENS_CORRG
        FILEOUT = QtWidgets.QFileDialog.getSaveFileName()
        FILEOUT=str(FILEOUT)
        if FILEOUT:
            f=open(FILEOUT, 'w')
            f.write("EAST,NORTH,FACPI,FACPW,FAREG,FAFORL1,FAFORL2,TOP_ICE,BOT_ICE,BATHYMETRY,BASEMENT,DENS_CORR \n")    
            mat_out=core.zeros((NG,12))
            mat_out[:,0]=YG
            mat_out[:,1]=XG
            mat_out[:,2]=FACPI
            mat_out[:,3]=FACPW
            mat_out[:,4]=FAREG
            mat_out[:,5]=FAFORL1
            mat_out[:,6]=FAFORL2
            mat_out[:,7]=TOP_ICE
            mat_out[:,8]=BOT_ICE
            mat_out[:,9]=BATHYMETRY
            mat_out[:,10]=BASEMENT 
            mat_out[:,11]=DENS_CORRG 
            lib.savetxt(f,mat_out,fmt='%.3f',delimiter=',')
            f.close()    

    def clear_memory(self):
        global MAT_DENS,DFDENS
        global MATDATPAR
        global FILEICE
        global FILEWATER
        global FILEGRAVITY
        global FILELAYER1
        global FILELAYER2
        
        pyplot.close('all')
        if 'MATDATPAR' in globals():
            del MATDATPAR
        if 'MAT_DENS' in globals():
            del MAT_DENS
        if 'FILEICE' in globals():
            del FILEICE
        if 'FILEWATER' in globals():
            del FILEWATER    
        if 'FILEGRAVITY' in globals():
            del FILEGRAVITY
        if 'FILELAYER1' in globals():
            del FILELAYER1   
        if 'FILELAYER2' in globals():
            del FILELAYER2
        self.lineinput_gravity_file.setText("")
        self.lineinput_ice_layer.setText("")
        self.lineinput_water_layer.setText("")
        self.lineinput_layer1.setText("")
        self.lineinput_layer2.setText("")
        self.linenumber_gravity_data.setText("")
        self.linenumber_ice_layer.setText("")
        self.linenumber_water_layer.setText("")
        self.linenumber_layer1.setText("")
        self.linenumber_layer2.setText("")
        self.linegravity_data_resolution.setText("")
        self.lineice_layer_resolution.setText("")
        self.linewater_layer_resolution.setText("")
        self.linelayer1_resolution.setText("")
        self.linelayer2_resolution.setText("")
        self.actionRun_density_inversion.setDisabled(True)
        self.actionSave_output_file.setDisabled(True)     
        self.pushButton_run_inversion.setDisabled(True)
        
    def exit_program(self):
        pyplot.close('all')
        sys.exit(0)             
            
            
if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    dmw = DesignerMainWindow()
    dmw.show()
    sys.exit(app.exec_())            
            

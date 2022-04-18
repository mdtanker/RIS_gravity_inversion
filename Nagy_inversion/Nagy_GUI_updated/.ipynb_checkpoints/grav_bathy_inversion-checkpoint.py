# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:01:01 2020

@author: matthewt
"""

# import python packages
import sys
import numpy as np
import matplotlib as mpl
import statistics as stat
from matplotlib import pyplot as plt
from matplotlib import axes
import csv
import pandas as pd
import scipy as sp
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
import mpl_toolkits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import Rbf
import matplotlib.gridspec as gridspec
from numpy import linalg
from numpy.linalg import lstsq
from scipy.sparse.linalg import lsqr
import gravbox
import warnings
from datetime import date
from grav_column_der import grav_column_der
from scipy import stats 

# import GUI program (QtDesign) packages
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTextEdit

#import .py geometry file from Qt Designer
import grav_bathy_inversion_GUI_geometry    #<---- add .py geometry file

from scipy.ndimage import filters
warnings.filterwarnings('ignore')

class InversionMainWindow(QMainWindow, grav_bathy_inversion_GUI_geometry.Ui_MainWindow):      #<---- create a class "InversionMainWindow" that will inherit all properties of QMainWindow from our geometry file
# use Object Oriented Programming because GUI's are interactive, so MainWindow will need to change based on user input
    
    def __init__(self):        # runs all sub-fucntions everytime something is done in the MainWindow
        # super will return parent object, InversionMainWindow
        super(InversionMainWindow, self).__init__()
        # calls to the geometry setup function defined in the geometry.py file
        self.setupUi(self)
        self.connectActions()

    def main(self):
        self.show()


    def connectActions(self):
        # from the geometry file, copy self.action"button_name" lines to connect the action of clicking the different buttons to specified functions, in the code below
        # change this line from geometry file:
            # self.actionInput_Gravity_Data.setObjectName("actionInput_Gravity_Data")
        # to this line
            # self.actionInput_Gravity_Data.triggered.connect(self.Input_Gravity_File_Func)
        # input data files
        self.actionInput_Gravity_Data.triggered.connect(self.Input_Gravity_Data_Func)
        self.actionInput_Ice_Surface.triggered.connect(self.Input_Ice_Surface_Func)
        self.actionInput_Water_Surface.triggered.connect(self.Input_Water_Surface_Func)
        self.actionInput_Bathymetry.triggered.connect(self.Input_Bathymetry_Func)
        self.actionInput_Basement_Surface.triggered.connect(self.Input_Basement_Surface_Func)
        self.actionInput_Moho.triggered.connect(self.Input_Moho_Func)
        self.pushButton_plot_input_data.clicked.connect(self.Grid_and_plot_input_data_Func) # plot and grid input data
        self.pushButton_plot_gravity_grids.clicked.connect(self.Grid_and_plot_gravity_grids_Func) # plot and grid gravity grids
        self.actionClear_Memory.triggered.connect(self.Clear_Memory_Func) # clear program memory
        self.actionSave_Input_Files.triggered.connect(self.Save_Input_Files_Func) # save inputs
        self.actionLoad_Saved_Input_Files.triggered.connect(self.Load_Saved_Input_Files_Func) # load saved inputs
        self.pushButton_run_inversion.clicked.connect(self.Run_Gravity_Inversion_Func) # run inversion
        #self.actionLoad_Saved_Input_Files.trigger() # automatically load files

    def Clear_Memory_Func(self):
        global FILEICE,FILEWATER,FILEBATHYMETRY,FILEBASEMENT,FILEMOHO,FILEGRAVITY
        
        plt.close('all')
        
        if 'FILEICE' in globals():
            del FILEICE
        if 'FILEWATER' in globals():
            del FILEWATER
        if 'FILEBATHYMETRY' in globals():
            del FILEBATHYMETRY
        if 'FILEBASEMENT' in globals():
            del FILEBASEMENT
        if 'FILEMOHO' in globals():
            del FILEMOHO
        if 'FILEGRAVITY' in globals():
            del FILEGRAVITY
        self.line_input_gravity_data_file.setText('')
        self.line_input_ice_surface_file.setText('')
        self.line_input_water_surface_file.setText('')
        self.line_input_bathymetry_file.setText('')
        self.line_input_basement_surface_file.setText('')
        self.line_input_moho_file.setText('')
        self.line_gravity_size.setText('')
        self.line_bathymetry_size.setText('')
        self.line_ice_size.setText('')
        self.line_water_size.setText('')
        self.line_basement_size.setText('')
        self.line_moho_size.setText('')
    
        print("Memory cleared, previously saved inputs still available as 'Saved_Input_Files.csv'")
        self.message_box.setText("Memory cleared, previously saved inputs still available as 'Saved_Input_Files.csv'") 
       
        
    def Load_Saved_Input_Files_Func(self):
    # define all the function for the above actions     
        # loads previously saved input files
        # should be able to make this more consise with OOP
        global FILEGRAVITY, FILEICE, FILEWATER, FILEBATHYMETRY, FILEBASEMENT, FILEMOHO
        global XG,YG,ZG,FA,FACP,NG
        global XI,YI,ZI,RHOI,CPI,NI
        global XW,YW,ZW,RHOW,CPW,NW
        global XBath,YBath,ZBath,RHOBath,CPBath,NBath
        global XBase,YBase,ZBase,RHOBase,CPBase,NBase
        global XM,YM,ZM,RHOM,CPM,NM
        global INPUT_ICE
        # if file already exits, then will continue to else statement, if it doesn't exit, will print "no saved file" and won't load anything
        try:
            with open('Saved_Input_Files.csv') as f:
                print('Saved previous inputs found!\n')
                self.message_box.setText('Saved previous inputs found!')
        except FileNotFoundError:
            print('no saved file')
            self.message_box.setText('no saved file')
        else:
            saved_inputs = []      # create empty list to fill with file names
            with open('Saved_Input_Files.csv') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        saved_inputs.append(row)  # reads the existing file, and writes the rows into a list
            FILEGRAVITY=saved_inputs[0][0]        # sets FILE_____ variable equal to correct position in list
            print("Gravity:     " + FILEGRAVITY) 
            FILEICE=saved_inputs[1][0]
            print("\nIce:     " + FILEICE)
            FILEWATER=saved_inputs[2][0]
            print("\nWater:     " + FILEWATER)
            FILEBATHYMETRY=saved_inputs[3][0]
            print("\nBathymetry:     " + FILEBATHYMETRY)
            FILEBASEMENT=saved_inputs[4][0]
            print("\nBasement:     " + FILEBASEMENT)
            FILEMOHO=saved_inputs[5][0]
            print("\nMoho:     " + FILEMOHO + "\nDone Importing")
            # if FILE_____ exists, will add that file name and path to the GUI input field
            if FILEGRAVITY:
                INPUT_GRAVITY=pd.read_csv(FILEGRAVITY, header=None, index_col=None, sep=',', names=('Y','X','Z','FA','FACP') )
                YG=np.array(INPUT_GRAVITY.Y) # 1st col, Easting = Y (geophysics convention?)
                XG=np.array(INPUT_GRAVITY.X) # 2nd col, Northing = X (geophysics convention?)
                ZG=-np.array(INPUT_GRAVITY.Z) # 3rd col, survey elevation
                FA=np.array(INPUT_GRAVITY.FA) # 4th col, Free air grav gridded
                FACP=np.array(INPUT_GRAVITY.FACP) #5th col, Free air grav gridded with only bathy control points
                NG=len(XG)      #<---- gives size of input data file
                self.line_input_gravity_data_file.setText(FILEGRAVITY) # adds text of filename that was just imported
                self. line_gravity_size.setText(str(NG)+" points")   # for adding box with number of point in gravity
            if FILEICE:
                INPUT_ICE=pd.read_csv(FILEICE, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
                XI=np.array(INPUT_ICE.X) 
                YI=np.array(INPUT_ICE.Y) 
                ZI=-np.array(INPUT_ICE.Z) 
                RHOI=np.array(INPUT_ICE.RHO) 
                CPI=np.array(INPUT_ICE.CP) 
                NI=len(XI)      
                self.line_input_ice_surface_file.setText(FILEICE)
                self. line_ice_size.setText(str(NI)+" points")
            if FILEWATER:
                INPUT_WATER=pd.read_csv(FILEWATER, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
                XW=np.array(INPUT_WATER.X) 
                YW=np.array(INPUT_WATER.Y) 
                ZW=-np.array(INPUT_WATER.Z) 
                RHOW=np.array(INPUT_WATER.RHO) 
                CPW=np.array(INPUT_WATER.CP) 
                NW=len(XW)      
                self.line_input_water_surface_file.setText(FILEWATER)
                self.line_water_size.setText(str(NW)+" points")
            if FILEBATHYMETRY:
                INPUT_BATHYMETRY=pd.read_csv(FILEBATHYMETRY, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
                XBath=np.array(INPUT_BATHYMETRY.X) 
                YBath=np.array(INPUT_BATHYMETRY.Y) 
                ZBath=-np.array(INPUT_BATHYMETRY.Z) 
                RHOBath=np.array(INPUT_BATHYMETRY.RHO) 
                CPBath=np.array(INPUT_BATHYMETRY.CP) 
                NBath=len(XBath)      
                self.line_input_bathymetry_file.setText(FILEBATHYMETRY)
                self.line_bathymetry_size.setText(str(NBath)+" points")
            if FILEBASEMENT:
                INPUT_BASEMENT=pd.read_csv(FILEBASEMENT, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
                XBase=np.array(INPUT_BASEMENT.X) 
                YBase=np.array(INPUT_BASEMENT.Y) 
                ZBase=-np.array(INPUT_BASEMENT.Z) 
                RHOBase=np.array(INPUT_BASEMENT.RHO) 
                CPBase=np.array(INPUT_BASEMENT.CP) 
                NBase=len(XBase)      
                self.line_input_basement_surface_file.setText(FILEBASEMENT)
                self.line_basement_size.setText(str(NBase)+" points")
            if FILEMOHO:
                INPUT_MOHO=pd.read_csv(FILEMOHO, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
                XM=np.array(INPUT_MOHO.X) 
                YM=np.array(INPUT_MOHO.Y) 
                ZM=-np.array(INPUT_MOHO.Z) 
                RHOM=np.array(INPUT_MOHO.RHO) 
                CPM=np.array(INPUT_MOHO.CP) 
                NM=len(XM) 
                self.line_input_moho_file.setText(FILEMOHO)
                self.line_moho_size.setText(str(NM)+" points")


    def Input_Gravity_Data_Func(self):
        global XG,YG,ZG,FA,FACP,NG           #<---- makes variable you define within function available outside of function
        global FILEGRAVITY, FILEGRAVITYSAVED
        if 'FILEGRAVITY' in globals():
            del FILEGRAVITY       #<---- deletes the file if there's already 1 loaded
            self.line_input_gravity_data_file.setText("")      #<---- empties the textbox
            self.line_gravity_size.setText("")                  #<---- empties the textbox
        FILEGRAVITY = QFileDialog.getOpenFileName(self)[0]     #<---- FILEGRAVITY will become the path to your gravity data .txt file you chose in the GUI
        if FILEGRAVITY:
            INPUT_GRAVITY=pd.read_csv(FILEGRAVITY, header=None, index_col=None, sep=',', names=('Y','X','Z','FA','FACP') )
            XG=np.array(INPUT_GRAVITY.X) # 2nd col, Northing = X (geophysics convention?)
            YG=np.array(INPUT_GRAVITY.Y) # 1st col, Easting = Y (geophysics convention?)
            ZG=-np.array(INPUT_GRAVITY.Z) # 3rd col, survey elevation
            FA=np.array(INPUT_GRAVITY.FA) # 4th col, Free air grav gridded
            FACP=np.array(INPUT_GRAVITY.FACP) #5th col, Free air grav gridded with only bathy control points
            NG=len(XG)      #<---- gives size of input data file
            self.line_input_gravity_data_file.setText(FILEGRAVITY) # adds text of filename that was just imported
            self. line_gravity_size.setText(str(NG)+" points")   # for adding box with number of point in gravity
            
            print("Successfully Loaded!")
            self.message_box.setText("Successfully Loaded!")


    def Input_Ice_Surface_Func(self):
        global XI,YI,ZI,RHOI,CPI,NI
        global FILEICE
        if 'FILEICE' in globals():
            del FILEICE
            self.line_input_ice_surface_file.setText("")
            self.line_ice_size.setText("")
        FILEICE = QFileDialog.getOpenFileName(self)[0]
        if FILEICE:
            INPUT_ICE=pd.read_csv(FILEICE, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
            XI=np.array(INPUT_ICE.X) 
            YI=np.array(INPUT_ICE.Y) 
            ZI= - np.array(INPUT_ICE.Z) 
            RHOI=np.array(INPUT_ICE.RHO) 
            CPI=np.array(INPUT_ICE.CP) 
            NI=len(XI)
            self.line_input_ice_surface_file.setText(FILEICE)
            self.line_ice_size.setText(str(NI)+" points")
            print("Successfully Loaded!")
            self.message_box.setText("Successfully Loaded!")


    def Input_Water_Surface_Func(self):
        global XW,YW,ZW,RHOW,CPW,NW
        global FILEWATER
        if 'FILEWATER' in globals():
            del FILEWATER
            self.line_input_water_surface_file.setText("")
            self.line_water_size.setText("")
        FILEWATER = QFileDialog.getOpenFileName(self)[0]
        if FILEWATER:
            INPUT_WATER=pd.read_csv(FILEWATER, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
            XW=np.array(INPUT_WATER.X) 
            YW=np.array(INPUT_WATER.Y) 
            ZW=-np.array(INPUT_WATER.Z) 
            RHOW=np.array(INPUT_WATER.RHO) 
            CPW=np.array(INPUT_WATER.CP) 
            NW=len(XW)
            self.line_input_water_surface_file.setText(FILEWATER)
            self.line_water_size.setText(str(NW)+" points")
            print("Successfully Loaded!")
            self.message_box.setText("Successfully Loaded!")


    def Input_Bathymetry_Func(self):
        global XBath,YBath,ZBath,RHOBath,CPBath,NBath
        global FILEBATHYMETRY
        if 'FILEBATHYMETRY' in globals():
            del FILEBATHYMETRY
            self.line_input_bathymetry_file.setText("")
            self.line_bathymetry_size.setText("")
        FILEBATHYMETRY = QFileDialog.getOpenFileName(self)[0]
        if FILEBATHYMETRY:
            INPUT_BATHYMETRY=pd.read_csv(FILEBATHYMETRY, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
            XBath=np.array(INPUT_BATHYMETRY.X) 
            YBath=np.array(INPUT_BATHYMETRY.Y) 
            ZBath=-np.array(INPUT_BATHYMETRY.Z) 
            RHOBath=np.array(INPUT_BATHYMETRY.RHO) 
            CPBath=np.array(INPUT_BATHYMETRY.CP) 
            NBath=len(XBath)      
            self.line_input_bathymetry_file.setText(FILEBATHYMETRY)
            self.line_bathymetry_size.setText(str(NBath)+" points")
            print("Successfully Loaded!")
            self.message_box.setText("Successfully Loaded!")


    def Input_Basement_Surface_Func(self):
        global XBase,YBase,ZBase,RHOBase,CPBase,NBase
        global FILEBASEMENT
        if 'FILEBASEMENT' in globals():
            del FILEBASEMENT
            self.line_input_basement_surface_file.setText("")
            self.line_basement_size.setText("")
        FILEBASEMENT = QFileDialog.getOpenFileName(self)[0]
        if FILEBASEMENT:
            INPUT_BASEMENT=pd.read_csv(FILEBASEMENT, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
            XBase=np.array(INPUT_BASEMENT.X) 
            YBase=np.array(INPUT_BASEMENT.Y) 
            ZBase=-np.array(INPUT_BASEMENT.Z) 
            RHOBase=np.array(INPUT_BASEMENT.RHO) 
            CPBase=np.array(INPUT_BASEMENT.CP) 
            NBase=len(XBase) 
            self.line_input_basement_surface_file.setText(FILEBASEMENT)
            self.line_basement_size.setText(str(NBase)+" points")
            self.message_box.setText("Successfully Loaded!")


    def Input_Moho_Func(self):
        global XM,YM,ZM,RHOM,CPM,NM
        global FILEMOHO
        if 'FILEMOHO' in globals():
            del FILEMOHO
            self.line_input_moho_file.setText("")
            self.line_moho_size.setText("")
        FILEMOHO = QFileDialog.getOpenFileName(self)[0]
        if FILEMOHO:
            INPUT_MOHO=pd.read_csv(FILEMOHO, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
            XM=np.array(INPUT_MOHO.X) 
            YM=np.array(INPUT_MOHO.Y) 
            ZM=-np.array(INPUT_MOHO.Z) 
            RHOM=np.array(INPUT_MOHO.RHO) 
            CPM=np.array(INPUT_MOHO.CP) 
            NM=len(XM) 
            self.line_input_moho_file.setText(FILEMOHO)
            self.line_moho_size.setText(str(NM)+" points")
            print("Successfully Loaded!")
            self.message_box.setText("Successfully Loaded!")


    def Grid_and_plot_input_data_Func(self):
        # make input data columns global
        global XG,YG,ZG,FA,FACP,NG
        global XI,YI,ZI,RHOI,CPI,NI
        global XW,YW,ZW,RHOW,CPW,NW
        global XBath,YBath,ZBath,RHOBath,CPBath,NBath
        global XBase,YBase,ZBase,RHOBase,CPBase,NBase
        global XM,YM,ZM,RHOM,CPM,NM
        global RESG, RESI, RESW, RESBATH, RESBASE, RESM
        global INDEXI, INDEXW, INDEXBath, INDEXBase, INDEXM
         
        #converts input resolutions to floats and assigns to variables
        try:
             RESG=int(self.line_gravity_resolution.text())
        except ValueError:
            print("PLEASE ENTER RESOLUTION VALUE, default is 10,000m\n")
            self.message_box.setText("PLEASE ENTER RESOLUTION VALUE, default is 10,000 m")
            RESG=10000
        try:
             RESI=int(self.line_ice_resolution.text())
        except ValueError:
            print("PLEASE ENTER RESOLUTION VALUE, default is 10,000m\n")
            self.message_box.setText("PLEASE ENTER RESOLUTION VALUE, default is 10,000 m")
            RESI=10000

        try:
            RESW=int(self.line_water_resolution.text())
        except ValueError:
            print("PLEASE ENTER RESOLUTION VALUE, default is 10,000m\n")
            self.message_box.setText("PLEASE ENTER RESOLUTION VALUE, default is 10,000 m")
            RESW=10000

        try:
            RESBATH=int(self.line_bathymetry_resolution.text())
        except ValueError:
            print("PLEASE ENTER RESOLUTION VALUE, default is 10,000m\n")
            self.message_box.setText("PLEASE ENTER RESOLUTION VALUE, default is 10,000 m")
            RESBATH=10000

        try:
             RESBASE=int(self.line_basement_resolution.text())
        except ValueError:
            print("PLEASE ENTER RESOLUTION VALUE, default is 10,000m\n")
            self.message_box.setText("PLEASE ENTER RESOLUTION VALUE, default is 10,000 m")
            RESBASE=10000

        try:
            RESM=int(self.line_moho_resolution.text())
        except ValueError:
            print("PLEASE ENTER RESOLUTION VALUE, default is 10,000m\n")
            self.message_box.setText("PLEASE ENTER RESOLUTION VALUE, default is 10,000 m")
            RESM=10000
        print("Done converting resolutions to int\n")
        plt.close()
        self.message_box.setText("Preparing grids")

        # number of control points in Bathy = sum of all numbers in CPBath (CP=1, non CP=0)
        NCPBath = int(np.sum(CPBath))
        NCPBase = int(np.sum(CPBase))
        NCPM = int(np.sum(CPM))
        print('done counting CPs')
        # coords of control points = array of zeros of length total number of control points
        XCBath = np.zeros(NCPBath)
        YCBath = np.zeros(NCPBath)
        XCBase = np.zeros(NCPBase)
        YCBase = np.zeros(NCPBase)
        XCM = np.zeros(NCPM)
        YCM = np.zeros(NCPM)
        print('done creating CP arrays')
        # index of layer = array of zeros of the length of the file
        INDEXBath=np.zeros(NBath)
        INDEXBase=np.zeros(NBase)
        INDEXM=np.zeros(NM)
        print('done creating index')
        # 
        ICPBath=-1
        for i in range(0,NBath):          # iterate over every point in Bath
            if CPBath[i] > 0.5:           # if the control point value is >.5
                ICPBath=ICPBath+1             #
                XCBath[ICPBath]=XBath[i]      # the X and Y coords are added to the respective index of XCBath 
                YCBath[ICPBath]=YBath[i]     
            INDEXBath[i]=i
        print('done indexing Bath')
        ICPBase=-1
        for i in range(0,NBase):
            if CPBase[i] > 0.5:
                ICPBase=ICPBase+1
                XCBase[ICPBase]=XBase[i]
                YCBase[ICPBase]=YBase[i]
            INDEXBase[i]=i
        print('done indexing Base')    
        ICPM=-1
        for i in range(0,NM):
            if CPM[i] > 0.5:
                ICPM=ICPM+1
                XCM[ICPM]=XM[i]
                YCM[ICPM]=YM[i]
            INDEXM[i]=i
        print('done indexing Moho')
    # Gridding / Plotting Input Layers
        # plotting in a for-loop
        print('starting plotting')
        plt.close('all')
        for i in range(0,6):
            title = ('Ice Surface Height', 'Water Surface Height', 'Bathymetry Depth', 'Basement Depth', 'Moho Depth', 'Obseved Gravity Anomaly')
            label = ('(meters)', '(meters)', '(meters)', '(meters)', '(meters)', '(mGals)')
            ax = ('ax1', 'ax2', 'ax3', 'ax4', 'ax5', 'ax6')
            ax = list(ax)
            cax = ('cax1', 'cax2', 'cax3', 'cax4', 'cax5', 'cax6')
            cax = list(cax)
            img = ('img1', 'img2', 'img3', 'img4', 'img5', 'img6')
            img = list(img)
            X = (XI, XW, XBath, XBase, XM, XG)
            Y = (YI, YW, YBath, YBase, YM, YG)
            Z = (ZI, ZW, ZBath, ZBase, ZM, FA)
            RES = (RESI, RESW, RESBATH, RESBASE, RESM, RESG)
            XC = (XCBath, XCBase, XCM)
            YC = (YCBath, YCBase, YCM)

            X_range=np.arange(min(X[i]),max(X[i])+0.0001,RES[i])
            Y_range=np.arange(min(Y[i]),max(Y[i])+0.0001,RES[i])
            East_grid, North_grid = np.meshgrid(Y_range,X_range)

            extent = x_min, x_max, y_min, y_max= [min(Y[i]), max(Y[i]), min(X[i]), max(X[i])]
            grid_Z = griddata((Y[i], X[i]), Z[i], (East_grid, North_grid), method='linear')
            
            fig1 = plt.figure(1, (10,10))
            ax[i] = fig1.add_subplot(3,2,i+1, adjustable='box', aspect=1)
           
            img[i] = ax[i].contourf(grid_Z, 100, cmap='jet', extent=extent)
            
            if i > 1 and i < 5:
                ax[i].plot(YC[i-2], XC[i-2], 'ko', markersize=2)
            else:
                pass
            
            ax[i].set_title(title[i])
            ax[i].set_xlabel('Easting (Km)')
            ax[i].set_ylabel('Northing (Km)')
            
            ax[i].set_ylim(-1385000,-485000)
            ax[i].set_xlim(-555000,345000)
            ax[i].set_yticks( ticks = np.arange(min(XI)+30000, max(XI)+30000, 100000))
            ax[i].set_yticklabels(labels = np.arange(int(0.001*(min(XI)+30000)), int(0.001*(max(XI)+30000)), int(100)))
            ax[i].set_xticks( ticks = np.arange(min(YI+40000), max(YI)+40000, 200000))
            ax[i].set_xticklabels(labels = np.arange(int(0.001*(min(YI)+40000)), int(0.001*(max(YI)+40000)), int(200)))
            #ax[i].margins(2,2)
            
            divider  = make_axes_locatable(ax[i])
            cax[i] = divider.append_axes('right', size='5%', pad = 0.1)
            cb1 = fig1.colorbar(img[i], label=label[i], cax=cax[i])
            cb1.set_ticks( ticks = np.arange(min(Z[i]), max(Z[i]), (abs(max(Z[i])-min(Z[i])))/10))
            
        plt.suptitle('Input Geologic Model Layers', fontsize='xx-large', y=1.02) 
        plt.tight_layout()
        plt.show()


    def Grid_and_plot_gravity_grids_Func(self):
        # make input data columns global
        global XG,YG,ZG,FA,FACP,NG
        global XI,YI,ZI,RHOI,CPI,NI
        global XW,YW,ZW,RHOW,CPW,NW
        global XBath,YBath,ZBath,RHOBath,CPBath,NBath
        global XBase,YBase,ZBase,RHOBase,CPBase,NBase
        global XM,YM,ZM,RHOM,CPM,NM
        global Z2I, Z2W, Z2Bath, Z2Base, Z2M
        global FA_Regional, FA_Residual, FA_CP_Tot,FA_CPI, FA_CPW, FA_CPBath, FA_CPBase, FA_CPM
        global RESG, RESI, RESW, RESBATH, RESBASE, RESM
        global East_gridG, North_gridG, XG_range, YG_range, extentG
        global Ice, Water, Bathymetry, Basement, Moho
        global RHOI_W, RHOW_Bath, RHOBath_Base, RHOBase_M
        global NGX, NGY
        global Base_GRIDVEC, Bath_GRIDVEC, M_GRIDVEC
        global GRID_POINTS
        global INDEXBath, INDEXBase, INDEXM
        

        #converts input resolutions to floats and assigns to variables
        try:
             RESG=int(self.line_gravity_resolution.text())
        except ValueError:
            print("PLEASE ENTER RESOLUTION VALUE, default is 10,000m")
            self.message_box.setText("PLEASE ENTER RESOLUTION VALUE, default is 10,000 m")
            RESG=10000
        try:
             RESI=int(self.line_ice_resolution.text())
        except ValueError:
            print("PLEASE ENTER RESOLUTION VALUE, default is 10,000m")
            self.message_box.setText("PLEASE ENTER RESOLUTION VALUE, default is 10,000 m")
            RESI=10000

        try:
            RESW=int(self.line_water_resolution.text())
        except ValueError:
            print("PLEASE ENTER RESOLUTION VALUE, default is 10,000m")
            self.message_box.setText("PLEASE ENTER RESOLUTION VALUE, default is 10,000 m")
            RESW=10000
        try:
            RESBATH=int(self.line_bathymetry_resolution.text())
        except ValueError:
            print("PLEASE ENTER RESOLUTION VALUE, default is 10,000m")
            self.message_box.setText("PLEASE ENTER RESOLUTION VALUE, default is 10,000 m")
            RESBATH=10000
        try:
             RESBASE=int(self.line_basement_resolution.text())
        except ValueError:
            print("PLEASE ENTER RESOLUTION VALUE, default is 10,000m")
            self.message_box.setText("PLEASE ENTER RESOLUTION VALUE, default is 10,000 m")
            RESBASE=10000
        try:
            RESM=int(self.line_moho_resolution.text())
        except ValueError:
            print("PLEASE ENTER RESOLUTION VALUE, default is 10,000m")
            self.message_box.setText("PLEASE ENTER RESOLUTION VALUE, default is 10,000 m")
            RESM=10000
        print("Done converting resolutions to int")

    # Gridding Residual Gravity Anomaly
        self.message_box.setText("Preparing grids")
        print("Preparing grids")

        # gridding preliminary data
        print("Gridding preliminary data")
        XG_range=np.arange(min(XG),max(XG)+0.0001,RESG)
        YG_range=np.arange(min(YG),max(YG)+0.0001,RESG)
        East_gridG, North_gridG = np.meshgrid(YG_range,XG_range)

        RHOIGRID=griddata((YI,XI),RHOI,(East_gridG, North_gridG), method='linear')
        RHOWGRID=griddata((YW,XW),RHOW,(East_gridG, North_gridG), method='linear')
        RHOBathGRID=griddata((YBath,XBath),RHOBath,(East_gridG, North_gridG), method='linear')
        RHOBaseGRID=griddata((YBase,XBase),RHOBase,(East_gridG, North_gridG), method='linear')
        
        I_GRID=griddata((YI,XI),-ZI,(East_gridG, North_gridG), method='linear')
        W_GRID=griddata((YW,XW),-ZW,(East_gridG, North_gridG), method='linear')
        Bath_GRID=griddata((YBath,XBath),ZBath,(East_gridG, North_gridG), method='linear')
        Base_GRID=griddata((YBase,XBase),ZBase,(East_gridG, North_gridG), method='linear')
        M_GRID=griddata((YM,XM),ZM,(East_gridG, North_gridG), method='linear')
       
        
        ###########################################################################
        # building vector of control points
        # number of control points in Bathy = sum of all numbers in CPBath (CP=1, non CP=0)
        NCPBath = int(np.sum(CPBath))
        NCPBase = int(np.sum(CPBase))
        NCPM = int(np.sum(CPM))
        
        # coords of control points = array of zeros of length total number of control points
        XCBath = np.zeros(NCPBath)
        YCBath = np.zeros(NCPBath)
        XCBase = np.zeros(NCPBase)
        YCBase = np.zeros(NCPBase)
        XCM = np.zeros(NCPM)
        YCM = np.zeros(NCPM)
        
        # index of layer = array of zeros of the length of the file
        INDEXBath=np.zeros(NBath)
        INDEXBase=np.zeros(NBase)
        INDEXM=np.zeros(NM)
        
        ICPBath=-1
        for i in range(0,NBath):          # iterate over every point in Bath
            if CPBath[i] > 0.5:           # if the control point value is >.5
                ICPBath=ICPBath+1             #
                XCBath[ICPBath]=XBath[i]      # the X and Y coords are added to the respective index of XCBath 
                YCBath[ICPBath]=YBath[i]     
            INDEXBath[i]=i
        ICPBase=-1
        for i in range(0,NBase):
            if CPBase[i] > 0.5:
                ICPBase=ICPBase+1
                XCBase[ICPBase]=XBase[i]
                YCBase[ICPBase]=YBase[i]
            INDEXBase[i]=i 
        ICPM=-1
        for i in range(0,NM):
            if CPM[i] > 0.5:
                ICPM=ICPM+1
                XCM[ICPM]=XM[i]
                YCM[ICPM]=YM[i]
            INDEXM[i]=i
            
        NGX=XG_range.size
        NGY=YG_range.size

        XGRIDVEC=np.zeros([NGX*NGY])
        YGRIDVEC=np.zeros([NGX*NGY])
        
        I_GRIDVEC=np.zeros([NGX*NGY])
        W_GRIDVEC=np.zeros([NGX*NGY])
        Bath_GRIDVEC=np.zeros([NGX*NGY])
        Base_GRIDVEC=np.zeros([NGX*NGY])
        M_GRIDVEC=np.zeros([NGX*NGY])
        
        
        RHOIGRIDVEC=np.zeros([NGX*NGY])
        RHOWGRIDVEC=np.zeros([NGX*NGY])
        RHOBathGRIDVEC=np.zeros([NGX*NGY])
        RHOBaseGRIDVEC=np.zeros([NGX*NGY])
        #RHOMGRIDVEC=np.zeros([NGX*NGY])   # ???? not sure if I need this for the last layer (Moho)
        
        INDGRID=-1
        for ix in range(0,NGX):
            for iy in range(0,NGY):
                INDGRID=INDGRID+1
                XGRIDVEC[INDGRID]=North_gridG[ix,iy]
                YGRIDVEC[INDGRID]=East_gridG[ix,iy]
                RHOIGRIDVEC[INDGRID]=RHOIGRID[ix,iy]
                RHOWGRIDVEC[INDGRID]=RHOWGRID[ix,iy]
                RHOBathGRIDVEC[INDGRID]=RHOBathGRID[ix,iy]
                RHOBaseGRIDVEC[INDGRID]=RHOBaseGRID[ix,iy]
                #RHOMGRIDVEC[INDGRID]=RHOMGRID[ix,iy]     # ???? not sure if I need this for the last layer (Moho)
                I_GRIDVEC[INDGRID]=I_GRID[ix,iy]
                W_GRIDVEC[INDGRID]=W_GRID[ix,iy]
                Bath_GRIDVEC[INDGRID]=-Bath_GRID[ix,iy]
                Base_GRIDVEC[INDGRID]=-Base_GRID[ix,iy]
                M_GRIDVEC[INDGRID]=-M_GRID[ix,iy]     # ???? not sure if I need this for the last layer (Moho)
                
        GRID_POINTS=np.zeros([NGX*NGY,2])
        GRID_POINTS[:,0]=YGRIDVEC
        GRID_POINTS[:,1]=XGRIDVEC

        # Density 
        intfunDENSI=LinearNDInterpolator(GRID_POINTS,RHOIGRIDVEC)
        RHOI_W=intfunDENSI(YW,XW)
        RHOI_W[np.isnan(RHOI_W)]=np.mean(RHOI_W[np.isfinite(RHOI_W)])
        
        intfunDENSW=LinearNDInterpolator(GRID_POINTS,RHOWGRIDVEC)
        RHOW_Bath=intfunDENSW(YBath,XBath)
        RHOW_Bath[np.isnan(RHOW_Bath)]=np.mean(RHOW_Bath[np.isfinite(RHOW_Bath)])
        
        intfunDENSBath=LinearNDInterpolator(GRID_POINTS,RHOBathGRIDVEC)
        RHOBath_Base=intfunDENSBath(YBase,XBase)
        RHOBath_Base[np.isnan(RHOBath_Base)]=np.mean(RHOBath_Base[np.isfinite(RHOBath_Base)])
        
        intfunDENSBase=LinearNDInterpolator(GRID_POINTS,RHOBaseGRIDVEC)
        RHOBase_M=intfunDENSBase(YM,XM)
        RHOBase_M[np.isnan(RHOBase_M)]=np.mean(RHOBase_M[np.isfinite(RHOBase_M)])
        
        #  other ??        
        intfunI=LinearNDInterpolator(GRID_POINTS,I_GRIDVEC)
        Ice=intfunI(YG,XG) 
        
        intfunW=LinearNDInterpolator(GRID_POINTS,W_GRIDVEC)
        Water=intfunW(YG,XG) 
        
        intfunBath=LinearNDInterpolator(GRID_POINTS,Bath_GRIDVEC)
        Bathymetry=intfunBath(YG,XG) 

        intfunBase=LinearNDInterpolator(GRID_POINTS,Base_GRIDVEC)
        Basement=intfunBase(YG,XG) 

        intfunM=LinearNDInterpolator(GRID_POINTS,M_GRIDVEC)
        Moho=intfunM(YG,XG) 

        
        #############################################
# ???????????????????????
        # calculate regional gravity field from control points
        print("Calculating regional gravity field from control points \n")
            # creating new arrays full of zeros of length NG
        FA_CPI=np.zeros(NG)
        FA_CPW=np.zeros(NG)
        FA_CPBath=np.zeros(NG)
        FA_CPBase=np.zeros(NG)
        FA_CPM=np.zeros(NG)
        FA_CP_Tot=np.zeros(NG)
        self.message_box.setText("Calculating regional field")
        TREND_TYPE = str(self.comboBox_gridding.currentText())

        Z2I=0*ZI
        Z2W=0*ZW
        Z2Bath=0*ZBath
        Z2Base=0*ZBase
        Z2M=0*ZM
        
        for i in range(0,NG):
            self.progressBar.setProperty("value", 100*(i+1)/NG)

            # going through gravity data point by point, for each point sum up the result of the function: gravbox
            FA_CPI[i]=np.sum(gravbox.gravbox(XG[i],    # x0 # sum function will cycle through all of the points in the layer
                                        YG[i],    # y0
                                        ZG[i],    # z0
                                        XI-0.5*RESI, # x1
                                        XI+0.5*RESI, # x2
                                        YI-0.5*RESI, # y1
                                        YI+0.5*RESI, # y2
                                        ZI,         # Z
                                        Z2I,         # z2
                                        RHOI         # rho
                                        ))
            FA_CPW[i]=np.sum(gravbox.gravbox(XG[i],
                                        YG[i],
                                        ZG[i],
                                        XW-0.5*RESW,
                                        XW+0.5*RESW,
                                        YW-0.5*RESW,
                                        YW+0.5*RESW,
                                        ZW,
                                        Z2W,
                                        RHOW-RHOI_W
                                        ))
            FA_CPBath[i]=np.sum(gravbox.gravbox(XG[i],
                                           YG[i],
                                           ZG[i],
                                           XBath-0.5*RESBATH,
                                           XBath+0.5*RESBATH,
                                           YBath-0.5*RESBATH,
                                           YBath+0.5*RESBATH,
                                           ZBath,
                                           Z2Bath,
                                           RHOBath-RHOW_Bath
                                           ))
            FA_CPBase[i]=np.sum(gravbox.gravbox(XG[i],
                                           YG[i],
                                           ZG[i],
                                           XBase-0.5*RESBASE,
                                           XBase+0.5*RESBASE,
                                           YBase-0.5*RESBASE,
                                           YBase+0.5*RESBASE,
                                           ZBase,
                                           Z2Base,
                                           RHOBase-RHOBath_Base
                                           ))
            FA_CPM[i]=np.sum(gravbox.gravbox(XG[i],
                                        YG[i],
                                        ZG[i],
                                        XM-0.5*RESM,
                                        XM+0.5*RESM,
                                        YM-0.5*RESM,
                                        YM+0.5*RESM,
                                        ZM,
                                        Z2M,
                                        RHOM-RHOBase_M
                                        ))
            FA_CP_Tot = - FA_CPI - FA_CPW + FA_CPBath + FA_CPBase + FA_CPM

        FA_DCP = FACP - FA_CP_Tot
        
        if TREND_TYPE=="Constant value":
            FA_Regional=np.nanmean(FA_DCP)*np.ones([NG])    # changed mp.mean to np.nanmean to fix issue of not griding Regional or Residual when layers had different cell sizes
        else:
            ATEMP=np.lib.column_stack((np.ones(XG.size), XG, YG))
            C,RESID,RANK,SIGMA=linalg.lstsq(ATEMP,FA_DCP)
            FA_Regional=C[0]*np.ones([NG])+C[1]*XG+C[2]*YG

        FA_Residual = FA-FA_Regional

#################################################
    # Plotting Gravity Grids

        # plotting in a for-loop
        for i in range(0,4):
            title = ('Observed Gravity Anomaly', 'Regional Gravity Trend', 'Residual Gravity Anomaly', 'Gravity Anomaly from Control Points')
            ax = ('ax1', 'ax2', 'ax3', 'ax4')
            ax = list(ax)
            cax = ('cax1', 'cax2', 'cax3', 'cax4')
            cax = list(cax)
            img = ('img1', 'img2', 'img3', 'img4')
            img = list(img)

            Z = (FA, FA_Regional, FA_Residual, FA_CP_Tot)

            XG_range=np.arange(min(XG),max(XG)+0.0001,RESG)
            YG_range=np.arange(min(YG),max(YG)+0.0001,RESG)
            East_gridG, North_gridG = np.meshgrid(YG_range,XG_range)
            extentG = xG_min, xG_max, yG_min, yG_max = [min(YG), max(YG), min(XG), max(XG)]
            
            grid_Z = griddata((YG, XG), Z[i], (East_gridG, North_gridG), method='linear')

            fig2 = plt.figure(2, (10,8))
            ax[i] = fig2.add_subplot(2,2,i+1, adjustable='box', aspect=1)
            img[i] = ax[i].contourf(grid_Z, 100, cmap='jet', extent=extentG)
            ax[i].set_title(title[i])
            ax[i].set_xlabel('Easting (Km)')
            ax[i].set_ylabel('Northing (Km)')
            ax[i].set_ylim(-1340000,-540000)
            ax[i].set_xlim(-495000,305000)
            ax[i].set_yticks( ticks = np.arange(min(XI)+30000, max(XI)+30000, 100000))
            ax[i].set_yticklabels(labels = np.arange(int(0.001*(min(XI)+30000)), int(0.001*(max(XI)+30000)), int(100)))
            ax[i].set_xticks( ticks = np.arange(min(YI+40000), max(YI)+40000, 200000))
            ax[i].set_xticklabels(labels = np.arange(int(0.001*(min(YI)+40000)), int(0.001*(max(YI)+40000)), int(200)))

            divider  = make_axes_locatable(ax[i])
            cax[i] = divider.append_axes('right', size='5%', pad = 0.05)
            cb2 = fig2.colorbar(img[i], label='(mGals)', cax=cax[i])
            if i==1:
                 cb2.set_ticks( ticks = np.arange(0,1,1))
            else:
                #cb2.set_ticks( ticks = np.arange(min(Z[i]), max(Z[i]), (abs(max(Z[i])-min(Z[i])))/10))
                #cb2.set_ticks( ticks = np.arange(min(Z[i]), max(Z[i]), (abs(max(Z[i])-min(Z[i])))/10))
                pass
        #plt.subplots_adjust(left=2, bottom=2, right=20, top=20)
        plt.suptitle('Gravity Grids', fontsize='xx-large', y=1.02)   
        plt.tight_layout()
        plt.show()
        print('Done Plotting Gravity Grids\n')
    

    def Save_Input_Files_Func(self):
        global FILEGRAVITY, FILEICE, FILEWATER, FILEBATHYMETRY, FILEBASEMENT, FILEMOHO
        if 'FILEGRAVITY' in globals():
            saved_inputs = [FILEGRAVITY]
        else:
            saved_inputs = ['']
        if 'FILEICE' in globals():
            saved_inputs.append(FILEICE)
        else:
            saved_inputs.append('')
        if 'FILEWATER' in globals():
            saved_inputs.append(FILEWATER)
        else:
            saved_inputs.append('')
        if 'FILEBATHYMETRY' in globals():
            saved_inputs.append(FILEBATHYMETRY)
        else:
            saved_inputs.append('')
        if 'FILEBASEMENT' in globals():
            saved_inputs.append(FILEBASEMENT)
        else:
            saved_inputs.append('')
        if 'FILEMOHO' in globals():
            saved_inputs.append(FILEMOHO)
        else:
            saved_inputs.append('')

        df = pd.DataFrame(saved_inputs)
        df.to_csv('Saved_Input_Files.csv', header=False, index=False)

        print('Input files saved as "Saved_Input_Files.csv"!')
        self.message_box.setText('Input files saved as "Saved_Input_Files.csv"!')
      
    # INVERSIONS    
    def Run_Gravity_Inversion_Func(self):
        # this will change the layers' surface elevations or densities to minimize gravity misfit between observed and forward model
        global Z2I, Z2W, Z2Bath, Z2Base, Z2M, NGX, NGY, GRID_POINTS, RHOIGRIDVEC, RHOWGRIDVEC, RHOBathGRIDVEC, RHOBaseGRIDVEC
        global RHOI_W, RHOW_Bath, RHOBath_Base, RHOBase_M
        global DENS_CORR, DENS_CORRG
        
        global TOP_ICE,BOT_ICE,BATHYMETRY,BASEMENT
        global FA_Inversion,Z_Corrected_Bathymetry,Z_Corrected_Basement
        global MAT_DENS,DFDENS
        global MATDATPAR
        
        global Ice, Water, Bathymetry, Basement, Moho
        global XG,YG,ZG,FA,FACP,NG
        global XI,YI,ZI,RHOI,CPI,NI
        global XW,YW,ZW,RHOW,CPW,NW
        global XBath,YBath,ZBath,RHOBath,CPBath,NBath
        global XBase,YBase,ZBase,RHOBase,CPBase,NBase
        global XM,YM,ZM,RHOM,CPM,NM
        global RESG, RESI, RESW, RESBATH, RESBASE, RESM
        global FA_Regional, FA_Residual, FA_CP_Tot, FA_CPI, FA_CPW, FA_CPBath, FA_CPBase, FA_CPM
        global INDEXI, INDEXW, INDEXBath, INDEXBase, INDEXM
        global East_gridG, North_gridG, XG_range, YG_range, extentG
        global NGX, NGY
        global FA_Forward_Bath, FA_Forward_Base, FA_Forward_Moho 
        global Base_GRIDVEC, Bath_GRIDVEC, ZSGRIDVEC , M_GRIDVEC 
        global Z_Bath_Base_Moho 
        # below are mostly duplicates
        global YCPI, YCPW, YCPBath, YCPBase, YCPM, XCPI, XCPW, XCPBath, XCPBase, XCPM
        global Surface_correction, leastsquaresol, Grav_Misfit
        try:
            FA_CPBath
        except NameError:
            print('\nInversion needs defined variables, running gravity grids portion of script\n')
            self.message_box.setText('Inversion needs defined variables, runing gravity grids portion of script')
            self.pushButton_plot_gravity_grids.click()
        else:
            print('\nInversion variable defined, continuing with inversion\n')
            self.message_box.setText('Inversion variable defined, continuing with inversion')
       
        Min_sed_thickness=float(self.line_min_sed_thickness.text())
        
        if 'MATDATPAR' in globals():
            del MATDATPAR
        if 'MAT_DENS' in globals():
            del MAT_DENS
        
        ZSGRIDVEC=np.zeros([NGX*NGY])
        
        Tolerance_Bathymetry=float(self.line_bathymetry_variation.text()) 
        Tolerance_Basement=float(self.line_basement_variation.text()) 
        Tolerance_Moho=float(self.line_moho_variation.text())
        
        Tolerance_Bathymetry_Contraints=float(self.line_bathymetry_constraints_variation.text())
        Tolerance_Basement_Contraints=float(self.line_basement_constraints_variation.text())
        Tolerance_Moho_Constraints=float(self.line_moho_constraints_variation.text())
        
        Density_Sediment_Variation=float(self.line_sediment_density_variation.text())
        Density_Crust_Variation=float(self.line_crust_density_variation.text())
        Density_Mantle_Variation=float(self.line_mantle_density_variation.text())
        
        Least_Squares=float(self.line_least_squares.text())
        Tolerance_Least_Squares=float(self.line_least_squares_tolerance.text())
        Max_Iterations=float(self.line_maximum_iterations.text())

        Z_Corrected_Bathymetry=ZBath
        Z_Bath_Base_Moho = np.concatenate((ZBath,ZBase,ZM))
        RHO_Bath_Base_Moho = np.concatenate((RHOBath,RHOBase,RHOM))
        
        MATDATPAR = (np.zeros([NG,NBath+NBase+NM])) # array with NG number of rows and NBath+NBase+NM number of columns
        
        MAT_DENS = (np.zeros([NG,NBath+NBase+NM]))
        
        FA_Inversion = FA - FA_Regional 
        Grav_Misfit=np.zeros([NG])        
        
        CHISQ1=np.Inf  # positive infinity
        DCHISQ=np.Inf  # positibe infinity
        
        ITER=0
        self.message_box.setText("Executing iteration "+str(ITER+1)) 
        while DCHISQ > 1+Tolerance_Least_Squares: # while DCHISQ is greater than 1 + least squares tolerance (0.02)
            ITER=ITER+1 
            print("iteration", ITER)
            FA_Forward=np.zeros([NG])    
            FA_Forward_Bath=np.zeros([NG])
            FA_Forward_Base=np.zeros([NG])
            FA_Forward_Moho=np.zeros([NG])
            for i in range(0,NG):  
                # for each gravity station, will sum up the gravity contributions from each layer's points and add it to a file for each layer
                self.progressBar.setProperty("value", 100*(i+1)/NG)
                FA_Forward_Bath[i]=np.sum(gravbox.gravbox(XG[i],YG[i],ZG[i],     XBath-0.5*RESBATH,XBath+0.5*RESBATH,YBath-0.5*RESBATH,YBath+0.5*RESBATH,    Z_Bath_Base_Moho[0:NBath],    Z2Bath,    RHO_Bath_Base_Moho[0:NBath] - RHOW_Bath)) 
                FA_Forward_Base[i]=np.sum(gravbox.gravbox(XG[i],YG[i],ZG[i],     XBase-0.5*RESBASE,XBase+0.5*RESBASE,YBase-0.5*RESBASE,YBase+0.5*RESBASE,     Z_Bath_Base_Moho[NBath:NBath+NBase],    Z2Base,     RHO_Bath_Base_Moho[NBath:NBath+NBase] - RHOBath_Base))
                FA_Forward_Moho[i]=np.sum(gravbox.gravbox(XG[i],YG[i],ZG[i],   XM-0.5*RESM,XM+0.5*RESM,YM-0.5*RESM,YM+0.5*RESM,    Z_Bath_Base_Moho[NBath+NBase:NBath+NBase+NM],    Z2M,     RHO_Bath_Base_Moho[NBath+NBase:NBath+NBase+NM] - RHOM-RHOBase_M))
                
                FA_Forward=FA_Forward_Bath+FA_Forward_Base+FA_Forward_Moho 
                
                #MATDATPAR is matrix array with NG number of rows and NBath+NBase+NM number of columns
                # uses vertical derivative of gravity to find least squares solution to minize gravity misfit for each grav station
                #SURFACE ELEVATION
                MATDATPAR[i,0:NBath]=grav_column_der(XG[i],YG[i],ZG[i],     XBath,YBath,     Z_Bath_Base_Moho[0:NBath],Z2Bath,     RESBATH,     RHOBath-RHOW_Bath)  # MATDATPAR[i,0:NBath] is i'th row and columns 0 to NBath
                MATDATPAR[i,NBath:NBath+NBase]=grav_column_der(XG[i],YG[i],ZG[i],     XBase,YBase,     Z_Bath_Base_Moho[NBath:NBath+NBase],Z2Base,     RESBASE,     RHOBase-RHOBath_Base)
                MATDATPAR[i,NBath+NBase:NBath+NBase+NM]=grav_column_der(XG[i],YG[i],ZG[i],     XM,YM,     Z_Bath_Base_Moho[NBath+NBase:NBath+NBase+NM],Z2M,     RESM,      RHOM-RHOBase_M)
                #DENSITY
                MAT_DENS[i,0:NBath]=grav_column_der(XG[i],YG[i],ZG[i],      XBath,YBath,     Z_Bath_Base_Moho[0:NBath],     Z2Bath,     RESBATH,     RHOBath-RHOW_Bath)
                MAT_DENS[i,NBath:NBath+NBase]=grav_column_der(XG[i],YG[i],ZG[i],      XBase,YBase,     Z_Bath_Base_Moho[NBath:NBath+NBase],    Z2Base,      RESBASE,      RHOBase-RHOBath_Base)
                MAT_DENS[i,NBath+NBase:NBath+NBase+NM]=grav_column_der(XG[i],YG[i],ZG[i],      XM,YM,      Z_Bath_Base_Moho[NBath+NBase:NBath+NBase+NM],    Z2M,      RESM,       RHOM-RHOBase_M)
                
    
            Grav_Misfit=FA_Inversion-FA_Forward # this is = FA - FA_Regional - FA_Forward
           
            # gives the amount that each column's Z1 needs to change by to have the smallest misfit
            Surface_correction=lsqr(MATDATPAR,Grav_Misfit,show=False)[0] # finds the least-squares solution to MATDATPAR and Grav_Misfit, assigns the first value to Surface_correction
            # gives the amount that each column's density needs to change by to have the smallest misfit
            Density_correction=-lsqr(MAT_DENS,Grav_Misfit,show=False)[0]
            

            # if necesarry correction is greater than tolerance, then correction equals tolerance, if it's less than tolerance, then correct by Surface_correction
            for i in range(0,NBath):
                if Surface_correction[i] > Tolerance_Bathymetry:
                    Surface_correction[i]=Tolerance_Bathymetry
                elif Surface_correction[i] < -Tolerance_Bathymetry:
                    Surface_correction[i]=-Tolerance_Bathymetry
            for i in range(NBath,NBath+NBase):
                if Surface_correction[i] > Tolerance_Basement:
                    Surface_correction[i]=Tolerance_Basement
                elif Surface_correction[i] < -Tolerance_Basement:
                    Surface_correction[i]=-Tolerance_Basement   
            for i in range(NBath+NBase,NBath+NBase+NM):
                if Surface_correction[i] > Tolerance_Moho:
                    Surface_correction[i]=Tolerance_Moho
                elif Surface_correction[i] < -Tolerance_Moho:
                    Surface_correction[i]=-Tolerance_Moho
            
            # if necesarry correction in greater than density tolerance, then correction equals tolerance, if it's less than density tolerance, then correct by Density_correction
            for i in range(0,NBath):
                if Density_correction[i] > Density_Sediment_Variation:
                    Density_correction[i]=Density_Sediment_Variation
                elif Density_correction[i] < -Density_Sediment_Variation:
                    Density_correction[i]=-Density_Sediment_Variation   
            for i in range(NBath,NBath+NBase):
                if Density_correction[i] > Density_Crust_Variation:
                    Density_correction[i]=Density_Crust_Variation
                elif Density_correction[i] < -Density_Crust_Variation:
                    Density_correction[i]=-Density_Crust_Variation
            for i in range(NBath+NBase,NBath+NBase+NM):
                if Density_correction[i] > Density_Mantle_Variation:
                    Density_correction[i]=Density_Mantle_Variation
                elif Density_correction[i] < -Density_Mantle_Variation:
                    Density_correction[i]=-Density_Mantle_Variation
                    
            # resetting the Z values with the above corrections 
            Z_Bath_Base_Moho=Z_Bath_Base_Moho+Surface_correction
            Z_Corrected_Bathymetry=Z_Bath_Base_Moho[0:NBath]   
            Z_Corrected_Basement=Z_Bath_Base_Moho[NBath:NBath+NBase]
            Z_Corrected_Moho=Z_Bath_Base_Moho[NBath+NBase:NBath+NBase+NM]
            # resetting the RHO values with the above corrections 
            RHO_Bath_Base_Moho=RHO_Bath_Base_Moho+Density_correction
            RHOSBathymetry=RHO_Bath_Base_Moho[0:NBath]   
            RHOSBasement=RHO_Bath_Base_Moho[NBath:NBath+NBase]
            RHOSMoho=RHO_Bath_Base_Moho[NBath+NBase:NBath+NBase+NM]
                                                               
            # if change at constrains is greater than allowed, force to be the max amount
            for i in range(0,NBath):
                if CPBath[i]==1:
                    if Z_Corrected_Bathymetry[i] > ZBath[i]+Tolerance_Bathymetry_Contraints:
                        Z_Corrected_Bathymetry[i]=ZBath[i]+Tolerance_Bathymetry_Contraints
                    elif Z_Corrected_Bathymetry[i] < ZBath[i]-Tolerance_Bathymetry_Contraints:
                        Z_Corrected_Bathymetry[i]=ZBath[i]-Tolerance_Bathymetry_Contraints
            for i in range(0,NBase):
                if CPBase[i]==1:
                    if Z_Corrected_Basement[i] > ZBase[i]+Tolerance_Basement_Contraints:
                        Z_Corrected_Basement[i]=ZBase[i]+Tolerance_Basement_Contraints
                    elif Z_Corrected_Basement[i] < ZBase[i]-Tolerance_Basement_Contraints:
                        Z_Corrected_Basement[i]=ZBase[i]-Tolerance_Basement_Contraints
            for i in range(0,NM):
                if CPM[i]==1:
                    if Z_Corrected_Moho[i] > ZM[i]+Tolerance_Moho_Constraints:
                        Z_Corrected_Moho[i]=ZM[i]+Tolerance_Moho_Constraints
                    elif Z_Corrected_Moho[i] < ZM[i]-Tolerance_Moho_Constraints:
                        Z_Corrected_Moho[i]=ZM[i]-Tolerance_Moho_Constraints 
            
            # incase layers are of different resolution, need to project corrected Z1 values onto lower layer's resolution
            # for basement
            Z_Corrected_Bathymetry_Grid=griddata((YBath, XBath), Z_Corrected_Bathymetry, (East_gridG, North_gridG), method='linear')            
            index1=-1
            for x in range(0,NGX):
                for y in range(0,NGY):
                    index1=index1+1
                    Bath_GRIDVEC[index1]=Z_Corrected_Bathymetry_Grid[x,y]       
            intfunZ_Corrected_regrid_Basement=LinearNDInterpolator(GRID_POINTS,Bath_GRIDVEC)
            Z_Bathymetry_reprojected=intfunZ_Corrected_regrid_Basement(YBase,XBase)
                # if basement is the same or above sediment, add min sed thickness to basement
            for i in range(0,NBase):
                if Z_Corrected_Basement[i] > Z_Bathymetry_reprojected[i]:
                    Z_Corrected_Basement[i] = Z_Bathymetry_reprojected[i]+100
            
            # for moho
            Z_Corrected_Basement_Grid=griddata((YM, XM), Z_Corrected_Basement, (East_gridG, North_gridG), method='linear')              
            index2=-1
            for x in range(0,NGX):
                for y in range(0,NGY):
                    index2=index2+1
                    Base_GRIDVEC[index2]=Z_Corrected_Basement_Grid[x,y]       
            intfunZ_Corrected_regrid_Moho=LinearNDInterpolator(GRID_POINTS,Base_GRIDVEC)
            Z_Basement_reprojected=intfunZ_Corrected_regrid_Moho(YM,XM)
                 # if moho is the same or above basement, add min sed thickness to basement
            for i in range(0,NM):
                if Z_Corrected_Moho[i] > Z_Basement_reprojected[i]:
                    Z_Corrected_Moho[i] = Z_Basement_reprojected[i]+100
            
            # for first iteration, divide infinity by mean square of gravity residuals, inversion will stop once this gets to Tolerance_Least_Squares (0.02)
            CHISQ2=np.mean(Grav_Misfit**2)
            DCHISQ=CHISQ1/CHISQ2
            CHISQ1=CHISQ2
            print ('mean of misfit^2 (CHISSQ2)',CHISQ2)
            print("DCHISQ=",DCHISQ)
            print("CHISQ1=",CHISQ1)
 

            print("Iteration "+ str(ITER)+" terminated with least-squares norm="+str(CHISQ2)+", executing iteration "+str(ITER+1))
            self.message_box.setText("Iteration "+ str(ITER)+" terminated with least-squares norm="+str(CHISQ2)+", executing iteration "+str(ITER+1))

            # stop the inversion if hit the max # iterations or it's below the Least Squares norm 
            if ITER==Max_Iterations:
                print("Inversion terminated after "+ str(ITER)+" iterations with least-squares norm="+str(CHISQ2)+ "because maximum number of iterations "+ str(Max_Iterations)+" reached")
                self.message_box.setText("Inversion terminated after "+ str(ITER)+" iterations with least-squares norm="+str(CHISQ2)+"because maximum number of iterations "+ str(Max_Iterations)+" reached") 
                break
            if CHISQ2 < Least_Squares:
                print("Inversion terminated after "+ str(ITER)+" iterations with least-squares norm="+str(CHISQ2)+" because least-squares norm < " + str(Least_Squares))
                self.message_box.setText("Inversion terminated after "+ str(ITER)+" iterations with least-squares norm="+str(CHISQ2)+" because least-squares norm < " + str(Least_Squares)) 
                break
        
            # end of inversion iteration WHILE loop
            
        if DCHISQ < 1+Tolerance_Least_Squares:
            print("terminated - no significant variation in least-squares norm ")
            self.message_box.setText("Inversion terminated after "+ str(ITER)+" iterations with least-squares norm="+str(CHISQ2)+" because tolerance on least-squares norm < " + str(Tolerance_Least_Squares))
       
        Bathymetry_Surface_Diff = ZBath - Z_Corrected_Bathymetry
        Basement_Surface_Diff = ZBase - Z_Corrected_Basement
        Moho_Surface_Diff = ZM - Z_Corrected_Moho
        
        Bathymetry_Density_Diff = RHOBath - RHOSBathymetry
        Basement_Density_Diff = RHOBase - RHOSBasement
        Moho_Density_Diff = RHOM - RHOSMoho
        
        XBath_range=np.arange(min(XBath),max(XBath)+0.0001,RESBATH)
        YBath_range=np.arange(min(YBath),max(YBath)+0.0001,RESBATH)
        East_gridBath, North_gridBath = np.meshgrid(YBath_range,XBath_range)
        
        XBase_range=np.arange(min(XBase),max(XBase)+0.0001,RESBASE)
        YBase_range=np.arange(min(YBase),max(YBase)+0.0001,RESBASE)
        East_gridBase, North_gridBase = np.meshgrid(YBase_range,XBase_range)
        
        XM_range=np.arange(min(XM),max(XM)+0.0001,RESM)
        YM_range=np.arange(min(YM),max(YM)+0.0001,RESM)
        East_gridM, North_gridM = np.meshgrid(YM_range,XM_range)
    
        #################################################
    # Plotting Inversion Grids
        # need 3 figures: 
            # 1: 3x3 with rows: bathy, base, moho,    columns: original, inverted, difference 
            # 2: 3x2 with rows: bathy, base, moho,    columns: original density, new density
            # 3: 3x1 observed grav, forward grav after inversion, misfit
        
        # plotting figure 1 (fig3)in a for-loop
        for i in range(0,9):
            title = ('Original Bathymetry', 'Inverted Bathymetry', 'Bathymetry Difference ',
                     'Original Basement', 'Inverted Basement', 'Basement Difference ',
                     'Original Moho', 'Inverted Moho', 'Moho Difference')
            ax = ('ax1', 'ax2', 'ax3', 'ax4', 'ax5', 'ax6', 'ax7', 'ax8', 'ax9')
            ax = list(ax)
            cax = ('cax1', 'cax2', 'cax3', 'cax4', 'cax5', 'cax6', 'cax7', 'cax8', 'cax9')
            cax = list(cax)
            img = ('img1', 'img2', 'img3', 'img4', 'img5', 'img6', 'img7', 'img8', 'img9')
            img = list(img)
            Z = (ZBath, Z_Corrected_Bathymetry, Bathymetry_Surface_Diff,
                 ZBase, Z_Corrected_Basement, Basement_Surface_Diff, 
                 ZM, Z_Corrected_Moho, Moho_Surface_Diff) 
            X = (XBath, XBath, XBath, 
                 XBase, XBase, XBase, 
                 XM, XM, XM)
            Y = (YBath, YBath, YBath,
                 YBase, YBase, YBase,
                 YM, YM, YM) 
            East_grid = (East_gridBath, East_gridBath, East_gridBath,
                         East_gridBase, East_gridBase, East_gridBase,
                         East_gridM, East_gridM, East_gridM)
            North_grid = (North_gridBath, North_gridBath, North_gridBath,
                          North_gridBase, North_gridBase, North_gridBase,
                          North_gridM, North_gridM, North_gridM)
        
            grid_Z = griddata((Y[i], X[i]), Z[i], (East_grid[i], North_grid[i]), method='linear')
            fig3 = plt.figure(3, (15,12))
            ax[i] = fig3.add_subplot(3,3,i+1, adjustable='box', aspect=1)
            extent = x_min, x_max, y_min, y_max= [min(Y[i]), max(Y[i]), min(X[i]), max(X[i])]
            img[i] = ax[i].contourf(grid_Z, 100, cmap='jet', extent=extent)
            ax[i].set_title(title[i])
            ax[i].set_xlabel('Easting (Km)')
            ax[i].set_ylabel('Northing (Km)')
            ax[i].set_ylim(-1385000,-485000)
            ax[i].set_xlim(-555000,345000)
            ax[i].set_yticks( ticks = np.arange(min(XI)+30000, max(XI)+30000, 100000))
            ax[i].set_yticklabels(labels = np.arange(int(0.001*(min(XI)+30000)), int(0.001*(max(XI)+30000)), int(100)))
            ax[i].set_xticks( ticks = np.arange(min(YI+40000), max(YI)+40000, 200000))
            ax[i].set_xticklabels(labels = np.arange(int(0.001*(min(YI)+40000)), int(0.001*(max(YI)+40000)), int(200)))
            divider  = make_axes_locatable(ax[i])
            cax[i] = divider.append_axes('right', size='5%', pad = 0.05)
            cb3 = fig3.colorbar(img[i], label="(meters)", cax=cax[i])
        plt.suptitle('Layer Surface - Gravity Inversion', fontsize='xx-large', y=1.02) 
        plt.tight_layout()
        plt.show()
        print('\nDone Plotting Inversion Grids #1\n')
      
        ###########################################
         # plotting figure 2 (fig4)in a for-loop

        for i in range(0,9):
            title = ('Original Sediment Density', 'Inverted Sediment Density', 'Sediment Density Difference',
                     'Original Crust Density', 'Inverted Crust Density', 'Crust Density Difference',
                     'Original Mantle Density', 'Inverted Mantle Density', 'Mantle Density Difference')
            ax = ('ax1', 'ax2', 'ax3', 'ax4', 'ax5', 'ax6', 'ax7', 'ax8', 'ax9')
            ax = list(ax)
            cax = ('cax1', 'cax2', 'cax3', 'cax4', 'cax5', 'cax6', 'cax7', 'cax8', 'cax9')
            cax = list(cax)
            img = ('img1', 'img2', 'img3', 'img4', 'img5', 'img6', 'img7', 'img8', 'img9')
            img = list(img)
            Z = (RHOBath, RHOSBathymetry, Bathymetry_Density_Diff,
                 RHOBase, RHOSBasement, Basement_Density_Diff, 
                 RHOM, RHOSMoho, Moho_Density_Diff) # need to swap in Moho terms  
            X = (XBath, XBath, XBath, 
                 XBase, XBase, XBase, 
                 XM, XM, XM)
            Y = (YBath, YBath, YBath,
                 YBase, YBase, YBase,
                 YM, YM, YM) # need to swap in Moho terms
            X_range = (XBath_range, XBath_range, XBath_range,
                       XBase_range, XBase_range,  XBase_range,
                       XM_range, XM_range, XM_range) # need to swap in Moho terms
            Y_range = (YBath_range, YBath_range, YBath_range,
                       YBase_range, YBase_range, YBase_range,
                       YM_range, YM_range, YM_range) # need to swap in Moho terms
            East_grid = (East_gridBath, East_gridBath, East_gridBath,
                         East_gridBase, East_gridBase, East_gridBase,
                         East_gridM, East_gridM, East_gridM)
            North_grid = (North_gridBath, North_gridBath, North_gridBath,
                          North_gridBase, North_gridBase, North_gridBase,
                          North_gridM, North_gridM, North_gridM)
        
            grid_Z = griddata((Y[i], X[i]), Z[i], (East_grid[i], North_grid[i]), method='linear')
            fig4 = plt.figure(4, (15,12))
            ax[i] = fig4.add_subplot(3,3,i+1, adjustable='box', aspect=1)
            extent = x_min, x_max, y_min, y_max= [min(Y[i]), max(Y[i]), min(X[i]), max(X[i])]
            img[i] = ax[i].contourf(grid_Z, 100, cmap='jet', extent=extent)
            ax[i].set_title(title[i])
            ax[i].set_xlabel('Easting (Km)')
            ax[i].set_ylabel('Northing (Km)')
            ax[i].set_ylim(-1385000,-485000)
            ax[i].set_xlim(-555000,345000)
            ax[i].set_yticks( ticks = np.arange(min(XI)+30000, max(XI)+30000, 100000))
            ax[i].set_yticklabels(labels = np.arange(int(0.001*(min(XI)+30000)), int(0.001*(max(XI)+30000)), int(100)))
            ax[i].set_xticks( ticks = np.arange(min(YI+40000), max(YI)+40000, 200000))
            ax[i].set_xticklabels(labels = np.arange(int(0.001*(min(YI)+40000)), int(0.001*(max(YI)+40000)), int(200)))
            divider  = make_axes_locatable(ax[i])
            cax[i] = divider.append_axes('right', size='5%', pad = 0.05)
            cb4 = fig4.colorbar(img[i], label="(g/cm^3)", cax=cax[i])
        plt.suptitle('Density - Gravity Inversion', fontsize='xx-large', y=1.02) 
        plt.tight_layout()
        plt.show()
        print('\nDone Plotting Inversion Grid #2\n')
        
        ###########################################
         # plotting figure 3 (fig5)in a for-loop
        for i in range(0,3):
            title = ('Observed Gravity', 'Forward Gravity Model', 'Gravity Error')
            ax = ('ax1', 'ax2', 'ax3')
            ax = list(ax)
            cax = ('cax1', 'cax2', 'cax3')
            cax = list(cax)
            img = ('img1', 'img2', 'img3')
            img = list(img)
            Z = (FA, FA_Forward+FA_Regional, FA_Inversion-FA_Forward)

            grid_Z = griddata((YG, XG), Z[i], (East_gridG, North_gridG), method='linear')
            fig5 = plt.figure(5, (12,12))
            ax[i] = fig5.add_subplot(3,1,i+1, adjustable='box', aspect=1)
            #extent = x_min, x_max, y_min, y_max= [min(Y[i]), max(Y[i]), min(X[i]), max(X[i])]
            img[i] = ax[i].contourf(grid_Z, 100, cmap='jet', extent=extentG)
            ax[i].set_title(title[i])
            ax[i].set_xlabel('Easting (Km)')
            ax[i].set_ylabel('Northing (Km)')
            ax[i].set_ylim(-1385000,-485000)
            ax[i].set_xlim(-555000,345000)
            ax[i].set_yticks( ticks = np.arange(min(XI)+30000, max(XI)+30000, 100000))
            ax[i].set_yticklabels(labels = np.arange(int(0.001*(min(XI)+30000)), int(0.001*(max(XI)+30000)), int(100)))
            ax[i].set_xticks( ticks = np.arange(min(YI+40000), max(YI)+40000, 200000))
            ax[i].set_xticklabels(labels = np.arange(int(0.001*(min(YI)+40000)), int(0.001*(max(YI)+40000)), int(200)))
            if i == 2:
                ax[i].text(-510000, -530000, 'average misfit: '+str(round(np.mean(FA_Inversion-FA_Forward),3)))
                ax[i].text(-510000, -575000, 'std: '+str(round(stat.stdev(FA_Inversion-FA_Forward),3)))
            divider  = make_axes_locatable(ax[i])
            cax[i] = divider.append_axes('right', size='5%', pad = 0.05)
            cb5 = fig5.colorbar(img[i], label="(mGals)", cax=cax[i])
        plt.suptitle('Gravity Inversion Results', fontsize='xx-large', y=1.02) 
        plt.tight_layout()
        plt.show()
        print('\nDone Plotting Inversion Grid #3\n')

if __name__=='__main__':
    app = QApplication(sys.argv)
    gui = InversionMainWindow()
    gui.show()

    sys.exit(app.exec_())
                  

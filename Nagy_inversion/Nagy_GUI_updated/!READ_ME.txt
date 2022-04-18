1) Create geometry file (sample_geometry.ui) in Qt Designer
	prompt -> designer
	main window -> save as "sample_geometry.ui"
	add "File", "Input File"
	add a Line Edit box to screen

2) Convert .ui file to .py
	prompt -> navigate to file with dir, cd. and cd..
	C:\your_folder> PYUIC5 sample_geometry.ui > sample_geometry.py
	or
	C:\your_folder> PYUIC5 -x sample_geometry.ui -o sample_geometry.py

3) Create Logic file in Spyder, use sample_inversion_logic.py as a guide

4) Logic file will need outside scripts for functions, and these functions may need outside scripts as well:
	lsq_linear.py     <------------- linear least-squares with bound constraints on independent variables
		common.py     <--------- functions used by least-squares algorithms
		trf_linear.py     <----- adaptation of Trust Region Reflective algorithm for a linear least-squares problem
			common.py
		bvls.py     <----------- Bounded-Variable Least-Squares algorithm
			common.py
	gravbox.py     <--------------- function to calculate the gravity field of prism	    
	grav_column_der.py     <------- function to calculate the derivative of a gravity field of a vertical column using with respect to the depth of the top of the column


	


# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:07:36 2022

@author: deepa
"""

import numpy as np
import math
import os
import pandas as pd

path_functions = 'D:\TAMU\Research\Asymmetric_2D_Dubins\Codes'
os.chdir(path_functions)

from Asymmetric_and_weighted_2D_Dubins_functions import *

# Declaring the parameters
rL = 1
rR = 3
muL = 0.5
muR = 4

ini_config = np.array([0, 0, 0])
fin_config = np.array([0, 0, math.pi])

# Generating random initial and final configurations
# ini_heading = np.mod(np.random.rand()*2*math.pi, 2*math.pi)
# fin_heading = np.mod(np.random.rand()*2*math.pi, 2*math.pi)
# fin_loc = np.random.rand()*20
# ini_heading = 3.21501456
# fin_heading = 3.97181889
# fin_loc = 6.59185584

# ini_heading = 0.41557635
# fin_heading = 3.18310131
# fin_loc = 0.54752094
# ini_config = np.array([0, 0, ini_heading])
# fin_config = np.array([fin_loc, 0, fin_heading])

# a, b, c, d, e, f = asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, r_sym, mu_sym, rL, rR, muL, muR)
# a, b, c, d = asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, r_sym, mu_sym, rL, rR, muL, muR)
_, _, opt_path_types, x_unweighted, y_unweighted, x_weighted, y_weighted =\
    asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, rL, rR, muL, muR)

#%% Comparison with different rL and rR values for unweighted and weighted Dubins

import numpy as np
import math
import os
import pandas as pd

path_functions = 'D:\TAMU\Research\Asymmetric_2D_Dubins\Codes'
os.chdir(path_functions)

from Asymmetric_and_weighted_2D_Dubins_functions import *

# Declaring the parameters
rL = 1
rR = 3
muL = 0
muR = 6

ini_config = np.array([0, 0, math.pi/2])
fin_config = np.array([rR, rR, 0])

_, _, opt_path_types, x_unweighted, y_unweighted, x_weighted, y_weighted = asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, rL, rR, muL, muR)

#%% Storing the obtained data in an excel sheet

# name_sheet = 'Teardrop_rL_1_rR_1_muL_0_2_muR_0_2.xlsx'
name_sheet = 'Teardrop_rL_1_rR_1_muL_1_muR_1.xlsx'
# name_sheet = 'Right_turn_rL_1_rR_3_muL_0_muR_6.xlsx'

# Storing the variables in an excel sheet
# Creating data frames corresponding to the configurations
df1 = pd.DataFrame(ini_config, columns = ['Initial config'])
df2 = pd.DataFrame(fin_config, columns = ['Final config'])
# Creating data frames corresponding to the coordinates along the path
df3 = pd.DataFrame(x_unweighted, columns = ['x coord unweighted - ' + str(opt_path_types[0].upper())])
df4 = pd.DataFrame(y_unweighted, columns = ['y coord unweighted - ' + str(opt_path_types[0].upper())])
df5 = pd.DataFrame(x_weighted, columns = ['x coord weighted - ' + str(opt_path_types[1].upper())])
df6 = pd.DataFrame(y_weighted, columns = ['y coord weighted - ' + str(opt_path_types[1].upper())])

# Writing onto an excel sheet
writer = pd.ExcelWriter(name_sheet, engine = 'xlsxwriter')
workbook = writer.book
worksheet = workbook.add_worksheet('Sheet 1')
writer.sheets['Sheet 1'] = worksheet
df1.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 0, index = False)
df2.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 1, index = False)
df3.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 2, index = False)
df4.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 3, index = False) 
df5.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 4, index = False)
df6.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 5, index = False) 
writer.save()
writer.close()

#%% Testing adjoint functions

from Asymmetric_and_weighted_2D_Dubins_functions_adjoint_vars import *

# Declaring the values for the adjoint variables
e = 1
lambda0 = 1.5
phi = -math.pi/4

plotting_adjoint_vars_weighted_unweighted_Dubins_CCC(ini_config, fin_config, r_sym, r_sym, mu_sym, mu_sym, e)

#%% Parametric study

import numpy as np
import math
import os
import pandas as pd

path_functions = 'D:\TAMU\Research\Asymmetric_2D_Dubins\Codes'
os.chdir(path_functions)

from Asymmetric_and_weighted_2D_Dubins_functions import *

alpha_i_range = np.linspace(0, 2*math.pi, 10, endpoint = False)
d_range = np.linspace(0, 20, 10)
# d_range = np.array([0])
alpha_f_range = np.linspace(0, 2*math.pi, 10, endpoint = False)
rR_range = np.linspace(0.5, 3, 10)
muL_range = np.linspace(0.1, 5, 10)
muR_range = np.linspace(0.1, 5, 10)

path_types, count_no_configs_opt_path =\
    parametric_study_weighted_Dubins_paths(alpha_i_range, d_range, alpha_f_range, rR_range,\
                                           muL_range, muR_range)

#%% Writing the data from the parametric study onto an excel file

path_types_mod = [i.upper() for i in path_types]
df1 = pd.DataFrame(path_types_mod, columns = ['Path type'])
df2 = pd.DataFrame(count_no_configs_opt_path, columns = ['Count optimal path for configurations'])
# Saving the variations in the parameters also
df3 = pd.DataFrame(alpha_i_range, columns = ['alpha_i_range'])
df4 = pd.DataFrame(d_range, columns = ['d_range'])
df5 = pd.DataFrame(alpha_f_range, columns = ['alpha_f_range'])
df6 = pd.DataFrame(rR_range, columns = ['rR_range'])
df7 = pd.DataFrame(muL_range, columns = ['muL_range'])
df8 = pd.DataFrame(muR_range, columns = ['muR_range'])

# Writing onto an excel sheet
writer = pd.ExcelWriter('Parametric study weighted Dubins.xlsx', engine = 'xlsxwriter')
workbook = writer.book
worksheet = workbook.add_worksheet('Sheet 1')
writer.sheets['Sheet 1'] = worksheet
df1.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 0, index = False)
df2.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 1, index = False)
df3.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 3, index = False)
df4.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 4, index = False)
df5.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 5, index = False)
df6.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 6, index = False)
df7.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 7, index = False)
df8.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 8, index = False)
writer.save()
writer.close()    

#%% Parametric study with fixed initial and final configuration

import numpy as np
import math
import os
import pandas as pd

path_functions = 'D:\TAMU\Research\Asymmetric_2D_Dubins\Codes'
os.chdir(path_functions)

from Asymmetric_and_weighted_2D_Dubins_functions import *

ini_config = np.array([0, 0, 0])
fin_config = np.array([0, 0, math.pi])
muL = 1
rR_range = np.linspace(0.5, 3, 26)
muR_range = np.linspace(0, 4, 21)

opt_path_types_matrix, path_cost_diff =\
    parametric_study_weighted_Dubins_paths_fixed_config(ini_config, fin_config,\
                                                        rR_range, muL, muR_range)
        
# Storing the dataset in an excel sheet
df2 = pd.DataFrame(rR_range, columns = ['rR \ muR'])
df1 = pd.DataFrame(muR_range)

# Writing to an excel sheet
# writer = pd.ExcelWriter('Parametric study weighted Dubins teardrop muL_0_1.xlsx',\
#                         engine = 'xlsxwriter')
writer = pd.ExcelWriter('Parametric study weighted Dubins teardrop muL_1.xlsx',\
                        engine = 'xlsxwriter')
# writer = pd.ExcelWriter('Parametric study weighted Dubins teardrop_fin_heading_3pi4 muL_0.xlsx',\
#                         engine = 'xlsxwriter')
# writer = pd.ExcelWriter('Parametric study weighted Dubins teardrop_fin_heading_3pi4 muL_0_1.xlsx',\
#                         engine = 'xlsxwriter')
workbook = writer.book
worksheet = workbook.add_worksheet('Sheet 1')
writer.sheets['Sheet 1'] = worksheet
df2.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 0, index = False)
df1.T.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 1,\
               header = False, index = False)

# Storing the rest of the matrix in the excel sheet
# Running through the columns
for i in range(len(muR_range)):
    
    df3 = pd.DataFrame(opt_path_types_matrix[:, i])
    df3.to_excel(writer, sheet_name = 'Sheet 1', startrow = 1, startcol = i + 1,\
                 header = False, index = False)

# Storing the percentage deviations
workbook = writer.book
worksheet = workbook.add_worksheet('Sheet 2')
writer.sheets['Sheet 2'] = worksheet
df2.to_excel(writer, sheet_name = 'Sheet 2', startrow = 0, startcol = 0, index = False)
df1.T.to_excel(writer, sheet_name = 'Sheet 2', startrow = 0, startcol = 1,\
               header = False, index = False)
    
# Storing the rest of the matrix in the excel sheet
# Running through the columns
for i in range(len(muR_range)):
    
    df3 = pd.DataFrame(path_cost_diff[:, i])
    df3.to_excel(writer, sheet_name = 'Sheet 2', startrow = 1, startcol = i + 1,\
                 header = False, index = False)

writer.save()
writer.close()

#%% Parametric study for varying configurations with fixed parameters

import numpy as np
import math
import os
import pandas as pd

path_functions = 'D:\TAMU\Research\Asymmetric_2D_Dubins\Codes'
os.chdir(path_functions)

from Asymmetric_and_weighted_2D_Dubins_functions import *

# Declaring parameters for the parametric study
fin_heading = math.pi/3
rL = 1
rR = 1.2
muL = 0
muR = 0.5
# Declaring variation in x and y coordinates
pos_x_var = np.linspace(-8, 8, 41)
pos_y_var = np.linspace(-8, 8, 41)

opt_path_types_matrix, path_cost\
    = parametric_study_weighted_Dubins_paths_fixed_param_vary_config(pos_x_var, pos_y_var,\
                                                                     fin_heading, rL, rR, muL, muR)
        
# Storing the dataset in an excel sheet
df2 = pd.DataFrame(pos_y_var, columns = ['y \ x'])
df1 = pd.DataFrame(pos_x_var)

# Writing to an excel sheet
# writer = pd.ExcelWriter('Parametric study weighted Dubins config rL_1 r_R_1_2 muL_0_muR_0_1.xlsx',\
#                         engine = 'xlsxwriter')
writer = pd.ExcelWriter('Parametric study weighted Dubins config rL_1 r_R_1_2 muL_0_muR_0_5.xlsx',\
                        engine = 'xlsxwriter')
workbook = writer.book
worksheet = workbook.add_worksheet('Sheet 1')
writer.sheets['Sheet 1'] = worksheet
df2.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 0, index = False)
df1.T.to_excel(writer, sheet_name = 'Sheet 1', startrow = 0, startcol = 1,\
               header = False, index = False)

# Storing the rest of the matrix in the excel sheet
# Running through the columns
for i in range(len(pos_x_var)):
    
    df3 = pd.DataFrame(opt_path_types_matrix[:, i])
    df3.to_excel(writer, sheet_name = 'Sheet 1', startrow = 1, startcol = i + 1,\
                 header = False, index = False)

writer.save()
writer.close()

#%% (Debugging) Case where RLR is not improved upon by RSLSR

import numpy as np
import math
import os
import pandas as pd

path_functions = 'D:\TAMU\Research\Asymmetric_2D_Dubins\Codes'
os.chdir(path_functions)

from Asymmetric_and_weighted_2D_Dubins_functions import *

# Declaring the parameters
rL = 1
rR = 3
muL = 5
muR = 5

ini_config = np.array([0, 0, 0])
fin_config = np.array([0, 0, 4.1887902])

# Generating random initial and final configurations
# ini_heading = np.mod(np.random.rand()*2*math.pi, 2*math.pi)
# fin_heading = np.mod(np.random.rand()*2*math.pi, 2*math.pi)
# fin_loc = np.random.rand()*20
# ini_heading = 3.21501456
# fin_heading = 3.97181889
# fin_loc = 6.59185584

# ini_heading = 0.41557635
# fin_heading = 3.18310131
# fin_loc = 0.54752094
# ini_config = np.array([0, 0, ini_heading])
# fin_config = np.array([fin_loc, 0, fin_heading])

# a, b, c, d, e, f = asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, r_sym, mu_sym, rL, rR, muL, muR)
# a, b, c, d = asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, r_sym, mu_sym, rL, rR, muL, muR)
_, _, opt_path_types, x_unweighted, y_unweighted, x_weighted, y_weighted =\
    asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, rL, rR, muL, muR)
    
#%% Debugging case where RLR was initially obtained to be optimal

import numpy as np
import math
import os
import pandas as pd

path_functions = 'D:\TAMU\Research\Asymmetric_2D_Dubins\Codes'
os.chdir(path_functions)

from Asymmetric_and_weighted_2D_Dubins_functions import *

# Declaring the parameters
rL = 1
d = 0
alpha_i = 1.2566370614359172
alpha_f = 5.026548245743669
rR = 0.5
muL = 0.1
muR = 0.1

ini_config = np.array([0, 0, alpha_i])
fin_config = np.array([d, 0, alpha_f])

# Generating random initial and final configurations
# ini_heading = np.mod(np.random.rand()*2*math.pi, 2*math.pi)
# fin_heading = np.mod(np.random.rand()*2*math.pi, 2*math.pi)
# fin_loc = np.random.rand()*20
# ini_heading = 3.21501456
# fin_heading = 3.97181889
# fin_loc = 6.59185584

# ini_heading = 0.41557635
# fin_heading = 3.18310131
# fin_loc = 0.54752094
# ini_config = np.array([0, 0, ini_heading])
# fin_config = np.array([fin_loc, 0, fin_heading])

# a, b, c, d, e, f = asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, r_sym, mu_sym, rL, rR, muL, muR)
# a, b, c, d = asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, r_sym, mu_sym, rL, rR, muL, muR)
_, _, opt_path_types, x_unweighted, y_unweighted, x_weighted, y_weighted = asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, rL, rR, muL, muR)

#%% Variation in optimal path with variation in penalty for same initial and final
# configurations - obtaining cost

import numpy as np
import math
import os
import pandas as pd

path_functions = 'D:\TAMU\Research\Asymmetric_2D_Dubins\Codes'
os.chdir(path_functions)

from Asymmetric_and_weighted_2D_Dubins_functions import *

# Declaring the parameters
rL = 1
rR = 1.2
muL = 0
muR = 0.1

ini_config = np.array([0, 0, 0])
fin_config = np.array([-3.2, 2, math.pi/3])

_, _, opt_path_types, x_unweighted, y_unweighted, x_weighted, y_weighted =\
    asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, rL, rR, muL, muR)
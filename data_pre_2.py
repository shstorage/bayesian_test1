import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle
import pandas as pd
import seaborn as sns
import data_pre_module_2 as dpm

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)

code_dir = os.path.abspath(".")

###################### 배관 두께측정 참값 데이터 생성 ###########################
# y1: 초기 데이터
# y2: 감육진행 데이터
# N_a = 13  #측정그리드 개수 - 축방향
# N_c = 12  #측정그리드 개수 - 원주방향
# Meas_err 감육측정에러 [행=원주12, 열=축13] [N_c,N_a]
# StdMeas 학습데이터 측정오차 난수

# t_i_1, t_i_2, t_i_3: 감육측정 시기 (첫번째, 두번째, 세번째), 
# 첫번째가 1이고 두번째 세번째는 년단위 숫자ex(1,3(2년뒤), 5(4년뒤))

###############################################################################
#계산용 주요 설정값 입력 -------------------------------------------------------
N_tn = 1000                       #train 데이터 개수
ThinDepth_range = np.array([5., 30.]) #감육깊이 상하한 %
ThinArea_range  = np.array([5., 50.]) #감육넓이 상하한 %
ThinAR_range =    np.array([0.5, 2.])    #감육 형상비 (축/원주)
StdMeas_range   = np.array([1.0, 10.]) #측정오차 상하한 %
N_ac_Case = np.array([[12,13]])       #측정그리드 개수 - 고정 (원주,축)
#출력파일 설정 -----------------------------------------------------------------
OutFileName = "TestData_12by13_Base.xlsx"
###############################################################################

# now = time.localtime()
# print("START - ", now.tm_hour, ":", now.tm_min, ":", now.tm_sec)

###############################################################################
#관련 변수정의------------------------------------------------------------------
ThinDepth = np.empty([N_tn])    #학습데이터 감육깊이 난수
ThinArea = np.empty([N_tn])     #학습데이터 감육면적 난수                  
ThinLoc_a = np.empty([N_tn])    #학습데이터 감육위치(축) 난수
ThinLoc_c = np.empty([N_tn])    #학습데이터 감육위치(원주) 난수
ThinAR = np.empty([N_tn])       #학습데이터 감육 형상비 난수
# StdMeas = np.empty([N_tn])      #학습데이터 측정오차 난수
N_a = N_ac_Case[0,1]            #측정그리드 개수 - 축방향
N_c = N_ac_Case[0,0]            #측정그리드 개수 - 원주방향
# Thin_Meas = np.zeros([N_c,N_a])    #감육측정결과 [행=원주12, 열=축13]
# Thin_true = np.zeros([N_c,N_a])    #감육측정결과 [행=원주12, 열=축13]
# TrainX = np.empty([N_tn, N_a*N_c]) #학습데이터 측정결과
# TrainY = np.empty([N_tn])          #학습데이터 참값(0 or 1)
###############################################################################

# Train Data 만들기 ###########################################################
#설정난수 만들기 --------------------------------------------------------------
for i in range(N_tn):
    #감육데이터
    ThinDepth[i] = np.random.uniform(ThinDepth_range[0], ThinDepth_range[1])
    ThinArea[i] =  np.random.uniform(ThinArea_range[0], ThinArea_range[1])
    ThinLoc_a[i] = np.random.uniform(0.,1.)  
    ThinLoc_c[i] = np.random.uniform(0.,1.)  
    ThinAR[i] = np.random.uniform(ThinAR_range[0], ThinAR_range[1])

StdMeas_1 = np.random.uniform(StdMeas_range[0], StdMeas_range[1], N_tn)  # 1차 학습데이터 측정오차 난수
StdMeas_2 = np.random.uniform(StdMeas_range[0], StdMeas_range[1], N_tn)  # 2차 학습데이터 측정오차 난수
StdMeas_3 = np.random.uniform(StdMeas_range[0], StdMeas_range[1], N_tn)  # 3차 학습데이터 측정오차 난수
    
#측정난수 만들기 --------------------------------------------------------------
Thin_true_delta = []
Meas_err_arr_1 = []
for i in range(N_tn) :
    #감육참값
    Thin_true = dpm.F_Thin_Pipe_eliptical(N_a, N_c, ThinDepth[i]/100., 
                                          ThinArea[i]/100., ThinAR[i], 
                                          ThinLoc_a[i], ThinLoc_c[i])
    Thin_true_delta.append(Thin_true)
    #감육측정에러
    Meas_err_1 = np.random.normal(0., StdMeas_1[i]/100., (N_c,N_a))  #감육측정에러 [행=원주12, 열=축13] [N_c,N_a] (StdMeas_1[i]/100.)/np.sqrt(2)
    Meas_err_arr_1.append(Meas_err_1)

Meas_err_arr_2 = []
for i in range(N_tn) :
    #감육측정에러
    Meas_err_2 = np.random.normal(0., StdMeas_2[i]/100., (N_c,N_a))  #감육측정에러 [행=원주12, 열=축13] [N_c,N_a] (StdMeas_1[i]/100.)/np.sqrt(2)
    Meas_err_arr_2.append(Meas_err_2)

Meas_err_arr_3 = []
for i in range(N_tn) :
    #감육측정에러
    Meas_err_3 = np.random.normal(0., StdMeas_3[i]/100., (N_c,N_a))  #감육측정에러 [행=원주12, 열=축13] [N_c,N_a] (StdMeas_1[i]/100.)/np.sqrt(2)
    Meas_err_arr_3.append(Meas_err_3)

Thin_true_delta = np.array(Thin_true_delta)
Meas_err_arr_1 = np.array(Meas_err_arr_1)
Meas_err_arr_2 = np.array(Meas_err_arr_2)
Meas_err_arr_3 = np.array(Meas_err_arr_3)

t_i_1 = 1 #첫번째 측정
t_i_2 = 3 #두번째 측정(2년뒤)
t_i_3 = 5 #세번째 측정(4년뒤)

y1, y2, y3 = dpm.make_true(N_a, N_c, N_tn, Thin_true_delta, t_i_1, t_i_2, t_i_3)

x = np.arange(N_a)
y = np.arange(N_c)
XX, YY = np.meshgrid(x, y)

plt.title("Contour plots for y3")
cp = plt.contourf(XX, YY, y3[0,:,:], levels=15, alpha=.75, cmap='jet')
plt.contour(XX, YY, y3[0,:,:], levels=15, linewidths=0.5, colors='black')
plt.colorbar(cp)
plt.show()

data1 = y1 + Meas_err_arr_1
data2 = y2 + Meas_err_arr_2
data3 = y3 + Meas_err_arr_3

data = []
for i, j, k in zip(data1,data2,data3):
    data.append(np.stack((i,j,k), axis=0))

data = np.array(data)

y_true = []
for i, j, k in zip(y1,y2,y3):
    y_true.append(np.stack((i,j,k), axis=0))

y_true = np.array(y_true)

with open(f'{code_dir}/data_1.pickle', 'wb') as f:
    pickle.dump(data1, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{code_dir}/data_2.pickle', 'wb') as f:
    pickle.dump(data2, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{code_dir}/data_3.pickle', 'wb') as f:
    pickle.dump(data3, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{code_dir}/data.pickle', 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f'{code_dir}/y_1.pickle', 'wb') as f:
    pickle.dump(y1, f, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f'{code_dir}/y_2.pickle', 'wb') as f:
    pickle.dump(y2, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{code_dir}/y_3.pickle', 'wb') as f:
    pickle.dump(y3, f, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f'{code_dir}/y_true.pickle', 'wb') as f:
    pickle.dump(y_true, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open('E:/Project/03_Wall_thinning/random/data_1.pickle', 'rb') as fb:
#     abc = pickle.load(fb)

common_levels = np.linspace(min(data1[0,:,:].min(), data2[0,:,:].min(),data3[0,:,:].min()), 
                            max(data1[0,:,:].max(), data2[0,:,:].max(),data3[0,:,:].max()), 15)

common_levels = common_levels.round(2)

# vmin = min(data1[0,:,:].min(), data2[0,:,:].min(),data3[0,:,:].min())
# vmax = max(data1[0,:,:].max(), data2[0,:,:].max(),data3[0,:,:].max())

plt.title("Contour plots for data1")
cp = plt.contourf(XX, YY, data1[0,:,:], levels=common_levels, alpha=.75, cmap='jet')
plt.contour(XX, YY, data1[0,:,:], levels=common_levels, linewidths=0.5, colors='black')
plt.colorbar(cp)
# plt.clim(0.75,1.5)
plt.show()

plt.title("Contour plots for data2")
cp = plt.contourf(XX, YY, data2[0,:,:], levels=common_levels, alpha=.75, cmap='jet')
plt.contour(XX, YY, data2[0,:,:], levels=common_levels, linewidths=0.5, colors='black')
plt.colorbar(cp)
plt.show()

plt.title("Contour plots for data3")
cp = plt.contourf(XX, YY, data3[0,:,:], levels=common_levels, alpha=.75, cmap='jet')
plt.contour(XX, YY, data3[0,:,:], levels=common_levels, linewidths=0.5, colors='black')
plt.colorbar(cp)
plt.show()


plt.figure(figsize=(10,5))
plt.title("Heatmap for svr data1")
df_1 = pd.DataFrame(data1[0,:,:])
sns.heatmap(df_1[::-1], annot=True, annot_kws={"size":8}, cmap='jet', fmt='.2f')
plt.show()
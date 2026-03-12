import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
import os

################################ 배관 두께측정 참값 데이터 생성 ################################################
# y1: 초기 데이터
# y2: 감육진행 데이터
# wtgn(wall thinning grid number): 3,5(7은 cos90, cos270에 걸리는 값이 0이라서 0.025/2값으로 대체) ...
# wtr(wall thinning rate): 0.15(5%감육)
# t_i_1, t_i_2, t_i_3: 감육측정 시기 (첫번째, 두번째, 세번째), 첫번째가 1이고 두번째 세번째는 년단위 숫자ex(1,3(2년뒤),5(4년뒤))


# 원주방향 배관 두께 데이터 초기 및 마지막 값 생산 함수
def make_true(wtgn, wtr):
    # y1 초기 데이터 생산
    y1 = np.linspace(0,2*np.pi,13)
    y1 = np.cos(y1)
    y1 = 0.1*y1 + 1   # 0.9 ~ 1.1 코사인함수 배관 원주방향 데이터 생성(중간값이 외호기준)
    
    # x2 감육진행 데이터 생산
    
    y2 = np.linspace(0,2*np.pi,13)
    y2_1 = np.cos(y2[:int((13-wtgn)/2)])
    y2_1 = 0.1*y2_1 + 1   # 0.9 ~ 1.1 코사인함수 배관 원주방향 데이터 생성(중간값이 외호기준) 왼쪽 감육안된부분 잘라서만들기
    
    if wtgn == 7:         # 외호기준 7개 그리드가 wtr(wall thinning rate) 만큼 감육된 부분 생성
        y2_2 = np.cos(y2[int((13-wtgn)/2):-int((13-wtgn)/2)])
        y2_2 = wtr*y2_2 + 1
        y2_2[0] = (y2_2[0] + y2_2[1])/2
        y2_2[-1] = (y2_2[-2] + y2_2[-1])/2
        
    else:
        y2_2 = np.cos(y2[int((13-wtgn)/2):-int((13-wtgn)/2)])  # 외호기준 3 or 5 개 그리드가 wtr(wall thinning rate) 만큼 감육된 부분 생성
        y2_2 = wtr*y2_2 + 1
    
    y2_3 = np.cos(y2[-int((13-wtgn)/2):])
    y2_3 = 0.1*y2_3 + 1   # 0.9 ~ 1.1 코사인함수 배관 원주방향 데이터 생성(중간값이 외호기준) 왼쪽 감육안된부분 잘라서만들기
    
    y2 = np.concatenate((y2_1,y2_2,y2_3))
    
    return y1, y2


# 원주방향 배관 두께 데이터 중간값 생산 함수
def make_mid_true(t_i_1, t_i_2, t_i_3, y_1, y_3):
    # 두번째 감육 초기 데이터 생산
    y_mid = y_1 - (t_i_2-t_i_1)/(t_i_3-t_i_1)*(y_1-y_3)

    return y_mid


################################ 배관 두께 측정 데이터 생성 ################################################
# n_sample: 생성하려는 샘플개수
# e_std: error 표준편차
# r_c: row or column 방향으로 데이터 생성(r or c 둘중에 고름)
# seq: 측정순서 1, 2, 3
# x1: 원주방향 인덱스 1~13
# x2: 시간방향 인덱스 1~13
# t_i: 감육측정년도 1~ (첫 측정1, 1년 단위로 2,3,4...)


# 원주방향 배관 두께 데이터 측정값 생산 함수
def make_measurement(y, n_sample, e_std, r_c, seq):

    if r_c == 'r':
        data = np.tile(y, (n_sample,1))  # 가로방향 원주방향
        error = np.random.normal(0,e_std,n_sample*13).reshape(-1,13) # 가로방향 error 데이터 생성        
    else:
        data = np.tile(y.reshape(-1,1), n_sample) # 세로방향 원주방향
        error = np.random.normal(0,e_std,n_sample*13).reshape(13,-1) # 세로방향 error 데이터 생성

    data = data + error

    #plt.figure(figsize=(10,1))
    plt.title(f'Wall Thickness Measurement_{seq}')
    plt.xlabel('Circumferential Direction')
    plt.ylabel('Wall Thickness')
    for i in range(data.shape[1]):
        plt.plot(data[:,i])
    plt.show()
    
    return data


################### 데이터 전처리함수 ###############################################################
# method: ['mmx', 'std', 'rbs', 'power_b', 'power_j']


# 데이터 전처리를 이용한 데이터 변환 함수
def data_tf(method, data):
    
    if method == 'mmx':
        scaler = MinMaxScaler()
    
    elif method == 'std':
        scaler = StandardScaler()
        
    elif method == 'rbs':
        scaler = RobustScaler()
        
    elif method == 'power_b':
        scaler = PowerTransformer(method='box-cox', standardize=True)
        
    elif method == 'power_j':
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    
    scaler.fit(data)
    data_dn = scaler.transform(data)
    
    return scaler, data_dn


# 전처리 된 데이터의 역변환 함수
def data_inver_tr(data, scaler):
    
    data_dn = scaler.inverse_transform(data)
    
    return data_dn


# 스케일러별 데이터 생성
def make_df_scaler(method, x):
    x_1_scaler, scaled_x_1 = data_tf(method=method, data=x)
    # y_scaler, scaled_y = data_tf(method=method, data=y.reshape(-1,1))
    
    # return x_scaler, y_scaler, scaled_x, scaled_y
    return x_1_scaler, scaled_x_1



################### 다양한 그래프 함수 ###############################################################
# y: 참값
# y_pred: 서포트벡터회귀로 추정한 값
# tn: 그래프 타이틀 이름
# seq: 측정된 순서(1,2,3) 첫번째 두번째 세번째
# y_la: y 레이블
# y_pred_la: y_pred 레이블
# tnc: time의 정규화 계수 조절
# p_i: picture index
# i_p: image path


# 원주방향 배관 두께 데이터 그래프 그리기
def c_wall_th_plot(y, tn, seq):
    plt.plot(y, label=f'm_{seq}')
    plt.title(f'{tn}')
    plt.xlabel('Circumferential Direction')
    plt.ylabel('Wall Thickness')
    plt.legend()

# 원주방향 배관 두께 데이터 그래프 이어서 그리기
def seq_consecutive_plot(y, y_pred, y_la, y_pred_la):
    
    plt.plot(y, label=f'{y_la}')
    plt.plot(y_pred, label=f'{y_pred_la}')
    plt.title('True vs Prediction')
    plt.axvline(x=12, color='r', linestyle='--', linewidth=1)
    plt.axvline(x=25, color='r', linestyle='--', linewidth=1)
    plt.xlabel('Circumferential Direction')
    plt.ylabel('Wall Thickness')
    plt.legend()
    plt.show()


# 비교그림 그리기
def para_search_plot(sc_method, x, y, C, G, E, tnc, m1, m2, m3, p_i, i_p, y_pred_acc):
    
    img_dir = os.path.abspath(f"./{i_p}")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)
    
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 8)) # sharey=True, figsize=(15, 10)

    if isinstance(C, list):
        var = C
    
    elif isinstance(G, list):
        var = G

    elif isinstance(E, list):
        var = E
    
    elif isinstance(tnc, list):
        var = tnc
   
    for r_i, v_i in enumerate(var):
        
        axes[r_i,0].plot(m1, label='m_1')
        axes[r_i,0].plot(m2, label='m_2')
        axes[r_i,0].plot(m3, label='m_3')
        axes[0,0].set_title('Measurement')
        axes[4,0].set_xlabel('Circumferential Direction')
        axes[r_i,0].set_ylabel('Wall Thickness')
        axes[r_i,0].set_ylim(0.8,1.15)
        axes[r_i,0].legend(fontsize=5)

        if isinstance(C, list):
            c_i = v_i
            g_i = G
            e_i = E
            tnc_i = tnc
        
        elif isinstance(G, list):
            c_i = C
            g_i = v_i
            e_i = E
            tnc_i = tnc
    
        elif isinstance(E, list):
            c_i = C
            g_i = G
            e_i = v_i
            tnc_i = tnc
            
        elif isinstance(tnc, list):
            c_i = C
            g_i = G
            e_i = E
            tnc_i = v_i
        
        svr_rbf = SVR(kernel='rbf', C=c_i, gamma=g_i, epsilon=e_i)
        
        if sc_method is not None:
            x_1_scaler, scaled_x_1 = make_df_scaler(method=sc_method, x=x[:,[0]])  # return 값 x_scaler, y_scaler, scaled_x, scaled_y
            x_train = np.concatenate((scaled_x_1.copy(), x[:,[1]].copy() * tnc_i), axis=1)
            y_train = y.copy()
        
        else:
            x_train = x.copy()
            x_train = x_train.astype(float)
            y_train = y.copy()
            x_train[:,1] = x_train[:,1].copy() * tnc_i
           
        svr_rbf.fit(x_train, y_train.ravel())
        y_pred = svr_rbf.predict(x_train)

        if sc_method is not None:
            # y_inver_pred = data_inver_tr(data=y_pred.reshape(-1,1), scaler=y_scaler)
            y_inver_pred = y_pred.copy()

        else:
            y_inver_pred = y_pred.copy()
        
        for j in range(3):
            axes[r_i,1].plot(y_inver_pred.reshape((-1,3), order='F')[:,j], label=f'm_{j+1}')
        axes[r_i,1].set_title(f'{sc_method} Prediction C={c_i}, G={g_i}, E={e_i}, TNC={tnc_i}')
        axes[4,1].set_xlabel('Circumferential Direction')
        axes[r_i,1].set_ylim(0.8,1.15)
        axes[r_i,1].legend(fontsize=5)
    
        
        axes[r_i,2].plot(y, label='y')
        axes[r_i,2].plot(y_inver_pred, label=f'{sc_method} y_inver_pred')
        axes[0,2].set_title('Measurement vs Prediction')
        axes[4,2].set_xlabel('Circumferential Direction')
        axes[r_i,2].axvline(x=12, color='r', linestyle='--', linewidth=1)
        axes[r_i,2].axvline(x=25, color='r', linestyle='--', linewidth=1)
        axes[r_i,2].set_ylim(0.8,1.15)
        axes[r_i,2].legend(fontsize=5)
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                             wspace=None, hspace=None)
        
        y_pred_acc.append(y_inver_pred.reshape((-1,3), order='F'))
    
    fig.tight_layout()
    # fig.suptitle("Wall thinning by Grid(3,5,7)", fontsize=20)
    plt.savefig(f'{img_dir}/{p_i}_{sc_method}_C_{c_i}_G_{g_i}_E_{e_i}_TNC_{tnc_i}.png')
    plt.show()
    
    return y_pred_acc


def time_seq_plot(sc_method, x, y, C, G, E, tnc, m1, m2, m3, p_i, i_p, y_pred_acc):

    img_dir = os.path.abspath(f"./{i_p}")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)
    
    if isinstance(C, list):
        var = C
    
    elif isinstance(G, list):
        var = G

    elif isinstance(E, list):
        var = E
    
    elif isinstance(tnc, list):
        var = tnc

    fig_1, axes_1 = plt.subplots(nrows=5, ncols=7, figsize=(15, 8)) # sharey=True, figsize=(15, 10)    

    for r_i, v_i in enumerate(var):
        
        if isinstance(C, list):
            c_i = v_i
            g_i = G
            e_i = E
            tnc_i = tnc
        
        elif isinstance(G, list):
            c_i = C
            g_i = v_i
            e_i = E
            tnc_i = tnc
    
        elif isinstance(E, list):
            c_i = C
            g_i = G
            e_i = v_i
            tnc_i = tnc
            
        elif isinstance(tnc, list):
            c_i = C
            g_i = G
            e_i = E
            tnc_i = v_i
        
        svr_rbf = SVR(kernel='rbf', C=c_i, gamma=g_i, epsilon=e_i)
        
        if sc_method is not None:
            x_1_scaler, scaled_x_1 = make_df_scaler(method=sc_method, x=x[:,[0]])  # return 값 x_scaler, y_scaler, scaled_x, scaled_y
            x_train = np.concatenate((scaled_x_1.copy(), x[:,[1]].copy() * tnc_i), axis=1)
            y_train = y.copy()
        
        else:
            x_train = x.copy()
            y_train = y.copy()
            x_train[:,1] = x_train[:,1].copy() * tnc_i
           
        svr_rbf.fit(x_train, y_train.ravel())
        y_pred = svr_rbf.predict(x_train)

        if sc_method is not None:
            # y_inver_pred = data_inver_tr(data=y_pred.reshape(-1,1), scaler=y_scaler)
            y_inver_pred = y_pred.copy()

        else:
            y_inver_pred = y_pred.copy()

        for j in range(7):
            axes_1[r_i,j].plot(y.reshape((-1,3), order='F')[j,:], label='measurement')
            axes_1[r_i,j].plot(y_inver_pred.reshape((-1,3), order='F')[j,:], label='prediction')
            
            axes_1[r_i,j].legend(fontsize=5)
            axes_1[r_i,j].set_ylim(0.8,1.15)
            axes_1[r_i,3].set_title(f'seq_1~7_{sc_method} Measurement VS Prediction C={c_i}, G={g_i}, E={e_i}, TNC={tnc_i}')
            
        axes_1[4,3].set_xlabel('Circumferential Direction')

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                             wspace=None, hspace=None)
        
    fig_1.tight_layout()
    plt.savefig(f'{img_dir}/{p_i}_{sc_method}_C_{c_i}_G_{g_i}_E_{e_i}_TNC_{tnc_i}_1.png')
    plt.show()

    fig_2, axes_2 = plt.subplots(nrows=5, ncols=6, figsize=(15, 8)) # sharey=True, figsize=(15, 10)

    for r_i, v_i in enumerate(var):
        
        if isinstance(C, list):
            c_i = v_i
            g_i = G
            e_i = E
            tnc_i = tnc
        
        elif isinstance(G, list):
            c_i = C
            g_i = v_i
            e_i = E
            tnc_i = tnc
    
        elif isinstance(E, list):
            c_i = C
            g_i = G
            e_i = v_i
            tnc_i = tnc
            
        elif isinstance(tnc, list):
            c_i = C
            g_i = G
            e_i = E
            tnc_i = v_i
        
        svr_rbf = SVR(kernel='rbf', C=c_i, gamma=g_i, epsilon=e_i)
        
        if sc_method is not None:
            x_1_scaler, scaled_x_1 = make_df_scaler(method=sc_method, x=x[:,[0]])  # return 값 x_scaler, y_scaler, scaled_x, scaled_y
            x_train = np.concatenate((scaled_x_1.copy(), x[:,[1]].copy() * tnc_i), axis=1)
            y_train = y.copy()
        
        else:
            x_train = x.copy()
            y_train = y.copy()
            x_train[:,1] = x_train[:,1].copy() * tnc_i
           
        svr_rbf.fit(x_train, y_train.ravel())
        y_pred = svr_rbf.predict(x_train)

        if sc_method is not None:
            # y_inver_pred = data_inver_tr(data=y_pred.reshape(-1,1), scaler=y_scaler)
            y_inver_pred = y_pred.copy()

        else:
            y_inver_pred = y_pred.copy()

        y_pred_acc.append(y_inver_pred.reshape((-1,3), order='F'))

        for j in range(6):
            axes_2[r_i,j].plot(y.reshape((-1,3), order='F')[j+7,:], label='measurement')
            axes_2[r_i,j].plot(y_inver_pred.reshape((-1,3), order='F')[j+7,:], label='prediction')
            
            axes_2[r_i,j].legend(fontsize=5)
            axes_2[r_i,j].set_ylim(0.8,1.15)
            axes_2[r_i,3].set_title(f'seq_8~13_{sc_method} Measurement VS Prediction C={c_i}, G={g_i}, E={e_i}, TNC={tnc_i}')

        axes_2[4,2].set_xlabel('Circumferential Direction')


        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                             wspace=None, hspace=None)    

    fig_2.tight_layout()
    # fig.suptitle("Wall thinning by Grid(3,5,7)", fontsize=20)
    plt.savefig(f'{img_dir}/{p_i}_{sc_method}_C_{c_i}_G_{g_i}_E_{e_i}_TNC_{tnc_i}_2.png')
    plt.show()
    
    return y_pred_acc
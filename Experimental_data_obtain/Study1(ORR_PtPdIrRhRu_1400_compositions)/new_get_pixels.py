import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from collections import Counter
import pandas as pd
from random import shuffle
from ase.build import fcc111

# =========================================================================
# 1. 환경 설정 및 기초 정의
# =========================================================================
# 크롭된 정밀 이미지들이 저장되어 있는 로컬 폴더 경로
cropped_dir = r"C:\Users\edb14\metalab\HEA_1"
static_mask = np.load(r"C:\Users\edb14\metalab\HEA_1\static_grid_mask.npy")

# 계산 및 데이터 분석에 사용할 원소 및 활동도 타겟 목록
elements = ['Pd', 'Ir', 'Rh', 'Ru', 'Pt']
activities = ['HER', 'ORR', 'OER']
all_labels = elements

# 데이터 저장용 딕셔너리 초기화
pixel_val_dict = {}
computed_values = {}
rmse_values = {}
raw_gauss_dict = {}

def rotated_gaussian(x, y, mux, muy, sigmax, sigmay, theta_deg):
    theta = np.radians(theta_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    x_rot = (x - mux) * cos_t + (y - muy) * sin_t
    y_rot = -(x - mux) * sin_t + (y - muy) * cos_t
    
    exponent = -(x_rot**2 / (2 * sigmax**2) + y_rot**2 / (2 * sigmay**2))
    normalization = np.sqrt(2 * np.pi * sigmax**2) * np.sqrt(2 * np.pi * sigmay**2)
    return np.exp(exponent) / normalization

# =========================================================================
# 2. ⚡ [핵심 개선] 크롭된 완벽한 이미지에서 직접 픽셀 값 매핑하기
# =========================================================================
# print("=" * 60)
# print(" 📊 크롭 이미지에서 51x51 격자 데이터 추출 시작")
# print("=" * 60)

for label in all_labels:
    # file_path = os.path.join(cropped_dir, f"{label}.png")
    file_path = os.path.join(cropped_dir, f"{label}_padding.png")
    
    if not os.path.exists(file_path):
        print(f"[경고] {label}.png 파일이 폴더에 존재하지 않아 스킵합니다.")
        continue
        
    # OpenCV를 이용해 크롭된 이미지 로드 및 RGB 변환
    bgr_img = cv2.imread(file_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    # 💡 기존의 복잡한 슬라이싱 평균 대신, 완벽한 정대칭 격자이므로 
    # 고성능 이미지 인터폴레이션을 통해 바로 목표하는 51x51 크기의 행렬로 리사이징합니다.
    resized_img = cv2.resize(rgb_img, (51, 51), interpolation=cv2.INTER_AREA)
    
    # 투명도(Alpha)를 결합한 4채널 형태로 변환하고 0~1 값으로 정규화
    rgba_img = np.zeros((51, 51, 4))
    rgba_img[:, :, :3] = resized_img / 255.0
    rgba_img[:, :, 3] = 1.0  # 알파 채널 기본값 1
    
    pixel_val_dict[label] = rgba_img
    # print(f" -> {label}.png 로드 및 51x51 격자 매핑 완료.")

# =========================================================================
# 3. 가우시안 시뮬레이션 합성 및 비교 함수
# =========================================================================
def process_computed_val(k, color, sigmax, sigmay, theta, mus):
    if k not in pixel_val_dict:
        return
        
    # 51x51 메쉬그리드 구성
    x, y = np.meshgrid(np.arange(51), np.arange(51))
    gauss_vals = np.zeros((51, 51))
    
    for mux, muy in mus:
        gau_val = rotated_gaussian(x, y, mux, muy, sigmax, sigmay, theta)
        gau_val *= 2 * np.pi * sigmax * sigmay
        gauss_vals += gau_val
    
    pure_density = gauss_vals.copy()
    pure_density[static_mask] = 0.0  # 배경 마스킹 영역은 강도를 0으로 차단
    raw_gauss_dict[k] = pure_density
    
    # 지정한 컬러맵 투영
    computed_val = colormaps[color](gauss_vals)
    
    # 경계 조건 및 흰색/검은색 노이즈 제거 마스킹
    computed_val[static_mask] = 0
    ref_pixels = pixel_val_dict[k]
    ref_pixels[static_mask] = 0
                
    computed_values[k] = computed_val

    # RMSE 계산 (노란색 십자가 마크 부분 무시 필터링)
    diff = (computed_values[k] - ref_pixels)
    diff[25, :, :] = 0  # 십자가 가로축 필터
    diff[:, 25, :] = 0  # 십자가 세로축 필터
    rmse = np.sqrt(np.mean(diff**2))
    rmse_values[k] = rmse
    print(f"[{k}] 시뮬레이션 완료 -> RMSE: {rmse:.4f}")

    # =========================================================================
    # ⚡ [수정 및 시각화 강화] 원본 데이터와 시뮬레이션 결과를 나란히 비교
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # 1행 2열로 창을 분할합니다.
    
    # 1) 왼쪽 창: 원본 크롭 데이터 이미지 + 현재 설정된 mus 피크 좌표 표시
    axes[0].imshow(ref_pixels)
    if len(mus) > 0:
        mus_np = np.array(mus)
        # mus_np[:, 0]는 x좌표, mus_np[:, 1]는 y좌표입니다.
        axes[0].scatter(mus_np[:, 0], mus_np[:, 1], color='cyan', edgecolors='black', s=40, label='Current mus')
        axes[0].legend(loc='upper right')
    axes[0].set_title(f"Original Data & Mus Peaks ({k})", fontsize=12, weight='bold')
    axes[0].axis('off')
    
    # 2) 오른쪽 창: 가우시안으로 피팅된 시뮬레이션 결과 이미지
    axes[1].imshow(computed_val)
    axes[1].text(0.5, 2, f'RMSE: {rmse:.4f}', color='black', fontsize=11, bbox=dict(facecolor='white', alpha=0.7))
    # 노란색 기준선(십자가 축) 표시
    axes[1].hlines(25.0, 8.0, 42.0, colors='yellow', linestyle='--', linewidth=5)
    axes[1].vlines(25.0, 8.0, 42.0, colors='yellow', linestyle='--', linewidth=5)
    axes[1].set_title(f"Simulated Gaussian Grid ({k})", fontsize=12, weight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    # plt.show()

## Pd
# sigma = 2.602
# sigma = 2.0
sigmax = 2.3
sigmay = 2.75
theta = 144
mus = [
       [26.0,8.5],
       [36.0,11.0],
       [42.0,18.8],
       [43.0,29.0],
       [37.5,37.5],
       [28.5,41.5],
       [18.8,39.1],
       [12.5,31.5],
       [11.5,21.5],
       [17.0,13.0],
       [31.1,18.7],
       [35.0,27.],
       [28.0,32.6],
       [21.0,28.0],
       [22.8,19.5],
       [28.0,25.2]
       ]

process_computed_val('Pd', 'Reds', sigmax, sigmay, theta, mus)

##Ir
# sigma = 2.391
sigmax = 2.05
sigmay = 2.55
theta = 220
mus = [
       [27.2,5.7],
       [37.5,8.5],
       [43.8,16.1],
       [44.5,26.5],
       [39.0,35.0],
       [30.0,39.0],
       [20.8,37.0],
       [14.8,29.0],
       [13.0,19.0],
       [18.5,10.0],
       [33.0,15.5],
       [36.5,24.1],
       [30.0,29.5],
       [23.0,25.0],
       [24.5,16.7],
       [29.5,22.0]
       ]

process_computed_val('Ir', 'Greens', sigmax, sigmay, theta, mus)

#Rh
sigmax = 2.3
sigmay = 2.6
theta = 142
mus = [
       [24.0,7.0],
       [33.5,9.5],
       [40.0,16.9],
       [39.9,27.0],
       [35.3,36.0],
       [26.0,39.5],
       [16.5,37.5],
       [10.0,29.5],
       [9.5,19.5],
       [15.0,10.5],
       [28.7,16.8],
       [31.5,24.8],
       [25.0,30.5],
       [18.0,26.0],
       [20.5,17.5],
       [25.0,23.0]
       ]

process_computed_val('Rh', 'Purples', sigmax, sigmay, theta, mus)

#Ru
sigmax = 2.2
sigmay = 2.7
theta = 5
mus = [
       [25.5,5.5],
       [34.8,8.0],
       [41.5,15.5],
       [42.3,26.0],
       [36.6,34.6],
       [27.5,37.5],
       [18.0,36.0],
       [11.7,28.2],
       [11.0,19.0],
       [16.0,10.0],
       [30.0,15.0],
       [33.4,23.0],
       [27.0,29.0],
       [20.0,24.3],
       [21.5,15.5],
       [26.6,21.2]
       ]

process_computed_val('Ru', 'Blues', sigmax, sigmay, theta, mus)

#Pt
sigmax = 2.25
sigmay = 2.6
theta = 216
mus = [
       [24.5,9.5],
       [34.0,11.5],
       [40.2,19.5],
       [41.0,29.5],
       [35.5,38.0],
       [26.5,42.0],
       [17.0,40.0],
       [11.0,32.0],
       [10.0,21.7],
       [15.0,13.0],
       [29.5,19.5],
       [32.0,27.5],
       [25.8,33.5],
       [18.0,29.0],
       [20.5,20.0],
       [25.0,25.8]
       ]

process_computed_val('Pt', 'Oranges', sigmax, sigmay, theta, mus)

'''
## Pd
# sigma = 2.602
# sigma = 2.0
sigmax = 2.4
sigmay = 2.95
theta = 144
mus = [
       [26.8,6.5],
       [37.5,9.5],
       [45.0,18.0],
       [46.0,29.5],
       [39.5,39.5],
       [29.5,43.5],
       [18.5,41.5],
       [11.0,32.0],
       [10.5,20.5],
       [16.5,11.5],
       [32.5,18.5],
       [36.5,27.5],
       [28.5,33.5],
       [20.5,28.5],
       [23.0,18.5],
       [28.5,25.0]
       ]

process_computed_val('Pd', 'Reds', sigmax, sigmay, theta, mus)

##Ir
# sigma = 2.391
sigmax = 2.4
sigmay = 2.9
theta = 216
mus = [
       [27.8,3.0],
       [39.0,6.0],
       [46.5,14.5],
       [47.5,26.5],
       [41.0,36.0],
       [30.5,41.2],
       [20.0,38.5],
       [13.5,30.0],
       [11.5,17.5],
       [17.5,7.5],
       [34.2,14.2],
       [38.0,23.9],
       [30.5,30.5],
       [22.5,25.0],
       [24.0,15.5],
       [30.0,21.5]
       ]

process_computed_val('Ir', 'Greens', sigmax, sigmay, theta, mus)

#Rh
sigmax = 2.7
sigmay = 3.0
theta = 144
mus = [
       [24.0,4.0],
       [34.5,7.0],
       [42.0,16.0],
       [42.0,27.0],
       [37.0,38.0],
       [26.0,41.5],
       [15.5,39.5],
       [8.0,30.5],
       [7.5,18.5],
       [13.5,8.5],
       [29.0,16.0],
       [32.5,24.5],
       [25.5,31.5],
       [17.0,26.5],
       [19.5,16.5],
       [24.5,23.5]
       ]

process_computed_val('Rh', 'Purples', sigmax, sigmay, theta, mus)

#Ru
sigmax = 2.6
sigmay = 3.05
theta = 0
mus = [
       [25.5,3.5],
       [36.0,5.5],
       [44.0,14.5],
       [45.0,26.0],
       [38.0,36.5],
       [28.0,40.0],
       [17.0,38.0],
       [10.0,29.0],
       [9.0,18.0],
       [14.5,8.0],
       [31.0,14.0],
       [34.5,23.5],
       [27.5,30.0],
       [19.0,25.0],
       [21.0,14.5],
       [26.5,21.0]
       ]

process_computed_val('Ru', 'Blues', sigmax, sigmay, theta, mus)

#Pt
sigmax = 2.6
sigmay = 2.8
theta = 216
mus = [
       [24.0,7.0],
       [35.0,9.8],
       [42.5,18.5],
       [43.0,30.0],
       [37.0,40.0],
       [27.0,44.0],
       [16.0,42.0],
       [9.0,33.0],
       [8.5,21.5],
       [14.0,11.5],
       [30.0,18.8],
       [33.0,28.0],
       [26.0,34.5],
       [17.2,29.5],
       [19.5,19.0],
       [25.0,26.0]
       ]

process_computed_val('Pt', 'Oranges', sigmax, sigmay, theta, mus)
'''
#print rmse
print(rmse_values)


total_gauss_density = np.zeros((51, 51))
for density_map in raw_gauss_dict.values():
    total_gauss_density += density_map

# 분모가 0인 곳(마스킹된 배경)에서 나누기 오류(NaN)가 나는 것을 방지하기 위해 
# 안전 필터를 적용합니다. (0인 곳은 1.0으로 임시 대체)
safe_total_density = np.where(total_gauss_density == 0, 1.0, total_gauss_density)

# 각 원소별 진정한 정량 조성 비율(0.0 ~ 1.0) 계산
normalized_gauss_vals = {}
for key, density_map in raw_gauss_dict.items():
    # 이제 진짜 강도가 높은 곳(진한 곳)이 높은 비율을 갖게 됩니다!
    normalized_gauss_vals[key] = density_map / safe_total_density

# 확인용 플롯 출력
for key, val in normalized_gauss_vals.items():
    plt.figure()
    plt.title(f"Normalized values for {key} (Fixed)")
    plt.imshow(val, cmap='viridis')
    plt.colorbar()

# atoms = [[[] for _ in range (51)] for _ in range(51)]

# for key, value in normalized_gauss_vals.items():
#     if key == 'Pd':
#         a = 46
#     elif key == 'Ir':
#         a = 77
#     elif key == 'Rh':
#         a = 45
#     elif key == 'Ru':
#         a = 44
#     else:
#         a = 78
        
#     for i, val in enumerate(value):
#         for j, v in enumerate(val):
#             for _ in range(int(512*v)):
#                 atoms[i][j].append(a)
            
# pixels = []
  
# for i in range(51):
#     for j in range(51):
#         if len(atoms[i][j]) > 0:
#             if len(atoms[i][j]) < 512:
#                 for _ in range(512-len(atoms[i][j])):
#                     atoms[i][j].append(78)
#             pixel = fcc111('Pt', size=(8, 8, 8) , a=3.9672029988415840, vacuum=10)
#             atoms_numbers = atoms[i][j]
#             shuffle(atoms_numbers)
#             pixel.set_atomic_numbers(atoms_numbers)
#             pixels.append(pixel)

# target_elements = [44, 45, 46, 77, 78]
# composition_list = []

# for pixel in pixels:
#     atom_numbers = pixel.get_atomic_numbers()
#     counts = Counter(atom_numbers)

#     row = [counts.get(num, 0) / 512.0 for num in target_elements]
#     composition_list.append(row)

# composition_array = np.array(composition_list)

# columns = ['Ru', 'Rh', 'Pd', 'Ir', 'Pt']
# df = pd.DataFrame(composition_array, columns=columns)
# df['Total'] = df[columns].sum(axis=1)
# df.to_csv('composition_new.csv', index_label='Pixel_Index')

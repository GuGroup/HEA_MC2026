import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

# =========================================================================
# 1. 경로 설정 및 파일 로드
# =========================================================================
cropped_dir = r"C:\Users\edb14\metalab\HEA_1"

orr_path = os.path.join(cropped_dir, "ORR.png")
mask_path = os.path.join(cropped_dir, "static_grid_mask.npy")
csv_path = os.path.join(cropped_dir, "composition.csv")

if not os.path.exists(orr_path) or not os.path.exists(mask_path):
    print("[오류] 파일 경로를 다시 확인해 주세요.")
    exit()

static_mask = np.load(mask_path)
orr_bgr = cv2.imread(orr_path)
orr_rgb = cv2.cvtColor(orr_bgr, cv2.COLOR_BGR2RGB)
orr_resized = cv2.resize(orr_rgb, (51, 51), interpolation=cv2.INTER_AREA) / 255.0

# =========================================================================
# 2. ⚡ [핵심 수정] R - B 채널 차이를 이용한 정밀 밸런스 복원
# =========================================================================
# 흰색/회색 배경은 R-B를 하면 0에 가깝게 소거되므로 데이터 왜곡을 막아줍니다.
r_channel = orr_resized[:, :, 0]
b_channel = orr_resized[:, :, 2]
rb_difference = r_channel - b_channel

# 오직 유효한 마스크 내부(데이터 구역)에서의 최솟값과 최댓값만 추출하여 기준으로 삼습니다.
valid_rb_values = rb_difference[~static_mask]
rb_min = valid_rb_values.min()
rb_max = valid_rb_values.max()

# 💡 R-B가 가장 작을 때(파란색) -> -11 pA / 가장 클 때(빨간색) -> -25 pA
actual_activity_map = np.interp(rb_difference, (rb_min, rb_max), (-11.0, -25.0))

# =========================================================================
# 3. 마스킹 및 nan 투명화 처리
# =========================================================================
activity_masked = actual_activity_map.copy()
activity_masked[static_mask] = np.nan

value_list = actual_activity_map[~static_mask].tolist()

print("=" * 60)
print(f"✅ ORR Activity 스케일 밸런스 튜닝 완료")
print(f" -> 유효 데이터 범위: {min(value_list):.2f} pA ~ {max(value_list):.2f} pA")
print("=" * 60)

# =========================================================================
# 4. 시각화 검증
# =========================================================================
# fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# axes[0].imshow(orr_resized)
# axes[0].set_title("1. Original Crop (ORR_perfect)", weight='bold')
# axes[0].axis('off')

# # bwr_r 컬러맵으로 밸런스가 잡힌 이미지 투영
# im = axes[1].imshow(activity_masked, cmap='bwr_r', vmin=-25, vmax=-11)
# axes[1].set_title("2. Fixed ORR Activity Map (Balanced)", weight='bold')
# axes[1].axis('off')

# cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
# cbar.set_label("Activity / pA", rotation=270, labelpad=15, weight='bold')

# plt.tight_layout()
# plt.show()

# =========================================================================
# 2. ⚡ [핵심 요청 1] 빨간 부분 = -1, 파란 부분 = 0 스케일링
# =========================================================================
# # 원본에서 -24 pA(진한 빨강) -> -1.0 / -12 pA(진한 파랑) -> 0.0 으로 선형 변환합니다.
# # np.interp는 x가 커질 때 y도 커져야 하므로 (-24, -12) 범위를 (-1.0, 0.0)으로 매핑합니다.
# scaled_activity_map = np.interp(actual_activity_map, (-25.0, -11.0), (-1.0, 0.0))

# # 안전하게 범위를 [-1.0, 0.0] 사이로 클리핑
# scaled_activity_map = np.clip(scaled_activity_map, -1.0, 0.0)
# =========================================================================
# 2. ⚡ [수정] 음수 데이터에 대한 로그 변환 및 -1 ~ 0 스케일링
# =========================================================================

# 1단계: 유효 데이터만 추출
valid_actual = actual_activity_map[~static_mask]

# 2단계: 로그 변환을 위해 양수 영역으로 평행이동 (Shift)
# 예: 가장 작은 음수보다 더 큰 값을 더해 모든 값을 안전한 양수(>0)로 만듭니다.
# 0으로 나누기나 log(0)을 방지하기 위해 최소값을 1 이상으로 맞추는 것이 좋습니다.
shift_val = abs(valid_actual.min()) + 1.0  # 예: -25였다면 26을 더해 1 이상으로 만듦
shifted_activity = actual_activity_map + shift_val

# 3단계: 로그 변환 적용 (데이터의 스펙트럼이 넓을 때 유용)
# log10 또는 자연로그 log를 사용하되, 마스크 영역은 나중에 처리하므로 전체 배열에 적용 가능
log_activity = np.log(shifted_activity)

# 4단계: 로그가 적용된 상태에서의 유효 범위 재추출
valid_log_values = log_activity[~static_mask]
log_min = valid_log_values.min()
log_max = valid_log_values.max()

# 5단계: 최종적으로 -1.0 ~ 0.0 범위로 선형 스케일링 (빨간색: -1, 파란색: 0)
# np.interp를 이용해 log_min ~ log_max 범위를 -1.0 ~ 0.0으로 매핑합니다.
scaled_activity_map = np.interp(log_activity, (log_min, log_max), (-1.0, 0.0))

# 안전하게 범위를 [-1.0, 0.0] 사이로 클리핑
scaled_activity_map = np.clip(scaled_activity_map, -1.0, 0.0)

# 마스크 영역은 다시 0이나 원하는 값으로 채우거나 유지 (이후 시각화에서 nan 처리)
# =========================================================================
# 3. ⚡ [핵심 요청 2] 데이터 저장하기 (.npy 형식)
# =========================================================================
# 51x51 행렬 형태 그대로 넘파이 바이너리 파일로 저장합니다. (용량도 작고 빠릅니다)
save_data_path = os.path.join(cropped_dir, "orr_log_scaled_activity.npy")
np.save(save_data_path, scaled_activity_map)
print(f"💾 스케일링된 맵 저장 완료: {save_data_path}")

print("-" * 60)

valid_activities = scaled_activity_map[~static_mask]

# 검증을 위해 기존 composition.csv 행 개수와 비교
df_comp = pd.read_csv(csv_path)
csv_row_count = len(df_comp)
extracted_count = len(valid_activities)

print("=" * 60)
print(f"📋 인덱스 매칭 검증")
print(f" -> composition.csv의 유효 행 개수: {csv_row_count} 개")
print(f" -> 추출된 유효 Activity 개수  : {extracted_count} 개")

if csv_row_count == extracted_count:
    print(" ✅ 검증 성공: CSV와 Activity 데이터 개수가 정확히 일치합니다!")
else:
    print(" ❌ 경고: 데이터 개수가 일치하지 않습니다. 마스크 파일을 다시 확인해 주세요.")
print("=" * 60)

# =========================================================================
# 4. ⚡ JSON 파일 구조화 및 저장
# =========================================================================
# Key: CSV의 Pixel_Index (문자열 타입), Value: 스케일링된 Activity 값
activity_dict = {
    str(idx): float(val) for idx, val in enumerate(valid_activities)
}

# json 파일로 내보내기 (들여쓰기를 주어 가독성 확보)
json_path = os.path.join(cropped_dir, "orr_log_matched_activity.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(activity_dict, f, indent=4)

# print(f"💾 CSV 매칭용 JSON 저장 완료: {json_path}")

loaded_activity = np.load(save_data_path)
loaded_mask = np.load(mask_path)

# 시각화를 위해 마스크 구역을 다시 np.nan으로 투명화 처리
visualization_map = loaded_activity.copy()
visualization_map[loaded_mask] = np.nan

# 그래프 그리기
plt.figure(figsize=(6, 5))

# vmin=-1(빨강), vmax=0(파랑)으로 고정하고 컬러맵을 'bwr_r'로 지정합니다.
im = plt.imshow(visualization_map, cmap='bwr_r', vmin=-1.0, vmax=0.0)
plt.title("Reloaded & Rescaled ORR Activity Map\n(Red: -1, Blue: 0)", weight='bold', pad=15)
# plt.axis('off')

# 컬러바 설정
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("Normalized Activity Scale", rotation=270, labelpad=15, weight='bold')
# 컬러바의 축 눈금(Ticks)을 깔끔하게 -1과 0 위주로 표시
cbar.set_ticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0])

plt.tight_layout()
plt.show()
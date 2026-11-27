import cv2
import os

# 1. 파일 경로 설정
img_path = r"C:\Users\edb14\metalab\HEA_1\adma202207635-sup-0001-suppmat.png"
img = cv2.imread(img_path)

if img is None:
    print(f"[오류] 이미지를 불러올 수 없습니다. 경로를 다시 확인해주세요: {img_path}")
    exit()

output_dir = r"C:\Users\edb14\metalab\HEA_1\cropped_perfect_grid"
os.makedirs(output_dir, exist_ok=True)

crop_width, crop_height = 284, 284
# crop_width, crop_height = 250, 250
BORDER_COLOR = [0, 0, 0]
saved_count = 0

# =========================================================================
# 🎯 [중요] 기준 좌표 및 간격 설정 (사용자의 이미지 해상도에 맞게 조정이 필요할 수 있습니다)
# 아래 좌표들은 기본 해상도 기준 예시이며, 혹시 위치가 맞지 않으면 이 숫자들만 미세하게 수정하면 됩니다.
# =========================================================================

# --- [C 영역: 아래쪽 원형 데이터 맵 2줄] ---
# 가장 첫 번째(맨 왼쪽 위) 그림인 'Pd' 그림의 십자가 정중앙 좌표를 직접 지정합니다.
c_labels = ["Pd", "Ir", "Rh", "Ru", "Pt"]
c_start_x = 137  # Pd 십자가의 X 좌표
c_start_y = 565  # Pd 십자가의 Y 좌표

c_col_gap = 281  # 가로로 다음 그림까지의 간격 (픽셀 단위)
c_row_gap = 180  # 세로로 다음 줄까지의 간격 (픽셀 단위)

# C 영역은 가로 5개(원소별), 세로 2줄(상단/하단 세트) 구조입니다.
c_centers = []
for row in range(1):      # 2줄
    for col in range(5):  # 5개 원소
        cx = c_start_x + (col * c_col_gap)
        cy = c_start_y + (row * c_row_gap)
        c_centers.append((cx, cy,c_labels[col]))


# --- [B 영역: 위쪽 전개도 그림들] ---
# B 영역의 맨 왼쪽 첫 번째 그림 십자가의 정중앙 좌표를 지정합니다.
b_labels = ["HER", "ORR", "OER"]
b_start_x = 508
b_start_y = 187  # B 영역의 Y 좌표

b_col_gap = 358  # 가로 간격

# B 영역은 가로로 5개 배치되어 있다고 가정합니다.
b_centers = []
for col in range(3):
    cx = b_start_x + (col * b_col_gap)
    cy = b_start_y
    b_centers.append((cx, cy, b_labels[col]))

# 모든 타겟 중심점 합치기
# all_targets = b_centers + c_centers
all_targets = c_centers
# =========================================================================
# 2. 수학적으로 계산된 좌표를 바탕으로 완벽한 대칭 크롭 수행
# =========================================================================
print("=" * 60)
print(" 📐 격자(Grid) 좌표 기반 100% 정밀 크롭 모드 작동 중...")
print(" - [ s ] : 이미지 저장 후 다음으로 이동")
print(" - [ 아무 키 ] : 저장하지 않고 건너뛰기")
print("=" * 60)

img_h, img_w = img.shape[:2]

for center_x, center_y, label in all_targets:
        
    # 1. ⚠️ max/min을 쓰지 않고 '이론적인' 시작/끝 좌표를 그대로 계산합니다. (음수가 나와도 무방)
    start_x = center_x - crop_width // 2
    start_y = center_y - crop_height // 2
    end_x = start_x + crop_width
    end_y = start_y + crop_height
    
    # 2. 이론적 좌표를 바탕으로 실제 이미지 밖으로 탈출한 픽셀 수(패딩량)를 정확히 계산합니다.
    pad_top = max(0, -start_y)
    pad_bottom = max(0, end_y - img_h)
    pad_left = max(0, -start_x)
    pad_right = max(0, end_x - img_w)
    
    # 3. 원본 이미지에서 실제로 잘라낼 수 있는 '안전한 범위'는 여기서 제한해 줍니다.
    safe_start_x = max(0, start_x)
    safe_start_y = max(0, start_y)
    safe_end_x = min(img_w, end_x)
    safe_end_y = min(img_h, end_y)
    
    # 4. 안전 영역 크롭 (이미지 경계선까지만 딱 자름)
    cropped_img = img[safe_start_y:safe_end_y, safe_start_x:safe_end_x]
    
    # 5. 모자란 부분에 패딩을 채워 넣음 (이제 좌측 음수 분량만큼 검은색 패딩이 생깁니다!)
    padded_img = cv2.copyMakeBorder(
        cropped_img, 
        pad_top, pad_bottom, pad_left, pad_right, 
        borderType=cv2.BORDER_CONSTANT, 
        value=BORDER_COLOR
    )
        
    # 미리보기 창 표시
    window_title = f"Grid Preview - {label} ({center_x}, {center_y})"
    cv2.imshow(window_title, padded_img)
    
    key = cv2.waitKey(0)
    if key == ord('s') or key == ord('S'):
        save_path = os.path.join(output_dir, f"{label}_padding.png")
        cv2.imwrite(save_path, cropped_img)
        print(f"[저장 완료] {label} 정중앙 크롭 성공 -> {save_path}")
        saved_count += 1s
    else:
        print(f"[건너뜀] {label} 제외")
        
    cv2.destroyWindow(window_title)

cv2.destroyAllWindows()
print(f"\n[모든 작업 완료] 오차 없이 정렬된 {saved_count}개의 이미지가 '{output_dir}' 폴더에 저장되었습니다.")
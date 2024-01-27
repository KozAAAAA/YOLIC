import cv2
import json
import numpy as np
from pathlib import Path

# define directories and images resolution
seg_dir = Path('./CarLast/validation/masks')
label_dir = Path('./CarLast/validation/yolic_labels')
cells_file = Path('./cell_design.json')
img_res = (224, 224)

# process file with cells of interests layout
json_file = open(str(cells_file))
data = json.load(json_file)
cois_data = data["COIs"]
coi_num = int(data["COIs"]["COINumber"])
classes_num = int(data["Labels"]["LabelNumber"])
json_file.close()

# generate masks for every cell of interests
masks = []
for coi in cois_data:
    if coi == "COINumber":
        continue
    mask = np.zeros(img_res)

    # reshape array of coordinates
    poly_cords = np.array(cois_data[coi][1:])
    num_cords = poly_cords.size // 2
    poly_cords = poly_cords.reshape(num_cords, 2)
    poly_cords = np.array([np.multiply(cords, img_res) for cords in poly_cords]).astype(np.int32)

    # get mask
    if cois_data[coi][0] == "polygon":    
        cv2.fillPoly(mask, [poly_cords],(255))
    else:
        [[x, y], [w, h]] = poly_cords
        cv2.rectangle(mask, [x, y], [x+w, y+h], (255), -1)
    masks.append(mask)
    
# generate yolic label for every segmentation image
for seg_file in seg_dir.iterdir():
    seg_img = cv2.imread(str(seg_file))

    # find segmentation labels in all cells of interests
    result = []
    for mask in masks:
        coi_result = [0] * (classes_num+1)
        values = seg_img[mask==255]
        bg = True
        for cls in range(1, classes_num+1):
            if cls in values:
                coi_result[cls] = 1
                bg = False
        if bg:
            coi_result[0] = 1
        result += coi_result

    # write yolic labels to file
    result_file = open(str(label_dir / seg_file.stem) + '.txt', 'w')
    result_file.write(''.join(str(r) + ' ' for r in result))
    result_file.close()
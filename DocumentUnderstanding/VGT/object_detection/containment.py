import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from ditod import VGTTrainer
# print( "vgttrainer.oggheight ", VGTTrainer.oggheight)
original_height = 612
original_width = 792
resized_height = 800
resized_width = 1035

# Sample bounding boxes
# boxes = [[288.1917, 484.7234, 428.0781, 531.2474],
#         [459.8793, 484.5547, 589.3247, 531.3502],
#         [116.5221, 484.6494, 247.5370, 531.5381],
#         [288.2242, 312.2604, 416.1836, 374.7731],
#         [288.3173, 142.4946, 398.9115, 188.5793],
#         [460.0410, 142.4497, 592.5703, 188.5733],
#         [116.4405, 142.4533, 257.9611, 188.4464],
#         [113.1688, 673.7156, 386.6188, 709.7101],
#         [548.3896, 751.4247, 582.9852, 760.9554],
#         [460.0588, 313.2395, 603.0294, 375.7906],
#         [116.3280, 313.1845, 256.6854, 359.6259],
#         [ 45.3760, 814.9465, 158.3074, 821.8260],
#         [593.6528, 815.2188, 644.5563, 821.0131],
#         [596.1397, 748.9673, 603.3596, 761.5773],
#         [ 30.5570,  18.4748, 675.1966, 819.7064],
#         [107.1593,  83.1975, 531.7543, 122.2687],
#         [462.1190, 379.7830, 601.8660, 454.9788],
#         [117.0643, 366.9367, 256.7705, 455.5495],
#         [117.3651, 196.0461, 255.4906, 285.2540],
#         [287.5542, 378.6000, 428.7285, 457.3341],
#         [287.1452, 194.2032, 429.0315, 286.6736],
#         [107.6298, 110.3996, 531.6072, 122.2589],
#         [314.8196,  87.0971, 366.3673,  97.8585],
#         [460.7277, 194.7030, 601.4795, 285.1799],
#         [107.2715,  83.2112, 284.1230, 103.1866],
#         [458.8961, 540.5170, 601.4069, 629.4706],
#         [ 55.5892,  18.7357, 663.0027,  70.5279],
#         [ 36.8430,  12.6907,  95.7245, 804.0374],
#         [116.4527, 543.9061, 256.4210, 627.2487],
#         [286.7128, 541.6981, 429.8167, 628.9529],
#         [ 36.1153, 719.2272, 673.5209, 810.8027],
#         [109.8641, 193.7978, 608.4915, 285.5848],
#         [103.7456, 372.2714, 610.5327, 458.7959],
#         [157.1413, 673.6105, 260.0317, 692.0986],
#         [ 91.1037, 541.2268, 637.1539, 634.0247],
#         [115.5911, 673.6794, 138.7808, 692.2928],
#         [548.2986, 748.6394, 604.1763, 761.5536],
#         [106.7816,  83.2200, 340.1469, 102.9798],
#         [116.3180, 313.2078, 159.9411, 324.2234],
#         [ 45.0810, 530.3650, 665.2291, 708.9802]]

# boxes = [[1.9856e+02, 3.3818e+02, 2.2667e+02, 3.4676e+02],
#         [4.9197e+02, 2.3208e+02, 5.0667e+02, 2.4070e+02],
#         [1.9684e+02, 2.3199e+02, 2.2851e+02, 2.4067e+02],
#         [4.7837e+02, 3.3819e+02, 5.1339e+02, 3.4676e+02],
#         [4.5980e+02, 4.6734e+02, 5.6357e+02, 5.1415e+02],
#         [5.9352e+02, 8.1532e+02, 6.4463e+02, 8.2119e+02],
#         [4.5987e+01, 8.1493e+02, 1.5876e+02, 8.2074e+02],
#         [3.4326e+02, 4.5000e+02, 4.4067e+02, 5.1448e+02],
#         [3.3179e+02, 3.8961e+02, 5.4237e+02, 4.2866e+02],
#         [5.9570e+02, 7.4866e+02, 6.0321e+02, 7.6135e+02],
#         [1.6253e+02, 7.1223e+01, 5.3497e+02, 1.1425e+02],
#         [4.5319e+02, 5.9922e+02, 5.9941e+02, 7.5185e+02],
#         [1.6371e+02, 7.1226e+01, 3.2998e+02, 1.1252e+02],
#         [3.4168e+02, 9.2033e+01, 5.3454e+02, 1.1511e+02],
#         [2.4715e-01, 6.0752e+01, 6.6036e+02, 8.2754e+02],
#         [3.3477e+02, 4.1740e+02, 5.2249e+02, 4.2862e+02],
#         [2.9279e+02, 6.0010e+02, 4.2235e+02, 7.1927e+02],
#         [1.0543e+02, 3.9920e+02, 2.7997e+02, 7.0413e+02],
#         [0.0000e+00, 2.6988e+02, 3.5491e+01, 8.1044e+02],
#         [5.4827e+02, 7.5117e+02, 5.8260e+02, 7.6086e+02],
#         [1.1582e+02, 5.9581e+02, 4.2008e+02, 7.1636e+02],
#         [3.3484e+02, 3.8956e+02, 5.3937e+02, 4.0570e+02],
#         [1.0695e+02, 4.1741e+02, 2.6590e+02, 5.5514e+02],
#         [1.0332e+02, 3.9492e+02, 5.0912e+02, 7.1025e+02],
#         [1.5807e+02, 1.1337e+02, 5.5370e+02, 3.7092e+02],
#         [1.0699e+02, 3.8970e+02, 2.4873e+02, 4.0589e+02],
#         [5.8194e+02, 7.4751e+02, 6.7091e+02, 7.6212e+02],
#         [1.2688e+02, 5.9987e+02, 2.7268e+02, 7.0451e+02],
#         [4.6510e+02, 7.2830e+02, 5.8257e+02, 7.6063e+02],
#         [4.6273e+02, 6.5792e+02, 5.8604e+02, 7.3847e+02],
#         [4.5662e+02, 5.9944e+02, 5.9982e+02, 6.6620e+02],
#         [0.0000e+00, 6.1490e+01, 6.6629e+02, 8.2800e+02],
#         [1.0426e+02, 3.8913e+02, 2.6718e+02, 4.5897e+02],
#         [1.0243e+02, 3.9566e+02, 4.0703e+02, 5.9591e+02],
#         [6.0826e+01, 4.1273e+02, 6.5500e+02, 5.9704e+02],
#         [1.3021e+02, 6.8103e+02, 2.3585e+02, 7.0488e+02],
#         [4.6534e+02, 6.4684e+02, 5.2174e+02, 6.5740e+02],
#         [4.6495e+02, 6.4693e+02, 5.7122e+02, 6.7178e+02],
#         [2.9669e+02, 6.6646e+02, 4.2230e+02, 7.1920e+02],
#         [1.2855e+02, 6.5790e+02, 2.3580e+02, 7.0460e+02],
#         [2.9737e+02, 5.9935e+02, 4.0589e+02, 6.3720e+02],
#         [9.9283e+01, 4.1651e+02, 5.9864e+02, 5.5704e+02],
#         [2.9777e+02, 5.9896e+02, 3.7071e+02, 6.0928e+02],
#         [5.7296e+02, 6.4838e+02, 6.6806e+02, 7.6462e+02],
#         [3.3475e+02, 4.1737e+02, 5.2249e+02, 4.2860e+02],
#         [4.5162e+02, 5.9867e+02, 6.3075e+02, 7.6197e+02],
#         [1.4041e+00, 3.5637e+02, 2.8465e+02, 7.3917e+02],
#         [5.7269e+01, 6.9119e+01, 6.0396e+02, 4.6750e+02],
#         [2.9367e+02, 6.3281e+02, 4.2244e+02, 7.1866e+02],
#         [5.9678e+02, 7.4869e+02, 6.1453e+02, 7.6143e+02],
#         [4.6409e+02, 6.8144e+02, 5.8687e+02, 7.3912e+02],
#         [1.0641e+02, 5.6725e+02, 2.7912e+02, 5.8976e+02]]

# boxes = [[385.6282, 575.8620, 404.6315, 591.4712],
#         [416.2117,  35.1908, 748.5856, 174.6790],
#         [411.4085, 206.2887, 746.7636, 308.4788],
#         [ 40.8395,  35.4667, 378.6684, 231.4235],
#         [ 35.2467, 578.1175, 249.9924, 590.8598],
#         [ 41.1264, 263.7213, 379.9325, 461.3715],
#         [ 40.9214,  35.9544,  56.3146,  52.2761],
#         [413.3896,  32.5783, 748.4283, 327.0828],
#         [ 39.5505,  35.6519, 378.3299,  70.2991],
#         [ 39.9944, 264.4403, 378.4658, 298.4931],
#         [ 40.0942,  35.4453, 378.6313, 231.1627],
#         [ 40.1628, 263.9309, 380.1836, 461.4005],
#         [411.2342,  36.2666, 426.4051,  52.4579],
#         [ 37.4972,  32.7813, 750.1583, 465.4253],
#         [ 36.3468, 258.5336, 735.2711, 483.1108],
#         [ 38.1805,  33.5716, 381.8362, 461.8158],
#         [ 41.1906, 264.2855,  56.3459, 280.3735],
#         [411.3876, 206.4326, 746.8021, 308.4673],
#         [415.4464,  35.2347, 748.6210, 174.6382],
#         [ 35.2216, 578.1929, 249.9919, 591.0222]]

boxes = [[288.1917, 484.7234, 428.0781, 531.2474],
        [459.8793, 484.5547, 589.3247, 531.3502],
        [116.5221, 484.6494, 247.5370, 531.5381],
        [288.2242, 312.2604, 416.1836, 374.7731],
        [288.3173, 142.4946, 398.9115, 188.5793],
        [460.0410, 142.4497, 592.5703, 188.5733],
        [116.4405, 142.4533, 257.9611, 188.4464],
        [113.1688, 673.7156, 386.6188, 709.7101],
        [548.3896, 751.4247, 582.9852, 760.9554],
        [460.0588, 313.2395, 603.0294, 375.7906],
        [116.3280, 313.1845, 256.6854, 359.6259],
        [ 45.3760, 814.9465, 158.3074, 821.8260],
        [593.6528, 815.2188, 644.5563, 821.0131],
        [596.1397, 748.9673, 603.3596, 761.5773],
        [ 30.5570,  18.4748, 675.1966, 819.7064],
        [107.1593,  83.1975, 531.7543, 122.2687],
        [462.1190, 379.7830, 601.8660, 454.9788],
        [117.0643, 366.9367, 256.7705, 455.5495],
        [117.3651, 196.0461, 255.4906, 285.2540],
        [287.5542, 378.6000, 428.7285, 457.3341],
        [287.1452, 194.2032, 429.0315, 286.6736],
        [107.6298, 110.3996, 531.6072, 122.2589],
        [314.8196,  87.0971, 366.3673,  97.8585],
        [460.7277, 194.7030, 601.4795, 285.1799],
        [107.2715,  83.2112, 284.1230, 103.1866],
        [458.8961, 540.5170, 601.4069, 629.4706],
        [ 55.5892,  18.7357, 663.0027,  70.5279],
        [ 36.8430,  12.6907,  95.7245, 804.0374],
        [116.4527, 543.9061, 256.4210, 627.2487],
        [286.7128, 541.6981, 429.8167, 628.9529],
        [ 36.1153, 719.2272, 673.5209, 810.8027],
        [109.8641, 193.7978, 608.4915, 285.5848],
        [103.7456, 372.2714, 610.5327, 458.7959],
        [157.1413, 673.6105, 260.0317, 692.0986],
        [ 91.1037, 541.2268, 637.1539, 634.0247],
        [115.5911, 673.6794, 138.7808, 692.2928],
        [548.2986, 748.6394, 604.1763, 761.5536],
        [106.7816,  83.2200, 340.1469, 102.9798],
        [116.3180, 313.2078, 159.9411, 324.2234],
        [ 45.0810, 530.3650, 665.2291, 708.9802]]

image = cv2.imread('./optvgt/page_3.jpg')




def is_contained(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    return (ox1 <= ix1 <= ox2) and (oy1 <= iy1 <= oy2) and (ox1 <= ix2 <= ox2) and (oy1 <= iy2 <= oy2)

def insert_box(box, hierarchy):
    for h_box in hierarchy:
        if is_contained(box, h_box['box']):
            insert_box(box, h_box['children'])
            return
    hierarchy.append({'box': box, 'children': []})

def create_hierarchy(boxes):
    hierarchy = []
    # Sort by area (width * height) descending
    sorted_boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    for box in sorted_boxes:
        insert_box(box, hierarchy)
    return hierarchy

def boxes_to_json(boxes):
    import json
    hierarchy = create_hierarchy(boxes)
    return json.dumps(hierarchy, indent=2)

json_representation = boxes_to_json(boxes)
print(json_representation)




def visualize_boxes(boxes):
    """
    Visualizes boxes sorted in descending order based on area with the origin at the top-left corner.
    The input boxes are specified by their top-left and bottom-right coordinates.

    Parameters:
    boxes (list of lists): Each inner list contains [x1, y1, x2, y2] where
                           (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    """
    # Convert (x1, y1, x2, y2) to (x, y, width, height)
    converted_boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
    
    # Calculate the area of each box
    areas = [(box[2] * box[3], box) for box in converted_boxes]
    
    # Sort boxes by area in descending order
    sorted_boxes = [box for _, box in sorted(areas, key=lambda x: x[0], reverse=True)]
    
    # Set up the plot
    fig, ax = plt.subplots()
    
    # Generate random colors for each box
    colors = [(random.random(), random.random(), random.random()) for _ in range(len(sorted_boxes))]
    
    # Plot each box
    for i, box in enumerate(sorted_boxes):
        x, y, width, height = box
        rect = plt.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.6)
        ax.add_patch(rect)
    
    # Set the limits of the plot to fit all boxes
    all_x = [box[0] for box in sorted_boxes]
    all_y = [box[1] for box in sorted_boxes]
    all_widths = [box[2] for box in sorted_boxes]
    all_heights = [box[3] for box in sorted_boxes]
    max_x = max([x + width for x, width in zip(all_x, all_widths)])
    max_y = max([y + height for y, height in zip(all_y, all_heights)])
    
    ax.set_xlim(0, max_x + 10)
    ax.set_ylim(0, max_y + 10)
    ax.set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.show()

visualize_boxes(boxes)




# Non-Maximum Suppression to mearge redundant boxees
# def calculate_iou(box1, box2):
#     x1, y1, w1, h1 = box1
#     x2, y2, w2, h2 = box2
    
#     intersect_x1 = max(x1, x2)
#     intersect_y1 = max(y1, y2)
#     intersect_x2 = min(x1 + w1, x2 + w2)
#     intersect_y2 = min(y1 + h1, y2 + h2)
    
#     intersect_area = max(0, intersect_x2 - intersect_x1) * max(0, intersect_y2 - intersect_y1)
#     area1 = w1 * h1
#     area2 = w2 * h2
    
#     iou = intersect_area / float(area1 + area2 - intersect_area)
#     return iou

# def non_max_suppression(boxes, threshold):
#     if len(boxes) == 0:
#         return []
    
#     sorted_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][1] + boxes[i][3])
#     selected_indices = []
    
#     while len(sorted_indices) > 0:
#         last = len(sorted_indices) - 1
#         i = sorted_indices[last]
#         selected_indices.append(i)
        
#         suppress = [last]
#         for pos in range(last):
#             j = sorted_indices[pos]
#             if calculate_iou(boxes[i], boxes[j]) > threshold:
#                 suppress.append(pos)
        
#         for idx in sorted(suppress, reverse=True):
#             sorted_indices.pop(idx)
    
#     return selected_indices

# threshold = 0.5  # IoU threshold for NMS
# selected_indices = non_max_suppression(boxes, threshold)
# nonsupressed_boxes = [boxes[i] for i in selected_indices]

# print("nonsupressed Boxes after NMS:")
# for box in nonsupressed_boxes:
#     print(box)

# visualize_boxes(nonsupressed_boxes)




# Crop Individual Boxes
def crop_and_display(image, bbox):
    x1, y1, x2, y2 = bbox
    
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(image.shape[1], int(round(x2)))
    y2 = min(image.shape[0], int(round(y2)))
    
    cropped_image = image[y1:y2, x1:x2]
    
    
    return cropped_image





for crops in boxes:
    cropped_image = crop_and_display(image, crops)
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)







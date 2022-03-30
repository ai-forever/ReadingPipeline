import numpy as np
from shapely.ops import nearest_points
from sklearn.cluster import KMeans
import cv2

from ocrpipeline.metrics import contour2shapely


def add_polygon_center(pred_img):
    """Add center coords for each polygons in the pred_img-dict.

    Args:
        pred_img (dict): The dictionary with predictions.
    """
    for idx, prediction in enumerate(pred_img['predictions']):
        contour = prediction['polygon']
        # compute the center of the contour
        M = cv2.moments(np.array([contour]))
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        pred_img['predictions'][idx]['polygon_center'] = [cX, cY]


def is_page_switched(cluster_centers):
    """Check if page X clusters not sorted."""
    if cluster_centers[0][0] > cluster_centers[1][0]:
        return True
    return False


def is_two_pages(cluster_centers, img_w, max_diff=.25):
    """Check if there are two pages on the image by comparing distance
    between K-means clusters of the image lines.
    """
    center1 = cluster_centers[0][0]
    center2 = cluster_centers[1][0]
    dist = center2 - center1
    diff_ratio = abs(dist) / img_w
    return diff_ratio >= max_diff


def add_page_idx_for_lines(pred_img, line_class_names, img_w, max_diff=.25):
    """Add page indexes for each contours in the pred_img-dict.
    Page is predicted using K-Means via line polygons.

    Args:
        pred_img (dict): The dictionary with predictions.
        line_class_names (list): The list of line class names.
        img_w (int): The image width.
    """
    x_coords = []
    indexes = []
    for idx, prediction in enumerate(pred_img['predictions']):
        contour_center = prediction['polygon_center']
        if prediction['class_name'] in line_class_names:
            x_coords.append(contour_center[0])
            indexes.append(idx)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(
        np.array(x_coords).reshape(-1, 1))
    if is_two_pages(kmeans.cluster_centers_, img_w, max_diff):
        if is_page_switched(kmeans.cluster_centers_):
            page_indexes = [0 if page == 1 else 1 for page in kmeans.labels_]
        else:
            # int() to make json serializable # int() to make json serializable
            page_indexes = [int(page) for page in kmeans.labels_]
    else:  # only one page on the image
        page_indexes = [0 for i in range(len(indexes))]

    for idx, page_idx in zip(indexes, page_indexes):
        pred_img['predictions'][idx]['page_idx'] = page_idx


def add_line_idx_for_lines(pred_img, line_class_names):
    """Add line indexes for line contours in the pred_img-dict.
    Line index is calculated by sorting line contours by their y-mean coords.

    Args:
        pred_img (dict): The dictionary with predictions.
        line_class_names (list): The list of line class names.
    """
    page_indexes = get_page_indexes(pred_img)
    for page_idx in page_indexes:
        y_means = []
        indexes = []
        for idx, prediction in enumerate(pred_img['predictions']):
            if (
                prediction['class_name'] in line_class_names
                and prediction['page_idx'] == page_idx
            ):
                y_means.append(prediction['polygon_center'][1])
                indexes.append(idx)
        indexes_y_means = sorted(
            zip(indexes, y_means), key=lambda x: x[1], reverse=False)
        for line_idx, (idx, y_mean) in enumerate(indexes_y_means):
            pred_img['predictions'][idx]['line_idx'] = line_idx


def get_polygons_distance(polygon1, polygon2):
    """Get distance between two polygons.

    Args:
        polygon1 (shapely.Polygon): The first polygon.
        polygon2 (shapely.Polygon): The second polygon.
    """
    if polygon1 is not None and polygon2 is not None:
        return polygon1.distance(polygon2)
    return None


def get_polygons_closest_points(contour1, contour2):
    """Get closest points between two polygons.

    Args:
        polygon1 (list of [x, y]): The first contour.
        polygon2 (list of [x, y]): The second contour.
    """
    polygon1 = contour2shapely(contour1)
    polygon2 = contour2shapely(contour2)
    if polygon1 is not None and polygon2 is not None:
        return [(geom.xy[0][0], geom.xy[1][0])
                for geom in nearest_points(polygon1, polygon2)]
    return None


def get_idx_of_line_closest_to_word(word_contour, pred_img, line_class_names):
    """Get the index of the line closest to the input word contour.

    Args:
        word_contour (list of [x, y] coords): The contour of the word.
        pred_img (dict): The dictionary with predictions.
        line_class_names (list): The list of line class names.
    """
    min_polygon_distance = np.inf
    idx_of_line = None
    word_shapely = contour2shapely(word_contour)

    indexes = []
    line_shapelys = []
    for idx, prediction in enumerate(pred_img['predictions']):
        if prediction['class_name'] in line_class_names:
            indexes.append(idx)
            line_shapelys.append(contour2shapely(prediction['polygon']))

    for idx, line_shapely in zip(indexes, line_shapelys):
        polygons_distance = get_polygons_distance(line_shapely, word_shapely)
        if polygons_distance is not None:
            if polygons_distance == 0:
                return 0, idx
            elif polygons_distance < min_polygon_distance:
                min_polygon_distance = polygons_distance
                idx_of_line = idx
    return min_polygon_distance, idx_of_line


def is_word_above_line(line_center, word_center):
    """Check if the word is above the line by comparing their central
    coords."""
    if word_center[1] <= line_center[1]:
        return True
    return False


def add_line_idx_for_words(pred_img, line_class_names, word_class_names):
    """Add line indexes for each word polygon in the pred_img-dict.
    The word contour must intersect with the line contour to determine the
    line index for the word.

    Args:
        pred_img (dict): The dictionary with predictions.
        word_class_names (list): The list of word class names.
        line_class_names (list): The list of line class names.
    """
    for prediction in pred_img['predictions']:
        if prediction['class_name'] in word_class_names:
            dist, idx = get_idx_of_line_closest_to_word(
                prediction['polygon'], pred_img, line_class_names)
            if (
                idx is not None
                and dist == 0
            ):
                prediction['page_idx'] = \
                    pred_img['predictions'][idx]['page_idx']
                prediction['line_idx'] = \
                    pred_img['predictions'][idx]['line_idx']


def add_column_idx_for_words(pred_img, word_class_names):
    """Add column indexes for word contours in the pred_img-dict.
    Column index is calculated by sorting word contours from one line
    by their x-mean coords.

    Args:
        pred_img (dict): The dictionary with predictions with page_idx and
            line_idx keys.
        word_class_names (list): The list of word class names.
    """
    page_indexes = get_page_indexes(pred_img)
    for page_idx in page_indexes:
        line_indexes = get_line_indexes(pred_img, page_idx)
        for line_idx in line_indexes:
            indexes = []
            x_means = []
            for idx, prediction in enumerate(pred_img['predictions']):
                if (
                    prediction.get('page_idx') == page_idx
                    and prediction.get('line_idx') == line_idx
                    and prediction['class_name'] in word_class_names
                ):
                    indexes.append(idx)
                    x_means.append(prediction['polygon_center'][0])
            indexes_x_means = sorted(
                zip(indexes, x_means), key=lambda x: x[1], reverse=False)
            for column_idx, (idx, x_mean) in enumerate(indexes_x_means):
                pred_img['predictions'][idx]['column_idx'] = column_idx


def get_page_indexes(pred_img):
    """Get list of sorted unique page indexes from pred_img-dict.

    Args:
        pred_img (dict): The dictionary with predictions.
    """
    unique_page_indexes = set()
    for prediction in pred_img['predictions']:
        page_idx = prediction.get('page_idx')
        if page_idx is not None:
            unique_page_indexes.add(page_idx)
    return sorted(list(unique_page_indexes), reverse=False)


def get_line_indexes(pred_img, page_idx):
    """Get list of sorted unique line indexes from a given page
    from pred_img-dict.

    Args:
        pred_img (dict): The dictionary with predictions with page_idx and
            line_idx keys.
        page_idx (int): The page index.
    """
    unique_line_indexes = set()
    for prediction in pred_img['predictions']:
        if prediction.get('page_idx') == page_idx:
            line_idx = prediction.get('line_idx')
            if line_idx is not None:
                unique_line_indexes.add(line_idx)
    return sorted(list(unique_line_indexes), reverse=False)


def get_column_indexes(pred_img, page_idx, line_idx):
    """Get list of sorted unique column indexes from a given page and line
    from pred_img-dict.

    Args:
        pred_img (dict): The dictionary with predictions with page_idx,
            line_idx and column_idx keys.
        page_idx (int): The page index.
        line_idx (int): The line index.
    """
    unique_column_indexes = set()
    for prediction in pred_img['predictions']:
        if (
            prediction.get('page_idx') == page_idx
            and prediction.get('line_idx') == line_idx
        ):
            column_idx = prediction.get('column_idx')
            if column_idx is not None:
                unique_column_indexes.add(column_idx)
    return sorted(list(unique_column_indexes), reverse=False)


def get_structured_text(pred_img, word_class_names):
    """Get list of texts from pred_img-dict, ordered by pages and lines.

    Args:
        pred_img (dict): The dictionary with predictions with page_idx,
            line_idx and column_idx keys.
        word_class_names (list): The list of word class names that would be
            selected as output texts.
    """
    structured_text = []
    for page_idx in get_page_indexes(pred_img):
        structured_page = []
        for line_idx in get_line_indexes(pred_img, page_idx):
            structured_line = []
            for column_idx in get_column_indexes(pred_img, page_idx, line_idx):
                for prediction in pred_img['predictions']:
                    if (
                        prediction.get('page_idx') == page_idx
                        and prediction.get('line_idx') == line_idx
                        and prediction.get('column_idx') == column_idx
                        and prediction['class_name'] in word_class_names
                    ):
                        structured_line.append(prediction['text'])
            structured_page.append(structured_line)
        structured_text.append(structured_page)
    return structured_text


def add_word_indexes(pred_img, word_class_names):
    """Add word indexes to pred json.

    Args:
        pred_img (dict): The dictionary with predictions with page_idx,
            line_idx and column_idx keys.
        word_class_names (list): The list of word class names.
    """
    word_idx = 0
    for page_idx in get_page_indexes(pred_img):
        for line_idx in get_line_indexes(pred_img, page_idx):
            for column_idx in get_column_indexes(pred_img, page_idx, line_idx):
                for prediction in pred_img['predictions']:
                    if (
                        prediction.get('page_idx') == page_idx
                        and prediction.get('line_idx') == line_idx
                        and prediction.get('column_idx') == column_idx
                        and prediction['class_name'] in word_class_names
                    ):
                        prediction['word_idx'] = word_idx
                        word_idx += 1

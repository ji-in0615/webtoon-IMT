import cv2
import opt
import imgproc

g_storage = []  # global variable to store divided image information.

# image shape
def bfs(img, h_criteria):
    h, w, _ = img.shape
    arr = []
    arr.append(h_criteria)
    mag = 1
    while arr:
        h_c = arr.pop(0)
        if h_c < 0 or h_c >= h:
            print('[error]Image out of index')
            return 'err'
        if h_c < h_criteria - opt.PIXEL_THRESHOLD or h_c >= h_criteria + opt.PIXEL_THRESHOLD:
            print('It takes too long times. so, cut down image as initial criteria')
            return h_criteria

        if (img[h_c] == 0).all() or (img[h_c] == 255).all(): return h_c
        arr.append(h_criteria + mag)
        arr.append(h_criteria - mag)
        mag += 1

# resize
def down_size_image(img, width=None):
    h, w, _ = img.shape
    ratio_h = int((width * h) // w)
    criteria_w = w / width
    img = cv2.resize(img, (width, ratio_h), interpolation=cv2.INTER_CUBIC)
    return img, criteria_w

# search pixel
def search_pixels(origin=None, copy=None, criteria_h=None, creteria_w=None, name=None, index=None):
    global g_storage
    h, _, _ = copy.shape
    if h <= criteria_h:  # recursive exit statement
        piece = origin[0:int(h * creteria_w), :, :]
        if h >= opt.MIN_SIZE:  # If final image is too small, It considers residue of image
            g_storage.append(piece)
        return

    n_h = bfs(copy, criteria_h)  # bfs to find 0 or 255 pixels per line
    piece = origin[0:int(n_h * creteria_w), :, :]
    g_storage.append(piece)

    search_pixels(origin=origin[int(n_h * creteria_w):, :, :], copy=copy[n_h:, :, :], criteria_h=criteria_h,
                  creteria_w=creteria_w, name=name, index=index + 1)  # recursive

# image cut
def cut_off_image(image=None, name=None, ratio=None):
    global g_storage
    g_storage = []
    if image.shape[1] >= 720:  # If image is too large, reduce size.
        h, w, _ = image.shape
        ratio_h = int((720 * h) // w)
        image = cv2.resize(image, (720, ratio_h), interpolation=cv2.INTER_CUBIC)

    copy = imgproc.cpImage(img=image)
    copy, criteria_w = down_size_image(copy, width=500)  # Resize image for searching pixels to decrease times.
    height, width, _ = image.shape
    copy_height, copy_width, _ = copy.shape
    criteria_h = int(copy_width * ratio)
    # Breadth-first search for cutting criteria line.
    search_pixels(origin=image, copy=copy, criteria_h=criteria_h, creteria_w=criteria_w, name=name, index=0)

    return g_storage

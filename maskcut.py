import cv2
import numpy as np
import dlib
import math

face_p = list(range(17, 68))
mouth_p = list(range(48, 68))
r_brow_p = list(range(17, 22))
l_brow_p = list(range(22, 27))
r_eye_p = list(range(36, 42))
l_eye_p = list(range(42, 48))
nose_p = list(range(27, 36))
jaw_p = list(range(0, 17))

mm = [
    mouth_p + [48,67]
]
nose = [
    nose_p + [27,31,27,35]
]
rb = [
    r_brow_p + [17,21],
]
lb = [
    l_brow_p + [22,26],
]
rr = [
    r_eye_p + [36,39] ,
]
ll = [
    l_eye_p +[42,45],
]
def get_landmarks(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 人脸数rects
    rects = detector(img_gray, 0)
    #人脸位置矩形框出
    # for _, d in enumerate(rects):
    #     cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 3)
    #返回68*2特征点
    s = np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
    print('人脸识别获取特征点成功!')
    return s

def annotate_landmarks(img, landmarks):
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4,color=(0, 0, 255))
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    # cv2.imshow('note',img)
    print('人脸标记完成!')
    return img

def draw_convex_hull(img, points, color):
    points = cv2.convexHull(points) # 得到凸包
    cv2.fillConvexPoly(img, points, color=color) # 绘制填充

def get_face_mask(img, landmarks):
    img2 = np.zeros(img.shape[0:2], dtype="uint8")
    img2.fill(255)
    for group in rr:
        draw_convex_hull(img2,landmarks[group],color=0)
    for group in ll:
        draw_convex_hull(img2,landmarks[group],color=0)
    for group in lb:
        draw_convex_hull(img2,landmarks[group],color=0)
    for group in rb:
        draw_convex_hull(img2,landmarks[group],color=0)
    for group in nose:
        draw_convex_hull(img2,landmarks[group],color=0)
    for group in mm:
        draw_convex_hull(img2,landmarks[group],color=0)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img2, kernel)  # 腐蚀
    # cv2.imshow('facemask', erosion)
    # cv2.imwrite('fmask.jpg', erosion)
    print('获取掩模完成!')
    return erosion

def get_hair_mask(img):
    #交互式门限处理提取头发区域
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gauss = cv2.GaussianBlur(gray, (9, 9), 1)
    maxvalue = 255
    def onthreshold(x):
        value = cv2.getTrackbarPos("value", "Threshold")
        a, binary = cv2.threshold(gauss, value, maxvalue, cv2.THRESH_BINARY)
        cv2.imshow("Threshold", binary)
        return binary
    cv2.namedWindow("Threshold",cv2.WINDOW_NORMAL)
    while(1):
        cv2.createTrackbar("value", "Threshold", 0, 255, onthreshold)
        if cv2.waitKey(0) == 13:
            final = onthreshold(1)
            cv2.destroyWindow("Threshold")
            break
    #获取头发区域掩模
    mask = np.zeros(img.shape[0:2], dtype="uint8")
    mask.fill(255)
    h, w, _ = img.shape
    #查找轮廓
    contours,hierarchy = cv2.findContours(final,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    c_max =[]
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        # 处理掉小的轮廓区域
        if (area < (h / 5 * w / 5)):
            c_min = []
            c_min.append(cnt)
            # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。
            continue
        c_max.append(cnt)
    cv2.drawContours(mask, c_max, -1, (0, 0, 0), thickness=-1)
    mask = 255 - mask
    # cv2.imshow('hairmask',mask)
    # cv2.imwrite("cuth.jpg", mask)
    print('获取掩模完成!')
    return mask

def maskcut(img2 , mask):
    img = img2.copy()
    rows, cols, channels = img.shape  # rows，cols是前景图片
    # 遍历替换
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] == 0:  # 0代表黑色的点
                img[i, j] = img[i, j]  # 此处替换颜色，为BGR通道
            else:
                img[i,j] = 255
    # 显示图片
    # cv2.imshow('mcut',img)
    # cv2.imwrite('fcut.jpg', img)
    print('图像分割完成!!')
    return img

def cartoonise(img,color,lamk):
    # 获取人脸掩模并进行分割
    facemask = get_face_mask(img, lamk)
    cutface = maskcut(img, facemask)
    # 人脸五官处理
    finaface = kmeans(cutface,8)
    # 五官的加入
    fist = hecheng(finaface,color,facemask)
    # 图像边缘
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # 图像梯度
    xgrad = cv2.Sobel(blurred, cv2.CV_16SC1, 1, 0)
    ygrad = cv2.Sobel(blurred, cv2.CV_16SC1, 0, 1)
    # 计算边缘
    final = cv2.Canny(xgrad, ygrad, 30, 30*3)
    # cv2.imshow("Canny", final)
    # 50和150参数必须符合1：3或者1：2
    edge_output = 255 - final
    img_edge = cv2.cvtColor(edge_output, cv2.COLOR_GRAY2RGB)
    img_cartoon = cv2.bitwise_and(img_edge, fist)
    # cv2.imshow("biany", img_edge)
    # cv2.imshow("fist", fist)
    # cv2.imshow("cartoon", img_cartoon)
    print('合成完成!!')
    return img_cartoon

def kmeans(img,k):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    # cv2.imshow(str(("spaceship K=4")), res2)
    return res2

def hecheng(fg,bg,mask):
    rows, cols, channels = fg.shape  # rows，cols是前景图片
    # 遍历替换
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] == 0:  # 0代表黑色的点
                bg[i, j] = fg[i, j]  # 此处替换颜色，为BGR通道
    # cv2.imshow('hecheng', bg)
    # cv2.imwrite('result.jpg', bg)
    print('融合完成!')
    return bg

def lunkua(img,landmarks):
    for i in range(5,11):
        pos = (landmarks[i, 0], landmarks[i, 1])
        try:
            pos2 = (landmarks[i+1, 0], landmarks[i+1, 1])
        except:
            pos2 = (landmarks[11, 0], landmarks[11, 1])
        cv2.line(img, pos, pos2,color=(0, 0, 0),thickness=1)
    # cv2.imshow('lunkua',img)
    # cv2.imwrite('lunkua.jpg', img)
    print('下巴轮廓填充完成!')
    return img

def zhengti(haha,lamk):
    rows, cols, _ = haha.shape
    zhong = [ lamk[21,1],lamk[33,1]]
    xia = [ lamk[33,1],lamk[8,1]]
    p1 = (lamk[21, 0], lamk[21, 1])
    p2 = (lamk[22, 0], lamk[22, 1])
    pz = ((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)
    pb = (lamk[33, 0], lamk[33, 1])
    px = (lamk[8, 0], lamk[8, 1])
    dis1 = math.sqrt((pz[0]-pb[0])*(pz[0]-pb[0]) + (pz[1]-pb[1])*(pz[1]-pb[1]))
    dis2 = math.sqrt((pb[0]-px[0])*(pb[0]-px[0]) + (pb[1]-px[1])*(pb[1]-px[1]))
    dis = dis1/dis2
    if (dis>1):
        p = zhong
    else:
        p =xia
    cropped = haha[p[0]:p[1], 0:cols]
    ss = cv2.resize(cropped,(cols,int(p[1]-p[0]+rows*0.04)),interpolation=cv2.INTER_NEAREST)
    a,b,__ = ss.shape
    outimg = np.zeros((int(rows*1.04),cols,3), np.uint8)
    for y in range(int(rows*1.04)):
        for x in range(cols):
            if y<p[0]:
                outimg[y, x] = haha[y, x]
            if  p[0] <= y < (p[0]+a):
                outimg[y,x] = ss[y-p[0],x]
            if  y >= (p[0]+a):
                outimg[y, x] = haha[y-a+p[1]-p[0], x]
    # cv2.imshow('zhenti',outimg)
    return outimg

def jubu(haha, lamk):
    rows, cols, _ = haha.shape
    yanjing = [lamk[38, 1], lamk[41, 1]]
    bizi = [lamk[29, 1], lamk[33, 1]]
    py1 = (lamk[36, 0], lamk[36, 1])
    py2 = (lamk[39, 0], lamk[39, 1])
    pb1 = (lamk[31, 0], lamk[31, 1])
    pb2 = (lamk[35, 0], lamk[35, 1])
    dis1 = math.sqrt((py1[0]-py2[0])*(py1[0]-py2[0]) + (py1[1]-py2[1])*(py1[1]-py2[1]))
    dis2 = math.sqrt((pb1[0]-pb2[0])*(pb1[0]-pb2[0]) + (pb1[1]-pb2[1])*(pb1[1]-pb2[1]))
    dis = dis1/dis2
    if (dis>1):
        p = yanjing
    else:
        p =bizi
    cropped = haha[p[0]:p[1], 0:cols]
    ss = cv2.resize(cropped, (cols, int(p[1] - p[0] + rows * 0.04)), interpolation=cv2.INTER_NEAREST)
    a, b, __ = ss.shape
    outimg = np.zeros((int(rows * 1.04), cols, 3), np.uint8)
    for y in range(int(rows * 1.04)):
        for x in range(cols):
            if y < p[0]:
                outimg[y, x] = haha[y, x]
            if p[0] <= y < (p[0] + a):
                outimg[y, x] = ss[y - p[0], x]
            if y >= (p[0] + a):
                outimg[y, x] = haha[y - a + p[1] - p[0], x]
    cv2.imshow('jubu', outimg)
    return outimg

def star(img):
    # 执行卡通化过程
    # 获取颜色图层
    color_img = kmeans(img,4)
    # 获取特征点
    lamk = get_landmarks(img)
    # 各个分量合成
    fina = cartoonise(img,color_img,lamk)
    # 获取头发掩模以及头发分割
    hairmask = get_hair_mask(img)
    haircut = maskcut(img, hairmask)
    # 头发处理
    hairgray = cv2.cvtColor(haircut, cv2.COLOR_RGB2GRAY)
    hairgray = cv2.cvtColor(hairgray, cv2.COLOR_RGB2BGR)
    # 对头发掩模腐蚀与膨胀
    erode = cv2.erode(hairmask, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=1)
    hehair = hecheng(hairgray, fina, dilate)
    # 轮廓修正
    lunk = lunkua(hehair,lamk)
    kz = zhengti(lunk,lamk)
    lamkz = get_landmarks(kz)
    final = jubu(kz,lamkz)
    # cv2.imwrite('ad/k50.jpg', lunk)
    # cv2.imwrite('ad/b50.jpg', final)

if __name__ == '__main__':
    img = cv2.imread('ad/4.jpg')
    star(img)
    cv2.waitKey()
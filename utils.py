import numpy as np
import cv2

def edgeOrientation(image):
  gX = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
  gY = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
  edge_orientation = np.arctan2(gY, gX) + np.pi
  
  return edge_orientation

def getRTable(template, degrees):
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(template,50,255)
    orientations = edgeOrientation(edges)
    r_table = [[] for i in range(degrees)]
    center_point = [int(template.shape[0]/2), int(template.shape[1]/2)]

    for (i, j), value in np.ndenumerate(edges):
        if value:
            # compute r and alpha for rtable
            r = np.sqrt((center_point[0] - i) ** 2 + (center_point[1] - j) ** 2)
            alpha = np.arctan2(i - center_point[0], j - center_point[1]) + np.pi
            
            # find index of rtable
            index = (int(degrees * orientations[i, j] / (2 * np.pi)))%degrees
            # print(index)
            r_table[index].append((r, alpha))

    return r_table

def computeAccumulator(match_im, r_table):

  edges = cv2.Canny(match_im,50,255)
  orientations = edgeOrientation(edges)
  accumulator = np.zeros(match_im.shape)
  r_table_degrees = len(r_table)
  vals = []
  for (i, j), value in np.ndenumerate(edges):
    #   check if significant pixel
      if value:
          index = (int(r_table_degrees * orientations[i, j] / (2 * np.pi)))%r_table_degrees

          r_row = r_table[index]
          for (r,alpha) in r_row:

              # find px place and  vote
              accum_i = int(i + r * np.sin(alpha))
              accum_j = int(j + r * np.cos(alpha))
              if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1] and accum_i > 0 and accum_j > 0:
                  accumulator[accum_i, accum_j] += 1

  return accumulator

def genHoughTransformMatch(image, template, rTable):
    acc = computeAccumulator(image,rTable)
    ridx, cidx = np.unravel_index(acc.argmax(), acc.shape)

    # find the half-width and height of template
    hheight = np.floor(template.shape[0] / 2) + 1
    hwidth = np.floor(template.shape[1] / 2) + 1

    # find coordinates of the box
    rs = int(max(ridx - hheight, 1))
    re = int(min(ridx + hheight, image.shape[0] - 1))
    cs = int(max(cidx - hwidth, 1))
    ce = int(min(cidx + hwidth, image.shape[1] - 1))

    return acc,rs,re,cs,ce

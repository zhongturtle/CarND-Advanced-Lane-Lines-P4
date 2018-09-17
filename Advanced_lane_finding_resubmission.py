

"""
This code was mostly taken from udacity carnd classroom Lesson 15 Advanced lane finding 

And the functionalities were tested and compared with the code in the following repository 
https://github.com/jeremy-shannon/CarND-Advanced-Lane-Lines/blob/master/project.ipynb
"""
#------------------- Importing Libraries ------------------------------------------------------------------------------------------------------------#
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

#------------------- Calibrate Camera Once  ---------------------------------------------------------------------------------------------------#

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

objp = np.zeros((6*9,3), np.float32) # first we construct a 3d array of all zeros, in our 2D image target we always have Z=0
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2) # so, we keep Z=0 as it is and we generate a grid corresponding to our x,y corners preknown numbers 

images = glob.glob('camera_cal/calibration*.jpg')
for i, filename in enumerate(images):
    img = cv2.imread(filename)
    img_size = (img.shape[1], img.shape[0]) # get image shape in the format Width, Height
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert image to gray in order for the next function to work properly
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None) # function to find the chessboard-corener in gray image
    if ret == True: # if corners were found
            imgpoints.append(corners) # add corners point to the image point array
            objpoints.append(objp) # add the prepared object points to the object points array 
  
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None) #calibrate camera and calculate camera matrix 
            
            
    
#------------------- Defining necessary functions ---------------------------------------------------------------------------------------------------#
def Undistort (image,mtx, dist): # function to undistort image based on object points and image points already determined
    
    cv2.undistort(image, mtx, dist, None, mtx) # undistort images remap the original image with distortion compensation 
    return image

def Birdeye (image):
    
    # defining checkpoints relative to image size 
    midpoint = image.shape[1]//2 
    farOffset = image.shape[1]//10
    nearOffset = image.shape[1]//2
    length = image.shape[0]//3
    bottom_margin = 0
    
    # defining source points and destination points which take a trapezoid of interest and map it to the flat image 
    src = np.float32([[midpoint-nearOffset, image.shape[0]-bottom_margin], [midpoint+nearOffset, image.shape[0]-bottom_margin],
                     [midpoint-farOffset,image.shape[0]-length], [midpoint+farOffset, image.shape[0]-length]])

    dst = np.float32([[25, image.shape[0]-25], [image.shape[1]-25, image.shape[0]-25],
                     [25,25], [image.shape[1]-25, 25]])
    
    img_size = (image.shape[1], image.shape[0]) # get image shape in the format Width, Height
    
    M = cv2.getPerspectiveTransform(src, dst) # calculating the linear transformation matrix to be applied for perspective shift
    Minv= np.linalg.inv(M) # calculating the inverse transformation matrix for remapping drawing to original perspective afterwards
    warped= cv2.warpPerspective(image, M, img_size) # map the images according to the transformation matrix
    return warped,Minv 



def pipeline(img, s_thresh=(90, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)# Convert to HLS color space and separate the V channel
    l_channel = hls[:,:,1] # saving L channel
    s_channel = hls[:,:,2] # saving S channel
   
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx)) # scale back the image values
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel and output an image value where one of both channels have a value
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    color_binary[(s_binary == 1) | (sxbinary ==1)]=1 
    color_binary=color_binary*255
    
    return color_binary
    
def Findlines (binary_warped):
    
    global left_fit
    global right_fit
    
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0) # sum the value of pixels over the x axis to get positional information 
    
    midpoint = np.int(histogram.shape[0]//2) # define x axis midpoint
    leftx_base = np.argmax(histogram[:midpoint]) # find the position of highest peak in the left half
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # find the position of highest peak in the right half
    
    nwindows = 9 # number of search windows
    window_height = np.int(binary_warped.shape[0]/nwindows) # height of search window
    nonzero = binary_warped.nonzero() # get the indices of all nonzeroelements in every dimension 
    nonzeroy = np.array(nonzero[0]) # indices of all nonzero elements in Y dimension
    nonzerox = np.array(nonzero[1]) # indices of all nonzero elements in X dimension
    leftx_current = leftx_base # set the starting search point as the position of left peak
    rightx_current = rightx_base # set the starting search point as the position of right peak
    margin = 50 # margin of search window
    minpix = 50 # minimum number of pixels to recenter
    
    left_lane_inds = [] 
    right_lane_inds = []
    
    if (left_fit!= [] and right_fit!=[]): # as they are global variables I am checking if this is the first frame or there is an existing history  
        
        # find indices that are out of line range but still within the margin of search        
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
            
            
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        if((left_fit[0]-right_fit[0])>1e-4 and (left_fit[1]-right_fit[1])>1e-1) or abs(left_fit[2]-right_fit[2])<(rightx_base-leftx_base) : # in case left and right lines coefficient are far so the lines are not parallel and we need a search window 
                left_lane_inds = []
                right_lane_inds = []
                for window in range(nwindows):
                     
                    #indices of current search window 
                    win_y_low = binary_warped.shape[0] - (window+1)*window_height 
                    win_y_high = binary_warped.shape[0] - window*window_height
                    win_xleft_low = leftx_current - margin
                    win_xleft_high = leftx_current + margin
                    win_xright_low = rightx_current - margin
                    win_xright_high = rightx_current + margin
                    
                    # get the non zero values inside the search window 
                    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                    left_lane_inds.append(good_left_inds)
                    right_lane_inds.append(good_right_inds)
                    if len(good_left_inds) > minpix:
                        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                    if len(good_right_inds) > minpix:        
                        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                 
                 # now we have indices for left and right lane line  
                left_lane_inds = np.concatenate(left_lane_inds)
                right_lane_inds = np.concatenate(right_lane_inds)
                
                # Again, extract left and right line pixel positions
                leftx = nonzerox[left_lane_inds]
                lefty = nonzeroy[left_lane_inds] 
                rightx = nonzerox[right_lane_inds]
                righty = nonzeroy[right_lane_inds] 
                
                # make a polynomial fit for both lines 
                left_fit = np.polyfit(lefty, leftx, 2)
                right_fit = np.polyfit(righty, rightx, 2)
    #-----
    else:
        for window in range(nwindows):
            #indices of current search window 
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # get the non zero values inside the search window 
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
         # now we have indices for left and right lane line  
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        # make a polynomial fit for both lines 
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
    # Generate x and y values for plotting and restore into the equation ax^2+ bx+c=0
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] ) # generate a grid with all points
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] # left fit line points
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] # right fit line points 
    
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    
    '''
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    '''
    
    y_eval = np.max(ploty)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/(rightx_base-leftx_base) # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    h = binary_warped.shape[0]
    if right_fit is not None and left_fit is not None:
        car_position = binary_warped.shape[1]/2
        l_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        r_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
    return left_curverad, right_curverad, center_dist, left_fit, right_fit
    


    
def draw_lane(original_img, binary_img, l_fit, r_fit, Minv):
    new_img = np.copy(original_img)
    
    if l_fit is None or r_fit is None:
        return original_img
    # Create an image to draw the lines on
    warp_zero=np.copy(binary_img)*0
    warp_zero =  np.zeros_like(binary_img[:,:,0].astype(np.uint8))
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w,z = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    colorwarp= cv2.fillPoly(color_warp, np.int_([pts]),color=(0,255,0))
    cv2.polylines(colorwarp, np.int32([pts_left]), False,(int(255),int(0), int(0)), 15)
    cv2.polylines(colorwarp, np.int32([pts_right]),False,(int(0),int(0), int(255)), 15)
    
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp= cv2.warpPerspective(colorwarp, Minv, (w, h))
    #plt.imshow(newwarp)
    
    
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.8, 0)
    return result, colorwarp

def draw_data(original_img, curv_rad, center_dist):
    new_img = np.copy(original_img)
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(new_img, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    return new_img


def process(frame): # video frame processing pipeline
    undist= Undistort (frame,mtx, dist) #undistort
    warped,Minv= Birdeye(undist) # warp prespective
    combined_binary1=pipeline(warped) # make it binary
    combined_binary2= combined_binary1[:,:,0] # make the binary 2D 
    left_curverad, right_curverad, center_dist, left_fit, right_fit= Findlines (combined_binary2) # findlines on binary
    curv_rad= (left_curverad+right_curverad)/2 # calculate average curvature of both lines
    result, newwarp= draw_lane(undist, combined_binary1,left_fit, right_fit, Minv) # draw lane on frame
    newimg= draw_data(result, curv_rad, center_dist) # draw measurment data on frame
    return newimg

#----------------------------------------------------------------------------------------------------------------------------------------------------#

global left_fit
global right_fit

left_fit=[]
right_fit=[]


 #--------------------------------------------------------------------------------------------------   

video_output1 = 'project_output_resubmission.mp4'
#video_input1 = VideoFileClip('project_video.mp4')#.subclip(36,50)
video_input1 = VideoFileClip('challenge_video.mp4')#.subclip(36,50)
processed_video = video_input1.fl_image(process)
processed_video.write_videofile(video_output1, audio=False)

#---------------------------------------------------------------------------------------------------
'''
cap = cv2.VideoCapture('challenge_video.mp4')
cv2.namedWindow('frames')

while(cap.isOpened()):
    ret, image = cap.read()

    if ret is True:
        image = process(image)
        cv2.imshow('frames', image) # show the combined image
        k= cv2.waitKey(5) & 0xFF
    
        if k==15:
         break
     
cv2.destroyAllWindows()
cap.release()        
'''
#--------------------------------------------------------------------------------------------------
'''
test= glob.glob('test_images/*.jpg')

for t in test :
    image= cv2.imread(t)
    image = process(image)
    plt.imshow(image)
'''

       
      

    

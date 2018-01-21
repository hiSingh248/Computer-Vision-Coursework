# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def calc_histogram(histogram):
   # returns the cumulative sum of the elements along a given axis.
   cdf=np.cumsum(histogram) 
   # normalizes the cumulative distributive function    
   cdf_normalized = cdf * histogram.max()/ cdf.max()
    # find the minimum histogram value (excluding 0) and masks the cdf     
   cdf_m = np.ma.masked_equal(cdf,0)
   cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())       
   cdf = np.ma.filled(cdf_m,0).astype('uint8')
   return cdf
   
def histogram_equalization(img_in):

   # splits channels of a color image
   
   b,g,r=cv2.split(img_in)
   # calculates histograms for blue channel
   histogram1 = cv2.calcHist([img_in],[0],None,[256],[0,256])
   #calculates histogram
   cdf=calc_histogram(histogram1)
   # calculates image for blue histogram
   img_b=cdf[b]
   
   # calculates histograms for green channel
   histogram2 = cv2.calcHist([img_in],[1],None,[256],[0,256])
   cdf=calc_histogram(histogram2)
   img_g=cdf[g]
 
   # calculates histograms for red channel
   histogram3 = cv2.calcHist([img_in],[1],None,[256],[0,256])
   cdf=calc_histogram(histogram3)
   img_r=cdf[r]
 
   img_out= cv2.merge((img_b,img_g,img_r))       
   #img_out = img_final # Histogram equalization result 
   
   return True, img_out
   
def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "1.jpg"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):
	
   # finds fourier transform
   f = np.fft.fft2(img_in)
   # shifts the result by N/2 in both the directions to bring the zero frequency component from top left corner to center
   fshift = np.fft.fftshift(f)

   magnitude_spectrum = 20*np.log(np.abs(fshift))

   # removing high frequency contents by applying a 20x20 rectangle window   
   rows, cols = img_in.shape
   crow = int(rows/2)
   ccol = int(cols/2)   
   original = np.copy(fshift)
   fshift[crow-10:crow+10, ccol-10:ccol+10] = 0
   # substracts the image with low frequencies removed from the original image and applies inverse shift to bring the zero frequency component from center to top left corner
   f_ishift = np.fft.ifftshift(original - fshift)
   # finds the inverse Fourier Transform
   img_back = np.fft.ifft2(f_ishift)
   # finds the inverse Fourier Transform
   img_back = np.abs(img_back)   
   img_out = img_back # Low pass filter result
	
   return True, img_out

def high_pass_filter(img_in):

  
   # finds fourier transform 
   f = np.fft.fft2(img_in)
   # shifts the result by N/2 in both the directions to bring the zero frequency component from top left corner to center
   fshift = np.fft.fftshift(f)
   magnitude_spectrum = 20*np.log(np.abs(fshift))
   # removing low frequency contents by applying a 20x20 rectangle window
   rows, cols = img_in.shape
   crow = int(rows/2)
   ccol = int(cols/2)   
   fshift[crow-10:crow+10, ccol-10:ccol+10] = 0

   # applies inverse shift to bring the zero frequency component from center to top left corner 
   f_ishift = np.fft.ifftshift(fshift)
   # finds the inverse Fourier Transform
   img_back = np.fft.ifft2(f_ishift)   
   img_back = np.abs(img_back) 


   img_out = img_back # High pass filter result
   
   return True, img_out

def ft(img, newsize=None):
    #finds the fourier transform 
    dft = np.fft.fft2(np.float32(img),newsize)
    # returns the shifted fourier transform
    return np.fft.fftshift(dft)
   
def ift(shift):
    # applies the inverse shift
    f_ishift = np.fft.ifftshift(shift)  
    # finds the inverse fourier transform  
    img_back = np.fft.ifft2(f_ishift)    
    return np.abs(img_back)

   
def deconvolution(img_in):
   
   # gets the gaussian kernel for de-blurring convoluted image
   gk = cv2.getGaussianKernel(21,5)
   gk = gk * gk.T
   # finds the fourier transform of convoluted image keeping equal size for both the transform 
   imf = ft(img_in, (img_in.shape[0],img_in.shape[1]))
   # finds the fourier transform of gaussian kernel keeping equal size for both the transform 
   gkf = ft(gk, (img_in.shape[0],img_in.shape[1])) 
   # divides the fourier transform of convoluted image by fourier transform of gaussian kernel
   imconvf = np.divide(imf,gkf)

   # reconstructs the blurred image from the FT
   deblurred = ift(imconvf)
   img_out = deblurred *255 # Deconvolution result
   
   return True, img_out

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], 0);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "2.jpg"
   output_name2 = sys.argv[4] + "3.jpg"
   output_name3 = sys.argv[4] + "4.jpg"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here
   
    # Read in input images
   img_in1 = cv2.imread(img_in1, cv2.IMREAD_COLOR);
   img_in2 = cv2.imread(img_in2, cv2.IMREAD_COLOR);
   
   # make images rectangular
   img_in1 = img_in1[:,:img_in1.shape[0]]
   img_in2 = img_in2[:img_in1.shape[0],:img_in1.shape[0]]

   # Set the total number of pyramid levels
   levels = 5
   
   # Generate Gaussian pyramids for img_in1 and img_in2
   gaussianPyramidA = [img_in1.copy()]
   gaussianPyramidB = [img_in2.copy()]
   for i in range(1, levels):
       gaussianPyramidA.append(cv2.pyrDown(gaussianPyramidA[i - 1]))
       gaussianPyramidB.append(cv2.pyrDown(gaussianPyramidB[i - 1]))

   # Generate the inverse Laplacian Pyramida for input_image1 and input_image2
   laplacianPyramidA = [gaussianPyramidA[levels-1]]
   laplacianPyramidB = [gaussianPyramidB[levels-1]]
   for i in range(levels - 1, 0, -1):
       sizeA = (gaussianPyramidA[i-1].shape[1], gaussianPyramidA[i-1].shape[0])
       sizeB = (gaussianPyramidB[i-1].shape[1], gaussianPyramidB[i-1].shape[0])       
       laplacianA = cv2.subtract(gaussianPyramidA[i - 1], cv2.pyrUp(gaussianPyramidA[i],dstsize=sizeA))
       laplacianB = cv2.subtract(gaussianPyramidB[i - 1], cv2.pyrUp(gaussianPyramidB[i],dstsize=sizeB))
       laplacianPyramidA.append(laplacianA)
       laplacianPyramidB.append(laplacianB)

   # Add the left and right halves of the Laplacian images in each level
   laplacianPyramidComb = []
   for laplacianA, laplacianB in zip(laplacianPyramidA, laplacianPyramidB):
       rows, cols, dpt = laplacianA.shape
       c_rows=int(rows/2)
       c_cols=int(cols/2)
       laplacianComb = np.hstack((laplacianA[:, 0:c_cols], laplacianB[:, c_cols:]))
       laplacianPyramidComb.append(laplacianComb)

   # Reconstruct the image from the Laplacian pyramid
   imgComb = laplacianPyramidComb[0]
   for i in range(1, levels):
       size = (laplacianPyramidComb[i].shape[1], laplacianPyramidComb[i].shape[0])
       imgComb = cv2.add(cv2.pyrUp(imgComb,dstsize=size), laplacianPyramidComb[i])

   

   img_out = imgComb # Blending result
   
   return True, img_out

def Question3():
   
   input_image1=sys.argv[2]
   input_image2=sys.argv[3]
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "5.jpg"
   print(output_name)
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
	  
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
         sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
         help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
         help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
         print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()

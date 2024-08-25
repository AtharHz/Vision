import cv2
import numpy as np
import matplotlib.pyplot as plt

I = cv2.imread("pasargadae.jpg", cv2.IMREAD_GRAYSCALE)
# print(I)
# print(I.shape)

levels = 256

# calculating histogram
def calc_hist(I, levels):
  hist = np.zeros(levels)
  flat=I.ravel()
  # print(flat)
  for i in flat:
    hist[i]+=1
  # ...
  return hist


# calculating CDF
def calc_cdf(hist, levels):
  cdf = np.zeros_like(hist)
  cdf[0]=hist[0]
  for i in range(1,levels):
    cdf[i]+=(hist[i]+cdf[i-1])
  # for j in range(levels):
  #   cdf[j]=cdf[j]
    #cdf[j]=cdf[j]/(625*940)
  return cdf

hist = calc_hist(I, levels)
cdf = calc_cdf(hist, levels)

# normalize CDF
cdf = cdf / cdf[-1]
# print("min    ", np.min(cdf))
# print("max    ", np.max(cdf))
# cdf_norm = ((cdf - np.min(cdf)) * 255) / (np.max(cdf) - np.min(cdf))
# print(cdf)
# cdf_normalized = cdf * float(hist.max()) / cdf.max()

# mapping
# cdf = (cdf - cdf.min())*255/(cdf.max() - cdf.min())
mapping = cdf * 255 / cdf[-1]
mapping = mapping.astype('uint8')
# print(cdf)
# print("cdf min ",cdf.min())
# print(cdf.max())

# print("*********")
# print(cdf)
# cdf = np.ma.filled(cdf,0).astype('uint8')
# cdf=cdf.astype('uint8')
# print("/////////////////")
# print(cdf)

# replace intensity
# equalized_image = hist * 255 / cdf[-1]
# equalized_image = cdf[I]
equalized_image = mapping[I]
# print(equalized_image)

equalized_image_hist = calc_hist(equalized_image, levels)
equalized_image_cdf = calc_cdf(equalized_image_hist, levels)

fig = plt.figure(figsize= (16, 8))
fig.add_subplot(2,3,1)
plt.imshow(I, cmap='gray')
plt.title('pasargadae')
plt.axis('off')

fig.add_subplot(2,3,2)
plt.plot(hist)
plt.title('Source histogram')

fig.add_subplot(2,3,3)
plt.plot(cdf)
plt.title('Source CDF')

fig.add_subplot(2,3,4)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized image')
plt.axis('off')

fig.add_subplot(2,3,5)
plt.plot(equalized_image_hist)
plt.title('Equalized histogram')


fig.add_subplot(2,3,6)
plt.plot(equalized_image_cdf)
plt.title('Equalized CDF')

plt.show()
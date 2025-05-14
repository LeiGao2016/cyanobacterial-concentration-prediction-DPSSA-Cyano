
from PIL import Image
#Drawing a comparison chart of before and after.
# 打开图像
# image1 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/LOCK9趋势浓度.png')
# image2 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/LOCK9趋势流量.png')
# image3 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/LOCK9趋势流速.png')
# image4 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/LOCK9趋势温度.png')
# image5 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/LOCK9趋势盐度.png')
#image6 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/Mannum周期浓度.png')
#image7 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/Mannum周期流量.png')
#image8 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/Mannum周期流速.png')
#image9 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/Mannum周期温度.png')
#image10 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/Mannum周期盐度.png')
image11 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/LOCK9浓度分解前后对比.png')
image12 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/LOCK9流量分解前后对比.png')
image13 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/LOCK9流速分解前后对比.png')
image14 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/LOCK9温度分解前后对比.png')
image15 = Image.open('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/LOCK9盐度分解前后对比.png')
# 获取图像的宽度和高度
# width1, height1 = image1.size
# width2, height2 = image2.size
# width3, height3 = image3.size
# width4, height4 = image4.size
# width5, height5 = image5.size
#width6, height6 = image6.size
#width7, height7 = image7.size
#width8, height8 = image8.size
#width9, height9 = image9.size
#width10, height10 = image10.size
width11, height11 = image11.size
width12, height12 = image12.size
width13, height13 = image13.size
width14, height14 = image14.size
width15, height15 = image15.size
# Create a new image whose width is the sum of the widths of the two images and whose height is the maximum height of the two images.
# merged_image = Image.new('RGB', (width1, height1*5))
#merged_image1 = Image.new('RGB', (width1, height1*5))
merged_image2 = Image.new('RGB', (width11, height11*5))
# 
# merged_image.paste(image1, (0, 0))

# merged_image.paste(image2, (0, height1))
# merged_image.paste(image3, (0, 2*height1))
# merged_image.paste(image4, (0, 3*height1))
# merged_image.paste(image5, (0, 4*height1))
#merged_image1.paste(image6, (0, 0*height1))
#merged_image1.paste(image7, (0, 1*height1))
#merged_image1.paste(image8, (0, 2*height1))
#merged_image1.paste(image9, (0, 3*height1))
#merged_image1.paste(image10, (0,4*height1))
merged_image2.paste(image11, (0, 0))
merged_image2.paste(image12, (0, height11))
merged_image2.paste(image13, (0, 2*height11))
merged_image2.paste(image14, (0, 3*height11))
merged_image2.paste(image15, (0, 4*height11))
# save the merged image to a file
# merged_image.save('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/LOCK9趋势汇总.jpg')
#merged_image1.save('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/Mannum周期汇总.jpg')
merged_image2.save('C:/Users/Lenovo/Desktop/论文资料/图们/新结果/LOCK9前后分解对比汇总.jpg')

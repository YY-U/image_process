import sys
import cv2 as cv
from PIL import Image
import collections, os, math
import numpy as np
#from scipy import signal

runcase = int(sys.argv[1])
#コマンドライン引数はデフォルトstr型なのでint型に変換


if( runcase == 1 ):
    image_name = sys.argv[2]
    #image_name='image0357.png'
    #image_name='iphone.jpg'
    #folder='./processed/test/' 
    #folder='./TrainingDataPath/' 
    #old_folder='./DSLR_image/DSLR_image/'+image_name
    old_folder='./Selfie/GT/5/'+image_name

    #new_folder='./processed/test/'
    #new_folder='./processed/vali_test1/'
    #results_folder='./DSLR_image/results/DSC_0050.jpeg'
    results_folder='./Selfie/LR/5/'

    if not os.path.exists(results_folder):
         os.makedirs(results_folder)#再帰的にディレクトリ作成，深い層まで一気に作成可能
        
    for image_num in range(1,2):
         #folder_nun='scene_'+str(scene_num) + '/'
         #folder_=folder+str(scene_num)+'/'


         #old_folder_=folder+'scene_'+str(image_num)+'/'
         
         #results_folder_=results_folder+'scene_'+str(scene_num+181+20)+'/'


         print('old_folder: ' + old_folder)
         #print('old_folder_: ' + old_folder_)
         print('results_folder: ' + results_folder)
         #print('results_folder_: ' + results_folder_)


         if not os.path.exists(results_folder):
             os.makedirs(results_folder)#再帰的にディレクトリ作成，深い層まで一気に作成可能
         #if not os.path.exists(results_folder_):
         #    os.makedirs(results_folder_)#再帰的にディレクトリ作成，深い層まで一気に作成可能

         image_path=old_folder
         im = cv.imread(image_path,3)
         #im = cv.imread(old_folder_,3)
         re_im=im[:,:,::-1]#BGR→RGB
         print('縮小リサイズ')
         #resized_image = cv.resize(im, (width*2, height*2), interpolation=cv.INTER_LINEAR)
         re_im = cv.resize(re_im, None, fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)
         #print('拡大リサイズ')
         #re_im = cv.resize(re_im, None, fx=4.00, fy=4.00, interpolation=cv.INTER_NEAREST)

         #re_im = cv.rotate(re_im, rotateCode=cv.ROTATE_90_CLOCKWISE)
         """
         cv.rotate(img, 90度単位の回転)#回転
         cv2.ROTATE_90_CLOCKWISE
         cv2.ROTATE_90_COUNTERCLOCKWISE
         cv2.ROTATE_180
         """
         #print('回転')
         #re_im = np.rot90(re_im,3)
         """
         np.rot90(img,1)#90
         np.rot90(img,2)#180
         np.rot90(img,3)#270
         """
         #re_im = cv.rotate(re_im)
         


         re_im=re_im[:,:,::-1]#RGB→BGR
         cv.imwrite(results_folder+image_name, re_im)
         print('cv.imwrite(results_folder+image_name, re_im)')

         #im = cv.cvtColor(im, cv.COLOR_BGR2RGB) #画像の色の順序をBGRからRGBに変換する

         #opencv 補間法#下ほど画質よし
         """
         cv2.INTER_NEAREST 最近傍補間 : 画像を拡大した際に最近傍にある画素をそのまま使う線形補間法
         cv2.INTER_LINEAR バイリニア補間（デフォルト）: 周辺2×2
         cv2.INTER_AREA 平均画素法(面積平均法)（縮小向き）
         cv2.INTER_CUBIC 4×4 の近傍領域を利用するバイキュービック補間：周辺4×4
         cv2.INTER_LANCZOS4 8×8 の近傍領域を利用する Lanczos法の補間：周辺8×8
         """



         #print('im[1]' + str(im[1]) )

         #for i in range(im[1]):


         """
         for image_n in range(0,60):#col_high_0000.png
             image_name='col_high_'+'%04d' % image_n +'.png'
             image_path=os.path.join(old_folder_,image_name)
             #print('image_path : '+str(image_path))
             

             #im_new=im.copy()
             #im_new=im_new[:,:,::-1]#RGB

             #if image_n %rate == 0:
             im = cv.imread(image_path,3)
             im=im[:,:,::-1]#RGB
             


             #image_new_n='col_high_'+'%04d' % image_n +'.png'
             cv.imwrite(new_folder_+image_name,im)
             #cv.imwrite('./result/im_new0001.png',im_new)
         """

elif( runcase == 2 ):
     #image_name='DSC_0050.JPG'
     #image_name='DSC_0050_rot270.JPG'
     #image_name='DSC_0050x0-25_rot270.JPG'
     image_name='DSC_0050x0-25_x4_rot270.JPG'
     #image_name='iphone.jpg'
     #folder='./processed/test/' 
     #folder='./TrainingDataPath/' 
     #old_folder='./DSLR_image/DSLR_image/'+image_name
     old_folder='./DSLR_image/results/LR_HR/'+image_name

     #new_folder='./processed/test/'
     #new_folder='./processed/vali_test1/'
     #results_folder='./DSLR_image/results/DSC_0050.jpeg'
     results_folder='./DSLR_image/results/LR_HR/'
     im = Image.open(old_folder)

     def crop_center(pil_img, crop_width, crop_height):
          img_width, img_height = pil_img.size
          return pil_img.crop( ((img_width - crop_width) // 2,(img_height - crop_height) // 2,(img_width + crop_width) // 2,(img_height + crop_height) // 2) )

     print('中心クロップ')
     im_new = crop_center(im, 300, 300)
     im_new.save(results_folder+'crop_'+image_name, quality=95)
    

elif( runcase == 3 ):#生成画像一括クロップ

     image_name = sys.argv[2]
     image_format = sys.argv[3] # .png , .jpg etc...

     image_number_start=sys.argv[4]
     image_s=int(image_number_start)
     #image_s='%04d' % image_s

     image_number_end=sys.argv[5]
     image_e=int(image_number_end)
     #image_e='%04d' % image_e

     model_mode_1 = 'TecoGAN'
     model_mode_2 = 'TecoGAN-dataset-0.5'
     #model_mode = 'TecoGAN'
     #model_mode = 'TecoGAN-dataset-0.5'
     image_folder = '2'
     #image_name='image0357.png'
     #image_name='iphone.jpg'
     #folder='./processed/test/' 
     #folder='./TrainingDataPath/' 
     #old_folder='./DSLR_image/DSLR_image/'+image_name
     
     for image_i in range(image_s,image_e+1):
         #GT_folder='./Selfie/GT/'+image_folder+'/'+image_name
         #GT_folder='./Selfie/GT/'+image_folder+'/'+image_name + str('%04d' % image_i) + str(image_format)
         #re_folder='./Selfie/results/'+model_mode+'/'+image_folder+'/'+image_folder+'/'+'output_'+image_name
         #re_folder='./Selfie/results/'+model_mode+'/'+image_folder+'/'+image_folder+'/'+'output_'+image_name + str('%04d' % image_i) + str(image_format)


         GT_f = './Selfie/GT/'+image_folder+'/'
         GT_im = GT_f + image_name + str('%04d' % image_i) + str(image_format)

         re_f_1 = './Selfie/results/'+model_mode_1+'/'+image_folder+'/'+image_folder+'/'
         re_im_1 = re_f_1 + 'output_'+image_name + str('%04d' % image_i) + str(image_format)
         #re_folder_1 = './Selfie/results/'+model_mode_1+'/'+image_folder+'/'+image_folder+'/'+'output_'+image_name + str('%04d' % image_i) + str(image_format)

         re_f_2 = './Selfie/results/'+model_mode_2+'/'+image_folder+'/'+image_folder+'/'
         re_im_2 = re_f_2 + 'output_'+image_name + str('%04d' % image_i) + str(image_format)
         #re_folder_2 = './Selfie/results/'+model_mode_2+'/'+image_folder+'/'+image_folder+'/'+'output_'+image_name + str('%04d' % image_i) + str(image_format)


         LR_f = './Selfie/LR/'+image_folder+'/'
         LR_im = LR_f + image_name + str('%04d' % image_i) + str(image_format)

         LRx4_f = './Selfie/LRx4/'+image_folder+'/'


         print('model_mode_1 : '+ str(model_mode_1) )
         print('model_mode_2 : '+ str(model_mode_2) )

         print('GT_im : '+ str(GT_im) )
         print('LR_im : '+ str(LR_im) )

         print('LRx4_f : '+ str(LRx4_f) )

         print('re_im_1 : '+ str(re_im_1) )
         print('re_im_2 : '+ str(re_im_2) )
            
         #new_folder='./processed/test/'
         #new_folder='./processed/vali_test1/'
         #results_folder='./DSLR_image/results/DSC_0050.jpeg'

         #results_folder='./Selfie/sub/GT_'+model_mode+'/'+image_folder+'/'
         #results_folder='./Selfie/sub/GT_'+model_mode+'/'+image_folder+'-1'+'/'

         #print('results_folder : '+str(results_folder) )

         #if not os.path.exists(results_folder):
         #    os.makedirs(results_folder)#再帰的にディレクトリ作成，深い層まで一気に作成可能

         GT = cv.imread(GT_im,3)
         GT=GT[:,:,::-1]#BGR→RGB

         LR = cv.imread(LR_im,3)
         LR=LR[:,:,::-1]#BGR→RGB

         print('LR:拡大リサイズ')
         LRx4 = cv.resize(LR, None, fx=4.00, fy=4.00, interpolation=cv.INTER_NEAREST)

         re_1 = cv.imread(re_im_1,3)
         re_1=re_1[:,:,::-1]#BGR→RGB

         re_2 = cv.imread(re_im_2,3)
         re_2=re_2[:,:,::-1]#BGR→RGB

         print('crop')

         #新しい配列に入力画像の一部を代入
         y_s=350
         y_e=700

         x_s=170
         x_e=508

         GT = GT[y_s:y_e,x_s:x_e]
         re_1 = re_1[y_s:y_e,x_s:x_e]
         re_2 = re_2[y_s:y_e,x_s:x_e]
         LRx4_crop = LRx4[y_s:y_e,x_s:x_e]

         point=str(y_s)+'~'+str(y_e)+'_'+str(x_s)+'~'+str(x_e)+'_'

         #sub = crop_center(sub, 300, 300)
         #im_new.save(results_folder+'crop_'+image_name, quality=95)

         GT = GT[:,:,::-1] #RGB→BGR

         re_1 = re_1[:,:,::-1] #RGB→BGR
         re_2 = re_2[:,:,::-1] #RGB→BGR

         LRx4 = LRx4[:,:,::-1] #RGB→BGR
         LRx4_crop = LRx4_crop[:,:,::-1] #RGB→BGR

         cv.imwrite(GT_f+'crop_'+point+image_name+str('%04d' % image_i)+image_format, GT)
         cv.imwrite(LRx4_f+image_name+str('%04d' % image_i)+image_format, LRx4)
         cv.imwrite(LRx4_f+'crop_'+point+image_name+str('%04d' % image_i)+image_format, LRx4_crop)
         cv.imwrite(re_f_1+'crop_'+point+image_name+str('%04d' % image_i)+image_format, re_1)
         cv.imwrite(re_f_2+'crop_'+point+image_name+str('%04d' % image_i)+image_format, re_2)


elif( runcase == 4 ):

     image_name = sys.argv[2]
     image_format = sys.argv[3]# .png , .jpg etc...

     image_number_start=sys.argv[4]
     image_s=int(image_number_start)
     #image_s='%04d' % image_s

     image_number_end=sys.argv[5]
     image_e=int(image_number_end)
     #image_e='%04d' % image_e

     #model_mode = 'TecoGAN'
     model_mode = 'TecoGAN-dataset-0.5'
     #image_folder = '2'
     #image_folder = '5'
     image_folder = '5-1'
     #image_name='image0357.png'
     #image_name='iphone.jpg'
     #folder='./processed/test/' 
     #folder='./TrainingDataPath/' 
     #old_folder='./DSLR_image/DSLR_image/'+image_name
     
     for image_i in range(image_s,image_e+1):
         #GT_folder='./Selfie/GT/'+image_folder+'/'+image_name
         GT_folder='./Selfie/GT/'+image_folder+'/'+image_name + str('%04d' % image_i) + str(image_format)
         #re_folder='./Selfie/results/'+model_mode+'/'+image_folder+'/'+image_folder+'/'+'output_'+image_name
         re_folder='./Selfie/results/'+model_mode+'/'+image_folder+'/'+image_folder+'/'+'output_'+image_name + str('%04d' % image_i) + str(image_format)

         print('model_mode : '+ str(model_mode) )
         print('GT_folder : '+ str(GT_folder) )
         print('re_folder : '+ str(re_folder) )
            
         #new_folder='./processed/test/'
         #new_folder='./processed/vali_test1/'
         #results_folder='./DSLR_image/results/DSC_0050.jpeg'

         #results_folder='./Selfie/sub/GT_'+model_mode+'/'+image_folder+'/'
         results_folder='./Selfie/sub/GT_'+model_mode+'/'+image_folder+'/'

         print('results_folder : '+str(results_folder) )

         if not os.path.exists(results_folder):
             os.makedirs(results_folder)#再帰的にディレクトリ作成，深い層まで一気に作成可能

         GT_im = cv.imread(GT_folder,3)
         #im = cv.imread(old_folder_,3)
         re_GT_im=GT_im[:,:,::-1]#BGR→RGB

         re_im = cv.imread(re_folder,3)
         #im = cv.imread(old_folder_,3)
         re_re_im=re_im[:,:,::-1]#BGR→RGB

         im_diff = re_GT_im.astype(int) - re_re_im.astype(int)
         #sub = abs(re_GT_im - re_re_im)
         #sub = np.abs(im_diff+128)#灰色#少し見にくいかな

         #0基準
         im_diff_abs = np.abs(im_diff)
         im_diff_abs_norm = im_diff_abs / im_diff_abs.max() * 255 #maxで正規化
         black_sub=im_diff_abs_norm

         #0→128 基準へ
         #im_diff_center = np.floor_divide(im_diff, 2) + 128
         im_diff_center_norm = im_diff / np.abs(im_diff).max() * 127.5 + 127.5 #maxで正規化
         gray_sub=im_diff_center_norm

         if sys.argv[6] == 'crop' :
             #def crop_center(pil_img, crop_width, crop_height):
                #img_width, img_height = pil_img.size
                #return pil_img.crop( ((img_width - crop_width) // 2,(img_height - crop_height) // 2,(img_width + crop_width) // 2,(img_height + crop_height) // 2) )

             print('クロップ')

             #新しい配列に入力画像の一部を代入
             y_s=350
             y_e=700

             x_s=170
             x_e=508

             black_sub = black_sub[y_s:y_e,x_s:x_e]
             gray_sub = gray_sub[y_s:y_e,x_s:x_e]

             point=str(y_s)+'~'+str(y_e)+'_'+str(x_s)+'~'+str(x_e)+'_'

             #sub = crop_center(sub, 300, 300)
             #im_new.save(results_folder+'crop_'+image_name, quality=95)

             black_sub = black_sub[:,:,::-1] #RGB→BGR
             gray_sub = gray_sub[:,:,::-1] #RGB→BGR
             cv.imwrite(results_folder+'crop_'+point+'_black_'+image_name+str('%04d' % image_i)+image_format, black_sub)
             cv.imwrite(results_folder+'crop_'+point+'_gray_'+image_name+str('%04d' % image_i)+image_format, gray_sub)
         elif sys.argv[6] == 'nocrop' :
             black_sub = black_sub[:,:,::-1] #RGB→BGR
             gray_sub=gray_sub[:,:,::-1]#RGB→BGR
             cv.imwrite(results_folder+'color_black_'+image_name+str('%04d' % image_i)+image_format, black_sub)
             cv.imwrite(results_folder+'color_gray_'+image_name+str('%04d' % image_i)+image_format, gray_sub)


elif( runcase == 5 ):#縮小一括リサイズ 拡大一括リサイズ

     image_name = sys.argv[2]
     image_format = sys.argv[3] # .png , .jpg etc...

     image_number_start=sys.argv[4]
     image_s=int(image_number_start)
     #image_s='%04d' % image_s

     image_number_end=sys.argv[5]
     image_e=int(image_number_end)
     #image_e='%04d' % image_e

     resize_method=sys.argv[6] #up or down

     #model_mode = 'TecoGAN'
     #model_mode = 'TecoGAN-dataset-0.5'
     image_folder = '5-1'
     
     for image_i in range(image_s,image_e+1):
         #GT_f = './Selfie/GT/'+image_folder+'/'
         #GT_im = GT_f + image_name + str('%04d' % image_i) + str(image_format)

         target_f = './Selfie/LR/'+image_folder+'/'
         target_im = target_f + image_name + str('%04d' % image_i) + str(image_format)
         
         #LRx0125_f = './Selfie/LRx0125/'+image_folder+'/'
         res_f = './Selfie/LRx4/'+image_folder+'/'
         #GTx05_f = './Selfie/GTx05/'+image_folder+'/'

         if not os.path.exists(res_f):
             os.makedirs(res_f)#再帰的にディレクトリ作成，深い層まで一気に作成可能

         #if not os.path.exists(GTx_f):
             #os.makedirs(GTx_f)#再帰的にディレクトリ作成，深い層まで一気に作成可能


         #print('GT_im : '+ str(GT_im) )
         #print('LRx0125_f : '+ str(LRx0125_f) )
         #print('GTx05_f : '+ str(GTx05_f) )
         print('target_f : '+ str(target_f) )
         print('target_im : '+ str(target_im) )
         print('results_f : '+ str(res_f) )

         target = cv.imread(target_im,3)
         target = target[:,:,::-1]#BGR→RGB
         if resize_method == 'down' :
             print('縮小リサイズ')
             resize_interpolation = cv.INTER_NEAREST
             print(str(resize_interpolation))
             #LR = cv.resize(GT, None, fx=0.125, fy=0.125, interpolation=cv.INTER_NEAREST)
             res = cv.resize(target, None, fx=0.5, fy=0.5, interpolation=resize_interpolation)
         elif resize_method == 'up' :
             print('拡大リサイズ')
             resize_interpolation = cv.INTER_NEAREST
             print(str(resize_interpolation))
             res = cv.resize(target, None, fx=4.0, fy=4.0, interpolation=resize_interpolation)
            

         #LR = LR[:,:,::-1] #RGB→BGR
         #GTx05 = GTx05[:,:,::-1] #RGB→BGR
         res = res[:,:,::-1] #RGB→BGR

         #cv.imwrite(LRx0125_f+image_name+str('%04d' % image_i)+image_format, LR)
         cv.imwrite(res_f+image_name+str('%04d' % image_i)+image_format, res)



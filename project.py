try:
    from tkinter import *
except:
    from Tkinter import *
import tensorflow
from PIL import Image, ImageOps
import numpy as np
import shutil
import os
import glob, os.path
#import time
import pygame, sys
pygame.init()
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from PIL import Image as pil_image
#associated to monochrome
def blackWhite():
    filelist = glob.glob(os.path.join('mono', '*.jpg'))
    filelist.sort()
    for i, imagepath in enumerate(filelist):
        print("    Status: %s / %s" %(i, len(filelist)), end="\r")
        Label (root, text="    Status: %s / %s" %(i+1, len(filelist)),font="none 10").grid(row=6,column=0, sticky=E)
        col =  image.load_img(imagepath, target_size=(224, 224))# open colour image
        gray = col.convert('L')
        bw = gray.point(lambda x: 0 if x<90 else 255, '1')
        bw.save("monochrome/result"+str(i)+".jpg")
#associated with clustering
def cluster():
    image.LOAD_TRUNCATED_IMAGES = True 
    model = VGG16(weights='imagenet', include_top=False)

    # Variables
    imdir = 'monochrome/'
    targetdir = 'outdir/'

    # Loop over files and get features
    filelist = glob.glob(os.path.join(imdir, '*.jpg'))
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
        print("    Status: %s / %s" %(i, len(filelist)), end="\r")
        Label (root, text="    Status: %s / %s" %(i+1, len(filelist)),font="none 10").grid(row=8,column=0, sticky=E)
        img = image.load_img(imagepath, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = np.array(model.predict(img_data))
        featurelist.append(features.flatten())
    #m=featurelist[0]-featurelist[1]
    #print(m)
    # Clustering
    kmeans = AgglomerativeClustering(n_clusters=None, affinity='euclidean', memory=None, connectivity=None, compute_full_tree=True, linkage='ward', distance_threshold=10).fit(np.array(featurelist))

    # Copy images renamed by cluster  
    # Check if target dir exists
    try:
        os.makedirs(targetdir)
    except OSError:
        pass
    # Copy with cluster name
    print("\n")
    for i, m in enumerate(kmeans.labels_):
        try:
            os.makedirs(targetdir+'cluster'+str(m))
        except OSError:
            pass
        print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
        shutil.copy(filelist[i], targetdir+'cluster'+str(m)+'/'+ str(m) + "_" + str(i) +".jpg")
    #print m no of tigers
    Label (root, text="  Verify Outdir:Clusters  "+str(len(next(os.walk('outdir'))[1])-1),font="none 10").grid(row=9,column=3, sticky=W)
   

    
    
#associated with cropping
def displayImage(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    width =  pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    # draw transparent box and blit it onto canvas
    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)
def setup(path):
    px = pygame.image.load(path)
    screen = pygame.display.set_mode( px.get_rect()[2:] )
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px

def mainLoop(screen, px):
    topleft = bottomright = prior = None
    n=0
    while n!=1:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                n=1
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    n=1
        if topleft:
            prior = displayImage(screen, px, topleft, prior)
    return ( topleft + bottomright )


#associated with browse button
def browse_button():
    filename = filedialog.askdirectory()
    out.insert(END,filename)
    print(filename)
    detection(filename)
    return filename
#detection
# Disable scientific notation for clarity
def detection(path):
    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    filelist = glob.glob(os.path.join(path, '*.jpg'))
    filelist.sort()
    for i, imagepath in enumerate(filelist):
        #print("    Status: %s / %s" %(i, len(filelist)), end="\r")
        Label (root, text="    Status: %s / %s" %(i+1, len(filelist)),font="none 10").grid(row=2,column=0, sticky=E)
        image = Image.open(imagepath)

        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # display the resized image
        #image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        tiger=np.matrix(prediction)
        #print(tiger)
        td=tiger.item((0,0))
        ntd=tiger.item((0,1))
        #print(td)
        #print(ntd)
        #print("Tiger        Non-Tiger")
        #print(prediction)
        if(td>ntd):
            #tiger prediction and moving
            print("Tiger Detected & moved")
            src=path+'/test_photo'+str(i)+'.jpg'
            f=next(os.walk('tiger_data'))[2]
            count=len(f)
            dest='tiger_data/test_photo'+str(count+1)+'.jpg'
            shutil.copyfile(src,dest)
        else: 
            #non-tiger
            print("Non Tiger Species Detected")

#cropping
def manualCrop():
    path='tiger_data'
    filelist = glob.glob(os.path.join(path, '*.jpg'))
    filelist.sort()
    count=0
    for i, imagepath in enumerate(filelist):
        count+=1
        #print("    Status: %s / %s" %(i, len(filelist)), end="\r")
        Label (root, text="    Status: %s / %s" %(i+1, len(filelist)),font="none 10").grid(row=4,column=0, sticky=E)
        output_loc = 'mono/test_photo'+str(count+1)+'.jpg'
        screen, px = setup(imagepath)
        try:
            left, upper, right, lower = mainLoop(screen, px)

            # ensure output rect always has positive width, height
            if right < left:
                left, right = right, left
            if lower < upper:
                lower, upper = upper, lower
            im = Image.open(imagepath)
            im = im.crop(( left, upper, right, lower))
            pygame.display.quit()
            im.save(output_loc)
        except:
            continue
root = Tk()
root.geometry("540x300")
root.title("Project Tiger")
v = StringVar()
photo1=PhotoImage(file="logo.gif")
Label (root, image=photo1).grid(row=0,column=3)
Label (root, text="  Select Folder : ",font="none 10").grid(row=1,column=0,sticky=W)
#initialisation
try:
    os.mkdir('tiger_data')
except:
    files=glob.glob('tiger_data/*')
    for f in files:
        os.remove(f)
    
    
try:
    os.mkdir('mono')
except:
    files=glob.glob('mono/*')
    for f in files:
        os.remove(f)
    
    
try:
    os.mkdir('monochrome')
except:
    files=glob.glob('monochrome/*')
    for f in files:
        os.remove(f)
    
try:
    os.mkdir('outdir')
except:
    folder='outdir'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

out=Text(root,width=40,height=1)
out.grid(row=1,column=3,sticky=E)
#Browse button
button2 = Button(text=" Browse", command=browse_button).grid(row=1, column=13, sticky=E)
Label (root, text="  Launch Tool :   ",font="none 10").grid(row=3,column=0, sticky=W)
Label (root, text="            Select the Falnk Region  ",font="none 10").grid(row=3,column=3, sticky=W)
button3 = Button(text=" Launch", command=manualCrop).grid(row=3, column=13, sticky=E)
Label (root, text="  To Monochrome :   ",font="none 10").grid(row=5,column=0, sticky=W)
Label (root, text="            Recommended for better accuracy  ",font="none 10").grid(row=5,column=3, sticky=W)
button4 = Button(text=" Convert", command=blackWhite).grid(row=5, column=13, sticky=E)
button4 = Button(text="       Identify       ", command=cluster).grid(row=7, column=3)

mainloop()
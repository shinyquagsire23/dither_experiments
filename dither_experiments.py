# pip3 install numpy python-opencv pillow

import numpy as np
import cv2
import random

# image path
path = 'input.png'
image = cv2.imread(path)

window_name = 'image'

TARGET_BITS = 4
TARGET_CHANNELS = 3

def clamp(val,a,b):
    if val < a:
        return a
    if val > b:
        return b
    return val

def get_new_val(old_val, nc):
    """
    Get the "closest" colour to old_val in the range [0,1] per channel divided
    into nc values.

    """

    return np.round(old_val * (nc - 1)) / (nc - 1)

def fs_dither(img, nc):
    """
    Floyd-Steinberg dither the image img into a palette with nc colours per
    channel.

    """

    arr = np.array(img, dtype=float) / 255

    new_height = img.shape[0]
    new_width = img.shape[1]

    for ir in range(new_height):
        for ic in range(new_width):
            # NB need to copy here for RGB arrays otherwise err will be (0,0,0)!
            old_val = arr[ir, ic].copy()
            new_val = get_new_val(old_val, nc)
            arr[ir, ic] = new_val
            err = old_val - new_val
            # In this simple example, we will just ignore the border pixels.
            if ic < new_width - 1:
                arr[ir, ic+1] += err * 7/16
            if ir < new_height - 1:
                if ic > 0:
                    arr[ir+1, ic-1] += err * 3/16
                arr[ir+1, ic] += err * 5/16
                if ic < new_width - 1:
                    arr[ir+1, ic+1] += err / 16

    carr = np.array(arr/np.max(arr, axis=(0,1)) * 255, dtype=np.uint8)
    return carr

def alter_pixel_dsarb_nodither(c, bits):
    """
    Downsample to `bits` bits per pixel without any dithering
    """
    r = (int(c[0]))
    g = (int(c[1]))
    b = (int(c[2]))

    ds_shift = np.array([8,8,8]) - bits

    r_ds = (int(r) >> ds_shift[0])
    g_ds = (int(g) >> ds_shift[1])
    b_ds = (int(b) >> ds_shift[2])
    r_ds = (r_ds << ds_shift[0]) + r_ds
    g_ds = (g_ds << ds_shift[1]) + g_ds
    b_ds = (b_ds << ds_shift[2]) + b_ds

    c_new = [r_ds, g_ds, b_ds]

    return c_new

def alter_pixel_dsarb(c, error, bits):
    """
    Downsample to `bits` bits per pixel with 'scanline' dithering
    """

    rgb = c + error
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    # Get the magnitude of the color that is outside of the range [0, 255]
    # We cascade this value into the next error
    over_r = -r if r < 0 else (r - 255) if r > 255 else 0
    over_g = -g if g < 0 else (g - 255) if g > 255 else 0
    over_b = -b if b < 0 else (b - 255) if b > 255 else 0

    # Clamp the color to [0, 255] now that we have the over/under magnitude
    r = clamp(r, 0, 255)
    g = clamp(g, 0, 255)
    b = clamp(b, 0, 255)

    ds_shift = np.array([8,8,8]) - bits
    ds_mult = np.array([255,255,255]) // ((1<<(bits))-1)

    r_ds = (int(r) >> ds_shift[0]) * ds_mult[0]
    g_ds = (int(g) >> ds_shift[1]) * ds_mult[1]
    b_ds = (int(b) >> ds_shift[2]) * ds_mult[2]

    # Cascade both the downsampling error (r - r_ds) and the clipping error (over_r)
    error[0] = r - r_ds + over_r
    error[1] = g - g_ds + over_g
    error[2] = b - b_ds + over_b

    # Make sure the error doesn't blow up endlessly
    error = np.clip(error, -255, 255)

    c_new = [r_ds, g_ds, b_ds]

    return c_new, error


def alter_image(image, fs):
    width = image.shape[1]
    height = image.shape[0]
    image_out = np.zeros((image.shape[0]*2,image.shape[1]*2,image.shape[2]), dtype=np.uint8)
    bits = np.array([TARGET_BITS,TARGET_BITS,TARGET_BITS])
    error_bits = np.array([TARGET_BITS,TARGET_BITS,TARGET_BITS]) # The lower the BPP, the higher this should be probably

    for y in range(0, image.shape[0]):
        #error = [random.randrange(0, (1<<bits[0])-1, 1),random.randrange(0, (1<<bits[1])-1, 1),random.randrange(0, (1<<bits[2])-1, 1)]
        #error = [random.randrange(-127, 127, 1),random.randrange(-127, 127, 1),random.randrange(-127, 127, 1)]

        #error = [random.randrange(-((1<<(error_bits[0]+1))-1), (1<<(error_bits[0]+1))-1, 1),random.randrange(-((1<<(error_bits[1]+1))-1), (1<<(error_bits[1]+1))-1, 1),random.randrange(-((1<<(error_bits[2]+1))-1), (1<<(error_bits[2]+1))-1, 1)]
        error = [0,0,0]
        if TARGET_CHANNELS == 1:
            error[1] = error[0]
            error[2] = error[0]

        last_c = image[y,0,:]

        for x in range(0, image.shape[1]):



            #if error[0] == 0:
            #    error = [random.randrange(-((1<<(error_bits[0]+1))-1), (1<<(error_bits[0]+1))-1, 1),random.randrange(-((1<<(error_bits[1]+1))-1), (1<<(error_bits[1]+1))-1, 1),random.randrange(-((1<<(error_bits[2]+1))-1), (1<<(error_bits[2]+1))-1, 1)]

            c = image[y,x,:]

            diff = last_c - c
            if diff[0]+diff[1]+diff[2] > 128:
                error = [random.randrange(-((1<<(error_bits[0]+1))-1), (1<<(error_bits[0]+1))-1, 1),random.randrange(-((1<<(error_bits[1]+1))-1), (1<<(error_bits[1]+1))-1, 1),random.randrange(-((1<<(error_bits[2]+1))-1), (1<<(error_bits[2]+1))-1, 1)]
                if TARGET_CHANNELS == 1:
                    error[1] = error[0]
                    error[2] = error[0]

            c_new_nodither = alter_pixel_dsarb_nodither(c, bits)
            c_new, error = alter_pixel_dsarb(c, error, bits)

            image_out[y,x,:] = c
            image_out[y,x+(width*1),:] = c_new_nodither
            image_out[y+height,x+(width*0),:] = fs[y,x,:]
            image_out[y+height,x+(width*1),:] = c_new
    return image_out


images = []
image_fs = fs_dither(image, 1<<TARGET_BITS)

for i in range(0, 60):
    images += [alter_image(image,image_fs)]
    print (i)

def loop():
    for i in range(0, len(images)):
        image_out = images[i]

        def text_help(a,b,c):
            a = cv2.putText(a, b, c, cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,0], 2, cv2.LINE_AA)
            return cv2.putText(a, b, c, cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,255,255], 1, cv2.LINE_AA)

        image_out = text_help(image_out, str(8*TARGET_CHANNELS) + 'bpp', (20,image_out.shape[0]//2-40))
        image_out = text_help(image_out, str(TARGET_BITS*TARGET_CHANNELS) + 'bpp no dither', (20+image.shape[1],image_out.shape[0]//2-40))
        image_out = text_help(image_out, str(TARGET_BITS*TARGET_CHANNELS) + 'bpp Floyd-Steinberg', (20+image.shape[1]*0,image_out.shape[0]-40))
        image_out = text_help(image_out, str(TARGET_BITS*TARGET_CHANNELS) + 'bpp scanline dither', (20+image.shape[1]*1,image_out.shape[0]-40))


        cv2.imshow(window_name, image_out)

        cv2.waitKey(1) # 1ms wait
        print ("frame")


while True:
    loop()

cv2.destroyAllWindows()
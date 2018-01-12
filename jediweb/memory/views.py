from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import numpy as np
# Create your views here.

_images = ['033b7c459edd347e83e7f6d7dec1dfa1_resize.jpg',
          '0bfcdf84e1cea32bf72a8cebd2dd3ecb_resize.jpg',
          '1eac0616da1d7b1fdaef4603694b8e30_resize.jpg',
          '272529824_ff1ee57e46_resize.jpg',
          '353f62a12c2d273e28abfa26da28385e_resize.jpg',
          '361775749_1e1f2f55bb_resize.jpg',
          '36da7c2c488add749d4eb7692efebf3f_resize.jpg',
          '382f00de93ed4565983f5f2f969fbcb7_resize.jpg',
          '40e8e92273d6e018d336568d754d28e2_resize.jpg',
          '47fa2a8868f4876fb6a1fa97f7a6a29b_resize.jpg',
          '5ac631bff1d4fe013e747c8b1f9a25f9_resize.jpg',
          '66f7cbeee2f762267d9e50c449bdc9c2_resize.jpg',
          '77d504505cc5ce7c620d2e5731ea3337_resize.jpg',
          '7d2fd4b6f3eced8f94085d28e960273a_resize.jpg',
          '7f80ceddd1f95f322b86e429fb7f4d69_resize.jpg',
          '84367a63f8dcf536981ea28312c4b3a6_resize.jpg',
          '8b0c389b3060f8dd4880f0c6eae9d20c_resize.jpg',
          '8bc14d78ac1221800779341a4ab487dc_resize.jpg',
          '938413b23f23ccd45f4bb774707fea8d_resize.jpg',
          'b4ba6282627a0ce39ac33d1b802f4c3c_resize.jpg',
          'liangyue_dc (30)_resize.jpg',
          'liangyue_dc (3)_resize.jpg',
          'liangyue_dc (4)_resize.jpg',
          'liangyue_wc (1)_resize.jpg']

def home(request):
  return render(request,'memory/index.html')

def images(request):
  return render(request,'memory/images.html')



def images_test(request):
  return render(request,'memory/images_test.html')

def get_images(request, n_images=6):

  data = {}

  imgs = list(range(len(_images)))
  np.random.shuffle(imgs)
  imgs = imgs[:n_images]

  # Save the images to session.
  request.session['ORDER_IMGS'] = imgs
  print(imgs)

  data['images'] = [_images[i] for i in imgs]

  return JsonResponse(data)



def get_images_test(request):
    imgs = request.session['ORDER_IMGS']
    data = {}
    data['images'] = [_images[i] for i in imgs]
    return JsonResponse(data)





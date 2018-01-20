import numpy as np

def get_next(request):

  image_ids = request.session['image_ids']
  algorithm = request.session['algorithm']

  print(image_ids)
  id =0

  if algorithm == 'eer':
    pass

  elif algorithm == 'imt':
    pass

  elif algorithm == 'jedi':
    pass
  else:
    id = image_ids[int(np.random.random_integers(0, len(image_ids), 1)[0])]


  return id
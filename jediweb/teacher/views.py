from django.shortcuts import render

# Create your views here.
###################################################################################################################
# Date: June 2015
# Author: Edward Johns (e.johns@imperial.ac.uk)
# This code may be freely distributed, but citations should be made to:
# E. Johns et al, "Becoming the Expert - Interactive Multi-Class Machine Teaching", in Proceedings of CVPR 2015
###################################################################################################################



# Import some Django modules
from django.shortcuts import render
from django.shortcuts import redirect

# Import some standard Python modules
import os
import numpy
import random

# Import our eer module (Expected Error Reduction)
from teacher import eer

# Import the User and UserResponse models (which are stored in the SQL database)
from teacher.models import User, UserResponse


# Set the name of the dataset
dataset_name = 'nhm-butterfly'
# Define the teaching and testing lengths
num_teaching_images = 20
num_testing_images = 20



# Load the class names
class_names = list(numpy.load(os.path.join('../Datasets/' + dataset_name + '/class_names.npy')))
num_classes = len(class_names)
# Load the number of samples (images) per class
class_num_samples = list(numpy.load(os.path.join('../Datasets/' + dataset_name + '/class_num_images.npy')))
class_num_samples = [int(i) for i in class_num_samples]
# Create a list of indices where each class starts
class_start = [sum(class_num_samples[:n]) for n in range(len(class_num_samples))]
# Create the class ids of all samples
sample_classes = [i for (i,k) in enumerate(class_num_samples) for j in range(k)]
# Load the image paths
image_paths = list(numpy.load(os.path.join('../Datasets/' + dataset_name + '/image_paths.npy')))



def index(request):

    # Has a mode been assigned?
    if 'mode' in request.POST:
        mode = int(request.POST['mode'])
        if mode == 0: # last mode was new user
            request.session.flush()
            createNewUser(request)
            request.session['teaching_image_num'] = 0
            return teaching(request)
        elif mode == 1: # last mode was teaching
            processTeachingAnswer(request)
            return feedback(request)
        elif mode == 2: # last mode was feedback
            teaching_image_num_ = int(request.session['teaching_image_num'])
            if teaching_image_num_ == num_teaching_images:
                request.session['testing_image_num'] = 1
                context = {'num_testing_images': num_testing_images}
                return render(request, 'teacher/endTeaching.html', context)
            else:
                return teaching(request)
        elif mode == 3: # last mode was endTeaching
            request.session['testing_image_num'] = 0
            return testing(request)
        elif mode == 4: # last mode was testing
            processTestingAnswer(request)
            testing_image_num_ = int(request.session['testing_image_num'])
            if testing_image_num_ == num_testing_images:
                return testResults(request)
            else:
                return testing(request)
    else: # No mode, therefore the user has just visited the website
        return render(request, 'teacher/newuser.html')


def createNewUser(request):

    # Create new user
    num_users = User.objects.count()
    user_id = num_users
    new_user = User.create(user_id)
    new_user.save()

    # Create X as an empty belief state (X is the machine's model of the student's distribution)
    X = numpy.zeros([len(sample_classes), num_classes])
    # Save X
    X_path = os.path.join('../User-Data/X_' + str(user_id) + '.npy')
    numpy.save(X_path, X)
    # Set L as an unlabelled set
    L = []
    request.session['L'] = L

    # Make the set of random testing samples (different for each user)
    testing_samples = []
    class_num_testing_samples = int(max(1, num_testing_images / num_classes))
    for i in range(num_classes):
        for j in range(class_num_testing_samples):
            while True:
                sample = random.randint(class_start[i], class_start[i] + class_num_samples[i] - 1)
                if sample not in testing_samples:
                    testing_samples.append(sample)
                    break
    random.shuffle(testing_samples)

    # Set up the session
    request.session['user_id'] = user_id
    request.session['testing_samples'] = testing_samples


def teaching(request):

    user_id_ = request.session['user_id']
    teaching_image_num_ = request.session['teaching_image_num']
    testing_samples_ = request.session['testing_samples']

    teaching_image_num = teaching_image_num_ + 1

    W = numpy.load(os.path.join('../Datasets/' + dataset_name + '/weight_matrix.npy'))
    Y = numpy.load(os.path.join('../Datasets/' + dataset_name + '/ground_truth.npy'))
    X_path = '../User-Data/X_' + str(user_id_) + '.npy'
    X = numpy.load(X_path)
    L = request.session['L']
    next_sample = int(eer.get_next_sample(X, Y, W, L, testing_samples_))
    image_path = image_paths[next_sample]
    teaching_class_id = sample_classes[next_sample]

    context = {'teaching_image_num': teaching_image_num, 'num_teaching_images': num_teaching_images, 'image_path': image_path, 'class_names': class_names}

    request.session['teaching_class_id'] = teaching_class_id
    request.session['teaching_image_id'] = next_sample
    request.session['teaching_image_num'] = teaching_image_num
    request.session['image_path'] = image_path

    return render(request, 'teacher/teaching.html', context)


def feedback(request):

    answer_ = int(request.POST['answer'])
    teaching_image_num_ = int(request.session['teaching_image_num'])
    teaching_class_id_ = int(request.session['teaching_class_id'])
    image_path_ = request.session['image_path']

    true_class_name = class_names[teaching_class_id_]
    answer_class_name = class_names[answer_]
    if answer_ == teaching_class_id_:
        is_correct = True
    else:
        is_correct = False

    context = {'class_names': class_names, 'teaching_image_num': teaching_image_num_, 'image_path': image_path_, 'true_class_name': true_class_name, 'answer_class_name': answer_class_name, 'is_correct': is_correct}

    return render(request, 'teacher/feedback.html', context)


def testing(request):

    testing_image_num_ = request.session['testing_image_num']
    testing_samples_ = request.session['testing_samples']

    testing_image_num = testing_image_num_ + 1

    testing_image_id = testing_samples_[testing_image_num - 1]
    image_path = image_paths[testing_image_id]
    testing_class_id = sample_classes[testing_image_id]

    request.session['testing_image_num'] = testing_image_num
    request.session['testing_image_id'] = testing_image_id
    request.session['testing_class_id'] = testing_class_id

    context = {'testing_image_num': testing_image_num, 'num_testing_images': num_testing_images, 'image_path': image_path, 'class_names': class_names}

    return render(request, 'teacher/testing.html', context)


def processTeachingAnswer(request):

    user_id_ = int(request.session['user_id'])
    teaching_image_id_ = int(request.session['teaching_image_id'])
    answer_ = int(request.POST['answer'])

    L_ = request.session['L']
    L_.append(teaching_image_id_)
    X_path = '../User-Data/X_' + str(user_id_) + '.npy'
    X_ = numpy.load(X_path)
    X_[teaching_image_id_][:] = 0
    X_[teaching_image_id_][answer_] = 1 # Always update X with the class answered by the student (rather than necessarily the ground truth)
    numpy.save(X_path, X_)
    request.session['L'] = L_


def processTestingAnswer(request):

    user_id_ = int(request.session['user_id'])
    testing_class_id_ = int(request.session['testing_class_id'])
    answer_ = int(request.POST['answer'])

    is_correct = True if testing_class_id_ == answer_ else False

    user_response = UserResponse.create(user_id_, is_correct)
    user_response.save()


def testResults(request):

    # Get the average score
    score_sum = 0
    finished_users = User.objects.filter(is_finished = True)
    for u in finished_users:
        finished_correct_responses = UserResponse.objects.filter(user_id = u.user_id).filter(is_correct = True)
        score_sum += len(finished_correct_responses)

    user_id_ = request.session['user_id']

    correct_responses = UserResponse.objects.filter(user_id = user_id_).filter(is_correct = True)
    score = len(correct_responses)

    user = User.objects.get(user_id = user_id_)
    user.score = score
    user.is_finished = True
    user.save()

    if len(finished_users) > 0:
        ave_score = float(score_sum) / len(finished_users)
    else:
        ave_score = score

    context = {'score': score, 'num_testing_images': num_testing_images, 'ave_score': ave_score}

    return render(request, 'teacher/testresults.html', context)

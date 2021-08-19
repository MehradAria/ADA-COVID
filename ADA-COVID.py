SEED = 27
import os
import io
import sys
import argparse
import random
import numpy as np
# from tensorflow import set_random_seed
import tensorflow as tf

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
# set_random_seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from sklearn.metrics import accuracy_score
import model
import optimizer

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

def pairwise_distance(feature, squared=False):
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                    array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

def masked_maximum(data, mask, dim=1):
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums

def triplet_loss_adapted_from_tf(y_true, y_pred):
    del y_true
    margin = 1.
    labels = y_pred[:, :1]

 
    labels = tf.cast(labels, dtype='int32')

    embeddings = y_pred[:, 1:]
    
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    # lshape=array_ops.shape(labels)
    # assert lshape.shape == 1
    # labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    # global batch_size  
    batch_size = array_ops.size(labels) # was 'array_ops.size(labels)'

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    semi_hard_triplet_loss_distance = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')
    
    return semi_hard_triplet_loss_distance

embedding_size = 64
step = 10
input_image_shape = (224, 224, 3)

def pil_loader(path):
    # Return the RGB variant of input image
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def one_hot_encoding(param):
    # Read the source and target labels from param
    s_label = param["source_label"]
    t_label = param["target_label"]

    # Encode the labels into one-hot format
    classes = (np.concatenate((s_label, t_label), axis = 0))
    num_classes = np.max(classes)
    if 0 in classes:
            num_classes = num_classes+1
    s_label = to_categorical(s_label, num_classes = num_classes)
    t_label = to_categorical(t_label, num_classes = num_classes)
    return s_label, t_label
            
def data_loader(filepath, inp_dims):
    # Load images and corresponding labels from the text file, stack them in numpy arrays and return
    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit() 
    img = []
    label = []
    with open(filepath) as fp:
        for line in fp:
            token = line.split()
            i = pil_loader(token[0])
            i = i.resize((inp_dims[0], inp_dims[1]), Image.ANTIALIAS)
            img.append(np.array(i))
            label.append(int(token[1]))
    img = np.array(img)
    label = np.array(label)
    return img, label

def batch_generator(data, batch_size):
    #Generate batches of data.
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size = batch_size, replace = False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr

def train(param):
    best_target_accuracy = 0.0
    net_name = param["network_name"]
    models = {}
    inp = Input(shape = (param["inp_dims"]))
    embedding = model.build_embedding(param, inp)
    classifier = model.build_classifier(param, embedding)
    discriminator = model.build_discriminator(param, embedding)

    if param["number_of_gpus"] > 1:
        models["combined_classifier"] = multi_gpu_model(model.build_combined_classifier(inp, classifier), gpus = param["number_of_gpus"])
        models["combined_discriminator"] = multi_gpu_model(model.build_combined_discriminator(inp, discriminator), gpus = param["number_of_gpus"])
        models["combined_model"] = multi_gpu_model(model.build_combined_model(inp, [classifier, discriminator]), gpus = param["number_of_gpus"])
    else:
        models["combined_classifier"] = model.build_combined_classifier(inp, classifier)
        models["combined_discriminator"] = model.build_combined_discriminator(inp, discriminator)
        models["combined_model"] = model.build_combined_model(inp, [classifier, discriminator])

    models["combined_classifier"].compile(optimizer = optimizer.opt_classifier(param), loss = triplet_loss_adapted_from_tf, metrics = ['accuracy'])
    models["combined_discriminator"].compile(optimizer = optimizer.opt_discriminator(param), loss = 'binary_crossentropy', metrics = ['accuracy'])
    models["combined_model"].compile(optimizer = optimizer.opt_combined(param), loss = {'class_act_last': 'categorical_crossentropy', 'dis_act_last': \
        'binary_crossentropy'}, loss_weights = {'class_act_last': param["class_loss_weight"], 'dis_act_last': param["dis_loss_weight"]}, metrics = ['accuracy'])

    Xs, ys = param["source_data"], param["source_label"]
    Xt, yt = param["target_data"], param["target_label"]

    # Source domain is represented by label 0 and Target by 1
    ys_adv = np.array(([0.] * ys.shape[0]))
    yt_adv = np.array(([1.] * yt.shape[0]))

    y_advb_1 = np.array(([1] * param["batch_size"] + [0] * param["batch_size"])) # For gradient reversal
    y_advb_2 = np.array(([0] * param["batch_size"] + [1] * param["batch_size"]))
    weight_class = np.array(([1] * param["batch_size"] + [0] * param["batch_size"]))
    weight_adv = np.ones((param["batch_size"] * 2,))
    S_batches = batch_generator([Xs, ys], param["batch_size"])
    T_batches = batch_generator([Xt, np.zeros(shape = (len(Xt),))], param["batch_size"])

    param["target_accuracy"] = 0

    optim = {}
    optim["iter"] = 0
    optim["acc"] = ""
    optim["labels"] = np.array(Xt.shape[0],)
    gap_last_snap = 0

    for i in range(param["num_iterations"]):        
        Xsb, ysb = next(S_batches)
        Xtb, ytb = next(T_batches)
        X_adv = np.concatenate([Xsb, Xtb])
        y_class = np.concatenate([ysb, np.zeros_like(ysb)])

        adv_weights = []
        for layer in models["combined_model"].layers:
            if (layer.name.startswith("dis_")):
                adv_weights.append(layer.get_weights())
          
        stats1 = models["combined_model"].train_on_batch(X_adv, [y_class, y_advb_1],\
                                sample_weight=[weight_class, weight_adv])            
        k = 0
        for layer in models["combined_model"].layers:
            if (layer.name.startswith("dis_")):                    
                layer.set_weights(adv_weights[k])
                k += 1

        class_weights = []        
        for layer in models["combined_model"].layers:
            if (not layer.name.startswith("dis_")):
                class_weights.append(layer.get_weights())  

        stats2 = models["combined_discriminator"].train_on_batch(X_adv, [y_advb_2])

        k = 0
        for layer in models["combined_model"].layers:
            if (not layer.name.startswith("dis_")):
                layer.set_weights(class_weights[k])
                k += 1

        if (((i + 1) % param["test_interval"] == 0) and (i > 19000)):
            ys_pred = models["combined_classifier"].predict(Xs)
            yt_pred = models["combined_classifier"].predict(Xt)
            ys_adv_pred = models["combined_discriminator"].predict(Xs)
            yt_adv_pred = models["combined_discriminator"].predict(Xt)

            source_accuracy = accuracy_score(ys.argmax(1), ys_pred.argmax(1))              
            target_accuracy = accuracy_score(yt.argmax(1), yt_pred.argmax(1))
            source_domain_accuracy = accuracy_score(ys_adv, np.round(ys_adv_pred))              
            target_domain_accuracy = accuracy_score(yt_adv, np.round(yt_adv_pred))

            log_str = "iter: {:05d}: \nLABEL CLASSIFICATION: source_accuracy: {:.5f}, target_accuracy: {:.5f}\
                    \nDOMAIN DISCRIMINATION: source_domain_accuracy: {:.5f}, target_domain_accuracy: {:.5f} \n"\
                                                         .format(i, source_accuracy*100, target_accuracy*100,
                                                      source_domain_accuracy*100, target_domain_accuracy*100)
            print(log_str)

            if param["target_accuracy"] < target_accuracy:              
                optim["iter"] = i
                optim["acc"] = log_str
                optim["labels"] = ys_pred.argmax(1)

                if (gap_last_snap >= param["snapshot_interval"]):
                    gap_last_snap = 0
                    with open(f"Log_{net_name}.txt", "a+") as My_Log:
                        My_Log.write(optim["acc"])
                    # if target_accuracy >= best_target_accuracy:
                    if target_accuracy > best_target_accuracy:
                        models["combined_classifier"].save(f"Best_Model_{net_name}.h5")
                        print('Target Accuracy Improved, Model Saved.')
                        best_target_accuracy = target_accuracy
                        with open(f"Best-ACC_{net_name}.txt", "a+") as Best_ACC:
                            Best_ACC.write(optim["acc"])
        gap_last_snap = gap_last_snap + 1;

if __name__ == "__main__":
    # Read parameter values from the console
    parser = argparse.ArgumentParser(description = 'Domain Adaptation')
    parser.add_argument('--number_of_gpus', type = int, nargs = '?', default = '1', help = "Number of gpus to run")
    parser.add_argument('--network_name', type = str, default = 'ResNet50', help = "Name of the feature extractor network")
    parser.add_argument('--dataset_name', type = str, default = 'COVID', help = "Name of the source dataset")
    parser.add_argument('--dropout_classifier', type = float, default = 0.25, help = "Dropout ratio for classifier")
    parser.add_argument('--dropout_discriminator', type = float, default = 0.25, help = "Dropout ratio for discriminator")    
    parser.add_argument('--source_path', type = str, default = 'Source.txt', help = "Path to source dataset")
    parser.add_argument('--target_path', type = str, default = 'Target.txt', help = "Path to target dataset")
    parser.add_argument('--lr_classifier', type = float, default = 0.0001, help = "Learning rate for classifier model")
    parser.add_argument('--b1_classifier', type = float, default = 0.9, help = "Exponential decay rate of first moment \
                                                                                             for classifier model optimizer")
    parser.add_argument('--b2_classifier', type = float, default = 0.999, help = "Exponential decay rate of second moment \
                                                                                            for classifier model optimizer")
    parser.add_argument('--lr_discriminator', type = float, default = 0.00001, help = "Learning rate for discriminator model")
    parser.add_argument('--b1_discriminator', type = float, default = 0.9, help = "Exponential decay rate of first moment \
                                                                                             for discriminator model optimizer")
    parser.add_argument('--b2_discriminator', type = float, default = 0.999, help = "Exponential decay rate of second moment \
                                                                                            for discriminator model optimizer")
    parser.add_argument('--lr_combined', type = float, default = 0.00001, help = "Learning rate for combined model")
    parser.add_argument('--b1_combined', type = float, default = 0.9, help = "Exponential decay rate of first moment \
                                                                                             for combined model optimizer")
    parser.add_argument('--b2_combined', type = float, default = 0.999, help = "Exponential decay rate of second moment \
                                                                                            for combined model optimizer")
    parser.add_argument('--classifier_loss_weight', type = float, default = 4, help = "Classifier loss weight")
    parser.add_argument('--discriminator_loss_weight', type = float, default = 1, help = "Discriminator loss weight")
    parser.add_argument('--batch_size', type = int, default = 32, help = "Batch size for training")
    parser.add_argument('--test_interval', type = int, default = 30, help = "Gap between two successive test phases")
    parser.add_argument('--num_iterations', type = int, default = 12000, help = "Number of iterations")
    parser.add_argument('--snapshot_interval', type = int, default = 30, help = "Minimum gap between saving outputs")
    parser.add_argument('--output_dir', type = str, default = 'Models', help = "Directory for saving outputs")
    args = parser.parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(list(np.arange(args.number_of_gpus))).strip('[]')

    # Initialize parameters
    param = {}
    param["number_of_gpus"] = args.number_of_gpus
    param["network_name"] = args.network_name
    param["inp_dims"] = [224, 224, 3]
    param["num_iterations"] = args.num_iterations
    param["lr_classifier"] = args.lr_classifier
    param["b1_classifier"] = args.b1_classifier
    param["b2_classifier"] = args.b2_classifier    
    param["lr_discriminator"] = args.lr_discriminator
    param["b1_discriminator"] = args.b1_discriminator
    param["b2_discriminator"] = args.b2_discriminator
    param["lr_combined"] = args.lr_combined
    param["b1_combined"] = args.b1_combined
    param["b2_combined"] = args.b2_combined        
    param["batch_size"] = int(args.batch_size/2)
    param["class_loss_weight"] = args.classifier_loss_weight
    param["dis_loss_weight"] = args.discriminator_loss_weight    
    param["drop_classifier"] = args.dropout_classifier
    param["drop_discriminator"] = args.dropout_discriminator
    param["test_interval"] = args.test_interval
    param["source_path"] = args.source_path
    param["target_path"] = args.target_path
    param["snapshot_interval"] = args.snapshot_interval
    # param["output_path"] = os.path.join("./Snapshot", args.output_dir)

    # Create directory for saving models and log files
    # if not os.path.exists(param["output_path"]):
        # os.mkdir(param["output_path"])
    
    # Load source and target data
    param["source_data"], param["source_label"] = data_loader(param["source_path"], param["inp_dims"])
    param["target_data"], param["target_label"] = data_loader(param["target_path"], param["inp_dims"])

    # Encode labels into one-hot format
    param["source_label"], param["target_label"] = one_hot_encoding(param)

    # Train data
    train(param)
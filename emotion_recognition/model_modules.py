from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, concatenate, add, Activation
from tensorflow.keras.models import Model,Sequential

# define a function than creates a VGG module
def vgg_block(layer_in, n_filters, n_conv):
    """
    function that creates a VGG module
    
    Params:
    layer_in: the prior layer in the model
    n_filters: the number of filter in the convolution layer
    n_conv: the number of convolution layer you want to stack together
    
    Returns:
    layer_out:
    """
    
    # add convolutional layers
    for i in range(n_conv):
        layer_in = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same', activation='relu')(layer_in)
        
    # add the maxpooling layer
    layer_out = MaxPooling2D(pool_size=(2,2), strides=(2,2))(layer_in)
    
    return layer_out

# create a function that adds the classifiers at the bottom of the model
def classifier(layer_in, num_classes, activation):
    """
    function than first flattens the previous layer in the model and than adds a fullyconnected layer
    
    Params:
    layer_in: the prior layer in the model
    num_classes: the number of units in the fully connected layer
    activation: the activation to be used in the fully connected layer
    
    Returns:
    layer_out
    """
    # flatten the input
    layer_in = Flatten()(layer_in)
    
    # add fully connecte layer
    # layer_in = Dense(units=num_classes, activation=activation)(layer_in)
    layer_out = Dense(units=num_classes, activation=activation)(layer_in)
    
    #return layer_in
    return layer_out

# define function that creates the naive form of inception module
def naive_inception_module(layer_in, filters_1, filters_2, filters_3):
    """
    function than implements the simplest form of the inception module
    
    Params:
    layer_in: the prior layer in the model
    filters_1: number of filters in the first conv2d layer
    filters_2: number of filters in the second conv2d layer
    filters_3: number of filters in the third conv2d layer
    
    Returns:
    layer_out: 
    """
    # 1x1 convolution
    conv_1 = Conv2D(filters=filters_1, kernel_size=(1,1), padding='same', activation='relu')(layer_in)
    
    # 3x3 convolution
    conv_2 = Conv2D(filters=filters_2, kernel_size=(3,3), padding='same', activation='relu')(layer_in)
    
    # 5x5 convolution
    conv_3 = Conv2D(filters=filters_3, kernel_size=(5,5), padding='same', activation='relu')(layer_in)
    
    # 3x3 maxpooling
    pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(layer_in)
    
    # concatenate the filters
    layer_out = concatenate([conv_1, conv_2, conv_3, pool], axis=-1)
    
    # return the concatenated layer
    return layer_out

# define function that creates the inception module with dimensionality reduction
def inception_module(inputs, filters_1, filters_2, filters_3):
    """
    function than implements the inception module with dimensionality reductione
    
    Params:
    inputs: the prior layer in the model
    filters_1: number of filters in the first conv2d layer
    filters_2: number of filters in the second conv2d layer
    filters_3: number of filters in the third conv2d layer
    
    Returns:
    layer_out: 
    """
    # define the first tower
    # the input first goes through the 1x1 convolution...
    tower_1 = Conv2D(filters=filters_1, kernel_size=(1,1), padding='same', activation='relu')(inputs)
    #... and than throught the 3x3 convolution
    tower_1 = Conv2D(filters=filters_1, kernel_size=(3,3), padding='same', activation='relu')(tower_1)

    # the second tower is structured very similarly. the only difference is that
    # the second con2d layer will have a kernel_size of 5 by 5
    tower_2 = Conv2D(filters=filters_2, kernel_size=(1,1), padding='same', activation='relu')(inputs)
    tower_2 = Conv2D(filters=filters_2, kernel_size=(5,5), padding='same', activation='relu')(tower_2)

    # in constrast with the previous two towers, the third one does not start with a 
    # con2d layer but with a Maxpooling insted. This occurs because the con2d with kernel
    # size = (1,1) is already present in the second step
    tower_3 = MaxPooling2D(pool_size=(3,3), padding='same', strides=(1,1))(inputs)
    tower_3 = Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu')(tower_3)
    
    # finally we can also add the single conv2d layer
    tower_4 = tower_3 = Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu')(inputs)
    
    # now we need to merge everything together and pass it to the next layer
    layer_out = concatenate([tower_1, tower_2, tower_3,tower_4], axis=1)    
    # return the concatenated layer
    return layer_out

# implement the identity residual module
def residual_module(layer_in, n_filters):
    """
    function that implements the identity residual module
    
    Params:
    layer_in: the prior layer in the model
    n_filters: number of filters to be used in the convolutional layers
    
    Returns:
    layer_out:
    """
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    # conv1
    conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    # conv2
    conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
    # add filters, assumes filters/channels last
    layer_out = add([conv2, merge_input])
    # activation function
    layer_out = Activation('relu')(layer_out)
    
    return layer_out


# Setup model for fine tuning
def setup_model(model, trainable):
    # Freeze the un-trainable layers of the model base
    for layer in model.layers[:(len(model.layers) - trainable)]:
        layer.trainable = False

    for layer in model.layers[(len(model.layers) - trainable):]:
        layer.trainable = True


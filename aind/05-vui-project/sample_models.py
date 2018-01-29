from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)
# from keras.layers import AtrousConvolution1D as DilConv1D # deprecated -> Conv1D with dilation_rate
from keras.layers import MaxPooling1D
from keras.layers import Dropout

# model 0
def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    # Add recurrent layer
    # Add softmax activation layer
    input_data = Input(name='the_input', shape=(None, input_dim))
    simp_rnn = GRU(output_dim, 
                   return_sequences=True, 
                   implementation=2, 
                   name='rnn')(input_data)
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model 1
def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    # Add recurrent layer ***** can be exchanged for LSTM or SimpleRNN
    # TODO: Add batch normalization 
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    # Add softmax activation layer
    input_data = Input(name='the_input', shape=(None, input_dim))
    simp_rnn = GRU(units, 
                   activation=activation, 
                   return_sequences=True, 
                   implementation=2, 
                   name='rnn')(input_data)
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim), name='td_dense')(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model 2
def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    # Add convolutional layer
    # Add batch normalization
    # Add a recurrent layer
    # TODO: Add batch normalization
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    # Add softmax activation layer
    input_data = Input(name='the_input', shape=(None, input_dim))
    cnn = Conv1D(filters, 
                 kernel_size, 
                 strides=conv_stride, 
                 padding=conv_border_mode,
                 activation='relu',
                 name='cnn')(input_data)
    bn_cnn = BatchNormalization(name='bn_cnn')(cnn)
    simp_rnn = GRU(units, 
                   activation='relu', 
                   return_sequences=True, 
                   implementation=2, 
                   name='rnn')(bn_cnn)
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim), name='td_dense')(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

# model 3
def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    # TODO: Add recurrent layers, each with batch normalization
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    # Add softmax activation layer
    input_data = Input(name='the_input', shape=(None, input_dim))
    simp_rnns = []
    bn_rnns = []
    for i in range(recur_layers):
        if i is 0:
            simp_rnns.append(GRU(units, 
                                 activation='relu', 
                                 return_sequences=True, 
                                 implementation=2, 
                                 name='rnn'+str(i))(input_data))
        else:
            simp_rnns.append(GRU(units, 
                                 activation='relu', 
                                 return_sequences=True, 
                                 implementation=2, 
                                 name='rnn'+str(i))(bn_rnns[i-1]))
        bn_rnns.append(BatchNormalization(name='bn_rnn'+str(i))(simp_rnns[i]))
    time_dense  = TimeDistributed(Dense(output_dim), name='td_dense')(bn_rnns[recur_layers-1])
    """
    if recur_layers == 1:
        simp_rnn = GRU(units, 
                       activation='relu',
                       return_sequences=True, 
                       implementation=2, 
                       name='rnn_1')(input_data)
        bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    if recur_layers == 2:
        simp_rnn_1 = GRU(units, 
                         activation='relu',
                         return_sequences=True, 
                         implementation=2, 
                         name='rnn_1')(input_data)
        bn_rnn_1 = BatchNormalization(name='bn_rnn_1')(simp_rnn_1)
        simp_rnn = GRU(units, 
                       activation='relu',
                       return_sequences=True, 
                       implementation=2, 
                       name='rnn_2')(bn_rnn_1)
        bn_rnn = BatchNormalization(name='bn_rnn_2')(simp_rnn)
    if recur_layers == 3:
        simp_rnn_2 = GRU(units, 
                         activation='relu',
                         return_sequences=True, 
                         implementation=2, 
                         name='rnn_1')(input_data)
        bn_rnn_2 = BatchNormalization(name='bn_rnn_1')(simp_rnn_2)
        simp_rnn_1 = GRU(units, 
                         activation='relu',
                         return_sequences=True, 
                         implementation=2, 
                         name='rnn_2')(bn_rnn_2)
        bn_rnn_1 = BatchNormalization(name='bn_rnn_2')(simp_rnn_1)
        simp_rnn = GRU(units, 
                       activation='relu',
                       return_sequences=True, 
                       implementation=2, 
                       name='rnn_3')(bn_rnn_1)
        bn_rnn = BatchNormalization(name='bn_rnn_3')(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim), name='td_dense')(bn_rnn)
    """
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model 4
def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    # TODO: Add bidirectional recurrent layer
    # *Add TimeDistributed(Dense(output_dim)) layer
    # Add softmax activation layer
    input_data = Input(name='the_input', shape=(None, input_dim))
    bidir_rnn = Bidirectional(GRU(units, 
                                  return_sequences=True, 
                                  name='gru'), 
                              merge_mode='concat', 
                              name='bidir_rnn')(input_data)
    time_dense = TimeDistributed(Dense(output_dim), name='td_dense')(bidir_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# final model
def final_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, dropout, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    # TODO: Specify the layers in your network
    # TODO: Add softmax activation layer
    input_data = Input(name='the_input', shape=(None, input_dim))
    cnn = Conv1D(filters, 
                 kernel_size, 
                 strides=conv_stride, 
                 padding=conv_border_mode,
                 activation='relu',
                 name='cnn')(input_data)
    dropout1 = Dropout(dropout)(cnn)
    bn_cnn = BatchNormalization(name='bn_cnn')(dropout1)
    simp_rnn = GRU(units, 
                   activation='relu', 
                   return_sequences=True, 
                   implementation=2, 
                   dropout=dropout, 
                   recurrent_dropout=dropout, 
                   name='rnn')(bn_cnn)
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    bidir_rnn1 = Bidirectional(GRU(units, 
                                   activation='relu', 
                                   return_sequences=True, 
                                   implementation=2),
                               merge_mode='concat',
                               name='bidir_rnn1')(bn_rnn)
    bidir_rnn2 = Bidirectional(GRU(units, 
                                   activation='relu', 
                                   return_sequences=True, 
                                   implementation=2), 
                               merge_mode='concat',
                               name='bidir_rnn2')(bidir_rnn1)
    dropout2 = Dropout(dropout)(bidir_rnn2)
    time_dense = TimeDistributed(Dense(output_dim), name='td_dense')(dropout2)
    dropout3 = Dropout(dropout)(time_dense)
    y_pred = Activation('softmax', name='softmax')(dropout3)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model 

# model 5 (custom 0)
def custom_model_0(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, rate, output_dim=29):
    """ 
    *** cnn with dropout ***
    """
    # Main acoustic input
    # Add convolutional layer
    # Add 1st dropout
    # Add batch normalization
    # Add a recurrent layer with additional dropout
    # Add batch normalization
    # Add a TimeDistributed(Dense(output_dim)) layer
    # Add 3rd dropout
    # Add softmax activation layer
    input_data = Input(name='the_input', shape=(None, input_dim))
    cnn = Conv1D(filters, 
                 kernel_size, 
                 strides=conv_stride, 
                 padding=conv_border_mode,
                 activation='relu',
                 name='cnn')(input_data)
    dropout_1 = Dropout(rate)(cnn)
    bn_cnn = BatchNormalization(name='bn_cnn')(dropout_1)
    simp_rnn = GRU(units, 
                   activation='relu', 
                   return_sequences=True, 
                   implementation=2, 
                   dropout=rate, 
                   recurrent_dropout=rate)(bn_cnn)
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim), name='td_dense')(bn_rnn)
    dropout_3 = Dropout(rate)(time_dense)
    y_pred = Activation('softmax', name='softmax')(dropout_3)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

# model 6 (custom 1)
def custom_model_1(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, dilation, output_dim=29):
    """ 
    *** dilated cnn layer ***
    """
    # Main acoustic input
    # Add dilated convolutional layer
    # Add batch normalization
    # Add a recurrent layer
    # Add batch normalization
    # Add a TimeDistributed(Dense(output_dim)) layer
    # Add softmax activation layer
    input_data = Input(name='the_input', shape=(None, input_dim))
    dcnn = Conv1D(filters, 
                  kernel_size, 
                  strides=conv_stride, 
                  padding=conv_border_mode,
                  dilation_rate=dilation,
                  activation='relu',
                  name='dcnn')(input_data)
    """
    dcnn = DilConv1D(filters, 
                            kernel_size,
                            atrous_rate=dilation,
                            strides=conv_stride, 
                            padding=conv_border_mode,
                            activation='relu',
                            name='dcnn')(input_data)
    """
    bn_dcnn = BatchNormalization(name='bn_dcnn')(dcnn)
    simp_rnn = GRU(units, 
                   activation='relu',
                   return_sequences=True, 
                   implementation=2, 
                   name='rnn')(bn_dcnn)
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim), name='td_dense')(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: dcnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, dilation)
    print(model.summary())
    return model

def dcnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation):
    """ Compute the length of the output sequence after dilated 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride // 2      # added: // 2

# model 7 (custom 2)
def custom_model_2(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, dilation, output_dim=29):
    """ 
    *** cnn with max_pooling ***
    """
    # Main acoustic input
    # Add convolutional layer
    # Add max pooling
    # Add batch normalization
    # Add a recurrent layer
    # Add batch normalization
    # Add a TimeDistributed(Dense(output_dim)) layer
    # Add softmax activation layer
    input_data = Input(name='the_input', shape=(None, input_dim))
    cnn = Conv1D(filters, 
                 kernel_size, 
                 strides=conv_stride, 
                 padding=conv_border_mode,
                 activation='relu',
                 name='cnn')(input_data)
    mp_cnn = MaxPooling1D(2, name='max_pool')(cnn)
    bn_cnn = BatchNormalization(name='bn_cnn')(mp_cnn)
    simp_rnn = GRU(units, 
                   activation='relu', 
                   return_sequences=True, 
                   implementation=2, 
                   name='rnn')(bn_cnn)
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim), name='td_dense')(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: dcnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, dilation)
    print(model.summary())
    return model

# model 8 (custom 3)
def custom_model_3(input_dim, units, recur_layers, output_dim=29):
    """ 
    *** deep bidirectional rnn ***
    """
    # Main acoustic input
    # Add bidirectional recurrent layers
    # Add a TimeDistributed(Dense(output_dim)) layer
    # Add softmax activation layer
    input_data = Input(name='the_input', shape=(None, input_dim))
    bidir_rnns = []
    bn_rnns = []
    for i in range(recur_layers):
        if i is 0:
            bidir_rnns.append(Bidirectional(GRU(units, 
                                                return_sequences=True, 
                                                name='gru'+str(i)),
                                            merge_mode='concat',
                                            name='bidir_rnn'+str(i))(input_data))
        else:
            bidir_rnns.append(Bidirectional(GRU(units, 
                                                return_sequences=True, 
                                                name='gru'+str(i)), 
                                            merge_mode='concat',
                                            name='bidir_rnn'+str(i))(bidir_rnns[i-1]))
    time_dense  = TimeDistributed(Dense(output_dim), name='td_dense')(bidir_rnns[recur_layers-1])
    """
    for i in range(recur_layers):
        if i is 0:
            bidir_rnns.append(Bidirectional(GRU(units, return_sequences=True, name='gru'+str(i)),
                                            merge_mode='concat',
                                            name='bidir_rnn'+str(i))(input_data))
        else:
            bidir_rnns.append(Bidirectional(GRU(units, return_sequences=True, name='gru'+str(i)), 
                                            merge_mode='concat',
                                            name='bidir_rnn'+str(i))(bn_rnns[i-1]))
        bn_rnns.append(BatchNormalization(name='bn_rnn'+str(i))(bidir_rnns[i]))
    time_dense  = TimeDistributed(Dense(output_dim), name='td_dense')(bn_rnns[recur_layers-1])
    """
    """
    if recur_layers == 1:
        bidir_rnn = Bidirectional(LSTM(units, return_sequences=True, name='lstm'), name='bidir_rnn')(input_data)
        bn_bidir_rnn = BatchNormalization(name='bn_bidir_rnn')(bidir_rnn)
    if recur_layers == 2:
        bidir_rnn_1 = Bidirectional(LSTM(units, return_sequences=True, name='lstm_1'), name='bidir_rnn_1')(input_data)
        bn_bidir_rnn_1 = BatchNormalization(name='bn_bidir_rnn_1')(bidir_rnn_1)
        bidir_rnn = Bidirectional(LSTM(units, return_sequences=True, name='lstm_2'), name='bidir_rnn_2')(bn_bidir_rnn_1)
        bn_bidir_rnn = BatchNormalization(name='bn_bidir_rnn_2')(bidir_rnn)
    if recur_layers == 3:
        bidir_rnn_2 = Bidirectional(LSTM(units, return_sequences=True, name='lstm_1'), name='bidir_rnn_1')(input_data)
        bn_bidir_rnn_2 = BatchNormalization(name='bn_bidir_rnn_1')(bidir_rnn_2)
        bidir_rnn_1 = Bidirectional(LSTM(units, return_sequences=True, name='lstm_2'), name='bidir_rnn_2')(bn_bidir_rnn_2)
        bn_bidir_rnn_1 = BatchNormalization(name='bn_bidir_rnn_2')(bidir_rnn_1)
        bidir_rnn = Bidirectional(LSTM(units, return_sequences=True, name='lstm_3'), name='bidir_rnn_3')(bn_bidir_rnn_1)
        bn_bidir_rnn = BatchNormalization(name='bn_bidir_rnn_3')(bidir_rnn)
    time_dense = TimeDistributed(Dense(output_dim), name='td_dense')(bn_bidir_rnn)
    """
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model
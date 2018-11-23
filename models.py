import tensorflow as tf
import utils
import abc
from utils import dropout_tf
class BaseModel(object):
    """Sequence-to-sequence base class.
    """

    def __init__(self,
                 hparams,
                 iterator,
                 mode=tf.contrib.learn.ModeKeys.TRAIN,
                 scope=None,
                 single_cell_fn=None):
        """Create the model.

        Args:
          hparams: Hyperparameter configurations.
          mode: TRAIN | EVAL | INFER
          iterator: Dataset Iterator that feeds data.
          source_vocab_table: Lookup table mapping source words to ids.
          target_vocab_table: Lookup table mapping target words to ids.
          reverse_target_vocab_table: Lookup table mapping ids to target words. Only
            required in INFER mode. Defaults to None.
          scope: scope of the model.
          single_cell_fn: allow for adding customized cell. When not specified,
            we default to model_helper._single_cell
        """
        self.iterator = iterator
        self.handle = iterator.handle
        self.mode = tf.placeholder(tf.bool)
        self.mode_ = mode
        # Initializer
        # initializer = tf.keras.initializers.glorot_normal
        initializer = tf.contrib.layers.xavier_initializer()
        tf.get_variable_scope().set_initializer(initializer)

        # Embeddings
        # TODO(ebrevdo): Only do this if the mode is TRAIN?
        self.init_embeddings(hparams, scope)

        ## When mode == Infer, each element of res is a scalar

        res = self.build_graph(hparams, scope=scope)

        self.positive_prob = res[-1][:,1]
        # self.eval_loss = res[1]
        if self.mode_== tf.contrib.learn.ModeKeys.TRAIN:
            self.labels = res[2]
            self.batch_size = tf.size(self.iterator.label)
            self.train_loss = res[1]
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels,tf.reshape(res[3],[-1])),'float'),name='accuracy')
            tf.summary.scalar('accuracy',self.accuracy)


        ## Learning rate
        self.global_step = tf.Variable(0, trainable=False)

        params = tf.trainable_variables()

        # if self.mode == 'training':
        if self.mode_ ==tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = tf.constant(hparams.learning_rate)
            opt = tf.train.AdamOptimizer(self.learning_rate)

            gradients = tf.gradients(
                self.train_loss,
                params,
                colocate_gradients_with_ops=True)
            from utils import gradient_clip
            clipped_gradients, gradient_norm_summary = gradient_clip(
                gradients, max_gradient_norm=10.0)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.update = opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step)

            # Summary
            self.train_summary = tf.summary.merge([
                                                      tf.summary.scalar("lr", self.learning_rate),
                                                      tf.summary.scalar("train_loss", self.train_loss),
                                                  ] + gradient_norm_summary)

        if self.mode_ == tf.contrib.learn.ModeKeys.INFER:

            self.infer_summary = self._get_infer_summary(hparams)

        # Saver
        self.saver = tf.train.Saver(tf.global_variables())

        # Print trainable variables
        print("# Trainable variables")
        for param in params:
            print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                    param.op.device))

        self.merge = tf.summary.merge_all()
    def init_embeddings(self, hparams, scope):
        """Init embeddings."""
        from utils import EmbeddingTransformer
        embeddings = EmbeddingTransformer(hparams.glove).embeddings
        embedding_encoder_glove = tf.get_variable('word_embeddings', shape=[embeddings.shape[0]-2,embeddings.shape[1]],initializer=tf.zeros_initializer,trainable=False)
        self.embedding_placeholder = tf.placeholder(tf.float32, embedding_encoder_glove.shape)
        self.embedding_init = embedding_encoder_glove.assign(self.embedding_placeholder)
        # embedding_encoder_glove = tf.Variable(
        #            initial_value=embeddings[1:],
        #            dtype=tf.float32,
        #            trainable=False,
        #            name="encoder_glove")
        embedding_encoder_pad = tf.Variable(initial_value=tf.zeros([1,embeddings.shape[1]]),
                                            dtype=tf.float32,
                                            trainable=False,
                                            name='encoder_pad')
        embedding_encoder_unk = tf.Variable(initial_value=tf.truncated_normal(shape=[1,embeddings.shape[1]],stddev=1e-5),
                                            dtype=tf.float32,
                                            trainable=True,
                                            name='encoder_unk')
        self.embedding_encoder = tf.concat([embedding_encoder_pad,embedding_encoder_unk,embedding_encoder_glove],axis=0)


    def train(self,handle,sess):
        # assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        # assert tf.cond(self.mode)
        return sess.run([self.merge,
                         self.update,
                         self.train_loss,
                         self.accuracy,
                         self.global_step,
                         self.batch_size],feed_dict={self.mode:True,self.handle:handle})

    def fully_connect_layer(self, _encoder_state,num_class):
        logits = tf.layers.dense(inputs=_encoder_state, units=num_class, activation=None, kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(1e-4,1e-4))

        return logits


    def build_graph(self, hparams, scope = None):
        """Subclass must implement this method.

          Creates classifier. The feature comes from sentence representation
          Args:
            hparams: Hyperparameter configurations.
            scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

          Returns:
            A tuple of the form (logits, loss),
        where:
          logits: float32 Tensor [batch_size x num_classes].
          loss: the total loss / batch_size.

        Raises
        """
        print("# creating %s graph ..." % self.mode)
        dtype = tf.float32

        with tf.variable_scope(scope or "classifier", dtype=dtype):
            #Encoder
            # concat_hidden, encoder_state = self._build_encoder(hparams)
            encoder_state = self._build_encoder(hparams)
            encoder_state = tf.layers.dense(encoder_state,units=64)
            dropout = dropout_tf(hparams.dropout,self.mode)
            encoder_state = tf.nn.dropout(encoder_state,keep_prob=1-dropout)

            logits = self.fully_connect_layer(encoder_state, 2)
            if self.mode_ != tf.contrib.learn.ModeKeys.INFER:
                from utils import get_device_str
                with tf.device(get_device_str(1 - 1, 1)):
                    labels = self.iterator.label
                    loss = self._compute_loss(logits, labels,0.0)
                    # loss = tf.Print(loss,['loss',loss])
            else:
                loss = None
                labels = None
            #return None is for compatability with other model
            sample_id = [tf.argmax(input=logits, axis=1)]
            sample_id = tf.transpose(sample_id)
            return logits, loss, labels, sample_id, tf.nn.softmax(logits)#last sample_id is a place ho

    @abc.abstractmethod
    def _build_encoder(self, hparams):
        """Subclass must implement this.

        Build and run an RNN encoder.

        Args:
          hparams: Hyperparameters configurations.

        Returns:
          A tuple of encoder_outputs and encoder_state.
        """
        pass

    def _build_encoder_cell(self, hparams, num_layers, num_residual_layers,
                            base_gpu=0):
        """Build a multi-layer RNN cell that can be used by encoder."""

        return utils.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=1.0,
            dropout=hparams.dropout,
            num_gpus=1,
            mode=self.mode,
            base_gpu=base_gpu,)

    def _compute_loss(self, logits, labels, label_smoothing=0.0):
        labels = tf.reshape(labels, [-1])
        one_hot_label = tf.one_hot(labels, 2)
        loss = tf.losses.softmax_cross_entropy(one_hot_label, logits, label_smoothing=label_smoothing)
        return loss



    def _get_infer_summary(self, hparams):
        return tf.no_op()

    def infer(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run([
            self.infer_logits, self.infer_summary, self.sample_id, self.softmax
        ])




class Model(BaseModel):
    """Sequence-to-sequence dynamic model.

    This class implements a multi-layer recurrent neural network as encoder,
    and a multi-layer recurrent neural network decoder.
    """
    def attention(self, concat_hidden):
        dropout = dropout_tf(0.1,self.mode)
        concat_hidden = tf.nn.dropout(concat_hidden, keep_prob=1-dropout)
        # print("concate_hidden",concat_hidden)
        _,seq_len, dim_feature = concat_hidden.get_shape().as_list()
        W = tf.Variable(tf.truncated_normal(
            [dim_feature, 100],0, 0.1), name='attention_w')
        v = tf.Variable(tf.truncated_normal([100],0,0.1), name='attention_v')
        b = tf.Variable(tf.zeros([100]), name='attention_bias')
        # concat_hidden = tf.transpose(concat_hidden, [1,0,2])
        Wh = tf.reshape(tf.matmul(tf.reshape(concat_hidden, [-1, dim_feature]), W), [tf.shape(concat_hidden)[0], seq_len, -1])
        vtan = tf.reduce_sum(tf.tanh(Wh+b)*v, -1)

        ah = tf.nn.softmax(vtan)
        fw = concat_hidden*tf.expand_dims(ah, -1)
        res = tf.reduce_sum(fw, 1)
        return res
    def _build_encoder(self, hparams):
        """Build an encoder."""

        iterator = self.iterator

        subject = iterator.subject
        bodyText = iterator.snippet
        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype
            # Look up embedding, emp_inp: [batch_size,max_time, num_units]
            encoder_emb_inp_subject = tf.nn.embedding_lookup(
                self.embedding_encoder, subject)
            encoder_emb_inp_bodyText = tf.nn.embedding_lookup(
                self.embedding_encoder,bodyText
            )

            # Encoder_outpus: [batch_size, batch_size,num_units]

            num_bi_layers = int(2 / 2)
            num_bi_residual_layers = int(0 / 2)
            print("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                  (num_bi_layers, num_bi_residual_layers))
            with tf.variable_scope('lstm.subject'):
                encoder_outputs_subject, bi_encoder_state_subject = (
                    self._build_bidirectional_rnn(
                        inputs=encoder_emb_inp_subject,
                        sequence_length=tf.shape(subject),
                        dtype=dtype,
                        hparams=hparams,
                        num_bi_layers=1,
                        num_bi_residual_layers=0
                    ))

            with tf.variable_scope('lstm.bodyText'):
                encoder_outputs_bodyText, bi_encoder_state_bodyText = (
                    self._build_bidirectional_rnn(
                        inputs=encoder_emb_inp_bodyText,
                        sequence_length=tf.shape(bodyText),
                        dtype=dtype,
                        hparams=hparams,
                        num_bi_layers=1,
                        num_bi_residual_layers=0
                    ))
            encoder_state_subject = tf.concat([state.h for state in bi_encoder_state_subject], axis=-1)
            encoder_state_bodyText = tf.concat([state.h for state in bi_encoder_state_bodyText], axis=-1)
            if hparams.attention:
                sentence_embedding_subject = self.attention(encoder_outputs_subject)
                sentence_embedding_bodyText = self.attention(encoder_outputs_bodyText)
                sentence_embedding = tf.concat([sentence_embedding_subject,sentence_embedding_bodyText],axis=1)
            else:
                sentence_embedding = tf.concat([encoder_state_subject,encoder_state_bodyText],axis=1)

        return sentence_embedding

    def _build_bidirectional_rnn(self, inputs, sequence_length,
                                 dtype, hparams,
                                 num_bi_layers,
                                 num_bi_residual_layers,
                                 base_gpu=0):
        """Create and call biddirectional RNN cells.

        Args:
          num_residual_layers: Number of residual layers from top to bottom. For
            example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
            layers in each RNN cell will be wrapped with `ResidualWrapper`.
          base_gpu: The gpu device id to use for the first forward RNN layer. The
            i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
            device id. The `base_gpu` for backward RNN cell is `(base_gpu +
            num_bi_layers)`.

        Returns:
          The concatenated bidirectional output and the bidirectional RNN cell"s
          state.
        """
        # Construct forward and backward cells
        fw_cell = self._build_encoder_cell(hparams,
                                           num_bi_layers,
                                           num_bi_residual_layers,
                                           base_gpu=base_gpu)
        bw_cell = self._build_encoder_cell(hparams,
                                           num_bi_layers,
                                           num_bi_residual_layers,
                                           base_gpu=(base_gpu + num_bi_layers))

        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            dtype=dtype,)
        # sequence_length=sequence_length)

        return tf.concat(bi_outputs, -1), bi_state

class GBDT_LSTM(Model):

    def build_graph(self, hparams, scope = None):
        """Subclass must implement this method.
        """
        dtype = tf.float32

        with tf.variable_scope(scope or "classifier", dtype=dtype):
            #Encoder
            # concat_hidden, encoder_state = self._build_encoder(hparams)
            encoder_state = super(GBDT_LSTM,self)._build_encoder(hparams)
            sparse_feature = tf.cast(self.iterator.sparse_feature,tf.float32)
            #encoder_concate = tf.concat([encoder_state,gbdt_features],axis=1)
            hidden_layer = tf.layers.dense(inputs=encoder_state, units=hparams.hidden_size, activation=tf.nn.relu, kernel_regularizer=None)
            dropout = dropout_tf(hparams.dropout,self.mode)
            hidden_layer = tf.nn.dropout(hidden_layer,keep_prob=1-dropout)
            full_layer = tf.concat([hidden_layer,sparse_feature],axis=1)
            logits = self.fully_connect_layer(full_layer, 2)
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                from utils import get_device_str
                with tf.device(get_device_str(1 - 1, 1)):
                    labels = self.iterator.label
                    loss = self._compute_loss(logits, labels,0.0)
                    # loss = tf.Print(loss,['loss',loss])
            else:
                loss = None
                labels = None
            #return None is for compatability with other model
            sample_id = [tf.argmax(input=logits, axis=1)]
            sample_id = tf.transpose(sample_id)
            return logits, loss, labels, sample_id, tf.nn.softmax(logits)#last sample_id is a place ho





class CNN(BaseModel):

    def cnn_layer(self,embedding,filter_sizes,num_filters,sequence_length):
        pooled_outputs = []
        embedded_chars_expanded = tf.expand_dims(embedding,-1)
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, 00, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        return h_pool_flat





    def _build_encoder(self,hparams,scope=None):
        iterator = self.iterator

        subject = iterator.subject
        bodyText = iterator.bodyText


        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype
            # Look up embedding, emp_inp: [batch_size,max_time, num_units]
            encoder_emb_inp_subject = tf.nn.embedding_lookup(
                self.embedding_encoder, subject)
            encoder_emb_inp_bodyText = tf.nn.embedding_lookup(
                self.embedding_encoder,bodyText
            )

        filter_sizes = [1,2,3,4]
        num_filters = 128
        with tf.name_scope('subject'):
            subject_encoding = self.cnn_layer(encoder_emb_inp_subject, filter_sizes,num_filters,100)
        with tf.name_scope('bodyText'):
            bodyText_encoding = self.cnn_layer(encoder_emb_inp_bodyText, filter_sizes,num_filters,400)

        sentence_embedding = tf.concat([subject_encoding,bodyText_encoding],axis=1)
        dropout = dropout_tf(hparams.cnn_dropout,self.mode)
        sentence_embedding = tf.nn.dropout(sentence_embedding,keep_prob=1-dropout)
        return sentence_embedding

    def build_graph(self, hparams, scope = None):

        dtype = tf.float32

        with tf.variable_scope(scope or "classifier", dtype=dtype):
            #Encoder

            sentence_embedding= self._build_encoder(hparams)



            logits = self.fully_connect_layer(sentence_embedding, 2)


            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                with tf.device(utils.get_device_str(0,1)):
                    labels = self.iterator.label
                    loss = self._compute_loss(logits, labels,0.0)
            else:
                loss = None
                labels = None
            #return None is for compatability with other model
            sample_id = [tf.argmax(input=logits, axis=1)]
            sample_id = tf.transpose(sample_id)
            return logits, loss, labels, sample_id, tf.nn.softmax(logits)












class QuasiFastText(Model):

    def _build_encoder(self, hparams):
        """Build an encoder."""

        iterator = self.iterator

        subject = iterator.subject
        bodyText = iterator.bodyText
        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype
            # Look up embedding, emp_inp: [batch_size,max_time, num_units]
            encoder_emb_inp_subject = tf.nn.embedding_lookup(
                self.embedding_encoder, subject)
            encoder_emb_inp_bodyText = tf.nn.embedding_lookup(
                self.embedding_encoder,bodyText
            )

        sentence_embedding_subject = self.attention(encoder_emb_inp_subject)
        sentence_embedding_bodyText = self.attention(encoder_emb_inp_bodyText)
        sentence_embedding = tf.concat([sentence_embedding_subject,sentence_embedding_bodyText],axis=1)

        return sentence_embedding


class DNN(BaseModel):


    def cnn_layer(self, embedding, filter_sizes, num_filters, sequence_length):

        pooled_outputs = []
        embedded_chars_expanded = tf.expand_dims(embedding,-1)
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, 300, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        return h_pool_flat





    def _build_subject_encoder(self,hparams,scope=None):
        iterator = self.iterator

        subject = iterator.subject
        snippet = iterator.snippet
        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype
            # Look up embedding, emp_inp: [batch_size,max_time, num_units]
            encoder_emb_inp_subject = tf.nn.embedding_lookup(
                self.embedding_encoder, subject)
            encoder_emb_inp_snippet = tf.nn.embedding_lookup(
                self.embedding_encoder,snippet)

        filter_sizes = [1,2,3,4]
        num_filters = 128
        with tf.name_scope('subject'):
            subject_encoding = self.cnn_layer(encoder_emb_inp_subject, filter_sizes,num_filters,30)
        with tf.name_scope('bodyText'):
            snippet_encoding = self.cnn_layer(encoder_emb_inp_snippet, filter_sizes,num_filters,500)
        email_embedding = tf.concat([subject_encoding,snippet_encoding],axis=1)
        dropout = dropout_tf(hparams.cnn_dropout,self.mode)
        email_embedding = tf.nn.dropout(email_embedding,keep_prob=1-dropout)
        return email_embedding

    def _build_encoder(self, hparams):
        """Build an encoder."""

        iterator = self.iterator

        feature = iterator.sparse_feature
        feature = tf.cast(feature,tf.float32)
        email_encoder = self._build_subject_encoder(hparams)
        #add batch norm
        regularizer = tf.contrib.layers.l2_regularizer(scale=hparams.l2)
        print(f"l2 regualrization constant is {hparams.l2}")

        temp_feature = tf.layers.dense(email_encoder,512,activation=None,kernel_regularizer=regularizer)
        temp_feature = tf.layers.batch_normalization(temp_feature,training=self.mode)
        temp_feature = tf.nn.relu(temp_feature)
        #features = tf.concat([tf.reshape(feature[:,0],shape=[-1,1]),temp_feature],axis=1)
        features = tf.concat([feature,temp_feature],axis=1)
        #features = temp_feature
        #add batch normalization

        features = tf.layers.dense(features,128,activation=None,kernel_regularizer=regularizer)
        features = tf.layers.batch_normalization(features,training=self.mode)
        features = tf.nn.relu(features)
        dropout = dropout_tf(hparams.cnn_dropout,self.mode)
        features = tf.nn.dropout(features,1-dropout)
        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype
            #features = tf.layers.batch_normalization(features,training = self.mode)
            #features = tf.layers.dense(features,training=(self.mode_==tf.contrib.learn.ModeKeys.TRAIN))
            #features = tf.contrib.layers.layer_norm(features)
            # Look up embedding, emp_inp: [batch_size,max_time, num_units]
            #temp_feature = tf.layers.dense(features,128,activation=tf.nn.relu)

        return features #
    def build_graph(self, hparams, scope = None):
        dtype=tf.float32
        with tf.variable_scope(scope or "classifier", dtype=dtype):
            #Encoder
            feature = self._build_encoder(hparams)

            logits = self.fully_connect_layer(feature, 2)


            if self.mode_ != tf.contrib.learn.ModeKeys.INFER:
                #with tf.device(utils.get_device_str(0,1)):
                labels = self.iterator.label
                loss = self._compute_loss(logits, labels, 0.0)
                loss += tf.losses.get_regularization_loss()
                tf.summary.scalar('loss',loss)
            else:
                loss = None
                labels = None
            #return None is for compatability with other model
            sample_id = [tf.argmax(input=logits, axis=1)]
            sample_id = tf.transpose(sample_id)
            return logits, loss, labels, sample_id, tf.nn.softmax(logits)

    def _compute_loss(self,logits,labels,label_smoothing=0):
        #add class weight to the sample
        labels = tf.reshape(labels, [-1])
        one_hot_label = tf.one_hot(labels, 2)
        ratio = 0.43/(1+0.43)
        class_weight  = tf.constant([[ratio,1-ratio]])
        weight_per_label =  tf.reshape(tf.matmul(one_hot_label, tf.transpose(class_weight)),[-1])
        loss = tf.losses.softmax_cross_entropy(one_hot_label, logits, weights=weight_per_label,label_smoothing=label_smoothing)
        return loss



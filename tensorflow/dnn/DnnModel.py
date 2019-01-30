class DNNModel:
    def __init__(self, config):
        self.config = config
        self.endpoints = {}
        self.graph = tf.Graph()
        
    def build_net(self, x_placeholder, y_placeholder):
        with self.graph.as_default():
            with tf.variable_scope(self.config["name"]):

                self.X = x_placeholder
                self.y = y_placeholder

                layer_output_li = []
                for idx, n in enumerate(self.config["n_li"][:-1]):
                    with tf.name_scope("Layer_" + str(idx)) as scope:
                        previous_dim = self.config["n_li"][idx]
                        next_dim = self.config["n_li"][idx + 1]
                        shape = [previous_dim, next_dim]
                        pre_layer_output = layer_output_li[-1] if idx > 0 else self.X
                        self.__set_weight_and_bias(idx, shape)
                        layer = self.__set_layer_endpoint(idx, pre_layer_output)
                        layer_output_li.append(layer)

            with tf.name_scope("Cost") as scope:
                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y))
                cost_sum = tf.summary.scalar("Cost", self.cost)
            
            self.predict = tf.argmax(self.logits, 1)
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
    def __set_weight_and_bias(self, idx, shape):
        if self.config["initializer_li"][idx] == "random_normal":
            self.endpoints["W_" + str(idx)] = tf.Variable(tf.random_normal(shape), name = "W_" + str(idx))
        elif self.config["initializer_li"][idx] == "xavier":
            self.endpoints["W_" + str(idx)] = tf.get_variable("W_" + str(idx), shape=shape, 
                                initializer=tf.contrib.layers.xavier_initializer())
        self.endpoints["b_" + str(idx)] = tf.Variable(tf.random_normal(shape[1:]), name = "b_" + str(idx))
        W_hist = tf.summary.histogram("W_hist_" + str(idx), self.endpoints["W_" + str(idx)])
        b_hist = tf.summary.histogram("b_hist_" + str(idx), self.endpoints["b_" + str(idx)])
    
    def __set_layer_endpoint(self, idx, pre_layer_output):
        W = self.endpoints["W_" + str(idx)]
        b = self.endpoints["b_" + str(idx)]
        if idx + 1 == len(self.config["n_li"][:-1]):
            self.logits = tf.matmul(pre_layer_output, W) + b
            layer_hist = tf.summary.histogram("Layer_hist_" + str(idx), self.logits)
            return self.logits
        if self.config["activation_li"][idx] == "sigmoid":
            self.endpoints["layer_" + str(idx)] = tf.sigmoid(tf.matmul(pre_layer_output, W) + b)
        elif self.config["activation_li"][idx] == "relu":
            self.endpoints["layer_" + str(idx)] = tf.nn.relu(tf.matmul(pre_layer_output, W) + b)
        layer_hist = tf.summary.histogram("Layer_hist_" + str(idx), self.endpoints["layer_" + str(idx)])       
        return self.endpoints["layer_" + str(idx)]
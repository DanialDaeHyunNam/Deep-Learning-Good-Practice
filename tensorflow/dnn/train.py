def train(model, X_train, y_train, lr=1e-4, epoch=15, batch_size=200):
    with model.graph.as_default():
        x_placeholder = tf.placeholder(tf.float32, shape=[None, model.config["n_features"]], name="X")
        y_placeholder = tf.placeholder(tf.float32, shape=[None, model.config["n_class"]], name="y")

    model.build_net(x_placeholder, y_placeholder)
    
    with tf.Session(graph=model.graph) as sess:
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(model.cost)
        init = tf.global_variables_initializer()
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs", sess.graph)
        sess.run(init)
        for step in range(epoch):
            total_batch = int(len(X_train)/batch_size)
            c_avg = 0
            for i in range(total_batch):
                batch_x = X_train[batch_size*i : batch_size*(i+1)]
                batch_y = y_train[batch_size*i : batch_size*(i+1)]
                summary, c, _  = sess.run([merged_summary, model.cost, train_op], 
                                              feed_dict={model.X: batch_x, model.y: batch_y})
                c_avg = c_avg + (c/total_batch)
                writer.add_summary(summary, i)
            print(step, c_avg)
        saver = tf.train.Saver()
        saver.save(sess, './checkpoint/' + model.config["name"] + '.chkp')
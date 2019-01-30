def predict(model, x_test):
    with tf.Session(graph=model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './checkpoint/' + model.config["name"] + '.chkp')
        return sess.run([model.predict], feed_dict={model.X : x_test})

def accuracy(model, x_test, y_test):
    with tf.Session(graph=model.graph) as sess: 
        saver = tf.train.Saver()
        saver.restore(sess, './checkpoint/' + model.config["name"] + '.chkp')
        return sess.run([model.accuracy], feed_dict={model.X : x_test, model.y : y_test})
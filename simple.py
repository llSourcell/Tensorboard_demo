import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

# This network is the same as the previous one except with an extra hidden layer + dropout
def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    # Add layer name scopes for better graph visualization
    with tf.name_scope("layer1"):
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu(tf.matmul(X, w_h))
    with tf.name_scope("layer2"):
        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h2))
    with tf.name_scope("layer3"):
        h2 = tf.nn.dropout(h2, p_keep_hidden)
        return tf.matmul(h2, w_o)

#Step 1 - Get Input Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

#Step 2 - Create input and output placeholders for data
X = tf.placeholder("float", [None, 784], name="X")
Y = tf.placeholder("float", [None, 10], name="Y")

#Step 3 - Initialize weights
w_h = init_weights([784, 625], "w_h")
w_h2 = init_weights([625, 625], "w_h2")
w_o = init_weights([625, 10], "w_o")

#Step 4 - Add histogram summaries for weights
tf.histogram_summary("w_h_summ", w_h)
tf.histogram_summary("w_h2_summ", w_h2)
tf.histogram_summary("w_o_summ", w_o)

#Step 5 - Add dropout to input and hidden layers
p_keep_input = tf.placeholder("float", name="p_keep_input")
p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")

#Step 6 - Create Model
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

#Step 7 Create cost function
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    # Add scalar summary for cost tensor
    tf.scalar_summary("cost", cost)

#Step 8 Measure accuracy
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1)) # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
    # Add scalar summary for accuracy tensor
    tf.scalar_summary("accuracy", acc_op)

#Step 9 Create a session
with tf.Session() as sess:
    # Step 10 create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    writer = tf.train.SummaryWriter("./logs/nn_logs", sess.graph) # for 0.8
    merged = tf.merge_all_summaries()

    # Step 11 you need to initialize all variables
    tf.initialize_all_variables().run()

    #Step 12 train the  model
    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        summary, acc = sess.run([merged, acc_op], feed_dict={X: teX, Y: teY,
                                          p_keep_input: 1.0, p_keep_hidden: 1.0})
        writer.add_summary(summary, i)  # Write summary
        print(i, acc)                   # Report the accuracy

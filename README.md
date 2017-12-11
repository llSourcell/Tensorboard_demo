# Tensorboard_demo
Tensorboard Demo by @Sirajology on Youtube

This is the code for the Tensorboard Video by @Sirajology on [Youtube](https://youtu.be/3bownM3L5zM). In this repo, there are 2 versions of the classifier, a simple one and a complex one. Input_data.py retrieves and formats the [MNIST character dataset](http://yann.lecun.com/exdb/mnist/).

#Challenge (Due by Sept 30 2016)

The challenge for this video is to use Tensorboard to visualize some audio data. See the [docs](https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html#audio_summary) for more info on how to do this. The first to do this who posts their repo in the comments section of the video gets a shoutout from me on Sept 30 2016 during the release of my video on that date!

### Requirements
*[Tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html)

### Set Up
Note: Can't use 'conda' to install tensorboard
Make Sure you have tensorflow AND tensorboard installed. These used to be combined in one package, but now need to be installed independently.

Using tensorflow-gpu seems to cause some issues. Use an environment that's running tensorflow (no-gpu)

##### Anaconda
Create an environment named "sandbox" for testing
```
conda create --name sandbox
activate sandbox
conda install tensorflow
pip install tensorflow-tensorboard
```

### Usage
##### Train your model

```
python simple.py
```
or 
```
python complex.py
```

##### Tensorboard
You can view the results in Tensorboard after training by typing the following into terminal

```
tensorboard --logdir=./logs/nn_logs
```
Terminal will output an address to visit in your browser. Go to that address to see your tensorboard. That's it!

# Credits

The Tensorflow team at Google! I've merely created a wrapper around this code to make it easy to use.

### Troubleshooting
```
WARNING:tensorflow:Found more than one graph event per run, or there was a metagraph containing a graph_def, as well as one or more graph events.  Overwriting the graph with the newest event.
```

If you run multiple times, it will create and save a new graph each time. Delete the files in the folder './logs/nn_logs'.

If the output directs you to HTTP://0.0.0.0.6006, you will probably see no output. This happens if you're running tensorflow, but not tensorboard. Install tensorboard and try again.

Currently, complex.py doesn't save any graph, so tensorboard has nothing to display. Working on a fix.

##### Hint:
If possible, train in an environment with tensorflow-gpu first, then use a different environment with tensorflow (no-gpu) to start tensorboard. The training will run much more quickly in tensorflow-gpu.





# Tensorboard_demo
Tensorboard Demo by @Sirajology on Youtube

This is the code for the Tensorboard Video by @Sirajology on [Youtube](https://youtu.be/3bownM3L5zM). In this repo, there are 2 versions of the classifier, a simple one and a complex one. Input_data.py retrieves and formats the [MNIST character dataset](http://yann.lecun.com/exdb/mnist/).

#Challenge (Due by Sept 30 2016)

The challenge for this video is to use Tensorboard to visualize some audio data. See the [docs](https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html#audio_summary) for more info on how to do this. The first to do this who posts their repo in the comments section of the video gets a shoutout from me on Sept 30 2016 during the release of my video on that date!

#Requirements
*[Tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html)

#Usage

Once you have Tensorflow installed, just run 

```
python simple.py
```
or 
```
python complex.py
```

to train your model. You can view the results in Tensorboard after training by typing the following into terminal

```
tensorboard --logdir=./logs/nn_logs
```
Terminal will output an address to visit in your browser. Go to that address to see your tensorboard. That's it!

# Credits

The Tensorflow team at Google! I've merely created a wrapper around this code to make it easy to use.


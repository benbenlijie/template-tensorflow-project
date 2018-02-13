# Tensorflow Project Template
This is a tensorflow project template that I summed up based on
[MrGemy95/Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) and my own practical experience.

# How to use
Do the following when you want to implement a model like style transfer model or something else:
-  Create a class named StyleTransfer that inherit the "base_model" class

```python

    class StyleTransferModel(BaseModel):
        def __init__(self, config, data_loader):
            super(TemplateModel, self).__init__(config, data_loader)
  ```
- Override these two functions "_build_train_model" and "_build_evaluate_model" and define the ```self.train_op``` and ```self.loss_op``` in theses two functions.
    
```python
     def _build_train_model(self):
        self.train_op = tf.constant([1, 2, 3], dtype=tf.float16)
        self.loss_op = tf.constant(0.1)
        tf.summary.scalar("test", self.loss_op)
            
     def _build_evaluate_model(self):
        pass

  ```
   
- Create a trainer that inherit from "base_train" class
```python

    class StyleTransferTrainer(BaseTrain):
        def __init__(self, sess, model, config):
            super(StyleTransferTrainer, self).__init__(sess, model, config)
```
- Override these two functions "train_step", "log_step"
```python

    def train_step(self):
        self.sess.run(self.model.train_op)

    def log_step(self, elapsed_time=0):
        loss = self.sess.run(self.model.loss_op)
        sys.stdout.write("total loss {}, secs/step {}".format(loss, elapsed_time))
        sys.stdout.flush()
        summary_str = self.sess.run(self.model.summary_op)
        self.model.summary.add_summary(summary_str)
        self.model.summary.flush()

```
- In main file create the data loader, model, trainer like this
```python
    image_loader = ImageDataLoader(config, True)

    model = StyleTransferModel(config, image_loader)
    model.init_train_model()
    with tf.Session() as sess:
        trainer = StyleTransferTrainer(sess, model, config)
        trainer.train()
```


Folder structure
--------------

```
├──  base
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_train.py - this file contains the abstract class of the trainer.
│
│
├── model               -This folder contains any model of your project.
│   └── example_model.py
│
│
├── trainer             -this folder contains trainers of your project.
│   └── example_trainer.py
│   
├──  mains              - here's the main/s of your project (you may need more than one main.
│                         
│  
├──  data _loader
│    ├── coco_data_loader.py   - this file contains custom tfrecords data loader inherit from RecordDataLoader.
│    ├── image_data_loader.py   - this file contains image data loader inherit from BaseDataLoader.
│    └── data_loader.py  - this file contains the abstract class of the data_loader.
│ 
└── utils
     ├── config.py
     └── utils.py

```


import json
import numpy as np
import tensorflow as tf
from keras.layers import *
import keras.optimizers
from keras import objectives
import keras.callbacks
from keras.callbacks import Callback, CallbackList
from .util           import ensure_list, NpEncoder, curry
from .util.tuning    import InvalidHyperparameterError
#from .util.layers    import LinearSchedule
from .util.layers    import *
import os.path
import imageio

from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
#from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend






# modified version
import progressbar
class DynamicMessage(progressbar.DynamicMessage):
    def __call__(self, progress, data):
        val = data['dynamic_messages'][self.name]
        if val:
            return '{}'.format(val)
        else:
            return 6 * '-'



class myCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, epoch, logs={}):
        #print(self.model)
        # self.loss
        print("ok")
        return
    


# def _convert_numpy_and_scipy(x):
#     if isinstance(x, np.ndarray):
#         dtype = None
#         if issubclass(x.dtype.type, np.floating):
#             dtype = backend.floatx()
#         return tensor_conversion.convert_to_tensor_v2_with_dispatch(
#             x, dtype=dtype
#         )


class Network:
    """Base class for various neural networks including GANs, AEs and Classifiers.
Provides an interface for saving / loading the trained weights as well as hyperparameters.

Each instance corresponds to a directory (specified in the `path` variable in the initialization),
which contains the learned weights as well as several json files for the metadata.
If a network depends on another network (i.e. AAE depends on SAE), it is customary to
save the network inside the dependent network. (e.g. Saving an AAE to mnist_SAE/_AAE)

PARAMETERS dict in the initialization argument is stored in the instance as well as 
serialized into a JSON string and is subsequently reloaded along with the weights.
This dict can be used while building the network, making it easier to perform a hyperparameter tuning.
"""
    def __init__(self,path,parameters={}):
        import subprocess
        subprocess.call(["mkdir","-p",path])
        self.built = False
        self.built_aux = False
        self.compiled = False
        self.loaded = False
        self.parameters = parameters
        self.custom_log_functions = {}
        self.metrics = []
        self.parameters["newLoss"] = 0
        self.transitions = None
        #self.parameters["newLoss"] = [None,None] # 1st ele for transitions, 2nd ele for states
        # self.parameters["newLoss0"] = tf.Variable([999], tf.int8)
        # self.parameters["newLoss1"] = tf.Variable([999], tf.int8)

        self.nets    = [None]
        self.losses  = [None]
        if parameters:
            # handle the test-time where parameters is not given
            self.path = os.path.join(path,"logs",self.parameters["time_start"])
            self.file_writer = tf.summary.FileWriter(self.path)
            self.epoch = LinearSchedule(schedule={
                0:0,
                self.parameters["epoch"]:self.parameters["epoch"],
            }, name="epoch")
            self.callbacks = [
                # keras.callbacks.LambdaCallback(
                #     on_epoch_end = self.save_epoch(path=self.path,freq=1000)),
                keras.callbacks.LambdaCallback(
                    on_epoch_end=self.epoch.update),
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.LambdaCallback(
                    on_epoch_end = self.bar_update,),
                keras.callbacks.TensorBoard(
                    histogram_freq = 0,
                    log_dir     = self.path+"/tensorboard",
                    write_graph = False)
            ]
        else:
            self.path = path
            self.callbacks = []




    def build(self,*args,**kwargs):
        """An interface for building a network. Input-shape: list of dimensions.
Users should not overload this method; Define _build_around() for each subclass instead.
This function calls _build bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        if self.built:
            # print("Avoided building {} twice.".format(self))
            return
        print("Building networks")
        self._build_around(*args,**kwargs)
        self.built = True
        print("Network built")
        return self

    def _build_around(self,*args,**kwargs):
        """An interface for building a network.
This function is called by build() only when the network is not build yet.
Users may define a method for each subclass for adding a new build-time feature.
Each method should call the _build_around() method of the superclass in turn.
Users are not expected to call this method directly. Call build() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        return self._build_primary(*args,**kwargs)

    def _build_primary(self,*args,**kwargs):
        pass

    def build_aux(self,*args,**kwargs):
        """An interface for building an additional network not required for training.
To be used after the training.
Input-shape: list of dimensions.
Users should not overload this method; Define _build_around() for each subclass instead.
This function calls _build bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        if self.built_aux:
            # print("Avoided building {} twice.".format(self))
            return
        print("Building auxiliary networks")
        self._build_aux_around(*args,**kwargs)
        self.built_aux = True
        print("Auxiliary network built")
        return self

    def _build_aux_around(self,*args,**kwargs):
        """An interface for building an additional network not required for training.
To be used after the training.
Input-shape: list of dimensions.
This function is called by build_aux() only when the network is not build yet.
Users may define a method for each subclass for adding a new build-time feature.
Each method should call the _build_aux_around() method of the superclass in turn.
Users are not expected to call this method directly. Call build() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        return self._build_aux_primary(*args,**kwargs)

    def _build_aux_primary(self,*args,**kwargs):
        pass

    def compile(self,*args,**kwargs):
        """An interface for compiling a network."""
        if self.compiled:
            # print("Avoided compiling {} twice.".format(self))
            return
        print("Compiling networks")
        self._compile(*args,**kwargs)
        self.compiled = True
        self.loaded = True
        print("Network compiled")
        return self


    def cust_loss( self, y_true, y_pred ) :
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(squared_difference, axis=-1)

    def _compile(self,optimizers):
        """An interface for compileing a network."""
        # default method.
        print(f"there are {len(self.nets)} networks.")
        print(f"there are {len(optimizers)} optimizers.")
        print(f"there are {len(self.losses)} losses.")
        
        
        assert len(self.nets) == len(optimizers)
        assert len(self.nets) == len(self.losses)
        for net, o, loss in zip(self.nets, optimizers, self.losses):
            print(f"compiling {net} with {o}, {loss}.")
            print(id(loss))
            print(net.losses)
            print(self.losses)
            net.compile(optimizer=o, loss=self.cust_loss, metrics=self.metrics)
            #net.compile(optimizer=o, loss=self.my_loss_fn, metrics=self.metrics)
        return

    def local(self,path=""):
        """A convenient method for converting a relative path to the learned result directory
into a full path."""
        return os.path.join(self.path,path)

    def save(self,path=""):
        """An interface for saving a network.
Users should not overload this method; Define _save() for each subclass instead.
This function calls _save bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        print("Saving the network to {}".format(self.local(path)))
        os.makedirs(self.local(path),exist_ok=True)
        self._save(path)
        print("Network saved")
        return self

    def _save(self,path=""):
        """An interface for saving a network.
Users may define a method for each subclass for adding a new save-time feature.
Each method should call the _save() method of the superclass in turn.
Users are not expected to call this method directly. Call save() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        for i, net in enumerate(self.nets):
            net.save_weights(self.local(os.path.join(path,f"net{i}.h5")))

        with open(self.local(os.path.join(path,"aux.json")), "w") as f:
            json.dump({"parameters":self.parameters,
                       "class"     :self.__class__.__name__,
                       "input_shape":self.net.input_shape[1:]}, f , skipkeys=True, cls=NpEncoder, indent=2)

    def save_epoch(self, freq=10, path=""):
        def fn(epoch, logs):
            if (epoch % freq) == 0:
                self.save(os.path.join(path,str(epoch)))
        return fn

    def load(self,allow_failure=False,path=""):
        """An interface for loading a network.
Users should not overload this method; Define _load() for each subclass instead.
This function calls _load bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        if self.loaded:
            # print("Avoided loading {} twice.".format(self))
            return

        if allow_failure:
            try:
                print("Loading networks from {} (with failure allowed)".format(self.local(path)))
                self._load(path)
                self.loaded = True
                print("Network loaded")
            except Exception as e:
                print("Exception {} during load(), ignored.".format(e))
        else:
            print("Loading networks from {} (with failure not allowed)".format(self.local(path)))
            self._load(path)
            self.loaded = True
            print("Network loaded")
        return self

    def _load(self,path=""):
        """An interface for loading a network.
Users may define a method for each subclass for adding a new load-time feature.
Each method should call the _load() method of the superclass in turn.
Users are not expected to call this method directly. Call load() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        with open(self.local(os.path.join(path,"aux.json")), "r") as f:
            data = json.load(f)
            _params = self.parameters
            self.parameters = data["parameters"]
            self.parameters.update(_params)
            self.build(tuple(data["input_shape"]))
            self.build_aux(tuple(data["input_shape"]))
        for i, net in enumerate(self.nets):
            net.load_weights(self.local(os.path.join(path,f"net{i}.h5")))

    def reload_with_shape(self,input_shape,path=""):
        """Rebuild the network for a shape that is different from the training time, then load the weights."""
        print(f"rebuilding the network with a new shape {input_shape}.")
        # self.build / self.loaded flag are ignored
        self._build_around(input_shape)
        self._build_aux_around(input_shape)
        for i, net in enumerate(self.nets):
            net.load_weights(self.local(os.path.join(path,f"net{i}.h5")))

    def initialize_bar(self):
        import progressbar
        widgets = [
            progressbar.Timer(format="%(elapsed)s"),
            " ", progressbar.Counter(), " | ",
            # progressbar.Bar(),
            progressbar.AbsoluteETA(format="%(eta)s"), " ",
            DynamicMessage("status", format='{formatted_value}')
        ]
        if "start_epoch" in self.parameters:
            start_epoch = self.parameters["start_epoch"] # for resuming
        else:
            start_epoch = 0
        self.bar = progressbar.ProgressBar(max_value=start_epoch+self.parameters["epoch"], widgets=widgets)

    def bar_update(self, epoch, logs):
        "Used for updating the progress bar."

        if not hasattr(self,"bar"):
            self.initialize_bar()
            #from colors import color
            from functools import partial
            #self.style = partial(color, fg="black", bg="white")

        tlogs = {}
        for k in self.custom_log_functions:
            tlogs[k] = self.custom_log_functions[k]()
        for k in logs:
            if k[:2] == "t_":
                tlogs[k[2:]] = logs[k]
        vlogs = {}
        for k in self.custom_log_functions:
            vlogs[k] = self.custom_log_functions[k]()
        for k in logs:
            if k[:2] == "v_":
                vlogs[k[2:]] = logs[k]

        if (epoch % 10) == 9:
            #self.bar.update(epoch+1, status = self.style("[v] "+"  ".join(["{} {:8.3g}".format(k,v) for k,v in sorted(vlogs.items())])) + "\n")
            self.bar.update(epoch+1, status = "[v] "+"  ".join(["{} {:8.3g}".format(k,v) for k,v in sorted(vlogs.items())]) + "\n")
        else:
            self.bar.update(epoch+1, status = "[t] "+"  ".join(["{} {:8.3g}".format(k,v) for k,v in sorted(tlogs.items())]))

    @property
    def net(self):
        return self.nets[0]

    @net.setter
    def net(self,net):
        self.nets[0] = net
        return net

    @property
    def loss(self):
        return self.losses[0]

    @loss.setter
    def loss(self,loss):
        self.losses[0] = loss
        return loss

    def add_metric(self, name,loss=None):
        if loss is None:
            loss = eval(name)
        thunk = lambda *args: loss
        thunk.__name__ = name
        self.metrics.append(thunk)



    def load_image(self, name):
        image = imageio.imread(f"{name}.png") / 255
        if len(image.shape) == 2:
            image = image.reshape(*image.shape, 1)
        image = self.output.normalize(image)
        return image


    def mse(self, x,y,axis=None):
        result = np.mean(np.square(x - y),axis=axis)
        if result.size == 1:
            return float(result)
        else:
            return result

    def autoencode_image(self, name, image):
        state = self.encode(np.array([image]))[0].round().astype(int)
        image_rec = self.decode(np.array([state]))[0]
        print(f"{name} (input) min:",image.min(),"max:",image.max(),)
        print(f"{name} (recon) min:",image_rec.min(),"max:",image_rec.max(),)
        # print(f"{name} BCE:",bce(image,image_rec))
        # print(f"{name} MAE:",mae(image,image_rec))
        print(f"{name} MSE:",self.mse(image,image_rec))
        # print(state)
        # if image_diff(image,image_rec) > image_threshold:
        #     print("Initial state reconstruction failed!")
        #     sys.exit(3)
        return state, image_rec


    def load_and_encode_image(self, name):
        image = self.load_image(name)
        #from latplan.util.noise        import gaussian
        # if noise is not None:
        #     print(f"adding gaussian noise N(0,{noise})")
        #     image = gaussian(image0, noise)
        # else:
        state, image = self.autoencode_image(name,image)
        
        return state, image


    def plot_image(self, images,w=10,path="reco_image.png",verbose=False):
        import matplotlib.pyplot as plt
        import math
        l = 0
        l = len(images)
        h = int(math.ceil(l/w))
        plt.figure(figsize=(w*1.5, h*1.5))
        for i,image in enumerate(images):
            ax = plt.subplot(h,w,i+1)
            try:
                plt.imshow(image,interpolation='nearest',cmap='gray', vmin = 0, vmax = 1)
            except TypeError:
                TypeError("Invalid dimensions for image data: image={}".format(np.array(image).shape))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        print(path) if verbose else None
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


    def normalize(self, x,save=True):
        parameters = {}
        if ("mean" in parameters) and save:
            mean = np.array(parameters["mean"][0])
            std  = np.array(parameters["std"][0])
        else:
            mean               = np.mean(x,axis=0)
            std                = np.std(x,axis=0)
            if save:
                parameters["mean"] = [mean.tolist()]
                parameters["std"]  = [std.tolist()]
        print("normalized shape:",mean.shape,std.shape)
        return (x - mean)/(std+1e-20)


    def normalize_transitions(self, pres,sucs):
        """Normalize a dataset for image-based input format.
    Normalization is performed across batches."""
        B, *F = pres.shape
        transitions = np.stack([pres,sucs], axis=1) # [B, 2, F]
        print(transitions.shape)

        normalized  = self.normalize(np.reshape(transitions, [-1, *F])) # [2BO, F]
        states      = normalized.reshape([-1, *F])
        transitions = states.reshape([-1, 2, *F])
        return transitions, states




    def load_sokoban(self, track,num_examples,objects=True,**kwargs):
        # track : sokoban_image-20000-global-global-2-train
        # num_examples : 20000
        # objects = False
        # load_sokoban("sokoban_image-20000-global-global-2-train", 20000, False)
        
        #with np.load(os.path.join(latplan.__path__[0],"puzzles",track+".npz")) as data:
        with np.load(os.getcwd()+"/latplan/puzzles/sokoban_image-20000-global-global-2-train.npz") as data:
            pres   = data['pres'][:num_examples] / 255 # [B,O,sH*sW*C]
            sucs   = data['sucs'][:num_examples] / 255 # [B,O,sH*sW*C]
            bboxes = data['bboxes'][:num_examples] # [B,O,4]
            picsize = data['picsize']
            print("loaded. picsize:",picsize)

        self.parameters["picsize"] = [picsize.tolist()]
        self.parameters["generator"] = None

        pres = pres[:,0].reshape((-1, *picsize))
        sucs = sucs[:,0].reshape((-1, *picsize))
        transitions, states = self.normalize_transitions(pres,sucs)

        return transitions, states



    
    def echodo(self, cmd,*args,**kwargs):
        import subprocess
        print(cmd,flush=True)
        subprocess.run(cmd,*args,**kwargs)




    def train(self,train_data,
              val_data      = None,
              train_data_to = None,
              val_data_to   = None,
              resume        = False,
              **kwargs):
        """Main method for training.
 This method may be overloaded by the subclass into a specific training method, e.g. GAN training."""

        if resume:
            #print("resuming the training")
            #print(self.path)
            #print(self.parameters["resume_from"])
            #self.load(allow_failure=False, path="logs/"+self.parameters["resume_from"])
            self.load(allow_failure=False, path="")
        else:
            input_shape = train_data.shape[1:]
            self.build(input_shape)
            self.build_aux(input_shape)

        import inspect

        # print(inspect.getsourcefile(self.nets[0]))
        # exit()

        epoch      = self.parameters["epoch"]
        batch_size = self.parameters["batch_size"]
        optimizer  = self.parameters["optimizer"]
        lr         = self.parameters["lr"]
        clipnorm   = self.parameters["clipnorm"]
        # clipvalue  = self.parameters["clipvalue"]
        if "start_epoch" in self.parameters:
            start_epoch = self.parameters["start_epoch"] # for resuming
        else:
            start_epoch = 0

        # batch size should be smaller / eq to the length of train_data
        batch_size = min(batch_size, len(train_data))

        def make_optimizer(net):
            return getattr(keras.optimizers,optimizer)(
                lr,
                clipnorm=clipnorm
                # clipvalue=clipvalue,
            )

        print("self.losses")
        print(self.losses)
        

        self.optimizers = list(map(make_optimizer, self.nets))
        self.compile(self.optimizers)


        if val_data     is None:
            val_data     = train_data
        if train_data_to is None:
            train_data_to = train_data
        if val_data_to  is None:
            val_data_to  = val_data

        def replicate(thing):
            if isinstance(thing, tuple):
                thing = list(thing)
            if isinstance(thing, list):
                assert len(thing) == len(self.nets)
                return thing
            else:
                return [thing for _ in self.nets]

        train_data    = replicate(train_data)
        train_data_to = replicate(train_data_to)
        val_data      = replicate(val_data)
        val_data_to   = replicate(val_data_to)

        plot_val_data = np.copy(val_data[0][:1])
        self.callbacks.append(
            keras.callbacks.LambdaCallback(
                on_epoch_end = lambda epoch,logs: \
                    self.plot_transitions(
                        plot_val_data,
                        self.path+"/",
                        epoch=epoch),
                on_train_end = lambda _: self.file_writer.close()))

        #self.callbacks.append(myCallback)

        def assert_length(data):
            l = None
            for subdata in data:
                if not ((l is None) or (len(subdata) == l)):
                    return False
                l = len(subdata)
            return True

        assert assert_length(train_data   )
        assert assert_length(train_data_to)
        assert assert_length(val_data    )
        assert assert_length(val_data_to )

        def make_batch(subdata):
            # len: 15, batch: 5 -> 3
            # len: 14, batch: 5 -> 2
            # len: 16, batch: 5 -> 3
            # the tail is discarded
            for i in range(len(subdata)//batch_size):
                yield subdata[i*batch_size:(i+1)*batch_size]

        index_array = np.arange(len(train_data[0]))

        clist = CallbackList(callbacks=self.callbacks)
        clist.set_model(self.nets[0])
        clist.set_params({
            "batch_size": batch_size,
            "epochs": start_epoch+epoch,
            "steps": None,
            "samples": len(train_data[0]),
            "verbose": 0,
            "do_validation": False,
            "metrics": [],
        })
        self.nets[0].stop_training = False

        def generate_logs(data,data_to):
            losses = []
            logs   = {}
            for i, (net, subdata, subdata_to) in enumerate(zip(self.nets, data, data_to)):
                evals = net.evaluate(subdata,
                                     subdata_to,
                                     batch_size=batch_size,
                                     verbose=0)
                logs_net = { k:v for k,v in zip(net.metrics_names, ensure_list(evals)) }
                losses.append(logs_net["loss"])
                logs.update(logs_net)
            if len(losses) > 2:
                for i, loss in enumerate(losses):
                    logs["loss"+str(i)] = loss
            logs["loss"] = np.sum(losses)
            return logs

        some_count = 0

        is_correct=0


        try:
            clist.on_train_begin()
            logs = {}

            self.transitions, states = self.load_sokoban("sokoban_image-20000-global-global-2-train", 20000, False)


            resset=False

            #loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

            def my_loss_fn(y_true, y_pred):
                squared_difference = tf.square(y_true - y_pred)
                return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`


            for epoch in range(start_epoch,start_epoch+epoch):
                np.random.shuffle(index_array)
                indices_cache       = [ indices for indices in make_batch(index_array) ]
                train_data_cache    = [[ train_subdata   [indices] for train_subdata    in train_data    ] for indices in indices_cache ]
                train_data_to_cache = [[ train_subdata_to[indices] for train_subdata_to in train_data_to ] for indices in indices_cache ]
                clist.on_epoch_begin(epoch,logs)
                for train_subdata_cache,train_subdata_to_cache in zip(train_data_cache,train_data_to_cache):
                    for net, train_subdata_batch_cache,train_subdata_to_batch_cache in zip(self.nets, train_subdata_cache,train_subdata_to_cache):
                        
                        if epoch > 0 and some_count==0:
                            self.bar.update(epoch+1, status = "[before train batch] "+"\n")
                        
                        

                        net.train_on_batch(train_subdata_batch_cache, train_subdata_to_batch_cache)
                        
                        #print(train_subdata_batch_cache.shape) (400, 2, 28, 28, 3)
                        

                        
                        exit()
                        with tf.GradientTape() as tape:

                            xx = tf.Variable(3.0)

                            #y_pred, loss_value = net(tf.convert_to_tensor(train_subdata_batch_cache, dtype=tf.float32))
                            out = net(tf.convert_to_tensor(train_subdata_batch_cache, dtype=tf.float32))

                            
                                
                            print("outt value") # Tensor("model_1/dmerge_5/concat:0", shape=(?, 2, 28, 28, 3), dtype=float32)
                            print(out)
                            # [<tf.Tensor 'my_regularizer_layer_1/mul:0' shape=() dtype=float32>, <tf.Tensor 'model_1/my_regularizer_layer_1/mul:0' shape=() dtype=float32>]

                            #loss_value = my_loss_fn(train_subdata_batch_cache, out)


                            # no losses if no layer ...( MyRegularizerLayer )
                            # print(self.loss) # is <function BaseActionMixinAMA4Plus._build_primary.<locals>.loss at 0x7f6653ceb3a0>

                            #print(net.losses)
                            # net.loss ==  self.loss   TRUE
                            #   (<function BaseActionMixinAMA4Plus._build_primary.<locals>.loss )
                            # print(self.lol) == Tensor("add_23:0", shape=(?,), dtype=float32)
                            # Update the weights of the model to minimize the loss value.

                        if some_count > 1:
                            print("okk")
                            [var.name for var in tape.watched_variables()]


                        #gradients = tape.gradient(self.loss, net.trainable_weights)
                        # optimizer.apply_gradients(zip(gradients, net.trainable_weights))
                        



                        #mett = net.train_on_batch(None, train_subdata_to_batch_cache)

                        # _disallow_inside_tf_function('train_on_batch')
                        # with self.distribute_strategy.scope(), \
                        #     training_utils.RespectCompiledTrainableState(self):
                        # iterator = data_adapter.single_batch_iterator(self.distribute_strategy, x,
                        #                                                 y, sample_weight,
                        #                                                 class_weight)
                        # self.train_function = self.make_train_function()
                        # logs = self.train_function(iterator)


                      
                        # #loss_1 = lambda y_true, y_pred: -1 * loss_1(y_true, y_pred)
                        # #loss_fn = loss_1
                        # with tf.GradientTape() as tape:
                        #     #logits = net(tf.convert_to_tensor(train_subdata_batch_cache, dtype=tf.float32), training=True)
                        #     # print("logits")
                        #     # print(logits)
                        #     #self.encoder()
                        #     #net(tf.convert_to_tensor(train_subdata_batch_cache, dtype=tf.float32))
                        #     theres = self.autoencoder.predict(train_subdata_batch_cache)
                        #     print("theres.loss")
                        #     print(theres.loss)
                        #     exit()
                        #     input_dtypes = nest.map_structure(lambda t: t.dtype, net.inputs)
                        #     nest.map_structure(math_ops.cast, train_subdata_batch_cache, input_dtypes)
                        #     exit()
                        #     #print(logits.shape)
                        #     print(self.loss)
                        #     # input_shape = (28, 28, 3)
                        #     # x = Input(shape=(2,*input_shape))








                        # problem_dir=self.path+"/sokoban-2-False/003-000"
                        # domainfile = self.path+"/"+"domain.pddl"

                        # ### test is_correct every 10 epochs, on 1st batch, until it is (then stop)
                        # if epoch % 10 == 0 and epoch != 0 and some_count == 0 and is_correct == 0:
                        #     from latplan.util.planner import puzzle_module
                        #     self.parameters["generator"] = "latplan.puzzles.sokoban"
                        #     p = puzzle_module(self)
                        #     state, image  = self.load_and_encode_image("init") # returns the state and the reconstruction
                        #     image_rec = self.render(image) # unnormalize the image
                        #     is_correct = p.validate_statesBIS(np.array([image_rec]))
                        #     self.bar.update(epoch+1, status = "[is_correct] "+"  "+str(is_correct) + "\n")

                                
                            #     self.bar.update(epoch+1, status = "[after train batch] "+"\n")

                        ### when is_correct has been set to one, compute new loss on every batch

                        # is_correct=1
                        # if is_correct == 1 and epoch > 0:
                        #     # print("some count {}".format(some_count))
                        #     # if some_count % 2 == 0:
                        #     #     self.parameters["newLoss0"].assign([0])
                        #     #     self.parameters["newLoss1"].assign([0])
                        #     # else:
                        #     #     self.parameters["newLoss0"].assign([0])
                        #     #     self.parameters["newLoss1"].assign([0])                     

                        #     # self.bar.update(epoch+1, status = "[Before CSVs gen] "+"\n")
                        #     # ## generate CVSs | make sure it's in the right folder !!!
                        #     # self.dump_actions(transitions,batch_size = 1000)
                        #     # self.bar.update(epoch+1, status = "[after CSVs gen] "+"\n")

                        #     # ## generate domain.pddl | as above + try to generate without neg pred
                        #     # self.echodo(["./pddl-ama3.sh",self.path])
                        #     # self.bar.update(epoch+1, status = "[after pddl-ama3 gen] "+"\n")
                            

                        #     # ## generates PROBLEM
                        #     # from latplan.util.planner import puzzle_module, init_goal_misc, setup_planner_utils
                        #     # setup_planner_utils(self, problem_dir, self.path, "ama3")
                        #     # self.parameters["generator"] = "latplan.puzzles.sokoban"
                        #     # p = puzzle_module(self)
                        #     # init, goal = init_goal_misc(p,1,noise=None, sae=self)
                        #     # bits = np.concatenate((init,goal))
                        #     # ig          = "problem.ig"
                        #     # problemfile = problem_dir+"/problem.pddl"
                        #     # os.path.exists(ig) or np.savetxt(ig,[bits],"%d")
                        #     # self.echodo(["helper/ama3-problem.sh","problem.ig",problemfile])
                        #     # self.bar.update(epoch+1, status = "[after problem gen] "+"\n")
                    

                        #     # planfile=problem_dir+"/problem.plan"
                        #     # tracefile=problem_dir+"/trace.plan"
                        #     # csvfile=problem_dir+"/plan.csv"
                        #     # pngfile=problem_dir+"/problem.png"
                        #     # jsonfile    = problem_dir+"/problem.json"
                        #     # logfile     = problem_dir+"/problem.log"
                        #     # npzfile     = problem_dir+"/problem.npz"
                        #     # negfile     = problem_dir+"/problem.negative"

                        #     # ## START PLANNING AND GENERATE (or not) the .PLAN file
                        #     # self.echodo(["./small-launch.sh", planfile, domainfile, problemfile])
                        #     # self.bar.update(epoch+1, status = "[finished planning] "+"\n")

                        #     # ## IF PLAN FILE WAS NOT GENERATED, write 0 0
                        #     # if not os.path.exists(planfile):
                        #     #     self.parameters["newLoss"][0] = 0.
                        #     #     self.parameters["newLoss"][1] = 0.
                        #     #     print("NO PLAN FOUND")
                                
                        #     # else:
                        #     #     print("YES PLAN FOUND")
                        #     #     ## GENERATE the TRACEFILE
                        #     #     found = True
                        #     #     print("start running a validator")
                        #     #     self.echodo(["arrival", domainfile, problemfile, planfile, tracefile])
                        #     #     print("finished running a validator")


                        #     #     ## GENERATE THE PLAN.CSV
                        #     #     print("start parsing the plan")
                        #     #     with open(csvfile,"w") as f:
                        #     #         self.echodo(["lisp/ama3-read-latent-state-traces.bin", tracefile, str(len(init))],
                        #     #             stdout=f)


                        #     #     ## AFFECT the plan to a variable 'plan'
                        #     #     plan = np.loadtxt(csvfile, dtype=int)
                        #     #     print("finished parsing the plan")

                        #     #     ## If plan has no length, then we pass (and we delete any existing plan file)
                        #     #     ## and we write 00 to the LossFile
                        #     #     if plan.ndim != 2:
                        #     #         assert plan.ndim == 1
                        #     #         print("Found a plan with length 0; single state in the plan.")
                        #     #         if os.path.exists(planfile):
                        #     #             os.remove(planfile)
                        #     #         self.parameters["newLoss"][0] = 0.
                        #     #         self.parameters["newLoss"][1] = 0.
                        #     #         continue

                        #     #     ## 
                        #     #     self.bar.update(epoch+1, status = "[in planner 6] "+"\n")

                        #     #     ## PLOT THE PLAN
                        #     #     print("start plotting the plan")
                        #     #     self.plot_plan(plan, pngfile, verbose=True)
                        #     #     print("finished plotting the plan")


                        #     #     # CREATE IMAGES OF THE PLAN 
                        #     #     print("start archiving the plan")
                        #     #     plan_images = self.decode(plan)
                        #     #     np.savez_compressed(npzfile,img_states=plan_images)
                        #     #     print("finished archiving the plan")
                        #     #     print("start visually validating the plan image : transitions")
                        #     #     # note: only puzzle, hanoi, lightsout have the custom validator, which are all monochrome.
                        #     #     plan_images = self.render(plan_images) # unnormalize the image
                                
                                
                        #     #     # VALIDATION OF TRANSITIONS
                        #     #     validation = p.validate_transitions([plan_images[0:-1], plan_images[1:]])
                        #     #     print(validation)
                        #     #     valid_trans = bool(np.all(validation))
                        #     #     print("finished visually validating the plan image : transitions")

                        #     #     # VALIDATION OF STATES
                        #     #     print("start visually validating the plan image : states")
                        #     #     validation_bis = p.validate_states(plan_images)
                        #     #     valid_states = bool(np.all(validation_bis))
                        #     #     self.bar.update(epoch+1, status = "[in planner 7] "+"\n")

                        #     #     ## WRITING IN LossFile in function of
                        #     #     ## states and transitions validations
                        #     #     if valid_trans:
                        #     #         self.parameters["newLoss"][0] = 1.
                        #     #     else:
                        #     #         self.parameters["newLoss"][0] = 0.

                        #     #     if valid_states:
                        #     #         self.parameters["newLoss"][1] = 1.
                        #     #     else:
                        #     #         self.parameters["newLoss"][1] = 0.

                        #     # self.bar.update(epoch+1, status = "[in planner 8] "+"\n")

                        #     # ## WE REMOVE THE PLAN FILE
                        #     # if os.path.exists(planfile):
                        #     #     os.remove(planfile) 




























                        #print(p.returncode)


                        some_count+=1

                if epoch > 0 and epoch % 10 == 0 and is_correct == 1:
                    self.save()

                some_count = 0

                logs = {}
                for k,v in generate_logs(train_data, train_data_to).items():
                    logs["t_"+k] = v
                for k,v in generate_logs(val_data,  val_data_to).items():
                    logs["v_"+k] = v
                clist.on_epoch_end(epoch,logs)
                if self.nets[0].stop_training:
                    break
            clist.on_train_end()

        except KeyboardInterrupt:
            print("learning stopped\n")
        finally:
            print("in finally")
            #self.save()
            self.loaded = True
        return self

    def evaluate(self,*args,**kwargs):

        return np.sum([
            { k:v for k,v in zip(net.metrics_names,
                                 ensure_list(net.evaluate(*args,**kwargs)))}["loss"]
            for net in self.nets
        ])

    def report(self,train_data,
               batch_size=1000,
               test_data=None,
               train_data_to=None,
               test_data_to=None):
        pass

    def save_array(self,name,data):
        print("Saving to",self.local(name))
        with open(self.local(name), "wb") as f:
            np.savetxt(f,data,"%d")


    def _plot(self,path,columns,epoch=None):
        """yet another convenient function. This one swaps the rows and columns"""
        columns = list(columns)
        if epoch is None:
            rows = []
            for seq in zip(*columns):
                rows.extend(seq)
            from .util.plot import plot_grid
            plot_grid(rows, w=len(columns), path=path, verbose=True)
            return
        else:
            # assume plotting to tensorboard

            if (epoch % 10) != 0:
                return

            _, basename = os.path.split(path)
            import tensorflow as tf
            import keras.backend.tensorflow_backend as K
            import imageio
            import skimage

            # note: [B,H,W] monochrome image is handled by imageio.
            # usually [B,H,W,1].
            for i, col in enumerate(columns):
                col = skimage.util.img_as_ubyte(np.clip(col,0.0,1.0))
                for k, image in enumerate(col):
                    self.file_writer.add_summary(
                        tf.Summary(
                            value = [
                                tf.Summary.Value(
                                    tag  = basename.lstrip("_")+"/"+str(i),
                                    image= tf.Summary.Image(
                                        encoded_image_string =
                                        imageio.imwrite(
                                            imageio.RETURN_BYTES,
                                            image,
                                            format="png")))]),
                        epoch)
            return

import tcav.model as model
import tensorflow as tf

#Taken from TCAV/models but adds image shape function and get gradient function
class KerasModelWrapper(model.ModelWrapper):
  """ ModelWrapper for keras models

    By default, assumes that your model contains one input node, one output head
    and one loss function.
    Computes gradients of the output layer in respect to a CAV.

    Args:
        sess: Tensorflow session we will use for TCAV.
        model_path: Path to your model.h5 file, containing a saved trained
          model.
        labels_path: Path to a file containing the labels for your problem. It
          requires a .txt file, where every line contains a label for your
          model. You want to make sure that the order of labels in this file
          matches with the logits layers for your model, such that file[i] ==
          model_logits[i]
  """

  def __init__(
      self,
      sess,
      model_path,
      labels_path,
  ):
    self.sess = sess
    super(KerasModelWrapper, self).__init__()
    self.import_keras_model(model_path)
    self.labels = tf.io.gfile.GFile(labels_path).read().splitlines()

    # Construct gradient ops. Defaults to using the model's output layer
    self.y_input = tf.compat.v1.placeholder(tf.int64, shape=[None])
    self.loss = self.model.loss_functions[0](self.y_input,
                                             self.model.outputs[0])
    self._make_gradient_tensors()

  def id_to_label(self, idx):
    return self.labels[idx]

  def label_to_id(self, label):
    return self.labels.index(label)

  def import_keras_model(self, saved_path):
    """Loads keras model, fetching bottlenecks, inputs and outputs."""
    self.ends = {}
    self.model = tf.keras.models.load_model(saved_path)
    self.get_bottleneck_tensors()
    self.get_inputs_and_outputs_and_ends()

  def get_bottleneck_tensors(self):
    self.bottlenecks_tensors = {}
    layers = self.model.layers
    for layer in layers:
      if 'input' not in layer.name:
        self.bottlenecks_tensors[layer.name] = layer.output

  def get_inputs_and_outputs_and_ends(self):
    self.ends['input'] = self.model.inputs[0]
    self.ends['prediction'] = self.model.outputs[0]  

  def get_image_shape(self):
    return (224, 224, 3) 

  def get_gradient(self, acts, y, bottleneck_name):
    """Return the gradient of the loss with respect to the bottleneck_name.

    Args:
      acts: activation of the bottleneck
      y: index of the logit layer
      bottleneck_name: name of the bottleneck to get gradient wrt.
      example: input example. Unused by default. Necessary for getting gradients
        from certain models, such as BERT.

    Returns:
      the gradient array.
    """
    return self.sess.run(self.bottlenecks_gradients[bottleneck_name], {
        self.bottlenecks_tensors[bottleneck_name]: acts,
        self.y_input: y
    })
 
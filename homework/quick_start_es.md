# Inicio rápido para principiantes

`Ascend` `GPU` `CPU` `Principiante` `Proceso completo`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/tutorials/source_en/quick_start.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

Aqui se explica las funciones básicas de MindSpore para implementar acciones básicas en deep learning. Si buscas algo en específico, mira los links de cada sección.

## Configurar la informacion de arranque

MindSpore usa `context.set_context` para configurar la informacion necesaria para funcionar, por ejemplo, el modo de ejecución, información del backend e información del hardware.

Importa el módulo  `context` y configura la información necesaria.

```python
import os
import argparse
from mindspore import context

parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

args = parser.parse_known_args()[0]
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
```

Este ejemplo se ejecuta en modo gráfico. Puedes cambiar la configuración de hardware segun necesites. Por ejemplo, si el cófigo se ejecuta en un procesador Ascend AI, cambia `--device_target` to `Ascend`. Esta regla se aplica tambien al código que se ejecute en la CPU y la GPU. Para más detalles sobre los parámetros, ver [context.set_context](https://www.mindspore.cn/docs/api/en/r1.6/api_python/mindspore.context.html).

## Descargando el Dataset

El dataset MNIST usado en este ejemplo consiste de 10 clases de imagenes en blanco y negro de 28 x 28 pixeles. Tiene un set de entrenamiento de 60,000 ejemplos, y un set de test de 10,000 ejemplos.

Pulsa [aqui](http://yann.lecun.com/exdb/mnist/) para descargar y extraer el dataset MNIST, despues coloca el dataset segun la siguiente estructura de directorio. El siguiente código de ejemplo descarga y extrae el dataset al directorio especificado.

```python
import os
import requests

def download_dataset(dataset_url, path):
    filename = dataset_url.split("/")[-1]
    save_path = os.path.join(path, filename)
    if os.path.exists(save_path):
        return
    if not os.path.exists(path):
        os.makedirs(path)
    res = requests.get(dataset_url, stream=True, verify=False)
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)

train_path = "datasets/MNIST_Data/train"
test_path = "datasets/MNIST_Data/test"

download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-labels-idx1-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-images-idx3-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-labels-idx1-ubyte", test_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-images-idx3-ubyte", test_path)
```

La estructura de directorio es la que sigue:

```text
    ./datasets/MNIST_Data
    ├── test
    │   ├── t10k-images-idx3-ubyte
    │   └── t10k-labels-idx1-ubyte
    └── train
        ├── train-images-idx3-ubyte
        └── train-labels-idx1-ubyte

    2 directories, 4 files
```

## Procesado de información

Los Datasets son cruciales para el entrenamiento del modelo. Un buen dataset puede mejorar la precisión y eficiencia del entrenamiento.
MindSpore dispone de un módulo API `mindspore.dataset` para el procesado de información para almacenar muestras y etiquetas. Antes de cargar un dataset, se suele procesar el dataset. `mindspore.dataset` integra métodos de procesamiento de información comunes.

Importa `mindspore.dataset` y otros módulos de MindSpore.

```python
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
```

El procesado del Dataset consiste de los siguientes pasos:

1. Define la función `create_dataset` para crear un dataset.
2. Define las operaciones de mejora y procesado de información para prepararse para el mapeo de la misma.
3. Use the map function to apply data operations to the dataset.
4. Realizar operaciones de mezcla y de lotes en la información.

```python
def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    # Definir el dataset.
    mnist_ds = ds.MnistDataset(data_path)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # Definir el mapeo que realizar.
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # Usar la funcionde mapeo para aplicar las operaciones de información al dataset.
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=[resize_op, rescale_op, rescale_nml_op, hwc2chw_op], input_columns="image", num_parallel_workers=num_parallel_workers)


    # Realizar operaciones de mezcla, lote y repetición.
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(count=repeat_size)

    return mnist_ds
```

En la información precedente, `batch_size` indica el numero de registros de información en cada grupo. Cada grupo contiene 32 registros de información.

> MindSpore soporta multiples operaciones de procesado y mejora de la información. Para más detalles, ver [Procesando información](https://www.mindspore.cn/docs/programming_guide/en/r1.6/pipeline.html) y [Mejora de información](https://www.mindspore.cn/docs/programming_guide/en/r1.6/augmentation.html).

## Creación de un modelo

Para usar MindSpore para la definición de una red neuronal, hereda `mindspore.nn.Cell`. `Cell` es la clase base de todas las redes neuronales (como por ejemplo `Conv2d-relu-softmax`).

Define cada capa de una red neuronal en el método `__init__` de antemano, y define el método `construct` para completar la construcción de la red neuronal. Segun la estructura LeNet, define las capas de la red de la siguiente manera:

```python
import mindspore.nn as nn
from mindspore.common.initializer import Normal

class LeNet5(nn.Cell):
    """
    Lenet network structure
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # Define la operacion requerida.
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        # Usa la operación definida para construir la red de envío.
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Instancia la red.
net = LeNet5()
```

## Optimizando los parámetros del modelo.

Para entrenar una red neuronal es necesario definir una función de perdida y un optimizador.

Las funciones de pérdidaLoss functions soportadas por MindSpore incluyen `SoftmaxCrossEntropyWithLogits`, `L1Loss`, y `MSELoss`. `SoftmaxCrossEntropyWithLogits` utiliza la función de pérdida de entropía cruzada.

```python
# Definir la función de pérdida.
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
```

> Para más información sobre como usar funciones de pérdida en mindspor , ver [Funciones de pérdida](https://www.mindspore.cn/tutorials/en/r1.6/optimization.html#loss-functions).

MindSpore soporta los optimizadores `Adam`, `AdamWeightDecay`, y `Momentum` . El siguiente código utiliza el optimizador `Momentum` como ejemplo.

```python
# Define el optimizador.
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
```

> Para más información sobre como usar un optimizador en mindspore, ver [Optimizador](https://www.mindspore.cn/tutorials/en/r1.6/optimization.html#optimizer).

## Entrenando y guardando el modelo

MindSpore provee el mecanismo de callback para ejecutar logica personalizada durante el entrenamiento. El siguiente código utiliza `ModelCheckpoint` del framework como ejemplo.
`ModelCheckpoint` puede guardar el modelo de red y sus parámetros para ajustes posteriores.

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
# Definir los parámetros de guardado de modelo.
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
# Usar dichos parámetros.
ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
```

La API `model.train` provista por MindSpore puede ser usada para entrenar la red fácilmente. `LossMonitor` puede monitorear los cambios en el valor `loss` durante el proceso de entrenamiento.

```python
# Importa las librerias necesarias para el entrenamiento del modelo.
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model
```

```python
def train_net(model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """Define a training method."""
    # Load the training dataset.
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)
```

`dataset_sink_mode` se usa para controlar si la información esta descargada. Descargar la información significa que dicha información se transmite directamente al dispositivo a traves de un canal para acelerar la velocidad de entrenamiento. Si `dataset_sink_mode` es True, la informacion se descarga. De lo contrario, la información no es descargada.

Valida la capacidad de generalización del modelo basandose en el resultado obtenido al ejecutar el dataset de testeo.

1. Lee el dataset de testeo usando la API `model.eval`.
2. Usa los parámetros del modelo guardado para la inferencia.

```python
def test_net(model, data_path):
    """Define a validation method."""
    ds_eval = create_dataset(os.path.join(data_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))
```

Cambia `train_epoch` a 1 para entrenar el data set en un epoch. En los métodos `train_net` y `test_net` , los datasets de entrenamiento previamente descargados son cargados. `mnist_path` es el directorio en el que se encuentra el dataset MNIST.

```python
train_epoch = 1
mnist_path = "./datasets/MNIST_Data"
dataset_size = 1
model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
train_net(model, train_epoch, mnist_path, dataset_size, ckpoint, False)
test_net(model, mnist_path)
```

Ejecuta el siguiente comando en la terminal para ejecutar el script:

```bash
python lenet.py --device_target=CPU
```

Donde,

`lenet.py`: Puedes copiar todo el código anterior a lenet.py (excluyendo el código para descargar el dataset). Generalmente, puedes poner la parte de los imports al inicio del código, colocar las definiciones de las clases, funciones y métodos despues del código y conectar las operaciones anteriores en el método main.

`--device_target=CPU`: especifica la plataforma de hardware. El valor del parámetro puede ser `CPU`, `GPU`, o `Ascend`, dependiendo de la plataforma de hardware en la que se quiera ejecutar el script.

Los valores de pérdida son mostrados durante el entrenamiento, tal como se muestra aqui. Aunque los valores de pérdida fluctuen, gradualmente disminuyen y la precisión gradualmente aumenta en general.Los valores de pérdida mostrados cada vez pueden ser diferentes debido a su aletoriedad.
El siguiente es un ejemplo de los valores de pérdida durante el entrenamiento:

```text
epoch: 1 step: 125, loss is 2.3083377
epoch: 1 step: 250, loss is 2.3019726
...
epoch: 1 step: 1500, loss is 0.028385757
epoch: 1 step: 1625, loss is 0.0857362
epoch: 1 step: 1750, loss is 0.05639569
epoch: 1 step: 1875, loss is 0.12366105
{'Accuracy': 0.9663477564102564}
```

La precisión del modelo se muestra en la terminal. En el ejemplo, la precisión llega al 96.6%, indicando un modelo de buena calidad. Según aumenta el número de epochs de la red (`train_epoch`), la precisión del modelo mejorara también.

## Cargando el modelo

```python
from mindspore import load_checkpoint, load_param_into_net
# Cargar el modelo guardado para el testeo.
param_dict = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
# Cargar los parámetros para la red.
load_param_into_net(net, param_dict)
```

> Para más información sobre cargar un modelo en mindspore, ver [Cargar el modelo](https://www.mindspore.cn/tutorials/en/r1.6/save_load_model.html#loading-the-model).

## Validando el modelo

Usa el modelo generado para predecir la clasificación de una imagen. El procedimiento es el siguiente:

> Las imagenes predecidas seran generadas de forma aleatoria, y los resultados pueden ser diferentes cada vez.

```python
import numpy as np
from mindspore import Tensor

# Define un dataset de test. Si batch_size se pone a 1, se obtiene una sola imagen.
ds_test = create_dataset(os.path.join(mnist_path, "test"), batch_size=1).create_dict_iterator()
data = next(ds_test)

# `images` indica la imagen del test, y `labels` indica la clasificación correcta de la imagen del test.
images = data["image"].asnumpy()
labels = data["label"].asnumpy()

# Usa la función model.predict para predecir la clasificación de la imagen.
output = model.predict(Tensor(data['image']))
predicted = np.argmax(output.asnumpy(), axis=1)

# Devuelve la clasificación predecida y la clasificación correcta.
print(f'Predicted: "{predicted[0]}", Actual: "{labels[0]}"')
```

```text
    Predicted: "6", Actual: "6"
```

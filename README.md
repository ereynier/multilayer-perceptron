# Multilayer Perceptron - 42

## Dépendance:
- sklearn
- pyyaml
- numpy
- matplotlib
- seaborn
- pandas
- (progressBar)

## Usage

    $ python main.py <data_path>

## Use object

### train:
    layers = [int list from all layers shapes]
    #eg: [input_shape, 32, 32, output_shape] = 2 hidden layers with 32 neurons for each 

    network = NeuralNetwork(layers)
    
    network.fit_(X_train, y_train, epoch=800, batch_size=500, plot=True, early_stopping=20, alpha=0.8)

    network.save_('weights')

    y_pred = network.predict_(X_test)

 ### load:

    with open(filename + ".layers", "r") as f:
        layers = str(f.read())
    layers = list(map(int, layers.split(', ')))

    network = NeuralNetwork(layers)
    
    network.load_(filename + ".npy")

    y_pred = network.predict_(X)
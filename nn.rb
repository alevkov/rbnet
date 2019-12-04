# Activation function Strategy.
module ActivationFunctions
  class Tanh
    def self.calc(val)
      Math.tanh(val)
    end
    def self.calc_derivative(val)
      1 - (Math.tanh(val) * Math.tanh(val))
    end
  end

  class Sigmoid
    def self.calc(val)
      1 / (1 + Math.exp(-val))
    end
    def self.calc_derivative(val)
      self.calc(val) * (1 - self.calc(val))
    end
  end
end

# Layer of Neurons within a Network (1 x N).
class Layer
  attr_accessor :neurons

  def initialize
    @neurons = []
  end

  def add(neuron)
    @neurons.push(neuron)
  end

  def get(idx)
    @neurons[idx]
  end

  def size
    @neurons.length
  end
end

class Neuron
  attr_accessor :output
  attr_accessor :gradient
  attr_accessor :output_weights
  attr_accessor :idx

  def initialize(outputs, idx, activation, eta)
    @output_weights = []
    @idx = idx
    @output = 0.0 # output of the Neuron
    @gradient = 0.0 # for Stochastic Gradient Descent
    @activation = activation # Strategy
    @eta = eta # learning rate

    outputs.times {
      # initialize with random weight.
      @output_weights.push(rand)
    }
  end

  def sum_derivatives_of_weights(next_layer)
    sum = 0.0

    (0...(next_layer.size - 1)).each { |i|
      sum += @output_weights[i] * next_layer.get(i).gradient
    }

    sum
  end

  def grad_output_layer(val)
    delta = val - @output
      @gradient = delta * @activation.calc_derivative(@output)
  end

  def grad_hidden_layer(next_layer)
    dow = sum_derivatives_of_weights(next_layer)
      @gradient = dow * @activation.calc_derivative(@output)
  end

  def update_input_weights(prev_layer)
    (0...prev_layer.size).each { |i|
      weight_delta = @eta * prev_layer.get(i).output * @gradient
      prev_layer.get(i).output_weights[@idx] += weight_delta
    }
  end

  def forward(prev_layer)
    sum = 0.0

    # sum the previous output values with weight multiplied
    prev_layer.neurons.each do |neuron|
      sum += (neuron.output * neuron.output_weights[@idx])
    end

    @output = @activation.calc(sum)
  end

end

class Network
  attr_accessor :epochs

  def initialize(shape, activation, eta)
    @layers = []
    @epochs = 0

    shape.each_with_index do |neurons_in_layer, layer_idx|
      layer = Layer.new
      num_outputs = neurons_in_layer == shape.last ? 0 : shape[layer_idx + 1]

      # Populate Layer with Neuron(s).
      # Add extra bias Layer (+1).
      (neurons_in_layer + 1).times do |neuron_idx|
        layer.add(Neuron.new(num_outputs, neuron_idx, activation, eta))
      end

      @layers.push(layer)
    end
  end

  def forward(_X)
    # Input shape must match shape of input Layer.
    if _X.length != @layers[0].size - 1
      puts 'Error! Input shape mismatch.'
      return
    end

    # Set output of input Neuron to input value provided.
    (0..._X.length).each { |i|
      @layers[0].neurons[i].output = _X[i]
    }

    # Feed forward starting with first hidden layer.
    (1...@layers.length).each { |i|
      (0...(@layers[i].size - 1)).each { |j|
        @layers[i].get(j).forward(@layers[i - 1])
      }
    }
  end

  def backprop(_y)
    (0...(@layers.last.size - 1)).each { |i|
      @layers.last.get(i).grad_output_layer(_y[i])
    }

    i = @layers.length - 2
    while i > 0

      (0...@layers[i].size).each { |j| # hidden
        @layers[i].get(j).grad_hidden_layer(@layers[i + 1])
      }

      i = i.pred
    end

    i = @layers.length - 1
    while i > 0

      (0...(@layers[i].size - 1)).each { |j|
        @layers[i].get(j).update_input_weights(@layers[i - 1])
      }

      i = i.pred
    end
  end

  def fit(_X, _y)
    (0..@epochs).each {
      (0..._X.length).each { |j|
        forward(_X[j])
        backprop(_y[j])
      }
    }
  end

  def predict(_X)
    forward(_X)
    results
  end

  def results
    res = []

    (0...(@layers.last.size - 1)).each { |i|
      res.push(@layers.last.get(i).output)
    }

    res
  end
end

class NetworkTester

  def main
    # Goal 1: train Network to function as logical gates and observe performance.
    # Goal 2: compare performance based on Strategy (ActivationFunctions)

    # Network setup
    shape1 = [2, 4, 4, 1] # 2 input, 2 hidden, 1 output
    shape2 = [2, 10, 1] # 2 input, 1 hidden, 1 output
    net1 = Network.new(shape1, ActivationFunctions::Tanh, 0.15) # Tanh ActivationFunctions Strategy
    net2 = Network.new(shape2, ActivationFunctions::Sigmoid, 0.15) # Sigmoid ActivationFunctions Strategy
    net1.epochs = 5000
    net2.epochs = 10000

    # Inputs
    _X = [[0, 1],
          [0, 0],
          [1, 0],
          [1, 1]]

    # **** Test 1: OR Gate ****

    _y = [1, 0, 1, 1] # training data
    puts "Training Net 1..."
    net1.fit(_X, _y)
    puts "Training Net 2..."
    net2.fit(_X, _y)

    puts "OR Gate Results:"
    puts "Net 1 RMS Error: #{calc_rms_error(net1, _X, _y)}"
    puts "Net 2 RMS Error: #{calc_rms_error(net2, _X, _y)}"

    puts "-----------"

    # **** Test 2: XOR Gate ****

    _y = [1, 0, 1, 0] # training data
    puts "Training Net 1..."
    net1.fit(_X, _y)
    puts "Training Net 2..."
    net2.fit(_X, _y)

    puts "XOR Gate Results:"
    puts "Net 1 RMS Error: #{calc_rms_error(net1, _X, _y)}"
    puts "Net 2 RMS Error: #{calc_rms_error(net2, _X, _y)}"

    puts "-----------"

    # **** Test 3: AND Gate ****

    _y = [0, 0, 0, 1] # training data
    puts "Training Net 1..."
    net1.fit(_X, _y)
    puts "Training Net 2..."
    net2.fit(_X, _y)

    puts "AND Gate Results:"
    puts "Net 1 RMS Error: #{calc_rms_error(net1, _X, _y)}"
    puts "Net 2 RMS Error: #{calc_rms_error(net2, _X, _y)}"
  end

  def calc_rms_error(net, input, target)
    error1 = 0.0

    (0...(target.length)).each { |i|
      delta1 = target[i] - net.predict(input[i])[0]
      error1 += delta1 * delta1
    }

    error1 /= target.length
    Math.sqrt(error1)
  end
end

NetworkTester.new.main

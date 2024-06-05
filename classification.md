---
layout: page
title: What is Image Classification?
permalink: classification
---

> Satoshi Nakamoto is the name used by the presumed pseudonymous person or persons who developed bitcoin, authored the bitcoin...

## Carousel

<div id="carouselExample" class="carousel slide relative" data-bs-ride="carousel">
  <div class="carousel-inner relative w-full overflow-hidden">
    <div class="carousel-item active float-left w-full">
      <img src="{{site.baseurl}}/assets/img/loss.jpeg" class="block w-full" alt="First Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/slow1.jpeg" class="block w-full" alt="Second Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/slow2.jpeg" class="block w-full" alt="Third Slide">
    </div>
  </div>
  <button class="carousel-control-prev absolute top-0 bottom-0 flex items-center justify-center p-0 text-center border-0 hover:outline-none hover:no-underline focus:outline-none focus:no-underline left-0" type="button" data-bs-target="#carouselExample" data-bs-slide="prev">
    <span class="carousel-control-prev-icon inline-block bg-no-repeat" aria-hidden="true"></span>
    <span class="visually-hidden">Previous</span>
  </button>
  <button class="carousel-control-next absolute top-0 bottom-0 flex items-center justify-center p-0 text-center border-0 hover:outline-none hover:no-underline focus:outline-none focus:no-underline right-0" type="button" data-bs-target="#carouselExample" data-bs-slide="next">
    <span class="carousel-control-next-icon inline-block bg-no-repeat" aria-hidden="true"></span>
    <span class="visually-hidden">Next</span>
  </button>
</div>


## Profile Image

<img class="mx-auto w-1/2" src="{{site.baseurl}}/assets/img/bean.jpg">

## Intro

Have you ever wondered how Apple FaceID distinguishes between your face and someone else’s? Or how Tesla Autopilot cars see other cars on the road? Or possibly you’re wondering how your favorite social media app can identify and apply filters to your face to give you and your friends dog ears or a new hairstyle. The answer involves a field of computer science called Computer Vision. Computer Vision is a subfield of Artificial Intelligence that involves training computers to process and extract details from digital images in order to perform different tasks such as recognizing objects and faces, tracking movement, and more.

One of the key tasks that computer vision can perform is classification. Image classification is the task of being able to accurately classify an object in an image into a certain category or “class.” This can look like trying to determine if an image of a furry household pet is a cat or a dog or a captcha identifying which boxes include bicycles. For computer vision’s application in a self-driving car, it must be able to identify objects and classify them in order to make further decisions correctly and safely. For example, as a car approaches an intersection, it needs to be able to differentiate between a stop sign and a speed limit sign. It also needs to be able to know the difference between a clear road and one where there are people illegally jaywalking.

## Neural Networks

So how are computers able to make these classifications? While these tasks are usually pretty simple for humans to do, it is much more complex for computers to perform. Computers use a type of technology, also known as a model, called neural networks. These neural networks were inspired by the way the human brain works as a giant network of connected neurons. By training these neural networks with hundreds to thousands or even millions of examples, they begin to learn patterns from the data and make accurate predictions.

There are many parts of a neural network that each play a role in achieving our goal. Let’s take a look at the main parts of the architecture.

### Input Layer

The input layer is the neural network's first interaction with the image it is trying to classify. The input layer does not do much more than verifying that the data is in the correct format and structure and passing it to the next layer.

### Output Layer

The output layer is the final output of the neural network. It returns a value that can be interpreted as a prediction that can take many different forms. In the context of image classification, the output often represents a probability that the image fits the classification. For a binary classification such as: “is this image a cat?”, we would expect the output layer to just be one value: the probability of the image being a cat. For a multiclass classification, the output layer can have multiple values representing the probability of each class.

### Hidden Layers

The hidden layers lie in between the input and output layers. There can be as few as one hidden layer but there are often multiple hidden layers in a neural network. Each hidden layer receives its inputs from the directly previous layer and sends its output to the next layer all the way until the output layer. We will discuss what happens here more in the next section.

### Weights and Biases

Each layer consists of several “nodes” that contain a “weight” and a “bias.” The weight is multiplied by the node’s input and the bias is added to produce the output at each layer.

## Math

Now that we have a basic understanding of the neural network structure, how do these things actually work? In this section, we will introduce a few of the concepts that are key to training neural networks. We will explore how loss functions are used to measure the accuracy of neural networks, how optimizations like gradient descent are used to minimize the loss, and how activation functions are used to model complex relationships in the data.

### Loss Functions

In order to determine whether or not a neural network is making accurate predictions, we need some way to measure its accuracy. We do this through what is called a loss function. One common loss function that is often used in statistics is the mean squared error. When modeling a linear regression, also known as a line of best fit, mean squared error is often used to minimize the distance between every data point and the nearest point on the line. In a linear regression, the slope and y-intercept are continuously updated until the loss function reaches its minimum value.

#### Common Loss Functions

**Mean Squared Error**: As mentioned above, Mean Squared Error (MSE) is a very important loss function. It is often used when trying to measure the accuracy of predictions for continuous variables. As the name suggests, it measures the average squared difference between actual and expected values.

The equation can be represented as:

\[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]

where \( n \) represents the total number of data points, \( y \) represents the actual observed value, and \( \hat{y} \) is our prediction from the model.

**Binary Cross Entropy**: Cross Entropy loss is a common loss function used specifically for classification problems. Binary Cross Entropy (BCE) is a simplified version when dealing with binary classification problems. This binary represents that there are only two options: yes/no, on/off, positive/negative test result, dog/cat, etc. When dealing with binary classification problems, we assign one of the labels with 1 (usually the ‘positive’ value) and the other with a 0.

The equation for BCE is:

\[ \text{BCE} = - \frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right) \]

where \( N \) is the number of observations, \( i \) is the \( i \)-th observation, \( \hat{y}_i \) is the predicted probability of the observation being positive (on a scale of 0-1), and \( y_i \) is the actual label.

When we break down this equation, the left side \( y_i \log(\hat{y}) \) measures how well we predicted positive values and the right side \( (1-y_i) \log(1-\hat{y}) \) measures how well we predicted negative values.

For example, if we correctly predicted an observation to be positive with a high probability, say 0.9, then we add \( -1 \cdot \log(0.9) + (0) \cdot \log(0.1) \approx 0.045 \) to our loss value. However, if we incorrectly predict a positive value as negative with a probability of 0.8 (this is a 0.2 probability of positive), then we add \( -1 \cdot \log(0.2) + (0) \cdot \log(0.8) \approx 0.698 \) to our loss value.

This loss function rewards correct predictions that are confident (high probabilities) with low loss values, penalizes incorrect predictions with high loss values, and has a medium loss value for uncertain probabilities.

### Activation Functions

Between each hidden layer, there is usually something called an “activation function” applied. This activation function serves two main purposes. First, it introduces nonlinearities. Without activation functions, neural networks consisting of chains of fully connected hidden layers could be simplified to a single hidden layer since each hidden layer is simply a linear transformation. These nonlinearities enable the neural network to learn more complex patterns. Second, activation functions are often differentiable, allowing us to effectively compute gradients to perform gradient descent.

#### Common Activation Functions

**Sigmoid**: The sigmoid function maps inputs to values between 0 and 1 using the following equation:

\[ \sigma(x) = \frac{1}{1 + e^{-x}} \]

This is very beneficial when we want to compute probabilities as it maps values that may not be interpretable to something that lies with a valid probability.

**Rectified Linear Unit (ReLU)**: ReLU is another very common activation function used in neural networks. ReLU is applied with the equation \( f(x) = \max(0, x) \). The simplicity of the function comes with both benefits and drawbacks. The main con is that it may lose information due to clipping off negative values. However, it introduces nonlinearities, is computationally effective, running much faster than most alternatives, and does not suffer from the ‘vanishing gradient’ problem.

### Gradient Descent and Learning Rate

Gradient descent is an optimization algorithm used in training neural networks. The algorithm works by finding the gradient of the loss function at the current position. The gradient tells us the direction and rate of change with respect to each model variable. By following the negative direction we can move further down the loss function to minimize the difference between our predicted and actual values. We can continue doing this until our loss value is low enough that we are making accurate predictions. One important factor to consider is our learning rate. The learning rate acts as a coefficient to determine how much we want to update our values by, how large of a ‘step’ to take. A learning rate that is too small may cause our algorithm to take too long to train our data. A learning rate too large may cause us to overshoot our goal.

## Small Learning Rate

<div id="carouselExample2" class="carousel slide relative" data-bs-ride="carousel">
  <div class="carousel-inner relative w-full overflow-hidden">
    <div class="carousel-item active float-left w-full">
      <img src="{{site.baseurl}}/assets/img/loss.jpeg" class="block w-full" alt="First Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/slow1.jpeg" class="block w-full" alt="Second Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/slow2.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/slow3.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/slow4.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/slow5.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/slow6.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/slow7.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/slow8.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/slow9.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/slow10.jpeg" class="block w-full" alt="Third Slide">
    </div>
  </div>
  <button class="carousel-control-prev absolute top-0 bottom-0 flex items-center justify-center p-0 text-center border-0 hover:outline-none hover:no-underline focus:outline-none focus:no-underline left-0" type="button" data-bs-target="#carouselExample2" data-bs-slide="prev">
    <span class="carousel-control-prev-icon inline-block bg-no-repeat" aria-hidden="true"></span>
    <span class="visually-hidden">Previous</span>
  </button>
  <button class="carousel-control-next absolute top-0 bottom-0 flex items-center justify-center p-0 text-center border-0 hover:outline-none hover:no-underline focus:outline-none focus:no-underline right-0" type="button" data-bs-target="#carouselExample2" data-bs-slide="next">
    <span class="carousel-control-next-icon inline-block bg-no-repeat" aria-hidden="true"></span>
    <span class="visually-hidden">Next</span>
  </button>
</div>

## Large Learning Rate

<div id="carouselExample3" class="carousel slide relative" data-bs-ride="carousel">
  <div class="carousel-inner relative w-full overflow-hidden">
    <div class="carousel-item active float-left w-full">
      <img src="{{site.baseurl}}/assets/img/loss.jpeg" class="block w-full" alt="First Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/fast.jpeg" class="block w-full" alt="Second Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/fast-2.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/fast-3.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/fast-4.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/fast-5.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/fast-6.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/fast-7.jpeg" class="block w-full" alt="Third Slide">
    </div>
    <div class="carousel-item float-left w-full">
      <img src="{{site.baseurl}}/assets/img/fast-8.jpeg" class="block w-full" alt="Third Slide">
    </div>
  </div>
  <button class="carousel-control-prev absolute top-0 bottom-0 flex items-center justify-center p-0 text-center border-0 hover:outline-none hover:no-underline focus:outline-none focus:no-underline left-0" type="button" data-bs-target="#carouselExample3" data-bs-slide="prev">
    <span class="carousel-control-prev-icon inline-block bg-no-repeat" aria-hidden="true"></span>
    <span class="visually-hidden">Previous</span>
  </button>
  <button class="carousel-control-next absolute top-0 bottom-0 flex items-center justify-center p-0 text-center border-0 hover:outline-none hover:no-underline focus:outline-none focus:no-underline right-0" type="button" data-bs-target="#carouselExample3" data-bs-slide="next">
    <span class="carousel-control-next-icon inline-block bg-no-repeat" aria-hidden="true"></span>
    <span class="visually-hidden">Next</span>
  </button>
</div>

## Convolutional Filters

So now that you know about neural networks you may still be wondering how these things allow computers to process image data. They do it with something called convolutional filters. A convolutional filter is a small matrix of values that is used to slide over images to identify specific patterns. This is made possible by representing an image as a matrix of values that represent each pixel as RGB or grayscale values. Convolutional filters serve two main roles. Different filters are able to detect different patterns which represent certain features such as edges, textures, or certain shapes in the image.

When the filter “slides” across the image, it performs a function called a convolution, which in essence is a matrix multiplication and then taking the sum of the values similar to a dot product. As the filter makes its way through the entire image matrix, it creates a new matrix known as a feature map. This feature map represents the locations in the image where the patterns or features are detected.

### Convolutional Filter Demo

## CNNs

We can repeat this pattern many times trying out different filters. By using many convolutional filters together, neural networks can learn to recognize more complex patterns and higher level features. While the initial convolutional filters may have learned to recognize edges and curves, after passing through multiple layers, the network begins to learn more complex patterns like rectangles or circles which could represent wheels on a car or a bicycle. Eventually, it recognizes these complex patterns together with other detected features to identify actual objects within the image. The combination of these filters building on one another over many iterations gives the Convolutional Neural Network the ability to understand images and classify objects within them.

This pattern recognition capability of Convolutional Neural Networks is what makes them so effective for various applications in computer vision. For example, in facial recognition applications like Apple FaceID, CNNs can detect facial features like our eyes or nose and distinguish between different individuals. In Tesla's Autopilot, CNNs can recognize other cars, pedestrians, and road signs, and respond in a safe way to enable autonomous driving. Additionally, in social media apps, CNNs power the filters that can detect and modify facial features which lets us have dog ears or new hairstyles in real-time. The ability and accuracy of CNNs to classify images have made them critical tools, driving advancements and innovations across these popular and impactful technologies.

## Adversarial Examples

While CNNs are very powerful tools, they also exhibit some vulnerabilities that we should be cautious of. One of these vulnerabilities is something called an adversarial example. Adversarial examples are digital images that are intentionally designed to trick neural networks and humans. Often by just changing a few pixels of an image, a CNN may make a completely different and incorrect classification prediction despite humans being unable to tell the difference between the original image and the new one.

One notable and well known example of this includes an image of a panda. The CNN predicts that it is a panda with over 50% confidence. However, another image that looks the exact same is misclassified by the CNN to be a gibbon with over 99% confidence. This example highlights the vulnerability that minute changes to an image can have major effects on the predictions of a neural network.

While misclassifying an image of a panda may not be that big of a problem, similar examples have more significant impacts. One such example includes a stop sign that had a small sticker on it that was misclassified to be a speed limit sign. Misclassification and other vulnerabilities in these Convolutional Neural Network systems have the potential to be disastrous if not resolved. With image classification being used in autonomous vehicles, medical imaging, security, and many more fields, it is vital that computer scientists find a way to mitigate these problems.

## Ethical Concerns

Beyond misclassification, other problems persist. One problem includes bias in training data. For example, many early facial recognition models tended to perform much higher on caucasian and lighter skinned people and had high error rates on people of color. While this wasn’t an intentional feature, unrepresentative data has the potential to harm marginalized communities. This continues to be an issue with future developments of artificial intelligence and machine learning that must be addressed.

Additionally, privacy invasion is another possible ethical issue. When used in surveillance and security systems, CNNs may track and analyze people’s movement and behavior without their knowledge or consent. Widespread access to this technology raises concerns about rights to privacy and ethical usage. As CNNs become more prevalent, it is important to ensure that these technologies are used responsibly and respect individual privacy rights. While CNNs have the power to lead the way in technology and innovation with image classification, it is essential to address the ethical concerns related to bias and privacy. Ensuring fairness and robustness in model training accuracy and implementing safeguards to protect privacy are critical steps toward a responsible and ethical use of neural networks in society.

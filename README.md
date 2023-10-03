# Neural Fader-Hardie Model

In this repository, we present a neural network-based variant of the [Fader-Hardie model](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf) for contractual customer churn.

## Architecture

### Modeling

We recall the Fader-Hardie model here. As noted above, we only consider the contractual churn case.

At the end of each period, a customer is subjected to a coin toss: "heads" the contract is canceled, "tails" the contract is renewed. The probability of "heads" remains constant over time for any single customer, but the probability of "heads" differs between customers. The latter assumption allows us to account for *heterogeneity* in churn rates. This also means that a customer does not have greater affinity to the product over time.

Our model here allows more features to be accounted in computing the probability of "heads". The flexibility of neural networks allow us to learn over various demographic, firmographic, and behavioral features, for example, countries, age, number of purchases etc.

### Training

The figure below shows how the model is trained. The main component of the setup is the *encoder*: this takes features of the customer and outputs a churn probability *p*. Thus, *p* here is a function of the input features, which allows us to model the heterogeneity of churn amongst customers. Through the magic of [backpropagation](https://en.wikipedia.org/wiki/Backpropagation), the model learns the churn probability of customers segmented by their demographic and behavioral features.

The hazard loss in this case (since we are operating in the world of contractual churn) is the log-likelihood function of a [geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution). This is the consequence of the coin toss analogy of contractual churn described above.

In order to train the model, we provide two labels:

* *periods*: how many periods has the subscription lasted (integer value at least 1), and
* *has_churned*: whether the customer has churned, or not (binary value, 1 for churned, 0 otherwise).

The second label lets us know which customer has an ongoing subscription (that is, not censored).

![Training of the model](/assets/images/training.png)

### Inference

Once we have trained the model, we can predict the probability of churn given the period *t*. The probability follows the geometric distribution.

![Forecasting churn using the model](/assets/images/inference.png)


## Why neural?

It is often a running joke that the easiest way to write a new machine learning paper is to take an existing model, and put a neural spin on it. **This piece of work is no different**.

Beyond the aforementioned facetious statement, there are reasons why you might want to use a neural model:

* **flexibility in introducing more features over time**: unlike the Fader-Hardie model, we don't have to slice our cohorts and fit a model per cohort. For instance, businesses typically treat churn differently across different regions around the world. In this case, countries can be included into the model via an embedding layer, which in turn allows the model to learn representations of country embeddings. The ability to leverage deep learning in this context is very powerful.

* **ability to try various feature extractors**: we can try various feature extraction modules depending on the dataset, given the
  flexibility of deep learning.

* **easy extension to sequences of cohorts by using a sequence based model**: if the model incorporates a recurrent neural network (RNN), then we can account for previous cohort probabilities of churn by incorporating them into the model.

* **leverage the goodness of composabilty via deep neural networks**: you can use various encoders to encode your features in your model depending on your dataset. As long as the hazard loss is used, the model should be able to create churn forecasts.

## Why not neural?

Of course, there are downsides:

* **data requirements**: like other deep learning algorithms, it is required that you have a large dataset.

* **can't fit into a spreadsheet**: the Fader-Hardie model has closed form solutions for the survival function. We do not, so our model cannot be used in a spreadsheet unlike the original, nor implemented in a SQL script.

* **limited interpretability**: like other black box models, our model cannot be interpreted easily.

## Improvements

The model used here is one example that relies on convolutional layers with a sparsity for feature selection. Other more exotic models such as [TabNet](https://arxiv.org/abs/1908.07442) can be used here too.

## Running

A shell script is provided to run the example
```shell
./run.sh all 
```
runs the data generation and training.

```shell
./run.sh generate
```
generates the synthetic churn data of a fictional SaaS product.

```shell
./run.sh train
```
runs the model training only.
